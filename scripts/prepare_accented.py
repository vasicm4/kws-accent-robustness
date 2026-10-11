"""Cut raw accented recordings into labelled 1 s clips.

Every recording follows the collection protocol: the 10 target words in
order, each said 1 or 3 times in a row (yes yes yes no no no ...). Some
speakers sent one file per word instead. Labels therefore come from the
position of each detected word, not from listening to it.

    uv run --extra accented python scripts/prepare_accented.py --src ~/Downloads/kws

On first run it writes data_accented/private/file_map.csv (source file ->
anonymous speaker id, optional word, optional crop). Edit it and re-run if
several files belong to the same speaker or a file needs cropping. Leave
`speaker` empty to skip a file, e.g. one replaced by a cleaned-up export.

A recording whose word count can't be matched to 1 or 3 reps is not
labelled; it is marked `review` in data_accented/sources.csv and its
detected segments are written to data_accented/private/labels/<tag>.auto.txt.
To fix it: open the recording in Audacity, File > Import > Labels that file,
delete the labels that aren't target words (merge any word split in two),
then File > Export > Labels as <tag>.txt in the same folder and re-run.
Labels that aren't a target word are filled in by order.

--verify transcribes every clip with Whisper and records disagreements in
clips.csv. Use it to choose what to listen to, never to drop clips: Whisper
is itself weaker on accented speech.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import TARGET_WORDS  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data_accented"
AUDIO_SUFFIXES = {".mp4", ".m4a", ".ogg", ".opus", ".wav", ".mp3", ".aac", ".flac"}
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
FRAME = 400  # 25 ms
HOP = 160  # 10 ms
MIN_GAP_S = 0.15  # closer segments are one word (e.g. the closure in "stop")
MIN_WORD_S = 0.08
BLIP_DB = 12.0  # segments this far below the median peak are breaths
CLICK_S = 0.13  # shorter than any target word
EDGE_S = 0.15
EDGE_TAP_S = 0.3
MAX_MERGE_GAP_S = 0.35
MAX_WORD_S = 1.2
SUSPECT_WORD_S = 1.6  # a longer "word" is two words or noise: labels after it shift
VALID_REPS = (1, 3)
SINGLE_WORD_PATTERN = re.compile(r"^New Recording(?: (\d+))?\.", re.IGNORECASE)


@dataclass
class Segment:
    start: int  # samples
    end: int
    peak_db: float
    label: str | None = None


def decode(path: Path) -> np.ndarray:
    """Any container/codec -> float32 mono at SAMPLE_RATE (PyAV bundles ffmpeg)."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
    chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32)


def frame_db(waveform: np.ndarray) -> np.ndarray:
    n_frames = max(1, 1 + (len(waveform) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = np.pad(waveform, (0, FRAME))[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def detect_segments(waveform: np.ndarray) -> list[Segment]:
    """Energy detection with a threshold set between the recording's own
    noise floor and its speech peaks, so a constant background hum doesn't
    swallow the whole file."""
    db = frame_db(waveform)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    threshold = max(peak - 35.0, floor + 0.3 * (peak - floor))
    active = db > threshold

    runs: list[list[int]] = []
    for i, is_active in enumerate(active):
        if not is_active:
            continue
        if runs and i - runs[-1][1] <= MIN_GAP_S * SAMPLE_RATE / HOP:
            runs[-1][1] = i
        else:
            runs.append([i, i])

    segments = []
    for first, last in runs:
        start, end = first * HOP, last * HOP + FRAME
        if end - start >= MIN_WORD_S * SAMPLE_RATE:
            segments.append(Segment(start, end, float(db[first : last + 1].max())))
    return segments


def is_junk(segment: Segment, median_peak: float, n_samples: int) -> bool:
    """Breaths (quiet), clicks (too short for any target word) and the tap
    on the record/stop button (short, at the very start or end)."""
    duration = segment.end - segment.start
    at_edge = segment.start < EDGE_S * SAMPLE_RATE or segment.end > n_samples - EDGE_S * SAMPLE_RATE
    return (
        segment.peak_db < median_peak - BLIP_DB
        or duration < CLICK_S * SAMPLE_RATE
        or (at_edge and duration < EDGE_TAP_S * SAMPLE_RATE)
    )


def fit_to_reps(segments: list[Segment], n_words: int, n_samples: int) -> list[Segment] | None:
    """Drop junk and re-join split words until the count is n_words times a
    valid rep count. None if no such count is reachable."""
    segments = list(segments)
    targets = sorted(n_words * r for r in VALID_REPS)

    while segments and len(segments) not in targets:
        if len(segments) < targets[0]:
            return None
        median_peak = float(np.median([s.peak_db for s in segments]))
        junk = [i for i, s in enumerate(segments) if is_junk(s, median_peak, n_samples)]
        if junk:
            segments.pop(min(junk, key=lambda i: segments[i].peak_db))
            continue

        gaps = [
            (segments[i + 1].start - segments[i].end, i)
            for i in range(len(segments) - 1)
            if segments[i + 1].end - segments[i].start <= MAX_WORD_S * SAMPLE_RATE
        ]
        if not gaps or min(gaps)[0] > MAX_MERGE_GAP_S * SAMPLE_RATE:
            return None
        _, i = min(gaps)
        left, right = segments[i], segments.pop(i + 1)
        segments[i] = Segment(left.start, right.end, max(left.peak_db, right.peak_db))

    return segments or None


def read_audacity_labels(path: Path) -> list[Segment]:
    segments = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0].startswith("\\"):  # skip spectral rows
            continue
        text = parts[2].strip().lower() if len(parts) > 2 else ""
        label = text if text in TARGET_WORDS else None
        start, end = (int(float(p) * SAMPLE_RATE) for p in parts[:2])
        segments.append(Segment(start, end, 0.0, label))
    return segments


def assign_labels(segments: list[Segment], words: list[str]) -> list[Segment] | None:
    if all(s.label for s in segments):
        return segments
    reps, rest = divmod(len(segments), len(words))
    if rest:
        return None
    for segment, word in zip(segments, [w for w in words for _ in range(reps)]):
        segment.label = segment.label or word
    return segments


def write_review_labels(path: Path, segments: list[Segment], offset: int) -> None:
    """Every detected segment as an Audacity label track, so fixing a
    recording means deleting the few wrong labels, not marking all words."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(segments, 1):
            start, end = (s.start + offset) / SAMPLE_RATE, (s.end + offset) / SAMPLE_RATE
            f.write(f"{start:.3f}\t{end:.3f}\t#{i} {s.peak_db:.0f}dB {end - start:.2f}s\n")


def cut_clip(waveform: np.ndarray, segment: Segment) -> np.ndarray:
    """1 s window centred on the word, taken from the recording itself so
    the clip keeps the speaker's real background (like Speech Commands)."""
    length = int(CLIP_SECONDS * SAMPLE_RATE)
    centre = (segment.start + segment.end) // 2
    start = min(max(0, centre - length // 2), max(0, len(waveform) - length))
    clip = waveform[start : start + length]
    return np.pad(clip, (0, length - len(clip)))


def default_file_map(sources: list[Path]) -> list[dict[str, str]]:
    """One speaker per file, except the per-word "New Recording N" files,
    which are one speaker (unnumbered = yes, 2 = no, ..., 10 = go)."""
    rows, speaker_of = [], {}
    for path in sources:
        match = SINGLE_WORD_PATTERN.match(path.name)
        group = "single-word" if match else path.name
        if group not in speaker_of:
            speaker_of[group] = f"spk{len(speaker_of) + 1:02d}"
        word = TARGET_WORDS[int(match.group(1) or 1) - 1] if match else ""
        rows.append(
            {"source": path.name, "speaker": speaker_of[group], "word": word,
             "start_s": "", "end_s": "", "device": ""}
        )
    return rows


def load_or_create_file_map(path: Path, sources: list[Path]) -> list[dict[str, str]]:
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        known = {r["source"] for r in rows}
        missing = [s.name for s in sources if s.name not in known]
        if missing:
            sys.exit(f"{path} has no row for: {', '.join(missing)}. Add them and re-run.")
        return rows
    rows = default_file_map(sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} (one speaker per file; edit it if that's wrong)")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_with_whisper(clip_rows: list[dict], out: Path, model_size: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    prompt = ", ".join(TARGET_WORDS) + "."
    for row in clip_rows:
        audio, _ = sf.read(out / row["path"], dtype="float32")
        segments, _ = model.transcribe(
            audio, language="en", initial_prompt=prompt, beam_size=5,
            condition_on_previous_text=False,
        )
        heard = re.sub(r"[^a-z ]", "", " ".join(s.text for s in segments).lower()).split()
        row["asr"] = " ".join(heard)
        row["asr_match"] = int(row["label"] in heard)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="folder of raw recordings")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true", help="cross-check clips with Whisper")
    parser.add_argument("--whisper-model", default="small")
    args = parser.parse_args()

    sources = sorted(
        p for p in args.src.expanduser().iterdir() if p.suffix.lower() in AUDIO_SUFFIXES
    )
    private = args.out / "private"
    file_map = load_or_create_file_map(private / "file_map.csv", sources)
    clips_dir = args.out / "clips"
    shutil.rmtree(clips_dir, ignore_errors=True)  # no stale clips from earlier runs

    clip_rows, source_rows = [], []
    file_map = [r for r in file_map if r["speaker"].strip()]  # empty = superseded, skip
    for row in file_map:
        source = args.src.expanduser() / row["source"]
        speaker, word = row["speaker"], row["word"].strip().lower()
        words = [word] if word else TARGET_WORDS
        tag = f"{speaker}_{word}" if word else speaker

        waveform = decode(source)
        offset = int(float(row["start_s"] or 0) * SAMPLE_RATE)
        end = int(float(row["end_s"]) * SAMPLE_RATE) if row["end_s"] else len(waveform)
        waveform = waveform[offset:end]

        labels_file = private / "labels" / f"{tag}.txt"
        if labels_file.exists():
            method = "audacity"
            segments = read_audacity_labels(labels_file)
            for s in segments:  # label times are relative to the uncropped file
                s.start, s.end = s.start - offset, s.end - offset
            detected = len(segments)
        else:
            method = "auto"
            raw = detect_segments(waveform)
            detected = len(raw)
            segments = fit_to_reps(raw, len(words), len(waveform))
            if segments and any(
                s.end - s.start > SUSPECT_WORD_S * SAMPLE_RATE for s in segments
            ):
                segments = None
        labelled = assign_labels(segments, words) if segments else None
        if not labelled and method == "auto":
            write_review_labels(private / "labels" / f"{tag}.auto.txt", raw, offset)

        status = "ok" if labelled else "review"
        source_rows.append(
            {"speaker": speaker, "word": word, "method": method, "detected": detected,
             "kept": len(labelled) if labelled else 0, "status": status}
        )
        print(f"{tag:12s} {method:8s} detected={detected:3d} -> "
              f"{len(labelled) if labelled else '-':>3}  {status}")
        if not labelled:
            continue

        rep_of: dict[str, int] = {}
        for segment in labelled:
            rep = rep_of[segment.label] = rep_of.get(segment.label, 0) + 1
            rel = Path("clips") / segment.label / f"{speaker}_{segment.label}_{rep}.wav"
            (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.out / rel, cut_clip(waveform, segment), SAMPLE_RATE, subtype="PCM_16")
            word_s = (segment.end - segment.start) / SAMPLE_RATE
            clip_rows.append(
                {"path": rel.as_posix(), "speaker": speaker, "label": segment.label,
                 "rep": rep, "word_s": round(word_s, 3),
                 "flag": "longer_than_clip" if word_s > CLIP_SECONDS else "",
                 "asr": "", "asr_match": ""}
            )

    if args.verify and clip_rows:
        verify_with_whisper(clip_rows, args.out, args.whisper_model)

    write_csv(args.out / "clips.csv", clip_rows, list(clip_rows[0]) if clip_rows else ["path"])
    write_csv(args.out / "sources.csv", source_rows, list(source_rows[0]))

    by_speaker: dict[str, dict[str, str]] = {}
    for row in file_map:
        by_speaker.setdefault(row["speaker"], {"speaker": row["speaker"], "device": row["device"]})
    for speaker, meta in by_speaker.items():
        meta["n_clips"] = sum(r["speaker"] == speaker for r in clip_rows)
    write_csv(args.out / "metadata.csv", list(by_speaker.values()), ["speaker", "device", "n_clips"])

    n_review = sum(r["status"] == "review" for r in source_rows)
    print(f"\n{len(clip_rows)} clips from {len(by_speaker)} speakers in {clips_dir}; "
          f"{n_review} recording(s) need review (see sources.csv)")
    if args.verify:
        mismatches = [r for r in clip_rows if r["asr_match"] == 0]
        print(f"Whisper disagrees on {len(mismatches)} clip(s); listen to those (clips.csv, asr_match=0)")


if __name__ == "__main__":
    main()
