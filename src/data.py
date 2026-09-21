from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
BACKGROUND_NOISE_DIR = "_background_noise_"


@dataclass(frozen=True)
class IndexEntry:
    """One dataset item: either a real wav (path set, offset None) or a
    synthesized silence crop (path, offset set, points into a noise file
    tracked separately)."""

    path: Path
    offset: int | None
    label: str


def build_label_map(target_words: list[str]) -> dict[str, int]:
    """{word: idx} for target_words + SILENCE_LABEL + UNKNOWN_LABEL, fixed order."""
    ordered_labels = list(target_words) + [SILENCE_LABEL, UNKNOWN_LABEL]
    return {label: idx for idx, label in enumerate(ordered_labels)}


def read_split_lists(root: Path) -> tuple[set[str], set[str]]:
    val_lines = (root / "validation_list.txt").read_text().splitlines()
    test_lines = (root / "testing_list.txt").read_text().splitlines()
    val_set = {line.strip() for line in val_lines if line.strip()}
    test_set = {line.strip() for line in test_lines if line.strip()}
    return val_set, test_set


def scan_label_dirs(root: Path) -> dict[str, list[Path]]:
    files_by_label: dict[str, list[Path]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        files_by_label[folder.name] = sorted(folder.glob("*.wav"))
    return files_by_label


def partition_by_split(
    files_by_label: dict[str, list[Path]],
    val_set: set[str],
    test_set: set[str],
) -> dict[str, dict[str, list[Path]]]:
    partitioned: dict[str, dict[str, list[Path]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for label, paths in files_by_label.items():
        train_files, val_files, test_files = [], [], []
        for path in paths:
            rel_path = f"{label}/{path.name}"
            if rel_path in val_set:
                val_files.append(path)
            elif rel_path in test_set:
                test_files.append(path)
            else:
                train_files.append(path)
        partitioned["train"][label] = train_files
        partitioned["val"][label] = val_files
        partitioned["test"][label] = test_files
    return partitioned


def sample_unknown(non_target_paths: list[Path], n: int, seed: int) -> list[Path]:
    if n > len(non_target_paths):
        raise AssertionError
    local_random = random.Random(seed)
    return local_random.sample(non_target_paths, n)


def build_silence_index(
    noise_dir: Path, n: int, clip_length: int, seed: int
) -> list[tuple[Path, int]]:
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not noise_files:
        raise ValueError(f"no noise files found in {noise_dir}")

    lengths = {path: sf.info(path).frames for path in noise_files}
    usable_files = [path for path in noise_files if lengths[path] >= clip_length]
    if not usable_files:
        raise ValueError(
            f"no noise files in {noise_dir} have >= {clip_length} samples"
        )

    local_random = random.Random(seed)
    crops: list[tuple[Path, int]] = []
    for _ in range(n):
        noise_path = local_random.choice(usable_files)
        max_offset = lengths[noise_path] - clip_length
        offset = local_random.randint(0, max_offset)
        crops.append((noise_path, offset))
    return crops


_SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}


def build_split_index(root: Path, split: str, config: dict) -> list[IndexEntry]:
    val_set, test_set = read_split_lists(root)
    files_by_label = scan_label_dirs(root)
    partitioned = partition_by_split(files_by_label, val_set, test_set)
    split_files = partitioned[split]

    target_word_set = set(TARGET_WORDS)
    entries: list[IndexEntry] = []
    non_target_paths: list[Path] = []
    target_counts: list[int] = []
    for label, paths in split_files.items():
        if label in target_word_set:
            entries.extend(IndexEntry(path=p, offset=None, label=label) for p in paths)
            target_counts.append(len(paths))
        else:
            non_target_paths.extend(paths)

    avg_target_count = (
        round(sum(target_counts) / len(target_counts)) if target_counts else 0
    )
    seed = config["seed"] + _SPLIT_SEED_OFFSETS[split]

    unknown_ratio = config.get("unknown_ratio", 1.0)
    n_unknown = min(round(avg_target_count * unknown_ratio), len(non_target_paths))
    unknown_paths = sample_unknown(non_target_paths, n_unknown, seed=seed)
    entries.extend(
        IndexEntry(path=p, offset=None, label=UNKNOWN_LABEL) for p in unknown_paths
    )

    silence_ratio = config.get("silence_ratio", 1.0)
    n_silence = round(avg_target_count * silence_ratio)
    clip_length = int(config["sample_rate"] * config.get("clip_duration_seconds", 1.0))
    noise_dir = root / BACKGROUND_NOISE_DIR
    silence_crops = build_silence_index(noise_dir, n_silence, clip_length, seed=seed)
    entries.extend(
        IndexEntry(path=p, offset=o, label=SILENCE_LABEL) for p, o in silence_crops
    )

    return entries


def load_waveform(path: Path, sample_rate: int) -> np.ndarray:
    waveform, native_sr = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if native_sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=sample_rate)
    return waveform.astype(np.float32)


def load_silence_waveform(
    noise_path: Path, offset: int, sample_rate: int, length: int
) -> np.ndarray:

    waveform, native_sr = sf.read(
        noise_path, start=offset, frames=length, dtype="float32", always_2d=False
    )
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if native_sr != sample_rate:
        raise ValueError(
            f"{noise_path} is at {native_sr}Hz, expected {sample_rate}Hz"
        )
    return waveform.astype(np.float32)


def fix_length(waveform: np.ndarray, target_length: int) -> np.ndarray:

    current_length = len(waveform)
    if current_length == target_length:
        return waveform
    if current_length > target_length:
        start = (current_length - target_length) // 2
        return waveform[start : start + target_length]
    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right))


_mel_transform_cache: dict[tuple, torchaudio.transforms.MelSpectrogram] = {}


def _get_mel_transform(
    sample_rate: int, n_fft: int, hop_length: int, n_mels: int
) -> torchaudio.transforms.MelSpectrogram:
    key = (sample_rate, n_fft, hop_length, n_mels)
    if key not in _mel_transform_cache:
        _mel_transform_cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    return _mel_transform_cache[key]


def extract_features(waveform: np.ndarray, config: dict) -> torch.Tensor:

    mel_transform = _get_mel_transform(
        sample_rate=config["sample_rate"],
        n_fft=config.get("n_fft", 400),
        hop_length=config.get("hop_length", 160),
        n_mels=config.get("n_mels", 40),
    )
    waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
    mel = mel_transform(waveform_t)
    return torchaudio.transforms.AmplitudeToDB()(mel)


class SpeechCommandsDataset(Dataset):

    def __init__(self, root: Path, split: str, config: dict, augment: bool = False):
        self.root = root
        self.split = split
        self.config = config
        self.augment = augment
        self.label_map = build_label_map(TARGET_WORDS)
        self.index: list[IndexEntry] = build_split_index(root, split, config)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.index[idx]
        sample_rate = self.config["sample_rate"]
        clip_length = int(
            sample_rate * self.config.get("clip_duration_seconds", 1.0)
        )

        if entry.label == SILENCE_LABEL:
            waveform = load_silence_waveform(
                entry.path, entry.offset, sample_rate, clip_length
            )
        else:
            waveform = load_waveform(entry.path, sample_rate)
            waveform = fix_length(waveform, clip_length)

        if self.augment:
            # TODO: hook into augment.py (waveform-domain noise/time-shift
            # here, SpecAugment after extract_features below) once it's
            # implemented. Raising rather than silently skipping augmentation.
            raise NotImplementedError("augment.py hooks not implemented yet")

        features = extract_features(waveform, self.config)
        label_idx = self.label_map[entry.label]
        return features, label_idx


class AccentedEvalDataset(Dataset):

    def __init__(self, metadata_csv: Path, audio_root: Path, config: dict):
        self.metadata_csv = metadata_csv
        self.audio_root = audio_root
        self.config = config
        self.label_map = build_label_map(TARGET_WORDS)
        with open(metadata_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        sample_rate = self.config["sample_rate"]
        clip_length = int(
            sample_rate * self.config.get("clip_duration_seconds", 1.0)
        )

        waveform = load_waveform(self.audio_root / row["path"], sample_rate)
        waveform = fix_length(waveform, clip_length)
        features = extract_features(waveform, self.config)
        label_idx = self.label_map.get(row["label"], self.label_map[UNKNOWN_LABEL])
        return features, label_idx
