from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import data
from fake_data import SAMPLE_RATE, _write_wav


def _speaker_id(path: Path) -> str:
    return path.name.split("_nohash_")[0]


def _speaker_ids(split_dict: dict[str, list[Path]]) -> set[str]:
    return {_speaker_id(p) for paths in split_dict.values() for p in paths}


def test_read_split_lists(corpus: Path) -> None:
    val_set, test_set = data.read_split_lists(corpus)
    assert val_set == {
        "yes/spkB_nohash_0.wav",
        "yes/spkB_nohash_1.wav",
        "no/spkB_nohash_0.wav",
    }
    assert test_set == {"no/spkD_nohash_0.wav", "no/spkD_nohash_1.wav"}


def test_scan_label_dirs_skips_background_noise(corpus: Path) -> None:
    files_by_label = data.scan_label_dirs(corpus)
    assert set(files_by_label.keys()) == {"yes", "no", "bed", "cat"}
    assert len(files_by_label["yes"]) == 4  # spkA x2 + spkB x2
    assert len(files_by_label["no"]) == 5  # spkC x2 + spkB x1 + spkD x2


def test_partition_by_split_basic(corpus: Path) -> None:
    val_set, test_set = data.read_split_lists(corpus)
    files_by_label = data.scan_label_dirs(corpus)
    partitioned = data.partition_by_split(files_by_label, val_set, test_set)

    assert len(partitioned["train"]["yes"]) == 2  # spkA
    assert len(partitioned["val"]["yes"]) == 2  # spkB
    assert len(partitioned["test"]["yes"]) == 0

    assert len(partitioned["train"]["no"]) == 2  # spkC
    assert len(partitioned["val"]["no"]) == 1  # spkB
    assert len(partitioned["test"]["no"]) == 2  # spkD

    assert len(partitioned["train"]["bed"]) == 2
    assert partitioned["val"]["bed"] == []
    assert partitioned["test"]["bed"] == []


def test_partition_by_split_keeps_speaker_together(corpus: Path) -> None:
    val_set, test_set = data.read_split_lists(corpus)
    files_by_label = data.scan_label_dirs(corpus)
    partitioned = data.partition_by_split(files_by_label, val_set, test_set)

    train_ids = _speaker_ids(partitioned["train"])
    val_ids = _speaker_ids(partitioned["val"])
    test_ids = _speaker_ids(partitioned["test"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # spkB says both "yes" and "no", both listed under validation_list.txt:
    # every one of their files must land in val, none in train.
    assert "spkB" in val_ids
    assert "spkB" not in train_ids
    assert "spkB" not in test_ids


def test_build_label_map() -> None:
    label_map = data.build_label_map(["yes", "no"])
    assert label_map == {
        "yes": 0,
        "no": 1,
        data.SILENCE_LABEL: 2,
        data.UNKNOWN_LABEL: 3,
    }
    assert len(set(label_map.values())) == len(label_map)  # all indices unique


def test_build_silence_index_bounds_and_determinism(corpus: Path) -> None:
    noise_dir = corpus / "_background_noise_"
    clip_length = SAMPLE_RATE
    noise_length = SAMPLE_RATE * 3

    crops = data.build_silence_index(noise_dir, n=5, clip_length=clip_length, seed=42)
    assert len(crops) == 5
    for path, offset in crops:
        assert path == noise_dir / "white_noise.wav"
        assert 0 <= offset <= noise_length - clip_length

    crops_again = data.build_silence_index(
        noise_dir, n=5, clip_length=clip_length, seed=42
    )
    assert crops == crops_again  # same seed -> identical crops


def test_fix_length_pads_short_waveform() -> None:
    short = np.zeros(100, dtype=np.float32)
    padded = data.fix_length(short, 150)
    assert padded.shape == (150,)
    assert np.array_equal(padded[25:125], short)
    assert np.all(padded[:25] == 0)
    assert np.all(padded[125:] == 0)


def test_fix_length_crops_long_waveform() -> None:
    long_wave = np.arange(200, dtype=np.float32)
    cropped = data.fix_length(long_wave, 150)
    assert cropped.shape == (150,)
    assert np.array_equal(cropped, long_wave[25:175])


def test_fix_length_exact_length_is_unchanged() -> None:
    exact = np.arange(100, dtype=np.float32)
    result = data.fix_length(exact, 100)
    assert np.array_equal(result, exact)


def test_speech_commands_dataset_shapes_and_labels(corpus: Path) -> None:
    config = {
        "seed": 0,
        "sample_rate": SAMPLE_RATE,
        "unknown_ratio": 1.0,
        "silence_ratio": 1.0,
    }

    for split in ("train", "val", "test"):
        ds = data.SpeechCommandsDataset(corpus, split, config, augment=False)
        assert len(ds) > 0

        for idx in (0, len(ds) - 1):
            features, label_idx = ds[idx]
            assert features.shape[0] == 1
            assert features.shape[1] == 40  # default n_mels
            assert 0 <= label_idx < len(ds.label_map)


def test_speech_commands_dataset_augment_not_implemented(corpus: Path) -> None:
    config = {"seed": 0, "sample_rate": SAMPLE_RATE}
    ds = data.SpeechCommandsDataset(corpus, "train", config, augment=True)
    with pytest.raises(NotImplementedError):
        ds[0]


def test_accented_eval_dataset(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_wav(audio_root / "spk01_yes_0.wav", SAMPLE_RATE)
    _write_wav(audio_root / "spk01_xyz_0.wav", SAMPLE_RATE)

    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text(
        "path,label\n"
        "spk01_yes_0.wav,yes\n"
        "spk01_xyz_0.wav,xyz\n"
    )

    config = {"seed": 0, "sample_rate": SAMPLE_RATE}
    ds = data.AccentedEvalDataset(metadata_csv, audio_root, config)
    assert len(ds) == 2

    features, label_idx = ds[0]
    assert features.shape[0] == 1
    assert features.shape[1] == 40
    assert label_idx == ds.label_map["yes"]

    _, unknown_label_idx = ds[1]
    assert unknown_label_idx == ds.label_map[data.UNKNOWN_LABEL]
