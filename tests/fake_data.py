"""A tiny fake Speech Commands corpus shared by the data and training tests."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
_seed_counter = itertools.count()


def _write_wav(path: Path, num_samples: int, seed: int | None = None) -> None:
    if seed is None:
        seed = next(_seed_counter)
    rng = np.random.default_rng(seed)
    waveform = rng.uniform(-0.1, 0.1, size=num_samples).astype(np.float32)
    sf.write(path, waveform, SAMPLE_RATE)


def make_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "speech_commands"
    root.mkdir()

    def add(word: str, speaker: str, rep: int) -> str:
        word_dir = root / word
        word_dir.mkdir(exist_ok=True)
        filename = f"{speaker}_nohash_{rep}.wav"
        _write_wav(word_dir / filename, SAMPLE_RATE)
        return f"{word}/{filename}"

    val_lines = [
        add("yes", "spkB", 0),
        add("yes", "spkB", 1),
        add("no", "spkB", 0),
    ]
    add("yes", "spkA", 0)
    add("yes", "spkA", 1)
    add("no", "spkC", 0)
    add("no", "spkC", 1)

    test_lines = [
        add("no", "spkD", 0),
        add("no", "spkD", 1),
    ]

    add("bed", "spkE", 0)
    add("bed", "spkE", 1)
    add("cat", "spkF", 0)
    add("cat", "spkF", 1)

    (root / "validation_list.txt").write_text("\n".join(val_lines) + "\n")
    (root / "testing_list.txt").write_text("\n".join(test_lines) + "\n")

    noise_dir = root / "_background_noise_"
    noise_dir.mkdir()
    _write_wav(noise_dir / "white_noise.wav", SAMPLE_RATE * 3)

    return root
