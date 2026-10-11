from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from augment import Augmenter, mix_at_snr, spec_augment, time_shift
from fake_data import SAMPLE_RATE


def test_time_shift_zero_fills_without_wrapping() -> None:
    waveform = np.arange(1, 11, dtype=np.float32)
    for seed in range(20):
        shifted = time_shift(waveform, max_shift=3, rng=np.random.default_rng(seed))
        assert shifted.shape == waveform.shape
        nonzero = shifted[shifted != 0]
        # The surviving samples are a contiguous, in-order run of the original.
        start = int(nonzero[0]) - 1
        np.testing.assert_array_equal(nonzero, waveform[start : start + len(nonzero)])
        assert len(nonzero) >= len(waveform) - 3


def test_time_shift_disabled_returns_input() -> None:
    waveform = np.ones(5, dtype=np.float32)
    assert time_shift(waveform, 0, np.random.default_rng(0)) is waveform


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 20.0])
def test_mix_at_snr_hits_requested_snr(snr_db: float) -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.3, 16000).astype(np.float32)
    noise = rng.normal(0, 0.05, 16000).astype(np.float32)

    added = mix_at_snr(signal, noise, snr_db) - signal

    measured = 20 * np.log10(np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(added**2)))
    assert measured == pytest.approx(snr_db, abs=1e-3)


def test_mix_at_snr_leaves_silent_clip_alone() -> None:
    silent = np.zeros(100, dtype=np.float32)
    noise = np.ones(100, dtype=np.float32)
    np.testing.assert_array_equal(mix_at_snr(silent, noise, 10.0), silent)


def test_spec_augment_masks_with_clip_mean() -> None:
    features = torch.randn(1, 40, 101) * 10 - 50
    original = features.clone()
    out = spec_augment(features, 2, 7, 2, 10, np.random.default_rng(0))

    assert out.shape == features.shape
    assert torch.equal(features, original)  # input not modified in place
    changed = out != features
    assert changed.any()
    assert torch.allclose(out[changed], features.mean().expand(int(changed.sum())))
    # At most 2 x 7 mel bins and 2 x 10 frames can be fully masked.
    assert (changed.all(dim=2)).sum() <= 14
    assert (changed.all(dim=1)).sum() <= 20


def test_spec_augment_off_returns_input() -> None:
    features = torch.randn(1, 40, 101)
    assert spec_augment(features, 0, 7, 0, 10, np.random.default_rng(0)) is features


def test_augmenter_without_noise_needs_no_noise_files(tmp_path: Path) -> None:
    augmenter = Augmenter(tmp_path, {"sample_rate": SAMPLE_RATE, "noise_prob": 0.0})
    waveform = np.ones(SAMPLE_RATE, dtype=np.float32)
    out = augmenter.waveform(waveform, np.random.default_rng(0))
    assert out.shape == waveform.shape


def test_augmenter_requires_noise_files_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no noise files"):
        Augmenter(tmp_path, {"sample_rate": SAMPLE_RATE, "noise_prob": 0.5})
