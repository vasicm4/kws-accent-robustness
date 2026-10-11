"""Train-time augmentation.

Waveform domain (E1): random time shift, then background noise mixed in at a
random SNR from the dataset's _background_noise_ recordings.
Spectrogram domain (E2): SpecAugment frequency and time masks on the log-mel.

Every function takes an explicit numpy Generator so tests and workers control
the randomness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Used when a key is missing from the config, like the feature defaults in data.py.
DEFAULTS = {
    "time_shift_ms": 100,
    "noise_prob": 0.8,
    "noise_snr_db_min": 5.0,
    "noise_snr_db_max": 20.0,
    "spec_freq_masks": 0,
    "spec_freq_mask_width": 7,
    "spec_time_masks": 0,
    "spec_time_mask_width": 10,
}


def time_shift(waveform: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Shift by a uniform offset in [-max_shift, max_shift] samples, zero-filling the gap.

    No wrap-around: the end of a word must not reappear at the start.
    """
    shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
    if shift == 0:
        return waveform
    shifted = np.zeros_like(waveform)
    if shift > 0:
        shifted[shift:] = waveform[:-shift]
    else:
        shifted[:shift] = waveform[-shift:]
    return shifted


def random_crop(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(0, len(noise) - length + 1))
    return noise[start : start + length]


def mix_at_snr(waveform: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise scaled so that RMS(waveform) / RMS(scaled noise) is snr_db.

    The signal RMS is over the whole clip, silence around the word included,
    so the SNR on the word itself is somewhat higher than snr_db.
    """
    signal_rms = np.sqrt(np.mean(waveform**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    if signal_rms == 0 or noise_rms == 0:
        return waveform
    gain = signal_rms / (noise_rms * 10 ** (snr_db / 20))
    return (waveform + gain * noise).astype(np.float32)


def spec_augment(
    features: torch.Tensor,
    freq_masks: int,
    freq_width: int,
    time_masks: int,
    time_width: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """SpecAugment masks on a (1, n_mels, frames) log-mel, without time warping.

    Each mask has a uniform width in [0, max width] and a uniform position.
    Masked cells get the clip's mean value: the features are in dB, where 0
    would be a loud value, not "no information".
    """
    if freq_masks == 0 and time_masks == 0:
        return features
    out = features.clone()
    fill = out.mean()
    _, n_mels, n_frames = out.shape
    for _ in range(freq_masks):
        width = int(rng.integers(0, min(freq_width, n_mels) + 1))
        start = int(rng.integers(0, n_mels - width + 1))
        out[:, start : start + width, :] = fill
    for _ in range(time_masks):
        width = int(rng.integers(0, min(time_width, n_frames) + 1))
        start = int(rng.integers(0, n_frames - width + 1))
        out[:, :, start : start + width] = fill
    return out


class Augmenter:
    """Applies the configured augmentations; one per training dataset."""

    def __init__(self, noise_dir: Path, config: dict):
        cfg = {k: config.get(k, v) for k, v in DEFAULTS.items()}
        sample_rate = config["sample_rate"]
        self.max_shift = int(sample_rate * cfg["time_shift_ms"] / 1000)
        self.noise_prob = cfg["noise_prob"]
        self.snr_range = (cfg["noise_snr_db_min"], cfg["noise_snr_db_max"])
        self.spec = (
            cfg["spec_freq_masks"],
            cfg["spec_freq_mask_width"],
            cfg["spec_time_masks"],
            cfg["spec_time_mask_width"],
        )
        # ~20 MB for the six Speech Commands recordings; loaded once, then shared
        # copy-on-write with the DataLoader workers.
        self.noises = []
        if self.noise_prob > 0:
            for path in sorted(noise_dir.glob("*.wav")):
                noise, sr = sf.read(path, dtype="float32", always_2d=False)
                if sr != sample_rate:
                    raise ValueError(f"{path} is at {sr} Hz, expected {sample_rate} Hz")
                self.noises.append(noise if noise.ndim == 1 else noise.mean(axis=1))
            if not self.noises:
                raise ValueError(f"noise_prob > 0 but no noise files in {noise_dir}")

    def waveform(self, waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        waveform = time_shift(waveform, self.max_shift, rng)
        if self.noises and rng.random() < self.noise_prob:
            noise = self.noises[int(rng.integers(len(self.noises)))]
            noise = random_crop(noise, len(waveform), rng)
            waveform = mix_at_snr(waveform, noise, rng.uniform(*self.snr_range))
        return waveform

    def features(self, features: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        return spec_augment(features, *self.spec, rng)
