"""The exported ONNX model (front end included) must match the PyTorch pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data import build_label_map, extract_features, TARGET_WORDS  # noqa: E402
from export_onnx import LogMelFrontend, export, onnx_logits, torch_logits  # noqa: E402
from model import KWSNet  # noqa: E402

CONFIG = {"sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 40}


def _waveforms(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    # Tones get a -60 dB noise floor like any real recording: in a perfectly
    # clean tone, bins far from it hold ~1e-10 of power, where float32 DFT
    # and FFT rounding differ by tenths of a dB.
    tones = [
        0.3 * np.sin(2 * np.pi * f * t) + 3e-4 * rng.standard_normal(16000)
        for f in (220.0, 1000.0, 3500.0)
    ]
    noise = [0.05 * rng.standard_normal(16000) for _ in range(n - len(tones) - 1)]
    silence_padded = np.zeros(16000)
    silence_padded[4000:12000] = 0.2 * rng.standard_normal(8000)
    return np.stack([*tones, *noise, silence_padded]).astype(np.float32)


def test_frontend_matches_torchaudio_features() -> None:
    waveforms = _waveforms()
    expected = torch.stack([extract_features(w, CONFIG) for w in waveforms])

    got = LogMelFrontend(CONFIG)(torch.from_numpy(waveforms))

    assert got.shape == expected.shape == (len(waveforms), 1, 40, 101)
    # dB values (range about -100..40); the only difference is float32 DFT
    # vs. FFT rounding, < 0.01 dB.
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize("strided_stem", [False, True])
def test_onnx_logits_match_pytorch(tmp_path: Path, strided_stem: bool) -> None:
    torch.manual_seed(0)
    model = KWSNet(n_classes=12, strided_stem=strided_stem).eval()
    path = tmp_path / "model.onnx"
    export(model, CONFIG, build_label_map(TARGET_WORDS), path)
    waveforms = _waveforms()

    expected = torch_logits(model, waveforms, CONFIG)
    got = onnx_logits(path, waveforms)

    assert got.shape == (len(waveforms), 12)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(got.argmax(axis=1), expected.argmax(axis=1))

