"""Export a trained run to ONNX, front end included: waveform in, logits out.

    uv run --extra serve python scripts/export_onnx.py runs/<run dir> [--out model.onnx]

The log-mel front end is rebuilt from plain ops so it lives inside the graph
and the server needs neither torch nor torchaudio: the STFT is a strided
conv1d with fixed DFT kernels, followed by the same mel filterbank and dB
conversion as data.extract_features. Input: (batch, 16000) float32 at
16 kHz, already cut or padded to 1 s. Output: (batch, 12) logits. Class
names and the expected sample rate are stored in the model's metadata.
The export is checked against the PyTorch pipeline on a few clips before saving.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data import extract_features  # noqa: E402

AMIN = 1e-10  # torchaudio AmplitudeToDB default floor


class LogMelFrontend(nn.Module):
    """data.extract_features as conv1d + matmul, so it exports to any ONNX opset.

    Matches torchaudio's MelSpectrogram defaults: periodic Hann window,
    center=True with reflect padding, power spectrum, HTK mel scale without
    normalisation, then 10 * log10(max(x, 1e-10)).
    """

    def __init__(self, config: dict):
        super().__init__()
        n_fft = config.get("n_fft", 400)
        self.hop_length = config.get("hop_length", 160)
        self.pad = n_fft // 2
        n_freqs = n_fft // 2 + 1
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_freqs, dtype=torch.float64).unsqueeze(1)
        angle = 2 * math.pi * k * n / n_fft
        # Rows 0..n_freqs-1: real part, n_freqs..: imaginary part (sign irrelevant for power).
        kernels = torch.cat([torch.cos(angle), torch.sin(angle)]) * window
        self.register_buffer("dft", kernels.float().unsqueeze(1))  # (2 * n_freqs, 1, n_fft)
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=float(config["sample_rate"] // 2),
            n_mels=config.get("n_mels", 40),
            sample_rate=config["sample_rate"],
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer("fbank", fbank.T.contiguous())  # (n_mels, n_freqs)
        self.n_freqs = n_freqs

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(waveform.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        spec = nn.functional.conv1d(x, self.dft, stride=self.hop_length)  # (B, 2F, frames)
        real, imag = spec[:, : self.n_freqs], spec[:, self.n_freqs :]
        mel = torch.matmul(self.fbank, real * real + imag * imag)  # (B, n_mels, frames)
        return (10.0 * torch.log10(torch.clamp(mel, min=AMIN))).unsqueeze(1)


class WaveformClassifier(nn.Module):
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.frontend = LogMelFrontend(config)
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(self.frontend(waveform))


def export(model: nn.Module, config: dict, label_map: dict[str, int], out: Path) -> None:
    """Write the waveform -> logits ONNX model with labels in its metadata."""
    wrapped = WaveformClassifier(model, config).eval()
    clip_length = int(config["sample_rate"] * config.get("clip_duration_seconds", 1.0))
    example = (torch.zeros(2, clip_length),)
    # The exporter warns about every torchvision op it can't register.
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            wrapped,
            example,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_shapes={"waveform": {0: torch.export.Dim("batch")}},
            dynamo=True,
            verbose=False,
        )
    program.save(str(out))

    proto = onnx.load(str(out))
    labels = [label for label, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    for key, value in {
        "labels": json.dumps(labels),
        "sample_rate": str(config["sample_rate"]),
        "clip_length": str(clip_length),
    }.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(out))


def torch_logits(model: nn.Module, waveforms: np.ndarray, config: dict) -> np.ndarray:
    """Reference: the training pipeline (torchaudio features, then the model)."""
    features = torch.stack([extract_features(w, config) for w in waveforms])
    with torch.inference_mode():
        return model.eval()(features).numpy()


def onnx_logits(path: Path, waveforms: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"waveform": waveforms.astype(np.float32)})[0]


def main() -> None:
    from evaluate_accented import load_run  # needs the training code; import late

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="default: <run_dir>/model.onnx")
    args = parser.parse_args()
    out = args.out or args.run_dir / "model.onnx"

    model, config, label_map = load_run(args.run_dir)
    export(model, config, label_map, out)

    rng = np.random.default_rng(0)
    waveforms = (0.1 * rng.standard_normal((8, 16000))).astype(np.float32)
    ref, got = torch_logits(model, waveforms, config), onnx_logits(out, waveforms)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    print(f"max |logit diff| vs PyTorch on 8 noise clips: {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
