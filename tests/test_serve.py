"""The FastAPI app returns the exported model's prediction for a WAV body."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from data import TARGET_WORDS, build_label_map  # noqa: E402
from export_onnx import export, torch_logits  # noqa: E402
from model import KWSNet  # noqa: E402
from serve.app import create_app  # noqa: E402

CONFIG = {"sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 40}


def _wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestClient, KWSNet]:
    torch.manual_seed(0)
    model = KWSNet(n_classes=12).eval()
    path = tmp_path_factory.mktemp("serve") / "model.onnx"
    export(model, CONFIG, build_label_map(TARGET_WORDS), path)
    return TestClient(create_app(path)), model


def test_predict_matches_pytorch(served: tuple[TestClient, KWSNet]) -> None:
    client, model = served
    # 1.2 s: the app must centre-cut it to 1 s, like the training pipeline.
    waveform = (0.1 * np.random.default_rng(0).standard_normal(19200)).astype(np.float32)
    expected = torch_logits(model, waveform[2400:18400][np.newaxis], CONFIG)[0]

    response = client.post("/predict", content=_wav_bytes(waveform, 16000))

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == list(build_label_map(TARGET_WORDS))[int(expected.argmax())]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["confidence"] == max(body["probabilities"].values())


@pytest.mark.parametrize(
    "content",
    [b"", b"not audio", _wav_bytes(np.zeros(8000, dtype=np.float32), 8000)],
    ids=["empty", "garbage", "8kHz"],
)
def test_predict_rejects_bad_input(served: tuple[TestClient, KWSNet], content: bytes) -> None:
    client, _ = served
    assert client.post("/predict", content=content).status_code == 400


def test_health(served: tuple[TestClient, KWSNet]) -> None:
    client, _ = served
    body = client.get("/health").json()
    assert body["status"] == "ok" and len(body["labels"]) == 12
