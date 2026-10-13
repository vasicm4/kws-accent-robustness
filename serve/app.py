"""Keyword-spotting inference API on ONNX Runtime.

    MODEL_PATH=runs/<run>/model.onnx uv run --extra serve uvicorn serve.app:create_app --factory

POST /predict with a WAV file as the request body (16 kHz, mono or stereo,
any length; it is centre-cut or zero-padded to 1 s like the training data):

    curl --data-binary @clip.wav -H "Content-Type: audio/wav" localhost:8000/predict

Only numpy, soundfile, onnxruntime and FastAPI are needed: the log-mel front
end is part of the ONNX graph (scripts/export_onnx.py).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request

MAX_BYTES = 1_000_000  # ~30 s of 16-bit 16 kHz mono; clips are 1 s


def fix_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Centre-cut or zero-pad to target_length (same as data.fix_length)."""
    if len(waveform) > target_length:
        start = (len(waveform) - target_length) // 2
        return waveform[start : start + target_length]
    pad = target_length - len(waveform)
    return np.pad(waveform, (pad // 2, pad - pad // 2))


def create_app(model_path: str | Path | None = None) -> FastAPI:
    model_path = Path(model_path or os.environ.get("MODEL_PATH", "model.onnx"))
    options = ort.SessionOptions()
    threads = os.environ.get("ORT_THREADS")
    if threads:
        options.intra_op_num_threads = int(threads)
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    meta = session.get_modelmeta().custom_metadata_map
    labels: list[str] = json.loads(meta["labels"])
    sample_rate = int(meta["sample_rate"])
    clip_length = int(meta["clip_length"])

    app = FastAPI(title="kws-accent-robustness")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_path.name, "labels": labels}

    @app.post("/predict")
    async def predict(request: Request) -> dict:
        body = await request.body()
        if not body:
            raise HTTPException(400, "send a WAV file as the request body")
        if len(body) > MAX_BYTES:
            raise HTTPException(413, f"audio larger than {MAX_BYTES} bytes")
        try:
            waveform, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
        except Exception as exc:  # soundfile raises several types for bad input
            raise HTTPException(400, f"could not decode audio: {exc}") from exc
        if sr != sample_rate:
            raise HTTPException(400, f"expected {sample_rate} Hz audio, got {sr} Hz")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        batch = fix_length(waveform, clip_length)[np.newaxis].astype(np.float32)

        logits = session.run(["logits"], {"waveform": batch})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(probs.argmax())
        return {
            "label": labels[best],
            "confidence": float(probs[best]),
            "probabilities": {label: float(p) for label, p in zip(labels, probs)},
        }

    return app
