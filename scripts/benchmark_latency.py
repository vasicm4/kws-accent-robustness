"""Per-clip inference latency: PyTorch vs. ONNX Runtime, batch size 1, CPU.

    uv run --extra serve python scripts/benchmark_latency.py runs/<run dir>

Both paths start from a 1 s waveform already in memory and end at logits:
PyTorch runs data.extract_features (torchaudio) and the model; ONNX Runtime
runs <run>/model.onnx, which contains the front end (scripts/export_onnx.py,
exported first if missing). File reading and resampling are excluded.
Clips are real Speech Commands test clips. Each setting is measured at 1
thread (one core, as on a small device) and at the library default.
Results are printed and written to reports/latency.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data import SpeechCommandsDataset, extract_features, fix_length, load_waveform  # noqa: E402
from evaluate_accented import load_run  # noqa: E402
from export_onnx import export  # noqa: E402
from train import resolve_data_root  # noqa: E402


def time_calls(fn, inputs: list, warmup: int) -> np.ndarray:
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        start = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - start)
    return np.array(times) * 1000


def summary(ms: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--clips", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "latency.json")
    args = parser.parse_args()

    model, config, label_map = load_run(args.run_dir)
    onnx_path = args.run_dir / "model.onnx"
    if not onnx_path.exists():
        export(model, config, label_map, onnx_path)

    clip_length = int(config["sample_rate"] * config.get("clip_duration_seconds", 1.0))
    test = SpeechCommandsDataset(resolve_data_root(config["data_root"]), "test", config)
    rng = np.random.default_rng(0)
    picks = rng.choice([i for i, e in enumerate(test.index) if e.offset is None], args.clips, replace=False)
    waveforms = [
        fix_length(load_waveform(test.index[i].path, config["sample_rate"]), clip_length)
        for i in picks
    ]

    def torch_infer(w: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(extract_features(w, config).unsqueeze(0)).numpy()

    default_threads = torch.get_num_threads()
    results = {}
    for threads in (1, default_threads):
        torch.set_num_threads(threads)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

        def onnx_infer(w: np.ndarray) -> np.ndarray:
            return session.run(["logits"], {"waveform": w[np.newaxis]})[0]

        agree = np.mean([torch_infer(w).argmax() == onnx_infer(w).argmax() for w in waveforms])
        for name, fn in (("pytorch", torch_infer), ("onnxruntime", onnx_infer)):
            results[f"{name}_threads{threads}"] = summary(time_calls(fn, waveforms, args.warmup))
        results[f"top1_agreement_threads{threads}"] = float(agree)
    torch.set_num_threads(default_threads)

    report = {
        "run": args.run_dir.name,
        "clips": args.clips,
        "batch_size": 1,
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "onnx_size_kib": onnx_path.stat().st_size / 1024,
        "params": sum(p.numel() for p in model.parameters()),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"run {args.run_dir.name}: {report['params']:,} params, "
          f"ONNX {report['onnx_size_kib']:.0f} KiB, batch 1, {args.clips} clips")
    print(f"{'':24s}{'p50 ms':>9s}{'p95 ms':>9s}{'mean ms':>9s}")
    for key, r in results.items():
        if key.startswith("top1"):
            print(f"{key:24s}{r:9.3f}")
        else:
            print(f"{key:24s}{r['p50_ms']:9.3f}{r['p95_ms']:9.3f}{r['mean_ms']:9.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
