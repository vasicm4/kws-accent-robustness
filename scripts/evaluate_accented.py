"""Evaluate a trained run on the accented clips and report the accent gap.

    uv run python scripts/evaluate_accented.py runs/20260923-215801_e0_baseline_seed0

Both sets are scored on the 10 target words only: the accented set has no
_silence_/_unknown_ clips, so the official test set is restricted to the
same words to make the gap like-for-like. Predicting _silence_ or
_unknown_ for a target word still counts as an error. Results go to
<run>/accented_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import TARGET_WORDS, AccentedEvalDataset, SpeechCommandsDataset  # noqa: E402
from evaluate import predict  # noqa: E402
from train import build_model, resolve_data_root  # noqa: E402

DEFAULT_CLIPS = REPO_ROOT / "data_accented" / "clips.csv"


def load_run(run_dir: Path) -> tuple[nn.Module, dict, dict[str, int]]:
    checkpoint = torch.load(run_dir / "best.pt", weights_only=False)
    config = dict(checkpoint["config"])
    # Runs from before these options existed trained the plain model.
    config.setdefault("strided_stem", False)
    config.setdefault("channels_last", False)
    label_map = checkpoint["label_map"]
    model = build_model(config, len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), config, label_map


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    target_ids = list(range(len(TARGET_WORDS)))
    return {
        "n": int(len(y_true)),
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=target_ids, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    model, config, label_map = load_run(args.run_dir)
    criterion = nn.CrossEntropyLoss()
    id_to_label = {i: label for label, i in label_map.items()}
    target_ids = {label_map[w] for w in TARGET_WORDS}

    official = SpeechCommandsDataset(
        resolve_data_root(config["data_root"]), "test", config
    )
    official_targets = [i for i, e in enumerate(official.index) if e.label in TARGET_WORDS]
    y_true_o, y_pred_o, _ = predict(
        model,
        DataLoader(Subset(official, official_targets), batch_size=args.batch_size),
        criterion,
    )

    accented = AccentedEvalDataset(args.clips, args.clips.parent, config)
    y_true_a, y_pred_a, _ = predict(
        model, DataLoader(accented, batch_size=args.batch_size), criterion
    )
    assert set(y_true_a) <= target_ids

    official_m = target_metrics(y_true_o, y_pred_o)
    accented_m = target_metrics(y_true_a, y_pred_a)

    per_speaker: dict[str, list[bool]] = defaultdict(list)
    per_word: dict[str, list[bool]] = defaultdict(list)
    confusions: dict[str, int] = defaultdict(int)
    for row, t, p in zip(accented.rows, y_true_a, y_pred_a):
        per_speaker[row["speaker"]].append(t == p)
        per_word[row["label"]].append(t == p)
        if t != p:
            confusions[f"{id_to_label[t]} -> {id_to_label[p]}"] += 1

    result = {
        "run": args.run_dir.name,
        "clips": str(args.clips.relative_to(REPO_ROOT)),
        "official_test_targets": official_m,
        "accented": accented_m,
        "gap": {k: official_m[k] - accented_m[k] for k in ("acc", "macro_f1")},
        "per_speaker_acc": {s: float(np.mean(v)) for s, v in sorted(per_speaker.items())},
        "per_word_acc": {
            w: float(np.mean(per_word[w])) for w in TARGET_WORDS if w in per_word
        },
        "official_per_word_acc": {
            w: float(np.mean(y_pred_o[y_true_o == label_map[w]] == label_map[w]))
            for w in TARGET_WORDS
        },
        "top_confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:15]),
    }
    (args.run_dir / "accented_metrics.json").write_text(json.dumps(result, indent=2))

    print(f"run {args.run_dir.name}")
    print(f"{'':22s}{'n':>6s}{'acc':>8s}{'macro_f1':>10s}")
    for name, m in (("official test (10 w)", official_m), ("accented", accented_m)):
        print(f"{name:22s}{m['n']:6d}{m['acc']:8.4f}{m['macro_f1']:10.4f}")
    print(f"{'gap':22s}{'':6s}{result['gap']['acc']:8.4f}{result['gap']['macro_f1']:10.4f}")
    print("\nper word (official -> accented):")
    for w in TARGET_WORDS:
        print(f"  {w:6s} {result['official_per_word_acc'][w]:.3f} -> "
              f"{result['per_word_acc'].get(w, float('nan')):.3f}")
    print("\nper speaker:", "  ".join(f"{s} {a:.2f}" for s, a in result["per_speaker_acc"].items()))
    print("\ntop confusions:", ", ".join(f"{k} ({v})" for k, v in result["top_confusions"].items()))
    print(f"\nwrote {args.run_dir / 'accented_metrics.json'}")


if __name__ == "__main__":
    main()
