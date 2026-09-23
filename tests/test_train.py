from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train
from evaluate import evaluate


def test_evaluate_scores_known_predictions() -> None:
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 1, 0])
    loader = DataLoader(TensorDataset(nn.functional.one_hot(y_pred, 3).float(), y_true), batch_size=3)

    metrics = evaluate(nn.Identity(), loader, nn.CrossEntropyLoss(), n_classes=3)

    assert metrics["acc"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 0.0) / 3)


def test_main_smoke(corpus: Path, tmp_path: Path) -> None:
    run_dir = train.main([
        "--set",
        f"data_root={corpus}",
        f"output_dir={tmp_path / 'runs'}",
        "max_epochs=2",
        "batch_size=4",
        "num_workers=0",
    ])

    for name in ("config.yaml", "train.log", "best.pt", "metrics.json"):
        assert (run_dir / name).exists(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert 1 <= metrics["best_epoch"] <= 2
    assert {"acc", "macro_f1", "loss"} <= metrics["test"].keys()

    checkpoint = torch.load(run_dir / "best.pt")
    assert checkpoint["epoch"] == metrics["best_epoch"]
    assert len(checkpoint["label_map"]) == 12


def _lrs_per_step(schedule: str, steps_per_epoch: int = 4) -> list[float]:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    config = {"lr_schedule": schedule, "warmup_epochs": 1, "max_epochs": 5}
    scheduler = train.make_scheduler(optimizer, config, steps_per_epoch)
    lrs = []
    for _ in range(5 * steps_per_epoch):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_cosine_schedule_warms_up_then_decays() -> None:
    lrs = _lrs_per_step("cosine")

    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    decay = lrs[4:]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert decay[-1] < 0.02


def test_constant_schedule_stays_at_peak_after_warmup() -> None:
    assert _lrs_per_step("constant")[4:] == pytest.approx([1.0] * 16)
