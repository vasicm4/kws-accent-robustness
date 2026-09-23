from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import SpeechCommandsDataset
from evaluate import evaluate
from model import KWSNet
from utils import load_config, make_generator, save_config, seed_worker, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the keyword-spotting CNN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline.yaml",
        help="YAML config file (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override config keys, e.g. --set lr=3e-3 seed=1",
    )
    return parser.parse_args(argv)


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_data_root(raw: str) -> Path:
    data_root = resolve_path(raw)
    if not (data_root / "validation_list.txt").exists():
        raise FileNotFoundError(
            f"Speech Commands not found at {data_root}. "
            "Run: uv run python scripts/download_data.py"
        )
    return data_root


def build_loaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    data_root = Path(config["data_root"])
    train_ds = SpeechCommandsDataset(data_root, "train", config, augment=config["augment"])
    val_ds = SpeechCommandsDataset(data_root, "val", config)
    test_ds = SpeechCommandsDataset(data_root, "test", config)

    num_workers = config["num_workers"]
    common = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=make_generator(config["seed"]),
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.label_map


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    model.train()
    start = time.perf_counter()
    total_loss, correct, seen = 0.0, 0, 0
    for features, labels in loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return {
        "loss": total_loss / seen,
        "acc": correct / seen,
        "seconds": time.perf_counter() - start,
    }


def git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return sha, bool(status.strip())


def make_run_dir(config: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(config["output_dir"]) / f"{stamp}_{config['name']}_seed{config['seed']}"
    run_dir.mkdir(parents=True)
    return run_dir


def make_logger(log_path: Path) -> Callable[[str], None]:
    """Print a line and append it to the run's train.log."""
    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_path, "a") as f:
            f.write(message + "\n")
    return log


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)
    config["data_root"] = str(resolve_data_root(config["data_root"]))

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.yaml")
    log = make_logger(run_dir / "train.log")
    sha, dirty = git_state()
    log(yaml.safe_dump(config, sort_keys=False))
    log(f"run dir: {run_dir}")
    log(f"git: {sha}{' (uncommitted changes)' if dirty else ''}")

    set_seed(config["seed"])
    train_loader, val_loader, test_loader, label_map = build_loaders(config)
    n_classes = len(label_map)
    log(
        f"clips: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} "
        f"| test {len(test_loader.dataset)} | classes {n_classes}"
    )

    model = KWSNet(n_classes=n_classes, dropout=config["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = run_dir / "best.pt"
    history: list[dict] = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    best_val: dict[str, float] = {}

    for epoch in range(1, config["max_epochs"] + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion)
        val_m = evaluate(model, val_loader, criterion, n_classes)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        log(
            f"epoch {epoch:2d} | train_loss {train_m['loss']:.4f} | train_acc {train_m['acc']:.4f} "
            f"| val_loss {val_m['loss']:.4f} | val_acc {val_m['acc']:.4f} "
            f"| val_f1 {val_m['macro_f1']:.4f} | {train_m['seconds']:.1f}s"
        )

        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs, best_val = val_m["macro_f1"], epoch, 0, val_m
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "git_sha": sha,
                    "git_dirty": dirty,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                log(f"early stopping: no val_f1 improvement for {bad_epochs} epochs")
                break

    # The model in memory is from the last epoch; report the best one.
    model.load_state_dict(torch.load(checkpoint_path)["model_state"])
    test_m = evaluate(model, test_loader, criterion, n_classes)
    log(
        f"best epoch {best_epoch} | val_acc {best_val['acc']:.4f} | val_f1 {best_val['macro_f1']:.4f} "
        f"| test_acc {test_m['acc']:.4f} | test_f1 {test_m['macro_f1']:.4f}"
    )

    metrics = {
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "val": best_val,
        "test": test_m,
        "git_sha": sha,
        "git_dirty": dirty,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
