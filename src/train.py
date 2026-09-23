from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from utils import load_config, set_seed

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


def build_loaders(config):
    pass

def worker_init(worker_id):
    pass

def evaluate():
    pass

def resolve_data_root(raw: str) -> Path:
    data_root = Path(raw).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    if not (data_root / "validation_list.txt").exists():
        raise FileNotFoundError(
            f"Speech Commands not found at {data_root}. "
            "Run: uv run python scripts/download_data.py"
        )
    return data_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)
    config["data_root"] = str(resolve_data_root(config["data_root"]))
    print(yaml.safe_dump(config, sort_keys=False))

    set_seed(config["seed"])
    # TODO: build loaders, model, optimizer; train with early stopping.


if __name__ == "__main__":
    main()
