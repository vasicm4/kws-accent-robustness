from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_num_threads(num_threads: int | None) -> None:
    """Intra-op threads for the main process; None keeps PyTorch's default.

    DataLoader workers always run single-threaded, so this only sizes the
    forward/backward pass.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _parse_override_value(raw: str, current: Any) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(current, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
        value = float(value)
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    for override in overrides or []:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"override {override!r} must look like KEY=VALUE")
        if key not in config:
            raise KeyError(f"unknown config key {key!r} in override {override!r}")
        config[key] = _parse_override_value(raw, config[key])

    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
