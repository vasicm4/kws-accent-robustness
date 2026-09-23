from __future__ import annotations

from pathlib import Path

import pytest

from utils import load_config, save_config

BASELINE = Path(__file__).resolve().parent.parent / "configs" / "baseline.yaml"


def test_baseline_config_has_numeric_floats() -> None:
    config = load_config(BASELINE)
    for key in ("lr", "weight_decay", "dropout", "clip_duration_seconds"):
        assert isinstance(config[key], float), key


def test_overrides_are_applied_and_typed() -> None:
    config = load_config(BASELINE, ["lr=3e-3", "seed=2", "augment=true", "weight_decay=0"])
    assert config["lr"] == pytest.approx(3e-3)
    assert config["seed"] == 2
    assert config["augment"] is True
    assert isinstance(config["weight_decay"], float)


def test_unknown_override_key_raises() -> None:
    with pytest.raises(KeyError):
        load_config(BASELINE, ["learning_rate=3e-3"])


def test_malformed_override_raises() -> None:
    with pytest.raises(ValueError):
        load_config(BASELINE, ["lr"])


def test_save_config_round_trips(tmp_path: Path) -> None:
    config = load_config(BASELINE, ["seed=7"])
    out = tmp_path / "config.yaml"
    save_config(config, out)
    assert load_config(out) == config
