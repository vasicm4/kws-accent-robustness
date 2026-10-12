"""Paired speaker-bootstrap comparison of experiments against a baseline.

    uv run python scripts/compare_runs.py e0_strided_stem e1_shift_noise e2_specaug

Reads <run>/predictions.npz (written by scripts/evaluate_accented.py) for every
run of each experiment name. In each bootstrap draw the same speakers are
resampled for every model, so per-speaker difficulty cancels out of the
difference. An experiment's metric in a draw is the mean over its seeds; the
CI therefore covers speaker sampling, not seed-to-seed variation (see
scripts/summarize_runs.py for the seed std).

Reported, as other - baseline, on the 10 target words: official test,
accented set, and the accent gap (official - accented; negative = smaller gap).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import TARGET_WORDS  # noqa: E402
from evaluate import confusion_metrics, group_confusions, paired_speaker_bootstrap  # noqa: E402

N_CLASSES = len(TARGET_WORDS) + 2
TARGET_IDS = list(range(len(TARGET_WORDS)))
METRICS = ("acc", "macro_f1")


def load_predictions(runs_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(runs_dir.glob("*/predictions.npz")):
        config = yaml.safe_load((path.parent / "config.yaml").read_text())
        groups[config["name"]].append(dict(np.load(path)))
    return groups


def set_arrays(run: dict, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, speaker, path) for 'official' (target words only) or 'accented'."""
    keep = run["official_is_target"] if which == "official" else slice(None)
    return tuple(run[f"{which}_{k}"][keep] for k in ("y_true", "y_pred", "speaker", "path"))


def group_draws(
    groups: list[list[dict]], which: str, n_boot: int, rng: np.random.Generator
) -> list[dict[str, np.ndarray]]:
    """Per experiment: mean over its seeds of each metric, per bootstrap draw."""
    y_true, _, speakers, paths = set_arrays(groups[0][0], which)
    y_preds, owner = [], []
    for g, runs in enumerate(groups):
        for run in runs:
            t, p, _, run_paths = set_arrays(run, which)
            if not (np.array_equal(run_paths, paths) and np.array_equal(t, y_true)):
                raise ValueError(f"{which}: runs were scored on different clips")
            y_preds.append(p)
            owner.append(g)
    draws = paired_speaker_bootstrap(
        y_true, y_preds, speakers, N_CLASSES, TARGET_IDS, n_boot, rng
    )
    owner_arr = np.array(owner)
    return [
        {k: np.mean([d[k] for d, o in zip(draws, owner_arr) if o == g], axis=0) for k in METRICS}
        for g in range(len(groups))
    ]


def point(runs: list[dict], which: str, metric: str) -> float:
    """Mean over seeds of the metric on the full set (not bootstrapped)."""
    values = []
    for run in runs:
        y_true, y_pred, _, _ = set_arrays(run, which)
        cm = group_confusions(y_true, y_pred, np.zeros(len(y_true)), N_CLASSES)[0]
        acc, macro_f1 = confusion_metrics(cm, TARGET_IDS)
        values.append(acc if metric == "acc" else macro_f1)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", help="experiment name to compare against")
    parser.add_argument("others", nargs="+", help="experiment names")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()

    available = load_predictions(args.runs_dir)
    names = [args.baseline, *args.others]
    missing = [n for n in names if not available.get(n)]
    if missing:
        sys.exit(f"no predictions.npz for {missing}; run scripts/evaluate_accented.py first")
    groups = [available[n] for n in names]

    rng = np.random.default_rng(args.boot_seed)
    official = group_draws(groups, "official", args.n_boot, rng)
    accented = group_draws(groups, "accented", args.n_boot, rng)

    print(f"paired speaker bootstrap, {args.n_boot} draws; other - {args.baseline}, "
          "mean over seeds; 95% CI")
    print(f"{'experiment':22s}{'seeds':>6s}  {'metric':9s}"
          f"{'official':>26s}{'accented':>26s}{'gap':>26s}")
    for g, name in enumerate(names[1:], start=1):
        for metric in METRICS:
            cells = []
            for which, draws in (("official", official), ("accented", accented)):
                diff = draws[g][metric] - draws[0][metric]
                est = point(groups[g], which, metric) - point(groups[0], which, metric)
                cells.append((est, np.percentile(diff, [2.5, 97.5])))
            gap_est = cells[0][0] - cells[1][0]
            gap_diff = (official[g][metric] - accented[g][metric]) - (
                official[0][metric] - accented[0][metric]
            )
            cells.append((gap_est, np.percentile(gap_diff, [2.5, 97.5])))
            row = "".join(f"{e:+9.4f} [{lo:+.4f}, {hi:+.4f}]" for e, (lo, hi) in cells)
            print(f"{name:22s}{len(groups[g]):6d}  {metric:9s}{row}")


if __name__ == "__main__":
    main()
