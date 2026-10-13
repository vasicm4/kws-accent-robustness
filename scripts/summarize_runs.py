"""Mean ± std over seeds for each experiment name in runs/.

    uv run python scripts/summarize_runs.py                 # every run
    uv run python scripts/summarize_runs.py e0_channels_last e0_strided_stem

Reads <run>/metrics.json and, when present, <run>/accented_metrics.json
(written by scripts/evaluate_accented.py). Runs are grouped by the config's
`name`; std is the sample std (ddof=1) across seeds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

COLUMNS = {
    # header: (format, getter)
    "val_f1": (".4f", lambda m, a: m["val"]["macro_f1"]),
    "test_f1": (".4f", lambda m, a: m["test"]["macro_f1"]),
    "acc_acc": (".4f", lambda m, a: a["accented"]["acc"] if a else None),
    "acc_f1": (".4f", lambda m, a: a["accented"]["macro_f1"] if a else None),
    "gap_acc": (".4f", lambda m, a: a["gap"]["acc"] if a else None),
    "gap_f1": (".4f", lambda m, a: a["gap"]["macro_f1"] if a else None),
    "best_ep": (".1f", lambda m, a: m["best_epoch"]),
    "epoch_s": (".1f", lambda m, a: float(np.median([h["train_seconds"] for h in m["history"]]))),
}


def load_runs(runs_dir: Path) -> dict[str, list[tuple[int, dict, dict | None]]]:
    groups: dict[str, list[tuple[int, dict, dict | None]]] = defaultdict(list)
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        accented_path = run_dir / "accented_metrics.json"
        accented = json.loads(accented_path.read_text()) if accented_path.exists() else None
        groups[config["name"]].append(
            (config["seed"], json.loads(metrics_path.read_text()), accented)
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="experiment names (default: all)")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    args = parser.parse_args()

    groups = load_runs(args.runs_dir)
    names = args.names or sorted(groups)

    print(f"{'experiment':22s}{'seeds':>8s}" + "".join(f"{c:>18s}" for c in COLUMNS))
    for name in names:
        runs = sorted(groups.get(name, []), key=lambda r: r[0])
        if not runs:
            print(f"{name:22s}  (no runs)")
            continue
        seeds = ",".join(str(seed) for seed, _, _ in runs)
        cells = []
        for fmt, get in COLUMNS.values():
            values = [v for _, m, a in runs if (v := get(m, a)) is not None]
            if not values:
                cells.append(f"{'-':>18s}")
            elif len(values) == 1:
                cells.append(f"{values[0]:>18{fmt}}")
            else:
                cell = f"{np.mean(values):{fmt}} ± {np.std(values, ddof=1):{fmt}}"
                cells.append(f"{cell:>18s}")
        print(f"{name:22s}{seeds:>8s}" + "".join(cells))


if __name__ == "__main__":
    main()
