"""Figures for the README and report, from evaluated runs.

    uv run python scripts/make_figures.py --main e0_strided_stem \\
        --experiments e0_strided_stem e1_shift_noise e2_specaug

Reads <run>/accented_metrics.json (scripts/evaluate_accented.py) and, for
the E3 figure, <run>/e3_finetune.json (scripts/finetune_accented.py).
Writes PNGs to reports/figures/. Only aggregate numbers are plotted; no audio.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL, ACCENTED = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(runs_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(runs_dir.glob("*/accented_metrics.json")):
        config = yaml.safe_load((path.parent / "config.yaml").read_text())
        run = json.loads(path.read_text())
        e3 = path.parent / "e3_finetune.json"
        run["e3"] = json.loads(e3.read_text()) if e3.exists() else None
        groups[config["name"]].append(run)
    return groups


def accent_gap(groups: dict, names: list[str], out: Path) -> None:
    """Official vs. accented accuracy per experiment: seed mean, seed range as error bar."""
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * len(names)))
    y = np.arange(len(names))[::-1]
    for offset, key, color, label in (
        (0.12, "official_test_targets", OFFICIAL, "Speech Commands test"),
        (-0.12, "accented", ACCENTED, "Serbian-accented"),
    ):
        values = [[r[key]["acc"] for r in groups[n]] for n in names]
        means = np.array([np.mean(v) for v in values])
        lo = np.maximum(means - np.array([min(v) for v in values]), 0)
        hi = np.maximum(np.array([max(v) for v in values]) - means, 0)
        ax.errorbar(means, y + offset, xerr=[lo, hi], fmt="o", ms=7, color=color,
                    ecolor=color, elinewidth=2, capsize=0, label=label)
    ax.set_yticks(y, [f"{n}  (n={len(groups[n])})" for n in names])
    ax.set_xlabel("Accuracy on the 10 target words (mean over seeds; bar = min-max)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def per_word(runs: list[dict], out: Path) -> None:
    words = list(runs[0]["official_per_word_acc"])
    official = np.mean([[r["official_per_word_acc"][w] for w in words] for r in runs], axis=0)
    accented = np.mean([[r["per_word_acc"][w] for w in words] for r in runs], axis=0)
    order = np.argsort(accented - official)  # biggest drop first
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(words))[::-1]
    for i, yi in zip(order, y):
        ax.plot([official[i], accented[i]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(official[order], y, s=45, color=OFFICIAL, label="Speech Commands test", zorder=2)
    ax.scatter(accented[order], y, s=45, color=ACCENTED, label="Serbian-accented", zorder=2)
    ax.set_yticks(y, [words[i] for i in order])
    ax.set_xlabel(f"Accuracy per word (mean over {len(runs)} seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def per_speaker(runs: list[dict], out: Path) -> None:
    speakers = list(runs[0]["per_speaker_acc"])
    acc = np.mean([[r["per_speaker_acc"][s] for s in speakers] for r in runs], axis=0)
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(np.arange(len(speakers)), acc[order], width=0.7, color=ACCENTED)
    ax.set_xticks(np.arange(len(speakers)), [speakers[i] for i in order], rotation=60, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"Accuracy (mean over {len(runs)} seeds)")
    ax.set_title("Serbian-accented set, per speaker", loc="left", color=INK, fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)


def confusion(runs: list[dict], key: str, title: str, out: Path) -> None:
    labels = runs[0]["labels"]
    cm = np.sum([np.array(r[key]["confusion"]) for r in runs], axis=0)
    rows = cm.sum(axis=1) > 0
    cm, row_labels = cm[rows], [l for l, keep in zip(labels, rows) if keep]
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.42 * len(row_labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), [l.strip("_") for l in labels], rotation=60, fontsize=8)
    ax.set_yticks(range(len(row_labels)), [l.strip("_") for l in row_labels], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def e3(runs: list[dict], out: Path) -> bool:
    results = [res for r in runs if r["e3"] for res in r["e3"]["results"]]
    if not results:
        return False
    ks = sorted({r["k"] for r in results})
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(ks))[::-1]
    base = [np.mean([r["heldout_base"]["acc"] for r in results if r["k"] == k]) for k in ks]
    ft = [np.mean([r["heldout_finetuned"]["acc"] for r in results if r["k"] == k]) for k in ks]
    for b, f, yi in zip(base, ft, y):
        ax.annotate("", xy=(f, yi), xytext=(b, yi),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.5})
    ax.scatter(base, y, s=45, color=MUTED, label="before fine-tuning", zorder=2)
    ax.scatter(ft, y, s=45, color=ACCENTED, label="after fine-tuning", zorder=2)
    # Keep a few points of context: a tight axis would make 0.7 points look huge.
    ax.set_xlim(min(base + ft) - 0.03, min(1.0, max(base + ft) + 0.03))
    n = len(results) // len(ks)
    ax.set_yticks(y, [f"k = {k} clip{'s' if k > 1 else ''}/word" for k in ks])
    ax.set_xlabel(f"Held-out accented accuracy (mean of {n} base-seed x split runs)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(out)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", required=True, help="experiment for the per-word/speaker figures")
    parser.add_argument("--experiments", nargs="+", required=True, help="rows of the accent-gap figure")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "figures")
    args = parser.parse_args()

    groups = load(args.runs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_runs = groups[args.main]
    written = [
        args.out_dir / "accent_gap.png",
        args.out_dir / "per_word.png",
        args.out_dir / "per_speaker.png",
        args.out_dir / "confusion_accented.png",
        args.out_dir / "confusion_official.png",
    ]
    accent_gap(groups, args.experiments, written[0])
    per_word(main_runs, written[1])
    per_speaker(main_runs, written[2])
    confusion(main_runs, "accented_report",
              f"Serbian-accented set, row-normalised ({args.main}, {len(main_runs)} seeds pooled)",
              written[3])
    confusion(main_runs, "official_test_12class",
              f"Speech Commands test, row-normalised ({args.main}, {len(main_runs)} seeds pooled)",
              written[4])
    if e3(main_runs, args.out_dir / "e3_finetune.png"):
        written.append(args.out_dir / "e3_finetune.png")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
