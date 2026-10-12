"""E3: few-shot fine-tuning on a few accented speakers, tested on the others.

    uv run python scripts/finetune_accented.py runs/<base run> [runs/<base run> ...]

For each base run, each split and each k:

1. Draw `--n-adapt` adaptation speakers (only from speakers with 3 reps, so
   k=3 is possible) with the split seed; every other speaker is held out.
2. Fine-tune the base checkpoint on k clips per word from each adaptation
   speaker (reps 1..k). Every batch mixes `--accented-batch` accented clips
   with `--replay-batch` Speech Commands training clips, so the model keeps
   the 12-class task (silence/unknown included) while it adapts. BatchNorm
   stays in training mode. Freezing it (eval-mode BN) was checked on the
   official *validation* split (seed-0 strided-stem base, k=3, 20 and 200
   steps, lr 1e-4 and 3e-5) and always forgot more: 0.930-0.936 val acc vs.
   0.937-0.944 with BN updating, from 0.946 before fine-tuning.
3. Score the base and the fine-tuned model on the held-out speakers (10
   target words, speaker-level paired bootstrap of the difference) and on the
   full official test set (12 classes, to see forgetting).

The fine-tuning settings are fixed in advance and not tuned: there is no
accented validation set, and tuning on the held-out speakers would leak.
Results go to <base run>/e3_finetune.json; a summary over all base runs,
splits and k is printed at the end.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data import TARGET_WORDS, AccentedEvalDataset, SpeechCommandsDataset  # noqa: E402
from evaluate import paired_speaker_bootstrap  # noqa: E402
from evaluate_accented import DEFAULT_CLIPS, load_run, percentile_ci  # noqa: E402
from train import resolve_data_root  # noqa: E402
from utils import make_generator, seed_worker, set_seed  # noqa: E402

TARGET_IDS = list(range(len(TARGET_WORDS)))
METRICS = ("acc", "macro_f1")


def stack(dataset, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """All features and labels of a dataset as two tensors (feature extraction runs once)."""
    features, labels = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, num_workers=4):
        features.append(x)
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def predict_tensor(model: nn.Module, features: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat(
        [model(features[i : i + batch_size]).argmax(dim=1) for i in range(0, len(features), batch_size)]
    ).numpy()


def scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def choose_adaptation_speakers(rows: list[dict], n_adapt: int, split_seed: int) -> list[str]:
    reps = {}
    for row in rows:
        reps[row["speaker"]] = max(reps.get(row["speaker"], 0), int(row["rep"]))
    eligible = sorted(s for s, r in reps.items() if r >= 3)
    rng = np.random.default_rng(split_seed)
    return sorted(rng.choice(eligible, size=n_adapt, replace=False).tolist())


def finetune(
    model: nn.Module,
    adapt_x: torch.Tensor,
    adapt_y: torch.Tensor,
    replay: DataLoader,
    args: argparse.Namespace,
    weight_decay: float,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    replay_iter = iter(replay)
    for _ in range(args.steps):
        try:
            replay_x, replay_y = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay)
            replay_x, replay_y = next(replay_iter)
        pick = torch.randint(len(adapt_x), (args.accented_batch,))
        x = torch.cat([adapt_x[pick], replay_x])
        y = torch.cat([adapt_y[pick], replay_y])
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return model.eval()


def run_base(base_dir: Path, args: argparse.Namespace) -> list[dict]:
    base_model, config, label_map = load_run(base_dir)
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    n_classes = len(label_map)
    data_root = resolve_data_root(config["data_root"])

    accented = AccentedEvalDataset(args.clips, args.clips.parent, config)
    acc_x, acc_y = stack(accented)
    rows = accented.rows
    speakers = np.array([r["speaker"] for r in rows])
    reps = np.array([int(r["rep"]) for r in rows])

    official_x, official_y = stack(SpeechCommandsDataset(data_root, "test", config))
    official_y = official_y.numpy()
    base_official = scores(official_y, predict_tensor(base_model, official_x), list(range(n_classes)))
    base_acc_pred = predict_tensor(base_model, acc_x)

    replay_ds = SpeechCommandsDataset(data_root, "train", config, augment=config.get("augment", False))
    replay = DataLoader(
        replay_ds,
        batch_size=args.replay_batch,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=make_generator(config["seed"]),
    )

    results = []
    for split, k in itertools.product(range(args.splits), args.k):
        adapt_speakers = choose_adaptation_speakers(rows, args.n_adapt, split_seed=split)
        is_adapt = np.isin(speakers, adapt_speakers)
        train_mask = is_adapt & (reps <= k)
        test_mask = ~is_adapt

        base_model.load_state_dict(base_state)
        model = finetune(
            base_model, acc_x[train_mask], acc_y[train_mask], replay, args,
            weight_decay=config["weight_decay"], seed=1000 * config["seed"] + 10 * split + k,
        )
        ft_acc_pred = predict_tensor(model, acc_x)
        ft_official = scores(official_y, predict_tensor(model, official_x), list(range(n_classes)))

        y_true = acc_y.numpy()[test_mask]
        base_held = scores(y_true, base_acc_pred[test_mask], TARGET_IDS)
        ft_held = scores(y_true, ft_acc_pred[test_mask], TARGET_IDS)
        base_draws, ft_draws = paired_speaker_bootstrap(
            y_true, [base_acc_pred[test_mask], ft_acc_pred[test_mask]], speakers[test_mask],
            n_classes, TARGET_IDS, args.n_boot, np.random.default_rng(args.boot_seed),
        )
        result = {
            "base_run": base_dir.name,
            "split": split,
            "k": k,
            "adapt_speakers": adapt_speakers,
            "n_adapt_clips": int(train_mask.sum()),
            "n_test_speakers": int(len(set(speakers[test_mask]))),
            "n_test_clips": int(test_mask.sum()),
            "heldout_base": base_held,
            "heldout_finetuned": ft_held,
            "heldout_delta": {m: ft_held[m] - base_held[m] for m in METRICS},
            "heldout_delta_ci95": {m: percentile_ci(ft_draws[m] - base_draws[m]) for m in METRICS},
            "official_base": base_official,
            "official_finetuned": ft_official,
            "official_delta": {m: ft_official[m] - base_official[m] for m in METRICS},
            "heldout_per_speaker_acc": {
                s: {
                    "base": float((base_acc_pred[speakers == s] == acc_y.numpy()[speakers == s]).mean()),
                    "finetuned": float((ft_acc_pred[speakers == s] == acc_y.numpy()[speakers == s]).mean()),
                }
                for s in sorted(set(speakers[test_mask]))
            },
        }
        results.append(result)
        ci = result["heldout_delta_ci95"]["acc"]
        print(
            f"{base_dir.name} split {split} k={k}: held-out acc {base_held['acc']:.4f} -> "
            f"{ft_held['acc']:.4f} ({result['heldout_delta']['acc']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]) "
            f"| official 12-class acc {base_official['acc']:.4f} -> {ft_official['acc']:.4f}",
            flush=True,
        )

    settings = {k: getattr(args, k) for k in ("steps", "lr", "accented_batch", "replay_batch", "n_adapt", "splits", "k")}
    (base_dir / "e3_finetune.json").write_text(
        json.dumps({"settings": settings, "results": results}, indent=2)
    )
    return results


def summarize(results: list[dict]) -> None:
    print(f"\nsummary: mean ± std over {len({r['base_run'] for r in results})} base runs x "
          f"{len({r['split'] for r in results})} splits (held-out speakers, 10 target words)")
    print(f"{'k':>3s}{'clips':>7s}{'base acc':>18s}{'ft acc':>18s}{'delta acc':>18s}"
          f"{'delta f1':>18s}{'official d acc':>18s}")
    for k in sorted({r["k"] for r in results}):
        rs = [r for r in results if r["k"] == k]

        def cell(values: list[float], sign: str = "") -> str:
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            return f"{np.mean(values):{sign}.4f} ± {std:.4f}"

        print(
            f"{k:3d}{rs[0]['n_adapt_clips']:7d}"
            f"{cell([r['heldout_base']['acc'] for r in rs]):>18s}"
            f"{cell([r['heldout_finetuned']['acc'] for r in rs]):>18s}"
            f"{cell([r['heldout_delta']['acc'] for r in rs], '+'):>18s}"
            f"{cell([r['heldout_delta']['macro_f1'] for r in rs], '+'):>18s}"
            f"{cell([r['official_delta']['acc'] for r in rs], '+'):>18s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_runs", type=Path, nargs="+")
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3], help="clips per word per adaptation speaker")
    parser.add_argument("--n-adapt", type=int, default=5, help="adaptation speakers per split")
    parser.add_argument("--splits", type=int, default=3, help="random adaptation/test splits (seeds 0..n-1)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accented-batch", type=int, default=32)
    parser.add_argument("--replay-batch", type=int, default=96)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=0)
    args = parser.parse_args()

    results = []
    for base_dir in args.base_runs:
        results.extend(run_base(base_dir, args))
    summarize(results)


if __name__ == "__main__":
    main()
