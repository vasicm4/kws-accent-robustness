"""Profile the training step on the E0 setup and write the report to reports/profiling/.

Three modes, all on the same model, optimizer and DataLoader as train.py:

    # epoch time split into waiting-for-data vs compute (used for the thread sweep)
    uv run python scripts/profile_epoch.py --mode epoch --set num_threads=4 num_workers=4

    # torch.profiler operator table over 50 training steps
    uv run python scripts/profile_epoch.py --mode torch

    # cProfile over one full epoch (num_workers=0 shows the data pipeline too)
    uv run python scripts/profile_epoch.py --mode cprofile --set num_workers=0

Pass --tag to keep reports from different settings apart.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import platform
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import train  # noqa: E402
from utils import load_config, set_num_threads, set_seed  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "profiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["epoch", "torch", "cprofile"], required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--set", dest="overrides", action="extend", nargs="+", default=[],
                        metavar="KEY=VALUE", help="config overrides, as in train.py")
    parser.add_argument("--epochs", type=int, default=2,
                        help="epoch mode: epochs to time; the first warms the file cache")
    parser.add_argument("--warmup-steps", type=int, default=10, help="torch mode")
    parser.add_argument("--steps", type=int, default=50, help="torch mode: steps to record")
    parser.add_argument("--tag", default="", help="suffix for the report file name")
    return parser.parse_args()


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def header(config: dict, mode: str) -> str:
    sha, dirty = train.git_state()
    return "\n".join([
        f"# profile_epoch.py --mode {mode}",
        f"# date: {datetime.now():%Y-%m-%d %H:%M}",
        f"# git: {sha}{' (uncommitted changes)' if dirty else ''}",
        f"# cpu: {cpu_name()} ({torch.get_num_threads()} torch threads, "
        f"{config['num_workers']} DataLoader workers)",
        f"# torch {torch.__version__}, batch_size {config['batch_size']}",
        "",
    ])


def write_report(name: str, tag: str, text: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}{'_' + tag if tag else ''}.txt"
    path.write_text(text)
    return path


def profile_epochs(setup: dict, epochs: int) -> str:
    lines = []
    for epoch in range(1, epochs + 1):
        m = train.train_one_epoch(**setup)
        lines.append(
            f"epoch {epoch} | {m['seconds']:.1f}s | data {m['data_seconds']:.1f}s "
            f"| compute {m['compute_seconds']:.1f}s"
        )
        print(lines[-1], flush=True)
    return "\n".join(lines) + "\n"


def profile_torch(setup: dict, warmup_steps: int, steps: int) -> str:
    model, loader = setup["model"], setup["loader"]
    optimizer, criterion = setup["optimizer"], setup["criterion"]
    model.train()
    batches = iter(loader)
    # The profiler drops its results once the schedule moves on, so grab them
    # when the recording window closes.
    traces = []
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=schedule(wait=0, warmup=warmup_steps, active=steps, repeat=1),
        on_trace_ready=traces.append,
    ) as prof:
        for _ in range(warmup_steps + steps):
            with record_function("data"):
                features, labels = next(batches)
            with record_function("forward"):
                loss = criterion(model(features), labels)
            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
            prof.step()

    averages = traces[0].key_averages()
    return "\n".join([
        f"## {steps} steps after {warmup_steps} warmup steps",
        "",
        "## By total CPU time (the data/forward/backward/optimizer rows are the step phases)",
        averages.table(sort_by="cpu_time_total", row_limit=25),
        "## By self CPU time (which operators the time is actually spent in)",
        averages.table(sort_by="self_cpu_time_total", row_limit=25),
    ])


def profile_cprofile(setup: dict) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    m = train.train_one_epoch(**setup)
    profiler.disable()

    stream = io.StringIO()
    stream.write(
        f"## one epoch: {m['seconds']:.1f}s | data {m['data_seconds']:.1f}s "
        f"| compute {m['compute_seconds']:.1f}s (cProfile overhead included)\n\n"
    )
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("cumulative").print_stats(40)
    return stream.getvalue()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    config["data_root"] = str(train.resolve_data_root(config["data_root"]))

    set_seed(config["seed"])
    set_num_threads(config["num_threads"])
    train_loader, _, _, label_map = train.build_loaders(config)
    model = train.build_model(config, len(label_map))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    setup = {
        "model": model,
        "loader": train_loader,
        "optimizer": optimizer,
        "scheduler": train.make_scheduler(optimizer, config, len(train_loader)),
        "criterion": nn.CrossEntropyLoss(),
    }

    start = time.perf_counter()
    if args.mode == "epoch":
        body = profile_epochs(setup, args.epochs)
    elif args.mode == "torch":
        body = profile_torch(setup, args.warmup_steps, args.steps)
    else:
        body = profile_cprofile(setup)
    print(f"done in {time.perf_counter() - start:.1f}s")

    if args.mode != "epoch":
        path = write_report(f"{args.mode}_profile", args.tag, header(config, args.mode) + body)
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
