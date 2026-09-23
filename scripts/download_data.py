"""Download Google Speech Commands v0.02 into data/speech_commands_v0.02/.

Source: https://arxiv.org/abs/1804.03209 (Warden, 2018), CC BY 4.0.
Skips the download if the dataset is already there. Standard library only.

    uv run python scripts/download_data.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "speech_commands_v0.02"

REQUIRED = [
    "validation_list.txt",
    "testing_list.txt",
    "_background_noise_",
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
]


def is_complete(dest: Path) -> bool:
    return all((dest / name).exists() for name in REQUIRED)


def download(url: str, archive: Path) -> None:
    partial = archive.with_name(archive.name + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        last_pct = -1
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct != last_pct:
                    print(f"\r  {pct:3d}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
                    last_pct = pct
    print()
    partial.rename(archive)


def extract(archive: Path, dest: Path) -> None:
    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging, filter="data")
    if not is_complete(staging):
        shutil.rmtree(staging)
        raise RuntimeError(
            f"{archive} does not look like Speech Commands v0.02; delete it and re-run"
        )
    staging.rename(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"default: {DEFAULT_DEST}")
    parser.add_argument("--url", default=URL, help=argparse.SUPPRESS)
    parser.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz after extracting")
    args = parser.parse_args(argv)

    dest: Path = args.dest.expanduser().resolve()
    if is_complete(dest):
        print(f"Speech Commands already present at {dest}")
        return 0
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} exists but is incomplete; delete it and re-run.", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "speech_commands_v0.02.tar.gz"
    if archive.exists():
        print(f"Reusing downloaded archive {archive}")
    else:
        print(f"Downloading {args.url} (~2.4 GB)")
        download(args.url, archive)

    print(f"Extracting to {dest}")
    if dest.exists():
        dest.rmdir()
    extract(archive, dest)

    if not args.keep_archive:
        archive.unlink()
    print(f"Done: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
