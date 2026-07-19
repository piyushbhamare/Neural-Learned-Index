#!/usr/bin/env python3
"""
download_data.py  —  Download SOSD benchmark datasets for NLI

Downloads the three uint64 binary datasets from the SOSD Harvard Dataverse
and places them in sosd_data/.

Usage:
    python3 download_data.py              # download all three datasets
    python3 download_data.py --check      # check which files are already present
    python3 download_data.py --dataset books   # download one dataset only

Each file is ~1.5 GB. Estimated total download: ~4.5 GB.
Existing files are skipped (re-run is safe).
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry
# Source: SOSD Harvard Dataverse
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JGVF9A
# ---------------------------------------------------------------------------
DATASETS = {
    "books": {
        "filename": "books_200M_uint64",
        "url": "https://dataverse.harvard.edu/api/access/datafile/4934329",
        "size_gb": 1.49,
        "description": "Book price keys — smooth, nearly sorted",
    },
    "facebook": {
        "filename": "fb_200M_uint64",
        "url": "https://dataverse.harvard.edu/api/access/datafile/4934328",
        "size_gb": 1.49,
        "description": "Facebook user IDs — clustered, non-uniform",
    },
    "wiki": {
        "filename": "wiki_ts_200M_uint64",
        "url": "https://dataverse.harvard.edu/api/access/datafile/4934330",
        "size_gb": 1.49,
        "description": "Wikipedia edit timestamps — quasi-periodic",
    },
}

OUT_DIR = Path(__file__).parent / "sosd_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {human(downloaded)} / {human(total_size)}",
              end="", flush=True)
    else:
        print(f"\r  Downloaded {human(downloaded)}", end="", flush=True)


def download(name: str, info: dict) -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / info["filename"]

    if dest.exists():
        size = dest.stat().st_size
        print(f"  ✓  {info['filename']}  already present  ({human(size)})")
        return True

    print(f"\n  Downloading  {info['filename']}  (~{info['size_gb']:.1f} GB)")
    print(f"  {info['description']}")
    print(f"  URL: {info['url']}")

    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(info["url"], tmp, reporthook=progress_hook)
        print()  # newline after progress bar
        tmp.rename(dest)
        print(f"  ✓  Saved  →  {dest}  ({human(dest.stat().st_size)})")
        return True
    except Exception as exc:
        print(f"\n  ✗  Download failed: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


# ---------------------------------------------------------------------------
# Manual-download fallback instructions
# ---------------------------------------------------------------------------
MANUAL_INSTRUCTIONS = """
If the automatic download fails (e.g. the Dataverse URL changed), download
the files manually from the SOSD project page and place them in sosd_data/:

  SOSD GitHub : https://github.com/learnedsystems/SOSD
  Harvard Dataverse : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JGVF9A

Files needed (uint64 binary format, ~1.5 GB each):
  sosd_data/books_200M_uint64
  sosd_data/fb_200M_uint64
  sosd_data/wiki_ts_200M_uint64

Verify after download:
  python3 scripts/validate_datasets.py sosd_data
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download SOSD datasets for NLI")
    parser.add_argument("--check", action="store_true",
                        help="Only check which files are present, don't download")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Download a single dataset instead of all three")
    args = parser.parse_args()

    print("\nNLI — SOSD Dataset Downloader")
    print("=" * 50)

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    if args.check:
        print(f"\nChecking  {OUT_DIR}/\n")
        all_ok = True
        for name, info in targets.items():
            dest = OUT_DIR / info["filename"]
            if dest.exists():
                print(f"  ✓  {info['filename']}  ({human(dest.stat().st_size)})")
            else:
                print(f"  ✗  {info['filename']}  MISSING")
                all_ok = False
        if all_ok:
            print("\nAll datasets present. Ready to run benchmarks.")
        else:
            print("\nRun  python3 download_data.py  to fetch missing files.")
        return

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Datasets to download: {', '.join(targets)}\n")

    ok = []
    failed = []
    for name, info in targets.items():
        if download(name, info):
            ok.append(name)
        else:
            failed.append(name)

    print("\n" + "=" * 50)
    if failed:
        print(f"  {len(ok)} succeeded,  {len(failed)} failed: {', '.join(failed)}")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)
    else:
        print(f"  All {len(ok)} dataset(s) ready.")
        print("\nNext step:")
        print("  python3 scripts/validate_datasets.py sosd_data")
        print("  ./run_all.sh sosd_data results 100000 1000000")


if __name__ == "__main__":
    main()
