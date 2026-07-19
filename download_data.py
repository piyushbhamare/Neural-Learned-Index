#!/usr/bin/env python3
"""
download_data.py  —  SOSD Dataset Helper for NLI Benchmark

Two modes:
  --sosd     Clone the official SOSD repository and run its own download
             script, which fetches the real benchmark datasets.
  --synth    Generate small synthetic datasets (no internet needed) for
             quick local testing of the benchmark pipeline.

Usage:
    python3 download_data.py --sosd          # real SOSD data (~4.5 GB)
    python3 download_data.py --synth         # synthetic data (instant, ~100 MB)
    python3 download_data.py --synth --keys 1000000   # choose key count
    python3 download_data.py --check         # check what is present
"""

import argparse
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).parent.resolve()
DATA_DIR = HERE / "sosd_data"

GREEN = "\033[92m"
YEL   = "\033[93m"
RED   = "\033[91m"
RST   = "\033[0m"

REAL_FILES = [
    "books_200M_uint64",
    "fb_200M_uint64",
    "wiki_ts_200M_uint64",
]

SOSD_REPO = "https://github.com/learnedsystems/SOSD.git"


# ─────────────────────────────────────────────────────────────────────────────
# Check
# ─────────────────────────────────────────────────────────────────────────────
def cmd_check():
    print(f"\nChecking  {DATA_DIR}/\n")
    DATA_DIR.mkdir(exist_ok=True)
    all_ok = True
    for name in REAL_FILES:
        p = DATA_DIR / name
        if p.exists():
            mb = p.stat().st_size // (1024 * 1024)
            print(f"  {GREEN}✓{RST}  {name}  ({mb} MB)")
        else:
            print(f"  {RED}✗{RST}  {name}  MISSING")
            all_ok = False

    synth_files = list(DATA_DIR.glob("synth_*.bin"))
    if synth_files:
        print()
        for p in sorted(synth_files):
            mb = p.stat().st_size // (1024 * 1024)
            print(f"  {YEL}~{RST}  {p.name}  (synthetic, {mb} MB)")

    print()
    if all_ok:
        print(f"  {GREEN}All real datasets present.{RST} Ready to run benchmarks.")
    else:
        print(f"  Run  python3 download_data.py --sosd   for real data (~4.5 GB)")
        print(f"  Run  python3 download_data.py --synth  for quick synthetic data")


# ─────────────────────────────────────────────────────────────────────────────
# Real SOSD data via official repo
# ─────────────────────────────────────────────────────────────────────────────
def cmd_sosd():
    print(f"\n{GREEN}Fetching real SOSD datasets via official repository{RST}")
    print(f"  Source: {SOSD_REPO}\n")

    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        print(f"{RED}[ERROR]{RST} git is not installed or not in PATH.")
        _print_manual()
        sys.exit(1)

    sosd_clone = HERE / "_sosd_repo"

    if sosd_clone.exists():
        print("  SOSD repo already cloned — pulling latest ...")
        subprocess.run(["git", "-C", str(sosd_clone), "pull"], check=True)
    else:
        print("  Cloning SOSD repo (shallow) ...")
        subprocess.run([
            "git", "clone", "--depth", "1", SOSD_REPO, str(sosd_clone)
        ], check=True)

    # SOSD ships its own download script
    dl_script = sosd_clone / "scripts" / "download.sh"
    if not dl_script.exists():
        print(f"{RED}[ERROR]{RST} Expected {dl_script} not found.")
        print("  The SOSD repo structure may have changed.")
        _print_manual()
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n  Running SOSD download script → output to {DATA_DIR}/\n")
    env = {**os.environ, "DATA_DIR": str(DATA_DIR)}
    result = subprocess.run(["bash", str(dl_script)], cwd=str(sosd_clone), env=env)

    if result.returncode != 0:
        print(f"\n{RED}[ERROR]{RST} SOSD download script failed.")
        _print_manual()
        sys.exit(1)

    print(f"\n{GREEN}Done.{RST} Verify with:  python3 scripts/validate_datasets.py sosd_data")


def _print_manual():
    print("""
Manual download instructions
─────────────────────────────────────────────────────────────────────────────
1. Visit the SOSD GitHub repository:
     https://github.com/learnedsystems/SOSD

2. Follow the "Data" section in their README to download:
     books_200M_uint64
     fb_200M_uint64
     wiki_ts_200M_uint64

3. Place all three files in:
     sosd_data/

4. Verify:
     python3 scripts/validate_datasets.py sosd_data
─────────────────────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (no internet, instant)
# ─────────────────────────────────────────────────────────────────────────────
SYNTH_DATASETS = {
    "books": {
        "filename":    "synth_books_uint64",
        "description": "Smooth, nearly sorted (models book-price distribution)",
        "generator":   "books",
    },
    "facebook": {
        "filename":    "synth_facebook_uint64",
        "description": "Clustered, non-uniform (models Facebook UID distribution)",
        "generator":   "facebook",
    },
    "wiki": {
        "filename":    "synth_wiki_uint64",
        "description": "Quasi-periodic with drift (models Wikipedia timestamp dist.)",
        "generator":   "wiki",
    },
}


def _gen_keys_books(n: int) -> list:
    """Smooth monotone distribution — polynomial + small noise."""
    rng = random.Random(42)
    mx = (1 << 63) - 1
    raw = sorted(int(mx * (i / n) ** 1.3 + rng.randint(0, mx // (n * 10)))
                 for i in range(n))
    # Deduplicate (sorted, so adjacent check is enough)
    keys = [raw[0]]
    for k in raw[1:]:
        if k > keys[-1]:
            keys.append(k)
        else:
            keys.append(keys[-1] + 1)
    return keys[:n]


def _gen_keys_facebook(n: int) -> list:
    """Clustered distribution — several dense bands."""
    rng = random.Random(43)
    n_clusters = 20
    cluster_size = n // n_clusters
    cluster_centers = sorted(rng.randint(0, (1 << 63) - 1) for _ in range(n_clusters))
    span = (1 << 63) // (n_clusters * 4)
    keys = []
    for c in cluster_centers:
        for _ in range(cluster_size):
            keys.append(max(0, c + rng.randint(-span, span)))
    keys += [rng.randint(0, (1 << 63) - 1) for _ in range(n - len(keys))]
    keys.sort()
    # Deduplicate
    out = [keys[0]]
    for k in keys[1:]:
        out.append(max(k, out[-1] + 1))
    return out[:n]


def _gen_keys_wiki(n: int) -> list:
    """Quasi-periodic with slow drift — simulates timestamp data."""
    rng = random.Random(44)
    period = (1 << 63) // 10
    keys = []
    t = 0
    for i in range(n):
        step = period // n + rng.randint(0, period // (n * 2))
        drift = int(period * 0.1 * i / n)
        t += step + drift
        keys.append(t % ((1 << 63) - 1))
    keys.sort()
    out = [keys[0]]
    for k in keys[1:]:
        out.append(max(k, out[-1] + 1))
    return out[:n]


_GENERATORS = {
    "books":    _gen_keys_books,
    "facebook": _gen_keys_facebook,
    "wiki":     _gen_keys_wiki,
}


def _write_sosd_binary(path: Path, keys: list):
    """Write keys in SOSD binary format: uint64 count then N uint64 keys."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(keys)))
        f.write(struct.pack(f"<{len(keys)}Q", *keys))


def cmd_synth(n_keys: int):
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n{GREEN}Generating synthetic datasets{RST}  ({n_keys:,} keys each)\n")

    for name, info in SYNTH_DATASETS.items():
        dest = DATA_DIR / info["filename"]
        print(f"  Generating  {info['filename']} ...", end="", flush=True)
        keys = _GENERATORS[info["generator"]](n_keys)
        _write_sosd_binary(dest, keys)
        mb = dest.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RST}  {mb:.1f} MB  — {info['description']}")

    print(f"""
{GREEN}Done.{RST}

Synthetic files use the same SOSD binary format as the real datasets.
The benchmark accepts them via the same path:

  ./run_all.sh sosd_data results 100000
  ./build/nli_benchmark sosd_data results 100000

Note: the filenames differ from the real datasets, so you will need to
pass the synthetic directory OR rename them to the expected names:

  cd sosd_data
  cp synth_books_uint64    books_200M_uint64
  cp synth_facebook_uint64 fb_200M_uint64
  cp synth_wiki_uint64     wiki_ts_200M_uint64

For full publication results use the real SOSD data:
  python3 download_data.py --sosd
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOSD dataset helper for NLI benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sosd",  action="store_true",
                       help="Download real SOSD data via official GitHub repo")
    group.add_argument("--synth", action="store_true",
                       help="Generate synthetic datasets locally (no internet)")
    group.add_argument("--check", action="store_true",
                       help="Check which dataset files are present")
    parser.add_argument("--keys", type=int, default=200_000,
                        help="Key count for synthetic datasets (default: 200000)")
    args = parser.parse_args()

    print("NLI — SOSD Dataset Helper")
    print("=" * 50)

    if args.check:
        cmd_check()
    elif args.sosd:
        cmd_sosd()
    elif args.synth:
        cmd_synth(args.keys)
    else:
        # No flag: show status and help
        cmd_check()
        print("\nOptions:")
        print("  python3 download_data.py --sosd         Real data via SOSD GitHub (~4.5 GB)")
        print("  python3 download_data.py --synth        Synthetic data, no internet (~100 MB)")
        print("  python3 download_data.py --synth --keys 1000000   Larger synthetic set")


if __name__ == "__main__":
    main()
