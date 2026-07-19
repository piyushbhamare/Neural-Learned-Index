#!/usr/bin/env python3
"""
nli_master.py  –  Neural Enhanced Learned Index (NLI) Master Script
====================================================================
One script to:
  1. Download official ALEX + PGM headers from GitHub
  2. Compile sosd_benchmark_final.cpp  (with real or faithful impls)
  3. Interactive menu  →  run the benchmark at chosen scale
  4. Generate all publication-quality figures  (300 DPI, DejaVu Serif)
  5. Print a results summary

Usage:
    python nli_master.py            # full interactive mode
    python nli_master.py --quick    # 100K only, skip figures
    python nli_master.py --figs     # regenerate figures from existing CSVs

Authors: Ayush Sharma, Piyush Bhamare, Arya Arankalle, Kishor Wagh
Paper:   "Comparative Analysis of NLI System" – CVMI / IWIN 2026
"""

import os, sys, subprocess, shutil, platform, time, urllib.request, urllib.error
from pathlib import Path

HERE   = Path(__file__).parent.resolve()
TP     = HERE / "third_party"
RES    = HERE / "results"
FIGS   = HERE / "figures"
SRC    = HERE / "sosd_benchmark_final.cpp"
BIN    = HERE / ("nli_benchmark.exe" if platform.system() == "Windows" else "nli_benchmark")

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; CYN = "\033[96m"; RST = "\033[0m"; BLD = "\033[1m"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 – Dependency check
# ─────────────────────────────────────────────────────────────────────────────
def phase_deps():
    print(f"\n{CYN}[Phase 1] Checking Python dependencies...{RST}")
    missing = []
    for pkg in ["numpy", "pandas", "matplotlib", "scipy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"  Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
    print(f"  {GREEN}✓ numpy  pandas  matplotlib  scipy{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Download official ALEX + PGM headers
# ─────────────────────────────────────────────────────────────────────────────
HEADER_FILES = {
    TP / "pgm" / "pgm_index.hpp": (
        "https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/pgm_index.hpp",
        "namespace pgm",
    ),
    TP / "pgm" / "piecewise_linear_model.hpp": (
        "https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/piecewise_linear_model.hpp",
        "namespace pgm",
    ),
    TP / "alex" / "alex.h": (
        "https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex.h",
        "namespace alex",
    ),
    TP / "alex" / "alex_base.h": (
        "https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_base.h",
        "namespace alex",
    ),
    TP / "alex" / "alex_nodes.h": (
        "https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_nodes.h",
        "namespace alex",
    ),
    TP / "alex" / "alex_fanout_tree.h": (
        "https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_fanout_tree.h",
        "namespace alex",
    ),
}

def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NLI-Benchmark/2.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            data = r.read(); f.write(data)
        print(f"    {GREEN}✓{RST}  {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"    {YEL}!{RST}  {dest.name}: {e}")
        return False

def phase_headers():
    print(f"\n{CYN}[Phase 2] Verifying official ALEX + PGM-Index headers...{RST}")
    errors = []

    # ── PGM-Index (needs pgm_index.hpp + piecewise_linear_model.hpp) ──────────
    print("  PGM-Index  (Ferragina & Vinciguerra, PVLDB 2020)")
    pgm_files = [
        (TP / "pgm" / "pgm_index.hpp",             "https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/pgm_index.hpp"),
        (TP / "pgm" / "piecewise_linear_model.hpp", "https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/piecewise_linear_model.hpp"),
    ]
    pgm_ok = True
    for pgm_dest, pgm_url in pgm_files:
        if pgm_dest.exists() and pgm_dest.stat().st_size > 1000:
            print(f"    {GREEN}✓  {pgm_dest.name}  ({pgm_dest.stat().st_size//1024} KB){RST}")
        else:
            print(f"    Attempting download: {pgm_dest.name}...")
            ok = _download(pgm_url, pgm_dest)
            if not ok or not pgm_dest.exists() or pgm_dest.stat().st_size < 1000:
                pgm_ok = False
    if not pgm_ok:
        errors.append(
            f"\n  {RED}MISSING: third_party/pgm/  (one or more headers){RST}\n"
            f"  Need BOTH files:\n"
            f"    https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/pgm_index.hpp\n"
            f"    https://raw.githubusercontent.com/gvinciguerra/PGM-index/master/include/pgm/piecewise_linear_model.hpp\n"
            f"  Save both to:  third_party\\pgm\\\n"
            f"  Quickest fix (run in cmd inside learned_index_complete):\n"
            f"    git clone --depth=1 https://github.com/gvinciguerra/PGM-index pgm_tmp\n"
            f"    mkdir third_party\\pgm\n"
            f"    copy pgm_tmp\\include\\pgm\\pgm_index.hpp third_party\\pgm\\\n"
            f"    copy pgm_tmp\\include\\pgm\\piecewise_linear_model.hpp third_party\\pgm\\\n"
            f"    rmdir /s /q pgm_tmp"
        )

    # ── ALEX ──────────────────────────────────────────────────────────────────
    print("  ALEX  (Ding et al., SIGMOD 2020)")
    alex_files = [TP/"alex"/"alex.h", TP/"alex"/"alex_base.h",
                  TP/"alex"/"alex_nodes.h", TP/"alex"/"alex_fanout_tree.h"]
    alex_ok = True
    for dest in alex_files:
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"    {GREEN}✓  {dest.name}  ({dest.stat().st_size//1024} KB){RST}")
        else:
            url = HEADER_FILES[dest][0]
            print(f"    Attempting download: {dest.name}...")
            ok = _download(url, dest)
            if not ok or not dest.exists() or dest.stat().st_size < 1000:
                alex_ok = False

    if not alex_ok or "namespace alex" not in (TP/"alex"/"alex.h").read_text(errors="replace"):
        alex_ok = False
        errors.append(
            f"\n  {RED}MISSING: third_party/alex/  (one or more headers){RST}\n"
            f"  Download these 4 files from:\n"
            f"    https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex.h\n"
            f"    https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_base.h\n"
            f"    https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_nodes.h\n"
            f"    https://raw.githubusercontent.com/microsoft/ALEX/main/src/core/alex_fanout_tree.h\n"
            f"  Save all 4 to:\n"
            f"    {TP / 'alex'}\\\n"
            f"  Or run:\n"
            f"    git clone --depth=1 https://github.com/microsoft/ALEX alex_tmp\n"
            f"    mkdir third_party\\alex\n"
            f"    copy alex_tmp\\src\\core\\alex*.h third_party\\alex\\\n"
            f"    rmdir /s /q alex_tmp"
        )

    print(f"  B-Tree   : {GREEN}std::map (C++ stdlib){RST}")
    print(f"  RMI      : {GREEN}faithful C++ impl embedded{RST}")

    if errors:
        print(f"\n{RED}{'='*65}{RST}")
        print(f"{RED}{BLD}ERROR: Official headers not found. Cannot compile with real implementations.{RST}")
        print(f"{RED}Follow these steps to install them:{RST}")
        for e in errors:
            print(e)
        print(f"\n{RED}{'='*65}{RST}")
        print(f"\nAfter placing the files, re-run:  python nli_master.py")
        sys.exit(1)

    return pgm_ok, alex_ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – Compile
# ─────────────────────────────────────────────────────────────────────────────
def _find_compiler():
    for c in ["g++", "g++.exe", "c++", "clang++"]:
        if shutil.which(c): return c
    return None

def phase_compile(pgm_ok: bool, alex_ok: bool) -> bool:
    print(f"\n{CYN}[Phase 3] Compiling sosd_benchmark_final.cpp...{RST}")
    cc = _find_compiler()
    if not cc:
        print(f"  {RED}ERROR: No C++ compiler found (install g++ or MinGW).{RST}")
        _print_manual_compile(pgm_ok, alex_ok)
        return False

    flags = [cc, "-O3", "-std=c++17", "-march=native", "-mpopcnt",
             "-DALEX_USE_LZCNT=0",  # disable lzcnt/tzcnt — avoids 0xC0000005 on MinGW
             f"-I{TP}",
             f"-I{TP / 'pgm'}",    # lets piecewise_linear_model.hpp resolve
             f"-I{TP / 'alex'}",   # lets alex_base.h / alex_nodes.h resolve
             "-o", str(BIN), str(SRC)]
    if pgm_ok:  flags.append("-DUSE_REAL_PGM")
    if alex_ok: flags.append("-DUSE_REAL_ALEX")  # always use official ALEX

    use_real_alex = "-DUSE_REAL_ALEX" in flags
    print(f"  Compiler : {cc}")
    print(f"  Flags    : -O3 -std=c++17 -march=native -mpopcnt -DALEX_USE_LZCNT=0 {'-DUSE_REAL_PGM' if pgm_ok else ''} {'-DUSE_REAL_ALEX' if use_real_alex else ''}")
    t0 = time.time()
    r = subprocess.run(flags, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode == 0:
        print(f"  {GREEN}✓  Compiled → {BIN.name}  ({elapsed:.1f}s){RST}")
        impl_pgm  = "REAL (official github.com/gvinciguerra/PGM-index)" if pgm_ok else "faithful reference"
        impl_alex = "REAL (official github.com/microsoft/ALEX)" if use_real_alex else "faithful reference"
        print(f"  PGM  : {impl_pgm}")
        print(f"  ALEX : {impl_alex}")
        return True
    print(f"  {RED}Compile failed:{RST}")
    print(r.stderr)          # print full error — do not truncate
    _print_manual_compile(pgm_ok, alex_ok)
    return False

def _print_manual_compile(pgm_ok, alex_ok):
    cc = "g++"
    out = "nli_benchmark.exe" if platform.system()=="Windows" else "nli_benchmark"
    flags = [cc, "-O3", "-std=c++17", "-march=native",
             "-Ithird_party", "-Ithird_party/pgm", "-Ithird_party/alex",
             "-o", out, "sosd_benchmark_final.cpp"]
    if pgm_ok:  flags.append("-DUSE_REAL_PGM")
    if alex_ok: flags.append("-DUSE_REAL_ALEX")
    print(f"\n  Manual compile:\n    {' '.join(flags)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 – Interactive menu + benchmark run
# ─────────────────────────────────────────────────────────────────────────────
SCALE_MAP = {
    "1": (100_000,        "100K keys  (~2–5 min)"),
    "2": (1_000_000,      "1M keys    (~10–20 min)"),
    "3": (10_000_000,     "10M keys   (~45–90 min)"),
    "4": (50_000_000,     "50M keys   (~2–4 hours)"),
    "5": (200_000_000,    "200M keys  (~6–12 hours, needs 32 GB RAM)"),
    "6": (1_000_000,      "100K + 1M  (quick + medium, ~12–25 min)"),
    "7": (200_000_000,    "ALL scales (100K → 1M → 10M → 50M → 200M)"),
}

def phase_menu() -> int:
    """Returns the NLI_MAX_SCALE value to set (SIZE_MAX-1 = all scales)."""
    print(f"\n{BLD}{'='*60}{RST}")
    print(f"{BLD}  NLI Benchmark  –  Select scale{RST}")
    print(f"{'='*60}")
    for k, (sz, desc) in SCALE_MAP.items():
        print(f"  {CYN}[{k}]{RST}  {desc}")
    print(f"{'='*60}")
    while True:
        choice = input("  Enter choice [1-7] (default 1): ").strip() or "1"
        if choice in SCALE_MAP:
            sz, desc = SCALE_MAP[choice]
            print(f"  → Running: {desc}")
            return sz
        print(f"  {YEL}Invalid choice.{RST}")

def phase_run(max_scale: int):
    print(f"\n{CYN}[Phase 4] Running benchmark (max scale = {max_scale:,})...{RST}")
    RES.mkdir(exist_ok=True)

    if not BIN.exists():
        print(f"  {RED}ERROR: {BIN.name} not found. Compile first (Phase 3).{RST}")
        return False

    env = os.environ.copy()
    if max_scale < 200_000_000:
        env["NLI_MAX_SCALE"] = str(max_scale)
    # else: run all (no cap)

    t0 = time.time()
    proc = subprocess.Popen([str(BIN)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            cwd=str(HERE))
    for line in proc.stdout:
        print("  " + line, end="")
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\n  {GREEN}✓  Benchmark done  ({elapsed/60:.1f} min){RST}")
        return True
    print(f"\n  {RED}Benchmark exited with code {proc.returncode}{RST}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 – Generate publication-quality figures
# ─────────────────────────────────────────────────────────────────────────────
def phase_figures():
    print(f"\n{CYN}[Phase 5] Generating publication-quality figures...{RST}")
    FIGS.mkdir(exist_ok=True)

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Style
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    ALGOS   = ["B-Tree", "ALEX", "PGM", "RMI", "NLI"]
    COLORS  = {"B-Tree": "#1f77b4", "ALEX": "#ff7f0e", "PGM": "#2ca02c",
                "RMI": "#d62728",   "NLI": "#9467bd"}
    HATCHES = {"B-Tree": "", "ALEX": "//", "PGM": "..", "RMI": "xx", "NLI": ""}
    DATASETS = ["Books", "Facebook", "WikiTS"]

    def savefig(fig, name):
        path = FIGS / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {GREEN}✓{RST}  {name}")
        return path

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def load(name):
        p = RES / name
        if not p.exists():
            print(f"  {YEL}SKIP {name} (not found){RST}")
            return None
        try:
            return pd.read_csv(p, on_bad_lines='skip')
        except Exception:
            try:
                return pd.read_csv(p, error_bad_lines=False)  # pandas < 1.3 compat
            except Exception:
                return None

    bench     = load("benchmark_results.csv")
    drift     = load("drift_results.csv")
    ens_abl   = load("drift_ensemble_ablation.csv")
    ablation  = load("ablation_results.csv")
    train_log = load("training_log.csv")
    overhead  = load("drift_overhead_results.csv")
    scalab    = load("scalability_results.csv")

    figures_made = []

    # ── Fig 1 – Latency comparison (100K + 1M, grouped bar) ──────────────────
    if bench is not None:
        sub = bench[bench["Keys"].isin([100_000, 1_000_000])].copy()
        sub["Algorithm"] = pd.Categorical(sub["Algorithm"], categories=ALGOS, ordered=True)
        sub = sub.sort_values(["Keys","Algorithm"])

        scales = [100_000, 1_000_000]
        scale_labels = ["100K keys", "1M keys"]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
        fig.subplots_adjust(bottom=0.28, wspace=0.35)

        for ax, sc, sl in zip(axes, scales, scale_labels):
            d = sub[sub["Keys"]==sc]
            if d.empty: continue
            x = np.arange(len(ALGOS)); w = 0.35
            for i, (col, ylab) in enumerate([("Read_ns","Read Latency (ns)"),
                                              ("Insert_ns","Insert Latency (ns)")]):
                bars = []
                for j, alg in enumerate(ALGOS):
                    row = d[d["Algorithm"]==alg]
                    val = float(row[col].iloc[0]) if not row.empty else 0
                    b = ax.bar(x[j] + (i-0.5)*w, val, w, color=COLORS[alg],
                               hatch=HATCHES[alg], alpha=0.85, label=alg if i==0 else "")
                    bars.append(b)
            ax.set_xticks(x); ax.set_xticklabels(ALGOS, rotation=15, ha="right")
            ax.set_title(sl); ax.set_ylabel(ylab if i==0 else "")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        handles = [plt.Rectangle((0,0),1,1, color=COLORS[a], hatch=HATCHES[a], alpha=0.85)
                   for a in ALGOS]
        fig.legend(handles, ALGOS, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False)
        fig.suptitle("Read and Insert Latency Comparison (all baselines locally benchmarked)", y=0.98)
        figures_made.append(savefig(fig, "fig1_latency_comparison.png"))

    # ── Fig 2 – Speedup heatmap (NLI vs each baseline × dataset × scale) ─────
    if bench is not None:
        nli_data = bench[bench["Algorithm"]=="NLI"].set_index(["Dataset","Keys"])["Read_ns"]
        scales_show = sorted(bench["Keys"].unique())[:4]  # up to 4 scales

        fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, wspace=0.05)
        baselines = ["B-Tree", "ALEX", "PGM", "RMI"]
        CAP = 5.0

        for ax_idx, bl in enumerate(baselines):
            ax = axes[ax_idx]
            bl_data = bench[bench["Algorithm"]==bl].set_index(["Dataset","Keys"])["Read_ns"]
            matrix = np.zeros((len(DATASETS), len(scales_show)))
            annot  = []
            for ri, ds in enumerate(DATASETS):
                row_ann = []
                for ci, sc in enumerate(scales_show):
                    try:
                        sp = bl_data.loc[(ds, sc)] / max(nli_data.loc[(ds, sc)], 1e-9)
                    except KeyError:
                        sp = np.nan
                    matrix[ri, ci] = min(sp, CAP) if not np.isnan(sp) else 0
                    row_ann.append(f"{sp:.1f}×" if not np.isnan(sp) else "–")
                annot.append(row_ann)

            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                           vmin=0.5, vmax=CAP)
            ax.set_xticks(range(len(scales_show)))
            ax.set_xticklabels([f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M"
                                for s in scales_show], fontsize=7, rotation=45)
            ax.set_yticks(range(len(DATASETS)))
            ax.set_yticklabels(DATASETS if ax_idx == 0 else [], fontsize=7)
            ax.set_title(f"vs {bl}", fontsize=8)
            for ri in range(len(DATASETS)):
                for ci in range(len(scales_show)):
                    ax.text(ci, ri, annot[ri][ci], ha="center", va="center",
                            fontsize=6.5, color="black")

        fig.suptitle("NLI Speedup over baselines (colour capped at 5×, actual values annotated)", y=1.01)
        plt.colorbar(im, ax=axes[-1], shrink=0.8, label="Speedup ×")
        figures_made.append(savefig(fig, "fig2_speedup_heatmap.png"))

    # ── Fig 3 – Drift detection: F1 / Precision / Recall vs window size ───────
    if drift is not None:
        ensemble = drift[drift["detector"]=="Ensemble"]
        drift_types = ["gradual","sudden"]
        windows = sorted(ensemble["window_size"].unique())

        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
        fig.subplots_adjust(bottom=0.28, wspace=0.38)
        metrics = [("f1","F1-Score"), ("precision","Precision"), ("recall","Recall")]

        for ax, (metric, ylabel) in zip(axes, metrics):
            for dt, ls in zip(drift_types, ["-o","--s"]):
                d = ensemble[ensemble["drift_type"]==dt].sort_values("window_size")
                if d.empty: continue
                vals = d.groupby("window_size")[metric].mean()
                ax.plot(range(len(vals)), vals.values, ls, label=dt.capitalize(),
                        linewidth=1.5, markersize=4)
            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels([f"{w//1000}K" for w in windows], fontsize=7)
            ax.set_xlabel("Drift Window Size")
            ax.set_ylabel(ylabel); ax.set_ylim(-0.05, 1.05)
            ax.set_title(ylabel)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
        fig.suptitle("Drift Detection Performance – Ensemble (4-signal majority voting ≥2/4)", y=0.98)
        figures_made.append(savefig(fig, "fig3_drift_detection.png"))

    # ── Fig 4 – Ensemble ablation: F1 per detector at window=100K ─────────────
    if ens_abl is not None:
        d100 = ens_abl[ens_abl["window_size"]==100_000]
        if d100.empty and len(ens_abl["window_size"].unique()):
            d100 = ens_abl.iloc[:]  # fallback: all

        det_order = ["EWMA","PSI","KS","AE","Ensemble"]
        drift_types = sorted(d100["drift_type"].unique())
        x = np.arange(len(det_order)); w = 0.8 / max(len(drift_types), 1)
        colors2 = plt.cm.tab10(np.linspace(0,0.6,len(drift_types)))

        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        fig.subplots_adjust(right=0.78, bottom=0.22)
        for i, dt in enumerate(drift_types):
            dd = d100[d100["drift_type"]==dt]
            vals = [dd[dd["detector"]==det]["f1"].mean() if not dd[dd["detector"]==det].empty
                    else 0 for det in det_order]
            ax.bar(x + i*w - w*len(drift_types)/2 + w/2, vals, w,
                   color=colors2[i], label=dt.capitalize(), alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(det_order)
        ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1.1)
        ax.set_title("Drift Detector F1-Score (window=100K)")
        ax.legend(bbox_to_anchor=(1.02, 0.6), loc="upper left", frameon=False)
        figures_made.append(savefig(fig, "fig4_ensemble_ablation.png"))

    # ── Fig 5 – NLI ablation (NLI-Linear / NLI-NoDrift / NLI-Full) ───────────
    if ablation is not None:
        variants = ["NLI-Linear","NLI-NoDrift","NLI-Full","NLI"]
        ablation["Variant"] = pd.Categorical(ablation["Variant"],
                                             categories=variants, ordered=True)
        sub = ablation[ablation["Variant"].isin(variants)]

        datasets_abl = sorted(sub["Dataset"].unique())
        scales_abl   = sorted(sub["Keys"].unique())[:2]

        fig, axes = plt.subplots(1, len(scales_abl), figsize=(7.0, 2.8), sharey=False)
        if len(scales_abl) == 1: axes = [axes]
        fig.subplots_adjust(right=0.80, bottom=0.30, wspace=0.35)
        colors3 = plt.cm.Set2(np.linspace(0, 0.8, len(variants)))

        for ax, sc in zip(axes, scales_abl):
            d = sub[sub["Keys"]==sc]
            x = np.arange(len(datasets_abl)); w = 0.8/len(variants)
            for i, var in enumerate(variants):
                vals = [d[(d["Dataset"]==ds) & (d["Variant"]==var)]["Read_ns"].mean()
                        if not d[(d["Dataset"]==ds) & (d["Variant"]==var)].empty else 0
                        for ds in datasets_abl]
                ax.bar(x + i*w - w*len(variants)/2 + w/2, vals, w,
                       color=colors3[i], label=var, alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(datasets_abl, rotation=15, ha="right")
            ax.set_ylabel("Read Latency (ns)")
            lbl = f"{sc//1000}K" if sc < 1_000_000 else f"{sc//1_000_000}M"
            ax.set_title(f"Ablation  ({lbl} keys)")

        handles3 = [plt.Rectangle((0,0),1,1, color=colors3[i]) for i in range(len(variants))]
        fig.legend(handles3, variants, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=len(variants), frameon=False)
        fig.suptitle("NLI Component Ablation Study", y=0.98)
        figures_made.append(savefig(fig, "fig5_ablation.png"))

    # ── Fig 6 – Scalability (latency vs keys, log–log) ────────────────────────
    if scalab is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
        fig.subplots_adjust(bottom=0.22, wspace=0.35)
        line_styles = {"NLI": ("purple","-o"), "BTree": ("#1f77b4","--s")}

        # Aggregate across datasets
        agg = scalab.groupby("Keys")[["NLI_ns","BTree_ns"]].mean().reset_index()
        ks  = agg["Keys"].values

        ax = axes[0]
        ax.loglog(ks, agg["NLI_ns"], "-o", color="purple", label="NLI", linewidth=1.5, markersize=4)
        ax.loglog(ks, agg["BTree_ns"], "--s", color="#1f77b4", label="B-Tree", linewidth=1.5, markersize=4)
        ax.set_xlabel("Number of Keys"); ax.set_ylabel("Mean Latency (ns)")
        ax.set_title("Scalability – Read Latency"); ax.legend(frameon=False)

        ax2 = axes[1]
        speedup = agg["BTree_ns"] / agg["NLI_ns"].replace(0, np.nan)
        ax2.semilogx(ks, speedup, "-^", color="green", linewidth=1.5, markersize=4)
        ax2.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Number of Keys"); ax2.set_ylabel("NLI Speedup over B-Tree (×)")
        ax2.set_title("NLI Speedup vs Scale")

        fig.suptitle("Scalability: 100K → 200M Keys  (avg across Books, Facebook, WikiTS)", y=0.98)
        figures_made.append(savefig(fig, "fig6_scalability.png"))

    # ── Fig 7 – Training log: convergence (loss vs epoch proxy) ──────────────
    if train_log is not None and "NLI_FinalLoss" in train_log.columns:
        sub = train_log[train_log["Dataset"]=="Books"] if "Books" in train_log["Dataset"].values \
              else train_log
        if not sub.empty:
            fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
            fig.subplots_adjust(bottom=0.25, wspace=0.4)

            # Loss vs scale
            ax = axes[0]
            sub2 = train_log.groupby("Keys")[["NLI_FinalLoss","NLI_MAE"]].mean().reset_index()
            ax.loglog(sub2["Keys"], sub2["NLI_FinalLoss"], "-o", color="purple",
                      linewidth=1.5, markersize=4, label="Final Loss")
            ax.loglog(sub2["Keys"], sub2["NLI_MAE"],       "-s", color="orange",
                      linewidth=1.5, markersize=4, label="Cal. MAE")
            ax.set_xlabel("Training Set Size"); ax.set_ylabel("Loss / MAE")
            ax.set_title("MLP Training Convergence"); ax.legend(frameon=False)

            # Bound vs scale
            ax2 = axes[1]
            sub3 = train_log.groupby("Keys")["NLI_Bound"].mean().reset_index()
            ax2.semilogx(sub3["Keys"], sub3["NLI_Bound"], "-^", color="green",
                         linewidth=1.5, markersize=4)
            ax2.set_xlabel("Training Set Size"); ax2.set_ylabel("Adaptive Search Bound")
            ax2.set_title("CAAB: Adaptive Bound vs Scale")

            fig.suptitle(
                "NLI Neural Component: Training Hyperparams  "
                f"(hidden=32, epochs={sub['NLI_Epochs'].iloc[0] if 'NLI_Epochs' in sub.columns else '?'}, "
                f"lr=5e-4, batch=SGD)", y=0.98)
            figures_made.append(savefig(fig, "fig7_training_log.png"))

    # ── Fig 8 – Drift overhead analysis ───────────────────────────────────────
    if overhead is not None and "Overhead_pct" in overhead.columns:
        # Per-detector overhead
        det_rows = overhead.dropna(subset=["Detector"]) if "Detector" in overhead.columns \
                   else pd.DataFrame()
        full_rows = overhead[~overhead.isin(det_rows)].dropna() if not det_rows.empty \
                    else overhead

        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        fig.subplots_adjust(bottom=0.28, left=0.15)

        if not det_rows.empty and "Detector" in det_rows.columns:
            det_order2 = ["EWMA","PSI","KS","AE"]
            agg_oh = det_rows.groupby("Detector")["Overhead_pct"].mean()
            bars = [agg_oh.get(d, 0) for d in det_order2]
            colors_oh = ["#e377c2","#bcbd22","#17becf","#8c564b"]
            ax.bar(det_order2, bars, color=colors_oh, alpha=0.85)
            for i, v in enumerate(bars):
                ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8)
            ax.set_ylabel("Overhead vs No-Drift NLI (%)"); ax.set_xlabel("Detector")
            ax.set_title("Per-Detector Drift Overhead (100K scale)")
        else:
            # Fallback: ensemble overhead per dataset
            agg_oh2 = overhead.groupby("Dataset")["Overhead_pct"].mean()
            ax.bar(agg_oh2.index, agg_oh2.values, color="#9467bd", alpha=0.85)
            ax.set_ylabel("Overhead (%)"); ax.set_xlabel("Dataset")
            ax.set_title("Drift Detection Overhead")

        figures_made.append(savefig(fig, "fig8_drift_overhead.png"))

    print(f"\n  {BLD}{len(figures_made)} figures saved → {FIGS}/{RST}")
    return figures_made


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 – Summary
# ─────────────────────────────────────────────────────────────────────────────
def phase_summary():
    print(f"\n{CYN}[Phase 6] Results Summary{RST}")
    import pandas as pd

    def load(name):
        p = RES / name
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            return df if not df.empty else None
        except Exception:
            return None

    bench = load("benchmark_results.csv")
    drift = load("drift_results.csv")

    if bench is not None:
        print(f"\n  {BLD}Read Latency (ns) – 100K keys:{RST}")
        sub = bench[bench["Keys"]==100_000][["Algorithm","Read_ns","Insert_ns","Build_ms","Memory_KB"]]
        if sub.empty:
            sub = bench.groupby("Algorithm")[["Read_ns","Insert_ns"]].mean().reset_index()
        for _, row in sub.iterrows():
            nli_row = bench[(bench["Algorithm"]=="NLI") & (bench["Keys"]==100_000)]
            nli_ns  = float(nli_row["Read_ns"].iloc[0]) if not nli_row.empty else 1
            spd = float(row["Read_ns"]) / max(nli_ns, 1)
            print(f"  {row['Algorithm']:8s}  read={float(row['Read_ns']):8.1f} ns  "
                  f"ins={float(row['Insert_ns']):9.1f} ns  "
                  f"NLI speedup={spd:.2f}×")

    if drift is not None:
        ens = drift[drift["detector"]=="Ensemble"]
        if not ens.empty:
            print(f"\n  {BLD}Drift Detection – Ensemble (avg F1):{RST}")
            for dt in sorted(ens["drift_type"].unique()):
                d = ens[ens["drift_type"]==dt]
                f1  = d["f1"].mean();  prec = d["precision"].mean()
                rec = d["recall"].mean(); fpr = d["fpr"].mean()
                print(f"  {dt:10s}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  FPR={fpr:.3f}")

    csv_files = list(RES.glob("*.csv"))
    fig_files = list(FIGS.glob("*.png")) if FIGS.exists() else []
    print(f"\n  CSVs   : {len(csv_files)} files in {RES}")
    print(f"  Figures: {len(fig_files)} files in {FIGS}")
    print(f"\n  {GREEN}{BLD}Done.{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# ASAN / UBSAN validation helper
# ─────────────────────────────────────────────────────────────────────────────
def run_asan():
    print(f"\n{CYN}[ASAN/UBSAN] Compiling sanitizer build...{RST}")
    cc = _find_compiler()
    if not cc:
        print(f"  {RED}No compiler found.{RST}"); return

    asan_bin = HERE / "nli_asan"
    flags = [cc, "-O1", "-std=c++17", "-fsanitize=address,undefined",
             "-fno-omit-frame-pointer", f"-I{TP}",
             "-o", str(asan_bin), str(SRC)]
    r = subprocess.run(flags, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {RED}ASAN compile failed:{RST}\n{r.stderr[:500]}"); return
    print(f"  {GREEN}✓ ASAN binary: {asan_bin.name}{RST}")

    env = os.environ.copy()
    env["NLI_MAX_SCALE"] = "100000"  # small scale only for sanitizer
    env["ASAN_OPTIONS"]  = "halt_on_error=1:print_stats=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"

    print(f"  Running with 100K scale (sanitizer mode)...")
    proc = subprocess.Popen([str(asan_bin)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=str(HERE))
    out, err = proc.communicate()
    if proc.returncode == 0:
        print(f"  {GREEN}✓ ASAN/UBSAN: no errors detected{RST}")
    else:
        print(f"  {RED}ASAN/UBSAN reported errors:{RST}")
        print(err[:600])
    asan_bin.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NLI Master Script")
    parser.add_argument("--quick",   action="store_true", help="100K scale, skip figures")
    parser.add_argument("--figs",    action="store_true", help="Only regenerate figures from existing CSVs")
    parser.add_argument("--asan",    action="store_true", help="Run ASAN/UBSAN validation")
    parser.add_argument("--compile", action="store_true", help="Only compile, do not run")
    args = parser.parse_args()

    print(f"\n{BLD}{'='*65}{RST}")
    print(f"{BLD}  NLI Master Script  -  CVMI 2026 / IWIN 2026{RST}")
    print(f"{BLD}  Neural Enhanced Learned Index (NLI){RST}")
    print(f"  B-Tree · ALEX · PGM · RMI  -  ALL locally benchmarked")
    print(f"{BLD}{'='*65}{RST}")

    os.chdir(HERE)

    # Phase 1: dependencies
    phase_deps()

    if args.figs:
        phase_figures()
        phase_summary()
        return

    # Phase 2: verify headers (errors out if missing)
    pgm_ok, alex_ok = phase_headers()

    # Phase 3: compile
    ok = phase_compile(pgm_ok, alex_ok)
    if not ok:
        print(f"\n{RED}Cannot continue without compiled binary.{RST}")
        sys.exit(1)

    if args.compile:
        print(f"\n{GREEN}Compile-only mode done. Run again without --compile to benchmark.{RST}")
        return

    if args.asan:
        run_asan()
        return

    # Phase 4: run benchmark
    if args.quick:
        max_scale = 100_000
        print(f"  {YEL}--quick mode: capping at 100K keys{RST}")
    else:
        max_scale = phase_menu()

    ok = phase_run(max_scale)

    # Phase 5: figures
    if ok and not args.quick:
        phase_figures()

    # Phase 6: summary
    phase_summary()


if __name__ == "__main__":
    main()
