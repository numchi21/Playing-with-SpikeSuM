"""
Build the limitation figure
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_figure")

EPOCHS, BLOCK = 5000, 1000
WARMUP = 600

# Theta grid per K, chosen to bracket the optimum on both sides.
GRID = {
    2: [0.12, 0.18, 0.25, 0.32, 0.44, 0.50, 0.56],
    4: [0.25, 0.32, 0.38, 0.44, 0.50, 0.56],
    6: [0.38, 0.44, 0.50, 0.55, 0.62, 0.68],
    8: [0.32, 0.38, 0.44, 0.50, 0.56, 0.62, 0.68],
}
GRID_QUICK = {
    2: [0.18, 0.32, 0.50],
    4: [0.32, 0.44, 0.56],
    6: [0.44, 0.55, 0.68],
    8: [0.44, 0.56, 0.68],
}


# -- metric --────────────────────────────────────────────────────────
def late_error(d):
    """Error in the post-warm-up window."""
    err = d["error"]
    cps = sorted(int(k) for k in d["change_points"])
    vals = []
    for i, s in enumerate(cps):
        e = cps[i + 1] if i + 1 < len(cps) else len(err)
        if s < WARMUP:
            continue
        seg = err[s:e]
        if len(seg) > 4:
            vals.append(sum(seg[len(seg) // 2:]) / len(seg[len(seg) // 2:]))
    if not vals:                       # short run with no late episodes
        seg = err[len(err) // 2:]
        return sum(seg) / len(seg)
    return sum(vals) / len(vals)


# -- execution --──────────────────────────────────────────────────────
def cells(grid):
    for K in sorted(grid):
        for th in grid[K]:
            yield {"K": K, "theta": th,
                   "out": os.path.join(OUT, "fix_K%d_t%.2f.json" % (K, th))}
    for K in sorted(grid):
        yield {"K": K, "theta": None,
               "out": os.path.join(OUT, "exo_K%d.json" % K)}


def command(c, epochs, block):
    cmd = [sys.executable, os.path.join(HERE, "run_cell.py"),
           "--k", str(c["K"]), "--epochs", str(epochs), "--block", str(block),
           "--seed", "1", "--out", c["out"]]
    cmd += ["--exo"] if c["theta"] is None else ["--theta", str(c["theta"])]
    return cmd


def run_all(grid, jobs, epochs, block):
    os.makedirs(OUT, exist_ok=True)
    todo = [c for c in cells(grid) if not os.path.exists(c["out"])]
    total = len(list(cells(grid)))
    print("%d cells, %d pending" % (total, len(todo)), flush=True)
    if not todo:
        return

    env = dict(os.environ, TQDM_DISABLE="1",
               OMP_NUM_THREADS=str(max(1, 8 // max(1, jobs))))
    running, t0 = [], time.time()
    while todo or running:
        while todo and len(running) < jobs:
            c = todo.pop(0)
            lab = "exo K=%d" % c["K"] if c["theta"] is None \
                else "K=%d theta=%.2f" % (c["K"], c["theta"])
            print("  starting %s" % lab, flush=True)
            running.append((lab, subprocess.Popen(
                command(c, epochs, block), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
        time.sleep(3)
        for item in list(running):
            lab, proc = item
            if proc.poll() is not None:
                running.remove(item)
                if proc.returncode:
                    print("  FAILED %s\n%s" % (
                        lab, proc.stderr.read().decode()[-600:]), flush=True)
                else:
                    print("  done %s" % lab, flush=True)
    print("all computed in %.0f min" % ((time.time() - t0) / 60), flush=True)


# -- figure --─────────────────────────────────────────────────────────
def build_figure(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    INK, GREY, AMBER, TEAL, GREEN = (
        "#18233B", "#64748B", "#C2410C", "#2E7D8F", "#15803D")
    KCOL = {2: "#0F766E", 4: "#2E7D8F", 6: "#B45309", 8: "#9A3412"}

    sweep, system, floors, sigmas = {}, {}, {}, {}
    for K in sorted(grid):
        sweep[K] = {}
        for th in grid[K]:
            f = os.path.join(OUT, "fix_K%d_t%.2f.json" % (K, th))
            if os.path.exists(f):
                sweep[K][th] = late_error(json.load(open(f)))
        f = os.path.join(OUT, "exo_K%d.json" % K)
        if os.path.exists(f):
            d = json.load(open(f))
            system[K] = {"theta": d["theta_final"], "error": late_error(d)}
        # within-episode floor of A~, from the theta=0.5 run
        f = os.path.join(OUT, "fix_K%d_t0.50.json" % K)
        if os.path.exists(f):
            d = json.load(open(f))
            A = d["A_trace"]
            cps = sorted(int(k) for k in d["change_points"])
            segs = []
            for i, s in enumerate(cps):
                e = cps[i + 1] if i + 1 < len(cps) else len(A)
                seg = A[s + 60:e]
                if len(seg) > 30:
                    segs.append(seg)
            if segs:
                floors[K] = sum(sum(s) / len(s) for s in segs) / len(segs)
                flat = [x for s in segs for x in s]
                mu = sum(flat) / len(flat)
                sd = (sum((x - mu) ** 2 for x in flat) / len(flat)) ** 0.5
                sigmas[K] = (0.5 - mu) / sd if sd else float("nan")

    Ks = [K for K in sorted(grid) if sweep.get(K) and K in system]
    if not Ks:
        raise SystemExit("not enough results in " + OUT)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": "#CBD5E1", "axes.labelcolor": INK,
        "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.30,
                          left=0.055, right=0.985, top=0.80, bottom=0.135)

    lo = min(min(v.values()) for v in sweep.values() if v) * 0.65
    hi = max(max(v.values()) for v in sweep.values() if v) * 2.2

    # A
    ax = fig.add_subplot(gs[0])
    for K in Ks:
        xs = sorted(sweep[K])
        ax.plot(xs, [sweep[K][x] for x in xs], "-o", ms=3.2, lw=1.5,
                color=KCOL[K], label="K = %d" % K)
        bx = min(sweep[K], key=sweep[K].get)
        ax.plot([bx], [sweep[K][bx]], "o", ms=8, mfc="none",
                mec=KCOL[K], mew=1.8, zorder=5)
    ax.axvline(0.5, color=AMBER, lw=1.4, ls="--", zorder=1)
    ax.text(0.512, hi * 0.55, "θ = 0.5\nthe repository's value",
            fontsize=8.5, color=AMBER, va="top", linespacing=1.4)
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ax.set_xlabel("surprise threshold θ")
    ax.set_ylabel("transition-matrix error")
    ax.set_title("A · The optimal threshold shifts with K",
                 fontsize=11, weight="bold", loc="left", pad=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              handlelength=1.3, borderaxespad=0.3)
    opt = " → ".join("%.2f" % min(sweep[K], key=sweep[K].get) for K in Ks)
    ax.text(0.03, 0.74, "open circles: measured optimum\n" + opt,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=GREY, linespacing=1.5)

    # B
    ax = fig.add_subplot(gs[1])
    for K in Ks:
        xs = sorted(sweep[K])
        ax.plot(xs, [sweep[K][x] for x in xs], "-", lw=1.2,
                color=KCOL[K], alpha=0.45)
        ax.plot([system[K]["theta"]], [system[K]["error"]], "*", ms=15,
                color=KCOL[K], mec="white", mew=0.8, zorder=6)
        if 0.50 in sweep[K]:
            ax.plot([0.5], [sweep[K][0.50]], "x", ms=7, mew=2,
                    color=AMBER, zorder=5)
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ax.set_xlabel("surprise threshold θ")
    ax.set_ylabel("transition-matrix error")
    ax.set_title("B · Where the network places it by itself",
                 fontsize=11, weight="bold", loc="left", pad=9)
    ax.legend(handles=[
        Line2D([], [], marker="*", ls="none", ms=13, color=INK,
               label="self-calibrated (no per-K tuning)"),
        Line2D([], [], marker="x", ls="none", ms=7, mew=2, color=AMBER,
               label="θ = 0.5 fixed")],
        frameon=False, fontsize=8.5, loc="upper left", borderaxespad=0.3)
    ax.text(0.97, 0.70,
            "stars land on the minimum\nof each curve",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=GREY, linespacing=1.5)

    # C
    ax = fig.add_subplot(gs[2])
    xs, wd = range(len(Ks)), 0.27
    base = [sweep[K].get(0.50, float("nan")) for K in Ks]
    auto = [system[K]["error"] for K in Ks]
    best = [min(sweep[K].values()) for K in Ks]
    ax.bar([x - wd for x in xs], base, wd, color=AMBER, label="θ = 0.5 (paper)")
    ax.bar(list(xs), auto, wd, color=TEAL, label="self-calibrated")
    ax.bar([x + wd for x in xs], best, wd, color="#CBD5E1",
           label="attainable optimum")
    for x, K, b, a in zip(xs, Ks, base, auto):
        if b == b:
            r = b / a
            ax.text(x - wd, b * 1.35, "%.1f×" % r, ha="center",
                    fontsize=9.5, weight="bold",
                    color=GREEN if r > 1.15 else GREY)
    ax.set_yscale("log"); ax.set_ylim(lo, hi * 1.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["K=%d" % K for K in Ks])
    ax.set_ylabel("transition-matrix error")
    ax.set_title("C · Gain over the paper's value",
                 fontsize=11, weight="bold", loc="left", pad=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              borderaxespad=0.3)

    fig.text(0.055, 0.945,
             "The K-dependence of the surprise threshold, and its fix",
             fontsize=15, weight="bold", color=INK)
    if floors:
        sub = ("With a fixed θ the floor of A͂ rises with K (%s), so θ = 0.5 "
               "sits at %s above it — conservative in one regime, buried in "
               "the noise in another. The network estimates K from the "
               "observed transitions and sets θ by itself."
               % (" → ".join("%.3f" % floors[K] for K in Ks if K in floors),
                  " / ".join("%.1fσ" % sigmas[K] for K in Ks if K in sigmas)))
    else:
        sub = ("The network estimates K from the observed transitions and "
               "sets θ by itself.")
    fig.text(0.055, 0.885, sub, fontsize=9, color=GREY)

    dst = os.path.join(HERE, "figure_limitation")
    fig.savefig(dst + ".png", dpi=200, facecolor="white")
    fig.savefig(dst + ".pdf", facecolor="white")
    print("\nfigure written: %s.png and %s.pdf" % (dst, dst))

    print("\n%4s %10s %12s %10s %9s" % ("K", "θ auto", "err auto", "err 0.5", "gain"))
    for K in Ks:
        b = sweep[K].get(0.50, float("nan"))
        print("%4d %10.4f %12.5f %10.5f %8.1f×"
              % (K, system[K]["theta"], system[K]["error"], b,
                 b / system[K]["error"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true",
                    help="fewer theta points and shorter runs")
    ap.add_argument("--figure-only", action="store_true",
                    help="run nothing, just redraw")
    args = ap.parse_args()

    grid = GRID_QUICK if args.quick else GRID
    epochs = 3000 if args.quick else EPOCHS    # comfortably past exo_freeze
    block = 1000

    if not args.figure_only:
        run_all(grid, args.jobs, epochs, block)
    build_figure(grid)
