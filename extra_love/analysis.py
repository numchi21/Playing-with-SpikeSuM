"""
Turns the pickles produced by scripts/spikesum.py into figures and numbers.

The change points and the seed are NOT stored in the result pickles, so they
have to be copied from what the script printed when it started.

The change points cannot be rebuilt:
simulation_utils.py line 204 re-seeds with random.randint(), which is never
seeded, so the switch times differ on every call.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# --- where things live ------------------------------------------------------
# This file sits in <repo>/extra_love/, so the repository root is one level up.
# Everything is resolved from here, never from the current working directory,
# so the script can be launched from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pkg"))

# ===========================================================================
# CONFIG - edit these to match the run you want to analyse
# ===========================================================================
SEED = 2640095231

CHANGE_POINTS = (
    "(0, 0) (1081, 3) (1837, 2) (2644, 1) (3688, 2) (3792, 0) (4693, 1) "
    "(5252, 2) (5538, 3) (5995, 1) (7132, 0) (7572, 2) (7727, 0) (7856, 1) "
    "(8120, 2) (8756, 3) (8881, 0)"
)

# these must match scripts/spikesum.py
NUMBER_ROOMS = 16
N_MOVES = 2
N_MAZE = 4

DEFAULT_GLOB = os.path.join(ROOT, "results", "SpikeSuM_info*.pkl")
OUTDIR = os.path.join(HERE, "output", "results_analysis")
# ===========================================================================


def to_numpy(x):
    """Accept torch tensors, lists of tensors, or plain arrays."""
    if isinstance(x, list):
        parts = [to_numpy(v) for v in x]
        return (np.concatenate([np.atleast_1d(p).ravel() for p in parts])
                if parts else np.array([]))
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_change_points(text: str):
    """Parse the '(0, 0) (1081, 3) ...' line printed at startup.

    Returns the sorted switch epochs and the maze active at the end.
    """
    pairs = sorted((int(a), int(b)) for a, b in
                   re.findall(r"\((\d+)\s*,\s*(\d+)\)", text))
    if not pairs:
        return [], 0
    return [e for e, _ in pairs], pairs[-1][1]


def rebuild_true_matrix(seed: int, maze_index: int, args):
    """Rebuild T* from the seed, or return None if that is not possible."""
    try:
        import simulation_utils
    except ImportError as exc:
        print(f"  (cannot rebuild T*: {exc})")
        return None
    sim = simulation_utils.create_simulation(
        epochs=1, number_rooms=args.number_rooms, volatility=0.0,
        n_moves=args.n_moves, n_maze=args.n_maze, seed=seed,
        Dirichlet=0, deter_start=None, symmetric=True)
    # create_trans_matrix stores [previous, next]; T_hat is [next, previous]
    return to_numpy(sim["transitions"][maze_index]).T


def learning_curve(info: dict) -> np.ndarray:
    error = to_numpy(info["error"]).ravel()
    if error.size and error[0] == 0.0:
        error = error[1:]   # written before any estimate exists
    return error


def label_from_path(path: str) -> str:
    base = os.path.basename(path).replace(".pkl", "")
    return base.split("SpikeSuM_info_")[-1]


def plot_learning_curve(error, change_points, label, outdir, window=25):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(error, lw=0.6, alpha=0.35, color="steelblue")
    if error.size > window:
        smooth = np.convolve(error, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window // 2, window // 2 + smooth.size), smooth,
                lw=1.5, color="navy", label=f"moving average ({window})")
        ax.legend(fontsize=9)
    for cp in change_points:
        ax.axvline(cp, color="crimson", ls="--", lw=0.8, alpha=0.65)
    ax.set_xlabel("presentation step")
    ax.set_ylabel(r"MSE$(T^*, \hat{T})$")
    ax.set_title(f"{label}  (dashed = rule switches)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"learning_curve_{label}.png"), dpi=140)
    plt.close(fig)


def plot_transition_matrices(T_hat, T_true, label, outdir):
    n = 1 if T_true is None else 2
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), squeeze=False)
    vmax = float(T_hat.max())
    if T_true is not None:
        im = axes[0][0].imshow(T_true, vmin=0, vmax=vmax, cmap="viridis")
        axes[0][0].set_title(r"ground truth $T^*$")
        fig.colorbar(im, ax=axes[0][0], fraction=0.046)
    ax = axes[0][n - 1]
    im = ax.imshow(T_hat, vmin=0, vmax=vmax, cmap="viridis")
    ax.set_title(r"decoded $\hat{T}$")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for a in axes[0]:
        a.set_xlabel("previous stimulus")
        a.set_ylabel("next stimulus")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"transition_matrices_{label}.png"), dpi=140)
    plt.close(fig)


def plot_comparison(curves, change_points, outdir, window=101):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, err in curves.items():
        if err.size > window:
            smooth = np.convolve(err, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window // 2, window // 2 + smooth.size), smooth,
                    lw=1.4, label=label)
        else:
            ax.plot(err, lw=1.0, label=label)
    for cp in change_points:
        ax.axvline(cp, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("presentation step")
    ax.set_ylabel(r"MSE$(T^*, \hat{T})$")
    ax.set_title(f"Comparison (moving average, {window} steps)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), dpi=140)
    plt.close(fig)


def recovery_stats(error, change_points):
    """How much does the error jump at a switch, and how fast does it recover?"""
    cps = [cp for cp in change_points if 50 < cp < error.size - 100]
    if not cps:
        return None
    baseline = np.array([error[cp - 50:cp].mean() for cp in cps])
    peak = np.array([error[cp:cp + 5].mean() for cp in cps])
    halftimes = []
    for cp, b, p in zip(cps, baseline, peak):
        target = b + 0.5 * (p - b)
        below = np.flatnonzero(error[cp:cp + 300] < target)
        if below.size:
            halftimes.append(int(below[0]))
    return {
        "n": len(cps),
        "before": baseline.mean(),
        "after": peak.mean(),
        "ratio": peak.mean() / baseline.mean() if baseline.mean() else float("nan"),
        "halftime": float(np.median(halftimes)) if halftimes else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse SpikeSuM result pickles.")
    ap.add_argument("pickle_file", nargs="*", default=None,
                    help=f"result pickles (default: {DEFAULT_GLOB})")
    ap.add_argument("--sim", default=None,
                    help="optional pickle of the simulation dictionary; "
                         "overrides --change-points and --seed")
    ap.add_argument("--change-points", default=CHANGE_POINTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--number-rooms", type=int, default=NUMBER_ROOMS)
    ap.add_argument("--n-moves", type=int, default=N_MOVES)
    ap.add_argument("--n-maze", type=int, default=N_MAZE)
    ap.add_argument("--module", default="best",
                    help="which context module's T_hat to plot: an index, "
                         "or 'best' (default) to pick the one closest to T*")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    files = args.pickle_file or sorted(glob.glob(DEFAULT_GLOB))

    # scripts/spikesum.py now saves the simulation next to the results, so the
    # change points and the true matrices no longer have to be pasted by hand.
    if args.sim is None:
        found = sorted(glob.glob(os.path.join(ROOT, 'results', 'simulation_*.pkl')))
        if found:
            args.sim = found[0]
            print('using {0} for the change points'.format(
                os.path.relpath(args.sim, ROOT)))
    if not files:
        raise SystemExit(
            f"No result pickles found.\n"
            f"Looked for: {DEFAULT_GLOB}\n"
            f"If they are somewhere else, pass them explicitly:\n"
            f"    python extra_love/analysis.py /path/to/SpikeSuM_info*.pkl")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Analysing {len(files)} file(s)\n")

    # --- ground truth, shared by every file of the same run ---------------
    if args.sim:
        with open(args.sim, "rb") as f:
            sim = pickle.load(f)
        if isinstance(sim, dict) and 'simulations' in sim:
            sim = sim['simulations'][0]          # written by the new spikesum.py
        else:
            sim = sim[0] if isinstance(sim, list) else sim
        change_points = sorted(sim["change_points"].keys())
        last_maze = sim["change_points"][change_points[-1]]
        T_true = to_numpy(sim["transitions"][last_maze]).T
        source = 'from the saved simulation' 
    else:
        change_points, last_maze = parse_change_points(args.change_points)
        T_true = rebuild_true_matrix(args.seed, last_maze, args)
        source = 'rebuilt from seed ' + str(args.seed)

    print(f"change points : {len(change_points)}")
    print(f"T*            : {source if T_true is not None else 'not available'}\n")

    # --- one file at a time ----------------------------------------------
    curves, rows = {}, []
    for path in files:
        label = label_from_path(path)
        with open(path, "rb") as f:
            info = pickle.load(f)

        error = learning_curve(info)
        curves[label] = error

        # T_hat has shape (batch, n_memory, R, R): one estimate per context
        # module. With several modules, module 0 is not necessarily the one
        # that encodes the maze active at the end of the run.
        T_all = to_numpy(info["SpikeSuM_module"]["T_hat"])
        T_all = T_all.reshape(-1, T_all.shape[-2], T_all.shape[-1])
        if args.module == "best" and T_true is not None:
            k = int(np.argmin([np.mean((t - T_true) ** 2) for t in T_all]))
        else:
            k = 0 if args.module == "best" else int(args.module)
        T_hat = T_all[k]
        if T_all.shape[0] > 1:
            print(f"{label}: {T_all.shape[0]} context modules, showing module {k}")

        plot_learning_curve(error, change_points, label, args.outdir)
        plot_transition_matrices(T_hat, T_true, label, args.outdir)

        stats = recovery_stats(error, change_points)
        rows.append({
            "run": label,
            "n_modules": T_all.shape[0],
            "module_shown": k,
            "error_start": round(float(error[:20].mean()), 6),
            "error_end": round(float(error[-100:].mean()), 6),
            "error_before_switch": round(float(stats["before"]), 6) if stats else "",
            "error_after_switch": round(float(stats["after"]), 6) if stats else "",
            "jump_ratio": round(float(stats["ratio"]), 2) if stats else "",
            "recovery_steps_median": int(stats["halftime"]) if stats else "",
            "mse_final_T": (round(float(np.mean((T_hat - T_true) ** 2)), 6)
                            if T_true is not None else ""),
        })

    if len(curves) > 1:
        plot_comparison(curves, change_points, args.outdir)

    # --- summary table ----------------------------------------------------
    cols = ["run", "n_modules", "module_shown", "error_start", "error_end",
            "error_before_switch", "error_after_switch", "jump_ratio",
            "recovery_steps_median", "mse_final_T"]

    def fmt(v):
        # matches the rounding already applied when the rows were built
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    widths = [max(len(c), max((len(fmt(r[c])) for r in rows), default=0)) for c in cols]
    print()
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(fmt(r[c]).rjust(w) for c, w in zip(cols, widths)))

    print("\nn_modules             : context modules in this run")
    print("module_shown          : which one the T_hat figure uses (best match to T*)")
    print("error_start / _end    : mean MSE over the first 20 and last 100 steps")
    print("error_before/_after   : mean MSE just before and just after a rule switch")
    print("jump_ratio            : after / before")
    print("recovery_steps_median : steps to halve the jump (lower is better)")
    print("mse_final_T           : error of the final decoded matrix")

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nfigures written to {os.path.relpath(args.outdir, ROOT)}/")
    print(f"table written to   {os.path.relpath(csv_path, ROOT)}")


if __name__ == "__main__":
    main()
    