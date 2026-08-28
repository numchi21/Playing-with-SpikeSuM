"""
module_map.py

The module axis of Fig 6E, from the pickles.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis import (  # noqa: E402  (also puts <repo>/pkg on sys.path)
    CHANGE_POINTS, DEFAULT_GLOB, HERE, N_MAZE, N_MOVES, NUMBER_ROOMS, ROOT,
    SEED, label_from_path, to_numpy,
)

OUTDIR = os.path.join(HERE, "output", "results_map")


def all_true_matrices(seed: int, number_rooms: int, n_moves: int,
                      n_maze: int) -> np.ndarray:
    """Regenerate every ground-truth transition matrix from the seed.

    create_simulation() draws the mazes right after torch.manual_seed(seed), so
    they are reproducible even though the switch times are not.
    Returned as [maze][next, previous] to match the orientation of T_hat.
    """
    import simulation_utils

    sim = simulation_utils.create_simulation(
        epochs=1, number_rooms=number_rooms, volatility=0.0, n_moves=n_moves,
        n_maze=n_maze, seed=seed, Dirichlet=0, deter_start=None, symmetric=True)
    return np.stack([to_numpy(t).T for t in sim["transitions"]])


def module_matrices(info: dict) -> np.ndarray:
    """The final T_hat of every context module, shape (n_memory, R, R)."""
    T = to_numpy(info["SpikeSuM_module"]["T_hat"])
    return T.reshape(-1, T.shape[-2], T.shape[-1])


def assignments(mse):
    """Greedy and one-to-one module -> rule assignments.

    Greedy lets each module pick its closest rule independently, so two modules
    can claim the same one and another rule is left with none. That is a real
    outcome, but it also appears when one module's second choice loses by a
    hair. The one-to-one assignment (Hungarian algorithm) minimises the total
    MSE subject to each rule being used at most once, which is the reading Fig
    6E implies. Comparing the two says whether the duplication is meaningful.
    """
    greedy = mse.argmin(axis=1)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(mse)
        optimal = np.full(mse.shape[0], -1, dtype=int)
        optimal[rows] = cols
    except ImportError:
        optimal = greedy
    return greedy, optimal


def plot_specialisation(mse: np.ndarray, label: str, outdir: str,
                        visits: dict | None = None,
                        assignment: np.ndarray | None = None) -> None:
    """Heat map of MSE[module, maze] with the chosen match per module marked."""
    n_mod, n_maze = mse.shape
    fig, ax = plt.subplots(figsize=(1.3 * n_maze + 3, 1.0 * n_mod + 2.4))
    im = ax.imshow(mse, cmap="viridis_r", aspect="auto")

    best = mse.argmin(axis=1) if assignment is None else assignment
    for i in range(n_mod):
        for j in range(n_maze):
            hit = (j == best[i])
            ax.text(j, i, f"{mse[i, j]:.4f}", ha="center", va="center",
                    fontsize=9, color="white" if hit else "0.75",
                    fontweight="bold" if hit else "normal")
            if hit:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="crimson", lw=2.2))

    ax.set_xticks(range(n_maze))
    ax.set_xticklabels([f"rule {j}" + (f"\n({visits[j]}x)" if visits else "")
                        for j in range(n_maze)])
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels([f"module {i}" for i in range(n_mod)])
    ax.set_title(f"{label}\nMSE of each module against each rule "
                 f"(red = best match)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="MSE")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"module_specialisation_{label}.png"), dpi=140)
    plt.close(fig)


def plot_module_grid(T_true: np.ndarray, T_mod: np.ndarray, label: str,
                     outdir: str, assignment: np.ndarray | None = None) -> None:
    """Ground-truth rules on the top row; each module under its assigned rule."""
    n_mod, n_maze = T_mod.shape[0], T_true.shape[0]
    if assignment is None:
        assignment = np.array(
            [[np.mean((T_mod[i] - T_true[j]) ** 2) for j in range(n_maze)]
             for i in range(n_mod)]).argmin(axis=1)
    best = assignment
    vmax = float(T_true.max())

    fig, axes = plt.subplots(n_mod + 1, n_maze,
                             figsize=(2.1 * n_maze, 2.1 * (n_mod + 1)),
                             squeeze=False)
    for j in range(n_maze):
        axes[0][j].imshow(T_true[j], vmin=0, vmax=vmax, cmap="viridis")
        axes[0][j].set_title(f"rule {j}", fontsize=10)
    axes[0][0].set_ylabel("ground truth", fontsize=10)

    for i in range(n_mod):
        for j in range(n_maze):
            ax = axes[i + 1][j]
            if j == best[i]:
                ax.imshow(T_mod[i], vmin=0, vmax=vmax, cmap="viridis")
            else:
                ax.set_facecolor("0.95")
        axes[i + 1][0].set_ylabel(f"module {i}", fontsize=10)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{label} — each module's final estimate, "
                 f"placed under the rule it matches", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"module_grid_{label}.png"), dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("pickle_file", nargs="*", default=None)
    ap.add_argument("--sim", default=None,
                    help="simulation pickle; found automatically in results/")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--change-points", default=CHANGE_POINTS)
    ap.add_argument("--number-rooms", type=int, default=NUMBER_ROOMS)
    ap.add_argument("--n-moves", type=int, default=N_MOVES)
    ap.add_argument("--n-maze", type=int, default=N_MAZE)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    files = args.pickle_file or sorted(glob.glob(DEFAULT_GLOB))
    if not files:
        raise SystemExit(f"No result pickles found.\nLooked for: {DEFAULT_GLOB}")
    os.makedirs(args.outdir, exist_ok=True)

    # scripts/spikesum.py saves the simulation next to the results. Prefer it:
    # regenerating from the CONFIG seed silently compares against a different
    # set of mazes whenever the run used another seed.
    sim_files = sorted(glob.glob(os.path.join(ROOT, 'results', 'simulation_*.pkl')))
    if args.sim or sim_files:
        with open(args.sim or sim_files[0], 'rb') as f:
            sim = pickle.load(f)
        if isinstance(sim, dict) and 'simulations' in sim:
            sim = sim['simulations'][0]
        elif isinstance(sim, list):
            sim = sim[0]
        T_true = np.stack([to_numpy(t).T for t in sim['transitions']])
        pairs = sorted((int(e), int(m)) for e, m in sim['change_points'].items())
        print(f"{T_true.shape[0]} ground-truth rules from "
              f"{os.path.relpath(args.sim or sim_files[0], ROOT)}")
    else:
        T_true = all_true_matrices(args.seed, args.number_rooms, args.n_moves,
                                   args.n_maze)
        pairs = sorted((int(a), int(b)) for a, b in
                       re.findall(r"\((\d+)\s*,\s*(\d+)\)", args.change_points))
        print(f"{T_true.shape[0]} ground-truth rules rebuilt from "
              f"seed {args.seed}")
    n_maze = T_true.shape[0]
    visits = {j: sum(1 for _, m in pairs if m == j) for j in range(n_maze)}
    print("rule presentations:", ", ".join(f"rule {j}: {n}x"
                                           for j, n in visits.items()))

    for path in files:
        label = label_from_path(path)
        with open(path, "rb") as f:
            info = pickle.load(f)
        T_mod = module_matrices(info)
        if T_mod.shape[0] < 2:
            print(f"\n{label}: single module, nothing to map — skipped")
            continue

        mse = np.array([[np.mean((m - t) ** 2) for t in T_true] for m in T_mod])
        greedy, optimal = assignments(mse)

        print(f"\n=== {label} : {T_mod.shape[0]} modules, "
              f"{T_true.shape[0]} rules ===")
        head = "          " + "".join(f"  rule {j}" for j in range(n_maze))
        print(head + "     closest   one-to-one")
        for i, row in enumerate(mse):
            cells = "".join(f"{v:8.4f}" for v in row)
            mark = " " if greedy[i] == optimal[i] else " *"
            print(f"module {i}  {cells}   -> rule {greedy[i]}"
                  f"      rule {optimal[i]}{mark}")

        distinct = len(set(greedy.tolist()))
        print(f"  closest match    : {distinct} distinct rule(s) "
              f"for {T_mod.shape[0]} modules")
        if not np.array_equal(greedy, optimal):
            gap = np.array([mse[i, optimal[i]] - mse[i, greedy[i]]
                            for i in range(len(greedy))])
            spread = float(mse.max() - mse.min())
            print("  one-to-one       : differs on the rows marked *")
            print(f"  cost of forcing it: {gap.sum():.4f} MSE, "
                  f"{100 * gap.sum() / spread:.0f}% of the spread in this table")
            if gap.sum() < 0.1 * spread:
                print("    small, so the duplication is a near-tie and the "
                      "one-to-one reading is the fair one")
            else:
                print("    large, so the duplication is real: those modules "
                      "genuinely converged on the same rule")
        else:
            print("  one-to-one       : same assignment, each module owns a "
                  "different rule")

        plot_specialisation(mse, label, args.outdir, visits, optimal)
        plot_module_grid(T_true, T_mod, label, args.outdir, optimal)

    print(f"\nfigures written to {os.path.relpath(args.outdir, ROOT)}/")


if __name__ == "__main__":
    main()