"""Run one (K, threshold) cell and save its metrics.

    python3 run_cell.py --k 8 --exo         --out r.json
    python3 run_cell.py --k 8 --theta 0.5   --out r.json

Self-contained: depends on nothing outside this repository.
"""
import argparse
import json
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pkg"))

import torch  # noqa: E402
import simulation_utils as su  # noqa: E402
import SpikeSuMC_network  # noqa: E402
import run_simulation as rs  # noqa: E402


def build_task(epochs, n_moves, seed, schedule, block, rooms=16):
    """A rule schedule with exact, known change points.

    create_simulation draws change points at random and forbids a switch until
    a rule has been seen 1/H times, so two conditions see different numbers of
    switches at different places and their latencies are not comparable. Here
    the sequence is explicit, so baseline and self-calibrated see an identical
    task and the only possible difference is the threshold.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    movements = su.create_move_list(rooms, n_moves)
    n_rules = max(schedule) + 1

    mazes = [list(torch.arange(rooms))]
    trans = [su.create_trans_matrix(mazes[0], movements, symmetric=True)]
    for _ in range(n_rules - 1):
        m = torch.arange(rooms)
        m = list(m[torch.randperm(m.shape[0])])
        mazes.append(m)
        trans.append(su.create_trans_matrix(m, movements, symmetric=True))

    torch.manual_seed(random.randint(0, int(1e8)))

    maze_seq, cps, cur = [], {}, None
    for e in range(epochs):
        r = schedule[min(e // block, len(schedule) - 1)]
        if r != cur:
            cps[e] = r
            cur = r
        maze_seq.append(r)

    room, rooms_seq = torch.tensor(0), []
    for e in range(epochs):
        room = su.draw_new_room(trans[maze_seq[e]], room)
        rooms_seq.append(room)

    return {"mazes": mazes, "transitions": trans, "epochs": epochs,
            "number_rooms": rooms, "Dirichlet": 0, "volatility": 0.0,
            "n_moves": n_moves, "change_points": cps, "rooms": rooms_seq,
            "maze": [maze_seq[0]] + maze_seq}


def run(n_moves, epochs, block, schedule, seed, exo, theta_fixed):
    with open(os.path.join(HERE, "scripts", "params",
                           "params_network.pkl"), "rb") as f:
        params = pickle.load(f)
    p = pickle.loads(pickle.dumps(params))
    p["n_memory"] = 1
    p["plot"] = False
    p["print"] = False
    p["batch_size"] = 1
    p["SpikeSuM_module"]["plot"] = False
    p["SpikeSuM_module"]["W"] = None
    p["SpikeSuM_module"]["exo_theta"] = bool(exo)

    freeze = p["SpikeSuM_module"].get("exo_freeze", 900)
    if exo and epochs <= freeze:
        raise SystemExit(
            "epochs=%d is too short: the threshold freezes at epoch %d and "
            "would never calibrate. Use more than %d epochs, or lower "
            "exo_freeze." % (epochs, freeze, freeze))
    if exo and block < 800:
        print("WARNING: block=%d. The histogram has tau=300, so with short "
              "episodes it mixes rules and inflates K_est. The constants were "
              "calibrated with episodes of 1000." % block)

    sim = build_task(epochs, n_moves, seed, schedule, block)
    net = SpikeSuMC_network.SpikeSuMC(p, None)
    net.SpikeSuM_module.tosave = None
    module = net.SpikeSuM_module
    if theta_fixed is not None:
        module.theta = torch.tensor([theta_fixed])

    a_trace, th_trace = [], []
    bucket = []
    inner = module.forward

    def traced_forward(*a, **kw):
        out = inner(*a, **kw)
        bucket.append((float(module.network_activity.flatten()[0]),
                       float(torch.as_tensor(module.theta).flatten()[0])))
        return out

    module.forward = traced_forward
    original_pre = rs.pre_step

    def traced_pre(sd, network, sims, epoch):
        if bucket:
            n = len(bucket)
            a_trace.append(sum(x[0] for x in bucket) / n)
            th_trace.append(sum(x[1] for x in bucket) / n)
            bucket.clear()
        return original_pre(sd, network, sims, epoch)

    rs.pre_step = traced_pre
    with torch.no_grad():
        rs.run_simulation([sim], net)
    rs.pre_step = original_pre

    err = [float(e.flatten()[0]) for e in module.info["error"]]
    return {"K": 2 * n_moves, "seed": seed, "epochs": epochs, "block": block,
            "exo": bool(exo), "theta_fixed": theta_fixed,
            "theta_final": th_trace[-1] if th_trace else None,
            "change_points": {int(k): int(v)
                              for k, v in sim["change_points"].items()},
            "A_trace": [round(x, 5) for x in a_trace],
            "theta_trace": [round(x, 5) for x in th_trace],
            "error": [round(x, 7) for x in err]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=(2, 4, 6, 8),
                    help="stochasticity of the rule: possible next rooms")
    ap.add_argument("--exo", action="store_true",
                    help="self-calibrating threshold")
    ap.add_argument("--theta", type=float, default=None,
                    help="fixed threshold instead")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = run(args.k // 2, args.epochs, args.block, [0, 1, 2, 0, 1],
              args.seed, args.exo, args.theta)
    print("K=%d  theta_final=%.4f  ->  %s"
          % (res["K"], res["theta_final"] or 0, args.out))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
