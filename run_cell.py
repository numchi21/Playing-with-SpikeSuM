"""Run one (K, threshold) cell and save its metrics.

    python3 run_cell.py --k 8 --theta 0.5    --out r.json   # baseline
    python3 run_cell.py --k 8 --exo          --out r.json   # exogenous
    python3 run_cell.py --k 8 --mean-c 0.05  --out r.json   # additive
    python3 run_cell.py --k 8 --hybrid       --out r.json   # both

Uses create_simulation, so change points are drawn with probability
`volatility` per epoch subject to a minimum dwell of 1/volatility.
"""
import argparse, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pkg"))

import torch
import simulation_utils as su
import SpikeSuMC_network
import run_simulation as rs


def run(n_moves, epochs, seed, volatility, exo, theta_fixed, mean_c, hybrid,
        mean_tau=300.0, n_memory=1, n_maze=3):
    with open(os.path.join(HERE, "scripts", "params",
                           "params_network.pkl"), "rb") as f:
        params = pickle.load(f)
    p = pickle.loads(pickle.dumps(params))
    p["n_memory"] = n_memory
    p["plot"] = False
    p["print"] = False
    p["batch_size"] = 1
    p["SpikeSuM_module"]["plot"] = False
    p["SpikeSuM_module"]["W"] = None
    # The hybrid needs both halves: the histogram for the margin and the
    # running mean for the level. exo_freeze is pushed past the end of the run
    # so K_est keeps updating instead of being frozen.
    p["SpikeSuM_module"]["exo_theta"] = bool(exo) or bool(hybrid)
    p["SpikeSuM_module"]["hybrid_theta"] = bool(hybrid)
    p["SpikeSuM_module"]["mean_theta"] = (mean_c is not None) or bool(hybrid)
    if hybrid:
        p["SpikeSuM_module"]["exo_freeze"] = 10 ** 9
    p["SpikeSuM_module"]["mean_tau"] = mean_tau
    if mean_c is not None:
        p["SpikeSuM_module"]["mean_c"] = mean_c

    freeze = p["SpikeSuM_module"].get("exo_freeze", 900)
    if exo and epochs <= freeze:
        raise SystemExit("epochs=%d is too short: the threshold freezes at "
                         "epoch %d and would never calibrate." % (epochs, freeze))
    if (mean_c is not None or hybrid) and epochs <= 400:
        raise SystemExit("epochs=%d is too short: the running mean needs a "
                         "200-epoch warm-up." % epochs)

    with torch.no_grad():
        sim = su.create_simulation(
            epochs=epochs, number_rooms=16, volatility=volatility,
            n_moves=n_moves, n_maze=n_maze, seed=seed, Dirichlet=0,
            deter_start=None, symmetric=True)

    net = SpikeSuMC_network.SpikeSuMC(p, None)
    net.SpikeSuM_module.tosave = None
    module = net.SpikeSuM_module
    if theta_fixed is not None:
        module.theta = torch.tensor([theta_fixed])

    a_trace, th_trace, bucket = [], [], []
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
    cps = {int(k): int(v) for k, v in sim["change_points"].items()}
    return {"K": 2 * n_moves, "seed": seed, "epochs": epochs,
            "volatility": volatility, "exo": bool(exo), "hybrid": bool(hybrid),
            "mean_c": mean_c, "mean_tau": mean_tau,
            "theta_fixed": theta_fixed,
            "theta_final": th_trace[-1] if th_trace else None,
            "change_points": cps, "n_change_points": len(cps),
            "A_trace": [round(x, 5) for x in a_trace],
            "theta_trace": [round(x, 5) for x in th_trace],
            "error": [round(x, 7) for x in err]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=(2, 4, 6, 8))
    ap.add_argument("--exo", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--mean-c", type=float, default=None)
    ap.add_argument("--mean-tau", type=float, default=300.0)
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--volatility", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-memory", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.k // 2, a.epochs, a.seed, a.volatility, a.exo, a.theta,
              a.mean_c, a.hybrid, a.mean_tau, a.n_memory)
    print("K=%d  theta_final=%.4f  change_points=%d  ->  %s"
          % (res["K"], res["theta_final"] or 0, res["n_change_points"], a.out))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
