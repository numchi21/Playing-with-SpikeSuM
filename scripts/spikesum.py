# coding: utf-8
"""
Run SpikeSuM-C on the volatile sequence task.
"""

import argparse
import os
import sys

import torch

path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.insert(1, '{0}/pkg/'.format(path))

import pickle  # noqa: E402  (must follow the sys.path line above)

import simulation_utils  # noqa: E402
import save_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--epochs', type=int, default=10000,
                   help='presentation steps per simulation')
    p.add_argument('--modules', type=int, nargs='+', default=[1, 2, 3, 4],
                   metavar='N', help='context module counts to run, one run each')
    p.add_argument('--rooms', type=int, default=16, help='number of stimuli R')
    p.add_argument('--n-moves', type=int, default=2,
                   help='K = 2 * n_moves possible transitions per stimulus')
    p.add_argument('--n-maze', type=int, default=4, help='number of rules')
    p.add_argument('--volatility', type=float, default=0.002,
                   help='H, switch probability per presentation step')
    p.add_argument('--batch-size', type=int, default=1,
                   help='simulations run in parallel')
    p.add_argument('--seed', type=int, default=None,
                   help='reproduce a previous run; random if omitted')
    p.add_argument('--plot', action='store_true',
                   help='draw the diagnostic figures during the run')
    p.add_argument('--save-drives', choices=['HD', 'PE'], default=None,
                   help='record the plasticity drives (SpikeSuM_module.tosave)')
    p.add_argument('--no-activity', dest='keep_activity', action='store_false',
                   help='skip the population activity traces (Fig 3 and '
                        'scripts/activity_plots.ipynb). They are recorded by '
                        'default and cannot be recovered without re-running.')
    p.add_argument('--t-hat-every', type=int, default=100, metavar='N',
                   help='store a T_hat snapshot every N steps, giving the time '
                        'axis of Fig 6E. 0 keeps only the final estimate.')
    return p.parse_args()


def compact(info):
    """Stack list-of-tensor entries into single tensors before pickling.

    forward() appends one small tensor per simulated millisecond, so these
    lists hold ~1e6 objects. Pickling them one by one costs roughly 24 times
    the size of the underlying numbers: 16.7 MB for a 200-epoch run instead of
    0.7 MB. Stacking first keeps every value and makes the traces cheap enough
    to record by default.
    """
    for key, value in list(info.items()):
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            try:
                info[key] = torch.stack([v.reshape(-1) for v in value]).squeeze()
            except RuntimeError:
                pass          # ragged entries: leave as they are
    return info


args = parse_args()

# These are read at import time by SpikeSuM_module, so they have to be set
# before it is imported -- hence the deferred imports below.
if args.keep_activity:
    os.environ['SPIKESUM_RECORD_ACTIVITY'] = '1'
if args.t_hat_every:
    os.environ['SPIKESUM_T_HAT_EVERY'] = str(args.t_hat_every)

import SpikeSuMC_network  # noqa: E402
from run_simulation import run_simulation  # noqa: E402

LOG_DIR = os.path.join(path, 'scripts', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(path, 'results'), exist_ok=True)
open(os.path.join(LOG_DIR, 'logs_single_run.txt'), 'w').close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_file = os.path.join(
    path, 'scripts', 'params',
    'params_network_n_moves_{0}_H_{1}.pkl'.format(args.n_moves, args.volatility))
if not os.path.exists(params_file):
    raise SystemExit(
        'No tuned parameters for n_moves={0}, H={1}.\n'
        'Looked for: {2}\n'
        'Available combinations are the params_network_*.pkl files in '
        'scripts/params/, or build a new one with Set_SpikeSuMM_params.ipynb.'
        .format(args.n_moves, args.volatility, params_file))
with open(params_file, 'rb') as f:
    params = pickle.load(f)

params['plot'] = args.plot
params['SpikeSuM_module']['plot'] = args.plot
params['batch_size'] = args.batch_size

# --- build the simulations --------------------------------------------------
# One seed per batch element. create_simulation seeds both torch and Python's
# random, so printing the seed is enough to reproduce the run exactly, change
# points included.
simulations, seeds = [], []
with torch.no_grad():
    for i in range(args.batch_size):
        seed = args.seed if args.seed is not None \
            else round(2**32 * torch.rand(1).item() - 1)
        seeds += [seed]
        print('Simulation {0}  seed {1}'.format(i + 1, seed))
        simulations += [simulation_utils.create_simulation(
            epochs=args.epochs, number_rooms=args.rooms,
            volatility=args.volatility, n_moves=args.n_moves,
            n_maze=args.n_maze, seed=seed, Dirichlet=0,
            deter_start=None, symmetric=True)]
        print('change points:')
        for key, value in sorted(simulations[i]['change_points'].items()):
            print((key, value), end=' ')
        print()

projection = 'rand' if params['SpikeSuM_module']['random_projection'] else 'onehot'

# The simulations are shared by every module count, so save them once. Nothing
# else records the change points or the ground-truth matrices, and they cannot
# be rebuilt from the result pickle.
save_utils.save(
    data={'simulations': simulations, 'seeds': seeds, 'args': vars(args)},
    file=os.path.join(path, 'results', 'simulation_moves_{0}_{1}'.format(
        args.n_moves, projection)),
    type_='pickle')

# --- run one simulation per module count ------------------------------------
for memories in args.modules:
    print('\n=== {0} context module(s) ==='.format(memories))
    params['n_memory'] = memories
    net = SpikeSuMC_network.SpikeSuMC(params, None)
    net.SpikeSuM_module.tosave = args.save_drives
    criteria, epoch = run_simulation(simulations, net)

    # The module info holds one entry per simulated millisecond for several
    # keys, which would make the pickle enormous. Keep the small summaries
    # plus whatever the user explicitly asked to record.
    info = net.SpikeSuM_module.info
    keep = ['error', 'T_hat']
    if args.keep_activity:
        keep += ['Activity', 'Activity_P1', 'Activity_P2', 'Activity_full']
    if args.t_hat_every:
        keep += ['T_hat_history']
    kept = compact({k: info[k] for k in keep if k in info})
    info.clear()
    info.update(kept)

    save_utils.save(
        data=net.info,
        file=os.path.join(path, 'results',
                          'SpikeSuM_info_moves_{0}_{1}_modules_{2}'.format(
                              args.n_moves, projection, memories)),
        type_='pickle')
    print('saved results/SpikeSuM_info_moves_{0}_{1}_modules_{2}.pkl'.format(
        args.n_moves, projection, memories))