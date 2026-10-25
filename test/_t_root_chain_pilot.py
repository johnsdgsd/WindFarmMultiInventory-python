# Root-policy multi-stage cut-chain preflight.
# Use states from the saved baseline 2000-path simulation (the root policy),
# enrich cuts backward at stages K..2, then re-cut stage 1 and measure root LB.
# Read-only with respect to results/.
#
#   python test/_t_root_chain_pilot.py
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine                                          # noqa: E402
from utils.io import load_cuts                                        # noqa: E402

BASE = ROOT / "results/base_T60_20261025_131043"
N_STATES = 200


def _stage_states(x_sim, k, n=N_STATES):
    idx = np.linspace(0, x_sim.shape[0] - 1, n, dtype=int)
    s = np.asarray([x_sim[i, k - 1] for i in idx], float)
    # exact duplicate removal, then keep up to n
    s = np.unique(np.round(s, 9), axis=0)
    return [np.asarray(x, float) for x in s]


def _stage1_anchors(prob, x_sim):
    Ly = prob.layout
    base = np.asarray(x_sim[0, 0], float)
    out = [base]
    q0 = Ly.q(base).copy()
    for r in range(min(4, Ly.N_R)):
        x = base.copy()
        q = q0.copy()
        q[r] = min(prob.Qmax[r], q[r] + 1.0)
        x[Ly.q_slice()] = q
        out.append(x)
    return out


def run_depth(K):
    pcfg = load_problem_config(ROOT / "config/problem.toml")
    scfg = load_solver_config(ROOT / "config/solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)
    x_sim = np.load(BASE / "sim_2000paths.npz")["x_in"]

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    t0 = time.time()
    try:
        eng.cuts = cuts
        eng.restore_cut_history()
        lb0 = eng.lower_bound()
        before2 = len(eng.cuts[2])
        before1 = len(eng.cuts[1])
        eng.round_id = 8000 + K

        counts = {}
        for k in range(K, 1, -1):
            states = _stage_states(x_sim, k)
            batch = [[] for _ in range(prob.T + 1)]
            batch[k] = states
            nadd = eng.backward(batch, it=1)
            counts[k] = (len(states), nadd, len(eng.cuts[k]))
            print(f"  K={K:2d} stage {k:2d}: states={len(states):3d} "
                  f"+{nadd:3d} cuts -> {len(eng.cuts[k]):5d}")

        anchors = _stage1_anchors(prob, x_sim)
        batch = [[] for _ in range(prob.T + 1)]
        batch[1] = anchors
        nadd1 = eng.backward(batch, it=1)
        lb = eng.lower_bound()
        print(f"  K={K:2d} stage  1: anchors={len(anchors):2d} +{nadd1:2d} cuts")
        print(f"K={K:2d}: baseline LB={lb0:.6f}  new LB={lb:.6f}  "
              f"delta={lb-lb0:+.6f} ({(lb-lb0)/abs(lb0)*100:+.3f}%)  "
              f"cuts1 {before1}->{len(eng.cuts[1])}  cuts2 {before2}->{len(eng.cuts[2])}  "
              f"time={time.time()-t0:.1f}s")
        return lb0, lb
    finally:
        eng.close()


def main():
    print("Root-policy cut-chain pilot: baseline 2000-path states, 200/stage")
    t0 = time.time()
    for K in (2, 5, 10):
        run_depth(K)
    print(f"total wall = {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
