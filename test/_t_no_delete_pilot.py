# Diagnostic: K=10 root-policy chain with LML1 disabled.
# Tests whether the K=10 negative result was caused by LML1 deleting cuts.
# Read-only with respect to results/.  python test/_t_no_delete_pilot.py
import dataclasses
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
    return [np.asarray(x, float) for x in np.unique(np.round(s, 9), axis=0)]


def _anchors1(prob, x_sim):
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


def main():
    pcfg = load_problem_config(ROOT / "config/problem.toml")
    scfg = dataclasses.replace(load_solver_config(ROOT / "config/solver.toml"),
                               cut_selection="none")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)
    x_sim = np.load(BASE / "sim_2000paths.npz")["x_in"]
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    t0 = time.time()
    try:
        eng.cuts = cuts
        lb0 = eng.lower_bound()
        print(f"baseline(no-LML1) LB={lb0:.6f} cuts1={len(eng.cuts[1])} cuts2={len(eng.cuts[2])}")
        eng.round_id = 8100
        K = 10
        for k in range(K, 1, -1):
            states = _stage_states(x_sim, k)
            batch = [[] for _ in range(prob.T + 1)]
            batch[k] = states
            eng.backward(batch, it=1)
            print(f"  stage {k}: states={len(states):3d} cuts[{k}]={len(eng.cuts[k])}")
        anchors = _anchors1(prob, x_sim)
        batch = [[] for _ in range(prob.T + 1)]
        batch[1] = anchors
        eng.backward(batch, it=1)
        lb = eng.lower_bound()
        print(f"no-delete K={K}: LB={lb:.6f} delta={lb-lb0:+.6f} "
              f"({(lb-lb0)/abs(lb0)*100:+.4f}%) cuts1={len(eng.cuts[1])} "
              f"cuts2={len(eng.cuts[2])} wall={time.time()-t0:.1f}s")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
