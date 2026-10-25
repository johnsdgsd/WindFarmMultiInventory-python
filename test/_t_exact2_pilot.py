# Exact one-step scenario-state cut chain:
#   - run 300 forward paths, one per stage-1 scenario, with baseline cuts;
#   - collect the exact stage-2 states x2(s) that the stage-1 LP evaluates;
#   - add stage-2 cuts at all x2(s) using current cuts[3];
#   - re-cut stage 1 at x1* using the enriched cuts[2];
#   - measure root LB.
# Read-only with respect to results/.  python test/_t_exact2_pilot.py
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


def _anchors1(prob, x1):
    Ly = prob.layout
    base = np.asarray(x1, float).copy()
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
    scfg = load_solver_config(ROOT / "config/solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    t0 = time.time()
    try:
        eng.cuts = cuts
        eng.restore_cut_history()
        lb0 = eng.lower_bound()
        print(f"baseline LB = {lb0:.6f}  cuts1={len(eng.cuts[1])} cuts2={len(eng.cuts[2])}")

        eng.round_id = 7000
        # one forward path per stage-1 scenario; x2 depends only on stage 1
        paths = [[s] * prob.T for s in range(len(prob.pool))]
        res = eng.forward_paths(paths)
        x1 = np.asarray(res[0][0][0], float)
        x2 = np.asarray([r[0][1] for r in res], float)
        x2 = np.unique(np.round(x2, 9), axis=0)
        print(f"forward paths={len(res)}  unique x2={len(x2)}  "
              f"forward time={time.time()-t0:.1f}s")

        # stage-2 cuts at the exact stage-1 LP successor states
        batch = [[] for _ in range(prob.T + 1)]
        batch[2] = [np.asarray(x, float) for x in x2]
        n2 = eng.backward(batch, it=1)
        print(f"stage2: +{n2} cuts  cuts2={len(eng.cuts[2])}")

        # stage-1 recut at the root state and q perturbations
        batch = [[] for _ in range(prob.T + 1)]
        batch[1] = _anchors1(prob, x1)
        n1 = eng.backward(batch, it=1)
        print(f"stage1: +{n1} cuts  cuts1={len(eng.cuts[1])}")

        lb = eng.lower_bound()
        print(f"new LB = {lb:.6f}  delta={lb-lb0:+.6f} "
              f"({(lb-lb0)/abs(lb0)*100:+.4f}%)  wall={time.time()-t0:.1f}s")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
