# Root first-stage q-grid cut enrichment on the saved baseline cuts.
# Read-only.  python test/_t_qgrid_pilot.py
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


def main():
    pcfg = load_problem_config(ROOT / "config/problem.toml")
    scfg = load_solver_config(ROOT / "config/solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)
    x_sim = np.load(BASE / "sim_2000paths.npz")["x_in"]
    base = np.asarray(x_sim[0, 0], float).copy()
    qbase = Ly.q(base).copy()

    vals = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    anchors = [base.copy()]
    for r in range(Ly.N_R):
        for v in vals:
            x = base.copy()
            q = qbase.copy()
            q[r] = min(prob.Qmax[r], float(v))
            x[Ly.q_slice()] = q
            anchors.append(x)
    # a few multi-warehouse small-order points
    for v in (0.1, 0.5, 1.0):
        x = base.copy()
        q = np.minimum(np.asarray(prob.Qmax, float), v)
        x[Ly.q_slice()] = q
        anchors.append(x)
    # exact duplicates
    uniq = np.unique(np.round(np.asarray(anchors), 9), axis=0)
    anchors = [np.asarray(x, float) for x in uniq]

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    t0 = time.time()
    try:
        eng.cuts = cuts
        eng.restore_cut_history()
        lb0 = eng.lower_bound()
        eng.round_id = 8200
        batch = [[] for _ in range(prob.T + 1)]
        batch[1] = anchors
        nadd = eng.backward(batch, it=1)
        lb = eng.lower_bound()
        print(f"anchors={len(anchors)} +{nadd} cuts1  cuts1={len(eng.cuts[1])}")
        print(f"baseline LB={lb0:.6f}  q-grid LB={lb:.6f}  "
              f"delta={lb-lb0:+.6f} ({(lb-lb0)/abs(lb0)*100:+.4f}%)  "
              f"wall={time.time()-t0:.1f}s")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
