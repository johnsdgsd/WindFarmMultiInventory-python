# _t_nogood_timebudget.py — q 区域 L_R 对单场景 MILP 时间预算的敏感性。
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils import copt_env as ce                                  # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"
BUDGETS = (0.1, 0.5, 1.0)
CASES = (("stage0-q", 1, "x0"), ("stage10-q", 10, 254))


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    Ly = prob.layout
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    for name, t, anchor in CASES:
        S = x0.copy() if anchor == "x0" else np.asarray(
            cuts[t][anchor].state, float)
        next_cuts = (np.array([c.alpha for c in cuts[t + 1]]),
                     np.vstack([c.beta for c in cuts[t + 1]])) \
            if t < prob.T and cuts[t + 1] else \
            (np.array([]), np.empty((0, prob.state_dim)))
        print(f"\n{name}  stage={t}  scenes={len(prob.pool)}")
        for tl in BUDGETS:
            sp = StageSubproblem(t, prob, scfg)
            vals, gaps, sts = [], [], []
            t0 = time.time()
            try:
                sp.assemble(next_cuts, key=None)
                for omega in prob.pool:
                    sp.set_point(S, omega, require_int=True)
                    for j, u in zip(qidx, ub):
                        sp.model.setInfo(COPT.Info.LB, [sp.x_in[j]], 0.0)
                        sp.model.setInfo(COPT.Info.UB, [sp.x_in[j]], float(u))
                    info = ce.solve_bound(
                        sp.model, time_limit=tl, rel_gap=0.0,
                        abs_gap=0.0, value="best_bnd", return_info=True)
                    vals.append(info["best_bnd"])
                    gaps.append(info["best_obj"] - info["best_bnd"])
                    sts.append(info["status"])
            finally:
                sp.free()
            print(f"  tl={tl:4.1f}s  L_R={np.mean(vals):10.3f}  "
                  f"gap_mean={np.mean(gaps):9.3e}  "
                  f"status={sorted(set(sts))}  wall={time.time()-t0:6.1f}s")
    print("\nOK _t_nogood_timebudget")


if __name__ == "__main__":
    main()
