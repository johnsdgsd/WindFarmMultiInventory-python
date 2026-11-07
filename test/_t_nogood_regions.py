# _t_nogood_regions.py — 多阶段 q 区域下界 L_R 与当前割包络比较。
#
# 固定阶段 t 的非 q 状态，只释放 q，逐场景解固定阶段 MIP：
#   L_R = mean_omega min_q Q_t(q, omega)
# 注意：这是 q 区域下界，不是全 40 维全局下界。
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
STAGES = (10, 20, 30, 45, 60)
TIME_LIMIT = 0.5


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    Ly = prob.layout
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    print("stage  anchor_it  V_cut       L_R         delta     status  time")
    for t in STAGES:
        cs = [c for c in cuts[t] if c.it == max(x.it for x in cuts[t])]
        c = cs[0]
        S = np.asarray(c.state, float)
        v_cut = float(max(z.alpha + z.beta @ S for z in cuts[t]))
        next_cuts = (np.array([z.alpha for z in cuts[t + 1]]),
                     np.vstack([z.beta for z in cuts[t + 1]])) \
            if t < prob.T and cuts[t + 1] else \
            (np.array([]), np.empty((0, prob.state_dim)))
        sp = StageSubproblem(t, prob, scfg)
        vals, statuses = [], []
        t0 = time.time()
        try:
            sp.assemble(next_cuts, key=None)
            for omega in prob.pool:
                sp.set_point(S, omega, require_int=True)
                for j, u in zip(qidx, ub):
                    sp.model.setInfo(COPT.Info.LB, [sp.x_in[j]], 0.0)
                    sp.model.setInfo(COPT.Info.UB, [sp.x_in[j]], float(u))
                info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                      rel_gap=0.0, abs_gap=0.0,
                                      value="best_bnd", return_info=True)
                vals.append(info["best_bnd"])
                statuses.append(info["status"])
        finally:
            sp.free()
        print(f"{t:5d} {c.it:9d} {v_cut:10.3f} {np.mean(vals):10.3f} "
              f"{np.mean(vals)-v_cut:9.3f} {sorted(set(statuses))} "
              f"{time.time()-t0:6.1f}s")
    print("OK _t_nogood_regions")


if __name__ == "__main__":
    main()
