# _t_nogood_globalL.py — 全 40 维状态的场景分解下界。
#
# L_global = mean_omega min_{S in state box} Q_1(S, omega)
# 每个场景独立释放全部 incoming state，解固定阶段 MIP，取 BestBnd。
# 同时计算只释放 q 的区域下界 L_q，比较两者。
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
TIME_LIMIT = 1.0


def _run(prob, scfg, cuts, x0, release_all: bool):
    Ly = prob.layout
    ub = np.asarray(Ly.upper(prob), float)
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
    vals, gaps, statuses, times = [], [], [], []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(x0, omega, require_int=True)
            if release_all:
                sp.model.setInfo(COPT.Info.LB, sp.x_in, [0.0] * Ly.dim)
                sp.model.setInfo(COPT.Info.UB, sp.x_in,
                                 [float(v) for v in ub])
            else:
                for j in qidx:
                    sp.model.setInfo(COPT.Info.LB, [sp.x_in[j]], 0.0)
                    sp.model.setInfo(COPT.Info.UB, [sp.x_in[j]], float(ub[j]))
            t0 = time.time()
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            times.append(time.time() - t0)
            if info is None or info.get("best_bnd") is None:
                statuses.append("FAIL")
                continue
            vals.append(info["best_bnd"])
            gaps.append(info["best_obj"] - info["best_bnd"])
            statuses.append(info["status"])
    finally:
        sp.free()
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "gap": float(np.mean(gaps)),
        "statuses": sorted(set(statuses)),
        "time": float(np.sum(times)),
        "n": len(vals),
    }


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    for name, all_ in (("L_global(all 40)", True), ("L_q(region)", False)):
        r = _run(prob, scfg, cuts, x0, all_)
        print(f"{name}: mean={r['mean']:.6f} min={r['min']:.6f} "
              f"max={r['max']:.6f} gap={r['gap']:.3e} "
              f"status={r['statuses']} n={r['n']} time={r['time']:.1f}s")
    print("OK _t_nogood_globalL")


if __name__ == "__main__":
    main()
