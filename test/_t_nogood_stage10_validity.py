# _t_nogood_stage10_validity.py — 阶段 10 q 区域割的随机 q 合法性抽样。
from __future__ import annotations

import sys
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
STAGE = 10
ANCHOR = 254
TIME_LIMIT = 0.5
N_RANDOM = 10


def _mean_fixed(prob, scfg, cuts, S):
    next_cuts = (np.array([c.alpha for c in cuts[STAGE + 1]]),
                 np.vstack([c.beta for c in cuts[STAGE + 1]]))
    sp = StageSubproblem(STAGE, prob, scfg)
    vals, sts = [], []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(S, omega, require_int=True)
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            vals.append(info["best_bnd"])
            sts.append(info["status"])
    finally:
        sp.free()
    return float(np.mean(vals)), sorted(set(sts))


def _L_region(prob, scfg, cuts, S):
    Ly = prob.layout
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    next_cuts = (np.array([c.alpha for c in cuts[STAGE + 1]]),
                 np.vstack([c.beta for c in cuts[STAGE + 1]]))
    sp = StageSubproblem(STAGE, prob, scfg)
    vals = []
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
    finally:
        sp.free()
    return float(np.mean(vals))


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    Ly = prob.layout
    c = cuts[STAGE][ANCHOR]
    S0 = np.asarray(c.state, float)
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    anchor = np.asarray(S0[Ly.q_slice()], int)
    L = _L_region(prob, scfg, cuts, S0)
    ell, st = _mean_fixed(prob, scfg, cuts, S0)
    print(f"stage={STAGE} anchor={ANCHOR} q={anchor.tolist()} "
          f"L_R={L:.6f} ell={ell:.6f} status={st}")
    qs = [anchor]
    rng = np.random.default_rng(20261109)
    while len(qs) < 1 + N_RANDOM:
        q = np.array([rng.integers(0, int(u) + 1) for u in ub], dtype=int)
        if not any(np.array_equal(q, z) for z in qs):
            qs.append(q)
    print("q                    Q(q)         L<=Q  binary  bin<=Q  hamming ham<=Q")
    for q in qs:
        S = S0.copy()
        S[Ly.q_slice()] = q
        Q, st = _mean_fixed(prob, scfg, cuts, S)
        d = int(np.sum(q != anchor))
        bc = ell if d == 0 else L
        hc = ell - (ell - L) * d
        print(f"{str(q.tolist()):20s} {Q:11.3f} {L <= Q + 1e-6!s:5s} "
              f"{bc:7.3f} {bc <= Q + 1e-6!s:5s} {hc:7.3f} "
              f"{hc <= Q + 1e-6!s:5s}")
        assert L <= Q + 1e-6 and bc <= Q + 1e-6 and hc <= Q + 1e-6
    print("OK _t_nogood_stage10_validity")


if __name__ == "__main__":
    main()
