# _t_nogood_validity.py — q 区域 no-good 割在随机整数 q 上的合法性抽样。
#
# 区域 R: 非 q 分量固定为 stage0 初始状态，q 可变。
# L_q 由每场景 min_q Q_1(q,omega) 的平均得到；逐点 Q(q) 用固定状态 MIP 检查。
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
TIME_LIMIT = 0.5
N_RANDOM = 10


def _mean_fixed(prob, scfg, cuts, S):
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
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


def _L_region(prob, scfg, cuts, x0):
    Ly = prob.layout
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
    vals = []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(x0, omega, require_int=True)
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
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    empty = np.zeros(Ly.n_q)
    q0 = x0[Ly.q_slice()]
    qs = [np.asarray(q0, int)]
    rng = np.random.default_rng(20261108)
    while len(qs) < 1 + N_RANDOM:
        q = np.array([rng.integers(0, int(u) + 1) for u in ub], dtype=int)
        if not any(np.array_equal(q, z) for z in qs):
            qs.append(q)

    Lq = _L_region(prob, scfg, cuts, x0)
    print(f"L_q={Lq:.6f}")
    # anchor = 初始 q；ell 单独算
    anchor = qs[0]
    S0 = x0.copy()
    S0[Ly.q_slice()] = anchor
    ell, st = _mean_fixed(prob, scfg, cuts, S0)
    print(f"anchor={anchor.tolist()} ell={ell:.6f} status={st}")
    print("q                    Q(q)         L<=Q   binary  binary<=Q  "
          "hamming  hamming<=Q")
    for q in qs:
        S = x0.copy()
        S[Ly.q_slice()] = q
        Q, st = _mean_fixed(prob, scfg, cuts, S)
        d = int(np.sum(q != anchor))
        bin_cut = ell if d == 0 else Lq
        ham_cut = ell - (ell - Lq) * d
        okL = Lq <= Q + 1e-6
        okB = bin_cut <= Q + 1e-6
        okH = ham_cut <= Q + 1e-6
        print(f"{str(q.tolist()):20s} {Q:11.3f}  {okL!s:5s} "
              f"{bin_cut:9.3f} {okB!s:10s} {ham_cut:9.3f} {okH!s:10s}")
        assert okL and okB and okH
    print("OK _t_nogood_validity")


if __name__ == "__main__":
    main()
