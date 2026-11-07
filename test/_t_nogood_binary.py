# _t_nogood_binary.py — 二元不匹配指示 No-good 割的 stage0 验证。
#
# 对比 Hamming 版本与二元 z 版本：
#   z=0 iff S=Sbar；z=1 otherwise
#   theta >= L + (ell-L) * (1-z)
# q 是 stage0 唯一自由的状态分量；L 是在 q 区域上有效的下界
#   L = mean_omega min_q Q_1(q, omega)
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from coptpy import COPT, LinExpr, quicksum

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


def _fixed_q(m, xout, qidx, qv):
    for j, v in zip(qidx, qv):
        m.setInfo(COPT.Info.LB, [xout[j]], float(v))
        m.setInfo(COPT.Info.UB, [xout[j]], float(v))


def _release_q(m, xin, qidx, ub):
    for j, u in zip(qidx, ub):
        m.setInfo(COPT.Info.LB, [xin[j]], 0.0)
        m.setInfo(COPT.Info.UB, [xin[j]], float(u))


def _ell_stage1(prob, scfg, cuts, S):
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
    vals, gaps = [], []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(S, omega, require_int=True)
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            vals.append(info["best_bnd"])
            gaps.append(info["best_obj"] - info["best_bnd"])
    finally:
        sp.free()
    return float(np.mean(vals)), max(gaps)


def _L_region_q(prob, scfg, cuts, x0):
    """L = mean_omega min_q Q_1(q,omega)；q 是 stage0 唯一自由状态分量。"""
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
            _release_q(sp.model, sp.x_in, qidx, ub)
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            vals.append(info["best_bnd"])
    finally:
        sp.free()
    return float(np.mean(vals)), float(np.min(vals))


def _stage0_with_binary_cut(prob, scfg, cuts, x0, q_anchor, ell, L,
                            *, onehot: bool):
    Ly = prob.layout
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    next_cuts = (np.array([c.alpha for c in cuts[1]]),
                 np.vstack([c.beta for c in cuts[1]]))
    sp = StageSubproblem(0, prob, scfg)
    sp.assemble(next_cuts, key=None)
    sp.set_state(x0, require_int=True)
    m = sp.model
    q = [sp.x_out[j] for j in qidx]
    d_terms = []
    if onehot:
        for r, (qv, u) in enumerate(zip(q_anchor, ub)):
            ys = [m.addVar(vtype=COPT.BINARY, name=f"y_{r}_{v}")
                  for v in range(int(u) + 1)]
            m.addConstr(quicksum(ys) == 1.0)
            m.addConstr(quicksum(v * ys[v] for v in range(int(u) + 1))
                        == q[r])
            d_terms.append(1.0 - ys[int(qv)])
    else:
        for r, (qv, u) in enumerate(zip(q_anchor, ub)):
            nb = int(np.ceil(np.log2(float(u) + 1.0)))
            bs = [m.addVar(vtype=COPT.BINARY, name=f"b_{r}_{k}")
                  for k in range(nb)]
            m.addConstr(quicksum((1 << k) * bs[k] for k in range(nb))
                        == q[r])
            for k, b in enumerate(bs):
                d_terms.append((1.0 - b) if (int(qv) >> k) & 1 else b)
    z = m.addVar(vtype=COPT.BINARY, name="z_mismatch")
    for d in d_terms:
        m.addConstr(z >= d, name="z_ge_d")
    m.addConstr(z <= quicksum(d_terms), name="z_le_sum_d")
    # theta >= L + (ell-L)(1-z)
    m.addConstr(sp._theta + (ell - L) * z >= ell, name="nogood_binary")
    info = ce.solve_bound(m, time_limit=60.0, rel_gap=0.0,
                          abs_gap=0.0, value="auto_strict",
                          exact_tol=1e-9, return_info=True)
    qv = [float(v) for v in ce.values(m, q)]
    sp.free()
    return info, qv


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    Ly = prob.layout
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))

    # 当前 stage0 的最优 q
    sp0 = StageSubproblem(0, prob, scfg)
    sp0.assemble((np.array([c.alpha for c in cuts[1]]),
                  np.vstack([c.beta for c in cuts[1]])), key=None)
    sp0.set_state(x0, require_int=True)
    info0 = ce.solve_bound(sp0.model, time_limit=60.0, rel_gap=0.0,
                           abs_gap=0.0, value="auto_strict",
                           exact_tol=1e-9, return_info=True)
    qroot = [int(round(v)) for v in ce.values(sp0.model,
                                             [sp0.x_out[j] for j in qidx])]
    sp0.free()
    print(f"stage0 base LB={info0['value']:.6f}  qroot={qroot}")

    L, Lmin = _L_region_q(prob, scfg, cuts, x0)
    print(f"L_region(q)={L:.6f}  per-scene min={Lmin:.6f}")

    # 历史锚点（阶段 1 最新割）
    c = [z for z in cuts[1] if z.it == max(y.it for y in cuts[1])][0]
    qa = [int(round(v)) for v in np.asarray(c.state, float)[Ly.q_slice()]]
    ell_a, ga = _ell_stage1(prob, scfg, cuts, np.asarray(c.state, float))
    print(f"history anchor q={qa}  ell={ell_a:.6f}  gap={ga:.3e}")

    # 当前 stage0 最优点
    Sroot = x0.copy()
    Sroot[Ly.q_slice()] = qroot
    ell_r, gr = _ell_stage1(prob, scfg, cuts, Sroot)
    print(f"root anchor q={qroot}  ell={ell_r:.6f}  gap={gr:.3e}")

    for name, qa_, ell_ in (("history", qa, ell_a), ("root", qroot, ell_r)):
        for onehot in (True, False):
            info, qv = _stage0_with_binary_cut(
                prob, scfg, cuts, x0, qa_, ell_, L, onehot=onehot)
            tag = "onehot" if onehot else "bit"
            print(f"{name:7s} {tag:6s} LB={info['value']:.6f} "
                  f"q={[int(round(v)) for v in qv]}")
    print("OK _t_nogood_binary")


if __name__ == "__main__":
    main()
