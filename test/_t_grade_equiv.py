# _t_grade_equiv.py — 等级模型单阶段子问题自检（纯 LP）：
#   N=Θn、n_out 列和、dem、fleet/W、F_new；.dj 与 duals 一致；
#   场景切换（A→B→A）无系数残留；check_stage 审计通过；
#   fleet/dem 右端 = N+fleet_tol，容差边界真实生效。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config     # noqa: E402
from code.problem_data import DefineProblem                          # noqa: E402
from code.subproblem import StageSubproblem, round_repair_n            # noqa: E402
from code.sim_analysis import (CHECKS, STATE_NUMERIC_CHECKS,          # noqa: E402
                               check_stage)
from utils import copt_env as ce                                      # noqa: E402

TOL = 1e-6


def _N_matrix(sp, Ly) -> np.ndarray:
    out = np.zeros((Ly.N_R, Ly.G, Ly.G + 1))
    for r in range(Ly.N_R):
        for i in range(Ly.G):
            for j in range(i, Ly.G + 1):
                out[r, i, j] = float(ce.values(sp.model, sp.N[r][i][j]))
    return out


def main() -> None:
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, G = prob.layout, prob.G
    sp = StageSubproblem(1, prob, scfg)
    x = prob.initial_state()
    n_in = Ly.n(x)
    ub = Ly.upper(prob)

    # fleet/dem 右端必须与模型口径一致：N_r + fleet_tol
    for r in range(Ly.N_R):
        for name in (f"fleet_{r}", f"dem_{r}"):
            row = sp._row(name)
            rhs = float(sp.model.getInfo(COPT.Info.UB, [row])[0])
            assert abs(rhs - (float(prob.N[r]) + prob.fleet_tol)) < 1e-12, (name, rhs)

    for si in (0, 1, 0):
        scene = prob.pool[si]
        th = np.asarray(scene[1], float)
        sp.set_point(x, scene)
        sp.solve_lp()
        N = _N_matrix(sp, Ly)
        assert np.max(np.abs(N - th * n_in[:, :, None])) < 1e-7
        assert np.max(np.abs(N.sum(axis=2) - n_in)) < 1e-7
        xo = sp.out_state_vector()
        n_out = Ly.n(xo)
        rec = sp.decision_values()
        assert np.max(np.abs(n_out - np.asarray(rec["n_out"]))) < 1e-7
        F = np.asarray(rec["F_new"], float)
        assert np.max(np.abs(F - N[:, :, G].sum(axis=1))) < 1e-7
        W_in = np.asarray(rec["W_in"], float)
        W_out = np.asarray(rec["W_out"], float)
        assert np.max(np.abs(W_in - (prob.N - n_in.sum(axis=1)))) < 1e-7
        assert np.min(W_in) >= -TOL and np.min(W_out) >= -TOL

        g1 = np.asarray(sp.gradient("dj"), float)
        g2 = np.asarray(sp.gradient("duals"), float)
        assert np.max(np.abs(g1 - g2)) < 1e-6, np.max(np.abs(g1 - g2))

        d = check_stage(prob, Ly, x, xo, rec["q"], rec["taup"], rec["taum"],
                        rec["S_end"], rec["y"], si, ub, F_new=rec["F_new"])
        state_tol = max(TOL, float(prob.state_repair_tol))
        viol = {k: v for k, v in d.items()
                if v > (state_tol if k in STATE_NUMERIC_CHECKS else TOL)}
        assert not viol, viol

    # fleet_tol 边界：N+tol/2 可行；N+10·tol 必须不可行（容差不是无限放行）
    tol = float(prob.fleet_tol)
    x_plus = x.copy()
    x_plus[Ly.ix_n(0, 0)] += tol / 2.0
    sp.set_point(x_plus, prob.pool[0])
    sp.solve_lp()
    xo_plus = sp.out_state_vector()
    assert Ly.n(xo_plus).sum(axis=1)[0] <= prob.N[0] + tol + 1e-9
    print(f"fleet_tol 内可行: Σn(仓1) = {Ly.n(xo_plus).sum(axis=1)[0]:.12f} "
          f"≤ N+fleet_tol = {prob.N[0] + tol:.12f}")

    # round + 小超额修复：复现用户报告的 N+1.114e-6
    n_rep = np.zeros((Ly.N_R, Ly.G))
    n_rep[0, 0] = float(prob.N[0]) + 1.114e-6
    n_fix = round_repair_n(n_rep, prob.N, prob.state_round_digits,
                           prob.state_repair_tol)
    assert abs(float(n_fix[0].sum()) - float(prob.N[0])) <= 1e-9
    assert np.all(n_fix >= 0.0)
    print(f"round+repair: Σn(仓1) {n_rep[0].sum():.12f} → {n_fix[0].sum():.12f}")

    n_bad_rep = n_rep.copy()
    n_bad_rep[0, 0] = float(prob.N[0]) + 10.0 * prob.state_repair_tol
    try:
        round_repair_n(n_bad_rep, prob.N, prob.state_round_digits,
                       prob.state_repair_tol)
    except ValueError as exc:
        print(f"超过 state_repair_tol 不修（预期）: {type(exc).__name__}")
    else:
        raise AssertionError("超过 state_repair_tol 的超额居然被修掉了")

    x_bad = x.copy()
    x_bad[Ly.ix_n(0, 0)] += 10.0 * tol
    sp.set_point(x_bad, prob.pool[0])
    try:
        sp.solve_lp()
    except ce.SolverError as exc:
        print(f"fleet_tol 外不可行（预期）: {type(exc).__name__}")
    else:
        raise AssertionError(f"Σn 超 N+10·fleet_tol 的子问题居然可行")

    print(f"scene switch A→B→A: N=Θn / n_out / dem / fleet / F_new / W 全部通过")
    print(f".dj vs duals: max|Δ| < 1e-6；check_stage: {len(CHECKS)} 项通过")
    print(f"state_dim={prob.state_dim}  N变量={sum(1 for r in sp.N for i in r for v in i if v is not None)}")
    print("OK _t_grade_equiv")


if __name__ == "__main__":
    main()
