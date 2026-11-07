# _t_cutlift.py — Lagrangian 割提升：截距合法、模型恢复、分数/整数状态验证。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402

TOL = 1e-5


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    rng = np.random.default_rng(31)
    x = np.asarray(prob.initial_state(), dtype=float).copy()
    x[Ly.off_I] = 4.0
    x[Ly.off_q] = 1.0
    n = np.zeros((prob.N_R, prob.G), dtype=int)
    for r in range(prob.N_R):
        cuts = np.sort(rng.choice(np.arange(1, int(prob.N[r])),
                                  size=prob.G - 1, replace=False))
        n[r] = np.diff(np.r_[0, cuts, int(prob.N[r])])
    x[Ly.off_n:Ly.off_n + Ly.n_n] = n.ravel()
    scene = prob.pool[5]
    sp = StageSubproblem(1, prob, scfg)
    try:
        sp.assemble(None, key=None)
        sp.set_point(x, scene, require_int=True)
        sp.solve_lp()
        lp0 = float(sp.obj)
        beta = np.asarray(sp.gradient("dj"), dtype=float)
        # 对标 SDDP.jl StrengthenedConicDuality：exact 口径必须 BestObj≈BestBnd。
        nu = sp.solve_lift(beta, time_limit=2.0, rel_gap=0.0, value="exact")
        assert nu is not None, "smoke lifting 没有返回精确 MIP 值"
        info = sp._last_lift_info
        assert info is not None and info["status"] == 1
        assert info["exact_ok"], info
        assert abs(info["best_obj"] - info["best_bnd"]) <= 1e-7 * max(
            1.0, abs(info["best_obj"]))
        # 恢复：再解 LP 应回到原值和原斜率
        sp.solve_lp()
        assert abs(float(sp.obj) - lp0) < 1e-6
        assert np.max(np.abs(np.asarray(sp.gradient("dj")) - beta)) < 1e-6
        # 整数锚点：lift cut 是合法下界，且不应弱于标准 LP 割
        sp.solve_mip()
        qmip = float(sp.obj)
        cut0 = nu + float(beta @ x)
        assert cut0 <= qmip + TOL, (cut0, qmip)
        assert cut0 >= lp0 - TOL, (cut0, lp0)
        # 分数状态：strengthened cut 可能高于 Q_LP（工具包只在整数可行域使用），
        # 这里只记录，不要求分数点合法。
        worst = 1e9
        for k in range(12):
            xp = x.copy()
            xp[Ly.off_I:Ly.off_I + Ly.n_I] = 1.0
            xp[Ly.off_q:Ly.off_q + Ly.n_q] = 1.0
            for r in range(prob.N_R):
                xp[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + prob.G] = (
                    rng.dirichlet(np.ones(prob.G)) * prob.N[r])
            sp.set_state(xp, require_int=False)
            sp.set_scene(scene)
            sp.solve_lp()
            q = float(sp.stage_cost())
            cut = nu + float(beta @ xp)
            worst = min(worst, cut - q)
        print(f"  [PASS] exact MIP BestObj={info['best_obj']:.6f} "
              f"BestBnd={info['best_bnd']:.6f}；"
              f"lift cut ≤ 固定状态 MIP {qmip:.4f}；"
              f"12 分数探针 min(cut−LP)={worst:.3e}（仅记录，MIP LB 不受影响）")
    finally:
        sp.free()
    print("OK _t_cutlift")


if __name__ == "__main__":
    main()
