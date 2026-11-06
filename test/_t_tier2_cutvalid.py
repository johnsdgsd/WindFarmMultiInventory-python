# _t_tier2_cutvalid.py — Tier2 后向 LP 割在分数状态/边界探针上仍为合法下界。
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
    rng = np.random.default_rng(19)
    x = np.asarray(prob.initial_state(), float).copy()
    x[Ly.off_I] = 4.0
    x[Ly.off_q] = 1.0
    n = np.zeros((prob.N_R, prob.G), dtype=int)
    for r in range(prob.N_R):
        cuts = np.sort(rng.choice(np.arange(1, int(prob.N[r])),
                                  size=prob.G - 1, replace=False))
        n[r] = np.diff(np.r_[0, cuts, int(prob.N[r])])
    x[Ly.off_n:Ly.off_n + Ly.n_n] = n.ravel()
    sp = StageSubproblem(1, prob, scfg)
    try:
        sp.assemble(None, key=None)
        sp.set_point(x, prob.pool[5], require_int=True)
        sp.solve_lp()
        beta = np.asarray(sp.gradient("dj"), float)
        q0 = float(sp.stage_cost())
        alpha = q0 - float(beta @ x)
        dual = np.asarray(sp.gradient("duals"), float)
        assert np.max(np.abs(beta - dual)) < 1e-5, np.max(np.abs(beta - dual))

        worst = np.inf
        nchk = 0
        for k in range(10):
            xp = x.copy()
            # 只扰动 n（分数）；I/o/q 保持小的可行值，避免随机状态本身
            # 使总库存容量不可行，掩盖割合法性检验。
            xp[Ly.off_I:Ly.off_I + Ly.n_I] = 1.0
            xp[Ly.off_q:Ly.off_q + Ly.n_q] = 1.0
            for r in range(prob.N_R):
                frac = rng.dirichlet(np.ones(prob.G)) * float(prob.N[r])
                xp[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + prob.G] = frac
            sp.set_state(xp, require_int=False)
            sp.set_scene(prob.pool[(7 * k) % len(prob.pool)])
            sp.solve_lp()
            q = float(sp.stage_cost())
            cut = float(alpha + beta @ xp)
            worst = min(worst, cut - q)
            assert cut <= q + TOL, (k, cut, q, cut - q)
            nchk += 1
        print(f"  [PASS] {nchk} 个分数探针：min(cut−Q)={worst:.3e}；"
              f".dj=duals 最大差 {np.max(np.abs(beta-dual)):.2e}")
    finally:
        sp.free()
    print("OK _t_tier2_cutvalid")


if __name__ == "__main__":
    main()
