# _t_integer_cutvalid.py — 整数模式的割由后向 LP 松弛生成，在分数状态上仍合法。
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
    pcfg = load_problem_config(ROOT / "config" / "smoke_integer" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_integer" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, G, NR = prob.layout, prob.G, prob.N_R
    ub = Ly.upper(prob)
    x0 = np.asarray(prob.initial_state(), float)
    sp = StageSubproblem(1, prob, scfg)
    rng = np.random.default_rng(7)
    try:
        sp.assemble(None, key=None)
        sp.set_state(x0, require_int=True)
        sp.set_scene(prob.pool[0])
        sp.solve_lp()          # 后向用 LP 松弛，即使在整数模式
        beta = np.asarray(sp.gradient("dj"), float)
        q0 = float(sp.stage_cost())
        alpha = q0 - float(beta @ x0)
        worst = np.inf
        nchk = 0
        for _ in range(12):
            xp = x0.copy()
            for r in range(NR):
                xp[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + G] = (
                    rng.dirichlet(np.ones(G)) * float(prob.N[r]))
            for j in range(Ly.dim):
                if Ly.off_n <= j < Ly.off_n + Ly.n_n:
                    continue
                xp[j] = min(max(xp[j] + rng.normal(0, 1), 0.0), float(ub[j]))
            sp.set_state(xp, require_int=False)
            sp.set_scene(prob.pool[0])
            sp.solve_lp()
            q = float(sp.stage_cost())
            cut = float(alpha + beta @ xp)
            worst = min(worst, cut - q)
            assert cut <= q + TOL, (cut, q, cut - q)
            nchk += 1
        print(f"  [PASS] {nchk} 个分数探针：min(cut−Q)={worst:.3e}")
    finally:
        sp.free()
    print("OK _t_integer_cutvalid")


if __name__ == "__main__":
    main()
