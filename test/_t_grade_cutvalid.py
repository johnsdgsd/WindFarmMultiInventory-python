# _t_grade_cutvalid.py — 纯 LP 等级子问题的割合法性抽检：
#   在锚点 x0 解同一场景的 LP，取 .dj 造割，再在扰动状态上解同一场景的 LP，
#   验证 cut(x) <= Q(x) + tol（Benders 割是 LP 值函数的下支撑）。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config     # noqa: E402
from code.problem_data import DefineProblem                          # noqa: E402
from code.subproblem import StageSubproblem                           # noqa: E402

TOL = 1e-6


def main() -> None:
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, G, NR = prob.layout, prob.G, prob.N_R
    ub = Ly.upper(prob)
    sp = StageSubproblem(1, prob, scfg)
    rng = np.random.default_rng(7)

    x0 = prob.initial_state()
    n_anchor = Ly.n(x0)
    worst = np.inf
    n_checks = 0
    for si in (0, 1, 2):
        x0 = np.minimum(np.maximum(x0, 0.0), ub)
        sp.set_point(x0, prob.pool[si])
        sp.solve_lp()
        beta = np.asarray(sp.gradient("dj"), float)
        obj0 = float(sp.stage_cost())
        alpha = obj0 - float(beta @ x0)

        for _ in range(12):
            # n：Dirichlet 按 N_r 切分（和恰为 N_r，fleet 可行）
            xp = x0.copy()
            for r in range(NR):
                w = rng.dirichlet(np.ones(G))
                xp[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + G] = w * float(prob.N[r])
            # I/o/q：小扰动并夹到界内
            for j in range(Ly.dim):
                if Ly.off_n <= j < Ly.off_n + Ly.n_n:
                    continue
                d = rng.normal(0.0, 1.0)
                xp[j] = min(max(xp[j] + d, 0.0), float(ub[j]))
            sp.set_point(xp, prob.pool[si])
            sp.solve_lp()
            q = float(sp.stage_cost())
            cut = float(alpha + beta @ xp)
            worst = min(worst, cut - q)
            n_checks += 1
            assert cut <= q + TOL, (si, cut, q, cut - q)

    print(f"割合法性抽检: {n_checks} 个（场景,扰动状态）点，"
          f"min(cut−Q) = {worst:.3e}（割是下界，应 ≤ +{TOL:g}）")
    print("OK _t_grade_cutvalid")


if __name__ == "__main__":
    main()
