# _t_cutlift_counterexample.py — 固定状态 MIP−LP 差距与截距合法性对照。
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
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    t = 1
    anchor = sorted([c for c in cuts[t] if c.it > 0],
                    key=lambda c: c.it)[-1]
    x = np.asarray(anchor.state, float)
    next_cuts = (np.array([c.alpha for c in cuts[t + 1]], float),
                 np.vstack([c.beta for c in cuts[t + 1]]))
    rng = np.random.default_rng(2026)
    probes = []
    for h in range(20):
        xp = x.copy()
        xp[:prob.N_R] = 1.0
        xp[prob.layout.off_q:prob.layout.off_q + prob.N_R] = 1.0
        for r in range(prob.N_R):
            a = rng.dirichlet(np.ones(prob.G)) * prob.N[r]
            xp[prob.layout.ix_n(r, 0):prob.layout.ix_n(r, 0) + prob.G] = a
        probes.append(xp)
    sp = StageSubproblem(t, prob, scfg)
    try:
        sp.assemble(next_cuts, key=None)
        sp.set_point(x, prob.pool[0], require_int=True)
        sp.solve_lp()
        lp = float(sp.obj)
        beta = np.asarray(sp.gradient("dj"), float)
        sp.solve_mip()
        mip = float(sp.obj)
        delta = mip - lp
        print(f"anchor it={anchor.it}  Q_LP={lp:.6f}  Q_MIP={mip:.6f}  "
              f"gap={delta:.6f}")

        # 1) 固定状态 MIP 值作为截距，检查分数探针
        alpha_mip = mip - float(beta @ x)
        worst_mip = 1e99
        for xp in probes:
            sp.set_state(xp, require_int=False)
            sp.set_scene(prob.pool[0])
            sp.solve_lp()
            q = float(sp.stage_cost())
            worst_mip = min(worst_mip, alpha_mip + float(beta @ xp) - q)
        print(f"fixed-state MIP intercept: min(cut−Q_LP)={worst_mip:.6f}"
              f"  {'INVALID' if worst_mip > 1e-6 else 'valid on probes'}")

        # 2) Lagrangian free-state 截距
        nu = sp.solve_lift(beta, time_limit=1.0, rel_gap=1e-3,
                           safety_margin=0.0)
        gain = None if nu is None else float(nu + beta @ x - lp)
        worst_lag = 1e99
        if nu is not None:
            for xp in probes:
                sp.set_state(xp, require_int=False)
                sp.set_scene(prob.pool[0])
                sp.solve_lp()
                q = float(sp.stage_cost())
                worst_lag = min(worst_lag, nu + float(beta @ xp) - q)
        print(f"Lagrangian free-state: gain_at_anchor={gain}  "
              f"min(cut−Q_LP)={worst_lag:.6f}"
              f"  {'INVALID' if worst_lag > 1e-6 else 'valid on probes'}")

        # 3) 从 LP 截距向 MIP 截距插值，找分数探针上仍合法的最大 θ
        alpha_lp = lp - float(beta @ x)
        best_theta = 0.0
        best_gain = 0.0
        for th in np.linspace(0.0, 1.0, 101):
            alpha = alpha_lp + th * delta
            ok = True
            for xp in probes:
                sp.set_state(xp, require_int=False)
                sp.set_scene(prob.pool[0])
                sp.solve_lp()
                q = float(sp.stage_cost())
                if alpha + float(beta @ xp) > q + 1e-6:
                    ok = False
                    break
            if ok:
                best_theta = th
                best_gain = th * delta
            else:
                break
        print(f"interpolated intercept: max valid theta={best_theta:.2f}  "
              f"valid gain={best_gain:.6f}")
    finally:
        sp.free()
    print("OK _t_cutlift_counterexample")


if __name__ == "__main__":
    main()
