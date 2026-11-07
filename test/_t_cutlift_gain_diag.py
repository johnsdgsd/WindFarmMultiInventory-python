# _t_cutlift_gain_diag.py — 只读诊断：在已有 Tier2 cuts 上测 Lagrangian 截距提升。
# 不训练、不写 results；输出每个 stage/anchor 的 lift_gain 分布。
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
STAGES = (1, 10, 30, 60)
N_ANCHORS = 3
N_SCENES = 30


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    rng = np.random.default_rng(2026)
    scenes = rng.choice(len(prob.pool), size=N_SCENES, replace=False).tolist()
    print(f"result={RESULT.name}  pool={len(prob.pool)}  scenes={N_SCENES}  "
          f"anchors/stage={N_ANCHORS}")
    print("stage anchor_it   lp_mean    lift_mean   lift_med   lift_max  "
          "positive  fallback")
    all_gains = []
    for t in STAGES:
        # 取该阶段最新的、非平凡割作为锚点
        cs = [c for c in cuts[t] if c.it > 0]
        cs = sorted(cs, key=lambda c: c.it, reverse=True)[:N_ANCHORS]
        if not cs:
            print(f"{t:5d} (no nontrivial cuts)")
            continue
        next_cuts = None
        if t < prob.T and cuts[t + 1]:
            next_cuts = (np.array([c.alpha for c in cuts[t + 1]], float),
                         np.vstack([c.beta for c in cuts[t + 1]]))
        for c in cs:
            x = np.asarray(c.state, float)
            sp = StageSubproblem(t, prob, scfg)
            gains, lps, fb = [], [], 0
            try:
                sp.assemble(next_cuts, key=None)
                sp.set_state(x, require_int=True)
                for si in scenes:
                    sp.set_scene(prob.pool[si])
                    sp.solve_lp()
                    lp = float(sp.obj)
                    beta = np.asarray(sp.gradient("dj"), float)
                    nu = sp.solve_lift(beta, time_limit=1.0,
                                       rel_gap=1e-3, safety_margin=0.0)
                    if nu is None:
                        fb += 1
                        continue
                    gains.append(float(nu + beta @ x - lp))
                    lps.append(lp)
            finally:
                sp.free()
            if gains:
                a = np.asarray(gains, float)
                all_gains.extend(gains)
                print(f"{t:5d} {c.it:8d} {np.mean(lps):10.3f} "
                      f"{a.mean():10.3e} {np.median(a):10.3e} "
                      f"{a.max():10.3e} {np.mean(a > 1e-7):8.1%} {fb:8d}")
            else:
                print(f"{t:5d} {c.it:8d} (no finite bound) fallback={fb}")
    a = np.asarray(all_gains, float)
    if a.size:
        print(f"ALL n={a.size} mean={a.mean():.6e} median={np.median(a):.6e} "
              f"max={a.max():.6e} positive={np.mean(a > 1e-7):.2%}")
    print("OK _t_cutlift_gain_diag")


if __name__ == "__main__":
    main()
