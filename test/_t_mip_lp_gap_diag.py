# _t_mip_lp_gap_diag.py — 只读诊断：固定状态 Q_MIP - Q_LP（已有 Tier2 cuts）。
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
STAGES = (1, 10, 30)
N_ANCHORS = 2
N_SCENES = 10


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    rng = np.random.default_rng(2026)
    scenes = rng.choice(len(prob.pool), size=N_SCENES, replace=False).tolist()
    print(f"result={RESULT.name}  stages={STAGES}  scenes={N_SCENES}")
    print("stage anchor  n   mean_Qlp    mean_Qmip   mean_diff  max_diff  "
          "optimal/fallback")
    all_diff = []
    for t in STAGES:
        cs = [c for c in cuts[t] if c.it > 0]
        cs = sorted(cs, key=lambda c: c.it, reverse=True)[:N_ANCHORS]
        next_cuts = None
        if t < prob.T and cuts[t + 1]:
            next_cuts = (np.array([c.alpha for c in cuts[t + 1]], float),
                         np.vstack([c.beta for c in cuts[t + 1]]))
        for c in cs:
            x = np.asarray(c.state, float)
            sp = StageSubproblem(t, prob, scfg)
            lps, mips, diffs = [], [], []
            n_opt = n_fb = 0
            try:
                sp.assemble(next_cuts, key=None)
                sp.set_state(x, require_int=True)
                for si in scenes:
                    sp.set_scene(prob.pool[si])
                    sp.solve_lp()
                    lp = float(sp.obj)
                    sp.solve_mip()
                    mip = float(sp.obj)
                    if sp.model.status == 1:
                        n_opt += 1
                    else:
                        n_fb += 1
                    lps.append(lp)
                    mips.append(mip)
                    diffs.append(mip - lp)
                    all_diff.append(mip - lp)
            finally:
                sp.free()
            if diffs:
                a = np.asarray(diffs, float)
                print(f"{t:5d} {c.it:6d} {len(a):2d} "
                      f"{np.mean(lps):10.3f} {np.mean(mips):10.3f} "
                      f"{a.mean():10.3f} {a.max():10.3f} "
                      f"{n_opt}/{n_fb}")
    a = np.asarray(all_diff, float)
    print(f"ALL n={a.size} mean={a.mean():.6f} median={np.median(a):.6f} "
          f"max={a.max():.6f} positive={np.mean(a > 1e-6):.2%}")
    print("OK _t_mip_lp_gap_diag")


if __name__ == "__main__":
    main()
