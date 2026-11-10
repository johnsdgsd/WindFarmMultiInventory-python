# _t_async_cutlift_valid.py — 抽查异步增强割：在锚点处不高于全场景整数真值。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

CFG = ROOT / "test" / "_cfg_smoke_tier2_cutlift"
RUN = ROOT / "results" / "smoke_tier2_cutlift_20261110_105656"
TOL = 1e-4


def main():
    prob = DefineProblem(load_problem_config(CFG / "problem.toml"))
    cfg = load_solver_config(CFG / "solver.toml")
    cuts, _ = load_cuts(RUN, prob.T, prob.state_dim)
    cut = next(c for c in cuts[prob.T] if c.it > 0)
    x = np.asarray(cut.state, dtype=float)
    cut_val = float(cut.alpha + cut.beta @ x)

    sub = make_stage_subproblem(prob.T, prob, cfg)
    try:
        sub.assemble(None)
        sub.set_state(x, require_int=True)
        vals = []
        for scene in prob.pool:
            sub.set_scene(scene)
            sub.solve_mip()
            vals.append(float(sub.obj))
        q_int = float(np.mean(vals))
    finally:
        sub.free()

    print(f"stage={cut.stage} cut={cut_val:.6f} int_mean={q_int:.6f} "
          f"cut-int={cut_val - q_int:.6e}")
    assert cut_val <= q_int + TOL, (cut_val, q_int)
    print("OK _t_async_cutlift_valid")


if __name__ == "__main__":
    main()
