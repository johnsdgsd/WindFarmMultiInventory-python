# _t_async_tier2_smoke.py — 异步引擎 + Tier2 前向 MIP/后向 LP 的冒烟。
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.async_engine import AsyncIterationEngine              # noqa: E402


def main():
    cdir = ROOT / "config" / "smoke_tier2"
    prob = DefineProblem(load_problem_config(cdir / "problem.toml"))
    cfg = load_solver_config(cdir / "solver.toml")
    cfg.cut_selection = "julia"
    cfg.n_processes = 2
    print(f"Tier2 async smoke T={prob.T} pool={len(prob.pool)} "
          f"integer={cfg.integer_scope}")
    eng = AsyncIterationEngine(prob, make_stage_subproblem, cfg,
                               n_workers=2, paths_per_task=1, log=print)
    hist = eng.train(5, max_seconds=180, verbose=True)
    print(f"done iterations={len(hist)} solves={eng.n_solves} "
          f"lb_best={eng.lb_best:.6f} "
          f"cuts={sum(len(c) for c in eng.cuts[1:])}")
    assert hist
    assert eng.lb_best > 0
    eng.close()
    print("ASYNC_TIER2_SMOKE_OK")


if __name__ == "__main__":
    main()
