# _t_async_smoke.py — AsyncIterationEngine 冒烟：2 worker、冒烟实例、5 个 iteration。
from __future__ import annotations

import os
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
    cdir = ROOT / "config" / "smoke"
    prob = DefineProblem(load_problem_config(cdir / "problem.toml"))
    cfg = load_solver_config(cdir / "solver.toml")
    cfg.cut_selection = os.environ.get("ASYNC_CUT_SELECTION", "none")
    cfg.n_processes = 2
    print(f"smoke T={prob.T} pool={len(prob.pool)} workers={cfg.n_processes}")
    eng = AsyncIterationEngine(prob, make_stage_subproblem, cfg,
                               n_workers=2, log=print)

    def on_iter(rec):
        vals = [c.value_at(eng.initial_state) for c in eng.cuts[1]]
        print(f"    [debug] stage1 cuts={len(vals)} "
              f"max@x0={max(vals) if vals else None:.3f}")

    eng.on_iter = on_iter
    hist = eng.train(5, max_seconds=120, verbose=True)
    print(f"done iterations={len(hist)} solves={eng.n_solves} "
          f"lb_best={eng.lb_best:.6f} cuts={sum(len(c) for c in eng.cuts[1:])}")
    assert hist, "没有完成任何 iteration"
    assert eng.lb_best > 0, "历史最大 LB 没有严格转正"
    n_cuts = sum(len(c) for c in eng.cuts[1:])
    if cfg.cut_selection == "none":
        assert n_cuts >= 1 + 5 * prob.T
    else:
        assert n_cuts >= 1 + prob.T   # LML1 会删割，不能按“每轮全留”断言
    eng.close()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
