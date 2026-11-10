# Benchmark Python Strengthened/lift MIP time on the same Tier2 states as Julia.
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402

CFG = ROOT / "test" / "_cfg_tier2_cutpool_2h"
CASES = json.loads((ROOT / "test" / "_lift_bench_states.json").read_text())


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def main():
    prob = DefineProblem(load_problem_config(CFG / "problem.toml"))
    cfg = load_solver_config(CFG / "solver.toml")
    cfg.n_processes = 1
    print(f"T={prob.T} pool={len(prob.pool)} scope={cfg.integer_scope}")
    print("stage scene lp_s mip_warm_s lift_s total_s lift_ok")
    for case in CASES:
        t, s = int(case["stage"]), int(case["scene"])
        x = np.asarray(case["state"], dtype=float)
        lps, warms, lifts, totals = [], [], [], []
        ok = []
        for rep in range(3):
            sub = make_stage_subproblem(t, prob, cfg)
            # no future cuts: same as Julia node without cuts
            sub.assemble((np.zeros(0), np.zeros((0, prob.state_dim))),
                         key=("bench", t))
            sub.set_point(x, prob.pool[s])
            t0 = time.perf_counter()
            sub.solve_lp()
            t1 = time.perf_counter()
            beta = np.asarray(sub.gradient("dj"), dtype=float)
            sub.solve_mip()  # warm start for released-state MIP
            t2 = time.perf_counter()
            nu = sub.solve_lift(
                beta, time_limit=5.0, rel_gap=0.0, abs_gap=0.0,
                safety_margin=1e-6, value="exact")
            t3 = time.perf_counter()
            lps.append(t1 - t0)
            warms.append(t2 - t1)
            lifts.append(t3 - t2)
            totals.append(t3 - t0)
            ok.append(nu is not None)
            sub.free()
        print(f"{t:2d} {s:4d} {med(lps):.4f} {med(warms):.4f} "
              f"{med(lifts):.4f} {med(totals):.4f} {ok}")
    print("PY_BENCH_DONE")


if __name__ == "__main__":
    main()
