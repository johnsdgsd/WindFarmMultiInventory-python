# _t_lb_mip_compare.py — 同一割集下比较阶段 0 的 LP 与整数 MIP 下界。
#
# 期望：MIP LB >= LP LB（LP 是 MIP 的松弛），且 MIP 解必须成功。
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils.async_engine import AsyncIterationEngine              # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/int_t2_formal_20261106_180256")
    args = ap.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    pcfg = load_problem_config(run / "config" / "problem.toml")
    scfg = load_solver_config(run / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, meta = load_cuts(run, prob.T, prob.state_dim)
    eng = AsyncIterationEngine(prob, make_stage_subproblem, scfg,
                               n_workers=1, paths_per_task=1)
    eng.cuts = list(cuts) + [[]]
    eng.n_iterations = int(meta.get("n_iterations", 0))
    try:
        lb_mip = eng._lower_bound()
        eng._lb_sub.solve_lp()
        lb_lp = float(eng._lb_sub.obj)
        print(f"cuts={sum(len(c) for c in eng.cuts[1:eng.T + 1])} "
              f"LP={lb_lp:.6f} MIP={lb_mip:.6f} diff={lb_mip - lb_lp:+.6f} "
              f"solver={eng.lb_solver}")
        assert np.isfinite(lb_mip)
        assert lb_mip >= lb_lp - 1e-5, "MIP 下界不应低于 LP 松弛"
    finally:
        if eng._lb_sub is not None:
            eng._lb_sub.free()
    print("LB_MIP_COMPARE_OK")


if __name__ == "__main__":
    main()
