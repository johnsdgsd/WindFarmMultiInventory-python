# _t_sync1_v2.py — 单 worker、单前向路径的同步 SDDP，用于和 async K=1 对照。
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
from utils import SDDPEngine                                     # noqa: E402


def main():
    cdir = ROOT / "config"
    prob = DefineProblem(load_problem_config(cdir / "problem.toml"))
    cfg = load_solver_config(cdir / "solver.toml")
    cfg.cut_selection = os.environ.get("CUT_SELECTION", "none")
    cfg.n_processes = 1
    cfg.forward_paths_per_process = 1
    eng = SDDPEngine(prob, make_stage_subproblem, cfg, results_dir=None,
                     log=print)
    hist = eng.train(20, verbose=True)
    print("sync1 done", len(hist), "LB", eng.lower_bound(),
          "solves", eng.n_solves, "cuts", sum(len(c) for c in eng.cuts[1:]))
    eng.close()


if __name__ == "__main__":
    main()
