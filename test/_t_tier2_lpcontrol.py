# _t_tier2_lpcontrol.py — Tier2 LP 控制跑通：前向 LP、N 连续、尾审计、无整数要求。
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (IntegerConfig, load_problem_config,   # noqa: E402
                         load_solver_config)
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils import simulate                                      # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    scfg = dataclasses.replace(scfg, integer=IntegerConfig(
        enabled=False, scope="tier2_full", rounding="cumulative_tail_nearest"))
    scfg.n_processes = 2
    prob = DefineProblem(pcfg)
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=Path(td),
                         log=lambda *a, **k: None)
        try:
            eng.train(1, verbose=False)
            st = simulate(eng, n_paths=5, seed=2026, chunk_size=5)
            assert st.n == 5 and np.isfinite(st.mean)
            assert eng.meta()["integer_mode"] is False
            assert eng.meta()["integer_scope"] == "tier2_full"
            print(f"  [PASS] Tier2 LP 控制：5 路径 UB={st.mean:.1f}，N 连续")
        finally:
            eng.close()
    print("OK _t_tier2_lpcontrol")


if __name__ == "__main__":
    main()
