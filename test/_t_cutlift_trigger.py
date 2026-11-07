# _t_cutlift_trigger.py — stagnation 触发：下一轮 lifting、记录 delta/gain/fallback。
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import CutLiftConfig, load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                                    # noqa: E402
from code.subproblem import make_stage_subproblem                              # noqa: E402
from utils.engine import SDDPEngine                                            # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    scfg = dataclasses.replace(scfg, cut_lift=CutLiftConfig(
        enabled=True, scope="all", mode="stagnation", value="exact",
        time_limit=2.0, rel_gap=0.0, abs_gap=0.0,
        lb_gain_threshold=1e9, min_round=1))
    scfg.n_processes = 2
    prob = DefineProblem(pcfg)
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg,
                         results_dir=Path(td), log=lambda *a, **k: None)
        try:
            hist = eng.train(3, verbose=False)
            assert hist[0]["lift_used"] is False
            assert hist[1]["lift_used"] is False
            assert hist[2]["lift_used"] is True
            assert hist[2]["lift_gain"] >= -1e-4
            m = eng.meta()
            assert m["cut_lift"]["lift_rounds"] == [3]
            assert m["cut_lift"]["mode"] == "stagnation"
            print(f"  [PASS] trigger round={m['cut_lift']['lift_rounds']}，"
                  f"delta_lb={hist[2]['delta_lb']:.3f}，"
                  f"lift_gain={hist[2]['lift_gain']:.3e}")
        finally:
            eng.close()
    print("OK _t_cutlift_trigger")


if __name__ == "__main__":
    main()
