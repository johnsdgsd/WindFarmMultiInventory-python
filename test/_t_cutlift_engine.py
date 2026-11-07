# _t_cutlift_engine.py — cut_lift 全量 smoke：后向可运行、meta 记录、无非法回退。
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
from utils.io import load_cuts                                                 # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    scfg = dataclasses.replace(scfg, cut_lift=CutLiftConfig(
        enabled=True, scope="all", value="exact", time_limit=2.0,
        rel_gap=0.0, abs_gap=0.0))
    scfg.n_processes = 2
    prob = DefineProblem(pcfg)
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg,
                         results_dir=Path(td), log=lambda *a, **k: None)
        try:
            eng.train(1, verbose=False)
            m = eng.meta()
            assert m["cut_lift"]["enabled"] is True
            assert m["cut_lift"]["scope"] == "all"
            assert m["cut_lift"]["n_fallback"] >= 0
            print(f"  [PASS] full cut_lift smoke：cuts={m['n_cuts']}，"
                  f"fallback={m['cut_lift']['n_fallback']}")
        finally:
            eng.close()
        cuts, meta = load_cuts(Path(td), prob.T, prob.state_dim)
        assert meta["cut_lift"]["enabled"] is True
        assert sum(len(c) for c in cuts) > 0
    print("OK _t_cutlift_engine")


if __name__ == "__main__":
    main()
