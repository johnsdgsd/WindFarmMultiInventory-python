# _t_multicut_resume.py — multi checkpoint/resume + v5 single 兼容/v5 multi 拒绝。
#   python test/_t_multicut_resume.py
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg = dataclasses.replace(scfg, n_processes=1, mode="multi")
    prob = DefineProblem(pcfg)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            eng.checkpoint()
        finally:
            eng.close()
        cuts, meta = load_cuts(td, prob.T, prob.state_dim, expected_mode="multi")
        eng2 = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                          log=lambda *a, **k: None)
        try:
            eng2.cuts = cuts
            assert eng2.mode == meta["mode"]
            assert eng2.n_theta == int(meta["n_theta"])
            eng2.restore_cut_history()
            eng2.train(1, start_iter=3, verbose=False)
        finally:
            eng2.close()

    # v5 单割只读兼容；multi 必须拒绝
    base = ROOT / "results" / "base_T60_20261026_204105"
    if (base / "cuts.json").is_file():
        cuts, _ = load_cuts(base, 60, 40, expected_mode="single")
        assert len(cuts) == 61
        try:
            load_cuts(base, 60, 40, expected_mode="multi")
        except ValueError:
            pass
        else:
            raise AssertionError("v5 cuts 未被 multi 拒绝")
    print("OK _t_multicut_resume")


if __name__ == "__main__":
    main()
