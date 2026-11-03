# _t_multicut_engine.py — smoke 多割：single/multi 跑通、组绑定、LML1、resume。
#   python test/_t_multicut_engine.py
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
    scfg.n_processes = 1
    prob = DefineProblem(pcfg)
    K = len(prob.pool)

    # single 回归
    s1 = dataclasses.replace(scfg, mode="single")
    e1 = SDDPEngine(prob, make_stage_subproblem, s1, results_dir=None,
                    log=lambda *a, **k: None)
    try:
        e1.train(2, verbose=False)
        assert e1.n_theta == 1 and e1.lower_bound() >= 0
    finally:
        e1.close()

    # multi
    sm = dataclasses.replace(scfg, mode="multi")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, sm, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            assert eng.n_theta == K
            for t in range(1, prob.T + 1):
                groups = {c.group for c in eng.cuts[t]}
                assert groups == set(range(K)), (t, len(groups))
        finally:
            eng.close()
        cuts, meta = load_cuts(td, prob.T, prob.state_dim, expected_mode="multi")
        assert meta["mode"] == "multi" and int(meta["n_theta"]) == K
        eng2 = SDDPEngine(prob, make_stage_subproblem, sm, results_dir=td,
                          log=lambda *a, **k: None)
        try:
            eng2.cuts = cuts
            n_hist = eng2.restore_cut_history()
            assert n_hist > 0
            eng2.train(1, start_iter=3, verbose=False)
            assert eng2.lower_bound() >= 0
        finally:
            eng2.close()
        assert (td / "cuts.json").is_file()
    print(f"OK _t_multicut_engine  K={K}")


if __name__ == "__main__":
    main()
