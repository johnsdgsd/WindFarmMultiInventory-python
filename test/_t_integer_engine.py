# _t_integer_engine.py — 整数引擎：前向 MIP/后向 LP、trial 整数、meta、resume。
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils import simulate                                      # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_integer" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_integer" / "solver.toml")
    scfg.n_processes = 1
    prob = DefineProblem(pcfg)
    Ly, T = prob.layout, prob.T
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            trial, _ = eng.forward(scfg.n_forward_paths)
            m = Ly.logistics_integer_mask()
            for t in range(1, T + 1):
                for x in trial[t]:
                    a = np.asarray(x, float)
                    assert np.max(np.abs(a[m] - np.rint(a[m]))) <= 1e-6
            results = eng.forward_paths([eng.sim_scene_paths(1, seed=2026)[0]],
                                        record=True)[0]
            for rec in results[3]:
                for key in ("q", "I_out", "taup", "taum", "S_end"):
                    a = np.asarray(rec[key], float)
                    assert np.max(np.abs(a - np.rint(a))) <= 1e-6, key
            print("  [PASS] 前向 MIP：trial/记录 I/o/q、y/tau/S_end 整数")
        finally:
            eng.close()

        cuts, meta = load_cuts(td, T, prob.state_dim)
        assert meta.get("integer_mode") is True
        assert meta.get("forward_solver") == "mip"
        assert meta.get("integer_scope") == "tier1_logistics"
        print("  [PASS] cuts meta 记录整数模式")

        eng2 = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                          log=lambda *a, **k: None)
        try:
            eng2.cuts = cuts
            eng2.lb_best = float(meta.get("lb_best", 0.0))
            eng2.restore_cut_history()
            eng2.train(1, start_iter=int(meta["n_iterations"]) + 1, verbose=False)
            assert eng2.lower_bound() >= 0
            st = simulate(eng2, n_paths=3, seed=2026, chunk_size=3)
            assert st.n == 3 and np.isfinite(st.mean)
            print(f"  [PASS] resume + 3 路径 MIP 仿真 UB={st.mean:.1f}")
        finally:
            eng2.close()
    print("OK _t_integer_engine")


if __name__ == "__main__":
    main()
