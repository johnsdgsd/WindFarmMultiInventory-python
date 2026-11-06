# _t_tier2_engine.py — Tier2 引擎：trial/N 整数、meta/版本、仿真；错误 n0 配置被拒。
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
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    scfg.n_processes = 2
    prob = DefineProblem(pcfg)
    Ly, T = prob.layout, prob.T
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            trial, _ = eng.forward(scfg.n_forward_paths)
            m = Ly.integer_mask("tier2_full")
            for t in range(1, T + 1):
                for x in trial[t]:
                    a = np.asarray(x, float)
                    assert np.max(np.abs(a[m] - np.rint(a[m]))) <= 1e-6
                    nrow = a[Ly.off_n:Ly.off_n + Ly.n_n].reshape(
                        prob.N_R, prob.G).sum(axis=1)
                    assert np.max(nrow - prob.N) <= prob.fleet_tol + 1e-6
            st = simulate(eng, n_paths=5, seed=2026, chunk_size=5)
            assert st.n == 5 and np.isfinite(st.mean)
            print(f"  [PASS] Tier2 训练/前向：trial 全整数；5 路径仿真 "
                  f"UB={st.mean:.1f}")
        finally:
            eng.close()

        cuts, meta = load_cuts(td, T, prob.state_dim)
        assert meta.get("integer_mode") is True
        assert meta.get("integer_scope") == "tier2_full"
        assert meta.get("integer_rounding") == "cumulative_tail_nearest"
        assert meta.get("n0_mode") == "integer_largest_remainder"
        print("  [PASS] cuts meta：Tier2 mode/scope/rounding/n0")

        # 错误 n0 配置必须被引擎拦下
        bad_p = dataclasses.replace(pcfg, n0_mode="continuous")
        bad_prob = DefineProblem(bad_p)
        try:
            SDDPEngine(bad_prob, make_stage_subproblem, scfg,
                       results_dir=None, log=lambda *a, **k: None)
        except ValueError:
            print("  [PASS] tier2_full + continuous n0 被拒绝")
        else:
            raise AssertionError("错误的 n0_mode 未被拒绝")
    print("OK _t_tier2_engine")


if __name__ == "__main__":
    main()
