# _t_tier2_simulate.py — Tier2 仿真落盘：N 字段、整数 n/N/F、尾取整审计。
from __future__ import annotations

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
from code.sim_analysis import CHECKS, STATE_NUMERIC_CHECKS       # noqa: E402
from simulate import Dump                                        # noqa: E402
from utils import simulate                                      # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402


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
            eng.train(1, verbose=False)
            sps = [eng.sim_scene_paths(3, seed=2026)[0]]
            res = eng.forward_paths(sps, record=True)
            dump = Dump(1, T, Ly, prob.L, with_N=True)
            dev, audit = dump.fill(0, res, prob, Ly, integer_mode=True,
                                   integer_scope="tier2_full")
            assert "N" in dump.a and dump.a["N"].shape == (
                1, T, prob.N_R, prob.G, prob.G + 1)
            assert np.max(np.abs(dump.a["N"] - np.rint(dump.a["N"]))) <= 1e-6
            assert np.max(np.abs(dump.a["F"] - np.rint(dump.a["F"]))) <= 1e-6
            assert dev <= 1e-6
            state_tol = 1e-6
            bad = {k: v for k, v in audit.items()
                   if v > (state_tol if k in STATE_NUMERIC_CHECKS else 1e-6)}
            assert not bad, bad
            print(f"  [PASS] Dump N 形状/整数/F 整数；五项偏差 {dev:.2e}；"
                  f"审计 {len(CHECKS)} 项全过")
            st = simulate(eng, n_paths=5, seed=2026, chunk_size=5)
            assert st.n == 5 and st.mean > 0
            print(f"  [PASS] simulate 5 路径 UB={st.mean:.1f}")
        finally:
            eng.close()
    print("OK _t_tier2_simulate")


if __name__ == "__main__":
    main()
