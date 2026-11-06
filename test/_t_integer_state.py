# _t_integer_state.py — 整数状态：I/o/q 吸附/拒绝小数，n 允许小数，S_end 整数。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils import copt_env as ce                                 # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_integer" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_integer" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    sp = StageSubproblem(1, prob, scfg)
    try:
        x0 = np.asarray(prob.initial_state(), float)
        sp.assemble(None, key=None)
        sp.set_state(x0, require_int=True)
        assert np.allclose(sp._state_vec[Ly.logistics_integer_mask()],
                           np.rint(sp._state_vec[Ly.logistics_integer_mask()]))

        bad = x0.copy()
        bad[Ly.ix_I(0)] += 0.5
        try:
            sp.set_state(bad, require_int=True)
        except ValueError:
            pass
        else:
            raise AssertionError("I 的 0.5 小数应被拒绝")

        frac_n = x0.copy()
        frac_n[Ly.ix_n(0, 0)] += 0.37
        sp.set_state(frac_n, require_int=True)      # n 未声明整数，允许
        sp.set_state(bad, require_int=False)        # 诊断允许分数
        print("  [PASS] I/o/q 整数校验；n 小数允许；require_int=False 放行诊断")

        sp.set_state(x0, require_int=True)
        sp.set_scene(prob.pool[0])
        sp.solve_mip()
        xo = sp.out_state_vector()
        m = Ly.logistics_integer_mask()
        assert np.max(np.abs(xo[m] - np.rint(xo[m]))) <= 1e-6
        dv = sp.decision_values()
        for key in ("q", "I_out", "taup", "taum", "S_end"):
            a = np.asarray(dv[key], float)
            assert np.max(np.abs(a - np.rint(a))) <= 1e-6, key
        y = np.asarray(dv["y"], float)
        assert np.max(np.abs(y - np.rint(y))) <= 1e-6
        n = Ly.n(xo)
        assert n.shape == (prob.N_R, prob.G)
        print("  [PASS] MIP 期末 I/o/q、y/tau/S_end 精确整数；n 可为小数")
    finally:
        sp.free()
    print("OK _t_integer_state")


if __name__ == "__main__":
    main()
