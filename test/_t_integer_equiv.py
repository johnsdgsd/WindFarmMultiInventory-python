# _t_integer_equiv.py — 整数模型 LP 松弛值 ≤ MIP 值；目标读取不粘滞。
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


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_integer" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_integer" / "solver.toml")
    prob = DefineProblem(pcfg)
    x0 = np.asarray(prob.initial_state(), float)

    vals = {}
    for tag, mip in (("lp", False), ("mip", True)):
        sp = StageSubproblem(1, prob, scfg)
        try:
            sp.assemble(None, key=None)
            sp.set_state(x0, require_int=bool(mip))
            sp.set_scene(prob.pool[0])
            sp.solve_mip() if mip else sp.solve_lp()
            vals[tag] = sp.obj
        finally:
            sp.free()
    assert vals["lp"] <= vals["mip"] + 1e-5, vals
    print(f"  [PASS] stage1 LP={vals['lp']:.6f} ≤ MIP={vals['mip']:.6f}")

    # 同一模型交替 LP/MIP/LP，确认目标读取
    sp = StageSubproblem(1, prob, scfg)
    try:
        sp.assemble(None, key=None)
        sp.set_state(x0, require_int=True)
        sp.set_scene(prob.pool[0])
        sp.solve_lp(); a = sp.obj
        sp.solve_mip(); b = sp.obj
        sp.solve_lp(); c = sp.obj
        assert abs(a - c) <= 1e-7 and b >= a - 1e-7, (a, b, c)
        print(f"  [PASS] 交替 LP/MIP/LP 目标 {a:.4f}/{b:.4f}/{c:.4f}")
    finally:
        sp.free()

    sp0 = StageSubproblem(0, prob, scfg)
    try:
        sp0.assemble(None, key=None)
        sp0.set_state(x0, require_int=True)
        sp0.solve_lp(); a = sp0.obj
        sp0.solve_mip(); b = sp0.obj
        assert a <= b + 1e-6, (a, b)
        print(f"  [PASS] stage0 LP={a:.4f} ≤ MIP={b:.4f}")
    finally:
        sp0.free()
    print("OK _t_integer_equiv")


if __name__ == "__main__":
    main()
