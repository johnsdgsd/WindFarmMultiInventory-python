# _t_tier2_equiv.py — Tier2 LP 松弛 ≤ MIP；交替 LP/MIP/LP 目标读取正确。
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import IntegerConfig, load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                                  # noqa: E402
from code.subproblem import StageSubproblem                                  # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    lpcfg = replace(scfg, integer=IntegerConfig(
        enabled=False, scope="tier2_full", rounding="cumulative_tail_nearest"))
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    rng = np.random.default_rng(11)
    x = np.asarray(prob.initial_state(), float).copy()
    x[Ly.off_I] = 5.0
    x[Ly.off_q] = 2.0
    # 随机整数 n，保证行和；同时让 W>0，目标非零
    nr, g = prob.N_R, prob.G
    n = np.zeros((nr, g), dtype=int)
    for r in range(nr):
        cuts = np.sort(rng.choice(np.arange(1, int(prob.N[r])), size=g - 1,
                                  replace=False))
        n[r] = np.diff(np.r_[0, cuts, int(prob.N[r])])
        n[r, -1] += 0
    x[Ly.off_n:Ly.off_n + Ly.n_n] = n.ravel()
    scene = prob.pool[3]

    vals = {}
    for tag, cfg, mip in (("mip", scfg, True), ("lp", lpcfg, False)):
        sp = StageSubproblem(1, prob, cfg)
        try:
            sp.assemble(None, key=None)
            sp.set_point(x, scene, require_int=bool(mip))
            sp.solve_mip() if mip else sp.solve_lp()
            vals[tag] = sp.obj
        finally:
            sp.free()
    assert vals["lp"] <= vals["mip"] + 1e-5, vals
    print(f"  [PASS] Q_LP={vals['lp']:.6f} ≤ Q_MIP={vals['mip']:.6f}")

    sp = StageSubproblem(1, prob, scfg)
    try:
        sp.assemble(None, key=None)
        sp.set_point(x, scene, require_int=True)
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
        sp0.set_state(x, require_int=True)
        sp0.solve_lp(); a = sp0.obj
        sp0.solve_mip(); b = sp0.obj
        assert a <= b + 1e-6, (a, b)
        print(f"  [PASS] stage0 LP={a:.4f} ≤ MIP={b:.4f}")
    finally:
        sp0.free()
    print("OK _t_tier2_equiv")


if __name__ == "__main__":
    main()
