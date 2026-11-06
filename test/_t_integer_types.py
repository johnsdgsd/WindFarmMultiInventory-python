# _t_integer_types.py — Tier 1 变量类型：I/o/q + 物流决策整数，n/N/theta 连续。
from __future__ import annotations

import sys
from pathlib import Path

from coptpy import COPT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402


def _count(vs):
    return sum(1 for v in vs if v.getType() == COPT.INTEGER)


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_integer" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_integer" / "solver.toml")
    lpcfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    prob = DefineProblem(pcfg)
    sp = StageSubproblem(1, prob, scfg)
    try:
        assert sp.integer_enabled and sp.int_mask.sum() == (prob.N_R * 3)
        assert _count(sp.x_in) == sp.int_mask.sum()
        assert _count(sp.x_out) == sp.int_mask.sum()
        assert _count(sp.taup) == prob.N_R
        assert _count(sp.taum) == prob.N_R
        assert _count([v for row in sp.y for v in row]) == prob.N_R ** 2
        assert _count(sp.S_end) == prob.N_R
        assert _count([v for r in sp.N for i in r for v in i if v is not None]) == 0
        nv = [sp.x_in[prob.layout.ix_n(r, i)] for r in range(prob.N_R)
              for i in range(prob.G)]
        assert _count(nv) == 0
        assert sp._theta.getType() != COPT.INTEGER
        print("  [PASS] Tier 1 类型：I/o/q、y/tau/S_end 整数；n/N/theta 连续")
    finally:
        sp.free()

    sp0 = StageSubproblem(0, prob, scfg)
    try:
        assert _count(sp0.x_in) == sp0.int_mask.sum()
        assert _count(sp0.x_out) == sp0.int_mask.sum()
        print("  [PASS] stage0 q_out/I/o 整数")
    finally:
        sp0.free()

    sp_lp = StageSubproblem(1, prob, lpcfg)
    try:
        assert not sp_lp.integer_enabled and sp_lp.int_mask.sum() == 0
        assert _count(sp_lp.x_in) == _count(sp_lp.x_out) == 0
        assert _count(sp_lp.taup) == _count(sp_lp.S_end) == 0
        print("  [PASS] LP 模式仍全连续")
    finally:
        sp_lp.free()
    print("OK _t_integer_types")


if __name__ == "__main__":
    main()
