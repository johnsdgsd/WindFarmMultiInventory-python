# _t_tier2_types.py — Tier 2 变量类型：全状态 + N 整数，theta 连续；LP 控制全连续。
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
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    prob = DefineProblem(pcfg)
    sp = StageSubproblem(1, prob, scfg)
    try:
        assert sp.tier2 and sp.integer_enabled
        assert sp.int_mask.sum() == prob.state_dim
        assert _count(sp.x_in) == prob.state_dim
        assert _count(sp.x_out) == prob.state_dim
        assert _count(sp.taup) == prob.N_R
        assert _count(sp.taum) == prob.N_R
        assert _count(sp.S_end) == prob.N_R
        nv = [v for r in sp.N for row in r for v in row if v is not None]
        assert len(nv) == prob.N_R * prob.G * (prob.G + 3) // 2
        assert _count(nv) == len(nv)
        assert sp._theta.getType() != COPT.INTEGER
        print(f"  [PASS] Tier2：状态 {prob.state_dim} 全整数；N {len(nv)} 个整数")
    finally:
        sp.free()

    # LP 控制：同一 tier2 模型、所有变量连续。
    from dataclasses import replace
    from code.config import IntegerConfig
    lp_cfg = replace(scfg, integer=IntegerConfig(
        enabled=False, scope="tier2_full", rounding="cumulative_tail_nearest"))
    sp = StageSubproblem(1, prob, lp_cfg)
    try:
        assert sp.tier2 and not sp.integer_enabled
        assert sp.int_mask.sum() == 0
        nv = [v for r in sp.N for row in r for v in row if v is not None]
        assert _count(sp.x_in) == _count(sp.x_out) == _count(nv) == 0
        assert _count(sp.taup) == _count(sp.S_end) == 0
        print("  [PASS] Tier2 LP 控制：全连续、尾累计结构仍在")
    finally:
        sp.free()
    print("OK _t_tier2_types")


if __name__ == "__main__":
    main()
