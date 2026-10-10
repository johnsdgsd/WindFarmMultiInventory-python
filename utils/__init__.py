# utils — SDDP 工具包。与具体问题解耦（不 import code/），可整目录迁移到其他项目。
"""对外接口：

    from utils import SDDPEngine, Cut, trivial_cut, make_solver, simulate

问题侧只需提供两个回调：
    build_stage(t, state_vec, scene, cuts_next, prob, it) -> 子问题对象
        （需有 solve(solver) / obj / out_state_vector() / stage_cost() / gradient()）
    state_dim, T, initial_state
"""

from .cut import Cut, FORMAT_VERSION, pack, trivial_cut
from .solver import SolverError, export_lp, make_solver
from .training import SDDPEngine
from .simulation import SimStats, simulate, simulate_paths

__all__ = [
    "Cut", "FORMAT_VERSION", "pack", "trivial_cut",
    "SolverError", "export_lp", "make_solver",
    "SDDPEngine",
    "SimStats", "simulate", "simulate_paths",
]
