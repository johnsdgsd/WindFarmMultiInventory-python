# utils — SDDP 工具包。与具体问题解耦（不 import code/），可整目录迁移到其他项目。
"""对外接口：

    from utils import SDDPEngine, Cut, trivial_cut, simulate

问题侧只需提供一个模块顶层工厂：

    make_sub(t, prob, cfg) -> 子问题对象

子问题对象须支持：
    assemble(cuts_next, key=None)   拼装未来成本割（key 相同则跳过）
    set_state(x) / set_scene(scene) / set_point(x, scene)
    solve_lp()                       # 纯 LP；solve_mip 仅保留给旧诊断脚本
    out_state_vector() / stage_cost() / gradient() / obj
    free()

模块分工：
    copt_env   coptpy 环境、参数与 ASCII 落盘兜底
    payload    割集跨进程传递（shared_memory + pickle 回退）
    worker     spawn worker 装配、2 个长驻模型、三类任务函数
    engine     前向并行 / 后向阶段串行 / 割池 / 下界 / 训练循环
    cut        割的表示与序列化
    io         结果目录、配置与割文件读写
    simulation 蒙特卡洛仿真与流式统计
    diagnostics 对偶提取交叉验证（诊断用）
"""

from .cut import Cut, FORMAT_VERSION, pack, trivial_cut
from .copt_env import SolverError, dump_lp
from .engine import SDDPEngine
from .simulation import SimStats, simulate, simulate_paths

__all__ = [
    "Cut", "FORMAT_VERSION", "pack", "trivial_cut",
    "SolverError", "dump_lp",
    "SDDPEngine",
    "SimStats", "simulate", "simulate_paths",
]
