# solver.py — 求解器工厂与失败兜底。与问题解耦，可整目录迁移。
"""求解器选择依据 SDDP算法Python实现/guide §7 的实测结论：

  · LP 下 COPT_DLL 与 CBC 的 .dj/pi 都正确；
  · MIP 下 COPT_DLL 拒绝返回对偶（安全），CBC 返回的是 LP 松弛的对偶（危险）；
  · 因此训练一律用 COPT_DLL，CBC 只作兜底。

整数模型的割由「同一模型 + mip=False 重解」的对偶给出（guide §6.2 路线 A），
零侵入：模型一个字不改，求解器侧忽略整性。
"""

from __future__ import annotations

import atexit
import contextlib
import io
import multiprocessing
import os
import sys

import pulp


class SolverError(RuntimeError):
    """子问题求解失败（不可行 / 无界 / 求解器报错）。"""


_hooked = False


def _silence_copt_banner() -> None:
    """压掉 COPT 的许可横幅。它由 COPT 原生库在**进程退出时**向 fd 1 打
    「Your COPT license expires in N days」（C 层 printf，Python 的
    redirect_stdout 拦不住），每个碰过 COPT 的进程各打一次。两种进程分开处理：
    spawn 出的 worker 是 daemon，合法输出只有异常（走 fd 2），直接把 fd 1 指向
    devnull；主进程在 atexit 里先冲掉自己的 stdout 再把 fd 1 指走，自己的输出
    一行不丢（Python 之后还要 flush，但缓冲已空）。"""
    global _hooked
    if multiprocessing.current_process().daemon:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        return
    if _hooked:
        return
    _hooked = True

    def _quiet_exit() -> None:
        sys.stdout.flush()
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)

    atexit.register(_quiet_exit)


def make_solver(*, mip: bool, use_copt: bool = True, msg: bool = False):
    """构造求解器实例。mip=False 时求解器忽略整性（用于 backward 取对偶）。"""
    if use_copt:
        try:
            # pulp.listSolvers 会往 stdout 打 COPT 的启动横幅，临时屏蔽
            with contextlib.redirect_stdout(io.StringIO()):
                avail = pulp.listSolvers(onlyAvailable=True)
            if "COPT_DLL" in avail:
                _silence_copt_banner()
                return pulp.COPT_DLL(msg=msg, mip=mip)
        except Exception:
            pass
    return pulp.PULP_CBC_CMD(msg=msg, mip=mip)


def export_lp(problem: pulp.LpProblem, path) -> None:
    """导出 LP 供 IIS 分析（不可行子问题的现场快照）。"""
    problem.writeLP(str(path))
