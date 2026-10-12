# copt_env.py — coptpy 环境、建模参数与失败兜底。与问题解耦，可整目录迁移。
"""三条实测约束写死在这里，别处不要绕开：

  ① COPT 的文件接口走窄字符路径：中文路径下 writeLp/writeMps/writeBin/writeSol/
     writeIIS/setLogFile 全部失败（长路径报 CoptError，短路径还会**静默不建文件**）。
     本仓库路径全中文 ⇒ 落盘一律先写 ASCII 临时目录再搬回（`dump_lp`）。
  ② 求解失败时 `objval` 与 `Attr.LpObjVal` 都静默返回上一次的陈旧值，只有
     `status` / `HasLpSol` 可靠 ⇒ 每次求解后必须走 `check_optimal`。
  ③ `m.objval` 对 LP 值有粘性：同一模型上跑过 solveLP() 后，`objval` 永远返回 LP 值，
     即使 solve() 已给出整数解。⇒ MIP 读 `BestObj`，LP 读 `LpObjVal`，不用 `objval`。
     （COPT使用手册.md §7.2）
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import tempfile
from pathlib import Path

import coptpy as cp
from coptpy import COPT


class SolverError(RuntimeError):
    """子问题求解失败（不可行 / 无界 / 求解器报错）。"""


# 曾经用 atexit + os.dup2(fd1 → NUL) 压 COPT 的许可横幅，**已移除**：
# 当 stdout 是控制台时，spawn 出的 daemon worker 里做 fd 1 重定向会让随后（或之后）
# 的 cp.Envr() 永久阻塞 —— worker 拿不到任务，主进程在 pool.map 上无限等，表现为
# 0% CPU 的"假死"。二分实测：worker 只 import coptpy = 正常；import + cp.Envr() = 正常；
# 先 dup2 再 cp.Envr() = 挂死。而 stdout 指向文件/管道时同样的重定向无害，所以这个坑
# 只在终端里直接跑才会踩到（重定向到文件跑则看不出来）。
# 代价：COPT 会在每个建过 Envr 的进程打两三行启动横幅。宁可多点输出，不要假死。

_ENV = None


def get_env() -> "cp.Envr":
    """进程内共享一个 COPT 环境（每个模型各自 CreateProb）。"""
    global _ENV
    if _ENV is None:
        _ENV = cp.Envr()
    return _ENV


def new_model(cfg=None) -> "cp.Model":
    """按配置建一个模型。`cfg` 是 SolverConfig（可为 None，则用 COPT 默认参数）。

    `Threads` 由配置给：多进程并行时必须是 1 —— 实测 3000 变量 LP 单线程
    0.020 s、自动 0.040 s，线程协调开销是净损失，并行度留给进程级。

    `RelGap` 是给**前向 MIP** 的收敛控制：前向只要一条可行策略，不必精确最优，而默认
    gap 1e-4 会让分支定界在某些状态上爆炸（实测第 115 期 15 s 出不来、RSS 1.8 GB）。
    `RelGap` 只作用于 MIP，LP 完全不受影响，所以可以在这里一次设好。

    `TimeLimit` 对 LP 也生效，**不能在这里设**——后向 LP 必须解到最优，被时限打断会让
    `check_optimal` 抛错、白白中止训练。改为只在 `solve(mip=True)` 期间临时设、解完恢复。
    """
    global _MIP_TIME_LIMIT
    m = get_env().createModel()
    m.setParam(COPT.Param.Logging, 0)
    if cfg is not None:
        m.setParam(COPT.Param.Threads, int(cfg.threads_per_process))
        m.setParam(COPT.Param.FeasTol, float(cfg.feas_tol))
        m.setParam(COPT.Param.DualTol, float(cfg.dual_tol))
        m.setParam(COPT.Param.IntTol, float(cfg.int_tol))
        m.setParam(COPT.Param.Presolve, int(cfg.presolve))
        m.setParam(COPT.Param.RelGap, float(cfg.mip_rel_gap))
        _MIP_TIME_LIMIT = float(cfg.mip_time_limit)
    return m


# 慢 MIP 日志阈值（秒）：设 SDDP_SLOW_MIP=5 就会把超过 5 s 的 MIP 记到 stderr
_SLOW_MIP = float(os.environ.get("SDDP_SLOW_MIP", "0") or 0)

# 前向 MIP 打满时限/未收到 gap 就采用可行解的次数（诊断用，跨进程各自计数）
MIP_FALLBACKS = 0
_MIP_TIME_LIMIT = 1e20      # 仅 MIP 期间生效，由 new_model 按配置覆盖


def solve(m: "cp.Model", *, mip: bool) -> float:
    """求解并返回目标值。mip=True 走整数求解，False 走 LP 松弛。

    **MIP 与 LP 的成败判据不同**：
      · LP（后向/下界）：必须精确最优，否则对偶不是合法次梯度 —— 不最优就抛。
      · MIP（前向）：只要一条**可行**策略来给上界、产试探点。到时限/节点限但已有可行解
        时采用该解即可，不影响算法正确性（割仍来自后向 LP，始终是合法下界）。

    返回值的取法：MIP 用 BestObj，LP 用 LpObjVal —— 不用 objval（对 LP 值有粘性）。
    """
    global MIP_FALLBACKS
    if not mip:
        m.solveLP()
        check_optimal(m, mip=False)
        return float(m.getAttr(COPT.Attr.LpObjVal))

    m.setParam(COPT.Param.TimeLimit, _MIP_TIME_LIMIT)
    _t0 = time.time()
    try:
        m.solve()
    finally:
        m.setParam(COPT.Param.TimeLimit, 1e20)   # 恢复，别约束随后的 LP
    _dt = time.time() - _t0
    if _SLOW_MIP and _dt >= _SLOW_MIP:
        try:
            _c = m.getAttr(COPT.Attr.Cols); _r = m.getAttr(COPT.Attr.Rows)
            _h = m.getAttr(COPT.Attr.HasMipSol)
        except Exception:
            _c = _r = _h = -1
        print(f"[slow MIP] {_dt:.1f}s  status={m.status}  HasMipSol={_h}  "
              f"cols={_c} rows={_r}  TimeLimit={_MIP_TIME_LIMIT}",
              file=sys.stderr, flush=True)
    if m.status != COPT.OPTIMAL:
        if int(m.getAttr(COPT.Attr.HasMipSol)) == 1:
            MIP_FALLBACKS += 1
            return float(m.getAttr(COPT.Attr.BestObj))
        raise SolverError(f"MIP 求解失败且无可行解：status={m.status}"
                          f"（OPTIMAL={COPT.OPTIMAL} INFEASIBLE={COPT.INFEASIBLE} "
                          f"TIMEOUT={COPT.TIMEOUT}）")
    return float(m.getAttr(COPT.Attr.BestObj))


def check_optimal(m: "cp.Model", *, mip: bool) -> None:
    """失败即抛。`status` 是唯一可靠的成败判据（`objval` 会是陈旧值）。"""
    if m.status != COPT.OPTIMAL:
        kind = "MIP" if mip else "LP"
        raise SolverError(f"{kind} 求解未达最优：status={m.status}"
                          f"（OPTIMAL={COPT.OPTIMAL} INFEASIBLE={COPT.INFEASIBLE} "
                          f"UNBOUNDED={COPT.UNBOUNDED}）")


def values(m: "cp.Model", var) -> list:
    """变量取值。给单个变量返回 float，给列表返回 list。"""
    if isinstance(var, (list, tuple)):
        return [float(v) for v in m.getInfo(COPT.Info.Value, list(var))]
    return float(m.getInfo(COPT.Info.Value, [var])[0])


def redcosts(m: "cp.Model", var) -> list:
    """变量的既约成本（= 状态锁界时的 ∂Q/∂x）。必须先解 LP，MIP 下取不到。"""
    if isinstance(var, (list, tuple)):
        return [float(v) for v in m.getInfo(COPT.Info.RedCost, list(var))]
    return float(m.getInfo(COPT.Info.RedCost, [var])[0])


_ASCII_DIR: Path | None = None


def ascii_dir() -> Path:
    """进程内共享的 ASCII 临时目录（COPT 落盘唯一可用的地方）。"""
    global _ASCII_DIR
    if _ASCII_DIR is None:
        _ASCII_DIR = Path(tempfile.gettempdir()) / "copt_out"
        _ASCII_DIR.mkdir(parents=True, exist_ok=True)
    return _ASCII_DIR


def dump_lp(m: "cp.Model", dest) -> Path | None:
    """把模型导出成 .lp 供 IIS 分析。COPT 写不进中文路径，故先落 ASCII 临时目录
    再搬到 dest；任一步失败都返回 None（兜底本身不能把训练带崩）。"""
    dest = Path(dest)
    tmp = ascii_dir() / dest.name
    try:
        m.writeLp(str(tmp))
        if not tmp.exists():          # 短路径下 COPT 可能静默不建文件
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dest))
        return dest
    except Exception:
        return None


def close_model(m: "cp.Model") -> None:
    """显式释放一个模型（长驻模型换新时必须调，否则内存不回收）。"""
    try:
        m.clear()
    except Exception:
        pass
