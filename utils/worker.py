# worker.py — spawn worker 的装配、长驻模型缓存与三类任务函数。与问题解耦。
"""问题侧通过 `make_sub(t, prob, cfg) -> 子问题` 工厂注入，utils 不 import code/。

**模型缓存（2 个模型/worker）**：
  · `stage0`  虚拟首阶段，结构特殊，常驻；
  · `shared`  服务所有 t = 1..T。base 结构在 t≥1 上完全相同（约束与目标系数全静态），
    唯一差异是挂载的割集 ⇒ 一个模型 + 按阶段换割行即可，不必每阶段建一个。
换割行的键是 `(round_id, t, pass_tag)`：键不变就跳过（同一阶段内所有场景/试探点复用
同一个模型，只改变量界与行右端）。

内存从最坏 (T+1) 个模型降到 2 个；实测换一次 2000 行割集（卸 2000 + 加 2000）约 0.23 s，
而重建同样规模的模型要 0.23 s 以上且随割数线性增长——换割与重建同阶，但省下的是
T 倍的内存。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import copt_env as ce
from . import payload as pl

# worker 侧全局状态：spawn 下每个子进程由 initializer 装配一次
_W: dict = {}


def init_worker(prob, make_sub, cfg, T: int, initial_state, results_dir) -> None:
    _W["prob"] = prob
    _W["make_sub"] = make_sub
    _W["cfg"] = cfg
    _W["T"] = int(T)
    _W["dim"] = int(prob.state_dim)
    _W["x0"] = np.round(np.asarray(initial_state, dtype=float).ravel(), 6)
    _W["results_dir"] = None if results_dir is None else Path(results_dir)
    _W["stage0"] = None
    _W["shared"] = None
    # 排查开关：worker 也会在超时后打印自己的线程栈（默认关闭）。主进程那份在 run.py。
    _sa = os.environ.get("SDDP_STACK_AFTER")
    if _sa:
        import faulthandler
        faulthandler.enable()
        faulthandler.dump_traceback_later(float(_sa), exit=True)


def _fail(where: str, t, scene_ind, sub, exc) -> Exception:
    """求解失败的兜底：把现场导出成 LP，供 AnalysisModel_IIS.py 分析冲突集。
    COPT 写不进中文路径，dump_lp 会先落 ASCII 临时目录再搬过来。"""
    out = _W.get("results_dir")
    path = None
    if out is not None:
        path = ce.dump_lp(sub.model, out / f"subproblem_{where}_t{t}_s{scene_ind}.lp")
    return ce.SolverError(
        f"{where} 求解失败：期 {t}，场景 {scene_ind}，现场 LP = {path}（{exc}）")


# ---------------------------------------------------------------- 模型缓存
def _shared_sub(t: int, cuts_next, key):
    """服务 t ≥ 1 的共享模型。首次调用时按 t=1 建 base；t≥1 的 base 结构完全相同，
    故 t 只进 key（决定换哪一套割行），不影响建模。"""
    sub = _W["shared"]
    if sub is None:
        sub = _W["shared"] = _W["make_sub"](1, _W["prob"], _W["cfg"])
    sub.assemble(cuts_next, key=key)
    return sub


def _stage0_sub(cuts_next, key):
    sub = _W["stage0"]
    if sub is None:
        sub = _W["stage0"] = _W["make_sub"](0, _W["prob"], _W["cfg"])
    sub.assemble(cuts_next, key=key)
    return sub


# ---------------------------------------------------------------- 前向任务
def forward_path_task(task):
    """走一条完整路径：t = 0..T 逐期解 LP 得策略，读期末状态作为下期入库状态。

    **一任务 = 一条路径**（用户要求的最小并行单位）。任务带的是**每阶段一个 ref**
    而不是一个整表 ref —— 前向要走遍所有阶段，故这块载荷整轮持有，由主管程统一释放。
    返回 (states[1..T], costs[1..T], total_cost, records, 0)；
    最后一个 0 是旧 MIP 回退计数，纯 LP 下恒为 0，保留五元组兼容仿真解包。
    """
    round_id, path_idx, scene_path, refs, record = task
    prob, T = _W["prob"], _W["T"]
    x = _W["x0"].copy()
    states, costs = [], []
    records = [] if record else None
    for t in range(0, T + 1):
        with pl.attached(refs[t] if t < T else None) as cuts_next:
            key = (round_id, t, "fwd")
            if t == 0:
                sub = _stage0_sub(cuts_next, key)
                sub.set_state(x)
                si = -1
            else:
                # ⚠ 必须记**求解前**的 x：states[t-1] 是第 t 期的入库状态，后向要拿它
                # 做子问题的输入点、并以此为锚算割的截距。记求解后的期末状态会整体
                # 错位一期 —— 割锚在「下一年」的状态上，阶段 0 面对的初始状态处外推成
                # 巨额负数，θ ≥ 0 恒胜，下界永远为 0（实测 15 轮 LB 全 0 的真凶）。
                states.append(x.copy())
                sub = _shared_sub(t, cuts_next, key)
                si = int(scene_path[t - 1])
                sub.set_point(x, prob.pool[si])
            try:
                sub.solve_lp()
            except Exception as exc:
                raise _fail("forward", t, si, sub, exc)
            x = sub.out_state_vector()
            if t >= 1:
                costs.append(sub.stage_cost())
                if record == "light":
                    records.append({**sub.replay_values(), "cost": costs[-1],
                                    "scene": si})
                elif record:
                    records.append({**sub.decision_values(), "cost": costs[-1],
                                    "scene": si})
    return states, costs, float(sum(costs)), records, 0


# ---------------------------------------------------------------- 后向任务
def backward_point_task(task):
    """一个试探点的后向：逐场景解 LP 松弛取 .dj，场景等概率平均得期望次梯度。

    Q(x̄) = E_ω[Q(x̄;ω)]，而 ∂(Σ p_ω Q(x̄;ω)) = Σ p_ω ∂Q(x̄;ω)（凸函数次微分可加），
    故场景平均的 .dj 就是期望值函数的一个合法次梯度。

    **一任务 = 一个试探点 = 一条割**（用户要求的最小产出单位）。返回 (beta, obj)，
    截距 alpha = obj − beta@x 由主管程算（x 只有主管程有）。
    """
    round_id, t, trial_idx, x, scene_inds, ref, pass_tag = task
    prob, dim = _W["prob"], _W["dim"]
    with pl.attached(ref) as cuts_next:
        sub = _shared_sub(t, cuts_next, (round_id, t, pass_tag))
        sub.set_state(x)
        grads = np.empty((len(scene_inds), dim))
        objs = np.empty(len(scene_inds))
        for k, si in enumerate(scene_inds):
            sub.set_scene(prob.pool[si])
            try:
                sub.solve_lp()
            except Exception as exc:
                raise _fail("backward", t, si, sub, exc)
            grads[k] = sub.gradient("dj")
            objs[k] = sub.obj
    return grads.mean(axis=0), float(objs.mean())


# ---------------------------------------------------------------- 下界任务
def lower_bound_task(task):
    """阶段 0 子问题解 LP。割只增不减 ⇒ 下界单调不减，且不依赖随机数。"""
    round_id, ref = task
    with pl.attached(ref) as cuts_next:
        sub = _stage0_sub(cuts_next, (round_id, 0, "lb"))
        sub.set_state(_W["x0"])
        try:
            sub.solve_lp()
        except Exception as exc:
            raise _fail("lower_bound", 0, -1, sub, exc)
        return float(sub.obj)


def free_models() -> None:
    """释放本 worker 的长驻模型（close 时调）。"""
    for k in ("stage0", "shared"):
        sub = _W.get(k)
        if sub is not None:
            sub.free()
            _W[k] = None
