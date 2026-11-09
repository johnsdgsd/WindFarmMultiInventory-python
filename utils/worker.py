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
    _W["integer"] = bool(getattr(cfg, "integer_enabled", False))
    _W["cut_lift"] = bool(getattr(cfg, "cut_lift_enabled", False))
    _W["cut_lift_cfg"] = getattr(cfg, "cut_lift", None)
    x0 = np.round(np.asarray(initial_state, dtype=float).ravel(), 6)
    if _W["integer"]:
        m = prob.layout.integer_mask(str(getattr(cfg, "integer_scope", "none")))
        x0[m] = np.rint(x0[m])
    _W["x0"] = x0
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
    int_mode = _W.get("integer", False)
    fb0 = int(ce.MIP_FALLBACKS)
    x = _W["x0"].copy()
    states, costs = [], []
    records = [] if record else None
    for t in range(0, T + 1):
        with pl.attached(refs[t] if t < T else None) as cuts_next:
            key = (round_id, t, "fwd")
            if t == 0:
                sub = _stage0_sub(cuts_next, key)
                sub.set_state(x, require_int=int_mode)
                si = -1
            else:
                # ⚠ 必须记**求解前**的 x：states[t-1] 是第 t 期的入库状态，后向要拿它
                # 做子问题的输入点、并以此为锚算割的截距。记求解后的期末状态会整体
                # 错位一期 —— 割锚在「下一年」的状态上，阶段 0 面对的初始状态处外推成
                # 巨额负数，θ ≥ 0 恒胜，下界永远为 0（实测 15 轮 LB 全 0 的真凶）。
                states.append(x.copy())
                sub = _shared_sub(t, cuts_next, key)
                si = int(scene_path[t - 1])
                sub.set_point(x, prob.pool[si], require_int=int_mode)
            try:
                sub.solve_forward()
            except Exception as exc:
                raise _fail("forward", t, si, sub, exc)
            x = sub.out_state_vector()
            if t >= 1:
                costs.append(sub.stage_cost())
                if record:
                    records.append({**sub.decision_values(), "cost": costs[-1],
                                    "scene": si})
    return states, costs, float(sum(costs)), records, int(ce.MIP_FALLBACKS) - fb0


# ---------------------------------------------------------------- 后向任务
def backward_point_task(task):
    """一个试探点的后向：逐场景解 LP 松弛取 .dj，场景等概率平均得期望次梯度。

    Q(x̄) = E_ω[Q(x̄;ω)]，而 ∂(Σ p_ω Q(x̄;ω)) = Σ p_ω ∂Q(x̄;ω)（凸函数次微分可加），
    故场景平均的 .dj 就是期望值函数的一个合法次梯度。

    **一任务 = 一个试探点 = 一条割**（用户要求的最小产出单位）。返回 (beta, obj)，
    截距 alpha = obj − beta@x 由主管程算（x 只有主管程有）。
    """
    if len(task) >= 7:
        round_id, t, trial_idx, x, scene_inds, ref = task[:6]
        lift_flag = bool(task[6])
    else:
        round_id, t, trial_idx, x, scene_inds, ref = task
        lift_flag = True
    prob, dim = _W["prob"], _W["dim"]
    int_mode = bool(_W.get("integer", False))
    lift = bool(lift_flag and _W.get("cut_lift", False))
    lcfg = _W.get("cut_lift_cfg")
    lift_value = str(getattr(lcfg, "value", "exact"))
    with pl.attached(ref) as cuts_next:
        sub = _shared_sub(t, cuts_next, (round_id, t, "bwd"))
        sub.set_state(x, require_int=int_mode)
        grads = np.empty((len(scene_inds), dim))
        vals = np.empty(len(scene_inds))
        lp_objs = np.empty(len(scene_inds))
        n_fb = 0
        for k, si in enumerate(scene_inds):
            sub.set_scene(prob.pool[si])
            try:
                sub.solve_lp()
            except Exception as exc:
                # COPT 偶发在“前向 MIP → 换割 → 后向 LP”后保留无效基；
                # 重建一次共享模型再试，仍失败才按真错误上报。
                old = _W.get("shared")
                if old is not None:
                    try:
                        old.free()
                    except Exception:
                        pass
                    _W["shared"] = None
                    sub = _shared_sub(t, cuts_next, (round_id, t, "bwd"))
                    sub.set_state(x, require_int=int_mode)
                    sub.set_scene(prob.pool[si])
                    try:
                        sub.solve_lp()
                    except Exception as exc2:
                        raise _fail("backward", t, si, sub, exc2)
                else:
                    raise _fail("backward", t, si, sub, exc)
            beta = np.asarray(sub.gradient("dj"), float)
            lp_obj = float(sub.obj)
            grads[k] = beta
            lp_objs[k] = lp_obj
            if lift:
                # 先解固定状态整数子问题，用其整数解作为释放状态 lift MIP 的起点。
                # 这只是热启动；是否接受 lift 值由 solve_lift 的 exact 口径决定。
                try:
                    sub.solve_mip()
                except Exception:
                    pass
                nu = sub.solve_lift(
                    beta, time_limit=float(lcfg.time_limit),
                    rel_gap=float(lcfg.rel_gap),
                    abs_gap=float(lcfg.abs_gap),
                    safety_margin=float(lcfg.safety_margin),
                    value=lift_value)
                if nu is None:
                    nu = lp_obj - float(beta @ x)
                    n_fb += 1
            else:
                nu = lp_obj - float(beta @ x)
            vals[k] = nu
        beta_mean = grads.mean(axis=0)
        obj_lift = float(np.mean(vals + grads @ x))
        if lift:
            # 同一 beta 下，LP 割（obj_lp）与提升割都是合法下界；取 max 保证
            # BestBnd 未解透时不会用一个比标准割更弱的割。
            obj_lp = float(np.mean(lp_objs))
            obj_used = max(obj_lift, obj_lp)
            gain = float(obj_used - obj_lp)
            return beta_mean, obj_used, n_fb, gain, obj_lp
        return beta_mean, obj_lift


# ---------------------------------------------------------------- 下界任务
def lower_bound_task(task):
    """阶段 0 子问题解 LP。割只增不减 ⇒ 下界单调不减，且不依赖随机数。"""
    round_id, ref = task
    int_mode = bool(_W.get("integer", False))
    with pl.attached(ref) as cuts_next:
        sub = _stage0_sub(cuts_next, (round_id, 0, "lb"))
        sub.set_state(_W["x0"], require_int=int_mode)
        try:
            if int_mode:
                return sub.solve_lb_mip()
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
