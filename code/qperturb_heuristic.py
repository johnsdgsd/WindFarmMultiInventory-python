# qperturb_heuristic.py — 旧的 q 局部扰动 + LP/MIP 重解启发式。
"""固定状态/场景，先解一次原阶段 LP/MIP，再按概率扰动 q 并固定 q 重解。

这不是纯构造启发式：它确实会解两次 LP/MIP。保留它用于和 SA、纯 LP 基线做
A/B；所有决策仍来自原 LP/MIP 子问题，因此约束/成本口径与 LP 完全一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from coptpy import COPT


@dataclass
class QPerturbResult:
    x_out: np.ndarray
    stage_cost: float
    objective: float
    base_x_out: np.ndarray
    base_stage_cost: float
    base_objective: float
    q_base: np.ndarray
    q_new: np.ndarray
    perturbed: bool
    failed: bool
    n_solves: int
    decisions: dict


def qperturb_step(sub, x, scene, *, rng=None) -> QPerturbResult:
    """先解 base LP/MIP；扰动 q 后固定 q 再解一次。

    `sub` 是已设置 cut 集合的 StageSubproblem；本函数只改 q 的变量界，结束时恢复。
    返回最终可行性解；若 q 未变化或重解失败，返回 base 解并标记。
    """
    cfg = sub.cfg
    prob = sub.prob_data
    Ly = sub.layout
    if sub.is_virtual:
        raise RuntimeError("qperturb 只用于真实阶段")
    if rng is None:
        rng = np.random.default_rng(20261111)
    q_prob = float(getattr(cfg, "heur_q_prob", 0.5))
    q_step = float(getattr(cfg, "heur_q_step", 1.0))
    resolve_prob = float(getattr(cfg, "heur_q_resolve_prob", 1.0))
    integer = bool(getattr(cfg, "integer_enabled", False))
    if q_step <= 0:
        raise ValueError(f"heur_q_step 非法：{q_step}")

    qvars = sub.x_out[Ly.q_slice()]
    ubq = np.asarray(Ly.upper(prob)[Ly.q_slice()], dtype=float)
    sub.set_point(x, scene)
    sub.solve_forward()
    n_solves = 1
    base_x = np.asarray(sub.out_state_vector(), dtype=float).copy()
    base_cost = float(sub.stage_cost())
    base_obj = float(sub.obj)
    base_dec = sub.decision_values()
    q_base = np.asarray(base_dec["q"], dtype=float).copy()
    # 按概率跳过扰动：直接使用 base 最优解，不再重解。
    if resolve_prob <= 0.0 or rng.random() >= resolve_prob:
        return QPerturbResult(base_x, base_cost, base_obj, base_x, base_cost,
                              base_obj, q_base, q_base.copy(), False, False,
                              n_solves, base_dec)
    if q_prob <= 0.0:
        return QPerturbResult(base_x, base_cost, base_obj, base_x, base_cost,
                              base_obj, q_base, q_base.copy(), False, False,
                              n_solves, base_dec)

    # 扰动上限不能超过原 LP 解的可用容量。
    I_out = Ly.I(base_x)
    pipe = Ly.o(base_x).sum(axis=0)
    qcap = np.minimum(ubq, np.maximum(prob.Imax - (I_out + pipe), 0.0))
    qcap = np.maximum(qcap, np.minimum(q_base, ubq))
    q_new = q_base.copy()
    for r in range(Ly.N_R):
        if rng.random() < q_prob:
            d = q_step if rng.random() < 0.5 else -q_step
            q_new[r] = min(float(qcap[r]), max(0.0, q_new[r] + d))
    if integer:
        q_new = np.rint(q_new)
    if np.allclose(q_new, q_base, atol=1e-9, rtol=0.0):
        # 全部不变时强制动一个可行分量（旧启发式的退化保护）。
        up = np.flatnonzero(qcap > q_base + 1e-9)
        dn = np.flatnonzero(q_base > 1e-9)
        choices = [("u", int(r)) for r in up] + [("d", int(r)) for r in dn]
        if not choices:
            return QPerturbResult(
                base_x, base_cost, base_obj, base_x, base_cost, base_obj,
                q_base, q_base.copy(), False, False, n_solves, base_dec)
        direction, r = choices[int(rng.integers(0, len(choices)))]
        q_new[r] = min(float(qcap[r]), max(
            0.0, q_base[r] + (q_step if direction == "u" else -q_step)))
        if integer:
            q_new = np.rint(q_new)
    if np.allclose(q_new, q_base, atol=1e-9, rtol=0.0):
        return QPerturbResult(base_x, base_cost, base_obj, base_x, base_cost,
                              base_obj, q_base, q_base.copy(), False, False,
                              n_solves, base_dec)

    failed = False
    try:
        sub.model.setInfo(COPT.Info.LB, qvars, [float(v) for v in q_new])
        sub.model.setInfo(COPT.Info.UB, qvars, [float(v) for v in q_new])
        sub.solve_forward()
        n_solves += 1
        xo = np.asarray(sub.out_state_vector(), dtype=float).copy()
        cost = float(sub.stage_cost())
        obj = float(sub.obj)
        dec = sub.decision_values()
        # 只接受 q 确实落在扰动目标上的解；极小浮点残差视为通过。
        q_final = np.asarray(dec["q"], dtype=float)
        if np.max(np.abs(q_final - q_new)) > 1e-6:
            failed = True
            xo, cost, obj, dec = base_x, base_cost, base_obj, base_dec
    except Exception:
        failed = True
        xo, cost, obj, dec = base_x, base_cost, base_obj, base_dec
    finally:
        sub.model.setInfo(COPT.Info.LB, qvars, [0.0] * Ly.N_R)
        sub.model.setInfo(COPT.Info.UB, qvars, [float(v) for v in ubq])
    return QPerturbResult(
        xo, cost, obj, base_x, base_cost, base_obj, q_base,
        q_new.copy(), bool(not failed), bool(failed), n_solves, dec)
