# sim_analysis.py — 仿真结果的**口径定义**：状态重建、五项成本拆解、约束审计。
#
# 放 code/（问题侧）而不放 utils/：这些函数直接用 ProblemData 与 StateLayout，
# 而 utils/ 的纪律是与问题解耦（不 import code/）。simulate.py（落盘）与
# analyze_sim.py（读盘分析）共用这里的定义，保证两边口径一致。
from __future__ import annotations

import numpy as np

# 与 code/subproblem.py 的目标逐项对应
TERMS = ("采购", "持有", "停机", "主动调拨", "被动调拨")

# 约束审计的核对项（名字对应 subproblem.py 的约束名）
CHECKS = ("pas", "dem", "age0", "F定义", "send_def", "tau_bal", "tau_lo", "tau_hi",
          "tau_src", "inv", "pipe", "pipe_last", "cap", "界.下", "界.上",
          "整性", "链.状态")


def out_state(Ly, x_in, rec) -> np.ndarray:
    """该期期末状态的完整向量，由 x_in 与决策按模型约束唯一确定。

    末期没有下一期状态可借，只能这样重建；其余期与下一期入库状态一致
    （「链.状态」核对项逐点验过）。
    """
    xo = np.empty(Ly.dim)
    xo[:Ly.n_I] = np.asarray(rec["I_out"], float)
    o = np.empty((Ly.L - 1, Ly.N_R))
    for l in range(Ly.L - 2):
        o[l] = Ly.o(x_in)[l + 1]
    o[Ly.L - 2] = Ly.q(x_in)                      # pipe_last: o_out[L−2] = q_in
    xo[Ly.off_o:Ly.off_o + Ly.n_o] = o.ravel()
    xo[Ly.off_n:Ly.off_n + Ly.n_n] = np.asarray(rec["n_out"], float).ravel()
    xo[Ly.off_q:Ly.off_q + Ly.n_q] = np.asarray(rec["q"], float)
    return xo


def F_of(prob, Ly, x_in, scene) -> np.ndarray:
    """失效台数 F = ⌊θ·n⌋（与 Fdef 两式定义一致）。"""
    return np.floor(np.asarray(scene[1], float) * Ly.n(x_in))


def cost_terms(prob, Ly, x_in, rec) -> np.ndarray:
    """该期的五项成本（与 subproblem.py 的目标逐项对应）。"""
    n_in, I_in, q_in = Ly.n(x_in), Ly.I(x_in), Ly.q(x_in)
    I_out = np.asarray(rec["I_out"], float)
    y = np.asarray(rec["y"], float)
    taup = np.asarray(rec["taup"], float)
    taum = np.asarray(rec["taum"], float)
    return np.array([
        prob.C * q_in.sum(),
        prob.h / 2.0 * (I_in.sum() + I_out.sum()),
        prob.Cs * (prob.N.sum() - n_in.sum()),
        prob.atc / 2.0 * (taup.sum() + taum.sum()),
        float((prob.tc_route * y).sum()),
    ])


def check_stage(prob, Ly, x, xo, q_out, taup, taum, S, y, scene_idx, ub) -> dict:
    """核对一期：返回 {核对项: 该期最大违约量}（≤ tol 视为合规）。

    参数全部是**数组**（不是 worker 的 rec 字典），这样 `simulate.py`（手上有 rec）
    与 `analyze_sim.py`（手上有 npz）能共用同一份判据。

    **F 的判据不能用 `floor(θ·n)` 硬比**：模型那条约束是 `θn − 1 ≤ F ≤ θn`、`F ∈ ℤ`，
    当 `θn` 落在整数上方 FeasTol(1e-6) 以内时（实测 4 / 6e7 个等式），求解器允许 F 取
    整数边界下方那一档。所以这里**从 age 行反解出模型自己的 F**，再查它是否落在
    `[θn − 1, θn]` 区间内（含容差）—— 这才是模型的真实约束。
    """
    NR, A, L = Ly.N_R, Ly.A, Ly.L
    n_in, I_in, o_in, q_in = Ly.n(x), Ly.I(x), Ly.o(x), Ly.q(x)
    n_out = Ly.n(xo)
    I_out = np.asarray(xo[:Ly.n_I], float)
    q_out = np.asarray(q_out, float)
    y = np.asarray(y, float)
    taup = np.asarray(taup, float)
    taum = np.asarray(taum, float)
    S = np.asarray(S, float)
    tau = taup - taum
    th = np.asarray(prob.pool[int(scene_idx)][1], float)
    # 模型自己的 F：a ≤ A−2 从 age 行反解；A−1 无 age 行，只能按定义重算
    F = np.empty((NR, A))
    F[:, :A - 1] = n_in[:, :A - 1] - n_out[:, 1:]
    F[:, A - 1] = np.floor(th[:, A - 1] * n_in[:, A - 1])
    o_out = np.empty_like(o_in)
    for l in range(L - 2):
        o_out[l] = o_in[l + 1]
    o_out[L - 2] = q_in
    prod = th * n_in

    d = {
        "pas": np.maximum(y.sum(axis=0) - I_in, 0).max(),
        "dem": np.maximum(y.sum(axis=1) + n_in.sum(axis=1) - F.sum(axis=1) - prob.N, 0).max(),
        "age0": np.abs(n_out[:, 0] - y.sum(axis=1)).max(),
        "F定义": max(float(np.maximum(prod - 1.0 - F, 0).max()),
                    float(np.maximum(F - prod, 0).max())),
        "send_def": np.abs(S - (I_in - y.sum(axis=0) + o_in[0])).max(),
        "tau_bal": abs(tau.sum()),
        "tau_lo": np.maximum(-(S + tau), 0).max(),
        "tau_hi": np.maximum(S + tau - prob.Imax, 0).max(),
        "tau_src": np.maximum(tau - (S.sum() - S), 0).max(),
        "inv": np.abs(I_out - (S + tau)).max(),
        "pipe": max([np.abs(o_out[l] - o_in[l + 1]).max() for l in range(L - 2)] or [0.0]),
        "pipe_last": np.abs(o_out[L - 2] - q_in).max(),
        "cap": np.maximum(I_out + o_out.sum(axis=0) + q_out - prob.Imax, 0).max(),
        "界.下": np.maximum(-x, 0).max(),
        "界.上": np.maximum(x - ub, 0).max(),
        "整性": max(np.abs(v - np.rint(v)).max() for v in
                    (n_in, I_in, o_in, q_in, n_out, I_out, q_out, y, taup, taum)),
    }
    return d


def chain_dev(Ly, x, xo, rec) -> float:
    """轨迹自洽：记录里重建的期末状态必须等于下一期的入库状态。"""
    return float(np.abs(np.asarray(xo, float) - out_state(Ly, x, rec)).max())
