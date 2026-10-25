# sim_analysis.py — 仿真口径：状态重建、五项成本拆解、等级转移审计。
#
# 放 code/（问题侧）而不放 utils/：这些函数直接用 ProblemData 与 StateLayout。
# simulate.py（落盘）与 analyze_sim.py（读盘分析）共用这里的定义。
#
# 等级模型口径：
#   N = Θ·n（Θ 从场景池取；不落盘，由 n_in + scene 重算）
#   n_out[0] = Y + N[0,0]；n_out[j] = Σ_i N[i,j]（j≥1）
#   F_new = Σ_i N[i,G]；W = N_r − Σ_i n_i；W_out = W_in + F_new − Y
# 纯 LP：不再检查整性，也不再用 age/rint。
from __future__ import annotations

import numpy as np

TERMS = ("采购", "持有", "停机", "主动调拨", "被动调拨")

CHECKS = ("pas", "fleet", "dem", "grade0", "grade", "N链接", "结构零",
          "F_new", "W守恒", "send_def", "tau_bal", "tau_lo", "tau_hi",
          "tau_src", "inv", "pipe", "pipe_last", "cap", "界.下", "界.上",
          "链.状态")

# 状态流转的 round/repair 会轻微改动 n_out；这三项用 state_repair_tol 审计。
STATE_NUMERIC_CHECKS = ("grade0", "grade", "W守恒")


def out_state(Ly, x_in, rec) -> np.ndarray:
    """由 x_in 与决策重建期末状态（末期没有下一期可借时也用这个）。"""
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
    """由 n_in 与场景 Θ 计算本期新失效 F_new = Σ_i N[i,G]。"""
    th = np.asarray(scene[1], float)
    n = Ly.n(x_in)
    N = th * n[:, :, None]
    return N[:, :, Ly.G].sum(axis=1)


def cost_terms(prob, Ly, x_in, rec) -> np.ndarray:
    """该期的五项成本（与 code/subproblem.py 的目标逐项对应）。"""
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


def check_stage(prob, Ly, x, xo, q_out, taup, taum, S, y, scene_idx, ub,
                F_new=None) -> dict:
    """核对一期，返回 {核对项: 该期最大违约量}（≤ TOL 视为合规）。

    N 由 n_in + 场景 Θ 重算（与模型里的 N=Θn 链接等价）。F_new 可选：
    传入仿真记录的 F 时，核对记录值与重算值；不传则该项为 0。
    fleet/dem 的右端与子问题一致为 N_r + fleet_tol：容差内的 LP 残差不算违规。
    grade0/grade/W守恒 会因状态 round/repair 有 ≤state_repair_tol 的偏差，
    simulate/analyze 对这三项用 STATE_NUMERIC_CHECKS 的阈值审计。
    """
    NR, G, L = Ly.N_R, Ly.G, Ly.L
    n_in, I_in, o_in, q_in = Ly.n(x), Ly.I(x), Ly.o(x), Ly.q(x)
    n_out, I_out, o_out = Ly.n(xo), np.asarray(xo[:Ly.n_I], float), Ly.o(xo)
    q_out = np.asarray(q_out, float)
    y = np.asarray(y, float)
    taup = np.asarray(taup, float)
    taum = np.asarray(taum, float)
    S = np.asarray(S, float)
    tau = taup - taum
    th = np.asarray(prob.pool[int(scene_idx)][1], float)

    N = th * n_in[:, :, None]                     # (NR,G,G+1)
    F_rec = N[:, :, G].sum(axis=1)
    Y = y.sum(axis=1)
    W_in = prob.N - n_in.sum(axis=1)
    W_out = prob.N - n_out.sum(axis=1)

    # 结构零：j<i 必须为 0
    zero_dev = 0.0
    for i in range(G):
        if i:
            zero_dev = max(zero_dev, float(np.max(np.abs(th[:, i, :i]))))
    row_dev = float(np.max(np.abs(N.sum(axis=2) - n_in)))
    f_dev = (float(np.max(np.abs(np.asarray(F_new, float) - F_rec)))
             if F_new is not None else 0.0)
    w_dev = float(np.max(np.abs(W_out - (W_in + F_rec - Y))))
    grade_dev = 0.0
    for j in range(1, G):
        grade_dev = max(grade_dev, float(np.max(np.abs(
            n_out[:, j] - N[:, :, j].sum(axis=1)))))
    grade0_dev = float(np.max(np.abs(n_out[:, 0] - Y - N[:, 0, 0])))
    # fleet/dem 右端与子问题一致：N + fleet_tol（纯 LP 的浮点残差口径）
    cap = prob.N + prob.fleet_tol
    dem_dev = float(np.maximum(Y + N[:, :, :G].sum(axis=(1, 2)) - cap, 0).max())
    fleet_dev = float(np.maximum(n_in.sum(axis=1) - cap, 0).max())

    o_out_calc = np.empty_like(o_in)
    for l in range(L - 2):
        o_out_calc[l] = o_in[l + 1]
    o_out_calc[L - 2] = q_in

    d = {
        "pas": float(np.maximum(y.sum(axis=0) - I_in, 0).max()),
        "fleet": fleet_dev,
        "dem": dem_dev,
        "grade0": grade0_dev,
        "grade": grade_dev,
        "N链接": row_dev,
        "结构零": zero_dev,
        "F_new": f_dev,
        "W守恒": w_dev,
        "send_def": float(np.abs(S - (I_in - y.sum(axis=0) + o_in[0])).max()),
        "tau_bal": float(abs(tau.sum())),
        "tau_lo": float(np.maximum(-(S + tau), 0).max()),
        "tau_hi": float(np.maximum(S + tau - prob.Imax, 0).max()),
        "tau_src": float(np.maximum(tau - (S.sum() - S), 0).max()),
        "inv": float(np.abs(I_out - (S + tau)).max()),
        "pipe": max((float(np.abs(o_out[l] - o_in[l + 1]).max())
                     for l in range(L - 2)), default=0.0),
        "pipe_last": float(np.abs(o_out[L - 2] - q_in).max()),
        "cap": float(np.maximum(I_out + o_out.sum(axis=0) + q_out - prob.Imax,
                                0).max()),
        "界.下": float(np.maximum(-x, 0).max()),
        "界.上": float(np.maximum(x - ub, 0).max()),
    }
    return d


def chain_dev(Ly, x, xo, rec) -> float:
    """轨迹自洽：记录的期末状态必须等于下一期的入库状态。"""
    return float(np.abs(np.asarray(xo, float) - out_state(Ly, x, rec)).max())
