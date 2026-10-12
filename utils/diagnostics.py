# diagnostics.py — 对偶提取与割的自检。与问题解耦（只依赖 make_stage 工厂）。
"""拿到 β 之后必须做的三件事（guide §10.1）：

  ① 交叉验证：.dj vs 通用式 c_s + Σ_c a_c·π_c（自动扫描含状态的约束）；
  ② 次梯度不等式：Q(x̄+h·e_j) ≥ Q(x̄) + β_j·h 对正负 h 都成立（h 用 1e-4，
     不能用 1e-6 —— 求解器容差 2e-6 除以极小 h 会放大成 100% 误差）；
  ③ 割的全局合法性：α + β·(s − x̄) ≤ Q(s) 在探测点上处处成立。

为什么不能用中心差分判定对错：折点处中心差分给出左右斜率的平均值，它只是一个
合法次梯度，不能用来判定真伪。次梯度不等式才是充分必要的判据。

`make_stage` 是问题侧的工厂 `(t, prob, cfg) -> 子问题`（code/subproblem.make_stage_subproblem）。
"""

from __future__ import annotations

import numpy as np

DIFF_H = 1e-4


def make_sub(make_stage, prob, cfg, t, state_vec, scene, cuts_next):
    sub = make_stage(t, prob, cfg)
    sub.assemble(cuts_next)
    if t >= 1:
        sub.set_point(state_vec, scene)
    else:
        sub.set_state(state_vec)
    return sub


def dual_crosscheck(make_stage, prob, cfg, t, state_vec, scene, cuts_next,
                    tol: float = 1e-6) -> dict:
    """① .dj（reduced cost）与 c_s + Σ a_c·π_c 逐维比对。"""
    sub = make_sub(make_stage, prob, cfg, t, state_vec, scene, cuts_next)
    sub.solve_lp()
    dj = sub.gradient("dj")
    du = sub.gradient("duals")
    scale = max(1.0, float(np.abs(dj).max()), float(np.abs(du).max()))
    err = float(np.abs(dj - du).max()) / scale
    return {"t": t, "Q": sub.obj, "max_rel_err": err, "ok": err <= tol,
            "n_nonzero_beta": int(np.count_nonzero(np.abs(dj) > 1e-9))}


def subgradient_inequality(make_stage, prob, cfg, t, state_vec, scene, cuts_next,
                           comps, h: float = DIFF_H, tol: float = 1e-6,
                           tol_rel: float = 1e-5) -> dict:
    """② 逐分量验证 β_j 落在左右单侧差商之间。

    判据是**次梯度不等式**：Q(x̄±h·e_j) − Q(x̄) ≥ ±h·β_j，即 left ≤ β_j ≤ right
    （left/right 为左右单侧差商）。中心差分在折点处给的是两斜率均值，只能算作一个
    合法次梯度、无法判伪，故不用它做判据。

    状态分量在 [0, ub] 边界上时，越界的一侧会被 model 夹回边界 → 该侧差商恒为 0，
    是假信号，因此该侧跳过。"why" 里记录跳过原因。
    """
    sub = make_sub(make_stage, prob, cfg, t, state_vec, scene, cuts_next)
    sub.solve_lp()
    beta = sub.gradient("dj")
    q0 = sub.obj
    ub = prob.layout.upper(prob)
    xbar = np.asarray(state_vec, dtype=float)
    out = []
    for j in comps:
        hj = min(h, 0.5 * float(ub[j]) if ub[j] > 0 else h)
        dan, ua = float(xbar[j]) - hj, float(xbar[j]) + hj
        can_dn, can_up = dan >= 0.0, ua <= float(ub[j])
        res = {"j": int(j), "beta": float(beta[j]), "left": None, "right": None,
               "ok": None, "why": ""}
        if not can_dn and not can_up:
            res["why"] = "两侧都越界，无法差分"
            out.append(res)
            continue
        try:
            if can_up:
                up = xbar.copy(); up[j] = ua
                su = make_sub(make_stage, prob, cfg, t, up, scene, cuts_next)
                su.solve_lp()
                res["right"] = (su.obj - q0) / hj
            if can_dn:
                dn = xbar.copy(); dn[j] = dan
                sd = make_sub(make_stage, prob, cfg, t, dn, scene, cuts_next)
                sd.solve_lp()
                res["left"] = (q0 - sd.obj) / hj
        except Exception as exc:
            res["why"] = f"±h 处不可行（{exc}）"
            out.append(res)
            continue
        # 容差按量级放缩：差商误差 ≈ 求解器容差/h，h=1e-4 时会被放大 1e4 倍
        scale = max(abs(float(beta[j])), 1.0)
        tol_j = tol + tol_rel * scale
        lhs = -np.inf if res["left"] is None else res["left"] - tol_j
        rhs = np.inf if res["right"] is None else res["right"] + tol_j
        res["ok"] = bool(lhs <= beta[j] <= rhs)
        if not can_up or not can_dn:
            res["why"] = "单侧在边界上，只用另一侧判定"
        out.append(res)
    return {"t": t, "rows": out,
            "ok": all(r["ok"] is not False for r in out),
            "n_checked": sum(1 for r in out if r["ok"] is True),
            "n_skipped": sum(1 for r in out if r["ok"] is None)}


def cut_validity(make_stage, prob, cfg, cut, probes, cuts_next, scene) -> dict:
    """③ 割在探测点上的合法性：α + β·(s − x̄) ≤ Q(s)。

    scene 必须与生成割时同分布（同一 (z,θ) 池即可）；单场景子问题的 Q 只是条件值
    函数，换场景比较的是另一条曲线，无意义。
    """
    worst, where = np.inf, None
    for k, s in enumerate(probes):
        sub = make_sub(make_stage, prob, cfg, cut.stage, s, scene, cuts_next)
        sub.solve_lp()
        slack = sub.obj - cut.value_at(s)
        if slack < worst:
            worst, where = float(slack), k
    return {"stage": cut.stage, "min_slack": worst, "argmin_probe": where,
            "ok": worst >= -1e-6}


def probe_states(rng, prob, n: int = 4, scale: float = 0.9):
    """生成 n 个**可行**探测状态。

    模型对月龄分布有硬耦合（Σ_a n_in[r,a] ≤ N_r + Σ_a F[r,a]），逐分量随机取整几乎
    必然越界 → 不可行。故月龄用多项分布按 N_r 切分（和恰好 = N_r，恒可行），其余
    分量在 [0, scale·ub] 内随机取整。
    """
    Ly = prob.layout
    ub = Ly.upper(prob)
    out = []
    for _ in range(n):
        s = np.zeros(Ly.dim)
        for r in range(prob.N_R):
            w = rng.random(Ly.A) + 1e-6
            cnt = rng.multinomial(int(prob.N[r]), w / w.sum())
            s[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + Ly.A] = cnt
        for j in range(Ly.dim):
            if Ly.off_n <= j < Ly.off_n + Ly.N_R * Ly.A:
                continue
            s[j] = float(np.rint(rng.random() * scale * float(ub[j])))
        out.append(s)
    return out
