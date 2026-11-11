# forward_heuristic.py — 可行域内的前向阶段模拟退火（SA）启发式。
"""固定 (x_t, scene) 后，用与 LP 子问题相同的约束和成本构造全部阶段决策。

设计要点：
  - 不调用任何 LP/MIP；SA 只访问阶段 0 的等式、界和当前活跃割的 alpha/beta；
  - N 由场景/状态确定：连续模型 N=Θn，Tier 2 用 nearest_tail_flow 的整数可行点；
  - S_end/I_out/o_out/n_out 全部由 LP 等式派生，避免自由搜索破坏可行性；
  - SA 搜索 q/y/tau，所有邻域动作都做与 LP 同口径的可行性审计；
  - 目标 = 五项阶段成本 + θ，θ=max_c(alpha_c+beta_c·x_next)；空割集时为 0。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .transition import nearest_tail_flow, round_repair_n, tail_cdf


class HeuristicInfeasible(Exception):
    """候选解不满足 LP 阶段约束；只用于预期的可行性失败。"""


@dataclass
class HeuristicResult:
    x_out: np.ndarray
    stage_cost: float          # 五项阶段成本，不含 θ
    objective: float           # 阶段成本 + θ（与 LP 前向目标一致）
    cost_terms: dict
    q_out: np.ndarray
    y: np.ndarray
    tau: np.ndarray
    taup: np.ndarray
    taum: np.ndarray
    S_end: np.ndarray
    I_out: np.ndarray
    o_out: np.ndarray
    n_out: np.ndarray
    N: np.ndarray
    F_new: np.ndarray
    W_in: np.ndarray
    W_out: np.ndarray
    sa_moves: int
    sa_accepted: int
    sa_time_s: float


@dataclass
class _Candidate:
    q: np.ndarray
    y: np.ndarray
    tau: np.ndarray
    taup: np.ndarray
    taum: np.ndarray
    S_end: np.ndarray
    I_out: np.ndarray
    n_out: np.ndarray
    o_out: np.ndarray
    x_out: np.ndarray
    Y: np.ndarray
    W_out: np.ndarray
    stage_cost: float
    theta: float
    objective: float
    cost_terms: dict


def _cut_arrays(cuts):
    """把活跃割转成 (alpha, beta)；也接受 assemble 的 (alpha,beta) 元组。"""
    if cuts is None:
        return None
    if isinstance(cuts, tuple) and len(cuts) == 2:
        a, b = cuts
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float)
        if a.size == 0:
            return None
        if b.ndim != 2 or b.shape[0] != a.size:
            raise ValueError(f"割 beta 形状 {b.shape} 与 alpha {a.shape} 不匹配")
        return a, b
    if len(cuts) == 0:
        return None
    a = np.asarray([float(c.alpha) for c in cuts], dtype=float)
    b = np.asarray([np.asarray(c.beta, dtype=float) for c in cuts], dtype=float)
    if b.ndim != 2 or b.shape[0] != a.size:
        raise ValueError(f"割 beta 形状 {b.shape} 与 alpha {a.shape} 不匹配")
    return a, b


def heuristic_step(*, prob, t: int, x, scene, cfg,
                   cuts_next=None, rng=None) -> HeuristicResult:
    """固定状态/场景，用可行 SA 构造全部阶段决策；不调用 LP/MIP。

    参数
    ----
    prob : ProblemData
    t    : 真实阶段号（async 共享模型的 self.t 恒为 1，必须显式传）
    x    : 第 t 期入库状态
    scene: (z, Θ)
    cfg  : SolverConfig
    cuts_next : 阶段 t+1 的活跃割（Cut 列表或 (alpha,beta)）；空 = θ=0
    rng  : numpy Generator；None 时用固定种子，便于单测
    """
    t0 = time.perf_counter()
    Ly = prob.layout
    NR, G, L = Ly.N_R, Ly.G, Ly.L
    x = np.asarray(x, dtype=float).ravel().copy()
    if x.size != Ly.dim:
        raise ValueError(f"状态维度 {x.size} ≠ {Ly.dim}")
    if int(t) < 1:
        raise ValueError("前向启发式只用于真实阶段 t≥1")

    integer = bool(getattr(cfg, "integer_enabled", False))
    scope = str(getattr(cfg, "integer_scope", "none"))
    tier2 = integer and scope == "tier2_full"
    int_tol = max(float(getattr(cfg, "int_tol", 1e-6)), 1e-7)
    audit_tol = max(float(getattr(prob, "fleet_tol", 1e-6)), 1e-7)
    if rng is None:
        rng = np.random.default_rng(20261111)

    target_frac = float(getattr(cfg, "heur_target_frac", 1.0))
    sa_iters = int(getattr(cfg, "heur_sa_iters", 20))
    sa_t0_frac = float(getattr(cfg, "heur_sa_t0_frac", 0.02))
    sa_cooling = float(getattr(cfg, "heur_sa_cooling", 0.9))
    sa_step = float(getattr(cfg, "heur_sa_step", 1.0))
    if not (0.0 < target_frac <= 1.0):
        raise ValueError(f"heur_target_frac 非法：{target_frac}")
    if sa_iters < 0:
        raise ValueError(f"heur_sa_iters 非法：{sa_iters}")
    if not (0.0 < sa_cooling < 1.0):
        raise ValueError(f"heur_sa_cooling 非法：{sa_cooling}")
    if sa_step <= 0:
        raise ValueError(f"heur_sa_step 非法：{sa_step}")

    I_in = Ly.I(x).copy()
    o_in = Ly.o(x).copy()
    n_in = Ly.n(x).copy()
    q_in = Ly.q(x).copy()
    th = np.asarray(scene[1], dtype=float)
    if th.shape != (NR, G, G + 1):
        raise ValueError(f"Θ 形状 {th.shape} ≠ {(NR, G, G + 1)}")

    # 固定健康转移。Tier 2 用与模型 Tail 行一致的整数可行点。
    if tier2:
        n_int = np.rint(n_in)
        dev = float(np.max(np.abs(n_in - n_int))) if n_in.size else 0.0
        if dev > int_tol:
            raise HeuristicInfeasible(
                f"Tier 2 状态 n 非整数，偏差 {dev:.3e}")
        try:
            N = nearest_tail_flow(n_int, th).astype(float)
        except ValueError as exc:
            raise HeuristicInfeasible(f"nearest_tail_flow 失败：{exc}") from exc
    else:
        N = th * n_in[:, :, None]

    F_new = N[:, :, G].sum(axis=1)
    W_in = prob.N - n_in.sum(axis=1)
    D = np.maximum(W_in, 0.0) + F_new

    # 管道移位与固定量。
    o_out = np.empty((L - 1, NR), dtype=float)
    for l in range(L - 2):
        o_out[l] = o_in[l + 1]
    o_out[L - 2] = q_in
    pipe_sum = o_out.sum(axis=0)
    ub_state = Ly.upper(prob)
    qmax = np.asarray(prob.Qmax, dtype=float)
    imax = np.asarray(prob.Imax, dtype=float)

    # 初始被动调拨：按路线成本从低到高分配本地/跨仓供给。
    supply = I_in.copy()
    demand = D.copy()
    if integer:
        supply = np.floor(np.maximum(supply, 0.0) + 1e-9)
        demand = np.floor(np.maximum(demand, 0.0) + 1e-9)
    rem_sup = supply.copy()
    rem_dem = demand.copy()
    y0 = np.zeros((NR, NR), dtype=float)
    edges = sorted((float(prob.tc_route[r, m]), r, m)
                   for r in range(NR) for m in range(NR))
    for _cost, r, m in edges:
        if rem_dem[r] <= 1e-12 or rem_sup[m] <= 1e-12:
            continue
        take = min(float(rem_dem[r]), float(rem_sup[m]))
        if integer:
            take = math.floor(take + 1e-9)
        if take <= 0:
            continue
        y0[r, m] += take
        rem_dem[r] -= take
        rem_sup[m] -= take

    # 初始订货：OTC，先算 y0 后的库存位置，再补到目标容量。
    S0 = I_in - y0.sum(axis=0) + o_in[0]
    ip0 = S0 + pipe_sum
    target = target_frac * imax
    if integer:
        target = np.floor(target + 1e-9)
    slack0 = np.maximum(imax - ip0, 0.0)
    q0 = np.minimum(np.maximum(target - ip0, 0.0),
                    np.minimum(qmax, slack0))
    if integer:
        q0 = np.floor(q0 + 1e-9)
    if int(t) > int(prob.T) - int(prob.L):
        q0[:] = 0.0
    tau0 = np.zeros(NR, dtype=float)

    cut_ab = _cut_arrays(cuts_next)

    def theta_of(xo):
        if cut_ab is None:
            return 0.0
        a, b = cut_ab
        return float(np.max(a + b @ np.asarray(xo, dtype=float)))

    def build(q, y, tau):
        q = np.asarray(q, dtype=float).copy()
        y = np.asarray(y, dtype=float).copy()
        tau = np.asarray(tau, dtype=float).copy()
        if q.shape != (NR,) or y.shape != (NR, NR) or tau.shape != (NR,):
            raise HeuristicInfeasible("决策数组形状不合法")
        if np.any(~np.isfinite(q)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(tau)):
            raise HeuristicInfeasible("决策含非有限值")
        if np.any(q < -audit_tol) or np.any(y < -audit_tol):
            raise HeuristicInfeasible("q/y 为负")

        out_src = y.sum(axis=0)
        if np.any(out_src > I_in + audit_tol):
            raise HeuristicInfeasible("pas 违约")
        Y = y.sum(axis=1)
        if np.any(Y > D + audit_tol):
            raise HeuristicInfeasible("dem 需求上限违约")

        S_end = I_in - out_src + o_in[0]
        if np.any(S_end < -audit_tol):
            raise HeuristicInfeasible("S_end 为负")
        if np.any(S_end > imax + qmax + audit_tol):
            raise HeuristicInfeasible("S_end 超过变量上界")
        I_out = S_end + tau
        if np.any(I_out < -audit_tol):
            raise HeuristicInfeasible("I_out 为负")
        if np.any(I_out > imax + audit_tol):
            raise HeuristicInfeasible("I_out 超过 Imax")
        if abs(float(tau.sum())) > audit_tol:
            raise HeuristicInfeasible("tau_bal 违约")
        tau_src_rhs = S_end.sum() - S_end
        if np.any(tau > tau_src_rhs + audit_tol):
            raise HeuristicInfeasible("tau_src 违约")
        if np.any(tau > np.minimum(np.sum(imax + qmax), imax) + audit_tol):
            raise HeuristicInfeasible("taup 上界违约")

        ip = I_out + pipe_sum
        if np.any(ip > imax + audit_tol):
            raise HeuristicInfeasible("cap 在 q=0 时仍违约")
        # q 是当前期新订单；容量不足时取可行的最大量（并保持整数模式）。
        q_eff = np.minimum(np.maximum(q, 0.0), qmax)
        q_eff = np.minimum(q_eff, np.maximum(imax - ip, 0.0))
        if integer:
            q_eff = np.floor(q_eff + 1e-9)
        if np.any(q_eff > qmax + audit_tol):
            raise HeuristicInfeasible("q 超过 Qmax")

        n_out = np.zeros((NR, G), dtype=float)
        n_out[:, 0] = N[:, 0, 0] + Y
        for j in range(1, G):
            n_out[:, j] = N[:, :, j].sum(axis=1)
        if tier2:
            n_int_out = np.rint(n_out)
            if np.max(np.abs(n_out - n_int_out)) > int_tol:
                raise HeuristicInfeasible("Tier 2 n_out 非整数")
            n_out = n_int_out
        else:
            try:
                n_out = round_repair_n(
                    n_out, prob.N, prob.state_round_digits,
                    prob.state_repair_tol)
            except ValueError as exc:
                raise HeuristicInfeasible(f"n_out round/repair 失败：{exc}") from exc

        xo = np.zeros(Ly.dim, dtype=float)
        xo[:Ly.n_I] = I_out
        xo[Ly.off_o:Ly.off_o + Ly.n_o] = o_out.ravel()
        xo[Ly.off_n:Ly.off_n + Ly.n_n] = n_out.ravel()
        xo[Ly.q_slice()] = q_eff
        if np.any(xo < -audit_tol) or np.any(xo > ub_state + audit_tol):
            raise HeuristicInfeasible("期末状态越界")
        if np.any(n_out.sum(axis=1) > prob.N + audit_tol):
            raise HeuristicInfeasible("n_out 行和超过机队")
        if np.any(n_out < -audit_tol):
            raise HeuristicInfeasible("n_out 为负")

        dem_lhs = Y + N[:, :, :G].sum(axis=(1, 2))
        if np.any(dem_lhs > prob.N + prob.fleet_tol + audit_tol):
            raise HeuristicInfeasible("dem 违约")
        if np.any(n_in.sum(axis=1) > prob.N + prob.fleet_tol + audit_tol):
            raise HeuristicInfeasible("fleet 违约")
        if np.any(np.abs(N.sum(axis=2) - n_in) > audit_tol):
            raise HeuristicInfeasible("N 行和违约")

        if tier2:
            ct = tail_cdf(th)
            for r in range(NR):
                for i in range(G):
                    for j in range(i + 1, G + 1):
                        val = float(N[r, i, j:].sum())
                        center = float(ct[r, i, j]) * float(n_in[r, i])
                        if abs(val - center) > 0.5 + audit_tol:
                            raise HeuristicInfeasible(
                                f"Tier 2 尾取整越界 r={r},i={i},j={j}")

        if integer:
            mask = Ly.integer_mask(scope)
            vals = [xo[mask], q_eff, y, tau, S_end, I_out]
            for arr in vals:
                if arr.size and np.max(np.abs(np.asarray(arr, float)
                                              - np.rint(arr))) > int_tol:
                    raise HeuristicInfeasible("整数模式含非整数决策/状态")

        theta = theta_of(xo)
        taup = np.maximum(tau, 0.0)
        taum = np.maximum(-tau, 0.0)
        order = float(prob.C) * float(q_in.sum())
        holding = 0.5 * float(prob.h) * float(I_in.sum() + I_out.sum())
        downtime = float(prob.Cs) * float(prob.N.sum() - n_in.sum())
        active = 0.5 * float(prob.atc) * float(taup.sum() + taum.sum())
        passive = float(np.sum(prob.tc_route * y))
        stage_cost = order + holding + downtime + active + passive
        W_out = prob.N - n_out.sum(axis=1)
        return _Candidate(
            q=q_eff, y=y, tau=tau, taup=taup, taum=taum, S_end=S_end,
            I_out=I_out, n_out=n_out, o_out=o_out.copy(), x_out=xo,
            Y=Y, W_out=W_out,
            stage_cost=float(stage_cost), theta=float(theta),
            objective=float(stage_cost + theta),
            cost_terms={"order": order, "holding": holding,
                        "downtime": downtime, "active": active,
                        "passive": passive,
                        "theta": float(theta)})

    initial = build(q0, y0, tau0)
    # 第二个便宜种子：q=0。终端附近 LP 常取 q=0，单靠逐步移动很难退到 0。
    try:
        q_zero = build(np.zeros(NR, dtype=float), y0, tau0)
    except HeuristicInfeasible:
        q_zero = None
    if q_zero is not None and q_zero.objective < initial.objective - 1e-12:
        initial = q_zero

    # SA：只在 q/y/tau 上做可行邻域移动；S_end 等始终由等式派生。
    if sa_iters <= 0:
        elapsed = time.perf_counter() - t0
        return _result(initial, N, F_new, W_in, 0, 0, elapsed)

    step = 1.0 if integer else float(sa_step)
    cur = initial
    best = initial
    accepted = 0
    temp = max(1e-12, sa_t0_frac * max(1.0, abs(best.objective)))
    max_q_move = int(t) <= int(prob.T) - int(prob.L)

    for _ in range(sa_iters):
        kind = int(rng.integers(0, 5))
        q = cur.q.copy()
        y = cur.y.copy()
        tau = cur.tau.copy()
        proposed = False

        if kind == 0 and max_q_move:
            r = int(rng.integers(0, NR))
            ub = min(float(qmax[r]),
                     max(float(imax[r] - (cur.I_out[r] + pipe_sum[r])), 0.0))
            mode = int(rng.integers(0, 3))
            if mode == 0:
                q[r] = 0.0
            elif mode == 1:
                q[r] = ub
            elif integer:
                q[r] = float(rng.integers(0, max(1, int(math.floor(ub + 1e-9)) + 1)))
            else:
                q[r] = float(rng.uniform(0.0, ub)) if ub > 0 else 0.0
            proposed = True
        elif kind == 1:
            m = int(rng.integers(0, NR))
            r1, r2 = (int(v) for v in rng.choice(NR, size=2, replace=False))
            d = step
            if y[r1, m] >= d - 1e-12 and cur.Y[r2] + d <= D[r2] + 1e-9:
                y[r1, m] -= d
                y[r2, m] += d
                proposed = True
        elif kind == 2:
            r = int(rng.integers(0, NR))
            m1, m2 = (int(v) for v in rng.choice(NR, size=2, replace=False))
            d = step
            spare = I_in[m2] - y[:, m2].sum()
            if y[r, m1] >= d - 1e-12 and spare >= d - 1e-12:
                y[r, m1] -= d
                y[r, m2] += d
                proposed = True
        elif kind == 3:
            r = int(rng.integers(0, NR))
            m = int(rng.integers(0, NR))
            d = step
            if rng.random() < 0.5 and y[r, m] >= d - 1e-12:
                y[r, m] -= d
                proposed = True
            elif (cur.Y[r] + d <= D[r] + 1e-9
                  and y[:, m].sum() + d <= I_in[m] + audit_tol):
                y[r, m] += d
                proposed = True
        elif kind == 4:
            r1, r2 = (int(v) for v in rng.choice(NR, size=2, replace=False))
            d = step
            tau[r1] -= d
            tau[r2] += d
            proposed = True

        if not proposed:
            temp = max(1e-12, temp * sa_cooling)
            continue
        try:
            cand = build(q, y, tau)
        except HeuristicInfeasible:
            temp = max(1e-12, temp * sa_cooling)
            continue
        delta = float(cand.objective - cur.objective)
        if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
            cur = cand
            accepted += 1
            if cand.objective < best.objective - 1e-12:
                best = cand
        temp = max(1e-12, temp * sa_cooling)

    elapsed = time.perf_counter() - t0
    return _result(best, N, F_new, W_in, sa_iters, accepted, elapsed)


def _result(c, N, F_new, W_in, moves, accepted, elapsed) -> HeuristicResult:
    return HeuristicResult(
        x_out=np.asarray(c.x_out, dtype=float).copy(),
        stage_cost=float(c.stage_cost),
        objective=float(c.objective),
        cost_terms=dict(c.cost_terms),
        q_out=np.asarray(c.q, dtype=float).copy(),
        y=np.asarray(c.y, dtype=float).copy(),
        tau=np.asarray(c.tau, dtype=float).copy(),
        taup=np.asarray(c.taup, dtype=float).copy(),
        taum=np.asarray(c.taum, dtype=float).copy(),
        S_end=np.asarray(c.S_end, dtype=float).copy(),
        I_out=np.asarray(c.I_out, dtype=float).copy(),
        o_out=np.asarray(c.o_out, dtype=float).copy(),
        n_out=np.asarray(c.n_out, dtype=float).copy(),
        N=np.asarray(N, dtype=float).copy(),
        F_new=np.asarray(F_new, dtype=float).copy(),
        W_in=np.asarray(W_in, dtype=float).copy(),
        W_out=np.asarray(c.W_out, dtype=float).copy(),
        sa_moves=int(moves), sa_accepted=int(accepted),
        sa_time_s=float(elapsed))
