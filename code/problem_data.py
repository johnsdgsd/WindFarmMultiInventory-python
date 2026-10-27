# problem_data.py — 等级模型问题数据：参数、Poisson 转移矩阵、Dirichlet 场景池。
"""等级模型（纯 LP）的问题数据。

- 运行等级 0..G-1，等级 G 为失效态；
- 等级增量 D ~ Poisson(lambda)，lambda = lambda0 * exp(beta·(z-zbar))；
- P 为 G×(G+1) 上三角，按 p_digits 四舍五入（对角吸收舍入误差）；
- 场景池每条 scene=(z, Θ)，Θ 为 (N_R,G,G+1)，逐行 Dirichlet(α=sP)；
- 场景池全阶段共享，每期独立抽场景号；因此每期转移矩阵不同但同分布。

旧月龄/Weibull/Beta 函数保留在本文件末尾的 legacy 区，仅用于历史脚本导入，
不参与等级模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import ProblemConfig
from .state_layout import StateLayout

COV_NAMES = ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")


# ---------------------------------------------------------------- 转移矩阵
def poisson_pmf(lam: float, k: int) -> float:
    """Poisson 质量函数 p_k = e^{-λ} λ^k / k!。"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def transition_matrix(lam: float, G: int, p_digits: int | None = 3) -> np.ndarray:
    """G×(G+1) 上三角转移矩阵。

    精确：P[i,j]=p_{j-i} (j<G)，P[i,G]=1-Σ_{k=0}^{G-1-i}p_k，j<i 为 0。
    若 p_digits 非 None：对 j>i 的分量（含失效列）四舍五入；对角
    P[i,i]=1-Σ_{j>i}P[i,j] 吸收舍入误差。若舍入后的非对角和 >1，先
    归一化非对角分量（对角置 0），保证非负、行和 1。
    """
    lam = float(lam)
    if lam < 0:
        raise ValueError(f"lambda 必须非负，实际 {lam}")
    p = np.array([poisson_pmf(lam, k) for k in range(G + 1)], dtype=float)
    P = np.zeros((G, G + 1), dtype=float)
    for i in range(G):
        off = []
        for j in range(i + 1, G + 1):
            if j < G:
                off.append(p[j - i])
            else:
                # P(D >= G-i)
                off.append(1.0 - float(p[: G - i].sum()))
        off = np.asarray(off, dtype=float)
        if p_digits is not None:
            off = np.round(off, int(p_digits))
            s = float(off.sum())
            if s > 1.0 + 1e-12:
                off = off / s
        P[i, i + 1:] = off
        P[i, i] = 1.0 - float(off.sum())
    # 数值清理：负零/极小负值；行和严格 1
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    if np.any(rs <= 0):
        raise ValueError("转移矩阵出现空行")
    P = P / rs
    return P


def cumulative_failure_prob(P: np.ndarray, T: int, init_mix) -> float:
    """按离散马尔可夫链计算 T 期内累计失效概率（P 的失败列）。"""
    G = P.shape[0]
    Q = P[:, :G]
    f = P[:, G]
    v = np.asarray(init_mix, dtype=float).copy()
    acc = 0.0
    for _ in range(int(T)):
        acc += float(v @ f)
        v = v @ Q
    return float(acc)


def expected_first_passage(P: np.ndarray, init_mix) -> float:
    """从初始等级分布出发的期望首达失效时间（月）。"""
    G = P.shape[0]
    Q = P[:, :G]
    M = np.linalg.solve(np.eye(G) - Q, np.ones(G))
    return float(np.asarray(init_mix, dtype=float) @ M)


# ---------------------------------------------------------------- PHM 因子
def _cov_factor(beta_cov, z_bar, z_r) -> float:
    """修正因子 f_r = exp(Σ_k β_k(z_k − z̄_k))；乘在 λ0 上。"""
    return float(np.exp(np.dot(beta_cov, np.asarray(z_r) - np.asarray(z_bar))))


# ---------------------------------------------------------------- 场景池
def sample_pool(lambda0: float, beta_cov, z_bar, cov, N_R: int, G: int,
                dirichlet_s: float, p_digits: int, pool_size: int,
                seed: int) -> list:
    """(z, Θ) 场景池。

    每条 scene：z 为 (3,N_R)；Θ 为 (N_R,G,G+1)。Θ 逐行 Dirichlet(sP)，
    只在 P>0 的支撑上抽样；结构零与 P=0 位置为 0。
    """
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(pool_size):
        z = np.empty((3, N_R))
        z[0] = rng.normal(cov["mu_T"], cov["sigma_T"], size=N_R)
        z[1] = rng.poisson(cov["lambda"], size=N_R)
        z[2] = 100.0 * rng.beta(cov["alpha_L"], cov["beta_L"], size=N_R)
        theta = np.zeros((N_R, G, G + 1))
        for r in range(N_R):
            f = _cov_factor(beta_cov, z_bar, z[:, r])
            P = transition_matrix(lambda0 * f, G, p_digits)
            for i in range(G):
                cols = [j for j in range(i, G + 1) if P[i, j] > 0.0]
                if not cols:
                    raise ValueError(f"P 第 {i} 行无正支撑")
                alpha = np.array([dirichlet_s * P[i, j] for j in cols])
                if len(cols) == 1:
                    vals = np.array([1.0])
                else:
                    vals = rng.dirichlet(alpha)
                theta[r, i, cols] = vals
        # 行和/结构零自检（构造期硬门槛）
        row_sum = theta.sum(axis=2)
        if np.max(np.abs(row_sum - 1.0)) > 1e-12:
            raise ValueError("Θ 行和偏离 1（Dirichlet 采样错误）")
        for i in range(G):
            if np.any(theta[:, i, :i] != 0.0):
                raise ValueError("Θ 下三角必须为 0")
        pool.append((z, theta))
    return pool


def _pool_mean_theta(pool, N_R: int, G: int) -> np.ndarray:
    acc = np.zeros((N_R, G, G + 1))
    for _, th in pool:
        acc += th
    return acc / len(pool)


def _pool_mean_P(pool, lambda0: float, beta_cov, z_bar, G: int,
                 p_digits: int) -> np.ndarray:
    acc = np.zeros((G, G + 1))
    for z, _ in pool:
        for r in range(z.shape[1]):
            f = _cov_factor(beta_cov, z_bar, z[:, r])
            acc += transition_matrix(lambda0 * f, G, p_digits)
    return acc / (len(pool) * z.shape[1])


# ---------------------------------------------------------------- 数据结构
@dataclass
class ProblemData:
    # ---- 维度 ----
    T: int
    G: int
    N_R: int
    L: int
    # ---- 机队与初值 ----
    N: np.ndarray
    n0: np.ndarray            # (N_R,G)，可为小数（初始等级分布）
    I0: np.ndarray
    o0: np.ndarray            # (L-1,N_R)
    init_mix: np.ndarray      # (G,)
    # ---- 成本 ----
    C: float
    h: float
    Cs: float
    atc: float
    tc: float
    # ---- 容量 ----
    Qmax: np.ndarray
    Imax: np.ndarray
    fleet_tol: float          # fleet/dem 右端的数值容差（台）
    state_round_digits: int   # 状态流转时 n 块保留的小数位
    state_repair_tol: float   # 只修正不超过该值的 Σn 超额（台）
    # ---- 被动调拨优先级 ----
    priority: dict
    tc_route: np.ndarray
    # ---- 退化/转移 ----
    lambda0: float
    dirichlet_s: float
    p_digits: int
    beta_cov: np.ndarray
    z_bar: np.ndarray
    cov: dict
    P_ref: np.ndarray         # λ0 下的 G×(G+1) 参考矩阵
    # ---- 场景池 ----
    pool: list
    seed: int
    # ---- 状态布局 ----
    layout: StateLayout = field(default=None, repr=False)

    @property
    def state_dim(self) -> int:
        return self.layout.dim

    def initial_state(self) -> np.ndarray:
        x = np.zeros(self.layout.dim)
        x[self.layout.off_I:self.layout.off_I + self.layout.n_I] = self.I0
        for l in range(self.L - 1):
            for r in range(self.N_R):
                x[self.layout.ix_o(l, r)] = self.o0[l, r]
        for r in range(self.N_R):
            for i in range(self.G):
                x[self.layout.ix_n(r, i)] = self.n0[r, i]
        return x


def default_priority(N_R: int) -> dict:
    """默认调拨优先级表。N_R=5 时沿用 MutiInventory 的实测表。"""
    if N_R == 5:
        return {1: [1, 2, 3, 4, 5], 2: [2, 1, 3, 4, 5], 3: [3, 1, 2, 4, 5],
                4: [4, 1, 2, 3, 5], 5: [5, 1, 2, 3, 4]}
    return {r: [r] + [m for m in range(1, N_R + 1) if m != r]
            for r in range(1, N_R + 1)}


def make_tc_route(priority: dict, tc: float, N_R: int) -> np.ndarray:
    """路线差异化被动调拨成本：m 在 r 优先级表中的位置 pos → (pos−1)·tc。"""
    route = np.zeros((N_R, N_R))
    for r in range(1, N_R + 1):
        for pos, m in enumerate(priority[r], start=1):
            route[r - 1, m - 1] = (pos - 1) * tc
    return route


def DefineProblem(cfg: ProblemConfig) -> ProblemData:
    """从 ProblemConfig 构造问题数据（等级模型）。"""
    T, G, N_R, L = cfg.T, cfg.G, cfg.N_R, cfg.L
    N = np.asarray(cfg.N, dtype=np.int64)
    Qmax = np.asarray(cfg.Qmax, dtype=np.int64)
    Imax = np.asarray(cfg.Imax, dtype=np.int64)
    init_mix = np.asarray(cfg.init_mix, dtype=float)
    I0 = np.asarray(cfg.I0, dtype=float)
    beta_cov = np.asarray(cfg.beta_cov, dtype=float)
    z_bar = np.asarray(cfg.z_bar, dtype=float)
    cov = dict(cfg.cov)

    if len(init_mix) != G:
        raise ValueError(f"init_mix 长度 {len(init_mix)} ≠ G={G}")
    if np.any(init_mix < 0) or abs(float(init_mix.sum()) - 1.0) > 1e-8:
        raise ValueError(f"init_mix 必须非负且和为 1，实际 {init_mix.tolist()}")
    if len(I0) != N_R or np.any(I0 < 0):
        raise ValueError(f"I0 必须是非负、长度 {N_R} 的列表，实际 {I0.tolist()}")
    if np.any(I0 > Imax):
        raise ValueError(f"I0 不得超过 Imax：I0={I0.tolist()}, Imax={Imax.tolist()}")

    priority = default_priority(N_R) if cfg.priority is None else cfg.priority
    tc_route = make_tc_route(priority, cfg.tc, N_R)

    # 初始等级分布：每仓 N_r * init_mix
    n0 = np.outer(N.astype(float), init_mix)
    o0 = np.zeros((L - 1, N_R), dtype=float)
    for r in range(N_R):
        if abs(float(n0[r].sum()) - float(N[r])) > 1e-8:
            raise ValueError(f"仓库 {r+1} 初始等级分布和 ≠ N={N[r]}")

    P_ref = transition_matrix(cfg.lambda0, G, cfg.p_digits)
    pool = sample_pool(cfg.lambda0, beta_cov, z_bar, cov, N_R, G,
                       cfg.dirichlet_s, cfg.p_digits,
                       cfg.pool_size, cfg.seed)

    return ProblemData(
        T=T, G=G, N_R=N_R, L=L, N=N, n0=n0, I0=I0, o0=o0,
        init_mix=init_mix,
        C=cfg.C, h=cfg.h, Cs=cfg.Cs, atc=cfg.atc, tc=cfg.tc,
        Qmax=Qmax, Imax=Imax, fleet_tol=cfg.fleet_tol,
        state_round_digits=cfg.state_round_digits,
        state_repair_tol=cfg.state_repair_tol,
        priority=priority, tc_route=tc_route,
        lambda0=cfg.lambda0, dirichlet_s=cfg.dirichlet_s,
        p_digits=cfg.p_digits, beta_cov=beta_cov, z_bar=z_bar,
        cov=cov, P_ref=P_ref, pool=pool, seed=cfg.seed,
        layout=StateLayout(N_R, G, L),
    )


# ---------------------------------------------------------------- 序列化
def to_json(prob: ProblemData) -> dict:
    return {
        "model": "markov_grade",
        "schema": 1,
        "T": prob.T, "G": prob.G, "N_R": prob.N_R, "L": prob.L,
        "N": prob.N.tolist(), "n0": prob.n0.tolist(),
        "I0": prob.I0.tolist(), "o0": prob.o0.tolist(),
        "init_mix": prob.init_mix.tolist(),
        "C": prob.C, "h": prob.h, "Cs": prob.Cs, "atc": prob.atc, "tc": prob.tc,
        "Qmax": prob.Qmax.tolist(), "Imax": prob.Imax.tolist(),
        "fleet_tol": prob.fleet_tol,
        "state_round_digits": prob.state_round_digits,
        "state_repair_tol": prob.state_repair_tol,
        "priority": {str(k): list(v) for k, v in prob.priority.items()},
        "tc_route": prob.tc_route.tolist(),
        "lambda0": prob.lambda0, "dirichlet_s": prob.dirichlet_s,
        "p_digits": prob.p_digits,
        "beta_cov": prob.beta_cov.tolist(), "z_bar": prob.z_bar.tolist(),
        "cov": prob.cov, "P_ref": prob.P_ref.tolist(),
        "seed": prob.seed,
        "pool": [[z.tolist(), th.tolist()] for z, th in prob.pool],
    }


def from_json(d: dict) -> ProblemData:
    if d.get("model") != "markov_grade":
        raise ValueError("不是等级模型的 problem_data.json；旧月龄结果请重新训练")
    N_R, G, L = int(d["N_R"]), int(d["G"]), int(d["L"])
    return ProblemData(
        T=int(d["T"]), G=G, N_R=N_R, L=L,
        N=np.asarray(d["N"], dtype=np.int64),
        n0=np.asarray(d["n0"], dtype=float),
        I0=np.asarray(d["I0"], dtype=float),
        o0=np.asarray(d["o0"], dtype=float),
        init_mix=np.asarray(d["init_mix"], dtype=float),
        C=float(d["C"]), h=float(d["h"]), Cs=float(d["Cs"]),
        atc=float(d["atc"]), tc=float(d["tc"]),
        Qmax=np.asarray(d["Qmax"], dtype=np.int64),
        Imax=np.asarray(d["Imax"], dtype=np.int64),
        fleet_tol=float(d["fleet_tol"]),
        state_round_digits=int(d["state_round_digits"]),
        state_repair_tol=float(d["state_repair_tol"]),
        priority={int(k): list(v) for k, v in d["priority"].items()},
        tc_route=np.asarray(d["tc_route"], dtype=float),
        lambda0=float(d["lambda0"]), dirichlet_s=float(d["dirichlet_s"]),
        p_digits=int(d["p_digits"]),
        beta_cov=np.asarray(d["beta_cov"], dtype=float),
        z_bar=np.asarray(d["z_bar"], dtype=float),
        cov={k: float(v) for k, v in d["cov"].items()},
        P_ref=np.asarray(d["P_ref"], dtype=float),
        pool=[(np.asarray(z, dtype=float), np.asarray(th, dtype=float))
              for z, th in d["pool"]],
        seed=int(d["seed"]),
        layout=StateLayout(N_R, G, L),
    )


# ---------------------------------------------------------------- 摘要/自检
def print_summary(prob: ProblemData, out=print) -> None:
    G = prob.G
    out("=" * 64)
    out("问题数据摘要（等级模型，纯 LP）")
    out("=" * 64)
    out(f"T = {prob.T}  G = {G}  N_R = {prob.N_R}  L = {prob.L}  "
        f"pool = {len(prob.pool)}  seed = {prob.seed}  状态维 = {prob.state_dim}")
    out(f"机队 N = {prob.N.tolist()}  总数 = {int(prob.N.sum())}  "
        f"init_mix = {prob.init_mix.tolist()}  I0 = {prob.I0.tolist()}  "
        f"fleet_tol = {prob.fleet_tol:g}  "
        f"state_round_digits = {prob.state_round_digits}  "
        f"state_repair_tol = {prob.state_repair_tol:g}")
    out(f"成本: C={prob.C}  h={prob.h}  Cs={prob.Cs}  atc={prob.atc}  tc={prob.tc}")
    out(f"λ0={prob.lambda0}  dirichlet_s={prob.dirichlet_s}  "
        f"p_digits={prob.p_digits}")
    out(f"PHM: beta_cov={prob.beta_cov.tolist()}  z_bar={prob.z_bar.tolist()}")
    c = prob.cov
    out(f"协变量: 油温 N({c['mu_T']}, {c['sigma_T']}²)  "
        f"启停 Poisson({c['lambda']})  满负荷 100·Beta({c['alpha_L']}, {c['beta_L']}) (%)")
    out("P_ref（λ0 下，四舍五入后）:")
    for i in range(G):
        out("  " + " ".join(f"{prob.P_ref[i, j]:.6g}" for j in range(G + 1))
            + f"   sum={prob.P_ref[i].sum():.6f}")
    P_exact = transition_matrix(prob.lambda0, G, None)
    out(f"P_ref vs 精确矩阵: max|Δ| = {np.max(np.abs(prob.P_ref - P_exact)):.3e}")

    pT = cumulative_failure_prob(prob.P_ref, prob.T, prob.init_mix)
    pT_exact = cumulative_failure_prob(P_exact, prob.T, prob.init_mix)
    mttf = expected_first_passage(prob.P_ref, prob.init_mix)
    total_N = float(prob.N.sum())
    years = prob.T / 12.0
    out(f"T={prob.T} 期累计失效: 3 位小数 {pT:.4%}（精确 {pT_exact:.4%}）")
    out(f"  N={total_N:.0f} 时约 {pT*total_N/years:.1f} 台/年（按 T/12 折算）；"
        f"首达约 {mttf:.1f} 月")

    # 池自检
    theta_bar = _pool_mean_theta(prob.pool, prob.N_R, G)
    Pz_bar = _pool_mean_P(prob.pool, prob.lambda0, prob.beta_cov, prob.z_bar,
                          G, prob.p_digits)
    max_row = 0.0
    max_zero = 0.0
    for z, th in prob.pool:
        max_row = max(max_row, float(np.max(np.abs(th.sum(axis=2) - 1.0))))
        for i in range(G):
            if i:
                max_zero = max(max_zero, float(np.max(np.abs(th[:, i, :i]))))
    dev = np.abs(theta_bar - Pz_bar[None, :, :])
    mask = Pz_bar >= 0.01
    rel = (float(np.max(dev[:, mask] / Pz_bar[mask][None, :]))
           if mask.any() else 0.0)
    out(f"池 Θ 行和 max|Δ|={max_row:.2e}  结构零 max|Δ|={max_zero:.2e}")
    out(f"E[Θ] vs E_z[P(z)]: max|Δ|={dev.max():.4f}  "
        f"P≥0.01 处 max 相对误差={rel:.2%}")

    # 协变量因子中心化
    rng = np.random.default_rng(prob.seed + 1000)
    n_mc = 100_000
    e_beta_L = float(np.mean(np.exp(prob.beta_cov[2] * (
        100.0 * rng.beta(c["alpha_L"], c["beta_L"], size=n_mc) - prob.z_bar[2]))))
    e_f_theory = (np.exp(prob.beta_cov[0] ** 2 * c["sigma_T"] ** 2 / 2)
                  * np.exp(c["lambda"] * (np.exp(prob.beta_cov[1]) - 1)
                           - prob.beta_cov[1] * c["lambda"])
                  * e_beta_L)
    e_f_pool = float(np.mean([_cov_factor(prob.beta_cov, prob.z_bar, z[:, r])
                              for z, _ in prob.pool for r in range(prob.N_R)]))
    dev_f = abs(e_f_pool - e_f_theory) / e_f_theory * 100
    out(f"修正因子 E[f]: 理论 {e_f_theory:.4f}  池内 {e_f_pool:.4f}  偏差 {dev_f:.2f}%")
    out("=" * 64)


# ---------------------------------------------------------------- Legacy（旧月龄模型）
# 以下函数只保留给历史脚本导入；等级模型不使用。
def weibull_hazard(a: int, m: float, eta: float) -> float:
    """旧月龄模型：Weibull 基准失效率。"""
    return (m / eta) * (a / eta) ** (m - 1)


def phm_hazard(a: int, m: float, eta: float, beta_cov, z, z_bar) -> float:
    """旧月龄模型：PHM 修正失效率。"""
    return weibull_hazard(a, m, eta) * float(
        np.exp(np.dot(beta_cov, np.asarray(z) - np.asarray(z_bar))))


def failure_prob(h: float) -> float:
    """旧月龄模型：p = 1 − exp(−h)。"""
    return 1.0 - float(np.exp(-h))


def beta_second(p: float) -> float:
    """旧月龄模型：Beta(2,b) 一阶矩匹配。"""
    return 2.0 * (1.0 - p) / p if p > 0 else float("inf")
