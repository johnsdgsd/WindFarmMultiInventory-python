# problem_data.py — 问题数据：参数、DefineProblem 默认值、威布尔/PHM 失效概率、
# (z, θ) 联合场景池、自检打印。与 Julia 版 code/problem_data.jl 逐项对应。
"""对照 docs/问题.md：

  §2    威布尔基准失效率 h0(a) = (m/η)(a/η)^(m−1)，不取月龄中点
  §2.2  中心化三协变量比例风险模型（docs/比例风险模型参数示例.md）
  §3.4.1 Beta(2, b) 一阶矩匹配 E[θ] = p
  §3.3.3 约束编号 (1)..(8) 由 code/model.py 注释引用，改编号必须两处同步

池按「各期独立同分布、全阶段共享」预采样：一条场景 = (z, θ)，z 为 3×N_R 协变量
矩阵（每仓库独立抽油温/启停/满负荷），θ 为 N_R×A 失效比例矩阵。θ[r,0]（月龄 0）
恒 0、θ[r,A−1]（到寿）恒 1。z 保留在场景里，供仿真后的策略分析使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ProblemConfig
from .state_layout import StateLayout

COV_NAMES = ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")

# 池矩匹配的 z 检验在小池下不可靠（见 print_summary 自检 2 的说明）
Z_THRESHOLD = 6.0
MIN_POOL_FOR_Z_TEST = 60


def weibull_hazard(a: int, m: float, eta: float) -> float:
    """威布尔基准失效率 h0(a) = (m/η)(a/η)^(m−1)，a = 0..A−1。

    m ≥ 1 时 h0(0) = 0（新件当月不失效）；m < 1 给出 h0(0) = ∞，无意义，构造时拒绝。
    """
    return (m / eta) * (a / eta) ** (m - 1)


def phm_hazard(a: int, m: float, eta: float, beta_cov, z, z_bar) -> float:
    """比例风险修正后的月龄失效率 h = h0(a)·exp(Σ β_k(z_k − z̄_k))。"""
    return weibull_hazard(a, m, eta) * float(np.exp(np.dot(beta_cov, np.asarray(z) - np.asarray(z_bar))))


def failure_prob(h: float) -> float:
    """周期内失效概率 p = 1 − exp(−h·Δt)，Δt = 1。"""
    return 1.0 - float(np.exp(-h))


def beta_second(p: float) -> float:
    """Beta 第二参数：固定 a = 2、一阶矩匹配 E[θ] = p。p = 0 时返回 Inf（θ 恒 0）。"""
    return 2.0 * (1.0 - p) / p if p > 0 else float("inf")


@dataclass
class ProblemData:
    # ---- 维度 ----
    T: int
    A: int
    N_R: int
    L: int
    # ---- 机队与初值 ----
    N: np.ndarray
    n0: np.ndarray            # (N_R, A) 期初月龄分布
    I0: np.ndarray
    o0: np.ndarray            # (L−1, N_R) 期初在途
    # ---- 成本 ----
    C: float
    h: float
    Cs: float
    atc: float
    tc: float
    # ---- 容量 ----
    Qmax: np.ndarray
    Imax: np.ndarray
    # ---- 被动调拨优先级（凸化：优先级 → 路线成本）----
    priority: dict
    tc_route: np.ndarray
    # ---- 退化与比例风险 ----
    m: float
    eta: float
    a_beta: float
    beta_cov: np.ndarray
    z_bar: np.ndarray
    cov: dict
    # ---- 派生 ----
    h0: np.ndarray
    p_bar: np.ndarray
    # ---- 噪声 ----
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
        for a in range(self.A):
            for r in range(self.N_R):
                x[self.layout.ix_n(r, a)] = self.n0[r, a]
        return x


def default_priority(N_R: int) -> dict:
    """默认调拨优先级表。N_R=5 时沿用 MutiInventory 的实测表，其余自动生成。"""
    if N_R == 5:
        return {1: [1, 2, 3, 4, 5], 2: [2, 1, 3, 4, 5], 3: [3, 1, 2, 4, 5],
                4: [4, 1, 2, 3, 5], 5: [5, 1, 2, 3, 4]}
    return {r: [r] + [m for m in range(1, N_R + 1) if m != r] for r in range(1, N_R + 1)}


def make_tc_route(priority: dict, tc: float, N_R: int) -> np.ndarray:
    """路线差异化被动调拨成本：m 在 r 优先级表中的位置 pos → (pos−1)·tc。

    本地满足成本 0、第 k 顺位外仓 (k−1)·tc —— 缺库存时 LP 自动先取高优先级仓库，
    把问题.md (5) 的二元序贯逻辑凸化（docs/建模与实现.md §4）。
    """
    route = np.zeros((N_R, N_R))
    for r in range(1, N_R + 1):
        # start=1 对齐 Julia 的 1-based enumerate：pos=1 是本地（成本 0）
        for pos, m in enumerate(priority[r], start=1):
            route[r - 1, m - 1] = (pos - 1) * tc
    return route


def _cov_factor(beta_cov, z_bar, z_r) -> float:
    """修正因子 f_r = exp(Σ_k β_k(z_k − z̄_k))（中心化 PHM）。"""
    return float(np.exp(np.dot(beta_cov, np.asarray(z_r) - np.asarray(z_bar))))


def sample_pool(h0, beta_cov, z_bar, cov, N_R, A, pool_size, seed) -> list:
    """(z, θ) 联合场景池。每条场景：每仓库独立抽 z，算修正因子 → p(z) → 抽 θ。

    θ[r,a] ~ Beta(2, b(p))，p = 1 − exp(−h0(a)·f_r)；θ[:,0] ≡ 0（新件不失效）、
    θ[:,A−1] ≡ 1（到寿全离场）。独立 Generator，seed 固定可复现。
    """
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(pool_size):
        z = np.empty((3, N_R))
        theta = np.zeros((N_R, A))
        theta[:, A - 1] = 1.0
        z[0] = rng.normal(cov["mu_T"], cov["sigma_T"], size=N_R)
        z[1] = rng.poisson(cov["lambda"], size=N_R)
        z[2] = 100.0 * rng.beta(cov["alpha_L"], cov["beta_L"], size=N_R)
        for r in range(N_R):
            f = _cov_factor(beta_cov, z_bar, z[:, r])
            for a in range(1, A - 1):
                p = failure_prob(h0[a] * f)
                theta[r, a] = rng.beta(2.0, beta_second(p))
        pool.append((z, theta))
    return pool


def pool_mean_p(pool, h0, beta_cov, z_bar, N_R, A) -> np.ndarray:
    """池内平均失效概率 p_bar[r,a] = E_z[p(z)]（自检与打印用）。"""
    p_bar = np.zeros((N_R, A))
    for z, _ in pool:
        for r in range(N_R):
            f = _cov_factor(beta_cov, z_bar, z[:, r])
            for a in range(A):
                p_bar[r, a] += failure_prob(h0[a] * f)
    return p_bar / len(pool)


def DefineProblem(cfg: ProblemConfig) -> ProblemData:
    """从 ProblemConfig 构造问题数据。参数校验全在 ProblemConfig 里（code/config.py）。"""
    T, A, N_R, L = cfg.T, cfg.A, cfg.N_R, cfg.L
    N = np.asarray(cfg.N, dtype=np.int64)
    Qmax = np.asarray(cfg.Qmax, dtype=np.int64)
    Imax = np.asarray(cfg.Imax, dtype=np.int64)
    beta_cov = np.asarray(cfg.beta_cov, dtype=float)
    z_bar = np.asarray(cfg.z_bar, dtype=float)
    cov = dict(cfg.cov)

    priority = default_priority(N_R) if cfg.priority is None else cfg.priority
    tc_route = make_tc_route(priority, cfg.tc, N_R)

    n0 = np.zeros((N_R, A), dtype=np.int64)
    if cfg.spread_init:
        for r in range(N_R):                      # 每仓独立摊平（异质机队）
            base, rem = divmod(int(N[r]), A)
            n0[r, :] = base
            n0[r, 0] += rem
    else:
        n0[:, 0] = N
    I0 = np.zeros(N_R, dtype=np.int64)
    o0 = np.zeros((L - 1, N_R), dtype=np.int64)

    h0 = np.array([weibull_hazard(a, cfg.m, cfg.eta) for a in range(A)])

    for r in range(N_R):
        if int(n0[r].sum()) != int(N[r]):
            raise ValueError(f"仓库 {r+1} 期初运行风机数 {int(n0[r].sum())} ≠ 总数 N={int(N[r])}")
        if I0[r] > Imax[r]:
            raise ValueError(f"仓库 {r+1} 期初库存 {I0[r]} 超过容量 Imax={Imax[r]}")

    pool = sample_pool(h0, beta_cov, z_bar, cov, N_R, A, cfg.pool_size, cfg.seed)
    p_bar = pool_mean_p(pool, h0, beta_cov, z_bar, N_R, A)

    return ProblemData(T=T, A=A, N_R=N_R, L=L, N=N, n0=n0, I0=I0, o0=o0,
                       C=cfg.C, h=cfg.h, Cs=cfg.Cs, atc=cfg.atc, tc=cfg.tc,
                       Qmax=Qmax, Imax=Imax,
                       priority=priority, tc_route=tc_route,
                       m=cfg.m, eta=cfg.eta, a_beta=cfg.a_beta,
                       beta_cov=beta_cov, z_bar=z_bar,
                       cov=cov, h0=h0, p_bar=p_bar, pool=pool, seed=cfg.seed,
                       layout=StateLayout(N_R, A, L))


# --------------------------------------------------------------- 序列化 / 打印
def to_json(prob: ProblemData) -> dict:
    return {
        "T": prob.T, "A": prob.A, "N_R": prob.N_R, "L": prob.L,
        "N": prob.N.tolist(), "n0": prob.n0.tolist(), "I0": prob.I0.tolist(),
        "o0": prob.o0.tolist(), "C": prob.C, "h": prob.h, "Cs": prob.Cs,
        "atc": prob.atc, "tc": prob.tc, "Qmax": prob.Qmax.tolist(),
        "Imax": prob.Imax.tolist(),
        "priority": {str(k): list(v) for k, v in prob.priority.items()},
        "tc_route": prob.tc_route.tolist(),
        "m": prob.m, "eta": prob.eta, "a_beta": prob.a_beta,
        "beta_cov": prob.beta_cov.tolist(), "z_bar": prob.z_bar.tolist(),
        "cov": prob.cov, "h0": prob.h0.tolist(), "p_bar": prob.p_bar.tolist(),
        "seed": prob.seed,
        "pool": [[z.tolist(), th.tolist()] for z, th in prob.pool],
    }


def from_json(d: dict) -> ProblemData:
    N_R, A, L = int(d["N_R"]), int(d["A"]), int(d["L"])
    return ProblemData(
        T=int(d["T"]), A=A, N_R=N_R, L=L,
        N=np.asarray(d["N"], dtype=np.int64), n0=np.asarray(d["n0"], dtype=np.int64),
        I0=np.asarray(d["I0"], dtype=np.int64), o0=np.asarray(d["o0"], dtype=np.int64),
        C=float(d["C"]), h=float(d["h"]), Cs=float(d["Cs"]),
        atc=float(d["atc"]), tc=float(d["tc"]),
        Qmax=np.asarray(d["Qmax"], dtype=np.int64),
        Imax=np.asarray(d["Imax"], dtype=np.int64),
        priority={int(k): list(v) for k, v in d["priority"].items()},
        tc_route=np.asarray(d["tc_route"], dtype=float),
        m=float(d["m"]), eta=float(d["eta"]), a_beta=float(d["a_beta"]),
        beta_cov=np.asarray(d["beta_cov"], dtype=float),
        z_bar=np.asarray(d["z_bar"], dtype=float),
        cov={k: float(v) for k, v in d["cov"].items()},
        h0=np.asarray(d["h0"], dtype=float),
        p_bar=np.asarray(d["p_bar"], dtype=float),
        pool=[(np.asarray(z, dtype=float), np.asarray(th, dtype=float)) for z, th in d["pool"]],
        seed=int(d["seed"]), layout=StateLayout(N_R, A, L),
    )


def print_summary(prob: ProblemData, out=print) -> None:
    """打印参数摘要 + 三项自检（构建规则 §1：print_summary 必带自检）。"""
    out("=" * 64)
    out("问题数据摘要（月龄状态多仓库备件调拨，docs/问题.md）")
    out("=" * 64)
    out(f"T = {prob.T}  A = {prob.A}  N_R = {prob.N_R}  L = {prob.L}  "
        f"pool = {len(prob.pool)}  seed = {prob.seed}  状态维 = {prob.state_dim}")
    out(f"机队 N = {prob.N.tolist()}  总数 = {int(prob.N.sum())}  "
        f"期初 W（待维修）= {[int(prob.N[r] - prob.n0[r].sum()) for r in range(prob.N_R)]}")
    out(f"成本: C={prob.C}  h={prob.h}  Cs={prob.Cs}  atc={prob.atc}  tc={prob.tc}")
    out(f"Weibull: m={prob.m}  η={prob.eta}  Beta({prob.a_beta}, b) 一阶矩匹配")
    out(f"PHM（中心化三协变量）: beta_cov = {prob.beta_cov.tolist()}  z_bar = {prob.z_bar.tolist()}")
    c = prob.cov
    out(f"协变量分布: 油温 N({c['mu_T']}, {c['sigma_T']}²)  "
        f"启停 Poisson({c['lambda']})  满负荷 100·Beta({c['alpha_L']}, {c['beta_L']}) (%)")
    out(f"h0: a=0 → {prob.h0[0]:.2e}  a={prob.A // 2} → {prob.h0[prob.A // 2]:.2e}  "
        f"a={prob.A - 1} → {prob.h0[-1]:.2e}")
    out(f"硬寿命截断强度: P(存活过 A={prob.A}) = exp(-(A/η)^m) = "
        f"{np.exp(-((prob.A / prob.eta) ** prob.m)):.3f}（宜接近 0）")

    # ---- 自检 1：修正因子中心化（E[f] 应 ≈ 1）----
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
    dev = abs(e_f_pool - e_f_theory) / e_f_theory * 100
    out(f"修正因子 E[f]: 理论 {e_f_theory:.4f}  池内 {e_f_pool:.4f}  偏差 {dev:.2f}%（阈值 3%）")
    if dev >= 3.0:
        raise ValueError("修正因子中心化自检失败，检查协变量采样")

    # ---- 自检 2：θ 池矩匹配（池内 E[θ] vs E_z[p(z)]，样本 SE 口径）----
    n = len(prob.pool)
    zmax, rel_at_zmax = 0.0, 0.0
    for r in range(prob.N_R):
        for a in range(1, prob.A - 1):
            ths = np.array([th[r, a] for _, th in prob.pool])
            se = ths.std(ddof=1) / np.sqrt(n)
            if se > 0:
                z = abs(ths.mean() - prob.p_bar[r, a]) / se
                if z > zmax:
                    zmax = z
                    rel_at_zmax = abs(ths.mean() - prob.p_bar[r, a]) / prob.p_bar[r, a]
    out(f"池矩匹配: max |E[θ]−E_z[p]|/SE = {zmax:.2f}（阈值 {Z_THRESHOLD:g}），"
        f"该处相对偏差 {rel_at_zmax*100:.1f}%，池 {n} 条")
    if zmax >= Z_THRESHOLD:
        # p 接近 1 时 Beta(2, b) 的密度在 x=1 附近发散，样本均值与样本 std 强负相关，
        # z 统计量在 n 小的时候严重重尾 —— 小池下 6σ 误报率很高，只作提示不作判据。
        if n < MIN_POOL_FOR_Z_TEST:
            out(f"  池只有 {n} 条（< {MIN_POOL_FOR_Z_TEST}），z 统计量重尾，本次不据此判失败；"
                f"要更可靠的池矩检验请把 pool_size 提到 {MIN_POOL_FOR_Z_TEST} 以上")
        else:
            raise ValueError(f"池矩匹配偏差 {zmax:.2f}σ 超阈值 6，检查 sample_pool 或换种子")

    # ---- 期望需求量级（期 1，按期初月龄分布；含到寿部分）----
    exp_fail = float(((prob.p_bar + np.where(np.arange(prob.A) == prob.A - 1,
                                             1.0 - prob.p_bar, 0.0)) * prob.n0).sum())
    out(f"期 1 期望需求（失效+到寿，按期初月龄分布）= {exp_fail:.2f} 台")
    out("=" * 64)
