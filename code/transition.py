# transition.py — Tier 2 的整数化工具：初始 n0 最大余数、尾累计取整参考值。
#
# 模型里的尾取整约束是线性的（Θ 只作 n 的系数），这里不参与建模；这些函数只用于
# 测试、仿真审计，以及验证 MIP 返回的 N 是否就是确定性取整，而不是在取整带内按成本
# 挑选出来的另一个 N。
from __future__ import annotations

import numpy as np

TAIL_ROUND_EPS = 1e-5


def largest_remainder_counts(total: int, mix) -> np.ndarray:
    """把 total 台按 mix 分配到各等级，保持总和不变。"""
    total = int(total)
    mix = np.asarray(mix, dtype=float)
    if total < 0 or np.any(mix < 0) or abs(float(mix.sum()) - 1.0) > 1e-8:
        raise ValueError(f"非法分配：total={total}, mix={mix.tolist()}")
    raw = total * mix
    base = np.floor(raw).astype(np.int64)
    deficit = total - int(base.sum())
    if deficit < 0:
        raise ValueError(f"分配基数超过 total：{base.tolist()}")
    frac = raw - base
    eligible = [j for j in range(len(mix)) if mix[j] > 0]
    order = sorted(eligible, key=lambda j: (-float(frac[j]), j))
    for j in order[:deficit]:
        base[j] += 1
    if int(base.sum()) != total:
        raise ValueError(f"最大余数分配后总和 {base.sum()} ≠ {total}")
    return base


def tail_cdf(theta) -> np.ndarray:
    """C[r,i,j] = Σ_{k=j}^{G} Θ[r,i,k]，形状同 Θ。"""
    th = np.asarray(theta, dtype=float)
    return np.cumsum(th[:, :, ::-1], axis=2)[:, :, ::-1]


def nearest_tail_flow(n, theta, eps: float = TAIL_ROUND_EPS) -> np.ndarray:
    """确定性尾累计取整得到 N，(N_R,G,G+1)，无效格为 0。

    规则：T[r,i,i]=n_i；对 j=i+1..G，
        T[r,i,j] = ceil(C[r,i,j]·n_i - 0.5)
    （平局向下），随后 N[r,i,j]=T_j-T_{j+1}。模型约束用 ±0.5 的
    线性带近似同一规则；eps 仅用于解释边界。
    """
    n = np.asarray(n, dtype=float)
    th = np.asarray(theta, dtype=float)
    if n.ndim != 2 or th.ndim != 3:
        raise ValueError(f"形状不匹配：n={n.shape}, theta={th.shape}")
    nr, g = n.shape
    if th.shape != (nr, g, g + 1):
        raise ValueError(f"Θ 形状 {th.shape} ≠ {(nr, g, g + 1)}")
    if np.max(np.abs(n - np.rint(n))) > 1e-7:
        raise ValueError("nearest_tail_flow 只接受整数 n")
    c = tail_cdf(th)
    out = np.zeros((nr, g, g + 1), dtype=np.int64)
    for r in range(nr):
        for i in range(g):
            ni = int(round(float(n[r, i])))
            if ni < 0:
                raise ValueError(f"n[{r},{i}]={ni} < 0")
            t = [ni]
            for j in range(i + 1, g + 1):
                center = float(c[r, i, j]) * ni
                t.append(int(np.ceil(center - 0.5)))
            t.append(0)
            for j in range(i, g + 1):
                v = t[j - i] - t[j - i + 1]
                if v < 0:
                    raise ValueError(f"尾取整出现负流量：r={r}, i={i}, j={j}")
                out[r, i, j] = v
    if np.any(out < 0):
        raise ValueError("N 出现负值")
    return out
