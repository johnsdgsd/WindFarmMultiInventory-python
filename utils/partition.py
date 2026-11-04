# partition.py — 自适应分区的纯函数工具（与问题/引擎解耦，便于单测）。
"""场景分区、簇权重、聚合代表和文档 §3 的贪心分裂规则。"""

from __future__ import annotations

import numpy as np


def validate_partition(parts, n_scenes: int, stage: int | None = None) -> None:
    """校验一个阶段的划分：非空、互斥、并集恰为 0..n_scenes-1。"""
    where = f"阶段 {stage}" if stage is not None else "分区"
    if not parts:
        raise ValueError(f"{where} 不能为空")
    seen = np.zeros(int(n_scenes), dtype=np.int8)
    for cluster in parts:
        if not cluster:
            raise ValueError(f"{where} 存在空簇")
        for s in cluster:
            si = int(s)
            if si < 0 or si >= n_scenes:
                raise ValueError(f"{where} 场景索引越界: {si}")
            if seen[si]:
                raise ValueError(f"{where} 场景 {si} 重复出现")
            seen[si] = 1
    if not seen.all():
        missing = int(np.flatnonzero(seen == 0)[0])
        raise ValueError(f"{where} 未覆盖场景 {missing}")


def cluster_weights(parts, n_scenes: int) -> np.ndarray:
    """等概率场景池下，每个簇的概率权重 |P_l|/n_scenes。"""
    return np.asarray([len(c) / float(n_scenes) for c in parts], dtype=float)


def aggregate_theta(pool, indices) -> np.ndarray:
    """簇内 Theta 的等权平均；单场景簇直接返回原场景的 Theta。"""
    inds = [int(i) for i in indices]
    if len(inds) == 1:
        return np.asarray(pool[inds[0]][1], dtype=float)
    return np.mean(np.asarray([pool[i][1] for i in inds], dtype=float), axis=0)


def greedy_split(indices, sigs: np.ndarray, qs: np.ndarray,
                 eps_sig: float, eps_obj: float) -> list[list[int]]:
    """文档 §3 Algorithm 1 的贪心聚类。

    加入条件：||sig_k - sig_rep|| <= eps_sig 或 |Q_k - Q_rep| <= eps_obj。
    代表固定为第一个加入该簇的场景，不更新为质心。
    """
    idx = [int(i) for i in indices]
    if len(idx) <= 1:
        return [idx]
    S = np.asarray(sigs, dtype=float)
    Q = np.asarray(qs, dtype=float)
    out: list[list[int]] = [[idx[0]]]
    reps = [idx[0]]
    for s in idx[1:]:
        joined = False
        for m, r in enumerate(reps):
            dist = float(np.linalg.norm(S[s] - S[r]))
            dq = abs(float(Q[s] - Q[r]))
            if dist <= eps_sig or dq <= eps_obj:
                out[m].append(s)
                joined = True
                break
        if not joined:
            out.append([s])
            reps.append(s)
    return out


def median_pairwise_distance(sigs: np.ndarray) -> float:
    """签名两两 Euclidean 距离的中位数；少于 2 条时返回 0。"""
    S = np.asarray(sigs, dtype=float)
    n = len(S)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n - 1):
        d = np.linalg.norm(S[i + 1:] - S[i], axis=1)
        vals.append(d)
    return float(np.median(np.concatenate(vals)))


def stall_trigger(lb_hist, window: int, rtol: float, abs_tol: float) -> bool:
    """LB_best 连续 window 轮提升不超过 max(abs_tol, rtol*|LB|) 时返回 True。"""
    if window < 1 or len(lb_hist) < window + 1:
        return False
    last = float(lb_hist[-1])
    delta = last - float(lb_hist[-window - 1])
    return delta <= max(float(abs_tol), float(rtol) * abs(last))


def failure_order(pool) -> np.ndarray:
    """按场景平均失效强度排序，供 initial_clusters>1 的确定性分位切分。"""
    score = np.asarray([
        float(np.asarray(scene[1])[:, :, -1].sum(axis=1).mean()) for scene in pool
    ], dtype=float)
    return np.argsort(score, kind="stable")


def initial_partition(pool, n_scenes: int, k: int) -> list[list[int]]:
    """把全部场景按失效强度分位切成 k 个近似等大的簇。"""
    k = max(1, min(int(k), int(n_scenes)))
    if k == 1:
        return [list(range(n_scenes))]
    order = failure_order(pool)
    return [order[i::k].tolist() for i in range(k)]


def resolve_theta_cap(mode: str, n_clusters: int, n_scenes: int,
                      refine_max_clusters: int = 0,
                      configured: int = 0) -> int:
    """max-K 共享模型建多少个 theta。

    refine 关闭时 cap 必须等于 active K（fixed A=L、full multi=n、single=1），
    保证与旧模型逐位一致；refine 开启时 cap 至少覆盖单阶段 K 上限。
    """
    if int(configured) > 0:
        return int(configured)
    base = 1 if mode == "single" else (int(n_scenes) if n_clusters == 0
                                       else int(n_clusters))
    return max(int(base), int(refine_max_clusters or 0))


def split_cluster(parts, idx: int, children, n_scenes: int,
                  stage: int | None = None) -> list[list[int]]:
    """把一个簇替换成若干子簇；其他簇成员和顺序不变。"""
    kids = [list(map(int, c)) for c in children]
    if not kids or any(not c for c in kids):
        raise ValueError("子簇不能为空")
    out = [list(c) for c in parts]
    out[idx:idx + 1] = kids
    validate_partition(out, n_scenes, stage)
    return out


def cluster_dispersion(sigs, indices) -> float:
    """簇内签名的两两 Euclidean 距离中位数（空/单点返回 0）。"""
    idx = [int(i) for i in indices]
    if len(idx) < 2:
        return 0.0
    return median_pairwise_distance(np.asarray(sigs, dtype=float)[idx])


def normalized_cluster_score(sigs, indices, eps: float) -> float:
    """跨阶段可比的离散度分数：簇内离散度 / 阈值。eps=0 时用 1 作分母。"""
    denom = float(eps) if eps > 0 else 1.0
    return cluster_dispersion(sigs, indices) / denom


def fixed_partition(pool, n_scenes: int, n_clusters: int,
                    mode: str = "failure_quantile") -> list[list[int]]:
    """固定分区（全阶段同一套簇）。

    L=1: 全场景一簇；L>=n: 自然顺序单场景簇；否则按 mode 切分。
    failure_quantile 是连续分位块；failure_stride 是 adaptive 旧实现（消融）。
    """
    n = int(n_scenes)
    L = n if int(n_clusters) == 0 else int(n_clusters)
    if L < 1 or L > n:
        raise ValueError(f"n_clusters={n_clusters} 必须在 [1, {n}] 或 0")
    if L == 1:
        parts = [list(range(n))]
    elif L >= n:
        parts = [[i] for i in range(n)]
    else:
        order = failure_order(pool)
        if mode == "failure_quantile":
            parts = [a.tolist() for a in np.array_split(order, L)]
        elif mode == "failure_stride":
            parts = [order[i::L].tolist() for i in range(L)]
        else:
            raise ValueError(f"L={L}<n 时 group_mode 不能是 {mode!r}")
    validate_partition(parts, n)
    if sorted(s for c in parts for s in c) != list(range(n)):
        raise AssertionError("fixed_partition 覆盖异常")
    return parts

