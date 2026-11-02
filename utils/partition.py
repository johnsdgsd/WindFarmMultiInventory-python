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
