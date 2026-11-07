# importance.py — 训练前向的全局场景 proposal（默认不启用；仿真不使用）。
"""最简 IS：所有阶段共用同一个场景分布 g。

g_j = floor/M + (1-floor) * score_j/Σscore
score_j = 包含场景 j 的 warmup 路径的平均总成本
路径权重 = Π_t (1/M) / g[scene_t]
"""
from __future__ import annotations

import hashlib

import numpy as np


class Proposal:
    def __init__(self, g: np.ndarray):
        g = np.asarray(g, dtype=float)
        if g.ndim != 1 or g.size < 2 or np.any(g <= 0):
            raise ValueError("proposal g 必须是一维正概率向量")
        g = g / g.sum()
        self.g = g
        self.M = int(g.size)
        self.log_g = np.log(g)
        self.log_f = -np.log(float(self.M))
        self.cdf = np.cumsum(g)
        self.cdf[-1] = 1.0
        self.fingerprint = hashlib.sha256(g.tobytes()).hexdigest()[:16]

    def sample_paths(self, rng: np.random.Generator, n_paths: int,
                     T: int) -> list[list[int]]:
        u = rng.random((n_paths, T))
        idx = np.searchsorted(self.cdf, u, side="right")
        idx = np.minimum(idx, self.M - 1)
        return idx.astype(int).tolist()

    def log_weights(self, scene_paths) -> np.ndarray:
        a = np.asarray(scene_paths, dtype=int)
        return (self.log_f - self.log_g[a]).sum(axis=1)

    def weights(self, scene_paths) -> np.ndarray:
        return np.exp(self.log_weights(scene_paths))


def build_global_proposal(scene_paths, totals, M: int,
                          uniform_floor: float = 0.05) -> Proposal:
    """用 warmup 路径构造全局场景分数；未出现场景得到均匀底部。"""
    if not (0.0 < float(uniform_floor) < 1.0):
        raise ValueError(f"uniform_floor 必须在 (0,1)：{uniform_floor}")
    sums = np.zeros(int(M), dtype=float)
    cnts = np.zeros(int(M), dtype=np.int64)
    for sp, total in zip(scene_paths, totals):
        a = np.asarray(sp, dtype=int)
        np.add.at(sums, a, float(total))
        np.add.at(cnts, a, 1)
    seen = cnts > 0
    mean_cost = float(np.mean(totals)) if len(totals) else 0.0
    score = np.full(int(M), mean_cost, dtype=float)
    score[seen] = sums[seen] / cnts[seen]
    score = np.maximum(score, 0.0)
    s = float(score.sum())
    if not np.isfinite(s) or s <= 0:
        q = np.full(int(M), 1.0 / M)
    else:
        q = score / s
    g = float(uniform_floor) / M + (1.0 - float(uniform_floor)) * q
    return Proposal(g)


def weight_stats(weights) -> dict:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0,
                "ess": 0.0, "ess_ratio": 0.0}
    ess = float(w.sum() ** 2 / np.sum(w * w))
    return {"min": float(w.min()), "median": float(np.median(w)),
            "max": float(w.max()), "mean": float(w.mean()),
            "ess": ess, "ess_ratio": ess / float(w.size)}
