# simulation.py — 蒙特卡洛仿真：分块、流式统计、95% CI。与问题解耦。
"""仿真口径与 Julia 版 simulate_in_chunks 一致：块内结果用完即弃、块种子 = seed+k，
因此结果依赖 chunk_size（分块只影响随机流切分，不影响统计量口径）。
每期累加的是**真实单期成本**（不含 theta），长期均值即策略的期望总成本上界。

仿真沿用引擎的前向任务（同一条求解路径），故策略口径与训练前向完全一致。
"""

from __future__ import annotations

import numpy as np


class SimStats:
    """总成本与逐期成本的流式统计量（不保留样本，几万条也不顶内存）。"""

    def __init__(self, T: int):
        self.T = int(T)
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.stage_sum = np.zeros(self.T)
        self.lo = float("inf")
        self.hi = float("-inf")

    def add(self, total: float, costs) -> None:
        self.n += 1
        self.total += total
        self.total_sq += total * total
        self.stage_sum += np.asarray(costs, dtype=float)
        self.lo = min(self.lo, total)
        self.hi = max(self.hi, total)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = (self.total_sq - self.n * self.mean ** 2) / (self.n - 1)
        return float(np.sqrt(max(var, 0.0)))

    @property
    def se(self) -> float:
        return self.std / np.sqrt(self.n) if self.n else float("nan")

    @property
    def ci95(self) -> float:
        return 1.959963984540054 * self.se

    def stage_mean(self) -> np.ndarray:
        return self.stage_sum / self.n if self.n else np.full(self.T, np.nan)


def simulate(engine, n_paths: int = 1000, seed: int | None = None,
             chunk_size: int = 1000, verbose: bool = False) -> SimStats:
    """用训练好的策略做蒙特卡洛前向模拟，返回总成本统计量。"""
    if n_paths < 1:
        raise ValueError("n_paths 至少为 1")
    base_seed = engine.seed if seed is None else int(seed)
    stats = SimStats(engine.T)

    done, blk = 0, 0
    while done < n_paths:
        n = min(chunk_size, n_paths - done)
        rng = np.random.default_rng(base_seed + blk)
        scene_paths = [rng.integers(0, engine.n_scenes, size=engine.T).tolist()
                       for _ in range(n)]
        for _, costs, total, _, _ in engine.forward_paths(scene_paths):
            stats.add(total, costs)
        done += n
        blk += 1
        if verbose:
            engine.log(f"  仿真 {done}/{n_paths}，均值 {stats.mean:.4f}")
    return stats


def simulate_paths(engine, scene_paths, record: bool = True):
    """按给定的场景路径序列仿真，返回原始结果（变量检查与个案分析用）。"""
    return engine.forward_paths([list(sp) for sp in scene_paths], record=record)
