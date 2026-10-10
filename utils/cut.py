# cut.py — Benders 割的表示与序列化。与问题解耦，可整目录迁移。
"""割是「下一期入库状态 x_out 的下界支撑超平面」：

    V_{t}(x) >= alpha + beta @ (x - state)

约定：`cuts[t]` 近似 V_t（第 t 期初的值函数），第 t-1 期子问题用 `cuts[t]` 约束其
未来成本变量 theta。周期索引 0..T：0 为虚拟首阶段（不持割），1..T 为真实周期。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMAT_VERSION = 1


@dataclass
class Cut:
    """一条割：V_stage(x) ≥ alpha + beta @ x（**截距式**）。

    截距式与点式 θ ≥ obj + β·(x − x̄) 代数等价；取截距式是因为子问题的割约束直接
    写成 `theta >= alpha + Σ beta_j·xout_j`，不必再展开 state。

    alpha : 截距，alpha = Q(x̄) - beta @ x̄（x̄ = state）
    beta  : 次梯度 ∂Q/∂x，与状态向量同序、同维
    state : 生成该割的 backward 试验点（只作留痕：复现、剪枝、策略分析用）
    stage : 生成时所在的期（割近似的是 V_stage）
    obj   : Q(x̄)，逐场景期望目标值
    it    : 创建该割的训练轮次（0 = 初始化平凡割）
    """

    alpha: float
    beta: np.ndarray
    state: np.ndarray
    stage: int
    obj: float
    it: int

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        self.beta = np.asarray(self.beta, dtype=float).ravel()
        self.state = np.asarray(self.state, dtype=float).ravel()
        self.obj = float(self.obj)
        if self.beta.shape != self.state.shape:
            raise ValueError(f"beta/state 维度不一致: {self.beta.shape} vs {self.state.shape}")

    def value_at(self, s) -> float:
        """割在状态 s 处给出的下界值。s 若传 (n, dim) 则返回 (n,)。"""
        arr = np.asarray(s, dtype=float)
        if arr.ndim == 1:
            return self.alpha + float(self.beta @ arr.ravel())
        return self.alpha + arr @ self.beta

    @property
    def dim(self) -> int:
        return int(self.beta.size)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta.tolist(),
                "state": self.state.tolist(), "stage": self.stage,
                "obj": self.obj, "it": self.it}

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        return cls(alpha=d["alpha"], beta=d["beta"], state=d["state"],
                   stage=d["stage"], obj=d["obj"], it=d["it"])


def trivial_cut(stage: int, state, dim: int) -> Cut:
    """平凡割 alpha=0, beta=0。

    目标非负 ⇒ V_t >= 0 处处成立 ⇒ theta >= 0 永远有效，可作为割集的初始元素。
    """
    return Cut(0.0, np.zeros(dim), np.asarray(state, dtype=float).ravel(), stage, 0.0, 0)


def pack(cuts, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """[Cut] -> (alpha (n,), beta (n, dim))。跨进程只传这份稠密数组。"""
    if not cuts:
        return np.zeros(0), np.zeros((0, dim))
    return (np.array([c.alpha for c in cuts], dtype=float),
            np.vstack([c.beta for c in cuts]))
