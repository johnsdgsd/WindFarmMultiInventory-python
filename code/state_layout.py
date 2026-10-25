# state_layout.py — 状态向量的布局与界。割系数、试验点、转移全按此顺序，只在这里定义。
"""状态向量 = I(N_R) | o((L−1)·N_R) | n(N_R·G) | q(N_R)。

等级模型（纯 LP）。约定：
  · I[r]      仓库 r 期初可用库存
  · o[l][r]   在途第 l 层（l=0 表示下期初到货的那一批），共 L−1 层
  · n[r][i]   仓库 r 处于健康等级 i 的运行风机数，i = 0..G−1
  · q[r]      上期下的订单（here-and-now，计费发生在它作为状态出现的这一期）
状态块顺序保留 I | o | n | q；文档里的 (I,n,o,q) 只是数学记号。
"""

from __future__ import annotations

import numpy as np


class StateLayout:
    def __init__(self, N_R: int, G: int, L: int):
        if L < 2:
            raise ValueError(f"当前实现要求 L ≥ 2（在途管 L−1 ≥ 1），实际 {L}")
        self.N_R, self.G, self.L = int(N_R), int(G), int(L)
        self.n_I = self.N_R
        self.n_o = (self.L - 1) * self.N_R
        self.n_n = self.N_R * self.G
        self.n_q = self.N_R
        self.off_I = 0
        self.off_o = self.n_I
        self.off_n = self.off_o + self.n_o
        self.off_q = self.off_n + self.n_n
        self.dim = self.off_q + self.n_q

    # ---- 分量下标 ----
    def ix_I(self, r: int) -> int:
        return self.off_I + r

    def ix_o(self, l: int, r: int) -> int:
        return self.off_o + l * self.N_R + r

    def ix_n(self, r: int, i: int) -> int:
        return self.off_n + r * self.G + i

    def ix_q(self, r: int) -> int:
        return self.off_q + r

    # ---- 切片视图 ----
    def I(self, v) -> np.ndarray:
        v = np.asarray(v)
        return v[self.off_I:self.off_I + self.n_I]

    def o(self, v) -> np.ndarray:
        """返回 (L−1, N_R) 视图。"""
        v = np.asarray(v)
        return v[self.off_o:self.off_o + self.n_o].reshape(self.L - 1, self.N_R)

    def n(self, v) -> np.ndarray:
        """返回 (N_R, G) 视图。"""
        v = np.asarray(v)
        return v[self.off_n:self.off_n + self.n_n].reshape(self.N_R, self.G)

    def q(self, v) -> np.ndarray:
        v = np.asarray(v)
        return v[self.off_q:self.off_q + self.n_q]

    def q_slice(self) -> slice:
        return slice(self.off_q, self.off_q + self.n_q)

    # ---- 界的打包 ----
    def lower(self, prob) -> np.ndarray:
        return np.zeros(self.dim)

    def upper(self, prob) -> np.ndarray:
        ub = np.empty(self.dim)
        ub[self.off_I:self.off_I + self.n_I] = prob.Imax
        ub[self.off_o:self.off_o + self.n_o] = np.tile(prob.Qmax, self.L - 1)
        ub[self.off_n:self.off_n + self.n_n] = np.repeat(prob.N, self.G)
        ub[self.off_q:self.off_q + self.n_q] = prob.Qmax
        return ub
