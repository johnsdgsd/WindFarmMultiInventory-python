# model.py — 单场景子问题。约束编号 (1)..(8) 对应 docs/问题.md §3.3.3。
"""时序（每期）：被动调拨（只用期初库存，缺货即停机）→ 主动调拨（只用被动调拨后的
剩余 + 本期到货）→ 调拨后库存经库存动态流入下一期。订货 q 是 here-and-now：
本期订货在见到本期失效前定下，L−1 期后到货。

对偶提取纪律（SDDP算法Python实现/guide §10.2）：
  · 状态声明为真变量，只用**变量界**锁死在试验点 → 读 .dj 即得 ∂Q/∂x；
  · 不写第二条锁行（承载源重复会让对偶劈开）；
  · 目标里状态有系数（q_in 的 C、I_in 的 h/2、n_in 的 −Cs）→ 交叉验证时必须加 c_s。
"""

from __future__ import annotations

import numpy as np
import pulp

CUT_EPS = 1e-12


def build_stage_subproblem(t, state_vec, scene, cuts_next, prob, it: int = 0):
    """引擎回调（须是模块顶层函数：多进程 spawn 按引用 pickle）。"""
    return StageSubproblem(t, state_vec, scene, cuts_next, prob, it)


class StageSubproblem:
    """第 t 期子问题。t = 0 为虚拟首阶段（钉初值 + 决定首期订货），t = 1..T 为真实周期。

    参数
    ----
    state_vec : 本期入库状态（615 维），全部变量界锁死在该点上
    scene     : 场景 (z, θ)；虚拟首阶段无噪声，传 None
    cuts_next : (alpha (n,), beta (n, dim))，是在 x_out 上对 V_{t+1} 的下界；末期传 None
    """

    def __init__(self, t, state_vec, scene, cuts_next, prob, it: int = 0):
        self.t = int(t)
        self.prob_data = prob
        self.layout = prob.layout
        # 输入状态舍入 6 位小数：F = θ·n 的浮点尾巴（~1e-15）会沿转移逐期放大，
        # 表现为求解器抖动与割/下界震荡
        self.state_vec = np.round(np.asarray(state_vec, dtype=float).ravel(), 6)
        if self.state_vec.size != self.layout.dim:
            raise ValueError(f"状态维度 {self.state_vec.size} ≠ {self.layout.dim}")
        self.scene = scene
        self.cuts_next = cuts_next
        self.it = int(it)
        self.is_virtual = self.t == 0
        self.theta = None
        self._state_coeff = None
        self._state_cost = None
        self._build()

    # -------------------------------------------------------------- 建模
    def _build(self) -> None:
        Ly = self.layout
        N_R, A, L = Ly.N_R, Ly.A, Ly.L
        prob = self.prob_data
        ub = Ly.upper(prob)
        xbar = self.state_vec

        p = pulp.LpProblem(f"stage_{self.t}", pulp.LpMinimize)
        self.prob = p

        # 状态：真变量，本次求解锁死在试验点。只用变量界锁死（路线 A）——
        # 每条限制只有一个承载源，读 .dj 即得 ∂Q/∂x。q 槽是订货量，保持整数。
        q_lo, q_hi = Ly.off_q, Ly.off_q + Ly.n_q
        xin, xout = [], []
        out_fixed = np.full(Ly.dim, np.nan)
        for j in range(Ly.dim):
            v = float(xbar[j])
            lo = min(max(v, 0.0), float(ub[j]))
            if abs(v - lo) > 1e-3:
                raise ValueError(f"状态分量 {j} 试验值 {v} 越界 [0, {ub[j]}]")
            xin.append(pulp.LpVariable(f"xin_{j}", lowBound=lo, upBound=lo))
            if self.is_virtual and not (q_lo <= j < q_hi):
                # 虚拟首阶段：除 q 外期末 = 期初，直接用界钉住，省 615 条约束
                xout.append(pulp.LpVariable(f"xout_{j}", lowBound=lo, upBound=lo))
                out_fixed[j] = lo
            else:
                cat = "Integer" if q_lo <= j < q_hi else "Continuous"
                xout.append(pulp.LpVariable(f"xout_{j}", lowBound=0.0,
                                            upBound=float(ub[j]), cat=cat))
        self.x_in, self.x_out = xin, xout
        self._out_fixed = out_fixed

        I_in = [xin[Ly.ix_I(r)] for r in range(N_R)]
        I_out = [xout[Ly.ix_I(r)] for r in range(N_R)]
        o_in = [[xin[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        o_out = [[xout[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        n_in = [[xin[Ly.ix_n(r, a)] for a in range(A)] for r in range(N_R)]
        n_out = [[xout[Ly.ix_n(r, a)] for a in range(A)] for r in range(N_R)]
        q_in = [xin[Ly.ix_q(r)] for r in range(N_R)]
        q_out = [xout[Ly.ix_q(r)] for r in range(N_R)]

        if self.is_virtual:
            # 虚拟首阶段：(2) 容量约束出侧，q_out 是唯一的自由决策
            for r in range(N_R):
                p += (I_out[r] + pulp.lpSum(o_out[l][r] for l in range(L - 1)) + q_out[r]
                      <= float(prob.Imax[r])), f"cap_{r}"
            self._finish(0.0)
            return

        # ---- 噪声：F[r,a] = round(θ[r,a]·n_in[r,a])，本期失效台数 ----
        theta_s = self.scene[1]
        n_in_val = Ly.n(self.state_vec)
        F = np.rint(theta_s * n_in_val).astype(np.int64)

        # ---- 决策 ----
        cap_sum = float(np.sum(prob.Imax + prob.Qmax))
        taup = [pulp.LpVariable(f"taup_{r}", lowBound=0, upBound=cap_sum, cat="Integer")
                for r in range(N_R)]
        taum = [pulp.LpVariable(f"taum_{r}", lowBound=0, upBound=cap_sum, cat="Integer")
                for r in range(N_R)]
        y = [[pulp.LpVariable(f"y_{r}_{m}", lowBound=0, upBound=int(prob.N[r]),
                              cat="Integer") for m in range(N_R)] for r in range(N_R)]
        S_end = [pulp.LpVariable(f"send_{r}", lowBound=0,
                                 upBound=float(prob.Imax[r] + prob.Qmax[r]))
                 for r in range(N_R)]
        tau = [taup[r] - taum[r] for r in range(N_R)]     # (1) τ = τ⁺ − τ⁻
        self.taup, self.taum, self.y, self.S_end, self.tau = taup, taum, y, S_end, tau

        # ---- 被动调拨（先）：各仓满足的总量不超过期初库存 ----
        for m in range(N_R):
            p += pulp.lpSum(y[r][m] for r in range(N_R)) <= I_in[m], f"pas_{m}"
        # ---- (4)+(5) 需求：被满足量 + 各月龄存活数（n−F）≤ 机队总数 ----
        # 月龄 A−1 项系数消为 0：其失效与到寿都进需求，到寿离场隐式成立
        for r in range(N_R):
            p += (pulp.lpSum(y[r]) + pulp.lpSum(n_in[r])
                  <= float(prob.N[r] + F[r].sum())), f"dem_{r}"
        # ---- S_end：被动调拨后剩余 + 本期到货 ----
        for r in range(N_R):
            p += (S_end[r] == I_in[r] - pulp.lpSum(y[s][r] for s in range(N_R))
                  + o_in[0][r]), f"send_def_{r}"

        # ---- 主动调拨（后）----
        p += pulp.lpSum(tau) == 0, "tau_bal"
        for r in range(N_R):
            p += S_end[r] + tau[r] >= 0, f"tau_lo_{r}"
            p += S_end[r] + tau[r] <= float(prob.Imax[r]), f"tau_hi_{r}"
            p += tau[r] <= pulp.lpSum(S_end[m] for m in range(N_R) if m != r), f"tau_src_{r}"

        # ---- (8) 月龄转移：新件进月龄 0，存活者月龄 +1 ----
        for r in range(N_R):
            p += n_out[r][0] == pulp.lpSum(y[r]), f"age0_{r}"
            for a in range(A - 1):
                p += n_out[r][a + 1] == n_in[r][a] - float(F[r, a]), f"age_{r}_{a}"
        # ---- (6) 库存动态 ----
        for r in range(N_R):
            p += I_out[r] == S_end[r] + tau[r], f"inv_{r}"
        # ---- (7) 在途滚动：逐期前移，本期订货进末层 ----
        for l in range(L - 2):
            for r in range(N_R):
                p += o_out[l][r] == o_in[l + 1][r], f"pipe_{l}_{r}"
        for r in range(N_R):
            p += o_out[L - 2][r] == q_in[r], f"pipe_last_{r}"
        # ---- (2) 容量：期末库存 + 在途 + 订货 ≤ Imax（钉出状态侧，RCR 自动成立）----
        for r in range(N_R):
            p += (I_out[r] + pulp.lpSum(o_out[l][r] for l in range(L - 1)) + q_out[r]
                  <= float(prob.Imax[r])), f"cap_{r}"

        # ---- 目标：采购 + 持有 + 停机 + 主动调拨 + 被动调拨 ----
        order = prob.C * pulp.lpSum(q_in)                       # 采购（决定处计费）
        holding = prob.h / 2 * pulp.lpSum(I_in[r] + I_out[r] for r in range(N_R))
        downtime = prob.Cs * pulp.lpSum(prob.N[r] - pulp.lpSum(n_in[r]) for r in range(N_R))
        active = prob.atc / 2 * pulp.lpSum(taup[r] + taum[r] for r in range(N_R))
        passive = pulp.lpSum(float(prob.tc_route[r][m]) * y[r][m]
                             for r in range(N_R) for m in range(N_R))
        self._finish(order + holding + downtime + active + passive)

    def _finish(self, stage_cost) -> None:
        """挂上未来成本变量 theta（末期为 0）并写目标。pulp 的 `p += obj` 是覆盖语义，
        目标只能写一次，故这里统一组装。"""
        Ly = self.layout
        obj = stage_cost
        if self.cuts_next is not None:
            alpha, beta = self.cuts_next
            theta = pulp.LpVariable("theta")
            self.theta = theta
            xout = self.x_out
            for i in range(len(alpha)):
                b = beta[i]
                nz = np.nonzero(np.abs(b) > CUT_EPS)[0]
                if nz.size:
                    expr = pulp.LpAffineExpression(
                        [(xout[j], float(b[j])) for j in nz])
                    self.prob += theta >= float(alpha[i]) + expr, f"cut_{i}"
                else:
                    self.prob += theta >= float(alpha[i]), f"cut_{i}"
            obj = obj + theta
        self.prob += obj

    # -------------------------------------------------------------- 求解
    def solve(self, solver) -> "StageSubproblem":
        try:
            self.prob.solve(solver)
        except Exception as exc:                      # PulpSolverError 等
            raise RuntimeError(f"stage {self.t} 求解异常：{exc}") from exc
        status = pulp.LpStatus[self.prob.status]
        if status != "Optimal":
            raise RuntimeError(f"stage {self.t} 求解失败：{status}")
        return self

    @property
    def obj(self) -> float:
        """Q(x̄)：子问题最优值（含 theta）。

        pulp 对**不含变量**的目标（纯常量，如虚拟首阶段无割时的 0）会返回 None，
        此时退回常量本身。
        """
        v = pulp.value(self.prob.objective)
        return float(self.prob.objective.constant) if v is None else float(v)

    def stage_cost(self) -> float:
        """本期真实成本（不含 theta），仿真累加与成本分解用。"""
        th = 0.0 if self.theta is None else float(pulp.value(self.theta))
        return self.obj - th

    def out_state_vector(self) -> np.ndarray:
        """期末状态。界钉死的分量（虚拟首阶段的非 q 槽）不进任何约束，预求解会把
        它们消掉、求解器不回值 —— 此时它们的值就等于期初值。"""
        out = np.empty(self.layout.dim)
        for j, v in enumerate(self.x_out):
            val = pulp.value(v)
            if val is None:
                val = self._out_fixed[j]
                if np.isnan(val):
                    raise RuntimeError(f"stage {self.t}：x_out[{j}] 求解器未返回值")
            out[j] = float(val)
        return np.round(out, 6)   # 输出即下期输入，同样杀浮点尾巴

    def decision_values(self) -> dict:
        """仿真记录用的决策快照。"""
        Ly = self.layout
        out = {"q": [float(pulp.value(v)) for v in self.x_out[Ly.q_slice()]],
               "I_out": [float(pulp.value(v)) for v in self.x_out[:Ly.n_I]]}
        if self.is_virtual:
            return out
        out.update({
            "taup": [float(pulp.value(v)) for v in self.taup],
            "taum": [float(pulp.value(v)) for v in self.taum],
            "S_end": [float(pulp.value(v)) for v in self.S_end],
            "y": [[float(pulp.value(v)) for v in row] for row in self.y],
            "n_out": Ly.n(self.out_state_vector()).tolist(),
        })
        return out

    # ------------------------------------------------------- 对偶（割的斜率）
    def gradient(self, method: str = "dj") -> np.ndarray:
        """∂Q/∂x̄，即 Benders 割的斜率。两种取法数学恒等，互为交叉验证。

        "dj"    状态用变量界锁死 ⇒ .dj 就是灵敏度（快路径，训练用这条）；
        "duals" 通用式 c_s + Σ_c a_c·π_c（a_c = 状态在约束右端的系数 = −左端系数），
                诊断用；漏掉 c_s 会让每条割系统性偏斜且不报错（guide 坑 1）。
        """
        if method == "dj":
            vals = [v.dj for v in self.x_in]
            if any(v is None for v in vals):
                raise RuntimeError(f"stage {self.t}：求解器未返回 reduced cost，"
                                   f"请确认在解 LP（mip=False）")
            return np.asarray(vals, dtype=float)
        if method == "duals":
            g = self.state_cost().copy()
            for name, a_c in self.state_coeff().items():
                pi = self.prob.constraints[name].pi
                if pi is None:
                    raise RuntimeError(f"stage {self.t}：约束 {name} 未返回对偶值，"
                                       f"请确认在解 LP 而非 MIP")
                g += a_c * float(pi)
            return g
        raise ValueError(f"未知 method: {method}")

    def state_cost(self) -> np.ndarray:
        """状态在目标里的系数 c_s（q_in 的 C、I_in 的 h/2、n_in 的 −Cs，均非零）。"""
        if self._state_cost is None:
            obj = self.prob.objective
            self._state_cost = np.array([float(obj.get(v, 0.0)) for v in self.x_in])
        return self._state_cost

    def state_coeff(self) -> dict:
        """自动扫描：{约束名: a_c 向量}，a_c = 状态在约束右端的系数 = −左端系数。

        构造完成后扫描，新增任何含状态的约束都不会漏项。
        """
        if self._state_coeff is None:
            idx = {v.name: j for j, v in enumerate(self.x_in)}
            coeff = {}
            for name, con in self.prob.constraints.items():
                a = np.zeros(self.layout.dim)
                hit = False
                for v, c in con.items():
                    j = idx.get(v.name)
                    if j is not None:
                        a[j] = -float(c)
                        hit = True
                if hit:
                    coeff[name] = a
            self._state_coeff = coeff
        return self._state_coeff
