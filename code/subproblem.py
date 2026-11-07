# subproblem.py — 等级模型单阶段子问题（纯 LP；Tier 1/2 整数扩展）。
"""状态布局 I | o | n | q；n[r,i] 为健康等级 i 的运行台数。

基线（scope=none）：
    N[r,i,j] = Θ[r,i,j] * n_in[r,i]          （Θ 只作系数）
    n_out[r,0] = Y[r] + N[r,0,0]             （Y=Σ_m y）
    n_out[r,j] = Σ_i N[r,i,j]                （j=1..G-1）
    F_new[r]   = Σ_i N[r,i,G]                （表达式/记录，不建 F 变量）
    dem[r]:    Σ_m y[r,m] + Σ_{i,j<G} N[r,i,j] <= N_r + fleet_tol
    fleet[r]:  Σ_i n_in[r,i] <= N_r + fleet_tol   （W_t≥0，容差只吸收 LP 残差）

Tier 2（scope=tier2_full）把 n/N/F 全整数化：删除逐格 Ndef 等式，改为
  · 每个源等级 i 的行和 Σ_j N[r,i,j] = n_in[r,i]；
  · 尾累计 T[r,i,j]=Σ_{k=j}^G N[r,i,k] 落在 C[r,i,j]·n_in 的最近整数带
    （C 为 Θ 的尾累计；右端只有 ±0.5 结构常数，Θ 只作 n 的系数）；
  · N 只建有效弧 j>=i，N 整数；F=Σ_i N[r,i,G] 自动整数。
这使 W_out=W_in+F-Y 也整数，避免 Tier 1 的分数停机底噪。

fleet_tol 是模型/审计的数值容差；基线/LP 控制的状态流转仍按 state_round_digits
round，并对 ≤state_repair_tol 的 Σn 超额从最大分量扣除；Tier 2 MIP 跳过该
分数修正，直接吸附整数并检查行和。

Θ 必须作为 n_in 的系数进入矩阵，不能把 N 算成右端常数——否则对偶看不见
∂N/∂n，Benders 割会越界。Tier 2 的尾累计行同样遵守这条。

保留的物流约束：pas / send_def / tau_* / inv / pipe / pipe_last / cap（出侧）。
容量取期末侧：I_out + Σ o_out + q_out <= Imax（文档 (1) 的期初侧在实现里由
上一期的出侧承担；用户确认保留出侧）。
"""

from __future__ import annotations

import numpy as np
from coptpy import COPT, LinExpr, quicksum

from utils import copt_env as ce
from .transition import tail_cdf

CUT_EPS = 1e-12
INF = COPT.INFINITY


def round_repair_n(n, N, digits: int, repair_tol: float) -> np.ndarray:
    """把 n 块 round 到 digits 位小数，并只修正不超过 repair_tol 的 Σn 超额。

    逐格 round 后行和仍可能略超 N（多个分量各带 0.5e-digits 的舍入误差）。
    这里从最大分量开始扣除超额，保证分量非负且 Σn ≤ N；超过 repair_tol 说明
    不是数值残差，直接报错。不缩放、不改动 n 之外的状态分量。
    """
    nr = np.round(np.asarray(n, dtype=float), int(digits)).copy()
    N = np.asarray(N, dtype=float)
    excess = nr.sum(axis=1) - N
    bad = excess > float(repair_tol) + 1e-12
    if np.any(bad):
        r = int(np.argmax(excess))
        raise ValueError(
            f"仓库 {r + 1} 期末 Σn={nr[r].sum():.12f} 超机队 "
            f"{excess[r]:.3e} > state_repair_tol={repair_tol:g}")
    for r in range(nr.shape[0]):
        rem = float(excess[r])
        if rem <= 1e-12:
            continue
        for j in np.argsort(-nr[r]):
            if rem <= 1e-12:
                break
            take = min(float(nr[r, j]), rem)
            nr[r, j] -= take
            rem -= take
        if rem > 1e-9:
            raise ValueError(
                f"仓库 {r + 1} 的 Σn 超额 {excess[r]:.3e} 无法从非负分量中扣除")
    return nr


def make_stage_subproblem(t, prob, cfg=None):
    """引擎工厂：只建 base，割由调用方用 assemble 拼装。"""
    return StageSubproblem(t, prob, cfg)


class StageSubproblem:
    """第 t 期子问题。t = 0 为虚拟首阶段，t = 1..T 为真实周期。"""

    def __init__(self, t: int, prob, cfg=None):
        self.t = int(t)
        self.prob_data = prob
        self.layout = prob.layout
        self.cfg = cfg
        self.is_virtual = (self.t == 0)
        self.integer_enabled = bool(getattr(cfg, "integer_enabled", False))
        self.integer_scope = str(getattr(cfg, "integer_scope", "none"))
        self.tier2 = self.integer_scope == "tier2_full"
        self.int_mask = (self.layout.integer_mask(self.integer_scope)
                         if self.integer_enabled
                         else np.zeros(self.layout.dim, dtype=bool))
        self.int_indices = np.flatnonzero(self.int_mask)
        self.int_tol = float(getattr(cfg, "int_tol", 1e-6))
        self._state_vec = None
        self._scene = None
        self._cut_rows = []
        self._cut_key = None
        self._state_coeff = {}
        self._pin_map = {}
        self._rows = {}
        self._obj = None
        self._last_mip = False
        self._last_lift_info = None
        self._link_rows = {}
        self._tail_rows = {}
        self._row_sum_rows = {}
        self._build()

    # -------------------------------------------------------------- base 构建
    def _build(self) -> None:
        Ly = self.layout
        N_R, G, L = Ly.N_R, Ly.G, Ly.L
        prob = self.prob_data
        ub = Ly.upper(prob)

        m = ce.new_model(self.cfg)
        self.model = m

        # 状态整数范围由 scope 的 int_mask 决定；仍只用变量界锁点。
        q_lo_j = Ly.off_q
        xin, xout = [], []
        for j in range(Ly.dim):
            kw = ({"vtype": COPT.INTEGER}
                  if self.integer_enabled and self.int_mask[j] else {})
            xin.append(m.addVar(lb=0.0, ub=float(ub[j]), name=f"xin_{j}", **kw))
            xout.append(m.addVar(lb=0.0, ub=float(ub[j]), name=f"xout_{j}", **kw))
            if self.is_virtual and not (q_lo_j <= j < q_lo_j + Ly.n_q):
                self._pin_map[j] = float(ub[j])
        self.x_in, self.x_out = xin, xout

        I_in = [xin[Ly.ix_I(r)] for r in range(N_R)]
        I_out = [xout[Ly.ix_I(r)] for r in range(N_R)]
        o_in = [[xin[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        o_out = [[xout[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        n_in = [[xin[Ly.ix_n(r, i)] for i in range(G)] for r in range(N_R)]
        n_out = [[xout[Ly.ix_n(r, i)] for i in range(G)] for r in range(N_R)]
        q_in = [xin[Ly.ix_q(r)] for r in range(N_R)]
        q_out = [xout[Ly.ix_q(r)] for r in range(N_R)]

        # 未来成本变量：base 里就位，由 assemble 决定它进不进目标。
        self._theta = m.addVar(lb=-INF, ub=INF, obj=0.0, name="theta")

        if self.is_virtual:
            # 虚拟首阶段：(2) 容量约束出侧；q_out 是唯一自由决策；目标只有 theta
            for r in range(N_R):
                m.addConstr(I_out[r] + quicksum(o_out[l][r] for l in range(L - 1))
                            + q_out[r] <= float(prob.Imax[r]), name=f"cap_{r}")
            return

        # ---- 决策变量：Tier 1 中物流决策整数，n/N/theta 连续 ----
        kw_int = {"vtype": COPT.INTEGER} if self.integer_enabled else {}
        cap_sum = float(np.sum(prob.Imax + prob.Qmax))
        taup, taum = [], []
        for r in range(N_R):
            up = float(min(cap_sum, prob.Imax[r]))
            dn = float(prob.Imax[r] + prob.Qmax[r])
            taup.append(m.addVar(lb=0.0, ub=up, name=f"taup_{r}", **kw_int))
            taum.append(m.addVar(lb=0.0, ub=dn, name=f"taum_{r}", **kw_int))
        y = [[m.addVar(lb=0.0, ub=float(min(prob.N[r], prob.Imax[mm])),
                       name=f"y_{r}_{mm}", **kw_int)
              for mm in range(N_R)] for r in range(N_R)]
        S_end = [m.addVar(lb=0.0, ub=float(prob.Imax[r] + prob.Qmax[r]),
                          name=f"send_{r}", **kw_int) for r in range(N_R)]
        tau = [taup[r] - taum[r] for r in range(N_R)]
        self.taup, self.taum, self.y, self.S_end, self.tau = taup, taum, y, S_end, tau

        # ---- 被动调拨（先）：各仓满足总量不超过期初库存 ----
        for mm in range(N_R):
            m.addConstr(quicksum(y[r][mm] for r in range(N_R)) - I_in[mm] <= 0.0,
                        name=f"pas_{mm}")
            self._note_state(f"pas_{mm}", [(Ly.ix_I(mm), -1.0)])

        # ---- 等级转移流量 N（有效弧 j>=i）----
        self.N = [[[None] * (G + 1) for _ in range(G)] for _ in range(N_R)]
        kw_N = ({"vtype": COPT.INTEGER}
                if self.tier2 and self.integer_enabled else {})
        for r in range(N_R):
            for i in range(G):
                for j in range(i, G + 1):
                    v = m.addVar(lb=0.0, ub=float(prob.N[r]),
                                 name=f"N_{r}_{i}_{j}", **kw_N)
                    self.N[r][i][j] = v
                    if not self.tier2:
                        row = m.addConstr(v - n_in[r][i] == 0.0,
                                          name=f"Ndef_{r}_{i}_{j}")
                        self._link_rows[(r, i, j)] = row
                        # 占位：左端 n_in 系数 -1 ⇒ a_c=+1；set_scene 改为 -Θ、a_c=+Θ
                        self._note_state(f"Ndef_{r}_{i}_{j}",
                                         [(Ly.ix_n(r, i), -1.0)])

        if self.tier2:
            # 行和：Σ_j N=n；尾累计行：T_j 与 C_j·n 的最近整数带。
            for r in range(N_R):
                for i in range(G):
                    row = m.addConstr(
                        quicksum(self.N[r][i][j] for j in range(i, G + 1))
                        - n_in[r][i] == 0.0, name=f"Nsum_{r}_{i}")
                    self._row_sum_rows[(r, i)] = row
                    self._note_state(f"Nsum_{r}_{i}",
                                     [(Ly.ix_n(r, i), -1.0)])
                    for j in range(i + 1, G + 1):
                        expr = quicksum(self.N[r][i][k]
                                        for k in range(j, G + 1))
                        row = m.addConstr(
                            expr, COPT.LESS_EQUAL,
                            0.5, name=f"Tail_{r}_{i}_{j}")
                        # 单行范围：下界 -0.5，上界 0.5−ε；set_scene 只改 n 的系数。
                        m.setInfo(COPT.Info.LB, [row], -0.5)
                        self._tail_rows[(r, i, j)] = row
                        self._note_state(f"Tail_{r}_{i}_{j}",
                                         [(Ly.ix_n(r, i), 0.0)])

        # ---- 下一期等级：列和 ----
        for r in range(N_R):
            m.addConstr(n_out[r][0] - quicksum(y[r][mm] for mm in range(N_R))
                        - self.N[r][0][0] == 0.0, name=f"grade0_{r}")
            for j in range(1, G):
                m.addConstr(n_out[r][j] - quicksum(self.N[r][i][j]
                                                   for i in range(j + 1)) == 0.0,
                            name=f"grade_{r}_{j}")

        # ---- 需求/待修：替换量 + 期末运行台数 <= 机队 ----
        for r in range(N_R):
            m.addConstr(quicksum(y[r][mm] for mm in range(N_R))
                        + quicksum(self.N[r][i][j] for i in range(G)
                                   for j in range(i, G))
                        <= float(prob.N[r] + prob.fleet_tol),
                        name=f"dem_{r}")

        # ---- W_t>=0：期初运行台数不超过机队（右端留数值容差）----
        for r in range(N_R):
            m.addConstr(quicksum(n_in[r][i] for i in range(G))
                        <= float(prob.N[r] + prob.fleet_tol),
                        name=f"fleet_{r}")
            self._note_state(f"fleet_{r}",
                             [(Ly.ix_n(r, i), 1.0) for i in range(G)])

        # ---- 发送/主动调拨/库存/在途/容量（原样保留）----
        for r in range(N_R):
            m.addConstr(S_end[r] - I_in[r]
                        + quicksum(y[s][r] for s in range(N_R)) - o_in[0][r] == 0.0,
                        name=f"send_def_{r}")
            self._note_state(f"send_def_{r}", [(Ly.ix_I(r), -1.0),
                                               (Ly.ix_o(0, r), -1.0)])

        m.addConstr(quicksum(tau) == 0.0, name="tau_bal")
        for r in range(N_R):
            m.addConstr(S_end[r] + tau[r] >= 0.0, name=f"tau_lo_{r}")
            m.addConstr(S_end[r] + tau[r] <= float(prob.Imax[r]), name=f"tau_hi_{r}")
            m.addConstr(tau[r] - quicksum(S_end[mm] for mm in range(N_R) if mm != r)
                        <= 0.0, name=f"tau_src_{r}")

        for r in range(N_R):
            m.addConstr(I_out[r] - S_end[r] - tau[r] == 0.0, name=f"inv_{r}")
        for l in range(L - 2):
            for r in range(N_R):
                m.addConstr(o_out[l][r] - o_in[l + 1][r] == 0.0,
                            name=f"pipe_{l}_{r}")
                self._note_state(f"pipe_{l}_{r}", [(Ly.ix_o(l + 1, r), -1.0)])
        for r in range(N_R):
            m.addConstr(o_out[L - 2][r] - q_in[r] == 0.0, name=f"pipe_last_{r}")
            self._note_state(f"pipe_last_{r}", [(Ly.ix_q(r), -1.0)])
        for r in range(N_R):
            m.addConstr(I_out[r] + quicksum(o_out[l][r] for l in range(L - 1))
                        + q_out[r] <= float(prob.Imax[r]), name=f"cap_{r}")

        # ---- 目标：采购 + 持有 + 停机 + 主动调拨 + 被动调拨 ----
        order = prob.C * quicksum(q_in)
        holding = prob.h / 2.0 * quicksum(I_in[r] + I_out[r] for r in range(N_R))
        downtime_const = prob.Cs * float(np.sum(prob.N))
        downtime = downtime_const - prob.Cs * quicksum(
            [n_in[r][i] for r in range(N_R) for i in range(G)])
        active = prob.atc / 2.0 * quicksum(taup[r] + taum[r] for r in range(N_R))
        passive = quicksum(float(prob.tc_route[r][mm]) * y[r][mm]
                           for r in range(N_R) for mm in range(N_R))
        m.setObjective(order + holding + downtime + active + passive, COPT.MINIMIZE)

    # ---------------------------------------------------------------- 内部记账
    def _note_state(self, name, lhs_pairs):
        """记录 约束名 → a_c 向量（a_c = 状态在约束右端的系数 = −左端系数）。"""
        a = np.zeros(self.layout.dim)
        for j, c in lhs_pairs:
            a[j] = -float(c)
        self._state_coeff[name] = a

    # ---------------------------------------------------------------- 拼装
    def assemble(self, cuts_next, key=None) -> "StageSubproblem":
        """把未来成本割拼装到 base 上。key 与上次相同则跳过。"""
        if key is not None and key == self._cut_key:
            return self
        m = self.model
        if self._cut_rows:
            m.remove(list(self._cut_rows))
            self._cut_rows = []

        if cuts_next is None or len(cuts_next[0]) == 0:
            # 末阶段或空割集：theta 钉在 0。空割集必须走这里，否则 LP 无界。
            m.setInfo(COPT.Info.LB, [self._theta], 0.0)
            m.setInfo(COPT.Info.UB, [self._theta], 0.0)
            m.setInfo(COPT.Info.Obj, [self._theta], 0.0)
        else:
            alpha, beta = cuts_next
            xout = self.x_out
            rows = []
            for i in range(len(alpha)):
                b = beta[i]
                nz = np.nonzero(np.abs(b) > CUT_EPS)[0]
                expr = LinExpr()
                expr.reserve(nz.size + 1)
                expr.addTerm(self._theta, 1.0)
                if nz.size:
                    expr.addTerms([xout[j] for j in nz], -b[nz])
                rows.append(m.addConstr(expr >= float(alpha[i]), name=f"cut_{i}"))
            self._cut_rows = rows
            m.setInfo(COPT.Info.LB, [self._theta], -INF)
            m.setInfo(COPT.Info.UB, [self._theta], INF)
            m.setInfo(COPT.Info.Obj, [self._theta], 1.0)
        self._cut_key = key
        return self

    # ---------------------------------------------------------------- 改参数
    def set_state(self, x, *, require_int: bool = False) -> "StageSubproblem":
        """锁状态：所有 x_in 的变量界。

        Tier 1 只对 I/o/q 掩码要求整数；n 块可为小数。`require_int=False`
        仅用于 LP 松弛/割有效性诊断，生产前向必须为 True。
        """
        Ly = self.layout
        v = np.asarray(x, dtype=float).ravel().copy()
        if v.size != Ly.dim:
            raise ValueError(f"状态维度 {v.size} ≠ {Ly.dim}")
        if require_int and self.integer_enabled:
            vals = v[self.int_indices]
            iv = np.rint(vals)
            dev = float(np.max(np.abs(vals - iv))) if vals.size else 0.0
            if dev > max(self.int_tol, 1e-6):
                j = int(self.int_indices[int(np.argmax(np.abs(vals - iv)))])
                raise ValueError(
                    f"状态分量 {j} 不是整数：{v[j]:.12g}（偏差 {dev:.3e}）")
            v[self.int_indices] = iv
        ub = Ly.upper(self.prob_data)
        lo = np.minimum(np.maximum(v, 0.0), ub)
        off = np.abs(v - lo) > 1e-3
        if off.any():
            bad = int(np.argmax(off))
            raise ValueError(f"状态分量 {bad} 试验值 {v[bad]} 越界 [0, {ub[bad]}]")
        m = self.model
        m.setInfo(COPT.Info.LB, self.x_in, [float(t) for t in lo])
        m.setInfo(COPT.Info.UB, self.x_in, [float(t) for t in lo])
        if self._pin_map:
            js = list(self._pin_map)
            m.setInfo(COPT.Info.LB, [self.x_out[j] for j in js],
                      [float(lo[j]) for j in js])
            m.setInfo(COPT.Info.UB, [self.x_out[j] for j in js],
                      [float(lo[j]) for j in js])
        self._state_vec = lo
        return self

    def set_scene(self, scene) -> "StageSubproblem":
        """按场景把 Θ 写入 Ndef 行的系数：N - Θ n_in = 0。

        Θ 只作 n_in 的系数；右端为 0。Θ=0 的结构零也显式 setCoeff(0)。
        """
        if self.is_virtual:
            return self
        if self._state_vec is None:
            raise RuntimeError("set_scene 之前必须先 set_state")
        Ly = self.layout
        th = np.asarray(scene[1], dtype=float)
        if th.shape != (Ly.N_R, Ly.G, Ly.G + 1):
            raise ValueError(f"Θ 形状 {th.shape} ≠ {(Ly.N_R, Ly.G, Ly.G + 1)}")
        m = self.model
        if self.tier2:
            ct = tail_cdf(th)
            for r in range(Ly.N_R):
                for i in range(Ly.G):
                    j = Ly.ix_n(r, i)
                    for jj in range(i + 1, Ly.G + 1):
                        c = float(ct[r, i, jj])
                        row = self._tail_rows[(r, i, jj)]
                        m.setCoeff(row, self.x_in[j], -c)
                        vec = np.zeros(Ly.dim)
                        vec[j] = c
                        self._state_coeff[f"Tail_{r}_{i}_{jj}"] = vec
        else:
            for r in range(Ly.N_R):
                for i in range(Ly.G):
                    j = Ly.ix_n(r, i)
                    for jj in range(i, Ly.G + 1):
                        c = float(th[r, i, jj])
                        m.setCoeff(self._link_rows[(r, i, jj)], self.x_in[j], -c)
                        vec = np.zeros(Ly.dim)
                        vec[j] = c
                        self._state_coeff[f"Ndef_{r}_{i}_{jj}"] = vec
        self._scene = scene
        return self

    def set_point(self, x, scene, *, require_int: bool = False) -> "StageSubproblem":
        self.set_state(x, require_int=require_int)
        self.set_scene(scene)
        return self

    # ---------------------------------------------------------------- 求解
    def solve_mip(self) -> "StageSubproblem":
        self._obj = ce.solve(self.model, mip=True)
        self._last_mip = True
        return self

    def solve_forward(self) -> "StageSubproblem":
        """前向策略：整数模式用 MIP，否则 LP。后向从不调用它。"""
        return self.solve_mip() if self.integer_enabled else self.solve_lp()

    def solve_lp(self) -> "StageSubproblem":
        if not self.is_virtual and self._scene is None:
            raise RuntimeError(f"stage {self.t}: 求解前未 set_scene")
        self._obj = ce.solve(self.model, mip=False)
        self._last_mip = False
        return self

    def solve_lb_mip(self) -> float:
        """阶段 0 的整数主问题下界。

        `auto_strict`：只有 MIP 证到最优（BestObj≈BestBnd）才读 BestObj；
        否则读 BestBnd。两者都是最小化问题的合法下界，绝不把未证 incumbent
        当 LB。
        """
        if not self.is_virtual:
            raise RuntimeError("solve_lb_mip 只用于阶段 0")
        info = ce.solve_bound(
            self.model,
            time_limit=float(getattr(self.cfg, "lb_time_limit", 60.0)),
            rel_gap=float(getattr(self.cfg, "lb_rel_gap", 1e-9)),
            abs_gap=float(getattr(self.cfg, "abs_gap", 1e-6)),
            value="auto_strict", exact_tol=1e-9, return_info=True)
        self._last_mip = True
        b = None if info is None else info.get("value")
        if b is None:
            raise RuntimeError("阶段 0 MIP 没有返回合法下界")
        return float(b)

    def solve_lift(self, pi, *, time_limit: float, rel_gap: float,
                   abs_gap: float = 0.0, safety_margin: float = 1e-6,
                   value: str = "exact") -> float | None:
        """固定 LP 斜率 pi 的 Lagrangian 整数截距（Strengthened Benders）。

        解 min_{x_in free, integer} [f(z) - pi·x_in]；只临时释放 x_in 并
        把 x_in 的目标系数从 c_s 改为 c_s-pi。

        value="exact"：对标 SDDP.jl StrengthenedConicDuality，MIP 必须
        BestObj≈BestBnd 才接受；否则返回 None，调用方回退 LP 割。
        value="best_bnd"：超时也可用 BestBnd（合法但可能较弱），由调用方
        与 LP 割取 max；仅用于训练提速/诊断。
        """
        if self.is_virtual:
            raise RuntimeError("solve_lift 只用于真实阶段")
        if self._state_vec is None or self._scene is None:
            raise RuntimeError("solve_lift 前必须 set_state 和 set_scene")
        if value not in ("exact", "best_bnd", "best_obj"):
            raise ValueError(f"未知 lift value={value!r}")
        pi = np.asarray(pi, dtype=float).ravel()
        if pi.size != self.layout.dim:
            raise ValueError(f"pi 维度 {pi.size} ≠ {self.layout.dim}")
        m = self.model
        xbar = np.asarray(self._state_vec, dtype=float).copy()
        ub = np.asarray(self.layout.upper(self.prob_data), dtype=float)
        cs = self.state_cost()
        xin = self.x_in
        # LP/MIP 当前解作为释放状态后 MIP 的显式可行起点。
        all_vars = list(m.getVars())
        all_vals = ce.values(m, all_vars)
        info = None
        try:
            m.setInfo(COPT.Info.LB, xin, [0.0] * self.layout.dim)
            m.setInfo(COPT.Info.UB, xin, [float(v) for v in ub])
            m.setInfo(COPT.Info.Obj, xin,
                      [float(cs[j] - pi[j]) for j in range(self.layout.dim)])
            try:
                m.setMipStart(all_vars, all_vals)
            except Exception:
                pass
            # exact 口径必须关掉相对 gap；否则 objective_value 仍可能是 incumbent。
            if value == "exact":
                lg, ag = 0.0, 0.0
            else:
                lg, ag = float(rel_gap), float(abs_gap)
            info = ce.solve_bound(
                m, time_limit=float(time_limit), rel_gap=lg,
                abs_gap=ag,
                restore_rel_gap=float(getattr(self.cfg, "mip_rel_gap", 1e-3)),
                restore_abs_gap=float(getattr(self.cfg, "abs_gap", 1e-6)),
                value=str(value), exact_tol=1e-9, return_info=True)
            self._last_mip = True
        finally:
            self._last_lift_info = info
            m.setInfo(COPT.Info.LB, xin, [float(v) for v in xbar])
            m.setInfo(COPT.Info.UB, xin, [float(v) for v in xbar])
            m.setInfo(COPT.Info.Obj, xin,
                      [float(v) for v in cs])
        b = None if info is None else info.get("value")
        if b is None:
            return None
        return float(b) - float(safety_margin)

    @property
    def obj(self) -> float:
        if self._obj is None:
            raise RuntimeError(f"stage {self.t}：尚未求解")
        return float(self._obj)

    def theta_value(self) -> float:
        return float(ce.values(self.model, self._theta))

    def stage_cost(self) -> float:
        """本期真实成本（不含 theta）。"""
        return self.obj - self.theta_value()

    def _snap_int_values(self, vals, label: str) -> np.ndarray:
        a = np.asarray(vals, dtype=float)
        iv = np.rint(a)
        dev = float(np.max(np.abs(a - iv))) if a.size else 0.0
        if dev > max(self.int_tol, 1e-6):
            raise RuntimeError(
                f"stage {self.t}: {label} 含非整数 {dev:.3e}（> IntTol）")
        return iv

    def out_state_vector(self) -> np.ndarray:
        """期末状态。虚拟阶段被钉住的分量用期初值。

        Tier 1 整数模式：I/o/q 精确吸附为整数并审计；n 仍按
        round_repair_n 处理（n 不声明整数）。LP 模式行为不变。
        """
        out = np.asarray(ce.values(self.model, self.x_out), dtype=float)
        if self._pin_map:
            for j in self._pin_map:
                out[j] = self._state_vec[j]
        Ly = self.layout
        if self.integer_enabled:
            out[self.int_indices] = self._snap_int_values(
                out[self.int_indices], "x_out")
            if not self.is_virtual:
                se = np.asarray(ce.values(self.model, self.S_end), dtype=float)
                self._snap_int_values(se, "S_end")
        n = out[Ly.off_n:Ly.off_n + Ly.n_n].reshape(Ly.N_R, Ly.G)
        if self.tier2 and self.integer_enabled:
            # Tier 2 的 n 由整数 N 的列和给出；不再做分数 round/repair。
            n = np.rint(n)
            excess = n.sum(axis=1) - self.prob_data.N
            if np.any(excess > float(self.prob_data.fleet_tol) + 1e-9):
                r = int(np.argmax(excess))
                raise RuntimeError(
                    f"stage {self.t}: 仓库 {r+1} 期末 Σn 超机队 "
                    f"{excess[r]:.3e} > fleet_tol={self.prob_data.fleet_tol:g}")
            out[Ly.off_n:Ly.off_n + Ly.n_n] = n.ravel()
        else:
            try:
                n_fixed = round_repair_n(
                    n, self.prob_data.N,
                    self.prob_data.state_round_digits,
                    self.prob_data.state_repair_tol)
            except ValueError as exc:
                raise RuntimeError(f"stage {self.t}: {exc}") from exc
            out[Ly.off_n:Ly.off_n + Ly.n_n] = n_fixed.ravel()
        return out

    def decision_values(self) -> dict:
        """仿真记录用的决策快照；整数模式下对物流决策精确吸附。"""
        Ly = self.layout

        def dec(vals, label):
            a = np.asarray(vals, dtype=float)
            if self.integer_enabled and self._last_mip:
                a = self._snap_int_values(a, label)
            return a

        out = {"q": [float(v) for v in dec(
                   ce.values(self.model, self.x_out[Ly.q_slice()]), "q")],
               "I_out": [float(v) for v in dec(
                   ce.values(self.model, self.x_out[:Ly.n_I]), "I_out")],
               "theta": self.theta_value()}
        if self.is_virtual:
            return out
        n_in = Ly.n(self._state_vec)
        n_out = Ly.n(self.out_state_vector())
        out.update({
            "taup": [float(v) for v in dec(ce.values(self.model, self.taup), "taup")],
            "taum": [float(v) for v in dec(ce.values(self.model, self.taum), "taum")],
            "S_end": [float(v) for v in dec(ce.values(self.model, self.S_end), "S_end")],
            "y": [[float(v) for v in dec(ce.values(self.model, row), "y")]
                  for row in self.y],
            "n_out": n_out.tolist(),
        })
        Nmat = np.zeros((Ly.N_R, Ly.G, Ly.G + 1), dtype=float)
        for r in range(Ly.N_R):
            for i in range(Ly.G):
                for j in range(i, Ly.G + 1):
                    val = float(ce.values(self.model, self.N[r][i][j]))
                    if self.tier2 and self.integer_enabled and self._last_mip:
                        val = float(self._snap_int_values([val], "N")[0])
                    Nmat[r, i, j] = val
        if self.tier2:
            out["N"] = Nmat.tolist()
        # 失效/待修诊断；Tier 2 的 N 来自整数转移，F 自动整数。
        F_new = [float(Nmat[r, :, Ly.G].sum()) for r in range(Ly.N_R)]
        W_in = [float(self.prob_data.N[r] - n_in[r].sum()) for r in range(Ly.N_R)]
        W_out = [float(self.prob_data.N[r] - n_out[r].sum()) for r in range(Ly.N_R)]
        out["F_new"] = F_new
        out["W_in"] = W_in
        out["W_out"] = W_out
        return out

    # ------------------------------------------------------- 对偶（割的斜率）
    def gradient(self, method: str = "dj") -> np.ndarray:
        """∂Q/∂x̄。dj 为生产路径；duals 为交叉验证。"""
        if method == "dj":
            if self._last_mip:
                raise RuntimeError(f"stage {self.t}：MIP 下取不到既约成本，请先 solve_lp")
            return np.asarray(ce.redcosts(self.model, self.x_in), dtype=float)
        if method == "duals":
            g = self.state_cost().copy()
            for name, a_c in self._state_coeff.items():
                pi = float(self.model.getInfo(COPT.Info.Dual, [self._row(name)])[0])
                g += a_c * pi
            return g
        raise ValueError(f"未知 method: {method}")

    def _row(self, name):
        if name not in self._rows:
            self._rows[name] = self.model.getConstrByName(name)
        return self._rows[name]

    def state_cost(self) -> np.ndarray:
        """状态在目标里的系数 c_s（q_in 的 C、I_in 的 h/2、n_in 的 −Cs）。"""
        Ly = self.layout
        c = np.zeros(Ly.dim)
        c[Ly.off_I:Ly.off_I + Ly.n_I] = self.prob_data.h / 2.0
        c[Ly.off_q:Ly.off_q + Ly.n_q] = self.prob_data.C
        c[Ly.off_n:Ly.off_n + Ly.n_n] = -self.prob_data.Cs
        return c

    def state_coeff(self) -> dict:
        return self._state_coeff

    def free(self) -> None:
        ce.close_model(self.model)
