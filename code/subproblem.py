# subproblem.py — 单阶段子问题（coptpy）。约束编号 (1)..(8) 对应 docs/问题.md §3.3.3。
"""时序（每期）：被动调拨（只用期初库存，缺货即停机）→ 主动调拨（只用被动调拨后的
剩余 + 本期到货）→ 调拨后库存经库存动态流入下一期。订货 q 是 here-and-now：
本期订货在见到本期失效前定下，L−1 期后到货。

**base 与拼装分离**：`__init__` 只建 base —— 约束 (1)..(8) 加**阶段成本目标**，不含
任何割。未来成本 theta 与割行由 `assemble()` 挂上去，换割集时先卸旧再挂新。阶段 1..T
的 base 结构完全相同（约束与目标系数全静态），所以一个模型可以服务所有 t ≥ 1，
只在切阶段时换割。

**可变量只有两处**（其余系数全静态，这是"改参数不重建模型"成立的前提）：
  · 615 个 x_in 的变量界（状态锁点）
  · Fdef_{r,a} 行上 n_in 的系数 = −θ_{r,a}（逐场景）
θ 逐场景不同 ⇒ `set_scene` 必须在 `set_state` 之后。

**失效量 F 必须是变量 + 约束，不能是算出来的数字**（负 gap 的根因，见 docs/割越界根因分析.md）：
把 `F = rint(θ·n(x))` 算成一个数塞进 dem/age 行的右端，等于让状态对右端的依赖**不经过
系数矩阵**。而对偶（.dj = 既约成本）只看得见状态作为**变量界**那条通道 ⇒ 割的斜率少
一整项 `Σ θ·∂Q/∂F`（实测 θ·ν 相对 .dj 达 5%），割在锚点外高于真值 ⇒ 下界虚高 ⇒ 负 gap。
写成 `Fdef: F − θ·n_in = 0` 后，n_in 带系数 −θ 进矩阵，对偶自动带回这一项。
右端是纯 0：这也是必须放弃 rint 的原因 —— 保留 rint 就得把残差
`rint(θn) − θn` 这个状态导出的常数写回右端，等于原样退回上面的坑。

对偶提取纪律（SDDP算法Python实现/guide §10.2）：
  · 状态声明为真变量，只用**变量界**锁死在试验点 → 读 .dj 即得 ∂Q/∂x；
  · 不写第二条锁行（承载源重复会让对偶劈开）；
  · 目标里状态有系数（q_in 的 C、I_in 的 h/2、n_in 的 −Cs）→ 交叉验证时必须加 c_s。
"""

from __future__ import annotations

import numpy as np
from coptpy import COPT, LinExpr, quicksum

from utils import copt_env as ce

CUT_EPS = 1e-12
INF = COPT.INFINITY


def make_stage_subproblem(t, prob, cfg=None):
    """引擎工厂：只建 base，割由调用方用 assemble 拼装。

    必须放在模块顶层（spawn 按引用 pickle）；引擎把它随 initializer 传给每个 worker。
    """
    return StageSubproblem(t, prob, cfg)


class StageSubproblem:
    """第 t 期子问题。t = 0 为虚拟首阶段（钉初值 + 决定首期订货），t = 1..T 为真实周期。"""

    def __init__(self, t: int, prob, cfg=None):
        self.t = int(t)
        self.prob_data = prob
        self.layout = prob.layout
        self.cfg = cfg
        self.is_virtual = (self.t == 0)
        self._state_vec = None
        self._scene = None
        self._cut_rows = []
        self._cut_key = None
        self._state_coeff = {}
        self._fdef_rows = ([], [])
        self._pin_map = {}
        self._rows = {}
        self._obj = None
        self._last_mip = False
        self._build()

    # -------------------------------------------------------------- base 构建
    def _build(self) -> None:
        Ly = self.layout
        N_R, A, L = Ly.N_R, Ly.A, Ly.L
        prob = self.prob_data
        ub = Ly.upper(prob)

        m = ce.new_model(self.cfg)
        self.model = m

        # 状态：真变量，界由 set_state 锁在试验点。只用变量界锁死（路线 A）——
        # 每条限制只有一个承载源，读 .dj 即得 ∂Q/∂x。
        # **全部声明为整数变量**：状态是台数/件数，物理上就是整数；声明 INTEGER 让前向
        # MIP 的下一期状态也停在整数格点上，round 不会再引入分数。对后向无影响 ——
        # 后向走 solveLP()，整数性本来就被松弛掉，对偶照取（§set_scene 的 θ 系数才是
        # 割合法性的关键，与整数性无关）。
        q_lo_j, q_hi_j = Ly.off_q, Ly.off_q + Ly.n_q
        xin, xout = [], []
        for j in range(Ly.dim):
            xin.append(m.addVar(lb=0.0, ub=float(ub[j]), vtype=COPT.INTEGER,
                                name=f"xin_{j}"))
            if self.is_virtual and not (q_lo_j <= j < q_hi_j):
                # 虚拟首阶段：除 q 外期末 = 期初，用界钉住，省 615 条约束
                xout.append(m.addVar(lb=0.0, ub=float(ub[j]), vtype=COPT.INTEGER,
                                     name=f"xout_{j}"))
                self._pin_map[j] = float(ub[j])          # 初值占位，set_state 再钉
            else:
                xout.append(m.addVar(lb=0.0, ub=float(ub[j]), vtype=COPT.INTEGER,
                                     name=f"xout_{j}"))
        self.x_in, self.x_out = xin, xout

        I_in = [xin[Ly.ix_I(r)] for r in range(N_R)]
        I_out = [xout[Ly.ix_I(r)] for r in range(N_R)]
        o_in = [[xin[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        o_out = [[xout[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
        n_in = [[xin[Ly.ix_n(r, a)] for a in range(A)] for r in range(N_R)]
        n_out = [[xout[Ly.ix_n(r, a)] for a in range(A)] for r in range(N_R)]
        q_in = [xin[Ly.ix_q(r)] for r in range(N_R)]
        q_out = [xout[Ly.ix_q(r)] for r in range(N_R)]

        # 未来成本变量：base 里就位，由 assemble 决定它进不进目标。
        # 下界取 −∞ 与旧实现一致（pulp 的 LpVariable("theta") 默认 lowBound=None
        # ⇒ −1e30 自由变量）；V ≥ 0 由割集里的平凡割 theta ≥ 0 承担，不靠变量界。
        self._theta = m.addVar(lb=-INF, ub=INF, obj=0.0, name="theta")

        if self.is_virtual:
            # 虚拟首阶段：(2) 容量约束出侧，q_out 是唯一的自由决策；目标只有 theta
            for r in range(N_R):
                m.addConstr(I_out[r] + quicksum(o_out[l][r] for l in range(L - 1))
                            + q_out[r] <= float(prob.Imax[r]), name=f"cap_{r}")
            return

        # ---- 噪声：F[r,a] = round(θ[r,a]·n_in[r,a])，由 set_scene 按状态与场景填 ----
        # 整数变量的界按物理上限收紧（都是约束蕴含的，不改变可行域）：
        #   松界会让分支定界在真实前向上爆掉 —— 实测第 115 期用 ub=N_r=10000 的 y
        #   时 MIP 15 s 出不来、RSS 涨到 1.8 GB；收紧后同一处瞬时求解。
        #   y[r][m]：仓 m 满足仓 r 的量，既受 r 的需求（≤N_r）也受 m 的库存（≤Imax_m）限制；
        #   taup[r]：净调入量，受 r 自身容量 τ ≤ Imax_r − S_end_r ≤ Imax_r 限制；
        #   taum[r]：净调出量，受 −τ ≤ S_end_r ≤ Imax_r + Qmax_r 限制。
        cap_sum = float(np.sum(prob.Imax + prob.Qmax))
        taup, taum = [], []
        for r in range(N_R):
            up = float(min(cap_sum, prob.Imax[r]))
            dn = float(prob.Imax[r] + prob.Qmax[r])
            taup.append(m.addVar(lb=0.0, ub=up, vtype=COPT.INTEGER, name=f"taup_{r}"))
            taum.append(m.addVar(lb=0.0, ub=dn, vtype=COPT.INTEGER, name=f"taum_{r}"))
        y = [[m.addVar(lb=0.0, ub=float(min(prob.N[r], prob.Imax[mm])),
                       vtype=COPT.INTEGER, name=f"y_{r}_{mm}")
              for mm in range(N_R)] for r in range(N_R)]
        S_end = [m.addVar(lb=0.0, ub=float(prob.Imax[r] + prob.Qmax[r]), name=f"send_{r}")
                 for r in range(N_R)]
        tau = [taup[r] - taum[r] for r in range(N_R)]      # (1) τ = τ⁺ − τ⁻
        self.taup, self.taum, self.y, self.S_end, self.tau = taup, taum, y, S_end, tau

        # ---- 被动调拨（先）：各仓满足的总量不超过期初库存 ----
        # 约束写成 Σy − I_in ≤ 0，故 I_in 的左端系数是 −1 ⇒ a_c = +1
        for mm in range(N_R):
            m.addConstr(quicksum(y[r][mm] for r in range(N_R)) - I_in[mm] <= 0.0,
                        name=f"pas_{mm}")
            self._note_state(f"pas_{mm}", [(Ly.ix_I(mm), -1.0)])

        # ---- 失效量 F：**整数**真变量，由两条线性不等式夹出来
        #        Fdef_hi: F − θ·n_in ≤ +0.5   （占位系数 −1，set_scene 改成 −θ）
        #        Fdef_lo: F − θ·n_in ≥ −0.5
        #      F ∈ ℤ ⇒ F = rint(θ·n)（四舍五入）。右端只有纯结构常数 ±0.5，
        #      没有一个是「由状态算出来的数」——θ 以**系数**进矩阵，对偶看得见 ∂F/∂x。
        #      为什么用 rint 而不是 ⌊·⌋：⌊θn⌋ 在每个桶上系统性少算约半台，且当每桶
        #      期望失效 θ·n < 1 时直接归零。实测（单仓 700~1100 台）rint 偏差 −1%~−3%，
        #      ⌊·⌋ 达 −20%~−35%，⌈·⌉ 因正反馈超发 18 倍。见 docs/系统参数定标.md §5。
        #      松弛后 F ∈ [θn−0.5, θn+0.5]，与连续期望值同量级。
        F_var = [[m.addVar(lb=0.0, ub=float(prob.N[r]), vtype=COPT.INTEGER,
                           name=f"F_{r}_{a}")
                  for a in range(A)] for r in range(N_R)]
        fdef_hi, fdef_lo = [], []
        for r in range(N_R):
            for a in range(A):
                fdef_hi.append(m.addConstr(F_var[r][a] - n_in[r][a] <= 0.5,
                                           name=f"Fdef_hi_{r}_{a}"))
                fdef_lo.append(m.addConstr(F_var[r][a] - n_in[r][a] >= -0.5,
                                           name=f"Fdef_lo_{r}_{a}"))
                self._note_state(f"Fdef_hi_{r}_{a}", [(Ly.ix_n(r, a), -1.0)])
                self._note_state(f"Fdef_lo_{r}_{a}", [(Ly.ix_n(r, a), -1.0)])
        self.F, self._fdef_rows = F_var, (fdef_hi, fdef_lo)

        # ---- (4)+(5) 需求：被满足量 + 各月龄存活数 − 失效量 ≤ 机队总数 ----
        # 月龄 A−1 项系数消为 0：其失效与到寿都进需求，到寿离场隐式成立。
        dem_rows = []
        for r in range(N_R):
            dem_rows.append(m.addConstr(quicksum(y[r]) + quicksum(n_in[r])
                                        - quicksum(F_var[r]) <= float(prob.N[r]),
                                        name=f"dem_{r}"))
            self._note_state(f"dem_{r}", [(Ly.ix_n(r, a), 1.0) for a in range(A)])
        self.dem_rows = dem_rows

        # ---- S_end：被动调拨后剩余 + 本期到货 ----
        for r in range(N_R):
            m.addConstr(S_end[r] - I_in[r]
                        + quicksum(y[s][r] for s in range(N_R)) - o_in[0][r] == 0.0,
                        name=f"send_def_{r}")
            self._note_state(f"send_def_{r}", [(Ly.ix_I(r), -1.0),
                                               (Ly.ix_o(0, r), -1.0)])

        # ---- 主动调拨（后）----
        m.addConstr(quicksum(tau) == 0.0, name="tau_bal")
        for r in range(N_R):
            m.addConstr(S_end[r] + tau[r] >= 0.0, name=f"tau_lo_{r}")
            m.addConstr(S_end[r] + tau[r] <= float(prob.Imax[r]), name=f"tau_hi_{r}")
            m.addConstr(tau[r] - quicksum(S_end[mm] for mm in range(N_R) if mm != r)
                        <= 0.0, name=f"tau_src_{r}")

        # ---- (8) 月龄转移：新件进月龄 0，存活者月龄 +1 且扣掉失效量 F ----
        for r in range(N_R):
            m.addConstr(n_out[r][0] - quicksum(y[r]) == 0.0, name=f"age0_{r}")
        age_rows = []
        for r in range(N_R):
            for a in range(A - 1):
                age_rows.append(m.addConstr(
                    n_out[r][a + 1] - n_in[r][a] + F_var[r][a] == 0.0,
                    name=f"age_{r}_{a}"))
                self._note_state(f"age_{r}_{a}", [(Ly.ix_n(r, a), -1.0)])
        self.age_rows = age_rows

        # ---- (6) 库存动态 ----
        for r in range(N_R):
            m.addConstr(I_out[r] - S_end[r] - tau[r] == 0.0, name=f"inv_{r}")
        # ---- (7) 在途滚动：逐期前移，本期订货进末层 ----
        for l in range(L - 2):
            for r in range(N_R):
                m.addConstr(o_out[l][r] - o_in[l + 1][r] == 0.0, name=f"pipe_{l}_{r}")
                self._note_state(f"pipe_{l}_{r}", [(Ly.ix_o(l + 1, r), -1.0)])
        for r in range(N_R):
            m.addConstr(o_out[L - 2][r] - q_in[r] == 0.0, name=f"pipe_last_{r}")
            self._note_state(f"pipe_last_{r}", [(Ly.ix_q(r), -1.0)])
        # ---- (2) 容量：期末库存 + 在途 + 订货 ≤ Imax（钉出状态侧，RCR 自动成立）----
        for r in range(N_R):
            m.addConstr(I_out[r] + quicksum(o_out[l][r] for l in range(L - 1))
                        + q_out[r] <= float(prob.Imax[r]), name=f"cap_{r}")

        # ---- 目标：采购 + 持有 + 停机 + 主动调拨 + 被动调拨 ----
        order = prob.C * quicksum(q_in)                            # 采购（决定处计费）
        holding = prob.h / 2.0 * quicksum(I_in[r] + I_out[r] for r in range(N_R))
        # 停机成本写成常量 −Cs·Σ n_in，避免 quicksum 里混进 numpy 标量
        downtime_const = prob.Cs * float(np.sum(prob.N))
        downtime = downtime_const - prob.Cs * quicksum(
            [n_in[r][a] for r in range(N_R) for a in range(A)])
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
        """把未来成本割拼装到 base 上。key 与上次相同则跳过（同阶段的反复求解复用）。

        换割集 = 先 remove 旧的割行再 addConstr 新的。实测 remove 不会使其它行句柄
        失效，所以 base 里的静态行句柄可以一直复用。
        """
        if key is not None and key == self._cut_key:
            return self
        m = self.model
        if self._cut_rows:
            m.remove(list(self._cut_rows))
            self._cut_rows = []

        if cuts_next is None or len(cuts_next[0]) == 0:
            # 末阶段或空割集：theta 钉在 0，目标里不带未来成本。
            # 空割集必须走这里——让 theta 进目标却没有任何割行约束它，LP 会无界。
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
                # 批量 addTerms 比逐项 quicksum 快 5 倍（实测 2000 条×615 非零：
                # 1186 ms → 223 ms），大割集下这是主成本，别改回逐项构造
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
    def set_state(self, x, *, require_int: bool = True) -> "StageSubproblem":
        """锁状态：615 个 x_in 的变量界（唯一承载源，路线 A）。

        `require_int=False` 只给诊断用：后向走 LP 松弛，松弛内部产生的 x_out 是**小数**，
        而割必须在那上面也成立 —— 验割合法性时要能扫小数点。生产路径（前向 MIP 的
        整数试验点、`initial_state`）恒为整数，默认的整数校验是防止将来悄悄混入小数。
        """
        Ly = self.layout
        v = np.round(np.asarray(x, dtype=float).ravel(), 6)
        if v.size != Ly.dim:
            raise ValueError(f"状态维度 {v.size} ≠ {Ly.dim}")
        if require_int:
            iv = np.rint(v)
            off_int = np.abs(v - iv)
            if off_int.max() > 1e-3:
                j = int(np.argmax(off_int))
                raise ValueError(f"状态分量 {j} 试验值 {v[j]} 不是整数"
                                 f"（状态变量已声明为 INTEGER）")
            v = iv
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
            # 虚拟首阶段：非 q 槽的 x_out 等于期初
            js = list(self._pin_map)
            m.setInfo(COPT.Info.LB, [self.x_out[j] for j in js], [float(lo[j]) for j in js])
            m.setInfo(COPT.Info.UB, [self.x_out[j] for j in js], [float(lo[j]) for j in js])
        self._state_vec = lo
        return self

    def set_scene(self, scene) -> "StageSubproblem":
        """按场景把 θ 写进 Fdef 行的系数（n_in 的系数取 −θ）。

        θ 只能作**系数**进矩阵：写成数字塞右端，对偶就看不见 ∂F/∂x，割的斜率会缺
        θ·∂Q/∂F（负 gap 的根因）。θ 逐场景不同 ⇒ 必须先 set_state 再 set_scene。
        """
        if self.is_virtual:
            return self
        if self._state_vec is None:
            raise RuntimeError("set_scene 之前必须先 set_state")
        Ly = self.layout
        th = np.asarray(scene[1], dtype=float)
        m = self.model
        hi, lo = self._fdef_rows
        for r in range(Ly.N_R):
            for a in range(Ly.A):
                j = Ly.ix_n(r, a)
                m.setCoeff(hi[r * Ly.A + a], self.x_in[j], -float(th[r, a]))
                m.setCoeff(lo[r * Ly.A + a], self.x_in[j], -float(th[r, a]))
                # a_c（状态在约束右端的系数）= +θ：对偶交叉验证要用，逐场景刷新
                vec = np.zeros(Ly.dim)
                vec[j] = float(th[r, a])
                self._state_coeff[f"Fdef_hi_{r}_{a}"] = vec
                self._state_coeff[f"Fdef_lo_{r}_{a}"] = vec
        self._scene = scene
        return self

    def set_point(self, x, scene, *, require_int: bool = True) -> "StageSubproblem":
        self.set_state(x, require_int=require_int)
        self.set_scene(scene)
        return self

    # ---------------------------------------------------------------- 求解
    def solve_mip(self) -> "StageSubproblem":
        self._obj = ce.solve(self.model, mip=True)
        self._last_mip = True
        return self

    def solve_lp(self) -> "StageSubproblem":
        self._obj = ce.solve(self.model, mip=False)
        self._last_mip = False
        return self

    @property
    def obj(self) -> float:
        """Q(x̄)：子问题最优值（含 theta）。"""
        if self._obj is None:
            raise RuntimeError(f"stage {self.t}：尚未求解")
        return float(self._obj)

    def theta_value(self) -> float:
        return float(ce.values(self.model, self._theta))

    def stage_cost(self) -> float:
        """本期真实成本（不含 theta），仿真累加与成本分解用。"""
        return self.obj - self.theta_value()

    def out_state_vector(self) -> np.ndarray:
        """期末状态。界钉死的分量（虚拟首阶段的非 q 槽）预求解可能不回值，
        此时它们的值等于期初值。"""
        out = np.asarray(ce.values(self.model, self.x_out), dtype=float)
        if self._pin_map:
            for j in self._pin_map:
                out[j] = self._state_vec[j]
        return np.round(out, 6)   # 输出即下期输入，同样杀浮点尾巴

    def decision_values(self) -> dict:
        """仿真记录用的决策快照。"""
        Ly = self.layout
        out = {"q": [float(v) for v in ce.values(self.model, self.x_out[Ly.q_slice()])],
               "I_out": [float(v) for v in ce.values(self.model, self.x_out[:Ly.n_I])],
               "theta": self.theta_value()}
        if self.is_virtual:
            return out
        out.update({
            "taup": [float(v) for v in ce.values(self.model, self.taup)],
            "taum": [float(v) for v in ce.values(self.model, self.taum)],
            "S_end": [float(v) for v in ce.values(self.model, self.S_end)],
            "y": [[float(v) for v in ce.values(self.model, row)] for row in self.y],
            "n_out": Ly.n(self.out_state_vector()).tolist(),
        })
        return out

    # ------------------------------------------------------- 对偶（割的斜率）
    def gradient(self, method: str = "dj") -> np.ndarray:
        """∂Q/∂x̄，即 Benders 割的斜率。两种取法数学恒等，互为交叉验证。

        "dj"    状态用变量界锁死 ⇒ .dj 就是灵敏度（快路径，训练用这条）；
        "duals" 通用式 c_s + Σ_c a_c·π_c（a_c = 状态在约束右端的系数），诊断用。
        """
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

    def _row(self, name: str):
        if name not in self._rows:
            self._rows[name] = self.model.getConstrByName(name)
        return self._rows[name]

    def state_cost(self) -> np.ndarray:
        """状态在目标里的系数 c_s（q_in 的 C、I_in 的 h/2、n_in 的 −Cs，均非零）。"""
        Ly = self.layout
        c = np.zeros(Ly.dim)
        c[Ly.off_I:Ly.off_I + Ly.n_I] = self.prob_data.h / 2.0
        c[Ly.off_q:Ly.off_q + Ly.n_q] = self.prob_data.C
        c[Ly.off_n:Ly.off_n + Ly.n_n] = -self.prob_data.Cs
        return c

    def state_coeff(self) -> dict:
        """{约束名: a_c 向量}，a_c = 状态在约束右端的系数 = −左端系数。"""
        return self._state_coeff

    def free(self) -> None:
        """显式释放底层模型（长驻模型换新时必须调）。"""
        ce.close_model(self.model)
