# CLAUDE.md — Tier2 全整数等级模型（Julia SDDP.jl）

## ① 一句话定义

用 SDDP.jl 复刻 Python 项目的 **Tier2 全整数**多仓库备件等级模型：状态
`I | o | n | q`（40 维全整数），等级转移用 `Nsum` + 累积尾最近整数带（Theta 只作
`n.in` 系数），所有物流决策与 N 整数；Python 是方程/参数的唯一事实源。本目录只做
模型构造与冒烟，不做自研训练/仿真/割存取。

## ② 目录与命令

```
julia_proj/tier2/
├── run.jl                       # 唯一入口：构建；可选调用一次内置 SDDP.train
├── code/problem_data.jl         # TOML + 导出 JSON、过渡矩阵、池、n0、摘要/自检
├── code/model.jl                # build_sddp_model()：T+1 阶段 Tier2 子问题
├── config/problem.toml          # 逐字复制自 Python Tier2 结果目录
├── data/problem_data.json       # 逐字复制：精确 300 条 (z,Θ) 池 + n0 + P_ref
└── test/
    ├── smoke.jl                 # 结构自检 + COPT/HiGHS × 两种对偶处理器
    ├── compare_python_stage.py  # Python 单阶段参考（生成 _python_stage.json）
    ├── validate_equivalence.jl  # 结构 + 单阶段 MIP/LP 等价
    └── _python_stage.json       # 上述 Python 脚本的对照夹具
```

```bash
cd "D:/WYJ/我的文件/论文/小论文/风场多仓库备件库存管理案例-python/julia_proj/tier2"
julia run.jl                  # T=60/pool=300 建图 + 摘要（不训练）
julia run.jl smoke            # T=2/pool=4 建图
julia run.jl smoke_train 1    # 冒烟 + 一次 ContinuousConicDuality 训练
julia test/smoke.jl           # 结构 + COPT/HiGHS × continuous/lagrangian
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/compare_python_stage.py
julia test/validate_equivalence.jl
```

依赖系统 Julia 1.12.1 的 SDDP 1.13、JuMP 1.29、COPT.jl 1.1.29（默认求解器；
随附 COPT 8.0.7；Python coptpy 为 COPT 8.0.3，单阶段结果仍逐位一致）、
HiGHS 1.19（可选交叉验证）、JSON、Distributions、TOML（stdlib）；无 `Project.toml`。

## ③ 当前模型语义（易错点）

阶段图 `SDDP.LinearPolicyGraph(stages=T+1)`：

- 节点 1 虚拟首阶段：`I/o/n` 的 `.out == .in`，`q.out` 自由（首期订货），
  出侧容量约束，阶段目标 0。
- 节点 2..T+1 真实周期 1..T。**订货成本是 `C * sum(q[r].in)`**：q 是 here-and-now
  状态，本期成本在“它作为状态出现”的这一期；本期 `q.out` 是下一期的新单。

状态（40 维，全部 `Int`，显式界）：

| 块 | 变量 | 界 | 初值 |
|---|---|---|---|
| I | `I[r]` | [0, Imax[r]] | I0[r] |
| o | `o[l,r]`, l=1..L−1 | [0, Qmax[r]] | 0 |
| n | `n[r,i]`, i=1..G | [0, N[r]] | n0[r,i]（最大余数整数） |
| q | `q[r]` | [0, Qmax[r]] | 0 |

决策（全部 `Int`）：`taup/taum[r]`、`y[r,m]`、`S_end[r]`、`Nflow[r,i,j]`（仅
`j ≥ i`）。Tier2 转移：

```
Nsum[r,i]:  sum_j N[r,i,j] = n[r,i].in                   (20 行)
Tail[r,i,j]: -0.5 <= sum_{k=j}^{G+1} N[r,i,k]
                      - C[r,i,j] * n[r,i].in <= 0.5      (50 行, j=i+1..G+1)
C[r,i,j] = sum_{k=j}^{G+1} Theta[r,i,k]
```

- **Theta 只作 `n.in` 的系数**（`set_normalized_coefficient`），右端只有 ±0.5；
  不要把 Theta 算成右端常数，也不要变量×参数。
- `N` 只建有效弧；**不要恢复 `Ndef` 行**。F 由 `sum_i N[r,i,G+1]` 派生，不建变量。
- `n_out` 用列和：`n_out[r,1] = sum_m y[r,m] + N[r,1,1]`；
  `n_out[r,j] = sum_{i=1}^{j} N[r,i,j]`（j=2..G）。
- `dem` 用 `sum_{i,j<G} N`，`fleet` 用 `sum_i n.in`，右端 `N + fleet_tol`。
- 其余物流约束与 Python V2 相同：pas/send_def/tau_*/inv/pipe/pipe_last/cap（出侧）。
- 目标：`C q.in + h/2(I.in+I.out) + Cs(N−sum n.in) + atc/2(taup+taum) +
  sum tc_route y`。
- **没有 min/max/floor 进状态转移**。

**SDDP.jl 的 `.in` 陷阱**：`@variable(..., SDDP.State, Int)` 只把界/整数性放在
`.out`；`.in` 无界。不能给 `.in` 设变量界——`SDDP.set_incoming_state` 调用
`JuMP.fix` 而不带 `force=true`，会报 “existing variable bounds”。本模型用 **80 条
`statein_*` 约束**补上 `.in` 界；这是 Python `x_in` 界/整数性的 SDDP.jl 表达
（`.out` 为 Int，`.in` 每步固定到上一期整数 `.out`）。已验证
`incoming_bounds=true`（226 行）与 `false`（146 行）在两个固定状态上的 MIP
目标、全部 `q/I/n/y/N/F` 变量、LP 松弛目标逐位一致，false 行数恰为 Python 的
146；所以 80 条
行不改变单阶段可行域/最优值。它们在整数状态上与 `.out` 界/转移冗余，但
`LagrangianDuality` 释放 `.in` 时保证有界。不要删（除非确定只用
`ContinuousConicDuality` 并接受差异）。注意 `LagrangianDuality` 在根子节点会把
`.in` 释放为连续变量（`incoming_state_bounds` 对根孩子是 `(-Inf,Inf,false)`）；
这是对整数问题超集的松弛，割仍合法；Python Tier2 用的
`ContinuousConicDuality` 不释放状态。

**状态键**：`node.states` 的键是逐分量的
`Symbol("I[1]")`、`Symbol("o[1,1]")`、`Symbol("n[1,1]")`、`Symbol("q[1]")`，
不是 `:I/:o/:n/:q`。

**求解器与对偶处理器**：

- 默认求解器：`COPT.Optimizer`（与 Python/reference 同为 COPT）；显式传
  `optimizer = HiGHS.Optimizer` 可切换 HiGHS 做交叉验证。
- 默认/对齐 Python Tier2：`SDDP.ContinuousConicDuality()`（前向 MIP、后向 LP 松弛，
  割是整数问题的合法下界）。
- 可选：`SDDP.LagrangianDuality(method=SDDP.LocalImprovementSearch.BFGS(n))`
  （SDDiP 风格；不是 Python Tier2 协议）。
- 单轮冒烟四种组合都通过，`numeric issues = 0`：
  COPT/continuous `125.44010938765132`，COPT/Lagrangian `125.44761360707344`，
  HiGHS/continuous `125.44010938763677`，HiGHS/Lagrangian `125.44823353005813`。
  COPT−HiGHS 差：continuous 约 1.5e-11，Lagrangian 约 6.2e-4。

## ④ 参数默认值

- **默认值只在 `config/problem.toml`**；代码没有第二份默认。缺键直接报错。
- Tier2 要求 `n0_mode = "integer_largest_remainder"`；`I0` 必填；`n0` 由最大余数
  逐仓生成并校验 `sum n0 = N`。
- 关键值：`T=60, G=4, N_R=5, L=3`；`N=[3680,2730,3780,3320,2520]`；
  `Qmax=[3,2,3,3,2]`；`Imax=[12,8,20,15,8]`；`init_mix=[0.420,0.369,0.163,0.048]`；
  `C=50,h=1,Cs=20,atc=5,tc=10`；`lambda0=0.007,dirichlet_s=20,p_digits=3`；
  `pool_size=300,seed=2026`。
- `state_round_digits` / `state_repair_tol` 只保留/打印，**Tier2 不做分数 round/repair**。
- 冒烟覆盖只改 `T=2, pool_size=4`；池取导出池前 4 条，不重新采样。
- `data/problem_data.json` 与 TOML 逐项交叉校验（N/Qmax/Imax/成本/λ0/s/β/z̄/
  init_mix/I0/seed/优先级/P_ref/n0/池长）；不一致直接报错。

## ⑤ 结果目录约定

- 本目录**不写 `results/`**，不保存割，不做仿真/UB/CI/策略提取。
- `SDDP.train` 默认会在当前工作目录写 `SDDP.log`；这是 SDDP.jl 的内置行为，本项目
  不做日志/结果目录管理。
- 不修改、不覆盖 Python 项目的 `code/` 与 `results/`；不碰
  `dmsvc.exe/DmServiceDMSERVER`。
- 不 git commit/push（除非用户明确要求）。

## ⑥ 已踩过的坑（按出现顺序）

1. **Julia 字符串里 `$var` 后接中文标点**（`$name，`）会被当成标识符的一部分 →
   ParseError。所有插值统一写 `$(var)`。
2. **JSON 对象的键是 String**：`Int(k)` 不行，必须 `parse(Int, k)`。
3. **SDDP.State 的 `Int` 只在 `.out`**；给 `.in` 设变量界会让 `JuMP.fix` 报
   “existing variable bounds”。用 80 条 `statein_*` 约束而不是变量界。已验证
   `incoming_bounds=true`（226 行）/`false`（146 行）的 MIP 目标与全部变量、
   LP 松弛目标逐位一致，false 行数 = Python 146，所以它们不改变单阶段
   可行域/最优值。
4. **`node.states` 的键是逐分量 Symbol**（`Symbol("n[1,1]")`），不是 `:n`。
5. **新建的非 direct SDDP 子问题没有挂 optimizer**；单独解一个 stage 前要
   `JuMP.set_optimizer(sp, COPT.Optimizer)`（默认；HiGHS 可选）。
6. **`node.parameterize` 只接收一个参数**（噪声 realization），不是 `(sp, ω)`。
7. **`tail_cdf` 必须初始化 `C[G]`**（不仅是 `C[G+1]`），否则 j=G 的尾带用错；
   已用 50 条 `Tail_*` 行对 `tail_cdf(scene 1)` 的系数自检覆盖。
8. **LP 对偶在活跃界/零点不唯一**：默认 COPT 下 Python 与 Julia 的 LP 梯度
   逐位一致（0.0/0.0）；若换可选 HiGHS，初始 I=0 点可能取不同合法次梯度
   （此前 max|Δ|=2.5），但 LP/MIP 目标与所有原变量仍逐位一致。
   不要把不同合法次梯度当成模型错误。
9. **Python 对照脚本中 `sub.solve_lp()` 会覆盖 `sub.obj`**：必须在 LP 之前抓
   MIP 目标/阶段成本，否则会把 LP 值记成 MIP 值（本次 158 曾被记成 156）。
10. **COPT `model.status` 是整数枚举**，不是字符串；与 `COPT.OPTIMAL` 比较。
11. **`SDDP.bellman_term` 的 theta 只有下界 0、没有上界**；无割时最优取 0，
    单阶段目标等于阶段成本。
12. **全量 T=60/pool=300 建图约 4 s，但单轮后向枚举 300×60**；不要误跑长训练。
    `run.jl` 在 `train` 且大实例时打印成本警告。

## ⑦ 当前进度与待办

**已完成（2026-11-09）**

- 读取并逐项对照 Python `problem_data.py`、`state_layout.py`、`transition.py`、
  `subproblem.py`、Tier2 配置与 `problem_data.json`；
- 全量 T=60/pool=300 建图（61 节点，40 状态）；
- 冒烟 T=2/pool=4 建图 + 结构自检；
- 一次 `SDDP.train`：COPT/HiGHS × continuous/Lagrangian 四种组合，
  `numeric issues=0`；LB 分别为 `125.44010938765132` /
  `125.44761360707344` / `125.44010938763677` / `125.44823353005813`；
- 结构等价：191 列/族一致，Python 146 行 → Julia 226 行（+80 statein），
  Nsum 20 / Tail 50 / N 70 一致；
- 单阶段 MIP 等价：initial 0/0，inventory 158/158；q/I/n/y/N/F 逐位一致；
- 单阶段 LP 松弛目标 0/0、156/156；默认 COPT 下两个状态的 LP 梯度均逐位一致
  （max|Δ|=0；可选 HiGHS 在初始退化点可能不同）；
- `incoming_bounds=true`（226 行）vs `false`（146 行）：MIP/LP 目标与全部变量
  逐位一致，false 行数 = Python 146，验证 80 条 `statein_*` 不改变单阶段可行域。

**未做（按任务范围，属于明确不实现）**

- 长训练、收敛曲线、LB/UB/gap 验收；
- 策略提取/策略分析；
- 蒙特卡洛仿真、UB/CI、2000/3000 路径；
- 割保存/加载/resume、LML1、割提升、重要性采样；
- 结果目录与 summary。

如果要继续做训练：直接 `SDDP.train(model; iteration_limit=..., time_limit=...,
duality_handler=SDDP.ContinuousConicDuality(), log_file=...)`，并在外层自行
管理结果目录；不要改本目录的模型方程，除非 Python Tier2 同步修改并重新导出
`problem_data.json`（模型结构/参数变了，旧割/旧结果不兼容）。

## ⑧ 文档过时清单

- `config/problem.toml` 头部仍写“纯 LP / 连续”；它是从 Python Tier2 结果目录
  **逐字复制**的，注释过时。以配置值和 `code/model.jl` 为准（Tier2 整数）。
- 参考 Julia 项目（`小论文/风场多仓库备件库存管理案例`）的 `docs/建模与实现.md`
  是**月龄/Weibull** 模型，与本等级模型无关；只借用其目录与 SDDP.jl 写法。
- Python 项目的 `docs/建模与实现.md` 描述的是 **V2 连续等级模型**；Tier2 的
  `Nsum`/`Tail`/整数语义以 `code/subproblem.py` 的 Tier2 分支和本目录 README/
  CLAUDE 为准。
- 本目录 README 的“验证结果”是 2026-11-09 的实测输出；若之后改模型或换
  SDDP/COPT/HiGHS 版本，需重跑 `test/smoke.jl` 与
  `test/validate_equivalence.jl`。
