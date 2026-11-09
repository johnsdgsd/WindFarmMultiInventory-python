# Tier2 全整数等级模型 — Julia SDDP.jl 实现

本目录用 **SDDP.jl** 复刻 Python 项目的 **Tier2 全整数**等级模型
（`integer_enabled=true` / `scope="tier2_full"` /
`rounding="cumulative_tail_nearest"`）。**Python 是方程与参数的唯一事实源**；
参考 Julia 项目（`小论文/风场多仓库备件库存管理案例`）只提供目录组织与
SDDP.jl 用法约定，不提供本模型的方程。

本目录只做**模型构造与冒烟**：没有自研训练循环、没有仿真、没有割存取/增强。
训练只用 SDDP.jl 内置的 `SDDP.train`；本项目默认只建图、不训练。

## 目录

```
julia_proj/tier2/
├── run.jl                       # 构建入口（可选一次内置 SDDP.train）
├── code/
│   ├── problem_data.jl          # TOML + 导出 problem_data.json；过渡矩阵、池、n0
│   └── model.jl                 # build_sddp_model()：T+1 阶段 Tier2 子问题
├── config/
│   └── problem.toml             # 逐字复制自 Python Tier2 结果目录
├── data/
│   └── problem_data.json         # 逐字复制自 Python Tier2 结果目录（精确场景池）
├── test/
│   ├── smoke.jl                 # 结构自检 + COPT/HiGHS × 两种对偶处理器
│   ├── compare_python_stage.py  # Python 单阶段参考（读取 Python 源码）
│   ├── _python_stage.json       # 上述脚本生成的对照夹具
│   └── validate_equivalence.jl  # Julia vs Python 单阶段结构/数值等价
├── README.md
└── CLAUDE.md
```

## 版本与依赖

- Julia 1.12.1（系统环境；无 `Project.toml`）
- SDDP 1.13.0，JuMP 1.29.1，COPT.jl 1.1.29（默认求解器），HiGHS 1.19.3（可选交叉验证）
- TOML（stdlib）、JSON、Distributions 0.25.x、Random/Statistics/Printf/LinearAlgebra

## 命令

```bash
cd "D:/WYJ/我的文件/论文/小论文/风场多仓库备件库存管理案例-python/julia_proj/tier2"

julia run.jl                  # 全量：T=60、pool=300，建图并打印摘要（不训练）
julia run.jl smoke            # 冒烟：T=2、pool=4（取导出池前 4 条），建图
julia run.jl smoke_train 1    # 冒烟 + 一次 SDDP.train（默认 COPT + ContinuousConicDuality）
julia run.jl train 1          # 全量建图 + 一次训练；全池后向很贵，见下方说明

julia test/smoke.jl           # 结构自检 + COPT/HiGHS × 两种对偶处理器
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/compare_python_stage.py
julia test/validate_equivalence.jl

# 可选：同一模型换 HiGHS（不改默认）
julia -e 'include("run.jl"); main(smoke=true, optimizer=HiGHS.Optimizer)'
```

`run.jl train` 在全量池/T=60 下会枚举 300 场景 × 60 阶段的后向；单轮就可能很慢。
快速验证请用 `smoke_train` 或 `test/smoke.jl`。本项目不实现训练结果目录、割保存、
仿真或策略提取；正式训练请直接用 SDDP.jl 内置 API 并在外层自行管理结果。

验证脚本依赖：`test/compare_python_stage.py` 需要 Python 项目的 MyRL 环境
（`coptpy`/COPT）来读 Python 源码并生成 `_python_stage.json`；只用于更新对照夹具。
Julia 侧默认求解器是 COPT.jl 1.1.29（与 Python 同为 COPT）；HiGHS 是可选的交叉验证。
`validate_equivalence.jl` 读取已有的 `_python_stage.json`。

## 数据来源（逐字复制，未改动 Python 结果）

| 文件 | 来源 | SHA-256 |
|---|---|---|
| `config/problem.toml` | `results/int_t2_formal_20261106_180256/config/problem.toml` | `fd2f74aaef53bc6af842b67d587f1bcca8693dbaab203007a8e1f664cc8763dd` |
| `data/problem_data.json` | `results/int_t2_formal_20261106_180256/problem_data.json` | `96c6e8cec2ce311d96f1b6f2e3e4fbeb944f262994ad0fe86940c6a92dc86aac` |

`problem_data.json` 含 Python Tier2 实例的**精确 300 条 (z, Θ) 池**、整数 n0、
`P_ref`、优先级/路线成本等。`DefineProblem` 在加载时逐项与 TOML 对照，任何不一致
直接报错；`P_ref`、n0（最大余数）、行和/结构零、池矩也做自检。因此 Julia 的随机
输入与 Python Tier2 逐位相同，而不是“分布相似”。

冒烟 `pool_size=4` 是**截取导出池的前 4 条**（等概率），不是重新采样。

如果删除 `data/problem_data.json`，`DefineProblem` 会退回 Julia 采样（相同分布，
但 NumPy PCG64 与 Julia MersenneTwister 不同，**不逐位一致**），并打印警告。

## 模型语义（与 Python `code/subproblem.py` 逐条对应）

阶段图：`SDDP.LinearPolicyGraph(stages = T + 1)`。默认求解器
`COPT.Optimizer`（与 Python/reference 同为 COPT）；显式传
`optimizer = HiGHS.Optimizer` 可切换 HiGHS 做交叉验证。

- 节点 1：虚拟首阶段。钉住 I/o/n 的 `.out == .in`，`q.out` 自由（首期订货），
  容量约束在出侧，阶段目标 0。
- 节点 2..T+1：真实周期 1..T。订货成本为 `C * sum(q[r].in)`（上一期下的单在本期
  计费）；`q.out` 是本期下的新单。

状态 `I | o | n | q`（40 维，全部整数）：

- `I[r] ∈ [0, Imax[r]]`：可用库存；
- `o[l,r] ∈ [0, Qmax[r]]`，l=1..L−1：在途；
- `n[r,i] ∈ [0, N[r]]`，i=1..G：等级 i 运行台数；
- `q[r] ∈ [0, Qmax[r]]`：上期订单（here-and-now 状态）。

决策（全部整数）：`taup[r], taum[r]`（主动调拨正/负部），`y[r,m]`（仓 m 满足
仓 r 的需求），`S_end[r]`（被动调拨后剩余），`Nflow[r,i,j]`（仅有效弧 j ≥ i）。

Tier2 转移（与 V2 的差别仅在这里）：

```
Nsum[r,i]:  sum_{j=i}^{G+1} N[r,i,j] = n[r,i].in
Tail[r,i,j], j=i+1..G+1:
    -0.5 <= sum_{k=j}^{G+1} N[r,i,k] - C[r,i,j] * n[r,i].in <= 0.5,
    C[r,i,j] = sum_{k=j}^{G+1} Theta[r,i,k]
n_out[r,1] = sum_m y[r,m] + N[r,1,1]
n_out[r,j] = sum_{i=1}^{j} N[r,i,j],  j=2..G
F_new[r]   = sum_i N[r,i,G+1]        （派生整数，不建变量）
```

`Theta` 只以 `n.in` 的**归一化系数**出现（`JuMP.set_normalized_coefficient`），
右端只有 ±0.5 结构常数；不存在变量×参数。`N` 只建有效弧；没有 `Ndef` 行。

其余约束/成本与 Python V2 完全相同：`pas`、`send_def`、`tau_ball/tau_lo/tau_hi/
tau_src`、`inv`、`pipe/pipe_last`、`cap`（出侧）、`dem`、`fleet`；目标为
采购 + 持有 + 停机 + 主动调拨 + 被动调拨。`W` 是导出量，停机项为
`Cs * sum_r (N[r] - sum_i n[r,i].in)`。

没有 `min/max/floor` 进入状态转移。所有 `.out` 状态都有显式界；因为 SDDP.jl 只把
`SDDP.State` 的界/整数性放在 `.out`，本模型额外加了 80 条 `.in` 界约束
（`statein_*`）。它们在整数状态上与 `.out` 界/转移冗余；加它们是
`LagrangianDuality` 释放 `.in` 时保持子问题有界（且不改变可行域）。

## 内置 SDDP API（未自研的部分）

- `SDDP.LinearPolicyGraph` / `SDDP.State` / `@stageobjective`：建图与状态；
- `SDDP.parameterize` + `JuMP.set_normalized_coefficient`：场景系数；
- `SDDP.train`、`SDDP.calculate_bound`：训练与下界；
- 求解器：`COPT.Optimizer`（默认，与 Python/reference 同为 COPT）；`HiGHS.Optimizer`
  可选做交叉验证。
- `SDDP.ContinuousConicDuality()`：**与 Python Tier2 后向 LP 松弛对应**（前向 MIP，
  后向松弛求对偶割，割是整数问题的合法下界）。这是默认处理器。
- `SDDP.LagrangianDuality(method = SDDP.LocalImprovementSearch.BFGS(n))`：可选的
  SDDiP 风格处理器；不是 Python Tier2 协议，只作为整数问题的备选。
- `SDDP.Serial()` / `SDDP.Threaded()`：并行方案。
- 验证脚本另用 `SDDP.bellman_term`、`JuMP.relax_integrality`、
  `JuMP.dual(JuMP.FixRef(...))` 做单阶段对照；这些不参与模型构造。

**没有实现**：自研前向/后向、割筛选/LML1、割提升、resume、仿真、UB/CI 报表、
策略提取、结果目录。

## 验证结果

### 结构等价（Python StageSubproblem(1) vs Julia node 2）

| 项 | Python Tier2 | Julia SDDP.jl |
|---|---|---|
| 状态布局/维数 | I(5)\|o(2×5)\|n(5×4)\|q(5)，40 | 同 |
| 总列数 | 191 | 191 |
| 状态列 | x_in 40 + x_out 40 | 40 in + 40 out |
| 控制列 | taup 5 + taum 5 + y 25 + S_end 5 + N 70 | 同 |
| theta 列 | 1 | 1（SDDP 匿名 Bellman theta） |
| 约束行数 | 146 | **226 = 146 + 80 statein_** |
| 约束行数（`incoming_bounds=false`） | 146 | **146**（与 Python 相同） |
| Nsum / Tail | 20 / 50 | 20 / 50 |
| 其余行族 | pas5, grade0 5, grade 15, dem5, fleet5, send_def5, tau_bal1, tau_lo5, tau_hi5, tau_src5, inv5, pipe5, pipe_last5, cap5 | 逐族一致 |
| 整数声明 | x_in+x_out+控制 = 190 列声明 Int | .out 40 + 控制 110 = 150 列声明 Int；`.in` 每次求解被上一期整数 `.out` 固定 |
| Theta 位置 | `n.in` 系数 | 同（50 条 Tail 行全部与 `tail_cdf(scene 1)` 一致，误差 <1e-12） |

唯一结构差异是 80 条 `statein_*` 行（Python 用 x_in 变量界，SDDP.jl 的
`JuMP.fix` 不允许 `.in` 同时有变量界，故写成约束）和 Python 多声明 40 个 x_in 的
`Int`（它们被 `set_state` 固定在整数点，实际是常数）。在整数状态下可行域一致。
`incoming_bounds=false` 时行数恰为 146，与 Python 相同；MIP 目标、全部变量与
LP 松弛目标在两个固定状态上逐位一致（见下方 §C）。

### 单阶段数值等价（固定整数状态、scene 0、theta=0）

两个状态：Python 初始状态（I=o=q=0, n=n0）和非平凡 inventory 状态
（I=[2,1,2,1,1], o=1, q=[1,0,1,0,1]）。

| 项 | initial | inventory |
|---|---:|---:|
| Python MIP stage cost | 0.0 | 158.0 |
| Julia MIP objective | 0.0 | 158.0 |
| q/I/n/y/N/F max\|Δ\| | 0.0 | 0.0 |
| Python LP 松弛目标 | 0.0 | 156.0 |
| Julia LP 松弛目标 | 0.0 | 156.0 |
| LP 状态梯度 max\|Δ\| | **0.0** | **0.0** |

默认 COPT（Julia COPT.jl 1.1.29/COPT 8.0.7，Python coptpy/COPT 8.0.3）的 LP
梯度也逐位一致。若改用可选 HiGHS，初始 I=0 等活跃界处 LP 对偶不唯一，可能取到
不同的合法次梯度（此前实测 max\|Δ\|=2.5），但 LP/MIP 目标与所有原变量仍逐位一致，
割仍有效。

### §C 80 条 `statein_*` 的可行性检查（COPT）

- `incoming_bounds=true`（226 行）vs `false`（146 行）：false 的行数 146 与 Python
  完全一致；
- 两个固定状态下，MIP 目标与全部 `q/I/n/y/N/F` 变量、LP 松弛目标逐位一致
  （max\|Δ\|=0）；
- 因此 `.out` Int + `.in` 固定到上一期整数 `.out` + 80 条 `statein_*` 共同构成
  Python `x_in` 界/整数性的 SDDP.jl 表示；80 条行本身只补界，**不改变单阶段可行域
  或最优值**，它们用于让 `LagrangianDuality` 释放的 `.in` 有界。

### 冒烟训练

`julia test/smoke.jl`（T=2, pool=4）：

- 40 个状态分量全整数；191 列中 150 个整数自由度；70 个 N；20 个 Nsum；50 个 Tail；
- 50 条 `Tail_*` 行全部与 `tail_cdf(scene 1)` 一致（例如 scene 1 的
  `C[1,1,2]=0.000148719...`，行内 `n.in` 系数 = `-C`）；
- `SDDP.train(1, COPT, ContinuousConicDuality)`：LB = `125.44010938765132`，
  `numeric issues = 0`；
- `SDDP.train(1, COPT, LagrangianDuality/BFGS(3))`：LB = `125.44761360707344`，
  `numeric issues = 0`；
- `SDDP.train(1, HiGHS, ContinuousConicDuality)`（可选交叉验证）：
  LB = `125.44010938763677`，`numeric issues = 0`；
- `SDDP.train(1, HiGHS, LagrangianDuality/BFGS(3))`（可选交叉验证）：
  LB = `125.44823353005813`，`numeric issues = 0`。
- COPT 与 HiGHS 的单轮 LB 差：continuous 约 1.5e-11；Lagrangian 约 6.2e-4
  （均为一次迭代的合法下界）。

全量建图（T=60, pool=300）：61 节点，40 状态，约 4 s；加载/校验 JSON 约 4 s。

## 与 Python Tier2 的已知差异（全部为实现层，非方程层）

1. **场景池**：默认加载 Python 导出的精确池；采样回退不与 Python 逐位一致。
2. **`.in` 整数性 / 80 条 `statein_*`**：SDDP.jl 的 `Int` 只落在 `.out`；`.in`
   在每次求解被上一期整数 `.out` 固定；Python 的 x_in 界用 80 条 `statein_*`
   约束表达。`.out` Int + `.in` 固定 + 80 条约束共同表示 Python 的 x_in 界/整数性。
   已验证 `incoming_bounds=true`（226 行）与 `false`（146 行）的 MIP 目标、
   全部变量、LP 松弛目标逐位一致，false 行数 = Python 146；固定 `.in` 处不改变
   可行域。用 `LagrangianDuality` 时，SDDP.jl 在根子节点的 Lagrangian 搜索里会把
   `.in` 释放为连续变量（`incoming_state_bounds` 对根孩子是 `(-Inf,Inf,false)`），
   `statein_*` 约束保证其有界；这是对整数问题超集的松弛，割仍是合法下界。
   Python Tier2 用 `ContinuousConicDuality`（状态固定），不受此影响。
3. **`statein_*` 行**：替代 Python 的 x_in 变量界；整数状态下冗余，但
   LagrangianDuality 释放 `.in` 时需要。§C 已给出 true/false 的可行性/最优值
   一致性验证。
4. **求解器**：默认 COPT（Julia COPT.jl 1.1.29/COPT 8.0.7；Python
   coptpy/COPT 8.0.3，同一求解器族）；HiGHS 1.19.3 可选做交叉验证。COPT 下
   单阶段 MIP/LP 目标与 LP 状态梯度都与 Python 逐位一致；HiGHS 在退化点可能取
   不同合法次梯度。等价测试把 Python COPT 的 `mip_rel_gap` 设为 0。
5. **`state_round_digits` / `state_repair_tol`**：按 Python Tier2 的口径保留在配置中
   并打印，但整数模型不做分数 round/repair；这两个键不参与模型。
6. **配置严格性**：本实现要求 `n0_mode="integer_largest_remainder"` 且 `I0` 必填
   （Tier2 配置都有）；Python 的 loader 对二者有旧默认。
7. **`F`**：Python Tier2 和本实现都从整数 N 派生，不建 F 变量。
8. **不支持 resume/旧割**：本实现不保存割；每次建图都是全新模型。

## 注意

- `config/problem.toml` 的文件头注释仍写着“纯 LP/连续”，因为它是从 Python Tier2
  结果目录逐字复制的；实际 Tier2 口径由 `n0_mode` 和 `model.jl` 的整数声明决定。
  以配置值和代码为准，不要按头部注释理解。
- 参考 Julia 项目的月龄/Weibull 模型与本模型无关，只借用了目录与 SDDP.jl 写法。
- 不修改、不覆盖 Python 项目的 `code/` 与 `results/`。
