# V2 连续基线（Julia / SDDP.jl）

把 Python 项目的 **V2 连续基线**（`code/subproblem.py` 的 `scope=none` /
`integer_enabled=false`）逐项翻译成 SDDP.jl 的 `LinearPolicyGraph`：

- \(T=60\)、\(G=4\)、\(N_R=5\)、\(L=3\)；
- 状态布局 `I(5) | o(2×5) | n(5×4) | q(5)`（40 维），全部连续；
- 等级转移流量 \(N_{r,i,j}=\Theta_{r,i,j}\,n_{r,i}\)，**\(\Theta\) 只作约束系数**；
- 无整数变量、无 `F` 变量、无 `F = round(Θn)` 固定；
- 默认求解器 **COPT**（与 Python V2 同求解器族；COPT.jl 不可用时自动回退
  HiGHS），这是求解器层选择，不改变任何方程；
- 只做模型构建。训练/仿真/割存取一律用 SDDP.jl 内置
  `SDDP.train`、`SDDP.simulate`、`SDDP.calculate_bound`、
  `SDDP.write_cuts_to_file` / `SDDP.read_cuts_from_file`。

Python 是方程与参数的唯一权威来源；本目录只负责 Julia 实现与等价性验证。
参考 Julia 项目（`风场多仓库备件库存管理案例`）只用于工程布局和 SDDP.jl 用法。

## 目录

```
julia_proj/v2_baseline/
├── run.jl                       # 入口：建模型；可选 smoke 训练
├── code/
│   ├── problem_data.jl          # 读 TOML、Poisson 转移、PHM、场景池加载、自检
│   └── model.jl                 # build_sddp_model()，T+1 节点纯 LP
├── data/
│   ├── pool_seed2026.txt        # Python 导出的 300 条 (z,Θ) 场景池（唯一场景来源）
│   └── pool_seed2026_meta.toml  # 首场景/均值/P_ref，用于加载自检
├── scripts/
│   └── export_pool.py           # 从 Python 项目导出上述场景池（只读 Python）
├── test/
│   ├── smoke.jl                 # T=2、池前 4 条、SDDP.train 1 轮
│   ├── smoke_both.jl            # 同一冒烟的 COPT vs HiGHS（同种子）对比
│   ├── dump_stage.py            # Python 单阶段 LP → _py_stage_dump.toml
│   └── compare_stage.jl         # 与 Python 逐行/逐变量/对偶等价性检查
├── README.md
└── CLAUDE.md
```

## 命令

Python 解释器（与 Python 项目一致）：

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
cd "D:/WYJ/我的文件/论文/小论文/风场多仓库备件库存管理案例-python/julia_proj/v2_baseline"

# 1) 从 Python 导出场景池（已导出；换 seed / 重新采样时才需要）
PYTHONDONTWRITEBYTECODE=1 "$P" scripts/export_pool.py

# 2) 建默认 T=60 模型（不训练）
julia run.jl

# 3) 冒烟：T=2、池前 4 条、SDDP.train 1 轮（默认 COPT）
julia run.jl smoke
julia test/smoke.jl
julia test/smoke_both.jl        # COPT vs HiGHS，同种子 1 轮

# 4) 等价性验证（含状态梯度/对偶对比）
PYTHONDONTWRITEBYTECODE=1 "$P" test/dump_stage.py
julia test/compare_stage.jl
```

`run.jl` 只有位置参数：无参数 = 建默认模型；`smoke [n]` = 小实例 + 内置训练
`n` 轮（默认 1）。它不实现自定义训练/仿真/存取。

## 配置与场景池

- 默认配置：Python 项目的 `config/problem.toml`（**只读**，不复制、不改写）。
  可用 `config_path=` 或环境变量 `V2_PROBLEM_TOML` 指向别的文件。
- 默认场景池：`data/pool_seed<seed>.txt`。NumPy `default_rng` 的随机流无法在
  Julia 逐位复现，所以**不默认用 Julia 采样**；必须加载 Python 导出的同一批
  300 条场景。池文件是场景的唯一事实源。
- 冒烟用 `pool_size=4` 取池文件前 4 条（等概率）；`seed` 覆盖会要求存在
  `data/pool_seed<新 seed>.txt`，否则报错。
- 只有显式 `allow_julia_pool=true` 才回退到 Julia RNG，并打印警告
  （场景不同，结果不可与 Python 逐场景对照）。

参数（来自 `config/problem.toml`，代码不复制第二份）：
`T=60, G=4, N_R=5, L=3`；`N=[3680,2730,3780,3320,2520]`；
`Qmax=[3,2,3,3,2]`；`Imax=[12,8,20,15,8]`；`C=50, h=1, Cs=20, atc=5, tc=10`；
`lambda0=0.007, dirichlet_s=20, p_digits=3`；`pool_size=300, seed=2026`；
`init_mix=[0.420,0.369,0.163,0.048]`；`I0=0`；`fleet_tol=1e-6`。

## 模型语义

**节点结构**：`stages = T+1`。节点 1 虚拟首阶段：`I/o/n.out` 用 `JuMP.fix`
钉在初值（等价于 Python 的 `_pin_map` 变量界），`q.out` 自由（决定首期订货），
阶段成本 0，只有 5 条 `cap_r`。节点 2..T+1 为真实期 1..T。

**状态**（全部连续，界写在 SDDP 的 `.out`）：

| 状态 | 界 | 初值 |
|---|---|---|
| `I[r]` | `[0, Imax[r]]` | `I0[r]` |
| `o[l,r]` | `[0, Qmax[r]]` | 0 |
| `n[r,i]` | `[0, N[r]]` | `N[r]·init_mix[i]` |
| `q[r]` | `[0, Qmax[r]]` | 0 |

`.in` 由 SDDP 在每次求解前钉入上一期的 `.out`；根节点初值在界内，故不需要
额外的 incoming 约束。

**约束**（行名沿用 Python 0-based 命名，便于逐行对照）：

| 行 | 含义 |
|---|---|
| `pas_m` | `Σ_r y[r,m] ≤ I[m].in` |
| `Ndef_r_i_j` | `N[r,i,j] − Θ[r,i,j]·n[r,i].in = 0`（Θ 作系数） |
| `grade0_r` | `n[r,1].out = Σ_m y[r,m] + N[r,1,1]` |
| `grade_r_j` | `n[r,j].out = Σ_{i≤j} N[r,i,j]`（j≥2） |
| `dem_r` | `Σ_m y + Σ_{i,j<G} N ≤ N_r + fleet_tol` |
| `fleet_r` | `Σ_i n[r,i].in ≤ N_r + fleet_tol` |
| `send_def_r` | `S_end[r] = I[r].in − Σ_s y[s,r] + o[1,r].in` |
| `tau_bal` / `tau_lo_r` / `tau_hi_r` / `tau_src_r` | 主动调拨守恒/下界/上界/来源 |
| `inv_r` | `I[r].out = S_end[r] + τ[r]` |
| `pipe_l_r` / `pipe_last_r` | 在途滚动 / 本期订货入管 |
| `cap_r` | `I[r].out + Σ_l o[l,r].out + q[r].out ≤ Imax[r]` |

**目标**（与 Python 相同，`q.in` 计费）：

```
C·Σq.in + h/2·Σ(I.in+I.out) + Cs·(ΣN_r − Σn.in)
+ atc/2·Σ(τ⁺+τ⁻) + Σ tc_route[r,m]·y[r,m]
```

**噪声**：真实期每期调用
`SDDP.parameterize(sp, prob.pool, fill(1/|pool|, |pool|))`，在 do-block 内对
全部 70 条 `Ndef` 行执行
`set_normalized_coefficient(Ndef, n[r,i].in, -Θ[r,i,j])`。
不能把 `N` 算成常数塞右端（对偶看不见 `∂N/∂n`，割会越界），也不能把 `Θ`
建成决策变量/与变量相乘。

## 验证结果

`test/compare_stage.jl` 在**同一状态、同一场景**下对比 Python
`StageSubproblem` 与 Julia 节点 2 的 LP：

| 检查项 | Python | Julia |
|---|---|---|
| 虚拟阶段列数 / 结构行数 | 81 / 5 | 81 / 5 |
| 真实阶段列数 / 结构行数 | 191 / 146 | 191 / 146 |
| 146 行行名 | — | 完全一致 |
| 行 lb/ub/系数 max\|Δ\| | — | 0 |
| 目标系数（共享变量）max\|Δ\| | — | 0 |
| 变量界（共享变量）max\|Δ\| | — | 0 |
| 阶段成本（同状态/场景） | 221.2889128862 | 221.2889128862（COPT；\|Δ\|=4.8e-11） |
| 阶段成本 HiGHS 重解 | 221.2889128862 | 221.2889128862（\|Δ\|=0） |
| 状态梯度 py(COPT) vs Julia(COPT) | — | max\|Δ\|=0，mean\|Δ\|=0 |
| 状态梯度 py(COPT) vs Julia(HiGHS) | — | max\|Δ\|=50，mean\|Δ\|=2.5 |
| 场景池首场景/均值/P_ref | — | 逐值一致（max\|Δ\|=0） |

冒烟（T=2、池 4、1 轮，同种子）：COPT 与 HiGHS 的 LB 均为
`140.0277958101`，`numeric issues = 0`；COPT 用时约 4.7 s、HiGHS 约 3.3 s
（仅参考）。COPT 与 HiGHS 的阶段成本一致；状态梯度（= Benders 割斜率）
在退化时不同，见下。

## 与 Python 的差异 / 边界

1. **随机流**：NumPy `default_rng(2026)` 无法在 Julia 逐位复现，故场景池由
   Python 导出后加载；首场景、均值、P_ref 已逐值核对。
2. **状态流转的 round/repair 未实现**：Python 在仿真/状态流转中对 n 块按
   `state_round_digits=6` round，并对 ≤`state_repair_tol` 的 Σn 超额做修正。
   本模型只建 SDDP 图，不做自定义仿真；同一 LP 下 `n.out` 与 Python 的差
   ≤ 2.176e-6（其余状态块为 0）。这两个参数仍读取并校验，但不参与建模。
3. **求解器（默认 COPT，HiGHS 可选/回退）**：默认 `COPT.Optimizer`，与
   Python V2 同求解器族；COPT.jl 不可用时自动回退 HiGHS。两求解器的
   **阶段成本一致**，但固定 `x_in` 的既约成本（= ∂Q/∂x，Benders 割斜率）
   在退化时**不唯一**：Python COPT vs Julia COPT `max|Δ|=0`；Python COPT vs
   Julia HiGHS `max|Δ|=50`（mean 2.5，主要出现在无当期成本的 `q.in`）。
   因此默认用 COPT 生成割；HiGHS 保留为可选/回退。任一最优对偶给出的割
   都是合法下界，但不同求解器的割数值与收敛轨迹可能不同。
4. **COPT 容差**：COPT 的 `MatrixTol=1e-10` 会把更小的 Θ 系数当 0（Python
   与 Julia-COPT 一致）；逐行比较仍用 `1e-10` 阈值。SDDP 的数值报告会提示
   “matrix range contains small coefficients”，冒烟 `numeric issues = 0`。
5. **算法层**：不使用 Python 引擎的 LML1 割筛选、割提升、重要性采样、
   自定义并行/前向路径；训练/仿真完全交给 SDDP.jl 内置实现。
6. **退化分量**：首期 `q.out` 不计成本，LP 可能有多个最优解；等价性检查
   不依赖其取值（本次两者恰好一致）。
7. 本目录不实现 Tier 1 / Tier 2 整数扩展，只覆盖 V2 连续基线。

## 内置 API（不在本目录重复实现）

```julia
include("code/problem_data.jl")
include("code/model.jl")

prob  = DefineProblem()                 # 默认读 Python config/problem.toml + 导出池
model = build_sddp_model(prob)          # 默认 COPT；LinearPolicyGraph，61 节点
# 可选/回退：model = build_sddp_model(prob; optimizer = HiGHS.Optimizer)

SDDP.train(model; iteration_limit = 100, print_level = 1)
lb   = SDDP.calculate_bound(model)
sims = SDDP.simulate(model, 1000)       # 返回路径；需要记录变量时传 Symbol 列表
SDDP.write_cuts_to_file(model, "cuts.json")
# 读回：用同一 builder 重建骨架，再
# SDDP.read_cuts_from_file(model, "cuts.json")
```
