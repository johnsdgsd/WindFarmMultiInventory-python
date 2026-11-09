# CLAUDE.md — V2 连续基线（Julia / SDDP.jl）交接文档

## ① 一句话定义

把 Python 项目的 **V2 连续基线**（等级模型 `scope=none`：T=60、G=4、
N_R=5、L=3、状态 `I|o|n|q` 40 维、`N=Θ·n` 连续、无整数变量、无 F 取整）
翻译成 SDDP.jl `LinearPolicyGraph`；**只建模型**，训练/仿真/割存取用 SDDP.jl
内置 API。Python 是方程与参数的唯一权威来源。默认求解器 **COPT**（与 Python
同族），COPT.jl 不可用时回退 HiGHS；这是求解器层选择，不改变方程。

## ② 目录与命令

```
julia_proj/v2_baseline/
├── run.jl                       # 唯一入口：建模型；smoke 时调用内置 SDDP.train
├── code/problem_data.jl         # TOML 读取/校验、Poisson 转移、PHM、池加载、自检
├── code/model.jl                # build_sddp_model()，T+1 节点纯 LP
├── data/pool_seed2026.txt       # Python 导出的 300 条 (z,Θ)，场景唯一事实源
├── data/pool_seed2026_meta.toml # 首场景/均值/P_ref，加载自检
├── scripts/export_pool.py       # 从 Python 导出池（只读 Python）
├── test/smoke.jl                # T=2、池前 4 条、SDDP.train 1 轮
├── test/smoke_both.jl           # 同一冒烟的 COPT vs HiGHS（同种子）
├── test/dump_stage.py           # Python 单阶段 LP dump
├── test/compare_stage.jl        # 逐行/逐变量/对偶等价性检查
├── README.md
└── CLAUDE.md
```

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
cd "D:/WYJ/我的文件/论文/小论文/风场多仓库备件库存管理案例-python/julia_proj/v2_baseline"

julia run.jl                     # 建 T=60 模型，不训练
julia run.jl smoke               # T=2、池 4、SDDP.train 1 轮（默认 COPT）
julia test/smoke.jl              # 同上
julia test/smoke_both.jl         # COPT vs HiGHS，同种子 1 轮
PYTHONDONTWRITEBYTECODE=1 "$P" scripts/export_pool.py   # 重新导出场景池
PYTHONDONTWRITEBYTECODE=1 "$P" test/dump_stage.py       # Python 单阶段 dump
julia test/compare_stage.jl      # 等价性检查
```

`run.jl` 只有位置参数：空 = 建默认模型；`smoke [n]` = 小实例 + 内置训练
`n` 轮。不要在本目录实现自定义训练/仿真/保存/割增强。

## ③ 当前模型语义

**节点**：`stages = T+1`。节点 1 虚拟首阶段：`JuMP.fix` 把 `I/o/n.out`
钉在初值（等价 Python `_pin_map`），`q.out` 自由，阶段成本 0，只有 5 条
`cap_r`。节点 2..T+1 为真实期 1..T。

**状态**（全部连续；SDDP 的界落在 `.out`，`.in` 每次求解前被钉入上一期
`.out`；根初值在界内，不需要额外 incoming 约束）：

- `I[r] ∈ [0, Imax[r]]`，初值 `I0[r]`
- `o[l,r] ∈ [0, Qmax[r]]`，初值 0
- `n[r,i] ∈ [0, N[r]]`，初值 `N[r]·init_mix[i]`（连续，不取整）
- `q[r] ∈ [0, Qmax[r]]`，初值 0

**决策**（连续）：`taup_r`、`taum_r`、`y_r_m`、`send_r`（=S_end）、
`N_r_i_j`（仅有效弧 `j≥i`）。`tau = taup - taum` 是表达式。

**约束**：`pas` / `Ndef` / `grade0` / `grade` / `dem` / `fleet` /
`send_def` / `tau_bal` / `tau_lo` / `tau_hi` / `tau_src` / `inv` /
`pipe` / `pipe_last` / `cap`，行名与 Python 子问题完全一致（0-based 名字）。

**噪声**：`SDDP.parameterize(sp, prob.pool, fill(1/n,n))`，do-block 内对全部
70 条 `Ndef` 行调用
`JuMP.set_normalized_coefficient(c_Ndef[r,i,j], n[r,i].in, -theta[r,i,j])`。
即 `N − Θ·n.in = 0`，`Θ` 只作系数。

**目标**：

```
C·Σq.in + h/2·Σ(I.in+I.out) + Cs·(ΣN_r − Σn.in)
+ atc/2·Σ(τ⁺+τ⁻) + Σ tc_route[r,m]·y[r,m]
```

`q.in` 计费：首期订货在节点 2 计费，末期的 `q.out` 不计费（与 Python 相同）。

**求解器**：`build_sddp_model` 默认 `COPT.Optimizer`；`HAS_COPT=false` 时回退
`HiGHS.Optimizer`。可显式传 `optimizer = HiGHS.Optimizer`。只换求解器，
方程/状态/目标不变。

**易错点**：

- **不要把 `N` 算成常数塞右端**：对偶看不见 `∂N/∂n`，Benders 割会系统性
  越界。必须走 `Ndef` + `set_normalized_coefficient`。
- **parameterize 内要设置全部 70 个系数（含 0）**，否则会残留上一个场景的
  系数。`set_normalized_coefficient` 是绝对赋值。
- **虚拟阶段用 `JuMP.fix` 钉 `.out`**，不要改成 `out == in` 等式行：Python
  用变量界钉，行结构不同；`JuMP.fix` 才能保持虚拟阶段 5 条结构行。
- **行名必须保留 Python 命名**：`Ndef_r_i_j` 等；否则等价性检查对不上。
- **首期 `q.out` 退化**：它不计成本，LP 可有多个最优解；不要用它的取值做
  等价性判断。
- **n 块的 Python round/repair** 未实现；同一 LP 下 `n.out` 差 ≤2.2e-6。

## ④ 参数默认值

唯一事实源：Python 项目的 `config/problem.toml`（只读）。可用
`config_path=` 或 `V2_PROBLEM_TOML` 覆盖路径。当前值：

- `T=60, G=4, N_R=5, L=3`
- `N=[3680,2730,3780,3320,2520]`, `Qmax=[3,2,3,3,2]`,
  `Imax=[12,8,20,15,8]`
- `C=50, h=1, Cs=20, atc=5, tc=10`
- `lambda0=0.007, dirichlet_s=20, p_digits=3`
- `beta_cov=[0.015,0.025,0.010]`, `z_bar=[60,5,65]`
- 协变量 `N(60,4²)` / `Poisson(5)` / `100·Beta(7.8,4.2)`
- `init_mix=[0.420,0.369,0.163,0.048]`, `I0=[0,0,0,0,0]`
- `fleet_tol=1e-6`, `state_round_digits=6`, `state_repair_tol=1e-4`
- `pool_size=300, seed=2026`

数据模块只保留与 Python `config.py::_opt` 一致的兼容默认：`I0` 缺省全 0、
`n0_mode` 缺省 continuous、N_R=5 的优先级表；其余缺键直接报错。

## ⑤ 结果目录约定

本目录**不写任何 results/**，不改 Python 的 `results/`。训练/仿真/割存取
只在用户明确要求时由 SDDP.jl 内置 API 写到调用者指定的路径；不要在本目录
新建结果目录，也不要把 Python 的割文件读进来（格式/模型族不同，且
`FORMAT_VERSION` 不兼容）。

## ⑥ 已踩过的坑

1. **Julia 字符串里 `$var。`/`$var；` 是解析错误**：中文标点会被当成变量名
   的一部分。所有中文串里的插值写成 `$(var)`。
2. **JuMP 1.29 没有 `JuMP.coefficients`**：用 `GenericAffExpr.terms`
   （`for (v,c) in f.terms`）。
3. **SDDP 训练前节点子问题没有求解器**：单阶段检查要先
   `JuMP.set_optimizer(node.subproblem, default_optimizer())`，再
   `SDDP.set_objective(node)`，最后 `JuMP.optimize!`。
4. **Ndef 约束忘了 `set_name`**：`@constraint` 容器默认可能匿名，等价性
   检查会丢掉 70 行。命名循环在 `model.jl`。
5. **虚拟阶段用等式行会多 35 行**：Python 用变量界钉，必须用 `JuMP.fix`。
6. **meta TOML 数组要逗号**：池文本文件用空格，TOML 数组必须 `, `；导出
   脚本分开处理。
7. **COPT `MatrixTol=1e-10`**：低于该量级的 Θ 系数被当 0（Python 与
   Julia-COPT 一致）；HiGHS 保留。逐行比较用 `1e-10` 阈值；SDDP 数值报告
   会提示 small coefficients，但 `numeric issues=0`。
8. **`dot` 需要 `using LinearAlgebra`**（`cumulative_failure_prob` 用到）。
9. **场景池不能默认用 Julia RNG**：NumPy 流不可复现；默认必须加载
   `data/pool_seed<seed>.txt`，否则报错。只有 `allow_julia_pool=true` 才回退。
10. **根节点状态初值必须满足子节点 `.out` 的界**，SDDP 在建图时会检查；
    当前初值均在界内。
11. **不同求解器的对偶可能不同**：阶段成本唯一，但固定 `x_in` 的既约成本
    （= ∂Q/∂x，Benders 割斜率）在退化时不唯一。Python COPT 与 Julia COPT
    完全一致（max|Δ|=0）；HiGHS 与它们差 max|Δ|=50（mean 2.5，主要在
    无当期成本的 `q.in`）。默认用 COPT 生成割；HiGHS 只作可选/回退。
12. **单阶段 COPT 对偶**：`JuMP.reduced_cost` 对 `JuMP.fix` 的 `x_in`
    可用；不要用 `JuMP.coefficients`（1.29 没有），行系数用
    `GenericAffExpr.terms`。

## ⑦ 当前进度与待办

**已完成**

- 数据加载/校验/自检：Python TOML + 导出池；池首场景、均值、P_ref 与
  Python 逐值一致（max|Δ|=0）。
- 模型：T+1 节点纯 LP；虚拟阶段 81 列/5 行；真实阶段 191 列/146 行。
- 等价性：146 行 lb/ub/系数、目标系数、变量界、阶段成本全部一致
  （阶段成本 |Δ|=4.8e-11；仅 n.out 因 Python round/repair 差 2.176e-6）。
- 冒烟：默认 COPT 训练 1 轮，numeric issues=0，LB=140.0277958101；同种子
  HiGHS 的 LB 同为 140.0277958101（`test/smoke_both.jl`）。
- 对偶：同一状态/场景下，Python COPT vs Julia COPT 的状态梯度 max|Δ|=0；
  HiGHS 与两者差 max|Δ|=50（mean 2.5，主要在 `q.in`）。默认 COPT 生成割。

**待办（不在本次范围）**

- 完整 T=60 训练（用 `SDDP.train`，轮数/时限自定）；报告
  `SDDP.calculate_bound` 的 LB，仿真用 `SDDP.simulate` 出 UB/CI。
- 与 Python V2 的 LB/UB 对比（注意 Python 用 COPT + 自定义引擎，数值路径
  不可能逐位一致）。
- 策略分析/热力图（用 `SDDP.DecisionRule` 或仿真记录，另写脚本）。

## ⑧ 文档过时清单

- **Python 文档是问题/方程的权威来源**：`docs/建模与实现.md`、
  `code/problem_data.py`、`code/subproblem.py`、`config/problem.toml`。
  本目录的 README/CLAUDE 只描述 Julia 实现与验证，若与 Python 冲突以 Python
  为准。
- Python `docs/建模与实现.md` 的 §5 示例里出现过 T=240、A=120 等旧参数；
  当前实际口径以 `config/problem.toml` 的 T=60/G=4 为准。
- `state_round_digits` / `state_repair_tol` 在本 Julia 模型中**只读取校验、
  不参与建模**（Python 的仿真/状态流转才会用）。这是已知差异，不是 bug。
- 参考 Julia 项目（`风场多仓库备件库存管理案例`）是月龄模型，与本 V2
  等级模型方程不同；只借用布局/写法，不要照搬其 T=240/A=120 参数。
