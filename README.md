# 风场多仓库备件库存管理案例 — Python 版（coptpy + 多进程 SDDP）

> **★ 2026-10-24 当前模型：等级模型（纯 LP）**
> 4 个运行健康等级 + 失效态 4；Poisson 增量 λ0=0.007；三位小数转移矩阵；
> 逐行 Dirichlet；`N=Θn`；前向/后向都解 LP；状态维 40（T=240、L=3）。
> `fleet`/`dem` 右端带 `fleet_tol=1e-6`；状态流转时 n 块 round 6 位，
> 超额 ≤`state_repair_tol=1e-4` 从最大分量扣除（不缩放），超过才报错。
> 当前实现见 `docs/建模与实现.md`；下文“Julia 原版/月龄/MIP/420 维”
> 等描述是旧模型历史，旧结果/割 v2/v3/v4 不可 resume（当前 `FORMAT_VERSION=5`）。
> 新模型验收：`test/_t_grade_calib.py`、`_t_grade_equiv.py`、`_t_grade_cutvalid.py`；
> 更新后的 `_t_config/_t_assembly/_t_smoke/_t_sweep/_t_engine/_t_det`；
> resume 验收：`_t_resume.py`；旧 age/rint/整数测试仅作历史。

Julia 原版（`风场多仓库备件库存管理案例`）的 Python 重写。模型、时序、结果目录口径
与 Julia 版逐项对齐；建模用 coptpy 直接建（原地改参数 + 自动热启动），SDDP 算法本体
（对偶提取、割的表示与存储、多进程调度）自己实现，放在可整目录迁移的 `utils/` 包里。

## 目录

```
├── config/
│   ├── problem.toml        # 问题参数（维度/机队/成本/退化/协变量/场景池）—— 唯一事实源
│   ├── solver.toml         # 求解参数（轮数/并行/仿真/容差/输出）—— 唯一事实源
│   └── smoke/              # 冒烟与自检用的极小实例，格式完全相同
├── run.py                  # 唯一训练入口
├── simulate.py             # ★ 唯一仿真入口：全局统计 + 详细数据落盘 + txt 报告
├── analyze_sim.py          # ★ 读仿真数据做事后分析（成本结构 / 调拨 / 订货 / 轨迹）
├── AnalysisModel_IIS.py    # LP 不可行冲突集分析（COPT/Gurobi），与 Julia 版同一份
├── COPT使用手册.md          # coptpy 用法与坑的完整清单（建模/改参数/对偶/存取/MIP/容差）
├── code/
│   ├── config.py           # 读 TOML + 校验 → ProblemConfig / SolverConfig
│   ├── problem_data.py     # ProblemData / DefineProblem(cfg) / 威布尔 / PHM / (z,θ) 池
│   ├── state_layout.py     # 状态向量布局：I | o | n | q
│   ├── subproblem.py       # StageSubproblem：base 构建 / 割拼装 / 改点求解 / 对偶
│   └── sim_analysis.py     # 仿真口径：状态重建 / 五项成本拆解 / 约束审计（两脚本共用）
├── utils/                  # 与问题解耦的工具包，可整目录搬到其他 SDDP 项目
│   ├── copt_env.py         # coptpy 环境、参数、ASCII 落盘兜底
│   ├── payload.py          # 割集跨进程传递（shared_memory + pickle 回退）
│   ├── worker.py           # spawn worker 装配、2 个长驻模型、三类任务函数
│   ├── engine.py           # SDDPEngine：前向并行 / 后向阶段串行 / 割池 / 训练循环
│   ├── cut.py              # Cut 类（alpha/beta/state/stage/obj/it）+ JSON 序列化
│   ├── simulation.py       # 蒙特卡洛：分块、流式统计、95% CI
│   ├── diagnostics.py      # 对偶自检：.dj vs π 通用式、次梯度不等式、割合法性
│   └── io.py               # 结果目录 / 配置副本 / 割存取 / 训练日志（原子写）
├── plot/plot_bounds.py     # 训练收敛图（表头解析，与 Julia 版同一份）
├── test/                   # 自检脚本（见文末「自检脚本」）
└── results/<tag>_<时间戳>/  # 每次训练一个自包含目录
```

## 命令

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"     # conda run 会报错，直接调 python.exe

$P run.py                                   # 用 config/ 的配置（默认值多为占位值）
$P run.py --config-dir config/smoke         # 冒烟：小实例快跑，验证链路连通
$P run.py 300                               # 覆盖轮数（其余仍走配置）
$P run.py 300 2000                          # 覆盖轮数 + 仿真路径数
$P run.py 200 resume results/<目录>          # 续训 200 轮

$P simulate.py results/<目录> [路径数] [块大小]   # 仿真一次：统计+落盘+报告
$P analyze_sim.py results/<目录>                # 读盘分析，不重跑
$P plot/plot_bounds.py                      # 默认画 results/ 下最新的一个目录
$P AnalysisModel_IIS.py <现场.lp> --solver copt
```

## 参数在配置文件里，不在代码里

参数的**唯一事实源是 `config/*.toml`**，代码里没有第二处默认值；缺键直接报错。
启动时会打印生效配置全文，并把 TOML **原样拷贝**进结果目录（`config/`，注释保留），
`simulate_only.py` / `simulate_inspect.py` 再从那份副本恢复口径。

`config/problem.toml`：`[dimension]` `[fleet]` `[cost]` `[degradation]` `[covariate]`
`[scenario]`，另可选 `[priority]`。

`config/solver.toml`：

| 键 | 默认 | 说明 |
|----|------|------|
| `[training] iterations` | 200 | 训练轮数 |
| `[training] forward_paths_per_process` | 1 | **实际前向路径数 = 本值 × n_processes** |
| `[training] n_backward_samples` | 30 | 后向每试探点采样场景数；0 = 全枚举 |
| `[training] checkpoint_interval` | 600.0 | 定时存割的间隔（秒）|
| `[training] seed` | 2026 | 随机种子 |
| `[training] cut_selection` | lml1 | 割筛选：`lml1`（docs/LML1割筛选算法逻辑.md）/ `none` |
| `[training] cut_select_eps` | 1e-6 | LML1 的 ε0：判两条割值是否并列 |
| `[training] cut_select_keep_rounds` | 0 | 试探点保留轮数；0 = 全留（`|S|` 无硬上界）|
| `[parallel] n_processes` | 0 | **0 = 自动（物理核数 // 2）**；1 = 串行（调试）|
| `[parallel] threads_per_process` | 1 | COPT `Threads`；多进程下必须 1 |
| `[simulation] n_simulations` | 10000 | 仿真路径数 |
| `[simulation] chunk_size` | 1000 | 仿真分块（块种子 = seed + k）|
| `[numerics] feas_tol / dual_tol / int_tol / presolve` | 1e-6 / 1e-6 / 1e-6 / -1 | COPT 容差 |
| `[output] tag / results_root` | base / results | 结果目录前缀与根目录 |

命令行位置参数是**显式覆盖**（覆盖了谁就在启动时回显谁），不是默认值。

## 结果目录（`results/<tag>_<时间戳>/`）

| 文件 | 内容 |
|------|------|
| `config/problem.toml`、`config/solver.toml` | 生效配置的**原样拷贝**（含注释）|
| `problem_data.json` | 问题数据 + 预采样场景池 |
| `cuts.json` | 全部割：`{meta, version, cuts: {期: [{alpha,beta,state,stage,obj,it}]}}` |
| `sddp_training_log.csv` | 训练日志，表头与 Julia 版一致，`plot_bounds.py` 直接解析 |
| `bounds.png` | 收敛图（`plot/plot_bounds.py` 画）|
| `sim_<n>paths.npz` | ★ 仿真详细数据：每期每变量的完整数值（`simulate.py` 写，≈35 MB/500 条）|
| `sim_<n>paths.json` | ★ 该次仿真的元数据：路径数/块/种子/UB/CI/LB/gap/耗时/审计结论 |
| `sim_report_<时间戳>.txt` | ★ 仿真 txt 报告（`simulate.py` 写）|
| `subproblem_<环节>_t<期>_s<场景>.lp` | 仅在求解失败时导出的现场 LP（供 IIS 分析）|

**只新增、不覆盖**：训练/仿真只往结果目录里写新文件，或先写 `.tmp` 再原子改名。
模型结构（约束/状态/目标）一旦改动，旧目录的割一律不兼容，必须全新训练，不要 resume。
`run.py` 会比对两边的问题配置，`load_cuts` 还会校验 `FORMAT_VERSION` ——
**维度对不上或版本对不上都直接拦下**（铁律 4）。

## 算法口径

- **前向**：一任务 = 一条路径，逐期解 MIP 得策略；收齐每阶段试探点与总成本（上界）。
- **后向**：**按阶段串行（T → 1），期内按试探点并行**；一任务 = 一个试探点 = 一条割
  （逐场景解 LP 松弛取 `.dj`，等概率平均）。本阶段割全算完再一起入池，然后推进到上一
  阶段 —— 这样期 t 立刻用上同轮期 t+1 的新割，一次反向扫描传遍全视界。
  （旧实现轮初打快照、全部任务一次铺满，信息每轮只传播一期，T=200 时下界上百轮不动。）
- **模型复用**：每 worker 只持 2 个 coptpy 模型——阶段 0 一个 + 服务所有 t≥1 的一个
  （阶段 1..T 的 base 结构完全相同，只换割行）。同一阶段内所有场景/试探点复用同一模型，
  只改变量界与行右端，**不重建模型**。
- **求解模式**：前向 `solve()`（**MILP**：全部状态变量 + `y`/`τ` 都是整数）取整数策略；
  后向 `solveLP()`（**同一模型的 LP 松弛**）读对偶。松弛值 ≤ 整数最优 ⇒ 松弛对偶割
  是整数问题的**合法下界**（标准 SDDP-for-MILP 框架，与 Julia 版
  `handler=:continuous` 同源）。
- **失效量 F 的写法是割合法性的关键**：`F` 必须是**整数变量**，由
  `F − θ·n_in ≤ 0` / `≥ −1` 两条线性式定义（右端只有纯结构常数）。θ 以**系数**
  进矩阵，对偶才看得见 `∂F/∂x`。曾写成 `F = rint(θ·n(x))` 塞进约束右端 ⇒ 割的斜率
  少一整项 ⇒ **系统性越界、负 gap**。推导与证据见 `docs/割越界根因分析.md`。
- **割筛选用 LML1**（`docs/LML1割筛选算法逻辑.md`）：每期按历史试探点，只保留「在该点取值
  最大、且生成最早」的那条割，其余**删除**。冒烟 12 轮实测割数 100 → 36。

## ★ 仿真（`simulate.py` / `analyze_sim.py`）

**生产与消费分离**：仿真跑**一次**（500 条 ≈ 8 分钟），把每期每变量的完整数值落成
`.npz`；此后所有统计都读盘，秒级出结果，**不重跑**。

```bash
$P simulate.py results/<目录> [路径数] [块大小]   # ① 跑仿真 → 统计 + 落盘 + txt 报告
$P analyze_sim.py results/<目录>                  # ② 读盘分析（可反复跑，不重仿真）
```

### `simulate.py` 出三样

1. **全局统计**：路径数、块大小、种子、规模、训练轮次/割数、UB 均值、标准差、
   标准误、95% CI、最大最小值、LB（只读割）、gap、求解次数、仿真用时；
2. **详细数据** `sim_<P>paths.npz` + `sim_<P>paths.json`；
3. **txt 报告** `sim_report_<时间戳>.txt`：上面这些 + 成本结构（五项占比 + 对拍）
   + **逐 (路径,期) 点的约束审计结论** + 逐期平均成本。

**分块**：块种子 = `seed + k`（与训练/旧脚本同一口径 ⇒ 结果依赖块大小，报告里记下来）。
分块同时把内存封在「一块」的量级 —— 块内结果要先攒成记录，所以**块越大越吃内存**；
路径数上万时把块调小（如 `simulate.py <目录> 10000 500`）。同名 `.npz` 已存在会
**拒绝覆盖**（铁律 1）。

### `.npz` 里有什么（形状 `(路径, 期, ...)`）

| 数组 | 内容 |
|---|---|
| `x_in` / `x_out` | **完整状态向量**（615 维 `I \| o \| n \| q`，用 `StateLayout` 切视图）|
| `q` / `taup` / `taum` / `S_end` / `y` | 该期**全部决策** |
| `terms` | 五项成本 `(采购, 持有, 停机, 主动调拨, 被动调拨)`，逐期 |
| `cost` / `theta` | `stage_cost` 与未来成本变量取值 |
| `scene` / `totals` | 场景号 / 单路径总成本 |

状态与决策都是整数 ⇒ 存 **float32 无损**（24 位尾数覆盖到 16.7M）。
`F` 不存：`F = ⌊θ·n⌋` 由 `(scene, x_in)` 现算即可，省 240 MB。

### `analyze_sim.py` 出五节

① **成本结构**：五项总量/占比、最贵的 8 期（含该期内部构成）、视界五等分、按仓库；
② **主动调拨 τ**：激活程度、`τ⁺/τ⁻` 并存（已知瑕疵，见 `docs/待查问题.md`）、按仓净流；
③ **被动调拨 y**：本地/跨仓、跨仓量按路线费是否**单调递减**（优先级梯度是否生效的判据）、
互补松弛；
④ **订货 q**：贴 `Qmax`/容量上限的比例、`q=0` 比例、**base-stock 形态**（订货前位置分箱
→ q）、对失效的**超前性**（`L=2` ⇒ 滞后 1 期应有正相关）；
⑤ **逐期轨迹**：库存 / 在途 / 停机 / 订货 / 失效 / 平均月龄。

### 口径只有一处

状态重建（`out_state`）、五项成本拆解（`cost_terms`）、约束审计（`check_stage`）全部
定义在 **`code/sim_analysis.py`**，`simulate.py` 与 `analyze_sim.py` 共用 —— 不会出现
「落盘按一套口径、分析按另一套」。

> 历史脚本 `simulate_only.py` / `simulate_inspect.py` 已被上面两个取代（前者只出统计、
> 后者只出 CSV 长表），保留未删。

## ★ 可复现性边界

同一 seed 两次跑，**结论统计等价但不逐位一致**（下界相对差 ~1e-2 量级）。原因是
LP 最优对偶可能不唯一（对偶退化）、共享模型热启动取到哪个最优对偶取决于模型使用史、
以及进程池任务分配不固定。第 1 轮的割跨进程数完全一致，第 2 轮起从退化点分岔。

**不影响正确性**（每个最优对偶都是合法次梯度，割总是有效下界）；但论文若要逐位可复现，
需每次求解前清基（丢掉热启动）或单进程跑。详见 `test/_t_det.py`。

## 自检脚本（`test/`）

临时脚本一律放这里，**不要写进项目根目录**。每个脚本自带 `sys.path` 引导，可直接从
根目录跑。建进程池的脚本都带 `if __name__ == "__main__":` 守卫（Windows spawn 必需）。

| 脚本 | 验什么 |
|------|--------|
| `_t_equiv.py` | **建模等价性（硬门槛）**：coptpy 重写 vs 旧 pulp 实现，obj/dj/π 逐位一致 |
| `_t_assembly.py` | 拼装契约：base 只含阶段成本、挂割/卸割、换割集、空割集 |
| `_t_config.py` | 配置生效性：改 TOML 必生效、缺键报错、派生值换算、非法值拦截 |
| `_t_sweep.py` | 后向串行扫描：T=8 六轮，断言**下界在 3 轮内 > 0** |
| `_t_det.py` | 可复现性边界：第 1 轮割跨进程一致、LB 统计等价、次梯度合法 |
| `_t_engine.py` | 引擎端到端：割自洽 + 试探点 + 变量记录 + 仿真 |
| `_t_smoke.py` | 对偶自检四项（`.dj` vs π、次梯度不等式、割合法性）|
| `_t_iis.py` | 造不可行子问题 → 引擎兜底导出 LP → 交冲突集分析 |
| `_t_solver.py` | coptpy 可用性（LP/MIP/对偶/参数/中文路径落盘）|
| `_t_copt_params.py` | COPT 容差默认值与边界、MatrixTol、Threads 实测 |
| `_t_mut_coptpy.py` / `_t_mut_dll.py` | 原地改参数 + 对偶差商；原生 DLL 对照 |
| `_t_copt_survey.py` | coptpy 全能力巡检 + 计时对照 |
| `_t_nint.py` | 状态变量类型（应全为 INTEGER）+ 前向路径的数值整性 |
| `_t_cutvalid.py` | **逐割验收**：同场景对拍（割是否越界）+ 抽样误差分离 |
| `_t_fparam.py` | F 参数化后，生产模型本身的割合法性扫描 |
| `_cvx_rint.py` / `_cvx_fix.py` / `_cvx_Fvar.py` | 负 gap 根因诊断（ν=∂Q/∂F、系数化等价性、三口径对照）；对**旧 v1 目录**跑，需绕过版本闸门 |

`test/_pulp_ref_model.py` 是**历史对照件，不是生产路径**，只被上面两个对照脚本引用。

```bash
$P test/_t_iis.py
$P AnalysisModel_IIS.py test/_tmp_iis/subproblem_backward_t1_s0.lp --solver copt
```
