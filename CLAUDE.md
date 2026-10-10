# CLAUDE.md

Julia 原版《风场多仓库备件库存管理案例》的 **Python 重写**：pulp 建模 + 自己实现的
多进程 SDDP 引擎。模型语义、时序、结果目录口径与 Julia 版逐项对齐；算法与对偶提取
依据 `SDDP算法Python实现/guide`（已实测的路线 A：状态用变量界锁死 → 读 `.dj` 取割
斜率；`mip=False` 零侵入取松弛对偶）。

**状态（2026-10-10）**：全链路已跑通 —— 数据层 / 子问题 / 引擎 / 仿真 / 结果落盘 /
绘图 / 变量记录 / IIS 导出。对偶自检四项全过。**尚未做**：系统参数定标、完整训练
（默认参数多为占位值）、策略分析、参数敏感性。

**进行中（2026-10-10）**：求解性能改造。已实测确认 coptpy 支持原地改参数 +
自动热启动（见「增量求解」一节与根目录 `COPT使用手册.md`），模型可直接用 coptpy
建，省掉 pulp→COPT 的 extract/LoadProb；后向串行扫描修 LB=0。**两项均未实施。**

接手先读「当前实现」「铁律」两节，细节见根目录 `README.md`（命令/结果约定）与
`docs/工具包说明.md`（utils API 与踩坑）。

## 目录与命令

```
风场多仓库备件库存管理案例-python/
├── run.py                  # 唯一训练入口（位置参数 / smoke / resume）
├── simulate_only.py        # 只仿真：从结果目录恢复问题与割，重跑蒙特卡洛
├── simulate_inspect.py     # 仿真 + 变量记录（CSV 长表，策略分析用）
├── AnalysisModel_IIS.py    # LP 冲突集分析（COPT/Gurobi），与 Julia 版同一份
├── COPT使用手册.md          # coptpy 8.0 用法与坑（建模/改参数/对偶/存取/MIP）
├── code/problem_data.py    # ProblemData + DefineProblem + 威布尔/PHM + (z,θ) 池
├── code/state_layout.py    # 状态布局 I | o | n | q
├── code/model.py           # StageSubproblem（单场景子问题）
├── utils/                  # SDDP 工具包，与问题解耦，可整目录迁移
├── test/                   # 自检脚本（_t_*.py），不进主链路
├── plot/plot_bounds.py     # 收敛图（表头解析，与 Julia 版同一份）
└── results/<tag>_<时间戳>/  # 每次训练一个自包含目录
```

求解器侧自检：`test/_t_solver.py`（可用性）、`test/_t_copt_survey.py`（coptpy 全能力
47 项）、`test/_t_mut_coptpy.py`（原地改参数 + 对偶差商 25 项）、`test/_t_mut_dll.py`
（原生 DLL 对照 21 项）、`test/_t_copt_params.py`（参数/容差/线程实测）。

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python"
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" run.py smoke         # 冒烟
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" run.py 200           # 训练 200 轮
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" run.py 200 resume results/<目录>
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" simulate_only.py results/<目录> [路径数] [块大小]
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" plot/plot_bounds.py
```

- 解释器固定用 **MyRL 环境**（pulp 3.3.0 / numpy 2.3.4 / COPT_DLL，许可到期
  2026-11-06）；`conda run` 会因插件报错，**必须直接调 python.exe**。
- 命令行只有位置参数，没有 `--flag`；默认值只在 `run.py` 的 `main` 签名里改。
- 训练默认走 `multiprocessing`（Windows spawn）。`n_processes=1` 退化为串行，
  traceback 完整，调试时用这个。

## 当前实现（要点，细节见 docs/）

**阶段结构**：阶段 0 为虚拟首阶段（钉初值 + 决定首期订货），1..T 为真实周期。
`cuts[t]` 近似 V_t，第 t−1 期子问题用 `cuts[t]` 约束未来成本变量 theta。

**状态**（615 维，N_R=5/A=120/L=2）：`I(N_R) | o((L−1)·N_R) | n(N_R·A) | q(N_R)`。
状态变量**只用变量界锁死在试验点**（路线 A），读 `.dj` 即得 ∂Q/∂x。

**子问题**：单场景（`F = round(θ·n_in)` 作为常数代入），决策 taup/taum/y/S_end 为整数，
q 是 here-and-now（期 τ 的订单在见到 θ_τ 前定下）；目标五项
`C·Σq_in + h/2·(ΣI_in+ΣI_out) + Cs·Σ(N−Σn_in) + atc/2·Σ(τ⁺+τ⁻) + Σ tc_route·y + theta`。

**割**：`Cut(alpha, beta, state, stage, obj, it)`，截距式 `V(x) ≥ alpha + beta @ x`
（`alpha = obj − beta @ state`）。不做筛选，每条割无条件入库。存 `cuts.json`（原子写）。

**两种求解模式**：前向/仿真 `mip=True` 取整数策略；后向/下界 `mip=False` 读松弛对偶。
割是**松弛值函数**的下界，对整数问题不保证严格有效 —— 与 Julia 版
`handler=:continuous` 同源。

## 铁律（用户定下的规矩，违背会被打回）

1. **不改 results/ 里的任何东西**：不删除、不覆盖、不挪动。训练/仿真只往里面
   **新增**目录或新文件。
2. **不要写乱七八糟的注释**：注释少而准，解释"为什么"（尤其非显然的），不写废话。
3. **达梦数据库服务不碰**（dmsvc.exe / DmServiceDMSERVER）：它占着约 10 GB commit，
   杀掉会引发连锁崩溃，与项目无关但在这台机器上。
4. 模型结构一旦改过（约束/状态/目标），**旧结果目录的割一律不兼容**，必须全新训练，
   不要 resume 旧目录；`simulate_only.py` 对旧目录同样失效。
5. Windows + 中文路径：Git Bash 里把中文路径放进变量会被编码搞坏，要么整条命令用
   引号包住绝对路径，要么用专用文件工具。
6. **临时/自检脚本一律放 `test/`**，不要写在项目根目录。

## 与 Julia 版的已知差异

| 项 | Julia 版 | Python 版 |
|----|---------|-----------|
| 建模 | JuMP | pulp（目标 `+=` 是覆盖语义，只能写一次）|
| 并行 | `SDDP.Threaded()`（`-t auto`）| `multiprocessing`（默认核数）|
| 场景池共享 | 全阶段共享 | 同 |
| 割存储 | JLD2/json | `cuts.json`（Cut 完整字段）|
| 前向路径数 | `InSampleMonteCarlo` | `n_forward_paths`（默认 1）|
| 后向场景 | — | 全枚举（可 `n_backward_samples` 抽样）|

## 增量求解：coptpy 原地改参数（2026-10-10 实测确认，方案已定型，待实施）

**用法与坑的完整清单见根目录 `COPT使用手册.md`**（每条都有 `test/_t_copt_survey.py`
等脚本复现）。这里只记结论与对本项目的意义。

**旧结论已推翻**：先前记的"coptpy 模型建成即不可变，只能整模重建，改界要靠 ctypes
直调原生 DLL"**是错的**——当时只测了 `getRow()` 的副本语义，没找到正门。正门是
`m.setInfo(COPT.Info.LB/UB, 对象, 值)`，COPT 的 LB/UB 定义本就同时覆盖列与行
（官方示例 `locationtransport_lr.py` 用的就是它）。

**实测能力**（`test/_t_mut_coptpy.py` 25/25 + `test/_t_copt_survey.py` 38/38 全过）：

| 能力 | 接口 | 与重建模型比对 |
|------|------|----------------|
| 改变量界 | `setInfo(Info.LB/UB, [var], v)` | 逐位一致 |
| 改行右端 | `setInfo(Info.LB/UB, [con], v)` | 逐位一致 |
| 改目标系数 | `setInfo(Info.Obj, [var], v)` | 逐位一致 |
| 改矩阵系数 | `setCoeff(con, var, v)` | 逐位一致 |
| 取对偶 | `getInfo(Info.Dual/RedCost/…, …)` | 差商验证 π/dj 即真灵敏度 |
| 追加割 | `addConstr` 后直接 `solveLP()` | 无需重建 |

**行界的内部表示**：`>=b` → LB=b/UB=+∞；`<=b` → LB=−∞/UB=b；`==b` → [b,b]。
改右端按此行别选 `Info.LB` 或 `Info.UB`。

**热启动是自动的**：原地改完界直接 `solveLP()`，COPT 复用上次最优基——400 变量链式
LP 首解 199 次单纯形迭代，改界后连续重解 **0 次迭代**。不需要手动 `setBasis`。

**计时**（真实冒烟规模子问题 180 列/100 行/324 非零）：

| 做法 | 建模 | 求解 |
|------|------|------|
| coptpy 直接建模 | 1.02 ms | 0.25 ms（首解） |
| pulp 建模 + `COPT_DLL`（现行） | 1.00 ms | 2.33 ms（每次含 extract+LoadProb） |
| coptpy 原地改界+重解 | — | **0.19 ms/次** |

建+解合计 coptpy **1.27 ms** vs pulp **3.33 ms**。600 列/400 行放大实例：原地
0.22 ms vs 重建 3.93 ms（17.7×）。

**★ 决定：不再实现 `utils/copt_dll.py`（ctypes 直调原生 DLL）。** 原生 DLL 能做的
coptpy 全都能做，原地改的耗时几乎相同（0.23 vs 0.22 ms），却多一层手写原型、
缓冲区大小核对（实测踩过越界写段错误）与 348 个函数的手工接线。旧计划里那条路线
整条作废。

**★ 四条必须遵守的安全规则**（详见手册 §0，都是实测踩出来的）：
1. **文件接口不认中文路径**：`writeLp/writeMps/writeBin/writeSol/writeIIS/setLogFile`
   遇非 ASCII 路径失败，短路径还会**静默不建文件**。本项目路径全中文 → 落盘一律
   先写 `tempfile.gettempdir()` 再搬回。这修正了旧计划里"失败兜底用 `COPT_WriteLp`
   导出现场"的写法。
2. **按求解类型读目标值，不要用 `objval`**。两重坑：① 求解失败时 `objval` 与
   `LpObjVal` **都静默返回上一次的陈旧值**；② **`objval` 对 LP 值有粘性**——一旦跑过
   `solveLP()`，此后无论 `solve()` 多少次，`objval` 永远返回 LP 值，而 `getValues()`
   返回的是整数解，两者脱节（实测 objval=1.5 而解向量算得 2.0，`Status` 照样报
   OPTIMAL 判不出来）。正确读法：**MIP 读 `BestObj`，LP 读 `LpObjVal`**；每次求解后
   先查 `status != OPTIMAL` 就中止。**同一模型交替跑 MIP/LP 没问题**，按类型分开读
   即可 —— 不要为这个开两个模型（内存优先）。注：pulp 路径本身安全
   （`getsolution` 查了 `LpStatus==1`）。
3. **多进程并行时每个 worker 设 `Threads=1`**（2026-10-10）：COPT 的 `Threads`
   默认 -1 = **自动用满核**。训练走 `multiprocessing` 默认核数进程，12 进程 × 28 线程
   严重超订。实测 3000 变量 LP 单线程 0.020 s vs 自动 0.040 s——**单线程反而快一倍**。
   并行度留给进程级，进程内不再并行。`SimplexThreads` 默认 -1 跟随 `Threads`。
4. `resetParam()` 一次复位**全部**参数；`tupledict + list` 传 `setInfo` 会
   `TypeError`（`addVars` 返回的正是 tupledict），要么 `list(d.values())` 要么传标量。

**跨期结构复用方案**（替代原 DLL 版）：每（期 × worker）一个长驻 coptpy `Model`，
首次建好结构，此后每轮只 `setInfo` 改状态界（615 个 xin）+ 改 `dem_r / age_{r,a}`
右端，`addConstr` 追新割，直接 `solveLP()` 取 `getRedcosts()`。虚拟首阶段单独一个小
模型。内存粗估 201 模型/worker × ~2-4 MB ≈ 0.8-1.5 GB/worker，1 路径可行、12 路径
要评估。

**建模源**：子问题改为**直接 coptpy 建模**（省掉 pulp 构造 + `extract()` +
`LoadProb()`），`code/model.py` 从 pulp 迁到 coptpy。迁移后必须用
`utils/diagnostics.py` 全套（`.dj` vs 通用式、次梯度不等式、割合法性）与旧 pulp
路径逐位对拍，确认对偶口径不变。

**与后向串行扫描正交**：串行扫描修 LB=0，增量求解修单次成本，两者独立、可叠加。

## 已踩过的坑（照做即可，完整清单见 docs/工具包说明.md §踩坑）

- **Julia→Python `enumerate` 差一**：`tc_route=(pos−1)·tc` 必须 `start=1`，否则本地
  调拨成本变成 −tc，负成本会把 LP 解带跑偏（已修）。
- **pulp 常量目标返回 None**（虚拟首阶段无割时），`obj` 属性要退回 `constant`。
- **预求解会消掉"只被界钉住"的变量**，虚拟首阶段 `xout` 求解器不回值 → 退回界值。
- **`pulp.LpProblem` 目标只能写一次**：多处加项统一在 `_finish()` 合成。
- **探测状态必须按约束构造**：本模型 Σ_a n_in[r,a] 与 N_r 有硬耦合，逐分量随机取整
  几乎必然不可行（用多项分布切分 N_r）。
- **差商容差按量级放缩**：h=1e-4 会把求解器容差放大 1e4 倍，绝对 1e-6 会误报。
- **Windows 控制台 GBK**：入口脚本统一 `sys.stdout.reconfigure(errors="replace")`。
- **失败兜底的 `export_lp` 要先建目录**：否则目录不存在时异常被吞、`path` 静默变
  `None`，最需要现场 LP 的时候反而没导出。
- **COPT 文件接口不认中文路径**（2026-10-10）：`writeLp/writeMps/writeBin/writeSol/
  writeIIS/setLogFile` 遇非 ASCII 路径失败，短路径**静默不建文件**。本项目路径全
  中文 → 落盘先写 `tempfile.gettempdir()` 再搬回。`writeLp` 并非"另一条路"，它只是
  `write` 的显式版本，同一个 C 文件接口。
- **求解失败时 coptpy 的 `objval` 是陈旧值**（2026-10-10）：`m.objval` 与
  `Attr.LpObjVal` 在不可行/失败后都静默返回**上一次**的值，`m.status` / `HasLpSol`
  才是可靠的守卫。每次 `solveLP()` 后必须查 status 再读结果（`getValues` 等会
  正确抛错，但目标值不会）。pulp 路径本身安全（`getsolution` 查了 `LpStatus==1`）。
- **coptpy 的 `objval` 对 LP 值有粘性**（2026-10-10）：同一模型上 `solveLP()` 之后
  再 `solve()`，`objval` 永远返回 LP 目标，而 `getValues()` 返回的是整数解——两者
  对不上，且 `Status` 照样报 OPTIMAL 判不出来。改用 `BestObj`（MIP）/ `LpObjVal`（LP），
  或干脆让前向 MIP 与后向 LP 各用一个模型。
- **把 MILP 松弛成 LP 有三种做法**（2026-10-10）：`m.solveLP()`（非破坏性，首选）、
  `m.setVarType(vars, COPT.CONTINUOUS)`（永久改，可改回）、逐变量混合松弛。三种都能
  提到松弛对偶。`IntTol` 调大不管用（照旧出整数解）。
- **COPT 容差默认值**（2026-10-10）：`FeasTol=1e-6`（可行性）、`DualTol=1e-6`（对偶
  可行性）、`IntTol=1e-6`（整数）、`MatrixTol=1e-10`（矩阵系数，更小的被当 0）、
  `RelGap=1e-4`/`AbsGap=1e-6`（MIP 双 gap）。实测 FeasTol 边界：冲突量 ≤1e-6 判可行、
  >1e-6 判不可行；MatrixTol 边界：(1e-11 及更小的系数被丢掉，本项目 (1−θ)~1e-5 安全。
  取参数用 `m.getParamInfo(COPT.Param.X)` → `(名, 当前, 默认, 最小, 最大)`；
  `-1` 在多数参数里表示"自动选择"而非"关"。完整 76 个参数见 `test/_t_copt_params.py`。
- **`m.getRow(con)` 是副本语义**：有 `setConstant/setCoeff` 但改完重解不生效。改行
  右端一律走 `m.setInfo(COPT.Info.LB/UB, [con], v)`；改矩阵系数用 `m.setCoeff(con,
  var, v)`（这个是有用的）。
- **`m.setInfo` 的容器分派**：`tupledict + list` → `TypeError`（`addVars` 返回的正是
  tupledict）。要么 `list(d.values())`，要么传标量广播。
- **`loadMatrix` 灌进来的变量/约束没有名字**，`getVarByName/getConstrByName` 取不到，
  只能 `m.getVars()[i]` 按位置索引。
- **`resetParam()` 不带参数**，一次复位全部参数，没有单项复位。
