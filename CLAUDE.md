# CLAUDE.md

> **★ 2026-10-24 当前模型：等级模型（纯 LP）**
> 4 个运行健康等级 + 失效态 4；Poisson 增量 λ0=0.007；三位小数转移矩阵；
> 逐行 Dirichlet；`N=Θn`；前向/后向都解 LP；状态维 40（T=240、L=3）；
> `fleet`/`dem` 右端带 `fleet_tol=1e-6`；状态流转时 n 块 round 6 位，超额
> ≤`state_repair_tol=1e-4` 从最大分量扣除（不缩放），超过才报错。
> 模型文档见 `docs/建模与实现.md`。**下文关于月龄/Weibull/Beta/F=rint/
> 前向 MILP/615/420 的段落是旧模型历史**，代码已按等级模型重写；
> 旧结果与割 v2/v3/v4 不兼容，当前 `FORMAT_VERSION=5`，不能 resume。
> 新模型验收：`test/_t_grade_calib.py`、`_t_grade_equiv.py`、`_t_grade_cutvalid.py`；
> 更新后的 `_t_config/_t_assembly/_t_smoke/_t_sweep/_t_engine/_t_det`；
> resume 验收：`_t_resume.py`；旧 age/rint/整数测试仅作历史。

风电机队齿轮箱备件的**多仓库库存与调拨**优化：风机按月龄状态退化（威布尔 +
比例风险），失效形成备件需求；N_R 个仓库共享库存（主动调拨 + 按优先级被动调拨），
订货有提前期 L。**coptpy 直接建模 + 自研多进程 SDDP 引擎**：前向是 **MILP**（全部
状态变量 + `y`/`τ` 都是整数变量），后向是它的 **LP 松弛** —— 松弛值 ≤ 整数最优，故
松弛对偶割天然是整数问题的合法下界。0 二元、0 Big-M。

**状态（2026-10-10）**：重构完成并全链路验收通过。建模从 pulp 迁到 coptpy（原地改
参数 + 自动热启动），参数全部进 TOML，后向改期间串行扫描（修好了 LB 长期为 0）。
**已修掉负 gap**：失效量 F 曾是「算成数字塞约束右端」，对偶看不见 ∂F/∂x ⇒ 割系统性
越界（生产 −5.10%）。改成 **F 是整数变量 + 两条线性式定义**后转正（冒烟 +0.23%）。
详见 `docs/割越界根因分析.md`。
**已做**：仿真工具整合成 `simulate.py`（一次跑完：统计 + 详细数据落 npz + txt 报告）
与 `analyze_sim.py`（读盘分析），取代原来散在 `test/` 的一次性脚本。
**已做**：**系统参数定标并落地**（2026-10-12）—— 由「质保期内失效占 10%」+「单仓年失效
≤100 台」+「规划周期 < 寿命」反解，取文献依据 `m=2.5`、`η=148 月`（≈ZF 厂家 100,000
运行小时）、`T=60 月`（=质保期）、`A=80 月`、`N=[1050,780,1080,950,720]`（共 4,580 台）。
配套把 `Fdef` 右端常数从 `0/−1` 改为 `±0.5`（即 `F` 由 `⌊θn⌋` 改为 `rint(θn)`，取整损失
从 −20%~−35% 降到 −1%~−3%），并改掉硬寿命截断的警告判据（改看「视界内是否触达 A」）。
完整推导、文献出处与校验见 `docs/系统参数定标.md`；复现脚本 `test/_t_param_calib.py`
（反解）与 `test/_t_roundoff.py`（取整精度）。**旧结果目录的状态维 615 ≠ 新 415，
割不兼容，必须全新训练。**
**尚未做**：完整训练、策略分析、参数敏感性。

接手先读「命令与配置」「当前实现」「铁律」三节；`COPT使用手册.md` 是 coptpy 用法与
坑的完整清单，`docs/工具包说明.md` 是 utils API 与踩坑。

## ★ 创新分支工作流（基线代码：ada18c3）

**2026-10-25 起，`main` 的 `ada18c3`（2026-10-25 15:54:21）是创新前代码基线。**
这一版是当前最合适的起点；任何算法创新都从它新建分支，不在 main 上直接改。
main 可以增加**只含文档/流程**的提交（例如本节的验收协议），代码基线仍以
`ada18c3` 为准。

1. **先确认基线**：`git switch main`，确认
   `git merge-base --is-ancestor ada18c3 HEAD` 成功；然后
   `git switch -c innov-<主题> main`。不要从上一个创新分支再开分支，避免把失败实验带进来。
2. 在新分支实现、跑测试和实验；结果目录只新增，不改/不删旧结果（铁律 1）。
3. 验证通过后提交并推送：`git push -u origin innov-<主题>`。
4. **可行**：保留分支，经用户确认后才合入 main；不直接 push main。
5. **不可行/无收益**：写清失败原因、参数和数据目录，然后 `git switch main` 回到基线；
   已推送的创新分支保留作证据，不删除、不合并。下一次创新重新从 main 开新分支。
6. 禁止 force-push、改写 main 历史、删除基线分支；`ada18c3` 是代码回退点。
7. 远端 `origin/main` 目前仍在 `4ccb80d`；需要远端基线时先单独 `git push origin main`，
   之后 main 只接受验证通过的合并。

### ★ 算法验收协议（2026-10-25 起）

- **统一 T60** 做算法 A/B；训练预算与基线保存割对齐：**3061.7 s**（基线
  `results/base_T60_20261025_131043` 的 `cuts.json` 检查点），或明确记录与基线
  相同的墙钟预算。
- 仿真口径固定：**2000 路径，seed=2026，chunk_size=1000**；用 `simulate.py`
  出详细报告和约束审计，再用逐路径配对比较 UB。
- 每个创新先跑 **pilot/预检**；预检不通过不投入 1 h/T240。所有进入 LB 的割必须
  合法；割生成/状态采样方式不限，但不能用场景子集割冒充全池 LB。
- 验收记录必须写：基线/对照目录、训练轮次与用时、LB/UB/gap、2000 路径 CI、
  cuts/LP 数、约束审计、失败原因或收益。报告写到**仓库外**：
  `D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python-创新文档\`
  （不在 git 内，避免分支回退覆盖）；分支里只保留代码和可复现 pilot。
- 不可行：先把创新分支推送，再回退 main；外部报告保留，下一轮重新设计。
  目标始终是缩小 gap、加快收敛，不用调参掩盖无收益。

当前分支说明：

- `main` / `report-2026-10-23` = `ada18c3` 代码基线；
- `dg-cser-replay` = DG-CSER 跨阶段回放实验（`32f47c5`，已推送）；period=5 同保存
  时长仅 13.08% vs 13.63%，**不是基线，不要从它继续创新**；
- `innov-mcl` = 单调割提升实验（`670daef`，已推送）；根 LB 仅 +0.0039，验收不通过，
  **不要从它继续创新**；
- `algorithm-innovation` 已删除。

## 命令与配置

```
风场多仓库备件库存管理案例-python/
├── config/
│   ├── problem.toml          # ★ 问题参数（唯一事实源）
│   ├── solver.toml           # ★ 求解参数（唯一事实源）
│   └── smoke/                # 冒烟/自检用的极小实例，格式相同
├── run.py                    # 唯一训练入口
├── simulate.py               # ★ 唯一仿真入口：全局统计 + 详细数据落 npz + txt 报告
├── analyze_sim.py            # ★ 读 npz 做事后分析（成本结构/调拨/订货/轨迹），不重仿真
├── AnalysisModel_IIS.py      # LP 冲突集分析（与 Julia 版同一份）
├── code/
│   ├── config.py             # 读 TOML + 校验 → ProblemConfig / SolverConfig
│   ├── problem_data.py       # ProblemData + DefineProblem(cfg) + 威布尔/PHM + (z,θ) 池
│   ├── state_layout.py       # 状态布局 I | o | n | q
│   ├── subproblem.py         # ★ coptpy 子问题：base 构建 / 割拼装 / 改点求解 / 对偶
│   └── sim_analysis.py       # 仿真口径：状态重建 / 五项成本 / 约束审计（两脚本共用）
├── utils/                    # SDDP 工具包，与问题解耦（不 import code/）
│   ├── copt_env.py           # coptpy 环境、参数、ASCII 落盘兜底
│   ├── payload.py            # 割集跨进程传递（shared_memory + pickle 回退）
│   ├── worker.py             # spawn worker 装配、2 个长驻模型、三类任务函数
│   ├── engine.py             # SDDPEngine：前向并行 / 后向阶段串行 / 割池 / 训练循环
│   ├── cut.py io.py simulation.py diagnostics.py
├── test/                     # 自检脚本（_t_*.py），不进主链路
├── plot/plot_bounds.py       # 收敛图（表头解析，与 Julia 版同一份）
└── results/<tag>_<时间戳>/    # 每次训练一个自包含目录
```

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"

$P run.py                                   # 用 config/ 的配置
$P run.py --config-dir config/smoke         # 换一套配置（冒烟）
$P run.py 300                               # 覆盖轮数（其余走配置）
$P run.py 300 2000                          # 覆盖轮数 + 仿真路径数
$P run.py 200 resume results/<目录>          # 续训

$P simulate.py results/<目录> [路径数] [块大小]   # 仿真一次：统计 + 落 npz + txt 报告
$P analyze_sim.py results/<目录>                # 读 npz 分析，秒级，不重仿真
$P plot/plot_bounds.py
```

**参数纪律（用户明确要求，别再犯）**：
- 默认值**只写在 `config/*.toml`**，代码里没有第二处。缺键直接报错，不静默兜底。
  这条是针对历史 bug 的回归：以前默认值在 `run.py` 的 `main()` 签名里，命令行位置
  参数会盖掉它，出现过"改了 120 却不生效"。
- 启动时打印**生效配置全文**，并把 TOML **原样拷贝**进结果目录
  （`results/<目录>/config/`，注释与结构全保留）。读回用同一套 loader。
- 两条派生规则：`n_processes = 0` → **物理核数 // 2**；
  `n_forward_paths = forward_paths_per_process × n_processes`，倍率默认 1
  ⇒ 每阶段试探点数恰好等于进程数 ⇒ **每个 worker 每阶段正好产出 1 条割**。
- `simulate_only.py` / `simulate_inspect.py` 从**结果目录里那份配置副本**恢复口径。

**割筛选**（`config/*.toml` 的 `[training]`，实现见 `utils/engine.py::_lml1_update`）：
按 `docs/LML1割筛选算法逻辑.md` 对每个历史试探点只保留「在该点取值最大的、且生成最早的」
那条割；**没被任何点选中的割直接删除**（文档原版是"全存但只用 S"，这里按用户要求真删）。
`cut_selection = "none"` 可关。实测冒烟 12 轮：割数 100 → **36**，中途还出现过下降。
`cut_select_keep_rounds = 0` 表示试探点全保留（文档原版）；设成正数则只留最近若干轮的点，
此时 `|S|` 才有硬上界 `keep_rounds × n_forward_paths`/阶段 —— **注意试探点本身也要占内存**
（每条 615 维），光删割不减半。

## 当前实现

**阶段结构**：阶段 0 为虚拟首阶段（钉初值 + 决定首期订货），1..T 为真实周期。
`cuts[t]` 近似 V_t，第 t−1 期子问题用 `cuts[t]` 约束未来成本变量 theta。

**状态**（615 维，N_R=5/A=120/L=2）：`I(N_R) | o((L−1)·N_R) | n(N_R·A) | q(N_R)`。
状态变量**只用变量界锁死在试验点**（路线 A），读 `.dj` 即得 ∂Q/∂x。
**615 个状态全部声明 `vtype=INTEGER`**（台数/件数物理上就是整数），`set_state`
据此拒绝小数试验点（`require_int=False` 仅供诊断扫 LP 松弛的小数点）。前向路径
因此全程停在整数格点（实测离整距离恒 0）。

**★ 失效量 F 必须是「整数变量 + 线性式定义」，不能是算出来的数字**（负 gap 根因）：

```
Fdef_hi: F[r,a] − θ[r,a]·n_in[r,a] ≤ 0        F ∈ ℤ  ⇒  F = ⌊θ·n⌋
Fdef_lo: F[r,a] − θ[r,a]·n_in[r,a] ≥ −1
dem_r  : Σ_m y[r,m] + Σ_a n_in[r,a] − Σ_a F[r,a] ≤ N_r
age_r,a: n_out[r,a+1] − n_in[r,a] + F[r,a] = 0
```
右端只有纯结构常数 `0` / `−1`，**没有一个是「由状态算出来的数」**，θ 以**系数**进
矩阵 ⇒ 对偶看得见 `∂F/∂x`。曾经写成 `F = rint(θ·n(x))` 塞右端，割的斜率少了
`Σ θ·∂Q/∂F`（实测相对 `.dj` 达 5%）⇒ 割在锚点外高于真值 ⇒ LB 虚高 ⇒ 负 gap。
**这与取整无关**（连续 `θ·n` 塞右端同样越界）；也**光把 F 建成变量、用变量界钉住
不行**（`.dj` 逐位不变，等于把数字换地方写）。修法与证据见 `docs/割越界根因分析.md`。
LP 松弛下 `F ∈ [⌊θ·n⌋, ⌊θ·n⌋+1]`（|F − θn| ≤ 1），与连续口径同量级。

**子问题（code/subproblem.py）base 与拼装分离**：
- `__init__` 只建 base——约束 (1)..(8) + **阶段成本目标**，不含任何割；
- `assemble(cuts_next, key)` 把未来成本 theta 与割行挂上去；`key` 不变则跳过；
- `set_state(x)` 改 615 个 xin 变量界（试验点须为整数）；`set_scene(scene)` 把 θ
  写进 `Fdef_hi/lo_{r,a}` 两族行的**系数**（`setCoeff(x_in[n], −θ)`，见上节）；
- 阶段 1..T 的 base **结构完全相同**，只有一个模型就能服务所有 t ≥ 1。

**引擎（utils/engine.py）**：
- **前向**：一任务 = 一条路径，t=0..T 逐期解 MIP；收齐 `trial[t]` 与总成本（上界）。
- **后向**：**按阶段串行（t = T..1），期内按试探点并行**。每阶段把该阶段 N 个试探点
  分给 worker，**每 worker 产一条割**（一任务 = 一个试探点 = 一条割：逐场景解 LP 取
  `.dj`，场景等概率平均）；本阶段割**全算完再一起入池**，然后才推进到 t−1。
- **期间串行是关键修复**：旧实现轮初打割快照、全部 (期 × 试探点) 一次铺满，期 t 用不上
  同轮期 t+1 的新割 → 终端信息每轮最多传播一期 → T=200 时 LB 上百轮不动（实测 13 轮全 0）。
  现在期 t 立即用上同轮期 t+1 的新割，一次反向扫描传遍全视界（实测 T=8 第 2 轮 LB 即 > 0）。

**模型缓存（utils/worker.py）**：每个 worker 只持 **2 个** coptpy 模型——
`stage0`（虚拟首阶段，常驻）+ `shared`（服务所有 t=1..T，键 `(round_id, t, pass_tag)`，
键变了才换割行）。换割集 = 先 `remove` 旧割行再 `addConstr` 新的，实测换 2000 行
约 0.23 s。内存从最坏 (T+1) 个模型降到 2 个。

**割载荷（utils/payload.py）**：一阶段一块 `shared_memory`，主管程每阶段写一次、map
结束 `unlink`；任务只带块名。按任务 pickle 的话每轮 IPC 量是割表本身的进程数倍。

**割**：`Cut(alpha, beta, state, stage, obj, it)`，截距式 `V(x) ≥ alpha + beta @ x`
（`alpha = obj − beta @ state`）。**LML1 筛选**（`cut_selection="lml1"`）：每个历史
试探点只留「在该点取值最大、且生成最早」的那条，其余删除。存 `cuts.json`（原子写，
带 `FORMAT_VERSION`；**`load_cuts` 会校验版本**，模型结构一改旧目录直接被拦，落实铁律 4）。

**两种求解模式**：前向 `solve()`（MILP，读 `BestObj`）取整数策略；后向 `solveLP()`
（同一模型的 LP 松弛，读 `LpObjVal`）取对偶。读目标值必须**按类型分开**，
**不要用 `objval`**（对 LP 值有粘性）。
**割的有效性依据**：松弛值 ≤ 整数最优 ⇒ 松弛对偶割是整数问题的合法下界（标准
SDDP-for-MILP 框架）。前提是 `Q^LP(·;ω)` 对状态凸 —— 而这要求**状态对约束的所有
依赖都走系数矩阵**，见上面 F 那一节。

## 仿真工具（生产与消费分离）

**仿真只跑一次、落盘；之后所有统计都读盘。** 别再为每个新需求往 `test/` 写一次性脚本重跑
仿真 —— 500 条要 ~8 分钟，是纯浪费。

```
simulate.py <结果目录> [路径数] [块大小]   # ① 跑仿真 → 统计 + 落 npz + txt 报告
analyze_sim.py <结果目录>                  # ② 读 npz 分析，秒级，可反复跑
```

- `simulate.py` 出：全局统计（n / UB / std / SE / 95% CI / LB / gap / 求解次数 / 耗时）、
  `sim_<P>paths.npz`（每期每变量的完整数值）+ `.json`（元数据）、`sim_report_<时间戳>.txt`
  （含**成本结构**与**逐 (路径,期) 点的约束审计**）。同名 npz 已存在则拒绝覆盖。
- `analyze_sim.py` 出五节：成本结构 / 主动调拨 τ / 被动调拨 y / 订货 q / 逐期轨迹。
- **口径只有一处**：状态重建、五项成本拆解、约束审计都在 `code/sim_analysis.py`，
  两个脚本共用，不会两边不一致。要加新统计就加到 `analyze_sim.py`，别另起脚本。
- 历史脚本 `simulate_only.py` / `simulate_inspect.py` 已被取代（保留未删）。

## 铁律（用户定下的规矩，违背会被打回）

1. **不改 results/ 里的任何东西**：不删除、不覆盖、不挪动。训练/仿真只往里面
   **新增**目录或新文件。
2. **不要写乱七八糟的注释**：注释少而准，解释"为什么"（尤其非显然的），不写废话。
3. **达梦数据库服务不碰**（dmsvc.exe / DmServiceDMSERVER）：它占着约 10 GB commit，
   杀掉会引发连锁崩溃，与项目无关但在这台机器上。
4. 模型结构一旦改过（约束/状态/目标），**旧结果目录的割一律不兼容**，必须全新训练，
   不要 resume 旧目录；`simulate_only.py` 对旧目录同样失效。resume 时
   `utils/io.problem_diff` 会比对两边的问题配置，不一致直接拦下。
5. Windows + 中文路径：Git Bash 里把中文路径放进变量会被编码搞坏，要么整条命令用
   引号包住绝对路径，要么用专用文件工具。
6. **临时/自检脚本一律放 `test/`**，不要写在项目根目录。
7. **建进程池的脚本必须有 `if __name__ == "__main__":` 守卫**（Windows spawn 会重新
   import 主模块，没有守卫会无限递归建池）。

## 与 Julia 版的已知差异

| 项 | Julia 版 | Python 版 |
|----|---------|-----------|
| 建模 | JuMP | coptpy（原地改参数 + 自动热启动）|
| 并行 | `SDDP.Threaded()`（`-t auto`）| `multiprocessing`（spawn，默认物理核//2）|
| 参数 | 脚本内 | `config/*.toml`，结果目录存副本 |
| 割存储 | JLD2/json | `cuts.json`（Cut 完整字段）|
| 后向 | 全阶段 | **期间串行 + 期内并行（试探点）** |
| 可复现 | — | 同进程数不保证逐位复现，见下 |

## 求解器（coptpy）要点 —— 完整清单见根目录 `COPT使用手册.md`

- **原地改参数可行**（旧结论"coptpy 不可变"是错的）：`setInfo(Info.LB/UB, 对象, 值)`
  同时覆盖列与行；`setCoeff(con, var, v)` 改矩阵系数；`addConstr` 追加割。
  实测 600 列实例原地改界+重解 0.22 ms vs 重建 3.93 ms（17.7×）。
- **热启动是自动的**：改完界直接 `solveLP()` 复用上次最优基——400 变量链式 LP 首解
  199 次单纯形迭代，改界后连续重解 **0 次迭代**。**不要手动 setBasis**。
- **`Threads` 默认 -1 = 自动用满核**，多进程下必须设 1（实测单线程反而快一倍：
  0.020 s vs 0.040 s）。并行度留给进程级。
- **MIP/LP 混合时必须按类型读目标值**：`objval` 对 LP 值有粘性（跑过 `solveLP()` 后
  永远返回 LP 值，即使 `solve()` 已给出整数解，且 `Status` 照样报 OPTIMAL 判不出来）。
- **原生 DLL 路线不必走**：ctypes 直调 `copt.dll` 能做到的 coptpy 全都能做，原地改的
  耗时几乎相同（0.23 vs 0.22 ms），却多一层手写原型与 348 个函数的手工接线。
  对照脚本 `test/_t_mut_dll.py` 保留作证据。

## ★ 已知边界：数值路径不逐位复现

同一 seed 两次跑，**结论统计等价但不逐位一致**（实测下界相对差 ~7e-3 量级）。
三个叠加因素：

1. **LP 最优对偶可能不唯一**（对偶退化）——目标值唯一，但 `.dj` 可取到多个最优对偶解；
2. 共享常驻模型 + 自动热启动 ⇒ 取到哪个最优对偶取决于该 worker 的**模型使用史**；
3. `multiprocessing.Pool.map` 不保证任务→worker 的固定映射。

实测：**第 1 轮的割跨进程数完全一致**，第 2 轮起从退化点开始分岔并向下游传播。

**不影响正确性**：每个最优对偶都是合法次梯度，割总是有效下界，收敛性不变。
**但论文若要逐位可复现**，需要每次求解前清基（丢掉热启动，正是这次改造要买的东西），
或者在单进程下跑。`test/_t_det.py` 把这条边界测准了。

## 已踩过的坑（照做即可，完整清单见 docs/工具包说明.md §踩坑）

- **Julia→Python `enumerate` 差一**：`tc_route=(pos−1)·tc` 必须 `start=1`，否则本地
  调拨成本变成 −tc，负成本会把 LP 解带跑偏（已修）。
- **容量约束 (2) 钉出侧**（`I.out + Σ_l o[l].out + q.out ≤ Imax`），不钉期初侧 ——
  期初口径先后被 q.in 滞后、主动调拨 τ 流入顶破，RCR 自动成立靠的是出侧。
- **`n[r,A].out` 切勿钉 0**：到寿离场已隐式含在需求约束里（系数 0），再钉 0 与
  trans 冲突（IIS 验证过）。
- **池矩匹配阈值 6σ 而非 4σ**：偏斜 Beta + 小池 + 多重检验下 z 到 3~4 正常。
- **COPT 文件接口不认中文路径**：`writeLp/writeMps/writeBin/writeSol/writeIIS/setLogFile`
  遇非 ASCII 路径失败，短路径还会**静默不建文件**。本项目路径全中文 → 落盘先写
  `tempfile.gettempdir()` 再搬回（`utils/copt_env.dump_lp`）。
- **求解失败时 `objval` 是陈旧值**：不可行/失败后 `objval` 与 `LpObjVal` 都静默返回
  上一次的值，只有 `m.status` 可靠。每次求解后必须查 `status != OPTIMAL` 就中止。
- **`m.getRow(con)` 是副本语义**：有 `setConstant/setCoeff` 但改完重解不生效。改行右端
  一律走 `setInfo`。
- **`remove()` 不会让其它行句柄失效**（实测删 A 后改 B、C 的右端仍精确生效），
  所以 base 里的静态行句柄可以一直复用。
- **空割集必须等同"无割"**：`assemble` 收到空 (alpha, beta) 要把 theta 钉在 0，
  否则 theta 进目标却无约束约束它 ⇒ **LP 无界**。
- **`setInfo` 的容器分派**：`tupledict + list` 会 `TypeError`（`addVars` 返回的正是
  tupledict）。要么 `list(d.values())` 要么传标量。
- **COPT 容差默认**：`FeasTol/DualTol/IntTol = 1e-6`、`MatrixTol = 1e-10`、
  `RelGap = 1e-4`/`AbsGap = 1e-6`。实测 FeasTol 边界：冲突量 ≤1e-6 判可行、
  >1e-6 判不可行；`(1−θ)~1e-5` 的系数远在 MatrixTol 安全区内。
- **★ resume 不恢复 LML1 历史会清空旧割**（2026-10-24）：`cuts.json` 只存 Cut，
  不存 `_pts/_mx/_rc`；直接 resume 后第一轮 `_lml1_update` 会把所有 `it>0` 的旧割
  当成无主割删掉。现在 `run.py` resume 调用 `engine.restore_cut_history()`，从每条
  割的 `state` 和 `alpha,beta` 重建历史。**LML1 的数学逻辑不变**：加载的旧割和
  新割一样参与筛选（强割支配时旧割仍会被删除），只是不会在第一轮被整体清空。
  验收 `test/_t_resume.py`。如果以后从别的地方加载 cuts 继续训练，也要先调用这个方法。
- **★ fleet/dem 的 LP 数值残差**（2026-10-24）：上一期 `n_out` 只满足到 FeasTol，
  下一期钉死后 `fleet`/`dem` 行会以 1e-6 量级冲突（IIS 实测 3.9e-7，后续又见
  1.114e-6）。修法分两层：
  1. `fleet`/`dem` 右端统一为 `N_r + fleet_tol`（默认 1e-6）；
  2. 状态流转时 n 块按 `state_round_digits=6` 位 round，若仍超 `N_r` 且超额
     ≤`state_repair_tol=1e-4`，从最大分量中扣除超额（只动一个分量，不缩放）；
     超过 `state_repair_tol` 才抛错，视为真不可行。
  **不要用按比例缩放掩盖**——它无法区分真不可行和数值残差。
  `FORMAT_VERSION=5`，旧割不可用。
- **★ `trial[t]` 必须是第 t 期的【入库】状态，别记成期末**（2026-10-10）：老代码（含重构前
  的 `training.py`）在 `x = out_state_vector()` **之后**才 `states.append(x)`，于是 `states[k]`
  是第 k+1 期的输出、`trial[t]` 成了第 t+1 期的输入。割会被锚在"下一年"的状态上，而阶段 0
  产出的是初始机队 → 线性外推成 −1e7 量级的巨额负数 → `theta ≥ 0` 恒胜 → **LB 永远为 0**。
  只在 `spread_init = false`（全机队月龄 0 这个极端点）时致命；`spread_init = true`
  （冒烟配置）月龄摊平，挪一个月几乎无差别，所以测试全绿而生产全黑。**教训：测试配置的
  数据分布要和生产一致，否则测不出对分布敏感的 bug。**
- **Windows 控制台 GBK**：入口脚本统一 `sys.stdout.reconfigure(errors="replace")`。
- **★ 别用 `os.dup2` 压 COPT 横幅（会让 worker 假死）**：曾经在 spawn worker 里用
  `os.dup2(os.open(os.devnull, O_WRONLY), 1)` 吞掉 COPT 的许可横幅。**当 stdout 是控制台时，
  这个重定向会让 worker 卡死**（后续 `cp.Envr()` 永久阻塞）→ worker 拿不到任务 → 主进程在
  `pool.map` 上无限等，表现为 **0% CPU 的假死**，日志停在表头。二分实测：worker 只
  `import coptpy` = 正常；`import` + `cp.Envr()` = 正常；**先 dup2 再 `cp.Envr()` = 挂死**。
  而 stdout 指向文件/管道时同样的重定向无害 —— 所以**只在终端里直接跑才会踩到**，重定向到
  文件跑则完全看不出来（这个不对称坑了很久）。已移除，代价是 COPT 会多打几行横幅。
- **数值路径不逐位复现**（见「已知边界」）：同一 seed 两次跑结果统计等价但不逐位一致，
  第 1 轮的割跨进程数完全一致、第 2 轮起从对偶退化点分岔。
- **进程假死时的排查手法**：`SDDP_TRACE=1` 看每轮各阶段边界；`SDDP_STACK_AFTER=<秒>`
  让主进程到点打印全部线程栈并退出（`faulthandler`）。
- **`resetParam()` 不带参数**，一次复位全部参数；`setParam` 返回 `None`，不能拿返回值判成败。

## 自检脚本（`test/`）

| 脚本 | 内容 |
|------|------|
| `_t_equiv.py` | **建模等价性（硬门槛）**：coptpy 重写 vs pulp 旧实现，obj/dj/π 逐位一致（35 项）|
| `_t_assembly.py` | 拼装契约：base 只含阶段成本、挂割/卸割、换割集不破坏句柄、空割集（15 项）|
| `_t_config.py` | 配置生效性：改 TOML 必生效、缺键报错、派生值换算、非法值拦截（28 项）|
| `_t_sweep.py` | 后向串行扫描验收：T=8 六轮，**断言 LB 在 3 轮内 > 0** |
| `_t_det.py` | 可复现性边界：第 1 轮割跨进程一致、LB 统计等价、次梯度合法 |
| `_t_engine.py` | 引擎端到端：割自洽 + 试探点 + 变量记录 + 仿真 |
| `_t_resume.py` | resume：从 cuts 重建 LML1 历史；弱割不清空旧割，强割仍按 LML1 删除 |
| `_t_smoke.py` | 对偶自检四项（.dj vs 通用式、次梯度不等式、割合法性）|
| `_t_iis.py` | 不可行子问题 → 失败兜底导出 LP → 交 AnalysisModel_IIS.py |
| `_t_solver.py` | coptpy 可用性（LP/MIP/对偶/参数/中文路径落盘）|
| `_t_copt_params.py` | 容差默认值与边界、MatrixTol、Threads 实测 |
| `_t_mut_coptpy.py` / `_t_mut_dll.py` | 原地改参数 + 对偶差商；原生 DLL 对照 |
| `_t_copt_survey.py` | coptpy 全能力巡检 + 计时对照（47 项）|

`test/_pulp_ref_model.py` 是**历史对照件，不是生产路径**，只被 `_t_equiv.py` /
`_t_copt_survey.py` 引用。
