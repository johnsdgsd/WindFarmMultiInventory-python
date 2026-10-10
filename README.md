# 风场多仓库备件库存管理案例 — Python 版（pulp + 多进程 SDDP）

Julia 原版（`风场多仓库备件库存管理案例`）的 Python 重写。模型、时序、结果目录口径
与 Julia 版逐项对齐；SDDP 算法本体（对偶提取、割的表示与存储、多进程调度）自己实现，
放在可整目录迁移的 `utils/` 包里。

## 目录

```
├── run.py                  # 唯一训练入口（位置参数、smoke、resume）
├── simulate_only.py        # 只仿真：从结果目录恢复问题与割，重跑蒙特卡洛
├── simulate_inspect.py     # 仿真 + 变量记录（CSV 长表，供策略分析）
├── AnalysisModel_IIS.py    # LP 不可行冲突集分析（COPT/Gurobi），与 Julia 版同一份
├── code/
│   ├── problem_data.py     # ProblemData / DefineProblem / 威布尔 / PHM / (z,θ) 场景池
│   ├── state_layout.py     # 状态向量布局：I | o | n | q
│   └── model.py            # StageSubproblem：单场景子问题（约束、目标、对偶）
├── utils/                  # 与问题解耦的工具包，可整目录搬到其他 SDDP 项目
│   ├── cut.py              # Cut 类（alpha/beta/state/stage/obj/it）+ JSON 序列化
│   ├── solver.py           # make_solver（COPT_DLL 优先 / CBC 兜底）、export_lp
│   ├── training.py         # SDDPEngine：前向/后向/下界/训练主循环，多进程
│   ├── simulation.py       # 蒙特卡洛：分块、流式统计、95% CI
│   ├── diagnostics.py      # 对偶自检：.dj vs π 通用式、次梯度不等式、割合法性
│   └── io.py               # 结果目录 / config / 割存取 / 训练日志（原子写）
├── plot/plot_bounds.py     # 训练收敛图（表头解析，与 Julia 版同一份）
├── test/                   # 自检与临场脚本（见文末「自检脚本」）
├── docs/工具包说明.md       # utils API 与踩坑
└── results/<tag>_<时间戳>/  # 每次训练一个自包含目录
```

## 命令

```bash
conda activate MyRL
```

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python"
python run.py                                    # 默认参数训练（默认值多为占位值）
python run.py smoke                              # 冒烟：小实例快跑，验证链路连通
python run.py 500                                # 训练 500 轮
python run.py 500 300                            # 训练 500 轮 + 仿真 300 路径
python run.py 200 resume results/base_2026xxxx_xxxxxx   # 续训 200 轮

python simulate_only.py results/<目录> [路径数] [块大小]
python simulate_inspect.py results/<目录> [路径数] [输出CSV]
python plot/plot_bounds.py                       # 默认画 results/ 下最新的一个目录
python AnalysisModel_IIS.py <现场.lp> --solver copt
```

用的是 MyRL 环境的解释器：

```bash
"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" run.py smoke
```

## 训练参数（`run.py` 的 `main` 签名，只有位置参数）

| 参数 | 默认 | 说明 |
|------|------|------|
| `iterations` | 200 | 训练轮数（位置参数 1）|
| `n_simulations` | 10000 | 仿真路径数（位置参数 2）|
| `T / A / N_R / L` | 24 / 120 / 5 / 2 | 周期数 / 最大月龄 / 仓库数 / 订货提前期 |
| `pool_size / seed` | 300 / 2026 | 场景池大小 / 随机种子 |
| `m / eta` | 2.5 / 120.0 | 威布尔形状 / 尺度（占位值，待定标）|
| `n_forward_paths` | 1 | 每轮前向路径条数（每条贡献一组试态点）|
| `n_backward_samples` | None | 后向场景抽样数；None = 枚举全部场景 |
| `chunk_size` | 1000 | 仿真分块大小（块种子 = seed + k）|
| `n_processes` | None | 进程数；None = 核数，1 = 串行（便于调试）|
| `checkpoint_interval` | 600.0 | 定时存割的间隔（秒）|

冒烟档（`run.py smoke`）会覆盖成：T=4、A=24、pool=30、iterations=10、
n_simulations=50、m=3.0、η=8.0、月龄摊平、n_forward_paths=2。

## 结果目录（`results/<tag>_<时间戳>/`）

| 文件 | 内容 |
|------|------|
| `config.txt` | key=value，结果目录自描述（含全部参数与 N/Qmax/Imax）|
| `problem_data.json` | 问题数据 + 预采样场景池，`simulate_only.py` 据此重建问题 |
| `cuts.json` | 全部割：`{meta, version, cuts: {期: [{alpha,beta,state,stage,obj,it}]}}` |
| `sddp_training_log.csv` | 训练日志，表头与 Julia 版一致，`plot_bounds.py` 直接解析 |
| `summary_<时间戳>.txt` | LB / UB / gap / std / 逐期均值报表 |
| `bounds.png` | 收敛图（`plot/plot_bounds.py` 画）|
| `inspect_<n>paths.csv` | 变量记录（`simulate_inspect.py` 写）|
| `subproblem_<环节>_t<期>_s<场景>.lp` | 仅在求解失败时导出的现场 LP（环节 = forward / backward / lower_bound，供 IIS 分析）|

**只新增、不覆盖**：训练/仿真只往结果目录里写新文件，或先写 `.tmp` 再原子改名。
模型结构（约束/状态/目标）一旦改动，旧目录的割一律不兼容，必须全新训练，不要 resume。

## 与 Julia 版的口径差异（有意为之）

- **求解模式**：前向/仿真用整数 MIP 得策略；后向与下界用**同一模型**加
  `mip=False` 重解、读 LP 松弛的对偶（`guide §6.2` 路线 A，零侵入）。
  与 Julia 版 `handler=:continuous` 同源：割是**松弛值函数**的下界，对整数问题
  不保证严格有效，gap 行为作为对照观察。
- **不做割筛选**：backward 产生的每条割无条件入库（去重/支配/level 剪枝推迟）。
- **后向按批次**：同一轮里期与期互相独立（cuts[t+1] 在本轮冻结），可一次铺满进程池。

## 自检脚本（`test/`）

临时脚本一律放这里，**不要写进项目根目录**。每个脚本自带 `sys.path` 引导，可直接
从根目录跑；除 `_t_conv.py` 外都是小实例、秒级完成。

| 脚本 | 验什么 |
|------|--------|
| `test/_t_smoke.py` | 子问题层：单期求解 + 对偶提取四项（`.dj` vs π、次梯度不等式、转移残差、割合法性）|
| `test/_t_engine.py` | 引擎层：训练 4 轮 + 前向记录 + 仿真 + 割自洽 |
| `test/_t_conv.py` | 收敛观察：冒烟参数 12 块 × 20 轮，逐块打印 LB / UB / gap（耗时较长）|
| `test/_t_iis.py` | 造不可行子问题 → 导出现场 LP |
| `test/_t_iis2.py` | 引擎失败兜底 → 导出现场 LP（全链路）|

两个 IIS 脚本产出的 LP 接着交给冲突集分析：

```bash
python test/_t_iis.py
python AnalysisModel_IIS.py test/_iis_probe.lp --solver copt
```
