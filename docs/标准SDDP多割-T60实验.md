# 标准 SDDP 全场景多割（V2）实现与 T60 pilot 结果

## 1. 范围

- 分支：`innov-multicut`（从 `exp-param-tune` / `4ec38f9` 新建，V2 参数）。
- 只实现标准 SDDP；不实现 adaptive 分区多割，不修改 `innov-adaptive-partition`。
- 单/多割切换：`SolverConfig.mode = "single" | "multi"`；没有 K、n_groups 等参数。
- `mode="multi"`：K 固定 `len(prob.pool)=300`；每个场景一条条件割、一个 theta；阶段 t−1 目标 `Σ_k p_k θ_k`，`p_k=1/300`。
- `mode="single"`：完全保持原单割行为。

## 2. 实现

- `Cut.group`：内部场景/theta 索引，single=0，multi=场景索引。
- `FORMAT_VERSION=6`；single 只读兼容 v5（group=0），multi 拒绝 v5。
- `StageSubproblem`：single 1 theta；multi 300 theta，权重 1/300；`assemble` 按 group 绑定；`theta_value = Σ p_k θ_k`，`stage_cost = obj − theta_value`。
- `utils/worker.py::backward_multicut_task`：每个试探点对 300 个场景逐个解 LP，返回 300 条 `(beta,obj)`，不平均。
- `utils/engine.py`：multi 每个试探点把 300 条割全部加入 `cuts[t]`；`n_solves` 按实际 300 次/点计；LML1 按 `(阶段, theta)` 独立维护，不跨 theta 删除。
- `utils/payload.py`：增加 group 数组；`utils/io.py`：v6/group；`run.py`/`simulate.py`：mode、K/meta 校验。
- 测试：
  - `test/_t_multicut_unit.py`（group 绑定/加权 theta）
  - `test/_t_multicut_engine.py`（smoke single/multi、K=30、LML1、resume）
  - `test/_t_multicut_cutvalid.py`（每个 group 只用该场景验证割）
  - `test/_t_multicut_resume.py`（multi resume、v5 single 兼容、v5 multi 拒绝）
  - 原有 `_t_config/_t_assembly/_t_smoke/_t_sweep/_t_engine/_t_grade_cutvalid/_t_resume` 在 `mode=single` 下全部通过。

## 3. T60 pilot

目录：`results/mcT60_pilot_20261103_113710/`

配置：V2 问题；`mode=multi`；`cut_selection=lml1`；`cut_select_keep_rounds=0`（只用 LML1，无额外封顶）；`max_train_seconds=600`；`n_simulations=200`；10 进程。

实际结果：

| 指标 | 多割 pilot |
|---|---:|
| 完成轮次 | **2**（超过 600 s 后当轮结束停止） |
| 训练墙钟 | 1,070.9 s（含 sim） |
| 每轮耗时 | 358 s / 470 s |
| LB | **4,130.8516** |
| UB（200 路径，独立 simulate.py） | **58,627.4599 ± 2,512** |
| gap | **92.95%** |
| 总割数 | **225,320** |
| cuts.json | **257 MB** |
| 累计训练 LP | 361,222 |
| 每轮新增割（理论） | 10 × 60 × 300 = 180,000 |
| 成本结构 | 采购 39.05%、持有 1.93%、停机 55.80%、主动 0.98%、被动 2.24% |
| 约束审计 | 全部通过；五项和 vs stage_cost 最大偏差 8.8e-09 |

同一 2 轮对比：单割基线第 2 轮 LB 1,959、每轮约 4.5–9.2 s；多割第 2 轮 LB 4,131，但每轮 358–470 s，慢约 40–50 倍。  
同墙钟对比：单割基线在约 600 s 时 LB 29,152；多割 pilot 在约 827 s 只有 LB 4,131。

## 4. 结论

1. **实现正确，但全 300 场景多割在 T60 当前预算下不可行**：
   - 每轮 18 万条割对象；LML1 后 2 轮已保留 22.5 万条割、cuts.json 257 MB；
   - 每轮 358–470 s（单割约 5–10 s）；
   - 600 s 只能完成 2 轮，LB 远低于单割/adaptive 同墙钟结果。
2. **不是 LML1 不正确**：LML1 已按 `(阶段, theta)` 独立工作；问题是 300 theta × 3000 条/阶段割行使每个场景 LP 变重，且割量本身增长快。
3. 按计划 gate（pilot 单轮 >150 s、600 s 内 <5 轮），**不启动 7,503 s 正式全 300 跑**；不把 pilot 的 92.95% gap 当作最终算法结果，只作为可行性证据。
4. 仍然保留 `mode="single"` 的标准 SDDP：单割回归全部通过；`simulate.py` 对 v5 单割只读兼容。

## 5. 后续建议

- 若要在 T60 上获得可比较的多割结果，建议缩小多割的作用范围（例如阶段 1-only K=300 根层多割），或引入场景簇 K（但本次用户明确不要 K 参数，未实现）。
- 或者优化 LP/割管理（例如共享束/多割专用 cut pooling），这属于新的算法方向，不在本分支。
- 本分支保留代码、测试和 pilot 证据；不把全 300 多割作为论文主实验结论。

## 6. 关键路径

- 代码：`code/subproblem.py`、`utils/worker.py`、`utils/engine.py`、`utils/cut.py`、`utils/payload.py`、`utils/io.py`、`run.py`、`simulate.py`
- 配置：`config/smoke_multi/`、`config/tune/multicut_t60_pilot/`、`config/tune/multicut_t60_full/`、`config/tune/multicut_t60_control/`
- pilot：`results/mcT60_pilot_20261103_113710/`（cuts 257 MB；sim_report/summary）
