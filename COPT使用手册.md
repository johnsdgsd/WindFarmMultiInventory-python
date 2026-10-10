# COPT 使用手册（coptpy 8.0）

本项目用它做 SDDP 的 LP/MIP 求解。本手册只写**本项目验证过**的用法与坑，
每条结论都有 `test/` 下的脚本复现：

| 脚本 | 验证内容 |
|------|----------|
| `test/_t_copt_survey.py` | 建模 / 改参数 / 求解 / 取对偶 / 存取 / MIP / 参数 / 计时（47 项断言） |
| `test/_t_copt_params.py` | 容差默认值与边界、MatrixTol、Threads（8 项断言） |
| `test/_t_mut_coptpy.py` | coptpy 原地改参数 + 对偶差商验证（25 项断言） |
| `test/_t_mut_dll.py` | 原生 DLL 同能力对照（21 项断言） |

复现：`"D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/_t_copt_survey.py`

环境：`COPT_HOME=C:\Program Files\copt80`，许可到期 2026-11-06。

---

## 0. 四条硬规则（先读这个）

1. **文件接口不认中文路径。** `writeLp / writeMps / writeBin / writeSol / writeIIS /
   setLogFile` 走窄字符路径，遇到非 ASCII 直接失败——长路径报 `CoptError: Fail to
   write to LP file`，短路径可能**静默不建文件**。本项目路径全是中文，所以落盘
   一律先写 ASCII 临时目录（`tempfile.gettempdir()`）再搬回项目。
2. **先验 `status`，再按求解类型读目标值——不要用 `objval`。** 两重坑：
   ① 求解失败时 `objval` 与 `LpObjVal` **都静默返回上一次的陈旧值**；
   ② 一旦跑过 `solveLP()`，`objval` 此后**永远返回 LP 值**，即使 `solve()` 已经给出
   整数解与整数解向量（`getValues()` 是对的，`objval` 对不上）。
   正确读法：MIP 读 `BestObj`，LP 读 `LpObjVal`；`status` 只判成败，判不出该读哪个
   （详见 §7.2）。
3. **`resetParam()` 不带参数**，一次复位全部参数。
4. **`tupledict` 不能配列表**。`setInfo(..., d, [list])` 会 `TypeError`——`addVars`
   返回的正是 tupledict。要么 `list(d.values())`，要么传标量。

---

## 1. 环境与模型

```python
import coptpy as cp
from coptpy import COPT

env = cp.Envr()                    # 一个进程一个 Envr 即可；可建多个模型
m = env.createModel()
m.setParam(COPT.Param.Logging, 0)  # 关日志；注意 Envr 没有 setParam，只能在 Model 上设
```

模型规模与类型：

```python
m.getAttr(COPT.Attr.Cols)     # 变量数
m.getAttr(COPT.Attr.Rows)     # 约束数
m.getAttr(COPT.Attr.Elems)    # 非零元
m.getAttr(COPT.Attr.IsMIP)    # 0/1
```

## 2. 建模

```python
x = m.addVar(lb=0.0, ub=10.0, obj=4.0, vtype=COPT.CONTINUOUS, name="x")
xs = m.addVars(range(400), lb=0.0, ub=1.0, obj=1.0, nameprefix="v")   # → tupledict

c1 = m.addConstr(x + y + w >= 3, name="c1")          # 比较式自带 sense
c2 = m.addConstr(2 * x + 3 * y <= 8, name="c2")      # >= / <= / == 都行
c3 = m.addBoundConstr(x + y, lb=0.0, ub=COPT.INFINITY, name="c3")   # 双边界约束

m.setObjective(4 * x + y + 100 * w, COPT.MINIMIZE)   # 方向也可在 addVar 里用 obj= 给
```

- 表达式加法：`cp.quicksum(...)` 求和、`expr.prod(dict)` 加权和。
- `CONSTRAINTS` 常数：`COPT.INFINITY = 1e30`（不是 `math.inf`）。
- 方向：`COPT.MINIMIZE` / `COPT.MAXIMIZE`；变量类型：`COPT.CONTINUOUS /
  INTEGER / BINARY / FREE`。
- 取对象：`m.getVarByName("x")`、`m.getConstrByName("c1")`、`m.getVars()`（VarArray，
  可按索引取，也能整体传给 `getInfo`）、`m.getConstrs()`。
- 读回：`m.getRow(c1)` 返回 `LinExpr`；`m.getA()` 返回 `scipy.sparse.csc_matrix`。

**批量装载**（适合从别处算好 CSC 矩阵直接灌）：

```python
m.loadMatrix(c, A, lhs, rhs, lb, ub)
m.setObjSense(COPT.MINIMIZE)
# c: 目标系数 (n,)
# A: scipy csc_matrix (m, n)
# lhs/rhs: 每行的下界/上界，单边用 ±COPT.INFINITY 补
# lb/ub: 变量界
# 代价：灌进来的变量与约束【没有名字】，getVarByName/getConstrByName 都取不到，
#       只能靠 m.getVars()[i] / m.getConstrs()[i] 按位置索引。
```

## 3. 求解与读结果

```python
m.solveLP()          # 解 LP 松弛（模型含整数变量时也解松弛，对偶可用）
m.solve()            # 解 MIP（模型必须含整数变量）

if m.status != COPT.OPTIMAL:      # ← 必做，见第 0 节规则 2
    raise RuntimeError(m.status)

m.objval                          # 目标值
m.getValues()                     # 全部变量值    list
m.getDuals()                      # 全部行对偶    list
m.getRedcosts()                   # 全部既约成本  list
m.getSlacks()                     # 全部行松弛/活跃度 list
```

按对象取（三种形态，同一组信息）：

```python
x.x, x.rc                        # 单变量：值、既约成本
c1.pi, c1.slack                  # 单约束：对偶、松弛
m.getInfo(COPT.Info.Value, [x, y, w])       # 列表 → list
m.getInfo(COPT.Info.Dual, [c1, c2])
m.getInfo(COPT.Info.Value, {1: x, 2: y})    # 字典 → tupledict
m.getInfo(COPT.Info.Value, m.getVars())     # VarArray → list
```

`COPT.Info` 可用成员：`Value / Dual / RedCost / Slack / LB / UB / Obj /
RelaxLB / RelaxUB / RelaxValue / DualFarkas / PrimalRay / SAObjLow / SAObjUp /
SALBLow / SALBUp / SAUBLow / SAUBUp`（后 8 个是灵敏度分析，需
`Param.ReqSensitivity = 1`）。

运行信息：`SimplexIter`、`SolvingTime`、`BestBnd`、`BestGap`、`NodeCnt`、
`HasLpSol`、`HasMipSol`、`MipStatus`、`ObjVal`（= `objval`）、`ObjConst`。

> **`.dj` 与 `getRedcosts()` 取的是同一列。** 本项目状态用变量界锁死（路线 A），
> `xin` 分量的既约成本即 ∂Q/∂x，直接用。

## 4. ★ 原地改参数（本项目加速的核心）

**模型一旦建好，参数可以原地改，改完直接重解，不用重建、不用重新 LoadProb。**

```python
m.setInfo(COPT.Info.LB,  [x], 2.0)          # 变量下界   （单个对象 + 标量）
m.setInfo(COPT.Info.UB,  [y, z], [0.5, 1.0])# 变量上界    （列表 + 列表）
m.setInfo(COPT.Info.LB,  [c1], 4.0)         # 行下界 = ≥ 行 / == 行的右端
m.setInfo(COPT.Info.UB,  [c2], 9.0)         # 行上界 = ≤ 行 / == 行的右端
m.setInfo(COPT.Info.Obj, [x], 50.0)         # 目标系数
m.setCoeff(c2, x, 3.0)                      # 矩阵系数：把 c2 里 x 的系数改成 3.0
```

`Info.LB / UB` 是本接口的**唯一统一原语**——COPT 的 LB/UB 定义本身就同时覆盖
"变量（列）"和"约束（行）"，所以改界只有一个入口。

**行界的内部表示**（决定该改 LB 还是 UB）：

| 建行时的 sense | 内部 LB | 内部 UB |
|---|---|---|
| `>= b` | `b` | `+inf` |
| `<= b` | `-inf` | `b` |
| `== b` | `b` | `b` |

`loadMatrix` 的 `lhs / rhs` 正是按这个直接给内部 LB/UB。

**容器分派陷阱**：`addVars` 返回 `tupledict`，`tupledict + list` 会走成"单值"分支
然后 `TypeError`。

```python
d = m.addVars(range(4), lb=0, ub=1)
m.setInfo(COPT.Info.UB, d, [1.0] * 4)              # ✗ TypeError
m.setInfo(COPT.Info.UB, list(d.values()), [1.0]*4) # ✓
m.setInfo(COPT.Info.UB, d, 1.0)                    # ✓ 标量广播
m.setInfo(COPT.Info.UB, m.getVars(), [1.0]*4)      # ✓ VarArray + list
```

**热启动是自动的。** 原地改完界直接 `solveLP()`，COPT 复用上一次的最优基——
400 变量链式 LP 首次求解 199 次单纯形迭代，改界后连续重解 **0 次迭代**。
不需要手动 `setBasis`。

**改参数会失效的旧写法**：`m.getRow(c)` 返回 `LinExpr`，有 `setConstant/setCoeff`
等方法，但那是**副本语义**，改完重解完全不生效（求解器日志里 RHS 与系数原封不动）。
改行界一律走 `setInfo`。

**计时（600 列 / 400 行）**：原地改界+重解 **0.22 ms** vs 重建模型+解 **3.93 ms**
（**17.7×**）；原生 DLL 同口径 0.23 ms vs 整模重载 1.18 ms。两者原地改的耗时
几乎相同，**没有理由再走 ctypes 直调 DLL**。

## 5. 增删模型要素

割是只增不改的，所以追加约束是常规操作：

```python
th = m.addVar(lb=-COPT.INFINITY, ub=COPT.INFINITY, obj=1.0, name="theta")
m.addConstr(th >= 1.0 + 0.5 * x - 2.0 * y, name="cut_0")   # 追加割
m.solveLP()                                                # 直接重解，无需重建

m.remove([m.getConstrByName("cut_0")])     # 删约束
m.remove([th])                             # 删变量
m.remove(constr_array)                     # 也接受列表 / 数组 / 字典
```

## 6. 存取

```python
m.write("C:/Temp/ascii/probe.lp")      # 按扩展名分发：.lp/.mps/.bin/.sol/.bas/.mst/.par
m.read("C:/Temp/ascii/probe.lp")       # 同扩展名分发

m.writeLp(p)   m.readLp(p)             # 显式版本，底层同一个 C 文件接口
m.writeMps(p)  m.readMps(p)
m.writeBin(p)  m.writeSol(p)  m.writeBasis(p)  m.writeMpsStr()
```

- **路径必须 ASCII**（规则 1）。`writeLp` 只是 `write` 的显式版本，救不了中文路径。
- `.lp / .mps / .bin` 往返后重解，目标值与对偶逐位一致。
- `writeBasis` 写出的 `.bas` 可用 `readBasis` 读回做热启动。
- `m.clone()` 复制模型，与原模型独立。

**IIS（不可行诊断）**：

```python
m.computeIIS()
m.getAttr(COPT.Attr.HasIIS)     # 1
m.getAttr(COPT.Attr.IISRows)    # ⚠ 只数【行】，不含变量界
m.writeIIS(path)                # 文件里才看得到完整冲突（含界）
```

可行模型上 `writeIIS` 会抛 `CoptError: (INVALID) Fail to write to IIS file`——
先 `computeIIS()` 再写。IISRows 只数行，所以"一行 + 一个变量界"的冲突会报
`IISRows = 1`，要看全貌必须读文件内容。

## 7. MIP 与 LP 松弛

```python
a = m.addVar(lb=0, ub=3, obj=1.0, vtype=COPT.INTEGER, name="a")
m.addConstr(...)
m.solve()        # MIP；读 Attr.MipStatus / HasMipSol
m.setMipStart([a, b], [0.0, 2.0])   # 给可行起点
```

### 7.1 三种"解 LP 松弛"的方式

| 方式 | 接口 | 破坏性 | 对偶可读 |
|------|------|--------|----------|
| 就地解松弛 | `m.solveLP()` | **否**，模型类型不动 | ✅ |
| 永久松弛 | `m.setVarType(vars, COPT.CONTINUOUS)` | 是（可 `setVarType(..., INTEGER)` 改回） | ✅ |
| 混合松弛 | `m.setVarType([部分变量], COPT.CONTINUOUS)` | 是 | ✅ |

实测（`min a+b s.t. 2a+2b>=3, a,b∈[0,2]∩Z`，松弛 1.5 / 整数 2.0）：

- `solveLP()` 直接给松弛解 `(0, 1.5)`，且 `IsMIP=1`、`Ints=2` 全不变——**非破坏性**，
  同一模型接着 `solve()` 照样出整数解。
- `setVarType(getVars(), CONTINUOUS)` 后 `IsMIP` 变 0、`Ints` 变 0，`solve()` 也只解
  LP；改回 `INTEGER` 后 `solve()` 恢复整数解，**可逆**。
- `m.getAttr(COPT.Attr.Ints)` 数整数变量个数，`IsMIP` 是否含整数变量。

**整数变量上跑 `solveLP()` 得到的对偶/既约成本是松弛对偶**，这正是本项目
"前向 `mip=True` 取整数策略、后向 `mip=False` 读松弛对偶"的实现基础。

`IntTol` 调大（如 0.9）想强行松弛是行不通的——实测照旧返回整数解，且语义含混，
不要用。

### 7.2 ★ 交替 solve()/solveLP() 会让 `objval` 与解向量脱节

**`m.objval`（= `m.ObjVal` = `Attr.ObjVal`）对 LP 值有粘性**：一旦这个模型上跑过
`solveLP()`，此后无论 `solve()` 多少次，`objval` 永远返回 **LP** 目标值。

```
solveLP() → objval=1.5, getValues()=(0, 1.5)          一致
紧接 solve() → objval=1.5, getValues()=(0, 2.0)        ✗ 脱节！
              BestObj=2.0, Status=1(OPTIMAL), HasMipSol=1
再 solve()   → objval 仍是 1.5（治不好）
```

要点：

- **`Status` 在这四种情形下都是 1（OPTIMAL），无法区分该读哪个目标值。**
- `getValues()` 跟**最近一次**求解走，是对的；`objval` 不跟。
- `BestObj` 是整数目标（`HasMipSol=1` 时有效），`LpObjVal` 是 LP 目标
  （`HasLpSol=1` 时有效）——**这两个才是各自场景下的正确读法**。
- 从没跑过 `solveLP()` 的模型，`objval` 正确（= 整数目标）。**混合使用才会脱节。**

所以读取目标值的正确姿势是按求解类型分别读，不要用 `objval`：

```python
m.solve()                                     # MIP
if m.status != COPT.OPTIMAL: raise ...
obj = m.getAttr(COPT.Attr.BestObj)            # 或 m.BestObj

m.solveLP()                                   # LP
if m.status != COPT.OPTIMAL: raise ...
obj = m.getAttr(COPT.Attr.LpObjVal)           # 或 m.LpObjVal
red = m.getRedcosts()                         # 松弛对偶，∂Q/∂x 出处
```

**同一模型上交替跑 MIP 与 LP 是可以的**，只要按求解类型分开读目标值——**不要为这个
开两个模型**（本项目内存优先：每（期 × worker）一个长驻模型已经占 0.8–1.5 GB/worker）。
这么读就安全：

```python
m.solve()    # 前向 MIP
assert m.status == COPT.OPTIMAL
obj = m.getAttr(COPT.Attr.BestObj)

m.solveLP()  # 后向 LP
assert m.status == COPT.OPTIMAL
obj = m.getAttr(COPT.Attr.LpObjVal)
grad = m.getRedcosts()
```

（COPT 官方文档把 `LpStatus/LpObjVal/BestObj/BestBnd` 标为"不推荐使用"，建议统一用
`Status/HasSol/ObjVal/ObjBound`——但 coptpy 8.0.3 的 `Attr` 枚举里没有
`HasSol/ObjBound`，且实测 `ObjVal` 有上述粘性，故本项目按求解类型分开读。）

## 8. 参数与容差

```python
m.setParam(COPT.Param.Logging, 0)      # 关日志（批量求解必关，否则刷屏）
m.setParam(COPT.Param.Threads, 4)      # 线程数；-1 = 自动
m.getParam(COPT.Param.Threads)         # 读回当前值
m.getParamInfo(COPT.Param.FeasTol)     # (名, 当前值, 默认值, 最小值, 最大值)
m.resetParam()                         # ⚠ 一次复位【全部】参数，没有单项复位
m.setLogFile("C:/Temp/ascii/run.log")  # 路径必须 ASCII
```

全量 76 个参数的默认/范围可跑 `test/_t_copt_params.py` 打印。`-1` 在多个参数里表示
**自动选择**（不是"关"）。

### 8.1 容差（默认值最重要，多数情况不用动）

| 参数 | 默认 | 范围 | 含义 |
|------|------|------|------|
| `FeasTol` | **1e-6** | 1e-9 ~ 1e-4 | 变量/约束取值的**可行性**容差 |
| `DualTol` | **1e-6** | 1e-9 ~ 1e-4 | **对偶解**的可行性容差 |
| `IntTol` | **1e-6** | 1e-9 ~ 1e-1 | 变量的**整数解**容差 |
| `MatrixTol` | **1e-10** | 0 ~ 1e-7 | 输入矩阵的**系数容差**（更小的系数被当 0） |
| `PDLPTol` | 1e-6 | 1e-12 ~ 1e-4 | 一阶算法（PDLP）收敛容差 |
| `NLPTol` | 1e-8 | 1e-13 ~ 1e-3 | 非线性规划最优相对容差 |
| `RelGap` | **1e-4** | 0 ~ inf | 整数规划最优**相对**容差 |
| `AbsGap` | 1e-6 | 0 ~ inf | 整数规划最优**绝对**容差 |

MIP 的双 gap 定义：

```
AbsGap = |primalBound − dualBound|
RelGap = AbsGap / max(|primalBound|, |dualBound|)
```

（最小化问题：`primalBound` = 当前可行解目标值上界，`dualBound` = 目标值下界。）

**实测 FeasTol 边界**（构造 `x ≥ 10` 与上界 `x ≤ 10−δ` 的必然冲突）：

| 冲突 δ | 结果 |
|--------|------|
| 1e-3 / 1e-5 | INFEASIBLE |
| **1e-6**（= FeasTol） | **OPTIMAL，Δ=1.000e-06** |
| 1e-7 | OPTIMAL，Δ=1.000e-07 |
| 1e-9 | OPTIMAL |

⇒ 冲突量 **≤ FeasTol 判可行**，解会把违反量塞进某个界里；`> FeasTol` 判不可行。
**建模别在右端留 1e-6 量级的缝隙**，否则"不可行"会被容差吞掉。

**实测 MatrixTol 边界**（`coef·a + b ≥ 1`，看系数是否还起作用）：

| 系数 | 行为 |
|------|------|
| 1e-3 / 1e-6 / 1e-9 | 系数保留（b 精确降到 1−coef） |
| 1e-11 / 1e-13 | **系数被当 0 丢掉**（b 恒为 1） |

⇒ 默认 1e-10。本项目的 `(1−θ) ~ 1e-5` 量级远在安全区内（Julia 版 HiGHS 的
"small coefficients" 警告在 COPT 这边对应 MatrixTol，量级差 5 个数量级，不会触发）。

### 8.2 ★ 线程：多进程并行时每个 worker 必须设 `Threads=1`

| 参数 | 默认 | 说明 |
|------|------|------|
| `Threads` | **-1（自动，用满核）** | 最大 128 |
| `SimplexThreads` | -1（跟随 `Threads`） | 对偶单纯形法线程数 |
| `MipTasks` | -1（自动） | MIP 求解任务数，最大 256 |

本项目训练走 `multiprocessing`，默认核数个子进程。**COPT 默认会自动用满核**，
12 进程 × 28 线程 = 严重超订，纯互相抢核。

实测（3000 变量链式 LP，单进程）：

| `Threads` | 耗时 | 迭代 |
|-----------|------|------|
| **1** | **0.020 s** | 3058 |
| -1（自动） | 0.040 s | 3134 |

**单线程反而快一倍**——LP 单纯形法本身并行收益有限，线程协调开销是净损失。
多进程场景下更是必须限制：

```python
m.setParam(COPT.Param.Threads, 1)          # 或 SimplexThreads=1
```

把并行度留给**进程级**（12 个 worker 各解一个 LP），不要进程内再并行。

### 8.3 其它常用参数默认值

| 参数 | 默认 | 取值 |
|------|------|------|
| `LpMethod` | -1 | -1 自动（LP 选对偶单纯形；MIP 选对偶单纯形或内点法）；1 对偶单纯形；2 内点法；3 直接 Crossover；4 并发求解；5 按特征自动选；6 PDLP |
| `Presolve` | -1 | -1 自动；0 关；1 少量快速；2 正常；3 多多益善；4 无限制（可能很耗时） |
| `Scaling` | -1 | -1 自动；0 不调整；1 调整系数矩阵数值 |
| `Dualize` | -1 | -1 自动；0 不建对偶模型；1 构建并求解对偶模型 |
| `DualPrice` | -1 | -1 自动；0 Devex；1 对偶最陡边 |
| `DualPerturb` | -1 | -1 自动；0 无摄动；1 允许目标函数摄动 |
| `Crossover` | 1 | -1 自动；0 不用；1 用（要用 `ReqSensitivity` 就得有基解） |
| `TimeLimit` / `SolTimeLimit` | 1e20 | 秒；1e20 相当于不限 |
| `NodeLimit` | -1 | 自动（不限）；MIP 节点数上限 |
| `ReqFarkasRay` | 0 | 不可行/无界时是否算 Farkas 射线或主元射线 |
| `ReqSensitivity` | 0 | 是否算灵敏度分析（`Info.SA*`），需最优基解 |
| `IISMethod` | -1 | -1 自动；0 质量优先；1 效率优先 |
| `MipStartMode` | -1 | 自动 |
| `Logging` | **1** | 0 关 / 1 开 |
| `LogToConsole` | 0 | 0/1 |
| `LogLevel` | 2 | 最大 3 |

> `Param.Dualize` 设为 1 会让 COPT 转去解对偶模型——**取对偶值时别开**，否则
> `Info.Dual` 的语义会绕一圈。

## 9. 计时参考

真实规模子问题（T=4/A=24/N_R=3 冒烟参数）：**180 列 / 100 行 / 324 非零**。

| 做法 | 建模 | 求解 |
|------|------|------|
| coptpy 直接建模 | 1.02 ms | 0.25 ms（首解） |
| pulp 建模 + `COPT_DLL` | 1.00 ms | 2.33 ms（每次含 extract + LoadProb） |
| coptpy 原地改界 + 重解 | — | **0.19 ms/次** |

同规模下"建+解"合计：coptpy **1.27 ms** vs pulp **3.33 ms**。

600 列 / 400 行放大实例：原地改界+重解 0.22 ms vs 重建+解 3.93 ms。

**结论**：直接 coptpy 建模省掉 pulp→COPT 的 `extract()` + `LoadProb()` 全量重载；
再叠加原地改参数，反复求解的场景（SDDP 后向每轮几万次 LP）收益是数量级的。

## 10. 本项目落地建议

- **新建子问题直接 coptpy 建模**，不再走 pulp。pulp 仅保留给旧路径对照。
- **每（期 × worker）持有一个长驻 `Model`**（一个模型同时承担前向 MIP 与后向 LP，
  不必开两个——内存优先）：首次建好结构，之后每轮只 `setInfo` 改状态界 +
  改 `dem_r / age_{r,a}` 的右端，`addConstr` 追新割，直接 `solveLP()` 取
  `getRedcosts()`。
- **目标值按求解类型分开读**：MIP 用 `BestObj`，LP 用 `LpObjVal`，**不用 `objval`**
  （§7.2 的粘性陷阱）。这样前向 MIP 与后向 LP 交替跑同一个模型是安全的。
- **每个 worker 设 `Threads=1`**（§8.2）：并行度靠多进程，进程内不再并行。
- **每次求解后立即查 `status`**，非 `OPTIMAL` 就中止并落盘诊断
  （`writeLp` 到 ASCII 临时目录再搬回项目）。
- **`_t_mut_dll.py` 的 ctypes 路线不必实现**——原生 DLL 能做的 coptpy 全都能做，
  且同样快，还少一层手写原型和越界风险。
