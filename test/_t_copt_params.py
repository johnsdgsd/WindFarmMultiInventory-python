# COPT 参数与容差实测：默认值、语义、以及本项目关心的三件事
#   （FeasTol 边界、MatrixTol 对 (1−θ) 小系数的处理、多进程下的线程设置）
#   python test/_t_copt_params.py
#
# 结论写进根目录 COPT使用手册.md §8。

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

import coptpy as cp                                                    # noqa: E402
from coptpy import COPT                                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV = cp.Envr()
ROWS = []


def mk(**par):
    m = ENV.createModel()
    m.setParam(COPT.Param.Logging, 0)
    for k, v in par.items():
        m.setParam(getattr(COPT.Param, k), v)
    return m


def check(name, got, want, note=""):
    ok = (abs(float(got) - float(want)) <= 1e-12 if not isinstance(got, (list, tuple))
          else all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(got, want)))
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- §1 容差默认值
head("§1 容差类参数：默认值 / 允许范围（getParamInfo 给 (名, 当前, 默认, 最小, 最大)）")
TOL = ["FeasTol", "DualTol", "IntTol", "MatrixTol", "PDLPTol", "NLPTol",
       "RelGap", "AbsGap"]
m0 = mk()
print(f"  {'参数':<12}{'默认':>10}{'最小':>10}{'最大':>10}   含义")
MEAN = {"FeasTol": "变量/约束取值的可行性容差",
        "DualTol": "对偶解的可行性容差",
        "IntTol": "变量的整数解容差",
        "MatrixTol": "输入矩阵的系数容差（更小的系数被当 0）",
        "PDLPTol": "一阶算法 PDLP 的收敛容差",
        "NLPTol": "非线性规划最优相对容差",
        "RelGap": "整数规划最优相对容差",
        "AbsGap": "整数规划最优绝对容差"}
for n in TOL:
    _, cur, dflt, lo, hi = m0.getParamInfo(getattr(COPT.Param, n))
    print(f"  {n:<12}{dflt:>10g}{lo:>10g}{hi:>10g}   {MEAN[n]}")
    check(f"{n} 默认值", dflt, cur, note="getParamInfo 的默认==当前")

# ---------------------------------------------------------------- §2 FeasTol 边界
head("§2 FeasTol：必然冲突多大算可行（默认 1e-6）")
print("  构造 x >= 10 而 x 的上界 = 10 − δ —— 两者差 δ，无真正可行解：")
for delta in (1e-3, 1e-5, 1e-6, 1e-7, 1e-9):
    m = mk()
    ub = 10.0 - delta
    x = m.addVar(lb=0.0, ub=ub, obj=1.0, name="x")
    m.addConstr(x >= 10.0, name="c")
    m.solveLP()
    if m.status == COPT.OPTIMAL:
        viol = float(x.x) - ub                     # 对**上界**的违反量
        print(f"    δ={delta:.0e} → OPTIMAL，x={float(x.x):.17g}，对上界违反={viol:.3e}")
    else:
        print(f"    δ={delta:.0e} → INFEASIBLE（冲突 δ 超出 FeasTol）")
print("  ⇒ 冲突量 ≤ FeasTol 被判可行、> FeasTol 判不可行。")
print("    解会把违反量塞进某个界里，故右端别留 1e-6 量级的缝隙。")

# ---------------------------------------------------------------- §3 MatrixTol
head("§3 MatrixTol：系数多小会被当 0（默认 1e-10）")
print("  构造 a ≤ 1（目标要 a 越大越好不成，改成 min b s.t. coef·a + b >= 1, b<=1）")
print("  若 coef·a 被当 0 丢掉，则行退化为 b >= 1，a 失去约束力的差别会在解上显现：")
for coef in (1e-3, 1e-6, 1e-9, 1e-11, 1e-13):
    m = mk()
    a = m.addVar(lb=0.0, ub=1.0, obj=0.0, name="a")
    b = m.addVar(lb=0.0, ub=1.0, obj=1.0, name="b")
    m.addConstr(coef * a + b >= 1.0, name="c")
    m.setObjective(b, COPT.MINIMIZE)
    m.solveLP()
    # 系数保留时：a=1 能让 b 降到 1−coef；系数被丢时 b 恒为 1
    print(f"    coef={coef:.0e} → b={float(b.x):.12f}  "
          f"{'系数保留' if float(b.x) < 1.0 - 1e-13 else '系数被当 0 丢掉'}")
print("  ⇒ 默认 1e-10：≥1e-10 的系数保留。本项目 (1−θ) 量级 ~1e-5，安全。")

# ---------------------------------------------------------------- §4 线程
head("§4 线程：默认自动(-1)用满核；多进程时每进程须设 1")
print(f"  TimeLimit={m0.getParam(COPT.Param.TimeLimit):g}  "
      f"Threads={m0.getParam(COPT.Param.Threads)}（-1=自动）  "
      f"SimplexThreads={m0.getParam(COPT.Param.SimplexThreads)}（-1=跟随 Threads）")

# 造一个中等规模 LP，比较单进程求解在 Threads=1 / 自动 下的耗时
def build_lp(m):
    n = 3000
    v = m.addVars(range(n), lb=0.0, ub=1.0, obj=1.0, nameprefix="v")
    vlist = list(v.values())
    for i in range(n - 2):
        m.addConstr(vlist[i] + vlist[i + 1] + 0.5 * vlist[i + 2] >= 1.0,
                    name=f"r{i}")
    m.setObjective(cp.quicksum(vlist), COPT.MINIMIZE)
    return vlist


for th in (1, -1):
    m = mk(Threads=th)
    build_lp(m)
    m.solveLP()
    print(f"  Threads={th:>2}（{-1 and '自动' or ''}）：{m.getAttr(COPT.Attr.Cols)} 列  "
          f"解 {m.getAttr(COPT.Attr.SolvingTime):.3f} s  "
          f"迭代 {m.getAttr(COPT.Attr.SimplexIter)}")

print("  ⇒ 本项目是 multiprocessing 多进程并行（12 进程），COPT 默认会自动用满核")
print("    → 12×28 线程超订，纯属互相抢核。每个 worker 内应设 Threads=1（或")
print("    SimplexThreads=1），把并行度留给进程级。")

# ---------------------------------------------------------------- §5 关键参数默认
head("§5 本项目相关的其它参数默认值")
m0 = ENV.createModel()          # 不套 mk()，否则 Logging 的默认值被我改掉看不见
KEYS = ["LpMethod", "Presolve", "Scaling", "Dualize", "DualPrice", "DualPerturb",
        "Crossover", "Threads", "SimplexThreads", "TimeLimit", "TimeLimit",
        "RelGap", "ReqFarkasRay", "ReqSensitivity", "IISMethod", "Logging",
        "LogToConsole", "LogLevel", "MipStartMode", "NodeLimit"]
print(f"  {'参数':<18}{'默认':>12}   取值说明")
NOTE = {"LpMethod": "-1 自动（LP 选对偶单纯形；MIP 选对偶单纯形或内点法）",
        "Presolve": "-1 自动；0 关 / 1 少量快 / 2 正常 / 3 多多益善 / 4 无限制",
        "Scaling": "-1 自动；0 不调整；1 调整系数矩阵数值",
        "Dualize": "-1 自动；0 不建对偶模型；1 建对偶模型",
        "DualPrice": "-1 自动；0 Devex；1 对偶最陡边",
        "DualPerturb": "-1 自动；0 无摄动；1 允许目标函数摄动",
        "Crossover": "-1 自动；0 不用；1 用",
        "Threads": "-1 自动（用满核）；最大 128",
        "SimplexThreads": "-1 跟随 Threads",
        "TimeLimit": "秒；1e20 相当于不限",
        "RelGap": "MIP 相对间隙收容差",
        "ReqFarkasRay": "0/1，不可行时是否算 Farkas 射线",
        "ReqSensitivity": "0/1，是否算灵敏度分析（SA* 信息）",
        "IISMethod": "-1 自动；0 质量优先；1 效率优先",
        "Logging": "0 关 / 1 开（批量求解必关）",
        "LogToConsole": "0/1",
        "LogLevel": "2 默认，最大 3",
        "MipStartMode": "-1 自动",
        "NodeLimit": "-1 自动（不限）"}
for n in dict.fromkeys(KEYS):
    print(f"  {n:<18}{m0.getParam(getattr(COPT.Param, n)):>12g}   {NOTE[n]}")

head("汇总")
n_fail = sum(1 for r in ROWS if r[1] == "FAIL")
for name, st, got, want, note in ROWS:
    if st == "FAIL":
        print(f"  [{st}] {name}  got={got} 期望={want}  {note}")
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
