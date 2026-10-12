# COPT Python 接口（coptpy）用法巡检：建模 / 原地改参数 / 求解 / 取对偶 / 存取 /
# MIP / 参数 / 热启动 / 计时。手册 COPT使用手册.md 的每条结论都由本文件跑出来。
#   python test/_t_copt_survey.py
#
# 基准 LP（与 test/_t_mut_coptpy.py 同题，期望值手算）：
#     min  4x + y + 100w
#     c1:  x + y + w >= 3      c2: 2x + 3y <= 8      0 <= x,y,w <= 10
#   解 x=1, y=2, w=0, obj=6；π=(10,−3)；dj=(0,0,90)

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

import tempfile                                                        # noqa: E402

import coptpy as cp                                                    # noqa: E402
from coptpy import COPT                                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

ENV = cp.Envr()
# COPT 的文件接口走窄字符路径：中文路径 writeLp/write/writeMps/writeBin/setLogFile
# 全部失败（短路径静默不建文件，长路径报 CoptError）。项目路径是中文，故落盘一律
# 先写 ASCII 临时目录再搬回项目 —— 这条对 LP/IIS/basis/log 都成立。
SCRATCH = Path(tempfile.gettempdir()) / "copt_survey"
ROWS = []


def new_model():
    m = ENV.createModel()
    m.setParam(COPT.Param.Logging, 0)
    return m


def close(a, b, tol=1e-9):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(u, v, tol) for u, v in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def check(name, got, want, tol=1e-9, note=""):
    ok = close(got, want, tol)
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))
    return ok


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- §1 建模
head("§1 建模：变量 / 约束 / 目标")
m = new_model()
x = m.addVar(lb=0.0, ub=10.0, obj=4.0, vtype=COPT.CONTINUOUS, name="x")
y = m.addVar(lb=0.0, ub=10.0, obj=1.0, name="y")            # obj/vtype 可省
w = m.addVars([0], lb=0.0, ub=10.0, obj=100.0, nameprefix="w")[0]
c1 = m.addConstr(x + y + w >= 3, name="c1")                 # 比较式直接带 sense
c2 = m.addConstr(2 * x + 3 * y <= 8, name="c2")             # <= / >= / ==
c3 = m.addBoundConstr(x + y, lb=0.0, ub=COPT.INFINITY, name="c3")   # 范围约束
m.setObjective(4 * x + y + 100 * w, COPT.MINIMIZE)          # 也可省，obj 已在 addVar
print(f"  变量数={m.getAttr(COPT.Attr.Cols)}  约束数={m.getAttr(COPT.Attr.Rows)}  "
      f"非零={m.getAttr(COPT.Attr.Elems)}  IsMIP={m.getAttr(COPT.Attr.IsMIP)}")
print(f"  取对象：getVarByName('y')={m.getVarByName('y')}  "
      f"getConstrByName('c1')={m.getConstrByName('c1')}")
print(f"  目标表达式 getObjective() = {m.getObjective()}")
check("变量数", m.getAttr(COPT.Attr.Cols), 3)
check("约束数", m.getAttr(COPT.Attr.Rows), 3)
check("目标系数读回", [v.obj for v in (x, y, w)], [4.0, 1.0, 100.0])
m.remove(c3)                                                # 多余的挂钩去掉
check("remove 约束后行数", m.getAttr(COPT.Attr.Rows), 2)

# ---------------------------------------------------------------- §2 求解
head("§2 求解与结果读取")
m.solveLP()
print(f"  objval={m.objval}  lpobjval={m.lpobjval}  status={m.status} "
      f"(COPT.OPTIMAL={COPT.OPTIMAL})")
print(f"  getValues()={m.getValues()}   getDuals()={m.getDuals()}   "
      f"getRedcosts()={m.getRedcosts()}   getSlacks()={m.getSlacks()}")
print(f"  getInfo 批式：Value={m.getInfo(COPT.Info.Value, [x, y, w])}  "
      f"Dual={m.getInfo(COPT.Info.Dual, [c1, c2])}")
print(f"  Attr：SimplexIter={m.getAttr(COPT.Attr.SimplexIter)}  "
      f"SolvingTime={m.getAttr(COPT.Attr.SolvingTime):.4f}s  "
      f"HasLpSol={m.getAttr(COPT.Attr.HasLpSol)}")
check("obj", m.objval, 6.0)
check("values", m.getValues(), [1.0, 2.0, 0.0])
check("duals", m.getDuals(), [10.0, -3.0])
check("redcosts", m.getRedcosts(), [0.0, 0.0, 90.0])
check("Info.Value 批式 == getValues", m.getInfo(COPT.Info.Value, [x, y, w]),
      m.getValues())
check("status", m.status, COPT.OPTIMAL)

# ---------------------------------------------------------------- §3 原地改参数
head("§3 原地改参数：变量界 / 行界 / 目标系数 / 矩阵系数（改完直接重解）")
base = m.objval
m.setInfo(COPT.Info.LB, [x], 2.0)                 # 变量下界（可一次改一批）
m.solveLP()
check("改变量界 x.lb=2 → obj", m.objval, 9.0, note="手算")
m.setInfo(COPT.Info.LB, [x], 0.0)

m.setInfo(COPT.Info.UB, [c2], 9.0)                # 行上界 = ≤ 行的右端
m.solveLP()
check("改行右端 c2.ub=9 → obj", m.objval, 3.0, note="手算")
m.setInfo(COPT.Info.UB, [c2], 8.0)

m.setInfo(COPT.Info.LB, [c1], 4.0)                # 行下界 = ≥ 行的右端
m.solveLP()
check("改行右端 c1.lb=4 → obj", m.objval, 16.0, note="手算")
m.setInfo(COPT.Info.LB, [c1], 3.0)

m.setInfo(COPT.Info.Obj, [w], 50.0)               # 目标系数
m.solveLP()
check("改目标系数 w.obj=50 → obj", m.objval, 6.0, note="仍是 x=1,y=2,w=0")
m.setInfo(COPT.Info.Obj, [w], 100.0)

m.setCoeff(c2, x, 3.0)                            # 矩阵系数：2x+3y<=8 → 3x+3y<=8
check("setCoeff 后行表达式读回", str(m.getRow(c2)).count("3.0"),
      2, note=f"{m.getRow(c2)}")
m.solveLP()
# x+y+w >= 3 与 3(x+y) <= 8 相容：x=0, y=8/3, w=1/3 → 36。手算式须带上 w
check("改矩阵系数 c2.x: 2→3 → obj", m.objval, 36.0, note="x=0, y=8/3, w=1/3")
m.setCoeff(c2, x, 2.0)
check("表格全部还原后 obj", m.solveLP() or m.objval, 6.0)

# 属性赋值也可（.pyi 里 lb/ub/obj 声明可写）；单对象形式 setInfo 同效
y.lb = 0.0
c1.setInfo(COPT.Info.LB, 3.0)
check("属性/单对象赋值后仍为基准", m.solveLP() or m.objval, 6.0)

# ---------------------------------------------------------------- §3b 陈旧值陷阱
head("§3b 求解失败时的陈旧值陷阱（增量求解必须无条件先查 status）")
m5 = new_model()
p = m5.addVar(lb=0.0, ub=10.0, obj=1.0, name="p")
q = m5.addVar(lb=0.0, ub=10.0, obj=1.0, name="q")
m5.addConstr(p + q >= 5, name="cc")
m5.solveLP()
obj_ok = m5.objval
bad = m5.addConstr(p + q <= 1, name="bad")          # 与 cc 直接冲突
m5.solveLP()
stale_objval = m5.objval
stale_attr = m5.getAttr(COPT.Attr.LpObjVal)
print(f"  可行时 objval={obj_ok}；加入冲突约束重解后：")
print(f"    status={m5.status}（COPT.INFEASIBLE={COPT.INFEASIBLE}）  "
      f"HasLpSol={m5.getAttr(COPT.Attr.HasLpSol)}")
print(f"    objval={stale_objval}  Attr.LpObjVal={stale_attr}  ← 两者都是上一次的陈旧值")
check("status 正确报不可行", m5.status, COPT.INFEASIBLE, note="唯一可靠的守卫")
check("HasLpSol 正确清零", m5.getAttr(COPT.Attr.HasLpSol), 0)
check("objval 是陈旧值（不能用它判断成败）", stale_objval, obj_ok,
      note="陷阱：不报错，静默返回旧值")
for fname, fn in (("getValues", m5.getValues), ("getRedcosts", m5.getRedcosts),
                  ("getDuals", m5.getDuals)):
    try:
        fn()
        ROWS.append((f"{fname} 应当抛错", "FAIL", "未抛错", "CoptError", ""))
    except Exception as exc:
        ROWS.append((f"{fname} 正确抛错", "PASS", type(exc).__name__, "CoptError", ""))
m5.remove([bad])
m5.solveLP()
check("删掉冲突约束后恢复", m5.objval, obj_ok)

# ---------------------------------------------------------------- §4 热启动
head("§4 热启动：原地改界后重解，单纯形迭代数与首次求解的关系")
m2 = new_model()
xs = m2.addVars(range(400), lb=0.0, ub=1.0, obj=1.0, nameprefix="v")
m2.addConstrs((xs[i] + xs[i + 1] >= 1.0 for i in range(399)), nameprefix="r")
m2.setObjective(cp.quicksum(xs), COPT.MINIMIZE)
m2.solveLP()
it_first = m2.getAttr(COPT.Attr.SimplexIter)
it_seq = []
for k in range(5):
    m2.setInfo(COPT.Info.UB, [xs[k]], 0.0)         # 连续改几个变量的上界
    m2.solveLP()
    it_seq.append(m2.getAttr(COPT.Attr.SimplexIter))
print(f"  首次求解迭代数={it_first}   改界后重解迭代数={it_seq}")
check("改界后重解迭代数不劣于首次", max(it_seq) <= it_first + 2, True,
      note=f"{it_seq} vs {it_first}")

# ---------------------------------------------------------------- §5 存取
head("§5 存取：write/read 各格式往返，解与对偶在磁盘上是否保真")
SCRATCH.mkdir(exist_ok=True)
for ext in (".lp", ".mps", ".bin"):
    p = SCRATCH / f"probe{ext}"
    m.write(str(p))
    m3 = new_model()
    m3.read(str(p))                                # read 按扩展名认格式
    m3.solveLP()
    check(f"{ext} 往返 obj", m3.objval, 6.0, note=f"文件 {p.stat().st_size} B")
p = SCRATCH / "probe.sol"
m.writeSol(str(p))
print(f"  writeSol → probe.sol（{p.stat().st_size} B，解析见 §2 的 getValues）")
m.writeBasis(str(SCRATCH / "probe.bas"))
print(f"  writeBasis → probe.bas（{SCRATCH.joinpath('probe.bas').stat().st_size} B）")
try:                                               # 可行模型没有 IIS，会抛 INVALID
    m.writeIIS(str(SCRATCH / "probe.iis"))
    print(f"  writeIIS → probe.iis（可行模型却写出来了，需留意）")
except Exception as exc:
    print(f"  writeIIS → 可行模型报 {type(exc).__name__}:{exc}（先 computeIIS 再写）")

mi = new_model()                                   # 造一个真的不可行模型走通 IIS 流程
u = mi.addVar(lb=0.0, ub=1.0, name="u")
mi.addConstr(u >= 2, name="bad_lo")
mi.addConstr(u <= 0, name="bad_hi")
mi.solveLP()
mi.computeIIS()
iis_path = SCRATCH / "probe.iis"
mi.writeIIS(str(iis_path))
iis_txt = iis_path.read_text(encoding="utf-8", errors="replace")
print(f"  不可行模型：status={mi.status}（COPT.INFEASIBLE={COPT.INFEASIBLE}）  "
      f"HasIIS={mi.getAttr(COPT.Attr.HasIIS)}  IISRows={mi.getAttr(COPT.Attr.IISRows)}")
print("  IIS 内容：", " | ".join(l.strip() for l in iis_txt.splitlines()
                                if l.strip() and not l.startswith("\\")
                                and l.strip() not in ("Minimize", "Subject To",
                                                      "Bounds", "END")))
# 最小冲突是 {bad_lo, u ≤ 1}：一行 + 一个变量界。IISRows 只数行，故为 1 而非 2
check("IIS 只含 1 行（另一条冲突是变量界）", mi.getAttr(COPT.Attr.IISRows), 1)
check("IIS 文件确实指向 bad_lo", "bad_lo" in iis_txt, True)
cl = m.clone()
cl.solveLP()
check("clone 后同解", cl.objval, 6.0)

print("  中文路径实测（项目路径本身是中文，必须知道哪些落盘接口会挂）：")
for label, fn, fname in [("writeLp", m.writeLp, "cn_probe.lp"),
                         ("writeMps", m.writeMps, "cn_probe.mps"),
                         ("writeBin", m.writeBin, "cn_probe.bin")]:
    cn_dir = Path(__file__).resolve().parent / "_scratch_cn"
    cn_dir.mkdir(exist_ok=True)
    target = cn_dir / fname
    try:
        fn(str(target))
        ok = target.exists()
        print(f"    {label:9s} → 项目中文路径：{'成功' if ok else '静默失败（无文件）'}")
    except Exception as exc:
        print(f"    {label:9s} → 项目中文路径：报错 {type(exc).__name__}")

# ---------------------------------------------------------------- §6 MIP
head("§6 MIP：整数变量、solve 与 solveLP（松弛）的差别、MIP start")
m4 = new_model()
a = m4.addVar(lb=0.0, ub=3.0, obj=1.0, vtype=COPT.INTEGER, name="a")
b = m4.addVar(lb=0.0, ub=3.0, obj=10.0, vtype=COPT.INTEGER, name="b")
m4.addConstr(2 * a + 3 * b >= 6, name="d")
m4.setObjective(a + 10 * b, COPT.MINIMIZE)
m4.solveLP()
lpv = (m4.objval, list(m4.getValues()))
m4.solve()
mipv = (m4.objval, list(m4.getValues()))
print(f"  松弛obj={lpv[0]}  整数obj={mipv[0]}  "
      f"MipStatus={m4.getAttr(COPT.Attr.MipStatus)}  "
      f"HasMipSol={m4.getAttr(COPT.Attr.HasMipSol)}")
check("MIP 解为整数", mipv[1], [3.0, 0.0], note="松弛解为 [0,2] 才最优")
check("MIP obj", mipv[0], 3.0)
check("松弛 obj", lpv[0], 3.0)
m4.setMipStart([a, b], [0.0, 2.0])                 # 给一个可行起点
m4.solve()
check("MIP start 后 obj 不变", m4.objval, 3.0)
m4.solveLP()                                       # 整数变量上跑 LP 松弛
check("整数模型上 solveLP() 仍可提对偶", len(m4.getDuals()), 1,
      note=f"Dual={m4.getDuals()} RedCost={m4.getRedcosts()}")

# ---------------------------------------------------------------- §6b MIP↔LP 切换
head("§6b 同一模型上交替 solve()/solveLP()：objval 对 LP 值有粘性，会与解向量脱节")
# min a+b  s.t. 2a+2b>=3, a,b∈[0,2]∩Z   松弛=1.5（a=0,b=1.5）  整数=2.0（a=0,b=2）


def _mk_relax():
    mm = new_model()
    p = mm.addVar(lb=0, ub=2, obj=1.0, vtype=COPT.INTEGER, name="a")
    q = mm.addVar(lb=0, ub=2, obj=1.0, vtype=COPT.INTEGER, name="b")
    mm.addConstr(2 * p + 2 * q >= 3, name="d")
    mm.setObjective(p + q, COPT.MINIMIZE)
    return mm


def _obj_from_values(mm):
    co = mm.getObjective()
    return sum(co.getCoeff(i) * mm.getValues()[i] for i in range(len(mm.getValues())))


m6 = _mk_relax()
m6.solveLP()
lp_only = (m6.objval, m6.LpObjVal, _obj_from_values(m6))
m6.solve()                                        # 紧接一次 MIP
mixed = {"objval": m6.objval, "ObjVal": m6.ObjVal, "LpObjVal": m6.LpObjVal,
         "BestObj": m6.BestObj, "status": m6.status,
         "HasLpSol": m6.getAttr("HasLpSol"), "HasMipSol": m6.getAttr("HasMipSol"),
         "values": list(m6.getValues()), "from_values": _obj_from_values(m6)}
print(f"  solveLP() 后：objval={lp_only[0]} LpObjVal={lp_only[1]} "
      f"解向量算得 {lp_only[2]}")
print(f"  紧接 solve()：status={mixed['status']} "
      f"HasMipSol={mixed['HasMipSol']} 解向量={mixed['values']} → 实算 {mixed['from_values']}")
print(f"    objval={mixed['objval']}  ← 仍是 LP 值 1.5，与解向量 (0,2) 对不上")
print(f"    BestObj={mixed['BestObj']}  ← 整数目标，正确")
check("solveLP 后 solve 的向量是整数解", mixed["values"], [0.0, 2.0])
check("该向量实算目标 = 2.0", mixed["from_values"], 2.0)
check("objval 却报 LP 值（粘性）", mixed["objval"], 1.5, note="陷阱：与解向量脱节")
check("BestObj 才是整数目标", mixed["BestObj"], 2.0)
check("status 无法区分（照样 OPTIMAL）", mixed["status"], COPT.OPTIMAL,
      note="陷阱：Status 不足以判断读哪个目标")
check("HasLpSol/HasMipSol 可区分", [mixed["HasLpSol"], mixed["HasMipSol"]], [1, 1])
m6.solve()                                        # 再解一次也治不好
check("再 solve() 一次 objval 仍粘在 LP 值", m6.objval, 1.5)
m6.solveLP()
check("切回 solveLP 后 objval 正确", m6.objval, 1.5, note="此例恰好 LP 值也是 1.5")

m7 = _mk_relax()
m7.solve()                                        # 全新模型，从不碰 solveLP
check("从不碰 solveLP 的模型 objval 正确", m7.objval, 2.0,
      note="对照：混合使用才会脱节")

# ---------------------------------------------------------------- §7 参数
head("§7 参数与调试开关")
m.setParam(COPT.Param.Logging, 0)
m.setParam(COPT.Param.Presolve, 1)
m.setParam(COPT.Param.Threads, 4)
print(f"  Logging={m.getParam(COPT.Param.Logging)}  "
      f"Presolve={m.getParam(COPT.Param.Presolve)}  "
      f"Threads={m.getParam(COPT.Param.Threads)}  "
      f"TimeLimit={m.getParam(COPT.Param.TimeLimit)}")
check("参数读回", [m.getParam(COPT.Param.Logging), m.getParam(COPT.Param.Presolve)],
      [0, 1])
m.resetParam()                                     # 注意：resetParam() 一次复位全部参数
print(f"  resetParam() 后 Threads={m.getParam(COPT.Param.Threads)}  "
      f"Logging={m.getParam(COPT.Param.Logging)}（全部回默认）")

# ---------------------------------------------------------------- §8 计时
head("§8 计时：coptpy 直接建模 vs pulp 建模 + extract + LoadProb")
# 这一节是**历史对照**：证明重构前那条 pulp 路径慢在哪。生产代码已全走 coptpy，
# pulp 只在这个计时对照里出现（参照件见 test/_pulp_ref_model.py）。
import dataclasses                                                     # noqa: E402
import numpy as np                                                     # noqa: E402
import pulp                                                            # noqa: E402

from code.config import load_problem_config                            # noqa: E402
from code.problem_data import DefineProblem                            # noqa: E402
import _pulp_ref_model as pulp_ref                                     # noqa: E402

pd_ = DefineProblem(dataclasses.replace(
    load_problem_config(ROOT / "config/smoke/problem.toml"),
    T=4, A=24, N_R=3, pool_size=10, spread_init=True))
st0 = pd_.initial_state()
sub = pulp_ref.StageSubproblem(1, st0, (None, np.zeros((pd_.N_R, pd_.A))),
                               None, pd_, 0)
NC = len(sub.prob.variables())
NR = len(sub.prob.constraints)
NNZ = sum(len(c) for c in sub.prob.constraints.values())
print(f"  真实子问题规模：{NC} 列 / {NR} 行 / {NNZ} 非零")

# (a) coptpy 直建同规模稀疏 LP
t0 = time.perf_counter()
mc = new_model()
vv = mc.addVars(range(NC), lb=0.0, ub=1.0, nameprefix="q")
rows = []
for i in range(NR):
    k = i % max(1, NC - 2)
    rows.append(mc.addConstr(vv[k] + 0.5 * vv[k + 1] - vv[k + 2] <= 1.0,
                             name=f"r{i}"))
mc.setObjective(cp.quicksum(vv), COPT.MINIMIZE)
t_copt_build = time.perf_counter() - t0
t0 = time.perf_counter()
mc.solveLP()
t_copt_solve = time.perf_counter() - t0

# (b) pulp 建同规模模型 + COPT_DLL 求解（= 重构前的路径）
t0 = time.perf_counter()
pp = pulp.LpProblem("probe", pulp.LpMinimize)
pv = [pulp.LpVariable(f"q{j}", 0, 1) for j in range(NC)]
for i in range(NR):
    k = i % max(1, NC - 2)
    pp += pv[k] + 0.5 * pv[k + 1] - pv[k + 2] <= 1.0, f"r{i}"
pp += pulp.lpSum(pv)
t_pulp_build = time.perf_counter() - t0
import contextlib                                                      # noqa: E402
import io                                                              # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    solver = pulp.COPT_DLL(msg=False, mip=False)
t0 = time.perf_counter()
pp.solve(solver)
t_pulp_solve = time.perf_counter() - t0

# (c) 直建之后原地改界 + 重解
# 注意 setInfo 的容器分派：tupledict + list 会 TypeError（被当成单值走 float()），
# 要么 list(dict.values()) 要么 tupledict + 标量。addVars 的返回值正是 tupledict。
vv_list = list(vv.values())
t0 = time.perf_counter()
for _ in range(10):
    mc.setInfo(COPT.Info.UB, vv_list, [1.0] * NC)
    mc.solveLP()
t_inc = (time.perf_counter() - t0) / 10

print(f"  coptpy 直接建模     {t_copt_build*1e3:8.2f} ms"
      f"     首解 {t_copt_solve*1e3:7.2f} ms")
print(f"  pulp 建模          {t_pulp_build*1e3:8.2f} ms"
      f"     解(含extract+LoadProb) {t_pulp_solve*1e3:7.2f} ms")
print(f"  coptpy 原地改界+重解 {t_inc*1e3:8.2f} ms/次")
print(f"  建 + 解 合计：coptpy {(t_copt_build+t_copt_solve)*1e3:.2f} ms  vs  "
      f"pulp {(t_pulp_build+t_pulp_solve)*1e3:.2f} ms")

# ---------------------------------------------------------------- 汇总
head("汇总")
wd = max(len(r[0]) for r in ROWS)
n_fail = 0
for name, st, got, want, note in ROWS:
    n_fail += st == "FAIL"
    g = got if isinstance(got, str) else (
        f"{got:.6f}" if not isinstance(got, list) else str([round(v, 4) for v in got]))
    print(f"  [{st}] {name:<{wd}}  got={g}  期望={want}  {note}")
print()
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
