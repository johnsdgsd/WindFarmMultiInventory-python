# coptpy 原地改模型参数的最小验证：变量界 / 约束右端能不能改、改了起不起作用、
# 对偶能不能提。这决定后向子问题是否走"建一次模型、反复改参数求解"的增量路线。
#   python test/_t_mut_coptpy.py
#
# 基准 LP（期望值全部手算，不用求解器自证）：
#     min  4x + y + 100w
#     c1:  x + y + w >= 3         行下界 = 3
#     c2:  2x + 3y   <= 8         行上界 = 8
#     0 <= x, y, w <= 10
#   解 x=1, y=2, w=0, obj=6；对偶 π=(10,−3)；既约成本 dj=(0,0,90)。
#   自洽：π1·3 + π2·8 = 30 − 24 = 6 = obj（强对偶）；dj_w = 100 − π1 = 90。
#
# 四条原地改的期望值（各自单独施加）：
#     var_lb  x.lb 0→2     → 9       var_ub  y.ub 10→0.5 → 10.5
#     con_lb  c1.lb 3→4    → 16      con_ub  c2.ub 8→9   → 3
#
# 注：行界与变量界都统一走 m.setInfo(COPT.Info.LB/UB, 对象, 值)——COPT 手册
# 明确 LB/UB 同时适用于"变量（列）"与"约束（行）"，这是 coptpy 唯一的原地改口。

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

import coptpy as cp                                                    # noqa: E402
from coptpy import COPT                                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV = cp.Envr()
INF = COPT.INFINITY


def new_model():
    m = ENV.createModel()
    m.setParam(COPT.Param.Logging, 0)
    return m

BASE = {"x_lb": 0.0, "x_ub": 10.0, "y_lb": 0.0, "y_ub": 10.0,
        "w_lb": 0.0, "w_ub": 10.0, "c1_lb": 3.0, "c1_ub": INF,
        "c2_lb": -INF, "c2_ub": 8.0}


def make_model(**over):
    """按 BASE 建新模型，over 的键覆盖对应界。返回 (模型, [x,y,w], [c1,c2])。"""
    b = {**BASE, **over}
    m = new_model()
    v = [m.addVar(lb=b["x_lb"], ub=b["x_ub"], obj=4.0, name="x"),
         m.addVar(lb=b["y_lb"], ub=b["y_ub"], obj=1.0, name="y"),
         m.addVar(lb=b["w_lb"], ub=b["w_ub"], obj=100.0, name="w")]
    x, y, w = v
    c = [m.addConstr(x + y + w >= b["c1_lb"], name="c1"),
         m.addConstr(2 * x + 3 * y <= b["c2_ub"], name="c2")]
    return m, v, c


def solve(m, v, c):
    m.solveLP()
    return {"obj": float(m.objval),
            "x": list(m.getInfo(COPT.Info.Value, v)),
            "pi": list(m.getInfo(COPT.Info.Dual, c)) if c else [],
            "dj": list(m.getInfo(COPT.Info.RedCost, v)),
            "slack": list(m.getInfo(COPT.Info.Slack, c)) if c else []}


ROWS = []


def close(a, b, tol):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(u, v, tol) for u, v in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def check(name, got, want, tol=1e-9, note=""):
    ok = close(got, want, tol)
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))
    return ok


print("=" * 78)
print("§1 基准解与对偶提取（m.getInfo 四类：Value / Dual / RedCost / Slack）")
print("=" * 78)
s0 = solve(*make_model())
print(f"  obj={s0['obj']:.6f}  x={s0['x']}  π={s0['pi']}  dj={s0['dj']}  "
      f"slack={s0['slack']}")
check("基准 obj", s0["obj"], 6.0)
check("基准 x", s0["x"], [1.0, 2.0, 0.0])
check("对偶 π", s0["pi"], [10.0, -3.0])
check("既约成本 dj", s0["dj"], [0.0, 0.0, 90.0])
check("强对偶 π1·3+π2·8", s0["pi"][0] * 3 + s0["pi"][1] * 8, 6.0)

print()
print("=" * 78)
print("§2 原地改 → 重解：与手算值比，再与「同改动重建的模型」逐位比")
print("=" * 78)


def case(name, expect_obj, mutate, rebuild_kw):
    """mutate(m, v, c) 在新建模型上原地改。行界与变量界都只用 setInfo 一条口径。"""
    m, v, c = make_model()
    mutate(m, v, c)
    got = solve(m, v, c)
    fs = solve(*make_model(**rebuild_kw))
    check(f"{name} · obj", got["obj"], expect_obj, note="手算")
    check(f"{name} · x", got["x"], fs["x"], note="vs 重建")
    check(f"{name} · obj vs 重建", got["obj"], fs["obj"], note="vs 重建")


case("A 变量下界 setInfo(LB,var) x.lb=2", 9.0,
     lambda m, v, c: m.setInfo(COPT.Info.LB, [v[0]], 2.0), {"x_lb": 2.0})
case("B 变量上界 setInfo(UB,var) y.ub=0.5", 10.5,
     lambda m, v, c: m.setInfo(COPT.Info.UB, [v[1]], 0.5), {"y_ub": 0.5})
case("C 行下界   setInfo(LB,con) c1.lb=4", 16.0,
     lambda m, v, c: m.setInfo(COPT.Info.LB, [c[0]], 4.0), {"c1_lb": 4.0})
case("D 行上界   setInfo(UB,con) c2.ub=9", 3.0,
     lambda m, v, c: m.setInfo(COPT.Info.UB, [c[1]], 9.0), {"c2_ub": 9.0})

# E：变量属性直接赋值（coptpy 的 .pyi 里 lb/ub 声明为可写，实测是否如此）
try:
    m, v, c = make_model()
    v[0].lb = 2.0
    got = solve(m, v, c)
    check("E 属性赋值 var.lb=2", got["obj"], 9.0, note="手算")
except Exception as exc:
    ROWS.append(("E 属性赋值 var.lb=2", "N/A", type(exc).__name__, "—", str(exc)[:44]))

# F：getRow 拿到的 LinExpr 是副本，改它不写回模型（预期"改了等于没改"）
m, v, c = make_model()
m.getRow(c[0]).setConstant(4.0)
got = solve(m, v, c)
check("F getRow 副本 setConstant(4) 不生效", got["obj"], 6.0, note="预期无效")
check("F 与原模型逐位同解", got["x"], s0["x"], note="预期无效")

print()
print("=" * 78)
print("§3 差商验证：原地改一次再读回来的 π / dj 是不是真的 ∂obj/∂参数")
print("=" * 78)
D = 1e-4
for label, mut, want, note in [
    ("∂obj/∂c1.LB == π1", lambda m, v, c: m.setInfo(COPT.Info.LB, [c[0]], 3.0 + D),
     s0["pi"][0], "行下界 +δ"),
    ("∂obj/∂c2.UB == π2", lambda m, v, c: m.setInfo(COPT.Info.UB, [c[1]], 8.0 + D),
     s0["pi"][1], "行上界 +δ"),
    ("∂obj/∂w.LB  == dj_w", lambda m, v, c: m.setInfo(COPT.Info.LB, [v[2]], D),
     s0["dj"][2], "变量下界 +δ"),
]:
    m, v, c = make_model()
    mut(m, v, c)
    got = (solve(m, v, c)["obj"] - s0["obj"]) / D
    check(label, got, want, tol=1e-4 * max(1.0, abs(want)), note=note)

print()
print("=" * 78)
print("§4 可逆性：四条改动全部还原后回到基准解")
print("=" * 78)
m, v, c = make_model()
m.setInfo(COPT.Info.LB, v, [2.0, 0.0, 0.0])
m.setInfo(COPT.Info.UB, v, [10.0, 0.5, 10.0])
m.setInfo(COPT.Info.LB, c, [4.0, -INF])
m.setInfo(COPT.Info.UB, c, [INF, 9.0])
solve(m, v, c)
m.setInfo(COPT.Info.LB, v, [0.0, 0.0, 0.0])
m.setInfo(COPT.Info.UB, v, [10.0, 10.0, 10.0])
m.setInfo(COPT.Info.LB, c, [3.0, -INF])
m.setInfo(COPT.Info.UB, c, [INF, 8.0])
back = solve(m, v, c)
check("还原后 obj", back["obj"], 6.0)
check("还原后 x", back["x"], [1.0, 2.0, 0.0])

print()
print("=" * 78)
print("§5 计时：原地改界+重解 vs 重建模型+解")
print("=" * 78)
K = 200                       # 拼成 600 列 / 400 行，接近真实子问题的量级


def make_big():
    m = new_model()
    v, c = [], []
    for k in range(K):
        a = m.addVar(lb=0, ub=10, obj=4.0, name=f"x{k}")
        b = m.addVar(lb=0, ub=10, obj=1.0, name=f"y{k}")
        w = m.addVar(lb=0, ub=10, obj=100.0, name=f"w{k}")
        c.append(m.addConstr(a + b + w >= 3, name=f"c1_{k}"))
        c.append(m.addConstr(2 * a + 3 * b <= 8, name=f"c2_{k}"))
        v += [a, b, w]
    return m, v, c


t0 = time.perf_counter()
mB, vB, cB = make_big()
t_build = time.perf_counter() - t0
t0 = time.perf_counter()
mB.solveLP()
t_first = time.perf_counter() - t0

n_it = 30
t0 = time.perf_counter()
for k in range(n_it):
    mB.setInfo(COPT.Info.UB, vB, [10.0] * len(vB))       # 等效两族批改
    mB.setInfo(COPT.Info.LB, cB, [3.0] * len(cB))
    mB.solveLP()
t_inplace = (time.perf_counter() - t0) / n_it

t0 = time.perf_counter()
for k in range(n_it):
    mm, vv, cc = make_big()
    mm.solveLP()
t_rebuild = (time.perf_counter() - t0) / n_it

print(f"  {K} 块（{3*K} 列 / {2*K} 行），每轮 = 变量界批改 + 行界批改")
print(f"    构建一次            {t_build*1e3:9.2f} ms")
print(f"    首次求解            {t_first*1e3:9.2f} ms")
print(f"    原地改界 + 重解     {t_inplace*1e3:9.2f} ms/次")
print(f"    重建模型 + 解       {t_rebuild*1e3:9.2f} ms/次")
print(f"    加速比              {t_rebuild/t_inplace:9.2f} ×")

print()
print("=" * 78)
print("汇总")
print("=" * 78)
w = max(len(r[0]) for r in ROWS)
n_fail = 0
for name, st, got, want, note in ROWS:
    n_fail += st == "FAIL"
    g = got if isinstance(got, str) else (f"{got:.6f}" if not isinstance(got, list)
                                         else str([round(x, 4) for x in got]))
    print(f"  [{st}] {name:<{w}}  got={g}  期望={want}  {note}")
print()
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
