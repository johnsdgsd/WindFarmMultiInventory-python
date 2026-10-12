# 拼装契约：base 只含阶段成本；assemble 挂割；换割集不破坏 base 的行句柄；
# 空割集必须等同"无割"（否则 theta 进目标却无约束 ⇒ LP 无界）。
#   python test/_t_assembly.py

import sys
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import StageSubproblem                           # noqa: E402
from utils import diagnostics as dg                                   # noqa: E402

prob = DefineProblem(load_problem_config(ROOT / "config/smoke/problem.toml"))
scfg = load_solver_config(ROOT / "config/smoke/solver.toml")
Ly = prob.layout
rng = np.random.default_rng(31337)
x0 = dg.probe_states(rng, prob, n=1)[0]
scene = prob.pool[9]

ROWS = []


def check(name, got, want, note=""):
    if isinstance(want, bool):
        ok = bool(got) == want
    elif isinstance(want, (int, str)):
        ok = got == want
    else:
        ok = abs(got - want) < 1e-6
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))
    return ok


def n_rows(sub):
    return int(sub.model.getAttr(COPT.Attr.Rows))


print("=" * 78)
print("§1 base 只含阶段成本：无割时 theta 不进目标，Q = 纯当期成本")
print("=" * 78)
sub = StageSubproblem(2, prob, scfg)
r_plain = n_rows(sub)
sub.set_point(x0, scene)
sub.solve_lp()
q_plain = sub.obj
print(f"  base 行数 {r_plain}，Q = {q_plain:.6f}，theta = {sub.theta_value()}")
check("无割时 theta 恒 0", sub.theta_value(), 0.0)
check("无割时 Q == 阶段成本", sub.obj, sub.stage_cost())

print()
print("=" * 78)
print("§2 assemble 挂割：theta 进目标并受割约束")
print("=" * 78)
# 平凡割 theta >= 0：真值函数本身非负，故挂上之后 Q 应当不变
sub.assemble((np.array([0.0]), np.zeros((1, Ly.dim))), key=("triv", 2))
r_cut = n_rows(sub)
sub.set_point(x0, scene)
sub.solve_lp()
print(f"  挂平凡割后行数 {r_cut}（base {r_plain} + 1），Q = {sub.obj:.6f}，"
      f"theta = {sub.theta_value():.6f}")
check("行数 = base + 割数", r_cut, r_plain + 1)
check("平凡割不改变 Q", sub.obj, q_plain)

# 一条真正来自下一期的割：在 x2_out 处解 t=3，用它的值与次梯度构 V_3 的下界支撑面。
# 截距式 alpha = Q(x̄) − beta@x̄；把 x̄ 写成 Q 会让割整体偏 beta@x̄，theta 掉到负值。
x_out = sub.out_state_vector().copy()               # t=2 的期末（= t=3 的期初）
nxt = StageSubproblem(3, prob, scfg)
nxt.assemble(None)
nxt.set_point(x_out, scene)
nxt.solve_lp()
b3 = nxt.gradient("dj").copy()
a3 = nxt.obj - float(b3 @ x_out)
sub.assemble((np.array([a3]), b3.reshape(1, -1)), key=("v3", 2))
sub.set_point(x0, scene)
sub.solve_lp()
print(f"  挂一条来自 t=3 的割：Q = {sub.obj:.6f}（base {q_plain:.6f}），"
      f"theta = {sub.theta_value():.6f}")
check("挂未来成本割后 Q 抬高", sub.obj > q_plain, True)
check("theta 非负", sub.theta_value() >= -1e-9, True)

print()
print("=" * 78)
print("§3 卸割：assemble(None) 回到纯阶段成本")
print("=" * 78)
sub.assemble(None, key=("b", 2))
sub.set_point(x0, scene)
sub.solve_lp()
print(f"  行数回到 {n_rows(sub)}，Q = {sub.obj:.6f}，theta = {sub.theta_value()}")
check("卸割后行数还原", n_rows(sub), r_plain)
check("卸割后 Q 还原", sub.obj, q_plain)

print()
print("=" * 78)
print("§4 换割集：行数稳定，base 的静态行句柄仍然有效")
print("=" * 78)
for k, n in enumerate((3, 8, 5)):
    a = rng.uniform(0, 100, size=n)
    b = rng.normal(0, 0.02, size=(n, Ly.dim))
    sub.assemble((a, b), key=("swap", k))
    ok = n_rows(sub) == r_plain + n
    print(f"  第 {k} 次换成交 {n} 条：行数 {n_rows(sub)}（期望 {r_plain + n}）"
          f"{'' if ok else '  <<< 行数不对'}")
    ROWS.append((f"换割集 {k} 行数", "PASS" if ok else "FAIL", n_rows(sub),
                 r_plain + n, ""))
# 静态句柄仍有效：改 dem_0 的右端必须仍然生效
r = sub._row("dem_0")
before = float(sub.model.getInfo(COPT.Info.UB, [r])[0])
sub.model.setInfo(COPT.Info.UB, [r], before + 1234.0)
after = float(sub.model.getInfo(COPT.Info.UB, [r])[0])
check("换割后 dem_0 句柄仍有效", after, before + 1234.0)
sub.model.setInfo(COPT.Info.UB, [r], before)

print()
print("=" * 78)
print("§5 空割集 == 无割（否则 theta 自由 + 目标系数 1 ⇒ LP 无界）")
print("=" * 78)
sub.assemble(None, key=("c", 2))
sub.set_point(x0, scene)
sub.solve_lp()
q_none = sub.obj
sub.assemble((np.zeros(0), np.zeros((0, Ly.dim))), key=("d", 2))
sub.set_point(x0, scene)
sub.solve_lp()
print(f"  None → Q = {q_none:.6f}；空 array → Q = {sub.obj:.6f}")
check("空割集与 None 同解", sub.obj, q_none)
check("空割集行数 == base", n_rows(sub), r_plain)

print()
print("=" * 78)
print("§6 原地复用：同 key 重复 assemble 不重复加行")
print("=" * 78)
a = rng.uniform(0, 100, size=4)
b = rng.normal(0, 0.02, size=(4, Ly.dim))
sub.assemble((a, b), key=("rep", 1))
n1 = n_rows(sub)
sub.assemble((a, b), key=("rep", 1))
sub.assemble((a, b), key=("rep", 1))
print(f"  同 key 调 3 次，行数 {n1} → {n_rows(sub)}")
check("同 key 不重复加行", n_rows(sub), n1)

print()
print("=" * 78)
print("汇总")
print("=" * 78)
n_fail = sum(1 for r in ROWS if r[1] == "FAIL")
for name, st, got, want, note in ROWS:
    line = f"  [{st}] {name}"
    if st == "FAIL":
        line += f"  got={got} 期望={want} {note}"
    print(line)
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
