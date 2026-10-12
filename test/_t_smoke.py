# 对偶自检四项：.dj vs 通用式、次梯度不等式、割合法性、平凡割合法性。
#   python test/_t_smoke.py
#
# 判据是**次梯度不等式**而不是中心差分：折点处中心差分给的是左右斜率均值，它只是
# 一个合法次梯度，判不出真伪（见 utils/diagnostics.py 顶部说明）。

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils import Cut, trivial_cut                                # noqa: E402
from utils import diagnostics as dg                               # noqa: E402

prob = DefineProblem(load_problem_config(ROOT / "config/smoke/problem.toml"))
scfg = load_solver_config(ROOT / "config/smoke/solver.toml")
Ly = prob.layout
rng = np.random.default_rng(4242)

ROWS = []


def check(name, ok, note=""):
    ROWS.append((name, "PASS" if ok else "FAIL", note))


x0 = dg.probe_states(rng, prob, n=1)[0]
scene = prob.pool[5]

print("=" * 78)
print("§1 无割：.dj 与通用式 c_s + Σa_c·π_c 交叉验证")
print("=" * 78)
r = dg.dual_crosscheck(make_stage_subproblem, prob, scfg, 2, x0, scene, None)
print(f"  Q = {r['Q']:.6f}  最大相对偏差 = {r['max_rel_err']:.3e}  "
      f"非零梯度维数 = {r['n_nonzero_beta']}")
check("无割交叉验证", r["ok"], f"rel={r['max_rel_err']:.2e}")

print()
print("=" * 78)
print("§2 带割：交叉验证")
print("=" * 78)
cuts = (np.array([0.0]), np.zeros((1, Ly.dim)))
r2 = dg.dual_crosscheck(make_stage_subproblem, prob, scfg, 2, x0, scene, cuts)
print(f"  Q = {r2['Q']:.6f}  最大相对偏差 = {r2['max_rel_err']:.3e}")
check("带割交叉验证", r2["ok"], f"rel={r2['max_rel_err']:.2e}")

print()
print("=" * 78)
print("§3 次梯度不等式（逐分量左右差商夹逼）")
print("=" * 78)
comps = [Ly.ix_I(0), Ly.ix_n(0, 3), Ly.ix_q(1), Ly.ix_o(0, 2)]
g = dg.subgradient_inequality(make_stage_subproblem, prob, scfg, 2, x0, scene,
                              None, comps, h=1e-3)
for row in g["rows"]:
    fmt = lambda v: "None" if v is None else f"{v:12.6f}"     # noqa: E731
    print(f"  分量 {row['j']:3d}  β={row['beta']:12.6f}  "
          f"左={fmt(row['left'])}  右={fmt(row['right'])}  "
          f"{'OK' if row['ok'] else 'BAD'} {row['why']}")
check("次梯度不等式", g["ok"], f"{g['n_checked']} 检查 / {g['n_skipped']} 跳过")

print()
print("=" * 78)
print("§4 割的全局合法性：α + β·(s − x̄) ≤ Q(s)")
print("=" * 78)
sub = make_stage_subproblem(2, prob, scfg)
sub.assemble(None)
sub.set_point(x0, scene)
sub.solve_lp()
beta = sub.gradient("dj").copy()
cut = Cut(sub.obj - float(beta @ x0), beta, x0.copy(), 2, sub.obj, 0)
probes = dg.probe_states(rng, prob, n=4, scale=0.8)
cv = dg.cut_validity(make_stage_subproblem, prob, scfg, cut, probes, None, scene)
print(f"  探测 4 点，最小间隙 = {cv['min_slack']:.6f}（显著为负即割越界）")
check("割合法性", cv["ok"], f"min_slack={cv['min_slack']:.3e}")

print()
print("=" * 78)
print("§5 平凡割 θ ≥ 0 合法（目标非负 ⇒ V ≥ 0）")
print("=" * 78)
tc = trivial_cut(2, x0, Ly.dim)
cv2 = dg.cut_validity(make_stage_subproblem, prob, scfg, tc, probes, None, scene)
print(f"  最小间隙 = {cv2['min_slack']:.6f}")
check("平凡割合法", cv2["ok"], f"min_slack={cv2['min_slack']:.3e}")

print()
print("=" * 78)
print("汇总")
print("=" * 78)
n_fail = sum(1 for r_ in ROWS if r_[1] == "FAIL")
for name, st, note in ROWS:
    print(f"  [{st}] {name}   {note}")
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
