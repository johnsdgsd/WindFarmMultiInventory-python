# 建模等价性验收（重构的硬门槛）：同一 (期, 状态, 场景, 割集) 下，
# 旧 pulp 子问题与新 coptpy 子问题必须给出逐位一致的目标值与对偶。
#   python test/_t_equiv.py
#
# 比对四项：obj、.dj 梯度、通用式梯度 c_s+Σa_c·π_c、逐约束 π。
# 任一项不一致就说明重构引入了偏差，必须先定位再往下走。

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import StageSubproblem                           # noqa: E402
import _pulp_ref_model as pulp_model                                  # noqa: E402
from utils.cut import Cut                                             # noqa: E402

TOL = 1e-9


def make_solver(mip: bool, msg: bool = False):
    """旧路径的求解器（pulp COPT_DLL / CBC 兜底）——只有对照件还用得上。"""
    import contextlib
    import io
    import pulp

    if "COPT_DLL" not in pulp.listSolvers(onlyAvailable=True):
        return pulp.PULP_CBC_CMD(msg=msg, mip=mip)
    with contextlib.redirect_stdout(io.StringIO()):
        return pulp.COPT_DLL(msg=msg, mip=mip)

cfg_p = load_problem_config(ROOT / "config/smoke/problem.toml")
cfg_s = load_solver_config(ROOT / "config/smoke/solver.toml")
prob = DefineProblem(cfg_p)
Ly = prob.layout
LP = make_solver(mip=False, msg=False)

ROWS = []


def check(name, got, want, tol=TOL):
    g, w = np.asarray(got, float), np.asarray(want, float)
    if g.shape != w.shape:
        ROWS.append((name, "FAIL", f"shape {g.shape}", f"shape {w.shape}", ""))
        return False
    scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
    abserr = float(np.abs(g - w).max()) if g.size else 0.0
    err = abserr / scale
    ok = err <= tol
    ROWS.append((name, "PASS" if ok else "FAIL", f"{abserr:.3e}", f"abs ≤ {tol:g}×{scale:g}",
                 f"rel={err:.2e}"))
    return ok


def probe_state(rng):
    """按约束构造可行探测点：月龄用多项分布切分 N_r（逐分量随机取整几乎必然越界）。"""
    s = np.zeros(Ly.dim)
    for r in range(prob.N_R):
        w = rng.random(Ly.A) + 1e-6
        cnt = rng.multinomial(int(prob.N[r]), w / w.sum())
        s[Ly.ix_n(r, 0):Ly.ix_n(r, 0) + Ly.A] = cnt
    ub = Ly.upper(prob)
    for j in list(range(Ly.off_I, Ly.off_I + Ly.n_I)) + \
            list(range(Ly.off_o, Ly.off_o + Ly.n_o)) + \
            list(range(Ly.off_q, Ly.off_q + Ly.n_q)):
        s[j] = float(np.rint(rng.random() * 0.5 * ub[j]))
    return s


def make_cuts(n, dim, rng):
    """造一批随机割（合法与否不重要——两条实现面对同一割集必须同解）。"""
    cuts = [Cut(0.0, np.zeros(dim), np.zeros(dim), 1, 0.0, 0)]     # 平凡割
    for i in range(n):
        beta = rng.normal(0, 0.05, size=dim)
        beta[rng.random(dim) < 0.9] = 0.0                          # 稀疏
        cuts.append(Cut(float(rng.uniform(0, 50)), beta, np.zeros(dim), 1, 0.0, i))
    return cuts


def pack_cuts(cuts, dim):
    return (np.array([c.alpha for c in cuts]),
            np.vstack([c.beta for c in cuts]) if cuts else np.zeros((0, dim)))


rng = np.random.default_rng(20261010)

print("=" * 78)
print("§1 新 coptpy 子问题自洽（先确保它能跑）")
print("=" * 78)
x0 = probe_state(rng)
sc0 = prob.pool[3]
sub = StageSubproblem(2, prob, cfg_s)
sub.assemble(None)
sub.set_point(x0, sc0)
sub.solve_lp()
print(f"  t=2 无割：obj={sub.obj:.6f}  梯度非零 {int((np.abs(sub.gradient())>1e-9).sum())} 维")

print()
print("=" * 78)
print("§2 obj / 梯度 / 对偶 逐位对拍（pulp vs coptpy）")
print("=" * 78)

for label, t, n_cuts in [("阶段 2 无割", 2, 0),
                         ("阶段 2 带 5 条割", 2, 5),
                         ("阶段 1 带 20 条割", 1, 20),
                         ("末阶段 T=4", prob.T, 0)]:
    x = probe_state(rng)
    scene = prob.pool[int(rng.integers(0, len(prob.pool)))]
    cuts = make_cuts(n_cuts, Ly.dim, rng)
    packed = pack_cuts(cuts, Ly.dim) if n_cuts else None

    old = pulp_model.build_stage_subproblem(t, x, scene, packed, prob, 0)
    old.solve(LP)

    new = StageSubproblem(t, prob, cfg_s)
    new.assemble(packed)
    new.set_point(x, scene)
    new.solve_lp()

    print(f"\n  [{label}]  pulp obj={old.obj:.10f}  coptpy obj={new.obj:.10f}")
    check(f"{label} · obj", new.obj, old.obj)

    g_old, g_new = old.gradient("dj"), new.gradient("dj")
    nz_old = int((np.abs(g_old) > 1e-9).sum())
    nz_new = int((np.abs(g_new) > 1e-9).sum())
    print(f"    梯度非零维数  pulp {nz_old}  coptpy {nz_new}")
    check(f"{label} · dj 梯度", g_new, g_old)

    d_old, d_new = old.gradient("duals"), new.gradient("duals")
    check(f"{label} · 通用式梯度", d_new, d_old)

    # 逐约束 π：两边的约束名集合应当一致
    names_old = set(old.state_coeff().keys())
    names_new = set(new.state_coeff().keys())
    check(f"{label} · state_coeff 键集", sorted(names_new) == sorted(names_old),
          True)
    if names_new == names_old:
        bad = []
        for nm in sorted(names_old):
            po = float(old.prob.constraints[nm].pi)
            pn = float(new.model.getInfo("Dual", [new._row(nm)])[0])
            if abs(po - pn) > 1e-8 * max(1.0, abs(po)):
                bad.append((nm, po, pn))
        check(f"{label} · 逐约束 π（{len(names_old)} 条）", len(bad), 0,
              tol=0)
        if bad:
            for nm, po, pn in bad[:5]:
                print(f"      π 不一致 {nm}: pulp={po:.10f} coptpy={pn:.10f}")

print()
print("=" * 78)
print("§3 虚拟首阶段（t=0）")
print("=" * 78)
x = probe_state(rng)
cuts0 = make_cuts(4, Ly.dim, rng)
packed0 = pack_cuts(cuts0, Ly.dim)
old0 = pulp_model.build_stage_subproblem(0, x, None, packed0, prob, 0)
old0.solve(LP)
new0 = StageSubproblem(0, prob, cfg_s)
new0.assemble(packed0)
new0.set_state(x)
new0.solve_lp()
print(f"  pulp obj={old0.obj:.10f}  coptpy obj={new0.obj:.10f}")
check("t=0 · obj", new0.obj, old0.obj)
check("t=0 · 期末状态", new0.out_state_vector(), old0.out_state_vector())
# 旧 pulp 路径在 t=0 取不出 .dj（x_in 全被界钉死，presolve 消掉后不回值）；
# 新 coptpy 路径能取到。是能力增强而非回退——lower_bound 只读 obj，不受影响。
try:
    g_old0 = old0.gradient("dj")
    check("t=0 · dj 梯度", new0.gradient("dj"), g_old0)
except RuntimeError as exc:
    g_new0 = new0.gradient("dj")
    print(f"  注：旧 pulp 路径 t=0 取不出 .dj（{exc}）；"
          f"新路径可读，非零维 {int((np.abs(g_new0) > 1e-9).sum())}")
    ROWS.append(("t=0 · dj 梯度", "N/A（旧路径不可用）", "—", "—",
                 "新路径能力更强"))

print()
print("=" * 78)
print("§4 前向 MIP 一致性（整数解与阶段成本）")
print("=" * 78)
x = probe_state(rng)
scene = prob.pool[7]
MIP = make_solver(mip=True, msg=False)
oldm = pulp_model.build_stage_subproblem(2, x, scene, None, prob, 0)
oldm.solve(MIP)
newm = StageSubproblem(2, prob, cfg_s)
newm.assemble(None)
newm.set_point(x, scene)
newm.solve_mip()
print(f"  pulp obj={oldm.obj:.10f}  coptpy obj={newm.obj:.10f}")
check("MIP · obj", newm.obj, oldm.obj)
check("MIP · 期末状态", newm.out_state_vector(), oldm.out_state_vector())
check("MIP · 阶段成本", newm.stage_cost(), oldm.stage_cost())

print()
print("=" * 78)
print("§5 原地复用：同一个模型改点重解，结果与新建模型一致")
print("=" * 78)
sub = StageSubproblem(2, prob, cfg_s)
for k in range(3):
    x = probe_state(rng)
    scene = prob.pool[int(rng.integers(0, len(prob.pool)))]
    cuts = make_cuts(6, Ly.dim, rng)
    packed = pack_cuts(cuts, Ly.dim)
    ref = StageSubproblem(2, prob, cfg_s)
    ref.assemble(packed)
    ref.set_point(x, scene)
    ref.solve_lp()
    sub.assemble(packed, key=(k, 2, "test"))
    sub.set_point(x, scene)
    sub.solve_lp()
    check(f"复用第 {k} 次 · obj", sub.obj, ref.obj)
    check(f"复用第 {k} 次 · dj", sub.gradient("dj"), ref.gradient("dj"))
    check(f"复用第 {k} 次 · out_state", sub.out_state_vector(),
          ref.out_state_vector())

print()
print("=" * 78)
print("汇总")
print("=" * 78)
wd = max(len(r[0]) for r in ROWS)
n_fail = 0
for name, st, got, want, note in ROWS:
    n_fail += st == "FAIL"
    flag = "" if st == "PASS" else "   <<<"
    if st == "FAIL":
        print(f"  [{st}] {name:<{wd}}  got={got}  期望={want}  {note}{flag}")
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
