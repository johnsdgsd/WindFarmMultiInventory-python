# 根因确认 + 修法验证。
#
# 诊断结论：set_scene 把 F = rint(θ·n(x)) **算成数字写进约束右端**，于是 θ 对状态的
# 依赖只存在于「右端常数」里 —— 而 .dj（既约成本）只反映**变量界**那条通道，
# ∂F/∂x 整条通道对偶看不见。割的斜率因此少一项 Σ ν·∂F/∂x（实测 ν = ∂Q/∂F ≈ −2000，
# θ·ν 相对 .dj 达 5%）。这与 rint 无关：连续 θn 口径同样越界（见 _cvx_rint.log）。
#
# Julia 原版没这个问题：它把 (1−θ) 写成**状态变量的系数**
#（docs/建模与实现.md (4)(8)：dem 系数 +1→(1−θ)，trans 系数 −1→−(1−θ)）。
#
# 本脚本做代数恒等的改写（只留取整残差在右端）：
#   dem_r  : Σy + Σ(1−θ)n_in ≤ N_r + Σ(rint(θn) − θn)
#   age_r,a: n_out[a+1] − (1−θ)n_in = −(rint(θn) − θn)
# 两式各自展开后与生产口径**完全等价** —— 所以先断言 Q 逐位不变，再看割是否合法。
#
#   python test/_cvx_fix.py [结果目录] [阶段t ...]
import sys
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

DEFAULT_DIR = "results/base_20261010_151432"
DEFAULT_STAGES = [100, 40, 199]


def set_rhs_form(sub, th):
    """生产口径：F 整体进右端（与 code/subproblem.set_scene 逐字一致）。"""
    Ly, prob, m = sub.layout, sub.prob_data, sub.model
    F = np.rint(th * Ly.n(sub._state_vec))
    m.setInfo(COPT.Info.UB, sub.dem_rows,
              [float(prob.N[r]) + float(F[r].sum()) for r in range(Ly.N_R)])
    rhs = [float(t) for t in (-F[:, :Ly.A - 1]).ravel()]
    m.setInfo(COPT.Info.LB, sub.age_rows, rhs)
    m.setInfo(COPT.Info.UB, sub.age_rows, rhs)


def set_coef_form(sub, th):
    """系数口径：θ 作为状态变量的系数，只有取整残差进右端。与 set_rhs_form 代数恒等。"""
    Ly, prob, m = sub.layout, sub.prob_data, sub.model
    prod = th * Ly.n(sub._state_vec)
    res = np.rint(prod) - prod                    # 取整残差：唯一留在右端的量
    for r in range(Ly.N_R):
        for a in range(Ly.A):
            m.setCoeff(sub.dem_rows[r], sub.x_in[Ly.ix_n(r, a)], 1.0 - th[r, a])
        m.setInfo(COPT.Info.UB, [sub.dem_rows[r]],
                  [float(prob.N[r] + res[r].sum())])
    k = 0
    for r in range(Ly.N_R):
        for a in range(Ly.A - 1):
            row = sub.age_rows[k]
            m.setCoeff(row, sub.x_in[Ly.ix_n(r, a)], -(1.0 - th[r, a]))
            m.setInfo(COPT.Info.LB, [row], [float(-res[r, a])])
            m.setInfo(COPT.Info.UB, [row], [float(-res[r, a])])
            k += 1


def eval_point(sub, x, thetas, setter):
    objs, grads = [], []
    for th in thetas:
        sub.set_state(x)
        setter(sub, th)
        sub.solve_lp()
        objs.append(sub.obj)
        grads.append(sub.gradient("dj"))
    return float(np.mean(objs)), np.mean(grads, axis=0)


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / DEFAULT_DIR
    if not d.is_absolute():
        d = ROOT / d
    stages = [int(v) for v in sys.argv[2:]] or DEFAULT_STAGES

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    cuts, _ = load_cuts(d, prob.T, prob.state_dim)

    rng = np.random.default_rng(7)
    si = sorted(rng.choice(len(prob.pool), size=4, replace=False).tolist())
    thetas = [prob.pool[i][1] for i in si]
    print(f"结果目录 {d.name}   T={prob.T}   诊断场景 {si}\n")

    for t in stages:
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        subs = {}
        for tag in ("rhs", "coef"):
            s = StageSubproblem(t, prob, scfg)
            s.assemble(cn, key=f"{tag}{t}")
            subs[tag] = s
        best, bn = None, -1.0
        for c in cuts[t]:
            if c.it == 0:
                continue
            v = float(Ly.n(c.state).sum())
            if v > bn:
                best, bn = c, v
        xb = np.asarray(best.state, dtype=float)
        nb = Ly.n(xb)
        prod = np.mean([th * nb for th in thetas], axis=0)
        prod = np.where(nb > 0.25 * nb.max(), prod, -np.inf)
        r0, a0 = np.unravel_index(np.argmax(prod), prod.shape)
        ia = Ly.ix_n(r0, a0)
        th0 = float(np.mean([th[r0, a0] for th in thetas]))
        span = max(1.0, 2.0 / max(th0, 1e-9))
        ks = np.unique(np.round(np.linspace(-span, span, 17)).astype(int))
        ks = ks[(xb[ia] + ks >= 0) & (xb[ia] + ks <= Ly.upper(prob)[ia])]

        print("=" * 78)
        print(f"阶段 {t}   锚点 it={best.it}   扫描桶 ({r0},{a0})  n={xb[ia]:.1f}  "
              f"θ̄={th0:.5f}  范围 ±{span:.0f}")
        print("=" * 78)
        # 锚点：两种口径的 Q 与 .dj
        Q_r, g_r = eval_point(subs["rhs"], xb, thetas, set_rhs_form)
        Q_c, g_c = eval_point(subs["coef"], xb, thetas, set_coef_form)
        print(f"  Q(锚点)  rhs={Q_r:,.4f}   coef={Q_c:,.4f}   "
              f"|差| {abs(Q_r - Q_c):.3e}   ← 应为 0（代数恒等）")
        print(f"  .dj[{r0},{a0}]  rhs={g_r[ia]:,.4f}   coef={g_c[ia]:,.4f}   "
              f"差 {g_c[ia]-g_r[ia]:,.4f}  （这就是漏掉的 θ·ν）")

        rows = []
        for k in ks:
            x = xb.copy()
            x[ia] = xb[ia] + k
            try:
                Qr, _ = eval_point(subs["rhs"], x, thetas, set_rhs_form)
                Qc, _ = eval_point(subs["coef"], x, thetas, set_coef_form)
            except Exception:
                break
            cr = Q_r + float(g_r @ (x - xb))
            cc = Q_c + float(g_c @ (x - xb))
            rows.append((int(k), Qr, cr, Qr - cr, Qc, cc, Qc - cc))
        w_r = min(r[3] for r in rows)
        w_c = min(r[6] for r in rows)
        idmax = max(abs(r[1] - r[4]) for r in rows)
        print(f"  模型等价性：max|Q_rhs − Q_coef| = {idmax:.3e}   ← 应为 0")
        print(f"  割合法性  rhs 口径 min(Q−c) = {w_r:>14,.2f}  "
              f"{'★ 越界' if w_r < -1e-6 * max(1.0, abs(Q_r)) else '合法'}")
        print(f"            coef 口径 min(Q−c) = {w_c:>14,.2f}  "
              f"{'★ 越界' if w_c < -1e-6 * max(1.0, abs(Q_c)) else '合法'}")
        print("        k:  " + " ".join(f"{r[0]:>11d}" for r in rows))
        print("     Q−c场:  " + " ".join(f"{r[3]:>11,.0f}" for r in rows))
        print("     Q−c系:  " + " ".join(f"{r[6]:>11,.0f}" for r in rows))
        print()
        for s in subs.values():
            s.free()


if __name__ == "__main__":
    main()
