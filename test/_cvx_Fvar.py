# F 用「变量 + 约束」参数化，两种钉法差别在哪。
#
# 用户提议：把失效量 F 建成模型里的变量，用约束固定住。方向对，但钉法决定成败：
#
#   A「界钉」：F 是变量，但只用变量界把它钉成 rint(θ·n̄) 这个**数字**。
#             ⇒ 等价于现状：n 没出现在任何含 F 的行里，.dj 一模一样，割照样越界。
#   B「式钉」：F 是变量，用**线性等式** F − θ_{r,a}·n_in[r,a] = res 定义它
#             （res = rint(θn) − θn 是取整残差，钉成常数）。
#             ⇒ n 以系数 −θ 进矩阵，.dj 自动带回θν，割恢复合法。
#
# 三口径对比（同一锚点、同一批场景）：
#   rhs  = 生产现状（F 算成数字写右端）
#   fvar = A：F 变量 + 界钉
#   fdef = B：F 变量 + 线性式钉
#
#   python test/_cvx_Fvar.py [结果目录] [阶段t ...]
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
DEFAULT_STAGES = [40, 100]


def add_F_block(sub):
    """给已建好的子问题加 F 变量，并把 dem/age 行改写成含 F 的形式：
         dem_r  : Σy + Σn_in − Σ_a F[r,a] ≤ N_r          （原来靠右端塞 N+ΣF）
         age_r,a: n_out[a+1] − n_in[a] + F[r,a] = 0      （原来靠右端塞 −F）
    """
    Ly, prob, m = sub.layout, sub.prob_data, sub.model
    Fv = [[m.addVar(lb=-COPT.INFINITY, ub=COPT.INFINITY, name=f"Fv_{r}_{a}")
           for a in range(Ly.A)] for r in range(Ly.N_R)]
    for r in range(Ly.N_R):
        m.setInfo(COPT.Info.UB, [sub.dem_rows[r]], [float(prob.N[r])])
        for a in range(Ly.A):
            m.setCoeff(sub.dem_rows[r], Fv[r][a], -1.0)
    k = 0
    for r in range(Ly.N_R):
        for a in range(Ly.A - 1):
            row = sub.age_rows[k]
            m.setInfo(COPT.Info.LB, [row], [0.0])
            m.setInfo(COPT.Info.UB, [row], [0.0])
            m.setCoeff(row, Fv[r][a], 1.0)
            k += 1
    return Fv


def add_F_def_rows(sub, Fv):
    """B 口径专用：给每个 F 变量加一条定义式 F − θ·n_in = res（θ 与 res 逐场景改）。"""
    Ly, m = sub.layout, sub.model
    rows = [[m.addConstr(Fv[r][a] == 0.0, name=f"Fdef_{r}_{a}")
             for a in range(Ly.A)] for r in range(Ly.N_R)]
    return rows


def set_rhs(sub, Fv, rows, th):
    Ly, prob, m = sub.layout, sub.prob_data, sub.model
    F = np.rint(th * Ly.n(sub._state_vec))
    m.setInfo(COPT.Info.UB, sub.dem_rows,
              [float(prob.N[r]) + float(F[r].sum()) for r in range(Ly.N_R)])
    rhs = [float(t) for t in (-F[:, :Ly.A - 1]).ravel()]
    m.setInfo(COPT.Info.LB, sub.age_rows, rhs)
    m.setInfo(COPT.Info.UB, sub.age_rows, rhs)


def set_fvar(sub, Fv, rows, th):
    """A：只用变量界把 F 钉成数字（内容与 set_rhs 写的数字完全相同）。"""
    Ly, m = sub.layout, sub.model
    F = np.rint(th * Ly.n(sub._state_vec))
    for r in range(Ly.N_R):
        for a in range(Ly.A):
            m.setInfo(COPT.Info.LB, [Fv[r][a]], [float(F[r, a])])
            m.setInfo(COPT.Info.UB, [Fv[r][a]], [float(F[r, a])])


def set_fdef(sub, Fv, rows, th):
    """B：F 由线性式定义，θ 作为 n_in 的系数进矩阵。"""
    Ly, m = sub.layout, sub.model
    prod = th * Ly.n(sub._state_vec)
    res = np.rint(prod) - prod                    # 只有取整残差留在右端
    for r in range(Ly.N_R):
        for a in range(Ly.A):
            m.setCoeff(rows[r][a], sub.x_in[Ly.ix_n(r, a)], -th[r, a])
            m.setInfo(COPT.Info.LB, [rows[r][a]], [float(res[r, a])])
            m.setInfo(COPT.Info.UB, [rows[r][a]], [float(res[r, a])])


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

    for t in stages:
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        # rhs 用原模型；fvar/fdef 用加了 F 变量块的模型
        s_rhs = StageSubproblem(t, prob, scfg); s_rhs.assemble(cn, key="rhs")
        s_fv = StageSubproblem(t, prob, scfg);  s_fv.assemble(cn, key="fv")
        FvA = add_F_block(s_fv)
        s_fd = StageSubproblem(t, prob, scfg);  s_fd.assemble(cn, key="fd")
        FvB = add_F_block(s_fd)
        rowsB = add_F_def_rows(s_fd, FvB)

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

        print("=" * 78)
        print(f"阶段 {t}   锚点 it={best.it}   扫描桶 ({r0},{a0})  "
              f"n={xb[ia]:.1f}  θ̄={th0:.5f}")
        print("=" * 78)

        panels = {"rhs": (s_rhs, None, None, set_rhs),
                  "fvar": (s_fv, FvA, None, set_fvar),
                  "fdef": (s_fd, FvB, rowsB, set_fdef)}
        res = {}
        for tag, (sub, Fv, rows, setter) in panels.items():
            objs, grads = [], []
            for th in thetas:
                sub.set_state(xb)
                setter(sub, Fv, rows, th)
                sub.solve_lp()
                objs.append(sub.obj)
                grads.append(sub.gradient("dj"))
            res[tag] = (float(np.mean(objs)), np.mean(grads, axis=0))

        Qr = res["rhs"][0]
        for tag in ("rhs", "fvar", "fdef"):
            Q, g = res[tag]
            print(f"  [{tag:>4}]  Q = {Q:>18,.6f}   "
                  f"|Q−Q_rhs| = {abs(Q-Qr):.3e}   .dj[{r0},{a0}] = {g[ia]:>15,.4f}")

        span = max(1.0, 2.0 / max(th0, 1e-9))
        ks = np.unique(np.round(np.linspace(-span, span, 17)).astype(int))
        ks = ks[(xb[ia] + ks >= 0) & (xb[ia] + ks <= Ly.upper(prob)[ia])]
        wr = {}
        for tag, (sub, Fv, rows, setter) in panels.items():
            Q0, g0 = res[tag]
            worst = None
            for k in ks:
                x = xb.copy()
                x[ia] = xb[ia] + k
                objs = []
                try:
                    for th in thetas:
                        sub.set_state(x)
                        setter(sub, Fv, rows, th)
                        sub.solve_lp()
                        objs.append(sub.obj)
                except Exception:
                    break
                Qk = float(np.mean(objs))
                v = Qk - (Q0 + float(g0 @ (x - xb)))
                worst = v if worst is None else min(worst, v)
            wr[tag] = worst
        print(f"\n  割合法性 min(Q−c) 扫描（k = {ks.min()} .. {ks.max()}，共 {len(ks)} 点）")
        for tag in ("rhs", "fvar", "fdef"):
            v = wr[tag]
            if v is None:
                print(f"    [{tag:>4}] —（扫描全不可行）")
            else:
                bad = v < -1e-6 * max(1.0, abs(Qr))
                print(f"    [{tag:>4}] {v:>16,.2f}   {'★ 越界' if bad else '合法'}")
        print()
        for s in (s_rhs, s_fv, s_fd):
            s.free()


if __name__ == "__main__":
    main()
