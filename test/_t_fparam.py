# 验收：F 参数化（Fdef: F − θ·n_in = 0）之后，割是否成为合法下界。
#
# 直接用**生产模型** StageSubproblem（不做任何手工改矩阵），在真实试探点（割的锚点）
# 上沿月龄方向扫描，比较真值 Q(k) 与割的直线外推 c(k)：
#   min(Q−c) < 0 ⇒ 割越界（下界不成立）；≥ 0 ⇒ 合法。
#
# 对照：旧口径见 test/_cvx_rint.log（rint/ceil/cont 三口径全部 ★越界）。
#   python test/_t_fparam.py [结果目录] [阶段t ...]
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

DEFAULT_DIR = "results/base_20261010_151432"
DEFAULT_STAGES = [199, 100, 40]


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
    print(f"结果目录 {d.name}   T={prob.T}   场景 {si}\n")

    all_ok = True
    for t in stages:
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        sub = StageSubproblem(t, prob, scfg)
        sub.assemble(cn, key=f"tfp{t}")

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

        def ev(x):
            objs, grads = [], []
            for j in si:
                sub.set_point(x, prob.pool[j])
                sub.solve_lp()
                objs.append(sub.obj)
                grads.append(sub.gradient("dj"))
            return float(np.mean(objs)), np.mean(grads, axis=0)

        Q0, g0 = ev(xb)
        rows = []
        for k in ks:
            x = xb.copy()
            x[ia] = xb[ia] + k
            try:
                Q, _ = ev(x)
            except Exception:
                break
            rows.append((int(k), Q, Q - (Q0 + float(g0 @ (x - xb)))))
        worst = min(r[2] for r in rows)
        ok = worst >= -1e-6 * max(1.0, abs(Q0))
        all_ok &= ok
        print(f"阶段 {t:>3}  锚点 it={best.it}  桶 ({r0},{a0}) n={xb[ia]:.1f} θ̄={th0:.5f}  "
              f"k∈[{rows[0][0]},{rows[-1][0]}]  Q(锚点)={Q0:,.2f}")
        print(f"         min(Q−c) = {worst:>16,.4f}   {'合法' if ok else '★ 越界'}")
        print("      k:  " + " ".join(f"{r[0]:>11d}" for r in rows))
        print("   Q−c:  " + " ".join(f"{r[2]:>11,.0f}" for r in rows))
        print()
        sub.free()

    print("=" * 66)
    print("总判定：" + ("全部合法 ✓" if all_ok else "★ 仍有越界 ✗"))


if __name__ == "__main__":
    main()
