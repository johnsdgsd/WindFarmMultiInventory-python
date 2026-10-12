# 诊断 LB 恒为 0：看割集本身有没有信息。
#
#   python test/_t_lb0.py <结果目录>
#
# 打印：逐期的割数 / alpha 分布 / beta 范数；阶段 0 复现 LB 并指出绑定割。
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


def main():
    d = Path(sys.argv[1])
    d = d if d.is_absolute() else ROOT / d
    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    print(f"{d.name}   轮次 {meta.get('n_iterations')}   总割数 {meta.get('n_cuts')}   "
          f"T={prob.T}  状态维 {Ly.dim}")

    print(f"\n{'期':>4} {'割数':>5} {'alpha最小':>14} {'alpha中位':>14} {'alpha最大':>14} "
          f"{'|beta|中位':>11} {'非平凡割':>8}")
    for t in range(1, prob.T + 1):
        cs = cuts[t]
        if not cs:
            continue
        al = np.array([c.alpha for c in cs])
        nb = np.array([np.linalg.norm(c.beta) for c in cs])
        nz = int((nb > 1e-9).sum())
        tg = [1, prob.T // 4, prob.T // 2, 3 * prob.T // 4, prob.T]
        if t in tg:
            print(f"{t:>4} {len(cs):>5} {al.min():>14,.2f} {np.median(al):>14,.2f} "
                  f"{al.max():>14,.2f} {np.median(nb):>11,.2f} {nz:>8}")

    # ★ 逐期的 obj = Q_t(x̄)：割在自己的锚点上的值。信息传没传回来就看它。
    print(f"\n  ★ 逐期【非平凡割的 obj 最大值】= 该期试探点上能达到的阶段值：")
    for k in range(0, prob.T, 6):
        seg = []
        for t in range(k + 1, min(k + 7, prob.T + 1)):
            ob = [c.obj for c in cuts[t] if c.it > 0]
            seg.append(max(ob) if ob else float("nan"))
        print(f"    期 {k+1:>3}-{k+len(seg):<3}: " +
              " ".join(f"{v:>12,.1f}" if v == v else f"{'—':>12}" for v in seg))

    print(f"\n  逐期 alpha 最大值：")
    row = []
    for t in range(1, prob.T + 1):
        row.append(max([c.alpha for c in cuts[t]], default=0.0))
    for k in range(0, prob.T, 10):
        seg = row[k:k + 10]
        print(f"    期 {k+1:>3}-{k+len(seg):<3}: " + " ".join(f"{v:>10,.1f}" for v in seg))

    # ---- 复现阶段 0 的 LB ----
    x0 = prob.initial_state()
    A1 = np.array([c.alpha for c in cuts[1]])
    B1 = np.vstack([c.beta for c in cuts[1]])
    s0 = StageSubproblem(0, prob, scfg)
    s0.assemble((A1, B1))
    s0.set_state(x0)
    s0.solve_lp()
    xs = s0.out_state_vector()
    print(f"\n  LB = {s0.obj:,.4f}   theta* = {s0.theta_value():,.4f}   "
          f"q_out* = {np.round(Ly.q(xs), 3).tolist()}   Qmax = {prob.Qmax.tolist()}")
    vals = np.array([c.value_at(xs) for c in cuts[1]])
    o = np.argsort(-vals)
    print(f"  阶段 1 的 {len(cuts[1])} 条割在 q* 处取值："
          f"最大 {vals[o][0]:,.4f}  第2 {vals[min(1,len(o)-1)]:,.4f}  "
          f"中位 {np.median(vals):,.4f}  最小 {vals.min():,.4f}")
    print(f"  前 5 名的轮次/alpha/在自锚点的值：")
    for i in o[:5]:
        c = cuts[1][int(i)]
        print(f"    轮 {c.it:>4}  alpha={c.alpha:>12,.2f}  在q*处={c.value_at(xs):>12,.4f}  "
              f"在自锚点={c.value_at(c.state):>12,.4f}  |beta|={np.linalg.norm(c.beta):>9,.3f}")
    s0.free()


if __name__ == "__main__":
    main()
