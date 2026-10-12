# 负 gap 排查：从 LB 的绑定点入手。
#   python test/_diag_lb.py [结果目录] [阶段2路径数K]
#
# 阶段 1：读回 cuts.json，复现 LB，找出是**哪条割**在顶住 theta、此时 q_out* 是多少，
#         并把 q_out* 与阶段 1 试探点的 q 分布对照（落在试探点没去过的区间 = 在外推）。
# 阶段 2：在 x* 这一点上判定 c* 是否越界——从 x* 出发按引擎自己的策略跑 K 条路径，
#         可行成本均值 C(x*) 是 V_1(x*) 的上界；c*(x*) > C(x*) ⇒ 割越界、根因锁定。
#
# 只读现有 artifacts，不改任何模型代码（rint 也不动）。
import json
import sys
import time
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


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / DEFAULT_DIR
    if not d.is_absolute():
        d = ROOT / d
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    x0 = prob.initial_state()
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    print(f"结果目录 {d.name}   轮次 {meta.get('n_iterations')}   "
          f"割数 {sum(len(c) for c in cuts)}   T={prob.T} 状态维 {Ly.dim}")

    # ---------------------------------------------------------- 阶段 1
    print("\n" + "=" * 78)
    print("阶段 1：解剖 LB 最优解")
    print("=" * 78)
    c1 = cuts[1]
    A1 = np.array([c.alpha for c in c1])
    B1 = np.vstack([c.beta for c in c1])
    s0 = StageSubproblem(0, prob, scfg)
    s0.assemble((A1, B1))
    s0.set_state(x0)
    s0.solve_lp()
    LB = s0.obj
    theta = s0.theta_value()
    xs = s0.out_state_vector()                 # = 初始状态（非 q 槽钉死）+ q_out*
    qstar = Ly.q(xs)
    print(f"  LB = {LB:,.4f}   theta = {theta:,.4f}   期 1 割数 {len(c1)}")
    print(f"  q_out* = {np.round(qstar, 3).tolist()}   Qmax = {prob.Qmax.tolist()}   "
          f"Imax = {prob.Imax.tolist()}")

    # 非 q 槽是否真的钉死在 x0
    keep = np.ones(Ly.dim, bool)
    keep[Ly.off_q:Ly.off_q + Ly.n_q] = False
    print(f"  非 q 槽 == 初始状态: {np.allclose(xs[keep], x0[keep])}"
          f"（应为 True）")

    # 各割在 x* 处的取值，排序
    vals = np.array([c.value_at(xs) for c in c1])
    order = np.argsort(-vals)
    print(f"\n  阶段 1 各割在 x* 处取值（降序前 6 / 后 2）：")
    for i in list(order[:6]) + list(order[-2:]):
        c = c1[i]
        tag = "  ← 绑定" if abs(vals[i] - theta) <= 1e-6 * max(1.0, abs(theta)) else ""
        own = c.value_at(c.state)
        print(f"    割#{i:<4} 轮 {c.it:<4} 值 {vals[i]:>18,.2f}   "
              f"在自锚点 {own:>18,.2f}  |差| {abs(vals[i]-own):>15,.2f}{tag}")
    gap12 = vals[order[0]] - (vals[order[1]] if len(order) > 1 else 0.0)
    print(f"  第 1 名 − 第 2 名 = {gap12:,.2f}"
          f"（很小 ⇒ LB 正好卡在某条割的折点上）")

    # q 分布对照：LB 的 q* vs 后向试探点的 q
    print(f"\n  q_out* 与阶段 1 试探点的 q 对照（看是否在外推）：")
    best = c1[int(order[0])]
    # 用阶段 1 割的锚点 state 代表"试探点"（每个锚点就是一个试探点）
    anc = np.vstack([c.state for c in c1])
    aq = anc[:, Ly.off_q:Ly.off_q + Ly.n_q]
    for r in range(prob.N_R):
        print(f"    仓{r}: q* = {qstar[r]:>9.3f}   试探点 q 范围 "
              f"[{aq[:, r].min():.3f}, {aq[:, r].max():.3f}]   "
              f"上界 Qmax = {prob.Qmax[r]}")

    # ---------------------------------------------------------- 阶段 2
    print("\n" + "=" * 78)
    print(f"阶段 2：从 x* 出发按引擎策略跑 {K} 条路径（串行），判定 c* 是否越界")
    print("=" * 78)
    rng = np.random.default_rng(20261010)
    totals = []
    t0 = time.time()
    for k in range(K):
        si_path = rng.integers(0, len(prob.pool), size=prob.T)
        x = xs.copy()
        total = 0.0
        for t in range(1, prob.T + 1):
            sub = StageSubproblem(t, prob, scfg)
            cn = None if t >= prob.T else (
                np.array([c.alpha for c in cuts[t + 1]]),
                np.vstack([c.beta for c in cuts[t + 1]]))
            sub.assemble(cn)
            sub.set_point(x, prob.pool[si_path[t - 1]])
            sub.solve_mip()
            total += sub.stage_cost()
            x = sub.out_state_vector()
        totals.append(total)
        if (k + 1) % 5 == 0:
            print(f"    {k+1}/{K} 条  均值 {np.mean(totals):,.1f}   "
                  f"{time.time()-t0:.0f}s", flush=True)
    C = float(np.mean(totals))
    se = float(np.std(totals, ddof=1) / np.sqrt(len(totals))) if len(totals) > 1 else 0.0
    cstar = float(vals[order[0]])
    print(f"\n  C(x*) = {C:,.1f} ± {1.96*se:,.1f}（95% CI，{K} 条）   "
          f"区间 [{min(totals):,.1f}, {max(totals):,.1f}]")
    print(f"  c*(x*) = {cstar:,.1f}（第 1 名割在 x* 处的值）")
    print(f"  LB     = {LB:,.1f}")
    print(f"  c*(x*) − C(x*) = {cstar - C:,.1f}")

    print("\n" + "=" * 78)
    if cstar > C + 1.96 * se:
        print("结论：★ 割越界 ★ —— c*(x*) 超过可行成本上界，它不是合法下界。")
        print("      LB 被这条割顶高了 ⇒ 负 gap 的直接原因。")
    else:
        print("结论：该割在 x* 处合法 —— LB 的偏高另有原因，转阶段 3B 查记账口径。")

    out = ROOT / "test" / "_diag_lb_out.json"
    out.write_text(json.dumps({
        "dir": d.name, "LB": LB, "theta": theta,
        "q_out_star": qstar.tolist(),
        "binding_cut": {"idx": int(order[0]), "it": c1[int(order[0])].it,
                        "alpha": float(c1[int(order[0])].alpha),
                        "obj": float(c1[int(order[0])].obj)},
        "c_star_at_xstar": cstar, "runner_up": float(vals[order[1]]),
        "C_xstar": C, "C_ci95": 1.96 * se, "K": K,
        "totals": [float(v) for v in totals],
        "x_star": xs.tolist(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n存档：{out}")


if __name__ == "__main__":
    main()
