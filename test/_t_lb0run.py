# 在内存里跑 2 轮 SDDP（单进程、单前向路径、不落盘），看割到底长什么样。
# 这是诊断 LB≡0 的最直接途径 —— 不用等生产跑的 600 s checkpoint。
#
#   python test/_t_lb0run.py [轮数]
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils import SDDPEngine                                      # noqa: E402


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "solver.toml")
    scfg.n_processes = 1              # 串行，进程内调试
    prob = DefineProblem(pcfg)
    print(f"T={prob.T} A={prob.A} 机队 {int(prob.N.sum())} 台   "
          f"前向路径 {scfg.forward_paths_per_process}×1  后向场景 {scfg.n_backward_samples}")

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    for k in range(1, n_iter + 1):
        eng.round_id = k
        trial, results = eng.forward(scfg.n_forward_paths)
        sim = float(np.mean([r[2] for r in results]))
        eng.backward(trial, it=k)
        lb = eng.lower_bound()
        print(f"  轮 {k}: 前向均值 {sim:>12,.1f}   LB {lb:>12,.4f}   "
              f"割数 {sum(len(c) for c in eng.cuts[1:])}")
    eng.close()

    print(f"\n{'期':>4} {'割数':>5} {'alpha最小':>14} {'alpha中位':>14} "
          f"{'alpha最大':>14} {'|beta|中位':>11}")
    for t in range(1, prob.T + 1):
        cs = eng.cuts[t]
        al = np.array([c.alpha for c in cs])
        nb = np.array([np.linalg.norm(c.beta) for c in cs])
        if t in (1, 5, 10, 20, 30, 40, 50, 55, 58, 59, 60):
            print(f"{t:>4} {len(cs):>5} {al.min():>14,.2f} {np.median(al):>14,.2f} "
                  f"{al.max():>14,.2f} {np.median(nb):>11,.2f}")

    # 非平凡割（it>0）的 alpha 逐期
    print(f"\n  逐期【非平凡割】的 alpha（看终端信息有没有传回来）：")
    for k in range(0, prob.T, 6):
        seg = []
        for t in range(k + 1, min(k + 7, prob.T + 1)):
            al = [c.alpha for c in eng.cuts[t] if c.it > 0]
            seg.append(max(al) if al else float("nan"))
        print(f"    期 {k+1:>3}-{k+len(seg):<3}: " +
              " ".join(f"{v:>12,.1f}" if v == v else f"{'—':>12}" for v in seg))

    # 阶段 1 的割在初始状态附近的取值
    x0 = prob.initial_state()
    Ly = prob.layout
    c1 = eng.cuts[1]
    print(f"\n  阶段 1 共 {len(c1)} 条割；在 x₀（q=0）处的取值：")
    vals = np.array([c.value_at(x0) for c in c1])
    print(f"    最大 {vals.max():,.2f}   中位 {np.median(vals):,.2f}   最小 {vals.min():,.2f}")
    for i in np.argsort(-vals)[:4]:
        c = c1[int(i)]
        bI = Ly.I(c.beta); bo = Ly.o(c.beta).ravel(); bn = Ly.n(c.beta); bq = Ly.q(c.beta)
        xI = Ly.I(c.state); xo = Ly.o(c.state).ravel(); xn = Ly.n(c.state); xq = Ly.q(c.state)
        print(f"    轮 {c.it:>3}  alpha {c.alpha:>13,.2f}  在x₀ {vals[int(i)]:>12,.2f}  "
              f"|beta| {np.linalg.norm(c.beta):>10,.3f}  obj {c.obj:>12,.2f}")
        print(f"        锚点 Σn {xn.sum():>6.0f}/{prob.N.sum():.0f}（停机 {prob.N.sum()-xn.sum():>4.0f}）"
              f"  ΣI {xI.sum():>4.0f}  Σo {xo.sum():>4.0f}  Σq {xq.sum():>5.0f}")
        print(f"        β_I {np.round(bI,1).tolist()}  β_q {np.round(bq,1).tolist()}  "
              f"β_n[a=0,1,2,10,30,59] {np.round(bn[:, [0,1,2,10,30,59]].mean(axis=0),1).tolist()}")

    # —— 复现阶段 0 的 LB：LP 选了什么 q_out？哪条割在卡？ ——
    from code.subproblem import StageSubproblem                       # noqa: E402
    from utils.cut import pack                                        # noqa: E402
    s0 = StageSubproblem(0, prob, scfg)
    s0.assemble(pack(c1, prob.state_dim))
    s0.set_state(x0)
    s0.solve_lp()
    xs = s0.out_state_vector()
    print(f"\n  阶段 0：LB = {s0.obj:,.4f}   q_out* = {np.round(Ly.q(xs),2).tolist()}"
          f"   Qmax = {prob.Qmax.tolist()}")
    v2 = np.array([c.value_at(xs) for c in c1])
    o2 = np.argsort(-v2)
    print(f"    在 q_out* 处各割取值：最大 {v2[o2][0]:,.4f}  第2 {v2[o2][1]:,.4f}  "
          f"中位 {np.median(v2):,.2f}")
    for i in o2[:3]:
        c = c1[int(i)]
        print(f"      轮 {c.it:>3}  alpha {c.alpha:>13,.2f}  在q*处 {v2[int(i)]:>12,.4f}  "
              f"β_q {np.round(Ly.q(c.beta),1).tolist()}")
    s0.free()


if __name__ == "__main__":
    main()
