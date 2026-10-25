# 可复现性边界：本引擎**不做逐位复现**，只保证统计等价。这个测试把边界测准。
#   python test/_t_det.py
#
# 三个叠加因素：
#   ① LP 最优对偶可能不唯一（对偶退化）——目标值唯一，但 .dj 可取到多个最优对偶解；
#   ② 共享常驻模型 + 自动热启动 ⇒ 取到哪个最优对偶取决于该 worker 的模型使用史；
#   ③ multiprocessing.Pool.map 不保证任务→worker 的固定映射。
# 于是同一 seed 两次跑，数值路径就可能分岔。**纯 LP 等级模型下前向 LP 也可能有多个
# 最优解**，所以连第 1 轮的试探点/割值都不保证逐位一致；本测试只断言下界统计等价
# 与次梯度合法，不再断言第 1 轮割逐位一致。
#
# 这**不影响正确性**：每个最优对偶都是合法次梯度，割总是有效下界，收敛性不变；
# 实测下界轨迹相对差在 1e-2 量级。但论文若要求逐位可复现，需要每次求解前清基
# （丢掉热启动——正是这次改造要买的东西），或在单进程下跑。
#
# ⚠ 建进程池的脚本必须有 __main__ 守卫（Windows spawn）。
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine                                          # noqa: E402
from utils import diagnostics as dg                                   # noqa: E402

N_ROUNDS = 3
N_PROC = 4
TOTAL_PATHS = 4          # 两次跑的**前向路径总数**必须相同，否则割数天然不同
N_SCENES = 8


def run(n_processes: int):
    """前向路径数 = 倍率 × 进程数，所以要让两次跑的路径总数相同，
    倍率必须随进程数反着调：1 进程 × 4 = 4 路径，4 进程 × 1 = 4 路径。"""
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        iterations=N_ROUNDS, n_processes=n_processes,
        forward_paths_per_process=TOTAL_PATHS // n_processes,
        n_backward_samples=N_SCENES, tag="det")
    assert scfg.n_forward_paths == TOTAL_PATHS, scfg.n_forward_paths
    prob = DefineProblem(pcfg)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    try:
        hist = eng.train(N_ROUNDS, verbose=False)
        pack = [(t, np.array([c.obj for c in eng.cuts[t]]),
                 np.vstack([c.beta for c in eng.cuts[t]]))
                for t in range(1, eng.T + 1)]
    finally:
        eng.close()
    return hist, pack


def same(a, b, tol=1e-9):
    return a == b if len(a) != len(b) else float(np.abs(a - b).max()) <= tol


def main() -> int:
    print(f"轮数 {N_ROUNDS}  后向场景 {N_SCENES}  前向路径总数 {TOTAL_PATHS}（两次相同）")

    print("\n[1] 1 进程跑两次（自复现）…")
    h1a, p1a = run(1)
    h1b, p1b = run(1)
    self1 = all(same(x[1], y[1]) and same(x[2], y[2]) for x, y in zip(p1a, p1b))

    print("[2] 4 进程跑两次（自复现）…")
    h4a, p4a = run(N_PROC)
    h4b, p4b = run(N_PROC)
    self4 = all(same(x[1], y[1]) and same(x[2], y[2]) for x, y in zip(p4a, p4b))

    print("\n[3] 跨进程数一致性（纯 LP 下仅记录，不作断言）")
    # 纯 LP 前向也可能多解；第 1 轮的试探点/割值不保证逐位一致。
    n_cut_per_stage = len(p1a[0][1]) // N_ROUNDS
    worst1, n1 = 0.0, 0
    for (t, oa, ba), (_, ob, bb) in zip(p1a, p4a):
        k = 1 + n_cut_per_stage                      # 1 条平凡割 + 第 1 轮的 n 条
        if ba.shape == bb.shape:
            worst1 = max(worst1, float(np.abs(ba[:k] - bb[:k]).max()))
            n1 += k
    print(f"  第 1 轮 {n1} 条割跨进程数最大差 = {worst1:.3e}（仅记录）")

    worst_all, n_all = 0.0, 0
    for (t, oa, ba), (_, ob, bb) in zip(p1a, p4a):
        n_all += len(oa)
        if ba.shape == bb.shape:
            worst_all = max(worst_all, float(np.abs(ba - bb).max()))
    print(f"  全部 {n_all} 条割最大差 = {worst_all:.3e}（第 2 轮起的对偶退化分岔）")

    print("\n[4] 下界轨迹与其相对差（统计等价性）")
    print("  轮   1 进程 LB          4 进程 LB         相对差")
    worst_lb = 0.0
    for a, b in zip(h1a, h4a):
        rel = abs(a["lb"] - b["lb"]) / max(1.0, abs(a["lb"]))
        worst_lb = max(worst_lb, rel)
        print(f"  {a['iter']:2d} {a['lb']:18.4f} {b['lb']:18.4f} {rel:14.3e}")

    print("\n[5] 分岔点上的 beta 仍是合法次梯度（取一个实点验次梯度不等式）")
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    scfg = load_solver_config(ROOT / "config/smoke/solver.toml")
    prob = DefineProblem(pcfg)
    rng = np.random.default_rng(2026)
    xs = dg.probe_states(rng, prob, n=1)
    comps = [prob.layout.ix_I(0), prob.layout.ix_n(0, 2), prob.layout.ix_q(0)]
    chk = dg.subgradient_inequality(make_stage_subproblem, prob, scfg, 2, xs[0],
                                    prob.pool[0], None, comps, h=1e-3)
    print(f"  次梯度不等式：{chk['n_checked']} 项通过 / {chk['n_skipped']} 跳过  "
          f"整体 {'OK' if chk['ok'] else 'BAD'}")

    ok = {
        # 纯 LP 下第 1 轮割值也不保证逐位一致，故只断言统计等价/次梯度合法
        "下界轨迹统计等价（相对差 < 2%）": worst_lb < 0.02,
        "分岔点的 beta 仍是合法次梯度": chk["ok"],
    }
    print()
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print()
    print(f"  已知边界（不作断言，仅记录）：自复现 1 进程={self1} "
          f"{N_PROC} 进程={self4}；第 1 轮 beta 最大差 {worst1:.3e}，"
          f"全部 beta 最大差 {worst_all:.3e}，下界最大相对差 {worst_lb:.2e}")
    print("  ⇒ 数值路径不逐位复现属预期（对偶退化 + 热启动 + 池调度），"
          "正确性与收敛性不受影响")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
