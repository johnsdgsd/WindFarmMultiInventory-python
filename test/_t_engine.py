# 引擎端到端：训练数轮 + 割自洽 + 试探点 + 变量记录 + 仿真。
#   python test/_t_engine.py
#
# 割自洽 = 截距式定义 alpha + beta@state == obj 逐条成立；不成立说明 alpha 算错了。
#
# ⚠ 建进程池的脚本必须有 __main__ 守卫（Windows spawn）。
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine, simulate                                # noqa: E402

N_ROUNDS = 4


def main() -> int:
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        iterations=N_ROUNDS, n_processes=2, n_simulations=30, chunk_size=30,
        tag="engine")
    prob = DefineProblem(pcfg)

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    t0 = time.time()
    hist = eng.train(N_ROUNDS, verbose=False)
    print(f"训练 {N_ROUNDS} 轮用时 {time.time() - t0:.1f} s，"
          f"割数 {sum(len(c) for c in eng.cuts)}，求解 {eng.n_solves} 次")

    print()
    print("=" * 78)
    print("§1 割自洽：alpha + beta@state == obj（截距式的定义）")
    print("=" * 78)
    worst, where = 0.0, None
    for t in range(1, eng.T + 1):
        for c in eng.cuts[t]:
            err = abs(c.alpha + float(c.beta @ c.state) - c.obj)
            if err > worst:
                worst, where = err, (t, c.it)
    print(f"  最差偏差 {worst:.3e}（期 {where[0]}，轮 {where[1]}）")
    ok_cut = worst < 1e-6

    print()
    print("=" * 78)
    print("§2 前向：一任务一条路径，收齐每阶段试探点")
    print("=" * 78)
    trial, results = eng.forward(scfg.n_forward_paths, record=True)
    print(f"  路径数 {len(results)}  每阶段试探点 {[len(trial[t]) for t in range(1, eng.T + 1)]}")
    print(f"  各路径总成本 {[round(r[2], 1) for r in results]}")
    recs = results[0][3]
    print(f"  路径 0 记录条数 {len(recs)}，第 2 期键 {sorted(recs[1].keys())[:6]} …")
    ok_trial = all(len(trial[t]) == scfg.n_forward_paths for t in range(1, eng.T + 1))

    print()
    print("=" * 78)
    print("§3 仿真（复用前向任务）")
    print("=" * 78)
    st = simulate(eng, n_paths=scfg.n_simulations, seed=scfg.seed,
                  chunk_size=scfg.chunk_size)
    lb = eng.lower_bound()
    print(f"  UB = {st.mean:.4f} ± {st.ci95:.4f}（{st.n} 条）   LB = {lb:.4f}")
    print(f"  逐期均值 {np.round(st.stage_mean(), 1).tolist()}")
    print(f"  各期都有成本: {bool((st.stage_mean() > 0).all())}")
    eng.close()

    ok = {
        "割自洽 alpha+beta@state == obj": ok_cut,
        "每阶段试探点数 == 前向路径数": ok_trial,
        "仿真样本数正确": st.n == scfg.n_simulations,
        "下界为正": lb > 0,
    }
    print()
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
