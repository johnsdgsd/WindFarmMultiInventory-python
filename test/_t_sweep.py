# 后向串行扫描验收：T=8 小实例训若干轮，断言下界在头几轮就离开 0。
#   python test/_t_sweep.py
#
# 旧实现（轮初打割快照 + 全部任务一次铺满）信息每轮最多传播一期，T=8 时要 ~8 轮才有
# 非零下界、T=200 时上百轮不动。改成期间串行后，一次反向扫描传遍全视界。
#
# ⚠ Windows spawn：建进程池的脚本必须有 __main__ 守卫，否则子进程重新 import 本模块时
#   会再建一次池，无限递归。所有 test/_t_*.py 都遵守这条。

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils import SDDPEngine, simulate                            # noqa: E402

T_SWEEP = 8
N_ROUNDS = 6


def main() -> int:
    pcfg = dataclasses.replace(
        load_problem_config(ROOT / "config/smoke/problem.toml"), T=T_SWEEP)
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        iterations=N_ROUNDS, n_processes=2, n_backward_samples=0,
        n_simulations=40, chunk_size=40, tag="sweep")
    prob = DefineProblem(pcfg)

    print("=" * 78)
    print(f"后向串行扫描：T={T_SWEEP}，{scfg.n_processes} 进程 × "
          f"{scfg.forward_paths_per_process} 路径 = 每阶段 {scfg.n_forward_paths} 个试探点")
    print("=" * 78)

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None, log=print)
    print(f"  状态维 {eng.state_dim}  场景池 {eng.n_scenes}  割初值 {len(eng.cuts[1])} 条")
    t0 = time.time()
    hist = eng.train(N_ROUNDS, verbose=True)
    elapsed = time.time() - t0

    print()
    print("  轮次  下界 LB        前向均值      割数    求解次数   用时(s)")
    for h in hist:
        print(f"  {h['iter']:4d}  {h['lb']:13.4f}  {h['sim']:13.4f}  "
              f"{h['n_cuts']:6d}  {h['solves']:10d}  {h['time']:8.1f}")

    lbs = [h["lb"] for h in hist]
    nz_round = next((i + 1 for i, v in enumerate(lbs) if v > 0), None)
    print()
    print(f"  下界首次为正出现在第 {nz_round} 轮（阈值：≤ 3 轮）")

    print()
    print("=" * 78)
    print("仿真（复用前向任务）")
    print("=" * 78)
    st = simulate(eng, n_paths=scfg.n_simulations, seed=scfg.seed,
                  chunk_size=scfg.chunk_size)
    lb = eng.lower_bound()
    gap = (st.mean - lb) / abs(st.mean) * 100 if st.mean else float("nan")
    print(f"  UB = {st.mean:.4f} ± {st.ci95:.4f}   LB = {lb:.4f}   gap = {gap:.2f}%")
    print(f"  逐期均值 {np.round(st.stage_mean(), 1).tolist()}")
    print(f"  割数 {sum(len(c) for c in eng.cuts)}   累计求解 {eng.n_solves}")
    print(f"  总用时 {elapsed:.1f} s")
    eng.close()

    ok = {
        "下界在 3 轮内 > 0": nz_round is not None and nz_round <= 3,
        f"仿真完成 {scfg.n_simulations} 条": st.n == scfg.n_simulations,
        "LB/UB 有限": bool(np.isfinite(lb) and np.isfinite(st.mean)),
    }
    print()
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
