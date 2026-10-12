# simulate_only.py — 只做仿真：从已有结果目录恢复问题、配置与割，重跑蒙特卡洛。
#
#   python simulate_only.py results/base_20261010_120000
#   python simulate_only.py results/base_20261010_120000 10000
#   python simulate_only.py results/base_20261010_120000 100000 2000
#
# 位置参数：<结果目录> [路径数] [块大小]。
# 问题与求解参数一律从**结果目录里那份 config/ 拷贝**读回（原样拷贝的 TOML），
# 所以恢复出来的口径与当初训练时逐字一致；路径数缺省取该目录的 n_simulations，
# 种子也沿用同一份配置。
#
# 不改 run.py。仿真分块跑，块内结果用完即弃，10 万条也不会把内存顶爆。
# 只往该目录新增一个 summary_<时间戳>.txt，不覆盖任何已有文件。

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")   # Windows 控制台 GBK

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem, print_summary            # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine, simulate                                # noqa: E402
from utils.io import (has_configs, list_result_dirs, load_cuts,       # noqa: E402
                      save_summary)


def _resolve(d: Path, n_sim: int | None, chunk: int | None):
    if not d.is_dir():
        raise SystemExit(f"结果目录不存在: {d}")
    if not has_configs(d):
        raise SystemExit(
            f"{d} 里没有 config/problem.toml —— 这是重构前的结果目录。\n"
            f"旧目录的割与当前模型口径不同，不能复用（铁律 4）；"
            f"请用重构后的训练重新生成结果目录。")
    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    return pcfg, scfg, (scfg.n_simulations if n_sim is None else n_sim), \
        (scfg.chunk_size if chunk is None else chunk)


def main(dir_arg: str, n_sim_arg: int | None = None, chunk_arg: int | None = None):
    d = Path(dir_arg)
    if not d.is_absolute():
        d = ROOT / d
    pcfg, scfg, n_sim, chunk = _resolve(d, n_sim_arg, chunk_arg)

    prob = DefineProblem(pcfg)
    print_summary(prob)
    print(f"结果目录: {d}")
    print(f"仿真参数: 路径数 = {n_sim}  块大小 = {chunk}  种子 = {scfg.seed}  "
          f"进程 = {scfg.n_processes}")

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=d,
                     log=lambda *a, **k: None)
    try:
        cuts, meta = load_cuts(d, prob.T, prob.state_dim)
        eng.cuts = cuts
        eng.last_iter = int(meta.get("n_iterations", 0))
        lb = eng.lower_bound()          # 只读割，未再训练

        t0 = time.time()
        st = simulate(eng, n_paths=n_sim, seed=scfg.seed, chunk_size=chunk)
        t_sim = time.time() - t0
    finally:
        eng.close()

    ub, margin = st.mean, st.ci95
    gap = (ub - lb) / ub * 100 if ub else float("nan")

    print("\n" + "=" * 64)
    print("              蒙特卡洛仿真（总成本，全周期）")
    print("=" * 64)
    print(f"样本量:          {st.n}")
    print(f"均值:            {ub:.4f}")
    print(f"95% 置信区间:    [{ub - margin:.4f}, {ub + margin:.4f}]")
    print(f"标准差:          {st.std:.4f}")
    print(f"最小值 / 最大值: {st.lo:.4f} / {st.hi:.4f}")
    print("=" * 64)
    print(f"下界 LB (只读割，未再训练) = {lb:12.4f}")
    print(f"间隙 (UB-LB)/UB            = {gap:11.2f} %")
    print(f"仿真用时                    = {t_sim:.1f} s")
    print("=" * 64)
    stage = st.stage_mean()
    print("逐期平均成本（第 1..T 期）：")
    print("  " + " ".join(f"{v:.4g}" for v in stage))

    text = "\n".join([
        f"tag = {scfg.tag}  n_cuts = {sum(len(c) for c in eng.cuts)}  "
        f"iterations = {eng.last_iter}  n_simulations = {n_sim}"
        f"  (sim-only, 从 {d.name} 恢复)",
        f"LB = {lb}",
        f"UB = {ub} ± {margin}",
        f"gap_percent = {gap}",
        f"std = {st.std}  min = {st.lo}  max = {st.hi}",
        f"sim_time_s = {t_sim}  chunk_size = {chunk}  seed = {scfg.seed}",
        "逐期平均成本: " + " ".join(f"{v:.6g}" for v in stage),
        f"T = {pcfg.T}  A = {pcfg.A}  N_R = {pcfg.N_R}  L = {pcfg.L}  "
        f"pool_size = {pcfg.pool_size}",
        f"C = {pcfg.C}  h = {pcfg.h}  Cs = {pcfg.Cs}  atc = {pcfg.atc}  "
        f"tc = {pcfg.tc}  m = {pcfg.m}  eta = {pcfg.eta}",
    ])
    path = save_summary(d, text + "\n")
    print(f"报表已写: {path}")
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("用法: python simulate_only.py <结果目录> [路径数] [块大小]")
        print("\nresults/ 下现有目录：")
        for p in list_result_dirs(ROOT / "results"):
            print("  ", p.name)
        raise SystemExit(0 if args else 1)
    main(args[0],
         int(args[1]) if len(args) >= 2 else None,
         int(args[2]) if len(args) >= 3 else None)
