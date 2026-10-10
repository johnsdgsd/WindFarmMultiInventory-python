# simulate_only.py — 只做仿真：从已有结果目录恢复问题数据与割，重跑蒙特卡洛。
#
#   python simulate_only.py results/base_20261009_120000
#   python simulate_only.py results/base_20261009_120000 10000
#   python simulate_only.py results/base_20261009_120000 100000 2000
#
# 位置参数：<结果目录> [路径数] [块大小]。路径数缺省取该目录 config.txt 里记的
# n_simulations，种子也沿用 config 里的 seed（与当初训练时同一批随机流）。
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

from code.model import build_stage_subproblem                         # noqa: E402
from code.problem_data import from_json, print_summary                 # noqa: E402
from utils import SDDPEngine, simulate                                # noqa: E402
from utils.io import (list_result_dirs, load_cuts, load_json, read_config,  # noqa: E402
                      save_summary)


def main(dir_arg: str, n_sim_arg: int | None, chunk: int,
         n_processes: int | None = None):
    d = Path(dir_arg)
    if not d.is_absolute():
        d = ROOT / d
    if not d.is_dir():
        raise SystemExit(f"结果目录不存在: {d}")
    if not (d / "problem_data.json").is_file():
        raise SystemExit(f"{d} 里没有 problem_data.json，无法重建问题 —— 训练时没存问题数据")
    cfg = read_config(d)

    prob = from_json(load_json(d, "problem_data.json"))
    n_sim = n_sim_arg if n_sim_arg is not None else int(cfg.get("n_simulations", 10000))
    seed = int(cfg.get("seed", prob.seed))
    tag = cfg.get("tag", d.name)
    print_summary(prob)
    print(f"结果目录: {d}")

    # 沿用训练时的求解器（config.txt 里 solver=copt/cbc）
    use_copt = cfg.get("solver", "copt") != "cbc"
    eng = SDDPEngine(prob, build_stage_subproblem, T=prob.T,
                     state_dim=prob.state_dim, initial_state=prob.initial_state(),
                     n_processes=n_processes, seed=seed, results_dir=d,
                     log=lambda *a, **k: None, use_copt=use_copt)
    try:
        cuts, meta = load_cuts(d, prob.T, prob.state_dim)
        eng.cuts = cuts
        lb = eng.lower_bound()          # 只读割，未再训练

        print(f"开始蒙特卡洛仿真: 路径数 = {n_sim}，块大小 = {chunk}，种子 = {seed}")
        t0 = time.time()
        st = simulate(eng, n_paths=n_sim, seed=seed, chunk_size=chunk)
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
        f"tag = {tag}  n_cuts = {sum(len(c) for c in eng.cuts)}  "
        f"iterations = {cfg.get('iterations', '-')}  n_simulations = {n_sim}"
        f"  (sim-only, 从 {d.name} 恢复)",
        f"LB = {lb}",
        f"UB = {ub} ± {margin}",
        f"gap_percent = {gap}",
        f"std = {st.std}  min = {st.lo}  max = {st.hi}",
        f"sim_time_s = {t_sim}  chunk_size = {chunk}  seed = {seed}",
        "逐期平均成本: " + " ".join(f"{v:.6g}" for v in stage),
    ] + [f"{k} = {cfg[k]}" for k in
         ("T", "A", "N_R", "L", "pool_size", "C", "h", "Cs", "atc", "tc", "m", "eta")
         if k in cfg])
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
         int(args[2]) if len(args) >= 3 else 2000)
