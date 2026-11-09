# _t_async_v2.py — 在 V2 基线配置上跑 iteration 级并行原型。
# 结果写入 results/asyncV2_<时间戳>/（只新增，不碰旧结果）。
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem, print_summary, to_json  # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.async_engine import AsyncIterationEngine              # noqa: E402
from utils.io import (TrainingLog, new_run_dir, save_cuts,        # noqa: E402
                      save_json, save_summary, write_configs)


def _make_effective_config(src_dir: Path, dst_dir: Path, *, workers: int,
                           iterations: int, seconds: float,
                           cut_selection: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_dir / "problem.toml", dst_dir / "problem.toml")
    text = (src_dir / "solver.toml").read_text(encoding="utf-8")
    text = re.sub(r'cut_selection\s*=\s*"[^"]+"',
                  f'cut_selection = "{cut_selection}"', text, count=1)
    text = text.replace("n_processes = 0", f"n_processes = {workers}")
    text = text.replace("iterations = 200", f"iterations = {iterations}")
    text = text.replace("max_train_seconds = 0",
                        f"max_train_seconds = {seconds:g}")
    text = text.replace("n_simulations = 10000", "n_simulations = 0")
    text = text.replace('tag = "base_T60"', 'tag = "asyncV2"')
    (dst_dir / "solver.toml").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--paths-per-task", type=int, default=1)
    ap.add_argument("--cut-selection", choices=["none", "lml1", "julia"],
                    default="none")
    ap.add_argument("--config-dir", default=str(ROOT / "config"))
    args = ap.parse_args()

    src = Path(args.config_dir)
    cfg_dir = ROOT / "test" / "_cfg_async_v2"
    _make_effective_config(src, cfg_dir, workers=args.workers,
                           iterations=args.iters, seconds=args.seconds,
                           cut_selection=args.cut_selection)
    pcfg = load_problem_config(cfg_dir / "problem.toml")
    scfg = load_solver_config(cfg_dir / "solver.toml")

    print("=" * 66)
    print(f"async V2  T={pcfg.T} pool={pcfg.pool_size} workers={scfg.n_processes} "
          f"paths/task={args.paths_per_task} cut={scfg.cut_selection} "
          f"seconds={args.seconds:g} iters<={args.iters}")
    print(scfg.describe())
    print("=" * 66)
    prob = DefineProblem(pcfg)
    print_summary(prob)

    d = new_run_dir(scfg.tag, str(ROOT / scfg.results_root))
    print(f"结果目录：{d}")
    write_configs(d, cfg_dir / "problem.toml", cfg_dir / "solver.toml")
    save_json(d, "problem_data.json", to_json(prob))

    log = TrainingLog(d, "sddp_training_log.csv")
    eng = AsyncIterationEngine(prob, make_stage_subproblem, scfg,
                               results_dir=d, log=print,
                               paths_per_task=args.paths_per_task)
    t0 = time.time()

    def on_iter(rec):
        log.row(rec["iter"], rec["sim"], rec["lb"], rec["time"], rec["solves"])

    eng.on_iter = on_iter
    hist = []
    try:
        hist = eng.train(args.iters, max_seconds=args.seconds,
                         start_iter=1, verbose=True)
    finally:
        log.close()
        cuts = (eng.cut_pool.all_by_stage()
                if eng.cut_pool is not None else eng.cuts)
        save_cuts(d, cuts[:eng.T + 1], eng.meta())
        lb = eng.lower_bound()
        summary = "\n".join([
            f"结果目录    : {d}",
            f"模式        : async_iteration (cut_selection={scfg.cut_selection})",
            f"训练轮次    : {len(hist)}（每轮 1 条前向路径）",
            f"下界 LB     : {lb:.6f}",
            f"历史 LB max : {eng.lb_best:.6f}",
            f"割数        : {sum(len(c) for c in eng.cuts[1:eng.T + 1])}",
            f"累计求解    : {eng.n_solves}",
            f"用时        : {time.time() - t0:.1f} s",
            "",
            "训练与仿真分离：UB/gap 需要另跑 simulate.py",
        ])
        print("\n" + summary)
        save_summary(d, summary + "\n")
        eng.close()
    print(f"完成：{d}")


if __name__ == "__main__":
    main()
