# _t_async_tier2.py — Tier2 全整数异步 iteration 引擎 2h runner。
# 前向 solve_mip，后向 solve_lp，Julia 式割池；训练后另跑 simulate.py。
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=7200.0)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cut-selection", default="julia")
    ap.add_argument("--last-k", type=int, default=0,
                    help="只 lift 最后 k 个阶段；>0 时覆盖 scope=tail")
    ap.add_argument("--lift-stages", default=None,
                    help="逗号分隔的阶段号；设置后覆盖 scope=selected")
    ap.add_argument("--tag", default=None,
                    help="结果目录前缀；默认用 solver.toml 的 [output].tag")
    ap.add_argument("--config-dir",
                    default=str(ROOT / "test" / "_cfg_tier2_cutpool_2h"))
    args = ap.parse_args()

    cdir = Path(args.config_dir)
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    scfg.cut_selection = args.cut_selection
    scfg.n_processes = args.workers
    if args.last_k > 0:
        if not scfg.cut_lift.enabled:
            raise SystemExit("--last-k 需要 [cut_lift].enabled=true")
        scfg.cut_lift.scope = "tail"
        scfg.cut_lift.last_k = int(args.last_k)
    if args.lift_stages:
        if not scfg.cut_lift.enabled:
            raise SystemExit("--lift-stages 需要 [cut_lift].enabled=true")
        stages = tuple(int(s) for s in args.lift_stages.split(",") if s.strip())
        if not stages:
            raise SystemExit("--lift-stages 不能为空")
        scfg.cut_lift.scope = "selected"
        scfg.cut_lift.lift_stages = stages
    if scfg.cut_selection != "julia":
        raise SystemExit("本 runner 只用于 cut_selection=julia")
    print(f"async Tier2 T={pcfg.T} pool={pcfg.pool_size} workers={args.workers} "
          f"scope={scfg.integer_scope} seconds={args.seconds:g}")
    print(scfg.describe())
    prob = DefineProblem(pcfg)
    print_summary(prob)

    run_tag = args.tag or scfg.tag or "tier2_async_cutpool_2h"
    d = new_run_dir(run_tag, str(ROOT / scfg.results_root))
    print(f"结果目录：{d}")
    write_configs(d, cdir / "problem.toml", cdir / "solver.toml")
    save_json(d, "problem_data.json", to_json(prob))

    log = TrainingLog(d, "sddp_training_log.csv")
    eng = AsyncIterationEngine(prob, make_stage_subproblem, scfg,
                               results_dir=d, log=print,
                               n_workers=args.workers, paths_per_task=1)
    t0 = time.time()
    eng.on_iter = lambda rec: log.row(rec["iter"], rec["sim"], rec["lb"],
                                      rec["time"], rec["solves"])
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
        text = "\n".join([
            f"结果目录    : {d}",
            "模式        : async_iteration, forward=MIP, backward=LP",
            f"integer     : {scfg.integer_scope}",
            f"训练轮次    : {len(hist)}",
            f"下界 LB     : {lb:.6f}",
            f"历史 LB max : {eng.lb_best:.6f}",
            f"割数        : {sum(len(c) for c in eng.cuts[1:eng.T + 1])}",
            f"累计求解    : {eng.n_solves}",
            f"割提升      : {scfg.cut_lift.describe()}",
            f"提升统计    : cuts={eng.n_lift_cuts}  mip={eng.n_lift_mip}  "
            f"fallback={eng.n_lift_fallback}",
            f"用时        : {time.time() - t0:.1f} s",
            "",
            "训练与仿真分离：UB/gap 需要另跑 simulate.py",
        ])
        print("\n" + text)
        save_summary(d, text + "\n")
        eng.close()
    print(f"完成：{d}")


if __name__ == "__main__":
    main()
