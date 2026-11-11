# _t_async_v2_heur.py — V2 连续模型（可选 SA 前向）异步训练 runner。
#
# 训练与仿真分离；结束后保存 cuts/meta/summary。配置目录里的 solver.toml
# 决定 forward_heuristic 是否开启，因此同一 runner 可跑基线和 SA 对照。
from __future__ import annotations

import argparse
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
    ap.add_argument("--config-dir", default=str(ROOT / "test" / "_cfg_v2_heur"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cdir = Path(args.config_dir)
    if not cdir.is_absolute():
        cdir = ROOT / cdir
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    scfg.n_processes = int(args.workers)
    print(f"V2 async T={pcfg.T} pool={pcfg.pool_size} "
          f"workers={args.workers} seconds={args.seconds:g}")
    print(scfg.describe())
    prob = DefineProblem(pcfg)
    print_summary(prob)

    tag = args.tag or scfg.tag or "v2_heur"
    d = new_run_dir(tag, str(ROOT / scfg.results_root))
    print(f"结果目录：{d}")
    write_configs(d, cdir / "problem.toml", cdir / "solver.toml")
    save_json(d, "problem_data.json", to_json(prob))

    log = TrainingLog(d, "sddp_training_log.csv")
    eng = AsyncIterationEngine(prob, make_stage_subproblem, scfg,
                               results_dir=d, log=print,
                               n_workers=args.workers, paths_per_task=1)
    eng.on_iter = lambda rec: log.row(rec["iter"], rec["sim"], rec["lb"],
                                      rec["time"], rec["solves"])
    hist = []
    t0 = time.time()
    try:
        hist = eng.train(args.iters, max_seconds=args.seconds,
                         start_iter=1, verbose=True)
    finally:
        log.close()
        cuts = (eng.cut_pool.all_by_stage()
                if eng.cut_pool is not None else eng.cuts)
        save_cuts(d, cuts[:eng.T + 1], eng.meta())
        lb = eng.lower_bound()
        meta = eng.meta()
        h = meta.get("forward_heuristic", {})
        text = "\n".join([
            f"结果目录    : {d}",
            "模式        : async_iteration, forward="
            f"{'MIP' if scfg.integer_enabled else 'LP'} / SA, backward=LP",
            f"训练轮次    : {len(hist)}",
            f"下界 LB     : {lb:.6f}",
            f"历史 LB max : {eng.lb_best:.6f}",
            f"割数        : {sum(len(c) for c in eng.cuts[1:eng.T + 1])}",
            f"累计求解    : {eng.n_solves}",
            f"启发式      : mode={h.get('mode', 'none')} "
            f"trials={h.get('n_trials', 0)} "
            f"used={h.get('n_used', 0)} fail={h.get('n_fail', 0)}",
            f"SA          : moves={h.get('n_sa_moves', 0)} "
            f"accepted={h.get('n_sa_accepted', 0)} "
            f"time={h.get('sa_time_s', 0.0):.3f}s",
            f"qperturb    : perturbed={h.get('n_qp_perturbed', 0)} "
            f"base={h.get('n_qp_base', 0)} "
            f"solves={h.get('n_qp_solves', 0)} "
            f"extra_trials={h.get('n_qp_extra_trials', 0)}",
            f"经验回放    : triggers={meta.get('experience_replay', {}).get('n_triggers', 0)} "
            f"cuts={meta.get('experience_replay', {}).get('n_cuts', 0)} "
            f"solves={meta.get('experience_replay', {}).get('n_solves', 0)} "
            f"fail={meta.get('experience_replay', {}).get('n_fail', 0)}",
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
