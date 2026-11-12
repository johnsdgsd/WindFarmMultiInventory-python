# _t_async_tier2_bandit.py — Tier2 全整数 + Julia 割池 + BanditDuality runner。
#
# 支持 --resume：从结果目录的 cuts.json/meta 恢复活跃割和 Bandit 状态，
# 继续训练；checkpoint_interval 由 solver.toml 控制。
from __future__ import annotations

import argparse
import csv
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
from utils.io import (TrainingLog, archive_log, load_cuts,        # noqa: E402
                      new_run_dir, save_cuts, save_json,
                      save_summary, write_configs)
from utils.io import has_configs, problem_diff                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(ROOT / "test" / "_cfg_tier2_bandit"))
    ap.add_argument("--seconds", type=float, default=28800.0)
    ap.add_argument("--iters", type=int, default=100000000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None,
                    help="结果目录；从其中 cuts.json/meta 恢复并续训")
    args = ap.parse_args()

    cdir = Path(args.config_dir)
    if not cdir.is_absolute():
        cdir = ROOT / cdir
    if args.resume:
        d = Path(args.resume)
        if not d.is_absolute():
            d = ROOT / d
        if not has_configs(d):
            raise SystemExit(f"{d} 没有 config/problem.toml，不能续训")
        cdir = d / "config"
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    scfg.n_processes = int(args.workers)
    if scfg.cut_selection != "julia":
        raise SystemExit("Bandit runner 只支持 cut_selection=julia")
    if not scfg.bandit_duality:
        raise SystemExit("需要 bandit_duality=true")
    if not scfg.cut_lift.enabled:
        raise SystemExit("需要 [cut_lift].enabled=true")
    print(f"Tier2 Bandit T={pcfg.T} pool={pcfg.pool_size} workers={args.workers} "
          f"seconds={args.seconds:g} resume={bool(args.resume)}")
    print(scfg.describe())
    prob = DefineProblem(pcfg)
    print_summary(prob)

    start_iter = 1
    if args.resume:
        cuts, meta = load_cuts(d, prob.T, prob.state_dim)
        for t, stage_cuts in enumerate(cuts):
            for i, c in enumerate(stage_cuts):
                if not hasattr(c, "uid") or c.uid is None:
                    c.uid = ("resume", t, i)
        diff = problem_diff(
            load_problem_config(d / "config" / "problem.toml"), pcfg)
        if diff:
            raise SystemExit(f"问题配置不一致：{diff}")
        archive_log(d)
    else:
        tag = args.tag or scfg.tag or "tier2_bandit"
        d = new_run_dir(tag, str(ROOT / scfg.results_root))
        write_configs(d, cdir / "problem.toml", cdir / "solver.toml")
        save_json(d, "problem_data.json", to_json(prob))
        cuts, meta = None, {}
    print(f"结果目录：{d}")

    log = TrainingLog(d, "sddp_training_log.csv")
    bandit_path = d / "bandit_log.csv"
    bandit_exists = bandit_path.is_file() and bandit_path.stat().st_size > 0
    bandit_f = bandit_path.open("a", newline="", encoding="utf-8")
    bandit_w = csv.writer(bandit_f)
    if not bandit_exists:
        bandit_w.writerow([
            "iter", "time_s", "lb", "sim", "arm_used", "next_arm",
            "reason", "reward", "score_continuous", "score_strengthened",
            "n_rewards_continuous", "n_rewards_strengthened"])

    def log_flush(*a, **k):
        print(*a, **k, flush=True)

    eng = AsyncIterationEngine(prob, make_stage_subproblem, scfg,
                               results_dir=d, log=log_flush,
                               n_workers=args.workers, paths_per_task=1)
    if args.resume:
        eng.cuts = list(cuts) + [[]]
        if eng.cut_pool is not None:
            eng.restore_cut_pool()
        eng.lb_best = float(meta.get("lb_best", 0.0))
        eng.last_iter = int(meta.get("n_iterations", 0))
        eng.restore_bandit(meta)
        start_iter = eng.last_iter + 1
        print(f"续训：起始轮次 {start_iter}，LB={eng.lb_best:.6f}，"
              f"当前臂={eng.bandit.current_arm if eng.bandit else 'n/a'}")
    def on_iter(rec):
        log.row(rec["iter"], rec["sim"], rec["lb"],
                rec["time"], rec["solves"])
        b = rec.get("bandit") or {}
        scores = b.get("scores") or {}
        nrew = b.get("n_rewards") or [0, 0]
        bandit_w.writerow([
            rec["iter"], f"{rec['time']:.6f}", f"{rec['lb']:.6f}",
            f"{rec['sim']:.6f}", rec.get("arm_used"),
            b.get("next_arm"), b.get("reason"),
            "" if b.get("reward") is None else f"{b['reward']:.9g}",
            scores.get("continuous", ""),
            scores.get("strengthened", ""),
            nrew[0] if len(nrew) > 0 else "",
            nrew[1] if len(nrew) > 1 else "",
        ])
        bandit_f.flush()

    eng.on_iter = on_iter
    t0 = time.time()
    hist = []
    try:
        hist = eng.train(args.iters, max_seconds=args.seconds,
                         start_iter=start_iter, verbose=True)
    finally:
        log.close()
        bandit_f.close()
        cuts_out = (eng.cut_pool.all_by_stage()
                    if eng.cut_pool is not None else eng.cuts)
        save_cuts(d, cuts_out[:eng.T + 1], eng.meta())
        lb = eng.lower_bound()
        text = "\n".join([
            f"结果目录    : {d}",
            "模式        : async_iteration, Julia cut pool, BanditDuality",
            f"integer     : {scfg.integer_scope}  n0={prob.n0_mode}",
            f"起始轮次    : {start_iter}",
            f"训练轮次    : {len(hist)}",
            f"下界 LB     : {lb:.6f}  (stage0 {eng.lb_solver.upper()})",
            f"历史 LB max : {eng.lb_best:.6f}",
            f"割数        : {sum(len(c) for c in eng.cuts[1:eng.T + 1])}",
            f"累计求解    : {eng.n_solves}",
            f"割提升      : {scfg.cut_lift.describe()}",
            f"提升统计    : cuts={eng.n_lift_cuts}  mip={eng.n_lift_mip}  "
            f"fallback={eng.n_lift_fallback}",
            f"Bandit      : {eng.bandit.meta() if eng.bandit else 'off'}",
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
