# _t_async_cutlift_tail_sweep.py — 同一割池起点，单核单轮扫描 tail-k lift。
#
# 起点：results/tier2_async_simcopy_20261110_003117/cuts.json
# 对每个 k：只 lift 最后 k 个阶段，跑 1 轮，比较同一轮后的 LB。
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem, to_json                 # noqa: E402
from code.subproblem import make_stage_subproblem                   # noqa: E402
from utils.async_engine import AsyncIterationEngine                 # noqa: E402
from utils.io import (load_cuts, new_run_dir, save_json,             # noqa: E402
                      save_summary, write_configs)

BASE_CFG = ROOT / "test" / "_cfg_tier2_cutpool_2h"
START_RUN = ROOT / "results" / "tier2_async_simcopy_20261110_003117"
K_LIST = [0, 1, 2, 3, 5, 10, 20, 40, 60]
ITERS = 1
WORKERS = 1


def run_one(prob, base_cfg, start_cuts, k: int, tag_prefix: str,
            value: str = "best_bnd", time_limit: float = 5.0):
    cfg = copy.deepcopy(base_cfg)
    cfg.n_processes = WORKERS
    cfg.cut_selection = "julia"
    cfg.cut_lift.enabled = k > 0
    cfg.cut_lift.scope = "tail"
    cfg.cut_lift.last_k = int(k)
    cfg.cut_lift.mode = "always"
    cfg.cut_lift.value = value
    cfg.cut_lift.time_limit = float(time_limit)
    if value == "exact":
        cfg.cut_lift.rel_gap = 0.0
        cfg.cut_lift.abs_gap = 0.0
    else:
        cfg.cut_lift.rel_gap = 1e-9
        cfg.cut_lift.abs_gap = 0.0
    cfg.cut_lift.safety_margin = 1e-6

    tag = f"{tag_prefix}_k{k}_1it"
    d = new_run_dir(tag, str(ROOT / cfg.results_root))
    print(f"\n=== k={k}  {d} ===")
    write_configs(d, BASE_CFG / "problem.toml", BASE_CFG / "solver.toml")
    save_json(d, "problem_data.json", to_json(prob))
    save_json(d, "sweep_params.json", {
        "k": k, "iters": ITERS, "workers": WORKERS,
        "start_run": str(START_RUN), "cut_lift": {
            "scope": cfg.cut_lift.scope, "last_k": cfg.cut_lift.last_k,
            "mode": cfg.cut_lift.mode, "value": cfg.cut_lift.value,
            "time_limit": cfg.cut_lift.time_limit,
            "rel_gap": cfg.cut_lift.rel_gap,
            "abs_gap": cfg.cut_lift.abs_gap}})

    eng = AsyncIterationEngine(prob, make_stage_subproblem, cfg,
                               results_dir=d, log=print,
                               n_workers=WORKERS, paths_per_task=1)
    n_start = eng.seed_initial_cuts(copy.deepcopy(start_cuts))
    t0 = time.time()
    lb0 = eng.lower_bound()
    hist = eng.train(ITERS, max_seconds=0, start_iter=1, verbose=True)
    lb1 = float(hist[-1]["lb"]) if hist else eng.lower_bound()
    wall = time.time() - t0
    meta = eng.meta()
    lift = meta.get("cut_lift", {})
    rec = {
        "k": int(k),
        "start_cuts": int(n_start),
        "lb0": float(lb0),
        "lb1": float(lb1),
        "delta_lb": float(lb1 - lb0),
        "wall_s": float(wall),
        "lift_cuts": int(lift.get("n_lift_cuts", 0)),
        "lift_mip": int(lift.get("n_lift_mip", 0)),
        "fallback": int(lift.get("n_fallback", 0)),
        "result_dir": str(d),
    }
    save_json(d, "sweep_result.json", rec)
    text = "\n".join([
        f"tail_k     : {k}",
        f"start_cuts : {n_start}",
        f"LB before  : {lb0:.6f}",
        f"LB after   : {lb1:.6f}",
        f"delta LB   : {lb1 - lb0:+.6f}",
        f"lift cuts  : {rec['lift_cuts']}",
        f"lift MIP   : {rec['lift_mip']}",
        f"fallback   : {rec['fallback']}",
        f"wall       : {wall:.1f} s",
    ])
    save_summary(d, text + "\n")
    print(text)
    eng.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="*", default=None,
                    help="只跑指定 k；默认跑完整 K_LIST")
    ap.add_argument("--value", choices=["exact", "best_bnd"],
                    default="best_bnd", help="增强割取值口径")
    ap.add_argument("--time-limit", type=float, default=5.0,
                    help="每次 lift MIP 的时限（秒）")
    ap.add_argument("--tag-prefix", default="tier2_tail_sweep",
                    help="结果目录前缀")
    ap.add_argument("--summary", default="test/_tier2_tail_sweep_summary.json",
                    help="汇总 JSON 路径（相对项目根目录）")
    args = ap.parse_args()
    ks = list(args.k) if args.k else list(K_LIST)

    prob = DefineProblem(load_problem_config(BASE_CFG / "problem.toml"))
    base_cfg = load_solver_config(BASE_CFG / "solver.toml")
    start_cuts, meta = load_cuts(START_RUN, prob.T, prob.state_dim)
    print(f"start cuts : {sum(len(c) for c in start_cuts[1:prob.T + 1])}  "
          f"from {START_RUN}")

    rows = []
    for k in ks:
        rows.append(run_one(prob, base_cfg, start_cuts, k,
                            tag_prefix=args.tag_prefix,
                            value=args.value,
                            time_limit=args.time_limit))

    out = ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "start_run": str(START_RUN),
        "iters": ITERS,
        "workers": WORKERS,
        "value": args.value,
        "time_limit": args.time_limit,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== sweep summary ===")
    print(" k   LB0          LB1          delta       lift_mip  fallback  wall_s")
    for r in rows:
        print(f"{r['k']:2d}  {r['lb0']:12.3f} {r['lb1']:12.3f} "
              f"{r['delta_lb']:+11.3f} {r['lift_mip']:9d} "
              f"{r['fallback']:8d} {r['wall_s']:7.1f}")
    print(f"\nsummary -> {out}")
    print("TAIL_SWEEP_DONE")


if __name__ == "__main__":
    main()
