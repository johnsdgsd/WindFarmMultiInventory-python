# _t_root_cut_lift.py — 固定割池，只 lift 阶段 1 的活跃割，直接看根 LB 变化。
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem, to_json                 # noqa: E402
from code.subproblem import make_stage_subproblem                   # noqa: E402
from utils import copt_env as ce                                    # noqa: E402
from utils.cut import Cut, pack                                     # noqa: E402
from utils.io import (load_cuts, new_run_dir, save_json,             # noqa: E402
                      save_summary, write_configs)

BASE_CFG = ROOT / "test" / "_cfg_tier2_cutpool_2h"
START_RUN = ROOT / "results" / "tier2_async_simcopy_20261110_003117"


def solve_root(prob, cfg, cuts1):
    sub = make_stage_subproblem(0, prob, cfg)
    try:
        sub.assemble(pack(cuts1, prob.state_dim), key=("root", id(cuts1)))
        sub.set_state(prob.initial_state())
        sub.solve_lp()
        xout = np.asarray(ce.values(sub.model, sub.x_out), dtype=float)
        lb = float(sub.obj)
        theta = float(sub.theta_value())
    finally:
        sub.free()
    return lb, theta, xout


def lift_cut(prob, cfg, cuts, cut, value, time_limit, recompute_beta=False):
    """对阶段 1 的割做逐场景 Lagrangian，返回 (新割, 统计)。

    recompute_beta=False：复用保存割的 beta（只强化已有截距）。
    recompute_beta=True ：逐场景从当前 LP 取 .dj，再平均（标准 strengthened cut）。
    """
    sub = make_stage_subproblem(1, prob, cfg)
    vals = []
    betas = []
    n_fb = 0
    t0 = time.time()
    try:
        sub.assemble(pack(cuts[2], prob.state_dim), key=("lift", id(cut)))
        x = np.asarray(cut.state, dtype=float)
        sub.set_state(x, require_int=True)
        beta_saved = np.asarray(cut.beta, dtype=float)
        for scene in prob.pool:
            sub.set_scene(scene)
            sub.solve_lp()
            lp_obj = float(sub.obj)
            beta = (np.asarray(sub.gradient("dj"), dtype=float)
                    if recompute_beta else beta_saved)
            nu = sub.solve_lift(
                beta, time_limit=float(time_limit), rel_gap=0.0,
                abs_gap=0.0, safety_margin=1e-6, value=value)
            if nu is None:
                nu = lp_obj - float(beta @ x)
                n_fb += 1
            vals.append(float(nu + beta @ x))
            betas.append(beta)
        beta_new = np.mean(np.vstack(betas), axis=0)
        alpha = float(np.mean(vals)) - float(beta_new @ x)
        old_val = float(cut.alpha + cut.beta @ x)
        if float(alpha + beta_new @ x) < old_val:
            alpha = old_val - float(beta_new @ x)
        new = Cut(alpha, beta_new, cut.state, 1,
                  alpha + float(beta_new @ x), cut.it)
    finally:
        sub.free()
    return new, {"fallback": n_fb, "lift_s": time.time() - t0,
                 "alpha_old": float(cut.alpha), "alpha_new": float(new.alpha),
                 "value_old": old_val, "value_new": float(new.alpha + new.beta @ x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", choices=["exact", "best_bnd"],
                    default="best_bnd")
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--max-cuts", type=int, default=3,
                    help="最多测试几个根节点活跃割；0 = 只报告活跃数")
    ap.add_argument("--recompute-beta", action="store_true",
                    help="逐场景重新取 LP 对偶再 lift，而不是复用保存割的 beta")
    args = ap.parse_args()

    prob = DefineProblem(load_problem_config(BASE_CFG / "problem.toml"))
    cfg = load_solver_config(BASE_CFG / "solver.toml")
    cuts, _ = load_cuts(START_RUN, prob.T, prob.state_dim)
    print(f"start cuts={sum(len(c) for c in cuts[1:prob.T + 1])} "
          f"stage1={len(cuts[1])} value={args.value}")

    lb0, theta0, xout = solve_root(prob, cfg, cuts[1])
    active = [(c, float(c.value_at(xout)))
              for c in cuts[1] if c.it > 0]
    active = [(c, v) for c, v in active
              if v >= theta0 - 1e-5]
    active.sort(key=lambda cv: cv[1] - theta0, reverse=True)
    print(f"root LB={lb0:.6f} theta={theta0:.6f} active={len(active)}")
    for c, v in active[:10]:
        print(f"  stage1 cut it={c.it} value={v:.6f} slack={v-theta0:+.3e}")

    if args.max_cuts <= 0:
        return
    candidates = [c for c, _ in active[:args.max_cuts]]
    mode = "regen" if args.recompute_beta else "savedbeta"
    d = new_run_dir(f"root_cut_lift_{args.value}_{mode}",
                    str(ROOT / cfg.results_root))
    write_configs(d, BASE_CFG / "problem.toml", BASE_CFG / "solver.toml")
    save_json(d, "problem_data.json", to_json(prob))
    print(f"result dir: {d}")

    rows = []
    changed = list(cuts[1])
    for c in candidates:
        new, st = lift_cut(prob, cfg, cuts, c, args.value, args.time_limit,
                           recompute_beta=args.recompute_beta)
        idx = next(i for i, old in enumerate(changed) if old is c)
        changed[idx] = new
        lb1, _, _ = solve_root(prob, cfg, changed)
        rec = {"stage": 1, "cut_it": int(c.it),
               "lb0": lb0, "lb1": lb1, "delta_lb": lb1 - lb0, **st}
        rows.append(rec)
        print(f"cut it={c.it}: alpha {st['alpha_old']:.3f}->"
              f"{st['alpha_new']:.3f}, fallback={st['fallback']}, "
              f"LB {lb0:.3f}->{lb1:.3f} ({lb1-lb0:+.6f}), "
              f"{st['lift_s']:.1f}s")

    lb_all, _, _ = solve_root(prob, cfg, changed)
    print(f"combined root LB={lb_all:.6f} delta={lb_all-lb0:+.6f}")
    save_json(d, "root_lift_result.json", {
        "value": args.value, "time_limit": args.time_limit,
        "lb0": lb0, "theta0": theta0, "n_active": len(active),
        "rows": rows, "lb_all": lb_all, "delta_all": lb_all - lb0,
    })
    save_summary(d, "\n".join([
        f"value      : {args.value}",
        f"root active: {len(active)}",
        f"LB before  : {lb0:.6f}",
        f"LB all     : {lb_all:.6f}",
        f"delta all  : {lb_all-lb0:+.6f}",
    ]) + "\n")
    print("ROOT_CUT_LIFT_DONE")


if __name__ == "__main__":
    main()
