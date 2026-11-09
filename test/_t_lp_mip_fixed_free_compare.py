# _t_lp_mip_fixed_free_compare.py — 同一阶段/状态/场景下比较：
#   LP 固定状态
#   固定状态 MILP
#   释放状态 MILP（Lagrangian：目标 f - β·x_in）
#
# 只读 Tier2 正式结果，不写 results/。beta 来自固定状态 LP 的对偶。
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coptpy as cp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils import copt_env as ce                                 # noqa: E402
from utils.cut import pack                                       # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


def make_sub(prob, scfg, cuts, t, x, scene, key):
    sub = make_stage_subproblem(t, prob, scfg)
    pn = (pack(cuts[t + 1], prob.state_dim) if t < prob.T
          else (np.zeros(0), np.zeros((0, prob.state_dim))))
    sub.assemble(pn, key=key)
    sub.set_state(x, require_int=True)
    sub.set_scene(prob.pool[scene])
    return sub


def solve_mip(m, *, time_limit, value="best_bnd"):
    info = ce.solve_bound(m, time_limit=time_limit, rel_gap=0.0, abs_gap=0.0,
                          value=value, return_info=True)
    return info


def free_state(m, sub, beta, prob, *, lift: bool):
    """释放 x_in；lift=True 时目标改为 f - beta·x。"""
    Ly = prob.layout
    ub = np.asarray(Ly.upper(prob), dtype=float)
    cs = np.asarray(sub.state_cost(), dtype=float)
    m.setInfo(cp.COPT.Info.LB, sub.x_in, [0.0] * Ly.dim)
    m.setInfo(cp.COPT.Info.UB, sub.x_in, [float(v) for v in ub])
    obj = cs - beta if lift else cs
    m.setInfo(cp.COPT.Info.Obj, sub.x_in,
              [float(obj[j]) for j in range(Ly.dim)])


def one_point(prob, scfg, cuts, t, x, scene, label,
              fixed_limit=20.0, free_limit=30.0):
    key = ("lp_mip", t, scene, label)
    sub = make_sub(prob, scfg, cuts, t, x, scene, key)
    sub.solve_lp()
    lp_fixed = float(sub.obj)
    beta = np.asarray(sub.gradient("dj"), dtype=float)
    bx = float(beta @ x)
    lp_free = lp_fixed - bx

    # 固定状态 MILP
    sub_f = make_sub(prob, scfg, cuts, t, x, scene, key + ("fixed",))
    sub_f.solve_lp()                      # 也为固定 MILP 提供基/热启动
    beta_f = np.asarray(sub_f.gradient("dj"), dtype=float)
    assert np.allclose(beta, beta_f)
    t0 = time.perf_counter()
    info_fixed = solve_mip(sub_f.model, time_limit=fixed_limit)
    t_fixed = time.perf_counter() - t0
    fixed_bnd = info_fixed.get("best_bnd")
    fixed_obj = info_fixed.get("best_obj")

    # 释放状态 MILP：Lagrangian 目标 f - beta·x
    sub_l = make_sub(prob, scfg, cuts, t, x, scene, key + ("lift",))
    sub_l.solve_lp()
    free_state(sub_l.model, sub_l, beta, prob, lift=True)
    t0 = time.perf_counter()
    info_lift = solve_mip(sub_l.model, time_limit=free_limit)
    t_lift = time.perf_counter() - t0
    lift_bnd = info_lift.get("best_bnd")
    lift_obj = info_lift.get("best_obj")

    def add(v, b):
        return None if v is None else float(v + b)

    rec = {
        "stage": t, "scene": scene, "label": label,
        "x_nz": int(np.count_nonzero(x)), "beta_norm": float(np.linalg.norm(beta)),
        "beta_dot_x": bx,
        "lp_fixed": lp_fixed, "lp_free": lp_free,
        "fixed_bnd": fixed_bnd, "fixed_obj": fixed_obj,
        "fixed_gap": info_fixed.get("best_gap"),
        "fixed_status": info_fixed.get("status"),
        "fixed_time": t_fixed, "fixed_exact": info_fixed.get("exact_ok"),
        "fixed_intercept_bnd": add(fixed_bnd, -bx),
        "fixed_intercept_obj": add(fixed_obj, -bx),
        "lift_bnd": lift_bnd, "lift_obj": lift_obj,
        "lift_gap": info_lift.get("best_gap"),
        "lift_status": info_lift.get("status"),
        "lift_time": t_lift, "lift_exact": info_lift.get("exact_ok"),
        "lift_cut_bnd": add(lift_bnd, bx),
        "lift_cut_obj": add(lift_obj, bx),
        "lp_free_minus_lp": lp_free - lp_fixed,
        "lift_gain_bnd": (None if lift_bnd is None else lift_bnd - lp_free),
        "lift_gain_obj": (None if lift_obj is None else lift_obj - lp_free),
    }
    return rec


def main():
    base = ROOT / "results" / "int_t2_formal_20261106_180256"
    pcfg = load_problem_config(base / "config" / "problem.toml")
    scfg = load_solver_config(base / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, meta = load_cuts(base, prob.T, prob.state_dim)

    cases = []
    # stage 1：取 it=193 的锚点（与 docs/割的提升.md 的严格反例同一阶段）
    c1 = [c for c in cuts[1] if c.it == 193]
    if c1:
        cases.append((1, np.asarray(c1[0].state, dtype=float), 0,
                      "stage1_it193"))
    # stage 30：取非平凡割中 beta 范数中位数的状态
    cs = [c for c in cuts[30] if c.it > 0 and np.linalg.norm(c.beta) > 1e-9]
    cs.sort(key=lambda c: float(np.linalg.norm(c.beta)))
    c30 = cs[len(cs) // 2]
    cases.append((30, np.asarray(c30.state, dtype=float), 0, "stage30_median"))

    print(f"T={prob.T} 维={prob.state_dim} 场景池={len(prob.pool)}")
    print(f"固定 MILP 时限={scfg.mip_time_limit:g}s（本测试用 20s）"
          f"，自由 MILP 时限 30s")
    recs = []
    for t, x, scene, label in cases:
        print(f"\n=== {label}: t={t}, scene={scene}, "
              f"x_nz={np.count_nonzero(x)}", flush=True)
        rec = one_point(prob, scfg, cuts, t, x, scene, label)
        recs.append(rec)
        print(f"LP fixed        = {rec['lp_fixed']:.6f}")
        print(f"LP free (f-βx)  = {rec['lp_free']:.6f}")
        print(f"fixed MILP      = bnd {rec['fixed_bnd']}  obj {rec['fixed_obj']}  "
              f"gap {rec['fixed_gap']}  t={rec['fixed_time']:.2f}s  "
              f"exact={rec['fixed_exact']}")
        print(f"  fixed intercept (用同一β，仅诊断) = "
              f"{rec['fixed_intercept_bnd']}")
        print(f"free lift MILP  = bnd {rec['lift_bnd']}  obj {rec['lift_obj']}  "
              f"gap {rec['lift_gap']}  t={rec['lift_time']:.2f}s  "
              f"exact={rec['lift_exact']}")
        print(f"  lift cut at x = bnd {rec['lift_cut_bnd']}  "
              f"obj {rec['lift_cut_obj']}")
        print(f"  lift gain vs LP free: bnd {rec['lift_gain_bnd']}  "
              f"obj {rec['lift_gain_obj']}")

    print("\n=== 汇总（固定状态差 / 合法提升差） ===")
    for r in recs:
        fd = (None if r["fixed_bnd"] is None
              else r["fixed_bnd"] - r["lp_fixed"])
        print(f"{r['label']}: LP={r['lp_fixed']:.6f}  "
              f"fixedMIP-LP={fd}  "
              f"lift_bnd-LPfree={r['lift_gain_bnd']}  "
              f"fixed_intercept-LPfree="
              f"{(None if r['fixed_intercept_bnd'] is None else r['fixed_intercept_bnd']-r['lp_free'])}")


if __name__ == "__main__":
    main()
