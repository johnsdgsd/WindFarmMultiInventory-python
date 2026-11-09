# _t_policy3_baseline.py — 第三项前向策略：
#   用基线 LP（base_T60，连续 N=Θn）的策略求 q/tau/y/S_end，
#   四舍五入后固定，放到 Tier2 整数模型里解一个修复 MIP；不可行则回退单阶段 MIP。
# 基线 cuts 是 v5，当前 loader 只收 v6，所以这里手工读基线 cuts（模型结构/维度相同）。
from __future__ import annotations

import argparse
import json
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
from utils.cut import Cut, pack                                  # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


def load_baseline_cuts(d: Path, T: int, dim: int):
    dct = json.load(open(d / "cuts.json", encoding="utf-8"))
    cuts = [[] for _ in range(T + 1)]
    for k, items in dct.get("cuts", {}).items():
        t = int(k)
        cuts[t] = [Cut.from_dict(c) for c in items]
    for t in range(T + 1):
        for c in cuts[t]:
            if c.dim != dim:
                raise ValueError(f"baseline cut dim {c.dim} != {dim}")
    return cuts, dct.get("meta", {})


def vals(m, vars_):
    return np.asarray(ce.values(m, vars_), dtype=float)


def policy3_path(lp_prob, lp_cfg, lp_cuts, t2_prob, t2_cfg, t2_cuts,
                 scene_path, path_idx):
    Ly = t2_prob.layout
    x0 = np.round(np.asarray(t2_prob.initial_state(), float), 6)
    # 基线 LP 的 stage 0 策略；只取 q_out 并取整
    s0 = make_stage_subproblem(0, lp_prob, lp_cfg)
    s0.assemble(pack(lp_cuts[1], t2_prob.state_dim), key=("p3b", 0, path_idx))
    s0.set_state(x0, require_int=False)
    s0.solve_lp()
    q0 = np.rint(vals(s0.model, s0.x_out[Ly.off_q:Ly.off_q + Ly.n_q]))
    s0.free()
    x = x0.copy()
    x[Ly.off_q:Ly.off_q + Ly.n_q] = q0
    total = 0.0
    fb = 0
    for t in range(1, t2_prob.T + 1):
        scene = int(scene_path[t - 1])
        # 基线 LP 决策
        pn_lp = (pack(lp_cuts[t + 1], t2_prob.state_dim) if t < t2_prob.T
                 else (np.zeros(0), np.zeros((0, t2_prob.state_dim))))
        s = make_stage_subproblem(t, lp_prob, lp_cfg)
        s.assemble(pn_lp, key=("p3b", t, scene, "lp"))
        s.set_state(x, require_int=False)
        s.set_scene(lp_prob.pool[scene])
        try:
            s.solve_lp()
        except Exception:
            s.free()
            raise
        qv = np.rint(vals(s.model, s.x_out[Ly.off_q:Ly.off_q + Ly.n_q]))
        tvp = np.rint(vals(s.model, s.taup))
        tvm = np.rint(vals(s.model, s.taum))
        yv = np.rint(vals(s.model, [v for row in s.y for v in row]))
        sv = np.rint(vals(s.model, s.S_end))
        s.free()

        # Tier2 修复 MIP：固定基线 LP 的物流决策，N/n/I/o 由 Tier2 整数模型填
        pn2 = (pack(t2_cuts[t + 1], t2_prob.state_dim) if t < t2_prob.T
               else (np.zeros(0), np.zeros((0, t2_prob.state_dim))))
        sub = make_stage_subproblem(t, t2_prob, t2_cfg)
        sub.assemble(pn2, key=("p3b", t, scene, "repair"))
        sub.set_state(x, require_int=True)
        sub.set_scene(t2_prob.pool[scene])
        fix = (list(sub.x_out[Ly.off_q:Ly.off_q + Ly.n_q])
               + list(sub.taup) + list(sub.taum)
               + [v for row in sub.y for v in row] + list(sub.S_end))
        fv = list(qv) + list(tvp) + list(tvm) + list(yv) + list(sv)
        try:
            sub.model.setInfo(cp.COPT.Info.LB, fix, [float(v) for v in fv])
            sub.model.setInfo(cp.COPT.Info.UB, fix, [float(v) for v in fv])
            sub.solve_mip()
        except Exception:
            fb += 1
            sub = make_stage_subproblem(t, t2_prob, t2_cfg)
            sub.assemble(pn2, key=("p3b", t, scene, "fallback"))
            sub.set_state(x, require_int=True)
            sub.set_scene(t2_prob.pool[scene])
            sub.solve_mip()
        total += float(sub.stage_cost())
        x = np.asarray(sub.out_state_vector(), dtype=float)
        sub.free()
    return total, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2-dir",
                    default="results/int_t2_resume2h_final2_20261108_213100")
    ap.add_argument("--lp-dir",
                    default="results/base_T60_20261026_204105")
    ap.add_argument("--paths", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    t2d = ROOT / args.t2_dir
    lpd = ROOT / args.lp_dir
    t2_prob = DefineProblem(load_problem_config(t2d / "config" / "problem.toml"))
    lp_prob = DefineProblem(load_problem_config(lpd / "config" / "problem.toml"))
    t2_cfg = load_solver_config(t2d / "config" / "solver.toml")
    lp_cfg = load_solver_config(lpd / "config" / "solver.toml")
    t2_cuts, t2_meta = load_cuts(t2d, t2_prob.T, t2_prob.state_dim)
    lp_cuts, lp_meta = load_baseline_cuts(lpd, lp_prob.T, lp_prob.state_dim)
    if lp_prob.T != t2_prob.T or lp_prob.state_dim != t2_prob.state_dim:
        raise SystemExit("baseline/Tier2 dimensions differ")
    rng = np.random.default_rng(args.seed)
    sps = [rng.integers(0, len(t2_prob.pool), size=t2_prob.T).tolist()
           for _ in range(args.paths)]
    print(f"Tier2 cuts {t2_meta.get('n_iterations')} / baseline cuts "
          f"{lp_meta.get('n_iterations')}; paths={args.paths}", flush=True)
    totals = []
    fb = 0
    t0 = time.time()
    for i, sp in enumerate(sps):
        val, f = policy3_path(lp_prob, lp_cfg, lp_cuts, t2_prob, t2_cfg,
                              t2_cuts, sp, i)
        fb += f
        totals.append(val)
        print(f"  path {i+1}/{len(sps)} total={val:.4f}", flush=True)
    arr = np.asarray(totals)
    se = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    print(f"policy3 baselineLP+round: mean={arr.mean():.4f} "
          f"std={arr.std(ddof=1) if arr.size>1 else 0:.4f} se={se:.4f} "
          f"fallback={fb} time={time.time()-t0:.1f}s")
    np.savez(ROOT / "test" / "_policy3_baseline_totals.npz", totals=arr)


if __name__ == "__main__":
    main()
