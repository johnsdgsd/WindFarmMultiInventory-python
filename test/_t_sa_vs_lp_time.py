# _t_sa_vs_lp_time.py — 同一状态/场景/割集下比较 SA 与 LP 的耗时和目标。
#
# 这是模型一致性门槛：SA 是启发式，目标不能低于同点 LP 最优；若低于，说明
# SA 的约束/成本/theta 口径与 LP 不一致。时间同时覆盖 LP 求解和 SA 搜索。
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.forward_heuristic import heuristic_step                # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.sim_analysis import check_stage                        # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

RUN = ROOT / "results" / "v2_static10_4h_20261110_223050"
CFG = ROOT / "test" / "_cfg_v2_heur"


def _set_I(x, Ly, I):
    y = x.copy()
    y[:Ly.n_I] = np.asarray(I, float)
    return y


def _set_n(x, Ly, n):
    y = x.copy()
    y[Ly.off_n:Ly.off_n + Ly.n_n] = np.asarray(n, float).ravel()
    return y


def _states(prob):
    Ly = prob.layout
    x0 = prob.initial_state()
    n0 = Ly.n(x0)
    all0 = np.zeros_like(n0)
    all0[:, 0] = prob.N
    return [("初始", x0),
            ("库存满", _set_I(x0, Ly, prob.Imax)),
            ("W全量", _set_n(x0, Ly, np.zeros_like(n0))),
            ("全0级", _set_n(x0, Ly, all0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--config-dir", default=str(CFG))
    args = ap.parse_args()

    run = Path(args.run)
    cdir = Path(args.config_dir)
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(run, prob.T, prob.state_dim)
    for t, cs in enumerate(cuts):
        for i, c in enumerate(cs):
            c.uid = ("time", t, i)

    Ly = prob.layout
    lp_times, sa_times, gaps = [], [], []
    rows = []
    for sname, x in _states(prob):
        for t in (1, 15, 30, 45, prob.T - prob.L,
                  prob.T - prob.L + 1, prob.T):
            t = max(1, min(int(t), prob.T))
            cc = cuts[t + 1] if t + 1 < len(cuts) else []
            for si in (0, 7, 19):
                scene = prob.pool[si]
                sub = make_stage_subproblem(1, prob, scfg)
                sub.set_cuts(cc)
                sub.set_point(x, scene)
                # 热启动一次，再计时 LP；SA 不依赖模型热启动。
                sub.solve_lp()
                lp_vals = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    sub.solve_lp()
                    lp_vals.append(time.perf_counter() - t0)
                lp_obj = float(sub.obj)
                sa_vals, sa_objs, sa_ok = [], [], []
                for k in range(args.reps):
                    t0 = time.perf_counter()
                    h = heuristic_step(
                        prob=prob, t=t, x=x, scene=scene, cfg=scfg,
                        cuts_next=cc,
                        rng=np.random.default_rng(1000 + 17 * k + si))
                    sa_vals.append(time.perf_counter() - t0)
                    sa_objs.append(float(h.objective))
                    aut = check_stage(
                        prob, Ly, x, h.x_out, h.q_out, h.taup, h.taum,
                        h.S_end, h.y, si, Ly.upper(prob), F_new=h.F_new,
                        integer_mode=False, integer_scope="none", N=None)
                    tol = max(float(prob.state_repair_tol), 1e-6)
                    bad = {k2: v for k2, v in aut.items() if v > tol}
                    sa_ok.append(not bad)
                sa_obj = float(np.median(sa_objs))
                gap = sa_obj - lp_obj
                if not all(sa_ok):
                    raise AssertionError(f"{sname} t={t} si={si}: "
                                         f"SA 解违反 LP 约束")
                # SA 不能优于精确 LP 最优（浮点容差内）。
                if gap < -1e-6 * max(1.0, abs(lp_obj)):
                    raise AssertionError(
                        f"{sname} t={t} si={si}: SA_obj {sa_obj} < "
                        f"LP_obj {lp_obj}（模型不一致）")
                lp_t = float(np.median(lp_vals))
                sa_t = float(np.median(sa_vals))
                lp_times.append(lp_t)
                sa_times.append(sa_t)
                gaps.append(gap)
                rows.append((sname, t, si, lp_obj, sa_obj, gap,
                             lp_t, sa_t))
                sub.free()

    lp = np.asarray(lp_times)
    sa = np.asarray(sa_times)
    gp = np.asarray(gaps)
    print(f"\npoints={len(rows)} reps={args.reps}")
    print("LP time  mean/median/p95 = "
          f"{lp.mean()*1e3:.3f}/{np.median(lp)*1e3:.3f}/"
          f"{np.percentile(lp,95)*1e3:.3f} ms")
    print("SA time  mean/median/p95 = "
          f"{sa.mean()*1e3:.3f}/{np.median(sa)*1e3:.3f}/"
          f"{np.percentile(sa,95)*1e3:.3f} ms")
    print(f"SA/LP time ratio median = {np.median(sa/lp):.3f}")
    print(f"SA objective gap mean/median/max = "
          f"{gp.mean():.6f}/{np.median(gp):.6f}/{gp.max():.6f}")
    print("SA time by stage (median ms):")
    for t in sorted({r[1] for r in rows}):
        vals = [r[7] for r in rows if r[1] == t]
        print(f"  t={t:3d}  {np.median(vals)*1e3:.3f}")
    print("SA_VS_LP_OK")


if __name__ == "__main__":
    main()
