# _t_rootcut_sample.py — 多状态/多场景统计 NodeLimit=0 根节点割的强度与耗时。
#
# 复用 _t_rootcut_probe.py 的单点流程；只做诊断，不写 results/。
# 对每个 (stage, state, scene)：
#   LP 松弛自由状态 = lp_free
#   根节点 L0 / L3 的 BestBnd
#   精确 B&B（短时限）的 BestBnd/BestObj
# 统计根节点相对短时限 B&B 的截距捕捉率和时间比。
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
sys.stdout.reconfigure(errors="replace")

from _t_rootcut_probe import (fail_prob, load_problem_config,          # noqa: E402
                              load_solver_config, DefineProblem, load_cuts,
                              run_point)


def quantile_states(cuts, t: int, n: int):
    cs = [(i, c) for i, c in enumerate(cuts[t])
          if np.count_nonzero(c.state) > 0 and np.linalg.norm(c.beta) > 1e-9]
    if not cs:
        return []
    cs.sort(key=lambda ic: float(np.linalg.norm(ic[1].beta)))
    if n >= len(cs):
        picks = cs
    else:
        qs = np.linspace(0.1, 0.9, n)
        picks = [cs[int(round(q * (len(cs) - 1)))] for q in qs]
    return [(idx, np.asarray(c.state, float), float(np.linalg.norm(c.beta)))
            for idx, c in picks]


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--result-dir",
                    default="results/int_t2_formal_20261106_180256")
    ap.add_argument("--stages", default="5,10,15,20,25,30,35,40,45,50,55")
    ap.add_argument("--states-per-stage", type=int, default=3)
    ap.add_argument("--scenes", default="hf,med,low")
    ap.add_argument("--root-time", type=float, default=20.0)
    ap.add_argument("--exact-time", type=float, default=2.0)
    ap.add_argument("--root-cut-level", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default="test/_rootcut_sample")
    args = ap.parse_args()

    base = Path(args.result_dir)
    if not base.is_absolute():
        base = ROOT / base
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    run_dir = out / time.strftime("sample_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    pcfg = load_problem_config(base / "config" / "problem.toml")
    scfg = load_solver_config(base / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(base, prob.T, prob.state_dim)

    fails = np.array([fail_prob(prob, i) for i in range(len(prob.pool))])
    hf = int(np.argmax(fails))
    med = int(np.argsort(fails)[len(fails) // 2])
    low = int(np.argsort(fails)[len(fails) // 10])
    smap = {"hf": hf, "med": med, "low": low}
    scenes = [(s, smap[s.strip()]) for s in args.scenes.split(",") if s.strip()]
    stages = [int(x) for x in args.stages.split(",") if x.strip()]

    print(f"result={base}")
    print(f"scenes: " + ", ".join(f"{k}={v}({fails[v]:.4g})" for k, v in scenes))
    print(f"stages={stages} states/stage={args.states_per_stage} "
          f"root L{args.root_cut_level} exact={args.exact_time:g}s")
    print(f"out={run_dir}")

    rows = []
    for t in stages:
        picks = quantile_states(cuts, t, args.states_per_stage)
        print(f"\nstage {t}: {len(picks)} states")
        for sidx, x, bn in picks:
            for name, scene in scenes:
                recs, lp_free, exact = run_point(
                    prob, scfg, cuts, t, sidx, x, bn, scene,
                    [0, args.root_cut_level], args.rounds,
                    args.root_time, args.exact_time)
                for r in recs:
                    r["scene_name"] = name
                    r["fail_prob"] = float(fails[scene])
                rows.extend(recs)
                ex = exact
                r3 = next((r for r in recs if r["mode"] == "root"
                           and r["level"] == args.root_cut_level), None)
                r0 = next((r for r in recs if r["mode"] == "root"
                           and r["level"] == 0), None)
                rg = (r3["best_bnd"] - lp_free) if r3 and r3.get("best_bnd") else float("nan")
                eg = ((ex["best_obj"] - lp_free) if ex and ex.get("best_obj") is not None
                      else float("nan"))
                print(f"  idx={sidx:5d} {name:>3} lp_free={lp_free:.3f} "
                      f"root0={r0.get('best_bnd') if r0 else None} "
                      f"root3={r3.get('best_bnd') if r3 else None} "
                      f"exact={ex.get('best_obj') if ex else None} "
                      f"gain={rg:.5f}/{eg:.5f} "
                      f"t={r3.get('time_s') if r3 else None:.3f}/"
                      f"{ex.get('time_s') if ex else None:.3f}s")

    keys = ["stage", "scene", "scene_name", "state_idx", "beta_norm", "x_nz",
            "fail_prob", "lp_fixed", "lp_free", "lp_time", "fixed_mip_ok",
            "fixed_mip_obj", "fixed_mip_time", "mode", "level", "rounds",
            "time_s", "status", "mip_status", "nodes", "has_sol", "best_obj",
            "best_bnd", "best_gap", "solving_time", "lp_obj", "beta_dot_x",
            "bound_minus_lpfree", "exact_bnd", "exact_obj", "error"]
    csv_path = run_dir / "rootcut_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 汇总 ----
    pts = {}
    for r in rows:
        pts.setdefault((r["stage"], r["scene"], r["state_idx"]), []).append(r)
    stats = []
    for k, rs in pts.items():
        ex = next((r for r in rs if r["mode"] == "exact"), None)
        r3 = next((r for r in rs if r["mode"] == "root"
                   and r["level"] == args.root_cut_level), None)
        r0 = next((r for r in rs if r["mode"] == "root" and r["level"] == 0), None)
        if not ex or not r3 or ex.get("best_obj") is None or r3.get("best_bnd") is None:
            continue
        lp = float(r3["lp_free"])
        rg = float(r3["best_bnd"]) - lp
        eg_hi = float(ex["best_obj"]) - lp
        eg_lo = (float(ex["best_bnd"]) - lp
                 if ex.get("best_bnd") is not None else float("nan"))
        stats.append({"stage": k[0], "scene_name": next(
            r["scene_name"] for r in rs if "scene_name" in r),
            "state_idx": k[2], "lp": lp, "root_gain": rg,
            "exact_lo": eg_lo, "exact_hi": eg_hi,
            "root_t": float(r3["time_s"]), "exact_t": float(ex["time_s"]),
            "root_bnd": float(r3["best_bnd"]),
            "exact_bnd": ex.get("best_bnd"), "exact_obj": ex.get("best_obj"),
            "root0_bnd": float(r0["best_bnd"]) if r0 and r0.get("best_bnd") else None})

    def summary(sub, name):
        if not sub:
            return
        n = len(sub)
        cap_hi = np.array([s["root_gain"] / s["exact_hi"]
                           if abs(s["exact_hi"]) > 1e-12 else np.nan
                           for s in sub])
        cap_lo = np.array([s["root_gain"] / s["exact_lo"]
                           if abs(s["exact_lo"]) > 1e-12 else np.nan
                           for s in sub])
        print(f"\n[{name}] n={n}  "
              f"mean gain root={np.nanmean([s['root_gain'] for s in sub]):.5f} "
              f"exact_hi={np.nanmean([s['exact_hi'] for s in sub]):.5f}")
        print(f"  capture vs exact_hi: median={np.nanmedian(cap_hi):.1%} "
              f"mean={np.nanmean(cap_hi):.1%} "
              f">50%={np.nanmean(cap_hi > 0.5):.0%}  >90%={np.nanmean(cap_hi > 0.9):.0%}")
        print(f"  capture vs exact_lo: median={np.nanmedian(cap_lo):.1%} "
              f"mean={np.nanmean(cap_lo):.1%}")
        print(f"  time root mean/med={np.mean([s['root_t'] for s in sub]):.3f}/"
              f"{np.median([s['root_t'] for s in sub]):.3f}s  "
              f"exact mean/med={np.mean([s['exact_t'] for s in sub]):.3f}/"
              f"{np.median([s['exact_t'] for s in sub]):.3f}s")

    summary(stats, "all")
    for st in stages:
        summary([s for s in stats if s["stage"] == st], f"stage {st}")
    for nm, _ in scenes:
        summary([s for s in stats if s["scene_name"] == nm], f"scene {nm}")

    srows = []
    for k in ["stage", "scene_name", "state_idx", "lp", "root_gain",
              "exact_lo", "exact_hi", "root_t", "exact_t", "root_bnd",
              "exact_bnd", "exact_obj", "root0_bnd"]:
        srows.append(k)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=srows)
        w.writeheader()
        w.writerows(stats)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
