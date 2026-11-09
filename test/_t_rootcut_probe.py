# _t_rootcut_probe.py — 在 Tier2 割提升子问题上测 COPT 的“只解根节点”MILP。
#
# 问题：solve_lift 的 Strengthened Benders 要解 min_x [f(x) - beta·x] 的整数
# 规划；正式训练里精确 B&B 可能几十秒。COPT 的 NodeLimit=0 是否能在根节点做完
# 割强化后停下，并给出可用的 BestBnd（合法下界）？
#
# 本脚本只做探针，不改训练链路、不写 results/：
#   - 从结果目录读 cuts/problem/solver 配置（只读）；
#   - 取阶段 t 的试探点 x 和 cuts[t+1]，复现一次后向 LP 得到 beta；
#   - 固定状态解一次 MIP 作为 warm start（与 worker 的 solve_lift 相同）；
#   - 对每个 RootCutLevel 用 NodeLimit=0 解，记录 BestBnd/BestObj/时间；
#   - 另跑一次完整 B&B 作为参照，检查根节点 BestBnd 是否是合法下界。
#
# 用法：
#   python test/_t_rootcut_probe.py
#   python test/_t_rootcut_probe.py --stages 10,30,54,60 --scenes 0,hf
#   python test/_t_rootcut_probe.py --exact-time 60 --root-time 20
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import coptpy as cp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import copt_env as ce                                      # noqa: E402
from utils.cut import pack                                            # noqa: E402
from utils.io import load_cuts                                        # noqa: E402


def finite(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def get_attr(m, attr):
    try:
        return m.getAttr(attr)
    except Exception:
        return None


def fail_prob(prob, scene_idx: int) -> float:
    """按初始等级分布加权的失效概率；只用于挑高失效场景。"""
    th = np.asarray(prob.pool[scene_idx][1], dtype=float)
    mix = np.asarray(prob.init_mix, dtype=float)
    return float(np.mean((th[:, :, prob.G] * mix[None, :]).sum(axis=1)))


def choose_state(cuts, t: int, how: str = "median"):
    """在 cuts[t] 里挑一个非平凡试探点（按 beta 范数）。"""
    cs = [(i, c) for i, c in enumerate(cuts[t])
          if np.count_nonzero(c.state) > 0 and np.linalg.norm(c.beta) > 1e-9]
    if not cs:
        raise RuntimeError(f"stage {t}: 没有非平凡割状态")
    cs.sort(key=lambda ic: float(np.linalg.norm(ic[1].beta)))
    if how == "first":
        idx, c = cs[0]
    elif how == "max":
        idx, c = cs[-1]
    else:
        idx, c = cs[len(cs) // 2]
    return np.asarray(c.state, dtype=float), int(idx), float(np.linalg.norm(c.beta))


def empty_pack(dim: int):
    return np.zeros(0), np.zeros((0, dim))


def new_sub(prob, scfg, t, pack_next, x, scene_idx, key):
    sub = make_stage_subproblem(t, prob, scfg)
    sub.assemble(pack_next, key=key)
    sub.set_state(x, require_int=True)
    sub.set_scene(prob.pool[scene_idx])
    return sub


def free_state_lift_model(sub, beta, prob):
    """释放 x_in，把目标改为 f(z) - beta·x_in（与 solve_lift 同一口径）。"""
    Ly = prob.layout
    m = sub.model
    xbar = np.asarray(sub._state_vec, dtype=float)
    ub = np.asarray(Ly.upper(prob), dtype=float)
    cs = np.asarray(sub.state_cost(), dtype=float)
    m.setInfo(cp.COPT.Info.LB, sub.x_in, [0.0] * Ly.dim)
    m.setInfo(cp.COPT.Info.UB, sub.x_in, [float(v) for v in ub])
    m.setInfo(cp.COPT.Info.Obj, sub.x_in,
              [float(cs[j] - beta[j]) for j in range(Ly.dim)])
    return m, xbar


def solve_variant(prob, scfg, t, pack_next, x, scene_idx, beta, mip_start,
                  *, mode: str, level: int, rounds: int, time_limit: float,
                  key):
    sub = new_sub(prob, scfg, t, pack_next, x, scene_idx, key)
    m, xbar = free_state_lift_model(sub, beta, prob)
    if mip_start is not None:
        try:
            m.setMipStart(list(m.getVars()), mip_start)
        except Exception:
            pass
    if mode == "root":
        m.setParam(cp.COPT.Param.NodeLimit, 0)
        m.setParam(cp.COPT.Param.RootCutLevel, int(level))
        m.setParam(cp.COPT.Param.RootCutRounds, int(rounds))
    # mode == exact：保持 COPT 默认的 NodeLimit=-1 / RootCutLevel=-1
    m.setParam(cp.COPT.Param.RelGap, 0.0)
    m.setParam(cp.COPT.Param.AbsGap, 0.0)
    m.setParam(cp.COPT.Param.TimeLimit, float(time_limit))
    t0 = time.time()
    try:
        m.solve()
    except Exception as exc:
        return {"mode": mode, "level": level, "rounds": rounds,
                "time_s": time.time() - t0, "error": f"{type(exc).__name__}: {exc}"}
    dt = time.time() - t0
    rec = {
        "mode": mode, "level": level, "rounds": rounds,
        "time_s": dt,
        "status": int(m.status),
        "mip_status": finite(get_attr(m, cp.COPT.Attr.MipStatus)),
        "nodes": finite(get_attr(m, cp.COPT.Attr.NodeCnt)),
        "has_sol": finite(get_attr(m, cp.COPT.Attr.HasMipSol)),
        "best_obj": finite(get_attr(m, cp.COPT.Attr.BestObj)),
        "best_bnd": finite(get_attr(m, cp.COPT.Attr.BestBnd)),
        "best_gap": finite(get_attr(m, cp.COPT.Attr.BestGap)),
        "solving_time": finite(get_attr(m, cp.COPT.Attr.SolvingTime)),
        "lp_obj": finite(get_attr(m, cp.COPT.Attr.LpObjVal)),
        "beta_dot_x": float(beta @ xbar),
    }
    try:
        m.setParam(cp.COPT.Param.NodeLimit, -1)
        m.setParam(cp.COPT.Param.RootCutLevel, -1)
        m.setParam(cp.COPT.Param.RootCutRounds, -1)
        m.setParam(cp.COPT.Param.TimeLimit, 1e20)
    except Exception:
        pass
    return rec


def run_point(prob, scfg, cuts, t, state_idx, x, beta_norm, scene_idx,
              level_list, rounds, root_time, exact_time):
    dim = prob.state_dim
    pack_next = (pack(cuts[t + 1], dim) if t < prob.T else empty_pack(dim))
    key0 = ("probe", t, scene_idx, state_idx, "lp")
    sub = new_sub(prob, scfg, t, pack_next, x, scene_idx, key0)
    t0 = time.time()
    sub.solve_lp()
    lp_fixed = float(sub.obj)
    beta = np.asarray(sub.gradient("dj"), dtype=float)
    lp_time = time.time() - t0
    lp_free = lp_fixed - float(beta @ x)

    # 与 worker.solve_lift 相同：固定状态先解整数子问题，作为释放状态 MIP 的起点。
    t0 = time.time()
    fixed_mip_ok = True
    try:
        sub.solve_mip()
        fixed_mip_obj = float(sub.obj)
        fixed_vals = ce.values(sub.model, list(sub.model.getVars()))
    except Exception as exc:
        fixed_mip_ok = False
        fixed_mip_obj = None
        fixed_vals = None
        fixed_err = f"{type(exc).__name__}: {exc}"
    fixed_time = time.time() - t0

    recs = []
    for lvl in level_list:
        rec = solve_variant(
            prob, scfg, t, pack_next, x, scene_idx, beta, fixed_vals,
            mode="root", level=lvl, rounds=rounds, time_limit=root_time,
            key=("probe", t, scene_idx, state_idx, "root", lvl))
        recs.append(rec)
    exact = solve_variant(
        prob, scfg, t, pack_next, x, scene_idx, beta, fixed_vals,
        mode="exact", level=-1, rounds=-1, time_limit=exact_time,
        key=("probe", t, scene_idx, state_idx, "exact"))
    recs.append(exact)

    base = {"stage": t, "scene": scene_idx, "state_idx": state_idx,
            "beta_norm": beta_norm, "lp_fixed": lp_fixed,
            "lp_free": lp_free, "lp_time": lp_time,
            "fixed_mip_ok": fixed_mip_ok, "fixed_mip_obj": fixed_mip_obj,
            "fixed_mip_time": fixed_time, "x_nz": int(np.count_nonzero(x))}
    for r in recs:
        r.update(base)
        r["bound_minus_lpfree"] = (None if r.get("best_bnd") is None
                                   else float(r["best_bnd"] - lp_free))
        r["exact_bnd"] = exact.get("best_bnd") if r["mode"] == "root" else None
        r["exact_obj"] = exact.get("best_obj") if r["mode"] == "root" else None
    return recs, lp_free, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir",
                    default="results/int_t2_formal_20261106_180256")
    ap.add_argument("--stages", default="10,30,54,60")
    ap.add_argument("--scenes", default="0,hf",
                    help="逗号分隔场景下标；hf = 按初始等级加权的最高失效概率场景")
    ap.add_argument("--state-select", choices=["first", "median", "max"],
                    default="median")
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--root-time", type=float, default=20.0)
    ap.add_argument("--exact-time", type=float, default=30.0)
    ap.add_argument("--out", default="test/_rootcut_probe")
    args = ap.parse_args()

    base = Path(args.result_dir)
    if not base.is_absolute():
        base = ROOT / base
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    run_dir = out / time.strftime("rootcut_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    pcfg = load_problem_config(base / "config" / "problem.toml")
    scfg = load_solver_config(base / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, meta = load_cuts(base, prob.T, prob.state_dim)

    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    scenes = []
    hf = int(np.argmax([fail_prob(prob, i) for i in range(len(prob.pool))]))
    for s in args.scenes.split(","):
        s = s.strip()
        if s:
            scenes.append(hf if s == "hf" else int(s))
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"结果目录 : {base}")
    print(f"T={prob.T} 状态维={prob.state_dim} 割数={sum(map(len, cuts))} "
          f"高失效场景={hf} ({fail_prob(prob, hf):.6g})")
    print(f"阶段 {stages}  场景 {scenes}  状态选择 {args.state_select}")
    print(f"根节点 levels={levels} rounds={args.rounds} time={args.root_time:g}s；"
          f"精确 B&B time={args.exact_time:g}s")
    print(f"输出 {run_dir}")

    rows = []
    for t in stages:
        x, sidx, bn = choose_state(cuts, t, args.state_select)
        print(f"\n阶段 {t}: state_idx={sidx} beta_norm={bn:.3f} x_nz={np.count_nonzero(x)}")
        for scene in scenes:
            recs, lp_free, exact = run_point(
                prob, scfg, cuts, t, sidx, x, bn, scene,
                levels, args.rounds, args.root_time, args.exact_time)
            rows.extend(recs)
            eb = exact.get("best_bnd")
            eo = exact.get("best_obj")
            print(f"  场景 {scene}: LP free={lp_free:.6f}  "
                  f"fixed MIP={recs[0]['fixed_mip_time']:.3f}s")
            for r in recs:
                if "error" in r:
                    print(f"    {r['mode']} L{r['level']}: ERROR {r['error']}")
                    continue
                label = ("exact" if r["mode"] == "exact"
                         else f"root L{r['level']}")
                print(f"    {label:<8} t={r['time_s']:7.3f}s  "
                      f"status={r['status']} mip={r['mip_status']} "
                      f"nodes={r['nodes']}  obj={r['best_obj']}  "
                      f"bnd={r['best_bnd']}  gap={r['best_gap']}")
            if eb is not None and eo is not None:
                print(f"    exact ref: bnd={eb:.6f} obj={eo:.6f}"
                      f" gain_bnd={eb-lp_free:.6f} gain_obj={eo-lp_free:.6f}")

    # ---------------------------------------------------------------- 落盘
    keys = ["stage", "scene", "state_idx", "beta_norm", "x_nz",
            "lp_fixed", "lp_free", "lp_time", "fixed_mip_ok", "fixed_mip_obj",
            "fixed_mip_time", "mode", "level", "rounds", "time_s", "status",
            "mip_status", "nodes", "has_sol", "best_obj", "best_bnd",
            "best_gap", "solving_time", "lp_obj", "beta_dot_x",
            "bound_minus_lpfree", "exact_bnd", "exact_obj", "error"]
    csv_path = run_dir / "rootcut_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 简洁汇总：每个点只看 L3 根节点与 exact 的对比。
    lines = []
    for t in stages:
        for scene in scenes:
            pt = [r for r in rows if r["stage"] == t and r["scene"] == scene]
            if not pt:
                continue
            lp_free = pt[0]["lp_free"]
            ex = next((r for r in pt if r["mode"] == "exact"), None)
            r3 = next((r for r in pt if r["mode"] == "root" and r["level"] == 3), None)
            lines.append(f"stage={t} scene={scene} lp_free={lp_free:.6f} "
                         f"exact_bnd={None if ex is None else ex.get('best_bnd')} "
                         f"exact_obj={None if ex is None else ex.get('best_obj')} "
                         f"rootL3_bnd={None if r3 is None else r3.get('best_bnd')} "
                         f"rootL3_time={None if r3 is None else r3.get('time_s')} "
                         f"exact_time={None if ex is None else ex.get('time_s')}")
    summary = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({
        "result_dir": str(base), "stages": stages, "scenes": scenes,
        "high_fail_scene": hf, "levels": levels, "rounds": args.rounds,
        "root_time": args.root_time, "exact_time": args.exact_time,
        "state_select": args.state_select,
        "n_cuts": sum(map(len, cuts)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + summary)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
