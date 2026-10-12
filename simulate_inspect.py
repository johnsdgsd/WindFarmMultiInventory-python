# simulate_inspect.py — 仿真 + 变量记录：把选定节点的状态/决策写成 CSV 长表，供策略分析。
#
#   python simulate_inspect.py results/base_20261009_120000 200
#   python simulate_inspect.py results/base_20261009_120000 200 out.csv
#
# 与 simulate_only.py 的分工：那个只做统计（均值/CI/gap），这个把每条路径每一期的
# 决策与状态落成表，用来检验 base-stock 形态、调拨触发条件、机队月龄轨迹。
#
# 列（每行 = 一条路径的一期）：
#   path, stage, scene, cost, t_plus, t_minus, y_total, y_local,
#   q_0.., I_out_0.., down_0.., I_in_total, mean_age, F_total
# 其中 down_r = N_r − Σ_a n_in[r,a]（本期停机台数，直接进 Cs·down 的停机成本）。

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")   # Windows 控制台 GBK

from code.config import load_problem_config, load_solver_config         # noqa: E402
from code.problem_data import DefineProblem                             # noqa: E402
from code.subproblem import make_stage_subproblem                       # noqa: E402
from utils import SDDPEngine, simulate_paths                            # noqa: E402
from utils.io import has_configs, load_cuts                             # noqa: E402


def row_from(prob, x_in, rec, path_k, t, x0):
    Ly = prob.layout
    n_in = Ly.n(x_in)
    q = rec["q"]
    I_out = rec["I_out"]
    y = np.asarray(rec["y"], dtype=float)
    ages = np.arange(prob.A)
    tot = n_in.sum()
    mean_age = float((n_in * ages).sum() / tot) if tot > 0 else float("nan")
    F = (prob.pool[rec["scene"]][1] * n_in).sum() if rec["scene"] >= 0 else 0.0
    row = {
        "path": path_k, "stage": t, "scene": rec["scene"], "cost": rec["cost"],
        "t_plus": float(np.sum(rec["taup"])), "t_minus": float(np.sum(rec["taum"])),
        "y_total": float(y.sum()), "y_local": float(np.trace(y)) if y.ndim == 2 else 0.0,
        "I_in_total": float(np.sum(Ly.I(x_in))),
        "mean_age": round(mean_age, 4),
        "F_total": float(F),
    }
    for r in range(prob.N_R):
        row[f"q_{r}"] = float(q[r])
        row[f"I_out_{r}"] = float(I_out[r])
        row[f"down_{r}"] = float(prob.N[r] - n_in[r].sum())
        row[f"S_end_{r}"] = float(rec["S_end"][r])
    return row


def main(dir_arg: str, n_paths: int, out_arg: str | None):
    d = Path(dir_arg)
    if not d.is_absolute():
        d = ROOT / d
    if not has_configs(d):
        raise SystemExit(f"{d} 里没有 config/problem.toml —— 重构前的结果目录，"
                         f"割口径不同，不能复用（铁律 4）")
    prob = DefineProblem(load_problem_config(d / "config" / "problem.toml"))
    scfg = load_solver_config(d / "config" / "solver.toml")
    seed = scfg.seed

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=d,
                     log=lambda *a, **k: None)
    x0 = prob.initial_state().copy()
    try:
        cuts, _ = load_cuts(d, prob.T, prob.state_dim)
        eng.cuts = cuts
        rng = np.random.default_rng(seed + 777)
        scene_paths = [rng.integers(0, eng.n_scenes, size=eng.T).tolist()
                       for _ in range(n_paths)]
        print(f"仿真 {n_paths} 条路径并记录变量…")
        results = simulate_paths(eng, scene_paths, record=True)
    finally:
        eng.close()

    rows = []
    for k, (states, costs, _, recs, _) in enumerate(results):
        for t in range(1, eng.T + 1):
            # states[t-1] 就是第 t 期的入库状态（t=1 时即 x0，阶段 0 把非 q 槽钉死了）
            x_in = states[t - 1]
            rows.append(row_from(prob, x_in, recs[t - 1], k, t, x0))

    out = Path(out_arg) if out_arg else d / f"inspect_{n_paths}paths.csv"
    if not out.is_absolute() and out_arg:
        out = ROOT / out
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已写 {len(rows)} 行 → {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        raise SystemExit("用法: python simulate_inspect.py <结果目录> [路径数] [输出CSV]")
    main(args[0], int(args[1]) if len(args) >= 2 else 200,
         args[2] if len(args) >= 3 else None)
