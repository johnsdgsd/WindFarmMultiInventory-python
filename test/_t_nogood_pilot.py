# _t_nogood_pilot.py — 非支撑型 No-good 割的 10 状态离散验证。
#
# 对阶段 t 的 10 个历史整数状态，固定状态、逐场景解 MILP，取 BestBound 的
# 场景均值 ell(Sbar)；与当前割包络 V_cut(Sbar) 比较，得到 no-good 割在锚点
# 可带来的逐点下界提升。L_t=0（成本非负），因此其他状态退化为 theta>=0。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils import copt_env as ce                                  # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"
STAGE = 10
N_SCENES = 300
N_STATES = 10
TIME_LIMIT = 0.5


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    states = [c for c in cuts[STAGE] if c.it == max(x.it for x in cuts[STAGE])]
    states = states[:N_STATES]
    assert states, "没有历史割锚点"
    next_cuts = (np.array([c.alpha for c in cuts[STAGE + 1]], dtype=float),
                 np.vstack([c.beta for c in cuts[STAGE + 1]]))
    print(f"stage={STAGE}  states={len(states)}  scenes={N_SCENES}  "
          f"time_limit={TIME_LIMIT}s  next_cuts={len(next_cuts[0])}")
    print("state  V_cut       ell_mean     delta      bestgap    status")
    rows = []
    sp = StageSubproblem(STAGE, prob, scfg)
    try:
        sp.assemble(next_cuts, key=None)
        for k, c in enumerate(states):
            S = np.asarray(c.state, dtype=float)
            v_cut = float(max(z.alpha + z.beta @ S for z in cuts[STAGE]))
            bnds, objs, gaps, statuses = [], [], [], []
            for j in range(N_SCENES):
                sp.set_point(S, prob.pool[j], require_int=True)
                info = ce.solve_bound(
                    sp.model, time_limit=TIME_LIMIT, rel_gap=0.0,
                    abs_gap=0.0, value="best_bnd", return_info=True)
                if info is None or info.get("best_bnd") is None:
                    statuses.append("FAIL")
                    continue
                bnds.append(info["best_bnd"])
                objs.append(info["best_obj"])
                gaps.append(info["best_obj"] - info["best_bnd"])
                statuses.append(info["status"])
            ell = float(np.mean(bnds))
            bg = float(np.mean(gaps)) if gaps else float("nan")
            rows.append((k, v_cut, ell, ell - v_cut, bg))
            print(f"{k:5d}  {v_cut:10.4f}  {ell:10.4f}  "
                  f"{ell - v_cut:10.4f}  {bg:9.3e}  "
                  f"{sorted(set(statuses))}")
    finally:
        sp.free()
    a = np.array([r[3] for r in rows])
    print(f"\nmean_delta={a.mean():.4f}  max_delta={a.max():.4f}  "
          f"min_delta={a.min():.4f}  positive={np.mean(a > 0):.0%}")
    print("L_t=0；其余状态 no-good 割值 <=0，故这 10 条割在其他状态不提供额外信息")
    print("OK _t_nogood_pilot")


if __name__ == "__main__":
    main()
