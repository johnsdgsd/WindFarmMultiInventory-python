# compare_python_stage.py — one-stage Python Tier2 reference for Julia equivalence.
#
# Reads the exported Tier2 instance (tier2/data/problem_data.json) and the
# Python Tier2 sources, solves StageSubproblem(1) at fixed integer states and
# scene 0 with no cuts (theta = 0), and writes test/_python_stage.json.
# Does not modify Python code or results.

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from coptpy import COPT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# <python-project>/julia_proj/tier2 -> <python-project>
PY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PY_ROOT))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_solver_config  # noqa: E402
from code.problem_data import from_json  # noqa: E402
from code.subproblem import StageSubproblem  # noqa: E402


def family(name: str) -> str:
    for pref in ("Nsum_", "Tail_", "Ndef_", "grade0_", "grade_", "dem_",
                 "fleet_", "send_def_", "tau_bal", "tau_lo_", "tau_hi_",
                 "tau_src_", "inv_", "pipe_last_", "pipe_", "cap_", "pas_"):
        if name.startswith(pref):
            return pref.rstrip("_")
    return name


def col_family(name: str) -> str:
    for pref in ("xin_", "xout_", "taup_", "taum_", "y_", "send_", "N_",
                 "theta"):
        if name.startswith(pref):
            return pref.rstrip("_")
    return name


data = json.loads((ROOT / "data" / "problem_data.json").read_text(encoding="utf-8"))
prob = from_json(data)
scfg = load_solver_config(PY_ROOT / "config" / "tune" / "int_t2_formal" /
                          "solver.toml", resolve=False)
# Exact MIP so the objective is comparable to HiGHS (not the 0.1% forward gap).
scfg.mip_rel_gap = 0.0
scfg.mip_time_limit = 300.0
scfg.int_tol = 1e-6

scenes = [prob.pool[0]]
Ly = prob.layout

cases = []
# Case 1: the Python initial state (I=o=q=0, n=n0).
x0 = prob.initial_state()
# Case 2: a non-trivial state exercising inventory, pipeline and transfers.
x1 = prob.initial_state()
x1[Ly.off_I:Ly.off_I + Ly.n_I] = [2, 1, 2, 1, 1]
for l in range(Ly.L - 1):
    for r in range(Ly.N_R):
        x1[Ly.ix_o(l, r)] = 1.0
x1[Ly.off_q:Ly.off_q + Ly.n_q] = [1, 0, 1, 0, 1]

for name, x in (("initial", x0), ("inventory", x1)):
    sub = StageSubproblem(1, prob, scfg)
    sub.assemble(None)                       # theta fixed at 0
    sub.set_point(x, scene=scenes[0], require_int=True)
    sub.solve_mip()
    status = sub.model.status
    assert status == COPT.OPTIMAL, (name, status)
    dec = sub.decision_values()
    mip_objective = float(sub.obj)
    mip_stage_cost = float(sub.stage_cost())
    # LP relaxation of the same integer subproblem: the Tier2 backward pass.
    sub.solve_lp()
    lp_objective = float(sub.obj)
    lp_gradient = [float(v) for v in sub.gradient("dj")]
    rows = [c.getName() for c in sub.model.getConstrs()]
    cols = [v.getName() for v in sub.model.getVars()]
    Nmat = np.asarray(dec["N"], dtype=float) if "N" in dec else None
    case = {
        "name": name,
        "state": [float(v) for v in x],
        "scene_index": 0,
        "objective": mip_objective,
        "stage_cost": mip_stage_cost,
        "theta": float(sub.theta_value()),
        "q_out": [float(v) for v in dec["q"]],
        "I_out": [float(v) for v in dec["I_out"]],
        "n_out": [[float(v) for v in row] for row in dec["n_out"]],
        "taup": [float(v) for v in dec["taup"]],
        "taum": [float(v) for v in dec["taum"]],
        "y": [[float(v) for v in row] for row in dec["y"]],
        "S_end": [float(v) for v in dec["S_end"]],
        "N": None if Nmat is None else Nmat.tolist(),
        "F_new": [float(v) for v in dec["F_new"]],
        "lp_objective": lp_objective,
        "lp_gradient": lp_gradient,
        "rows": len(rows),
        "cols": len(cols),
        "integer_cols": len(cols) - 1,   # theta is the only continuous column

        "rows_by_family": dict(Counter(family(n) for n in rows)),
        "cols_by_family": dict(Counter(col_family(n) for n in cols)),
    }
    cases.append(case)
    print(f"[{name}] objective={case['objective']:.10f} "
          f"rows={case['rows']} cols={case['cols']} theta={case['theta']:.1f}")
    print("  rows_by_family:", case["rows_by_family"])
    print("  cols_by_family:", case["cols_by_family"])

out = {
    "scene_index": 0,
    "pool_size": len(prob.pool),
    "P_ref": prob.P_ref.tolist(),
    "cases": cases,
}
(HERE / "_python_stage.json").write_text(json.dumps(out, indent=2),
                                          encoding="utf-8")
print("written:", HERE / "_python_stage.json")
