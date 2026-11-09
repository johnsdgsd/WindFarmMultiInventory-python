# compare_python_stage_extra.py — extended one-stage Python Tier2 reference.
#
# Companion to compare_python_stage.py: 5 fixed integer states x 3 scenes
# (0, 1, 7), full variable/row/bound dump for validate_equivalence_extra.jl.
# Read-only w.r.t. Python code/results.
#
# Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/compare_python_stage_extra.py

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from coptpy import COPT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PY_ROOT))

from code.config import load_solver_config  # noqa: E402
from code.problem_data import from_json  # noqa: E402
from code.subproblem import StageSubproblem  # noqa: E402
from utils import copt_env as ce  # noqa: E402


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
scfg.mip_rel_gap = 0.0
scfg.mip_time_limit = 300.0
scfg.int_tol = 1e-6

SCENES = [0, 1, 7]
Ly = prob.layout

# ---- fixed integer test states (all within Python x bounds and fleet) ----
states = {}
states["initial"] = prob.initial_state()

x = prob.initial_state()
x[Ly.off_I:Ly.off_I + Ly.n_I] = [2, 1, 2, 1, 1]
for l in range(Ly.L - 1):
    for r in range(Ly.N_R):
        x[Ly.ix_o(l, r)] = 1.0
x[Ly.off_q:Ly.off_q + Ly.n_q] = [1, 0, 1, 0, 1]
states["inventory"] = x

x = prob.initial_state()
x[Ly.off_I:Ly.off_I + Ly.n_I] = [12, 8, 20, 15, 8]          # Imax, no pipeline/order
states["I_max"] = x

x = prob.initial_state()
x[Ly.off_I:Ly.off_I + Ly.n_I] = [4, 3, 5, 3, 2]
for l in range(Ly.L - 1):
    for r in range(Ly.N_R):
        x[Ly.ix_o(l, r)] = 1.0
x[Ly.off_q:Ly.off_q + Ly.n_q] = [1, 1, 1, 1, 1]
states["pipeline_mid"] = x

x = prob.initial_state()
x[Ly.off_I:Ly.off_I + Ly.n_I] = [5, 3, 7, 4, 2]
x[Ly.ix_o(0, 0):Ly.ix_o(0, Ly.N_R)] = [2, 1, 2, 1, 1]
x[Ly.ix_o(1, 0):Ly.ix_o(1, Ly.N_R)] = [1, 2, 1, 2, 1]
x[Ly.off_q:Ly.off_q + Ly.n_q] = [2, 1, 2, 1, 2]
n = np.rint(prob.n0).astype(int).copy()
move = (n[:, 0] * 0.10).astype(int)
n[:, 0] -= move
n[:, 2] += move
for r in range(Ly.N_R):
    for i in range(Ly.G):
        x[Ly.ix_n(r, i)] = float(n[r, i])
states["mixed"] = x

x = prob.initial_state()
n = np.zeros((Ly.N_R, Ly.G), dtype=int)
n[:, Ly.G - 1] = prob.N
for r in range(Ly.N_R):
    for i in range(Ly.G):
        x[Ly.ix_n(r, i)] = float(n[r, i])
states["all_grade3"] = x

# Same corner but with the fleet row slack (10 units offline): probes whether
# gradient differences at all_grade3 are dual degeneracy at the tight row.
x = prob.initial_state()
n = np.zeros((Ly.N_R, Ly.G), dtype=int)
n[:, Ly.G - 1] = prob.N - 10
for r in range(Ly.N_R):
    for i in range(Ly.G):
        x[Ly.ix_n(r, i)] = float(n[r, i])
states["grade3_slack"] = x

# ---- model-wide column / row dump (built once, before assemble) ----
scfg0 = load_solver_config(PY_ROOT / "config" / "tune" / "int_t2_formal" /
                           "solver.toml", resolve=False)
base = StageSubproblem(1, prob, scfg0)
col_info = []
for v in base.model.getVars():
    col_info.append({
        "name": v.getName(),
        "family": col_family(v.getName()),
        "lb": float(v.getInfo(COPT.Info.LB)),
        "ub": float(v.getInfo(COPT.Info.UB)),
        "type": v.getType(),
    })
row_info = []
for c in base.model.getConstrs():
    row_info.append({
        "name": c.getName(),
        "family": family(c.getName()),
        "lb": float(c.getInfo(COPT.Info.LB)),
        "ub": float(c.getInfo(COPT.Info.UB)),
    })
base.free()

cases = []
for name, xs in states.items():
    xs = np.asarray(xs, dtype=float)
    for si in SCENES:
        sub = StageSubproblem(1, prob, scfg)
        sub.assemble(None)
        sub.set_point(xs, scene=prob.pool[si], require_int=True)
        sub.solve_mip()
        status = sub.model.status
        assert status == COPT.OPTIMAL, (name, si, status)
        dec = sub.decision_values()
        mip_objective = float(sub.obj)
        mip_stage_cost = float(sub.stage_cost())
        x_out = [float(v) for v in ce.values(sub.model, sub.x_out)]
        # All MIP values must be integer (sanity check on Python side too).
        Nvals = [dec["N"][r][i][j] for r in range(Ly.N_R) for i in range(Ly.G)
                 for j in range(i, Ly.G + 1)]
        ctrl_vals = (dec["taup"] + dec["taum"] + dec["S_end"]
                     + [v for row in dec["y"] for v in row] + Nvals)
        # LP relaxation of the same integer subproblem.
        sub.solve_lp()
        lp_objective = float(sub.obj)
        lp_gradient = [float(v) for v in sub.gradient("dj")]
        lp_x_out = [float(v) for v in ce.values(sub.model, sub.x_out)]
        cases.append({
            "name": name,
            "scene_index": int(si),
            "state": xs.tolist(),
            "objective": mip_objective,
            "stage_cost": mip_stage_cost,
            "theta": float(sub.theta_value()),
            "x_out": x_out,
            "q_out": [float(v) for v in dec["q"]],
            "I_out": [float(v) for v in dec["I_out"]],
            "n_out": [[float(v) for v in row] for row in dec["n_out"]],
            "o_out": [[x_out[Ly.ix_o(l, r)] for r in range(Ly.N_R)]
                      for l in range(Ly.L - 1)],
            "taup": [float(v) for v in dec["taup"]],
            "taum": [float(v) for v in dec["taum"]],
            "y": [[float(v) for v in row] for row in dec["y"]],
            "S_end": [float(v) for v in dec["S_end"]],
            "N": dec["N"],
            "F_new": [float(v) for v in dec["F_new"]],
            "W_in": [float(v) for v in dec["W_in"]],
            "W_out": [float(v) for v in dec["W_out"]],
            "mip_all_integer": all(abs(v - round(v)) <= 1e-6 for v in ctrl_vals)
                                and all(abs(v - round(v)) <= 1e-6 for v in x_out),
            "lp_objective": lp_objective,
            "lp_gradient": lp_gradient,
            "lp_x_out": lp_x_out,
        })
        print(f"[{name}/scene {si}] MIP={mip_objective:.6f} "
              f"stage={mip_stage_cost:.6f} LP={lp_objective:.6f} "
              f"int_ok={cases[-1]['mip_all_integer']}")

# ---- an intentionally infeasible state: both models must agree ----
# I=Imax, o=Qmax at both layers, q=Qmax, n=n0.  Only grade-3 units can fail
# (P[i,G]=0 for i<G at p_digits=3), so the dem row caps total inbound transfer
# by F~6.5; the out-side capacity needs ~3*sum(Qmax)=39 of outflow -> infeasible.
x_inf = prob.initial_state()
x_inf[Ly.off_I:Ly.off_I + Ly.n_I] = [12, 8, 20, 15, 8]
for l in range(Ly.L - 1):
    for r in range(Ly.N_R):
        x_inf[Ly.ix_o(l, r)] = float(prob.Qmax[r])
x_inf[Ly.off_q:Ly.off_q + Ly.n_q] = [3, 2, 3, 3, 2]
for r in range(Ly.N_R):
    for i in range(Ly.G):
        x_inf[Ly.ix_n(r, i)] = float(prob.n0[r, i])
sub = StageSubproblem(1, prob, scfg)
sub.assemble(None)
sub.set_point(x_inf, scene=prob.pool[0], require_int=True)
try:
    sub.solve_mip()
    inf_status = int(sub.model.status)
except Exception:
    inf_status = int(sub.model.status)
print("infeasible case Python status:", inf_status, "(COPT INFEASIBLE=2)")
assert inf_status == 2, inf_status

out = {
    "scenes": SCENES,
    "pool_size": len(prob.pool),
    "state_dim": Ly.dim,
    "state_ub": Ly.upper(prob).tolist(),
    "rows": len(row_info),
    "cols": len(col_info),
    "col_family_counts": dict(Counter(c["family"] for c in col_info)),
    "row_family_counts": dict(Counter(r["family"] for r in row_info)),
    "col_info": col_info,
    "row_info": row_info,
    "cases": cases,
    "infeasible": {"state": x_inf.tolist(), "scene_index": 0,
                   "python_status": inf_status},
}
(HERE / "_python_stage_extra.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")
print("written:", HERE / "_python_stage_extra.json")
