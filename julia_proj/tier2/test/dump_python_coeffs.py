# dump_python_coeffs.py — exact Python Tier2 coefficient dump for the Julia check.
#
# Dumps (scene 0, initial state):
#   * stage 1: all 146 rows (name, lb, ub, coefficients), all 191 columns
#     (name, lb, ub, type), objective coefficients + constant;
#   * stage 0 (virtual): all 5 rows, 81 columns with their post-pin bounds,
#     objective (theta = 0).
#
# Read-only w.r.t. Python code/results. Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/dump_python_coeffs.py

from __future__ import annotations

import json
import sys
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

data = json.loads((ROOT / "data" / "problem_data.json").read_text(encoding="utf-8"))
prob = from_json(data)
scfg = load_solver_config(PY_ROOT / "config" / "tune" / "int_t2_formal" /
                          "solver.toml", resolve=False)


def dump(sub):
    m = sub.model
    constrs = m.getConstrs()
    vars_ = m.getVars()
    A = m.getA()
    rows = []
    for i, c in enumerate(constrs):
        row = A.getrow(i)
        idx = row.nonzero()[1]
        rows.append({
            "name": c.getName(),
            "lb": float(c.getInfo(COPT.Info.LB)),
            "ub": float(c.getInfo(COPT.Info.UB)),
            "coeffs": {vars_[j].getName(): float(A[i, j]) for j in idx},
        })
    cols = [{
        "name": v.getName(),
        "lb": float(v.getInfo(COPT.Info.LB)),
        "ub": float(v.getInfo(COPT.Info.UB)),
        "type": v.getType(),
    } for v in vars_]
    obj = m.getObjective()
    obj_coeffs = {obj.getVar(i).getName(): float(obj.getCoeff(i))
                  for i in range(obj.getSize())}
    obj_const = float(obj.getConstant())
    return {"rows": rows, "cols": cols, "obj": obj_coeffs,
            "obj_const": obj_const, "A_shape": list(A.shape)}


x0 = prob.initial_state()
sub1 = StageSubproblem(1, prob, scfg)
sub1.assemble(None)
sub1.set_point(x0, scene=prob.pool[0], require_int=True)
stage1 = dump(sub1)
stage1["scene_index"] = 0
stage1["state_name"] = "initial"

sub0 = StageSubproblem(0, prob, scfg)
sub0.assemble(None)
sub0.set_state(x0, require_int=True)
stage0 = dump(sub0)
stage0["state_name"] = "initial"

out = {"stage1": stage1, "stage0": stage0}
(HERE / "_python_coeffs.json").write_text(json.dumps(out, indent=1),
                                          encoding="utf-8")
print(f"stage1: {len(stage1['rows'])} rows x {stage1['A_shape'][1]} cols; "
      f"objective terms {len(stage1['obj'])}, const {stage1['obj_const']}")
print(f"stage0: {len(stage0['rows'])} rows x {stage0['A_shape'][1]} cols; "
      f"objective terms {len(stage0['obj'])}, const {stage0['obj_const']}")
print("written:", HERE / "_python_coeffs.json")
