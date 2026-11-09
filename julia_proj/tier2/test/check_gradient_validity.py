# check_gradient_validity.py — the differing Julia/Python LP gradients at a
# degenerate state must both be valid subgradients of the same LP value function.
#
# Reads both fixtures (_python_stage_extra.json, _julia_stage_extra.json), then
# for every case where |g_py - g_jl| > 1e-6 evaluates the Python LP at the base
# and at single I/o perturbations (+1) and checks
#     Q(x') >= Q(x) + g . (x' - x)
# for both gradients. A convex value function has a kink with a whole
# subdifferential interval at such a state, so both may pass while differing.
#
# Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/check_gradient_validity.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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
scfg.mip_rel_gap = 0.0
scfg.mip_time_limit = 300.0
scfg.int_tol = 1e-6
py = json.loads((HERE / "_python_stage_extra.json").read_text(encoding="utf-8"))
jl_path = HERE / "_julia_stage_extra.json"
if not jl_path.is_file():
    raise SystemExit("先运行 julia test/validate_equivalence_extra.jl 生成 "
                     "_julia_stage_extra.json")
jl = json.loads(jl_path.read_text(encoding="utf-8"))
jl_by = {(c["name"], int(c["scene_index"])): c for c in jl["cases"]}
Ly = prob.layout
ub = Ly.upper(prob)
ok = True
n_checked = 0


def lp_value(x, scene_index):
    sub = StageSubproblem(1, prob, scfg)
    sub.assemble(None)
    sub.set_point(np.asarray(x, dtype=float), scene=prob.pool[scene_index],
                  require_int=True)
    sub.solve_lp()
    return float(sub.obj)


for case in py["cases"]:
    key = (case["name"], int(case["scene_index"]))
    jrec = jl_by[key]
    g_py = np.asarray(case["lp_gradient"], dtype=float)
    g_jl = np.asarray(jrec["lp_gradient"], dtype=float)
    dmax = float(np.max(np.abs(g_py - g_jl)))
    if dmax <= 1e-6:
        continue
    n_checked += 1
    x = np.asarray(case["state"], dtype=float)
    q0 = lp_value(x, case["scene_index"])
    print(f"\n{key[0]}/scene {key[1]}: |g_py-g_jl|={dmax:.6g}, Q0={q0:.6f}")
    dirs = []
    diff_idx = np.flatnonzero(np.abs(g_py - g_jl) > 1e-6)
    for j in diff_idx:
        if x[j] < ub[j] - 1e-9:
            d = np.zeros(Ly.dim)
            d[j] = 1.0
            dirs.append((f"state[{j + 1}]+1", d))
    for label, sl in (("I_all+1", slice(0, 5)), ("o_all+1", slice(5, 15)),
                      ("q_all+1", slice(35, 40))):
        d = np.zeros(Ly.dim)
        d[sl] = 1.0
        if np.all(x[sl] + 1.0 <= ub[sl] + 1e-9):
            dirs.append((label, d))
    worst_py = np.inf
    worst_jl = np.inf
    qps = []
    for label, d in dirs:
        try:
            qp = lp_value(x + d, case["scene_index"])
        except Exception as exc:
            print(f"  [skip] {label}: perturbed LP infeasible ({type(exc).__name__})")
            continue
        qps.append(qp)
        sp = qp - q0 - float(g_py @ d)
        sj = qp - q0 - float(g_jl @ d)
        worst_py = min(worst_py, sp)
        worst_jl = min(worst_jl, sj)
        if sp < -1e-6:
            ok = False
            print(f"  [FAIL] py gradient invalid on {label}: slack={sp:.3e}")
        if sj < -1e-6:
            ok = False
            print(f"  [FAIL] jl gradient invalid on {label}: slack={sj:.3e}")
    if qps:
        print(f"  {len(qps)}/{len(dirs)} directions feasible, Q' range "
              f"[{min(qps):.6f}, {max(qps):.6f}]; worst subgradient slack: "
              f"py={worst_py:.3e}, jl={worst_jl:.3e}")
    else:
        print("  no feasible perturbation found")
        ok = False

print()
print(f"{n_checked} case(s) with differing gradients checked.")
print("RESULT:", "PASS (both are valid subgradients on all tested directions)"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
