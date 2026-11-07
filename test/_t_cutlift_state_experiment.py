# _t_cutlift_state_experiment.py — 状态级：支配割提升前后 V_cut / Q_LP / Q_MIP。
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"
STAGE = 1
DOM_IDX = 64          # 在 cut64 的锚点，cut64 是支配割（与 cut71 同值）
LIFT_IDX = 64


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    scfg_tight = dataclasses.replace(scfg, mip_rel_gap=1e-9, mip_time_limit=20.0)
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    c1 = cuts[STAGE]
    cdom = c1[DOM_IDX]
    clift = c1[LIFT_IDX]
    S = np.asarray(cdom.state, float)
    vals = np.array([c.alpha + c.beta @ S for c in c1])
    V_std = float(vals[DOM_IDX if vals[DOM_IDX] >= vals.max() - 1e-9
                        else int(np.argmax(vals))])
    dom_check = int(np.argmax(vals))
    next_cuts = (np.array([c.alpha for c in cuts[STAGE + 1]], float),
                 np.vstack([c.beta for c in cuts[STAGE + 1]]))
    print(f"state=cut{DOM_IDX}.state  cuts={len(c1)}  dominant={dom_check}  "
          f"V_std={V_std:.6f}  cut_alpha={cdom.alpha:.6f}")

    # Q_LP/Q_MIP at S over all 300 scenes; alpha at the cut's own anchor
    qlps, qmips = [], []
    alphas, bnds, objs, bnd_gaps = [], [], [], []
    spS = StageSubproblem(STAGE, prob, scfg_tight)
    spL = StageSubproblem(STAGE, prob, scfg)
    t0 = time.time()
    try:
        spS.assemble(next_cuts, key=None)
        spL.assemble(next_cuts, key=None)
        for si in range(len(prob.pool)):
            # Q at S
            spS.set_state(S, require_int=True)
            spS.set_scene(prob.pool[si])
            spS.solve_lp()
            qlps.append(float(spS.obj))
            spS.solve_mip()
            qmips.append(float(spS.obj))
            # lift alpha at the cut's anchor
            xa = np.asarray(clift.state, float)
            spL.set_state(xa, require_int=True)
            spL.set_scene(prob.pool[si])
            spL.solve_lp()
            lp = float(spL.obj)
            pi = np.asarray(spL.gradient("dj"), float)
            spL.solve_mip()
            nu = spL.solve_lift(pi, time_limit=1.0, rel_gap=1e-9,
                                safety_margin=0.0, value="best_bnd")
            if nu is None:
                continue
            alphas.append(float(nu + pi @ xa - lp))
            b = float(spL.model.getAttr(COPT.Attr.BestBnd))
            o = float(spL.model.getAttr(COPT.Attr.BestObj))
            bnds.append(b)
            objs.append(o)
            bnd_gaps.append(o - b)
    finally:
        spS.free()
        spL.free()
    alpha = float(np.mean(alphas)) if alphas else 0.0
    qlp = float(np.mean(qlps))
    qmip = float(np.mean(qmips))
    # Lifted dominant cut: same slope, alpha + alpha
    vals_lift = vals.copy()
    vals_lift[DOM_IDX] += alpha
    V_lift = float(vals_lift.max())
    print(f"alpha_n={len(alphas)}  alpha_mean={alpha:.6f}  "
          f"alpha_median={np.median(alphas):.6f}  alpha_max={max(alphas):.6f}")
    print(f"Q_LP_mean={qlp:.6f}  Q_MIP_mean={qmip:.6f}")
    print(f"V_std={V_std:.6f}  V_lift={V_lift:.6f}  dV={V_lift-V_std:.6f}")
    print(f"gap_std LP/MIP={qlp-V_std:.6f}/{qmip-V_std:.6f}  "
          f"gap_lift LP/MIP={qlp-V_lift:.6f}/{qmip-V_lift:.6f}  "
          f"closed LP/MIP={V_lift-V_std:.6f}/{V_lift-V_std:.6f}")
    print(f"cut intercept before/after={cdom.alpha:.6f}/{cdom.alpha+alpha:.6f}")
    print(f"lift Bnd-Obj mean/median/max={np.mean(bnd_gaps):.3e}/"
          f"{np.median(bnd_gaps):.3e}/{max(bnd_gaps):.3e}  "
          f"relative={np.mean(bnd_gaps)/max(abs(np.mean(objs)),1e-9):.3e}")
    print(f"elapsed={time.time()-t0:.1f}s")
    print("OK _t_cutlift_state_experiment")


if __name__ == "__main__":
    main()
