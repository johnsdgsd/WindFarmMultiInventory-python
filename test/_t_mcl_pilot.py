# MCL preflight: use the saved T60 baseline cuts as a terminal/suffix envelope,
# lift them backward through the real stage dynamics, and test the root LB.
# Read-only with respect to results/; no result directory is written.
#
#   python test/_t_mcl_pilot.py
#
# The direct-sharing pilot already showed zero root-LB change. This script tests
# the stronger lifting variant: A_s = cuts from source stage s; for k=s-1..1,
# solve stage k at selected anchors with future envelope A_{k+1}, add the new
# cuts, select a bounded envelope A_k, then solve the stage-0 LB with cuts[1].
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine                                          # noqa: E402
from utils.io import load_cuts                                        # noqa: E402

BASE = ROOT / "results/base_T60_20261025_131043"
K_ENV = 100          # envelope size after each lifted stage
N_ANCHOR = 4         # anchors per stage from baseline simulation
SEED_PATHS = (0, 500, 1000, 1500)


def _value(cut, states):
    return max(float(cut.alpha + cut.beta @ np.asarray(x, float)) for x in states)


def _top(cuts, states, k):
    if not cuts:
        return []
    order = sorted(cuts, key=lambda c: _value(c, states), reverse=True)
    out, seen = [], set()
    for c in order:
        key = c.beta.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _anchors(prob, x_sim, k):
    """Feasible anchors for stage k from the saved baseline simulation."""
    Ly = prob.layout
    a = [np.asarray(x_sim[i, k - 1], float).copy() for i in SEED_PATHS]
    if k == 1:
        # all baseline paths share x1; add q-perturbations around the root order
        base = a[0]
        qs = Ly.q(base)
        for r in range(min(3, Ly.N_R)):
            x = base.copy()
            q = qs.copy()
            q[r] = min(prob.Qmax[r], q[r] + 1.0)
            x[Ly.q_slice()] = q
            a.append(x)
    return a


def run_lift(source_s: int):
    pcfg = load_problem_config(ROOT / "config/problem.toml")
    scfg = load_solver_config(ROOT / "config/solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)
    npz = np.load(BASE / "sim_2000paths.npz")
    x_sim = npz["x_in"]

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    try:
        eng.cuts = cuts
        eng.restore_cut_history()
        lb0 = eng.lower_bound()
        print(f"source s={source_s}: baseline LB = {lb0:.6f}")

        # A[s] starts with all saved cuts at the source stage.
        envelope = {source_s: list(eng.cuts[source_s])}
        eng.round_id = 9000 + source_s
        for k in range(source_s - 1, 0, -1):
            eng.cuts[k + 1] = list(envelope[k + 1])
            anchors = _anchors(prob, x_sim, k)
            batch = [[] for _ in range(prob.T + 1)]
            batch[k] = anchors
            n_add = eng.backward(batch, it=1)
            # Keep a bounded, diverse envelope for the next lower stage.
            envelope[k] = _top(eng.cuts[k], anchors, K_ENV)
            eng.cuts[k] = list(envelope[k])
            if k % 10 == 0 or k <= 2:
                print(f"  k={k:3d}  anchors={len(anchors)}  +{n_add:3d} cuts  "
                      f"envelope={len(envelope[k])}")
        eng.cuts[1] = list(envelope[1])
        lb1 = eng.lower_bound()
        print(f"source s={source_s}: lifted LB = {lb1:.6f}  "
              f"delta = {lb1 - lb0:+.6f}  ({(lb1-lb0)/abs(lb0)*100:+.3f}%)")
        print(f"  cuts[1] = {len(eng.cuts[1])}")
        return lb0, lb1
    finally:
        eng.close()


def main():
    t0 = time.time()
    for s in (60, 50):
        run_lift(s)
    print(f"pilot wall = {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
