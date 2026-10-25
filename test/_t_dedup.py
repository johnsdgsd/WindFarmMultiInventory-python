# Exact beta dedup: identical beta rows with smaller alpha are redundant.
# Verify the root LB is unchanged and report row-count reduction.
#   python test/_t_dedup.py
import dataclasses
import sys
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
from utils.payload import _dedup_exact                                # noqa: E402

BASE = ROOT / "results/base_T60_20261025_131043"


def main():
    pcfg = load_problem_config(ROOT / "config/problem.toml")
    scfg = load_solver_config(ROOT / "config/solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(BASE, prob.T, prob.state_dim)

    raw = uni = 0
    for t in range(1, prob.T + 1):
        if not cuts[t]:
            continue
        a = np.array([c.alpha for c in cuts[t]], float)
        b = np.vstack([c.beta for c in cuts[t]])
        a2, b2 = _dedup_exact(a, b)
        raw += len(a)
        uni += len(a2)
    print(f"stage rows: raw={raw}  unique-beta={uni}  "
          f"reduction={(1-uni/raw)*100:.1f}%")

    def lb(dedup):
        cfg = dataclasses.replace(scfg, dedup_cuts=dedup)
        eng = SDDPEngine(prob, make_stage_subproblem, cfg, results_dir=None,
                         log=lambda *a, **k: None)
        try:
            eng.cuts = cuts
            eng.restore_cut_history()
            return eng.lower_bound()
        finally:
            eng.close()

    l0 = lb(False)
    l1 = lb(True)
    print(f"root LB no-dedup={l0:.9f}  dedup={l1:.9f}  diff={l1-l0:+.3e}")
    ok = abs(l1 - l0) <= 1e-7
    print(f"[{'PASS' if ok else 'FAIL'}] dedup preserves root LB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
