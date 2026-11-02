# _t_partition_calib.py — 只读预检：pi 签名尺度、Q 差、贪心簇数。
#   python test/_t_partition_calib.py results/base_T60_20261026_204105 [stages]
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.io import load_cuts                                    # noqa: E402
from utils.partition import (greedy_split, median_pairwise_distance,  # noqa: E402
                             stall_trigger)
from utils.payload import put_cuts                                # noqa: E402
from utils.worker import scene_signatures_task                    # noqa: E402


def main():
    src = Path(sys.argv[1]).resolve()
    pcfg = load_problem_config(src / "config" / "problem.toml")
    scfg = load_solver_config(src / "config" / "solver.toml")
    scfg.n_processes = 1
    scfg.partition.enabled = False
    if len(sys.argv) > 3:
        scfg.partition.signature = sys.argv[3]
    eps_obj = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    prob = DefineProblem(pcfg)
    T, dim, n = prob.T, prob.state_dim, len(prob.pool)
    stages = ([int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [1, 5, 15, 30, 45, 59])
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    try:
        cuts, _ = load_cuts(src, T, dim)
        eng.cuts = cuts
        states = eng.forward_paths([[0] * T])[0][0]
        RELS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0)
        print("stage | med|pi| | med dist | Q range | " +
              " ".join(f"L@{q:g}" for q in RELS))
        for t in stages:
            x = np.asarray(states[t - 1], dtype=float)
            blk = put_cuts(eng.cuts[t + 1] if t < T else [], dim)
            try:
                _, sigs, qs = scene_signatures_task(
                    (999, t, 0, x, list(range(n)), blk.ref))
            finally:
                blk.close()
            med = median_pairwise_distance(sigs)
            ns = np.linalg.norm(sigs, axis=1)
            counts = []
            for rel in RELS:
                parts = greedy_split(list(range(n)), sigs, qs, rel * med, eps_obj)
                counts.append(len(parts))
            print(f"{t:5d} | {np.median(ns):9.3f} | {med:9.3f} | "
                  f"{qs.min():.3f}..{qs.max():.3f} | " +
                  " ".join(f"{c:3d}" for c in counts))
        print("说明：L@x 是 eps_sig=x*med_dist、eps_obj=1 时从单簇贪心分裂的簇数")
        print("OK _t_partition_calib")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
