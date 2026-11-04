# _t_refine_jensen.py — 多状态 Jensen 预检：聚合簇割下界 ≤ 全场景平均 Q。
#   python test/_t_refine_jensen.py
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (MulticutConfig, RefineConfig,  # noqa: E402
                         load_problem_config, load_solver_config)
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem, make_stage_subproblem  # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.partition import aggregate_theta, cluster_weights      # noqa: E402
from utils.payload import attached, put_cuts                      # noqa: E402

TOL = 1e-4


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    cfg = dataclasses.replace(
        scfg, mode="multi",
        multicut=MulticutConfig(n_clusters=2, group_mode="failure_quantile"),
        refine=RefineConfig(enabled=True, warmup_rounds=999,
                            max_clusters_per_stage=4, theta_cap=4,
                            min_child_size=1),
        payload_mode="pickle")
    prob = DefineProblem(pcfg)
    T = prob.T
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, cfg,
                         results_dir=Path(td), log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            trial, _ = eng.forward(1)
            pts = 0
            worst = -1e9
            for t in range(1, T + 1):
                x = np.asarray(trial[t][0], float)
                blk = put_cuts(eng.cuts[t + 1] if t < T else [], eng.state_dim)
                try:
                    with attached(blk.ref) as cuts_next:
                        sp = StageSubproblem(t, prob, cfg,
                                             n_theta_cap=eng.theta_cap,
                                             active_k=eng.K[t + 1] if t < T else 0,
                                             theta_weights=eng.weights[t + 1]
                                             if t < T else ())
                        try:
                            qs = []
                            for inds in eng.parts[t]:
                                sp.set_state(x)
                                sp.set_scene((None, aggregate_theta(prob.pool, inds)))
                                sp.assemble(cuts_next, key=f"j{ints(inds)}",
                                            active_k=eng.K[t + 1] if t < T else 0,
                                            weights=eng.weights[t + 1] if t < T else ())
                                sp.solve_lp()
                                qs.append(float(sp.obj))
                            mean_q = 0.0
                            for k in range(len(prob.pool)):
                                sp.set_state(x)
                                sp.set_scene(prob.pool[k])
                                sp.assemble(cuts_next, key=f"jk{t}_{k}",
                                            active_k=eng.K[t + 1] if t < T else 0,
                                            weights=eng.weights[t + 1] if t < T else ())
                                sp.solve_lp()
                                mean_q += float(sp.obj)
                            mean_q /= len(prob.pool)
                            w = cluster_weights(eng.parts[t], len(prob.pool))
                            gap = mean_q - float(np.dot(w, qs))
                            worst = max(worst, gap)
                            assert gap >= -TOL, (t, gap)
                            pts += 1
                        finally:
                            sp.free()
                finally:
                    blk.close()
            print(f"  [PASS] {pts} 个 (阶段,状态) Jensen gap≥0，max gap={worst:.6f}")
        finally:
            eng.close()
    print("OK _t_refine_jensen")


def ints(x):
    return "_".join(str(int(v)) for v in x[:3])


if __name__ == "__main__":
    main()
