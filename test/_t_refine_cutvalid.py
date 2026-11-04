# _t_refine_cutvalid.py — 细化后新簇割的支撑性 + 加权 Jensen 下界。
#   python test/_t_refine_cutvalid.py
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
        refine=RefineConfig(enabled=True, max_refines=2,
                            max_clusters_per_stage=4, theta_cap=4,
                            warmup_rounds=999, min_child_size=1,
                            eps_obj=0.0),
        payload_mode="pickle")
    prob = DefineProblem(pcfg)
    T, dim = prob.T, prob.state_dim
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, cfg,
                         results_dir=Path(td), log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            trial, _ = eng.forward(cfg.n_forward_paths)
            t = 3
            cl = eng.parts[t][0]
            kids = [cl[:len(cl) // 2], cl[len(cl) // 2:]]
            cand = {"stage": t, "cluster": 0, "kids": kids,
                    "oldK": eng.K[t], "newK": eng.K[t] + 1,
                    "eps_sig": 1.0, "median": 1.0, "disp": 1.0, "score": 1.0,
                    "scan_s": 0.0, "scan_stages": 1,
                    "lb_before": eng.lower_bound(),
                    "lb_best_before": eng.lb_best}
            eng._apply_refinement(cand, trial, 3)
            x = np.asarray(trial[t][0], float)
            blk = put_cuts(eng.cuts[t + 1], dim)
            try:
                with attached(blk.ref) as cuts_next:
                    sp = StageSubproblem(
                        t, prob, cfg, n_theta_cap=eng.theta_cap,
                        active_k=eng.K[t + 1],
                        theta_weights=eng.weights[t + 1])
                    try:
                        qs = []
                        for inds in eng.parts[t]:
                            sp.set_state(x)
                            sp.set_scene((None, aggregate_theta(prob.pool, inds)))
                            sp.assemble(cuts_next, key=f"cv{inds[0]}",
                                        active_k=eng.K[t + 1],
                                        weights=eng.weights[t + 1])
                            sp.solve_lp()
                            qs.append(float(sp.obj))
                        mean_q = 0.0
                        for k in range(len(prob.pool)):
                            sp.set_state(x)
                            sp.set_scene(prob.pool[k])
                            sp.assemble(cuts_next, key=f"cvk{k}",
                                        active_k=eng.K[t + 1],
                                        weights=eng.weights[t + 1])
                            sp.solve_lp()
                            mean_q += float(sp.obj)
                        mean_q /= len(prob.pool)
                        w = cluster_weights(eng.parts[t], len(prob.pool))
                        jensen = float(np.dot(w, qs))
                        assert jensen <= mean_q + TOL, (jensen, mean_q)
                        for l in range(eng.K[t]):
                            cs = [c for c in eng.cuts[t] if c.group == l]
                            c = max(cs, key=lambda z: z.it)
                            assert c.value_at(x) <= qs[l] + TOL, (l, c.value_at(x), qs[l])
                        print(f"  [PASS] t={t} 新 K={eng.K[t]}：逐簇支撑 + "
                              f"Jensen {jensen:.2f} ≤ {mean_q:.2f}")
                    finally:
                        sp.free()
            finally:
                blk.close()
        finally:
            eng.close()
    print("OK _t_refine_cutvalid")


if __name__ == "__main__":
    main()
