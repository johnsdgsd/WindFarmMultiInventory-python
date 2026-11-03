# _t_fcmc_cutvalid.py — 固定簇割：逐簇支撑性 + Jensen 加权下界。
#   python test/_t_fcmc_cutvalid.py
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (MulticutConfig, load_problem_config,  # noqa: E402
                         load_solver_config)
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem, make_stage_subproblem  # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402
from utils.partition import aggregate_theta                      # noqa: E402
from utils.payload import attached, put_cuts                     # noqa: E402

TOL = 1e-4


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    scfg = dataclasses.replace(
        scfg, mode="multi",
        multicut=MulticutConfig(n_clusters=4, group_mode="failure_quantile"))
    prob = DefineProblem(pcfg)
    T, dim = prob.T, prob.state_dim
    ub = prob.layout.upper(prob)

    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=Path(td),
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            states = eng.forward_paths([[1] * T])[0][0]
            t = min(2, T - 1)
            x = np.asarray(states[t - 1], float)
            x2 = x.copy()
            moved = 0
            for j in np.flatnonzero(ub > 1e-6)[:6]:
                step = 0.25 if x[j] + 0.25 <= ub[j] else -0.25
                if 0.0 <= x[j] + step <= ub[j]:
                    x2[j] = x[j] + step
                    moved += 1
            assert moved > 0

            blk = put_cuts(eng.cuts[t + 1], dim)
            try:
                with attached(blk.ref) as cuts_next:
                    sp = StageSubproblem(t, prob, scfg)

                    def solve(xx, theta):
                        sp.set_state(xx)
                        sp.set_scene((None, theta))
                        sp.solve_lp()
                        return float(sp.obj), sp.gradient("dj").copy()

                    try:
                        for xa in (x, x2):
                            qs, betas = [], []
                            for inds in eng.clusters:
                                q, b = solve(xa, aggregate_theta(prob.pool, inds))
                                qs.append(q)
                                betas.append(b)
                            mean_q = float(np.mean([
                                solve(xa, prob.pool[k][1])[0]
                                for k in range(len(prob.pool))]))
                            jensen = float(np.dot(eng.cluster_weights, qs))
                            assert jensen <= mean_q + TOL, (jensen, mean_q)
                            wcut = 0.0
                            for l in range(len(eng.clusters)):
                                c = max((c for c in eng.cuts[t]
                                         if c.group == l),
                                        key=lambda c: c.it)
                                val = c.value_at(xa)
                                assert val <= qs[l] + TOL, (l, val, qs[l])
                                wcut += eng.cluster_weights[l] * val
                            assert wcut <= mean_q + TOL, (wcut, mean_q)
                        print(f"  阶段 {t}：4 簇支撑割 + 加权 Jensen 下界通过（2 个状态）")
                    finally:
                        sp.free()
            finally:
                blk.close()
        finally:
            eng.close()
    print("OK _t_fcmc_cutvalid")


if __name__ == "__main__":
    main()
