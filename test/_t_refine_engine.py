# _t_refine_engine.py — 强制细化：K/LML1/其他阶段隔离、stage1、resume、simulate。
#   python test/_t_refine_engine.py
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
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils import simulate                                       # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def _cfg(scfg):
    return dataclasses.replace(
        scfg, mode="multi",
        multicut=MulticutConfig(n_clusters=2, group_mode="failure_quantile"),
        refine=RefineConfig(enabled=True, max_refines=4,
                            max_clusters_per_stage=4, theta_cap=4,
                            warmup_rounds=999, stall_window=2,
                            cooldown_rounds=0, recovery_timeout=0,
                            min_child_size=1, eps_obj=0.0),
        payload_mode="pickle")


def _force(eng, trial, t, it):
    cl = eng.parts[t][0]
    assert len(cl) >= 2
    kids = [cl[:len(cl) // 2], cl[len(cl) // 2:]]
    cand = {"stage": t, "cluster": 0, "kids": kids, "oldK": eng.K[t],
            "newK": eng.K[t] + 1, "eps_sig": 1.0, "median": 1.0,
            "disp": 1.0, "score": 1.0, "scan_s": 0.0, "scan_stages": 1,
            "lb_before": eng.lower_bound(), "lb_best_before": eng.lb_best}
    return eng._apply_refinement(cand, trial, it)


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    cfg = _cfg(scfg)
    prob = DefineProblem(pcfg)
    T = prob.T
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, cfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            trial, _ = eng.forward(cfg.n_forward_paths)
            other = {s: [id(c) for c in eng.cuts[s]] for s in range(1, T + 1)}
            t = 3
            oldK = eng.K[t]
            ev = _force(eng, trial, t, 3)
            assert eng.K[t] == oldK + 1
            assert {c.group for c in eng.cuts[t]} == set(range(eng.K[t]))
            assert all(len(eng._pts_g[t][g]) >= 1 for g in range(eng.K[t]))
            for s in range(1, T + 1):
                if s != t:
                    assert eng.K[s] == len({c.group for c in eng.cuts[s]})
                    assert [id(c) for c in eng.cuts[s]] == other[s], s
            assert eng.lower_bound() >= -1e-6
            # stage 1：stage0 的 active K 必须切到新 K1
            _force(eng, trial, 1, 4)
            assert eng.theta_spec(0).active_k == eng.K[1]
            assert eng.lower_bound() >= -1e-6
            eng.train(1, start_iter=5, verbose=False)
            lb = eng.lower_bound()
            assert lb >= -1e-6
            eng.checkpoint()
            print(f"  [PASS] 强制细化 t={t}→K{eng.K[t]}，stage1→K{eng.K[1]}，"
                  f"其他阶段不动，LB={lb:.2f} best={eng.lb_best:.2f}")
        finally:
            eng.close()

        cuts, meta = load_cuts(td, T, prob.state_dim, expected_mode="multi")
        assert meta["_format_version"] == 7
        assert meta["layouts"]["theta_cap"] == 4
        assert [meta["layouts"]["K"][str(s)] for s in range(1, T + 1)] == \
            [int(meta["n_theta_by_stage"][s - 1]) for s in range(1, T + 1)]
        eng2 = SDDPEngine(prob, make_stage_subproblem, cfg, results_dir=td,
                          log=lambda *a, **k: None)
        try:
            eng2.check_cut_meta(meta)
            eng2.cuts = cuts
            eng2.lb_best = float(meta.get("lb_best", 0.0))
            eng2.restore_cut_history()
            eng2.restore_refine_state(meta)
            assert eng2.K[1:T + 1] == [meta["layouts"]["K"][str(s)]
                                       for s in range(1, T + 1)]
            eng2.train(1, start_iter=int(meta["n_iterations"]) + 1, verbose=False)
            assert eng2.lower_bound() >= -1e-6
            st = simulate(eng2, n_paths=5, seed=2026, chunk_size=5)
            assert np.isfinite(st.mean) and st.n == 5
            print(f"  [PASS] resume 安装 per-stage K，续训+5 路径仿真 UB={st.mean:.1f}")
        finally:
            eng2.close()
    print("OK _t_refine_engine")


if __name__ == "__main__":
    main()
