# _t_importance.py — 全局 proposal 构造/权重，以及训练 warmup 后 IS 激活。
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (ImportanceSamplingConfig, load_problem_config,  # noqa: E402
                         load_solver_config)
from code.problem_data import DefineProblem                              # noqa: E402
from code.subproblem import make_stage_subproblem                        # noqa: E402
from utils.engine import SDDPEngine                                      # noqa: E402
from utils.importance import (build_global_proposal, weight_stats)       # noqa: E402


def main():
    # toy：g 行和 1、所有元素正、权重期望附近为 1
    paths = [[0, 1, 0], [1, 1, 2], [2, 0, 1], [0, 2, 2]]
    totals = [10.0, 20.0, 30.0, 40.0]
    prop = build_global_proposal(paths, totals, 3, uniform_floor=0.05)
    assert np.isclose(prop.g.sum(), 1.0)
    assert np.all(prop.g > 0)
    w = prop.weights(paths)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    st = weight_stats(w)
    assert st["ess"] > 0
    print(f"  [PASS] proposal g={np.round(prop.g,4).tolist()} "
          f"weights={np.round(w,3).tolist()}")

    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    scfg = dataclasses.replace(scfg, importance_sampling=ImportanceSamplingConfig(
        enabled=True, target="training", build_after_round=2,
        uniform_floor=0.05))
    scfg.n_processes = 2
    prob = DefineProblem(pcfg)
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg,
                         results_dir=Path(td), log=lambda *a, **k: None)
        try:
            eng.train(3, verbose=False)
            assert eng.is_proposal is not None
            assert eng.is_stats.get("ess", 0) > 0
            meta = eng.meta()
            assert meta["importance_sampling"]["active"] is True
            assert meta["importance_sampling"]["stats"]["fingerprint"]
            print(f"  [PASS] warmup 后 IS 激活；ESS="
                  f"{meta['importance_sampling']['stats']['ess']:.1f}")
        finally:
            eng.close()
    print("OK _t_importance")


if __name__ == "__main__":
    main()
