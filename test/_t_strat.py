# Stratified forward sampling smoke: paths cover distinct failure-intensity strata.
#   python test/_t_strat.py
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


def main():
    pcfg = load_problem_config(ROOT / "config/strat/problem.toml")
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/strat/solver.toml"), n_processes=1)
    prob = DefineProblem(pcfg)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    try:
        paths = eng.sample_scene_paths(10)
        ranks = [int(np.where(eng.scene_order == s)[0][0])
                 for p in paths for s in p]
        print(f"paths={len(paths)} lens={set(map(len, paths))} "
              f"scene rank range=[{min(ranks)},{max(ranks)}]")
        # every path must draw only from its stratum (30 scenes for 10 paths)
        ok = all(all((k * 30) <= int(np.where(eng.scene_order == s)[0][0])
                     < min((k + 1) * 30, len(eng.scene_order))
                     for s in paths[k]) for k in range(10))
        print(f"[{'PASS' if ok else 'FAIL'}] each path stays in its stratum")
        return 0 if ok else 1
    finally:
        eng.close()


if __name__ == "__main__":
    sys.exit(main())
