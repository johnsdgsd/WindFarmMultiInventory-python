# _t_multicut_cutvalid.py — 多割场景割抽检：group=k 的割只用场景 k 验证。
#   python test/_t_multicut_cutvalid.py
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem, make_stage_subproblem  # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.payload import put_cuts                                # noqa: E402

TOL = 1e-6


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    scfg = dataclasses.replace(scfg, mode="multi")
    prob = DefineProblem(pcfg)
    T, dim = prob.T, prob.state_dim
    with tempfile.TemporaryDirectory() as td:
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=Path(td),
                         log=lambda *a, **k: None)
        try:
            eng.train(2, verbose=False)
            states = eng.forward_paths([[1] * T])[0][0]
            t = min(2, T - 1)
            x = np.asarray(states[t - 1], float)
            xp = np.asarray(states[t], float) if t < T else x
            blk = put_cuts(eng.cuts[t + 1], dim)
            try:
                sp = StageSubproblem(t, prob, scfg)
                n_checked = 0
                for k in (0, 1, 2, len(prob.pool) - 1):
                    sp.set_state(xp)
                    sp.set_scene(prob.pool[k])
                    sp.assemble((blk._arrays[0], blk._arrays[1], blk._arrays[2]),
                                key=f"cv{k}")
                    sp.solve_lp()
                    q = float(sp.obj)
                    cs = [c for c in eng.cuts[t] if c.group == k and c.it > 0]
                    assert cs, f"group {k} 无割"
                    c = cs[-1]
                    val = c.value_at(xp)
                    assert val <= q + TOL, (k, val, q, val - q)
                    n_checked += 1
                print(f"多割合法性抽检: {n_checked} 个 (场景,状态) 点")
            finally:
                blk.close()
        finally:
            eng.close()
    print("OK _t_multicut_cutvalid")


if __name__ == "__main__":
    main()
