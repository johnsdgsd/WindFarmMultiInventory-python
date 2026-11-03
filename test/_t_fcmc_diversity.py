# _t_fcmc_diversity.py — T60 固定簇诊断：不同簇的割斜率是否真的不同。
#   python test/_t_fcmc_diversity.py [L] [轮数]
# 只训 1–2 轮（不落盘）。若中位相对差 < 2e-2，说明簇割几乎相同，多割不会有 LB 收益。
from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                     # noqa: E402
from code.subproblem import make_stage_subproblem               # noqa: E402
from utils.engine import SDDPEngine                             # noqa: E402


def main(L: int = 8, rounds: int = 1):
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "tune" / f"fcmc_t60_L{L}"
                              / "solver.toml")
    prob = DefineProblem(pcfg)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=print)
    try:
        eng.train(rounds, verbose=True)
        T = prob.T
        stages = sorted({1, 2, T // 2, T - 1, T})
        print(f"\n固定簇诊断 L={L} group={eng.group_mode} "
              f"规模={[len(c) for c in eng.clusters]}")
        for t in stages:
            by_state = {}
            for c in eng.cuts[t]:
                if c.it == 0:
                    continue
                key = tuple(np.round(c.state, 6))
                by_state.setdefault(key, {})[c.group] = c
            rho, alpha_rel = [], []
            for key, gd in by_state.items():
                if len(gd) != eng.n_theta:
                    continue
                for g1, g2 in itertools.combinations(range(eng.n_theta), 2):
                    c1, c2 = gd[g1], gd[g2]
                    d = np.linalg.norm(c1.beta - c2.beta)
                    den = max(np.linalg.norm(c1.beta), np.linalg.norm(c2.beta),
                              1e-12)
                    rho.append(d / den)
                    alpha_rel.append(abs(c1.alpha - c2.alpha) /
                                     max(abs(c1.alpha), abs(c2.alpha), 1e-12))
            if not rho:
                print(f"  阶段 {t:3d}: 无完整对照点")
                continue
            print(f"  阶段 {t:3d}: 点数 {len(rho):4d}  "
                  f"beta 相对差 中位 {np.median(rho):.4f}  "
                  f"p90 {np.percentile(rho, 90):.4f}  "
                  f"alpha 相对差 中位 {np.median(alpha_rel):.4f}")
    finally:
        eng.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         int(sys.argv[2]) if len(sys.argv) > 2 else 1)
