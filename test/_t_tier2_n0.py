# _t_tier2_n0.py — Tier 2 初始 n0 最大余数整数化。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config          # noqa: E402
from code.problem_data import DefineProblem          # noqa: E402
from code.transition import largest_remainder_counts # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "tune" / "int_t2_formal" /
                               "problem.toml")
    prob = DefineProblem(pcfg)
    assert prob.n0_mode == "integer_largest_remainder"
    n0 = np.asarray(prob.n0, dtype=float)
    assert n0.shape == (prob.N_R, prob.G)
    assert np.all(n0 >= 0)
    assert np.max(np.abs(n0 - np.rint(n0))) <= 1e-9
    assert np.array_equal(n0.sum(axis=1).astype(np.int64), prob.N)
    raw = np.outer(prob.N.astype(float), prob.init_mix)
    assert np.all(np.abs(n0 - raw) < 1.0)
    for r in range(prob.N_R):
        assert np.array_equal(n0[r].astype(np.int64),
                              largest_remainder_counts(int(prob.N[r]),
                                                       prob.init_mix))
    print(f"  [PASS] T60 n0 整数、行和={n0.sum(axis=1).tolist()}，"
          f"最大偏差 {np.max(np.abs(n0-raw)):.4f}")
    print("OK _t_tier2_n0")


if __name__ == "__main__":
    main()
