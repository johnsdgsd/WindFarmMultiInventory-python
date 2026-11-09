# dump_pool_fingerprint.py — Python fingerprint of the full 300-scene pool,
# P_ref, pool means/row sums and the transition matrix at several lambdas.
#
# Read-only. Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/dump_pool_fingerprint.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PY_ROOT))

from code.problem_data import from_json, transition_matrix  # noqa: E402
from code.transition import tail_cdf  # noqa: E402

data = json.loads((ROOT / "data" / "problem_data.json").read_text(encoding="utf-8"))
prob = from_json(data)
pool = prob.pool
G, N_R = prob.G, prob.N_R

row_max = 0.0
zero_max = 0.0
n_tiny = 0
max_tiny = 0.0
for _, th in pool:
    row_max = max(row_max, float(np.max(np.abs(th.sum(axis=2) - 1.0))))
    for i in range(G):
        if i:
            zero_max = max(zero_max, float(np.max(np.abs(th[:, i, :i]))))
    c = tail_cdf(th)
    for r in range(N_R):
        for i in range(G):
            for j in range(i + 1, G + 1):
                v = abs(float(c[r, i, j]))
                if 0.0 < v < 1e-10:
                    n_tiny += 1
                    max_tiny = max(max_tiny, v)

mean_theta = np.mean([th for _, th in pool], axis=0)
mean_P = np.mean([transition_matrix(prob.lambda0 * np.exp(
                      np.dot(prob.beta_cov, z[:, r] - prob.z_bar)), G,
                      prob.p_digits)
                  for z, _ in pool for r in range(N_R)], axis=0)
lams = [0.0, 0.001, 0.007, 0.05, 0.5]
trans = {str(lam): transition_matrix(lam, G, prob.p_digits).tolist()
         for lam in lams}

out = {
    "pool_size": len(pool),
    "scene0_z": pool[0][0].tolist(),
    "scene0_theta": pool[0][1].tolist(),
    "scene1_theta": pool[1][1].tolist(),
    "scene7_theta": pool[7][1].tolist(),
    "mean_theta": mean_theta.tolist(),
    "mean_P": mean_P.tolist(),
    "row_sum_max": row_max,
    "struct_zero_max": zero_max,
    "P_ref": prob.P_ref.tolist(),
    "n_tail_coeff_below_1e-10": n_tiny,
    "max_tail_coeff_below_1e-10": max_tiny,
    "transition": trans,
    "transition_lambdas": lams,
}
(HERE / "_python_pool_fingerprint.json").write_text(json.dumps(out),
                                                    encoding="utf-8")
print("pool", len(pool), "row_sum_max", row_max, "struct_zero_max", zero_max)
print("tail coefficients in (0,1e-10):", n_tiny, "max", max_tiny)
print("mean theta shape", mean_theta.shape, "mean P shape", mean_P.shape)
print("written:", HERE / "_python_pool_fingerprint.json")
