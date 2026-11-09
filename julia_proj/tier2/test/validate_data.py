# validate_data.py — independent check of the exported Tier2 instance.
#
# Read-only: loads the Julia project's data/problem_data.json and re-derives the
# Python Tier2 ProblemData from config/tune/int_t2_formal/problem.toml, then
# asserts bit-identical agreement. Does not touch Python code/results.
#
# Run:
#   "D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe" test/validate_data.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PY_ROOT))

from code.config import load_problem_config  # noqa: E402
from code.problem_data import DefineProblem, from_json, transition_matrix  # noqa: E402
from code.state_layout import StateLayout  # noqa: E402
from code.transition import largest_remainder_counts, tail_cdf  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}")
    ok = ok and bool(cond)


data_path = ROOT / "data" / "problem_data.json"
cfg_path = ROOT / "config" / "problem.toml"
raw = json.loads(data_path.read_text(encoding="utf-8"))
prob = from_json(raw)
cfg = load_problem_config(cfg_path)

print("=" * 78)
print("1. Dimensions / layout / integer initial state")
print("=" * 78)
check("T=60, G=4, N_R=5, L=3",
      (prob.T, prob.G, prob.N_R, prob.L) == (60, 4, 5, 3),
      f"got {(prob.T, prob.G, prob.N_R, prob.L)}")
ly = prob.layout
check("layout I(5)|o(2*5)|n(5*4)|q(5) offsets",
      (ly.off_I, ly.off_o, ly.off_n, ly.off_q, ly.dim) == (0, 5, 15, 35, 40),
      f"got {(ly.off_I, ly.off_o, ly.off_n, ly.off_q, ly.dim)}")
check("n0_mode = integer_largest_remainder",
      prob.n0_mode == "integer_largest_remainder", prob.n0_mode)
n0_int = np.all(np.abs(prob.n0 - np.rint(prob.n0)) < 1e-9)
check("n0 all integer", n0_int, f"n0={prob.n0.tolist()}")
row_ok = True
for r in range(prob.N_R):
    want = largest_remainder_counts(int(prob.N[r]), prob.init_mix)
    got = np.rint(prob.n0[r]).astype(int)
    row_ok = row_ok and np.array_equal(got, want) and got.sum() == int(prob.N[r])
check("n0 row = Python largest_remainder and sums to N", row_ok,
      f"rows sums={np.rint(prob.n0).sum(axis=1).astype(int).tolist()} N={prob.N.tolist()}")
check("I0 all zero and integer", np.all(prob.I0 == np.rint(prob.I0)),
      f"I0={prob.I0.tolist()}")
check("o0 all zero", np.all(prob.o0 == 0.0), f"o0={prob.o0.tolist()}")
print("n0 =", np.rint(prob.n0).astype(int).tolist())

print()
print("=" * 78)
print("2. Transition matrix P_ref")
print("=" * 78)
P_fresh = transition_matrix(prob.lambda0, prob.G, prob.p_digits)
check("P_ref bit-identical to transition_matrix",
      np.array_equal(prob.P_ref, P_fresh),
      f"max|d|={np.max(np.abs(prob.P_ref - P_fresh)):.3e}")
check("P_ref row sums == 1",
      np.allclose(prob.P_ref.sum(axis=1), 1.0, atol=1e-15),
      f"{prob.P_ref.sum(axis=1).tolist()}")
check("P_ref lower triangle zero",
      np.max(np.abs(np.tril(prob.P_ref[:, :prob.G], -1))) == 0.0)
print("P_ref =")
for row in prob.P_ref:
    print("  ", [f"{x:.9g}" for x in row])

print()
print("=" * 78)
print("3. Exported pool == freshly re-sampled Python Tier2 pool")
print("=" * 78)
fresh = DefineProblem(cfg)
check("pool length 300", len(prob.pool) == 300 == len(fresh.pool),
      f"json={len(prob.pool)} fresh={len(fresh.pool)}")
max_z = max_th = 0.0
exact = True
for (z1, t1), (z2, t2) in zip(prob.pool, fresh.pool):
    exact = exact and np.array_equal(z1, z2) and np.array_equal(t1, t2)
    max_z = max(max_z, float(np.max(np.abs(z1 - z2))))
    max_th = max(max_th, float(np.max(np.abs(t1 - t2))))
check("all 300 (z, Theta) scenes bit-identical", exact,
      f"max|dz|={max_z:.3e} max|dTheta|={max_th:.3e}")

# Row sums / structural zeros / first-scene fingerprint.
max_row = max_zero = 0.0
for _, th in prob.pool:
    max_row = max(max_row, float(np.max(np.abs(th.sum(axis=2) - 1.0))))
    for i in range(prob.G):
        if i:
            max_zero = max(max_zero, float(np.max(np.abs(th[:, i, :i]))))
check("Theta row sums == 1", max_row <= 1e-10, f"max|d|={max_row:.3e}")
check("Theta structural lower triangle == 0", max_zero == 0.0,
      f"max|d|={max_zero:.3e}")

# Pool mean vs E_z[P(z)] (same diagnostic as print_summary).
th_bar = np.mean([th for _, th in prob.pool], axis=0)
Pz = np.mean([transition_matrix(prob.lambda0 * np.exp(
                  np.dot(prob.beta_cov, z[:, r] - prob.z_bar)), prob.G,
                  prob.p_digits)
              for z, _ in prob.pool for r in range(prob.N_R)], axis=0)
dev = np.abs(th_bar - Pz[None, :, :])
rel = dev[:, Pz >= 0.01] / Pz[Pz >= 0.01][None, :]
print(f"E[Theta] vs E_z[P(z)]: max|d|={dev.max():.6f} "
      f"max rel(P>=0.01)={rel.max():.4%}")

z0, th0 = prob.pool[0]
c0 = tail_cdf(th0)
print("scene 0 z =", z0.tolist())
print("scene 0 Theta[1,1,:] =", th0[0, 0, :].tolist(), "row sum",
      float(th0[0, 0, :].sum()))
print("scene 0 C[1,1,:]    =", c0[0, 0, :].tolist())
check("scene 0 Theta[1,1] row sum == 1", abs(th0[0, 0, :].sum() - 1) <= 1e-12)
check("scene 0 C == cumsum(Theta) reversed",
      np.array_equal(c0, np.cumsum(th0[:, :, ::-1], axis=2)[:, :, ::-1]))

print()
print("=" * 78)
print("4. Config cross-check (JSON vs TOML)")
print("=" * 78)
pairs = [("T", prob.T, cfg.T), ("G", prob.G, cfg.G),
         ("N_R", prob.N_R, cfg.N_R), ("L", prob.L, cfg.L),
         ("seed", prob.seed, cfg.seed), ("p_digits", prob.p_digits, cfg.p_digits),
         ("lambda0", prob.lambda0, cfg.lambda0),
         ("dirichlet_s", prob.dirichlet_s, cfg.dirichlet_s),
         ("C", prob.C, cfg.C), ("h", prob.h, cfg.h), ("Cs", prob.Cs, cfg.Cs),
         ("atc", prob.atc, cfg.atc), ("tc", prob.tc, cfg.tc)]
for name, a, b in pairs:
    check(f"{name}: json {a} == toml {b}", a == b)
check("N", prob.N.tolist() == list(cfg.N))
check("Qmax", prob.Qmax.tolist() == list(cfg.Qmax))
check("Imax", prob.Imax.tolist() == list(cfg.Imax))
check("init_mix", np.array_equal(prob.init_mix, np.asarray(cfg.init_mix)))
check("fleet_tol", prob.fleet_tol == cfg.fleet_tol)
check("priority", prob.priority == fresh.priority)
check("tc_route / (tc*route order) matches",
      np.array_equal(prob.tc_route, fresh.tc_route))

print()
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
