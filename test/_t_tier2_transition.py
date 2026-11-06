# _t_tier2_transition.py — Tier2 尾累计整数转移：N 行和、尾带、确定性与 F/W 整数。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from code.transition import nearest_tail_flow, tail_cdf          # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, NR, G = prob.layout, prob.N_R, prob.G
    rng = np.random.default_rng(2026)
    sp = StageSubproblem(1, prob, scfg)
    worst_tail = 0.0
    worst_helper = 0.0
    worst_sum = 0.0
    n_check = 0
    try:
        for k in range(12):
            x = np.asarray(prob.initial_state(), dtype=float).copy()
            # 随机整数 n，每仓行和严格等于 N_r
            n = np.zeros((NR, G), dtype=int)
            for r in range(NR):
                cuts = np.sort(rng.choice(np.arange(1, int(prob.N[r])),
                                          size=G - 1, replace=False))
                n[r] = np.diff(np.r_[0, cuts, int(prob.N[r])])
            x[Ly.off_n:Ly.off_n + Ly.n_n] = n.ravel()
            for sidx in [k % len(prob.pool), (17 * k + 5) % len(prob.pool)]:
                sc = prob.pool[sidx]
                th = np.asarray(sc[1], float)
                sp.assemble(None, key=None)
                sp.set_point(x, sc, require_int=True)
                sp.solve_mip()
                dv = sp.decision_values()
                N = np.asarray(dv["N"], float)
                assert np.max(np.abs(N - np.rint(N))) <= 1e-7
                assert np.max(np.abs(N.sum(axis=2) - n)) <= 1e-7
                for r in range(NR):
                    for i in range(G):
                        if i:
                            assert np.max(np.abs(N[r, i, :i])) <= 1e-9
                ct = tail_cdf(th)
                for r in range(NR):
                    for i in range(G):
                        for j in range(i + 1, G + 1):
                            t = float(N[r, i, j:].sum())
                            worst_tail = max(
                                worst_tail,
                                abs(t - float(ct[r, i, j]) * n[r, i]))
                nh = nearest_tail_flow(n, th)
                worst_helper = max(worst_helper,
                                   float(np.max(np.abs(N - nh))))
                worst_sum = max(worst_sum,
                                float(np.max(np.abs(np.asarray(dv["n_out"]).sum(1)
                                                    - (n.sum(1) - N[:, :, G].sum(1)
                                                       + np.asarray(dv["y"]).sum(1))))))
                assert np.max(np.abs(np.asarray(dv["F_new"])
                                     - N[:, :, G].sum(1))) <= 1e-7
                assert np.max(np.abs(np.asarray(dv["W_in"])
                                     + np.asarray(dv["F_new"])
                                     - np.asarray(dv["y"]).sum(1)
                                     - np.asarray(dv["W_out"]))) <= 1e-7
                n_check += 1
    finally:
        sp.free()
    # 尾最近取整不应像 floor/ceil 那样系统性偏置 F
    eff = []
    for sc in prob.pool:
        th = np.asarray(sc[1], float)
        for _ in range(3):
            ntest = np.zeros((NR, G), dtype=int)
            for r in range(NR):
                cuts_ = np.sort(rng.choice(np.arange(1, int(prob.N[r])),
                                           size=G - 1, replace=False))
                ntest[r] = np.diff(np.r_[0, cuts_, int(prob.N[r])])
            nh = nearest_tail_flow(ntest, th)
            eff.append(float(nh[:, :, G].sum(1).sum()
                             - (th[:, :, G] * ntest).sum()))
    mean_bias = float(np.mean(eff))
    assert abs(mean_bias) < 0.2, mean_bias
    print(f"  [PASS] F 尾取整平均偏差 {mean_bias:+.4f} 台/仓·源")
    assert worst_tail <= 0.5 + 1e-6, worst_tail
    assert worst_helper <= 1e-7, worst_helper
    print(f"  [PASS] {n_check} 个 (状态,场景)：尾误差 {worst_tail:.3e}，"
          f"N 与确定性 helper 最大差 {worst_helper:.3e}，"
          f"n_out 行和审计 {worst_sum:.3e}")
    print("OK _t_tier2_transition")


if __name__ == "__main__":
    main()
