# _t_grade_calib.py — 等级模型数学层自检（不训练）：
#   P 行和/上三角/3 位小数；λ0=0.007 的 20 年失效率/首达时间；
#   场景池 Θ 行和/结构零/E[Θ]=E_z[P(z)]；状态维 N_R(G+L+1)。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config                       # noqa: E402
from code.problem_data import (DefineProblem, cumulative_failure_prob,  # noqa: E402
                               expected_first_passage, transition_matrix,
                               _pool_mean_P, _pool_mean_theta)


def main() -> None:
    cfg = load_problem_config(ROOT / "config" / "problem.toml")
    prob = DefineProblem(cfg)
    G = prob.G

    # ---- P ----
    P = prob.P_ref
    assert P.shape == (G, G + 1), P.shape
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(P >= 0)
    assert np.allclose(np.tril(P[:, :G], -1), 0.0)
    # 3 位小数：每个元素应是 0.001 的整数倍
    assert np.max(np.abs(P * 1000.0 - np.rint(P * 1000.0))) < 1e-9
    expected = np.array([
        [0.993, 0.007, 0.000, 0.000, 0.000],
        [0.000, 0.993, 0.007, 0.000, 0.000],
        [0.000, 0.000, 0.993, 0.007, 0.000],
        [0.000, 0.000, 0.000, 0.993, 0.007],
    ])
    assert np.allclose(P, expected, atol=1e-12), P

    # ---- 20 年失效率/首达（标定口径固定为「全新机队」，与实验的 init_mix 无关）----
    fresh = np.array([1.0, 0.0, 0.0, 0.0])
    pT = cumulative_failure_prob(P, 240, fresh)
    P_exact = transition_matrix(prob.lambda0, G, None)
    pT_exact = cumulative_failure_prob(P_exact, 240, fresh)
    mttf = expected_first_passage(P, fresh)
    assert 0.0893 <= pT <= 0.0897, pT
    assert 0.0900 <= pT_exact <= 0.0904, pT_exact
    assert 560.0 <= mttf <= 580.0, mttf

    # ---- 状态维/初值 ----
    assert prob.state_dim == prob.N_R * (G + prob.L + 1) == 40
    assert np.allclose(prob.n0.sum(axis=1), prob.N)
    assert np.allclose(prob.n0, np.outer(prob.N, prob.init_mix))

    # ---- 池 ----
    theta_bar = _pool_mean_theta(prob.pool, prob.N_R, G)
    Pz_bar = _pool_mean_P(prob.pool, prob.lambda0, prob.beta_cov,
                          prob.z_bar, G, prob.p_digits)
    dev = np.abs(theta_bar - Pz_bar[None, :, :]).max()
    for _, th in prob.pool:
        assert th.shape == (prob.N_R, G, G + 1)
        assert np.max(np.abs(th.sum(axis=2) - 1.0)) < 1e-12
        for i in range(G):
            if i:
                assert np.all(th[:, i, :i] == 0.0)
    assert dev < 0.01, dev

    print(f"P_ref (λ0={prob.lambda0}, p_digits={prob.p_digits}):")
    print(P)
    print(f"20 年累计失效: 3 位小数 {pT:.4%}  精确 {pT_exact:.4%}")
    print(f"期望首达: {mttf:.1f} 月")
    print(f"状态维: {prob.state_dim}  init_mix={prob.init_mix.tolist()}")
    print(f"E[Θ] vs E_z[P(z)] max|Δ|={dev:.4f}  ({len(prob.pool)} 条场景)")
    print("OK _t_grade_calib")


if __name__ == "__main__":
    main()
