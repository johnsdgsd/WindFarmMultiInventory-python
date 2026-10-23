# test/_t_fleet035.py — 机队 ×0.35 参数组的先验自检（不训练）：
#   ① 蜜月期（rint 口径下首台失效出现在第几期）
#   ② F 峰值 / ΣQmax 的实际比值，以及缺口从第几期开始
#   ③ Imax ≥ L·Qmax 的管道自洽条件
#
# 用 pool 里的场景在「不做任何决策、全部失效都不修」的极端下推 n 轨迹，
# 得到 F 的上界序列 —— 真实策略下 F 只会更小，所以这是保守判据。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402


def main():
    prob = DefineProblem(load_problem_config(ROOT / "config" / "problem.toml"))
    NR, A, T, L = prob.N_R, prob.A, prob.T, prob.L
    N = prob.N.astype(float)
    print(f"N={prob.N.tolist()}  ΣN={prob.N.sum():,.0f}")
    print(f"Qmax={prob.Qmax.tolist()}  ΣQmax={prob.Qmax.sum()}")
    print(f"Imax={prob.Imax.tolist()}  ΣImax={prob.Imax.sum()}")
    print(f"Imax/Qmax={np.round(prob.Imax/prob.Qmax, 2).tolist()}   "
          f"L·Qmax={[L*q for q in prob.Qmax]}  "
          f"自洽 {'✓' if (prob.Imax >= L*prob.Qmax).all() else '✗ 管道会先把订购率压住'}")

    # ① 蜜月期：h0(a)·N_r 何时越过 rint 的 0.5 阈值
    a = np.arange(1, A)
    h0 = (prob.m / prob.eta) * (a / prob.eta) ** (prob.m - 1.0)
    print("\n① 蜜月期（rint 要求每桶 θ·n ≥ 0.5）")
    print("     仓       N     首个 h0·N≥0.5 的月龄")
    for r in range(NR):
        k = np.argmax(h0 * N[r] >= 0.5)
        print(f"   {r+1:>4} {N[r]:>7,.0f}   {a[k] if h0[k]*N[r] >= 0.5 else '>A':>10}")

    # ② F 序列：全不修的最坏情形（失效的空位不补），按 pool 的期望 θ 逐期推
    print("\n② F 序列（全不修上界，pool 300 场景平均；真实策略下 F 只会更小）")
    theta_bar = np.mean([th for _, th in prob.pool], axis=0)      # (NR, A)
    n = np.zeros((NR, A))
    n[:, 0] = N
    print("     期      F      F−ΣQmax    累计缺口")
    cum = tot = 0.0
    first = None
    for t in range(T):
        F_age = np.rint(theta_bar * n)                           # (NR, A) 逐月龄失效
        Ft = F_age.sum()
        gap = Ft - prob.Qmax.sum()
        if gap > 0 and first is None:
            first = t + 1
        cum += max(gap, 0.0)
        tot += Ft
        if t + 1 in (1, 5, 8, 9, 10, 15, 20, 30, 40, 45, 50, 53, 55, 57, 58, 59, 60):
            print(f"   {t+1:>5} {Ft:>7.1f} {gap:>+10.1f} {cum:>10.1f}"
                  f"{'   ← 首次超顶' if t + 1 == first else ''}")
        nxt = np.zeros_like(n)                                   # 全不修：n_out[0] = 0
        nxt[:, 1:] = n[:, :-1] - F_age[:, :A - 1]
        n = nxt
    print(f"   全期失效合计 {tot:,.0f} 台（占 ΣN 的 {tot/prob.N.sum():.2%}）；"
          f"首超顶在第 {first} 期；累计缺口 {cum:,.0f} 台")


if __name__ == "__main__":
    main()
