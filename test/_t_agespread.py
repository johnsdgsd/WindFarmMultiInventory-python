# 期初月龄摊开宽度 K 的扫描：rint 口径下逐期失效量、峰值、零失效期数。
#
#   python test/_t_agespread.py [K列表]
#
# 目的：rint 要求每个 (仓, 月龄) 桶 θ·n ≳ 0.5，所以「摊多宽」不是自由选择 ——
# 摊太宽每桶台数不足，整桶归零。本脚本给出可行区间与对应峰值，用于定 Qmax。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config      # noqa: E402
from code.problem_data import DefineProblem      # noqa: E402


def profile(prob, K, mode="rint"):
    """期初把每仓机队均摊在月龄 0..K−1，保持满编演化，返回逐期失效数（系统级）。"""
    NR, A, T = prob.N_R, prob.A, prob.T
    th = np.mean([t for _, t in prob.pool], axis=0)
    n = np.zeros((NR, A), dtype=float)
    for a in range(K):
        n[:, a] = prob.N / K
    out = []
    for _ in range(T):
        e = th * n
        if mode == "rint":
            f = np.rint(e)
        elif mode == "exact":
            f = e
        else:
            f = np.ceil(e)
        out.append(float(f.sum()))
        nxt = np.zeros_like(n)
        nxt[:, 0] = f.sum(axis=1)              # 满编：失效各自补回 0 月龄
        nxt[:, 1:] = n[:, :-1] - f[:, :-1]     # 存活者月龄 +1（rint 口径扣整数台）
        n = nxt
    return np.asarray(out)


def main():
    ks = [int(v) for v in sys.argv[1:]] or [1, 5, 10, 15, 20, 30, 40, 60]
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    prob = DefineProblem(pcfg)
    print(f"T={prob.T} A={prob.A} η={prob.eta} 机队 {int(prob.N.sum())} 台  "
          f"Qmax 合计 {int(prob.Qmax.sum())}")
    print(f"\n{'K':>4} {'合计失效':>9} {'峰值/月':>8} {'零失效期':>8} "
          f"{'末期失效':>9} {'峰值/仓-最大仓':>14} {'rint丢失':>9}")
    for K in ks:
        r = profile(prob, K, "rint")
        e = profile(prob, K, "exact")
        peak_wh = 0.0
        # 单仓峰值：用最大仓 N=1080 按比例估
        peak_wh = r.max() * prob.N.max() / prob.N.sum()
        print(f"{K:>4} {r.sum():>9.0f} {r.max():>8.1f} {int((r < 0.5).sum()):>8} "
              f"{r[-1]:>9.1f} {peak_wh:>14.1f} {r.sum()/max(e.sum(),1e-9)-1:>8.1%}")

    # 逐期曲线（选定几个 K）
    for K in [k for k in (1, 10, 20, 30) if k in ks]:
        r = profile(prob, K, "rint")
        e = profile(prob, K, "exact")
        print(f"\n  K={K} 逐期（每 6 期）：")
        idx = list(range(0, prob.T, 6))
        print(f"    {'期':>4} " + " ".join(f"{i+1:>7}" for i in idx))
        print(f"    {'rint':>4} " + " ".join(f"{r[i]:>7.1f}" for i in idx))
        print(f"    {'精确':>4} " + " ".join(f"{e[i]:>7.1f}" for i in idx))


if __name__ == "__main__":
    main()
