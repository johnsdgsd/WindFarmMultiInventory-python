# 逐仓逐期失效量（rint 口径），用于定 Qmax / Imax。
#
#   python test/_t_demand.py [K]
#
# 判据：峰值月失效 > Qmax ⇒ 缺货必然发生 ⇒ Q(满编空仓) > 0 ⇒ LB 能爬起来。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config      # noqa: E402
from code.problem_data import DefineProblem      # noqa: E402


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    prob = DefineProblem(pcfg)
    NR, A, T = prob.N_R, prob.A, prob.T
    th = np.mean([t for _, t in prob.pool], axis=0)

    n = np.zeros((NR, A))
    if K <= 0:                                  # 0 = 全新机队，全在月龄 0
        n[:, 0] = prob.N
    else:
        n[:, :K] = prob.N[:, None] / K
    F = np.zeros((T, NR))
    for t in range(T):
        e = np.rint(th * n)
        F[t] = e.sum(axis=1)
        nxt = np.zeros_like(n)
        nxt[:, 0] = e.sum(axis=1)
        nxt[:, 1:] = n[:, :-1] - e[:, :-1]
        n = nxt

    print(f"K={K}  T={T}  机队 {prob.N.tolist()}  现行 Qmax {prob.Qmax.tolist()}")
    print(f"\n{'期':>4} " + " ".join(f"{r+1:>6}" for r in range(NR)) + f" {'合计':>7}")
    for t in list(range(3, T, 3)) + [T - 1]:
        print(f"{t+1:>4} " + " ".join(f"{F[t, r]:>6.0f}" for r in range(NR))
              + f" {F[t].sum():>7.0f}")
    print(f"\n  {'仓':>3} {'峰值':>6} {'均值':>6} {'合计':>7} {'现行Qmax':>9} {'峰值/Qmax':>10}")
    for r in range(NR):
        q = prob.Qmax[r]
        print(f"  {r+1:>3} {F[:, r].max():>6.0f} {F[:, r].mean():>6.2f} {F[:, r].sum():>7.0f} "
              f"{q:>9} {F[:, r].max()/q:>10.2f}")
    print(f"  系统 峰值 {F.sum(axis=1).max():.0f}  均值 {F.sum()/T:.2f}  "
          f"Qmax 合计 {prob.Qmax.sum()}")


if __name__ == "__main__":
    main()
