# 取整口径的代价：rint / ceil / floor 相对「精确期望失效数 θ·n」差多少。
#
#   python test/_t_round3.py
#
# 用生产配置，模拟「机队保持满编、失效当期补新件」的月龄演化，逐期累加三种口径。
# 覆盖两种期初月龄：全 0（现状 spread_init=false）与 0..T−1 均匀（现实混龄）。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config      # noqa: E402
from code.problem_data import DefineProblem      # noqa: E402


def evolve(prob, n0):
    """机队保持满编的月龄演化，返回逐期 (精确, rint, ceil, floor)。"""
    NR, A, T = prob.N_R, prob.A, prob.T
    th = np.mean([t for _, t in prob.pool], axis=0)      # (NR, A) 池平均 θ
    n = n0.copy()
    rows = []
    for _ in range(T):
        e = th * n                                        # 逐桶期望失效
        f_exact = float(e.sum())
        f_rint = float(np.rint(e).sum())
        f_ceil = float(np.ceil(e).sum())
        f_floor = float(np.floor(e).sum())
        rows.append((f_exact, f_rint, f_ceil, f_floor))
        # 失效与到寿都离场，补新件进 a=0；到寿档（A−1）θ=1 全走
        leaving = e / np.maximum(th, 1e-300) * 0.0        # 占位，下面直接算
        nxt = np.zeros_like(n)
        nxt[:, 0] = e.sum(axis=1)                         # 满编 ⇒ 各仓失效各自补回 0 月龄
        nxt[:, 1:] = n[:, :-1] * (1.0 - th[:, :-1])
        n = nxt
    return np.asarray(rows)


def report(tag, prob, n0):
    rows = evolve(prob, n0)
    tot = rows.sum(axis=0)
    print(f"\n【{tag}】")
    print(f"  {'期':>4} {'精确 θ·n':>10} {'rint':>8} {'ceil':>8} {'floor':>8}")
    for t in list(range(1, prob.T + 1, 6)) + [prob.T]:
        e, ri, ce, fl = rows[t - 1]
        print(f"  {t:>4} {e:>10.2f} {ri:>8.0f} {ce:>8.0f} {fl:>8.0f}")
    print(f"  合计  精确 {tot[0]:>10.1f}  rint {tot[1]:>8.0f} ({tot[1]/tot[0]-1:+.1%})  "
          f"ceil {tot[2]:>8.0f} ({tot[2]/tot[0]-1:+.1%})  "
          f"floor {tot[3]:>8.0f} ({tot[3]/tot[0]-1:+.1%})")
    # 有货桶数与单桶期望：rint 什么时候失灵
    th = np.mean([t for _, t in prob.pool], axis=0)
    n = n0.copy()
    occ, emax = [], []
    for _ in range(prob.T):
        e = th * n
        occ.append(int((n > 1e-9).sum()))
        emax.append(float(e.max()))
        nxt = np.zeros_like(n)
        nxt[:, 0] = e.sum(axis=1)
        nxt[:, 1:] = n[:, :-1] * (1.0 - th[:, :-1])
        n = nxt
    print(f"  非空桶数 {occ[0]} → {occ[-1]}    单桶最大期望 θ·n {emax[0]:.3f} → {emax[-1]:.3f}")


def main():
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    prob = DefineProblem(pcfg)
    Ly, NR, A = prob.layout, prob.N_R, prob.A
    print(f"T={prob.T} A={A}  η={prob.eta} m={prob.m}  机队 {prob.N.tolist()}")

    n0 = np.zeros((NR, A)); n0[:, 0] = prob.N
    report("期初全 0 月龄（现状）", prob, n0)

    n1 = np.zeros((NR, A))
    for a in range(prob.T):                     # 0..T−1 均匀，留出 A−T 的到寿余量
        n1[:, a] = prob.N / prob.T
    report(f"期初 0..{prob.T-1} 均匀（现实混龄）", prob, n1)


if __name__ == "__main__":
    main()
