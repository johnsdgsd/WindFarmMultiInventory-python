# 定位约束审计里 age 项那 1 单位的违约：是模型错了，还是「重算 ⌊θ·n⌋」在边界上翻了个？
#
#   python test/_t_agecheck.py <结果目录> [sim_Ppaths.npz]
#
# 判据：模型自己的 age 行是 n_out[a+1] = n_in[a] − F，F 是**整数变量**。
# 审计用 floor(θ·n) 重算 F —— 若 θ·n 落在整数附近（1e-6 容差内），求解器的
# 行可行性容差允许 F 取到整数 k 而 floor 给出 k−1，于是审计报出 ±1。
# 本脚本把违约点的原始数值全打印出来，直接用 θ·n 的小数部分定性。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402
from utils.io import load_json                         # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python test/_t_agecheck.py <结果目录> [sim_Ppaths.npz]")
    d = Path(sys.argv[1])
    d = d if d.is_absolute() else ROOT / d
    f = (d / sys.argv[2]) if len(sys.argv) > 2 else sorted(d.glob("sim_*paths.npz"))[-1]
    prob = DefineProblem(load_problem_config(d / "config" / "problem.toml"))
    Ly, A, T = prob.layout, prob.A, prob.T
    z = np.load(f)
    P = int(load_json(d, f.stem + ".json")["n_paths"])
    x_in, x_out, scene = z["x_in"].astype(float), z["x_out"].astype(float), z["scene"].astype(int)
    print(f"{f.name}   {P} 路径 × {T} 期 × {A} 月龄桶"
          f"（共 {P*T*A*(Ly.N_R)} 个 age 等式）")

    hits, worst = [], 0.0
    for t in range(T):
        th_all = np.array([prob.pool[scene[p, t]][1] for p in range(P)])     # (P,NR,A)
        n_in = x_in[:, t, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, Ly.N_R, A)
        n_out = x_out[:, t, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, Ly.N_R, A)
        prod = th_all * n_in
        Ff = np.floor(prod)
        pred = n_in[:, :, :A - 1] - Ff[:, :, :A - 1]
        got = n_out[:, :, 1:]
        diff = got - pred
        idx = np.argwhere(np.abs(diff) > 0.5)
        worst = max(worst, float(np.abs(diff).max()))
        for p, r, a in idx:
            Fmodel = n_in[p, r, a] - got[p, r, a]
            hits.append((t + 1, p, r, a, th_all[p, r, a], n_in[p, r, a],
                         prod[p, r, a], Ff[p, r, a], Fmodel, diff[p, r, a]))

    print(f"违约点 {len(hits)} 个    最大偏差 {worst:.0f}\n")
    if not hits:
        print("没有违约 ⇒ 审计报的那 1 单位不在 age 项上（看看是不是别的项）")
        return
    print("   期  路径 仓 月龄        θ            n        θ·n     floor   模型F  偏差"
          "      θ·n 距最近整数")
    for (t, p, r, a, th, n, pr, ff, fm, df) in hits[:25]:
        frac = abs(pr - round(pr))
        print(f"  {t:>4} {p:>4} {r:>2} {a:>4}  {th:.9f} {n:>8.0f} {pr:>12.9f} "
              f"{ff:>7.0f} {fm:>7.0f} {df:>5.0f}   {frac:.2e}")
    if len(hits) > 25:
        print(f"  …（其余 {len(hits)-25} 个略）")
    fr = np.array([abs(h[6] - round(h[6])) for h in hits])
    print(f"\n违约点的 |θ·n − 最近整数|：最大 {fr.max():.3e}，中位 {np.median(fr):.3e}")
    from code.config import load_solver_config
    sc = load_solver_config(d / "config" / "solver.toml")
    print(f"FeasTol={sc.feas_tol:g}  IntTol={sc.int_tol:g}  DualTol={sc.dual_tol:g}")
    print("判定：" + ("全部落在容差边界上 ⇒ 是「重算 ⌊θ·n⌋」在整数附近翻档，"
                    "**不是模型违约**（模型自己的 F 是整数变量，n_out = n_in − F 恒精确相等）"
                    if fr.max() <= max(sc.feas_tol, sc.int_tol) * 10 + 1e-9
                    else "★ 有非边界的偏差 ⇒ 需要进一步查"))


if __name__ == "__main__":
    main()
