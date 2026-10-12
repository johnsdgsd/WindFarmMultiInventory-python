# n 的整性：**变量类型**与**数值整性**是两件事，分开查。
#
# 变量类型：_build 里只有 q 块写 vtype=INTEGER，n 一直是连续变量（改前改后相同）。
# 数值整性：改前 age 行是 n_out[a+1] = n_in[a] − rint(θ·n_in[a])，整数进 ⇒ 整数出；
#           改后是 n_out[a+1] = (1−θ)·n_in[a]，θ 是分数 ⇒ 一般不再整数。
#           本脚本用生产配置（spread_init=false，初值全为整数）逐步 MIP 前向，直接量。
#
#   python test/_t_nint.py [步数]
import sys
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402


def frac(v):
    """到最近整数的最大距离：0 = 全整数。"""
    a = np.asarray(v, dtype=float)
    return float(np.abs(a - np.rint(a)).max())


def main():
    n_step = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    print(f"生产配置 T={prob.T} A={prob.A} N_R={prob.N_R} "
          f"spread_init={pcfg.spread_init}   状态维 {Ly.dim}\n")

    # ---- 变量类型：直接从 COPT 读回 ----
    sub = StageSubproblem(1, prob, scfg)
    sub.assemble(None)
    # getType() 返回 'C'/'I'/'B' 字符，不是数字常量
    kinds = {}
    for j in range(Ly.dim):
        kinds.setdefault(str(sub.x_out[j].getType()), []).append(j)
    for t, js in sorted(kinds.items()):
        tag = "  ← q 块" if Ly.off_q in js else ("  ← n 块" if Ly.off_n in js else "")
        print(f"  x_out 类型 {t:>2}（C=连续 I=整数 B=二元）  共 {len(js):>3} 个{tag}")

    # ---- 数值整性：逐步 MIP 前向（无割，只看状态怎么走；θ 一取分数就露馅） ----
    print(f"\n逐步 MIP 前向（初值全整数，看 n 块是否保持整数）：")
    sub.assemble(None, key="none")          # theta 钉 0，模型只做当期最优
    x = np.asarray(prob.initial_state(), dtype=float)
    print(f"  期  0   n 块最大离整距离 = {frac(Ly.n(x)):.6f}   "
          f"Σn = {Ly.n(x).sum():,.1f}")
    for t in range(1, n_step + 1):
        j = t % len(prob.pool)              # 换着场景取，避免单一 θ
        sub.set_point(x, prob.pool[j])
        sub.solve_mip()
        x = sub.out_state_vector()
        nb = Ly.n(x)
        th = np.asarray(prob.pool[j][1], dtype=float)
        old = Ly.n(sub._state_vec) - np.rint(th * Ly.n(sub._state_vec))   # 改前口径
        print(f"  期 {t:>2}   n 块最大离整距离 = {frac(nb):.6f}   "
              f"Σn = {nb.sum():,.1f}   "
              f"（改前口径 n_in − rint(θn) 的离整距离 {frac(old):.6f}）")

    print("\n结论：x_out 若全为 I（整数）且离整距离恒为 0 ⇒ 状态全程停在整数格点上。")
    sub.free()


if __name__ == "__main__":
    main()
