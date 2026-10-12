# 把某一期的割**按状态块展开**：alpha / I / o / n / q 各贡献多少，
# 以及在初始状态 x₀ 处各项怎么凑出割值。诊断 LB≡0 的直接工具。
#
#   python test/_t_cutblk.py <结果目录> [期号=1] [展开条数=3]
#
# 关注点：割在 x₀ 处读 0，是哪一块「退掉了成本却没补上收益」。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def blocks(Ly, v):
    """状态向量 -> (I, o, n, q) 四块。"""
    return Ly.I(v), Ly.o(v), Ly.n(v), Ly.q(v)


def main():
    d = Path(sys.argv[1])
    d = d if d.is_absolute() else ROOT / d
    t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, NR, A = prob.layout, prob.N_R, prob.A
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    x0 = prob.initial_state()
    I0, o0, n0, q0 = blocks(Ly, x0)

    print(f"{d.name}  轮次 {meta.get('n_iterations')}  总割 {meta.get('n_cuts')}")
    print(f"T={prob.T} A={A} N_R={NR} L={prob.L}  状态维 {Ly.dim}")
    print(f"x₀: ΣI={I0.sum():.1f}  Σo={o0.sum():.1f}  Σn={n0.sum():.1f}/{prob.N.sum():.0f}  "
          f"Σq={q0.sum():.1f}   停机={prob.N.sum()-n0.sum():.1f}")

    if t0 == 0:
        # 全阶段扫描：割的锚点长什么样、在 x₀ 处能读到多少
        print(f"\n{'期':>4} {'割数':>5} {'αmax':>14} {'obj最大':>14} {'锚点停机':>9} "
              f"{'锚点Σq':>8} {'在x₀最大取值':>14}")
        for t in range(1, prob.T + 1):
            if not cuts[t]:
                continue
            ob = max((c.obj for c in cuts[t]), default=0.0)
            c_ob = max(cuts[t], key=lambda c: c.obj)
            dt = prob.N.sum() - Ly.n(c_ob.state).sum()
            vv = max(c.value_at(x0) for c in cuts[t])
            print(f"{t:>4} {len(cuts[t]):>5} {al[t-1] if False else max(c.alpha for c in cuts[t]):>14,.0f} "
                  f"{ob:>14,.0f} {dt:>9.1f} {Ly.q(c_ob.state).sum():>8.1f} {vv:>14,.1f}")
        return

    cs = cuts[t0]
    if not cs:
        print(f"阶段 {t0} 无割"); return
    al = np.array([c.alpha for c in cs])
    nb = np.array([np.linalg.norm(c.beta) for c in cs])
    vals = np.array([c.value_at(x0) for c in cs])
    o_ = np.argsort(-vals)
    print(f"\n阶段 {t0}：{len(cs)} 条割   alpha ∈ [{al.min():,.2f}, {al.max():,.2f}]   "
          f"|β| 中位 {np.median(nb):,.3f}")
    print(f"在 x₀ 处取值：最大 {vals[o_][0]:,.4f}  第2 {vals[o_][1]:,.4f}  "
          f"中位 {np.median(vals):,.2f}  最小 {vals.min():,.2f}")

    for rank, i in enumerate(o_[:topk]):
        c = cs[int(i)]
        bI, bo, bn, bq = blocks(Ly, c.beta)
        xI, xo, xn, xq = blocks(Ly, c.state)
        print(f"\n{'='*78}")
        print(f"【第 {rank+1} 名】轮 {c.it}   alpha={c.alpha:,.2f}  |β|={nb[i]:,.3f}  "
              f"obj=Q(x̄)={c.obj:,.2f}")
        print(f"  锚点 x̄: ΣI={xI.sum():.1f} Σo={xo.sum():.1f} Σn={xn.sum():.1f} "
              f"(停机 {prob.N.sum()-xn.sum():.1f}) Σq={xq.sum():.1f}")
        print(f"  在自锚点处取值 {c.value_at(c.state):,.4f}   "
              f"（应 = obj，差 {c.value_at(c.state)-c.obj:,.4e}）")

        # —— 块级分解 ——
        parts = [("α 截距   ", c.alpha, None),
                 ("β_I·I₀  ", float(bI @ I0), bI),
                 ("β_o·o₀  ", float(bo.ravel() @ o0.ravel()), bo.ravel()),
                 ("β_n·n₀  ", float(bn.ravel() @ n0.ravel()), bn.ravel()),
                 ("β_q·q₀  ", float(bq @ q0), bq)]
        print(f"  {'项':<10} {'贡献':>16}  系数（按仓）")
        for name, val, coef in parts:
            if coef is None:
                print(f"  {name:<10} {val:>16,.2f}")
            else:
                cc = coef.reshape(NR, -1).sum(axis=1) if coef.size != NR else coef
                print(f"  {name:<10} {val:>16,.2f}  " +
                      " ".join(f"{v:>10.4g}" for v in cc))
        tot = sum(v for _, v, _ in parts)
        print(f"  {'合计':<10} {tot:>16,.4f}   ← 应等于割在 x₀ 的取值 {c.value_at(x0):,.4f}")

        # —— n 块的逐月龄剖面：β_n(θ) 里有没有 θ 的痕迹 ——
        bn_r = bn.reshape(NR, A)
        nz = np.nonzero(np.abs(bn_r).sum(axis=0) > 1e-9)[0]
        print(f"  n 块：非零月龄 {len(nz)}/{A} 个"
              + (f"（a ∈ {nz.min()}..{nz.max()}）" if len(nz) else ""))
        if len(nz):
            show = nz if len(nz) <= 12 else np.unique(np.linspace(0, len(nz)-1, 12).astype(int))
            print(f"    {'a':>5} " + " ".join(f"{int(nz[k]):>9}" for k in show))
            for r in range(NR):
                print(f"    仓{r+1:<2} " + " ".join(f"{bn_r[r, nz[k]]:>9.4g}" for k in show))
            # 与 θ(a) 的相关系数：割的月龄梯度是否随失效率走
            th = np.mean([th_ for _, th_ in prob.pool], axis=0)
            for r in range(NR):
                m_ = np.abs(bn_r[r, nz]) > 1e-9
                if m_.sum() > 2 and np.std(bn_r[r, nz][m_]) > 0 and np.std(th[r, nz][m_]) > 0:
                    cc = np.corrcoef(bn_r[r, nz][m_], th[r, nz][m_])[0, 1]
                    print(f"    仓{r+1:<2} corr(β_n, θ) = {cc:+.4f}")


if __name__ == "__main__":
    main()
