# 取整对失效量的影响：**只做分析，不改任何定标结果、不改模型**。
#
# 模型现在用两条式定义 F（code/subproblem.py）：
#     Fdef_hi: F − θ·n ≤ 0     Fdef_lo: F − θ·n ≥ −1     F ∈ ℤ   ⇒ F = ⌊θn⌋
# 换成 rint 只需把右端常数改成 +0.5 / −0.5（不碰结构）。
# 本脚本并行跑：真值（不取整）/ ⌊⌋ / rint / ⌈⌉，比较**失效量的数量与时序**。
#
#   python test/_t_roundoff.py [m] [eta] [A] [T]
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

# 定标结果（docs/系统参数定标.md §5），仅作输入，不改动
N0 = np.array([290, 230, 145, 350, 85])


def theta(a, m, eta):
    """月龄 a 的月失效概率 = 1 − S(a+1)/S(a)，S(a)=exp(−(a/η)^m)。"""
    return 1.0 - np.exp(-((a + 1) / eta) ** m + (a / eta) ** m)


def rnd(x, mode):
    return {"floor": np.floor, "ceil": np.ceil, "rint": np.rint}.get(mode, lambda v: v)(x)


def run(m, eta, A, T, N, mode="floor"):
    NR = len(N)
    n = np.zeros((NR, A))
    n[:, 0] = N                                      # spread_init = false
    th = theta(np.arange(A), m, eta)
    truth = np.zeros((NR, T))
    model = np.zeros((NR, T))
    n_t, n_m = n.copy(), n.copy()
    for t in range(T):
        ft, fm = th * n_t, rnd(th * n_m, mode)
        truth[:, t], model[:, t] = ft.sum(axis=1), fm.sum(axis=1)
        for nm, F in ((n_t, ft), (n_m, fm)):
            new = np.zeros_like(nm)
            new[:, 1:] = nm[:, :-1] - F[:, :-1]
            new[:, 0] = F.sum(axis=1)
            nm[:] = new
    return truth, model


def main():
    m = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 148.0
    A = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    # 第 5 个参数：各仓等量台数（不给则用定标稿的异质分布）
    N = (np.full(len(N0), int(sys.argv[5])) if len(sys.argv) > 5 else N0)
    print(f"参数：m={m}  η={eta}  A={A}  T={T}  N={N.tolist()}（共 {N.sum()} 台）")
    print(f"θ(1)={theta(1,m,eta):.2e}  θ({A-1})={theta(A-1,m,eta):.5%}"
          f"  ⇒ 满编时 θ·n 跨度 {theta(1,m,eta)*N.min():.2e} ~ "
          f"{theta(A-1,m,eta)*N.max():.2f}")
    print(f"失效概率：质保内(前 {T} 期) {1-np.exp(-(T/eta)**m):.2%}"
          f"   累计 {T} 期后累计 {1-np.exp(-(2*T/eta)**m):.2%}")

    tr, md = run(m, eta, A, T, N, "rint")
    _, mf = run(m, eta, A, T, N, "floor")
    base = tr.sum()
    print("\n" + "=" * 80)
    print("精度诊断（spread_init=false）")
    print("=" * 80)
    print(f"  60 期累计失效：真值 {base:.1f} 台   "
          f"rint {md.sum():.0f} 台（{md.sum()/base-1:+.1%}）   "
          f"⌊⌋ {mf.sum():.0f} 台（{mf.sum()/base-1:+.1%}）")
    print(f"\n  逐仓：")
    print(f"  {'仓':>3} {'N_r':>5} {'真值':>8} {'rint':>7} {'偏差':>8} {'首次失效期(真/rint)':>20}")
    for r in range(len(N)):
        ft, fm = tr[r], md[r]
        i_t = int(np.argmax(ft > 0) + 1) if (ft > 0).any() else -1
        i_m = int(np.argmax(fm > 0) + 1) if (fm > 0).any() else -1
        print(f"  {r+1:>3} {int(N[r]):>5} {ft.sum():>8.1f} {fm.sum():>7.0f} "
              f"{fm.sum()-ft.sum():>+8.0f}      {i_t if i_t>0 else '—':>8} / "
              f"{i_m if i_m>0 else '—'}")
    print(f"\n  逐期（每 5 期取值，真值 → rint）：")
    ta, ma = tr.sum(axis=0), md.sum(axis=0)
    print("     期:  " + " ".join(f"{t+1:>5}" for t in range(0, T, 5)))
    print("   真值 :  " + " ".join(f"{ta[t]:>5.1f}" for t in range(0, T, 5)))
    print("   rint :  " + " ".join(f"{ma[t]:>5.0f}" for t in range(0, T, 5)))
    err = np.where(ta > 1e-9, ma / np.maximum(ta, 1e-9), np.nan)
    fin = ~np.isnan(err)
    print(f"\n  逐期相对误差（仅取真值>0 的期）：均值 {np.nanmean(err[fin]):.1%}"
          f"   中位 {np.nanmedian(err[fin]):.1%}"
          f"   最小 {np.nanmin(err[fin]):.1%}   最大 {np.nanmax(err[fin]):.1%}")

    print("\n" + "=" * 80)
    print("精度随【机队规模】的变化（m、η、A、T 不变，N 等比缩放）")
    print("=" * 80)
    print(f"  {'倍数':>6} {'N 合计':>8} {'年失效':>9} {'真值累计':>10} {'rint累计':>10} "
          f"{'rint偏差':>10} {'⌊⌋累计':>9} {'⌊⌋偏差':>9}")
    for s in (1, 2, 3, 5, 10, 20, 34.5):
        Ns = np.maximum(np.rint(N0 * s).astype(int), 1)
        t_, r_ = run(m, eta, A, T, Ns, "rint")
        _, f_ = run(m, eta, A, T, Ns, "floor")
        tb = t_.sum()
        yr = (Ns.sum() / 12) if False else None
        print(f"  {s:>6.1f} {Ns.sum():>8,} {'':>9} {tb:>10.1f} {r_.sum():>10.0f} "
              f"{(r_.sum()/tb-1):>+9.1%} {f_.sum():>9.0f} {(f_.sum()/tb-1):>+9.1%}")
    print("\n  注：年失效 ≈ N合计/10.93（MTTF 年）——N 放大 34.5 倍回到 38,000 台时"
          "年失效回到 ~3,500 台")


if __name__ == "__main__":
    main()
