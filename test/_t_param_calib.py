# 系统参数定标测算：按老师的三个约束反解 (m, η, W, A, N)。
#
#   ① 质保期内失效占全部失效 ≈ 10%
#   ② 年失效台数 ∈ [十几, 100]
#   ③ 规划周期 T < 寿命（现在 T=200 月 > η=120 月，倒挂）
#
# 模型口径（code/problem_data.py）：
#   h0(a) = (m/η)(a/η)^(m−1)     S(a) = exp(−(a/η)^m)     MTTF = η·Γ(1+1/m)
#   年失效 = N·12/MTTF(月)       质保占比 = [1−S(W)]/[1−S(A)]
#
# **另一个必须一起看的量：每桶的 θ·n**。模型的失效量是 F = ⌊θ·n⌋（整数变量），
# 只有当「每桶台数 n × 月失效率 θ」≳1 时这个取整才不把失效抹掉。机队小 + 月龄桶细
# ⇒ θ·n ≪ 1 ⇒ F ≡ 0 ⇒ 模型认为机队永不失效。下面逐候选打印这个诊断量。
#
#   python test/_t_param_calib.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.problem_data import weibull_hazard       # noqa: E402


def share_in_warranty(m, eta, W, A):
    """质保期内失效 / 全部失效。"""
    S = lambda a: np.exp(-(a / eta) ** m)
    return (1.0 - S(W)) / (1.0 - S(A))


def solve_eta_for_share(m, W, target=0.10, ratio=2.0):
    """给定 m、质保期 W、目标占比与 A=ratio·η，反解 η。"""
    # 1 − S(W) = target·(1 − S(ratio·η))；S(ratio·η) = exp(−ratio^m) 与 η 无关
    rhs = target * (1.0 - np.exp(-ratio ** m))
    x = (-np.log(1.0 - rhs)) ** (1.0 / m)          # W/η
    return W / x


def audit(m, eta, A, N, NR, bin_w, W, T):
    """按稳态月龄分布算各项指标。bin_w = 月龄分箱宽度（月）。"""
    a = np.arange(A)
    S = np.exp(-(a / eta) ** m)
    nr = N / NR                                     # 每仓台数
    n_a = nr * S / S.sum()                          # 稳态每桶台数（替换式更新）
    # 按 bin_w 分箱：箱内条件失效率 θ_bin = 1 − S(a_end)/S(a_start)，n_bin = Σn_a
    nb = A // bin_w
    th_b, n_b = [], []
    for k in range(nb):
        i0, i1 = k * bin_w, min((k + 1) * bin_w, A)
        s0, s1 = S[i0], S[i1] if i1 < A else 0.0
        th_b.append(1.0 - (s1 / s0 if s0 > 0 else 0.0))
        n_b.append(n_a[i0:i1].sum())
    th_b, n_b = np.array(th_b), np.array(n_b)
    # θ_bin 是**整箱**（bin_w 个月）的失效概率，除以箱宽才是月失效率
    exp_f = (n_b * th_b).sum() * NR / bin_w         # 每月期望失效（真值）
    mdl_f = (np.floor(n_b * th_b)).sum() * NR / bin_w   # 模型口径 ⌊θ·n⌋
    mttf_m = eta * float(np.math.gamma(1 + 1 / m)) if hasattr(np, "math") else \
        eta * float(__import__("math").gamma(1 + 1 / m))
    return {
        "月龄桶数": A, "箱宽": bin_w, "箱数": nb,
        "每箱台数(中位)": float(np.median(n_b)),
        "θ·n 中位": float(np.median(n_b * th_b)),
        "θ·n 最小": float((n_b * th_b).min()),
        "每期失效(真值)": exp_f, "每期失效(模型⌊⌋)": mdl_f,
        "取整损失": 1 - mdl_f / max(exp_f, 1e-12),
        "年失效": exp_f * 12, "年失效(模型)": mdl_f * 12,
        "MTTF(月)": mttf_m, "MTTF(年)": mttf_m / 12,
        "质保占比": share_in_warranty(m, eta, W, A),
        "P(过A)": float(np.exp(-(A / eta) ** m)),
        "T<η": T < eta,
    }


def main():
    W = 60          # 质保期（月）= 5 年
    print("=" * 78)
    print(f"① 先由「质保期内失效占 10%」反解 η（取 A = 2η ⇒ P(过A)≈0）")
    print("=" * 78)
    print(f"   质保期 W = {W} 月（5 年）    A = 2η")
    print(f"   {'m':>5} {'反解 η(月)':>12} {'η(年)':>8} {'MTTF(月)':>10} {'MTTF(年)':>9} "
          f"{'P(过A)':>9}   年失效=N/MTTF年")
    rows = []
    for m in (2.0, 2.5, 3.0):
        eta = solve_eta_for_share(m, W, 0.10, 2.0)
        mttf = eta * float(__import__("math").gamma(1 + 1 / m))
        rows.append((m, eta, mttf))
        print(f"   {m:>5.1f} {eta:>12.1f} {eta/12:>8.2f} {mttf:>10.1f} {mttf/12:>9.2f} "
              f"{np.exp(-2.0**m):>9.4f}")
    print("\n   ⇒ 与 Julia 原版记录的 η=120~144 月 相符的是 m=2.5（η≈148）与 m=3（η≈127）")

    print("\n" + "=" * 78)
    print("② 再由「年失效 十几~100 台」反解机队 N = 年失效 × MTTF(年)")
    print("=" * 78)
    for m, eta, mttf in rows:
        print(f"   m={m}  η={eta:.0f} 月（MTTF {mttf/12:.2f} 年）："
              f"  λ=15 台/年 ⇒ N={15*mttf/12:>7,.0f}    "
              f"λ=50 ⇒ N={50*mttf/12:>7,.0f}    λ=100 ⇒ N={100*mttf/12:>7,.0f}")

    print("\n" + "=" * 78)
    print("③ 关键诊断：机队缩小后，⌊θ·n⌋ 会不会把失效整段抹掉？")
    print("=" * 78)
    print("   （模型 F=⌊θ·n⌋；θ·n≪1 时 F≡0 ⇒ 机队永不失效。用稳态月龄分布、"
          "分箱宽度 bin_w 评估）")
    hdr = (f"   {'m':>4} {'η':>5} {'N':>6} {'A':>5} {'箱宽':>5} {'箱数':>5} "
           f"{'每箱台数':>9} {'θ·n中位':>9} {'年失效真值':>11} {'年失效模型':>11} {'取整损失':>9}")
    print(hdr)
    for (m, eta, N, A, bw, tag) in (
            (2.5, 120.0, 38000, 120, 1, "← 当前配置(摊平)"),
            (2.5, 120.0, 38000, 120, 6, ""),
            (2.5, 148.0, 38000, 300, 1, ""),
            (2.5, 148.0, 38000, 300, 6, "← 大机队+分箱，可接受"),
            (2.5, 148.0, 2000, 300, 12, ""),
            (2.5, 148.0, 2000, 300, 6, ""),
            (2.5, 148.0, 1000, 300, 12, ""),
            (2.5, 148.0, 1000, 300, 6, "← 老师口径的机队，全被抹掉"),
            (2.5, 148.0, 546, 300, 12, "← λ=50台/年"),
            (2.5, 148.0, 546, 300, 24, "")):
        d = audit(m, eta, A, N, 5, bw, W, 60)
        print(f"   {m:>4.1f} {eta:>5.0f} {N:>6,} {A:>5} {bw:>5} {d['箱数']:>5} "
              f"{d['每箱台数(中位)']:>9.1f} {d['θ·n 中位']:>9.3f} "
              f"{d['年失效']:>11,.0f} {d['年失效(模型)']:>11,.0f} "
              f"{d['取整损失']:>9.1%}   {tag}")

    print("\n" + "=" * 78)
    print("④ 推荐参数点（m=2.5 分支）")
    print("=" * 78)
    for N, A, bw in ((2000, 300, 6), (2000, 300, 12), (3000, 300, 6)):
        d = audit(2.5, 148.0, A, N, 5, bw, W, 60)
        print(f"   N={N:,}  A={A}(箱宽{bw}月,{d['箱数']}箱)  "
              f"年失效 {d['年失效']:,.0f}（模型 {d['年失效(模型)']:,.0f}，"
              f"损失 {d['取整损失']:.0%}）  质保占比 {d['质保占比']:.1%}  "
              f"P(过A)={d['P(过A)']:.3f}  每箱 {d['每箱台数(中位)']:.1f} 台")


if __name__ == "__main__":
    main()
