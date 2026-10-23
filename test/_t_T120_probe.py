# test/_t_T120_probe.py — 探讨性推演：质保期由 5 年拉到 10 年会发生什么。
#
#   python test/_t_T120_probe.py
#
# **只在内存里构造 T=120 / A=140 的问题，不读写任何配置文件、不碰 results/**。
# 用同一套机队/退化参数，把月龄队列推 120 期（全不修上界 + 补货版），看：
#   ① 累计失效率怎么变（非线性？）
#   ② 峰值月失效 vs ΣQmax=53 的缺口
#   ③ 缺口累计成多少停机存量
#   ④ 到寿档、状态维、rint 蜜月期这些结构性约束会不会被顶破
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402

cfg0 = load_problem_config(ROOT / "config" / "problem.toml")

print("=" * 78)
print("一、退化端：把视界从 60 拉到 120，失效量怎么变（解析式，与 T 无关）")
print("=" * 78)
N_tot = int(np.sum(cfg0.N))
m, eta = cfg0.m, cfg0.eta
for T in (60, 120):
    cum = 1.0 - np.exp(-(T / eta) ** m)
    print(f"  T = {T:>3} 月（{T/12:.0f} 年）：累计失效率 {cum:>7.3%}   累计失效 "
          f"{N_tot*cum:>7,.0f} 台   年均 {N_tot*cum/(T/12):>7,.0f} 台/年")
print()
print("  逐年增量（百分点）：")
prev = 0.0
for y in range(1, 11):
    t = 12 * y
    cum = 1.0 - np.exp(-(t / eta) ** m)
    print(f"    第 {y:>2} 年  累计 {cum:>7.3%}   本年新增 {(cum-prev):>6.3%} "
          f"= {N_tot*(cum-prev):>6,.0f} 台")
    prev = cum

print()
print("=" * 78)
print("二、产能端：月度失效 F(t) 与订货上限 ΣQmax 的缺口")
print("=" * 78)
S_Qmax = 53
print(f"  当前 ΣQmax = {S_Qmax}（本组参数）")
print()
print("   月龄 t    解析 F(t)    F−ΣQmax")
first = None
for t in (40, 50, 55, 60, 70, 80, 90, 100, 110, 120):
    h0 = (m / eta) * (t / eta) ** (m - 1.0)
    F = N_tot * h0
    if F > S_Qmax and first is None:
        # 反解 F(t)=ΣQmax 的时点
        tt = eta * (S_Qmax / (N_tot * m / eta)) ** (1.0 / (m - 1.0))
        first = tt
    print(f"   {t:>6}   {F:>9.1f}   {F-S_Qmax:>+9.1f}"
          f"{'   ← 首次超顶' if t == 60 and first else ''}")
print(f"\n  ⇒ F 首次超过 ΣQmax 的时点 ≈ 第 {first:.0f} 期（改造后现参数是第 52 期）")

# 累计缺口（把 F 当连续量做梯形积分，t=50..120）
ts = np.arange(50, 121, dtype=float)
Ft = N_tot * (m / eta) * (ts / eta) ** (m - 1.0)
print(f"  ⇒ 第 50~120 期累计缺口 ∫(F−ΣQmax)⁺ ≈ {np.sum(np.maximum(Ft-S_Qmax,0)):,.0f} 台")
print(f"     占机队 {np.sum(np.maximum(Ft-S_Qmax,0))/N_tot:.1%}")

print()
print("  要让缺口比值回到 1.25，ΣQmax 需要提到 F峰值/1.25 ：")
for T in (60, 120):
    h0 = (m / eta) * (T / eta) ** (m - 1.0)
    print(f"    T = {T:>3}：F({T}) = {N_tot*h0:>6.1f}  ⇒  ΣQmax ≈ {N_tot*h0/1.25:>6.1f}"
          f"   （现用 {S_Qmax}，需 ×{N_tot*h0/1.25/S_Qmax:.1f}）")

print()
print("=" * 78)
print("三、结构端：T=120 会顶破哪些现有设定")
print("=" * 78)
A_now = cfg0.A
dim = lambda A: cfg0.N_R + (cfg0.L - 1) * cfg0.N_R + cfg0.N_R * A + cfg0.N_R
print(f"  A 现状 = {A_now}；硬约束 A ≥ T+1 = 121 ⇒ 必须抬到 ≥ 121。"
      f"取 A=140 时状态维 {dim(A_now)} → {dim(140)}（×{dim(140)/dim(A_now):.2f}）")
print()
print("  到寿档：spread_init = 0 ⇒ 期初全在月龄 0 ⇒ 全体在 t = A−1 同时到寿。")
print(f"    A = {A_now}  ：全体在第 {A_now} 期同时到寿（T=120 时仍在视界内）✗ 悬崖重现")
print(f"    A = 140      ：到寿点在第 140 期，出视界 ✓（这正是改造前 T=200/A=120 踩过的坑）")
print()
print("  但 A=140 只解决\"撞档\"，不解决\"同步老化\"：机队仍是同生同死，")
print("  峰谷形态会更极端（失效在视界末段集中到 F 峰值）。")
print()
print("  rint 与摊平：若靠 spread_init = K 打散初始队列，每桶台数 ÷K，蜜月期拉长。")
for K in (0, 1, 5, 10, 20):
    per_bucket = N_tot / (m > 0) / cfg0.N_R / max(K, 1)
    # 需要 h0(a)·n ≥ 0.5
    need = 0.5 / per_bucket
    aa = eta * (need / (m / eta)) ** (1.0 / (m - 1.0)) if need > 0 else 0.0
    print(f"    spread_init = {K:>2}：每桶 {per_bucket:>6,.0f} 台 → 首台失效在第 {aa:>5.1f} 期"
          f"（占 120 期视界 {aa/120:>5.1%}）")

print()
print("=" * 78)
print("四、补货版推演：缺口累计成多少停机（ΣQmax 保持 53）")
print("=" * 78)
cfg = replace(cfg0)
cfg.T, cfg.A = 120, 140
prob = DefineProblem(cfg)
NR, A, T = prob.N_R, prob.A, prob.T
Qmax = np.asarray(prob.Qmax, float)
n = np.zeros((NR, A))
n[:, 0] = prob.N
theta_bar = np.mean([th for _, th in prob.pool], axis=0)      # (NR, A) —— A=140 全覆盖
D = np.zeros(NR)                                              # 各仓待补台数
print(f"  问题规模：T={T} A={A} 状态维={prob.layout.dim}"
      f"（无提前期、装机 = min(待补, Qmax) 的上界推演）")
print("     期       F      装机     停机D")
for t in range(1, T + 1):
    fail = np.rint(theta_bar * n)                             # (NR, A)
    F = fail.sum(axis=1)
    demand = D + F                                            # 只能给坏了的机器换
    q = np.minimum(Qmax, demand)
    D = demand - q
    nxt = np.zeros_like(n)
    nxt[:, 1:] = np.maximum(n[:, :-1] - fail[:, :-1], 0.0)     # 存活者月龄 +1
    nxt[:, 0] = q
    n = nxt
    if t % 10 == 0 or t == 60:
        print(f"   {t:>5} {F.sum():>8.1f} {q.sum():>8.1f} {D.sum():>9.1f}"
              f"{'   ← 60 期（= 现在的视界末）' if t == 60 else ''}")
print(f"\n  ⇒ 第 120 期停机存量 {D.sum():,.0f} 台 = 机队的 {D.sum()/sum(prob.N):.1%}"
      f"（对照：改造后 60 期视界的峰值仅 53.6 台 = 0.33%）")
print(f"  ⇒ 第 60 期停机 {D.sum():,.0f} 台（同上：改造后 60 期实测峰值 53.6 台）")
