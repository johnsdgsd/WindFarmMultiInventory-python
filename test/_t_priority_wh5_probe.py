# _t_priority_wh5_probe.py — 诊断两个待查问题（只读 sim npz，不重跑仿真）：
#   ① 被动调拨 y 的优先级梯度为何在聚合上不单调；
#   ② 仓 5 为何「净调出最多、停机台·期却最高」。
#
# 用法: python test/_t_priority_wh5_probe.py [结果目录]
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402
from utils.io import load_json                         # noqa: E402

TOL = 1e-6
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "base_20261012_220300"
if not D.is_absolute():
    D = ROOT / D

prob = DefineProblem(load_problem_config(D / "config" / "problem.toml"))
Ly, NR, A, T = prob.layout, prob.N_R, prob.A, prob.T
f = sorted(D.glob("sim_*paths.npz"))[-1]
meta = load_json(D, f.stem + ".json")
P = int(meta["n_paths"])
z = np.load(f)
x_in = z["x_in"].astype(float)
y = z["y"].astype(float)
q = z["q"].astype(float)
taup, taum = z["taup"].astype(float), z["taum"].astype(float)
terms = z["terms"]
scene = z["scene"].astype(int)
I_in = x_in[:, :, :Ly.n_I]
n_in = x_in[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A)
F = np.zeros((P, T, NR))
for t in range(T):
    th = np.array([prob.pool[scene[p, t]][1] for p in range(P)])
    F[:, t, :] = np.rint(th * n_in[:, t]).sum(axis=2)

tc = np.asarray(prob.tc_route, float)
print(f"文件 {f.name}  路径 {P} × {T} 期")
print("tc_route（行=收货仓，列=供出仓）：")
for r in range(NR):
    print("  r" + str(r + 1), " ".join(f"{tc[r, m]:>4.0f}" for m in range(NR)))


def head(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ============================================================ A. 顺位聚合
head("A. 跨仓被动调拨按【优先级顺位】聚合（顺位 = 该源在本仓 tc_route 中的升序位置）")
print("      收货仓  本地(0)    顺位2     顺位3     顺位4     顺位5   |  跨仓合计")
for r in range(NR):
    order = np.argsort(tc[r], kind="stable")
    vals = [y[:, :, r, m].sum() for m in order]
    costs = [tc[r, m] for m in order]
    print(f"  r{r+1}   " + " ".join(f"{v:>10,.0f}" for v in vals) +
          f"   | {sum(vals[1:]):>10,.0f}")
    print("         " + " ".join(f"{'(' + format(c, '.0f') + ')' if c == 0 else format(c, '.0f'):>10}"
                                for c in costs))
print("\n  ⇒ 若优先级严格生效，每行应近似单调递减；中段反转 = 便宜源已被分光，不代表违反")

# ============================================================ B. 互补松弛：违例量与金额
head("B. 互补松弛检验：用了贵源时，便宜源是否真的没货了？")
used = y.sum(axis=2)                      # (P,T,NR) 各源被消耗的量
tot_passive_cost = float(terms[:, :, 4].sum())
viol_ev = viol_units = 0
excess_money = 0.0
excess_by_seg = np.zeros(5)
units_by_seg = np.zeros(5)
forced_units = 0.0                        # 所有便宜源都空、只能取贵源的量
for r in range(NR):
    order = np.argsort(tc[r], kind="stable")
    for k in range(1, NR):
        m = order[k]
        take = y[:, :, r, m]
        mask = take > TOL
        # 更便宜的源里，还剩库存（即"本可用而没用"）的最低成本
        best = np.full_like(take, np.inf)
        for j in range(k):
            mj = order[j]
            has = (I_in[:, :, mj] - used[:, :, mj]) > TOL
            best = np.where(has, np.minimum(best, tc[r, mj]), best)
        avail_cheap = np.isfinite(best)
        bad = mask & avail_cheap
        n_bad = int(bad.sum())
        u_bad = float(take[bad].sum())
        viol_ev += n_bad
        viol_units += u_bad
        if n_bad:
            excess_money += float((take[bad] * (tc[r, m] - best[bad])).sum())
            seg = np.minimum(np.arange(T) // (T // 5), 4)
            for s in range(5):
                sel = bad[:, seg == s]
                units_by_seg[s] += float(take[:, seg == s][sel].sum())
                excess_by_seg[s] += float((take[:, seg == s][sel] *
                                           (tc[r, m] - best[:, seg == s][sel])).sum())
        forced_units += float(take[mask & ~avail_cheap].sum())
tot_cross = float(y.sum()) - float(np.trace(y, axis1=2, axis2=3).sum())
print(f"  违例事件 {viol_ev:,} 次  违例量 {viol_units:,.0f} 件"
      f"（占跨仓量 {viol_units/max(tot_cross,1):.2%}）")
print(f"  「被迫取贵源」（便宜源全空）{forced_units:,.0f} 件"
      f"（占跨仓量 {forced_units/max(tot_cross,1):.2%}）")
print(f"  违例多付金额上界 {excess_money:,.0f}（占被动调拨总成本 {excess_money/tot_passive_cost:.3%}，"
      f"占总成本 {excess_money/float(terms.sum()):.4%}）")
print("  分段（1-12/13-24/25-36/37-48/49-60）：")
print("     违例件数 " + " ".join(f"{v:>12,.0f}" for v in units_by_seg))
print("     违例金额 " + " ".join(f"{v:>12,.0f}" for v in excess_by_seg))

# ============================================================ C. 精确运输 LP 对照
head("C. 精确运输问题对照：固定各仓满足量，重解最小成本分配（scipy/HiGHS）")
try:
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    cells = np.array([(p, t) for p in range(P) for t in range(T)])
    idx = rng.choice(len(cells), size=3000, replace=False)
    real_tot = opt_tot = 0.0
    ncells = 0
    c = tc.ravel()
    for i in idx:
        p, t = cells[i]
        served = y[p, t].sum(axis=1)
        if served.sum() <= TOL:
            continue
        A_eq = np.zeros((NR, NR * NR))
        for r in range(NR):
            A_eq[r, r * NR:(r + 1) * NR] = 1.0
        b_eq = served
        A_ub = np.zeros((NR, NR * NR))
        for m in range(NR):
            A_ub[m, m::NR] = 1.0
        b_ub = I_in[p, t]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method="highs")
        if not res.success:
            continue
        ncells += 1
        real_tot += float((y[p, t] * tc).sum())
        opt_tot += float(res.fun)
    print(f"  抽样 {ncells} 个 (路径,期) 点：实现成本 {real_tot:,.0f}  vs  "
          f"最优分配 {opt_tot:,.0f}  ⇒ 超额 {real_tot - opt_tot:,.0f}"
          f"（{(real_tot - opt_tot)/max(opt_tot, 1):.4%}）")
except ImportError:
    print("  scipy 不可用，跳过")

# ============================================================ D. 分仓画像
head("D. 分仓时间画像（每 12 期一档，全路径均值；F=失效, 满足=Σy, D=停机台数）")
served_r = y.sum(axis=2)
Dn = prob.N[None, None, :] - n_in.sum(axis=3)
supplied = y.sum(axis=2).sum(axis=0)  # unused
out_to = np.zeros((NR, NR))
for m in range(NR):
    for r in range(NR):
        out_to[r, m] = y[:, :, r, m].sum()
print("   仓    指标 " + " ".join(f"{a+1:>9}-{b:<9}" for a, b in
                                  [(0, 12), (12, 24), (24, 36), (36, 48), (48, 60)]))
segs = [(0, 12), (12, 24), (24, 36), (36, 48), (48, 60)]
for r in range(NR):
    rows = {
        "F   ": F[:, :, r].mean(axis=0),
        "满足": served_r[:, :, r].mean(axis=0),
        "D   ": Dn[:, :, r].mean(axis=0),
        "I_in": I_in[:, :, r].mean(axis=0),
        "q   ": q[:, :, r].mean(axis=0),
        "τ净 ": (taup[:, :, r] - taum[:, :, r]).mean(axis=0),
        "供出": y[:, :, :, r].sum(axis=2).mean(axis=0),
    }
    for k, v in rows.items():
        print(f"   {r+1 if k == 'F   ' else ' '} {k} " +
              " ".join(f"{v[a:b].mean():>18.1f}" for a, b in segs))

print("\n  被动调拨来源矩阵（行=收货仓, 列=供出仓；件，全期）")
print("       " + " ".join(f"   m{m+1:>8}" for m in range(NR)))
for r in range(NR):
    print(f"   r{r+1}  " + " ".join(f"{y[:, :, r, m].sum():>10,.0f}" for m in range(NR)))

print("\n  仓 5 的库存被谁取走（被动 y 合计）与主动 τ⁻ 调出：")
for m in [4]:
    for r in range(NR):
        if r != m:
            print(f"     r{r+1} ← m{m+1}: {y[:, :, r, m].sum():>10,.0f}")
    print(f"     主动调出 τ⁻: {taum[:, :, m].sum():>10,.0f}   主动调入 τ⁺: {taup[:, :, m].sum():>10,.0f}")

print("\n  仓 5 停机的时间分布（占本仓总停机台·期）：")
for a, b in segs:
    print(f"     {a+1:>3}-{b:<3} {Dn[:, a:b, 4].sum():>12,.0f}"
          f"  {Dn[:, a:b, 4].sum()/Dn[:, :, 4].sum():>7.1%}")

print("\n  取整对失效的影响（rint(θ·n) 逐桶归零率；仓 5 机队最小）")
for r in range(NR):
    tot_b = 0
    zero_b = 0
    for t in range(T):
        th = np.array([prob.pool[scene[p, t]][1] for p in range(P)])
        raw = th[:, r, :] * n_in[:, t, r, :]
        tot_b += raw.size
        zero_b += int((raw < 0.5).sum())
    print(f"     仓{r+1}: 桶数 {tot_b:,}  归零率 {zero_b/tot_b:.1%}   平均 F {F[:, :, r].mean():.2f}")
