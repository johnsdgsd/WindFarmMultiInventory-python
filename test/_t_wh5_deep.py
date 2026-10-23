# _t_wh5_deep.py — 仓 5 反常下钻：净调出最多却停机最多，钱和时间都花在哪，
# 以及「调出的是不是当期剩下的、接盘方是不是更缺」。
# 只读 sim npz。
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
S_end = z["S_end"].astype(float)
terms = z["terms"]
scene = z["scene"].astype(int)
I_in = x_in[:, :, :Ly.n_I]
n_in = x_in[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A)
F = np.zeros((P, T, NR))
for t in range(T):
    th = np.array([prob.pool[scene[p, t]][1] for p in range(P)])
    F[:, t, :] = np.rint(th * n_in[:, t]).sum(axis=2)

served = y.sum(axis=2)                    # (P,T,NR) 本仓需求被满足
Dn = prob.N[None, None, :] - n_in.sum(axis=3)
tau_n = taup - taum
unmet = np.maximum(F - served, 0.0)


def head(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


m5 = 4
head("① 仓 5 的调出是不是「当期被动满足后的剩余」？又去了哪里？")
print(f"  仓5 全期：τ⁻ {taum[:, :, m5].sum():,.0f}  τ⁺ {taup[:, :, m5].sum():,.0f}"
      f"  净 {tau_n[:, :, m5].sum():,.0f}")
print(f"  调出时的 S_end 均值 {S_end[:, :, m5][taum[:, :, m5] > TOL].mean():.2f}"
      f"（S_end = 被动满足后剩余 + 本期到货；Imax=36）")
print(f"  调出时的本仓未满足量均值 {unmet[:, :, m5][taum[:, :, m5] > TOL].mean():.2f}"
      f"（>0 说明当期本仓就有缺口）")
exp_cells = (taum[:, :, m5] > TOL)
print(f"  调出期数占比 {exp_cells.mean():.1%}；其中本仓同期有缺口（unmet>0）占 "
      f"{(unmet[:, :, m5][exp_cells] > TOL).mean():.1%}")
print(f"  「本仓下一期缺口」在调出期的均值 "
      f"{unmet[:, 1:, m5][exp_cells[:, :-1]].mean():.2f}，"
      f"非调出期 {unmet[:, 1:, m5][~exp_cells[:, :-1]].mean():.2f}")

print("\n  分仓净调拨（全期，件）与末期 20 期：")
print("     仓    净τ(全期)   净τ(49-60)   未满足(全期)  未满足(49-60)   I_in(49-60)")
for r in range(NR):
    print(f"   {r+1:>3} {tau_n[:, :, r].sum():>12,.0f} {tau_n[:, 48:, r].sum():>12,.0f}"
          f" {unmet[:, :, r].sum():>14,.0f} {unmet[:, 48:, r].sum():>14,.0f}"
          f" {I_in[:, 48:, r].mean():>13.2f}")

head("② 逐期画像（第 36..60 期，全路径均值）")
print("     t     仓1: F/满足/D/I/q/τ | 仓4（净入）        | 仓5（净出）")
for t in range(35, T):
    cells = []
    for r in (0, 3, 4):
        cells.append(f"{F[:, t, r].mean():>5.1f}/{served[:, t, r].mean():>5.1f}/"
                     f"{Dn[:, t, r].mean():>4.1f}/{I_in[:, t, r].mean():>5.1f}/"
                     f"{q[:, t, r].mean():>4.1f}/{tau_n[:, t, r].mean():>+5.2f}")
    print(f"   {t+1:>4}  " + "  |  ".join(cells))

head("③ 若仓 5 不调出，能补上多少停机？（上界）")
keep = np.minimum(taum[:, :, m5], np.roll(unmet[:, :, m5], -1, axis=1))
keep[:, -1] = 0.0
print(f"  min(当期调出量, 下期未满足量) 全期合计 {keep.sum():,.0f} 件·期"
      f"（= 停机台·期的 {keep.sum()/Dn[:, :, m5].sum():.1%}）")
print(f"  分时段：")
for a, b in [(0, 12), (12, 24), (24, 36), (36, 48), (48, 60)]:
    print(f"     {a+1:>3}-{b:<3} 可补 {keep[:, a:b].sum():>10,.0f}"
          f"   该段停机 {Dn[:, a:b, m5].sum():>10,.0f}"
          f"   覆盖率 {keep[:, a:b].sum()/max(Dn[:, a:b, m5].sum(),1):>6.1%}")

head("④ 相关性：仓 5 的净调出与什么同步？（逐 (路径,期) 样本）")
for r in range(NR):
    a = tau_n[:, :, r].ravel()
    pairs = [("期初库存", I_in[:, :, r].ravel()),
             ("当期失效F", F[:, :, r].ravel()),
             ("当期未满足", unmet[:, :, r].ravel()),
             ("下期未满足", np.roll(unmet[:, :, r], -1, axis=1).ravel()),
             ("Qmax−q余量", (prob.Qmax[r] - q[:, :, r]).ravel())]
    s = "  ".join(f"{nm} {np.corrcoef(a, v)[0,1]:+.2f}" for nm, v in pairs)
    print(f"   仓{r+1}: {s}")

head("⑤ 成本对冲：仓 5 的调拨成本 vs 它自己的停机成本")
c5_active = float(terms[:, :, 3].sum())     # 全部主动成本（分仓不可拆，用占比近似）
print(f"  仓5 主动调拨成本（按 (τ⁺+τ⁻) 份额）: "
      f"{c5_active * (taup[:,:,m5]+taum[:,:,m5]).sum() / (taup+taum).sum():,.0f}")
print(f"  仓5 停机成本: {Dn[:, :, m5].sum()*prob.Cs:,.0f}")
print(f"  仓5 采购成本: {prob.C * x_in[:, :, Ly.ix_q(m5)].sum():,.0f}")
