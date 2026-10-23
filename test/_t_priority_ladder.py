# _t_priority_ladder.py — 优先级阶梯的正面检验：抽干率 / 跨仓拉货时的可选项。
# 「跨仓量单调递减」是错的判据（受供给容量混淆），这里换成：
#   (a) 各源被抽干（用光）的比率 —— 越靠前应越高；
#   (b) 每次跨仓拉货时，更便宜的源里还剩几个有货。
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
P = int(load_json(D, f.stem + ".json")["n_paths"])
z = np.load(f)
x_in = z["x_in"].astype(float)
y = z["y"].astype(float)
I_in = x_in[:, :, :Ly.n_I]
tc = np.asarray(prob.tc_route, float)

used = y.sum(axis=2)                       # (P,T,NR) 各源被消耗
cross = y.copy()
for r in range(NR):
    cross[:, :, r, r] = 0.0                # 只看跨仓

print("(a) 各源被抽干（库存用光）的比率 —— 越靠前（别人越愿意用）应越高")
print("     仓   有货的(路径,期)数    抽干率    平均用掉比例   跨仓供出量")
for m in range(NR):
    has = I_in[:, :, m] > TOL
    drained = (used[:, :, m] >= I_in[:, :, m] - TOL) & has
    frac_used = np.where(has, used[:, :, m] / np.maximum(I_in[:, :, m], 1), np.nan)
    print(f"   {m+1:>3} {int(has.sum()):>16,} {drained.sum()/max(has.sum(),1):>10.1%}"
          f" {np.nanmean(frac_used):>13.1%} {cross.sum(axis=(0,1,2))[m]:>12,.0f}")

print("\n   注：跨仓供出量 = 被其他仓按优先级取走的量；仓 5 是所有人的末位，"
      "仓 1 是所有人的首位。")

print("\n(b) 每次跨仓拉货时，更便宜的源里还剩几个有货？（0 = 全部抽干，被迫取贵源）")
hist = np.zeros(5, dtype=np.int64)
tot = 0
for r in range(NR):
    order = np.argsort(tc[r], kind="stable")
    for k in range(1, NR):
        m = order[k]
        take = cross[:, :, r, m]
        mask = take > TOL
        n_avail = np.zeros(mask.shape, dtype=np.int64)
        for j in range(k):
            mj = order[j]
            n_avail += ((I_in[:, :, mj] - used[:, :, mj]) > TOL).astype(np.int64)
        for v in range(k + 1):
            hist[v] += int((mask & (n_avail == v)).sum())
        tot += int(mask.sum())
print("     更便宜源仍有货的个数: " +
      "  ".join(f"{i}个:{hist[i]/max(tot,1):>6.1%}" for i in range(5)))
print(f"     合计跨仓 (路径,期,仓,源) 次数 {tot:,}")

print("\n(c) 跨仓拉货的时机：是否都在「系统有人缺货」的期？")
served = y.sum(axis=2)
Dn = prob.N[None, None, :] - x_in[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A).sum(axis=3)
sys_unmet = (Dn + 0).sum(axis=2)           # 期初待维修合计（当期缺货压力）
cross_vol = cross.sum(axis=(2, 3))
tot_vol = cross_vol.sum()
for thr in (0.5, 1.5, 3.0):
    m = sys_unmet > thr
    print(f"     系统期初停机 > {thr:>3.1f} 台的期：占跨仓量 "
          f"{cross_vol[m].sum()/tot_vol:>6.1%}（占全部 (路径,期) 的 {m.mean():>5.1%}）")
