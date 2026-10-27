# 画 151 轮 + 续训 231 轮的收敛曲线（累计训练时间）。
#   python test/_plot_convergence.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "base_T60_20261026_204105"
OUT = D
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"   # SVG 里文字转路径，换机器不会掉字


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) == 6:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    return np.array(rows)


a1 = load(D / "sddp_training_log_prev_20261027_112414.csv")   # 1..151
a2 = load(D / "sddp_training_log.csv")                        # 152..231
off = float(a1[-1, 3])                                        # 3713.337 s
t = np.r_[a1[:, 3], a2[:, 3] + off] / 60.0                    # minutes
lb = np.r_[a1[:, 2], a2[:, 2]]
sim = np.r_[a1[:, 1], a2[:, 1]]
it = np.r_[a1[:, 0], a2[:, 0]]

w = 9
med = np.array([np.median(sim[max(0, i - w // 2):i + w // 2 + 1])
                for i in range(len(sim))])

UB151, LB151, GAP151 = 36370.8685, 31282.3236, 13.9907
UB231, LB231, GAP231 = 35833.3228, 31573.2123, 11.8887
cut = off / 60.0

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(11, 10.5),
    gridspec_kw={"height_ratios": [1.05, 0.85, 0.9]})

# ---- (a) 全程 LB ----
ax1.plot(t, lb, color="#1f77b4", lw=1.9, label="第一阶段（1–151 轮）")
ax1.plot(t[t >= cut], lb[t >= cut], color="#d62728", lw=2.3,
         label="续训（152–231 轮）")
ax1.axvline(cut, color="#555555", ls="--", lw=1.1)
ax1.scatter([cut, t[-1]], [LB151, LB231], color=["#1f77b4", "#d62728"],
            zorder=5, s=45)
ax1.text(0.55, 0.16,
         "151 轮：LB 31,282，gap 13.99%\n"
         "231 轮：LB 31,573，gap 11.89%\n"
         "第二小时：LB +290.9，UB(同场景) −537.6",
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.95))
ax1.set_ylabel("下界 LB（万元）")
ax1.set_title("T60 收敛曲线：R00 参数，151 轮 + 续训 80 轮（累计训练时间）",
              fontsize=13, pad=10)
ax1.grid(alpha=0.25)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_ylim(0, 33500)

# ---- (b) 尾部放大 ----
ax2.plot(t, lb, color="#1f77b4", lw=1.8)
ax2.plot(t[t >= cut], lb[t >= cut], color="#d62728", lw=2.2)
ax2.axvline(cut, color="#555555", ls="--", lw=1.1)
ax2.scatter([cut, t[-1]], [LB151, LB231], color=["#1f77b4", "#d62728"],
            zorder=5, s=40)
ax2.annotate(f"151 轮  {LB151:,.1f}", xy=(cut, LB151),
             xytext=(33, 31760), fontsize=10, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
ax2.annotate(f"231 轮  {LB231:,.1f}  (+290.9)", xy=(t[-1], LB231),
             xytext=(73, 31720), fontsize=10, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
ax2.set_xlim(28, t[-1] + 2)
ax2.set_ylim(30400, 31950)
ax2.set_ylabel("LB（尾部放大）")
ax2.grid(alpha=0.25)

# ---- (c) 策略端：前向均值 + 最终 UB ----
ax3.plot(t, sim, color="#bbbbbb", lw=0.8, alpha=0.8,
         label="前向均值（10 路径/轮，噪声大）")
ax3.plot(t, med, color="#2ca02c", lw=1.8, label="前向均值滚动中位数（9 轮）")
ax3.axhline(UB151, color="#1f77b4", ls=":", lw=1.5)
ax3.axhline(UB231, color="#d62728", ls=":", lw=1.5)
ax3.text(t[-1] - 2, UB151 + 900, f"151 轮最终 UB {UB151:,.0f}", fontsize=9,
         color="#1f77b4", ha="right",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax3.text(t[-1] - 2, UB231 - 2600, f"231 轮最终 UB {UB231:,.0f}", fontsize=9,
         color="#d62728", ha="right",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax3.axvline(cut, color="#555555", ls="--", lw=1.1)
ax3.set_xlabel("累计训练时间（分钟）")
ax3.set_ylabel("总成本（万元/路径）")
ax3.set_title("策略端：前向均值（首轮 >60,000，已截断）与两种割集的最终 3000 路径 UB",
              fontsize=11)
ax3.grid(alpha=0.25)
ax3.legend(loc="upper right", fontsize=9)
ax3.set_xlim(0, t[-1] + 2)
ax3.set_ylim(28000, 60000)

fig.tight_layout()
png = OUT / "convergence_151_231.png"
svg = OUT / "convergence_151_231.svg"
csv = OUT / "convergence_151_231.csv"
fig.savefig(png, dpi=170, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
np.savetxt(csv, np.c_[it, t, lb, sim],
           delimiter=",", fmt=["%d", "%.4f", "%.4f", "%.4f"],
           header="iteration,cum_time_min,LB,forward_mean_10paths",
           comments="")
print("saved", png)
print("saved", svg)
print("saved", csv)
print(f"rows {len(t)}  time {t[0]:.2f}–{t[-1]:.2f} min  "
      f"LB {lb[0]:.0f}–{lb[-1]:.0f}  "
      f"LB@cut {lb[np.argmin(np.abs(t-cut))]:.1f}")
