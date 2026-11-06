# _plot_lb_compare.py — T60 各路线当前下界 vs 训练墙钟，统一一张图。
#   python test/_plot_lb_compare.py
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(r"D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python-创新文档\figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def rows(path: Path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if s and s[0].isdigit():
            out.append((int(s[0]), float(s[3]), float(s[2])))  # iter, t, lb
    return out


def combine(prev: Path, main: Path, skip_until: int | None = None):
    """prev 段 + main 段；main 的 time 从 0 重新计时，需加 offset。
    skip_until 用于 A L8 续训重做 214-229 时只保留 iter>229。"""
    r1, r2 = rows(prev), rows(main)
    if not r1 or not r2:
        return r1 or r2
    off = r1[-1][1]
    return r1 + [(i, t + off, lb) for i, t, lb in r2
                 if skip_until is None or i > skip_until]


def main():
    specs = [
        ("V2 标准单割", ROOT / "results/base_T60_20261026_204105",
         "sddp_training_log_prev_20261027_112414.csv", "sddp_training_log.csv", 151,
         "#1f77b4"),
        ("自适应场景分区", ROOT / "results/partT60_20261102_235747",
         None, "sddp_training_log.csv", None, "#2ca02c"),
        ("固定簇 A L8", ROOT / "results/fcmcT60_full_L8_20261103_165647",
         "sddp_training_log_prev_20261103_183513.csv", "sddp_training_log.csv", 229,
         "#ff7f0e"),
        ("固定簇 A L16", ROOT / "results/fcmcT60_full_L16_20261104_102936",
         None, "sddp_training_log.csv", None, "#d62728"),
        ("固定簇+细化 B n=10", ROOT / "results/refineB_t60_full_n10_20261103_215301",
         None, "sddp_training_log.csv", None, "#9467bd"),
    ]
    curves = []
    for name, d, prev, main_name, skip, color in specs:
        if prev:
            data = combine(d / prev, d / main_name, skip)
        else:
            data = rows(d / main_name)
        if not data:
            print("missing", name, d)
            continue
        arr = np.asarray([(t, lb) for _, t, lb in data], float)
        curves.append((name, arr, color))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=160,
                                  gridspec_kw={"width_ratios": [1.05, 1.0]})
    for name, arr, color in curves:
        label = f"{name}（最终 {arr[-1, 1]:,.0f}）"
        ax.plot(arr[:, 0], arr[:, 1], lw=1.6, color=color, label=label)
        ax2.plot(arr[:, 0], arr[:, 1], lw=1.8, color=color, label=label)
        ax.scatter([arr[-1, 0]], [arr[-1, 1]], s=20, color=color, zorder=5)
        ax2.scatter([arr[-1, 0]], [arr[-1, 1]], s=22, color=color, zorder=5)
    for a in (ax, ax2):
        a.axvline(7503.3, color="0.55", ls="--", lw=0.9)
        a.grid(alpha=0.25)
    ax.set_xlabel("训练墙钟时间 (s)")
    ax.set_ylabel("当前下界 LB")
    ax.set_title("(a) 全程")
    ax2.set_xlim(500, 7700)
    ax2.set_ylim(28000, 31800)
    ax2.set_xlabel("训练墙钟时间 (s)")
    ax2.set_ylabel("当前下界 LB")
    ax2.set_title("(b) 500 s 以后放大")
    ax2.legend(loc="lower right", fontsize=8.3, framealpha=0.92)
    fig.suptitle("T60 下界收敛（当前 cut pool；训练墙钟）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = OUT / "lb_convergence_T60.png"
    svg = OUT / "lb_convergence_T60.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    print("saved", png)
    print("saved", svg)

    # 同时间点表：600/1800/3600/5400/7503
    marks = [600, 1800, 3600, 5400, 7503.3]
    csv_path = OUT / "lb_convergence_T60_checkpoints.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + [f"LB@{m:g}s" for m in marks] + ["final_LB", "final_t"])
        for name, arr, _ in curves:
            vals = []
            for m in marks:
                idx = np.searchsorted(arr[:, 0], m, side="right") - 1
                vals.append("" if idx < 0 else f"{arr[idx, 1]:.2f}")
            w.writerow([name] + vals + [f"{arr[-1, 1]:.2f}", f"{arr[-1, 0]:.1f}"])
    print("saved", csv_path)


if __name__ == "__main__":
    main()
