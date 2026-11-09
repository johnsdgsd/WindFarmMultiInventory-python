# _plot_lb_tier1_tier2.py — 画 V2 基线 / Tier1 / Tier2 的下界收敛图（横轴为训练墙钟）。
#
# 只读结果目录里的 sddp_training_log*.csv，不修改 results/。
# 基线目录有续训归档日志：prev 段 + 当前段；当前段的 time 列从 0 重新计时，
# 因此按段拼接并累加上一段的末尾时间。
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(errors="replace")

BASE = ROOT / "results" / "base_T60_20261026_204105"
T1 = ROOT / "results" / "int_t1_formal_20261106_132002"
T2 = ROOT / "results" / "int_t2_formal_20261106_180256"
OUT = (ROOT.parent / "风场多仓库备件库存管理案例-python-创新文档" / "figures")


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def parse_log(path: Path):
    """返回 [(iter, ub, lb, time_s), ...]，按日志表头定位列。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        head = line.split()
        if len(head) < 4 or head[0] != "iteration" or "bound" not in head:
            continue
        i_ub = head.index("simulation")
        i_lb = head.index("bound")
        i_t = head.index("time") if "time" in head else len(head)
        if max(i_ub, i_lb) >= i_t:
            raise SystemExit(f"{path}: 表头列序无法定位 time: {line!r}")
        rows = []
        for row in lines[i + 1:]:
            try:
                vals = [float(c) for c in row.split()]
            except ValueError:
                if rows:
                    break
                continue
            if len(vals) < 4:
                if rows:
                    break
                continue
            rows.append((vals[0], vals[i_ub], vals[i_lb], vals[i_t]))
        return rows
    return []


def load_dir(d: Path):
    """拼接目录内 prev 段 + 当前段；time 列在续训段从 0 重新计时。"""
    segs = sorted(d.glob("sddp_training_log_prev_*.csv"),
                  key=lambda p: p.stat().st_mtime)
    segs.append(d / "sddp_training_log.csv")
    out = []
    last_t = 0.0
    for p in segs:
        if not p.is_file():
            continue
        rows = parse_log(p)
        if not rows:
            continue
        offset = last_t if rows[0][3] < last_t - 1e-9 else 0.0
        for it, ub, lb, t in rows:
            out.append((it, ub, lb, t + offset))
        if out:
            last_t = out[-1][3]
    out.sort(key=lambda r: (r[3], r[0]))
    # 段边界偶有同轮重复；保留最后一版
    ded = []
    for r in out:
        if ded and abs(ded[-1][0] - r[0]) < 1e-12:
            ded[-1] = r
        else:
            ded.append(r)
    return np.array(ded, dtype=float)


def lb_at(rows, target):
    """墙钟 target 及之前最后一轮的 LB/轮次；没有则返回 None。"""
    m = rows[rows[:, 3] <= target + 1e-12]
    if len(m) == 0:
        return None
    r = m[-1]
    return float(r[2]), float(r[0]), float(r[3])


def main():
    setup_font()
    if not OUT.is_dir():
        raise SystemExit(f"输出目录不存在: {OUT}")

    base = load_dir(BASE)
    t1 = load_dir(T1)
    t2 = load_dir(T2)
    data = [
        ("V2 基线", base, "#1f77b4"),
        ("Tier1 整数物流", t1, "#ff7f0e"),
        ("Tier2 全整数", t2, "#2ca02c"),
    ]
    for name, rows, _ in data:
        if len(rows) == 0:
            raise SystemExit(f"{name}: 没解析到数据")
        print(f"{name}: {len(rows)} 点，末轮 {rows[-1,0]:.0f}，"
              f"t={rows[-1,3]:.1f}s，LB={rows[-1,2]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax1, ax2 = axes
    for name, rows, color in data:
        label = f"{name}（{rows[-1,0]:.0f} 轮，{rows[-1,2]:,.2f}）"
        ax1.plot(rows[:, 3], rows[:, 2], color=color, lw=1.8, label=label)
        ax2.plot(rows[:, 3], rows[:, 2], color=color, lw=1.8, label=label)
        ax1.plot(rows[-1, 3], rows[-1, 2], "o", color=color, ms=5)
        ax2.plot(rows[-1, 3], rows[-1, 2], "o", color=color, ms=5)

    ax1.set_xlim(0, 7800)
    ax1.set_ylim(0, 33000)
    ax2.set_xlim(1000, 7800)
    ax2.set_ylim(23500, 32000)
    for ax, title in ((ax1, "(a) 全程"), (ax2, "(b) 1,000 s 后放大")):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("训练墙钟时间 (s)")
        ax.set_ylabel("下界 LB")
        ax.grid(alpha=0.3, lw=0.5)
    fig.suptitle("T60 下界收敛：V2 基线 vs Tier1 整数物流 vs Tier2 全整数",
                 fontsize=13)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    png = OUT / "lb_convergence_T60_tier1_tier2.png"
    svg = OUT / "lb_convergence_T60_tier1_tier2.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    # 墙钟检查点：每个版本取该时刻之前最后一轮（与验收报告口径一致）
    checkpoints = [600, 1800, 3600, 5400, 7200]
    rows_out = []
    for cp in checkpoints:
        rec = {"checkpoint_s": cp}
        for key, arr in (("baseline", base), ("tier1", t1), ("tier2", t2)):
            v = lb_at(arr, cp)
            if v is not None:
                rec[f"{key}_lb"] = f"{v[0]:.4f}"
                rec[f"{key}_iter"] = f"{v[1]:.0f}"
                rec[f"{key}_time"] = f"{v[2]:.2f}"
        rows_out.append(rec)
    # 最终值
    rec = {"checkpoint_s": "final"}
    for key, arr in (("baseline", base), ("tier1", t1), ("tier2", t2)):
        rec[f"{key}_lb"] = f"{arr[-1,2]:.4f}"
        rec[f"{key}_iter"] = f"{arr[-1,0]:.0f}"
        rec[f"{key}_time"] = f"{arr[-1,3]:.2f}"
    rows_out.append(rec)

    csv_path = OUT / "lb_convergence_T60_tier1_tier2_checkpoints.csv"
    keys = ["checkpoint_s", "baseline_lb", "baseline_iter", "baseline_time",
            "tier1_lb", "tier1_iter", "tier1_time",
            "tier2_lb", "tier2_iter", "tier2_time"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("\n墙钟检查点（该时刻之前最后一轮）：")
    for r in rows_out:
        print(r)
    print(f"\nPNG: {png}\nSVG: {svg}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
