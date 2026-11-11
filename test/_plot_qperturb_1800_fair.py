# _plot_qperturb_1800_fair.py — qperturb 单解 vs 2h 单独运行的非分块/分块基线。
#
# 4h 的 static/no-block 是当时两个训练同时启动、各 10 worker，存在 CPU 争用，
# 只画成虚线作参考，不应当和单独运行的曲线按同一墙钟直接比较。
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

QP = ROOT / "results" / "v2_qperturb_1800_repA_20261111_133802"
NOBLOCK2 = ROOT / "results" / "asyncV2_20261109_215549"
BLOCK2 = ROOT / "results" / "v2_stageblock_b10_2h_20261110_141902"
OUT_EXT = ROOT.parent / "风场多仓库备件库存管理案例-python-创新文档" / "figures"


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def parse_log(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        head = line.split()
        if len(head) < 4 or head[0] != "iteration" or "bound" not in head:
            continue
        i_lb = head.index("bound")
        i_t = head.index("time") if "time" in head else None
        if i_t is None:
            raise SystemExit(f"{path}: 表头没有 time 列")
        rows = []
        for line2 in lines[i + 1:]:
            try:
                vals = [float(c) for c in line2.split()]
            except ValueError:
                if rows:
                    break
                continue
            if len(vals) <= max(i_lb, i_t):
                if rows:
                    break
                continue
            rows.append((vals[0], vals[i_lb], vals[i_t]))
        if rows:
            return np.asarray(rows, dtype=float)
    return np.empty((0, 3), dtype=float)


def lb_at(rows, target):
    m = rows[rows[:, 2] <= target + 1e-9]
    return None if len(m) == 0 else m[-1]


def main():
    setup_font()
    primary = [
        ("qperturb 单解", QP, "#d62728", "-"),
        ("非分块 2h 基线", NOBLOCK2, "#1f77b4", "-"),
        ("分块 B=10 2h 基线", BLOCK2, "#2ca02c", "-"),
    ]
    data = []
    for name, d, color, ls in primary:
        rows = parse_log(d / "sddp_training_log.csv")
        if len(rows) == 0:
            raise SystemExit(f"没解析到 {d}")
        rows = rows[rows[:, 2] <= 1800.0 + 1e-9]
        data.append((name, rows, color, ls, False))
        print(f"{name}: 末轮 {rows[-1,0]:.0f}, "
              f"t={rows[-1,2]:.1f}s, LB={rows[-1,1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax1, ax2 = axes
    for name, rows, color, ls, is4 in data:
        label = f"{name}（{rows[-1,1]:,.1f}）"
        style = dict(color=color, lw=1.7 if not is4 else 1.3, ls=ls,
                     alpha=1.0 if not is4 else 0.8)
        ax1.plot(rows[:, 2], rows[:, 1], label=label, **style)
        ax2.plot(rows[:, 2], rows[:, 1], **style)

    ax1.set_xlim(0, 1820)
    ax1.set_ylim(0, 32500)
    ax2.set_xlim(600, 1820)
    zoom = [r[r[:, 2] >= 600.0] for _, r, _, _, _ in data]
    zoom = [r for r in zoom if len(r)]
    ymin = min(float(r[:, 1].min()) for r in zoom)
    ymax = max(float(r[:, 1].max()) for r in zoom)
    ax2.set_ylim(ymin - 40, ymax + 60)
    for ax, title in ((ax1, "(a) 全程"), (ax2, "(b) 600 s 后放大")):
        ax.axvline(600, color="0.55", ls=":", lw=1.0)
        ax.axvline(1800, color="0.55", ls="--", lw=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("训练墙钟时间 (s)")
        ax.set_ylabel("下界 LB")
        ax.grid(alpha=0.3, lw=0.5)
    fig.suptitle("qperturb 单解 vs 2h 单独运行的非分块/分块基线"
                 "：T60 下界收敛（≤1800 s）", fontsize=13)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    out_dir = QP
    png = out_dir / "lb_compare_qperturb_1800_fair.png"
    svg = out_dir / "lb_compare_qperturb_1800_fair.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    if OUT_EXT.is_dir():
        for ext, src in ((".png", png), (".svg", svg)):
            dst = OUT_EXT / f"lb_compare_qperturb_1800_fair{ext}"
            if dst.exists():
                dst = OUT_EXT / f"lb_compare_qperturb_1800_fair_v2{ext}"
            dst.write_bytes(src.read_bytes())

    checkpoints = [600, 900, 1200, 1500, 1800]
    rows_out = []
    for cp in checkpoints:
        rec = {"checkpoint_s": cp}
        for name, rows, _, _, _ in data:
            v = lb_at(rows, cp)
            if v is not None:
                rec[f"{name}_lb"] = f"{v[1]:.4f}"
                rec[f"{name}_iter"] = f"{v[0]:.0f}"
                rec[f"{name}_time"] = f"{v[2]:.2f}"
        rows_out.append(rec)
    csv_path = out_dir / "lb_compare_qperturb_1800_fair_checkpoints.csv"
    keys = ["checkpoint_s"]
    for name, _, _, _, _ in data:
        keys += [f"{name}_lb", f"{name}_iter", f"{name}_time"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nPNG: {png}\nSVG: {svg}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
