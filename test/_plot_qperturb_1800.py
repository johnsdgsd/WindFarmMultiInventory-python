# _plot_qperturb_1800.py — qperturb 单解 vs 非分块/分块的历史下界曲线。
#
# 只读取结果目录中的 sddp_training_log.csv；图与检查点 CSV 只新增文件。
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
EARLY_NOBLOCK = ROOT / "results" / "asyncV2_20261109_215549"
LATE_NOBLOCK = ROOT / "results" / "v2_noblock_4h_20261110_223050"
BLOCK = ROOT / "results" / "v2_static10_4h_20261110_223050"
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
    specs = [
        ("qperturb 单解（本次）", QP, "#d62728"),
        ("最早异步非分块", EARLY_NOBLOCK, "#1f77b4"),
        ("后来非分块 4h", LATE_NOBLOCK, "#9467bd"),
        ("分块 static-10 4h", BLOCK, "#2ca02c"),
    ]
    data = []
    for name, d, color in specs:
        rows = parse_log(d / "sddp_training_log.csv")
        if len(rows) == 0:
            raise SystemExit(f"没解析到 {d}")
        rows = rows[rows[:, 2] <= 1800.0 + 1e-9]
        data.append((name, rows, color))
        print(f"{name}: {len(rows)} 点，末轮 {rows[-1,0]:.0f}, "
              f"t={rows[-1,2]:.1f}s, LB={rows[-1,1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax1, ax2 = axes
    for name, rows, color in data:
        label = f"{name}（{rows[-1,1]:,.1f}）"
        ax1.plot(rows[:, 2], rows[:, 1], color=color, lw=1.7, label=label)
        ax2.plot(rows[:, 2], rows[:, 1], color=color, lw=1.7, label=label)

    ax1.set_xlim(0, 1820)
    ax1.set_ylim(0, 32500)
    ax2.set_xlim(600, 1820)
    zoom = [r[r[:, 2] >= 600.0] for _, r, _ in data]
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
    fig.suptitle("qperturb 单解 vs 非分块/分块：T60 下界收敛（≤1800 s）",
                 fontsize=13)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    out_dir = QP
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "lb_compare_qperturb_1800_v2.png"
    svg = out_dir / "lb_compare_qperturb_1800_v2.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    if OUT_EXT.is_dir():
        # 复制到外部文档目录；若已有旧图，加后缀避免覆盖。
        for ext, src in ((".png", png), (".svg", svg)):
            dst = OUT_EXT / f"lb_compare_qperturb_1800{ext}"
            if dst.exists():
                dst = OUT_EXT / f"lb_compare_qperturb_1800_v2{ext}"
            dst.write_bytes(src.read_bytes())

    checkpoints = [600, 900, 1200, 1500, 1800]
    rows_out = []
    for cp in checkpoints:
        rec = {"checkpoint_s": cp}
        for key, rows, _ in data:
            v = lb_at(rows, cp)
            if v is not None:
                rec[f"{key}_lb"] = f"{v[1]:.4f}"
                rec[f"{key}_iter"] = f"{v[0]:.0f}"
                rec[f"{key}_time"] = f"{v[2]:.2f}"
        rows_out.append(rec)
    csv_path = out_dir / "lb_compare_qperturb_1800_checkpoints_v2.csv"
    keys = ["checkpoint_s"]
    for name, _, _ in data:
        keys += [f"{name}_lb", f"{name}_iter", f"{name}_time"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("\n检查点：")
    for r in rows_out:
        print(r)
    print(f"\nPNG: {png}\nSVG: {svg}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
