# _plot_lb_compare_b10.py — static-10 与无分块的 LB 收敛对比。
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np              # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B10 = ROOT / "results" / "v2_stageblock_b10_2h_20261110_141902"
NO = ROOT / "results" / "asyncV2_20261109_215549"


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def read_log(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        h = line.split()
        if h[:1] == ["iteration"] and "bound" in h:
            ilb, it = h.index("bound"), h.index("time")
            break
    rows = []
    for line in lines[i + 1:]:
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) < 4:
            continue
        rows.append((v[it], v[ilb]))
    rows.sort()
    return np.array(rows)


def step_at(rows, t):
    x = rows[rows[:, 0] <= t]
    return float(x[-1, 1]) if len(x) else float("nan")


def main():
    setup_font()
    b = read_log(B10 / "sddp_training_log.csv")
    n = read_log(NO / "sddp_training_log.csv")
    grid = np.linspace(0, min(b[-1, 0], n[-1, 0]), 1200)
    delta = np.array([step_at(b, t) - step_at(n, t) for t in grid])
    # 前 5 分钟是异步初始化瞬态，不能代表稳定收敛差异。
    mask = grid >= 300.0
    peak_i = int(np.nanargmax(np.where(mask, delta, -np.inf)))
    peak_t, peak_d = float(grid[peak_i]), float(delta[peak_i])

    b_ub = json.loads((B10 / "sim_3000paths.json").read_text(encoding="utf-8"))
    n_ub = json.loads((NO / "sim_3000paths.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    ax.plot(b[:, 0] / 60, b[:, 1], color="tab:blue", lw=1.8, label="static-10")
    ax.plot(n[:, 0] / 60, n[:, 1], color="tab:orange", lw=1.8,
            label="无分块")
    ax.axhline(float(b[-1, 1]), color="tab:blue", lw=0.9, ls=":")
    ax.axhline(float(n[-1, 1]), color="tab:orange", lw=0.9, ls=":")
    ax.set_title("V2 连续版：分块 B=10 vs 无分块（跳过初始瞬态）")
    ax.set_xlabel("训练时间（分钟）")
    ax.set_ylabel("下界 LB")
    ax.set_xlim(2, b[-1, 0] / 60)
    lo = min(step_at(b, 120), step_at(n, 120))
    ax.set_ylim(lo - 100, max(float(b[-1, 1]), float(n[-1, 1])) + 60)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right")
    ax.annotate(f"final {b[-1,1]:,.0f}", xy=(b[-1,0]/60, b[-1,1]),
                xytext=(-10, 12), textcoords="offset points",
                color="tab:blue", fontsize=8, ha="right")
    ax.annotate(f"final {n[-1,1]:,.0f}", xy=(n[-1,0]/60, n[-1,1]),
                xytext=(-10, -18), textcoords="offset points",
                color="tab:orange", fontsize=8, ha="right")

    ax = axes[1]
    ax.plot(grid[mask] / 60, delta[mask], color="tab:green", lw=1.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(peak_t / 60, color="tab:red", lw=0.8, ls=":")
    ax.annotate(f"峰值 {peak_d:+.0f} @ {peak_t/60:.1f} min",
                xy=(peak_t / 60, peak_d), xytext=(10, 8),
                textcoords="offset points", fontsize=9, color="tab:red")
    ax.set_title("5 min 后同期差值：LB(static-10) − LB(无分块)")
    ax.set_xlabel("训练时间（分钟）")
    ax.set_ylabel("LB 差值")
    ax.set_xlim(5, grid[-1] / 60)
    ax.grid(alpha=0.3, lw=0.5)

    fig.suptitle("V2 下界收敛对比：static-10 vs 无分块", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = B10 / "lb_compare_b10_vs_noblock.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    stats = {
        "b10_final_lb": float(b[-1, 1]),
        "noblock_final_lb": float(n[-1, 1]),
        "b10_final_ub": float(b_ub["mean"]),
        "noblock_final_ub": float(n_ub["mean"]),
        "delta_final": float(b[-1, 1] - n[-1, 1]),
        "delta_peak": peak_d,
        "delta_peak_time_s": peak_t,
    }
    (B10 / "lb_compare_b10_vs_noblock.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
