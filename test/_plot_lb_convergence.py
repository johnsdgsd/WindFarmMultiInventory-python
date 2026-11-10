# _plot_lb_convergence.py — 画 V2 连续版下界收敛图（全轮 + 末段放大）。
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np              # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "asyncV2_20261109_215549"


def parse_log(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        head = line.split()
        if head[:1] == ["iteration"] and "bound" in head:
            i_lb = head.index("bound")
            i_t = head.index("time") if "time" in head else None
            break
    else:
        raise SystemExit(f"no header in {path}")
    rows = []
    for line in lines[i + 1:]:
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) < 4:
            continue
        rows.append((v[0], v[i_lb], v[i_t] if i_t is not None else np.nan))
    rows.sort(key=lambda r: r[0])
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def main():
    setup_font()
    it, lb, t = parse_log(RUN / "sddp_training_log.csv")
    meta = json.loads((RUN / "sim_3000paths.json").read_text(encoding="utf-8"))
    ub = float(meta["mean"])
    lb_final = float(meta["lb"])
    lmax = np.maximum.accumulate(lb)

    last = 300
    m = it > it.max() - last
    slope = (lmax[-1] - lmax[m][0]) / max(1.0, it[-1] - it[m][0])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    ax.plot(it, lb, color="tab:blue", lw=1.0, alpha=0.35, label="当前 LB")
    ax.plot(it, lmax, color="tab:blue", lw=2.0, label="历史最大 LB")
    ax.axhline(ub, color="tab:orange", lw=1.5, ls="--",
               label=f"3000 路径仿真 UB = {ub:,.0f}")
    ax.axhline(lb_final, color="tab:green", lw=1.0, ls=":",
               label=f"末轮 LB = {lb_final:,.0f}")
    ax.set_title("V2 连续版：下界收敛（全轮）", fontsize=11)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("期望总成本")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax = axes[1]
    k = max(0, len(it) - 350)
    ax.plot(it[k:], lb[k:], color="tab:blue", lw=1.0, alpha=0.4)
    ax.plot(it[k:], lmax[k:], color="tab:blue", lw=2.0)
    ax.axhline(ub, color="tab:orange", lw=1.5, ls="--")
    ax.axhline(lb_final, color="tab:green", lw=1.0, ls=":")
    ax.set_title(f"末 {last} 轮放大：平均 +{slope:.3f}/轮", fontsize=11)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("下界 LB")
    ax.grid(alpha=0.3, lw=0.5)

    fig.suptitle("SDDP V2 连续版下界收敛", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RUN / "lb_convergence.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    stats = {
        "n_iterations": int(len(it)),
        "wall_s": float(t[-1]),
        "lb_final": lb_final,
        "ub_3000": ub,
        "gap_pct": float(meta["gap_pct"]),
        "lb_last_300_delta": float(lmax[-1] - lmax[m][0]),
        "lb_last_300_per_iter": float(slope),
        "lb_last_100_delta": float(
            lmax[-1] - lmax[it > it.max() - 100][0]),
        "lb_at_600": float(lmax[np.searchsorted(it, 600) - 1]),
        "lb_at_900": float(lmax[np.searchsorted(it, 900) - 1]),
        "lb_at_final": float(lmax[-1]),
    }
    (RUN / "lb_convergence_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
