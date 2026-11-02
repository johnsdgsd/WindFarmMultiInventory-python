# _plot_partition.py — 自适应分区结果 vs 基线：LB-时间、LB-迭代、每轮耗时、簇数。
#   python test/_plot_partition.py results/partT60_pilot_xxx
# 基线默认用 base_T60_20261026_204105 的两段日志。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "base_T60_20261026_204105"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"


def load_log(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) == 6:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    return np.asarray(rows)


def load_segments(d: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 (iteration, cum_time, lb, sim, raw_time, solves)，按归档顺序拼接。"""
    files = sorted(d.glob("sddp_training_log_prev_*.csv")) + [d / "sddp_training_log.csv"]
    arrs, off = [], 0.0
    for f in files:
        if not f.is_file():
            continue
        a = load_log(f)
        if len(a) == 0:
            continue
        a = a.copy()
        a[:, 3] += off
        off = float(a[-1, 3])
        arrs.append(a)
    if not arrs:
        raise FileNotFoundError(d)
    return np.vstack(arrs)


def events(d: Path, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 partition_log.csv 还原细化事件及当时的平均/最大簇数。"""
    p = d / "partition_log.csv"
    if not p.is_file():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cur = {t: 1 for t in range(1, T + 1)}
    rows = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        q = line.split(",")
        if len(q) < 4:
            continue
        it, st, new = int(q[0]), int(q[1]), int(q[3])
        cur[st] = new
        rows[it] = (float(np.mean(list(cur.values()))),
                    int(max(cur.values())))
    if not rows:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    its = np.array(sorted(rows))
    return its, np.array([rows[i][0] for i in its]), np.array([rows[i][1] for i in its])


def main():
    d = Path(sys.argv[1]).resolve()
    T = 60
    new = load_segments(d)
    base = load_segments(BASE)
    # 日志列：iteration, simulation, bound, time, solves, pid
    it, sim, lb, t, sol = new[:, 0], new[:, 1], new[:, 2], new[:, 3], new[:, 4]
    bit, bsim, blb, bt, bsol = base[:, 0], base[:, 1], base[:, 2], base[:, 3], base[:, 4]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.plot(bt / 3600, blb, color="#777777", lw=2.0, label="基线 V2（231 轮）")
    ax.plot(t / 3600, lb, color="#d62728", lw=2.0, label=d.name)
    ax.set_xlabel("累计训练时间（小时）")
    ax.set_ylabel("下界 LB")
    ax.set_title("自适应分区 vs V2 基线：LB–时间")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "lb_vs_time.png", dpi=170, bbox_inches="tight")
    fig.savefig(d / "lb_vs_time.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.plot(bit, blb, color="#777777", lw=2.0, label="基线 V2")
    ax.plot(it, lb, color="#d62728", lw=2.0, label=d.name)
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("下界 LB")
    ax.set_title("自适应分区 vs V2 基线：LB–迭代")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "lb_vs_iteration.png", dpi=170, bbox_inches="tight")
    fig.savefig(d / "lb_vs_iteration.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.plot(bsol, blb, color="#777777", lw=2.0, label="基线 V2")
    ax.plot(sol, lb, color="#d62728", lw=2.0, label=d.name)
    ax.set_xlabel("累计实际 LP 解次数")
    ax.set_ylabel("下界 LB")
    ax.set_title("自适应分区 vs V2 基线：LB–LP 次数")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "lb_vs_lp.png", dpi=170, bbox_inches="tight")
    fig.savefig(d / "lb_vs_lp.svg", bbox_inches="tight")
    plt.close(fig)

    dt = np.diff(t)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(it[1:], dt, color="#1f77b4", lw=0.8)
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("每轮用时（秒）")
    ax.set_title("每轮训练耗时")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(d / "time_per_iteration.png", dpi=170, bbox_inches="tight")
    fig.savefig(d / "time_per_iteration.svg", bbox_inches="tight")
    plt.close(fig)

    eit, lm, lx = events(d, T)
    if len(lm):
        fig, ax = plt.subplots(figsize=(11, 5.6))
        ax.step(eit, lm, where="post", label="mean L")
        ax.step(eit, lx, where="post", label="max L")
        ax.set_xlabel("迭代轮次")
        ax.set_ylabel("每阶段簇数（事件后）")
        ax.set_title("自适应分区簇数")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "clusters_vs_iteration.png", dpi=170, bbox_inches="tight")
        fig.savefig(d / "clusters_vs_iteration.svg", bbox_inches="tight")
        plt.close(fig)

    # 同时把新结果整理成 CSV（不覆盖 npz）
    out = np.c_[it, t, lb, sim, sol]
    np.savetxt(d / "partition_compare.csv", out, delimiter=",",
               header="iteration,cum_time_s,lb,forward_mean,solves",
               comments="", fmt=["%d", "%.4f", "%.6f", "%.6f", "%.0f"])
    print("saved plots/csv to", d)


if __name__ == "__main__":
    main()
