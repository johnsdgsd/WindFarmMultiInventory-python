# _t_partition_summary.py — 读取一次分区实验的结果目录，和 V2 基线做时间对齐摘要。
#   python test/_t_partition_summary.py results/partT60_xxx
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "base_T60_20261026_204105"


def load_segments(d: Path):
    files = sorted(d.glob("sddp_training_log_prev_*.csv")) + [d / "sddp_training_log.csv"]
    arrs, off = [], 0.0
    for f in files:
        if not f.is_file():
            continue
        rows = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) == 6:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
        a = np.asarray(rows)
        if len(a) == 0:
            continue
        a = a.copy()
        a[:, 3] += off
        off = float(a[-1, 3])
        arrs.append(a)
    return np.vstack(arrs) if arrs else np.zeros((0, 6))


def lb_at(a, sec):
    if len(a) == 0:
        return float("nan"), -1
    i = np.where(a[:, 3] >= sec)[0]
    if len(i) == 0:
        return float(a[-1, 2]), int(a[-1, 0])
    return float(a[i[0], 2]), int(a[i[0], 0])


def main():
    d = Path(sys.argv[1]).resolve()
    new = load_segments(d)
    base = load_segments(BASE)
    print(f"new = {d.name}")
    print(" t(s) |   new LB   |  base LB   |  new iter | base iter")
    for sec in (600, 1800, 3600, 7503):
        nlb, ni = lb_at(new, sec)
        blb, bi = lb_at(base, sec)
        print(f"{sec:5d} | {nlb:10.2f} | {blb:10.2f} | {ni:8d} | {bi:8d}")
    print(f"new final: iter={int(new[-1,0])} time={new[-1,3]:.1f}s LB={new[-1,2]:.4f} "
          f"solves={int(new[-1,4])}")
    print(f"base final: iter={int(base[-1,0])} time={base[-1,3]:.1f}s LB={base[-1,2]:.4f} "
          f"solves={int(base[-1,4])}")


if __name__ == "__main__":
    main()
