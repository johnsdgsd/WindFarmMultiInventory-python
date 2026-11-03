# _summ_fcmc.py — 汇总固定簇 pilot 的 LB/时间/割数（读结果目录，不重跑）。
#   python test/_summ_fcmc.py results/<dir> [...]
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")


def _summary(d: Path) -> dict:
    fs = sorted(d.glob("summary_*.txt"))
    if not fs:
        return {}
    txt = fs[-1].read_text(encoding="utf-8", errors="replace")
    out = {}
    pats = {"rounds": r"训练轮次\s*:\s*(\d+)",
            "lb": r"下界 LB\s*:\s*([\d.]+)",
            "ub": r"上界 UB\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
            "gap": r"间隙 gap\s*:\s*([\d.]+)",
            "cuts": r"割数\s*:\s*(\d+)",
            "wall": r"用时\s*:\s*([\d.]+)"}
    for k, p in pats.items():
        m = re.search(p, txt)
        if not m:
            continue
        if k == "ub":
            out["ub"], out["ci"] = float(m.group(1)), float(m.group(2))
        else:
            out[k] = float(m.group(1)) if k in ("lb", "gap", "wall") else int(m.group(1))
    return out


def _log(d: Path):
    p = d / "sddp_training_log.csv"
    if not p.is_file():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                     float(parts[3]), int(parts[4])))
    return np.asarray(rows, float) if rows else None


def main(dirs):
    print(f"{'目录':32s} {'轮':>5s} {'LB':>12s} {'UB':>12s} {'gap%':>7s} "
          f"{'割数':>8s} {'cuts MB':>8s} {'墙钟':>7s} "
          f"{'LB@60':>10s} {'LB@120':>10s} {'LB@300':>10s} {'LB@600':>10s}")
    for d in dirs:
        d = Path(d)
        s = _summary(d)
        lg = _log(d)
        cj = d / "cuts.json"
        mb = cj.stat().st_size / 1e6 if cj.is_file() else float("nan")
        if lg is None:
            print(f"{d.name:32s}  (无训练日志)")
            continue
        lbck = []
        for sec in (60, 120, 300, 600):
            mask = lg[:, 3] <= sec
            lbck.append(lg[mask, 2][-1] if mask.any() else float("nan"))
        print(f"{d.name:32s} {s.get('rounds', len(lg)):5d} "
              f"{s.get('lb', lg[-1, 2]):12.2f} {s.get('ub', float('nan')):12.2f} "
              f"{s.get('gap', float('nan')):7.2f} {s.get('cuts', 0):8d} "
              f"{mb:8.1f} {s.get('wall', lg[-1, 3]):7.1f} "
              + " ".join(f"{v:10.1f}" for v in lbck))


if __name__ == "__main__":
    main(sys.argv[1:])
