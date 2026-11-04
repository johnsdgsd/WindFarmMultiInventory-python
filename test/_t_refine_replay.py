# _t_refine_replay.py — 在固定 A 的 277 轮日志上回放停滞触发次数。
#   python test/_t_refine_replay.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.partition import stall_trigger  # noqa: E402

D = ROOT / "results" / "fcmcT60_full_L8_20261103_165647"


def _rows(p: Path):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if s and s[0].isdigit():
            out.append((int(s[0]), float(s[3]), float(s[2])))
    return out


def replay(window: int, warmup: int = 20):
    # 续训从第 213 轮 checkpoint 重做 214-229：前半只保留 1..213，避免重复轮次。
    r1 = [r for r in _rows(D / "sddp_training_log_prev_20261103_183513.csv")
          if r[0] <= 213]
    r2 = _rows(D / "sddp_training_log.csv")
    off = r1[-1][1]
    rows = r1 + [(i, t + off, lb) for i, t, lb in r2]
    best = 0.0
    hist, events = [], []
    for i, _t, lb in rows:
        best = max(best, lb)
        if i <= warmup:
            continue
        hist.append(best)
        if stall_trigger(hist, window, 1e-3, 1.0):
            events.append(i)
            hist = []
    return events


def main():
    e10 = replay(10)
    e20 = replay(20)
    print(f"n=10 events={len(e10)}: {e10}")
    print(f"n=20 events={len(e20)}: {e20}")
    assert len(e10) >= 10, len(e10)
    assert 3 <= len(e20) <= 7, len(e20)
    assert e10 and e20 and e10[0] > 100
    print("OK _t_refine_replay")


if __name__ == "__main__":
    main()
