# _pair_fcmc.py — 逐路径配对 UB 比较（读 simulate.py 落的 npz）。
#   python test/_pair_fcmc.py <ref_dir> <cmp_dir> [...]
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")


def _load(d: Path, name: str):
    p = d / name
    if not p.is_file():
        raise SystemExit(f"缺少 {p} —— 先跑 simulate.py")
    z = np.load(p)
    return z["totals"].astype(float), z["scene"]


def main(ref: str, comps, name: str = "sim_200paths.npz"):
    rd = Path(ref)
    rtot, rscene = _load(rd, name)
    print(f"参照 {rd.name}: n={len(rtot)} 均值={rtot.mean():.2f} "
          f"SE={rtot.std(ddof=1)/np.sqrt(len(rtot)):.2f}")
    print(f"{'对比目录':34s} {'均值':>12s} {'Δ=对比-参照':>12s} "
          f"{'SE(Δ)':>9s} {'95%CI':>18s} {'胜/平/负':>9s}")
    for c in comps:
        cd = Path(c)
        ctot, cscene = _load(cd, name)
        if ctot.shape != rtot.shape:
            raise SystemExit(f"{cd.name}: 路径数 {len(ctot)} ≠ 参照 {len(rtot)}")
        if not np.array_equal(cscene, rscene):
            raise SystemExit(f"{cd.name}: 场景路径与参照不同，不能配对")
        diff = ctot - rtot
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n)
        lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
        win, tie, loss = int((diff < -1e-9).sum()), int((np.abs(diff) <= 1e-9).sum()), \
            int((diff > 1e-9).sum())
        print(f"{cd.name:34s} {ctot.mean():12.2f} {diff.mean():12.2f} "
              f"{se:9.2f} [{lo:8.2f}, {hi:8.2f}] {win:4d}/{tie:3d}/{loss:4d}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
