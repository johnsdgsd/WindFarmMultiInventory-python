# 读两份 2000 路径仿真，做逐路径配对比较（同 seed / 同 chunk）。
#   python test/_t_compare_replay.py <基线目录> <回放目录>
# 只读 npz/json，不写结果目录。
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")


def load(d):
    d = Path(d)
    if not d.is_absolute():
        d = ROOT / d
    z = np.load(d / "sim_2000paths.npz")
    meta = json.loads((d / "sim_2000paths.json").read_text(encoding="utf-8"))
    return z, meta


def main(base_dir, replay_dir):
    a, ma = load(base_dir)
    b, mb = load(replay_dir)
    ta, tb = a["totals"], b["totals"]
    if len(ta) != len(tb):
        raise SystemExit(f"路径数不同：{len(ta)} vs {len(tb)}")
    d = tb - ta
    mean, std = float(d.mean()), float(d.std(ddof=1))
    se = std / np.sqrt(len(d))
    ci = 1.959963984540054 * se
    print("=" * 72)
    print("逐路径配对比较（回放 − 基线；同 seed 同 chunk）")
    print("=" * 72)
    print(f"路径数            {len(d)}")
    print(f"基线 UB           {ta.mean():.4f}   LB {ma['lb']:.4f}  "
          f"gap {ma['gap_pct']:.4f}%")
    print(f"回放 UB           {tb.mean():.4f}   LB {mb['lb']:.4f}  "
          f"gap {mb['gap_pct']:.4f}%")
    print(f"UB 差（回放−基线） {mean:+.4f} ± {ci:.4f}（95% CI）")
    print(f"  标准误 {se:.4f}，配对 t = {mean/se:+.2f}，"
          f"回放更低的路径占比 {float((d < 0).mean()):.2%}")
    print(f"LB 差             {mb['lb'] - ma['lb']:+.4f}")
    print(f"gap 差            {mb['gap_pct'] - ma['gap_pct']:+.4f} pp")
    print("\n五项成本（2000 路径总量，回放 − 基线）")
    for i, k in enumerate(("采购", "持有", "停机", "主动调拨", "被动调拨")):
        va = float(a["terms"][:, :, i].sum())
        vb = float(b["terms"][:, :, i].sum())
        print(f"  {k:<6} {va:>14,.1f} → {vb:>14,.1f}   {vb-va:>+12,.1f}")
    print("\n逐期平均成本差（回放−基线，第 1..T 期）")
    sd = b["cost"].mean(axis=0) - a["cost"].mean(axis=0)
    print("  " + " ".join(f"{v:+.3g}" for v in sd))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("用法: python test/_t_compare_replay.py <基线目录> <回放目录>")
    main(sys.argv[1], sys.argv[2])
