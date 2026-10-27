# 读一个结果目录，输出参数调优用的紧凑摘要（不重仿真）。
#   python test/_tune_report.py results/<dir>
# 需要该目录已有 sim_*paths.npz（simulate.py 产物）。
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config  # noqa: E402


def load_log(d: Path) -> np.ndarray:
    rows = []
    for line in (d / "sddp_training_log.csv").read_text(
            encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) == 6:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    return np.array(rows)


def parse_report(d: Path) -> dict:
    fs = sorted(d.glob("sim_report_*.txt"))
    if not fs:
        return {}
    txt = fs[-1].read_text(encoding="utf-8", errors="replace")
    out = {"file": fs[-1].name}
    for key, pat in [("ub", r"均值 UB\s*:\s*([-\d.]+)"),
                     ("lb", r"下界 LB\s*:\s*([-\d.]+)"),
                     ("gap", r"间隙 gap\s*:\s*([-\d.]+)"),
                     ("paths", r"路径数\s*:\s*(\d+)")]:
        m = re.search(pat, txt)
        out[key] = float(m.group(1)) if m else None
    for key, pat in [("proc", r"采购\s+[\d,.]+\s+([\d.]+)%"),
                     ("hold", r"持有\s+[\d,.]+\s+([\d.]+)%"),
                     ("down", r"停机\s+[\d,.]+\s+([\d.]+)%"),
                     ("act", r"主动调拨\s+[\d,.]+\s+([\d.]+)%"),
                     ("pas", r"被动调拨\s+[\d,.]+\s+([\d.]+)%")]:
        m = re.search(pat, txt)
        out[key] = float(m.group(1)) if m else None
    out["audit"] = "全部通过" in txt
    return out


def stats(d: Path) -> dict:
    fs = sorted(d.glob("sim_*paths.npz"), key=lambda p: p.stat().st_size)
    if not fs:
        return {}
    z = np.load(fs[-1], allow_pickle=True)
    pcfg = load_problem_config(d / "config" / "problem.toml")
    F, W, S, q, terms = z["F"], z["W_in"], z["S_end"], z["q"], z["terms"]
    n = terms.shape[0]
    totF = F.sum() / n
    t = terms.sum(axis=(0, 1)) / n
    yearly = [F[:, 12 * k:12 * (k + 1)].sum() / n for k in range(5)]
    w = W.sum() / n
    qmax = np.asarray(pcfg.Qmax)
    imax = np.asarray(pcfg.Imax)
    # 剔除前 4 + 后 3 期的内区成本结构
    ti = terms[:, 4:-3, :].sum(axis=(0, 1)) / n
    return {
        "npz": fs[-1].name, "npaths": n, "fails": totF, "yearly": yearly,
        "wbar": w / totF, "wshare": float((W > 1e-9).mean()),
        "inv": float(S.mean()), "q0": float((q <= 1e-9).mean()),
        "qmax": float((q >= qmax - 1e-9).mean()),
        "util": float(q.sum() / (60 * qmax.sum() * n)),
        "terms": t, "per_fail": t / totF, "shares": 100 * t / t.sum(),
        "inner_shares": 100 * ti / ti.sum(),
        "imax_hit": float((S >= imax - 1e-6).mean()),
    }


def main() -> None:
    d = Path(sys.argv[1])
    a = load_log(d)
    r = parse_report(d)
    s = stats(d)
    print(f"dir {d.name}")
    if len(a):
        for T in (600, 1200, 1800, 2400, 3000, 3600):
            if a[:, 3].max() >= T - 1:
                i = np.argmin(np.abs(a[:, 3] - T))
                print(f"  LB@{T:4d}s = {a[i, 2]:12.1f}  (iter {int(a[i, 0])}, "
                      f"t={a[i, 3]:.0f})")
        print(f"  final LB = {a[-1, 2]:12.1f}  iter {int(a[-1, 0])}  "
              f"t={a[-1, 3]:.0f} s")
    if r:
        print(f"  UB={r.get('ub')}  LB={r.get('lb')}  gap={r.get('gap')}%  "
              f"paths={r.get('paths')}  audit={r.get('audit')}")
    if s:
        print(f"  fails={s['fails']:.1f}  yearly={['%.1f' % x for x in s['yearly']]}")
        print(f"  Wbar={s['wbar']:.3f}  W>0={s['wshare']:.3f}  inv={s['inv']:.2f}  "
              f"q0={s['q0']:.3f}  q@Qmax={s['qmax']:.3f}  util={s['util']:.3f}")
        print("  shares      " + "  ".join(
            f"{k}={v:.1f}%" for k, v in zip(("proc", "hold", "down", "act", "pas"),
                                            s["shares"])))
        print("  inner(5-57) " + "  ".join(
            f"{k}={v:.1f}%" for k, v in zip(("proc", "hold", "down", "act", "pas"),
                                            s["inner_shares"])))
        print("  per-failure " + "  ".join(
            f"{k}={v:.1f}" for k, v in zip(("proc", "hold", "down", "act", "pas"),
                                           s["per_fail"])))


if __name__ == "__main__":
    main()
