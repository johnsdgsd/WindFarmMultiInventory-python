#!/usr/bin/env python3
"""SDDP 训练收敛图：读结果目录里的 sddp_training_log.csv（SDDP 的文本日志），
画上界、下界两条曲线，PNG 存回同一个结果目录。

    python plot/plot_bounds.py                      # results/ 下最新的一个目录
    python plot/plot_bounds.py results/T24_2026xxxx_xxxxxx
    python plot/plot_bounds.py --en results/T24_...  # 英文标签（投稿用）

日志里的表固定六列：iteration / simulation / bound / time (s) / solves / pid，
simulation 是前向仿真的上界估计，bound 是下界。列位置按表头名字找，不写死索引。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Windows 下 Python 默认按本地代码页编码 stdout，中文会输出成乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

L = {
    "zh": {
        "ub": "上界 UB",
        "lb": "下界 LB",
        "x": "迭代轮数",
        "y": "期望总成本",
        "title": "SDDP 训练收敛：{name}",
        "nodata": "没解析到数据行（{path}）",
    },
    "en": {
        "ub": "Upper bound",
        "lb": "Lower bound",
        "x": "Iteration",
        "y": "Expected total cost",
        "title": "SDDP convergence: {name}",
        "nodata": "no data rows parsed ({path})",
    },
}


def setup_font() -> str | None:
    """挑一个可用的中文字体，返回字体名；没有则返回 None（调用方退回英文）。"""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def parse_log(path: Path) -> list[tuple[float, float, float]]:
    """返回 [(iteration, UB, LB), ...]。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        head = line.split()
        if len(head) < 4 or head[0] != "iteration" or "bound" not in head:
            continue
        i_ub, i_lb = head.index("simulation"), head.index("bound")
        # 表头里 "time (s)" 占两个 token、数据行占一个 —— 只有排在它之前的列
        # 才能按表头索引直接取
        i_time = head.index("time") if "time" in head else len(head)
        if max(i_ub, i_lb) >= i_time:
            raise SystemExit(f"表头列序与预期不符，无法定位上下界列: {line!r}")
        rows = []
        for row in lines[i + 1:]:
            try:
                vals = [float(c) for c in row.split()]
            except ValueError:
                if rows:                      # 表尾的分隔线 / 后续摘要
                    break
                continue                      # 表头的分隔线，继续往下找
            if len(vals) < 4:
                if rows:
                    break
                continue
            rows.append((vals[0], vals[i_ub], vals[i_lb]))
        return rows
    return []


def plot_one(d: Path, dpi: int, lang: str, out: Path) -> dict:
    log = d / "sddp_training_log.csv"
    if not log.is_file():
        raise SystemExit(f"找不到训练日志: {log}")
    rows = parse_log(log)
    if not rows:
        raise SystemExit(L[lang]["nodata"].format(path=log))

    it = [r[0] for r in rows]
    ub = [r[1] for r in rows]
    lb = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(it, ub, color="tab:orange", lw=1.6, label=L[lang]["ub"])
    ax.plot(it, lb, color="tab:blue", lw=1.8, label=L[lang]["lb"])

    ax.set_xlabel(L[lang]["x"])
    ax.set_ylabel(L[lang]["y"])
    ax.set_title(L[lang]["title"].format(name=d.name), fontsize=11)
    ax.margins(x=0.01)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)

    return dict(n=len(rows), it0=it[0], it1=it[-1], lb=lb[-1], ub=ub[-1], out=out)


def newest_results_dir() -> Path:
    cands = [p for p in RESULTS.iterdir()
             if p.is_dir() and (p / "sddp_training_log.csv").is_file()]
    if not cands:
        raise SystemExit(f"{RESULTS} 下没有含训练日志的目录")
    return max(cands, key=lambda p: (p / "sddp_training_log.csv").stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="画 SDDP 训练的上下界收敛图，PNG 存进对应结果目录")
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="结果目录（缺省用 results/ 下最新的一个）")
    ap.add_argument("--en", action="store_true", help="英文标签")
    ap.add_argument("--dpi", type=int, default=150, help="图片分辨率（默认 150）")
    args = ap.parse_args()

    lang = "en" if args.en else ("zh" if setup_font() else "en")
    if lang == "en":
        plt.rcParams["axes.unicode_minus"] = False

    for d in (args.dirs or [newest_results_dir()]):
        d = d if d.is_absolute() else ROOT / d
        if not d.is_dir():
            raise SystemExit(f"目录不存在: {d}")
        out = d / "bounds.png"
        r = plot_one(d, args.dpi, lang, out)
        print(f"{d.name}:  {r['n']} 个点（迭代 {r['it0']:.0f}–{r['it1']:.0f}），"
              f"末值 LB {r['lb']:.2f} / UB {r['ub']:.2f}  →  {r['out']}")


if __name__ == "__main__":
    sys.exit(main())
