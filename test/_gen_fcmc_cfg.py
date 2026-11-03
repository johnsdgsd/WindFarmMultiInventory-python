# _gen_fcmc_cfg.py — 生成固定分区簇多割的 T60 实验配置（一次生成，可复现）。
#   python test/_gen_fcmc_cfg.py
"""源配置 = config/tune/multicut_t60_pilot（T60 V2，mode=multi，600 s，LML1）。
新配置只改 [multicut]、tag 和 max_train_seconds。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "config" / "tune"
SRC = TUNE / "multicut_t60_pilot"

MULTICUT = """# 固定分区簇多割：n_clusters=L，每簇一个 theta/一条聚合代表割
[multicut]
n_clusters = {L}
group_mode = "{gm}"
"""


def _ensure_multicut(text: str, L: int, gm: str) -> str:
    block = MULTICUT.format(L=L, gm=gm)
    m = re.search(r"^\[multicut\]", text, flags=re.M)
    if m:
        nxt = text.find("\n[", m.end())
        if nxt < 0:
            return text[:m.start()] + block
        return text[:m.start()] + block + "\n" + text[nxt + 1:]
    m2 = re.search(r"^seed\s*=\s*\d+\s*$", text, flags=re.M)
    if not m2:
        raise RuntimeError("solver.toml 里找不到 seed 行")
    return text[:m2.end()] + "\n\n" + block + text[m2.end():]


def _set(text: str, key: str, val: str) -> str:
    out, n = re.subn(rf"^{key}\s*=.*$", f"{key} = {val}", text,
                     count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"未找到唯一的 {key}")
    return out


def main():
    base_problem = (SRC / "problem.toml").read_text(encoding="utf-8")
    base_solver = (SRC / "solver.toml").read_text(encoding="utf-8")

    # 旧配置显式写出 [multicut]，避免依赖 loader 默认值
    for name in ("multicut_t60_pilot", "multicut_t60_full", "multicut_t60_control"):
        d = TUNE / name
        txt = (d / "solver.toml").read_text(encoding="utf-8")
        if not re.search(r"^\[multicut\]", txt, flags=re.M):
            txt = _ensure_multicut(txt, 0, "scene")
            (d / "solver.toml").write_text(txt, encoding="utf-8")

    for tag, L, gm in (("L1", 1, "failure_quantile"),
                       ("L4", 4, "failure_quantile"),
                       ("L8", 8, "failure_quantile"),
                       ("L16", 16, "failure_quantile"),
                       ("L32", 32, "failure_quantile"),
                       ("stride8", 8, "failure_stride")):
        d = TUNE / f"fcmc_t60_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "problem.toml").write_text(base_problem, encoding="utf-8")
        txt = base_solver
        txt = _ensure_multicut(txt, L, gm)
        txt = _set(txt, "max_train_seconds", "600")
        txt = _set(txt, "tag", f'"fcmcT60_{tag}"')
        (d / "solver.toml").write_text(txt, encoding="utf-8")
        print(d.relative_to(ROOT))

    # 正式对照：与 V2 基线同墙钟、同仿真路径数
    d = TUNE / "fcmc_t60_full_L8"
    d.mkdir(parents=True, exist_ok=True)
    (d / "problem.toml").write_text(base_problem, encoding="utf-8")
    txt = _ensure_multicut(base_solver, 8, "failure_quantile")
    txt = _set(txt, "max_train_seconds", "7503.3")
    txt = _set(txt, "n_simulations", "3000")
    txt = _set(txt, "tag", '"fcmcT60_full_L8"')
    (d / "solver.toml").write_text(txt, encoding="utf-8")
    print(d.relative_to(ROOT))


if __name__ == "__main__":
    main()
