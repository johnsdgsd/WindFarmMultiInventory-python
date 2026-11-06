# _gen_integer_cfg.py — 生成 Tier 1 整数 SDDP 的冒烟/T60 配置。
#   python test/_gen_integer_cfg.py
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config"


def _add_integer(text: str, enabled: bool, scope: str = "tier1_logistics") -> str:
    block = f"[integer]\nenabled = {'true' if enabled else 'false'}\nscope = \"{scope}\"\n\n"
    if re.search(r"^\[integer\]", text, flags=re.M):
        raise RuntimeError("配置里已有 [integer]")
    return text.replace("[parallel]", block + "[parallel]", 1)


def _set(text: str, key: str, val: str) -> str:
    out, n = re.subn(rf"^{key}\s*=.*$", f"{key} = {val}", text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"找不到唯一的 {key}")
    return out


def _write(name: str, p_src: Path, s_src: Path, *, enabled: bool,
           secs: float, sims: int, tag: str) -> None:
    d = CFG / "tune" / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p_src, d / "problem.toml")
    txt = s_src.read_text(encoding="utf-8")
    txt = _add_integer(txt, enabled)
    txt = _set(txt, "max_train_seconds", f"{secs:g}")
    txt = _set(txt, "n_simulations", str(sims))
    txt = _set(txt, "tag", f'"{tag}"')
    (d / "solver.toml").write_text(txt, encoding="utf-8")
    print(d.relative_to(ROOT))


def main():
    p = CFG / "problem.toml"
    s = CFG / "solver.toml"

    smoke = CFG / "smoke_integer"
    smoke.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG / "smoke" / "problem.toml", smoke / "problem.toml")
    txt = (CFG / "smoke" / "solver.toml").read_text(encoding="utf-8")
    txt = _add_integer(txt, True)
    txt = _set(txt, "tag", '"smoke_integer"')
    (smoke / "solver.toml").write_text(txt, encoding="utf-8")
    print(smoke.relative_to(ROOT))

    _write("int_t1_pilot_lp", p, s, enabled=False, secs=600,
           sims=200, tag="int_t1_pilot_lp")
    _write("int_t1_pilot", p, s, enabled=True, secs=600,
           sims=200, tag="int_t1_pilot")
    _write("int_t1_formal", p, s, enabled=True, secs=7503.3,
           sims=3000, tag="int_t1_formal")


if __name__ == "__main__":
    main()
