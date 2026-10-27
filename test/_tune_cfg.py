# 生成 config/tune/<tag>/ 实验配置：复制主配置并按 key 覆盖。
#   python test/_tune_cfg.py <tag> k=v ...
# 规则：iterations/max_train_seconds/n_simulations/chunk_size/tag/checkpoint_interval
#       覆盖 solver.toml，其余覆盖 problem.toml；tag 自动加引号。
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER_KEYS = {"iterations", "max_train_seconds", "n_simulations",
               "chunk_size", "tag", "checkpoint_interval"}


def set_key(text: str, key: str, val: str) -> str:
    pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    if not pat.search(text):
        raise KeyError(f"key {key} not found")
    return pat.sub(f"{key} = {val}", text, count=1)


def main() -> None:
    tag = sys.argv[1]
    over = dict(a.split("=", 1) for a in sys.argv[2:])
    d = ROOT / "config" / "tune" / tag
    d.mkdir(parents=True, exist_ok=True)
    p = (ROOT / "config" / "problem.toml").read_text(encoding="utf-8")
    s = (ROOT / "config" / "solver.toml").read_text(encoding="utf-8")
    for k, v in over.items():
        if k in SOLVER_KEYS:
            if k == "tag":
                v = f'"{v}"'
            s = set_key(s, k, v)
        else:
            p = set_key(p, k, v)
    (d / "problem.toml").write_text(p, encoding="utf-8")
    (d / "solver.toml").write_text(s, encoding="utf-8")
    print("wrote", d)


if __name__ == "__main__":
    main()
