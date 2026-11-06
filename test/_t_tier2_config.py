# _t_tier2_config.py — Tier2 scope/rounding/n0_mode 与版本门禁。
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (IntegerConfig, load_problem_config,  # noqa: E402
                         load_solver_config)
from utils.cut import FORMAT_VERSION                         # noqa: E402
from utils.io import problem_diff                             # noqa: E402


def main():
    p = load_problem_config(ROOT / "config" / "smoke_tier2" / "problem.toml")
    s = load_solver_config(ROOT / "config" / "smoke_tier2" / "solver.toml")
    assert p.n0_mode == "integer_largest_remainder"
    assert s.integer_scope == "tier2_full"
    assert s.integer_rounding == "cumulative_tail_nearest"
    assert FORMAT_VERSION == 6
    print("  [PASS] Tier2 config/meta 口径；FORMAT_VERSION=6")

    for kw in ({"scope": "bad"}, {"scope": "tier2_full", "rounding": "none"},
               {"enabled": True, "scope": "none"}):
        try:
            IntegerConfig(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 integer 配置未拒绝：{kw}")
    print("  [PASS] 非法 scope/rounding/enabled 被拒绝")

    try:
        dataclasses.replace(p, n0_mode="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 n0_mode 未拒绝")
    q = dataclasses.replace(p, n0_mode="continuous")
    assert "n0_mode" in problem_diff(p, q)
    print("  [PASS] n0_mode 校验与 problem_diff")
    print("OK _t_tier2_config")


if __name__ == "__main__":
    main()
