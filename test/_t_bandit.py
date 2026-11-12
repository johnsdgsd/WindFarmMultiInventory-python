# _t_bandit.py — BanditDuality 控制器与 SDDP.jl 规则的单元自检。
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.bandit import BanditDuality  # noqa: E402


def main():
    b = BanditDuality(seed=7)
    assert b.current_arm == "continuous"

    r1 = b.observe(100.0, 1.0, "continuous")
    assert r1["arm_used"] == "continuous"
    assert r1["next_arm"] == "continuous"
    assert r1["reward"] is None
    assert b.arms[0].rewards == []

    # 与 SDDP.jl 一样：第一轮不更新奖励；第二轮无进展时切到下一个臂。
    r2 = b.observe(100.0, 3.0, "continuous")
    assert abs(r2["reward"] - 0.0) < 1e-15
    assert r2["reason"] == "no_progress"
    assert r2["next_arm"] == "strengthened"
    assert len(b.arms[0].rewards) == 1

    # 增强臂第一次使用后，奖励仍记在上一轮臂；该臂尚未有观测时，
    # SDDP.jl 会在未观测臂中随机选择（只有 strengthened）。
    r3 = b.observe(120.0, 6.0, "strengthened")
    assert abs(r3["reward"] - 20.0 / 3.0) < 1e-12
    assert r3["next_arm"] == "strengthened"
    assert b.arms[1].rewards == []

    # 下一轮完成增强臂后，奖励归到 strengthened。
    r4 = b.observe(130.0, 9.0, "strengthened")
    assert abs(r4["reward"] - 10.0 / 3.0) < 1e-12
    assert len(b.arms[1].rewards) == 1

    b.reset()
    assert b.current_arm == "continuous"
    assert all(len(a.rewards) == 0 for a in b.arms)
    print("BANDIT_OK")


if __name__ == "__main__":
    main()
