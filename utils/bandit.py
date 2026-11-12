# bandit.py — SDDP.jl 风格的双臂 BanditDuality 控制器。
"""只在两个对偶处理器之间选择：continuous / strengthened。

控制器本身不生成割；主引擎按 current_arm 调度：

  · continuous  -> 普通 LP 割
  · strengthened -> 当前 cut_lift（release_integrality=true 时对标
                    SDDP.jl StrengthenedConicDuality）

奖励/切换规则尽量逐项复刻 SDDP.jl 1.13：
  reward = |bound_t - bound_{t-1}| / max(time_t - time_{t-1}, 0.1)
  no-progress (atol=1e-6, rtol=sqrt(eps)) -> 切下一个臂
  否则 score=mean+std(ddof=1)，按 softmax 采样
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class _Arm:
    name: str
    rewards: list[float] = field(default_factory=list)


class BanditDuality:
    """双臂 bandit；接口按“每次完成一个 iteration”调用。"""

    def __init__(self, arms: Sequence[str] = ("continuous", "strengthened"),
                 *, seed: int = 2026, atol: float = 1e-6,
                 rtol: float | None = None):
        if len(arms) < 2:
            raise ValueError("BanditDuality 至少需要两个臂")
        self.arm_names = tuple(str(a) for a in arms)
        self.arms = [_Arm(name) for name in self.arm_names]
        self.last_arm_index = 0
        self.current_arm = self.arm_names[0]
        self._bounds: list[float] = []
        self._times: list[float] = []
        self._used_arms: list[int] = []
        self.atol = float(atol)
        self.rtol = (math.sqrt(np.finfo(float).eps) if rtol is None
                     else float(rtol))
        self.rng = np.random.default_rng(int(seed))
        self.history: list[dict] = []

    # ------------------------------------------------------------------ 选择
    def _scores(self) -> list[float]:
        scores = []
        for arm in self.arms:
            n = len(arm.rewards)
            if n == 0:
                scores.append(float("nan"))
            elif n < 2:
                scores.append(float(np.mean(arm.rewards)))
            else:
                mu = float(np.mean(arm.rewards))
                sd = float(np.std(arm.rewards, ddof=1))
                scores.append(mu if np.isnan(sd) else mu + sd)
        return scores

    def _choose(self) -> int:
        scores = self._scores()
        nan_idx = [i for i, s in enumerate(scores) if np.isnan(s)]
        if nan_idx:
            return int(nan_idx[int(self.rng.integers(0, len(nan_idx)))])
        z = np.exp(np.asarray(scores) - max(scores))
        z /= z.sum()
        r = float(self.rng.random())
        acc = 0.0
        for i, p in enumerate(z):
            acc += float(p)
            if r <= acc:
                return i
        return len(z) - 1

    def _is_no_progress(self, a: float, b: float) -> bool:
        return (abs(a - b)
                <= self.atol + self.rtol * max(abs(a), abs(b)))

    # ------------------------------------------------------------------ 记录
    def observe(self, bound: float, elapsed: float,
                arm_name: str | None = None) -> dict:
        """记录一个完成的 iteration，更新奖励并选择下一次的臂。

        异步引擎里 iteration 的完成顺序不一定等于派发顺序；这里按完成
        顺序记录，奖励记到“上一个完成 iteration 所用臂”上。这与 SDDP.jl
        的顺序执行版本在单 worker/顺序完成时一致。
        """
        arm_index = (self.arm_names.index(arm_name)
                     if arm_name in self.arm_names else self.last_arm_index)
        self._bounds.append(float(bound))
        self._times.append(float(elapsed))
        self._used_arms.append(int(arm_index))
        reward = None
        if len(self._bounds) >= 2:
            dt = max(self._times[-1] - self._times[-2], 0.1)
            reward = abs(self._bounds[-1] - self._bounds[-2]) / dt
            # SDDP.jl 在下一轮 backward 开头奖励上一轮用的臂。
            prev_arm = self._used_arms[-2]
            self.arms[prev_arm].rewards.append(float(reward))
            if self._is_no_progress(self._bounds[-1], self._bounds[-2]):
                self.last_arm_index = (self.last_arm_index + 1) % len(self.arms)
                switch_reason = "no_progress"
            else:
                self.last_arm_index = self._choose()
                switch_reason = "softmax"
        else:
            # SDDP.jl 初始 logs_seen=1：第二轮先不更新，仍用第一个臂。
            switch_reason = "warmup"
        self.current_arm = self.arm_names[self.last_arm_index]
        rec = {
            "bound": float(bound), "time": float(elapsed),
            "arm_used": self.arm_names[arm_index],
            "next_arm": self.current_arm,
            "reward": reward,
            "reason": switch_reason,
            "scores": dict(zip(self.arm_names, self._scores())),
            "n_rewards": [len(a.rewards) for a in self.arms],
        }
        self.history.append(rec)
        return rec

    @classmethod
    def from_meta(cls, meta: dict, *, seed: int = 2026) -> "BanditDuality":
        """从 meta()/cuts.json 恢复控制器（用于失败后续训）。"""
        m = dict(meta or {})
        obj = cls(m.get("arms") or ("continuous", "strengthened"),
                  seed=seed, atol=float(m.get("atol", 1e-6)),
                  rtol=m.get("rtol"))
        reward_history = m.get("reward_history") or []
        for arm, rewards in zip(obj.arms, reward_history):
            arm.rewards[:] = [float(x) for x in rewards]
        obj.last_arm_index = int(m.get("last_arm_index", 0))
        obj.current_arm = str(m.get("current_arm", obj.arm_names[0]))
        obj.history = list(m.get("history") or [])
        obj._bounds = [float(h["bound"]) for h in obj.history]
        obj._times = [float(h["time"]) for h in obj.history]
        obj._used_arms = [
            obj.arm_names.index(h["arm_used"])
            if h.get("arm_used") in obj.arm_names else obj.last_arm_index
            for h in obj.history
        ]
        return obj

    def reset(self) -> None:
        for arm in self.arms:
            arm.rewards.clear()
        self.last_arm_index = 0
        self.current_arm = self.arm_names[0]
        self._bounds.clear()
        self._times.clear()
        self._used_arms.clear()
        self.history.clear()

    def meta(self) -> dict:
        return {
            "arms": list(self.arm_names),
            "current_arm": self.current_arm,
            "last_arm_index": int(self.last_arm_index),
            "atol": self.atol,
            "rtol": self.rtol,
            "n_observations": len(self.history),
            "reward_counts": [len(a.rewards) for a in self.arms],
            "reward_history": [list(a.rewards) for a in self.arms],
            "history": list(self.history),
        }
