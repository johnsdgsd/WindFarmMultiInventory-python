# cut_pool.py — SDDP.jl 风格的割池：缓存全部割 + non-dominated count + 复活。
"""本模块只做“割选择”，不改割的数学定义。

与 LML1 的核心区别：
  · 所有发现过的割都保留在 `all_cuts` 缓存里；
  · 每条割有 `non_dominated_count`，表示它当前支配多少个历史采样状态；
  · count<1 的割从活跃池移除（不再进 LP），但对象不删除；
  · 一个已移除的旧割如果在新采样状态上重新占优，会被加回活跃池；
  · 只有 `cut_deletion_minimum` 条待删除割凑齐时，才真正从 LP 移除。

Julia SDDP.jl 的实现见：
  SDDP.jl/src/plugins/bellman_functions.jl::_cut_selection_update
本类是它的单目标、单 belief、最小化版本；不改变割的截距/斜率。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cut import Cut


@dataclass
class _SampledState:
    state: np.ndarray
    dominating: Cut
    best: float


@dataclass
class StageCutPool:
    stage: int
    deletion_minimum: int = 1
    all_cuts: list[Cut] = field(default_factory=list)
    active: list[Cut] = field(default_factory=list)
    sampled: list[_SampledState] = field(default_factory=list)
    pending: set[int] = field(default_factory=set)
    _active_ids: set[int] = field(default_factory=set)
    _next_uid: int = 0

    def _uid(self, cut: Cut) -> int:
        uid = getattr(cut, "uid", None)
        if uid is None:
            uid = ("pool", self.stage, self._next_uid)
            self._next_uid += 1
            cut.uid = uid
        return uid

    def seed(self, cuts) -> None:
        """初始化平凡割：保留在活跃池，但不创建采样状态。"""
        for c in cuts:
            self._uid(c)
            c.non_dominated_count = 1
            self.all_cuts.append(c)
            self.active.append(c)
            self._active_ids.add(id(c))

    def active_cuts(self) -> list[Cut]:
        return self.active

    def _deactivate(self, cut: Cut, out: list[Cut]) -> None:
        if id(cut) in self._active_ids:
            self._active_ids.discard(id(cut))
            self.active = [c for c in self.active if c is not cut]
            self.pending.discard(self._uid(cut))
            cut.non_dominated_count = 0
            out.append(cut)

    def _activate(self, cut: Cut, out: list[Cut]) -> None:
        if id(cut) not in self._active_ids:
            self._active_ids.add(id(cut))
            self.active.append(cut)
            self.pending.discard(self._uid(cut))
            out.append(cut)

    def add(self, cut: Cut) -> tuple[list[Cut], list[Cut]]:
        """加入一条新割，返回 (本步新激活, 本步被移出活跃池) 的割列表。"""
        uid = self._uid(cut)
        cut.non_dominated_count = 1
        activated: list[Cut] = []
        deactivated: list[Cut] = []

        # 新割先在自己的采样点上成为最优割。
        s_new = _SampledState(np.asarray(cut.state, dtype=float).copy(),
                              cut, float(cut.value_at(cut.state)))

        # 1) 新割如果支配旧采样点，就接管这些点的最优割。
        for old in self.sampled:
            h = float(cut.value_at(old.state))
            if h >= old.best:
                old.dominating.non_dominated_count -= 1
                cut.non_dominated_count += 1
                old.dominating = cut
                old.best = h
        self.sampled.append(s_new)

        # 2) 旧的不活跃割若在新采样点上占优，则复活。
        for old in self.all_cuts:
            if id(old) in self._active_ids:
                continue
            h = float(old.value_at(s_new.state))
            if h >= s_new.best:
                s_new.dominating.non_dominated_count -= 1
                old.non_dominated_count += 1
                s_new.dominating = old
                s_new.best = h
                self._activate(old, activated)

        # 3) 新割加入缓存/活跃池。
        self.all_cuts.append(cut)
        self._activate(cut, activated)

        # 4) count<1 的活跃割进入待删除队列；凑够阈值再统一移出 LP。
        for c in self.all_cuts:
            if id(c) in self._active_ids and c.non_dominated_count < 1:
                self.pending.add(self._uid(c))
            elif c.non_dominated_count >= 1:
                self.pending.discard(self._uid(c))

        if len(self.pending) >= max(1, int(self.deletion_minimum)):
            by_uid = {self._uid(c): c for c in self.all_cuts}
            todo = [by_uid[u] for u in self.pending if u in by_uid]
            self.pending.clear()
            for c in todo:
                self._deactivate(c, deactivated)
        dead = {id(c) for c in deactivated}
        activated = [c for c in activated if id(c) not in dead]
        return activated, deactivated


class CutPool:
    """按 stage 管理的 SDDP.jl 风格割池。"""

    def __init__(self, T: int, deletion_minimum: int = 1):
        self.T = int(T)
        self.pools = {t: StageCutPool(t, int(deletion_minimum))
                      for t in range(1, self.T + 1)}

    def seed(self, t: int, cuts) -> None:
        self.pools[t].seed(cuts)

    def add(self, t: int, cut: Cut):
        return self.pools[t].add(cut)

    def active(self, t: int) -> list[Cut]:
        return self.pools[t].active_cuts()

    def all_cuts(self, t: int) -> list[Cut]:
        return self.pools[t].all_cuts

    def all_by_stage(self) -> list[list[Cut]]:
        out = [[] for _ in range(self.T + 2)]
        for t in range(1, self.T + 1):
            out[t] = list(self.all_cuts(t))
        return out

    def active_by_stage(self) -> list[list[Cut]]:
        out = [[] for _ in range(self.T + 2)]
        for t in range(1, self.T + 1):
            out[t] = self.active(t)
        return out

    def counts(self) -> tuple[int, int]:
        active = sum(len(p.active) for p in self.pools.values())
        total = sum(len(p.all_cuts) for p in self.pools.values())
        return active, total
