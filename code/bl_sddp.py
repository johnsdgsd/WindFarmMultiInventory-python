# bl_sddp.py — Batch Learning SDDP（docs/BL-SDDP.md）的本项目实现。
"""后向 pass 的经验回放：

  · 每个阶段只保留本阶段的 trial 状态（阶段库状态，无需奖励/完整转移）；
  · 每 Z 轮从历史 trial 状态中取一批 K 个，用**当前** cuts[t+1] 重解并重建割；
  · 回放严格在本阶段内，不做跨阶段共享；
  · 支撑超平面更新只需状态；这里额外记录生成时 Q(x) 以支持论文的 B 规则
    （选择 tightest 的割：V(x) − Q(x) 最小，即当前包络与新割最接近）。

两种模式（cfg.bl_mode）：
  after   —— 论文原意：标准后向（当前试探点）之后再对历史点做一次后向；
  unified —— 当前试探点 + 历史点合并为一批，只做一次后向（省一次标准后向）。

回放割仍走正常 Benders 构造和 LML1 竞争；合法性不改变。
"""

from __future__ import annotations

import time

import numpy as np


class BatchLearning:
    def __init__(self, prob, cfg, *, log=print, rng=None):
        self.T = int(prob.T)
        self.mode = str(cfg.bl_mode)
        self.period = int(cfg.bl_period)
        self.batch = int(cfg.bl_batch)      # 0 = full batch
        self.selection = str(cfg.bl_selection)
        self.memory = int(cfg.bl_memory)
        self.rng = rng if rng is not None else np.random.default_rng(
            int(cfg.seed) + 104729)
        self.log = log
        # M[t] = [(round, state, Q_at_insert), ...]
        self.M = [[] for _ in range(self.T + 1)]
        self.runs = 0
        self.last_stats = {}

    def meta(self) -> dict:
        return {"enabled": True, "mode": self.mode, "period": self.period,
                "batch": self.batch, "selection": self.selection,
                "memory": self.memory,
                "sizes": [len(x) for x in self.M],
                "runs": self.runs, "last": self.last_stats}

    @staticmethod
    def _key(x) -> bytes:
        return np.asarray(x, dtype=float).tobytes()

    def remember(self, trial, it: int, cuts) -> None:
        """在标准后向之后记录本轮 trial 状态及其新割的 Q(x)。"""
        for t in range(1, self.T + 1):
            pts = trial[t]
            if not pts:
                continue
            new = [c for c in cuts[t] if c.it == it]
            qmap = {self._key(c.state): float(c.obj) for c in new}
            for x in pts:
                x = np.asarray(x, dtype=float).copy()
                q = qmap.get(self._key(x))
                if q is None:
                    # new cut was filtered/absent: use current envelope value
                    q = max(float(c.alpha + c.beta @ x) for c in cuts[t])
                self.M[t].append((int(it), x, float(q)))
            if len(self.M[t]) > self.memory:
                del self.M[t][:-self.memory]

    def _envelope(self, cuts, x) -> float:
        return max(float(c.alpha + c.beta @ x) for c in cuts)

    def _select(self, t: int, it: int, cuts):
        cand = [e for e in self.M[t] if e[0] < it]
        if not cand:
            cand = list(self.M[t])
        if not cand:
            return []
        if self.batch <= 0:
            chosen = cand
        elif self.selection == "recent":
            chosen = cand[-self.batch:]
        elif self.selection == "tight":
            # V(x_i) − Q_i 越小越“紧”；与论文 B 规则一致
            # 论文 R2/B：V(x_i) − Q_i 越小越“紧”，优先回放
            scored = []
            for e in cand:
                x = e[1]
                scored.append((self._envelope(cuts, x) - e[2], e))
            scored.sort(key=lambda z: z[0])
            chosen = [e for _, e in scored[:self.batch]]
        else:
            idx = self.rng.choice(len(cand), size=min(self.batch, len(cand)),
                                  replace=False)
            chosen = [cand[int(i)] for i in idx]
        # 同一个状态只回放一次（阶段 1 的 trial 全相同）
        out, seen = [], set()
        for _, x, _ in chosen:
            key = self._key(np.round(x, 9))
            if key in seen:
                continue
            seen.add(key)
            out.append(np.asarray(x, dtype=float).copy())
        return out

    def replay(self, engine, it: int, current=None) -> dict:
        t0 = time.time()
        batch = [[] for _ in range(self.T + 1)]
        n_states = 0
        for t in range(1, self.T + 1):
            pts = self._select(t, it, engine.cuts[t])
            if current is not None:
                pts = list(current[t]) + pts
            # one replay per state at this stage
            uniq, seen = [], set()
            for x in pts:
                key = np.round(np.asarray(x, dtype=float), 9).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(np.asarray(x, dtype=float).copy())
            batch[t] = uniq
            n_states += len(uniq)
        n_added = 0
        if n_states:
            saved = engine.round_id
            try:
                # 与标准后向使用不同的 worker 模型键，强制用当前割重新装配
                engine.round_id = int(it) + 10_000_000
                n_added = int(engine.backward(batch, it=it))
            finally:
                engine.round_id = saved
        dt = time.time() - t0
        stats = {"bl_cuts": n_added, "bl_states": int(n_states),
                 "bl_time": float(dt), "bl_mem": int(sum(map(len, self.M)))}
        self.runs += 1
        self.last_stats = stats
        self.log(f"  [BL-SDDP {it}] 回放状态 {n_states} -> 割 {n_added}  "
                 f"({dt:.1f} s)")
        return stats
