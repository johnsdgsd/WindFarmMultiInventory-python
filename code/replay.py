# replay.py — DG-CSER：诊断引导的跨阶段经验回放（docs/创新-跨阶段经验回放机制.md）。
"""SDDP 的割只在其锚点附近紧。前向采样一旦长期落在同一片状态区域，远处状态就
没有机会生成割，表现为某些阶段的割松弛 δ_t 长期卡住。

本模块在每个回放周期做四件事：
  1. 诊断：用本轮前向轨迹算 δ_t = c_t + θ_t − θ_{t−1}，并按未来成本构成归因；
  2. 取状态：从历史经验池取目标阶段的本地状态，以及同组其它阶段的跨阶段状态；
  3. 跨阶段准入：修复到目标阶段的可行域、按特征覆盖度去重、按新颖度/多样性选批；
  4. 重解：把选中的状态交给 engine.backward，用**当前** Q_{t+1} 重新生成割。

只搬运状态，不搬运割或动作。跨阶段状态在目标阶段的可行性与来源阶段无关（状态界
和 fleet 约束不随阶段变），重解得到的割和标准后向割一样是合法下界。
"""

from __future__ import annotations

import time

import numpy as np

from code.sim_analysis import TERMS, cost_terms


class CrossStageReplay:
    """DG-CSER 状态。由 run.py 创建并传给 SDDPEngine(replay=...)。"""

    def __init__(self, prob, cfg, *, log=print):
        self.prob = prob
        self.cfg = cfg
        self.log = log
        self.T = int(prob.T)
        self.Ly = prob.layout
        self.ub = np.asarray(self.Ly.upper(prob), dtype=float)
        self.N = np.asarray(prob.N, dtype=float)
        self.Imax_sum = float(np.asarray(prob.Imax, dtype=float).sum())
        self.Qmax_sum = float(np.asarray(prob.Qmax, dtype=float).sum())

        self.period = int(cfg.replay_period)
        self.k_local = int(cfg.replay_k_local)
        self.k_cross = int(cfg.replay_k_cross)
        self.quantile = float(cfg.replay_coverage_quantile)
        self.max_pool = int(cfg.replay_max_pool)
        self.diversity_eps = float(cfg.replay_diversity_eps)
        self.n_forward = int(cfg.n_forward_paths)
        self.hybrid_stages = {int(t) for t in cfg.replay_hybrid_stages}
        self.top_targets = int(cfg.replay_top_targets)
        self.hybrid_start = int(cfg.replay_hybrid_start)
        self.hybrid_keep_rounds = int(cfg.replay_hybrid_keep_rounds)

        self.groups = [[int(t) for t in g] for g in cfg.replay_groups]
        self.group_labels = [[str(x) for x in g] for g in cfg.replay_group_labels]
        if len(self.group_labels) != len(self.groups):
            raise ValueError("[replay] groups 与 group_labels 长度不一致")
        self.stage_group = [None] * (self.T + 1)
        self.stage_label = [() for _ in range(self.T + 1)]
        for gi, stages in enumerate(self.groups):
            for t in stages:
                if not 1 <= t < self.T:
                    raise ValueError(
                        f"[replay] 组内阶段必须落在 1..{self.T - 1}（末期无未来成本），"
                        f"实际 {t}")
                if self.stage_group[t] is not None:
                    raise ValueError(f"[replay] 阶段 {t} 同时出现在多个组")
                self.stage_group[t] = gi
                self.stage_label[t] = tuple(self.group_labels[gi])
        for t in self.hybrid_stages:
            if not 1 <= t <= self.T:
                raise ValueError(f"[replay] hybrid_stages 越界：{t}")

        # exp[t] 的元素 = (round_id, x, feature)；只保留最近 max_pool 条。
        self.exp = [[] for _ in range(self.T + 1)]
        self.active_targets = self._static_targets()
        self.tau = np.zeros(self.T + 1)
        self.delta = np.zeros(self.T + 1)
        self.delta_comp = np.zeros((self.T + 1, len(TERMS)))
        self.weights = np.zeros(self.T + 1)
        self.diag_labels = [[] for _ in range(self.T + 1)]
        self.rng = np.random.default_rng(int(cfg.seed) + 7919)
        self.runs = 0
        self.last_stats: dict = {}

    # ---------------------------------------------------------------- 引擎挂接
    def setup(self, engine) -> None:
        engine.hybrid_stages = set(self.hybrid_stages)
        engine.hybrid_start = self.hybrid_start
        engine.hybrid_keep_rounds = self.hybrid_keep_rounds

    def meta(self) -> dict:
        return {"enabled": True, "period": self.period,
                "k_local": self.k_local, "k_cross": self.k_cross,
                "coverage_quantile": self.quantile,
                "max_pool": self.max_pool,
                "diversity_eps": self.diversity_eps,
                "groups": self.groups, "group_labels": self.group_labels,
                "hybrid_stages": sorted(self.hybrid_stages),
                "hybrid_start": self.hybrid_start,
                "hybrid_keep_rounds": self.hybrid_keep_rounds,
                "top_targets": self.top_targets,
                "active_targets": list(self.active_targets),
                "runs": self.runs, "last": self.last_stats}

    # ---------------------------------------------------------------- 特征/修复
    def _feature(self, x) -> np.ndarray:
        """阶段无关的归一化特征：状态/上界 + fleet/失效/库存/订货/在途压力。"""
        x = np.asarray(x, dtype=float)
        z = x / np.where(self.ub > 0.0, self.ub, 1.0)
        n = self.Ly.n(x)
        fleet = n.sum(axis=1) / self.N
        scal = np.array([
            n.sum() / self.N.sum(),                 # 总运行台数占比
            1.0 - float(fleet.mean()),               # 平均停机/失效占比
            n[:, -1].sum() / self.N.sum(),           # 最差运行等级占比
            self.Ly.I(x).sum() / self.Imax_sum,      # 总库存压力
            self.Ly.q(x).sum() / self.Qmax_sum,      # 总在途订单压力
            self.Ly.o(x).sum() / (self.Qmax_sum * (self.Ly.L - 1)),
        ])
        return np.concatenate([z, scal])

    def _repair(self, x) -> np.ndarray:
        """把来源状态修到目标阶段的定义域：界内 + Σn ≤ N（跨阶段只依赖这些）。"""
        x = np.clip(np.asarray(x, dtype=float).copy(), 0.0, self.ub)
        n = self.Ly.n(x)                              # view，原地改 x
        s = n.sum(axis=1)
        over = s > self.N + 1e-12
        if over.any():
            n[over] *= (self.N[over] / s[over])[:, None]
        return x

    # ---------------------------------------------------------------- 经验池
    def _add_entry(self, t: int, rnd: int, x, feat=None) -> None:
        if feat is None:
            feat = self._feature(x)
        self.exp[t].append((int(rnd), np.asarray(x, dtype=float).copy(), feat))
        if len(self.exp[t]) > self.max_pool:
            del self.exp[t][:-self.max_pool]

    def _static_targets(self) -> list[int]:
        return sorted({t for g in self.groups for t in g})

    def _targets(self) -> list[int]:
        return list(self.active_targets)

    def _refresh_tau(self, t: int) -> None:
        """覆盖阈值 τ_t：目标阶段经验池最近邻距离的分位数。"""
        F = np.asarray([e[2] for e in self.exp[t]], dtype=float)
        if len(F) < 2:
            self.tau[t] = 0.0
            return
        if len(F) > 300:
            idx = self.rng.choice(len(F), size=300, replace=False)
            F = F[idx]
        sq = np.einsum("ij,ij->i", F, F)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (F @ F.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2)
        np.fill_diagonal(D, np.inf)
        self.tau[t] = float(np.quantile(D.min(axis=1), self.quantile))

    # ---------------------------------------------------------------- 诊断
    def _diagnose(self, results, lb_before: float, it: int) -> dict:
        T = self.T
        delta_sum = np.zeros(T + 1)
        comp_sum = np.zeros((T + 1, len(TERMS)))
        npath = len(results)
        if npath == 0:
            raise ValueError("回放诊断收到空的前向结果")

        for states, costs, _, records, _ in results:
            if records is None:
                raise RuntimeError("回放需要前向记录（engine.forward(record=True)）")
            terms = np.empty((T, len(TERMS)))
            for t in range(1, T + 1):
                terms[t - 1] = cost_terms(self.prob, self.Ly, states[t - 1],
                                          records[t - 1])
                self._add_entry(t, it, states[t - 1])
            for t in range(1, T + 1):
                th_t = float(records[t - 1].get("theta", 0.0))
                th_prev = (float(lb_before) if t == 1
                           else float(records[t - 2].get("theta", 0.0)))
                delta = float(costs[t - 1]) + th_t - th_prev
                fut = terms[t - 1:].sum(axis=0)
                ft = float(fut.sum())
                comp = delta * fut / ft if abs(ft) > 1e-12 else np.zeros(len(TERMS))
                delta_sum[t] += delta
                comp_sum[t] += comp

        self.delta = delta_sum / npath
        self.delta_comp = comp_sum / npath
        pos = np.maximum(self.delta, 0.0)
        den = float(pos.sum())
        self.weights = pos / den if den > 1e-12 else np.zeros_like(pos)
        for t in range(1, T + 1):
            v = np.maximum(self.delta_comp[t], 0.0)
            if v.sum() <= 1e-12:
                self.diag_labels[t] = []
            else:
                order = np.argsort(v)[::-1]
                self.diag_labels[t] = [TERMS[i] for i in order[:2]
                                       if v[i] >= 0.15 * v.sum()]

        order = [t for t in range(1, T) if self.weights[t] > 0]
        order.sort(key=lambda t: self.weights[t], reverse=True)
        # 静态组是成熟模型的诊断先验；当前 δ_t 的头部阶段额外做本地回放。
        # 非静态组阶段没有同组来源，只回放本地状态（_cross 返回空）。
        dynamic = order[:self.top_targets] if self.top_targets > 0 else []
        self.active_targets = sorted(set(self._static_targets()) | set(dynamic))
        for t in self.active_targets:
            self._refresh_tau(t)

        top = [{"stage": int(t), "weight": float(self.weights[t]),
                "delta": float(self.delta[t]),
                "labels": list(self.diag_labels[t])} for t in order[:6]]
        return {"top": top,
                "tau": {int(t): float(self.tau[t])
                        for t in self.active_targets}}

    # ---------------------------------------------------------------- 取状态
    def _local(self, t: int, it: int) -> list[np.ndarray]:
        """目标阶段历史本地状态（不含本轮；首轮不够时退回当前池）。"""
        hist = [e for e in self.exp[t] if e[0] < it]
        if not hist:
            hist = list(self.exp[t])
        k = min(self.k_local, len(hist))
        if k <= 0:
            return []
        idx = self.rng.choice(len(hist), size=k, replace=False)
        return [hist[int(i)][1].copy() for i in idx]

    def _cross(self, t: int, it: int) -> tuple[list[np.ndarray], dict]:
        """同组其它阶段的跨阶段状态：修复 → 覆盖度 → 标签 → 多样性。"""
        gi = self.stage_group[t]
        if gi is None or self.k_cross <= 0:
            return [], {"cand": 0, "accept": 0}
        srcs = [s for s in self.groups[gi] if s != t]
        # 严格历史：本轮之前的状态；本轮源阶段状态不回放到目标阶段。
        cand = [e for s in srcs for e in self.exp[s] if e[0] < it]
        if not cand:
            return [], {"cand": 0, "accept": 0}
        if len(cand) > 3000:
            idx = self.rng.choice(len(cand), size=3000, replace=False)
            cand = [cand[int(i)] for i in idx]

        Ftarget = np.asarray([e[2] for e in self.exp[t]], dtype=float)
        if len(Ftarget) == 0:
            return [], {"cand": len(cand), "accept": 0}
        xs, feats = [], []
        for e in cand:
            x = self._repair(e[1])
            xs.append(x)
            feats.append(self._feature(x))
        C = np.asarray(feats, dtype=float)
        sqc = np.einsum("ij,ij->i", C, C)
        sqt = np.einsum("ij,ij->i", Ftarget, Ftarget)
        D2 = sqc[:, None] + sqt[None, :] - 2.0 * (C @ Ftarget.T)
        np.maximum(D2, 0.0, out=D2)
        dmin = np.sqrt(D2.min(axis=1))
        tau = float(self.tau[t])
        ok = dmin > tau + 1e-12

        # 标签匹配：目标阶段与来源阶段同组，成本标签天然重合；仍显式检查，
        # 防止未来把不同标签的阶段混进同一组。
        target_lab = set(self.stage_label[t])
        source_ok = np.ones(len(cand), dtype=bool)
        for i, e in enumerate(cand):
            # 候选来自哪个阶段，只有其组标签可用；用组标签做匹配。
            source_ok[i] = bool(target_lab & set(self.group_labels[gi]))

        sel: list[int] = []
        for i in np.argsort(-dmin):
            i = int(i)
            if not ok[i] or not source_ok[i]:
                continue
            f = C[i]
            if any(float(np.linalg.norm(f - C[j])) <= self.diversity_eps
                   for j in sel):
                continue
            sel.append(i)
            if len(sel) >= self.k_cross:
                break
        return [xs[i] for i in sel], {"cand": len(cand), "accept": len(sel)}

    def _build_batch(self, it: int):
        batch = [[] for _ in range(self.T + 1)]
        cross_states = [[] for _ in range(self.T + 1)]
        n_local = n_cross = n_cand = n_accept = 0
        for t in self._targets():
            loc = self._local(t, it)
            cr, st = self._cross(t, it)
            batch[t] = loc + cr
            cross_states[t] = cr
            n_local += len(loc)
            n_cross += len(cr)
            n_cand += int(st["cand"])
            n_accept += int(st["accept"])
        return batch, cross_states, {"local": n_local, "cross": n_cross,
                                     "cand": n_cand, "accept": n_accept}

    # ---------------------------------------------------------------- 主流程
    def run(self, engine, results, lb_before: float, it: int) -> dict:
        t0 = time.time()
        diag = self._diagnose(results, lb_before, it)
        batch, cross_states, bstat = self._build_batch(it)
        n_points = int(sum(len(x) for x in batch))
        n_added = 0
        if n_points:
            n_added = int(engine.backward(batch, it=it, pass_tag="replay"))
        # 跨阶段状态进入目标阶段经验池，后续覆盖度/本地批可复用；本地状态已在池中。
        for t in self._targets():
            for x in cross_states[t]:
                self._add_entry(t, it, x)
        dt = time.time() - t0
        stats = {"replay_cuts": n_added, "replay_points": n_points,
                 "replay_local": int(bstat["local"]),
                 "replay_cross": int(bstat["cross"]),
                 "replay_cand": int(bstat["cand"]),
                 "replay_accept": int(bstat["accept"]),
                 "replay_time": float(dt),
                 "replay_targets": int(len(self._targets())),
                 "replay_top": diag["top"],
                 "replay_tau": diag["tau"]}
        self.runs += 1
        self.last_stats = stats
        top = ", ".join(f"{x['stage']}:{x['weight']:.2f}" for x in diag["top"][:3])
        self.log(f"  [回放 {it}] 高松弛 {top or '无'} | 点 {n_points}"
                 f"（本地 {bstat['local']} / 跨 {bstat['cross']}，"
                 f"候选 {bstat['cand']} → 接受 {bstat['accept']}）"
                 f" | 新增割 {n_added} | {dt:.1f} s")
        return stats
