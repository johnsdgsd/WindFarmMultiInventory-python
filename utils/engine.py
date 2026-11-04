# engine.py — SDDP 引擎：前向并行、后向阶段串行、割池、下界与训练循环。与问题解耦。
"""算法骨架（阶段 0 = 虚拟首阶段，1..T 为真实周期；cuts[t] 近似 V_t）：

    每轮：
      (a) 前向 —— 采样 N 条路径，每路径独立走完 T+1 期解 LP 得策略，
          收齐试探点 trial[t] 与前向总成本（上界估计）；
      (b) 后向 —— **按阶段串行，期内并行**：t 从 T 递减到 1，把该阶段的 N 个试探点
          平均分给 worker；single 对一个试探点枚举场景后平均成一条割，multi 保留每个
          场景一条割并绑定到对应 theta；本阶段所有割算完再一起入库，然后才推进到 t−1；
      (c) 下界 —— 阶段 0 子问题（割口径与后向一致）。

**为什么必须"期间串行"**：旧实现轮初打一份割快照、全部 (期 × 试探点) 任务一次铺满，
于是期 t 用不上同轮期 t+1 的新割 —— 终端信息每轮最多向前传播一期，T=200 时下界上百轮
不动（实测 13 轮全 0）。改成期间串行后，期 t 立即用上同轮期 t+1 的新割，一次反向扫描
就把信息传遍全视界。

**并行度来自试探点数**：n_forward_paths 默认 = 进程数 × 1 ⇒ 每个阶段恰好 N = 进程数 个
试探点 ⇒ 每个 worker 每阶段正好产出 1 条割，无空转。
"""

from __future__ import annotations

import csv
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .cut import Cut, trivial_cut
from .io import TrainingLog, save_cuts
from .partition import (cluster_dispersion, cluster_weights, fixed_partition,
                        greedy_split, median_pairwise_distance,
                        normalized_cluster_score, resolve_theta_cap,
                        split_cluster, stall_trigger, validate_partition)
from .payload import StagePayload, ThetaSpec, put_cuts
from .worker import (backward_cluster_multicut_task, backward_multicut_task,
                     backward_point_task, forward_path_task, init_worker,
                     lower_bound_task, scene_signatures_task)


def _default_ctx():
    return mp.get_context("spawn")


class SDDPEngine:
    """问题无关的 SDDP 引擎。

    参数
    ----
    prob          : 问题数据；须有 `.pool`（等概率场景序列）与 `.state_dim`
    make_sub      : 模块顶层工厂 (t, prob, cfg) -> 子问题对象
    cfg           : SolverConfig（进程数/线程数/容差/前向路径数/后向场景数）
    results_dir   : 结果目录（求解失败时的现场 LP 落这里）
    """

    def __init__(self, prob, make_sub, cfg, *, results_dir=None, log=print):
        self.prob = prob
        self.make_sub = make_sub
        self.cfg = cfg
        self.T = int(prob.T)
        self.state_dim = int(prob.state_dim)
        self.initial_state = np.round(
            np.asarray(prob.initial_state(), dtype=float).ravel(), 6)

        self.seed = int(cfg.seed)
        self.rng = np.random.default_rng(self.seed)
        self.n_processes = int(cfg.n_processes)
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.checkpoint_interval = float(cfg.checkpoint_interval)
        self.max_train_seconds = float(getattr(cfg, "max_train_seconds", 0.0))
        self.log = log if log is not None else (lambda *a, **k: None)

        self.n_scenes = len(prob.pool)
        self.mode = str(getattr(cfg, "mode", "single"))
        self.refine = getattr(cfg, "refine", None)
        self.refine_enabled = bool(getattr(self.refine, "enabled", False))
        self._use_shm = str(getattr(cfg, "payload_mode", "pickle")) == "shm"
        mc = getattr(cfg, "multicut", None)
        self.group_mode = "scene" if mc is None else str(mc.group_mode)
        self.n_clusters_cfg = 0 if mc is None else int(mc.n_clusters)
        if self.mode == "single":
            base_parts = [list(range(self.n_scenes))]
            base_K = 1
            self.grouped = False
        elif self.n_clusters_cfg == 0:
            base_parts = [[i] for i in range(self.n_scenes)]
            base_K = self.n_scenes
            self.grouped = False
        else:
            base_parts = fixed_partition(prob.pool, self.n_scenes,
                                         self.n_clusters_cfg, self.group_mode)
            base_K = len(base_parts)
            self.grouped = True
        refine_cap = 0
        configured = 0
        if self.refine_enabled:
            refine_cap = int(self.refine.max_clusters_per_stage)
            configured = int(self.refine.theta_cap)
        self.theta_cap = resolve_theta_cap(
            self.mode, self.n_clusters_cfg, self.n_scenes, refine_cap, configured)
        if self.theta_cap < base_K:
            raise ValueError(f"theta_cap={self.theta_cap} < base K={base_K}")
        # 每阶段一份独立副本；细化只改一个阶段，不影响其他阶段。
        self.parts = [[]] + [ [list(c) for c in base_parts] for _ in range(self.T) ]
        self.K = [0] + [base_K] * self.T + [0]
        self.weights = [np.zeros(0)] + [
            cluster_weights(base_parts, self.n_scenes) if self.mode != "single"
            else np.array([1.0]) for _ in range(self.T)
        ] + [np.zeros(0)]
        self.layout_gen = [0] * (self.T + 2)
        self.cut_gen = [0] * (self.T + 2)
        # 兼容别名：旧测试/日志看全局 L。
        self.clusters = base_parts
        self.cluster_weights = self.weights[1]
        self.n_theta = self.theta_cap
        _k = int(getattr(cfg, "n_backward_samples", 0))
        if 0 < _k < self.n_scenes:
            self.log(
                f"[警告] 后向只抽 {_k}/{self.n_scenes} 个场景：割只对该子样本子问题成立，"
                f"LB 可能高于全池 UB；要全池有效 LB 请把 n_backward_samples 设为 0")
        if self.mode == "multi" and _k != 0:
            raise ValueError("mode=multi 要求 n_backward_samples=0（全池枚举）")
        self.cuts: list[list[Cut]] = [[] for _ in range(self.T + 1)]
        for t in range(1, self.T + 1):
            self.cuts[t] = [
                trivial_cut(t, self.initial_state, self.state_dim, group=g)
                for g in range(self.K[t])
            ]

        self.n_solves = 0
        self.n_signature_solves = 0
        self.n_cut_dropped = 0
        self.lb_best = 0.0          # 历史最大合法 LB（LML1 删割时当前 LB 可能下降）
        self.round_id = 0
        self.last_iter = 0          # 最后一个完成的轮次（续训据此定起点）
        self.on_iter = None
        # 割筛选（docs/LML1割筛选算法逻辑.md）：每期维护历史试探点，以及每个点上
        # 「目前取值最大的、且生成最早」的那条割。未被任何点选中的割直接删除。
        self.cut_selection = str(getattr(cfg, "cut_selection", "none"))
        self.cut_select_eps = float(getattr(cfg, "cut_select_eps", 1e-6))
        self.cut_select_keep_rounds = int(getattr(cfg, "cut_select_keep_rounds", 0))
        self._pts: list[list] = [[] for _ in range(self.T + 1)]   # 历史试探点（single）
        self._mx: list[list] = [[] for _ in range(self.T + 1)]    # 该点处当前最大割值
        self._rc: list[list] = [[] for _ in range(self.T + 1)]    # 保留的最旧最大割
        # multi：每个 (阶段, theta) 独立维护 LML1 历史；长度随 K[t] 变化。
        self._pts_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        self._mx_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        self._rc_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        # 细化状态
        self._lb_hist: list[float] = []
        self.n_refines = 0
        self.n_attempts = 0
        self.n_noops = 0
        self.n_ineffective = 0
        self.last_refine_iter = 0
        self.last_attempt_iter = 0
        self.refine_events: list[dict] = []
        self._refine_rng = np.random.default_rng(self.seed + 10000)
        self._last_refine_lb_before = 0.0
        self.refine_log = None
        self._pool = None
        if self.n_processes > 1:
            self._pool = _default_ctx().Pool(
                self.n_processes, initializer=init_worker,
                initargs=(prob, make_sub, cfg, self.T, self.initial_state,
                          self.results_dir, self.theta_cap))
        else:
            init_worker(prob, make_sub, cfg, self.T, self.initial_state,
                        self.results_dir, self.theta_cap)

    # ---------------------------------------------------------------- 调度
    def _map(self, fn, tasks):
        if not tasks:
            return []
        if self._pool is None:
            return [fn(t) for t in tasks]
        # 任务粒度较粗（后向一个试探点要解完它全部场景）→ 小块以摊平负载
        chunksize = max(1, len(tasks) // (2 * self.n_processes))
        return self._pool.map(fn, tasks, chunksize=chunksize)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ 前向 pass
    def sample_scene_paths(self, n_paths: int) -> list[list[int]]:
        return [self.rng.integers(0, self.n_scenes, size=self.T).tolist()
                for _ in range(n_paths)]

    def theta_spec(self, t: int) -> ThetaSpec:
        """stage-t 子问题消费 cuts[t+1]；t=T 用 active_k=0。"""
        if t >= self.T:
            return ThetaSpec(0, (), 0, 0)
        s = t + 1
        return ThetaSpec(int(self.K[s]),
                         tuple(float(v) for v in self.weights[s]),
                         int(self.layout_gen[s]),
                         int(self.cut_gen[s]))

    def stage_specs(self) -> list[ThetaSpec]:
        return [self.theta_spec(t) for t in range(self.T + 1)]

    def forward(self, n_paths: int, *, record: bool = False):
        """采样路径并求解。返回 (trial, results)。

        trial[t] = 第 t 期入库状态列表（即该阶段的全部试探点）；
        results[k] = (states, costs, total, records)。
        """
        if n_paths < 1:
            raise ValueError("n_paths 至少为 1")
        scene_paths = self.sample_scene_paths(n_paths)
        results = self.forward_paths(scene_paths, record=record)
        trial = [[] for _ in range(self.T + 1)]
        for states, _, _, _, _ in results:
            for t in range(1, self.T + 1):
                trial[t].append(states[t - 1])
        return trial, results

    def forward_paths(self, scene_paths, *, record: bool = False):
        """按给定场景路径跑前向。前向要走遍所有阶段，故整轮持有每阶段的割块，
        map 结束后统一释放。"""
        blocks = self.forward_refs()
        refs = [b.ref for b in blocks]
        specs = self.stage_specs()
        try:
            tasks = [(self.round_id, k, list(sp), refs, specs, record)
                     for k, sp in enumerate(scene_paths)]
            results = self._map(forward_path_task, tasks)
        finally:
            for b in blocks:
                b.close()
        self.n_solves += len(scene_paths) * (self.T + 1)
        return results

    # ------------------------------------------------------------ 后向 pass
    def backward(self, trial, it: int) -> int:
        """按阶段串行（t = T..1），期内按试探点并行。返回新增割数。

        每个阶段：本阶段全部试探点的割算完，一起入库，再推进到 t−1 —— 这样期 t−1
        用到的 cuts[t] 已含本轮新割。
        """
        n_added = 0
        for t in range(self.T, 0, -1):
            n_added += self._backward_stage(t, trial[t], it)
        return n_added

    def _backward_stage(self, t: int, points, it: int, *,
                        pass_tag: str = "bwd", clusters=None) -> int:
        """单阶段后向；细化后的立即重建也复用它。"""
        if not points:
            return 0
        parts = self.parts[t] if clusters is None else clusters
        if len(parts) != self.K[t]:
            raise ValueError(f"stage {t}: clusters {len(parts)} ≠ K[t]={self.K[t]}")
        blk = put_cuts(self.cuts[t + 1] if t < self.T else [], self.state_dim,
                       use_shm=self._use_shm)

        spec = self.theta_spec(t)
        try:
            if self.grouped:
                tasks = [(self.round_id, t, i, np.asarray(x, dtype=float),
                          parts, blk.ref, spec) for i, x in enumerate(points)]
                results = self._map(backward_cluster_multicut_task, tasks)
            elif self.mode == "multi":
                inds = list(range(self.n_scenes))
                tasks = [(self.round_id, t, i, np.asarray(x, dtype=float), inds,
                          blk.ref, spec) for i, x in enumerate(points)]
                results = self._map(backward_multicut_task, tasks)
            else:
                inds = self._sample_scenes(len(points))
                tasks = [(self.round_id, t, i, np.asarray(x, dtype=float),
                          inds[i], blk.ref, spec) for i, x in enumerate(points)]
                results = self._map(backward_point_task, tasks)
        finally:
            blk.close()
        new_cuts = []
        K = len(parts) if self.mode == "multi" else 1
        if self.mode == "multi":
            self.n_solves += len(points) * (K if self.grouped else self.n_scenes)
            for (_, _, _, x, _, _, _), (betas, objs) in zip(tasks, results):
                kk = K if self.grouped else self.n_scenes
                for k in range(kk):
                    beta = betas[k]
                    obj = float(objs[k])
                    c = Cut(float(obj) - float(beta @ x), beta, x, t,
                            obj, it, group=k)
                    self.cuts[t].append(c)
                    new_cuts.append(c)
            if self.cut_selection == "lml1":
                self._lml1_update_multi(t, points, new_cuts)
        else:
            self.n_solves += sum(len(s) for s in inds)
            for (_, _, _, x, _, _, _), (beta, obj) in zip(tasks, results):
                c = Cut(float(obj) - float(beta @ x), beta, x, t, float(obj), it)
                self.cuts[t].append(c)
                new_cuts.append(c)
            if self.cut_selection == "lml1":
                self._lml1_update(t, points, new_cuts)
        self.cut_gen[t] += 1
        return len(new_cuts)

    # ------------------------------------------------------------ LML1 割筛选
    def restore_cut_history(self) -> int:
        """从已加载的 cuts 重建 LML1 的试探点历史。

        cuts.json 只保存 Cut，不保存 _pts/_mx/_rc。resume 后若不重建，第一次
        _lml1_update 会把所有 it>0 的旧割当成“无主割”整体删掉。这里把每条
        非平凡割的 state 作为试探点、其在 state 处的取值作为 mx、割本身作为
        rc 恢复，让旧割和新割一样参与后续 LML1 的竞争与删除；只是不再在
        resume 第一轮被整体清空。
        """
        self._pts = [[] for _ in range(self.T + 1)]
        self._mx = [[] for _ in range(self.T + 1)]
        self._rc = [[] for _ in range(self.T + 1)]
        self._pts_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        self._mx_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        self._rc_g = [[[] for _ in range(self.K[t])] for t in range(self.T + 1)]
        n = 0
        for t in range(1, self.T + 1):
            for c in self.cuts[t]:
                if c.it == 0:
                    continue
                x = np.asarray(c.state, dtype=float).ravel()
                if x.size != self.state_dim:
                    raise ValueError(
                        f"stage {t}: 割的 state 维度 {x.size} ≠ {self.state_dim}")
                if c.stage != t:
                    raise ValueError(f"stage {t}: cut.stage={c.stage} 不一致")
                if self.mode == "multi":
                    if not 0 <= c.group < self.K[t]:
                        raise ValueError(
                            f"stage {t}: cut group {c.group} 越界 K={self.K[t]}")
                    self._pts_g[t][c.group].append(x)
                    self._mx_g[t][c.group].append(float(c.alpha + c.beta @ x))
                    self._rc_g[t][c.group].append(c)
                else:
                    self._pts[t].append(x)
                    self._mx[t].append(float(c.alpha + c.beta @ x))
                    self._rc[t].append(c)
                n += 1
        return n

    def _lml1_update(self, t: int, new_points, new_cuts) -> None:
        """LML1：每个历史试探点只留「在该点取值最大的、生成最早」的那条割，
        没被任何点选中的割直接删除（docs/LML1割筛选算法逻辑.md）。

        为什么只让**本轮新割**与历史点竞争：已被选中的割要么本轮被更强的挤掉，
        要么保持；老割之间的大小关系不会因为又过了一轮而改变，重复比较无意义。

        删除是安全的：割集是下界的下包络近似，删掉若干条只会让近似变松（下界仍
        合法），不会让它失效。保留 `it == 0` 的平凡割以保证 theta 有下界。
        """
        pts, mx, rc = self._pts[t], self._mx[t], self._rc[t]
        n_old = len(pts)
        for x, c in zip(new_points, new_cuts):
            px = np.asarray(x, dtype=float).ravel()
            pts.append(px)
            mx.append(float(c.alpha + c.beta @ px))   # 新割在其生成点必是最大
            rc.append(c)
        if n_old == 0 and len(pts) == 0:
            return
        M = np.asarray(mx, dtype=float).copy()        # 一次成型，避免逐条 vstack
        X = np.vstack(pts)
        tol = self.cut_select_eps * np.maximum(1.0, np.abs(M))
        for c in new_cuts:
            vals = X @ c.beta + c.alpha
            better = vals > M + tol
            if better.any():
                idx = np.nonzero(better)[0]
                M[idx] = vals[idx]
                tol[idx] = self.cut_select_eps * np.maximum(1.0, np.abs(vals[idx]))
                for i in idx:
                    rc[i] = c
        mx[:] = M.tolist()
        if self.cut_select_keep_rounds > 0:           # 可选：只保留最近若干轮的试探点
            keep_n = self.cut_select_keep_rounds * int(self.cfg.n_forward_paths)
            if len(pts) > keep_n:
                del pts[:-keep_n], mx[:-keep_n], rc[:-keep_n]
        alive = {id(c) for c in rc}
        before = len(self.cuts[t])
        self.cuts[t] = [c for c in self.cuts[t]
                        if c.it == 0 or id(c) in alive]
        self.n_cut_dropped += before - len(self.cuts[t])

    def _lml1_update_multi(self, t: int, new_points, new_cuts) -> None:
        """按 (t, theta) 独立的 LML1：不同场景的割绝不互相比较/删除。"""
        by_g = {g: [] for g in range(self.K[t])}
        for c in new_cuts:
            by_g[c.group].append(c)
        for g, cuts_g in by_g.items():
            if len(cuts_g) != len(new_points):
                raise RuntimeError(f"stage {t} group {g}: 新割数 {len(cuts_g)} "
                                   f"≠ 试探点数 {len(new_points)}")
            pts = self._pts_g[t][g]
            mx = self._mx_g[t][g]
            rc = self._rc_g[t][g]
            for x, c in zip(new_points, cuts_g):
                px = np.asarray(x, dtype=float).ravel()
                pts.append(px)
                mx.append(float(c.alpha + c.beta @ px))
                rc.append(c)
            if not pts:
                continue
            M = np.asarray(mx, dtype=float).copy()
            X = np.vstack(pts)
            tol = self.cut_select_eps * np.maximum(1.0, np.abs(M))
            for c in cuts_g:
                vals = X @ c.beta + c.alpha
                better = vals > M + tol
                if better.any():
                    idx = np.nonzero(better)[0]
                    M[idx] = vals[idx]
                    tol[idx] = self.cut_select_eps * np.maximum(1.0, np.abs(vals[idx]))
                    for i in idx:
                        rc[i] = c
            mx[:] = M.tolist()
            if self.cut_select_keep_rounds > 0:
                keep_n = self.cut_select_keep_rounds * int(self.cfg.n_forward_paths)
                if len(pts) > keep_n:
                    del pts[:-keep_n], mx[:-keep_n], rc[:-keep_n]
        alive = {id(c) for g in range(self.K[t]) for c in self._rc_g[t][g]}
        before = len(self.cuts[t])
        self.cuts[t] = [c for c in self.cuts[t]
                        if c.it == 0 or id(c) in alive]
        self.n_cut_dropped += before - len(self.cuts[t])

    # ------------------------------------------------------------ 细化（B）
    def _sample_refine_path(self) -> list[int]:
        return self._refine_rng.integers(0, self.n_scenes, size=self.T).tolist()

    def _scan_refinement_candidates(self, it: int):
        """沿一条细路径扫描各阶段，按归一化 π 离散度选一个 (stage, cluster)。"""
        cfg = self.refine
        t0 = time.time()
        path = self._sample_refine_path()
        states = self.forward_paths([path])[0][0]
        nproc = max(1, self.n_processes)
        chunk = max(1, int(np.ceil(self.n_scenes / (2.0 * nproc))))
        cands = []
        for t in range(max(1, int(cfg.min_refine_stage)), self.T,
                       max(1, int(cfg.candidate_stage_stride))):
            x = np.asarray(states[t - 1], dtype=float)
            blk = put_cuts(self.cuts[t + 1], self.state_dim, use_shm=self._use_shm)
            spec = self.theta_spec(t)
            try:
                tasks = []
                for k0 in range(0, self.n_scenes, chunk):
                    inds = list(range(k0, min(k0 + chunk, self.n_scenes)))
                    tasks.append((self.round_id, t, 0, x, inds, blk.ref, spec,
                                  str(cfg.signature)))
                results = self._map(scene_signatures_task, tasks)
            finally:
                blk.close()
            sigs = np.vstack([r[0] for r in results])
            qs = np.concatenate([r[1] for r in results])
            if len(sigs) != self.n_scenes:
                raise RuntimeError(f"stage {t}: 签名数 {len(sigs)} ≠ {self.n_scenes}")
            self.n_signature_solves += self.n_scenes
            med = median_pairwise_distance(sigs)
            eps = (float(cfg.eps_sig) if cfg.eps_sig > 0
                   else float(cfg.eps_sig_rel) * med)
            for ci, cluster in enumerate(self.parts[t]):
                if len(cluster) < max(2, int(cfg.min_child_size) * 2):
                    continue
                kids = greedy_split(cluster, sigs, qs, eps, cfg.eps_obj)
                if len(kids) < 2:
                    continue
                if any(len(c) < int(cfg.min_child_size) for c in kids):
                    continue
                if cfg.max_children_per_event > 0 and \
                        len(kids) > int(cfg.max_children_per_event):
                    m = int(cfg.max_children_per_event)
                    kids = kids[:m - 1] + [sum(kids[m - 1:], [])]
                if len(kids) < 2:
                    continue
                newK = self.K[t] - 1 + len(kids)
                if newK > int(cfg.max_clusters_per_stage):
                    continue
                cands.append({
                    "stage": int(t), "cluster": int(ci), "kids": kids,
                    "oldK": int(self.K[t]), "newK": int(newK),
                    "eps_sig": float(eps), "median": float(med),
                    "disp": float(cluster_dispersion(sigs, cluster)),
                    "score": float(normalized_cluster_score(sigs, cluster, eps)),
                })
        scan_s = time.time() - t0
        n_stages = len(range(max(1, int(cfg.min_refine_stage)), self.T,
                             max(1, int(cfg.candidate_stage_stride))))
        for c in cands:
            c["scan_s"] = scan_s
            c["scan_stages"] = n_stages
        if not cands:
            return None
        cands.sort(key=lambda c: (-c["score"], c["stage"], c["cluster"]))
        return cands[0]

    def _log_refine_event(self, ev: dict) -> None:
        if self.results_dir is None:
            return
        path = self.results_dir / "refine_log.csv"
        fields = ["iteration", "selected_stage", "selected_cluster", "trigger_lb_best",
                  "trigger_delta", "threshold", "old_K", "new_K", "group_sizes",
                  "cluster_hash", "cuts_cleared", "cuts_rebuilt", "scan_s",
                  "rebuild_s", "scan_solves", "lb_before", "lb_after", "delta_lb",
                  "lb_best_before", "lb_best_after", "capped", "noop"]
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: ev.get(k, "") for k in fields})

    def _apply_refinement(self, cand: dict, trial, it: int) -> dict:
        t = int(cand["stage"])
        old = {
            "parts": self.parts[t], "K": self.K[t], "weights": self.weights[t],
            "cuts": self.cuts[t], "pts": self._pts_g[t], "mx": self._mx_g[t],
            "rc": self._rc_g[t], "lgen": self.layout_gen[t],
            "cgen": self.cut_gen[t],
        }
        t0 = time.time()
        new_parts = split_cluster(self.parts[t], cand["cluster"], cand["kids"],
                                  self.n_scenes, t)
        self.parts[t] = new_parts
        self.K[t] = len(new_parts)
        self.weights[t] = cluster_weights(new_parts, self.n_scenes)
        self.layout_gen[t] += 1
        removed = len(self.cuts[t])
        self.cuts[t] = [trivial_cut(t, self.initial_state, self.state_dim, group=g)
                        for g in range(self.K[t])]
        self._pts_g[t] = [[] for _ in range(self.K[t])]
        self._mx_g[t] = [[] for _ in range(self.K[t])]
        self._rc_g[t] = [[] for _ in range(self.K[t])]
        self.n_cut_dropped += removed
        try:
            self._backward_stage(t, trial[t], it, pass_tag="rebuild",
                                 clusters=new_parts)
            self.cut_gen[t] += 1
            lb_after = self.lower_bound()
        except Exception:
            self.parts[t] = old["parts"]
            self.K[t] = old["K"]
            self.weights[t] = old["weights"]
            self.cuts[t] = old["cuts"]
            self._pts_g[t] = old["pts"]
            self._mx_g[t] = old["mx"]
            self._rc_g[t] = old["rc"]
            self.layout_gen[t] = old["lgen"]
            self.cut_gen[t] = old["cgen"]
            raise
        rebuild_s = time.time() - t0
        self.n_refines += 1
        self.last_refine_iter = it
        ev = {
            "iteration": int(it), "selected_stage": t,
            "selected_cluster": int(cand["cluster"]),
            "trigger_lb_best": float(cand.get("lb_best_before", 0.0)),
            "trigger_delta": float(cand.get("trigger_delta", 0.0)),
            "threshold": float(cand.get("threshold", 0.0)),
            "old_K": int(cand["oldK"]), "new_K": int(cand["newK"]),
            "group_sizes": [len(c) for c in new_parts],
            "cluster_hash": self._hash_parts(new_parts),
            "cuts_cleared": int(removed),
            "cuts_rebuilt": int(len(self.cuts[t]) - self.K[t]),
            "scan_s": float(cand.get("scan_s", 0.0)),
            "rebuild_s": float(rebuild_s),
            "scan_solves": int(self.n_scenes * int(cand.get("scan_stages", 0))),
            "lb_before": float(cand.get("lb_before", 0.0)),
            "lb_after": float(lb_after),
            "delta_lb": float(lb_after - cand.get("lb_before", 0.0)),
            "lb_best_before": float(cand.get("lb_best_before", self.lb_best)),
            "lb_best_after": float(self.lb_best),
            "capped": False, "noop": False,
        }
        self.refine_events.append(ev)
        self._log_refine_event(ev)
        return ev

    def _maybe_refine(self, it: int, elapsed: float, lb_current: float, trial):
        cfg = self.refine
        if not self.refine_enabled or cfg is None:
            return None
        if self.refine_events:
            last = self.refine_events[-1]
            if not last.get("judged") and \
                    it - int(last.get("iteration", 0)) >= int(cfg.stall_window):
                gain = self.lb_best - float(last.get("lb_best_before", 0.0))
                last["judged"] = True
                last["gain"] = float(gain)
                if gain < float(cfg.min_event_gain):
                    self.n_ineffective += 1
        if self.n_refines >= cfg.max_refines or self.n_attempts >= cfg.max_attempts:
            return None
        if self.n_noops >= 2 or self.n_ineffective >= cfg.max_ineffective_events:
            return None
        if it < int(cfg.warmup_rounds):
            return None
        if self.max_train_seconds > 0 and \
                elapsed > self.max_train_seconds - cfg.min_remaining_seconds:
            return None
        if it - max(self.last_refine_iter, self.last_attempt_iter) <= cfg.cooldown_rounds:
            return None
        if self.last_refine_iter > 0 and \
                it - self.last_refine_iter < cfg.recovery_timeout:
            if lb_current < cfg.recovery_frac * self._last_refine_lb_before:
                return None
        val = self.lb_best if cfg.stall_on == "lb_best" else lb_current
        self._lb_hist.append(float(val))
        if len(self._lb_hist) > int(cfg.stall_window) + 1:
            del self._lb_hist[:-(int(cfg.stall_window) + 1)]
        if not stall_trigger(self._lb_hist, int(cfg.stall_window),
                             float(cfg.stall_rtol), float(cfg.stall_abs)):
            return None
        delta = self._lb_hist[-1] - self._lb_hist[-int(cfg.stall_window) - 1]
        threshold = max(float(cfg.stall_abs),
                        float(cfg.stall_rtol) * abs(self._lb_hist[-1]))
        self.n_attempts += 1
        self.last_attempt_iter = it
        cand = self._scan_refinement_candidates(it)
        if cand is None:
            self.n_noops += 1
            return None
        self.n_noops = 0
        cand["lb_before"] = float(lb_current)
        cand["lb_best_before"] = float(self.lb_best)
        cand["trigger_delta"] = float(delta)
        cand["threshold"] = float(threshold)
        ev = self._apply_refinement(cand, trial, it)
        self._lb_hist = []
        self._last_refine_lb_before = float(lb_current)
        return ev

    def _sample_scenes(self, n: int) -> list[list[int]]:
        """给 n 个试探点各抽一份场景子集。主进程抽 ⇒ 与进程数无关、可复现。"""
        k = int(self.cfg.n_backward_samples)
        if k == 0 or k >= self.n_scenes:
            base = list(range(self.n_scenes))
            return [base] * n
        return [sorted(self.rng.choice(self.n_scenes, size=k, replace=False).tolist())
                for _ in range(n)]

    # ------------------------------------------------------------------ 下界
    def lower_bound(self) -> float:
        """阶段 0 子问题值（**当前割集**）。

        LML1 会删除未被任何试探点选中的割，因此当前 LB 可能下降；每个当前值仍是
        真问题的合法下界。历史最大值另存 `self.lb_best`，只作诊断/报告，本方法
        返回当前值，保持「LB = 当前割集的阶段 0 最优值」的语义。
        """
        blk = put_cuts(self.cuts[1], self.state_dim, use_shm=self._use_shm)
        spec = self.theta_spec(0)
        try:
            lb = float(self._map(
                lower_bound_task, [(self.round_id, blk.ref, spec)])[0])
        finally:
            blk.close()
        self.n_solves += 1
        self.lb_best = max(self.lb_best, lb)
        return lb

    # -------------------------------------------------------------- 主循环
    def train(self, n_iter: int, *, start_iter: int = 1, verbose: bool = True,
              trace: bool = False) -> list[dict]:
        """trace=True 时打印每轮各阶段的边界——程序"看起来卡住"时用它定位。"""
        history: list[dict] = []
        t0 = time.time()
        last_ckpt = t0
        log = self.log
        try:
            for k in range(start_iter, start_iter + n_iter):
                self.round_id = k
                if trace:
                    log(f"  [轮 {k}] 前向 {self.cfg.n_forward_paths} 条路径 …")
                trial, results = self.forward(self.cfg.n_forward_paths)
                sim = float(np.mean([r[2] for r in results]))
                n_fb = int(sum(r[4] for r in results))
                if trace:
                    log(f"  [轮 {k}] 后向（{self.T} 期串行，每期 "
                        f"{len(trial[1])} 个试探点）…")
                self.backward(trial, it=k)
                if trace:
                    log(f"  [轮 {k}] 下界 …")
                lb = self.lower_bound()
                elapsed = time.time() - t0
                ev = None
                if self.refine_enabled:
                    ev = self._maybe_refine(k, elapsed, lb, trial)
                elapsed = time.time() - t0
                self.last_iter = k
                rec = {"iter": k, "lb": lb, "lb_best": self.lb_best,
                       "sim": sim, "time": elapsed,
                       "solves": self.n_solves, "mip_fallback": n_fb,
                       "n_cuts": sum(len(c) for c in self.cuts[1:]),
                       "n_dropped": self.n_cut_dropped,
                       "mode": self.mode, "n_theta": self.n_theta,
                       "theta_cap": self.theta_cap,
                       "K_min": min(self.K[1:self.T+1]), "K_max": max(self.K[1:self.T+1]),
                       "n_refines": self.n_refines,
                       "refine": ev}
                history.append(rec)
                if self.on_iter is not None:
                    self.on_iter(rec)
                if verbose:
                    rtxt = ""
                    if ev is not None:
                        rtxt = (f" | refine t={ev['selected_stage']} "
                                f"K {ev['old_K']}→{ev['new_K']} "
                                f"LB {ev['lb_before']:.1f}→{ev['lb_after']:.1f}")
                    log(f"  轮 {k:5d} | LB = {lb:14.4f} | 前向均值 = {sim:14.4f} "
                        f"| 割数 = {rec['n_cuts']:6d} | {elapsed:7.1f} s"
                        + (f" | MIP 回退 {n_fb}" if n_fb else "") + rtxt)
                if self.results_dir is not None and \
                        time.time() - last_ckpt >= self.checkpoint_interval:
                    self.checkpoint()
                    last_ckpt = time.time()
                if self.max_train_seconds > 0 and \
                        time.time() - t0 >= self.max_train_seconds:
                    log(f"达到训练时限 {self.max_train_seconds:.0f} s："
                        f"完成 {k} 轮后停止")
                    break
        except KeyboardInterrupt:
            log("已中断，正在保存割 …")
        finally:
            if self.results_dir is not None:
                self.checkpoint()
        if verbose:
            log(f"训练用时 {time.time() - t0:.1f} s，累计求解 {self.n_solves} 次")
        return history

    # --------------------------------------------------------------- 落盘
    def checkpoint(self) -> Path | None:
        if self.results_dir is None:
            return None
        return save_cuts(self.results_dir, self.cuts, self.meta())

    @staticmethod
    def _hash_parts(groups) -> str:
        payload = ";".join(",".join(str(int(s)) for s in g) for g in groups)
        return hashlib.sha1(payload.encode("ascii")).hexdigest()

    def layouts_meta(self) -> dict:
        """per-stage 布局（v7 的权威来源）。"""
        return {
            "schema": 1,
            "theta_cap": int(self.theta_cap),
            "K": {str(t): int(self.K[t]) for t in range(1, self.T + 1)},
            "parts": {str(t): [list(map(int, c)) for c in self.parts[t]]
                      for t in range(1, self.T + 1)},
            "weights": {str(t): [float(v) for v in self.weights[t]]
                        for t in range(1, self.T + 1)},
            "sizes": {str(t): [len(c) for c in self.parts[t]]
                      for t in range(1, self.T + 1)},
            "hash": {str(t): self._hash_parts(self.parts[t])
                     for t in range(1, self.T + 1)},
            "gen": {str(t): int(self.layout_gen[t])
                    for t in range(1, self.T + 1)},
        }

    def refine_meta(self) -> dict:
        try:
            rng_state = self._refine_rng.bit_generator.state
        except Exception:
            rng_state = None
        return {"enabled": bool(self.refine_enabled),
                "n_refines": int(self.n_refines),
                "n_attempts": int(self.n_attempts),
                "n_noops": int(self.n_noops),
                "n_ineffective": int(self.n_ineffective),
                "last_refine_iter": int(self.last_refine_iter),
                "last_attempt_iter": int(self.last_attempt_iter),
                "lb_hist": [float(v) for v in self._lb_hist],
                "last_refine_lb_before": float(self._last_refine_lb_before),
                "rng_state": rng_state,
                "events": list(self.refine_events)}

    def meta(self) -> dict:
        return {"T": self.T, "state_dim": self.state_dim, "seed": self.seed,
                "model": "markov_grade",
                "G": getattr(self.prob, "G", None),
                "lambda0": getattr(self.prob, "lambda0", None),
                "dirichlet_s": getattr(self.prob, "dirichlet_s", None),
                "n_scenes": self.n_scenes,
                "mode": self.mode,
                "n_theta": self.n_theta,
                "theta_cap": int(self.theta_cap),
                "n_theta_by_stage": [int(self.K[t]) for t in range(1, self.T + 1)],
                "multicut": self.multicut_meta(),
                "layouts": self.layouts_meta(),
                "refine": self.refine_meta(),
                "n_iterations": self.last_iter,
                "n_cuts": sum(len(c) for c in self.cuts[1:]),
                "n_solves": self.n_solves,
                "lb_best": self.lb_best}

    def check_cut_meta(self, meta: dict) -> None:
        """割文件身份校验 + 安装 per-stage 布局。

        v7 以 meta.layouts 为准；v6 全局布局只读模拟时按 cfg 重建 uniform parts。
        v6/v5 不允许 refine；v7 的 max(K) 不得超过当前 theta_cap。
        """
        ver = int(meta.get("_format_version",
                           7 if "layouts" in meta else 6))
        m_mode = str(meta.get("mode", "single"))
        if m_mode != self.mode:
            raise SystemExit(
                f"割的 mode={m_mode} 与当前 {self.mode} 不一致，不能使用")
        mc = meta.get("multicut") or {}
        m_ncl = int(mc.get("n_clusters", 0))
        m_gm = str(mc.get("group_mode", "scene"))
        cur = self.multicut_meta()
        if m_ncl != cur["n_clusters"] or m_gm != cur["group_mode"]:
            raise SystemExit(
                f"割的初始布局 n_clusters/group_mode={m_ncl}/{m_gm} 与当前 "
                f"{cur['n_clusters']}/{cur['group_mode']} 不一致，不能使用")
        if ver >= 7 and self.grouped:
            m_sizes = list(mc.get("cluster_sizes", []))
            m_hash = str(mc.get("cluster_hash", ""))
            if m_sizes != cur["cluster_sizes"] or m_hash != cur["cluster_hash"]:
                raise SystemExit(
                    "割的初始簇成员与当前固定分区不一致（cluster_sizes/hash 不符）")
        if ver >= 7:
            layouts = meta.get("layouts")
            if not isinstance(layouts, dict):
                raise SystemExit("v7 割文件缺少 layouts 元数据")
            if int(layouts.get("theta_cap", -1)) != int(self.theta_cap):
                raise SystemExit(
                    f"割的 theta_cap={layouts.get('theta_cap')} 与当前 "
                    f"{self.theta_cap} 不一致")
            parts_new, K_new, w_new, gen_new = {}, {}, {}, {}
            for t in range(1, self.T + 1):
                key = str(t)
                if key not in layouts.get("parts", {}):
                    raise SystemExit(f"v7 layouts 缺少 stage {t}")
                groups = [[int(s) for s in c] for c in layouts["parts"][key]]
                validate_partition(groups, self.n_scenes, t)
                K = int(layouts.get("K", {}).get(key, len(groups)))
                if K != len(groups):
                    raise SystemExit(f"stage {t}: K={K} ≠ len(parts)={len(groups)}")
                if K > self.theta_cap:
                    raise SystemExit(f"stage {t}: K={K} > theta_cap={self.theta_cap}")
                w = np.asarray(layouts.get("weights", {}).get(key, []), float)
                w_ref = (np.array([1.0]) if self.mode == "single"
                         else cluster_weights(groups, self.n_scenes))
                if w.size != K or not np.allclose(w, w_ref, atol=1e-10):
                    raise SystemExit(f"stage {t}: weights 与 parts 不一致")
                h = str(layouts.get("hash", {}).get(key, ""))
                if h != self._hash_parts(groups):
                    raise SystemExit(f"stage {t}: cluster_hash 不符")
                parts_new[t] = groups
                K_new[t] = K
                w_new[t] = w
                gen_new[t] = int(layouts.get("gen", {}).get(key, 0))
            self.parts = [[]] + [parts_new[t] for t in range(1, self.T + 1)]
            self.K = [0] + [K_new[t] for t in range(1, self.T + 1)] + [0]
            self.weights = [np.zeros(0)] + [w_new[t] for t in range(1, self.T + 1)] \
                + [np.zeros(0)]
            self.layout_gen = [0] + [gen_new[t] for t in range(1, self.T + 1)] + [0]
            self.clusters = self.parts[1]
            self.cluster_weights = self.weights[1]
        elif ver == 6:
            if self.refine_enabled:
                raise SystemExit("v6 全局布局不能用于 refine 模式")
            m_k = int(meta.get("n_theta", 1))
            if m_k != self.n_theta:
                raise SystemExit(
                    f"割的全局 K={m_k} 与当前 theta_cap={self.n_theta} 不一致")
            if self.grouped:
                m_sizes = list(mc.get("cluster_sizes", []))
                m_hash = str(mc.get("cluster_hash", ""))
                if m_sizes != cur["cluster_sizes"] or m_hash != cur["cluster_hash"]:
                    raise SystemExit(
                        "割的簇成员与当前固定分区不一致（cluster_sizes/hash 不符）")
            for t in range(1, self.T + 1):
                self.parts[t] = [list(c) for c in self.clusters]
                self.K[t] = len(self.parts[t])
                self.weights[t] = (np.array([1.0]) if self.mode == "single"
                                   else cluster_weights(self.clusters, self.n_scenes))
                self.layout_gen[t] = 0
        else:
            if self.mode != "single":
                raise SystemExit(f"v{ver} 旧割只支持 single 只读，不能用于 multi")
            for t in range(1, self.T + 1):
                self.parts[t] = [list(range(self.n_scenes))]
                self.K[t] = 1
                self.weights[t] = np.array([1.0])
                self.layout_gen[t] = 0
            self.clusters = self.parts[1]
            self.cluster_weights = self.weights[1]

    def restore_refine_state(self, meta: dict) -> None:
        """resume 后恢复触发历史/计数/RNG；v6 或 refine 关闭时清空。"""
        rf = meta.get("refine") or {}
        self.n_refines = int(rf.get("n_refines", 0))
        self.n_attempts = int(rf.get("n_attempts", 0))
        self.n_noops = int(rf.get("n_noops", 0))
        self.n_ineffective = int(rf.get("n_ineffective", 0))
        self.last_refine_iter = int(rf.get("last_refine_iter", 0))
        self.last_attempt_iter = int(rf.get("last_attempt_iter", 0))
        self._lb_hist = [float(v) for v in rf.get("lb_hist", [])]
        self._last_refine_lb_before = float(rf.get("last_refine_lb_before", 0.0))
        self.refine_events = list(rf.get("events", []))
        st = rf.get("rng_state")
        if isinstance(st, dict):
            try:
                self._refine_rng.bit_generator.state = st
            except Exception:
                pass

    def multicut_meta(self) -> dict:
        """初始全局布局的身份（cfg 层）；B 的当前布局看 layouts。"""
        if self.mode == "single":
            return {"n_clusters": 0, "group_mode": "scene",
                    "cluster_sizes": [], "cluster_hash": ""}
        groups = self.clusters if self.clusters is not None else \
            [[i] for i in range(self.n_scenes)]
        sizes = [len(g) for g in groups]
        return {"n_clusters": len(groups) if self.grouped else 0,
                "group_mode": self.group_mode if self.grouped else "scene",
                "cluster_sizes": sizes, "cluster_hash": self._hash_parts(groups)}

    def open_log(self, name: str = "sddp_training_log.csv") -> TrainingLog:
        if self.results_dir is None:
            raise RuntimeError("未指定 results_dir，无法写训练日志")
        return TrainingLog(self.results_dir, name)

    # --------------------------------------------- 前向割块（仿真/外部复用）
    def forward_refs(self):
        """当前割表的每阶段 ref（前向路径任务用）。调用方负责 close。"""
        return [put_cuts(self.cuts[t + 1], self.state_dim,
                         use_shm=self._use_shm) for t in range(0, self.T)]

    # ------------------------------------------------------ 仿真用采样器
    def sim_scene_paths(self, n_paths: int, seed: int | None = None):
        """仿真/复现用的场景路径。与训练共用 rng 会打乱训练随机流，故单独一个 Generator。"""
        base = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(base)
        return [rng.integers(0, self.n_scenes, size=self.T).tolist()
                for _ in range(n_paths)]
