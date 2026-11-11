# async_engine.py — iteration 级并行 SDDP 原型。
"""每个 worker 持有一条路径的完整 forward + backward；主进程维护全局割池，
并把新割作为增量广播给 worker。worker 只有在下一次领取 iteration 时才需要
接收增量割，因此路径之间是异步推进的，而不是 Python 旧引擎的 10 路径同轮批处理。

边界：
  · 支持 cut_selection = "none" / "lml1" / "julia"；
  · 支持后向 Strengthened Benders 割提升（cut_lift），触发由主进程决定；
  · 每个 iteration 仍是一个前向路径、后向枚举 n_backward_samples=0 的全部场景；
  · 下界在主进程重建 stage 0 LP，使用全局活跃割；
  · worker 持“缓存全部割 + 活跃割”的两层池，支持 Julia 式割的复活；
  · worker 的模型缓存与 code/subproblem.py 的 assemble 契约不变。
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
import time
from pathlib import Path

import numpy as np

from .cut import Cut, pack, trivial_cut
from .cut_pool import CutPool


def _solve_forward(sub, cfg):
    """前向求解器由问题配置决定：Tier2 用 MIP，连续模型用 LP。"""
    if bool(getattr(cfg, "integer_enabled", False)):
        sub.solve_mip()
    else:
        sub.solve_lp()


def _scene_indices(cfg, round_id: int, n_scenes: int):
    k = int(getattr(cfg, "n_backward_samples", 0))
    if k <= 0 or k >= n_scenes:
        return list(range(n_scenes))
    rng = np.random.default_rng(int(cfg.seed) + int(round_id) * 100003)
    return sorted(rng.choice(n_scenes, size=k, replace=False).tolist())


def _worker_main(wid, prob, make_sub, cfg, T, initial_state, initial_cuts,
                 task_q, result_q, pool_version):
    cuts = [list(c) for c in initial_cuts]
    # 所有发现过的割都缓存；cuts[s] 只是当前进入 LP 的活跃割。
    all_cuts = [{getattr(c, "uid", None): c for c in cuts[s]}
                for s in range(T + 2)]
    active_uids = [{getattr(c, "uid", None) for c in cuts[s]}
                   for s in range(T + 2)]
    # cut_lift 与割池正交：这里只改变后向生成的割，不改变池的选择规则。
    lift_enabled = bool(getattr(cfg, "cut_lift_enabled", False))
    lcfg = getattr(cfg, "cut_lift", None)
    lift_scope = str(getattr(lcfg, "scope", "all"))
    lift_last_k = int(getattr(lcfg, "last_k", 0))
    lift_stages = frozenset(
        int(t) for t in (getattr(lcfg, "lift_stages", ()) or ()))
    lift_value = str(getattr(lcfg, "value", "exact"))
    lift_time = float(getattr(lcfg, "time_limit", 5.0))
    lift_rel = float(getattr(lcfg, "rel_gap", 0.0))
    lift_abs = float(getattr(lcfg, "abs_gap", 0.0))
    lift_margin = float(getattr(lcfg, "safety_margin", 1e-6))
    heur_on = bool(getattr(cfg, "forward_heuristic", False))
    heur_mode = str(getattr(cfg, "forward_heuristic_mode", "sa"))
    heur_q_dual = bool(getattr(cfg, "heur_q_dual_trials", False))
    heur_lp_p0 = float(getattr(cfg, "heur_lp_p0", 0.5))
    heur_lp_p1 = float(getattr(cfg, "heur_lp_p1", 0.95))
    heur_ramp = int(getattr(cfg, "heur_ramp_iters", 1000))
    heur_alpha = float(getattr(cfg, "heur_alpha", 1.0))
    heur_rng = np.random.default_rng(int(cfg.seed) + wid * 7919)
    heur_trials = 0
    heur_used = 0
    heur_fail = 0
    sa_moves = 0
    sa_accepted = 0
    sa_time_s = 0.0
    qp_perturbed = 0
    qp_base = 0
    qp_solves = 0
    qp_extra_trials = 0
    # 一个模型对应一个「未来割阶段」s：求解第 t=s−1 期时使用 cuts[s]。
    # s=1 由虚拟阶段 0 使用；s=2..T+1 是普通真实阶段模型（base 同 t=1）。
    cut_models = {}

    def cut_model(s: int):
        nonlocal cut_models
        sub = cut_models.get(s)
        if sub is not None:
            return sub
        if s == 1:
            sub = make_sub(0, prob, cfg)
        else:
            sub = make_sub(1, prob, cfg)
        sub.set_cuts(cuts[s], key=("init", s))
        cut_models[s] = sub
        return sub

    def add_cut(s: int, c):
        all_cuts[s][c.uid] = c
        if c.uid not in active_uids[s]:
            cuts[s].append(c)
            active_uids[s].add(c.uid)
        sub = cut_models.get(s)
        if sub is not None:
            sub.append_cuts([c])

    def deactivate(s: int, uids):
        if not uids:
            return
        ids = set(uids)
        cuts[s] = [c for c in cuts[s] if c.uid not in ids]
        active_uids[s].difference_update(ids)
        sub = cut_models.get(s)
        if sub is not None:
            sub.remove_cuts(uids)

    def activate(s: int, uids):
        for uid in uids:
            if uid in active_uids[s]:
                continue
            c = all_cuts[s].get(uid)
            if c is None:
                raise KeyError(f"worker {wid}: activate unknown cut {uid} at stage {s}")
            cuts[s].append(c)
            active_uids[s].add(uid)
            sub = cut_models.get(s)
            if sub is not None:
                sub.append_cuts([c])

    local_version = 0
    sync_calls = 0
    sync_skipped = 0
    sync_wait_s = 0.0

    def apply_ack(ack):
        nonlocal local_version
        _, new_cuts, activate_uids, deactivate_uids = ack[:4]
        if len(ack) > 4:
            local_version = int(ack[4])
        for s, gone in enumerate(deactivate_uids):
            deactivate(s, gone)
        for s, ds in enumerate(new_cuts):
            for c in ds:
                all_cuts[s][c.uid] = c
        for s, uids in enumerate(activate_uids):
            activate(s, uids)

    def sync_block(round_id: int, new_cuts, *, force: bool = False):
        """阶段块边界：把本块新割交给主进程，并取回最新全局割增量。

        force=False 时，如果全局版本没变且没有新割，就跳过这次往返。
        """
        nonlocal sync_calls, sync_skipped, sync_wait_s
        if (not force and not any(new_cuts)
                and int(pool_version.value) == local_version):
            sync_skipped += 1
            return
        sync_calls += 1
        t0 = time.perf_counter()
        result_q.put(("sync", wid, int(round_id), new_cuts))
        ack = task_q.get()
        sync_wait_s += time.perf_counter() - t0
        if ack is None:
            raise RuntimeError("worker received shutdown while waiting for block ack")
        if ack[0] != "ack":
            raise RuntimeError(f"worker expected ack, got {ack[0]!r}")
        apply_ack(ack)

    default_block_size = int(getattr(cfg, "stage_block_size", 0) or 0)

    while True:
        cmd = task_q.get()
        if cmd is None:
            break
        block_size = default_block_size
        task_version = None
        if cmd[0] == "iter":
            _, round_id, scene_paths, deltas, deletions = cmd[:5]
            lift_this = bool(cmd[5]) if len(cmd) > 5 else False
            if len(cmd) > 6:
                block_size = int(cmd[6])
            if len(cmd) > 7:
                task_version = int(cmd[7])
            for s, gone in enumerate(deletions):
                deactivate(s, gone)
            for s, ds in enumerate(deltas):
                for c in ds:
                    add_cut(s, c)
        elif cmd[0] == "iterj":
            (_, round_id, scene_paths, new_cuts, activate_uids,
             deactivate_uids) = cmd[:6]
            lift_this = bool(cmd[6]) if len(cmd) > 6 else False
            if len(cmd) > 7:
                block_size = int(cmd[7])
            if len(cmd) > 8:
                task_version = int(cmd[8])
            for s, gone in enumerate(deactivate_uids):
                deactivate(s, gone)
            for s, ds in enumerate(new_cuts):
                for c in ds:
                    all_cuts[s][c.uid] = c
            for s, uids in enumerate(activate_uids):
                activate(s, uids)
        else:
            continue
        if task_version is not None:
            local_version = task_version
        lift_this = bool(lift_this and lift_enabled)
        try:
            x0 = np.asarray(initial_state, dtype=float).copy()
            K = len(scene_paths)
            # trial[t][k] = 第 t 期入库状态（第 k 条路径）
            trial = [[None] * K for _ in range(T + 1)]
            totals = np.zeros(K)
            n_solves = 0
            heur_trials = 0
            heur_used = 0
            heur_fail = 0
            sa_moves = 0
            sa_accepted = 0
            sa_time_s = 0.0
            qp_perturbed = 0
            qp_base = 0
            qp_solves = 0
            qp_extra_trials = 0

            # 概率随轮次升高：LP 概率 p0→p1；其余走 SA/qperturb。
            k_ramp = min(max(int(round_id), 1), max(1, heur_ramp))
            lp_prob = (heur_lp_p0 + (heur_lp_p1 - heur_lp_p0)
                       * (k_ramp / max(1, heur_ramp)) ** heur_alpha)

            # forward：同一切割快照下顺序走 K 条路径
            for k, scene_path in enumerate(scene_paths):
                x = x0.copy()
                for t in range(0, T + 1):
                    if block_size and t > 1 and (t - 1) % block_size == 0:
                        sync_block(round_id, [[] for _ in range(T + 2)])
                    # 求解第 t 期用 cuts[t+1] 作为未来成本
                    sub = cut_model(t + 1)
                    if t == 0:
                        sub.set_state(x)
                        _solve_forward(sub, cfg)
                        n_solves += 1
                        x = sub.out_state_vector()
                        continue

                    trial[t][k] = x.copy()
                    scene = prob.pool[int(scene_path[t - 1])]
                    use_heur = heur_on and heur_rng.random() >= lp_prob
                    if use_heur and heur_mode == "qperturb":
                        heur_trials += 1
                        try:
                            qr = sub.qperturb_step(
                                x, scene, rng=heur_rng)
                        except Exception:
                            heur_fail += 1
                        else:
                            n_solves += int(qr.n_solves)
                            qp_solves += int(qr.n_solves)
                            x = qr.x_out
                            totals[k] += float(qr.stage_cost)
                            if qr.perturbed:
                                heur_used += 1
                                qp_perturbed += 1
                                # 主线继续扰动解；把 base 解作为额外的下一期试探点。
                                if heur_q_dual and t < T:
                                    trial[t + 1].append(
                                        np.asarray(qr.base_x_out,
                                                   dtype=float).copy())
                                    qp_extra_trials += 1
                            else:
                                qp_base += 1
                            continue
                    elif use_heur:
                        heur_trials += 1
                        h = sub.heuristic_step(
                            x, scene, stage_t=t,
                            cuts_next=cuts[t + 1], rng=heur_rng)
                        if h is not None:
                            heur_used += 1
                            sa_moves += int(h.sa_moves)
                            sa_accepted += int(h.sa_accepted)
                            sa_time_s += float(h.sa_time_s)
                            x = h.x_out
                            totals[k] += float(h.stage_cost)
                            continue
                        # 仅预期不可行才回退到原有 LP/MIP。
                        heur_fail += 1

                    sub.set_point(x, scene)
                    _solve_forward(sub, cfg)
                    n_solves += 1
                    x = sub.out_state_vector()
                    totals[k] += float(sub.stage_cost())

            # backward：每个阶段对 K 个试探状态各枚举场景，生成 K 条割
            new_cuts = [[] for _ in range(T + 2)]
            lift_stats = {"n_cuts": 0, "n_mip": 0, "n_fb": 0, "gain_sum": 0.0}
            inds = _scene_indices(cfg, round_id, len(prob.pool))
            for t in range(T, 0, -1):
                sub = cut_model(t + 1)
                stage_lift = bool(lift_this and (
                    lift_scope == "all"
                    or (lift_scope == "last" and t == T)
                    or (lift_scope == "tail" and t > T - lift_last_k)
                    or (lift_scope == "selected" and t in lift_stages)))
                stage_cuts = []
                # 主线每路径一个点；qperturb dual 会在下一期追加 base 解。
                for k, xb in enumerate(
                        np.asarray(p, dtype=float) for p in trial[t]
                        if p is not None):
                    sub.set_state(xb)
                    grads = np.empty((len(inds), prob.state_dim))
                    objs = np.empty(len(inds))
                    lift_vals = np.empty(len(inds)) if stage_lift else None
                    for j, si in enumerate(inds):
                        sub.set_scene(prob.pool[si])
                        sub.solve_lp()
                        n_solves += 1
                        beta_j = np.asarray(sub.gradient("dj"), dtype=float)
                        grads[j] = beta_j
                        lp_obj = float(sub.obj)
                        objs[j] = lp_obj
                        if stage_lift:
                            nu = sub.solve_lift(
                                beta_j, time_limit=lift_time,
                                rel_gap=lift_rel, abs_gap=lift_abs,
                                safety_margin=lift_margin, value=lift_value)
                            if nu is None:
                                nu = lp_obj - float(beta_j @ xb)
                                lift_stats["n_fb"] += 1
                            lift_vals[j] = nu
                    beta = grads.mean(axis=0)
                    obj_lp = float(objs.mean())
                    if stage_lift:
                        # 与同步 worker 相同：提升值可能与 LP 割互有强弱，取 max。
                        obj_lift = float(np.mean(lift_vals + grads @ xb))
                        obj = max(obj_lift, obj_lp)
                        lift_stats["n_cuts"] += 1
                        lift_stats["n_mip"] += len(inds)
                        lift_stats["gain_sum"] += obj - obj_lp
                    else:
                        obj = obj_lp
                    c = Cut(float(obj) - float(beta @ xb), beta, xb, t, obj,
                            int(round_id))
                    c.uid = ("w", wid, int(round_id), t, k)
                    stage_cuts.append(c)
                # 割阶段 t 的模型 v=t 只服务第 t−1 期；当前后向正在用 t+1。
                for c in stage_cuts:
                    add_cut(t, c)
                new_cuts[t].extend(stage_cuts)
                if (block_size and t > 1
                        and (T - t + 1) % block_size == 0):
                    sync_block(round_id, new_cuts, force=True)
                    new_cuts = [[] for _ in range(T + 2)]
            result_q.put(("ok", wid, int(round_id), float(totals.mean()),
                          n_solves, new_cuts, lift_stats,
                          {"calls": sync_calls, "skipped": sync_skipped,
                           "wait_s": sync_wait_s},
                          {"trials": heur_trials, "used": heur_used,
                           "fail": heur_fail, "sa_moves": sa_moves,
                           "sa_accepted": sa_accepted,
                           "sa_time_s": sa_time_s,
                           "qp_perturbed": qp_perturbed,
                           "qp_base": qp_base, "qp_solves": qp_solves,
                           "qp_extra_trials": qp_extra_trials}))
        except Exception:
            result_q.put(("err", wid, int(round_id), traceback.format_exc()))


class AsyncIterationEngine:
    """Iteration-level asynchronous SDDP prototype."""

    def __init__(self, prob, make_sub, cfg, *, results_dir=None, log=print,
                 n_workers: int | None = None, paths_per_task: int = 1):
        self.cut_selection = str(getattr(cfg, "cut_selection", "none"))
        if self.cut_selection not in ("none", "lml1", "julia"):
            raise ValueError(f"cut_selection 只能是 none/lml1/julia："
                             f"{self.cut_selection!r}")
        self.prob = prob
        self.make_sub = make_sub
        self.cfg = cfg
        self.T = int(prob.T)
        self.state_dim = int(prob.state_dim)
        self.initial_state = np.round(
            np.asarray(prob.initial_state(), dtype=float).ravel(), 6)
        self.n_workers = int(n_workers or cfg.n_processes)
        if self.n_workers < 1:
            raise ValueError("n_workers 必须 ≥ 1")
        self.paths_per_task = int(paths_per_task)
        if self.paths_per_task < 1:
            raise ValueError("paths_per_task 必须 ≥ 1")
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.log = log if log is not None else (lambda *a, **k: None)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.n_scenes = len(prob.pool)
        self.cuts = [[] for _ in range(self.T + 2)]
        for t in range(1, self.T + 1):
            c = trivial_cut(t, self.initial_state, self.state_dim)
            c.uid = ("init", t, 0)
            self.cuts[t].append(c)
        self.cut_pool = None
        self.cut_deletion_minimum = int(getattr(cfg, "cut_deletion_minimum", 1))
        if self.cut_selection == "julia":
            self.cut_pool = CutPool(self.T + 1, self.cut_deletion_minimum)
            for t in range(1, self.T + 1):
                self.cut_pool.seed(t, self.cuts[t])
                self.cuts[t] = self.cut_pool.active(t)
        # cut_lift 与割池、调度解耦：主进程只决定“本轮是否 lift”，
        # scope/value/time_limit 等由 worker 在生成割时执行。
        self.cut_lift_cfg = getattr(cfg, "cut_lift", None)
        self.cut_lift_enabled = bool(getattr(cfg, "cut_lift_enabled", False))
        self._cl_scope = str(getattr(self.cut_lift_cfg, "scope", "all"))
        self._cl_last_k = int(getattr(self.cut_lift_cfg, "last_k", 0))
        self._cl_stages = frozenset(
            int(t) for t in (getattr(self.cut_lift_cfg, "lift_stages", ()) or ()))
        self._cl_mode = str(getattr(self.cut_lift_cfg, "mode", "always"))
        self._cl_threshold = float(
            getattr(self.cut_lift_cfg, "lb_gain_threshold", 20.0))
        self._cl_min_round = int(getattr(self.cut_lift_cfg, "min_round", 50))
        self.n_lift_cuts = 0
        self.n_lift_mip = 0
        self.n_lift_fallback = 0
        self.n_block_syncs = 0
        self._stage_block_static = int(getattr(
            cfg, "stage_block_size", 0) or 0)
        self._stage_block_adaptive = bool(getattr(
            cfg, "stage_block_adaptive", False))
        self._stage_block_sizes = tuple(int(x) for x in (
            getattr(cfg, "stage_block_sizes", ()) or ()))
        self._stage_block_switch_s = tuple(float(x) for x in (
            getattr(cfg, "stage_block_switch_s", ()) or ()))
        self._t0 = None
        self._block_hist = {}
        self._iter_block = {}
        self._lift_rounds = []
        self._lift_gains = []
        self._last_lb = None
        self._prev_lb_delta = None
        self.n_solves = 0
        self.n_iterations = 0
        self.last_iter = 0
        self.lb_best = 0.0
        self.on_iter = None
        # LML1 的全局历史；只在主进程维护。
        self.cut_select_eps = float(getattr(cfg, "cut_select_eps", 1e-6))
        self.cut_select_keep_rounds = int(
            getattr(cfg, "cut_select_keep_rounds", 0))
        self._pts = [[] for _ in range(self.T + 2)]
        self._mx = [[] for _ in range(self.T + 2)]
        self._rc = [[] for _ in range(self.T + 2)]
        self.n_cut_dropped = 0
        self._ctx = mp.get_context("spawn")
        self._task_q = []
        self._result_q = self._ctx.Queue()
        self._pool_version = self._ctx.Value("q", 0)
        self.n_sync_wait_s = 0.0
        self.n_sync_skipped = 0
        self.n_heur_trials = 0
        self.n_heur_used = 0
        self.n_heur_fail = 0
        self.n_sa_moves = 0
        self.n_sa_accepted = 0
        self.sa_time_s = 0.0
        self.n_qp_perturbed = 0
        self.n_qp_base = 0
        self.n_qp_solves = 0
        self.n_qp_extra_trials = 0
        self._proc = []
        # 每个 worker 已持有的割 uid。不能用「全局前缀长度」，因为不同 worker 的
        # 完成顺序会让全局列表的追加顺序与本 worker 的本地顺序不同。
        self._view_uids = []
        # julia 模式额外区分“已缓存”和“当前活跃”，前者需要保留以支持复活。
        self._view_all = []
        self._view_active = []
        self._lb_sub = None
        self._lb_key = None
        self._closed = False

    def _start_workers(self):
        initial = [list(c) for c in self.cuts]
        for wid in range(self.n_workers):
            q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_worker_main,
                args=(wid, self.prob, self.make_sub, self.cfg, self.T,
                      self.initial_state, initial, q, self._result_q,
                      self._pool_version),
                daemon=True)
            p.start()
            self._task_q.append(q)
            self._proc.append(p)
            self._view_uids.append({c.uid for lst in initial for c in lst})
            self._view_all.append({c.uid for lst in initial for c in lst})
            self._view_active.append({c.uid for lst in initial for c in lst})

    def _block_size_now(self) -> int:
        if not self._stage_block_adaptive:
            return self._stage_block_static
        elapsed = 0.0 if self._t0 is None else time.time() - self._t0
        for i, sw in enumerate(self._stage_block_switch_s):
            if elapsed < sw:
                return int(self._stage_block_sizes[i])
        return int(self._stage_block_sizes[-1])

    def _lift_now(self, k: int) -> bool:
        """本轮是否做增强割。

        always 直接开；stagnation 用“最近一个已完成 iteration 的 LB 增量”判断。
        异步环境下没有唯一的“上一轮”，这是刻意采用的全局近似口径。
        """
        if not self.cut_lift_enabled:
            return False
        if self._cl_mode == "always":
            return True
        return (self._prev_lb_delta is not None
                and int(k) >= self._cl_min_round
                and self._prev_lb_delta < self._cl_threshold)

    def _plain_delta(self, wid: int):
        """worker 尚未持有的割，以及主进程已经删掉的割。"""
        seen = self._view_uids[wid]
        deltas = []
        deletions = []
        for t in range(self.T + 2):
            gids = {c.uid for c in self.cuts[t]}
            new = [c for c in self.cuts[t] if c.uid not in seen]
            gone = [uid for uid in seen if uid not in gids]
            for c in new:
                seen.add(c.uid)
            for uid in gone:
                seen.discard(uid)
            deltas.append(new)
            deletions.append(gone)
        return deltas, deletions

    def _julia_delta(self, wid: int):
        """同步“全部割缓存”和“当前活跃割”两类增量。"""
        seen_all = self._view_all[wid]
        seen_active = self._view_active[wid]
        new_cuts, activate, deactivate = [], [], []
        for t in range(self.T + 2):
            pool = self.cut_pool.pools.get(t)
            active_now = [] if pool is None else pool.active
            active_ids = {c.uid for c in active_now}
            # 只把“新且当前活跃”的割发给 worker；缓存中的非活跃割等复活时再发。
            new = []
            for c in active_now:
                if c.uid not in seen_all:
                    new.append(c)
                    seen_all.add(c.uid)
            act = [c for c in active_now if c.uid not in seen_active]
            deact = [uid for uid in seen_active if uid not in active_ids]
            for c in act:
                seen_active.add(c.uid)
            for uid in deact:
                seen_active.discard(uid)
            new_cuts.append(new)
            activate.append([c.uid for c in act])
            deactivate.append(deact)
        return new_cuts, activate, deactivate

    def _send_ack(self, wid: int):
        """只同步割，不派新 iteration；用于阶段块边界。"""
        if self.cut_pool is not None:
            new, activate, deactivate = self._julia_delta(wid)
        else:
            new, deactivate = self._plain_delta(wid)
            activate = [[] for _ in range(self.T + 2)]
        self._task_q[wid].put(
            ("ack", new, activate, deactivate, int(self._pool_version.value)))

    def _dispatch(self, wid: int, k: int):
        if self.cut_pool is not None:
            return self._dispatch_julia(wid, k)
        deltas, deletions = self._plain_delta(wid)
        scene_paths = [self.rng.integers(0, self.n_scenes,
                                        size=self.T).tolist()
                       for _ in range(self.paths_per_task)]
        blk = self._block_size_now()
        self._iter_block[int(k)] = blk
        self._task_q[wid].put(
            ("iter", k, scene_paths, deltas, deletions, self._lift_now(k),
             blk, int(self._pool_version.value)))

    def _dispatch_julia(self, wid: int, k: int):
        new_cuts, activate, deactivate = self._julia_delta(wid)
        scene_paths = [self.rng.integers(0, self.n_scenes,
                                        size=self.T).tolist()
                       for _ in range(self.paths_per_task)]
        blk = self._block_size_now()
        self._iter_block[int(k)] = blk
        self._task_q[wid].put(
            ("iterj", k, scene_paths, new_cuts, activate, deactivate,
             self._lift_now(k), blk, int(self._pool_version.value)))

    def _merge(self, wid: int, new_cuts):
        if self.cut_pool is not None:
            self._merge_julia(wid, new_cuts)
        else:
            # worker 已经把它自己生成的新割放进本地池；先记录 uid，再交给 LML1。
            seen = self._view_uids[wid]
            for t in range(self.T + 2):
                if new_cuts[t]:
                    self.cuts[t].extend(new_cuts[t])
                    for c in new_cuts[t]:
                        seen.add(c.uid)
                    if self.cut_selection == "lml1":
                        self._lml1_update(
                            t, [np.asarray(c.state, dtype=float)
                                for c in new_cuts[t]], new_cuts[t])
        if any(new_cuts[t] for t in range(self.T + 2)):
            self._pool_version.value += 1

    def _merge_julia(self, wid: int, new_cuts):
        # 源 worker 自己把新割加入了本地活跃池；主进程随后按 Julia 规则筛选。
        seen_all = self._view_all[wid]
        seen_active = self._view_active[wid]
        for t in range(self.T + 2):
            for c in new_cuts[t]:
                seen_all.add(c.uid)
                seen_active.add(c.uid)   # worker 生成时立即加入活跃池
            if t == 0 or t > self.T:
                continue
            for c in new_cuts[t]:
                _, deactivated = self.cut_pool.add(t, c)
                self.n_cut_dropped += len(deactivated)
            self.cuts[t] = self.cut_pool.active(t)

    def restore_cut_pool(self) -> int:
        """resume 后把已加载的活跃割装入 Julia 式缓存池。"""
        if self.cut_pool is None:
            return 0
        self.cut_pool = CutPool(self.T + 1, self.cut_deletion_minimum)
        n = 0
        for t in range(1, self.T + 1):
            self.cut_pool.seed(t, self.cuts[t])
            self.cuts[t] = self.cut_pool.active(t)
            n += len(self.cuts[t])
        return n

    def seed_initial_cuts(self, cuts_by_stage) -> int:
        """从割文件建立起始割池；必须在 train() 前调用。

        cuts.json 保存的是 all_cuts，无法恢复原 active 标记，因此载入的割全部
        视为活跃。用于“同一割池、不同 lift 阶段”的受控单轮对比。
        """
        if self._proc:
            raise RuntimeError("seed_initial_cuts 必须在 start_workers 之前调用")
        self.cuts = [[] for _ in range(self.T + 2)]
        for t in range(1, self.T + 1):
            self.cuts[t] = list(cuts_by_stage[t])
        self.cut_pool = None
        if self.cut_selection == "julia":
            self.cut_pool = CutPool(self.T + 1, self.cut_deletion_minimum)
            for t in range(1, self.T + 1):
                self.cut_pool.seed(t, self.cuts[t])
                self.cuts[t] = self.cut_pool.active(t)
        self._lb_sub = None
        self.lb_best = 0.0
        return sum(len(c) for c in self.cuts[1:self.T + 1])

    def _lml1_update(self, t: int, new_points, new_cuts) -> None:
        """与 utils/engine.py 的 LML1 同一数学逻辑（不在这里改写算法）。

        主进程是全局割池的唯一权威；worker 本地可能暂时保留已被删除的旧割，
        这些旧割仍然合法，只是近似更松。下一次 dispatch 会把删除 uid 发给
        持有它的 worker。
        """
        pts, mx, rc = self._pts[t], self._mx[t], self._rc[t]
        n_old = len(pts)
        for x, c in zip(new_points, new_cuts):
            px = np.asarray(x, dtype=float).ravel()
            pts.append(px)
            mx.append(float(c.alpha + c.beta @ px))
            rc.append(c)
        if n_old == 0 and len(pts) == 0:
            return
        M = np.asarray(mx, dtype=float).copy()
        X = np.vstack(pts)
        tol = self.cut_select_eps * np.maximum(1.0, np.abs(M))
        for c in new_cuts:
            vals = X @ c.beta + c.alpha
            better = vals > M + tol
            if better.any():
                idx = np.nonzero(better)[0]
                M[idx] = vals[idx]
                tol[idx] = self.cut_select_eps * np.maximum(
                    1.0, np.abs(vals[idx]))
                for i in idx:
                    rc[i] = c
        mx[:] = M.tolist()
        if self.cut_select_keep_rounds > 0:
            keep_n = self.cut_select_keep_rounds * max(1, self.paths_per_task)
            if len(pts) > keep_n:
                del pts[:-keep_n], mx[:-keep_n], rc[:-keep_n]
        alive = {id(c) for c in rc}
        before = len(self.cuts[t])
        self.cuts[t] = [c for c in self.cuts[t]
                        if c.it == 0 or id(c) in alive]
        self.n_cut_dropped += before - len(self.cuts[t])

    def _lower_bound(self) -> float:
        if self._lb_sub is None:
            self._lb_sub = self.make_sub(0, self.prob, self.cfg)
        key = (int(self.n_iterations), "lb")
        self._lb_sub.assemble(pack(self.cuts[1], self.state_dim), key=key)
        self._lb_sub.set_state(self.initial_state)
        self._lb_sub.solve_lp()
        self.n_solves += 1
        lb = float(self._lb_sub.obj)
        self.lb_best = max(self.lb_best, lb)
        return lb

    def lower_bound(self) -> float:
        """当前全局割池的阶段 0 下界。"""
        return self._lower_bound()

    def meta(self) -> dict:
        n_all = (sum(len(p.all_cuts) for p in self.cut_pool.pools.values())
                 if self.cut_pool is not None else
                 sum(len(c) for c in self.cuts[1:self.T + 1]))
        return {"T": self.T, "state_dim": self.state_dim,
                "seed": int(self.cfg.seed), "model": "markov_grade",
                "G": getattr(self.prob, "G", None),
                "lambda0": getattr(self.prob, "lambda0", None),
                "dirichlet_s": getattr(self.prob, "dirichlet_s", None),
                "n_scenes": self.n_scenes,
                "n_iterations": self.last_iter,
                "n_cuts": sum(len(c) for c in self.cuts[1:self.T + 1]),
                "n_all_cuts": n_all,
                "n_solves": self.n_solves,
                "lb_best": self.lb_best,
                "mode": "async_iteration",
                "cut_selection": self.cut_selection,
                "stage_block_size": int(getattr(
                    self.cfg, "stage_block_size", 0) or 0),
                "stage_block_adaptive": self._stage_block_adaptive,
                "stage_block_sizes": list(self._stage_block_sizes),
                "stage_block_switch_s": list(self._stage_block_switch_s),
                "block_hist": dict(self._block_hist),
                "n_block_syncs": self.n_block_syncs,
                "n_sync_skipped": self.n_sync_skipped,
                "sync_wait_s": self.n_sync_wait_s,
                "forward_heuristic": {
                    "enabled": bool(getattr(self.cfg, "forward_heuristic", False)),
                    "mode": str(getattr(
                        self.cfg, "forward_heuristic_mode", "sa")),
                    "lp_p0": float(getattr(self.cfg, "heur_lp_p0", 0.5)),
                    "lp_p1": float(getattr(self.cfg, "heur_lp_p1", 0.95)),
                    "ramp_iters": int(getattr(
                        self.cfg, "heur_ramp_iters", 1000)),
                    "alpha": float(getattr(self.cfg, "heur_alpha", 1.0)),
                    "target_frac": float(getattr(
                        self.cfg, "heur_target_frac", 1.0)),
                    "sa_iters": int(getattr(self.cfg, "heur_sa_iters", 20)),
                    "sa_t0_frac": float(getattr(
                        self.cfg, "heur_sa_t0_frac", 0.02)),
                    "sa_cooling": float(getattr(
                        self.cfg, "heur_sa_cooling", 0.9)),
                    "sa_step": float(getattr(self.cfg, "heur_sa_step", 1.0)),
                    "n_trials": self.n_heur_trials,
                    "n_used": self.n_heur_used,
                    "n_fail": self.n_heur_fail,
                    "n_sa_moves": self.n_sa_moves,
                    "n_sa_accepted": self.n_sa_accepted,
                    "sa_time_s": self.sa_time_s,
                    "q_dual_trials": bool(getattr(
                        self.cfg, "heur_q_dual_trials", False)),
                    "n_qp_perturbed": self.n_qp_perturbed,
                    "n_qp_base": self.n_qp_base,
                    "n_qp_solves": self.n_qp_solves,
                    "n_qp_extra_trials": self.n_qp_extra_trials,
                },
                "n_cut_dropped": self.n_cut_dropped,
                "integer_mode": bool(getattr(self.cfg, "integer_enabled", False)),
                "integer_scope": str(getattr(self.cfg, "integer_scope", "none")),
                "integer_rounding": str(getattr(self.cfg, "integer_rounding", "none")),
                "n0_mode": str(getattr(self.prob, "n0_mode", "continuous")),
                "forward_solver": ("mip" if getattr(self.cfg, "integer_enabled", False)
                                   else "lp"),
                "lower_bound_solver": "lp",
                "cut_lift": {
                    "enabled": self.cut_lift_enabled,
                    "scope": self._cl_scope,
                    "last_k": self._cl_last_k,
                    "lift_stages": sorted(self._cl_stages),
                    "mode": self._cl_mode,
                    "value": str(getattr(self.cut_lift_cfg, "value", "exact")),
                    "time_limit": float(getattr(
                        self.cut_lift_cfg, "time_limit", 5.0)),
                    "rel_gap": float(getattr(
                        self.cut_lift_cfg, "rel_gap", 0.0)),
                    "abs_gap": float(getattr(
                        self.cut_lift_cfg, "abs_gap", 0.0)),
                    "safety_margin": float(getattr(
                        self.cut_lift_cfg, "safety_margin", 1e-6)),
                    "lb_gain_threshold": self._cl_threshold,
                    "min_round": self._cl_min_round,
                    "n_lift_cuts": self.n_lift_cuts,
                    "n_lift_mip": self.n_lift_mip,
                    "n_fallback": self.n_lift_fallback,
                    "lift_rounds": list(self._lift_rounds),
                    "lift_gains": list(self._lift_gains),
                }}

    def checkpoint(self):
        if self.results_dir is None:
            return None
        from .io import save_cuts
        cuts = (self.cut_pool.all_by_stage()
                if self.cut_pool is not None else self.cuts)
        return save_cuts(self.results_dir, cuts[:self.T + 1], self.meta())

    def train(self, n_iter: int, *, max_seconds: float = 0.0,
              start_iter: int = 1, verbose: bool = True) -> list[dict]:
        self._start_workers()
        history = []
        t0 = time.time()
        self._t0 = t0
        last_ckpt = t0
        next_k = int(start_iter)
        last_k = int(start_iter) + int(n_iter) - 1
        outstanding = 0
        accepting = True
        try:
            for wid in range(self.n_workers):
                if next_k > last_k:
                    accepting = False
                    break
                self._dispatch(wid, next_k)
                next_k += 1
                outstanding += 1
            while outstanding > 0:
                kind, wid, k, *rest = self._result_q.get()
                if kind == "sync":
                    self._merge(wid, rest[0])
                    self._send_ack(wid)
                    self.n_block_syncs += 1
                    continue
                outstanding -= 1
                if kind == "err":
                    raise RuntimeError(f"async worker {wid} iter {k} failed:\n{rest[0]}")
                if len(rest) >= 6:
                    (total, n_solves, new_cuts, lift_stats, sync_stats,
                     heur_stats) = rest[:6]
                elif len(rest) >= 5:
                    total, n_solves, new_cuts, lift_stats, sync_stats = rest[:5]
                    heur_stats = {}
                elif len(rest) >= 4:
                    total, n_solves, new_cuts, lift_stats = rest[:4]
                    sync_stats, heur_stats = {}, {}
                else:
                    total, n_solves, new_cuts = rest[:3]
                    lift_stats, sync_stats, heur_stats = {}, {}, {}
                self._merge(wid, new_cuts)
                self.n_sync_wait_s += float(sync_stats.get("wait_s", 0.0))
                self.n_sync_skipped += int(sync_stats.get("skipped", 0))
                self.n_heur_trials += int(heur_stats.get("trials", 0))
                self.n_heur_used += int(heur_stats.get("used", 0))
                self.n_heur_fail += int(heur_stats.get("fail", 0))
                self.n_sa_moves += int(heur_stats.get("sa_moves", 0))
                self.n_sa_accepted += int(heur_stats.get("sa_accepted", 0))
                self.sa_time_s += float(heur_stats.get("sa_time_s", 0.0))
                self.n_qp_perturbed += int(heur_stats.get("qp_perturbed", 0))
                self.n_qp_base += int(heur_stats.get("qp_base", 0))
                self.n_qp_solves += int(heur_stats.get("qp_solves", 0))
                self.n_qp_extra_trials += int(
                    heur_stats.get("qp_extra_trials", 0))
                self.n_solves += int(n_solves)
                self.n_iterations += 1
                self.last_iter = max(self.last_iter, int(k))
                lift_used = int(lift_stats.get("n_cuts", 0)) > 0
                lift_gain = float(lift_stats.get("gain_sum", 0.0))
                if lift_used:
                    self.n_lift_cuts += int(lift_stats.get("n_cuts", 0))
                    self.n_lift_mip += int(lift_stats.get("n_mip", 0))
                    self.n_lift_fallback += int(lift_stats.get("n_fb", 0))
                    self._lift_rounds.append(int(k))
                    self._lift_gains.append(lift_gain)
                lb = self._lower_bound()
                self._prev_lb_delta = (None if self._last_lb is None
                                       else lb - self._last_lb)
                self._last_lb = lb
                elapsed = time.time() - t0
                blk = self._iter_block.pop(int(k), self._block_size_now())
                self._block_hist[str(blk)] = self._block_hist.get(str(blk), 0) + 1
                rec = {"iter": int(k), "lb": lb, "sim": float(total),
                       "time": elapsed, "solves": self.n_solves,
                       "n_cuts": sum(len(c) for c in self.cuts[1:self.T + 1]),
                       "n_workers": self.n_workers,
                       "paths_per_task": self.paths_per_task,
                       "block_size": int(blk),
                       "sync_wait_s": float(sync_stats.get("wait_s", 0.0)),
                       "sync_skipped": int(sync_stats.get("skipped", 0)),
                       "lift_used": bool(lift_used),
                       "lift_gain": lift_gain,
                       "lift_fallback": int(lift_stats.get("n_fb", 0))}
                history.append(rec)
                if self.on_iter is not None:
                    self.on_iter(rec)
                if verbose:
                    lift_txt = ""
                    if self.cut_lift_enabled:
                        lift_txt = (f" | lift={int(lift_used)}"
                                    f" gain={lift_gain:+.1f}"
                                    f" fb={rec['lift_fallback']}")
                    blk_txt = f" | B={blk}" if self._stage_block_adaptive else ""
                    self.log(f"  async iter {k:5d} | LB={lb:14.4f} | "
                             f"sim={total:14.4f} | cuts={rec['n_cuts']:6d} | "
                             f"{elapsed:7.1f} s{blk_txt}{lift_txt}")
                ckpt_iv = float(getattr(self.cfg, "checkpoint_interval", 0) or 0)
                if (self.results_dir is not None and ckpt_iv > 0
                        and time.time() - last_ckpt >= ckpt_iv):
                    self.checkpoint()
                    last_ckpt = time.time()
                if (accepting and next_k <= last_k
                        and (not max_seconds or elapsed < max_seconds)):
                    self._dispatch(wid, next_k)
                    next_k += 1
                    outstanding += 1
                elif self._task_q[wid] is not None:
                    # 不再派新任务，worker 处理完当前任务后退出
                    self._task_q[wid].put(None)
                    self._task_q[wid] = None
            if not accepting:
                pass
            for q in self._task_q:
                if q is not None:
                    q.put(None)
            for p in self._proc:
                p.join(timeout=10)
            if self._lb_sub is not None:
                self._lb_sub.free()
                self._lb_sub = None
            if self.results_dir is not None:
                self.checkpoint()
        finally:
            for q in self._task_q:
                if q is not None:
                    try:
                        q.put(None)
                    except Exception:
                        pass
            for p in self._proc:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=2)
            self._closed = True
        return history

    def close(self):
        if self._closed:
            return
        for q in self._task_q:
            if q is not None:
                try:
                    q.put(None)
                except Exception:
                    pass
        for p in self._proc:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2)
        self._closed = True
