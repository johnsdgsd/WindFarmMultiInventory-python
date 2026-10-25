# engine.py — SDDP 引擎：前向并行、后向阶段串行、割池、下界与训练循环。与问题解耦。
"""算法骨架（阶段 0 = 虚拟首阶段，1..T 为真实周期；cuts[t] 近似 V_t）：

    每轮：
      (a) 前向 —— 采样 N 条路径，每路径独立走完 T+1 期解 LP 得策略，
          收齐试探点 trial[t] 与前向总成本（上界估计）；
      (b) 后向 —— **按阶段串行，期内并行**：t 从 T 递减到 1，把该阶段的 N 个试探点
          平均分给 worker，每 worker 对一个试探点枚举场景解 LP 取对偶、平均成一条割；
          本阶段所有割算完再一起入库，然后才推进到 t−1；
      (c) 下界 —— 阶段 0 子问题（割口径与后向一致）。

**为什么必须"期间串行"**：旧实现轮初打一份割快照、全部 (期 × 试探点) 任务一次铺满，
于是期 t 用不上同轮期 t+1 的新割 —— 终端信息每轮最多向前传播一期，T=200 时下界上百轮
不动（实测 13 轮全 0）。改成期间串行后，期 t 立即用上同轮期 t+1 的新割，一次反向扫描
就把信息传遍全视界。

**并行度来自试探点数**：n_forward_paths 默认 = 进程数 × 1 ⇒ 每个阶段恰好 N = 进程数 个
试探点 ⇒ 每个 worker 每阶段正好产出 1 条割，无空转。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .cut import Cut, trivial_cut
from .io import TrainingLog, save_cuts
from .payload import StagePayload, put_cuts
from .worker import (backward_point_task, forward_path_task, init_worker,
                     lower_bound_task)


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
        self.forward_sampling = str(getattr(cfg, "forward_sampling", "random"))
        self.log = log if log is not None else (lambda *a, **k: None)

        self.n_scenes = len(prob.pool)
        # 按每场景的总失效强度排序，stratified 前向按分位分层取场景
        scores = []
        G = int(getattr(prob, "G", 0))
        for _, theta in prob.pool:
            th = np.asarray(theta, dtype=float)
            scores.append(float(th[:, :, G].sum(axis=1).mean()))
        self.scene_order = np.argsort(np.asarray(scores))
        _k = int(getattr(cfg, "n_backward_samples", 0))
        if 0 < _k < self.n_scenes:
            self.log(
                f"[警告] 后向只抽 {_k}/{self.n_scenes} 个场景：割只对该子样本子问题成立，"
                f"LB 可能高于全池 UB；要全池有效 LB 请把 n_backward_samples 设为 0")
        self.cuts: list[list[Cut]] = [[] for _ in range(self.T + 1)]
        for t in range(1, self.T + 1):
            self.cuts[t].append(trivial_cut(t, self.initial_state, self.state_dim))

        self.n_solves = 0
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
        self._pts: list[list] = [[] for _ in range(self.T + 1)]   # 历史试探点
        self._mx: list[list] = [[] for _ in range(self.T + 1)]    # 该点处当前最大割值
        self._rc: list[list] = [[] for _ in range(self.T + 1)]    # 保留的最旧最大割
        self._pool = None
        if self.n_processes > 1:
            self._pool = _default_ctx().Pool(
                self.n_processes, initializer=init_worker,
                initargs=(prob, make_sub, cfg, self.T, self.initial_state,
                          self.results_dir))
        else:
            init_worker(prob, make_sub, cfg, self.T, self.initial_state,
                        self.results_dir)

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
        if self.forward_sampling != "stratified":
            return [self.rng.integers(0, self.n_scenes, size=self.T).tolist()
                    for _ in range(n_paths)]
        # 每条路径负责一个失效强度分位层；层内按阶段独立抽样。
        strata = np.array_split(self.scene_order, n_paths)
        out = []
        for stratum in strata:
            pool = stratum if len(stratum) else self.scene_order
            out.append(self.rng.choice(pool, size=self.T, replace=True).tolist())
        return out

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
        try:
            tasks = [(self.round_id, k, list(sp), refs, record)
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
            points = trial[t]
            if not points:
                continue
            inds = self._sample_scenes(len(points))
            blk = put_cuts(self.cuts[t + 1] if t < self.T else [], self.state_dim)
            try:
                tasks = [(self.round_id, t, i, np.asarray(x, dtype=float), inds[i],
                          blk.ref) for i, x in enumerate(points)]
                results = self._map(backward_point_task, tasks)
            finally:
                blk.close()
            self.n_solves += sum(len(s) for s in inds)
            new_cuts = []
            for (_, _, _, x, _, _), (beta, obj) in zip(tasks, results):
                c = Cut(float(obj) - float(beta @ x), beta, x, t, float(obj), it)
                self.cuts[t].append(c)
                new_cuts.append(c)
                n_added += 1
            if self.cut_selection == "lml1":
                self._lml1_update(t, points, new_cuts)
        return n_added

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
        n = 0
        for t in range(1, self.T + 1):
            for c in self.cuts[t]:
                if c.it == 0:
                    continue
                x = np.asarray(c.state, dtype=float).ravel()
                if x.size != self.state_dim:
                    raise ValueError(
                        f"stage {t}: 割的 state 维度 {x.size} ≠ {self.state_dim}")
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
        blk = put_cuts(self.cuts[1], self.state_dim)
        try:
            lb = float(self._map(lower_bound_task, [(self.round_id, blk.ref)])[0])
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
                self.last_iter = k
                elapsed = time.time() - t0
                rec = {"iter": k, "lb": lb, "lb_best": self.lb_best,
                       "sim": sim, "time": elapsed,
                       "solves": self.n_solves, "mip_fallback": n_fb,
                       "n_cuts": sum(len(c) for c in self.cuts[1:]),
                       "n_dropped": self.n_cut_dropped}
                history.append(rec)
                if self.on_iter is not None:
                    self.on_iter(rec)
                if verbose:
                    log(f"  轮 {k:5d} | LB = {lb:14.4f} | 前向均值 = {sim:14.4f} "
                        f"| 割数 = {rec['n_cuts']:6d} | {elapsed:7.1f} s"
                        + (f" | MIP 回退 {n_fb}" if n_fb else ""))
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

    def meta(self) -> dict:
        return {"T": self.T, "state_dim": self.state_dim, "seed": self.seed,
                "model": "markov_grade",
                "G": getattr(self.prob, "G", None),
                "lambda0": getattr(self.prob, "lambda0", None),
                "dirichlet_s": getattr(self.prob, "dirichlet_s", None),
                "n_scenes": self.n_scenes,
                "n_iterations": self.last_iter,
                "n_cuts": sum(len(c) for c in self.cuts[1:]),
                "n_solves": self.n_solves,
                "lb_best": self.lb_best}

    def open_log(self, name: str = "sddp_training_log.csv") -> TrainingLog:
        if self.results_dir is None:
            raise RuntimeError("未指定 results_dir，无法写训练日志")
        return TrainingLog(self.results_dir, name)

    # --------------------------------------------- 前向割块（仿真/外部复用）
    def forward_refs(self):
        """当前割表的每阶段 ref（前向路径任务用）。调用方负责 close。"""
        return [put_cuts(self.cuts[t + 1], self.state_dim) for t in range(0, self.T)]

    # ------------------------------------------------------ 仿真用采样器
    def sim_scene_paths(self, n_paths: int, seed: int | None = None):
        """仿真/复现用的场景路径。与训练共用 rng 会打乱训练随机流，故单独一个 Generator。"""
        base = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(base)
        return [rng.integers(0, self.n_scenes, size=self.T).tolist()
                for _ in range(n_paths)]
