# training.py — SDDP 引擎：多进程前向/后向、割生成、下界与训练主循环。与问题解耦。
"""算法骨架（与 SDDP.jl 同构）：

    阶段 0..T：0 为虚拟首阶段（钉初值 + 决定首期订货），1..T 为真实周期。
    cuts[t] 近似 V_t；第 t-1 期子问题用 cuts[t] 约束其 theta。

    每轮：
      (a) 前向 pass —— 采样路径，逐期解 MIP 得策略，得到试态点 trial[t]；
      (b) 后向 pass —— 对每个试态点枚举场景，逐场景解 LP 松弛取 .dj，
          取场景平均得期望次梯度 → 一条割；期与期之间按批次，互不依赖；
      (c) 下界 —— 阶段 0 子问题（mip=False，与割的口径一致）。

为什么按批次后向（而非边算边加割）：批次后向后，同一轮的 T 个期任务彼此独立，
可一次性铺满进程池；割的有效性不依赖加入顺序（cuts[t+1] 在本轮内冻结，仍是 V_{t+1}
的合法下界近似），只是收敛速度略有差异。

两种求解模式（guide §6.2 路线 A，零侵入）：
    前向/仿真 → mip=True   取整数策略；
    后向/下界 → mip=False  读 LP 松弛的对偶（COPT_DLL 在 MIP 下会拒绝返回对偶）。
代价：割是「松弛值函数」的下界，对整数问题不保证严格有效 —— 与 Julia 版
handler=:continuous 同源，作为对照观察（见 docs/建模与实现.md）。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .cut import Cut, pack, trivial_cut
from .io import TrainingLog, save_cuts
from .solver import SolverError, export_lp, make_solver

# worker 侧全局状态：spawn 下每个子进程由 initializer 装配一次，
# 场景池只随进程创建传一次，任务里只带场景下标。
_W: dict = {}


def _init_worker(prob, build_stage, T: int, state_dim: int, initial_state,
                 use_copt: bool, results_dir) -> None:
    _W["prob"] = prob
    _W["build"] = build_stage
    _W["T"] = int(T)
    _W["dim"] = int(state_dim)
    _W["init"] = np.round(np.asarray(initial_state, dtype=float), 6)   # 与子问题口径一致
    _W["results_dir"] = results_dir
    _W["solver_mip"] = make_solver(mip=True, use_copt=use_copt)
    _W["solver_lp"] = make_solver(mip=False, use_copt=use_copt)


def _build(t: int, state, scene, cuts_next, it: int = 0):
    return _W["build"](t, state, scene, cuts_next, _W["prob"], it)


def _fail(where: str, t, scene_ind, sub, exc) -> SolverError:
    """求解失败的兜底：把现场导出成 LP，供 AnalysisModel_IIS.py 分析冲突集。"""
    out = _W.get("results_dir")
    if out is not None:
        path = Path(out) / f"subproblem_{where}_t{t}_s{scene_ind}.lp"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            export_lp(sub.prob, path)
        except Exception:
            path = None
    else:
        path = None
    return SolverError(f"{where} 求解失败：期 {t}，场景 {scene_ind}，现场 LP = {path}（{exc}）")


# ------------------------------------------------------------------ 路径任务
def _path_task(task):
    """走一条完整路径：逐期解 MIP 得策略，读期末状态作为下一期入库状态。

    stage 0 无噪声（钉初值 + 决定 q_0）；stage t>=1 先用场景 ω_t 解子问题。
    返回 (states[1..T], costs[1..T], total_cost, records)。
    """
    cuts_payload, scene_inds, record = task
    T, prob = _W["T"], _W["prob"]
    x = _W["init"].copy()
    states, costs, records = [], [], ([] if record else None)

    for t in range(0, T + 1):
        cuts_next = cuts_payload[t + 1] if t < T else None
        scene = None if t == 0 else prob.pool[scene_inds[t - 1]]
        sub = _build(t, x, scene, cuts_next)
        try:
            sub.solve(_W["solver_mip"])
        except RuntimeError as exc:
            raise _fail("forward", t, -1 if scene is None else scene_inds[t - 1], sub, exc)
        x = sub.out_state_vector()
        if t >= 1:
            states.append(x.copy())
            costs.append(sub.stage_cost())
            if record:
                records.append({**sub.decision_values(), "cost": costs[-1],
                                "scene": int(scene_inds[t - 1]) if scene_inds else -1})
    return states, costs, float(sum(costs)), records


# ------------------------------------------------------------------ 后向任务
def _backward_task(task):
    """一个试态点的后向：逐场景解 LP 松弛，取 .dj 平均得期望次梯度。

    Q(x̄) = E_ω[Q(x̄;ω)]，而 ∂(Σ p_ω Q(x̄;ω)) = Σ p_ω ∂Q(x̄;ω)（凸函数次微分可加），
    故场景平均的 .dj 就是期望值函数的一个合法次梯度。场景等概率 ⇒ 直接取均值。
    """
    t, state_vec, scene_inds, cuts_next = task
    prob = _W["prob"]
    dim = _W["dim"]
    grads = np.empty((len(scene_inds), dim))
    objs = np.empty(len(scene_inds))
    for k, si in enumerate(scene_inds):
        sub = _build(t, state_vec, prob.pool[si], cuts_next)
        try:
            sub.solve(_W["solver_lp"])
        except RuntimeError as exc:
            raise _fail("backward", t, si, sub, exc)
        grads[k] = sub.gradient()
        objs[k] = sub.obj
    return grads.mean(axis=0), float(objs.mean())


# ---------------------------------------------------------------- 下界任务
def _lower_bound_task(task):
    cuts_payload = task
    sub = _build(0, _W["init"], None, cuts_payload[1])
    try:
        sub.solve(_W["solver_lp"])
    except RuntimeError as exc:
        raise _fail("lower_bound", 0, -1, sub, exc)
    return float(sub.obj)


class SDDPEngine:
    """问题无关的 SDDP 引擎。

    参数
    ----
    prob          : 问题数据；须有 `.pool`（场景序列，等概率）
    build_stage   : 模块顶层函数 (t, state_vec, scene, cuts_next, prob, it) -> 子问题
    T             : 真实周期数（阶段 0..T 共 T+1 个）
    state_dim     : 状态向量维度
    initial_state : 初始状态向量
    n_processes   : 进程数；1 = 串行（便于调试），None = 核数
    """

    def __init__(self, prob, build_stage, *, T: int, state_dim: int, initial_state,
                 n_processes: int | None = None, seed: int = 0, use_copt: bool = True,
                 log=print, results_dir=None, checkpoint_interval: float = 600.0):
        self.prob = prob
        self.build_stage = build_stage
        self.T = int(T)
        self.state_dim = int(state_dim)
        self.initial_state = np.asarray(initial_state, dtype=float).ravel()
        if self.initial_state.size != self.state_dim:
            raise ValueError(f"initial_state 维度 {self.initial_state.size} ≠ {self.state_dim}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.use_copt = bool(use_copt)
        self.log = log if log is not None else (lambda *a, **k: None)
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.checkpoint_interval = float(checkpoint_interval)

        np_ = n_processes if n_processes is not None else (mp.cpu_count() or 1)
        self.n_processes = max(1, int(np_))

        self.n_scenes = len(prob.pool)
        self.cuts: list[list[Cut]] = [[] for _ in range(self.T + 1)]
        for t in range(1, self.T + 1):
            self.cuts[t].append(trivial_cut(t, self.initial_state, self.state_dim))

        self.n_solves = 0
        self.on_iter = None       # 可选回调 on_iter(rec)，逐轮落 CSV / 自定义进度条
        self._pool = None
        if self.n_processes > 1:
            self._pool = mp.get_context("spawn").Pool(
                self.n_processes, initializer=_init_worker,
                initargs=(prob, build_stage, self.T, self.state_dim, self.initial_state,
                          self.use_copt, self.results_dir))
        else:
            _init_worker(prob, build_stage, self.T, self.state_dim, self.initial_state,
                         self.use_copt, self.results_dir)

    # ---------------------------------------------------------------- 调度
    def _map(self, fn, tasks):
        if self._pool is None:
            return [fn(t) for t in tasks]
        # 任务粒度较细（后向每个试态点一次求解）→ 动态分块，压 IPC 次数
        chunksize = max(1, len(tasks) // (4 * self.n_processes))
        return self._pool.map(fn, tasks, chunksize=chunksize)

    def _cuts_payload(self) -> list:
        dim = self.state_dim
        return [None] + [pack(self.cuts[t], dim) for t in range(1, self.T + 1)]

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

    def forward(self, n_paths: int = 1, record: bool = False):
        """采样路径并求解。返回 (trial, results)：trial[t] 是第 t 期入库状态列表。"""
        if n_paths < 1:
            raise ValueError("n_paths 至少为 1")
        scene_paths = self.sample_scene_paths(n_paths)
        cuts_payload = self._cuts_payload()
        tasks = [(cuts_payload, scene_paths[k], record) for k in range(n_paths)]
        results = self._map(_path_task, tasks)
        self.n_solves += n_paths * (self.T + 1)
        trial = [[] for _ in range(self.T + 1)]
        for states, _, _, _ in results:
            for t in range(1, self.T + 1):
                trial[t].append(states[t - 1])
        return trial, results

    # ------------------------------------------------------------ 后向 pass
    def backward(self, trial, it: int = 0, n_samples: int | None = None) -> int:
        """对每个试态点生成一条割。n_samples=None 表示枚举全部场景。

        整轮后向共用一份割快照：期与期在批内互不依赖，cuts[t+1] 在本轮冻结。
        """
        payload = self._cuts_payload()
        tasks = []
        for t in range(self.T, 0, -1):
            cuts_next = payload[t + 1] if t < self.T else None
            for x in trial[t]:
                if n_samples is None or n_samples >= self.n_scenes:
                    inds = list(range(self.n_scenes))
                else:
                    inds = sorted(self.rng.choice(self.n_scenes, size=int(n_samples),
                                                  replace=False).tolist())
                tasks.append((t, np.asarray(x, dtype=float), inds, cuts_next))
        if not tasks:
            return 0
        results = self._map(_backward_task, tasks)
        self.n_solves += sum(len(t[2]) for t in tasks)

        n_added = 0
        for (t, x, _, _), (beta, obj) in zip(tasks, results):
            alpha = obj - float(beta @ x)
            self.cuts[t].append(Cut(alpha, beta, x, t, obj, it))
            n_added += 1
        return n_added

    # ------------------------------------------------------------------ 下界
    def lower_bound(self) -> float:
        """阶段 0 子问题值。割只增不减 ⇒ LB 单调不减，且不依赖随机数。"""
        lb = float(self._map(_lower_bound_task, [self._cuts_payload()])[0])
        self.n_solves += 1
        return lb

    # -------------------------------------------------------------- 主循环
    def train(self, n_iter: int, *, n_forward_paths: int = 120,
              n_backward_samples: int | None = None, start_iter: int = 1,
              verbose: bool = True) -> list[dict]:
        history: list[dict] = []
        t0 = time.time()
        last_ckpt = t0
        log = self.log
        try:
            for k in range(start_iter, start_iter + n_iter):
                trial, results = self.forward(n_paths=n_forward_paths)
                sim = float(np.mean([r[2] for r in results]))
                self.backward(trial, it=k, n_samples=n_backward_samples)
                lb = self.lower_bound()
                elapsed = time.time() - t0
                rec = {"iter": k, "lb": lb, "sim": sim, "time": elapsed,
                       "solves": self.n_solves,
                       "n_cuts": sum(len(c) for c in self.cuts[1:])}
                history.append(rec)
                if self.on_iter is not None:
                    self.on_iter(rec)
                if verbose:
                    log(f"  轮 {k:5d} | LB = {lb:14.4f} | 前向均值 = {sim:14.4f} "
                        f"| 割数 = {rec['n_cuts']:6d} | {elapsed:7.1f} s")
                if self.results_dir is not None and \
                        time.time() - last_ckpt >= self.checkpoint_interval:
                    self.checkpoint()
                    last_ckpt = time.time()
        except KeyboardInterrupt:
            log("已中断，正在保存割 …")
        finally:
            if self.results_dir is not None:
                self.checkpoint()
        if verbose:
            log(f"训练用时 {time.time() - t0:.1f} s，累计求解 {self.n_solves} 次")
        return history

    def checkpoint(self) -> Path | None:
        if self.results_dir is None:
            return None
        return save_cuts(self.results_dir, self.cuts, self.meta())

    def meta(self) -> dict:
        return {"T": self.T, "state_dim": self.state_dim, "seed": self.seed,
                "n_scenes": self.n_scenes, "n_cuts": sum(len(c) for c in self.cuts[1:]),
                "n_solves": self.n_solves}

    # --------------------------------------------------------------- 日志
    def open_log(self, name: str = "sddp_training_log.csv") -> TrainingLog:
        if self.results_dir is None:
            raise RuntimeError("未指定 results_dir，无法写训练日志")
        return TrainingLog(self.results_dir, name)

    def log_row(self, f: TrainingLog, rec: dict) -> None:
        f.row(rec["iter"], rec["sim"], rec["lb"], rec["time"], rec["solves"])
