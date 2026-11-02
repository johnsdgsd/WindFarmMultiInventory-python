# config.py — 读 TOML 配置并校验。默认值只存在于 config/*.toml，代码里不复写第二份。
"""为什么"缺键即报错"：以前默认值散在 run.py 的 main() 签名里，命令行位置参数会盖掉
它，出现过"改了 120 却不生效"。现在唯一事实源是 TOML —— 键不存在就抛错，绝不静默兜底。

两份配置各管一事：
    problem.toml → ProblemConfig   问题本身的参数（维度/机队/成本/退化/协变量/场景池）
    solver.toml  → SolverConfig    怎么解（轮数/并行/仿真/容差/输出）

`resolve()` 把两个"相对值"换算成实际值：
    n_processes = 0                      → 物理核数 // 2
    n_forward_paths = forward_paths_per_process × n_processes
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ 取键助手
def _req(d: dict, key: str, where: str):
    if key not in d:
        raise KeyError(f"配置 {where} 缺少必填项 {key!r}。默认值请写在 config/*.toml，"
                       f"不要在代码里兜底")
    return d[key]


def _opt(d: dict, key: str, default, where: str):
    """可选键：只用于向后兼容的新增字段（旧结果目录的配置没有这些键）。
    决定问题结构的参数仍必须走 _req，不得在代码里兜底。"""
    return d[key] if key in d else default


def _section(d: dict, name: str, where: str) -> dict:
    sec = _req(d, name, where)
    if not isinstance(sec, dict):
        raise TypeError(f"配置 {where} 的 [{name}] 必须是表（节），实际 {type(sec).__name__}")
    return sec


def physical_cores() -> int:
    """物理核数。psutil 不可用时退到逻辑核的一半（可能高估，但不会失控）。"""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


# ------------------------------------------------------------------ 问题配置
@dataclass
class ProblemConfig:
    # 维度
    T: int
    G: int              # 运行健康等级数；失效态为 G
    N_R: int
    L: int
    # 机队
    N: list
    Qmax: list
    Imax: list
    fleet_tol: float          # fleet/dem 右端的数值容差（台）
    state_round_digits: int   # 状态流转时 n 块保留的小数位
    state_repair_tol: float   # 只修正不超过该值的 Σn 超额（台）
    # 初始等级分布（长度 G、非负、和为 1）；[1,0,..] = 全新机队
    init_mix: list
    # 各仓初始可用库存（长度 N_R，台）；旧配置缺省全 0
    I0: list
    # 成本
    C: float
    h: float
    Cs: float
    atc: float
    tc: float
    # 退化/转移
    lambda0: float       # Poisson 基准均值（月度）
    dirichlet_s: float   # 逐行 Dirichlet 集中度
    p_digits: int        # 转移矩阵保留小数位
    beta_cov: list
    z_bar: list
    # 协变量分布（cov["lambda"] 是启停次数均值，与 lambda0 不是同一个量）
    cov: dict
    # 场景池
    pool_size: int
    seed: int
    # 可选
    priority: dict | None = None
    # 来源留痕（拷贝进结果目录时用）
    source_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        if self.T <= 0:
            raise ValueError(f"[dimension] T 必须为正，实际 {self.T}")
        if self.G < 2:
            raise ValueError(f"[dimension] G（运行等级数）必须 ≥ 2，实际 {self.G}")
        if self.N_R <= 0:
            raise ValueError(f"[dimension] N_R 必须为正，实际 {self.N_R}")
        if self.L < 2:
            raise ValueError(f"[dimension] 当前实现要求 L ≥ 2（在途管 L−1 ≥ 1），实际 {self.L}")

        for name in ("N", "Qmax", "Imax"):
            v = getattr(self, name)
            if len(v) != self.N_R:
                raise ValueError(f"[fleet] {name} 长度 {len(v)} ≠ N_R={self.N_R}；"
                                 f"改 N_R 时必须同步改写这三个列表")
            if any(x < 0 for x in v):
                raise ValueError(f"[fleet] {name} 不得含负值：{v}")
        if any(int(x) == 0 for x in self.Qmax):
            raise ValueError(f"[fleet] Qmax 不得含 0（仓库无法订货）：{self.Qmax}")
        if not (0.0 <= float(self.fleet_tol) <= 1e-3):
            raise ValueError(f"[fleet] fleet_tol 必须在 [0, 1e-3] 台，"
                             f"实际 {self.fleet_tol}")
        if not (0 <= int(self.state_round_digits) <= 12):
            raise ValueError(f"[fleet] state_round_digits 必须在 0..12，"
                             f"实际 {self.state_round_digits}")
        if not (float(self.fleet_tol) <= float(self.state_repair_tol) <= 1e-2):
            raise ValueError(f"[fleet] state_repair_tol 必须在 "
                             f"[fleet_tol, 1e-2] 台，实际 {self.state_repair_tol}")

        if len(self.init_mix) != self.G:
            raise ValueError(f"[fleet] init_mix 长度 {len(self.init_mix)} ≠ G={self.G}")
        if any(float(x) < 0 for x in self.init_mix):
            raise ValueError(f"[fleet] init_mix 不得含负值：{self.init_mix}")
        s = float(sum(self.init_mix))
        if abs(s - 1.0) > 1e-8:
            raise ValueError(f"[fleet] init_mix 必须和为 1，实际 {s}")

        if len(self.I0) != self.N_R:
            raise ValueError(f"[fleet] I0 长度 {len(self.I0)} ≠ N_R={self.N_R}")
        if any(float(x) < 0 for x in self.I0):
            raise ValueError(f"[fleet] I0 不得含负值：{self.I0}")
        bad = [r + 1 for r, (x, cap) in enumerate(zip(self.I0, self.Imax))
               if float(x) > float(cap)]
        if bad:
            raise ValueError(f"[fleet] I0 超过 Imax：仓库 {bad}，"
                             f"I0={self.I0}, Imax={self.Imax}")

        if self.lambda0 <= 0:
            raise ValueError(f"[degradation] lambda0 必须为正，实际 {self.lambda0}")
        if self.dirichlet_s <= 0:
            raise ValueError(f"[degradation] dirichlet_s 必须为正，实际 {self.dirichlet_s}")
        if not (0 <= int(self.p_digits) <= 6):
            raise ValueError(f"[degradation] p_digits 必须在 0..6，实际 {self.p_digits}")
        if len(self.beta_cov) != 3 or len(self.z_bar) != 3:
            raise ValueError(f"[degradation] PHM 要求 3 维协变量，实际 "
                             f"beta_cov={self.beta_cov}、z_bar={self.z_bar}")

        need = ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")
        missing = [k for k in need if k not in self.cov]
        if missing:
            raise KeyError(f"[covariate] 缺少 {missing}")
        for k in ("sigma_T", "lambda", "alpha_L", "beta_L"):
            if self.cov[k] <= 0:
                raise ValueError(f"[covariate] {k} 必须为正，实际 {self.cov[k]}")

        if self.pool_size < 2:
            raise ValueError(f"[scenario] pool_size 至少为 2，实际 {self.pool_size}")

        if self.priority is not None:
            keys = sorted(int(k) for k in self.priority)
            if keys != list(range(1, self.N_R + 1)):
                raise ValueError(f"[priority] 必须恰覆盖仓库 1..N_R（N_R={self.N_R}），"
                                 f"实际键 {keys}")


def load_problem_config(path) -> ProblemConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    dim = _section(raw, "dimension", path.name)
    fleet = _section(raw, "fleet", path.name)
    cost = _section(raw, "cost", path.name)
    deg = _section(raw, "degradation", path.name)
    cov = _section(raw, "covariate", path.name)
    scen = _section(raw, "scenario", path.name)
    prio = raw.get("priority")
    N_R = int(_req(dim, "N_R", "problem.toml/dimension"))
    return ProblemConfig(
        T=int(_req(dim, "T", "problem.toml/dimension")),
        G=int(_req(dim, "G", "problem.toml/dimension")),
        N_R=N_R,
        L=int(_req(dim, "L", "problem.toml/dimension")),
        N=[int(v) for v in _req(fleet, "N", "problem.toml/fleet")],
        Qmax=[int(v) for v in _req(fleet, "Qmax", "problem.toml/fleet")],
        Imax=[int(v) for v in _req(fleet, "Imax", "problem.toml/fleet")],
        fleet_tol=float(_req(fleet, "fleet_tol", "problem.toml/fleet")),
        state_round_digits=int(_req(fleet, "state_round_digits",
                                    "problem.toml/fleet")),
        state_repair_tol=float(_req(fleet, "state_repair_tol",
                                    "problem.toml/fleet")),
        init_mix=[float(v) for v in _req(fleet, "init_mix", "problem.toml/fleet")],
        I0=[float(v) for v in _opt(fleet, "I0", [0.0] * N_R,
                                   "problem.toml/fleet")],
        C=float(_req(cost, "C", "problem.toml/cost")),
        h=float(_req(cost, "h", "problem.toml/cost")),
        Cs=float(_req(cost, "Cs", "problem.toml/cost")),
        atc=float(_req(cost, "atc", "problem.toml/cost")),
        tc=float(_req(cost, "tc", "problem.toml/cost")),
        lambda0=float(_req(deg, "lambda0", "problem.toml/degradation")),
        dirichlet_s=float(_req(deg, "dirichlet_s", "problem.toml/degradation")),
        p_digits=int(_req(deg, "p_digits", "problem.toml/degradation")),
        beta_cov=[float(v) for v in _req(deg, "beta_cov", "problem.toml/degradation")],
        z_bar=[float(v) for v in _req(deg, "z_bar", "problem.toml/degradation")],
        cov={k: float(_req(cov, k, "problem.toml/covariate"))
             for k in ("mu_T", "sigma_T", "lambda", "alpha_L", "beta_L")},
        pool_size=int(_req(scen, "pool_size", "problem.toml/scenario")),
        seed=int(_req(scen, "seed", "problem.toml/scenario")),
        priority={int(k): [int(x) for x in v] for k, v in prio.items()} if prio else None,
        source_path=path,
    )


# ------------------------------------------------------------------ 求解配置
@dataclass
class PartitionConfig:
    """自适应分区（ITER-SDDP）参数。

    这里的默认值只用于兼容**没有 [partition] 节的旧结果目录**（simulate/resume 要能读回
    旧配置）；新配置必须在 solver.toml 里显式写全。本轮只实现 cut_mode="single"。
    """

    enabled: bool = False
    cut_mode: str = "single"       # single=单粗割；multi 留给第二阶段
    adaptive: bool = True
    initial_clusters: int = 1
    signature: str = "pi"          # pi=严格文档；beta=敏感性对照
    eps_sig: float = 0.0           # 绝对签名距离阈值；0=用 eps_sig_rel
    eps_sig_rel: float = 0.1       # 每阶段签名两两距离中位数的倍数
    eps_obj: float = 0.0           # Q 差值阈值
    stall_window: int = 10
    stall_rtol: float = 1.0e-3
    stall_abs: float = 1.0
    warmup_rounds: int = 10
    max_clusters: int = 0          # 0=不限
    refine_points: int = 1

    def __post_init__(self) -> None:
        if self.cut_mode not in ("single", "multi"):
            raise ValueError(f"[partition] cut_mode 只能是 single/multi，"
                             f"实际 {self.cut_mode!r}")
        if self.enabled and self.cut_mode == "multi":
            raise NotImplementedError(
                "[partition] cut_mode='multi' 本阶段未实现；先用 single，"
                "多割在第二阶段单独实现")
        if self.signature not in ("pi", "beta"):
            raise ValueError(f"[partition] signature 只能是 pi/beta，"
                             f"实际 {self.signature!r}")
        if self.initial_clusters < 1:
            raise ValueError(f"[partition] initial_clusters 必须 ≥1，"
                             f"实际 {self.initial_clusters}")
        if self.refine_points != 1:
            raise ValueError(f"[partition] refine_points 首版只支持 1，"
                             f"实际 {self.refine_points}")
        if self.eps_sig < 0 or self.eps_sig_rel < 0 or self.eps_obj < 0:
            raise ValueError("[partition] eps_sig/eps_sig_rel/eps_obj 不得为负")
        if self.stall_window < 1:
            raise ValueError(f"[partition] stall_window 必须 ≥1，"
                             f"实际 {self.stall_window}")
        if self.stall_rtol < 0 or self.stall_abs < 0:
            raise ValueError("[partition] stall_rtol/stall_abs 不得为负")
        if self.warmup_rounds < 0:
            raise ValueError("[partition] warmup_rounds 不得为负")
        if self.max_clusters < 0:
            raise ValueError(f"[partition] max_clusters 不得为负（0=不限），"
                             f"实际 {self.max_clusters}")
        if self.enabled and not (self.eps_sig > 0 or self.eps_sig_rel > 0):
            raise ValueError("[partition] enabled=true 时必须给 eps_sig>0 或 eps_sig_rel>0")

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        eps = (f"eps_sig={self.eps_sig:g}" if self.eps_sig > 0
               else f"eps_sig_rel={self.eps_sig_rel:g}")
        return (f"on  mode={self.cut_mode} sig={self.signature} {eps} "
                f"eps_obj={self.eps_obj:g} K0={self.initial_clusters} "
                f"n={self.stall_window} maxK={self.max_clusters or 'inf'}")


@dataclass
class SolverConfig:
    # 训练
    iterations: int
    forward_paths_per_process: int
    n_backward_samples: int
    checkpoint_interval: float
    max_train_seconds: float
    seed: int
    cut_selection: str
    cut_select_eps: float
    cut_select_keep_rounds: int
    partition: PartitionConfig
    # 并行
    n_processes: int
    threads_per_process: int
    # 仿真
    n_simulations: int
    chunk_size: int
    # 数值
    feas_tol: float
    dual_tol: float
    int_tol: float
    presolve: int
    mip_rel_gap: float
    mip_time_limit: float
    # 输出
    tag: str
    results_root: str
    source_path: Path | None = field(default=None, repr=False)

    def resolve(self, n_physical: int | None = None) -> "SolverConfig":
        """把 n_processes=0 与 forward_paths_per_process 换算成实际值（原地改）。"""
        if self.n_processes == 0:
            self.n_processes = max(1, (n_physical or physical_cores()) // 2)
        if self.n_processes < 1:
            raise ValueError(f"[parallel] n_processes 必须 ≥ 1（0 表示自动），"
                             f"实际 {self.n_processes}")
        if self.forward_paths_per_process < 1:
            raise ValueError(f"[training] forward_paths_per_process 必须 ≥ 1，"
                             f"实际 {self.forward_paths_per_process}")
        return self

    @property
    def n_forward_paths(self) -> int:
        """前向路径数 = 倍率 × 进程数。默认 1 倍 ⇒ 每阶段试探点数 == 进程数。"""
        return self.forward_paths_per_process * self.n_processes

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"[training] iterations 不得为负，实际 {self.iterations}")
        if self.n_backward_samples < 0:
            raise ValueError(f"[training] n_backward_samples 不得为负（0 = 全枚举），"
                             f"实际 {self.n_backward_samples}")
        if self.cut_selection not in ("none", "lml1"):
            raise ValueError(f"[training] cut_selection 只能是 none/lml1，"
                             f"实际 {self.cut_selection!r}")
        if self.cut_select_eps < 0:
            raise ValueError(f"[training] cut_select_eps 不得为负，"
                             f"实际 {self.cut_select_eps}")
        if self.cut_select_keep_rounds < 0:
            raise ValueError(f"[training] cut_select_keep_rounds 不得为负，"
                             f"实际 {self.cut_select_keep_rounds}")
        if self.max_train_seconds < 0:
            raise ValueError(f"[training] max_train_seconds 不得为负（0 = 不限），"
                             f"实际 {self.max_train_seconds}")
        if self.threads_per_process < 1:
            raise ValueError(f"[parallel] threads_per_process 必须 ≥ 1，"
                             f"实际 {self.threads_per_process}")
        if self.chunk_size < 1:
            raise ValueError(f"[simulation] chunk_size 必须 ≥ 1，实际 {self.chunk_size}")
        for k in ("feas_tol", "dual_tol", "int_tol"):
            if getattr(self, k) <= 0:
                raise ValueError(f"[numerics] {k} 必须为正，实际 {getattr(self, k)}")
        if self.mip_rel_gap < 0:
            raise ValueError(f"[numerics] mip_rel_gap 不得为负，实际 {self.mip_rel_gap}")
        if self.mip_time_limit <= 0:
            raise ValueError(f"[numerics] mip_time_limit 必须为正，"
                             f"实际 {self.mip_time_limit}")
        if self.partition.enabled and self.n_backward_samples != 0:
            raise ValueError(
                "[partition] enabled=true 时必须 n_backward_samples=0（全池枚举）；"
                "场景子采样割只对子样本成立，不能冒充全池下界")

    def describe(self) -> str:
        return "\n".join([
            f"训练   轮数 {self.iterations}  前向路径 {self.n_forward_paths}"
            f"（{self.forward_paths_per_process} × {self.n_processes} 进程）"
            f"  后向场景 {self.n_backward_samples or '全枚举'}"
            f"  割筛选 {self.cut_selection}"
            f"  时限 {self.max_train_seconds or '不限'} s",
            f"分区   {self.partition.describe() if hasattr(self.partition, 'describe') else ('on' if self.partition.enabled else 'off')}",
            f"并行   进程 {self.n_processes}  每进程 COPT 线程 {self.threads_per_process}",
            f"仿真   路径 {self.n_simulations}  块大小 {self.chunk_size}",
            f"数值   FeasTol {self.feas_tol:g}  DualTol {self.dual_tol:g}  "
            f"IntTol {self.int_tol:g}  Presolve {self.presolve}",
            f"输出   {self.results_root}/{self.tag}_<时间戳>",
        ])


def load_solver_config(path, *, resolve: bool = True) -> SolverConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    tr = _section(raw, "training", path.name)
    pa = _section(raw, "parallel", path.name)
    si = _section(raw, "simulation", path.name)
    nu = _section(raw, "numerics", path.name)
    ou = _section(raw, "output", path.name)
    # [partition] 是新增节：旧结果目录没有它，用兼容默认（关闭）读回。
    pt = raw.get("partition")
    if pt is None:
        partition = PartitionConfig()
    else:
        if not isinstance(pt, dict):
            raise TypeError(f"配置 {path.name} 的 [partition] 必须是表")
        w = "solver.toml/partition"
        partition = PartitionConfig(
            enabled=bool(_req(pt, "enabled", w)),
            cut_mode=str(_req(pt, "cut_mode", w)),
            adaptive=bool(_req(pt, "adaptive", w)),
            initial_clusters=int(_req(pt, "initial_clusters", w)),
            signature=str(_req(pt, "signature", w)),
            eps_sig=float(_req(pt, "eps_sig", w)),
            eps_sig_rel=float(_req(pt, "eps_sig_rel", w)),
            eps_obj=float(_req(pt, "eps_obj", w)),
            stall_window=int(_req(pt, "stall_window", w)),
            stall_rtol=float(_req(pt, "stall_rtol", w)),
            stall_abs=float(_req(pt, "stall_abs", w)),
            warmup_rounds=int(_req(pt, "warmup_rounds", w)),
            max_clusters=int(_req(pt, "max_clusters", w)),
            refine_points=int(_req(pt, "refine_points", w)),
        )
    cfg = SolverConfig(
        iterations=int(_req(tr, "iterations", "solver.toml/training")),
        forward_paths_per_process=int(
            _req(tr, "forward_paths_per_process", "solver.toml/training")),
        n_backward_samples=int(_req(tr, "n_backward_samples", "solver.toml/training")),
        checkpoint_interval=float(_req(tr, "checkpoint_interval", "solver.toml/training")),
        max_train_seconds=float(_opt(tr, "max_train_seconds", 0.0,
                                     "solver.toml/training")),
        seed=int(_req(tr, "seed", "solver.toml/training")),
        cut_selection=str(_req(tr, "cut_selection", "solver.toml/training")),
        cut_select_eps=float(_req(tr, "cut_select_eps", "solver.toml/training")),
        cut_select_keep_rounds=int(_req(tr, "cut_select_keep_rounds", "solver.toml/training")),
        partition=partition,
        n_processes=int(_req(pa, "n_processes", "solver.toml/parallel")),
        threads_per_process=int(_req(pa, "threads_per_process", "solver.toml/parallel")),
        n_simulations=int(_req(si, "n_simulations", "solver.toml/simulation")),
        chunk_size=int(_req(si, "chunk_size", "solver.toml/simulation")),
        feas_tol=float(_req(nu, "feas_tol", "solver.toml/numerics")),
        dual_tol=float(_req(nu, "dual_tol", "solver.toml/numerics")),
        int_tol=float(_req(nu, "int_tol", "solver.toml/numerics")),
        presolve=int(_req(nu, "presolve", "solver.toml/numerics")),
        mip_rel_gap=float(_req(nu, "mip_rel_gap", "solver.toml/numerics")),
        mip_time_limit=float(_req(nu, "mip_time_limit", "solver.toml/numerics")),
        tag=str(_req(ou, "tag", "solver.toml/output")),
        results_root=str(_req(ou, "results_root", "solver.toml/output")),
        source_path=path,
    )
    return cfg.resolve() if resolve else cfg


# ------------------------------------------------------------------ 目录入口
def config_dir(root) -> Path:
    return Path(root) / "config"


def load_all(root, *, resolve: bool = True) -> tuple[ProblemConfig, SolverConfig]:
    """从 <root>/config/ 读两份配置。"""
    d = config_dir(root)
    return (load_problem_config(d / "problem.toml"),
            load_solver_config(d / "solver.toml", resolve=resolve))
