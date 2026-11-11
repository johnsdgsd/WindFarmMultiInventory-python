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
    # 初始 n0 口径：continuous（旧模型）或 integer_largest_remainder（Tier 2）
    n0_mode: str
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

        if self.n0_mode not in ("continuous", "integer_largest_remainder"):
            raise ValueError(f"[fleet] n0_mode 非法：{self.n0_mode!r}")

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
        n0_mode=str(_opt(fleet, "n0_mode", "continuous", "problem.toml/fleet")),
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
class IntegerConfig:
    """整数模式。

    scope=tier1_logistics：只整数化物流变量，n/N 连续。
    scope=tier2_full：n/N/F 全整数；enabled=false 时仍是同一 Tier 2 模型的 LP 松弛。
    """

    enabled: bool = False
    scope: str = "tier1_logistics"
    rounding: str = "none"

    def __post_init__(self) -> None:
        if self.scope not in ("tier1_logistics", "tier2_full", "none"):
            raise ValueError(f"[integer] scope 非法：{self.scope!r}")
        if self.enabled and self.scope == "none":
            raise ValueError("[integer] enabled=true 时 scope 不能为 none")
        if self.scope == "tier2_full":
            if self.rounding != "cumulative_tail_nearest":
                raise ValueError(
                    "[integer] tier2_full 的 rounding 必须为 "
                    "'cumulative_tail_nearest'")
        elif self.rounding != "none":
            raise ValueError(
                f"[integer] scope={self.scope} 时 rounding 必须为 none")

    def describe(self) -> str:
        tail = (f"  rounding={self.rounding}"
                if self.scope == "tier2_full" else "")
        return f"{'on' if self.enabled else 'off'}  scope={self.scope}{tail}"


@dataclass
class ImportanceSamplingConfig:
    """训练前向的重要性采样：默认关闭；仿真始终均匀 MC。"""

    enabled: bool = False
    target: str = "training"
    build_after_round: int = 50
    uniform_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.target != "training":
            raise ValueError(
                f"[importance_sampling] target 只能是 training，实际 {self.target!r}")
        if self.build_after_round < 1:
            raise ValueError(
                "[importance_sampling] build_after_round 必须 ≥ 1，"
                f"实际 {self.build_after_round}")
        if not (0.0 < float(self.uniform_floor) < 1.0):
            raise ValueError(
                "[importance_sampling] uniform_floor 必须在 (0,1)，"
                f"实际 {self.uniform_floor}")

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        return (f"on  target={self.target}  build_after_round="
                f"{self.build_after_round}  uniform_floor={self.uniform_floor:g}")


@dataclass
class CutLiftConfig:
    """后向割的 Strengthened Benders 截距提升；默认关闭。

    value="exact" 对标 SDDP.jl StrengthenedConicDuality：MIP 必须
    BestObj≈BestBnd 才接受，否则回退 LP 割。value="best_bnd" 只用于
    训练提速/诊断，超时值由 worker 与 LP 割取 max。
    """

    enabled: bool = False
    scope: str = "all"
    last_k: int = 0
    lift_stages: tuple[int, ...] = ()
    mode: str = "always"
    value: str = "exact"
    time_limit: float = 5.0
    rel_gap: float = 0.0
    abs_gap: float = 0.0
    safety_margin: float = 1e-6
    lb_gain_threshold: float = 20.0
    min_round: int = 50

    def __post_init__(self) -> None:
        if self.scope not in ("all", "last", "tail", "selected"):
            raise ValueError(
                f"[cut_lift] scope 只能是 all/last/tail/selected，"
                f"实际 {self.scope!r}")
        if int(self.last_k) < 0:
            raise ValueError(
                f"[cut_lift] last_k 不得为负，实际 {self.last_k!r}")
        if self.scope == "tail" and int(self.last_k) < 1:
            raise ValueError(
                "[cut_lift] scope=tail 必须设置 last_k ≥ 1")
        stages = tuple(int(t) for t in self.lift_stages)
        if any(t < 1 for t in stages):
            raise ValueError(
                f"[cut_lift] lift_stages 必须是正阶段号，实际 {stages!r}")
        if self.scope == "selected" and not stages:
            raise ValueError(
                "[cut_lift] scope=selected 必须设置非空 lift_stages")
        self.lift_stages = stages
        if self.mode not in ("always", "stagnation"):
            raise ValueError(f"[cut_lift] mode 只能是 always/stagnation，实际 {self.mode!r}")
        if self.value not in ("exact", "best_bnd"):
            raise ValueError(
                f"[cut_lift] value 只能是 exact/best_bnd，实际 {self.value!r}"
                "（best_obj 是未证 incumbent，已禁用）")
        if float(self.time_limit) <= 0:
            raise ValueError("[cut_lift] time_limit 必须为正")
        if float(self.rel_gap) < 0 or float(self.abs_gap) < 0:
            raise ValueError("[cut_lift] rel_gap/abs_gap 不得为负")
        if self.value == "exact" and (float(self.rel_gap) != 0.0
                                      or float(self.abs_gap) != 0.0):
            raise ValueError(
                "[cut_lift] value=exact 要求 rel_gap=0 且 abs_gap=0；"
                "要限时用 best_bnd")
        if float(self.safety_margin) < 0:
            raise ValueError("[cut_lift] safety_margin 不得为负")
        if float(self.lb_gain_threshold) < 0:
            raise ValueError("[cut_lift] lb_gain_threshold 不得为负")
        if int(self.min_round) < 0:
            raise ValueError("[cut_lift] min_round 不得为负")

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        scope = self.scope
        if scope == "tail":
            scope = f"tail(last_k={self.last_k})"
        elif scope == "selected":
            scope = f"selected({list(self.lift_stages)})"
        return (f"on  scope={scope}  mode={self.mode}  value={self.value}  "
                f"time_limit={self.time_limit:g}s  rel_gap={self.rel_gap:g}"
                + (f"  threshold={self.lb_gain_threshold:g}  min_round={self.min_round}"
                   if self.mode == "stagnation" else ""))


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
    stage_block_size: int
    stage_block_adaptive: bool
    forward_heuristic: bool
    forward_heuristic_mode: str
    heur_lp_p0: float
    heur_lp_p1: float
    heur_ramp_iters: int
    heur_alpha: float
    heur_target_frac: float
    heur_sa_iters: int
    heur_sa_t0_frac: float
    heur_sa_cooling: float
    heur_sa_step: float
    heur_q_prob: float
    heur_q_step: float
    heur_q_resolve_prob: float
    heur_q_dual_trials: bool
    stage_block_sizes: tuple[int, ...]
    stage_block_switch_s: tuple[float, ...]
    integer: IntegerConfig
    importance_sampling: ImportanceSamplingConfig
    cut_lift: CutLiftConfig
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
    lb_rel_gap: float
    lb_time_limit: float
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

    @property
    def integer_enabled(self) -> bool:
        return bool(self.integer.enabled)

    @property
    def integer_scope(self) -> str:
        return str(self.integer.scope)

    @property
    def integer_rounding(self) -> str:
        return str(self.integer.rounding)

    @property
    def importance_enabled(self) -> bool:
        return bool(self.importance_sampling.enabled)

    @property
    def cut_lift_enabled(self) -> bool:
        return bool(self.cut_lift.enabled)

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"[training] iterations 不得为负，实际 {self.iterations}")
        if self.n_backward_samples < 0:
            raise ValueError(f"[training] n_backward_samples 不得为负（0 = 全枚举），"
                             f"实际 {self.n_backward_samples}")
        if self.cut_selection not in ("none", "lml1", "julia"):
            raise ValueError(f"[training] cut_selection 只能是 none/lml1/julia，"
                             f"实际 {self.cut_selection!r}")
        if self.cut_select_eps < 0:
            raise ValueError(f"[training] cut_select_eps 不得为负，"
                             f"实际 {self.cut_select_eps}")
        if self.cut_select_keep_rounds < 0:
            raise ValueError(f"[training] cut_select_keep_rounds 不得为负，"
                             f"实际 {self.cut_select_keep_rounds}")
        if self.stage_block_size < 0:
            raise ValueError(f"[training] stage_block_size 不得为负（0 = 关闭），"
                             f"实际 {self.stage_block_size}")
        if self.forward_heuristic_mode not in ("sa", "qperturb"):
            raise ValueError("[training] forward_heuristic_mode 只能是 sa/qperturb，"
                             f"实际 {self.forward_heuristic_mode!r}")
        if not (0.0 <= self.heur_lp_p0 <= 1.0
                and 0.0 <= self.heur_lp_p1 <= 1.0):
            raise ValueError("[training] heur_lp_p0/heur_lp_p1 必须在 [0,1]")
        if self.heur_ramp_iters < 1:
            raise ValueError("[training] heur_ramp_iters 必须 ≥ 1")
        if self.heur_alpha <= 0:
            raise ValueError("[training] heur_alpha 必须为正")
        if not (0.0 < self.heur_target_frac <= 1.0):
            raise ValueError("[training] heur_target_frac 必须在 (0,1]")
        if self.heur_sa_iters < 1:
            raise ValueError("[training] heur_sa_iters 必须 ≥ 1")
        if self.heur_sa_t0_frac <= 0:
            raise ValueError("[training] heur_sa_t0_frac 必须为正")
        if not (0.0 < self.heur_sa_cooling < 1.0):
            raise ValueError("[training] heur_sa_cooling 必须在 (0,1)")
        if self.heur_sa_step <= 0:
            raise ValueError("[training] heur_sa_step 必须为正")
        if not (0.0 <= self.heur_q_prob <= 1.0):
            raise ValueError("[training] heur_q_prob 必须在 [0,1]")
        if self.heur_q_step <= 0:
            raise ValueError("[training] heur_q_step 必须为正")
        if not (0.0 <= self.heur_q_resolve_prob <= 1.0):
            raise ValueError("[training] heur_q_resolve_prob 必须在 [0,1]")
        if self.stage_block_adaptive:
            if not self.stage_block_sizes:
                raise ValueError(
                    "[training] stage_block_adaptive=true 需要非空 stage_block_sizes")
            if len(self.stage_block_switch_s) != len(self.stage_block_sizes) - 1:
                raise ValueError(
                    "[training] stage_block_switch_s 长度必须等于 "
                    "stage_block_sizes 长度减 1")
            if any(int(x) <= 0 for x in self.stage_block_sizes):
                raise ValueError("[training] stage_block_sizes 必须全为正")
            sw = [float(x) for x in self.stage_block_switch_s]
            if any(sw[i] >= sw[i + 1] for i in range(len(sw) - 1)):
                raise ValueError(
                    "[training] stage_block_switch_s 必须严格递增")
            if any(x <= 0 for x in sw):
                raise ValueError(
                    "[training] stage_block_switch_s 必须全为正")
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
        if self.lb_rel_gap < 0:
            raise ValueError(f"[numerics] lb_rel_gap 不得为负，"
                             f"实际 {self.lb_rel_gap}")
        if self.lb_time_limit <= 0:
            raise ValueError(f"[numerics] lb_time_limit 必须为正，"
                             f"实际 {self.lb_time_limit}")

    def stage_block_desc(self) -> str:
        if not self.stage_block_adaptive:
            return str(self.stage_block_size or "off")
        parts = []
        t0 = 0.0
        for i, b in enumerate(self.stage_block_sizes):
            t1 = (self.stage_block_switch_s[i]
                  if i < len(self.stage_block_switch_s) else None)
            span = f"{t0:g}s+" if t1 is None else f"{t0:g}-{t1:g}s"
            parts.append(f"{span}:B={b}")
            if t1 is not None:
                t0 = float(t1)
        return "adaptive[" + ", ".join(parts) + "]"

    def describe(self) -> str:
        return "\n".join([
            f"训练   轮数 {self.iterations}  前向路径 {self.n_forward_paths}"
            f"（{self.forward_paths_per_process} × {self.n_processes} 进程）"
            f"  后向场景 {self.n_backward_samples or '全枚举'}"
            f"  割筛选 {self.cut_selection}"
            f"  阶段块 {self.stage_block_desc()}"
            f"  整数 {self.integer.describe()}"
            f"  前向启发式 {'on:' + self.forward_heuristic_mode if self.forward_heuristic else 'off'}"
            f"  重要性采样 {self.importance_sampling.describe()}"
            f"  割提升 {self.cut_lift.describe()}"
            f"  时限 {self.max_train_seconds or '不限'} s",
            f"并行   进程 {self.n_processes}  每进程 COPT 线程 {self.threads_per_process}",
            f"仿真   路径 {self.n_simulations}  块大小 {self.chunk_size}",
            f"数值   FeasTol {self.feas_tol:g}  DualTol {self.dual_tol:g}  "
            f"IntTol {self.int_tol:g}  Presolve {self.presolve}"
            f"  LB MIP gap {self.lb_rel_gap:g} / {self.lb_time_limit:g}s",
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
    in_raw = raw.get("integer")
    if in_raw is None:
        integer = IntegerConfig()             # 旧配置/基线默认 LP
    else:
        if not isinstance(in_raw, dict):
            raise TypeError(f"配置 {path.name} 的 [integer] 必须是表")
        scope = str(_req(in_raw, "scope", "solver.toml/integer"))
        rounding = (str(_req(in_raw, "rounding", "solver.toml/integer"))
                    if scope == "tier2_full"
                    else str(_opt(in_raw, "rounding", "none",
                                  "solver.toml/integer")))
        integer = IntegerConfig(
            enabled=bool(_req(in_raw, "enabled", "solver.toml/integer")),
            scope=scope,
            rounding=rounding,
        )
    is_raw = raw.get("importance_sampling")
    if is_raw is None:
        importance = ImportanceSamplingConfig()
    else:
        if not isinstance(is_raw, dict):
            raise TypeError(f"配置 {path.name} 的 [importance_sampling] 必须是表")
        importance = ImportanceSamplingConfig(
            enabled=bool(_req(is_raw, "enabled",
                              "solver.toml/importance_sampling")),
            target=str(_req(is_raw, "target",
                            "solver.toml/importance_sampling")),
            build_after_round=int(_req(is_raw, "build_after_round",
                                       "solver.toml/importance_sampling")),
            uniform_floor=float(_req(is_raw, "uniform_floor",
                                     "solver.toml/importance_sampling")),
        )
    cl_raw = raw.get("cut_lift")
    if cl_raw is None:
        cut_lift = CutLiftConfig()
    else:
        if not isinstance(cl_raw, dict):
            raise TypeError(f"配置 {path.name} 的 [cut_lift] 必须是表")
        cut_lift = CutLiftConfig(
            enabled=bool(_req(cl_raw, "enabled", "solver.toml/cut_lift")),
            scope=str(_req(cl_raw, "scope", "solver.toml/cut_lift")),
            last_k=int(_opt(cl_raw, "last_k", 0, "solver.toml/cut_lift")),
            lift_stages=tuple(int(t) for t in _opt(
                cl_raw, "lift_stages", [], "solver.toml/cut_lift")),
            mode=str(_opt(cl_raw, "mode", "always", "solver.toml/cut_lift")),
            value=str(_opt(cl_raw, "value", "exact", "solver.toml/cut_lift")),
            time_limit=float(_req(cl_raw, "time_limit", "solver.toml/cut_lift")),
            rel_gap=float(_req(cl_raw, "rel_gap", "solver.toml/cut_lift")),
            abs_gap=float(_opt(cl_raw, "abs_gap", 0.0, "solver.toml/cut_lift")),
            safety_margin=float(_opt(cl_raw, "safety_margin", 1e-6,
                                     "solver.toml/cut_lift")),
            lb_gain_threshold=float(_opt(cl_raw, "lb_gain_threshold", 20.0,
                                         "solver.toml/cut_lift")),
            min_round=int(_opt(cl_raw, "min_round", 50,
                               "solver.toml/cut_lift")),
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
        stage_block_size=int(_opt(tr, "stage_block_size", 0,
                                  "solver.toml/training")),
        stage_block_adaptive=bool(_opt(tr, "stage_block_adaptive", False,
                                       "solver.toml/training")),
        forward_heuristic=bool(_opt(tr, "forward_heuristic", False,
                                    "solver.toml/training")),
        forward_heuristic_mode=str(_opt(
            tr, "forward_heuristic_mode", "sa",
            "solver.toml/training")),
        heur_lp_p0=float(_opt(tr, "heur_lp_p0", 0.5, "solver.toml/training")),
        heur_lp_p1=float(_opt(tr, "heur_lp_p1", 0.95, "solver.toml/training")),
        heur_ramp_iters=int(_opt(tr, "heur_ramp_iters", 1000,
                                 "solver.toml/training")),
        heur_alpha=float(_opt(tr, "heur_alpha", 1.0, "solver.toml/training")),
        heur_target_frac=float(_opt(tr, "heur_target_frac", 1.0,
                                    "solver.toml/training")),
        heur_sa_iters=int(_opt(tr, "heur_sa_iters", 20,
                               "solver.toml/training")),
        heur_sa_t0_frac=float(_opt(tr, "heur_sa_t0_frac", 0.02,
                                   "solver.toml/training")),
        heur_sa_cooling=float(_opt(tr, "heur_sa_cooling", 0.9,
                                   "solver.toml/training")),
        heur_sa_step=float(_opt(tr, "heur_sa_step", 1.0,
                                "solver.toml/training")),
        heur_q_prob=float(_opt(tr, "heur_q_prob", 0.5,
                               "solver.toml/training")),
        heur_q_step=float(_opt(tr, "heur_q_step", 1.0,
                               "solver.toml/training")),
        heur_q_resolve_prob=float(_opt(
            tr, "heur_q_resolve_prob", 1.0, "solver.toml/training")),
        heur_q_dual_trials=bool(_opt(tr, "heur_q_dual_trials", False,
                                     "solver.toml/training")),
        stage_block_sizes=tuple(int(x) for x in _opt(
            tr, "stage_block_sizes", [], "solver.toml/training")),
        stage_block_switch_s=tuple(float(x) for x in _opt(
            tr, "stage_block_switch_s", [], "solver.toml/training")),
        integer=integer,
        importance_sampling=importance,
        cut_lift=cut_lift,
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
        lb_rel_gap=float(_opt(nu, "lb_rel_gap", 1e-9, "solver.toml/numerics")),
        lb_time_limit=float(_opt(nu, "lb_time_limit", 60.0,
                                 "solver.toml/numerics")),
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
