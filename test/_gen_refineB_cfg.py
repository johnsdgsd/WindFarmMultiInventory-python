# _gen_refineB_cfg.py — 生成阶段 B（固定簇 + 停滞细化）的 T60 配置。
#   python test/_gen_refineB_cfg.py
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE = ROOT / "config" / "tune"
SRC = TUNE / "fcmc_t60_full_L8"

TMPL = """# 阶段 B：固定分区多割 + LB 停滞触发细化（{tag}）
[training]
mode = "multi"
iterations = 100000
forward_paths_per_process = 1
n_backward_samples = 0
checkpoint_interval = 600.0
max_train_seconds = {secs}
cut_selection = "lml1"
cut_select_eps = 1e-6
cut_select_keep_rounds = 0
seed = 2026

[multicut]
n_clusters = 8
group_mode = "failure_quantile"

[io]
payload_mode = "pickle"

[refine]
enabled = true
max_refines = 8
max_clusters_per_stage = 32
theta_cap = 32
max_attempts = 16
max_children_per_event = 8
min_child_size = 2
candidate_stage_stride = 1
min_remaining_seconds = 120
min_event_gain = 1.0
max_ineffective_events = 2
warmup_rounds = 20
stall_on = "lb_best"
stall_window = {window}
stall_rtol = 1e-3
stall_abs = 1.0
cooldown_rounds = 10
recovery_frac = 0.995
recovery_timeout = 30
min_refine_stage = 1
signature = "pi"
eps_sig = 0.0
eps_sig_rel = 0.5
eps_obj = 20.0

[parallel]
n_processes = 0
threads_per_process = 1

[simulation]
n_simulations = {sims}
chunk_size = 1000

[numerics]
feas_tol = 1e-6
dual_tol = 1e-6
int_tol = 1e-6
presolve = -1
mip_rel_gap = 0.001
mip_time_limit = 20.0

[output]
tag = "{tag}"
results_root = "results"
"""

VARIANTS = [
    # name, stall_window, seconds, sims
    ("refineB_t60_n10_pilot600", 10, 600, 200),
    ("refineB_t60_n20_pilot600", 20, 600, 200),
    ("refineB_t60_n10_pilot1800", 10, 1800, 200),
    ("refineB_t60_n20_pilot1800", 20, 1800, 200),
    ("refineB_t60_full_n10", 10, 7503.3, 3000),
    ("refineB_t60_full_n20", 20, 7503.3, 3000),
]


def main():
    base_problem = (SRC / "problem.toml").read_text(encoding="utf-8")
    for name, window, secs, sims in VARIANTS:
        d = TUNE / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "problem.toml").write_text(base_problem, encoding="utf-8")
        (d / "solver.toml").write_text(
            TMPL.format(tag=name, window=window, secs=secs, sims=sims),
            encoding="utf-8")
        print(d.relative_to(ROOT))


if __name__ == "__main__":
    main()
