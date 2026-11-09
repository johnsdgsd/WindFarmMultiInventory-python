# _t_tier2_timing.py — 用 Tier2 正式割池跑一轮，拆分前向/后向/下界墙钟。
#
# 单进程、单前向路径，避免多进程排队和负载均衡影响；只读结果目录，不写 results/。
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


class TimedEngine(SDDPEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.t_forward = 0.0
        self.t_backward = 0.0
        self.t_lb = 0.0

    def forward(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return super().forward(*a, **kw)
        finally:
            self.t_forward += time.perf_counter() - t0

    def backward(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return super().backward(*a, **kw)
        finally:
            self.t_backward += time.perf_counter() - t0

    def lower_bound(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return super().lower_bound(*a, **kw)
        finally:
            self.t_lb += time.perf_counter() - t0


def main():
    base = ROOT / "results" / "int_t2_formal_20261106_180256"
    pcfg = load_problem_config(base / "config" / "problem.toml")
    scfg = load_solver_config(base / "config" / "solver.toml")
    scfg.n_processes = 1
    scfg.forward_paths_per_process = 1
    scfg.max_train_seconds = 0.0
    scfg.n_simulations = 0
    prob = DefineProblem(pcfg)
    cuts, meta = load_cuts(base, prob.T, prob.state_dim)
    print(f"T={prob.T} 维={prob.state_dim} 割={sum(map(len, cuts))} "
          f"n_scenes={len(prob.pool)} 进程={scfg.n_processes} "
          f"前向路径={scfg.n_forward_paths}", flush=True)

    eng = TimedEngine(prob, make_stage_subproblem, scfg,
                      results_dir=None, log=lambda *a, **k: None)
    try:
        eng.cuts = cuts
        eng.lb_best = float(meta.get("lb_best", 0.0))
        n_hist = eng.restore_cut_history()
        print(f"LML1 历史恢复 {n_hist} 条", flush=True)
        solves0 = eng.n_solves
        start = int(meta.get("n_iterations", 0)) + 1
        t0 = time.perf_counter()
        hist = eng.train(1, start_iter=start, verbose=False)
        total = time.perf_counter() - t0
        ds = eng.n_solves - solves0
        t_f, t_b, t_lb = eng.t_forward, eng.t_backward, eng.t_lb
        other = total - t_f - t_b - t_lb
        print(f"\n一轮（{start}）总墙钟 {total:.3f} s，求解 {ds} 次")
        print(f"  前向   {t_f:8.3f} s  {t_f/total*100:6.2f}%")
        print(f"  后向   {t_b:8.3f} s  {t_b/total*100:6.2f}%")
        print(f"  下界   {t_lb:8.3f} s  {t_lb/total*100:6.2f}%")
        print(f"  其他   {other:8.3f} s  {other/total*100:6.2f}%")
        if hist:
            print(f"  末轮 LB {hist[-1]['lb']:.4f}，割数 {hist[-1]['n_cuts']}")
        print("\n明细（单进程单路径；含首轮 worker 建模型时间）：")
        print("  前向 = 1 条路径 × 61 次 MIP 求解")
        print("  后向 = 1 个试探点 × 300 场景 × 60 阶段 = 18,000 次 LP")
        print("  下界 = 阶段 0 的 1 次 LP")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
