# run.py — 唯一训练入口。路径全部由 __file__ 推导，在任何目录下都能跑。
#
#   python run.py                              # 全部用 main 的默认值
#   python run.py 500                          # 训练 500 轮（仿真路径数仍用默认）
#   python run.py 500 300                      # 训练 500 轮 + 仿真 300 路径
#   python run.py smoke                        # 冒烟：小实例快跑，验证链路连通
#   python run.py 200 resume results/base_20261008_120000    # 续训 200 轮
#
# 求解器默认 COPT_DLL；命令行加一个位置参数 "cbc" 即改用 CBC，位置不限：
#   python run.py smoke cbc                    # 冒烟 + CBC
#   python run.py 500 cbc                      # 训练 500 轮 + CBC
#   python run.py 200 resume results/<目录> cbc   # 续训 + CBC
# 求解器写进 config.txt 的 solver= 字段（copt/cbc），simulate_only 与
# simulate_inspect 会自动沿用。
#
# 位置参数只在给了的时候才传给 main，没给就交给 main 的默认值 —— 别在这里写死
# 200/1000 之类的兜底，那会盖掉 main 里改过的默认值。
#
# 训练与仿真走多进程（multiprocessing，Windows 为 spawn）。进程数默认核数，
# 用 n_processes 覆盖；n_processes=1 时退化为串行，便于调试与 traceback。

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Windows 控制台默认 GBK，落不进的字（如 β̄ 之类组合符号）会直接抛 UnicodeEncodeError
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

from code.model import build_stage_subproblem                      # noqa: E402
from code.problem_data import DefineProblem, print_summary, to_json  # noqa: E402
from utils import SDDPEngine, simulate                             # noqa: E402
from utils.io import (archive_log, load_cuts, new_run_dir, read_config,  # noqa: E402
                      save_json, save_summary, write_config)

RESULTS_ROOT = str(ROOT / "results")


def main(*, iterations: int = 200, n_simulations: int = 10000, pool_size: int = 300,
         T: int = 200, A: int = 120, N_R: int = 5, L: int = 2, seed: int = 2026,
         m: float = 2.5, eta: float = 120.0, n_forward_paths: int = 120,
         n_backward_samples: int | None = None, chunk_size: int = 1000,
         tag: str = "base", smoke: bool = False, n_processes: int | None = 12,
         checkpoint_interval: float = 600.0, resume_dir: str | None = None,
         results_root: str = RESULTS_ROOT, use_copt: bool = True):
    spread_init = False
    if smoke:
        # 短寿命 + 月龄摊平：让失效/到寿/调拨/订货链路在少数几期内全部被激活
        T, pool_size, iterations = 4, 30, 10
        A, n_simulations, tag = 24, 50, "smoke"
        m, eta, spread_init, n_forward_paths = 3.0, 8.0, True, 2
        print("=" * 64)
        print(f"冒烟模式：T = {T}, A = {A}, pool_size = {pool_size}, "
              f"iterations = {iterations}, n_simulations = {n_simulations}, "
              f"m = {m}, η = {eta}")
        print("=" * 64)

    prob = DefineProblem(T=T, A=A, N_R=N_R, L=L, pool_size=pool_size, seed=seed,
                         m=m, eta=eta, spread_init=spread_init)
    print_summary(prob)

    t0 = time.time()
    if resume_dir is not None:
        d = Path(resume_dir)
        if not d.is_absolute():
            d = ROOT / d
    else:
        d = new_run_dir(tag, results_root)
        print(f"结果目录：{d}")

    # 工作进程在建池时装配 results_dir（求解失败要往那里导出现场 LP），
    # 所以目录必须先定下来再建引擎。
    if not use_copt:
        print("求解器：CBC（COPT_DLL 关闭）")
    eng = SDDPEngine(
        prob, build_stage_subproblem, T=prob.T, state_dim=prob.state_dim,
        initial_state=prob.initial_state(), n_processes=n_processes, seed=seed,
        results_dir=d, checkpoint_interval=checkpoint_interval,
        use_copt=use_copt)
    start_iter = 1
    try:
        if resume_dir is not None:
            cuts, meta = load_cuts(d, prob.T, prob.state_dim)
            eng.cuts = cuts
            start_iter = int(read_config(d).get("iterations", 0) or 0) + 1
            archive_log(d)
            print(f"续训：从 {d} 读回 {sum(len(c) for c in cuts)} 条割，"
                  f"起始轮次 {start_iter}（重启下界 = {eng.lower_bound():.4f}）")

        write_config(d, [
            ("tag", tag), ("iterations", iterations), ("pool_size", pool_size),
            ("T", T), ("A", A), ("N_R", N_R), ("L", L), ("seed", seed),
            ("spread_init", spread_init), ("n_forward_paths", n_forward_paths),
            ("n_backward_samples", "all" if n_backward_samples is None
             else n_backward_samples),
            ("C", prob.C), ("h", prob.h), ("Cs", prob.Cs),
            ("atc", prob.atc), ("tc", prob.tc),
            ("m", prob.m), ("eta", prob.eta), ("a_beta", prob.a_beta),
            ("beta_cov", list(np.asarray(prob.beta_cov).round(6))),
            ("z_bar", list(np.asarray(prob.z_bar).round(6))),
            ("N", list(np.asarray(prob.N).astype(int))),
            ("Qmax", list(np.asarray(prob.Qmax).astype(int))),
            ("Imax", list(np.asarray(prob.Imax).astype(int))),
            ("solver", "copt" if use_copt else "cbc"),
        ])
        save_json(d, "problem_data.json", to_json(prob))

        # 逐轮回调里写日志：引擎只在 verbose 时打印，这里统一由本脚本落 CSV + 终端
        log = eng.open_log()

        def on_iter(rec):
            log.row(rec["iter"], rec["sim"], rec["lb"], rec["time"], rec["solves"])
            print(f"  轮 {rec['iter']:5d} | LB = {rec['lb']:14.4f} | "
                  f"前向均值 = {rec['sim']:14.4f} | 割数 = {rec['n_cuts']:6d} | "
                  f"{rec['time']:7.1f} s")

        eng.on_iter = on_iter
        hist = eng.train(iterations, n_forward_paths=n_forward_paths,
                         n_backward_samples=n_backward_samples,
                         start_iter=start_iter, verbose=False)
        log.close()

        lb = eng.lower_bound()
        print(f"训练完成：{len(hist)} 轮，用时 {time.time() - t0:.1f} s，"
              f"累计求解 {eng.n_solves} 次，LB = {lb:.4f}")

        print(f"仿真 {n_simulations} 条路径（chunk = {chunk_size}）…")
        st = simulate(eng, n_paths=n_simulations, seed=seed, chunk_size=chunk_size)
        ub, ci = st.mean, st.ci95
        gap = (ub - lb) / abs(ub) * 100 if ub else float("nan")
        print(f"UB = {ub:.4f} ± {ci:.4f}（95%），gap = {gap:.2f}%")

        stage = st.stage_mean()
        lines = [
            f"结果目录    : {d}",
            f"训练轮次    : {len(hist)}（起始 {start_iter}）",
            f"下界 LB     : {lb:.6f}",
            f"上界 UB     : {ub:.6f} ± {ci:.6f} (95% CI, {st.n} 条路径)",
            f"间隙 gap    : {gap:.4f} %",
            f"总成本 std  : {st.std:.6f}   最小 {st.lo:.4f}   最大 {st.hi:.4f}",
            f"割数        : {sum(len(c) for c in eng.cuts)}",
            f"累计求解    : {eng.n_solves}",
            f"用时        : {time.time() - t0:.1f} s",
            "",
            "逐期平均成本（第 1..T 期）：",
            "  " + " ".join(f"{v:.4g}" for v in stage),
        ]
        print("\n".join(lines))
        save_summary(d, "\n".join(lines) + "\n")
        eng.checkpoint()
        print(f"已保存：{d}")
    finally:
        eng.close()
    return d


if __name__ == "__main__":
    args = sys.argv[1:]
    use_cbc = "cbc" in args
    if use_cbc:
        args = [a for a in args if a != "cbc"]
    if args and args[0] == "smoke":
        main(smoke=True, use_copt=not use_cbc)
    elif len(args) >= 2 and args[1] == "resume":
        if len(args) < 3:
            raise SystemExit("resume 需指定结果目录：python run.py 200 resume results/<目录名>")
        main(iterations=int(args[0]), resume_dir=args[2], tag=Path(args[2]).name,
             use_copt=not use_cbc)
    elif not args:
        main(use_copt=not use_cbc)
    elif len(args) == 1:
        main(iterations=int(args[0]), use_copt=not use_cbc)
    else:
        main(iterations=int(args[0]), n_simulations=int(args[1]),
             use_copt=not use_cbc)
