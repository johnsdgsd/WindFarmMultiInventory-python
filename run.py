# run.py — 唯一训练入口。路径全部由 __file__ 推导，在任何目录下都能跑。
#
#   python run.py                                   # 用 config/ 的配置
#   python run.py --config-dir config/smoke         # 换一套配置
#   python run.py 300                               # 覆盖轮数（其余仍走配置）
#   python run.py 300 2000                          # 覆盖轮数 + 仿真路径数
#   python run.py --config-dir config/smoke 20      # 冒烟配置 + 覆盖轮数
#   python run.py 200 resume results/<目录>          # 续训
#
# 参数的唯一事实源是 config/*.toml：默认值只写在那里，代码里没有第二处。
# 命令行位置参数是**显式覆盖**（覆盖谁就在启动时回显谁），不是默认值。
#
# 训练与仿真走多进程（Windows 为 spawn）。进程数默认物理核数的一半，用
# [parallel].n_processes 改；n_processes=1 退化为串行，便于调试与 traceback。

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Windows 控制台默认 GBK，落不进的字（如 β̄ 之类组合符号）会直接抛 UnicodeEncodeError
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

# 卡死排查开关：设 SDDP_STACK_AFTER=<秒> 后，若 N 秒内没跑完就打印所有线程的调用栈
# 再退出。程序"看起来卡住"时用它一次定位到阻塞点（多进程下 worker 的栈要各自 dump）。
_stack_after = os.environ.get("SDDP_STACK_AFTER")
if _stack_after:
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(float(_stack_after), exit=True)
    print(f"[调试] {_stack_after} s 后会打印全部线程栈并退出")

from code.config import load_problem_config, load_solver_config     # noqa: E402
from code.problem_data import DefineProblem, print_summary, to_json  # noqa: E402
from code.subproblem import make_stage_subproblem                   # noqa: E402
from utils import SDDPEngine, simulate                              # noqa: E402
from utils.io import (archive_log, has_configs, load_cuts, new_run_dir,  # noqa: E402
                      problem_diff, save_json, save_summary, write_configs)


def _load(config_dir: Path):
    return (load_problem_config(config_dir / "problem.toml"),
            load_solver_config(config_dir / "solver.toml"))


def main(*, config_dir: str = "config", iterations: int | None = None,
         n_simulations: int | None = None, resume_dir: str | None = None) -> Path:
    cdir = Path(config_dir)
    if not cdir.is_absolute():
        cdir = ROOT / cdir
    pcfg, scfg = _load(cdir)

    print("=" * 66)
    print(f"生效配置（来自 {cdir}）")
    print("=" * 66)
    print(f"问题   T={pcfg.T}  A={pcfg.A}  N_R={pcfg.N_R}  L={pcfg.L}  "
          f"池={pcfg.pool_size}  m={pcfg.m}  eta={pcfg.eta}")
    print(scfg.describe())

    overrides = []
    if iterations is not None:
        overrides.append(f"iterations: {scfg.iterations} → {iterations}")
        scfg.iterations = int(iterations)
    if n_simulations is not None:
        overrides.append(f"n_simulations: {scfg.n_simulations} → {n_simulations}")
        scfg.n_simulations = int(n_simulations)
    if overrides:
        print("命令行覆盖：" + "；".join(overrides))
    print("=" * 66)

    prob = DefineProblem(pcfg)
    print_summary(prob)

    t0 = time.time()
    if resume_dir is not None:
        d = Path(resume_dir)
        if not d.is_absolute():
            d = ROOT / d
        if not has_configs(d):
            raise SystemExit(f"{d} 里没有 config/problem.toml，不是本版本的结果目录，"
                             f"无法续训")
        rpcfg, _ = _load(d / "config")
        diff = problem_diff(rpcfg, pcfg)
        if diff:
            raise SystemExit(
                f"问题配置与结果目录不一致（{', '.join(diff)}）——模型已变，旧割不再"
                f"兼容（铁律 4），必须换新目录全新训练")
    else:
        d = new_run_dir(scfg.tag, str(ROOT / scfg.results_root))
        print(f"结果目录：{d}")
        write_configs(d, cdir / "problem.toml", cdir / "solver.toml")
        save_json(d, "problem_data.json", to_json(prob))

    # 工作进程在建池时装配 results_dir（求解失败要往那里导出现场 LP），
    # 所以目录必须先定下来再建引擎。
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=d, log=print)
    start_iter = 1
    try:
        if resume_dir is not None:
            cuts, meta = load_cuts(d, prob.T, prob.state_dim)
            eng.cuts = cuts
            start_iter = int(meta.get("n_iterations", 0)) + 1
            archive_log(d)
            print(f"续训：从 {d} 读回 {sum(len(c) for c in cuts)} 条割，"
                  f"起始轮次 {start_iter}（重启下界 = {eng.lower_bound():.4f}）")
            write_configs(d, cdir / "problem.toml", cdir / "solver.toml")

        log = eng.open_log()

        def on_iter(rec):
            log.row(rec["iter"], rec["sim"], rec["lb"], rec["time"], rec["solves"])
            fb = rec.get("mip_fallback", 0)
            note = f" | MIP 近似 {fb}" if fb else ""
            nd = rec.get("n_dropped", 0)
            if nd:
                note += f" | 已筛除 {nd}"
            print(f"  轮 {rec['iter']:5d} | LB = {rec['lb']:14.4f} | "
                  f"前向均值 = {rec['sim']:14.4f} | 割数 = {rec['n_cuts']:6d} | "
                  f"{rec['time']:7.1f} s{note}")

        eng.on_iter = on_iter
        hist = eng.train(scfg.iterations, start_iter=start_iter, verbose=False,
                         trace=bool(os.environ.get("SDDP_TRACE")))
        log.close()

        lb = eng.lower_bound()
        print(f"训练完成：{len(hist)} 轮，用时 {time.time() - t0:.1f} s，"
              f"累计求解 {eng.n_solves} 次，LB = {lb:.4f}")

        print(f"仿真 {scfg.n_simulations} 条路径（chunk = {scfg.chunk_size}）…")
        st = simulate(eng, n_paths=scfg.n_simulations, seed=scfg.seed,
                      chunk_size=scfg.chunk_size)
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
    kw = {}
    if "--config-dir" in args:
        i = args.index("--config-dir")
        if i + 1 >= len(args):
            raise SystemExit("--config-dir 需要一个路径参数")
        kw["config_dir"] = args[i + 1]
        args = args[:i] + args[i + 2:]
    if "resume" in args:
        i = args.index("resume")
        if i + 1 >= len(args):
            raise SystemExit("resume 需指定结果目录：python run.py [轮数] resume results/<目录>")
        kw["resume_dir"] = args[i + 1]
        if i > 0:
            kw["iterations"] = int(args[0])
    elif len(args) >= 1:
        kw["iterations"] = int(args[0])
        if len(args) >= 2:
            kw["n_simulations"] = int(args[1])
    main(**kw)
