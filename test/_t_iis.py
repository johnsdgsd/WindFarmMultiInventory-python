# IIS 全链路：引擎后向撞上不可行子问题 → 失败兜底导出现场 LP → 交给冲突集分析。
#   python test/_t_iis.py
#   python AnalysisModel_IIS.py test/_tmp_iis/subproblem_backward_t1_s*.lp --solver copt
#
# ⚠ 建进程池的脚本必须有 __main__ 守卫（Windows spawn）。
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine, SolverError                             # noqa: E402


def main() -> int:
    pcfg = dataclasses.replace(
        load_problem_config(ROOT / "config/smoke/problem.toml"), T=3, N_R=2,
        N=[10000, 8000], Qmax=[300, 100], Imax=[1000, 400], pool_size=8)
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        n_processes=1, n_backward_samples=0)
    prob = DefineProblem(pcfg)
    Ly = prob.layout

    x0 = prob.initial_state().copy()
    # 仓库 0 塞成年状态：月龄 1/2 的 θ≈0，ΣF 补不上多出来的 2000 → dem 必冲突
    Ly.n(x0)[0, :] = 0
    Ly.n(x0)[0, 1] = pcfg.N[0]
    Ly.n(x0)[0, 2] = 2000.0
    n_in = Ly.n(x0)
    print(f"仓库0 月龄和 = {n_in[0].sum():.0f} > N[0] = {pcfg.N[0]}")
    print(f"仓库0 ΣF 各场景 = "
          f"{[int(np.rint(th[0] * n_in[0]).sum()) for _, th in prob.pool[:5]]} …")
    sys.stdout.flush()

    out = ROOT / "test" / "_tmp_iis"
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=out,
                     log=lambda *a, **k: None)
    trial = [[] for _ in range(prob.T + 1)]
    trial[1].append(x0)                      # 直接把越界态塞进 stage 1 的后向
    ok, lp = False, []
    try:
        eng.backward(trial, it=0)
        print("后向居然全过 —— 加狠扰动")
    except SolverError as exc:
        print(f"引擎兜底触发：{exc}")
        lp = list(out.glob("subproblem_backward_t1_s*.lp"))
        print(f"现场 LP 已导出 {len(lp)} 个：{[p.name for p in lp]}")
        ok = len(lp) > 0
    finally:
        eng.close()
    print()
    print(f"  [{'PASS' if ok else 'FAIL'}] 不可行子问题被捕获且导出了现场 LP")
    if ok:
        print(f"  下一步：python AnalysisModel_IIS.py {lp[0]} --solver copt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
