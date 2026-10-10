# IIS 全链路：引擎后向撞上不可行子问题 → 失败兜底导出现场 LP → 交给冲突集分析。
#   python test/_t_iis2.py
#   python AnalysisModel_IIS.py test/_tmp_iis/subproblem_backward_t1_s0.lp --solver copt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.problem_data import DefineProblem          # noqa: E402
from code.model import build_stage_subproblem        # noqa: E402
from utils import SDDPEngine, SolverError            # noqa: E402

prob = DefineProblem(T=3, A=12, N_R=2, pool_size=8, spread_init=True, m=3.0, eta=8.0)
Ly = prob.layout
x0 = prob.initial_state().copy()
# 仓库 0 塞成年状态：月龄 1/2 的 θ≈0，ΣF 补不上多出来的 2000 → dem 必冲突
Ly.n(x0)[0, :] = 0
Ly.n(x0)[0, 1] = 10000.0
Ly.n(x0)[0, 2] = 2000.0
n_in = Ly.n(x0)
print("仓库0 月龄和 =", n_in[0].sum(), "> N[0] =", prob.N[0])
print("仓库0 ΣF 各场景 =", [int(np.rint(th[0] * n_in[0]).sum()) for _, th in prob.pool])
sys.stdout.flush()

out = Path(__file__).resolve().parent / "_tmp_iis"
eng = SDDPEngine(prob, build_stage_subproblem, T=prob.T, state_dim=prob.state_dim,
                 initial_state=x0, n_processes=1, seed=0, results_dir=out,
                 log=lambda *a, **k: None)
trial = [[] for _ in range(prob.T + 1)]
trial[1].append(x0)                      # 直接把越界态塞进 stage 1 的后向
try:
    eng.backward(trial, 0)
    print("后向居然全过 —— 加狠扰动")
    sys.exit(1)
except SolverError as exc:
    print("引擎兜底触发:", exc)
finally:
    eng.close()

lps = sorted(out.glob("*.lp"))
print("导出现场 LP:", [p.name for p in lps], [p.stat().st_size for p in lps])
