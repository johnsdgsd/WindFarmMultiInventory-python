# IIS 现场导出：故意造一个不可行子问题（Σn_in 超过 N + ΣF）→ 导出 LP 供冲突集分析。
#   python test/_t_iis.py
#   python AnalysisModel_IIS.py test/_iis_probe.lp --solver copt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.problem_data import DefineProblem          # noqa: E402
from code.model import build_stage_subproblem        # noqa: E402
from utils import make_solver, trivial_cut           # noqa: E402
from utils.solver import export_lp                   # noqa: E402

prob = DefineProblem(T=3, A=12, N_R=2, pool_size=8, spread_init=True, m=3.0, eta=8.0)
Ly = prob.layout
xs = prob.initial_state().copy()
Ly.n(xs)[0, 0] -= 100          # 把月龄 0 的一部分挪到月龄 1，保持各分量在界内
Ly.n(xs)[0, 1] += 100
Ly.n(xs)[0, 1] += 2000         # 但总和超过 N → dem 约束必然冲突（Σn_in > N + ΣF）
print("月龄和 =", Ly.n(xs)[0].sum(), " > N =", prob.N[0])

slp = make_solver(mip=False)
triv = trivial_cut(2, xs, Ly.dim)
cn = ([triv.alpha], triv.beta.reshape(1, -1))
sub = build_stage_subproblem(1, xs, prob.pool[0], cn, prob)
try:
    sub.solve(slp)
    print("居然可行 —— 换更狠的扰动")
except RuntimeError as exc:
    print("已判不可行:", exc)
    out = Path(__file__).resolve().parent / "_iis_probe.lp"
    export_lp(sub.prob, out)
    print("现场 LP 已导出:", out.resolve(), out.stat().st_size, "bytes")
    sys.exit(0)
sys.exit(1)
