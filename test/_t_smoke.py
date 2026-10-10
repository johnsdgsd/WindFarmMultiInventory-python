# 子问题层自检：单期求解 + 对偶提取四项（.dj vs π、次梯度不等式、转移残差、割合法性）。
#   python test/_t_smoke.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.problem_data import DefineProblem          # noqa: E402
from code.model import build_stage_subproblem        # noqa: E402
from utils import Cut, make_solver, trivial_cut      # noqa: E402
from utils import diagnostics as dg                  # noqa: E402

prob = DefineProblem(T=4, A=24, N_R=3, pool_size=30, spread_init=True,
                     m=3.0, eta=8.0)
Ly = prob.layout
print(f"T={prob.T} A={prob.A} N_R={prob.N_R} dim={Ly.dim} pool={len(prob.pool)}")
x0 = prob.initial_state()
n0 = Ly.n(x0)
print("x0: I 合计 =", Ly.I(x0).sum(), " o 合计 =", Ly.o(x0).sum(),
      " q =", Ly.q(x0), " n 合计 =", n0.sum(), " N 合计 =", float(np.sum(prob.N)))
print("x0 月龄分布(仓0) =", n0[0][:12].astype(int), "...")

smip = make_solver(mip=True)
slp = make_solver(mip=False)

# ---- 构造内部状态（各分量都离边界 > 1）----
xs = x0.copy()
Iv = Ly.I(xs); Iv[:] = [30.0, 20.0, 15.0]
ov = Ly.o(xs); ov[:] = 4.0
qv = Ly.q(xs); qv[:] = 2.0
nv = Ly.n(xs)
mv = int(nv[1, 0] * 0.3); nv[1, 0] -= mv; nv[1, 7] += mv
print("内部状态: I=", Ly.I(xs), " o=", Ly.o(xs), " q=", Ly.q(xs),
      " n 合计 =", Ly.n(xs).sum())

scene = prob.pool[3]
triv = trivial_cut(stage=2, state=xs, dim=Ly.dim)
cuts_next = ([triv.alpha], triv.beta.reshape(1, -1))

# ---- 虚拟首阶段：无割（纯常量目标）----
sub0 = build_stage_subproblem(0, x0, None, None, prob).solve(smip)
print("t=0 obj =", sub0.obj, " q_out =", [round(v, 3) for v in sub0.decision_values()["q"]])

# ---- 真实期 ----
for x, tag in ((x0, "初值"), (xs, "内部")):
    sub = build_stage_subproblem(1, x, scene, cuts_next, prob).solve(smip)
    dv = sub.decision_values()
    print(f"[{tag}] t=1 obj={sub.obj:.4f} stage_cost={sub.stage_cost():.4f} "
          f"tau+={dv['taup']} tau-={dv['taum']} S_end={[round(v,2) for v in dv['S_end']]}")
    print(f"      q_out={dv['q']}  被动 y 合计={np.sum(dv['y']):.2f}  I_out={np.round(dv['I_out'],2)}")

# ---- 对偶提取自检（内部状态）----
r1 = dg.dual_crosscheck(build_stage_subproblem, prob, slp, 1, xs, scene, cuts_next)
print("① dj vs duals:", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in r1.items()})

comps = [Ly.ix_I(0), Ly.ix_n(1, 5), Ly.ix_n(2, 0), Ly.ix_q(2), Ly.ix_o(0, 1),
         Ly.ix_n(0, 23)]
r2 = dg.subgradient_inequality(build_stage_subproblem, prob, slp, 1, xs, scene,
                               cuts_next, comps)
print(f"② 次梯度不等式：checked={r2['n_checked']} skipped={r2['n_skipped']} ok={r2['ok']}")
for row in r2["rows"]:
    print("   ", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()})

# ---- 状态转移一致性：n_out[a+1] = n_in[a] − F[a] ----
F = np.rint(scene[1] * Ly.n(xs)).astype(int)
nout = Ly.n(sub.out_state_vector())
print("③ 月龄转移残差 max =", int(np.abs(nout[:, 1:] - (Ly.n(xs)[:, :-1] - F[:, :-1])).max()))
print("   库存动态残差 max =", float(np.abs(Ly.I(sub.out_state_vector())
      - (np.asarray(dv["S_end"]) + np.asarray(dv["taup"]) - np.asarray(dv["taum"]))).max()))
print("   需求约束松弛 =", [round(sub.prob.constraints[f"dem_{r}"].value(), 4)
                           for r in range(prob.N_R)])

# ---- 割的合法性 ----
rng = np.random.default_rng(0)
cut = Cut(alpha=0.0, beta=np.zeros(Ly.dim), state=xs, stage=1, obj=0.0, it=0)
probes = dg.probe_states(rng, prob, n=3)
print("3b 平凡割合法性:", dg.cut_validity(build_stage_subproblem, prob, slp, cut, probes, cuts_next, scene))
