# 直接求解「第 1 期在 x₀ 处」的后向 LP，看它凭什么把目标做到 0。
#
#   python test/_t_q1.py <结果目录> [期号=1] [场景数=3]
#
# 打印每场景的 obj / theta / 本期成本，以及 F、停机、安装 y、订货 q、I_out，
# 定位「零成本」是靠哪个自由度实现的。
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from utils import copt_env as ce                                  # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils.cut import pack                                        # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def main():
    d = Path(sys.argv[1])
    d = d if d.is_absolute() else ROOT / d
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_sc = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, NR, A = prob.layout, prob.N_R, prob.A
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    x0 = prob.initial_state()          # 第 1 期收到的状态 = x₀
    print(f"阶段 {t} 在 x₀ 处求解（割来自 cuts[{t+1}]，共 {len(cuts[t+1]) if t < prob.T else 0} 条）")
    print(f"x₀：Σn={Ly.n(x0).sum():.0f}/{prob.N.sum():.0f}  ΣI={Ly.I(x0).sum():.0f}  "
          f"Σq={Ly.q(x0).sum():.0f}")

    sub = StageSubproblem(t, prob, scfg)
    nxt = pack(cuts[t + 1], prob.state_dim) if t < prob.T else None
    sub.assemble(nxt)
    sub.set_state(x0)
    for k in range(n_sc):
        scene = prob.pool[k]
        sub.set_scene(scene)
        sub.solve_lp()
        dv = sub.decision_values()
        nout = np.asarray(dv["n_out"], dtype=float)
        F = np.asarray(ce.values(sub.model, [v for row in sub.F for v in row]),
                       dtype=float).reshape(NR, A)
        thn = np.asarray(prob.pool[k][1], dtype=float) * Ly.n(x0)
        print(f"\n  场景 {k}: obj={sub.obj:,.4f}  theta={sub.theta_value():,.4f}  "
              f"本期成本={sub.stage_cost():,.4f}  "
              f"ΣF={F.sum():.1f}  Σrint(θn)={np.rint(thn).sum():.1f}  "
              f"Σ(θn−0.5)⁺={np.maximum(thn-0.5,0).sum():.1f}")
        print(f"    F 相对区间下界（rint 口径 [θn−0.5, θn+0.5]）："
              f"中位 {np.median((F-thn).ravel()):+.3f}  "
              f"贴下界的桶 {int(((F-thn) <= -0.49).sum())}/{NR*A}")
        print(f"    Σn_out={nout.sum():,.1f}  停机={prob.N.sum()-nout.sum():,.1f}  "
              f"I_out={np.round(dv['I_out'],2).tolist()}  q_out={np.round(dv['q'],2).tolist()}")
        print(f"    τ⁺={np.round(dv['taup'],2).tolist()}  τ⁻={np.round(dv['taum'],2).tolist()}")
        ysum = np.asarray(dv["y"], dtype=float).sum(axis=0)
        print(f"    安装入各仓 Σ_y[m,r]={np.round(ysum,2).tolist()}")
    sub.free()


if __name__ == "__main__":
    main()
