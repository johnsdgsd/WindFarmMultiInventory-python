# _t_strengthened_state.py — Strengthened Benders 状态级小规模验证。
#
# 选最近一轮生成的割（其割值 = 当前 Q_LP 在锚点的 300 场景均值），
# 对锚点状态计算标准割值 V_before，再做 exact lift，检查：
#   1) MIP 必须 BestObj≈BestBnd；
#   2) 提升后截距/状态价值的变化；
#   3) 失败场景回退，不把未证 incumbent 当截距。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"
CASES = ((60, 1921), (10, 254))
N_SCENES = 30
TOL = 1e-7


def run_case(prob, cuts, scfg, stage: int, idx: int, n_scenes: int) -> None:
    c = cuts[stage][idx]
    S = np.asarray(c.state, dtype=float)
    vals = np.array([z.alpha + z.beta @ S for z in cuts[stage]])
    order = np.argsort(-vals)
    v_before = float(vals[idx])
    margin = float(vals[order[0]] - vals[order[1]])
    assert int(order[0]) == idx, f"stage {stage} cut {idx} 不是支配割"
    next_cuts = (np.array([z.alpha for z in cuts[stage + 1]], dtype=float),
                 np.vstack([z.beta for z in cuts[stage + 1]])) if stage < prob.T \
        else (np.array([]), np.empty((0, prob.state_dim)))
    print(f"\nstage={stage} cut={idx} it={c.it}  "
          f"V_before={v_before:.6f}  margin={margin:.6f}  "
          f"scenes={n_scenes}")
    sp = StageSubproblem(stage, prob, scfg)
    lps, alphas, gaps = [], [], []
    n_fb = 0
    try:
        sp.assemble(next_cuts, key=None)
        # 标准割值用全 300 场景复算；提升只用前 n_scenes 个场景控制验证耗时。
        for si in range(len(prob.pool)):
            sp.set_point(S, prob.pool[si], require_int=True)
            sp.solve_lp()
            lp = float(sp.obj)
            lps.append(lp)
            if si >= n_scenes:
                continue
            pi = np.asarray(sp.gradient("dj"), dtype=float)
            nu = sp.solve_lift(pi, time_limit=5.0, rel_gap=0.0,
                               safety_margin=0.0, value="exact")
            info = sp._last_lift_info
            if nu is None:
                n_fb += 1
                continue
            assert info is not None and info["exact_ok"], info
            assert abs(info["best_obj"] - info["best_bnd"]) <= TOL * max(
                1.0, abs(info["best_obj"]))
            gaps.append(info["best_obj"] - info["best_bnd"])
            alphas.append(float(nu + pi @ S - lp))
    finally:
        sp.free()
    lp_mean = float(np.mean(lps))
    a = np.asarray(alphas, dtype=float)
    std_at_S = lp_mean
    v_lift = std_at_S + float(a.mean()) if a.size else v_before
    v_after = max(v_before, v_lift)
    print(f"  Q_LP(S)={lp_mean:.6f}  fresh_std={std_at_S:.6f}  "
          f"alpha mean={a.mean():.6f} median={np.median(a):.6f} "
          f"min={a.min():.6f} max={a.max():.6f}")
    print(f"  V_lift={v_lift:.6f}  V_after=max(V_before,V_lift)={v_after:.6f}  "
          f"dV={v_after - v_before:.6f}  fallback={n_fb}/{n_scenes}")
    print(f"  BestBnd-BestObj max={max(gaps) if gaps else 0:.3e}  "
          f"exact_n={a.size}")
    assert a.size > 0
    # 回退是工具包允许的稳定路径：MIP 没证到最优就回退 LP 割。
    # 终端阶段应全部精确；中层阶段允许回退，报告实际比例。
    if stage == 60:
        assert n_fb == 0, f"终端 exact MIP 回退 {n_fb}/{n_scenes}"
    assert max(gaps) <= 5e-3, ("BestBnd−BestObj 过大，"
                               "提升量无法分辨", max(gaps))
    assert abs(std_at_S - v_before) <= 1e-4, (
        "最近一轮割值应与当前 Q_LP 一致；否则该割不能作为本次提升对象",
        std_at_S, v_before)


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    for stage, idx in CASES:
        run_case(prob, cuts, scfg, stage, idx, N_SCENES)
    print("\nOK _t_strengthened_state")


if __name__ == "__main__":
    main()
