# _t_forward_heuristic.py — SA 前向可行性与 LP 口径一致性自检。
#
# 直接用 code.forward_heuristic.heuristic_step 构造状态矩阵上的阶段解；
# 用 code.sim_analysis.check_stage 审计 LP 的全部阶段约束；再核对五项成本
# 与 objective=stage_cost+theta。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.forward_heuristic import heuristic_step                # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.sim_analysis import check_stage                        # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

V2_CFG = ROOT / "test" / "_cfg_v2_heur"
T2_CFG = ROOT / "test" / "_cfg_tier2_cutpool_2h"
V2_CUT_RUN = ROOT / "results" / "v2_static10_4h_20261110_223050"


def _set_I(x, Ly, I):
    y = x.copy()
    y[:Ly.n_I] = np.asarray(I, dtype=float)
    return y


def _set_q(x, Ly, q):
    y = x.copy()
    y[Ly.q_slice()] = np.asarray(q, dtype=float)
    return y


def _set_n(x, Ly, n):
    y = x.copy()
    y[Ly.off_n:Ly.off_n + Ly.n_n] = np.asarray(n, dtype=float).ravel()
    return y


def _audit(prob, cfg, x, t, scene_idx, cuts, tag):
    Ly = prob.layout
    scene = prob.pool[scene_idx]
    h = heuristic_step(prob=prob, t=t, x=x, scene=scene, cfg=cfg,
                       cuts_next=cuts, rng=np.random.default_rng(123))
    assert h is not None
    aut = check_stage(
        prob, Ly, x, h.x_out, h.q_out, h.taup, h.taum, h.S_end, h.y,
        scene_idx, Ly.upper(prob), F_new=h.F_new,
        integer_mode=bool(cfg.integer_enabled),
        integer_scope=str(cfg.integer_scope),
        N=h.N if cfg.integer_scope == "tier2_full" else None)
    # 状态流转的 round/repair 在 grade/W 上允许 state_repair_tol 级残差。
    tol = max(float(prob.state_repair_tol), 1e-6)
    bad = {k: float(v) for k, v in aut.items() if v > tol}
    assert not bad, f"{tag}: 审计失败 {bad}"

    # 成本逐项必须与 LP 子问题的目标系数一致。
    n_in = Ly.n(x)
    q_in = Ly.q(x)
    I_in = Ly.I(x)
    y = h.y
    order = prob.C * float(q_in.sum())
    holding = 0.5 * prob.h * float(I_in.sum() + h.I_out.sum())
    downtime = prob.Cs * float(prob.N.sum() - n_in.sum())
    active = 0.5 * prob.atc * float(h.taup.sum() + h.taum.sum())
    passive = float(np.sum(prob.tc_route * y))
    cost = order + holding + downtime + active + passive
    assert abs(cost - h.stage_cost) <= 1e-7 * max(1.0, abs(cost)), (
        f"{tag}: 阶段成本不一致 {cost} vs {h.stage_cost}")
    theta = 0.0
    if cuts:
        theta = max(float(c.alpha + np.asarray(c.beta) @ h.x_out)
                    for c in cuts)
    assert abs(h.objective - (h.stage_cost + theta)) <= 1e-7 * max(
        1.0, abs(h.objective)), f"{tag}: objective 不含 theta"
    assert abs(h.cost_terms["theta"] - theta) <= 1e-7 * max(
        1.0, abs(theta)), f"{tag}: theta 不一致"
    if t > prob.T - prob.L:
        assert np.allclose(h.q_out, 0.0), f"{tag}: 终端前 q 未置 0"
    if cfg.integer_enabled:
        ids = Ly.integer_indices(cfg.integer_scope)
        vals = np.asarray(h.x_out, float)[ids]
        assert np.max(np.abs(vals - np.rint(vals))) <= cfg.int_tol
    return h


def _states(prob, tier2=False):
    Ly = prob.layout
    x0 = prob.initial_state()
    out = [("初始", x0)]
    out.append(("库存满", _set_I(x0, Ly, prob.Imax)))
    out.append(("q满", _set_q(x0, Ly, prob.Qmax)))
    n0 = Ly.n(x0)
    out.append(("机队全0级", _set_n(x0, Ly,
                                    np.column_stack([prob.N,
                                                     np.zeros((prob.N_R,
                                                               prob.G - 1))]))))
    out.append(("W全量", _set_n(x0, Ly, np.zeros_like(n0))))
    if tier2:
        # 保持整数行和为 N_r。
        n = np.zeros_like(n0, dtype=float)
        n[:, 0] = prob.N
        out.append(("Tier2全0级", _set_n(x0, Ly, n)))
    return out


def _run_case(name, cdir, cuts_by_stage):
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    print(f"\n=== {name}: T={prob.T} G={prob.G} N_R={prob.N_R} "
          f"scope={scfg.integer_scope} ===")
    for sname, x in _states(prob, scfg.integer_scope == "tier2_full"):
        for t in (1, max(1, prob.T // 2), max(1, prob.T - prob.L),
                  prob.T - prob.L + 1, prob.T):
            t = min(int(t), prob.T)
            if t < 1:
                continue
            si = int((hash((name, sname, t)) & 0x7fffffff)
                     % len(prob.pool))
            cuts = cuts_by_stage[t + 1] if t + 1 < len(cuts_by_stage) else []
            h = _audit(prob, scfg, x, t, si, cuts,
                       f"{name}/{sname}/t{t}")
            print(f"  {sname:12s} t={t:3d} cost={h.stage_cost:12.2f} "
                  f"obj={h.objective:12.2f} q={np.round(h.q_out, 2).tolist()} "
                  f"acc={h.sa_accepted}/{h.sa_moves} "
                  f"time={h.sa_time_s*1e3:.2f}ms")


def main():
    v2_cuts = [[] for _ in range(1000)]
    if V2_CUT_RUN.is_dir():
        p = load_problem_config(V2_CUT_RUN / "config" / "problem.toml")
        prob = DefineProblem(p)
        cuts, _ = load_cuts(V2_CUT_RUN, prob.T, prob.state_dim)
        v2_cuts = cuts
        for t, cs in enumerate(v2_cuts):
            for i, c in enumerate(cs):
                c.uid = ("unit", t, i)
    _run_case("V2", V2_CFG, v2_cuts)
    # Tier2 先用空割集验证尾部整数与全部约束；有 cuts 时也可直接传入。
    _run_case("Tier2", T2_CFG, [[] for _ in range(1000)])
    print("\nFORWARD_HEURISTIC_OK")


if __name__ == "__main__":
    main()
