# _t_policy2_roll12.py — 12 期滚动窗口 MILP 前向（Tier2）。
#
# 每个 t：当前期用真实场景；未来 H-1 期用场景池均值 theta 作为预测，
# 建联合多阶段 MILP；终端 cuts[t+H] 近似 V_{t+H}。
# 解出后只执行第一阶段的决策，再滚动到 t+1。
# 这是因果的期望值滚动策略；不是完美信息，也不是完整场景树。
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import coptpy as cp
from coptpy import LinExpr, quicksum

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from code.transition import tail_cdf                             # noqa: E402
from utils import copt_env as ce                                 # noqa: E402
from utils.cut import pack                                       # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

INF = cp.COPT.INFINITY


def add_stage(m, prob, cfg, tag, xin, scene, H):
    """在 m 里加一个真实阶段；xin 是 40 个常数或 Var。返回 vars。"""
    Ly = prob.layout
    N_R, G, L = Ly.N_R, Ly.G, Ly.L
    ub = Ly.upper(prob)
    xout = [m.addVar(lb=0.0, ub=float(ub[j]), vtype=cp.COPT.INTEGER,
                     name=f"{tag}_xout_{j}") for j in range(Ly.dim)]
    I_in = [xin[Ly.ix_I(r)] for r in range(N_R)]
    I_out = [xout[Ly.ix_I(r)] for r in range(N_R)]
    o_in = [[xin[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
    o_out = [[xout[Ly.ix_o(l, r)] for r in range(N_R)] for l in range(L - 1)]
    n_in = [[xin[Ly.ix_n(r, i)] for i in range(G)] for r in range(N_R)]
    n_out = [[xout[Ly.ix_n(r, i)] for i in range(G)] for r in range(N_R)]
    q_in = [xin[Ly.ix_q(r)] for r in range(N_R)]
    q_out = [xout[Ly.ix_q(r)] for r in range(N_R)]

    cap_sum = float(np.sum(prob.Imax + prob.Qmax))
    taup, taum = [], []
    for r in range(N_R):
        up = float(min(cap_sum, prob.Imax[r]))
        dn = float(prob.Imax[r] + prob.Qmax[r])
        taup.append(m.addVar(lb=0.0, ub=up, vtype=cp.COPT.INTEGER,
                             name=f"{tag}_taup_{r}"))
        taum.append(m.addVar(lb=0.0, ub=dn, vtype=cp.COPT.INTEGER,
                             name=f"{tag}_taum_{r}"))
    y = [[m.addVar(lb=0.0, ub=float(min(prob.N[r], prob.Imax[mm])),
                   vtype=cp.COPT.INTEGER, name=f"{tag}_y_{r}_{mm}")
          for mm in range(N_R)] for r in range(N_R)]
    S_end = [m.addVar(lb=0.0, ub=float(prob.Imax[r] + prob.Qmax[r]),
                      vtype=cp.COPT.INTEGER, name=f"{tag}_send_{r}")
             for r in range(N_R)]
    tau = [taup[r] - taum[r] for r in range(N_R)]

    N = [[[None] * (G + 1) for _ in range(G)] for _ in range(N_R)]
    for r in range(N_R):
        for i in range(G):
            for j in range(i, G + 1):
                N[r][i][j] = m.addVar(lb=0.0, ub=float(prob.N[r]),
                                      vtype=cp.COPT.INTEGER,
                                      name=f"{tag}_N_{r}_{i}_{j}")

    # pas
    for mm in range(N_R):
        m.addConstr(quicksum(y[r][mm] for r in range(N_R)) - I_in[mm] <= 0.0,
                    name=f"{tag}_pas_{mm}")
    # Nsum / Tail
    ct = tail_cdf(np.asarray(scene[1], dtype=float))
    for r in range(N_R):
        for i in range(G):
            m.addConstr(quicksum(N[r][i][j] for j in range(i, G + 1))
                        - n_in[r][i] == 0.0, name=f"{tag}_Nsum_{r}_{i}")
            for j in range(i + 1, G + 1):
                c = float(ct[r, i, j])
                expr = quicksum(N[r][i][k] for k in range(j, G + 1))
                if isinstance(n_in[r][i], (int, float)):
                    expr = expr - c * float(n_in[r][i])
                else:
                    expr = expr - c * n_in[r][i]
                row = m.addConstr(expr, cp.COPT.LESS_EQUAL, 0.5,
                                  name=f"{tag}_Tail_{r}_{i}_{j}")
                m.setInfo(cp.COPT.Info.LB, [row], -0.5)

    # grade
    for r in range(N_R):
        m.addConstr(n_out[r][0] - quicksum(y[r][mm] for mm in range(N_R))
                    - N[r][0][0] == 0.0, name=f"{tag}_grade0_{r}")
        for j in range(1, G):
            m.addConstr(n_out[r][j] - quicksum(N[r][i][j]
                                                for i in range(j + 1)) == 0.0,
                        name=f"{tag}_grade_{r}_{j}")
    # dem / fleet
    for r in range(N_R):
        m.addConstr(quicksum(y[r][mm] for mm in range(N_R))
                    + quicksum(N[r][i][j] for i in range(G)
                               for j in range(i, G))
                    <= float(prob.N[r] + prob.fleet_tol),
                    name=f"{tag}_dem_{r}")
        m.addConstr(quicksum(n_in[r][i] for i in range(G))
                    <= float(prob.N[r] + prob.fleet_tol),
                    name=f"{tag}_fleet_{r}")
    # logistics
    for r in range(N_R):
        m.addConstr(S_end[r] - I_in[r]
                    + quicksum(y[s][r] for s in range(N_R)) - o_in[0][r] == 0.0,
                    name=f"{tag}_send_def_{r}")
    m.addConstr(quicksum(tau) == 0.0, name=f"{tag}_tau_bal")
    for r in range(N_R):
        m.addConstr(S_end[r] + tau[r] >= 0.0, name=f"{tag}_tau_lo_{r}")
        m.addConstr(S_end[r] + tau[r] <= float(prob.Imax[r]),
                    name=f"{tag}_tau_hi_{r}")
        m.addConstr(tau[r] - quicksum(S_end[mm] for mm in range(N_R)
                                      if mm != r) <= 0.0,
                    name=f"{tag}_tau_src_{r}")
    for r in range(N_R):
        m.addConstr(I_out[r] - S_end[r] - tau[r] == 0.0,
                    name=f"{tag}_inv_{r}")
    for l in range(L - 2):
        for r in range(N_R):
            m.addConstr(o_out[l][r] - o_in[l + 1][r] == 0.0,
                        name=f"{tag}_pipe_{l}_{r}")
    for r in range(N_R):
        m.addConstr(o_out[L - 2][r] - q_in[r] == 0.0,
                    name=f"{tag}_pipe_last_{r}")
        m.addConstr(I_out[r] + quicksum(o_out[l][r] for l in range(L - 1))
                    + q_out[r] <= float(prob.Imax[r]),
                    name=f"{tag}_cap_{r}")

    order = prob.C * quicksum(q_in)
    holding = prob.h / 2.0 * (quicksum(I_in) + quicksum(I_out))
    downtime = (prob.Cs * float(np.sum(prob.N))
                - prob.Cs * quicksum(n_in[r][i] for r in range(N_R)
                                     for i in range(G)))
    active = prob.atc / 2.0 * quicksum(taup + taum)
    passive = quicksum(float(prob.tc_route[r][mm]) * y[r][mm]
                       for r in range(N_R) for mm in range(N_R))
    cost = order + holding + downtime + active + passive
    return {"xout": xout, "taup": taup, "taum": taum, "y": y,
            "S_end": S_end, "N": N, "cost": cost}


def stage_cost_from(prob, xin, st):
    def val(v):
        return float(v)
    return None


def solve_roll12(prob, cfg, cuts, scene_path, path_idx, H=12,
                 time_limit=10.0):
    Ly = prob.layout
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    # stage 0：当前单阶段策略选初始订货
    s0 = make_stage_subproblem(0, prob, cfg)
    s0.assemble(pack(cuts[1], prob.state_dim), key=("p2", 0, path_idx))
    s0.set_state(x0, require_int=True)
    s0.solve_mip()
    x = np.asarray(s0.out_state_vector(), float)
    s0.free()
    total = 0.0
    fallbacks = 0
    # 未来期的预测场景：场景池均值 theta；当前期仍用真实场景。
    th_mean = np.mean([np.asarray(th, float) for _, th in prob.pool], axis=0)
    mean_scene = (None, th_mean)
    for t in range(1, prob.T + 1):
        h = min(H, prob.T - t + 1)
        m = ce.new_model(cfg)
        stages = []
        # 首期状态用固定变量表示，避免把 -c*n_in 变成行内常数；
        # 否则 Tail 行的 Lb/Ub 会被 c*n_in 平移，范围 -0.5..0.5 失效。
        xin = [m.addVar(lb=float(x[j]), ub=float(x[j]),
                        vtype=cp.COPT.INTEGER, name=f"x0_{j}")
               for j in range(Ly.dim)]
        for k in range(h):
            tt = t + k
            sc = (prob.pool[int(scene_path[tt - 1])] if k == 0
                  else mean_scene)
            st = add_stage(m, prob, cfg, f"s{k}", xin, sc, h)
            stages.append(st)
            xin = st["xout"]
        cost = sum((st["cost"] for st in stages[1:]), stages[0]["cost"])
        theta = None
        tcut = t + h
        if tcut <= prob.T and cuts[tcut]:
            theta = m.addVar(lb=-INF, ub=INF, obj=1.0, name="theta")
            xlast = stages[-1]["xout"]
            for c in cuts[tcut]:
                expr = theta - quicksum(float(c.beta[j]) * xlast[j]
                                        for j in range(Ly.dim)
                                        if abs(c.beta[j]) > 1e-12)
                m.addConstr(expr >= float(c.alpha), name="cut")
            cost = cost + theta
        m.setObjective(cost, cp.COPT.MINIMIZE)
        m.setParam(cp.COPT.Param.RelGap, 0.001)
        m.setParam(cp.COPT.Param.AbsGap, 0.0)
        m.setParam(cp.COPT.Param.TimeLimit, float(time_limit))
        try:
            m.solve()
            ok = int(m.getAttr(cp.COPT.Attr.HasMipSol)) == 1
        except Exception:
            ok = False
        if not ok:
            # 回退到单阶段 MIP
            fallbacks += 1
            ce.close_model(m)
            sub = make_stage_subproblem(t, prob, cfg)
            pn = (pack(cuts[t + 1], prob.state_dim) if t < prob.T
                  else (np.zeros(0), np.zeros((0, prob.state_dim))))
            sub.assemble(pn, key=("p2", t, path_idx, "fb"))
            sub.set_state(x, require_int=True)
            sub.set_scene(prob.pool[int(scene_path[t - 1])])
            sub.solve_mip()
            total += float(sub.stage_cost())
            x = np.asarray(sub.out_state_vector(), float)
            sub.free()
            continue
        # 第一阶段成本
        st = stages[0]
        def vv(var):
            return float(m.getInfo(cp.COPT.Info.Value, [var])[0])
        I_in = [float(x[Ly.ix_I(r)]) for r in range(Ly.N_R)]
        I_out = [vv(st["xout"][Ly.ix_I(r)]) for r in range(Ly.N_R)]
        n_in = [float(x[Ly.ix_n(r, i)]) for r in range(Ly.N_R)
                for i in range(Ly.G)]
        n_out = [vv(st["xout"][Ly.ix_n(r, i)]) for r in range(Ly.N_R)
                 for i in range(Ly.G)]
        q_in = [float(x[Ly.ix_q(r)]) for r in range(Ly.N_R)]
        taup = [vv(v) for v in st["taup"]]
        taum = [vv(v) for v in st["taum"]]
        yv = [[vv(v) for v in row] for row in st["y"]]
        cost_t = (prob.C * sum(q_in)
                  + prob.h / 2.0 * (sum(I_in) + sum(I_out))
                  + prob.Cs * (float(np.sum(prob.N)) - sum(n_in))
                  + prob.atc / 2.0 * (sum(taup) + sum(taum))
                  + sum(float(prob.tc_route[r][mm]) * yv[r][mm]
                        for r in range(Ly.N_R) for mm in range(Ly.N_R)))
        total += cost_t
        x = np.asarray([vv(v) for v in st["xout"]], float)
        x = np.rint(x)
        ce.close_model(m)
    return total, fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",
                    default="results/int_t2_resume2h_final2_20261108_213100")
    ap.add_argument("--paths", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=10.0)
    args = ap.parse_args()
    d = ROOT / args.dir
    prob = DefineProblem(load_problem_config(d / "config" / "problem.toml"))
    cfg = load_solver_config(d / "config" / "solver.toml")
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    rng = np.random.default_rng(args.seed)
    sps = [rng.integers(0, len(prob.pool), size=prob.T).tolist()
           for _ in range(args.paths)]
    print(f"rounds={meta.get('n_iterations')} H={args.horizon} "
          f"paths={args.paths}", flush=True)
    t0 = time.time()
    totals = []
    fb = 0
    for i, sp in enumerate(sps):
        val, f = solve_roll12(prob, cfg, cuts, sp, i, args.horizon,
                              args.time_limit)
        totals.append(val)
        fb += f
        print(f"  path {i+1}/{len(sps)} total={val:.4f} fallback={f}",
              flush=True)
    arr = np.asarray(totals)
    se = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    print(f"policy2 roll{args.horizon}: mean={arr.mean():.4f} "
          f"std={arr.std(ddof=1) if arr.size>1 else 0:.4f} se={se:.4f} "
          f"fallback={fb} time={time.time()-t0:.1f}s")
    np.savez(ROOT / "test" / "_policy2_roll12_totals.npz", totals=arr)


if __name__ == "__main__":
    main()
