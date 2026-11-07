# _t_nogood_masterloop.py — master-guided 二元 No-good 割的 stage0 循环。
#
# 对每次 stage0 master 选出的 q，求固定状态 ell(q)，加入一条二元 No-good 割，
# 再解 master。比较 L=0 与区域下界 L=Lq 两种设置。
# 区域 r 固定为非 q 分量；q 是 stage0 唯一自由状态分量。
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from coptpy import COPT, quicksum

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem                      # noqa: E402
from utils import copt_env as ce                                  # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

RESULT = ROOT / "results" / "int_t2_formal_20261106_180256"
TIME_LIMIT = 0.5
MAX_ITERS = 8


def _ell_q(prob, scfg, cuts, S):
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
    vals = []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(S, omega, require_int=True)
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            vals.append(info["best_bnd"])
    finally:
        sp.free()
    return float(np.mean(vals))


def _L_region(prob, scfg, cuts, x0):
    Ly = prob.layout
    qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
    ub = np.asarray(Ly.upper(prob), float)[qidx]
    next_cuts = (np.array([c.alpha for c in cuts[2]]),
                 np.vstack([c.beta for c in cuts[2]]))
    sp = StageSubproblem(1, prob, scfg)
    vals = []
    try:
        sp.assemble(next_cuts, key=None)
        for omega in prob.pool:
            sp.set_point(x0, omega, require_int=True)
            for j, u in zip(qidx, ub):
                sp.model.setInfo(COPT.Info.LB, [sp.x_in[j]], 0.0)
                sp.model.setInfo(COPT.Info.UB, [sp.x_in[j]], float(u))
            info = ce.solve_bound(sp.model, time_limit=TIME_LIMIT,
                                  rel_gap=0.0, abs_gap=0.0,
                                  value="best_bnd", return_info=True)
            vals.append(info["best_bnd"])
    finally:
        sp.free()
    return float(np.mean(vals))


class Stage0Master:
    def __init__(self, prob, scfg, cuts, x0, L):
        Ly = prob.layout
        self.qidx = list(range(Ly.off_q, Ly.off_q + Ly.n_q))
        self.ub = np.asarray(Ly.upper(prob), float)[self.qidx]
        self.L = float(L)
        self.base = StageSubproblem(0, prob, scfg)
        self.base.assemble((np.array([c.alpha for c in cuts[1]]),
                            np.vstack([c.beta for c in cuts[1]])), key=None)
        self.base.set_state(x0, require_int=True)
        m = self.m = self.base.model
        q = self.q = [self.base.x_out[j] for j in self.qidx]
        self.y = []
        for r, u in enumerate(self.ub):
            ys = [m.addVar(vtype=COPT.BINARY, name=f"y_{r}_{v}")
                  for v in range(int(u) + 1)]
            m.addConstr(quicksum(ys) == 1.0)
            m.addConstr(quicksum(v * ys[v] for v in range(int(u) + 1))
                        == q[r])
            self.y.append(ys)
        self.anchors = set()
        self.ncut = 0

    def add_cut(self, qv, ell):
        qv = [int(v) for v in qv]
        if tuple(qv) in self.anchors:
            return False
        d = []
        for r, v in enumerate(qv):
            d.append(1.0 - self.y[r][v])
        z = self.m.addVar(vtype=COPT.BINARY,
                          name=f"z_{self.ncut}")
        for t in d:
            self.m.addConstr(z >= t)
        self.m.addConstr(z <= quicksum(d))
        # ell_eff >= L
        ell_eff = max(float(ell), self.L)
        self.m.addConstr(self.base._theta + (ell_eff - self.L) * z
                         >= ell_eff)
        self.anchors.add(tuple(qv))
        self.ncut += 1
        return True

    def solve(self):
        info = ce.solve_bound(self.m, time_limit=60.0, rel_gap=0.0,
                              abs_gap=0.0, value="auto_strict",
                              exact_tol=1e-9, return_info=True)
        qv = [int(round(v)) for v in
              ce.values(self.m, self.q)]
        return float(info["value"]), qv

    def free(self):
        self.base.free()


def main():
    pcfg = load_problem_config(RESULT / "config" / "problem.toml")
    scfg = load_solver_config(RESULT / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RESULT, prob.T, prob.state_dim)
    x0 = np.round(np.asarray(prob.initial_state(), float), 6)
    t0 = time.time()
    Lq = _L_region(prob, scfg, cuts, x0)
    print(f"L_q={Lq:.6f}  (time={time.time()-t0:.1f}s)")
    cache = {}

    def ell(S):
        key = tuple(int(round(v)) for v in S[prob.layout.q_slice()])
        if key not in cache:
            cache[key] = _ell_q(prob, scfg, cuts, S)
        return cache[key]

    for tag, L in (("L=0", 0.0), ("L=Lq", Lq)):
        print(f"\n--- master-guided {tag} ---")
        m = Stage0Master(prob, scfg, cuts, x0, L)
        try:
            for k in range(MAX_ITERS):
                lb, qv = m.solve()
                S = x0.copy()
                S[prob.layout.q_slice()] = qv
                e = ell(S)
                print(f"iter {k}: LB={lb:.6f}  q={qv}  ell={e:.6f}  "
                      f"ncut={m.ncut}")
                if not m.add_cut(qv, e):
                    print("  anchor already exists; stop")
                    break
        finally:
            m.free()
    print(f"\nOK _t_nogood_masterloop  cache={len(cache)}  "
          f"time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
