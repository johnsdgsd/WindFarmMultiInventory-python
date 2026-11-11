# _t_qperturb_heuristic.py — 旧 q 扰动 + LP 重解启发式的可行性与耗时自检。
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                      # noqa: E402
from code.sim_analysis import check_stage                        # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.io import load_cuts                                   # noqa: E402

RUN = ROOT / "results" / "v2_static10_4h_20261110_223050"
CFG = ROOT / "test" / "_cfg_v2_qperturb"


def main():
    pcfg = load_problem_config(CFG / "problem.toml")
    scfg = load_solver_config(CFG / "solver.toml")
    prob = DefineProblem(pcfg)
    cuts, _ = load_cuts(RUN, prob.T, prob.state_dim)
    for t, cs in enumerate(cuts):
        for i, c in enumerate(cs):
            c.uid = ("qp-test", t, i)
    Ly = prob.layout
    x0 = prob.initial_state()

    lp_t, qp_t, ratios, gaps = [], [], [], []
    try:
        for t in (1, 10, 30, 45, 56, 57, 60):
            cc = cuts[t + 1] if t + 1 < len(cuts) else []
            for si in (0, 7, 19):
                scene = prob.pool[si]
                # LP 基线计时（同一模型热启动一次）
                sub = make_stage_subproblem(1, prob, scfg)
                sub.set_cuts(cc)
                sub.set_point(x0, scene)
                sub.solve_lp()
                t0 = time.perf_counter()
                sub.solve_lp()
                lp_t.append(time.perf_counter() - t0)
                base_obj = float(sub.obj)
                sub.free()

                sub = make_stage_subproblem(1, prob, scfg)
                sub.set_cuts(cc)
                t0 = time.perf_counter()
                qr = sub.qperturb_step(
                    x0, scene, rng=np.random.default_rng(100 + t + si))
                qp_t.append(time.perf_counter() - t0)
                assert qr.n_solves == 2
                assert qr.perturbed and not qr.failed
                assert not np.allclose(qr.q_new, qr.q_base)
                assert qr.objective >= base_obj - 1e-6 * max(1.0, abs(base_obj))
                dec = qr.decisions
                aut = check_stage(
                    prob, Ly, x0, qr.x_out, dec["q"], dec["taup"],
                    dec["taum"], dec["S_end"], dec["y"], si,
                    Ly.upper(prob), F_new=dec["F_new"],
                    integer_mode=False, integer_scope="none", N=None)
                tol = max(float(prob.state_repair_tol), 1e-6)
                bad = {k: v for k, v in aut.items() if v > tol}
                assert not bad, (t, si, bad)
                gaps.append(qr.objective - base_obj)
                ratios.append(qp_t[-1] / lp_t[-1])
                print(f"t={t:3d} si={si:2d} base={base_obj:12.2f} "
                      f"qp={qr.objective:12.2f} qbase={np.round(qr.q_base, 2)} "
                      f"qnew={np.round(qr.q_new, 2)} "
                      f"LP={lp_t[-1]*1e3:7.3f}ms QP={qp_t[-1]*1e3:7.3f}ms")
                sub.free()
    finally:
        pass

    lp = np.asarray(lp_t)
    qp = np.asarray(qp_t)
    print(f"\npoints={len(lp)}")
    print(f"LP time median={np.median(lp)*1e3:.3f}ms")
    print(f"QP time median={np.median(qp)*1e3:.3f}ms")
    print(f"QP/LP time ratio median={np.median(ratios):.3f}")
    print(f"QP-base objective gap mean={np.mean(gaps):.3f} "
          f"min={np.min(gaps):.3f} max={np.max(gaps):.3f}")
    print("QPERTURB_OK")


if __name__ == "__main__":
    main()
