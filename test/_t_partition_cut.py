# _t_partition_cut.py — 分区粗割的 Jensen/等价性硬门槛。
#   默认用 smoke 配置训练几轮；也可指定结果目录读取 V2 基线 cuts：
#     python test/_t_partition_cut.py
#     python test/_t_partition_cut.py results/base_T60_20261026_204105
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils.engine import SDDPEngine                                   # noqa: E402
from utils.io import load_cuts                                        # noqa: E402
from utils.partition import cluster_weights, initial_partition         # noqa: E402
from utils.payload import put_cuts                                    # noqa: E402
from utils.worker import (backward_cluster_task, backward_point_task,  # noqa: E402
                          init_worker, scene_signatures_task)


def main():
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if src is None:
        cdir = ROOT / "config" / "smoke"
    else:
        cdir = src / "config"
    pcfg = load_problem_config(cdir / "problem.toml")
    scfg = load_solver_config(cdir / "solver.toml")
    scfg.n_processes = 1
    scfg.partition.enabled = False
    prob = DefineProblem(pcfg)
    T, dim, n = prob.T, prob.state_dim, len(prob.pool)

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    try:
        if src is None:
            eng.train(3, verbose=False)
        else:
            cuts, _ = load_cuts(src, T, dim)
            eng.cuts = cuts
        # 一条 forward 路径，取各阶段输入状态
        states = eng.forward_paths([[0] * T])[0][0]
        if len(sys.argv) > 2:
            stages = [int(v) for v in sys.argv[2].split(",")]
        else:
            stages = [1, max(1, T // 2), T]
        print(f"problem={'smoke' if src is None else src.name}  T={T}  "
              f"scenes={n}  stages={stages}")
        print("stage |   exact Q   |  L=1 gap  |  L=4 gap  | L=16 gap  | L=all obj/beta")
        worst = np.inf
        for t in stages:
            x = np.asarray(states[t - 1], dtype=float)
            blk = put_cuts(eng.cuts[t + 1] if t < T else [], dim)
            try:
                be, qe = backward_point_task((999, t, 0, x, list(range(n)), blk.ref))
                gaps = []
                for k in (1, 4, 16):
                    parts = initial_partition(prob.pool, n, k)
                    w = cluster_weights(parts, n)
                    bc, qc = backward_cluster_task((999, t, 0, x, parts, w, blk.ref))
                    gap = float(qe - qc)
                    worst = min(worst, gap)
                    assert gap >= -1e-6, (t, k, gap)
                    gaps.append(gap)
                parts = [[i] for i in range(n)]
                w = np.full(n, 1.0 / n)
                ba, qa = backward_cluster_task((999, t, 0, x, parts, w, blk.ref))
                # 目标值必须一致；β 在 LP 退化时可以取不同最优对偶，不要求逐位相同。
                eq_q = abs(float(qe - qa))
                eq_b = float(np.max(np.abs(be - ba)))
                assert eq_q <= 1e-6, (t, eq_q)
                print(f"{t:5d} | {qe:11.4f} | {gaps[0]:9.4f} | {gaps[1]:9.4f} | "
                      f"{gaps[2]:9.4f} | {eq_q:.2e}/{eq_b:.2e}")
            finally:
                blk.close()
        # 签名可读、不含 cut 行；一次 LP 同时给出 Q/pi
        blk = put_cuts(eng.cuts[2] if T >= 2 else [], dim)
        try:
            inds, sigs, objs = scene_signatures_task(
                (999, 1, 0, np.asarray(states[0], float), list(range(3)), blk.ref))
            assert sigs.shape[0] == 3 and sigs.shape[1] > 0
            print(f"signature: shape={sigs.shape} objs={np.round(objs, 4).tolist()}")
        finally:
            blk.close()
        print(f"min Jensen gap = {worst:.6f}（应 >= -1e-6）")
        print("OK _t_partition_cut")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
