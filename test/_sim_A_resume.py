# _sim_A_resume.py — 方案 A（v6 全局布局）的分块可续跑 3000 路径仿真。
#   python test/_sim_A_resume.py <结果目录> [总路径] [块大小] [进程数]
# 每完成一块原子写 sim_partial_<P>.npz；被杀后重跑从已完成块继续。
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config  # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils import SDDPEngine                                      # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def _save(path: Path, totals, stage_sum, done):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, totals=np.asarray(totals),
                        stage_sum=np.asarray(stage_sum), done=int(done))
    tmp.replace(path)


def main(d_arg, n_paths=3000, chunk=1000, nproc=8):
    d = Path(d_arg)
    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    scfg.n_processes = int(nproc)
    prob = DefineProblem(pcfg)
    T = prob.T
    partial = d / f"sim_partial_{n_paths}.npz"
    totals, stage_sum, done = [], np.zeros(T), 0
    if partial.is_file():
        z = np.load(partial)
        totals = z["totals"].tolist()
        stage_sum = z["stage_sum"]
        done = int(z["done"])
        print(f"续跑：已有 {done}/{n_paths} 路径", flush=True)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=d,
                     log=lambda *a, **k: None)
    try:
        cuts, meta = load_cuts(d, T, prob.state_dim, expected_mode="multi")
        eng.check_cut_meta(meta)
        eng.cuts = cuts
        lb = eng.lower_bound()
        print(f"LB={lb:.6f}  K={eng.n_theta}", flush=True)
        blk = done // chunk
        t0 = time.time()
        while done < n_paths:
            n = min(chunk, n_paths - done)
            rng = np.random.default_rng(int(scfg.seed) + blk)
            sps = [rng.integers(0, eng.n_scenes, size=T).tolist() for _ in range(n)]
            res = eng.forward_paths(sps)
            for _, costs, total, _, _ in res:
                totals.append(float(total))
                stage_sum += np.asarray(costs, dtype=float)
            done += n
            blk += 1
            _save(partial, totals, stage_sum, done)
            print(f"  完成 {done}/{n_paths}  用时 {time.time()-t0:.0f} s", flush=True)
        arr = np.asarray(totals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se = std / np.sqrt(len(arr))
        ci = 1.959963984540054 * se
        gap = (mean - lb) / abs(mean) * 100
        np.savez_compressed(d / f"sim_{n_paths}paths_resume.npz",
                            totals=arr, stage_sum=stage_sum, done=done, lb=lb)
        js = {"dir": d.name, "n_paths": int(done), "chunk_size": chunk,
              "seed": int(scfg.seed), "lb": lb, "mean": mean, "std": std,
              "ci95": ci, "gap_pct": gap, "n_processes": int(nproc),
              "note": "chunk-resume simulation (phase A); not paired npz"}
        (d / f"sim_{n_paths}paths_resume.json").write_text(
            json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
        txt = (f"LB={lb:.6f}\nUB={mean:.6f} ± {ci:.6f} (95%, {done} paths)\n"
               f"gap={gap:.4f}%\n")
        (d / f"sim_{n_paths}paths_resume.txt").write_text(txt, encoding="utf-8")
        print(txt, flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 3000,
         int(a[2]) if len(a) > 2 else 1000,
         int(a[3]) if len(a) > 3 else 8)
