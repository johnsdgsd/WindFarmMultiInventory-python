# _t_partition_engine.py — 自适应分区端到端 smoke：细化、持久化、resume。
#   python test/_t_partition_engine.py
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import make_stage_subproblem                 # noqa: E402
from utils.engine import SDDPEngine                               # noqa: E402
from utils.io import load_cuts                                    # noqa: E402
from utils.partition import validate_partition                    # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    scfg.partition.enabled = True
    scfg.partition.adaptive = True
    scfg.partition.warmup_rounds = 0
    scfg.partition.stall_window = 1
    scfg.partition.stall_rtol = 0.0
    scfg.partition.stall_abs = 0.0
    scfg.partition.eps_sig_rel = 0.1
    scfg.partition.eps_obj = 0.0
    scfg.partition.max_clusters = 8
    prob = DefineProblem(pcfg)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            hist = eng.train(5, verbose=False)
            assert len(hist) == 5
            assert eng.n_refines > 0, "细化没有发生"
            assert eng.lower_bound() > 0
            for t in range(1, prob.T + 1):
                validate_partition(eng.partitions[t], len(prob.pool), t)
            assert max(len(p) for p in eng.partitions[1:]) > 1
            eng.checkpoint()
            meta_part = eng.meta().get("partition")
            assert meta_part and meta_part["n_refines"] == eng.n_refines
        finally:
            eng.close()
        cuts, meta = load_cuts(td, prob.T, prob.state_dim)
        eng2 = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                          log=lambda *a, **k: None)
        try:
            eng2.cuts = cuts
            eng2.load_partition_state(meta)
            assert eng2.n_refines == meta["partition"]["n_refines"]
            for t in range(1, prob.T + 1):
                validate_partition(eng2.partitions[t], len(prob.pool), t)
            eng2.train(1, start_iter=hist[-1]["iter"] + 1, verbose=False)
        finally:
            eng2.close()
        assert (td / "partition_log.csv").is_file()
    print("OK _t_partition_engine")


if __name__ == "__main__":
    main()
