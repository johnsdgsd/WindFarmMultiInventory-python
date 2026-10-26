# BL-SDDP smoke: per-stage trial memory -> batch replay -> current cuts.
# 两种模式都验：after = 标准后向 + 历史回放；unified = 当前+历史一次后向。
#   python test/_t_bl.py
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.bl_sddp import BatchLearning                              # noqa: E402
from code.config import load_problem_config, load_solver_config    # noqa: E402
from code.problem_data import DefineProblem                        # noqa: E402
from code.subproblem import make_stage_subproblem                  # noqa: E402
from utils import SDDPEngine                                       # noqa: E402


def run(mode):
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        iterations=4, n_processes=2,
        bl_enabled=True, bl_mode=mode, bl_period=2, bl_batch=3,
        bl_selection="tight", bl_memory=50)
    prob = DefineProblem(pcfg)
    bl = BatchLearning(prob, scfg, log=lambda *a, **k: None)
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None, bl=bl)
    try:
        hist = eng.train(4, verbose=False)
        fired = [r["iter"] for r in hist if r.get("bl_cuts")]
        mem = [len(x) for x in bl.M[1:]]
        stats = dict(bl.last_stats)
        worst = max(abs(c.alpha + float(c.beta @ c.state) - c.obj)
                    for t in range(1, eng.T + 1) for c in eng.cuts[t])
        lb = eng.lower_bound()
        cuts = [len(eng.cuts[t]) for t in range(1, eng.T + 1)]
    finally:
        eng.close()
    print(f"[{mode}] fired={fired} mem={mem} stats={stats}")
    print(f"[{mode}] cut self-consistency worst={worst:.3e}  LB={lb:.4f}")
    print(f"[{mode}] cuts per stage {cuts}")
    return fired == [2, 4] and worst < 1e-6 and lb > 0


def main():
    ok = all(run(m) for m in ("after", "unified"))
    print(f"[{'PASS' if ok else 'FAIL'}] BL replay fired, cuts consistent, LB>0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
