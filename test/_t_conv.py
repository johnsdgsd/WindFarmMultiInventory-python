# 收敛观察：冒烟参数下分块多轮训练，逐块打印 LB / UB / gap 走向。
#   python test/_t_conv.py
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.problem_data import DefineProblem          # noqa: E402
from code.model import build_stage_subproblem        # noqa: E402
from utils import SDDPEngine, simulate               # noqa: E402

if __name__ == "__main__":
    prob = DefineProblem(T=4, A=24, N_R=3, pool_size=30, spread_init=True,
                         m=3.0, eta=8.0)
    eng = SDDPEngine(prob, build_stage_subproblem, T=prob.T,
                     state_dim=prob.state_dim, initial_state=prob.initial_state(),
                     n_processes=6, seed=2026, log=lambda *a: None)
    t0 = time.time()
    for blk in range(12):
        eng.train(20, n_forward_paths=3)
        lb = eng.lower_bound()
        st = simulate(eng, n_paths=200, seed=2026, chunk_size=100)
        print(f"轮 {20 * (blk + 1):4d} | LB {lb:14.2f} | UB {st.mean:14.2f} "
              f"± {st.ci95:10.2f} | gap {(st.mean - lb) / st.mean * 100:5.2f}% "
              f"| 割 {sum(len(c) for c in eng.cuts):5d} | {time.time() - t0:6.1f}s",
              flush=True)
    eng.close()
