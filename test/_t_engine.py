# 引擎端到端冒烟：训练 4 轮 + 前向记录 + 仿真，顺带验证割自洽（截距式定义）。
#   python test/_t_engine.py
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.problem_data import DefineProblem          # noqa: E402
from code.model import build_stage_subproblem        # noqa: E402
from utils import SDDPEngine, simulate               # noqa: E402

prob = DefineProblem(T=3, A=12, N_R=2, pool_size=8, spread_init=True, m=3.0, eta=8.0)
eng = SDDPEngine(prob, build_stage_subproblem, T=prob.T, state_dim=prob.layout.dim,
                 initial_state=prob.initial_state(), n_processes=1, seed=7, log=print)
print("引擎就绪：场景", eng.n_scenes, "状态维", eng.state_dim, "进程", eng.n_processes)

t0 = time.time()
hist = eng.train(4, n_forward_paths=1)
print(f"训练用时 {time.time() - t0:.1f}s")
for h in hist:
    print("  ", h)

trial, res = eng.forward(n_paths=2, record=True)
print("前向路径总成本:", [round(r[2], 2) for r in res])
print("阶段 1 试态点个数:", len(trial[1]), " 状态前 6 维:",
      np.round(trial[1][0][:6], 3))

st = simulate(eng, n_paths=20, seed=99, chunk_size=5, verbose=False)
print(f"仿真 20 条：均值 {st.mean:.3f} 标准差 {st.std:.3f} ±{st.ci95:.3f}")
print("逐期均值:", np.round(st.stage_mean(), 2))
print("割数:", {t: len(c) for t, c in enumerate(eng.cuts) if c})

# 割自洽：alpha + beta @ state == obj（截距式的定义）
ok = all(abs(c.alpha + c.beta @ c.state - c.obj) < 1e-6 for cs in eng.cuts for c in cs)
print("割自洽（alpha + beta@state == obj）:", ok)
eng.close()
