# DG-CSER 经验回放冒烟：诊断 → 本地/跨阶段选点 → 重解出割 → 割仍合法。
#   python test/_t_replay.py
#
# 这里不追回放是否一定降 gap（那要生产级长跑），只验收：
#   1. 回放按周期触发，且只在高松弛组阶段产点；
#   2. 跨阶段点修复到目标阶段界内、Σn ≤ N；
#   3. 新割满足 alpha + beta@state == obj，且全部割锚点合法；
#   4. 高松弛阶段跳过 LML1，割数不被筛选吃掉。
#
# ⚠ 建进程池的脚本必须有 __main__ 守卫（Windows spawn）。
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.replay import CrossStageReplay                              # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine                                          # noqa: E402

N_ROUNDS = 4
PERIOD = 2


def main() -> int:
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    base = load_solver_config(ROOT / "config/smoke/solver.toml")
    scfg = dataclasses.replace(
        base, iterations=N_ROUNDS, n_processes=2, n_simulations=20,
        chunk_size=20, tag="replay_test",
        replay_enabled=True, replay_period=PERIOD,
        replay_k_local=2, replay_k_cross=2, replay_max_pool=100,
        replay_groups=[[3, 4, 5], [6, 7]],
        replay_group_labels=[["采购"], ["停机", "被动调拨", "持有"]],
        replay_hybrid_stages=[3, 4, 5, 6, 7])
    prob = DefineProblem(pcfg)
    replay = CrossStageReplay(prob, scfg, log=lambda *a, **k: None)

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None, replay=replay)
    t0 = time.time()
    hist = eng.train(N_ROUNDS, verbose=False)
    elapsed = time.time() - t0
    print(f"训练 {N_ROUNDS} 轮用时 {elapsed:.1f} s，"
          f"割数 {sum(len(c) for c in eng.cuts)}，求解 {eng.n_solves} 次")
    for rec in hist:
        if rec.get("replay_cuts"):
            print(f"  轮 {rec['iter']}: 回放点 {rec['replay_points']}"
                  f"（本地 {rec['replay_local']}/跨 {rec['replay_cross']}）"
                  f" → +{rec['replay_cuts']} 割，{rec['replay_time']:.2f} s")

    print()
    print("=" * 74)
    print("§1 触发周期与阶段范围")
    print("=" * 74)
    fired = [rec["iter"] for rec in hist if rec.get("replay_cuts")]
    want = [k for k in range(1, N_ROUNDS + 1) if k % PERIOD == 0]
    print(f"  触发轮次 {fired}，期望 {want}")
    ok_fire = fired == want

    targets = set(replay._targets())
    all_states = []
    for t in range(1, eng.T + 1):
        for c in eng.cuts[t]:
            if c.it > 0:
                all_states.append(c.state)
    print(f"  回放目标阶段 {sorted(targets)}，hybrid {sorted(eng.hybrid_stages)}")
    ok_hybrid = (eng.hybrid_stages == set(scfg.replay_hybrid_stages)
                 and eng.hybrid_start == scfg.replay_hybrid_start
                 and eng.hybrid_keep_rounds == scfg.replay_hybrid_keep_rounds)

    print()
    print("=" * 74)
    print("§2 割自洽与锚点可行")
    print("=" * 74)
    worst = 0.0
    for t in range(1, eng.T + 1):
        for c in eng.cuts[t]:
            worst = max(worst, abs(c.alpha + float(c.beta @ c.state) - c.obj))
    ub = np.asarray(prob.layout.upper(prob), float)
    lb = np.asarray(prob.layout.lower(prob), float)
    N = np.asarray(prob.N, float)
    bound_bad = 0.0
    fleet_bad = 0.0
    for x in all_states:
        x = np.asarray(x, float)
        bound_bad = max(bound_bad, float(np.maximum(lb - x, 0).max()),
                        float(np.maximum(x - ub, 0).max()))
        fleet_bad = max(fleet_bad,
                        float(np.maximum(prob.layout.n(x).sum(axis=1) - N, 0).max()))
    print(f"  割截距最差偏差 {worst:.3e}；锚点越界 {bound_bad:.3e}；"
          f"Σn−N 最大 {fleet_bad:.3e}")
    ok_cut = worst < 1e-6 and bound_bad < 1e-9 and fleet_bad < 1e-6

    print()
    print("=" * 74)
    print("§3 回放选点规模 / 覆盖阈值 / 下界")
    print("=" * 74)
    ok_size = all(rec["replay_local"] <= len(targets) * scfg.replay_k_local
                  and rec["replay_cross"] <= len(targets) * scfg.replay_k_cross
                  for rec in hist if rec.get("replay_cuts"))
    for rec in hist:
        if rec.get("replay_cuts"):
            print(f"  轮 {rec['iter']}: 候选 {rec['replay_cand']}，"
                  f"接受 {rec['replay_accept']}，τ={rec['replay_tau']}")
    print(f"  LB = {eng.last_lb:.4f}（历史最大 {eng.lb_best:.4f}）")
    ok_lb = eng.last_lb > 0
    hybrid_targets = sorted(targets & set(replay.hybrid_stages))
    ok_count = all(len(eng.cuts[t]) > N_ROUNDS * scfg.n_forward_paths
                   for t in hybrid_targets)
    print(f"  hybrid 阶段割数 {[len(eng.cuts[t]) for t in hybrid_targets]}")
    print(f"  每轮前向割 {N_ROUNDS * scfg.n_forward_paths}；"
          f"高松弛阶段应多于它: {ok_count}")
    eng.close()

    ok = {
        "回放按周期触发": ok_fire,
        "hybrid 阶段集合生效": ok_hybrid,
        "新割截距自洽 / 锚点合法": ok_cut,
        "回放批大小受限": ok_size,
        "下界为正": ok_lb,
        "高松弛阶段割未被 LML1 筛掉": ok_count,
    }
    print()
    for k, v in ok.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
