# resume 的 LML1 历史恢复：从 cuts 的 state/obj/it 重建 _pts/_mx/_rc。
# ① 弱新割不能触发第一轮“无主割整体清空”；
# ② 强新割仍按 LML1 删除旧割——LML1 的数学逻辑不变。
#   python test/_t_resume.py
#
# ⚠ Windows spawn：建进程池的脚本必须有 __main__ 守卫。
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config       # noqa: E402
from code.problem_data import DefineProblem                           # noqa: E402
from code.subproblem import make_stage_subproblem                     # noqa: E402
from utils import SDDPEngine                                          # noqa: E402
from utils.cut import Cut                                             # noqa: E402


def main() -> None:
    pcfg = load_problem_config(ROOT / "config/smoke/problem.toml")
    scfg = dataclasses.replace(
        load_solver_config(ROOT / "config/smoke/solver.toml"),
        iterations=2, n_processes=2, n_simulations=10, chunk_size=10, tag="resume")
    prob = DefineProblem(pcfg)

    # 先正常训 2 轮，拿到一批割
    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                     log=lambda *a, **k: None)
    eng.train(2, verbose=False)
    old = [list(c) for c in eng.cuts]
    eng.close()

    # 模拟 resume：新引擎只拿到 cuts，_pts/_mx/_rc 是空的
    eng2 = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                      log=lambda *a, **k: None)
    eng2.cuts = [list(c) for c in old]
    n_hist = eng2.restore_cut_history()
    n_old = sum(1 for t in range(1, eng2.T + 1)
                for c in eng2.cuts[t] if c.it > 0)
    assert n_hist == n_old, (n_hist, n_old)

    old_stage1 = [c for c in old[1] if c.it > 0]
    assert old_stage1, "smoke 两轮后 stage 1 没有非平凡割"
    old_ids = {id(c) for c in old_stage1}
    x_new = eng2.initial_state

    # ① 弱的、锚在新点上的新割：不能触发“无主割清空”
    low = Cut(0.0, np.zeros(eng2.state_dim), x_new, 1, 0.0, 999)
    eng2.cuts[1].append(low)
    eng2._lml1_update(1, [x_new], [low])
    now_ids = {id(c) for c in eng2.cuts[1]}
    survived = len(old_ids & now_ids)
    print(f"restore: {n_hist} 条历史记录；弱新割后 stage 1 旧割 "
          f"{len(old_ids)} 条存活 {survived} 条")
    assert old_ids <= now_ids, (len(old_ids), survived)

    # ② 强割（在任意点取值 1e9）支配所有旧割：旧割必须按 LML1 被删除，
    #    证明加载的割仍然受 LML1 筛选，而不是被特殊保护。
    strong = Cut(1e9, np.zeros(eng2.state_dim), x_new, 1, 1e9, 1000)
    eng2.cuts[1].append(strong)
    eng2._lml1_update(1, [x_new], [strong])
    now_ids = {id(c) for c in eng2.cuts[1]}
    left = len(old_ids & now_ids)
    print(f"强新割后 stage 1 旧割剩余 {left} 条（应为 0，LML1 仍筛选旧割）")
    assert not (old_ids & now_ids), (len(old_ids), left)
    eng2.close()
    print("OK _t_resume")


if __name__ == "__main__":
    main()
