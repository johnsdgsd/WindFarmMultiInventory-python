# _t_multicut_unit.py — single/multi theta 与 Cut.group 的单测（不跑引擎）。
#   python test/_t_multicut_unit.py
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils.cut import Cut                                         # noqa: E402


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    prob = DefineProblem(pcfg)
    x = np.round(np.asarray(prob.initial_state(), float), 6)
    K = len(prob.pool)

    s1 = StageSubproblem(1, prob, dataclasses.replace(scfg, mode="single"))
    assert s1.n_theta == 1
    assert np.allclose(s1.theta_weights, [1.0])

    sm = StageSubproblem(1, prob, dataclasses.replace(scfg, mode="multi"))
    assert sm.n_theta == K
    assert abs(float(sm.theta_weights.sum()) - 1.0) < 1e-12

    # Cut.group 序列化
    c = Cut(1.5, np.zeros(prob.state_dim), x, 1, 2.5, 7, group=3)
    d = c.to_dict()
    c2 = Cut.from_dict(d)
    assert c2.group == 3 and abs(c2.alpha - 1.5) < 1e-12

    # 多割 group 绑定：每个 theta 只受自己的 alpha 约束
    cuts = [Cut(float(10 + g), np.zeros(prob.state_dim), x, 1, 0.0, 1, group=g)
            for g in range(K)]
    sm.set_state(x)
    sm.set_scene(prob.pool[0])
    sm.assemble((np.array([c.alpha for c in cuts]),
                 np.vstack([c.beta for c in cuts]),
                 np.array([c.group for c in cuts], float)), key="unit")
    sm.solve_lp()
    # theta_value = sum p_g * alpha_g
    expect = float(np.dot(sm.theta_weights, np.arange(10, 10 + K)))
    assert abs(sm.theta_value() - expect) < 1e-7, (sm.theta_value(), expect)
    assert abs(sm.stage_cost() - (sm.obj - expect)) < 1e-7

    # 只改 group 3 的割，其他 group 的 theta 值不变
    cuts2 = list(cuts)
    cuts2[3] = Cut(1000.0, np.zeros(prob.state_dim), x, 1, 0.0, 2, group=3)
    sm.assemble((np.array([c.alpha for c in cuts2]),
                 np.vstack([c.beta for c in cuts2]),
                 np.array([c.group for c in cuts2], float)), key="unit2")
    sm.solve_lp()
    expect2 = float(np.dot(sm.theta_weights,
                           np.array([10 + g for g in range(K)])
                           + np.eye(K)[3] * (1000.0 - 13.0)))
    assert abs(sm.theta_value() - expect2) < 1e-7

    # 缺 group 必须拒绝
    bad = [c for c in cuts if c.group != 5]
    try:
        sm.assemble((np.array([c.alpha for c in bad]),
                     np.vstack([c.beta for c in bad]),
                     np.array([c.group for c in bad], float)), key="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("missing group 未被拒绝")

    print(f"OK _t_multicut_unit  K={K}")


if __name__ == "__main__":
    main()
