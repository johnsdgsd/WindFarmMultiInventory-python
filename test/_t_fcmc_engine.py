# _t_fcmc_engine.py — 固定分区多割：L 扫描、组绑定、LML1 隔离、resume 身份校验。
#   python test/_t_fcmc_engine.py
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (MulticutConfig, load_problem_config,  # noqa: E402
                         load_solver_config)
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import StageSubproblem, make_stage_subproblem  # noqa: E402
from utils.cut import Cut, trivial_cut                           # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


def _cfg(scfg, L, gm="failure_quantile"):
    return dataclasses.replace(
        scfg, mode="multi", multicut=MulticutConfig(n_clusters=L, group_mode=gm))


def _eng(prob, cfg, results_dir=None, log=None):
    return SDDPEngine(prob, make_stage_subproblem, cfg,
                      results_dir=results_dir,
                      log=log or (lambda *a, **k: None))


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    prob = DefineProblem(pcfg)
    T, K, dim = prob.T, len(prob.pool), prob.state_dim

    # single 回归：不受 [multicut] 影响
    e1 = _eng(prob, scfg)
    try:
        e1.train(2, verbose=False)
        assert e1.n_theta == 1 and not e1.grouped
        assert e1.lower_bound() >= 0
    finally:
        e1.close()
    print("  [PASS] single 回归")

    # L 扫描：分组、实际权重、全部阶段 group 完整
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg4 = _cfg(scfg, 4)
        e4 = _eng(prob, cfg4, results_dir=td)
        try:
            e4.train(2, verbose=False)
            assert e4.n_theta == 4 and e4.grouped
            sizes = [len(c) for c in e4.clusters]
            assert sizes == [8, 8, 7, 7], sizes
            assert np.allclose(e4.cluster_weights, np.asarray(sizes) / K)
            assert abs(float(e4.cluster_weights.sum()) - 1.0) < 1e-12
            for t in range(1, T + 1):
                assert {c.group for c in e4.cuts[t]} == set(range(4)), t
            print(f"  [PASS] L=4：簇 {sizes}，每阶段 4 组割")
        finally:
            e4.close()

        # resume：同 L 可续训；换 L / 换 group_mode 硬失败
        cuts, meta = load_cuts(td, T, dim, expected_mode="multi")
        assert meta["multicut"]["n_clusters"] == 4
        assert meta["multicut"]["cluster_sizes"] == [8, 8, 7, 7]
        assert meta["multicut"]["cluster_hash"]
        e4b = _eng(prob, cfg4, results_dir=td)
        try:
            e4b.check_cut_meta(meta)
            e4b.cuts = cuts
            n_hist = e4b.restore_cut_history()
            assert n_hist > 0
            e4b.train(1, start_iter=3, verbose=False)
            assert e4b.lower_bound() >= 0
        finally:
            e4b.close()
        print(f"  [PASS] L=4 resume：重建 {n_hist} 条历史并可续训")

        for cfg2, what in ((_cfg(scfg, 8), "L=8"),
                           (_cfg(scfg, 4, "failure_stride"), "stride")):
            e = _eng(prob, cfg2, results_dir=None)
            try:
                try:
                    e.check_cut_meta(meta)
                except SystemExit:
                    pass
                else:
                    raise AssertionError(f"{what} 与 L=4 分位簇不兼容，应拒绝")
            finally:
                e.close()
        print("  [PASS] 换 L / 换 group_mode 拒绝加载旧割")

        # 旧全场景多割 meta（无 multicut 块）仍可被 full multi 接受
        efull = _eng(prob, dataclasses.replace(scfg, mode="multi"))
        try:
            assert efull.n_theta == K
            efull.check_cut_meta({"mode": "multi", "n_theta": K})
        finally:
            efull.close()
        print("  [PASS] 旧全场景多割 meta 兼容")

    # 组绑定与权重：只有 group1 有强割 ⇒ θ1=100，其他 θ=0
    sp = StageSubproblem(1, prob, cfg4)
    try:
        x0 = np.asarray(prob.initial_state(), float)
        alpha = np.array([0.0, 100.0, 0.0, 0.0])
        beta = np.zeros((4, dim))
        groups = np.arange(4)
        sp.assemble((alpha, beta, groups), key="bind")
        sp.set_state(x0)
        sp.set_scene(prob.pool[0])
        sp.solve_lp()
        w = np.asarray([8, 8, 7, 7]) / 30.0
        assert abs(sp.theta_value() - 100.0 * w[1]) < 1e-7, sp.theta_value()
        assert abs(sp.obj - sp.stage_cost() - sp.theta_value()) < 1e-7
        for bad in ((alpha[:3], beta[:3], groups[:3]),
                    (alpha, beta, np.array([0, 1, 2, 9]))):
            try:
                sp.assemble(bad, key="bad")
            except ValueError:
                pass
            else:
                raise AssertionError("缺组/越界 group 应报错")
            sp.assemble((alpha, beta, groups), key="bind")
    finally:
        sp.free()
    print("  [PASS] 组绑定/权重：theta_value=Σp_lθ_l，缺组/越界拦截")

    # LML1 隔离：group0 被更强的新割替换，group1 的历史割保留
    e = _eng(prob, _cfg(scfg, 2), results_dir=None)
    try:
        t = 1
        triv = [trivial_cut(t, e.initial_state, dim, group=g) for g in range(2)]
        e.cuts[t] = list(triv)
        e._pts_g[t] = [[], []]
        e._mx_g[t] = [[], []]
        e._rc_g[t] = [[], []]
        x0 = np.zeros(dim)
        x1 = np.ones(dim)
        c0a = Cut(10.0, np.zeros(dim), x0, t, 10.0, 1, group=0)
        c1a = Cut(5.0, np.zeros(dim), x0, t, 5.0, 1, group=1)
        e.cuts[t] += [c0a, c1a]          # 生产路径：新割先入池，再交给 LML1
        e._lml1_update_multi(t, [x0], [c0a, c1a])
        assert c0a in e._rc_g[t][0] and c1a in e._rc_g[t][1]
        c0b = Cut(12.0, np.zeros(dim), x1, t, 12.0, 2, group=0)
        c1b = Cut(4.0, np.zeros(dim), x1, t, 4.0, 2, group=1)
        e.cuts[t] += [c0b, c1b]
        e._lml1_update_multi(t, [x1], [c0b, c1b])
        assert c0b in e._rc_g[t][0] and c0a not in e._rc_g[t][0]
        assert c1a in e._rc_g[t][1] and c1b in e._rc_g[t][1]
        alive = {id(c) for c in e.cuts[t]}
        assert {id(c0b), id(c1a), id(c1b)} <= alive and id(c0a) not in alive
        assert all(id(c) in alive for c in triv)
    finally:
        e.close()
    print("  [PASS] LML1 按 (阶段, 簇) 隔离，平凡割保留")

    print("OK _t_fcmc_engine")


if __name__ == "__main__":
    main()
