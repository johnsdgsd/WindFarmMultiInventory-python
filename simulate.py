# simulate.py — **唯一仿真入口**：跑一次蒙特卡洛，出三样东西。
#
#   python simulate.py results/<目录> [路径数] [块大小]
#
#   ① 全局统计：路径数、均值、标准差、95% CI、LB（只读割）、gap、求解次数、耗时；
#   ② 详细数据：results/<目录>/sim_<P>paths.npz，含**每期每个变量的完整数值**
#      （状态向量、全部决策、五项成本、theta、场景），供事后分析决策/成本结构，
#      不必重跑（500 条要 ~8 分钟）；
#   ③ txt 报告：results/<目录>/sim_report_<时间戳>.txt，含全局统计、逐期成本、
#      成本结构、约束审计、关键量分布。
#                                               
# 深入分析（调拨是否按优先级、订货形态、按仓分解）读 ② 的 npz：
#   python analyze_sim.py results/<目录>
#
# **分块**：块种子 = seed + k（与训练/旧脚本同一口径），故结果依赖 chunk_size ——
# 报告里会记下来。分块同时把内存封在「一块」的量级；块越大越吃内存（块内结果要
# 先攒成记录），路径数上万时把块调小。
#
# 已存在同名 .npz 则拒绝覆盖（铁律 1：只新增）。
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config        # noqa: E402
from code.problem_data import DefineProblem                            # noqa: E402
from code.sim_analysis import (CHECKS, TERMS, chain_dev, check_stage,  # noqa: E402
                               cost_terms, out_state)
from code.subproblem import make_stage_subproblem                      # noqa: E402
from utils import SDDPEngine                                           # noqa: E402
from utils.io import has_configs, load_cuts                            # noqa: E402

TOL = 1e-6


class Dump:
    """把逐期逐变量攒成定长数组（float32 对 ≤16.7M 的整数无损）。"""

    def __init__(self, n_paths: int, T: int, Ly, L: int):
        P, NR = n_paths, Ly.N_R
        sh = {"x_in": (P, T, Ly.dim), "x_out": (P, T, Ly.dim),
              "q": (P, T, NR), "taup": (P, T, NR), "taum": (P, T, NR),
              "S_end": (P, T, NR), "y": (P, T, NR, NR)}
        self.a = {k: np.zeros(s, np.float32) for k, s in sh.items()}
        self.a["cost"] = np.zeros((P, T))
        self.a["terms"] = np.zeros((P, T, 5))
        self.a["theta"] = np.zeros((P, T))
        self.a["scene"] = np.zeros((P, T), np.int16)
        self.a["totals"] = np.zeros(P)

    def fill(self, k0: int, results, prob, Ly) -> tuple[float, list]:
        """写入第 k0 起的若干条路径，返回 (五项和 vs stage_cost 最大偏差, 审计累加)。"""
        dev, audit = 0.0, []
        ub = Ly.upper(prob)
        for i, (states, costs, total, recs, _) in enumerate(results):
            k = k0 + i
            for t in range(1, len(states) + 1):
                x = np.asarray(states[t - 1], float)
                r = recs[t - 1]
                xo = out_state(Ly, x, r)
                self.a["x_in"][k, t - 1] = x
                self.a["x_out"][k, t - 1] = xo
                self.a["q"][k, t - 1] = np.asarray(r["q"], float)
                self.a["taup"][k, t - 1] = np.asarray(r["taup"], float)
                self.a["taum"][k, t - 1] = np.asarray(r["taum"], float)
                self.a["S_end"][k, t - 1] = np.asarray(r["S_end"], float)
                self.a["y"][k, t - 1] = np.asarray(r["y"], float)
                tm = cost_terms(prob, Ly, x, r)
                self.a["terms"][k, t - 1] = tm
                self.a["cost"][k, t - 1] = float(r["cost"])
                self.a["theta"][k, t - 1] = float(r.get("theta", 0.0))
                self.a["scene"][k, t - 1] = int(r["scene"])
                dev = max(dev, abs(tm.sum() - float(r["cost"])))
                a = check_stage(prob, Ly, x, xo, r["q"], r["taup"], r["taum"],
                                r["S_end"], r["y"], r["scene"], ub)
                if t < len(states):
                    a["链.状态"] = chain_dev(Ly, x, states[t], r)
                audit.append(a)
            self.a["totals"][k] = float(total)
        return dev, audit

    @staticmethod
    def _ub(Ly, prob):
        return Ly.upper(prob)


def main(dir_arg: str, n_paths: int, chunk_size: int | None):
    d = Path(dir_arg)
    if not d.is_absolute():
        d = ROOT / d
    if not has_configs(d):
        raise SystemExit(f"{d} 里没有 config/problem.toml，不是本版本的结果目录")

    npz = d / f"sim_{n_paths}paths.npz"
    if npz.exists():
        raise SystemExit(f"{npz.name} 已存在 —— 不覆盖（铁律 1）。换路径数或先改名。")

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly, T = prob.layout, prob.T
    chunk = int(chunk_size or scfg.chunk_size)
    print(f"结果目录 {d.name}")
    print(f"仿真 {n_paths} 条路径，块大小 {chunk}，种子 {scfg.seed}，"
          f"T={T} A={prob.A} N_R={prob.N_R}")

    eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=d,
                     log=lambda *a, **k: None)
    dump = Dump(n_paths, T, Ly, prob.L)
    audit: list[dict] = []
    dev = 0.0
    t0 = time.time()
    try:
        cuts, meta_cuts = load_cuts(d, prob.T, prob.state_dim)
        eng.cuts = cuts
        lb = eng.lower_bound()
        done, blk = 0, 0
        while done < n_paths:
            n = min(chunk, n_paths - done)
            rng = np.random.default_rng(scfg.seed + blk)
            sps = [rng.integers(0, eng.n_scenes, size=T).tolist() for _ in range(n)]
            res = eng.forward_paths(sps, record=True)
            dv, au = dump.fill(done, res, prob, Ly)
            dev = max(dev, dv)
            audit.extend(au)
            done += n
            blk += 1
            print(f"  已仿真 {done}/{n_paths}  用时 {time.time()-t0:.0f} s", flush=True)
    finally:
        n_solves = eng.n_solves
        eng.close()
    wall = time.time() - t0

    tot = dump.a["totals"]
    mean = float(tot.mean())
    std = float(tot.std(ddof=1))
    se = std / np.sqrt(len(tot))
    ci = 1.959963984540054 * se
    gap = (mean - lb) / abs(mean) * 100 if mean else float("nan")

    # ---- 约束审计汇总 ----
    bad = {}
    for c in CHECKS:
        m = max((a[c] for a in audit if c in a), default=0.0)
        if m > TOL:
            bad[c] = m
    # ---- 成本结构 ----
    terms = dump.a["terms"]
    tsum = terms.sum(axis=(0, 1))
    stage_mean = dump.a["cost"].mean(axis=0)

    np.savez_compressed(npz, **dump.a)
    meta = {"dir": d.name, "n_paths": n_paths, "chunk_size": chunk, "T": T,
            "A": prob.A, "N_R": prob.N_R, "L": prob.L, "state_dim": Ly.dim,
            "seed": int(scfg.seed), "terms": list(TERMS), "lb": lb,
            "mean": mean, "std": std, "ci95": ci, "gap_pct": gap,
            "wall_s": wall, "n_solves": n_solves,
            "cost_coef": {"C": prob.C, "h": prob.h, "Cs": prob.Cs,
                          "atc": prob.atc, "tc": prob.tc},
            "terms_vs_stage_cost_max_dev": dev,
            "audit_violations": bad, "bytes": int(npz.stat().st_size)}
    (d / f"sim_{n_paths}paths.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 报告
    L_ = []
    A_ = L_.append
    A_("=" * 72)
    A_("蒙特卡洛仿真报告")
    A_("=" * 72)
    A_(f"结果目录      : {d}")
    A_(f"路径数        : {n_paths}    块大小 {chunk}（块种子 = seed + k，"
       f"结果依赖块划分）")
    A_(f"随机种子      : {scfg.seed}")
    A_(f"问题规模      : T={T}  A={prob.A}  N_R={prob.N_R}  L={prob.L}  "
       f"状态维 {Ly.dim}  池 {len(prob.pool)}")
    A_(f"训练轮次      : {meta_cuts.get('n_iterations')}    "
       f"割数 {sum(len(c) for c in cuts)}")
    A_("")
    A_("— 总成本 —")
    A_(f"均值 UB       : {mean:.4f}")
    A_(f"标准差        : {std:.4f}     标准误 {se:.4f}")
    A_(f"95% 置信区间  : [{mean - ci:.4f}, {mean + ci:.4f}]")
    A_(f"最小值/最大值 : {tot.min():.4f} / {tot.max():.4f}")
    A_(f"下界 LB       : {lb:.4f}      （只读割，未再训练）")
    A_(f"间隙 gap      : {gap:.4f} %    "
       f"（UB−LB = {mean-lb:,.1f} = {(mean-lb)/se if se else 0:+.2f} 个标准误）")
    A_(f"求解次数      : {n_solves}     仿真用时 {wall:.1f} s")
    A_("")
    A_("— 成本结构（五项，对应 code/subproblem.py 的目标）—")
    s = tsum.sum()
    for i, k in enumerate(TERMS):
        A_(f"  {k:<8}{tsum[i]:>18,.1f}   {tsum[i]/s:>7.2%}")
    A_(f"  {'合计':<8}{s:>18,.1f}   {1.0:>7.2%}")
    A_(f"  对拍：五项之和 vs stage_cost 最大偏差 {dev:.3e}")
    A_("")
    A_("— 约束审计（逐 (路径,期) 点核对全部约束 + 整性 + 轨迹链）—")
    if not bad:
        A_("  全部通过 ✓   " + " / ".join(CHECKS))
    else:
        A_("  ★ 有违规：")
        for k, v in sorted(bad.items()):
            A_(f"    {k:<12} 最大违约量 {v:,.6g}")
    A_("")
    A_("— 逐期平均成本（第 1..T 期）—")
    A_("  " + " ".join(f"{v:.4g}" for v in stage_mean))
    A_("")
    A_(f"数据文件      : {npz.name}  ({npz.stat().st_size/1e6:.1f} MB)")
    A_(f"元数据        : sim_{n_paths}paths.json")
    A_("深入分析      : python analyze_sim.py " + dir_arg)
    text = "\n".join(L_) + "\n"

    rep = d / f"sim_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    rep.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"报告已写: {rep}")
    return rep


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("用法: python simulate.py <结果目录> [路径数] [块大小]")
    main(args[0], int(args[1]) if len(args) > 1 else 1000,
         int(args[2]) if len(args) > 2 else None)
