# 割合法性验收。
#
# 判据要分清两件事，否则会把抽样噪声误判成越界：
#
#   ① 【算法性质】逐场景的 Q_t(·;ω) 是否凸 —— 凸则「场景均值的割」必是「场景均值的
#      值函数」的下界。这条是割合法性的**充要条件**，必须零越界。
#      对拍方式与后向任务完全一致：同一条场景子集 S 既造割又算真值
#         c_S(x) = mean_j Q(x̄;ω_j) + (mean_j g_j)·(x − x̄)      ← 与 worker 同构
#         slack  = mean_j Q(x;ω_j) − c_S(x)  ≥ 0 ？
#
#   ② 【统计误差】存盘的割用的是训练那一批场景，与任何别的样本都不重合。
#      拿全池均值去比，差值里混着抽样噪声，**不是**越界。单列出来只作量级参考。
#
#   python test/_t_cutvalid.py [结果目录] [检查期数] [每期取几个锚点] [场景子集大小]
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils import copt_env as ce                                  # noqa: E402
from utils.io import load_cuts                                    # noqa: E402


def newest_dir(root: Path) -> Path:
    ds = [p for p in root.iterdir() if p.is_dir() and (p / "cuts.json").is_file()]
    return max(ds, key=lambda p: p.stat().st_mtime)


def main():
    root = ROOT / "results"
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_dir(root)
    if not d.is_absolute():
        d = ROOT / d
    n_stage = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_anchor = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n_sc = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    ub = Ly.upper(prob)
    print(f"结果目录 {d.name}   轮次 {meta.get('n_iterations')}   T={prob.T}   "
          f"割数 {sum(len(c) for c in cuts)}")
    print(f"参数 m={pcfg.m} eta={pcfg.eta} A={pcfg.A} N_R={prob.N_R} "
          f"spread_init={pcfg.spread_init}   每场景子集 {n_sc} 条\n")

    rng = np.random.default_rng(11)
    si = sorted(rng.choice(len(prob.pool), size=n_sc, replace=False).tolist())
    print(f"对拍场景子集 S = {si}\n")

    ts = [t for t in range(1, prob.T + 1) if len(cuts[t]) > 1]
    if not ts:
        raise SystemExit("该目录没有任何有效割")
    pick = ts if len(ts) <= n_stage else \
        [ts[int(round(v))] for v in np.linspace(0, len(ts) - 1, n_stage)]

    worst_alg, worst_stat = np.inf, -np.inf
    n_checked, n_bad = 0, 0
    for t in pick:
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        sub = StageSubproblem(t, prob, scfg)
        sub.assemble(cn, key=f"cv{t}")

        def vec_at(x):
            """逐场景解 LP，返回 (Q_j, g_j) —— 与后向任务同一口径。

            require_int=False：割必须在 LP 松弛走过的**小数**点上也不越界
            （松弛里 x_out 是连续变量），所以诊断要能扫小数点。
            """
            qs, gs = [], []
            for j in si:
                sub.set_point(x, prob.pool[j], require_int=False)
                sub.solve_lp()
                qs.append(sub.obj)
                gs.append(sub.gradient("dj"))
            return np.array(qs), np.array(gs)

        real = [c for c in cuts[t] if c.it > 0]
        idx = rng.choice(len(real), size=min(n_anchor, len(real)), replace=False)
        worst_t, stat_t, qmax_t = np.inf, -np.inf, 1.0
        fdev = None
        for i in idx:
            c = real[int(i)]
            xb = np.asarray(c.state, dtype=float)
            q0, g0 = vec_at(xb)
            bar = q0.mean()
            gbar = g0.mean(axis=0)
            if fdev is None:
                # 整数 F 的两式定义在 LP 松弛下落在哪：F ∈ [θn−1, θn]，应贴住上界 θn。
                # 必须用**同一个场景**重解后再读 F，否则比的是不同 θ 的 F。
                j0 = si[0]
                sub.set_point(xb, prob.pool[j0], require_int=False)
                sub.solve_lp()
                Fv = np.array([[ce.values(sub.model, sub.F[r][a]) for a in range(Ly.A)]
                               for r in range(Ly.N_R)])
                tgt = np.asarray(prob.pool[j0][1], dtype=float) * Ly.n(xb)
                fdev = float(np.abs(Fv - tgt).max())
            # 方向：θ 只经月龄对状态有依赖，故以月龄方向为主，另加两个随机方向
            nb = Ly.n(xb)
            prod = np.mean([prob.pool[j][1] * nb for j in si], axis=0)
            prod = np.where(nb > 0.25 * nb.max(), prod, -np.inf)
            r0, a0 = np.unravel_index(np.argmax(prod), prod.shape)
            th0 = float(np.mean([prob.pool[j][1][r0, a0] for j in si]))
            dirs = []
            e = np.zeros(Ly.dim); e[Ly.ix_n(r0, a0)] = 1.0; dirs.append(e)
            for _ in range(2):
                dd = np.zeros(Ly.dim)
                js = rng.choice(Ly.dim, size=5, replace=False)
                dd[js] = rng.choice([-1.0, 1.0], size=5)
                dirs.append(dd)
            span = max(2.0, 2.0 / max(th0, 1e-9))
            for dv in dirs:
                for k in np.linspace(-span, span, 9):
                    x = xb + k * dv
                    if np.abs(x - np.clip(x, 0.0, ub)).max() > 1e-6:
                        continue
                    try:
                        qs, _ = vec_at(x)
                    except Exception:
                        continue
                    qm = qs.mean()
                    qmax_t = max(qmax_t, abs(qm))
                    n_checked += 1
                    s_alg = qm - (bar + float(gbar @ (x - xb)))     # ① 算法性质
                    s_sta = qm - c.value_at(x)                      # ② 存盘割 vs 全子集
                    if s_alg < -1e-6 * qmax_t:
                        n_bad += 1
                    worst_t = min(worst_t, s_alg)
                    stat_t = max(stat_t, -s_sta)
            worst_alg = min(worst_alg, worst_t)
            worst_stat = max(worst_stat, stat_t)
        # 阈值按量级取：LP 容差 1e-6 作用在 Q 的数据量级 qmax_t 上，与总量判定同口径
        tol_t = 1e-6 * qmax_t
        print(f"  期 {t:>3}  割 {len(real):>4} 条  取 {len(idx)} 个锚点  "
              f"|Q|~{qmax_t:,.0f}  ①同场景 min(Q−c) = {worst_t:>14,.4f}   "
              f"{'★ 越界' if worst_t < -tol_t else '合法'}   "
              f"②存盘割与子集均值最大差 {stat_t:>12,.1f}   "
              f"[松弛后 |F−θn|≤{fdev:.3f}]")
        sub.free()

    print("\n" + "=" * 78)
    print(f"① 算法性质：共查 {n_checked} 个 (割,点)，越界 {n_bad} 个，"
          f"全体最小余量 = {worst_alg:,.4f}")
    print(f"   判定：" + ("全部合法 ✓ —— 逐场景值函数凸，割是合法下界"
                        if n_bad == 0 else f"★ {n_bad} 处越界 ✗ —— 仍有非凸源"))
    print(f"② 统计误差参考（存盘割用的训练场景 ≠ 这里的子集，差值含抽样噪声，"
          f"不是越界）：最大 {worst_stat:,.1f}")


if __name__ == "__main__":
    main()
