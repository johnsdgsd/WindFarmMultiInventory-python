# 割系统性越界的根因验证。
#
# 怀疑链：F = rint(θ·n(x)) 只依赖状态 n，而它是**阶梯函数** ⇒ LP 值函数 Q_t(x;ω) 沿
# 月龄方向不连续、更不凸。割的斜率 .dj 是在**当前 F 数值冻结**下取到的导数，等于
# 「F 冻结版 Q̃」的次梯度；它与真值 Q 的差是 ν·ΔF（ν = ∂Q/∂F），这一项从不进割。
# 于是：F(x) < F(锚点) 处，割预测值可以**高于**真值 Q(x) —— 割越界。
#
#   python test/_cvx_rint.py [结果目录] [场景数] [半宽步数]
#
# A 部分：直接测 ν = ∂Q/∂F（把 dem/age 行右端按 F 的定义手工 +1）。ν ≈ 0 ⇒ 假设证伪。
# B 部分：在**真实试探点**（cuts[t] 的锚点）上沿 n[r0,a0] 扫描，每步重解 LP 得真值，
#         与「期望割」的直线外推对比。三种口径对照：
#           rint（生产） / ceil / 连续 θn（仿射 ⇒ Q 必然凸，割必然合法）
#         只有非连续口径出现 Q(k) < c(k)，才把根因锁死在取整上。
#
# 只读现有 artifacts，**不改任何生产代码**（rint 保持原样）。
import json
import sys
from pathlib import Path

import numpy as np
from coptpy import COPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config, load_solver_config   # noqa: E402
from code.problem_data import DefineProblem                       # noqa: E402
from code.subproblem import StageSubproblem                       # noqa: E402
from utils.io import load_cuts                                    # noqa: E402

DEFAULT_DIR = "results/base_20261010_151432"
STAGES = [199, 100, 40, 2]               # B 部分逐阶段检查（由浅到深）


def set_F(sub, F):
    """按给定的 F 矩阵直接写 dem/age 行右端（绕开 set_scene 的 rint，仅供诊断）。"""
    Ly, prob, m = sub.layout, sub.prob_data, sub.model
    m.setInfo(COPT.Info.UB, sub.dem_rows,
              [float(prob.N[r]) + float(F[r].sum()) for r in range(Ly.N_R)])
    rhs = [float(t) for t in (-F[:, :Ly.A - 1]).ravel()]
    m.setInfo(COPT.Info.LB, sub.age_rows, rhs)
    m.setInfo(COPT.Info.UB, sub.age_rows, rhs)


def round_F(prods, mode):
    if mode == "rint":
        return np.rint(prods)
    if mode == "ceil":
        return np.ceil(prods)
    if mode == "floor":
        return np.floor(prods)
    return prods                                    # "cont"：连续 θn


def pick_anchor(cuts_t, Ly):
    """挑一个「机队还活着」的试探点：Σn 最大的锚点。

    取 `it>0` 的第一条会拿到全灭状态（n≡0 ⇒ F≡0，且 F+1 就把 age 行顶到 n_out<0），
    那样既测不出 F 通道、又会假不可行。
    """
    best, bn = None, -1.0
    for c in cuts_t:
        if c.it == 0:
            continue
        s = float(Ly.n(c.state).sum())
        if s > bn:
            best, bn = c, s
    return best


def eval_point(sub, x, thetas, mode):
    """返回 (Q, g, F)：逐场景解 LP 平均（与后向任务同口径）。"""
    objs, grads, F = [], [], None
    for th in thetas:
        sub.set_state(x)
        F = round_F(th * sub.layout.n(x), mode)
        set_F(sub, F)
        sub.solve_lp()
        objs.append(sub.obj)
        grads.append(sub.gradient("dj"))
    return float(np.mean(objs)), np.mean(grads, axis=0), F


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / DEFAULT_DIR
    if not d.is_absolute():
        d = ROOT / d
    n_sc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    half = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    pcfg = load_problem_config(d / "config" / "problem.toml")
    scfg = load_solver_config(d / "config" / "solver.toml")
    prob = DefineProblem(pcfg)
    Ly = prob.layout
    x0 = prob.initial_state()
    cuts, meta = load_cuts(d, prob.T, prob.state_dim)
    print(f"结果目录 {d.name}  轮次 {meta.get('n_iterations')}  T={prob.T}  "
          f"割数 {sum(len(c) for c in cuts)}")
    print(f"默认参数 m={pcfg.m} eta={pcfg.eta} N={prob.N.tolist()} "
          f"Cs={prob.Cs} C={prob.C} h={prob.h} atc={prob.atc}\n")

    rng = np.random.default_rng(7)
    si = sorted(rng.choice(len(prob.pool), size=n_sc, replace=False).tolist())
    thetas = [prob.pool[i][1] for i in si]
    print(f"诊断场景 {si}")

    # ================================================================ A 部分
    print("\n" + "=" * 78)
    print("A：ν = ∂Q/∂F —— 把 F[r,a] 手工 +1 重解 LP（这是割里缺的那一项）")
    print("=" * 78)
    for t in (prob.T, 100):
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        sub = StageSubproblem(t, prob, scfg)
        sub.assemble(cn, key=f"nu{t}")
        c_ = pick_anchor(cuts[t], Ly)
        if c_ is None:
            sub.free()
            continue
        xb = np.asarray(c_.state, dtype=float)
        th = thetas[0]
        sub.set_state(xb)
        nb = Ly.n(xb)
        F0 = round_F(th * nb, "rint")
        set_F(sub, F0)
        sub.solve_lp()
        Q0 = sub.obj
        grads = sub.gradient("dj")
        print(f"\n  阶段 {t}（锚点 it={c_.it}）  Q = {Q0:,.4f}   "
              f"Σn = {nb.sum():,.0f}   ΣF = {F0.sum():,.0f}   ΣI = {Ly.I(xb).sum():,.0f}")
        # 挑 F 最大的几个桶做 +1 扰动（跳过 F+1 > n 的桶：会把 n_out 顶成负、假不可行）
        cand = [k for k in np.argsort(-F0.ravel())[:12]
                if F0.ravel()[k] + 1.0 <= nb.ravel()[k] + 1e-9]
        for k in cand[:4]:
            r, a = np.unravel_index(k, F0.shape)
            Fb = F0.copy()
            Fb[r, a] += 1.0
            sub.set_state(xb)
            set_F(sub, Fb)
            try:
                sub.solve_lp()
            except Exception as exc:
                print(f"    F[{r},{a}]={F0[r,a]:>5.0f}  +1 ⇒ 不可行（{type(exc).__name__}）")
                continue
            nu = sub.obj - Q0
            # 该桶在 n 方向上的「平均每单位 n 漏掉的斜率」= ν·θ
            print(f"    F[{r},{a}]={F0[r,a]:>5.0f}  n={nb[r,a]:>6.0f}  +1 ⇒ ΔQ = {nu:>14,.2f}"
                  f"   θ·ν = {nu * th[r, a]:>12,.4f}   "
                  f"（该维 .dj = {grads[Ly.ix_n(r, a)]:>12,.4f}）")
        sub.free()

    # ================================================================ B 部分
    print("\n" + "=" * 78)
    print("B：真实试探点上沿 n[r0,a0] 扫描 —— 真值 Q(k) vs 割的直线外推 c(k)")
    print("=" * 78)
    print("  判据：Q(k) − c(k) < 0 ⇒ 割在 LP 值函数上越界（生产口径切出来的是假下界）\n")
    out = {"dir": d.name, "scenes": si, "stages": {}}
    for t in STAGES:
        if t > prob.T:
            continue
        cn = None if t >= prob.T else (
            np.array([c.alpha for c in cuts[t + 1]]),
            np.vstack([c.beta for c in cuts[t + 1]]))
        ncut = 0 if cn is None else len(cn[0])
        sub = StageSubproblem(t, prob, scfg)
        sub.assemble(cn, key=f"cvx{t}")
        c_ = pick_anchor(cuts[t], Ly)
        if c_ is None:
            sub.free()
            continue
        xb = np.asarray(c_.state, dtype=float)
        nb = Ly.n(xb)
        # 选扫描桶：F 贡献最大的那个（dem 通道最活跃），且 n 足够大以容纳扫描摆动
        prod = np.mean([th * nb for th in thetas], axis=0)
        prod = np.where(nb > 0.25 * nb.max(), prod, -np.inf)
        r0, a0 = np.unravel_index(np.argmax(prod), prod.shape)
        th0 = float(np.mean([th[r0, a0] for th in thetas]))
        ia = Ly.ix_n(r0, a0)
        # 半宽：让 F 大致摆动 ±(half/2) 格
        span = max(1.0, (half / 2.0) / max(th0, 1e-9))
        ks = np.unique(np.round(np.linspace(-span, span, 2 * half + 1)).astype(int))
        # 只保留「F 单调、且不会把 age 行顶成 n_out<0」的一段（n 减少方向最安全）
        ks = ks[(xb[ia] + ks >= 0) & (xb[ia] + ks <= Ly.upper(prob)[ia])]
        if ks.size < 3:
            sub.free()
            continue

        print(f"  阶段 {t:>3}  割数 {ncut:>5}  锚点 it={c_.it}  "
              f"Σn={nb.sum():,.0f}  ΣF={np.mean([round_F(th*nb,'rint')[0].sum() for th in thetas]):,.0f}  "
              f"扫描桶 ({r0},{a0})  n={xb[ia]:.1f}  θ̄={th0:.5f}  "
              f"F̄={prod[r0,a0]:.2f}  步长范围 ±{span:.0f}")
        rec = {}
        for mode in ("rint", "ceil", "cont"):
            Q0, g0, _ = eval_point(sub, xb, thetas, mode)
            rows = []
            for k in ks:
                x = xb.copy()
                x[ia] = xb[ia] + k
                try:
                    Q, _, F = eval_point(sub, x, thetas, mode)
                except Exception:
                    break           # n 增大方向会被 dem 行余量截断，到界即止
                c = Q0 + float(g0 @ (x - xb))
                rows.append((int(k), float(F[r0, a0]), Q, c, Q - c))
            worst = min(r[4] for r in rows)
            rec[mode] = {"Q0": Q0, "rows": rows, "worst": worst}
            print(f"    [{mode:>4}] Q(锚点)={Q0:>16,.2f}   "
                  f"min(Q−c) = {worst:>16,.2f}   "
                  f"{'★ 越界' if worst < -1e-6 * max(1.0, abs(Q0)) else '合法'}")
            if mode == "rint":
                print("        k:  " + " ".join(f"{r[0]:>11d}" for r in rows))
                print("        F:  " + " ".join(f"{r[1]:>11.0f}" for r in rows))
                print("     Q−c:  " + " ".join(f"{r[4]:>11,.0f}" for r in rows))
        out["stages"][str(t)] = rec
        print()
        sub.free()

    fp = ROOT / "test" / "_cvx_rint_out.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"存档：{fp}")


if __name__ == "__main__":
    main()
