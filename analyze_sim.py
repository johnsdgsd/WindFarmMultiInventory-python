# analyze_sim.py — 读 simulate.py 落盘的 .npz 做**事后分析**，不重跑仿真。
#
#   python analyze_sim.py results/<目录> [sim_Ppaths.npz]
#
# 输出五节（口径定义在 code/sim_analysis.py，与 simulate.py 落盘时完全一致）：
#   ① 成本结构：五项总量/占比、最贵的期、视界分段、按仓库
#   ② 主动调拨 τ：激活程度、τ⁺/τ⁻ 并存（已知瑕疵）、按仓净流
#   ③ 被动调拨 y：本地/跨仓、按优先级路线的单调性、互补松弛
#   ④ 订货 q：贴限情况、base-stock 形态、对失效的超前性
#   ⑤ 逐期轨迹：库存/停机/月龄/在途
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402
from code.sim_analysis import CHECKS, check_stage      # noqa: E402
from code.state_layout import StateLayout              # noqa: E402
from utils.io import load_json                         # noqa: E402

SEG = 5
TOL = 1e-6
AUDIT_PATHS = 40        # 约束审计抽查的路径数上限（逐点 Python 循环，全量太慢）


def head(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python analyze_sim.py <结果目录> [sim_Ppaths.npz]")
    d = Path(sys.argv[1])
    if not d.is_absolute():
        d = ROOT / d
    if len(sys.argv) > 2:
        f = Path(sys.argv[2])
        f = f if f.is_absolute() else d / f
    else:
        c = sorted(d.glob("sim_*paths.npz"))
        if not c:
            raise SystemExit(f"{d} 里没有 sim_*paths.npz —— 先跑 "
                             f"python simulate.py {sys.argv[1]} <路径数>")
        f = c[-1]

    meta = load_json(d, f.stem + ".json")
    prob = DefineProblem(load_problem_config(d / "config" / "problem.toml"))
    Ly, NR, A, T = prob.layout, prob.N_R, prob.A, prob.T
    P = int(meta["n_paths"])
    z = np.load(f)
    x_in = z["x_in"].astype(float)
    x_out = z["x_out"].astype(float)
    q, taup, taum = z["q"].astype(float), z["taup"].astype(float), z["taum"].astype(float)
    y, terms = z["y"].astype(float), z["terms"]
    scene = z["scene"].astype(int)
    n_in = x_in[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A)
    I_in, I_out = x_in[:, :, :Ly.n_I], x_out[:, :, :Ly.n_I]
    o_in = x_in[:, :, Ly.off_o:Ly.off_o + Ly.n_o].reshape(P, T, prob.L - 1, NR)
    o_out = x_out[:, :, Ly.off_o:Ly.off_o + Ly.n_o].reshape(P, T, prob.L - 1, NR)
    F = np.zeros((P, T, NR))
    for t in range(T):
        th = np.array([prob.pool[scene[p, t]][1] for p in range(P)])   # (P,NR,A)
        F[:, t, :] = np.floor(th * n_in[:, t]).sum(axis=2)

    print(f"{f.name}   路径 {P} × {T} 期   种子 {meta['seed']} "
          f"块 {meta.get('chunk_size', '—')}")
    if "mean" in meta:
        print(f"UB {meta['mean']:,.1f} ± {meta['ci95']:,.1f}   LB {meta['lb']:,.1f}   "
              f"gap {meta['gap_pct']:.3f}%")
    else:
        print(f"（这份 .npz 由旧脚本落盘，无全局统计元数据；"
              f"总成本均值 {z['totals'].mean():,.1f}）")
    print(f"系数 C={prob.C} h={prob.h} Cs={prob.Cs} atc={prob.atc} tc={prob.tc}")
    print(f"N={prob.N.tolist()}  Qmax={prob.Qmax.tolist()}  Imax={prob.Imax.tolist()}")

    # ================================================================ 约束审计
    step = max(1, P // min(P, AUDIT_PATHS))
    head(f"约束审计（门禁：复算全部约束 + 整性 + 轨迹链）"
         f"[抽查 {len(range(0, P, step))}/{P} 条]")
    ub = Ly.upper(prob)
    S_end = z["S_end"].astype(float)
    bad = {}
    npt = 0
    for p in range(0, P, step):
        for t in range(T):
            x, xo = x_in[p, t], x_out[p, t]
            d = check_stage(prob, Ly, x, xo, q[p, t], taup[p, t], taum[p, t],
                            S_end[p, t], y[p, t], scene[p, t], ub)
            if t + 1 < T:
                d["链.状态"] = float(np.abs(x_in[p, t + 1] - xo).max())
            npt += 1
            for c in CHECKS:
                if c in d:
                    bad[c] = max(bad.get(c, 0.0), float(d[c]))
    viol = {k: v for k, v in bad.items() if v > TOL}
    print(f"  共核对 {npt:,} 个 (路径,期) 点，age/F 类等式 {npt * Ly.n_n:,} 条")
    if not viol:
        print("  全部通过 ✓   " + " / ".join(CHECKS))
    else:
        print("  ★ 有违规（阈值 1e-6）：")
        for k, v in sorted(viol.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<12} 最大违约量 {v:,.6g}")

    # ================================================================ ① 成本结构
    head("① 成本结构")
    tsum = terms.sum(axis=(0, 1))
    s = tsum.sum()
    names = meta["terms"]
    for i, k in enumerate(names):
        print(f"  {k:<8}{tsum[i]:>18,.1f}   {tsum[i]/s:>7.2%}")
    print(f"  {'合计':<8}{s:>18,.1f}   {1.0:>7.2%}")
    per = terms.sum(axis=0)
    tots = per.sum(axis=1)
    print("\n  最贵的 8 期（列为该期内部占比）")
    print("     期      该期成本      采购    持有    停机  主动调拨 被动调拨  占总成本")
    for t in sorted(np.argsort(-tots)[:8]):
        row = per[t]
        print(f"   {t+1:>5} {tots[t]:>13,.0f}   " +
              " ".join(f"{row[i]/max(tots[t],1e-9):>6.1%}" for i in range(5)) +
              f"  {tots[t]/s:>7.2%}")
    print(f"\n  视界 {SEG} 等分")
    print("     区间        段占比     采购    持有    停机  主动调拨 被动调拨")
    seg = T // SEG
    for k in range(SEG):
        a, b = k * seg, (T if k == SEG - 1 else (k + 1) * seg)
        row = per[a:b].sum(axis=0)
        print(f"   {a+1:>4}-{b:<4}  {row.sum()/s:>9.2%}   " +
              " ".join(f"{row[i]/max(row.sum(),1e-9):>6.1%}" for i in range(5)))
    print(f"\n  按仓库（停机 = 台·期）")
    print("     仓        N    停机台·期   停机成本    持有成本    采购成本  "
          "主动调拨成本")
    for r in range(NR):
        dn = (prob.N[r] - n_in[:, :, r].sum(axis=2)).sum()
        hold = prob.h / 2.0 * (I_in[:, :, r] + I_out[:, :, r]).sum()
        order = prob.C * x_in[:, :, Ly.ix_q(r)].sum()
        act = prob.atc / 2.0 * (taup[:, :, r] + taum[:, :, r]).sum()
        print(f"   {r+1:>4} {prob.N[r]:>9,.0f} {dn:>11,.0f} {dn*prob.Cs:>11,.0f} "
              f"{hold:>11,.0f} {order:>11,.0f} {act:>11,.0f}")

    # ================================================================ ② 主动调拨
    head("② 主动调拨 τ = τ⁺ − τ⁻")
    both = (taup > TOL) & (taum > TOL)
    nz = (taup + taum) > TOL
    print(f"  非零 (路径,期,仓) 占比 {nz.mean():.1%}")
    if both.any():
        print(f"  ★ τ⁺/τ⁻ 并存 {int(both.sum())} 次（占 {both.mean():.3%}）—— 成本为正 ⇒ "
              f"最优不应并存；疑似前向 MIP 的 rel_gap 放过的次优解，见 docs/待查问题.md")
        print(f"    并存时 min(τ⁺,τ⁻) 合计 {np.minimum(taup, taum)[both].sum():,.0f} 件")
    else:
        print("  τ⁺/τ⁻ 并存 = 0 ✓（成本为正 ⇒ 最优不并存）")
    print(f"\n     仓      调出(τ⁻)     调入(τ⁺)          净     调出次数  调入次数")
    for r in range(NR):
        so, si = taum[:, :, r].sum(), taup[:, :, r].sum()
        print(f"   {r+1:>4} {so:>12,.0f} {si:>12,.0f} {si-so:>12,.0f}   "
              f"{(taum[:,:,r]>TOL).sum():>8d} {(taup[:,:,r]>TOL).sum():>9d}")
    print(f"  合计 |τ| = {(taup.sum()+taum.sum())/2:,.0f} 件   "
          f"件均成本 {terms[:,:,3].sum()/max((taup.sum()+taum.sum())/2,1e-9):.2f}")

    # ================================================================ ③ 被动调拨
    head("③ 被动调拨 y（优先级凸化：先用便宜的源）")
    yl = np.trace(y, axis1=2, axis2=3).sum()
    yt = y.sum()
    print(f"  总量 {yt:,.0f} 件：本地 {yl:,.0f} ({yl/yt:.1%})，"
          f"跨仓 {yt-yl:,.0f} ({1-yl/yt:.1%})，成本 {terms[:,:,4].sum():,.0f}")
    print(f"\n     仓      本地   跨仓来源（按 tc_route 升序）")
    mono = 0
    for r in range(NR):
        row = y[:, :, r, :].sum(axis=(0, 1))
        order = [m for m in np.argsort(prob.tc_route[r]) if m != r]
        vals = [row[m] for m in order]
        mono += int(all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))
        print(f"   {r+1:>4} {row[r]:>9,.0f}   " +
              "  ".join(f"m{m+1}:{row[m]:>7,.0f}" for m in order))
    print(f"  → {mono}/{NR} 个收货仓的跨仓量随路线费**单调递减**"
          f"（优先级梯度是否生效的判据）")
    out_m = y.sum(axis=2)
    used_tot = viol = 0
    for r in range(NR):
        order = [m for m in np.argsort(prob.tc_route[r]) if m != r]
        for k in range(1, len(order)):
            m = order[k]
            buyers = y[:, :, r, m] > TOL
            used_tot += int(buyers.sum())
            for mc in order[:k]:
                viol += int((buyers & (out_m[:, :, mc] < I_in[:, :, mc] - TOL)).sum())
    print(f"  互补松弛：用源 {used_tot} 次，其中「便宜源未耗尽就用贵源」{viol} 次"
          f"（占 {viol/max(used_tot,1):.1%}）—— 同上，多半是 MIP rel_gap 放过的次优解")

    # ================================================================ ④ 订货 q
    head("④ 订货 q（here-and-now，L−1 期后到货）")
    print(f"     仓        均值     Qmax  贴Qmax   q=0占比  均值/本仓失效率")
    for r in range(NR):
        print(f"   {r+1:>4} {q[:,:,r].mean():>11,.1f} {prob.Qmax[r]:>8,.0f} "
              f"{(q[:,:,r]>=prob.Qmax[r]-TOL).mean():>7.1%} "
              f"{(q[:,:,r]<=TOL).mean():>8.1%}   "
              f"{q[:,:,r].mean()/max(F[:,:,r].mean(),1e-9):>7.2f}")
    print(f"  全局：贴 Qmax {(q>=prob.Qmax[None,None,:]-TOL).mean():.1%}   "
          f"q=0 {(q<=TOL).mean():.1%}   均值 {q.mean():,.1f}   失效均值 {F.mean():,.1f}"
          f"   比值 {q.mean()/max(F.mean(),1e-9):.2f}")
    load = I_out + o_out.sum(axis=2) + q
    slack = prob.Imax[None, None, :] - load
    print(f"  容量 I_out+Σo_out+q vs Imax：贴限 {(slack<=TOL).mean():.1%}，"
          f"平均余额 {np.round(slack.mean(axis=(0,1)),1).tolist()}")
    print(f"\n  base-stock 形态（订货前位置 I_out+Σo_out 分箱 → q 均值；"
          f"位置越小越缺货，q 应越大）")
    pos = I_out + o_out.sum(axis=2)
    for r in range(NR):
        p_, q_ = pos[:, :, r].ravel(), q[:, :, r].ravel()
        idx = np.clip(np.digitize(p_, np.linspace(0, prob.Imax[r], 6)) - 1, 0, 4)
        cells = []
        for k in range(5):
            if (idx == k).sum() > 5:
                cells.append(f"{p_[idx==k].mean():>6,.0f}→{q_[idx==k].mean():>6,.0f}")
            else:
                cells.append(f"{'—':>6}→{'—':>6}")
        print(f"    仓{r+1}: " + "  ".join(cells))
    print(f"\n  订货的超前性（q_t 与本仓失效 F 的 Pearson r；L={prob.L} ⇒ 滞后 1 期应有正相关）")
    for lag in range(3):
        rs = []
        for r in range(NR):
            a = q[:, :T - lag, r].ravel()
            b = F[:, lag:, r].ravel()
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        print(f"    滞后 {lag} 期: " + "  ".join(f"仓{i+1} {v:+.3f}" for i, v in enumerate(rs)))

    # ================================================================ ⑤ 逐期轨迹
    head("⑤ 逐期轨迹（全路径均值；每 10 期取一个点）")
    tr = {"库存 I": I_in.sum(axis=(0, 2)), "在途 o": o_in.sum(axis=(0, 2, 3)),
          # 停机：先按路径求 (N − Σn)，再对路径取均值 —— 直接 axis=(0,2,3) 求和会得到负数
          "停机": (prob.N.sum() - n_in.sum(axis=3).sum(axis=2)).mean(axis=0),
          "订货 q": q.sum(axis=(0, 2)), "失效 F": F.sum(axis=(0, 2))}
    ages = np.arange(A)
    tot_n = n_in.sum(axis=(0, 2, 3))
    tr["平均月龄"] = np.array([float((n_in[:, t] * ages).sum() / tot_n[t])
                              if tot_n[t] > 0 else np.nan for t in range(T)])
    for k, v in tr.items():
        print(f"  {k:<8}" + " ".join(f"{v[t]:>8,.0f}" if not np.isnan(v[t]) else "     nan"
                                     for t in range(0, T, 10)))

    # ================================================================ ⑦ 决策检验
    head("⑦ 决策合理性检验")
    load = I_out + o_out.sum(axis=2) + q                 # 容量约束左端 I_out+Σo_out+q
    util = load / prob.Imax[None, None, :]
    down = prob.N[None, None, :] - n_in.sum(axis=3)      # 停机台数
    tau_n = taup - taum                                  # 净流入（共享池，不区分对手方）
    DIE = 119                                            # 0 基下标：第 120 期＝到寿集中爆发

    # ---- A. 大停机前有没有提前备货？缺口是不是被容量堵死的？----
    print("  A. 大停机（第 120 期）之前有没有囤货？缺口是不是被 ΣImax 堵死的？")
    print("       仓   Imax    [100,119]平均load   占用率   贴满占比   全期贴满   第120期期初I")
    for r in range(NR):
        u = util[:, 99:DIE, r]
        print(f"     {r+1:>2} {prob.Imax[r]:>7,.0f} {load[:, 99:DIE, r].mean():>15,.0f} "
              f"{u.mean():>8.1%} {(u >= 1 - TOL).mean():>10.1%} "
              f"{(util[:, :, r] >= 1 - TOL).mean():>10.1%} "
              f"{I_in[:, DIE, r].mean():>14,.0f}")
    print(f"     第 120 期期初总库存 {I_in[:, DIE, :].sum(axis=1).mean():>9,.0f} 件   "
          f"总库容 ΣImax={prob.Imax.sum():,.0f}   第 120 期失效 {F[:, DIE, :].sum(axis=1).mean():,.0f} 台")
    print(f"     ⇒ 缺口是否结构性的：失效 {F[:, DIE, :].sum(axis=1).mean():,.0f} 台 vs "
          f"库容 {prob.Imax.sum():,.0f} 件 = {F[:, DIE, :].sum(axis=1).mean()/prob.Imax.sum():.1f} 倍")

    # ---- B. 订货 ----
    print("\n  B. 订货 q：是不是 order-up-to 型？（q>0 时 load=I_out+Σo_out+q 应贴住同一个 S*）")
    for r in range(NR):
        m = q[:, :, r] > TOL
        if m.sum() < 10:
            continue
        L = load[:, :, r][m]
        print(f"     仓{r+1}: q>0 占 {m.mean():>5.1%}   此时 load 均值 {L.mean():>7,.0f}  "
              f"标准差 {L.std():>7,.0f}   "
              f"分位 5%/50%/95% = {np.percentile(L,5):>6,.0f}/{np.percentile(L,50):>6,.0f}/"
              f"{np.percentile(L,95):>6,.0f}   Imax={prob.Imax[r]:,.0f}")
        m0 = ~m
        if m0.sum() > 10:
            L0 = load[:, :, r][m0]
            print(f"           q=0 占 {m0.mean():>5.1%}   此时 load 均值 {L0.mean():>7,.0f}  "
                  f"贴满 Imax 占 {(L0 >= prob.Imax[r] - TOL).mean():>5.1%}  "
                  f"位置已 ≥0.9·Imax 占 {(L0 >= 0.9*prob.Imax[r]).mean():>5.1%}")
    print("     q_t 对【本仓下期失效】的响应（按 F_{t+1} 四分位分箱看 q 均值）：")
    for r in range(NR):
        fn = F[:, 1:, r].ravel()
        qq = q[:, :T - 1, r].ravel()
        qt = np.quantile(fn, [0, .25, .5, .75, 1.0])
        bins = np.clip(np.digitize(fn, qt[1:4]), 0, 3)
        cells = [f"{(bins==k).sum():>7d}个→q均值 {qq[bins==k].mean():>6,.1f}"
                 if (bins == k).sum() > 10 else "—" for k in range(4)]
        print(f"     仓{r+1}: " + "  ".join(cells))

    # ---- C. 主动调拨（共享池：τ_r>0 从池中取，τ_r<0 向池中放）----
    print("\n  C. 主动调拨 τ：方向对不对？（净流入 vs 该仓的库存/停机）")
    print("       仓   净流入τ均值   corr(τ, 期初库存)   corr(τ, 停机台数)   调出次数  调入次数")
    for r in range(NR):
        a = tau_n[:, :, r].ravel()
        b = I_in[:, :, r].ravel()
        c = down[:, :, r].ravel()
        print(f"     {r+1:>2} {tau_n[:,:,r].mean():>12,.1f} "
              f"{np.corrcoef(a,b)[0,1]:>16.3f} {np.corrcoef(a,c)[0,1]:>18.3f} "
              f"{(taum[:,:,r]>TOL).sum():>10d} {(taup[:,:,r]>TOL).sum():>10d}")
    print("     （期望：净流入与期初库存负相关、与停机正相关 ⇒ 从富余仓调向缺货仓）")

    # ---- D. 被动调拨：本地优先吗？----
    print("\n  D. 被动调拨 y：本地用光了才跨仓吗？")
    for r in range(NR):
        cross = y[:, :, r, :].sum(axis=2) - y[:, :, r, r]
        used = cross > TOL
        if used.sum() < 10:
            continue
        local_full = (y[:, :, r, r] >= I_in[:, :, r] - TOL)[used]
        print(f"     仓{r+1}: 跨仓次数 {used.sum():>7d}   其中「本地库存已用光」占 "
              f"{local_full.mean():>6.1%}   本地量/期初库存均值 "
              f"{(y[:,:,r,r]/np.maximum(I_in[:,:,r],1)).mean():>5.1%}")
    yd = np.trace(y, axis1=2, axis2=3)
    print(f"\n     本地 y/期初库存 的平均饱和度 {(yd/np.maximum(I_in.sum(axis=2),1)).mean():.1%}"
          f"（若≈100% 说明本地总是先被用光）")


if __name__ == "__main__":
    main()
