# test/_t_why55.py — 追查第 55 期停机暴增的来源：是「补不上」还是「不想补」？
#
#   python test/_t_why55.py results/<目录>
#
# 逐期拆出守恒链：失效 F → 安装 Σy → 停机存量 D = N − Σn，并对照产能上限 ΣQmax。
# 停机是**存量**：D_t = D_{t−1} + F_t − 安装_t，所以单期缺口会逐期累积。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config            # noqa: E402
from code.problem_data import DefineProblem            # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python test/_t_why55.py <结果目录>")
    d = Path(sys.argv[1])
    if not d.is_absolute():
        d = ROOT / d
    f = sorted(d.glob("sim_*paths.npz"))[-1]
    z = np.load(f)

    prob = DefineProblem(load_problem_config(d / "config" / "problem.toml"))
    Ly, NR, A, T, L = prob.layout, prob.N_R, prob.A, prob.T, prob.L
    P = z["x_in"].shape[0]

    x_in = z["x_in"].astype(float)
    x_out = z["x_out"].astype(float)
    q = z["q"].astype(float)
    y = z["y"].astype(float)
    scene = z["scene"].astype(int)

    n_in = x_in[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A)
    n_out = x_out[:, :, Ly.off_n:Ly.off_n + Ly.n_n].reshape(P, T, NR, A)
    n_arr = n_out[:, :, :, 0].sum(axis=2)                  # 本期新装机队（月龄 0 出栏）

    F = np.zeros((P, T, NR))
    for t in range(T):
        th = np.array([prob.pool[scene[p, t]][1] for p in range(P)])
        F[:, t, :] = np.rint(th * n_in[:, t]).sum(axis=2)

    N = prob.N.sum()
    D = N - n_in.sum(axis=3).sum(axis=2)                   # (P,T) 停机存量
    # 逐期净增 ΔD_t = F_{t−1} − Σy_{t−1}（x_in(t) = x_out(t−1)）
    dD = np.zeros(T)
    dD[1:] = D[:, 1:].mean(axis=0) - D[:, :-1].mean(axis=0)

    print(f"{f.name}   路径 {P} × {T} 期")
    print(f"机队 {N:,.0f}  ΣQmax={prob.Qmax.sum():,.0f}  ΣImax={prob.Imax.sum():,.0f}  L={L}")
    print()
    hdr = (f"{'期':>4} {'失效F':>8} {'订货q':>8} {'安装Σy':>8} {'停机D':>8} "
           f"{'ΔD':>7} {'F−ΣQmax':>8} {'F−Σy':>8} {'库存I_in':>9} {'在途':>7}")
    print(hdr)
    print("-" * len(hdr))
    for t in range(T):
        Fm = F[:, t, :].sum(axis=1).mean()
        qm = q[:, t, :].sum(axis=1).mean()
        ym = y[:, t, :, :].sum(axis=(1, 2)).mean()
        Dm = D[:, t].mean()
        Ii = x_in[:, t, :Ly.n_I].sum(axis=1).mean()
        oi = x_in[:, t, Ly.off_o:Ly.off_o + Ly.n_o].sum(axis=1).mean()
        mark = "  ←" if t + 1 in (43, 49, 55, 60) else ""
        print(f"{t+1:>4} {Fm:>8.1f} {qm:>8.1f} {ym:>8.1f} {Dm:>8.1f} "
              f"{dD[t]:>+7.1f} {Fm-prob.Qmax.sum():>+8.1f} {Fm-ym:>+8.1f} "
              f"{Ii:>9.1f} {oi:>7.1f}{mark}")

    # —— 分期解析：缺口是「订不到」（q 贴 Qmax）还是「不想订」（q 远低于 Qmax）
    print()
    peak = int(np.argmax(D.mean(axis=0)))
    print(f"停机峰值在第 {peak+1} 期，均值 {D[:, peak].mean():.1f} 台")
    for label, t in (("第 43 期", 42), ("第 49 期", 48), ("第 55 期", 54)):
        Fm = F[:, t, :].sum(axis=1)
        qm = q[:, t, :].sum(axis=1)
        print(f"\n{label}：失效 {Fm.mean():.1f}  订货 {qm.mean():.1f} "
              f"（ΣQmax={prob.Qmax.sum()}，贴限占 {(qm >= prob.Qmax.sum()-1e-6).mean():.1%}）")
        print("   分仓：  " + "  ".join(
            f"仓{r+1} F={F[:,t,r].mean():>6.1f} q={q[:,t,r].mean():>6.1f}"
            f"(Qmax={prob.Qmax[r]})" for r in range(NR)))

    # —— 末期窗口：q 是「贴 Qmax」被限、还是「贴 Imax」被限、还是自愿少订？
    print(f"\n末 10 期分仓：q 被谁限制？"
          f"（贴Qmax = q_out+1/2 的 Slack≈0；贴Imax = I_out+Σo_out+q 顶到 Imax）")
    Lo = x_out[:, :, :Ly.n_I]
    oo = x_out[:, :, Ly.off_o:Ly.off_o + Ly.n_o].reshape(P, T, L - 1, NR)
    for t in range(T - 10, T):
        print(f"\n   期{t+1}")
        for r in range(NR):
            qm = q[:, t, r].mean()
            lim = min(prob.Qmax[r], prob.Imax[r] - Lo[:, t, r].mean()
                      - oo[:, t, :, r].sum(axis=1).mean())
            print(f"     仓{r+1}  I_in={x_in[:,t,Ly.ix_I(r)].mean():>7.1f}  "
                  f"I_out={Lo[:,t,r].mean():>7.1f}  "
                  f"Σo={oo[:,t,:,r].sum(axis=1).mean():>7.1f}  "
                  f"q={qm:>6.1f}  Qmax={prob.Qmax[r]:>3}  Imax={prob.Imax[r]:>4}  "
                  f"容量余量={prob.Imax[r]-Lo[:,t,r].mean()-oo[:,t,:,r].sum(axis=1).mean()-qm:>7.1f}"
                  f"  {'← 贴Qmax' if qm >= prob.Qmax[r]-1e-6 else ''}")

    # —— 末期窗口：D 的增量是「订不到」还是「不想订」
    lo = T - 20
    print(f"\n末 20 期（{lo+1}..{T}）逐期：F vs q vs ΣQmax")
    short = (F.sum(axis=2) - prob.Qmax.sum()).mean(axis=0)
    for t in range(lo, T):
        print(f"   期{t+1:>3}  F={F[:,t,:].sum(axis=1).mean():>7.1f}  "
              f"q={q[:,t,:].sum(axis=1).mean():>7.1f}  "
              f"缺口F−ΣQmax={short[t]:>+7.1f}  D={D[:,t].mean():>7.1f}")


if __name__ == "__main__":
    main()
