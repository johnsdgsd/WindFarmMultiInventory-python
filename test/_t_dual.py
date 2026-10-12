# 读后向现场 LP 的**逐约束对偶**，定位哪一行塌成 0。
#
#   python test/_t_dual.py <现场 LP 路径> <场景号>
#
# 诊断目标：F 的两条定义式（Fdef_hi/lo）的对偶是否非零 —— 它是「月龄通道」
# ∂Q/∂n 里 θ·∂Q/∂F 那一项的来源。若 F 停在松弛区间内部、两条式都松，该项为 0，
# 割就会退化成「只含当期停机成本」。纯读，不碰任何生产代码。
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import coptpy as cp
from coptpy import COPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import load_problem_config        # noqa: E402
from code.problem_data import DefineProblem        # noqa: E402


def main():
    src = Path(sys.argv[1])
    src = src if src.is_absolute() else ROOT / src
    s_idx = int(sys.argv[2])
    pcfg = load_problem_config(ROOT / "config" / "problem.toml")
    prob = DefineProblem(pcfg)
    Ly, NR, A = prob.layout, prob.N_R, prob.A

    # COPT 的文件接口走不了中文路径 ⇒ 先复制到 ASCII 临时目录
    tmp = Path(tempfile.gettempdir()) / "dual_diag"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / "p.lp"
    shutil.copyfile(src, dst)

    env = cp.Envr()
    m = env.createModel()
    m.setParam(COPT.Param.Logging, 0)
    m.readLp(str(dst))
    print(f"{src.name}   场景 {s_idx}")
    print(f"  规模：{m.getAttr(COPT.Attr.Rows)} 行 × {m.getAttr(COPT.Attr.Cols)} 列   "
          f"整数列 {m.getAttr(COPT.Attr.Ints)}")
    m.solveLP()
    print(f"  求解 status = {m.status}（{COPT.OPTIMAL}=最优）")
    if m.status != COPT.OPTIMAL:
        print("  ★ 这个现场 LP 本身不可解 —— 它是一次**求解失败**的转储，"
              "不能用来读对偶。")
        return
    print(f"  LP 目标 = {m.getAttr(COPT.Attr.LpObjVal):,.4f}")

    def dual(name):
        try:
            return float(m.getInfo(COPT.Info.Dual, [m.getConstrByName(name)])[0])
        except Exception:
            return float("nan")

    def val(name):
        try:
            return float(m.getInfo(COPT.Info.Value, [m.getVarByName(name)])[0])
        except Exception:
            return float("nan")

    print("\n  —— 各约束族的对偶（非零 ⇒ 该行在承载信息）——")
    fam = {
        "pas_mm":     [f"pas_{m_}" for m_ in range(NR)],
        "dem_r":      [f"dem_{r}" for r in range(NR)],
        "inv_r":      [f"inv_{r}" for r in range(NR)],
        "cap_r":      [f"cap_{r}" for r in range(NR)],
        "age_0_a":    [f"age_0_{a}" for a in range(0, A - 1, 8)],
        "Fdef_hi_0_a": [f"Fdef_hi_0_{a}" for a in range(0, A, 8)],
        "Fdef_lo_0_a": [f"Fdef_lo_0_{a}" for a in range(0, A, 8)],
    }
    for k, names in fam.items():
        d = np.array([dual(n) for n in names])
        print(f"    {k:<14} " + " ".join(f"{v:>10.3g}" for v in d))

    print("\n  —— F 停在松弛区间的哪里？（rint 口径 ⇒ 区间 [θn−0.5, θn+0.5]）——")
    th = np.asarray(prob.pool[s_idx][1], float)
    print(f"    {'仓':>3} {'月龄':>4} {'n=xin':>8} {'θ':>10} {'θ·n':>9} "
          f"{'区间下界':>9} {'区间上界':>9} {'F 实际':>9} {'离最近边界':>11}")
    r0 = 0
    for a in list(range(0, A, 10)) + [A - 2, A - 1]:
        n_val = val(f"xin_{Ly.ix_n(r0, a)}")
        F_val = val(f"F_{r0}_{a}")
        lo, hi = th[r0, a] * n_val - 0.5, th[r0, a] * n_val + 0.5
        gap = min(abs(F_val - lo), abs(F_val - hi))
        print(f"    {r0+1:>3} {a:>4} {n_val:>8.1f} {th[r0, a]:>10.6f} "
              f"{th[r0, a]*n_val:>9.3f} {lo:>9.3f} {hi:>9.3f} {F_val:>9.3f} {gap:>11.3f}")

    print("\n  —— 状态取值（看锚点是不是「满编」）——")
    n_tot = [sum(val(f"xin_{Ly.ix_n(r, a)}") for a in range(A)) for r in range(NR)]
    print(f"    各仓 Σn   = {[round(v) for v in n_tot]}")
    print(f"    各仓 N    = {prob.N.tolist()}")
    print(f"    停机台数  = {[int(prob.N[r] - n_tot[r]) for r in range(NR)]}")
    print(f"    各仓 I    = {[round(val(f'xin_{Ly.ix_I(r)}')) for r in range(NR)]}")
    print(f"    各仓 q    = {[round(val(f'xin_{Ly.ix_q(r)}')) for r in range(NR)]}")

    # —— 机队落在哪些月龄桶、每桶 θ·n 多大（决定 rint 是否还在传递 ∂F/∂n）——
    print(f"\n  —— 有货的月龄桶（rint 只在 θ·n ≳ 0.5 时才不为零）——")
    n_all = np.array([[val(f"xin_{Ly.ix_n(r, a)}") for a in range(A)]
                      for r in range(NR)])
    occ = [(r, int(a), n_all[r, a], th[r, a], th[r, a] * n_all[r, a])
           for r in range(NR) for a in range(A) if n_all[r, a] > 1e-6]
    print(f"    有货桶数 {len(occ)} / {NR*A}   每桶台数 中位 "
          f"{np.median([o[2] for o in occ]):.1f}  最大 {max(o[2] for o in occ):.1f}")
    print(f"    θ·n 分布：中位 {np.median([o[4] for o in occ]):.3f}  "
          f"最大 {max(o[4] for o in occ):.3f}  "
          f"≥1 的桶 {sum(1 for o in occ if o[4] >= 1)} 个  "
          f"≥0.5 的桶 {sum(1 for o in occ if o[4] >= 0.5)} 个")
    print(f"    {'仓':>3} {'月龄':>4} {'台数':>7} {'θ':>10} {'θ·n':>8} {'F':>6} "
          f"{'Fdef_hi对偶':>12} {'Fdef_lo对偶':>12} {'F离下界':>9} {'F离上界':>9}")
    for r, a, nv, tv, pv in sorted(occ, key=lambda o: -o[4])[:10]:
        fv = val(f"F_{r}_{a}")
        dhi = dual(f"Fdef_hi_{r}_{a}")
        dlo = dual(f"Fdef_lo_{r}_{a}")
        print(f"    {r+1:>3} {a:>4} {nv:>7.1f} {tv:>10.6f} {pv:>8.3f} {fv:>6.1f} "
              f"{dhi:>12.4g} {dlo:>12.4g} {fv-(pv-0.5):>9.4f} {(pv+0.5)-fv:>9.4f}")


if __name__ == "__main__":
    main()
