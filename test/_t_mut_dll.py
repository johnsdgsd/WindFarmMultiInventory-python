# 原生 copt.dll 原地改模型参数的最小验证（ctypes 直调，绕开 coptpy）：
# 变量界 / 约束右端能不能改、改了起不起作用、对偶能不能提、比整模重载快多少。
#   python test/_t_mut_dll.py
#
# 同一个基准 LP（期望值手算，与 test/_t_mut_coptpy.py 同题，可交叉对拍）：
#     min  4x + y + 100w
#     c1:  x + y + w >= 3         行下界 = 3
#     c2:  2x + 3y   <= 8         行上界 = 8
#     0 <= x, y, w <= 10
#   解 x=1, y=2, w=0, obj=6；对偶 π=(10,−3)；既约成本 dj=(0,0,90)。
#
# 行界口径：LoadProb 只收 rowSense + rowBound，COPT 内部转成 [LB, UB]——
# 'G' → [bound, +∞)，'L' → (−∞, bound]，'E' → [bound, bound]。
# 故改右端 = COPT_SetRowLower（≥/E 行）或 COPT_SetRowUpper（≤/E 行）。

import ctypes as C
import sys
import time

sys.stdout.reconfigure(errors="replace")

from pulp.apis.copt_api import COPT_DLL_loadlib                       # noqa: E402

INF = 1e30
LIB = COPT_DLL_loadlib()                    # 复用 pulp 已验证的路径/许可装载

# ---- 1. 函数原型（签名逐条对照 copt.h，出错就让 ctypes 直接报，不要静默） ----
P_D = C.POINTER(C.c_double)
P_I = C.POINTER(C.c_int)
SIG = {
    "COPT_CreateEnv":       ([C.POINTER(C.c_void_p)], C.c_int),
    "COPT_DeleteEnv":       ([C.POINTER(C.c_void_p)], C.c_int),
    "COPT_CreateProb":      ([C.c_void_p, C.POINTER(C.c_void_p)], C.c_int),
    "COPT_DeleteProb":      ([C.POINTER(C.c_void_p)], C.c_int),
    "COPT_LoadProb":        ([C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_double,
                              P_D, P_I, P_I, P_I, P_D, C.c_char_p, P_D, P_D,
                              C.c_char_p, P_D, P_D,
                              C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)],
                             C.c_int),
    "COPT_SolveLp":         ([C.c_void_p], C.c_int),
    "COPT_SetIntParam":     ([C.c_void_p, C.c_char_p, C.c_int], C.c_int),
    "COPT_GetIntAttr":      ([C.c_void_p, C.c_char_p, P_I], C.c_int),
    "COPT_GetDblAttr":      ([C.c_void_p, C.c_char_p, P_D], C.c_int),
    "COPT_GetLpSolution":   ([C.c_void_p, P_D, P_D, P_D, P_D], C.c_int),
    "COPT_SetColLower":     ([C.c_void_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_SetColUpper":     ([C.c_void_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_SetRowLower":     ([C.c_void_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_SetRowUpper":     ([C.c_void_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_GetColInfo":      ([C.c_void_p, C.c_char_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_GetRowInfo":      ([C.c_void_p, C.c_char_p, C.c_int, P_I, P_D], C.c_int),
    "COPT_WriteLp":         ([C.c_void_p, C.c_char_p], C.c_int),
}
F = {}
for name, (argt, restype) in SIG.items():
    fn = getattr(LIB, name)
    fn.argtypes, fn.restype = argt, restype
    F[name] = fn
print(f"copt.dll 已装载，{len(F)} 个函数原型就位（{LIB._name}）")

# ---- 2. 基准 LP 的 CSC 数据（列序 x,y,w；行序 c1,c2） ----
NC, NR, NNZ = 3, 2, 5
OBJ = (C.c_double * NC)(4.0, 1.0, 100.0)
COL_LB = (C.c_double * NC)(0.0, 0.0, 0.0)
COL_UB = (C.c_double * NC)(10.0, 10.0, 10.0)
COL_BEG = (C.c_int * NC)(0, 2, 4)
COL_CNT = (C.c_int * NC)(2, 2, 1)
COL_IDX = (C.c_int * NNZ)(0, 1, 0, 1, 0)          # 行号
COL_VAL = (C.c_double * NNZ)(1.0, 2.0, 1.0, 3.0, 1.0)
ROW_SENSE = C.c_char_p(b"GL")                       # c1 ≥, c2 ≤
ROW_BOUND = (C.c_double * NR)(3.0, 8.0)
COL_NAME = (C.c_char_p * NC)(b"x", b"y", b"w")
ROW_NAME = (C.c_char_p * NR)(b"c1", b"c2")
ALL_C = (C.c_int * NC)(0, 1, 2)
ALL_R = (C.c_int * NR)(0, 1)

ROWS = []


def close(a, b, tol=1e-9):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(u, v, tol) for u, v in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def check(name, got, want, tol=1e-9, note=""):
    ok = close(got, want, tol)
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))
    return ok


def new_prob():
    env, prob = C.c_void_p(), C.c_void_p()
    rc = F["COPT_CreateEnv"](C.byref(env))
    assert rc == 0, rc
    rc = F["COPT_CreateProb"](env, C.byref(prob))
    assert rc == 0, rc
    rc = F["COPT_SetIntParam"](prob, b"Logging", 0)
    assert rc == 0, rc
    rc = F["COPT_LoadProb"](prob, NC, NR, 1, C.c_double(0.0), OBJ,
                            COL_BEG, COL_CNT, COL_IDX, COL_VAL, None,
                            COL_LB, COL_UB, ROW_SENSE, ROW_BOUND, None,
                            COL_NAME, ROW_NAME)
    assert rc == 0, f"LoadProb rc={rc}"
    return env, prob


def solve(prob, nc=NC, nr=NR):
    """nc/nr 必须与模型实际行列数一致——COPT 按模型维度往缓冲区里写，给小了就是越界写。"""
    rc = F["COPT_SolveLp"](prob)
    if rc != 0:
        return {"status": f"SolveLp rc={rc}"}
    st, obj = C.c_int(), C.c_double()
    F["COPT_GetIntAttr"](prob, b"LpStatus", C.byref(st))
    F["COPT_GetDblAttr"](prob, b"LpObjVal", C.byref(obj))
    x = (C.c_double * nc)(); slack = (C.c_double * nr)()
    pi = (C.c_double * nr)(); dj = (C.c_double * nc)()
    F["COPT_GetLpSolution"](prob, x, slack, pi, dj)
    return {"status": st.value, "obj": obj.value, "x": list(x),
            "pi": list(pi), "dj": list(dj), "slack": list(slack)}


def bounds(prob, nc=NC, nr=NR):
    """把模型内部真正的 [LB, UB] 读回来——证明改动落到了数据上，而非碰巧重解一致。"""
    ca, ra = (C.c_int * nc)(*range(nc)), (C.c_int * nr)(*range(nr))
    clb, cub = (C.c_double * nc)(), (C.c_double * nc)()
    rlb, rub = (C.c_double * nr)(), (C.c_double * nr)()
    F["COPT_GetColInfo"](prob, b"LB", nc, ca, clb)
    F["COPT_GetColInfo"](prob, b"UB", nc, ca, cub)
    F["COPT_GetRowInfo"](prob, b"LB", nr, ra, rlb)
    F["COPT_GetRowInfo"](prob, b"UB", nr, ra, rub)
    return {"col": (list(clb), list(cub)), "row": (list(rlb), list(rub))}


def batch(fname, prob, idx, vals):
    """批改：一次调用改一批界。返回 rc（非 0 直接抛，别让改动静默丢失）。"""
    n = len(idx)
    rc = F[fname](prob, n, (C.c_int * n)(*idx), (C.c_double * n)(*vals))
    assert rc == 0, f"{fname} rc={rc}"
    return rc


env, prob = new_prob()

print()
print("=" * 78)
print("§1 LoadProb 后行界的内部表示（决定改右端该调哪个函数）")
print("=" * 78)
b0 = bounds(prob)
print(f"  列 LB={b0['col'][0]}  UB={b0['col'][1]}")
print(f"  行 LB={b0['row'][0]}  UB={b0['row'][1]}      （±{INF:g} 即无穷）")

print()
print("=" * 78)
print("§2 基准解与对偶")
print("=" * 78)
s0 = solve(prob)
print(f"  status={s0['status']} obj={s0['obj']:.6f} x={s0['x']} π={s0['pi']} "
      f"dj={s0['dj']} slack={s0['slack']}")
check("基准 obj", s0["obj"], 6.0)
check("基准 x", s0["x"], [1.0, 2.0, 0.0])
check("对偶 π", s0["pi"], [10.0, -3.0])
check("既约成本 dj", s0["dj"], [0.0, 0.0, 90.0])
check("强对偶 π1·3+π2·8", s0["pi"][0] * 3 + s0["pi"][1] * 8, 6.0)

print()
print("=" * 78)
print("§3 四条原地改路径（改 → 重解 → 手算值；再读回界证明落到数据上）")
print("=" * 78)


def case(name, fname, idx, val, expect_obj, what):
    batch(fname, prob, idx, [val])
    got = solve(prob)
    check(f"{name} · obj", got["obj"], expect_obj, note="手算")
    readback = bounds(prob)
    side = "LB" if "Lower" in fname else "UB"
    isrow = "Row" in fname
    real = readback["row" if isrow else "col"][0 if side == "LB" else 1][idx[0]]
    check(f"{name} · {what}读回", real, val, note=f"{'行' if isrow else '列'} {side}")


case("A 变量下界 SetColLower(x,2)", "COPT_SetColLower", [0], 2.0, 9.0, "x.LB")
case("B 变量上界 SetColUpper(y,0.5)", "COPT_SetColUpper", [1], 0.5, 10.5, "y.UB")

# 逐个还原，保证每条用例的起点都是基准
batch("COPT_SetColLower", prob, [0], [0.0])
batch("COPT_SetColUpper", prob, [1], [10.0])
check("还原后 obj", solve(prob)["obj"], 6.0)

case("C 行下界 SetRowLower(c1,4)", "COPT_SetRowLower", [0], 4.0, 16.0, "c1.LB")
batch("COPT_SetRowLower", prob, [0], [3.0])
check("还原后 obj", solve(prob)["obj"], 6.0)

case("D 行上界 SetRowUpper(c2,9)", "COPT_SetRowUpper", [1], 9.0, 3.0, "c2.UB")
batch("COPT_SetRowUpper", prob, [1], [8.0])
back = solve(prob)
check("全部还原后 obj", back["obj"], 6.0)
check("全部还原后 x", back["x"], [1.0, 2.0, 0.0])
check("全部还原后 π", back["pi"], [10.0, -3.0])

print()
print("=" * 78)
print("§4 差商验证：原地改一次再读回来的 π / dj 是不是真的 ∂obj/∂参数")
print("=" * 78)
D = 1e-4
for label, fname, idx, base_v, want, note in [
    ("∂obj/∂c1.LB == π1", "COPT_SetRowLower", [0], 3.0, s0["pi"][0], "行下界 +δ"),
    ("∂obj/∂c2.UB == π2", "COPT_SetRowUpper", [1], 8.0, s0["pi"][1], "行上界 +δ"),
    ("∂obj/∂w.LB  == dj_w", "COPT_SetColLower", [2], 0.0, s0["dj"][2], "列下界 +δ"),
]:
    batch(fname, prob, idx, [base_v + D])
    got = (solve(prob)["obj"] - s0["obj"]) / D
    batch(fname, prob, idx, [base_v])
    check(label, got, want, tol=1e-4 * max(1.0, abs(want)), note=note)

print()
print("=" * 78)
print("§5 计时：原地改界+重解 vs 整模重新 LoadProb（600 列 / 400 行拼块实例）")
print("=" * 78)
K = 200
B_NC, B_NR = 3 * K, 2 * K
B_OBJ = (C.c_double * B_NC)(*[c for _ in range(K) for c in (4.0, 1.0, 100.0)])
B_LB = (C.c_double * B_NC)(*[0.0] * B_NC)
B_UB = (C.c_double * B_NC)(*[10.0] * B_NC)
_b_cnt = [2 if i % 3 != 2 else 1 for i in range(B_NC)]
_b_beg, _acc = [], 0
for _c in _b_cnt:
    _b_beg.append(_acc)
    _acc += _c
B_BEG = (C.c_int * B_NC)(*_b_beg)
B_CNT = (C.c_int * B_NC)(*_b_cnt)
B_IDX = (C.c_int * (5 * K))(*[r for k in range(K)
                              for r in (2 * k, 2 * k + 1, 2 * k, 2 * k + 1, 2 * k)])
B_VAL = (C.c_double * (5 * K))(*[v for _ in range(K) for v in (1.0, 2.0, 1.0, 3.0, 1.0)])
B_SENSE = C.c_char_p(b"GL" * K)
B_BOUND = (C.c_double * B_NR)(*[3.0, 8.0] * K)
B_ALL_C = (C.c_int * B_NC)(*range(B_NC))
B_ALL_R = (C.c_int * B_NR)(*range(B_NR))


def load_big(prob):
    """整模重载。CSC 索引每次都必须自洽——B_BEG 是 B_CNT 的前缀和，写错会越界。"""
    rc = F["COPT_LoadProb"](prob, B_NC, B_NR, 1, C.c_double(0.0), B_OBJ,
                            B_BEG, B_CNT, B_IDX, B_VAL, None,
                            B_LB, B_UB, B_SENSE, B_BOUND, None, None, None)
    assert rc == 0, f"LoadProb rc={rc}"


t0 = time.perf_counter()
envB, probB = C.c_void_p(), C.c_void_p()
F["COPT_CreateEnv"](C.byref(envB)); F["COPT_CreateProb"](envB, C.byref(probB))
F["COPT_SetIntParam"](probB, b"Logging", 0)
load_big(probB)
t_load = time.perf_counter() - t0
t0 = time.perf_counter()
solve(probB, B_NC, B_NR)
t_first = time.perf_counter() - t0

n_it = 30
t0 = time.perf_counter()
for _ in range(n_it):
    batch("COPT_SetColUpper", probB, list(range(B_NC)), [10.0] * B_NC)
    batch("COPT_SetRowLower", probB, list(range(B_NR)), [3.0, -INF] * K)
    F["COPT_SolveLp"](probB)
t_inplace = (time.perf_counter() - t0) / n_it

t0 = time.perf_counter()
for _ in range(n_it):
    load_big(probB)
    F["COPT_SolveLp"](probB)
t_reload = (time.perf_counter() - t0) / n_it

print(f"  {B_NC} 列 / {B_NR} 行；每轮 = 列界批改 + 行界批改")
print(f"    建环境+LoadProb     {t_load*1e3:9.2f} ms")
print(f"    首次求解            {t_first*1e3:9.2f} ms")
print(f"    原地改界 + 重解     {t_inplace*1e3:9.2f} ms/次")
print(f"    重新 LoadProb + 解  {t_reload*1e3:9.2f} ms/次")
print(f"    加速比              {t_reload/t_inplace:9.2f} ×")

print()
print("=" * 78)
print("汇总")
print("=" * 78)
w = max(len(r[0]) for r in ROWS)
n_fail = 0
for name, st, got, want, note in ROWS:
    n_fail += st == "FAIL"
    g = got if isinstance(got, str) else (f"{got:.6f}" if not isinstance(got, list)
                                         else str([round(x, 4) for x in got]))
    print(f"  [{st}] {name:<{w}}  got={g}  期望={want}  {note}")
print()
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
