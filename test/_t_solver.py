# 求解器可用性自检：coptpy 能否装载、能否解 LP/MIP、对偶能否读出、参数能否生效。
#   python test/_t_solver.py
#
# 装/卸 COPT、换 conda 环境、许可到期之后跑一次。

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

import coptpy as cp                                                    # noqa: E402
from coptpy import COPT                                                # noqa: E402

from utils import copt_env as ce                                       # noqa: E402

print(f"coptpy {cp.__version__ if hasattr(cp, '__version__') else ''}  "
      f"COPT {COPT.VERSION_MAJOR}.{COPT.VERSION_MINOR}.{COPT.VERSION_TECHNICAL}")

ROWS = []


def check(name, ok, note=""):
    ROWS.append((name, "PASS" if ok else "FAIL", note))


# ---- LP：解 + 对偶 ----
m = ce.new_model()
x = m.addVar(lb=0.0, ub=3.0, obj=1.0, name="x")
c = m.addConstr(x >= 1.0, name="c")
m.setObjective(x, COPT.MINIMIZE)
obj = ce.solve(m, mip=False)
check("LP 求解", abs(obj - 1.0) < 1e-9, f"obj={obj}")
check("LP 取值", abs(ce.values(m, x) - 1.0) < 1e-9)
check("LP 对偶可读", abs(float(m.getInfo(COPT.Info.Dual, [c])[0]) - 1.0) < 1e-9)

# ---- MIP ----
m2 = ce.new_model()
a = m2.addVar(lb=0.0, ub=3.0, obj=1.0, vtype=COPT.INTEGER, name="a")
b = m2.addVar(lb=0.0, ub=3.0, obj=10.0, vtype=COPT.INTEGER, name="b")
m2.addConstr(2 * a + 3 * b >= 6, name="d")
m2.setObjective(a + 10 * b, COPT.MINIMIZE)
obj2 = ce.solve(m2, mip=True)
check("MIP 求解", abs(obj2 - 3.0) < 1e-9, f"obj={obj2}")
check("MIP 解为整数", [ce.values(m2, v) for v in (a, b)] == [3.0, 0.0])

# ---- 参数生效（注意 setParam 返回 None，不能拿返回值判成败）----
m.setParam(COPT.Param.Threads, 1)
m.setParam(COPT.Param.FeasTol, 1e-8)
check("Threads 可设", m.getParam(COPT.Param.Threads) == 1)
check("容差可设", abs(m.getParam(COPT.Param.FeasTol) - 1e-8) < 1e-12)

# ---- 落盘：中文路径必须走 ASCII 临时目录 ----
from pathlib import Path as _P                                        # noqa: E402
tmp_cn = ROOT / "test" / "_tmp_solver_cn"
tmp_cn.mkdir(parents=True, exist_ok=True)
dest = tmp_cn / "probe.lp"
got = ce.dump_lp(m, dest)
check("dump_lp 能落到中文路径（经 ASCII 中转）",
      got is not None and _P(got).is_file(), f"{got}")
if got is not None:
    _P(got).unlink()
tmp_cn.rmdir()

print()
n_fail = 0
for name, st, note in ROWS:
    n_fail += st == "FAIL"
    print(f"  [{st}] {name}   {note}")
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
