# 求解器可用性自检：make_solver 当前会挑到谁、能否解 LP、对偶能否读出。
# 装/卸 COPT、切 conda 环境这类改动之后跑一次。

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pulp

from utils.solver import make_solver

print("available:", pulp.listSolvers(onlyAvailable=True))

for mip in (False, True):
    s = make_solver(mip=mip)
    p = pulp.LpProblem("t", pulp.LpMinimize)
    x = pulp.LpVariable("x", 0, 3)
    p += x >= 1
    p += x
    p.solve(s)
    print(f"mip={mip} -> {type(s).__name__:16s} status={pulp.LpStatus[p.status]} "
          f"x={x.value()} dj={x.dj}")
