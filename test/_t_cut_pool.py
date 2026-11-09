# _t_cut_pool.py — SDDP.jl 风格割池的单元测试（缓存、移出、复活）。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.cut import Cut, trivial_cut           # noqa: E402
from utils.cut_pool import CutPool              # noqa: E402


def cut(alpha, beta, state):
    return Cut(alpha, np.asarray(beta, float), np.asarray(state, float),
               1, alpha, 1)


def main():
    p = CutPool(1, deletion_minimum=1)
    triv = trivial_cut(1, np.zeros(2), 2)
    triv.uid = ("triv",)
    p.seed(1, [triv])
    assert [c.alpha for c in p.active(1)] == [0.0]

    a = cut(10.0, [0, 0], [0.0, 0.0])
    a.uid = "a"
    act, deact = p.add(1, a)
    assert a in p.active(1)
    assert [c.alpha for c in p.active(1)] == [0.0, 10.0]

    b = cut(20.0, [0, 0], [0.0, 0.0])
    b.uid = "b"
    act, deact = p.add(1, b)
    assert any(c is b for c in p.active(1))
    assert not any(c is a for c in p.active(1)), "b 在相同点支配 a，a 应从活跃池移除"
    assert [c.alpha for c in p.active(1)] == [0.0, 20.0]
    assert len(p.all_cuts(1)) == 3, "被移除的割必须留在缓存"

    # 新点取值很低；旧的不活跃割 a 在该点更高，应复活。
    d = cut(0.0, [0, 0], [1.0, 1.0])
    d.uid = "d"
    p.add(1, d)
    assert any(c is a for c in p.active(1)), "旧割在新采样点占优时必须复活"
    assert any(c is b for c in p.active(1))
    assert not any(c is d for c in p.active(1)), \
        "d 未支配任何采样点，应被移出活跃池"
    assert len(p.all_cuts(1)) == 4
    assert len(p.active(1)) == 3
    assert any(c is d for c in p.all_cuts(1)), "被移出的 d 也必须留在缓存"
    print("OK _t_cut_pool")


if __name__ == "__main__":
    main()
