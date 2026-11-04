# _t_refine_unit.py — 细化纯函数：theta cap、切簇、离散度、停滞触发。
#   python test/_t_refine_unit.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.partition import (cluster_dispersion, normalized_cluster_score,
                             resolve_theta_cap, split_cluster, stall_trigger,
                             validate_partition)


def main():
    assert resolve_theta_cap("single", 0, 30) == 1
    assert resolve_theta_cap("multi", 0, 30) == 30
    assert resolve_theta_cap("multi", 8, 30) == 8
    assert resolve_theta_cap("multi", 8, 30, refine_max_clusters=32) == 32
    assert resolve_theta_cap("multi", 8, 30, refine_max_clusters=32,
                             configured=16) == 16
    print("  [PASS] theta_cap 解析")

    parts = [[0, 1, 2, 3], [4, 5, 6]]
    new = split_cluster(parts, 0, [[0, 1], [2, 3]], 7, stage=3)
    validate_partition(new, 7, 3)
    assert new == [[0, 1], [2, 3], [4, 5, 6]]
    assert parts == [[0, 1, 2, 3], [4, 5, 6]]     # 不原地改
    try:
        split_cluster(parts, 0, [[0, 1], []], 7)
    except ValueError:
        pass
    else:
        raise AssertionError("空子簇应报错")
    print("  [PASS] split_cluster 覆盖/顺序/非空")

    sigs = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    assert cluster_dispersion(sigs, [0, 1]) > 0
    assert cluster_dispersion(sigs, [2]) == 0.0
    assert normalized_cluster_score(sigs, [0, 1], 0.0) == \
        cluster_dispersion(sigs, [0, 1])
    print("  [PASS] 簇离散度/归一化分数")

    assert not stall_trigger([10.0], 2, 1e-3, 1.0)
    assert stall_trigger([100.0, 100.0, 100.0], 2, 1e-3, 1.0)
    assert not stall_trigger([100.0, 100.0, 110.0], 2, 1e-3, 1.0)
    print("OK _t_refine_unit")


if __name__ == "__main__":
    main()
