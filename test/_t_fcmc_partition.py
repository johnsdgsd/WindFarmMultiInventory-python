# _t_fcmc_partition.py — 固定分区纯函数：覆盖/权重/端点/分位与 stride。
#   python test/_t_fcmc_partition.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.partition import (aggregate_theta, cluster_weights,  # noqa: E402
                             failure_order, fixed_partition,
                             validate_partition)


def _pool(n: int):
    # 失效强度随场景序号单调变化，便于验证分位块是连续切片
    return [(None, np.full((2, 3, 4), float(i))) for i in range(n)]


def main():
    n = 30
    pool = _pool(n)

    for L in (1, 4, 7, 16, 30):
        parts = fixed_partition(pool, n, L, "failure_quantile")
        validate_partition(parts, n)
        sizes = [len(p) for p in parts]
        assert len(parts) == L
        assert max(sizes) - min(sizes) <= 1, sizes
        w = cluster_weights(parts, n)
        assert abs(float(w.sum()) - 1.0) < 1e-12
        assert np.allclose(w, np.asarray(sizes) / n)
    print("  [PASS] 覆盖/互斥/非空、簇大小差≤1、权重=len/n")

    assert fixed_partition(pool, n, 1, "failure_quantile") == [list(range(n))]
    print("  [PASS] L=1 单簇全场景")

    parts_n = fixed_partition(pool, n, n, "failure_quantile")
    assert parts_n == [[i] for i in range(n)]
    assert fixed_partition(pool, n, 0, "scene") == parts_n
    print("  [PASS] L=n（以及引擎口径 0 → n）自然顺序单例")

    order = failure_order(pool)
    parts = fixed_partition(pool, n, 4, "failure_quantile")
    flat = [s for p in parts for s in p]
    assert flat == order.tolist(), "连续分位块必须保持失效强度顺序"
    print("  [PASS] failure_quantile 是排序后的连续块")

    stride = fixed_partition(pool, n, 4, "failure_stride")
    assert stride != parts
    assert all(len(p) in (7, 8) for p in stride)
    print("  [PASS] failure_stride 与分位块不同（消融口径）")

    th = aggregate_theta(pool, parts[0])
    assert np.allclose(th, np.mean([pool[i][1] for i in parts[0]], axis=0))
    th1 = aggregate_theta(pool, [7])
    assert np.array_equal(th1, pool[7][1])
    print("  [PASS] 聚合代表 = 簇内 Θ 均值；单例原样返回")

    for bad in (-1, n + 1):
        try:
            fixed_partition(pool, n, bad, "failure_quantile")
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_clusters={bad} 应报错")
    try:
        fixed_partition(pool, n, 4, "scene")
    except ValueError:
        pass
    else:
        raise AssertionError("L<n 时 group_mode=scene 应报错")
    print("  [PASS] 非法 L / group_mode 拦截")

    print("OK _t_fcmc_partition")


if __name__ == "__main__":
    main()
