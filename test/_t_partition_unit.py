# _t_partition_unit.py — 自适应分区纯函数单测（不跑 LP）。
#   python test/_t_partition_unit.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.partition import (aggregate_theta, cluster_weights, greedy_split,
                             initial_partition, median_pairwise_distance,
                             stall_trigger, validate_partition)

PASS = 0


def ok(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1
    print("  [PASS]", msg)


def main():
    global PASS
    # 合成池：场景 0..4，每条 Theta 形状 (2,2,3)
    rng = np.random.default_rng(7)
    pool = [(None, rng.dirichlet([1, 1, 1], size=(2, 2)).reshape(2, 2, 3) * 0
             + rng.dirichlet([1, 1, 1], size=(2, 2)).reshape(2, 2, 3))
            for _ in range(5)]
    parts = initial_partition(pool, 5, 2)
    validate_partition(parts, 5, 1)
    ok(sorted(s for c in parts for s in c) == list(range(5)), "initial_partition 覆盖全部场景")
    w = cluster_weights(parts, 5)
    ok(abs(w.sum() - 1.0) < 1e-12, "簇权重和为 1")
    th = aggregate_theta(pool, parts[0])
    ok(th.shape == (2, 2, 3), "aggregate_theta 形状正确")

    # 分裂规则：大签名差 + 小 Q 差 => 不分裂（退化保护）
    sigs = np.array([[0.0, 0.0], [10.0, 0.0]])
    qs = np.array([1.0, 1.0])
    out = greedy_split([0, 1], sigs, qs, 1.0, 0.5)
    ok(out == [[0, 1]], "大签名差但 Q 差 <= eps_obj 时合并")

    # 小签名差 + 大 Q 差 => 不分裂（贪心 OR 条件）
    out = greedy_split([0, 1], np.array([[0.0], [0.2]]), np.array([0.0, 10.0]),
                       1.0, 0.5)
    ok(out == [[0, 1]], "签名接近时即使 Q 差大也合并（OR 规则）")

    # 两者都大 => 分裂
    out = greedy_split([0, 1], np.array([[0.0], [10.0]]), np.array([0.0, 10.0]),
                       1.0, 0.5)
    ok(out == [[0], [1]], "签名和 Q 都大时分裂")

    # 代表固定为第一个加入者：场景 0 代表，1 加入，2 与 1 近但与 0 远 => 仍加入 0 簇
    sigs = np.array([[0.0], [5.0], [5.2]])
    qs = np.array([0.0, 0.0, 0.0])
    out = greedy_split([0, 1, 2], sigs, qs, 1.0, 0.5)
    ok(out == [[0, 1, 2]], "代表固定为第一个加入场景（非质心）")

    # 校验重复/漏场景
    for bad in ([[0, 1], [1, 2]], [[0, 1], []]):
        try:
            validate_partition(bad, 3, 1)
        except ValueError:
            PASS += 1
            print("  [PASS] 非法分区被拒绝")
        else:
            raise AssertionError(f"非法分区未拒绝: {bad}")

    ok(median_pairwise_distance(np.array([[0.0], [0.0], [0.0]])) == 0.0,
       "中位距离为 0 时返回 0")
    ok(stall_trigger([1.0, 1.0], 1, 0.0, 0.0), "停滞窗口触发")
    ok(not stall_trigger([1.0, 2.0], 1, 0.0, 0.0), "LB 提升时不触发")

    print(f"\n{PASS} 项通过")
    print("OK _t_partition_unit")


if __name__ == "__main__":
    main()
