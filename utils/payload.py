# payload.py — 割集跨进程传递。与问题解耦，可整目录迁移。
"""为什么走共享内存：一个阶段的割集 = |cuts| × (1+dim) 个 float64。按任务 pickle 的话
后向每阶段 N 个任务各带一份 ⇒ IPC 量是割集本身的 N 倍（默认进程数 10 ⇒ 每阶段 ~100 MB、
每轮 20 GB），必然成为瓶颈。走 shm 后主管程每阶段只写一次。

**粒度是一个阶段一块**，不是一轮一块：后向只用到 cuts[t+1]（9.8 MB 量级），一轮一块
会在每次换阶段时重写整张表，总量退化成 O(T²)。前向确实要跨所有阶段，由主管程为每阶段
各建一块、整轮持有，map 结束后统一释放。

worker 在任务内映射、任务结束即关，不跨任务持有；主管程在 map 返回后才 unlink
（此刻 worker 的映射都已关闭，Windows 上 unlink 才是安全的）。

shm 不可用时回退为任务内直接 pickle。
"""

from __future__ import annotations

import contextlib
from collections import namedtuple
from multiprocessing import shared_memory

import numpy as np

# 每个阶段子问题的 theta 规格：只用前 active_k 个 theta；weights 是未来阶段
# 分区 P_{t+1} 的权重；layout_gen/cut_gen 变更时强制重新拼装割行。
ThetaSpec = namedtuple("ThetaSpec", "active_k weights layout_gen cut_gen")


class StagePayload:
    """一个阶段的割集 (alpha (n,), beta (n, dim), group (n,))。

    use_shm=False 时直接 pickle 传数组。B 实验默认 pickle，与固定 A 的历史口径
    一致；shm 只作独立性能对照。修复过的 nbytes 必须包含 group 数组，否则三次
    写入会越界、被 except 吞掉后静默退回 pickle。
    """

    def __init__(self, alpha, beta, groups=None, use_shm: bool = False):
        a = np.ascontiguousarray(alpha, dtype=float)
        b = np.ascontiguousarray(beta, dtype=float)
        g = (np.zeros(a.size, dtype=float) if groups is None
             else np.ascontiguousarray(groups, dtype=float))
        if g.size != a.size:
            raise ValueError(f"group 数 {g.size} ≠ 割数 {a.size}")
        self._arrays = [a, b, g]
        self._shm = None
        nbytes = int(sum(arr.nbytes for arr in self._arrays))
        if nbytes == 0:
            self.ref = {"kind": "empty"}
        elif not use_shm:
            self.ref = {"kind": "pickle", "arrays": self._arrays}
        else:
            try:
                self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
                off = 0
                for arr in self._arrays:
                    if arr.nbytes:
                        v = np.ndarray(arr.shape, dtype=arr.dtype,
                                       buffer=self._shm.buf, offset=off)
                        v[...] = arr
                        off += arr.nbytes
                self.ref = {"kind": "shm", "name": self._shm.name,
                            "shapes": [arr.shape for arr in self._arrays]}
            except Exception:
                self.close()
                self.ref = {"kind": "pickle", "arrays": self._arrays}

    @property
    def n_cuts(self) -> int:
        return int(self._arrays[0].size)

    def close(self) -> None:
        """主管程在 map 结束后调。"""
        if self._shm is not None:
            with contextlib.suppress(Exception):
                self._shm.close()
            with contextlib.suppress(Exception):
                self._shm.unlink()
            self._shm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@contextlib.contextmanager
def attached(ref):
    """worker 侧：映射一块割集，用完即关。返回 (alpha,beta,groups) 或 None。

    None 表示"没有割集"——调用方必须把 None 与"空割集"区分开：
    空割集会让 theta 进目标却无约束，LP 无界。
    """
    if ref is None or ref.get("kind") == "empty":
        yield None
        return
    if ref.get("kind") == "pickle":
        a, b, g = ref["arrays"]
        yield (a, b, g)
        return
    shm = shared_memory.SharedMemory(name=ref["name"])
    try:
        arrs, off = [], 0
        for shape in ref["shapes"]:
            arr = np.ndarray(shape, dtype=float, buffer=shm.buf, offset=off)
            arrs.append(arr)
            off += arr.nbytes
        yield (arrs[0], arrs[1], arrs[2])
    finally:
        with contextlib.suppress(Exception):
            shm.close()


def put_cuts(cuts, dim: int, use_shm: bool = False) -> StagePayload:
    """[Cut] -> StagePayload（含 group 数组）。"""
    if not cuts:
        return StagePayload(np.zeros(0), np.zeros((0, dim)),
                            np.zeros(0, dtype=float), use_shm=use_shm)
    return StagePayload(np.array([c.alpha for c in cuts], dtype=float),
                        np.vstack([c.beta for c in cuts]).astype(float),
                        np.array([c.group for c in cuts], dtype=float),
                        use_shm=use_shm)
