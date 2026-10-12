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
from multiprocessing import shared_memory

import numpy as np


class StagePayload:
    """一个阶段的割集 (alpha (n,), beta (n, dim))。"""

    def __init__(self, alpha, beta):
        a = np.ascontiguousarray(alpha, dtype=float)
        b = np.ascontiguousarray(beta, dtype=float)
        self._arrays = [a, b]
        self._shm = None
        nbytes = int(a.nbytes + b.nbytes)
        if nbytes == 0:
            self.ref = {"kind": "empty"}
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
    """worker 侧：映射一块割集，用完即关。返回 (alpha, beta) 或 None。

    None 表示"没有割集"——调用方必须把 None 与"空割集"区分开：
    空割集会让 theta 进目标却无约束，LP 无界。
    """
    if ref is None or ref.get("kind") == "empty":
        yield None
        return
    if ref.get("kind") == "pickle":
        a, b = ref["arrays"]
        yield (a, b)
        return
    shm = shared_memory.SharedMemory(name=ref["name"])
    try:
        arrs, off = [], 0
        for shape in ref["shapes"]:
            arr = np.ndarray(shape, dtype=float, buffer=shm.buf, offset=off)
            arrs.append(arr)
            off += arr.nbytes
        yield (arrs[0], arrs[1])
    finally:
        with contextlib.suppress(Exception):
            shm.close()


def put_cuts(cuts, dim: int) -> StagePayload:
    """[Cut] -> StagePayload。"""
    if not cuts:
        return StagePayload(np.zeros(0), np.zeros((0, dim)))
    return StagePayload(np.array([c.alpha for c in cuts], dtype=float),
                        np.vstack([c.beta for c in cuts]).astype(float))
