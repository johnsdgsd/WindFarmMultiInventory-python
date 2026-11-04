# _t_payload_shm.py — 割载荷 shm/pickle 两种模式往返一致；shm nbytes 含 group。
#   python test/_t_payload_shm.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from utils.payload import StagePayload, attached, put_cuts  # noqa: E402
from utils.cut import Cut, trivial_cut                     # noqa: E402


def main():
    rng = np.random.default_rng(0)
    a = rng.normal(size=7)
    b = rng.normal(size=(7, 3))
    g = np.array([0, 1, 1, 2, 2, 2, 0], float)
    for use in (False, True):
        blk = StagePayload(a, b, g, use_shm=use)
        want = "shm" if use else "pickle"
        assert blk.ref["kind"] == want, (use, blk.ref["kind"])
        with attached(blk.ref) as got:
            assert got is not None
            aa, bb, gg = got
            assert np.array_equal(aa, a) and np.array_equal(bb, b)
            assert np.array_equal(gg, g)
        blk.close()
    print("  [PASS] pickle / shm 往返一致")

    cuts = [trivial_cut(1, np.zeros(3), 3, group=k) for k in range(3)]
    blk = put_cuts(cuts, 3, use_shm=True)
    assert blk.ref["kind"] == "shm"
    with attached(blk.ref) as got:
        aa, bb, gg = got
        assert aa.shape == (3,) and bb.shape == (3, 3)
        assert np.array_equal(gg, np.array([0, 1, 2], float))
    blk.close()
    print("  [PASS] put_cuts 携带 group 且 shm 不再回退")

    empty = put_cuts([], 3, use_shm=True)
    assert empty.ref["kind"] == "empty"
    empty.close()
    print("OK _t_payload_shm")


if __name__ == "__main__":
    main()
