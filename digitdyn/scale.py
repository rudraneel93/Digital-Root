"""Scale reduction for the block certificate.

Bad_L(sigma; e): some x in [0, e) is carried by the low map g_sigma on
[0, b^L) to the top exit.

Lemma (scale reduction). Let L > L' >= 1, and let sigma, e satisfy
sigma + (b-1) L < b^{L'} and e <= b^{L'}, with e > sigma + (b-1)L as well.
If Bad_L(sigma; e) then Bad_{L'}(sigma + j; e) for every j in
{0, 1, ..., (b-1)(L - L')}.

This module computes the bad set at a base scale L0 with a fixed entry window
e, and its longest run of consecutive bad values.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .certify_b import digit_sums, up_fates


def bad_set(b, L0, sigmas, e):
    M = b ** L0
    sx = digit_sums(M, b)
    rounds = int(np.ceil(np.log2(M + 2))) + 1
    out = np.zeros(len(sigmas), dtype=bool)
    for i, sg in enumerate(sigmas):
        if sg + (b - 1) * L0 >= M:
            out[i] = True  # outside the valid range: treat as bad (conservative)
            continue
        up = up_fates(int(sg), M, sx, rounds)
        out[i] = bool(up[:e].any())
    return out


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    b, L0, smax, e = (int(v) for v in sys.argv[1:5])
    t0 = time.time()
    sig = np.arange(smax + 1)
    mask = bad_set(b, L0, sig, e)
    bad = np.flatnonzero(mask)
    print(f"base {b}, scale L0={L0}, sigma <= {smax}, window e={e}: "
          f"{bad.size} bad values, longest run {longest_run(mask)} [{time.time() - t0:.0f}s]")
    if bad.size:
        print("  first bad:", bad[:10].tolist(), " last bad:", bad[-5:].tolist())
    np.save(f"digitdyn/data/bad_b{b}_L{L0}_s{smax}_e{e}.npy", mask)


if __name__ == "__main__":
    main()
