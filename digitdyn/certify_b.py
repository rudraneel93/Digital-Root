"""Computer-assisted proof of eventual periodicity for all n < b^D.

Write n = H * b^L + x with 0 <= x < b^L and sigma = s_b(H). Then
s_b(n) = sigma + s_b(x) and (b even) n has the parity of x, so while H is
fixed the orbit follows the low map
    g(x) = x + sigma + s_b(x)  (x odd),   x - sigma - s_b(x)  (x even),
and H changes by +-1 exactly when g leaves [0, b^L) (steps are < b^L).

Certificate: for every sigma in [0, SIGMA_MAX], no x < E reaches an exit at
the top under g with that fixed sigma; E bounds every x that can occur right
after an upward exit (x' = y + step - b^L < step <= SIGMA_MAX + (b-1)L).

Consequence: after H increases it can never increase again before it has
decreased, so along every orbit H <= H_0 + 1, the orbit is bounded, hence
eventually periodic. For n < b^D we have H_0 < b^(D-L), so every H on the
orbit is <= b^(D-L), whose digit sum is at most SIGMA_MAX = (b-1)(D-L).

Usage: python3 -m digitdyn.certify_b <b> <D> <L>
"""
from __future__ import annotations

import sys
import time

import numpy as np


def digit_sums(M, b):
    x = np.arange(M)
    s = np.zeros(M, dtype=np.int64)
    while x.any():
        s += x % b
        x //= b
    return s


def up_fates(sigma, M, sx, rounds):
    """Boolean array: True where x reaches the top exit under g_sigma."""
    x = np.arange(M)
    step = sigma + sx
    g = np.where(x % 2 == 1, x + step, x - step)
    UP, DOWN = M, M + 1
    nxt = np.empty(M + 2, dtype=np.int64)
    nxt[:M] = np.where(g >= M, UP, np.where(g < 0, DOWN, g))
    nxt[UP], nxt[DOWN] = UP, DOWN
    for _ in range(rounds):
        nxt = nxt[nxt]
    return nxt[:M] == UP


def certify(b, D, L, verbose=True):
    assert b % 2 == 0
    M = b ** L
    SIGMA_MAX = (b - 1) * (D - L)
    E = SIGMA_MAX + (b - 1) * L + 1
    assert SIGMA_MAX + (b - 1) * L < M, "steps must be smaller than b^L"
    sx = digit_sums(M, b)
    rounds = int(np.ceil(np.log2(M + 2))) + 1
    bad = []
    t0 = time.time()
    for sigma in range(SIGMA_MAX + 1):
        up = up_fates(sigma, M, sx, rounds)
        hits = np.flatnonzero(up[:E])
        if hits.size:
            bad.append((sigma, int(hits[0]), int(hits.size)))
        if verbose and sigma % max(1, SIGMA_MAX // 8) == 0:
            print(f"  sigma={sigma}/{SIGMA_MAX}  bad so far={len(bad)}  [{time.time() - t0:.0f}s]",
                  flush=True)
    return bad, SIGMA_MAX, E


def main():
    b, D, L = (int(v) for v in sys.argv[1:4])
    print(f"base {b}: all n < {b}^{D}, blocks of {b}^{L} = {b ** L:,} states")
    bad, smax, E = certify(b, D, L)
    if not bad:
        print(f"CERTIFIED: for every sigma <= {smax}, no entry x < {E} exits at the top.")
        print(f"=> every orbit of f_{b} starting below {b}^{D} is bounded and eventually periodic.")
    else:
        print(f"NOT CERTIFIED: {len(bad)} values of sigma admit an exit at the top; first {bad[:8]}")


if __name__ == "__main__":
    main()
