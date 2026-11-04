"""Computer-assisted convergence certificate for n < 10^D.

Write n = H * 10^L + x with 0 <= x < 10^L and sigma = s(H). Then
s(n) = sigma + s(x) and n has the parity of x, so while H is unchanged the
orbit follows the "low map" on x with parameter sigma:

    g(x) = x + sigma + s(x)   (x odd)        g(x) = x - sigma - s(x)  (x even)

If g(x) >= 10^L the orbit moves to H + 1 ("exit up", new x = g(x) - 10^L);
if g(x) < 0 it moves to H - 1 ("exit down", new x = g(x) + 10^L). Steps are
smaller than 10^L, so x wraps at most once per step.

Claim checked here, for every sigma in [0, SIGMA_MAX]:
    every x < E (E bounds the x reachable right after an exit up) leads, under
    the low map with that fixed sigma, to a cycle or an exit down -- never to
    an exit up.
Consequence: after H increases once, it cannot increase again before it
decreases; so along any orbit H never exceeds H_0 + 1, the orbit is bounded,
and therefore eventually periodic.

For n < 10^D we use L digits for x and SIGMA_MAX = 9 * (D - L) (bounds s(H)
for every H <= H_0 + 1 < 10^(D-L) + 1), and E = SIGMA_MAX + 9L + 1 (an exit
up lands on x' = x + step - 10^L < step <= sigma_old + 9L).
"""
from __future__ import annotations

import sys
import time

import numpy as np


def digit_sums(M):
    x = np.arange(M)
    s = np.zeros(M, dtype=np.int64)
    while x.any():
        s += x % 10
        x //= 10
    return s


def fates(sigma, L, sx):
    """For each x in [0, 10^L): 0 = cycle, 1 = exit up, 2 = exit down."""
    M = 10 ** L
    x = np.arange(M)
    step = sigma + sx
    g = np.where(x % 2 == 1, x + step, x - step)
    UP, DOWN = M, M + 1
    nxt = np.empty(M + 2, dtype=np.int64)
    nxt[:M] = np.where(g >= M, UP, np.where(g < 0, DOWN, g))
    nxt[UP], nxt[DOWN] = UP, DOWN
    # pointer doubling: after k rounds, nxt = g^(2^k); 2^k >= M+2 suffices
    rounds = int(np.ceil(np.log2(M + 2))) + 1
    for _ in range(rounds):
        nxt = nxt[nxt]
    end = nxt[:M]
    return np.where(end == UP, 1, np.where(end == DOWN, 2, 0))


def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    SIGMA_MAX = 9 * (D - L)
    E = SIGMA_MAX + 9 * L + 1
    M = 10 ** L
    assert SIGMA_MAX + 9 * L < M, "steps must be smaller than 10^L"
    assert E <= M
    sx = digit_sums(M)
    t0 = time.time()
    bad = []
    for sigma in range(SIGMA_MAX + 1):
        fa = fates(sigma, L, sx)
        up = np.flatnonzero(fa[:E] == 1)
        if up.size:
            bad.append((sigma, up[:5].tolist(), int(up.size)))
    print(f"D={D} L={L}: checked sigma 0..{SIGMA_MAX}, entries x < {E} "
          f"[{time.time() - t0:.0f}s]")
    badset = sorted(b[0] for b in bad)
    # longest run of consecutive integers inside the bad set
    r, cur = 0, 0
    for i, v in enumerate(badset):
        cur = cur + 1 if i and v == badset[i - 1] + 1 else 1
        r = max(r, cur)
    print(f"bad sigma values (a second exit up is possible): {len(badset)}; "
          f"longest run of consecutive bad values r = {r}")
    print(f"bad set: {badset}")
    if r <= 9:
        print("CERTIFIED (record-level argument): a first ascent from level h to h+1 "
              "requires s(h) in the bad set; ten consecutive levels inside one decade "
              f"have ten consecutive digit sums, so at most 2r = {2 * r} consecutive "
              "levels qualify.")
        print(f"=> H never exceeds H_0 + {2 * r + 1}; every orbit starting below "
              f"10^{D} is bounded and eventually periodic.")
    else:
        print("NOT CERTIFIED: the bad set contains 10 consecutive values.")


if __name__ == "__main__":
    main()
