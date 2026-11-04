"""Cycle profile c_L(sigma) of the low map and the prefix-lifting predictions.

Low map on x in [0, 10^L) with parameter sigma (= digit sum of the prefix H):
    g(x) = x + sigma + s(x)  (x odd),   g(x) = x - sigma - s(x)  (x even);
leaving [0, 10^L) means the prefix changes. c_L(sigma) = number of cycles of g
inside [0, 10^L); c2_L(sigma) = number of those of length 2.

Prefix-lifting theorem: cycles of f below 10^D whose terms share one prefix
H < 10^(D-L) are counted exactly by sum_sigma c_L(sigma) * N(D-L, sigma), where
N(d, sigma) = #{d-digit strings with digit sum sigma}. Dividing by 10^D gives a
density = sum_sigma (c_L(sigma)/10^L) * P_{D-L}(sigma), P the digit-sum
distribution of a uniform random (D-L)-digit string.

Outputs digitdyn/data/profile_L{L}.npz and prediction tables.
"""
from __future__ import annotations

import json
import os
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


def cycles_for_sigma(sigma, M, sx, rounds):
    x = np.arange(M)
    step = sigma + sx
    g = np.where(x % 2 == 1, x + step, x - step)
    OUT = M
    nxt = np.empty(M + 1, dtype=np.int64)
    nxt[:M] = np.where((g >= M) | (g < 0), OUT, g)
    nxt[OUT] = OUT
    # 2-cycles: odd x, g(x) inside and even, g(g(x)) == x, g(x) != x
    gx = nxt[:M]
    inside = gx < M
    gg = np.where(inside, nxt[np.minimum(gx, M)], OUT)
    two = int(np.count_nonzero((x % 2 == 1) & inside & (gg == x) & (gx != x)))
    # all cycles: cyclic nodes are the image of g^(2^r) for 2^r >= M
    p = nxt.copy()
    for _ in range(rounds):
        p = p[p]
    cyc_nodes = np.unique(p[:M])
    cyc_nodes = cyc_nodes[cyc_nodes < M]
    if cyc_nodes.size == 0:
        return 0, two, 0
    # label each cyclic node by the minimum of its cycle (min-propagation by doubling)
    lab = np.full(M + 1, np.iinfo(np.int64).max)
    lab[cyc_nodes] = cyc_nodes
    q = nxt.copy()
    for _ in range(rounds):
        lab = np.minimum(lab, lab[q])
        q = q[q]
    n_cycles = int(np.unique(lab[cyc_nodes]).size)
    return n_cycles, two, int(cyc_nodes.size)


def digit_sum_dist(d, smax):
    """P(digit sum = sigma) for a uniform random d-digit string, float64."""
    p = np.zeros(smax + 1)
    p[0] = 1.0
    unif = np.full(10, 0.1)
    for _ in range(d):
        p = np.convolve(p, unif)[: smax + 1]
    return p


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    SMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    M = 10 ** L
    rounds = int(np.ceil(np.log2(M + 1))) + 1
    sx = digit_sums(M)
    c = np.zeros(SMAX + 1, dtype=np.int64)
    c2 = np.zeros(SMAX + 1, dtype=np.int64)
    nodes = np.zeros(SMAX + 1, dtype=np.int64)
    t0 = time.time()
    for sigma in range(SMAX + 1):
        c[sigma], c2[sigma], nodes[sigma] = cycles_for_sigma(sigma, M, sx, rounds)
        if sigma % 250 == 0:
            print(f"  sigma={sigma} cycles={c[sigma]} two={c2[sigma]} [{time.time() - t0:.0f}s]",
                  flush=True)
    os.makedirs("digitdyn/data", exist_ok=True)
    np.savez(f"digitdyn/data/profile_L{L}.npz", c=c, c2=c2, nodes=nodes, L=L)
    # predictions
    rows = []
    for D in list(range(L + 1, 60)) + list(range(60, 401, 10)):
        d = D - L
        P = digit_sum_dist(d, SMAX)
        mass = P.sum()  # how much of the distribution is covered by sigma <= SMAX
        dens_all = float((c * P).sum() / M)
        dens_two = float((c2 * P).sum() / M)
        rows.append(dict(D=D, density_cycles=dens_all, density_2cycles=dens_two,
                         mass_covered=float(mass)))
    json.dump(rows, open(f"digitdyn/data/predictions_L{L}.json", "w"), indent=1)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
