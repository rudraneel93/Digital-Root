#!/usr/bin/env python3
"""Standalone checker for the computer-assisted part of the paper (Theorem certres).

It imports nothing from the digitdyn package, so a referee can check the
proof without trusting the exploratory code. It recomputes, for every
prefix digit sum sigma <= SMAX and every entry x0 < E = SMAX + (b-1)L + 1,
the highest point of the orbit of the half-line map

    g(x) = x + sigma + s_b(x)   (x odd),   x - sigma - s_b(x)   (x even),

and checks that it is below b^L. By Theorem cert of the paper this proves
that every orbit of f_b starting below b^D, with D = SMAX/(b-1) + L, is
eventually periodic.

Termination and memoisation. Each orbit is followed until it becomes negative
(it has left the block at the bottom) or reaches a state already seen (it has
entered a cycle, or joined an orbit whose maximum is already known). The
highest point of the forward orbit of a state depends only on that state, so
storing it (memoisation) is exact. If the program finishes, every listed
orbit has been followed completely.

Usage:
    python3 verify_certificates.py 10 19944 8      # all n < 10^2224 (~2 min on 8 cores)
    python3 verify_certificates.py 2 3000 24       # all n < 2^3024
    python3 verify_certificates.py 10 19944 8 --compare digitdyn/data/climb_profile_b10_S19944_L8.json
"""
import json
import sys
from multiprocessing import Pool, cpu_count


def digit_sum(x, b):
    total = 0
    while x:
        total += x % b
        x //= b
    return total


def highest_point(args):
    """Highest point over the orbits of all entries x0 < E for one sigma."""
    b, sigma, E = args
    best = {}                    # state -> highest point of its forward orbit
    worst = 0
    for x0 in range(E):
        path, on_path, x = [], {}, x0
        while x >= 0 and x not in best and x not in on_path:
            on_path[x] = len(path)
            path.append(x)
            step = sigma + digit_sum(x, b)
            x = x + step if x % 2 else x - step
        if x < 0:
            tail = -1
        elif x in best:
            tail = best[x]
        else:                    # a new cycle: path[on_path[x]:]
            start = on_path[x]
            cycle_max = max(path[start:])
            for y in path[start:]:
                best[y] = cycle_max
            path = path[:start]
            tail = cycle_max
        for y in reversed(path):
            tail = max(tail, y)
            best[y] = tail
        worst = max(worst, best[x0])
    return sigma, worst


def main():
    b, smax, L = (int(v) for v in sys.argv[1:4])
    if b % 2:
        sys.exit("the certificate applies to even bases")
    E = smax + (b - 1) * L + 1
    assert smax + (b - 1) * L < b ** L, "steps must be smaller than the block"
    with Pool(cpu_count()) as pool:
        heights = dict(pool.imap_unordered(highest_point, [(b, s, E) for s in range(smax + 1)], chunksize=4))
    top = max(heights.values())
    D = smax // (b - 1) + L
    print(f"base {b}, L = {L}, sigma <= {smax}, entries < {E}: highest point {top:,} (block {b ** L:,})")
    ok = top < b ** L
    if "--compare" in sys.argv:
        ref = json.load(open(sys.argv[sys.argv.index("--compare") + 1]))["heights"]
        same = all(heights[s] == ref[s] for s in range(smax + 1))
        print("matches the stored profile exactly" if same else "DIFFERS from the stored profile")
        ok = ok and same
    print(f"VERIFIED: every orbit of f_{b} starting below {b}^{D} is eventually periodic." if ok
          else "NOT VERIFIED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
