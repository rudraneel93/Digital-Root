"""Exact number of 2-cycles below 10^D via digit dynamic programming.

By the 2-cycle theorem, an odd a lies on a 2-cycle iff t = s(a) is odd and
the addition a + t has exactly s(t)/9 carries (this forces 9 | t). For each
admissible t (odd multiple of 9, t <= 9D) we count D-digit strings a
(leading zeros allowed, a odd) with digit sum t and exactly s(t)/9 carries
when adding t, processing digits from least significant upward with state
(carry, digit sum so far, carries so far). Carries beyond the needed count
are pruned.

Output: T(D) = number of odd a < 10^D on a 2-cycle, and T(D)/10^D.
Cross-checked against brute force for small D.
"""
from __future__ import annotations

import sys
from fractions import Fraction


def s(n):
    return sum(map(int, str(n)))


def count_for_t(D, t):
    need = s(t) // 9
    tdig = [int(c) for c in reversed(str(t))]
    # dp: dict (carry, digit_sum, carries) -> count
    dp = {(0, 0, 0): 1}
    for i in range(D):
        ti = tdig[i] if i < len(tdig) else 0
        nd = {}
        for (c, ds, cc), w in dp.items():
            for a in range(10):
                if i == 0 and a % 2 == 0:
                    continue
                nds = ds + a
                if nds > t:
                    break
                tot = a + ti + c
                nc = 1 if tot >= 10 else 0
                ncc = cc + nc
                if ncc > need:
                    continue
                key = (nc, nds, ncc)
                nd[key] = nd.get(key, 0) + w
        dp = nd
    total = 0
    for (c, ds, cc), w in dp.items():
        # digits of t beyond D can't occur since t <= 9D < 10^D for D >= 2
        if ds == t and cc == need:
            total += w
    return total


def T(D):
    return sum(count_for_t(D, t) for t in range(9, 9 * D + 1, 18))


def brute(D):
    N = 10 ** D
    cnt = 0
    for a in range(1, N, 2):
        t = s(a)
        if t % 2 == 1:
            b = a + t
            if s(b) == t:
                cnt += 1
    return cnt


def main():
    for D in range(2, 6):
        assert T(D) == brute(D), (D, T(D), brute(D))
    print("DP matches brute force for D = 2..5")
    Ds = [int(x) for x in sys.argv[1:]] or [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200]
    for D in Ds:
        v = T(D)
        print(f"D={D:4d}  T={v}  T/10^D={v / 10 ** D:.6f}", flush=True)


if __name__ == "__main__":
    main()
