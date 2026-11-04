"""Digit-sum parity map f(n) = n + s(n) (n odd), n - s(n) (n even).

Checks, exhaustively for 1 <= n <= N:
  1. every orbit reaches a cycle (or 0), with the maximum excursion
     max(orbit) - n, to guide a boundedness proof;
  2. the carry characterisation of 2-cycles: a in a 2-cycle (a odd) iff
     a % 9 == 0, s(a) odd, and carries(a + s(a)) == s(s(a)) / 9;
  3. the two-step identity f(f(a)) - a = 9 * (carries(a + s(a)) - 1) - (s(s(a)) - 9)
     for odd a with s(a) odd (i.e. f(f(a)) = a - s(s(a)) + 9 c).
"""
from __future__ import annotations

import sys
from collections import Counter


def s(n):
    t = 0
    while n:
        t += n % 10
        n //= 10
    return t


def f(n):
    return n + s(n) if n & 1 else n - s(n)


def carries(a, b):
    c = cnt = 0
    while a or b or c:
        d = a % 10 + b % 10 + c
        c = 1 if d >= 10 else 0
        cnt += c
        a //= 10
        b //= 10
    return cnt


def main(N):
    cycle_of = {}          # n -> canonical cycle (min element, length)
    max_exc = (0, 0)
    lengths = Counter()
    for n in range(1, N + 1):
        seen = {}
        x, i = n, 0
        top = n
        while x not in seen and x not in cycle_of:
            seen[x] = i
            x = f(x)
            i += 1
            top = max(top, x)
            if x == 0:
                break
        if x == 0:
            cyc = (0, 1)
        elif x in cycle_of:
            cyc = cycle_of[x]
        else:  # new cycle found
            c = [x]
            y = f(x)
            while y != x:
                c.append(y)
                y = f(y)
            cyc = (min(c), len(c))
            for y in c:
                cycle_of[y] = cyc
            lengths[len(c)] += 1
        for y in seen:
            cycle_of.setdefault(y, cyc)
        if top - n > max_exc[0]:
            max_exc = (top - n, n)
    print(f"N={N}: all orbits end in a cycle or 0")
    print(f"  distinct cycles by length: {dict(sorted(lengths.items()))}")
    print(f"  max excursion max(orbit)-n = {max_exc[0]} (start n={max_exc[1]})")

    # 2-cycle characterisation, exhaustive over odd a <= N
    bad = 0
    two = 0
    for a in range(1, N + 1, 2):
        b = f(a)
        in2 = (b % 2 == 0) and f(b) == a and b != a
        pred = (a % 9 == 0 and s(a) % 2 == 1 and carries(a, s(a)) * 9 == s(s(a)))
        two += in2
        bad += in2 != pred
    print(f"  2-cycles with odd point <= N: {two}; characterisation mismatches: {bad}")

    # two-step identity
    bad2 = 0
    for a in range(1, N + 1, 2):
        if s(a) % 2 == 1:
            if f(f(a)) != a - s(s(a)) + 9 * carries(a, s(a)):
                bad2 += 1
    print(f"  two-step identity f(f(a)) = a - s(s(a)) + 9*carries(a, s(a)): mismatches {bad2}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6)
