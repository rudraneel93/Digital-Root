"""The digit-sum parity map in base b:
    f_b(n) = n + s_b(n) (n odd),   n - s_b(n) (n even).

Checks, for several bases b:
  * odd b: every orbit reaches 0 (no cycles other than {0});
  * even b: 2-cycle characterisation s_b(s_b(a)) = (b-1) C_b(a, s_b(a)),
    carry balance E C_K = (s_b(t) - t b^-K)/(b-1), the general level equation,
    and conditional reversibility with shift b-1;
  * even b: exact 2-cycle counts T_b(D) by digit DP.
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction


def digits(n, b):
    out = []
    while n:
        out.append(n % b)
        n //= b
    return out


def s(n, b):
    t = 0
    while n:
        t += n % b
        n //= b
    return t


def f(n, b):
    return n + s(n, b) if n & 1 else n - s(n, b)


def carries(x, y, b):
    c = cnt = 0
    while x or y or c:
        d = x % b + y % b + c
        c = 1 if d >= b else 0
        cnt += c
        x //= b
        y //= b
    return cnt


def borrows(x, y, b):
    br = cnt = 0
    while x or y:
        d = x % b - y % b - br
        br = 1 if d < 0 else 0
        cnt += br
        x //= b
        y //= b
    return cnt


def check_odd_base(b, N):
    for n in range(1, N):
        x, seen = n, set()
        while x and x not in seen:
            seen.add(x)
            x = f(x, b)
        if x != 0:
            return n
    return None


def check_two_cycle(b, N):
    bad = 0
    for a in range(1, N, 2):
        y = f(a, b)
        on2 = y % 2 == 0 and f(y, b) == a and y != a
        t = s(a, b)
        pred = t % 2 == 1 and s(t, b) == (b - 1) * carries(a, t, b)
        bad += on2 != pred
    return bad


def check_balance(b, K=None):
    K = K or (5 if b <= 4 else 3)
    bad = 0
    for t in [1, 3, 5, b - 1, b + 1, 2 * b - 1, b * b - 1, 3 * b + 1]:
        if t >= b ** K:
            continue
        tot = 0
        for a in range(b ** K):
            c = cc = 0
            for i in range(K):
                d = (a // b ** i) % b + (t // b ** i) % b + c
                c = 1 if d >= b else 0
                cc += c
            tot += cc
        E = Fraction(tot, b ** K)
        bad += E != (Fraction(s(t, b)) - Fraction(t, b ** K)) / (b - 1)
    return bad


def check_level_and_reverse(b, trials=3000):
    rng = random.Random(b)
    bad = 0
    for _ in range(trials):
        n = rng.randrange(b ** rng.randint(2, 30))
        n -= n % (b - 1)
        if n == 0:
            continue
        S = s(n, b)
        k = S // (b - 1)
        kap = s((b - 1) * k, b) // (b - 1)
        if s(n + S, b) != (b - 1) * (k + kap - carries(n, S, b)):
            bad += 1
        if n >= S and s(n - S, b) != (b - 1) * (k - kap + borrows(n, S, b)):
            bad += 1
    rev_bad = rev_n = 0
    for x in range(1, 200000):
        if s(x, b) % 2 == 0 and s(f(x, b) + b - 1, b) == s(x, b):
            rev_n += 1
            rev_bad += f(f(x, b) + b - 1, b) != x + b - 1
    return bad, rev_n, rev_bad


def T(D, b):
    """Odd a < b^D on a 2-cycle: s_b(a) = t odd, carries(a, t) = s_b(t)/(b-1)."""
    total = 0
    for t in range(1, (b - 1) * D + 1, 2):
        st = s(t, b)
        if st % (b - 1):
            continue
        need = st // (b - 1)
        tdig = digits(t, b)
        dp = {(0, 0, 0): 1}
        for i in range(D):
            ti = tdig[i] if i < len(tdig) else 0
            nd = {}
            for (c, ds, cc), w in dp.items():
                for a in range(b):
                    if i == 0 and a % 2 == 0:
                        continue
                    nds = ds + a
                    if nds > t:
                        break
                    nc = 1 if a + ti + c >= b else 0
                    ncc = cc + nc
                    if ncc > need:
                        continue
                    key = (nc, nds, ncc)
                    nd[key] = nd.get(key, 0) + w
            dp = nd
        total += sum(w for (c, ds, cc), w in dp.items() if ds == t and cc == need)
    return total


def T_brute(D, b):
    cnt = 0
    for a in range(1, b ** D, 2):
        y = f(a, b)
        cnt += y % 2 == 0 and f(y, b) == a and y != a
    return cnt


def main():
    print("odd bases: first n whose orbit does NOT reach 0 (None = all reach 0)")
    for b in (3, 5, 7, 9, 11):
        print(f"  b={b}: {check_odd_base(b, 3000)}", flush=True)
    print("\neven bases: exact checks")
    for b in (2, 4, 6, 8, 10, 16):
        N = min(200000, b ** 7)
        print(f"  b={b:2d}: 2-cycle characterisation mismatches {check_two_cycle(b, N)}; "
              f"carry-balance mismatches {check_balance(b)}; "
              f"level/reversal {check_level_and_reverse(b)}", flush=True)
    print("\nDP vs brute force for T_b(D)")
    for b in (2, 4, 6):
        for D in range(2, 7 if b == 2 else 5):
            if b ** D > 3 * 10 ** 5:
                continue
            assert T(D, b) == T_brute(D, b), (b, D)
    print("  DP agrees with brute force")


if __name__ == "__main__":
    main()
