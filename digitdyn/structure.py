"""Exact checks of the four structural facts behind the digit-sum parity map.

1. s(9m) = 9 * #{i : m_i > 9 * 0.m_{i-1}...m_0}  (= 9 * borrows(10m - m)).
2. Reversibility: if s(x) is even and s(f(x) + 9) == s(x), then
   f(f(x) + 9) == x + 9.
3. Level change: for a multiple of 9 with s = 9k, 1 <= k <= 10,
   s(n + 9k) = 9(k + 1 - carries(n, 9k)), s(n - 9k) = 9(k - 1 + borrows(n, 9k)).
4. Prefix lifting: cycles whose terms share one high part H (n = H*10^L + x)
   correspond exactly to cycles of the low map g_sigma(x) = x +- (sigma + s(x))
   that stay inside [0, 10^L), with sigma = s(H). Hence
     #{such cycles below 10^D} = sum_sigma c_L(sigma) * N(D-L, sigma),
   where c_L(sigma) counts low-map cycles and N(d, sigma) counts d-digit
   strings (leading zeros allowed) with digit sum sigma.
"""
from __future__ import annotations

from fractions import Fraction

from .explore import carries, f, s


def borrows(a, b):
    cnt = br = 0
    while a or b:
        d = a % 10 - b % 10 - br
        br = 1 if d < 0 else 0
        cnt += br
        a //= 10
        b //= 10
    return cnt


def k_formula(m):
    """#{i : m_i > 9 * 0.m_{i-1}...m_0} computed exactly with fractions."""
    digits = [int(c) for c in reversed(str(m))]
    r = Fraction(0)
    cnt = 0
    for d in digits:
        if d > 9 * r:
            cnt += 1
        r = (d + r) / 10
    return cnt


def check1(M):
    bad = [m for m in range(1, M) if s(9 * m) != 9 * k_formula(m) or s(9 * m) != 9 * borrows(10 * m, m)]
    return bad


def check2(M):
    tested = bad = 0
    for x in range(1, M):
        if s(x) % 2 == 0 and s(f(x) + 9) == s(x):
            tested += 1
            if f(f(x) + 9) != x + 9:
                bad += 1
    return tested, bad


def check3(M):
    bad = 0
    for n in range(9, M, 9):
        S = s(n)
        k = S // 9
        if 1 <= k <= 10:
            if s(n + S) != 9 * (k + 1 - carries(n, S)):
                bad += 1
            if n >= S and s(n - S) != 9 * (k - 1 + borrows(n, S)):
                bad += 1
    return bad


def low_cycles(sigma, L):
    """Cycles of x -> x + (sigma + s(x)) [x odd], x - (sigma + s(x)) [x even]
    that never leave [0, 10^L). Returned as frozensets."""
    M = 10 ** L
    state = [0] * M  # 0 unvisited, 1 on stack, 2 done
    out = []
    for x0 in range(M):
        if state[x0]:
            continue
        path, pos = [], {}
        x = x0
        while True:
            if x < 0 or x >= M:
                break  # exits: no cycle along this path
            if state[x] == 2:
                break
            if x in pos:
                out.append(frozenset(path[pos[x]:]))
                break
            pos[x] = len(path)
            path.append(x)
            st = sigma + s(x)
            x = x + st if x % 2 else x - st
        for y in path:
            state[y] = 2
    return out


def N_digits(d, sigma):
    """Number of d-digit strings (leading zeros allowed) with digit sum sigma."""
    dp = [1] + [0] * sigma
    for _ in range(d):
        nd = [0] * (sigma + 1)
        for t, w in enumerate(dp):
            if w:
                for a in range(10):
                    if t + a > sigma:
                        break
                    nd[t + a] += w
        dp = nd
    return dp[sigma]


def actual_constant_H_cycles(D, L):
    """Brute force: cycles with all terms < 10^D sharing one value of n // 10^L."""
    N = 10 ** D
    found = set()
    for n in range(1, N):
        x = n
        seen = set()
        while x not in seen and x > 0:
            seen.add(x)
            x = f(x)
        if x == 0:
            continue
        c = [x]
        y = f(x)
        while y != x:
            c.append(y)
            y = f(y)
        if max(c) < N and len({v // 10 ** L for v in c}) == 1:
            found.add(min(c))
    return len(found)


def main():
    print("1. s(9m) digit formula, m < 10^6: mismatches =", len(check1(10 ** 6)))
    t, b = check2(10 ** 6)
    print(f"2. reversibility, x < 10^6: {t} applicable cases, {b} failures")
    print("3. level-change rule, multiples of 9 < 10^6: failures =", check3(10 ** 6))
    L = 3
    for D in (5, 6):
        pred = 0
        for sigma in range(0, 9 * (D - L) + 1):
            c = len(low_cycles(sigma, L))
            if c:
                pred += c * N_digits(D - L, sigma)
        act = actual_constant_H_cycles(D, L)
        print(f"4. prefix lifting, D={D}, L={L}: predicted {pred}, actual {act}")


if __name__ == "__main__":
    main()
