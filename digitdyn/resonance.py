"""Carry resonance: the probability that adding t to a random number gives
exactly s(t)/9 carries, via the carries transfer matrices, and the resulting
approximation of the 2-cycle density.

For a digit d of t and incoming carry c in {0,1}, a uniformly random digit of
a produces an outgoing carry with probability (d + c)/10 (d + c of the ten
digit values overflow). With z marking a carry, the transfer matrix is
    M(d) = [[ (10-d)/10,  d z/10     ],
            [ (9-d)/10,   (d+1) z/10 ]]       (rows: carry in 0/1; cols: out 0/1)
Above the top digit of t the chain continues with d = 0. Then
    rho(t) = [z^{s(t)/9}]  e_0^T  M(t_0) M(t_1) ... M(t_{l-1}) M(0)^K  1,
the probability of exactly s(t)/9 carries (K large).

Approximate 2-cycle density below 10^D (ignoring the weak dependence between
s(a) = t, the parity of a and the carries):
    T(D)/10^D  ~  sum_{t = 9 mod 18}  P_D(s(a) = t) * (1/2) * rho_odd(t),
where rho_odd uses an odd units digit for a (5 of 10 values).
"""
from __future__ import annotations

import json
import sys

import numpy as np


def poly_mul(p, q):
    return np.convolve(p, q)


def rho(t, tail=12, odd_units=True):
    """Probability of exactly s(t)/9 carries when adding t to a random number
    (a's units digit uniform over odd digits if odd_units)."""
    digits = [int(ch) for ch in reversed(str(t))] + [0] * tail
    need = sum(int(ch) for ch in str(t)) // 9
    # state vector: polynomials in z for carry 0 / carry 1
    v0, v1 = np.array([1.0]), np.array([0.0])
    for i, d in enumerate(digits):
        if i == 0 and odd_units:
            # units digit a0 in {1,3,5,7,9}: carry iff a0 + d >= 10
            n_carry = sum(1 for a0 in (1, 3, 5, 7, 9) if a0 + d >= 10)
            p_c = n_carry / 5
            n0 = v0 * (1 - p_c)
            n1 = np.concatenate([[0.0], v0 * p_c])
            v0, v1 = n0, n1
            continue
        a00 = (10 - d) / 10; a01 = d / 10
        a10 = (9 - d) / 10; a11 = (d + 1) / 10
        n0 = poly_add(v0 * a00, v1 * a10)
        n1 = np.concatenate([[0.0], poly_add(v0 * a01, v1 * a11)])
        v0, v1 = n0, n1
    tot = poly_add(v0, v1)
    return float(tot[need]) if need < len(tot) else 0.0


def poly_add(p, q):
    n = max(len(p), len(q))
    r = np.zeros(n)
    r[: len(p)] += p
    r[: len(q)] += q
    return r


def digit_sum_dist(d, smax):
    p = np.zeros(smax + 1)
    p[0] = 1.0
    u = np.full(10, 0.1)
    for _ in range(d):
        p = np.convolve(p, u)[: smax + 1]
    return p


def approx_density(D):
    smax = 9 * D
    P = digit_sum_dist(D, smax)
    tot = 0.0
    for t in range(9, smax + 1, 18):
        if P[t] > 1e-15:
            tot += P[t] * 0.5 * rho(t)
    return tot


def main():
    exact = {6: 0.018879, 8: 0.013812, 10: 0.013101, 12: 0.012434, 15: 0.013855,
             20: 0.019908, 30: 0.013760, 50: 0.013827, 75: 0.012426, 100: 0.011919,
             150: 0.012165, 200: 0.014819}
    print(" t     rho(t)")
    for t in (9, 27, 45, 63, 81, 99, 117, 459, 549, 999, 1899, 1989, 9999):
        print(f"{t:5d}  {rho(t):.4f}")
    print("\n D    exact     carry-resonance approximation")
    out = []
    for D in sorted(exact):
        a = approx_density(D)
        out.append((D, exact[D], a))
        print(f"{D:4d}  {exact[D]:.6f}  {a:.6f}")
    json.dump(out, open("digitdyn/data/resonance_vs_exact.json", "w"))


if __name__ == "__main__":
    main()
