"""Carry-count variance V(t) and the local-limit approximation of rho(t).

C(a,t) for a uniform odd a is an additive functional of the two-state carry
chain with P(carry out | carry in c, digit d) = (d+c)/10. We compute its
exact pmf (as in resonance.rho), its mean and variance, and compare rho(t)
with the Gaussian local value exp(-(m-mu)^2/2V)/sqrt(2 pi V).
"""
from __future__ import annotations

import math
import sys

import numpy as np


def carry_pmf(t, tail=14):
    digits = [int(ch) for ch in reversed(str(t))] + [0] * tail
    v0, v1 = np.array([1.0]), np.array([0.0])
    for i, d in enumerate(digits):
        if i == 0:
            pc = sum(1 for a0 in (1, 3, 5, 7, 9) if a0 + d >= 10) / 5
            v0, v1 = v0 * (1 - pc), np.concatenate([[0.0], v0 * pc])
            continue
        n0 = np.zeros(max(len(v0), len(v1)))
        n0[:len(v0)] += v0 * (10 - d) / 10
        n0[:len(v1)] += v1 * (9 - d) / 10
        n1 = np.zeros(len(n0) + 1)
        n1[1:len(v0) + 1] += v0 * d / 10
        n1[1:len(v1) + 1] += v1 * (d + 1) / 10
        v0, v1 = n0, n1
    tot = np.zeros(max(len(v0), len(v1)))
    tot[:len(v0)] += v0
    tot[:len(v1)] += v1
    return tot


def stats(t):
    p = carry_pmf(t)
    k = np.arange(len(p))
    mu = float((k * p).sum())
    V = float(((k - mu) ** 2 * p).sum())
    m = sum(map(int, str(t))) // 9
    rho = float(p[m]) if m < len(p) else 0.0
    gauss = math.exp(-(m - mu) ** 2 / (2 * V)) / math.sqrt(2 * math.pi * V) if V > 0 else float('nan')
    return mu, V, m, rho, gauss


def window_average(D, fn, width=4.0):
    """Average fn(t) over t = 9 mod 18 with the Gaussian law of s(a), a < 10^D."""
    mean, sd = 4.5 * D, math.sqrt(8.25 * D)
    lo, hi = int(mean - width * sd), int(mean + width * sd)
    num = den = 0.0
    for t in range(lo - lo % 18 + 9, hi, 18):
        w = math.exp(-(t - mean) ** 2 / (2 * sd * sd))
        num += w * fn(t)
        den += w
    return num / den


if __name__ == "__main__":
    for t in [9, 99, 999999, 1000000 + 9 * 7, 123456789, 55555 * 18 + 9, 909090909]:
        mu, V, m, r, g = stats(t)
        print(f"t={t:>11}: mean={mu:7.3f} target={m:3d} V={V:7.4f} rho={r:.5f} gauss={g:.5f}")
