"""Tests for every computable claim in the manuscript.

Run from the repository root:   python3 -m pytest digitdyn/tests -q
Each test checks a theorem or computational statement against brute force.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from digitdyn.count2 import T, brute
from digitdyn.explore import carries, f, s
from digitdyn.resonance import rho
from digitdyn.structure import (N_digits, borrows, k_formula, low_cycles)


def _cycle_of(n):
    seen = set()
    x = n
    while x not in seen:
        seen.add(x)
        x = f(x)
    c = [x]
    y = f(x)
    while y != x:
        c.append(y)
        y = f(y)
    return c


# Lemma 2.1: carries and borrows
def test_carry_borrow_lemma():
    rng = random.Random(1)
    for _ in range(5000):
        a, b = rng.randrange(10 ** 12), rng.randrange(10 ** 12)
        assert s(a + b) == s(a) + s(b) - 9 * carries(a, b)
        hi, lo = max(a, b), min(a, b)
        assert s(hi - lo) == s(hi) - s(lo) + 9 * borrows(hi, lo)


# Proposition 2.3: worst case to reach a multiple of 9 below 10^6 is 14 steps
def test_reduction_to_nine_counterexample():
    orbit = [11]
    while orbit[-1] % 9:
        orbit.append(f(orbit[-1]))
    assert orbit == [11, 13, 17, 25, 32, 27]
    x, k = 899457, 0
    while x % 9:
        x = f(x)
        k += 1
    assert k == 14


# Theorem 3.1: s(9m) = 9 * borrows(10m, m) = 9 * digit formula
def test_self_similarity_formula():
    for m in range(1, 200000):
        assert s(9 * m) == 9 * borrows(10 * m, m) == 9 * k_formula(m)


# Theorem 4.2: general level equation, including levels k > 10
def test_level_equation_general():
    rng = random.Random(0)
    big = 0
    for _ in range(4000):
        D = rng.randint(2, 60)
        n = rng.randrange(10 ** (D - 1), 10 ** D)
        n -= n % 9
        if n == 0:
            continue
        S = s(n)
        k = S // 9
        kap = s(9 * k) // 9
        big += k > 10
        assert s(n + S) == 9 * (k + kap - carries(n, S))
        if n >= S:
            assert s(n - S) == 9 * (k - kap + borrows(n, S))
    assert big > 1000


# Theorem 5.1: conditional reversibility
def test_conditional_reversibility():
    n_cases = 0
    for x in range(1, 300000):
        if s(x) % 2 == 0 and s(f(x) + 9) == s(x):
            n_cases += 1
            assert f(f(x) + 9) == x + 9
    assert n_cases > 5000


# Theorem 6.2: characterisation of 2-cycles (brute force below 10^6)
def test_two_cycle_characterisation():
    for a in range(1, 10 ** 6, 2):
        b = f(a)
        on_two_cycle = b % 2 == 0 and f(b) == a and b != a
        t = s(a)
        pred = t % 2 == 1 and s(t) == 9 * carries(a, t)
        assert on_two_cycle == pred


# Section 6: exact DP for T(D) agrees with enumeration
@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_dp_matches_bruteforce(D):
    assert T(D) == brute(D)


def test_dp_published_values():
    assert T(6) == 18879
    assert T(8) == 1381234


# Theorem 7.1: prefix lifting counts constant-prefix cycles exactly (L = 3)
def test_prefix_lifting_exact():
    L = 3
    for D in (4, 5):
        pred = sum(len(low_cycles(sig, L)) * N_digits(D - L, sig)
                   for sig in range(0, 9 * (D - L) + 1))
        N = 10 ** D
        found = {0}  # the fixed point 0 has prefix 0
        for n in range(1, N):
            c = _cycle_of(n)
            if max(c) < N and len({v // 10 ** L for v in c}) == 1:
                found.add(min(c))
        assert pred == len(found)


# Corollary 7.2: cross-prefix cycles are at most 18 D 10^(D-L)
def test_cross_prefix_bound():
    D, L = 5, 3
    N = 10 ** D
    all_cycles, crossing = set(), set()
    for n in range(0, N):
        c = _cycle_of(n)
        if max(c) < N:
            key = min(c)
            all_cycles.add(key)
            if len({v // 10 ** L for v in c}) > 1:
                crossing.add(key)
    assert len(crossing) <= 18 * D * 10 ** (D - L)


# Proposition 8.1: rho(10^j - 1) = 9/10
@pytest.mark.parametrize("j", [1, 2, 3, 4, 5])
def test_resonance_peak(j):
    assert abs(rho(10 ** j - 1) - 0.9) < 1e-12


# Theorem 10.1: carry balance, exact, both versions
@pytest.mark.parametrize("t", [9, 27, 45, 99, 117, 459, 999, 1233, 5679])
def test_carry_balance(t):
    K = 4

    def ck(a):
        c = cnt = 0
        for i in range(K):
            d = (a // 10 ** i) % 10 + (t // 10 ** i) % 10 + c
            c = 1 if d >= 10 else 0
            cnt += c
        return cnt

    allv = range(10 ** K)
    E_all = Fraction(sum(ck(a) for a in allv), len(allv))
    assert E_all == (Fraction(s(t)) - Fraction(t, 10 ** K)) / 9
    if t % 2 == 1:
        odds = range(1, 10 ** K, 2)
        E_odd = Fraction(sum(ck(a) for a in odds), len(odds))
        assert E_odd == (Fraction(s(t)) + 1 - Fraction(t + 1, 10 ** K)) / 9


# Table 7 (corrected statistics) and Proposition 5.2 (census) spot checks
def test_corrected_table_rows():
    def run(N):
        cyc = set()
        for n in range(9, N + 1, 9):
            cyc.add(min(_cycle_of(n)))
        return len(cyc)
    assert run(500) == 20
    assert run(10000) == 171


# Section 'Other bases': odd bases collapse to 0
@pytest.mark.parametrize("b", [3, 5, 7, 9, 11])
def test_odd_bases_reach_zero(b):
    from digitdyn.baseb import check_odd_base
    assert check_odd_base(b, 2000) is None


# Even bases: 2-cycle characterisation, carry balance, level eq., reversal
@pytest.mark.parametrize("b", [2, 4, 6, 8, 16])
def test_even_base_structure(b):
    from digitdyn.baseb import check_balance, check_level_and_reverse, check_two_cycle
    assert check_two_cycle(b, min(100000, b ** 7)) == 0
    assert check_balance(b) == 0
    lvl_bad, rev_n, rev_bad = check_level_and_reverse(b, trials=1000)
    assert lvl_bad == 0 and rev_bad == 0 and rev_n > 1000


@pytest.mark.parametrize("b,D", [(2, 4), (2, 6), (4, 3), (4, 4), (6, 3)])
def test_base_dp_matches_bruteforce(b, D):
    from digitdyn.baseb import T as Tb, T_brute
    assert Tb(D, b) == T_brute(D, b)


# Binary switch lemma: reset/block make a switch bit control exactly one carry
def test_binary_switch_mechanism():
    rng = random.Random(3)
    from digitdyn.baseb import carries as cb
    for _ in range(3000):
        L = 24
        t = rng.randrange(1, 2 ** L)
        tb = [(t >> i) & 1 for i in range(L)]
        cand = [i for i in range(2, L - 2, 4) if tb[i - 1] != tb[i]]
        if not cand:
            continue
        i = rng.choice(cand)
        a = rng.randrange(2 ** L)
        a = a & ~((1 << (i - 1)) | (1 << (i + 1)))          # set reset and block bits
        a |= (tb[i - 1] << (i - 1)) | (tb[i + 1] << (i + 1))
        a0, a1 = a & ~(1 << i), a | (1 << i)
        # flipping the switch bit changes the carry count by exactly 1
        assert cb(a1, t, 2) - cb(a0, t, 2) == 1


# Constant-level runs are short (Theorem 'shortruns')
def test_constant_level_runs_bounded():
    rng = random.Random(11)
    checked = 0
    for D in (30, 60, 120):
        for _ in range(40):
            x = rng.randrange(10 ** (D - 1), 10 ** D)
            while x % 9:
                x = f(x)
            seen = set()
            while x not in seen:
                seen.add(x)
                S = s(x)
                k = S // 9
                kap = s(9 * k) // 9
                ell = len(str(9 * k))
                start = x
                while True:
                    y = f(x)
                    if s(y) != S or (y % 2) != (x % 2) or y in seen:
                        break
                    x = y
                    seen.add(x)
                if x != start:
                    checked += 1
                    assert abs(x - start) < 2 * 10 ** (ell + kap)
                x = f(x)
    assert checked > 20


# Block certificate (computer-assisted convergence): soundness on small ranges.
# Whenever the certificate holds, brute force must confirm H <= H_0 + 1.
@pytest.mark.parametrize("b,D,L", [(2, 18, 9), (4, 8, 5), (10, 5, 3), (10, 6, 4)])
def test_block_certificate_sound(b, D, L):
    from digitdyn.baseb import f as fb
    from digitdyn.certify_b import certify
    bad, _, _ = certify(b, D, L, verbose=False)
    assert not bad
    M = b ** L
    for n in range(1, b ** D):
        H0, x, seen = n // M, n, set()
        while x not in seen:
            seen.add(x)
            x = fb(x, b)
            assert x // M <= H0 + 1


def test_block_certificate_fails_when_blocks_too_small():
    # with L = 8 in base 2 the certificate fails, and a genuine 2-level climb exists
    from digitdyn.baseb import f as fb
    from digitdyn.certify_b import certify
    bad, _, _ = certify(2, 16, 8, verbose=False)
    assert bad
    M, worst = 2 ** 8, 0
    for n in range(1, 2 ** 16):
        H0, x, seen, hmax = n // M, n, set(), n // M
        while x not in seen:
            seen.add(x)
            x = fb(x, 2)
            hmax = max(hmax, x // M)
        worst = max(worst, hmax - H0)
    assert worst == 2


def test_climb_method_matches_block_certificate():
    from digitdyn.climbcert import max_height
    from digitdyn.certify_b import certify
    for b, D, L in [(10, 6, 4), (2, 16, 8), (2, 18, 9), (4, 8, 5)]:
        smax = (b - 1) * (D - L)
        E = smax + (b - 1) * L + 1
        climb_ok = max(max_height(b, sg, E) for sg in range(smax + 1)) < b ** L
        bad, _, _ = certify(b, D, L, verbose=False)
        assert climb_ok == (not bad)


def test_uprun_corrected_statement():
    """Prop uprun (corrected): along an ascent from odd 9|n with s(n) even < 198,
    the digit sum never increases except possibly by 9 at the last term."""
    from digitdyn.explore import s
    seen_increase = False
    for n0 in range(9, 400_000, 18):           # odd multiples of 9
        if s(n0) % 2:
            continue
        run = [n0]
        while run[-1] % 2 == 1 and s(run[-1]) % 2 == 0:
            run.append(run[-1] + s(run[-1]))
        if run[-1] % 2 == 0:
            run.pop()                            # keep only the odd terms
        sums = [s(x) for x in run]
        for j in range(1, len(sums)):
            if sums[j] > sums[j - 1]:
                assert j == len(sums) - 1 and sums[j] == sums[j - 1] + 9
                seen_increase = True
    assert seen_increase                         # e.g. 8901 -> 8919


def test_constant_level_run_odd_level_alternates():
    from digitdyn.explore import f, s
    n = 8919                                     # level 3, odd
    assert f(n) == 8946 and s(8946) == s(n) and f(8946) == n


def test_exact_carry_variance_formula():
    """Lemma exactvar: Var C = (11/9) sum q_i(1-q_i), q_{i+1} = (t_i + q_i)/10."""
    import random
    from digitdyn.variance import stats
    random.seed(5)
    for _ in range(60):
        t = random.randrange(1, 10 ** random.randint(1, 15))
        d = [int(c) for c in reversed(str(t))] + [0] * 40
        q = sum(1 for a0 in (1, 3, 5, 7, 9) if a0 + d[0] >= 10) / 5
        tot = 0.0
        for i in range(1, len(d)):
            tot += q * (1 - q)
            q = (d[i] + q) / 10
        assert abs(11 / 9 * tot - stats(t)[1]) < 1e-9


def test_extremal_digit_average_bound():
    """Lemma kappa: sum phi(x_i) <= 30n/121 + 5/22 for every digit sequence."""
    import random
    random.seed(6)
    phi = lambda x: x * (1 - x)
    for trial in range(2000):
        x = random.random()
        n = random.randint(1, 60)
        if trial % 3 == 0:
            digs = [random.choice((4, 5)) for _ in range(n)]
        else:
            digs = [random.randrange(10) for _ in range(n)]
        s = 0.0
        for dgt in digs:
            s += phi(x)
            x = (dgt + x) / 10
        assert s <= 30 * n / 121 + 5 / 22 + 1e-12
    # the 45-cycle attains the constant
    x, s = 0.5, 0.0
    for i in range(2000):
        s += phi(x); x = ((4, 5)[i % 2] + x) / 10
    assert abs(s / 2000 - 30 / 121) < 1e-3


def test_sharp_local_limit_trend():
    import math, random
    from digitdyn.variance import stats
    random.seed(8)
    for nd, tol in ((30, 0.03), (150, 0.01)):
        for _ in range(20):
            t = random.randrange(10 ** (nd - 1), 10 ** nd); t += (9 - t % 18) % 18
            mu, V, m, r, g = stats(t)
            assert abs(r * math.sqrt(2 * math.pi * V) - 1) < tol


def test_basin_census_small():
    from digitdyn.basins import run
    r = run(20000, margin=20000)
    cyc = r["distinct_cycles_by_length"]
    assert cyc[1] == 1 and cyc[2] > cyc.get(4, 0)
    assert abs(sum(r["start_fraction_by_cycle_length"].values()) - 1) < 1e-9


def test_climb_profile_certifies_10_pow_2224():
    """The saved profile behind Thm certres: every sigma <= 19944 stays below 10^8,
    and spot-checked sigmas recompute exactly."""
    import json
    from digitdyn.climbcert import max_height
    d = json.load(open("digitdyn/data/climb_profile_b10_S19944_L8.json"))
    assert d["E"] == 19944 + 9 * 8 + 1 and len(d["heights"]) == 19945
    assert max(d["heights"]) < 10 ** 8
    for sg in (0, 1952, 9960):
        assert max_height(10, sg, d["E"]) == d["heights"][sg]


def test_standalone_checker_agrees_and_fails_correctly():
    import subprocess, sys
    ok = subprocess.run([sys.executable, "digitdyn/verify_certificates.py", "10", "63", "5"], capture_output=True, text=True)
    assert ok.returncode == 0 and "VERIFIED: every orbit of f_10 starting below 10^12" in ok.stdout
    bad = subprocess.run([sys.executable, "digitdyn/verify_certificates.py", "2", "8", "8"], capture_output=True, text=True)
    assert bad.returncode == 1 and "NOT VERIFIED" in bad.stdout
