"""Local maxima and minima of the normalised 2-cycle main term (Table 'osc').

Computes sqrt(ln D) * (1/2) * sum_t P_D(t) rho(t) near D = (10^j - 1)/4.5
(peaks) and D = 2.5 * 10^j / 4.5 (troughs), using a Gaussian approximation to
the digit-sum law P_D (validated against the exact convolution at
D = 200, 1000, 5000). Run:  python3 -m digitdyn.oscillation [jmax]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from .resonance import approx_density, rho


def dens_gauss(D):
    mu, var = 4.5 * D, 8.25 * D
    sd = math.sqrt(var)
    lo = max(9, int(mu - 12 * sd))
    hi = int(mu + 12 * sd)
    t0 = lo + ((9 - lo) % 18)
    tot = 0.0
    for t in range(t0, hi + 1, 18):
        p = math.exp(-(t - mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
        tot += p * 0.5 * rho(t)
    return tot


def norm(D):
    return math.sqrt(math.log(D)) * dens_gauss(D)


def main():
    jmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for D in (200, 1000, 5000):
        print(f"validation D={D}: exact-P {approx_density(D):.6f}  gaussian-P {dens_gauss(D):.6f}")
    rows = []
    print(" j   D_peak     norm(peak)   D_trough   norm(trough)   ratio")
    for j in range(2, jmax + 1):
        Dp, Dt = (10 ** j - 1) / 4.5, 2.5 * 10 ** j / 4.5
        cand_p = sorted({max(2, int(Dp * x)) for x in np.linspace(0.85, 1.04, 20)})
        cand_t = sorted({max(2, int(Dt * x)) for x in np.linspace(0.7, 1.4, 20)})
        bp = max(cand_p, key=norm)
        bt = min(cand_t, key=norm)
        r = (j, bp, norm(bp), bt, norm(bt), norm(bp) / norm(bt))
        rows.append(r)
        print(f"{j:2d} {bp:9d}   {r[2]:.5f}   {bt:9d}   {r[4]:.5f}      {r[5]:.3f}", flush=True)
    json.dump(rows, open("digitdyn/data/oscillation.json", "w"))


if __name__ == "__main__":
    main()
