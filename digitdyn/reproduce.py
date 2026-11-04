"""Regenerate every table, number and figure in the manuscript.

Run from the repository root:   python3 -m digitdyn.reproduce
Takes about 10-15 minutes on a laptop (the exact counts up to D = 200 and
the L = 5 profile dominate). Outputs go to digitdyn/data/ and
digitdyn/manuscript/figures/.
"""
from __future__ import annotations

import runpy
import sys
import time

STEPS = [
    ("Orbit census to 2*10^6: convergence, cycle lengths, excursion, 2-cycle and "
     "two-step checks (Prop. census, Table 'cycles')", "digitdyn.explore", ["2000000"]),
    ("Exact T(D) for D up to 200 by digit DP (Table 'density')", "digitdyn.count2", []),
    ("Structural identities and prefix-lifting check (Thms self, level, reverse, lift)",
     "digitdyn.structure", []),
    ("Low-map profile c_L(sigma), L = 5 (Fig. 'profile')", "digitdyn.profile", ["5", "2500"]),
    ("Carry-resonance values and approximation (Table 'density', Fig. 'rho')",
     "digitdyn.resonance", []),
    ("Largest excursion over 300 seeded random starts per digit length (Fig. 'excursion')",
     "digitdyn.excursion", []),
    ("Oscillation of the normalised 2-cycle density (Table 'osc'; slow: large D)",
     "digitdyn.oscillation", []),
    ("Base-b structure checks (Thms oddbase, evenbase)", "digitdyn.baseb", []),
    ("Exact 2-cycle densities in bases 2, 4, 6 (Fig. 'bases')", "digitdyn.basedensity", []),
    ("Block certificate: all n < 2^128 (base 2)", "digitdyn.certify_b", ["2", "128", "18"]),
    ("Block certificate: all n < 10^20 (base 10)", "digitdyn.certify_b", ["10", "20", "6"]),
    ("Climb certificate: all n < 10^341 (base 10, Thm certres)", "digitdyn.climbcert", ["10", "3000", "8"]),
    ("Climb certificate: all n < 2^3024 (base 2, Thm certres)", "digitdyn.climbcert", ["2", "3000", "24"]),
    ("Basin census: cycles vs starting values (Table 'cycles', Fig. 'basins')", "digitdyn.basins", ["200000", "1000000", "2000000"]),
    ("Carry-variance formula and sharp local limit check (Sec. 'osclaw')", "digitdyn.variance", []),
    ("All figures", "digitdyn.figures", []),
]


def main():
    t0 = time.time()
    for title, module, args in STEPS:
        print(f"\n=== {title}\n    python3 -m {module} {' '.join(args)}", flush=True)
        sys.argv = [module] + args
        runpy.run_module(module, run_name="__main__")
    from digitdyn import figures
    figures.fig_carry_balance()
    figures.fig_normalized()
    figures.fig_bases()
    print(f"\nDone in {time.time() - t0:.0f}s. Run the tests with: python3 -m pytest digitdyn/tests -q")


if __name__ == "__main__":
    main()
