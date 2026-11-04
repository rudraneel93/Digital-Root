"""Exact 2-cycle densities T_b(D)/b^D in even bases (digit DP from baseb)."""
import math, json, sys
from .baseb import T

def main():
    out = {}
    plan = {2: [4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 200],
            4: [3, 5, 8, 12, 16, 24, 32, 48, 64, 80],
            6: [3, 5, 8, 12, 16, 24, 32, 48]}
    for b, Ds in plan.items():
        out[b] = []
        for D in Ds:
            v = T(D, b) / b ** D
            out[b].append((D, v))
            print(f"b={b} D={D:4d}  T/b^D={v:.6f}  sqrt(lnD)*T/b^D={v*math.sqrt(math.log(D)):.5f}", flush=True)
    json.dump(out, open("digitdyn/data/base_density.json", "w"))

if __name__ == "__main__":
    main()
