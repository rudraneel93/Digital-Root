"""Largest excursion max(orbit) - n over random starting values (Figure 'excursion').

For each number of digits D in (10, 20, 50, 100, 200, 400), draws 300 uniform
random D-digit starting values with random.Random(0) (fixed seed) and records
the largest excursion and the largest number of steps to reach a cycle.
Run: python3 -m digitdyn.excursion
"""
from __future__ import annotations

import json
import random

from .explore import f


def main(seed: int = 0, starts: int = 300):
    rng = random.Random(seed)
    rows = []
    for D in (10, 20, 50, 100, 200, 400):
        worst = steps = 0
        for _ in range(starts):
            n = rng.randrange(10 ** (D - 1), 10 ** D)
            seen, x, top, k = set(), n, n, 0
            while x not in seen and x > 0:
                seen.add(x)
                x = f(x)
                top = max(top, x)
                k += 1
            worst = max(worst, top - n)
            steps = max(steps, k)
        rows.append(dict(D=D, max_excursion=worst, max_steps=steps))
        print(f"D={D:4d}: max excursion {worst:7d}, max steps {steps}", flush=True)
    json.dump(dict(seed=seed, starts=starts, rows=rows),
              open("digitdyn/data/excursion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
