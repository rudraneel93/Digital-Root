"""Sampled main term at peaks and troughs with standard errors (Remark 'osc ratios').
Peak: D = floor(10^j/4.5); trough: D = floor(2 y_j / 9), y_j = 4545... (j digits).
Ratio R = (rho_p/rho_t) * sqrt(log D_p / log D_t); standard error by the delta method.
"""
import json
import math
import random
import sys

from .variance import stats


def sample(D, n, rng):
    mu = 9 * D // 2
    sd = math.isqrt(33 * D // 4)
    jitter = min(10 ** 12, max(1, sd // 1000))
    vals = []
    for _ in range(n):
        t = mu + int(rng.gauss(0, 1) * 2 ** 40) * sd // 2 ** 40 + rng.randrange(jitter)
        t += (9 - t % 18) % 18
        vals.append(stats(t)[3])
    m = sum(vals) / n
    v = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(v / n)


def main():
    n = int(sys.argv[1])
    js = [int(v) for v in sys.argv[2].split(",")]
    rng = random.Random(7)
    rows = []
    for j in js:
        Dp = 10 ** j * 2 // 9
        Dt = int(("45" * 80)[:j]) * 2 // 9
        mp, sp = sample(Dp, n, rng)
        mt, st = sample(Dt, n, rng)
        c = math.sqrt(math.log(Dp) / math.log(Dt))
        R = mp / mt * c
        se = R * math.sqrt((sp / mp) ** 2 + (st / mt) ** 2)
        rows.append({"j": j, "peak": mp, "peak_se": sp, "trough": mt, "trough_se": st, "ratio": R, "ratio_se": se})
        print(f"j={j}: ratio {R:.3f} +- {se:.3f}", flush=True)
    json.dump(rows, open("digitdyn/data/osc_sampling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
