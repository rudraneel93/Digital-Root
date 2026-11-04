"""Parallel climb certificate with per-sigma maxima.

python3 -m digitdyn.climbcert_par <b> <SMAX> <L> <jobs>
Writes data/climb_profile_b<b>_S<SMAX>_L<L>.json with the max climb height for
every sigma (entries x0 < E = SMAX + (b-1)L + 1), and prints the verdict.
"""
import json
import sys
import time
from multiprocessing import Pool

from .climbcert import max_height


def work(args):
    b, sg, E = args
    return sg, max_height(b, sg, E)


def main():
    b, SMAX, L, jobs = (int(v) for v in sys.argv[1:5])
    E = SMAX + (b - 1) * L + 1
    t0 = time.time()
    res = {}
    with Pool(jobs) as pool:
        for k, (sg, h) in enumerate(pool.imap_unordered(work, [(b, s, E) for s in range(SMAX, -1, -1)], chunksize=4)):
            res[sg] = h
            if k % 1000 == 0:
                print(f"  {k}/{SMAX + 1} sigmas done, max so far {max(res.values()):,} [{time.time() - t0:.0f}s]", flush=True)
    top = max(res.values())
    worst = max(res, key=res.get)
    D = SMAX // (b - 1) + L
    out = f"digitdyn/data/climb_profile_b{b}_S{SMAX}_L{L}.json"
    json.dump({"b": b, "SMAX": SMAX, "L": L, "E": E, "max": top, "argmax_sigma": worst,
               "heights": [res[s] for s in range(SMAX + 1)]}, open(out, "w"))
    print(f"base {b}: sigma <= {SMAX}, entries < {E}: max climb {top:,} at sigma={worst}; b^L = {b ** L:,}")
    print(("CERTIFIED: every orbit of f_%d starting below %d^%d is eventually periodic." % (b, b, D))
          if top < b ** L else "NOT CERTIFIED at this L")


if __name__ == "__main__":
    main()
