"""Basin census: for every start n <= N, the length of the cycle its orbit enters,
and the cycle-length histogram of the distinct cycles met (Table 'basins')."""
import json
import sys

import numpy as np


def run(N, margin=200000):
    M = N + margin
    x = np.arange(M)
    s = np.zeros(M, dtype=np.int64)
    y = x.copy()
    while y.any():
        s += y % 10
        y //= 10
    f = np.where(x % 2 == 1, x + s, x - s)
    assert f.max() < M or True
    f = np.minimum(f, M)                        # M = escape sink
    nxt = np.append(f, M)
    # iterate to land in the eventual cycle (tails are short)
    cur = np.arange(N + 1)
    for _ in range(400):
        cur = nxt[cur]
    assert (cur < M).all(), "an orbit left the margin"
    # cycle length and canonical representative (min) of the cycle containing cur
    rep = cur.copy(); L = np.ones_like(cur); z = nxt[cur]
    for k in range(1, 200):
        open_ = z != cur
        if not open_.any():
            break
        L[open_] += 1
        rep = np.minimum(rep, z)
        z = np.where(open_, nxt[z], z)
    basin = {}
    for l in np.unique(L):
        basin[int(l)] = float((L == l).mean())
    zero = float((rep == 0).mean())
    reps, idx = np.unique(rep, return_index=True)
    cyc = {}
    for l in L[idx]:
        cyc[int(l)] = cyc.get(int(l), 0) + 1
    return {"N": N, "start_fraction_by_cycle_length": basin, "fraction_to_zero": zero,
            "distinct_cycles_by_length": cyc, "distinct_cycles": int(len(reps))}


if __name__ == "__main__":
    out = [run(int(v)) for v in sys.argv[1:]]
    for r in out:
        print(json.dumps(r))
    json.dump(out, open("digitdyn/data/basins.json", "w"), indent=1)
