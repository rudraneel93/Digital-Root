"""Certificate via exact climb heights (no block arrays).

For every sigma <= SMAX and every entry x0 < E = SMAX + (b-1)L + 1, follow the
low map g_sigma(x) = x +- (sigma + s_b(x)) on the half-line until it falls below
0 or enters a cycle, and record the highest point reached. If every height is
< b^L, the hypothesis of the block certificate (Theorem cert) holds, because
g_sigma on [0, b^L) agrees with the half-line map until the orbit leaves.

Usage: python3 -m digitdyn.climbcert <b> <SMAX> <L>
"""
import sys
import time


def sb(x, b):
    t = 0
    while x:
        t += x % b
        x //= b
    return t


def max_height(b, sg, E):
    memo = {}                     # state -> max height of its forward orbit
    worst = 0
    for x0 in range(E):
        path = []
        pos = {}
        x = x0
        while x >= 0 and x not in memo and x not in pos:
            pos[x] = len(path)
            path.append(x)
            S = sg + sb(x, b)
            x = x + S if x % 2 else x - S
        if x < 0:
            tail = -1
        elif x in memo:
            tail = memo[x]
        else:                     # closed a new cycle starting at index pos[x]
            i = pos[x]
            cyc = max(path[i:])
            for y in path[i:]:
                memo[y] = cyc
            path = path[:i]
            tail = cyc
        for y in reversed(path):
            tail = max(tail, y)
            memo[y] = tail
        worst = max(worst, memo[x0])
    return worst


def main():
    b, SMAX, L = (int(v) for v in sys.argv[1:4])
    E = SMAX + (b - 1) * L + 1
    t0 = time.time()
    top = 0
    ratio = 0.0
    for sg in range(SMAX + 1):
        h = max_height(b, sg, E)
        top = max(top, h)
        if sg >= 10:
            ratio = max(ratio, h / (sg + 1) ** 2)
        if sg % 500 == 0:
            print(f"  sigma={sg} max height so far {top:,} [{time.time() - t0:.0f}s]", flush=True)
    D = SMAX // (b - 1) + L
    ok = top < b ** L
    print(f"base {b}: sigma <= {SMAX}, entries < {E}: max climb height {top:,} "
          f"(max h/(sigma+1)^2 for sigma>=10: {ratio:.3f}); b^L = {b ** L:,}")
    if ok:
        print(f"CERTIFIED: every orbit of f_{b} starting below {b}^{D} is bounded and eventually periodic.")
    else:
        print("NOT CERTIFIED at this L")


if __name__ == "__main__":
    main()
