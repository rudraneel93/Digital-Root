"""Figures for the manuscript (PDF, print-oriented, single light theme).

Palette: validated reference categorical slots 1-2 (blue #2a78d6, orange
#eb6834) on white; text in primary/secondary ink; recessive grid.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .explore import f, s
from .resonance import approx_density, rho

OUT = "digitdyn/manuscript/figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "mathtext.fontset": "stix",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def fig_mountain():
    """The 15-cycle through 1791: ascent on the odd rail, descent on the even
    rail, and the reversing shift x -> x + 9 mapping one onto the other."""
    c = [1791]
    while True:
        y = f(c[-1])
        if y == c[0]:
            break
        c.append(y)
    odd = [v for v in c if v % 2 == 1]
    asc = [v for v in c if v % 2 == 1]     # odd rail: n -> n + s(n)
    desc = [v for v in c if v % 2 == 0]    # even rail: n -> n - s(n)
    assert c[: len(asc)] == asc, "cycle should start with its ascent"
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ia = np.arange(len(asc))
    idsc = np.arange(len(asc), len(c))
    ax.plot(np.r_[ia, idsc, len(c)], np.r_[asc, desc, c[0]], color=GRID, lw=1.0, zorder=1)
    ax.plot(ia, asc, "o", ms=6, color=BLUE, mec="white", mew=1.2, zorder=3, label="ascent (odd rail, $+s(n)$)")
    ax.plot(idsc, desc, "s", ms=5.5, color=ORANGE, mec="white", mew=1.2, zorder=3, label="descent (even rail, $-s(n)$)")
    for i, v in zip(ia, asc):
        ax.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=6.5, color=INK2)
    for i, v in zip(idsc, desc):
        ax.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("step")
    ax.set_ylabel("$n$")
    ax.set_title("A 15-cycle: one ascent, one descent; descent = ascent reversed and shifted by 9")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xlim(-0.7, len(c) - 0.3)
    ax.set_ylim(1770, 1945)
    save(fig, "mountain_cycle.pdf")


def fig_profile():
    d = np.load("digitdyn/data/profile_L5.npz")
    c2 = d["c2"].astype(float)
    sig = np.arange(len(c2))
    w = 18
    sm = np.convolve(c2, np.ones(w) / w, mode="same")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot(sig, c2 / 1e5, color=BLUE, lw=0.5, alpha=0.35)
    ax.plot(sig[w:-w], sm[w:-w] / 1e5, color=BLUE, lw=1.8)
    for t9 in (99, 999, 1999):
        x = t9 - 22
        ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.annotate(f"$\\sigma+22\\approx{t9}$", (x, 0.0272), ha="left", va="center", fontsize=7,
                    color=INK2, xytext=(4, 0), textcoords="offset points",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xlabel(r"prefix digit sum $\sigma$")
    ax.set_ylabel(r"$c_2(\sigma)/10^5$")
    ax.set_title(r"2-cycles of the low map ($L=5$): peaks where $s(n)\approx\sigma+22$ ends in nines", pad=10)
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 0.0295)
    save(fig, "profile_c2.pdf")


def fig_density():
    exact = json.load(open("digitdyn/data/resonance_vs_exact.json"))
    De = [r[0] for r in exact]
    ve = [r[1] for r in exact]
    Ds = np.unique(np.round(np.logspace(np.log10(6), np.log10(10000), 160)).astype(int))
    curve = [approx_density(int(D)) for D in Ds]
    json.dump(dict(D=Ds.tolist(), density=curve), open("digitdyn/data/density_curve.json", "w"))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(Ds, curve, color=BLUE, lw=1.6, label="carry-resonance formula")
    ax.plot(De, ve, "o", ms=6, color=ORANGE, mec="white", mew=1.2, label="exact count (digit DP)")
    for D9 in (22, 222, 2222):
        ax.axvline(D9, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.annotate(f"$4.5D\\approx{int(round(4.5 * D9))}$", (D9, 0.0222), ha="left", va="center",
                    fontsize=7, color=INK2, xytext=(4, 0), textcoords="offset points",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xscale("log")
    ax.set_xlabel("number of digits $D$ (log scale)")
    ax.set_ylabel(r"$T(D)/10^D$")
    ax.set_title("Density of 2-cycles below $10^D$ (peaks when $4.5D$ is just below a power of 10)")
    ax.set_ylim(0.0, 0.0235)
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, "density_2cycles.pdf")


def fig_rho():
    ts = np.arange(9, 3000, 18)
    r = [rho(int(t)) for t in ts]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.plot(ts, r, color=BLUE, lw=1.0)
    ax.plot(ts, r, "o", ms=2.2, color=BLUE)
    for t9 in (99, 999, 1999, 2999):
        ax.axvline(t9, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel(r"digit sum $t$ (odd multiples of 9)")
    ax.set_ylabel(r"$\rho(t)$")
    ax.set_title(r"Carry resonance: probability of exactly $s(t)/9$ carries in $a+t$ (equal to $0.9$ at $t=10^j-1$)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 3000)
    save(fig, "carry_resonance.pdf")


def fig_cycle_lengths():
    lengths = {2: 37374, 3: 81, 4: 4759, 5: 72, 8: 605, 14: 17, 15: 284, 16: 18}
    ks = list(lengths)
    vs = [lengths[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.bar([str(k) for k in ks], vs, color=BLUE, width=0.6)
    ax.set_yscale("log")
    for i, v in enumerate(vs):
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7, color=INK2)
    ax.set_xlabel("cycle length")
    ax.set_ylabel("distinct cycles (log)")
    ax.set_title(r"Cycles reached from $1 \leq n \leq 2\cdot10^6$")
    ax.grid(axis="x", visible=False)
    save(fig, "cycle_lengths.pdf")


def fig_excursion():
    digits = [10, 20, 50, 100, 200, 400]
    exc = [661, 1031, 5568, 5921, 10592, 20505]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(digits, exc, "o", ms=6, color=BLUE, mec="white", mew=1.2)
    xs = np.array([0, 420])
    ax.plot(xs, 51.5 * xs, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.annotate(r"$\approx 51\,D$", (300, 51.5 * 300), textcoords="offset points", xytext=(6, -10),
                fontsize=7.5, color=INK2)
    ax.set_xlabel("number of digits $D$ of the start")
    ax.set_ylabel(r"max excursion $\max(\mathrm{orbit})-n$")
    ax.set_title("Largest climb over 300 random starts")
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 23000)
    save(fig, "excursion.pdf")


def main():
    fig_mountain()
    fig_profile()
    fig_rho()
    fig_cycle_lengths()
    fig_excursion()
    fig_density()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()


def carry_pmf(t, tail=14):
    """Exact distribution of the number of carries in a + t, a with uniform
    digits and odd units digit (same chain as resonance.rho)."""
    digits = [int(ch) for ch in reversed(str(t))] + [0] * tail
    v0, v1 = np.array([1.0]), np.array([0.0])
    for i, d in enumerate(digits):
        if i == 0:
            p_c = sum(1 for a0 in (1, 3, 5, 7, 9) if a0 + d >= 10) / 5
            v0, v1 = v0 * (1 - p_c), np.concatenate([[0.0], v0 * p_c])
            continue
        a00, a01, a10, a11 = (10 - d) / 10, d / 10, (9 - d) / 10, (d + 1) / 10
        n0 = np.zeros(max(len(v0), len(v1)))
        n0[: len(v0)] += v0 * a00
        n0[: len(v1)] += v1 * a10
        n1 = np.zeros(max(len(v0), len(v1)) + 1)
        n1[1: len(v0) + 1] += v0 * a01
        n1[1: len(v1) + 1] += v1 * a11
        v0, v1 = n0, n1
    n = max(len(v0), len(v1))
    tot = np.zeros(n)
    tot[: len(v0)] += v0
    tot[: len(v1)] += v1
    return tot


def fig_carry_balance():
    t = 4_536_271_809_123_457
    t += (9 - t % 18) % 18
    pmf = carry_pmf(t)
    m = s(t) // 9
    ks = np.arange(len(pmf))
    mean = float((ks * pmf).sum())
    keep = pmf > 1e-4
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    cols = [ORANGE if k == m else BLUE for k in ks[keep]]
    ax.bar(ks[keep], pmf[keep], color=cols, width=0.7)
    top = pmf.max()
    ax.axvline(mean, color=INK2, lw=0.9, ls=(0, (3, 3)), ymax=0.80)
    # mean label: right of the line, at the top of the plot area
    ax.annotate(f"mean $= {mean:.3f}$", xy=(mean, top * 1.18), xytext=(mean + 1.9, top * 1.30),
                fontsize=7.5, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    # required value label: left of the highlighted bar
    ax.annotate(f"required $s(t)/9 = {m}$,  $P = {pmf[m]:.3f}$", xy=(m - 0.36, top * 0.97),
                xytext=(m - 6.6, top * 1.30), fontsize=7.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_ylim(0, top * 1.42)
    ax.set_xlim(-0.5, ks[keep].max() + 0.7)
    ax.set_xlabel(r"number of carries in $a+t$")
    ax.set_ylabel("probability")
    ax.set_title("Carry balance: the 2-cycle condition asks for the mean number of carries")
    ax.grid(axis="x", visible=False)
    save(fig, "carry_balance.pdf")
    return t, m, mean, float(pmf[m])


def fig_normalized():
    d = json.load(open("digitdyn/data/density_curve.json"))
    D = np.array(d["D"])
    v = np.array(d["density"])
    exact = json.load(open("digitdyn/data/resonance_vs_exact.json"))
    De = np.array([r[0] for r in exact])
    ve = np.array([r[1] for r in exact])
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.plot(D, v * np.sqrt(np.log(D)), color=BLUE, lw=1.6, label="resonance formula")
    ax.plot(De, ve * np.sqrt(np.log(De)), "o", ms=6, color=ORANGE, mec="white", mew=1.2,
            label="exact")
    ax.set_xscale("log")
    ax.set_xlabel("number of digits $D$ (log scale)")
    ax.set_ylabel("√(ln $D$) × $T(D)/10^D$")
    ax.set_title("Normalised 2-cycle density √(ln $D$) × $T(D)/10^D$: bounded above and below")
    ax.set_ylim(0, 0.045)
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "normalized_density.pdf")


def fig_bases():
    """Normalised 2-cycle density in bases 2, 4, 6, 10 against log_b of the
    typical digit sum (b-1)D/2: resonance peaks align at integers."""
    import math
    base = json.load(open("digitdyn/data/base_density.json"))
    ex10 = json.load(open("digitdyn/data/resonance_vs_exact.json"))
    series = {2: base["2"], 4: base["4"], 6: base["6"], 10: [(r[0], r[1]) for r in ex10]}
    styles = {2: (BLUE, "o"), 4: (ORANGE, "s"), 6: ("#1baf7a", "^"), 10: ("#eda100", "D")}
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    for b, pts in series.items():
        pts = [(D, v) for D, v in pts if D >= 4]
        xs = [math.log((b - 1) * D / 2, b) for D, _ in pts]
        ys = [v * math.sqrt(math.log(D)) for D, v in pts]
        m = sum(ys) / len(ys)
        ys = [y / m for y in ys]
        col, mk = styles[b]
        ax.plot(xs, ys, "-", color=col, lw=1.2, alpha=0.8)
        ax.plot(xs, ys, mk, color=col, ms=5.5, mec="white", mew=1.0, label=f"base {b}")
    for j in range(1, 7):
        ax.axvline(j, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("log$_b$ of the typical digit sum $(b-1)D/2$")
    ax.set_ylabel("normalised density / mean")
    ax.set_title("2-cycle density in four bases: peaks where the typical digit sum is near $b^j$")
    ax.legend(loc="upper left", fontsize=7.5, ncol=4)
    ax.set_ylim(0.6, 1.55)
    save(fig, "bases_resonance.pdf")


def fig_osclaw():
    """Oscillation law (Theorem osclaw / Corollary osc). (a) The law along D,
    sampled densely inside the resonance neighbourhoods of D = 10^j/4.5, with
    the proved limsup, the typical value and the liminf, and sampled main-term
    values. (b) Universal spike shape: the law at D = (10^j/4.5)(1 - 10^(-r j/2))
    against r for several j."""
    import math
    from decimal import Decimal, getcontext
    getcontext().prec = 200
    phi = lambda x: x * (1 - x)
    K0 = 1 / (12 * math.sqrt(11 * math.pi))
    hi, mid, lo = K0 * math.sqrt(6), K0 * math.sqrt(3), K0 * math.sqrt(726 / 301)

    def pred(D):
        y = Decimal(9) * Decimal(D) / 2
        lg = math.log10(D)
        h = lg / 2 - math.log10(math.log(D))
        B, i = 0.0, max(1, int(math.ceil(h)))
        while i < lg + 25:
            q = y / (Decimal(10) ** i)
            B += phi(float(q - int(q)))
            i += 1
        return math.sqrt(lg) / (36 * math.sqrt(22 * math.pi / 9 * (h / 6 + B)))

    Ds = {int(Decimal(10) ** Decimal(float(u))) for u in np.linspace(2.0, 100.0, 1500)}
    for j in range(3, 101):
        c = Decimal(10) ** j / Decimal("4.5")
        for k in np.linspace(0.5, j, 40):
            e = Decimal(10) ** Decimal(-float(k))
            Ds.add(int(c * (1 - e)))
            Ds.add(int(c * (1 + e)))
    Ds = sorted(d for d in Ds if d > 50)
    xs = [math.log10(d) for d in Ds]
    ys = [pred(d) for d in Ds]
    samp_peak = {8: .3980, 12: .3343, 16: .2937, 24: .2425, 32: .2128, 48: .1761, 64: .1530, 96: .1259}
    samp_tr = {8: .2688, 12: .2226, 16: .1940, 24: .1595, 32: .1385, 48: .1135, 64: .0985, 96: .0806}
    px = [math.log10(10 ** j * 2 // 9) for j in samp_peak]
    py = [math.sqrt(x) * r / 36 for x, r in zip(px, samp_peak.values())]
    tx = [math.log10(int(("45" * 60)[:j]) * 2 // 9) for j in samp_tr]
    ty = [math.sqrt(x) * r / 36 for x, r in zip(tx, samp_tr.values())]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.7, 1]})
    ax.plot(xs, ys, "-", color=BLUE, lw=0.5, alpha=0.9, label="the law (Theorem 10.10)")
    for v, lab in ((hi, r"$\limsup=K_0\sqrt{6}$"), (mid, r"typical $K_0\sqrt{3}$"),
                   (lo, r"$\liminf=K_0\sqrt{726/301}$")):
        ax.axhline(v, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(1.5, v + 0.0003, lab, va="bottom", fontsize=6.5, color=INK2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.plot(px, py, "o", ms=5.5, color=ORANGE, mec="white", mew=1.0, label="sampled, $D=\\lfloor 10^j/4.5\\rfloor$")
    ax.plot(tx, ty, "s", ms=5, color="#1baf7a", mec="white", mew=1.0, label="sampled, $4.5D\\approx 4545\\cdots$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.017, 0.038)
    ax.set_xlabel(r"$\log_{10} D$")
    ax.set_ylabel(r"$\sqrt{\log_{10}D}\;T(D)/10^D$")
    ax.set_title("(a) spikes at $D\\approx10^j/4.5$", loc="left")

    rs = np.linspace(0.0, 1.0, 60)
    cols = {20: BLUE, 40: ORANGE, 80: "#1baf7a"}
    for j, col in cols.items():
        c = Decimal(10) ** j / Decimal("4.5")
        ys2 = [pred(int(c * (1 - Decimal(10) ** Decimal(-max(r, 0.02) * j / 2)))) for r in rs]
        bx.plot(rs, ys2, "-", color=col, lw=1.3, label=f"$j={j}$")
    bx.axhline(hi, color=INK2, lw=0.8, ls=(0, (4, 3)))
    bx.axhline(mid, color=INK2, lw=0.8, ls=(0, (4, 3)))
    bx.set_xlabel("$r$: leading nines $\\approx r\\cdot j/2$")
    bx.set_title("(b) spike shape", loc="left")
    bx.set_ylim(0.017, 0.038)
    bx.set_yticklabels([])
    bx.legend(loc="upper left", fontsize=6.5)
    h1, l1 = ax.get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.45, -0.08))
    fig.subplots_adjust(wspace=0.08, bottom=0.22)
    save(fig, "oscillation_law.pdf")


def fig_basins():
    """Distinct cycles versus starting values, by cycle length (n <= 2*10^6)."""
    data = json.load(open("digitdyn/data/basins.json"))[-1]
    cyc = {int(k): v for k, v in data["distinct_cycles_by_length"].items() if k != "1"}
    st = {int(k): v for k, v in data["start_fraction_by_cycle_length"].items() if k != "1"}
    tot = sum(cyc.values())
    lens = sorted(cyc)
    y = np.arange(len(lens))
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    a = [cyc[l] / tot for l in lens]
    b = [st.get(l, 0) for l in lens]
    ax.barh(y + 0.2, a, height=0.38, color=BLUE, label="share of distinct cycles", edgecolor="white", linewidth=1.0)
    ax.barh(y - 0.2, b, height=0.38, color=ORANGE, label="share of starting values $n$", edgecolor="white", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([str(l) for l in lens])
    ax.invert_yaxis()
    ax.set_ylabel("cycle length")
    ax.set_xlabel("share (logarithmic scale)")
    for yi, ai, bi in zip(y, a, b):
        if ai > 0.05 or bi > 0.05:
            ax.text(ai * 1.15, yi + 0.2, f"{100 * ai:.1f}%", va="center", fontsize=7, color=INK2)
            ax.text(bi * 1.15, yi - 0.2, f"{100 * bi:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_xlim(2e-4, 3)
    ax.set_title(r"Cycles and their basins for $1\leq n\leq 2\cdot10^6$")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "basins.pdf")


if __name__ == "__main__":
    fig_osclaw()
    fig_basins()
