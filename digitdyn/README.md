# digitdyn: code for "Carries, Rails and Resonance"

Reproduces every computation in R. Das, *Carries, Rails and Resonance: the
Structure of the Digit-Sum Parity Map* (manuscript in `manuscript/`), which
studies f(n) = n + s(n) for odd n and n − s(n) for even n, where s is the
decimal digit sum.

## Requirements

Python ≥ 3.10 with `numpy`, plus `matplotlib` (figures only) and `pytest`
(tests only).

```bash
pip install numpy matplotlib pytest
```

## Quick start (from the directory that contains `digitdyn/`)

```bash
python3 -m pytest digitdyn/tests -q      # 50 tests, about 40 s
python3 -m digitdyn.reproduce            # every table and figure, about 10-15 min
```

## What each file supports

| File | Manuscript content |
|---|---|
| `explore.py` | Orbit census up to 2·10⁶: convergence, cycle lengths, largest excursion, exhaustive check of the 2-cycle characterisation and the two-step identity |
| `count2.py` | Exact T(D) by digit dynamic programming (Table of 2-cycle densities); verified against enumeration for D ≤ 5 |
| `structure.py` | Exact checks of the s(9m) formula, the level equation, conditional reversibility, and prefix lifting (constant-prefix cycle counts) |
| `profile.py` | Low-map cycle profile c_L(σ) for L = 5 and the prefix-lifting predictions |
| `resonance.py` | Carry-resonance probabilities ρ(t) and the resonance approximation to T(D)/10^D |
| `excursion.py` | Largest excursion over 300 random starts per digit length, fixed seed `random.Random(0)` |
| `oscillation.py` | Peaks and troughs of the normalised 2-cycle density up to D ≈ 6·10⁶ (oscillation table) |
| `baseb.py` | The map in base b: odd bases reach 0; exact checks of the even-base structure theorem; 2-cycle counts T_b(D) by digit DP |
| `basedensity.py` | Exact 2-cycle densities in bases 2, 4, 6 (multi-base resonance figure) |
| `figures.py` | All data figures (PDF) |
| `certify_b.py` | Block certificate (computer-assisted proof of eventual periodicity below b^D): e.g. `python3 -m digitdyn.certify_b 10 30 7` proves it for all n < 10^30 (about 3.5 min) |
| `climbcert.py` | Climb certificate (memoised half-line orbits, no block arrays): `python3 -m digitdyn.climbcert 10 3000 8` proves eventual periodicity for all n < 10^341 (~20 s); `... 2 3000 24` for all n < 2^3024 (~30 s). Outputs in `data/climbcert_b*.txt` |
| `variance.py` | Exact carry-count variance V(t) = (11/9) sum q_i(1-q_i) (Lemma `lem:exactvar`), sharp local limit check, and window averages behind the oscillation law (Theorem `thm:osclaw`) |
| `basins.py` | Basin census: share of distinct cycles vs share of starting values by cycle length (`data/basins.json`) |
| `periodic_sample.py` | Sampled proportion of periodic points among random D-digit numbers |
| `scale.py` | Bad sets of the low maps at a fixed scale (scale-reduction lemma, Lemma `lem:scale`) |
| `certify.py` | Earlier, coarser certificate attempt (superseded by `certify_b.py`) |
| `tests/test_digitdyn.py` | Tests of every computable claim against brute force |
| `reproduce.py` | Runs everything above in order |

## Status of results

The manuscript separates **theorems** (proved), **computational statements**
(exact computations over a stated finite range) and **conjectures**. Global
eventual periodicity of f is a conjecture; the code checks it only for the
stated ranges.

## Archiving

Before publication, push this directory to a public repository and archive a
tagged release (for example on Zenodo) to obtain a DOI; then replace the
placeholder in the manuscript's "Data and code" section.

## Verifying the certificates

The computer-assisted part of the paper (Theorem `thm:certres`) is fully determined by
`certify_b.py` and `climbcert.py`. To check it independently:

1. `python3 -m pytest digitdyn/tests` checks both certificate methods against direct
   orbit enumeration on small parameter sets, including a case `(2,16,8)` where the
   certificate must fail.
2. `python3 -m digitdyn.climbcert 10 3000 8` and `python3 -m digitdyn.climbcert 2 3000 24`
   regenerate `data/climbcert_b10.txt` and `data/climbcert_b2.txt`.
3. `shasum -a 256 -c SHA256SUMS` confirms that the released sources and certificate
   outputs are the ones used for the paper.
