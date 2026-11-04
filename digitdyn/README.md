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
python3 -m pytest digitdyn/tests -q      # 59 tests, about 15 s
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
| `oscillation.py` | Peaks and troughs of the normalised 2-cycle density up to D ≈ 6·10⁷ (oscillation table) |
| `baseb.py` | The map in base b: odd bases reach 0; exact checks of the even-base structure theorem; 2-cycle counts T_b(D) by digit DP |
| `basedensity.py` | Exact 2-cycle densities in bases 2, 4, 6 (multi-base resonance figure) |
| `figures.py` | All data figures (PDF) |
| `certify_b.py` | Block certificate (computer-assisted proof of eventual periodicity below b^D): e.g. `python3 -m digitdyn.certify_b 10 30 7` proves it for all n < 10^30 (about 3.5 min) |
| `climbcert.py` | Climb certificate (memoised half-line orbits, no block arrays): `python3 -m digitdyn.climbcert 10 3000 8` checks σ ≤ 3000 (all n < 10^341, ~20 s); `... 2 3000 24` proves all n < 2^3024 (~30 s). Outputs in `data/climbcert_b*.txt` |
| `verify_certificates.py` | Standalone checker for the two proof computations (no package imports) |
| `climbcert_par.py` | Parallel climb certificate with the max climb per σ: `python3 -m digitdyn.climbcert_par 10 19944 8 8` proves all n < 10^2224 (~2 min on 8 cores); profile in `data/climb_profile_b10_S19944_L8.json` (Fig. `climb`) |
| `osc_sampling.py` | Sampled peak/trough ratios with standard errors (Remark `rem:slow`) |
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

## Archive

The release used for the paper is archived on Zenodo as v1.0.2,
[doi:10.5281/zenodo.23940129](https://doi.org/10.5281/zenodo.23940129)
(all versions: [10.5281/zenodo.23938784](https://doi.org/10.5281/zenodo.23938784)).
The preprint is [doi:10.5281/zenodo.23938983](https://doi.org/10.5281/zenodo.23938983).

## Verifying the certificates

The computer-assisted part of the paper (Theorem `thm:certres`) consists of exactly
two finite computations; everything else is a cross-check.

| Proof computation | Input `(b, σ_max, L)` | Output (highest point) | Proves |
| --- | --- | --- | --- |
| base 10 | `(10, 19944, 8)` | 20,020,006 < 10^8 | all n < 10^2224 eventually periodic |
| base 2 | `(2, 3000, 24)` | 2,107,392 < 2^24 | all n < 2^3024 eventually periodic |

To check it independently:

1. **Standalone checker** (93 lines, shares no code with the package):
   ```bash
   python3 digitdyn/verify_certificates.py 10 19944 8 --compare digitdyn/data/climb_profile_b10_S19944_L8.json
   python3 digitdyn/verify_certificates.py 2 3000 24
   python3 digitdyn/verify_certificates.py 2 8 8     # must report NOT VERIFIED
   ```
   The first command recomputes all 19,945 per-σ maxima and compares them with the
   stored profile (about 2 minutes on 8 cores).
2. **Cross-checks:** `python3 -m pytest digitdyn/tests` compares the block and climb
   methods with direct orbit enumeration on small parameter sets, including the case
   `(2,16,8)` where the certificate must fail.
3. **Integrity:** `shasum -a 256 -c SHA256SUMS` (inside `digitdyn/`) confirms that the
   sources, certificate outputs and manuscript are the ones used for the paper.
