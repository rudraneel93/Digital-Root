# Digital Root: the digit-sum parity map

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23940129.svg)](https://doi.org/10.5281/zenodo.23940129)

Code, data and manuscript for the study of the **digit-sum parity map**

```
f(n) = n + s(n)   if n is odd
f(n) = n − s(n)   if n is even
```

where `s(n)` is the sum of the decimal digits of `n`. Example orbits:

```
5 → 10 → 9 → 18 → 9            (enters the 2-cycle {9, 18})
1791 → 1809 → … → 1800 → 1791    (a 15-cycle)
```

The main reference is the manuscript

> R. Das, *Carries, Rails and Resonance: the Structure of the Digit-Sum Parity Map*
> ([`digitdyn/manuscript/manuscript.pdf`](digitdyn/manuscript/manuscript.pdf)).

## Main results

The manuscript separates three kinds of statement: proved theorems,
exact computations over a stated range, and conjectures.

**Proved**

- **2-cycles via carries.** An odd `a` lies on a 2-cycle exactly when `s(a)` is
  odd and adding `s(a)` to `a` produces exactly `s(s(a))/9` carries.
- **Carry balance.** That required carry count is within 1/9 of the *average*
  number of carries.
- **Exact density of 2-cycles.** The number of 2-cycles below `N` is of exact
  order `N / sqrt(log log N)`, so 2-cycles have density zero, but only barely.
- **Oscillation law.** The normalised density `sqrt(log10 D) · T(D)/10^D` has
  no limit. Its limit points fill the interval `[K0·sqrt(726/301), K0·sqrt(6)]`
  with `K0 = 1/(12·sqrt(11π))`, so the peak-to-trough ratio is
  `sqrt(301)/11 ≈ 1.577`. For almost every `D` it is close to `K0·sqrt(3)`.
  Two new ingredients go into this: an exact formula for the variance of the
  carry count, `Var C = (11/9) Σ q_i(1 − q_i)`, and the sharp bound
  `Σ x_i(1 − x_i) ≤ (30/121)·n + 5/22`, attained by the digits `4545…`.
- **Structure.** Self-similarity on multiples of 9, `s(9m) = 9 × borrows(10m − m)`;
  a level equation for how the digit sum changes at each step; a conditional
  time-reversal identity; and a convergence criterion: an orbit is eventually
  periodic if and only if its digit sums stay bounded.
- **Other bases.** In odd bases every orbit reaches 0. In every even base,
  base 2 included, 2-cycles have density of exact order `(log D)^(−1/2)`.

**Proved by computer**

- Every orbit starting below **10^2224** (base 10), and below **2^3024** in
  base 2, is eventually periodic. The proof uses the block and climb
  certificates, which are checked by two independent methods.

**Open**

- Whether *every* orbit is eventually periodic. It would follow from a
  uniform quantitative strengthening, the "quadratic climbs" conjecture.
- Whether all cycles, not only 2-cycles, have density zero. About 4–6% of
  numbers up to 512 digits are periodic, and 4-cycles are almost as common
  as 2-cycles.

## Repository layout

| Path | Contents |
| --- | --- |
| [`digitdyn/`](digitdyn/) | Python package reproducing every computation in the manuscript ([README](digitdyn/README.md)) |
| [`digitdyn/manuscript/`](digitdyn/manuscript/) | LaTeX source, PDF and figures |
| [`digitdyn/data/`](digitdyn/data/) | Data behind the tables and figures, including the certificate outputs |
| [`digitdyn/tests/`](digitdyn/tests/) | 59 tests checking theorems against brute force |
| [`Digital Root/`](Digital%20Root/) | The original exploratory project (2025): cycle tables, plots and an early write-up |

## Quick start

From the repository root:

```bash
pip install numpy matplotlib pytest
python3 -m pytest digitdyn/tests                 # 59 tests, about 15 s
python3 -m digitdyn.climbcert_par 10 19944 8 8   # certificate: all n < 10^2224 (~2 min, 8 cores)
python3 digitdyn/verify_certificates.py 10 19944 8   # standalone check of the same proof
python3 -m digitdyn.climbcert 2 3000 24          # certificate: all n < 2^3024
python3 -m digitdyn.reproduce                    # regenerate every table and figure
```

To confirm the released files are the ones used for the paper:

```bash
cd digitdyn && shasum -a 256 -c SHA256SUMS
```

## A note on the original `Digital Root/` folder

`Digital Root/` holds the project's two earlier write-ups (June and September
2025) and the original README. The manuscript supersedes them and corrects
them in its introduction ("Earlier versions"):

- **Reaching a multiple of 9.** The June write-up conjectured that every
  orbit does this within five steps, and the old README stated it as a
  theorem with *two* steps. Both are **false**: n = 899457 needs 14 steps.
  It is true that every term after the first even term is a multiple of 9.
- **Convergence.** That every orbit converges to a cycle or 0 is a conjecture
  in the June write-up but a "theorem" in the old README. It is **not
  proved**: it is the open conjecture above, verified below 10^2224.
- **Cycle counts.** The September write-up reports 9,001 distinct cycles
  reached from the multiples of 9 up to 200,000, and suggests linear growth.
  Exact enumeration gives **4,149**, and the number of 2-cycles grows like
  N/sqrt(log log N). Tables 7 and 8 of the manuscript replace the earlier
  counts.

## Citation

If you use this code or these results, please cite the manuscript and the
archived software:

```
R. Das, Carries, Rails and Resonance: the Structure of the Digit-Sum Parity Map,
manuscript (2026), https://github.com/rudraneel93/Digital-Root

R. Das, digitdyn: code and data for "Carries, Rails and Resonance",
version v1.0.2, Zenodo (2026), https://doi.org/10.5281/zenodo.23940129
```

- **v1.0.2 exactly**, the version used in the manuscript:
  [10.5281/zenodo.23940129](https://doi.org/10.5281/zenodo.23940129)
- **All versions**, which always resolves to the latest:
  [10.5281/zenodo.23938784](https://doi.org/10.5281/zenodo.23938784)

The preprint itself is archived at
[10.5281/zenodo.23938983](https://doi.org/10.5281/zenodo.23938983) (all versions).

GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
