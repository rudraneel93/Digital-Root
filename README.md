# Digital Root: the digit-sum parity map

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

- Every orbit starting below **10^341** (base 10), and below **2^3024** in
  base 2, is eventually periodic. The proof uses the block and climb
  certificates, which are checked by two independent methods.

**Open**

- Whether *every* orbit is eventually periodic. The manuscript reduces this to
  a precise "quadratic climbs" conjecture.
- Whether all cycles, not only 2-cycles, have density zero. About 4–6% of
  numbers up to 512 digits are periodic, and 4-cycles are almost as common
  as 2-cycles.

## Repository layout

| Path | Contents |
| --- | --- |
| [`digitdyn/`](digitdyn/) | Python package reproducing every computation in the manuscript ([README](digitdyn/README.md)) |
| [`digitdyn/manuscript/`](digitdyn/manuscript/) | LaTeX source, PDF and figures |
| [`digitdyn/data/`](digitdyn/data/) | Data behind the tables and figures, including the certificate outputs |
| [`digitdyn/tests/`](digitdyn/tests/) | 57 tests checking theorems against brute force |
| [`Digital Root/`](Digital%20Root/) | The original exploratory project (2025): cycle tables, plots and an early write-up |

## Quick start

From the repository root:

```bash
pip install numpy matplotlib pytest
python3 -m pytest digitdyn/tests                 # 57 tests, about 15 s
python3 -m digitdyn.climbcert 10 3000 8          # certificate: all n < 10^341
python3 -m digitdyn.climbcert 2 3000 24          # certificate: all n < 2^3024
python3 -m digitdyn.reproduce                    # regenerate every table and figure
```

To confirm the released files are the ones used for the paper:

```bash
cd digitdyn && shasum -a 256 -c SHA256SUMS
```

## A note on the original `Digital Root/` folder

The early write-up in `Digital Root/` states two results that the later
analysis corrects:

- *"Every orbit reaches a multiple of 9 within two steps."* This is **false**.
  Every orbit does reach a multiple of 9 after its first even term, but this
  can take up to 14 steps.
- *"Every orbit converges to a cycle or 0."* This is **not proved**. It is
  the open conjecture above, verified for all starting values below 10^341.

The cycle census in that folder has also been recomputed: Tables 7 and 8 of
the manuscript replace the earlier counts.

## Citation

If you use this code or these results, please cite the manuscript:

```
R. Das, Carries, Rails and Resonance: the Structure of the Digit-Sum Parity Map,
manuscript (2026), https://github.com/rudraneel93/Digital-Root
```

An archived release with a DOI will be added.

## License

MIT, see [LICENSE](LICENSE).
