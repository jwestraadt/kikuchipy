# Phase 1 -- `sht-square-grid-transform`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_fft.py tests/test_indexing/test_spherical_grid.py tests/test_indexing/test_spherical_sht.py -n 0     # first run: warm the numba cache in one process
uv run pytest tests/test_indexing/test_spherical_fft.py tests/test_indexing/test_spherical_grid.py tests/test_indexing/test_spherical_sht.py -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pytest --cov=kikuchipy.indexing._spherical --cov-report=term-missing tests/test_indexing/test_spherical_fft.py tests/test_indexing/test_spherical_grid.py tests/test_indexing/test_spherical_sht.py
uv run pre-commit run --files <the new/changed .py files>
```

Required assertions (each is a named test):

FFT sizing
- `fast_size(n) >= max(1, n)` and `fast_size(n)` is 13-smooth for `n in range(0, 1101)`; `fast_size(n) == smallest 13-smooth >= n` on the same range (any counter-example is recorded below and pinned to a C++-generated fixture); `fast_size(n) == max(1, n)` for `n <= 16`; spot values `105, 175, 245, 315 -> themselves`, `109 -> 110`, `17 -> 18`; `fast_size(-1)` raises `ValueError`.
- `fast_bandwidths(16, 512)` contains 32, 38, 41, 53, 63, 68, 74, 88, 95, 113, 122, 123, 158 and not 55; every returned `bw` satisfies `fast_size(2*bw - 1) == 2*bw - 1`.

Grids
- `validate_dim`: `dim < 3` and even `dim` raise `ValueError`.
- `normals(dim, layout)` unit length to 1e-15; centre pixel is `(0, 0, 1)`; asymmetric axis probes: `normals(dim, "lambert")[j, i] == square_to_sphere(i/(dim-1), j/(dim-1))` for `(i, j) in {(1, 0), (0, 1), (dim-1, 1)}` (index along axis 1 moves x, along axis 0 moves y); `ring_number` equals Chebyshev distance.
- `ring_indices`: `offsets[0] == 0`, `offsets[y+1] - offsets[y] == max(1, 8y)`, `offsets[-1] == dim*dim`, every pixel appears exactly once in `flat`; `flat[offsets[y]]` is pixel `(dim//2, dim//2 + y)` and `flat[offsets[y] + 1]` is `(dim//2 + 1, dim//2 + y)`; the azimuths `atan2(y, x)` of the ring's normals start at 0 and increase by `2 pi/(8y)` per slot (mod `2 pi`, 1e-12).
- Legendre normals: `z == cos_latitudes[ring]` and azimuths identical to Lambert normals of the same `dim` (1e-14).
- `square_to_sphere(sphere_to_square(v)) == v` for random upper-hemisphere `v` (1e-13); `sphere_to_square(+z) == (0.5, 0.5)`; `_lambert2vector(x, y) == square_to_sphere((x+1)/2, (y+1)/2)` and `sphere_to_square(v) == (_vector2lambert(v)/sqrt(pi/2) + 1)/2` (1e-12, two separate tests).
- `lambert_cos_latitudes(dim)` equals `1 - (2y/(dim-1))^2` with `rtol=1e-15`; `legendre_cos_latitudes(dim)[0] == 1`, strictly decreasing, last `== 0.0`; roots equal a transcribed bisection (`legendre::roots`) to 1e-13 for `n in {5, 69, 385}`.
- Ring solid angles: `sum_y ring_solid_angles(dim, layout)[y] * N_phi(y) * (0.5 if y == Nt-1 else 1) * 2/n_grid_points(dim) == 1` (1e-12) for both layouts at `dim in {9, 33, 101, 201}`.
- Lambert pixel solid angles at `dim in {11, 21, 51, 101}`: positive; 8-fold symmetric; pole pixel `== 2/pi` within `rel 1e-2` (converges from above; the geodesic quad through the four diagonal corner images has area `4r^2` against the equal-area cell `2 pi r^2`); every pixel with `ring >= 1` within 6 % of `n_grid_points/2/(dim-1)^2`; `sum(interior) + sum(edge)/2 + sum(corner)/4 == dim*dim - 2*dim + 2` with `rtol=1e-6` (Mazonka's `arg(prod)` cancels catastrophically as pixels shrink; the measured error vs `dim` is recorded below).
- Weights: `sum(w_hat[k]) == 1` (`cbrt(eps)/64`) for every skip set `k`; `w_hat[k, k] == 0`; `w[k, y] == 4 pi w_hat[k, y]/max(1, 8y)`; Legendre: all `Nw` rows identical **and** `w_hat[1:-1] == leggauss(dim-2) weights (descending)[:-1]`, `w_hat[-1] == leggauss weight of the zero root / 2` to 1e-14 for `dim in {19, 35, 101, 201, 401}`; `quadrature_weights(201, "lambert")` succeeds and `quadrature_weights(401, "lambert")` raises `ValueError`; the smallest tripping Lambert `dim` (bracket 259-301 per the reviewers) is recorded below.

SHT
- **Analyze oracle**: sample the complex `Y_l^m` (`sph_harm_y(n=l, m=m, theta=arccos(z), phi=arctan2(y, x))`, keywords; self-check `sph_harm_y(1, 1, pi/2, 0).real == -0.5*sqrt(3/(2 pi))`) on `normals(dim, layout)` for `(l, m) in {(1,0),(2,1),(3,2),(3,3),(6,4),(9,5),(12,8),(15,12)}`; `analyze(north, south)` gives `alm[m, l] == 1` and all other entries `< tol`, with `tol = 1e-10` for Lambert (`bw 32`, `dim 65`) and for Legendre `m % 4 != 0` (`bw 68`, `dim 71`); the Legendre `m % 4 == 0` cases are recorded (expected `1e-9 .. 1e-6`, the Nyquist property) and asserted `< 1e-6`.
- **Synthesize oracle**: `alm` with a single unit entry at `[m, l]` for the same set; `synthesize(alm)` equals `Y_l^m` evaluated at the grid normals (north and south) to 1e-11 (Lambert) / 1e-10 (Legendre, `m % 4 != 0`).
- **Condon-Shortley confirmation** (signed): analysing `Re Y_l^m` for `l in 1..6`, `m in 0..l` on Legendre `bw 8`, `dim 51`: `alm[m, l] == +1.0` (`m == 0`) or `+0.5` (`m > 0`), imaginary part `< 1e-10`, `abs=1e-8`; a dropped phase gives `-0.5`. Gate: `pytest.importorskip("scipy", minversion="1.15")` on the top-level package (never on `scipy.special`, which has no `__version__`); the DoD requires this test to have *run* (not skipped) on the dev machine.
- Round trip per `square_sht.cpp:90-148`: Legendre default set with `max |Re/Im delta| < 5e-3`, `mean < 5e-5` (C++ tolerances, uniform(-1,1) draw as specified in `plan.md`); Lambert default set with scale-free `max |delta| / max |alm_in| < 1e-11` (reviewer-measured `5e-13` at bw 16, `1.5e-12` at bw 64; loose tolerances would let a wrong weight-set index through).
- `analyze(ones, ones)`: `alm[0, 0] == sqrt(4 pi)` -- Legendre `abs 1e-12`, Lambert (`dim <= 201`) `abs 1e-10`; all others `< 1e-10`.
- Numba DFT path vs `scipy.fft` path agree to 1e-12 for Legendre `bw in {16, 68, 128}` and Lambert `bw 32` (both paths forced on the same input); `.py_func` of every kernel equals the compiled result.
- `analyze(..., bandwidth=k)` bitwise equal (`np.array_equal`) to `analyze(...)[:k, :k]`.
- Real data (Ni master `[:, ::2, ::2]`, `dim 201`, `bw 100`, Lambert): `sum |alm[m, :]|^2` over `m % 4 != 0` `< 1e-25 * total` (reviewer-measured `7.6e-34`); `analyze(north, south)` odd-`(l+m)` power `== 0.0` exactly and `analyze(north, 0.5*south)` odd-`(l+m)` power `> 1e-3 * total`; determinations recorded: `alm[0, 0]/sqrt(4 pi)` vs the `lambert_solid_angles`-weighted mean, and Parseval `sum_l (|a_0l|^2 + 2 sum_{m>0} |a_ml|^2)` vs `4 pi <f^2>` (asserted only `rel < 1e-2` -- different quadratures of a non-band-limited uint8 image; guards factor-of-N normalisation errors).
- Timing baseline recorded below: `analyze` wall time for `(bw, layout, dim) in {(68, legendre, 71), (128, legendre, 131), (384, legendre, 387)}` after JIT (informational).

## Weekly
- `uv run pytest --weekly tests/test_indexing/test_spherical_sht.py`: full Legendre `bw 4..384` and Lambert `bw 4..128` round-trip sweeps.

## Manual
- Headers: every new file carries the kikuchipy GPL header; `_grid.py`, `_sht.py`, `_fft.py` carry the delimited EMSphInx notice (CMU/Lenthe, GPL-2.0-or-later conveyed under GPL-3.0-or-later, "changed by Johan Westraadt, 2026-08") listing the ported functions.
- Docstrings: numpydoc, types in signatures only, comment lines <= 72 chars, three import blocks; no new public names in `indexing/__init__.pyi`.
- Coverage of `src/kikuchipy/indexing/_spherical/` >= 95 % (kernels via `.py_func`).
- Adversarial review findings addressed or explicitly deferred with reason.
- Known limitation stated: the Ni real-data fixture is fully square-symmetric and north == south; axis conventions and the antisymmetric branch are locked by the synthetic probes above, not by real data.

## Definition of done
All Phase 1 boxes in `specs/roadmap.md` ticked, default suite green on Windows (this machine) with `-n 4` after a `-n 0` warm-up, weekly sweep run once locally and green, the Condon-Shortley test executed (not skipped), PR opened into fork `develop`; determination results (smallest tripping Lambert `dim`, Legendre `m % 4 == 0` errors, `fast_size` vs 13-smooth counter-examples, Lambert solid-angle sum error vs `dim`, DC/Parseval ratios, timing) recorded below. "PR merged" is tracked in the roadmap.

## Recorded results
(filled in during implementation)

- **`fast_size` vs 13-smooth counter-examples (2026-08-16)**: the count of `n` in
  `[0, 1100]` where `fast_size(n) != smallest 13-smooth >= n` is **12**, namely
  `n in range(757, 769)`, all of which return **770** where the smallest
  13-smooth size is **768**. This is EMSphInx' behaviour, not a transcription
  error: `EMSphInx/include/util/fft.hpp:438-491` (commit `60f3517`) was compiled
  verbatim with g++ 15.2.0 and tabulated for `n in [0, 1100]`; the Python port
  agrees with the C++ on **all 1101 values (0 mismatches)**. Cause: `768 = 2^8*3`
  has nine prime factors, while `maxIter = log2(log2(v2x)) = 3` for `v2x = 1024`
  limits the set-product iteration to products of at most `2^3 = 8` primes, so
  the best reachable candidate is `770 = 2*5*7*11`. Parity wins per `plan.md`:
  the 12 sizes are pinned in `tests/test_indexing/test_spherical_fft.py` as
  `EMSPHINX_COUNTEREXAMPLES` (asserted equal to the C++ value 770 by
  `test_fast_size_matches_emsphinx_on_the_counterexamples`) and skipped by
  `test_fast_size_is_the_smallest_13_smooth_size_not_smaller_than_n`. 770 is
  itself 13-smooth, so no transform is slowed. `fast_bandwidths(16, 512)` is
  bit-for-bit unaffected: its 66 bandwidths are identical to a brute-force
  13-smooth oracle, because the only bandwidths whose `2*bw - 1` falls in
  `[757, 768]` are `bw in 379..384` and the target 768 is even, so the odd
  `2*bw - 1` could never equal it under either rule. The 66 contain all 20
  `nml.hpp` (lines 298 and 415) bandwidths and not 55.

### 2026-08-16 -- amended tolerances (adversarial reference measurements)

A faithful NumPy reference of `EMSphInx/include/sht/square_sht.hpp` was built and
every numeric assertion of the Phase 1 tests measured against it. The tolerances
below are amended; each entry gives the amended tolerance and the measurement
that motivates it. No other spec text is changed.

- Lambert round trip (`test_lambert_round_trip_is_scale_free_accurate` and the
  weekly `test_lambert_round_trip_sweep`): the scale-free
  `max |delta| / max |alm_in| < 1e-11` is reachable only for `dim <= 129`.
  Measured 3.9e-13 (bw 16), 7.9e-13 (bw 64), 3.6e-9 (bw 100, `dim` 201) and
  2.9e-6 (bw 128, `dim` 257). The bound is now `dim`-dependent: **1e-11 for
  `dim <= 129`, 1e-8 for `dim <= 201`, 1e-5 above** -- measured, then pinned with
  roughly one order of magnitude of margin. The weekly sweep now uses the same
  bound instead of the C++ round-trip tolerances.
- Legendre analyze oracle, `m % 4 != 0`
  (`test_legendre_analyze_returns_one_for_a_single_harmonic`): the "other
  entries" bound is raised from 1e-10 to **1e-8**. Worst measured other entry
  8.6e-10, at `(l, m) = (9, 5)` in `(m, l) = (5, 67)`, from the
  `m_lim = min(bw, 4y + 1)` truncation on ring 1. The diagonal keeps 1e-10.
  Lambert measured 2.7e-11 at `(12, 8)` and keeps 1e-10 for both checks.
- Legendre analyze oracle, `m % 4 == 0` (the Nyquist ring case, renamed
  `test_legendre_analyze_nyquist_orders_stay_below_1e_10`): **tightened from 1e-6
  to 1e-10** for the diagonal and the other entries alike. The defect is
  quadratic in `analyze`, so at bw 68, `dim` 71 the measured diagonal error is
  1e-16 .. 6e-16 and the worst other entry 2.8e-12.
- Lambert `analyze(ones, ones)` (`test_lambert_analyze_of_one_gives_sqrt_four_pi`):
  the "others" bound is relaxed from 1e-10 to **1e-9 for the `dim` 201 case
  only** (measured 5.98e-11, a 1.7x margin against 1e-10). `dim` 65 keeps 1e-10,
  and `alm[0, 0]` keeps `abs 1e-10` on both.
- Legendre weights vs `leggauss`
  (`test_legendre_weights_are_gauss_legendre_with_a_halved_equator`): the
  relative tolerances are replaced by **absolute** ones, since the smallest
  Gauss-Legendre weights at `dim` 401 are themselves of order 1e-5. Measured
  worst differences at `dim` 401: 9.1e-15 for the bulk weights (now `rtol=0`,
  `atol=1e-13`) and 1.35e-15 for the halved equator weight (now `abs=1e-14`).
- Synthesize oracle: the compared pixels are masked to the rings which carry the
  order, `4 * y >= m`, because `synthesize()` writes order `m` only where
  `m < m_lim(y) = min(bw, 4y + 1)`; the Nyquist ring `y == m // 4` of orders
  `m % 4 == 0` stays excluded on top of that, and the mask is now applied on the
  Legendre layout too. With the extended mask all eight `(l, m)` pairs agree to
  1e-16 .. 7e-15 on both layouts, well inside the unchanged 1e-11 (Lambert) and
  1e-10 (Legendre) bounds.

Assertions added or sharpened at the same time, none of which relaxes a spec
tolerance: every Lambert weight set must reproduce the Chebyshev moment system it
is solved from, `A[j, i] = T_j(2 x_i^2 - 1)` against
`b_j = int_0^1 T_j(2 x^2 - 1) dx = 1, -1/(4 j^2 - 1)`, to `atol` 1e-10 for `dim`
in {33, 65}; the dual-path tests patch `numba_ring_dft_max_dim` on the class
*before* constructing each transformer and assert the two report a different
`uses_numba_ring_dft` (a new property on `SphericalHarmonicTransform`); every
`.py_func` test first asserts that the kernel carries a `py_func`, so an
implementation without `@njit` fails loudly; the bw 384 timing case is
`@pytest.mark.weekly`; and all recorded determinations use the `record_property`
fixture instead of `print`.

### 2026-08-16 -- `_grid.py` implementation determinations

Measured on this machine (Windows 11, numpy/LAPACK of the uv-managed venv) with
`uv run pytest tests/test_indexing/test_spherical_grid.py -n 0`: **172 passed**.

- **Smallest tripping Lambert `dim` for the precision guard** (odd `dim` in
  259..301, guard `|sum(w_hat) - 1| > cbrt(eps)/64 = 9.462e-08`): **277**.
  259..275 all pass, 277..301 all fail. Worst `|sum(w_hat) - 1|` over the
  `(dim-2)//4 + 1` skip sets: 1.097e-11 (`dim` 201), 1.597e-08 (259),
  6.623e-08 (275), **1.238e-07 (277)**, 2.200e-06 (301), 1.079e-03 (401).
  This confirms the reviewers' 259-301 bracket and the constitution's
  "Lambert usable only for `dim <~ 275`".
- **Lambert solid-angle sum error vs `dim`** (`sum(interior) + sum(edge)/2 +
  sum(corner)/4` against `dim*dim - 2*dim + 2`), relative:

  | `dim` | sum | target | relative error |
  | --- | --- | --- | --- |
  | 11 | 101.000000000 | 101 | 7.1e-14 |
  | 21 | 401.000000000 | 401 | 7.6e-13 |
  | 51 | 2501.000000136 | 2501 | 5.4e-11 |
  | 101 | 10001.000007107 | 10001 | 7.1e-10 |
  | 201 | 40001.000515390 | 40001 | 1.3e-08 |

  The error grows as roughly `dim^4`, i.e. Mazonka's `arg(product)` cancels
  catastrophically as the pixels shrink; the spec's `rtol=1e-6` holds with a
  factor 80 of margin at `dim` 201.
- **Pole pixel value at `dim` 401**: `lambert_solid_angles(401)[200, 200] =
  0.636626153478`, i.e. `2/pi` (0.636619772368) plus 1.002e-05 relative --
  convergence from above, as `O(1/(dim-1)^2)`: 0.646211616050 (`dim` 11,
  +1.507e-02), 0.639009750756 (21, +3.754e-03), 0.637001812298 (51,
  +6.001e-04), 0.636715270827 (101, +1.500e-04), 0.636643654627 (201,
  +3.751e-05). Cross-checked against an independent spherical-excess area
  (sum of the four interior angles minus `2 pi`) of the same four corner
  images: agreement 1.5e-15 at `dim` 11 and 4.1e-07 at `dim` 401.

Test corrections required by the implementation (each is a test bug, not a
tolerance relaxation of a correct assertion):

1. `test_n_rings_and_n_grid_points_match_emsphinx`: the `dim` 201 case was
   parametrised with `n_grid_points = 79402`, which contradicts the test's own
   second assertion `n_grid_points == 2*dim*dim - 4*(dim-1) = 80002`. Corrected
   to 80002.
2. `test_lambert_cos_latitudes_equal_the_closed_form`: `rtol` 1e-15 -> 1e-14.
   The spec *mandates* the integer recursion (one rounding); the closed form
   `1 - (2y/(dim-1))^2` rounds three times and cancels near the equator, where
   it is 11 ulp (1.9e-15 relative, `dim` 101 and 201) to 25 ulp (4.3e-15,
   `dim` 401) away from the exactly rounded value the recursion produces. No
   implementation of the mandated recursion can meet 1e-15 here.
3. `test_lambert_pole_pixel_solid_angle_converges_to_two_over_pi`: `rel` 1e-2 ->
   2e-2, because the exact geodesic quadrilateral at `dim` 11 is 0.646212, i.e.
   1.507e-02 above `2/pi` (see the table above); 1e-2 is unattainable at
   `dim` 11 for any correct implementation. `dim` 21, 51 and 101 stay well
   inside 1e-2.
4. `test_weights_are_scaled_by_four_pi_over_the_ring_point_count`: for the
   Legendre layout the expected unscaled set is now the `skip = 0` one for
   every row. The test contradicted both the spec ("Legendre: solve `skip = 0`
   once and replicate") and `test_legendre_weight_sets_are_all_the_skip_zero_set`
   in the same class: the skip-`k` Legendre systems have genuinely different
   solutions (measured `max |w_k - w_0|` = 0.129 at `dim` 9 and 0.0075 at
   `dim` 33, `skip = 1`), so the two assertions could not both hold.

### 2026-08-16 -- `_sht.py` implementation determinations

Measured on this machine (Windows 11, uv-managed venv) with
`uv run pytest tests/test_indexing/test_spherical_sht.py -n 0`: **106 passed**
(3.2 s). Trio (`_fft`, `_grid`, `_sht`): **315 passed** with `-n 0` (11.8 s) and
with `-n 4` (67.3 s). Weekly (`--weekly ... -n 4`): **613 passed in 206.7 s**
(3 min 27 s). `--doctest-modules src/kikuchipy/indexing/_spherical`: 2 passed.

- **`analyze` timing baseline after JIT** (best of five, single thread, idle
  machine; the `record_property` values from the test run, which is one call
  under `-n 4` load, are given in brackets):

  | `bw`, layout, `dim` | path | `analyze` |
  | --- | --- | --- |
  | 68, legendre, 71 | numba ring DFT | **0.38 ms** [0.4-0.9 ms] |
  | 128, legendre, 131 | numba ring DFT | **2.61 ms** [2.7-3.7 ms] |
  | 384, legendre, 387 | `scipy.fft` | **28.8 ms** [110 ms] |

  The `dim` 387 case is on the `scipy.fft` path because
  `numba_ring_dft_max_dim` is 131.
- **Legendre `m % 4 == 0` analyze oracle errors** (bw 68, `dim` 71, the
  Nyquist-ring property), `|alm[m, l] - 1|` and the worst other entry:

  | `(l, m)` | diagonal | worst other |
  | --- | --- | --- |
  | (1, 0) | 8.882e-16 | 2.002e-15 |
  | (6, 4) | 1.670e-16 | 1.145e-15 |
  | (12, 8) | 1.620e-16 | **2.825e-12** |
  | (15, 12) | 4.596e-16 | 8.572e-16 |

  Both columns are far inside the amended 1e-10 bound, confirming that the
  defect is quadratic in `analyze`.
- **Lambert scale-free round trip** `max |delta| / max |alm_in|`, which
  reproduces the reviewers' reference measurements to the quoted digits:
  **3.942e-13** (bw 16, `dim` 33), **7.846e-13** (bw 64, `dim` 129),
  **3.568e-09** (bw 100, `dim` 201) and **2.899e-06** (bw 128, `dim` 257);
  mean errors 3.8e-15, 4.5e-15, 2.7e-11 and 1.9e-08. Legendre round trip
  (EMSphInx tolerances 5e-3 max / 5e-5 mean): max 9.3e-04 / mean 3.6e-06
  (bw 68), 5.7e-04 / 1.2e-06 (bw 128), 3.9e-04 / 6.5e-07 (bw 158) and
  6.6e-04 / 2.7e-07 (bw 384).
- **Ni master pattern** (`[:, ::2, ::2]`, `dim` 201, bw 100, Lambert):
  relative power of the `m % 4 != 0` orders **5.658e-32**; odd-`(l + m)` power
  exactly 0.0 for equal hemispheres and **1.015e-01** of the total for
  `analyze(north, 0.5 * south)`.
  - **DC**: `alm[0, 0] / sqrt(4 pi) = -1886.942645014` against the
    `lambert_solid_angles`-weighted mean **43.924569969**. On the centred
    `dim` 101 sub grid (bw 50) the same determination is
    **44.145295117 vs 44.224670475**, i.e. 1.8e-03 relative.
  - **Parseval**: `sum_l (|a_0l|^2 + 2 sum_(m>0) |a_ml|^2) = 8.459978e+10`
    against `4 pi <f^2> = 3.463972e+04` of the raw uint8 image, and against
    `4 pi <f_bl^2> = 8.459421e+10` of the band-limited function the
    coefficients represent, i.e. **6.6e-05 relative**. On the `dim` 101 sub
    grid the band-limited ratio is 1.000237 and the raw ratio 0.808937 (at
    `dim` 51, 0.718232).

**Determination: the Lambert ring quadrature amplifies out-of-band content
long before the `sum(w_hat)` guard trips.** The exact `dim` 201 Lambert weight
set (`skip = 0`) oscillates: `max |w_hat|` grows as 3.925e-02 (`dim` 101),
4.980e-02 (121), 3.188e-01 (141), 2.597e+00 (161), 2.321e+01 (181),
**2.218e+02 (201)**, 2.348e+04 (241), while `sum(w_hat) - 1` stays at
5.5e-13 at `dim` 201 and the condition number of the Chebyshev-Vandermonde
system is only 1.348e+08 there. The `dim` 201 weights were re-solved with
`mpmath` at 60 decimal digits: `max |w_hat| = 221.75969605`, agreeing with the
float64 solve to 2.4e-07 absolute, so the oscillation is the exact solution and
not round-off. Consequence: a band-limited function round trips to 3.6e-09
(above), but the non-band-limited uint8 master pattern has its out-of-band
content amplified by ~1e3 in `analyze`, which is why its DC comes out as
-1886.9 instead of 43.9. This sharpens the constitution's "Lambert usable only
for `dim <~ 275`" (the `sum(w_hat)` guard) to: **for non-band-limited input the
Lambert layout is only trustworthy up to `dim ~ 141`**, one more argument for
Phase 2's `toLegendre` regridding.

Test corrections required by the implementation (both in
`TestNickelMasterPattern`; each is an unmeasured test expectation, not a
tolerance relaxation of a correct assertion):

1. `test_dc_coefficient_matches_the_weighted_mean_at_dim_101` (renamed after review): the `rel=1e-2`
   guard was asserted on the `dim` 201 transform, where the amplification above
   makes the DC -1886.9 against a weighted mean of 43.9 (a sign flip, so no
   tolerance can express the intended "factor-of-N normalisation" guard). The
   `dim` 201 determination is still recorded; the guard is now asserted on the
   centred `dim` 101 sub grid (`max |w_hat| = 0.039`), where it holds to
   1.8e-03.
2. `test_parseval_sum_matches_the_band_limited_mean_square` (renamed after review): the reference
   `4 pi <f^2>` of the raw uint8 image cannot be matched to 1 % at any `dim`,
   because the master pattern is not band limited -- 28 % (`dim` 51) and 19 %
   (`dim` 101) of its power sits above the band limit, before the `dim` 201
   amplification adds six orders of magnitude. The raw determination is still
   recorded; Parseval is now asserted against `4 pi <f_bl^2>` of
   `synthesize(analyze(f))`, measured with the independent Mazonka per-pixel
   solid angles, which holds to 6.6e-05 at `dim` 201 (2.4e-04 at `dim` 101) and
   still guards every `4 pi`, `1 / N_phi(y)` and `sqrt(4 pi)` factor.

No other test was touched and no tolerance of a correct assertion was relaxed.

### 2026-08-16 -- adversarial review of the implementation (3 reviewers + bug injection)

- Fidelity: the reviewer compiled EMSphInx's own `square_sht.hpp`/`fft.hpp` (g++ 15.2) and compared numerically -- **no divergence** in grids, ring order, weights, `analyze`, `synthesize` or `fast_size`. The two-sided precision guard (deliberate deviation) moves the first tripping Lambert `dim` from 285 (C++, signed test) to 277 (port); pinned by two new tests.
- Bug injection: 21 mutations (incl. dropped `(-1)^m` in analyze/synthesize, weight set 0 for all m, clockwise rings, slot shift, transposed `alm`, missing `xN_phi`, reversed latitudes, `m_lim` off-by-one, north/south swap, unhalved equator weights, conjugated DFT table, dropped Nyquist real-only case) -- 17 killed by the original suite; the 4 survivors (one-sided guard, `cache=True`, `nogil=True`, `workers=1`) now have dedicated tests; all 21 killed.
- Conventions review: notice blocks corrected (`Namelist::sanityCheck()`, `square_sht.hpp:375-381`, literal "GPL-2.0-or-later"), `Raises` sections completed, uncovered numba branches exercised, path frozen at construction, timing test given real bounds, CRLF line endings normalised. Coverage 100 % (341 tests).
- Lambert weight oscillation: at `dim` 201 the exact Sneeuw `skip = 0` weight set has `max |w_hat| = 221.8` (mpmath 60-digit re-solve agrees to 2.4e-7), i.e. the interpolatory rule amplifies out-of-band content of a non-band-limited uint8 image ~1e3x even though `sum(w_hat) - 1 = 5.5e-13`; band-limited inputs are integrated exactly. Consequence for Phase 2+: treat the Lambert layout as a *master-pattern container only*, never as the analysis grid for real images -- regrid to Legendre first.

