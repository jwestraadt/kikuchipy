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
