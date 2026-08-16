# Phase 3 -- `sht-wigner-d`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_wigner.py tests/test_indexing/test_spherical_euler.py -n 0     # first run: warm the numba cache in one process
uv run pytest tests/test_indexing/test_spherical_wigner.py tests/test_indexing/test_spherical_euler.py -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pytest --cov=kikuchipy.indexing._spherical --cov=kikuchipy.data.emsphinx --cov-report=term-missing tests/test_indexing/test_spherical_wigner.py tests/test_indexing/test_spherical_euler.py
uv run pre-commit run --files <the new/changed .py files>
uv run sphinx-build -b html doc doc/_build/html     # exit 0; `fukushima2016wigner` renders on doc/_build/html/user/bibliography.html (`:all:`)
```

`eps` below is `numpy.finfo(float).eps = 2.220446049250313e-16`; "bitwise" means `==` on every defined slot (`numpy.array_equal` after masking the NaN slots, whose positions are asserted separately). Every "measured" value comes from the pre-implementation reference measurements recorded at the end of this file; the assertion tolerances are the C++ ones where the C++ has one, otherwise measured-then-pinned with margin.

Required assertions (each is a named test):

Reference tables (`kikuchipy.data.emsphinx.wigner_reference_tables`)
- Each of the seven dicts has exactly 165 keys, equal to `{(j, k, m): 0 <= j < 5, |k| <= j, |m| <= j}`; every value is finite; `as_array(table)` has shape `(5, 9, 9)`, NaN exactly on the 240 slots `j < max(|k|, |m|)`.
- Hand-checked entries: `D_PI_2[(0, 0, 0)] == 1.0`, `D_PI_2[(1, 1, 0)] == 0.7071067811865476`, `D_PI_2[(1, 0, 1)] == -0.7071067811865476`, `D_PI_2[(4, 4, 4)] == 0.0625`, `D_PI_3[(2, 2, 1)] == 0.649519052838329`, `D_PI_3[(2, 1, 0)] == 0.5303300858899106`, `D_2PI_3[(3, -2, 1)] == 0.2567449488305466`, `D_PRIME_PI_3[(2, 1, 0)] == -0.6123724356957945`, `D_PRIME2_PI_3[(2, 1, 0)] == -2.1213203435596424`, `D_3_2_1 == 0.19764235376052370824993084652704 - 0.34232659844072882091060611425050j` (Python float equality of the parsed literals).

Wigner d -- Mathematica tables (port of `wigner.cpp:112-394`, `testDjkm`)
- For all `j < 5`, `k, m in (-5, 5)` (`-4..4`): `wigner_d_half_pi(j, k, m)` and `wigner_d(j, k, m, 0.0, False)` vs `D_PI_2[j, k, m]`; `wigner_d(j, k, m, 0.5, False)` vs `D_PI_3[j, k, m]`; `wigner_d(j, k, m, 0.5, True)` vs `D_PI_3[j, m, k]` (`d^j_{k,m}(-beta) = d^j_{m,k}(beta)`); `wigner_d(j, k, m, -0.5, False)` vs `D_2PI_3[j, k, m]`; `wigner_d(j, k, m, -0.5, True)` vs `D_2PI_3[j, m, k]`; **`abs <= 2 eps`** (`wigner.cpp:287`; measured worst 1.1e-16 for the `pi/2` special case and 3.3e-16 for the general function -- a 1.33x margin, so the shared-helper/association rule of `requirements.md` is load-bearing); undefined entries: both NaN.
- `wigner_d_sign(j, k, m) * D_PI_2[j, |k|, |m|] == D_PI_2[j, k, m]` **exactly** on the 165 defined triples (`:342-370`; measured 0 mismatches); the 240 both-NaN slots are skipped, as `wigner.cpp:359` does (`nan == nan` is `False`), and their positions are asserted by the NaN-pattern test above.
- `wigner_D(3, 2, 1, (pi/3, pi/2, pi/6)) == D_3_2_1` to `abs <= 2 eps` (`:372-390`; measured 1.1e-16).

Wigner d -- symmetries, edge cases, high degree
- Identities of `wigner.hpp:301-315` on 200 random `(j, k, m, t)` (`j <= 40`, `|k|, |m| <= j`, `t` uniform in `(-1, 1)`), each ~~**exact**~~ **exact when `k != 0` and `m != 0`, otherwise `abs <= 8 eps`** (corrected 2026-08-16: the three identities marked below are exact only when both orders are non-zero -- see Recorded results): `wigner_d(j, k, m, t, True) == wigner_d(j, m, k, t, False)`; `wigner_d(j, -k, -m, t, False) == (-1)^(k-m) wigner_d(j, k, m, t, False)`; `wigner_d(j, k, -m, t, False) == (-1)^(j+k) wigner_d(j, k, m, -t, False)`; `wigner_d(j, -k, m, t, False) == (-1)^(j+m) wigner_d(j, k, m, -t, False)`; `wigner_d(j, m, k, t, False) == (-1)^(k-m) wigner_d(j, k, m, t, False)`; `wigner_d(j, k, m, t, nB)` is NaN iff `j < max(|k|, |m|)` (also for `wigner_d_half_pi`, `wigner_d_prime`, `wigner_d_prime2`, and for `j < 0`).
- `t = +1`: `wigner_d(j, k, m, 1.0, nB) == (k == m)` to `abs 1e-14` for `j <= 8`, both `nB` (`0 ** 0 = 1`; the recursion reproduces `P_l(1) = 1` only to rounding: measured 1.0e-15), no NaN; `t = -1`: `wigner_d(j, k, m, -1.0, False) == (-1)^(j+k) (m == -k)` to `abs 1e-14` (measured 1.0e-15); `wigner_d_table(8, +-1.0, False)` has no NaN in a defined slot.
- Unitarity `sum_{m=-j}^{j} wigner_d(j, k, m, t, False)^2 == 1` to `abs 1e-12` for `j in {15, 63, 127, 511}`, `k in {0, j // 2, j}`, `t = cos(0.9708055194)` (measured `<= 3.1e-14` at `j = 511`).
- Closed form (Fukushima eq. 1 with `scipy.special.eval_jacobi` and `gammaln`; no version gate) for `(j, k, m) in {(1, 1, 0), (2, 2, 1), (5, 3, 1), (15, 10, 4), (63, 40, 20), (127, 100, 90), (300, 250, 200), (511, 0, 0), (511, 511, 0)}` at `beta in {0.9708055194, 2.5}` and `(511, 400, 300)` at `0.9708055194`: `rel <= 1e-10` (measured worst 6.4e-13; the values span 1e-115 .. 1e-1); int64 sanity: all finite for `j = 511`.
- **Pinned underflow limitation** (`requirements.md`, Scope): `wigner_d(511, 400, 300, cos(2.5), False) == 0.0` exactly, with the true value `-1.209e-184` (closed form and `mpmath`) in the test's comment -- the seed `c2 ** (k+m)` underflows (`c2 = cos(1.25)`, `c2 ** 700 == 0.0`), so the recursion returns exactly zero; the test exists so that a later phase raising the bandwidth sees the limitation named rather than re-tuning the closed-form tolerance around it. Weekly: the closed-form scan below.

Wigner d -- tables (port of `wigner.cpp:405-555`, `testTables(15)`)
- `beta = 0.9708055194` (`pi/2/phi`, `:412`), `t = cos(beta)`, `bw in {1, 2, 3, 15, 32}` (68, 88, 113 weekly): `wigner_d_table(bw, t, nB)[k, m, j, 0] == wigner_d(j, k, m, t, nB)` and `[k, m, j, 1] == wigner_d(j, k, m, -t, nB)` **bitwise** for both `nB` on the defined slots (`:416-461`; measured 0 mismatches for both `nB` at `bw` 15 (both the pure-Python and the numba transcription), 32 (11440 defined slots) and 68 (107134 defined slots) -- so the default-suite bandwidths are all measured, and 88/113 are extrapolations checked weekly; the same section warns that one re-association breaks this at 1.7e-16, so the assertion is deliberately zero-margin); NaN exactly on `j < max(k, m)`.
- `wigner_d_table_pre(bw, t, nB, *wigner_d_table_factors(bw))` **bitwise** equal to `wigner_d_table(bw, t, nB)` for both `nB` (`:463-513`), NaN exactly on `j < max(k, m)`; with `out=` given (a `np.full(nan)` buffer, and a buffer from a previous call at a different `t`), the same array object is returned, its contents are bitwise equal and its NaN pattern is exactly `j < max(k, m)`; a wrong `out` shape, dtype or a non-C-contiguous view raises `ValueError`, and so does `out=np.empty((bw, bw, bw, 2))` (representative-slot tripwire, `requirements.md`; `bw` 15 and `bw` 2).
- `wigner_d_table_factors(bw)`: `e_km[k, m] == wigner_d_half_pi(k, k, m) * 2 ** k` **exactly** for `m <= k < 20` (`d^k_{k,m}(pi/2) = 2^-k e_km`, `wigner.hpp:281`; power-of-two scaling is exact, measured exact); `w_jkm[k, m, i] == 1 / (sqrt((i+k)(i-k)(i+m)(i-m)) (i-1))` and `b_jkm[k, m, i] == w_jkm[k, m, i] * (sqrt((i+k-1)(i-k-1)(i+m-1)(i-m-1)) * i)` bitwise for `m <= k`, `i >= k + 2`; NaN elsewhere.
- `wigner_d_half_pi_table(bw, transpose=False)[k, m, j] == wigner_d_half_pi(j, k, m)` and `wigner_d_half_pi_table(bw, transpose=True)[m, k, j] == wigner_d_half_pi(j, k, m)` **bitwise** (`:515-552`); `transpose=True` result `== transpose=False result .transpose(1, 0, 2)` bitwise; `table[k, m, j] == (-1)^(k-m) table[m, k, j]` exactly; NaN exactly on `j < max(k, m)` for both `transpose` values (all three table constructors pin the same NaN pattern).
- `wigner_d_half_pi_table(15, False) == wigner_d_table(15, 0.0, False)[..., 0]` to `abs 1e-15` (measured 7.2e-16, not bitwise: different seed formula) and `wigner_d_table(15, 0.0, False)[..., 0] == [..., 1]` bitwise (`pi - pi/2 = pi/2`).
- `negative_beta` swap: `wigner_d_table(bw, t, True)[k, m, j, s] == wigner_d_table(bw, t, False)[m, k, j, s]` bitwise.
- `bandwidth < 1` and `|cos_beta| > 1` raise `ValueError`.

`rotate_harmonics` (no C++ unit test exists, `wigner.cpp:77`)
- Random `alm` (`np.random.default_rng(0).uniform(-1, 1)` real and imaginary parts, `l >= m`, `m == 0` real) at `bw in {8, 16, 32}`; random `zyz` (`alpha, gamma in [-pi, pi)`, `beta in [-pi, pi]`).
- Identity: `rotate_harmonics(alm, (0, 0, 0)) == alm` to `abs 1e-14` (measured 6.5e-16); pure z: `rotate_harmonics(alm, (a, 0, g))[m, l] == alm[m, l] exp(i m (a + g))` to `abs 1e-14`, also for `beta = -0.0`; glide `rotate_harmonics(alm, (a + pi, -b, g + pi)) == rotate_harmonics(alm, (a, b, g))` to `abs 1e-13` (measured 1.5e-15); **wrap** `rotate_harmonics(alm, (a, b + 2 pi, g)) == rotate_harmonics(alm, (a, b, g))` and `(a, 4.0, g) == (a, 4.0 - 2 pi, g)` to `abs 1e-13` (the recorded deviation; the unwrapped C++ differs by 2.2).
- Inverse: `rotate_harmonics(rotate_harmonics(alm, (a, b, g)), (-g, -b, -a)) == alm` to `abs 1e-13` (measured 7.9e-16); via the quaternion `quaternion_to_zyz((~Q).data)` likewise (4.3e-15).
- Composition: `rotate_harmonics(rotate_harmonics(alm, z1), z2) == rotate_harmonics(alm, quaternion_to_zyz((Q2 * Q1).data))` to `abs 1e-12` (measured 2.0e-15) **and** the other order `Q1 * Q2` differs by more than 0.1 (a guard against a symmetric mistake), for three random `(z1, z2)` pairs.
- Structure: per-degree power `|b^l_0|^2 + 2 sum_{m>0} |b^l_m|^2` equals that of `alm` to `rel 1e-12` (measured 5.3e-15); `blm[0, :]` imaginary part `<= 1e-14` (1.7e-16); entries of the input with `l < m` do not change the output (fill them with garbage first) and the output has them exactly zero; the output is a new array (input untouched); non-square or 1-D `alm` and a `zyz` of the wrong shape raise `ValueError`.
- **Brute-force `wigner_D` sum (dependency-free, runs on the CI "oldest" job)**: `rotate_harmonics(alm, zyz)[m, l] == sum_{n=-l}^{l} a^l_n wigner_D(l, m, n, zyz)` with `a^l_{-n} = (-1)^n conj(a^l_n)`, at `bw in {6, 8}` for `zyz in {(0.7, 1.1, -2.3), (-2.0, -0.6, 1.3), (0.4, pi, -1.0), (2.5, 0.0, 0.3)}` (positive, negative, `pi` and zero `beta`), `abs <= 1e-13` (measured worst 9.0e-16 at `bw` 8, 5.7e-16 at `bw` 6, 1.4e-15 for `beta = 0`); guard: the transposed sum `sum_n a^l_n wigner_D(l, n, m, zyz)` differs by more than 0.1 for the three `beta != 0` cases (measured 0.45-2.7; at `beta = 0` the two coincide, which is why that case is not a guard). This is the assertion that kills the transposed `dBeta[n, m, j]` read of the kernel (an inner automorphism, invisible to every identity above and to the Ni fixture) when the scipy oracle below is skipped.
- **Direction oracle** (`pytest.importorskip("scipy", minversion="1.15")` for `sph_harm_y`): with `f(n) = sum_l sum_m [alm]` evaluated by `sph_harm_y` (real function, `m > 0` doubled), `bw` 16, 200 random unit directions `n`, `R = Rotation(zyz_to_quaternion(zyz))`: `eval(rotate_harmonics(alm, zyz), n) == eval(alm, ((~R) * Vector3d(n)).data)` and `== eval(alm, n @ R.to_matrix()[0])` (i.e. `A n` with `A = R.to_matrix().T`) to `abs 1e-12` (measured ~~1.9e-14~~ 5.3e-14 / 2.9e-13 / 1.6e-13 for the three `zyz` of the test; corrected 2026-08-16: 1.9e-14 was the first case only, the thinnest margin is 3.4x -- see Recorded results); the wrong direction `eval(alm, (R * Vector3d(n)).data)` differs by more than 1 (guard, ~~all three~~ asserted on the two `zyz` with `beta != pi` only; corrected 2026-08-16: `(0.4, pi, -1.0)` is an involution, `w == 0.0`, so `R * n == (~R) * n` bit for bit and the guard is vacuous there).
- **Through Phase 1's transform** (same `importorskip`): `sht = SphericalHarmonicTransform(32, "lambert")` (`dim` 65), `alm` random at `bw` 16 zero-padded to 32, `north, south = sht.synthesize(rotate_harmonics(alm, zyz))` equals `eval(alm, ((~R) * Vector3d(normals)).data)` on `_grid.normals(65, "lambert")` (north) and its southern mirror (`z` negated) **on the pixels with `_grid.ring_number(65) >= 4`** (i.e. `4 y >= 15`, the highest order present -- Phase 1's Lambert `synthesize` writes order `m` on ring `y` only for `m < min(bw, 4y + 1)`, so rings 1-3 are not pointwise evaluations of the series -- measured 9.8e-4 / 6.5e-7 / 2.3e-10 on rings 1 / 2 / 3, so an unmasked whole-grid comparison fails at 9.8e-4 (the drafting text's `abs 1e-10` conflated this with the round trip, which inverts the same truncation); the mask is the rule Phase 1's own synthesize oracle adopted) to **`abs 1e-11`** (measured 1.9e-13 north / 9.3e-14 south for `zyz = (0.7, 1.1, -2.3)`, 1.7e-13 for `(-2.0, -0.6, 1.3)`, 3.5e-14 for `(0.4, pi, -1.0)`, of which 1.8e-13 is the `sph_harm_y`/orix pointwise evaluation itself over 4225 directions and 2.5e-14 the transform; function scale 13.9); guard: the wrong direction `eval(alm, (R * Vector3d(normals)).data)` differs by more than 1 on the same pixels (measured 14.8; ~~all three~~ asserted on the two `zyz` with `beta != pi` only -- corrected 2026-08-16, same involution reason as the bullet above). Weekly variant with `analyze` in the loop (no `sph_harm_y`, whole grid): `sht.analyze(*sht.synthesize(rotate_harmonics(alm, zyz)))` vs `rotate_harmonics(alm, zyz)` to `abs 1e-11` (measured 4.4e-15, 9.1e-15, 5.6e-15 for the three `zyz`; the round trip inverts the same per-ring truncation).
- Real data (Ni master, `kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")`, `mp.data[:, ::4, ::4]`, `dim` 101, `SphericalHarmonicTransform(50, "lambert", dim=101)`): relative change `||rotate_harmonics(alm, z) - alm|| / ||alm||` for the 4-fold about z (`z = (0, 0, pi/2)` and `(pi/2, 0, 0)`), the 2-fold about x (`quaternion_to_zyz(Rotation.from_axes_angles(Vector3d.xvector(), pi).data)`, which is the `beta = pi` branch: `(pi, pi, 0)`) and the 2-fold about `[110]` **`< 1e-12`** (measured 2.2e-15, 2.3e-15, 3.2e-15; exact grid symmetries; the `beta = pi` cases exercise table slot 1 at `t = -1`); the 3-fold about `[111]` (`from_axes_angles([1, 1, 1], 2 pi / 3)`) is **recorded** and asserted `< 0.2` (measured 7.92e-2 -- aliasing of the non-band-limited uint8 image, whose grid is not 3-fold invariant), while a 90 deg rotation about `[111]` (not a symmetry) is asserted `> 0.3` (measured 0.394) as the discriminating control.

Derivatives (port of `wigner.cpp:563-868`, `testDerivatives`, and Phase 7 pinning)
- For all `j < 5`, `k, m in (-5, 5)`: `wigner_d_prime(j, k, m, 0.5, False)` vs `D_PRIME_PI_3[j, k, m]`, `wigner_d_prime(j, k, m, -0.5, False)` vs `D_PRIME_2PI_3`, `wigner_d_prime2(...)` vs `D_PRIME2_PI_3`/`D_PRIME2_2PI_3`; negative beta: `wigner_d_prime(j, k, m, +-0.5, True) == (-1)^(|k|+|m|+1) D_PRIME_*[j, k, m]` and `wigner_d_prime2(j, k, m, +-0.5, True) == (-1)^(|k|+|m|) D_PRIME2_*[j, k, m]` (`:815-824`, `neg`/`-neg`); **`abs <= 24 eps`** (`:793`; measured worst 1.0e-15 for `d'` and 3.6e-15 for `d''`, i.e. 1.5x margin on `d''`); undefined: both NaN (skipped in the comparison as `wigner.cpp:828` does, positions asserted separately).
- `wigner_d_prime2(j, j, m, 0.5, False)` is **finite** for `(j, m) in {(0, 0), (1, 0), (2, 1), (3, 3)}` -- the slots whose C++ `d2Coef` radicand `(j-k-1)(j+k+2)` is `-2, -4, -6, -8` -- for the dispatcher **and** for `wigner_d_prime2.py_func` (which must not raise `ValueError`), values `0.0, -0.6123724356957947, -1.299038105676658, 0.421875` to `24 eps` (`requirements.md`, second deviation); likewise `wigner_d_prime.py_func(0, 0, 0, 0.5, False)` is finite.
- Finite differences (5-point stencils, `h = 1e-3`) for all `j < 15`, `|k|, |m| <= j`, `beta in {0.9708055194, -0.9708055194, 2.5, -2.5}`: `|fd1 - wigner_d_prime| <= 1e-7` (measured 4.2e-9, `max |d'| = 3.8`), `|fd2 - wigner_d_prime2| <= 1e-6` (measured 2.0e-8, `max |d''| = 49.8`).
- **Phase 7 formula pinning** (test-local transcription of `sht_xcorr.hpp:1009-1041`; `d1P`/`d1N` at `:1038-1039`, `d2P`/`d2N` at `:1040-1041`): with `table = wigner_d_table(15, t, nB)`, `csc = (-1 if nB else 1) / sqrt(1 - t^2)`, for all `m, n < 15`, `j >= max(m, n)`: `d1P == wigner_d_prime(j, m, n, t, nB)`, `d1N == (-1)^(j+m) wigner_d_prime(j, m, -n, t, nB)`, `d2P == wigner_d_prime2(j, m, n, t, nB)`, `d2N == (-1)^(j+m) wigner_d_prime2(j, m, -n, t, nB)` to `abs 1e-12` for `beta in {0.9708055194, -0.9708055194, 2.5, -2.5}` (measured at `bw` 15: worst 8.9e-16 / 3.1e-15 / 2.8e-14 / 5.7e-14 for `d1P` / `d1N` / `d2P` / `d2N`, the last at `beta = +-2.5`; 17x margin. The earlier `bw` 12 figure of 3.6e-14 is superseded).

Euler (`_euler.py`; port of `test/xtal/rotations.cpp:288-318`)
- `zyz_to_quaternion(zyz) == Rotation.from_euler(zyz_to_bunge(zyz)).data` to **`abs 1e-14`** on 1000 random triples (`rng(0)`, `alpha, gamma in [0, 2 pi)`, `beta in [0, pi]`; measured 8.6e-16), both with `w >= 0`; the wrong relation `(alpha - pi/2, beta, gamma + pi/2)` differs by more than 0.5 for at least one triple (guard; measured 2.0).
- Grid of `rotations.cpp:134-138` with `n = 25` (`phi = pi i / 24, i = 0..48`; `theta = phi[:25]`) and `n = 15`: `zyz_to_quaternion(bunge_to_zyz(eu))` vs the test-local `_emsphinx_eu2qu(eu)` (transcription of `rotations.hpp:410-429` incl. normalisation and `orientAxis`), `max |q - q'| <= 10 eps` component-wise (`:289`, measured 9.4e-16 at `n = 25`, 8.9e-16 at 15); `zyz_to_bunge(quaternion_to_zyz(_emsphinx_eu2qu(eu)))` re-converted with `_emsphinx_eu2qu` equals `_emsphinx_eu2qu(eu)` to `10 eps` (`euDelta`, `:313-323`; measured 1.03e-15); vs orix `Rotation.from_euler(eu).data` **as rotations** (`min(max|q - q'|, max|q + q'|) <= 10 eps`; measured 9.4e-16) with the number of sign-flipped degenerate cases recorded (`w = 0` cases where orix keeps the sign that `orientAxis` flips: 2402 of 60025 at `n = 25`, 814 of 12615 at `n = 15`).
- `_emsphinx_eu2qu(eu) == Rotation.from_euler(eu).data` **bitwise** on the 1000 random triples (measured 0.0), pinning `explore-emsphinx-xtal-util-vs-orix.md` 1.2.
- `quaternion_to_zyz(zyz_to_quaternion(zyz))`: re-converted quaternion equals the original to `abs 1e-14` (measured 8.9e-16) and the angles agree mod `2 pi` to `abs 1e-13`; ranges `[0, 2 pi) x [0, pi] x [0, 2 pi)`; degenerate (both branches of `qu2zyz`, 2000 random `(a, g)` pairs each): `beta = 0 -> ((a + g) % (2 pi), 0.0, 0.0)` and `beta = pi -> ((a - g) % (2 pi), pi, 0.0)` with **Python modulo** (into `[0, 2 pi)`, as the `+2 pi` wrap of `rotations.hpp:1019` does -- not `math.fmod`, which is negative whenever `a < g`: `fmod(1 - 2, 2 pi) = -1.0` where the port returns `5.283185307179586`), the first component to `abs 1e-14` measured against `exp(1j * alpha)` or mod `2 pi` (measured 8.9e-16 on both branches; `atan2` does not round-trip exactly, 978/2000 resp. 479/2000 pairs differ in the last bit), `beta` and `gamma` **exact** (`== 0.0` / `== pi`, `== 0.0`; measured exact in 2000/2000), `beta in {1e-9, 1e-7, pi - 1e-7, pi - 1e-9}` round trip to `abs 1e-14` in quaternion space (general branch, measured `<= 2.2e-16`); `q` and `-q` give the same angles; a `(..., 5)` input raises `ValueError`.
- `zyz_to_quaternion`: `w >= 0` for all inputs (also `signbit`-clean, i.e. never `-0.0`, on the random triples and the grid); pi rotations through `orientAxis` (`w` set to exactly `0.0`, all four components `signbit`-clean): `(0, pi, 0) -> (0, 0, 1, 0)`, `(pi/2, pi, pi/2) -> (0, 0, 1, 0)`, `(0.3, pi, 0.3) -> (0, 0, 1, 0)` (`+y` half of the equator), `(0, pi, pi) -> (0, 1, 0, 0)` and `(pi, pi, 0) -> (0, 1, 0, 0)` (`+x` rule when `y = z = 0`), asserted exactly (measured exact with the transcription); a `(..., 2)` input raises `ValueError`; shapes `(3,) -> (4,)`, `(n, 3) -> (n, 4)`, `(a, b, 3) -> (a, b, 4)`.
- `zyz_to_bunge((0, 0, 0)) == (pi/2, 0, -pi/2)`, `bunge_to_zyz(zyz_to_bunge(x)) == x` ~~exactly~~ to `abs <= 1e-15` with `beta` bitwise (corrected 2026-08-16: `(x + pi/2) - pi/2` differs from `x` for 38138 of 100000 random triples, worst 4.44e-16, so no "pure affine, no wrapping" implementation passes an exact assertion -- see Recorded results), no wrapping.
- `rotation_from_zyz(zyz).data == (~Rotation(zyz_to_quaternion(zyz))).data` exactly; `== (~Rotation.from_euler(zyz_to_bunge(zyz))).data` to `abs 1e-14`; `rotation_to_zyz(rotation_from_zyz(zyz)) == zyz` mod `2 pi` to `abs 1e-13`; shapes `(3,) -> Rotation (1,)`, `(n, 3) -> (n,)`, `(a, b, 3) -> (a, b)`; the docstring states the sign is provisional until Phase 5.
- `wrap_beta`: `4.0 -> 4 - 2 pi`, `-4.0 -> 2 pi - 4`, `7.0 -> 7 - 2 pi`, `pi -> pi`, `-pi -> -pi`, `0.0 -> 0.0`, `-0.0 -> -0.0` (`signbit`), `1.0 -> 1.0` (all to `abs 1e-15` or exact where stated).

Kernels and conventions
- Every `@njit` kernel of `_wigner.py` and `_euler.py` has a `py_func` (`hasattr` guard) whose result equals the compiled result **bitwise** on the `bw` 15 table, `wigner_d` on 50 random arguments, `rotate_harmonics` at `bw` 8, and 100 random Euler triples / quaternions; every kernel is compiled with `cache=True, nogil=True` and without `parallel`/`fastmath`, asserted exactly as Phase 1 does (`tests/test_indexing/test_spherical_sht.py`, `test_kernels_are_compiled_with_cache_and_nogil`): `kernel.targetoptions.get("nogil") is True`, `type(kernel._cache).__name__ == "FunctionCache"` (`cache` never appears in `targetoptions`; numba 0.65.1: `{'nogil': True, 'nopython': True, 'boundscheck': None}`), `not kernel.targetoptions.get("parallel", False)`, `not kernel.targetoptions.get("fastmath", False)`, parametrised over every kernel name of `_wigner.py` and `_euler.py`.
- Timing/memory baselines recorded via `record_property` (informational): `wigner_d_table` at `bw in {68, 88, 113}` (weekly: 158) **including its `np.full(nan)` allocation** (measured cost of the fill alone: 0.90 / 1.96 / 4.08 / 11.9 ms at 68 / 88 / 113 / 158, ~0.65x the recursion), `wigner_d_table_pre` with `out=` at the same (no allocation -- the pair backs the `out=` re-use argument with two numbers), the additive peak `w_jkm + b_jkm + table` (`4 bw^3` doubles: 10.1 / 21.8 / 46.2 MB) from `.nbytes`, `wigner_d_half_pi_table(bw, True)`, `wigner_d_table_factors`, `rotate_harmonics` at `bw in {68, 88}`, `zyz_to_quaternion` on 1e5 triples; table sizes in MB from `.nbytes`.

## Weekly
- `uv run pytest --weekly tests/test_indexing/test_spherical_wigner.py`: table vs scalar bitwise at `bw in {68, 88, 113}` for both `nB` and `beta in {0.9708055194, 0.3, 2.5, pi/2, 1e-3, pi - 1e-3}` (68 measured, 88/113 extrapolated -- record the result); the `analyze`-in-the-loop rotation oracle; the **closed-form underflow scan**: `wigner_d(j, k, m, cos(beta), False)` vs the `gammaln` + `eval_jacobi` closed form on the stride-7 grid `0 <= m <= k <= j`, `j = 511`, `beta in {2.5, 3.0, 0.9708055194}` and the stride-3 grid at `j = 127`, `beta in {2.5, 3.0, 3.13, pi - 1e-3}`: every entry the recursion returns as exactly `0.0` has closed-form `|d| < 1e-130` (measured largest 5.3e-143 at `(511, 343, 301)`, `beta` 2.5; 1.3e-139 at `(511, 175, 112)`, `beta` 3.0; 2.6e-276 at `(127, 60, 39)`, `beta` `pi - 1e-3`; none zeroed at `j = 127` for 2.5/3.0 or at `j = 511` for 0.9708), and every entry with closed-form `|d| >= 1e-100` agrees to `rel <= 1e-8` (measured worst 2.1e-10 at `(511, 70, 21)`, `beta` 2.5, and 3.1e-10 at `(127, 39, 0)`, `pi - 1e-3` -- the closed form's own `eval_jacobi` accuracy at these degrees is of the same order, so the bound is loose on purpose); timing at `bw` 158.

## Manual
- Headers: every new file carries the kikuchipy GPL header; `_wigner.py` and `_euler.py` carry the delimited EMSphInx notice (CMU/Lenthe, GPL-2.0-or-later conveyed under GPL-3.0-or-later, "changed by Johan Westraadt, 2026-08") listing the ported functions with line ranges (`_euler.py` also records that `zyz2eu`/`eu2zyz` were not ported as written); `wigner_reference_tables.py` carries the notice for `test/sht/wigner.cpp`.
- `src/kikuchipy/indexing/_spherical/__init__.py` lists ``_euler`` and ``_wigner`` in its `Submodules` block (alphabetical, one line each) and still imports nothing.
- `doc/user/bibliography.bib` parses and `fukushima2016wigner` renders on `doc/_build/html/user/bibliography.html` (Sphinx build exit 0); its DOI is a ResearchGate DOI (`10.13140/RG.2.2.31922.20160`, as `wigner.hpp:45-46` cites it), so a `-b linkcheck` complaint on that one URL is expected and allowed.
- Docstrings freeze the conventions listed in `requirements.md` (d/D definitions and the Mathematica/Wikipedia relation, `b^l_m` sum, `g(n) = f((~R) * n)`, composition order, table layouts, NaN slots incl. the caller-owned invariant of `out=`, beta wrap, memory formula and additive peaks, the underflow limitation, the two `.py_func`-driven deviations of the derivatives, provisional sign of `rotation_from_zyz`); numpydoc, types in signatures only, comment lines <= 72 chars, three import blocks; no new public names in `indexing/__init__.pyi`; no CHANGELOG entry.
- Coverage of `src/kikuchipy/indexing/_spherical/_wigner.py`, `_euler.py` and `src/kikuchipy/data/emsphinx/` >= 95 % (kernels via `.py_func`).
- Adversarial review findings addressed or explicitly deferred with reason; the bug-injection list of `plan.md` 4.1 fully killed.
- Known limitations stated: `rotate_harmonics` allocates `2 bw^3` doubles (906 MB at `bw` 384); the naive recursion zeroes `d^j_{k,m}` whose seed `c2^(k+m) s2^(k-m)` underflows (below ~1e-139 at `bw` 512, `beta >= 2.5`; nothing lost at `bw <= 128` for `beta <= 3.0`) -- pinned, not fixed; the through-Phase-1 oracle is masked to `ring_number >= 4` because Phase 1's Lambert `synthesize` is not pointwise on the inner rings; the Ni real-data fixture's 3-fold check is an aliasing-limited discrimination, not a precision test; the direction of `rotation_from_zyz` (the `~`) is provisional until Phase 5.

## Definition of done
All Phase 3 boxes in `specs/roadmap.md` ticked, default suite green on Windows (this machine) with `-n 4` after a `-n 0` warm-up, weekly run once locally and green, the `sph_harm_y` oracle tests executed (not skipped), `sphinx-build -b html` exit 0 with `fukushima2016wigner` rendered, PR opened into fork `develop`; determination results (worst Mathematica-table deltas for `d`, `d'`, `d''`, whether table == scalar is bitwise on this machine at every asserted `bw`, the sign-flip counts of the grid test, the Ni 3-fold/90-deg values, the through-Phase-1 per-ring errors, timings incl. the NaN fill and table sizes) recorded below. "PR merged" is tracked in the roadmap.

## Recorded results

### 2026-08-16 -- pre-implementation reference measurements (spec drafting)

A faithful pure-Python transcription of `wigner.hpp` (`d`, `d(pi/2)`, `dSign`,
`D`, `dTable`, `dTable(pi/2, trans)`, `rotateHarmonics`, `dPrime`, `dPrime2`)
and of `rotations.hpp` (`zyz2qu`, `qu2zyz`, `eu2qu`, `orientAxis`), plus a
numba version of `dTable`, were run against the Mathematica tables parsed from
`test/sht/wigner.cpp` and against orix 0.14.2 / numpy 2.4.6 / scipy 1.17.1 /
numba 0.65.1 on this machine. Every tolerance above is derived from these
numbers; they are to be re-measured with the real implementation and appended
below.

- Mathematica tables (`testDjkm`, `2 eps = 4.44e-16`): worst `|delta|` per case
  `[pi/2 special, pi/2, pi/3, -pi/3, 2pi/3, -2pi/3]` =
  `[1.11e-16, 3.33e-16, 3.33e-16, 3.33e-16, 3.33e-16, 3.33e-16]` -- pass, 1.33x
  margin. `dSign`: 0 mismatches. `D^3_{2,1}(pi/3, pi/2, pi/6)`: 1.11e-16.
- Derivative tables (`testDerivatives`, `24 eps = 5.33e-15`): worst per case
  `[d' pi/3, d' 2pi/3, d'' pi/3, d'' 2pi/3, and the four negative-beta twins]`
  = `[9.99e-16, 9.99e-16, 3.55e-15, 3.55e-15, 9.99e-16, 9.99e-16, 3.55e-15,
  3.55e-15]` -- pass, 1.5x margin on `d''`.
- Tables vs scalar at `beta = 0.9708055194`, `bw` 15: `dTable` bitwise equal to
  `d()` for `nB` false and true (0 mismatches, max diff 0.0); `dTable(pi/2)`
  bitwise equal to `d(j,k,m)` for both `trans`; `dTable(t = 0)[..., 0]` vs
  `dTable(pi/2)`: 7.2e-16 (not bitwise, different seed formula);
  `dTable(t = 0)[..., 0] == [..., 1]` bitwise. A numba transcription of `dTable`
  is bitwise equal to the Python one **only** when `b_jkm` is formed as
  `w * (sqrt(...) * j)`; the re-association `(w * sqrt(...)) * j` gives
  1.7e-16 differences (hence the shared-helper rule).
- `d'`/`d''` vs 5-point finite differences (`h = 1e-3`, `j < 15`, all `k, m`):
  worst 3.8e-9 / 1.75e-8 at `beta = +-0.9708055194`, 4.2e-9 / 1.96e-8 at
  `+-2.5` (`max |d'| = 3.8`, `max |d''| = 49.8`). With `h = 1e-4`: 1.4e-11 /
  1.8e-7; `h = 1e-5`: 8.7e-11 / 1.9e-5 (rounding dominates `d''`), so `h = 1e-3`
  is the pinned step.
- Phase 7 formula pinning at `bw` 12: `|d1P - d'(m, n)| <= 4.4e-16`,
  `|d1N - (-1)^(j+m) d'(m, -n)| <= 2.7e-15`, `|d2P - d''(m, n)| <= 2.1e-14`,
  `|d2N - (-1)^(j+m) d''(m, -n)| <= 3.6e-14` for `beta in {+-0.9708, +-2.5}`.
- High degree: closed form (`eval_jacobi`, `gammaln`) vs recursion, relative:
  2.1e-14 (300, 250, 200 @ 0.9708), 1.9e-14 (same @ 2.5), 5.7e-13 (511, 400,
  300 @ 0.9708), 3.4e-15 / 6.4e-14 (511, 0, 0 @ 0.9708 / 2.5), 6.1e-13 / 6.0e-13
  (511, 511, 0; values 4e-44 and 1.9e-115), 5.1e-14 / 7.8e-14 (200, 5, 3);
  `(511, 400, 300) @ 2.5`: the recursion returns 0.0 but the closed form does **not** underflow (`-1.209e-184`; the drafting note "underflows to 0.0 in both" was wrong -- corrected by the review re-measurement below). Unitarity
  `sum_m d^2 - 1`: 1e-15 (`j` 63), 8e-15 (127), 1.8e-14 (300), 3.1e-14
  (511, `k = j`).
- `rotateHarmonics` (`bw` 16, band-limited `bw` 8 input, `zyz = (0.7, 1.1,
  -2.3)`): identity 6.5e-16; direction `max |g(n) - f((~R) n)| = 1.9e-14`,
  `max |g(n) - f(R.to_matrix() n)| = 8.1`; `beta = 0` phase
  `exp(i m (a + g))`: 8.7e-15 (the opposite sign: 7.5); composition
  `rotate(rotate(a, z1), z2)` vs `rotate(a, zyz(q2 * q1))`: 2.0e-15, vs
  `zyz(q1 * q2)`: 2.4; inverse `(-g, -b, -a)`: 7.9e-16, conjugate quaternion:
  4.3e-15; glide `(a + pi, -b, g + pi)`: 1.5e-15; per-degree power: 5.3e-15;
  `m = 0` imaginary: 1.7e-16. Beta outside `[-pi, pi]`: `beta = 4.0` vs
  `4.0 - 2 pi` differ by 2.17 and `beta = -1` vs `-1 + 2 pi` by 2.0 in the
  unwrapped C++ algorithm (`beta = +pi` vs `-pi`: 0.0) -- the reason for the
  wrap. `t = +-1` and `beta = -0.0`: no NaN, `-0.0 == +0.0` result.
- Ni master (`dim` 101, `bw` 50, Lambert): relative change under the 4-fold
  about z 2.23e-15 (both `(0, 0, pi/2)` and `(pi/2, 0, 0)`; 2.30e-15 for
  `(0.3, 0, pi/2 - 0.3)`), 2-fold about x 2.29e-15, 2-fold about `[110]`
  3.18e-15, 3-fold about `[111]` **7.92e-2**, 90 deg about `[111]` (not a
  symmetry) **3.94e-1**. At `dim` 201 / `bw` 100 the exact symmetries hold to
  2.1e-14 but the 3-fold and the control both give 1.33 (the amplified
  out-of-band content of Phase 1's finding dominates), which is why the test
  uses `dim` 101.
- Euler: `zyz2qu(zyz)` vs `Rotation.from_euler((a + pi/2, b, g - pi/2)).data`
  8.6e-16 on 1000 random triples (both `w >= 0`), vs `(a - pi/2, b, g + pi/2)`
  2.0; `eu2qu` transcription vs orix `from_euler`: 0.0 (bitwise); `zyz2qu` vs
  `eu2qu((a + pi/2, b, g - pi/2))`: 8.6e-16. Grid test (`n = 25`, 60025 triples):
  `zyz2qu` vs `eu2qu` 9.4e-16, `qu2zyz` round trip (`euDelta`) 1.03e-15, vs
  orix component-wise 2.0 (2402 sign-flipped `w = 0` cases) but 9.4e-16 as
  rotations; `n = 15` (12615 triples): 8.9e-16 / 8.9e-16 / 814 flips. Round trip
  `qu2zyz(zyz2qu(z))`: 8.9e-16 (quaternion) and 8.9e-16 (angles mod `2 pi`);
  `beta = 0 -> ((a + g) % 2 pi, 0, 0)` and `beta = pi -> ((a - g) % 2 pi, pi, 0)`
  with `beta`, `gamma` exact and `alpha` to 8.9e-16 (the drafting note said
  "exact" for one `(a, g)` pair; the 2000-pair re-measurement below shows the
  last bit differs in about half of the pairs); `beta = 1e-9, 1e-7, pi - 1e-7`:
  general branch, `<= 2.2e-16`;
  orix `to_euler` route agrees with `qu2zyz` to 8.9e-16 away from degeneracy.
  EMSphInx `quat::mul` (`pijk = +1`) vs orix `Quaternion.__mul__`: 1.4e-17;
  `Rotation.to_matrix()` of `zyz2qu(zyz)` equals `(Rz(a) Ry(b) Rz(g))^T` to
  1.1e-16 and `Rotation * Vector3d` equals `to_matrix() @ v` (2.2e-16);
  `Rotation.from_euler(bunge).to_matrix()` equals the passive Bunge matrix
  `(Rz(phi1) Rx(Phi) Rz(phi2))^T` (2.2e-16); glide identity
  `zyz2qu(a, b, g) == zyz2qu(a + pi, -b, g + pi)`: 7.2e-16.
- Edges: `d(j, k, m, +-1)` vs `delta_{km}` / `(-1)^(j+k) delta_{m,-k}`: 1.0e-15
  (not exact), no NaN in the `t = +-1` tables; `e_km == d^k_{k,m}(pi/2) 2^k`
  exact for `k < 20`; `zyz2qu` pi rotations: `(0, pi, 0)`, `(pi/2, pi, pi/2)`,
  `(0.3, pi, 0.3) -> (0, 0, 1, 0)`; `(0, pi, pi)`, `(pi, pi, 0) -> (0, 1, 0, 0)`,
  all components `signbit`-clean.
- Numba baselines (this machine, best of 5, JIT excluded): `dTable` 1.39 ms
  (`bw` 68, 5.0 MB), 2.94 ms (88, 10.9 MB), 6.40 ms (113, 23.1 MB), 17.9 ms
  (158, 63.1 MB); `rotateHarmonics` loop 0.35 ms (68), 0.72 ms (88) on top of
  the table. Pure Python `dTable` at `bw` 68: 0.09 s (the reference is fast
  enough to be a test oracle at `bw` 15).

### 2026-08-16 -- adversarial review of the spec (2 critics) and re-measurements

Both critics transcribed `wigner.hpp` and `rotations.hpp` independently and
reproduced every number above; their findings and the re-measurements that
settle them (this machine, same library versions):

- **Through-Phase-1 oracle (blocker)**: the whole-grid comparison cannot pass.
  Lambert `synthesize` writes order `m` on ring `y` only for
  `m < min(bw, 4y + 1)`, so for `bw`-16 content on `dim` 65 the pointwise error
  vs `sph_harm_y` is, per ring (rotated by `(0.7, 1.1, -2.3)`, north / south):
  ring 0 7.8e-15 / 8.4e-15, ring 1 9.8e-4 / 4.0e-4, ring 2 6.5e-7 / 5.2e-7,
  ring 3 9.1e-11 / 2.3e-10, ring 4 1.3e-14 / 1.6e-14, and **1.9e-13 / 9.3e-14
  over all rings `>= 4`** (worst at ring 24, where the pointwise
  `|g(n) - f((~R) n)|` without any transform is already 1.8e-13 over the 4225
  grid directions; the transform's own contribution on rings `>= 4` is
  2.5e-14). Unrotated control: 8.3e-4 whole grid, 2.3e-14 rings `>= 4`. Content
  sweep at `dim` 65: `bw` 16 -> 8.3e-4, 8 -> 1.7e-5, 4 -> 1.7e-15; `dim` 129
  with `bw`-16 content: 4.6e-5 (the inner rings always have 8, 16, 24 samples).
  Other `zyz`: `(-2.0, -0.6, 1.3)` 1.7e-13, `(0.4, pi, -1.0)` 3.5e-14 on rings
  `>= 4`. Wrong direction on the same pixels: 14.8. `analyze` in the loop:
  4.4e-15 / 9.1e-15 / 5.6e-15. Frozen: mask `ring_number(65) >= 4`, `abs 1e-11`.
- **Brute-force `wigner_D` sum**: `|rotate - sum_n a_n D(l, m, n)|` = 5.7e-16
  (`bw` 6) and 9.0e-16 (`bw` 8) for `beta` 1.1 / -0.6 / `pi`, 1.4e-15 for
  `beta = 0`; the transposed sum `D(l, n, m)` differs by 0.61 / 0.45 / 2.7
  (`bw` 6) and 1.07 / 0.70 / 2.3 (`bw` 8), and by 0 at `beta = 0`. The critic's
  injection of the transposed `dBeta[n, m, j]` read survived identity, pure-z,
  glide, inverse, composition, power, `m = 0` and the Ni fixture (it is
  conjugation by the 2-fold about z); this sum and the `sph_harm_y` oracle are
  the only assertions that kill it.
- **`dPrime2` on `k == j` slots**: the C++ `d2Coef` radicand
  `(j-k-1)(j+k+2)` is `-2, -4, -6, -8` at `(0,0,0), (1,1,0), (2,2,1), (3,3,3)`;
  `math.sqrt` raises there in the `.py_func` (numba: NaN, C++: discarded by the
  ternary). Guarded transcription at `t = 0.5`: `0.0`, `-0.6123724356957947`,
  `-1.299038105676658`, `0.421875` (inside the `24 eps` table test).
- **`qu2zyz` degenerate branches**: `alpha` vs `(a + g) % 2pi` (`beta = 0`) and
  `(a - g) % 2pi` (`beta = pi`): worst 8.9e-16 on 2000 pairs each (978 resp. 479
  pairs differ in the last bit -- not exact); `beta`, `gamma` exact in
  2000/2000; `exp(i alpha)` form 9.6e-16 / 9.5e-16; `fmod(a - g, 2 pi)` is
  negative in 993/2000 pairs, `qu2zyz` returns the `+2 pi` value
  (`a = 1, g = 2 -> 5.283185307179586`).
- **Table vs scalar bitwise** at `beta = 0.9708055194`, extended: `bw` 32 both
  `nB` 0 mismatches on 11440 defined slots (NaN pattern exact); `bw` 68 both
  `nB` 0 mismatches on 107134 slots (4.0 / 4.7 s pure Python). 88 / 113 not
  measured (weekly).
- **Phase 7 pinning at `bw` 15** (the asserted bandwidth): `d1P` 8.9e-16, `d1N`
  2.3e-15 (0.9708) / 3.1e-15 (2.5), `d2P` 2.8e-14, `d2N` 2.8e-14 (0.9708) /
  5.7e-14 (2.5), identical for the negative betas.
- **High-degree underflow**: `d^511_{400,300}(2.5)` closed form
  `-1.209067e-184` (`mpmath` 50 digits: `-1.2090675e-184`), recursion `0.0`;
  `d^511_{511,0}(2.5)` 1.853566e-115 both (rel 6.7e-13); seed
  `c2 ** 610 = 1.7e-306`, `** 620 = 1.7e-311`, `** 700 = 0.0` at `beta` 2.5.
  Stride-7 scan at `j = 511` (2775 points): `beta` 2.5 zeroes 379 points,
  largest true `|d|` 5.3e-143 at `(343, 301)`; `beta` 3.0 zeroes 50, largest
  1.3e-139 at `(175, 112)`; `beta` 0.9708 zeroes none. Stride-3 at `j = 127`
  (946 points): 2.5 and 3.0 zero none, 3.13 zeroes 38 (largest 1.4e-280), `pi -
  1e-3` zeroes 20 (largest 2.6e-276). Worst relative agreement of entries with
  `|d| >= 1e-100`: 2.1e-10 (511, 70, 21 @ 2.5), 1.4e-12 (511 @ 3.0), 3.5e-12
  (511 @ 0.9708), 9.1e-13 / 7.3e-12 / 3.1e-10 / 4.6e-11 (127 @ 3.0 / 3.13 /
  `pi - 1e-3` / 2.5).
- **NaN fill cost**: `np.full((bw, bw, bw, 2), nan)` best of 7: 0.90 ms (`bw`
  68, 5.0 MB), 1.96 ms (88, 10.9 MB), 4.08 ms (113, 23.1 MB), 11.9 ms (158,
  63.1 MB) -- 0.65-0.67x of the recorded `dTable` kernel times, which excluded
  it. Additive peaks: `4 bw^3` doubles = 10.1 / 21.8 / 46.2 / 126 MB at
  68 / 88 / 113 / 158 (1.81 GB at 384); correlator `5 bw^3` = 12.6 / 27.3 /
  57.7 MB at 68 / 88 / 113.
- Citations corrected: `sht_xcorr.hpp:1009-1041` (`d2P`/`d2N` at
  `:1040-1041`); `include/modality/ebsd/detector.hpp:454-459` (`northPoleQuat`,
  tilt quaternion commented out at `:457`) and `include/idx/base.hpp:133`;
  numba `cache` is not in `targetoptions` (Phase 1's `_cache` check is used).
- Verified-good and kept unchanged (both critics): conventions and layouts,
  ZYZ<->Bunge determination and the upstream `zyz2eu` erratum, direction and
  composition, all bitwise decisions, the C++ tolerances and their margins, the
  Ni fixture values, `northPoleQuat` identity, `rEps`/`thr`, the reference-table
  plan and hand-checked literals.

### 2026-08-16 -- test-suite critic findings, re-measured

Ten findings on the (failing) Phase 3 test suite, each re-measured here
with an independent faithful transcription of `wigner.hpp` `d()` and
with orix, on this machine. Library drift from the drafting
measurements above: **orix 0.15.0** (not 0.14.2), numpy 2.5.2, scipy
1.18.0, **numba 0.67.0** (not 0.65.1), Python 3.13.12.

- **Symmetry identities 6, 7, 8 are not exact.** `d^j_{k,-0}` *is*
  `d^j_{k,0}`, so when `k == 0` or `m == 0` one side of the identity
  fails to trip the reduction branch it is meant to trip and the two
  sides run the recursion at `t` and at `-t`. Over 20000 random
  `(j, k, m, t)` draws (`j <= 40`): eq 6 408 inexact, eq 7 219, eq 8
  205, worst `|delta| = 7.216e-16 = 3.25 eps`; **0 inexact draws have
  both orders non-zero**. Eqs 5 and 9 are exact (0 of 20000). With the
  test's own `seed=0`/200 cases all three fail (3 / 2 / 1 cases, e.g.
  `(18, -3, 0, 0.05862432)` and `(11, 0, 7, 0.25301292)`). Frozen: `==`
  when `k != 0 and m != 0`, else `abs <= 8 eps` (2.5x margin).
- **`bunge_to_zyz(zyz_to_bunge(x))` is one ulp off.** `(x + pi/2)
  - pi/2 != x` in float64 for **38138 of 100000** random triples,
  worst 4.44e-16. `beta` is untouched by both maps and is still
  asserted bitwise. Frozen: `abs <= 1e-15`.
- **The wrong-direction guards are vacuous at `zyz = (0.4, pi, -1.0)`.**
  That rotation has `w == 0.0` (`zyz_to_quaternion` sets it exactly,
  through the `|w| <= rEps` branch), i.e. an involution, so
  `max |R n - (~R) n| = 0.0` (2.36e-16 with orix' own unrounded `w`),
  against 1.902 and 1.571 for `(0.7, 1.1, -2.3)` and
  `(-2.0, -0.6, 1.3)`. Both guards (pointwise and through the Phase 1
  transform) now run on those two `zyz` only; the positive assertions
  still run on all three.
- **Direction oracle margins:** 5.3e-14 / 2.9e-13 / 1.6e-13 for the
  three `zyz` against the `abs 1e-12` bound (thinnest margin 3.4x).
  Kept at 1e-12 (the spec's bound); the figure is recorded per case by
  `record_property`.
- **The NaN guard of `wigner_d_prime`/`wigner_d_prime2` was untested.**
  Deleting `if j < max(|k|, |m|): return nan` from either left the
  whole suite green: the compiled kernel still returns NaN through
  Numba's `sqrt(-2)`, only the `py_func` raises (`ValueError: math
  domain error` at `(j, k, m) = (0, 1, 0)`). The NaN-pattern test is
  now parametrised over the dispatcher **and** its `py_func`.
- **`np.allclose(..., rtol=1e-12)` kept the default `atol=1e-8`.** The
  per-degree powers span 0.075-25.2, so the effective bound was 1e-8
  absolute, 1e5 weaker than the spec's `rel 1e-12` (measured 5.3e-15);
  a uniform +1e-9 perturbation passed. Now `atol=0`.
- **`< 0` instead of `signbit` is an EQUIVALENT mutation**, not an
  untested one: `w = cos(beta/2) * cos(sigma)` can never be `+-0.0` in
  float64 (`|cos(x)| >= 6.12e-17` for every double `x`; the closest
  double to `pi/2` gives exactly that, the next one 2.83e-16, and the
  product of two such factors cannot underflow), and `zyz2qu` sets
  `qu[0] = +0.0` in its `|w| <= rEps` branch anyway. Recorded in
  `plan.md` 4.1 and dropped from the kill list.
- Minor: `test_derivatives_match_mathematica` asserted `not
  isnan(got)` but not `not isnan(want)`, so a finite value on an
  undefined slot passed through `max(worst, nan) == worst`; both
  `KERNEL_NAMES` literals are now asserted equal to the modules' real
  `CPUDispatcher` members, so a kernel added during the implementation
  cannot escape the flag and `py_func` tests.
- **Numba has no `math.fmod` in nopython mode** (`Unknown attribute
  'fmod' of type Module(math)`, on 0.67.0 here as on 0.65.1), so
  `_wrap_beta` must use `np.fmod`, which is the C `fmod` and preserves
  `-0.0`. Noted in the `_wrap_beta` docstring. The flag test needs no
  change on 0.67: `targetoptions` is still
  `{'nogil': True, 'nopython': True, 'boundscheck': None}` and
  `type(kernel._cache).__name__` is still `FunctionCache`.

### 2026-08-16 -- test-skeleton determinations

Resolved while writing the failing suite; re-verified here where the
measurement does not need the (unwritten) implementation.

- **orix is 0.15.0 on this machine**, not the 0.14.2 of the drafting
  measurements. `Rotation.from_euler` still defaults to
  `direction="lab2crystal"` and the test-local `_emsphinx_eu2qu` is
  still **bitwise** equal to it (max diff 0.0 on 1000 triples), so
  `explore-emsphinx-xtal-util-vs-orix.md` 1.2 still holds.
- **The closed-form oracle must be evaluated in log space.** Written
  as the direct product `exp(pre) * c2**(k+m) * s2**(k-m) *
  eval_jacobi(...)` it underflows exactly where the recursion does and
  returns `-0.0` at `(511, 400, 300)`/`beta = 2.5` instead of the
  recorded `-1.209e-184` (`c2 ** 700 == 0.0` kills it too). Combining
  the prefactor, both half-angle powers and `log|jacobi|` in log space
  reproduces `-1.209067502248811e-184` exactly, so the underflow pin
  test compares the recursion's `0.0` against a genuinely independent
  value. The helper reproduces `D_PI_3`/`D_2PI_3` to 3.3e-16/4.7e-16.
- **`min(max|q - q'|, max|q + q'|)` is per quaternion, not global.**
  Taken globally it is `min(2.0, 2.0) = 2.0` on both grids and can
  never pass `10 eps`. Per quaternion, then maxed over the grid:
  **2.833e-16** at `n = 25` and at `n = 15` (the spec's 9.4e-16 was a
  different aggregation; both pass `10 eps = 2.22e-15`).
- **Sign-flip counts reproduce exactly:** **2402 of 60025** at
  `n = 25` and **814 of 12615** at `n = 15`, recorded by
  `record_property`.
- **`beta = pi` degenerate `alpha`: 1.13e-15** over 2000 pairs, not
  the 8.9e-16 of the drafting run. Still inside the spec's `abs
  1e-14`, so the tolerance is kept.
- **`beta = pi - 1e-9` is not the "general branch".** `zyz2qu`'s
  `|w| <= rEps` test fires (`w ~ 4.8e-10 < 1.49e-8`), the quaternion
  is collapsed to an exact pi rotation and `qu2zyz` takes its
  `beta = pi` branch. The near-degenerate test therefore asserts only
  the quaternion-space round trip (`< 1e-14`, measured 0.0-3.3e-16)
  and not which branch is taken.
- `out=np.empty(...)` is nondeterministic (it may hand back NaN in the
  two tripwire slots), so the rejection test writes `0.0` into exactly
  those slots first and additionally tests `np.zeros(...)`.
- `_a_jkm_1_pre` is omitted (plan 3.1 lists only `_0_pre`/`_2_pre`):
  the table kernels group type 1 with type 0 via
  `isType0 = !signbit(t)`, so it is unreachable. 30 kernels in
  `_wigner.py`, 6 in `_euler.py`.
- `e_km` is `(bw, bw)`, matching "Table layouts"; the C++ allocates
  `bw^3` for `pE` but uses only `bw^2`.
- The weekly closed-form scan strides are `k in range(0, j+1, stride)`,
  `m in range(0, k+1, stride)`, giving exactly the 2775 (`j = 511`,
  stride 7) and 946 (`j = 127`, stride 3) point counts recorded above.
- `fukushima2016wigner` already exists in `doc/user/bibliography.bib`
  (added by the specs commit `1235bc58`), so plan amendment 5 needs no
  further action.

(implementation results follow)
