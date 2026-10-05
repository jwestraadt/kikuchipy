# NLPAR -- `feat-NLPAR`: validation

**Status note (requirements D13): drafted 2026-10-04 with read-only
verification and ZERO test execution.** The spec stage is read-only
analysis plus `specs/` writes. Every tolerance marked `MTP`
(measured-then-pinned) is filled at the owning stage's
failing-tests/implementation gates with `pytest.approx(measured,
rel=0.05)` or the ~2x margin convention; until then the test carries
a `None` placeholder that raises "unfilled MEASURED-THEN-PINNED
placeholder" (the HREBSD precedent,
`specs/2026-09-07-hrebsd-dic/validation.md`). **The numbers quoted
as "seed (2026-09-11)" in the tables below are the drafting seeds of
the parked plan `specs/_research/plan-nlpar-2026-09-11.md`, measured
there with a pure-NumPy reference on the drafting machine. They are
SEEDS, not pins:** a seed tells the measurer what scale a correct
implementation should reproduce; the gate measures its own value,
records it in the ledger with date, machine and recipe, and pins
that. A seed that a measurement refutes is amended in
`requirements.md` with the same date. Bitwise expectations are not
MTP: where the oracle is PyEBSDIndex's own float32 kernel the
expectation is equality, and a fallback ulp band exists only so that
a platform difference is recorded rather than silently loosened.

Branch policy (requirements D11): one fork PR `feat-NLPAR ->
develop`, expected `#17`, merged only on the user's go with ubuntu
and windows CI green (macOS is red for the known pre-existing
ebsdsim step and the pseudo-symmetry `23 == 22` test). Pushes of
`feat-NLPAR` trigger the fork's on-push CI (`.github/workflows/
tests.yml:22-49`, `os: [ubuntu-latest, windows-latest,
macos-latest]` at line 43, oldest job at line 48), which is an extra signal: the
gates below are LOCAL gates recorded here with numbers, recipes and
the machine ID. Tests marked [download] need pooch and the network
once (cached after); [skipif pyebsdindex] tests skip when the
optional oracle is absent, so the minimum-requirement CI job stays
meaningful through the test-local NumPy transcription.

Oracle principle. There are four independent oracles, used where
each is strongest: (1) PyEBSDIndex's two static `njit` kernels
`NLPAR.sigma_numba` (`nlpar_cpu.py:752-818`) and `NLPAR.nlpar_nb`
(`:820-936`), public-domain NRL code, called COMPILED on in-memory
arrays, expected BITWISE equal on float32 by construction; (2) a test-local
float64 NumPy transcription `nlpar_reference` of the paper's three
formulas (Brewick, Wright and Rowenhorst 2019, `brewick2019nlpar`),
independent of both kernels, used where PyEBSDIndex deviates from
the paper or from our recorded policy; (3) analytic identities whose
answer is exact by construction (constant map, radius 0, injected
sigma, huge lambda, power-of-two scaling, closed-form lambda); (4)
synthetic generators with seeded `numpy.random.default_rng` whose
expected statistics are derived (iid-noise sigma recovery, two-grain
boundary). Real data (`nickel_ebsd_large`, `si_wafer`) measure the
EFFECT (ADP, IQ, Hough quality), never correctness.

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_signals -k nlpar -n 0
uv run pytest tests/test_signals -k nlpar -n 4
```

The `-n 0` run comes first in every session that compiles a kernel
(numba `cache=True` caches are written by one process) and because
`import pyebsdindex` redirects `NUMBA_CACHE_DIR` to a shared
`~/.pyebsdindex/numbacache`, which defeats per-worker isolation
under xdist (the Hough `test_reflector_list(s)` flake is this). A red
test under `-n 4` is re-run alone before it counts as a failure
(numba-cache flake rule, plan.md stage-independent recipe).

Test modules (two new files plus an append to the root `conftest.py`;
no fixture files under `tests/`, fixtures are generated in-test or
come from `src/kikuchipy/data/**`):

| module | classes | owns |
|---|---|---|
| `tests/test_signals/test_util/test_nlpar.py` | `TestKernels` (V0), `TestSigmaOracle` (V2), `TestAveragingOracle` (V3), `TestLambdaOracle` (V6 kernel/oracle half), `TestDepthAndHalo` (V7 driver half), `TestPolicyOracles` (V10), `TestPerformance` (V11) | the kernels in `src/kikuchipy/pattern/_nlpar.py`, the drivers, the PyEBSDIndex oracles |
| `tests/test_signals/test_ebsd_nlpar.py` | `TestIdentities` (V1), `TestNoiseOracle` (V4), `TestTwoGrain` (V5), `TestLambdaMethod` (V6 method half), `TestLazyAndContracts` (V7), `TestSigmaMethod` (V7, `get_nlpar_sigma` contracts), `TestRealData` (V8), `TestSiWafer` (V9) | `EBSD.average_non_local_neighbour_patterns`, `EBSD.get_nlpar_sigma`, `EBSD.get_nlpar_lambda` |
| root `conftest.py` (appended) | the four generator functions and their same-named fixtures; the shared MTP placeholder `EXP_KERNEL_ULP` and its fixture `exp_kernel_ulp` | the synthetic fixtures and the one placeholder both modules share |

The oracle import is always in-test: `from pyebsdindex import
nlpar_cpu` inside a module-scoped fixture `pyebsdindex_kernels`,
guarded by `pytest.mark.skipif(dependency_version["pyebsdindex"] is
None, ...)` exactly as `tests/test_signals/test_ebsd_hough_indexing.py:35-37`
does. `src/` never imports pyebsdindex (requirements D10). ONE
oracle route (decided 2026-10-04, spec review E1-F1): every parity
test calls the COMPILED dispatchers `NLPAR.sigma_numba` /
`NLPAR.nlpar_nb`, on the small synthetic maps and on
`nickel_ebsd_large` alike, after the fixture has warmed both kernels
once on a `(3, 3 | 4, 4)` float32 map and recorded the warm-up time
through `record_property("pyebsdindex_jit_warmup_s", ...)` (cold
compile measured 2026-10-04: 4.0 s `sigma_numba` + 3.2 s `nlpar_nb`;
both kernels carry `parallel=True`, so the compile is the expensive
part of the default-suite budget and the `-n 0`-first rule above
covers its write to the shared numba cache). The oracle's `.py_func`
is NEVER an oracle: measured 2026-10-04 on a (7, 7 | 8, 8) float32
map, `sigma_numba.py_func` reproduces `sigma` bitwise but not `dout`,
and `nlpar_nb.py_func` differs from the compiled kernel on 2586 of
3136 pixels (max abs 1.07e-4, for a float and a float32 `lam` alike),
because NumPy and numba promote `float32 ** int32` and `2.0 *
float32` oppositely and numba promotes the `n2 += 1.0` counter to
float64 (ledger entry 2). Test names carry `_compiled_`.

**Oracle call recipe** (one helper per kernel in `test_nlpar.py`,
verified against `nlpar_cpu.py:754, :822, :487-502` on 2026-10-04):
`data2d = patterns.astype(np.float32).reshape(nrows * ncols, h * w)`
(pattern index `col + ncols * row`, the C order of the navigation
axes); `indices = np.flatnonzero(~signal_mask.ravel()).astype(
np.int64)` or `np.arange(h * w, dtype=np.int64)`; `sigma, dout, nout =
NLPAR.sigma_numba(data2d, 1, nrows, ncols, np.array([0, nrows],
np.int64), np.array([0, ncols], np.int64), indices,
saturation_protect)` (sigma `(nrows, ncols)` float32; `dout`/`nout`
`(nrows, ncols, 9)` float32 in COMPACT slot order, row outer, column
inner, trailing zeros for out-of-map slots; the self slot holds
`nout = h * w` and `dout = -sqrt(h * w / 2)`; a visited pair without
a kept pixel holds `nout = 1e-12`); `dataout = NLPAR.nlpar_nb(data2d,
float(lam), int(sr), np.float32(dthresh), sigma, nrows, ncols,
np.array([0, 0, ncols, nrows], np.int64), indices,
saturation_protect, np.float32(0.0))` with a FRESH `calclim` array
per call (the default is a mutable array the kernel writes into),
then `dataout.reshape(nrows, ncols, h, w)`; `dataout` is
`np.zeros_like(data, np.float32)` of the FULL block (`:848`) and a
non-default `calclim = [cstart, rstart, ncolcalc, nrowcalc]` leaves
the rows and columns outside it at zero (`:870-876` loop over
`calclim` only), so a `calclim` comparison slices `dataout.reshape(
nrows, ncols, h, w)[rstart:rstart + nrowcalc, cstart:cstart +
ncolcalc]` as PyEBSDIndex's own driver does (`:507-509`; spec review
E3-R3-4); every navigation axis
`>= 2 sr + 1` (else `IndexError` inside `nlpar_nb`, measured
2026-10-04). `lam` is passed as a Python float (requirements D7.3:
PyEBSDIndex's driver would round it to float32 first, `:297`; the
kernels are called directly).

**Fixture generators** (spec review E1-F15; sharing mechanism decided
2026-10-04, spec review E2-F1/F2-F3): the four generators are plain
functions in the ROOT `conftest.py` (the `dummy_signal` precedent,
`conftest.py:195-205`), each exposed by a same-named fixture that
returns the callable (`@pytest.fixture def identical_plus_gaussian():
return _identical_plus_gaussian`); both test modules take them as
fixtures; nothing is imported from one test module into another
(pytest runs with `--import-mode=importlib`, `pyproject.toml:167`,
and `tests/` has no `__init__.py`, so a sibling `from test_nlpar
import ...` raises `ModuleNotFoundError` at collection). The
`pyebsdindex_kernels` warm-up fixture stays module-scoped in
`test_nlpar.py`, the only module that calls the oracle kernels. The
one MTP placeholder asserted in both modules, `EXP_KERNEL_ULP` (V0
and V1), is defined once in the root `conftest.py` beside the
generators and exposed by the fixture `exp_kernel_ulp` (requirements
D1.9; spec review F3-R3-3). The
root `conftest.py` is collected by every `pytest tests/test_signals -k
nlpar` run, the oldest-matrix recipe included.

- `identical_plus_gaussian(nav_shape, sig_shape, sigma=8.0,
  dtype=np.float32, seed=0, sigma_right=None)`: base `np.linspace(
  40.0, 200.0, h * w, dtype=np.float32).reshape(h, w)` (a smooth ramp;
  no in-package data), noise `default_rng(seed).normal(0, sigma, nav +
  sig)` in float32; `sigma_right` given replaces the noise sigma on
  the right half of the columns of every pattern (the two-noise-level
  map of V4 `test_mask_polarity_separates`); `dtype=np.uint8` returns
  `np.clip(np.rint(x), 0, 255).astype(np.uint8)`.
- `two_grain(nav_shape=(10, 16), sig_shape=(32, 32), delta=30.0,
  sigma=8.0, dtype=np.float32, seed=1)`: grain A = the ramp base,
  grain B = base + delta on the right half of the columns, the same
  noise model.
- `random_uniform_saturated(nav_shape, sig_shape, frac=0.05, seed=2,
  one_block_only=False)`: `rng.integers(20, 240, nav + sig,
  dtype=np.uint8)` with `round(frac * h * w)` pixels per pattern set
  to 255 at seeded positions (never all pixels of a pattern;
  `frac=0.0` plants none and is the unsaturated random map of V1);
  `one_block_only=True` plants the 255s in the first `(5, 8)`
  navigation block only (the V7/M11 fixture).
- `exact_duplicates(nav_shape, sig_shape, seed=3, block=False)`:
  `rng.integers(20, 240, nav + sig, dtype=np.uint8)` with NO
  saturated pixels, then patterns (1, 1) -> (1, 2), (3, 4) -> (4, 4),
  (5, 2) -> (5, 3) copied exactly; `block=True` additionally copies
  pattern (3, 4) into its eight 3x3 neighbours, so (3, 4) has no
  neighbour with `d2 > 0` and takes sigma `np.float32(1e12)` while
  each neighbour keeps a finite sigma from its outer neighbours.

Oracle fixtures use `nav_shape=(7, 8)` (every axis `>= 7`, so sr 3 is
legal) with `sig_shape=(12, 12)`; V4 uses `(12, 12 | 32, 32)` float32;
V1 `(4, 5 | 6, 6)`; V0 and V10 shapes as stated there.

Budget: default-suite additions <= ~90 s per worker including the
PyEBSDIndex JIT fixture (recorded at the Stage A implementation
gate), weekly additions <= 5 min.

**MTP placeholder inventory (drafted; the failing-tests gate of each
stage confirms the names in the test modules).** Stage A, shared
(root `conftest.py`, fixture `exp_kernel_ulp`; asserted by V0 and
V1): `EXP_KERNEL_ULP`; Stage A
(`test_util/test_nlpar.py`): `SIGMA_PARITY_ULP`,
`AVERAGE_PARITY_ULP`, `BORDER_BAND_MIN_DIFF`,
`PYEBSDINDEX_JIT_WARMUP_S` (recorded, never asserted);
(`test_ebsd_nlpar.py`): `REFERENCE_MAX_ABS_GREY`,
`NOISE_SIGMA_RATIO_BAND`, `D_MEAN_BAND`, `D_STD_BAND`,
`UNIT_WEIGHT_FRACTION_BAND`, `NOISE_REDUCTION_TOL`,
`TWO_GRAIN_CONTRAST_MIN`, `TWO_GRAIN_BOUNDARY_RESIDUAL_TOL`. Stage B
(`test_util/test_nlpar.py`): `LAMBDA_CLOSED_FORM_REL`,
`LAMBDA_PHANTOM_RATIO`; (`test_ebsd_nlpar.py`): `LAMBDA_NI_RAW`,
`LAMBDA_NI_CORRECTED`, `ADP_BEFORE`, `ADP_AFTER_AUTO`,
`ADP_AFTER_07`, `IQ_BEFORE`, `IQ_AFTER_AUTO`, `HOUGH_PQ_GAIN`,
`HOUGH_FIT_GAIN`, `HOUGH_NMATCH_GAIN`, `HOUGH_CM_GAIN`,
`HOUGH_MISO_MEDIAN_AFTER`, `SI_SIGMA_CV`, `SI_NEFF_MEDIAN`,
`SI_IQ_GAIN`. Stage C adds none (documentation stage). Anything
asserted bitwise has no placeholder.

### V0 -- Kernel discipline (`tests/test_signals/test_util/test_nlpar.py`, `TestKernels`) -- Stage A

Pins: the five kernels of `src/kikuchipy/pattern/_nlpar.py`
(`_window_bounds`, `_nlpar_sigma_kernel`, `_nlpar_distances_kernel`,
`_nlpar_weights_kernel`, `_nlpar_weighted_sum_kernel`, requirements
D7) are compiled with exactly `@njit(cache=True, nogil=True)`, no
`parallel`, no `fastmath`, and every one of them agrees with its own
`.py_func`. Fixture: random float32 maps from `default_rng(0)` of
shape (5, 6 | 8, 8) plus the border cases (1, 7 | 4, 4) and
(2, 2 | 4, 4). Gating: default. Runtime: < 5 s after the first
compile.

- `test_kernel_names_lists_every_njit_kernel_of_the_module`: the
  literal `KERNEL_NAMES` equals `_njit_kernel_names(_nlpar)` (the
  helper of `tests/test_indexing/test_spherical_euler.py:72-85`,
  copied, so a kernel added later cannot escape the flag and
  `py_func` tests). [D7]
- `test_kernels_are_compiled_with_cache_and_nogil` (parametrised over
  `KERNEL_NAMES`): `targetoptions["nogil"] is True`, `type(kernel.
  _cache).__name__ == "FunctionCache"`, `parallel` and `fastmath`
  falsy (the assertions of `test_spherical_euler.py:517-529`). [D7]
- `test_<kernel>_py_func_equals_the_compiled_kernel` (one per
  kernel): compiled and `.py_func` outputs `np.array_equal` for
  `_window_bounds`, `_nlpar_sigma_kernel`, `_nlpar_distances_kernel`
  and `_nlpar_weighted_sum_kernel` (BITWISE: no transcendental, all
  float32 literals cast explicitly so NumPy 1.23 and NumPy 2 agree);
  `_nlpar_weights_kernel`, the only kernel calling `exp`, within
  `EXP_KERNEL_ULP` float32 ulps (the shared `exp_kernel_ulp`
  fixture; MTP; seed 1 ulp from the 2026-09-11
  finding that numba's `exp` and NumPy's differ by one float32 ulp;
  the measured value, 0 or 1, is recorded). The `py_func` arm first
  asserts `hasattr(kernel, "py_func")` so a stub without `@njit`
  fails loudly (`_py_func` helper, `test_spherical_euler.py:88-96`).
  [D7]
- `test_kernel_literals_are_float32`: `_nlpar_sigma_kernel` and
  `_nlpar_distances_kernel` on float32 input return float32 arrays
  through BOTH routes (a Python-float literal promotes the `.py_func`
  result to float64 under NumPy 1.23 and makes the bitwise arm fail
  on the oldest job only); and an `ast` walk of `_nlpar.py` asserts
  that no `ast.Pow` node occurs inside a function decorated with
  `njit` (requirements D7.3: numba types `float32 ** int32` as
  float32, NumPy as float64, measured 2026-10-04; squares are
  products). [D7]
- `test_window_bounds_matches_pyebsdindex_expressions_and_clamps`
  (parametrised over `(n, r, center)` incl. `n < 2 r + 1`):
  `_window_bounds(center, r, n, shift=True)` equals PyEBSDIndex's
  `winstart`/`winend` expressions (`nlpar_cpu.py:872-873, :877-878`)
  whenever `n >= 2 r + 1` and `(0, n)` otherwise; `shift=False` equals
  the clipped `nn_r_start`/`nn_r_end` (`:770-774`); for `shift=True`
  the window length `stop - start` is `min(2 r + 1, n)` in every
  case; for `shift=False` it is `min(c + r, n - 1) - max(c - r, 0) +
  1` (`r + 1` at a corner, `2 r` one step in; spec review C3-R3-F5).
  [D4]
- `test_nlpar_code_never_names_pyebsdindex`: an `ast` walk of the
  source of `kikuchipy.pattern._nlpar` finds no `ast.Import` or
  `ast.ImportFrom` node whose module name starts with `pyebsdindex`
  (including inside functions), and the bodies of
  `EBSD.average_non_local_neighbour_patterns`, `EBSD.get_nlpar_sigma`
  and `EBSD.get_nlpar_lambda` (present from Stage A as the
  `NotImplementedError` stub of requirements D5.7, so the audit runs
  unchanged in both stages) with their docstrings removed (the
  `ast.get_docstring` node dropped before `ast.unparse`) contain no
  `pyebsdindex` token; the module header's NRL notice and the Notes
  paragraph MAY name PyEBSDIndex (requirements D10.4, D12.6), and the
  Hough import elsewhere in `ebsd.py` is legitimate and untouched.
  [D10]
- `_nlpar_normalized_distances` is a NumPy driver, not a kernel: the
  `KERNEL_NAMES` test asserts it is absent from `_njit_kernel_names(
  _nlpar)`; its arithmetic is pinned by V2. [D2/D7]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `EXP_KERNEL_ULP` | 1 float32 ulp | |
| all other kernels vs `py_func` | bitwise (not MTP) | |

### V1 -- Analytic identities (`tests/test_signals/test_ebsd_nlpar.py`, `TestIdentities`) -- Stage A

Pins the formulas through the public method against answers known
exactly, with the test-local float64 `nlpar_reference(data,
search_radius, lam, dthresh, sigma=None, signal_mask=None,
saturation_protect=True)` (a direct NumPy transcription of sigma_i^2
= min_j ||p_i - p_j||^2 / (2 N), d_ij = [sum (p_i - p_j)^2 - N
(s_i^2 + s_j^2)] / [sqrt(2 N) (s_i^2 + s_j^2)], w_ij = exp(-max(d_ij
- dthresh, 0) / lambda^2), self weight 1; search window per axis =
the shifted-inward `(2r + 1)` window when `n >= 2r + 1` and the whole
axis otherwise (the `_window_bounds(shift=True)` semantics,
implemented independently in the reference); sigma window the
clipped 3x3). `N` in both formulas is the PER-PAIR count `n_ij` of
pixels that are unmasked and, with `saturation_protect=True`, both
strictly below the threshold (`0.9961 * max` in the sigma pass,
`0.999 * max` in the search pass, `max` the global maximum; `max + 1`
when protection is off), exactly requirements D3.1/D3.5; the
reference also applies the `d2 > 0` guard, the 1e12 fallback and
`d = +inf` for `n_ij == 0`, so that it is the policy oracle on the
saturated fixture (M12). Fixture: `random_uniform_saturated((4, 5),
(6, 6), frac=0.0, seed=1)` (the unsaturated random map, uint8 and
its `astype(np.float32)`) and `random_uniform_saturated((4, 5), (6,
6))` (5 % saturated) where stated. Gating: default. Runtime: < 3 s.

- `test_constant_map_is_an_identity` (parametrised `search_radius`
  in {1, 3}: at 1 the 3x3 window has 9 weights `fl(1/9)` and shifts
  at every border point; at the default 3 the window is the whole
  (4, 5) map, 20 weights; redesigned 2026-10-04, spec review
  C3-R3-F1, the compiled-oracle probe re-run by the fixer). Two
  protection modes, two routes (requirements D2.4, D3.5). BOX-MEAN
  arms with `saturation_protect=False`: every pattern identical ->
  every pair kept (`n2 = h * w`) with `d2 = 0` -> no neighbour passes
  `d2 > 0` -> every sigma takes the 1e12 fallback -> `d = -sqrt(n2 /
  2) < 0` -> every weight 1 -> the window mean of identical patterns
  is the pattern. uint8 input: output BITWISE equal to the input for
  `dtype_out=None` (`rint` absorbs the accumulation error) and, for
  `dtype_out="float32"`, per pixel `abs(out - p) <= n_window *
  np.spacing(np.float32(p))` with `n_window` the number of window
  points; float32 input: the same linear bound (sequential float32
  accumulation of `n_window` products `fl(fl(1/n) p)` does not sum to
  exactly `p`; probe 2026-10-04: worst 2-3 / 3-4 / 11-13 ulp at 9 /
  20 / 49 weights, so the bound is linear in `n_window`, NOT 2 ulp;
  structural, derived from the `(n - 1) u` sequential-sum bound, not
  MTP). EXCLUSION arm with the default `saturation_protect=True`:
  every pixel equals the global maximum, the strict `<` thresholds
  exclude every pixel, every pair has `n2 == 0`, so `d = +inf` and
  weight 0 on every non-self slot (D3.4) and the output is the self
  pattern EXACTLY: asserted as bitwise identity for `dtype_out=
  "float32"` and for uint8, `get_nlpar_sigma()` equal to
  `np.float32(1e12)` everywhere, and `_nlpar_weights_kernel` on the
  `_nlpar_distances_kernel` output exactly 0.0 on every non-self slot
  (the `n_window * spacing` bound never binds here; the compiled
  oracle on the same map keeps `nout` at its `1e-12` seed and sigma at
  `1e12` and returns the box mean by its weight-1 convention, ledger
  entry 4). The exact bitwise identity under box-mean weights is
  pinned by `test_injected_tiny_sigma_is_an_identity` (self weight
  exactly 1, all others exactly 0). A per-pattern rescale (M14) turns
  a constant pattern into NaN or zeros and dies here in both modes.
  This test is NOT an S6 killer: with the `d2 > 0` guard dropped the
  protection-on arm still has `n2 == 0` (sigma unchanged, `d = +inf`)
  and the protection-off arm gets sigma 0, `dnorm = 0 <= 1e-8`, `d =
  1e6 n2` and weight 0, so the identity holds either way (V2 and V10
  kill S6). [D2/D3/D6]
- `test_search_radius_zero_warns_and_is_a_no_op`: `search_radius=0`
  (and `(0, 0)`) warns with the `average_neighbour_patterns` wording
  (`ebsd.py:1028-1034` precedent), returns `None`, leaves `data`
  untouched; a per-axis `(0, 2)` radius is NOT a no-op. [D1/D4]
- `test_injected_tiny_sigma_is_an_identity`: `sigma=1e-6` (scalar)
  drives `dnorm = 2e-12 sqrt(2 N) ~ 1.7e-11 < 1e-8` at N 36, so every
  non-self pair takes the `d = 1e6 * n2` branch of the distances
  kernel (requirements D3.4, the oracle's `:914-915`) and its weight
  `exp(-1e6 n2 / lam^2)` is exactly 0.0 in float32 (`np.exp(
  np.float32(-104.0)) == 0.0`, verified 2026-10-04); self weight 1 ->
  output equals input after the dtype policy (BITWISE for float32
  output; for uint8 output `rint` of the exact value is the value).
  The branch itself is pinned at kernel level by V10
  `test_tiny_dnorm_branch_gives_1e6_n2`. [D2/D3]
- `test_huge_lambda_is_the_shifted_window_box_mean` (parametrised
  `search_radius` in {1, (1, 2), 2}; the default 3 is NOT used
  because on (4, 5) the window is then the whole map and the shift,
  clamp and zero-extend policies coincide): `lam=1e6`, `dthresh=0` ->
  every weight exp(-d/1e12) rounds to 1 in float32 for |d| < ~60 (the
  random fixture's |d| stays under 10) -> output equals the
  UNWEIGHTED mean over the shift-inward window, compared to a
  test-local `shifted_box_mean` within `n_window * np.spacing(
  np.float32(ref))` per pixel before rounding (the same linear
  accumulation bound as the constant-map test; structural, not MTP).
  At sr 1 both axes shift at every border point (row 0 averages rows
  0-2 shifted vs rows 0-1 clamped); at `(1, 2)` the rows shift while
  the 5-column axis is the whole axis; at 2 both axes are the whole
  map and the test asserts the plain map mean. Kills the clamp (M8)
  and zero-extend (M9) policies on the border band and the
  self-weight-0 mutant (M5) everywhere. [D3/D4]
- `test_power_of_two_scaling_is_exact`: float32 data scaled by
  2^-8, 2^-4 and 2^4 give normalised distances and weights BITWISE
  equal to the unscaled run and `dtype_out="float32"` output equal
  to the scaled unscaled-output (IEEE power-of-two scaling is exact
  through squares, sums, ratios and the global saturation max). The
  scale range keeps `dnorm = (s_i^2 + s_j^2) sqrt(2 n2)` far above
  the absolute `1e-8` guard (at 2^-8 on uint8-range noise sigma ~8,
  dnorm ~ 2 (8/256)^2 sqrt(2 N) ~ 0.09 at N 1024). This is the V1
  scale-invariance test of the `d2 > 0` duplicate-guard decision:
  with PyEBSDIndex's absolute `d2 >= 1e-3` the 2^-8 arm degrades to
  a box filter (D3, recorded deviation). [D3/D7]
- `test_weight_formula_on_injected_distances` (parametrised
  `dthresh` in {0, 0.5}): `_nlpar_weights_kernel` on a hand-built d
  array in {-3, 0, 0.25, 0.5, 2, 50} equals `exp(-max(d - dthresh,
  0) / lam^2)` to `EXP_KERNEL_ULP` (the shared `exp_kernel_ulp`
  fixture of the root `conftest.py`); the `dthresh=0.5` arm has
  weights exactly 1.0 for d in {-3, 0, 0.25, 0.5} (kills "drop
  max(., 0)", M6, and "exp(-d/lam)", M7). [D3/D5]
- `test_reference_agrees_with_the_method_on_random_maps`
  (parametrised; the pyebsdindex-free killer of M1-M4, M8, M9, M12,
  M13, M16, M20 and S3): float64 `nlpar_reference` vs the method's
  `dtype_out="float32"` output on the (4, 5 | 6, 6) random fixture,
  each arm called with the SAME arguments on both sides: `sr` in
  {1, 2}, `lam` in {0.7, 2.5}; `dthresh` in {0, 0.5}; `signal_mask`
  in {None, the inscribed circle (True outside)};
  `saturation_protect` in {True, False} on the saturated variant
  (`random_uniform_saturated((4, 5), (6, 6))`, 5 % pixels at the map
  max); `search_radius` in {(1, 2), (2, 1)} (row, col order); and a
  7-point 1-D scan at `sr=2`. Assertions: max abs difference <=
  `REFERENCE_MAX_ABS_GREY` grey levels on a 255 scale (MTP; seed 1e-3
  grey levels, pinned at ~2x the measured value at the Stage A gate:
  float32 weights carrying 1e-6 to 1e-5 relative error applied to
  values spread 20-240 can move a pixel by a few 1e-4 grey levels, so
  the earlier 1e-4 "bound" was not structural, spec review C2-F6);
  the integer output equals `rint` + clip of the reference at every
  pixel whose reference value lies farther than
  `REFERENCE_MAX_ABS_GREY` from a half-integer, and "zero pixels
  differ by >= 2 grey levels" everywhere (a hard bound; kills the
  truncation mutant M13 on a fixture whose mean fractional part is
  0.5 by construction); and every non-default
  arm (`dthresh=0.5`, the mask, `saturation_protect=False`, each
  asymmetric radius) DIFFERS from the default arm's output, with the
  `(1, 2)` and `(2, 1)` arms asserted unequal to each other, so each
  keyword is seen to be forwarded (M12, M16, S3). [D1/D3/D4/D6/D9]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `REFERENCE_MAX_ABS_GREY`, float32 vs float64 reference, max abs | 1e-3 grey levels (MTP; ~2x the measured value) | |
| pixels differing by 1 after `rint` | only where the reference is within `REFERENCE_MAX_ABS_GREY` of a half-integer; none by >= 2 (bound) | |
| constant map (`saturation_protect=False`) and box mean, float32 output vs exact | <= `n_window` float32 ulp (structural; probe 2026-10-04: 2-3 / 3-4 / 11-13 ulp at 9 / 20 / 49 weights) | |
| constant map (`saturation_protect=True`), every dtype | bitwise (the `n2 == 0` exclusion route; not MTP) | |
| everything else in V1 | bitwise or exact | |

### V2 -- PyEBSDIndex sigma parity (`tests/test_signals/test_util/test_nlpar.py`, `TestSigmaOracle`) -- Stage A [skipif pyebsdindex]

Pins `_nlpar_sigma_kernel` + `_nlpar_normalized_distances` (sigma
AND the normalised 3x3 distances for the lambda fit; requirements
D2.5 as amended 2026-10-04: the kernel returns raw `d2`/`n2`/`valid`,
the NumPy driver normalises) against `NLPAR.sigma_numba(data, nn=1,
nrows, ncols, rowstartcount, colstartcount, indices,
saturation_protect)` (`nlpar_cpu.py:752-818`): 3x3 window CLIPPED at
map edges (`:770-774`), saturation threshold `0.9961 * max(data)`
(`:767`), `+1.0` when protection is off (`:765`), duplicate guard
`d2 >= 1e-3` (`:796`), `mind` seed 1e24 so an all-duplicate pixel
gets sigma 1e12 (`:776, :804`), float32 accumulators (`:784-785`),
normalisation of `dout` by `nout * (s_i^2 + s_j^2)` and `(s_i^2 +
s_j^2) sqrt(2 nout)` only where `nout > 0` (`:812-817`). Our kernel
mirrors this order of operations (the driver reproduces the compiled
oracle's float64 denominator, `2.0 * nout` being float64 under numba,
and stores float32); differences are confined to the `d2 > 0` guard
(identical on integer data, V1 covers floats) and to three slot
conventions the comparison masks: PyEBSDIndex enumerates in-map
neighbours COMPACTLY (row outer, column inner, `:779-802`) where
ours has a fixed nine-slot layout with `valid`; its self slot holds
`nout = npix` (the total pixel count, `:761, :786`) and `dout =
-sqrt(npix / 2)` where ours holds `n2 = n_kept`, `d = -inf` (both
weigh exactly 1 after `max(. - dthresh, 0)`); a visited pair without
a kept pixel keeps `nout = 1e-12 > 0` and a finite tiny negative
`dout` where ours has `n2 = 0`, `d = +inf` (probe 2026-10-04:
interior self slot `dout = -5.656854 = -sqrt(64 / 2)`, corner `nout =
[64, 64, 64, 64, 0, 0, 0, 0, 0]`).

Fixtures: the generators of the Automated section at `nav_shape=(7,
8)`, `sig_shape=(12, 12)` (every axis `>= 2 sr + 1` for sr 3):
`identical_plus_gaussian(..., dtype=np.uint8)`, `two_grain((7, 8),
(12, 12))`, `random_uniform_saturated((7, 8), (12, 12))`,
`exact_duplicates((7, 8), (12, 12))`. Masks: none, and an "automask"
circle (True outside the inscribed circle, kikuchipy polarity,
converted to PyEBSDIndex's kept-`indices` by `np.flatnonzero(~mask)`).
`saturation_protect` on/off. Real data: `nickel_ebsd_large` raw and
background-corrected (`remove_static_background` +
`remove_dynamic_background`, both defaults). Every arm calls the
compiled oracle through the `pyebsdindex_kernels` fixture (Automated
section). Gating: skipif pyebsdindex; the Ni arms are [A, download]
(Stage A, like the rest of V2: they are the only parity tests that
drive the chunked eager route of D8 on a real map, spec review
C2-F1). Runtime: small maps < 2 s after warm-up; Ni compiled < 5 s
after warm-up.

- `test_sigma_parity_compiled_<generator>_<mask>_<protect>`
  (parametrised): `sigma` BITWISE equal (`np.array_equal`); and for
  each point, `ours_d[valid]` taken in slot order with the SELF SLOT
  EXCLUDED on both sides (`slot != 4` for ours, the matching compact
  position for PyEBSDIndex) equals `dout[:n_valid]` BITWISE on every
  slot where `nout >= 1`; slots where PyEBSDIndex holds `nout =
  1e-12` are asserted `n2 == 0` and `d == +inf` on our side. Fallback
  `SIGMA_PARITY_ULP` (MTP, expected 0; if a platform forces a
  non-zero value the ledger says why). [D2]
- `test_sigma_parity_compiled_nickel_ebsd_large_<raw|corrected>`
  [download]: the same comparison on (55, 75 | 60, 60) uint8 (max
  253, verified 2026-10-04); records the oracle's wall time through
  `record_property` (seed 0.12 s single-thread for `sigma_numba` at
  sr=3 context, 2026-09-11). [D2]
- `test_sigma_window_is_clipped_not_shifted`: on `identical_plus_
  gaussian((7, 8), (12, 12))` (seed 0 fixed): (a) ours equals the
  test-local CLIPPED formula BITWISE at every border pixel (the
  corner pixel's sigma is the min over its THREE in-map neighbours,
  an edge pixel's over FIVE; the M20 killer); (b) the test-local
  shifted-3x3 variant (the search-window policy applied to the sigma
  window, M20) satisfies `shifted <= clipped` at every border pixel
  (a min over a superset of candidates); (c) strict inequality holds
  on at least 25 % of the border pixels (the shifted window's extra
  candidates win with probability 5/8 per corner and 3/8 per edge
  pixel on iid noise, so an "at every border pixel" assertion would
  fail against a correct kernel; spec review C2-F4). [D2/D4]
- `test_sigma_fallback_and_duplicates`: on `exact_duplicates((7, 8),
  (12, 12))` the duplicated pair contributes nothing to `mind` (sigma
  of (1, 1) equals the min over its remaining neighbours, test-local,
  and is finite); on `exact_duplicates(..., block=True)` pixel (3, 4),
  whose eight neighbours are all exact copies, gets exactly
  `np.float32(1e12)` while each of those neighbours keeps a finite
  sigma from its outer neighbours. [D2/D3]
- `test_sigma_saturation_threshold_constant`: on `random_uniform_
  saturated` with protection on, the kernel's effective threshold is
  the float64 product `0.9961 * max` (requirements D3.5:
  `np.float64(max_value) * np.float64(0.9961)`, the compiled oracle's
  promotion) and NOT `0.999 * max` (a uint16 arm
  places values in [65280, 65469], excluded by 0.9961 x 65535 =
  65279.4 and kept by 0.999 x 65535 = 65469.5, so the two constants
  are distinguishable; on uint8 data with max 253 both exclude only
  253, computed 2026-10-04). Protection off excludes nothing (the
  `+1` rule). [D2/D3]
- `test_sigma_mask_is_forwarded`: the automask arm's sigma differs
  from the unmasked arm's (kills "mask not forwarded to sigma",
  M16), and `signal_mask` with the polarity flipped (`~mask`)
  differs from both (M15). [D9]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| sigma vs `sigma_numba`, all arms | bitwise (fallback `SIGMA_PARITY_ULP`, expected 0) | |
| normalised 3x3 distances, neighbour slots with `nout >= 1` (self slot excluded) | bitwise | |
| `sigma_numba` wall time, Ni, single thread | 0.12 s (recorded, never gated) | |
| uint16 two-threshold arm | 0.9961 excludes [65280, 65469], 0.999 keeps them (computed 2026-10-04) | |

### V3 -- PyEBSDIndex averaged-pattern parity (`tests/test_signals/test_util/test_nlpar.py`, `TestAveragingOracle`) -- Stage A [skipif pyebsdindex]

Pins `_nlpar_distances_kernel` + `_nlpar_weights_kernel` +
`_nlpar_weighted_sum_kernel` (through the eager driver) against
`NLPAR.nlpar_nb(data, lam, sr, dthresh, sigma, nrows, ncols, calclim,
indices_in, saturation_protect, diff_offset=0)`
(`nlpar_cpu.py:820-936`): search window always `(2 sr + 1)^2`,
SHIFTED INWARD at edges (`winstart/winend`, `:872-873, :877-878`),
saturation threshold `0.999 * max(data)` (`:869`), per-pair `n2`
counted over the unmasked pixels where BOTH values are below the
threshold (`:907-909`), `d2 -= n2 (s_0 + s_1)`, `dnorm = (s_1 + s_0)
sqrt(2 n2)`, `dnorm <= 1e-8 -> d2 = 1e6 n2` (`:910-915`), self slot
`-1e6` so that `max(. - dthresh, 0)` gives weight exactly 1
(`:893, :924-925`), the exponent `-1.0 * w * lam2` evaluated in
float64 and stored float32 (`:847, :925`), weights normalised by
their float32 sum and the output accumulated in float32 in window
order (`:921-934`). Our driver reproduces the operation order. Two
injections: (a) PyEBSDIndex's own `sigma` from V2 fed to both
(isolates the averaging kernels); (b) end to end with our sigma (the
method's actual route). Parameters: sr in {1, 2, 3}, lam in {0.7,
2.5} (Python floats, Automated recipe), dthresh in {0, 0.5} (both
kernels take `np.float32(dthresh)`; the 0.5 arm pins that `dthresh`
reaches the averaging weights, #824's defect 3), protection on/off,
masks none/automask. Fixtures: the V2 generators at `(7, 8 | 12, 12)`
and `nickel_ebsd_large` raw and background-corrected, every arm on
the compiled oracle. Gating: skipif pyebsdindex; Ni arms [A,
download] (Stage A; the eager Ni run is chunked `((47, 8), (47, 28))`
by `get_dask_array`, so these arms pin the chunked eager route of D8
end to end). Runtime: small maps < 5 s after warm-up; Ni compiled
~1-2 s after warm-up (oracle seed 0.54 s single thread at sr=3).

- `test_average_parity_compiled_<generator>_sr<sr>_lam<lam>_<arms>`
  (arms = dthresh, protection, mask, injected|end_to_end): float32
  output BITWISE equal (`np.array_equal`) to `nlpar_nb`'s `dataout`;
  fallback `AVERAGE_PARITY_ULP` per pixel (MTP, expected 0), plus the
  dtype-policy consequence "zero pixels differ by >= 2 grey levels
  after `rint`" (a hard bound, not MTP). [D3/D4]
- `test_average_parity_compiled_nickel_ebsd_large_<raw|corrected>_
  <injected|end_to_end>` [download]: the same on (55, 75 | 60, 60) at
  sr=3, lam in {0.7, 2.5}; the compiled oracle runs once per
  parametrisation and its wall time is recorded. [D3/D4]
- `test_border_band_differs_from_clamp_and_zero_extend`: the
  discriminator. Three policies on `identical_plus_gaussian` at
  sr=2 with the SAME injected sigma and weights: ours (shift
  inward, == PyEBSDIndex bitwise), test-local `clamp` (window
  clipped at the map edge, normalised over the surviving slots, M8)
  and test-local `zero_extend` (zeros outside the map plus
  `window_sums` normalisation, the `average_neighbour_patterns`
  policy, M9). Interior (>= sr from every edge) BITWISE identical
  across the three; on the border band (within sr of an edge) ours
  differs from each alternative by at least `BORDER_BAND_MIN_DIFF`
  grey levels at the worst pixel (MTP; the scale is set by the
  generator's noise sigma 8 x the weight redistribution, expected
  O(1) grey level; measured then pinned at half the measured
  minimum). The "must differ" arm proves the test can see the
  policy; the bitwise PyEBSDIndex arm proves which policy is in
  force. [D4]
- `test_saturation_arm_separates_per_pair_n2_from_global_n`: on
  `random_uniform_saturated` with protection on, the per-pair `n2`
  varies pair to pair (asserted: at least two distinct values); a
  test-local mutant that uses the global unmasked pixel count N in
  `d2 -= n2 (s_0 + s_1)` and `sqrt(2 n2)` (M12) is not bitwise equal
  to the oracle, ours is. With protection off every `n2 == N` and
  the mutant is bitwise equal, which the test asserts too, so the
  arm is known to discriminate only under saturation. [D3]
- `test_calclim_from_block_info`: fixture `identical_plus_gaussian(
  (9, 8), (12, 12))`, `sr=2`, row chunks `(3, 3, 3)` (depth `max(2,
  5 - 3) = 2`, no rechunk), columns unchunked, `lam=1.0`, `dthresh=0`,
  `saturation_protect=False` on BOTH sides so that the oracle's
  block-local `np.max(data)` is irrelevant (protection on is covered
  by V10 `test_saturation_max_is_global`); hand-built haloed blocks
  rows `[0, 5)`, `[1, 8)`, `[4, 9)` with `block_info` dicts
  `{"chunk-location": (k, 0), "num-chunks": (3, 1)}` and the
  whole-map sigma sliced to the block rows as the second operand. The
  chunk wrapper's `block_info -> which sides carry a halo -> kept
  region` mapping equals PyEBSDIndex's `calclim = [cstart, rstart,
  ncolcalc, nrowcalc]` semantics (`:835-843`): the kept regions are
  map rows `[0, 3)`, `[3, 6)`, `[6, 9)`, i.e. `calclim = [0, rstart,
  8, 3]` with `rstart` 0, 2, 2 (requirements D8.5: the first/last
  chunk along an axis has no halo on its outer side;
  `array-location` is in overlapped coordinates under `map_overlap`
  and under `map_blocks` on an overlapped array, measured 2026-10-04,
  and is not read). Assertions: the wrapper returns ONLY the kept
  region (shape `(3, 8, 12, 12)` asserted; S1); it equals the
  `calclim` region of `nlpar_nb` on the same haloed block bitwise,
  i.e. `dataout.reshape(5 or 7, 8, 12, 12)[rstart:rstart + 3, 0:8]`
  (the oracle returns the FULL block, `(5, 8, 12, 12)` / `(7, 8, 12,
  12)` / `(5, 8, 12, 12)`, zero outside the `calclim` rows, so the
  test slices it as PyEBSDIndex's driver does, `:507-509`; spec
  review E3-R3-4; every block
  has `>= 2 sr + 1 = 5` rows, so the oracle is legal); the three kept
  regions concatenated equal the whole-map `nlpar_nb` run bitwise;
  the wrapper raises on `block_info=None`. [D4/D8]
- `test_small_map_padding_slots_are_inf`: `_nlpar_distances_kernel`
  on a (2, 2 | 4, 4) and a (3, 5 | 4, 4) map at sr=3 returns `+inf`
  (weight exactly 0) in every slot with `wr >= len_r` or `wc >=
  len_c` (slot `= wr * (2 rc + 1) + wc`, `(wr, wc)` the window-local
  indices, `len_r`/`len_c` the in-map window lengths `min(2 r + 1,
  n)`; plan.md module 2.4) and finite values in the `len_r x len_c`
  in-map slots; the weighted-sum kernel on those weights equals the
  whole-axis box mean at `lam=1e6` (S2; PyEBSDIndex is not an oracle
  here, its index would be out of bounds). [D4]
- `pyebsdindex_kernels` fixture: module scope; imports
  `pyebsdindex.nlpar_cpu` only; warms `sigma_numba` and `nlpar_nb`
  on a `(3, 3 | 4, 4)` float32 map; `record_property(
  "pyebsdindex_jit_warmup_s", t)`; the ledger carries the first
  measured value (`PYEBSDINDEX_JIT_WARMUP_S`, recorded never gated;
  cold compile 4.0 s + 3.2 s measured 2026-10-04 outside pytest) so
  the default-suite budget line can be checked.

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| float32 output vs `nlpar_nb`, all arms | bitwise (fallback `AVERAGE_PARITY_ULP`, expected 0) | |
| pixels differing by >= 2 grey levels after `rint` | 0 (bound) | |
| `BORDER_BAND_MIN_DIFF`, worst border pixel vs clamp / zero-extend | O(1) grey level (expected class; MTP) | |
| `nlpar_nb` wall time, Ni, sr=3, single thread | 0.54 s (recorded) | |
| `PYEBSDINDEX_JIT_WARMUP_S`, both kernels | unmeasured (recorded) | |

### V4 -- iid-noise oracle (`tests/test_signals/test_ebsd_nlpar.py`, `TestNoiseOracle`) -- Stage A

Pins the statistics the paper predicts on `identical_plus_gaussian(
(12, 12), (32, 32), sigma=8.0, dtype=np.float32)` (Automated
section): the smooth ramp base repeated on a (12, 12) map plus N(0,
sigma_true) float32 noise, `sigma_true = 8` (NEVER 1: with sigma 1
the "sigma unsquared" mutant M2 has `s == s^2` and passes), N = 32 x
32 = 1024 pixels unless stated. Derived expectations (2026-10-04,
analytic; the bands are MTP around the measured values and these
derivations only say what scale to expect): each of the 8
neighbour estimates ||p_i - p_j||^2 / (2 N) has mean sigma^2 and
std sigma^2 sqrt(2/N) = 0.0442 sigma^2 at N 1024; the min over 8
correlated draws sits 1.0 to 1.4 std below the mean, so
sigma_hat / sigma_true is expected in 0.969-0.978 (N 1024) and
0.983-0.988 (N 3600). With exact sigma the normalised d has mean 0
and std exactly 1; a 5 % low sigma^2 bias shifts the mean to about
+0.05 sqrt(N/2) ~ +1.2 at N 1024, so the fraction of neighbour
weights exactly 1.0 (d <= 0) is expected near 0.1, not 0.5. Output
noise variance is sigma^2 sum_j w_ij^2 for independent iid noise.
Gating: default. Runtime: < 5 s.

- `test_sigma_recovery_median_ratio`: median over the interior of
  `get_nlpar_sigma() / sigma_true` inside `NOISE_SIGMA_RATIO_BAND`
  (MTP; seed derived 0.969-0.978 at N 1024). Kills the sign-flipped
  correction (M1) and the unsquared sigma (M2) by an order of
  magnitude. [D2]
- `test_normalised_distance_moments`: mean and std of the 3x3
  normalised distances (pass 1 output, via `_nlpar_normalized_
  distances` on the kernel's raw accumulators) inside
  `D_MEAN_BAND` / `D_STD_BAND` (MTP; derived class mean ~+1.2, std
  ~1 at N 1024; a `sqrt(n2)` denominator (M3) or `d2 / n2` (M4)
  changes the std by sqrt(2) or by N). [D3]
- `test_fraction_of_unit_weights`: fraction of neighbour weights
  equal to exactly 1.0 inside `UNIT_WEIGHT_FRACTION_BAND` (MTP;
  derived class ~0.1); asserted > 0 (kills "drop max(., 0)", M6,
  whose weights are never exactly 1 for d > 0 and exceed 1 for
  d < 0). [D3]
- `test_noise_reduction_matches_the_weights`: variance of (output -
  base) over the interior equals sigma_true^2 x mean_i sum_j w_ij^2
  (weights from `_nlpar_weights_kernel` normalised as the driver
  does) within `NOISE_REDUCTION_TOL` relative (MTP; finite-sample
  class ~5 % at 144 patterns x 1024 pixels). [D3]
- `test_noise_reduction_is_monotone_in_lambda`: residual variance
  strictly decreases over lam in {0.5, 1.0, 2.0, 4.0} (ordering, no
  tolerance). [D5]
- `test_mask_polarity_separates`: on `identical_plus_gaussian((12,
  12), (32, 32), sigma=8.0, sigma_right=24.0)` (the right half of
  every pattern three times noisier) with `signal_mask` True on the
  right half, `get_nlpar_sigma(signal_mask=mask,
  saturation_protect=False)` equals `get_nlpar_sigma(
  saturation_protect=False)` on the left-half-only map `s.isig[:16,
  :]` (bitwise, same pixels, same order; protection OFF on both sides
  because with protection on the global maximum pixel, always
  excluded, moves when the right half is cropped away and changes the
  kept set, spec review C2-F8); the flipped polarity gives a
  different sigma (M15), larger by about a factor 3. [D9]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `NOISE_SIGMA_RATIO_BAND`, N 1024 | derived 0.969-0.978 (2026-09-11 "-1.0..-1.4 sqrt(2/N) in sigma^2"; arithmetic 2026-10-04) | |
| `D_MEAN_BAND` / `D_STD_BAND`, N 1024 | derived ~+1.2 / ~1.0 (2026-10-04) | |
| `UNIT_WEIGHT_FRACTION_BAND` | derived ~0.1 (2026-10-04) | |
| `NOISE_REDUCTION_TOL` | ~5 % class | |

### V5 -- Two-grain sharp boundary (`tests/test_signals/test_ebsd_nlpar.py`, `TestTwoGrain`) -- Stage A (tutorial figure Stage C)

Pins the paper's headline property: NLPAR does not average across a
grain boundary that a Gaussian/box window blurs. Generator
`two_grain(nav_shape=(10, 16), sig_shape=(32, 32), delta=30.0,
sigma=8.0, seed=1)` (Automated section): two base patterns differing
by a mean contrast Delta = 30 grey levels over the whole pattern
(left 8 columns grain A, right 8 grain B), plus N(0, 8) noise.
Derived cross-boundary distance (2026-10-04,
reproducing the 2026-09-11 derivation): d = [N (Delta^2 + 2
sigma^2) - N 2 sigma^2] / [sqrt(2 N) 2 sigma^2] = 159.1 at N 1024,
so -d / lam^2 = -325 at lam 0.7 and -159 at lam 1.0, both below the
float32 `exp` underflow point (between -103.9 and -104.0, verified
2026-10-04): cross-boundary weights are EXACTLY 0.0 for lam <= 1.2;
at lam 2.5 the weight is ~9e-12 x (1/sum), i.e. < 1e-9 grey levels,
not exactly zero. Gating: default. Runtime: < 5 s.

- `test_cross_boundary_weights_are_exactly_zero` (lam in {0.7,
  1.0}): every weight between a grain-A and a grain-B pattern is
  `== 0.0` (kernel-level, exact). [D3/D5]
- `test_contrast_is_retained`: the A-vs-B mean-pattern contrast
  after averaging (lam in {0.7, 2.5} in Stage A; a `lam=None` arm is
  added in Stage B) is >= `TWO_GRAIN_
  CONTRAST_MIN` of the before contrast (MTP; seed 0.995), while the
  comparison arm, a test-local `scipy.ndimage.correlate` of the map
  with the normalised (5, 5) Gaussian window (std 1, the
  `kp.filters.Window("gaussian", (5, 5), std=1)` coefficients) along
  the navigation axes and NO rescale, retains < 0.9 (its
  cross-boundary contrast loss follows from the window weights: the
  two boundary columns mix 2 of 5 columns across the boundary);
  asserted as the ordering NLPAR > Gaussian. `average_neighbour_
  patterns` itself is not the comparison arm because it rescales
  every pattern to the dtype range (`ebsd.py:966-968`,
  `chunk.py:148-163`), which would measure the rescale, not the
  blur (spec review E2-F18); it is shown in the tutorial only. [D3]
- `test_boundary_columns_stay_within_the_within_grain_residual`:
  the residual (output - own base) on the two boundary columns is
  bounded by `TWO_GRAIN_BOUNDARY_RESIDUAL_TOL` x the interior
  residual rms (MTP; expected ~1.0-1.3 because boundary patterns
  have fewer same-grain neighbours in the window and so average
  less). [D3/D4]
- Tutorial figure (Stage C, manual): the two-grain map before,
  after NLPAR and after the Gaussian window, with the sigma map.

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| cross-boundary d at Delta 30, sigma 8, N 1024 | derived 159 (2026-09-11; 159.1 recomputed 2026-10-04) | |
| cross-boundary weights, lam <= 1.2 | exactly 0.0 (exact) | |
| `TWO_GRAIN_CONTRAST_MIN` | 0.995 | |
| `TWO_GRAIN_BOUNDARY_RESIDUAL_TOL` | ~1.0-1.3 class | |

### V6 -- Lambda optimisation (`tests/test_signals/test_util/test_nlpar.py`, `TestLambdaOracle`; `tests/test_signals/test_ebsd_nlpar.py`, `TestLambdaMethod`) -- Stage B

Pins `_nlpar_optimize_lambda` (Nelder-Mead on `scipy.optimize.
minimize`, bounds [1e-3, 10], `fatol` 1e-4, stride 2 above 1e6
points, the PyEBSDIndex settings of `nlpar_cpu.py:164-169`) with the
PHANTOM-FREE objective mean_i |tw - 1 / (1 + sum_{j in map} exp(-max(
d_ij - dthresh, 0) / lam^2))| over the clipped 3x3 neighbourhood
(requirements D5), and `EBSD.get_nlpar_lambda(target_weight=0.34,
...)` / `lam=None`. PyEBSDIndex's `loptfunc` (`:106-112`) differs in
two recorded ways: it uses `max(d2, dthresh)` and it counts the
out-of-map "phantom" slots of `dout` (zeros where `nout == 0`,
`:756-757, :812`) as neighbours of weight 1, which lowers lambda at
map edges. Gating: default (the Ni arms [download]). Runtime: < 10 s
on the constructed fields; Ni end to end ~2 s per target.

- `test_closed_form_lambda_on_constructed_distances` (parametrised
  c in {2, 5, 12}, tw in {0.5, 0.34, 0.25}): a `d` field of shape
  (10, 10, 9) constructed directly (no map): all eight non-self slots
  `= np.float32(c)`, the self slot `-inf`, `valid` all True (so no
  border point has fewer than 8 neighbours and the closed form holds
  at every point); `_nlpar_optimize_lambda(d, valid, tw,
  np.float32(0.0))`. The target is met when 1 / (1 + 8 exp(-c /
  lam^2)) = tw, i.e. lambda(tw) = sqrt(-c / ln((1/tw - 1) / 8)).
  Optimiser result within `LAMBDA_CLOSED_FORM_REL` (MTP; Nelder-Mead
  `fatol` 1e-4 on the objective gives a 1e-3 class in lambda; seed
  1e-3 relative). Closed-form table computed 2026-10-04: c=2: 0.9807 /
  1.1884 / 1.4280; c=5: 1.5506 / 1.8790 / 2.2578; c=12: 2.4022 /
  2.9110 / 3.4978 for tw 0.5 / 0.34 / 0.25. [D5]
- `test_phantom_free_objective_excludes_missing_neighbours`
  (pyebsdindex-free M17 killer; value-level, spec review E2-F6): the
  constructed field above (`c = 2`) with `valid=False` on the
  out-of-map slots of a (10, 10) map's border points (the pattern of
  a real map: 4 corners x 5, 32 edge points x 3) and those slots
  carrying `d = 0.0` (PyEBSDIndex's `dout` convention for unvisited
  slots). Assertions at fixed `lam = np.array([1.0])` and `np.array(
  [2.0])`: `_nlpar_lambda_objective` equals its value on the same
  field with those slots set to `+inf` EXACTLY (exclusion through
  `valid`, not through the stored distance), and differs from a
  test-local phantom-counting objective (invalid slots weight 1) by
  more than 1e-3 absolute (measured 2026-10-04: 0.049 at `lam` 1.0,
  0.042 at `lam` 2.0; the border points' self weights change by
  O(1)). A minimiser comparison alone cannot kill M17: both
  minimisers sit at the kink of the all-valid closed form and differ
  by < 1e-4 (probe 2026-10-04). The arm "minimiser equals the
  all-valid closed form within `LAMBDA_CLOSED_FORM_REL`" is kept.
  [D5]
- `test_mixed_c_field_distinguishes_mean_from_median` (value-level,
  no optimiser call; spec review E2-F7/C2-F2): a constructed field of
  100 points, 70 with every non-self slot at `c = 2` and 30 at
  `c = 12` (all valid, self slot `-inf`), at fixed `lam = np.array(
  [1.5])`: `_nlpar_lambda_objective` equals `np.mean(np.abs(tw - 1 /
  S_i))` over the 100 per-point terms EXACTLY (same `w.sum(axis=-1)`
  over nine slots) and differs from `np.median` of the same terms by
  more than 1e-3 (measured 2026-10-04: mean 0.2616, median 0.1068,
  the median is the `c = 2` term alone), killing M19 (median over
  points instead of `np.mean`, `nlpar_cpu.py:111`; PyEBSDIndex's
  `np.median` at `:181` is over its three target fits). An
  optimiser-based version cannot kill: a 50/50 split makes the median
  of an even-length two-valued array equal the mean, and on any
  split the mean objective has a kink minimum at 1.1884 that
  Nelder-Mead from `x0 = 1.0` returns (measured 2026-10-04 on 70/30:
  1.1884, the global minimum near 2.9 is not reached). [D5]
- `test_lambdas_ascend_with_decreasing_target`: lambda(0.5) <
  lambda(0.34) < lambda(0.25) on the constructed field and on
  `identical_plus_gaussian` (ordering; this is why PyEBSDIndex's
  median of three fits equals its 0.34 fit). [D5]
- `test_bound_hit_warns` (both arms re-verified numerically at the
  Stage B failing-tests gate; spec review F2-F1/C2-F3, measured
  2026-10-04 with scipy 1.17.1): UPPER arm: a (10, 10, 9) field with
  ONE non-self slot at `np.float32(1.0)` and the other seven at
  `+inf` (the `n2 == 0` convention, weight exactly 0), self slot
  `-inf`, all valid: the self weight `1 / (1 + exp(-1 / lam^2))` stays
  above 0.5 > 0.34 for every lambda, so the objective decreases
  monotonically toward the bound and bounded Nelder-Mead stops at 10
  (measured `x = 10.0`, `nit` 9); the optimiser returns `>= 9.9` and
  `pytest.warns(UserWarning, match="upper bound")`. NOT the all-slots
  `c = 200` field (closed form 11.9 > 10, but `8 exp(-200 / lam^2)`
  is below `2^-53` for `lam < ~2.27`, so `1 + 8 w` rounds to exactly
  1.0 in float64 (`exp(-200) = 1.4e-87` is representable; the
  flatness is the rounding of the sum, not an underflow, spec review
  F3-R3-2), the objective is flat 0.66
  around `x0 = 1.0` and Nelder-Mead terminates at `1.000000`,
  measured `nit` 10), and NOT a per-slot mix of `c = 1` and `c = 250`
  (its minimiser is 1.176, inside the bounds, measured). LOWER arm:
  every non-self slot at `np.float32(1e-7)` (closed form `sqrt(1e-7 /
  1.41617) = 2.66e-4 < 1e-3`; measured `x = 1e-3`): the optimiser
  returns `<= 1.01e-3` with `match="lower bound"`. NOT `c = 1e-4`
  (closed form 0.0084, inside the bounds, measured 0.0084, no
  warning); the lower bound is hit only for `c < 1.416e-6` at
  `tw = 0.34`. Message per requirements D5.4: "NLPAR lambda
  optimisation hit the {which} bound ({lam:.4f}); the target weight
  {tw} is not supported by the data". [D5]
- `test_stride_above_1e6_points`: the stride lives in
  `_nlpar_optimize_lambda`, which applies `d[::2, ::2]`, `valid[::2,
  ::2]` before calling the objective when `d.shape[0] * d.shape[1]
  >= 1e6` (requirements D5.1, spec review C2-F12). On a constructed
  float32 field of shape (1000, 1001, 9) (36 MB, `>= 1e6` points)
  `_nlpar_optimize_lambda(d, valid, tw, dthresh) ==
  _nlpar_optimize_lambda(d[::2, ::2], valid[::2, ::2], tw, dthresh)`
  EXACTLY (the strided call is below 1e6 points and is not strided
  again, so both run the same objective on the same array); on a
  (999, 1000, 9) two-valued field (`< 1e6`) constructed so that the
  strided and full grids have different group ratios (`c = 2` on
  even rows, `c = 12` on odd rows: the full field is 50/50, the
  strided one all `c = 2`) the two calls are `!=` (a mutant that
  always strides, or never does, dies in one of the two arms;
  `stride = 1 if sigma.size < 1e6 else 2`, `nlpar_cpu.py:164`). [D5]
- `test_objective_equals_test_local_loptfunc_on_pyebsdindex_dout`
  [skipif pyebsdindex]: PyEBSDIndex's `dout`/`nout` from the compiled
  `sigma_numba` on `identical_plus_gaussian((7, 8), (12, 12))`; our
  objective evaluated on `dout` with the slots masked by `nout >= 1`
  and the self slot replaced by weight 1 equals a test-local
  transcription of `loptfunc` with the same mask, the `+ 1e-12` term
  DROPPED and `max(d - dthresh, 0)` in place of `max(d, dthresh)`
  EXACTLY (requirements D5.1 fixes the form: `lam` is the float64
  `(1,)` array `minimize` passes, `w = np.exp(-np.maximum(d -
  dthresh, np.float32(0.0)) / lam ** 2)` in float64, invalid slots
  zeroed, `S = w.sum(axis=-1)` over the nine compact slots of the
  SAME `dout`-shaped array in the same order, `float(np.mean(...))`;
  the test-local `loptfunc` uses that same `np.sum(axis=-1)` over the
  same nine slots, not `1 + sum of the others`, which is not bitwise
  (probe 2026-10-04: 17 of 56 points differ); `dthresh=0`), and
  `minimize` from the same start with the same options returns the
  same lambda exactly (deterministic Nelder-Mead). [D5]
- `test_phantom_deviation_is_measured_and_pinned` [skipif
  pyebsdindex, download]: on `nickel_ebsd_large` raw and background-
  corrected, lambda(0.34) from PyEBSDIndex's raw `dout` with the
  unmasked `loptfunc` (phantoms counted, `max(d2, dthresh)`) vs
  ours; the ratio ours / theirs inside `LAMBDA_PHANTOM_RATIO` (MTP;
  seeds give 1.1387 / 1.1164 = 1.020 and 2.5787 / 2.5246 = 1.021).
  The phantom count on (55, 75) is 4 x 5 + (2 x 53 + 2 x 73) x 3 =
  776 of 37125 slots (reproduced analytically 2026-10-04). The oracle
  arm of M17 (the pyebsdindex-free arm is above); M18 is killed in
  the `dthresh > 0` arm of V10. [D5]
- `test_get_nlpar_lambda_on_nickel_ebsd_large` [download]:
  `LAMBDA_NI_RAW` and `LAMBDA_NI_CORRECTED` (MTP; seeds 1.1387 and
  2.5787 phantom-free), `lam=None` in the averaging method uses the
  same value (bitwise equal output to `lam=<that value>`) and logs it
  at INFO through `logging` (caplog pin). [D1/D5]
- `test_lam_none_logs_the_optimised_lambda` (default, no download):
  on `identical_plus_gaussian((12, 12), (32, 32))`, `caplog.at_level(
  logging.INFO, logger="kikuchipy.pattern._nlpar")` captures exactly
  one record matching `NLPAR: optimised lambda [0-9.]+ for target
  weight 0.34`, `capsys.readouterr().out == ""` (no `print`), and the
  logged value equals `get_nlpar_lambda()` on the same signal. [D1/D5]
- `test_target_weight_is_forwarded_through_lam_none`: on the same
  synthetic map, `average_non_local_neighbour_patterns(lam=None,
  target_weight=0.5, inplace=False, dtype_out="float32")` is bitwise
  equal to `lam=s.get_nlpar_lambda(target_weight=0.5)` and differs
  from the `target_weight=0.34` run (the keyword is seen to reach the
  optimiser; a hard-coded 0.34 dies). [D1/D5]
- `test_get_nlpar_lambda_accepts_injected_sigma_and_mask`:
  `sigma=` array and `signal_mask=` change the result; a scalar
  `sigma` is broadcast; `sigma=s.get_nlpar_sigma()` reproduces the
  `sigma=None` value exactly (the injected sigma feeds
  `_nlpar_normalized_distances`, requirements D2.5). [D1/D5/D9]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `LAMBDA_CLOSED_FORM_REL` | 1e-3 relative (Nelder-Mead class) | |
| closed form lambda(0.34), c = 2 / 5 / 12 | 1.1884 / 1.8790 / 2.9110 (computed 2026-10-04) | |
| bound arms: one slot `c = 1` + seven `+inf`; all slots `c = 1e-7` | 10.0 / 1e-3 (measured 2026-10-04, scipy 1.17.1; exact, not MTP) | |
| mean vs median of the per-point terms, 70/30 field, `lam` 1.5 | 0.2616 vs 0.1068 (computed 2026-10-04) | |
| phantom-free vs phantom-counting objective, border-masked field, `lam` 1.0 / 2.0 | 0.049 / 0.042 apart (computed 2026-10-04) | |
| `LAMBDA_NI_RAW`, phantom-free, tw 0.34 | 1.1387 | |
| `LAMBDA_NI_CORRECTED`, phantom-free, tw 0.34 | 2.5787 | |
| with phantoms (PyEBSDIndex objective), raw / corrected | 1.1164 / 2.5246 | |
| `LAMBDA_PHANTOM_RATIO`, ours / theirs | 1.020 / 1.021 (parked-plan Reference facts quote a second pair 1.121 / 1.098 = 1.021 at the same target; configuration of that pair unverified) | |

### V7 -- Lazy, chunking, 1-D, determinism, dtype, inplace (`tests/test_signals/test_ebsd_nlpar.py`, `TestLazyAndContracts`, `TestSigmaMethod`; `tests/test_signals/test_util/test_nlpar.py`, `TestDepthAndHalo`) -- every bullet tagged [A] (Stage A) or [B] (Stage B lazy path)

Pins the dask path (two passes, each `da.overlap.overlap(...,
boundary="none")` followed by `da.map_blocks` with explicit
`chunks=`/`meta=` and a core-only chunk wrapper, requirements D8.2-3
as amended 2026-10-04: pass 1 depth 1 for sigma and the raw 3x3
accumulators, pass 2 depth from the NLPAR depth helper with `sigma`
as a second navigation-chunked operand like `window_sums` in
`ebsd.py:1063-1065`; `ensure_minimum_chunksize` rechunk from
`dask.array.overlap`, importable on dask 2026.3.0, verified
2026-10-04) and the method contract inherited from
`average_neighbour_patterns` (`ebsd.py:1018-1019, 1095-1122`). The
eager route of Stage A already runs through this path (an in-memory
signal above 8 MB is chunked by `get_dask_array`); Stage A raises
`NotImplementedError` only for a LAZY INPUT and for
`lazy_output=True`, so every [B] arm below needs one of those. Stage
A drives the real multi-chunk `overlap` + `map_blocks` route without
pyebsdindex through the two `TestDepthAndHalo` driver tests below
and, on a real map, through the V2/V3 Ni parity arms [A, download]
(spec review C2-F1). Chunk notation: every chunking below names the
NAVIGATION axes; the signal axes are always one chunk
(`da.from_array(x, chunks=c + (-1, -1))` in the driver tests, the
same on the `LazyEBSD` data of the method arms; a 2-tuple of ints is
a chunk shape, a tuple of tuples the explicit chunks; spec review
E3-R3-7). Route through the METHOD: a lazy input passes
`get_dask_array(rechunk=True)`, whose `_reduce_chunks` keeps the
chunks of the navigation axis with the smaller chunksize and
re-chunks the other axis to one chunk on these small fixtures
(requirements D8.1; measured 2026-10-04 on (10, 16 | 16, 16):
`(5, 8)` -> `((5, 5), (16,))`, `((3, 3, 4), (7, 7, 2))` -> `((3, 3,
4), (16,))`, `((1, 9), (2, 14))` -> `((1, 9), (16,))`, `(1, 1)` ->
`((1,) * 10, (16,))`, `(2, 3)` -> `((2,) * 5, (16,))`, `(4, 4)` ->
`((4, 4, 2), (16,))`; Ni `((26, 26, 3), (75,))` -> `((26, 26, 3),
(40, 35))`; the issue-230 tuple unchanged), so the method-level [B]
arms pin the ROW chunkings and the column and both-axes chunkings
are pinned by the two [A] `TestDepthAndHalo` driver tests, which
call the drivers directly (spec review F3-R3-1/C3-R3-F2). Gating:
default; Ni arms [download]. Runtime:
< 20 s (the Ni lazy == eager arms dominate).

- [A] `test_pass_one_driver_on_multichunk_dask_array_equals_the_
  kernel` (`TestDepthAndHalo`, pyebsdindex-free; parametrised over
  the navigation chunkings `(5, 8)`, `((3, 3, 4), (7, 7, 2))`, `((1,
  9), (2, 14))`, `(1, 1)`, `(2, 3)`): `_nlpar_sigma(da.from_array(x,
  chunks=c + (-1, -1)), mask_indices=all, max_value=global max,
  saturation_protect=True)` on `random_uniform_saturated((10, 16),
  (16, 16), one_block_only=True)` returns `sigma`, `d2`, `n2` and
  `valid` BITWISE equal to `_nlpar_sigma_kernel` called once on the
  whole array (the pass-1 driver at depth 1 with the core-only
  wrapper, `chunks=` and `meta=` as in plan.md module 2.11; a
  per-chunk maximum, M11, a halo returned with the core, S1, or a
  wrong depth, S8, all break the equality). [D8]
- [A] `test_pass_two_driver_on_multichunk_dask_array_equals_single_
  chunk` (`TestDepthAndHalo`, pyebsdindex-free; the same chunkings
  and fixture, `sr` in {1, 3}, `lam=1.0`, `dthresh=0`,
  `dtype_out="float32"`): `_nlpar_average(da.from_array(x,
  chunks=c + (-1, -1)), sigma, ...)` computed is BITWISE equal to the
  single-chunk run (`chunks=-1` on the navigation axes). At `sr=3` on
  10 rows chunked `(3, 3, 4)` the depth is `max(3, 7 - 3) = 4` and
  `ensure_minimum_chunksize` rechunks the rows to `(6, 4)`; on the
  `(1, 1)` chunking the depth is 6 on both axes (`7 - 1`) and the
  rechunk gives rows `(10,)` and columns `(6, 10)` (measured
  2026-10-04), so a depth `= r` mutant (M10; with depth 3 nothing is
  rechunked and the 3-row edge chunk plus a 3-row halo cannot reach
  row 6), a skipped minimum-chunksize rechunk (S4; `overlap` refuses
  a depth larger than a chunk, or the `map_blocks` `chunks=` no
  longer matches) and a depth off by one (S8) die here in Stage A;
  the saturated pixels sit in one block only, so a per-chunk maximum
  (M11) dies too. Both driver tests bypass `get_dask_array`, so they
  are the ONLY arms that see column and both-axes chunkings
  (requirements D8.1, D8.6). [D8]
- [A] `test_depth_helper` (`TestDepthAndHalo`): for `search_radius`
  r and an axis of length n with chunks c: unchunked axis -> depth 0;
  `2 r + 1 >= n` -> the axis is rechunked to a single chunk; else
  depth `max(r, 2 r + 1 - min(c[0], c[-1]))`, then `chunks_out =
  ensure_minimum_chunksize(depth, c)`. Cases (rule values re-computed
  and the rechunks measured 2026-10-04): (55,) chunks (26, 26, 3) at
  r=3 -> 4 (NOT 3: the shifted window of the last three rows reaches
  2 r = 6 rows inward, past the 3-row chunk plus a 3-row halo; this
  is the mutant M10 "depth sr with tiny edge chunk"), `chunks_out ==
  (26, 25, 4)`; (26, 26, 3) at r=4 -> max(4, 9 - 3) = 6, `chunks_out
  == (26, 23, 6)`; (55,) chunks (47, 8) at r=3 -> 3 and at r=4 ->
  max(4, 9 - 8) = 4 (the first term wins; `chunks_out == (47, 8)`);
  (5,) at r=2 -> single chunk; (3,) at r=3 -> single chunk; the
  pass-1 depth is 1 for any chunked axis. [D8]
- [A] `test_argument_validation` (parametrised over `(kwargs,
  match)`, each `pytest.raises(ValueError, match=...)` on an eager
  (4, 5 | 6, 6) signal, for `average_non_local_neighbour_patterns`
  and `get_nlpar_sigma` where the keyword exists in Stage A; EVERY
  `get_nlpar_lambda` arm (`target_weight`, `dthresh`, `sigma`,
  `signal_mask`) is [B], because in Stage A that method is the
  `NotImplementedError` stub of requirements D5.7 and raises before
  any validation, spec review E3-R3-5): `search_radius=-1` and `1.5`
  -> "search_radius
  must be a non-negative int"; `search_radius=(1, 2, 3)` on a 2-D map
  -> "one radius per navigation axis"; `lam=0.0`, `lam=-1` -> "lam
  must be > 0"; `dthresh=-0.1` -> "dthresh must be >= 0";
  `target_weight=0.0`, `1.0` -> "0 < target_weight < 1" (a
  `get_nlpar_lambda`-only keyword, so [B]); `sigma=0.0`, `-1.0`,
  `True` ->
  "sigma must be > 0"; `sigma=np.zeros(nav_shape, np.float32)` and an
  array with one `np.nan` element -> "sigma must be > 0" (array
  elements checked after the cast, spec review E2-F9);
  `sigma=np.ones((2, 2))` -> "navigation shape";
  `search_radius=True` and `np.int64(2)`: the first -> "search_radius
  must be a non-negative int", the second accepted and equal to the
  `2` run bitwise (spec review E2-F15); `dtype_out=bool` and
  `dtype_out=complex` -> "dtype_out must be an integer or floating
  dtype" (spec review E2-F16); `signal_mask=np.ones((2, 2), bool)` ->
  "signal shape"; `signal_mask=np.ones(sig_shape, bool)` -> "excludes
  every pixel"; 0-D navigation -> "nothing to average" (also with
  `search_radius=3`, the 0-D check precedes the radius broadcast); an
  integer 0/1 mask is accepted and gives the bool mask's output
  bitwise. Fragments frozen in requirements D1.6. [D1/D2/D6/D9]
- [A, deleted in Stage B] `test_stage_a_guards_raise_not_implemented`:
  `lam=None` -> `NotImplementedError` matching "lambda optimisation
  lands in Stage B"; a lazy input, and `lazy_output=True` with
  `inplace=False`, -> "lazy NLPAR lands in Stage B"; `get_nlpar_sigma`
  on a lazy input -> the same; `get_nlpar_lambda()` with default
  arguments -> "lambda optimisation lands in Stage B" (the fifth
  guard, the Stage A stub; requirements D5.7); the method's guards
  fire AFTER the D1.6
  validation (a lazy signal with `search_radius=-1` raises
  `ValueError`, not `NotImplementedError`), the stub before any. [D5]
- [B] `test_lazy_equals_eager_<chunking>` (parametrised):
  `dtype_out="float32"` output of the lazy route BITWISE equal to the
  eager route on `random_uniform_saturated((10, 16), (16, 16),
  one_block_only=True)` (saturated pixels in ONE block only, the
  first (5, 8) block, so a per-chunk maximum, M11, is caught) for the
  INPUT chunkings single, regular `(5, 8)`, irregular tiny edge
  `((3, 3, 4), (7, 7, 2))`, `((1, 9), (2, 14))`, and `(1, 1)`,
  `(2, 3)`, `(4, 4)` navigation chunks of the `LazyEBSD` (the parked
  plan's pins; the parametrisation ids are the input chunkings).
  What the method PROCESSES is the `_reduce_chunks` output of the V7
  header: the row chunks `(5, 5)`, `(3, 3, 4)`, `(1, 9)`, `(1,) *
  10`, `(2,) * 5`, `(4, 4, 2)` with the 16 columns as one chunk, so
  these arms pin lazy == eager over the ROW chunkings (regular,
  tiny-edge, thinner than the depth); the saturated block, the
  `(1,) * 10` rows and the `(3, 3, 4)` rows survive the re-chunking,
  so the M11, S1, S4 and S8 [B] arms stand, while the column and
  both-axes coverage is the two [A] driver tests above (requirements
  D8.1, D8.6; spec review F3-R3-1/C3-R3-F2). The integer output is
  then bitwise equal too (same float32 input to `rint`). [D8]
- [B] `test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3`
  [download]: `LazyEBSD` rechunked EXPLICITLY to `{0: (26, 26, 3),
  1: (75,)}` at sr=3 and sr=4, lazy == eager bitwise; through the
  method `_reduce_chunks` processes it as `((26, 26, 3), (40, 35))`
  (measured 2026-10-04), so the 3-row last chunk still exercises the
  depth rule (`(26, 25, 4)` at sr 3, `(26, 23, 6)` at sr 4) and the
  M10 [B] killer stands. The default
  `get_dask_array(chunk_bytes=8e6, rechunk=True)` chunking of this
  dataset is `((47, 8), (47, 28), (60,), (60,))` on dask 2026.3.0
  (measured 2026-10-04; the parked plan's `(26, 26, 3)` is therefore
  an explicitly constructed case, not the default), so a second arm
  runs the default chunking as the method itself produces it. [D8]
- [B] `test_issue_230_irregular_chunks` [download-free]: the chunk
  tuple of `tests/test_signals/test_ebsd.py:1619`, `((3, 3, 4, 3, 4,
  3, 4, 3, 3, 4, 3, 4, 3, 4, 3, 4), (75,), (6,), (6,))` on a
  `da.zeros((55, 75, 6, 6), dtype=uint8)` `LazyEBSD` plus a noisy
  variant of the same chunking (`_reduce_chunks` leaves this tuple
  unchanged, measured 2026-10-04): `average_non_local_neighbour_
  patterns(inplace=False, lazy_output=True)` builds and computes
  without error and equals the eager run bitwise (the `ensure_
  minimum_chunksize` rechunk handles chunks thinner than the depth;
  the regression class of pyxem/kikuchipy#230, cited from that test's
  docstring, the issue text itself unverified today). [D8]
- [A, lazy arm B] `test_one_dimensional_navigation_equals_a_one_row_
  map`: a (1, n) map and the same patterns as an n-point 1-D scan
  (`search_radius` int applies to the one axis) give bitwise equal
  output (eager [A]; lazy input [B]); `get_nlpar_sigma()` on the
  n-point scan returns shape `(n,)` and equals the `(1, n)` map's
  sigma row bitwise [A]; the per-axis ORDER of a 2-D tuple is pinned
  in V1 (`(1, 2)` vs `(2, 1)`, S3). [D1/D9]
- [A, lazy arm B] `test_map_smaller_than_the_window`: (2, 2) and
  (3, 5) maps at sr=3: the window is clamped to the map (PyEBSDIndex's
  unchecked index would read out of bounds here, recorded quirk),
  output equals the test-local reference with the whole-axis window
  [A]; lazy == eager [B]. [D4]
- [B] `test_scheduler_and_thread_invariance`: `scheduler=
  "synchronous"`, `"threads"` with `num_workers` 1 and 4 give bitwise
  identical output (no `parallel=True`, no cross-chunk accumulation).
  [D7/D8]
- [A] `test_dtype_round_trip` (uint8, uint16, float32, float64 input;
  `inplace=False`): default output dtype equals the input dtype;
  integer outputs equal `np.clip(np.rint(f32), omin, omax)` of the
  `dtype_out="float32"` run (BITWISE), float64 output is the float32
  average cast up (BITWISE), `dtype_out="float32"` returns the raw
  average; the `rint` arm on a fixture with mean fractional part 0.5
  differs from `floor` on ~half the pixels (M13). Pins D6's "no
  per-pattern rescale": the mean of the output equals the mean of the
  input within 0.5 grey levels (a convex combination; a rescale, M14,
  moves the mean by tens of grey levels on the two-grain map). [D6]
- [A] `test_inplace_with_dtype_out_changes_the_data_dtype`: a uint8
  signal with `inplace=True, dtype_out="float32"` ends with
  `s.data.dtype == np.float32` and data bitwise equal to the
  `inplace=False` float32 result (requirements D1.5: `self.data =
  result.compute()` when the dtype changes, the `downsample`
  precedent; a `da.store` into the old uint8 buffer would truncate);
  uint8 -> uint8 in place (the `store` path) equals the
  `inplace=False` result bitwise. Extra M13/M14 killer. [D1/D6]
- [A, lazy arms B] `test_inplace_lazy_output_contract`: [A]
  `lazy_output=True` with `inplace=True` raises `ValueError` with the
  `average_neighbour_patterns` message (`ebsd.py:1019`) on an eager
  signal; `inplace=False` returns `EBSD`; `inplace=True` returns
  `None` and modifies `data`; [B] `inplace=False, lazy_output=True`
  returns `LazyEBSD` (the `test_lazy_output` shape of
  `test_ebsd.py:1683-1692`); a lazy input with `lazy_output=None`
  stays lazy and, with `inplace=True`, keeps its original chunks
  (`rechunk(old_chunks)`, `ebsd.py:1107`); [B] a lazy input with
  `lazy_output=False, inplace=True` ends with `isinstance(s.data,
  np.ndarray)` (the signal is eager afterwards) and data bitwise
  equal to the eager `inplace=False` run (requirements D1.5 (a); the
  `average_neighbour_patterns` precedent would `store` into a dask
  target here, spec review C2-F15). [D1]
- [A, download] `test_inplace_equals_inplace_false_on_a_multichunk_
  eager_signal`: `nickel_ebsd_large` (eager, default chunking `((47,
  8), (47, 28))` under `get_dask_array`), `lam=2.5`, `sr=3`, two
  arms: one under `dask.config.set(scheduler="synchronous")` (a
  deterministic task order, the stronger pin) and one under the
  `"threads"` scheduler with the default worker count: `inplace=True`
  (the `store(self.data, compute=True)` path into the buffer the
  graph's chunks view) is BITWISE equal to `inplace=False` in both
  (requirements D1.5 (b): a halo read scheduled after a neighbour's
  store would otherwise read averaged data; measured 2026-10-04 for
  the identical `average_neighbour_patterns` mechanism, bitwise under
  synchronous and threads 1/2/20, 0 differing patterns, ledger entry
  4; if this ever differs the recorded fallback is `self.data =
  result.compute()` for every dtype). [D1/D8]
- [A] `test_custom_attributes_are_carried`: `xmap`, `detector`,
  `static_background` survive `inplace=False` (`_get_custom_
  attributes`, `ebsd.py:1111`). [D1]
- [A] `test_show_progressbar_registers_and_unregisters`: the
  `ProgressBar` registration mirrors `ebsd.py:1096-1101, 1118-1119`
  (both branches covered). [D1]
- [A] `test_sigma_argument_contract`: `sigma=` scalar, `sigma=` array
  of the navigation shape (skips pass 1, bitwise equal to passing the
  `get_nlpar_sigma()` result), wrong shape raises. [D1/D2]
- [A] `TestSigmaMethod::test_returns_float32_of_navigation_shape`: a
  (4, 5) map -> float32 of shape (4, 5); a 1-D scan of 7 -> (7,).
  `TestSigmaMethod::test_equals_the_pass_one_of_the_method`:
  `average_non_local_neighbour_patterns(sigma=s.get_nlpar_sigma(),
  lam=1.0, inplace=False, dtype_out="float32")` is bitwise equal to
  the `sigma=None` run (same kernel, same arguments, D1.4).
  `TestSigmaMethod::test_forwards_mask_and_protection`: on the
  saturated fixture a circular `signal_mask` and `saturation_protect=
  False` each change the sigma map (M16). [D1/D2]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| lazy vs eager, every chunking | bitwise (not MTP) | |
| depth for (26, 26, 3) at r=3 / r=4; (47, 8) at r=4 | 4 / 6; 4 (rule, re-computed 2026-10-04) | |
| default chunking of `nickel_ebsd_large` at 8e6 bytes | `((47, 8), (47, 28))` (measured 2026-10-04, dask 2026.3.0) | |
| `_reduce_chunks` on the lazy inputs of the method arms (V7 header) | row chunks kept, columns one chunk; Ni `((26, 26, 3), (75,))` -> `((26, 26, 3), (40, 35))` (measured 2026-10-04, dask 2026.3.0) | |
| output mean vs input mean | within 0.5 grey levels (bound) | |

### V8 -- Real-data effect (`tests/test_signals/test_ebsd_nlpar.py`, `TestRealData`) -- Stage B [download], Hough arms [skipif pyebsdindex], full-map Hough weekly

Pins that NLPAR improves `nickel_ebsd_large` (4125 patterns, uint8,
max 253, min 21, verified 2026-10-04) by the two kikuchipy map
metrics and by Hough indexing quality, at the default `search_radius
= 3` on the background-corrected signal (`remove_static_background()`
then `remove_dynamic_background()`, defaults). Rule for Hough pins
(binding): **pin only measured improvements, at >= 0.5 x the
measured gain; where a metric does not improve, pin "not worse" and
say so in the ledger.** The full map runs in the default suite
(estimated 0.2-0.4 s with dask threads; the ADP and IQ maps are
cheap); the Hough before/after runs on the slicing subset
`s.inav[::5, ::5]` in the default suite and on the full map weekly.
The subset has data shape `(11, 15, 60, 60)` (165 patterns,
navigation (11, 15) rows x cols; hyperspy repr `(15, 11|60, 60)`),
carries the detector, the static background and an `xmap` of
`size == 165` whose points are in the same row-major order as the
patterns (orix reports the extent shape `(51, 71)` under step
slicing; the test compares orientations by point ORDER, never by
`xmap.shape`); measured 2026-10-04 (ledger entry 3; the roadmap
Phase 11 precedent, `specs/roadmap.md:105`, uses the same `[::5,
::5]` 165-point subset). `extract_grid((11, 15))` is NOT used: on
this dataset it returns data `(13, 10, 60, 60)` under an axes
manager claiming `(11, 15)` and an `xmap` of shape `(49, 64)`
(`ebsd.py:329-332` reverses `grid_shape` before `grid_indices`),
an internally inconsistent signal (ledger entry 3; a kikuchipy
quirk the suite does not rely on and does not fix).

- `test_adp_improves`: `get_average_neighbour_dot_product_map()`
  mean before (`ADP_BEFORE`, MTP; seed 0.600) and after with `lam=
  None` (`ADP_AFTER_AUTO`, seed 0.904 at lambda ~2.52) and `lam=0.7`
  (`ADP_AFTER_07`, seed 0.766); asserted after > before (ordering)
  and each inside its pin. [D5/D12]
- `test_iq_improves`: `get_image_quality()` mean before/after
  (`IQ_BEFORE`, `IQ_AFTER_AUTO`, MTP, no seed: "IQ measured"), after
  > before. [D12]
- `test_hough_quality_before_and_after` [skipif pyebsdindex]:
  `hough_indexing` with the stored detector and an `EBSDIndexer`
  from `detector.get_indexer(phase_list)` on the 165-point grid
  before and after NLPAR; medians of `pq`, `fit`, `nmatch`, `cm`
  gains (`HOUGH_*_GAIN`, MTP; seeds none) under the rule above; and
  the median misorientation to the STORED orientations of the
  dataset's `xmap` (`HOUGH_MISO_MEDIAN_AFTER`, MTP) with the
  before-value recorded beside it. The stored xmap's provenance is
  recorded, not assumed: it carries props `scores` and `z` (verified
  2026-10-04); the roadmap Phase 11 line calls it "Hough + refined
  (0.8.0 provenance), NOT a DI reference" while the parked plan
  calls it "DI-refined"; the ledger entry that fills the pin states
  which it is after reading `kikuchipy-data`'s provenance. [D12]
- `test_hough_quality_full_map` (`@pytest.mark.weekly`): the same on
  all 4125 patterns; recorded. [D12]
- `test_sigma_map_is_plausible`: `get_nlpar_sigma()` on the
  corrected signal has a median in the grey-level class of the
  dataset's noise (recorded, not gated; feeds the tutorial sigma-map
  figure) and the raw-vs-corrected ratio is recorded. [D2/D12]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `ADP_BEFORE`, corrected Ni | 0.600 | |
| `ADP_AFTER_AUTO` (lambda ~2.52) | 0.904 | |
| `ADP_AFTER_07` (lambda 0.7) | 0.766 | |
| `IQ_BEFORE` / `IQ_AFTER_AUTO` | unmeasured | |
| `HOUGH_PQ_GAIN`, `HOUGH_FIT_GAIN`, `HOUGH_NMATCH_GAIN`, `HOUGH_CM_GAIN`, 165-pt grid | unmeasured; pin >= 0.5 x measured gain or "not worse" | |
| `HOUGH_MISO_MEDIAN_AFTER` vs stored xmap (before recorded beside it) | unmeasured | |
| full-map Hough (weekly) | recorded only | |

### V9 -- Low-signal weekly/local (`tests/test_signals/test_ebsd_nlpar.py`, `TestSiWafer`) -- Stage B [download], weekly

`kp.data.si_wafer(allow_download=True, lazy=True)`: (50, 50 | 480,
480) uint8, 311 MB zipped download (`_data.py:392-449`; cached on
the drafting machine, verified 2026-10-04). The suite downloads
NOTHING by itself: a missing cache is a skip naming the
`allow_download=True` call (the HREBSD V5 precedent). Marked
`@pytest.mark.weekly`; runs lazily with the method's own chunking.

- `test_si_wafer_sigma_cv`: coefficient of variation of the sigma
  map (`SI_SIGMA_CV`, MTP, recorded class: a single crystal should
  have a flat sigma map). [D2]
- `test_si_wafer_effective_neighbours`: median N_eff = 1 / sum_j
  w_ij^2 over the window at the auto lambda (`SI_NEFF_MEDIAN`, MTP,
  recorded). [D3/D5]
- `test_si_wafer_iq_gain`: `get_image_quality()` mean ratio after /
  before (`SI_IQ_GAIN`, MTP; asserted > 1). [D12]
- Runtime recorded (seed 5-10 s multi-threaded plus the download).

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| `SI_SIGMA_CV` | unmeasured | |
| `SI_NEFF_MEDIAN` | unmeasured | |
| `SI_IQ_GAIN` | unmeasured, > 1 | |
| si_wafer runtime, lazy, default threads | 5-10 s | |

### V10 -- Policy oracles where PyEBSDIndex is not the oracle (`tests/test_signals/test_util/test_nlpar.py`, `TestPolicyOracles`) -- Stage A (arms tagged [B] land with Stage B)

Each recorded deviation from PyEBSDIndex (requirements D3) has a
test that pins OUR behaviour and, where PyEBSDIndex is available,
asserts the deviation is real (so a later "parity" refactor cannot
silently adopt the oracle's behaviour). Fixtures: hand-built
(3, 3 | 4, 4) float32 maps. Gating: default; the oracle arms skipif
pyebsdindex and compiled. Runtime: < 3 s after warm-up.

- `test_n2_zero_pair_gets_weight_zero`: a neighbour whose every
  pixel is saturated (all values equal the map max, protection on):
  `n2 == 0` -> `d = +inf` -> weight exactly 0.0, the pattern of
  interest unaffected by it; PyEBSDIndex gives `d2 = 1e6 * 0 = 0`
  and weight 1 (`nlpar_cpu.py:915`), asserted not-equal in the
  oracle arm. [D3]
- [B] `test_dthresh_is_consistent_between_kernel_and_objective`:
  with `dthresh=0.5` the weights the averaging kernel produces and
  the weights inside the lambda objective are the same function (the
  objective's mean self-weight recomputed from the kernel's weights
  on the same map is bitwise what the objective returns); the
  PyEBSDIndex-style `max(d2, dthresh)` objective (M18) evaluated on
  the same distances differs (asserted). Stage B because it needs
  `_nlpar_lambda_objective`. [D3/D5]
- `test_tiny_dnorm_branch_gives_1e6_n2`: `_nlpar_distances_kernel`
  with an injected `sigma2 = 1e-12` everywhere (sigma 1e-6) returns
  `d == np.float32(1e6) * n2` on every non-self in-map slot (the
  `dnorm <= 1e-8` branch of requirements D3.4, the oracle's
  `:914-915`), the self slot `-inf`; the oracle arm is a test-local
  NumPy transcription of `nlpar_cpu.py:901-916` (float32 `d2`, `n2`,
  `dnorm`, the `<= 1e-8` branch, `sigma1 + sigma0` in float32)
  evaluated on the same block and asserted BITWISE equal to the
  kernel's `d` on the non-self slots; no compiled call (the pre-`exp`
  weights are not observable through `nlpar_nb`'s output, spec
  review E2-F20). [D3]
- `test_duplicate_neighbour_is_skipped_in_sigma_but_averaged`: an
  exact duplicate neighbour does not set sigma (`d2 > 0` guard) yet
  receives weight exactly 1 in the average (d < 0 -> `max(., 0)` =
  0), matching PyEBSDIndex on integer data (`d2 >= 1e-3`) and, on
  float data in [0, 1] with d2 in (0, 1e-3), differing from it
  (oracle arm asserts the difference). [D3]
- [A, lazy arm B] `test_saturation_max_is_global`: [A] arm:
  `_nlpar_average_chunk` called directly on the two hand-built haloed
  blocks of one (3, 6 | 4, 4) map whose halves have different maxima,
  with hand-built `block_info` dicts, once with `max_value` = the
  global maximum and once with each block's own maximum: the outputs
  differ on the block with the lower maximum (M11 killer) and the
  global-maximum outputs concatenated equal the single-block run
  bitwise. [B] arm: a lazy two-chunk `LazyEBSD` of the same map gives
  bitwise the same output as the eager run (the threshold is
  `max(data)` over the WHOLE map, computed once before the overlap).
  PyEBSDIndex's tile-local max is the recorded quirk; the oracle runs
  whole arrays in this suite so parity is unaffected. [D3/D8]
- `test_uint16_two_threshold_arm`: uint16 data with values in
  [65280, 65469] and the max at 65535: those pixels are excluded
  from the sigma estimate (0.9961 x 65535 = 65279.4) and kept in
  the averaging distances (0.999 x 65535 = 65469.5); a single-
  constant mutant changes one of the two (asserted both ways).
  Protection off keeps every pixel in both. [D3]
- `test_sigma_fallback_value_is_1e12`: literal pin of the fallback
  (`np.float32(1e12)`) and of the two module-level threshold
  constants read by the kernels: `SIGMA_SATURATION_FACTOR == 0.9961`
  (a Python float; it meets the float64 `np.float64(max_value)` in
  the sigma kernel, requirements D3.5) and `AVERAGE_SATURATION_FACTOR
  == np.float32(0.999)` with `type(...) is np.float32` (it meets the
  float32 `max_value` in the averaging kernel). [D3/D7]

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| all V10 arms | exact / bitwise (not MTP) | |

### V11 -- Performance (`tests/test_signals/test_util/test_nlpar.py`, `TestPerformance`; `benchmarks/`) -- Stage B, recorded never gated

- `test_runtime_is_recorded` [download]: `record_property` of the
  wall time of `get_nlpar_sigma()` and of `average_non_local_
  neighbour_patterns(lam=2.5, inplace=False)` on `nickel_ebsd_large`
  at sr=3, default dask threads, best of 3 after one warm-up; no
  assertion. [D13]
- pytest-benchmark entry `benchmarks/pattern/test_nlpar.py`
  (pytest-benchmark 5.2.3 installed, verified 2026-10-04; the
  existing tree has `benchmarks/indexing/` only, so the folder is
  new) with the same two routes at sr in {1, 3}. [D13]
- Numbers are copied into the Performance table below with date,
  machine and recipe.

| quantity | seed (2026-09-11) | pin (date, machine, recipe) |
|---|---|---|
| pure-NumPy `nlpar_reference`, full Ni map, sr=3 | 6-14 s (test-local reference; explains why V1 uses small maps) | |
| PyEBSDIndex `sigma_numba` + `nlpar_nb`, Ni, sr=3, single thread | 0.12 s + 0.54 s | |
| ours, Ni, sr=3, dask threads | 0.2-0.4 s (estimate, not a measurement) | |
| ours, si_wafer, lazy, dask threads | 5-10 s (estimate) | |

## Local-gated and weekly

- [download] V8 Ni arms: the default suite fetches `nickel_ebsd_large
  (allow_download=True)` as the spherical suites already do
  (`tests/test_indexing/test_spherical_emsphinx_regression.py:672`,
  tech-stack.md:50); cached by pooch after the first run. Full-map
  Hough `@pytest.mark.weekly`.
- [download, weekly] V9 `si_wafer` (above): skips with the
  `allow_download=True` instruction when uncached; never downloads
  by itself.
- Local-only ledger entry, never a test: the `ni_gain` series
  (`kp.data.ni_gain(number, allow_download=True)`, the five-gain
  Ni calibration series at 480 x 480 if present locally) run once at
  Stage B with `get_nlpar_sigma()` and the auto lambda per gain,
  recorded as a sigma-vs-gain table in the ledger for the tutorial's
  parameter-guidance paragraph. Not gated, not weekly; if the series
  is not cached the ledger says so.
- [B, weekly; skipif pyebsdindex, download] PyEBSDIndex FILE-BASED
  end-to-end oracle (`TestAveragingOracle::test_file_based_
  pyebsdindex_end_to_end`; the class is Stage A, this one test lands
  with Stage B's weekly deliverables, plan.md 3.4): kikuchipy writes
  `nickel_ebsd_large` (corrected) to
  a kikuchipy h5ebsd file in `tmp_path`; PyEBSDIndex reads it
  (`ebsd_pattern.py:110-111`: `vendor >= 'kikuchipy'` ->
  `KIKUCHIPYH5`, class at `:1671`); `NLPAR(filename=..., lam=2.5,
  searchradius=3).calcnlpar()` runs its driver; the comparison is
  made on the float32 KERNEL output (`nlpar_nb` called on the array
  PyEBSDIndex read, with its sigma), NEVER on the file PyEBSDIndex
  writes: its writer's default `clip` path casts with `astype`
  (truncation, `ebsd_pattern.py:164`) and its other paths rescale
  (`:166-181`), so a written file cannot be bitwise compared to our
  `rint` policy (D6). The test asserts: patterns read back by
  PyEBSDIndex equal ours bitwise (the h5ebsd round trip), and the
  kernel output equals our float32 output bitwise. Also the tile-
  local saturation max of PyEBSDIndex's driver is in play here (its
  `chunksize` rows): the test sets `chunksize` to the full map so
  the threshold is global, and records that a smaller chunksize
  breaks parity on patterns near the tile boundary (the recorded
  quirk, not a defect of ours).
- Local oldest-matrix run once per stage (the ONE recipe string of
  requirements D10.5 and plan.md sections 0.3 and 5): `uv run
  --isolated --python 3.10 --with "numpy==1.23.0" --with
  "numba==0.57" --with "orix==0.12.1" --with "pyebsdindex==0.3.9.2"
  --with "dask==2021.8.1" --with "scikit-image==0.21.0" pytest
  tests/test_signals -k nlpar`; the CI oldest job (`tests.yml:48`)
  pins the same floors plus `hyperspy==2.2`; the dask pin is
  mandatory because the lazy path relies on
  `dask.array.overlap.ensure_minimum_chunksize`, `overlap` and
  `map_blocks` semantics; the first two were verified present and
  identical at the dask `2021.08.1` tag on 2026-10-04 (D8.4), and
  this run is the executable confirmation of all three.
  Both PyEBSDIndex kernels carry identical signatures and
  flags in 0.3.9.2 and 0.3.10.1 (parked-plan finding 2026-09-11;
  re-verified today for 0.3.10.1 only, 0.3.9.2 unverified today), so
  no version gate beyond the pyproject floor
  (`pyproject.toml:79`, `pyebsdindex >= 0.3.9.2, != 0.3.10`).
- Clean-replay grep gate, run at every stage close and before the
  fan-out replay (the executable form of the session plan's "`git
  grep ... develop..` finds nothing new" wording, since `git grep`
  takes a tree and `develop..` fails with "unable to resolve
  revision", verified 2026-10-04): `git diff develop...HEAD -- src
  tests doc examples benchmarks conftest.py CHANGELOG.rst | grep -E
  "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|
  validation\.md|tech-stack\.md| [DV][0-9]"` prints nothing (the root
  `conftest.py`, `benchmarks/` and `CHANGELOG.rst` are carried
  verbatim by the clean replay, so they are in the pathspec; the
  ` [DV][0-9]` alternative catches a bare D/V number; spec review
  E3-R3-2); comments in code state the fact or the measurement, never
  a D/V number.
- Full existing suite green before every stage-closing commit:
  `uv run pytest tests -n 4` (after a `-n 0` run of the NLPAR
  modules), recorded with counts.

## Requirement-to-test mapping

D-numbers as agreed with the requirements drafter: D1 API, D2 sigma,
D3 distances/weights, D4 window/edges, D5 lambda, D6 dtype, D7
kernels, D8 dask, D9 masks/1-D, D10 dependencies, D11 branch policy,
D12 docs, D13 measurement policy. The numbering matches
requirements.md (confirmed 2026-10-04 by the fixer); the V-numbers
and test names are the stable keys, and the test names here are the
naming authority for plan.md section 6 (requirements D1.9).

| Decision | Requirement (short) | Killer / evidence | Stage |
|---|---|---|---|
| D1 | `average_non_local_neighbour_patterns` / `get_nlpar_sigma` / `get_nlpar_lambda` signatures, `#824` keyword mapping in the docstring, `X \| Y` hints, no `print`, inplace/lazy_output/show_progressbar contract (incl. the lazy-input `lazy_output=False` case and the multi-chunk eager `store`), radius-0 warning, argument validation with the D1.6 message fragments (array `sigma`, `dtype_out`, bool/NumPy-integer radii), 0-D raises, in-place dtype change | V7 `test_argument_validation`, `test_stage_a_guards_raise_not_implemented`, `test_inplace_lazy_output_contract`, `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`, `test_inplace_with_dtype_out_changes_the_data_dtype`, `test_custom_attributes_are_carried`, `test_show_progressbar_registers_and_unregisters`, `test_sigma_argument_contract`, `TestSigmaMethod::test_returns_float32_of_navigation_shape`, `test_equals_the_pass_one_of_the_method`; V1 `test_search_radius_zero_warns_and_is_a_no_op`, reference agreement (per-axis radius order); V6 `test_lam_none_logs_the_optimised_lambda`, `test_target_weight_is_forwarded_through_lam_none` (caplog, capsys, no print); the conventions reviewer's `git grep -n -E "Optional\[\|Union\[\|print\("` on the three new files | A/B |
| D2 | sigma: clipped 3x3 min, `d2 > 0` guard, 1e12 fallback, 0.9961 threshold, mask forwarded, raw accumulators + NumPy normalisation, self-slot convention | V2 all (sigma and `_nlpar_normalized_distances` vs `dout`); V4 `test_sigma_recovery_median_ratio`, `test_mask_polarity_separates`; V7 `test_argument_validation` (sigma shape and sign), `TestSigmaMethod::test_forwards_mask_and_protection`; V10 `test_sigma_fallback_value_is_1e12` | A |
| D3 | distances and weights: per-pair n2, `max(d - dthresh, 0)`, self weight 1, `n2 == 0 -> +inf`, `dnorm <= 1e-8 -> 1e6 n2`, global saturation max, 0.999 threshold, no per-pattern rescale | V3 parity (incl. the `dthresh=0.5` arm) + `test_saturation_arm_separates_per_pair_n2_from_global_n`; V1 `test_weight_formula_on_injected_distances`, `test_power_of_two_scaling_is_exact`, reference agreement (`dthresh`, `saturation_protect` arms); V4 moments and unit-weight fraction; V5; V10 all incl. `test_tiny_dnorm_branch_gives_1e6_n2` | A |
| D4 | search window per axis, shift inward, clamp when the map is smaller than the window, 3x3 sigma window clipped, `_window_bounds(center, radius, n, shift)` | V0 `test_window_bounds_matches_pyebsdindex_expressions_and_clamps`; V3 `test_border_band_differs_from_clamp_and_zero_extend`, `test_calclim_from_block_info`, `test_small_map_padding_slots_are_inf`; V2 `test_sigma_window_is_clipped_not_shifted`; V7 `test_map_smaller_than_the_window`; V1 `test_huge_lambda_is_the_shifted_window_box_mean` | A |
| D5 | lambda: phantom-free objective (`S_i = 1 + sum`), target 0.34 default, Nelder-Mead bounds [1e-3, 10], stride, bound warning rule and message, logging, Stage A guards | V6 all incl. `test_phantom_free_objective_excludes_missing_neighbours`, `test_stride_above_1e6_points`, `test_lam_none_logs_the_optimised_lambda`; V7 `test_stage_a_guards_raise_not_implemented` (A); V4 `test_noise_reduction_is_monotone_in_lambda`; V8 `test_adp_improves` (auto vs 0.7) | B |
| D6 | dtype: input dtype by default via `rint` + clip, `dtype_out="float32"` raw, no rescale, in-place dtype change | V7 `test_dtype_round_trip`, `test_inplace_with_dtype_out_changes_the_data_dtype`; V1 `test_reference_agrees_with_the_method_on_random_maps`, `test_constant_map_is_an_identity`; V3 "zero pixels differ by >= 2" | A |
| D7 | kernels: five `@njit(cache=True, nogil=True)`, no parallel/fastmath, float32 literals, no `**` on float32, explicit float64 casts, `exp` isolated, compiled-oracle parity | V0 all (incl. the `ast.Pow` walk); V2/V3 compiled parity; V7 `test_scheduler_and_thread_invariance` | A |
| D8 | dask: two `overlap` + `map_blocks` passes with core-only wrappers, depth helper, `ensure_minimum_chunksize`, sigma operand, `block_info` halo sides from `chunk-location`/`num-chunks`, `boundary="none"` | V7 `TestDepthAndHalo::test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel`, `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk`, `test_depth_helper` (A); V2/V3 Ni compiled parity arms (A, download; the chunked eager route on a real map); V7 `test_lazy_equals_eager_*`, `test_issue_230_irregular_chunks` (B); V3 `test_calclim_from_block_info`; V10 `test_saturation_max_is_global` | A (depth helper, `block_info` wrappers, the multi-chunk drivers and the chunked eager route) / B (lazy input and output) |
| D9 | masks (True = excluded, cast with `np.asarray(..., bool)`) in sigma and distances only; 1-D navigation; per-axis radius in (row, col) order | V2 `test_sigma_mask_is_forwarded`; V4 `test_mask_polarity_separates`; V1 reference agreement (mask, `(1, 2)`/`(2, 1)`, 1-D arms); V7 `test_one_dimensional_navigation_equals_a_one_row_map` (`get_nlpar_sigma` shape `(n,)`), `test_argument_validation` (mask shape, all-excluded, integer mask); V6 `test_get_nlpar_lambda_accepts_injected_sigma_and_mask` | A |
| D10 | dependencies: `src/` never imports pyebsdindex for NLPAR; oracle tests skipif and compiled; no new required dependency; NRL derivation notice in the module docstring; GPL header | V0 `test_nlpar_code_never_names_pyebsdindex` (source of `_nlpar.py` and of the three methods); every oracle class carries the skipif marker (asserted by the conventions reviewer; `uv run pytest tests/test_signals -k nlpar` with pyebsdindex uninstalled in an isolated env is the recorded check at Stage A) | A |
| D11 | branch policy: `feat-NLPAR -> develop` PR #17, merge on the user's go, ubuntu/windows green, fan-out after merge, take-ours convergence with `#824`, supersedes 2026-09-11 decision 1 | Definition of done items; `gh pr view` output in the ledger | C |
| D12 | docs: `nlpar.ipynb`, gallery example, CHANGELOG with the fork PR link, `:cite:` key, Notes paragraph "Differences from PyEBSDIndex and from upstream PR #824", `hybrid_indexing.ipynb` never edited | Manual section; nbval on `nlpar.ipynb`; `sphinx-build -b html` exit 0; `git diff --name-only` never lists the never-sweep notebooks; V8 numbers quoted in the notebook match the ledger | C |
| D13 | measurement policy: MTP, seeds are not pins, recorded baselines never gates, numba-cache flake rule, oldest matrix per stage, clean-replay grep gate | this document; ledger entries per gate; V11 | A-C |

### Mutant killers (plan.md section 6 list; each must die by a NAMED default-suite test; the same rows, with module and class, are in plan.md section 6)

Stage A unless marked (B); `reference agreement` = V1
`TestIdentities::test_reference_agrees_with_the_method_on_random_maps`.

| mutant | killer |
|---|---|
| M1 sign of the sigma correction flipped (`d2 += n2 (s_0 + s_1)`) | V4 `TestNoiseOracle::test_normalised_distance_moments`; reference agreement; V2 `test_sigma_parity_compiled_*` (oracle, normalised distances) |
| M2 sigma used unsquared in d | V4 `TestNoiseOracle::test_sigma_recovery_median_ratio` (sigma_true 8, never 1); reference agreement; V3 `test_average_parity_compiled_*` (oracle) |
| M3 `sqrt(n2)` for `sqrt(2 n2)` | V4 `test_normalised_distance_moments` (std off by sqrt 2); reference agreement; V3 parity |
| M4 `d2 / n2` for the normalised form | V4 moments; reference agreement; V3 parity |
| M5 self weight 0 or self excluded | V1 `TestIdentities::test_injected_tiny_sigma_is_an_identity`, `test_huge_lambda_is_the_shifted_window_box_mean` |
| M6 drop `max(., 0)` | V1 `TestIdentities::test_weight_formula_on_injected_distances`; V4 `TestNoiseOracle::test_fraction_of_unit_weights`; [B] V10 `TestPolicyOracles::test_dthresh_is_consistent_between_kernel_and_objective` |
| M7 `exp(-d / lam)` for `exp(-d / lam^2)` | V1 `test_weight_formula_on_injected_distances` (`lam != 1`); [B] V6 `TestLambdaOracle::test_closed_form_lambda_on_constructed_distances` |
| M8 clamp instead of shift inward | V3 `TestAveragingOracle::test_border_band_differs_from_clamp_and_zero_extend` (oracle); V1 `test_huge_lambda_is_the_shifted_window_box_mean`; reference agreement |
| M9 zero-extend + window sums | same three |
| M10 (A, lazy arm B) depth sr with a tiny edge chunk | V7 `TestDepthAndHalo::test_depth_helper`, `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk[((3, 3, 4), (7, 7, 2))]` (A, pyebsdindex-free); [B] `TestLazyAndContracts::test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3` |
| M11 (A, lazy arm B) per-chunk saturation max | V10 `TestPolicyOracles::test_saturation_max_is_global` ([A] wrapper arm, [B] lazy arm); V7 `TestDepthAndHalo::test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel`, `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` (A, saturated pixels in one block); [B] `test_lazy_equals_eager_<chunking>` (row chunkings through the method; the saturated block survives `_reduce_chunks`) |
| M12 global N under saturation | V3 `TestAveragingOracle::test_saturation_arm_separates_per_pair_n2_from_global_n` (oracle); reference agreement `[saturation_protect=True]` on the saturated fixture |
| M13 `floor` for `rint` | V7 `TestLazyAndContracts::test_dtype_round_trip`; reference agreement |
| M14 per-pattern rescale | V7 `test_dtype_round_trip` (mean preserved), `test_inplace_with_dtype_out_changes_the_data_dtype`; V1 `test_constant_map_is_an_identity`, `test_injected_tiny_sigma_is_an_identity` |
| M15 mask polarity | V4 `TestNoiseOracle::test_mask_polarity_separates`; V2 `TestSigmaOracle::test_sigma_mask_is_forwarded` (oracle) |
| M16 mask not forwarded to sigma | V2 `test_sigma_mask_is_forwarded` (oracle); reference agreement `[signal_mask=circle]`; V7 `TestSigmaMethod::test_forwards_mask_and_protection` |
| M17 (B) phantom neighbours counted | V6 `TestLambdaOracle::test_phantom_free_objective_excludes_missing_neighbours` (value-level at fixed `lam`: exclusion exact, phantom-counting objective > 1e-3 apart); `test_phantom_deviation_is_measured_and_pinned` (oracle, download) |
| M18 (B) optimiser `max(d, dthresh)` | V10 `test_dthresh_is_consistent_between_kernel_and_objective` (dthresh 0.5) |
| M19 (B) objective statistic: median over points for `np.mean` | V6 `TestLambdaOracle::test_mixed_c_field_distinguishes_mean_from_median` (value-level on a 70/30 field at `lam` 1.5: `np.mean` exact, `np.median` > 1e-3 apart; no optimiser) |
| M20 shifted 3x3 sigma window | V2 `TestSigmaOracle::test_sigma_window_is_clipped_not_shifted` (clipped formula bitwise at every border pixel; shifted variant `<=` with strict inequality on >= 25 % of border pixels); reference agreement (the reference clips) |
| M21 `fastmath=True` | V0 `TestKernels::test_kernels_are_compiled_with_cache_and_nogil` (reviewed-only beyond the flag: no numeric test is promised to see it) |
| M22 float64 accumulation | V3 `test_average_parity_compiled_*` bitwise parity (the only killer; recorded reviewed-only for the formula, since float64 is "more correct" and every band test passes) |
| S1 halo returned with the core | V3 `TestAveragingOracle::test_calclim_from_block_info` (oracle); V7 `TestDepthAndHalo::test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel`, `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` (A, pyebsdindex-free); [B] V7 `test_lazy_equals_eager_<chunking>` (row chunkings through the method) |
| S2 `+inf` padding missing for a map smaller than the window | V7 `TestLazyAndContracts::test_map_smaller_than_the_window`; V3 `TestAveragingOracle::test_small_map_padding_slots_are_inf` |
| S3 1-D misrouted or per-axis radius in (col, row) order | reference agreement `[(1, 2)]` and `[(2, 1)]` (asserted unequal); V7 `test_one_dimensional_navigation_equals_a_one_row_map` (eager A, lazy [B]) |
| S4 (A, lazy arm B) lazy path skipping the minimum-chunksize rechunk | V7 `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk[(1, 1)]`, `[((3, 3, 4), (7, 7, 2))]` (A); [B] `TestLazyAndContracts::test_issue_230_irregular_chunks`; `test_lazy_equals_eager_<chunking>[(1, 1)]` (rows `(1,) * 10` through the method, thinner than the depth 6) |
| S5 sigma^2 from the mean or median of the eight estimates for the minimum | V2 `test_sigma_parity_compiled_*` (oracle); V4 `test_sigma_recovery_median_ratio` (mean estimator ratio ~1.00, outside the band); V5 `TestTwoGrain::test_cross_boundary_weights_are_exactly_zero` |
| S6 duplicate guard dropped (`d2 > 0` removed) | V2 `TestSigmaOracle::test_sigma_fallback_and_duplicates` (oracle); V10 `TestPolicyOracles::test_duplicate_neighbour_is_skipped_in_sigma_but_averaged` (NOT V1 `test_constant_map_is_an_identity`: a constant map returns itself with the guard dropped in both protection modes, spec review C3-R3-F1) |
| S7 saturation constant swapped (0.999 in sigma or 0.9961 in averaging) | V10 `test_uint16_two_threshold_arm`; V2 `test_sigma_saturation_threshold_constant` (oracle); V10 `test_sigma_fallback_value_is_1e12` (module constants) |
| S8 (A, lazy arm B) depth off by one (`r - 1` or `r + 1`, halo logic unchanged) | V7 `TestDepthAndHalo::test_depth_helper`, `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` (A); [B] `test_lazy_equals_eager_<chunking>` (row chunkings through the method) |

## Performance (recorded baselines, never gates -- D13)

| measurement | recipe | recorded value |
|---|---|---|
| `get_nlpar_sigma()` on `nickel_ebsd_large`, sr context 3, dask threads | V11 `test_runtime_is_recorded`, best of 3 after warm-up | (unmeasured; PyEBSDIndex seed 0.12 s single thread, 2026-09-11) |
| `average_non_local_neighbour_patterns(lam=2.5)` on Ni, sr=3, dask threads | same | (unmeasured; estimate 0.2-0.4 s; PyEBSDIndex `nlpar_nb` seed 0.54 s single thread) |
| pure-NumPy `nlpar_reference` on the full Ni map, sr=3 | the V1 test-local reference run once at Stage A to justify its small-map use | (6-14 s seed, 2026-09-11) |
| `si_wafer` lazy end to end | V9, weekly | (5-10 s estimate) |
| PyEBSDIndex JIT warm-up, both kernels | `pyebsdindex_kernels` fixture `record_property` | (unmeasured) |
| default-suite addition per worker | `uv run pytest tests/test_signals -k nlpar -n 0 --durations=0` | (budget <= ~90 s) |

## Manual

- User review of the plan.md section 0 constitution amendments at
  the spec commit (the approval gate of the parked plan Step 1 is
  the user's approval of `plan.md`).
- Tutorial `doc/tutorials/nlpar.ipynb` (Stage C) renders and passes
  nbval locally (`doc/tutorials/run_nbval.sh` entry,
  `tutorials_sanitize.cfg` regexes for timings and chunk counts;
  stored outputs only if the estimated Read the Docs execution
  exceeds ~2 min, tech-stack.md:52). Review items, each ticked by
  eye with the figure named:
  - the two-grain synthetic map: before / NLPAR / Gaussian
    `average_neighbour_patterns`, showing boundary preservation vs
    Gaussian blur (V5), with the weight map of one boundary pattern;
  - the sigma map of the synthetic map and of `nickel_ebsd_large`
    (raw and background-corrected), with a colour bar in grey
    levels;
  - the lambda-vs-target curve on `nickel_ebsd_large` (targets 0.1
    to 0.9), marking 0.34 and the PyEBSDIndex-style value, so the
    reader sees the monotonicity V6 asserts;
  - before/after patterns (three positions: interior, edge, grain
    boundary) and the IQ/ADP maps before/after (V8 numbers quoted
    from the ledger, IQ at <= 3 decimals);
  - background removal + Hough indexing before/after on the 165-pt
    grid with the IPF maps and the `pq`/`cm` histograms (V8);
  - `si_wafer` section runs lazily or quotes the ledger if the RTD
    budget forbids the download;
  - the parameter-guidance paragraph (search radius, lambda,
    dthresh, saturation) and the "Differences from PyEBSDIndex and
    from upstream PR #824" paragraph match the method docstring
    wording;
  - the acknowledgement of PyEBSDIndex / NRL and the
    `:cite:`brewick2019nlpar`` render (bibliography key at
    `doc/user/bibliography.bib:11`);
  - `hybrid_indexing.ipynb` is NOT edited; `nlpar.ipynb` links to
    it.
- Gallery example `examples/pattern_processing/nlpar.py` runs under
  sphinx-gallery and shows one before/after pair.
- One end-user smoke run from a notebook: `s.average_non_local_
  neighbour_patterns()` on `nickel_ebsd_large` with the progress
  bar, eyeballing that grain boundaries stay sharp and the logged
  lambda is the ledger's value.

## Definition of done

- Per stage: failing tests committed first (skeletons raise
  `NotImplementedError`, placeholders `None`); implementation;
  two adversarial reviewers (fidelity vs the paper, both PyEBSDIndex
  kernels and EMsoftOO `mod_NLPAR.f90`; conventions/integration) +
  bug injection of the stage's mutants applied ALONE in the main
  tree, each dying by its named test, survivors -> strengthened
  tests with kills verified by re-injection; disposition tables
  appended to plan.md; `uvx pre-commit run --files <explicit list,
  never specs/>` clean; coverage 100 % of
  `src/kikuchipy/pattern/_nlpar.py` with the command output
  recorded (`uv run pytest tests/test_signals -k nlpar --cov=
  kikuchipy.pattern._nlpar --cov-report=term-missing -n 0`); full
  existing suite green (`uv run pytest tests -n 4`, counts
  recorded); the local oldest-matrix recipe run and recorded; the
  numba-cache flake rule applied (`-n 0`, then `-n 4`, red tests
  re-run alone); the clean-replay grep gate recorded; signed commits
  (`git commit -s`, trailer `Co-Authored-By: Claude Fable 5.1
  <noreply@anthropic.com>`) pushed to `origin/feat-NLPAR` after each
  stage, a failing-tests commit never pushed alone; roadmap feature-
  path boxes ticked with ledger evidence.
- Every `MTP` placeholder in the inventory above replaced by a dated
  measured value in Recorded results (recipe + machine ID); every
  seed either confirmed or amended in requirements.md with the
  date; the V8 Hough pins each labelled "improvement (>= 0.5 x
  measured gain)" or "not worse" in the ledger.
- Stage A: V0, V1, V2, V3 (both incl. their `nickel_ebsd_large` arms
  [A, download]), V4, V5, the V7 [A] arms (incl. `TestSigmaMethod`,
  `test_argument_validation`,
  `test_stage_a_guards_raise_not_implemented`, the two
  `TestDepthAndHalo` multi-chunk driver tests and
  `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`)
  and the V10 [A] arms green with pyebsdindex installed AND with it
  absent (skips counted and recorded); the root `conftest.py`
  generator fixtures and the shared `exp_kernel_ulp` fixture in
  place;
  `EBSD.average_non_local_neighbour_patterns` (eager
  input, eager output; the Stage A guards of requirements D5.7 in
  place) and `EBSD.get_nlpar_sigma` shipped, `EBSD.get_nlpar_lambda`
  present as the `NotImplementedError` stub (the fifth guard);
  CHANGELOG API bullet with the fork PR link.
- Stage B: V6, the V7 [B] arms (lazy), the V10 [B] arms, V8, V9
  (weekly), V11 recorded; `EBSD.get_nlpar_lambda` and `lam=None`
  shipped; the Stage A guards and their pin deleted; performance table
  filled; `ni_gain` ledger entry written or "not cached" recorded.
- Stage C: tutorial + gallery example + CHANGELOG tutorial bullet;
  nbval green; `sphinx-build -b html` exit 0; the Phase-11-style
  validation matrix + failure-mode review recorded; the Manual list
  ticked.
- **The three spec documents re-submitted to adversarial review
  after Stage C** (fidelity + conventions critics, fixer disposition
  table appended to plan.md).
- `git log origin/feat-NLPAR..feat-NLPAR` empty; PR `feat-NLPAR ->
  develop` open (`gh pr view` recorded, number confirmed, CHANGELOG
  link updated if not #17) with ubuntu and windows CI green and the
  macOS failures identified as the two known pre-existing ones;
  merge awaits the user. Fan-out (merge develop into hrebsd-dic;
  replay as 2 clean commits onto `feat-spherical-indexing-nlpar`) is
  AFTER the merge and outside this definition of done.
- Memory notes updated after the fan-out, per the session plan:
  `nlpar-parked-plan.md` (done; base, PR policy, `#824` alignment,
  fan-out shas, PyEBSDIndex quirks), `feat-spherical-indexing-
  branch.md` (the sibling branch `feat-spherical-indexing-nlpar`),
  `hrebsd-dic-project.md` (develop merged in with NLPAR),
  `hrosm-parked-plan.md` (follow the develop-first plus fan-out
  pattern; amend its Step 0 when HROSM resumes).

## Recorded results

Append-only ledger. Every entry is dated, names the machine (CPU
class + OS at minimum), the exact recipe or command, and which `MTP`
placeholder or decision it fills. Refutations of frozen decisions
are recorded here AND amended in requirements.md with the same
date.

### 1. 2026-10-04 (drafting)

Machine: the 20-core Windows 11 laptop of the spherical and HREBSD
phases, Intel64 Family 6 Model 186 (Raptor Lake), 20 logical cores,
Windows 11 build 26200, `.venv` Python 3.13.12, numpy 2.4.6, scipy
1.17.1, numba 0.65.1, dask 2026.3.0, hyperspy 2.4.0, orix 0.14.2,
pooch 1.9.0, pytest-benchmark 5.2.3, pyebsdindex 0.3.10.1 (all
printed by `uv run --no-sync python -c "import ...; print(
__version__)"` today). No test was executed and nothing was timed;
everything below is a line reading or a read-only Python one-liner.

1. **PyEBSDIndex oracle kernels re-verified** (`.venv/Lib/
   site-packages/pyebsdindex/nlpar_cpu.py`): `NLPAR.sigma_numba`
   `py_func` spans lines 752-818, `NLPAR.nlpar_nb` 820-936,
   `opt_lambda_cpu` 94-184 (`inspect.getsourcelines`). Both kernels
   are decorated `@staticmethod` + `@numba.jit(nopython=True,
   cache=True, fastmath=False, parallel=True)` (`:753, :821`);
   `targetoptions` read from the dispatchers: `{'nopython': True,
   'fastmath': False, 'parallel': True, 'boundscheck': None}`, and
   both expose `.py_func`. Constants: `0.9961` (`:767`), `0.999`
   (`:869`), `d2 >= 1.e-3` (`:796`), `mind = 1e24` (`:776`), `dnorm >
   1.e-8` else `1e6 * n2` (`:912-915`), self slot `-1.0e6` (`:893`),
   `max(w - dthresh, 0)` then `exp(-1.0 * w * lam2)` (`:924-925`),
   `lam2 = 1.0 / lam**2` float64 (`:847`). `loptfunc` uses `np.maximum(
   d2, dthresh)` (`:106-107`); `stride = 1 if sigma.size < 1e6 else
   2` (`:164`); Nelder-Mead with `bounds=[[0.001, 10.0]]`,
   `fatol 0.0001` (`:168-169`); the median of the three target fits
   becomes `self.lam` (`:181`; corrected from `:180` on 2026-10-04 by
   the round-3 fixer, F3-R3-5, `:180` is the `if autoupdate == True:`
   line). Phantom slots: `dout`/`nout` are
   allocated `(nn*2+1)**2` wide and zero-filled (`:756-757`); the
   normalisation loop skips `nout == 0` (`:812`), so out-of-map
   slots stay 0 and weigh 1 in `loptfunc`. 0.3.9.2 was NOT
   re-verified today (the 2026-09-11 finding "identical flags in
   0.3.9.2" stands as recorded).
2. **PyEBSDIndex reads kikuchipy h5ebsd and its writer truncates**:
   `ebsd_pattern.py:110-111` (`vendor >= 'kikuchipy'` ->
   `KIKUCHIPYH5(path)`, class at `:1671`); the pattern-cast helper's
   `clip` branch is `pats.clip(minval, maxval).astype(typeout)`
   (`:164`, truncation) while `fullscale`/`scale` use `np.around`
   (`:171, :180`); which branch `write_data` takes by default was
   not traced (unverified), hence the rule "compare on the kernel
   output, never on the written file".
3. **Upstream PR #824 diff** (`scratchpad/pr824.diff`, 914 lines,
   6 files: `doc/dev/code_style.rst`, `src/kikuchipy/pattern/
   _nlpar.py`, `pattern/chunk.py`, `signals/ebsd.py`, `tests/test_io/
   test_kikuchipy_h5ebsd.py`, `tests/test_signals/test_ebsd.py`):
   border replication `np.pad(..., mode="edge")` on patterns and
   sigma (diff lines 267-276); per-pattern min-max rescale through
   `_rescale_neighbour_averaged_patterns` with `window_sums=ones`
   (354-360); `dthresh=0.0` passed to the lambda fit only (564) and
   `loptfunc` with `np.maximum(d2, dthresh)` (171-172); the search
   window cast `astype(bool)` (506); `np.std(..., astype=...)`
   fallback for sigma when pyebsdindex is absent (120);
   `print` in the optimiser (225-230); `InputError` undefined (100).
   Observation for the requirements drafter (not in the task's
   defect list): `indices = np.arange(npix) * ~signal_mask.flatten()`
   (104) maps masked pixels to index 0 instead of dropping them.
   Recorded, not reproduced (D3/D4/D6). (Three line numbers
   corrected in place 2026-10-04 by the round-2 fixer, F2-F7,
   verified with `awk`: 100, 104, 120.)
4. **kikuchipy facts**: `average_neighbour_patterns` at
   `src/kikuchipy/signals/ebsd.py:954-1122`, `downsample` starts at
   `:1124`; the `lazy_output and inplace` guard at `:1018-1019`; the
   radius-0 warning at `:1028-1034`; `window_sums` second operand
   `:1063-1065`; `map_overlap(..., boundary="none")` `:1082-1093`;
   `rechunk(old_chunks)` `:1107`; `_get_custom_attributes` `:1111`;
   `extract_grid` takes `(n columns, n rows)` (`:279-292`; its measured
   behaviour on `nickel_ebsd_large` is in entry 3 item 5 and the suite
   does not use it).
   `pattern/chunk.py:130` `_average_neighbour_patterns`, `:148`
   `_rescale_neighbour_averaged_patterns` (`@njit(cache=True,
   fastmath=True, nogil=True)`, the rescale we do NOT reuse).
   `signals/util/_dask.py:198-216` `_get_chunk_overlap_depth`.
   Gating pattern `tests/test_signals/test_ebsd_hough_indexing.py:
   35-37`; kernel-flag helpers `tests/test_indexing/test_spherical_
   euler.py:72-96, 509-529`. `brewick2019nlpar` at
   `doc/user/bibliography.bib:11-19`. `pyproject.toml:79`
   `pyebsdindex >= 0.3.9.2, != 0.3.10`; `weekly` marker
   `pyproject.toml:203-204`, `--weekly` option `conftest.py:89-92`
   (repo root; there is no `tests/conftest.py`); `dummy_signal`
   fixture (3, 3 | 3, 3) `conftest.py:196-205`. CI matrix
   `.github/workflows/tests.yml:43` ubuntu/windows/macos (line
   number corrected 2026-10-04 by the fixer, F1-F16), oldest
   dependencies line 48 (`dask==2021.8.1 ... numba==0.57
   numpy==1.23.0 orix==0.12.1 pooch==1.3.0 pyebsdindex==0.3.9.2
   scikit-image==0.21.0`, Python 3.10), `pyebsdindex[gpu]` install at
   line 100. `tests/test_signals/test_util/` exists (`test_array_
   tools.py`, `test_dask.py`, `test_overwrite_hyperspy_methods.py`);
   `benchmarks/` holds `indexing/` only. Issue-230 chunk tuple at
   `tests/test_signals/test_ebsd.py:1617-1620`
   (`test_average_neighbour_patterns_lazy`, docstring "Fixes
   https://github.com/pyxem/kikuchipy/issues/230"; the issue text
   itself unverified today).
5. **Data facts (read-only Python)**: `kp.data.nickel_ebsd_large()`
   is cached: navigation (55, 75), signal (60, 60), uint8, max 253,
   min 21; its `xmap` has shape (55, 75) and props `['scores',
   'z']` (provenance NOT settled: roadmap Phase 11 says Hough +
   refined 0.8.0, the parked plan says DI-refined; V8 fills this at
   Stage B). `get_dask_array(signal=s, chunk_bytes=8e6,
   rechunk=True).chunks == ((47, 8), (47, 28), (60,), (60,))` on
   dask 2026.3.0: **the parked plan's `(26, 26, 3)` last-chunk case
   is NOT this call's default today**, so V7 pins it as an explicit
   rechunk and adds the measured default as a second arm.
   `s.extract_grid((11, 15))` was read off the axes manager as
   "navigation shape (15, 11) rows x cols, 165 patterns"; CORRECTED
   in place 2026-10-04 by the round-2 fixer (F2-F2, re-measured): the
   call returns an internally inconsistent signal, data `(13, 10, 60,
   60)` (130 patterns), axes manager `(11, 15)`, `xmap` `(49, 64)`;
   the suite uses `s.inav[::5, ::5]` instead (entry 3). `kp.data.
   si_wafer(lazy=True)` is cached: (50, 50 | 480, 480) uint8.
   `ensure_minimum_chunksize` imports from `dask.array.overlap`.
6. **Arithmetic (not measurements)**: closed-form lambda(tw) =
   sqrt(-c / ln((1/tw - 1)/8)): c=2 -> 0.9807 / 1.1884 / 1.4280,
   c=5 -> 1.5506 / 1.8790 / 2.2578, c=12 -> 2.4022 / 2.9110 /
   3.4978 for tw 0.5 / 0.34 / 0.25. float32 `exp` underflows to
   exactly 0.0 at -104.0 and returns the denormal 1e-45 at -103.9.
   Two-grain cross-boundary d at Delta 30, sigma 8, N 1024 = 159.1;
   d / lam^2 = 325 / 159 / 110 / 25.5 at lam 0.7 / 1.0 / 1.2 / 2.5.
   iid-noise sigma^2 estimator std factor sqrt(2/N) = 0.0442 (N
   1024), 0.0236 (N 3600), 0.00295 (N 230400); the -1.0 to -1.4 std
   bias of the min over 8 neighbours gives sigma_hat / sigma_true
   0.969-0.978 (N 1024), 0.983-0.988 (N 3600), 0.998-0.999 (N
   230400). Phantom slots on (55, 75): 776 of 37125, reproducing the
   parked plan's count. Saturation thresholds: uint8 max 253 ->
   252.0 (0.9961) and 252.7 (0.999), both exclude only 253; uint16
   max 65535 -> 65279.4 and 65469.5.
7. **Seed provenance caveat**: the parked plan carries two
   with/without-phantom lambda pairs at target 0.34 for
   `nickel_ebsd_large`: 1.098 / 1.121 in its Reference facts and
   1.1164 / 1.1387 (raw) plus 2.5246 / 2.5787 (corrected) in its V6
   line. The V6 pair is the binding seed (task instruction); the
   ratio is 1.021 in both, so only the configuration of the first
   pair is unverified. The Stage B measurement records which route
   (raw vs corrected, tile-local vs global max) each corresponds to.
8. **Open reconciliation items for the fixer** (also flagged
   inline): (a) oracle route split `.py_func` for small maps vs
   compiled for Ni (Automated section); (b) the D-numbering
   assumption (Requirement-to-test mapping); (c) the `(26, 26, 3)`
   chunking is explicit, not default (V7); (d) stored-xmap
   provenance (V8); (e) `#824` masked-index observation (item 3).

### 2. 2026-10-04 (spec review round 1, fixer)

Machine: as entry 1. Read-only Python (`uv run --no-sync python`);
the critics' probe `scratchpad/critic_numerics_probe.py` re-run by
the fixer; nothing under the repository was executed or written
outside `specs/2026-10-04-nlpar/`. Resolutions of entry 1 item 8:
(a) ONE oracle route, compiled (item 1 below); (b) D-numbering
matches requirements.md; (c) V7 pins the explicit `(26, 26, 3)`
rechunk and the measured default as two arms (unchanged); (d) V8
fills the provenance at Stage B (unchanged); (e) recorded in
requirements Context defect 9 (unchanged).

1. **PyEBSDIndex `.py_func` is NOT a bitwise oracle** (critic E1,
   re-run): on a (7, 7 | 8, 8) float32 map with `lam=0.7`, `sr=3`,
   `dthresh=0`, `NLPAR.sigma_numba` compiled vs `.py_func`: `sigma`
   bitwise True, `dout` bitwise False, `nout` bitwise True;
   `NLPAR.nlpar_nb` compiled vs `.py_func`: bitwise False, 2586 of
   3136 pixels differ, max abs 1.07e-4, identical outcome for a
   Python-float and a `np.float32` `lam`. Cause (numba type
   inference on transcriptions of `:788-793` and `:904-909`): numba
   types `float32 ** int32` as float32 while NumPy gives float64;
   numba types `2.0 * float32` as float64 while NumPy 2 gives
   float32; numba promotes the `n2 += 1.0` counter of `sigma_numba`
   to float64 (compiled counter `64.000000000001`, `.py_func`
   `64.0`). Interior self slot `dout = -5.656854 = -sqrt(64 / 2)`,
   `nout = 64`; corner `nout = [64, 64, 64, 64, 0, 0, 0, 0, 0]`
   (compact enumeration). Cold compile 4.0 s (`sigma_numba`) + 3.2 s
   (`nlpar_nb`) in the critic's run; cached in the fixer's re-run.
   Fills: the oracle route (Automated section), requirements D7.5,
   D10.2.
2. **dask route for the two passes** (fixer): on dask 2026.3.0,
   `overlap(x, depth={0: 1, 1: 1, 2: 0, 3: 0}, boundary="none")` on a
   `(10, 12, 4, 4)` float32 array with chunks `((3, 3, 4), (5, 7),
   (4,), (4,))` gives overlapped chunks `((4, 5, 5), (6, 8))`;
   `map_blocks(f, overlapped, chunks=x.chunks[:2] + ((4,), (9,)),
   dtype=float32, meta=np.empty((0, 0, 4, 9), float32))` with `f`
   returning its core computes shape `(10, 12, 4, 9)` with the core
   values in place, makes zero calls with `block_info=None`, and
   `block_info[0]` carries `array-location`, `chunk-location`,
   `num-chunks`, `shape`. `ensure_minimum_chunksize(6, (26, 26, 3))
   == (26, 23, 6)`, `(4, (26, 26, 3)) == (26, 25, 4)`, `(5, (47, 8))
   == (47, 8)`. Depth rule values: `(47, 8)` r3 -> 3, r4 -> 4, r5 ->
   5; `(26, 26, 3)` r3 -> 4, r4 -> 6. Fills: requirements D8.2, D8.3,
   D8.5, V7 `test_depth_helper` cases.
3. **Oracle source facts re-read** (fixer, `nlpar_cpu.py`):
   `sigma_numba(data, nn, nrows, ncols, rowstartcount, colstartcount,
   indices, saturation_protect=True)` (`:754`), `n0 = float32(
   shpdata[-1])` (`:761`), `nout[j, i, count] = n0` for every slot
   before the non-self branch (`:786`); `nlpar_nb(data, lam, sr,
   dthresh, sigma, nrows, ncols, calclim=<mutable default>,
   indices_in, saturation_protect=True, diff_offset=float32(0))`
   (`:822-824`), `calclim = [cstart, rstart, ncolcalc, nrowcalc]`
   (`:835`); the driver casts `lam` and `dthresh` to float32
   (`:297, :303`) and calls `sigma_numba(data, 1, nrowchunk,
   ncolchunk, [0, nrowchunk], [0, ncolchunk], indices,
   saturation_protect)` (`:487-490`); `loptfunc` adds `1e-12` to the
   weight sum (`:109`) and takes `np.mean` over points (`:111`); the
   median at `:181` is over the three target fits. Fills: the oracle
   call recipe, requirements D2.5, D5.1, D7.3.
4. **Small facts** (fixer): `git grep -n -E "specs/" develop.. -- src`
   fails with "unable to resolve revision: develop.." (exit 128), so
   the executable gate is the `git diff develop...HEAD | grep` form;
   `doc/tutorials/tutorials_sanitize.cfg` has 9 `[regexN]` sections;
   `.github/workflows/tests.yml` has the `os:` matrix at line 43 and
   the oldest dependencies at line 48 (entry 1 item 4 corrected in
   place); the approved session plan's verification grep (its line
   250) names `tests/test_signals/test_util/test_nlpar.py`.

### 3. 2026-10-04 (spec review round 2, fixer)

Machine: as entry 1; scipy 1.17.1. Read-only Python
(`scratchpad/fixer2_probe.py`, `uv run --no-sync python`); nothing
under the repository was executed or written outside
`specs/2026-10-04-nlpar/`. Disposition table: plan.md section 10,
round 2.

1. **Bound arms of the lambda optimiser** (F2-F1, C2-F3): bounded
   Nelder-Mead (`x0 = 1.0`, `fatol` 1e-4, bounds `[1e-3, 10]`) on the
   D5.1 objective, (10, 10, 9) fields, `tw = 0.34`: every non-self
   slot at `c = 200` -> `x = 1.000000`, `fun = 0.66`, `nit` 10 (flat
   objective, the start point is returned; NOT a bound hit); four
   slots at `c = 1` and four at `c = 250` -> `x = 1.1761`, `fun`
   2.9e-6 (inside the bounds); one slot at `c = 1`, seven at `+inf`
   -> `x = 10.0`, `fun` 0.1625, `nit` 9 (UPPER bound hit); all slots
   at `c = 1e-4` -> `x = 0.00840` (no bound), at `1e-6` -> `x =
   0.001`, at `1e-7` -> `x = 0.001` (LOWER bound hit). Closed form
   `lam(0.34) = sqrt(c / 1.41617)`: 2.66e-4 at `c = 1e-7`; the lower
   bound is hit for `c < 1.416e-6`. Fills: V6 `test_bound_hit_warns`,
   requirements D5.4.
2. **Mean vs median objective** (C2-F2, E2-F7): 70/30 field (`c = 2`
   / `c = 12`, all valid) at fixed `lam = 1.5`: mean of the per-point
   terms 0.2616, median 0.1068; the mean-objective minimiser from
   `x0 = 1.0` is 1.1884 (the kink of the majority group's closed
   form), `fun` 0.1975, so "strictly between 1.1884 and 2.9110" is
   false and the V6 test is value-level. Fills: V6 `test_mixed_c_
   field_distinguishes_mean_from_median`, M19 rows.
3. **Phantom-free objective** (E2-F6): on the (10, 10) border-masked
   constant field (`c = 2`, invalid slots at `d = 0.0`), the objective
   equals the `+inf` variant EXACTLY at `lam` 1.0 (0.186599) and 2.0
   (0.138274); the phantom-counting objective is 0.137427 / 0.180023,
   i.e. 0.049 / 0.042 apart. Fills: V6 `test_phantom_free_objective_
   excludes_missing_neighbours`, M17 rows.
4. **Constant-map accumulation** (E2-F2, C2-F6): sequential float32
   sum of `k` copies of `fl(fl(1/k) p)`, `p` in 1..254: worst 2 ulp
   (`k` 9), 3 ulp (`k` 20), 11 ulp (`k` 49) in this probe (critic
   E2's probe, a slightly different operation order: 3 / 4 / 13). The
   "<= 2 ulp" bound was wrong beyond 9 weights; the bound is linear
   in `n_window`. Fills: V1 constant-map and box-mean bullets and
   table.
5. **`extract_grid` vs `inav` on `nickel_ebsd_large`** (F2-F2):
   `s.extract_grid((11, 15), return_indices=True)` -> data `(13, 10,
   60, 60)`, `axes_manager.navigation_shape (11, 15)`, indices `(2,
   13, 10)`, `xmap.shape (49, 64)`, repr `(11, 15|60, 60)`:
   internally inconsistent (`ebsd.py:329-332` reverses `grid_shape`
   before `grid_indices`). `s.inav[::5, ::5]` -> data `(11, 15, 60,
   60)`, `navigation_shape (15, 11)` (hyperspy x|y), `xmap.size 165`,
   `xmap.shape (51, 71)` (orix extent shape under step slicing),
   `xmap.phases_in_data.names ['ni']`, detector `EBSDDetector`,
   static background `(60, 60)` carried. Fills: V8 header, plan 3.4
   and 4.1.5, requirements Scope and D12.1; entry 1 item 5 corrected.
6. **Test-sharing facts** (E2-F1, F2-F3): `pyproject.toml:167`
   `"--import-mode=importlib"`; no `__init__.py` anywhere under
   `tests/`; no `tests/conftest.py` or `tests/test_signals/conftest.py`
   (the only conftest is the repository root's, `dummy_signal` at
   `conftest.py:195-205`). Fills: the fixture-generator mechanism.
7. **Line checks** (F2-F7, F2-F8, C2-F11, C2-F5): `pr824.diff` line
   100 `raise InputError(...)`, 104 `indices = np.arange(npix) *
   ~signal_mask.flatten()`, 120 `sigma = np.std(..., astype =
   np.float32)`; `gnomonic_correction.py:52-54` is the
   `NUMBA_CACHE_DIR` redirect (assignment at 54), `band_detect.py:
   60-62` likewise; the parked plan's status paragraph spans lines
   3-10 with `## Context` at line 12; `nlpar_cpu.py:785` `n2 =
   np.float32(1.0e-12)` (sigma kernel) vs `:903` `n2 = np.float32(
   0.0)` (averaging kernel), `:797` `s0 = d2 / np.float32(n2 * 2.0)`,
   `:763-767` `mxval = np.max(data)` then `mxval *= 0.9961` (float64
   literal), `:865-869` `mxval *= np.float32(0.999)`.

### 4. 2026-10-04 (spec review round 3, fixer)

Machine: as entry 1; dask 2026.3.0. Read-only Python (`uv run
--no-sync python`): the fixer's `scratchpad/fixer3_probe.py` and
critic C3's `scratchpad/critic3_constant_map_probe.py` and
`scratchpad/critic3_store_race_probe.py` re-run; one fetch of
`https://raw.githubusercontent.com/dask/dask/2021.08.1/dask/array/
overlap.py`; nothing under the repository was executed or written
outside `specs/2026-10-04-nlpar/`. Disposition table: plan.md
section 10, round 3.

1. **`get_dask_array(rechunk=True)` on a lazy input runs
   `_reduce_chunks`** (F3-R3-1, C3-R3-F2; `_dask.py:142-151,
   163-195`): on a 2-D navigation the axis with the smaller chunksize
   keeps its chunks and the other is re-chunked `"auto"` under the
   8e6-byte limit. `LazyEBSD(da.zeros(..., chunks=c, dtype=uint8))`
   through `get_dask_array(signal=s, chunk_bytes=8e6, rechunk=True)
   .chunks[:2]`: (10, 16 | 16, 16): `((3, 3, 4), (7, 7, 2))` ->
   `((3, 3, 4), (16,))`; `((5, 5), (8, 8))` -> `((5, 5), (16,))`;
   `((1, 9), (2, 14))` -> `((1, 9), (16,))`; `((1,) * 10, (1,) *
   16)` -> `((1,) * 10, (16,))`; `((2,) * 5, (3, 3, 3, 3, 3, 1))` ->
   `((2,) * 5, (16,))`; `((4, 4, 2), (4,) * 4)` -> `((4, 4, 2),
   (16,))`; (55, 75 | 60, 60): `((26, 26, 3), (75,))` -> `((26, 26,
   3), (40, 35))`; (55, 75 | 6, 6): the issue-230 tuple unchanged.
   Signal axes always `((h,), (w,))`. Fills: requirements D8.1, D8.6;
   V7 header and [B] bullets; plan 0.2, 3.3, section 6 rows.
2. **`ensure_minimum_chunksize`** (E3-R3-1, C3-R3-F4): `(4, (3, 3,
   4)) == (6, 4)`, `(6, (7, 7, 2)) == (7, 9)`, `(6, (1,) * 16) ==
   (6, 10)`, `(6, (1,) * 10) == (10,)`, `(1, (1,) * 10)` unchanged,
   `(4, (26, 26, 3)) == (26, 25, 4)`. The function at the dask
   `2021.08.1` tag has the same signature, docstring examples and
   body as 2026.3.0 (`for c in chunks` loop, `if new > size + (size -
   c)` branch, `output[-1] += new` tail, the `ValueError` for a depth
   larger than the array), and `overlap(x, depth, boundary)` is
   defined in the same file. Fills: requirements D8.3, D8.4; plan 2.8,
   2.11, 7.5; V7 `test_pass_two_driver_*`.
3. **Eager in-place `store` on a multi-chunk in-memory signal**
   (C3-R3-F3): `average_neighbour_patterns(inplace=True)` on
   `EBSD(rng.integers(20, 240, (55, 75, 60, 60), uint8))` (chunked
   `((47, 8), (47, 28))` by `get_dask_array`) equals `inplace=False`
   bitwise under `scheduler="synchronous"` and `"threads"` with
   `num_workers` 1, 2 and 20: 0 differing patterns, max abs 0; a toy
   `map_overlap(depth=1, boundary="none")` + `store` into its own
   (40, 40) float32 source equals the computed result under the same
   four settings. Fills: requirements D1.5 (b); V7
   `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`.
4. **Constant map on the compiled oracle** (C3-R3-F1): (4, 5 | 6, 6)
   float32 map at 100, `sr=1`, `lam=1.0`: `saturation_protect=True`
   -> sigma all `1e12`, `nout` of every neighbour slot at the
   `1e-12` seed (no kept pair); `saturation_protect=False` -> sigma
   all `1e12`, `nout = 36`; `100 < 0.9961 * 100` and `100 <
   float32(100) * float32(0.999)` are both False. Fills: requirements
   D2.4, D3.5; V1 `test_constant_map_is_an_identity`; S6 rows.
5. **Arithmetic** (F3-R3-2): `8 exp(-200) = 1.1e-86` (representable
   in float64, the minimum normal is 2.2e-308); `8 exp(-200 / lam^2)
   < 2^-53` iff `lam < 2.2699`, so `1 + 8 w == 1.0` exactly and the
   D5.4 objective is flat around `x0 = 1.0`. Fills: V6
   `test_bound_hit_warns` wording.
6. **Line checks** (F3-R3-5, E3-R3-6): `nlpar_cpu.py:180` `if
   autoupdate == True:`, `:181` `self.lam = np.median(
   lamopt_values)`; `_data.py:392` `def si_wafer(`, `:449` `return
   load(file_path, **kwargs)`; `git diff feat-spherical-indexing
   develop -- src/kikuchipy/signals/ebsd.py`: hunks `@@ -1673,17
   +1673,13 @@` and `@@ -1776,12 +1772,8 @@`, 4 insertions, 12
   deletions; `ebsd.py:1081` `dtype_range[dtype_out.type]`. Fills:
   ledger entry 1 item 1, requirements Scope, V9, plan 1, 2.7, 2.12.

This section is filled at each stage's failing-tests gate
(placeholder inventory confirmed), implementation gate
(measurements + pins with recipes and machine) and review gate
(re-measurements, disposition pointers), each in its own numbered,
dated subsection.
