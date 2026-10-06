# HROSM -- `feat-HROSM`: validation

**Status note (requirements D17): drafted 2026-10-06 with read-only
verification and ZERO test execution.** Every tolerance marked `MTP`
(measured-then-pinned) is filled at the owning stage's failing-tests
or implementation gate with `pytest.approx(measured, rel=0.05)` or
the ~2x margin convention (`tech-stack.md:51`); until then the test
carries a `None` placeholder that raises "unfilled MEASURED-THEN-PINNED
placeholder" (the NLPAR and HREBSD precedent). Numbers quoted as
"seed" are drafting values (parked plan 2026-09-28, the requirements
probes of 2026-10-06, or arithmetic in this file); a seed tells the
measurer the expected scale, it is never a pin. Bitwise expectations
are not MTP unless the row says "bitwise (MTP)": those are bitwise by
design, with a recorded fallback (an ulp band plus a pinned count)
used only if a platform or a compiler difference is measured, so that
it is recorded rather than silently loosened.

Branch and PR policy: requirements D14 (one fork PR `feat-HROSM ->
develop`, expected #20, merge on Johan's go, fan-out after the merge).
Pushes trigger the fork's on-push CI (`.github/workflows/tests.yml`,
`timeout-minutes: 20` at `:36`, `PYTEST_ARGS: --cov-branch
--cov-report=xml --reruns 2 -n 4 --cov=kikuchipy` at `:39`, oldest job
pins at `:48`); the gates below are LOCAL gates recorded in the
ledger with numbers, recipe and machine.

**Oracle principle.** Five independent oracles, each where it is
strongest:
1. **EMsoft's own output**: the shipped references written by the
   EMsoftOO binaries on this machine (D13.3) and Johan's historical
   files behind the local gate (D13.5). Bitwise for the orientation
   stage (CPU arithmetic), tolerance-based for the dictionary stage.
2. **Test-local literal transcriptions** of the Fortran loops
   (`getKAMMap`, `getOrientationSimilarityMap`, `grow_region_`,
   `grain_dilate_`, `EMforDS_`), written from the EMsoftOO source by
   the test writer, never copied from the implementation; they are
   the oracle for every input shape the files do not cover.
3. **Analytic fields** whose answer is exact by construction
   (gradient fields, constant-pair fields, uniform top lists).
4. **orix** as an independent library: `Orientation.angle_with`,
   private `cu2ho`, `Rotation.from_homochoric`, the Hamilton product,
   `Symmetry.proper_subgroup`, one `random_vonmises` cross-check.
5. **Seeded synthetic generators** (`numpy.random.default_rng`) whose
   expected statistics are derived: averaging recovery, sub-grain
   physics sanity.

**Clean-replay rule for tests (A4, D14.5).** Test names, comments,
docstrings, parametrise ids and MTP constant names state the fact
("EMsoft credits the vertical neighbour one column to the right"),
never a spec ID (D-, V-, M-, K-, R-numbers) or a spec file name. New
code and tests also avoid identifiers matching `[DVKMR][0-9]+` after a
non-word character (`R0`, `D4`, `M1`), which the gate's grep flags;
use `ball_identity`, `divisor` and so on.

## Automated (default suite; run from Git Bash)

```
uv run --no-sync pytest tests/test_indexing tests/test_signals -k hrosm -n 0
uv run --no-sync pytest tests/test_indexing tests/test_signals -k hrosm -n 4
```

`-k` matching is case-insensitive, so classes named `TestHROSM...` in
the two existing modules are selected. The `-n 0` run comes first in
every session (numba `cache=True` files of the existing kernels that
`get_patterns` and the NCC metric use are written by one process; the
root `conftest.py` autouse fixture `_keep_numba_cache_dir_per_worker`
(`:131-150`) exists because `import pyebsdindex` redirects
`NUMBA_CACHE_DIR`). A red test under `-n 4` is re-run alone before it
counts (numba-cache flake rule, D15.5).

Gated runs (each also with `-n 0` first):

```
KIKUCHIPY_EMSOFT_DATA=C:/Users/westraadt.1/Software/EMSOFT/EMsoftData \
  uv run --no-sync pytest tests/test_indexing -k hrosm -n 0
KIKUCHIPY_EMSOFT_BIN=<the chosen Bin directory, D13.4> \
  uv run --no-sync pytest tests/test_indexing -k hrosm -n 0
uv run --no-sync pytest tests/test_indexing tests/test_signals -k hrosm -n 4 --weekly
```

**Wheel rule.** The wheel job runs the suite from the installed
package (`pyproject.toml:146-151` force-includes `tests/` and
`conftest.py`), so no test uses a repository-relative path:
references through `Path(kp.data.__file__).parent / "emsoft_hrosm"`
or `Dataset(...).fetch_file_path()`, sources for the static checks
through the module's `__file__`. `_hrosm/` docstring examples run in
the doctest job (<= 5 s).

**Test modules** (`tests/` has no `__init__.py` and runs with
`--import-mode=importlib`: nothing is imported from one test module
into another; shared generators live in the root `conftest.py`; every
EMsoft-file arm lives in `test_hrosm_emsoft_regression.py`, which
holds the one reference loader and the degenerate-pixel helper):

| module | classes | owns | stage |
|---|---|---|---|
| `tests/test_indexing/test_hrosm_kam.py` | `TestEMsoftQuaternions`, `TestCompatKAM` (V1), `TestCorrectKAM` (V3) | `_emsoft_quaternions.py`, `_kam.py` | A |
| `tests/test_indexing/test_hrosm_segmentation.py` | `TestSegmentationRule` (V4), `TestSyntheticGrains` (V5), `TestDilateAndBoxes` (V6) | `_segmentation.py` | A |
| `tests/test_indexing/test_hrosm_averaging.py` | `TestCenterPixel` (V6), `TestRecovery`, `TestGROD`, `TestGrainTable` (V8), `TestCompatEM` (V9) | `_averaging.py`, `_directional_statistics.py`, `_grains.py` | A |
| `tests/test_indexing/test_hrosm_sampling.py` | `TestMisorientationBall`, `TestBallSpacing`, `TestEMsampleRFZ` (V7) | `_sampling.py` | A |
| `tests/test_indexing/test_hrosm_osm.py` | `TestCompatOSM` (V10), `TestGrainAwareOSM` (V11) | `_osm.py` (synthetic lists only) | A |
| `tests/test_indexing/test_hrosm_emsoft_regression.py` | `TestModuleDiscipline` (V0), `TestCompatKAMOnEMsoftFiles` (V2), `TestCompatOSMOnEMsoftFiles` (V10), `TestClusterStage` (V12), `TestEndToEnd` (V13), `TestReferenceFiles`, `TestEMsoftFileReader`, `TestEMsoftProgramLock`, `TestRegenerateReferences` (V14) | shipped and local EMsoft oracles, `_emsoft_file.py`, `create_hrosm_reference.py`, the lock | A (V13 B) |
| `tests/test_signals/test_ebsd_hrosm.py` | `TestSubgrainContrast` (V15), `TestValidation`, `TestOutput`, `TestContracts`, `TestInvariance`, `TestMessages` (V16) | `_driver.py`, `EBSD.hrosm` | B |
| `tests/test_indexing/test_orientation_similarity_map.py` (append) | `TestHROSMKeywords` (V11) | the two new keywords; existing tests untouched | B |
| `tests/test_indexing/test_dictionary_indexing.py` (append) | `TestDictionaryIndexing::test_verbose_false_silences_the_core_for_hrosm` (V16) | `_dictionary_indexing(verbose=)` | B |
| root `conftest.py` (append) | generator functions + same-named fixtures; `emsoft_data_file(relpath, md5)`, `emsoft_bin_dir`, `emsoft_program`, `_emsoft_program_lock(path=None)` | synthetic inputs and the two gates | A (Stage B adds `hrosm_synthetic_signal`) |

**Gates** (D13.1):

| gate | condition | skip message names | used by |
|---|---|---|---|
| CI | always (default suite) | -- | everything not listed below |
| [download] | `kp.data.nickel_ebsd_large(allow_download=True)`, pooch-cached (precedent `test_spherical_emsphinx_regression.py`, `tech-stack.md:50`) | -- | V13 arms, V14 PC route pin |
| weekly | `@pytest.mark.weekly` (`weekly.yml:88`, `pytest --weekly --reruns 2 -n 4`; that workflow is disabled on the fork, so these arms run only in the local `--weekly` stage gate, D15.1 as amended in spec review round 2) | -- | full grids, V13 one-grain arm, V15 full size |
| local + weekly | `@pytest.mark.weekly` AND `KIKUCHIPY_EMSOFT_DATA` set (the variable marks this machine; any runner without it skips) | the variable | Al arms; V13 full map (reads no EMsoft data file) |
| local | `KIKUCHIPY_EMSOFT_DATA` set; fixture `emsoft_data_file(relpath, md5)` asserts the md5 (cached per session) | the variable, or the missing file | V2, V10, V12 local arms |
| bin | `KIKUCHIPY_EMSOFT_BIN` set and holding the 5 exes and 2 DLLs of D13.1, `EMdatapathname` read from `~/.config/EMsoft/EMsoftConfig.json`, an OpenCL GPU; every run inside `_emsoft_program_lock` (D13.1 as amended 2026-10-06: modelled on `conftest.py:702-750`, file `kikuchipy-emsoft-program.lock` in `tempfile.gettempdir()`, `_EMSOFT_LOCK_TIMEOUT = 3600.0`, a 30 s `os.utime` heartbeat while held, `_EMSOFT_LOCK_STALE = 300.0`); run directories only under `<EMdatapathname>/kikuchipy_hrosm/` | the variable, the missing program, or no GPU | V7 N 20 arm, V14 regeneration |

**Root-conftest generators** (plain functions, each exposed by a
same-named fixture returning the callable, the `dummy_signal`
precedent; all phases `Phase("ni", point_group="m-3m")`, step 1 um,
`CrystalMap` built with `create_coordinate_arrays`; rotations composed
with orix `Rotation.__mul__`, angles in radians via `np.deg2rad`):

- `hrosm_gradient_xmap(shape=(6, 7), delta_x=0.4, delta_y=0.1,
  euler0=(10.0, 20.0, 30.0), scramble_seed=None, absent=(),
  phase_id=None, rotations_per_point=1)`: rotation at `(y, x)` =
  `Rz(x * delta_x) * Rx(y * delta_y) * g0`, `g0 =
  Rotation.from_euler(np.deg2rad(euler0))`, degrees for the deltas.
  Each 4-neighbour pair angle is exactly `delta_x` (horizontal) or
  `delta_y` (vertical) for deltas below 45 deg. `scramble_seed`
  left-multiplies every pixel by `Oh.proper_subgroup[rng.integers(24)]`
  (`default_rng(scramble_seed)`); `absent` (flat indices) are written
  with `is_in_data=False`; `phase_id` (array) adds a second phase
  `"ni2"` (m-3m) for the given pixels; `rotations_per_point > 1`
  appends random extra rotations after the first.
- `hrosm_constant_pair_xmap(shape=(4, 5), phi=0.3, euler0=(10.0, 20.0,
  30.0))`: rotation `Rz((x + y) * phi) * g0`, every 4-neighbour pair
  angle exactly `phi` deg.
- `hrosm_grain_xmap(shape=(12, 16), n_grains=2, gradient=0.2,
  boundary_angle=30.0)`: grain A = columns `< W // 2` with `g_A =
  Rz(x * gradient) * g0`; grain B = the other columns with
  `R[111](boundary_angle) * g_A`; `n_grains=3` makes rows `>= H // 2`
  of the right half grain C, `R[100](45) * g_A`. Returns `(xmap,
  truth)`, `truth` int32 (1..n). The generator asserts that every pair
  of distinct grains is disoriented by more than 20 deg unless
  `boundary_angle` is passed below 20 (the leakage arm).
- `hrosm_top_lists(shape, n=10, pool=15, seed=40)`: per pixel
  `default_rng(seed).permutation(pool)[:n] + 1`, int32 `(H * W, n)`,
  1-based, no duplicates within a row.
- `write_emsoft_layout_file(path, kind, arrays)` (added 2026-10-06,
  spec review; `kind` in {"dot_product", "hrosm"}; `arrays` a dict of
  NumPy arrays and namelist values): writes a minimal EMsoft-layout
  HDF5 file with h5py: Fortran `(x, y)` arrays stored so h5py sees `(H,
  W)`, `(3, N)` Euler arrays, `TopMatchIndices` padded to a multiple of
  `numexptsingle`, namelist strings as object arrays of bytes,
  `NMLfiles/HROSMNML` text for `kind="hrosm"`; returns `path`. Used by
  `TestEMsoftFileReader`.
- `hrosm_synthetic_signal(xmap, sig_shape=(32, 32), pc=(0.42, 0.22,
  0.50), noise=0.0, seed=50)`: `mp =
  kp.data.nickel_ebsd_master_pattern_small(projection="lambert",
  hemisphere="both")` (shipped, 401 px, 20 kV; loaded once per
  session), `det = kp.detectors.EBSDDetector(sig_shape, pc=pc,
  sample_tilt=70)`, `s = mp.get_patterns(xmap.rotations.reshape(
  *xmap.shape), det, energy=20, compute=True)`; `noise > 0` adds
  `default_rng(seed).normal(0, noise * (max - min))` in float32.
  Returns `(s, det, mp)`. Added in Stage B (commit 4), its only users
  being V15 and V16.

**MTP placeholder inventory** (drafted; the failing-tests gate of
each stage confirms the names). Stage A,
`test_hrosm_emsoft_regression.py`: `SHIPPED_KAM_NONDEGENERATE_DIFF`,
`SHIPPED_REFINED_KAM_NONDEGENERATE_DIFF`, `SHIPPED_OSM_DIFF`,
`SHIPPED_GRAIN_ID_FROM_EULER_DIFF`, `CENTER_AVOR_MAX_ULP`,
`WAT_AVOR_MAX_DEG`, `WAT_KAPPA_REL`, `REFERENCE_TOTAL_BYTES` (seed
~841,000);
local: `NI6_KAM_NONDEGENERATE_DIFF`, `NI6_KAM_MAX_ULP`,
`NI6_DI_KAM_NONDEGENERATE_DIFF`, `GRX810_KAM_NONDEGENERATE_DIFF`,
`AL_KAM_NONDEGENERATE_DIFF`, `AL_KAM_MAX_ULP`, `NI6_OSM_EDGE_PIXELS`,
`NI6_WAT_AVOR_MAX_DEG`, `NI6_WAT_KAPPA_REL`; bin: `GPU_ARRAY_POLICY`,
`REGENERATION_RUNTIME_S` (recorded). `test_hrosm_averaging.py`:
`RECOVERY_ANGLE_FACTOR`, `KAPPA_RATIO_BAND_N300`,
`KAPPA_RATIO_BAND_N30`, `WRONG_SIDE_MAX_KAPPA_RATIO`,
`WATSON_UNDERFLOW_SEED` (seed 80; V9, added in spec review round 2).
`test_hrosm_sampling.py`: `BALL_SPACING_DEFAULT_DEG` (seed 0.15918),
`BALL_SPACING_N10_DEG` (seed 0.31765), `BALL_SPACING_N2_DEG` (seed
1.60130). Stage B, `test_hrosm_emsoft_regression.py` (split
2026-10-06, spec review: the two arms use different masters):
`E2E_ONE_GRAIN_DISORIENTATION_MEDIAN_DEG`,
`E2E_ONE_GRAIN_DISORIENTATION_P99_DEG` (weekly) and
`E2E_FULL_MAP_OSM_PEARSON_MIN`,
`E2E_FULL_MAP_DISORIENTATION_MEDIAN_DEG`,
`E2E_FULL_MAP_DISORIENTATION_P99_DEG`, `E2E_FULL_MAP_CI_PEARSON_MIN`
(local + weekly); `test_ebsd_hrosm.py`: `SUBGRAIN_CONTRAST_MIN`,
`SUBGRAIN_CONTRAST_RATIO`, `SUBGRAIN_MEDIAN_ERROR_DEG`,
`SUBGRAIN_STEP_TOL_DEG`, `CHUNK_INVARIANCE` (the "bitwise" or
fallback decision of D8.12). Stage C adds none.

Budget (D15.1 as amended 2026-10-06): CI-style wall time of the HROSM
selection <= 25 s (binding) and default-suite additions <= 60 s
serial (ceiling), per-file seeds in "CI budget" below; weekly
additions: local serial seconds recorded per stage with a scaled CI
estimate (target <= 5 min), never a gate, since the fork's Weekly
workflow is disabled (D15.1 as amended in spec review round 2).

### V0 -- Module discipline (`test_hrosm_emsoft_regression.py`, `TestModuleDiscipline`) -- Stage A

Pins D10.3 (no numba kernels), D11 (licence, nothing LGPL), D1.7
(style). Static `ast` and text checks over
`src/kikuchipy/indexing/_hrosm/*.py`; runtime < 1 s; CI.

- `test_hrosm_package_imports_no_numba`: no `ast.Import` /
  `ast.ImportFrom` of a module starting with `numba`. If a stage gate
  measures a hot loop over budget and adds a kernel (D10.3), this test
  is replaced, in the same commit, by the NLPAR V0 arms
  (`KERNEL_NAMES` equals the module's njit names, `cache=True,
  nogil=True`, no `parallel`/`fastmath`, `.py_func` parity bitwise),
  helpers copied from `tests/test_indexing/test_spherical_euler.py`.
- `test_emsoft_derived_modules_carry_the_bsd_notice` (parametrised
  over the seven Stage A modules `_emsoft_quaternions`, `_kam`,
  `_segmentation`, `_averaging`, `_directional_statistics`,
  `_sampling`, `_osm`; the Stage B failing-tests commit adds `_driver`
  to the parametrisation, amended 2026-10-06, spec review): the
  module's consecutive `#` comment lines are joined first (strip the
  leading `#` and one space, collapse runs of whitespace to one space),
  because the 72-character comment rule wraps the long sentences (the
  `signals/util/_master_pattern.py:20-28` precedent); the joined text
  contains "The following copyright notice is included because the
  following functionality in this file is derived and adapted from
  EMsoftOO:", "Marc De Graef Research Group/Carnegie Mellon
  University", "All rights reserved.", "Changes by the kikuchipy
  developers"; the year span is "2014-2026" in
  `_directional_statistics` and "2013-2026" elsewhere; every module
  (incl. `_grains`, `_emsoft_file`) starts with the GPL-3.0-or-later
  header.
- `test_public_docstrings_link_no_private_name_and_no_spec_id` (added
  2026-10-06, spec review; the existing
  `test_spherical_indexer.py:1441-1467` check scans only the spherical
  modules): over the docstrings of the eight public names (Stage B
  adds `EBSD.hrosm` and `orientation_similarity_map`), no Sphinx role
  to a private name (the role regex of
  `test_spherical_indexer.py:1446`) and no match of the clean-replay
  pattern of `plan.md` section 5 (spec paths, spec file names, and
  `[^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)`).
- `test_no_lgpl_routine_is_ported`: no token `r8_normal_01`,
  `r8_uniform_01`, `r8vec_normal_01`, `BesselI0`, `BesselI1`,
  `BesselIn`, `SSORT`, `DSYEV` outside comments naming what replaced
  them; `numpy.random.default_rng` is the only RNG constructor.
- `test_print_only_in_the_driver`: `print` calls occur only in
  `_driver.py`.
- `test_modules_use_postponed_annotations_and_no_optional_union`: each
  module has `from __future__ import annotations`; no `Optional[` or
  `Union[`.
- `test_emsoft_compatible_is_never_a_module_global`: no module-level
  assignment whose target name contains `emsoft_compatible`
  (case-insensitive).

### V1 -- Compat KAM against the loop transcription (`test_hrosm_kam.py`, `TestEMsoftQuaternions`, `TestCompatKAM`) -- Stage A

Pins D3.2-D3.4, K1, K2, D10.2. Oracle: the test-local literal
transcription `emsoft_kam_loop(euler32, H, W, operators)` of the
pseudo-code of requirements D3.2 (pure Python loop over `t = 1..n`,
`lstore`/`pstore` as written, `ii`, `jj` with the off-by-one, the
spurious identity term, `dis` = the `getDisorientation_` double loop
over EMsoft's operator list in `quatmult` term order, unclipped
`acos`, NaN never selected, `ac = 1000.0`), then the multiplier
sequence of D3.4 in float64 with `third(x) = (x * 4) / 3`, `sngl`,
and `float32(float64(kam) * (180 / np.pi))`. Fixtures: random float32
Euler maps from `default_rng(10)` (`phi1, phi2` uniform [0, 2 pi),
`Phi` uniform [0, pi], cast to float32), the near-identity map (`g0`
perturbed by at most 0.01 deg per pixel, `default_rng(13)`), the
constant-pair and gradient generators. Input maps are built as
`CrystalMap(Rotation.from_euler(eu32.astype(np.float64)), ...)` with
`create_coordinate_arrays(shape)`; the implementation's compat route
(`to_euler()` -> float32 -> `eq_`, D3.2) recovers `eu32` exactly, and
1 x n, n x 1, 1 x 1 maps return `(1, n)`, `(n, 1)`, `(1, 1)` (D1.9).
CI; < 3 s.

`TestEMsoftQuaternions` (the ingredients):
- `test_operator_table_for_m3m_is_emsoft_order`: `pgnum 32` gives 24
  quaternions, identity first, in the order of `SYM_Qsymop` columns
  1, 5-10, 17-24, 2-4, 11-16 (literal table transcribed in the test
  from `mod_quaternions.f90:53-64ff` with `sq22 =
  0.7071067811865475244`, `sq32 = 0.8660254037844386467`, `half =
  0.5`, exact zeros); as a set equal up to sign to orix
  `Oh.proper_subgroup` within 1e-15.
- `test_eq_matches_orix_from_euler`: 1000 random float32 Euler
  triplets (`default_rng(14)`) promoted to float64: EMsoft `eq_` (q0
  >= 0) equals `Rotation.from_euler(eu64).data` within 1e-15 per
  component; `(10, 20, 30)` deg gives `(0.92542, -0.17101, 0.03015,
  -0.33682)` to 5 decimals.
- `test_euler_round_trip_reproduces_the_shipped_float32_angles` (the
  compat input route, D3.2 and `plan.md` 7.2 item 16 as decided
  2026-10-06, spec review; replaces the drafted
  `from_euler == eq_` pin, refuted at 1 float64 ulp in ~30 % of
  rows): on the shipped `EulerAngles` and `RefinedEulerAngles`, both
  in `regression_hrosm_large_refined.npz` (loaded from
  `Path(kp.data.__file__).parent / "emsoft_hrosm"`),
  `Rotation.from_euler(eu32.astype(np.float64)).to_euler().astype(
  np.float32)` equals `eu32` bitwise on every row, and the EMsoft `eq_`
  port of the round-tripped angles equals `eq_(eu32)` bitwise (seed:
  exact on Ni6 28,086 x 2, GRX810 139,181, Al 501,592 rows).
- `test_point_group_number_rejects_unmapped_or_mismatched_groups`
  (added 2026-10-06, spec review; K11, D1.5 item 6): `Oh` maps to 32
  and `O` (432) to 30; a `Symmetry` built from Oh's operators but
  given a name outside the EMsoft map raises `ValueError` "point
  group"; with the EMsoft operator table
  monkeypatched to drop one operator of `pgnum 32`, `Oh` raises
  "point group". The Stage A measurer records whether any orix proper
  group's operator set differs from EMsoft's (then a real group is
  added as a case).
- `test_quatmult_term_order_matches_orix_product`: the EMsoft term
  order of D3.2 equals orix `Rotation.__mul__` within 1 float64 ulp
  per component on 1000 random pairs, and `i * j == k`.

`TestCompatKAM`:
- `test_matches_the_loop_transcription_bitwise[shape]`, shapes `(1,
  1)`, `(1, 5)`, `(5, 1)`, `(2, 2)`, `(3, 3)`, `(4, 7)`, `(7, 9)` plus
  the near-identity `(6, 8)` arm: `kernel_average_misorientation_map(
  xmap, emsoft_compatible=True)` `np.array_equal` the transcription
  (float32 deg, every pixel incl. degenerate ones; both sides are
  NumPy). Fallback (recorded, never silent): if an arm differs by 1
  float32 ulp only on pixels where the two sides call `np.arccos`
  through different array shapes (NumPy's SIMD paths), the pin becomes
  `<= 1` ulp with the count recorded.
- `test_matches_the_loop_transcription_on_duplicated_orientations`
  (added 2026-10-06, spec review; K2): a `(4, 6)` map whose float32
  Euler triplets repeat exactly in neighbouring pixels (every second
  column a copy of its left neighbour, two rows copies of the row
  above), bitwise equal to the transcription on every pixel; the
  fixture includes at least one duplicated pair whose unclipped
  transcription angle is NOT 0 (all near-identity products round
  above 1, so NaN is never selected and a non-identity angle wins;
  found by scanning float32 triplets from `default_rng(15)`, and the
  test asserts its presence), so a clipped `arccos` changes the
  result.
- `test_constant_pair_angle_field[shape]`, shapes `(4, 5)`, `(6, 3)`,
  `phi = 0.3`: every pixel equals `phi` except `(0, 0)` = `phi + s`
  and `(0, W - 1)` = `phi + s / 2`, `s` the correct-mode angle of
  pixel `(0, W - 1)` to the identity (deg); `rtol=3e-7`.
- `test_vertical_pair_is_credited_one_column_right`: constant-pair
  field `(5, 5)`, pixel `(2, 1)` additionally rotated by 3 deg about
  z; the set of pixels whose compat KAM changes (vs the unperturbed
  map, bitwise comparison) is exactly `{(2, 0), (2, 1), (2, 2), (1,
  2), (3, 1)}`; the correct-mode set is `{(2, 0), (2, 1), (2, 2), (1,
  1), (3, 1)}`.
- `test_first_row_last_column_is_compared_with_the_identity`: two
  `(4, 5)` constant-pair maps (`phi = 0.3`) with `euler0` `(10, 20,
  30)` and `(40, 50, 60)` deg have the same pair angles but different
  `s`: pixel `(0, 0)` differs between them by `s2 - s1` and `(0, 4)`
  by `(s2 - s1) / 2`; every other pixel equals `phi` in both (`rtol=
  3e-7`). (A transformed map is never compared bitwise: rotating the
  inputs changes pair angles at the ulp level.)
- `test_degrees_output_rounds_through_float32_radians`: with the same
  input, `kam_deg == np.float32(np.float64(kam_rad) * (180 / np.pi))`
  bitwise, `kam_rad` from `degrees=False`; both float32 `(H, W)`
  (drafting arithmetic: 25.4 % of uniform [0, 0.2] rad sums differ
  between this rounding and `np.float32(acc * rtod)`).
- `test_rejects_absent_points_and_several_phases`: compat raises
  `ValueError` with "emsoft_compatible requires every map point" (one
  absent point) and "one phase" (two phases).

### V2 -- Compat KAM on EMsoft files (`test_hrosm_emsoft_regression.py`, `TestCompatKAMOnEMsoftFiles`) -- Stage A

Pins D3.6 against the binary. A pixel is DEGENERATE if any pair
credited to it (left, right, up, shifted down, spurious; D3.3) has a
correct-mode angle below 1e-6 rad (test-local helper
`degenerate_pixels(euler, H, W)`). On non-degenerate pixels the count
of float32 values differing from the file is pinned (seed 0 unless
stated); on degenerate pixels the differing count and the maximum
difference are recorded with `record_property`, never asserted.

| test | input -> compared with | gate | pin (seed) |
|---|---|---|---|
| `test_shipped_di_kam_on_nondegenerate_pixels` | `regression_hrosm_large_refined` `EulerAngles` -> `regression_hrosm_large_di` `KAM` (55, 75) (file placement per D13.3 as amended 2026-10-06) | CI | `SHIPPED_KAM_NONDEGENERATE_DIFF` (0; fallback <= 2 ulp, count pinned) |
| `test_shipped_refined_kam_on_nondegenerate_pixels` | `regression_hrosm_large_refined` `RefinedEulerAngles` -> `regression_hrosm_large_center` `kam` | CI | `SHIPPED_REFINED_KAM_NONDEGENERATE_DIFF` (0; fallback <= 2 ulp, count pinned) |
| `test_ni6_hrosm_kam_on_nondegenerate_pixels` | `DItutorial/Ni/dp-Ni6-refined.h5` `RefinedEulerAngles` -> `dp-Ni6-refined_HROSM.h5` `kam` (151 x 186 (H x W) map, 28,086 px) | local | `NI6_KAM_NONDEGENERATE_DIFF` (6 of 27,992) and `NI6_KAM_MAX_ULP` (2); degenerate seed 37 of 94, max 22.5 deg |
| `test_ni6_di_kam_on_nondegenerate_pixels` | `dp-Ni6-refined.h5` `EulerAngles` -> `KAM` | local | `NI6_DI_KAM_NONDEGENERATE_DIFF` (0 of 6,774); degenerate seed 7,079 of 21,312 |
| `test_grx810_di_kam_on_nondegenerate_pixels` | `OSM/GRX810_HROSM/dp-GRX-refined.h5` `EulerAngles` -> `KAM` (337 x 413) | local | `GRX810_KAM_NONDEGENERATE_DIFF` (0 of 4,182); degenerate seed 358 of 134,999 |
| `test_al_di_kam_on_nondegenerate_pixels` | `Al_HROSM/dp-full.h5` `EulerAngles` -> `KAM` (689 x 728 (H x W)) | local + weekly | `AL_KAM_NONDEGENERATE_DIFF` (16 of 210,294) and `AL_KAM_MAX_ULP` (2); degenerate seed 9,838 of 291,298, max 90.0 deg (spec-review probe 2026-10-06, float64 literal transcription, 96.8 s unoptimised) |

Map shape from `ipf_wd`, `ipf_ht` (or `ROI`) through the private
reader (D12.1); local files are opened only through
`emsoft_data_file` with the md5s of D13.5. The `GRX810` HROSM file
is NOT an oracle (D13.5) and no test opens it. Runtime CI < 1 s;
Al ~5 s (estimate, recorded).

### V3 -- Correct KAM (`test_hrosm_kam.py`, `TestCorrectKAM`) -- Stage A

Pins D3.1, D3.7. Tolerance: `np.abs(kam - expected) <=
np.spacing(expected)` (1 float32 ulp) with `expected =
np.float32(np.float32(mean_rad) * (180 / np.pi))` from the float64
analytic or orix mean. CI; < 1 s.

- `test_two_axis_gradient_field_is_exact[shape]`: gradient generator
  `delta_x = 0.4`, `delta_y = 0.1`, shapes `(6, 7)`, `(1, 5)`, `(5,
  1)`, `(2, 2)`, `(1, 1)`. Expected (deg): interior `(2 dx + 2 dy) /
  4`; top/bottom row interior `(2 dx + dy) / 3`; left/right column
  interior `(dx + 2 dy) / 3`; corners `(dx + dy) / 2`; `(1, n)` maps
  `dx` inside and `dx` at both ends; `(n, 1)` maps `dy`; `(1, 1)` NaN.
- `test_matches_orix_angle_with_on_scrambled_variants`: gradient map
  `(5, 6)`, `scramble_seed=11`, each pixel further left-multiplied by
  a random rotation of at most 2 deg (`default_rng(12)`); reference =
  per pixel, mean over existing 4-neighbours of `Orientation(...,
  Oh).angle_with(other)` (orix, float64 rad).
- `test_absent_neighbours_are_excluded_and_isolated_pixels_are_nan`:
  `(4, 5)` gradient map, `absent=(1, 5, 13)` (flat `y * 5 + x`;
  corrected 2026-10-06, spec review round 2: the drafted `(1, 5, 7,
  11)` made all four neighbours of `(1, 1)` absent): absent pixels
  NaN; pixel `(0, 0)` (both neighbours, flat 1 and 5, absent) NaN;
  pixel `(1, 1)` (neighbours flat 1, 5 absent, 7, 11 present) is the
  mean over its 2 present neighbours, `(delta_x + delta_y) / 2` within
  the V3 tolerance.
- `test_pairs_across_phases_are_skipped`: `(4, 6)` gradient map,
  columns 3-5 phase `"ni2"`: columns 2 and 3 use 3 (interior) or 2
  (edge rows) neighbours.
- `test_first_rotation_is_used_with_several_rotations_per_point`:
  `rotations_per_point=3` gives the same map as 1.
- `test_degrees_false_returns_float32_radians`.
- `test_identical_neighbours_give_zero_not_nan` (added 2026-10-06,
  spec review; rewritten in round 2 for the D3.1/D10.4 near-one snap):
  two `(3, 4)` maps, each of one repeated rotation: `below`, the first
  rotation of `Rotation.from_euler(eu32.astype(np.float64))`, `eu32`
  float32 Euler triples from `default_rng(13)` (uniform over `[0, 2
  pi) x [0, pi) x [0, 2 pi)`), whose float64 self-dot `max_j |<S_j o,
  o>|` (Oh proper operators) is `< 1`, and `above`, the first whose
  self-dot is `> 1` (the test asserts both sides; about 5 % and 33 %
  of rotations, round-2 probe); plus the duplicated-orientation
  fixture of V1. Correct KAM is finite everywhere and exactly 0.0
  where all neighbours are identical, on both maps (without the snap:
  NaN on `above`, S16; with the old clip: ~3e-6 deg on `below`, S18).

### V4 -- Segmentation rule (`test_hrosm_segmentation.py`, `TestSegmentationRule`) -- Stage A

Pins D4.1-D4.4, R6. Oracle: test-local literal transcription
`emsoft_grow_regions(kam32, gangle)` of `grow_region_driver_` /
`grow_region_` (`mod_cluster.f90:445-562`): raster seeds (y outer, x
inner), seed rejected with `nGrains -= 1` if `kam > gangle`, LIFO
stack, label on push, the 8 neighbour offsets in the order of
`:512-513`, test `abs(kam_n - kam_popped) <= gangle` in float32,
`count == 1 -> -1`, `nGrains -= 1` at the end, `-1 -> 0`. Fixtures:
background value 100.0 (a component of background pixels never holds
a pixel `<= 5`). Threshold 5.0 unless stated. CI; < 2 s.

- `test_matches_the_flood_fill_transcription[shape-seed]`: random
  float32 KAM uniform [0, 12] (`default_rng(20 + i)`), shapes `(1,
  1)`, `(1, 6)`, `(6, 1)`, `(5, 5)`, `(9, 11)`, `(17, 23)`; labels
  `np.array_equal`, `dtype == np.int32`, the count of grains equal.
- `test_diagonal_neighbours_join`: two pixels kam 1.0 touching only
  diagonally -> one grain of 2 pixels.
- `test_criterion_is_chained_on_kam_differences`: a row `[1, 5, 9,
  13, 17]` -> one grain of 5 pixels.
- `test_component_without_a_pixel_at_or_below_threshold_is_unassigned`:
  `[6, 7, 8]` -> all 0; `[6, 6, 2]` -> one grain of 3.
- `test_seed_at_exactly_the_threshold_is_accepted`: `[5.0, 5.0]` ->
  one grain.
- `test_singletons_are_unassigned`: an isolated pixel kam 1.0 -> 0.
- `test_threshold_tie_is_decided_in_float32`: adjacent
  `np.float32(0.01)` and `np.float32(5.01)`: float32 `|b - a|` is
  exactly 5.0 (joins: one grain); the float64 difference of the same
  values is 5.000000229 (drafting arithmetic).
- `test_labels_follow_the_first_qualifying_pixel_in_raster_order`: a
  component whose first raster pixel is above the threshold but whose
  first qualifying pixel comes after another component's gets label
  2.
- `test_nan_kam_is_never_assigned`: NaN pixels 0, and they separate
  components.
- `test_phases_are_segmented_separately`: equal KAM across a phase
  border gives two grains; `phase_id == -1` pixels are 0.
- `test_rule_is_the_same_in_both_modes`: `emsoft_compatible` True and
  False give equal labels with `dilate=False`.

### V5 -- Correct KAM plus segmentation on synthetic grains (`test_hrosm_segmentation.py`, `TestSyntheticGrains`) -- Stage A

Pins D4.1 + D3.1 end to end, D4.7. CI; < 1 s.

- `test_two_and_three_grain_maps_are_recovered_up_to_permutation[n]`:
  `hrosm_grain_xmap(n_grains=n)`, n in {2, 3}; correct KAM then
  `segment_grains_kam(threshold=5.0)`: interior pixels (no
  4-neighbour in another true grain) of each true grain carry exactly
  one label, distinct across grains (a bijection); boundary pixels are
  0 or the label of an adjacent true grain.
- `test_boundary_below_four_thresholds_leaks`: `boundary_angle=15.0`
  -> grains A and B share one label (the documented leak, D4.7).
- `test_compat_kam_gives_the_same_interior_grains`: compat KAM on the
  same map recovers the same interior bijection.

### V6 -- Dilate, bounding boxes, centre pixel (`TestDilateAndBoxes` in `test_hrosm_segmentation.py`; `TestCenterPixel` in `test_hrosm_averaging.py`) -- Stage A

Pins D4.5, D4.6, D5.3, K3, K4. Oracle for compat dilate: the test-local
transcription of `grain_dilate_` (`mod_cluster.f90:403-442`: zero-padded
copy, loop over 1-based `i = 1..W-1`, `j = 1..H-1`, 3 x 3 max written
to `(i + 1, j + 1)` when non-zero, simultaneous). CI; < 1 s.

- `test_compat_dilate_matches_the_emsoft_window_transcription[seed]`:
  random label maps (labels 0..4, `default_rng(30 + i)`, shapes `(7,
  9)`, `(1, 5)`, `(5, 1)`, `(2, 2)`) equal the transcription and the
  `maximum_filter` form of D4.5 bitwise.
- `test_compat_dilate_skips_the_first_row_and_column_and_overwrites`:
  a grain label 1 touching row 0 and column 0 leaves row 0 and column
  0 unchanged; label 3 next to label 2 takes one pixel of 2.
- `test_compat_dilate_can_remove_a_grain`: a 2-pixel label 1 inside
  label 2 vanishes; `grain_bounding_boxes` row for label 1 has height
  and width 0; `average_grain_orientations` gives that grain `kappa
  == -1.0`, `valid` False, identity rotation.
- `test_correct_dilate_fills_only_unassigned_pixels`: assigned labels
  unchanged; an unassigned pixel next to labels 2 and 3 becomes 3; a
  pixel two steps from any label stays 0 (simultaneous update).
- `test_correct_dilate_respects_phases`: an unassigned pixel of phase
  `"ni2"` next to only phase `"ni"` labels stays 0.
- `test_bounding_boxes_are_zero_based_row_col_height_width`: int64
  `(n, 4)`; an L-shaped grain at rows 2-5, columns 1-3 gives `(2, 1,
  4, 3)`.
- `test_emsoft_grain_roi_converts_to_bounding_boxes`: EMsoft
  `(x0, y0, w, h)` 1-based -> `(y0 - 1, x0 - 1, h, w)`.

`TestCenterPixel`:
- `test_compat_center_is_the_box_centre_rounded_up`: a box with 1-based
  `x0 = 1, w = 4, y0 = 1, h = 3` picks 1-based `(3, 2)` = 0-based
  `(1, 2)`; `avor` is EMsoft `eq_` of that pixel's float32 Euler
  angles (not reduced), `kappa == 1.0`.
- `test_compat_center_may_lie_outside_the_grain`: a ring-shaped grain:
  the compat centre pixel has another label; correct `center` lies in
  the grain.
- `test_correct_center_is_the_grain_pixel_nearest_the_centroid`: ties
  broken by the lowest flat index.
- `test_center_kappa_is_one_and_never_gated`: `min_kappa=1e9` keeps
  every grain with `method="center"` and `"mean"`.

### V7 -- Misorientation ball (`test_hrosm_sampling.py`, `TestMisorientationBall`, `TestEMsampleRFZ`) -- Stage A

Pins D6, K8. CI; < 3 s (N 20 grid built twice).

- `test_count_order_and_shells[n_steps]`, N in {1, 2, 6, 20}, 5 deg:
  length `(2N + 1)**3`; element at index `(i + N)(2N + 1)**2 + (j +
  N)(2N + 1) + (k + N)` is the cube point `(i, j, k) dx`; shell `m =
  max(|i|, |j|, |k|)` holds 1 point at `m = 0` and `24 m**2 + 2`
  otherwise (N 6: 1, 26, 98, 218, 386, 602, 866; N 20 outer 9,602).
- `test_outer_shell_is_exactly_max_angle`: 5 deg, N 20: maximum angle
  within 1e-9 deg of 5 (seed 5.000000000008); shell 1 at 0.24997 deg
  (1e-5); within-shell angle spread <= 1e-10 deg (seed 6e-12);
  `edge = 0.035163745447257796`, `dx = 0.0017581872723628899`
  (`rel=1e-15`, through the private grid helper); arms `max_angle` in
  {1, 10, 30} with the same exactness.
- `test_cube_to_ball_matches_orix_cu2ho`: the own port of
  `Lambert3DCubeForwardDouble` + `GetPyramidDouble` equals orix's
  private `orix.quaternion._conversions.cu2ho` within 1e-12 on the N
  20 grid and 1000 random cube points (`default_rng(60)`) including
  the origin, face centres, edges and pyramid boundaries `|x| = |y|`;
  `Rotation.from_homochoric` of the result is the ball.
- `test_composition_is_conj_ball_times_center`: centre = Rodrigues
  axis `(1, 2, 3) / sqrt(14)`, magnitude 0.1;
  `misorientation_ball(center)[i] == ~ball_identity[i] * center`
  within 1e-15; its Rodrigues vector equals `(-v + rho0 + rho0 x v) /
  (1 + v . rho0)` within 1e-12; `center * ~ball_identity[i]` and
  `ball_identity[i] * center` differ from it by more than 1e-3 rad for
  some `i`.
- `test_center_none_is_the_identity`.
- `test_compat_storage_is_float32_rodrigues`: with
  `emsoft_compatible=True` each point equals the float32 Rodrigues
  4-vector `(axis, tan(w / 2))` of the correct point converted back in
  float64 (`w = 2 atan(float64(tan32))`), `(0, 0, 1, 0)` for the zero
  vector (the identity), within 1e-15; compat and correct differ
  (not `array_equal`) by at most 2e-7 rad.
- `test_argument_validation`: `max_angle` 0, -1, 180 and `n_steps` 0,
  `True` raise `ValueError`.

`TestBallSpacing` (D20.1, D20.5; added 2026-10-06, spec review; CI,
< 1 s apart from the default pin, which builds the N 20 grid once):
- `test_spacing_equals_brute_force_nearest_neighbour[n_steps]`, N in
  {1, 2, 3}, 5 deg: equals the mean over points of the minimum
  off-diagonal entry of `ball.angle_with_outer(ball, degrees=True)`
  (orix `Rotation.angle_with_outer`, present in 0.12.1) within 1e-12
  deg (seed N 2: 1.60130, brute force equal to 5e-13).
- `test_spacing_is_invariant_under_the_grain_composition`: the same
  brute force on `misorientation_ball(center, max_angle=5,
  n_steps=2)` for a random centre (`default_rng(61)`) equals
  `misorientation_ball_spacing(5, 2)` within 1e-9 deg.
- `test_default_spacing_pin`: `misorientation_ball_spacing()` ==
  `BALL_SPACING_DEFAULT_DEG` (MTP, seed 0.15918 at 5 deg / N 20),
  `misorientation_ball_spacing(5, 10)` == `BALL_SPACING_N10_DEG` (seed
  0.31765), `(5, 2)` == `BALL_SPACING_N2_DEG` (seed 1.60130), each
  `rel=1e-4`; float return type.
- `test_spacing_decreases_with_n_steps_and_grows_with_max_angle`:
  strictly decreasing over N in {2, 3, 4, 6} at 5 deg; strictly
  increasing over `max_angle` in {1, 2, 5, 10} at N 4.
- `test_spacing_argument_validation`: the `misorientation_ball` cases
  and fragments.

`TestEMsampleRFZ` (the binary's own ball; `qu` and `eu` are the moved
ball, `ro` the unshifted grid, 9-decimal text, `mod_so3.f90:2513-2635`):
- `test_shipped_n6_ball_matches_in_order` (CI, `regression_hrosm_
  ball_n6.npz`, 2,197 points, `pgnum 32`, `maxmisor 5`, centre as
  above): ours with q0 >= 0 equals `qu` per component within 6e-10,
  IN ORDER; `ro` equals the unshifted grid's Rodrigues 4-vectors
  within 6e-10 (angle-0 point compared by its angle only).
- `test_binary_n20_matches_in_order` (bin): runs `EMsampleRFZ` with
  `nsteps 20` (and `nsteps 6`) in a run directory under
  `kikuchipy_hrosm/`; the same assertions on 68,921 points, plus `eu`
  in degrees (`mod_so3.f90:2588-2590`, `writeOrientationstoFile_`:
  `e%e_copyd() / dtor`, format F17.9; settled 2026-10-06, spec review)
  equal to ours through `Rotation.from_euler(np.deg2rad(eu))` within
  2e-8 rad.

### V8 -- Averaging recovery, GROD and `GrainTable` (`test_hrosm_averaging.py`, `TestRecovery`, `TestGROD`, `TestGrainTable`) -- Stage A

Pins D5.1-D5.4, D5.7-D5.9, D2.1, R5. **Sampler (test-local, exact).**
orix `Rotation.random_vonmises` is NOT the sampler (ledger entry 1):
its density is `exp(2 alpha cos w) / 0F1(1.5, alpha**2)` with `w` the
rotation angle (a Watson density on S^3 with `kappa_W = 4 alpha`;
VMF-equivalent `8 alpha` near the mode), it draws from the global
`np.random` state, and its rejection loop takes > 88 s at `alpha =
1000, N = 30`. `sample_s3(mu, kappa, n, kind, rng)`: half-angle
`theta` on a `2**16 + 1` grid over `[0, pi / 2]` with log-density
`f(cos theta) - f(1) + 2 log(sin theta)` (`f(t) = kappa t` for VMF,
`kappa t**2` for Watson), inverse-CDF by `np.interp` of
`rng.random(n)`; axis = normalised `rng.standard_normal((n, 3))`; `x
= mu * (cos theta, sin theta * axis)` (orix product). The VMF mass
beyond `pi / 2` is below `exp(-kappa)` and is dropped. Spread grid by
the orix-equivalent `alpha` in {20, 100, 1000} (rms rotation angle
~15, ~7, ~2.2 deg): VMF `kappa = 8 alpha`, Watson `kappa = 4 alpha`;
`mean` uses the Watson samples. Samples are wrapped as one grain in a
`(1, N)` `CrystalMap` (`grain_id` all 1). Scrambling: LEFT =
`Oh.proper_subgroup[j] * x`, RIGHT = `x * Oh.proper_subgroup[j]`, `j =
rng.integers(24)`. Error = symmetry-aware angle (orix `angle_with`) of
the recovered mean to `mu`. `mu` = Euler `(37, 51, 113)` deg.

| quantity | seed | pin |
|---|---|---|
| `RECOVERY_ANGLE_FACTOR` (band = factor x `2 / sqrt(8 alpha N)` rad) | 4 | MTP |
| `KAPPA_RATIO_BAND_N300` (`kappa_hat / kappa` in `[1/b, b]`) | 1.25 | MTP |
| `KAPPA_RATIO_BAND_N30` | 1.6 | MTP |
| `WRONG_SIDE_MAX_KAPPA_RATIO` (`kappa_hat` on wrong-side samples / `kappa_hat` on the same samples scrambled on the model's side) | 0.2 | MTP |

`TestRecovery` (seed `default_rng(70)`, `n_em=25`, `n_iter=40`,
`seed=0` for the method):
- `test_sampler_matches_orix_random_vonmises`: `np.random.seed(0)`;
  orix `alpha = 20`, N 300 vs ours Watson `kappa = 80`, N 3000: the
  mean of `cos w` agrees within 4 standard errors of the orix sample
  (0.12 s measured for the orix draw).
- `test_recovers_left_scrambled_variants[method-alpha-n]`: method in
  {mean, vmf, watson}; DEFAULT arms `alpha100-n300` (all methods) and
  `alpha1000-n30` (watson, mean); the full grid `alpha` {20, 100,
  1000} x N {30, 300} is weekly. Error <= band; `kappa_hat` within the
  ratio band for vmf/watson (`mean`'s kappa never asserted, D5.2).
- `test_right_scrambled_variants_are_not_recovered[method]`: `alpha
  100`, N 300, correct mode: `kappa_hat` (for `mean` the VMF
  closed-form value of D5.2) on RIGHT-scrambled samples <=
  `WRONG_SIDE_MAX_KAPPA_RATIO` x `kappa_hat` on the same samples
  scrambled LEFT (23 of 24 variants cannot be aligned, so the
  resultant collapses; a direction error is not asserted, since the
  unscrambled 1/24 can still pull the mean near `mu`).
- `test_watson_is_antipodally_symmetric`: `mu` = 179.8 deg about `(1,
  2, 3) / sqrt(14)`, Watson `kappa = 400`, N 300, no scrambling, so
  about half the samples have their sign flipped by `q0 >= 0`: Watson
  error <= band and `kappa_hat` within the band; the VMF error on the
  same samples is recorded (`record_property`), not asserted.
- `test_mean_draws_nothing`: `method="mean"` gives identical tables for
  `seed` 0, 1 and None.
- `test_one_generator_is_consumed_across_grains_in_label_order`: a
  2-grain map with `method="watson", seed=0`: grain 2 equals a
  single-grain run of grain 2 with `seed=rng`, `rng = default_rng(0)`
  first advanced by `standard_normal((n_em, 4))` (the free function's
  `seed` accepts a `Generator`, as `EBSD.hrosm`'s does).

`TestGROD` (D5.8 and D20.2-D20.3 as amended 2026-10-06, spec review):
- `test_max_grod_is_the_largest_angle_to_the_grain_reference`:
  `max_grod` (float32 deg) equals the maximum over the grain of a
  per-pixel orix `Orientation(pixel, Oh).angle_with(Orientation(
  reference, Oh), degrees=True)` loop within 1 float32 ulp, the
  reference being `table.rotation[g - 1]`, in both modes, for every
  method.
- `test_warning_lists_the_largest_ten_in_descending_max_grod`: 12
  grains (explicit `grain_id`), grain `k` (k = 1..12) a 1 x 3 row
  whose outer pixels are rotated by +-(3.0 + 0.1 k) deg about z from
  its middle pixel, `method="center"` (correct mode: the middle pixel
  is the reference, so `max_grod_k` = 3.0 + 0.1 k within 1 float32
  ulp), `max_angle=3`: exactly one `UserWarning` containing
  "misorientation ball", "max GROD", the count "12" and "3"
  (`max_angle`), no spacing clause; the ten listed labels are 12, 11,
  ..., 3 in that order with their max GROD and `n_pixels` 3; labels 1
  and 2 are absent.
- `test_no_warning_within_max_angle`: spread 1 deg, `max_angle=3`.
- `test_warning_is_strict_at_max_angle`: a 2-pixel grain,
  `method="center"`: run once and read `m = table.max_grod.max()`
  (float32); `max_angle=float(m)` gives no warning, `max_angle=float(
  np.nextafter(m, np.float32(0)))` (the float32 neighbour on NumPy 1.23
  and 2 alike; `np.nextafter(m, 0)` is float64 on NumPy 1.23; corrected
  2026-10-06, spec review round 2) gives one (`max_angle` does not
  change `max_grod`; the comparison is float64, D5.8).
- `test_correct_center_pixel_has_zero_grod` (D20.2):
  `grain_reference_orientation_deviation_map(xmap, grain_id, table)`
  with a correct-mode `"center"` table is exactly 0.0 at each grain's
  centre pixel (exact by the D3.1 near-one snap; round 2: the first
  grain is a 3 x 3 block whose centre pixel holds the V3 `below`
  rotation exactly and whose other pixels are it rotated by 0.5 deg
  about z, so the old clip would give ~3e-6 deg there; S18).
- `test_compat_center_grod_is_measured_from_the_box_centre_pixel`: a
  ring-shaped grain (the V6 fixture) in compat `"center"`: the map
  equals the per-pixel orix angle to the K4 box-centre pixel's
  rotation (outside the grain), not to the centroid-nearest pixel;
  the two differ by more than 0.1 deg on some pixel.
- `test_variant_scrambled_pixels_have_the_same_grod`: the same grain
  with `scramble_seed=11` (random LEFT operators) gives the same map
  within 1 float32 ulp.
- `test_deviation_map_and_max_grod_agree_bitwise`: the per-grain
  `np.nanmax` of the public map equals `table.max_grod` bitwise; NaN
  at label 0, invalid grains and absent points (V16 extends the
  agreement to the driver's props).

`TestGrainTable`:
- `test_fields_dtypes_and_label_order`: fields and dtypes of D2.1
  (`n_pixels` int64 `(n,)`, `bounding_box` int64 `(n, 4)`, `rotation`
  `Rotation (n,)`, `phase_id` int32, `kappa` float64, `max_grod`
  float32, `valid` bool, `method` str); index `i` is label `i + 1`;
  `n_grains` property; the dataclass is frozen
  (`dataclasses.FrozenInstanceError` on assignment).
- `test_invalid_grains_are_identity_with_nan_max_grod`: `valid ==
  (kappa != -1)`.
- `test_center_kappa_is_one`: `method="center"` gives `kappa == 1.0`.
- `test_from_crystal_map_round_trip` (added 2026-10-06, spec review,
  so that Stage A covers `_grains.py`): a hand-built `CrystalMap`
  carrying `grain_id` and the props written by
  `_broadcast_grain_props(table, grain_id)` (with a vanished label and
  an invalid grain) gives `GrainTable.from_crystal_map(xmap)` equal to
  `table` field by field, `method` None; the vanished label's row is
  the D2.1 default in both (`n_pixels` 0, box `(0, 0, 0, 0)`, identity,
  `phase_id` -1, `kappa` -1.0, `max_grod` NaN, `valid` False; added
  2026-10-06, spec review round 2), and `n_grains` equals
  `grain_id.max()`.
- `test_map_grid_restores_one_row_one_column_and_one_point_maps`
  (D1.9): maps from `create_coordinate_arrays` with shapes (1, 6), (6,
  1), (1, 1), (2, 5) give `(ny, nx)` (1, 6), (6, 1), (1, 1), (2, 5)
  and the flat-index grid, although orix reports `xmap.shape` (6,),
  (6,), (), (2, 5).
- `test_map_grid_spans_points_not_in_the_data` (added 2026-10-06,
  spec review round 2; D1.9): (3, 4) maps from
  `create_coordinate_arrays` with `is_in_data` False on row 0 (flat
  0-3), on the last column (3, 7, 11), and on every point but flat 5:
  each grid is (3, 4) with -1 exactly at the absent positions and the
  in-data points numbered 0, 1, ... in raster order (the last case:
  `grid[1, 1] == 0`, all else -1), although orix reports `xmap.shape`
  (2, 4), (3, 3), (1, 1); a (1, 5) map with flat 0 absent gives (1,
  5), a (5, 1) map with flat 4 absent (5, 1); `xmap[1:3, 1:3]` of a
  full (3, 4) map gives the (3, 4) grid with 0..3 at `[1:3, 1:3]` and
  -1 elsewhere (D1.9 consequence); and
  `kernel_average_misorientation_map` of a (3, 4) gradient map with
  row 0 absent returns (3, 4), NaN on row 0, and at `(2, 1)` (no
  absent neighbour) the full map's value bitwise. Probe 2026-10-06
  (round 2): this grid rule gives these shapes on orix 0.12.1 (numpy
  1.23.0) and 0.14.2.

### V9 -- Compat EM quirks (`test_hrosm_averaging.py`, `TestCompatEM`) -- Stage A

Pins D5.4-D5.7, K5-K7, D5.10, R5. The tests read `EMResult`, the
diagnostics record of D5.5 (named 2026-10-06, spec review: `mu`,
`kappa`, `log_likelihood` (per-init final `L`), `n_iterations`
(per-init), `best_init`) returned by `_em_emsoft` and `_em_correct`.
Oracle: the test-local transcription `emsoft_em(x, operators, kind,
n_em, n_iter, rng)` of `EMforDS_`, `Estep`, `Mstep`, `getQandL`,
`logCp` and the kappa table (`mod_dirstats.f90:805-1285`, the table at
`mod_dirstats.f90:243-261`; corrected 2026-10-06, spec review) written
from the source in EMsoft operator order, `Qi = Li = 0.0`, `rng =
default_rng(seed)`. Runtime rule (amended 2026-10-06, spec review): the
transcription keeps EMsoft's loop order over inits and iterations and
is vectorised with NumPy over samples and operators; the default arms
run `n_em=3` (`n_iter=40`), the `n_em=25` arms are weekly. CI; < 3 s.

- `test_matches_the_seeded_transcription[kind-case]`: kind in {vmf,
  watson}; cases `alpha20-n30`, `alpha1000-n50`, LEFT-scrambled
  (`default_rng(80)`), `n_em=3` (weekly siblings at `n_em=25`): `mu`,
  `kappa`, per-init `L`, per-init iteration counts and the chosen
  init equal the transcription bitwise (MTP: seed bitwise; fallback 4
  float64 ulp on `mu`/`kappa` with identical counts and init index,
  recorded).
- `test_compat_model_is_right_sided[kind]`: `alpha 100`, N 300:
  compat recovers RIGHT-scrambled samples (error to the class `{mu
  S_j}` <= the V8 band, `kappa_hat` within the ratio band) and on the
  same samples scrambled LEFT gives `kappa_hat` <=
  `WRONG_SIDE_MAX_KAPPA_RATIO` x the right-scrambled value; correct
  mode is the mirror image (V8).
- `test_vmf_at_realistic_kappa_runs_every_iteration_and_keeps_the_first_init`:
  VMF `kappa = 1e4` (alpha 1250), N 50: every init runs `n_iter`
  iterations, `Q` is non-finite, the chosen init index is 0.
- `test_watson_underflow_keeps_the_previous_q_and_exits_at_the_second_iteration`
  (rewritten 2026-10-06, spec review round 2; D5.5): Watson `kappa =
  1e4`, N 50, samples NOT scrambled (`sample_s3` with `default_rng(80)`;
  LEFT-scrambled samples cannot be aligned by the right-sided compat
  model, so `kappa_hat` stays small and `Phi` never underflows),
  `n_em=3`, `n_iter=40`, `seed=0`. The test-local transcription runs
  with a per-iteration trace (`Q_i`, `L_i`, whether `min(Phi) <= 0`).
  Precondition (asserted): at least one init whose iteration-2 `Phi`
  underflows; if `default_rng(80)` gives none, the Stage A writer takes
  the first `default_rng(80 + k)`, `k = 1, 2, ...`, that does and
  pins it as `WATSON_UNDERFLOW_SEED` (MTP, seed 80). For every such init the trace has `Q_2 == Q_1`,
  `L_2 == L_1` and stops at `i = 2`, and `_em_emsoft` reports
  `n_iterations == 2` exactly and `log_likelihood` equal to the trace's
  `L_1` for that init; the whole `EMResult` equals the transcription
  under the pin of `test_matches_the_seeded_transcription` (bitwise;
  its recorded 4-ulp fallback, if taken, also covers `L`, with
  iteration counts and the init index identical). Per-init `L`,
  iteration counts and the chosen init are recorded (`record_property`, MTP), never asserted as
  constants: `Qi`, `Li` persist across inits, so an init whose
  iteration-1 `Phi` already underflows inherits the previous init's
  `L`, and the chosen init depends on the draws.
- `test_final_representative_side`: for `mu` = Euler `(37, 51, 113)`
  deg, compat returns the `mu * S_i` with the largest `|q0|` (first
  index on ties, q0 >= 0), correct returns `S_i * mu` with the largest
  `|q0|`; the two differ.
- `test_kappa_gate_is_strict[mode-kind]`: run once (seed 0), read
  `kappa_hat`; rerun with `min_kappa=kappa_hat` -> `valid` False,
  `kappa == -1.0`, identity rotation, `max_grod` NaN; with
  `min_kappa=np.nextafter(kappa_hat, 0)` -> kept.
- `test_nan_kappa_is_rejected_and_inf_is_kept`: all-identical samples
  give `y = 1`, `kappa = +inf` (kept); `_apply_kappa_gate(np.nan,
  5.0)` is False and `_apply_kappa_gate(np.inf, 5.0)` True.
- `test_kappa_lookup_and_closed_forms`: for `y` in
  `np.linspace(0.05, 0.99, 95)` the private estimator equals the
  test's brute-force table (`xAp = 0.001 + (i - 1) * 0.001`, `i =
  1..35000`, EMsoft's order; `yAp` VMF `I2(k) / I1(k)`, Watson
  `(I1(k/2) / (I0(k/2) - I1(k/2))) / k`, Bessel ratios from
  `scipy.special.ive`, first minimum, index 1 -> 2) for `y < 0.94` and
  the closed forms of D5.4 for `y >= 0.94`, bitwise; and the table's
  `xAp` differs from `0.001 * i` on 11,504 of 35,000 entries (the order
  matters).
- `test_log_normaliser_branches`: `logCp` equals the D5.4 formulas
  (`C = -3.675754132818690967`, `C2 = 4.1746562059854348688`, `C2W =
  5.4243952068443172530`) on both sides of `kappa = 30` (VMF) and 20
  (Watson) within 1e-12 relative.

### V10 -- Compat OSM (`test_hrosm_osm.py`, `TestCompatOSM`; `test_hrosm_emsoft_regression.py`, `TestCompatOSMOnEMsoftFiles`) -- Stage A

Pins D7.2, D7.3, K9. Oracle: test-local transcription
`emsoft_osm_loop(top1based, n, H, W)` of `getOrientationSimilarityMap`
(`mod_DIsupport.f90:172-275`): the loop of D3.2 with `dis` replaced
by `vectormatch` (`mod_math.f90:3620-3645`) on the first `n` 1-based
indices, float32 `localosm`, the multipliers of D3.4 in float32 with
`third(x) = (x * np.float32(4)) / np.float32(3)`, no division by `n`.
CI arms < 1 s.

- `test_matches_the_loop_transcription_bitwise[shape]`:
  `hrosm_top_lists(shape, n=10, pool=15)`, shapes `(1, 1)`, `(1, 6)`,
  `(6, 1)`, `(2, 2)`, `(5, 7)`, `(9, 11)`; `n` 10 and 5.
- `test_edge_multiplier_follows_the_source_order`: a `(3, 4)` box,
  `n = 3`, lists built so that top-edge pixel `(0, 1)` accumulates
  `2 + 2 + 1 = 5` (left pair, right pair, shifted down pair of pixel
  `(0, 0)`): its value is exactly `np.float32(1.6666666)` =
  `(np.float32(1.25) * 4) / 3`, not `np.float32(1.6666667)` =
  `np.float32(1.25) * np.float32(4 / 3)` (drafting arithmetic: the two
  forms differ at accumulated counts 5, 7, 10, 14, 17, 20, ...).
- `test_compat_osm_is_not_divided_by_n`: identical lists everywhere,
  `n = 10`, every pixel exactly 10.0 (incl. corners and edges).

`TestCompatOSMOnEMsoftFiles` (1-based `TopMatchIndices[:N]`, map from
the namelist):

| test | lists -> compared with | gate | pin (seed) |
|---|---|---|---|
| `test_shipped_osm_matches[osm-10]`, `[osm_05-5]` | `regression_hrosm_large_di` `TopMatchIndices[:, :n]` -> `OSM`, `OSM_05` | CI | `SHIPPED_OSM_DIFF` (0 if the generating build keeps the source order; if it folds `4/3`, every differing pixel is a `third` pixel and differs by exactly 1 ulp, and `x * float32(4 / 3)` reproduces it; count pinned) |
| `test_grx810_osm_matches[OSM]`, `[OSM_20]` | `dp-GRX-refined.h5` (`nosm` from its namelist; 20) | local | bitwise (seed 139,181 of 139,181 for the form measured) |
| `test_al_osm_matches` | `Al_HROSM/dp-full.h5` `OSM` | local + weekly | bitwise (seed 501,592 of 501,592) |
| `test_ni6_osm_differs_only_on_third_pixels_by_one_ulp` | `dp-Ni6-refined.h5` `OSM` (`nosm 20`) | local | every differing pixel is a `third` pixel, 1 ulp, `x * float32(4 / 3)` matches it; `NI6_OSM_EDGE_PIXELS` (seed 214) |

"`third` pixels": those the D3.4 sequence passes through `third`
(top and bottom row interior, left and right column interior, and
`acc[n - W]`).

### V11 -- Grain-aware correct OSM and the public routing (`test_hrosm_osm.py`, `TestGrainAwareOSM` [A]; `test_orientation_similarity_map.py`, `TestHROSMKeywords` [B])

Pins D7.1, D7.5, R7. CI; < 1 s.

`TestGrainAwareOSM` (array function):
- `test_equals_the_legacy_function_on_one_grain[shape]`: shapes `(6,
  7)`, `(3, 3)`, `(2, 5)`; random lists, `grain_id` all 1, all
  re-indexed: equal to `orientation_similarity_map(xmap)` (legacy
  path, `_orientation_similarity_map.py:95-152`) bitwise (float32).
- `test_cross_grain_neighbours_are_excluded`: two grains (columns
  0-3, 4-7 of `(5, 8)`), grain A lists all `1..10`, grain B all
  `11..20`: every A pixel is 10.0 including column 3; counting the
  cross-grain neighbour would lower column 3.
- `test_pixel_without_an_in_grain_neighbour_is_nan`.
- `test_not_reindexed_neighbours_are_excluded`.
- `test_range_is_zero_to_n`: disjoint lists give 0.0, identical give
  `n`.

`TestHROSMKeywords` (Stage B, public function):
- `test_defaults_run_the_legacy_path_unchanged`: the three existing
  tests' inputs give identical arrays with and without
  `grain_id=None, emsoft_compatible=False`; a `(2, 5)` map gives the
  same `(2, 5)` array bitwise with and without the keywords (amended
  2026-10-06, spec review: the drafted `(1, 6)` arm cannot pass, since
  the legacy function raises "footprint.ndim" on one-row, one-column
  and one-point maps, D7.5); a full 3 x 3 footprint with
  `center_index=4` still runs and equals the legacy value computed by
  `generic_filter` in the test; `from_n_best=5` still returns the
  legacy 3-D array.
- `test_grain_id_routes_to_the_grain_aware_map` and
  `test_emsoft_compatible_routes_to_the_emsoft_table`.
- `test_cannot_be_combined_with_footprint_center_index_or_from_n_best`:
  `ValueError` "cannot be combined" for each.
- `test_normalize_divides_by_n` and `test_n_best_sets_n`.
- `test_absent_points_and_keep_n_one_work_without_squeeze`: a `(1,
  6)` map with one absent point and a squeezed `keep_n == 1` prop
  returns float32 `(1, 6)`.

### V12 -- EMHROSM cluster stage (`test_hrosm_emsoft_regression.py`, `TestClusterStage`) -- Stage A

Pins D4 (rule, K3 dilate, boxes), D5.3 (K4), D5.5 (K5-K7 bands), D12.2,
K12 as written by the binary, R3. Scenarios `center`, `center_dilate`,
`wat` (shipped, `regression_hrosm_large_<scenario>.npz`, map 55 x 75,
`gangle 5`, `misorang 5`, `nsamples 20`, `nosm 10`). CI; < 2 s.

- `test_grain_ids_from_the_reference_kam_are_bitwise[scenario]`:
  `segment_grains_kam(ref["kam"], threshold=5.0, dilate=<scenario>,
  emsoft_compatible=True)` equals `grainID` bitwise; label count
  equals `nGrains`.
- `test_grain_ids_from_the_euler_angles[scenario]`: compat KAM from
  the shipped `RefinedEulerAngles` then segmentation: differing
  pixels `SHIPPED_GRAIN_ID_FROM_EULER_DIFF` (seed 0; any difference
  must lie in the 8-neighbourhood of a KAM pixel counted by V2).
- `test_pixel_counts_and_boxes[scenario]`: `npixels` and the converted
  `grainROI` equal ours bitwise.
- `test_center_average_is_the_box_centre_pixel[center-center_dilate]`:
  `avor` equals EMsoft `eq_` of the K4 pixel's float32 angles,
  `kappa == 1.0`; `CENTER_AVOR_MAX_ULP` (seed 0 = bitwise; fallback 1
  float64 ulp per component, count recorded).
- `test_watson_average_within_bands`: `wat` `avor` vs our compat
  Watson (seed 0): symmetry-aware angle <= `WAT_AVOR_MAX_DEG` (seed
  0.01 deg) and `|kappa / kappa_ref - 1| <= WAT_KAPPA_REL` (seed 0.01)
  per grain; the binary's seed is the clock (`mod_cluster.f90:266`), so
  never bitwise.
- `test_new_euler_lies_in_the_grain_ball[scenario]`: for re-indexed
  pixels (`newCI > 0`), `Rotation.from_euler(newEuler.astype(
  np.float64))` is within `5 + 1e-3` deg of `Rotation(avor[g - 1])`
  (no symmetry: the ball is a misorientation of the raw average);
  `newEuler` is radians (degrees would leave the ball).
- `test_unprocessed_pixels_are_zero_filled[scenario]`: pixels with
  `grainID == 0` or of a skipped grain (`npixels < 10` or `kappa ==
  -1`) have `newOSM == 0`, `newEuler == 0`, `newCI == 0`.

Local (Ni6 HROSM run of 2026-04-11: `gangle 10.0`, `dilate .TRUE.`,
`orav 'averageWAT'`, 151 x 186 (H x W) map, 62 grains):
- `test_ni6_clustering_is_bitwise`: the file's own `kam` -> labels,
  `npixels`, `grainROI` bitwise (seed 0 of 28,086 differ).
- `test_ni6_watson_average_within_bands`: `NI6_WAT_AVOR_MAX_DEG`
  (seed 0.01), `NI6_WAT_KAPPA_REL` (seed 0.01); kappa range
  8.19-34,896, none rejected.
- `test_ni6_small_grains_are_not_reindexed`: grains 35 and 62 have
  fewer than 10 pixels and only zero `newOSM`, `newEuler`, `newCI`.

### V13 -- End-to-end tolerance against EMHROSM (`test_hrosm_emsoft_regression.py`, `TestEndToEnd`) -- Stage B

Pins D8 (driver in compat mode) against the binary; tolerance-based
because the dictionary stage is kikuchipy-native (parked decision 4).
Inputs: `nickel_ebsd_large` [download], static then dynamic
background removed (as the reference script); the detector of the
dataset with `det.pc = det.pc_average` (one PC, the reference route);
input `xmap` from the shipped `RefinedEulerAngles` (float64 promoted,
phase Ni m-3m, shape (55, 75)); `average="center"`, `threshold=5`,
`max_angle=5`, `n_steps=20`, `keep_n=20`, `n_osm=10`, `pc="single"`,
`emsoft_compatible=True`. Compared pixels `P`: grain pixels of grains
re-indexed by both sides whose box has `W * H >= 32`
(`numdictsingle`, D8.10). Disorientation = symmetry-aware angle
between our rotation and `Rotation.from_euler(newEuler)`.

Pins split 2026-10-06 (spec review): the two arms use different
masters (401 px shipped vs the 1001 px EMsoft master), so their
tolerances differ.

| quantity | seed | pin |
|---|---|---|
| `E2E_ONE_GRAIN_DISORIENTATION_MEDIAN_DEG` | 0.3 | MTP (weekly) |
| `E2E_ONE_GRAIN_DISORIENTATION_P99_DEG` | 1.0 | MTP (weekly) |
| `E2E_FULL_MAP_OSM_PEARSON_MIN` (r of our `osm` vs `newOSM` over `P`) | 0.8 | MTP (local + weekly) |
| `E2E_FULL_MAP_DISORIENTATION_MEDIAN_DEG` | 0.3 | MTP (local + weekly) |
| `E2E_FULL_MAP_DISORIENTATION_P99_DEG` | 1.0 | MTP (local + weekly) |
| `E2E_FULL_MAP_CI_PEARSON_MIN` (r of our `scores[:, 0]` vs `newCI`) | 0.5 | MTP (local + weekly) |

- `test_one_grain_against_emhrosm` (weekly, [download]; moved out of
  the default suite 2026-10-06, spec review, for the CI budget):
  master `nickel_ebsd_master_pattern_small(projection="lambert",
  hemisphere="both")`; the grain with the smallest box satisfying `W
  * H >= 32` and `npixels >= 10` in the `center` reference, run
  through `_driver._hrosm(..., grains=[label])` (D8.15): disorientation
  median and p99 within the one-grain pins; every other pixel keeps
  the compat fill (identity, `osm` 0.0).
- `test_full_map_against_emhrosm` (local + weekly, [download]: minutes
  even on this machine, so runners without `KIKUCHIPY_EMSOFT_DATA`
  skip it; it reads no EMsoft data file): master `kp.data.ebsd_master_pattern("ni", hemisphere="both",
  projection="lambert", energy=20, allow_download=True)` (skips with
  the instruction when offline); the four full-map pins; runtime
  recorded.

### V14 -- Reference files, reader and regenerate-and-diff (`test_hrosm_emsoft_regression.py`, `TestReferenceFiles`, `TestEMsoftFileReader`, `TestRegenerateReferences`) -- Stage A

Pins D12, D13.2-D13.6. `TestReferenceFiles` is cloned from
`test_spherical_emsphinx_regression.py:920-1061` (not its module
docstring, which names a `specs/` path). Frozen key table (arrays
uncompressed `np.savez`, `allow_pickle=False`; N = 4,125; `n` = grain
count):

| file | arrays |
|---|---|
| `regression_hrosm_large_di.npz` | `TopMatchIndices` int32 (4125, 10) 1-based; `KAM`, `OSM`, `OSM_05` float32 (55, 75); `CI` float32 (4125,) |
| `regression_hrosm_large_refined.npz` | `RefinedEulerAngles` float32 (4125, 3); `RefinedDotProducts` float32 (4125,); `EulerAngles` float32 (4125, 3) |
| `regression_hrosm_large_{center,center_dilate,wat}.npz` | `nGrains` int32 (); `grainID` int32 (55, 75); `npixels` int32 (n,); `grainROI` int32 (n, 4); `avor` float64 (n, 4); `kappa` float64 (n,); `kam`, `newOSM`, `newCI` float32 (55, 75); `newEuler` float32 (55, 75, 3); `hrosm_namelist` str |
| `regression_hrosm_ball_n6.npz` | `qu` float64 (2197, 4); `ro` float64 (2197, 4) |

**Budget arithmetic** (D13.3 as amended 2026-10-06, spec review; this
table is THE layout, no conditional trimming): the drafted D13.3 put
`EulerAngles` in the DI file (280,500 B of arrays, over the 250 kB
cap); `EulerAngles` lives in the refined file (DI 231,000 B, refined
115,500 B), `newQuat` (optional in D12.2, the float32 `eq_` of
`newEuler`) and `maxGROD` are not shipped (scenario files ~118 kB
each) and the ball file ships `qu` and `ro` only (140,600 B; `eu` is
checked by the bin arm). Total seed ~841,000 B, pinned as measured in
`REFERENCE_TOTAL_BYTES`; every file < 250,000 B.

Provenance in every file (0-d unless stated): `program_md5` str
(`name=md5` pairs joined by `;`, sorted, exactly `EMDI.exe`,
`EMFitOrientation.exe`, `EMHROSM.exe`, `EMgetOSM.exe`,
`EMsampleRFZ.exe`, `EMsoftOOLib.dll`, `EMOpenCLLib.dll`),
`emsoft_version` str, `emsoft_commit` str, `master_md5` str (the
cached original), `master_run_md5` str (the copy with both `xtalname`
datasets rewritten to `Ni.xtal`, D13.2.2; added 2026-10-06, spec
review), `patterns_md5` str, `pc` float64 (4,), `namelist` str,
`gpu_name` str, `numdictsingle` int64, `numexptsingle` int64,
`kikuchipy_version` str.

`TestReferenceFiles` (CI):
- `test_registry_lists_every_reference`: `emsoft_hrosm/
  regression_hrosm_*.npz` registry keys equal the directory glob of
  the installed package; each `Dataset(key).has_correct_hash`.
- `test_scenario_set_is_complete`: quad equality of the script's
  `SCENARIOS`, the registry, the glob and the module's frozen table.
- `test_references_load_without_pickle[name]`,
  `test_frozen_keys_and_dtypes[name]`.
- `test_provenance_pins[name]`: `master_md5 ==
  "8b69c071a036ad3488d465093b67fe4d"` (amended in the same ledger
  entry if the fallback master is used); `master_run_md5` a 32-hex
  value different from `master_md5`, identical across files;
  `numdictsingle == numexptsingle == 32`; `program_md5` has the 7
  names with 32-hex values, identical across files; `emsoft_version`
  starts with `"6_0_"`; the namelist text holds `nnk = 20`, `nosm =
  10`, `ncubochoric = 100`, and every path value starts with
  `kikuchipy_hrosm/` or is `'undefined'` or a `tmpfile` name; scenario
  namelists hold their `orav` and `dilate`.
- `test_pc_route_is_reproduced` ([download]): from `nickel_ebsd_large`
  with `det.pc = det.pc_average`: `pc = det.pc_emsoft()[0]` (a `(1,
  3)` array; corrected 2026-10-06, spec review), `xpc, ypc = pc[:2] /
  det.binning`, `L = pc[2]`, `delta = det.px_size * det.binning`, each
  rounded through `.6g`, equals the stored `pc` (measured 2026-10-06:
  4.6044, 17.182, 240.996, 8.0).
- `test_each_file_within_budget`: every file `< 250_000` B; total
  recorded and `<= REFERENCE_TOTAL_BYTES` (MTP, seed 841,000).
- `test_script_is_import_safe`: with both environment variables
  deleted (`monkeypatch.delenv`) and `builtins.open` patched to raise,
  `import kikuchipy.data.emsoft_hrosm.create_hrosm_reference`
  succeeds and exposes `SCENARIOS`.

`TestEMsoftFileReader` (CI; synthetic HDF5 files written with h5py in
`tmp_path` in EMsoft's layout: Fortran `(x, y)` arrays stored so h5py
sees `(H, W)`, `TopMatchIndices` padded to a multiple of
`numexptsingle`, namelist strings as object arrays of bytes):
- `test_dot_product_file_shapes_and_padding`: map `(ipf_ht, ipf_wd)`;
  `TopMatchIndices` sliced to `[:N]`, kept 1-based; `(3, N)` read as
  `(N, 3)`; namelist scalars typed.
- `test_roi_sets_the_map_shape`: `sum(ROI) != 0` gives `(ROI(4),
  ROI(3))`.
- `test_euler_datasets_are_returned_in_radians`: `EulerAngles`,
  `RefinedEulerAngles`, `newEuler` are returned unchanged (radians).
- `test_hrosm_file_and_dilate_from_the_namelist_text`: `dilate`
  parsed from `NMLfiles/HROSMNML` (`dilate = .TRUE.` -> True) even
  though `NMLparameters/HROSMNameList` lacks it; absent `newQuat` and
  `maxGROD` read as None.

`TestEMsoftProgramLock` (CI, < 1 s; added 2026-10-06, spec review,
D13.1): with a lock path in `tmp_path` and the heartbeat patched to
0.05 s: a lock file whose mtime is set (`os.utime`) older than the
stale age is taken over; while one holder is inside the context, the
file's mtime advances (heartbeat) and a second acquirer with a short
patched timeout raises `TimeoutError` naming the path; the file is
removed on release.

`TestRegenerateReferences`:
- `test_the_mismatch_message_names_the_programs_and_the_arrays` (CI):
  a stand-in file (one array dropped, one changed, one widened) makes
  the diff helper's message name `program_md5`, "the contract is
  bitwise", the changed array and the dtype change.
- `test_regenerated_references_are_bitwise` (bin, runtime ~10-25 min,
  `REGENERATION_RUNTIME_S` recorded): `create_hrosm_reference.main(
  output_dir=tmp_path, bin_dir=emsoft_bin_dir)` (never the default
  output directory, which is the package's own; amended 2026-10-06,
  spec review, the `test_spherical_emsphinx_regression.py:1547`
  precedent) in a new `kikuchipy_hrosm/<YYYYmmdd-HHMMSS>/` run
  directory, compared with `Path(kp.data.__file__).parent /
  "emsoft_hrosm"`, whose md5s are asserted unchanged after the run,
  reproduces every shipped file (and `newQuat` of the run equals
  float32 `eq_` of its `newEuler`): CPU-only arrays (`KAM`, `OSM`,
  `OSM_05`, `grainID`, `kam`, `npixels`, `grainROI`, ball lists)
  bitwise; GPU-derived arrays (`TopMatchIndices`, `EulerAngles`,
  `RefinedEulerAngles`, `RefinedDotProducts`, `CI`, `newOSM`,
  `newEuler`, `newCI`, `wat` `avor`/`kappa`) per `GPU_ARRAY_POLICY`
  (MTP: each array "bitwise" or a band, decided at the first
  regeneration; seeds: bitwise except `TopMatchIndices` tie swaps,
  counted, and `wat` within the V12 bands). The script's acid bands
  (top-1 median <= ~1.5 deg, refined median <= 0.5 deg vs the stored
  `xmap`) and the `OSM_10 == OSM` self-check abort it before writing;
  no new entry appears in the `EMdatapathname` root.

### V15 -- Physics sanity: a sub-grain boundary (`test_signals/test_ebsd_hrosm.py`, `TestSubgrainContrast`) -- Stage B

Pins the purpose of HROSM (sub-degree contrast a global dictionary
cannot resolve) in correct mode. Fixture (module scope, computed
once): map `(10, 16)`; grain A columns 0-7 with `g_A` = Euler `(10,
20, 30)` deg on columns 0-3 and `Rz(0.5 deg) * g_A` on columns 4-7;
grain B columns 8-15 with `g_B` = Euler `(60, 45, 10)` deg (generator
asserts > 20 deg to `g_A`); `hrosm_synthetic_signal(sig_shape=(32,
32), noise=0.02, seed=50)`. Global dictionary: the union of
`misorientation_ball(c, max_angle=10, n_steps=7)` (3,375 each, ~1.43
deg spacing) around `c_A = Rv(0.7 deg) * g_A` and `c_B = Rv(0.7 deg) *
g_B`, `Rv` about `(2, -1, 3) / sqrt(14)` (no grid point on the truth),
simulated with the same detector; `s.dictionary_indexing(dict,
keep_n=10)` gives the input `xmap` and the global OSM
(`orientation_similarity_map(xmap, n_best=10)`). HROSM: `s.hrosm(
xmap, mp, det, energy=20, max_angle=2.0, n_steps=8, keep_n=10,
n_osm=10)` (4,913 orientations, 0.25 deg shells); GROD warnings
ignored. Contrast `c(osm)` = median over rows 1-8 of columns 1, 2, 5,
6 minus median over rows 1-8 of columns 3, 4.

| quantity | seed | pin |
|---|---|---|
| `SUBGRAIN_CONTRAST_MIN` (HROSM `c`, of 10) | 3.0 | MTP |
| `SUBGRAIN_CONTRAST_RATIO` (HROSM `c` / max(global `c`, 0.5)) | 2.0 | MTP |
| `SUBGRAIN_MEDIAN_ERROR_DEG` (HROSM rotation vs truth, grain A non-boundary pixels) | 0.15 | MTP |
| `SUBGRAIN_STEP_TOL_DEG` (median angle between columns 2 and 5 minus 0.5) | 0.15 | MTP |

- `test_grains_are_segmented_without_splitting_the_subgrain`: the
  interior of A (columns 0-6) holds one label, B another.
- `test_hrosm_osm_shows_the_subboundary`: contrast pins; global
  contrast recorded.
- `test_subgrain_step_is_recovered`: median error and step pins; the
  global top-1 step is recorded.
- `test_full_size_subgrain_demonstration` (weekly): map `(20, 30)`,
  60 x 60 px, sub-boundary between columns 7 and 8, grain B columns
  15-29, `max_angle=5`, `n_steps=20`; the same four pins (separate MTP
  values recorded), runtime recorded.

### V16 -- API, output and driver contracts (`test_signals/test_ebsd_hrosm.py`, `TestValidation`, `TestOutput`, `TestContracts`, `TestInvariance`, `TestMessages`; `test_dictionary_indexing.py`) -- Stage B

Pins D1.3-D1.6, D2.2-D2.3, D8.1-D8.14, K10-K12, R1-R4, R7. Base fixture
(module scope): `hrosm_grain_xmap(shape=(8, 10), n_grains=2,
gradient=0.2)` with each rotation further left-multiplied by a random
rotation of at most 0.3 deg (`default_rng(90)`, the "indexing
error"), `hrosm_synthetic_signal` 32 x 32, noise-free; calls use
`max_angle=1.0, n_steps=3` (343 orientations) unless stated; one
cached `hrosm` result per mode (`run_correct`, `run_compat` with
`average="center"`, `pc="single"`). CI; ~16 s.

`TestValidation`:
- `test_signatures_are_frozen`: `inspect.signature` of the eight public
  names (incl. `misorientation_ball_spacing` and
  `grain_reference_orientation_deviation_map`, D20),
  `orientation_similarity_map` and `EBSD.hrosm` equals D1.3 (names,
  order, kinds, defaults; `emsoft_compatible` keyword-only, default
  False; no `show_progressbar`, no `max_chunk_bytes`).
- `test_arguments_are_validated_in_order[case]`: one row per D1.5
  check with its fragment ("two navigation dimensions", "xmap shape",
  "detector shape", "one PC or one PC per map point", "n_osm",
  "keep_n", "average", "pc", the mask checks of D1.5 item 5 as
  amended 2026-10-06 ("navigation mask must be a NumPy array" for a
  list, never `AttributeError`; "signal mask must be a NumPy array";
  "boolean"; "navigation mask shape"; "signal mask shape"; "at least
  one pattern"), "emsoft_compatible requires every map point", "one
  phase", "pc='single'"); `True` for an integer argument raises; only
  `ValueError`, never `TypeError`; a case violating two checks reports
  the earlier one; a one-row EBSD signal with a one-row map passes the
  "xmap shape" check (D1.9).

`TestOutput`:
- `test_one_crystal_map_with_the_input_frame`: same shape, `x`/`y`,
  phase ids, phase list, `is_in_data`, scan unit; input props not
  carried.
- `test_properties_dtypes_shapes_and_fills[mode]`: exactly the 10
  props of D2.3 with their dtypes and shapes; correct fills: `osm`
  NaN, `scores` NaN rows, `simulation_indices` -1 rows, rotation =
  input rotation where not re-indexed; compat: `osm` 0.0, `scores`
  0.0 rows, -1 rows, identity rotation; `grain_id` 0 unassigned, 1..n
  (EMsoft convention); `kam` equals
  `kernel_average_misorientation_map` of the input (same mode);
  `reindexed` bool.
- `test_grod_props_equal_the_public_deviation_map[mode]` (D20.2,
  added 2026-10-06, spec review): `out.prop["grod"]` equals
  `grain_reference_orientation_deviation_map(xmap, grain_id, table)`
  bitwise (the INPUT `xmap`, `grain_id = out.prop["grain_id"]` on the
  grid; flattened, NaN positions equal) and `grain_max_grod`
  equals the broadcast `table.max_grod`, `table` from
  `GrainTable.from_crystal_map(out)`; a valid grain skipped by
  `min_pixels` still has finite `grod` and `grain_max_grod`.
- `test_simulation_indices_are_local_to_the_grain_ball`: for every
  re-indexed pixel `p` of grain `g`, `misorientation_ball(table.
  rotation[g - 1], max_angle=1.0, n_steps=3)[sim[p, 0]]` equals
  `rotations[p]` (compat: after the float32 Rodrigues storage).
- `test_grain_table_round_trips_through_the_crystal_map`:
  `GrainTable.from_crystal_map(out)` equals
  `average_grain_orientations(...)` field by field (`method` None);
  after `orix.io.save` / `orix.io.load` (h5) the same (orix 0.12.1
  round trip of bool and 2-D props verified 2026-10-06, spec review:
  no version gate).
- `test_default_average_is_mean`: the default call's `grain_kappa`
  equals the `mean` table's.

`TestContracts`:
- `test_each_grain_is_matched_against_its_own_ball[mode]`: every
  re-indexed rotation lies within `max_angle + 1e-9` deg (correct) or
  `max_angle + 1e-4` deg (compat: the K8 float32 Rodrigues storage
  moves ball points by up to 2e-7 rad, ~1.1e-5 deg, V7; amended
  2026-10-06, spec review round 2) of its `grain_orientation` (no
  symmetry) and within 0.5 deg of the truth (symmetry-aware).
- `test_navigation_masked_dictionary_indexing_map_keeps_its_grid`
  (added 2026-10-06, spec review round 2; D1.9, D8.1): `dictionary =
  mp.get_patterns(xmap.rotations, det, energy=20, compute=True)` (the
  base fixture's 80 rotations), `xmap_di = s.dictionary_indexing(
  dictionary, navigation_mask=mask, keep_n=1)` with `mask` True on row
  0 (so `xmap_di.is_in_data = ~mask`, `_dictionary_indexing.py:141-144`,
  and orix reports `xmap_di.shape == (7, 10)`); then
  `s.hrosm(xmap_di, mp, det, energy=20, max_angle=1.0, n_steps=3,
  verbose=0)` passes the "xmap shape" check, the output's
  `is_in_data` equals `xmap_di`'s, and on the `_map_grid` grid (8, 10)
  row 0 holds no point while rows 1-7 carry two grains at columns 0-3
  and 6-9 (as on the unmasked base map; the boundary columns 4-5 stay
  unassigned), each re-indexed
  rotation within 0.5 deg of the base truth at the same (row, col).
  Cost ~2 s (80 simulated 32 x 32 patterns; cached master).
- `test_navigation_mask_true_excludes_and_fills`: mask True on 3
  interior pixels of grain A (not adjacent to grain B) whose patterns
  are replaced by constant 0: those pixels have `kam` NaN, `grain_id`
  0, `reindexed` False, the input rotation, `osm` NaN, `scores` NaN,
  `simulation_indices` -1; every output of grain B equals the
  unmasked run bitwise (its KAM, label, average and ball do not
  change).
- `test_absent_points_behave_like_masked_points`: `is_in_data=False`
  and `phase_id == -1` give the same fills.
- `test_min_pixels_boundary_is_inclusive`: `min_pixels` equal to grain
  A's pixel count re-indexes A; that count + 1 skips A (fills of
  D2.3).
- `test_invalid_grains_are_skipped`: `average="vmf", min_kappa=1e12`:
  no grain re-indexed, all-NaN `osm`, one `UserWarning` "no grain
  re-indexed".
- `test_compat_domain_is_the_bounding_box`: a spy on
  `_dictionary_indexing` records the experimental block size per
  grain: box `W * H` in compat, `n_pixels` in correct mode; compat
  copies back only the grain's pixels.
- `test_pc_policy[case]`: one-PC detector used as is; a per-point
  detector (pc of the base +- 0.002, `default_rng(91)`) with
  `pc="grain"`: a spy on `EBSDMasterPattern.get_patterns` sees, per
  grain, one PC equal to the mean PC of that grain's domain within
  1e-15; `pc="single"` sees `detector.pc_average`; compat with a
  per-point detector and `pc="grain"` raises "pc='single'".
- `test_multi_phase_uses_the_master_by_phase_name`: grain B as phase
  `"ni2"` (m-3m): its grains are skipped with one warning "no master
  pattern"; renaming the master's phase to neither raises "master
  pattern phase"; a single-phase map uses the master whatever its
  name.
- `test_compat_dilate_vanished_grain_is_not_reindexed`: a map whose
  compat dilate removes a grain runs without error; that label has
  `valid` False.

`TestInvariance`:
- `test_results_do_not_depend_on_n_per_iteration`: `n_per_iteration`
  None, 50 and 343 give identical outputs (`CHUNK_INVARIANCE`, MTP:
  seed bitwise; fallback of D8.12: identical top-1 except exact score
  ties, scores within 2 float32 ulp, recorded).
- `test_lazy_signal_equals_eager`: `s.as_lazy()` gives the same
  output (same pin).
- `test_seeded_runs_are_identical`: two `average="watson", seed=0`
  runs equal bitwise.
- `test_default_n_per_iteration`: `_default_n_per_iteration(sig_size,
  ball_size)` (D8.6) gives 17,777 for 60 x 60 (ball 68,921), 277 for
  480 x 480, the ball size when the floor exceeds it, and at least 1.
- `test_simulation_indices_are_int32_on_both_paths`: single-pass and
  chunked runs give int32.

`TestMessages`:
- `test_verbose_levels`: `verbose=0`: `capsys` out and err empty;
  `verbose=1`: one information line holding the ball size "343", the
  radius `max_angle` and the mean spacing `misorientation_ball_spacing(
  1.0, 3)` as printed (D20.1), one grain-level progress bar, one timing
  line, no `dictionary_indexing` line; `verbose=2`: also one
  `dictionary_indexing` message per re-indexed grain.
- `test_warnings_are_issued_once_per_call`: the coverage warning
  ("misorientation ball"), "no master pattern", "no grain re-indexed"
  each at most once.

Coverage-warning arms (D20.3 as amended 2026-10-06, spec review;
added in the spec review). Each patches
`EBSDMasterPattern.get_patterns` with a spy that records the call and
then raises a test-local `_StopRun` exception, so no dictionary is
simulated and each arm costs only the orientation stage; warnings are
recorded with `warnings.catch_warnings(record=True)` in call order.
`max_angle` is set below the measured max GROD of grain A of the base
fixture (`max_angle = 0.5 * grain_A_max_grod`, read from an
`average_grain_orientations` table of the same map):
- `test_coverage_warning_precedes_the_first_simulation`: exactly one
  "misorientation ball" warning was recorded before the spy's first
  call (the spy reads the recorded list when it is called); a run
  without the spy also warns exactly once.
- `test_coverage_warning_lists_grains_in_descending_max_grod`: a
  fixture variant with three grains of distinct spreads above
  `max_angle`: the message holds the count, `max_angle`, the spacing
  `misorientation_ball_spacing(max_angle, n_steps)` as printed and the
  three `(label, max GROD, n_pixels)` entries in descending max GROD.
- `test_coverage_warning_is_silent_at_equality`: `max_angle =
  float(m)`, `m` (float32) the largest max GROD of the grains to be
  re-indexed: no coverage warning; `max_angle = float(np.nextafter(m,
  np.float32(0)))`: one (corrected 2026-10-06, spec review round 2:
  `np.nextafter(m, 0)` is the float64 neighbour on NumPy 1.23; the
  comparison is float64, D5.8).
- `test_skipped_grains_do_not_trigger_the_coverage_warning`
  (fixture variant specified 2026-10-06, spec review round 2: the base
  fixture is symmetric, so grain B would also exceed `max_angle` and
  share A's pixel count): `hrosm_grain_xmap(shape=(8, 11), n_grains=2,
  gradient=0.2)` with every grain-B pixel (`truth == 2`) set to the
  one rotation `R[111](30) * g0` and the 0.3 deg indexing error applied
  to grain A only; the boundary columns 4-5 drop out, so A has 32 and B
  40 pixels (asserted from `average_grain_orientations`), and B's max
  GROD is ~0 (below `max_angle = 0.5 * A's max GROD`). With
  `min_pixels = n_pixels_A + 1`, A is skipped and B re-indexed: no
  coverage warning before the spy's first call, although
  `average_grain_orientations` on the same map warns and lists A.

`test_dictionary_indexing.py`:
- `TestDictionaryIndexing::test_verbose_false_silences_the_core_for_hrosm`:
  `_dictionary_indexing(..., verbose=False)` writes nothing to
  stdout or stderr, never calls `sleep` (a monkeypatched spy on
  `kikuchipy.indexing._dictionary_indexing.sleep`; added 2026-10-06,
  spec review round 2, D8.7) and returns the same arrays as
  `verbose=True`; the existing tests are untouched.

## Local-gated and weekly

- **local** (`KIKUCHIPY_EMSOFT_DATA`; md5s of D13.5 asserted):
  V2 Ni6 HROSM, Ni6 DI, GRX810 DI arms; V10 GRX810 and Ni6 arms; V12
  Ni6 arms. Run once per stage, outcomes in the ledger.
- **local + weekly**: the Al arms of V2 and V10 (501,592 px) and V13
  `test_full_map_against_emhrosm` (any runner without the variable
  skips them; amended 2026-10-06, spec review); run in the local
  `--weekly` gate with `KIKUCHIPY_EMSOFT_DATA` set.
- **bin** (`KIKUCHIPY_EMSOFT_BIN`, lock held): V7
  `test_binary_n20_matches_in_order`, V14
  `test_regenerated_references_are_bitwise`. Run once at Stage A after
  the reference generation and once per reference-file change; the
  binary used (D13.4) and its md5s are recorded.
- **weekly** (`--weekly`): V8 full recovery grid (18 arms), the V9
  `n_em=25` transcription siblings, V13 `test_one_grain_against_emhrosm`,
  V15 `test_full_size_subgrain_demonstration`. The fork's Weekly
  workflow is `disabled_inactivity` (last run 2026-08-10; its notebook
  job failed on the last three runs), so these arms run only in the
  local `--weekly` gate, once per stage (amended 2026-10-06, spec
  review round 2); their local serial seconds (`--durations=0`) are
  recorded with a scaled CI estimate (local seconds x the last on-push
  ubuntu py3.13 job time / the local full-suite `-n 4` time; target
  <= 5 min), never a gate. Re-enabling the workflow is `plan.md` 7.4
  item 26. The performance baselines are local ledger runs, not tests.
- **Not built**: a Ni6 end-to-end arm against
  `dp-Ni6-refined_HROSM.h5` `newOSM` (patterns `EDAX-Ni.h5`, 103 MB;
  the master was regenerated after the run, D13.5): `plan.md` 7.2
  item 11 default; Johan may override at approval (amended
  2026-10-06, spec review).
- **Local-only ledger entry, never a test**: one `si_wafer` run at
  Stage B (binned 4 x to 120 x 120, `n_steps=10`), runtime and grain
  count recorded for the tutorial's cost table, or "not cached".
- **Oldest matrix** (D15.4, A6), one recorded local run per stage:
  `uv run --isolated --python 3.10 --extra tests --with
  "dask==2021.8.1" --with "diffsims==0.5.2" --with "hyperspy==2.2"
  --with "matplotlib==3.6" --with "numba==0.57" --with
  "numpy==1.23.0" --with "orix==0.12.1" --with "pooch==1.3.0" --with
  "pyebsdindex==0.3.9.2" --with "scikit-image==0.21.0" pytest
  tests/test_indexing tests/test_signals -k hrosm -n 0 -q -p
  no:cacheprovider`; expected all default arms green (APIs verified
  present in orix 0.12.1, D15.4, incl. `Rotation.angle_with_outer`,
  `scipy.spatial.cKDTree` and the `orix.io` round trip of bool and 2-D
  props, re-verified 2026-10-06, spec review); the arms to watch: the
  orix `cu2ho` comparison, and the float32 boundary arms V8
  `test_warning_is_strict_at_max_angle` and V16
  `test_coverage_warning_is_silent_at_equality` (NumPy 1.23 promotes
  all-scalar ufunc calls with value-based casting; both arms write
  `np.nextafter(m, np.float32(0))` and the code compares in float64,
  D5.8; probe 2026-10-06, round 2: on numpy 1.23.0 `np.nextafter(
  np.float32(0.4999), 0)` is `float64` and a float32 array compared
  with it gives `[False]`, with `np.float32(0)` `[True]`); and
  `_map_grid`'s use of orix's private `_original_shape` (present in
  0.12.1, probe round 2).
- **Clean-replay grep** (D14.5) at every stage close and before the
  replay: `git diff develop...HEAD -- src tests doc examples
  benchmarks conftest.py CHANGELOG.rst pyproject.toml | grep -E "^\+"
  | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|
  tech-stack\.md|mission\.md|roadmap\.md|[^A-Za-z0-9_][DVKMR][0-9]+(
  [^0-9]|$)"` prints nothing, with `*.ipynb` excluded from the diff
  and the notebook's cell sources checked through `nbformat` instead
  (stored base64 outputs give false hits; `plan.md` section 5).
- **Full existing suite** green before every stage-closing commit:
  `uv run --no-sync pytest tests -n 4` after the `-n 0` HROSM run,
  counts recorded.

## Requirement-to-test mapping

| Decision | Requirement (short) | Killer / evidence | Stage |
|---|---|---|---|
| D1 | package layout, eight public names + `EBSD.hrosm`, frozen signatures, validation order and fragments, warnings, style, test modules, the map grid (D1.9) | V16 `test_signatures_are_frozen`, `test_arguments_are_validated_in_order`, `test_warnings_are_issued_once_per_call`; V0 style arms and `test_public_docstrings_link_no_private_name_and_no_spec_id` (no mutant; the docstring rule); V1 `test_point_group_number_rejects_unmapped_or_mismatched_groups`; V8 `test_map_grid_restores_one_row_one_column_and_one_point_maps`, `test_map_grid_spans_points_not_in_the_data` (S19); V16 `test_navigation_masked_dictionary_indexing_map_keeps_its_grid`; V11 `TestHROSMKeywords`; `__all__` read by the conventions reviewer | A/B |
| D2 | `GrainTable` fields (incl. the defaults of a label without pixels, round 2), one `CrystalMap` output, props, fills, grain-local indices | V8 `TestGrainTable`; V16 `TestOutput` (all) | A/B |
| D3 | KAM correct mode, compat loop, vectorised form, multipliers, parity policy, inputs | V1 all; V2 all; V3 all | A |
| D4 | segmentation rule, csgraph closed form, threshold dtype, not-reproduced items, dilate both modes, boxes, leakage | V4 all; V5 all; V6 `TestDilateAndBoxes`; V12 grain-id arms | A |
| D5 | methods, `mean`, `center`, correct EM, compat EM, final representative, gate, GROD, seeds | V6 `TestCenterPixel`; V8 `TestRecovery`, `TestGROD`; V9 all; V12 average arms | A |
| D6 | grid, cube-to-ball port, composition side, float32 Rodrigues storage, once per call | V7 all; V16 `test_simulation_indices_are_local_to_the_grain_ball` | A/B |
| D7 | grain-aware correct OSM, compat table, multiplier order, driver domain, public routing | V10 all; V11 all; V16 `test_compat_domain_is_the_bounding_box` | A/B |
| D8 | absent points (incl. a navigation-masked `dictionary_indexing` map, round 2), orientation stage, ball per grain, domain, eager block, chunked dictionary, `verbose` (incl. no `sleep`), PC, skips, no thinning, results, determinism, multi-phase | V13; V15; V16 `TestContracts`, `TestInvariance`, `TestMessages`; `test_verbose_false_silences_the_core_for_hrosm` (S5, S20) | B |
| D9 | the twelve switches | K1: V1, V2; K2: V1 (incl. `test_matches_the_loop_transcription_on_duplicated_orientations`), V2, V3 `test_identical_neighbours_give_zero_not_nan` (correct side); K3: V6, V12; K4: V6, V12; K5-K7: V9, V12 `wat`; K8: V7, V16; K9: V10; K10: V13, V16 `test_compat_domain_is_the_bounding_box`; K11: V1 `test_rejects_absent_points_and_several_phases`, `test_point_group_number_rejects_unmapped_or_mismatched_groups`, V16 `test_arguments_are_validated_in_order`; K12: V12 `test_unprocessed_pixels_are_zero_filled`, V16 fills | A/B |
| D10 | dtypes, compat precision, no numba, the near-one snap (round 2) | V0 `test_hrosm_package_imports_no_numba`; V1 `test_degrees_output_rounds_through_float32_radians`; V3 dtype arms, `test_identical_neighbours_give_zero_not_nan` (S16, S18); V16 dtypes | A/B |
| D11 | no new dependency, BSD block, nothing LGPL ported | V0 licence and LGPL arms; oldest-matrix run; `pyproject.toml` diff shows no new dependency | A |
| D12 | private reader | V14 `TestEMsoftFileReader`; every V2/V10/V12 local arm | A |
| D13 | gates, script, shipped files, binary choice, historical files, regeneration | V14 all; V2, V7, V10, V12 shipped and local arms; ledger records the binary and md5s | A |
| D14 | branch, PR #20, merge rule, fan-out, clean-replay grep, standing constraints | Definition of done; clean-replay grep; `gh pr view` in the ledger | C |
| D15 | CI budget, placement (weekly arms local while the fork's Weekly workflow is disabled, round 2), oldest matrix, flake rule | "CI budget" table measured per stage; the local `--weekly` gate's seconds and scaled estimate; oldest-matrix runs; `-n 0` then `-n 4` | A-C |
| D16 | tutorial, registration, gallery example, docstrings, CHANGELOG, API reference | Manual; nbval on `hrosm.ipynb`; `sphinx-build -b html` exit 0 | C |
| D17 | measurement policy | this document; ledger entries per gate | A-C |
| D18 | model rule, workflow per stage, approval gate | Definition of done; ledger records any escalation | A-C |
| D19 | R1-R7 | R1 V16 `TestOutput`; R2 V16 `test_default_average_is_mean`, V12/V13 pass `"center"`; R3 V12, V16 fills; R4 V16 `test_multi_phase_uses_the_master_by_phase_name`; R5 V8 left/right arms, V9 `test_compat_model_is_right_sided`; R6 V4, V5 leakage; R7 V11 `TestHROSMKeywords`, V16 verbose | A/B |
| D20 | ball spacing (public), GROD to the ball centre (public map, one private helper), pre-run coverage warning (one helper, two call sites, before any simulation, re-indexed grains only, strict) | V7 `TestBallSpacing` (all); V8 `TestGROD` (`test_max_grod_is_the_largest_angle_to_the_grain_reference`, `test_warning_lists_the_largest_ten_in_descending_max_grod`, `test_warning_is_strict_at_max_angle`, `test_correct_center_pixel_has_zero_grod`, `test_compat_center_grod_is_measured_from_the_box_centre_pixel`, `test_variant_scrambled_pixels_have_the_same_grod`, `test_deviation_map_and_max_grod_agree_bitwise`); V16 `test_grod_props_equal_the_public_deviation_map`, the four coverage-warning arms of `TestMessages`, `test_verbose_levels`; mutants S9-S14 | A/B |

### Mutant killers (`plan.md` section 6 list; each applied alone in the main tree; numbering and areas of the parked plan)

Every killer below is a DEFAULT-suite test (no `--weekly`, no
environment variable). No mutant is reachable only from a gated
path. Short names: `kam` = `tests/test_indexing/test_hrosm_kam.py`,
`seg` = `..._segmentation.py`, `avg` = `..._averaging.py`, `smp` =
`..._sampling.py`, `osm` = `..._osm.py`, `reg` =
`..._emsoft_regression.py`, `sig` = `tests/test_signals/
test_ebsd_hrosm.py`.

| M | mutant | default-suite killer(s) | why it dies |
|---|---|---|---|
| M1 | compat KAM: vertical pair credited to `t - W` (the correct pixel) | `kam::TestCompatKAM::test_vertical_pair_is_credited_one_column_right`; `kam::TestCompatKAM::test_matches_the_loop_transcription_bitwise`; `reg::TestCompatKAMOnEMsoftFiles::test_shipped_di_kam_on_nondegenerate_pixels` | the changed-pixel set becomes `(1, 1)` instead of `(1, 2)` |
| M2 | compat KAM: `jj` computed without the off-by-one | `kam::TestCompatKAM::test_first_row_last_column_is_compared_with_the_identity`; transcription arms with `W >= 2` | the identity comparison at `(W, 1)` disappears, so `(0, 0)` and `(0, W - 1)` no longer depend on `euler0` |
| M3 | compat KAM: corner `(1, 1)` multiplier `third` instead of `* 4` | `kam::TestCompatKAM::test_constant_pair_angle_field`; transcription arms | `(0, 0)` becomes `(phi + s) / 3` |
| M4 | compat KAM: the spurious angle taken against `q[0, 0]` instead of the identity | `kam::TestCompatKAM::test_constant_pair_angle_field`, `test_first_row_last_column_is_compared_with_the_identity` | `(0, 0)` gets `phi + (W - 1) phi`, independent of `euler0` |
| M5 | compat KAM: degrees as `kam32 * np.float32(rtod)` (float32 product) | `kam::TestCompatKAM::test_degrees_output_rounds_through_float32_radians`; transcription arms; shipped DI KAM | the rounding differs on ~14 % of values |
| M6 | correct KAM: fixed divisor 4 | `kam::TestCorrectKAM::test_two_axis_gradient_field_is_exact` | edges need 3, corners 2 |
| M7 | correct KAM: angle not symmetry-reduced | `kam::TestCorrectKAM::test_matches_orix_angle_with_on_scrambled_variants` | scrambled neighbours are up to 180 deg apart unreduced |
| M8 | growth: 4-connectivity | `seg::TestSegmentationRule::test_diagonal_neighbours_join`; `test_matches_the_flood_fill_transcription` | the diagonal pair becomes two singletons |
| M9 | growth: edge test `(a <= t) & (b <= t)` instead of `abs(a - b) <= t` | `seg::TestSegmentationRule::test_criterion_is_chained_on_kam_differences`; transcription arms | the grain stops after `[1, 5]` |
| M10 | growth: seed gate removed | `seg::TestSegmentationRule::test_component_without_a_pixel_at_or_below_threshold_is_unassigned`; transcription arms | `[6, 7, 8]` becomes a grain |
| M11 | growth: singleton kept | `seg::TestSegmentationRule::test_singletons_are_unassigned`; transcription arms | the isolated pixel gets a label |
| M12 | growth: comparison in float64 | `seg::TestSegmentationRule::test_threshold_tie_is_decided_in_float32` | 5.000000229 > 5 splits the pair |
| M13 | compat dilate: touches row/column 0 or does not overwrite | `seg::TestDilateAndBoxes::test_compat_dilate_skips_the_first_row_and_column_and_overwrites`; `test_compat_dilate_matches_the_emsoft_window_transcription` | row 0 changes, or label 2 keeps its pixel |
| M14 | correct dilate overwrites labelled pixels | `seg::TestDilateAndBoxes::test_correct_dilate_fills_only_unassigned_pixels` | an assigned pixel changes label |
| M15 | compat centre `x0 + (w - 1) // 2` | `avg::TestCenterPixel::test_compat_center_is_the_box_centre_rounded_up`; `reg::TestClusterStage::test_center_average_is_the_box_centre_pixel` | even width picks column 1, not 2 |
| M16 | ball count `(2N)**3` or `N` points per half edge mis-set | `smp::TestMisorientationBall::test_count_order_and_shells`; `smp::TestEMsampleRFZ::test_shipped_n6_ball_matches_in_order` | length and shells differ |
| M17 | composition side swapped (`center * ~ball` or `ball * center`) | `smp::TestMisorientationBall::test_composition_is_conj_ball_times_center`; `smp::TestEMsampleRFZ::test_shipped_n6_ball_matches_in_order` | the in-order element comparison fails by > 1e-3 rad |
| M18 | edge from the small-angle form `(pi w**3 / 6)**(1/3) / 2` | `smp::TestMisorientationBall::test_outer_shell_is_exactly_max_angle`; shipped N 6 ball | the outer shell misses 5 deg by `w**3 / 60` rad = ~6e-4 deg |
| M19 | correct EM: symmetry side mixed (right side in the E or M step) | `avg::TestRecovery::test_recovers_left_scrambled_variants[vmf-alpha100-n300]`, `[watson-alpha100-n300]` | left-scrambled samples fall into right-side centres |
| M20 | correct-mode Watson (`_em_correct`): linear term `t` instead of `t**2` | `avg::TestRecovery::test_watson_is_antipodally_symmetric`; `avg::TestRecovery::test_recovers_left_scrambled_variants[watson-alpha100-n300]` | sign-flipped samples cannot be fitted: about half of the samples carry the opposite sign after `q0 >= 0`, so their true variant gets a negative linear term and no responsibility (corrected 2026-10-06, spec review round 2: the compat transcription arm exercises `_em_emsoft`, not the mutated `_em_correct`) |
| M21 | kappa gate `>=` instead of `>` (in `_apply_kappa_gate`) | `avg::TestCompatEM::test_kappa_gate_is_strict` | `min_kappa == kappa_hat` keeps the grain |
| M22 | compat: stale `Q`/`L` branch removed | `avg::TestCompatEM::test_watson_underflow_keeps_the_previous_q_and_exits_at_the_second_iteration` | an underflowing iteration gives `Q` NaN or `-inf`, so the init whose iteration-2 `Phi` underflows no longer exits at `i = 2` (its iteration count differs from the transcription's 2; arm rewritten 2026-10-06, spec review round 2) |
| M23 | `mean` without variant alignment | `avg::TestRecovery::test_recovers_left_scrambled_variants[mean-alpha100-n300]` | the Markley mean of scrambled variants is ~49 deg off |
| M24 | compat OSM divided by `n` | `osm::TestCompatOSM::test_compat_osm_is_not_divided_by_n`; transcription arms; `reg::TestCompatOSMOnEMsoftFiles::test_shipped_osm_matches` | 1.0 instead of 10.0 |
| M25 | compat OSM edge multiplier `x * float32(4 / 3)` | `osm::TestCompatOSM::test_edge_multiplier_follows_the_source_order` | 1.6666667 instead of 1.6666666 |
| M26 | correct OSM counts cross-grain neighbours | `osm::TestGrainAwareOSM::test_cross_grain_neighbours_are_excluded` | column 3 drops below 10 |
| M27 | `min_pixels` boundary off by one (`<=` skips) | `sig::TestContracts::test_min_pixels_boundary_is_inclusive` | grain A with exactly `min_pixels` is skipped |
| M28 | navigation mask polarity flipped or masked props leak | `sig::TestContracts::test_navigation_mask_true_excludes_and_fills` | masked pixels get re-indexed or non-NaN props |
| M29 | the reader applies `np.deg2rad` to `EulerAngles`, `RefinedEulerAngles`, `newEuler` | `reg::TestEMsoftFileReader::test_euler_datasets_are_returned_in_radians` (the shipped `.npz` files do not pass through the reader; the local V2/V12 arms reinforce) | the returned arrays are the stored ones divided by 57.3 |
| M30 | ball not re-centred per grain (`ball_g = ball_identity`) | `sig::TestContracts::test_each_grain_is_matched_against_its_own_ball`; `sig::TestSubgrainContrast::test_subgrain_step_is_recovered` | every grain's rotations lie near the identity, not its average |

Supplementary mutants (`plan.md` section 6, S1-S20):

| S | mutant | default-suite killer(s) | why it dies |
|---|---|---|---|
| S1 | correct final representative `mu * S_i` instead of `S_i * mu` | `avg::TestCompatEM::test_final_representative_side`; `avg::TestRecovery::test_recovers_left_scrambled_variants[vmf-alpha100-n300]` | the representative is not left-equivalent to `mu` |
| S2 | compat domain = the grain's pixels (box dropped) | `sig::TestContracts::test_compat_domain_is_the_bounding_box` | block size `n_pixels` instead of `W * H` |
| S3 | compat fill = NaN and the input rotation | `sig::TestOutput::test_properties_dtypes_shapes_and_fills[compat]` | `osm` NaN instead of 0.0, rotation not the identity |
| S4 | compat ball without the float32 Rodrigues round trip | `smp::TestMisorientationBall::test_compat_storage_is_float32_rodrigues` | compat equals correct bitwise |
| S5 | `_dictionary_indexing(verbose=False)` still prints | `test_dictionary_indexing.py::TestDictionaryIndexing::test_verbose_false_silences_the_core_for_hrosm` | `capsys` not empty |
| S6 | GROD warning `>=` | `avg::TestGROD::test_warning_is_strict_at_max_angle` | warns at `max_angle == max_grod` |
| S7 | default keywords routed to the grain-aware map | `test_orientation_similarity_map.py::TestHROSMKeywords::test_defaults_run_the_legacy_path_unchanged` | the custom footprint / `center_index` and `from_n_best` arms raise "cannot be combined" instead of running the legacy path (amended 2026-10-06: the `(1, 6)` arm was dropped, the legacy function raises on it) |
| S8 | `default_rng(seed)` re-created per grain | `avg::TestRecovery::test_one_generator_is_consumed_across_grains_in_label_order` | grain 2 repeats grain 1's draws |
| S9 | spacing `2 * arcsin(d / 2)` instead of `4 * arcsin(d / 2)` | `smp::TestBallSpacing::test_spacing_equals_brute_force_nearest_neighbour`; `test_default_spacing_pin` | the value halves |
| S10 | spacing from `query(q, k=1)` (self distance) | `smp::TestBallSpacing::test_spacing_equals_brute_force_nearest_neighbour`; `test_spacing_decreases_with_n_steps_and_grows_with_max_angle` | spacing 0.0 everywhere |
| S11 | GROD without symmetry reduction | `avg::TestGROD::test_variant_scrambled_pixels_have_the_same_grod`; `test_max_grod_is_the_largest_angle_to_the_grain_reference` | scrambled pixels get angles up to 180 deg |
| S12 | compat `"center"` GROD measured from the centroid-nearest pixel | `avg::TestGROD::test_compat_center_grod_is_measured_from_the_box_centre_pixel` | the ring grain's map differs by > 0.1 deg |
| S13 | coverage warning issued after the first grain's simulation | `sig::TestMessages::test_coverage_warning_precedes_the_first_simulation` | the spy's first call sees no warning recorded |
| S14 | driver coverage warning `>=` instead of `>` | `sig::TestMessages::test_coverage_warning_is_silent_at_equality` | warns at `max_angle == max_grod` |
| S15 | compat `arccos` clipped to [0, 1] | `kam::TestCompatKAM::test_matches_the_loop_transcription_on_duplicated_orientations` | the guaranteed non-zero duplicated pair becomes 0 |
| S16 | correct-mode near-one snap dropped (redefined 2026-10-06, spec review round 2) | `kam::TestCorrectKAM::test_identical_neighbours_give_zero_not_nan` | NaN on the rotation whose self-dot rounds above 1 |
| S17 | compat K11 checks removed (absent points, several phases) | `kam::TestCompatKAM::test_rejects_absent_points_and_several_phases`; (B) `sig::TestValidation::test_arguments_are_validated_in_order` | no `ValueError` |
| S18 | correct-mode snap replaced by the clip to [0, 1] | `kam::TestCorrectKAM::test_identical_neighbours_give_zero_not_nan`; `avg::TestGROD::test_correct_center_pixel_has_zero_grod` | the rotation whose self-dot rounds below 1 gives ~3e-6 deg, not 0.0 |
| S19 | `_map_grid` from the in-data `row`/`col` (the drafted rule) | `avg::TestGrainTable::test_map_grid_spans_points_not_in_the_data`; (B) `sig::TestContracts::test_navigation_masked_dictionary_indexing_map_keeps_its_grid` | the absent-row-0 map shrinks to (2, 4); `hrosm` raises "xmap shape" |
| S20 | `_dictionary_indexing(verbose=False)` keeps `sleep(0.2)` | `test_dictionary_indexing.py::TestDictionaryIndexing::test_verbose_false_silences_the_core_for_hrosm` | the `sleep` spy is called |

Stage A mutants M1-M26 (M26 at the array function), S1, S4, S6, S8,
S9-S12, S15-S19; Stage B M27-M30, S2, S3, S5, S7, S13, S14, S20 (rows
S9-S17 added 2026-10-06, spec review; S18-S20 in round 2, S16
redefined).
The bug-injection gate records, per row, the killer arm that fired
(`-n 4 -x`, then the named arms alone without `-x`); a survivor gets a
strengthened test verified by re-injection.

## CI budget

Two numbers per stage gate (D15.1 as amended 2026-10-06, spec review,
with the measured CI job times of 17 min 21 s / 20 min on `develop`):
(1) the BINDING CI-style wall time `uv run --no-sync --with pytest-cov
pytest <the HROSM modules> -n 4 -q -p no:cacheprovider
--cov=kikuchipy --cov-branch --cov-report=` <= 25 s (MTP; NLPAR's
trimmed equivalent measured 22.0-25.8 s), median of three runs; (2)
serial seconds of the DEFAULT suite (`uv run --no-sync pytest <file>
-n 0 -q -p no:cacheprovider --durations=0`, warm numba caches, this
machine) <= 60 s (A5 ceiling), with the summed worker-seconds. Seeds
are estimates.

| file | stage | default arms (heavy arms moved out) | seed s |
|---|---|---|---|
| `test_hrosm_kam.py` | A | all of V1, V3 | 3 |
| `test_hrosm_segmentation.py` | A | all of V4-V6 (dilate part) | 2 |
| `test_hrosm_averaging.py` | A | V8 default arms (8 recovery runs), V9 at `n_em=3`, `TestCenterPixel`, `TestGROD`, `TestGrainTable`; full grid and `n_em=25` weekly | 8 |
| `test_hrosm_sampling.py` | A | V7 except the bin arm, `TestBallSpacing` (one N 20 grid) | 3.5 |
| `test_hrosm_osm.py` | A | V10, V11 array arms | 2 |
| `test_hrosm_emsoft_regression.py` | A | V0, V2/V10/V12 shipped arms, V14 CI arms and the lock test (~4 s); V13 weekly, local, bin and weekly arms skip | 4 |
| `tests/test_signals/test_ebsd_hrosm.py` | B | V15 small arm (~8 s, fixture once), V16 (~16 s, cached runs; the coverage-warning arms stop at the first `get_patterns`, ~1 s; the navigation-masked `dictionary_indexing` arm ~2 s, round 2); full-size V15 weekly | 25 |
| `test_orientation_similarity_map.py` (append) | B | `TestHROSMKeywords` | 1 |
| `test_dictionary_indexing.py` (append) | B | one test | 0.5 |
| **total** | | | **49** |

If the CI-style time exceeds 25 s (or the serial total 60 s, or an
on-push CI job 18 min), arms move to `@pytest.mark.weekly` in this
order, each keeping every mutant row killed by the default suite
(re-verified by re-injection): the V8 `alpha1000-n30` arms, then V15
at `(8, 12)` with `n_steps=6`, then the V16 `n_per_iteration` and
lazy arms reduced to one case each.

## Performance (recorded baselines, never gates -- D8.13, D17)

| measurement | recipe | seed / recorded value |
|---|---|---|
| compat KAM, Ni6 map (28,086 px) | NumPy, `TestCompatKAMOnEMsoftFiles` local arm, `record_property` | 0.29 s measured 2026-10-06 (requirements D3.8 probe) |
| compat KAM, Al map (501,592 px) | local + weekly arm | ~5 s estimate for the vectorised implementation; an unoptimised float64 transcription took 96.8 s (spec-review probe 2026-10-06) |
| segmentation (csgraph), Al map | weekly local, timed in the KAM arm | unmeasured |
| ball N 20 (68,921 points) | `test_count_order_and_shells[20]` | unmeasured |
| Watson EM, one grain of 2,000 px, 25 x 40 | `TestRecovery` weekly arm scaled | ~1 s estimate (parked design) |
| `EBSD.hrosm` on `nickel_ebsd_large` (4,125 px, 60 x 60), N 20 | local ledger run (and the V13 full-map local + weekly arm) | NCC 2.0e12 flops (~40 s) + simulation ~3 s per grain (one 68,921-pattern `get_patterns` measured 1.9 s here) |
| `ni_gain` (29,800 px) | local ledger only | 1.5e13 flops, ~5 min estimate |
| `si_wafer` binned 4 x, `n_steps=10` | local ledger entry (Local-gated section) | unmeasured; unbinned 7.9e13 flops (hours) |
| V14 regeneration | bin arm | ~10-25 min estimate (D13.6; `plan.md` 7.3 item 23) |
| default-suite addition (serial) | "CI budget" recipe | ceiling <= 60 s; seed 49 s (47 before round 2) |
| CI-style HROSM run (`-n 4`, coverage) | "CI budget" CI-style recipe | gate <= 25 s (MTP); unmeasured |
| on-push CI job times | `gh run view <id> --json jobs` per stage push | each <= 18 min (`develop` de27741a: 17 min 21 s max) |
| weekly additions (round 2: no CI runner, the fork's Weekly workflow is disabled) | local `--weekly` gate with `--durations=0`, serial seconds; scaled CI estimate = local x (last on-push ubuntu py3.13 job time / local full-suite `-n 4` time) | target <= 5 min estimated; unmeasured |

## Manual

- Johan's approval of `plan.md` (with the seven defaults of D19) before
  any test or code (A9).
- Reference generation run (Stage A, this machine): the
  `create_hrosm_reference.py` run directory, binary (D13.4) and its
  seven md5s, `emsoft_version`/`emsoft_commit`, GPU name, the acid
  band values, the `OSM_10 == OSM` self-check and the file sizes
  recorded in the ledger; EMsoft programs never concurrent; run
  directory under `kikuchipy_hrosm/` only; the historical files and
  the EMsoftOO checkout untouched.
- Tutorial `doc/tutorials/hrosm.ipynb` (Stage C) renders and passes
  nbval locally (`run_nbval.sh` entry between `hough_indexing` and
  `hybrid_indexing`; sanitize regexes only as needed); stored outputs
  if the estimated Read the Docs run exceeds ~2 min (MTP). Review items,
  each ticked by eye with the figure named:
  - global OSM (from `pattern_matching.ipynb`'s dictionary route plus
    NCC refinement on `nickel_ebsd_large`) next to the HROSM OSM, same
    colour scale 0..10;
  - KAM in both modes, the grain map, the GROD map and the GROD
    warning text;
  - the synthetic sub-grain demonstration (V15 setup), global vs
    HROSM;
  - the parameter-guidance paragraph and cost table (`max_angle`,
    `n_steps`, `(2N + 1)**3` scaling) with ledger numbers;
  - "Differences from EMsoftOO's EMHROSM" in words (no K-numbers),
    matching the `EBSD.hrosm` Notes;
  - the `:cite:` keys render; links to `pattern_matching.ipynb` and
    `spherical_indexing.ipynb` (never edited).
- Gallery example `examples/indexing/hrosm.py` (+ `README.rst`) runs
  under sphinx-gallery in <= 30 s (MTP) and shows one OSM pair.
- `sphinx-build -b html` exits 0; the API reference lists the eight
  names and `EBSD.hrosm`.
- One end-user smoke run from a notebook: `s.hrosm(...)` on
  `nickel_ebsd_large` with `verbose=1`, eyeballing the progress bar,
  the warning and that the HROSM OSM shows sub-grain contrast.

## Definition of done

- Per stage: failing tests committed first (skeletons raise
  `NotImplementedError`, MTP placeholders `None`), never pushed alone;
  implementation; adversarial review (fidelity against
  `mod_DIsupport.f90`, `mod_cluster.f90`, `mod_so3.f90`,
  `mod_dirstats.f90`, `mod_HROSM.f90`, `mod_DI.f90`; conventions and
  integration) and bug injection of the stage's mutants applied ALONE
  in the main tree, each dying by its named default-suite test,
  survivors strengthened and re-injected; disposition tables appended
  to `plan.md`; Workflow agents per D18 (Fable only as a logged
  escalation).
- Gates per stage: `-n 0` then `-n 4` (red re-run alone); coverage
  100 % of `src/kikuchipy/indexing/_hrosm/` (Stage A: its modules)
  from the default suite only, no `--weekly` and no gate variable
  (unified with `plan.md` section 5, 2026-10-06, spec review:
  `COVERAGE_FILE="$SCRATCH/.coverage.hrosm" uv run --no-sync coverage
  run -m pytest $B_TESTS -n 0 -q -p no:cacheprovider`, then
  `COVERAGE_FILE="$SCRATCH/.coverage.hrosm" uv run --no-sync coverage
  report -m --include="src/kikuchipy/indexing/_hrosm/*"`; the
  reference script is omitted by the existing `pyproject.toml`
  `omit`); full existing suite green with counts; oldest-matrix run
  recorded; local and bin arms run once and recorded; the CI budget
  measured (CI-style <= 25 s, serial <= 60 s, on-push CI jobs <= 18
  min);
  `SKIP=licenseheaders uvx pre-commit run --files <explicit non-specs
  files>` clean; clean-replay grep empty; signed commits with the
  trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`,
  explicit pathspecs, pushed to `origin/feat-HROSM`; roadmap boxes
  ticked with ledger evidence.
- Every MTP placeholder replaced by a dated measured value (recipe and
  machine); every seed confirmed or amended in `requirements.md` with
  the date.
- Stage A: V0-V12 and V14 green (V0 over the seven Stage A modules,
  `_driver` added by the Stage B failing-tests commit; V11 array arms;
  V13 not yet), shipped references and registry entries committed, the
  reader and the script in place, CHANGELOG API bullet with the fork
  PR link.
- Stage B: V11 routing arms, V13 (one-grain weekly, full map local +
  weekly; both run once locally), V15, V16 green; performance table filled; the `si_wafer` ledger entry
  written or "not cached".
- Stage C: tutorial, gallery example, CHANGELOG tutorial bullet;
  nbval green; `sphinx-build -b html` exit 0; the Manual list ticked;
  the three spec documents re-submitted to adversarial review with
  the fixer's disposition table in `plan.md`.
- `git log origin/feat-HROSM..feat-HROSM` empty; PR `feat-HROSM ->
  develop` open (`gh pr view` recorded, number confirmed, CHANGELOG
  link rewritten if not #20) with ubuntu and windows CI green and the
  macOS failure identified as the known `test_ni_proper_oh_count` (23
  == 22); merge awaits Johan.
- `git diff --name-only develop...HEAD` never lists
  `doc/tutorials/{hybrid_indexing,spherical_indexing,load_save_data,
  pattern_matching,nlpar}.ipynb`, `specs/_research/
  plan-upstream-merge-0.13.1.md` or `AGH__Si_indent_1_512x672.h5oina`.
- After the merge (outside this definition, D14.4 as amended
  2026-10-06): `hrebsd-dic` merge with the gates of `plan.md` 1.2 (no
  conflict markers; `$B_TESTS` at `-n 0` then `-n 4`; `pytest tests -k
  hrebsd -n 4`; the full suite `-n 4`; nbval on `hrosm.ipynb`;
  pre-commit on the resolved files); the clean replay onto
  `feat-spherical-indexing-hrosm` with the equivalence gate and the
  worktree tests (`PYTHONPATH=<worktree>/src`, `kikuchipy.__file__`
  guard); memory notes updated.

## Recorded results

Append-only ledger. Every entry is dated, names the machine (CPU class
and OS at minimum), the exact recipe, and which MTP placeholder or
decision it fills. Refutations of frozen decisions are recorded here
AND amended in `requirements.md` with the same date. An escalation to
Fable (D18.3) is logged with its reason.

### 1. 2026-10-06 (drafting)

Validation drafter, Opus 5.5; Windows 11 Enterprise 10.0.26200, the
20-core workstation of the NLPAR ledger, `.venv` (orix 0.14.2, numpy
2.4.6, scipy 1.17.1, numba 0.65.1, as stated by the task; not
re-printed). Read-only: no test executed, nothing written outside
this file. Inputs read: the parked plan, the five exploration
reports, `requirements.md` (1,396 lines), the NLPAR validation
(template), `conftest.py`, `tests.yml`, `weekly.yml`,
`test_spherical_emsphinx_regression.py`,
`test_orientation_similarity_map.py`, `test_dictionary_indexing.py`,
`_orientation_similarity_map.py`.

1. **Drafted**: V0-V16 with test files, classes, gates, oracles,
   assertions and fixtures; the root-conftest generators; the MTP
   inventory; the mutant-killer table (M1-M30 and S1-S8, every row
   with a default-suite killer); the CI budget (seed 54.5 s of 60 s);
   the requirement map D1-D19.
2. **orix `random_vonmises` probe** (orix 0.14.2 source and timings,
   `np.random.seed(0)`): density `exp(2 alpha cos w) / hyp0f1(1.5,
   alpha**2)` with `w = angle_with(reference)`, rejection sampling
   from `Rotation.random(int(alpha) * n)`, global `np.random.rand`.
   Times: `alpha 20, N 30` 0.02 s; `20, 300` 0.12 s; `100, 30` 0.11 s;
   `100, 300` 1.42 s; `1000, 30` did not finish in 88 s. rms rotation
   angle: 14.8-15.2 deg (alpha 20), 6.6-7.0 deg (alpha 100). Hence the
   test-local sampler of V8 and the kappa mapping VMF `8 alpha`,
   Watson `4 alpha`; orix kept as one cross-check arm.
3. **float32 constants** (`uv run --no-sync python`): for `x = 0.25
   k`, `(x * 4) / 3` and `x * float32(4 / 3)` differ at `k` = 5, 7,
   10, 14, 17, 20, 23, 25, 28, ... (`k = 5`: 1.6666666 vs 1.6666667);
   `float32(0.01)` and `float32(5.01)` differ by exactly 5.0 in
   float32 and by 5.000000229105353 in float64; on 100,000 uniform [0,
   0.2] rad sums (`default_rng(0)`) `float32(float64(float32(acc)) *
   rtod)` differs from `float32(acc * rtod)` in 25.4 % and from the
   float32-`rtod` product in 14.0 %.
4. **Shipped-file arithmetic**: see V14 (DI file 280.5 kB as D13.3
   lists it; ~840 kB in total after the trimming defaults of
   `plan.md` 7.2 item 10, which V14's key table applies).
4a. **Aligned with `plan.md`** (written in parallel, read at the end of
   drafting): its mutant code changes (section 6, M1-M30) and the
   supplementary S1-S8 got named default-suite killers here; item 16
   (the `from_euler == eq_` input route) got its own V1 arm. Two
   placements differ: `plan.md`'s Stage A test list puts the V2, V10
   and V12 file arms in the unit modules, while this document keeps
   every EMsoft-file arm in `test_hrosm_emsoft_regression.py` (one
   loader and one degenerate-pixel helper, no cross-module imports);
   `plan.md` 7.3 item 20 (`-k hrosm` misses the two appended modules)
   is met here by naming (`TestHROSMKeywords`,
   `..._for_hrosm`; `-k` is case-insensitive).
5. **Points for the requirements fixer** (not edited here):
   (a) D5 Pins and D15.4 cite `Rotation.random_vonmises` with "kappa
   20/100/1000"; those are orix `alpha` values (item 2), and the
   sampler is impractical at 1000. (b) D13.3 file list vs its 250 kB
   per-file and ~800 kB total budget (V14). (c) V13's CI arm needs a
   private driver entry point with a grain subset; D8 does not name
   one. (d) V9 needs a private EM diagnostics record (per-init `L`,
   iteration counts, chosen init); D5 does not name one. (e) D3.6
   says "bitwise on non-degenerate pixels", while its own Ni6 HROSM
   probe has 6 of 27,992 non-degenerate pixels at 1-2 ulp: V2 pins
   that count and the ulp bound rather than zero. (f) D7.3 + D13.4:
   whether the recommended `c127868` build folds `4/3` is unknown, so
   the shipped-OSM pin is MTP (V10). (g) D1.5's "point group whose
   EMsoft operator set differs from orix's" names no example; the V16
   case is chosen at Stage A from the operator-table comparison.
   (h) V8 needs `average_grain_orientations(seed=...)` to accept a
   `Generator` (as `EBSD.hrosm`'s `seed` does); D1.3 leaves the free
   function's `seed` type open.
6. **Remaining MTP**: every placeholder of the inventory; the `eu`
   unit of the EMsampleRFZ text; `GPU_ARRAY_POLICY`; the per-file CI
   seconds; the oldest-matrix outcome of the orix-io round trip.

### 2. 2026-10-06 (spec review round 1, fixer)

Fixer, Opus 5.5; Windows 11 Enterprise 10.0.26200, the 20-core
workstation of entry 1, `.venv` (orix 0.14.2, numpy 2.4.6, scipy
1.17.1), Git Bash. Inputs: the 46 findings of three critics (F1-F16,
C1-C16, E1-E14). Each was re-verified before editing; dispositions in
`plan.md` section 10. No test or code written; only the three spec
documents, the HROSM blocks of `specs/{mission,roadmap,tech-stack}.md`
(still byte-equal to `plan.md` section 0, no BOM) changed.

1. **Probes re-run (read-only, session scratchpad):**
   (a) legacy `orientation_similarity_map` on `create_coordinate_arrays`
   maps: (1, 6) -> `xmap.shape` (6,), `RuntimeError` "footprint.ndim
   (2) must match len(axes) (1)"; (6, 1) the same; (1, 1) -> shape (),
   `xmap.row` raises "not enough values to unpack"; (2, 5) -> (2, 5).
   (b) Ni6 `dp-Ni6-refined.h5`: `Rotation.from_euler(eu32.astype(
   float64)).data != eq_(eu32)` on 8,930 of 28,086 rows
   (`RefinedEulerAngles`) and 8,404 (`EulerAngles`), max 2.2e-16; the
   round trip `from_euler(...).to_euler().astype(float32) == eu32` on
   28,086/28,086 rows of both, GRX810 139,181/139,181, Al
   501,592/501,592, and `eq_` of the round trip equals `eq_(eu32)` on
   every row. (c) Al `dp-full.h5` `EulerAngles` -> `KAM` with the
   critic's float64 literal transcription (96.8 s): non-degenerate 16
   of 210,294 differ (max 2 ulp), degenerate 9,838 of 291,298 (max
   90.0 deg). (d) orix 0.12.1 + numpy 1.23.0 (isolated py3.10, scipy
   1.13.1): `Rotation.angle_with_outer`, `from_homochoric`, `cu2ho`,
   `scipy.spatial.cKDTree` present; `orix.io` save/load keeps bool
   (12,), float32 (12, 5), float64 (12, 4), int32 (12,) props. (e) The
   cached master `ni_mc_mp_20kv.h5`: both `xtalname` datasets
   `ni/ni.xtal` (variable-length ASCII, shape (1,)), version
   `4_3_0_0`; `EMsoftData/Xtal` holds `Ni.xtal` (SG 225, a = 0.35236
   nm) and no subdirectory; `EMsoftConfig.json` `EMXtalFolderpathname`
   = `.../EMsoftData/Xtal`. (f) `0.001 * i != 0.001 + (i - 1) * 0.001`
   for 11,504 of 35,000 `i`. (g) CI run 37508408494 (`develop`
   de27741a): ubuntu py3.10 oldest 17 min 21 s, windows py3.13 16 min
   55 s, ubuntu py3.13 14 min 48 s, windows py3.14 11 min 20 s, wheel
   10 min 15 s, macOS red as known.
2. **Source facts re-read:** `mod_OSM.f90:177-179, 479-491` and
   `mod_HROSM.f90:768-779` (TIFFs written unconditionally under
   `EMdatapathname`); `mod_FitOrientation.f90:263-277` (required
   keys); `mod_dirstats.f90:243-261` (the kappa table; not
   `mod_cluster.f90`); `mod_so3.f90:2588-2590` (EMsampleRFZ `eu` in
   degrees); `mod_crystallography.f90:2652-2656` (crystal path);
   `ebsd.py:2505-2522` (mask checks); `conftest.py:680-750` (EMSphInx
   lock, 600/900 s, mtime never refreshed); `_ebsd_detector.py:
   1704-1705` (`pc_emsoft` 2-D); `create_emsphinx_reference.py:311-312`
   (default output = package dir); `_dictionary_indexing.py:97-128`
   (int64 on both paths); `hrebsd-dic` `_hrebsd/_segmentation.py:
   283-312` (`_map_grid`) and `_reference.py:204-209` (labels -1).
3. **Changes here:** D20 carried into every section (eight public
   names; V7 `TestBallSpacing`; V8 `TestGROD` rewritten to the one
   warning contract plus the deviation-map arms; V16 coverage-warning
   arms with a `get_patterns` spy that stops the run; D20 mapping row;
   S9-S14); K2/K11 killers (V1 duplicated orientations and point-group
   arms, V3 identical neighbours; S15-S17); V0 BSD test joins wrapped
   comment lines and is staged (`_driver` in Stage B), new docstring
   test; V1 Euler round-trip pin replaces the refuted `from_euler ==
   eq_` pin; V2 Al seed 16 (<= 2 ulp) and the `EulerAngles` file fixed;
   V9 citation, `EMResult`, `_apply_kappa_gate`, `xAp` order, `n_em=3`
   default arms; V11 legacy arm on (2, 5); V13 pins split and moved
   (one grain weekly, full map local + weekly); V14 layout fixed (~841
   kB seed), `master_run_md5`, the PC route, `main(output_dir=tmp_path,
   ...)`, `TestEMsoftProgramLock`; the CI budget restated (CI-style <=
   25 s binding, serial <= 60 s ceiling, job times <= 18 min; seed 47
   s); coverage command unified (default suite only).
4. **Remaining MTP** (unchanged apart from these): `eu` unit settled
   (degrees); the orix-io round trip settled (works on 0.12.1); new
   placeholders `BALL_SPACING_DEFAULT_DEG`, `BALL_SPACING_N10_DEG`,
   `BALL_SPACING_N2_DEG`, `AL_KAM_MAX_ULP`, the six split `E2E_*`
   names; `REFERENCE_TOTAL_BYTES` seed 841,000; the CI-style time.

### 3. 2026-10-06 (spec review round 2, fixer)

Fixer, Opus 5.5; Windows 11 Enterprise 10.0.26200, the workstation of
entry 1, `.venv` (orix 0.14.2, numpy 2.4.6), an isolated py3.10 env
(orix 0.12.1, numpy 1.23.0), Git Bash, `gh`. Inputs: the 13 findings
of one critic (R2-1 to R2-13). Each was re-verified before editing;
dispositions in `plan.md` section 10 (round 2): 13 fixed, 0 rejected.
No test or code written; only the three spec documents and the HROSM
block of `specs/tech-stack.md` (two bullets, still byte-equal to
`plan.md` 0.3, no BOM) changed; `mission.md` and `roadmap.md`
unchanged.

1. **Probes (read-only, session scratchpad):**
   (a) orix grid: `create_coordinate_arrays((3, 4))` maps with
   `is_in_data` False on row 0 -> `shape (2, 4)`, `size 8`, `row` `[0 0
   0 0 1 1 1 1]`; on the last column -> `(3, 3)`; all but flat 5 ->
   `(1, 1)`, `size 1`; `xmap[1:3, 1:3]` -> `shape (2, 2)`,
   `_original_shape (3, 4)`, `is_in_data.size` 12. The D1.9 rule
   (`_original_shape`, the one-row/one-column rule, -1 off the data)
   gives (3, 4) for all three, (1, 6), (6, 1), (1, 1), (2, 5) for the
   full maps, (1, 5) and (5, 1) with an absent end point, and the (3,
   4) grid with 0..3 at `[1:3, 1:3]` for the slice, identically on
   orix 0.12.1/numpy 1.23.0 and orix 0.14.2/numpy 2.4.6.
   (b) Near-one dot: 10,000 random float32-Euler rotations (Oh proper
   operators, float64): self-dot `< 1` on 4.68 %, max angle
   3.4150946e-06 deg after `float32(float32(rad) * rtod)`; orix
   `angle_with` max 0.0; the 4-eps snap gives 0.0. 100,000 rotations:
   `< 1` on 4.95 % (unscrambled) and 2.46 % (left-scrambled), max
   deficit 2.0 and 1.5 eps; `> 1` on 33.5 % and 53.0 %.
   (c) NumPy 1.23.0 (isolated py3.10): `type(np.nextafter(np.float32(
   0.4999), 0))` float64, with `np.float32(0)` float32; a float32
   array `>` the float64 neighbour -> `[False]`, `>` the float32
   neighbour -> `[True]`; `astype(np.float64) >` -> `[True]`.
   (d) `gh workflow list --repo jwestraadt/kikuchipy --all`: Weekly
   `disabled_inactivity` (id 282579366); `gh run list --workflow
   weekly.yml`: newest 2026-08-10; jobs of 2026-07-27, 08-03, 08-10:
   `test-documentation-notebooks` failure, `weekly-tests` success (3
   min 52 s, 3 min 53 s, 3 min 59 s).
2. **Source facts re-read:** EMsoftOO `c127868`
   `mod_dirstats.f90:836` (`Qi`, `Li` declared once in `EMforDS_`),
   `:865-932` (inits and iterations; `L_All(init) = L(i)`; exit when
   `i >= 2` and `|Q(i) - Q(i-1)| < 0.01`), `:934` (`maxloc`),
   `:1125-1174` (`getQandL_`: `oldQ = Q; oldL = L`, reuse when
   `minval(Phi) <= 0`), `:1177-1220` (`Density_`); kikuchipy `develop`
   `_dictionary_indexing.py:24` (`from time import sleep, time`),
   `:135` (`sleep(0.2)` unconditional), `:141-144` (`is_in_data =
   ~mask` on full coordinates); `_merge_crystal_maps.py:104`
   (`xmap._original_shape`); `signals/util/_crystal_map.py:28-61`
   (`_xmap_is_compatible_with_signal` compares `xmap.shape`); orix
   0.14.2 `crystal_map.py:340-347` (`size`, `shape` over in-data
   points), `:381-429` (`row`/`col` with the one-row rule and the
   minimum shift), `:687` (slicing clears `is_in_data`), `:1089-1106`
   (`_data_shape_from_coordinates`); orix `Orientation.angle_with`
   (`arccos(2 d^2 - 1)` with `nan_to_num`).
3. **Changes here:** V3 identical-neighbours arm on two seeded
   rotations (one per rounding side) and the absent-neighbour list
   `(1, 5, 13)`; V8 centre-pixel arm on the below-1 rotation, the
   strict-warning arm with `np.nextafter(m, np.float32(0))`, the
   round-trip arm with the vanished-label defaults, new
   `test_map_grid_spans_points_not_in_the_data`; V9 Watson underflow
   arm rewritten (unscrambled, traced transcription, precondition,
   structural asserts, recorded per-init `L`; `WATSON_UNDERFLOW_SEED`
   added to the MTP inventory); V13 and the gates table worded for a
   disabled Weekly workflow; V16 own-ball tolerance per mode, new
   `test_navigation_masked_dictionary_indexing_map_keeps_its_grid`,
   equality arm with the float32 neighbour, the skipped-grain arm's
   fixture variant `(8, 11)`; the `_dictionary_indexing` arm's `sleep`
   spy; mapping rows D1, D2, D8, D10, D15; killer rows M20 (correct
   module, second killer), M22 (iteration count), S16 redefined, S18-S20
   added; Local-gated weekly bullets and the oldest-matrix watch list;
   DoD Stage B wording; CI budget seed 49 s; a Performance row for the
   weekly additions.
4. **Remaining MTP** (added): `WATSON_UNDERFLOW_SEED` (seed 80); the
   weekly additions' local seconds and scaled CI estimate (target <= 5
   min). Unchanged otherwise.

### 4. 2026-10-06 (approval, main loop)

Johan approved `plan.md` (AskUserQuestion, after rounds 1 and 2): every
default of plan section 7 adopted (R1-R7, items 8-16, 24-27); R2
`average="mean"`, R3 EMsoft labels (0 unassigned, 1..n) and item 26
(weekly CI stays disabled, heavy arms run in the local stage gates)
confirmed explicitly. D20 (ball spacing, GROD reference, pre-run
coverage warning) added by Johan after the first draft and carried
into plan and validation by the round-1 fixer. Commit 1 follows.
