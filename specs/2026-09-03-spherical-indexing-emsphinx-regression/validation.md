# Phase 10 -- `spherical-indexing-emsphinx-regression`: validation

## Automated (default suite; run from Git Bash)

`uv run pytest tests/test_indexing/test_spherical_emsphinx_regression.py -n 4`
(no numba warm-up needed -- no kernels this phase).

All in `tests/test_indexing/test_spherical_emsphinx_regression.py`
unless stated. Bands follow the Phase 6 margin convention
(~1.7-2.1x on the measured worst value); every measured number is
dated in "Recorded results". Budget: **<= 8 s wall under `-n 4`**
(xdist spreads the module across workers, each rebuilding the
session-scoped harmonics fixture once -- the dominant fixed cost).

### The reference files (`TestReferenceFiles`, CI)

- `test_registry_lists_every_reference`: the `regression_*.npz`
  glob of `Path(kikuchipy.data.__file__).parent / "emsphinx"` (the
  installed package -- NOT `src/`, which the `-build-install-wheel`
  job does not have) equals the eight registry keys, and each
  file's md5 matches its `_registry_hashes` entry.
- `test_scenario_set_is_complete` (quad-equality): {the imported
  script module's scenario names} == {registry `regression_*` keys}
  == {the directory glob} == {the test module's frozen table} --
  a scenario added or renamed without regenerating fails on CI.
- `test_references_load_without_pickle`:
  `np.load(path, allow_pickle=False)` succeeds for all eight
  (no object arrays; kills pickled provenance).
- `test_frozen_keys_and_dtypes`: exactly the frozen key set
  (`phi1 phi phi2 metric iq phase` + `emsphinx_commit bw normed
  refine nregions gausbckg delta vendor route dataset scan_shape
  scan_steps sample_tilt pc namelist master_sht master_md5
  patterns_md5 preprocessing subset_slice emsphinx_compatible
  manufacturer flip kikuchipy_version`); result arrays float32
  except `phase` uint8; `pc` float64 `(3,)`; string provenance as
  fixed-width unicode; row counts 9 x6, 20, 165 matching
  `scan_shape` products.
- `test_provenance_pins`: `emsphinx_commit ==
  "60f351741036c63a59a6061a7ac2fca4f60f2c64"` (the test module's
  OWN literal -- the stored value is probed, so this check has
  power only through the independent duplication, per the module
  comment), `bw == 68`, `normed == True`, `sample_tilt == 70.0`,
  `phase` all zero, and the per-scenario
  `refine/nregions/gausbckg/vendor/delta/dataset` match the frozen
  scenario table.
- `test_each_file_within_budget`: **every** file < 100 kB (the
  tech-stack rule is per file); the total recorded via
  `record_property` (drafting candidates: 50,450 B over nine with
  the pre-review key set).

### Reference integrity guards (`TestReferenceIntegrity`, CI, refs only)

Class docstring: frozen bytes vs frozen bytes -- these exercise no
kikuchipy code and can only fire at regeneration or if the shipped
data is edited; they are **not** the CI parity surface (that is
orientations/scores/IQ vs our engine, the kwargs table, the route
pins and the stretch diagnostic).

- `test_emsoft_route_is_close_but_distinct_from_bruker`:
  m-3m misorientation max between `emsoft_d500` and `refined_nr10`
  < 1e-3 deg (measured 5.83e-5, 17x) AND the `metric` arrays are
  not identical (the `.6g` pc rounding differs at 1e-6).
- `test_coarse_and_refined_share_the_preprocessing`: `iq` of
  `coarse_nr10` bitwise equals `refined_nr10` (measured True).
- `test_refined_metric_exceeds_coarse_per_point`:
  `refined.metric - coarse.metric > 0` for all 9 (measured deltas
  +0.0032 to +0.0187; the drafting 0.02 upper bound is dropped --
  7 % headroom violated the margin convention and only strict
  positivity is meaningful).
- `test_preprocessing_scenarios_are_distinct`: the `iq` arrays of
  `refined_nr10` vs `nr0`, `nr7` and `nr10_gb` are pairwise not
  equal (measured IQ ranges 0.173-0.204 / 0.289-0.327 /
  0.18565-0.22098 / 0.187-0.216).

### Route pins (`TestRoutePins`, CI -- real conversion code)

- `test_namelist_matches_the_declared_route`: for each small
  scenario, `EMSphInxNamelist.from_kwargs(**<frozen table>)
  .to_string()` equals `str(ref["namelist"])` **exactly** -- one
  comparison pinning vendor/delta/pctr/bw/normed/refine/nregions/
  gausbckg/circmask/thetac/patdims/scandims and the file names
  (kills a wrong-vendor generation and any kwargs-table drift).
- `test_pc_matches_the_stored_namelist`: `EMSphInxNamelist
  .from_string(str(ref["namelist"])).to_detector(sample_tilt=70)
  .pc` equals `ref["pc"]` -- exact float64 equality, safe ONLY
  because `to_string` quantised at `.6g` before the round trip
  (stated in the test comment). Large-scenario recomputes live in
  the pooch-gated class.

### Ours vs theirs, small map (`TestOursVsTheirsSmall`, CI)

Shared fixtures: background-removed `nickel_ebsd_small`;
session-scoped `MasterPatternHarmonics.from_file` on the in-package
`.sht` (bw 384, resized by the indexer -- the `idx.hpp:182` parity
route); `detector.pc = ref["pc"]`; kwargs from
`scenario_kwargs(ref)`; one cached `spherical_indexing` run per
scenario (~0.15-0.3 s each measured warm). Misorientations with
**both sides** `Orientation(..., symmetry.Oh)` and `degrees=True`
(a bare `Rotation` on the theirs side raises `AttributeError`,
orix 0.14.2 -- review-measured); theirs =
`Rotation.from_euler(np.stack([phi1, phi, phi2], 1)
.astype(np.float64))` (EMSphInx `qu2eu` Bunge radians, `from_euler`
defaults).

- `test_scenario_kwargs_derive_from_provenance`: `scenario_kwargs`
  output equals the frozen per-scenario table literal (the
  structural killer for provenance-ignoring mutants -- the refine
  flag has no band signature, see "Recorded results" mutant
  measurements).
- `test_sample_tilt_binding`: `detector.sample_tilt ==
  harmonics.sample_tilt == ref["sample_tilt"] == 70.0`.
- `test_orientations_agree[<scenario>]` (6 params):
  - coarse_nr10: median < **1.0** (measured 0.510, 1.96x), at
    least 8/9 < **1.25** (measured max 0.622, 2.01x), all < **4.0**
    -- the single-cell-jump ceiling: one correlation-grid cell at
    bw 68 is 360/`fast_size(135)` = 2.667 deg, so a lone
    neighbour-cell argmax on another platform is a legitimate
    near-tie (Phase 6's outlier convention);
  - every refined scenario: median < **0.7** (measured 0.310-0.341,
    >= 2.05x), max < **0.75** (measured 0.335-0.367, >= 2.05x) --
    plain maxima: refinement re-converges a near-tie.
- `test_scores_correlate[<scenario>]`: Pearson r > **0.85**
  (measured 0.935-0.969); mean |ours - metric| < **0.03** (measured
  0.0088-0.0139); max |ours - metric| < **0.07** (measured
  0.0226-0.0330). Never equality: both metrics are normalised but
  ~2 % apart systematically.
- `test_iq_is_float32_equal[<scenario>]`: max |xmap.iq - ref.iq| <
  **1e-3** -- the preprocessing discriminator. Ladder: measured
  parity <= 1.4e-8 on this machine; one uint8 gray level in one
  pixel moves that pattern's IQ by up to 5.2e-5 (AHE histogram
  cascade -- review re-measurement, 12x the critic's 4.04e-6), the
  numba-fastmath background removal has ~4 truncation-boundary
  pixels per map, so the platform-drift budget is ~2e-4; the
  smallest real signature is 1.5e-2 (gausbckg) and the smallest
  kwargs-mutant signature 2.27e-2 (nr7 vs nr10) -- 1e-3 is ~5x
  above the drift budget, >= 15x below the smallest kill. The
  assertion message names `ref["patterns_md5"]` so platform drift
  fails as "your background-removed patterns differ from the
  reference machine's".
- `test_stretch_emulation_collapses_the_residual` (the D7 evidence
  pin): re-run the refined anchor with the item-31 stretch emulated
  in pc space (`pc' = (0.5/w + pc_x (w-1)/w, 0.5/h + pc_y (h-1)/h,
  pc_z (w-1)/w)`); assert its median misorientation vs the same
  reference is < **0.2** deg (the original mission gate; measured
  0.0940, 2.1x) and < the unmodified anchor's median (measured
  0.3404); record both via `record_property`.
- Measured medians/maxima/r recorded per scenario via
  `record_property`.

### Ours vs theirs, large map (`TestOursVsTheirsLarge`)

- `test_the_twenty_point_scenario` (default suite,
  `pytest.importorskip("pooch")`, `allow_download=True`): the
  `[::15, ::15]` subset after full-map background removal, full-map
  detector with `pc = ref["pc"]`; median < **0.7** (measured
  0.325), max < **0.8** (measured 0.373, 2.15x), r > **0.90**
  (measured 0.973), score mean/max |diff| < 0.03/0.07 (measured
  0.0110/0.0287), IQ < 1e-3 (measured 7.3e-9); plus the large
  namelist/pc route pins (measured pc
  `(0.42326, 0.213633, 0.502074)`).

### Namelist module (`tests/test_indexing/test_spherical_namelist.py`, CI)

- `test_emsoft_delta_invariance_of_the_conversion` (the retired
  delta axis's content, D1): EMsoft-vendor `from_kwargs ->
  to_string -> from_string -> to_detector(70).pc` is bit-identical
  across delta {125, 250, 500} (measured True) and the `.6g`
  `pctr` `sDst` values are the decimal-exact 3755.3 / 7510.6 /
  15021.2.

## Local-gated (KIKUCHIPY_EMSPHINX_DIR; skipped on CI)

- `test_regenerated_references_are_bitwise` (calls the conftest
  `emsphinx_program("IndexEBSD")` -- gate, "not built" skip and
  the machine-wide binary lock -- and passes the resolved path
  into `create_emsphinx_reference.main(tmp_path, program=...)`)
  over the six small scenarios; every regenerated file md5-equals
  the shipped one (measured foundation: two full drafting sweeps
  were md5-identical on all files, under an unchanged fftw.wisdom);
  on mismatch the assertion message names the fftw.wisdom state as
  suspect #1 (FFTW_PATIENT + machine-wide wisdom -- D4) and then
  carries the per-array diff. Runtime ~8 s (derived from the
  measured 10.8 s full sweep minus the large-map runs; measure and
  record the small-only figure at implementation).

## Weekly

`uv run pytest --weekly tests/test_indexing/test_spherical_emsphinx_regression.py`

- `test_the_165_point_scenario` (pooch): median < **0.7** (measured
  0.339), p95 < **0.9** (measured 0.441, 2.04x), max < **1.0**
  (measured 0.487, 2.05x), r > **0.88** (measured 0.944), score
  mean/max |diff| < 0.03/0.07 (measured 0.0118/0.0364), IQ < 1e-3
  (measured 7.4e-9).
- `test_regenerated_references_are_bitwise_all` (weekly + gated):
  the full eight-scenario regeneration incl. the large subsets,
  md5-compared (measured 10.8 s end-to-end + the full-map
  background removal).

## Manual

- Run `uv run python src/kikuchipy/data/emsphinx/
  create_emsphinx_reference.py` once on this machine during
  implementation (it takes the machine-wide lock itself and prints
  the fftw.wisdom md5 before/after); commit the eight `.npz` and
  the registry md5s; spot-check one file with `np.load` (keys,
  shapes, provenance incl. the namelist text).
- Confirm `pre-commit` leaves the `.npz` files untouched (the GPL
  licenseheaders hook has no `files:` regex -- the protection is
  licenseheaders' own extension whitelist, verify it holds) and
  the doctest ignore-glob covers the script.
- Eyeball the PR diff: no `specs/` leakage into runtime code, no
  public-API surface beyond the two D9 docstring Notes.

## Definition of done

- [ ] Spec + plan-0 amendments committed (incl. the mission
      criterion 2 re-anchor with the item-31 decomposition, the
      criterion 5 CHANGELOG reconcile and the roadmap Phase 10
      five-box rewrite).
- [ ] Failing tests committed first (reference names/keys/bands
      asserted before the refs exist).
- [ ] `create_emsphinx_reference.py` committed; eight `.npz`
      generated on this machine and committed; registry md5s in;
      `pyproject.toml` coverage omit in; the namelist
      delta-invariance test in; the two D9 docstring Notes in.
- [ ] Default suite green without `KIKUCHIPY_EMSPHINX_DIR` and
      without pooch (large tests skip cleanly); gated +
      weekly-gated suites green on this machine.
- [ ] Adversarial review (fidelity to the recorded constraints,
      conventions, bug-injection list of plan 5.2) passed with
      fixes applied; new test module's helper coverage 100 %.
- [ ] `pre-commit run --files <changed>` clean; PR #11 opened with
      the licence statement and the no-CHANGELOG rationale.

## Recorded results

### 2026-09-03 -- drafting-probe measurements (spec phase; binaries @ 60f3517, Windows, this machine)

NB (revision 2026-09-03): the `small_refined_nr4` and
`small_refined_emsoft_d250` rows below are **drafting-only** -- the
final matrix replaces nr4 with nr7 and retires the d250 reference
(see the review re-measurements section); the `.ang` decimal counts
are corrected there (5/3/1, not 6/~4/1).

Scripts (session scratchpad, not committed): `p10_gen.py` (the full
scenario sweep + determinism rerun + candidate `.npz` writer),
`p10_ours.py` (kikuchipy vs the candidates + ref self-consistency),
`p10_extra.py` (the gausbckg scenario + coarse/refined IQ and
metric cross-checks + npz byte-stability). Inputs:
`kp.data.nickel_ebsd_small()` and `nickel_ebsd_large()` (both
backgrounds removed), the in-package
`emsphinx/ni_small_20kv_bw384.sht` (sample_tilt 70.0), writer and
namelist from Phase 9 (`write_emsphinx_patterns` defaults,
`EMSphInxNamelist.from_kwargs`, `bw` 68, `normed`/`gausbckg` per
scenario, `nthread=1 batchsize=1`, `data_file=out.h5`,
`vendor_file=out.ang`).

**Datafile layout** (all nine runs): `Scan 1/EBSD/Data/{IQ, Metric,
Phi, Phi1, Phi2}` float32 + **`Phase` uint8** (Phase 9 D7 recorded
float32 for all -- corrected); `Phase` all zero.

**`.ang` vs datafile** (all nine runs): Euler max |diff| 4.6e-6 to
5.0e-6 rad (5-decimal fixed text -- half-ULP bound 5e-6); ci vs
Metric max |diff| 4.0e-4 to 5.0e-4 (ci = Metric at 3 decimals --
bound 5e-4); `.ang` iq column = IQ at one decimal -- unique values
{0.2} (small nr10/nr4/gb/emsoft), {0.3} (small nr0), {0.1, 0.2}
(large) -- Phase 9's "iq column constant 0.2" explained.

**Binary runs** (exit 0 everywhere; `IndexEBSD.exe` stdout timing):

| scenario | index time | Metric range | IQ range |
|---|---|---|---|
| small_coarse_nr10 | 0.0630 s (142.9 pat/s) | 0.5643-0.6623 | 0.17266-0.20363 |
| small_refined_nr10 | 0.0783 s (115.0 pat/s) | 0.5749-0.6669 | 0.17266-0.20363 |
| small_refined_nr0 | 0.0773 s (116.5 pat/s) | 0.5756-0.6758 | 0.28900-0.32686 |
| small_refined_nr4 (drafting-only) | 0.0778 s (115.7 pat/s) | 0.5447-0.6218 | 0.19984-0.24123 |
| small_refined_nr10_gb | 0.0822 s (109.5 pat/s) | 0.569012-0.663151 | 0.18734-0.21588 |
| small_refined_emsoft_d500 | 0.0821 s (109.6 pat/s) | 0.5749-0.6669 | 0.17266-0.20363 |
| small_refined_emsoft_d250 (drafting-only) | (bitwise == d500) | == d500 | == d500 |
| large20_refined_nr10 | 0.1815 s (110.2 pat/s) | 0.5196-0.7134 | 0.14612-0.21072 |
| large165_refined_nr10 | 1.400 s (116.8 pat/s) | 0.4685-0.7134 | 0.13403-0.23172 |

(The gb Metric range, blank at drafting, was filled from the same
stored run during the review.)

Process wall clock 0.18-0.20 s per small run, 0.31 s large-20,
1.53 s large-165. **Whole sweep (8 scenarios + determinism rerun)
10.8 s end-to-end** on a warm cache; the gausbckg scenario adds
~0.2 s.

**Determinism**: `small_refined_nr10` rerun -- `phi1/phi/phi2/
metric/iq` all bitwise identical. Full-script rerun: all candidate
`.npz` files **md5-identical** across sweeps. `np.savez` of
identical arrays two seconds apart: byte-identical (NumPy writes
timestamp-free zip entries). NB (revision): both reruns ran under
a warm, unchanged `C:/ProgramData/fftw.wisdom` -- see the review
section for the FFTW_PATIENT qualification.

**Effective pc** (`.6g` namelist round trip, float64): small Bruker
`(0.425139, 0.213367, 0.500707)`; small EMsoft route
`(0.425139, 0.21336666..., 0.50070666...)` (differs in the 6th
decimal); large `(0.42326, 0.213633, 0.502074)` (full-map
`pc_average`).

**Delta pair** (EMsoft vendor, same physical geometry, `sDst`
15021.2 vs 7510.6 at `.6g`): Euler/Metric/IQ **bitwise identical**
(max |diff| 0.0). EMsoft d500 vs the Bruker anchor: Euler max
|diff| 9.5e-7 rad; m-3m misorientation max **5.83e-5 deg**; metric
arrays differ in the last float32 digits (0.6669433 vs 0.6669446).
(Revision consequence: the pair is inert -- the d250 reference is
retired, D1.)

**Ours vs theirs** (kikuchipy `spherical_indexing`, harmonics
`from_file` at bw 384 indexed at 68, `detector.pc = ref["pc"]`,
kwargs per scenario; m-3m `angle_with`; wall = warm kikuchipy run):

| scenario | n | mis median | mis p95 | mis max | score r | score \|d\| mean | score \|d\| max | IQ \|d\| max | wall |
|---|---|---|---|---|---|---|---|---|---|
| small_coarse_nr10 | 9 | 0.5096 | 0.6044 | 0.6219 | 0.9413 | 0.01392 | 0.03299 | 7.0e-9 | 0.31 s |
| small_refined_nr10 | 9 | 0.3404 | 0.3634 | 0.3643 | 0.9515 | 0.01257 | 0.02702 | 7.0e-9 | 0.19 s |
| small_refined_nr0 | 9 | 0.3104 | 0.3351 | 0.3353 | 0.9693 | 0.00882 | 0.02259 | 1.4e-8 | 0.15 s |
| small_refined_nr4 (drafting-only) | 9 | 0.3336 | 0.3609 | 0.3617 | 0.9453 | 0.01059 | 0.02717 | 5.7e-9 | 0.14 s |
| small_refined_nr10_gb | 9 | 0.3406 | 0.3567 | 0.3570 | 0.9566 | 0.01258 | 0.02962 | 7.3e-9 | ~0.15 s |
| small_refined_emsoft_d500 | 9 | 0.3404 | 0.3634 | 0.3643 | 0.9515 | 0.01257 | 0.02702 | 7.0e-9 | 0.14 s |
| small_refined_emsoft_d250 (drafting-only) | 9 | 0.3404 | 0.3634 | 0.3643 | 0.9515 | 0.01257 | 0.02702 | 7.0e-9 | 0.14 s |
| large20_refined_nr10 | 20 | 0.3248 | 0.3678 | 0.3725 | 0.9725 | 0.01098 | 0.02871 | 7.3e-9 | 0.22 s |
| large165_refined_nr10 | 165 | 0.3390 | 0.4409 | 0.4867 | 0.9444 | 0.01175 | 0.03643 | 7.4e-9 | 1.25 s |

(The gb p95, blank at drafting, was measured during the review.)

Score ranges overlap on the same scale (e.g. small refined ours
0.5479-0.6746 vs Metric 0.5749-0.6669): both normalised, ~2 %
apart, never equal -- Pearson + |diff| bands, not equality. IQ is
float32-equal (ours float64 0.17266215867... vs stored float32
0.172662153...): equality-band, not correlation.

**Ref self-consistency** (m-3m misorientation max, deg):
refined vs coarse 0.5841; nr10 vs nr0 0.0863; nr10 vs nr4 0.0666;
nr10 vs nr10_gb 0.0305; emsoft_d500 vs d250 0.0; emsoft_d500 vs
Bruker 5.83e-5. Coarse vs refined `iq` **bitwise equal**; refined
metric per-point above coarse: deltas +0.00318 to +0.01868 (9/9
up). The preprocessing scenarios are therefore indistinguishable by
orientation bands (0.03-0.09 deg gaps) and sharply distinguishable
by IQ (1e-2-1e-1 range shifts) -- the IQ band carries the
discrimination.

**Band-survivor mutants measured** (plan 5.2 honesty check):
ours-refined vs the coarse ref -- median 0.5752, max 0.6570, score
|d| mean 0.0102/max 0.0202; ours-coarse vs the refined ref --
median 0.4308, max 0.6678, score |d| mean 0.0227/max 0.0381. Both
sit inside every plausible band (the refined max band would need
< 0.62 to catch one, a knife edge against the 0.364 measurement),
so the refine-flag mutant is killed structurally by the frozen
kwargs-table test, not by bands.

**Candidate `.npz` files** (drafting key set -- WITHOUT `gausbckg`
and the review-added provenance keys; md5s therefore provisional,
the shipped set is pinned at implementation; nr4/d250 rows are
drafting-only):

| candidate | bytes | md5 |
|---|---|---|
| small_coarse_nr10.npz | 5211 | 22bb34349607f5356419129a7e3a82cb |
| small_refined_nr10.npz | 5211 | 8dd2a3e79056057e624c75fe5c0e8738 |
| small_refined_nr0.npz | 5211 | b33ca4dccc974548beb60937f6d2a158 |
| small_refined_nr4.npz | 5211 | 26f99dc3fd7e909f1344f829af1bfd36 |
| small_refined_nr10_gb.npz | 5211 | 58739cebe16c1e2af8e6c26850602764 |
| small_refined_emsoft_d500.npz | 5211 | 2d28b611bbdcf63bfe61192c2ab19f91 |
| small_refined_emsoft_d250.npz | 5211 | ea45a3a645a1722b5082c440b8780890 |
| large20_refined_nr10.npz | 5462 | bf4ee093c1edcec3c1a8c50f347ec4ba |
| large165_refined_nr10.npz | 8511 | 358b1657262c5af1f06b92c979e27912 |

Total **50,450 B** (the final key set adds ~2 kB/file; each file
stays far under the per-file 100 kB rule).

**Environment**: `nickel_ebsd_large` served from the local pooch
cache; `MasterPatternHarmonics.from_file` on the in-package `.sht`
reports phase Ni m-3m, `sample_tilt` 70.0, `bandwidth` 384 (the
resize-at-index parity route of research item 39). The
kikuchipy-vs-IndexEBSD canonical-route anchor (median 0.3404 / max
0.3643, r 0.9515) supersedes the Phase 9 Bruker-route context
numbers (0.341/0.363, r 0.9607) as this phase's baseline --
essentially unchanged, now on the canonical route with the rounded
pc.

### 2026-09-03 -- adversarial-review re-measurements (revision; binaries @ 60f3517, Windows, this machine)

Scripts (session scratchpad, not committed): the fidelity critic's
`p10c_gen.py`/`p10c_an1.py`/`p10c_ours.py`/`p10c_large.py`/
`p10c_stretch.py`/`p10c_stretch2.py`/`p10c_nr.py`, re-run and
extended by the revision's `p10r_quick.py`/`p10r_nr7.py`/
`p10r_iqsens.py`. Every drafting headline number reproduced
bitwise/exactly on the critic's independent sweep before these
deltas were measured.

**Item-31 stretch decomposition** (`p10c_stretch.py`/
`p10c_stretch2.py`, re-run by the revision -- the D7 basis).
Emulating EMSphInx's un-ported `bilinearCoeff` stretch in pc space
(`pc' = (0.5/w + pc_x f, 0.5/h + pc_y f, pc_z f)`, `f = (w-1)/w =
0.98333`):

| scenario | baseline med/max (deg) | stretch-emulated med/max | r baseline -> emulated |
|---|---|---|---|
| small coarse nr10 | 0.5096 / 0.6219 | 0.0745 / 0.1288 | 0.9413 -> 0.9775 |
| small refined nr10 | 0.3404 / 0.3643 | 0.0940 / 0.1166 | 0.9515 -> 0.9727 |
| large20 refined | 0.3248 / 0.3725 | 0.0784 / 0.1429 | 0.9725 -> 0.9762 |
| large165 refined | 0.3390 / 0.4867 | 0.0717 / 0.2190 | 0.9444 -> 0.9650 |

Scale scan (small refined median): f 0.975 -> 0.5073, 0.98 ->
0.2634, **0.98333 -> 0.0940**, 0.99 -> 0.3277, 1.0 (xy shift only)
-> 0.8278, 1.01 -> 1.3547 -- a sharp minimum exactly at the
predicted `(w-1)/w`. Separated terms: z-only scaling 0.5469, xy-only
0.1422 -- the full stretch is what collapses it. Against the
STORED xmap the emulation is *worse*: large20 0.3653 -> 0.5838,
large165 0.3800 -> 0.5761 deg median -- kikuchipy's convention is
the physical one; the port keeps it (D7).

**`nregions` 7 -- the mosaic-AHE remainder path** (`p10r_nr7.py`;
60 % 7 = 4, never exercised by the dividing drafting set): mis
median 0.3346 / p95 0.3633 / max 0.3667 deg, r 0.9347, score
|d| mean 0.01166 / max 0.02640, IQ max |diff| **6.6e-9**; Metric
range 0.562044-0.637833, IQ range 0.18565-0.22098; nr7-ref-IQ vs
nr10-ref-IQ max gap **2.27e-2** (the kwargs-mutant signature).
All inside the refined bands -- nr7 replaces nr4 in the matrix
(D1).

**One-gray-level IQ sensitivity** (`p10r_iqsens.py` -- the IQ band
ladder): a single +-1 uint8 bump in one pixel of one pattern,
24-trial scan over random interior pixels: worst IQ shift of that
pattern **5.197e-5** (typical 2e-6-4e-5; 5 of 24 exactly 0.0 --
the AHE rank mapping absorbs some bumps). The conventions critic's
4.04e-6 single measurement understates the worst case by 12x --
hence the band at **1e-3**, not the critic's proposed 1e-4
(~5x above the ~2e-4 drift budget of ~4 boundary pixels, >= 15x
below the 1.5e-2 smallest real signature).

**`.ang` column precision** (`p10r_quick.py`, vs `tsl.hpp:783-794`
`std::fixed` setprecision(5)/(1)/(3)): measured data line
`6.08360 1.56232 4.14436 0.00000 0.00000 0.2 0.662 0` -- decimals
per column [5, 5, 5, 5, 5, 1, 3, 0]; anchor max |ang Euler - h5| =
4.685e-6 rad, |ci - Metric| = 4.143e-4. The drafting "6-decimal"/
"~4-decimal" descriptions were off by one; the measured diffs were
right and are exactly the 5-decimal/3-decimal half-ULP bounds. The
generation cross-check tolerances (1e-5 rad, 1e-3) are exactly 2x
the deterministic bounds.

**Delta-axis inertness, kikuchipy side** (`p10r_quick.py`): the
EMsoft-route `from_kwargs -> to_string -> from_string ->
to_detector(70).pc` is **bit-identical** for delta {125, 250, 500}
(`array([0.42513883, 0.21336667, 0.50070667])`; `pctr` `sDst`
3755.3 / 7510.6 / 15021.2 -- exact decimal halving at `.6g`).
Together with the drafting bitwise binary result: no shipped test
could distinguish a wrongly generated d250 reference -- the axis is
retired (D1) and the invariance pinned in the namelist module.

**Repack byte-identity needs the reshape** (`p10r_quick.py`):
`/patterns` is `(9, 60, 60)`; `np.array_equal(patterns,
signal.data)` (shape `(3, 3, 60, 60)`) is **False**;
`np.array_equal(patterns, signal.data.reshape(-1, 60, 60))` is
**True**; the flipped reshape is False. The D4 guard is spelled
with the reshape.

**gb gap-fills** (`p10r_quick.py`/`p10r_nr7.py`): Metric range
0.569012-0.663151; ours-vs-theirs mis p95 0.3567 (max 0.3570) --
both merged into the drafting tables above.

**FFTW wisdom state** (the D4/D8 qualification): the machine-wide
`C:/ProgramData/fftw.wisdom` at measurement time: 391,837 B, md5
**edf8c26f975e50e372d387e81bcd288a** (mtime 2026-09-03 00:57). The
binaries plan with FFTW_PATIENT and import/export this file at
start/exit, so "bitwise deterministic" holds *given this wisdom*;
every drafting and review rerun ran warm against it. The
generation script prints the md5 before/after its sweep; a changed
wisdom is suspect #1 in any regenerate-and-diff mismatch.

**Repo-config verifications** (revision): `[tool.coverage.run]` has
`source = ["src/kikuchipy"]` and **no `omit`** -- the existing
`create_emsphinx_sht_fixtures.py` reports 0 %, so the omit of plan
4.2 is required, not pre-existing; the GPL licenseheaders hook has
**no `files:` regex** (only an `exclude:`) -- `.npz` protection is
licenseheaders' own extension whitelist; the in-package registry
block is grouped by directory inside `# fmt: off` with a
hand-aligned md5 column (longest existing key 62 chars, longest new
key 49); `tests/test_indexing/test_spherical_namelist.py` exists
(the delta pin's home); the conftest lock file is
`<tempdir>/kikuchipy-emsphinx-program.lock` (stdlib
`O_CREAT|O_EXCL`, stale takeover) and the `emsphinx_program`
"not built" skip lives inside the returned callable.
