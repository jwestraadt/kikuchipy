# Phase 10 -- `spherical-indexing-emsphinx-regression`: requirements

Branch `spherical-indexing-emsphinx-regression` (roadmap Phase 10).
The phase turns the Phase 9 interop tooling into a shipped,
CI-runnable regression suite: `IndexEBSD.exe` @ 60f3517 is run once
on kikuchipy-written inputs, its outputs are frozen as small
in-package `.npz` reference files, and CI tests compare kikuchipy's
own `spherical_indexing` against them on identical inputs -- no
binaries needed after generation.

Every "measured" number below was produced on 2026-09-03 by the
drafting probes (`p10_gen.py`, `p10_ours.py`, `p10_extra.py`) or by
the 2026-09-03 adversarial-review re-measurements (`p10c_*.py`,
`p10r_*.py`; all session scratchpad, not committed; recipes and
full outputs in `validation.md` "Recorded results") against the
built binaries in `c:/Users/westraadt.1/Repos/EMSphInx/build/
Release/` @ 60f3517. **Revised 2026-09-03 after adversarial review**
(fidelity + conventions critics); the revision's decisive
re-measurements -- the research-item-31 stretch decomposition, the
retired delta axis, the `nregions` 7 remainder-path scenario, the
one-gray-level IQ sensitivity, the `.ang` column precision -- are
recorded in the dated review section of `validation.md`.

## Scope

In scope:

- **`src/kikuchipy/data/emsphinx/create_emsphinx_reference.py`** --
  the committed, pinned reference-generation script (D4): repacks
  background-removed `nickel_ebsd_small` and two `nickel_ebsd_large`
  subsets with `write_emsphinx_patterns` on the **canonical route**
  (writer default `Manufacturer` EMsoft, rows unflipped; nml vendor
  Bruker with `pc_average`; `nthread=1 batchsize=1` -- the recorded
  Phase 9 constraint), writes namelists via
  `EMSphInxNamelist.from_kwargs` for the frozen scenario matrix
  (D1), runs `IndexEBSD.exe`, and converts each run's datafile
  output into one provenance-carrying `.npz` (D2). Runnable only on
  a machine with the binaries (`KIKUCHIPY_EMSPHINX_DIR`); run once,
  output committed (the `create_emsphinx_sht_fixtures.py`
  precedent).
- **Eight `.npz` reference files** shipped in
  `src/kikuchipy/data/emsphinx/` with md5 entries in
  `src/kikuchipy/data/_registry.py` (no URL -- in-package). Drafting
  candidates measured 5,211-8,511 B each with the drafting key set;
  the final key set (D2, + the verbatim namelist text and the md5
  provenance strings) adds ~2 kB/file -- every file far below the
  constitution's **per-file < 100 kB** rule (D2). (corrected
  2026-09-03 at implementation: the final key set adds ~21 kB/file,
  not ~2 kB -- `to_string()` emits 4,228-4,232 characters of
  commented template and NumPy stores unicode as UCS-4, so the
  `namelist` array alone is ~16.9 kB. Shipped: 26,292-29,632 B per
  file, 214,027 B total; the per-file rule still holds with 3.4x
  margin.)
- **Bidirectional regression tests**
  (`tests/test_indexing/test_spherical_emsphinx_regression.py`):
  - *ours-vs-theirs, CI, no binaries* -- **the real CI parity
    surface**: `EBSD.spherical_indexing` on the identical inputs
    (the canonical-route repack is byte-identical to
    `signal.data.reshape(-1, h, w)`, so CI needs no HDF5 file) vs
    the shipped refs -- orientations, scores and IQ with
    measured-then-pinned bands, the frozen kwargs table, the
    namelist/pc route recomputes, and the stretch-emulated
    decomposition diagnostic (D5, D6, D7). Small-map scenarios in
    the plain default suite; the large 20-point subset
    default-suite behind `pytest.importorskip("pooch")`; the
    165-point subset weekly.
  - *reference integrity guards, CI, refs only*: frozen bytes vs
    frozen bytes -- they can only fire at regeneration or if the
    shipped data is edited, and are labelled so (D6): coarse/refined
    IQ bitwise equal, refined metric per-point above coarse,
    scenario distinctness, provenance pins, the scenario-set
    completeness quad-equality.
  - *theirs-on-ours*: already delivered by Phase 9's local-gated
    acceptance tests; Phase 10 adds only the **regenerate-and-diff**
    test -- rerunning the generation script reproduces the shipped
    files **md5-identical** (measured across full reruns, under an
    unchanged FFTW wisdom -- D4/D8), small sweep local-gated, large
    scenarios local-gated + weekly (D8).
- **One namelist unit test** added to
  `tests/test_indexing/test_spherical_namelist.py`: the EMsoft-route
  delta invariance of the PC conversion, pinned as the pure
  kikuchipy round trip it is (D1 -- the retired delta axis).
- **Two `Notes` sentences** on `EBSD.spherical_indexing` and
  `SphericalIndexer` carrying the measured parity numbers, so
  something shippable cites them (D9; `specs/` is stripped from any
  upstream PR, so the numbers must live in the code they describe).
- **`pyproject.toml`**: `[tool.coverage.run]` gains
  `omit = ["src/kikuchipy/data/*/create_*.py"]` -- no omit exists
  today, so the never-imported generation scripts (the existing
  `create_emsphinx_sht_fixtures.py` included) land at 0 % and move
  the codecov project status (D9).
- **Constitution amendments** listed in plan section 0 (NOT applied
  at drafting): roadmap Phase 10 box rewrite with the measured
  bands and the two recorded matrix amendments (delta axis retired,
  `nregions` 4 -> 7), tech-stack data bullet + Numerics stretch
  consequence, **mission success-criterion 2 re-anchoring on the
  measured decomposition** (D7) and the criterion-5 CHANGELOG
  reconcile, research addenda.

Out of scope (confirmed):

- **Refine-only (`msk & 0x02`) reference scenarios** -- FORBIDDEN by
  the Phase 7 research addendum (`explore-emsphinx-core-algorithm.md`
  section 8 item 40): the shipped `refineImage` discards its
  refinement and stores a zero or stale score, so its outputs are
  not valid references. Structurally unreachable here anyway: the
  refine-only work item requires a scan-file input (string
  `scandims`), which `EMSphInxNamelist` refuses
  (`NotImplementedError`) and `from_kwargs` cannot emit. The
  during-indexing `refine = .TRUE.` path is the refined route (D1).
- **A second-`delta` reference scenario** -- retired by measurement
  (D1): the EMsoft-route delta pair is bitwise inert through the
  binary *and* bit-identical in the round-tripped pc, so a d250
  reference would differ from d500 only in provenance scalars and
  no CI test could detect a wrongly generated one.
- `normed = .FALSE.` scenarios (the roadmap fixes `normed` T; the
  un-normalised metric is compared nowhere).
- New acceptance tests against the binaries (Phase 9's acid test and
  negative controls own that surface; D8).
- Any public API or signal-surface change, docs page or CHANGELOG
  entry -- **except** the two docstring `Notes` sentences above (D9
  -- data, a maintainer script, tests and the parity Notes).
- A pseudo-symmetry scenario (Phase 8 deferred; the roadmap re-scope
  records that Phase 10's scenarios do not exercise it).

## Decisions

### D1 -- The scenario matrix (frozen)

Eight references. All use `bw` 68, `normed` T, `gausbckg` F unless
stated, `circmask` -1, `thetac` 0, `nthread=1 batchsize=1`, the
canonical file route (`Manufacturer` EMsoft, rows unflipped), master
= in-package `emsphinx/ni_small_20kv_bw384.sht` (`sample_tilt` 70.0),
uint8 patterns (the Phase 9 D2 constraint), namelist vendor Bruker
with the unrounded `pc_average` unless stated:

| reference file | dataset | vendor | delta | refine | nregions | gausbckg |
|---|---|---|---|---|---|---|
| `regression_small_coarse_nr10.npz` | small 3x3 | Bruker | 500 | F | 10 | F |
| `regression_small_refined_nr10.npz` | small 3x3 | Bruker | 500 | T | 10 | F |
| `regression_small_refined_nr0.npz` | small 3x3 | Bruker | 500 | T | 0 | F |
| `regression_small_refined_nr7.npz` | small 3x3 | Bruker | 500 | T | **7** | F |
| `regression_small_refined_nr10_gb.npz` | small 3x3 | Bruker | 500 | T | 10 | **T** |
| `regression_small_refined_emsoft_d500.npz` | small 3x3 | **EMsoft** | 500 | T | 10 | F |
| `regression_large20_refined_nr10.npz` | large `[::15, ::15]` = 4x5 | Bruker | 500 | T | 10 | F |
| `regression_large165_refined_nr10.npz` | large `[::5, ::5]` = 11x15 | Bruker | 500 | T | 10 | F |

Design rationale (one-factor-at-a-time around the canonical anchor
`small_refined_nr10`, which is exactly the Phase 9 acid-test
configuration):

- **refine axis**: F vs T at `nregions` 10 (the roadmap's "refine
  F/T on the during-indexing path"). Only the during-indexing
  refine; see Scope for the refine-only ban.
- **nregions axis**: {0, 7, 10} at refine T. 0 disables the mosaic
  AHE, 10 is the `IndexEBSD` default (60/10 = 6, the dividing-tile
  mosaic == CLAHE regime Phase 5 pinned). **7 replaces the drafting
  matrix's 4** (review finding): 4 and 10 both divide 60, so the
  drafting matrix never compared the mosaic-AHE *remainder* path
  (research item 35: the mosaic equals skimage CLAHE only for
  dividing tiles) against the binary; `nregions` 7 (60 % 7 = 4)
  exercises it -- measured: mis 0.3346/0.3667 deg, r 0.9347, IQ max
  |diff| 6.6e-9, IQ range 0.18565-0.22098 (the roadmap's recorded
  {0, 4, 10} set is amended in the plan 0.1 box rewrite).
- **gausbckg scenario** (an addition beyond the roadmap's letter,
  flagged in plan 6.4): without it the ported 2-D Gaussian
  background fit is never exercised end-to-end against the binary.
  One scenario closes the preprocessing coverage; +5,211 B.
- **vendor axis**: `emsoft_d500` exercises the EMsoft PC conversion
  end-to-end. The EMsoft-route effective pc differs from the Bruker
  route's in the 6th decimal (`.6g` rounding of different
  representations: `(0.425139, 0.21336667, 0.50070667)` vs
  `(0.425139, 0.213367, 0.500707)`); measured consequence 5.8e-5
  deg max between `emsoft_d500` and the Bruker anchor.
- **delta axis: RETIRED by measurement** (review finding; the
  drafting matrix carried `emsoft_d250`). The roadmap constraint
  said the sweep must use the EMsoft vendor route (delta cancels
  bitwise for the fractional vendors -- Phase 9 D6) or become a
  sanityCheck-window probe. Measured: the EMsoft route is *also*
  inert -- the d250/d500 runs are **bitwise identical** through the
  binary (`from_kwargs` scales `sDst = PCz*h*delta` and `idx.hpp:
  219-221` divides it back out), and the `.6g`-round-tripped pc is
  **bit-identical for delta 500/250/125** (`sDst` 15021.2/7510.6/
  3755.3 -- exact decimal halving at 6 significant figures). A d250
  reference would therefore differ from d500 only in provenance
  scalars; no CI test (the pc recompute included) could detect a
  wrongly generated one, and its "bitwise identical" self-test
  would compare frozen bytes to frozen bytes. The delta invariance
  is what it always was -- a property of the kikuchipy conversion
  -- and is pinned as a **namelist round-trip unit test** in
  `tests/test_indexing/test_spherical_namelist.py` (bit-identical
  `to_detector` pc and the decimal-exact `.6g` `sDst` across delta
  {125, 250, 500}); both drafting deltas sat inside sanityCheck's
  [5, 90] mm window anyway, so nothing was probed there either. The
  roadmap box rewrite (plan 0.1) records the null measurement and
  the retirement.
- **large subsets**: exactly the point sets the existing suite
  uses -- `[::15, ::15]` (20 points; default suite, pooch-gated,
  matching `test_the_twenty_point_subset`) and `[::5, ::5]` (165
  points; weekly, matching the 165-point weekly tests). Repacked as
  their own regular scans (4x5 and 11x15) after full-map background
  removal, with the **full map's** `pc_average` (the existing
  tests' convention) and `scan_steps` = navigation-axis scale x
  step (recorded provenance, not load-bearing). Only the canonical
  refined scenario per subset: the matrix's axes are covered on the
  small map, and the large refs exist to give the bands statistics
  (165 points) and a default-suite real-map guard (20 points).
- **No wrong-flip / negative-control refs**: garbage routes are
  Phase 9's negative controls, not reference data.

### D2 -- Reference file format (frozen)

- One `np.savez` (uncompressed) file per scenario, name
  `regression_<scenario>.npz` in `src/kikuchipy/data/emsphinx/`.
  Uncompressed because the files are tiny, zlib output is a
  cross-version wildcard, and NumPy's zip writing is
  timestamp-free -- **measured: repeated `np.savez` of the same
  arrays is byte-identical, and two full generation sweeps produced
  md5-identical files for every reference** (the regenerate-
  and-diff test's foundation, D8; md5-stable across numpy
  1.23.5/1.26.4/2.0.2 -- review-verified). Loadable with
  `np.load(..., allow_pickle=False)` (no object arrays; the string
  provenance is stored as fixed-width unicode arrays).
- **Per-scenario files, not one combined `.npz`** (flagged decision,
  plan 6.13): a single combined file would save ~8 registry lines
  and the ~5 kB/file zip overhead, but per-scenario files keep
  per-file md5 granularity -- a regenerate-and-diff or registry
  failure names the drifted scenario directly -- and the
  completeness quad-equality test (D6) pins set == table either
  way. The ~98 % overhead on ~250 B of result payload is accepted
  and recorded.
- Result arrays, exactly as stored in the datafile (D3), one row
  per scan point in row-major scan order (== `CrystalMap` order):
  `phi1`, `phi`, `phi2` (float32, Bunge radians), `metric`
  (float32), `iq` (float32), `phase` (uint8, all 0).
- Provenance arrays (the task's "attrs", realised as 0-d/1-d
  arrays). The drafting set: `emsphinx_commit` (full 40-hex sha,
  read from the checkout by the script -- not a constant), `bw`
  (68), `normed` (True), `refine`, `nregions`, `gausbckg`, `delta`,
  `vendor`, `route` (the canonical-route sentence), `dataset`
  (`"nickel_ebsd_small"` / `"nickel_ebsd_large_20pt"` /
  `"nickel_ebsd_large_165pt"`), `scan_shape` (`(rows, cols)`),
  `scan_steps`, `sample_tilt` (70.0), and **`pc`** -- float64
  `(3,)`, the kikuchipy-frame pc **the binary actually used**: the
  namelist's `.6g`-rounded values read back through
  `from_string(to_string(nml)).to_detector(sample_tilt=70).pc`
  (measured small Bruker: `(0.425139, 0.213367, 0.500707)`; large:
  `(0.42326, 0.213633, 0.502074)`; EMsoft route: the 1e-6-different
  triple above). **Added by the review** (the drafting set left
  silent-regeneration channels open -- a sibling `.sht`, changed
  preprocessing or writer defaults, a changed subset slice):
  - `namelist` -- the **exact namelist text the binary consumed**
    (~1.5 kB): the machine-checkable route pin. It carries
    vendor/delta/pctr/bw/normed/refine/nregions/gausbckg/circmask/
    thetac/patdims/scandims and the file names in one string, so
    no separate keys are needed for those;
  - `master_sht` (the in-package name) and `master_md5` -- the
    sibling `ni_20kv_bw384.sht` would otherwise yield a plausible
    but different reference;
  - `patterns_md5` -- md5 of the exact uint8 `/patterns` array the
    binary indexed (== the bytes of `signal.data.reshape(-1, h,
    w)`, D4): names the input in IQ-band failure messages (D5);
  - `preprocessing` -- the recipe sentence
    (`remove_static_background()` + `remove_dynamic_background()`,
    kikuchipy defaults) whose silent default change would
    invalidate every reference and band;
  - `subset_slice` (`"::15,::15"` / `"::5,::5"` / `""`),
    `emsphinx_compatible` (True), `manufacturer` (`"EMsoft"`),
    `flip` (False), `kikuchipy_version`.
  CI tests set `detector.pc = ref["pc"]` so both engines see the
  same rounded geometry (D5). **No timestamps or hostnames inside
  the files** -- bitwise reproducibility beats build metadata; the
  generation date lives in `validation.md` and the git history.
- The drafting candidates measured in `validation.md` predate the
  final key set (no `gausbckg`, none of the review-added keys), so
  the drafting md5s are provisional; the shipped md5s are pinned
  when the script regenerates on the branch. The result arrays are
  unaffected (bitwise determinism, D4).
- Budget: the constitution's rule is **"each < 100 kB"**
  (tech-stack "Tests, docs, data") -- asserted **per file**, with
  the total recorded via `record_property` (drafting candidates
  summed to 50,450 B over nine; the final eight with the extended
  key set are estimated ~60 kB total; corrected 2026-09-03 at
  implementation: **measured 214,027 B total**, see the correction
  in Scope). Registry: eight
  `"emsphinx/regression_*.npz"` md5 entries in `_registry_hashes`,
  no `_registry_urls` entries (in-package, the `.sht` precedent).

### D3 -- Parsed source: the datafile HDF5, not the `.ang` (frozen)

- The script parses **`Scan 1/EBSD/Data/{Phi1, Phi, Phi2, Metric,
  IQ, Phase}` of the namelist `datafile`** (`out.h5`). Measured
  dtypes: Phi1/Phi/Phi2/Metric/IQ float32, **Phase uint8** --
  correcting Phase 9 D7's recorded "float32" for Phase (research
  addendum, plan 0.4).
- Why not the `.ang` (measured on the canonical route, all
  scenarios; decimal counts corrected by the review against
  `tsl.hpp:783-794` -- `std::fixed` with `setprecision(5)` Euler,
  `setprecision(1)` iq, `setprecision(3)` ci):
  - `.ang` Euler columns are **5-decimal** fixed text: measured max
    |ang - h5| = 4.6e-6 to 5.0e-6 rad, the worst case exactly the
    half-ULP bound 5e-6. The `.h5` is the full float32 source.
  - the `.ang` "ci" column is the **Metric rounded to 3 decimals**
    (measured max |ci - Metric| = 4.0e-4 to 5.0e-4, bound 5e-4) --
    so the metric column *is* the normalised cross-correlation
    score, now measured (the task's open question). Small-map
    refined range 0.5749-0.6669, matching the Phase 9 acid scores
    (mean 0.6283).
  - the `.ang` "iq" column is the **IQ rounded to one decimal**
    (unique values {0.2} on the small map, {0.1, 0.2} on the large,
    {0.3} at `nregions` 0) -- which retroactively explains Phase
    9's "iq column constant 0.2" observation. Useless for
    regression; the `.h5` IQ is the real DCT image quality.
- The `.ang` is still written (`vendor_file="out.ang"`) and used as
  a **generation-time cross-check** (script asserts |ang - h5|
  Euler <= 1e-5 rad, |ci - Metric| <= 1e-3 -- each tolerance
  exactly 2x its deterministic half-ULP rounding bound), read with
  the Phase 9 `read_ang` convention (`np.loadtxt(comments="#")`).
  The frozen Phase 9 fixture is not abandoned -- it keeps serving
  the gated acceptance tests and this cross-check; the *reference
  payload* comes from the `.h5` (flagged decision, plan 6.3).

### D4 -- `create_emsphinx_reference.py`: contract (frozen)

- Location `src/kikuchipy/data/emsphinx/` next to
  `create_emsphinx_sht_fixtures.py` (the run-once-output-committed
  precedent); already excluded from the doctest job by the existing
  `--ignore-glob=src/kikuchipy/data/emsphinx/*.py`; added to the
  new coverage `omit` (D9). **Import-safe module**: importing it
  performs no env-var lookup, no binary probe and no HDF5 open --
  everything lives inside `main()` and its helpers -- so the CI
  completeness test can import the scenario table (D6).
- `main(output_dir=None, program=None)` under an
  `if __name__ == "__main__":` guard (default output: the script's
  own directory, i.e. the shipped location). `program` is the
  `IndexEBSD` executable path; when `None`, `main` resolves it via
  **`KIKUCHIPY_EMSPHINX_DIR`** (the same env var as the gated
  tests; `build/Release/IndexEBSD.exe` or bare `IndexEBSD` for
  non-Windows). The gated regenerate-and-diff test passes the path
  it got from the conftest `emsphinx_program` fixture instead --
  the fixture's "not built" skip lives inside that callable, so
  the test must call it (plan 3.1). `main()` itself **acquires the
  same machine-wide lock file the conftest uses**
  (`<tempdir>/kikuchipy-emsphinx-program.lock`, stdlib
  `O_CREAT|O_EXCL` with the stale-takeover rule), so the documented
  manual invocation serialises against a concurrently running gated
  suite -- the drafting spec left the manual path unlocked (review
  finding).
- Hard-asserts `git -C <dir> rev-parse HEAD` ==
  `60f351741036c63a59a6061a7ac2fca4f60f2c64` before any run; the
  stored `emsphinx_commit` is the *probed* value, so a stale pin or
  a wrong checkout both die at generation. NB the CI test
  `test_provenance_pins` compares `ref["emsphinx_commit"]` against
  the **test module's own pin literal** -- probed value and script
  pin are equal by construction, so that CI check has power only
  because the test module maintains its literal independently of
  the script's (stated in the test's comment; plan 5.2).
- Per scenario, in a temporary run directory: repack with
  `write_emsphinx_patterns` (all defaults = the canonical route),
  **guard-assert `/patterns` dtype uint8 and bytes ==
  `signal.data.reshape(-1, h, w)`** -- the repack flattens the
  navigation shape, so the comparison needs the reshape (measured:
  `array_equal` is False against the raw `(rows, cols, h, w)` data
  and False against a flipped reshape); record `patterns_md5` and
  `master_md5` into the provenance (D2); copy the in-package
  `.sht`; build the namelist with `from_kwargs(...,
  data_file="out.h5", vendor_file="out.ang", n_thread=1,
  batch_size=1, bandwidth=68, normalize=True, ...)` and store its
  `to_string()` text verbatim as `ref["namelist"]`; run the binary
  with `cwd=<rundir>`, require exit 0; parse per D3; run the `.ang`
  cross-check; write the `.npz` per D2.
- Generation-time acceptance guards (each was measured; they catch
  route/geometry mutants before a bad reference is ever written):
  - every small-map scenario: refined-or-coarse vs the stored
    `nickel_ebsd_small` xmap **median < 1.2 deg** (the Phase 9 acid
    band; a wrong flip pairing measures ~39.6, a non-uint8 repack
    ~38.9);
  - `small_coarse_nr10` vs `small_refined_nr10`: IQ **bitwise
    equal**, refined metric per-point above coarse (measured deltas
    +0.0032 to +0.0187).
- Determinism, measured -- and **qualified by the review**: with
  `nthread=1 batchsize=1` the binary is bitwise reproducible
  run-to-run **given a fixed FFTW wisdom state**. The programs plan
  with `FFTW_PATIENT` (`square_sht.hpp:384`, `sht_xcorr.hpp:366`)
  and import/export the machine-wide
  `getSharedDataDir() + "fftw.wisdom"` at start/exit
  (`util/fft.hpp:71-108, 281-330`); PATIENT picks the algorithm by
  timing, and different algorithms round differently -- so the bit
  pattern is a function of the wisdom, and the drafting reruns were
  bitwise identical because the wisdom was warm and unchanged (its
  md5 is recorded beside the reference md5s in `validation.md`).
  The script prints the wisdom file's md5 before and after the
  sweep; the regenerate-and-diff failure message names a changed
  wisdom as suspect #1 (D8). The wisdom race remains the
  concurrency rationale: the script runs its scenarios
  sequentially in one process and takes the machine-wide lock
  (above).
- Runtime, measured (this machine, drafting matrix): **10.8 s
  end-to-end** for the full sweep + the determinism rerun (index
  times 0.063-0.082 s small at 109-143 pat/s, 0.18 s large-20,
  1.4 s large-165); the final matrix swaps two small scenarios in
  and out (~unchanged cost).
- Licence: **kikuchipy GPL header only** (plan 6.2): the script
  executes the binaries and calls Phase 9's kikuchipy API; no
  EMSphInx expression is transcribed. The docstring embeds the
  generation recipe and the EMSphInx commit **inline** and cites no
  `specs/` path -- `specs/` is fork-only and stripped upstream, so
  a `src/` reference would dangle (the
  `create_emsphinx_sht_fixtures.py` precedent; review finding).
  The `.npz` files are program output with provenance arrays, not
  licence-stamped source.

### D5 -- The CI ours-vs-theirs harness (frozen)

- Inputs reconstructed without any binary artefact:
  - patterns: background-removed `nickel_ebsd_small` (resp. the
    large subsets per D1) -- on the canonical route the repack is
    **byte-identical to `signal.data.reshape(-1, h, w)`** (D4), so
    indexing the signal *is* indexing the repacked file;
  - harmonics: `MasterPatternHarmonics.from_file` on the in-package
    `.sht`, indexed at `bandwidth=68` -- the resize-from-the-stored-
    bw-384 semantics that `IndexEBSD` itself uses (`idx.hpp:182`;
    research item 39 forbids building bw-68 spectra directly).
    Built once per test process in a **session-scoped fixture**:
    under `pytest -n 4` (`--dist load`) each xdist worker rebuilds
    it once, and the `from_file` + resize is the dominant fixed
    cost (plan 6.8's budget);
  - detector: `signal.detector.deepcopy()` with
    **`detector.pc = ref["pc"]`** (the `.6g`-rounded triple the
    binary used, D2) -- not the unrounded `pc_average`; the ~3e-4 px
    difference is sub-band, so the choice is pinned by the
    provenance-recompute test, not by the bands;
  - `sample_tilt` binding: the detector's 70.0, the harmonics'
    `.sht`-derived `sample_tilt` and `ref["sample_tilt"]` are
    asserted equal in one named test -- the Phase 6 binding guard's
    regression-suite face (a mismatch indexes ~4.7 deg wrong at
    higher scores).
- Indexing kwargs are derived from the reference's provenance by
  one shared helper (`bandwidth=int(ref["bw"])`,
  `normalize=bool(ref["normed"])`, `refine`, `n_regions`,
  `gaussian_background`, `circular_mask=False`,
  `emsphinx_compatible=bool(ref["emsphinx_compatible"])`), and the
  helper's output per scenario is pinned against a frozen table in
  the test module -- the structural killer for provenance-ignoring
  mutants, because the refine flag has no IQ signature and the
  measured cross-comparisons stay inside plausible bands
  (validation "Recorded results": ours-coarse vs the refined ref
  medians 0.43, ours-refined vs the coarse ref 0.58 -- both
  band-survivors).
- Comparisons per scenario (all measured, bands in D6):
  - orientations: **both sides `Orientation(..., symmetry.Oh)`** --
    `Orientation(ours.data, Oh).angle_with(Orientation(
    Rotation.from_euler(np.stack([phi1, phi, phi2],
    1).astype(np.float64)).data, Oh), degrees=True)`. A bare
    `Rotation` on the theirs side raises `AttributeError` on orix
    0.14.2 (review-measured); the stored angles are EMSphInx
    `qu2eu` Bunge **radians** (`orientation_map.hpp:673-686`), so
    `from_euler` takes its defaults (no `degrees=`, default
    direction -- the construction already used in `src/` and on
    the orix 0.12.1 oldest job);
  - scores: `xmap.scores` vs `metric` -- Pearson r plus mean/max
    absolute difference. The two are on the same scale (both
    normalised) but **not equal**: measured mean |diff|
    0.0088-0.0139, max 0.0226-0.0364, r 0.935-0.973 -- so
    correlation-plus-band, not equality (re-stating the task's
    premise with the measurement: it is not that our score is
    un-normalised, it is that the two normalised metrics differ by
    ~2 % systematically);
  - IQ: `xmap.prop["iq"]` vs `iq` -- **near-equality**: measured max
    |diff| 6.6e-9 to 1.4e-8 across all scenarios (float32
    quantisation of the stored value). IQ is computed from the
    *processed* pattern, so it is the sharp discriminator for the
    preprocessing axes: `nregions` 0/7/10 and gausbckg move the IQ
    range by 1.5e-2 to 1.3e-1 (0.289-0.327 at nr0, 0.18565-0.22098
    at nr7, 0.187-0.216 with gausbckg, 0.173-0.204 at the
    default), while the misorientation gaps between those
    scenarios are only 0.03-0.09 deg. **The IQ band is what makes
    the nregions/gausbckg scenarios discriminating** -- but it is
    bound to the exact uint8 pattern bytes, whose numba-`fastmath`
    background removal can drift by 1-2 ULP across CI hosts: the
    band and its ladder live in D6, and the IQ assertion message
    names `ref["patterns_md5"]` so a platform drift fails as "your
    background-removed patterns differ from the reference
    machine's", not as an unexplained IQ miss (review finding);
  - **stretch-emulated decomposition diagnostic** (new, the D7
    evidence pin): one extra run of the refined anchor with the
    research-item-31 stretch emulated in pc space
    (`pc' = (0.5/w + pc_x (w-1)/w, 0.5/h + pc_y (h-1)/h,
    pc_z (w-1)/w)`) asserts the median misorientation vs the same
    reference **collapses** (< 0.2 deg -- the original criterion-2
    number -- and < the unmodified anchor's median; measured
    0.0940 vs 0.3404), with both values recorded via
    `record_property`.
- Weekly/default split: small scenarios + large-20 in the default
  suite (large-20 behind `importorskip("pooch")` +
  `allow_download=True`, the existing convention), large-165 behind
  `@pytest.mark.weekly`.

### D6 -- Measured agreement and the pinned bands (frozen)

Measured 2026-09-03 (binaries @ 60f3517, this machine; full tables
in `validation.md`), bands with the Phase 6 margin convention
(~1.7-2.1x on the measured worst):

| scenario | mis median (meas) | mis max (meas) | score r (meas) | assert |
|---|---|---|---|---|
| small coarse nr10 | 0.510 | 0.622 | 0.941 | median < 1.0, >= 8/9 < 1.25, all < 4.0, r > 0.85 |
| small refined nr10 | 0.340 | 0.364 | 0.951 | median < 0.7, max < 0.75, r > 0.85 |
| small refined nr0 | 0.310 | 0.335 | 0.969 | same refined bands |
| small refined nr7 | 0.335 | 0.367 | 0.935 | same refined bands |
| small refined nr10 gb | 0.341 | 0.357 | 0.957 | same refined bands |
| small refined emsoft d500 | 0.340 | 0.364 | 0.951 | same refined bands |
| large20 refined | 0.325 | 0.373 | 0.973 | median < 0.7, max < 0.8, r > 0.90 |
| large165 refined (weekly) | 0.339 | p95 0.441 / max 0.487 | 0.944 | median < 0.7, p95 < 0.9, max < 1.0, r > 0.88 |

- **Coarse outlier clause** (review finding): at bw 68 one
  correlation-grid cell is 360/`fast_size(135)` = 2.667 deg, larger
  than the whole coarse band -- a single argmax landing one cell
  over on another CI platform is a legitimate near-tie, so the
  coarse scenario uses the Phase 6 convention (>= 8/9 under the
  tight max, all under a single-cell-jump ceiling of 4.0 deg =
  measured max + one cell, ~1.9x). (corrected 2026-09-03 at
  implementation: the arithmetic is wrong. Measured max + one cell
  is 0.622 + 2.667 = **3.29** deg, not 4.0, and 4.0 deg is **6.4x**
  the measured max, not 1.9x. The ceiling stays 4.0 -- that sum
  rounded up to a round number -- and the constant's comment in the
  test module now says so.) The refined scenarios keep plain
  maxima: their >= 2x margins stand, and Newton refinement
  re-converges a near-tie.
- Scores, all scenarios: mean |ours - metric| < 0.03 (measured
  0.0088-0.0139), max |ours - metric| < 0.07 (measured
  0.0226-0.0364).
- IQ, all scenarios: max |ours - iq| < **1e-3**. The ladder
  (review-measured, replacing the drafting 1e-6 knife-edge): parity
  on this machine <= 1.4e-8; but the reference IQ is bound to the
  exact uint8 pattern bytes, and **one gray level in one pixel
  shifts that pattern's IQ by up to 5.2e-5** (24-trial scan;
  typical 2e-6-4e-5, some zeros -- the AHE histogram cascade), the
  background removal is numba `fastmath` float32 with ~4 pixels per
  map within 4 ULP of the truncation boundary, so the realistic
  cross-platform drift budget is ~2e-4; the smallest *real*
  signature the band must catch is 1.5e-2 (gausbckg vs default;
  nregions mutants >= 2.27e-2). 1e-3 sits ~5x above the drift
  budget and >= 15x below the smallest kill -- and the assertion
  message names `ref["patterns_md5"]` (D5). NB the review critic
  proposed 1e-4 from a measured 4.04e-6 one-pixel sensitivity; the
  revision re-measured the sensitivity 12x worse (5.2e-5 worst of
  24), which forces the wider band -- recorded in `validation.md`.
- **Route pins (CI, real kikuchipy code -- the tests that actually
  exercise the conversion)**:
  - `test_namelist_matches_the_declared_route`:
    `EMSphInxNamelist.from_kwargs(**<frozen table>)`.`to_string()`
    equals `ref["namelist"]` **exactly** per scenario -- one string
    comparison pinning vendor, delta, pctr, bw, normed, refine,
    nregions, gausbckg, circmask, thetac, patdims/scandims and the
    file names (kills a wrong-vendor or wrong-delta generation and
    every kwargs-table drift);
  - `test_pc_matches_the_stored_namelist`:
    `from_string(ref["namelist"]).to_detector(sample_tilt=70).pc`
    equals `ref["pc"]` -- exact float64 equality, which is safe
    **only** because `to_string` quantised at `.6g` before the
    round trip (say so in the test comment, lest the next reader
    loosen it for the wrong reason; review finding). Small
    scenarios in CI; the large recompute lives with the
    pooch-gated test.
- **Reference integrity guards** (CI, refs only -- frozen bytes vs
  frozen bytes: they exercise no kikuchipy code and **can only fire
  at regeneration or if the shipped data is edited**; labelled so
  in the class docstring and not counted as parity surface --
  review finding):
  - `emsoft_d500` vs `refined_nr10`: orientation gap max < 1e-3 deg
    (measured 5.83e-5) and metric arrays **not** identical (the
    routes differ in the 6th decimal of pc);
  - `coarse_nr10` vs `refined_nr10`: `iq` bitwise equal (same
    preprocessing), orientations not identical, and
    `refined metric - coarse metric > 0` per point (measured
    +0.0032..+0.0187) -- the drafting upper bound 0.02 is
    **dropped**: it had 7 % headroom on the measured worst
    (+0.018680), violating the spec's own margin convention, and
    only strict positivity is meaningful (review finding);
  - `refined_nr10` vs `nr0`/`nr7`/`nr10_gb`: `iq` arrays pairwise
    **not** equal (the preprocessing signature; ranges above);
  - provenance pins: `emsphinx_commit` == the test module's own
    40-hex literal (D4 caveat), `bw` 68, `normed` True,
    `sample_tilt` 70.0, `phase` all zero, per-scenario
    refine/nregions/gausbckg/vendor/delta/dataset vs the frozen
    table;
  - **completeness quad-equality** (review finding): {the script
    module's scenario names} == {the `regression_*` registry keys}
    == {the `regression_*.npz` glob of
    `Path(kikuchipy.data.__file__).parent / "emsphinx"`} == {the
    test module's frozen table} -- so adding or renaming a scenario
    without regenerating and registering fails on CI, not only on a
    machine with binaries. The glob anchor is the installed package
    directory, **not** `src/` -- the `-build-install-wheel` CI job
    runs the suite from the installed wheel where `src/` does not
    exist (review finding).

### D7 -- Re-anchoring the a-priori parity numbers (recorded amendment)

Mission success criterion 2 and the pre-Phase-5 roadmap carried
guessed gates: *refined median < 0.2 deg, coarse median < 0.5 deg,
scores Pearson r > 0.98*. Measured on identical inputs at `bw` 68
(this phase, eight scenarios, 9-165 points): refined median
0.31-0.34, coarse median 0.51, r 0.935-0.973.

**The residual is now decomposed by measurement** (review finding --
the drafting draft mis-attributed it to "peak-interpolation and
Newton-path differences" plus two common-mode terms; the
re-measurements are in `validation.md`, dated review section).
Emulating EMSphInx's deliberately un-ported `bilinearCoeff` pixel
stretch (research item 31: `x = X (w-1)`; its exact pc-space
equivalent is `pc' = (0.5/w + pc_x (w-1)/w, 0.5/h + pc_y (h-1)/h,
pc_z (w-1)/w)`) collapses the disagreement to its floor:

| scenario | as shipped (med/max deg) | stretch-emulated |
|---|---|---|
| small coarse | 0.5096 / 0.6219 | 0.0745 / 0.1288 |
| small refined | 0.3404 / 0.3643 | 0.0940 / 0.1166 |
| large20 refined | 0.3248 / 0.3725 | 0.0784 / 0.1429 |
| large165 refined | 0.3390 / 0.4867 | 0.0717 / 0.2190 |

A scale scan has its sharp minimum exactly at the predicted
`f = (w-1)/w = 0.98333` (0.975 -> 0.507, 0.98 -> 0.263,
**0.98333 -> 0.094**, 0.99 -> 0.328, 1.0 -> 0.828 deg median). So
~75-85 % of the ours-vs-theirs residual is the documented item-31
sampling-convention difference, and the remaining **~0.07-0.09 deg
median is the true peak-interpolation/Newton floor** -- under the
original < 0.2 gate. **The port keeps kikuchipy's convention**: the
emulation *worsens* agreement with the stored xmap (large20 0.3653
-> 0.5838, large165 0.3800 -> 0.5761 deg median), i.e. kikuchipy's
convention is the physical one and the stretch is the EMSphInx
quirk (exactly as tech-stack's Numerics bullet recorded a priori).

The amendment (plan 0.3, applied in the spec commit): criterion 2's
numbers become the measured-then-pinned bands of D6 **with the
decomposition recorded beside them** -- the original < 0.2 gate is
stated as met under item-31 emulation, pinned by the shipped
diagnostic test (D5), not discarded as unachievable. Supporting
facts kept from drafting: IQ agrees to 1.4e-8 (the whole
preprocessing + back-projection front end is float32-identical),
the scores agree to ~2 % with r ~0.95, and the binary's own two
correct flip routes differ from each other by 0.01-0.02 deg.
Flagged prominently for user review -- this edits `mission.md`
(plan 6.1).

### D8 -- Theirs-on-ours: what Phase 10 adds (frozen)

Phase 9's local-gated suite already proves acceptance ("their
binaries on our files"): the acid test, its negative controls, the
`PatternRepack.exe` bitwise pins and the namelist template parity.
Phase 10 adds exactly one gated surface, **regenerate-and-diff**:

- `TestRegenerateReferences` (local-gated on
  `KIKUCHIPY_EMSPHINX_DIR`, holding the conftest binary lock via
  the `emsphinx_program` fixture and passing its resolved
  executable into `main(program=...)`, D4): import
  `create_emsphinx_reference`, call `main(tmp_path, program=...)`
  for the small-map scenarios, and assert every regenerated file is
  **md5-identical** to the shipped one (measured foundation: two
  full sweeps were md5-identical on all files, under an unchanged
  FFTW wisdom). On mismatch the test reports, in order: (1) the
  FFTW wisdom state as suspect #1 (`FFTW_PATIENT` + the machine-
  wide wisdom make the bit pattern a function of the wisdom, D4),
  (2) the per-array diff (`np.load` both, compare
  keys/dtypes/bytes) -- the array-level fallback is diagnostic,
  not an acceptance tolerance. The large-map regeneration (needs
  pooch + ~2 s of indexing + the full-map background removal)
  carries `@pytest.mark.weekly` on top of the gate.
- Rationale: bitwise regeneration is simultaneously the strongest
  possible theirs-on-ours acceptance (the binary still accepts and
  identically indexes our files) and the guard that the shipped
  refs match the pinned commit. If a future machine, FFTW build,
  **FFTW wisdom state** or **numpy `.npy`/zip-writing change**
  (review addition) breaks bitwise reproduction, the recorded
  fallback is to re-derive tolerance bands from that machine's
  measurements and amend this spec -- not to loosen silently
  (plan 6.6).

### D9 -- Placement, licences, exports, CHANGELOG (frozen)

- New files: `src/kikuchipy/data/emsphinx/create_emsphinx_reference.py`
  (GPL kikuchipy header only, D4), eight
  `src/kikuchipy/data/emsphinx/regression_*.npz`,
  `tests/test_indexing/test_spherical_emsphinx_regression.py`
  (kikuchipy header). Changed: `src/kikuchipy/data/_registry.py`
  (eight md5 lines); **`pyproject.toml`** (`[tool.coverage.run]`
  gains `omit = ["src/kikuchipy/data/*/create_*.py"]` -- verified:
  no `omit` exists today and `source = ["src/kikuchipy"]` puts the
  never-imported generation scripts at 0 %, moving the codecov
  project status; the drafting claim that the existing fixtures
  script "is excluded from coverage" was wrong -- review finding);
  **`tests/test_indexing/test_spherical_namelist.py`** (the
  delta-invariance conversion pin, D1);
  **`src/kikuchipy/signals/ebsd.py`** and
  **`src/kikuchipy/indexing/_spherical/_indexer.py`** -- two
  `Notes` sentences each on `EBSD.spherical_indexing` /
  `SphericalIndexer`: agreement with `IndexEBSD` @ 60f3517 on
  identical inputs at bw 68 (refined median ~0.34 deg -- ~0.09
  after emulating EMSphInx's documented detector-sampling stretch
  -- scores r ~0.95, image quality equal to float32), enforced by
  the shipped regression references (review finding: `specs/` is
  stripped from upstream PRs, so nothing shippable would otherwise
  carry the numbers Phase 11 cites). No `specs/` path, per the
  `src/` rule. No `__init__.pyi` change, no docs pages.
- **No CHANGELOG entry** (flagged autonomous decision, plan 6.1):
  the roadmap gate rule exempts phases with no user-facing change;
  this phase ships un-exported test data, a maintainer script,
  tests and two docstring `Notes` sentences (documentation of
  measured behaviour, deliberately below the CHANGELOG bar).
  Mission criterion 5's blanket "a CHANGELOG entry" is reconciled
  **in the same spec commit** by carrying the roadmap's exemption
  into the criterion (plan 0.3; review finding -- the drafting
  draft shipped a documented contradiction). The PR (#11)
  description carries the rationale; if review disagrees, the
  fallback is a single `Added` line naming the regression suite.
- Tests import the refs via
  `Dataset("emsphinx/regression_<name>.npz").fetch_file_path()`
  (the Phase 2 `.sht` convention; in-package files need no pooch)
  and `np.load(..., allow_pickle=False)`.
- No new dependencies, no numba kernels, no `scipy.fft`; the oldest
  CI job needs no version gates (npz IO is numpy-stable; h5py is
  not needed by the CI tests at all -- only the script and the
  gated tests touch HDF5).
- PR **#11** into fork `develop`; the GPL statement is unchanged
  from prior phases (the new script is kikuchipy-only, D4).

### D10 -- Mission criterion 2 tie-in (recorded)

With this phase, every clause of criterion 2 is delivered and
measured: agreement with `IndexEBSD.exe` @ 60f3517 on kikuchipy's
Ni datasets is now a shipped, CI-enforced regression (D5/D6, at the
re-anchored bands of D7 with the item-31 decomposition), and "a
kikuchipy-written `.sht` and a kikuchipy-repacked pattern file are
accepted by the EMSphInx binaries" is enforced by Phase 2/9's gated
tests plus this phase's regenerate-and-diff (D8). The roadmap Phase
10 box rewrite (plan 0.1) records the measured numbers.

**Phase 11 hand-off caveat** (review finding, recorded here so the
tutorial does not inherit a mismatch): the shipped parity numbers
are **bw 68 against the in-package `ni_small_20kv_bw384.sht`**;
Phase 11 plans the full `ebsd_master_pattern("ni")` at bw <= 190,
where these numbers do not transfer. The tutorial must either
reproduce the reference configuration for its parity claim or scope
the citation ("median ~0.34 deg at bw 68, enforced by the shipped
regression suite" -- the D9 docstring `Notes` are the citable,
shippable carrier).

## Context

- Constitution: `specs/mission.md` (success criterion 2 -- amended
  per D7/plan 0.3; criterion 5 -- CHANGELOG exemption reconciled,
  D9), `specs/tech-stack.md` (Tests/docs/data bullet: in-package
  `.npz` refs **each < 100 kB**, md5 in `_registry.py`, no URL,
  `nthread=1 batchsize=1` -- all honoured, the per-file rule now
  asserted per file; Numerics bullet: the item-31 stretch
  non-port, now with its measured consequence; amended per plan
  0.2), `specs/roadmap.md` Phase 10 box (the recorded constraints:
  uint8-only regression files, the canonical route -- honoured; the
  EMsoft-vendor delta sweep and the `nregions` {0, 4, 10} set --
  **amended by measurement**, plan 0.1).
- Research: `specs/_research/explore-emsphinx-core-algorithm.md`
  section 8 items **31** (the `bilinearCoeff` stretch -- D7's
  decomposition), 39 (resize-from-stored-bw parity -- D5), 40
  (refine-only outputs forbidden -- Scope), 44-47 (Manufacturer,
  uint8-only, flip/vendor decoupling, canonical route -- D1/D4);
  `explore-emsphinx-programs-and-formats.md` section 7 (output
  HDF5 -- corrected by D3's Phase-dtype and column-precision
  measurements).
- Phase 9 deliverables composed: `write_emsphinx_patterns`,
  `EMSphInxNamelist` (`from_kwargs`/`to_detector`/round trip),
  the conftest `emsphinx_dir`/`emsphinx_program` (binary lock) and
  `read_ang` fixtures, the recorded anchors
  (`specs/2026-09-02-sht-interop/validation.md`): default-route
  acid median 0.7245/max 0.9479, scores mean 0.6283;
  kikuchipy-vs-IndexEBSD context 0.341/0.363 deg, r 0.9607 on the
  Bruker route -- re-baselined on the canonical route this phase
  (measured: 0.340/0.364, r 0.951 -- the Phase 9 context numbers
  carry over almost unchanged).
- Phase 2 fixture: `emsphinx/ni_small_20kv_bw384.sht`
  (`sample_tilt` 70.0) -- the master of every scenario, now pinned
  per reference by `master_sht`/`master_md5` (D2).
- Existing suite conventions reused: `misorientation` via
  `Orientation.angle_with` (never `reduce()`), measured-then-pinned
  bands, `record_property` for recorded-only values, weekly marker,
  `importorskip("pooch")`, the `[::15, ::15]`/`[::5, ::5]` large
  subsets of `test_spherical_refinement.py`.
- Downstream: Phase 11's tutorial cites the regression suite as the
  parity evidence **subject to the D10 configuration caveat**; the
  deferred Phase 8 adds no scenario here (roadmap re-scope).
