# Phase 9 -- `sht-interop`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_namelist.py tests/test_indexing/test_spherical_pattern_repack.py tests/test_io/test_oxford_binary.py -n 4
uv run pytest tests/test_indexing tests/test_signals tests/test_io -k spherical -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical
uv run pre-commit run --files <changed files>
uv run sphinx-build -b html doc doc/_build/html
uv run sphinx-build -b linkcheck doc doc/_build/linkcheck
```

Everything below runs on CI (no EMSphInx binaries) except the
`KIKUCHIPY_EMSPHINX_DIR`-gated section.

`tests/test_indexing/test_spherical_pattern_repack.py`:

- `TestWriteEmsphinxPatterns`
  - `test_layout_is_contiguous_alloc_early`: written file's
    `/patterns` has `get_layout() == h5py.h5d.CONTIGUOUS`,
    `get_alloc_time() == h5py.h5d.ALLOC_TIME_EARLY`, **zero
    filters** (the functional guard -- compression is fatal to the
    reader, D2), no chunks, and `id.get_offset()` is a
    non-negative int.
  - `test_default_route_dataset_equals_signal_bytes`: default
    (`manufacturer="EMsoft"`, auto `flip=False`) `/patterns` bytes
    == `nickel_ebsd_small().data.reshape(9, 60, 60)` exactly
    (`np.array_equal` on the raw dataset).
  - `test_flip_true_reverses_rows`: `flip=True` dataset ==
    `data[:, ::-1, :]` (rows, not columns -- the column mutant
    dies here).
  - `test_manufacturer_auto_flip_table`: the frozen mapping
    `{EDAX: False, EMsoft: False, Oxford: True, Bruker: True,
    "Bruker Nano": True, "DREAM.3D": True}` -- asserted by writing
    each manufacturer and comparing dataset bytes to the expected
    orientation.
  - `test_manufacturer_is_a_root_scalar_vlen_dataset`: value
    equals the manufacturer string; dtype is a variable-length
    string **with ASCII cset** (`h5py.check_string_dtype(...)
    .encoding == "ascii"` / `ds.id.get_type().get_cset() ==
    h5py.h5t.CSET_ASCII` -- UTF-8 is the h5py default and is fatal
    to the binary, D1); dataspace scalar; it is a dataset at the
    file root (not an attribute/group).
  - `test_unknown_manufacturer_raises` (`"kikuchipy"`, `"tsl"`,
    `"TSL"` -- none are Manufacturer strings),
    `test_bad_dtype_raises` (int8, float64),
    `test_uint16_write_warns` + `test_float32_write_warns`
    (UserWarning naming the EMSphInx uint8-only read),
    `test_binning_must_divide` (ValueError).
  - `test_byteswapped_input_writes_native`: a `>u2` (and `>f4`)
    input signal produces a **native-order** dataset whose bytes
    equal the byteswapped-to-native data (the binary rejects
    big-endian dtypes outright -- measured, D1/D2).
  - `test_overwrite_false_leaves_file` /
    `test_overwrite_true_replaces` / `test_suffix_defaulting`
    (`filename` without suffix gains `.h5`; parent directory
    created): the kikuchipy writer conventions (H2 fix; mirrors
    `MasterPatternHarmonics.save`).
  - `test_navigation_orders`: 2-D `(2, 3)` nav row-major flatten;
    1-D and 0-d navigation shapes.
  - `test_lazy_write_equals_eager`: a rechunked lazy signal
    produces byte-identical `/patterns`;
    `test_lazy_write_does_not_materialise`: `tracemalloc` peak
    during the lazy write of a synthetic map stays under a bound
    well below the full-map bytes (Phase 6 pattern; the bound is
    measured on the tests-first run, then pinned here with the
    ~1.7x margin convention).
- `TestBinning`
  - `test_binavg_rounds_half_away_from_zero`: frozen fixture with
    exact-half block means (e.g. blocks summing to 2 with
    `binning=2`: mean 0.5 -> 1, where banker's gives 0); also the
    nickel bin-2 output equals the frozen NumPy recipe
    `floor(block_sum/4 + 0.5).astype(uint8)` (the recipe measured
    bitwise-equal to `PatternRepack.exe`; banker's differs on
    1003/8100 pixels).
  - `test_binavg_accumulates_in_float64` (blocks whose uint8 sums
    exceed 255), `test_binavg_binning_one_copies`.
  - `test_binfloat_sums_in_float32`: block **sum**, output float32,
    for uint8/uint16/float32 inputs; `binning == 1` casts to
    float32 without summing (the completed-dead-code contract,
    D1; NumPy pairwise summation is the defined accumulation
    order for float32 inputs).

`tests/test_indexing/test_spherical_namelist.py`:

- `TestNameListParser` (the `test/util/nml.cpp` port, D4)
  - `test_scalar_parsing`: bools `.true./.false.`, ints
    `12345/+12345/-12345`, doubles `1.2345/+1.2345/-1.2345/1.23e4`,
    strings incl. `'str \'with single quotes\''` and
    `'str "with single quotes"'`.
  - `test_vector_parsing`: scalar-as-vector length 1; bool/int/
    string lists; the mixed `1, 2, 3., 4` list promoted to all
    doubles `[1.0, 2.0, 3.0, 4.0]`.
  - `test_second_string_whitespace_stripped`: in a quoted list the
    first string keeps its spaces, the second and later have *all*
    whitespace removed (`'a b', 'c d e'` -> `["a b", "cde"]` --
    the sticky-`skipws` quirk, measured through the binary, D4).
  - `test_partial_parsing`: `fully_parsed()` true after all gets;
    the two-token file with one get -> `unused_tokens() ==
    "tokentwo"`.
  - `test_error_cases` (each raises -- the **eleven** `nml.cpp`
    cases): first-line kv, missing comma, missing leading space,
    duplicate key, missing `=`, missing string delimiter,
    string-then-number list, double quotes, unquoted string,
    int/bool mix, double/bool mix.
  - `test_error_messages_match_the_binary`: the messages probed
    through `IndexEBSD.exe` (Recorded results) appear verbatim:
    *"missing leading space in namelist line ..."* (zero spaces or
    tab), *"error parsing line '...' from name list"* (**two or
    more** leading spaces -- a different message, measured; also
    the whitespace-only line), *"key \"circmask\" was defined
    twice ..."*, *"namelist files cannot have key value pairs in
    the first line"*, *"missing comma between previous entry and
    namelist line ..."*, *"couldn't parse token ... (strings must
    be in single quotes, e.g. key = 'value')"*, *"bad delimeter
    (expected '=') ..."* (the indented comment), and the
    unused-token warning wording.
  - `test_indented_comment_is_not_a_comment`: `" ! text"` raises
    (the comment test is `line.front()`, column 0 only -- measured
    exit 1 `bad delimeter (expected '=') in namelist line 6 " ! an
    indented comment"`, D4);
    `test_column0_comment_is_skipped`: `"! text"` at column 0 is
    skipped.
  - `test_whitespace_only_line_raises`: a `"   "` line raises
    *"error parsing line '   ' from name list"* (measured -- the
    literal `line.empty()` port; `line.strip()` would diverge).
  - `test_get_int_rejects_doubles` /
    `test_get_double_accepts_ints` (C++ `:226-229` strictness).
  - `test_keys_are_case_insensitive` (lookup of `BW` finds `bw`).
- `TestNamelistTemplate`
  - `test_defaults_to_string_matches_index_ebsd_template`:
    `EMSphInxNamelist.defaults().to_string()` line-list equals the
    frozen 119-line `IndexEBSD.exe -t` capture (string constant
    with provenance comment; comparison after `splitlines()` --
    the capture is CRLF from the Windows binary, md5
    `49ddf0e7d9b2d758d918c20a7f900a6d`). Line 1 is the fixed
    ` &EMSphInx` (no name parameter in v1, D5).
  - `test_doubles_format_like_cpp_streams`: `50.0 -> "50"`,
    `1.5 -> "1.5"`, `0.42513885 -> "0.425139"`, `12345678.9 ->
    "1.23457e+07"` (the `.6g` rule).
  - `test_to_string_without_qualmap_has_no_terminator` (the
    `:464-468` quirk) and with qualmap the last line is `" /"`.
  - `test_write_rejects_spaced_second_master`: `master_files =
    ["a.sht", "b c.sht"]` -> `ValueError` from `to_string`/`write`
    (the parser would strip the space on read-back, D4/D5); a
    spaced *first* master writes fine.
- `TestNamelistRoundTrip`
  - defaults, the acid-test namelist **and a non-empty-`ipath`
    variant** survive `from_string(to_string())` field-for-field
    (raw path storage, D5 -- the C++ would double-prefix); reading
    the acid namelist yields its literal values (8-decimal pctr
    parsed in full).
  - `test_derived_paths`: `pat_path == ipath + pat_file`,
    `master_paths == [ipath + f, ...]`;
    `test_psymfile_double_ipath_quirk_is_ported`: with `psym_file`
    non-empty, `pat_path` carries `ipath` twice (the `:247` quirk,
    observable on the derived property).
  - `test_optional_keys`: missing
    `ipath/psymfile/opath/vendorfile/ipfmap/qualmap` parse to
    their empty defaults; missing `bw` (required) raises
    *"couldn't find `bw' in namelist"*.
  - `test_patdset_scanname_always_optional`: namelists without
    `patdset`/`scanname` parse regardless of whether `patfile`
    points at an existing HDF5 file (recorded deviation -- the C++
    requiredness is filesystem-dependent, measured both ways, D5);
    a present `scanname` with numeric `scandims` is consumed
    (no unused-token warning) and round-trips.
  - `test_scandims_three_or_four` (3 -> square steps), non-integer
    dims raise, string scandims -> `NotImplementedError`.
  - `test_vendor_whitelist` (`tsl` accepted, `TSL` rejected),
    `test_extra_key_warns_with_its_name`,
    `test_sanity_check_bounds` (parametrised: each of the
    **thirteen** checks probed just inside and just outside; the
    negativity bounds are live in the binary -- `nregions = -5` ->
    `unreasonable AHE nregions`, `nthread = -1` -> `negative
    thread count`, both measured).
- `TestVendorConversions`
  - `test_conversion_table_square_and_rectangular`: the frozen D6
    table on `(60, 60)`, `(48, 60)` **and `(60, 48)`**; measured
    reference triples asserted with `pytest.approx` (e.g. Bruker
    `(0.42513885, 0.21336699, 0.50070692)` <-> `cX -4.49166896,
    cY 17.19798032, sDst 15021.20746804` at `delta` 500 -- triple
    derived from the *unrounded* `pc_average`; recomputing from
    the rounded pc gives 17.1979806/15021.2076, hence approx, D6).
  - `test_emsoft_equals_kikuchipy_pc_emsoft_version_4`: exact,
    rectangular included, **on a detector constructed with
    `binning=1, px_size=delta`** -- the equality's precondition
    (kikuchipy multiplies by `_binning` and `px_size`; measured:
    the nickel fixture's `binning=8, px_size=1.0` breaks it, D6).
  - `test_tsl_oxford_deviate_from_kikuchipy_on_rectangular`
    (EMSphInx-TSL == kikuchipy `pc_oxford()`; EMSphInx-Oxford z !=
    kikuchipy `pc_tsl()` z -- the frozen deviation rows, asserted
    on both `(48, 60)` and `(60, 48)`; equal on square detectors).
    (corrected 2026-09-02: the *inequality* holds on `(48, 60)`
    only. Measured in the tests-first run -- on `(60, 48)`
    kikuchipy's `min(nrows, ncols)/nrows` factor reproduces
    EMSphInx's `h/w` scaling exactly, so EMSphInx-Oxford **equals**
    `pc_tsl()` there. The test is parametrised over the three
    shapes with the measured relation per shape; see the third
    dated section of "Recorded results".)
  - `test_delta_invariant_for_fractional_vendors`: two `delta`
    values give identical `to_detector().pc` for
    Bruker/tsl/EDAX/Oxford pctr (exact; measured bitwise through
    the binary -- Bruker delta 250 vs 500 `.ang` identical) and
    different pc for EMsoft (delta is live there, D6).
  - `test_pctr_pc_round_trip` for all four vendors.
- `TestToFromKwargs`
  - `test_kwargs_keys_are_live_parameters`: every `to_kwargs()`
    key is a parameter of `EBSD.spherical_indexing` (via
    `inspect.signature`) and of `SphericalIndexer.__init__` where
    applicable; values map `bw/normed/refine/nregions/gausbckg`
    correctly.
  - `test_positive_circmask_maps_to_false_with_warning`
    (`0 -> True`, `-1 -> False`, `> 0 -> False` + UserWarning
    naming the **lost processor-side mask** -- EMSphInx keeps a
    radius-`r` CircMask in the image processor while
    `Geometry::circ` stays false, `imprc.hpp:108-122` /
    `idx.hpp:230, 254`; kikuchipy has no fixed-radius knob, D6).
  - `test_batchsize_zero_maps_to_none_chunksize`.
  - `test_roimask_nonempty_raises` (out-of-scope grammar).
  - `test_to_detector_requires_sample_tilt` (TypeError without),
    `test_to_detector_fields` (shape `(h, w)`, pc, px_size=delta,
    tilt=thetac).
  - `test_from_kwargs_bruker_pctr_is_pc_average_verbatim`,
    `test_from_kwargs_delta_default_is_30mm_detector`
    (`30000/patdims[0]`; nickel detector px_size 1.0 must NOT leak
    in), `test_from_kwargs_passes_its_own_sanity_check`,
    `test_from_kwargs_rejects_azimuthal_twist`,
    `test_from_kwargs_patdims_order` (rectangular: `(ncols,
    nrows)`), `test_from_kwargs_scandims_order` (**non-square**
    navigation shape, e.g. `(2, 3)`: `scan_dims ==
    scan_shape[::-1]` -- x then y, `ebsd/nml.hpp:397` /
    `idx.hpp:238`; a square scan cannot catch the transposition),
    `test_from_kwargs_thread_batch_passthrough` (`n_thread=1,
    batch_size=1` land in the namelist -- the acid/Phase 10
    configuration), `test_from_kwargs_circular_mask_inverse`
    (`True -> circ_rad 0`, `False -> -1`),
    `test_from_kwargs_output_names_default_empty` (`vendor_file`/
    `ipf_name`/`qual_name` all `""` -- no surprise PNG outputs;
    the resulting `to_string` has no `" /"` terminator, which the
    parser accepts).

`tests/test_io/test_oxford_binary.py` additions:

- `test_get_scan_info_in_package_file`: `patterns.ebsp` -> 9/9
  patterns, `(60, 60)` uint8, 3600 pattern bytes, 32400 total,
  version 2, `beam_x == beam_y == [0.0, 1.5, 3.0]` (measured --
  the file's 1.5 um step), regular (the measured `EBSPDims.exe`
  facts: 3 x and 3 y coordinates).
- `test_get_scan_info_irregular`: the staggered fixture (beam_x
  `[0, 1, 2, 0.5, 1.5, 2.5]`, beam_y `[0, 1]`, written by the
  module-local helper -- the conftest fixture cannot express it,
  plan 3.2) -> 6 x / 2 y, `is_regular_grid` False, sorted unique
  lists equal the measured `X: 0 0.5 1 1.5 2 2.5` / `Y: 0 1`.
- `test_get_scan_info_exact_value_distinctness`: `1.0` vs
  `1.0 + 1e-12` count as two coordinates (the `std::set<double>`
  semantics; module-local helper file).
- `test_get_scan_info_version0_has_no_beams` (`beam_x is None`,
  regular False), `test_get_scan_info_not_all_present`
  (present-count regularity; `all_patterns_present` False),
  `test_get_scan_info_key_set` (frozen dict keys) -- these three
  on the existing conftest `oxford_binary_file` variants,
  unchanged.

## Local-gated (KIKUCHIPY_EMSPHINX_DIR; skipped on CI)

Class `TestAgainstEmsphinxBinaries` in
`test_spherical_pattern_repack.py` (plus the template test in
`test_spherical_namelist.py`), using the shared `emsphinx_program`
and `read_ang` conftest fixtures (promoted from
`test_spherical_sht_file.py:160-190`, D9). **Every invocation is
`subprocess.run([...], cwd=tmp_path, ...)`** -- `IndexEBSD.exe -t`
writes to a hard-coded relative path and all namelist paths resolve
against the cwd (D9).

- `test_index_ebsd_accepts_kikuchipy_repack` -- **the acid test**,
  on the **canonical default route**: write the background-removed
  `nickel_ebsd_small` repack (`manufacturer="EMsoft"`, unflipped),
  the `from_kwargs` namelist (`bw` 68, defaults, `n_thread=1,
  batch_size=1`, Bruker pctr = `pc_average`, delta from the 30 mm
  rule (= 500), thetac 0, scandims 3,3,1.5,
  `vendor_file="out.ang"`) and the in-package
  `ni_small_20kv_bw384.sht` into `tmp_path`; run
  `IndexEBSD.exe <nml>` with `cwd=tmp_path`; assert exit 0, then
  `read_ang` the `.ang`: refined-vs-stored-xmap **median < 1.2
  deg**, **max < 1.6 deg** (measured on this route 0.7245 /
  0.9479 -- margins 1.66x / 1.69x, Phase 6 convention), scores
  mean `pytest.approx(0.628, rel=0.05)` (measured 0.6283 on this
  route; the earlier 0.713/0.947/0.6304 anchors belong to the
  Bruker-flip route and stay recorded, not asserted); runtime and
  per-point values `record_property`ed, never asserted (recorded
  112-120 pat/s refined across runs; coarse 135.7 pat/s measured
  on the Bruker route).
- `test_wrong_flip_pairing_is_discriminated`: the same run with
  `flip` forced wrong -> exit 0 but median **> 10 deg** (measured
  ~39.6) -- the flip contract is observable, not cosmetic.
- `test_index_ebsd_rejects_missing_manufacturer`: a
  Manufacturer-less twin -> non-zero exit, stderr/stdout contains
  `doesn't have a Manufacturer string`.
- `test_index_ebsd_rejects_unknown_manufacturer`: `"kikuchipy"`
  -> non-zero exit, `unknown EBSD vendor`.
- `test_vendor_namelists_are_equivalent`: EMsoft/EDAX/Oxford
  namelists from the D6 conversions -> `.ang` Euler columns
  bitwise-equal to the Bruker run (measured max |diff| 0.0).
- `test_pattern_repack_binary_parity`: `PatternRepack.exe
  patterns.ebsp out.h5 [1|2]` vs our writer (`flip=True`, binAvg)
  -- `/patterns` bitwise equal at binning 1 and 2 (Manufacturer
  ignored: the binary writes none).
- `test_ebsp_dims_binary_parity`: `EBSPDims.exe patterns.ebsp`
  stdout counts (`found 9 patterns`, `3 x and 3 y coordinates`)
  match `get_scan_info`'s numbers.
- `test_index_ebsd_template_matches_ours` (in
  `test_spherical_namelist.py`): run `IndexEBSD.exe -t` with
  `cwd=tmp_path`; the written `IndexEBSD.nml` equals
  `defaults().to_string()` line-wise (the CI fixture's live twin).

## Weekly

None -- no phase-specific weekly load (the large-map repack +
IndexEBSD regression sweep is Phase 10's; `--weekly -k spherical`
must simply stay green).

## Manual

- Run the acid test once from a clean checkout of the branch
  (env var set) and eyeball the `IndexEBSD.exe` stdout geometry
  block (`Vertical Flip: true` on the default route, `Pattern
  Center`, `Scintillator Distance`) against D6's expectations.
- `sphinx-build -b html`: the three new reference pages render;
  the three CHANGELOG entries display with the PR #10 link.

## Definition of done

- [ ] Spec (`requirements.md`, this file) and the plan-0
      constitution amendments committed on `sht-interop`.
- [ ] Failing tests committed first; then implementation
      (`_namelist.py`, `_pattern_repack.py`, `get_scan_info`,
      exports).
- [ ] Full automated section green (`-n 4`); local-gated section
      green on this machine; coverage of the new modules == 100 %
      (recorded, the Phase 1-6 precedent).
- [ ] Adversarial review (fidelity + conventions + the plan-5
      bug-injection list) done and fixes applied.
- [ ] `pre-commit run --files <changed>` clean;
      `--doctest-modules` green; `sphinx-build -b html` and
      `-b linkcheck` exit 0.
- [ ] CHANGELOG entries (three, PR #10); roadmap Phase 9 boxes
      ticked with measured numbers; signed commits pushed; PR #10
      opened into fork `develop` (GPL-only statement included).

## Recorded results

### 2026-09-02 -- drafting-probe measurements (spec phase; binaries @ 60f3517, Windows, this machine)

Scripts (session scratchpad `p9/`, not committed): `acid_test.py`
(repack variants + IndexEBSD runs), `acid_test2.py` (flip/vendor
decoupling, output-file dump, parser probes), `probes3.py`
(binning, Manufacturer errors, EBSPDims irregular), `probes4.py`
(vendor conversions, uint16 corruption). Inputs:
`kp.data.nickel_ebsd_small()` (background-removed for indexing
runs), `src/kikuchipy/data/oxford_binary/patterns.ebsp`,
`src/kikuchipy/data/emsphinx/ni_small_20kv_bw384.sht`
(`primary_angle` 70.0 verified via `_sht_file.read_sht`; surfaces
publicly as `MasterPatternHarmonics.sample_tilt`).

**PatternRepack.exe** on `patterns.ebsp`:
- bin 1: `/patterns` `(9, 60, 60)` uint8, layout CONTIGUOUS (1),
  alloc EARLY (1), data offset **2144**, file **34544 B**
  (= 2144 + 9*3600), root keys `['patterns']` -- **no
  Manufacturer**; rows equal `nickel_ebsd_small().data[:, ::-1, :]`
  (hard-coded flip).
- bin 2: `(9, 30, 30)` uint8, **10244 B**; bitwise equal to
  `floor(block_sum/4 + 0.5).astype(uint8)` of the flipped data
  (`True`); vs `np.round` (banker's): **1003 of 8100 pixels
  differ**.

**EBSPDims.exe**:
- `patterns.ebsp`: `found 9 patterns ... width: 60 / hegiht: 60 /
  type: 8 bit / bytes: 3600 ... found 3 x and 3 y coordinates`
  (regular -> no lists).
- staggered synthetic v2 file (6 patterns, x
  `[0, 1, 2, 0.5, 1.5, 2.5]`, y `[0, 0, 0, 1, 1, 1]`):
  `found 6 x and 2 y coordinates`, then `X: 0 0.5 1 1.5 2 2.5`
  and `Y: 0 1`.

**kikuchipy-written repack** (h5py low-level): layout 1 / alloc 1 /
offset **6160** / file 38560 B with the Manufacturer dataset;
`ni_small` detector `pc_average = [0.42513885, 0.21336699,
0.50070692]`, `sample_tilt` 70, `tilt` 0.

**IndexEBSD.exe acid runs** (`bw` 68, normed+refine, `nregions`
10, `circmask` -1, `delta` 500, `thetac` 0, scandims 3,3,1.5,
`nthread=1 batchsize=1`, vendor Bruker + `pc_average`, master =
in-package `ni_small` .sht):

| run | exit | index time | misorientation vs stored xmap (deg) |
|---|---|---|---|
| rows flipped, Manufacturer Bruker, refine | 0 | 0.0807085 s (111.512 pat/s) | per-point 0.652 0.947 0.795 0.709 0.713 0.684 0.875 0.610 0.902 -> **median 0.713 / max 0.947** |
| same, refine=.FALSE. | 0 | 0.0663445 s (135.656 pat/s) | median 0.783 / max 1.007 |
| rows as-is, Manufacturer Bruker (wrong pairing), refine | 0 | 0.0803978 s (111.943 pat/s) | **median 39.618 / max 53.304**, scores ~0.22 |
| rows as-is, Manufacturer **EDAX**, nml vendor Bruker, refine | 0 | 0.0867682 s | median 0.725 / max 0.948 (the decoupled route; equivalent, not bitwise -- ~0.01-0.02 deg per point vs the Bruker-flip route) |
| rows as-is, Manufacturer **EMsoft** (the writer's default route), refine | 0 | ~0.08 s | median 0.725 / max 0.948; `.ang` Eulers **bitwise identical** to the EDAX-manufacturer run (max diff 0.0 -- same vendorFlip read path) |

Refined `.ang` scores, **Bruker-flip route** (col 7): `0.669 0.583
0.646 0.674 0.573 0.655 0.633 0.615 0.626` (mean **0.6304**); iq
column constant 0.2; `.ang` header is TSL-convention (`y-star
0.786633`). Stdout geometry block (EDAX run): `Scintillator
Distance: 15021.2 microns`, `Pattern Center: -4.49167, 17.198
fractional pixels`, `Vertical Flip: true`, `Side Length: 135`.
(Default-route per-point anchors: second dated section below.)

**Vendor-equivalence runs** (same flipped Bruker file): namelists
with `EMsoft (-4.49166896, 17.19798032, 15021.20746804)`,
`EDAX (0.42513885, 0.78663301, 0.50070692)`,
`Oxford (0.42513885, 0.78663301, 0.50070692)` -> `.ang` Euler
columns **max |diff| 0.0** vs the Bruker run (bitwise).

**uint16 twin** (same values as uint16, Manufacturer Bruker):
exit 0 but median **38.88 / max 56.41 deg**, scores 0.156-0.231 --
the buffered `NATIVE_UINT8` read corruption (`pattern.hpp:515`;
mmap gate `:494` dead).

**Manufacturer errors** (both exit 1): PatternRepack's own output
-> `repacked_ni_small.h5 doesn't have a Manufacturer string`;
`Manufacturer="kikuchipy"` -> `unknown EBSD vendor: kikuchipy`.

**Namelist parser probes through IndexEBSD.exe**: no leading space
-> exit 1 `missing leading space in namelist line 5 "patdims    =
60, 60,"`; duplicate key -> exit 1 `key "circmask" was defined
twice in the name list`; extra key `extrakey = 42` -> **exit 0**
with `warning: some namelist parameters weren't used: extrakey`
(and the file indexes); first-line kv -> exit 1 `namelist files
cannot have key value pairs in the first line`; missing comma ->
exit 1 `missing comma between previous entry and namelist line 7
...`; double-quoted string -> exit 1 `couldn't parse token
""bruker"" ... (strings must be in single quotes, e.g. key =
'value')`.

**`IndexEBSD.exe -t` template**: 119 lines, CRLF (Windows text
mode), md5 `49ddf0e7d9b2d758d918c20a7f900a6d`; defaults render at
stream precision (`delta      = 50`, `pctr       = 0, 0, 15000`,
`scandims   = 256, 256, 1, 1`); the master-file line carries a
trailing `", "`; terminator `" /"` present (qualmap non-empty in
defaults).

**Vendor conversion cross-check (pure Python)**: on `(48, 60)`
(nrows, ncols), Bruker pc `(0.4251, 0.2134, 0.5007)`, delta 500:
internal triple `cX -4.494, cY 13.7568, sDst 12016.8`;
EMSphInx-TSL `(0.4251, 0.62928, 0.40056)` == kikuchipy
`pc_oxford()`; EMSphInx-Oxford `(0.4251, 0.7866, 0.40056)` vs
kikuchipy `pc_tsl()` `(0.4251, 0.7866, 0.5007)` (z differs);
EMSphInx-EMsoft `(-4.494, 13.7568, 12016.8)` == kikuchipy
`pc_emsoft(version=4)` exactly (v5 flips the x sign; see the
second dated section for the `binning`/`px_size` precondition).

**kikuchipy vs IndexEBSD context** (identical inputs, harmonics
via `MasterPatternHarmonics.from_file` on the same `.sht`,
`detector.pc = detector.pc_average`, `spherical_indexing` defaults
at `bw` 68; IndexEBSD reference = the **Bruker-flip** run): per-point
0.360 0.356 0.308 0.350 0.324 0.341 0.363 0.262 0.309 deg ->
**median 0.341 / max 0.363**; scores kikuchipy `0.6575 0.5738
0.6198 0.6746 0.5479 0.6295 0.6362 0.6050 0.6192` vs IndexEBSD
above, Pearson r **0.9607**; kikuchipy vs stored xmap median
0.533. Recorded as Phase 10 context only (its mission gates are
refined median < 0.2 deg, r > 0.98 -- evaluated there with its own
harness, re-baselined on the canonical default route).

**Environment note**: HDF5 dataset-byte assertions are
h5py-version-stable; whole-file md5s are not pinned (superblock
variance). The `-t` template capture is CRLF because the Windows
binary writes in text mode; all comparisons are line-wise.

### 2026-09-02 -- revision-probe re-measurements (spec revision; binaries @ 60f3517, Windows, this machine)

Scripts (session scratchpad `p9r/`, not committed): `probe_pure.py`
(flip/bin commutation, pc_emsoft precondition, delta cancellation,
precision), `probe_binary.py` (default-route anchor, parser error
probes, sanity-bound probes, delta no-op). Run after the fidelity
and conventions reviews to settle the disputed numbers.

**Default-route acid anchor** (`Manufacturer` EMsoft, rows
unflipped, nml vendor Bruker + unrounded `pc_average`, `bw` 68,
`nthread=1 batchsize=1`, delta 500, `cwd`=run dir): exit 0, stdout
`Vertical Flip: true`, `Pattern Center: -4.49167, 17.198`,
`Scintillator Distance: 15021.2 microns`, `0.0774412s to index
(116.217 pat/s)`; refined-vs-stored-xmap per-point `[0.6477,
0.9479, 0.7988, 0.7051, 0.7245, 0.6889, 0.8748, 0.6159, 0.9015]`
deg -> **median 0.7245 / max 0.9479**; `.ang` scores `[0.662,
0.580, 0.645, 0.667, 0.575, 0.654, 0.633, 0.614, 0.625]`, mean
**0.6283**. Confirms both reviews: the draft's 0.713/0.947/0.6304
anchors belong to the Bruker-flip route; the acid bands are now
anchored on this route (median < 1.2 = 1.66x, max < 1.6 = 1.69x,
approx(0.628, rel=0.05)).

**Bruker-flip route re-run** (same session): exit 0, `Vertical
Flip: false`, `0.0747929s (120.332 pat/s)`, median 0.7131 / max
0.9474, scores mean 0.6304 -- reproduces the drafting table.

**Delta no-op through the binary** (D6): the default-route
namelist with `delta = 250` (all else equal): exit 0, stdout
`Scintillator Distance: 7510.6 microns` (halved), `.ang` Euler
columns **bitwise equal** to the delta-500 run, scores identical.
`delta` cancels for fractional vendors; only EMsoft pctr and the
sanity window see it.

**Parser probes through IndexEBSD.exe** (all on the working acid
namelist):
- indented comment `" ! an indented comment"` -> exit 1 `bad
  delimeter (expected '=') in namelist line 6 " ! an indented
  comment"` (comment rule is column 0 only -- `nml.hpp:307`
  `line.front()`; the doc comment at `:291` is wrong).
- whitespace-only line `"   "` -> exit 1 `error parsing line '   '
  from name list` (not skipped -- literal `line.empty()`).
- two leading spaces `"  bw         = 68,"` -> exit 1 `error
  parsing line '  bw         = 68,' from name list` (a different
  message than the zero-space `missing leading space ...`).

**Sanity negativity bounds are live** (rejects conventions L5,
whose `size_t` premise contradicts the `int32_t` declarations at
`ebsd/nml.hpp:61-63, 79-80`): `nregions = -5` -> exit 1
`unreasonable AHE nregions`; `nthread = -1` -> exit 1 `negative
thread count`. The `< 0` checks fire; they stay ported and probed.

**Flip/bin commutation** (pure NumPy, nickel `(9, 60, 60)` uint8):
for `binning` in {2, 3, 4, 5, 6}, `binAvg(flip(x)) ==
flip(binAvg(x))` and `binFloat(flip(x)) == flip(binFloat(x))`
**bitwise** in every case -- the "flip applied after binning"
mutant is unkillable and was removed from plan 5.3 (replaced by
the flip-axis mutant).

**`pc_emsoft(version=4)` precondition** (pure Python): the nickel
detector carries `binning=8, px_size=1.0` and a `(3, 3)` pc array;
its `pc_emsoft(version=4)` is ~8x the EMSphInx triple (e.g. cX
-35.3 vs -4.49). A detector built with `binning=1, px_size=500,
pc=pc_average` gives `(-4.49166896, 17.19798032, 15021.20746804)`
== the EMSphInx Bruker->internal triple exactly. The named test
constructs its detector that way.

**Precision of the frozen triple** (conventions L7): from the
8-decimal rounded pc, `(0.5 - 0.21336699) * 60 = 17.1979806` and
`0.50070692 * 60 * 500 = 15021.2076`; the recorded
`17.19798032 / 15021.20746804` derive from the unrounded
`pc_average` (`0.21336699472343632...`). Tests quote input and
output from the same unrounded source and assert with
`pytest.approx`.
### 2026-09-02 -- tests-first run (skeleton + failing suite; binaries @ 60f3517, Windows, this machine)

Probe script (session scratchpad, not committed): `p9_probe.py`
(pure Python conversions on the live kikuchipy detectors, the
in-package `patterns.ebsp` through `OxfordBinaryFileReader`, the
`binAvg` rounding count and the `.6g` formatting). The
`IndexEBSD.exe -t` capture was re-taken in a temporary directory:
119 lines, CRLF, md5 `49ddf0e7d9b2d758d918c20a7f900a6d` --
unchanged, and now embedded verbatim (LF) as
`INDEX_EBSD_TEMPLATE` in `tests/test_indexing/
test_spherical_namelist.py`, with a guard test for the trailing
space of its `masterfile` line.

**Confirmed unchanged**: `pc_average` of `nickel_ebsd_small` is
`[0.42513885, 0.21336699, 0.50070692]` (unrounded
`0.4251388...`/`0.2133669947...`/`0.5007069...`) and its Bruker ->
internal triple at `delta` 500 on the 60 x 60 detector is
`(-4.491668961692965, 17.19798031659382, 15021.207468035665)`
(from the rounded pc: `17.1979806`/`15021.2076`, hence
`pytest.approx`). `binAvg` bin 2 of the flipped nickel data
differs from `np.round` on **1003 of 8100** pixels.
`patterns.ebsp`: version 2, 9/9 patterns, `(60, 60)` uint8, 3600
B/pattern, `beam_x == beam_y == [0, 1.5, 3]`. `.6g` renders
`50.0 -> "50"`, `1.5 -> "1.5"`, `0.42513885 -> "0.425139"`,
`12345678.9 -> "1.23457e+07"`.

**Correction to the TSL/Oxford deviation rows (D6)**: measured on
detectors built with `binning=1, px_size=delta=500` and Bruker pc
`(0.4251, 0.2134, 0.5007)`:

| shape (nrows, ncols) | EMSphInx-TSL vs kp `pc_oxford()` | EMSphInx-Oxford vs kp `pc_tsl()` |
|---|---|---|
| (60, 60) | equal | equal |
| (48, 60) | equal | **z differs** (0.40056 vs 0.5007) |
| (60, 48) | equal | **equal** (0.625875 both) |

kikuchipy's `_pc_bruker2tsl` divides z by `min(nrows, ncols)/nrows`
(`_ebsd_detector.py:2326-2330`): on `(60, 48)` that factor is
`48/60`, exactly EMSphInx's `h/w`, so the two agree there; on
`(48, 60)` it is a no-op and they disagree. The spec sentence
"asserted on both `(48, 60)` and `(60, 48)`" is corrected in place
in both `requirements.md` D6 and the `TestVendorConversions` bullet
above: `test_tsl_oxford_deviate_from_kikuchipy_on_rectangular` is
parametrised over the three shapes and asserts the measured
relation per shape (equality of the x and y components always, and
of z only where the table says so). The "never delegate to the
kikuchipy helpers" decision is unaffected -- one orientation
disagrees, which is enough to make delegation wrong.

Also confirmed (same probe): EMSphInx-EMsoft equals kikuchipy
`pc_emsoft(version=4)` **exactly** (max |diff| 0.0) on all three
shapes under `binning=1, px_size=delta`, and `version=5` flips the
x sign; `delta` 250 vs 500 gives bitwise identical
`to_detector`-style pc for EDAX/tsl/Oxford/Bruker and different pc
for EMsoft.

**h5py availability (plan 5.2)**: `h5py.h5d.CONTIGUOUS`,
`ALLOC_TIME_EARLY`, `h5p.DATASET_CREATE.set_alloc_time`,
`h5py.h5t.CSET_ASCII` and `string_dtype(encoding="ascii")` all
exist in the environment's h5py 3.16 and have been part of the
low-level API since h5py 2.x, so **no `importorskip` or version
gate is added**.

**Tests-first status**: 265 CI tests fail, every one of them on
`NotImplementedError` from the three stubs (11 of them surface as
`pytest.warns` "DID NOT WARN" and 2 as `pytest.raises` regex
mismatches chained from that same `NotImplementedError`), and the
13 locally gated tests likewise. The only new test which passes is
`test_the_template_master_line_keeps_its_trailing_space`, the
tripwire on the frozen template constant itself. Existing suites
stay green: `tests/test_indexing tests/test_io -k "spherical or
sht or emsphinx or oxford" -n 4` gives 2545 passed / 708 skipped
with the new modules excluded, and the six refactored
`test_spherical_sht_file.py` binary tests pass with
`KIKUCHIPY_EMSPHINX_DIR` set.

**Open at implementation time**: the lazy-write `tracemalloc`
bound is provisional. The test asserts the peak stays under
`LAZY_PEAK_FRACTION = 0.5` of the full map bytes (a 64 x 32 map of
60 x 60 uint8 patterns, 7 372 800 B) and records the measured peak
with `record_property`; the band is to be re-pinned on that
measurement with the Phase 6 margin convention once the writer
exists.

### 2026-09-02 -- test-quality review fixes (skeleton unchanged; pure-Python measurements, this machine)

Probe scripts (session scratchpad, not committed): `check3.py`
(Bruker composition), `check6.py` (binAvg accumulator search),
`check8.py` (reader scalar types), `check10.py` (lazy-write peak).
Run to settle the ten findings of the test-quality critic; no
implementation exists yet, so all four are pure Python.

**Bruker is not the bitwise identity** (critic finding 3). With
`geom = pctr_to_geometry` and `pc = geometry_to_pc` as the D6
formulas, `pc(geom(x))` at `delta` 500:

| input pc | w, h | round trip |
|---|---|---|
| `(0.4251, 0.2134, 0.5007)` | 60, 60 | `(0.4251, 0.21340000000000003, 0.5007)` |
| `(0.4251, 0.2134, 0.5007)` | 60, 48 | `(0.4251, 0.21340000000000003, 0.5006999999999999)` |
| `(0.4251, 0.2134, 0.5007)` | 48, 60 | `(0.4251, 0.21340000000000003, 0.5007)` |
| nickel `pc_average` | 60, 60 | y `0.21336699472343634` vs `...32` |

`np.array_equal` is `False` in every case, so a spec-faithful
`_pctr_to_pc = geometry ∘ pctr` fails the two `np.array_equal`
assertions. Resolved by **short-circuiting `"Bruker"`** in
`_pctr_to_pc`/`_pc_to_pctr` (documented in both docstrings, the
module docstring and an in-place note at `requirements.md` D6);
`test_bruker_is_the_identity_on_the_kikuchipy_pc` now asserts both
directions and names the reason.

**binAvg accumulator discrimination** (critic finding 6). The
fixture of `test_binavg_accumulates_in_float64` has a maximum block
sum of **1020**, exact in `float32` and in `uint16`, so a `float32`
(or `int32`) accumulator survives it; only an *input-dtype*
accumulator dies. For `uint8` input the `float32`/`float64`
distinction is unobservable at any binning a legal pattern shape
allows (`255 * binning**2 < 2**24` up to `binning = 256`). The
uint8 test is therefore renamed
`test_binavg_does_not_overflow_the_input_dtype`, and the C++
`std::vector<double>` is pinned by a new
`test_binavg_accumulates_in_float64` on a **`float32`** block
`[[2**24, 1], [1, 2]]`: 64-bit sum 16777220 -> mean **4194305.0**
(exact in `float32`), while every 32-bit accumulation of it gives
16777218 -> **4194304.5** (pairwise and left-to-right alike, both
measured). The named assertion of the `validation.md` bullet above
keeps its name; the bullet's parenthetical ("blocks whose uint8
sums exceed 255") describes the renamed test.

**Reader scalar types** (critic finding 8).
`OxfordBinaryFileReader` on the in-package `patterns.ebsp`:
`signal_shape` is `(np.int32(60), np.int32(60))`, `n_patterns` is
`np.int64(9)`, `n_bytes` is `np.int32(3600)`; `isinstance(np.int32(60), int)`
and `isinstance(np.bool_(True), bool)` are both **`False`**. So
`get_scan_info` must cast to plain Python scalars -- now stated in
its docstring and asserted for all five integer keys.

**Lazy-write peak** (critic finding 10; numpy 2.4.6, dask 2026.3.0,
h5py 3.16.0). The exact test fixture `(64, 32, 60, 60)` uint8,
chunks `(8, 32, 60, 60)`, 7 372 800 B, into a contiguous
early-allocated data set:

| route | peak B | peak / full |
|---|---|---|
| `da.store`, default (threaded) scheduler | 1 181 159 | **0.160** |
| `da.store`, `scheduler="synchronous"` | 1 061 764 | **0.144** |
| explicit `for blk in arr.blocks` loop | 2 070 985 | **0.281** |
| full materialisation (control) | 7 497 272 | **1.017** |

`LAZY_PEAK_FRACTION = 0.5` stands: at least **1.78x** margin over
the worst streaming route, not thread-count sensitive, and the
materialising mutant misses by 2x. The band is no longer
provisional and the "Open at implementation time" note above is
resolved.

**Named-assertion naming map** (critic finding 5, cosmetic). The
`test_uint16_write_warns` / `test_float32_write_warns` pair of the
D2 bullet and of plan 5 is realised as the single parametrised
`test_non_uint8_write_warns[uint16|float32]`; coverage is
identical, only the node id differs. Likewise
`test_conversion_table_square_and_rectangular` is realised as
`test_conversion_table_square` plus
`test_conversion_table_rectangular`, and `test_vendor_whitelist` as
`test_vendor_whitelist_accepts` plus `test_vendor_whitelist_rejects`.

**Messages now pinned verbatim** (critic finding 5). Three C++
strings which the suite matched only by a substring are frozen as
module constants and matched with `re.escape`: `expected a filename
or dimensions + resolution for 'scandims' in namelist`, `scan
dimensinos must be non-negative integers` (the C++ typo is part of
the wording, `ebsd/nml.hpp:263`) and `some namelist parameters
weren't used: extrakey` (`index_ebsd.cpp:83`). Two uncovered
element-count messages gain named tests: `patdims must be 2
elements` (`:272`) and `pctr    must be 3 elements` (`:276`, four
spaces).
