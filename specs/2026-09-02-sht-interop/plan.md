# Phase 9 -- `sht-interop`: plan

Branch `sht-interop` off `develop` (after the Phase 7 merge,
jwestraadt/kikuchipy#9; Phase 8 deferred per the 2026-09-02
re-scope). Models: plan/spec on Fable 5 (xhigh, ultracode); tests,
implementation, adversarial review and fixes by Opus 5 agents
(xhigh, ultracode). Autonomous mode (approval gate waived for
spherical-indexing phases; decisions flagged in section 6). Tests
are written (failing) before the code they exercise. No numba
kernels this phase (no `-n 0` warm-up needed beyond convention).
Every number in `requirements.md` was measured on 2026-09-02 with
the drafting probes (`acid_test.py`, `acid_test2.py`, `probes3.py`,
`probes4.py`, scratchpad `p9/`) or the revision probes
(`probe_pure.py`, `probe_binary.py`, scratchpad `p9r/`) against the
built EMSphInx binaries @ 60f3517 (not committed; recipes and full
outputs in `validation.md` "Recorded results").

## 0. Constitution amendments (applied 2026-09-02 in the spec commit)

1. `specs/roadmap.md`, Phase 9 boxes: rewrite the first box to name
   the delivered pieces with the measured facts
   (`write_emsphinx_patterns` (PatternRepack port **plus the root
   `Manufacturer` dataset the C++ program omits** -- `IndexEBSD.exe`
   refuses a Manufacturer-less file, measured; vlen-**ASCII**
   Manufacturer -- h5py's default vlen UTF-8 is fatal with a
   misleading H5Dread error, measured; contiguous `/patterns` with
   alloc-time-early and **zero filters** (gzip is fatal, measured;
   alloc/layout are PatternRepack byte-parity only),
   layout/alloc/offset pinned via h5py -- PatternRepack's own repack
   of the in-package `patterns.ebsp`: offset 2144, 34544 B, rows
   flipped; native byte order enforced (`>u2` rejected by the
   binary, measured); `binAvg` == block mean rounded half away from
   zero, bitwise vs `PatternRepack.exe` bin 2, 1003/8100 pixels
   differ under banker's rounding; `binFloat` = float32 block *sum*,
   unreachable in the shipped binary (`binToFloat=false` const) so
   NumPy-pinned; per-manufacturer auto flip -- EDAX/EMsoft written
   unflipped and read-flipped, Oxford/Bruker pre-flipped, both
   routes measured correct, wrong pairing ~39.6 deg median;
   flip/bin commute exactly for divisible binning (measured b 2-6);
   uint16/float32 warned -- the buffered `NATIVE_UINT8` HDF5 read
   corrupts them, measured 38.9 deg median garbage, and the mmap
   path is dead code (`0 != getNfilters()` inversion,
   `pattern.hpp:494`); kikuchipy overwrite/suffix/directory
   conventions), `get_scan_info` in `oxford_binary`
   (EBSPDims contract: distinct exact-value beam-x/y sets +
   `len(x)*len(y) == n_present` regularity + coordinate lists;
   measured on `patterns.ebsp`: 3 x 3 regular), `EMSphInxNamelist`
   (nml.hpp parser semantics incl. the `test/util/nml.cpp` suite
   and the error cases probed through `IndexEBSD.exe` -- among them
   the column-0-only `!` comment rule, the unskipped
   whitespace-only line and the two-leading-spaces message; EBSD
   fields with `defaults()`/`sanity_check()` (13 checks, the
   negativity bounds live -- `nregions=-5`/`nthread=-1` exit 1,
   measured)/`to_string` at `.6g` -- line-parity with the captured
   `IndexEBSD -t` template, 119 lines; the four vendor PC
   conversions ported from `detector.hpp:85, 249-279`, pinned by
   bitwise-identical `.ang` output across EMsoft/EDAX/Oxford/Bruker
   namelists and delta-250-vs-500 Bruker runs (delta cancels for
   fractional vendors, measured), EMsoft == kikuchipy
   `pc_emsoft(version=4)` under `binning=1, px_size=delta`,
   TSL/Oxford deviate from kikuchipy's `pc_tsl`/`pc_oxford` on
   rectangular detectors -- frozen table, both orientations; the
   `circmask > 0` semantics (processor-side CircMask kept at
   radius r, `Geometry::circ` false -- `imprc.hpp:108-122`,
   `idx.hpp:230, 254`), the `tsl`-lowercase whitelist, the
   double-`ipath` quirk (reproduced on the derived `pat_path`,
   storage is raw), the qualmap-conditional `" /"` terminator));
   rewrite the test box with the canonical-route acid numbers:
   "IndexEBSD.exe indexes a kikuchipy-written repack + nml +
   in-package ni .sht (default route: Manufacturer EMsoft,
   unflipped; nthread=1 batchsize=1, bw 68): refined vs stored
   xmap median 0.7245 / max 0.9479 deg (assert median < 1.2,
   max < 1.6), scores mean 0.6283 (approx 0.628 rel 0.05), 112-120
   pat/s recorded; Bruker-flip route recorded separately (0.713/
   0.947, mean 0.6304, equivalent not bitwise);
   kikuchipy-vs-IndexEBSD context 0.341/0.363 deg, r 0.9607 on the
   Bruker route (Phase 10's gate, recorded only); namelist round
   trip incl. template line-parity; out-of-scope list confirmed in
   mission.md".
2. `specs/tech-stack.md`, Numerics, PatternRepack bullet: replace
   the one-sentence contract with the measured one -- "PatternRepack
   contract (`write_emsphinx_patterns(filename, signal, *,
   manufacturer=...)`): root **scalar vlen-ASCII** dataset
   `Manufacturer` (which `PatternRepack.exe` itself omits --
   `IndexEBSD.exe` requires it, measured; vlen UTF-8 -- the h5py
   default -- is fatal with a misleading H5Dread error, measured;
   accepted strings are the reader's flip table `{EDAX, Oxford,
   Bruker, Bruker Nano, DREAM.3D, EMsoft}`, distinct from the
   namelist vendor whitelist `{EMsoft, EDAX, tsl, Oxford,
   Bruker}`), `/patterns` `(n, h, w)` contiguous with
   alloc-time-early and **zero filters** (filters are the real
   reader constraint -- gzip is fatal, measured; the mmap gate is
   dead code (`0 != getNfilters()`, `pattern.hpp:494`) and every
   HDF5 read is buffered through `NATIVE_UINT8`, so only uint8
   pattern files are read correctly; uint16/float32 writes warn;
   byte-swapped dtypes are rejected by the binary, so the writer
   casts to native order), per-manufacturer auto vertical flip
   (EDAX/EMsoft unflipped + read-flip, Oxford/Bruker pre-flipped;
   both measured correct on nickel_ebsd_small, wrong pairing ~39.6
   deg median; flip and divisible binning commute exactly --
   measured), binning as in-dtype `binAvg` (block mean, round half
   away from zero -- `floor(x+0.5)`, bitwise vs `PatternRepack.exe`)
   or float32 `binFloat` block **sum** (dead code in the shipped
   binary, NumPy-pinned; `binning == 1` under `bin_to_float=True`
   casts to float32 -- a completed-dead-code deviation); kikuchipy
   `overwrite`/suffix/`_ensure_directory` conventions; EMsoft raw
   `.data` input is out of scope."
   Append an interop bullet: "`EMSphInxNamelist`: nml.hpp parse
   semantics (first line skipped, `!` comments at column 0 only
   (code contradicts its own doc comment -- measured), leading
   space exactly one (two+ raise a different message than zero),
   whitespace-only lines raise, `,` terminators, lowercase keys,
   `.true./.false.`, int-subset-of-double, `get_int` rejects
   doubles while `get_double` accepts ints, whitespace stripped
   inside 2nd+ quoted strings -- ported, writer guards against it);
   `to_string` at C++ `.6g` stream precision, line-parity with
   `IndexEBSD -t`; raw path storage with derived
   `pat_path`/`master_paths` (the C++ double-prefixes `ipath` on
   round trip; the psymfile double-`ipath` quirk lives on the
   derived property); `patdset`/`scanname` always-optional (the
   C++ requiredness is filesystem-dependent -- measured);
   vendor PC conversions are EMSphInx's own formulas (EMsoft ==
   kikuchipy `pc_emsoft(version=4)` only at `binning=1,
   px_size=delta`; kikuchipy `pc_tsl`/`pc_oxford` deviate on
   rectangular detectors -- never delegate; `delta` cancels
   exactly for TSL/Oxford/Bruker -- measured bitwise through the
   binary); `to_detector` requires an explicit `sample_tilt` (the
   nml has none; IndexEBSD takes it from the master's sig,
   `idx.hpp:218`; cite `MasterPatternHarmonics.sample_tilt`);
   `from_kwargs` default vendor Bruker (pctr == kikuchipy
   `pc_average` verbatim), `delta=None -> 30000/patdims[0]` (30 mm
   detector, always inside sanityCheck's [5, 90] mm; px_size 1.0
   fixtures would be rejected), x-then-y `patdims`/`scandims`
   (reversed kikuchipy shapes), `n_thread`/`batch_size`
   passthrough, empty output names by default."
3. `specs/roadmap.md` Phase 10 box: append the recorded
   constraints -- "(repacked regression files must be uint8 -- the
   EMSphInx HDF5 read corrupts other dtypes, Phase 9 D2; **the
   canonical route for byte-stable references is the writer
   default: `Manufacturer` EMsoft, rows unflipped, nml vendor
   Bruker with `pc_average`, `nthread=1 batchsize=1`** -- the two
   correct flip routes are equivalent but not bitwise, ~0.01-0.02
   deg, Phase 9 D1/D7; the two-`delta` sweep is a geometric no-op
   for TSL/Oxford/Bruker namelists (delta cancels -- measured
   bitwise) so it must use the EMsoft vendor route or be recorded
   as a sanityCheck-window probe only, Phase 9 D6)".
4. `specs/_research/explore-emsphinx-programs-and-formats.md`
   addenda: 1.6 -- PatternRepack writes **no `Manufacturer`**, its
   output needs one added before `IndexEBSD` accepts it (measured
   error text); `binToFloat`/`flip` are hard-coded consts
   (`:116-117`), so `binFloat` is dead code in the binary; binAvg
   rounding is `std::round` = half away from zero (bitwise probe).
   2 (pattern reading) -- the HDF5 mmap gate `0 !=
   props.getNfilters()` (`pattern.hpp:494`) is inverted and
   unsatisfiable for contiguous datasets: every HDF5 pattern file
   takes the buffered branch, which reads `NATIVE_UINT8` for any
   dtype (`:515`) -- uint16/float32 HDF5 patterns are corrupted
   (measured end-to-end); **filters are the real constraint**
   (gzip-compressed `/patterns` is fatal, chunked-unfiltered and
   alloc-late are fine -- measured); byte-swapped dtypes are
   rejected (`NATIVE_*` comparison `:476-480`, measured); a vlen
   **UTF-8** `Manufacturer` fails with a misleading `H5Dread`
   error (`GetVendor` fallback `:631-635` throws into the outer
   handler `:519` -- measured). 1.9/3 -- namelist quirks: the `!`
   comment rule is **column 0 only** (`nml.hpp:307` `line.front()`;
   the doc comment `:291` says first-non-space -- code wins,
   measured), whitespace-only lines are not skipped (measured
   error), two+ leading spaces raise a different message than zero
   (measured both), whitespace is stripped inside the 2nd+ string
   of a quoted list (sticky `skipws`, `:364-368` -- measured,
   decisive); the vendor whitelist accepts lowercase `tsl` only
   (`TSL` rejected, template comment notwithstanding), `"tsl"` is
   duplicated in the parse check and dead in the `idx.hpp:227`
   Bruker branch; **`circmask > 0` keeps the processor-side
   circular mask at radius r** (`imprc.hpp:108-122` builds
   `CircMask` and sets `msk=true` for any `r >= 0`, fed at
   `idx.hpp:254`) while only `Geometry::circ` stays false
   (`maskPattern(circRad == 0)`, `idx.hpp:230`) -- earlier
   "silently disables the mask" wording was wrong; `patdset`/
   `scanname` requiredness is filesystem-dependent
   (`isHdf5(patFile)` gates, `ebsd/nml.hpp:242-246`, `:253-254` --
   measured both ways); `sanityCheck` has 13 checks and its
   negativity bounds are live (`int32_t` fields -- `nregions=-5`
   and `nthread=-1` exit 1, measured); the double-`ipath`
   prefix when `psymfile` is set (`nml.hpp:247`), and `parse_nml`
   re-prefixes `ipath` on every read so the C++ round trip
   double-prefixes; `to_string` emits the `" /"` terminator only
   inside the qualmap block (`:464-468`); stream doubles print at
   6 significant digits; `delta` cancels out of the geometry for
   TSL/Oxford/Bruker pctr (measured bitwise: Bruker delta 250 ==
   delta 500 `.ang`). Section 8 (gotchas): new items for the
   Manufacturer gap + ASCII cset, the uint8-only HDF5 read, the
   flip/vendor decoupling (file `Manufacturer` drives the flip,
   nml `vendor` drives only the PC interpretation -- measured with
   an EDAX-manufacturer file under a Bruker-vendor namelist), and
   the two flip routes being equivalent-but-not-bitwise
   (~0.01-0.02 deg; canonical = writer default, Phase 10).
5. `specs/mission.md`: no changes (the out-of-scope list already
   covers this phase and the deliverables table names no probe
   function; confirmation recorded in requirements Scope).

## 1. `_namelist.py` -- tests in `tests/test_indexing/test_spherical_namelist.py`

1. Licence block per D8 (EMSphInx notice with the ported line
   ranges, modification notice, not-ported list, recorded
   deviations); module docstring with the namelist grammar summary
   and the recorded quirks (D4/D5/D6).
2. **`_NameList`** (private, D4): `read` (stream/string), the typed
   variant storage with used-flags, `get_bool/int/double/string` (+
   plural forms) with C++-strict typing and messages,
   `fully_parsed`/`unused_tokens`; every `nml.cpp` behaviour listed
   in validation, plus the column-0 comment rule, the unskipped
   whitespace-only line, the two-leading-spaces message and the
   2nd+-string whitespace stripping.
3. **`EMSphInxNamelist`** (D5): fields (incl. `scan_file`/
   `scan_name`) + `defaults()`; raw path storage with the derived
   `pat_path`/`master_paths` properties (double-`ipath` quirk on
   `pat_path`); `read`/`from_string` = `parse_nml` semantics
   (optional-key try/except set, always-optional
   `patdset`/`scanname` deviation, scandims 3-or-4 doubles with
   the integer check, the string-scandims `NotImplementedError`,
   vendor whitelist with lowercase `tsl`, `sanity_check()` with
   all 13 checks, unused-token `UserWarning`); `to_string`/`write`
   = the template verbatim at `.6g` with the fixed ` &EMSphInx`
   first line, the qualmap-terminator quirk and the
   spaced-2nd-master `ValueError` guard; `__eq__` or an equivalent
   comparison helper for round-trip tests.
4. **Conversions** (D6): private `_pctr_to_pc(vendor, pctr, w, h,
   delta)` / `_pc_to_pctr(...)` implementing the frozen table;
   `to_detector(sample_tilt=...)`, `to_kwargs()` (incl. the
   `circ_rad > 0 -> False` processor-mask-loss warning and the
   roimask raise), `from_kwargs(...)` (vendor Bruker default,
   `delta=None -> 30000/patdims[0]`, x-then-y dims orders,
   `n_thread`/`batch_size` passthrough, empty output names,
   inverse `circular_mask` map, azimuthal/twist raise,
   sanity-by-construction).
5. Tests (assertion detail in `validation.md`): `TestNameListParser`
   (the ported `nml.cpp` suite: good-file scalars/vectors, partial
   parsing, the eleven error cases; the binary-probed error
   messages incl. the indented comment, the whitespace-only line
   and the two-leading-spaces variant; the 2nd+-string whitespace
   stripping), `TestNamelistTemplate` (defaults + line-parity with
   the frozen 119-line template constant; the qualmap-terminator
   quirk; `.6g` formatting cases), `TestNamelistRoundTrip`
   (defaults, the acid namelist and a non-empty-`ipath` variant
   through `from_string(to_string())`; the read of the acid
   namelist matches its literal values; the derived-path
   properties incl. the psymfile double-prefix; the always-optional
   `patdset`/`scanname` deviation; the spaced-master writer
   guard), `TestVendorConversions` (the frozen table on square,
   `(48, 60)` **and `(60, 48)`** detectors; the preconditioned
   `pc_emsoft(version=4)` equality; the kikuchipy
   `pc_tsl`/`pc_oxford` deviation rows -- asserted as *inequality*
   on rectangular, equality on square; delta-invariance for the
   fractional vendors; round-trip `_pc_to_pctr(_pctr_to_pc(...))`),
   `TestToFromKwargs` (kwargs keys == live signature names via
   `inspect.signature`; the circmask map + warning;
   batchsize<->chunksize; required `sample_tilt`; the from_kwargs
   delta rule and `sanity_check()` pass; patdims *and* scandims
   orders on rectangular shapes; roimask/scandims-string raises).
   Local-gated here: `test_index_ebsd_template_matches_ours`
   (namelist-only, D9 fixtures).

## 2. `_pattern_repack.py` -- tests in `tests/test_indexing/test_spherical_pattern_repack.py`

1. Licence block per D8; docstring with the D1/D2 contract and
   quirks (Manufacturer deviation + ASCII cset, uint8-only reader
   note, flip table with the flip/bin commutation note,
   canonical-route note for Phase 10).
2. **`write_emsphinx_patterns`** (D1): filename-first signature,
   suffix defaulting + `_ensure_directory` + the `overwrite`
   protocol (`kikuchipy.io._util`), manufacturer whitelist, flip
   resolution, dtype checks + native-byte-order cast + the
   non-uint8 `UserWarning`, binning validation,
   `binAvg`/`binFloat` (float64 accumulate / float32 accumulate,
   `floor(x+0.5)` for integer dtypes, float32 output for
   `bin_to_float=True` at any binning), low-level h5py dataset
   creation (contiguous + ALLOC_TIME_EARLY, zero filters,
   vlen-ASCII scalar `Manufacturer`), eager and lazy write paths
   (chunk-streamed slabs for dask input).
3. Tests: `TestWriteEmsphinxPatterns` -- layout pins (layout ==
   CONTIGUOUS, alloc == EARLY, offset defined, no filters, dtype,
   shape `(n, h, w)`), dataset bytes == `signal.data.reshape(...)`
   for the default route and == flipped for `flip=True`, navigation
   orders (2-D, 1-D, 0-d), lazy == eager bytes + the tracemalloc
   bound, manufacturer/dtype/binning `ValueError`s, the
   byte-swapped-input native-order pin, the uint16/float32 warning,
   `Manufacturer` value + vlen dtype + **ASCII cset** + scalar
   dataspace, the overwrite protocol (False leaves the file
   untouched, True replaces, suffixless filename gains `.h5`);
   `TestBinning` -- the frozen binAvg oracle incl. exact-half
   fixtures (kills `np.round`), binFloat sum/dtype incl. the
   `binning == 1` float32 cast, non-divisor raise;
   `TestAgainstEmsphinxBinaries` (local-gated, D9 fixtures,
   `cwd=tmp_path`) -- the acid test and its negative controls
   (validation), our `flip=True` dataset equals
   `PatternRepack.exe`'s bin-1 output bitwise, our `binAvg` bin-2
   equals its bin-2 output bitwise, and `EBSPDims.exe` on
   `patterns.ebsp` agrees with `get_scan_info`'s counts (parsed
   from its stdout).

## 3. `get_scan_info` -- tests in `tests/test_io/test_oxford_binary.py`

1. Function in `oxford_binary/_api.py` (kikuchipy header only, D8),
   exported from the plugin `__init__.pyi`; docstring shows the
   EBSPDims-equivalent report and cites the program as reference.
2. Tests: in-package `patterns.ebsp` (9 patterns, `(60, 60)` uint8,
   3600 B, 3 x / 3 y, regular -- the measured EBSPDims output);
   the staggered-row irregular fixture (6 patterns, 6 x / 2 y,
   `is_regular_grid` False, the sorted coordinate lists asserted)
   and the near-duplicate fixture (`1.0` vs `1.0 + 1e-12` distinct)
   -- both written by a **module-local helper** modelled on
   `src/kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py`
   (the conftest `oxford_binary_file` fixture derives `beam_x/y`
   from navigation indices and cannot express either file); the
   conftest `oxford_binary_file` variants **used as-is, no conftest
   change**: version 0 (no footer -> `beam_x is None`, regular
   False), `all_present=False` (present-count semantics), uint16;
   dict keys frozen.

## 4. API, docs, CHANGELOG

1. Existing files this change must touch (complete list):
   - `src/kikuchipy/indexing/__init__.pyi`: import + sorted
     `__all__` entries for `EMSphInxNamelist`,
     `write_emsphinx_patterns`.
   - `src/kikuchipy/indexing/__init__.py`: module docstring
     extended -- it currently describes dictionary/spherical
     *indexing* only and now also hosts two EMSphInx file
     utilities.
   - `src/kikuchipy/indexing/_spherical/__init__.py`: the
     alphabetical Submodules docstring list gains `_namelist` and
     `_pattern_repack` entries.
   - `src/kikuchipy/io/plugins/oxford_binary/__init__.pyi`:
     `__all__ = ["file_reader", "get_scan_info"]`.
   - `tests/test_indexing/test_spherical_indexer.py`
     (`TestExports`, `:929-996`): the lazy-loader parametrize list
     gains `"EMSphInxNamelist"` and `"write_emsphinx_patterns"`;
     both module tuples `(_indexer, _back_projection, _fft)` gain
     `_namelist` and `_pattern_repack`; the phase tuple
     `("Phase 5", ..., "Phase 8")` gains `"Phase 9"` and
     `"Phase 10"`.
   - `conftest.py`: gains the shared `emsphinx_program` and
     `read_ang` fixtures (D9); no `.ebsp` fixture changes (plan
     3.2).
   - `tests/test_indexing/test_spherical_sht_file.py`: its private
     `_emsphinx_dir`/`_emsphinx_program` helpers refactored onto
     the conftest fixtures (behaviour identical).
2. CHANGELOG: three entries per D8 (writer / namelist / probe, each
   with the PR #10 fork link). Reference docs render
   (`sphinx-build -b html` exit 0; the three new pages generated
   from `__all__`).
3. Apply the section-0 amendments in the spec commit; tick roadmap
   boxes only as gates complete.

## 5. Adversarial review and fixes

1. Fidelity reviewer refutes against the C++ line by line:
   `binAvg`/`binFloat` accumulation dtypes and rounding
   (`pattern_repack.cpp:50-98`), the flip/binning equivalence
   (`:105-113`, `:210-283`), the HDF5 property list (`:199-208`),
   `GetVendor` contract (`pattern.hpp:608-637`), the parser state
   machine (`nml.hpp:292-426` -- comma rule, comment rule at
   column 0, escape handling, sticky-skipws string lists,
   promotion, mixed-type errors), `parse_nml` optional/required key
   sets and quirks (`ebsd/nml.hpp:236-315`), the template writer
   against the captured `-t` bytes, `sanityCheck` bounds
   (`:621-639`, 13 checks), the conversion formulas
   (`detector.hpp:85, 249-279`) and the `idx.hpp:218-254` /
   `imprc.hpp:108-122` mapping (vendor switch, circmask both
   sides, flip-from-manufacturer). Conventions reviewer: licence
   blocks (incl. the probe's kikuchipy-only stance, 6.3), numpydoc,
   lazy_loader exports, CHANGELOG, no phase numbers public.
2. Confirm on the tests-first run that the oldest-supported h5py
   exposes `h5p.set_alloc_time`/`string_dtype(encoding="ascii")`
   (D9); add an `importorskip`/version gate only if it does not.
   Measure the lazy-write `tracemalloc` peak on the tests-first
   run and pin the band (D9).
3. Test-quality reviewer runs the **bug-injection list** -- every
   mutant must die by a named test:
   - **binAvg uses banker's rounding** (`np.round`): dies by
     `test_binavg_rounds_half_away_from_zero` (exact-half fixture)
     and the local bitwise `PatternRepack.exe` bin-2 test.
   - **binAvg divides by `binning` not `binning**2`**: dies by the
     binAvg oracle values.
   - **binAvg accumulates in input dtype** (uint8 overflow): dies
     by the oracle's high-value block fixture (sums > 255).
   - **binFloat averages instead of sums**: dies by
     `test_binfloat_sums_in_float32`.
   - **binFloat output not float32 / bin-1 not cast**: dies by the
     dtype pins.
   - **flip table inverted** (EDAX -> True): dies by
     `test_default_route_dataset_equals_signal_bytes` (EMsoft
     default writes unflipped) and, locally, by the acid test's
     `median < 1.2` (measured wrong-pairing median ~39.6).
   - **flip reverses columns instead of rows**: dies by
     `test_flip_true_reverses_rows` (`data[:, ::-1, :]`, not
     `data[..., ::-1]`). (The earlier "flip applied after binning"
     mutant was removed: for divisible binning the operations
     commute bitwise -- measured for binning 2-6 on the nickel
     fixture, both paths -- so no test can kill it; the docstring
     records the equivalence instead.)
   - **Manufacturer omitted / written as attribute or group**:
     dies by `test_manufacturer_is_a_root_scalar_vlen_dataset`;
     locally by the `IndexEBSD.exe` error-message test.
   - **Manufacturer fixed-length or UTF-8**: dies by the vlen
     dtype + ASCII cset pins (the C++ tolerates fixed-length but
     dies on UTF-8 with a misleading error -- the CI pin is the
     guard, D1/D2).
   - **manufacturer whitelist widened** (accepts "kikuchipy"): dies
     by `test_unknown_manufacturer_raises`; the measured binary
     error is the rationale.
   - **chunked or compressed dataset**: dies by the layout pin
     (CONTIGUOUS, zero filters); compression is the measured
     fatal case.
   - **alloc time not EARLY**: dies by `get_alloc_time` pin.
   - **byte-order cast dropped**: dies by
     `test_byteswapped_input_writes_native` (a `>u2` signal writes
     a native-order dataset).
   - **overwrite protocol ignored** (always truncates): dies by
     `test_overwrite_false_leaves_file`.
   - **dataset name changed**: dies by every h5py test opening
     `/patterns`.
   - **column-major flatten of the navigation**: dies by the
     bytes-equality test on the 2-D nickel signal (rows 1..8
     permute).
   - **non-uint8 warning dropped**: dies by
     `test_uint16_write_warns`.
   - **lazy path materialises**: dies by the `tracemalloc` peak
     bound (byte equality cannot see materialisation);
     **reorders**: dies by lazy==eager bytes on a rechunked lazy
     signal.
   - **parser accepts a duplicate/uppercase-distinct key**: dies by
     the dup-key raise test + case-insensitivity test.
   - **comma rule dropped**: dies by the missing-comma raise test.
   - **leading-space rule dropped**: dies by its raise test; the
     two-leading-spaces message test discriminates the second
     failure mode.
   - **comment rule reads first non-space char** (the C++ doc
     comment's version): dies by
     `test_indented_comment_is_not_a_comment` (raises the measured
     `bad delimeter` error).
   - **whitespace-only lines skipped** (`line.strip()`): dies by
     `test_whitespace_only_line_raises`.
   - **2nd+-string whitespace preserved** (quirk 'fixed'): dies by
     `test_second_string_whitespace_stripped`.
   - **first-line kv accepted**: dies by its raise test.
   - **escape handling broken**: dies by the `vStrSgl` case.
   - **int list not promoted with doubles**: dies by the
     `vDoubles` mixed-list case.
   - **bool/number mix accepted**: dies by its raise test.
   - **`get_int` accepts doubles** (`bw = 68.0` parses): dies by
     `test_get_int_rejects_doubles` (C++ strictness).
   - **`get_double` rejects ints**: dies by the pctr-with-ints
     round trip (the C++ accepts them, `:228`).
   - **unused-token warning dropped**: dies by
     `test_extra_key_warns_with_its_name`.
   - **`TSL` accepted / `tsl` rejected**: dies by the whitelist
     test pair.
   - **`patdset` unconditionally required**: dies by
     `test_patdset_scanname_always_optional`.
   - **paths stored prefixed (C++-style)**: dies by the
     non-empty-`ipath` round-trip test (the C++ storage
     double-prefixes).
   - **double-`ipath` quirk 'fixed'**: dies by
     `test_psymfile_double_ipath_quirk_is_ported` (constructed
     namelist with both set; the derived `pat_path` shows the
     doubled prefix).
   - **spaced-master writer guard dropped**: dies by
     `test_write_rejects_spaced_second_master`.
   - **`.6g` replaced by repr**: dies by the template line-parity
     fixture (`delta      = 50` would become `50.0`).
   - **qualmap terminator quirk 'fixed'**: dies by
     `test_to_string_without_qualmap_has_no_terminator` and the
     template fixture (with qualmap -> terminator present).
   - **scandims 3-vs-4 handling wrong** (`dims.back()` semantics):
     dies by the 3-element round trip (`scan_steps == (s, s)`).
   - **sanityCheck bound off** (detW window, bw range, the live
     negativity checks): dies by the parametrised
     `test_sanity_check_bounds` (each of the 13 checks probed one
     unit either side; `nregions=-5`/`nthread=-1` measured live in
     the binary).
   - **vendor conversion sign/axis flipped**: dies by the frozen
     conversion-table test; locally by the bitwise `.ang`
     equivalence runs.
   - **EMsoft conversion uses the v5 sign**: dies by
     `test_emsoft_equals_kikuchipy_pc_emsoft_version_4`
     (preconditioned detector, D6).
   - **TSL yStar/zStar scaled by `h`** (kikuchipy-style): dies by
     the rectangular-detector table rows (both orientations).
   - **conversions delegated to kikuchipy `pc_*`**: dies by the
     rectangular deviation rows on `(48, 60)` *and* `(60, 48)`
     (equalities would flip; the `(60, 48)` rows exercise the
     `min(nrows, ncols)/nrows` branch that is a no-op on
     `(48, 60)`).
   - **delta leaks into fractional-vendor geometry**: dies by
     `test_delta_invariant_for_fractional_vendors` (two deltas ->
     identical `to_detector().pc`; measured bitwise through the
     binary).
   - **circmask quirk inverted** (`> 0 -> True`): dies by
     `test_positive_circmask_maps_to_false_with_warning` (the
     warning names the lost processor-side mask, D6).
   - **`to_detector` defaults `sample_tilt`**: dies by
     `test_to_detector_requires_sample_tilt` (TypeError).
   - **from_kwargs delta uses `px_size`**: dies by
     `test_from_kwargs_passes_its_own_sanity_check` on the nickel
     detector (px_size 1.0 -> detW 0.06 mm would raise).
   - **patdims order swapped**: dies by the rectangular
     from_kwargs/to_detector round trip (shape `(h, w)` vs
     `(w, h)`).
   - **scandims order swapped**: dies by
     `test_from_kwargs_scandims_order` (non-square navigation
     shape; the 3x3 fixtures cannot catch it).
   - **probe regularity uses header count** (`n_patterns`): dies by
     the `all_present=False` fixture test.
   - **probe dedupes with a tolerance**: dies by the staggered
     fixture (0.5-offsets are distinct exact values) plus the
     near-duplicate fixture (1.0 vs 1.0+1e-12 stay distinct).
   - **probe drops the coordinate arrays**: dies by the irregular
     fixture's list assertions.
4. Fix, re-run (`-n 4`), `pre-commit run --files <changed>`,
   `--doctest-modules src/kikuchipy/indexing/_spherical`, coverage
   of the new modules **== 100 %** (recorded, the Phase 1-6
   precedent), `sphinx-build -b html` + `-b linkcheck` exit 0.

## 6. Open questions -- decided 2026-09-02 (autonomous mode), flagged for review

1. **The writer always adds `Manufacturer`** (deviation from
   `pattern_repack.cpp`, which writes none): without it the file is
   dead on arrival (`IndexEBSD.exe` exit 1, measured). The
   alternative -- a `manufacturer=None` passthrough reproducing the
   C++ byte-for-byte -- ships a file no EMSphInx program accepts;
   rejected. The deviation is named in the licence block and
   docstring.
2. **Defaults `manufacturer="EMsoft"` (writer) and
   `vendor="Bruker"` (namelist `from_kwargs`)**: the writer default
   makes `/patterns` byte-identical to `signal.data` (auto
   `flip=False`) -- the cheapest write and the strongest CI
   assertion; the namelist default makes `pctr` kikuchipy's
   `pc_average` verbatim (identity conversion). This mixed pairing
   **is the canonical Phase 10 route** (plan 0.3) and the acid
   test's route; its bands are anchored on its own measurements
   (median 0.7245 / max 0.9479 / scores mean 0.6283). The pairing
   is valid by construction -- the file `Manufacturer` controls
   only the read-flip, the nml `vendor` only the PC interpretation
   (decoupling measured: EDAX-manufacturer file + Bruker-vendor
   nml indexes correctly). The writer keyword is named
   `manufacturer` (the dataset's literal name; kikuchipy's own
   word for the field) to end the collision with the namelist
   `vendor` key. Alternatives rejected: one shared default
   ("Bruker" everywhere forces flipped writes and weakens the
   byte-equality test to a flipped comparison; "EMsoft" everywhere
   puts pixel-unit pctr in every nml where fractional kikuchipy pc
   is the project's lingua franca).
3. **`get_scan_info` carries the kikuchipy header only** (no
   EMSphInx notice): it executes and transcribes no EMSphInx code
   -- kikuchipy's own reader (pre-dating this project) supplies the
   footers, and the distinct-set/product diagnostic is an idea,
   not expression. The program is cited as the modelled-on
   reference. If the review disagrees, the fallback is the full
   delimited notice; flagged for exactly that call.
4. **Probe return type is a plain `dict`** (documented keys):
   matches the plugin's list-of-dict reader convention, avoids a
   public class with its own docs page for a diagnostic helper.
   Rejected: dataclass (nicer autocomplete, more surface), printing
   like the C++ (kikuchipy functions return, not print). Named
   `get_scan_info` because the payload is more than the grid.
5. **Non-uint8 writes warn instead of raising**: the C++ program
   writes uint16/float32 happily -- refusing would deviate from the
   ported contract -- but the shipped reader corrupts them
   (measured), so silence would set a trap. `UserWarning` with the
   measured consequence named. Phase 10 pins uint8.
6. **`to_string` keeps the C++ `.6g` stream precision, no
   full-precision mode**: byte-parity with `IndexEBSD -t` is worth
   more than the ~1e-6 fractional pctr loss (~3e-4 px, vs the
   0.33 deg mean-PC floor); *reading* full-precision namelists
   loses nothing (measured: the 8-decimal acid nml parses
   exactly). Revisit only if Phase 10's regression needs
   sub-1e-6 pctr reproducibility (it will not -- it writes the
   nml it reads back).
7. **`sample_tilt` is a required keyword on `to_detector`**: the
   nml has no sample tilt; `IndexEBSD` binds it from the master's
   sig (`idx.hpp:218`), and Phase 6 measured a 5-deg silent
   mis-index from a tilt mismatch. A 70.0 default would be exactly
   that trap. Rejected: reading it from a harmonics argument
   (couples the detector builder to a master object it does not
   otherwise need; the caller passes the public
   `harmonics.sample_tilt` explicitly).
8. **String-valued `scandims` and non-empty `roimask` raise**:
   scan-file reading (`.ang`/`.ctf`/H5 `OrientationMap`) and the
   ROI grammar are mission out-of-scope; parse-and-store keeps
   round-trip fidelity, conversion refuses loudly. Rejected:
   silent ignore (data loss), porting `orientation_map.hpp`
   (a phase of its own, no consumer).
9. **The generic parser stays private** (`_NameList`): the public
   surface is the EBSD namelist class; no other EMSphInx namelist
   exists in scope. Public export deferred until a second consumer
   appears.
10. **Regularity over *present* patterns** in `get_scan_info`
    (`n_patterns_present`, with `all_patterns_present` reported
    separately): EBSPDims cannot open sparse files at all (its
    reader throws on a zero offset), so any sparse behaviour is an
    extension; counting absent header slots as "irregular" would
    conflate two different diagnoses the dict already separates.
11. **No new shipped fixtures**: the in-package `patterns.ebsp` +
    `ni_small_20kv_bw384.sht` cover the real-data paths; synthetic
    regular `.ebsp` variants come from the existing conftest
    fixture, and the two inexpressible ones (staggered,
    near-duplicate) from a module-local helper (plan 3.2); the
    `-t` template is a string constant in the test module
    (provenance-commented), not a data file.
12. **`batch_size <-> chunksize` and `n_thread` unmapped in
    `to_kwargs`**: `batchsize` 0 means auto in both worlds
    (`BatchEstimate` / Phase 6 default `chunksize=None`);
    `nthread` has no kikuchipy knob (dask's scheduler owns
    workers) -- documented in `to_kwargs`, not smuggled into an
    unrelated parameter. `from_kwargs` still accepts
    `n_thread`/`batch_size` (default 0) so deterministic
    `nthread=1 batchsize=1` namelists can be built without
    hand-editing (the acid test and Phase 10 need exactly that).
13. **Paths stored raw, prefixed forms derived**
    (`pat_path`/`master_paths`): the C++ stores `ipath`-prefixed
    values and therefore double-prefixes on its own
    `from_string(to_string(x))`; porting that storage would break
    the round-trip contract for any non-empty `ipath`. The
    double-`ipath` psymfile quirk remains observable on the
    derived property. Rejected: faithful prefixed storage
    (round-trip broken), dropping the quirk (fidelity loss).
14. **`patdset`/`scanname` always optional, always consumed**: the
    C++ requiredness depends on `H5::H5File::isHdf5(patFile)` --
    a filesystem probe (measured both ways) that a pure parser
    must not reproduce. Always-consume also kills the spurious
    unused-token warning and lets `scanname` round-trip. Rejected:
    filesystem probing (non-deterministic parsing), always
    required (rejects namelists the binary accepts).
15. **The writer follows kikuchipy's `overwrite`/suffix/directory
    conventions** (`MasterPatternHarmonics.save`,
    `io/_io.py:439-460`): silent truncation of an existing file
    would be the only writer in the package that does it.
16. **The `.ang` reader for gated tests is
    `np.loadtxt(path, comments="#")`** (shared `read_ang`
    fixture): `orix.io.load` reads the file but warns
    (8-column/vendor guess) and names the score/iq columns
    `unknown2`/`unknown1`; the plain array is what the assertions
    index. Phase 10 reuses the fixture.

## 7. Commit and PR

1. Signed commits in gate order: spec + section-0 amendments;
   failing tests; implementation; review fixes. Tick the Phase 9
   boxes in `specs/roadmap.md` with the measured numbers as each
   gate completes; push; PR **#10** into fork `develop` with the
   template, the GPL-only statement (BSD opt-out impossible for
   `_namelist.py`/`_pattern_repack.py`), and the CHANGELOG entries.
