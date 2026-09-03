# Phase 9 -- `sht-interop`: requirements

Branch `sht-interop` (roadmap Phase 9, **slimmed 2026-09-02 to the
interop pieces**; the visualisation half is deferred with Phase 8
until after Phase 11). Everything Phase 10 needs to drive
`IndexEBSD.exe` from kikuchipy-written files is delivered here.

Revised 2026-09-02 after the fidelity and conventions reviews; the
re-measurements backing the revision are in `validation.md`
"Recorded results", second dated section.

## Scope

In scope:

- **`kp.indexing.write_emsphinx_patterns(filename, signal, *,
  manufacturer="EMsoft", binning=1, bin_to_float=False, flip=None,
  overwrite=None)`** in the new module
  `src/kikuchipy/indexing/_spherical/_pattern_repack.py` -- the port
  of `EMSphInx/programs/pattern_repack.cpp` @ 60f3517 (`binFloat`
  `:50-67`, `binAvg` `:76-98`, `flipPat` `:105-113`, the HDF5 layout
  `:183-208`, the write loops `:210-293`) **plus the root
  `Manufacturer` dataset the C++ program itself omits** (D2 defect;
  without it `IndexEBSD.exe` refuses the file -- measured).
  Filename-first argument order follows the io-writer convention
  (`file_writer(filename, signal)`, `np.save(file, arr)`).
- **`kp.io.plugins.oxford_binary.get_scan_info(filename)`** -- the
  `EBSPDims` equivalent (`programs/ebsp_dims.cpp:41-106`): distinct
  `beam_x`/`beam_y` sets from the `.ebsp` pattern footers and the
  irregular-grid diagnostic (`sx.size() * sy.size() != numPat`,
  `:97-103`), built on kikuchipy's own `OxfordBinaryFileReader`
  (D3). Named `get_scan_info` (not `get_scan_grid`) because the
  dict also carries dtype/bytes/version facts beyond the grid.
- **`kp.indexing.EMSphInxNamelist`** in the new module
  `src/kikuchipy/indexing/_spherical/_namelist.py`: read/write of
  `IndexEBSD` `.nml` files -- a private generic parser `_NameList`
  porting `include/util/nml.hpp` (`read` `:292-426`, the typed
  `Variant` get/set `:226-249`, `fullyParsed`/`unusedTokens`
  `:505-520`), pinned by the `test/util/nml.cpp` round-trip suite and
  by error probes run through the shipped binary (D4) -- plus the
  EBSD field set, defaults, `parse_nml` semantics, `sanityCheck` and
  the `to_string` template of `include/modality/ebsd/nml.hpp`
  (`:52-160`, `:186-218`, `:236-315`, `:320-470`, `:621-639`),
  byte-pinned against the captured `IndexEBSD -t` template (D5).
- **`EMSphInxNamelist.to_kwargs()` / `.from_kwargs()` /
  `.to_detector()`** mapping namelist fields to
  `SphericalIndexer`/`EBSD.spherical_indexing` arguments and an
  `EBSDDetector`, with the four vendor pattern-centre conversions of
  `include/modality/ebsd/detector.hpp` (`patternCenter` `:85`, TSL
  `:249-254`, Oxford `:261-267`, Bruker `:274-279`; selected as
  `idx.hpp:221-229`) ported as formulas -- **not** delegated to
  kikuchipy's `pc_*` helpers, which deviate on rectangular detectors
  (D6, measured).
- **Tests**: `tests/test_indexing/test_spherical_namelist.py`
  (`_namelist.py`) and
  `tests/test_indexing/test_spherical_pattern_repack.py`
  (`_pattern_repack.py`) -- one test module per source module, the
  per-module precedent of `test_spherical_sht_file.py` etc.; all CI
  assertions on file bytes/h5py; additions to
  `tests/test_io/test_oxford_binary.py` (the probe); and the
  local-gated `IndexEBSD.exe`/`PatternRepack.exe`/`EBSPDims.exe`
  end-to-end tests behind `KIKUCHIPY_EMSPHINX_DIR` (skip helpers
  promoted from `tests/test_indexing/test_spherical_sht_file.py:
  160-190` into shared `conftest.py` fixtures -- D9) (D7, D9).
- **CHANGELOG** `Added` entries (three, one per public name, PR
  **#10**), constitution amendments listed in plan section 0 (not
  yet applied), research addenda.

Out of scope (confirmed against `mission.md`'s v1 list): the EMSphInx
**ROI string grammar** (`roimask` is parsed and round-tripped as a
string; a non-empty value raises in `to_kwargs`), **`Geometry::ecp()`**,
the **`.ctf` writer**, the **IPF/XC PNG writers** (`ipfmap`/`qualmap`
are round-tripped as file names only), EMsoft raw **`.data`** pattern
input to the repack equivalent, reading scan files for
`scandims`/`findScanFile` (a string-valued `scandims` raises
`NotImplementedError`; `xtal/orientation_map.hpp` is not ported),
`psymfile` consumption (parsed and stored; Phase 8), the whole
visualisation half of the original Phase 9 (sht2png equivalents,
`plot_power_spectrum`, `SphericalBackProjector.plot`, xcorr volume
plot, Sphinx-Gallery example, `extractBunge` -- deferred with Phase 8
per the 2026-09-02 re-scope), and `IndexEBSD.exe` accuracy *parity*
gates (Phase 10; this phase's acid test proves the files are
**accepted and index sanely**, D7).

## Decisions

Every "measured" number below was produced on 2026-09-02 by the
drafting probes (`acid_test.py`, `acid_test2.py`, `probes3.py`,
`probes4.py` in the session scratchpad `p9/`, not committed) or by
the revision probes (`probe_pure.py`, `probe_binary.py`, scratchpad
`p9r/`; recipes and full outputs in `validation.md` "Recorded
results") against the built binaries in
`c:/Users/westraadt.1/Repos/EMSphInx/build/Release/` @ 60f3517.

### D1 -- `write_emsphinx_patterns`: contract (frozen)

- ```
  write_emsphinx_patterns(
      filename: str | Path,
      signal: EBSD | LazyEBSD,
      *,
      manufacturer: str = "EMsoft",
      binning: int = 1,
      bin_to_float: bool = False,
      flip: bool | None = None,
      overwrite: bool | None = None,
  ) -> None
  ```
  The keyword is `manufacturer` because that is literally the name
  of the dataset written and the word kikuchipy already uses for
  this field (`io/_io.py:256-269`, every plugin
  `specification.yaml`); it also stops the collision with the
  namelist's `vendor` key, which takes a **different** value set
  and controls a different thing (D6).
- **File handling** (the kikuchipy writer convention --
  `MasterPatternHarmonics.save` and `signal.save`,
  `io/_io.py:439-460`): `".h5"` is appended when `filename` has no
  suffix; the parent directory is created via
  `kikuchipy.io._util._ensure_directory`; an existing file is
  handled by `overwrite` -- `None` asks via
  `kikuchipy.io._util._overwrite` (not overwriting when input is
  unavailable), `True` truncates, `False` returns without writing.
- Writes the two-object HDF5 file EMSphInx's
  `PatternFile::Read`/`GetVendor` consume:
  - root scalar dataset **`Manufacturer`**, variable-length
    **ASCII** string (`H5::StrType(0, H5T_VARIABLE)` on a scalar
    dataspace -- `pattern.hpp:608-637` reads it; h5py:
    `string_dtype(encoding="ascii")`). The charset is load-bearing:
    h5py's default vlen **UTF-8** string makes `IndexEBSD.exe` exit
    1 with a misleading `H5 error attempting to read EBSD patterns
    ... H5Dread failed` (measured; `GetVendor`'s fallback read at
    `pattern.hpp:631-635` throws into `PatternFile::Read`'s outer
    handler `:519`) -- the CI pin asserts the ASCII cset, not
    merely "variable-length" (D9). Accepted values are exactly the
    reader's flip-table strings (`pattern.hpp:465-471`):
    `{"EDAX", "Oxford", "Bruker", "Bruker Nano", "DREAM.3D",
    "EMsoft"}`; any other value raises `ValueError` here because
    `IndexEBSD.exe` errors with `unknown EBSD vendor: <s>` (measured
    with `"kikuchipy"`). Note this is a **different set** from the
    namelist `vendor` whitelist (D6).
  - **`/patterns`**, shape `(n, h_out, w_out)`, **contiguous**
    layout with **`H5D_ALLOC_TIME_EARLY`** (`pattern_repack.cpp:201`;
    h5py low-level `h5p`/`h5d` API -- measured on this machine:
    `get_layout() == 1` (CONTIGUOUS), `get_alloc_time() == 1`
    (EARLY), `id.get_offset()` defined (6160 for the acid file)),
    no chunks, **no filters** -- filters are the functional
    constraint: a gzip-compressed dataset makes `IndexEBSD.exe`
    exit 1 (`H5Dread failed`, measured), while chunked-unfiltered
    and contiguous alloc-late files are read fine, so
    contiguous+EARLY is byte-parity with `PatternRepack.exe` and
    zero-filters is the guard (D2). Dtype = the signal dtype in
    **native byte order** (or float32, below). `n =
    prod(navigation_shape)` in row-major navigation order (the C++
    writes each pattern at its `extract` scan index, `:214`); a
    0-d navigation signal writes `n = 1`.
- **Dtypes**: `uint8`, `uint16`, `float32` (the `ImageSource::Bits`
  set, `:184-190`); anything else raises `ValueError`. A non-native
  byte order on the input (`>u2`, `>f4`) is **cast to native**
  before writing (`dtype.newbyteorder("=")` equivalent):
  `PatternFile::Read` compares against `NATIVE_*` types
  (`pattern.hpp:476-480`) and a big-endian `/patterns` makes
  `IndexEBSD.exe` exit 1 with `only uint8, uint16, and float hdf
  patterns are supported` (measured). When the written dtype is not
  `uint8` a **`UserWarning`** is emitted: EMSphInx 60f3517 reads
  HDF5 patterns through a buffered `NATIVE_UINT8` read and corrupts
  non-uint8 data (D2, measured: a uint16 twin of the acid file
  indexes 38.9 deg median wrong at scores ~0.21 vs 0.63-0.67). The
  write itself proceeds (the C++ program writes such files happily;
  only the warning is ours).
- **Binning** (`binning >= 1`, must divide both signal dimensions,
  else `ValueError` -- `:140-143`, `:175-178`):
  - `bin_to_float=False` -> **`binAvg`** (`:76-98`): per output
    pixel, the `binning^2` block sum accumulated in float64, divided
    by `binning**2`, and for integer dtypes rounded **half away from
    zero** (`std::round`; values are non-negative, so
    `floor(x + 0.5)` -- **never `np.round`**: measured bitwise
    against `PatternRepack.exe <ebsp> out.h5 2` on the in-package
    `patterns.ebsp`, where banker's rounding differs on 1003 of
    8100 pixels), cast back to the input dtype. `binning == 1`
    copies.
  - `bin_to_float=True` -> output dtype **always float32**: at
    `binning > 1` via **`binFloat`** (`:50-67`) -- the block **sum**
    (not average) accumulated in float32 (`:191-192` switches the
    dataset dtype); at `binning == 1` as a pure cast. The
    `binning == 1` cast is a **deliberate completion of dead
    code**: the C++ `main` never routes `binning == 1` through
    `binFloat` (the raw-copy path `:212-218` writes the input
    dtype uncast), so "float output" there is unreachable -- like
    the `Manufacturer` deviation, we complete the contract rather
    than copy the gap. The whole mode is unreachable in the shipped
    binary (`const bool binToFloat = false`, `:116`), so it is
    pinned against a NumPy oracle, not against `PatternRepack.exe`
    (D2). For float32 *inputs* the oracle defines the accumulation
    order (NumPy pairwise summation; the C++ per-row
    `std::accumulate` may differ in ulps -- the path is dead in the
    binary, the oracle is authoritative).
- **Flip** (`flipPat`, `:105-113`: whole-row vertical mirror). The
  C++ applies flip before binning; for divisible binning the two
  operations **commute exactly** (block partitions are preserved
  under row reversal and block sums are order-independent --
  verified bitwise for `binning` 2-6 on the nickel fixture, both
  `binAvg` and `binFloat`), so the order is unobservable and the
  contract is stated as the equivalence
  `binned(flip(x)) == flip(binned(x))`, not as an internal order.
  - `flip=None` (default) resolves **per manufacturer** so that the
    pattern EMSphInx ultimately interpolates is the **row-reversed
    kikuchipy pattern** -- EMSphInx's internal convention is
    origin-bottom-left (`Geometry::flip`, `detector.hpp:62`, means
    "origin top left", and it is set from the vendor table at
    `idx.hpp:231`), which is why `PatternRepack.exe` hard-codes
    `flip=true` (`:117`). Vendors whose `vendorFlip` is true
    (`{"EDAX", "EMsoft"}`, `pattern.hpp:465`, `:470`) get
    **unflipped** writes -- the read chain applies the flip
    (measured stdout: `Vertical Flip: true` on the default route);
    vendors whose `vendorFlip` is false (`{"Oxford", "Bruker",
    "Bruker Nano", "DREAM.3D"}`) get **pre-flipped** writes
    (measured stdout: `Vertical Flip: false`). Measured both ways
    on `nickel_ebsd_small`: rows-as-written + `Manufacturer` EMsoft
    or EDAX and rows-flipped + `Manufacturer` Bruker both index
    correctly (refined medians 0.725 resp. 0.713 deg vs the stored
    xmap); the *wrong* pairing indexes ~39.6 deg median wrong at
    scores ~0.22 (D7). An explicit `flip=True/False` overrides
    (documented: overriding breaks orientation parity; `flip=True`
    is `PatternRepack.exe`'s own hard-coded behaviour).
  - The two correct routes are **equivalent but not bitwise
    identical** downstream (the C++ flips vendor-flip files at
    interpolation time, not in memory): measured per-point
    differences ~0.01-0.02 deg between the routes. **The canonical
    route is the writer default** (`manufacturer="EMsoft"`,
    unflipped): Phase 10's byte-stable references, the acid test
    and every pinned band in D7 use it (plan 0.3 writes this into
    the roadmap Phase 10 box).
- **Default `manufacturer="EMsoft"`**: with it, `flip` resolves to
  `False` and `/patterns` is **byte-identical to
  `signal.data.reshape(n, h, w)`** -- the cheapest write, the
  easiest CI byte assertion, and the vendor string EMSphInx's own
  ecosystem uses for simulated-pattern files. (The namelist's
  default `vendor` is independently `"Bruker"` -- D6; the two knobs
  control different things -- file flip vs pattern-centre
  interpretation -- and any combination is valid, measured: an
  EDAX-`Manufacturer` file indexed with a Bruker-`vendor` namelist
  is one of the two correct routes above.)
- **Lazy signals** are streamed chunk-wise into the pre-allocated
  dataset (alloc-time-early makes the dataset writable slab by
  slab); no full-map materialisation (pinned by a `tracemalloc`
  peak bound, the Phase 6 pattern -- band measured then pinned at
  implementation time, D9). Deterministic output: the file bytes
  depend only on the data, parameters and HDF5 library version
  (dataset bytes are pinned in CI, whole-file md5 is not -- h5py
  superblock details may vary across versions).
- The C++ prints a summary and timings (`:154-172`); the port is
  silent (kikuchipy io convention). Not ported: the `.up1/.up2`,
  `.data` and `.ebsp` *input* paths (`PatternFile::Read` -- the
  input is a kikuchipy signal; `kp.load` covers `.ebsp`), the
  stdout report, the `argv` CLI.

### D2 -- Recorded EMSphInx facts the contract rests on (measured)

- **`PatternRepack.exe` output is not consumable by
  `IndexEBSD.exe`**: the program writes only `/patterns`
  (`pattern_repack.cpp:199-202`; h5py-verified root keys
  `['patterns']`), and `IndexEBSD.exe` on that file exits 1 with
  `repacked_ni_small.h5 doesn't have a Manufacturer string`
  (`pattern.hpp:623`). The kikuchipy writer therefore **always
  writes `Manufacturer`** -- a deliberate, loudly documented
  deviation from the program it ports (the C++ contract is
  completed, not copied).
- **The HDF5 memory-map path of `PatternFile::Read` is dead code**:
  the gate requires `0 != props.getNfilters()` (`pattern.hpp:494`)
  -- inverted relative to its own comment ("we can't memory map
  compressed datasets"), and contiguous datasets can never carry
  filters, so the condition is unsatisfiable and every HDF5 pattern
  file takes the `BufferedPatternFile` branch (`:511-517`).
  Consequence 1: the reader's real constraint is **zero filters**,
  not layout -- measured: chunked+gzip exits 1 (`H5Dread failed`),
  chunked-without-filters and plain contiguous alloc-**late** files
  both exit 0 -- so `alloc-time-early` is kept for **byte-layout
  parity with `PatternRepack.exe`** (measured: layout 1 / alloc 1 /
  offset 2144 on its output), not because the reader needs it, and
  the zero-filter pin is the functional guard (compression is the
  obvious lazy-write temptation).
  Consequence 2: the buffered branch reads with
  `H5::PredType::NATIVE_UINT8` regardless of the dataset dtype
  (`:515`), so **only uint8 HDF5 patterns are read correctly** --
  verified end-to-end: the uint16 acid twin indexes garbage
  (D1). The dtype whitelist itself compares against `NATIVE_*`
  types (`:476-480`), so byte-swapped datasets are rejected
  outright (`>u2` -> exit 1 `only uint8, uint16, and float hdf
  patterns are supported`, measured -- hence D1's native-order
  cast). Phase 10's regression files must be uint8.
- `PatternRepack.exe`'s repack of the in-package
  `src/kikuchipy/data/oxford_binary/patterns.ebsp` (the
  `nickel_ebsd_small` patterns): `/patterns` `(9, 60, 60)` uint8,
  contiguous, alloc EARLY, data offset 2144, file 34544 B
  = 2144 + 9*3600, and the rows equal
  `nickel_ebsd_small().data[..., ::-1, :]` (the hard-coded
  `flip=true`) -- the bitwise oracle for the writer's
  `flip=True` route and for `binAvg` (bin 2: `(9, 30, 30)` uint8,
  10244 B, bitwise equal to the frozen floor(x+0.5) recipe).
- `GetVendor` details ported into the writer's docstring: it also
  accepts a `" Manufacturer"` dataset name with a stray leading
  space (EDAX quirk, `:617`) and falls back to a 128-byte
  fixed-length string read (`:631-635`) -- so a fixed-length ASCII
  `Manufacturer` would still be read (measured: `S6` exits 0), but
  a vlen **UTF-8** one is fatal with a misleading error (D1,
  measured), so the vlen-**ASCII** form is the contract (CI pins
  dtype *and* cset).

### D3 -- `get_scan_info`: the EBSPDims probe (frozen)

- `kp.io.plugins.oxford_binary.get_scan_info(filename: str | Path)
  -> dict`, implemented in the plugin's `_api.py` on top of
  `OxfordBinaryFileReader` (no EMSphInx code is executed or
  transcribed -- D8 licence stance), exported through the plugin's
  `__init__.pyi` next to `file_reader` (the first
  non-`file_reader`/`file_writer` export in any of the 15 io
  plugins -- deliberate, recorded; a diagnostic probe belongs next
  to the reader it describes).
- Returned keys (frozen):
  - `"n_patterns"` -- header slot count (the reader's guess),
  - `"n_patterns_present"`, `"all_patterns_present"`,
  - `"signal_shape"` `(nrows, ncols)`, `"dtype"`,
    `"pattern_bytes"` (`n_bytes` per pattern),
    `"total_bytes"` (`pattern_bytes * n_patterns_present`),
  - `"version"` (`.ebsp` version),
  - `"beam_x"`, `"beam_y"` -- **sorted unique** float64 arrays of
    the per-pattern footer coordinates over the *present* patterns
    (the EBSPDims `std::set<double>` semantics, `ebsp_dims.cpp:78`,
    `:86-87`: exact-value distinctness, no tolerance), or `None`
    when the footer has no such field (version 0, or a false
    `has_beam_*` flag),
  - `"is_regular_grid"` -- `len(beam_x) * len(beam_y) ==
    n_patterns_present` (the EBSPDims product test `:97` applied to
    the patterns actually present -- EBSPDims itself cannot open a
    missing-pattern file at all, its `OxfordPatternFile` ctor
    throws, so the extension to sparse files is ours and flagged);
    `False` when either set is `None`.
- The irregular-grid *diagnostic* is the data itself: EBSPDims
  prints the two coordinate lists only when irregular (measured on
  a staggered-row synthetic `.ebsp`: `found 6 x and 2 y
  coordinates` then `X: 0 0.5 1 1.5 2 2.5` / `Y: 0 1`); the probe
  always returns the sets and the flag, and the docstring shows the
  EBSPDims-equivalent report. Measured on the in-package
  `patterns.ebsp`: 9 patterns, 60x60, 8-bit, 3600 B/pattern,
  3 x and 3 y coordinates, regular -- exactly EBSPDims' output.
- Compressed `.ebsp` patterns need no new handling: kikuchipy's
  reader already raises `NotImplementedError` on them
  (`oxford_binary/_api.py:113-115`), which the probe inherits
  (EBSPDims' `pattern.hpp:844-852` sparse/compressed throw is the
  C++ analogue).
- Not ported: the batch-100 extraction loop (`:76-92`; kikuchipy's
  memmap already exposes the footers -- `np.unique` over the memmap
  fields is equivalent and measured to agree), the stdout report,
  the MB/GB pretty-printer.

### D4 -- `_NameList`: the generic namelist parser (frozen)

Private class in `_namelist.py`, a faithful port of
`util/nml.hpp` `NameList::read` (`:292-426`) with the `Variant` type
rules; every rule below is pinned by the ported `test/util/nml.cpp`
suite (`:63-345`, **eleven** error cases) and the starred ones were
additionally confirmed through `IndexEBSD.exe` (exit 1 + message,
"Recorded results"):

- Line 1 is skipped and must not contain `=` (*"namelist files
  cannot have key value pairs in the first line"*); comment lines
  start with `!` as the **first character of the line**\*
  (`nml.hpp:307` tests `line.front()`; the doc comment at `:291`
  says "first character after white space" and the code contradicts
  it -- a recorded C++ doc/code mismatch. Measured: an indented
  `" ! comment"` line is *not* a comment and exits 1 with `bad
  delimeter (expected '=') ...`); empty lines and the exact line
  `" /"` are skipped. **Whitespace-only lines are not skipped**\*
  (`:306` is a literal `line.empty()` test; measured: a `"   "`
  line exits 1 with `error parsing line '   ' from name list` --
  the idiomatic `if not line.strip(): continue` would diverge).
- Every key line: **one leading space** required\* -- zero spaces or
  a tab raise *"missing leading space in namelist line ..."*, while
  **two or more** leading spaces raise the *different* message
  *"error parsing line '...' from name list"*\* (the `noskipws`
  key extraction fails on the second space; both measured and both
  pinned); `key` lowercased (case-insensitive lookups, `ToLower`
  `:284-287`), duplicate keys raise\*, `=` delimiter required, a
  **comma must end every entry line except the last**\* (the
  reverse-scan rule `:313-327`).
- Values: single-quoted strings (backslash-escaped `\'` inside;
  comma-separated lists; a bare `''` is the empty string; a
  double-quoted or unquoted string raises\*); otherwise
  comma-tokenised `.true.`/`.false.` (case-insensitive after the
  lowercase transform), ints, doubles (`tryParse` exact-consumption
  semantics `:273-279`: `"12345"`/`"+12345"`/`"-12345"` int,
  `"1.23e4"` double); a mixed int/double list promotes ints to
  doubles (`:413-419`); mixing bools with numbers raises
  (`:410`).
- **Quoted-list whitespace quirk** (ported): from the **second**
  string of a quoted list onward, *all* whitespace inside the
  string is stripped -- `nml.hpp:364-368` sets sticky `std::skipws`
  before the inter-string delimiters, so the per-character
  extraction at `:347` skips whitespace from then on. Measured
  (decisive): `masterfile = 'a.sht', 'ni small stripped.sht',`
  opens `nismallstripped.sht` (exit 0 with that file on disk); the
  same spaced name in *first* position is kept verbatim. Ported
  faithfully and pinned; `EMSphInxNamelist.to_string`/`write` guard
  the writer side (D5) -- a real Windows-path hazard.
- Typed access, C++-strict: `get_bool`/`get_int`/`get_string` accept
  only their own type; **`get_double` silently accepts ints**
  (`:228`); wrong type raises with the C++ message (*"stored type
  isn't integer"* etc.); a missing key raises *"couldn't find `X'
  in namelist"* (`:440-457`). Vector variants `get_*s`.
- Used-flag bookkeeping: `fully_parsed()` / `unused_tokens()`
  (`:505-520`; comma-joined, lowercased keys, map order =
  lexicographic) -- `EMSphInxNamelist.read` warns with the unused
  list exactly as `IndexEBSD` does (measured\*: an `extrakey = 42`
  line still indexes, exit 0, with `warning: some namelist
  parameters weren't used: extrakey`).
- Not ported: `NML_USE_H5` (`writeParameters`/`writeFile`,
  `:525-575` -- they serve the *output* scan file, Phase 10's
  reading concern, not ours to write), the raw `fileLines` copy
  beyond what `unused`-warning parity needs (we keep the raw text
  for round-trip introspection -- a port convenience, documented).

### D5 -- `EMSphInxNamelist`: fields, defaults, writer (frozen)

- Attributes mirror the C++ struct (`nml.hpp:54-88`) with pythonic
  names kept close to the namelist keys: `ipath`, `pat_file`,
  `pat_dset`, `master_files` (list), `psym_file`, `pat_dims`
  `(width, height)`, `circ_rad`, `gaus_bckg`, `n_regions`, `delta`,
  `vendor`, `pctr` `(3,)`, `thetac`, `scan_dims` `(2,)`,
  `scan_steps` `(2,)`, `scan_file`, `scan_name` (the C++
  `scanFile`/`scanName`, `:72-73`), `roi_mask` (string, opaque),
  `bw`, `normed`, `refine`, `n_thread`, `batch_size`, `opath`,
  `data_file`, `vendor_file`, `ipf_name`, `qual_name`. `defaults()`
  reproduces `Namelist::defaults()` (`:186-218`) exactly (640x480,
  `circRad` -1, `nRegions` 10, `delta` 50, EMsoft `(0, 0, 15000)`,
  `thetac` 10, 256x256x1x1, `bw` 68, normed/refine true,
  `SphInx_Scan.h5`, `reindexed.ang`, `ipf.png`, `qual.png`).
- **Path storage is raw** (deviation, recorded): the C++
  `parse_nml` stores `ipath`-prefixed `patFile`/`masterFiles`/
  `scanFile` (`:241`, `:247-248`, `:252`) and `to_string` writes
  the stored values back, so the C++ `from_string(to_string(x))`
  **double-prefixes** whenever `ipath` is non-empty. The port
  stores every path exactly as written in the file and exposes the
  prefixed forms as derived read-only properties `pat_path` /
  `master_paths` (with the double-`ipath` quirk reproduced *there*:
  when `psym_file` is non-empty, `pat_path` carries `ipath` twice,
  `:247` -- ported observable, recorded; harmless when `ipath` is
  empty, the recommended usage). This keeps the round-trip contract
  exact for non-empty `ipath` (named test) while still showing
  exactly what the binary would open.
- `read(path)` / `from_string(text)` implement `parse_nml`
  (`:236-315`) faithfully, with one recorded deviation:
  - optional keys by try/except exactly as the C++: `ipath`,
    `psymfile`, `opath`, `vendorfile`, `ipfmap`, `qualmap` (a
    missing `vendorfile`/`ipfmap`/`qualmap` means "no such
    output");
  - **`patdset` and `scanname` are unconditionally optional and
    always consumed** -- a recorded deviation: the C++ reads
    `patdset` only `if(H5::H5File::isHdf5(patFile))` (`:242-246`)
    and `scanname` only in the string-`scandims` branch
    (`:253-254`), making requiredness **filesystem-dependent**
    (measured: an existing HDF5 `patfile` without `patdset` exits 1
    `couldn't find 'patdset' in namelist`; a *non-existent*
    `patfile` without `patdset` parses fine and fails later at
    open). A pure parser cannot mirror that without touching the
    filesystem; always-consume also avoids a spurious unused-token
    warning and lets a `scanname`-carrying namelist round-trip
    (the C++ would warn-and-drop it when `scandims` is numeric).
    Everything else required (missing-key error of D4);
  - **`scandims`** as 3 or 4 doubles (3 -> square step, `dims[2]`
    reused, `:266-267`; non-integer scan dimensions raise
    `:263`); a *string* `scandims` (scan-file route, `:252-256`)
    raises `NotImplementedError` (out of scope -- no
    `OrientationMap` port). `scan_file`/`scan_name` are therefore
    never populated by `read` in v1; they are storable, `scan_name`
    round-trips through `to_string` (below), `scan_file` does not
    (the C++ `to_string` always writes numeric `scandims`, `:397`
    -- "fortran version can't be string");
  - the namelist `vendor` whitelist (`:290-295`): `{"EMsoft",
    "EDAX", "tsl", "Oxford", "Bruker"}` -- **lowercase `"tsl"`,
    and `"TSL"` is rejected** even though the template's comment
    advertises it (C++ quirk, `"tsl"` appears twice in the C++
    condition; recorded and ported);
  - `sanityCheck` (`:621-639`) as `sanity_check()`, all
    **thirteen** checks with the C++ bounds and messages (empty
    patfile, empty master list, per-file empty master name, pattern
    dims `[2, 16384]`, `circRad >= -1`, `nregions` in
    `[0, min(patdims)]`, detector width `delta * patDims[0] / 1000`
    in `[5, 90]` mm, `thetac` in `[-60, 60]`, positive scan dims,
    `bw` in `[16, 512]`, non-negative `nthread`, non-negative
    `batchsize`, non-empty datafile), run at the end of `read`
    exactly as `parse_nml` does. The three negativity checks are
    **live** in the binary (the struct fields are `int32_t`,
    `nml.hpp:61-63`, `:79-80`; measured: `nregions = -5` exits 1
    `unreasonable AHE nregions`, `nthread = -1` exits 1 `negative
    thread count`), so they are ported and probed like every other
    bound. The unused-token warning after it.
- `to_string()` / `write(path)` port the commented template of
  `Namelist::to_string` (`:320-470`) verbatim: the fixed first line
  ` &EMSphInx` (the C++ default parameter `nml = "EMSphInx"`,
  `:130`; no name parameter in v1), section banners, comments, key
  ordering and padding, the conditional omission of the optional
  blocks (`ipath`, `patdset`, `psymfile`, `scanname`, `opath`,
  `vendorfile`, `ipfmap`, `qualmap`), doubles formatted as C++
  `std::ostream::operator<<` does (**`format(v, ".6g")`**), bools
  as `.TRUE.`/`.FALSE.`, the master-file list with its trailing
  `", "`, LF line endings, **and the quirk that the closing
  `" /"` line is emitted only inside the `qualmap` block**
  (`:464-468`) -- an empty `qual_name` produces a file with no
  terminator, which the parser accepts (ported, recorded).
  **Writer guard** (D4's quoted-list quirk): `to_string`/`write`
  raise `ValueError` when any element after the first of a quoted
  string list -- in practice `master_files[1:]` -- contains
  whitespace, because the C++ parser would silently strip it on
  read-back (measured).
  **Byte-parity pin**: `EMSphInxNamelist.defaults().to_string()`
  equals the captured `IndexEBSD.exe -t` template (119 lines,
  Windows binary writes CRLF -- comparison is line-wise; CRLF md5
  of the capture `49ddf0e7d9b2d758d918c20a7f900a6d`) line for
  line. The `.6g` formatting means a full-precision `pctr`
  round-trips with ~1e-6 fractional loss (~3e-4 px on a 60 px
  detector -- negligible against the 0.33 deg mean-PC floor;
  measured: 8-decimal `pctr` values in a hand-written namelist are
  parsed fully, so *reading* loses nothing). No full-precision
  writer mode in v1 (open question 6).
- Round-trip contract: `from_string(x.to_string())` equals `x` on
  every field for values representable at `.6g` (defaults, the
  acid-test namelist, **and a non-empty-`ipath` variant** -- the
  raw-storage decision above makes this exact where the C++ would
  double-prefix); the generic parser round-trip is the `nml.cpp`
  suite (D4).

### D6 -- Vendor conversions, `to_kwargs`/`from_kwargs`/`to_detector` (frozen)

- Internal geometry triple (EMSphInx `Geometry`): `cX`, `cY` in
  pixels relative to the detector centre, `sDst` in microns.
  Ported conversions (with `w = patDims[0]`, `h = patDims[1]`,
  `delta` the pixel size):

  | vendor | cX | cY | sDst | source |
  |---|---|---|---|---|
  | EMsoft | `p0` | `p1` | `p2` | `detector.hpp:85` |
  | EDAX / tsl | `p0*w - w/2` | `p1*w - h/2` | `p2*w*delta` | `:249-254` |
  | Oxford | `(p0-0.5)*w` | `(p1-0.5)*h` | `p2*w*delta` | `:261-267` |
  | Bruker | `(p0-0.5)*w` | `(0.5-p1)*h` | `p2*h*delta` | `:274-279` |

  and kikuchipy (Bruker) pc from the triple:
  `PCx = cX/w + 0.5`, `PCy = 0.5 - cY/h`, `PCz = sDst/(h*delta)`
  -- i.e. **Bruker is the identity on kikuchipy's `pc`**
  (tech-stack: kikuchipy `pc` enters directly).
  (corrected 2026-09-02: that identity is **algebraic, not
  bitwise**. Measured -- composing the two formulas returns
  `0.2134` as `0.21340000000000003` on all three test shapes and
  `0.5007` as `0.5006999999999999` on `(60, 48)`, and the nickel
  `pc_average` in the last ulp of its y component. Since
  `test_bruker_is_the_identity_on_the_kikuchipy_pc` and
  `test_from_kwargs_bruker_pctr_is_pc_average_verbatim` assert
  `np.array_equal`, and D6 says `pctr` is `pc_average`
  **verbatim**, `_pctr_to_pc`/`_pc_to_pctr` **short circuit
  `"Bruker"`** and return the input unchanged rather than composing
  `_pctr_to_geometry` with the projection; see the appended
  "Recorded results" of `validation.md`.) **`delta` cancels
  exactly for the three fractional vendors** (TSL/Oxford/Bruker
  fold it into `sDst` and the projection divides it back out):
  measured through the binary, Bruker namelists with `delta` 250
  and 500 produce bitwise-identical `.ang` Euler angles (`sDst`
  halves, geometry unchanged), and `to_detector().pc` is
  delta-invariant for those vendors (named test). Only the
  **EMsoft** route (pixel/micron-unit `pctr`) and `sanityCheck`'s
  `[5, 90]` mm window see `delta` -- Phase 10's planned two-`delta`
  sweep must therefore use the EMsoft vendor route (or is a
  sanity-window probe only); recorded in the plan 0.3 amendment.
- **Measured equivalence through the binary** (the strongest pin):
  on the acid geometry, namelists with vendor
  `EMsoft (-4.49166896, 17.19798032, 15021.20746804)`,
  `EDAX (0.42513885, 0.78663301, 0.50070692)` and
  `Oxford (0.42513885, 0.78663301, 0.50070692)` produce
  **bitwise-identical `.ang` Euler angles** to the Bruker run
  (max |diff| 0.0). Stdout confirms the internal triple:
  `Pattern Center: -4.49167, 17.198 fractional pixels`,
  `Scintillator Distance: 15021.2 microns`. (Precision note: the
  reference triple derives from the **unrounded**
  `detector.pc_average`; recomputing from the 8-decimal rounded pc
  gives `cY 17.1979806` / `sDst 15021.2076` -- tests quote input
  and output from the same unrounded source and assert with
  `pytest.approx`.)
- **kikuchipy `pc_*` helpers are NOT used** for TSL/Oxford:
  measured on a rectangular `(48, 60)` detector, EMSphInx-TSL
  `(0.4251, 0.62928, 0.40056)` equals kikuchipy `pc_oxford()`, and
  EMSphInx-Oxford y equals kikuchipy `pc_tsl()` y while z differs
  (`w` vs `h` scaling) -- the two code bases disagree about which
  formula belongs to which vendor name on non-square detectors.
  The port follows **EMSphInx's own formulas** (the goal is what
  `IndexEBSD.exe` will do with the numbers); the deviation table
  is frozen in a named test, exercised on **both** rectangular
  orientations -- `(48, 60)` and `(60, 48)` -- because kikuchipy's
  `_pc_bruker2tsl` divides z by `min(nrows, ncols)/nrows`
  (`_ebsd_detector.py:2326-2330`), a no-op when `nrows < ncols`,
  so a single orientation hides that branch. (corrected
  2026-09-02: that branch makes the two **agree** on `(60, 48)` --
  `48/60` is exactly EMSphInx's `h/w`, measured -- so the
  Oxford-vs-`pc_tsl` *deviation* is a `(48, 60)` row only; the
  named test carries the measured relation per shape. Recorded
  results, third dated section.) **EMsoft equals
  kikuchipy `pc_emsoft(version=4)` exactly only when
  `binning == 1` and `px_size == delta`** (kikuchipy multiplies by
  `self._binning` and `self.px_size`, `_ebsd_detector.py:
  2317-2324`; EMSphInx uses the already-binned `patDims` and the
  namelist `delta` -- measured: the nickel detector's
  `binning=8, px_size=1.0` gives `pc_emsoft(4)` values ~8x off);
  the equality test constructs its detector with `binning=1,
  px_size=delta, pc=pc_average` and then holds exactly,
  rectangular included; kikuchipy's default v5 has the opposite
  `cX` sign.
- `to_detector(*, sample_tilt: float) -> EBSDDetector`:
  `shape=(patDims[1], patDims[0])`, `pc = [PCx, PCy, PCz]` (from
  the table), `px_size=delta`, `tilt=thetac`,
  `sample_tilt=sample_tilt`. **`sample_tilt` is a required
  keyword**: the namelist has no sample tilt -- `IndexEBSD` takes
  it from the master's `sig` (`idx.hpp:218`), and Phase 6's binding
  guard showed a silent default indexes ~5 deg wrong at *higher*
  scores; the docstring points at the public
  `MasterPatternHarmonics.sample_tilt` (which `from_file` maps
  from the `.sht` `primary_angle`,
  `_master_pattern_harmonics.py:1805`; 70.0 in the in-package Ni
  files -- measured).
- `to_kwargs() -> dict` for
  `EBSD.spherical_indexing`/`SphericalIndexer` (keys are exactly
  their parameter names): `bandwidth=bw`, `normalize=normed`,
  `refine=refine`, `n_regions=nRegions`,
  `gaussian_background=gausBckg`,
  **`circular_mask = (circ_rad == 0)`** -- kikuchipy's
  `circular_mask` mirrors EMSphInx's `Geometry::circ` flag, which
  `idx.hpp:230` sets as `maskPattern(circRad == 0)`. A *positive*
  radius leaves `Geometry::circ` false but **still masks in the
  image processor**: `PatternProcessor::setSize`
  (`imprc.hpp:108-122`) builds a `CircMask(w, h, r)` and sets
  `msk = true` for any `r >= 0` (`idx.hpp:254` passes `circRad`
  straight through). kikuchipy has no fixed-radius processor mask,
  so `circ_rad > 0` maps to `False` **and loses that processor
  mask** -- `to_kwargs` warns about exactly that (the mask is
  applied at radius `r` in EMSphInx, not silently dropped there;
  earlier drafts had this wrong). `chunksize = batch_size or None`
  (0 -> the `BatchEstimate` default, matching Phase 6). `n_thread`
  has no kikuchipy equivalent (dask owns the workers) --
  documented, not returned. A non-empty `roi_mask` raises
  `ValueError` (out-of-scope grammar) (corrected 2026-09-02: except
  `"0"`, which is the program's own spelling of *no* region of
  interest -- `RoiSelection::from_string` returns an empty selection
  for it, `idx/roi.h:592`, and the template's own comment advertises
  "0 (or omitted) to index the entire scan". Measured through
  `IndexEBSD.exe`: `roimask = '0'` is accepted exactly as `''` is,
  while `'1'` gives "odd number of points in ROI string". `"0"`
  therefore maps to the whole scan and does not raise; it is still
  written back as `'0'`, since the string is stored opaquely, where
  the C++ writer normalises it to `''`); `emsphinx_compatible` is
  not emitted (both defaults are `True`, the parity configuration).
- `from_kwargs(*, pattern_file, pat_dset="patterns", master_files,
  detector, scan_shape, scan_steps, data_file, vendor="Bruker",
  delta=None, n_thread=0, batch_size=0, vendor_file="",
  ipf_name="", qual_name="", **indexing_kwargs) ->
  EMSphInxNamelist` (classmethod): builds a complete namelist for
  Phase 10.
  - `vendor="Bruker"` default -- `pctr` is then kikuchipy's
    `detector.pc_average` verbatim (identity, no precision games);
    any of the four vendors selectable (conversion table above,
    inverted).
  - **`delta=None` resolves to `30000 / pat_dims[0]`** (detector
    width exactly 30 mm -- always inside `sanityCheck`'s `[5, 90]`
    mm window; `detector.px_size` is NOT used blindly: kikuchipy
    fixtures carry `px_size=1.0`, which the C++ rejects as a
    0.06 mm detector -- measured constraint; geometry-neutral for
    the fractional vendors by the cancellation above, choice
    documented).
  - `thetac = detector.tilt`; a non-zero `azimuthal`/`twist`
    raises as everywhere in this project.
  - `pat_dims = (ncols, nrows) = detector.shape[::-1]`, and
    likewise **`scan_dims = (n_scan_cols, n_scan_rows) =
    scan_shape[::-1]`** and `scan_steps = (step_x, step_y)`: the
    namelist is x-then-y (`ebsd/nml.hpp:397` writes
    `scandims = x, y, sx, sy`; `idx.hpp:238` builds
    `OrientationMap(scanDims[0], scanDims[1])`), the opposite of
    kikuchipy's `(nrows, ncols)` navigation shape -- both orders
    pinned by rectangular named tests (a 3x3 fixture cannot catch
    a transposition).
  - `n_thread`/`batch_size` pass straight through (default 0 =
    auto in both worlds) so the acid test and Phase 10 can emit
    the deterministic `nthread=1 batchsize=1` configuration;
    `circular_mask` (from `indexing_kwargs`) maps inversely to
    `circ_rad`: `True -> 0`, `False -> -1` (the forward map's
    exact inverse on its image).
  - The three output-name keywords default to **empty** (`""` = no
    vendor/`ipf`/`qual` output; `defaults()`' `reindexed.ang`/
    `ipf.png`/`qual.png` would make every Phase 10 run write PNG
    maps) -- callers wanting the `.ang` pass
    `vendor_file="out.ang"` explicitly; an empty `qual_name` also
    exercises the terminator-less `to_string` file (D5).
  - The result passes `sanity_check()` by construction (named
    test).

### D7 -- The acid test: measured end-to-end acceptance (frozen gates)

Recipe (validation "Recorded results" has the scripts): repack
`nickel_ebsd_small` with backgrounds removed
(`remove_static_background()` + `remove_dynamic_background()`,
uint8) on the **canonical route** (writer default:
`manufacturer="EMsoft"`, rows unflipped), write the matching
namelist via `from_kwargs` (`bw` 68, `normed`/`refine` `.TRUE.`,
`nregions` 10, `gausbckg` `.FALSE.`, `circmask` -1, `nthread` 1,
`batchsize` 1, `vendor` Bruker + `pc_average`, `delta` 500 = the
30 mm rule, `thetac` 0, `scandims` 3,3,1.5,
`vendor_file="out.ang"`), master =
`src/kikuchipy/data/emsphinx/ni_small_20kv_bw384.sht`, run
`IndexEBSD.exe <nml>` with `cwd=tmp_path`.

- **It indexes** (exit 0). Both correct routes were measured and
  are recorded separately; every pinned band below is anchored on
  the **default route** the test actually runs:
  - default route (`Manufacturer` EMsoft, unflipped; stdout
    `Vertical Flip: true`): refined-vs-stored-xmap per-point
    `[0.648, 0.948, 0.799, 0.705, 0.725, 0.689, 0.875, 0.616,
    0.902]` deg -- **median 0.7245 / max 0.9479**; `.ang` scores
    (col 7) `[0.662, 0.580, 0.645, 0.667, 0.575, 0.654, 0.633,
    0.614, 0.625]`, mean **0.6283**; `.ang` Eulers bitwise
    identical to the EDAX-`Manufacturer` twin (same vendorFlip
    read path).
  - Bruker route (rows flipped, `Manufacturer` Bruker; stdout
    `Vertical Flip: false`): per-point `[0.652, 0.947, 0.795,
    0.709, 0.713, 0.684, 0.875, 0.610, 0.902]` deg -- median
    0.713 / max 0.947; coarse (refine=.FALSE.) median 0.783 / max
    1.007; scores mean 0.6304. Equivalent but not bitwise
    (~0.01-0.02 deg per point vs the default route).
  - Runtimes (this machine, single thread, recorded only):
    112-120 pat/s refined, ~136 pat/s coarse across the recorded
    runs; whole process ~0.2-0.6 s.
- Context (recorded, NOT a Phase 9 gate -- Phase 10 owns parity):
  kikuchipy `spherical_indexing` on the identical inputs (same
  `.sht` via `from_file`, `bw` 68, defaults,
  `detector.pc = detector.pc_average`) vs the `IndexEBSD.exe`
  refined output on the **Bruker route**: **median 0.341 / max
  0.363 deg**, score Pearson r **0.9607**; kikuchipy vs stored
  xmap median 0.533. Phase 10 re-baselines these on the canonical
  route with its own harness.
- **Frozen local-gated assertions** (Phase 6 margin convention,
  ~1.7x on the measured values): the end-to-end test writes the
  three files into `tmp_path`, runs `IndexEBSD.exe` with
  `cwd=tmp_path` (exit 0 required), and asserts
  refined-vs-stored-xmap **median < 1.2** (measured 0.7245, 1.66x),
  **max < 1.6** deg (measured 0.9479, 1.69x) and scores mean
  `pytest.approx(0.628, rel=0.05)` (measured 0.6283); the run
  configuration is `nthread=1 batchsize=1` (the Phase 10
  convention). Runtime recorded via `record_property`, never
  asserted.
- The negative controls are tests too (each was measured): the
  wrong flip pairing gives median ~39.6 deg at scores ~0.22 (not
  asserted at those values -- asserted as `median > 10 deg`, a
  discrimination check); a `Manufacturer`-less file exits non-zero
  with `doesn't have a Manufacturer string`; an unknown
  `Manufacturer` exits non-zero with `unknown EBSD vendor`.
- Output-file facts recorded for Phase 10 (not asserted here): the
  `.ang` is TSL-convention (header `y-star 0.786633` = the
  converted value), 8 columns `eu1 eu2 eu3 x y iq ci phase`, read
  in tests with `np.loadtxt(path, comments="#")` (the shared
  `read_ang` helper, D9 -- `orix.io.load` works but warns and
  names the columns `unknown1/2`); `datafile` HDF5 layout
  `Manufacturer="EMSphInx"`, `Version`, `NMLfiles/IndexEBSD` (raw
  lines), `NMLparameters/IndexEBSD/*` (typed: int32 / float64 /
  uint32-bools / vlen str), `Scan 1/EBSD/Data/{IQ, Metric, Phase,
  Phi, Phi1, Phi2}` float32, `Scan 1/{IPF Map, IQ Map, XC Map}`
  uint8.

### D8 -- Placement, licences, exports, CHANGELOG (frozen)

- Modules: `indexing/_spherical/_namelist.py` (D4+D5+D6) and
  `indexing/_spherical/_pattern_repack.py` (D1); probe in
  `io/plugins/oxford_binary/_api.py`. Public names via
  `indexing/__init__.pyi` (sorted `__all__` gains
  `EMSphInxNamelist`, `write_emsphinx_patterns`) and the plugin's
  `__init__.pyi` (`__all__ = ["file_reader", "get_scan_info"]`).
  No signal-method surface (nothing pattern-shaped to hang on
  `EBSD` -- these are file utilities). Existing files the change
  must touch are enumerated in plan 4.1.
- Licence provenance per file (mission legal terms; BSD opt-out
  impossible for the first two -- stated in the PR):
  - `_pattern_repack.py`: kikuchipy GPL header + delimited EMSphInx
    notice (`programs/pattern_repack.cpp` -- `binFloat` `:50-67`,
    `binAvg` `:76-98`, `flipPat` `:105-113`, HDF5 layout
    `:183-208`; plus `pattern.hpp` `GetVendor` `:608-637` and the
    vendor-flip table `:463-471` as the `Manufacturer` contract),
    with the modification notice and the D2 deviation ("writes the
    Manufacturer dataset the program omits") named in the block.
  - `_namelist.py`: kikuchipy GPL header + delimited EMSphInx
    notice (`util/nml.hpp` `:226-520`; `modality/ebsd/nml.hpp`
    `:52-160`, `:186-218`, `:236-315`, `:320-470`, `:621-639`;
    `modality/ebsd/detector.hpp` `:85`, `:249-279`;
    `idx.hpp:218-231`, `:254` and `imprc.hpp:108-122` for the
    kwargs/mask semantics), modification notice, not-ported list
    (scan-file reading, ROI grammar, H5 dumping) and the recorded
    deviations (raw path storage, always-optional
    `patdset`/`scanname`).
  - `oxford_binary/_api.py`: **kikuchipy header only** -- the probe
    reimplements `ebsp_dims.cpp`'s *output contract* on kikuchipy's
    pre-existing reader; no EMSphInx code is transcribed (the
    distinct-set/product idea is not expression). The docstring
    cites the program (`EMSphInx programs/ebsp_dims.cpp`) as the
    modelled-on reference, like `related_projects.rst` does.
    Flagged autonomous decision (plan 6.3).
- CHANGELOG (`Unreleased -> Added`, **three entries, one per
  public name** -- the tech-stack "one per feature" rule; each with
  the PR **#10** fork link
  `` (`#10 <https://github.com/jwestraadt/kikuchipy/pull/10>`_) ``):
  1. "``kikuchipy.indexing.write_emsphinx_patterns()``: write EBSD
     patterns to the repacked HDF5 layout EMSphInx's ``IndexEBSD``
     reads (``PatternRepack`` equivalent plus the required
     ``Manufacturer`` dataset)."
  2. "``kikuchipy.indexing.EMSphInxNamelist``: read and write
     ``IndexEBSD`` namelist files and convert them to and from
     spherical-indexing arguments and an ``EBSDDetector``."
  3. "``kikuchipy.io.plugins.oxford_binary.get_scan_info()``:
     probe the scan grid and layout of an ``.ebsp`` file
     (``EBSPDims`` equivalent)."
- Docs: numpydoc for the three public names (generated reference
  from `__all__`); no phase numbers in public docstrings; the
  writer/namelist docstrings carry the EMSphInx-quirk notes of
  D1/D2/D6 (`:cite:lenthe2019spherical` where orientations are
  discussed). No tutorial changes (Phase 11).

### D9 -- CI gating, determinism, dependencies (frozen)

- The EMSphInx binaries exist only on this machine: every test that
  runs an `.exe` is gated on **`KIKUCHIPY_EMSPHINX_DIR`** with
  `pytest.skip`. The gate helpers are promoted to shared
  `conftest.py` fixtures this phase (one copy for Phases 9 and 10,
  and the third consumer was coming): `emsphinx_program` (factory
  fixture wrapping the `_emsphinx_dir`/`_emsphinx_program` logic of
  `test_spherical_sht_file.py:160-190`, `build/Release/<name>.exe`
  or bare `<name>` for non-Windows) and `read_ang` (callable
  fixture, `np.loadtxt(path, comments="#")`);
  `test_spherical_sht_file.py`'s private copies are refactored onto
  them (plan 4.1). **Every binary invocation passes
  `cwd=tmp_path`** to `subprocess.run`: `IndexEBSD.exe -t` writes
  to the hard-coded relative `"IndexEBSD.nml"`
  (`index_ebsd.cpp:67`) and all namelist paths resolve against the
  process cwd, so a bare invocation races under `-n 4` and litters
  the checkout (Phase 0 already had to clean one up).
- **Everything else asserts on file bytes and h5py properties** so
  CI covers the full contract: dataset layout/alloc/offset/dtype/
  shape pins (incl. **zero filters** -- the functional guard, D2 --
  and **native byte order** on a byte-swapped input, D1),
  dataset-bytes equality against `signal.data`, the frozen
  `binAvg`/`binFloat` NumPy oracles (with half-value fixtures so
  rounding mode is discriminated), the flip table, the
  `Manufacturer` vlen-**ASCII** pin (dtype *and* cset -- UTF-8 is
  the h5py default and is fatal, D1), the overwrite protocol, the
  `tracemalloc` lazy-write bound (Phase 6 pattern; band measured
  on the tests-first run, then pinned), the `nml.cpp` suite, the
  119-line template parity fixture (stored as a string constant in
  the test module with its provenance comment), the conversion
  table incl. the rectangular-detector deviation rows on both
  orientations and the preconditioned `pc_emsoft(version=4)`
  equality, the delta-invariance test, and the probe on the
  in-package `patterns.ebsp` plus synthetic fixtures (regular
  variants from the `oxford_binary_file` conftest fixture as-is;
  the staggered and near-duplicate files from a module-local
  helper modelled on
  `src/kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py`
  -- the conftest fixture derives `beam_x/y` from navigation
  indices and cannot express them, plan 3.2).
- No numba kernels, no `scipy.fft`, no new dependencies (h5py and
  the low-level `h5py.h5p/h5d/h5s/h5t` API are part of the h5py
  runtime dependency). Everything is deterministic; no seeds
  needed beyond the fixtures'.
- The oldest-supported job needs no version gates: h5py low-level
  alloc-time API and `string_dtype(encoding="ascii")` exist across
  the supported h5py range (an `importorskip` is still added if
  the tests-first run finds otherwise -- plan 5.2 records the
  check).

## Context

- Constitution: `specs/mission.md` (deliverables table rows
  `PatternRepack`/`EBSPDims`, out-of-scope list confirmed above),
  `specs/tech-stack.md` (PatternRepack contract bullet -- amended
  per plan 0; Bruker == kikuchipy `pc`; delta cancellation),
  `specs/roadmap.md` Phase 9 (slimmed; box rewrite in plan 0.1).
- Research: `specs/_research/explore-emsphinx-programs-and-formats.md`
  sections 1.5-1.9 (IndexEBSD `-t`, PatternRepack, EBSPDims, the
  namelist table), 2 (pattern formats), 7 (output HDF5); addenda in
  plan 0.4.
- C++ read for this spec: `programs/pattern_repack.cpp` (all),
  `programs/ebsp_dims.cpp` (all), `include/util/nml.hpp` (all),
  `include/modality/ebsd/nml.hpp` (all),
  `include/modality/ebsd/pattern.hpp:400-554, 590-637, 754-913`,
  `include/modality/ebsd/detector.hpp:52-166, 244-326`,
  `include/modality/ebsd/idx.hpp:218-260`,
  `include/modality/ebsd/imprc.hpp:100-140`,
  `test/util/nml.cpp` (all).
- Prior phases composed: the Phase 2 `.sht` fixtures
  (`ni_small_20kv_bw384.sht`, `sample_tilt` 70.0 via
  `MasterPatternHarmonics.from_file` -- measured) and
  `MasterPatternHarmonics.from_file`; the Phase 6/7 indexing
  defaults the kwargs map onto (signatures verified against the
  live `SphericalIndexer.__init__` and `EBSD.spherical_indexing`);
  the in-package `patterns.ebsp` (== `nickel_ebsd_small` -- the
  writer's and probe's real-data fixture); the
  `KIKUCHIPY_EMSPHINX_DIR` gate of Phase 2's tests.
- Downstream driver: Phase 10 feeds `IndexEBSD.exe` with a
  kikuchipy-written repacked file + namelist on
  `nickel_ebsd_small`/`large` (`nthread=1 batchsize=1`, the
  canonical default route) -- the D7 acid test is that flow, run
  today; D2's uint8-only finding, D1's canonical-route decision
  and D6's delta-cancellation note are Phase 10 constraints
  recorded now (plan 0.3).
- CI lessons applied: no whole-file md5 pins across h5py versions
  (dataset bytes only); no float knife edges (the rounding fixture
  uses exact .5 sums); line-wise template comparison (CRLF is the
  Windows binary's, not ours); measured-then-pinned bands with the
  Phase 6 margin convention; binaries local-gated with skips and
  `cwd=tmp_path`.
