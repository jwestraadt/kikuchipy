# EMSphInx supporting programs & file formats — exhaustive reference for a pure-Python (kikuchipy) port

Repo: `c:/Users/westraadt.1/Repos/EMSphInx` @ branch `master`, HEAD `60f3517` ("Add Visual Studio 2022 CMake build instructions").
Built binaries: `c:/Users/westraadt.1/Repos/EMSphInx/build/Release/{IndexEBSD,MasterXcorr,mp2sht,sht2png,ShtWisdom,PatternRepack,EBSPDims,EMSphInxEBSD}.exe`.

**Caveat found while exploring:** `build/Release/emsphinx_cuda.lib` exists and the `benchmarks/*.nml` contain `backend`/`gpudevice` keys plus a `CUDA Stage Timing` block in the logs — those artefacts were produced by a `feature/GPU` branch build (logs say `Git Branch : feature/GPU`, `Commit Hash : 98c251a`). The **checked-out `master` source has no CUDA and no `backend`/`gpudevice` namelist keys** (`grep -rn "backend|gpudevice|CUDA"` over non-build sources returns nothing). Treat the benchmark nml/logs/h5 as *reference outputs from a superset build*; the extra keys would be reported as "unused namelist parameters" by master's `IndexEBSD`.

---

## 1. Program inventory

`programs/CMakeLists.txt:39-47` builds 7 CLI tools + 1 GUI. Only 5 are `install`ed (`:47`): `IndexEBSD MasterXcorr mp2sht sht2png ShtWisdom` (component `clt`). `PatternRepack` and `EBSPDims` are built but not installed (dev utilities). The GUI is gated behind `EMSPHINX_BUILD_GUIS` (`:49-63`).

| Target | Source | Kind | Links |
|---|---|---|---|
| `IndexEBSD` | `programs/index_ebsd.cpp` | **CLI** | FFTW + HDF5 (`:98`) |
| `MasterXcorr` | `programs/master_xcorr.cpp` | **CLI** | FFTW + HDF5 (`:99`) |
| `PatternRepack` | `programs/pattern_repack.cpp` | **CLI** | HDF5 only (`:100`) |
| `EBSPDims` | `programs/ebsp_dims.cpp` | **CLI** | HDF5 only (`:101`) |
| `mp2sht` | `programs/mp2sht.cpp` | **CLI** | FFTW + HDF5 (`:102`) |
| `sht2png` | `programs/sht2png.cpp` | **CLI** | FFTW + HDF5 (`:103`) |
| `ShtWisdom` | `programs/sht_wisdom.cpp` | **CLI** | FFTW only (`:104`) |
| `EMSphInxEBSD` | `programs/ebsd_wizard.cpp` | **GUI (wxWidgets) — SKIP for Python port** | + `core base propgrid html` (`:107`) |

### 1.1 `IndexEBSD` — main EBSD indexer (CLI)

- **Source**: `programs/index_ebsd.cpp` (198 lines).
- **CLI** (`:59-74`): exactly 2 argv.
  - `IndexEBSD input.nml` → index.
  - `IndexEBSD -t` → writes a template namelist to `./IndexEBSD.nml` via `emsphinx::ebsd::Namelist::defaults()` + `to_string()`.
- **Headers used**: `util/threadpool.hpp`, `util/timer.hpp`, `util/nml.hpp`, `util/sysnames.hpp`, `modality/ebsd/pattern.hpp`, `modality/ebsd/nml.hpp`, `modality/ebsd/idx.hpp` (`:38-46`). `Real = double` (`:52`).
- **Flow**:
  1. Parse namelist (`nml.from_string`), warn about unused keys (`:81-84`).
  2. `emsphinx::ebsd::IndexingData<Real> idxData(nml)` (`:91`) — this constructor does *all* the setup (§3.4, `include/modality/ebsd/idx.hpp:175-311`), and importantly **creates + writes the output HDF5 header up front** (`idx.hpp:177` → `nml.writeFileHeader()`) so a permission failure surfaces before hours of compute.
  3. Print banner: git branch/hash/version, geometry, pattern source, master patterns (point group, zRot, equatorial mirror), bandwidth, `fft::fastSize(2*bw-1)` side length, ROI string, thread count, batch size (`:98-141`).
  4. `ThreadPool pool(threadCount)`, schedule `numPat/batchSize` (+1) copies of `idxData.workItem` (`:148-159`). NB the loop computes `start`/`end` but does **not** pass them — each work item pulls the next batch from the (mutex-protected) `PatternFile::extract` itself.
  5. Progress loop with 1 s updates using `idxData.idxCtr` / `idxData.numIdx` (`:162-177`).
  6. `idxData.save(tmStart, tmEnd, total)` (`:187`).
- **Namelist parameters**: see §1.9 table.
- **Outputs**: HDF5 datafile, optional vendor `.ang`/`.ctf`, optional IPF png, optional quality png (§4).

### 1.2 `MasterXcorr` — pseudo-symmetry prediction (CLI) → §5

- `programs/master_xcorr.cpp`. Headers: `util/timer.hpp`, `idx/master.hpp`, `sht/sht_xcorr.hpp`, `xtal/diagram.hpp`, `constants.hpp` (`:35-39`).
- **Argument parsing bug worth knowing**: the usage string (`:44`) claims `scanFile bandWidth cutoff masterFile1 [masterFile2]` but the code reads `argv[1]=bandWidth`, `argv[2]=cutoff`, `argv[3]=masterFile1`, `argv[4]=masterFile2` (`:53-57`), with `argc` required to be 4 or 5. So the real usage is `MasterXcorr <bw> <cutoff> <master1.h5> [master2.h5]`.
- Inputs are **EMsoft `*.h5` master patterns** (read via `emsphinx::MasterPattern<double>(file)`), not `.sht`.
- Hard-coded: `outputFile = "pseudo_sym.h5"` (`:60`), `factor = 0.95` (`:61`), bandwidth clamp `[53, 313]` (`:64-65`), duplicate-maxima merge tolerance 2° (`:160`), symmetry-operator rejection cosine 0.999 (`:237`), n-fold classification cutoff 0.05 (`:238`).

### 1.3 `mp2sht` — EMsoft master pattern HDF5 → `.sht` (CLI) → §2.3

- `programs/mp2sht.cpp`. `mp2sht inputFile outputFile` (`:44-49`). Usage text says `*.spx` for the output but the format written is the v1.1 `.sht` (legacy naming).
- Headers: `idx/master.hpp`, `sht_file.hpp` (`:37-38`). `sht_file.hpp` is **not in the repo** — it is fetched at configure time by `CMakeLists.txt:151-162` (`FetchContent` from `https://github.com/EMsoft-org/SHTfile`) and configured into `build/_deps/shtfile-build/sht_file.hpp` from `build/_deps/shtfile-src/sht_file.in.hpp`.
- Hard-coded: `bw = 384`, `nrm = true` (`:53-54`).

### 1.4 `sht2png` — `.sht` → PNG previews + header dump (CLI) → §6

- `programs/sht2png.cpp`. `sht2png inputFile sqLegOut [sterOut]` (`:72-78`).
- Headers: `idx/master.hpp`, `sht_file.hpp`, `util/image.hpp`, and `miniz/miniz.c` compiled inline with `MINIZ_NO_STDIO/NO_TIME/NO_ZLIB_APIS` (`:39-46`).

### 1.5 `ShtWisdom` — FFTW plan pre-computation (CLI) → §7

- `programs/sht_wisdom.cpp`. `ShtWisdom bandWidth` (`:42-47`). Only header: `util/fft.hpp` (`:37`).

### 1.6 `PatternRepack` — raw pattern file → HDF5 (CLI)

- `programs/pattern_repack.cpp`. `PatternRepack inputFile outputFile [binning]` (`:120-126`); input `*.up1 | *.up2 | *.data | *.ebsp`, output `*.hdf`.
- Compile-time switches at top of `main`: `binToFloat = false` (`:116`), `flip = true` (`:117`).
- Algorithm: `PatternFile::Read(inputFile)` (`:147`); print `numPat/width/height/type/imBytes/total MB` (`:154-172`); check `binning` divides pattern dims (`:175`); create HDF5 dataset **`/patterns`**, dtype from pixel type (uint8/uint16/float, `:184-190`), shape `(numPat, height/bin, width/bin)` (`:195`), with `props.setAllocTime(H5D_ALLOC_TIME_EARLY)` so `getOffset()` works later for memory-mapped reads (`:200-202`); then loop, `flipPat` each pattern vertically (`:105-113`, `:215`), bin with `binAvg` (in-type average, `:76-98`) or `binFloat` (sum→float, `:50-67`), and write via hyperslab (`:216-217`).
- Python equivalent: trivial with h5py + numpy; the only subtlety is the **vertical flip** and that the HDF5 dataset is created contiguous/early-allocated.

**Addendum (2026-09-02, `specs/2026-09-02-sht-interop/requirements.md` D1, D2)**: PatternRepack writes **no `Manufacturer`**, its output needs one added before `IndexEBSD` accepts it (measured error text); `binToFloat`/`flip` are hard-coded consts (`:116-117`), so `binFloat` is dead code in the binary; binAvg rounding is `std::round` = half away from zero (bitwise probe).

### 1.7 `EBSPDims` — Oxford `.ebsp` scan-dimension prober (CLI)

- `programs/ebsp_dims.cpp`. `EBSPDims inputFile` (`:45-49`). Constructs `emsphinx::ebsd::OxfordPatternFile` directly (`:53`), prints pattern count/width/height/type/bytes/total size, then streams all patterns in batches of 100 collecting the per-pattern `x`/`y` stage coordinates into `std::set`s (`:76-92`), and reports `sx.size()` × `sy.size()`; if that product ≠ `numPat` it prints the full sorted coordinate lists (`:96-103`). Purpose: recover the scan grid (`scandims`) for an `.ebsp` with no companion `.ctf`.

### 1.8 `EMSphInxEBSD` — GUI (**skip**)

- `programs/ebsd_wizard.cpp` is a 43-line wxWidgets `wxApp` that instantiates `IndexingFrame` (`include/wx/IndexingFrame.h`) and sets the sphinx icon. All GUI logic lives in `include/wx/*.h` (20 files: `EbsdNamelistWizard.h`, `IdxParamPanel.h`, `MasterPatternSelectPanel.h`, `MasterFileList.hpp`, `MPConvertDlg.h`, `PatternCenterPanel.h`, `PatternLoadPanel.h`, `PatternPreviewPanel.h`, `RoiSelectionPanel.h`, `ScanDimsPanel.h`, `WisdomPrompt.h`, `BibtexDialog.h`, `PeriodicTablePanel.h`, `ValidityWizard.h`, …). Everything the GUI does maps 1:1 onto the CLI namelist (documented in `documentation/emsphinxebsd.rst`), so a notebook can replace it entirely.

### 1.9 Complete `IndexEBSD` namelist (from `include/modality/ebsd/nml.hpp`)

Struct fields `:54-88`; parser `Namelist::parse_nml` `:236-315`; writer `to_string` `:320-470`; defaults `:186-218`; validation `sanityCheck` `:621-639`.

| Key | Type | Required? | Meaning / parse notes | Default (`defaults()`) |
|---|---|---|---|---|
| `ipath` | string | optional (`:238`) | prefix prepended to `patfile` and every `masterfile` | `""` |
| `patfile` | string | **yes** | pattern file; `ipath + patfile` (`:241`) | `scan.h5` |
| `patdset` | string | only if `patfile` is HDF5 (`:243`) | h5 path to the 3D pattern dataset | `Scan 1/EBSD/Data/Pattern` |
| `masterfile` | string list (`getStrings`, `:240`) | **yes** | one or more `.sht` files; index in list = phase index | `{master.h5}` |
| `psymfile` | string | optional (`:239`) | EMsoft quaternion angle file of pseudo-symmetry operators; **only valid for single-phase** (`idx.hpp:191`) | `""` |
| `patdims` | 2 ints (`:273-276`) | **yes** | binned detector `w, h`; validated `[2,16384]` (`:627`) | `640, 480` |
| `circmask` | int (`:277`) | **yes** | `-1` none, `0` largest inscribed circle, `>0` radius px; `>= -1` enforced (`:629`) | `-1` |
| `gausbckg` | bool (`:278`) | **yes** | 2D Gaussian background subtraction | `.FALSE.` |
| `nregions` | int (`:279`) | **yes** | AHE tile count, `0` = off; `<= min(patdims)` (`:630`) | `10` |
| `delta` | double (`:282`) | **yes** | binned pixel size µm; detector width `delta*patdims[0]/1000` must be in `[5,90]` mm (`:631-632`) | `50.0` |
| `thetac` | double (`:283`) | **yes** | camera elevation, `[-60,60]` deg (`:633`) | `10.0` |
| `vendor` | string (`:284`) | **yes** | one of `EMsoft`, `EDAX`, `tsl`, `Oxford`, `Bruker` (`:290-295`; note `TSL` uppercase is **rejected** — only lowercase `tsl` matches) | `EMsoft` |
| `pctr` | 3 doubles (`:286-288`) | **yes** | interpretation per vendor (§3.5) | `0,0,15000` |
| `scandims` | string **or** 3–4 doubles (`:251-268`) | **yes** | if it parses as a filename → read `.ang`/`.ctf`/`.h5` scan file for dims + pattern centre + tilt (`readScanFile`, `:550-578`), and **pctr/thetac are then set to NAN** (`:256`); else `w,h,step` or `w,h,sx,sy` | `256,256,1,1` |
| `scanname` | string | only when `scandims` is an h5 file (`:254`) | h5 scan group | `""` |
| `roimask` | string (`:269`) | **yes** (may be `''` or `'0'`) | ROI string, see §3.5 | `''` |
| `bw` | int (`:298`) | **yes** | bandwidth; `[16,512]` (`:635`) | `68` |
| `normed` | bool (`:299`) | **yes** | normalized vs unnormalized spherical XC | `.TRUE.` |
| `refine` | bool (`:300`) | **yes** | Newton refinement vs 3×3×3 subpixel interpolation | `.TRUE.` |
| `nthread` | int (`:301`) | **yes** | `0` = `ThreadPool::Concurrency()` | `0` |
| `batchsize` | int (`:302`) | **yes** | `0` = `Indexer::BatchEstimate(bw, nThread, numIdx)` (`idx.hpp:235`) | `0` |
| `opath` | string | optional (`:305`) | output prefix | `""` |
| `datafile` | string (`:306`) | **yes** | output HDF5 (must be h5) | `SphInx_Scan.h5` |
| `vendorfile` | string | optional (`:307`) | `.ang` or `.ctf`; omitted → not written | `reindexed.ang` |
| `ipfmap` | string | optional (`:308`) | PNG (writer is always PNG regardless of extension) | `ipf.png` |
| `qualmap` | string | optional (`:309`) | PNG of the (normalized) cross-correlation metric | `qual.png` |

Namelist syntax is Fortran-ish, parsed by `include/util/nml.hpp` (`nml::NameList::read`, comment char `!`, `:142`). Values are `Variant{Bool,Int,Double,String}` (`:59-104`); bools are `.TRUE.`/`.FALSE.`; unused keys reported by `fullyParsed()`/`unusedTokens()` (`:196-200`). `writeParameters(H5::Group)` and `writeFile(H5::Group, name)` (`:205-210`) dump the parsed values and the raw file lines into the output HDF5.

**Addendum (2026-09-02, `specs/2026-09-02-sht-interop/requirements.md` D4, D5, D6; also amends the §3.5 mask note)** -- namelist quirks: the `!` comment rule is **column 0 only** (`nml.hpp:307` `line.front()`; the doc comment `:291` says first-non-space -- code wins, measured), whitespace-only lines are not skipped (measured error), two+ leading spaces raise a different message than zero (measured both), whitespace is stripped inside the 2nd+ string of a quoted list (sticky `skipws`, `:364-368` -- measured, decisive); the vendor whitelist accepts lowercase `tsl` only (`TSL` rejected, template comment notwithstanding), `"tsl"` is duplicated in the parse check and dead in the `idx.hpp:227` Bruker branch; **`circmask > 0` keeps the processor-side circular mask at radius r** (`imprc.hpp:108-122` builds `CircMask` and sets `msk=true` for any `r >= 0`, fed at `idx.hpp:254`) while only `Geometry::circ` stays false (`maskPattern(circRad == 0)`, `idx.hpp:230`) -- earlier "silently disables the mask" wording was wrong; `patdset`/`scanname` requiredness is filesystem-dependent (`isHdf5(patFile)` gates, `ebsd/nml.hpp:242-246`, `:253-254` -- measured both ways); `sanityCheck` has 13 checks and its negativity bounds are live (`int32_t` fields -- `nregions=-5` and `nthread=-1` exit 1, measured); the double-`ipath` prefix when `psymfile` is set (`nml.hpp:247`), and `parse_nml` re-prefixes `ipath` on every read so the C++ round trip double-prefixes; `to_string` emits the `" /"` terminator only inside the qualmap block (`:464-468`); stream doubles print at 6 significant digits; `delta` cancels out of the geometry for TSL/Oxford/Bruker pctr (measured bitwise: Bruker delta 250 == delta 500 `.ang`).

---

## 2. The SHT master-pattern file format (v1.1)

Definition: `build/_deps/shtfile-src/sht_file.in.hpp` (2247 lines; configured to `build/_deps/shtfile-build/sht_file.hpp` with `@SHT_FILE_VERS@` → software-version string). Upstream: `https://github.com/EMsoft-org/SHTfile`, fetched by `CMakeLists.txt:151-162`. Version constants: `VERSION_MAJOR = 1`, `VERSION_MINOR = 1` (`:57-58`). ReadMe.md:13 states EMSphInx ≥0.2 uses **file version 1.1**, identical to the [SHTdatabase](https://github.com/EMsoft-org/SHTdatabase); EMSphInx 0.1 used an older, incompatible format.

### 2.1 Overall stream layout (`File::write` `:1995-2005`, `File::read` `:2010-2031`)

```
FileHeader        (40 B fixed + doi string + notes string)
MasterPatternData (8 B fixed + numXtal × CrystalData + numXtal × SimulationData)
HarmonicsData     (8 B fixed + doubCnt × float64)
uint32 crc        (CRC-32C over everything above, seed 0x00000000)
```
Every variable-length string is **zero-padded up to a multiple of 8 bytes**, while the stored length field is the *unpadded* length (`setDoi` `:1182-1188`, `setFormula` `:1470-1476`, etc.). Readers must round up: `if(len%8) len += 8-(len%8)` (`:1136-1137`, `:1414-1418`).

CRC is **CRC-32C with the *normal* (non-reflected) polynomial 0x1edc6f41**, computed with a 256-entry LUT and reflected-style update `crc = (crc>>8) ^ LUT[(crc&0xFF)^byte]`, `crc = ~crc` at both ends (`detail::crc32c` `:947-1005`). This is *not* the standard reflected Castagnoli CRC32C — a Python port must reimplement this exact LUT/loop (LUT literal at `:967-1000`) or copy the table.

Byte order: detected from the magic bytes. `*sht` = little-endian, `*SHT` = big-endian (`FileHeader::fileBig` `:1163-1168`); mismatch triggers a full `byteSwap()` after read (`:2025-2029`). (The 64-bit `byteSwap` at `:925-934` is buggy — uses `||` instead of `|` — irrelevant for LE files but worth not copying.)

### 2.2 Field-by-field byte layout

**FileHeader — 40 bytes fixed** (`:138-257`, ctor `:1053-1063`):

| Off | Type | Field | Notes |
|---|---|---|---|
| 0 | 4×char | `magicBytes` | `*sht` (LE) / `*SHT` (BE) |
| 4 | 2×int8 | `fileVersion` | `{1,1}` |
| 6 | 2×int8 | `resBytes` | must be 0 |
| 8 | 8×char | `softwareVersion` | e.g. `ve49ad6b` (git hash of writer) |
| 16 | int8 | `modality` | `Modality` enum: 0 Unknown, **1 EBSD**, 2 ECP, 3 TKD, 0x11 PED, 0x21 Laue (`:97-104`) |
| 17 | 3×int8 | `resBytes2` | must be 0 |
| 20 | float32 | `beamEnergy` | keV; sanity `[0, 10000]` (`:1091-1092`) |
| 24 | float32 | `primaryAngle` | deg, `[-360,360]` |
| 28 | float32 | `secondaryAngle` | deg |
| 32 | float32 | `reservedParam` | |
| 36 | int16 | `doiLen` | unpadded byte length |
| 38 | int16 | `noteLen` | unpadded byte length |
| 40 | char[pad8(doiLen)] | doi utf8 | |
| … | char[pad8(noteLen)] | notes utf8 | |

**MasterPatternData — 8 bytes fixed** (`:475-539`):

| Off | Type | Field |
|---|---|---|
| 0 | int8 | `numXtal` (# crystals averaged) |
| 1 | uint8 | `sgEff` (**effective** space group of the spherical function; `[1,230]`) |
| 2 | int8 | `pijk` (+1/−1 quaternion convention; EMsoft always **+1**, `:2062`) |
| 3 | int8 | `rotSense` (97=`'a'` active, 112=`'p'` passive; EMsoft always **passive**, `:2063`) |
| 4 | int8 | `modality` |
| 5 | int8 | `vendor` (`Vendor` enum: 0 Unknown, **1 EMsoft**, `:106-109`) |
| 6 | int16 | `simMetaSize` (bytes per SimulationData record; 0 = none) |

then `numXtal` × `CrystalData`, then `numXtal` × `SimulationData` (write order `:1575-1585`).

**CrystalData — 72 bytes fixed + atoms + 5 strings** (`:320-472`):

| Off | Type | Field | Units |
|---|---|---|---|
| 0 | uint8 | `sgNum` | 1–230 |
| 1 | int8 | `sgSet` | origin choice (1 or 2 for the 24 dual-origin groups listed `:1235-1244`) |
| 2 | int8 | `sgAxis` | `Axis` enum 1–6 (orth ABC/BAC/CAB/CBA/BCA/ACB, mono B/nB/C/nC/A/nA) `:322-329` |
| 3 | int8 | `sgCell` | `Cell` enum 1–3 (mono cell 1/2/3, trig hex/rhomb, tet CF, trigHex H) `:332-336` |
| 4/8/12 | 3×float32 | `oriX/Y/Z` | origin shift in **24ths** of a/b/c |
| 16 | 6×float32 | `lat` | `{a,b,c,α,β,γ}`, **a,b,c in nm**, angles in degrees |
| 40 | 4×float32 | `rot` | quaternion `{w,x,y,z}` applied to the spherical signal; unit-norm enforced (`:1321-1325`) |
| 56 | float32 | `weight` | averaging weight |
| 60 | int16 | `numAtoms` | |
| 62/64/66/68/70 | 5×int16 | `formulaLen`,`matNameLen`,`structSymLen`,`refsLen`,`noteLen` | unpadded lengths |
| 72 | `numAtoms` × AtomData(32 B) | | |
| … | 5 × pad8 strings | form, name, symb, refs, note (in that order) | |

**AtomData — 32 bytes** (`:260-317`):

| Off | Type | Field | Units |
|---|---|---|---|
| 0/4/8 | 3×float32 | `x,y,z` | **24ths of a/b/c** (so real fractional = value/24); `[0,24)` |
| 12 | float32 | `occ` | (0,1] |
| 16 | float32 | `charge` | atomic units, `[-18, +Z]` |
| 20 | float32 | `debWal` | Debye-Waller, **nm²** |
| 24 | float32 | `resFp` | reserved |
| 28 | int8 | `atZ` | atomic number `[1,118]` |
| 29 | 3×int8 | reserved | must be 0 |

**SimulationData / `EMsoftED` — 88 bytes** (used for EBSD/ECP/TKD, `:741-849`):

| Off | Type | Field |
|---|---|---|
| 0 | 8×char | `emsoftVersion` |
| 8/12/16 | 3×float32 | `sigStart`, `sigEnd`, `sigStep` (deg; `NAN` for "full" single-angle mode) |
| 20 | float32 | `omega` |
| 24 | float32 | `keV` |
| 28 | float32 | `eHistMin` |
| 32 | float32 | `eBinSize` |
| 36 | float32 | `depthMax` (nm) |
| 40 | float32 | `depthStep` (nm) |
| 44 | float32 | `thickness` (nm; `+inf` for bulk) |
| 48 | int64 | `totNumEl` (= `totnum_el` × `multiplier`, `:2216-2218`) |
| 56 | int16 | `numSx` (MC grid) |
| 58 | 2×int8 | reserved |
| 60/64/68 | 3×float32 | `c1`, `c2`, `c3` (Bethe) |
| 72 | float32 | `sigDbDiff` |
| 76 | float32 | `dMin` (nm) |
| 80 | int16 | `numPx` (master-pattern half size) |
| 82 | int8 | `latGridType` (**1 = square Lambert, 2 = square Legendre**) |
| 83 | 5×int8 | reserved |

`EMsoftXD` (32 B, Laue only) also exists (`:852-904`): `emsoftVersion`, `lambdaMin/Max` (nm), `kappaVMF`, `intFactor`, `numPx`, `patchW`.

**HarmonicsData — 8 bytes fixed + payload** (`:542-637`):

| Off | Type | Field |
|---|---|---|
| 0 | int16 | `bw` (bandwidth) |
| 2 | int8 | `zRot` (z-rotational order used for compression) |
| 3 | int8 | `cmpFlg` bitmask: `0x01` inversion, `0x02` equatorial mirror, `0x04` mirror with +y normal (`Nmm`), `0x08` mirror at φ=90/zRot (`-42m`, `31m`, `3m1`-rotated) — `0x04` and `0x08` are mutually exclusive (`:559-567`, `:1678`) |
| 4 | int32 | `doubCnt` (number of float64 stored) |
| 8 | `doubCnt` × float64 | packed harmonics |

### 2.3 Harmonics storage / compression

Uncompressed layout is a dense `bw × bw` complex array with **`a^l_m` at `alm[m*bw + l]`** (i.e. m-major rows, `l < m` entries are padding, `master.hpp:173-176`).

`NumHarm(b, n, f)` (`:1672-1698`) and `PackHarm`/`UnpackHarm` (`:1706-1831`) implement:
- skip whole rows where `n>1 && m % n != 0` (systematic zeros from z-rotational symmetry);
- within a row skip `l` where `inv && l%2 != 0` (inversion) or `mirZ && (l+m)%2 != 0` (equatorial mirror);
- **storage type per row**: complex (2 doubles per surviving `(l,m)`) by default; **strictly real** (1 double) if `mirY (0x04)`; if `mirX (0x08)` then rows with `m % (2n) == 0` are real, all others **strictly imaginary** (1 double each).

`SpaceGroupRot(sg)` and `SpaceGroupCmp(sg)` are 230-entry LUTs (`:1838-1849`, `:1858-1869`) that derive `zRot` and `cmpFlg` from the effective space group, **assuming standard settings** (monoclinic unique-axis b, orthorhombic abc). A Python port can copy these two tables verbatim.

### 2.4 `mp2sht`: EMsoft HDF5 → `.sht`

Two stages.

**Stage A — harmonics** (`mp2sht.cpp:53-55`):
```cpp
emsphinx::MasterSpectra<double> spec(emsphinx::MasterPattern<double>(argv[1]), /*bw*/384, /*nrm*/true);
```
- `MasterPattern<Real>::read` (`include/idx/master.hpp:242-347`) reads from the EMsoft h5:
  - `/NMLparameters/MCCLNameList/sig` → sample tilt; `/NMLparameters/MCCLNameList/EkeV` → kV (`:249-250`).
  - `/CrystalData` group → `Phase::readMaster` (lattice + space group → point group) (`:261`).
  - **`EMData/MCOpenCL/accum_e`** — 3D `(nE, ny, nx)` uint32 Monte-Carlo histogram (`:265-271`). Energy weights = column sums over the spatial slice, normalised to sum 1 (`:274-278`). *Note the summation loops over `slicePts = dims[0]*dims[1]` and strides by `eCounts.size()`, i.e. it treats the array as `[slicePts][nE]` — effectively "sum over everything but the energy axis".*
  - Detects `EMData/EBSDmaster` vs `EMData/ECPmaster` by iterating `EMData` children (`:282-295`).
  - Reads **`EMData/EBSDmaster/mLPNH`** and **`EMData/EBSDmaster/mLPSH`** (`:299-300`), expected 4D `{atom, energy, x, y}` for EBSD (3D `{atom,x,y}` for ECP, with a dummy energy axis inserted and weights = `{1}`) (`:306-316`). Both hemispheres must have identical shapes (`:305`).
  - **Sums over atoms** (unweighted plain `+`, `:335-338`) — note this differs from `scripts/master_sphere.py:78-82` which weights atoms by occupancy.
  - **Energy-weighted average** using the MC weights (`:341-346`) → `nh`, `sh` (square **Lambert**, side `dim = dims[2]`).
- `MasterSpectra` ctor (`master.hpp:550-595`):
  - target Legendre grid side `dimLg = bw + 2 + (bw even ? 1 : 0)` (odd) (`:551`); `toLegendre(dimLg)` if the pattern is Lambert (`:552-554`). `toLegendre` first rescales the Lambert grid by √2 via `image::Rescaler` (FFT-based), then bilinearly interpolates onto Legendre-latitude grid normals (`:381-416`).
  - If `nrm`: builds per-pixel solid-angle weights from `square::solidAngles`, halves weights on all four square edges (equator double-counting), makes the **area-weighted mean 0 and area-weighted stdev 1 across both hemispheres jointly** (`:557-584`).
  - `square::DiscreteSHT<Real>::Legendre(dim).analyze(nh, sh, alm, bw, bw)` (`:593`).
- **Stage B — metadata + write** (`mp2sht.cpp:58-130`). Reads, directly from the same h5 file:

| h5 path | → | slot |
|---|---|---|
| `CrystalData/SpaceGroupNumber` | `iprm[3]`, and copied to `iprm[0]` = **effective sg** | `:70` |
| `CrystalData/SpaceGroupSetting` | `iprm[4]` | `:71` |
| `CrystalData/LatticeParameters` | `fprm[4..9]` (a,b,c,α,β,γ) | `:72` |
| `CrystalData/Natomtypes` | `iprm[5]` | `:73` |
| `CrystalData/AtomData` | `aCd` (`nAt*5` floats: x,y,z,occ,DW) | `:76` |
| `CrystalData/Atomtypes` | `aTy` | `:77` |
| `NMLparameters/MCCLNameList/sig` | `fprm[10]` (sigStart) | `:80` |
| — | `fprm[11]=fprm[12]=NAN` (sigEnd/sigStep) | `:81-82` |
| `…/MCCLNameList/omega` | `fprm[13]` | `:83` |
| `…/MCCLNameList/EkeV` | `fprm[14]` | `:84` |
| `…/MCCLNameList/Ehistmin` | `fprm[15]` | `:85` |
| `…/MCCLNameList/Ebinsize` | `fprm[16]` | `:86` |
| `…/MCCLNameList/depthmax` | `fprm[17]` | `:87` |
| `…/MCCLNameList/depthstep` | `fprm[18]` | `:88` |
| — | `fprm[19] = +inf` (thickness) | `:89` |
| `…/BetheList/c1,c2,c3,sgdbdiff` | `fprm[20..23]` | `:90-93` |
| `…/EBSDMasterNameList/dmin` | `fprm[24]` | `:94` |
| `…/MCCLNameList/totnum_el` | `iprm[6]` | `:96` |
| `…/MCCLNameList/multiplier` | `iprm[7]` | `:97` |
| `…/MCCLNameList/numsx` | `iprm[8]` | `:98` |
| `…/EBSDMasterNameList/npx` | `iprm[9]` | `:99` |
| — | `iprm[10] = 1` (**latGridType = square Lambert**) | `:100` |

  Header scalars: `iprm[1] = Modality::EBSD`, `iprm[2] = bw = 384`, `fprm[0] = spec.getKv()`, `fprm[1] = spec.getSig()`, `fprm[2] = fprm[3] = 0` (`:60-65`).
  Formula string = concatenation of unique element symbols in ascending Z (via `std::set<int32_t>`, no multiplicity — the code comment admits this, `:102-115`).
  Fixed strings: `doi = "https://doi.org/10.1016/j.ultramic.2019.112841"`, `note = "created with mp2sht"`, `emVers = "5_0_0_0"` (`:119-121`). **All of name/structure-symbol/references/note for the crystal are written empty** (`:122-126`).
  `initFileEMsoft(iprm, fprm, doi, note, (double*)spec.data())` (`:127`, impl `sht_file.in.hpp:2047-2078`) sets the header + derives `zRot`/`cmpFlg` from `SpaceGroupRot/Cmp(sgEff)` and packs the harmonics. `addDataEMsoft(...)` (`:128`, impl `:2139-2230`) appends the crystal (converting atom fractional coords to 24ths, with exact special-casing of 1/6, 1/3, 2/3, 5/6 → 4, 8, 16, 20, `:2158-2177`; `charge` forced to 0; `rot` = identity; `weight` = 1) and the `EMsoftED` block.

**Note the asymmetry**: `mp2sht` reads *only* `sig` for the header's primary angle, and hard-codes `sigEnd`/`sigStep` to NaN, thickness to +inf, `latGridType` to 1 — i.e. `.sht` files produced by `mp2sht` always describe a single-tilt, bulk, square-Lambert EMsoft EBSD simulation.

### 2.5 Reading a `.sht` back (what `IndexEBSD` actually uses)

`MasterSpectra<Real>::read` (`master.hpp:619-640`) keeps only: `beamEnergy` → kv, `primaryAngle` → sig, `harmonics.bw`, `PointGroup(mpData.sgEff())`, lattice parameters when `numXtal == 1`, and the unpacked `alm`. Phase **name is cleared** (`:631`) — hence the empty `MaterialName`/`Formula` in the benchmark `.ang`.

---

## 3. Pattern input formats & ROI/mask semantics

All in `include/modality/ebsd/pattern.hpp` (919 lines).

### 3.1 Class hierarchy (`:51-258`)

- `PatternFile : ImageSource` — `numPat()`, `flipY()`, `width/height/pixBytes/imBytes/numPix`, abstract `extract(char* out, size_t cnt) -> vector<size_t>` (thread-safe, returns the scan indices of the patterns written).
- `ContigousPatternFile` (in-memory pointer walk, `:714-725`), `StreamedPatternFile` (istream read, `:732-741`), `ChunkedPatternFile` (declared, unused), `BufferedPatternFile` (owns a `std::vector<char>`, `:188-201`), `IfStreamedPatternFile` (`std::ifstream` + byte offset; `setOffset` computes `num = (fileBytes - offset)/imBytes` and seeks, `:746-750`), `OxfordPatternFile` (`:221-258`).
- `Bits` enum: `U8`(1B), `U16`(2B), `F32`(4B), `UNK` (`setShape` `:350-362`).

### 3.2 Format dispatch: `PatternFile::Read(name, aux, px, py)` (`:427-554`), by **lowercased extension**

| Ext | Reader | Details |
|---|---|---|
| `up1` | `IfStreamedPatternFile`, `Bits::U8` | header via `detail::UpHeader::read` (`:296-341`); `setOffset(header.dStart)`; **`flp = true`** (EDAX needs vertical flip) (`:433-451`) |
| `up2` | same, `Bits::U16` | idem |
| `h5`/`hdf`/`hdf5` | dataset `aux` must be **3D** `(numPat, h, w)` (`:458-459`) | see §3.3 |
| `data` | `IfStreamedPatternFile`, `Bits::F32`, offset 0, `flp=false` (`:528-534`) | **raw headerless float32**; `px`,`py` MUST be supplied by the caller (from `patdims`) |
| `ebsp` | `OxfordPatternFile` (`:535-551`) | see §3.5 |
| anything else | `throw std::runtime_error("couldn't find EBSD pattern reader for '...'")` |

**Not supported**: `.oh5`, `.h5oina`, `.osc`, `.bcf`, `.hdf5` from Oxford's new format. A `.h5oina` would only work if it happened to have a root `Manufacturer` dataset (it does not — h5oina uses `/1/EBSD/...` and a `Manufacturer` attribute, not a root dataset), so `GetVendor` (`:608-637`) would throw. This is a **gap a Python port should close** — kikuchipy already reads h5oina/oh5.

### 3.3 HDF5 pattern reader specifics (`:452-527`)

- Vendor string is required: `GetVendor(name)` iterates root children looking for a dataset named `Manufacturer` **or `" Manufacturer"`** (some EDAX files have a stray leading space, `:613-622`), then reads it as variable-length string with a fixed-length 128-byte fallback (`:627-636`).
- **Flip decision by vendor** (`:463-471`): `EDAX` → flip; `EMsoft` → flip; `Oxford`, `Bruker`, `Bruker Nano`, `DREAM.3D` → no flip; **anything else throws** `"unknown EBSD vendor: ..."`.
- dtype mapping: `NATIVE_UINT8`/`NATIVE_UCHAR` → U8, `NATIVE_UINT16` → U16, `NATIVE_FLOAT` → F32; anything else throws (`:474-480`).
- **Fast path**: if the dataset has no external files, layout is COMPACT or CONTIGUOUS, `getNfilters() != 0` (**this condition is inverted — the comment says "we can't memory map compressed datasets" but the test requires filters to be present**, `:494`), and `getOffset() != HADDR_UNDEF`, it memory-maps/streams the raw bytes at that offset (`:496-507`). Otherwise it falls back to reading the whole dataset into RAM as `BufferedPatternFile` (`:511-517`), reading with `H5::PredType::NATIVE_UINT8` regardless of dtype (byte-wise copy).
- `PatternFile::SearchH5(name)` (`:556-603`) recursively walks the file and returns every 3D dataset with `dims[2] > 4` as a candidate pattern dataset (the `>4` filters out RGB(A) coordinate-system images). This is what the GUI's dataset dropdown uses.
- `PatternFile::GetFileDims(name, w, h, bit, num, aux)` (`:381-419`): for `up1/up2/ebsp` it fully opens the file; for `data` it returns `bit=F32`, `num = file size in bytes`, `w=h=-1`; for h5 it reads the 3D extents + dtype; other extensions → returns `false`.
- `PatternFile::FromImages(fmt, px, py)` (`:644-707`): `printf`-style filename template `fmt` with `(i, j)` indices; **only `.bmp` is implemented** (8-bit or 24-bit via `include/util/bmp.hpp`); it also writes a stray debug file `test.raw` (`:699-700`).

**Addendum (2026-09-02, `specs/2026-09-02-sht-interop/requirements.md` D2)**: the HDF5 mmap gate `0 != props.getNfilters()` (`pattern.hpp:494`) is inverted and unsatisfiable for contiguous datasets: every HDF5 pattern file takes the buffered branch, which reads `NATIVE_UINT8` for any dtype (`:515`) -- uint16/float32 HDF5 patterns are corrupted (measured end-to-end); **filters are the real constraint** (gzip-compressed `/patterns` is fatal, chunked-unfiltered and alloc-late are fine -- measured); byte-swapped dtypes are rejected (`NATIVE_*` comparison `:476-480`, measured); a vlen **UTF-8** `Manufacturer` fails with a misleading `H5Dread` error (`GetVendor` fallback `:631-635` throws into the outer handler `:519` -- measured).

### 3.4 Oxford `.ebsp` reader (`OxfordPatternFile`, `:754-913`)

Layout (as reverse-engineered by the authors; comments at `:755-761`, `:788-792`, `:809`):
- 8-byte header, **or 9 bytes if `header[0] == 0xFC`** (ebsp "version 4", patched 2023-02-21, `:767-771`, `:804-808`, `:813-817`).
- Then `numPat` × `uint64` absolute file offsets, one per pattern (in *acquisition* order, **not** necessarily ascending).
- Each pattern block: `uint32 leadIn`, `uint32 height`, `uint32 width`, `uint32 bytes` (16 B header) — **note the read order is leadIn, height, width, bytes** (`:795-798`, `:881-884`) — then `leadIn` padding bytes, then `bytes` of pixel data, then an 18-byte tail: `uint8 ix`, `double x` (µm), `uint8 iy`, `double y` (µm) (`:899-904`).
- `numPat` is computed from **file size**: `numPat = fsize / (bytes + 42)` where 42 = 8 (offset entry) + 16 (block header) + 18 (tail) (`:809-810`).
- Bit depth inferred: `bytes == w*h` → U8, `bytes == w*h*2` → U16, else throw (`:824-832`).
- Offsets are converted to a permutation: `blockBytes = 16 + leadIn + bytes + 18`; `off = (offsets[i] - minOffset)/blockBytes` must be exact, and `idx[off] = i` gives "which scan index lives at each sequential file position" (`:834-852`). Extraction therefore reads sequentially and returns out-of-order indices — this is exactly why `extract()` returns an index vector rather than assuming contiguity.
- `flp = false`.
- `extract(out, cnt, vx, vy)` optionally harvests the per-pattern stage coordinates — used by `EBSPDims`.

### 3.5 ROI / mask semantics

Two independent masks:

**(a) Scan-level ROI** — `include/idx/roi.h`, `emsphinx::RoiSelection`.
- Shapes: `Rectangle`, `Ellipse`, `Polygon` (`DrawMode`, `:43-47`); optional `inv` inversion flag.
- **String grammar** (`to_string` `:565-585`, `from_string` `:589-627`, docs `emsphinxebsd.rst:145-164`):
  - `""` or `"0"` → no ROI (index everything).
  - optional leading `i` → inverted (select the *excluded* region).
  - optional leading `e` → ellipse (bounding box given as a rectangle).
  - 4 comma-separated ints → rectangle `x0, y0, dx, dy` (stored internally as two corner points `p0=(x0,y0)`, `p1=(x0+dx, y0+dy)`).
  - >4 ints (even count) → polygon vertex list, **must be closed** (first point repeated as last) else throws `"polygon not closed in ROI string"`.
  - examples from the docs: `"12, 34, 44, 45"`; `"12, 34, 12, 79, 56, 79, 56, 34, 12, 34"`; `"ie10, 20, 100, 100"`.
- `buildMask(w, h) -> std::vector<char>` (`:435-505`): initialises to `inv ? 1 : 0`, clips the shape bounding box into `[0,w]×[0,h]`, then:
  - Rectangle: fills `[yMin,yMax) × [xMin,xMax)` — **half-open, and pixels on the max edges are excluded**.
  - Ellipse: `(2dx)²/(2a)² + (2dy)²/(2b)² <= 1` using doubled integer arithmetic.
  - Polygon: non-zero **winding number** test (`orient2` cross product, `:430`, algorithm from geomalgorithms.com a03).
- Consumed at `include/modality/ebsd/idx.hpp:205-215`:
  ```
  idxMask = roi.buildMask(scanDims[0], scanDims[1]);
  numIdx  = count(idxMask != 0);
  if(numIdx == 0) { fill(idxMask, 1); numIdx = idxMask.size(); }   // empty ROI ⇒ index everything
  if(refine) for(char& v : idxMask) if(v==1) v = 3;                 // bitmask: 0x01 index, 0x02 refine
  ```
  The work item checks `msk[i]`: `0x01` → full index (+`0x02` → refine during index), `0x02` alone → refine-only from an existing orientation (`idx.hpp:408-450`).

**(b) Detector-level circular mask** — `circmask` in the namelist → `Geometry::maskPattern(circRad == 0)` (`idx.hpp:230`) and `PatternProcessor::setSize(w, h, circRad, gausBckg, nRegions)` (`idx.hpp:254`, impl `include/modality/ebsd/imprc.hpp:108-146`):
- `r == -1` → no mask, plain `gaussian::BckgSub2D<Real>(w,h)`.
- `r == 0` → `BckgSub2D::CircMask(w,h)` (largest inscribed circle).
- `r > 0` → `BckgSub2D::CircMask(w,h,r)`.
- The mask (`bkg.msk`) is used both for the Gaussian background fit **and** as the AHE validity mask (`imprc.hpp:157`, `:176`, `:180`, `:186`).
- Note `Geometry::circ` only becomes `true` for `circRad == 0` (`idx.hpp:230`), whereas the image processor honours any `r > 0`.

**Pattern preprocessing order** (`imprc.hpp:166-190`): optional 2D Gaussian background fit + subtract → (if AHE) rescale to uint8 over min/max → adaptive histogram equalisation with the mask → float output. If neither is enabled it is a plain cast to `Real`.

**Pattern-centre conventions** (`include/modality/ebsd/detector.hpp:250-279`, `idx.hpp:221-229`):
```
EMsoft : cX = xpc; cY = ypc; sDst = L                         (pixels, pixels, µm)
EDAX   : cX = x* · w - w/2 ; cY = y* · w - h/2 ; sDst = z*·w·pX
Oxford : cX = (x*-0.5)·w   ; cY = (y*-0.5)·h   ; sDst = z*·w·pX
Bruker : cX = (x*-0.5)·w   ; cY = (0.5-y*)·h   ; sDst = z*·h·pX
```
and the inverse used when writing the vendor file (`idx.hpp:245-248`):
```
ratio = w/h ; xStar = (cX + w/2)/w ; yStar = (cY + h/2)/w ; zStar = sDst/(pX·w)   [EDAX convention]
```
Cross-vendor conversion in `ebsd::Calibration<Real>::setVendor` (`include/xtal/orientation_map.hpp:176-199`).

---

## 4. Output formats and `orientation_map.hpp`

Written by `IndexingData<Real>::save` (`include/modality/ebsd/idx.hpp:318-370`).

### 4.1 The output HDF5 (`datafile`)

Created in two passes. **Pass 1**, `Namelist::writeFileHeader()` (`nml.hpp:581-617`), called from the `IndexingData` constructor:
```
/Manufacturer          str = "EMSphInx"
/Version               str = emsphinx::Version   (e.g. "master:60f3517")
/NMLfiles/IndexEBSD    str[nLines]  — the raw namelist file, one string per line
/NMLparameters/IndexEBSD/<key>      — every parsed key as a typed dataset (int32/float64/uint32-for-bool/str)
/EMheader/Date         str "%a %b %d %Y"
/EMheader/HostName     str  (getComputerName())
/EMheader/UserName     str  (getUserName())
/EMheader/ProgramName  str "IndexEBSD (index_ebsd.cpp)"
/EMheader/Version      str
```
**Pass 2**, `save()` (`idx.hpp:320-368`) reopens `H5F_ACC_RDWR` and adds:
```
/EMheader/StartTime    str "%a %b %d %Y %T"
/EMheader/StopTime     str
/EMheader/PatPerS      float64
/Scan 0i/…             one group per orientation map (zero-padded to ceil(log10(nMaps+1)) digits)
```
Number of maps: `om.size() = phases.size() + extraScans`, where `extraScans = (1 == phases.size()) ? phases[0].pseudoSym().size() : 1` (`idx.hpp:249-250`). Single-phase, no psym ⇒ exactly one `Scan 1`.

Each `Scan i` group contains `OrientationMap::writeH5` output (`orientation_map.hpp:618-700`) plus 3 image datasets (`idx.hpp:365-367`):
```
Scan i/EBSD/Header/nColumns                                  uint32 scalar
Scan i/EBSD/Header/nRows                                     uint32 scalar
Scan i/EBSD/Header/Step X, Step Y                            float32 scalar (µm)
Scan i/EBSD/Header/Operator, Scan ID                         var-len str
Scan i/EBSD/Header/Sample Tilt, Working Distance             float32 scalar
Scan i/EBSD/Header/Grid Type                                 var-len str = "SqrGrid"
Scan i/EBSD/Header/Notes, Sample ID                          var-len str (created empty)
Scan i/EBSD/Header/Pattern Center Calibration/x-star,y-star,z-star  float32 scalar
Scan i/EBSD/Header/Phase/<n>/…                               Phase::writeEBSD: MaterialName, Symmetry,
                                                             Lattice Constant a/b/c/alpha/beta/gamma
Scan i/EBSD/Data/Phase                                       uint8 [w*h]
Scan i/EBSD/Data/Phi1, Phi, Phi2                             float32 [w*h]  (Bunge Euler, RADIANS)
Scan i/EBSD/Data/Metric                                      float32 [w*h]  (spherical cross correlation)
Scan i/EBSD/Data/IQ                                          float32 [w*h]
Scan i/IPF Map                                               uint8  [h, w, 3]
Scan i/XC Map                                                uint8  [h, w]   ← note: created with DataSpace(2, dims)
Scan i/IQ Map                                                uint8  [h, w]
```
Conventions: data arrays are **1D, row-major, length `w*h`**; Euler angles in **radians** (`qu2eu`); the map is a `SqrGrid`; pattern centre is stored in the **EDAX/TSL convention** (`idx.hpp:244`).

Confirmed against `benchmarks/GPU_test_cpu.h5` (dumped live): exactly the above, plus a `feature/GPU`-only `/EMheader/ComputeBackend` and `NMLparameters/IndexEBSD/{backend,gpudevice}`.

Two implementation quirks to reproduce or fix in Python: `save()` recomputes `ipfMap/xcMap/iqMap` from **`om.front()`** inside the per-map loop rather than `om[i]` (`idx.hpp:351-353`), so for multi-phase/psym runs every `Scan i` gets the *first* map's images; and `XC Map`/`IQ Map` are created with a 2-D dataspace taken from a 3-element `dims` array (`:366-367`).

**Addendum (2026-09-03, `specs/2026-09-03-spherical-indexing-emsphinx-regression/requirements.md` D3)**: the `Scan 1/EBSD/Data` dtypes are confirmed by measurement on the Phase 10 canonical route -- `Phase` is **uint8** exactly as tabled above (Phase 9 D7, `specs/2026-09-02-sht-interop/requirements.md`, had recorded float32 for it -- corrected by measurement), Phi1/Phi/Phi2/Metric/IQ are float32; the datafile, not the `.ang`, is the Phase 10 reference payload.

### 4.2 Vendor files

`OrientationMap::write(fileName)` (`orientation_map.hpp:584-613`) dispatches by extension: `.hdf/.h5/.hdf5` → `writeH5` into a group literally named `"Scan 1"`; otherwise try `toTSL().write()` (`.ang`), then `toHKL().write()` (`.ctf`).

**`.ang`** — `include/xtal/vendor/tsl.hpp`, `writeAngHeader` (`:702-762` region) / `writeAngData`:
```
# TEM_PIXperUM     %.6f
# x-star / y-star / z-star / WorkingDistance   %.6f
#
# Phase %d / MaterialName / Formula / Info / Symmetry (TSL sym code)
# NumberFamilies %d
# LatticeConstants a b c alpha beta gamma   (%.3f, ANGSTROMS — nm*10 at toTSL():329)
# hklFamilies … (h k l useIdx intensity showBands)     [none written by EMSphInx]
# ElasticConstants ×6 rows of 6                        [all zeros]
# Categories
#
# GRID:            SqrGrid
# XSTEP: / YSTEP: / NCOLS_ODD: / NCOLS_EVEN: / NROWS:
#
# OPERATOR:  / # SAMPLEID:  / # SCANID:
```
then one row per point, row-major `j` outer / `i` inner:
`phi1 Phi phi2 x y iq ci phase [sem [fit]]` with widths/precisions `%9.5f %9.5f %9.5f %12.5f %12.5f %7.1f %6.3f %2d`. Euler angles in **radians** (TSL native). `toTSL()` (`:297-354`) allocates with `tokenCount = 8`, i.e. **no SEM/Fit columns**; `ci` ← EMSphInx `metric`, `iq` ← `imQual`; `x[i] = xStep*i`, `y[j] = yStep*j`; phase indices are copied 0-based; `phaseList[i].num = i+1`, `sym = pg.tslNum()`.

Verified against `benchmarks/GPU_test_cpu.ang`: 34 header lines then 149776 data rows (149810 total), `Symmetry 43` (m-3m), `LatticeConstants 2.866 …` (Fe bcc), `x-star 0.480000 y-star 0.505000 z-star 0.560001`, `NCOLS_ODD/EVEN 407`, `NROWS 368`, and exactly **5727 rows with CI > 0** — matching the ROI `63, 21, 83, 69` ⇒ 83×69 = 5727 indexed points. Un-indexed points are written as all-zero rows (with `-0.00000` for phi1).

**Addendum (2026-09-03, `specs/2026-09-03-spherical-indexing-emsphinx-regression/requirements.md` D3)**: the `.ang` data columns are written under `std::fixed` (`tsl.hpp:783-794`): Euler at `setprecision(5)` (measured max |ang - h5| = 5.0e-6 rad vs the datafile, exactly the half-ULP bound), ci = the Metric at `setprecision(3)` (max 5.0e-4), iq = the IQ at `setprecision(1)` ({0.1, 0.2, 0.3} observed -- Phase 9's "constant 0.2" explained); the `.ang` is a text-rounded cross-check only, never the Phase 10 payload.

**`.ctf`** — `include/xtal/vendor/hkl.hpp`, `writeCtfHeader` (`:468-507`):
```
Channel Text File
Prj <project>
Author\t / JobMode\t (="Grid") / XCells / YCells / [ZCells] / XStep / YStep / [ZStep]
AcqE1 / AcqE2 / AcqE3
Euler angles refer to Sample Coordinate system (CS0)! Mag\t… Coverage\t… Device\t… KV\t… TiltAngle\t… TiltAxis\t…
Phases\t<n>
a;b;c \t alpha;beta;gamma \t <name> \t <laue> \t <space>     (lengths in ANGSTROMS)
Phase\tX\tY\t…\tEuler1\tEuler2\tEuler3\t…                     (column header, only present columns listed)
```
`toHKL()` (`:409-464`) allocates `CTF_Phase|CTF_Error|CTF_Euler|CTF_X|CTF_Y` only; phase indices are converted **0→1 based**; Euler angles converted to **degrees**; `err` ← `metric`, `bc` ← `imQual` (but `bc`/`mad`/`bands`/`bs` are not allocated so not written); `x[i] = xStep*(i+1)`, `y[j] = yStep*(j+1)`.

### 4.3 IPF and quality images

`idx.hpp:340-345`:
```cpp
Real n[3] = {0,0,1};                                   // IPF reference = sample Z
auto h2r = xtal::sph2rgb<Real>;                        // HSL-like → RGB
ipfMap = om.front().ipfColor(n, h2r);                  // RGB uint8, 3 bytes/px
xcMap  = image::to8Bit(metric…);  iqMap = image::to8Bit(imQual…);
writePng(ipfMap, w, h, 3, opath+ipfName);
writePng(xcMap , w, h, 1, opath+qualName);
```
`writePng` (`idx.hpp:51-63`) uses miniz `tdefl_write_image_to_png_file_in_memory_ex` at `MZ_BEST_COMPRESSION`, no flip. **It always emits PNG regardless of the file extension you give it.** The benchmarks name them `*_IPF.tiff` / `*_CI.tiff` — I verified their magic bytes are `89 50 4E 47 0D 0A 1A 0A … IHDR`, i.e. **they are PNGs with a `.tiff` name**. A Python port should just write real PNGs (or honour the extension).

`OrientationMap::ipfColor` (`orientation_map.hpp:708-728`): normalises the reference direction, rotates it into the crystal frame per pixel (`qu[i].rotateVector`), then `PointGroup::ipfColor(nx, color, h2r)`. Pixels whose `phase >= phsList.size()` (i.e. failed, `phase = (uint_fast8_t)-1`) are left black. There is a latent bug: the function takes an `alpha` flag and computes `NUM = alpha?4:3` but always writes 3 bytes per pixel with stride 3 (`:720-727`); and the vector-returning overload constructs a **`const`** vector and casts away constness (`:737-738`).

### 4.4 Reading vendor files (for `scandims = <file>` and the GUI prefill)

`OrientationMap<Real>::read` (`orientation_map.hpp:471-576`) order:
1. If HDF5 **and** root contains an `EMheader` group → EMsoft/EMSphInx dot-product format: reads `<aux>/EBSD/Header/{nColumns,nRows,Step X,Step Y,Operator,Scan ID,Sample Tilt,Working Distance}`, `…/Pattern Center Calibration/{x-star,y-star,z-star}`, `…/Header/Phase/<n>` via `Phase::readEBSD`, then `<aux>/EBSD/Data/{Phase,Phi1,Phi,Phi2,CI,IQ}` (`:484-554`). **Note it reads `CI`, not `Metric`** — so EMSphInx's own output h5 is not fully round-trippable through this path.
2. Else `tsl::OrientationMap::CanRead` (`.ang` yes, `.h5/.hdf/.hdf5` yes, `.osc` **no**) → `readAng` or `readH5` (`tsl.hpp:223-236`, `:719-…`). The TSL h5 reader requires `EBSD/Header/{…, Camera Elevation Angle, …}` and **all ten** data arrays incl. `X Position`, `Y Position`, `SEM Signal`, `Fit` (`tsl.hpp` readH5).
3. Else `hkl::OrientationMap::CanRead` (`.ctf`) → `readCtf`.
4. Else throw.

Hexagonal grids are rejected on conversion into `xtal::OrientationMap` (`:243`, `:364`).

---

## 5. `MasterXcorr` — pseudo-symmetry prediction

Reference paper: https://doi.org/10.1107/S1600576719011233 (cited in `ReadMe.md:7`).

**What it computes**: the full SO(3) cross-correlation of a master pattern with itself (or with a second phase's master pattern), i.e. `xc(g) = ∫_{S²} f(r) · h(g·r) dΩ` sampled on a ZYZ Euler grid, then extracts and refines all local maxima. Non-trivial maxima that are *not* crystallographic symmetry operators of the phase are candidate pseudo-symmetry operators.

**Algorithm** (`programs/master_xcorr.cpp`):
1. Read both master patterns from EMsoft h5 (`:69-70`).
2. `MasterSpectra<double> p1(mp1, bw, /*nrm*/true)`, same for `p2` (`:72-73`); `p1.removeDC(); p2.removeDC()` (`:75-76`, sets `a^0_0 = 0`).
3. `emsphinx::sphere::Correlator<double> s2Corr(bw)` (`:80`); `s2Corr.correlate(p1.data(), p2.data(), p1.mirror(), p1.nFold(), eu, /*ref*/false)` (`:82`) — **only the symmetry of the first pattern is exploited**, as the usage note says (`:50`).
4. Peak normalisation: for autocorrelation (`masterFile1 == masterFile2`), the reference maximum is forced to the identity grid point `idxIdent = (bw-1)*sl² + (bw/2)*sl + (bw/2)` with `sl = 2bw-1` (`:87-90`); `vMax = s2Corr.refinePeak(...)` (`:91`).
5. Scan the whole `bw × sl × sl` correlation volume (`getXC()`), keep voxels `>= vMax·cutoff·0.95`, test each against its 3×3×3 neighbourhood via `extractNeighborhood<1>` (`:107-140`); convert the grid index to ZYZ Euler angles with `indexEuler` and then to a quaternion with `xtal::zyz2qu` (`:144-145`).
6. Duplicate suppression: any new maximum within **2°** (misorientation from `|q1·q2|`) of an existing one replaces it if brighter (`:148-166`).
7. Sort by intensity descending, then `refinePeak` each one (Newton) and re-sort (`:174-190`); print `intensity  qu` (4 decimals) for all with `intensity >= cutoff` (`:192-195`).

**Outputs**:
- **stdout**: timing lines, `maximum intensity: <vMax>`, then `<relative intensity> <w x y z>` per surviving maximum.
- **`pseudo_sym.h5`** (`:200-203`): a single dataset `"Cross Correlation"`, `float64`, shape `(bw, 2bw-1, 2bw-1)` = the **raw un-normalised** correlation volume (the normalisation line is commented out, `:198`).
- **`pseudo_sym.xdmf`** (`:206-225`): an XDMF v2.2 wrapper describing a `3DCoRectMesh` with origin `0 0 0` and spacing `res = 360/(2bw-1)` degrees in each axis, referencing `pseudo_sym.h5:/Cross Correlation` as a cell-centred scalar — drop into ParaView.
- **`true.svg`** (`:229-230`): `xtal::Diagram diag(mp1, black)` → the master pattern's true point-group stereogram (north hemisphere).
- **`pseudo.svg`** (`:239-280`): the same diagram with each detected non-symmetry operator overlaid. For each maximum: reject if `|q·op| > 0.999` for any of the `numRotOps()` rotational symmetry operators (~2.5°) (`:243-250`); reject duplicates against already-drawn operators with the same cut (`:255-261`); colour scaled by intensity (`c.rgb[0] = c.rgb[1] = intensity`, giving yellow→white); compute the rotation order `order = π/acos(q.w)`, `nFld = round(order)`; if crystallographic draw an n-fold rotor symbol at 0.67 scale (`addRotor`), else draw a dot (`addDot`) (`:264-274`). **Bug**: the crystallographic test is `std::fabs(nFld - nFld) < rotCut` (`:267`) — always 0 < 0.05 — so the `addDot` branch is dead code and every operator is drawn as an n-fold rotor.
- Only the north hemisphere SVG is saved (`getHemi()` defaults to `nth = true`); a lot of commented-out misorientation-histogram code follows (`:282-378`).

The output of this program is what feeds `psymfile` in the indexing namelist (an EMsoft **quaternion** angle file; `MasterData::addPseudoSym(std::string)` in `master.hpp:222-233` requires `Rotation::Quaternion` and skips the identity).

---

## 6. `sht2png`, `xtal::Diagram`, `util/svg.hpp` — visualisation

### 6.1 `sht2png`

`sht2png inputFile sqLegOut [sterOut]`:
1. `MasterSpectra<double> spec; spec.read(argv[1])` (`:81-82`); prints `kV sig`.
2. Reconstructs a real-space **square Legendre** master pattern at `dim = bw + (bw even ? 3 : 2)` (odd) via `square::DiscreteSHT<double>::synthesize(spec.data(), nh, sh)` (`:87-91`).
3. Rescales both hemispheres jointly to 8-bit using **only the north hemisphere's** min/max (`:94-98` — `minMaxSh` is computed but the `std::min`/`std::max` calls both use `minMaxNh`, a copy-paste bug) and writes a horizontally concatenated `2·dim × dim` grayscale PNG (`nh | sh`) to `argv[2]` (`:99-109`).
4. If a 3rd output is given: `sqMp.toLambert()` (Legendre→Lambert, `master.hpp:423-473`), then for each pixel of a `dim × dim` canvas with `X,Y ∈ [-1,1]` (Y negated for image convention) computes the stereographic-inverse direction `n = (2X, 2Y, 1-h²)/(h²+1)` for `h² = X²+Y² <= 1`, square-Lambert-projects it, bilinearly interpolates, and emits a **gray+alpha** (2-channel) PNG, again `nh | sh` concatenated → `2·dim × dim` (`:115-175`).
5. Reopens the same file as a raw `sht::File` and pretty-prints **everything**: file version, software version, modality, beam energy, both angles, reserved param, notes, doi; then `numXtal`, `sgEff`, rotation sense + pijk, `simMetaSize` + vendor + modality; per crystal: sg/setting, axis/cell choice, origin shift, `abc`, `abg`, `rot` quaternion, weight, formula/name/structure-symbol/refs/note strings, atom list (`Z: x/24 y/24 z/24 occ charge DW`); and the full `EMsoftED` block including `latGridType` decoded as "square lambert"/"square legendre" (`:179-274`).

This program is the best available spec-by-example for a Python `.sht` reader — mirroring its printout is an excellent unit test.

### 6.2 `xtal::Diagram` (`include/xtal/diagram.hpp`, 620 lines)

- Projections: `Type::Stereo` (conformal) or `Type::Lambert` (equal-area) (`:47-50`).
- Constructors: empty (`:55`); from a `PointGroup` (`:60`); **from a `MasterPattern`** (`:66`, impl `:180-219`) — builds a `dim × dim` gray+alpha `svg::Image` per hemisphere by inverse-projecting each canvas pixel, square-Lambert-projecting the direction, and bilinearly interpolating the master pattern, rescaled to 8-bit over the joint min/max of both hemispheres; then overlays the point-group symmetry at 0.67 scale; from an arbitrary colour function (`:73`, impl `:227-267`) producing RGBA; and `IpfLegend(pg, t)` static factory that binds `PointGroup::ipfColor` with `sph2rgb` (`:272-275`).
- Drawing API: `addRotor(x,y,z,fld,scl)` (fld ∈ {2,3,4,6,-1,-2,-3,-4,-6}), `addMirror(x,y,z)`, `addInversion(scl)`, `add(PointGroup, scl)`, `addDot(n, color, scl)`, `addLabel(str, scl)`, `setColor(r,g,b)` (`:84-119`).
- Output: `getHemi(nth).write(fileName)` → an SVG file. Also contains `CatmullRom2Bezier` for smooth great-circle (mirror-trace) rendering (`:155`).

### 6.3 `include/util/svg.hpp` (1353 lines)

A self-contained SVG DOM: `SVG` canvas with `write(fileName)` (`:65`); abstract `Element`; `Group`; `Transform` (translate/rotate/scale, `:171-208`); `Color` (`:228`), `Stroke` (`:263`), `Fill` (`:305`); shapes `Path` (with control points), `Rect`, `Circle`, `Ellipse`, `Line`, `Polyline`, `Polygon`, `Text` (`:424-640`); and `Image` (`:340-362`) with `PixelType` ∈ {Gray=1, GrayAlpha=2, RGB=3, RGBA=4}.

`Image::write` (`:1007-1026`) is the notable one: it PNG-encodes the raw buffer in memory with miniz and embeds it as `xlink:href="data:image/png;base64,…"` using `base64::encode` from `include/util/base64.hpp` (`encode` `:46-79`, `decode` `:86-…`, standard alphabet `A-Za-z0-9+/` with `=` padding). So a `Diagram` SVG is a single self-contained file with the master-pattern projection embedded as a base64 PNG.

`include/util/bmp.hpp` (580 lines) is only used to read `.bmp` pattern stacks (`PatternFile::FromImages`); it parses BITMAPFILEHEADER + BITMAPINFO/V4/V5 headers (`Header` struct `:43-117`) and `readImage(is, buf, gry)` handles 8/24-bit.

`include/util/image.hpp` (719 lines) supplies `image::BiPix<Real>` (bilinear coefficients + `interpolate`), `image::Rescaler` (FFT-based resampling), `image::to8Bit`, and `image::ImageQualityCalc<Real>` (the IQ metric used in the output maps).

---

## 7. `ShtWisdom` — FFTW wisdom (irrelevant to a Python port)

`programs/sht_wisdom.cpp`:
```
ShtWisdom <bandWidth>      # "some reasonable values are 63, 95, 158, 263"
```
1. For every ring `y ∈ [0, bw/2+2)` of the largest spherical grid, plan a 1-D real FFT of length `max(1, 8y)` with `FFTW_PATIENT` (`:51-56`).
2. For each fast bandwidth in the hard-coded table `{25,32,38,41,53,63,68,74,88,95,113,122,123,158,172,188,203,221,263,284,313,338,365,368,438,473,515}` up to `bw`, plan a `SepRealFFT3D` with `FFTW_PATIENT` (`:60-68`).

It produces **no data file of its own** — `include/util/fft.hpp` accumulates FFTW wisdom and persists it to a system location (on this machine `C:\ProgramData\fftw.wisdom`; the benchmark logs all end with `failed to write wisdom to C:\ProgramData\fftw.wisdom`, a permissions issue that is harmless). The docs (`emsphinxebsd.rst:42-50`) expose the same thing as Tools ▸ Clear/Build/Import/Export Wisdom and note wisdom is hardware-specific.

**Verdict for the Python port: irrelevant.** NumPy/SciPy/`pyfftw` have their own planning; if you use `pyfftw` you could optionally expose `pyfftw.export_wisdom()`/`import_wisdom()`, but with `numpy.fft`/`scipy.fft` there is nothing to do. The only thing worth carrying across is the **"fast bandwidth" table** above (and `scripts/gen_bw.cpp`, which generates it: all 7-smooth numbers ≤2000 that are odd, mapped to `bw = (n+1)/2 ≥ 25`, so that `2·bw-1` is a product of {2,3,5,7}).

---

## 8. `data/Ni {20kV 75.7deg}.sht`

74 828 bytes. Fully parsed (live) — this is the exact reference target for a Python `.sht` reader:

```
magic            b'*sht'          (little endian)
fileVersion      1.1
resBytes         [0, 0]
softwareVersion  'e49ad6b'        (stored as "ve49ad6b")
modality         1 (EBSD)
beamEnergy       20.0 keV
primaryAngle     75.69999694824219 deg     ← the "75.7deg" in the filename
secondaryAngle   0.0        reservedParam 0.0
doiLen 46  noteLen 19
doi              "https://doi.org/10.1016/j.ultramic.2019.112841"   (padded to 48 B)
notes            "created with mp2sht"                              (padded to 24 B)
--- MasterPatternData @ offset 112 ---
numXtal 1   sgEff 225 (Fm-3m)   pijk +1   rotSense 'p'   modality 1   vendor 1 (EMsoft)   simMetaSize 88
--- CrystalData ---
sgNum 225  sgSet 1  sgAxis 1  sgCell 1   originShift (0,0,0)
lat  a=b=c=0.35236 nm, alpha=beta=gamma=90
rot  (1,0,0,0)   weight 1.0
numAtoms 1   strlens (form=2, name=0, symb=0, refs=0, note=0)
atom0: Z=28 (Ni), frac (0,0,0), occ 1.0, charge 0.0, DW 0.0035 nm^2
form "Ni"; name/symb/refs/note all empty
--- EMsoftED @ offset 232 ---
emsoftVersion '5_0_0_0'
sigStart 75.7   sigEnd NaN   sigStep NaN   omega 0.0
keV 20.0   eHistMin 5.0   eBinSize 1.0
depthMax 100.0 nm   depthStep 1.0 nm   thickness +inf
totNumEl 2 000 000 000   numSx 501
c1 4.0  c2 8.0  c3 50.0  sigDbDiff 1.0  dMin 0.05 nm
numPx 500   latGridType 1 (square Lambert)
--- HarmonicsData @ offset 320 ---
bw 384   zRot 4   cmpFlg 0x7 (inversion | equatorial mirror | +y mirror)   doubCnt 9312
harmonics: 9312 float64 = 74 496 B, offsets 328 .. 74 823
--- CRC @ 74824 --- 0xf2af93ef
```
Sanity check on the compression: sg 225 → `SpaceGroupRot[224] = 4`, `SpaceGroupCmp[224] = 0x7` — matches the LUTs (`sht_file.in.hpp:1847`, `:1867`). Dense storage would be `384·384 = 147 456` complex doubles ≈ 2.36 MB; the file stores 9 312 real doubles — a **~253×** reduction from the m3m symmetry.

**Role**: it is the tutorial/regression master pattern. `documentation/emsphinxebsd.rst:490-524` ("Example Data") names it explicitly as the master pattern for the Hikari Ni scan sequence used in the indexing paper, with the suggested wizard walk-through: pattern file `HikariNiSequence.h5` → `Scan 10`; master `Ni {20kV 75.7deg}.sht`; Binning 1, Binned Pixel Size 475 µm; default scan geometry; **bandwidth 53** with refinement on. Data links: full 10-scan sequence (~600 MB) `https://kilthub.cmu.edu/ndownloader/files/14503052`; scan-10-only (~80 MB) `http://vbff.materials.cmu.edu/wp-content/uploads/2019/10/Hikari_Scan10.zip`.

There are no other `.sht` files in the repo, and `test/` contains no `.sht` fixtures (unit tests are `test/{diagram,dict}.cpp`, `test/sht/{sht_xcorr,square_sht,wigner}.cpp`, `test/util/*`, `test/xtal/*`).

---

## 9. `benchmarks/` — reference data for regression tests

16 files. Two runs of the **same** dataset (a 407×368 EDAX `.up1` scan of BCC Fe at 12 kV) differing only in `backend = 'CPU'` vs `'CUDA'`. Produced by a `feature/GPU` build (`Commit Hash 98c251a`), 2025-03-23.

### 9.1 Inputs (both `.nml` are byte-identical apart from `backend` and the output paths)

```
patfile    = 'C:\Users\westraadt.1\Desktop\EMSphInx\map20240214000909769.up1'   ← NOT in the repo
masterfile = 'C:\Users\westraadt.1\Desktop\EMSphInx\Fe_bcc-master-12kV.sht'     ← NOT in the repo
patdims    = 96, 96
circmask   = 0            (largest inscribed circle)
gausbckg   = .FALSE.
nregions   = 4
delta      = 333          (µm; 96 × 333 = 31.97 mm detector)
pctr       = -1.92, 0.48, 17902.1     vendor = 'EMsoft'
thetac     = 8
scandims   = 407, 368, 1, 1
roimask    = '63, 21, 83, 69'         → 83 × 69 = 5727 indexed points of 149 776
bw         = 55                       (side length 2·55-1 = 109 → padded to 110)
normed     = .TRUE.       refine = .TRUE.
backend    = 'CPU' | 'CUDA'   gpudevice = 0      ← feature/GPU only
nthread    = 0            batchsize = 30
datafile   = benchmarks\GPU_test_{cpu,cuda}.h5
vendorfile = benchmarks\GPU_test_{cpu,cuda}.ang
ipfmap     = benchmarks\GPU_test_{cpu,cuda}_IPF.tiff     (actually PNG)
qualmap    = benchmarks\GPU_test_{cpu,cuda}_CI.tiff      (actually PNG)
```
**Blocker for reuse as-is**: neither the `.up1` patterns nor the `Fe_bcc-master-12kV.sht` master pattern is in the repo (absolute Desktop paths). The `.ang`/`.h5` outputs *are* committed and can serve as golden files once the inputs are located, but a self-contained regression test would need to be rebuilt around `data/Ni {20kV 75.7deg}.sht` + the Hikari Ni scan.

### 9.2 Log contents (UTF-16LE; 99.5 % of the CUDA logs is repeated progress spam)

Derived geometry echoed by `IndexEBSD` (identical in both):
```
Sample Tilt           : 70 degrees          ← read from the .sht primaryAngle
Scintillator Distance : 17902.1 microns
Camera Tilt           : 8 degrees
Camera                : 96 x 96 with 333 micron pixels
Pattern Center        : -1.92, 0.48 fractional pixels
Circular Mask         : true
Vertical Flip         : true                ← EDAX up1
Scan Dimensions       : 407 x 368 ; Resolution 1 x 1 micron
Pattern bitdepth      : 8 ; Total Patterns : 149776
AHE grid points       : 4
Point Group           : m3m ; Z Rotational Symmetry : 4 ; Equatorial Mirror : yes
Bandwidth 55 ; Side Length 110 ; ROI Mask 63, 21, 83, 69 ; Batch Size 30
```

Timing results (5727 patterns indexed each time):

| Log | Backend | Threads | Wall | Rate |
|---|---|---|---|---|
| `GPU_test_cpu.log` | CPU | 20 | **8.7 s** | **655.1 pat/s** |
| `GPU_test_cuda.log` | CUDA | — | 58.0 s | 98.7 pat/s |
| `GPU_test_cuda_timed.log` | CUDA | — | 57.5 s | 99.6 pat/s |
| `GPU_test_cuda_refine_opt.log` | CUDA | — | 56.5 s | 101.3 pat/s |
| `GPU_test_cuda_refine_opt2.log` | CUDA | — | 60.1 s | 95.3 pat/s |
| `GPU_test_cuda_refine_final.log` | CUDA | — | 61.4 s | 93.3 pat/s |

(i.e. the CUDA backend on this machine was **~6.6× slower** than the 20-thread CPU path.)

CUDA stage breakdown, e.g. `GPU_test_cuda_refine_final.log` (7 740 patterns / 258 batches timed):
```
Upload      0.009 s ( 0.0%)   Preprocess  0.352 s ( 0.6%)   IQ          0.480 s ( 0.8%)
Backproject 0.543 s ( 0.9%)   SHT        12.249 s (20.2%)   Coarse Corr 17.316 s (28.6%)
Refine     29.549 s (48.8%)   Pseudo Sym  0.000 s ( 0.0%)   Ranking     0.031 s ( 0.1%)
Output      0.038 s ( 0.1%)   Other/Host  0.828 s ( 1.3%)
```
Across all four CUDA logs the split is stable: **SHT 18–20 %, coarse correlation 29–30 %, Newton refinement 49–50 %**. That is the single most useful number for planning a Python port: *refinement dominates*, coarse correlation is next, and the forward SHT is a fifth of the budget; pattern preprocessing, back-projection and IQ together are under 2.5 %.

### 9.3 Golden outputs

- `GPU_test_cpu.ang` / `GPU_test_cuda.ang` — 11 234 238 B each, 149 810 lines (34 header + 149 776 data). They **differ** (first difference at byte 646 794, line 8 645) — i.e. CPU vs CUDA results are not bit-identical, so a Python regression test must compare with a tolerance (e.g. misorientation < some ε and CI within a few %), not exact equality.
- `GPU_test_cpu.h5` (3 948 848 B) / `GPU_test_cuda.h5` (3 954 784 B) — structure dumped in §4.1.
- `*_IPF.tiff` (12 998 / 14 593 B) and `*_CI.tiff` (5 034 / 1 877 B) — PNG-encoded 407×368 RGB / grayscale.
- Header values recoverable from the `.ang` for cross-checking a Python pattern-centre conversion: `x-star 0.480000`, `y-star 0.505000`, `z-star 0.560001`, `TEM_PIXperUM 1.000000`, `WorkingDistance 0.000000`, `Symmetry 43`, `LatticeConstants 2.866 2.866 2.866 90 90 90`, `GRID: SqrGrid`, `XSTEP/YSTEP 1.000000`, `NCOLS_ODD/EVEN 407`, `NROWS 368`, `OPERATOR: unknown`. Verify against the formulas in §3.5: `xStar = (cX + w/2)/w = (-1.92+48)/96 = 0.48` ✓; `yStar = (cY + h/2)/w = (0.48+48)/96 = 0.505` ✓; `zStar = sDst/(pX·w) = 17902.1/(333·96) = 0.56000...` ✓.

---

## 10. What `documentation/*.rst` says that a kikuchipy tutorial notebook should mirror

`documentation/` contains only `index.rst` (81 lines), `emsphinxebsd.rst` (525 lines), `license.rst`, plus Sphinx `conf.py`/`Makefile`/`make.bat` and `images/emsphinxebsd/*.png`. There is **no CLI documentation at all** — `ReadMe.md:75` explicitly says "The functionality of the commandline programs is mostly equivalent to the GUI and instructions are printed by running the program with no arguments." So the RST describes the GUI wizard, and that wizard's 6-panel flow is exactly the skeleton for a notebook.

**Workflow to mirror (from `emsphinxebsd.rst`), in order:**

1. **(Optional, once) Convert master pattern** — `Tools ▸ Convert Master Pattern...` (`:46`) = `mp2sht` = EMsoft `.h5` → `.sht`. In kikuchipy this is "read an EMsoft/EMsoftED master pattern with `kp.load`, energy-average, project to the sphere, compute the SHT, (optionally) write `.sht`".
2. **(Optional, once) Build wisdom** — `Tools ▸ Build Wisdom...` (`:43`) — skip in Python (§7). Note the doc's warning that first-run indexer initialisation "may take several minutes" while DFTs are planned (`:14`).
3. **Panel 1 — Experimental Pattern Selection** (`:220-273`): choose the pattern file. Supported: HDF5 (`.h5/.hdf/.hdf5`), EDAX (`.up1/.up2`), Oxford (`.ebsp`), EMsoft raw (`.data`). Dimensions/bitdepth/count auto-detected for all but `.data`. Companion-file auto-discovery: an h5 EBSD scan file, or a `.ang` beside a `.up1/.up2`, or a `.ctf` beside a `.ebsp` (`:236-239`, code at `nml.hpp:496-543`) — from which it prefills **pattern centre, detector tilt, scan dimensions, scan pixel size, and IQ/CI maps for ROI drawing** (`:243-247`). Image-processing preview: N evenly spaced patterns, raw on the left, processed on the right, live parameter tweaking (`:258-269`); optional "Compute Image Quality Map for ROI Selection".
4. **Panel 2 — Master Pattern Selection** (`:275-320`): ordered list of `.sht` files; list index = output phase index. A persistent "library" browsable/sortable by File / Formula / Name / S.Syb / kV / Tilt / Laue / SG#, filterable by kV, tilt, SG# and composition. *(A notebook analogue: a small pandas table over a folder of `.sht` headers — trivial once you have a header parser from §2.2.)*
5. **Panel 3 — Detector Geometry** (`:322-383`): the **binning ↔ pixel size ↔ detector width** triangle (`:329-368`) — this is the single most confusing part for users and deserves an explicit notebook cell. Worked example: 640×480 detector, 50 µm pixels, 4×4 binning ⇒ pattern 160×120, effective pixel 200 µm, detector width 32 mm; the table at `:362-368` shows the four equivalent parameterisations. Then the pattern centre with the vendor conversion table (`:123-131`):

   | | EMsoft | Bruker | EDAX | Oxford |
   |---|---|---|---|---|
   | `pctr.x` | pixels | detector widths | detector widths | detector widths |
   | `pctr.y` | pixels | detector widths | detector widths | detector **heights** |
   | `pctr.z` | microns | detector **heights** | detector widths | detector widths |
   | origin | centre | top left | bottom left | bottom left |

   References: EMsoft tutorial paper https://doi.org/10.1184/R1/7792505 and forward-model paper https://doi.org/10.1017/S1431927613001840, plus the vendor-convention paper https://doi.org/10.1007/s40192-019-00137-4 (cited in the generated nml, `nml.hpp:381`).
6. **Panel 4 — Scan Geometry** (`:385-439`): scan `w,h,dx,dy`, and interactive ROI drawing over an IQ/CI map with rectangle / ellipse / polygon, invert flag, coverage percentage. The ROI string grammar (`:145-164`) is in §3.5 — a notebook can use a matplotlib/ipywidgets selector, or just accept the same string.
7. **Panel 5 — Indexing Parameters** (`:166-193`, `:441-469`): bandwidth (cost scales as `bw³·ln(bw³)`, grid is `(2bw-1)³`; use "fast" sizes), with the recommended tiers spelled out — **53, 63, 68, 74** "fast but somewhat noise sensitive"; **88, 95, 113, 123** "trade-off"; **158, 172, 203, 221, 263** "maximum noise robustness but slow" (`:177-179`); `normed` ("slightly slower but suggested when pseudo-symmetry is anticipated or to index against multiple phases", `:181-184`); `refine` ("Newton's method … otherwise a sub-pixel maximum is interpolated from the 3×3×3 box surrounding the maximum in the Euler angle grid", `:466-469`); `nthread` ("performance peaks at ~1.5× the number of real cores", `:187-189`); `batchsize` (`:190-193`).
8. **Panel 6 — Summary** and outputs (`:195-204`, `:471-488`): `datafile` (required h5), `vendorfile` (`.ang`/`.ctf`, optional), `ipfmap` (Z-reference IPF PNG), `qualmap` (normalized spherical cross-correlation PNG).
9. **Example data section** (`:490-524`) — reproduce the exact Ni tutorial with the shipped `data/Ni {20kV 75.7deg}.sht`, `Hikari_Scan10.zip`, binning 1 / 475 µm pixels, bw 53, refinement on. **This is the obvious anchor for a kikuchipy tutorial notebook** since both the master pattern and the experimental data are public, and the resulting orientation map can be visually and statistically compared against the published indexing paper.
10. Citations to surface in the notebook: EBSD indexing https://doi.org/10.1016/j.ultramic.2019.112841 (also baked into every `mp2sht` `.sht` as the DOI field) and pseudo-symmetry prediction https://doi.org/10.1107/S1600576719011233 (`index.rst:7-8`, `ReadMe.md:6-7`); AHE https://doi.org/10.1016/S0734-189X(87)80186-X (`:99`). Licence: GPL-2.0, with a note in `ReadMe.md:10` that "The central indexing algorithm is covered by a provisional patent application" — worth flagging before shipping a reimplementation.

---

## Appendix A — Recommended Python port ordering (derived from the above)

| Priority | Piece | Source of truth | Notes |
|---|---|---|---|
| 1 | `.sht` reader (header + crystal + sim + unpack harmonics + CRC verify) | `sht_file.in.hpp:138-637`, `:1672-1871`, `:2010-2031` | Golden file: `data/Ni {20kV 75.7deg}.sht` (§8). Copy the two 230-entry LUTs and the CRC-32C LUT verbatim. |
| 2 | `.sht` writer + EMsoft-h5 → `.sht` (`mp2sht`) | `mp2sht.cpp`, `master.hpp:242-347`, `:550-595`, `sht_file.in.hpp:2047-2230` | Watch: atom-sum is unweighted; energy weights from `accum_e`; 24ths encoding of atom coords; pad-to-8 strings. |
| 3 | Pattern readers: `up1/up2`, EMsoft/EDAX/Bruker/Oxford h5, `.data`, `.ebsp` | `pattern.hpp:381-554`, `:754-913` | kikuchipy already covers most; `.ebsp` version-4 handling and the offset-permutation logic are the parts worth transcribing. **Add `.h5oina`/`.oh5` — EMSphInx has no reader.** |
| 4 | Namelist parse/emit (`nml`) | `util/nml.hpp`, `modality/ebsd/nml.hpp` | For a notebook a dataclass + optional `.nml` I/O for interop with the C++ binaries. |
| 5 | ROI mask | `idx/roi.h:435-505` | 3 shapes, invert flag, half-open rectangle, winding-number polygon. |
| 6 | Detector geometry / pattern-centre conversions | `detector.hpp:250-279`, `idx.hpp:244-248` | 4 vendor conventions both directions. |
| 7 | Output writers: h5, `.ang`, `.ctf`, IPF/quality PNG | `orientation_map.hpp:584-700`, `tsl.hpp`, `hkl.hpp`, `idx.hpp:318-370` | Regression against `benchmarks/GPU_test_cpu.{ang,h5}` (with tolerance — CPU and CUDA runs already differ). |
| 8 | `sht2png`-equivalent visualisation | `sht2png.cpp`, `diagram.hpp`, `svg.hpp` | In Python: `synthesize` to a Legendre/Lambert grid + matplotlib; `Diagram`/`svg.hpp` can be replaced with matplotlib stereograms + orix symmetry markers. |
| 9 | `MasterXcorr`-equivalent pseudo-symmetry | `master_xcorr.cpp` + `sht/sht_xcorr.hpp` | Note the two bugs identified in §5 (usage string, dead `addDot` branch). Output = HDF5 volume + XDMF + 2 SVGs + a quaternion list that feeds `psymfile`. |
| — | `ShtWisdom` | — | **Skip entirely** (§7); keep only the fast-bandwidth table. |
| — | `PatternRepack`, `EBSPDims` | `pattern_repack.cpp`, `ebsp_dims.cpp` | Both are ~20-line numpy/h5py scripts; useful as notebook utilities. |
| — | `EMSphInxEBSD` GUI | `ebsd_wizard.cpp` + `include/wx/*` | **Skip** — replaced by the notebook (§10). |
