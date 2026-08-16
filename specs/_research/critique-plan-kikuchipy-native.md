# Adversarial review of the EMSphInx→kikuchipy plan

All findings below were verified by reading the files. Verified-correct claims are listed at the end.

---

## BLOCKERS

### B1. The ZYZ→Bunge Euler mapping is wrong. Every indexed orientation would be incorrect.
**Evidence.** Plan §C.6 and `specs/tech-stack.md` state: *"correlator ZYZ (α,β,γ) → Bunge (α−π/2, β, γ+π/2); final orientation = `~Rotation.from_euler(bunge)`"* and claim orix `from_euler` ≡ EMSphInx `eu2qu` was *"verified bit-identical"*.

The `eu2qu` ≡ `from_euler` half is true. The ZYZ offset half is **false**. Numerically (3000 random ZYZ triples, comparing `zyz2qu(z)` from `EMSphInx/include/xtal/rotations.hpp:973-989` against `eu2qu(...)` from `rotations.hpp:409-429`, taking the min over ±q):

```
PLAN mapping (α−π/2, β, γ+π/2)  max err = 1.386      <-- wrong
CORRECT      (α+π/2, β, γ−π/2)  max err = 8.9e-16    <-- right
```

Worked case `zyz=(0, π/2, 0)`: `zyz2qu` → `(0.7071, 0, −0.7071, 0)`; plan's route → `(0.7071, 0, +0.7071, 0)`. Opposite 90° rotations about **y**.

Root cause: `eu2qu` uses `delta = (eu[0]−eu[2])/2` while `zyz2qu` uses `delta = (eu[2]−eu[0])/2` — the operand order is **reversed** (`rotations.hpp:411-412` vs `:976-977`). The plan copied `zyz2eu` at `rotations.hpp:1025-1029`, which carries the `eu2zyz` offsets (an EMSphInx bug); its `//@note : equivalent to eu[0] -= pi/2 ...` comment at `:971` is also wrong. EMSphInx itself is unaffected because `indexer.hpp:266` calls `xtal::zyz2qu` **directly** and never routes through `zyz2eu` — only the port would be wrong.

**Fix.** `_rotation_from_zyz`: `eu = (α + π/2, β, γ − π/2)`, then `Rotation.from_euler(eu)`, then conjugate (`indexer.hpp:267-268`). Add a REQ in `specs/2026-08-22-*/requirements.md`: a Python transcription of both `zyz2qu` and `eu2qu` must agree to 1e-14 over ≥1000 random triples *before* `_correlator.py` is written. Also add a `_zyz_from_rotation`/`_rotation_from_zyz` round-trip test at β≈0 and β≈π (where `qu2zyz`'s `chi <= thr` branches at `:445-455` kick in).

### B2. `scipy.fft.next_fast_len(n, real=True)` is not EMSphInx `fastSize`. The published "fast bandwidth" table is wrong for 7 of 17 entries.
**Evidence.** EMSphInx `fastSize` (`util/fft.hpp:438-491`) admits factors {2,3,5,7,11,13}. scipy's pocketfft real transforms admit only {2,3,5}. Measured on scipy 1.17.1:

| bw | 2bw−1 | `real=True` | `real=False` |
|---|---|---|---|
| 53 | 105 | **108** | 105 |
| 74 | 147 | **150** | 147 |
| 88 | 175 | **180** | 175 |
| 95 | 189 | **192** | 189 |
| 123 | 245 | **250** | 245 |
| 158 | 315 | **320** | 315 |
| 172 | 343 | **360** | 343 |
| 221 | 441 | **450** | 441 |
| 263 | 525 | **540** | 525 |
| 284 | 567 | **576** | 567 |

This invalidates, as written: the tech-stack fast-size list; §C.3's `slP = next_fast_len(2*bw−1, real=True)` plus its "warn if `slP != 2*bw−1`"; the §C.5 memory table (bw=88: `fxc` 43.1 → 46.8 MB); F4's `padded sizes {54..64}` and the `< 360/(2·slP)·1.5` unrefined tolerance; and F7's golden comparison at **bw=53 and bw=68** — bw 53 is exactly the EMSphInx tutorial default and the plan's own `nickel_ebsd_large` reference bandwidth, where EMSphInx uses slP=105 and the port would use 108. Different coarse peak grid ⇒ different `refine=.FALSE.` results and different `interpPeak` starts.

**Fix.** Decide and record in `specs/2026-08-22-*/plan.md`: the cube is `slP³` so the real axis binds. Either (a) benchmark `real=False` sizes (7/11-smooth) and accept pocketfft's Bluestein/generic path on the `m` axis, or (b) accept 5-smooth `slP` and regenerate the fast-size table from `next_fast_len(2*bw−1, real=True) == 2*bw−1`. Then recompute the memory table, and for F7 pin the golden run to a bandwidth where both agree (63, 68, 113, 122, 188, 203, 313) — bw **68** works for both, bw **53 does not**.

### B3. Wrong licence attributed to the `.sht` codec — SHTfile is BSD-3-Clause, not GPL-2.0-or-later.
**Evidence.** `EMSphInx/build/_deps/shtfile-src/sht_file.in.hpp:1-34` and `.../LICENSE:1-4`:
> `Copyright (c) 2019, De Graef Group, Carnegie Mellon University` / `Author William C. Lenthe` / `Redistribution and use in source and binary forms ... 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. ... 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products...`

The plan's tech-stack mandates *"in **every** ported file: ... GPL-2.0-or-later; ... relicensed under GPL-3.0-or-later"*. Applied to `_sht_file.py` — whose content (CRC-32C LUT `sht_file.in.hpp:967-1000`, `SpaceGroupRot`/`SpaceGroupCmp` LUTs `:1838-1869`, `PackHarm`/`UnpackHarm` `:1706-1831`, all explicitly "copied verbatim" per F2) derives from SHTfile — this **misstates the licence**, **omits the BSD-3 conditions list and warranty disclaimer that clause 1 requires be retained**, and asserts a relicensing that BSD-3 does not authorise in that form.

This is precisely what kikuchipy reviewers are instructed to check (`doc/dev/licensing_considerations.rst`, absent locally — see G5). It also needlessly forecloses an option: BSD-3 code *may* live in kikuchipy's permissive area, whereas GPL-derived code may not.

**Fix.** Split the attribution policy in `specs/tech-stack.md`:
- `_sht_file.py` (and any SHTfile-derived LUT): reproduce the SHTfile BSD-3 notice + conditions + disclaimer verbatim in the module docstring; do **not** claim GPL relicensing; flag to maintainers that this module could be BSD-3-licensed.
- `_square_grid.py`, `_sht.py`, `_wigner.py`, `_correlator.py`, `_back_projection.py`, `_harmonics.py`, `_indexer.py`, `_pseudo_symmetry.py` (from `EMSphInx/include/**`, headers at e.g. `include/sht/square_sht.hpp:1-33`): GPL-2.0-or-later → GPL-3.0-or-later as planned, plus the GPLv2 §2(a) "changed files + date" notice.
- Note `include/miniz/miniz.c:3-25` is **MIT**, not BSD-3 as `EMSphInx/ReadMe.md` claims — do not repeat that error if miniz is ever touched.

---

## MAJOR

### M1. "32-entry dicts keyed on `Symmetry.name`" will KeyError or silently produce wrong `fNf`/`fMr`.
**Evidence.** orix 0.14.2 exposes **38** point-group objects, all with distinct names:
`['1','-1','211','121','112','m11','1m1','11m','2/m','222','mm2','mmm','4','-4','4/m','422','4mm','-42m','4/mmm','3','-3','321','312','32','3m','-3m','6','-6','6/m','622','6mm','-6m2','6/mmm','23','m-3','432','-43m','m-3m']`

These are **setting-specific**, and the setting is exactly what `zRot()`/`zMirror()` depend on: `112` has z-rotation order 2 while `211`/`121` have 1; `11m` has an equatorial mirror while `m11`/`1m1` do not. A 32-entry dict misses six names and, worse, would give the wrong systematic-zero mask (`m % fNf`, `(m+j) % 2` — `sht_xcorr.hpp:701-705`) for any non-standard setting, corrupting the correlation with no error raised.

Related: F4's test list `{112, 11m, 112/m, 3, 4, 4/m, 6, 6/m, m-3m}` names **`112/m`, which does not exist in orix** (the object is `2/m`). Not executable as written.

**Fix.** Build 38 entries keyed on `Symmetry.name`, and make the cross-check in `specs/2026-08-18-*/validation.md` normative rather than advisory: for every group assert `_z_rotation_order(s)` equals the order of the highest-order rotation axis parallel to `Vector3d.zvector()` derived from `s.get_axis_orders()`, and `_has_equatorial_mirror(s)` equals the existence of an improper diad with normal ‖ z. Rewrite F4's list with orix names.

### M2. Memory table under-counts peak per-thread usage by roughly 2×.
**Evidence.** §C.5 counts `fxc` **or** the real cube as separate line items and sums them. But `scipy.fft.irfftn(fxc, s=(slP,)*3)` simultaneously holds: the `(slP, slP, bwP)` complex128 input (91.5 MB at bw=113), pocketfft's intermediate complex workspace for the k/n axes (~same again), and the `slP³` float64 output (91.1 MB) — and §C.3 then makes a **contiguous copy** of `xc_full[:bwP]` (another 45.8 MB) rather than a view. `overwrite_x=True` is a hint, not a guarantee, and is not honoured when the output dtype/shape differ.

Realistic bw=113 peak is ~550-600 MB/thread, so the quoted "8 dask threads ≈ 2.3 GB" is closer to **4.5-5 GB** — enough to OOM a 16 GB laptop once the master harmonics, `rDen`, sphere buffers and dask's own graph are added.

**Fix.** Measure peak RSS per thread with `tracemalloc`/`memory_profiler` for bw ∈ {63, 88, 113} in F4's `validation.md` and replace the table with measurements. Keep `xc = xc_full[:bwP]` as a **view** (it is already C-contiguous as a leading-axis slice) and state that explicitly. Add a documented guard in `SphericalIndexer.__init__` that warns when `bandwidth³ × n_threads` exceeds available memory.

### M3. Tutorial computes bw=384 harmonics from a master pattern that cannot support it.
**Evidence.** F8 step 4 does `mp = kp.data.nickel_ebsd_master_pattern_small(...)` then `h = mp.get_spherical_harmonics(bandwidth=384)`. Verified: that dataset is `(2, 401, 401)` **uint8**, single energy. A 401² square-Lambert grid supports Lambert bandwidth `(dim−1)/2 = 200` (`square_sht.hpp:112`, `:344`). `MasterSpectra` at bw=384 first upsamples to `round(√2·387) = 547` (`master.hpp:383`) then regrids to a 387 Legendre grid — so coefficients for `l` ≈ 200-383 are pure interpolation artefact, compounded by 8-bit quantisation. The subsequent `h.resize(68).to_master_pattern()` "truncation demo" would then be demonstrating interpolation noise, not bandwidth truncation, in the flagship tutorial.

**Fix.** Either use `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both", energy=20)` (1001², float32 — already in the `develop` pooch cache, so no download for the author, `allow_download=True` for readers), or cap the tutorial at `bandwidth=190`. Add a `warnings.warn` in `from_master_pattern` when `bandwidth > (source_dim − 1) / 2`, and a unit test for that warning.

### M4. Score thresholds are asserted a priori for a metric whose scale the plan itself changes.
**Evidence.** D.3 and F5 assert `scores > 0.3`, `mean(scores) > 0.3`, and the benchmark asserts `xmap.scores.mean() > 0.3`. But the exploration report §3.10 establishes the metric is *not* a bounded NCC (`Indexer::sum2` is computed and never consumed, `indexer.hpp:315, 326-331`), and §9.2 of the second report measured EMSphInx CPU 0.32-0.74 vs CUDA **4.43-4.77** on the *same* data — the scale is not even stable across EMSphInx backends. The plan simultaneously changes normalisation (fixing bugs 8.7 mean-÷totW and 8.8 quartered corners), which shifts `a₀₀` and hence the denominator.

**Fix.** Make every score assertion a two-step: first PR records the measured value in `validation.md`, subsequent assertions use `pytest.approx(measured, rel=0.05)`. Assert *ordering* properties that are scale-free instead (true match > any pseudo-match; refined ≥ unrefined; monotone decrease with added noise) — this is the idiom kikuchipy already uses (`tests/test_indexing/test_ebsd_refinement.py:581-604`: `assert xmap_ref.scores.mean() > s.xmap.scores.mean()`).

### M5. `Orientation.reduce()` / `Misorientation.reduce()` against the pinned "oldest" CI job.
**Evidence.** `.github/workflows/tests.yml:48` pins `orix==0.12.1` for the oldest job. `reduce()` exists in orix 0.14.2 (verified) but `map_into_symmetry_reduced_zone` was only deprecated in 0.14 — i.e. `reduce()` is a later addition. F4, F5 and C.6 all use `.reduce()` in **tests**, which run on the oldest job.

**Fix.** Verify `reduce` in orix 0.12.1 before writing F4. If absent, either use `Orientation.angle_with(other, degrees=True)` (which applies symmetry, as `tests/test_signals/test_ebsd_hough_indexing.py:75-87` relies on) or bump `orix` in `pyproject.toml` and update the oldest pin — the latter needs maintainer agreement and belongs in the issue.

### M6. "PatternRepack and EBSPDims need no new code" is not supported.
**Evidence.**
- `EBSD.downsample` (`src/kikuchipy/signals/ebsd.py:1113`) documents *"rescale intensities to fill the data type range"* and *"Contrast between patterns is lost."* PatternRepack's `binAvg` (`pattern_repack.cpp:76-98`) is a plain in-type average with **no** rescaling; `binFloat` (`:50-67`) sums to float. Not equivalent.
- PatternRepack applies a **vertical flip** (`flip = true`, `:117`, `:215`). Nothing in the plan's `kp.load → downsample → save` chain does.
- kikuchipy has **no reader for EMsoft raw headerless `.data` float32** — verified plugin list: `bruker_h5ebsd, ebsd_directory, edax_binary, edax_h5ebsd, emsoft_ebsd, emsoft_*_master_pattern, kikuchipy_h5ebsd, nordif, nordif_calibration_patterns, oxford_binary, oxford_h5ebsd`. PatternRepack accepts `.data`.
- EBSPDims: `oxford_binary/_api.py:282-322` `get_navigation_shape_and_step_size` infers nrows/ncols from **first/last footers assuming a regular grid**; EBSPDims (`ebsp_dims.cpp:76-103`) collects the full `std::set` of x and y and explicitly reports when `sx.size()*sy.size() != numPat`. kikuchipy will silently return a wrong shape in exactly the irregular case EBSPDims exists to diagnose.

**Fix.** Since the user's brief requires these programs be *covered*, add to F8 a small `_spherical`-adjacent utility or a documented notebook recipe: `EBSD.rebin`/manual `reshape().mean()` for intensity-preserving binning, an explicit `data[..., ::-1, :]` flip note, and an `EBSPDims`-equivalent cell that reads all footers and prints the distinct x/y sets with the consistency check. Either document `.data` as unsupported or add a 20-line reader. State honestly in `specs/roadmap.md` that these are *partially* covered.

---

## MINOR

- **m1. Working tree is dirtier than stated.** The plan says only `IndexEBSD.nml` is stray. `git status` also shows modified `doc/tutorials/hybrid_indexing.ipynb` and `doc/tutorials/load_save_data.ipynb`. `git merge upstream/develop` will refuse or conflict. Add "stash/commit/discard the two modified notebooks" to Phase 0.
- **m2. `sht2png`'s header dump is dropped.** `sht2png.cpp:179-274` pretty-prints the entire header/crystal/simulation block including `latGridType` decoded. The plan's one-line `__repr__` doesn't cover it, and the report itself calls that printout *"the best available spec-by-example for a Python `.sht` reader"*. Add `MasterPatternHarmonics.print_file_info()` or make F2's test assert against the full parsed dict.
- **m3. No image quality in the result.** EMSphInx writes `iq` into every `.ang`/`.h5` (`idx.hpp:365`, `Result::iq`). `SphericalIndexer.index_patterns` returns only rotations/scores/phase. Add `"iq"` to `CrystalMap.prop` (populated from `EBSD.get_image_quality`), matching the real-data report's recommendation.
- **m4. Open question 6 is already answered by the repo.** `.github/workflows/tests.yml:48` pins `numba==0.57 numpy==1.23.0 orix==0.12.1` on Python 3.10. Delete that half of OQ6 and state the constraint as fact.
- **m5. Schedule.** Spec folders run 2026-08-16 → 2026-09-02 for a discrete SHT, Wigner-d tables with derivatives, a symmetry-reduced SO(3) correlator, analytic Newton refinement, a back-projection LUT, a binary `.sht` codec, pseudo-symmetry, a golden regression and a tutorial — the C++ originals are ~4000 lines of tuned code. Either drop the dates or mark them aspirational.
- **m6. `_resample_dct` normalisation unstated.** scipy's `idctn(type=2)` carries a `1/(2N)` factor tied to the **output** length, so truncate/pad changes amplitude; EMSphInx compensates explicitly (`master.hpp:365-367`, `0.5/nhScaled.size()`). Cancels for unit-std patterns but not for `to_master_pattern` intensities. Specify the factor and unit-test constant-image preservation.
- **m7. `_legendre_ring_weights` formula is ambiguous.** State it as `w_y = 4π · ŵ_y / max(1, 8y)` with `ŵ_y = w_leggauss,y / 2`, equator halved, pole 0, `Σŵ = 1` — and make the `Σŵ_y == 1` assertion (`square_sht.hpp:1058`) a hard test.
- **m8. Missing interop pieces:** no `.nml` reader/writer (F7's own generation scripts need one), no ROI-string equivalent for `roimask` (`idx/roi.h:565-627` grammar: rect / `e`llipse / polygon / `i`nvert), no `.ctf` writer (orix writes `.ang`/`.h5` only). Say explicitly in `specs/roadmap.md` that these are out of scope and why.
- **m9. `specs/` divergence.** Committing `specs/` to the fork's `develop` and excluding it upstream guarantees a conflict on every `merge upstream/develop`. Put specs on an orphan branch or a sibling repo.
- **m10. Sphere→detector direction is under-specified.** F5 says "`detector.sample_to_detector` + gnomonic bounds" but never states the projection step (rotate into the detector frame, divide by the frame z, then `convert_gnomonic_to_pixel_coords`), nor the row-order/flip convention. The plan *does* mandate a round-trip test against `_get_direction_cosines_from_detector`, which will catch it — make that test a REQ with a stated tolerance rather than a parenthetical.

---

## WHAT IS GOOD — KEEP

**Verified-correct facts.** These all checked out and should be kept as-is:
- `_get_direction_cosines_from_detector(detector, signal_mask=None) -> np.ndarray` exists at `src/kikuchipy/signals/util/_master_pattern.py:82` with exactly that signature. Reusing it as the sphere-direction source and as the back-projection oracle is the single best decision in the plan.
- `src/kikuchipy/indexing/__init__.pyi` exports exactly **9** names — count confirmed.
- `doc/dev/licensing_considerations.rst` is genuinely **absent** from the local clone (13 files in `doc/dev/`), so "add upstream, merge" as the first action is correct and necessary.
- `dim = bw + (2 if bw odd else 3)`; verified `dim−2 ≥ bw` for bw ∈ {16, 53, 68, 384}.
- `norm="forward"` on `scipy.fft.irfftn` is exactly the unnormalised FFTW-equivalent (`×slP³`) — a subtle detail the plan got right.
- `Rotation.from_euler(euler, direction='lab2crystal', degrees=False)` is the real orix 0.14.2 signature with that default; `eu2qu` ≡ `from_euler` confirmed.
- `scipy.special.sph_harm_y` exists in scipy 1.17.1 (and `sph_harm` is gone) — correctly chosen as the oracle.
- `original_metadata` of the packaged Ni master pattern has keys `['BetheList','CrystalData','EBSDMasterNameList','MCCLNameList']` — **no `EMData`**, so the plan's diagnosis that `accum_e` energy weights require a reader change is correct and the proposed backwards-compatible addition is the right shape.
- `nickel_ebsd_small` detector PC is `(3,3,3)` / nav `(3,3)`; `EBSDDetector.sample_to_detector` exists.
- `dictionary_indexing` ends at `ebsd.py:1984` — correct insertion point for `spherical_indexing`.
- `oxford_binary.get_navigation_shape_and_step_size` genuinely exists.
- orix `Symmetry.name` values are unique across all 38 groups — so name-keyed dicts are *viable* (just need 38 entries, per M1).

**Architecture and process.** Keep essentially unchanged:
- No new hard dependencies; `scipy.fft` over FFTW/pyfftw/shtns. The pyshtools analysis (§C.2) is honest and lands on the right answer for the right reason (the square-Legendre grid, m-major layout, `d(π/2)` layout and symmetry-reduced SO(3) correlation are all EMSphInx-specific).
- Private `_spherical/` package, `_`-prefixed modules, public names only via `lazy_loader.attach_stub` in `__init__.pyi` with sorted `__all__` — matches `indexing/_refinement/` precisely.
- dask `map_blocks` across patterns + `nogil` numba kernels, `parallel=False`, `scipy.fft(workers=1)`, thread-local correlator buffers, one-pattern-per-thread mirroring EMSphInx. Correct for kikuchipy and avoids oversubscription.
- Public surface (`EBSD.spherical_indexing`, `SphericalIndexer`, `MasterPatternHarmonics`, `EBSDDetector.get_spherical_indexer`, `find_pseudo_symmetry`) mirrors the Hough/DI/refinement API shape exactly.
- Cataloguing and **fixing** the EMSphInx bugs (8.7 mean ÷totW, 8.8 quartered corners, 8.9 south-hemisphere index, `interpPeak` `x[2]`) while documenting the divergence.
- Test strategy: deterministic `default_rng(0)` synthetic tests mirroring `test/sht/*.cpp`, real-data tests on shipped `nickel_ebsd_small` + `nickel_ebsd_master_pattern_small`, `@pytest.mark.weekly` for heavy work, `.py_func` variants for numba coverage, benchmark under `benchmarks/indexing/`, `CHANGELOG.rst` Unreleased→Added with PR link, `credits` + `.zenodo.json`, `git commit -s`, merge-not-rebase — every one of these matches documented kikuchipy convention.
- Rejecting the `benchmarks/GPU_test_*` files as a baseline (missing inputs, different branch, incompatible metric scale) and generating a fresh serial CPU reference instead.
- Escalating the CMU provisional-patent question to maintainers before any upstream merge, and refusing to place this code in a BSD-3 area.

**Correction priority:** B1 first (it silently poisons every downstream tolerance), then B2 (it fixes the size tables the specs are written against), then B3 (cheap, and it unblocks the F2 licence header). M1 and M4 should be folded into F2/F4 requirements before any code is written.
