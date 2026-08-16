ADVERSARIAL REVIEW — plan "port EMSphInx spherical indexing (CPU) into kikuchipy"

Everything below was checked against files. Verified-correct claims are listed at the end; do not re-litigate them.

================================================================
BLOCKERS
================================================================

B1. `slP`/`fast_bandwidths` use `next_fast_len(..., real=True)` — WRONG, and it silently breaks the EMSphInx regression tests.
Evidence: measured in the repo venv (scipy 1.17.1):
  bw  2bw-1  next_fast_len  next_fast_len(real=True)
  53   105        105              108
  55   109        110              120
  88   175        175              180
  123  245        245              250
  158  315        315              320
EMSphInx `fastSize` (EMSphInx/include/util/fft.hpp:438-491) allows {2,3,5,7,11,13}-smooth and imposes no even/real constraint, so it returns 105/110/175/245/315. Consequences:
 (a) plan section C's `slP = scipy.fft.next_fast_len(2*bw-1, real=True)` produces a *different correlation grid* than EMSphInx for bw 53, 55, 88, 123, 158 — i.e. most of the recommended list. The Feature 4 argument "both interpolate the same 3x3x3 neighbourhood" then fails outright.
 (b) `fast_bandwidths()` is defined in Feature 4 as `2*bw-1 == next_fast_len(2*bw-1, real=True)`, which *excludes* 53, 88, 123, 158 — contradicting the plan's own docstring guidance ("fast values 53, 63, 68, 74, 88, 95, 113, 122, 123, 158") three lines later.
 (c) the parenthetical "pocketfft's fast sizes are {2,3,5,7,11}-smooth" is only true for `real=False`.
Fix: drop `real=True`. Then implement EMSphInx `fastSize` verbatim anyway, because scipy omits radix 13 and will diverge whenever `2*bw-1` has a 13 factor (e.g. 2bw-1=767=13·59 → scipy 768, EMSphInx 768 by luck; not guaranteed in general). Make `fast_bandwidths` a filter over the EMSphInx rule, and add a test asserting `fast_size(n) == fastSize(n)` over n in [16, 1100].

B2. The ring-DFT-matrix design is infeasible at the master-pattern bandwidth (Feature 1/2 default path).
Evidence: plan section C says ring DFTs go "inside numba with precomputed per-ring cos/sin matrices `(Nt, 8y_max, m_max)` ... padded array is simplest for numba". At bw=384 → dim=387, Nt=194, 8*(Nt-1)=1544, m_max=384: 194·1544·384·8 B = 920 MB **per** matrix, ×2 (cos, sin) = 1.8 GB. Even a ragged/offset layout is Σ_y 8y·min(bw,4y+1) ≈ 5.8e7 complex entries ≈ 0.9 GB, and costs ~40× more flops than per-ring FFTs. Yet `MasterPatternHarmonics.from_master_pattern(..., bandwidth=384)` is on the *default* test suite (Feature 2 mp2sht parity) and in the tutorial (Feature 8 cell 4). The plan's own budget ("bw 384 ... ~2e8, sub-second") is off by more than an order of magnitude and ignores memory entirely.
Fix: split the implementation. Use the numba DFT-matrix kernel only for indexing bandwidths (bw ≤ ~128 → table ≈ 11 MB at bw 68, fine), and `scipy.fft.rfft` per ring for the master-pattern SHT. Note `scipy.fft` already releases the GIL, so the stated GIL rationale for the matrix approach does not apply to the master-pattern path (which runs once, not per pattern). State the crossover bandwidth in tech-stack.md and test both kernels agree to 1e-12.

B3. The fixed-PC restriction is never surfaced in the API, yet every real-data assertion depends on it.
Evidence: `src/kikuchipy/signals/util/_master_pattern.py:83-124` branches on `detector.navigation_shape == (1,)` vs per-PC. A back-projection LUT is by construction single-PC — that is exactly where EMSphInx's speed comes from. But `kp.data.nickel_ebsd_small()` and `nickel_ebsd_large()` both carry **per-point** PC arrays ((3,3,3) and (55,75,3)), and their stored `xmap`s were produced with those per-point PCs. The plan quietly substitutes `pc_average` in every test (D.1, Feature 3, 4, 5) and then asserts sub-degree agreement against those maps. Meanwhile `EBSD.spherical_indexing`'s signature and docstring say nothing about it, and no `ValueError` is specified for a multi-PC detector — a user passing `s.detector` directly will get either a crash inside the LUT builder or silently wrong results.
Fix: (i) validate `detector.navigation_shape == (1,)` in `SphericalBackProjector.__init__` with an explicit error pointing at `detector.pc_average`; (ii) document the limitation in the `spherical_indexing` docstring Notes and in mission.md as a known scope boundary vs `refine_orientation` (which does support per-point PC); (iii) budget the PC-spread contribution into the tolerance vs the stored xmaps rather than treating it as noise.

================================================================
MAJORS
================================================================

M1. `idctn(type=3)` is not FFTW REDFT01 — verified false.
Evidence: tech-stack.md says "DCT via `scipy.fft.dctn(type=2)` / `idctn(type=3)` (identical to FFTW REDFT10/REDFT01, unnormalised)". Measured:
  `dct(x, type=2)` == REDFT10  -> True
  `dct(x, type=3)` == REDFT01  -> True
  `idct(x, type=3)` == REDFT01 -> False  (ratio non-constant: [0.114, 0.045, 0.0]; scipy's idct type 3 is dct type 2 / 2N — a different transform, not a scale factor)
This is not cosmetic: the DCT pair drives `image::Rescaler`, `MasterPattern::resize`, `toLegendre`, the Feature 3 back-projection resampler and the DCT image quality — every place the plan asserts byte- or 1e-3-level EMSphInx parity. Fix: `dctn(type=2)` / `dctn(type=3)`, and add the test the plan already sketches (`test_rescale_dct_matches_fftw_convention`) as a *first* test in Feature 3, not a late one.

M2. Performance targets are internally inconsistent and not credible; worse, one of them is a merge gate.
Evidence: Feature 4 acceptance says ">= 5 patterns/s per core at bw 68"; section C time model says "50-100 ms/pattern/thread at bw 68 (>= 10 pat/s/core), i.e. nickel_ebsd_large (4125) in ~1 min on 8 cores". These differ by 2×. The anchor is EMSphInx's own log: 655 pat/s on 20 threads at bw 55 ≈ 30 ms/pattern/thread in hand-tuned C++ with FFTW and per-thread FFTW wisdom. Scaling ~bw⁴ from 55→68 gives ≈70 ms/thread in C++. Assuming numba + scipy.fft matches optimized C++/FFTW within 1× is not defensible; 2-4× is the honest band → 3-7 pat/s/core, and `nickel_ebsd_large` in 5-15 min on 8 cores. Additionally the refinement cost is under-modelled: `dTablePre` rebuilds a bw³×2 table (5 MB at bw 68) *per Newton iteration*, up to 15 iterations → ~75 MB of write traffic per pattern, on top of a bw²·(bw/2) derivative loop. EMSphInx's own profile puts refinement at 49% of runtime **on GPU**; on CPU in Python it will likely exceed the plan's "refinement < 3x the coarse time" acceptance.
Fix: convert the Feature 4 throughput number from a pass/fail gate into a recorded baseline in validation.md, with a much lower hard floor (e.g. 2 pat/s/core) that triggers the fallback list rather than blocking the branch. Restate the tutorial runtime honestly.

M3. "Coarse: all 9 within 0.5 deg of EMSphInx, differences arise only from tiny numerical noise" is unjustified.
Evidence: the coarse result is `np.argmax` over a discrete grid with cell size 360/slP = 2.67° at bw 68, followed by a tri-quadratic fit on the winning 3×3×3 box. When two neighbouring cells are near-tied — routine for a 60×60 noisy pattern — a sub-percent difference in the DCT rescaler, the bilinear LUT weights, or the `np.linalg.solve` ring weights flips which box is chosen, and the answer moves by ~2.7°, not by "numerical noise". One flipped point out of nine breaks `all < 0.5 deg`.
Fix: assert robust statistics for the coarse comparison (`median < 0.5°`, `>= 8/9 < 1.5°`) and reserve tight tolerances for the *refined* comparison, where both implementations converge to the same continuous maximum and the argument does hold.

M4. `scipy.special.sph_harm_y` is used in tests but is newer than the declared floor.
Evidence: `pyproject.toml:61` declares `scipy >= 1.7`; `requires-python >= 3.10`. `sph_harm_y` landed in scipy 1.15; the legacy `sph_harm` was **removed** in 1.17 (confirmed: `hasattr(scipy.special,'sph_harm')` is False in the repo venv), so there is no fallback. The CI "oldest" job (ubuntu, py3.10) does not pin scipy, so resolution is version-dependent, and any user/dev with scipy 1.7-1.14 gets a collection error.
Fix: use `scipy.special.lpmv` / `sph_legendre_p` with an explicit phase factor (both are old and stable), or guard with a version check, or negotiate a scipy floor bump with maintainers (a user-facing change that must go in the CHANGELOG and `doc/user/installation.rst`).

M5. The Condon-Shortley statement is under-determined and one test depends on the missing half.
Evidence: the plan asserts "no Condon-Shortley in the ALF recursion, hence the `(-1)^m` applied to odd `m` in analyze/synthesize (`square_sht.hpp:439, 554`)". But `:439` applies `(-1)^m` to the *ring FFT data*, not to the ALF, so the net convention of the stored `alm` is the composition of both and is stated by neither half of the sentence. `scipy.special.sph_harm_y` **includes** Condon-Shortley. `test_analyze_matches_scipy_sph_harm` is then written with "sign convention: no Condon-Shortley" and a tolerance of `|1 - value| < 1e-8`, which will fail by a sign on odd m if the composition is CS-including.
Fix: make the convention an *output* of a determination test (evaluate Y_l^m with `sph_harm_y`, analyze, print the empirical per-m sign, then freeze it as a constant), rather than an asserted input. Document the frozen result in the `SphericalHarmonicTransform` class docstring.

M6. The licensing/patent conversation is scheduled ~4 features too late.
Evidence: risk 6 and open question 1 defer the upstream issue to Feature 4. By then Features 1-3 — the bulk of the code derived from `square_sht.hpp`, `wigner.hpp`, `detector.hpp`, `master.hpp` — already exist. The two things that can kill the project outright are (a) the CMU provisional patent on "the central indexing algorithm" (EMSphInx ReadMe, Financial Support section) and (b) kikuchipy's dual-license policy in `doc/dev/licensing_considerations.rst`, which forbids GPL-derived code in BSD areas and instructs reviewers to ask contributors to *prefer* BSD. Confirmed: that file is **absent** from the local `doc/dev/` (13 files, no `licensing_considerations.rst`) — the clone predates it, so the plan is right that a merge is needed, but wrong to defer acting on the policy.
Fix: move into Feature 0, as blocking checkboxes: merge `upstream/develop`; read `licensing_considerations.rst`; open a kikuchipy issue referencing EMSphInx issue #7 (hakonanes, 2020, still unanswered); email pyxem.team@gmail.com asking explicitly (i) is GPL-derived code acceptable in `indexing/`, (ii) does the CMU provisional patent status need resolving before merge. Do not start Feature 1 until (i) is answered.

M7. `PatternRepack` is not actually ported — the user asked for it as functionality.
Evidence: user request lists PatternRepack among the programs to port "as pure code-driven functionality usable from notebooks like the rest of kikuchipy". Feature 7 demotes it to "docs section mapping `PatternRepack` -> `kp.load(...).save('*.h5')`/`EBSD.downsample`". That mapping does not work: verified that `kikuchipy_h5ebsd/patterns.h5` has root datasets `manufacturer` (lowercase, value `b'kikuchipy'`) and `version` — and `PatternFile::GetVendor` (EMSphInx/include/modality/ebsd/pattern.hpp:608-637) searches for a root dataset literally named `Manufacturer` or `" Manufacturer"` and throws `"doesn't have a Manufacturer string"` otherwise, then throws again unless the value is one of EDAX/EMsoft/Oxford/Bruker/Bruker Nano/DREAM.3D (`pattern.hpp:463-471`). PatternRepack's actual job — emit that header plus a contiguous, `H5D_ALLOC_TIME_EARLY` `/patterns` dataset, with optional vertical flip and binning — is exactly the code the plan already needs for section D.2, so it is being written and then thrown away as a script.
Fix: promote it to a small public function (`kikuchipy.io.write_emsphinx_patterns(signal, filename, vendor="Bruker", binning=1, flip=False)` or an `emsphinx_patterns` writer plugin), tested and exported. ~30 lines, and it is what makes the whole reference-generation recipe reproducible by a reviewer instead of only on this machine.

================================================================
MINORS
================================================================

m1. `EBSPDims` mapping is partial. `src/kikuchipy/io/plugins/oxford_binary/_api.py:282-322` infers the grid from the first, second and last patterns' `beam_x`/`beam_y` and assumes an equal step. `EBSPDims` (EMSphInx/programs/ebsp_dims.cpp:76-103) collects the *full* set of unique x and y and prints the sorted coordinate lists when `sx.size()*sy.size() != numPat` — the diagnostic for interrupted/non-rectangular scans, which is the entire reason the program exists. Either port that diagnostic or say plainly in roadmap.md that only the happy path is covered.

m2. The `tutorials_sanitize.cfg` change in Feature 4 is unnecessary. Verified: `regex2` is `\d+.\d+ (?=patterns/s)`, which already sanitizes `"  Indexing speed: {x:.5f} patterns/s"`. `"Indexing N pattern(s) in M chunk(s)"` is deterministic given the data, so needs no sanitizing at all. Drop it from the changed-files list.

m3. Feature 1 acceptance is self-contradictory: "`analyze` of one 387x387 pair < 50 ms after JIT (bw 384 < 1 s)". A 387×387 Legendre pair *is* bw 384 — two budgets 20× apart for the same operation. Neither accounts for the one-time `computeWeightsSkip` solve, an `np.linalg.solve` on a 193×193 `cos(2jθ)` matrix at dim 387, which the exploration report itself calls "the dominant construction cost", nor for its conditioning (EMSphInx guards with `sum(w)-1 <= cbrt(eps)/64` — verified at square_sht.hpp:1057).

m4. Feature 0's spec folder is self-cancelling: `specs/2026-08-16-constitution/` is described as holding "only mission.md, tech-stack.md, roadmap.md at `specs/` root" — i.e. an empty directory — and it contains none of the `plan.md`/`requirements.md`/`validation.md` the user's workflow mandates *per feature*. Either drop the folder or give it real contents.

m5. `MasterXcorr`'s stereogram outputs (`true.svg`, `pseudo.svg`: master pattern + point-group symmetry markers + detected operators) are not covered by Feature 6 or 7 — only the HDF5 volume and XDMF, both marked "optional". orix supplies `StereographicPlot`, `Symmetry.plot()` and `_symmetry_marker`, so this is cheap and is the actual human-readable output of that program.

m6. `navigation_mask` polarity is unstated. kikuchipy's convention (`src/kikuchipy/signals/ebsd.py:1870-1874`) is "only patterns equal to **False** are indexed" — the opposite of the naive reading, and `_refinement` passes `~navigation_mask` in places. Fix the docstring text now, not at review time.

m7. `.sht` "read -> write -> bytes identical" needs restating. The header carries `softwareVersion` (the writer's git hash, `ve49ad6b` in the shipped Ni file) and `notes`, while `MasterPatternHarmonics.save()` defaults `notes="created with kikuchipy"`. The byte-identity test must be read → write-preserving-original-metadata → compare; say so explicitly or the test is unimplementable as written.

m8. The `accum_e` energy-weighting path is effectively untested by default. `nickel_ebsd_master_pattern_small` is uint8 with a **single** 20 keV bin, so weighting is a no-op; and kikuchipy's reader does not expose `accum_e` at all (raw h5py needed). The plan tests it "indirectly through the parity test" only on the weekly 305 MB path. Add a direct unit test of the summation against a synthetic `accum_e`.

m9. `uv run pre-commit run --all-files` assumes `pre-commit` is in the uv env; kikuchipy's `[dev]` extra is doc+tests+coverage+ruff+black+isort+hatch. Verify or use `uv tool run pre-commit`.

m10. Even-`slP` glide parity is left undecided ("use exact `+slP//2 mod slP` when slP even - decide with test"). For all recommended bandwidths slP is odd (105/125/135/175/225/245/315 — measured), so the EMSphInx off-by-one at `sht_xcorr.hpp:530-531` is dormant; but the plan's own xcorr test list includes bw 54-64, where padding produces even slP (bw 55 → 110). Decide now: reproduce EMSphInx behaviour for parity, and put any "fix" behind a flag so regressions stay comparable.

m11. There is no memory row for bw=68 — the default bandwidth — in section C's table (only 63, 88, 113). At bw 68, slP=135, bwP=68: `fxc` ≈ 19.8 MB, `xc` ≈ 9.9 MB, `rDen` ≈ 9.9 MB. Add it, since it is what the docstring will quote.

m12. The CMU commercial-license block (Center for Technology Transfer and Enterprise Creation, innovation@cmu.edu) is part of the notice in every EMSphInx header. GPLv2 §1 requires keeping notices intact. The plan's attribution block omits it. Either preserve it verbatim or reference the upstream header explicitly.

================================================================
WHAT IS GOOD — KEEP
================================================================

G1. The license analysis is correct and load-bearing. GPL-2.0-or-later → GPL-3.0-or-later via the "or any later version" clause is the right reading; the requirement to carry a GPLv2 §2(a) modification notice with author and date in every derived file is correct and often missed. Keep the exact attribution block wording.

G2. The four "repo facts verified while planning" that I re-checked are all correct:
 - kikuchipy h5ebsd carries lowercase root `manufacturer` = `b'kikuchipy'` → not readable by `IndexEBSD.exe` (confirmed both sides).
 - `IndexEBSD.nml` is untracked in the working tree; `hybrid_indexing.ipynb` and `load_save_data.ipynb` are locally modified (`git status`).
 - Sample tilt comes from the `.sht` `primaryAngle`, and the shipped Ni file is 75.7° while kikuchipy's Ni MC is 70° → a fresh `.sht` is required. This is the single highest-value catch in the plan; without it every regression would be silently 5.7° off.
 - Legendre roots ≡ positive half of `leggauss(dim-2)`, but ring weights must still come from the Sneeuw solve. Correct and non-obvious.

G3. The EMSphInx bug inventory is accurate where I spot-checked it: `interpPeak`'s bounds check is literally `max(|x[0]|, max(|x[1]|, |x[0]|))` with x[2] never tested (sht_xcorr.hpp:421); `MasterSpectra`'s mean divides by `totW` while the stdev divides by `totW*2` (master.hpp:571-581); `alm[0]=0` is commented out (master.hpp:594); the weight scaling `w0/(8i)` with `w0 == 4π` for odd dim (square_sht.hpp:1050-1062). Deciding per-bug whether to reproduce or fix — and gating the choice on the parity tests — is the right posture.

G4. Ordering by numerical risk. Validating the SHT round trip and the Wigner tables against EMSphInx's own C++ unit tests (`test/sht/*.cpp`, seed-0 mt19937_64, tolerances 5e-3 max / 5e-5 mean, cbrt(FLT_EPSILON) for refinement) *before* anything touches `EBSD` is exactly right, and those tests are genuinely portable because the C++ side is deterministic.

G5. The convention-lock strategy — isolating every sign/offset decision (`_zyz_to_rotation`, the ZYZ↔Bunge ±π/2, the pseudo-symmetry `q0*q`) in a single helper each, then pinning it empirically with a forward-projection test through `mp.get_patterns` at known rotations — turns the highest-risk class of defect into a one-line fix. Keep this verbatim.

G6. Zero new runtime dependencies. Rejecting pyshtools (wrong grid), shtns (CeCILL, no Windows wheels), pyfftw/FFTW (GPL, and the only forced-GPL link in the chain), and rocket-fft (no cp314 wheels) in favour of numpy/scipy/numba is correct on licensing, packaging and Windows grounds simultaneously.

G7. kikuchipy convention fidelity. Private `_spherical/` package with a warning docstring, exports via `__init__.pyi` `lazy_loader.attach_stub`, `@njit(cache=True, nogil=True)` with `fastmath` deferred until tolerances pass, `.py_func` tests for every kernel, dask `map_blocks` with the threaded scheduler rather than `parallel=True`, `-s` sign-off, branch off `develop`, `Unreleased -> Added` changelog entries with PR links, credits + `.zenodo.json`. Line references I checked (`ebsd.py:1600/1827/1986/2376`, `_master_pattern.py:83/544/593/695/730`) are all exact.

G8. The reference-generation recipe (section D). Repacking with an accepted `Manufacturer`, disabling EMSphInx-side preprocessing (`nregions=0`, `gausbckg=.FALSE.`) so both sides see identical inputs, `nthread=1`/`batchsize=1` for determinism, recording the EMSphInx commit hash in the `.npz` metadata, and the correct observation that Bruker fractional PC makes `delta` geometrically irrelevant (verified: `fx`, `fy`, `sDst` all scale with `pX`, so `sampleDir` is scale-invariant) — this is careful work.

G9. Not using the shipped `benchmarks/GPU_test_*` as a baseline. Their inputs are absent from the machine, the build was `feature/GPU@98c251a` not master, and CPU vs CUDA `Metric` differ by ~7× in scale. Correctly rejected.

G10. Incremental delivery shape. Eight independently mergeable branches, each with its own dated spec folder and a real user-visible artifact, riskiest kernels first, tutorial last — this matches the user's SDD requirement and keeps the license conversation reviewable in small pieces.

================================================================
SUGGESTED SEQUENCING CHANGES
================================================================
1. Feature 0 gains: merge `upstream/develop` (to obtain `licensing_considerations.rst`); upstream issue + maintainer email on GPL-area placement and the CMU patent, as a blocking checkbox.
2. Feature 1 gains: `fast_size` as a verbatim port of `fastSize` with a parity test; explicit dual-path SHT (numba matrices ≤128, per-ring `scipy.fft.rfft` above); a determination test for the Condon-Shortley composition; corrected acceptance criteria.
3. Feature 3 leads with `test_rescale_dct_matches_fftw_convention` using `dctn(type=2)`/`dctn(type=3)`; `SphericalBackProjector` raises on multi-PC detectors.
4. Feature 4 converts the throughput gate to a recorded baseline; coarse-vs-EMSphInx tolerances become robust statistics.
5. `write_emsphinx_patterns` moves from a section-D script into Feature 2 or 7 as tested public API.