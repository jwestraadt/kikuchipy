# HROSM -- `feat-HROSM`: plan

**Status (2026-10-06):** drafted on `feat-HROSM` (= fork `develop` @
de27741a = `origin/develop`; `hrebsd-dic` b64cc18f,
`feat-spherical-indexing` 6723aaf0, `feat-spherical-indexing-nlpar`
e49b3d85, `upstream/develop` 31666938, `stash@{0}` present and
untouched; untracked and never staged:
`AGH__Si_indent_1_512x672.h5oina`,
`specs/_research/plan-upstream-merge-0.13.1.md`). **APPROVED by
Johan 2026-10-06** (AskUserQuestion, after spec review rounds 1 and 2):
this file as written, with every default of section 7 (R1-R7, items
8-16 and 24-27) adopted; he confirmed R2 (`average="mean"`), R3
(EMsoft labels 0 and 1..n) and item 26 (weekly CI stays disabled;
heavy arms run in the local stage gates) explicitly. Stage A may
start. The design is the parked plan
`specs/_research/plan-hrosm-2026-09-28.md` as carried into
`requirements.md` (D1-D20 govern; this file never contradicts them and
lists the inconsistencies it found in section 7.3; spec review rounds
1 and 2 are dispositioned in section 10); `validation.md`
holds the V0-V16 oracle suite, the named mutant killers and the
append-only Recorded results ledger. Route (Johan, 2026-10-06): base
fork `develop`, one fork PR `feat-HROSM -> develop`, fan-out by a
merge into `hrebsd-dic` and a clean replay onto a new
`feat-spherical-indexing-hrosm` stacked on
`feat-spherical-indexing-nlpar` (section 1). Models (owner rule of
2026-10-05, D18): this spec on Opus 5.5 xhigh with ultracode; tests,
implementation, adversarial review, bug injection and fixes as
Workflow agents `{model: 'opus', effort: 'medium'}`, at most 10 agents
per workflow; Fable only to escalate ONE step after repeated errors
AND an inconsistency (a gate fails again after a fix round, reviewers
contradict each other, or spec and code still disagree after a fix),
the reason logged in the ledger, then back to Opus; commits and
pushes by the main loop only. Lean by design (Johan: the NLPAR build
"took forever"): three stages, three workflows per code stage instead
of NLPAR's five, at most one critic round unless a blocker survives,
no gold-plating. Section 0 is applied to the working tree now
(uncommitted) and is committed with the three spec documents in
commit 1 after the approval.

**Status (2026-10-07, spec re-review after Stage C):** Stage A committed
(1e9471e1, d8b837a2) and Stage B committed (5f3e899d, 05b9ee6c), both
pushed. 953358e2 (CI times, Stage B gates box) is local and goes out
with commit 6. Stage C is built and its close gate passed (validation
ledger entries 28-29). Decisions taken during the build that change this
file are folded in below and dated "2026-10-07, spec re-review":
nsamples 10 (D13), VMF over G+- (D5.4), Watson kappa recorded only
(V12), partial-bitwise regenerate check (D13.6), budget on pytest time
(D15.1), platform-dependent pins (entry 19), chunk bound 16 ulp (D8.12),
parallel simulation tasks (D8.6), metric and energy checked before any
work (D1.5 check 7), tutorial outputs stored with n_steps=10 for the
real map (section 4). Open: the Chen et al. (2015) bibliography entry
(7.2 item 12; section 13 row A18).

Drafting measurements (2026-10-06, read-only `git diff --numstat`,
`git grep`, `ls`, `gh pr list`, this machine; details in section 1):
`git merge-base develop feat-spherical-indexing-nlpar` = 4ed31813,
54 non-specs files differ (+1339/-1273), and in the files HROSM
touches the divergence is docstring/citation/packaging only, with no
overlapping or adjacent hunk (no replay conflict expected);
`git merge-base develop hrebsd-dic` = de27741a (`hrebsd-dic` =
`develop` + HREBSD; 40 non-specs files, +29121/-6), with four
same-anchor appends expected (section 1.2); `gh pr list --repo
jwestraadt/kikuchipy --state all`: #19 newest, merged, so HROSM
expects #20. Both EMsoft builds hold all five programs and both DLLs
(`C:/Users/westraadt.1/EMSOFT/EMsoftOOBuild/Release/Bin/` and
`C:/Users/westraadt.1/Software/EMSOFT/EMsoftOO/build-ifx-release/Bin/`,
the latter `EMsoftConfig.json`'s `EMsoftLibraryLocation`);
`EMdatapathname = C:/Users/westraadt.1/Software/EMSOFT/EMsoftData`,
`EMtmppathname = C:/Users/westraadt.1/.config/EMsoft/tmp`. Namelist
keys read from `C:\Users\westraadt.1\Repos\EMsoftOO\NamelistTemplates\`:
EMHROSM `&HROSMdata` (`gangle`, `misorang`, `nsamples`, `nosm`,
`dilate`, `orav`, `numEM`, `numIter`, `dpfile`, `OSMfile`, `OSMtiff`,
`IPFmap`, `angfile`, `ctffile`, `maxRAMmem`); EMgetOSM `&getOSM`
(`nmatch`, `dpweighted`, `dotproductfile`, `tiffname`);
EMFitOrientation `&RefineOrientations` (`nthreads`, `dotproductfile`,
`newdotproductfile`, `usemasterpatternfile`, `tmpfile`, `inRAM`,
`matchdepth`, `method`, `niter`, `nmis`, `step`, `PCcorrection`);
EMsampleRFZ `&RFZlist` (`samplemode`, `pgnum`, `maxmisor`, `nsteps`,
`rodrigues`, `quoutname`, `euoutname`, `rooutname`). Repository rules
that bind the new code: `tests/test_indexing/test_spherical_indexer.py::
TestExports::test_all_is_sorted` asserts `kp.indexing.__all__ ==
sorted(...)`; `CHANGELOG.rst` lists entries newest first (HROSM
bullets go to the top of `Unreleased -> Added`, `:19`);
`doc/user/bibliography.bib` is alphabetical by key except the trailing
`aanes*` block; CI runs `pytest src --doctest-modules`
(`.github/workflows/tests.yml:123`), so every `_hrosm/` docstring
example runs on CI.

## 0. Constitution amendments (applied in the working tree 2026-10-06; committed in commit 1)

The texts below are appended verbatim (Edit tool, no BOM; `roadmap.md`
line 1 unchanged and BOM-free). Nothing else in the three files
changes. `specs/2026-08-16-constitution/upstream-issue.md` and
`specs/_research/plan-upstream-merge-0.13.1.md` are never touched.

### 0.1 `specs/mission.md`, appended after line 75 (end of file)

```
## Fork feature path: HROSM (recorded 2026-10-06)

The fork also carries HROSM (high angular resolution orientation
similarity maps) as a feature path on branch `feat-HROSM`, spec
`specs/2026-10-06-hrosm/`: a NumPy port of EMsoftOO's program EMHROSM
(M. De Graef, 2025, BSD-3; no paper exists) that segments an indexed
orientation map into grains on its kernel average misorientation
(KAM), averages each grain's orientation, re-indexes each grain's
patterns against a fine misorientation ball around that average (5 deg
radius, 0.25 deg shells, 68,921 orientations by default) with
kikuchipy's own `EBSDMasterPattern.get_patterns` and normalised
cross-correlation, and builds a grain-aware orientation similarity map
from the best matches, as `EBSD.hrosm()` plus eight public building
blocks in `kikuchipy.indexing`, with a tutorial
`doc/tutorials/hrosm.ipynb` and a gallery example. It is correct by
default; `emsoft_compatible=True` reproduces the EMsoftOO develop
binary's orientation-stage arithmetic, defects included, as twelve
numbered switches, proven against EMsoft's own programs and files on
this machine (gated tests and shipped references). The input map can
come from any indexing method, spherical indexing included. Like
NLPAR, this path is built once on fork `develop` (one PR, `feat-HROSM
-> develop`) and fanned out by a merge into `hrebsd-dic` and a clean
replay onto `feat-spherical-indexing-hrosm`, stacked on
`feat-spherical-indexing-nlpar`; `specs/roadmap.md` carries the gates
and `specs/tech-stack.md` the rules.
```

### 0.2 `specs/roadmap.md`, appended after line 166 (end of file)

```
---

# Feature path: HROSM (branch `feat-HROSM`; spec `2026-10-06-hrosm`)

Not a Phase 13: HROSM is not in the EMSphInx dependency chain above.
Base is fork `develop` (de27741a); one fork PR `feat-HROSM -> develop`
(expected jwestraadt/kikuchipy#20, confirmed with `gh pr list` at PR
time), merged only on Johan's go (no advance go for HROSM) with
ubuntu/windows CI green; macOS known red on the pre-existing
`test_ni_proper_oh_count` (`23 == 22`). Johan approves
`specs/2026-10-06-hrosm/plan.md` before any test or code is written. A
box ticks only when the work is committed on `feat-HROSM` (`git log`).
Gate list per code stage: spec recorded -> failing tests committed ->
implementation (Stage A: plus the EMsoft reference run) -> adversarial
review + bug injection + fixes -> pre-commit clean -> CHANGELOG entry
-> pushed. Stage C is documentation: it skips the failing-tests gate
and keeps the CHANGELOG gate (it ships a tutorial).

## Stage A -- orientation engine (no dictionary indexing)
- [ ] `src/kikuchipy/indexing/_hrosm/` (`_emsoft_quaternions`, `_kam`, `_segmentation`, `_grains`, `_averaging`, `_directional_statistics`, `_sampling`, `_osm` with both OSM forms as array functions, `_emsoft_file`); public `GrainTable`, `average_grain_orientations`, `grain_bounding_boxes`, `grain_reference_orientation_deviation_map`, `kernel_average_misorientation_map`, `misorientation_ball`, `misorientation_ball_spacing`, `segment_grains_kam` in `kikuchipy.indexing`; the EMsoftOO BSD-3 block in every EMsoft-derived module; no numba kernels
- [ ] EMsoft references: `src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py` (import safe) run once on this machine (EMDI, EMFitOrientation, EMgetOSM, EMHROSM `center`/`center_dilate`/`wat`, EMsampleRFZ N 6 and 20; pre-flight and acid bands passed); the shipped `.npz` files with provenance (`program_md5` of every program and DLL), md5s in `_registry.py`
- [ ] Tests: `tests/test_indexing/test_hrosm_{kam,segmentation,averaging,sampling,osm,emsoft_regression}.py` (compat bitwise against literal transcriptions and the shipped references, correct mode against analytic fields and orix, recovery bands, reader on synthetic EMsoft-layout files, reference-file quad equality); the bin-gated arms (EMsampleRFZ N 20, regenerate-and-diff) and the local-gated arms (Ni6, GRX810, Al) run once and recorded; every tolerance measured then pinned in `validation.md`
- [ ] Adversarial review (fidelity vs EMsoftOO `mod_DIsupport`, `mod_cluster`, `mod_so3`, `mod_dirstats`, `mod_Lambert`, `mod_quaternions`; conventions/integration) + bug injection (plan section 6, Stage A rows) + fixes; coverage 100 % of the Stage A `_hrosm/` modules recorded
- [ ] Gates: `-n 0` then `-n 4` (red tests re-run alone), full suite, doctests, `SKIP=licenseheaders` pre-commit on explicit files, oldest-matrix recipe, clean-replay grep, default-suite budget measured; CHANGELOG "Added" bullet with the fork PR link; signed commits pushed (the failing-tests commit never alone)

## Stage B -- per-grain re-indexing and `EBSD.hrosm()`
- [ ] `src/kikuchipy/indexing/_hrosm/_driver.py` and `EBSD.hrosm()` directly after `dictionary_indexing` in `signals/ebsd.py` (validation order, absent points, grains per phase, the ball composed per grain, eager experimental block, chunked lazy dictionary, `pc="grain"|"single"`, `verbose` 0/1/2, the GROD coverage warning before any simulation, warnings, one output `CrystalMap` with the documented props)
- [ ] Backwards-compatible upstream touches: `orientation_similarity_map(..., *, grain_id=None, emsoft_compatible=False)` (legacy path bitwise unchanged at the defaults) and `_dictionary_indexing(..., verbose=True)`
- [ ] Tests: `tests/test_signals/test_ebsd_hrosm.py` (contracts, masks, PC policy, skips, `n_per_iteration` and lazy/eager invariance, determinism, warnings, physics sanity on a synthetic sub-grain map, end-to-end tolerance against the EMHROSM references: one grain weekly, full map local + weekly) and additions to `test_orientation_similarity_map.py` and `test_dictionary_indexing.py`; performance baselines recorded as local ledger runs, never gated
- [ ] Adversarial review + bug injection (Stage B rows) + fixes; coverage 100 % of `_hrosm/` re-recorded
- [ ] Gates as Stage A; CHANGELOG bullet extended with `EBSD.hrosm()` and the two keywords; signed commits pushed

## Stage C -- tutorial
- [ ] `doc/tutorials/hrosm.ipynb` (`nickel_ebsd_large`: dictionary indexing and refinement as in `pattern_matching.ipynb`, then `EBSD.hrosm()`; global OSM next to the HROSM OSM; KAM, grain map, GROD and its warning; synthetic sub-grain demonstration; parameter guidance and cost; differences from EMsoftOO's EMHROSM in words); `pattern_matching.ipynb` and `spherical_indexing.ipynb` linked, never edited
- [ ] Registration: `doc/tutorials/index.rst` after `pattern_matching`, `NOTEBOOKS` entry in `run_nbval.sh`, `tutorials_sanitize.cfg` sections (if any) numbered from `[regex20]`, stored outputs if > ~2 min on the RTD builder; gallery example `examples/indexing/hrosm.py` with the new section file `examples/indexing/README.rst`
- [ ] Validation matrix + failure-mode review (clean-kernel execute, nbval, html render inspection, linkcheck, name/spell pass) + fixes; `sphinx-build -b html` exit 0
- [ ] CHANGELOG tutorial bullet; the three spec documents re-submitted to review and the amendments folded in (definition of done); then the signed commit pushed

## Fan-out (plan section 1; after the merge)
- [ ] Fork PR `feat-HROSM -> develop` opened with the PR template (number confirmed; CHANGELOG links rewritten if not #20); roadmap tick commit "Tick HROSM boxes in roadmap (jwestraadt/kikuchipy#20)"
- [ ] PR merged on Johan's go (merge commit; ubuntu/windows CI green); merge sha M recorded here
- [ ] `hrebsd-dic`: `git merge --no-ff develop`; the stub imports and `__all__` resolved in sorted order, the other append-type conflicts HREBSD first then HROSM; `segment_grains`/`segment_grains_kam` See Also cross-references added on `hrebsd-dic` only; the HROSM tests, `-k hrebsd`, then the full suite green; nbval on `hrosm.ipynb`; pushed; still never merged into `develop`
- [ ] `feat-spherical-indexing-hrosm`: new branch off `feat-spherical-indexing-nlpar` (e49b3d85, untouched), clean replay of M with `pick.ps1`/`gate.ps1` as two commits ("Add high angular resolution orientation similarity maps (HROSM)", "Add HROSM tutorial"; `Staged-from:` trailers), equivalence gate and the clean-replay grep clean, worktree suite == baseline + HROSM tests; pushed, no PR; `feat-spherical-indexing` stays at 6723aaf0
```

### 0.3 `specs/tech-stack.md`, appended after line 101 (end of file)

```
## HROSM feature path (branch `feat-HROSM`, spec `specs/2026-10-06-hrosm/`; recorded 2026-10-06)

Everything above applies on `feat-HROSM` too, with these additions and scopings.

- **Base and branch policy.** `feat-HROSM` is cut from fork `develop` at de27741a (= `origin/develop`: spherical indexing, pseudo-symmetry, spherical GPU, NLPAR #17/#18, the HyperSpy 2.5 test fix #19; 24 commits behind `upstream/develop`; no HREBSD code). The feature merges back through one fork PR `feat-HROSM -> develop` (expected #20; merge commit, only on Johan's go, ubuntu/windows CI green; macOS red only on the pre-existing `test_ni_proper_oh_count`, `23 == 22`). Johan approves `specs/2026-10-06-hrosm/plan.md` before any test or code (no waiver). Update rule: merge `develop` into `feat-HROSM` if needed, never rebase; **never merge `feat-spherical-indexing*` or `hrebsd-dic` into `develop`**. The upstream 0.13.1 merge stays parked (`specs/_research/plan-upstream-merge-0.13.1.md`, untracked, untouched); the 18.9 GB `AGH__Si_indent_1_512x672.h5oina` in the repository root is not ignored on `develop` and is never staged (explicit pathspecs only).
- **Fan-out after the merge (sha M).** (1) `hrebsd-dic` receives HROSM by `git merge --no-ff develop` (never the reverse). Expected conflicts: the `kikuchipy.indexing` stub (imports and `__all__`, resolved in sorted order, which `TestExports::test_all_is_sorted` enforces), the `ebsd.py` import block, the top of `CHANGELOG.rst` `Unreleased -> Added`, the `run_nbval.sh` notebook list, and the `tutorials_sanitize.cfg` EOF sections (HROSM added `[regex20]` and `[regex21]` in Stage C; `hrebsd-dic` uses 10-13; keep both, HREBSD first, no renumbering; amended 2026-10-07, spec re-review); HROSM adds no `bibliography.bib` key as built, so no conflict there; every other append keeps both sides, HREBSD first then HROSM. (2) A new branch `feat-spherical-indexing-hrosm`, stacked on `feat-spherical-indexing-nlpar` (e49b3d85; it and `feat-spherical-indexing` at 6723aaf0 stay untouched), receives M by the staging replay (`C:\Users\westraadt.1\Repos\_staging\pick.ps1`, then `gate.ps1`, in a worktree with `PYTHONPATH=<worktree>\src` and the `kikuchipy.__file__` guard) as two clean commits, feature and tutorial, with `Staged-from: jwestraadt/kikuchipy#20 (<M>)` trailers, `specs/` stripped, no PR; the equivalence gate compares the `+`/`-` lines of `git diff M^1 M -- . ':!specs'` with the replay diff. An upstream HROSM PR cut from that branch would carry the NLPAR commits unless NLPAR goes upstream first.
- **EMsoft oracles.** Two opt-in gates, never on CI. `KIKUCHIPY_EMSOFT_BIN` names a directory holding `EMDI.exe`, `EMFitOrientation.exe`, `EMHROSM.exe`, `EMgetOSM.exe`, `EMsampleRFZ.exe`, `EMsoftOOLib.dll` and `EMOpenCLLib.dll` (EMDI and EMHROSM need an OpenCL GPU; the data path is `EMdatapathname` of `~/.config/EMsoft/EMsoftConfig.json`, which the programs prepend to every relative path). `KIKUCHIPY_EMSOFT_DATA` names an EMsoft data root holding Johan's historical files (read-only inputs, md5 asserted). A test whose variable or file is missing skips, naming it. Both names are distinct from `KIKUCHIPY_EMSPHINX_DIR`, `KIKUCHIPY_LOCAL_MASTERS_DIR`, `KIKUCHIPY_NO_GPU_TESTS` and `hrebsd-dic`'s `KIKUCHIPY_LOCAL_DATA_DIR`. EMsoft programs never run concurrently: every run holds `_emsoft_program_lock` (root `conftest.py`, modelled on `_emsphinx_program_lock`, file `kikuchipy-emsoft-program.lock` in `tempfile.gettempdir()`, but with a 3600 s wait, a 30 s mtime heartbeat while held and a 300 s stale age, since a reference run holds it for ~36-48 min (measured 2026-10-06/07: 2,148.8 s for the shipped run, 2,846.8 s for the regenerate arm on a shared machine; ledger entries 11, 15), within the 3600 s wait), since they share `EMtmppathname` and one GPU. Run directories live only under `<EMdatapathname>/kikuchipy_hrosm/`: every namelist path key starts with `kikuchipy_hrosm/<ts>/` (the programs prepend `EMdatapathname`, and EMgetOSM/EMHROSM always write a TIFF), each program runs with its working directory in the run directory, and a new entry in the `EMdatapathname` root aborts the run. Johan's EMsoftOO checkout (`C:\Users\westraadt.1\Repos\EMsoftOO`) stays on `develop` and is only read (his branch `feature/emhrosm-grod-precheck` through `git -C ... show`).
- **Program and DLL md5 rule.** The EMsoft `.exe` files are ~50 kB launchers; the code lives in `EMsoftOOLib.dll` and `EMOpenCLLib.dll`. Every shipped EMsoft reference records `program_md5` of every program AND both DLLs, `emsoft_version`, `emsoft_commit` (from the DLL string), `gpu_name`, the master and pattern md5s, the PC and the namelist texts; the regenerate-and-diff test names the differing md5s on a mismatch. A changed binary means a new reference generation, new registry md5s and a ledger entry, never a widened tolerance.
- **EMsoftOO BSD-3 attribution.** Modules with EMsoftOO-derived code (`_emsoft_quaternions`, `_kam`, `_segmentation`, `_averaging`, `_directional_statistics`, `_sampling`, `_osm`, `_driver` in `src/kikuchipy/indexing/_hrosm/`) carry kikuchipy's GPL header plus the delimited third-party block of `src/kikuchipy/signals/util/_master_pattern.py:20-57`: the rationale line, the derived routines with their EMsoftOO files, the EMsoftOO BSD-3 notice verbatim ("Copyright (c) 2013-2026, Marc De Graef Research Group/Carnegie Mellon University", 2014-2026 for `mod_dirstats.f90`; "All rights reserved."; the three conditions and the disclaimer) and "Changes by the kikuchipy developers, <date>: ported from Fortran to NumPy; defects reproduced only behind `emsoft_compatible`." Burkardt's LGPL random-number routines, J-P Moreau's Bessel routines, SLATEC `SSORT` and LAPACK `DSYEV` are not ported (NumPy `Generator`, `scipy.special.ive`, NumPy sorting, `numpy.linalg.eigh`). The `EBSD.hrosm` Notes and the CHANGELOG acknowledge EMsoftOO's EMHROSM (M. De Graef) and Johan's EMsoftOO branch.
- **Numerics (HROSM-scoped, like the NLPAR and HREBSD scopings).** The float64 rule under "Numerics" is EMSphInx-scoped. HROSM keeps orientation maths in float64 and stores maps in float32 (`kam`, `osm`, `grod`, `max_grod`, as EMsoft does); `emsoft_compatible=True` mirrors EMsoft's precision exactly (KAM pairs and sums in float64 then `sngl`, OSM in float32, the ball as float32 Rodrigues vectors, unclipped `acos`); correct mode sets a symmetry-reduced dot within 4 float64 eps of 1 to 1 before `2 arccos` (identical orientations give exactly 0, never NaN). `emsoft_compatible` is a keyword on every entry point, never a module global. No numba kernels are planned; a kernel enters only if a stage gate measures a hot loop over budget, then under the numba rule above with `.py_func` parity.
- **Oldest-matrix recipe (local, once per stage, recorded; the full CI pin set of `.github/workflows/tests.yml:48`):** `uv run --isolated --python 3.10 --extra tests --with "dask==2021.8.1" --with "diffsims==0.5.2" --with "hyperspy==2.2" --with "matplotlib==3.6" --with "numba==0.57" --with "numpy==1.23.0" --with "orix==0.12.1" --with "pooch==1.3.0" --with "pyebsdindex==0.3.9.2" --with "scikit-image==0.21.0" pytest <the HROSM test modules> -n 0 -q -p no:cacheprovider`. Verified 2026-10-06 in an isolated py3.10 environment (scipy 1.13.1 resolved; CI pins no scipy): `Rotation.random_vonmises`, `Rotation.from_homochoric`, `orix.quaternion._conversions.cu2ho`, `Symmetry.proper_subgroup`, `Orientation.angle_with(degrees=...)`, `Rotation.angle_with_outer`, the Hamilton `Rotation.__mul__`, the `orix.io` save/load round trip of bool and 2-D props, `scipy.sparse.csgraph.connected_components`, `scipy.ndimage.maximum_filter`, `scipy.special.ive`, `scipy.spatial.cKDTree`. Tests never use `Orientation.reduce()`.
- **Numba-cache flake rule.** The root `conftest.py` autouse fixture `_keep_numba_cache_dir_per_worker` stays; gates run `-n 0` first, then `-n 4`, and a red test under `-n 4` is re-run alone before it counts. The two known pre-existing flakes of the NLPAR section never block a gate.
- **CI budget.** The fork's `tests.yml` job limit is 20 min (fork-only; 15 upstream and on the staging branches); on `develop` de27741a the jobs already take up to 17 min 21 s (ubuntu py3.10 oldest; windows py3.13 16 min 55 s). HROSM's default-suite additions are gated by the CI-style time of the HROSM selection (`-n 4` with coverage, this machine; amended 2026-10-06, Stage A gates: pytest's own reported time, the work HROSM adds, not the shell wall time, which includes the xdist and coverage start-up the whole suite pays once: measured 18.2-19.3 s pytest vs 40.2-40.6 s wall) <= 25 s, the NLPAR post-trim equivalent, with 60 s serial (`-n 0`, warm caches) as the ceiling, both measured at every stage gate and pinned in the ledger; heavy arms are `@pytest.mark.weekly` (balls with `n_steps=20` and real dictionary indexing, the V13 one-grain arm), local + weekly (the full-map run) or local ledger runs (performance baselines, `si_wafer`), and regenerate-and-diff is bin gated. The fork's Weekly workflow (`weekly.yml`, the only CI home of `--weekly` and nbval) is `disabled_inactivity` since 2026-08 (its notebook job failed on the last three runs), so the weekly arms run in the local `--weekly` stage gate and nbval of `hrosm.ipynb` in the local Stage C and fan-out gates; the weekly additions' local seconds are recorded with a scaled CI estimate (target <= 5 min), never a gate. The on-push CI run of each stage push (the Stage B push at the latest, before Stage C) is checked against 18 min on every job; a job above 18 min triggers the NLPAR trim at once (arms to weekly, one named killer per mutant kept).
- **Fixtures and reference data.** Test data from `src/kikuchipy/data/**` (`nickel_ebsd_small`, `nickel_ebsd_master_pattern_small`; `nickel_ebsd_large(allow_download=True)` once cached; `ebsd_master_pattern("ni")` weekly) or generated in the test with fixed seeds (HROSM generators are plain functions in the root `conftest.py` exposed as fixtures; test modules never import each other). EMsoft references ship uncompressed under `src/kikuchipy/data/emsoft_hrosm/` (each `.npz` <= 250 kB, md5 in `_registry.py`, no URL), written by the import-safe `create_hrosm_reference.py` (excluded from doctests by an `--ignore-glob` and from coverage by the existing `omit`).
- **CHANGELOG.** Fork PR-link convention `` (`#20 <https://github.com/jwestraadt/kikuchipy/pull/20>`_) ``, entries at the top of `Unreleased -> Added` (newest first), the number confirmed with `gh pr list` at PR time and rewritten if it differs.
- **Clean-replay rule (enforced at every stage gate and before the replay).** Nothing under `src/`, `tests/`, `doc/`, `examples/`, `benchmarks/`, nor the root `conftest.py`, `CHANGELOG.rst` or `pyproject.toml`, may name a `specs/` path, a spec file name (`requirements.md`, `plan.md`, `validation.md`, `tech-stack.md`, `mission.md`, `roadmap.md`) or a spec ID (D-, V-, M-, K-, R-numbers); comments, docstrings and test names state the fact ("EMsoft credits the vertical neighbour to iii-W+1"), never the ID. Gate: `git diff develop...HEAD -- src tests doc examples benchmarks conftest.py CHANGELOG.rst pyproject.toml ':!*.ipynb' | grep -E "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md|mission\.md|roadmap\.md|[^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)"` prints nothing, and the same pattern over the cell sources of `doc/tutorials/hrosm.ipynb` (read with `nbformat`, since stored base64 outputs give false hits) finds nothing; a Fortran literal such as `180.D0` is written `180/pi`.
- **Type hints and output.** `X | Y` (no `Optional`/`Union`); `print` only behind `verbose`, as `dictionary_indexing` does; no `print()` in tests; no module-scope import of `kikuchipy.signals` inside `_hrosm/` (the `ebsd.py` import of the driver would be circular).
```

### 0.4 `specs/_research/plan-hrosm-2026-09-28.md`, status header (replaces lines 3-17, from `**Status (2026-09-28)` through the "To resume" list; line 1 and everything from `## Context` unchanged)

```
**Status (2026-10-06): SUPERSEDED** by `specs/2026-10-06-hrosm/`
(`requirements.md`, `plan.md`, `validation.md`), which carries this
design into execution with these changes. The route changed (Johan,
2026-10-06): `feat-HROSM` is cut from fork `develop` (de27741a), not
from `feat-spherical-indexing`; it ends in one fork PR `feat-HROSM ->
develop` and fans out by a merge into `hrebsd-dic` and a clean replay
onto a new `feat-spherical-indexing-hrosm` stacked on
`feat-spherical-indexing-nlpar`. Decisions 2 and 6 below are
therefore superseded, and "Step 0" drops out because `develop` tracks
`specs/`. The spec folder is re-dated from `2026-09-28-hrosm` to
`2026-10-06-hrosm`. The "Process" section is replaced by the model
rule of 2026-10-05: spec work on Opus 5.5 xhigh with ultracode; tests,
implementation, review, bug injection and fixes as Workflow agents on
`{model: 'opus', effort: 'medium'}`, at most 10 agents per workflow;
Fable only as an escalation on repeated errors and an inconsistency.
Design amendments made while drafting the spec are dated in
`requirements.md`. The body below stays as the design record and is
not edited.
```

## 1. Branch policy and CI implications (binding, user decision 2026-10-06; requirements D14, D15)

- **Base and target.** `feat-HROSM` is cut from fork `develop` @
  de27741a (verified 2026-10-06 equal to `origin/develop`). All work
  (spec, tests, implementation, fixes, tutorial) commits onto
  `feat-HROSM` in gate order. ONE fork PR `feat-HROSM -> develop`
  after Stage C: `gh pr create --repo jwestraadt/kikuchipy --base
  develop --head feat-HROSM` with `.github/PULL_REQUEST_TEMPLATE.md`;
  expected #20, confirmed at PR time with `gh pr list --repo
  jwestraadt/kikuchipy --state all --limit 3`; the CHANGELOG bullets
  carry `` (`#20 <https://github.com/jwestraadt/kikuchipy/pull/20>`_) ``
  and are rewritten in the tick commit if the number differs. Update
  rule: merge `develop` into `feat-HROSM` if `develop` moves (never
  rebase); never merge `feat-spherical-indexing*` or `hrebsd-dic` into
  `develop`.
- **Merge gate.** Only on Johan's go (no advance authorisation, unlike
  NLPAR), as a merge commit, ubuntu and windows CI green; macOS red on
  `test_ni_proper_oh_count` (`23 == 22`) is pre-existing and recorded,
  not a blocker (the NLPAR merge run 37426647339 had no other macOS
  failure). The fork's tests workflow runs on every push, so pushes
  are an extra signal; the local gates of `validation.md` carry the
  recorded verification. **Push policy:** per stage; a failing-tests
  commit is never pushed alone.
- **Supersession.** D14 replaces the parked plan's decisions 2 and 6
  and its Step 0 (`develop` already tracks `specs/`; no `.gitignore`,
  pre-commit-exclude or `.git/info/exclude` edits). Section 0.4 records
  this in the parked plan's header.

### 1.1 Divergence `develop` vs `feat-spherical-indexing-nlpar` in HROSM's files (measured 2026-10-06)

Merge base 4ed31813; `git diff --stat develop
feat-spherical-indexing-nlpar -- . ':!specs'`: 54 files, +1339/-1273.
Numstat is `+`/`-` going from `develop` to the staging branch.

| file (HROSM's change on `develop`) | numstat | staging hunks | HROSM hunk | replay conflict |
|---|---|---|---|---|
| `src/kikuchipy/signals/ebsd.py` (driver import after `:58`; `hrosm` after `:2558`) | +16/-8 | IQ doctest `:1925-1931`; Hough `:cite:` Notes `:2235-2251`, `:2334-2346`; on staging `dictionary_indexing` is at `:2408`, `spherical_indexing` at `:2567` (+8) | `:58`, `:2558` | none (no overlapping or adjacent hunk) |
| `CHANGELOG.rst` (bullets at the top of `Unreleased -> Added`, `:20`) | +16/-0 | `0.13.1 (2026-09-04)` section at `:105` | `:20` | none |
| `doc/user/bibliography.bib` (only if 7.2 item 12: `chen2015parameter` after `chen2015dictionary`, `:54`) | +26/-0 | `rowenhorst2024fast` at `:251`; EOF `aanes2026kikuchipy`, `aanes2026kikuchipy_arxiv` | `:54` | none |
| `pyproject.toml` (`--ignore-glob` after `:181`) | +2/-2 | ebsdsim floors `:72`, `:83` | `:182` | none |
| `.gitignore` (not touched) | +35/-1 | `specs/` and raw data (`*.npz`, `*.h5`, ...) ignored, `src/kikuchipy/data/**` re-included | -- | n/a; the shipped `.npz` stay addable |
| `.pre-commit-config.yaml` (not touched) | +2/-2 | exclude without `specs/`; ruff `v0.16.6` (`develop`: `v0.15.15`) | -- | n/a; `gate.ps1` may reformat, recorded |
| `src/kikuchipy/detectors/_ebsd_detector.py` (read only) | +7/-3 | docstring | -- | n/a |
| `tests/test_indexing/test_spherical_emsphinx_regression.py` (read only; `TestReferenceFiles` is cloned, its module docstring is not) | +1/-2 | module docstring `specs/` path | -- | n/a |
| `.github/workflows/{tests,weekly}.yml` (not touched; fork-only timeout 20 on `develop`, 15 on staging) | +6/-8, +4/-4 | -- | -- | n/a |
| `doc/conf.py` (not touched) | +1/-1 | pyxem intersphinx URL | -- | n/a |

No divergence: `src/kikuchipy/indexing/{__init__.pyi,
_orientation_similarity_map.py,_dictionary_indexing.py}`,
`src/kikuchipy/signals/ebsd_master_pattern.py`,
`src/kikuchipy/data/{_registry.py,_data.py}`, `conftest.py`,
`doc/tutorials/{index.rst,run_nbval.sh,tutorials_sanitize.cfg}`,
`examples/` (no `examples/indexing/` on either branch),
`tests/test_indexing/test_{orientation_similarity_map,dictionary_indexing}.py`.
All new paths (`_hrosm/`, `data/emsoft_hrosm/`, the test modules, the
notebook, `examples/indexing/`) exist on neither branch. **Expected
replay conflicts: none.**

### 1.2 Fan-out step 1: `hrebsd-dic` (after the merge sha M exists)

`git switch hrebsd-dic && git merge --no-ff develop -m "Merge develop
(HROSM) into hrebsd-dic"`. Measured 2026-10-06 against
`git diff develop hrebsd-dic` (merge base de27741a):

| file | `hrebsd-dic` change | HROSM change | conflict | resolution |
|---|---|---|---|---|
| `src/kikuchipy/indexing/__init__.pyi` | +12: six `._hrebsd.*` imports after `:18`; `hrebsd_gnd`, `hrebsd_kam`, `hrebsd_pc_shift`, `hrebsd_strain_stress` after `"find_pseudo_symmetry_operators"` (`:57`); `segment_grains`, `voigt_stiffness` after `"read_emsphinx_psym_file"` (`:60`) | five `._hrosm.*` import lines after `:18`; eight `__all__` entries (section 2.11) | yes, same anchors | sorted order (`test_all_is_sorted`): imports `._hough_indexing`, `._hrebsd.*`, `._hrosm.*`, `._merge_crystal_maps`; `__all__` `... "find_pseudo_symmetry_operators", "grain_bounding_boxes", "grain_reference_orientation_deviation_map", "hrebsd_gnd", "hrebsd_kam", "hrebsd_pc_shift", "hrebsd_strain_stress", "kernel_average_misorientation_map", "merge_crystal_maps", "misorientation_ball", "misorientation_ball_spacing", "orientation_similarity_map", "read_emsphinx_psym_file", "segment_grains", "segment_grains_kam", "voigt_stiffness", ...` (neither new name collides on `hrebsd-dic`, checked 2026-10-06) |
| `src/kikuchipy/signals/ebsd.py` | +451: six import lines `:59-64`; `hrebsd_dic` after `spherical_indexing` (`:3045+`) | three `kikuchipy.indexing._hrosm` import statements at `:59-63` (`_driver._hrosm`, `_emsoft_quaternions.emsoft_point_group_number as _emsoft_point_group_number`, `_grains._map_grid as _hrosm_map_grid`); `hrosm` at `:2564` (as built; amended 2026-10-07, spec re-review) | import block only | `_hrebsd` imports, then the three `_hrosm` imports (ruff isort order) |
| `CHANGELOG.rst` | +55 at the top of `Unreleased -> Added` (`:21`) | two bullets at the same place | yes | HREBSD bullets first, then HROSM (the NLPAR precedent, 1532813d) |
| `doc/tutorials/run_nbval.sh` | +31: licence header, the `KIKUCHIPY_LOCAL_DATA_DIR` export, `"hrebsd_dic.ipynb"`, `"hrebsd_si_indent.ipynb"` after `"hough_indexing.ipynb"` | `"hrosm.ipynb"` after `"hough_indexing.ipynb"` (`:9`) | yes | `hrebsd_dic`, `hrebsd_si_indent`, `hrosm` |
| `doc/tutorials/tutorials_sanitize.cfg` | +33 at EOF: `[regex10]`-`[regex13]` | `[regex20]`, `[regex21]` at EOF (added 2026-10-07, Stage C; amended 2026-10-07, spec re-review) | yes, EOF | keep both, HREBSD's `[regex10]`-`[regex13]` first, then `[regex20]`-`[regex21]`; no renumbering |
| `doc/user/bibliography.bib` | +83: nine keys alphabetically, `ernould2020global` (+2 more) in the gap `chen2015dictionary`..`foden2019indexing` (`:52-54`) | not touched as built (amended 2026-10-07, spec re-review; open item 7.2 item 12, section 13 row A18: only option (i) adds `chen2015parameter` in this gap) | no (yes only under option (i)) | under option (i): alphabetical, `chen2015parameter`, then the `ernould*` keys |
| `doc/tutorials/index.rst` | +9: new "Strain and lattice rotation" section after Indexing (`:53`) | `hrosm` after `pattern_matching` (`:44`) | no (8 lines apart) | -- |
| `specs/{mission,roadmap,tech-stack}.md` | blocks inserted before NLPAR's (`:52`, `:125`, `:83`); roadmap `:1-4` | EOF appends after NLPAR's | no | -- |
| `.gitignore` | `*.h5oina` | not touched | no | -- |

No `hrebsd-dic` change in `conftest.py`, `pyproject.toml`,
`_registry.py`, `_orientation_similarity_map.py`,
`_dictionary_indexing.py`, `ebsd_master_pattern.py` or the two
existing test modules HROSM extends.

**Names.** `git grep -i` over `hrebsd-dic` (`src`, `tests`,
`conftest.py`, `doc`, `examples`) finds none of
`kernel_average_misorientation`, `segment_grains_kam`,
`grain_bounding_boxes`, `average_grain_orientations`,
`misorientation_ball`, `GrainTable`, `hrosm`, `emsoft_compatible`,
`KIKUCHIPY_EMSOFT`, `_emsoft_program_lock`, `emsoft_hrosm`,
`examples/indexing`; `grod` occurs only in notebooks. The prop name
`grain_id` is used by `_hrebsd/{_engine,_gnd,_kam}.py` with 0-based
labels and -1; HROSM's `grain_id` is EMsoft's (0 unassigned, 1..n;
R3). Coexistence per D1.2 (amended 2026-10-06, spec review), no
rename: `hrebsd_kam` and `hrebsd_gnd` need HREBSD-only props
(`rotation_vector`, `Fe`) besides `grain_id` and raise without them,
but `EBSD.hrebsd_dic(grain_labels=...)` takes 0-based labels with -1
outside any grain and would accept HROSM's `grain_id` silently. After
the merge, ONE `hrebsd-dic`-only commit "Cross-reference the KAM and
misorientation grain segmentations" adds a one-sentence See Also entry
to `segment_grains` and to `segment_grains_kam`, and to the latter's
Notes the sentence "pass `grain_id - 1` (unassigned 0 becomes -1) as
`grain_labels` of `EBSD.hrebsd_dic`" (`segment_grains` does not exist
on `develop`, so the cross-reference cannot live there).

**Gates on `hrebsd-dic`:** no conflict markers (`git grep -n -E
'^(<{7} |={7}$|>{7} )'`); `uv run --no-sync pytest $B_TESTS -n 0`,
then `-n 4` (section 5); `uv run --no-sync pytest tests -k hrebsd -n
4`; the full suite `-n 4` (baseline re-measured on b64cc18f before the
merge; seed 5212 passed / 0 failed at 1532813d); nbval on
`hrosm.ipynb`; `SKIP=licenseheaders uvx pre-commit run --files
<resolved non-specs files>`. Push. `hrebsd-dic` is never merged into
`develop`.

### 1.3 Fan-out step 2: clean replay onto `feat-spherical-indexing-hrosm`

1. Baseline: `git worktree add ../kikuchipy-hrosm-clean -b
   feat-spherical-indexing-hrosm feat-spherical-indexing-nlpar`; in the
   worktree, `PYTHONPATH=<worktree>\src` and the guard `uv run
   --no-sync python -c "import kikuchipy; assert
   kikuchipy.__file__.startswith(r'<worktree>')"`; full suite `-n 4`
   recorded as the baseline (seed 4515 passed / 1251 skipped, measured
   on 8571c081; e49b3d85 added tutorial content only).
2. `powershell -File C:\Users\westraadt.1\Repos\_staging\pick.ps1 M`
   (first-parent `cherry-pick -m 1 -n`, strips `specs/`, reports
   conflicts outside `specs/`; expected none, section 1.1).
3. Two commits, split by path, messages in
   `C:\Users\westraadt.1\Repos\_staging\msg-20a.txt` and `msg-20b.txt`
   (format of `msg-17a.txt`: what, why, oracles, tests; ending with
   `Staged-from: jwestraadt/kikuchipy#20 (<full M sha>)` and the
   session's attribution trailer):
   - **"Add high angular resolution orientation similarity maps
     (HROSM)"**: `src/`, `tests/`, `conftest.py`, `pyproject.toml`
     (`doc/user/bibliography.bib` only if 7.2 item 12 is resolved by
     option (i); amended 2026-10-07, spec re-review), and `CHANGELOG.rst` with
     the API bullet only (the tutorial bullet, the first HROSM bullet
     since it sits ABOVE the API bullet (newest first),
     removed from the working
     file with the Edit tool, `git add CHANGELOG.rst`, restored after
     the commit; the NLPAR precedent 03c3ca47 + 8571c081 split the
     CHANGELOG 15 + 3 lines).
   - **"Add HROSM tutorial"**: `doc/tutorials/hrosm.ipynb`,
     `doc/tutorials/index.rst`, `doc/tutorials/run_nbval.sh`,
     `doc/tutorials/tutorials_sanitize.cfg` (`[regex20]`, `[regex21]`;
     amended 2026-10-07, spec re-review),
     `examples/indexing/{README.rst,hrosm.py}`, the CHANGELOG tutorial
     bullet.
   `powershell -File C:\Users\westraadt.1\Repos\_staging\gate.ps1`
   before each commit (no `specs/`, no conflict markers, ruff check and
   format with the main `.venv` ruff, nbformat validation, pre-commit
   with `SKIP=licenseheaders` under the staging `v0.16.6` config;
   reformat and record if the ruff versions disagree).
4. Equivalence gate: the `+`/`-` lines of `git diff M^1 M -- .
   ':!specs'` equal those of `git diff feat-spherical-indexing-nlpar
   feat-spherical-indexing-hrosm` apart from recorded resolutions and
   reformatting (count recorded; NLPAR: 9577/9577); the clean-replay
   grep of section 5 run on `git diff feat-spherical-indexing-nlpar
   feat-spherical-indexing-hrosm -- src tests doc examples benchmarks
   conftest.py CHANGELOG.rst pyproject.toml ':!*.ipynb'` plus the
   notebook source check prints nothing.
5. Tests in the worktree (guarded `PYTHONPATH`): `$B_TESTS -n 0`, `-n
   4`, the full suite (== baseline + the HROSM tests), nbval on
   `hrosm.ipynb`.
6. `git push -u origin feat-spherical-indexing-hrosm`; no PR; `git
   worktree remove ../kikuchipy-hrosm-clean`. The push triggers the
   fork CI there with the staging 15-min limit: recorded, not a gate.
   `feat-spherical-indexing` (6723aaf0) and
   `feat-spherical-indexing-nlpar` (e49b3d85) are verified unchanged
   with `git rev-parse`.

Recorded consequence of the stacking: an upstream HROSM PR cut from
`feat-spherical-indexing-hrosm` would carry the NLPAR commits unless
NLPAR goes upstream first.

### 1.4 CI implications and the budget (D15)

- `.github/workflows/tests.yml` on `develop`: `timeout-minutes: 20`
  (`:36`, fork-only), `PYTEST_ARGS: --cov-branch --cov-report=xml
  --reruns 2 -n 4 --cov=kikuchipy` (`:39`), doctests `pytest src
  --doctest-modules` (`:123`), matrix ubuntu/windows/macos x 3.13/3.14
  plus py3.10 oldest (`:48` pins), minimum-requirement and
  wheel-install jobs. The wheel ships `src/kikuchipy/**` (the `.npz`
  references included) and force-includes `tests/` and `conftest.py`
  (`pyproject.toml:146-151`), so every HROSM test must pass from the
  installed wheel (no repository-relative paths; references through
  `Path(kp.data.__file__).parent / "emsoft_hrosm"` and
  `kikuchipy.data._data.Dataset(...).fetch_file_path()`, as
  `test_spherical_emsphinx_regression.py:408, 438` does).
- Evidence for the budget (re-measured 2026-10-06, spec review): on
  `develop` de27741a (run 37508408494) the jobs take 17 min 21 s
  (ubuntu py3.10 oldest), 16 min 55 s (windows py3.13), 14 min 48 s
  (ubuntu py3.13), 11 min 20 s (windows py3.14), 10 min 15 s (wheel),
  so the 20 min limit leaves under 3 min; NLPAR's default tests cost
  ~43 worker-s and 22.0-25.8 s CI-style wall here after the trim (NLPAR
  ledger entry 20).
- HROSM budget (D15.1, amended 2026-10-06, spec review): the binding
  gate is the CI-style time as pytest reports it (amended 2026-10-06,
  ledger entry 14) of the HROSM selection (`uv run
  --no-sync --with pytest-cov pytest <selection> -n 4 -q -p
  no:cacheprovider --cov=kikuchipy --cov-branch --cov-report=`, this
  machine) <= 25 s (measured medians 11.81 s Stage A, 17.78 s Stage B;
  entries 17, 26; amended 2026-10-07, spec re-review); 60 s serial
  (`-n 0`, warm caches) stays the A5 ceiling; both measured at every
  stage gate (section 5) and pinned, the summed worker-seconds
  recorded; `_hrosm/` doctests <= 5 s; the V13 one-grain arm is
  weekly, the V13 full map local + weekly, and the performance
  baselines are local ledger runs, never tests; nbval of `hrosm.ipynb`
  <= ~5 min on this machine (measured 2026-10-07: 62.0-66.3 s pytest
  time quiet, 148-228 s loaded; entries 28-29). **Weekly is local (amended
  2026-10-06, spec review round 2; D15.1).** The fork's Weekly workflow
  (`weekly.yml`, id 282579366; `weekly-tests` job 15 min limit at
  `:60`, nbval job 30 min) is `disabled_inactivity`; its last scheduled
  runs (2026-07-27, 08-03, 08-10) failed in `test-documentation-notebooks`
  while `weekly-tests` passed (~4 min). No CI runner therefore runs the
  `@pytest.mark.weekly` arms or nbval of `hrosm.ipynb`: they run in the
  local `--weekly` gate of section 5 and the local nbval gates (Stage C,
  1.2, 1.3). The weekly additions' local serial seconds
  (`--durations=0`) are recorded per stage with a scaled CI estimate
  (local seconds x the last on-push ubuntu py3.13 job time / the local
  full-suite `-n 4` time; target <= 5 min), never a gate. Re-enabling
  the workflow is 7.4 item 26. The on-push CI run
  of each stage push is checked against 18 min on every job before the
  next stage starts (the Stage B push at the latest before Stage C);
  a job above 18 min applies the trim order of `validation.md` "CI
  budget" at once.
- The bin and local arms skip on CI with a reason naming the variable;
  the gate runs that use them are recorded locally (section 5).

## 2. Stage A -- orientation engine (bit-level; no dictionary indexing)

Deliverables: the `_hrosm/` modules below except `_driver.py`, the eight
public names (D20 added `misorientation_ball_spacing` and
`grain_reference_orientation_deviation_map`), the EMsoft reference
script, run and shipped files, six test modules, the CHANGELOG "Added"
bullet (Stage A part), roadmap Stage A ticks, ledger entries.

**Files.** Create:
`src/kikuchipy/indexing/_hrosm/{__init__,_emsoft_quaternions,_kam,
_segmentation,_grains,_averaging,_directional_statistics,_sampling,
_osm,_emsoft_file}.py`;
`src/kikuchipy/data/emsoft_hrosm/{__init__.py,create_hrosm_reference.py}`
and the shipped `regression_hrosm_*.npz` (2.12);
`tests/test_indexing/test_hrosm_{kam,segmentation,averaging,sampling,
osm,emsoft_regression}.py`. Modify:
`src/kikuchipy/indexing/__init__.pyi` (2.11),
`src/kikuchipy/data/_registry.py` (rows after `:38`, the last
`emsphinx/` row, same column alignment under `# fmt: off`),
`pyproject.toml` (`"--ignore-glob=src/kikuchipy/data/emsoft_hrosm/*.py",`
after `:181`; the coverage `omit` at `:161` already covers
`create_*.py`), `conftest.py` (2.13), `CHANGELOG.rst`,
`doc/user/bibliography.bib` (only if 7.2 item 12 is adopted),
`specs/roadmap.md` (ticks), `specs/2026-10-06-hrosm/validation.md`
(ledger). Private helper names below are recorded at plan level: an
implementer may rename one only together with its section 6 row.

Build modules, in implementation order:

1. **`_hrosm/__init__.py`**: docstring listing the submodules and the
   private-package warning, as `indexing/_spherical/__init__.py`;
   imports nothing.
2. **`_emsoft_quaternions.py`** (D1.1, D3.2, K2, K11): the literal
   `SYM_Qsymop` table (`mod_quaternions.f90:53-64ff`; EMsoft constants
   `sq22 = 0.7071067811865475244`, `sq32 = 0.8660254037844386467`,
   `half = 0.5`, exact zeros), `PGROT` (`mod_symmetry.f90:404-407`);
   `emsoft_symmetry_operators(pgnum: int) -> np.ndarray` ((n, 4)
   float64 in `QSym_Init_` order, `mod_quaternions.f90:2542-2746`;
   `432` = `SYM_Qsymop` columns 1, 5-10, 17-24, 2-4, 11-16);
   `emsoft_point_group_number(point_group: Symmetry) -> int` (orix
   point group by name -> EMsoft 1-32; `ValueError` "point group" when
   unmapped or when the `{+-q}` operator set differs from
   `point_group.proper_subgroup`, K11);
   `emsoft_euler_to_quaternion(euler)` (`eq_`,
   `mod_rotations.f90:5731-5775`: half-angle cos/sin of `Phi`,
   `phi1 - phi2`, `phi1 + phi2`, `(cPhi cp, -sPhi cm, -sPhi sm, -cPhi
   sp)`, negated if q0 < 0); `emsoft_quaternion_multiply(a, b)`
   (`quatmult` term order of D3.2, `mod_quaternions.f90:1078-1126`);
   `emsoft_disorientation_angle(a, b, operators) -> float64 (n,)`
   (K2: for `j`, then `k`, in EMsoft order, `x = |scalar((S_j a) *
   conj(S_k b))|`, `angle = 2.0 * arccos(x)` unclipped under
   `np.errstate(invalid="ignore")`, `ac = 1000.0` replaced only when
   `angle < ac`, so NaN is never selected; vectorised over pairs,
   looped over the |G|^2 = 576 cubic products; measured 0.29 s for the
   28,086-pixel Ni6 map).
3. **`_emsoft_file.py`** (D12): `read_emsoft_dot_product_file(path) ->
   dict`, `read_emsoft_hrosm_file(path) -> dict`,
   `parse_namelist_text(text: str) -> dict` (EMHROSM's `dilate` lives
   only in the `NMLfiles/HROSMNML` text). Datasets, shapes, the
   `TopMatchIndices[:N]` slice (kept 1-based), the
   `(ipf_ht, ipf_wd)`/ROI map shape and the Fortran-order conversions
   exactly as D12.1-D12.3; h5py only; no file access at import.
4. **`create_hrosm_reference.py` and the reference run**: section 2.12
   (runs in parallel with modules 5-6 in the build workflow).
5. **`_kam.py`** (D3): public `kernel_average_misorientation_map(xmap,
   *, degrees=True, emsoft_compatible=False) -> np.ndarray` ((ny, nx)
   float32). Helpers: `_correct_kam(q, present, phase_id,
   operators_by_phase) -> float64 radians` (D3.1: 4-neighbour mean
   over existing, present, same-phase neighbours, pair angle
   `_dot_to_angle(max_j |<S_j o_a, o_b>|)` over the phase's
   `proper_subgroup`, NaN without a neighbour); `_dot_to_angle(d) ->
   float64 radians` (D3.1 as amended in spec review round 2: `d =
   np.where(d >= 1 - 4 * np.finfo(np.float64).eps, 1.0, d)`, then `2 *
   np.arccos(d)`; no separate clip; shared with `_grod_map`);
   `_emsoft_pair_angles(q,
   operators) -> (pair_h (H, W-1), pair_v (H-1, W), spurious)` (K2
   angles; `spurious = dis(identity, q[0, W-1])` with the identity
   quaternion `eq_(0, 0, 0)`); `_emsoft_neighbour_sum(pair_h, pair_v,
   spurious, H, W, dtype)` (D3.3, verbatim; for W == 1 pixel t gets `2
   dis(e(t-1), e(t))`, `e(0)` the identity); `_emsoft_edge_multipliers(
   acc, H, W, third)` (D3.4 sequence, verbatim; `third(x) = (x * 4.0) /
   3.0` in float64 for KAM). Output: correct `float32(float32(mean) *
   RTOD)`, compat `float32(float64(float32(acc)) * RTOD)`, `RTOD = 180 /
   np.pi` (= `57.29577951308232`); `degrees=False` returns
   `float32(mean)` / `float32(acc)`. Compat input (D3.2 input route,
   frozen 2026-10-06, spec review) = the first rotation per point as
   `emsoft_euler_to_quaternion(xmap.rotations.to_euler().astype(
   np.float32).astype(np.float64))`, never `xmap.rotations.data`
   directly (orix `from_euler` is 1 ulp off `eq_` in ~30 % of rows; the
   float32 round trip is exact on every row of the four EMsoft files).
   The grid comes from `_map_grid` (D1.9; `_grains.py`, written first by
   implementer 1), so 1 x n, n x 1, 1 x 1 maps return `(1, n)`, `(n,
   1)`, `(1, 1)`. Compat validation (K11, D1.5 fragments): dense map,
   one phase, EMsoft operator set.
6. **`_segmentation.py`** (D4): public `segment_grains_kam(kam, *,
   threshold=5.0, dilate=False, phase_id=None,
   emsoft_compatible=False) -> np.ndarray` (int32, 0 or 1..n) and
   `grain_bounding_boxes(grain_id) -> np.ndarray` ((n, 4) int64 row0,
   col0, height, width; a label without pixels gets height = width =
   0). Helpers: `_kam_difference_labels(kam, threshold, phase_id)`
   (D4.1-D4.3: `scipy.sparse.csgraph.connected_components` on edges
   over the offsets `((0, 1), (1, 0), (1, 1), (1, -1))` between finite,
   same-phase pixels with `np.abs(kam_a - kam_b) <= kam.dtype.type(
   threshold)` in KAM's dtype; keep components of size >= 2 that
   contain a pixel with `kam <= threshold`; labels 1..n by raster order
   of each component's first such pixel); `_dilate_emsoft(labels)` (K3,
   D4.5 expression: `M = maximum_filter(g, size=3, mode="constant",
   cval=0); out = g.copy(); out[1:, 1:] = np.where(M[1:, 1:] != 0,
   M[1:, 1:], g[1:, 1:])`); `_dilate_correct(labels, present,
   phase_id)` (unassigned present pixels take the largest same-phase
   8-neighbour label, simultaneous, never overwriting). The flood fill
   is a test oracle only.
7. **`_sampling.py`** (D6): public `misorientation_ball(center=None,
   *, max_angle=5.0, n_steps=20, emsoft_compatible=False) -> Rotation`
   (`(2N + 1)**3` rotations, element `i` = `~Q_v[i] * center`, the
   identity if `center is None`). Helpers: `_cubochoric_grid(
   max_angle_rad, n_steps) -> (M, 3)` (`edge = 0.5 * (np.pi * (w -
   np.sin(w))) ** (1 / 3)`, `dx = edge / N`, `i` slowest, `k`
   fastest); `_cubochoric_to_homochoric(cu)` (own port of
   `Lambert3DCubeForwardDouble`, `mod_Lambert.f90:1022-1113`, with
   `GetPyramidDouble`, `:1376-1432`, as `ch_` calls it,
   `mod_rotations.f90:5327-5353`), then `Rotation.from_homochoric`;
   `_emsoft_rodrigues_round_trip(q) -> q` (K8: the Rodrigues 4-vector
   `(axis, tan(w/2))` stored as float32, `(0, 0, 1, 0)` when `|v'| =
   0`, converted back in float64 with EMsoft's `rq_`). Public
   `misorientation_ball_spacing(max_angle=5.0, n_steps=20) -> float`
   (D20.1): the raw float64 ball `misorientation_ball(None,
   max_angle=max_angle, n_steps=n_steps)`, `d, _ =
   scipy.spatial.cKDTree(q).query(q, k=2)`, `angle = 4 *
   np.arcsin(d[:, 1] / 2)`, return `float(np.rad2deg(angle).mean())`
   (degrees); same argument validation and fragments as
   `misorientation_ball`; See Also both ways; docstring states the mean
   spacing (0.15918 deg at 5 deg / N 20) next to the radial shell step
   `max_angle / n_steps` (0.25 deg).
8. **`_directional_statistics.py`** (D5.4-D5.6, D5.10): the kappa
   tables (`xAp = 0.001 + (i - 1) * 0.001`, `i = 1..35000`, EMsoft's
   operation order; `yAp` from `scipy.special.ive`, Watson
   `(I1(k/2) / (I0(k/2) - I1(k/2))) / k`), `_kappa_from_y(y, kind)`,
   `_log_cp(kappa, kind)` (constants `C`, `C2`, `C2W` of D5.4),
   `EMResult` (NamedTuple: `mu`, `kappa`, `log_likelihood`,
   `n_iterations`, `best_init`; D5.5),
   `_em_correct(x, operators, kind, n_em, n_iter, rng) -> EMResult` (log
   space, left side, `eigh`, best finite `L`), `_em_emsoft(x, operators,
   kind, n_em, n_iter, rng) -> EMResult` (K5-K6 literal transcription
   with `Qi = Li = 0.0`), `_final_representative(mu, operators, emsoft_compatible)`
   (K7, D5.6 Voronoi form: compat `mu * S_i` with the largest `|q0|`,
   first index on ties; correct `S_i * mu`; q0 >= 0).
9. **`_grains.py` and `_averaging.py`** (D2.1, D5, D20.2-D20.3):
   `_map_grid(xmap) -> (grid, (ny, nx))` (D1.9; first, used by every
   module); `GrainTable` (frozen dataclass, fields `n_pixels`,
   `bounding_box`, `rotation`, `phase_id`, `kappa`, `max_grod`, `valid`,
   `method`; property `n_grains`; classmethod `from_crystal_map`);
   `_broadcast_grain_props(table, grain_id) -> dict` (the
   `grain_orientation`, `grain_kappa`, `grain_max_grod` props of D2.3).
   Public `average_grain_orientations(xmap, grain_id, *,
   method="mean", max_angle=5.0, n_em=25, n_iter=40, min_kappa=5.0,
   seed: int | np.random.Generator | None = None,
   emsoft_compatible=False) -> GrainTable` = the private core
   `_average_grains(...)` (same arguments, returns the table and never
   warns) plus the standalone coverage warning; helpers
   `_center_pixel(...)` (correct nearest-to-centroid; compat K4 box
   centre `(x0 + w // 2, y0 + h // 2)`), `_mean_orientation(...)`
   (D5.2), `_apply_kappa_gate(kappa, min_kappa) -> bool` (strict `>`,
   D5.7), `_grod_map(xmap, grain_id, rotation, valid,
   operators_by_phase) -> float32 (ny, nx)` degrees (D20.2: the
   symmetry-reduced angle to `GrainTable.rotation`, the ball centre,
   through `_kam._dot_to_angle`, so a correct `"center"` grain's own
   centre pixel is exactly 0.0;
   NaN at label 0, invalid grains, absent points; the ONLY GROD
   implementation, so `max_grod`, the public map and the driver's
   props agree bitwise), `_coverage_warning_message(labels, max_grod,
   n_pixels, max_angle, spacing=None) -> str | None` (D5.8/D20.3
   contract: None unless some `max_grod > max_angle`; else fragments
   "misorientation ball" and "max GROD", the count, `max_angle`, the
   spacing clause only if given, the largest max GROD, up to ten
   `(label, max GROD, n_pixels)` entries in descending max GROD with
   ties by ascending label, the hint `max_angle >= <largest rounded up
   to 0.5 deg>`); the standalone warning covers every valid grain,
   without spacing. Public `grain_reference_orientation_deviation_map(
   xmap, grain_id, grains) -> np.ndarray` = `_grod_map` with
   `grains.rotation`/`grains.valid`. One
   `numpy.random.default_rng(seed)` consumed across grains in label
   order (D5.9; a `Generator` passed in is used as is).
10. **`_osm.py`** (D7.1-D7.3, array functions only; the public routing
    is Stage B): `_osm_grain_aware(simulation_indices, grain_id,
    reindexed, n) -> float32 (H, W)` (sorted rows, broadcast equality,
    integer counts, float64 mean, NaN without an in-grain re-indexed
    neighbour); `_osm_emsoft(simulation_indices, H, W, n) -> float32
    (H, W)` (`vectormatch` counts on the first `n` indices,
    `_emsoft_neighbour_sum(..., spurious=0, dtype=np.float32)`,
    `_emsoft_edge_multipliers` with `third(x) = (x * np.float32(4)) /
    np.float32(3)` in float32, not divided by `n`).
11. **Exports** (`src/kikuchipy/indexing/__init__.pyi`): after `:18`
    insert `from ._hrosm._averaging import average_grain_orientations,
    grain_reference_orientation_deviation_map`, `from ._hrosm._grains
    import GrainTable`, `from ._hrosm._kam import
    kernel_average_misorientation_map`, `from ._hrosm._sampling import
    misorientation_ball, misorientation_ball_spacing`, `from
    ._hrosm._segmentation import grain_bounding_boxes,
    segment_grains_kam`; `__all__` (eight entries): `"GrainTable"`
    after `"EMSphInxNamelist"` (`:45`), `"average_grain_orientations"`
    before `"compute_refine_orientation_projection_center_results"`
    (`:53`), `"grain_bounding_boxes"`,
    `"grain_reference_orientation_deviation_map"` and
    `"kernel_average_misorientation_map"` after
    `"find_pseudo_symmetry_operators"` (`:57`), `"misorientation_ball"`
    and `"misorientation_ball_spacing"` after `"merge_crystal_maps"`
    (`:58`), `"segment_grains_kam"` after `"read_emsphinx_psym_file"`
    (`:60`). Every public docstring: numpydoc, Examples that run in < 1
    s on `nickel_ebsd_small` or a synthetic map, no Sphinx role to a
    private name and no spec ID (both enforced by V0
    `test_public_docstrings_link_no_private_name_and_no_spec_id`).
12. **Reference generation** (D13.2-D13.4; run by the reference
    runner of `hrosm-a-build`, alone on the GPU, before the measurer):
    - *Script.* `src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py`,
      import safe (no environment lookup or file access at import;
      module constant `SCENARIOS` naming the shipped files; the
      `create_emsphinx_reference.py:28-32` pattern); CLI `--out <dir>`
      (default `src/kikuchipy/data/emsoft_hrosm`), `--run-dir <dir>`
      (default `<EMdatapathname>/kikuchipy_hrosm/<YYYYmmdd-HHMMSS>/`),
      `--scenarios`. `main(output_dir=None, bin_dir=None,
      scenarios=None)` mirrors `create_emsphinx_reference.py:259-316`
      and its `_program_lock` (`:775`): with `bin_dir=None` it resolves
      `KIKUCHIPY_EMSOFT_BIN` and takes its own copy of the lock
      (`LOCK_NAME = "kikuchipy-emsoft-program.lock"`, the file the
      conftest lock uses, with the same 3600 s wait, 30 s heartbeat and
      300 s stale age of 2.13; `src/` cannot import the root
      `conftest.py`) for the whole run; the gated V14 test passes
      `bin_dir` while it already holds the conftest lock, so it never
      dead-locks. Each program runs as `<bin>/<prog>.exe <prog>.nml`
      with `cwd=<run>` (EMHROSM writes `center.txt`/`WAT.txt` into the
      working directory) and every namelist path key prefixed
      `kikuchipy_hrosm/<ts>/` (the binaries prepend `EMdatapathname`);
      the script lists the `EMdatapathname` root first and aborts,
      naming the program, if a new entry appears there after any
      program; `run.log` records command, exit code, stdout, stderr and
      wall time per program.
    - *Binary.* `KIKUCHIPY_EMSOFT_BIN=C:/Users/westraadt.1/Software/
      EMSOFT/EMsoftOO/build-ifx-release/Bin` (c127868, 6_0_20260525_0;
      7.2 item 8); the provenance records which ran.
    - *Pre-flight* (abort with a message naming the failed check):
      P1 the five programs and two DLLs present, md5s recorded; P2
      `EMsoftConfig.json` readable, `EMdatapathname` and
      `EMtmppathname` exist; P2b `<EMXtalFolderpathname>/Ni.xtal`
      exists with `CrystalData/SpaceGroupNumber == 225` and
      `LatticeParameters[0]` within 1e-6 of 0.35236 (nm) (D13.2.2); P3
      `nickel_ebsd_large` cached and the cached master
      `ni_mc_mp_20kv.h5` md5 `8b69c071a036ad3488d465093b67fe4d`; P4 a
      dry `EMDI` run into `<run>/preflight/` with `ncubochoric 10`
      against the PATCHED master copy: exit code 0 (the master and
      crystal are readable and an OpenCL device answers),
      `TopMatchIndices` of shape (N_pad, 20) with `N_pad % 32 == 0`
      and N_pad >= 4125; P5 no other EMsoft program running (the lock);
      P6 every path value of every generated namelist starts with
      `kikuchipy_hrosm/` (or is `'undefined'`, or is a `tmpfile`
      name). If P4 still rejects the master: fallback EMMCOpenCL +
      EMEBSDmaster from the `DItutorial/Ni` namelists with `sig 70`
      (+30-60 min), recorded, and P3 and the V14 `master_md5` pin
      amended in the same ledger entry.
    - *Inputs* (D13.2.1-3): `nickel_ebsd_large`, static then dynamic
      background removed as in `pattern_matching.ipynb`, written with
      the NORDIF writer (`io/plugins/nordif/_api.py:433`) as
      `<run>/patterns/Pattern.dat`; master copied into `<run>/` and
      both `xtalname` datasets of the COPY (`EMData/EBSDmaster/
      xtalname`, `NMLparameters/MCCLNameList/xtalname`, both
      `ni/ni.xtal` in the cached file) rewritten to `Ni.xtal` with h5py
      (same dtype and shape), `master_run_md5` recorded; PC from
      `det.pc = det.pc_average`, `pc = det.pc_emsoft()[0]` (a `(1, 3)`
      array), `xpc, ypc = pc[:2] / det.binning`, `L = pc[2]`, `delta =
      det.px_size * det.binning` (measured 4.6044, 17.1820, 240.9956,
      8.0), `thetac 0`, `omega 0`, `.6g` round trip stored.
    - *Runs* (namelists generated from the EMsoftOO templates with the
      listed keys changed, full text stored; `<r>` below is
      `kikuchipy_hrosm/<ts>`): (1) `EMDI.nml`: `ipf_wd 75, ipf_ht 55,
      ROI 0 0 0 0, nnk 20, nosm 10, nism 5, hipassw 0.05, nregions 4,
      maskpattern 'y', maskradius 29, ncubochoric 100, energymin 15,
      energymax 20, exptnumsx 60, exptnumsy 60, numsx 60, numsy 60,
      binning 1, scalingmode 'not', numdictsingle 32, numexptsingle
      32, nthreads 20, platid 1, devid 1, inputtype 'NORDIF', flipy
      .FALSE.`, `exptfile '<r>/patterns/Pattern.dat'`, `masterfile
      '<r>/ni_mc_mp_20kv.h5'`, `datafile '<r>/dp.h5'`, `tmpfile
      'EMEBSDDict_tmp.data'`, `keeptmpfile 'n'`, `ctffile`/`angfile`
      `'undefined'`; top-1 acid band: median disorientation of
      `EulerAngles` to the stored `xmap` <= ~1.5 deg (measured
      0.5880-0.5993 deg, entries 7, 11, 15); on failure
      retry once with `flipy .TRUE.` (the patterns file unchanged),
      record it, else abort. (2) `EMFitOrientation.nml`:
      `dotproductfile '<r>/dp.h5'`, `newdotproductfile
      '<r>/dp-refined.h5'`, `usemasterpatternfile
      '<r>/ni_mc_mp_20kv.h5'`, `angfile '<r>/dp-refined.ang'`, `tmpfile
      'EMFitOrientation_tmp.data'`, `inRAM .FALSE.`, `method 'FIT',
      niter 1, nmis 1, step 0.03, matchdepth 1, PCcorrection 'off',
      nthreads 20` (`readNameList` aborts without `angfile` or
      `ctffile`, `tmpfile`, and `usemasterpatternfile` when
      `newdotproductfile` is set, `mod_FitOrientation.f90:263-277`);
      refined acid band: median <= 0.5 deg (measured 0.3700-0.3755
      deg). (3) `EMgetOSM.nml`:
      `dotproductfile '<r>/dp-refined.h5'`, `tiffname '<r>/osm_'`
      (required, `mod_OSM.f90:177-179`), `nmatch = 10 5 0 0 0` ->
      `OSM_10`, `OSM_05`; self-check `OSM_10 == OSM` bitwise. (4)
      `EMHROSM.nml` x 3 with `dpfile '<r>/dp-refined.h5'`, `gangle 5.0,
      misorang 5.0, nsamples 10 (amended 2026-10-06 from 20: EMHROSM's
      per-grain memory leak, requirements D13), nosm 10, numEM 25, numIter 40,
      maxRAMmem 1.0`, `OSMfile '<r>/hrosm_<scenario>.h5'`, `OSMtiff
      '<r>/hrosm_<scenario>.tiff'` (written unconditionally,
      `mod_HROSM.f90:778`), `IPFmap`, `angfile`, `ctffile`
      `'undefined'`: `center` (`dilate .FALSE., orav 'center'`),
      `center_dilate` (`dilate .TRUE.`), `wat` (`orav 'averageWAT'`).
      (5) `EMsampleRFZ.nml` x 2: `samplemode 'MIS', pgnum 32, maxmisor
      5, nsteps 6` and `20`, `rodrigues` = unit axis `(1, 2,
      3)/sqrt(14)` with magnitude 0.1, `quoutname`, `euoutname`,
      `rooutname` = `'<r>/ball_n<N>_{qu,eu,ro}.txt'` (text, 9 decimals;
      `ro` = the unshifted grid, `qu`/`eu` = the moved ball, `eu` in
      degrees).
    - *Shipped files* (D13.3 as amended 2026-10-06 and 2026-10-07; the
      script's own byte-stable uncompressed zip writer (`np.savez`
      stamps the time; entry 7; amended 2026-10-07, spec re-review), no pickled objects,
      strings as `np.str_` arrays,
      byte-stable; `TestReferenceFiles` asserts < 250,000 B each):
      exactly the V14 key table, no conditional trimming:
      `regression_hrosm_large_di.npz` (242,474 B), `..._refined.npz`
      (holds `EulerAngles`; 126,586 B), `..._center.npz` (132,636 B),
      `..._center_dilate.npz` (132,688 B), `..._wat.npz` (132,628 B; no
      `newQuat`, no `maxGROD`), `regression_hrosm_ball_n6.npz` (`qu`,
      `ro`; 151,396 B), each with the provenance keys of D13.3 (incl.
      `master_run_md5`); total 918,408 B, pinned
      `REFERENCE_TOTAL_BYTES = 920_000` (measured, entries 11-12). The
      N 20 lists and `eu` are never shipped (bin gate).
    - *Registry and ledger*: the six md5 rows in `_registry.py`; one
      ledger entry with the provenance, acid-band values, timings
      (measured 2,148.8 s (35.8 min; EMDI 1,382.6 s, EMFitOrientation
      103.8 s, EMHROSM 196.8 / 258.8 / 189.0 s; entry 11), sizes and
      md5s; the run directory is kept (EMsoftData is outside the repo).
    - *Deviations* (as built, entry 7 item 7; amended 2026-10-07, spec
      re-review): comment-free namelists (the template comments hold
      example paths the path check rejects); `dpweighted = .FALSE.`
      unquoted for EMgetOSM; EMDI-only `namelist` provenance plus a
      per-scenario `hrosm_namelist`; `logging` instead of `print`; a run
      fails on "ended abnormally", "Fatal error" or "forrtl: severe" in a
      program's output, because EMsoft exits 0.
13. **Root `conftest.py`** (appended at EOF after the NLPAR block, under
    `# ------------------------------- HROSM ------------------------------ #`):
    `_EMSOFT_LOCK_TIMEOUT = 3600.0`, `_EMSOFT_LOCK_STALE = 300.0`,
    `_EMSOFT_LOCK_HEARTBEAT = 30.0` (D13.1 as amended 2026-10-06),
    `_emsoft_program_lock(path=None)` (modelled on
    `conftest.py:702-750`, file `kikuchipy-emsoft-program.lock`; a
    daemon thread `os.utime`s the lock every heartbeat while held and
    stops on release; a lock older than the stale age is taken over; no
    PID test); fixtures (the names of `validation.md`)
    `emsoft_bin_dir`, `emsoft_data_file(relpath, md5)` (skip naming
    the variable or the missing file; md5 asserted, cached per
    session), `emsoft_program` (runs one program under the lock with
    `cwd` in the run directory, returns the `CompletedProcess`);
    `_EMSOFT_LOCK_NAME = "kikuchipy-emsoft-program.lock"` and the
    `emsoft_program_lock` fixture; the synthetic generators as private
    plain functions named with a leading underscore
    (`_hrosm_gradient_xmap`, `_hrosm_constant_pair_xmap`,
    `_hrosm_grain_xmap`, `_hrosm_top_lists`,
    `_write_emsoft_layout_file`, `_hrosm_synthetic_signal`), each
    exposed by a fixture of the name without the underscore (as built;
    amended 2026-10-07, spec re-review; private helpers `_hrosm_crystal_map`,
    `_hrosm_master_pattern`, `_emsoft_config`, `_md5_of_file` besides);
    the fixtures are (`hrosm_gradient_xmap`,
    `hrosm_constant_pair_xmap`, `hrosm_grain_xmap`, `hrosm_top_lists`,
    `write_emsoft_layout_file(path, kind, arrays)`, which writes
    minimal EMsoft-layout DI and HROSM HDF5 files with h5py so
    `_emsoft_file.py` is covered in the default suite);
    `hrosm_synthetic_signal` is added in Stage B (commit 4).
14. **CHANGELOG** (top of `Unreleased -> Added`, newest first): one
    bullet naming the eight public names, what they do, the EMsoftOO
    EMHROSM acknowledgement and the fork PR link; extended in Stage B.

**Tests (Stage A): the module table of `validation.md` "Test modules"
is the authority** (amended 2026-10-06, spec review; one reference
loader and one degenerate-pixel helper, no cross-module imports):
`test_hrosm_kam.py` (V1 compat vs the test-local loop transcription
incl. 1 x n, n x 1, 1 x 1 and duplicated orientations; the Euler
round-trip pin; the point-group checks; V3 analytic fields, orix
`angle_with`, identical neighbours); `test_hrosm_segmentation.py` (V4
vs a flood-fill transcription incl. ties at exactly the threshold; V5;
V6 dilate and boxes); `test_hrosm_averaging.py` (V6 centre pixel; V8
recovery, `TestGROD` incl. the coverage-warning contract and the
public deviation map, `TestGrainTable` incl. the `from_crystal_map`
round trip; V9); `test_hrosm_sampling.py` (V7 incl. `TestBallSpacing`,
the shipped N 6 lists and the bin-gated N 20 run); `test_hrosm_osm.py`
(V10 and V11 array arms on synthetic lists only);
`test_hrosm_emsoft_regression.py` (EVERY EMsoft-file arm: V0 module
discipline, V2 shipped and local KAM, V10 shipped and local OSM, V12
cluster stage, V14 `TestReferenceFiles`, `TestEMsoftFileReader`,
`TestEMsoftProgramLock`, `TestRegenerateReferences`; Stage B adds V13).

**Not built in Stage A (or ever, unless re-decided):** numba kernels
(D10.3; V0 stays reserved), a public EMsoft reader, the 1e6 stack
truncation, `nGrains = -1`, the dead `ScanGrain_` rule, EMsoft's clock
seed and Burkardt RNG, misorientation-based segmentation (R6), GPU,
the dictionary-batch thinning (D8.10).

**Stage A gates:** failing tests committed (commit 2) ->
implementation + reference run + measured pins -> adversarial review,
bug injection (section 6, Stage A rows) and fixes -> section 5 gate
list: `-n 0`, `-n 4` (red re-run alone), coverage 100 % of the Stage
A `_hrosm/` modules, full suite, doctests, pre-commit, oldest matrix,
clean-replay grep, budget, never-sweep check; bin arms (V7 N 20, V14)
and local arms (Ni6, GRX810; Al with `--weekly`) run once and
recorded -> CHANGELOG bullet -> roadmap ticks -> commit 3 -> push 1 +
2 + 3.

**Workflow shape (all agents `{model: 'opus', effort: 'medium'}`):**

| workflow | agents | content |
|---|---|---|
| `hrosm-a-tests` | 6 | skeleton (stubs raising `NotImplementedError` with final signatures incl. the private interfaces named in this section (`_map_grid`, `_dot_to_angle`, `EMResult`, `_apply_kappa_gate`, `_grod_map`, `_coverage_warning_message`, `_average_grains`), headers with the GPL + BSD-3 blocks, exports, `pyproject.toml`, the conftest section, lock and Stage A generators, the script skeleton with `SCENARIOS`); writers (1) `test_hrosm_kam.py` + `test_hrosm_segmentation.py`, (2) `test_hrosm_averaging.py` + `test_hrosm_sampling.py`, (3) `test_hrosm_osm.py` (synthetic arms only) + `test_hrosm_emsoft_regression.py` (V0, V2, the V10 file arms, V12, V14, the reader, the lock test; the one reference loader and degenerate-pixel helper), each running only its modules `-n 0` and expecting `NotImplementedError`/missing-file failures and no collection errors; critic (read-only: would a plausibly wrong implementation pass, does every Stage A mutant have a NAMED killer, are the MTP placeholders inventoried, no spec IDs); fixer (+ ledger "Stage A failing-tests gate") |
| `hrosm-a-build` | 6 | phase 1: reference runner (module 3 + 12, the run, registry) and implementer 1 (`_map_grid`, then modules 2, 5, 6); phase 2: implementer 2 (7, 8, the rest of 9) and implementer 3 (10, 11, 14); phase 3: measurer (every MTP pin with date, machine, recipe; the D3.6 degenerate counts; the D7.3 multiplier order of the chosen binary; budgets); phase 4: gate runner (section 5, numbers not adjectives) |
| `hrosm-a-harden` | 8 | phases, strictly in order (the injector must never mutate code under active fixes): (1) fidelity reviewer (EMsoftOO source line by line against D3-D7 and the K table) and conventions/integration reviewer (numpydoc, hints, headers, exports, clean-replay rule), in parallel; (2) two skeptics per finding batch (a finding dies only if both refute it with evidence); (3) fixer (survivors, killer tests the review demands, disposition table appended to this file); (4) mutant injector alone (section 6 Stage A rows, in the main tree, sequential batches of 3, each dead by its named killer, file restored, slice re-run green); (5) strengthener (survivors -> sharper tests, kill verified by re-injection); (6) close-gate runner (section 5 list on the final tree; ledger/roadmap/CHANGELOG check) |

## 3. Stage B -- per-grain re-indexing and `EBSD.hrosm()`

Deliverables: `_driver.py`, `EBSD.hrosm`, the two
`orientation_similarity_map` keywords, `_dictionary_indexing`'s
`verbose`, the tests, performance baselines, the extended CHANGELOG
bullet, roadmap Stage B ticks.

**Files.** Create: `src/kikuchipy/indexing/_hrosm/_driver.py`,
`tests/test_signals/test_ebsd_hrosm.py`. Modify:
`src/kikuchipy/signals/ebsd.py` (`from kikuchipy.indexing._hrosm._driver
import _hrosm` after the `_hough_indexing` import block, `:58`; the
method after `:2558`, between `dictionary_indexing` (`:2400-2557`) and
`spherical_indexing` (`:2559`)),
`src/kikuchipy/indexing/_orientation_similarity_map.py`,
`src/kikuchipy/indexing/_dictionary_indexing.py`,
`tests/test_indexing/test_orientation_similarity_map.py`,
`tests/test_indexing/test_dictionary_indexing.py`, `conftest.py`
(`hrosm_synthetic_signal`, simulated from
`nickel_ebsd_master_pattern_small`),
`tests/test_indexing/test_hrosm_emsoft_regression.py` (V13 `TestEndToEnd`;
the V0 BSD-notice parametrisation gains `_driver`),
`CHANGELOG.rst`, roadmap, ledger.

1. **`_dictionary_indexing(..., verbose: bool = True)`** (D8.7, R7):
   last keyword; `False` suppresses the two `print`s (`:77-85`,
   `:136-139`), the `ProgressBar` (`:92`), the `tqdm` bar (`:105`,
   `disable=not verbose`) and the unconditional `sleep(0.2)` before the
   speed message (`:135`; 0.2 s per grain otherwise; added 2026-10-06,
   spec review round 2); not exposed by `EBSD.dictionary_indexing`;
   results bitwise identical either way.
2. **`orientation_similarity_map(..., *, grain_id=None,
   emsoft_compatible=False)`** (D7.5, R7): at the defaults the legacy
   body runs unchanged (bitwise; existing tests untouched); otherwise
   `footprint`, `center_index`, `from_n_best` must keep their defaults
   ("cannot be combined"), `n_best` sets `n`, the result comes from
   `_osm_grain_aware` (`grain_id` given; `reindexed` = present points)
   or `_osm_emsoft` (compat; dense map required), `normalize=True`
   divides by `n` after the table, absent points and `keep_n == 1`
   work, the grid from `_map_grid`, result `(ny, nx)` float32, never
   squeezed. The legacy path keeps its recorded limitations (D7.5,
   amended 2026-10-06: one-row, one-column and one-point maps raise
   "footprint.ndim"). The import of
   `_hrosm._osm` is inside the function or at module scope (no cycle:
   `_osm.py` imports nothing from `kikuchipy.indexing` beyond
   `_hrosm`).
3. **`_driver._hrosm(signal, xmap, master_pattern, detector, energy, *,
   grains: Sequence[int] | None = None, **keywords) -> CrystalMap`** (D8;
   `grains` is the test-only subset of D8.15), in order (amended
   2026-10-06, spec review, to D8.2 and D20.3): (a) absent mask
   (`is_in_data == False`, `phase_id == -1`, `navigation_mask ==
   True`); (b) KAM -> `segment_grains_kam(phase_id=...)` ->
   `grain_bounding_boxes` -> the private averaging core
   `_average_grains` (never warns) with GROD from `_grod_map`, all with
   the call's `emsoft_compatible`; (c) skip decisions: `n_pixels <
   min_pixels`, `valid == False`, empty labels, phases without the
   master (one "no master pattern" warning; "master pattern phase"
   raises when no phase matches, R4), labels outside `grains` when
   given; (d) `spacing = misorientation_ball_spacing(max_angle,
   n_steps)` and at most ONE coverage `UserWarning` from
   `_coverage_warning_message(...)` over the grains to be re-indexed
   only, with the spacing (D20.3: fragments "misorientation ball" and
   "max GROD", strict `>`, up to ten entries in descending max GROD);
   then, at `verbose >= 1`, the info line (ball size `(2N + 1)**3`,
   radius `max_angle`, the spacing, the number of grains to re-index);
   nothing has been built or simulated before this point; (e) the raw
   cubochoric ball once (68,921 x 32 B = 2.2 MB at the defaults); per
   grain `ball_g = ~R_v * avor_g` (orix Hamilton product), compat then
   `_emsoft_rodrigues_round_trip`; (e2) domain: the grain's present
   pixels (correct) or its bounding box in box raster order (compat,
   K10); (f) the
   experimental block `signal.data` reshaped `(n_map, sig_size)`,
   indexed by the domain and computed eagerly to float32 (dask fancy
   indexing for lazy signals); metric from `signal._prepare_metric(
   metric, None, signal_mask, None, False, ball_size)` with
   `n_experimental_patterns` overridden to the block size (`:4484`
   sets the map size), then `metric.navigation_mask = None` (Stage B
   code review F1; amended 2026-10-07, spec re-review); (g) `det_g` per grain: a one-PC detector as is;
   a per-point detector with `pc="grain"` -> `det_g =
   detector.deepcopy(); det_g.pc = <mean PC of the domain>`; with
   `pc="single"` -> `detector.pc_average`; (h) the dictionary
   `master_pattern.get_patterns(ball_g, det_g, energy,
   dtype_out="float32", compute=False, chunk_shape=...)`, chunked along
   the dictionary axis by `n_per_iteration` (default
   `_default_n_per_iteration(sig_size, ball_size)` = `clip(floor(256e6
   / (4 sig_size)), 1, ball_size)`: 17,777 at 60 x 60;
   `chunk_shape=_simulation_chunks(ball_size, n_per_iteration)`, tasks
   of >= 1,024 patterns, at most one per CPU, tiling each iteration
   chunk; ledger entry 22, code review F2; amended 2026-10-07, spec re-review); (i) `_dictionary_indexing(
   experimental=block, experimental_nav_shape=(n_domain,),
   dictionary=sim.data, step_sizes=(1.0,), dictionary_xmap=sim.xmap,
   metric=metric, keep_n=keep_n, n_per_iteration=..., verbose=verbose
   >= 2)` under `dask.config.set(**{"array.slicing.split_large_chunks":
   False})`, never with a navigation mask; (j) results:
   `rotations_g = ball_g[sim_idx[:, 0]]`, OSM per D7.4 from
   `sim_idx[:, :n_osm]`, copied back for the grain's pixels only;
   `scores` float32, `simulation_indices` int32 (cast: both
   `_dictionary_indexing` paths return int64, D8.11); (k) output
   `CrystalMap` per D2.2-D2.3 (fills: correct NaN/-1/input rotation;
   compat 0.0/-1/identity, K12), `grod` from the same `_grod_map`
   array, and `_broadcast_grain_props`; (l) `verbose`: 0 silent, 1 the
   info line of (d) + one `tqdm` over grains + the timing, 2 also each
   grain's dictionary message; (m) "no grain re-indexed" warning with
   the all-NaN OSM.
4. **`EBSD.hrosm(...)`**: signature of D1.3 verbatim; validation in
   the order and with the fragments of D1.5, then check 7, the metric
   and master-pattern energy (D1.5 as amended 2026-10-07); then the
   driver; numpydoc
   with the Notes "Differences from EMsoftOO's EMHROSM" in words
   (D9.4), the keyword map to the EMHROSM namelist, the acknowledgement
   (D11.4), See Also `misorientation_ball_spacing` and
   `grain_reference_orientation_deviation_map`, and `:cite:` keys
   `marquardt2017quantitative`, `chen2015dictionary`,
   `singh2016orientation`; a doctest (amended 2026-10-06, spec review:
   `nickel_ebsd_small` has 9 points, below the default `min_pixels=10`,
   and `verbose=1` prints) `s = kp.data.nickel_ebsd_small(); mp =
   kp.data.nickel_ebsd_master_pattern_small(projection="lambert",
   hemisphere="both"); out = s.hrosm(s.xmap, mp, s.detector,
   energy=20, n_steps=2, keep_n=5, n_osm=5, min_pixels=2, verbose=0)`
   (125 orientations) under ~2 s, asserting only stable facts
   (`out.prop["reindexed"].any()`, `out.prop["osm"].dtype`), any
   coverage warning filtered in the example.
5. **Tests**: `test_ebsd_hrosm.py` (V16 contracts: validation order
   and messages, masks, absent points, PC policy, `min_pixels`
   boundary, `kappa = -1` skip, vanished labels, multi-phase skip and
   warnings, `n_per_iteration` invariance, lazy == eager, seeded
   determinism, `verbose` 0/1/2 through `capsys`, the output contract
   and `GrainTable.from_crystal_map`, the `orix.io` round trip, compat
   K10/K11/K12, the coverage-warning order (before the first
   `get_patterns` call), skip exclusion and equality; V15 physics
   sanity, small default arm + weekly full size; V13 end-to-end vs the
   shipped `center` reference in `test_hrosm_emsoft_regression.py`: one
   grain weekly, the full map local + weekly (7.2 item 14 as amended);
   additions to `test_orientation_similarity_map.py` (V11 public
   routing, legacy bitwise at the defaults on a 2-D map, "cannot be
   combined", `normalize`, absent points, `keep_n = 1`, no squeeze) and
   to `test_dictionary_indexing.py` (`verbose=False` silent and
   bitwise equal).
6. **Performance baselines** (recorded, never gated; local ledger
   runs on this machine, NOT tests, amended 2026-10-06, spec review):
   `nickel_ebsd_large` full map at the defaults: total, per-grain
   simulation and NCC times, grains re-indexed, peak memory;
   `n_steps=10`; compat box-pixel overhead. Seeds (D8.13): NCC 2.0e12
   flops (~40 s), simulation ~3 s per grain at 60 x 60 (one
   `get_patterns` of 68,921 60 x 60 patterns measured 1.9 s here,
   2026-10-06), `n_steps=10` 7.4x cheaper, boxes +20-50 % pixels.
   Measured 2026-10-07 (entries 21-22, 24; amended 2026-10-07, spec
   re-review): defaults 186.3 s after the hot-spot fix (273.2 s before),
   `n_steps=10` 21.8 s, compat +91 % pixels / +13 % time, peak 1.24 GB;
   the spare implementer slot of `hrosm-b-build` was used for that hot
   spot.

**Stage B gates:** as Stage A, plus the `EBSD.hrosm` doctest, coverage
100 % of all of `_hrosm/` (incl. `_driver.py`), the budget re-measured
for the whole feature (CI-style <= 25 s, serial <= 60 s), the
oldest-matrix run, the weekly and local + weekly arms run once
locally, the on-push CI job times recorded and checked against 18 min
(section 1.4); commits 4 + 5; push.

**Workflow shape:** `hrosm-b-tests` (5: skeleton for `_driver.py`, the
method stub, the two keyword stubs, `hrosm_synthetic_signal` and the
`_driver` parameter of the V0 BSD-notice test; writers (1)
`test_ebsd_hrosm.py`, (2) the two existing modules + V13; critic;
fixer), `hrosm-b-build` (5: implementer 1 driver + method; implementer
2 OSM routing + `verbose`; measurer incl. V13/V15 pins and the
performance baselines as local ledger runs; gate runner; one spare
implementer slot for a measured hot spot), `hrosm-b-harden` (8, the
six phases of `hrosm-a-harden`, section 6 Stage B rows).

## 4. Stage C -- tutorial

Documentation stage: skips the failing-tests gate, keeps the
validation matrix, failure-mode review and the CHANGELOG gate.

**Files.** Create: `doc/tutorials/hrosm.ipynb`,
`examples/indexing/README.rst`, `examples/indexing/hrosm.py`. Modify:
`doc/tutorials/index.rst` (`    hrosm` after `:44` `pattern_matching`),
`doc/tutorials/run_nbval.sh` (`  "hrosm.ipynb"\` after `:9`
`"hough_indexing.ipynb"`), `doc/tutorials/tutorials_sanitize.cfg` (only
if nbval needs a new pattern; sections `[regex20]`, `[regex21]`, ...
appended at EOF after `[regex9]`), `CHANGELOG.rst`, roadmap, ledger,
and the spec re-review amendments.

1. **Notebook cell outline** (hidden first cell; thumbnail tag; black
   at 77 via the `black-jupyter --line-length=77` hook; `_ =
   dask.config.set(num_workers=8)` before verbose cells; drift-safe
   print precision; no bare repr with a memory address; no spec IDs;
   `pattern_matching.ipynb` and `spherical_indexing.ipynb` linked,
   never edited): (1) title, what HROSM does, credit to EMsoftOO's
   EMHROSM, the three `:cite:` keys; (2) `nickel_ebsd_large`
   (download), backgrounds removed, the dictionary route of
   `pattern_matching.ipynb` with `nickel_ebsd_master_pattern_small`
   and `refine_orientation`, the global OSM; (3) KAM in both modes,
   `segment_grains_kam`, `average_grain_orientations`, the GROD map and
   warning; (4) `s.hrosm(...)`, the HROSM OSM next to the global OSM,
   scores, `reindexed`; (5) the synthetic sub-grain demonstration (the
   V15 generator: a 0.5 deg sub-boundary invisible to a 1.4 deg global
   dictionary, visible in HROSM, the step recovered); (6) parameter
   guidance and cost (`max_angle`, `n_steps` -> `(2N + 1)**3`, ledger
   timings); (7) differences from EMsoftOO's EMHROSM in words and a
   short `emsoft_compatible=True` demonstration on the KAM.
   (As built, 2026-10-07; ledger entries 28-29; amended 2026-10-07, spec
   re-review.) (3) also shows the `threshold=2` split of the merged grain
   (two parts 9.3 deg apart) and the IPF-X map with its colour key. (4)
   runs `max_angle=5, n_steps=10` at `verbose=1` (the N 20 cost quoted in
   words). (5) builds its own 20 x 30 map (60 x 60 px, 2 % noise,
   sub-boundary 0.5 deg about [001] from column 8, global dictionary = two
   balls of 10 deg / N 7 offset by 0.7 deg, 0.9 deg mean spacing; the
   notebook cannot import the conftest generator) and re-indexes it at the
   defaults with `keep_n=10`. (7) differences in words plus the
   compat-minus-correct KAM map. 49 cells, 23 code.
2. **Runtime and stored outputs.** Measure the nbval runtime on this
   machine (budget ~5 min, MTP; 7.2 item 15) and estimate the ~2-vCPU
   Read the Docs builder; store outputs iff > ~2 min there (`uv run
   --no-sync --with ipykernel jupyter nbconvert --to notebook
   --execute --inplace doc/tutorials/hrosm.ipynb`, then strip the
   notebook-level `metadata.widgets`).
   Decided 2026-10-07 (amended 2026-10-07, spec re-review; entry 28):
   outputs stored (Read the Docs estimate about 4-11 min). Then `jupyter
   nbconvert --to notebook --inplace --coalesce-streams` (no re-execution)
   and the per-cell `metadata.execution` removed. No `metadata.widgets`
   was written.
3. **Gallery example** `examples/indexing/hrosm.py` (header and `# %%`
   cells as `examples/pattern_processing/neighbour_pattern_averaging.py`,
   `hs.preferences.General.show_progressbar = False`): a synthetic
   two-grain map simulated from `nickel_ebsd_master_pattern_small`,
   `EBSD.hrosm(max_angle=2, n_steps=4)` (729 orientations), runtime <=
   30 s (measured 6.6-6.8 s quiet, entry 28); figure: 2x2 OSM and GROD
   panels (entry 29; amended 2026-10-07, spec re-review); `examples/indexing/README.rst`: title "Indexing" and one
   sentence ("These examples cover indexing of EBSD patterns and the
   analysis of the resulting orientation maps."), the section layout
   of `examples/pattern_processing/README.rst`.
4. **CHANGELOG** second bullet: "Tutorial on high angular resolution
   orientation similarity maps (HROSM), ``doc/tutorials/hrosm.ipynb``,
   and a gallery example" + fork link.
5. **Validation matrix + failure-mode review**: clean-kernel execute,
   nbval (`uv run --no-sync --with nbval pytest -v --nbval
   doc/tutorials/hrosm.ipynb --nbval-sanitize-with
   doc/tutorials/tutorials_sanitize.cfg`), `uv run --no-sync
   sphinx-build -b html doc doc/_build/html` exit 0, html render
   inspection of every figure, linkcheck of the notebook's links,
   name/spell pass, the notebook clean-replay check, the never-sweep
   check.

**Stage C gates:** the validation matrix of item 5 green; nbval
runtime and the stored-outputs decision recorded; `$B_TESTS` `-n 4`
(`-n 2` on a loaded machine, section 5 as amended 2026-10-07)
and the `_hrosm/` doctests still green; `SKIP=licenseheaders uvx
pre-commit run --files` on the notebook, the two example files,
`index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg` (if touched) and
`CHANGELOG.rst`; the clean-replay grep and the notebook source check;
the never-sweep check -> CHANGELOG tutorial bullet -> spec re-review
(amendments folded in) -> roadmap Stage C ticks -> commit 6 -> push ->
PR (order fixed 2026-10-06, spec review: the re-review's amendments
land in commit 6).

**Workflow shape:** `hrosm-c-docs` (6: notebook author; gallery
author; measurer (runtimes, stored-outputs decision); validation-matrix
reviewer; fixer; close-gate runner) then `hrosm-c-spec-review` (2:
spec re-submission critic reading the three spec documents against the
shipped tree and returning dated amendments; fixer). Then commit 6,
push, PR.

## 5. Stage-independent recipe

**Per code stage:** `hrosm-<s>-tests` -> main loop: commit (failing
tests + skeletons; not pushed) -> `hrosm-<s>-build` ->
`hrosm-<s>-harden` -> main loop: commit (implementation, pins, fixes,
CHANGELOG, roadmap ticks, ledger, disposition rows) -> push both.
Every workflow: <= 10 agents, all `{model: 'opus', effort:
'medium'}`; agents read and write only their explicit file list, run
no git command that changes state, never sweep a notebook, write no
em-dashes, use `X | Y` hints, add no `print()` outside `verbose`, and
never name a spec ID outside `specs/`. Escalation: if a gate fails
again after a fix round AND an inconsistency appears (reviewers
contradict, or spec and code disagree after a fix), the main loop
re-runs that one step with `model: 'fable'`, logs why in the ledger,
and returns to Opus.

**Test selections** (Git Bash; the scratch directory is the session
scratchpad):

```
A_TESTS="tests/test_indexing/test_hrosm_kam.py tests/test_indexing/test_hrosm_segmentation.py tests/test_indexing/test_hrosm_averaging.py tests/test_indexing/test_hrosm_sampling.py tests/test_indexing/test_hrosm_osm.py tests/test_indexing/test_hrosm_emsoft_regression.py"
B_TESTS="$A_TESTS tests/test_signals/test_ebsd_hrosm.py tests/test_indexing/test_orientation_similarity_map.py tests/test_indexing/test_dictionary_indexing.py"
```

(Stage A gates use `$A_TESTS`, later gates `$B_TESTS`. D15.4's `-k
hrosm` selection would miss the keyword tests in the two existing
modules; the explicit selection is a superset.)

**Gate commands:**

```
uv run --no-sync pytest $B_TESTS -n 0 -q -p no:cacheprovider
uv run --no-sync pytest $B_TESTS -n 4 -q -p no:cacheprovider        # red tests re-run alone
COVERAGE_FILE="$SCRATCH/.coverage.hrosm" uv run --no-sync coverage run -m pytest $B_TESTS -n 0 -q -p no:cacheprovider
COVERAGE_FILE="$SCRATCH/.coverage.hrosm" uv run --no-sync coverage report -m --include="src/kikuchipy/indexing/_hrosm/*"
uv run --no-sync pytest src/kikuchipy/indexing/_hrosm src/kikuchipy/indexing/_orientation_similarity_map.py --doctest-modules -q -p no:cacheprovider
uv run --no-sync pytest src/kikuchipy/signals/ebsd.py --doctest-modules -k hrosm -q -p no:cacheprovider    # Stage B on
uv run --no-sync pytest tests -n 4                                    # full suite
SKIP=licenseheaders uvx pre-commit run --files <explicit changed files, never specs/>
uv run --isolated --python 3.10 --extra tests --with "dask==2021.8.1" --with "diffsims==0.5.2" --with "hyperspy==2.2" --with "matplotlib==3.6" --with "numba==0.57" --with "numpy==1.23.0" --with "orix==0.12.1" --with "pooch==1.3.0" --with "pyebsdindex==0.3.9.2" --with "scikit-image==0.21.0" pytest $B_TESTS -n 0 -q -p no:cacheprovider
git diff develop...HEAD -- src tests doc examples benchmarks conftest.py CHANGELOG.rst pyproject.toml ':!*.ipynb' | grep -E "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md|mission\.md|roadmap\.md|[^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)"   # must print nothing
uv run --no-sync python -c "import nbformat,re,sys; p=re.compile(r'specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md|mission\.md|roadmap\.md|[^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)'); nb=nbformat.read('doc/tutorials/hrosm.ipynb', as_version=4); bad=[(i, l) for i, c in enumerate(nb.cells) for l in c.source.splitlines() if p.search(l)]; print(bad); sys.exit(bool(bad))"   # Stage C on
KIKUCHIPY_EMSOFT_DATA="C:/Users/westraadt.1/Software/EMSOFT/EMsoftData" uv run --no-sync pytest $B_TESTS --weekly -n 4 -q -p no:cacheprovider --durations=0     # weekly and local + weekly, once per stage; the only run of these arms while the fork's Weekly workflow is disabled (1.4); the weekly arms' seconds summed and recorded
```

(Coverage: pytest-cov is not in `.venv`, so `coverage run` replaces
it; `COVERAGE_FILE` keeps `.coverage` out of the tree. The coverage
gate is measured on the default suite only, without `--weekly` and
without any gate variable (the same two commands in `validation.md`
"Definition of done"; unified 2026-10-06, spec review): nothing in
`_hrosm/` branches on the variables, which live in the fixtures and
the omitted script, and every `_hrosm/` branch has a default-suite
arm. Stage A excludes the not-yet-written `_driver.py` by
construction.)

(Amended 2026-10-07, spec re-review; ledger entries 10, 13, 17, 23, 26,
29.) When another session loads the machine, the full suite and the
selection gates run at `-n 2` (memory: 20 tests failed with
`_ArrayMemoryError` at `-n 4`, all passing alone). The CI-style budget
runs stay at `-n 4` on a quiet machine, and the ledger states the `-n`
used.

**Budget measurement** (D15.1 as amended 2026-10-06): (1) the binding
CI-style gate `uv run --no-sync --with pytest-cov pytest <the HROSM
selection> -n 4 -q -p no:cacheprovider --cov=kikuchipy --cov-branch
--cov-report=` time as pytest reports it (amended 2026-10-06, ledger
entry 14) <= 25 s (measured medians 11.81 s Stage A, 17.78 s Stage B;
entries 17, 26), three runs, the median
recorded; (2) `uv run --no-sync pytest <the new HROSM test node ids>
-n 0 -q -p no:cacheprovider --durations=0`, wall time minus a one-test
run of the same selection (fixed overhead), recorded with the summed
worker-seconds and the slowest ten durations, <= 60 s for the feature;
the Stage A share recorded.

**Bin and local gates** (never concurrent with other GPU or EMsoft
work; `-n 0`; `-rs` shows that no gated arm skipped):

```
KIKUCHIPY_EMSOFT_BIN="C:/Users/westraadt.1/Software/EMSOFT/EMsoftOO/build-ifx-release/Bin" uv run --no-sync pytest tests/test_indexing/test_hrosm_sampling.py tests/test_indexing/test_hrosm_emsoft_regression.py -n 0 -q -p no:cacheprovider -rs
KIKUCHIPY_EMSOFT_DATA="C:/Users/westraadt.1/Software/EMSOFT/EMsoftData" uv run --no-sync pytest $A_TESTS -n 0 -q -p no:cacheprovider -rs [--weekly for Al]
KIKUCHIPY_EMSOFT_BIN="<same>" uv run --no-sync python src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py --out src/kikuchipy/data/emsoft_hrosm   # Stage A reference run only
```

**Never-sweep and hygiene check** (every close gate): `git status
--short` and `git diff --name-only HEAD` list none of
`doc/tutorials/{hybrid_indexing,spherical_indexing,load_save_data,
pattern_matching,nlpar}.ipynb`,
`specs/2026-08-16-constitution/upstream-issue.md`,
`specs/_research/plan-upstream-merge-0.13.1.md` (stays untracked) or
`AGH__Si_indent_1_512x672.h5oina` as staged; `git stash list` still
shows `stash@{0}`; `head -c 3 specs/roadmap.md` is `# R` (no BOM);
`git rev-parse develop hrebsd-dic feat-spherical-indexing
feat-spherical-indexing-nlpar` unchanged until the merge (de27741a,
b64cc18f, 6723aaf0, e49b3d85).

**Known flakes** (never block a gate; re-run alone): the numba-cache
rule (tech-stack HROSM section) and the two NLPAR-recorded flakes
(`test_refine_orientation_projection_center_local_nlopt`,
`TestCalculateMasterPattern::test_shape`).

## 6. Adversarial review and mutation list

Each mutant is applied ALONE in the main tree (the editable install's
`.pth` points at the main `src`), in sequential batches of three, and
must die by its NAMED killer, whose test id lives in `validation.md`
"Mutant killers" (the naming authority; not duplicated here). A
renamed test or helper renames its row in both files in the same
commit. "Killer modules" below says where the killers live (`kam`,
`seg`, `avg`, `smp`, `osm`, `reg`, `sig` as in `validation.md`; the
EMsoft-file arms are all in `reg`; corrected 2026-10-06, spec review);
"V" their arm. Stage A unless marked (B).

| # | mutant: exact code change | killer modules, V |
|---|---|---|
| M1 | `_kam._emsoft_neighbour_sum`: `down[1:] = v[:-1]` -> `down[:] = v` (vertical pair credited to the pixel above, `t - W`, the correct index) | `kam` V1; `reg` V2 (shipped DI `KAM`) |
| M2 | `_kam._emsoft_pair_angles`: `spurious = 0.0` (the `jj` fixed: pixel (W, 1) no longer compared with the identity) | `kam` V1; `reg` V2 |
| M3 | `_kam._emsoft_edge_multipliers`: `acc[0] *= 4.0` -> `acc[0] = third(acc[0])` (corner (1, 1) treated as an edge) | `kam` V1 |
| M4 | `_kam._emsoft_pair_angles`: the spurious angle taken against `q[0, 0]` instead of the identity quaternion | `kam` V1; `reg` V2 |
| M5 | `_kam.kernel_average_misorientation_map` compat: `np.float32(np.float64(kam32) * RTOD)` -> `kam32 * np.float32(RTOD)` | `kam` V1 (degrees rounding); `reg` V2 (shipped DI `KAM`) |
| M6 | `_kam._correct_kam`: `total / count` -> `total / 4` | `kam` V3 (edge and corner pixels of the analytic field) |
| M7 | `_kam._correct_kam`: pair angle over `operators[:1]` (identity only) | `kam` V3 (orix `angle_with` on symmetry-scrambled maps) |
| M8 | `_segmentation._kam_difference_labels`: offsets `((0, 1), (1, 0), (1, 1), (1, -1))` -> `((0, 1), (1, 0))` | `seg` V4 (diagonal chain) |
| M9 | same: edge test `np.abs(a - b) <= t` -> `(a <= t) & (b <= t)` | `seg` V4 (chained gradient) |
| M10 | same: the requirement that a component contain a pixel with `kam <= threshold` dropped | `seg` V4 (seed above the threshold) |
| M11 | same: `size >= 2` -> `size >= 1` | `seg` V4 (singleton) |
| M12 | same: the difference in float64 (`np.abs(a.astype(np.float64) - b.astype(np.float64)) <= float(threshold)`) | `seg` V4 (constructed float32 tie at exactly the threshold) |
| M13 | `_segmentation._dilate_emsoft`: (a) `out[1:, 1:] = ...` -> `out[:, :] = np.where(M != 0, M, g)` (first row and column dilated); (b) `M[1:, 1:] != 0` -> `(M[1:, 1:] != 0) & (g[1:, 1:] == 0)` (no overwrite) | `seg` V6 (border and overwrite arms); `reg` V12 (`center_dilate`) |
| M14 | `_segmentation._dilate_correct`: `M` written into every present pixel with `M != 0` (labels overwritten) | `seg` V6 (labels invariant) |
| M15 | `_averaging._center_pixel` compat: `(x0 + w // 2, y0 + h // 2)` -> `(x0 + (w - 1) // 2, y0 + (h - 1) // 2)` | `avg` V6 (even widths); `reg` V12 (`center` `avor`) |
| M16 | `_sampling._cubochoric_grid`: `np.arange(-N, N + 1)` -> `np.arange(-N, N)` (`(2N)**3` points) | `smp` V7 (count, shells) |
| M17 | `_sampling.misorientation_ball` (and the driver's composition): `~ball * center` -> `center * ~ball` | `smp` V7 (Rodrigues-formula composition, shipped N 6 `qu` order) |
| M18 | `_sampling._cubochoric_grid`: `edge = 0.5 * (np.pi * (w - np.sin(w))) ** (1 / 3)` -> `0.5 * (np.pi * w**3 / 6) ** (1 / 3)` | `smp` V7 (max angle exactly `w`, shell angles, N 6 lists) |
| M19 | `_directional_statistics._em_correct`: E-step centres `S_j * mu` -> `mu * S_j` and M-step `conj(S_j) * x` -> `x * conj(S_j)` | `avg` V8 (left-scrambled recovery) |
| M20 | same, Watson: `f(t) = t**2` -> `f(t) = t` | `avg` V8 (Watson recovery with random sign flips; left-scrambled Watson recovery `alpha100-n300`) |
| M21 | `_averaging._apply_kappa_gate`: `kappa > min_kappa` -> `kappa >= min_kappa` | `avg` V9 gate boundary (kappa equal to `min_kappa`) |
| M22 | `_directional_statistics._em_emsoft`: the "keep the previous `Q`, `L` when `min(Phi) <= 0`" branch removed | `avg` V9 (unscrambled Watson at kappa 1e4: an init whose iteration-2 `Phi` underflows exits at `i = 2`; round 2) |
| M23 | `_averaging._mean_orientation`: variant alignment skipped (sign-fixed mean of the raw quaternions) | `avg` V8 (scrambled mean) |
| M24 | `_osm._osm_emsoft`: result divided by `n` | `osm` V10 (transcription); `reg` V10 (shipped `OSM`) |
| M25 | `_osm._osm_emsoft`: `third(x) = (x * np.float32(4)) / np.float32(3)` -> `x * np.float32(4 / 3)` | `osm` V10 (test-local transcription in CI); `reg` V10 GRX810/Al (local) |
| M26 | `_osm._osm_grain_aware`: the same-grain mask dropped | `osm` V11 array arm; (B) `test_orientation_similarity_map.py`, V11 public arm |
| M27 (B) | `_driver`: skip test `n_pixels < min_pixels` -> `n_pixels <= min_pixels` | `sig` V16 (a grain of exactly `min_pixels` is re-indexed) |
| M28 (B) | `_driver`: `present` built from `navigation_mask` instead of `~navigation_mask` | `sig` V16 (mask polarity; no `np.empty` leak) |
| M29 | `_emsoft_file`: `np.deg2rad` applied to `EulerAngles`, `RefinedEulerAngles` and `newEuler` (radians read as degrees) | `reg` reader arm (default); `reg` local V2/V12 reinforce; (B) `reg` V13 (weekly) |
| M30 (B) | `_driver`: `ball_g = ball_identity` (ball not composed with the grain average) | `sig` V15 (step recovered) + V16 (each grain matched against its own ball) |

Supplementary mutants (switches and contracts the parked list did not
cover):

| # | mutant: exact code change | killer modules, V |
|---|---|---|
| S1 | `_directional_statistics._final_representative` correct mode: `S_i * mu` -> `mu * S_i` | `avg` V8/V9 (disorientation to the truth under left symmetry) |
| S2 (B) | `_driver` compat domain = the grain's pixels (box dropped) | `sig` V16 (block size per grain) |
| S3 (B) | `_driver` compat fill = NaN OSM/scores and the input rotation | `sig` V16 (compat fill) |
| S4 | `misorientation_ball(emsoft_compatible=True)`: `_emsoft_rodrigues_round_trip` skipped | `smp` V7 (compat ball == float32-Rodrigues transcription bitwise, differs from correct) |
| S5 (B) | `_dictionary_indexing(verbose=False)` still prints the matching line | `test_dictionary_indexing.py` (`capsys`) |
| S6 | `_coverage_warning_message`: `max_grod > max_angle` -> `>=` (standalone call site) | `avg` V8 (GROD boundary) |
| S7 (B) | `orientation_similarity_map` at the default keywords routed to `_osm_grain_aware` | `test_orientation_similarity_map.py`, V11 (a custom footprint and `from_n_best` still run on the legacy path) |
| S8 | `average_grain_orientations`: `default_rng(seed)` re-created per grain | `avg` V8 (one generator across grains) |

D20 mutants (requirements D20.5; added 2026-10-06, spec review):

| # | mutant: exact code change | killer modules, V |
|---|---|---|
| S9 | `misorientation_ball_spacing`: `4 * np.arcsin(d / 2)` -> `2 * np.arcsin(d / 2)` | `smp` V7 (brute force at N 1-3; default pin) |
| S10 | `misorientation_ball_spacing`: `query(q, k=2)` and column 1 -> `query(q, k=1)` and column 0 (self distance, spacing 0) | `smp` V7 (brute force; monotonicity) |
| S11 | `_grod_map`: angle without symmetry reduction (`operators[:1]`) | `avg` V8 (variant-scrambled pixels give the same GROD; orix `angle_with` loop) |
| S12 | `_average_grains` compat `"center"`: GROD reference = the grain pixel nearest the centroid instead of the K4 box-centre pixel | `avg` V8 (compat centre GROD measured from the box-centre pixel) |
| S13 (B) | `_driver`: the coverage warning issued after the first grain's `get_patterns` call instead of before the ball | `sig` V16 (spy order) |
| S14 (B) | `_driver` call site of `_coverage_warning_message`: strict `>` -> `>=` | `sig` V16 (silent at equality) |

K2 and K11 mutants (added 2026-10-06, spec review: no row covered them):

| # | mutant: exact code change | killer modules, V |
|---|---|---|
| S15 | `_emsoft_quaternions.emsoft_disorientation_angle` (compat): `np.arccos(x)` -> `np.arccos(np.clip(x, 0, 1))` | `kam` V1 (duplicated orientations vs the unclipped transcription) |
| S16 | `_kam._dot_to_angle`: the near-one snap dropped (`2 * np.arccos(d)` directly; redefined 2026-10-06, spec review round 2, as the snap replaced the clip) | `kam` V3 (identical neighbours, the above-1 rotation: NaN instead of 0) |
| S17 | compat KAM accepts absent points or several phases (the K11 checks removed) | `kam` V1 (`test_rejects_absent_points_and_several_phases`); (B) `sig` V16 (validation order); (B) `sig::TestValidation::test_compat_input_is_checked_before_any_work[absent, not_indexed, masked, two_phases]` (added 2026-10-07, Stage B close gate: the driver repeats the checks as its first work step, so only a spy on the KAM and `get_patterns` separates them; the variant without the "one phase" raise survived the order arm alone; amended 2026-10-07, spec re-review) |

Round-2 mutants (added 2026-10-06, spec review round 2):

| # | mutant: exact code change | killer modules, V |
|---|---|---|
| S18 | `_kam._dot_to_angle`: the snap replaced by the old clip `2 * np.arccos(np.clip(d, 0, 1))` | `kam` V3 (identical neighbours, the below-1 rotation: ~3e-6 deg instead of 0); `avg` V8 (correct centre pixel GROD exactly 0) (amended 2026-10-07, Stage A close gate: the below-1 case is a SYMMETRY-EQUIVALENT pair; a stored rotation's self-dot never rounds below 1 in the production sum, measured on 1000 seeded rotations, while 213-243 of 23,000 rotation/operator pairs do; the killers build such pairs) |
| S19 | `_grains._map_grid`: grid from the in-data `xmap.row.max() + 1`, `xmap.col.max() + 1` with the `xmap.size == 1` shortcut (the drafted rule) | `avg` V8 (grid spans points not in the data); (B) `sig` V16 (map from a navigation-masked `dictionary_indexing`) |
| S20 (B) | `_dictionary_indexing(verbose=False)` keeps the `sleep(0.2)` | `test_dictionary_indexing.py` (`sleep` spy not called) |
| S21 | `_directional_statistics._em_correct` VMF: operator set `G` instead of `G+- = {S_j} u {-S_j}` (added 2026-10-06, Stage A failing-tests gate; requirements D5.4 antipodal amendment) | `avg` V8 (`test_vmf_treats_q_and_minus_q_as_one_orientation`; all-24-operator left-scrambled VMF recovery) |
| S22 (B) | `EBSD.hrosm` compat check ignores `navigation_mask` (a strengthening found during the build, entry 26 item 2; amended 2026-10-07, spec re-review) | `sig::TestValidation::test_compat_input_is_checked_before_any_work[masked]` (the KAM spy is called) |

Review scope beyond mutants: the fidelity reviewer re-reads
`mod_DIsupport.f90:172-275, 358-466`, `mod_cluster.f90:134-317,
403-562`, `mod_so3.f90:1569-1635, 2118-2166, 4117-4219`,
`mod_dirstats.f90:805-1285`, `mod_Lambert.f90:1022-1113, 1376-1432`,
`mod_quaternions.f90:53-64ff, 1078-1126, 2542-2746` and
`mod_HROSM.f90:428-760` against D3-D9 and the K table, checks the
float32/float64 promotion chain of every compat path (it decides
bitwise parity), and confirms that every "Differences" item in the
`EBSD.hrosm` Notes is a recorded K switch or deviation.

## 7. Open questions

### 7.1 The seven recorded defaults (R1-R7), approved by Johan 2026-10-06 (all defaults; R2, R3 confirmed explicitly; ledger entry 4)

Adopted provisionally (requirements D19); approving this file approves
them unless Johan names one to switch.

| R | default adopted | alternative | consequence of switching |
|---|---|---|---|
| R1 | `EBSD.hrosm` returns ONE `CrystalMap` with broadcast per-grain props, plus `GrainTable.from_crystal_map` | `tuple[CrystalMap, GrainTable]` | an extra return value; fewer props |
| R2 | `EBSD.hrosm(average="mean")` (deterministic, symmetry-aware); parity tests pass `average="center"` | EMsoft's default `"center"` | the default follows one pixel (in compat possibly outside the grain) |
| R3 | `grain_id` uses EMsoft's labels: 0 unassigned, grains 1..n | `hrebsd-dic`'s 0-based labels with -1 | a conversion in every EMsoft comparison; same convention as HREBSD's `grain_id` |
| R4 | multi-phase v1: one master; other phases' grains skipped with one warning | `dict[str, EBSDMasterPattern]` | a multi-phase driver and tests in Stage B |
| R5 | EMsoft's right-side symmetry in the VMF/Watson steps (K5) is a defect, reproduced in compat only | treat it as a convention in both modes | correct-mode VMF/Watson would fail the left-scrambled recovery |
| R6 | EMsoft's KAM-difference segmentation in both modes; a misorientation rule is a later follow-up | misorientation rule in correct mode | correct grains diverge from EMsoft's; a new oracle needed |
| R7 | backwards-compatible keywords on upstream code: `_dictionary_indexing(verbose=)`, `orientation_similarity_map(grain_id=, emsoft_compatible=)` | private copies, upstream code untouched | duplicated code; no public EMsoft-compatible OSM for DI maps |

### 7.2 Decided by default (confirmed by the named measurement; Johan may override)

8. **Reference binary** (D13.4). Default the develop-HEAD build
   `C:/Users/westraadt.1/Software/EMSOFT/EMsoftOO/build-ifx-release/Bin/`
   (6_0_20260525_0, commit c127868, the source the K table cites;
   `EMsoftConfig.json` points there; writes `newQuat`). Alternative the
   Release/Bin build (6_0_20260413_0, 975a1fc, DLL md5s in D13.4).
   CONFIRMED 2026-10-06 (entries 7, 11): c127868 used for every shipped
   file.
9. **Compat OSM edge multiplier order on that binary** (D7.3). The
   Windows ifx Ni6 run matched `x * float32(4/3)`, GRX810 and Al `(x *
   4) / 3`. The code keeps `(x * 4) / 3` (the Fortran text, two builds
   of three); if the chosen binary folds, the shipped-reference V10
   arm pins "bitwise except the straight-edge pixels, count pinned",
   and the test-local transcription stays M25's CI killer. Measured by
   the measurer on the reference `OSM`. Alternative: follow the chosen
   binary (breaks GRX810/Al bitwise parity).
   MEASURED 2026-10-06 (entry 8 item 3, entry 12): the c127868 build FOLDS
   (`x * float32(4/3)`); the shipped V10 arm pins `SHIPPED_OSM_DIFF =
   {"OSM": 79, "OSM_05": 87}` straight-edge points at 1 ulp, each
   reproduced by the folded form; code keeps `(x * 4) / 3`; M25's CI
   killer is the test-local transcription.
10. **Reference-file sizes** (D13.3; DECIDED 2026-10-06, spec review,
    D13.3 amended). The drafted list put the di file at ~280 kB
    (`TopMatchIndices[:, :10]` int32 165 kB + `EulerAngles` 49.5 kB +
    four (55, 75) float32 maps 66 kB), above the 250 kB cap, so the
    layout is now fixed, not conditional: (a) `EulerAngles` lives in
    `regression_hrosm_large_refined.npz` (di ~231 kB, refined ~116 kB);
    (b) `newQuat` is not shipped (the float32 `eq_` of `newEuler`;
    checked in V14 only), nor `maxGROD`; each scenario file ~118 kB;
    (c) the ball file ships `qu` and `ro` only (~141 kB; `eu` checked by
    the bin arm). Total seed ~841,000 B, pinned as measured
    (`REFERENCE_TOTAL_BYTES`); the `center_dilate` re-indexing arrays
    stay (V12 uses them).
11. **Ni6 as an end-to-end oracle** (D13.5 leaves it open). Default no:
    the EMHROSM run's patterns (`EDAX-Ni.h5`, 103 MB) are local and its
    master was regenerated on 2026-05-23, after the run, so its inputs
    cannot be reproduced; Ni6 stays the local KAM/segmentation/WAT and
    DI-OSM oracle (D13.5). Alternative: a weekly-local tolerance arm
    with the regenerated master.
12. **Chen et al. (2015), IEEE Signal Process. Lett. 22, 1152-1155**
    (D16.4). Default: add `chen2015parameter` alphabetically after
    `chen2015dictionary` (`bibliography.bib:43-54`), DOI verified first,
    cited in `average_grain_orientations`; conflict-free on the replay
    (staging hunks at `:251` and EOF), one alphabetical resolution on
    the `hrebsd-dic` merge (1.2). Alternative: no entry, the method
    credited through EMsoftOO only.
    OPEN DISCREPANCY (recorded 2026-10-07, spec re-review; section 13 row
    A18): the built tree has neither the entry nor the citation and no
    ledger entry records a change. The main loop resolves it before commit
    6: (i) implement this default, or (ii) record here and in requirements
    D16.4 "Not added (decided 2026-10-07, <who>): the method is credited
    through EMsoftOO only (the item 12 alternative)", with a ledger entry
    either way. **Resolved 2026-10-07 (main loop, before commit 6): option (i)**: `chen2015parameter` (Chen, Wei, Newstadt, De Graef, Simmons, Hero, IEEE Signal Process. Lett. 22(8), 1152-1155, 2015, DOI 10.1109/LSP.2014.2387206, verified) added after `chen2015dictionary` and cited in `average_grain_orientations`; the bibliography is therefore touched: no conflict on the clean replay, one alphabetical conflict expected on the `hrebsd-dic` merge (its entries sit in the same `chen2015dictionary`..`foden2019indexing` gap).
13. **orix 0.12.1 round trip of the bool and 2-D props** (D2.3).
    VERIFIED 2026-10-06 (spec review; orix 0.12.1, numpy 1.23.0,
    isolated py3.10): bool `(12,)`, float32 `(12, 5)`, float64 `(12, 4)`
    and int32 `(12,)` props survive `orix.io.save`/`load`. No version
    gate.
14. **V13 placement** (amended 2026-10-06, spec review, for the CI
    budget of 1.4). The one-grain arm (the smallest grain with `W*H >=
    32` and `n_pixels >= min_pixels`, `n_steps=10` since the D13
    `nsamples` amendment) is
    `@pytest.mark.weekly`; the full map is local + weekly. M29 and M30
    keep default-suite killers. CONFIRMED 2026-10-07 (entry 21): one
    grain = label 23 (20 points), median 0.0, p99 0.4999 deg.
15. **Tutorial runtime.** The executed cells use the defaults if nbval
    of `hrosm.ipynb` takes <= ~5 min on this machine (MTP); otherwise
    `n_steps=10` (9,261 orientations) in the executed cells, the N 20
    numbers quoted from the ledger. Outputs stored iff > ~2 min on the
    RTD builder.
    DECIDED 2026-10-07 (Stage C, entry 28): the real-map cell uses
    `n_steps=10` (21.8 s) rather than the defaults (186.3 s); with it,
    nbval takes 62-66 s quiet and 148-228 s loaded. The synthetic
    demonstration runs the defaults. Outputs are stored (Read the Docs
    estimate 4-11 min).
16. **Compat KAM input route** (DECIDED 2026-10-06, spec review; D3.2
    amended). The drafted seed "`from_euler == eq_` bitwise" is refuted:
    orix differs from `eq_` by 1 float64 ulp in 8,930 of 28,086 Ni6
    `RefinedEulerAngles` rows (8,404 `EulerAngles`). Compat therefore
    converts `xmap.rotations` (first per point) through `to_euler()` ->
    float32 -> `eq_`; the round trip `Rotation.from_euler(eu32.astype(
    np.float64)).to_euler().astype(np.float32) == eu32` holds bitwise
    on every row of Ni6 (28,086 x 2), GRX810 (139,181) and Al (501,592),
    and V1 pins it on the shipped angles.

### 7.3 Inconsistencies found in `requirements.md` (items 17-19 and 21 fixed by dated amendments in spec review round 1, 2026-10-06; 20, 22, 23 recorded)

17. FIXED (D13.3 amended): D13.3 listed `EulerAngles` in
    `regression_hrosm_large_di.npz`, which put that file at ~280 kB,
    above its own 250 kB cap (item 10).
18. FIXED (D16.4, D14.4 amended): D16.4 said an HROSM bibliography
    entry at EOF would conflict on the replay; the file is alphabetical
    (except the `aanes*` block), so the entry goes after
    `chen2015dictionary`: no replay conflict, but one alphabetical
    conflict on the `hrebsd-dic` merge (item 12).
19. FIXED (D14.5 amended): D14.5 ran the clean-replay grep over `doc`,
    whose stored notebook outputs (base64 PNG) produce false hits for
    `[^A-Za-z0-9_][DVKMR][0-9]+`; the gate excludes `*.ipynb` and checks
    the notebook's cell sources with `nbformat` (section 5).
20. D15.4's oldest recipe selects `-k hrosm`; the two appended test
    modules are still selected because their classes and tests carry
    "hrosm" in their names (`TestHROSMKeywords`, `..._for_hrosm`; `-k`
    is case-insensitive); the gates use the explicit `$B_TESTS`
    selection anyway.
21. FIXED (D1.2 amended): D1.2 asked for See Also cross-references
    between `segment_grains` and `segment_grains_kam`; `segment_grains`
    exists only on `hrebsd-dic`, so the cross-reference is a
    `hrebsd-dic`-only commit after the merge (1.2).
22. Scope puts both OSM forms in Stage A as array functions, while the
    parked plan and the task outline put the correct OSM in Stage B;
    this plan follows `requirements.md` (array functions in Stage A,
    the public routing and the driver use in Stage B).
23. D13.6 seeds the reference run at 10-15 min; EMDI + EMFitOrientation
    + three EMHROSM runs at ~5 min each seed ~20-25 min (MTP either
    way). MEASURED 2026-10-06 (entry 11): 2,148.8 s (35.8 min) at
    `nsamples 10`, EMDI alone 1,382.6 s. Both seeds were low.

### 7.4 For Johan

24. DECIDED 2026-10-06 (approval): default taken. **Approval scope.** Approving this `plan.md` also approves the
    section 0 texts as applied to the working tree and commit 1 of all
    three spec documents, without a second round. Default: yes.
25. DECIDED 2026-10-06 (approval): default taken. **Fan-out timing.** After the merge on your go, the `hrebsd-dic`
    merge and the clean replay run immediately, in order, stopping at
    the first red gate. Default: yes.
26. DECIDED 2026-10-06 (approval): default taken. **Weekly CI on the fork** (added 2026-10-06, spec review round 2).
    `weekly.yml` is `disabled_inactivity` (last run 2026-08-10; the
    notebook job failed on the last three runs, `weekly-tests` passed
    in ~4 min). Default: leave it disabled; the `--weekly` arms and
    nbval of `hrosm.ipynb` run in the local stage gates and their
    runtime is a local measurement plus a scaled estimate (1.4, D15.1).
    Alternative: `gh workflow enable weekly.yml --repo
    jwestraadt/kikuchipy`, then `gh workflow run weekly.yml --repo
    jwestraadt/kikuchipy --ref feat-HROSM` once per stage push and the
    `weekly-tests` time recorded; the notebook job would stay red for
    reasons unrelated to HROSM until fixed separately.
27. DECIDED 2026-10-06 (approval): default taken. **Sliced crystal maps** (added 2026-10-06, spec review round 2;
    D1.9). The map grid spans every point of the map (orix's original
    grid), so a map from `dictionary_indexing(navigation_mask=...)`
    with absent edge rows aligns with its signal. Consequence: a map
    sliced with `xmap[r0:r1, c0:c1]` keeps the original grid, so
    `s.hrosm(xmap_sliced)` works on the unsliced signal (the rest is
    absent) and raises "xmap shape" on `s.inav[...]`. Default: that.
    Alternative: when the grid differs from the signal, fall back to
    the bounding box of the points in the data if it matches (one more
    branch and test).

## 8. Commits

Signed (`git commit -s`), explicit pathspecs only, the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (always the
one the session specifies), no em-dashes in messages, `SKIP=
licenseheaders uvx pre-commit run --files <explicit non-specs files>`
before each code commit, `roadmap.md` line 1 checked for a BOM after
every spec-touching commit. Between workflows the tree is clean apart
from the untracked `AGH__Si_indent_1_512x672.h5oina` and
`specs/_research/plan-upstream-merge-0.13.1.md`.

1. **"Add HROSM spec and constitution amendments"** --
   `specs/2026-10-06-hrosm/{requirements,plan,validation}.md`, the
   section 0 appends to `specs/{mission,roadmap,tech-stack}.md`, the
   status header of `specs/_research/plan-hrosm-2026-09-28.md`. Only
   after Johan approves this plan; pushed with commits 2-3.
2. **"Add HROSM Stage A skeletons and failing tests"** -- the `_hrosm/`
   stubs, the script skeleton, exports, `pyproject.toml`, the conftest
   section, the six test modules, ledger entry. Never pushed alone.
3. **"Implement HROSM Stage A: KAM, grain segmentation and averaging,
   misorientation ball, OSM and EMsoft references"** --
   implementation, the shipped `.npz` files and registry rows, measured
   pins, review fixes, CHANGELOG bullet (link #20 until confirmed),
   `bibliography.bib` if item 12 holds, roadmap Stage A ticks, ledger
   entries, disposition rows here. Push 1 + 2 + 3.
4. **"Add HROSM Stage B failing tests: per-grain re-indexing and
   EBSD.hrosm"** -- stubs, `test_ebsd_hrosm.py`, the additions to the
   two existing modules, the generator. Never pushed alone.
5. **"Implement HROSM Stage B: per-grain re-indexing and
   EBSD.hrosm"** -- driver, method, keywords, pins, performance
   baselines, CHANGELOG bullet extended, roadmap Stage B ticks. Push 4
   + 5.
6. **"Add HROSM tutorial notebook and gallery example"** -- notebook,
   `index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg` if touched,
   `examples/indexing/{README.rst,hrosm.py}`, CHANGELOG tutorial
   bullet, roadmap Stage C ticks, and the spec re-review amendments
   (the re-review runs BEFORE this commit, section 4). Push; the PR is
   opened.
7. **"Tick HROSM boxes in roadmap (jwestraadt/kikuchipy#20)"** -- on
   `feat-HROSM` after the number is confirmed (CHANGELOG links
   rewritten if it is not #20); ticks "PR opened".

After the merge (section 1): on `hrebsd-dic` the merge commit "Merge
develop (HROSM) into hrebsd-dic" and "Cross-reference the KAM and
misorientation grain segmentations"; on
`feat-spherical-indexing-hrosm` the two replay commits; on `develop`
one fork-local commit "Tick HROSM fan-out boxes in roadmap" (the
242bfcb4 precedent).

## 9. Definition of done

- Per code stage: failing tests committed first; implementation;
  adversarial review (two reviewers + two skeptics) and bug injection
  (every section 6 mutant of the stage dead by its named killer, or
  recorded reviewed-only with the reason) + fixes; pre-commit clean on
  the explicit list; coverage 100 % of `_hrosm/` (Stage A: its
  modules) from the default suite with the command output recorded;
  `-n 0` then `-n 4` green (red tests re-run alone, flakes
  attributed); full suite green; doctests green; oldest-matrix run
  recorded; clean-replay grep empty; budget (CI-style <= 25 s, serial
  <= 60 s) recorded; on-push CI job times recorded (each <= 18 min);
  never-sweep check clean; signed commits pushed together; roadmap
  stage boxes ticked.
- Stage A: the reference run done once with pre-flight and acid bands
  recorded; V14 regenerate-and-diff run once per shipped file with the
  program and DLL md5s; the bin arm V7 N 20 and the local arms (Ni6,
  GRX810; Al weekly-local) run once, outcomes in the ledger.
- Every MTP placeholder in `validation.md` replaced by a dated measured
  value (recipe, machine); every seed of 2026-09-28 or 2026-10-06
  confirmed or amended; refutations of frozen decisions recorded in the
  ledger AND amended in `requirements.md` with the same date (7.3
  items 17-23 included).
- `EBSD.hrosm` and the eight public names documented (numpydoc,
  `:cite:`, Notes with the differences from EMHROSM in words), in the
  API reference; `git grep -n -E "Optional\[|Union\[|print\(" --
  src/kikuchipy/indexing/_hrosm tests/test_indexing/test_hrosm_*.py
  tests/test_signals/test_ebsd_hrosm.py` finds only `verbose`-guarded
  prints in `_driver.py`.
- Tutorial registered and nbval-wired; `sphinx-build -b html` exit 0;
  the gallery example renders; CHANGELOG bullets (feature, tutorial)
  with the confirmed fork link.
- Open questions 8-16 confirmed by their measurements with dated
  records; R1-R7 and 24-27 answered by Johan (2026-10-06, entry 4;
  approval of this file).
- The three spec documents re-submitted to review after Stage C, the
  findings folded in (disposition rows appended below).
- `git log origin/feat-HROSM..feat-HROSM` empty; PR `feat-HROSM ->
  develop` open with ubuntu/windows CI green (macOS red attributed to
  `test_ni_proper_oh_count`), every CI job time recorded. The merge
  and the fan-out are tracked in the roadmap, not in this definition.
- Memory notes updated after the fan-out (`hrosm-parked-plan.md`,
  `feat-spherical-indexing-branch.md`, `hrebsd-dic-project.md`, the
  index `MEMORY.md`).

Section 10 onward: spec-review and stage-review disposition tables,
appended by the fixers.

## 10. Spec-review disposition table (2026-10-06, fixer, round 1)

Three critics, 46 findings; each re-verified by the fixer (probes and
source reads in `validation.md` Recorded results entry 2). Totals: 44
fixed, 2 partly, 0 rejected. Duplicates are fixed once and
cross-referenced.

| id | severity | disposition | what changed or why |
|---|---|---|---|
| F1 | blocker | fixed | D20 carried into plan (2.7 spacing, 2.9 GROD map + helpers, 2.11 eight exports, 1.2 merged `__all__`, 0.1/0.2/2.14/9 "eight") and validation (V7 `TestBallSpacing`, V8 `TestGROD`, V16 coverage arms, D20 mapping row, MTP names); mutants S9-S14 with killers; constitution blocks re-applied (byte-equal, no BOM) |
| F2 | major | fixed | Section 3 item 3 reordered to D8.2/D20.3: private core `_average_grains` (never warns) -> skips -> spacing -> one coverage warning over re-indexed grains -> info line -> ball; 2.9 helper contract; V8 warning test rewritten (descending max GROD, "misorientation ball") |
| F3 | blocker | verified, fixed | Legacy OSM raises "footprint.ndim" on (1, 6), (6, 1), (1, 1); V11 arm now (2, 5) plus custom footprint and `from_n_best`; S7 "why it dies" restated; D7.5 records the limitation |
| F4 | major | verified, fixed | New D1.9 `_map_grid` (in `_grains.py`, hrebsd-dic precedent); D1.5 shape check and D3.7 use it; plan 2.5/2.9; V1 note and V8 `test_map_grid_restores_one_row_one_column_and_one_point_maps` |
| F5 | major | verified, fixed | Master copy's two `xtalname` datasets (vlen ASCII) rewritten to `Ni.xtal`; P2b asserts `Xtal/Ni.xtal` SG 225, a 0.35236 nm; `master_run_md5` in provenance and V14; fallback amends P3/V14 pin in the same ledger entry (D13.2.2, 2.12) |
| F6 | major | verified, fixed | Every path key of every namelist prefixed `kikuchipy_hrosm/<ts>/` (full list in D13.2 and 2.12 *Runs*), `cwd = <run>`, EMsoftData-root new-entry abort, P6; tech-stack bullet updated (merged with E4) |
| F7 | minor | fixed | D8.11 and 3(j): both `_dictionary_indexing` paths return int64; the driver casts |
| F8 | minor | verified, fixed | Compat route frozen as `to_euler()` -> float32 -> `eq_` (D3.2, 2.5, 7.2 item 16); V1 pin is the round trip (exact on Ni6, GRX810, Al); Context bullet corrected |
| F9 | minor | verified, fixed | Al seed 16 of 210,294 (<= 2 ulp), degenerate 9,838 of 291,298; D3.6 fallback <= 2 ulp, count pinned; `AL_KAM_MAX_ULP` added |
| F10 | minor | verified, fixed | V9 cites `mod_dirstats.f90:243-261`; D5.4, 2.8, V9 use `xAp = 0.001 + (i - 1) * 0.001` and Watson `(I1 / (I0 - I1)) / k` (11,504 of 35,000 entries differ from `0.001 i`) |
| F11 | minor | verified, fixed | V7 bin arm: `eu` in degrees (`mod_so3.f90:2588-2590`) via `from_euler(np.deg2rad(eu))` |
| F12 | minor | verified, fixed | D2.3, 7.2 item 13, V16: orix 0.12.1 round trip verified, no version gate |
| F13 | minor | fixed | V0 BSD test joins consecutive `#` lines before matching |
| F14 | minor | fixed | V2 row reads `regression_hrosm_large_refined` `EulerAngles` (see C8/E5) |
| F15 | minor | fixed | (a) D1.5 mask checks and fragments defined (ndarray, bool, shape, not all True; all `ValueError`) and in V16; (b) `sample_generators.py:137` (verified); (c) Ni6 "151 x 186 (H x W)" everywhere; (d) "no overlapping or adjacent hunk" in header, 1.1, D14.4(b) |
| F16 | minor | fixed | D1.2 reworded (`EBSD.hrebsd_dic(grain_labels=)` would accept `grain_id` silently); 1.2 hrebsd-dic-only commit adds the `grain_id - 1` sentence |
| C1 | blocker | fixed | As F1; requirements Scope and D16.5 say eight; header "D1-D20 govern" |
| C2 | blocker | fixed | As F2; verbose=1 info line content (ball size, radius, spacing) in 3(d)/(l) and V16 `test_verbose_levels` |
| C3 | blocker | fixed | As F1; V7/V8/V16 arms named, D20 mapping row, MTP inventory, CI budget seconds; S9-S14 in section 6 and the killer table |
| C4 | major | fixed | One contract (dated D5.8, D1.6, D20.3): `_coverage_warning_message` always carries "misorientation ball" and "max GROD", descending max GROD, ties by label; standalone over all valid grains without spacing, driver over re-indexed grains with spacing; V8 grains with distinct spreads 3.0 + 0.1 k |
| C5 | major | fixed | V1 duplicated-orientation arm (with an asserted non-zero unclipped pair), V3 identical-neighbours arm, V1 point-group test (unmapped name; table patched); S15-S17; whether any real orix group differs is measured at Stage A (fragment kept for unmapped names) |
| C6 | major | fixed | Stage A test list replaced by validation's module table; section 6 column now "killer modules" with `reg` for every EMsoft-file arm; writer (3) owns `test_hrosm_osm.py` + `test_hrosm_emsoft_regression.py` (V0, V2, V10 files, V12, V14) |
| C7 | major | fixed | V0 BSD test over the seven Stage A modules; `_driver` added in the Stage B failing-tests commit (V0, 3 Files, `hrosm-b-tests`, DoD) |
| C8 | major | fixed | D13.3 amended (dated) to the single layout; V1 names the refined file; 7.2 item 10 and 7.3 item 17 recorded as decided |
| C9 | minor | fixed | 2.13 uses `emsoft_data_file(relpath, md5)`; `write_emsoft_layout_file` added to validation's generator list with its signature |
| C10 | minor | fixed | One coverage recipe (default suite only) in section 5 and validation DoD |
| C11 | minor | fixed | D5 Pins (test-local sampler, alpha mapping), D1.3 `seed` type, D5.5 `EMResult`, D8.15 `grains` subset, D3.6 seeds amended (dated) |
| C12 | minor | verified, fixed | `cKDTree` and `angle_with_outer` added to D11.1, D15.4 and the tech-stack verified list (re-applied) |
| C13 | minor | fixed | D14.4(a) gates as 1.2, constitution-ends claim dropped (measured hunks cited); D16.4 bib placement; validation DoD after-merge line |
| C14 | minor | fixed | `E2E_ONE_GRAIN_*` and `E2E_FULL_MAP_*` pins |
| C15 | minor | fixed | Ni6 end-to-end arm "not built (7.2 item 11 default; Johan may override)" in D13.5 and validation |
| C16 | minor | fixed | V0 `test_public_docstrings_link_no_private_name_and_no_spec_id`; D1 mapping row |
| E1 | blocker | fixed | As F1/C1/C3 |
| E2 | blocker | fixed | As F2/C4; V16 order (spy), skip-exclusion, equality and descending-order arms; the spy raises after recording so the arms stay cheap |
| E3 | major | partly | Verified the CI times; D15.1/1.4/tech-stack restated: CI-style <= 25 s binding, 60 s serial ceiling, job times <= 18 min checked per stage push; V13 one-grain arm weekly; performance baselines local ledger runs. Variant: the V13 full map is local + weekly, not CI weekly (minutes even here, over the 5 min weekly line on a CI runner) |
| E4 | major | fixed | As F6; EMFitOrientation `usemasterpatternfile`/`angfile`/`tmpfile`/`inRAM`, EMgetOSM `tiffname`, EMHROSM `OSMtiff`; flip route = EMDI `flipy` (recorded in provenance) |
| E5 | major | fixed | As C8; `REFERENCE_TOTAL_BYTES` seed 841,000 pinned as measured; the `center_dilate` trimming sentence deleted; `maxGROD` never shipped; `newQuat` checked in V14 |
| E6 | major | fixed | As C6/C9; `hrosm_synthetic_signal` placed in Stage B in both documents (its only users are V15/V16) |
| E7 | major | fixed | V8 `test_from_crystal_map_round_trip` (Stage A), V1 point-group test, coverage recipe unified, V0 staged |
| E8 | major | partly | `_EMSOFT_LOCK_TIMEOUT = 3600`, 30 s `os.utime` heartbeat, `_EMSOFT_LOCK_STALE = 300`, same in the script, `TestEMsoftProgramLock` (D13.1, 2.13, tech-stack). PID liveness check rejected: on Windows `os.kill(pid, 0)` terminates the process; the heartbeat alone marks a live holder |
| E9 | major | fixed | Named: `EMResult`, `_apply_kappa_gate`, `_default_n_per_iteration`, `_hrosm(..., grains=None)`, `_grod_map`, `_coverage_warning_message`, `_average_grains`, `_map_grid`; `seed` typed in D1.3 |
| E10 | minor | verified, fixed | `pc = det.pc_emsoft()[0]` in D13.2.3, 2.12, V14 |
| E11 | minor | fixed | Doctest call fixed (`min_pixels=2`, `verbose=0`, `keep_n=5`, `n_osm=5`), stable assertions |
| E12 | minor | verified, fixed | V14 `main(output_dir=tmp_path, bin_dir=...)`, shipped md5s asserted unchanged |
| E13 | minor | fixed | Stage C order docs -> spec re-review -> ticks -> commit 6 -> push -> PR (section 4, commit 6, roadmap box) |
| E14 | minor | fixed | Harden workflows get six ordered phases (injector alone). V9 transcription vectorised over samples and operators with EMsoft's loop order over inits and iterations, default arms at `n_em=3`, `n_em=25` weekly; the "< 3 s" stays a seed measured at the Stage A gate |

Nothing unresolved that blocks approval. Open for Johan with this plan:
R1-R7 (7.1), items 24-25 (7.4), and the two variants above (E3
full map local + weekly; E8 no PID check).

## 10. Spec-review disposition table (2026-10-06, fixer, round 2)

One critic, 13 findings (7 major, 6 minor); each re-verified by the
fixer before editing (probes and source reads in `validation.md`
Recorded results entry 3). Totals: 13 fixed, 0 rejected. Where a
finding offered two fixes, the column says which was taken and why.

| id | severity | disposition | what changed or why |
|---|---|---|---|
| R2-1 | major | verified, fixed | Probe (orix 0.12.1 + numpy 1.23.0 and 0.14.2): row 0 absent -> `xmap.shape` (2, 4); last column absent -> (3, 3); one in-data point of 12 -> (1, 1). D1.9 rewritten: `_map_grid` from `xmap._original_shape` (already read at `_merge_crystal_maps.py:104`) with orix's one-row/one-column rule, -1 where not in the data, no `size == 1` shortcut; the same rule gives (3, 4), (1, 5), (5, 1) on both orix versions. D1.5 (message), D8.1, D8.8 aligned. V8 `test_map_grid_spans_points_not_in_the_data` (three absent patterns, (1, 5)/(5, 1) with an absent end, a sliced map, KAM on the row-0-absent map); V16 `test_navigation_masked_dictionary_indexing_map_keeps_its_grid`; mutant S19. Recorded consequence: a sliced map keeps orix's original grid, so it works on the unsliced signal and raises "xmap shape" on a sliced one (7.4 item 27, alternative named) |
| R2-2 | major | verified, fixed (option b) | Probe: self-dot `< 1` for 4.7-5.0 % (max deficit 2 eps), `> 1` for 33-53 % of random float32-Euler rotations (unscrambled and left-scrambled), up to 3.4150946e-06 deg; orix `angle_with` 0.0 on all. Taken: a frozen near-one snap `d = 1 if d >= 1 - 4 eps` in one helper `_dot_to_angle` (D3.1, K2 row, D10.4, D20.2, D20.5, plan 2.5/2.9, tech-stack numerics bullet), so the exact-zero claims (D20.2, V3, V8) hold and match orix. Why not (a): relaxing to `<= 1e-5` deg would also have had to loosen the 1-ulp scrambled-GROD arm at a centre pixel and drop "its own GROD is 0". The snap makes "clip dropped" equivalent, so S16 is redefined (snap dropped: NaN on the above-1 rotation) and S18 added (old clip: ~3e-6 deg on the below-1 rotation); V3 uses two seeded rotations, one per rounding side; V8 centre-pixel arm uses the below-1 rotation |
| R2-3 | major | verified, fixed | `EMforDS_` (`mod_dirstats.f90:836, 903-905`) declares `Qi`, `Li` once; `getQandL_` (`:1152-1173`) overwrites them whenever `minval(Phi) > 0`. V9 arm rewritten: unscrambled samples, `n_em=3`, a traced transcription; precondition "some init underflows at iteration 2" asserted (seed `WATSON_UNDERFLOW_SEED`, MTP 80); for those inits `Q_2 == Q_1`, `L_2 == L_1`, exit at 2, and `_em_emsoft` has `n_iterations == 2` and `L` = `L_1`; per-init `L` and the chosen init recorded, not asserted. D5.5 Watson sentence amended ("measured"; LEFT-scrambled samples never reach the regime in compat). M22 killer row restated (iteration count); the `alpha1000-n50` transcription arm dropped as an M22 killer (no asserted underflow) |
| R2-4 | major | verified, fixed | Probe (isolated py3.10, numpy 1.23.0): `np.nextafter(np.float32(0.4999), 0)` is `float64`; `float32 array > float(that)` -> `[False]`; with `np.float32(0)` -> `[True]`; `astype(float64)` comparison -> `[True]`. Both: V8 and V16 arms use `np.nextafter(m, np.float32(0))`; D5.8/D20.3 freeze `np.asarray(max_grod).astype(np.float64) > float(max_angle)`; both arms on the oldest-matrix watch list |
| R2-5 | major | verified, fixed | Flat `y * 5 + x`: (1, 1)'s neighbours are 1, 5, 7, 11. Now `absent=(1, 5, 13)`: (0, 0) NaN, (1, 1) the mean over 7 and 11, `(delta_x + delta_y) / 2` |
| R2-6 | major | verified, fixed (option b, a listed) | `gh workflow list --all`: Weekly `disabled_inactivity`; runs 2026-07-27, 08-03, 08-10 failed in `test-documentation-notebooks`, `weekly-tests` green (3 min 52 s - 3 min 59 s). D13.1, D15.1, D15.3, plan 1.4, section 5 `--weekly` line (now with `KIKUCHIPY_EMSOFT_DATA` and `--durations=0`), the tech-stack CI bullet (byte-equal to 0.3, no BOM), validation gates table, budget line, Local-gated, V13, Performance row: weekly arms and nbval of `hrosm.ipynb` run only in local gates; the 5 min line becomes a scaled estimate, never a gate. Re-enabling is 7.4 item 26 (default: leave disabled) |
| R2-7 | minor | fixed | D15.1, plan 0.3 and `specs/tech-stack.md`: "the on-push CI run of each stage push (the Stage B push at the latest, before Stage C)" |
| R2-8 | minor | fixed | DoD Stage B: "V13 (one-grain weekly, full map local + weekly; both run once locally)" |
| R2-9 | minor | verified, fixed | `_dictionary_indexing.py:135` `sleep(0.2)` unconditional (`from time import sleep, time`, `:24`). D8.7 and section 3 item 1 skip it at `verbose=False`; the `test_dictionary_indexing.py` arm asserts a `sleep` spy is never called; mutant S20 (B) |
| R2-10 | minor | fixed | V16 own-ball arm: `max_angle + 1e-4` deg in compat (K8 moves points by <= 2e-7 rad, V7), 1e-9 in correct mode |
| R2-11 | minor | verified, fixed | Base fixture symmetric (A and B 32 px each, same gradient and noise). Arm now specifies `shape=(8, 11)`, grain B one constant rotation `R[111](30) * g0`, noise on A only: A 32 px, B 40 px (asserted), B's max GROD ~0; `min_pixels = n_pixels_A + 1` skips A only |
| R2-12 | minor | fixed | (a) D8.8 uses `(ny, nx)` from `_map_grid`. (b) D2.1 and D4.6 define a label without pixels: `n_pixels` 0, box (0, 0, 0, 0), identity, `phase_id` -1, `kappa` -1.0, `max_grod` NaN, `valid` False, in `average_grain_orientations` and `from_crystal_map`; `n_grains = grain_id.max()` (the highest label never vanishes under compat dilate); V8 round-trip arm asserts the defaults |
| R2-13 | minor | verified, fixed | M20 mutates `_em_correct` (section 6); the compat transcription arm runs `_em_emsoft`. Killers now `test_watson_is_antipodally_symmetric` and `test_recovers_left_scrambled_variants[watson-alpha100-n300]` (a default arm) |

Default-suite seed after round 2: 49 s serial (+2 s, the V16
navigation-masked arm). Nothing unresolved that blocks approval. Open
for Johan with this plan: R1-R7 (7.1), items 24-27 (7.4; 26 and 27
new: re-enabling the fork's Weekly workflow, and sliced crystal
maps), and the two round-1 variants (E3, E8).

## 11. Stage A code-review disposition table (2026-10-07, fixer)

Eleven surviving findings (1 major, 10 minor) plus one refuted by both
skeptics. Every fix was re-verified against the code (and, for F3 and
F4, the EMsoftOO source) before editing; each new killer test was run
against the pre-fix code and failed there (11 failures, restored
after). Six HROSM modules after the fixes, `-n 0`: 293 passed, 31
skipped in 7.14 s; `_hrosm` doctests 8 passed; ruff check and format
clean.

| id | severity | disposition | one line |
|---|---|---|---|
| F1 | minor | fixed | D9.1 ("everything else identical") governs: `"mean"` now uses the orix quaternions and operators in both modes (`_average_grains`); killer `test_mean_is_identical_in_both_modes` (bitwise rotation, kappa, max_grod); `emsoft_compatible` doc says "mean" is the same in both modes |
| F2 | minor | fixed | Correct dilation's present mask is `phase_id >= 0` when `phase_id` is given (D8.1, D4.5), finite KAM only without it; killer `test_correct_dilate_fills_a_present_point_with_a_nan_kam` (and an absent point stays 0); `dilate`/`kam`/`phase_id` docs updated |
| F3 | minor | fixed | Compat `_final_representative` applies EMsoft's float64 `qr_` -> `ra_` -> `aq_` -> normalise round trip (`mod_dirstats.f90:949-961`, `mod_rotations.f90` `qr_`, `ra_`, `aq_`, `rq_`, thresholds 1e-10 and 1e-12) per D5.5 "literal transcription"; the test-local V9 transcription gets the same step (`_emsoft_qr_rq`); killer `test_compat_final_representative_round_trips_rodrigues`; side test now bitwise against the round trip |
| F4 | minor | fixed | Horizontal compat pairs call `_vectormatch(right, left)` as `vectormatch(lnm, cp, lp)` (`mod_DIsupport.f90:235`); the test-local loop oracle had the same swap and is corrected; killer `test_pair_order_follows_the_source_with_duplicates` (1 x 2 and 2 x 2 maps with a duplicate) |
| C1 | major | fixed | CHANGELOG names the four functions that take `emsoft_compatible=True` |
| C2 | minor | fixed | Raises sections on `segment_grains_kam` and `grain_bounding_boxes`; `kam` doc: floating dtype, float64 for an integer KAM map |
| C3 | minor | fixed | `average_grain_orientations` Raises lists `max_angle`, `n_em`, `n_iter`, `min_kappa` |
| C4 | minor | fixed | `GrainTable.kappa` doc covers mean (VMF concentration of the mean resultant length), vmf/watson (EM estimate), center (1.0), rejected (-1.0) |
| C5 | minor | fixed | `from_crystal_map` no longer names the Stage B method; it describes the four properties (to become a `:meth:` link when `EBSD.hrosm` lands) |
| C6 | minor | fixed | `read_local_dot_product_file` and `read_local_hrosm_file` return arrays made read only (`_read_only`, recursive), like `load_reference` |
| C7 | - | refuted-by-both | Not acted on |
| C8 | minor | fixed | CHANGELOG uses "high angular resolution orientation similarity maps (HROSM)", as `_hrosm/__init__.py` |
| C9 | minor | fixed | Banner comment shortened to 72 characters |

Spec clarifications recommended (not made; the fixes follow the
existing text): D5.6 could name the compat Rodrigues round trip
(last-bit change, identity snap below a 1e-10 vector norm); D5.2 could
say compat `"mean"` uses the correct-mode inputs and operators.

## 12. Stage B code-review disposition table (2026-10-07, fixer)

Nine surviving findings (all minor; none refuted by both skeptics, C6
refuted by one). Each was re-verified against the working tree before
editing. The new killers were run against the pre-fix code with the
fix lines replaced by `pass` and failed there (stale mask arm 1,
before-work arms 2; the tiling arm cannot import the pre-fix
helper), restored after. HROSM selection `-n 0`: 435 passed, 34
skipped in 14.31 s; CI-style `-n 4` with pytest-cov, branch: 15.46,
14.52, 14.65 s (median 14.65 s <= 25 s); `_hrosm` and
`_orientation_similarity_map` doctests 8 passed, `ebsd.py -k hrosm`
1 passed; ruff check and format clean.

| id | severity | disposition | one line |
|---|---|---|---|
| F1 | minor | fixed | Driver sets `metric.navigation_mask = None` after `_prepare_metric` (D8 step 5 "no navigation mask"); killer `TestContracts::test_stale_navigation_mask_of_a_metric_is_not_applied` (NCC instance with a map-sized mask; a spy on the matching core asserts mask None and metric size = block size, then stops) |
| F2 | minor | fixed | `_simulation_task_size` replaced by `_simulation_chunks(ball_size, n_per_iteration)`: explicit task sizes tiling every iteration chunk (at most one per CPU, >= 1,024 unless single, near equal), passed as `chunk_shape`; results unchanged (per-pattern projection; chunk invariance arms green); test `TestInvariance::test_simulation_tasks_tile_each_iteration_chunk` (4 cases incl. 68,921 / 17,777 at 16 CPUs) |
| F3 | minor | fixed | `EBSD.hrosm` check 7, after the D1.5 checks 1-6: `_prepare_metric(metric, None, signal_mask, None, False, ball_size)` and `master_pattern._get_master_pattern_arrays_from_energy(energy)`; killer `TestValidation::test_metric_and_energy_are_checked_before_any_work[metric, energy]` (no grain large enough; spies assert KAM and simulation never called); validation cases `metric_unknown`, `compat_before_metric` |
| C1 | minor | fixed | `EBSD.hrosm` Raises lists `NotImplementedError` for a master pattern not in the Lambert projection, and ValueError now names the metric and energy |
| C2 | minor | fixed | Same root cause and fix as F3: an invalid metric now raises ValueError even when no grain is re-indexed |
| C3 | minor | fixed | `orientation_similarity_map` Returns describes the (n rows, n columns) float32 never-squeezed output of the new paths; Raises section added (n_best, keyword combinations, compat points, grain_id); `versionchanged:: 0.14` note |
| C4 | minor | fixed | Banner comments shortened to 72 characters (11 in `test_ebsd_hrosm.py`, 1 each in `test_orientation_similarity_map.py` and `test_hrosm_emsoft_regression.py`) |
| C5 | minor | fixed | `OUTPUT_PROPS` comment says "keep_n", 4 for quaternions, otherwise None |
| C6 | minor | fixed | Kept `_KEYWORD_DEFAULTS` (the merge base of the driver's `**keywords`); `test_signatures_are_frozen` now asserts it equals the keyword-only defaults of `EBSD.hrosm` |

Spec clarification recommended (not made; the fix adds a check after
the listed ones and changes no listed behaviour): D1.5 could list a
check 7, "metric (as `dictionary_indexing`) and master-pattern energy,
before any work", so the ordered list matches the code.

## 13. Post-implementation spec review (2026-10-07)

Spec re-review after Stage C (`hrosm-c-spec-review`: critic, then
this fixer). The critic read the three spec documents against the
built tree (HEAD 953358e2 plus the Stage C working tree) and returned
46 dated amendments, A1-A46. Every amendment was checked against the
code, the tests or the ledger before it was applied (pins against the
test modules' constants, test names and classes by `grep`, the
`ebsd.py` import lines and `hrosm` line, the `tutorials_sanitize.cfg`
EOF sections, `git diff develop -- doc/user/bibliography.bib` empty).
All edits are marked "amended 2026-10-07, spec re-review" or carry
the date of the decision they record. Result: 45 applied (A18 and A41
as recorded open items for the main loop), 1 needing no text change
(A46), 0 rejected. Ledger entry 30 records it.

| id | file(s) | disposition | one line |
|---|---|---|---|
| A1 | requirements | applied | Status 2026-10-07 paragraph inserted; the 2026-10-06 text kept, prefixed "(drafting status, superseded)" |
| A2 | plan | applied | Status 2026-10-07 paragraph after the header; it also names the open Chen item (A18) |
| A3 | validation | applied | Status 2026-10-07 paragraph after the drafting status note |
| A4 | requirements | applied | D1.5 check (7) (metric and energy before any work); masked points count as absent in check (6) |
| A5 | requirements | applied | D4.5 "present" defined for correct-mode dilate (Stage A F2) |
| A6 | requirements | applied | D5.2 `"mean"` identical in both modes (Stage A F1); killer class named `TestCenterPixel`, where the test lives |
| A7 | requirements | applied | D5.6 EMsoft Rodrigues round trip of the compat representative (Stage A F3) |
| A8 | requirements | applied | D8.5 metric navigation-mask reset (Stage B F1) |
| A9 | requirements | applied | D8.6 `_simulation_chunks` replaces the open `chunk_shape` clause |
| A10 | requirements | applied | D8.13 measured costs; the si_wafer/ni_gain pointer goes to the validation Definition of done, where A41 records the gap |
| A11 | requirements, plan 0.3, tech-stack | applied | lock hold ~36-48 min (measured); plan 0.3 and `specs/tech-stack.md` byte-equal again (the mirror also took plan 0.3's CI-budget bullet, amended at the Stage A gates but never copied to tech-stack) |
| A12 | requirements | applied | D13.2 measured acid bands and the script deviations |
| A13 | requirements | applied | D13.3 writer, provenance, measured sizes, `REFERENCE_TOTAL_BYTES` |
| A14 | requirements | applied | D13.4 decided at approval; the used build and md5 prefixes |
| A15 | requirements | applied | D13.6 regenerate-check-only policy and runtime |
| A16 | requirements | applied | D15.1 pytest time, measured medians, Stage B on-push CI |
| A17 | requirements | applied | D16.1 stored-outputs decision and the `n_steps=10` real-map cell; D16.2 `[regex20]`, `[regex21]` |
| A18 | requirements D16.4, plan 7.2 item 12, 1.2, 1.3 | applied as a recorded open discrepancy | the approved `chen2015parameter` default is not built; the main loop resolves (i) or (ii) before commit 6 and records it in the ledger; the bibliography rows say "not touched as built" with the option (i) exception |
| A19 | requirements | applied | D19 heading: approved 2026-10-06 |
| A20 | requirements | applied | D20.1 pinned spacings and the 0.15 s cost |
| A21 | plan 1.2, requirements D14.4(a), plan 0.3 / tech-stack | applied | sanitize EOF conflict now known; bibliography not touched as built; three `_hrosm` import statements at `:59-63`, `hrosm` at `:2564` |
| A22 | plan 1.3 | applied | feature commit: bibliography only under A18 (i); tutorial commit lists `tutorials_sanitize.cfg`; the tutorial CHANGELOG bullet is the first HROSM bullet |
| A23 | plan 1.4, 5 | applied | pytest time and measured medians; nbval runtime measured |
| A24 | plan 2.12 | applied | writer, measured sizes and total, measured timings and acid bands, *Deviations* bullet |
| A25 | plan 2.13, validation | applied | private generators exposed by fixtures without the underscore; `_EMSOFT_LOCK_NAME`, `emsoft_program_lock`, private helpers |
| A26 | plan 3 | applied | 3(f) mask reset, 3(h) `_simulation_chunks`, item 4 check 7, item 6 measured baselines |
| A27 | plan 4 | applied | notebook as built (items 3-5, 7), outputs stored and normalised, gallery 2x2 figure and runtime |
| A28 | plan 5, 4 | applied | `-n 2` under memory pressure; the Stage C gates line points to it |
| A29 | plan 7.1, 7.4, 9 | applied | approval recorded; items 24-27 DECIDED; 24-27 answered |
| A30 | plan 7.2 | applied | items 8, 9, 14, 15 confirmed or decided, with entries |
| A31 | plan 7.3 | applied | item 23 measured 2,148.8 s |
| A32 | validation | applied | pinned-values paragraph after the MTP inventory; removed constants named |
| A33 | validation V12 | applied | Ni6 Watson arm scoping (kappa >= 50, loose count 23, grain 1) |
| A34 | validation V13 | applied | six pins with measured values; full-map facts and runtimes |
| A35 | validation V14 | applied | writer, measured sizes, budget pin 920,000, runtime, regenerate constants, "passed once"; the drafted text moved into a marked history note |
| A36 | validation V15 | applied | pins with measured values, full-size pins, the lowered-contrast note |
| A37 | validation V6, V9, V10, V16, mapping | applied with placement corrections | V16 class lists extended (`test_grains_subset_reindexes_only_the_chosen_grains` is in `TestContracts`); `test_correct_dilate_fills_a_present_point_with_a_nan_kam` and `test_mean_is_identical_in_both_modes` (`TestCenterPixel`) in V6; the two round-trip tests in V9; `test_pair_order_follows_the_source_with_duplicates` in V10 (it lives in `test_hrosm_osm.py::TestCompatOSM`, not V3); mapping rows D1, D4, D5, D7, D8 |
| A38 | validation V9 | applied | old bullet struck and pointed to the amendment; the `EMSOFT_OPERATORS` sign-safe subset recorded |
| A39 | validation V7, V8 | applied | spacing pins; V8 pin column measured, `WRONG_SIDE_MAX_KAPPA_RATIO` 0.1 |
| A40 | validation CI budget, Performance | applied | pytest time and measured medians; performance rows filled; unmeasured rows say "not measured" or "not run" |
| A41 | validation Definition of done | applied as a recorded gap | the main loop appends a ledger entry (si_wafer run or "not run", weekly scaled estimate with its stated basis) before commit 6 |
| A42 | validation Automated, oldest matrix | applied | recorded gates use `$A_TESTS` / `$B_TESTS`; `-n 2` under memory pressure |
| A43 | validation Manual | applied | gallery runtime and 2x2 figure; outputs stored |
| A44 | plan 6, validation mutant killers | applied | S17 (B) killer `test_compat_input_is_checked_before_any_work[...]`; new row S22 (B) |
| A45 | requirements D7.3 | applied | c127868 folds; per-key pins 79 / 87 |
| A46 | plan 0.2, roadmap | no change needed | the box text stays true; the tick cites `[regex20]`, `[regex21]` and stored outputs (entry 28) through ledger entry 30 |
