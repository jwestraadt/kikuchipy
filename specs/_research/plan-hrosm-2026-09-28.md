# PARKED PLAN: `feat-HROSM`: port EMsoftOO's EMHROSM (high angular resolution OSM) into the kikuchipy fork

**Status (2026-09-28): designed, NOT approved, NOT started.** Johan parked it to be implemented **after
the NLPAR implementation** ("Write it to a spec plan separate after NLPAR implementation"). The NLPAR
plan is `specs/_research/plan-nlpar-2026-09-11.md`.

Nothing below has been executed: there is no `feat-HROSM` branch, no commits and no spec folder.

**To resume:**
1. Check that `feat-NLPAR` has been merged into `feat-spherical-indexing`.
2. Re-verify:
   - the base sha (`git rev-parse feat-spherical-indexing`; it was `6723aaf0` before NLPAR);
   - that `specs/` is tracked there;
   - the EMsoft binaries, `EMsoftConfig.json` data path and cached master listed below;
   - the orix version.
3. Start at "Step 0". Step 1 still needs Johan's approval of the spec `plan.md`, and of the
   "Recorded defaults" at the end of this file.

## Context

Johan asked (2026-09-28): "Applying the HROSM analysis from EMsoftOO in kikuchipy". He chose a full
spec-driven (SDD) feature port rather than a recipe.

EMHROSM (`EMsoftOO/Source/DictionaryIndexing/EMHROSM.f90`, `EMOpenCLLib/program_mods/mod_HROSM.f90`,
BSD-3, De Graef 2025) works in four steps:
1. Take a refined dictionary-indexing (DI) orientation map.
2. Cluster it into grains.
3. Average each grain's orientation.
4. For each grain, re-index its patterns against a fine local dictionary (a 5° misorientation ball
   with 0.25° shells, 68,921 orientations), then build a new orientation similarity map (OSM) from
   the top-10 matches.

The result is sub-degree orientation contrast (sub-grains, deformation gradients) that a global
dictionary at about 1.4° spacing cannot resolve.

Johan has worked on this program before:
- His EMsoftOO branch `feature/emhrosm-grod-precheck` adds a correct 4-neighbour KAM (`c85982a`), a
  per-grain max-GROD pre-check with a warning, and a `maxGROD` output (`f6270d0`). None of this is on
  EMsoftOO develop.
- He has earlier EMHROSM runs in `C:/Users/westraadt.1/Software/EMSOFT/EMsoftData/OSM/`.

The port serves the same workflow as his spherical indexing (SI) + NCC + plane-fitted-PC recipe
(earlier in this session): the input orientation map can come from any indexing method.

**Goal:** a kikuchipy-native HROSM behind the fork's SDD process. It must be correct by default and
bit-level faithful to EMsoft behind `emsoft_compatible=True`, proven against EMsoft's own binaries,
and come with a tutorial. The work ends as a PR into `feat-spherical-indexing`.

## Decisions (Johan, AskUserQuestion 2026-09-28)

1. **Scope:** full SDD feature port, including a tutorial.
2. **Branch:** `feat-HROSM` off `feat-spherical-indexing`: the tip after the NLPAR merge, or
   `6723aaf0` (verified 2026-09-28) if NLPAR has not landed. NLPAR brings the fork's `specs/` tree,
   otherwise the Step 0 fallback re-adopts it. The work ends in a PR into `feat-spherical-indexing`,
   merged only on Johan's go.
3. **Fidelity:** correct by default. `emsoft_compatible=True` reproduces the upstream EMsoftOO develop
   binary exactly, including its defects. Every defect is recorded as a numbered switch.
4. **Parity:** algorithmic parity. Clustering, KAM, sampling, averaging and OSM are ported exactly
   and tested bit for bit. The per-grain dictionary step uses kikuchipy-native
   `EBSDMasterPattern.get_patterns` + NCC `EBSD.dictionary_indexing`. EMsoft's hi-pass/adhisteq
   preprocessing and its energy-weighted simulation are not ported. End-to-end comparison with
   `EMHROSM.exe` is therefore tolerance-based.
5. **Averaging:** all three `orav` methods ('center', von Mises-Fisher EM, Watson EM), plus a
   deterministic symmetry-aware mean as the default.
6. **Sequencing:** implement after NLPAR. `feat-NLPAR` lands in `feat-spherical-indexing` first and
   carries the specs re-adoption. `feat-HROSM` then branches off the updated staging branch, and its
   roadmap block goes after NLPAR's.

## Reference facts (verified 2026-09-28; details go into the spec)

### Pipeline (`mod_HROSM.f90:428-760`)

**Inputs and grain loop**
- Reads the dot-product file's `RefinedEulerAngles` (float32, radians).
- Calls `Cluster_T(gangle=5, dilate=F, orav='center', numEM=25, numIter=40)`.
- Loops over grains, skipping those with `kappa == -1` or `npixels < 10`.

**Per grain**
- Sampling ball: `sample_isoCubeFilled(misorang=5, nsamples=20)`, then `SampleIsoMisorientation`
  around the grain average.
- `OSMDIdriver` runs on the grain's bounding box and keeps `nnk` matches.
- The OSM uses `nosm = 10` of those matches.
- For pixels with `grainID == i`, `newOSM`, `newEuler` (radians, not fundamental-zone reduced) and
  `newCI` (maximum normalised dot product) are copied back into the main map.

**Outputs:** `nGrains, grainID, npixels, grainROI, avor, kappa, kam, newOSM, newEuler, newQuat, newCI`
(Fortran `(x,y)`, which h5py/numpy sees as `(H,W)`).

### Clustering (`mod_cluster.f90`)
- It grows regions on the scalar KAM map, not on orientations. KAM is `getKAMMap` of the refined
  angles in degrees, with no threshold.
- Seeds are taken in raster order and must have `kam <= gangle`.
- Growth is 8-connected and chained on `|kam(n) - kam(popped)| <= gangle`.
- Singleton grains are rejected. The final label is 0 (unassigned) or 1..n.
- A closed form is equivalent: connected components of size ≥ 2 containing a pixel with
  `kam <= gangle`, labelled in raster order of their first such pixel.
- `dilate` is a 3×3 maximum filter. It skips the first row and column and overwrites existing
  labels.
- `'center'` picks the pixel at the bounding-box centre, which may lie outside the grain.
- VMF/Watson EM (`mod_dirstats.f90`):
  - It has a symmetry-side mix: the E/M steps and the FZ loop use `mu⊗S`, while `getQandL` uses
    `S⊗mu`.
  - VMF normalisation is wrong (`C = exp(logCp)` inside `exp`).
  - Q/L go stale or uninitialised when `Phi` underflows.
  - `kappa` is kept only if it exceeds 5; otherwise the grain gets `kappa = -1`.
  - The random seed comes from the clock (a seed of 0 gives NaN), and the RNG routines are
    Burkardt's LGPL code (use a NumPy `Generator` instead).

### KAM and OSM share one defect (`mod_DIsupport.f90`: `getKAMMap` 358-466, `getOrientationSimilarityMap` 172-275)
The 1-D loop has three errors:
- It credits the vertical pair to `iii-W+1` instead of `iii-W`.
- `jj` is off by one for the last column.
- Pixel `(W,1)` is compared with the identity; this only affects KAM.

Divisor table: interior /4, edges /3, corners `(1,1)` /1 (so a sum, not a mean), `(W,1)` /2,
`(1,H)` /3, `(W,H)` /2. The OSM is not normalised by `nosm`. Exact NumPy emulations are in the
exploration reports and go into the spec.

### Sampling (`mod_so3.f90`)
- Cubochoric cube with half-edge `0.5*(pi*(w - sin w))**(1/3)` and step `dx = edge/N`, giving
  `(2N+1)^3` points (68,921 at the defaults).
- The shells are exact constant-misorientation shells, and the largest angle is exactly `w`.
- No clipping and no fundamental-zone reduction.
- Rotation to the grain: `q = conj(q_v) ⊗ q0` (left multiplication, Hamilton product,
  `epsijk = +1`). Stored as float32 Rodrigues vectors.
- orix `get_sample_local` is infeasible here: it builds the full SO(3) grid, which is 61 GB at 0.25°.

### OSMDIdriver (`mod_DI.f90:1757-2629`)
- One PC for the whole map; the per-pixel PC code was removed in `92c1b21`.
- **Dictionary thinning defect** at line 2479: if the bounding box has `W*H < Nd` pixels, only the
  first `W*H` of every `Nd` dictionary patterns are simulated.
- The OSM is computed over the whole bounding box, so it includes pixels from other grains.
- The ROI offset is off by one (+1) when the DI run used an ROI.

### Conventions (orix ↔ EMsoft)
- orix `eu2qu` and EMsoft `eq_` agree component for component; both force q0 ≥ 0.
- Crystal symmetry is applied on the left in both (`S*o`). orix
  `point_group.proper_subgroup` matches EMsoft `QSym_Init`.
- orix `Quaternion.mean` is not symmetry-aware: it is 49° off on scrambled variants, and 0.23° off
  after aligning the variants first.

### kikuchipy gaps
- There is no orientation KAM on this base (`hrebsd_kam` is on hrebsd-dic only).
- There is no grain segmentation on this base (`segment_grains` is on hrebsd-dic only).
- `orientation_similarity_map` (`src/kikuchipy/indexing/_orientation_similarity_map.py:30-152`) is
  correct but not grain-aware. It breaks when `is_in_data` is False, when `keep_n=1`, and on 1-D maps.
- `dictionary_indexing`: a `navigation_mask` value of True means excluded, and masked points get
  garbage (`np.empty`) properties.
- `get_patterns` accepts one PC, or one PC per simulated pattern.
- orix's EMsoft reader hard-codes `EBSDIndexingNameListType`. Real EMsoftOO files use
  `EMDINameList`, so an own h5py reader is needed.

### Oracles on this machine
- **Binaries:** `C:/Users/westraadt.1/EMSOFT/EMsoftOOBuild/Release/Bin/` holds `EMDI`,
  `EMFitOrientation`, `EMHROSM`, `EMgetOSM`, `EMsampleRFZ` (.exe) and `EMsoftOOLib.dll` /
  `EMOpenCLLib.dll`. They were built from upstream develop-era code (buggy KAM, no `maxGROD`).
  EMDI and EMHROSM need OpenCL (RTX 2000 Ada).
- **EMsoft data path:** `C:/Users/westraadt.1/Software/EMSOFT/EMsoftData` (from `EMsoftConfig.json`).
- **Master pattern:** the cached kikuchipy file
  `%LOCALAPPDATA%/kikuchipy/.../ebsd_master_pattern/ni_mc_mp_20kv.h5` (305 MB) is a complete EMsoft
  MC + master file with sig 70, so EMsoft can use it directly.
- **Historical real-data run:** `EMsoftData/OSM/GRX810_HROSM/`, containing
  `dp-GRX-refined.h5` (337×413 Ni map, `KAM`, `OSM`, `OSM_20`, `TopMatchIndices` 50,
  `RefinedEulerAngles`, EMsoft 6_0_20250527) and `dp-GRX-refined_HROSM.h5` (EMsoft 6_0_20250528):
  - Usable as a local-only bit-level oracle for EMsoft-compatible KAM and OSM on real data.
  - Its `grainID` predates the switch to KAM-difference region growing (`0b8e885`, 2025-06-25), so
    it is **not** a clustering target.
- **Al map:** `EMsoftData/Al_HROSM/dp-full.h5` (about 500,000 points) is a large-map check for KAM
  and OSM.

## Design (becomes the spec's D-numbers)

### Package and public API
New private package `src/kikuchipy/indexing/_hrosm/`, laid out like `_spherical/`:
- `_kam.py`
- `_segmentation.py`
- `_grains.py` (`GrainTable`)
- `_averaging.py`
- `_directional_statistics.py`
- `_sampling.py`
- `_osm.py`
- `_driver.py`
- `_emsoft_file.py` (minimal h5py reader for dot-product and HROSM files)

Public names are added to `src/kikuchipy/indexing/__init__.pyi`. `emsoft_compatible` is a keyword
everywhere, never a module global.

```python
kernel_average_misorientation_map(xmap, *, degrees=True, emsoft_compatible=False) -> np.ndarray  # (ny,nx) float32
segment_grains_kam(kam, *, threshold=5.0, dilate=False, emsoft_compatible=False) -> np.ndarray  # int32, 0 = unassigned
grain_bounding_boxes(grain_id) -> np.ndarray                  # (n,4) int64 row0, col0, height, width (0-based)
average_grain_orientations(xmap, grain_id, *, method="mean", max_angle=5.0, n_em=25, n_iter=40,
                           min_kappa=5.0, seed=None, emsoft_compatible=False) -> GrainTable
misorientation_ball(center=None, *, max_angle=5.0, n_steps=20, emsoft_compatible=False) -> Rotation  # ((2N+1)^3,)
orientation_similarity_map(..., *, grain_id=None, emsoft_compatible=False)  # existing function, 2 new keywords

EBSD.hrosm(xmap, master_pattern, detector, energy=None, *, threshold=5.0, max_angle=5.0,
           n_steps=20, keep_n=20, n_osm=10, average="mean", n_em=25, n_iter=40, min_kappa=5.0,
           min_pixels=10, dilate=False, metric="ncc", signal_mask=None, navigation_mask=None,
           pc="grain", n_per_iteration=None, max_chunk_bytes=256e6, seed=None,
           emsoft_compatible=False, show_progressbar=None, verbose=1) -> CrystalMap
```

- **`GrainTable`:** a frozen dataclass with `n_grains`, `n_pixels`, `bounding_box`, `orientation`,
  `kappa`, `max_grod`, `indexed` and `method`. Its `from_crystal_map()` rebuilds it from the
  broadcast properties.
- **Output of `EBSD.hrosm`:** one `CrystalMap` with the input's coordinates, phases and
  `is_in_data`. Its `rotations` are the best local match, or the input rotation where a grain was not
  re-indexed. Properties:
  - `osm` (float32; NaN where not re-indexed)
  - `scores` and `simulation_indices` (`(n, keep_n)`; indices are local to each grain's ball)
  - `grain_id`, `kam`, `grod`, `reindexed`
  - broadcast per-grain values `grain_orientation`, `grain_kappa`, `grain_max_grod`, `grain_n_pixels`
- **GROD pre-check (Johan's `f6270d0`):** always computes `max_grod`, and issues one `UserWarning`
  listing grains whose spread exceeds `max_angle`.

### Per-grain driver
- **Ball:** computed once around the identity (68,921 × 32 B = 2.2 MB), then rotated per grain as
  `ball_g = (~R0) * avor_g`.
- **Pixels indexed:**
  - Correct mode: the grain's own pixels only, from an eager float32 experimental block.
  - EMsoft-compatible mode: the whole bounding box, with only this grain's pixels copied back.
- **Dictionary:** lazy `get_patterns` chunked by `max_chunk_bytes` (17,777 patterns per chunk at
  60×60), fed to the existing `_dictionary_indexing` core. That core gets a backwards-compatible
  `verbose` keyword (a change to upstream code, recorded) so a loop over hundreds of grains stays
  quiet under one grain-level progress bar.
- **PC:** `pc="grain"` takes the mean PC of the grain's points from a plane-fitted per-point detector
  (a documented deviation from EMsoft). `pc="single"` and compat mode use one PC.
- **Multi-phase maps (v1):** one master pattern. Grains are segmented per phase, and grains of other
  phases are skipped with a warning.
- **Determinism:**
  - Results must not change with `n_per_iteration`; a test pins this bit for bit.
  - VMF/Watson are the only random parts, and they use the seeded `Generator`.

### Numerics and `emsoft_compatible` switches
| Piece | Correct (default) | Compat switch |
|---|---|---|
| KAM | 4-neighbour mean over existing, present neighbours (= Johan's `c85982a`); `2*arccos(clip(d))`; NaN if no neighbour | K1 the 1-D table incl. identity at `(W,1)`; K2 unclipped `acos` |
| Segmentation | same KAM-difference region growing (closed form, `scipy.sparse.csgraph`), no 1e6 stack truncation | (same algorithm; fed K1 KAM) |
| Dilate | fill unassigned pixels only, max label on ties | K3 EMsoft window incl. skipped row/col and overwrite |
| `center` | grain pixel nearest the grain centroid | K4 bounding-box centre pixel |
| VMF/Watson | log-sum-exp, symmetry on the LEFT, `eigh`, best finite likelihood | K5 right-side `mu⊗S`; K6 getQandL defects idealised as "first init wins" (VMF runs all `n_iter`, WAT 2); K7 right-side FZ reduction |
| `mean` (default) | align variants to the correct centre (`S*o`), normalised mean; kappa from VMF closed form | n/a |
| Ball | float64 quaternions, `conj(q_v)*q0` | K8 float32 Rodrigues storage |
| OSM | grain-aware 4-neighbour mean of top-`n_osm` intersections, vectorised, 0..n range, NaN if no in-grain neighbour | K9 EMsoft table over the bounding box, un-normalised |
| Indexing domain | grain pixels only | K10 bounding box + 1024-batch dictionary thinning emulation |
| Inputs | masks/absent points allowed; per-grain PC | K11 dense map + single PC required (ValueError otherwise) |
| Unfilled pixels | NaN OSM, input rotation kept | K12 zero OSM, identity rotation |

The segmentation rule is the same in both modes, matching Johan's own EMsoftOO branch, which fixed the
KAM but kept the growth rule. The rule can leak across boundaries below about 4× the threshold. That
is documented, and misorientation-based segmentation is a recorded follow-up.

### Performance (estimates, to be measured and pinned in Stage B)
NCC work is about `n_px × 68,921 × sig_size × 2` flops:

| Map | Pixels × pattern size | Flops | Estimated time |
|---|---|---|---|
| nickel_ebsd_large | 4,125 × 60×60 | 2.0e12 | ~40 s |
| ni_gain | 29,800 × 60×60 | 1.5e13 | ~5 min |
| si_wafer, unbinned | 2,500 × 480×480 | 7.9e13 | hours (with simulation) |

- Simulation costs about 3 s per grain at 60×60, so si_wafer is weekly-only or binned.
- Cost scales with `(2N+1)^3`: `n_steps=10` is 7.4× cheaper.
- GPU is deferred; the `SimilarityMetric` interface is where it would plug in.

## Validation catalogue (spec `validation.md`; values marked MTP are measured, then pinned)

**Gates**
- **CI:** always runs.
- **bin:** runs only when `KIKUCHIPY_EMSOFT_BIN` is set, serialised by a
  `kikuchipy-emsoft-program.lock` (a clone of `conftest.py:677-722` `_emsphinx_program_lock`).
- **local:** runs only when `KIKUCHIPY_EMSOFT_DATA` is set (Johan's historical files).
- **weekly:** `@pytest.mark.weekly`.

| V | Pins | Oracle / comparison | Gate |
|---|---|---|---|
| V0 | numba kernel discipline (`cache`, `nogil`, no fastmath), `.py_func` parity | bitwise | CI |
| V1 | compat KAM (K1/K2) | bitwise vs a test-local literal transcription, including 1-row/1-col maps | CI |
| V2 | compat KAM on real data | bitwise vs the shipped DI `KAM` (nickel_ebsd_large ref) and vs `GRX810 kam` / `Al dp-full KAM` | CI / local |
| V3 | correct KAM | analytic constant-curvature field + orix `angle_with` bookkeeping, 1e-10 | CI |
| V4 | segmentation | bitwise vs a literal flood-fill transcription; edge cases at exactly `gangle`; vs `EMHROSM` `grainID` (V12) | CI |
| V5 | correct KAM + segmentation on synthetic 2/3-grain maps with sub-degree gradients | label equality up to permutation | CI |
| V6 | dilate + center (K3/K4 and correct forms) | transcription + `scipy.ndimage`; outside-grain centre asserted in compat, inside in correct | CI |
| V7 | sampling ball | count/shells/edge; orix `from_homochoric(cu2ho)` 1e-12; `EMsampleRFZ` `MIS` (`eu`/`qu`/`ro`) set match + composition side; shipped N=6 list | CI / bin |
| V8 | VMF/Watson/mean recovery | orix `random_vonmises` (kappa 20/100/1000, N 30/300), scrambled by random `Oh` ops; wrong-side scrambling must fail | CI |
| V9 | EM compat quirks K5-K7 | bitwise vs seeded transcription; early-exit iteration count at kappa 1e4 | CI |
| V10 | compat OSM K9 | bitwise vs shipped DI `OSM`/`OSM_05` (EMgetOSM) and `GRX810 OSM_20`; float32 multiplier op-order measured then pinned | CI / local |
| V11 | correct OSM | bitwise = `orientation_similarity_map` on single-grain full maps; cross-grain neighbours excluded; works with `is_in_data=False`, `keep_n=1` | CI |
| V12 | EMHROSM cluster stage | shipped `center`, `center_dilate`, `wat` refs: `kam`/`grainID`/`avor`(center) bitwise; wat `avor`/`kappa` MTP (clock seed) | CI |
| V13 | end-to-end tolerance vs EMHROSM | kikuchipy-native DI vs `newOSM`/`newEuler`/`newCI`: OSM Pearson r, disorientation median/p99, CI r (MTP; seeds 0.8 / 0.3° / 1.0°); only grains with `W*H >= numdictsingle` | CI subset / weekly |
| V14 | regenerate-and-diff | `create_hrosm_reference.py` reproduces shipped refs: CPU arrays bitwise, GPU-derived measured then pinned; `program_md5` per exe and DLL | bin |
| V15 | physics sanity | synthetic 20×30 map, 40×40 px: 0.5° sub-boundary invisible to a 1.4° global-dictionary OSM, visible in HROSM (drop ≥ MTP), 0.5° step recovered (median ≤ 0.15° MTP) | CI (weekly at full size) |
| V16 | API contracts | mask polarity (garbage props never leak), per-grain vs single PC, `min_pixels`, `kappa=-1` skip, `n_per_iteration` and lazy-vs-eager invariance, determinism, runtimes recorded (never gated) | CI |

### Reference data (`src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py`, import-safe like `create_emsphinx_reference.py`)
**Input map and run directory**
- nickel_ebsd_large (55×75, 60×60), static + dynamic background removed, written with the NORDIF
  writer.
- Run directory: `EMsoftData/kikuchipy_hrosm/<run>/`.
- The cached EMsoft Ni master is copied in, with its md5 recorded.
- PC comes from `det.pc_emsoft()`, divided by binning: `xpc, ypc, L, delta = 4.62, 17.16, 240.96, 8.0`.

**Namelist settings**
- `EMDI`: `nnk 20`, `nosm 10`, `ncubochoric 100`, `ROI 0`.
- `numexptsingle = numdictsingle = 32`. This removes the thinning defect from the reference for all
  but tiny grains, and keeps `Ne == Nd` for the OpenCL kernel; a pre-flight check verifies this.

**Runs**
1. `EMDI`
2. `EMFitOrientation` (`FIT`, `PCcorrection off`)
3. `EMgetOSM` (nmatch 10, 5). Self-check: `OSM_10 == OSM`.
4. `EMHROSM` (center, center+dilate, wat)
5. `EMsampleRFZ` (MIS 5°/N20 and N6, generic centre)

**Acid bands:** top-1 dictionary result vs the shipped refined xmap ≤ ~1.5°, refined result ≤ 0.5°.
These catch sign, flip and PC errors early.

**Shipped files**
- Uncompressed `.npz` files, at most 250 kB each and about 800 kB in total (recorded as above the
  EMSphInx precedent).
- md5s recorded in `_registry.py`.
- Provenance stored in each file: `program_md5` of every exe and DLL, `emsoft_version`, master and
  patterns md5, PC, namelists, GPU name.
- A `TestReferenceFiles` quad-equality check, cloned from `test_spherical_emsphinx_regression.py`.

### Mutants (each applied alone in the main tree; each must be killed by a named test)
| Mutants | Area | What is changed |
|---|---|---|
| M1-M5 | compat KAM | correct vertical index, `jj` formula, corner factor, dropping the identity term, float64 `rtod` |
| M6-M7 | correct KAM | fixed divisor 4, unreduced angle |
| M8-M12 | growth | 4-conn in compat, non-chained criterion, no seed gate, singleton kept, float64 tie |
| M13-M15 | dilate / center | dilate border/overwrite, correct dilate overwriting, centre `(w-1)//2` |
| M16-M18 | ball | count, composition side, small-angle edge formula |
| M19-M23 | averaging | EM side mix, Watson linear term, kappa gate `>=`, stale Q/L removed, unaligned mean |
| M24-M26 | OSM | compat `/nosm`, multiplier op-order, cross-grain counting |
| M27-M30 | driver | `min_pixels` boundary, mask polarity, `newEuler` in degrees, ball not re-centred |

The full table with killer test names goes into `plan.md` section 7.

## Process (model assignment per memory)

- **Planning and spec:** Fable, effort xhigh, ultracode Workflows.
- **Tests, implementation, review, bug injection, fixes:** Workflow agents with `model: "opus"`,
  effort xhigh, at most 15 agents per workflow.
- **Commits and pushes:** by the main loop only.
- **Per-stage sequence:** failing-tests workflow → commit → implement → adversarial review (two
  sceptics per finding) → mutants (in the main tree, batches of 3) → closing gates → commit + push.
  A commit that only adds failing tests is never pushed on its own.

### Step 0: branch mechanics (after NLPAR has merged)
**Pre-flight**
- The working tree is clean and `stash@{0}` is present.
- The NLPAR PR is merged into `feat-spherical-indexing`. Record the new tip sha.
- Record the hashes of the parked plans (`plan-upstream-merge-0.13.1.md`, this file, and the NLPAR
  plan if it still exists).

**Steps**
1. `git switch -c feat-HROSM feat-spherical-indexing`.
2. Verify that `specs/` is tracked (`git ls-files specs/roadmap.md`), that the `.gitignore` specs rule
   is gone, and that `.pre-commit-config.yaml` already excludes `specs/`. These three changes arrive
   with the NLPAR merge, so no re-adoption commit is needed.
3. Append this file and the upstream-merge plan to `.git/info/exclude` (local only). Check that
   `git status --short -- specs` shows neither of them.

**Fallback if NLPAR has not merged when HROSM starts:** run the NLPAR plan's Step 0 with the name
swapped (`git checkout develop -- specs`, the two ignore lines, the pre-commit exclude), as fork-local
commit 0 "Re-adopt the fork specs tree on feat-HROSM". develop is never merged in.

### Step 1: spec, then Johan's approval, then commit 1
**Spec-drafting workflow `hrosm-spec-draft`**
- Three drafters (requirements, plan, validation) work from this plan and the four exploration
  reports.
- They use `hrebsd-dic:specs/2026-09-07-hrebsd-dic/*.md` as the template.
- Three read-only critics check the draft: factual (against the repo, EMsoftOO source and
  binaries), completeness (every quirk has a V and an M), and executability.
- A fixer resolves findings, with a disposition table. At most two rounds and 11 agents.

**Spec folder and constitution amendments**
- Spec folder: `specs/2026-09-28-hrosm/`.
- `mission.md`: add a "Fork feature path: HROSM" paragraph.
- `roadmap.md`: add a `---` then `# Feature path: HROSM (branch feat-HROSM; spec 2026-09-28-hrosm)`
  with Stage A/B/C checkboxes, appended at the end of the file after NLPAR's block.
- Re-date the spec folder if work resumes much later.
- `tech-stack.md`: add an "HROSM feature path" section covering:
  - base sha and update rule
  - EMsoft oracle conventions: `KIKUCHIPY_EMSOFT_BIN`, `KIKUCHIPY_EMSOFT_DATA`, the program lock,
    the DLL md5 rule
  - the BSD-3 attribution rule
  - the oldest-matrix recipe:
    `uv run --isolated --python 3.10 --with "numpy==1.23.0" --with "numba==0.57" --with "orix==0.12.1" pytest tests -k hrosm`
  - the numba-cache flake rule: run `-n 0`, then `-n 4`, and re-run red tests alone

**Gate:** Johan approves the spec's `plan.md`. Then commit 1: "Add HROSM spec and constitution
amendments".

### Stage A: orientation engine (bit-level; no DI)
- **Code:** `_hrosm/` modules `_kam`, `_segmentation`, `_grains`, `_averaging`,
  `_directional_statistics`, `_sampling`, the compat OSM function, `_emsoft_file`; also
  `create_hrosm_reference.py` and the reference-generation run on this machine.
- **Tests:** `tests/test_indexing/test_hrosm_{kam,segmentation,averaging,sampling,osm}.py` and
  `test_hrosm_emsoft_regression.py`, covering V0-V10, V12, V14 and M1-M25.
- **Commits:**
  - 2: "Add HROSM Stage A skeletons and failing tests"
  - 3: "Implement HROSM Stage A: KAM, grain clustering, sampling ball, grain averaging and EMsoft
    oracles" (includes the shipped refs, registry, CHANGELOG bullet, roadmap ticks)
- Push after commit 3.

### Stage B: per-grain DI and `EBSD.hrosm()`
- **Code:** `_driver.py`, `_osm.py` (correct mode), `EBSD.hrosm`, the new keywords on
  `orientation_similarity_map`, the `_dictionary_indexing` `verbose` keyword, CHANGELOG.
- **Tests:** `tests/test_signals/test_ebsd_hrosm.py`, covering V11, V13, V15, V16 and M26-M30.
- **Runs:** the weekly full-map run; performance baselines recorded.
- **Commits:** 4 and 5. Push.

### Stage C: tutorial
**Notebook `doc/tutorials/hrosm.ipynb`**
- nickel_ebsd_large: DI + NCC refinement, then HROSM.
- Global OSM next to HROSM; KAM, GROD and grain maps; the V15 synthetic sub-grain demo; parameter
  guidance (`max_angle`, `n_steps`, cost); an EMsoft-compat table.
- Links to `pattern_matching.ipynb` and `spherical_indexing.ipynb`, which are never edited.
- Outputs are stored because the runtime is over 2 minutes.

**Registration and extras**
- Register the notebook in `index.rst`, `run_nbval.sh` and the sanitize regexes.
- Gallery example `examples/pattern_matching/hrosm.py`.
- CHANGELOG tutorial bullet.

**Commit 6:** "Add HROSM tutorial notebook".

**Closing steps**
1. Re-submit the spec documents to review.
2. Open the PR: `gh pr create --repo jwestraadt/kikuchipy --base feat-spherical-indexing`. The number
   will follow NLPAR's PR and is confirmed when it is opened; CHANGELOG entries use the fork PR-link
   convention.
3. Watch CI. ubuntu and windows should be green; the macOS ebsdsim step is red for a pre-existing,
   unrelated reason.
4. Final commit: "Tick HROSM boxes in roadmap (jwestraadt/kikuchipy#N)".
5. Merge only on Johan's go.

### Definition of done
- **Roadmap:** all boxes ticked, with ledger evidence.
- **Measured values:** every MTP replaced by a dated measured value (recipe and machine).
- **Test records:** coverage 100 % of `_hrosm/`; one oldest-matrix run per stage; V14 run once per
  reference file.
- **Spec and branch:** the spec is re-reviewed after Stage C; `git log origin/feat-HROSM..feat-HROSM`
  is empty.
- **PR:** open with ubuntu/windows CI green.
- **Memory:** a project memory note written (branch, base, oracles, K-switch list, sequencing after
  NLPAR).

## Standing constraints
**Git and files**
- `stash@{0}` is never popped.
- `specs/2026-08-16-constitution/upstream-issue.md` is never edited.
- The parked plans are never committed or touched.
- `doc/tutorials/{hybrid_indexing,spherical_indexing,load_save_data,pattern_matching}.ipynb` are never
  swept into commits.
- Explicit pathspecs only.
- Signed commits carrying the attribution trailer the session specifies at commit time.

**Writing and tooling**
- No em-dashes in prose.
- The main `.venv` stays free of CuPy.
- Pre-commit runs as `uvx pre-commit run --files <explicit list, never specs/>`.
- Spec files are written with the Write/Edit tools only (no BOM on `roadmap.md`).

**EMsoft**
- EMsoft programs are never run concurrently (use the lock).
- Run directories stay under `EMsoftData/kikuchipy_hrosm/`.
- Johan's EMsoftOO checkout stays on `develop` and is never modified.
- The historical `OSM/` and `Al_HROSM/` files are read-only inputs.

## Risks
| Risk | Handling |
|---|---|
| The ifx `-fp-model` flag reorders floating-point operations, so KAM/OSM might not match bit for bit | V10 measures the op order from real files first; if needed, fall back to a 1-ulp float32 pin and record the count of differing pixels |
| EMsoftOO rejects the cached kikuchipy master | pre-flight EMDI read; fallback EMMCOpenCL + EMEBSDmaster from `DItutorial/Ni` namelists (+30-60 min) |
| Wrong NORDIF flip or PC sign in the reference | acid bands abort the reference sweep |
| `Ne != Nd` breaks the OpenCL kernel | set both to 32; pre-flight check on the top-1 acid band |
| VMF/Watson seeds come from the clock | `grainID`/`kam` compared bit for bit, `avor`/`kappa` within a tolerance; compat EM compared bit for bit only against the seeded transcription |
| GPU top-k ties can reorder between regenerations | bit-for-bit contract for CPU-only arrays; GPU arrays measured |
| Reference size | at most 250 kB per file, N=6 ball shipped, N=20 gated |
| Numba cache flake under xdist, Windows 8k-character command limit, editable install pointing at the main `src` | run `-n 0` then `-n 4`; in-tree mutants; short pathspec lists |
| NLPAR not merged when HROSM starts, or NLPAR changes shared files (`_dictionary_indexing`, CHANGELOG, roadmap) | Step 0 fallback; rebase-free (branch off the updated staging branch); re-read `_dictionary_indexing.py` before adding `verbose`; parked-plan hashes checked at every close gate |
| Tutorial runtime on RTD | stored outputs; synthetic demo at 40×40 px |
| si_wafer unbinned run is infeasible in CI | weekly/binned only; tutorial uses nickel_ebsd_large |

## Recorded defaults (the spec lists them; flag any you want changed before Step 1)
**Output and API**
1. Output: one `CrystalMap` carrying broadcast per-grain properties, plus
   `GrainTable.from_crystal_map`. The alternative would be returning a tuple.
2. `EBSD.hrosm` defaults to `average="mean"`. Parity tests pass `average="center"` explicitly, as
   EMsoft's default does.
3. `grain_id` uses EMsoft's convention: 0 means unassigned, grains are numbered from 1. hrebsd's
   `segment_grains` uses 0-based labels with -1.
4. Multi-phase v1: one master pattern; grains of other phases are skipped with a warning.

**Algorithms**
5. K5 (right-side symmetry in the EM steps) is treated as a defect and applies only in compat mode.
6. Segmentation keeps EMsoft's KAM-difference rule in both modes. Misorientation-based segmentation
   is a follow-up.

**Upstream code touched**
7. The shared `_dictionary_indexing` core gets a `verbose` keyword, and
   `orientation_similarity_map` gets `grain_id` and `emsoft_compatible`. Both are backwards
   compatible.

## Verification (whole feature)
- **Test suites:**
  - `pytest tests -k hrosm` passes, both with and without the binary/local gates (`-n 0`, then
    `-n 4`).
  - The spherical and full suites are unchanged and pass.
  - Coverage of `_hrosm/` is 100 %.
  - The oldest-matrix recipe passes.
- **Oracle runs, with outcomes recorded in the ledger:**
  - V14 regenerate-and-diff (bin gate).
  - V2/V10 on GRX810 and Al (local gate).
  - V13 full map (weekly).
- **Lint and docs:**
  - `uvx pre-commit run --files <changed>` is clean.
  - `sphinx-build -b html` exits 0.
  - nbval passes on `hrosm.ipynb`.
- **Branch hygiene:**
  - `git diff --name-only` never lists the never-sweep notebooks.
  - develop, hrebsd-dic and feat-spherical-indexing are untouched until the PR merges. Check with
    `git rev-parse` against the shas recorded at Step 0; before NLPAR they were
    71a1b2f3 / 02a529c0 / 6723aaf0.
