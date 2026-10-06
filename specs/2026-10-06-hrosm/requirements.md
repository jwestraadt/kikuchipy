# HROSM -- `feat-HROSM`: requirements

Status (2026-10-06): spec drafted, NOT approved. Johan approves `plan.md`
before any test or code is written (no waiver for HROSM); the seven
recorded defaults of the parked plan are adopted provisionally as D19
(R1-R7) and listed in `plan.md` section 7 as "pending Johan's approval with
this plan". Branch `feat-HROSM`, cut from fork `develop` @ `de27741a`
(= `origin/develop`: spherical indexing PRs #1-#12, pseudo-symmetry,
spherical GPU, NLPAR #17/#18, HyperSpy 2.5 test fix #19; 24 commits behind
`upstream/develop`; no HREBSD code). Spec folder `specs/2026-10-06-hrosm/`,
one folder for three stages (A orientation engine, B per-grain re-indexing
and `EBSD.hrosm()`, C tutorial). The feature ports EMsoftOO's EMHROSM ("high
angular resolution orientation similarity map", `Source/DictionaryIndexing/
EMHROSM.f90` + `Source/EMOpenCLLib/program_mods/mod_HROSM.f90`, BSD-3, De
Graef 2025): segment an indexed orientation map into grains, average each
grain's orientation, re-index each grain's patterns against a fine local
dictionary (a 5 deg misorientation ball, 0.25 deg shells, 68,921
orientations) and build a new orientation similarity map (OSM) from the top
matches. Correct by default; `emsoft_compatible=True` reproduces the
EMsoftOO develop binary's orientation-stage arithmetic including its
defects, each a numbered switch K1-K12 (D9).

**Relation to the parked plan.** `specs/_research/plan-hrosm-2026-09-28.md`
(parked 2026-09-28, unexecuted) and its five exploration reports are the
substance of this spec; every fact the implementer needs is copied in, with
line references re-verified on 2026-10-06 against fork `develop` @
`de27741a`, `hrebsd-dic` @ `b64cc18f`, `feat-spherical-indexing-nlpar` @
`e49b3d85` and the EMsoftOO checkout (`C:\Users\westraadt.1\Repos\EMsoftOO`,
`develop` @ `c127868`, 2026-05-24). Superseded by user decisions: parked
decisions 2 (branch off and PR into `feat-spherical-indexing`) and 6
(sequencing through that branch, specs re-adoption) by D14 (Johan,
2026-10-06); the parked Step 0 is dropped (`develop` tracks `specs/`); the
parked "Process" section by D18 (model rule, 2026-10-05). Design amendments
made while drafting, each dated in place: K10 no longer emulates the
dictionary-batch thinning (D8.10); the compat OSM edge multiplier follows
the source order because the binaries disagree (D7.3); compat KAM parity is
bitwise on non-degenerate pixels only (D3.6); K7 uses the Voronoi form of
the Rodrigues fundamental zone (D5.6); `EBSD.hrosm` drops
`show_progressbar` and `max_chunk_bytes` (D1.3); `GrainTable` stores
`rotation` + `phase_id` and `valid` (D2.1); `segment_grains_kam` gains
`phase_id` (D4.1); no numba kernels (D10.3); the gallery example moves to a
new `examples/indexing/` section (D16.3). Spec review round 1 (2026-10-06;
`plan.md` section 10), each marked "amended 2026-10-06, spec review" in
place: D1.2 (coexistence wording), D1.3 (`seed` type), D1.5 (grid shape,
mask checks), D1.6, D1.9 (new: the map grid), D2.3, D3.2 (compat input
route), D3.6, D3.7, D5.4, D5.5, D5.7, D5.8, D5 Pins, D6.6, D7.5, D8.6,
D8.11, D8.15 (new), D11.1, D13.1, D13.2, D13.3, D13.5, D14.4, D14.5,
D15.1, D15.3, D15.4, D16.4, D16.5, D20.2, D20.3 and the conventions
Context. Spec review round 2 (2026-10-06; `plan.md` section 10, round
2), each marked "spec review round 2" in place: D1.5 and D1.9 (the grid
spans every point), D2.1 and D4.6 (labels without pixels), D3.1, K2
and D10.4 (the near-one snap replaces the clip), D5.5 (Watson at kappa
1e4: per-init `L` and the chosen init measured), D5.8 and D20.3 (float64
comparison), D8.7 (`sleep(0.2)`), D8.8 (per-point detector shape),
D13.1, D15.1 and D15.3 (the fork's Weekly workflow is disabled; the
on-push CI check), D20.2 and D20.5.

Drafting measurements of 2026-10-06 (read-only probes, session scratchpad,
not shipped) are seeds per D17; they are quoted in D3.6, D4.1, D6.1, D7.3,
D13.5 and D15.4.

## Scope

In scope (three stages on one branch; `plan.md` sections 2-4):

- **Stage A -- orientation engine (no dictionary indexing).** Private package
  `src/kikuchipy/indexing/_hrosm/` (D1.1): EMsoft quaternion tables and
  arithmetic, KAM in both modes (D3), KAM-difference segmentation, dilate
  and bounding boxes (D4), grain averaging (`mean`, `center`, von
  Mises-Fisher and Watson EM, GROD pre-check) and `GrainTable` (D2, D5), the
  misorientation ball (D6), both OSM forms as array functions (D7), the
  private EMsoft file reader (D12), `create_hrosm_reference.py`, the
  reference run on this machine and the shipped references (D13). Public
  (eight names): `kernel_average_misorientation_map`, `segment_grains_kam`,
  `grain_bounding_boxes`, `average_grain_orientations`,
  `grain_reference_orientation_deviation_map`, `misorientation_ball`,
  `misorientation_ball_spacing`, `GrainTable` (D20 added two, 2026-10-06).
- **Stage B -- per-grain re-indexing.** `_driver.py` and `EBSD.hrosm()`
  (D8), the two new keywords of `orientation_similarity_map` (D7.5), the
  backwards-compatible `verbose` of `_dictionary_indexing` (D8.7),
  end-to-end tolerance pins against EMHROSM output, physics-sanity and
  contract tests, performance baselines (recorded, never gated).
- **Stage C -- tutorial.** `doc/tutorials/hrosm.ipynb` and its
  registration, `examples/indexing/hrosm.py` with the new section file
  `examples/indexing/README.rst`, the CHANGELOG tutorial bullet (D16).
- **Data.** Synthetic maps generated in the tests with fixed seeds;
  `kp.data.nickel_ebsd_small()` (shipped, (3, 3 | 60, 60));
  `kp.data.nickel_ebsd_large(allow_download=True)` ((55, 75 | 60, 60)
  uint8, one PC per point, stored refined `xmap` with props `scores`, `z`;
  `_data.py:129`); `kp.data.nickel_ebsd_master_pattern_small()` (shipped,
  401 px, 20 kV); `kp.data.ebsd_master_pattern("ni")` (cached EMsoft MC +
  master `ni_mc_mp_20kv.h5`, 305 MB; weekly); the shipped EMsoft references
  (`src/kikuchipy/data/emsoft_hrosm/`, D13.3); the historical EMsoft files
  behind the local gate (D13.5).
- **Constitution amendments** (`plan.md` section 0), appended at the end of
  `specs/mission.md` (75 lines), `specs/roadmap.md` (166 lines, after
  NLPAR's block) and `specs/tech-stack.md` (101 lines, after the NLPAR
  section) by the main loop in the spec commit.

Out of scope (each with its revisit path):

- **GPU**: no device path; `SimilarityMetric` is the seam.
- **Misorientation-based segmentation**: both modes keep EMsoft's
  KAM-difference rule (R6); `segment_grains` on `hrebsd-dic` (not on
  `develop`) is the existing alternative. Revisit: a `rule=` keyword after a
  measured need (boundaries below ~4x the threshold leak, D4.7).
- **Multi-master multi-phase**: one master pattern; grains of other phases
  are skipped with a warning (R4). Revisit: `dict[str, EBSDMasterPattern]`.
- **EMsoft's dictionary-stage physics**: hi-pass (`mod_filters.f90:
  974-1024, 1043-1079`), `adhisteq` (`:455-580`), energy-weighted
  simulation (`program_mods/mod_EBSD.f90:3473-3667`), dot products of
  non-negative patterns, the OpenCL kernel; kikuchipy's `get_patterns` +
  NCC replace them, so the end-to-end comparison is tolerance-based
  (parked decision 4, unchanged).
- **Super-resolution**; per-pixel PC inside a grain (D8.8).
- **EMsoft I/O**: no public reader (D12 is private; orix's
  `emsoft_h5ebsd.py:165` expects `EBSDIndexingNameListType`, EMsoftOO files
  carry `EMDINameList`); no `.ang`/`.ctf`/TIFF/IPF outputs.
- **EMsoft defects not reproduced** (quirk catalogue rows 8, 9, 17, 22-25).
- **Upstream** PRs; `hrebsd-dic` beyond the fan-out (D14.4); the untracked
  `specs/_research/plan-upstream-merge-0.13.1.md` (never touched) and the
  18.9 GB root file `AGH__Si_indent_1_512x672.h5oina` (never committed; NOT
  ignored on `develop`, so explicit pathspecs only).
- **New required dependencies**: none (D11.1).

## Decisions

Each decision states what is decided, why, the alternative rejected and the
`validation.md` arm that pins it. (frozen) = changed only by a dated
amendment here; (recorded) = a fact or policy, amendable at a gate with a
ledger entry; (recorded, pending Johan's approval) = a parked default
(D19). K-numbers are the switches of D9, R-numbers the defaults of D19.

### D1 -- API surface, naming and placement (frozen)

1. **Package** `src/kikuchipy/indexing/_hrosm/`, `__init__.py` holding only
   a docstring that lists the submodules (as `_spherical/__init__.py`):
   - `_emsoft_quaternions.py`: literal EMsoft tables and float64 arithmetic
     for the compat paths: `SYM_Qsymop` and the `QSym_Init_` selection
     (`mod_quaternions.f90:53-64ff, 2542-2746`), `PGrot`
     (`mod_symmetry.f90:404-407`), `eq_` (`mod_rotations.f90:5731-5775`),
     the `quatmult` term order (`mod_quaternions.f90:1078-1126`), the
     `getDisorientation_` double loop (`mod_so3.f90:4117-4219`).
   - `_kam.py` (D3), `_segmentation.py` (D4), `_grains.py` (D2),
     `_averaging.py` (dispatcher, `mean`, `center`, GROD; D5),
     `_directional_statistics.py` (VMF/Watson EM; D5), `_sampling.py` (D6),
     `_osm.py` (D7), `_driver.py` (D8), `_emsoft_file.py` (D12).
2. **Public names** in `src/kikuchipy/indexing/__init__.pyi` (imports after
   `._hough_indexing`, sorted `__all__`, which drives the API reference):
   `GrainTable`, `average_grain_orientations`, `grain_bounding_boxes`,
   `grain_reference_orientation_deviation_map` (D20.2),
   `kernel_average_misorientation_map`, `misorientation_ball`,
   `misorientation_ball_spacing` (D20.1), `segment_grains_kam`; two
   keyword-only additions to
   `orientation_similarity_map`; the method `EBSD.hrosm`. Collision check
   (2026-10-06): `develop`'s `__all__` (19 names) shares none; `hrebsd-dic`
   adds `hrebsd_gnd`, `hrebsd_kam`, `hrebsd_pc_shift`,
   `hrebsd_strain_stress`, `segment_grains`, `voigt_stiffness`, none
   identical. Coexistence there after the fan-out: `segment_grains`
   (misorientation, 0-based labels, -1 unlabelled) and `segment_grains_kam`
   (KAM difference, EMsoft labels) are distinct; each See Also names the
   other with one sentence, added on `hrebsd-dic` only (`segment_grains`
   does not exist on `develop`; `plan.md` 1.2). The prop name `grain_id` is
   shared with HREBSD's output under another label convention (R3); no
   rename (amended 2026-10-06, spec review): `hrebsd_kam` and `hrebsd_gnd`
   read `grain_id` together with HREBSD-only props (`rotation_vector`,
   `Fe`) and raise without them, but `EBSD.hrebsd_dic(grain_labels=...)`
   takes 0-based labels with -1 outside any grain (`_hrebsd/_reference.py:
   204-209` on `hrebsd-dic`) and would silently treat HROSM's unassigned
   pixels (0) as a grain, so the `hrebsd-dic`-only See Also/Notes of
   `segment_grains_kam` states the conversion `grain_id - 1`.
   hrebsd-dic has no GROD or orientation-KAM code.
3. **Signatures, frozen** (keyword-only after the data; `emsoft_compatible:
   bool = False` everywhere, never a module global):

   ```python
   kernel_average_misorientation_map(xmap, *, degrees=True,
       emsoft_compatible=False) -> np.ndarray       # (ny, nx) float32
   segment_grains_kam(kam, *, threshold=5.0, dilate=False, phase_id=None,
       emsoft_compatible=False) -> np.ndarray       # int32, 0 or 1..n
   grain_bounding_boxes(grain_id) -> np.ndarray     # (n, 4) int64, 0-based
                                                    # row0, col0, h, w
   average_grain_orientations(xmap, grain_id, *, method="mean",
       max_angle=5.0, n_em=25, n_iter=40, min_kappa=5.0,
       seed: int | np.random.Generator | None = None,
       emsoft_compatible=False) -> GrainTable       # "mean" | "center" |
                                                    # "vmf" | "watson"
   misorientation_ball(center=None, *, max_angle=5.0, n_steps=20,
       emsoft_compatible=False) -> Rotation         # ((2N + 1)**3,)
   misorientation_ball_spacing(max_angle=5.0, n_steps=20) -> float
                                                    # mean NN angle [deg]
   grain_reference_orientation_deviation_map(xmap, grain_id, grains)
       -> np.ndarray                                # (ny, nx) float32 [deg]
   orientation_similarity_map(xmap, n_best=None,
       simulation_indices_prop="simulation_indices", normalize=False,
       from_n_best=None, footprint=None, center_index=2, *,
       grain_id=None, emsoft_compatible=False) -> np.ndarray

   def hrosm(self, xmap: CrystalMap, master_pattern: EBSDMasterPattern,
             detector: EBSDDetector, energy: int | float | None = None, *,
             threshold: float = 5.0,     # EMsoft gangle [deg]
             max_angle: float = 5.0,     # misorang [deg]
             n_steps: int = 20,          # nsamples
             keep_n: int = 20,           # nnk of the EMDI run
             n_osm: int = 10,            # nosm
             average: str = "mean",      # orav (R2)
             n_em: int = 25, n_iter: int = 40,  # numEM, numIter
             min_kappa: float = 5.0,     # mod_cluster.f90:299
             min_pixels: int = 10,       # mod_HROSM.f90:592
             dilate: bool = False,
             metric: SimilarityMetric | str = "ncc",
             signal_mask: np.ndarray | None = None,
             navigation_mask: np.ndarray | None = None,
             pc: str = "grain",          # "grain" | "single"
             n_per_iteration: int | None = None,
             seed: int | np.random.Generator | None = None,
             emsoft_compatible: bool = False,
             verbose: int = 1) -> CrystalMap
   ```

   Dropped from the parked signature (amended 2026-10-06, leanness):
   `show_progressbar` (`verbose` follows the fork's `spherical_indexing`: 0
   silent, 1 information + grain-level progress bar + timing, 2 also each
   grain's dictionary-indexing message) and `max_chunk_bytes` (D8.6). The
   docstring Notes carry the keyword map to the EMHROSM namelist. The free
   function's `seed` takes a `Generator` like `EBSD.hrosm`'s (amended
   2026-10-06, spec review; V8 passes one).
4. **Placement.** `EBSD.hrosm` directly after `dictionary_indexing`
   (`ebsd.py:2400-2557`), before `spherical_indexing` (`:2559`); on
   `hrebsd-dic`, `hrebsd_dic` sits between `spherical_indexing` and
   `refine_orientation_spherical`, so the insertions never touch. The driver
   import goes after the `_hough_indexing` import (`ebsd.py:53-58`); the
   method only validates, calls the driver and returns.
5. **Validation** (`ValueError`, never `TypeError`; tested fragments in
   quotes), in order: (1) two navigation dimensions ("two navigation
   dimensions"); the map's grid shape `(ny, nx)` from `_map_grid` (D1.9),
   not `xmap.shape`, equals the navigation shape (rows, cols) ("xmap
   shape"; amended 2026-10-06, spec review: orix flattens one-row and
   one-column maps; round 2: the grid spans every point, so a map whose
   edge rows or columns are not in the data passes, and the message of
   a sliced map says to pass the unsliced signal); (2) `master_pattern._is_suitable_for_projection(raise_if_not=
   True)`; `detector.shape` = signal shape ("detector shape");
   `detector.navigation_shape` in {`(1,)`, `(ny, nx)`} ("one PC or one PC
   per map point"); (3) `threshold > 0`, `0 < max_angle < 180`, integers
   `n_steps >= 1`, `keep_n >= 1`, `1 <= n_osm <= keep_n` ("n_osm"), `keep_n
   <= (2 n_steps + 1)**3` ("keep_n"), `n_em >= 1`, `n_iter >= 1`,
   `min_kappa >= 0`, `min_pixels >= 1`, `n_per_iteration` None or `>= 1`,
   `verbose` in {0, 1, 2}; `bool` rejected where an int is expected; (4)
   `average` in {"mean", "center", "vmf", "watson"} ("average"), `pc` in
   {"grain", "single"} ("pc"); (5) masks (amended 2026-10-06, spec review:
   `dictionary_indexing`'s checks at `ebsd.py:2505-2522` test the
   navigation-mask shape before the type, so a list raises
   `AttributeError`, and check only the type of the signal mask), each mask
   in this order: a NumPy array ("navigation mask must be a NumPy array" /
   "signal mask must be a NumPy array"), dtype bool ("boolean"), the
   navigation shape `(ny, nx)` or the signal shape ("navigation mask shape"
   / "signal mask shape"), the navigation mask not all True ("at least one
   pattern"); (6) compat (K11): an absent point
   ("emsoft_compatible requires every map point"), several phases ("one
   phase"), a multi-PC detector with `pc="grain"` ("pc='single'"), a proper
   point group that has no EMsoft number or whose EMsoft operator set
   differs from orix's ("point group"; whether any orix proper group
   differs is recorded at Stage A from the operator-table comparison; the
   branch is tested by patching the EMsoft table). The free functions apply
   their subsets with the same fragments.
6. **Warnings** (`UserWarning`, at most once per call): the GROD coverage
   warning (fragments "misorientation ball" and "max GROD"; D5.8, D20.3;
   in `EBSD.hrosm` issued by the driver after the skip decisions and before
   the ball is built or anything is simulated, counting only grains that
   will be re-indexed; amended 2026-10-06, spec review); grains of a phase
   without the master (D8.14); no grain re-indexed (all-NaN OSM returned).
7. **Style**: `from __future__ import annotations`; hints `X | Y`, never
   `Optional`/`Union`; numpydoc; comment/docstring lines <= 72 characters;
   `print` only behind `verbose`, as `dictionary_indexing`/
   `spherical_indexing` do; no `print` in tests.
8. **Test modules** (recorded): `tests/test_indexing/
   test_hrosm_{kam,segmentation,averaging,sampling,osm}.py`,
   `tests/test_indexing/test_hrosm_emsoft_regression.py`,
   `tests/test_signals/test_ebsd_hrosm.py`, additions to
   `test_orientation_similarity_map.py` and `test_dictionary_indexing.py`.
   `tests/` has no `__init__.py` and runs with `--import-mode=importlib`,
   so shared generators are root-`conftest.py` functions exposed as
   fixtures.
9. **Map grid (new, amended 2026-10-06, spec review; grid over ALL
   points amended 2026-10-06, spec review round 2).** orix computes
   `xmap.shape`, `xmap.size`, `xmap.row` and `xmap.col` over the points
   in the data only, and `row`/`col` subtract their minimum
   (`crystal_map.py:339-429`, orix 0.14.2): a (3, 4) map whose row 0 is
   not in the data reports `shape (2, 4)`, `size 8` and `row` from 0;
   with the last column absent `(3, 3)`; with one in-data point of 12
   `(1, 1)` (measured 2026-10-06, round 2, orix 0.12.1 and 0.14.2).
   orix also flattens a one-row and a one-column map to `xmap.shape ==
   (n,)` and a one-point map to `()` (`xmap.row` raises "not enough
   values to unpack" there). Every HROSM function reads the 2-D grid
   from one private helper `_map_grid(xmap) -> (grid, (ny, nx))` in
   `_hrosm/_grains.py`, built over every point, in the data or not:
   - `shape = tuple(xmap._original_shape)` (orix's all-points shape from
     the coordinates, set at construction; present in orix 0.12.1 and
     0.14.2, and already read by kikuchipy at
     `_merge_crystal_maps.py:104`); `len(shape) == 2`: `(ny, nx) =
     shape`; `len(shape) == 1`: `(n, 1)` if `xmap.x is None` else `(1,
     n)` (orix's own `row`/`col` rule, `crystal_map.py:395-400`);
     `len(shape) == 0`: `(1, 1)`.
   - `grid = np.full(ny * nx, -1, np.int64); grid[xmap.is_in_data] =
     np.arange(xmap.size)`, reshaped to `(ny, nx)`: the index into the
     in-data arrays (`xmap.rotations`, `xmap.phase_id`,
     `xmap.prop[...]`), -1 where the point is not in the data. No
     `xmap.size == 1` shortcut, no minimum shift.
   So absent edge rows and columns keep their place: the map that
   `dictionary_indexing(navigation_mask=...)` writes
   (`_dictionary_indexing.py:141-144`, `is_in_data = ~mask` on the full
   coordinate arrays) stays aligned with the signal. Outputs are `(ny,
   nx)`, so 1 x n, n x 1 and 1 x 1 maps return `(1, n)`, `(n, 1)` and
   `(1, 1)`; tests build such maps with `create_coordinate_arrays`. The
   `hrebsd-dic` `_hrebsd/_segmentation.py:283-312` helper (in-data
   `row`/`col`) has the shrinking flaw and is not followed. Recorded
   consequence: orix slicing (`xmap[r0:r1, c0:c1]`, `__getitem__`
   `:687`) only clears `is_in_data`, so a sliced map keeps the original
   grid with -1 outside the slice; `EBSD.hrosm` accepts it against the
   unsliced signal (the sliced-away points are absent) and rejects it
   against a sliced signal with "xmap shape" (`plan.md` 7.4 item 27).

### D2 -- `GrainTable` and the `CrystalMap` output (frozen; output form R1)

1. **`GrainTable`**, frozen dataclass in `_grains.py`, grains in label order
   (index `i` = label `i + 1`):

   | field | type, shape | meaning |
   |---|---|---|
   | `n_pixels` | int64 (n,) | pixels per grain after any dilation |
   | `bounding_box` | int64 (n, 4) | row0, col0, height, width, 0-based |
   | `rotation` | `Rotation` (n,) | grain average; identity where not `valid` |
   | `phase_id` | int32 (n,) | phase of the grain's pixels; -1 for a label without pixels |
   | `kappa` | float64 (n,) | concentration; -1.0 rejected; 1.0 for `center` |
   | `max_grod` | float32 (n,) | max GROD [deg]; NaN where not `valid` |
   | `valid` | bool (n,) | `kappa != -1` |
   | `method` | str or None | averaging method; None when rebuilt |

   Property `n_grains`; classmethod `from_crystal_map(xmap)` rebuilds the
   table exactly from the props of D2.3 (first pixel per label; `n_pixels`,
   `bounding_box` recomputed from `grain_id`), since VMF/Watson results are
   stochastic. **A label without pixels** (possible only after compat
   dilate, D4.5-D4.6; amended 2026-10-06, spec review round 2) has
   `n_pixels` 0, `bounding_box` `(0, 0, 0, 0)`, the identity `rotation`,
   `phase_id` -1, `kappa` -1.0, `max_grod` NaN and `valid` False, in
   `average_grain_orientations` and in `from_crystal_map` alike (no
   pixel carries its props, so these are the defaults); `n_grains` =
   `grain_id.max()`, recoverable from the map because the highest label
   never vanishes under compat dilate (no larger neighbour overwrites
   it). Renamed from the parked `orientation`/`indexed` (amended
   2026-10-06): one `Orientation` cannot carry several phases' symmetries,
   and `indexed` reads like `CrystalMap.is_indexed`. EMsoft's `grainROI` is
   `(x0, y0, w, h)`, 1-based, inclusive (`mod_cluster.f90:382-398`), so
   `bounding_box = (y0 - 1, x0 - 1, h, w)`.
2. **`EBSD.hrosm` returns ONE `CrystalMap`** (R1) with the input's
   coordinates, phase ids, phase list, `is_in_data` and scan unit; the
   input's props are not carried (no stale `scores`). One rotation per
   point: the grain ball's best match where re-indexed, elsewhere the input
   rotation (correct) or the identity (compat, K12: EMsoft leaves `newEuler
   = 0`, `mod_HROSM.f90:546-551`).
3. **Properties** (`n` = map size):

   | prop | dtype, shape | correct: not re-indexed / outside grains | compat |
   |---|---|---|---|
   | `osm` | float32 (n,) | NaN | 0.0 (K12) |
   | `scores` | float32 (n, keep_n) | NaN row | 0.0 row |
   | `simulation_indices` | int32 (n, keep_n) | -1 row | -1 row |
   | `grain_id` | int32 (n,) | 0 = unassigned, grains 1..n (R3) | same |
   | `kam` | float32 (n,) | NaN if absent or no valid neighbour | EMsoft value |
   | `grod` | float32 (n,) | NaN outside grains and invalid grains | same |
   | `reindexed` | bool (n,) | False | same |
   | `grain_orientation` | float64 (n, 4) | NaN outside grains | same |
   | `grain_kappa` | float64 (n,) | NaN outside grains | same |
   | `grain_max_grod` | float32 (n,) | NaN outside grains | same |

   `simulation_indices` are 0-based into THAT grain's ball,
   `misorientation_ball(GrainTable.rotation[g], ...)` (grain-local).
   EMsoft's `newCI` (`mod_HROSM.f90:633, 759`) is `scores[:, 0]`, not
   duplicated. `kam`, `grod` in degrees. The parked `grain_n_pixels`
   broadcast is dropped (amended 2026-10-06; recomputable). orix 0.14.2
   and orix 0.12.1 (with numpy 1.23.0, isolated py3.10) round-trip bool
   `(n,)`, float32 `(n, 5)`, float64 `(n, 4)` and int32 `(n,)` props through
   `orix.io.save/load` (measured 2026-10-06; 0.12.1 verified in the spec
   review, so no version gate).

Pins: V16 output contract, `TestGrainTable` round trip.

### D3 -- Kernel average misorientation, KAM (frozen; parity level MTP)

1. **Correct mode** (Johan's EMsoftOO `c85982a` "Parallelize KAM and grain
   orientation averaging", which rewrote `getKAMMap` as a mean over existing
   neighbours, extended): per present pixel, the mean disorientation angle
   to its 4 nearest neighbours that exist, are present (D8.1) and share its
   phase id; NaN if none (`c85982a` leaves 0; recorded deviation). Pair
   angle `2 arccos(d')` in float64 with `d = max_j |<S_j o_a, o_b>|`, `S_j`
   the phase's `point_group.proper_subgroup` (single-sided, equal to orix
   `angle_with`; `2 arccos d` follows EMsoft rather than orix's `arccos(2d^2
   - 1)`), and the **near-one snap** `d' = 1.0 if d >= 1 - 4 eps else d`,
   `eps = np.finfo(np.float64).eps` (amended 2026-10-06, spec review round
   2; it replaces the clip to [0, 1], since `d >= 0` as a maximum of
   absolute values and every `d > 1` is snapped). Why: for identical
   orientations the identity term is the computed `|o|^2`, which rounds
   below 1 (by at most 2 eps) for 4.7-5.0 % and above 1 for 33-53 % of
   random float32-Euler rotations, unscrambled or left-scrambled
   (measured 2026-10-06 on 10,000 and 100,000 rotations); unsnapped that
   gives up to 3.4e-6 deg after `float32(float32(rad) * rtod)` or NaN,
   snapped exactly 0.0, as orix `angle_with` gives on all of them. Angles
   below `2 arccos(1 - 4 eps)` = 8.4e-8 rad (4.8e-6 deg) become 0. One
   private helper `_dot_to_angle(d) -> float64 radians` in `_kam.py`
   holds the snap and the `2 arccos`; `_correct_kam` and `_grod_map`
   (D20.2) both call it. Output `float32(float32(mean) * rtod)` degrees, or
   `float32(mean)` radians with `degrees=False`; `rtod = 180/pi` float64.
   Several rotations per point: the first is used.
2. **Compat mode (K1 + K2), normative transcription** of `getKAMMap`
   (`mod_DIsupport.f90:358-466`) as called by `mod_cluster.f90:152-153` and
   `mod_DIfiles.f90:2104-2120`. `W = ipf_wd` columns (x fastest), `H =
   ipf_ht` rows, `n = W H`, Euler float32 radians promoted with `dble`, flat
   1-based `t = (y - 1) W + x`:

   ```
   acc(1..n) = 0 (float64); lstore(1..W) = Euler (0, 0, 0); pstore = same
   for t = 1..n:
       ii = t mod W; if ii == 0: ii = W        # column, 1-based
       jj = t div W + 1                        # row; one too large if ii == W
       if ii == 1 and jj > 1: lstore = pstore  # previous row, copied first
       if ii == 1: cp = e(t); pstore(ii) = cp
       else: lp = cp; cp = e(t); pstore(ii) = cp
             d = dis(lp, cp); acc(t - 1) += d; acc(t) += d
       if jj > 1: d = dis(lstore(ii), cp); acc(t - W + 1) += d; acc(t) += d
   ```

   So the vertical pair of rows y-1, y at column x goes to `t - W + 1`
   (pixel (x+1, y-1); for x = W pixel (1, y)) instead of `t - W`; at t = W
   `jj = 2` while `lstore` is still zero, so pixel (W, 1) is compared with
   the identity and that angle is added to (1, 1) and (W, 1). Then the
   multipliers of D3.4, `kam = sngl(acc)` and (callers) `kam_deg =
   float32(float64(kam) * rtod)`. **Input route (amended 2026-10-06, spec
   review; frozen):** compat reads the first rotation per point as
   `xmap.rotations.to_euler()` cast to float32, then `eq_` in float64
   (EMsoft stores Euler angles as float32 and promotes them with `dble`).
   orix `Rotation.from_euler(eu32.astype(float64))` is NOT bitwise equal to
   `eq_`: it differs by 1 float64 ulp in 8,930 of 28,086 rows of the Ni6
   `RefinedEulerAngles` (8,404 for `EulerAngles`; 29.6 % of 100,000 random
   float32 triples), which moves compat KAM on a few pixels; the round trip
   `Rotation.from_euler(eu32.astype(float64)).to_euler().astype(float32)
   == eu32` holds bitwise on every row of the Ni6 (28,086 x 2), GRX810
   (139,181) and Al (501,592) files, and `eq_` of the round trip equals
   `eq_(eu32)` bitwise there. `dis(a, b)` (K2) is `getDisorientation_`
   without `fix1` (`mod_so3.f90:4190-4213`): `Mu = eq_(a)`, `qu = eq_(b)`
   in float64 (`eq_` = `(cPhi cp, -sPhi cm, -sPhi sm, -cPhi sp)` from the
   half-angle cos/sin of `Phi`, `phi1 - phi2`, `phi1 + phi2`, negated if
   the first component is negative; `epsijk = +1`); `ac = 1000.0`
   (`:4163`); for every `j`, `k` of the EMsoft operator list, `Mus = S_j *
   Mu`, `qus = S_k * qu` in `quatmult` order (component 0 `(a0 b0 - a1 b1)
   - (a2 b2 + a3 b3)`, 1 `(a0 b1 + a1 b0) + (a2 b3 - a3 b2)`, 2 `(a0 b2 +
   a2 b0) + (a3 b1 - a1 b3)`, 3 `(a0 b3 + a3 b0) + (a1 b2 - a2 b1)`), `x =
   |scalar(Mus * conj(qus))|` (the reversed product has the same scalar bit
   for bit: one evaluation), `a = 2.0 * acos(x)` UNCLIPPED, `if a < ac: ac
   = a`. A NaN (`x > 1` by rounding) is never selected; if all
   near-identity combinations round above 1, the pair gets the smallest
   non-identity symmetry angle (90 deg for cubic), observed in the oracles
   (D3.6). Operator values are EMsoft's (`sq22 = 0.7071067811865475244D0`,
   `sq32 = 0.8660254037844386467D0`, `half = 0.5`, exact zeros) selected
   by `QSym_Init_` from `PGrot(pgnum)`; e.g. `432` (prot 30) is
   `SYM_Qsymop` columns 1, 5-10, 17-24, 2-4, 11-16 in that order.
3. **Vectorised form** (normative for W >= 2; the loop is the oracle and
   covers W = 1). 0-based `[y, x]`, flat `p = y W + x`:

   ```python
   def emsoft_neighbour_sum(pair_h, pair_v, spurious, H, W, dtype):
       # pair_h[y, x]: (y, x)-(y, x + 1), (H, W - 1)
       # pair_v[y, x]: (y, x)-(y + 1, x), (H - 1, W)
       # spurious: dis(identity, q[0, W - 1]) for KAM, 0 for OSM
       left = np.zeros((H, W), dtype); left[:, 1:] = pair_h
       up = np.zeros((H, W), dtype); up[1:, :] = pair_v
       up[0, W - 1] = spurious
       right = np.zeros((H, W), dtype); right[:, :-1] = pair_h
       v = np.zeros(H * W, dtype); v[: (H - 1) * W] = pair_v.ravel()
       down = np.zeros(H * W, dtype); down[1:] = v[:-1]  # from flat p - 1
       down[0] = spurious
       # EMsoft's accumulation order per pixel: left, up, right, down
       return ((left + up) + right).ravel() + down
   ```

   For W == 1 pixel t gets `2 dis(e(t - 1), e(t))`, `e(0)` the identity
   (KAM) or the zero list (OSM).
4. **Multiplier sequence** (`mod_DIsupport.f90:437-455` KAM, `:246-264`
   OSM) on the flat array, 0-based, in source order; `third(x) = (x * 4) /
   3` (KAM float64; OSM float32 per D7.3):

   ```python
   acc *= 0.25
   acc[1 : W - 1] = third(acc[1 : W - 1])                  # top interior
   acc[n - W + 1 : n - 1] = third(acc[n - W + 1 : n - 1])  # bottom
   for jj in range(1, H - 1):
       acc[W * jj] = third(acc[W * jj])                    # left column
   for jj in range(2, H):
       acc[W * jj - 1] = third(acc[W * jj - 1])            # right column
   acc[0] *= 4.0; acc[W - 1] *= 2.0; acc[n - 1] *= 2.0
   acc[n - W] = third(acc[n - W])
   ```

   Effective divisors for W, H >= 2 (1-based (x, y)): interior 4; edges 3;
   (1, 1) 1 (a sum incl. the identity term); (W, 1) 2; (W, H) 2; (1, H) 3.
   One-row/one-column maps get EMsoft's overlapping multipliers (H = 1
   multiplies the interior by (4/3)^2), reproduced by the sequence.
5. **Only the bottom-row interior and (W, H) are right** in EMsoft's KAM;
   interior pixels use their left neighbour's "down" pair; left-column
   pixels take the right edge of the previous row. The bookkeeping is
   inherited from EMsoft 5 (`EMsoftHDFLib/commonmod.f90:604, 621`) and
   shared by the OSM (D7).
6. **Parity policy (amended 2026-10-06 from the parked "bitwise"; seeds
   amended 2026-10-06, spec review).** A pixel is DEGENERATE if a pair
   credited to it (left, right, up, shifted down, spurious) has a
   correct-mode angle below 1e-6 rad. Compat KAM is pinned on
   non-degenerate pixels as a count of differing float32 values plus a
   maximum ulp distance (MTP; target 0; measured with a float64 NumPy
   literal transcription: Ni6 HROSM `kam` 6 of 27,992 at <= 2 float32 ulp;
   Ni6 DI `KAM` 0 of 6,774; GRX810 DI `KAM` 0 of 4,182; Al DI `KAM` 16 of
   210,294 at <= 2 ulp). On degenerate pixels the binary's value depends
   on last-ulp rounding of 576 symmetry products (possibly FMA in the ifx
   build): the pair gives about 0 or the smallest non-identity angle;
   differing counts are recorded, never pinned (seeds: Ni6 HROSM 37 of 94,
   max difference 22.5 deg = 90/4; Ni6 DI 7,079 of 21,312; GRX810 DI 358 of
   134,999; Al DI 9,838 of 291,298, max 90.0 deg). Fallback if a
   non-degenerate pin fails: <= 2 float32 ulp, count pinned.
7. **Inputs**: correct mode accepts absent points and several phases
   (pairs across phases or touching an absent point skipped); compat needs
   a dense single-phase map (K11). 1 x n, n x 1 and 1 x 1 maps work through
   `_map_grid` (D1.9) and return `(1, n)`, `(n, 1)`, `(1, 1)` (correct 1 x
   1 is NaN; amended 2026-10-06, spec review).
8. **Cost**: correct, `2n` pair angles over `|G|` operators, vectorised;
   compat, `|G|^2` products per pair (576 cubic): measured 2026-10-06 0.29
   s for the Ni6 map (28,086 px) in NumPy; Al (501,592 px) ~5 s estimate.

Pins: V1 (compat vs the loop, incl. 1-row/1-col maps), V2 (shipped DI
reference; local Ni6/GRX810/Al), V3 (analytic constant-curvature field,
orix `angle_with` bookkeeping).

### D4 -- Segmentation, dilate, centre pixel, bounding boxes (frozen; rule R6)

1. **Rule (both modes, R6)**, closed form of `grow_region_driver_`/
   `grow_region_` (`mod_cluster.f90:445-562`; `0b8e885`, 2025-06-25,
   "replaces recursive region growing by iterative region growing"): graph
   on pixels with finite KAM (and, with `phase_id`, equal ids >= 0), edges
   between 8-neighbours with `|kam_a - kam_b| <= threshold` (float32; EMsoft
   `gangle` is float32); grains = components of size >= 2 containing a
   pixel with `kam <= threshold`; labels 1..n in increasing raster order (y
   outer, x inner) of each component's FIRST pixel with `kam <= threshold`;
   all else 0. Derivation: a seed with `kam > gangle` decrements `nGrains`,
   the driver's count test restores it, the seed stays 0 and can be
   absorbed (`:526-529, 471-475`); labels are set on push and the criterion
   is symmetric, so a flood fill equals the component; a singleton seed has
   no edge (a qualifying neighbour would have joined it, or it that
   neighbour's grain); `-1` singletons are reset to 0 (`:167-169`); NaN
   fails every comparison (isolated, 0). Measured 2026-10-06: the closed
   form plus K3 dilate on the Ni6 HROSM file's own `kam` (gangle 10.0)
   equals its `grainID` bitwise (62 grains, 0 of 28,086 pixels differ;
   `npixels`, `grainROI` equal).
2. **Implementation**: `scipy.sparse.csgraph.connected_components` over the
   offsets (0, 1), (1, 0), (1, 1), (1, -1), then relabelling; O(n). The
   flood fill is a test oracle only.
3. **`threshold`** is in the KAM's unit (degrees), compared in float32 when
   `kam` is float32.
4. **Not reproduced**: the 1e6 stack limit (`:42, 548-556`; a large grain
   left partial); `nGrains = -1` when no grain is accepted and the last
   raster pixel is a lone high-KAM seed (`:471-481`; we report 0 grains);
   the O(n^2) `count()` per seed.
5. **Dilate.** Compat (K3, `grain_dilate_`, `:403-442`, applied at
   `:209-212`, boxes re-run): `M = maximum_filter(g, size=3,
   mode="constant", cval=0)`; `out = g.copy(); out[1:, 1:] =
   np.where(M[1:, 1:] != 0, M[1:, 1:], g[1:, 1:])` (simultaneous; first row
   and column untouched; labels overwritten by larger neighbours; a small
   grain can vanish). Correct: unassigned present pixels take the largest
   same-phase 8-neighbour label (simultaneous); labels never overwritten.
   `n_pixels` and boxes after dilation in both modes (`:215-218`).
6. **`grain_bounding_boxes`**: labels 1..max, (row0, col0, height, width)
   int64; a label without pixels (possible after compat dilate, where
   EMsoft's `getROI_` would return a negative width) gets the box `(0, 0,
   0, 0)` and the `GrainTable` defaults of D2.1 (`phase_id` -1, `kappa =
   -1.0`, `valid = False`; amended 2026-10-06, spec review round 2), and
   is never re-indexed.
7. **Leakage (Notes)**: KAM is about theta/4 on both sides of a straight
   boundary of angle theta, so boundaries below ~4x the threshold can be
   crossed by the chained rule, which compares KAM values, not orientations.

Pins: V4 (compat transcription incl. ties at the threshold), V5
(synthetic two/three-grain maps), V6 (dilate, centre), V12 (shipped
references; local Ni6).

### D5 -- Grain averaging and the GROD pre-check (frozen; default R2, K5 R5)

1. **Methods** `"mean"` (`EBSD.hrosm` default, R2), `"center"`, `"vmf"`,
   `"watson"` (EMsoft `orav` `center` (its default), `averageVMF`,
   `averageWAT`; parity tests pass `average="center"`). Pixels in raster
   order (`mod_cluster.f90:269-283`), quaternions with q0 >= 0, no prior
   variant reduction.
2. **`mean`** (deterministic, both modes): each pixel replaced by `S_j o_i`
   maximising `|<S_j o_i, o_ref>|` (`o_ref` = correct-mode centre pixel),
   sign fixed to a non-negative dot, arithmetic mean normalised (not
   Markley's eigenvector; equal to O(spread^2)). `kappa` from `y = |sum| /
   N` through the VMF rule of D5.4, never gated. Why: orix
   `Quaternion.mean` (Markley) is not symmetry-aware: 49 deg off on
   scrambled variants, 0.23 deg after aligning (parked measurement).
3. **`center`**: correct, the grain pixel nearest the centroid (row, col),
   lowest flat index on ties, `kappa = 1.0`; compat (K4, `:228-241`), pixel
   `(x0 + w // 2, y0 + h // 2)` of the 1-based box, possibly outside the
   grain, raw `eq_` quaternion, `kappa = 1.0`.
4. **VMF/Watson EM, correct** (`mod_dirstats.f90:805-1285`; Chen et al.
   2015): `n_em` inits, `mu` = normalised `Generator.standard_normal(4)`,
   q0 >= 0, `kappa = 30` (`:871-877`). E-step in log space, `log R(n, j) =
   logCp(kappa) + kappa f(<S_j mu, x_n>)` normalised over `j` by
   log-sum-exp, `f(t) = t` (VMF), `t^2` (Watson). M-step on `v_nj =
   conj(S_j) x_n`: VMF `gamma = sum R v`, `mu = gamma / |gamma|`, `y =
   |gamma| / N`; Watson `T = (1/N) sum R v v^T`, `mu` = top eigenvector
   (`numpy.linalg.eigh`), `y` = top eigenvalue. `kappa`: for `y >= 0.94`
   VMF `(15 - 3y + sqrt(15 + 90y + 39y^2)) / (16(1 - y))`, Watson `(5y - 11
   - sqrt(39 - 12y + 9y^2)) / (8(y - 1))`; else the nearest entry of the
   table (amended 2026-10-06, spec review: EMsoft's operation order, which
   differs from `0.001 i` in float64 for 11,504 of 35,000 entries) `xAp =
   0.001 + (i - 1) * 0.001`, `i = 1..35000`, `yAp = I2(k) / I1(k)` (VMF) or
   `(I1(k/2) / (I0(k/2) - I1(k/2))) / k` (Watson, divided by `k` last),
   first minimum, index 1 -> 2 (`mod_dirstats.f90:243-261, 1112-1120`),
   Bessel ratios from `scipy.special.ive` (the scaling cancels in each
   ratio).
   `logCp` (`:1244-1285`): VMF `kappa > 30`: `C2 - kappa +
   log(kappa^4.5 / (-105 + 8 kappa (-15 + 16 kappa (-3 + 8 kappa))))`, else
   `C + log(kappa / I1(kappa))`; Watson `kappa > 20`: `C2W - kappa +
   log(kappa^4.5 / (525 + 4 kappa (45 + 8 kappa (3 + 4 kappa))))`, else
   `-kappa/2 - log(I0(kappa/2) - I1(kappa/2))`; `C =
   -3.675754132818690967`, `C2 = 4.1746562059854348688`, `C2W =
   5.4243952068443172530`. `L = sum_n logsumexp_j(...) - log|G|`, `Q = sum
   R log Phi` (`0 log 0 = 0`); stop when `i >= 2` and `|Q_i - Q_{i-1}| <
   0.01` (`:922-929`); best init = largest finite `L`, first on ties; final
   `mu` per D5.6 (left side).
5. **Compat EM (K5-K7)**, literal transcription in EMsoft operator order:
   E-step centres `mu * S_j` (`:1002`), M-step `x_n * conj(S_j)` (`:1051,
   1068`) (K5, right side); `getQandL` (`:1125-1174`) uses `S_j * mu` (left,
   `:1162`), VMF `C = exp(logCp)` then `exp(C + kappa t)` (`:1156`), `Phi /
   |G|`, keeps the previous `Q`, `L` when `min(Phi) <= 0` (`:1166-1173`);
   the previous values are `EMforDS_`'s `Qi`, `Li` (`:836`), uninitialised
   in the binary and 0.0 here (recorded deviation, K6); `Mu` not sign-fixed
   between iterations (`:918`); best init `maxloc(L_All)` = first maximum,
   NaN ignored, index 1 if all NaN (`:934`); final K7 (`:949-961`). At
   realistic `kappa ~ 1e4` the transcription reproduces the parked
   analysis without a separate idealisation: VMF `C` underflows, `exp(kappa
   t)` overflows, `Q` is NaN/Inf, all `n_iter` iterations run, the first
   init wins. Watson (amended 2026-10-06, spec review round 2): once `Phi`
   underflows for far variants, `getQandL` keeps the previous `Q` and
   `L`, i.e. `EMforDS_`'s `Qi`, `Li`, declared once (`:836`) and so
   carried across iterations AND inits; an init whose iteration-2 `Phi`
   underflows has `Q_2 = Q_1`, `L_2 = L_1` and exits at `i = 2`; an init
   whose iteration-1 `Phi` already underflows inherits the previous
   init's `L` (0.0 for the first init); per-init `L` and the chosen init
   are measured (V9, MTP), not frozen. This regime needs samples the
   right-sided model (K5) can align: LEFT-scrambled samples keep
   `kappa_hat` small in compat (14-17 in the round-2 critic's
   approximate emulation) and `Phi` never underflows.
   **Diagnostics record (named 2026-10-06, spec review; private):** both
   EM routines (`_em_correct`, `_em_emsoft`) return `EMResult`, a
   `typing.NamedTuple` in `_directional_statistics.py` with `mu` float64
   `(4,)` (q0 >= 0, after D5.6), `kappa` float, `log_likelihood` float64
   `(n_em,)` (each init's final `L`), `n_iterations` int64 `(n_em,)` and
   `best_init` int (0-based); V9 reads it.
6. **Final representative (K7; amended 2026-10-06).** EMsoft takes the
   first `mu * S_i` (EMsoft order) whose Rodrigues vector `IsinsideFZ`
   (`mod_so3.f90:818-882`). For a proper group the Rodrigues FZ is the
   Voronoi cell of the identity (`q in RFZ <=> |q0| >= |<q, s>|` for all
   `s` in G) and `scalar(mu S_i) = scalar(S_i mu)`, so compat takes `mu *
   S_i` with the largest `|q0|`, first index on ties (boundary ties
   measure-zero), q0 >= 0; correct takes `S_i * mu` (left, an equivalent
   orientation under EMsoft's own convention). No FZ test is ported.
7. **Gate**: `vmf`/`watson` keep a grain if `kappa > min_kappa` (strict;
   NaN fails, +Inf passes; `mod_cluster.f90:299-306`), else identity,
   `kappa = -1.0`, `valid = False`, skipped by the driver
   (`mod_HROSM.f90:570`). `mean`, `center` never gate. One private helper
   `_apply_kappa_gate(kappa: float, min_kappa: float) -> bool` (True =
   keep) holds the comparison (named 2026-10-06, spec review; V9 injects a
   NaN through it).
8. **GROD pre-check** (Johan's EMsoftOO `f6270d0` "EMHROSM: add GROD
   coverage precheck and persist maxGROD", not on EMsoftOO develop): per
   pixel the angle to its grain's reference orientation (D20.2; orix
   `angle_with` semantics, degrees, float32; EMsoft `fix1=.TRUE.` gives the
   same angle); `max_grod` per valid grain, both modes. **Warning contract
   (amended 2026-10-06, spec review; one contract for both call sites,
   replacing "the first ten labels"):** one private helper
   `_coverage_warning_message(labels, max_grod, n_pixels, max_angle,
   spacing=None) -> str | None` returns None when no entry has `max_grod >
   max_angle` (strict; compared in float64 as
   `np.asarray(max_grod).astype(np.float64) > float(max_angle)`, frozen
   2026-10-06, spec review round 2, so NumPy 1.23's value-based casting
   and NumPy 2's NEP 50 agree: a float32 array against a Python float
   would otherwise compare in float32), else the message of D20.3: fragments
   "misorientation ball" and "max GROD", the count, `max_angle`, the
   spacing clause only when `spacing` is given, the largest max GROD, up to
   ten `(label, max GROD, n_pixels)` entries in DESCENDING max GROD (ties
   by ascending label) and the hint. `average_grain_orientations` warns
   with it over every valid grain, without spacing (it has neither
   `n_steps` nor `min_pixels`); the driver calls the private averaging
   core, which never warns, and warns once over the grains it will
   re-index, with the spacing (D20.3). Pixels outside the ball saturate on
   its surface.
9. **Seeds**: `numpy.random.default_rng(seed)`; ONE generator consumed
   across grains in label order (EMsoft carries its seed across grains,
   `mod_cluster.f90:266-298`); `None` is non-deterministic; `mean` and
   `center` draw nothing.
10. **Not ported** (D11.3): the clock seed `values(8)` (`:266-267`; seed 0
    is a fixed point of the Lehmer generator, NaN), Burkardt's RNG,
    EMsoft's Bessel routines, `DSYEV` (eigenvector sign irrelevant for
    Watson), the `center.txt`/`VMF.txt`/`WAT.txt` files, `R_All`, the
    ignored `muhat` input.

Pins (amended 2026-10-06, spec review): V8 (recovery on samples of the
test-local exact sampler `sample_s3` scrambled by random LEFT operators,
spreads given as orix-equivalent `alpha` 20/100/1000 with VMF `kappa = 8
alpha` and Watson `kappa = 4 alpha`, N 30/300; orix
`Rotation.random_vonmises` is one cross-check arm only, being a Watson
density on the global `np.random` state and too slow at `alpha = 1000`;
right-side scrambling must fail in correct mode), V9 (compat vs a seeded
transcription; iteration counts at kappa 1e4), V12 (`wat` reference
`avor`/`kappa` within MTP bands), the GROD and coverage-warning tests (V8
`TestGROD`, V16 `TestMessages`).

### D6 -- Misorientation ball (frozen)

1. **Grid** (`sample_isoCubeFilled_`, `mod_so3.f90:1569-1635`): `w =
   max_angle` in radians, `edge = (pi (w - sin w))^(1/3) * 0.5` (`:1612`),
   `dx = edge / N`, cubochoric `(i, j, k) dx`, `i, j, k = -N..N`, `i`
   slowest, `k` fastest (0-based index `(i + N)(2N + 1)^2 + (j + N)(2N + 1)
   + (k + N)`): `(2N + 1)^3` points (68,921 at N = 20; EMsoft's "N^3"
   comments are wrong). The cube maps onto the homochoric ball whose
   surface is exactly `w`; shell `m = max(|i|, |j|, |k|)` holds `24 m^2 +
   2` points at the constant angle `w_m - sin w_m = (m/N)^3 (w - sin w)`.
   Measured 2026-10-06 (5 deg, N 20): `edge = 0.035163745447257796`, `dx =
   0.0017581872723628899`, max angle 5.000000000008 deg, shell 1 at
   0.24997 deg, within-shell spread <= 6e-12 deg, shell counts 1, 26, 98,
   ..., 9,602. No clipping, no FZ reduction.
2. **Conversions**: cu -> ho is an own port of EMsoft's
   `Lambert3DCubeForwardDouble` (`mod_Lambert.f90:1022-1113`) with
   `GetPyramidDouble` (`:1376-1432`), as `ch_` calls it
   (`mod_rotations.f90:5327-5353`); ho -> quaternion by orix
   `Rotation.from_homochoric` (public in 0.12.1 and 0.14.2). Test oracle:
   orix's private `orix.quaternion._conversions.cu2ho` (both versions) at
   1e-12.
3. **Composition** (`SampleIsoMisorientation_`, `mod_so3.f90:2118-2166`,
   formula `:2154`): `v' = (-v + rho0 + rho0 x v) / (1 + v.rho0)`, `rho0`
   the grain average's Rodrigues vector; with the Hamilton product
   (`epsijk = +1`, `mod_global.f90:107-108`) this is `q = conj(q_v) * q0`
   (verified 2026-10-06 with orix: `~Q_v * Q_0` matches the formula, `Q_0 *
   ~Q_v` and `Q_v * Q_0` do not). The port composes quaternions (no
   Rodrigues singularity at 180 deg, `:2156-2160`). The cube is
   centrosymmetric, so the SET equals `{q_v * q0}`; the order matters for
   the EMsampleRFZ oracle. `misorientation_ball(center)` element `i` is
   `~Q_v[i] * center`; `center=None` means the identity.
4. **Storage**: correct, float64 quaternions; compat (K8), each moved point
   as EMsoft's float32 Rodrigues 4-vector `(axis, tan(w/2))` (`FZarray`,
   `mod_DI.f90:1902, 2112-2116`; `(0, 0, 1, 0)` if `|v'| = 0`) converted
   back in float64 (`:2481-2483`), the source of the dictionary
   orientations and the output rotations (`mod_HROSM.f90:630-632`).
5. **Driver**: the raw grid once per call (68,921 x 32 B = 2.2 MB),
   composed per grain (D8.3).
6. **orix `get_sample_local` is not used**: it samples all of SO(3) first
   (orix 0.14.2 `sample_generators.py:137`, the default cubochoric branch;
   corrected 2026-10-06, spec review) and keeps angles below `grid_width` (about
   1.9e9 points, 61 GB at 0.25 deg) and multiplies `center * rot`.

Pins: V7 (count, shells, edge, orix `cu2ho` 1e-12, composition side;
`EMsampleRFZ` `MIS` set match at its 9-decimal text precision, N 6
shipped, N 20 bin-gated).

### D7 -- Orientation similarity map, OSM (frozen; public routing R7)

1. **Correct, grain-aware**: per re-indexed grain pixel, the mean over its
   4 nearest neighbours that exist, are in the same grain and were
   re-indexed, of `|top_n(p) & top_n(q)|` (first `n = n_osm` indices, set
   intersection, no rank weighting); integer counts, mean in float64,
   stored float32; NaN without such a neighbour; range 0..n (kikuchipy's
   `normalize=False` default since 0.5). Vectorised (sorted rows, broadcast
   equality). On a single-grain full map it equals the legacy
   `orientation_similarity_map` bitwise (`np.nanmean` of integer counts
   into float32, `_orientation_similarity_map.py:117-152`).
2. **Compat (K9)**, `getOrientationSimilarityMap` (`mod_DIsupport.f90:
   172-275`): the loop of D3.2 with `dis` replaced by `vectormatch`
   (`mod_math.f90:3620-3645`: entries of one list present in the other, no
   duplicates) on the first `lnm = min(nosm, nnk)` 1-based indices, float32
   accumulation (`localosm`, `:193`), multipliers of D3.4 in float32, NOT
   divided by `nosm` (range 0..n). The `lstore = 0` comparison adds 0
   (indices >= 1): vectorised form D3.3 with `spurious = 0`. Boxes with W
   = 1 or H = 1 get EMsoft's overlapping multipliers.
3. **Edge multiplier order (amended 2026-10-06).** Fortran lets
   `x * 4.0/3.0` become `x * (4.0/3.0)`, and the builds on record differ:
   GRX810 DI (6_0_20250527) and Al DI (6_0_20260813) match `(x * 4) / 3`
   in float32 bitwise (139,181/139,181 and 501,592/501,592); Ni6 DI
   (6_0_20260411, Windows ifx) matches `x * float32(4/3)` (28,086/28,086;
   214 edge pixels differ between the forms). Compat uses `(x * 4) / 3`
   (the Fortran text, two builds of three); V10 pins GRX810/Al bitwise and
   records Ni6 as "bitwise except the straight-edge pixels, 1 ulp"; V14
   measures the chosen binary first (the 2026-04-13 Release/Bin build is
   expected to fold). For KAM both orders give the same float32 (measured).
4. **Domain in the driver**: correct, the grain's re-indexed pixels;
   compat (K10), the whole bounding box in box raster order (pixels of
   other grains and unassigned pixels, matched against this ball, count as
   neighbours), copied back for the grain only (`mod_HROSM.f90:626-636`;
   box OSM at `mod_DI.f90:2578-2579`).
5. **Public routing (R7)**: `orientation_similarity_map` gains keyword-only
   `grain_id` and `emsoft_compatible`; at their defaults the legacy path
   runs unchanged (bitwise; existing tests untouched). Otherwise the new
   functions run; `footprint`, `center_index`, `from_n_best` must keep
   their defaults ("cannot be combined"); `n_best` sets `n`;
   `normalize=True` divides by `n` after the table; absent points and
   `keep_n == 1` work; result `(ny, nx)` float32, no `squeeze`. Legacy
   limitations recorded, not fixed: absent points raise in the `reshape`
   (`:104-105`), a squeezed `keep_n == 1` prop fails the unpack (`:97`),
   the result is squeezed (`:128`), and one-row, one-column and one-point
   maps raise `RuntimeError` "footprint.ndim (2) must match len(axes)"
   because orix collapses their shape (measured 2026-10-06, spec review;
   a (2, 5) map returns (2, 5)). The new paths take the grid from
   `_map_grid` (D1.9).

Pins: V10 (shipped `OSM`, `OSM_05`; local GRX810, Al, Ni6), V11 (equality
with the legacy function on one grain, cross-grain neighbours excluded,
absent points, `keep_n = 1`).

### D8 -- Per-grain driver and `EBSD.hrosm` (frozen; multi-phase R4, verbose R7)

1. **Absent points**: `is_in_data == False`, `phase_id == -1` or
   `navigation_mask == True` (True = excluded, `ebsd.py:2443-2447`): KAM
   NaN, label 0, never simulated or matched, input rotation kept;
   anywhere on the map, edge rows and columns included, since the grid
   spans every point (D1.9 as amended in spec review round 2). No
   navigation mask is passed to `_dictionary_indexing` (D8.5), so its
   `np.empty` fill of masked points (`_dictionary_indexing.py:149-154`)
   cannot leak.
2. **Orientation stage**: KAM -> `segment_grains_kam(phase_id=...)` ->
   boxes -> private averaging core (GROD per D20.2) -> skip decisions
   (D8.9, D8.14) -> `misorientation_ball_spacing` -> the one coverage
   warning (D20.3), all with the call's `emsoft_compatible` and all before
   the ball (D8.3) is built or anything is simulated.
3. **Ball**: raw grid once (D6.5); per grain `ball_g = ~R_v * avor_g` (orix
   `Rotation.__mul__`, Hamilton), compat then K8-rounded.
4. **Domain**: correct, the grain's present pixels (flat, raster order), so
   no other grain's pattern meets this ball; compat (K10), the box.
5. **Experimental block**: `self.data` as `(n_map, sig_size)` indexed by the
   domain, computed eagerly to float32 NumPy (dask fancy indexing for lazy
   signals), because the chunked loop re-evaluates the experimental graph
   per iteration (`_dictionary_indexing.py:105-117`). The metric is
   prepared per grain with `n_experimental_patterns` = block size
   (`EBSD._prepare_metric`, `ebsd.py:4459-4498`, sets the map size at
   `:4484`; the driver overrides it), `signal_mask` set, no navigation
   mask, `rechunk=False`.
6. **Dictionary per grain**: `master_pattern.get_patterns(ball_g, det_g,
   energy, dtype_out="float32", compute=False, chunk_shape=...)`
   (`ebsd_master_pattern.py:99-331`, lazy, dictionary `xmap` at `:290`),
   chunked along the dictionary axis by `n_per_iteration`; default
   `clip(floor(256e6 / (4 sig_size)), 1, ball size)` (module constant 256
   MB: 17,777 at 60 x 60, 4 iterations for 68,921; 277 at 480 x 480),
   computed by the private `_default_n_per_iteration(sig_size: int,
   ball_size: int) -> int` (named 2026-10-06, spec review; V16 tests it);
   the exact `chunk_shape` form is checked at Stage B.
7. **`_dictionary_indexing(..., verbose: bool = True)`** (R7, backwards
   compatible): False suppresses its two `print`s (`:77-85, 136-139`), the
   `tqdm` bar (`:105`), the `ProgressBar` (`:92`) and the unconditional
   `sleep(0.2)` before the speed message (`:135`; it only keeps the tqdm
   bar's background off that print, and would cost 0.2 s per grain;
   amended 2026-10-06, spec review round 2); not exposed by
   `EBSD.dictionary_indexing`. The driver passes `verbose >= 2`; at
   `verbose >= 1` it prints one info message, one `tqdm` over grains and
   the timing.
8. **PC**: a one-PC detector is used as is (EMsoft's model: one PC; the
   per-pattern PC code was removed in `92c1b21`, 2025-07-07, "invalid
   assumption"). With one PC per map point (`detector.navigation_shape ==
   (ny, nx)` from `_map_grid`, D1.9, not `xmap.shape`; amended 2026-10-06,
   spec review round 2; e.g. plane-fitted), `pc="grain"` simulates each
   grain with
   the mean PC of its domain (`det_g = detector.deepcopy(); det_g.pc =
   mean`; documented deviation), `pc="single"` with `detector.pc_average`
   (`_ebsd_detector.py:465`). `get_patterns` takes one PC or one per
   simulated pattern (`ebsd_master_pattern.py:194-198`), hence one detector
   per grain. Compat with a multi-PC detector requires `pc="single"` (K11).
9. **Skips**: `n_pixels < min_pixels` (EMsoft `< 10`, `mod_HROSM.f90:592`),
   `valid == False` (`:570`), a label without pixels, grains of a phase
   without the master (D8.14); they keep the fill of D2.3.
10. **Dictionary-batch thinning NOT reproduced (amended 2026-10-06; amends
    the parked K10).** `OSMDIdriver` simulates only `min(ppend, W*H)`
    patterns of every `Nd = numdictsingle` batch (`mod_DI.f90:2479`), so a
    box with `W*H < Nd` (default 1024) sees zero vectors for most of the
    ball (a 10 x 10 box keeps ~6,800 of 68,921 orientations, in streaks), a
    leftover of `92c1b21` (`DIdriver` has no cap). Not emulated: it depends
    on a GPU batch size with no kikuchipy meaning, would need a compat-only
    public keyword, and the dictionary stage is tolerance-based anyway. The
    reference run sets `numdictsingle = numexptsingle = 32`, so the binary
    thins only boxes below 32 pixels; V13 compares grains with `W*H >=
    numdictsingle` only.
11. **Results**: `rotations_g = ball_g[sim_idx[:, 0]]`; OSM per D7.4 from
    `sim_idx[:, :n_osm]`; copy back by flat index; `scores` float32,
    `simulation_indices` int32 (both paths of `_dictionary_indexing`
    return int64: `argtopk`, and the chunked loop's `np.hstack` promotes
    its int32 seed array of `:97`; the driver casts; corrected 2026-10-06,
    spec review). EMsoft's
    ROI +1 offset (`mod_HROSM.f90:470, 616`) does not apply (no DI ROI).
12. **Determinism**: all but VMF/Watson is deterministic; results must not
    change with `n_per_iteration` nor lazy vs eager (MTP: seed bitwise;
    fallback: identical top-1 except at exact score ties, scores within 2
    float32 ulp, recorded; BLAS may sum differently for other chunk
    shapes); dask threaded scheduler.
13. **Cost seeds** (MTP at Stage B, never gated): NCC ~ `n_px x 68,921 x
    sig_size x 2` flops: nickel_ebsd_large 2.0e12 (~40 s), ni_gain 1.5e13
    (~5 min), si_wafer (480 x 480) 7.9e13 (hours with simulation);
    simulation ~3 s per grain at 60 x 60; cost scales with `(2N + 1)^3`
    (`n_steps=10` is 7.4x cheaper); compat boxes add 20-50 % pixels.
14. **Multi-phase v1 (R4)**: one master; grains segmented per phase; a
    single-phase map uses the master whatever the names; on a multi-phase
    map the master is matched to `xmap.phases` by `Phase.name`, other
    phases' grains are skipped with one warning ("no master pattern"), no
    match raises ("master pattern phase").
15. **Private driver entry point (named 2026-10-06, spec review).**
    `_driver._hrosm(signal, xmap, master_pattern, detector, energy, *,
    grains: Sequence[int] | None = None, **keywords) -> CrystalMap`, the
    keywords being those of D1.3 after validation. `grains` (labels 1..n)
    restricts the re-indexing to those grains, all other grains taking
    the fills of D2.3 as if skipped; the orientation stage (KAM,
    segmentation, averaging, GROD, the coverage warning over the selected
    grains) is unchanged. Used by tests only (V13); `EBSD.hrosm` never
    passes it.

Pins: V13 (end-to-end tolerance vs EMHROSM), V15 (physics sanity), V16
(masks, PC policy, skips, invariances, determinism, verbose output,
warnings), the `_dictionary_indexing(verbose=False)` test.

### D9 -- The `emsoft_compatible` switch set (frozen)

1. One keyword, twelve behaviours; everything else is identical in both
   modes. EMsoft lines refer to EMsoftOO `develop` @ `c127868`; the
   computational core is unchanged since `975a1fc` (the Release/Bin build;
   inspected 2026-10-06: `mod_DIsupport.f90`, `mod_cluster.f90` differ only
   in headers, `mod_dirstats.f90` only by test output).

   | K | piece | correct (default) | `emsoft_compatible=True` | EMsoft source | applies in |
   |---|---|---|---|---|---|
   | K1 | KAM bookkeeping | mean over existing present same-phase 4-neighbours, NaN if none | 1-D loop (D3.2): vertical pair to `t - W + 1`, `jj` off by one in the last column, (W, 1) vs identity, multipliers D3.4 | `mod_DIsupport.f90:406-455` | KAM, segmentation, `hrosm` |
   | K2 | KAM pair angle | single-sided, `2 arccos` with the near-one snap of D3.1, float64 | literal double loop, EMsoft operators and `quatmult` order, unclipped `acos`, NaN never selected, `ac` from 1000 | `mod_so3.f90:4163-4213` | same |
   | K3 | dilate | unassigned present pixels take the largest same-phase neighbour label | 3 x 3 max for centres x in [2, W], y in [2, H] (1-based), overwrites labels | `mod_cluster.f90:403-442` | `segment_grains_kam(dilate=True)` |
   | K4 | `center` pixel | grain pixel nearest the centroid | box centre `(x0 + w // 2, y0 + h // 2)`, may lie outside | `mod_cluster.f90:228-241` | `average="center"` |
   | K5 | EM symmetry side | `S_j mu`, `conj(S_j) x` (left) | `mu S_j`, `x conj(S_j)` (right) | `mod_dirstats.f90:1002, 1051, 1068` | `vmf`, `watson` |
   | K6 | EM `Q`, `L` | log-sum-exp, left side, best finite `L` | `S_j mu` in `getQandL`, VMF `C = exp(logCp)`, stale `Q`/`L` from 0.0, `maxloc` first max | `mod_dirstats.f90:836, 934, 1125-1174` | same |
   | K7 | final representative | `S_i mu` with max `|q0|` | `mu S_i` with max `|q0|`, first in EMsoft order | `mod_dirstats.f90:949-961` | same |
   | K8 | ball storage | float64 quaternions | float32 Rodrigues 4-vector, back to float64 | `mod_DI.f90:1902, 2112-2116, 2481-2483` | `misorientation_ball`, `hrosm` |
   | K9 | OSM | grain-aware mean over in-grain re-indexed neighbours, NaN if none | EMsoft table, float32, un-normalised, `(x * 4) / 3` | `mod_DIsupport.f90:221-264` | `orientation_similarity_map`, `hrosm` |
   | K10 | indexing domain | the grain's pixels | the bounding box, grain pixels copied back; thinning not reproduced | `mod_HROSM.f90:615-636` | `hrosm` |
   | K11 | inputs | absent points, multi-phase, per-grain PC | dense single-phase map, one PC, EMsoft operator set; else `ValueError` | `mod_DI.f90` (one PC) | every compat entry point |
   | K12 | fill | input rotation; `osm`, `scores` NaN | identity; `osm`, `scores` 0.0 | `mod_HROSM.f90:546-551` | `hrosm` |

2. **Not reproduced in either mode**: quirk catalogue rows 8, 9, 17, 22-25,
   28, EMsoft's `nosm > nnk` clamp (`mod_DIsupport.f90:201-209`; here
   `n_osm <= keep_n` is validated), OpenCL/`SSORT` tie order, and the
   dictionary physics (Scope).
3. **No switch** for the segmentation rule (EMsoft's in both modes, R6;
   Johan's EMsoftOO branch fixed the KAM and kept the rule) or the GROD
   (not an EMsoft develop output).
4. **Docs**: the `EBSD.hrosm` Notes give this table in words ("Differences
   from EMsoftOO's EMHROSM"), the tutorial a short version; neither names a
   K-number (D14.5).

Pins: named tests per K row in V1-V12 and V16; the mutants of `plan.md`
section 6.

### D10 -- dtype, precision and kernel policy (frozen policy)

1. **HROSM scoping of the float64 rule** (`tech-stack.md:21` is
   EMSphInx-scoped, as NLPAR and HREBSD scoped it): orientation maths in
   float64; stored maps float32 (`kam`, `osm`, `grod`, `max_grod`; EMsoft
   stores float32); grain orientations float64; NCC in float32 (the
   metric's default); `get_patterns(dtype_out="float32")`.
2. **Compat precision mirrors EMsoft**: KAM pairs and sums float64, `sngl`,
   `float32(float64(kam) * rtod)`; OSM float32; EM float64; ball K8 float32
   Rodrigues; threshold comparisons float32 when KAM is float32.
3. **No numba kernels planned** (amended 2026-10-06, leanness: the compat
   double loop takes 0.29 s on 28,086 pixels in NumPy). A kernel is added
   only if a stage gate measures a hot loop over budget, then under
   `tech-stack.md:44` (`cache=True, nogil=True`, no `parallel`, no
   `fastmath`) with `.py_func` parity (V0 then applies).
4. **Near-one snap** (amended 2026-10-06, spec review round 2, from
   "Clipping"): correct mode sets the symmetry-reduced dot to 1 when it
   is `>= 1 - 4 eps` (D3.1), so `arccos` never sees a value above 1
   (`acos(1 + ulp)` is NaN, `mod_so3.f90:4199`) and identical
   orientations give exactly 0; the same helper serves KAM and GROD;
   compat does neither (K2).

### D11 -- Dependencies and licensing (frozen)

1. **No new dependency**: numpy, scipy (`sparse.csgraph.
   connected_components`, `ndimage.maximum_filter`, `special.ive`,
   `spatial.cKDTree` (D20.1; added 2026-10-06, spec review); `>=
   1.7`, `pyproject.toml:64`), dask, orix `>= 0.12.1`, h5py (reader) are
   required already; nothing EMsoft is imported at runtime.
2. **Headers**: every new module carries kikuchipy's GPL-3.0-or-later
   header. Modules with EMsoft-derived code (`_emsoft_quaternions`, `_kam`,
   `_segmentation`, `_averaging`, `_directional_statistics`, `_sampling`,
   `_osm`, `_driver`) add the delimited block of the
   `signals/util/_master_pattern.py:20-57` convention (`tech-stack.md:45`):
   the rationale line "The following copyright notice is included because
   the following functionality in this file is derived and adapted from
   EMsoftOO:" and the list of derived routines with their files; between
   `# ####` rules the EMsoftOO BSD-3 notice verbatim ("Copyright (c)
   2013-2026, Marc De Graef Research Group/Carnegie Mellon University", for
   `_directional_statistics.py` "2014-2026" as `mod_dirstats.f90:2`; "All
   rights reserved."; the three conditions and the disclaimer of
   `mod_HROSM.f90:1-27`); then "Changes by the kikuchipy developers, <date
   of the implementation commit>: ported from Fortran to NumPy; defects
   reproduced only behind `emsoft_compatible`." BSD-3 code may enter a GPL
   work with its notice kept.
3. **Not ported**: Burkardt's `r8_normal_01`, `r8_uniform_01`,
   `r8vec_normal_01` (`mod_math.f90:1763, 1868, 2109`, each "distributed
   under the GNU LGPL license") -> NumPy `Generator`; J-P Moreau's
   `BesselIn/I0/I1` (`:748-909`) -> `scipy.special`; SLATEC `SSORT` ->
   NumPy; LAPACK `DSYEV` -> `numpy.linalg.eigh`; `opencl/DictIndx.cl` ->
   kikuchipy NCC.
4. **Johan's EMsoftOO branch** `feature/emhrosm-grod-precheck` (`c85982a`
   KAM, `f6270d0` GROD, `fd52d53` OpenMP; 2026-02-13) is his own work; its
   semantics are re-implemented and acknowledged in the method Notes.
5. EMsoft binaries and files are used only by gated tests and the
   reference script (D13).

### D12 -- Private EMsoft file reader (frozen)

`_hrosm/_emsoft_file.py`, h5py only, for tests and the reference script.

1. **Dot-product files** (`Scan 1/EBSD/Data`, `NMLparameters/EMDINameList`,
   `EMheader/Version`): map shape `(ipf_ht, ipf_wd)`, or `(ROI(4), ROI(3))`
   when `sum(ROI) != 0` (`mod_cluster.f90:134-141`); `PointGroupNumber`;
   `EulerAngles` (N, 3) float32 radians = the top-1 dictionary orientation
   written as `float32(eulerarray * dtor)` (`mod_DIfiles.f90:2223-2235`),
   the exact input of the file's `KAM` (`:2104-2120`); `RefinedEulerAngles`
   (N, 3) float32 radians (`:1873-1883`); `TopMatchIndices` (N_pad, nnk)
   int32 1-based, padded to a multiple of `numexptsingle` (Ni6 28,672 vs
   28,086; GRX810 139,264 vs 139,181; Al 501,760 vs 501,592) -> `[:N]`,
   kept 1-based; `TopDotProductList`; `KAM` (H, W) float32 degrees; `OSM`,
   `OSM_nn` (H, W) float32; `CI`; namelist `nnk`, `nosm`, `ipf_wd`,
   `ipf_ht`, `ROI`, `numdictsingle`, `numexptsingle`, `xpc`, `ypc`, `L`,
   `delta`, `thetac`, `energymin`, `energymax`. Strings are object arrays
   of bytes.
2. **HROSM files** (`EMData/HROSM`, `EMheader/HROSM/Version`): `nGrains`,
   `grainID` (H, W) int32, `npixels`, `grainROI` (n, 4) int32 1-based `(x0,
   y0, w, h)`, `avor` (n, 4) float64, `kappa`, `kam` (H, W) float32
   degrees, `newOSM`, `newEuler` (H, W, 3) float32 radians, not
   FZ-reduced, 0 where not re-indexed, `newCI`; optional `newQuat` (develop
   builds after `975a1fc`) and `maxGROD` (Johan's branch). `dilate` is NOT
   in `NMLparameters/HROSMNameList`; it is parsed from the text
   `NMLfiles/HROSMNML`.
3. **Fortran order**: `(x, y)` arrays read as `(H, W)`, `(3, N)` as `(N,
   3)`; flat `y W + x` = EMsoft's `(iy - 1) W + ix` minus 1.

### D13 -- Reference data, oracles and gates (frozen policy; values MTP)

1. **Gates.**
   - **CI**: the default suite. **weekly**: `@pytest.mark.weekly`
     (`weekly.yml:88`, `pytest --weekly --reruns 2 -n 4`; that workflow
     is disabled on the fork, so these arms run in the local `--weekly`
     stage gate, D15.1 as amended in spec review round 2).
   - **bin**: `KIKUCHIPY_EMSOFT_BIN` names a directory with `EMDI.exe`,
     `EMFitOrientation.exe`, `EMHROSM.exe`, `EMgetOSM.exe`,
     `EMsampleRFZ.exe`, `EMsoftOOLib.dll`, `EMOpenCLLib.dll`; the data path
     is `EMdatapathname` of `~/.config/EMsoft/EMsoftConfig.json` (the
     binaries prepend it to every path); an OpenCL GPU is needed (EMDI,
     EMHROSM). Skip naming what is missing. Every run holds
     `_emsoft_program_lock(path=None)`, modelled on `conftest.py:702-750`
     `_emsphinx_program_lock` (exclusive create of
     `kikuchipy-emsoft-program.lock` in `tempfile.gettempdir()`; `path`
     only for its own unit test): the programs share `EMtmppathname`
     (`EMEBSDDict_tmp.data`) and one GPU. **HROSM constants and heartbeat
     (amended 2026-10-06, spec review: the EMSphInx 600 s wait and 900 s
     stale age are sized for programs of seconds, while the reference run
     holds the lock ~10-25 min and the EMSphInx lock never refreshes its
     mtime, so a second run would take it over):** `_EMSOFT_LOCK_TIMEOUT =
     3600.0` s; while held, a daemon thread touches the lock file
     (`os.utime`) every 30 s and stops on release; a lock whose mtime is
     older than `_EMSOFT_LOCK_STALE = 300.0` s belongs to a dead holder and
     is taken over. No PID test (on Windows `os.kill(pid, 0)` terminates
     the process). `create_hrosm_reference.py` carries the same constants
     and logic.
   - **local**: `KIKUCHIPY_EMSOFT_DATA` names an EMsoft data root holding
     the files of D13.5; each md5 asserted; skip naming the missing file.
   Both variables are new and distinct from `KIKUCHIPY_EMSPHINX_DIR`,
   `KIKUCHIPY_LOCAL_MASTERS_DIR`, `KIKUCHIPY_NO_GPU_TESTS` (`develop`) and
   `KIKUCHIPY_LOCAL_DATA_DIR` (`hrebsd-dic`).
2. **Script** `src/kikuchipy/data/emsoft_hrosm/create_hrosm_reference.py`,
   import safe like `create_emsphinx_reference.py:28-32` (no environment
   lookup or file access at import; tests import `SCENARIOS`); excluded
   from doctests by a new `--ignore-glob=src/kikuchipy/data/emsoft_hrosm/
   *.py` (`pyproject.toml`, next to `:181`) and from coverage by the
   existing `omit` (`:161`). `main()` runs in `<EMdatapathname>/
   kikuchipy_hrosm/<YYYYmmdd-HHMMSS>/`:
   1. Patterns: `nickel_ebsd_large`, static then dynamic background
      removed as in `pattern_matching.ipynb`, NORDIF writer
      (`io/plugins/nordif/_api.py:433`), EMsoft `inputtype 'NORDIF'`; flip
      decided by the acid band through EMDI's own `flipy` key (template
      default `.FALSE.`; one retry with `flipy = .TRUE.`, the patterns
      file unchanged; the value used is in the stored namelist and
      provenance; decided 2026-10-06, spec review).
   2. Master: copy of the cached `ni_mc_mp_20kv.h5` (md5
      `8b69c071a036ad3488d465093b67fe4d`; MC + master, sig 70, 20 kV,
      EMsoft `4_3_0_0`; EMsoftOO's readers guard namelist datasets with
      `H5Lexists`) into `<run>/`. **Crystal file (amended 2026-10-06, spec
      review):** EMDI (`mod_DI.f90:422-423`) and EMHROSM
      (`mod_HROSM.f90:497-499`) read the crystal from
      `<EMXtalFolderpathname>/<xtalname>` (`mod_crystallography.f90:2656`;
      no `useXtalName`), and the cached master's two `xtalname` datasets
      (`EMData/EBSDmaster/xtalname`, `NMLparameters/MCCLNameList/xtalname`)
      read `ni/ni.xtal`, a path that does not exist under
      `EMsoftData/Xtal` (no subdirectories). The script therefore asserts
      that `<EMXtalFolderpathname>/Ni.xtal` exists with
      `CrystalData/SpaceGroupNumber == 225` and `LatticeParameters[0]`
      within 1e-6 of 0.35236 nm (verified 2026-10-06; the file is
      Johan's `EMmkxtal` Ni of 2025-04-26), then rewrites both `xtalname`
      datasets IN THE COPY to `Ni.xtal` (h5py; both are variable-length
      ASCII strings of shape (1,), measured 2026-10-06, so the shorter
      value needs no padding; same dtype and shape). The
      cached original is never modified (`master_md5` pins it); the
      patched copy's md5 is recorded as `master_run_md5`. Pre-flight EMDI
      read; fallback EMMCOpenCL + EMEBSDmaster from the `DItutorial/Ni`
      namelists (+30-60 min), which changes `master_md5`: P3 and the V14
      `master_md5` pin are then amended in the same ledger entry.
   3. PC (corrected 2026-10-06, spec review: `pc_emsoft()` returns a 2-D
      `(n_pc, 3)` array, `_ebsd_detector.py:1704-1705`): `det.pc =
      det.pc_average`; `pc = det.pc_emsoft()[0]`; `xpc, ypc = pc[:2] /
      det.binning`; `L = pc[2]`; `delta = det.px_size * det.binning`;
      measured 2026-10-06: 4.6044, 17.1820, 240.9956, 8.0 (the parked
      4.62/17.16/240.96 were another PC); `thetac 0`, `omega 0`; `.6g`
      round trip stored.
   **Paths, working directory and output containment (amended 2026-10-06,
   spec review).** The binaries prepend `EMdatapathname` to every path key,
   and EMgetOSM and EMHROSM write a TIFF unconditionally
   (`mod_OSM.f90:479-491`, `mod_HROSM.f90:768-779`; both name keys default
   to `'undefined'`, so a template run writes `EMsoftData/undefined*`).
   Every path key of every namelist therefore starts with
   `kikuchipy_hrosm/<ts>/` (`<ts>` = `YYYYmmdd-HHMMSS`); each program runs
   with `cwd = <run>/` (EMHROSM's cluster stage writes `center.txt`/
   `WAT.txt` into the working directory, `mod_cluster.f90:229, 255`); the
   script lists the `EMdatapathname` root before the first program and
   aborts if a new entry appears after any program. Keys per program:
   EMDI `exptfile` (`.../patterns/Pattern.dat`), `masterfile`
   (`.../ni_mc_mp_20kv.h5`, the patched copy), `datafile` (`.../dp.h5`),
   `tmpfile 'EMEBSDDict_tmp.data'` (relative to `EMtmppathname`),
   `keeptmpfile 'n'`, `ctffile`/`angfile` left `'undefined'` (EMDI skips
   them, `mod_DI.f90:1652-1662`); EMFitOrientation `dotproductfile`
   (`.../dp.h5`), `newdotproductfile` (`.../dp-refined.h5`),
   `usemasterpatternfile` (`.../ni_mc_mp_20kv.h5`; required with
   `newdotproductfile`), `angfile` (`.../dp-refined.ang`; one of
   `angfile`/`ctffile` is required), `tmpfile
   'EMFitOrientation_tmp.data'`, `inRAM .FALSE.`
   (`mod_FitOrientation.f90:263-277`); EMgetOSM `dotproductfile`
   (`.../dp-refined.h5`), `tiffname` (`.../osm_`); EMHROSM `dpfile`
   (`.../dp-refined.h5`), `OSMfile` (`.../hrosm_<scenario>.h5`), `OSMtiff`
   (`.../hrosm_<scenario>.tiff`), `IPFmap`, `angfile`, `ctffile` left
   `'undefined'`; EMsampleRFZ `quoutname`, `euoutname`, `rooutname`
   (`.../ball_n<N>_{qu,eu,ro}.txt`; `mod_sampleRFZ.f90:595-619`).
   4. `EMDI.nml`: `ipf_wd 75, ipf_ht 55, ROI 0 0 0 0, nnk 20, nosm 10,
      nism 5, hipassw 0.05, nregions 4, maskpattern 'y', maskradius 29,
      ncubochoric 100, energymin 15, energymax 20, exptnumsx 60, exptnumsy
      60, numsx 60, numsy 60, binning 1, scalingmode 'not', numdictsingle
      32, numexptsingle 32` (Ne == Nd, multiples
      of 16, required by the OpenCL kernel), `nthreads 20, platid 1, devid
      1`. Acid band: top-1 median disorientation to the stored `xmap` <=
      ~1.5 deg (MTP).
   5. `EMFitOrientation.nml`: `method 'FIT', niter 1, nmis 1, step 0.03,
      matchdepth 1, PCcorrection 'off'`; acid band median <= 0.5 deg.
   6. `EMgetOSM.nml`: `nmatch = 10 5 0 0 0` -> `OSM_10`, `OSM_05`;
      self-check `OSM_10 == OSM` bitwise.
   7. `EMHROSM.nml`: `center` (gangle 5, misorang 5, nsamples 20, nosm 10,
      dilate .FALSE., orav 'center'), `center_dilate`, `wat` (orav
      'averageWAT', numEM 25, numIter 40); `maxRAMmem 1.0`.
   8. `EMsampleRFZ.nml`: `samplemode 'MIS', pgnum 32, maxmisor 5, nsteps
      6` (shipped) and `20` (bin-gated), `rodrigues` a generic centre (unit
      axis (1, 2, 3)/sqrt(14), magnitude 0.1), `quoutname`, `euoutname`,
      `rooutname` (text, 9 decimals; `ro` writes the UNSHIFTED grid, `qu`
      and `eu` the moved ball; `mod_so3.f90:2513-2635`).
3. **Shipped files** (uncompressed `np.savez`, no timestamps, byte-stable;
   md5s in `src/kikuchipy/data/_registry.py`, no URL; package
   `emsoft_hrosm/__init__.py` with a docstring). **Layout (amended
   2026-10-06, spec review: the drafted list put the DI file at ~280 kB,
   over its own 250 kB cap; this is the single layout, frozen as the V14
   key table, no conditional trimming):**
   `regression_hrosm_large_di.npz` (`TopMatchIndices[:, :10]` int32
   1-based, `KAM`, `OSM`, `OSM_05`, `CI`; ~231 kB),
   `regression_hrosm_large_refined.npz` (`RefinedEulerAngles`,
   `RefinedDotProducts`, `EulerAngles`; ~116 kB),
   `regression_hrosm_large_{center,center_dilate,wat}.npz` (`nGrains`,
   `grainID`, `npixels`, `grainROI`, `avor`, `kappa`, `kam`, `newOSM`,
   `newEuler`, `newCI`, the namelist text; `newQuat`, the float32 `eq_` of
   `newEuler`, is not shipped and is checked in V14 only; `maxGROD` is
   never written by the chosen binaries and never shipped; ~118 kB each),
   `regression_hrosm_ball_n6.npz` (2,197 points; `qu`, `ro`; `eu` checked
   by the bin arm only; ~141 kB).
   Budget: each < 250 kB, total recorded and pinned (seed ~841,000 B;
   above the EMSphInx `< 100 kB` precedent, recorded). Provenance in every
   file: `program_md5`
   (each exe AND DLL; the exes are 50 kB launchers), `emsoft_version`,
   `emsoft_commit` (from the DLL string), `master_md5` (the cached
   original), `master_run_md5` (the patched copy, D13.2.2), `patterns_md5`,
   `pc`, `namelist`, `gpu_name`, `numdictsingle`, `numexptsingle`,
   `kikuchipy_version`. `TestReferenceFiles` (quad equality of script
   `SCENARIOS`, registry keys, directory glob, frozen test table) is cloned
   from `test_spherical_emsphinx_regression.py:920-1062`, NOT its module
   docstring, which on `develop` names a `specs/` path (`:21-24`; fixed
   only on the staging branch).
4. **Binary choice (recorded; open in `plan.md` section 7).** Two Windows
   ifx builds: `C:/Users/westraadt.1/EMSOFT/EMsoftOOBuild/Release/Bin/`
   (DLL version `6_0_20260413_0`, commit `975a1fc`, no `newQuat`;
   `EMsoftOOLib.dll` md5 `da007b873434bbaaf0a25ecda49b2725`,
   `EMOpenCLLib.dll` `570c619807905d43b178e205cd373958`) and
   `C:/Users/westraadt.1/Software/EMSOFT/EMsoftOO/build-ifx-release/Bin/`
   (`6_0_20260525_0`, commit `c127868` = EMsoftOO develop HEAD, with
   `newQuat`, built 2026-05-25; `EMsoftConfig.json`'s
   `EMsoftLibraryLocation` points there). Recommended: the `c127868` build
   (it is the source the K-table cites); provenance records which ran.
5. **Historical files (local gate; read-only)** under
   `C:/Users/westraadt.1/Software/EMSOFT/EMsoftData`:

   | file | md5 | EMsoft | use |
   |---|---|---|---|
   | `DItutorial/Ni/dp-Ni6-refined_HROSM.h5` | `0d8a77ac950f8d0db9c9a76ab13a84b4` | 6_0_20260411_0 | clustering, dilate, boxes; `kam`; WAT `avor`/`kappa` bands |
   | `DItutorial/Ni/dp-Ni6-refined.h5` | `dbfa6da1e5e9c08be6864bb1b9b8820b` | 6_0_20260411_0 | `RefinedEulerAngles` of that run; DI `KAM`, `OSM` (nosm 20) |
   | `OSM/GRX810_HROSM/dp-GRX-refined.h5` | `48bd1600f4a417fe4978b8ca4fd5f5f5` | 6_0_20250527_0 | DI `KAM`, `OSM`, `OSM_20` (337 x 413) |
   | `Al_HROSM/dp-full.h5` | `cdb94c405f9dfa8079c9828769f24075` | 6_0_20260813_0 | DI `KAM`, `OSM` (689 x 728; weekly-local) |

   **A8 decision: the Ni6 HROSM file IS a local clustering oracle.** Its
   run (2026-04-11; `gangle 10.0`, `misorang 5`, `nsamples 20`, `nosm 10`,
   `dilate .TRUE.`, `orav 'averageWAT'`, `numEM 25`, `numIter 40`; 151 x
   186 (H x W) map; 62 grains, kappa 8.19-34,896, none rejected, grains 35 and 62
   under 10 pixels) postdates `0b8e885`, and its `grainID`, `npixels`,
   `grainROI` were reproduced bitwise (D4.1). Its `dpfile`
   `DItutorial/Ni/dp-Ni6.h5` was overwritten on 2026-05-23 (EMDI
   6_0_20260523_0); the input survives as `dp-Ni6-refined.h5` (EMDI
   2026-04-11 20:14, EMFitOrientation `PCcorrection 'On'`), evidenced by
   the KAM match (D3.6). Not an end-to-end `newOSM` oracle in CI: the
   patterns `EDAX-Ni.h5` (md5 `c03a565488f4ca1218bbb4ff2f491a2f`) are 103
   MB and `MasterPatterns/Ni-master-20kV.h5` was regenerated on 2026-05-23,
   after the run; a Ni6 end-to-end arm is not built (`plan.md` 7.2 item 11
   default; Johan may override at approval; amended 2026-10-06, spec
   review).
   **GRX810's HROSM file is NOT an oracle** (`dp-GRX-refined_HROSM.h5`, md5
   `afc63b4d7a2dedd0dd95930d3575f9d2`, 6_0_20250528_0): it predates
   `0b8e885`, and its `kam` is not reproducible from the local dp copy
   (1,494 of 139,022 non-degenerate pixels differ, up to 360 deg); its
   `dpfile` `GRX/20250131_GRX810_ODS_Crept/dp-GRX-refined.h5` is gone.
6. **Regenerate-and-diff (bin, V14)**: the script reproduces the shipped
   files; CPU-only arrays (`KAM`, `OSM*`, `grainID`, `kam`, `npixels`,
   `grainROI`, EMsampleRFZ lists) bitwise; GPU-derived arrays
   (`TopMatchIndices`, Euler angles, `newOSM`, `newEuler`, `newCI`, `wat`
   `avor`/`kappa`) measured at the first regeneration and pinned (bitwise
   or a band); a mismatch names `program_md5` per exe and DLL. Run time
   ~10-15 min (MTP).

Pins: V2, V7, V10, V12-V14, `TestReferenceFiles`, the local arms.

### D14 -- Branch, PR and fan-out policy (frozen, user decision 2026-10-06)

Supersedes the parked plan's decisions 2 and 6.

1. **Base**: `feat-HROSM` off fork `develop` @ `de27741a`; no specs
   re-adoption (the parked Step 0 with its `.gitignore`, pre-commit and
   `.git/info/exclude` edits is dropped). Update rule: merge `develop` into
   `feat-HROSM` if needed; never merge `feat-spherical-indexing*` or
   `hrebsd-dic` into `develop`.
2. **One fork PR** `feat-HROSM -> develop` (jwestraadt/kikuchipy) after
   Stage C with the PR template; expected #20 (`gh pr list`, 2026-10-06:
   #19 newest, merged), confirmed at PR time; CHANGELOG link `` (`#20
   <https://github.com/jwestraadt/kikuchipy/pull/20>`_) ``, rewritten if
   the number differs. Signed commits (`git commit -s`) with the trailer
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, explicit
   pathspecs, `SKIP=licenseheaders uvx pre-commit run --files <explicit
   non-specs files>`; pushed per stage; a failing-tests commit never alone.
3. **Merge only on Johan's go**, merge commit, ubuntu and windows CI green;
   macOS red on the pre-existing `test_ni_proper_oh_count` (23 == 22) is
   not a blocker.
4. **Fan-out after the merge (sha M)**; worked example: the NLPAR fan-out
   (`specs/2026-10-04-nlpar/plan.md` section 1, roadmap "Fan-out").
   (a) **`hrebsd-dic`**: `git merge --no-ff develop` (never the reverse);
       append-type conflicts resolved HREBSD first, then HROSM. Expected
       (from `git diff develop hrebsd-dic`, 2026-10-06):
       `indexing/__init__.pyi` (imports at `:16`, `__all__` at `:55`,
       where `segment_grains` and `segment_grains_kam` are adjacent);
       `ebsd.py` import block (hrebsd's `_hrebsd` imports at `:59-64` vs
       HROSM's after `_hough_indexing`); `CHANGELOG.rst` Unreleased
       (hrebsd +55 lines at `:18`); `run_nbval.sh` (`hrebsd_dic`,
       `hrebsd_si_indent`, `hrosm` all between `hough_indexing` and
       `hybrid_indexing`; hrebsd also rewrote the header);
       `tutorials_sanitize.cfg` (EOF, only if HROSM adds sections);
       `bibliography.bib` (if HROSM adds a key: one alphabetical
       resolution in the `chen2015dictionary`..`foden2019indexing` gap).
       None expected in `conftest.py`, `pyproject.toml`, `_registry.py`,
       `_orientation_similarity_map.py`, `_dictionary_indexing.py`,
       `doc/tutorials/index.rst` (hrebsd adds a section after Indexing;
       HROSM inserts inside it), the `ebsd.py` methods (D1.4) or the three
       constitution files (`hrebsd-dic`'s hunks sit at `mission.md:52`,
       `roadmap.md:1-4, 125`, `tech-stack.md:83`, HROSM appends at EOF;
       amended 2026-10-06, spec review). Gates (as `plan.md` 1.2): no
       conflict markers; `pytest $B_TESTS -n 0`, then `-n 4`; `pytest tests
       -k hrebsd -n 4`; the full suite `-n 4`; nbval on `hrosm.ipynb`;
       pre-commit on the resolved files; push; never merged into
       `develop`.
   (b) **Clean replay** onto a NEW branch `feat-spherical-indexing-hrosm`
       STACKED on `feat-spherical-indexing-nlpar` (`e49b3d85`) with
       `C:\Users\westraadt.1\Repos\_staging\{pick.ps1,gate.ps1}`
       (`cherry-pick -m 1 -n` of M, `specs/` stripped, two commits "Add
       high angular resolution orientation similarity maps (HROSM)" and
       "Add HROSM tutorial" with `Staged-from: jwestraadt/kikuchipy#20 (M)`
       trailers), the equivalence gate on the `+`/`-` lines of `git diff
       M^1 M -- . ':!specs'`, worktree tests with `PYTHONPATH=<worktree>\
       src` and a `kikuchipy.__file__` guard; pushed, no PR;
       `feat-spherical-indexing` (`6723aaf0`) and
       `feat-spherical-indexing-nlpar` untouched. The fork-only `tests.yml`
       timeout (20 min on `develop`, 15 on the staging branches) is
       excluded. Divergence `develop` vs `feat-spherical-indexing-nlpar` in
       HROSM's files (merge base `4ed31813`; 54 non-specs files differ
       overall): `ebsd.py` (two docstring hunks, IQ doctest `:1925-1931`
       and Hough citations `:2235-2346`; no overlapping or adjacent hunk
       with the HROSM insertion at `:2558`, no conflict expected);
       `CHANGELOG.rst` (a 0.13.1 section at `:105`; HROSM bullets go into
       Unreleased at `:20`, no overlapping or adjacent hunk);
       `bibliography.bib` (inserts at `:251` and EOF; an HROSM key goes
       alphabetically after `chen2015dictionary`, `:54`, so no conflict;
       amended 2026-10-06, spec review); `pyproject.toml` (ebsdsim floors
       `:72, :83`; the new `--ignore-glob` near `:181` is clear);
       `.gitignore` (the staging branch ignores `*.npz` except under
       `src/kikuchipy/data/**`, so the shipped references are kept);
       `_ebsd_detector.py` (docstring only). No divergence in
       `indexing/__init__.pyi`, `_orientation_similarity_map.py`,
       `_dictionary_indexing.py`, `ebsd_master_pattern.py`, `_registry.py`,
       `conftest.py`, `doc/tutorials/{index.rst,run_nbval.sh,
       tutorials_sanitize.cfg}`. Because of the stacking, an upstream HROSM
       PR cut from that branch would carry the NLPAR commits unless NLPAR
       goes upstream first (recorded).
5. **Clean-replay rule (A4)**, enforced at every stage gate and before the
   replay: nothing under `src/`, `tests/`, `doc/`, `examples/`,
   `benchmarks/`, nor the root `conftest.py`, `CHANGELOG.rst`,
   `pyproject.toml`, names a `specs/` path, a spec file name
   (`requirements.md`, `plan.md`, `validation.md`, `tech-stack.md`,
   `mission.md`, `roadmap.md`) or a spec ID (D-, V-, M-, K-, R-numbers);
   comments, docstrings and test names state the fact ("EMsoft credits the
   vertical neighbour to iii-W+1",
   `test_compat_vertical_pair_is_credited_one_column_right`), never "K1".
   Gate: `git diff develop...HEAD -- src tests doc examples benchmarks
   conftest.py CHANGELOG.rst pyproject.toml ':!*.ipynb' | grep -E "^\+" |
   grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|
   tech-stack\.md|mission\.md|roadmap\.md|
   [^A-Za-z0-9_][DVKMR][0-9]+([^0-9]|$)"` (one pattern, wrapped here at
   the `|`) prints nothing, and the same pattern over the cell sources
   of `doc/tutorials/hrosm.ipynb` (read with `nbformat`; stored base64
   outputs give false hits; amended 2026-10-06, spec review) finds
   nothing (`git grep ... develop..` fails: it takes a tree); a Fortran
   double literal such as `180.D0` is rewritten as `180/pi`.
6. **Standing constraints**: `stash@{0}` never touched;
   `specs/2026-08-16-constitution/upstream-issue.md` never edited;
   `doc/tutorials/{hybrid_indexing,spherical_indexing,load_save_data,
   pattern_matching,nlpar}.ipynb` never in a diff; the untracked
   `specs/_research/plan-upstream-merge-0.13.1.md` and
   `AGH__Si_indent_1_512x672.h5oina` never staged; the main `.venv`
   CuPy-free; EMsoft programs never concurrent (lock), run directories only
   under `<EMdatapathname>/kikuchipy_hrosm/`; the EMsoftOO checkout stays on
   `develop`, read-only (Johan's branch read with `git -C ... show`).

Pins: the definition of done in `validation.md` (clean-replay grep, `git
log origin/feat-HROSM..feat-HROSM` empty, CI state recorded).

### D15 -- CI budget, test placement and oldest matrix (frozen policy; values MTP)

1. **Budget (A5; amended 2026-10-06, spec review, with the measured CI
   times)**: the fork's `tests.yml` (`timeout-minutes: 20`, `:36`,
   fork-only; 15 upstream and on the staging branches) runs the default
   suite with `--cov-branch --reruns 2 -n 4`. On `develop` de27741a (run
   37508408494) the jobs already take 17 min 21 s (ubuntu py3.10
   oldest), 16 min 55 s (windows py3.13) and 14 min 48 s (ubuntu py3.13);
   NLPAR's trimmed default suite costs ~43 worker-s and 22.0-25.8 s
   CI-style wall here (NLPAR ledger entry 20). HROSM's default-suite
   additions are gated by (a) the CI-style wall time (`uv run --no-sync
   --with pytest-cov pytest <the HROSM test selection> -n 4 -q -p
   no:cacheprovider --cov=kikuchipy --cov-branch --cov-report=`, this
   machine) <= 25 s (MTP; the NLPAR-measured equivalent) and (b) the A5
   ceiling of 60 s serial (`-n 0`, warm caches), both measured at every
   stage gate and pinned in the ledger; the summed worker-seconds are
   recorded. The on-push CI run of each stage push (the Stage B push at
   the latest, before Stage C; amended 2026-10-06, spec review round 2:
   the PR opens only after Stage C) is checked against 18 min on every
   job; a job above 18 min applies the trim order of `validation.md` "CI
   budget" at once. **Weekly arms run locally (amended 2026-10-06, spec
   review round 2).** The fork's Weekly workflow (`weekly.yml`, id
   282579366) is `disabled_inactivity`; its last scheduled runs
   (2026-07-27, 08-03, 08-10) failed in the notebook job while the
   `weekly-tests` job passed (3 min 59 s on 2026-08-10). So no CI runner
   executes the `@pytest.mark.weekly` arms or nbval of `hrosm.ipynb`; they
   run in the local `--weekly` stage gate (`plan.md` section 5) and the
   local nbval gates (Stage C, fan-out). Budget for the weekly additions:
   their local serial seconds (`--durations=0`) recorded per stage, with
   a scaled CI estimate (local seconds x the ratio of the last on-push
   ubuntu py3.13 job time to the local full-suite `-n 4` time), a record,
   never a gate; the 5 min line of the earlier draft is kept as the
   estimate's target. Re-enabling the workflow is Johan's choice
   (`plan.md` 7.4 item 26).
2. **Default suite**: Stage A unit tests on synthetic maps and the shipped
   references (KAM, segmentation, dilate/centre, ball N <= 6, averaging at
   small N, OSM both modes, reader), reference integrity, Stage B contracts
   on small synthetic maps (e.g. 8 x 10 maps, 32 x 32 px, ball N 2-4); at
   least one named killer per mutant row.
3. **Weekly**: N = 20 balls with real dictionary indexing (V15 full
   size), the V13 one-grain arm (moved out of the default suite, amended
   2026-10-06, spec review), parametrised grids beyond the representative
   subset (all run in the local `--weekly` gate, D15.1). **local +
   weekly** (run on this machine with `--weekly` and
   `KIKUCHIPY_EMSOFT_DATA`, which marks this machine; any runner without
   the variable skips them): the Al
   arms and the V13 full map (minutes even here; it reads no EMsoft data
   file). **Local ledger runs, never tests**: the performance baselines
   (full map at N 20, `n_steps=10`, compat box overhead) and `si_wafer`.
   **bin**: EMsampleRFZ N 20, V14. **local**: Ni6, GRX810, Al (Al also
   weekly).
4. **Oldest matrix (A6)**, one recorded local run per stage, the full CI pin
   set (`tests.yml:48`): `uv run --isolated --python 3.10 --extra tests
   --with "dask==2021.8.1" --with "diffsims==0.5.2" --with "hyperspy==2.2"
   --with "matplotlib==3.6" --with "numba==0.57" --with "numpy==1.23.0"
   --with "orix==0.12.1" --with "pooch==1.3.0" --with
   "pyebsdindex==0.3.9.2" --with "scikit-image==0.21.0" pytest
   tests/test_indexing tests/test_signals -k hrosm -n 0 -q -p
   no:cacheprovider`. Verified 2026-10-06 in an isolated py3.10 env (orix
   0.12.1, numpy 1.23.0, numba 0.57, scipy 1.13.1 resolved):
   `Rotation.random_vonmises(shape, alpha, reference)`,
   `Rotation.from_homochoric`, `orix.quaternion._conversions.cu2ho` and
   `cu2ho_single`, `Symmetry.proper_subgroup` (Oh -> 24),
   `Orientation.angle_with(other, degrees)`, Hamilton `Rotation.__mul__`
   (i * j = k), `Orientation.dot_outer`, `Rotation.mean`,
   `Rotation.angle_with_outer` (D20.5 tests), the `orix.io` save/load
   round trip of bool and 2-D props (D2.3),
   `scipy.sparse.csgraph.connected_components`, `scipy.ndimage.
   {maximum_filter,grey_dilation}`, `scipy.special.ive`,
   `scipy.spatial.cKDTree` (D20.1; the last three items added and
   re-verified 2026-10-06, spec review). No fallback
   needed; the CI oldest job pins no scipy and every scipy API used
   predates the `>= 1.7` floor. Tests avoid `Orientation.reduce()`
   (`tech-stack.md:17`).
5. **Numba-cache flake rule (A10)**: the root `conftest.py` autouse fixture
   `_keep_numba_cache_dir_per_worker` (`:131-150`) exists; gates run `-n
   0`, then `-n 4`, and re-run red tests alone before they count.

### D16 -- Documentation (frozen)

1. **Tutorial** `doc/tutorials/hrosm.ipynb`: `nickel_ebsd_large`
   (download) indexed by the dictionary route of `pattern_matching.ipynb`
   plus NCC refinement, then `EBSD.hrosm()`; global OSM next to the HROSM
   OSM; KAM (both modes), grain map, GROD and the warning; the synthetic
   sub-grain demonstration (a 0.5 deg sub-boundary invisible to a 1.4 deg
   global dictionary, visible in HROSM); parameter guidance (`max_angle`,
   `n_steps`, cost table); "Differences from EMsoftOO's EMHROSM" in words;
   links to `pattern_matching.ipynb` and `spherical_indexing.ipynb` (never
   edited). Rules (`tech-stack.md:52`): hidden first cell, thumbnail tag,
   black at 77, `dask.config.set(num_workers=8)` before verbose cells,
   stored outputs (over ~2 min on Read the Docs, MTP), no bare reprs with
   memory addresses.
2. **Registration**: `doc/tutorials/index.rst` Indexing gallery after
   `pattern_matching` (`:44`); `run_nbval.sh` `NOTEBOOKS` between
   `"hough_indexing.ipynb"` and `"hybrid_indexing.ipynb"` (`:9-10`);
   `tutorials_sanitize.cfg` regexes only as nbval needs (9 exist).
3. **Gallery example** `examples/indexing/hrosm.py` + new
   `examples/indexing/README.rst` ("Indexing") (amended 2026-10-06:
   `examples/pattern_matching/` does not exist on `develop`; sections are
   directories with a README under `examples_dirs="../examples"`,
   `doc/conf.py:373-375`): a synthetic two-grain map simulated from the
   shipped `nickel_ebsd_master_pattern_small`, `max_angle=2`, `n_steps=4`
   (729 orientations), runtime <= 30 s (MTP).
4. **Docstrings**: numpydoc; `EBSD.hrosm` cites
   `marquardt2017quantitative` (OSM), `chen2015dictionary` (DI, VMF
   mixture), `singh2016orientation` (cubochoric sampling) with `:cite:`
   (`bibliography.bib:190, 43, 280`). No EMHROSM paper exists (EMsoftOO has
   only `NamelistTemplates/EMHROSM.template` and `mod_HROSM.f90:428-441`):
   the Notes cite "EMsoftOO, program EMHROSM (M. De Graef, 2025),
   https://github.com/EMsoft-org/EMsoftOO". Chen et al., IEEE Signal
   Process. Lett. 22 (2015) 1152-1155 (the `mod_dirstats.f90` header) is
   cited only if added to the bibliography (`plan.md` 7.2 item 12; DOI
   verified first; the entry goes alphabetically after
   `chen2015dictionary`, `bibliography.bib:54`, not at EOF, so it does not
   conflict on the replay and needs one alphabetical resolution on the
   `hrebsd-dic` merge; amended 2026-10-06, spec review).
5. **CHANGELOG** `Unreleased -> Added`: one API bullet (method, the eight
   public functions/class, the two keywords, EMsoftOO acknowledgement) and
   one tutorial bullet, each ending with the fork PR link (eight since
   D20, 2026-10-06).
6. **API reference**: generated from `__all__` and the `EBSD` class.

### D17 -- Measurement policy (recorded)

1. **MTP**: every tolerance measured at the owning gate, pinned with
   `pytest.approx(measured, rel=0.05)` or the ~2x margin
   (`tech-stack.md:51`); bitwise pins say so. Real-data effects are pinned
   only as measured improvements, else "not worse"; the ledger says which.
2. **Seeds are seeds**: the parked plan's numbers (2026-09-28) and the
   drafting probe (2026-10-06) calibrate; none becomes a pin unmeasured.
3. **Seeds carried**: V13 OSM Pearson r >= 0.8, per-pixel disorientation
   median <= 0.3 deg and p99 <= 1.0 deg; V15 0.5 deg step recovered with
   median error <= 0.15 deg; V8 band ~ `2 / sqrt(kappa N)` scaled; the
   probe numbers of D3.6, D4.1, D6.1, D7.3, D13.5.
4. **Ledger**: `validation.md` "Recorded results", append-only, numbered,
   dated, recipe and machine; amendments dated in place here; an escalation
   to Fable (D18.3) is logged with its reason.

### D18 -- Process and model assignment (frozen, user rule 2026-10-05)

Replaces the parked "Process" section.

1. Planning and spec: Opus 5.5, effort xhigh, ultracode.
2. Tests, implementation, adversarial review, bug injection and fixes:
   Workflow agents `{model: 'opus', effort: 'medium'}`, at most 10 agents
   per workflow; commits and pushes by the main loop only.
3. Fable only to escalate ONE step after repeated errors AND an
   inconsistency (a gate fails again after a fix round, reviewers
   contradict, spec and code still disagree after a fix); the reason is
   recorded; back to Opus afterwards.
4. Lean (Johan: the NLPAR build "took forever"): three stages; per stage
   tests -> implementation -> review + bug injection -> fixes -> gates ->
   commit; no gold-plating.
5. Gate: Johan approves `plan.md` before any test or code (A9, no waiver).

### D19 -- The seven recorded defaults (recorded, pending Johan's approval)

| R | default adopted | where | alternative | consequence of switching |
|---|---|---|---|---|
| R1 | one `CrystalMap` with broadcast per-grain props + `GrainTable.from_crystal_map` | D2.2 | `tuple[CrystalMap, GrainTable]` | extra return value; fewer props |
| R2 | `EBSD.hrosm(average="mean")`; parity tests pass `"center"` | D5.1 | EMsoft's default `"center"` | default follows the centre pixel (outside the grain possible in compat) |
| R3 | `grain_id`: 0 unassigned, grains 1..n (EMsoft) | D2.3, D4.1 | hrebsd's 0-based labels with -1 | conversion for every EMsoft comparison; same convention as hrebsd's `grain_id` |
| R4 | one master; other phases' grains skipped with a warning | D8.14 | `dict[str, EBSDMasterPattern]` | multi-phase driver and tests in Stage B |
| R5 | K5 (right-side symmetry in the EM) is a defect, compat only | D5.5, D9 | treat it as a convention in both modes | correct VMF/Watson fails left-scrambled recovery |
| R6 | KAM-difference segmentation in both modes; misorientation rule later | D4.1 | misorientation rule in correct mode | correct grains diverge from EMsoft's; new oracle |
| R7 | `_dictionary_indexing(verbose=)`, `orientation_similarity_map(grain_id=, emsoft_compatible=)`, backwards compatible | D7.5, D8.7 | private copies, upstream code untouched | duplicated code; no public EMsoft-compatible OSM for DI maps |

### D20 -- Ball spacing and the pre-run GROD coverage check (frozen, user requirement 2026-10-06)

Johan (2026-10-06, after the first draft): "i need a function that
computes the mean angular spacing between the misorientation ball sampling
that is used for the run. Also the grod needs to be computed for each grain
from the grain average/central pixel. IF the misorientation ball is smaller
that this value a warning must be made before starting the run." Refines
D1.2, D1.3, D1.6, D5.8, D6 and D8.2; where they differ, D20 wins.

1. **Ball spacing, public** `misorientation_ball_spacing(max_angle=5.0,
   n_steps=20) -> float` in `_sampling.py`: the mean, over all `(2N + 1)^3`
   orientations of the ball, of the rotation angle (degrees, float64) to
   the nearest OTHER orientation of the same ball.
   - Computed once on the raw float64 grid (`misorientation_ball(None,
     max_angle=..., n_steps=...)`). The angle between two ball points is
     invariant under the per-grain composition `~R_v * avor_g` (the
     relative rotation is conjugated by `avor_g`), so the value is the same
     for every grain and both modes; the K8 float32 storage moves it by a
     negligible amount (MTP, recorded, never used).
   - Nearest neighbours by `scipy.spatial.cKDTree(q).query(q, k=2)` on the
     unit quaternions (column 1 = nearest other point). Every ball point has
     `q0 >= cos(w / 2) > 0` for `max_angle < 180`, so the chord `d` is
     monotone in the angle and `angle = 4 arcsin(d / 2)` (from `<qa, qb> =
     1 - d^2 / 2` and `angle = 2 arccos|<qa, qb>|`).
   - Validation as `misorientation_ball` (same fragments). Cost at the
     defaults (68,921 points): well under 1 s (MTP), so it runs inside the
     default suite and inside every `EBSD.hrosm` call.
   - Measured 2026-10-06 (main loop; grid of D6.1 through orix `cu2ho` and
     `Rotation.from_homochoric`, equal to the ported conversion at 1e-12 per
     D6.2): 5 deg / N 20: mean 0.15918 deg (min 0.14649, max 0.24997, the
     centre point whose nearest neighbour is shell 1); 5 deg / N 10:
     0.31765; 10 deg / N 20: 0.31812; 2 deg / N 20: 0.06368; 5 deg / N 2
     (125 points): 1.60130, equal to the all-pairs brute force
     (`angle_with_outer`) to 5e-13. So the mean spacing is about 0.64 x
     `max_angle / n_steps`, smaller than the radial shell step `max_angle /
     n_steps` (0.25 deg at the defaults) because neighbours within a shell
     are closer than neighbours across shells. The docstring states both
     numbers. Pinned values: re-measured through the ported conversion at
     Stage A (expected identical to these digits).
   - `misorientation_ball` and `EBSD.hrosm` name it under See Also; the
     `verbose >= 1` info message of `EBSD.hrosm` prints the ball size
     `(2N + 1)^3`, the radius `max_angle` and this spacing before the grain
     loop.
2. **GROD reference** (refines D5.8): for each pixel of a valid grain, GROD
   is the disorientation angle (minimum over the phase's proper point
   group operators, applied on the left as everywhere in this spec; the
   pair angle of D3.1 with its near-one snap, through `_dot_to_angle`;
   degrees, float32) between the pixel orientation and the grain's
   reference orientation, which is exactly the orientation the ball is
   centred on: the grain average for `"mean"`, `"vmf"` and `"watson"`, the
   centre pixel for `"center"` (correct mode: the grain pixel nearest the
   centroid, so its own GROD is exactly 0.0 by the snap, also when the
   pixel is stored as another symmetry variant; amended 2026-10-06, spec
   review round 2; compat K4: the bounding-box centre pixel,
   possibly outside the grain). `max_grod` is the maximum over the grain's
   present pixels. Computed for EVERY valid grain in both modes, in the
   orientation stage, including grains later skipped by `min_pixels` or the
   phase rule (their GROD is still reported).
   - Public `grain_reference_orientation_deviation_map(xmap, grain_id,
     grains) -> np.ndarray` in `_averaging.py`: `(ny, nx)` float32 degrees,
     NaN at label 0, invalid grains and absent points; `grains` is the
     `GrainTable` of `average_grain_orientations` (its `rotation` per grain
     is the reference). One implementation: the driver and
     `average_grain_orientations` (for `max_grod`) call the same private
     helper, `_grod_map(xmap, grain_id, rotation, valid, operators_by_phase)
     -> np.ndarray` ((ny, nx) float32 degrees, in `_averaging.py`; named
     2026-10-06, spec review), so the map, `GrainTable.max_grod` and the
     output props `grod` / `grain_max_grod` agree bitwise. In compat
     `"center"` the reference is the K4 box-centre pixel's raw `eq_`
     quaternion (`GrainTable.rotation`), never the centroid.
   - Symmetry: pixels stored as different symmetry variants of the same
     orientation get the same GROD (the left-scrambling test of V8 applies).
3. **Pre-run coverage warning** (refines D1.6, D5.8, D8.2): after the
   orientation stage and BEFORE the ball is built, any dictionary is
   simulated or any pattern is matched, `EBSD.hrosm` issues ONE
   `UserWarning` if any grain that will be re-indexed (valid, `n_pixels >=
   min_pixels`, phase with the master) has `max_grod > max_angle`
   (strict; equality does not warn; compared in float64 as frozen in
   D5.8, round 2). The run then continues.
   - Message (tested fragments in quotes): "misorientation ball", the
     number of grains concerned, the radius `max_angle`, the mean spacing of
     D20.1, the largest max GROD, up to ten `(label, max GROD, n_pixels)`
     entries in descending max GROD, and the hint that pixels beyond the
     radius can only match orientations on the ball surface and that
     `max_angle >= <largest max GROD rounded up to 0.5 deg>` covers every
     grain (cost grows with `(2N + 1)^3`; keeping the spacing fixed means
     `n_steps` grows with `max_angle`).
   - Both modes (Johan's EMsoftOO `f6270d0` adds the same check; upstream
     EMsoft has none; a warning changes no number, so no K-switch).
   - **Two call sites, one helper (amended 2026-10-06, spec review).**
     The message comes from `_coverage_warning_message` (D5.8) and always
     carries both fragments "misorientation ball" and "max GROD".
     `average_grain_orientations` keeps a standalone warning for callers
     outside `EBSD.hrosm`: it counts every valid grain and has no spacing
     clause (the function has neither `n_steps` nor `min_pixels`). The
     driver calls the private averaging core, which never warns, applies
     the skip decisions, computes `misorientation_ball_spacing(max_angle,
     n_steps)` and warns once over the grains it will re-index, with the
     spacing; a run therefore warns exactly once. Entries are sorted by
     descending max GROD, ties by ascending label.
4. **Output**: no new props (`grod`, `grain_max_grod` exist, D2); the
   spacing is not stored in the `CrystalMap` (no scalar metadata there).
5. **Pins and mutants** (to be placed in validation.md and plan.md section
   6): spacing equals a brute-force mean nearest-neighbour angle (all
   pairs, orix `Rotation.angle_with_outer` or NumPy) on small balls (N = 1,
   2, 3) at 1e-12; equals the brute force on `misorientation_ball(center)`
   for a random centre at 1e-9 (invariance); default value pinned (MTP);
   decreases with `n_steps` and grows with `max_angle`. GROD: centre pixel
   exactly 0.0 in correct `"center"` mode (by the D3.1 snap; the arm uses
   a centre rotation whose self-dot rounds below 1, round 2); equals a
   per-pixel orix `Orientation.
   angle_with` loop on a synthetic grain; variant-scrambled pixels give the
   same GROD; map, `max_grod` and props agree. Warning: fires once, before
   the first `get_patterns` call (spy records the order); lists the
   offending grains; silent at `max_grod == max_angle`; skipped grains do
   not trigger it. Mutants: `2 arcsin(d / 2)` instead of `4 arcsin(d / 2)`;
   `k=1` (self distance); GROD without symmetry reduction; GROD to the
   centroid instead of the ball centre in compat `"center"`; warning issued
   after the first grain; `>=` instead of `>`. Placed 2026-10-06 (spec
   review): `validation.md` V7 `TestBallSpacing`, V8 `TestGROD`, V16
   `TestMessages`; mutants S9-S14 of `plan.md` section 6.

## Context

### EMHROSM algorithm (EMsoftOO develop @ `c127868`, re-verified 2026-10-06)

`mod_HROSM.f90` (868 lines; outline `:428-441`):

1. **Read the DI file** (`:463-465`, `getRefinedEulerAngles=.TRUE.`);
   `nosm` overridden from the HROSM namelist (`:468`); `ROIoffset =
   ROI(1:2)` (`:470`). Only `RefinedEulerAngles` (float32 radians) are
   used: top-1, CI and masks are ignored; missing refined angles leave the
   array unallocated.
2. **Monte Carlo, master, crystal** (`:472-499`); detector from the DI
   namelist, binned size `numsx = exptnumsx / binning` (`:501-524`).
3. **`Cluster_T`** (`:528`, `mod_cluster.f90`): map size (`:134-141`), KAM
   in degrees (`:152-153`), growth (`:164, 445-562`), `-1 -> 0`
   (`:167-169`), boxes (`:203-204, 382-398`), optional dilate and boxes
   again (`:209-212`), `npixels` (`:215-218`), averaging (`:227-315`).
4. **Per grain** (`:569-655`, after one ball count at `:560-566`): skip
   `kappa == -1` (`:570`); ball `sample_isoCubeFilled(misorang, nsamples)`
   + `SampleIsoMisorientation(qr(avor_i))` (`:579-587`); skip `npixels <
   10` (`:592`); in RAM if the block is below `maxRAMmem` GB (`:605-612`);
   `dinl = savedinl`, ROI = box + offset (`:615-617`); `OSMDIdriver`
   (`:619-620`); copy back `mainOSM`, `mainEuler = real(re(rodarray))`,
   `mainResult` for `grainID == i` (`:626-636`).
5. **Output** `EMData/HROSM` (`:719-760`): `nGrains`, `grainID`,
   `npixels`, `grainROI`, `avor`, `kappa`, `kam`, `newOSM`, `newEuler`
   (radians, not FZ-reduced), `newQuat` (float32 `eq_` of `newEuler`),
   `newCI`; then a 0..255 TIFF of `mainOSM` (`:769-772`).
6. **Defaults** (`:166-180`, `NamelistTemplates/EMHROSM.template`):
   `nsamples 20`, `nosm 10`, `gangle 5.0`, `misorang 5.0`, `maxRAMmem
   1.0`, `dilate .FALSE.`, `orav 'center'`, `numEM 25`, `numIter 40`.

`OSMDIdriver` (`mod_DI.f90:1757-2629`; an older commented copy at
`:2631-3471`), per `W x H` box: experimental patterns hi-pass filtered
(float64 FFT, `1 - exp(-w(i^2 + j^2))`, DC only scaled), rescaled to
0..255, `adhisteq(nregions)`, masked, L2-normalised without mean
subtraction; dictionary patterns simulated per ball orientation with the
energy-weighted Lambert interpolation (`CalcEBSDPatternSingleFull`),
optional gamma, NO hi-pass (commented out), 0..255, `adhisteq`, masked
before and after, L2-normalised; float32 inner products on the GPU
(`opencl/DictIndx.cl`; correct only for `Ne == Nd`, multiples of 16); top
`nnk` merged per batch (`SSORT`, not stable); `CI` = best dot product;
`rodarray` = `FZarray` of the best match (`(0, 0, 1, 0)` for index 0); OSM
over the box (`:2578-2579`). One PC for the map (`92c1b21`; the comment at
`mod_HROSM.f90:461-462` is stale).

### EMsoft quirk catalogue (mapped to the K-switches)

| # | defect or behaviour | file:line | handling |
|---|---|---|---|
| 1 | KAM vertical pair credited to `iii - W + 1` | `mod_DIsupport.f90:432` | K1 |
| 2 | KAM `jj = iii/W + 1` one too large in the last column | `:409` | K1 |
| 3 | KAM pixel (W, 1) compared with Euler (0, 0, 0) | `:396, 411, 427-433` | K1 |
| 4 | KAM divisor multipliers tuned to the buggy counts | `:439-455` | K1 |
| 5 | `acos` unclipped; NaN never selected; `ac` starts at 1000 | `mod_so3.f90:4163, 4199-4210` | K2 |
| 6 | growth on chained KAM differences, 8-connected | `mod_cluster.f90:512-513, 547` | rule, both modes (R6) |
| 7 | seed gate `kam <= gangle`; singletons rejected | `:471-475, 526-529` | rule, both modes |
| 8 | stack limit 1e6 truncates a grain | `:42, 548-556` | not reproduced |
| 9 | `nGrains = -1` corner case | `:471-481` | not reproduced |
| 10 | dilate skips first row/column, overwrites labels | `:431-437` | K3 |
| 11 | `center` = box centre, may lie outside the grain | `:232-234` | K4 |
| 12 | EM E/M steps on the right (`Mu * S_j`) | `mod_dirstats.f90:1002, 1051, 1068` | K5 |
| 13 | `getQandL` on the left; VMF `C = exp(logCp)` | `:1156, 1162` | K6 |
| 14 | stale/uninitialised `Q`, `L` when `Phi` underflows | `:836, 1166-1173` | K6 (initial 0.0) |
| 15 | final FZ representative on the right (`7039173`, 2025-07-08) | `:949-961` | K7 |
| 16 | `kappa > 5` gate, else identity and `kappa = -1` | `mod_cluster.f90:299-306` | `min_kappa`, both modes |
| 17 | clock seed (0-999 ms; 0 gives NaN); LGPL RNG | `:266-267`; `mod_math.f90:1868` | not ported |
| 18 | ball stored as float32 Rodrigues | `mod_DI.f90:1902, 2116` | K8 |
| 19 | OSM same bookkeeping, float32, not divided by `nosm` | `mod_DIsupport.f90:221-264` | K9 |
| 20 | OSM `x * 4.0/3.0` evaluation order build dependent | `:251-264` | `(x * 4) / 3` (D7.3) |
| 21 | box-wide indexing and OSM incl. foreign pixels | `mod_HROSM.f90:615-636`; `mod_DI.f90:2578` | K10 |
| 22 | dictionary thinning `min(ppend, W*H)` | `mod_DI.f90:2479` | not reproduced (D8.10) |
| 23 | ROI offset +1 with a DI ROI | `mod_HROSM.f90:470, 616` | not applicable |
| 24 | TIFF sized from `dinl`, map from `cluster` | `:769-772` | not applicable |
| 25 | `dinl = savedinl` reverts binning | `:502-507, 552, 615` | not applicable |
| 26 | unprocessed pixels `newEuler`, `newOSM`, `newCI` = 0 | `:546-551` | K12 |
| 27 | one PC; per-pixel PC removed | `mod_DI.f90`; `92c1b21` | K11; `pc="grain"` deviates |
| 28 | `center.txt` `I5` overflow; `R_All` unused; O(n^2) driver | `mod_cluster.f90:239, 471`; `mod_dirstats.f90:853-854` | not ported |

The KAM/OSM bookkeeping is inherited from EMsoft 5
(`C:\Users\westraadt.1\Repos\EMsoft\Source\EMsoftHDFLib\commonmod.f90`: OSM
`:356-455` with "THIS NEEDS TO BE VERIFIED !!!!!" at `:430`; KAM `:604,
621`).

### orix <-> EMsoft conventions (verified 2026-10-06 unless stated)

- **Euler -> quaternion**: EMsoft `eq_` (double branch, `epsijk = +1`,
  `mod_global.f90:107-108`) equals orix `Rotation.from_euler(eu)` with
  `direction="lab2crystal"` component for component, both q0 >= 0 (e.g.
  (10, 20, 30) deg -> (0.92542, -0.17101, 0.03015, -0.33682)), but not
  bit for bit: orix's numba `eu2qu` differs from `eq_` by 1 float64 ulp in
  about 30 % of rows (measured 2026-10-06, spec review; Ni6 8,930 of
  28,086), so compat converts through float32 Euler angles (D3.2 input
  route).
- **Product**: EMsoft `quatmult` (`epsijk = +1`) and orix
  `Rotation.__mul__` are the Hamilton product (i * j = k).
- **Symmetry side**: crystal symmetry on the LEFT in both
  (`getDisorientation_` `S_j * Mu`, `mod_so3.f90:4191, 4194`; orix `S *
  o`). `point_group.proper_subgroup` is the rotational group `QSym_Init_`
  selects (identity first, no +-q duplicates); VALUES and ORDER differ
  (EMsoft's literal constants and order), hence EMsoft's table in compat.
- **Disorientation**: `min_{j,k} angle(S_j a (S_k b)^-1)` equals the
  single-sided `min_j angle(a b^-1 S_j)` (conjugation invariance).
- **Rodrigues FZ** of a proper group = Voronoi cell of the identity (D5.6).
- **Ball composition** `q = conj(q_v) * q0` (D6.3); orix
  `get_sample_local` does `center * rot`.
- **Constants**: `cPi = 3.14159265358979323846D0` (`mod_global.f90:151`)
  is `np.pi`; `rtod = 180.D0/cPi` (`:170`) = `57.29577951308232`.
- **Means**: orix `Quaternion.mean` is Markley's, not symmetry-aware (D5.2).

### kikuchipy on `develop` @ `de27741a`: what exists and the gaps

- **No orientation KAM, no grains**: `hrebsd_kam` (mrad, HREBSD props) and
  `segment_grains` exist only on `hrebsd-dic` (`_hrebsd/_kam.py:126`,
  `_hrebsd/_segmentation.py:105`); the local `indexing/_hrebsd/` directory
  holds only an ignored `__pycache__`.
- **OSM** (`_orientation_similarity_map.py:30-152`): correct (Marquardt et
  al. 2017), not grain-aware, per-pixel lambda, breaks on absent points
  and squeezed `keep_n == 1` props (D7.5).
- **Dictionary indexing** (`_dictionary_indexing.py:36-169`,
  `ebsd.py:2400-2557`, `_prepare_metric` `:4459-4498`): NCC =
  `da.einsum("ik,mk->im", ...)` of zero-mean normalised float32 patterns;
  unconditional output, int64/int32 index paths and the masked `np.empty`
  fill (D8.1, D8.5, D8.7, D8.11).
- **`get_patterns`** (`ebsd_master_pattern.py:99-331`): PC rule (D8.8; the
  check is `len(detector.pc) > 1`), at most 2-D rotations (`:203-208`),
  `get_chunking` with `chunk_shape`/`chunk_bytes` (`:213-223`), rescale
  only if `dtype_out` differs from the master's (`:226-233`).
- **Detector** (`_ebsd_detector.py`): `pc_average` `:465`,
  `navigation_shape` `:614`, `crop` `:986`, `deepcopy` `:1034`,
  `extrapolate_pc` `:1315`, `fit_pc` `:1427`, `pc_emsoft` `:1660`; no
  `__getitem__`. `nickel_ebsd_large`: 4,125 distinct PCs (std ~0.003),
  binning 8, `px_size` 1.0, sample tilt 70, tilt 0.
- **Precedents to clone**: `conftest.py:684-700` (`emsphinx_dir`),
  `:702-750` (lock), `:753-775` (`emsphinx_program`);
  `create_emsphinx_reference.py`; `TestReferenceFiles`;
  `_registry.py:25-56`; `pyproject.toml:161, 177-181`. `.gitignore` on
  `develop` ignores neither `*.h5oina` nor `*.npz`.

### Oracle inventory

Binaries and configuration: D13.1, D13.4; Release/Bin exe md5s
(2026-10-06): `EMDI.exe 7161cf6b1fc3b16512b8f4c3a2d68172`,
`EMFitOrientation.exe db698904ee3d4db3f67b1bc43191affc`, `EMHROSM.exe
9d47a0b2833fde3c47147d9ae98af9bc`, `EMgetOSM.exe
f3dfdba0c1e7edac5975de0e2adb1fbf`, `EMsampleRFZ.exe
c66e97eedd4b7b7ec08e2898573e2372` (the develop-HEAD build's are recorded by
the script). `EMsoftConfig.json`: `EMdatapathname =
C:/Users/westraadt.1/Software/EMSOFT/EMsoftData`, `EMtmppathname =
C:/Users/westraadt.1/.config/EMsoft/tmp`. Cached master
`%LOCALAPPDATA%/kikuchipy/kikuchipy/Cache/develop/data/
ebsd_master_pattern/ni_mc_mp_20kv.h5` (305,510,476 B); the in-package
`ni_mc_mp_20kv_uint8_gzip_opts9.h5` (uint8, one energy bin) is unusable by
EMDI; the 75.7 deg masters in `DItutorial/Ni/` have the wrong tilt for
`nickel_ebsd_large` (tilt from the MC file's `sig`,
`program_mods/mod_EBSD.f90:2371`). Historical files: D13.5. Johan's
EMsoftOO branch is not built here; V3 pins the correct KAM analytically.

### Licensing

D11. EMsoftOO `License.txt` and the module headers carry the same
three-clause text, holder "Marc De Graef Research Group/Carnegie Mellon
University" (2013-2026; `mod_dirstats.f90` 2014-2026; `EMHROSM.f90`
2016-2026); kikuchipy already carries an EMsoft BSD-3 block
(`signals/util/_master_pattern.py:20-57`).

### Constitution pointers

`specs/tech-stack.md`: float64 rule `:21` (EMSphInx-scoped; HROSM scoping
D10), numba `:44`, licence block `:45`, `py_func` `:49`, reference
conventions `:50`, MTP `:51`, notebooks `:52`, CHANGELOG `:53`, process
`:70-84`, NLPAR section `:86-101` (format of the HROSM section);
`specs/roadmap.md` NLPAR block `:126-166` (HROSM's Stage A/B/C and Fan-out
boxes go after it); `specs/mission.md` ends with NLPAR's paragraph (HROSM's
goes after it). Texts: `plan.md` section 0.

### Sources

EMsoftOO (`develop` `c127868`): `Source/DictionaryIndexing/EMHROSM.f90`,
`Source/EMOpenCLLib/program_mods/{mod_HROSM,mod_DI}.f90`,
`Source/EMsoftOOLib/{mod_cluster,mod_DIsupport,mod_dirstats,mod_so3,
mod_quaternions,mod_rotations,mod_Lambert,mod_symmetry,mod_global,
mod_math,mod_filters}.f90`, `Source/EMsoftOOLib/program_mods/{mod_DIfiles,
mod_OSM,mod_sampleRFZ,mod_EBSD}.f90`, `NamelistTemplates/{EMHROSM,
EMsampleRFZ,EMgetOSM,EMFitOrientation}.template`, `License.txt`; Johan's
`feature/emhrosm-grod-precheck` (`c85982a`, `f6270d0`); EMsoft 5
`commonmod.f90`. kikuchipy `develop` `de27741a`, `hrebsd-dic` `b64cc18f`,
`feat-spherical-indexing-nlpar` `e49b3d85`; orix 0.14.2 and 0.12.1.
Papers: `marquardt2017quantitative`, `chen2015dictionary`,
`singh2016orientation`; Chen et al., IEEE Signal Process. Lett. 22 (2015)
1152-1155. The parked plan `specs/_research/plan-hrosm-2026-09-28.md` and
its five exploration reports (session scratchpad, not shipped); the NLPAR
spec `specs/2026-10-04-nlpar/` (template); user decisions of 2026-10-05
(model rule) and 2026-10-06 (branch route).
