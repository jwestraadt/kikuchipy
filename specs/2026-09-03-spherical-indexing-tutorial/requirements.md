# Phase 11 -- `spherical-indexing-tutorial`: requirements

Branch `spherical-indexing-tutorial` (roadmap Phase 11). The phase
ships the user-facing tutorial for the whole merged spherical API:
`doc/tutorials/spherical_indexing.ipynb` with stored outputs, its
docs wiring (`index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg`),
a `.sht` section in `load_save_data.ipynb` (verified absent -- Phase
2 did not add one; `grep -i sht doc/tutorials/load_save_data.ipynb`
has zero matches), and the 0.14 CHANGELOG consolidation. The
pseudo-symmetry section is **omitted** until Phase 8 lands (roadmap
re-scope 2026-09-02).

Every "measured" number below was produced on 2026-09-03 by the
drafting probes (`p11_time.py`, `p11_time2.py`, `build_nb.py` and
the executed prototype `spherical_indexing_executed.ipynb`; all in
the session scratchpad, not committed; recipes and full outputs in
`validation.md` "Recorded results") on this machine (20 logical
cores, Windows, warm numba caches, dask pinned to 8 workers inside
the notebook). The complete draft notebook was **executed
end-to-end** with `jupyter nbconvert --to notebook --execute`
(measured 72-81 s kernel time) and then re-run under
`pytest --nbval` against its own stored outputs with the repo's
existing `tutorials_sanitize.cfg`: **23 of 24 code cells passed
as-is**; the one failure (a `dask.config.set` memory-address repr)
is fixed in the cell inventory below (D4).

**Revised 2026-09-03** after the adversarial drafting review (a
content critic and a conventions critic; both reports applied,
disputed numbers re-measured with `p11_revise_probe.py` -- see
`validation.md` "Re-measurements"). The largest changes: the
shipped `nickel_ebsd_large` crystal map is **Hough-indexed and
refined, not dictionary indexing** (D3/D5 -- the drafting spec and
prototype mislabeled it); the bandwidth sweep now reports an
accuracy proxy instead of cross-bandwidth scores (D4); the parity
paragraph says 0.34 deg and names its configuration (D6); the
stored-outputs decision carries a quantified rebuttal of the dev
guide's output-less recommendation (D7); several recorded numbers
were corrected (chunk line 275/15, not 375/11; nine CHANGELOG
entries, not eight; six NOTEBOOKS entries, not five).

## Scope

In scope (the roadmap Phase 11 box, item by item):

- **`doc/tutorials/spherical_indexing.ipynb`** -- ~50 cells (25
  markdown, 25 code, 7 figures + the IPF color-key inset), stored
  outputs, ~1 MB executed (in family with `hybrid_indexing.ipynb`
  1.16 MB and `pattern_matching.ipynb` 2.0 MB). Content frozen by
  the cell inventory in plan section 1 (drafted, executed and
  nbval-verified as the scratch prototype; the review deltas --
  color-key cell, sweep redesign, misorientation stats -- are
  frozen in the inventory and re-executed at implementation).
  Wall clock measured ~80 s end-to-end **on the 20-core drafting
  machine** -- the roadmap's <= 3 min budget on 8 threads is met
  there with 2.2x margin. The margin is a property of that
  machine, not of the notebook: ~42 s of the kernel time is two
  cells that scale with real cores, so a 2-4 vCPU runner
  plausibly lands at 3-5 min. The pre-agreed fallbacks (bw 53,
  `s.inav[:40, :30]`, `refine=False` for sweep cells) therefore
  stay **live options**, drawn on only in the sweep design itself
  (D4).
- **`doc/tutorials/index.rst`**: `spherical_indexing` added to the
  Indexing `nbgallery`, after `pattern_matching` (D10).
- **`doc/tutorials/run_nbval.sh`**: `"spherical_indexing.ipynb"` in
  the `NOTEBOOKS` array, after `pc_fit_plane.ipynb` (the array is
  alphabetical) (D7).
- **`doc/tutorials/tutorials_sanitize.cfg`**: two new rules,
  `regex8` (PyOpenCL True/False) and `regex9` (chunk counts) (D7).
- **`doc/tutorials/load_save_data.ipynb`**: one new "Supported file
  formats" subsection *EMSphInx spherical harmonics master pattern
  (.sht)* -- one markdown cell, one live `kp.load` cell on the
  in-package `.sht` -- plus its row in the format table (D8). NB the
  constitution forbids sweeping the user's uncommitted notebook
  edits into a commit; `git status` at drafting is clean, but the
  implementation re-checks and commits only the hunks this spec
  describes.
- **`CHANGELOG.rst`**: the **nine** spherical `Unreleased -> Added`
  entries (PRs #5/#8/#9/#10; counted in the committed file: three
  interop, one refinement, one indexing, one bandwidth helper,
  three `.sht`) consolidated into **three**, plus the tutorial
  entry -- **four bullets total**; exact replacement text frozen in
  D9.
- **Builds**: `uv run sphinx-build -b html doc doc/_build/html`
  exits 0; `-b linkcheck` (shared doctrees) reports every
  phase-introduced link ok in its `output.json` -- overall exit 0
  is **not** achievable today and is not the gate (D11, validation
  V6; the review's correction). The docs build does **not**
  execute this notebook -- D11.

Out of scope (confirmed):

- **A pseudo-symmetry section** (roadmap re-scope 2026-09-02: Phase
  8 is deferred until after Phase 11; the tutorial neither mentions
  nor links pseudo-symmetry handling).
- Sphinx-Gallery examples, `plot_power_spectrum` /
  `SphericalBackProjector.plot` conveniences, stereographic plots --
  the deferred Phase 9 visualisation half. The tutorial plots
  everything it needs with plain matplotlib (`power_spectrum()`
  array + `semilogy`, `to_master_pattern().data` + `imshow`,
  `unproject` outputs + `imshow`), which is also what keeps it
  independent of that deferral.
- Any `src/` change: no new API, no docstring edits, no data files.
  The parity numbers the tutorial cites are already carried by the
  Phase 10 D9 docstring `Notes` on `EBSD.spherical_indexing` /
  `SphericalIndexer` and by the shipped regression suite.
- Running dictionary indexing inside the notebook (a live DI run
  would blow the budget for no added content). NB the review
  established there is **no** shipped DI reference either: the
  `nickel_ebsd_large` crystal map ships from **Hough indexing with
  PyEBSDIndex followed by orientation and PC refinement** (upstream
  CHANGELOG 0.8.0, PRs pyxem#578/#584), so the tutorial validates
  against live Hough plus that shipped refined map (D3/D5) and
  cross-links `pattern_matching.ipynb` for DI itself.
- Running the EMSphInx binaries inside the notebook (the interop
  section writes the input files and states what `IndexEBSD` does
  with them; the binaries are not on CI/Binder/user machines).
- Bibliography changes: `lenthe2019spherical` and friends shipped
  in Phase 0 and render on the Bibliography page; the notebook
  links the DOI directly, the exemplars' convention (D1).

## Decisions

### D1 -- Section list and framing (frozen)

The notebook mirrors `hough_indexing.ipynb` (hidden first markdown
cell, `# Title`, intro with a Note admonition, imports cell with
`%matplotlib inline`, `## Section` flow, thumbnail-tagged figure,
`## What's next?`), minus pseudo-symmetry:

1. *(hidden)* the standard "part of the kikuchipy documentation"
   cell (`"nbsphinx": "hidden"` metadata).
2. `# Spherical indexing` -- what it is (a **port of EMSphInx's own
   `IndexEBSD` program** -- EMSphInx is itself a CPU program, so
   the intro must not imply "CPU re-implementation" is the novelty;
   [Lenthe et al. 2019 DOI link]), refined orientations without a
   dictionary; the transform prose frozen as: the correlation is
   *expanded in generalized spherical harmonics and its maximum
   located with fast Fourier transforms over the three Euler
   angles* (not "FFTs in generalized spherical harmonics"); a Note
   admonition stating the port's agreement with `IndexEBSD` is
   pinned by a shipped regression suite (forward link to the
   interop section). Citations are markdown DOI links, the
   exemplars' convention -- no `:cite:` in notebooks. Any
   cost-vs-dictionary-indexing claim is hedged or attributed to
   Lenthe 2019, never asserted as a measurement of this notebook.
3. *(load data)* `nickel_ebsd_large(allow_download=True)` + static
   and dynamic background removal (link to the pattern processing
   tutorial **and** to the Hough tutorial's pre-indexing maps
   section -- the maps live there and are linked, not duplicated;
   both links required).
4. `## Master pattern on the sphere` --
   `nickel_ebsd_master_pattern_small(projection="lambert",
   hemisphere="both")`, `get_spherical_harmonics(bandwidth=188)`
   (D2), the `(d - 1)/2` resolution rule stated in prose with the
   bandwidth-choice sentence frozen in D2 (never "the largest
   bandwidth below this limit with fast transforms" -- 193 is a
   fast bandwidth below 200, measured), a Note admonition on
   full-resolution masters via `kikuchipy.data.ebsd_master_pattern()`
   with the exact call form (D2), one sentence on where master
   patterns for other phases come from (the EMSphInx SHT database,
   via the relative `../user/related_projects.rst` link), the
   power spectrum (semilogy of `power_spectrum()`), and the
   original-vs-synthesized comparison
   (**`to_master_pattern(dim=401)`** so both panels are 401 px --
   like-for-like, measured 0.04 s; the default 377-px grid is not
   used for the figure). The simulation-energy prose says **20.1
   keV** (the `EkeV` in the shipped master's Monte Carlo metadata,
   measured -- `Ehistmin` 20.0, one 0.1 keV bin), matching the
   harmonics repr; never "20 keV" against a 20.1 repr.
5. `## Calibrate detector-sample geometry` -- the dataset's shipped
   calibrated detector, `det.pc = det.pc_average` (spherical
   indexing takes one PC per map, as EMSphInx does), link to the
   Hough tutorial's PC-estimation section; a prose paragraph on the
   `sample_tilt` binding (the harmonics carry the simulation's 70
   deg and the indexer checks it against the detector -- the Phase
   6 measured "mismatch indexes ~5 deg wrong at higher scores"
   guard, stated as user guidance without the internals).
6. `## Back-projection onto the sphere` -- public
   `SphericalBackProjector(det, bandwidth=68)`, `unproject` of one
   pattern with `return_image_quality=True`, a three-panel figure
   (pattern | north Legendre grid | `window_mask()[0]`), IQ printed
   at 3 decimals (D7), prose on the window and the never-seen
   south hemisphere.
7. `## Perform indexing` -- the `_ = dask.config.set(num_workers=8)`
   cell with the memory-model rationale (D4), then
   `s.spherical_indexing(harmonics, det, bandwidth=68)` at the
   defaults (`refine=True` -- stated as EMSphInx's own default),
   the `CrystalMap` repr, the **score-semantics sentence frozen in
   D4** (the scores are *not* bounded normalized
   cross-correlations -- the shipped docstring's own warning), the
   scores + IQ maps figure, the commented `orix.io.save` export
   cell (the Hough exemplar's). Timing prose is machine-scoped
   ("about 20 s on eight dask workers of a 20-core desktop CPU")
   or points at the printed `Indexing speed` line -- never a bare
   absolute wall clock (no exemplar states one).
8. `### Choosing the bandwidth` -- `fast_bandwidths(50, 130)`, the
   coarse sweep over {53, 68, 88} on `s.inav[:40, :30]` printing
   **the median misorientation of each coarse solution to the
   refined headline map** (via `xmap[:30, :40]` CrystalMap
   slicing) plus `patterns/s` (D4 -- the redesign: scores are
   comparable only within one bandwidth, so a cross-bandwidth
   score trend is a change of metric, not of accuracy; the
   drafting sweep's "higher bandwidths give higher scores" claim
   is dropped), prose: the coarse error is bounded by the Euler
   half-cell (1.7 / 1.3 / 1.0 deg at 53 / 68 / 88), the tail
   tightens with bandwidth at roughly cubic cost, refinement
   removes most of the grid error -- and the closing residual
   sentence frozen in D4 (PC calibration *plus* band limit, per
   the docstring; never "geometry rather than bandwidth").
9. `## Validate indexing results` -- phase list from
   `mp.phase.deepcopy()` (D5; the nm-lattice `ValueError`
   constraint stated explicitly, D5), Hough indexing with the
   Hough tutorial's indexer settings, one sentence + repr noting
   the single `not_indexed` Hough point (the exemplar comments on
   indexing coverage; ours must too), the IPF **color key** cell
   (`ckey` repr) then the three-way IPF-Z figure (spherical |
   Hough | shipped map (Hough + refinement)) with the color key
   as an inset axes (the `hough_indexing` cell-51
   `plot_ipf_color_key` shape) -- **the thumbnail cell** (D6) --
   and the two misorientation-angle maps with printed medians
   *and 99th percentiles* (D5; the `not_indexed` point masked;
   one prose sentence on the >2 deg outliers the `vmax=2` color
   scale saturates).
10. `## Refining orientations from another source` --
    `refine_orientation_spherical(xmap_hough, harmonics, det)`,
    the overlaid histogram (Hough vs refined-Hough misorientation
    to the spherical solution) and the printed median 0.042 deg:
    both engines converge to the same maxima.
11. `## Interoperability with EMSphInx` -- `tempfile.mkdtemp()`
    (the `load_save_data` precedent), `harmonics.save(... .sht)` +
    `kp.load` round trip, `write_emsphinx_patterns`,
    `EMSphInxNamelist.from_kwargs(...).write(...)` + the first 5
    namelist lines printed, the frozen parameter-map table and the
    frozen scoped parity paragraph (D6), the what-`IndexEBSD`-does
    sentence frozen as "*runs the same indexing on the same
    patterns and master with the original C++ program*" (never
    "reproduces the indexing of this tutorial" -- unexecuted, and
    contradicted by the 0.34 deg parity paragraph one sentence
    later), then the hidden cleanup cell (`import os` +
    `os.remove` loop + `os.rmdir` + a final `temp_dir.exists()`
    line whose stored `False` output nbval re-verifies;
    `{"nbsphinx": "hidden"}` metadata and the
    `# Remove files written to disk in this tutorial` comment --
    the `load_save_data` cell-109 / `pattern_matching` cell-95
    precedent, both hidden).
12. `## What's next?` -- Hough (faster, no master, less precise),
    dictionary indexing (noisiest patterns, multi-phase
    discrimination; any relative-cost wording hedged or cited to
    Lenthe 2019, not asserted), hybrid refinement (per-pattern
    PC); one closing sentence that multi-phase spherical indexing
    takes a list of harmonics. **No pseudo-symmetry pointer**
    (re-scope).

### D2 -- Master pattern source: in-package small master at bw 188 (frozen)

- The executed path uses
  `nickel_ebsd_master_pattern_small(projection="lambert",
  hemisphere="both")` -- in-package, no download, 401 px, the master
  behind every Phase 5-10 measured number and behind the shipped
  regression `.sht`. `ebsd_master_pattern("ni")` (full resolution)
  is **relegated to a Note admonition** whose code line mirrors
  `hybrid_indexing.ipynb`'s exactly (a bare call returns a
  multi-energy stack, so the call form is load-bearing):

  ```python
  mp = kp.data.ebsd_master_pattern(
      "ni", projection="lambert", energy=20, allow_download=True
  )
  ```

  Grounds (corrected by the review): the file is a 305.5 MB pooch
  download (verified in the local cache under both `0.9.0` and
  `develop`) -- heavy for Binder and for a first-time reader; the
  in-package master is the configuration every shipped Phase 5-10
  number and the regression `.sht` were measured on; and the
  tutorial's accuracy story does not need it. **Weekly-CI
  feasibility is NOT a ground**: the weekly nbval runner already
  downloads this exact file today for `hybrid_indexing.ipynb`
  (verified: its cell calls `ebsd_master_pattern("ni", ...,
  allow_download=True)` and it is in the `NOTEBOOKS` array) --
  the drafting spec's claim to the contrary was wrong. This
  amends the roadmap box's "uses the full Ni master ... or caps
  bw <= 190" **to the second alternative** -- flagged (plan 7.1).
- Harmonics bandwidth **188**: the largest `fast_bandwidths()`
  member below both the master's information limit
  `(401 - 1)/2 = 200` **and the roadmap's cap 190**. The notebook
  prose must carry both constraints or neither -- frozen sentence:
  "*We use 188, a fast bandwidth comfortably under this limit --
  ample for these patterns*" (or equivalent that does not claim
  maximality). It must **never** say "the largest bandwidth below
  this limit with fast transforms": measured
  `fast_bandwidths(150, 210)` = `[158 163 172 176 182 188 193
  203]`, so 193 is a fast bandwidth below 200 and the claim is
  disproved by the very function the notebook teaches two
  sections later. Measured: no warning at 188; at 384 the Phase 2
  warning fires ("exceeds 200, the largest harmonic degree a
  square Lambert master pattern of side length 401 carries..." --
  verified verbatim in the probe).
  `get_spherical_harmonics(bandwidth=188)` measured **0.08 s** --
  the Lambert dim <= 275 analysis guard is never approached
  because the master is regridded to Legendre dim 191 internally
  (Phase 2 `toLegendre`), which the tutorial does not mention
  (internals).
- Indexing then resizes 188 -> 68 internally, the same
  resize-from-a-larger-stored-bandwidth semantics `IndexEBSD`
  itself uses on `.sht` files (research item 39). NB the tutorial's
  bw-68 spectra are therefore *not bitwise* the regression suite's
  (which resize the mp2sht bw-384 `.sht`); the parity paragraph is
  scoped accordingly (D6).

### D3 -- Dataset and geometry (frozen)

- `nickel_ebsd_large(allow_download=True)`: (75, 55) x (60, 60),
  the Hough exemplar's own dataset, already downloaded by the
  weekly nbval job for `hough_indexing`/`hybrid_indexing`/
  `pattern_matching` (15.4 MB), shipping a calibrated per-point-PC
  detector (`sample_tilt` 70), a static background, **and a
  crystal map** (`s.xmap`, props `scores`/`z`) whose provenance is
  **Hough indexing with PyEBSDIndex followed by orientation and PC
  refinement** (upstream CHANGELOG 0.8.0, PRs pyxem#578/#584;
  corroborated: no `simulation_indices` prop, and
  `pattern_matching.ipynb` never writes `s.xmap`). The drafting
  spec called this "a dictionary-indexing crystal map" -- **wrong**,
  and the tutorial must not: the shipped map is the free
  *refined-reference* for validation (D5), refined with per-point
  PCs (PC std ~0.0028, measured), i.e. the same Hough engine plus
  an independent refinement -- not an independent indexing method.
  It is also the dataset of the Phase 10 `large20`/`large165`
  regression references, which keeps the parity citation honest on
  this very map.
- Geometry: `det = s.detector.deepcopy(); det.pc = det.pc_average`
  -- the Phase 5-10 test convention. No PSO PC estimation section:
  the Hough tutorial owns that story on the same dataset and is
  linked (flagged, plan 7.3).

### D4 -- Indexing configuration, worker pinning, sweep (frozen)

- Headline run: `s.spherical_indexing(harmonics, det,
  bandwidth=68)` -- every other argument at its default, so the
  tutorial teaches the defaults (`refine=True`, `normalize=True`,
  `n_regions=10`, `circular_mask=False`). Measured: **20.0-20.3 s
  for 4125 patterns at 8 workers (204-230 patterns/s)**; info
  message prints "Indexing 4125 pattern(s) in **275** chunk(s) of
  up to **15** pattern(s)" and "Estimated memory per worker: 54
  MB" (corrected by the review: the drafting spec recorded
  "375/11", the *unpinned* 20-worker probe's numbers; the pinned
  prototype's stored output and two independent re-runs all say
  275/15 -- re-measured). Timing prose in the notebook is
  machine-scoped or deferred to the printed `Indexing speed` line
  (D1.7).
- **Score semantics** (frozen teaching point, from the shipped
  `spherical_indexing` docstring Notes): the `"scores"` property
  is *the spherical correlation at the (refined) maximum, divided
  by the rotation-dependent window denominator when
  `normalize=True` -- not a bounded normalized cross-correlation*;
  it is comparable only within one detector geometry and
  bandwidth, and cannot be fed to `orientation_similarity_map` or
  compared with `dictionary_indexing` NCC scores. The drafting
  prototype's "the normalized spherical cross-correlation" is
  **wrong** and must not appear.
- **`_ = dask.config.set(num_workers=8)`** in a visible cell before
  any indexing, with prose on the per-worker memory model. Three
  reasons, recorded: (i) the info-message chunk line becomes
  machine-independent (`_n_workers()` honours the config --
  verified in `_indexer.py:404`), which stored-output nbval needs;
  (ii) it suppresses the >= 2 GiB memory warning a 20-core machine
  gets at bw 68 x 20 workers (measured: the warning fired in the
  unpinned probe), which would otherwise be a machine-dependent
  stderr output; (iii) it anchors the roadmap's "on 8 threads"
  budget. The `_ =` assignment is load-bearing: the bare expression
  outputs `<dask.config.set at 0x...>` and was the **only nbval
  failure of the prototype** (memory address differs per run).
- Bandwidth sweep (**redesigned by the review** -- the drafting
  sweep printed mean score per bandwidth and claimed "higher
  bandwidths give higher scores (sharper correlation peaks)",
  which is exactly the cross-bandwidth score comparison the
  docstring forbids, and a change of metric rather than of
  accuracy): coarse only (`refine=False`), on `s.inav[:40, :30]`
  (1200 patterns), over {53, 68, 88} -- the pre-agreed fallback
  shape adopted as the design (full-map sweeps measured 12.6-45 s
  per bandwidth; the subset keeps the whole sweep at ~20 s). The
  accuracy proxy is the misorientation of each coarse solution to
  the refined headline map's corresponding subset, extracted with
  **CrystalMap slicing `xmap[:30, :40]`** (measured bitwise-equal
  to a refined subset run; NB a manual
  `Orientation.reshape/.flatten` route produced garbage medians of
  ~41 deg in the probe -- CrystalMap slicing is the correct and
  the pedagogical route; the `s.inav[:40, :30]` x-then-y vs
  `xmap[:30, :40]` row-then-column mapping is one teaching
  sentence). Printed per bandwidth (frozen):
  `f"Bandwidth {bw}: median {np.median(ang):.2f} deg, p99
  {np.percentile(ang, 99):.2f} deg to the refined map, {speed:.1f}
  patterns/s"`. Measured: median **0.37 / 0.37 / 0.25 deg**, p99
  **1.40 / 0.71 / 0.69 deg**, at **401 / 228 / 103 patterns/s**
  (bw 53 / 68 / 88; Euler half-cells 1.7 / 1.3 / 1.0 deg). The
  mean score is **not printed** (it would invite the invalid
  comparison; the score-semantics point is D4's frozen sentence
  above). Prose: the coarse error is bounded by the half-cell and
  its tail tightens with bandwidth at roughly cubic cost;
  refinement removes most of the remaining grid error -- which is
  why the default bw 68 *with refinement* is the compromise. The
  default-68 rationale is stated as **EMSphInx's own namelist
  default** and nothing more -- never the drafting prototype's
  invented pattern-size rationale ("for 60 pixel wide patterns"),
  and the unmeasured per-quality guidance ("for difficult, noisy
  data a higher bandwidth can pay off; for high-quality patterns
  53 is often enough") is dropped, or kept only hedged and
  attributed to Lenthe 2019 -- this notebook measures neither.
- Closing residual sentence (frozen, per the docstring -- the
  drafting "the remaining error is dominated by the geometry
  calibration rather than the bandwidth" contradicts the shipped
  docstring's "a larger bandwidth shrinks it"): "*the residual
  against the shipped orientations is set by the PC calibration
  together with the band limit -- a larger bandwidth still
  shrinks it (the `spherical_indexing` docstring's measurement:
  median 0.51 deg at bandwidth 68 vs 0.45 at 88)*" -- attributed,
  so it cannot clash with the tutorial's own 0.43 deg print (a
  different comparison on this map).
- The refined full-map headline plus the coarse sweep gives the
  refine axis its narrative without a separate refine-vs-coarse
  full-map pair (budget).

### D5 -- Validation content (frozen)

- **Hough in-notebook**: phase list built from
  **`mp.phase.deepcopy()`** -- *not* a hand-built Å-lattice Phase.
  Measured constraint: `refine_orientation_spherical` refuses a
  crystal map whose phase has different lattice parameters from the
  master's (`ValueError` "... must be the same, but have different
  lattice parameters"; kikuchipy's EMsoft readers keep nm, so a
  3.5236 Å Hough phase against the 0.35236 nm master phase dies --
  and `hough_indexing.ipynb` cell 25 builds exactly that Å phase).
  The notebook **states the constraint**, frozen: "*building the
  phase list from the master pattern's phase keeps the lattice
  parameters identical (kikuchipy's EMsoft readers use nm);
  `refine_orientation_spherical()` raises a `ValueError` if they
  differ, so a phase built as in the [Hough indexing
  tutorial](hough_indexing.ipynb) (in Å) would be refused*" -- one
  sentence plus the cross-link, so a reader coming from the Hough
  tutorial is not left with an unexplained error. Verified to
  index identically (median vs spherical 0.229 deg both ways).
  Indexer settings copied from the Hough exemplar (`nBands=10,
  tSigma=2, rSigma=2`, reflector list). Measured 4.4-5.2 s
  (922 patterns/s CPU).
- **Shipped-map reference** (relabeled by the review -- see D3;
  never "dictionary" or "DI"): the shipped `s.xmap`, presented as
  "*the Hough-indexed, dynamically refined orientations shipped
  with the dataset (refined with per-point PCs)*". IPF panel
  title: **"Hough + refinement (shipped)"**. The two comparisons
  are *not* two independent engines vs a third -- prose must not
  claim independence; the honest framing is: live Hough (same
  engine as the shipped map, coarser), and the shipped refined map
  (per-point PCs vs the tutorial's `pc_average` -- most of the
  0.43 deg is that PC difference plus refinement). No
  `pattern_matching.ipynb` attribution for the shipped map (it
  never writes `s.xmap`).
- Measured comparisons (the printed lines are frozen at median +
  p99, both 2 decimals, computed on the **`not_indexed`-masked**
  arrays for the Hough pair): spherical vs Hough median **0.23
  deg**, p99 **0.75 deg** (indexed 4124 points; 25 points > 2 deg,
  max 59.99 -- one prose sentence covers these outliers, which the
  `vmax=2` maps saturate: they are points where the two engines
  picked different solutions, plus twins/grain boundaries);
  spherical vs shipped median **0.43 deg**, p99 **1.53 deg** (33
  points > 2 deg, max 59.98; consistent with the Phase 7 165-pt
  refined 0.456 vs the same stored map). The single `not_indexed`
  Hough point (flat index 3334, spurious 47.3 deg against its
  identity rotation) is masked out of the Hough stats and set to
  NaN in the Hough angle map. Misorientations via
  `Orientation(..., symmetry.Oh).angle_with(..., degrees=True)` --
  the constitution's convention, never `reduce()`.
- **Refinement demo**: `refine_orientation_spherical(xmap_hough,
  harmonics, det, bandwidth=68)` -- measured 3.3-3.5 s
  (1167 patterns/s), median misorientation to the spherical
  solution drops 0.229 -> **0.042 deg** (p99 0.128; masked p99
  0.127 -- the print uses the same mask as above): the
  convergence-to-the-same-maxima story. The one `not_indexed`
  Hough point keeps its input row (the documented contract, stated
  in one sentence).

### D6 -- Interop section, thumbnail, parity wording (frozen)

- Files written to `tempfile.mkdtemp()` and removed at the end
  (the `load_save_data` precedent; nothing lands in
  `doc/tutorials/`). Measured: `.sht` 18,388 B, the repacked
  pattern file 14.9 MB (deleted in the cleanup cell), whole
  section < 0.1 s.
- `EMSphInxNamelist.from_kwargs(..., bandwidth=68, n_thread=8)`
  with `scan_shape=(55, 75)`/`scan_steps=(1.5, 1.5)`; the notebook
  prints only the first 5 of the 110 namelist lines (the full
  template is reference material, not tutorial content).
- **Thumbnail**: the three-way IPF figure cell carries
  `{"nbsphinx-thumbnail": {"tooltip": "Spherical harmonic indexing
  of EBSD patterns"}, "tags": ["nbsphinx-thumbnail"]}` -- the
  `hough_indexing` metadata shape exactly.
- **Parameter map** (frozen markdown table -- corrected by the
  review against the actual `from_kwargs(...).to_string()` output,
  which writes exactly these 19 keys: `patfile patdset masterfile
  patdims circmask gausbckg nregions delta pctr vendor thetac
  scandims roimask bw normed refine nthread batchsize datafile`;
  the drafting table's `ipath` and `scanstep` keys **do not
  exist** -- `ipath` is a read-only derived-path concept never
  written, and the steps live inside `scandims = 75, 55, 1.5,
  1.5`): `patfile`/`patdset` -> `write_emsphinx_patterns()`;
  `masterfile` -> `MasterPatternHarmonics.save()`; `patdims` ->
  detector shape; `scandims` -> navigation shape *and* scan steps
  (x-then-y note); `pctr`/`thetac` -> `EBSDDetector`
  (`pc_average` conversion, tilt); **`vendor`** -> a
  `from_kwargs` *argument* (default `"Bruker"`, for which `pctr`
  is the kikuchipy PC verbatim) -- not read off the detector;
  **`delta`** -> *not* taken from `detector.px_size`
  (deliberately, per the docstring -- kikuchipy fixtures carry
  `px_size=1.0`, which `IndexEBSD` would reject); defaults to
  `30000 / pat_dims[0]`, a detector exactly 30 mm wide (measured:
  `delta = 500` here); `bw` -> `bandwidth`; `normed` ->
  `normalize`; `refine` -> `refine`; `nregions` -> `n_regions`;
  `gausbckg` -> `gaussian_background`; `circmask` ->
  `circular_mask` (default off, `-1`); `nthread`/`batchsize` ->
  dask `num_workers`/`chunksize`.
- **Scoped parity paragraph** (frozen wording, honouring Phase 10
  D10 -- the shipped numbers are for the 401-px in-package master
  resized from the `ni_small_20kv_bw384.sht` fixture at bandwidth
  68, and do not transfer to other configurations; the tutorial's
  own harmonics are bw-188 sourced from the Lambert master, not
  that `.sht`, so the claim is attributed to the suite and its
  configuration and **not** to the tutorial's run. Revised by the
  review: the drafting notebook said "on this dataset and
  configuration ... about 0.3°" -- *both* halves wrong: the
  tutorial's configuration is not the suite's, and the Phase 10
  D6 medians are 0.31-0.34 with the docstring saying "about 0.34
  degrees", so 0.3 rounds the wrong way. **The notebook carries
  this paragraph verbatim** -- the drafting spec declared the
  wording frozen while the prototype carried different text; the
  spec text below is the single normative string):

  > The agreement between the two implementations is measured and
  > pinned by a regression suite shipped with kikuchipy, which
  > compares both engines on these nickel datasets at bandwidth
  > 68, against the master pattern shipped as
  > `ni_small_20kv_bw384.sht`: orientations agree to a median
  > misorientation of about 0.34°, correlation scores to about
  > 2%, and image qualities to floating-point precision. These
  > numbers are for that configuration; your own bandwidth and
  > master pattern will differ. Most of the remaining orientation
  > difference is a documented half-pixel detector-sampling
  > convention difference in EMSphInx; accounting for it, the two
  > agree to better than 0.1°.

  (Numbers: Phase 10 D6/D7 -- refined medians 0.31-0.34 deg,
  scores mean |diff| < 0.03 at r 0.94-0.97, IQ <= 1.4e-8, stretch
  decomposition 0.07-0.09 deg. The docstring `Notes` on
  `EBSD.spherical_indexing` carry "about 0.34 degrees" in `src/`.)

### D7 -- Stored outputs and the nbval treatment (frozen)

- **Outputs are stored.** The repo precedent is two-mode:
  `nbsphinx_execute = "auto"` (verified, `doc/conf.py:168`)
  executes only output-less notebooks at docs-build time --
  `hough_indexing` ships output-less and executes on RTD;
  `pattern_matching`/`hybrid_indexing` ship stored outputs and are
  *not* executed at build time. An expensive indexing notebook
  follows the second mode (the constitution's tech-stack bullet
  says exactly this, "as `pattern_matching.ipynb` does").
- **Quantified rebuttal of the dev guide's output-less
  recommendation** (added by the review, which correctly noted
  `doc/dev/building_writing_documentation.rst` recommends storing
  output only for notebooks "too computationally intensive for the
  Read the Docs server ... 15 minutes and 3 GB" and that this
  spec's decision must rebut it, not ignore it): (i) the 15-min
  limit is **per build of the whole docs**, which already
  live-executes every output-less notebook including
  `hough_indexing`'s full 4125-pattern Hough run; (ii) the two
  heavy cells (indexing 21.2 s + sweep 20.7 s on 8 workers of 20
  real cores) are compute-bound and scale with real cores -- on
  the ~2-vCPU RTD builder a x4-6 slowdown puts this notebook
  alone at an estimated **4-6 min, a third of the whole build
  budget** (the 15.4 MB `nickel_ebsd_large` pooch download is
  already paid by `hough_indexing` there); (iii) the *same* dev
  guide's operative paragraph says "For computationally expensive
  notebooks however, we store the cell outputs so the
  documentation doesn't take too long to build" -- which is the
  policy `pattern_matching.ipynb` and `hybrid_indexing.ipynb`
  (this notebook's exact class: full-map indexing) actually
  follow; (iv) stored outputs are what the weekly nbval job
  compares digit-for-digit, which is this phase's determinism
  story. The tech-stack amendment (plan 0.2) records the
  threshold so the constitution and the dev guide stop
  disagreeing silently.
- **Production recipe**: execute once from the repo root with
  `uv run --with ipykernel jupyter nbconvert --to notebook
  --execute --inplace doc/tutorials/spherical_indexing.ipynb`
  (ipykernel is not in the project venv -- verified `No such
  kernel named python3` without the `--with`), then **strip the
  `metadata.widgets` block** nbconvert adds at the notebook level
  (measured: a 17.7 kB `application/vnd.jupyter.widget-state+json`
  block of per-run tqdm model ids; none of
  `hough_indexing`/`pattern_matching`/`hybrid_indexing` carries
  one -- they store widget-*view* outputs only -- and nbval does
  not compare notebook-level metadata, so committing it would be
  17.7 kB of diff churn per re-execution that no gate catches):
  `python -c "import json, io; p =
  'doc/tutorials/spherical_indexing.ipynb'; nb =
  json.load(open(p, encoding='utf-8'));
  nb['metadata'].pop('widgets', None); json.dump(nb, open(p, 'w',
  encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)"`
  plus a trailing newline (or the `jq` equivalent) -- frozen as
  part of the recipe. Measured: the stored form nbconvert
  produces (tqdm placeholder frames, dask ProgressBar as
  consecutive per-update stream entries) matches the *committed*
  `pattern_matching.ipynb` form (its `remove_static_background`
  cell stores the same `" 0%| ..."` placeholder frames), and
  **nbval passes against it**: nbval coalesces consecutive stream
  outputs and collapses carriage returns before comparing, so the
  update-count difference between runs cancels. Evidence: the
  executed prototype re-run under `pytest --nbval` with the
  repo's sanitize file -- 23/24 pass, the failure being D4's
  since-fixed `dask.config.set` repr; the review's independent
  re-run of the fixed prototype: **24/24**.
- **Existing sanitize rules already cover** the dask ProgressBar
  times (`regex1`), all three `patterns/s` prints -- the indexing
  speed, the sweep lines, the refinement speed -- (`regex2`), the
  `Refining 4124 orientation(s)` line (`regex6`; 4124, not 4125 --
  one Hough point is `not_indexed`; corrected by the review), and
  figure-size reprs (`regex5`).
- **Two new rules** (frozen text, appended to
  `tutorials_sanitize.cfg`):

  ```
  [regex8]
  regex: (?<=PyOpenCL: )(True|False)
  replace: BOOL
  # Example: PyOpenCL: False -> PyOpenCL: BOOL
  # (the weekly workflow installs pyopencl; local runs may not)

  [regex9]
  regex: in \d+ chunk\(s\)( of up to \d+ pattern\(s\))?
  replace: in N chunk(s)
  # Example: Indexing 4125 pattern(s) in 275 chunk(s) of up to 15
  # pattern(s) -> Indexing 4125 pattern(s) in N chunk(s)
  ```

  (The comment example is corrected by the review: the pinned
  chunk line is 275/15, not the drafting spec's 375/11.)
  `regex8` is required: `hough_indexing()` prints its info message
  regardless of `verbose` (measured), and `PyOpenCL:` is True on
  the weekly runner (it installs pyopencl) and False here.
  `regex9` is **also required, not belt-and-braces** (the review's
  correction): it is the only rule covering the Hough info block's
  `in 8 chunk(s)` (pyebsdindex-internal, machine-dependent) and it
  additionally armours both spherical chunk lines (`in 275
  chunk(s) of up to 15 pattern(s)` appears in the indexing *and*
  the refinement info blocks; `regex6` covers only the `Refining
  N` count) against dask-heuristic changes across versions.
  Sanitize rules apply to both sides of the comparison, so they
  can only make existing notebooks' checks more lenient -- no
  regression risk to the other **six** NOTEBOOKS entries
  (counted: hough_indexing, hybrid_indexing,
  mandm2021_sunday_short_course, pattern_matching,
  pc_extrapolate_plane, pc_fit_plane -- the drafting spec said
  five).
- **Printed-precision rule** (drift armour): every number printed
  by the notebook is held to a precision safe under the Phase 10
  fastmath drift ladder -- IQ at 3 decimals (one uint8 gray level
  moves a pattern's IQ by up to 5.2e-5), mean scores at 3
  decimals, medians at 2-3 decimals. A value landing on a rounding
  boundary on another machine is the recorded residual nbval risk;
  the fix ladder is in `validation.md` (failure modes).

### D8 -- The `load_save_data.ipynb` `.sht` section (frozen)

- Verified: no `sht` match in the committed notebook -- the roadmap
  conditional resolves to **in scope**.
- One new `### EMSphInx spherical harmonics master pattern (.sht)`
  subsection inserted after `### EMsoft TKD master pattern HDF5`
  (cell 60) and before `### kikuchipy h5ebsd` -- the master-pattern
  cluster, and alphabetically EMSphInx follows EMsoft. Two cells:
  - markdown: master patterns stored as spherical harmonic
    coefficients in EMSphInx's `.sht` format are read with
    `kp.load()` as an `EBSDMasterPattern` synthesized from the
    harmonics; files are available from the EMSphInx master
    pattern library (related-projects link); writing goes through
    `EBSDMasterPattern.get_spherical_harmonics()` /
    `MasterPatternHarmonics.save()` (link), see the spherical
    indexing tutorial (relative notebook link);
  - code (live, in-package -- no download):

    ```python
    mp_sht = kp.load(
        data_path / "emsphinx/ni_small_20kv_bw384.sht"
    )
    mp_sht
    ```

    (`data_path = Path("../../src/kikuchipy/data")` is already
    defined in cell 4; measured `kp.load` of a `.sht` < 0.1 s.)
- The format table in cell 46 gains the row
  `| [EMSphInx spherical harmonics master pattern (.sht)](#EMSphInx-spherical-harmonics-master-pattern-(.sht)) | Yes | No |`
  -- link text **identical to the heading it anchors** (the
  review's correction; every other row does this), Write **No**
  because the io plugin is read-only (`specification.yaml:
  writes: False`, verified); the markdown sentence carries the
  `MasterPatternHarmonics.save()` route, the same shape as other
  read-only rows with adjacent write guidance. Frozen placement
  and formatting (the review flagged both as unspecified): the
  row goes **between the EMsoft TKD master pattern row and the
  kikuchipy h5ebsd row**, mirroring section order; the existing
  rows are **not re-padded** -- the new row is longer than the
  current 73-char padding width, and markdown renders a ragged
  column identically, so the diff stays two cells + one table
  line (V9's hunk check depends on this). (Anchor text with
  parentheses is checked against nbsphinx's anchor generation at
  implementation -- precedent exists in the same table,
  `(#Oxford-Instruments-h5ebsd-(H5OINA))`; fall back to a plain
  nearby anchor if the parenthesised one does not resolve --
  validation "html render" step.)
- `load_save_data.ipynb` stores outputs (42 of 52 code cells) and
  is **not** in the nbval NOTEBOOKS array -- the new cell follows
  both facts: stored output, no nbval wiring change.

### D9 -- CHANGELOG consolidation (frozen text)

"Consolidation" concretely: the **nine** spherical `Added` entries
accreted by PRs #5/#8/#9/#10 tell the story in accretion order
(three `.sht` entries, three interop entries, one indexing entry,
its bandwidth helper split out, refinement split from indexing --
3+3+1+1+1 = 9; the drafting spec said eight, corrected by counting
the committed file). They are rewritten into **three** entries --
one per user-facing capability, descending-chronological as the
file requires, every original PR link preserved, no entry for
Phase 10 (its no-CHANGELOG call is recorded in that spec) -- plus
the new tutorial entry, **four bullets total**. A further warrant
for curating `Unreleased` (the review's suggestion): `RELEASE.rst`
line 35 instructs "Review and clean up `CHANGELOG.rst` as per Keep
a Changelog" before a release -- an in-repo instruction, not just
the external convention. The `Unreleased -> Added` block becomes
exactly:

```rst
Added
-----
- Tutorial on spherical indexing, ``doc/tutorials/spherical_indexing.ipynb``, and a
  section on reading EMSphInx ``*.sht`` master pattern files in the load/save
  tutorial.
  (`#12 <https://github.com/jwestraadt/kikuchipy/pull/12>`_)
- Interoperability with EMSphInx's ``IndexEBSD`` program:
  ``kikuchipy.indexing.write_emsphinx_patterns()`` writes EBSD patterns to the
  repacked HDF5 layout it reads (``PatternRepack`` equivalent plus the required
  ``Manufacturer`` dataset), ``kikuchipy.indexing.EMSphInxNamelist`` reads and writes
  its namelist input files and converts them to and from spherical-indexing arguments
  and an ``EBSDDetector``, and ``kikuchipy.io.plugins.oxford_binary.get_scan_info()``
  probes the scan grid and layout of an ``.ebsp`` file (``EBSPDims`` equivalent).
  (`#10 <https://github.com/jwestraadt/kikuchipy/pull/10>`_)
- Spherical indexing of EBSD patterns against one or more master patterns, a CPU port
  of EMSphInx's ``IndexEBSD``: ``EBSD.spherical_indexing()``,
  ``kikuchipy.indexing.SphericalIndexer``,
  ``kikuchipy.indexing.SphericalBackProjector`` and
  ``kikuchipy.indexing.fast_bandwidths()``. Newton refinement of the correlation
  maximum on the sphere is on by default (EMSphInx's own default; pass
  ``refine=False`` for coarse-only indexing), and
  ``EBSD.refine_orientation_spherical()`` refines orientations from any source, e.g.
  Hough indexing.
  (`#8 <https://github.com/jwestraadt/kikuchipy/pull/8>`_,
  `#9 <https://github.com/jwestraadt/kikuchipy/pull/9>`_)
- EBSD master patterns as spherical harmonics:
  ``EBSDMasterPattern.get_spherical_harmonics()`` returning the new class
  ``kikuchipy.indexing.MasterPatternHarmonics`` (``mp2sht`` equivalent), read from
  and written to EMSphInx ``*.sht`` files, and ``kp.load("*.sht")`` reading a
  ``.sht`` master pattern directly (io plugin ``emsphinx_master_pattern``).
  (`#5 <https://github.com/jwestraadt/kikuchipy/pull/5>`_)
```

The empty `Fixed/Changed/Removed/Deprecated` headers stay. This
rewrites shipped (fork-merged) changelog text -- **flagged** (plan
7.5); the defence is that `Unreleased` is exactly the block Keep a
Changelog expects to be curated before a release, and the upstream
PR series will rewrite these links wholesale anyway (the recorded
upstream-submission checklist).

### D10 -- `index.rst` placement (frozen)

`spherical_indexing` goes into the Indexing `nbgallery` after
`pattern_matching`, before `hybrid_indexing`:

```rst
    hough_indexing
    pattern_matching
    spherical_indexing
    hybrid_indexing
```

Rationale: the mission's "sits next to the Hough and dictionary
indexing tutorials"; the gallery is ordered pedagogically (the
three primary methods, then hybrid which composes the first two,
then the PC series) -- flagged (plan 7.4).

### D11 -- CI and build impact (recorded)

- **Docs build does not execute the notebook**: stored outputs +
  `nbsphinx_execute = "auto"`; `nbsphinx_allow_errors = True`
  besides. RTD (`readthedocs.yaml`) builds get the stored outputs;
  build-time cost is rendering only (~1 MB page with 7 figures).
- **No docs job in `tests.yml`** (verified: only doctests run
  there); the notebook executes on CI only in the **weekly**
  `test-documentation-notebooks` job via `run_nbval.sh`
  (ubuntu, py3.13, `pip install -e .[doc,tests,all]` + nbval +
  pyopencl, 30-min timeout). Cost added, bounded (the review asked
  for a number, not "dwarfed"): ~80 s on the 20-core drafting
  machine, of which ~42 s is the two core-scaling cells; on the
  ~4-vCPU ubuntu runner estimate x3-4 on those -> **~3-5 min
  added** to a job whose 30-min timeout is shared by the six
  existing notebooks (one of which, `pattern_matching`, runs full
  DI, and another, `hybrid_indexing`, downloads the 305.5 MB Ni
  master -- so both the compute and the download precedents are
  larger than this addition). If the weekly job approaches its
  timeout, the recorded mitigation ladder is: sweep to two
  bandwidths, then headline on an `inav` subset (the roadmap's
  pre-agreed fallbacks). Estimate recorded as an estimate; the
  first weekly run after merge is the measurement.
  `nickel_ebsd_large` is already downloaded by the existing
  entries, and the small master is in-package. No new downloads.
- The weekly runner has pyopencl -> `PyOpenCL: True` in the Hough
  cell -> `regex8` (D7). The runner's core count differs ->
  `regex9` + the D4 worker pin.
- **linkcheck**: new external links are doi.org (Lenthe 2019),
  github.com/EMsoft-org/EMSphInx, pyebsdindex.readthedocs.io,
  orix.readthedocs.io -- all already whitelisted-by-precedent
  domains in the docs. NB (the review's correction): a full
  `sphinx-build -b linkcheck` of this repo does **not** exit 0 --
  `doc/conf.py` sets no `linkcheck_ignore`/retries, and
  `CHANGELOG.rst` alone carries 312 pyxem PR URLs that rate-limit
  -- so the gate is scoped to *this phase's* links via the
  linkcheck `output.json` (validation V6), with the doctree cache
  shared between the html and linkcheck builders so the
  output-less notebooks are not re-executed twice. Internal links
  are relative (`hough_indexing.ipynb`,
  `../reference/generated/*.rst`, `../user/related_projects.rst`
  -- the prototype's one absolute kikuchipy.org link is replaced
  by the relative form at implementation, recorded in plan 1.3).
- Binder: the notebook is runnable there (in-package master, 15 MB
  download, 8 dask threads on 1-2 cores is GIL-mild); the heavy
  cells are the same ones every indexing tutorial has.

### D12 -- Gates for a documentation phase (recorded)

Phase 11 skips the failing-tests gate (documentation-only; the
"tests" surface is the validation matrix in `validation.md`) but
keeps: plan approved (autonomous mode, flags in plan section 7) ->
spec recorded -> implementation -> adversarial review + fixes ->
pre-commit clean -> **CHANGELOG entry** (the D9 tutorial line) ->
PR opened (#12). NB the roadmap's gate sentence as written says
documentation-only phases "skip the failing-tests **and
CHANGELOG** gates" -- keeping the CHANGELOG gate here would
contradict the constitution, so plan 0.1 **amends the gate
sentence** (the CHANGELOG gate applies when a documentation phase
ships a user-visible deliverable, as this one does; Phase 0's
constitution-only work is what the skip was written for). The
review flagged the unamended contradiction. Bug injection has
limited meaning for a notebook; the review instead executes the
validation matrix and the failure-mode list (validation.md),
which is what the roadmap's "adversarial review" gate means here.

## Context

- Constitution: `specs/roadmap.md` Phase 11 box (rewritten with
  measured numbers in plan 0.1; the "full Ni master or cap 190"
  alternative resolved per D2; the fallback list recorded as live
  options, machine-scoped margin) **and the gate sentence amended
  per D12** (CHANGELOG gate for doc phases with user-visible
  deliverables); `specs/tech-stack.md` "Tests, docs, data" tutorial
  bullet (hidden first cell, thumbnail tag, black 77 -- enforced
  by the `black-jupyter --line-length=77` pre-commit hook,
  verified in `.pre-commit-config.yaml` -- stored outputs,
  NOTEBOOKS entry, sanitize cfg: all honoured; amended with the
  measured conventions in plan 0.2); `specs/mission.md` (the
  tutorial deliverable sentence -- satisfied; no amendment).
- Phase 10 hand-off: D10 ("the tutorial must either match that
  configuration or scope its parity claim") -- **scoped**, D6; the
  D9 docstring `Notes` are the in-`src/` carrier of the cited
  numbers.
- Exemplars: `doc/tutorials/hough_indexing.ipynb` (structure,
  hidden cell, thumbnail metadata shape, no-PSO-duplication),
  `pattern_matching.ipynb` (stored-outputs mode, nbval + sanitize
  precedent, tqdm/ProgressBar stored form), `load_save_data.ipynb`
  (temp-dir + cleanup precedent, format-section template,
  `data_path`).
- Verified repo facts this spec relies on: `nbsphinx_execute =
  "auto"` + `nbsphinx_allow_errors = True` (`doc/conf.py`);
  `run_nbval.sh` array + weekly-only invocation
  (`.github/workflows/weekly.yml`); the six existing sanitize
  rules; `hough_indexing.ipynb` committed output-less vs
  `pattern_matching.ipynb` with 40 output cells;
  `emsphinx_master_pattern` plugin `writes: False`; pre-commit
  `black-jupyter` at 77 on `.ipynb`; the in-package `.sht` files
  under `src/kikuchipy/data/emsphinx/`; the GPL `licenseheaders`
  hook **does** rewrite `.sh` files (probed: it prepends a 19-line
  `##` header to the header-less `doc/tutorials/run_nbval.sh`,
  shebang kept first; `.ipynb`/`.rst`/`.cfg` are outside its
  extension map) while pre-commit.ci skips the hook entirely
  (`ci: skip: [licenseheaders]`) and no local git pre-commit hook
  is installed in this clone -- so the only channel that would
  stamp `run_nbval.sh` is an explicit `pre-commit run --files`
  over it, which V8 therefore avoids (validation V8).
- Downstream: Phase 8, when un-deferred, appends its
  pseudo-symmetry section to this notebook (a new `##` section
  before "What's next?") and adds its own CHANGELOG entry; nothing
  in this phase blocks or presupposes it.
