# Phase 11 -- `spherical-indexing-tutorial`: plan

Branch `spherical-indexing-tutorial` off `develop` (after the Phase
10 merge, jwestraadt/kikuchipy#11). Models: plan/spec on Fable 5
(xhigh, ultracode); implementation, adversarial review and fixes by
Opus 5 agents (xhigh, ultracode). Autonomous mode (approval gate
waived for spherical-indexing phases; decisions flagged in section
7). Documentation-only phase: the failing-tests gate is skipped per
the roadmap rule; the CHANGELOG gate applies (requirements D12 --
section 0.1 amends the roadmap's gate sentence, which as written
would skip it too). Every number in `requirements.md` was measured
on 2026-09-03 with the drafting probes (`p11_time.py`,
`p11_time2.py`, and the revision probe `p11_revise_probe.py`) and
the executed prototype (`build_nb.py` ->
`spherical_indexing.ipynb`/`spherical_indexing_executed.ipynb`;
session scratchpad, not committed; recipes and full outputs in
`validation.md` "Recorded results" and "Re-measurements"). The
prototype executed end-to-end (72-81 s kernel time) and passed
23/24 nbval cells against its own stored outputs with the repo's
existing sanitize file; the 24th is fixed in the inventory below,
and the review's independent re-run of the fixed prototype passed
24/24. This plan was **revised 2026-09-03** after the adversarial
drafting review; the inventory below is the post-review normative
content (color-key cell, sweep redesign, misorientation stats,
relabeled shipped-map reference, cleanup-cell metadata).

## 0. Constitution amendments (applied 2026-09-03 in the spec commit)

1. `specs/roadmap.md`, Phase 11 box: rewrite into six deliverable
   boxes with the measured facts (applied in the spec commit,
   ticked as gates complete), **plus** the gate-sentence amendment
   below:
   - box 1: "`doc/tutorials/spherical_indexing.ipynb` (~50 cells,
     stored outputs, ~1 MB, 7 figures + IPF color key; in-package
     401-px master at harmonics bw 188, a fast bandwidth under
     both the (401-1)/2 = 200 information limit and the cap 190 --
     the full `ebsd_master_pattern('ni')` is a Note only, 305 MB
     download; indexes `nickel_ebsd_large` (4125 patterns) at bw
     68 refined in ~20 s on 8 pinned dask workers (204-230 pat/s)
     on the 20-core drafting machine, coarse bw {53, 68, 88}
     sweep on a 1200-pattern subset ~20 s reporting median
     misorientation to the refined map (0.37/0.37/0.25 deg) +
     speed, Hough ~5 s, `refine_orientation_spherical` on the
     Hough map ~3.4 s (median to the spherical solution 0.229 ->
     0.042 deg); validation vs in-notebook Hough (median 0.23
     deg) and the shipped Hough+refined xmap (0.43 deg -- NOT a
     DI reference, per its 0.8.0 provenance); interop cells write
     `.sht`/repack/namelist to a temp dir; parity claim scoped to
     the Phase 10 regression suite per its D10 (0.34 deg at bw 68
     vs the `ni_small_20kv_bw384.sht` master); measured total
     ~80 s end-to-end on the drafting machine -- the <= 3 min /
     8-thread budget holds 2.2x there; fallbacks (bw 53, `inav`
     subset, `refine=False`) stay live for slower runners; the
     pseudo-symmetry section is omitted until Phase 8 lands
     (re-scope 2026-09-02))".
   - box 2: "`doc/tutorials/index.rst` entry (Indexing gallery,
     after `pattern_matching`)".
   - box 3: "nbval wiring: `NOTEBOOKS` entry in `run_nbval.sh`
     (alphabetical slot), `tutorials_sanitize.cfg` + regex8
     (PyOpenCL bool) + regex9 (chunk counts); stored outputs pass
     nbval on the drafting machine 24/24 after the
     `_ = dask.config.set(num_workers=8)` fix (the bare repr's
     memory address was the one failure)".
   - box 4: "`.sht` section in `load_save_data.ipynb` (verified
     absent from Phase 2: zero grep matches) + format-table row
     (Read Yes / Write No -- plugin `writes: False`); CHANGELOG
     consolidation 9 -> 3 entries + the tutorial entry = 4
     bullets (PR #12 link)".
   - box 5: "adversarial review (validation matrix + failure-mode
     list -- clean-kernel execute, nbval, html render inspection,
     linkcheck of the phase's links via output.json, name/spell
     pass) + fixes; `sphinx-build -b html` exit 0".
   - box 6: "signed commits pushed; PR #12 opened".
   - Gate-sentence amendment (requirements D12): the rule
     "Documentation-only phases (Phase 0) skip the failing-tests
     and CHANGELOG gates" becomes "Documentation-only phases skip
     the failing-tests gate; they also skip the CHANGELOG gate
     unless they ship a user-visible documentation deliverable
     (Phase 0 did not; Phase 11 does)". Without this, keeping the
     D9 CHANGELOG entry would contradict the constitution as
     written (review finding).
2. `specs/tech-stack.md`, "Tests, docs, data", the tutorial-rules
   sentence: extend with the measured conventions -- "tutorial
   notebook rules: hidden first cell, thumbnail tag, black at 77
   (enforced by the `black-jupyter --line-length=77` pre-commit
   hook), registered in `doc/tutorials/index.rst`; outputs are
   stored for notebooks whose estimated execution on the ~2-vCPU
   Read the Docs builder would consume a significant share of the
   15-min build limit (>~2 min there -- `pattern_matching.ipynb`,
   `hybrid_indexing.ipynb` and `spherical_indexing.ipynb` store;
   `hough_indexing.ipynb` stays output-less and executes live;
   this quantifies the dev guide's
   `building_writing_documentation.rst` two-mode rule --
   `nbsphinx_execute = 'auto'` then skips stored-output notebooks
   at docs-build time), produced with
   `uv run --with ipykernel jupyter nbconvert --to notebook
   --execute --inplace <nb>` (ipykernel is not a project
   dependency) followed by stripping the notebook-level
   `metadata.widgets` block nbconvert adds (17.7 kB of per-run
   tqdm model ids; no committed tutorial carries one), the
   notebook is added to the `NOTEBOOKS` array in
   `doc/tutorials/run_nbval.sh` and non-deterministic output is
   covered by `doc/tutorials/tutorials_sanitize.cfg` (timings by
   regex1/regex2, PyOpenCL by regex8, chunk counts by regex9 --
   nbval coalesces carriage-return stream updates, so dask
   ProgressBar cells compare on their final line); notebooks pin
   `dask.config.set(num_workers=8)` before verbose indexing so
   chunk lines and memory warnings are machine-independent, print
   floats at drift-safe precision (IQ <= 3 decimals -- the
   one-gray-level IQ sensitivity is 5.2e-5), and never leave a
   bare expression whose repr embeds a memory address".
3. `specs/mission.md`: no amendment (the tutorial-deliverable
   sentence is satisfied as written).
4. Research addendum, `specs/_research/explore-kikuchipy-conventions.md`
   (docs section): the two-mode notebook precedent measured --
   `hough_indexing` committed output-less (executed by nbsphinx
   auto), `pattern_matching` with stored outputs (skipped by
   nbsphinx, compared by weekly nbval); the dev guide's 15-min/3-GB
   RTD threshold and its "so the documentation doesn't take too
   long to build" paragraph, quantified per requirements D7; nbval's
   stream coalescing + CR collapse is what lets stored multi-update
   ProgressBar streams pass; `hough_indexing()` prints its info
   message regardless of `verbose` (PyOpenCL line --
   machine-dependent); `EBSD.spherical_indexing`'s chunk line
   depends on `dask.config num_workers` (honoured via
   `_n_workers()`); nbconvert adds a notebook-level
   `metadata.widgets` state block that must be stripped before
   commit; the shipped `nickel_ebsd_small`/`nickel_ebsd_large`
   crystal maps are Hough+refined (upstream 0.8.0, pyxem#578/#584),
   **not** dictionary indexing; the GPL `licenseheaders` hook
   rewrites `.sh` files (probed on `run_nbval.sh`) but not
   `.ipynb`/`.rst`/`.cfg`, and pre-commit.ci skips it.
5. Tick the roadmap Phase 11 boxes only as gates complete (spec
   commit, implementation, review, builds, PR #12).

## 1. The notebook -- `doc/tutorials/spherical_indexing.ipynb`

1. Recreate the drafting prototype in `doc/tutorials/` (the
   builder `build_nb.py` and the executed prototype live in the
   drafting session's scratchpad; the cell inventory below plus
   requirements D1-D6 is the normative content -- re-typing from
   it is acceptable and expected, and the review deltas marked
   *(revised)* below are **not** in the prototype). ~50 cells (25
   markdown, 25 code); markdown cells at the D1 section list; code
   cells, in order, with their load-bearing content:
   1. imports: `%matplotlib inline`; `tempfile`, `Path`, `dask`,
      `matplotlib.pyplot`, `numpy`; `kikuchipy as kp`,
      `from orix import plot`,
      `from orix.crystal_map import PhaseList`,
      `from orix.quaternion import Orientation, symmetry`;
      `plt.rcParams.update({"figure.facecolor": "w",
      "font.size": 15})`.
   2. `s = kp.data.nickel_ebsd_large(allow_download=True)` +
      `remove_static_background()` + `remove_dynamic_background()`
      + `s` repr.
   3. `mp = kp.data.nickel_ebsd_master_pattern_small(
      projection="lambert", hemisphere="both")` + repr.
   4. `harmonics = mp.get_spherical_harmonics(bandwidth=188)` +
      repr (the class repr is one line: `MasterPatternHarmonics:
      bw = 188, ni (m-3m), 20.1 keV, 70.0 deg` -- measured; no
      `describe()` dump, it is 60 lines of reference material).
   5. power spectrum: `harmonics.power_spectrum()` + semilogy.
   6. round trip *(revised)*: `mp_synth =
      harmonics.to_master_pattern(dim=401)` -- `dim=401` so both
      panels are 401 px, like-for-like (the default 377-px grid
      made the drafting comparison unequal); two-panel imshow
      `mp.data[0]` vs `mp_synth.data[0]`.
   7. `det = s.detector.deepcopy(); det.pc = det.pc_average; det`.
   8. back-projection: `projector =
      kp.indexing.SphericalBackProjector(det, bandwidth=68)`;
      `north, south, iq = projector.unproject(pattern,
      return_image_quality=True)`; three-panel imshow (pattern,
      `north`, `projector.window_mask()[0]`);
      `print(f"Image quality: {iq:.3f}")`.
   9. `_ = dask.config.set(num_workers=8)` -- the assignment is
      load-bearing (nbval; requirements D4).
   10. `xmap = s.spherical_indexing(harmonics, det, bandwidth=68)`
       (defaults otherwise; verbose info + ProgressBar +
       "Indexing speed" are the stored output).
   11. `xmap` repr.
   12. scores + IQ maps via `xmap.get_map_data(...)`, two-panel
       imshow with colorbars.
   13. commented `orix.io.save` export cell.
   14. `kp.indexing.fast_bandwidths(bandwidth_min=50,
       bandwidth_max=130)`.
   15. the sweep *(revised, requirements D4)*: `from time import
       time` (mid-cell import, exemplar precedent); `s_sub =
       s.inav[:40, :30]`; `ori_sub =
       Orientation(xmap[:30, :40].rotations, symmetry.Oh)`
       (CrystalMap slicing -- NOT a manual
       `Orientation.reshape`/`.flatten` route, which the revision
       probe measured to scramble the map order); loop over
       `[53, 68, 88]`, `refine=False, verbose=0`, computing `ang
       = Orientation(xmap_bw.rotations,
       symmetry.Oh).angle_with(ori_sub, degrees=True)` and
       printing `f"Bandwidth {bw}: median {np.median(ang):.2f}
       deg, p99 {np.percentile(ang, 99):.2f} deg to the refined
       map, {speed:.1f} patterns/s"`. No mean-score print
       (cross-bandwidth scores are not comparable).
   16. `phase_list = PhaseList(mp.phase.deepcopy())` + repr (the
       nm-lattice `ValueError` constraint stated in the adjacent
       markdown, requirements D5).
   17. `indexer = det.get_indexer(phase_list, [[1, 1, 1], [2, 0, 0],
       [2, 2, 0], [3, 1, 1]], nBands=10, tSigma=2, rSigma=2)`;
       `xmap_hough = s.hough_indexing(phase_list=phase_list,
       indexer=indexer, verbose=0)` + repr (the info message
       prints anyway -- regex8/regex9 cover it; adjacent markdown
       notes the one `not_indexed` point the repr shows).
   17b. *(revised, new cell)* `ckey =
       plot.IPFColorKeyTSL(symmetry.Oh)` + `ckey` repr (the
       `hough_indexing` cell-49 shape; prose: colors encode the
       crystal direction along sample Z).
   18. the three-way IPF figure *(revised)* (spherical | Hough |
       `s.xmap` titled **"Hough + refinement (shipped)"** --
       never "Dictionary"), plus the color key as an inset axes
       (`fig.add_axes(..., projection="ipf",
       symmetry=symmetry.Oh)` + `plot_ipf_color_key`, the
       `hough_indexing` cell-51 shape; exact inset coordinates
       decided at implementation against the rendered page, V5),
       **thumbnail metadata** per requirements D6.
   19. misorientation stats *(revised, requirements D5)*: `mask =
       xmap_hough.phase_id.flatten() != -1`; median + p99 prints
       at 2 decimals for spherical-vs-Hough (masked) and
       spherical-vs-shipped; two-panel angle maps (`vmax=2`, the
       Hough panel's `not_indexed` point set to NaN); one prose
       sentence on the >2 deg outliers the color scale saturates.
   20. `xmap_hough_ref = s.refine_orientation_spherical(
       xmap_hough, harmonics, det, bandwidth=68)`.
   21. overlaid histogram + `print(f"Median after refinement:
       {np.median(angles_ref):.3f} deg")`.
   22. interop: `temp_dir = Path(tempfile.mkdtemp())`;
       `harmonics.save(temp_dir / "ni_20kv.sht")`;
       `mp_sht = kp.load(temp_dir / "ni_20kv.sht")` + repr.
   23. `kp.indexing.write_emsphinx_patterns(temp_dir /
       "patterns.h5", s)`; `nml =
       kp.indexing.EMSphInxNamelist.from_kwargs(
       pattern_file="patterns.h5", master_files=["ni_20kv.sht"],
       detector=det, scan_shape=(55, 75), scan_steps=(1.5, 1.5),
       data_file="out.h5", bandwidth=68, n_thread=8)`;
       `nml.write(temp_dir / "IndexEBSD.nml")`; print the first 5
       lines of `nml.to_string()`.
   24. cleanup *(revised)*: `{"nbsphinx": "hidden"}` metadata and
       the `# Remove files written to disk in this tutorial`
       opening comment (the `load_save_data` cell-109 /
       `pattern_matching` cell-95 precedent -- both hidden; the
       drafting spec froze neither, so the housekeeping would
       have rendered on the public page); `import os` **inside
       the cell** (the `pattern_matching` precedent; the drafting
       inventory's imports cell never listed `os`, though the
       prototype had it here); `os.remove` loop over `temp_dir` +
       `os.rmdir`; final line `temp_dir.exists()` whose stored
       `False` output gives the cleanup a compared, deterministic
       output (validation FM4).
2. Frozen prose blocks from requirements: the intro + Note with
   the port/transform wording (D1.2), the bandwidth-choice and
   20.1 keV sentences (D2/D1.4), the master-pattern-sources
   sentence (D1.4), the full-master Note incl. call form (D2),
   the sample-tilt binding paragraph (D1.5), the score-semantics
   sentence (D4), the sweep prose and residual sentence (D4), the
   phase-list constraint sentence (D5), the shipped-map
   provenance framing (D3/D5), the parity paragraph **verbatim**
   and parameter-map table (D6), the `IndexEBSD` sentence (D1.11),
   the "What's next?" triple (D1.12 -- no pseudo-symmetry
   pointer, hedged cost claims).
3. Link hygiene: all internal links relative
   (`hough_indexing.ipynb#Calibrate-detector-sample-geometry`,
   the Hough tutorial's pre-indexing-maps anchor from the
   load-data section (requirements D1.3),
   `../reference/generated/kikuchipy.signals.EBSD.spherical_indexing.rst`,
   ..., `../user/related_projects.rst` -- replacing the prototype's
   absolute kikuchipy.org related-projects URL; also linked from
   the master-pattern section for the SHT database, D1.4);
   external links: Lenthe 2019 DOI, EMSphInx GitHub, PyEBSDIndex
   docs, orix docs. Anchor case: cross-notebook anchors use the
   heading's own case (`#Calibrate-detector-sample-geometry`) --
   this matches all eight committed cross-notebook links, and
   deliberately *not* the dev guide's lowercase example; recorded
   so review does not "correct" it.
4. Format: `black-jupyter --line-length=77` via
   `pre-commit run --files doc/tutorials/spherical_indexing.ipynb`
   **before executing** (measured on the prototype: three trivial
   reformat hunks -- an import-block blank line and two call
   unwraps -- so run the hook first, then execute, so stored
   outputs match the committed source). Hidden first cell,
   kernelspec `python3` metadata (the exemplars').
5. Execute once from the repo root:
   `uv run --with ipykernel jupyter nbconvert --to notebook
   --execute --ExecutePreprocessor.timeout=600 --inplace
   doc/tutorials/spherical_indexing.ipynb`
   (ipykernel is not in the project venv -- verified failure
   without `--with`; the notebook is cwd-independent: data via
   `kp.data`, files via `tempfile`), then strip the notebook-level
   `metadata.widgets` block per the requirements D7 recipe (no
   committed tutorial carries one; nbval does not compare it, so
   only this step keeps the diff churn out). Expected ~80 s +
   kernel start. Commit source + outputs together.

## 2. Docs wiring

1. `doc/tutorials/index.rst`: insert `spherical_indexing` in the
   Indexing `nbgallery` after `pattern_matching` (requirements
   D10).
2. `doc/tutorials/run_nbval.sh`: insert
   `"spherical_indexing.ipynb"\` after `"pc_fit_plane.ipynb"\`
   (alphabetical array order).
3. `doc/tutorials/tutorials_sanitize.cfg`: append regex8 + regex9
   exactly as frozen in requirements D7.
4. Re-run nbval locally against the committed notebook:
   `uv run --with nbval --with ipykernel pytest -v --nbval
   doc/tutorials/spherical_indexing.ipynb --nbval-sanitize-with
   doc/tutorials/tutorials_sanitize.cfg` -- must be **25/25** (25
   code cells after the color-key cell; the drafting evidence is
   24/24 on the 24-cell prototype; the full
   `./doc/tutorials/run_nbval.sh` sweep is weekly-CI's job and is
   not run locally -- pattern_matching alone is a long DI run).

## 3. `load_save_data.ipynb` -- the `.sht` section

1. Insert the two cells and the table row per requirements D8
   (after the EMsoft TKD subsection, before kikuchipy h5ebsd).
   Edit the JSON surgically with a **frozen format contract** (the
   review's addition -- the committed file is `nbformat: 4,
   nbformat_minor: 4`, `indent=1`, LF line endings, trailing
   newline, `source` as list-of-lines, and **no `id` key on any
   cell**, so an `nbformat`-library edit would attach `id`s and
   bump the minor version, rewriting the whole file): use
   `json.load` / `json.dump(nb, f, indent=1, ensure_ascii=False)`
   + trailing newline, add no `id` keys, keep `nbformat_minor: 4`;
   verify with `git diff --stat` (expected: one file, ~2 cells +
   1 table line of hunks). A full re-execution of
   `load_save_data.ipynb` is out of scope -- the notebook stores
   outputs from a prior session and only the new cell is executed,
   matching how the file mixes execution counts today (its stored
   counts are already non-monotonic: 10, 12, 13...; the new
   cell's stored output is produced by running just that cell
   against the in-package file **with cwd `doc/tutorials/`** --
   `data_path = Path("../../src/kikuchipy/data")` resolves only
   from there -- or committed output-less; decide at
   implementation by which renders correctly, recorded in
   validation).
2. Verify the anchor in the format-table row resolves in the built
   html (requirements D8's fallback note).
3. Constitution guard: commit only the new-section hunks; verify
   `git diff` for `load_save_data.ipynb` contains nothing else
   (the tech-stack "never swept into a commit" rule for the user's
   notebook edits -- the tree is clean at drafting, re-check at
   implementation).

## 4. CHANGELOG

1. Replace the `Unreleased -> Added` block with the frozen D9
   text (nine originals -> three consolidated entries + the
   tutorial entry = four bullets, PR #12 links). Keep the empty
   subsection headers.
2. Cross-check every PR URL against the fork
   (`jwestraadt/kikuchipy` pulls 5, 8, 9, 10, 12).

## 5. Builds and checks (the validation matrix's automated half)

1. `uv run sphinx-build -b html -d doc/_build/doctrees doc
   doc/_build/html` -- exit 0. NB this is a **multi-minute** gate:
   with no `-W`/`nitpicky` anywhere, warnings never fail, and a
   cold build live-executes every output-less notebook (incl.
   `hough_indexing`'s full run) and downloads data -- budget for
   it, and keep the doctree dir for step 2. Warning criterion per
   validation V4 (grep the log for the changed files -- there is
   no "no new warnings" baseline to diff against). Then eyeball
   `doc/_build/html/tutorials/spherical_indexing.html` (thumbnail
   present in the gallery, figures render, color key visible and
   not obscuring a panel, admonitions styled, parameter table
   renders, anchors work, cleanup cell NOT rendered) and the
   load_save_data page's new section.
2. `uv run sphinx-build -b linkcheck -d doc/_build/doctrees doc
   doc/_build/linkcheck` (shared doctrees -- otherwise the
   linkcheck builder re-executes the output-less notebooks). The
   gate is **not** exit 0 (the repo's 312 pyxem PR URLs in
   `CHANGELOG.rst` rate-limit and there is no `linkcheck_ignore`;
   a full clean run is unachievable today): per validation V6,
   grep `doc/_build/linkcheck/output.json` for the URLs this
   phase introduces and require status ok/working/redirect. The
   fork PR link `#12` cannot resolve until section 8 opens the
   PR -- that single check is sequenced after PR creation (V10).
3. `pre-commit run --files` over the changed doc files +
   `CHANGELOG.rst` -- but **excluding `doc/tutorials/run_nbval.sh`
   and never `specs/`**: probed, the GPL `licenseheaders` hook
   prepends a 19-line `##` header to the header-less
   `run_nbval.sh` (its extension map covers `.sh`; `.ipynb`,
   `.rst`, `.cfg` return "File not supported"), which would be an
   unrelated hunk in an upstream file that pre-commit.ci itself
   skips (`ci: skip: [licenseheaders]`) -- and no local git
   pre-commit hook is installed, so the explicit `--files` run is
   the only channel that would stamp it. `run_nbval.sh`'s change
   is validated by V3's diff inspection instead. black-jupyter
   must be a no-op on the committed notebooks (1.4's ordering).
4. Fresh-kernel re-execution check: run the nbconvert command of
   1.5 a second time on a copy and `pytest --nbval` the committed
   file (2.4) -- the two together are the "clean kernel +
   deterministic outputs" gate.

## 6. Adversarial review and fixes

1. Conventions reviewer refutes against the exemplars and the
   constitution: hidden first cell + thumbnail metadata shape
   (`hough_indexing`'s exactly), black-77 (hook idempotence),
   stored-outputs mode vs `nbsphinx_execute="auto"`, temp-dir
   file hygiene (nothing written into `doc/tutorials/`), relative
   links, alphabetical `NOTEBOOKS`, gallery placement, the
   CHANGELOG's Keep-a-Changelog shape and preserved PR links, the
   re-scope (no pseudo-symmetry anywhere -- grep the notebook for
   `pseudo`), and that no `src/` file changed.
2. Fidelity reviewer refutes the notebook's *claims* against the
   recorded measurements: every printed number in stored outputs
   matches `validation.md`'s tables; the parity paragraph's
   numbers against Phase 10 D6/D7 (0.34 deg / 2% / float
   precision / < 0.1 deg under the stretch decomposition; the
   paragraph is carried verbatim from D6) and its scoping against
   Phase 10 D10; the `(d-1)/2` prose against the Phase 2 warning;
   the score-semantics and residual sentences against the shipped
   `spherical_indexing` docstring Notes; the shipped-map labeling
   against the pyxem#578/#584 provenance; the memory/speed prose
   against the Phase 6/7 baselines; the parameter-map table
   against `EMSphInxNamelist.from_kwargs(...).to_string()`'s 19
   actual keys; the `.sht` table row's Write=No against
   `specification.yaml`.
3. Validation-matrix executor: runs `validation.md`'s matrix
   (execute clean-kernel, nbval, html render inspection,
   linkcheck, name/spell check) and probes the failure-mode list
   (each mode has a stated detection channel -- see
   `validation.md`; for a notebook these replace bug injection).
4. Fix, re-execute the notebook if any code cell changed (outputs
   must always match source -- never hand-edit an output), re-run
   2.4 and 5.1-5.3.

## 7. Open questions -- decided 2026-09-03 (autonomous mode), flagged for review

1. **The full Ni master is a Note, not the executed path** (D2):
   305.5 MB download weight for Binder/first-time readers, and
   the in-package 401-px master is the one every shipped number
   was measured on. (The drafting rationale "infeasible for the
   weekly nbval runner" was **wrong** -- `hybrid_indexing`
   downloads that exact file on the weekly runner today; grounds
   corrected in D2.) The roadmap box offered both alternatives;
   this resolves it to "cap bw <= 190" -> bw 188. Fallback: a
   gated "full-resolution" cell would need a skip mechanism no
   other tutorial has -- rejected.
2. **The validation comparison uses the shipped `s.xmap`, not a
   live DI run** (D3/D5): live DI is minutes against a ~20 s
   budget item, and -- established by the review -- there is no
   DI content to validate against anyway: the shipped map is
   Hough+refined (pyxem#578/#584), and the tutorial labels it so.
   Hough runs live (5 s); DI is cross-linked, not run.
3. **No PC-estimation section** (D3): the same dataset's PSO
   calibration is the Hough tutorial's opening act; duplicating
   it would double the notebook for zero new API coverage. The
   shipped calibration + `pc_average` + a link replaces it.
4. **Gallery position after `pattern_matching`** (D10) -- hough,
   DI, spherical, hybrid. Alternative (after `hough_indexing`)
   rejected: the intro contrasts spherical with DI, which then
   reads better already-introduced.
5. **The CHANGELOG consolidation rewrites shipped entries**
   (D9) -- the most user-visible call of the phase: 8 accreted
   entries -> 4 curated ones, all PR links kept, tutorial entry
   added. Fallback: append-only (just the tutorial entry) if
   review rejects rewriting history in `Unreleased`.
6. **`_ = dask.config.set(num_workers=8)` is a visible cell**
   (D4): teaches the memory model, pins the budget, and is what
   makes stored-output nbval machine-independent (chunk line +
   memory warning). Alternative (hide it in the hidden first
   cell) rejected: the hidden cell is markdown by convention, and
   silently altered user parallelism would be a docs anti-pattern.
7. **Two sanitize rules are added, not `nbval-ignore-output`
   tags** (D7): rules keep the outputs compared (the
   pattern_matching precedent uses exactly one ignore tag, for a
   genuinely non-comparable cell); regex8 is strictly required
   (PyOpenCL True on the weekly runner), regex9 is recorded as
   belt-and-braces.
8. **Outputs are produced by nbconvert, not a live JupyterLab
   session** (D7): measured to match the committed
   pattern_matching stored form and to pass nbval; it is also the
   only scriptable, reviewable recipe. The tech-stack amendment
   (0.2) writes it down.
9. **The Hough phase list is built from `mp.phase`** (D5): the
   measured `refine_orientation_spherical` lattice-parameter
   check (nm vs Å) turns this from style into correctness; the
   tutorial states the constraint in one sentence.
10. **PR number #12** is assumed (the fork's next; #11 was Phase
    10). If another PR lands first, the D9 links and the roadmap
    box are renumbered in the same commit that opens the PR.
11. **The bandwidth sweep reports an accuracy proxy, not scores**
    (D4, revision): the drafting sweep's cross-bandwidth score
    trend was the one comparison the shipped docstring calls
    invalid. The proxy (median/p99 misorientation of each coarse
    solution to the refined headline map, via CrystalMap slicing)
    was measured for this revision: 0.37/0.37/0.25 deg median,
    1.40/0.71/0.69 deg p99.
12. **Stored outputs stand against the dev guide's output-less
    recommendation via a quantified rebuttal** (D7, revision):
    est. 4-6 min on the ~2-vCPU RTD builder for this notebook
    alone (a third of the 15-min build budget), the dev guide's
    own "so the documentation doesn't take too long to build"
    paragraph, and the pattern_matching/hybrid_indexing
    precedents; the tech-stack amendment (0.2) records the
    threshold. Fallback if review rejects: ship output-less like
    `hough_indexing` and drop regex8/regex9 + the worker-pin
    nbval rationale -- not taken.
13. **V8 excludes `run_nbval.sh`** (5.3, revision): probed --
    the GPL licenseheaders hook would prepend an 18+1-line header
    to it; pre-commit.ci skips the hook and no local git hook is
    installed, so exclusion (plus V3's diff inspection) is the
    clean resolution rather than committing an unrelated header
    hunk.

## 8. Commit and PR

1. Signed commits in gate order: spec + section-0 amendments
   (roadmap box rewrite, tech-stack amendment, research addendum);
   implementation (notebook + wiring + load_save_data + CHANGELOG
   in one commit -- they are one user-facing change); review
   fixes. Tick roadmap boxes as gates complete; push; PR **#12**
   into fork `develop` with the template. Licence statement: no
   new source files; the notebook is documentation for
   GPL-covered functionality and carries no licence header (the
   exemplars' convention -- notebooks are not stamped by
   `licenseheaders`, whose extension whitelist excludes `.ipynb`).
2. PR description records: the measured ~80 s wall clock and its
   <= 3 min budget, the nbval evidence, the D9 consolidation
   rationale, and the Phase 10 D10 parity-scoping compliance.
