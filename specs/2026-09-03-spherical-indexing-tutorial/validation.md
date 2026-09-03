# Phase 11 -- `spherical-indexing-tutorial`: validation

Documentation-only phase: the failing-tests gate is skipped
(roadmap rule); its place is taken by the validation matrix below,
which the adversarial review executes (plan 6.3). Bug injection has
limited meaning for a notebook -- the failure-mode list replaces it:
each mode names its detection channel, and the review probes a
sample of them.

## Validation matrix (all must pass before the PR)

| # | check | command / procedure | pass criterion |
|---|---|---|---|
| V1 | clean-kernel execution | `uv run --with ipykernel jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace doc/tutorials/spherical_indexing.ipynb` from the repo root, then the `metadata.widgets` strip (requirements D7) | exit 0, no error outputs, no unexpected stderr (the only stderr-ish outputs are tqdm placeholder frames); the cleanup cell's stored output is `False` (`temp_dir.exists()` -- the temp dir was removed); no top-level `metadata.widgets` key remains; wall clock recorded and <= 3 min on a drafting-class (many-core) machine -- on slower machines record and compare against the D11 estimate instead |
| V2 | deterministic re-execution (nbval) | `uv run --with nbval --with ipykernel pytest -v --nbval doc/tutorials/spherical_indexing.ipynb --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg` | 25/25 code cells pass |
| V3 | nbval wiring | inspect `run_nbval.sh` (entry after `pc_fit_plane.ipynb`; no licenseheaders header -- see V8) and `tutorials_sanitize.cfg` (regex8/regex9 exactly as requirements D7) | diff matches the frozen text |
| V4 | html build | `uv run sphinx-build -b html -d doc/_build/doctrees doc doc/_build/html` (multi-minute: live-executes the output-less notebooks; keep the doctrees for V6) | exit 0; `grep -iE "spherical_indexing|load_save_data|tutorials/index|CHANGELOG" <build log> \| grep -i warning` returns nothing (there is no repo-wide "no new warnings" baseline -- `nitpicky` is off and no `-W` anywhere, so the criterion is scoped to the changed files) |
| V5 | html render inspection | open `doc/_build/html/tutorials/spherical_indexing.html` and `.../index.html` and `.../load_save_data.html` | gallery shows the new card with the IPF thumbnail + tooltip; 7 figures render; the IPF color key renders and does not obscure a map panel; both Note admonitions styled; parameter table renders; the hidden cells (first cell, cleanup cell) do NOT render; internal links resolve (spot-click the reference links, the hough anchor links incl. the pre-indexing-maps one, the related-projects link); the load_save_data `.sht` section renders and its table-row anchor resolves (requirements D8 fallback if not) |
| V6 | linkcheck | `uv run sphinx-build -b linkcheck -d doc/_build/doctrees doc doc/_build/linkcheck` (shared doctrees), then grep `doc/_build/linkcheck/output.json` for each URL this phase introduces | every phase-introduced URL (doi.org Lenthe 2019, EMSphInx GitHub + SHTdatabase, PyEBSDIndex docs, orix docs) has status ok/working/redirect in `output.json`; overall exit code is NOT the criterion (312 pre-existing pyxem PR URLs rate-limit; recorded, not fixed); the fork PR link #12 is checked in V10 after the PR exists |
| V7 | name/spell pass | manual read of all markdown: EMSphInx (capital S, capital I), PyEBSDIndex, EMsoft, kikuchipy (lower k), HyperSpy, Lenthe, De Graef, Bunge, Legendre, Lambert, Radon/Hough; degree signs and units consistent with the exemplars | no misspelling; no "pseudo" match in the notebook (`grep -i pseudo` returns nothing -- the re-scope guard); no "dictionary" label on the shipped map (`grep -i dictionary` hits only the DI cross-references, never the shipped-map panel/print -- the D3 guard) |
| V8 | style hooks | `uv run pre-commit run --files doc/tutorials/spherical_indexing.ipynb doc/tutorials/load_save_data.ipynb doc/tutorials/index.rst doc/tutorials/tutorials_sanitize.cfg CHANGELOG.rst` -- **`run_nbval.sh` deliberately excluded**: the GPL licenseheaders hook prepends a 19-line header to it (probed; pre-commit.ci skips the hook, no local git hook installed; plan 5.3 / 7.13) | all hooks pass with **no modifications** (black-jupyter ran before execution, plan 1.4) |
| V9 | file hygiene | `git status` after V1; `git diff --stat` for `load_save_data.ipynb` | nothing untracked in `doc/tutorials/` (NB temp files go to `%TEMP%`, not the repo -- the cleanup check is V1's stored `False`, not this row); the `load_save_data.ipynb` diff contains only the new-section hunks (plan 3.1's format contract: ~2 cells + 1 table line, no `id` keys, `nbformat_minor` still 4) |
| V10 | CHANGELOG | render check of the Unreleased block (V4 covers rst validity); after the PR is opened (plan 8), PR links 5/8/9/10/12 resolve on the fork | matches the frozen D9 text; all five PR URLs resolve (sequenced after PR creation -- #12 cannot exist before it) |
| V11 | size/scale | `ls -l doc/tutorials/spherical_indexing.ipynb` | ~1 MB (family: hybrid 1.16 MB, pattern_matching 2.0 MB); way below the 6.3 MB workshop-notebook precedent |

## Failure modes (the notebook analogue of bug injection)

Each mode: what breaks -> which matrix row catches it.

1. **A code cell edited after execution** (source/output mismatch)
   -> V2 fails on that cell (nbval compares stored vs fresh).
2. **Machine-dependent output stored** (chunk line without the
   worker pin, memory warning on a many-core machine, PyOpenCL
   bool, a memory-address repr) -> V2 on any other machine; the
   drafting evidence already caught the `dask.config.set` repr
   this way, and the D4 pin + regex8/regex9 close the known
   channels. Residual: a printed float landing on a rounding
   boundary under cross-platform fastmath drift (IQ ladder
   5.2e-5, scores ~1e-5) -- fix ladder: re-execute on the failing
   machine; if it recurs, drop one decimal on that print and
   re-execute (never hand-edit the output).
3. **Forgotten `verbose=0` on the sweep/Hough cells** -> V2 (extra
   progress bars still pass via CR-collapse, but the Hough info
   block's chunk line and PyOpenCL line must be regex-covered --
   V3 checks the rules exist) + V5 (page noise).
4. **Temp files leak** (cleanup cell dropped, reordered or
   broken) -> V1: the cleanup cell ends with `temp_dir.exists()`
   whose stored output must be `False` (and V2 re-compares it on
   every nbval run). NB V9 cannot catch this -- the files go to
   `%TEMP%`, outside the repo (`.ipynb_checkpoints` is gitignored
   besides), which is why the notebook itself carries the check;
   a *dropped* cell is caught by the plan-6.1 inventory diff.
5. **Thumbnail metadata malformed or on the wrong cell** -> V5
   (gallery card falls back to a generic image or wrong figure).
6. **A reference link typo** (`.rst` target misspelled) -> V4
   warning + V5 spot-click; external typo -> V6.
7. **The notebook not added to `index.rst`** -> V4 warning
   ("document isn't included in any toctree") + V5.
8. **run_nbval entry misspelled** -> V3; (the weekly job would
   also fail loudly on a missing file).
9. **Sweep cell accidentally refining** (`refine=False` dropped)
   -> V1 wall clock grows ~1.3x and the stored speeds drop ~30%
   vs the recorded table -> fidelity review (plan 6.2) catches
   the mismatch against "Recorded results".
10. **Hough phase built with Å lattice** -> V1 fails loudly:
    `refine_orientation_spherical` raises the measured
    lattice-parameter `ValueError` (the strongest single guard in
    the notebook's own execution path).
11. **`sample_tilt` mismatch introduced** (e.g. detector edited to
    65 deg) -> V1 fails loudly (the Phase 6 binding guard raises).
12. **Parity paragraph over-claims** (numbers not the suite's, or
    unscoped per Phase 10 D10) -> plan 6.2 fidelity review against
    Phase 10 D6/D7/D10 -- reviewed-only, no automated channel;
    recorded as such.
13. **Pseudo-symmetry mention sneaks in** -> V7's grep guard.
14. **CHANGELOG entry lost in the consolidation** (a capability no
    longer named) -> plan 6.1 conventions review checks the three
    consolidated entries cover all **nine** originals' API names
    (three `.sht`, three interop, indexing, bandwidth helper,
    refinement); V10 link check.
15. **load_save_data regression** (unrelated stored cells touched
    by the JSON edit) -> V9's hunk check + V5's render of the
    whole page.
16. **black-jupyter reformats after execution** (hook/output
    drift) -> V8's "no modifications" criterion.

## Recorded results (drafting, 2026-09-03, this machine)

Machine: Windows 11, 20 logical cores, warm numba caches, local
data cache warm (`nickel_ebsd_large` present). Probes and the
prototype builder in the session scratchpad: `p11_time.py`,
`p11_time2.py`, `build_nb.py`, `spherical_indexing.ipynb` (source,
48 cells), `spherical_indexing_executed.ipynb` (1,009,925-
1,013,805 B across runs).

### End-to-end

- `jupyter nbconvert --to notebook --execute`: **88.0 s** then
  **79.5 s** wall for the full notebook (incl. ~7 s uv/kernel
  startup); kernel time ~72-81 s. Budget <= 3 min on 8 threads:
  **met, 2.2x margin on this 20-core machine** (machine-scoped per
  the revised Scope -- ~42 s of the kernel time scales with real
  cores) -- the pre-agreed fallbacks (bw 53, `inav[:40, :30]`,
  `refine=False`) were not needed beyond the sweep's own design
  and stay live for slower runners.
- nbval on the executed prototype with the repo's existing
  sanitize file: **23 passed, 1 failed** in 62 s -- the failure
  being `dask.config.set(num_workers=8)`'s bare repr
  (`<dask.config.set at 0x17e6...>` vs `0x260a...`), fixed by the
  `_ =` assignment (requirements D4). The review's independent
  re-run of the fixed prototype: **24 passed in 65 s** (V2
  re-proves on the branch at 25 cells).

### Per-section timings (probe measurements, 8 pinned workers)

| section | measured |
|---|---|
| imports | 1.8 s |
| load `nickel_ebsd_large` + static+dynamic background | 3.3 s |
| `get_spherical_harmonics(bandwidth=188)` | **0.08 s** |
| `power_spectrum()` + `to_master_pattern()` (prototype default (2, 377, 377); the committed notebook passes `dim=401` -> (2, 401, 401), re-measured 0.04 s) | 0.04 s |
| `SphericalBackProjector` + one `unproject` | 0.06 s |
| `spherical_indexing` 4125 patterns, bw 68, refined | **20.0-20.3 s** (204-230 patterns/s; 18.3 s / 230 pat/s at 20 unpinned workers) |
| bw sweep coarse `inav[:40, :30]` = 1200 patterns | 53: 2.9-3.0 s (401.4 pat/s), 68: 5.4-5.5 s (222.0), 88: 11.5-12.6 s (104.4) |
| Hough indexing (PyEBSDIndex, CPU) | 4.7-5.2 s (913-924 patterns/s) |
| `refine_orientation_spherical` on the Hough map | 3.3-3.5 s (1167 patterns/s) |
| interop (save `.sht` 18,388 B + `kp.load` + repack 14.86 MB + namelist 110 lines) | < 0.1 s |
| 7 figures + IPF color mapping | ~2-4 s total |
| full-map refined at bw 88 (probe only; NOT in the notebook) | 45.0 s -- the sweep stays coarse-on-subset |

### Values printed in the stored outputs (fidelity anchors)

(Post-revision anchors; entries marked *(prototype)* are the
drafting prototype's stored values, retained where the redesign
did not change the printed line. The drafting version of this
section recorded two numbers wrongly -- the chunk line "375/11"
and "Refining 4125" -- corrected below and re-measured in the
dated section.)

- Sweep *(redesigned, D4)*: median to refined **0.37 / 0.37 /
  0.25 deg**, p99 **1.40 / 0.71 / 0.69 deg** at bw 53 / 68 / 88
  (re-measured; mean scores 0.521 / 0.616 / 0.660 are no longer
  printed).
- Spherical vs Hough (masked): median **0.23 deg**, p99 **0.75
  deg**; vs the shipped Hough+refined xmap: median **0.43 deg**,
  p99 **1.53 deg** (consistent with the Phase 7 165-pt refined
  0.456 vs the same stored map).
- Refined-Hough vs spherical median: **0.042 deg** (p99 0.128;
  masked p99 0.127); refined map props `scores`/`iq`, mean score
  0.6300 *(prototype)*.
- Back-projection IQ of `s.inav[0, 0]`: 0.349 (printed at 3
  decimals -- the drift ladder) *(prototype)*.
- Cleanup cell output: `False` (`temp_dir.exists()`).
- Info message (stored): "Indexing 4125 pattern(s) in **275**
  chunk(s) of up to **15** pattern(s)" (the drafting record said
  375/11 -- those are the *unpinned* 20-worker probe numbers, a
  transcription error caught by the review; the prototype's
  stored output and two re-runs all say 275/15), "Estimated
  memory per worker: 54 MB", "Refinement: Newton (on)";
  "Refining **4124** orientation(s) in 275 chunk(s) of up to 15
  pattern(s)" (4124, not 4125 -- one Hough point is
  `not_indexed`); Hough info: "PyOpenCL: False", "in 8
  chunk(s)".
- `MasterPatternHarmonics` repr: `MasterPatternHarmonics: bw =
  188, ni (m-3m), 20.1 keV, 70.0 deg`.
- bw-384 warning text verified ("exceeds 200, the largest
  harmonic degree a square Lambert master pattern of side length
  401 carries..."), fired only in the probe -- the notebook stays
  at 188 and stores no warning.

### Repo facts verified at drafting

- `load_save_data.ipynb`: zero `sht` matches -> the D8 section is
  in scope. `data_path = Path("../../src/kikuchipy/data")` (cell
  4); cleanup precedent (cells 37/93/109); format table in cell
  46; plugin `emsphinx_master_pattern` `writes: False`.
- `hough_indexing.ipynb` committed with **no stored outputs**
  (nbsphinx auto-executes it); `pattern_matching.ipynb` with 40
  output cells incl. tqdm `" 0%|"` placeholder frames and
  final-line dask ProgressBar streams; `hybrid_indexing.ipynb` 17
  output cells; thumbnail metadata shape confirmed on
  `hough_indexing` cell 51.
- `nbsphinx_execute = "auto"`, `nbsphinx_allow_errors = True`
  (`doc/conf.py:168-169`); no docs build in `tests.yml`; nbval
  runs only in `weekly.yml` (`[doc,tests,all]` + nbval +
  pyopencl, 30-min timeout).
- Existing sanitize rules regex1-regex7 cover: dask "Completed |
  TIME", `patterns/s`, `comparisons/s`, tqdm 100% lines, figure
  sizes, "Refining N", "Matching M/N".
- `_n_workers()` honours `dask.config.set(num_workers=...)`
  (`_indexer.py:393-406`) -- the D4 pin's mechanism.
- Unpinned on 20 workers, bw 68 raises the "2.00 GiB > 2 GiB"
  memory `UserWarning` (measured) -- the pin suppresses it.
- `refine_orientation_spherical` refuses an Å-lattice Hough phase
  against the nm master phase (measured `ValueError`) -- the D5
  phase-list rule.
- pyebsdindex 0.3.10.1 in the venv (the `!= 0.3.10` pin allows
  it); Hough runs CPU (`PyOpenCL: False`).
- `ebsd_master_pattern("ni")` = `ni_mc_mp_20kv.h5`, **305.5 MB**,
  cached locally under `0.9.0` and `develop` -- the D2 Note-only
  call.
- ipykernel absent from the project venv (`No such kernel named
  python3` without `--with ipykernel`); black-jupyter at 77 on the
  prototype: three trivial reformat hunks (plan 1.4 orders the
  hook before execution).
- Executed prototype size 1,009,925 B / 48 cells / 7 figures (the
  post-revision notebook adds a color-key cell and markdown: ~50
  cells, 25 code).

## Re-measurements (2026-09-03, spec revision after adversarial drafting review)

Probe: `p11_revise_probe.py` (+ two follow-up snippets) in the
session scratchpad; same machine and pin (20 logical cores,
Windows, 8 dask workers). Every disputed number was re-executed
rather than argued:

- **Info messages** (the drafting record's two errors, both
  confirmed against the executed prototype's stored output AND a
  fresh run): headline indexing prints "Indexing 4125 pattern(s)
  in **275** chunk(s) of up to **15** pattern(s)", "Estimated
  memory per worker: 54 MB"; refinement prints "Refining **4124**
  orientation(s) in 275 chunk(s) of up to 15 pattern(s)". The
  drafting "375/11" matches 20 unpinned workers (375 x 11 = 4125,
  as 275 x 15 = 4125) -- a probe-transcription error.
- **`fast_bandwidths(150, 210)`** = `[158 163 172 176 182 188 193
  203]` -- 193 is fast and < 200, disproving the prototype's
  "largest bandwidth below this limit" claim (requirements D2).
- **Sweep redesign numbers** (coarse on `s.inav[:40, :30]` vs the
  refined headline map's `xmap[:30, :40]` slice): median
  **0.37 / 0.37 / 0.25 deg**, p99 **1.40 / 0.71 / 0.69 deg**, at
  **401 / 228 / 103 patterns/s** for bw 53 / 68 / 88 (Euler side
  lengths 105 / 135 / 175 -> half-cells 1.7 / 1.3 / 1.0 deg).
  Control: a refined run on the subset itself reproduces the
  slice's rotations (coarse-vs-refined-subset median
  0.3712043957... = coarse-vs-slice median, bitwise). NB a manual
  `Orientation.reshape(*xmap.shape)[:30, :40].flatten()` route
  gives garbage (~41.6 deg median) -- CrystalMap slicing is the
  correct extraction and the one the notebook teaches.
- **Misorientation stats** (mask = `phase_id != -1`, 1
  `not_indexed` Hough point at flat index 3334, spurious 47.3 deg
  vs its identity rotation): spherical vs Hough (4124 indexed):
  median **0.229**, p99 **0.75**, max **59.99**, 25 points > 2
  deg; spherical vs shipped map (4125): median **0.429**, p99
  **1.53**, max **59.98**, 33 points > 2 deg; refined-Hough vs
  spherical: median **0.042**, p99 0.128 (masked 0.127).
- **Timings** (re-run): headline indexing 20.2 s, Hough 4.4 s,
  refinement 3.5 s -- inside the drafting ranges.
- **`to_master_pattern(dim=401)`**: shape (2, 401, 401), 0.04 s
  -- the like-for-like figure costs nothing extra.
- **Namelist** (`from_kwargs(...).to_string()`): 19 keys
  `patfile patdset masterfile patdims circmask gausbckg nregions
  delta pctr vendor thetac scandims roimask bw normed refine
  nthread batchsize datafile` -- no `ipath`, no `scanstep`;
  `delta = 500` (= 30000/60, not `det.px_size`), `vendor =
  'Bruker'`, `scandims = 75, 55, 1.5, 1.5`.
- **Shipped-data provenance** (requirements D3): `s.xmap.prop` =
  `['scores', 'z']` (no `simulation_indices`); shipped detector
  per-point PC std = (0.0028, 0.0027, 0.0020); upstream CHANGELOG
  0.8.0 names Hough + refinement (pyxem#578/#584).
- **Master metadata** (requirements D1.4): `MCCLNameList.EkeV` =
  **20.1**, `Ehistmin` = 20.0 -- the harmonics repr's "20.1 keV"
  is the simulation's own beam energy, not a rounding artefact.
- **licenseheaders probe** (V8): `pre-commit run licenseheaders
  --files doc/tutorials/run_nbval.sh` modified the file (19-line
  `##` GPL header prepended after the shebang); working tree
  restored with `git checkout --` immediately. No local git
  pre-commit hook is installed (`.git/hooks/` has samples only).
- **`metadata.widgets`**: present in the executed prototype
  (17,692 B, `application/vnd.jupyter.widget-state+json`); absent
  from all three committed exemplars -> the D7 strip step.

## Definition of done

- [ ] Spec commit: this folder + the plan section-0 amendments
      (roadmap Phase 11 box rewrite + gate-sentence amendment,
      tech-stack tutorial-rules amendment, research addendum) on
      the branch.
- [ ] `doc/tutorials/spherical_indexing.ipynb` committed with
      stored outputs per plan 1 (black-77 first, executed once,
      `metadata.widgets` stripped, V1/V2 green).
- [ ] `index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg`
      changes committed per plan 2 (V3).
- [ ] `load_save_data.ipynb` `.sht` section + table row committed
      per plan 3 (V5, V9).
- [ ] `CHANGELOG.rst` Unreleased/Added block replaced by the D9
      text (V10).
- [ ] Validation matrix V1-V11 all green; results (wall clocks,
      nbval counts, build exit codes) recorded in this file's
      "Recorded results" as a dated implementation section.
- [ ] Adversarial review (plan 6) run; findings fixed;
      re-validated.
- [ ] Roadmap Phase 11 boxes ticked as gates complete; signed
      commits pushed; PR #12 opened into fork `develop`.
