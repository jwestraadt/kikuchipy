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

## Recorded results (implementation, 2026-09-03, this machine)

Same machine as drafting (Windows 11, 20 logical cores, 8 pinned
dask workers, warm caches), `.venv` Python 3.13.12, pyebsdindex
0.3.10.1. Everything below was executed, not estimated.

### Build and gate wall clocks

- **V1** `jupyter nbconvert --to notebook --execute
  --ExecutePreprocessor.timeout=600 --inplace`: exit 0, **77.8 s**
  wall (incl. uv/kernel startup); a second clean-kernel run on a
  scratch copy: exit 0, **73.1 s**. No error outputs, no stderr
  streams, 7 figures, cleanup cell stored output `False`. Both
  runs printed **bit-identical numbers** in every compared line.
  The <= 3 min budget is met with 2.3x margin on this machine.
- **V2** `pytest --nbval ... --nbval-sanitize-with
  doc/tutorials/tutorials_sanitize.cfg`: **25 passed** (52.8-55.0 s
  across three runs -- after the first execute, after the markdown
  polish, and after the `metadata.execution` strip).
- **V4** html build: **exit 0**, `build succeeded`, 2 min 40 s cold
  / 14.1 s incremental (see deviation 2 on `nbsphinx_execute`).
- **V6** linkcheck (shared doctrees): finished with problems, as
  the spec predicts; 963 rows, 461 working / 65 redirected / 399
  unchecked / 38 broken.
- **V8** `pre-commit run --files` over the five notebook/rst/cfg
  files (`run_nbval.sh` excluded): all hooks **Passed, no
  modifications**, on three separate invocations.

### Per-cell execution timing (from `metadata.execution`, run 1)

Captured before the block was stripped (deviation 3). Sum of cell
busy times **53.0 s**; the remaining ~25 s of the 77.8 s wall is
uv resolution, kernel start and inter-cell overhead.

| cell | s | cell | s |
|---|---|---|---|
| C1 imports | 2.13 | C13 export (commented) | 0.00 |
| C2 load + backgrounds | 3.39 | C14 `fast_bandwidths` | 0.02 |
| C3 master pattern | 0.02 | **C15 bw sweep** | **17.98** |
| C4 harmonics bw 188 | 0.07 | C16 phase list | 0.01 |
| C5 power spectrum | 0.19 | C17 Hough indexing | 4.73 |
| C6 round trip dim=401 | 0.26 | C18 color key | 0.01 |
| C7 detector | 0.00 | C19 three-way IPF | 0.27 |
| C8 back-projection | 0.17 | C20 misorientations | 0.13 |
| C9 worker pin | 0.00 | C21 refine Hough | 2.98 |
| **C10 indexing** | **20.33** | C22 histogram | 0.11 |
| C11 `xmap` repr | 0.01 | C23 `.sht` round trip | 0.04 |
| C12 score/IQ maps | 0.12 | C24 patterns + namelist | 0.01 |
| | | C25 cleanup | 0.01 |

### Stored numbers (all frozen anchors re-confirmed)

- Info: "Indexing 4125 pattern(s) in **275** chunk(s) of up to
  **15** pattern(s)", "Estimated memory per worker: 54 MB";
  refinement: "Refining **4124** orientation(s) in 275 chunk(s) of
  up to 15 pattern(s)"; Hough: "PyOpenCL: False", "in 8 chunk(s)".
- Harmonics repr `MasterPatternHarmonics: bw = 188, ni (m-3m),
  20.1 keV, 70.0 deg`; back-projection `Image quality: 0.349`.
- Sweep: median **0.37 / 0.37 / 0.25 deg**, p99 **1.40 / 0.71 /
  0.69 deg** at bw 53 / 68 / 88 -- identical to the spec.
- Misorientation prints: to Hough (masked) median **0.23**, p99
  **0.75**; to shipped (refined) median **0.43**, p99 **1.53** deg.
- `Median after refinement: 0.042 deg`; cleanup `False`.
- Speeds (sanitized by regex2, quoted nowhere in prose) ran
  *faster* than the drafting table: indexing 206.3 patterns/s
  (recorded 204-230), sweep 459.9 / 246.6 / 114.4 (recorded 401 /
  228 / 103), Hough 909.1 (913-924), refinement 1523.9 (1167).
- **V11** size **1,004,294 B** (0.96 MiB), 50 cells (25 markdown,
  25 code), 7 figures -- in family with hybrid 1.16 MB.

### V3 / V5 / V7 / V9 / V10

- **V3**: `run_nbval.sh` gains one line after `pc_fit_plane.ipynb`
  (trailing `\` kept, no licenseheaders header);
  `tutorials_sanitize.cfg` gains regex8 + regex9 byte-for-byte as
  frozen in D7; `index.rst` gains one line between
  `pattern_matching` and `hybrid_indexing`.
- **V5**: the gallery card renders with thumbnail
  `tutorials_spherical_indexing_36_0.png` (the three-way IPF cell)
  and tooltip "Spherical harmonic indexing of EBSD patterns", in
  the Indexing gallery between Pattern matching and Hybrid
  indexing; 7 figures, 2 Note admonitions, 1 parameter table; the
  IPF color key sits to the right of the three panels and obscures
  none of them (inspected in the rendered PNG); **neither hidden
  cell renders** (0 matches for the hidden-cell text, the cleanup
  comment and `os.rmdir`); every internal link resolves
  (`hough_indexing.html#Calibrate-detector-sample-geometry`,
  `#Pre-indexing-maps`, `../user/related_projects.html`, the 12
  `../reference/generated/*.html` targets, the two same-page
  anchors). The `load_save_data` `.sht` section renders with its
  stored output and the **parenthesised table anchor resolves**
  (`id="EMSphInx-spherical-harmonics-master-pattern-(.sht)"`), so
  the D8 fallback was not needed.
- **V7**: `grep -i pseudo` on the notebook = **0**; the four
  "dictionary" hits are all DI cross-references (intro, score
  semantics, What's next) -- none labels the shipped map. EMSphInx
  (22), PyEBSDIndex (4), EMsoft (4), kikuchipy (33), Lenthe (3),
  Legendre, Lambert, Newton, Euler, Hough all correctly cased;
  "Dask" in prose matches the repo's 11 existing uses.
- **V9**: `git status` clean apart from the five modified files
  and the new notebook; nothing untracked in `doc/tutorials/`.
  The `load_save_data.ipynb` diff is **+33 / -0**: one table line
  plus two cells, no `id` keys added, `nbformat_minor` still 4.
- **V10**: the rendered `Unreleased -> Added` block is exactly the
  frozen D9 text -- four bullets, PR links 5/8/9/10/12, empty
  Fixed/Changed/Removed/Deprecated headers kept. Failure mode 14
  checked by hand: the three consolidated entries name all nine
  originals' APIs (3 `.sht`, 3 interop, indexing + bandwidth
  helper + refinement). The #12 link is V10's post-PR check.

### Deviations, with measurements

1. **V4's scoped grep is not empty**, and the criterion is met
   only after classifying the hits. `sphinx-build` exits 0. The
   grep returns (a) **7 sphinx-codeautolink "Could not match
   transformation of `X`" warnings** on `spherical_indexing.rst`
   -- a repo-wide class: **129 such warnings across 35 files** in
   the same build, 8 on `adaptive_histogram_equalization`, 7 on
   `pc_orientation_dependence`, 6 on `load_save_data`, 5 on
   `hough_indexing`; they are emitted for every notebook's import
   block and ours is in family; and (b) one **pre-existing**
   `load_save_data.ipynb: "nbsphinx-thumbnail" in cell 102:
   Unsupported output type in output 0: "stream"` -- its thumbnail
   cell was index **100** at `HEAD` with the same single `stream`
   output, so all this phase changed is the reported index (+2
   from the two inserted cells). No warning is attributable to
   this phase's content, and there is no "isn't included in any
   toctree" warning (FM7 clear).
2. **The html and linkcheck builds ran with
   `-D nbsphinx_execute=never`** instead of validation.md's
   default configuration. Grounds: both files under test store
   outputs, so `nbsphinx_execute = "auto"` skips them too and the
   rendered pages are identical; `never` avoids live-executing the
   ten unrelated output-less notebooks (`hough_indexing`,
   `pattern_processing`, `feature_maps`,
   `geometrical_ebsd_simulations`,
   `kinematical_ebsd_simulations`, `multivariate_analysis`,
   `pc_calibration_moving_screen_technique`, `reference_frames`,
   `virtual_backscatter_electron_imaging`,
   `visualizing_patterns`). Side effect, recorded so it is not
   mistaken for a regression: under `never` those notebooks have
   no outputs, so nbsphinx emits 11 `"nbsphinx-thumbnail" ... No
   outputs` warnings and their gallery cards fall back to the
   broken-thumbnail SVG. Neither affects the two changed pages.
3. **nbconvert also writes a per-cell `metadata.execution`
   block**, which D7's recipe does not mention. Measured: 25 cells
   carried one (~6.5 kB), and **no committed tutorial carries one**
   (`metadata.execution` count over `doc/tutorials/*.ipynb` before
   this phase: zero). It is the same per-run churn as
   `metadata.widgets` (17,693 B, stripped per the recipe), so it
   was stripped too; the remaining cell metadata keys are exactly
   the exemplars' `nbsphinx`, `nbsphinx-thumbnail`, `tags`. nbval
   after the strip: 25 passed.
4. **`load_save_data`'s new code cell stores its output with
   `execution_count: null`** (rendered `[ ]:`). Stored output was
   chosen because `nbsphinx_execute = "auto"` does not execute a
   notebook that already has outputs, so an output-less cell would
   render as an input with no result beside 42 cells that all show
   one. No execution number was invented: the file's counts run
   3-53 in one sequence, so any renumbering would either duplicate
   a number or touch the 49 following cells and break V9's hunk
   budget, and two cells (48, 110) already carry `null`.
5. **The D8 code snippet is one line, not two**:
   `mp_sht = kp.load(data_path / "emsphinx/ni_small_20kv_bw384.sht")`
   is 63 characters, so `black-jupyter --line-length=77` unwraps
   the spec's wrapped form, and V8 requires the hook to make no
   modification.
6. **The misorientation print carries labels** ("To Hough
   (masked):" / "To shipped (refined):"). D5 froze the statistics
   (median + p99, two decimals, `not_indexed`-masked for the Hough
   pair), not the label text.
7. **The refinement median is printed from the masked array.**
   Plan 1.21 froze `np.median(angles_ref)` while D5 says "the
   print uses the same mask as above"; both are honoured by
   defining `angles_ref = ori_ref.angle_with(ori_sph,
   degrees=True)[mask]`. The value is 0.042 deg either way.
8. **The IPF color key is placed beside the panels, not over
   them.** Plan 1.18 left the inset coordinates to implementation
   against the rendered page (V5). Measured shape:
   `fig.tight_layout(rect=[0, 0, 0.87, 1])` then
   `fig.add_axes([0.87, 0.06, 0.13, 0.8], projection="ipf",
   symmetry=symmetry.Oh)` -- still the `hough_indexing` cell-51
   `add_axes` + `plot_ipf_color_key` + transparent-patch shape,
   but in reserved figure space, which makes "does not obscure a
   map panel" true by construction rather than by luck. The panel
   titles use `ax.set_title(title, fontsize=13)`: `Axes.set()`
   rejects `fontsize`, and "Hough + refinement (shipped)"
   overflows a 4.3 in panel at the notebook's `font.size` of 15.
9. **The notebook is `nbformat_minor` 5 with cell ids**
   (`cell-00` ... `cell-49`, deterministic), matching
   `hough_indexing` -- the structural exemplar -- and the drafting
   prototype. The repo is split: 7 tutorials use minor 5 with ids,
   11 use minor 4 without (including `pattern_matching` and
   `hybrid_indexing`).
10. **Two markdown-only edits were made after execution**: "so
    this call is EMSphInx's own namelist defaults" -> "so this
    call *reproduces* EMSphInx's own namelist defaults", and
    `"scores"` gained its gloss ("the correlation at the best
    orientation") in the two-property sentence. Markdown is
    neither executed nor compared by nbval, so stored outputs
    still match their sources; re-proved by a full nbval run
    afterwards (25 passed) and a rebuilt html page.
11. **Linkcheck: `spherical_indexing.ipynb` has one `broken`
    row**, `https://github.com/pyxem/kikuchipy/blob/develop/doc/
    tutorialsspherical_indexing.ipynb` (404). It is the
    `nbsphinx_prolog`'s "view it on Github" link, whose
    `env.doc2path(env.docname, base=None)` yields a Windows path
    (`tutorials\name`) whose backslash is dropped in the URL --
    **every one of the 19 tutorial notebooks carries exactly this
    row in the same build**, untouched ones included, and it
    cannot occur on the Linux RTD builder. All six
    phase-introduced URLs are ok: Lenthe 2019 DOI `redirected`,
    EMSphInx GitHub `working`, SHT database `working`, PyEBSDIndex
    docs `redirected`, both orix targets `working`.

## Recorded results (adversarial review fixes, 2026-09-03, this machine)

Same machine, venv and pin as the implementation section. Every
finding of the content review and the conventions review was
re-measured here before it was applied; the probe is
`p11_fix_probe.py` (session scratchpad) plus the follow-up
snippets quoted below. The notebook was re-executed in place after
the edits, so stored outputs match their sources.

### Findings applied, with the measurement that carried them

1. **Power spectrum (cells 9-10) -- applied.** `power_spectrum()`
   returns 188 values, `P[l]` for `l = 0 ... 187`. The **odd**
   degrees are numerically zero (median 1.37e-30, max 3.19e-30):
   the master is centrosymmetric. The **even** degrees do not fall
   off monotonically: `P[0] = 9.145` (the constant term, **73.3 %**
   of the total power), `P[2] = 1.80e-05`, rising to a maximum of
   **2.06e-01 at l = 44**, then decaying to 3.47e-03 at l = 186;
   even-band means 9.26e-3 / 8.88e-2 / 6.86e-2 / 1.76e-2 / 6.83e-3
   over [2,20) / [20,50) / [50,90) / [90,130) / [130,188). Plotting
   all 188 degrees forces a **31.5-decade** y-axis (the odd spikes),
   which is why the old PNG was a picket fence; the even-only plot
   spans 5.7 decades and shows the envelope. Fix: cell 10 plots
   `degrees = np.arange(0, power.size, 2)` against `power[degrees]`
   with xlabel "Even harmonic degree $l$"; the prose now says the
   odd degrees vanish (~1e-30), `l = 0` is the mean, the rest peaks
   near `l = 45` and decays, with **under 1 % of the total power
   above l = 150** (measured 0.802 %). The refuted sentence ("shows
   how the spectral content falls off with the harmonic degree")
   is gone.
2. **Download size (cell 9 Note) -- applied.** The shipped table in
   `src/kikuchipy/data/_data.py:564-580` lists **0.2-3.0 GB** per
   phase (`steel_r` 3.0, `steel_sigma` 1.5, `alpha_almnsi` 1.1,
   `steel_sigma2` 0.8, `steel_chi` 0.6, four at 0.5; only
   ni/si/austenite/ferrite are 0.3). "about 300 MB each" was wrong
   by up to 10x in the one admonition that warns about download
   weight -> "0.2 to 3 GB each depending on the phase, about 300 MB
   for nickel". `(1001, 1001)` and "bandwidths up to 500" verified
   correct ((1001-1)/2 = 500).
3. **Round trip (cell 11) -- applied.** Measured on the two 401-px
   arrays: Pearson **r = 0.9599**, z-scored NRMSE **0.283**,
   gradient energy 0.2873 -> 0.2093, i.e. **-27.1 %**; the rendered
   panel pair is visibly smoother on the right (inspected). "The
   difference is barely visible: a bandwidth of 188 retains
   practically all of the diffraction signal" -> "Fine detail is
   smoothed by the band limit, but the bands and zone axes that
   indexing correlates on come back intact", which is what the
   figure shows and still carries the argument.
4. **Image quality (cell 17) -- applied.** `xmap.prop["iq"][:4] =
   [0.2107, 0.2096, 0.1915, 0.1985]` against the bare
   `projector.unproject(..., return_image_quality=True)` of the same
   four patterns `[0.3491, 0.3220, 0.3204, 0.3243]` -- **+65.7 %**
   for pattern (0, 0); the map spans 0.119-0.232, so the printed
   0.349 is off the top of the cell-23 colorbar. Cause confirmed in
   `src/kikuchipy/indexing/_spherical/_indexer.py:576-593` and its
   module docstring (lines 150-155): indexing back-projects
   `_preprocess_pattern(...)` and the docstring records the same
   three bands (0.173-0.204 at `n_regions = 10`, 0.289-0.327 at
   `n_regions = 0`, 0.766-0.779 raw), while
   `SphericalBackProjector.__init__` takes no preprocessing
   argument. The sentence now names the `"iq"` property and says
   the map values are computed after the adaptive histogram
   equalization and come out lower than the printed one. The
   "cosine transform" attribution is correct (`_dct_image_quality`,
   `dctn(..., type=2)`) and was kept.
5. **Bimodal refinement histogram (cell 43) -- applied.** 4124
   masked points: **46.4 %** in [0, 0.02) deg, 3.3 % in
   [0.02, 0.04), 4.0 % in [0.04, 0.06), 5.7 % in [0.06, 0.08),
   **24.0 %** in [0.08, 0.10), 14.1 % in [0.10, 0.12), 1.8 % in
   [0.12, 0.14), **0 %** in [0.14, 1.0), 0.6 % (26 points) above
   1 deg. Quartiles p25 0.0000, p50 **0.0419**, p75 0.0933, p90
   0.1060, p99 **0.1275** -- the printed median does sit in the
   trough between the two modes. Prose now names the second mode:
   "just under half of the points land on the spherical solution to
   within 0.02 deg, and almost all of the rest within 0.15 deg of
   it" (measured 99.4 % below 0.14 deg). The frozen print
   (plan 1.21) was not touched.
6. **Namelist table (cell 47) -- applied.** `from_kwargs(...)`
   writes 19 keys; the table mapped 17. Two rows added: `datafile`
   -> `data_file` (a **required** `from_kwargs` argument, verified
   at `_namelist.py:1098-1116`, and the one cell 46 passes as
   `data_file="out.h5"`), and `roimask` (no `from_kwargs`
   argument; stays at the `__init__` default `''`, the whole scan
   -- `_namelist.py:1420`). The table now covers all 19 keys and
   renders as 16 data rows.
7. **Default-68 attribution (cell 30) -- applied.** D4 freezes the
   rationale as "EMSphInx's own namelist default and nothing more";
   "the compromise EMSphInx settles on" ascribed a trade-off
   judgement to EMSphInx that no cited source states. Now: "is the
   compromise -- and EMSphInx's own namelist default", which keeps
   D4's "is the compromise" clause and drops the attribution.
8. **`load_save_data` naming -- applied.** `doc/user/
   related_projects.rst:50-51` and `spherical_indexing.ipynb`
   (cells 9 and 43) call it the **SHT database**; the new
   `load_save_data` markdown was the only place in `doc/` saying
   "EMSphInx master pattern library" (grep: 1 hit, now 0). Changed
   to "the SHT database", inside the already-new cell, so the diff
   stays +33/-0. NB this edits a string frozen in requirements D8;
   recorded here as a deviation rather than as a spec correction,
   since D8's wording is not *wrong*, only inconsistent with the
   page it links and with the tutorial.
9. **`.sht` round trip grid (cell 43) -- applied.** `kp.load`
   synthesizes on `to_master_pattern`'s default grid,
   `2 * bandwidth + 1` (docstring, `_master_pattern_harmonics.py:
   1955-1966`): 2*188+1 = **377** here, and 2*384+1 = **769** for
   the bw-384 `.sht` in `load_save_data` -- both stored outputs
   confirm it. The sentence now says so, so the 377 after cell 11's
   `dim=401` no longer reads as lost resolution.
10. **`not_indexed` point in the IPF panel (cell 36) -- applied.**
    `ckey.orientation2color` mapped the fill identity rotation to
    an ordinary colour, so the Hough panel showed no trace of the
    failure cell 33 describes and cell 38 NaNs. Measured: the point
    is flat index **3334** (row 44, column 34) and was painted
    **pure red** `[1.0, 1.1e-16, 0.0]` -- the `[001]` corner of the
    key, not a subtle artefact, in a map of pastel grains. Added
    `rgb[xm.phase_id.flatten() == -1] = 1` inside the loop (white,
    which is orix's own `not_indexed` phase colour -- see the cell
    32 repr) and cell 33 now says the point is left white in the
    maps and masked out of the comparisons. Verified: the thumbnail
    figure still renders and the colour key still sits beside the
    three panels.

### Findings NOT applied, with the measurement that decides it

11. **`load_save_data`'s `[ ]:` prompt (content 11 / conventions
    2)** -- kept as deviation 4, and its precedent claim is
    **sharpened**: the file's 53 code cells carry counts 3-53,
    **strictly increasing in document order** with one gap (10 ->
    12) and no duplicates; the two other `execution_count: null`
    cells are index 48 (fully commented out) and 112 (empty), and
    **neither has an output**, so ours is the only cell in the file
    that renders an *empty output prompt*. The three options were
    measured: renumbering to 32 touches the 49 following cells and
    breaks V9's hunk budget; inventing the next free integer (54)
    would be the only decreasing step in an otherwise monotone file
    and would assert an execution position that never happened;
    `null` states truthfully that the output was produced outside
    the numbered run. Kept `null`.
12. **Cell-id scheme (conventions 1)** -- kept as deviation 9. The
    reviewer marks it optional and schema-valid; `cell-00 ...
    cell-49` is deterministic across re-executions (uuid4 ids
    would churn the diff on every run), which is the property this
    phase's re-execution loop depends on.
13. **Roadmap Phase 11 checkboxes (conventions 4)** -- outside this
    task's write permission (`specs/roadmap.md` is not the
    validation file); they are ticked at PR time as phases 5/6/7/10
    were.

### Corrections to the implementation section's deviations

- **Deviation 2 over-attributes the 11 "No outputs" thumbnail
  warnings** (conventions info 3, confirmed and quantified).
  Measured over `doc/tutorials/*.ipynb`: **ten** notebooks are
  fully output-less (`feature_maps`,
  `geometrical_ebsd_simulations`, `hough_indexing`,
  `kinematical_ebsd_simulations`, `multivariate_analysis`,
  `pattern_processing`, `pc_calibration_moving_screen_technique`,
  `reference_frames`, `virtual_backscatter_electron_imaging`,
  `visualizing_patterns`) and only those are newly affected by
  `-D nbsphinx_execute=never`. The eleventh,
  `hybrid_indexing.ipynb`, ships **17 cells with outputs**, so
  `nbsphinx_execute = "auto"` does not execute it either and its
  output-less thumbnail cell 82 emits the same warning **on RTD
  today** -- pre-existing, not an artefact of the override. The
  conclusion of deviation 2 (nothing attributable to this phase) is
  unaffected.
- **Deviation 4's precedent** -- see item 11 above.

### Re-run validation matrix (after the fixes)

| # | result |
|---|---|
| V1 | `nbconvert --execute --inplace`: **exit 0, 81 s** wall; then `metadata.widgets` (17,662 B) and the 25 `metadata.execution` blocks (5,350 B) stripped, as in deviation 3. 50 cells (25 markdown / 25 code), execution counts 1-25 contiguous, **0 error outputs, 0 non-stdout streams**, 7 figures, cleanup cell stored output `False`, no `metadata.widgets`, cell metadata exactly `nbsphinx` x2 + the thumbnail pair on cell 36. Cell busy time 56.9 s (C10 indexing 21.07 s, C15 sweep 19.94 s, C17 Hough 4.68 s, C21 refinement 4.44 s). Budget <= 3 min met. |
| V1b | clean-kernel re-execution of a scratch copy: **exit 0, 83 s**. Compared cell by cell after nbval's own treatment (coalesce consecutive streams, collapse carriage returns, apply all nine sanitize rules): **0 of 25 code cells differ**. Unsanitized, the only differing lines are the four speed prints (indexing 198.9 vs 196.9, sweep 405.6/220.3/104.2 vs 399.2/225.2/98.2, Hough 922.0 vs 909.7, refinement 991.6 vs 1198.7 patterns/s), all covered by `regex2`. |
| V2 | `pytest --nbval ... --nbval-sanitize-with tutorials_sanitize.cfg`: **25 passed** in 60.3 s. |
| V3 | unchanged by the fixes and re-inspected: `index.rst` +1 line between `pattern_matching` and `hybrid_indexing`; `run_nbval.sh` +1 line after `pc_fit_plane.ipynb` with the trailing `\` kept and no licence header; `tutorials_sanitize.cfg` +12 lines, regex8/regex9 byte-for-byte as D7. |
| V4 | `sphinx-build -b html -D nbsphinx_execute=never`: **exit 0**, `build succeeded`, 22 s incremental. Scoped grep: **7** sphinx-codeautolink "Could not match transformation" on `spherical_indexing.rst` (`time`, `tempfile`, `pathlib`, `orix.quaternion`, `orix.crystal_map`, `numpy`, `matplotlib.pyplot`), 6 of the same class on `load_save_data.rst`, and the pre-existing `load_save_data.ipynb` cell-102 `"nbsphinx-thumbnail" ... Unsupported output type ... "stream"`. No toctree, undefined-label, duplicate-target or docutils warning from any changed file -- the same classification as deviation 1. |
| V5 | rendered page: **7 figures** (`_10_0`, `_12_0`, `_16_1`, `_23_0`, `_36_0`, `_38_1`, `_42_1`), 2 `admonition note`, 1 table with **17 rows** (header + 16), gallery card with thumbnail `tutorials_spherical_indexing_36_0.png` and tooltip "Spherical harmonic indexing of EBSD patterns" between Pattern matching and Hybrid indexing, IPF colour key beside the three panels, **neither hidden cell renders** (0 matches for the hidden-cell text, the cleanup comment and `os.rmdir`). Every new string renders (`Even harmonic degree`, `centrosymmetric`, `0.2 to 3 GB`, `zone axes that indexing correlates on`, the `iq` qualifier, `EMSphInx's own namelist default`, `two modes`, `left white in the maps below`, `2 * bandwidth + 1`, `377 here`, the `roimask` and `datafile` rows) and every refuted string is gone (`barely visible`, `compromise EMSphInx settles on`, `about 300 MB each`, `falls off with the harmonic degree`). `load_save_data`: the `.sht` section renders, says **SHT database**, stores `(2|769, 769)`, and the parenthesised anchor `id="EMSphInx-spherical-harmonics-master-pattern-(.sht)"` still resolves. The two rewritten figures were inspected as PNGs: the power spectrum is now a legible rise-and-decay curve, and the round-trip pair visibly supports its new caption. |
| V6 | `sphinx-build -b linkcheck` (shared doctrees): **963 rows, 461 working / 66 redirected / 399 unchecked / 37 broken**, "finished with problems" as the spec predicts. The six phase-introduced URLs are all ok in `output.json` -- Lenthe 2019 DOI `redirected` (linkinghub.elsevier.com), EMSphInx GitHub `working`, SHT database `working`, PyEBSDIndex docs `redirected`, orix `orix.io.save` and `CrystalMap` `working`. The two same-page anchors are `unchecked` as before. The 37 broken rows classify as **19** `nbsphinx_prolog` Windows-path artefacts -- exactly one per tutorial notebook, ours included (deviation 11) -- plus 17 pre-existing external rows (rate-limited DOIs, an EMsoft wiki anchor, a Stack Overflow link, a diffsims anchor) and the `jwestraadt/kikuchipy/pull/12` link, which cannot resolve until the PR exists (V10's post-PR check). |
| V7 | `grep -i pseudo` = **0**. Four `dictionar*` hits, all DI cross-references (intro x2, score semantics, What's next); none labels the shipped map, whose panel title is still "Hough + refinement (shipped)". Name counts over the file unchanged by the edits: EMSphInx 22, PyEBSDIndex 4, EMsoft 4, kikuchipy 33; no misspelling variant matches. |
| V8 | `pre-commit run --files` over the five notebook/rst/cfg files (`run_nbval.sh` excluded): ruff, ruff-format, black-jupyter, licenseheaders all **Passed, no modifications**, both before execution (plan 1.4's ordering) and after. |
| V9 | `git status --short`: the six modified files (`CHANGELOG.rst`, `index.rst`, `load_save_data.ipynb`, `run_nbval.sh`, `tutorials_sanitize.cfg`, this file) plus the untracked notebook; nothing else, nothing untracked in `doc/tutorials/`. `load_save_data.ipynb` still **+33 / -0**, `nbformat_minor` 4, no `id` keys. Both notebooks are byte-identical to `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`, pure LF, no trailing whitespace, no empty cells. |
| V10 | `CHANGELOG.rst` untouched by the fixes (+27 / -32 as before); the #12 link check stays sequenced after PR creation. |
| V11 | **948,974 B** (0.90 MiB), down from 1,004,294 B: the even-degree spectrum PNG is much smaller than the 188-spike one. Still in family with hybrid 1.16 MB. |

### Two markdown-only edits after execution (deviation 10's precedent)

After the re-execution, two of the new sentences were split for
readability -- the power-spectrum sentence into "…all odd degrees
vanish … and we plot the even ones." + "The constant term at
$l = 0$ …", and the image-quality sentence into two ("Indexing
measures it on the adaptive histogram equalized pattern it
back-projects rather than on the raw one …"), which also removed a
"computed … computed there" repetition. Markdown is neither
executed nor compared by nbval, so stored outputs still match
their sources; re-proved afterwards by **nbval 25 passed**
(58.9 s), `pre-commit` **Passed, no modifications**, and a rebuilt
html page (**exit 0**, `build succeeded`; the scoped grep returns
only the same 7 codeautolink warnings) in which both paragraphs
render with well-formed MathJax spans (`\(l\)`, `\(10^{-30}\)`,
`\(l = 0\)`, `\(l = 45\)`). Final file: 948,974 B, canonical JSON,
pure LF, 50 cells, cell metadata exactly `nbsphinx` x2 +
`nbsphinx-thumbnail`/`tags`.

### Working tree after the fixes

`git status --short` is exactly the six intended modifications
(`CHANGELOG.rst`, `doc/tutorials/index.rst`,
`doc/tutorials/load_save_data.ipynb`, `doc/tutorials/run_nbval.sh`,
`doc/tutorials/tutorials_sanitize.cfg`, this file) plus the one
untracked `doc/tutorials/spherical_indexing.ipynb`; `git diff
--numstat` is 27/32 CHANGELOG, 1/0 index.rst, 33/0 load_save_data,
1/0 run_nbval.sh, 12/0 sanitize cfg, and this file append-only
(0 deletions). No `src/` file changed; nothing untracked in
`doc/tutorials/` besides the notebook.

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
