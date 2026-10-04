# PARKED PLAN: `feat-NLPAR` -- non-local pattern averaging (NLPAR) in the kikuchipy fork

**Status (2026-09-11):** designed, NOT approved, NOT started. Johan parked
it at the approval gate ("save this plan in specs, for implementation
later"). Saved untracked under `specs/_research/` (ignored by the staging
branch's `.gitignore`). Nothing below was executed: no `feat-NLPAR` branch,
no commits, no spec folder. To resume: re-verify the base sha
(`feat-spherical-indexing` @ `6723aaf0`, upstream `31666938`) and the
installed PyEBSDIndex version, then start at "Step 0: branch mechanics";
Step 1 still needs Johan's approval of the spec `plan.md`.

## Context

Johan asked (2026-09-11): "on the develop branch there is a spec for NLPAR
implementation, branch off this branch called feat-NLPAR, then implement
(planning-spec (Fable, xhigh+ultracode) then tests-implementation-testing-
adversarial review-fixes (OPUS 5 xhigh+ultracode))."

Finding: **no NLPAR spec exists** on any branch (develop, hrebsd-dic, every
origin/upstream ref), in the stash, or on disk; `specs/` on develop has zero
matches for nlpar/non-local/denois. Only upstream's tutorials cite
`brewick2019nlpar` (bibliography key already present) and
`hybrid_indexing.ipynb`'s last cell points readers to PyEBSDIndex's NLPAR.
Johan's decisions (AskUserQuestion, 2026-09-11):
1. **Write the spec fresh** in the planning phase from Brewick, Wright and
   Rowenhorst 2019, with EMsoftOO `mod_NLPAR.f90` (BSD-3) as an equation
   cross-check and PyEBSDIndex `nlpar_cpu.py` (public-domain NRL code) as the
   numerical oracle.
2. **Scope: full CPU NLPAR + tutorial** (sigma estimation, lambda
   optimisation, search window, edge policy, lazy/dask, sigma map, CHANGELOG,
   tutorial with a re-indexing demonstration). No GPU backend.
3. **Data: synthetic + `nickel_ebsd_large` + `si_wafer`** (si_wafer
   weekly/gated).

Base: `feat-spherical-indexing` @ `6723aaf0` = upstream `pyxem/kikuchipy`
develop `31666938` (v0.14.dev1) + 16 spherical staging commits + 2 fork-local
ignore commits (`c0b6b7f2` ignores `specs/`, `6723aaf0` ignores raw data).
The fork's `specs/` tree (58 files) exists only on develop/hrebsd-dic, so
`feat-NLPAR` must re-adopt it before a spec can be recorded.

Standing constraints: `stash@{0}` (Johan's develop-only edits) is never
popped; `specs/2026-08-16-constitution/upstream-issue.md` never edited;
untracked `specs/_research/plan-upstream-merge-0.13.1.md` (parked plan) never
committed or touched; `doc/tutorials/{hybrid_indexing,spherical_indexing,
load_save_data}.ipynb` never swept; explicit pathspecs only; signed commits
with trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`;
no em-dashes in prose; main `.venv` stays CuPy-free; pre-commit via
`uvx pre-commit run --files <explicit list, never specs/>`.

## Reference facts the spec builds on (verified 2026-09-11)

**Paper / algorithm.** sigma_i^2 = min over the (up to 8) nearest neighbours
of ||p_i - p_j||^2 / (2 N); d_ij = [sum_k (p_i-p_j)^2 - N (s_i^2+s_j^2)] /
[sqrt(2N) (s_i^2+s_j^2)]; w_ij = exp(-max(d_ij - dthresh, 0) / lambda^2),
self-weight 1, normalised over the (2 SR+1)^2 window; lambda chosen so the
mean weight of the pattern of interest in its 3x3 neighbourhood hits a
target (PyEBSDIndex: median of the fits to 0.5/0.34/0.25 = the 0.34 fit,
since lambda is monotone in the target).

**PyEBSDIndex 0.3.10.1** (`.venv/.../pyebsdindex/nlpar_cpu.py`; public
domain, NRL; derivative works must carry a change notice + acknowledge NRL):
static `njit` kernels callable on in-memory arrays: `NLPAR.sigma_numba`
(:752-818; 3x3 window **clipped** at edges; saturation protect excludes
pixels >= 0.9961*max; `d2 >= 1e-3` duplicate guard; if no neighbour passes,
sigma = 1e12 -> uniform window average) and `NLPAR.nlpar_nb` (:820-936;
search window always (2sr+1)^2, **shifted inward** at edges; `0.999*max`
saturation; `dnorm <= 1e-8 -> d = 1e6 n2`; float32 accumulation; per-column
pair memo saves only ~3/48 of the work). Quirks NOT to port: window index
goes out of bounds when an axis is shorter than 2sr+1 (no bounds check);
`diff_offset` accumulates across threads; lambda optimisation counts
"phantom" zero-distance neighbours at map edges (776/37125 entries on
nickel_ebsd_large; +2.05 % on lambda at target 0.34: 1.098 vs phantom-free
1.121); `opt_lambda` uses `max(d2, dthresh)` while the averaging kernel uses
`max(d2 - dthresh, 0)`; saturation thresholds are tile-local. Both kernels
have identical flags in 0.3.9.2 (the CI oldest pin), so no version gate.
PyEBSDIndex reads kikuchipy h5ebsd files (`ebsd_pattern.py:105-111`), so a
file-based end-to-end oracle is also possible. Numba `exp` differs by 1
float32 ulp between compiled and `py_func`.

**EMsoftOO `mod_NLPAR.f90`** (BSD-3; `C:\Users\westraadt.1\Repos\EMsoftOO\
Source\EMsoftOOLib\mod_NLPAR.f90`): same formula, defaults sw=3,
lambda=0.375, but sigma^2/S/sqrt(2N) truncated to integers (line 301), a
10000.0 sentinel that kills border smoothing, a stale-row bug on the last
row, no lambda optimisation. Equation cross-check only; no code ported.

**kikuchipy reuse points.** `EBSD.average_neighbour_patterns`
(`src/kikuchipy/signals/ebsd.py:954-1122`): the inplace/lazy_output/
show_progressbar contract, `get_dask_array(chunk_bytes=8e6, rechunk=True)`,
`da.overlap.map_overlap(..., boundary="none")`, `window_sums`-style second
dask operand, `LazyEBSD(..., **self._get_custom_attributes())`;
`_dask.py:198-216` `_get_chunk_overlap_depth`; signal-mask convention True =
excluded; `dependency_version["pyebsdindex"]` gating
(`tests/test_signals/test_ebsd_hough_indexing.py:35-37`); no `parallel=True`
anywhere in `src/`; the spherical tests' strict kernel-flag + `py_func`
tests (`tests/test_indexing/test_spherical_euler.py:72-96, 509-529`).
Data: `nickel_ebsd_large` (55,75|60,60) uint8, max 253, prefetched in CI;
`si_wafer` (50,50|480,480) uint8, 311 MB pooch download; `dummy_signal`
(3,3|3,3). Measured on this machine: PyEBSDIndex sigma 0.12 s and averaging
0.54 s single-thread on nickel_ebsd_large at sr=3.

**Fork spec conventions** (from `develop:specs/{mission,tech-stack,
roadmap}.md` and the GPU/HREBSD exemplars): dated folder with
`requirements.md` (Scope, D-numbered decisions marked (frozen)/(recorded),
Context), `plan.md` (section 0 constitution amendments with exact line
targets, numbered build modules, implementation order, deliberately-not-
built, adversarial review + mutation list, open questions decided vs FOR THE
USER, commits), `validation.md` (Automated / Local-gated / Weekly /
requirement-to-test table / Performance baselines / Manual / Definition of
done / append-only numbered ledger); every tolerance MTP (measured then
pinned, `pytest.approx(measured, rel=0.05)` or the ~2x margin); 4-commit
cadence per stage (spec -> failing tests + skeletons -> implementation +
CHANGELOG + ticks -> push); `@njit(cache=True, nogil=True)`, no parallel,
`.py_func`-tested, 100 % coverage of new private modules recorded; oldest
matrix py3.10/numpy 1.23/numba 0.57/orix 0.12.1/pyebsdindex 0.3.9.2; two
adversarial reviewers + mutation list + disposition table; the three spec
docs re-submitted to review as definition of done.

## Design (recommended; becomes the spec's D-numbers)

**API** (three methods after `average_neighbour_patterns` in `ebsd.py`):
```python
def nlpar(self, search_radius: int | tuple[int, ...] = 3, lam: float | None = None,
          dthresh: float = 0.0, target_weight: float = 0.34,
          sigma: float | np.ndarray | None = None, signal_mask: np.ndarray | None = None,
          saturation_protect: bool = True, dtype_out=None,
          show_progressbar=None, inplace: bool = True, lazy_output=None) -> EBSD | LazyEBSD | None
def get_nlpar_sigma(self, signal_mask=None, saturation_protect=True, show_progressbar=None) -> np.ndarray
def get_nlpar_lambda(self, target_weight=0.34, dthresh=0.0, sigma=None, signal_mask=None,
                     saturation_protect=True, show_progressbar=None) -> float
```
`lam=None` optimises for `target_weight` (phantom-free objective, same
`max(d - dthresh, 0)` weight as the averaging kernel) and logs the value;
`signal_mask` affects distances/sigma only (the weighted sum averages all
pixels, as PyEBSDIndex); `search_radius` per navigation axis; radius 0
everywhere warns and no-ops like `average_neighbour_patterns`; 0-D
navigation raises. Output: input dtype by default (rounded, see the
reconciled decisions; no per-pattern rescale: a convex combination stays in
range; parity with PyEBSDIndex `rescale=False`), `dtype_out="float32"` for
the raw average.

**Numerics.** PyEBSDIndex parity by construction: 3x3 sigma window clipped
at map edges, search window shifted inward (clamped safely when an axis is
shorter than 2r+1), float32 pair accumulators, global `max(data)` saturation
threshold with PyEBSDIndex's two constants (0.9961 sigma / 0.999 average),
`dnorm <= 1e-8` guard, sigma = 1e12 fallback. Recorded deviations: global
(not tile-local) saturation max; no pair memo; no
`diff_offset`/`backsub`/`stem_scale`; lambda objective ignores missing
neighbours; duplicate guard and `n2 == 0` policy per the reconciled
decisions below.

**Kernels** in new `src/kikuchipy/signals/util/_nlpar.py`, all
`@njit(cache=True, nogil=True)` (no fastmath, no parallel): `_window_bounds`,
`_nlpar_sigma_kernel` (sigma + normalised 3x3 distances), `_nlpar_distances_
kernel` (window distances for the kept region only), `_nlpar_weights_kernel`
(the only kernel with `exp`, so the others are bitwise `py_func`-testable),
`_nlpar_weighted_sum_kernel`. Python drivers: chunk wrappers declaring
`block_info=None` (chunk location -> which sides carry a halo -> PyEBSDIndex's
`calclim` for free), NLPAR-specific depth helper (unchunked axis 0; `2r+1 >=
n` -> single chunk; else `max(r, 2r+1 - min(first, last chunk))`; pass 1 depth
2), `ensure_minimum_chunksize` rechunk (fixes the issue-#230 irregular-chunk
class), `_nlpar_optimize_lambda` (Nelder-Mead, bounds [1e-3, 10], stride 2
above 1e6 points). Two `map_overlap` passes: pass 1 (sigma + 3x3 distances,
always eager, output ndim kept to avoid drop/new_axis metadata bugs) and
pass 2 (weighted average, `sigma` as a second nav-chunked dask operand like
`window_sums`). Module docstring carries the NRL derivation notice with the
change list and date; `EBSD.nlpar` Notes and the CHANGELOG acknowledge
PyEBSDIndex; module is GPL like the rest of `signals/`.

**Estimated cost**: nickel_ebsd_large ~0.2-0.4 s with dask threads; si_wafer
~5-10 s multi-threaded (memory-bound) plus download.

**Tests** (two new files): `tests/test_signals/test_util/test_nlpar.py`
(kernel flags, `py_func` bitwise / 1-ulp, PyEBSDIndex kernel oracles on
random and Ni data with mask/saturation on/off, end-to-end float32 parity,
optimiser parity incl. the documented +2 % phantom deviation, depth rules,
calclim from block_info) and `tests/test_signals/test_ebsd_nlpar.py`
(independent NumPy transcription of the formulas on small maps incl. 1-D
nav and asymmetric radius; constant map identity; iid-noise sigma recovery
and noise reduction; two-grain sharp boundary not blurred vs box averaging;
lazy == eager across regular/irregular/sub-depth chunkings; inplace/
lazy_output/dtype/mask/radius/sigma-argument/duplicate/small-map contracts;
nickel_ebsd_large IQ + ADP improve (pinned); gated Hough before/after and
file-based PyEBSDIndex end-to-end). All tolerances MTP.

**Reconciled design decisions** (architecture vs validation reports):
- Integer output = `np.rint` + clip to the input range (unbiased; a
  truncating cast shifts the mean by -0.5 grey level); parity with
  PyEBSDIndex is asserted on the float32 kernel output, never on written
  files (its writer truncates). Recorded deviation.
- Duplicate guard = `d2 > 0` (scale-free) instead of PyEBSDIndex's absolute
  `d2 >= 1e-3`: identical on integer data, and float data in [0, 1] no longer
  degrades to a box filter (V1 scale-invariance test). Recorded deviation.
- A pair with `n2 == 0` (every pixel saturated or masked) gets `d = +inf`
  (weight 0); PyEBSDIndex gives it weight 1 (`d = 1e6 * 0`). Recorded.
- Saturation: global `max(data)` with PyEBSDIndex's two constants (0.9961
  sigma / 0.999 average) for parity on uint16/float too; tile-local max is
  the recorded deviation (the oracle runs whole arrays, so parity holds).
- Lambda objective excludes phantom (out-of-map) neighbours and uses the
  averaging kernel's `max(d - dthresh, 0)` form; the ~2 % difference from
  PyEBSDIndex at target 0.34 is measured and documented (V6).
- Kernel literal discipline: float32 literals cast explicitly so `py_func`
  parity holds under NumPy 1.23 (oldest job) and NumPy 2 alike; the `exp`
  kernel is pinned at 1 ulp, all others bitwise.
- No pyebsdindex version gate beyond the pyproject floor: 0.3.9.2's two
  kernels are identical; the local oldest-matrix recipe verifies it.

**Validation catalogue for `validation.md`** (V-numbers; each tolerance MTP
unless marked bitwise; drafting seeds measured 2026-09-11 with a pure-NumPy
reference on this machine are recorded as seeds, not pins):
- V0 kernel discipline: `KERNEL_NAMES` completeness, `nogil`, `cache`, no
  `parallel`/`fastmath`; `py_func` bitwise (exp kernel 1 ulp).
- V1 analytic identities (test-local float64 reference `nlpar_reference`):
  constant map identity; `search_radius=0` identity; injected `sigma=1e-6`
  identity (self-weight 1, neighbours underflow to 0); `lam=1e6` equals the
  shifted-window box mean; power-of-two scale invariance (bitwise); weight
  formula on injected d incl. `dthresh` in {0, 0.5}.
- V2 PyEBSDIndex sigma parity (`NLPAR.sigma_numba`): synthetic generators
  (identical + Gaussian sigma 8; two-grain; random uniform with 5 % saturated
  and exact-duplicate variants), masks none/automask, protection on/off;
  nickel_ebsd_large raw and background-corrected; expect bitwise, fallback
  ulp pin; clipped-3x3 edge policy asserted against a shifted variant.
- V3 PyEBSDIndex averaged-pattern parity (`NLPAR.nlpar_nb`, injected
  PyEBSDIndex sigma, and end-to-end with our sigma): sr in {1,2,3}, lam in
  {0.7, 2.5}; expect bitwise on float32, fallback per-pixel ulp pin and
  "zero pixels differ by >= 2 grey levels" after rounding; border band vs
  clamp/zero-extend alternatives (must differ), interior identical; the
  saturation arm separates per-pair n2 from a global-N mutant; PyEBSDIndex
  JIT warm-up in a module fixture (measured, recorded).
- V4 iid-noise oracle (identical patterns + N(0, 8), never sigma_true = 1):
  sigma_hat/sigma_true median band (derived bias ~ -1.0..1.4 sqrt(2/N) in
  sigma^2), normalised-d mean/std bands, fraction of weights exactly 1,
  noise reduction vs the weight-derived expectation, monotonicity in lambda,
  mask-polarity separation.
- V5 two-grain sharp boundary: cross-boundary weights underflow to exactly 0
  (derived d ~ 159 at Delta 30, sigma 8, N 1024), contrast retained >= 0.995,
  boundary columns within the within-grain residual bound; Gaussian
  `average_neighbour_patterns` blurs (tutorial figure).
- V6 lambda: closed-form optimiser check on constructed d (c in {2,5,12}:
  lambda(tw) = sqrt(-c / ln((1/tw - 1)/8))); exact equality with a test-local
  `loptfunc` re-implementation on PyEBSDIndex's own `dout`; end-to-end on
  nickel_ebsd_large (seeds: 1.1164 / 2.5246 with phantoms, 1.1387 / 2.5787
  without); lambdas ascend with decreasing target; bound-hit warning.
- V7 lazy/chunking/1-D/determinism/dtype/inplace: lazy == eager bitwise over
  single, regular, irregular tiny-edge and both-axes chunkings and the
  (26,26,3) last-chunk case on nickel_ebsd_large; issue-230 regression; 1-D
  scan == (1, n) run; map smaller than the window; scheduler/thread
  invariance; uint8/uint16/float32/float64 round trip with `rint`; custom
  attributes carried; `lazy_output=True` + `inplace=True` raises.
- V8 real-data effect (default subset, weekly full): background-corrected
  nickel_ebsd_large, ADP seed 0.600 -> 0.904 (lam auto ~2.52) / 0.766 (0.7),
  IQ measured; Hough on `extract_grid((11, 15))` before/after: `pq`, `fit`,
  `nmatch`, `cm` medians and misorientation to the stored (DI-refined, not
  Hough) orientations; pin only measured improvements (>= 0.5x measured
  gain), "not worse" otherwise, said so in the ledger.
- V9 low-signal weekly/local: si_wafer sigma CV, N_eff, IQ gain (weekly,
  `allow_download=True`); ni_gain series local-only ledger entry.
- V10 policy oracles where PyEBSDIndex is not the oracle: `n2 == 0` pair,
  `dthresh > 0` consistency, duplicate neighbours, global saturation rule,
  uint16 two-threshold arm.
- V11 performance: pytest-benchmark entry and `record_property` runtimes,
  recorded never gated (seeds: pure-NumPy full map 6-14 s; PyEBSDIndex
  kernels 0.12 + 0.54 s single-thread).
Mutants M1-M22 (sign of correction; sigma unsquared; sqrt(n2); d2/n2; self
weight 0/excluded; drop max(.,0); exp(-d/lam); clamp for shift; zero-extend
+ window sums; depth sr with tiny edge chunk; per-chunk saturation max;
global N under saturation; floor for rint; per-pattern rescale; mask
polarity; mask not forwarded to sigma; phantoms counted; optimiser
`max(d, dthresh)`; mean for median; shifted 3x3 sigma window; fastmath;
float64 accumulation) each die by a named default-suite test or are
recorded reviewed-only with the reason (M22 only breaks the bitwise pin).
Budget: default-suite additions <= ~90 s per worker incl. the PyEBSDIndex
JIT fixture; weekly <= 5 min.

**Docs**: `doc/tutorials/nlpar.ipynb` (formulas + acknowledgement; synthetic
two-grain demo with sigma map and boundary preservation; nickel_ebsd_large
sigma map, lambda-vs-target curve, before/after patterns, IQ/ADP maps,
background removal + Hough indexing before/after; si_wafer lazily; parameter
guidance and PyEBSDIndex differences), registered in `index.rst` after
`pattern_processing`, `run_nbval.sh`, sanitize regexes as needed, stored
outputs if > ~2 min on the RTD builder; gallery example
`examples/pattern_processing/nlpar.py`; CHANGELOG Unreleased/Added bullets
with the fork PR link (decision 1 below). `hybrid_indexing.ipynb` is NOT
edited (never-sweep rule); the new tutorial links to it instead.

## Process (approved model assignment)

Planning/spec: Fable, xhigh + ultracode Workflow. Tests, implementation,
adversarial review, bug injection, fixes: Workflow agents with
`model: "opus"`, effort `xhigh`. Commits and pushes by the main loop only,
after each workflow returns. Per-stage sequence: failing-tests workflow ->
commit -> implementation workflow -> adversarial-review workflow ->
bug-injection workflow -> closing-gates workflow -> commit + push.

### Step 0: branch mechanics (commit 0, fork-local)
Pre-flight: clean tree at `6723aaf0`, `stash@{0}` present, record
`H0 = git hash-object specs/_research/plan-upstream-merge-0.13.1.md` (and
the hash of this file).
1. `git switch -c feat-NLPAR feat-spherical-indexing`.
2. `git checkout develop -- specs` (all 58 files, byte-identical: `git diff
   --quiet develop -- specs`; `upstream-issue.md` blob `e60097e2`, not the
   stash's `82791b0f`). Take develop, not hrebsd-dic (HREBSD is never merged;
   its roadmap has a stray BOM).
3. `.gitignore`: delete the two lines `# Fork-local design documents ...` /
   `specs/` (keep the raw-data rules). `.pre-commit-config.yaml:20`:
   `exclude: ^\.github/pull_request_template.md` -> `exclude:
   ^(\.github/pull_request_template.md|specs/)` (ruff rev v0.16.6 untouched).
4. Guard the parked plans: append `specs/_research/plan-upstream-merge-0.13.1.md`
   and `specs/_research/plan-nlpar-2026-09-11.md` to `.git/info/exclude`
   (local only); verify `git check-ignore -v specs/x.md` exits 1,
   `git status --short -- specs` shows nothing staged for the parked files,
   their hashes unchanged.
5. `git add .gitignore .pre-commit-config.yaml` (specs already staged by the
   checkout; never `git add specs`); `git commit -s` "Re-adopt the fork specs
   tree on feat-NLPAR" (body: fork-local, never in an upstream PR).

### Step 1: spec (Fable workflow) -> user approval -> commit 1
Workflow `nlpar-spec-draft`: three parallel drafters (requirements / plan /
validation) from the three design reports (architecture, validation,
process; their substance is captured in this file) using
`hrebsd-dic:specs/2026-09-07-hrebsd-dic/*.md` as the shape; then three
read-only critics (factual vs repo/git/.venv, completeness, executability)
with evidence-required findings; a fixer with a disposition table; loop
until dry, max 2 rounds (<= 11 agents). Spec folder
`specs/2026-09-11-nlpar/` (re-date if resumed much later). Constitution
amendments (plan section 0, applied in the same commit): `mission.md` "Fork
feature path: NLPAR" paragraph after line 53; `roadmap.md` a `---` rule then
`# Feature path: NLPAR (branch feat-NLPAR; spec 2026-09-11-nlpar)` with
Stage A/B/C boxes after line 125 (not a Phase 13: NLPAR is not in the
EMSphInx chain); `tech-stack.md` "NLPAR feature path" section after line 85
(base sha, update rule = merge feat-spherical-indexing in, never develop;
branch/PR policy; dtype policy scoped like HREBSD's; PyEBSDIndex licensing
+ derivation-notice rule; oldest-matrix recipe `uv run --isolated --python
3.10 --with "numpy==1.23.0" --with "numba==0.57" --with "orix==0.12.1"
--with "pyebsdindex==0.3.9.2" pytest tests/test_signals -k nlpar`;
pyebsdindex-optional rule: `src/` never imports it, oracle tests skipif +
in-test import of `pyebsdindex.nlpar_cpu` calling the kernels' `.py_func`;
numba-cache flake rule: `-n 0` then `-n 4`, re-run red tests alone; fixtures
only under `src/kikuchipy/data/**` or generated; CHANGELOG uses the fork
PR-link convention). **Gate: Johan approves `plan.md`**, then commit 1
"Add NLPAR spec and constitution amendments" (specs only; no BOM; roadmap
line 1 unchanged).

### Steps 2-4: three build stages (Opus workflows)
- **Stage A, engine**: kernels, sigma estimation, chunk wrappers, depth
  helper, eager `EBSD.nlpar()` + `get_nlpar_sigma()`; commit 2 "Add NLPAR
  Stage A skeletons and failing tests" (skeletons with `NotImplementedError`,
  test modules, validation ledger entry; never pushed alone), commit 3
  "Implement NLPAR Stage A: kernels, sigma estimation and EBSD.nlpar"
  (+ measured pins, review fixes, CHANGELOG API bullet, roadmap ticks). Push.
- **Stage B, optimisation and scale**: `get_nlpar_lambda()` + `lam=None`,
  lazy/dask path, eager == lazy pins incl. irregular and sub-depth chunks,
  nickel_ebsd_large subset default / full weekly, si_wafer weekly/gated,
  performance baselines recorded; commits 4 and 5. Push.
- **Stage C, tutorial**: notebook, index/nbval/sanitize registration,
  gallery example, CHANGELOG tutorial bullet, roadmap ticks; commit 6
  "Add NLPAR tutorial notebook". Documentation stage: skips the failing-tests
  gate, keeps the Phase-11-style validation matrix + failure-mode review.
  Push; then the three spec docs are re-submitted to review, the fork PR
  (#17 expected) is opened into `feat-spherical-indexing` with the template,
  CI watched (ubuntu/windows green expected), and the roadmap tick commit
  names the PR. Merge only on Johan's go.

Per-stage workflows (all agents `model: "opus"`, effort `xhigh`; every
workflow <= 15 agents): `nlpar-stage-tests` (skeleton agent; one writer per
test module running only its module `-n 0`; test-critic asking whether a
plausibly wrong implementation would still pass and whether every plan-7
mutant has a NAMED killer; fixer + ledger entry); `nlpar-stage-implement`
(implementers in dependency order kernels -> drivers -> EBSD methods;
measurer filling MTP pins with date/machine/recipe; gate runner: `-n 0`,
`-n 4` with red tests re-run alone, coverage 100 % of `_nlpar.py`, full
suite, pre-commit on explicit files, oldest-matrix recipe);
`nlpar-stage-review` (fidelity reviewer against the paper, both PyEBSDIndex
kernels and EMsoftOO; conventions/integration reviewer; each finding batch
refuted by two skeptics, a finding dies only if both refute; fixer adds the
named killer tests and the disposition table); `nlpar-stage-mutants`
(plan-section-7 mutants applied ALONE in the main tree, sequential batches
of 3, each must die by its named test, survivors -> strengthened tests with
kills verified by re-injection; no worktrees by default because the editable
`.pth` points at the main `src`); `nlpar-stage-close` (gates, CHANGELOG/
roadmap/ledger checks, never-sweep check via `git diff --name-only`, spec
re-submission critic).

Mutation-list seeds for plan section 7 (each must die by a named test): sigma
correction sign flipped; sigma used unsquared in d; self-weight not forced;
window clipped instead of shifted (or vice versa for the 3x3 sigma window);
depth off by one; halo region written instead of zeroed; float64 instead of
float32 accumulation (kills the bitwise oracle); saturation threshold on
the wrong constant; mask ignored in n2; duplicate guard removed; lambda
objective using `max(d2, dthresh)`; phantom neighbours counted; missing
`+inf` padding for small maps; rounding replaced by truncation; 1-D nav
misrouted; lazy path skipping the rechunk.

### Step 5: definition of done
All roadmap boxes ticked with evidence in the ledger; every MTP replaced by
a dated measured value (recipe + machine); coverage 100 % of `_nlpar.py`
recorded; oldest-matrix run recorded; the three spec docs re-submitted to
review after Stage C; `git log origin/feat-NLPAR..feat-NLPAR` empty; PR
opened into `feat-spherical-indexing` with CI green on ubuntu/windows
(merge awaits Johan); memory note written (branch, base, the PR policy and
its consequence for the staging branch, PyEBSDIndex quirks).

## Decisions by Johan (AskUserQuestion, 2026-09-11)
1. **Branch/PR policy: PR from `feat-NLPAR` into `feat-spherical-indexing`
   at the end** (after Stage C). Signed commits are pushed to
   `origin/feat-NLPAR` after each stage (CI on push is extra signal; a
   failing-tests commit is never pushed alone); one fork PR opened with
   `gh pr create --repo jwestraadt/kikuchipy --base feat-spherical-indexing`
   using the PR template, expected number #17; merged as a merge commit on
   Johan's go once ubuntu/windows CI is green (macOS stays red on the known
   pseudo-symmetry `23 == 22` test, recorded). Consequences recorded in the
   spec: the merge brings the fork `specs/` tree and the ignore/exclude
   commits onto `feat-spherical-indexing`; upstream PRs continue to be cut
   per commit (the 16 spherical commits and the NLPAR code/tutorial commits),
   skipping the fork-local commits (ignore rules, specs re-adoption, spec
   commit, roadmap ticks), so the staging property is kept at the commit
   level. CHANGELOG bullets therefore use the fork PR-link convention
   (`` (`#17 <https://github.com/jwestraadt/kikuchipy/pull/17>`_) ``,
   confirmed at PR open) rather than the spec-folder wording, and the
   roadmap gate list keeps "PR opened -> PR merged" with a final
   "Tick NLPAR boxes in roadmap (jwestraadt/kikuchipy#17)" commit.
2. **`hybrid_indexing.ipynb` stays untouched**; `nlpar.ipynb` links to it.

## Risks
| Risk | Handling |
|---|---|
| Parked plans swept once `specs/` is un-ignored | `.git/info/exclude` entries, explicit pathspecs, hashes re-checked at every close gate |
| Three divergent roadmaps (develop, hrebsd-dic, feat-NLPAR) | never merge develop into feat-NLPAR; all appends at EOF; cherry-pick the spec commit if ever ported |
| BOM in spec files | Write/Edit tools only, never PowerShell `Set-Content`; gate on `roadmap.md` line 1 |
| pyebsdindex absent on the minimum-requirement CI job | skipif marker; NumPy transcription oracle keeps that job meaningful |
| Numba cache flake under `-n 4` (pyebsdindex import redirects NUMBA_CACHE_DIR) | `-n 0` first; red tests re-run alone; oracle imports only `nlpar_cpu` and uses `.py_func` |
| `py_func` vs compiled `exp` differs by 1 ulp | `exp` isolated in one kernel with a 1-ulp pin; others bitwise |
| Lazy irregular / sub-depth chunks | `ensure_minimum_chunksize` rechunk; pins on (1,1), (2,3), (4,4), #230 tuple |
| Tutorial runtime on RTD | compute on nickel_ebsd_large only; si_wafer numbers quoted from the ledger; stored outputs if > ~2 min |
| Oldest matrix numba 0.57 | no post-0.57 features in kernels; one recorded local run per stage |
| Mutant false survivors in worktrees | in-tree sequential injection; worktree variant only with `PYTHONPATH` + `kikuchipy.__file__` guard |

## Verification (whole feature)
Spherical and full suites unchanged and green (`-n 0` then `-n 4`);
`pytest tests -k nlpar` green with the PyEBSDIndex oracles both installed
and skipped; coverage 100 % of `_nlpar.py`; oldest-matrix recipe green;
`uvx pre-commit run --files <changed>` clean; `sphinx-build -b html` exit 0
and nbval green on `nlpar.ipynb`; `git diff --name-only` never lists the
never-sweep notebooks; develop, hrebsd-dic and feat-spherical-indexing
untouched until the PR merge (`git rev-parse`: 71a1b2f3 / 02a529c0 /
6723aaf0).
