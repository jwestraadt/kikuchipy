# NLPAR -- `feat-NLPAR`: requirements

Status (2026-10-04): spec drafted, spec-review rounds 1, 2 and 3
folded (three critics per round; disposition tables in `plan.md`
section 10; amendments dated 2026-10-04 in place; round-2 and round-3
measurements in validation.md ledger entries 3 and 4), awaiting the
user's approval of `plan.md`.
Branch `feat-NLPAR` off fork `develop` @ `18d59c07`
(= `origin/develop`; `upstream/develop` is at `31666938`, 24 commits
ahead, merge parked). Spec folder `specs/2026-10-04-nlpar/`, one
folder for all three build stages (A engine, B optimisation and scale,
C tutorial). The feature: non-local pattern averaging (NLPAR) of EBSD
patterns after Brewick, Wright and Rowenhorst, Ultramicroscopy 200
(2019) 50-61, doi 10.1016/j.ultramic.2019.02.013 (bibliography key
`brewick2019nlpar`, `doc/user/bibliography.bib:11-19`), as a CPU
method of `kikuchipy.signals.EBSD` with its own numba kernels, a
sigma map getter, a lambda optimiser, a tutorial and a gallery
example.

**Relation to the parked plan.** The design was written on 2026-09-11
and parked unexecuted in `specs/_research/plan-nlpar-2026-09-11.md`
(algorithm facts, PyEBSDIndex quirks, API, kernels, numerics, V0-V11
with drafting seeds, mutants M1-M22, stages, standing constraints).
That file is the substance of this spec; its facts are carried inline
with the line references re-verified today against the installed
PyEBSDIndex 0.3.10.1. Two of its decisions are superseded by the
user's decisions of 2026-10-04 (approved session plan
`C:/Users/westraadt.1/.claude/plans/how-do-run-a-cuddly-possum.md`):
(1) base and PR target are fork `develop`, not
`feat-spherical-indexing` (D11; supersedes decision 1 of the
2026-09-11 plan, whose Step 0 "re-adopt the specs tree" drops out
because `develop` already tracks `specs/`); (2) the public names
follow upstream PR pyxem/kikuchipy#824 (D1). Everything else stands.

**Relation to upstream PR #824.** jorgenasorhaug opened
pyxem/kikuchipy#824 on 2026-09-27 (2 commits, 6 files; read
2026-10-04 with `gh`: OPEN, CHANGES_REQUESTED by hakonanes, updated
2026-10-03). The fork builds the same feature under the same module
path and method name so that the two converge, with its own kernels
and the design below; #824's defects are RECORDED, never reproduced
(Context), and the convergence rule is take-ours (D11.6). No review
is posted on #824 unless the user says so.

**Drafting caveat (D13).** Nothing was executed while drafting beyond
read-only inspection and three cheap re-measurements dated 2026-10-04
(installed versions; the Ni dataset's shape, dtype and range from the
cached file; the phantom-neighbour count by arithmetic). Every
tolerance is `MEASURED-THEN-PINNED` (MTP), filled at the owning
stage's gate with the recipe and machine recorded in validation.md.
Numbers quoted as "seed" are the parked plan's 2026-09-11 drafting
seeds (pure-NumPy reference); seeds, not pins. Facts marked
"(unverified)" are verified at the gate that depends on them.

**Reference-implementation status, recorded.** PyEBSDIndex 0.3.10.1
(`.venv/Lib/site-packages/pyebsdindex/nlpar_cpu.py`; public domain,
US Naval Research Laboratory, author David Rowenhorst) is the
numerical oracle: its two static numba kernels `NLPAR.sigma_numba`
(:752-818) and `NLPAR.nlpar_nb` (:820-936) run on in-memory arrays and
are called from tests as an optional dependency. EMsoftOO
`mod_NLPAR.f90` (BSD-3, local clone) is an equation cross-check only.
PyEBSDIndex is file-oriented and carries quirks that a kikuchipy
method must not inherit (Context, "PyEBSDIndex quirk catalogue");
every deviation from it is a recorded decision below, and parity is
asserted on float32 kernel output, never on written files.

## Scope

In scope (three stages on one branch; plan.md sections 2-4):

- **Stage A -- engine.** New private module
  `src/kikuchipy/pattern/_nlpar.py` (D1.1) with the numba kernels
  (D7), the sigma estimate (D2), distances and weights (D3), the
  shifted-inward search window with per-axis radius (D4), chunk
  wrappers and the depth helper (D8), the output dtype policy (D6);
  `EBSD.average_non_local_neighbour_patterns()` with the frozen
  signature (D1.3) for in-memory signals and `EBSD.get_nlpar_sigma()`
  (D1.4). Three Stage A guards raise `NotImplementedError` naming
  Stage B until the optimiser and the lazy path land: `lam=None`, a
  lazy input signal and `lazy_output=True` (recorded stage split,
  D5.7; the default is still `None`); `EBSD.get_nlpar_lambda()` is
  present from Stage A as a stub whose body raises the lambda
  message (the fifth guard of D5.7, so the method order of D1.2 and
  the V0 source audit hold from Stage A on; spec review F3-R3-4,
  2026-10-04). The eager path already runs
  through the chunked dask route of D8 (an in-memory signal larger
  than 8 MB is chunked by `get_dask_array`), so the depth helper and
  the chunk wrappers are Stage A deliverables, pinned in Stage A by
  the compiled PyEBSDIndex parity arms on `nickel_ebsd_large` (V2/V3,
  [A, download]; the eager Ni run is chunked `((47, 8), (47, 28))`)
  and by two pyebsdindex-free driver tests on multi-chunk dask arrays
  (V7 `TestDepthAndHalo`, spec review C2-F1).
- **Stage B -- optimisation and scale.** `EBSD.get_nlpar_lambda()`
  and `lam=None` (D5), the lazy/dask path with eager == lazy pins over
  regular, irregular and sub-depth chunkings (D8), `nickel_ebsd_large`
  in the default suite ([download]: full-map ADP/IQ, the phantom-ratio
  arm, Hough indexing on the `s.inav[::5, ::5]` subset of 165
  patterns, navigation (11, 15) rows x cols) with the full-map Hough
  weekly, `si_wafer` weekly and gated (D13), performance baselines
  recorded (V11).
- **Stage C -- tutorial.** `doc/tutorials/nlpar.ipynb`, its
  registration, the gallery example
  `examples/pattern_processing/nlpar.py`, the CHANGELOG tutorial
  bullet (D12).
- **Data**: synthetic generators (identical patterns plus Gaussian
  noise, two-grain maps, random maps with saturated and duplicate
  variants), the fixture `dummy_signal` (3, 3|3, 3)
  (`conftest.py:196-205`), `kp.data.nickel_ebsd_large`
  ((55, 75 | 60, 60) rows x cols, NumPy order; hyperspy repr
  `(75, 55|60, 60)`; uint8, min 21, max 253; re-measured 2026-10-04
  from the cached file; `_data.py:129-175`, download-backed, pooch)
  and `kp.data.si_wafer` ((50, 50 | 480, 480) uint8, 311 MB download,
  weekly and locally gated; `_data.py:392-449`; shape verified
  2026-10-04 from the cache, validation.md ledger entry 1 item 5).
  Fixtures live only under `src/kikuchipy/data/**` or are generated in
  the test; the four synthetic generators are plain functions in the
  root `conftest.py` exposed as same-named fixtures (D1.9).
- **Constitution amendments** listed in plan.md section 0 (appended
  at the end of `specs/mission.md`, `specs/roadmap.md`,
  `specs/tech-stack.md`; applied by the main loop in the spec commit).

Deliberately out of scope (each recorded with its revisit path):

- **GPU backend.** None (user decision 2, 2026-09-11). PyEBSDIndex has
  an OpenCL NLPAR; a CuPy port would follow the Phase 12 gate pattern
  if ever wanted.
- **PyEBSDIndex as a runtime dependency.** Never imported by `src/`
  for NLPAR (D10.1). #824 calls it for sigma and lambda;
  PyEBSDIndex#83 (opened 2026-10-03, 0 comments today) asks for
  array-level functions; even if answered, the fork keeps its kernels.
- **A `Window` or boolean search-window argument.** The search window
  is the full `(2 r_y + 1) x (2 r_x + 1)` rectangle (D4.1), as in the
  paper and PyEBSDIndex; hakonanes doubted a `Window` makes sense here
  (#824 review, 2026-09-27). Revisit path: a boolean mask keyword
  once a use case is measured.
- **Per-system extras of PyEBSDIndex**: `diff_offset`, `backsub`
  (Gaussian background fit), `stem_scale` (sqrt transform), `automask`
  (default circular mask), the file reader and writer, `nn > 1` sigma
  windows, `reset_sigma`, the pair memo. kikuchipy's background
  removal, `signal_mask` and `rescale_intensity` cover these jobs.
- **Navigation mask** (excluding scan points from being candidates):
  not in PyEBSDIndex, not in v1 (D9.5).
- **Edits to `doc/tutorials/hybrid_indexing.ipynb`** (never-sweep
  rule, tech-stack.md:11): its last cell points to PyEBSDIndex's NLPAR
  tutorial (`:1704-1705`); the new tutorial links to it (D12.5).
- **#824's `doc/dev/code_style.rst` note**: arrives with an upstream
  merge if #824 lands; not duplicated.
- **The upstream 0.13.1 merge**: stays parked (untracked
  `specs/_research/plan-upstream-merge-0.13.1.md`, never touched).
- **New required dependencies**: none (D10.1).

## Decisions

Each decision states what is decided, why, the alternative rejected,
and the validation.md test that pins it. (frozen) = changed only by a
dated amendment in this file; (recorded) = a fact or a policy the
build follows, amendable at a gate with a ledger entry.

### D1 -- API surface, naming and placement (frozen)

1. **Module**: `src/kikuchipy/pattern/_nlpar.py` (private, leading
   underscore), holding the kernels (D7) and the drivers (chunk
   wrappers, depth helper, optimiser). It sits next to
   `src/kikuchipy/pattern/chunk.py`, which holds the sibling
   `_average_neighbour_patterns` (`chunk.py:130-144`). Rationale:
   hakonanes' review of #824 asked for "their own module ... under
   `src/kikuchipy/pattern/_nlpar.py`" (inline comment on chunk.py:148,
   2026-09-27), and the fork converges with upstream by using the same
   path. Alternative rejected: the parked plan's
   `src/kikuchipy/signals/util/_nlpar.py` (superseded by the
   2026-10-04 decision). The module is never exported through
   `pattern/__init__.pyi` (private helpers only).
2. **Methods** in `src/kikuchipy/signals/ebsd.py`, inserted directly
   after `average_neighbour_patterns` (which ends at `ebsd.py:1122`)
   and before `downsample` (`ebsd.py:1124`), in this order:
   `average_non_local_neighbour_patterns`, `get_nlpar_sigma`,
   `get_nlpar_lambda`. The module import joins the existing
   `kikuchipy.pattern` import block (`ebsd.py:81-92`).
3. **Signature, frozen** (defaults decided in D2-D6):

   ```python
   def average_non_local_neighbour_patterns(
       self,
       search_radius: int | tuple[int, ...] = 3,
       lam: float | None = None,
       dthresh: float = 0.0,
       target_weight: float = 0.34,
       sigma: float | np.ndarray | None = None,
       signal_mask: np.ndarray | None = None,
       saturation_protect: bool = True,
       dtype_out: str | np.dtype | type | None = None,
       show_progressbar: bool | None = None,
       inplace: bool = True,
       lazy_output: bool | None = None,
   ) -> EBSD | LazyEBSD | None
   ```

   Keyword map to #824, carried in the docstring: `window_shape` =
   `(2 * r_y + 1, 2 * r_x + 1)` of our `search_radius`; `lamda` =
   `lam`; `lam=None` optimises (as #824's `lamda=None` does through
   PyEBSDIndex); #824's `window` (boolean search mask, default
   "circular") and `dask_config_kwargs` have no counterpart (D4.1,
   D8.8); our `dthresh`, `target_weight`, `saturation_protect` and
   `dtype_out` have none in #824. PyEBSDIndex names map one to one
   (`searchradius`, `lam`, `dthresh`, `saturation_protect`), its
   `mask` (1 = use) being `~signal_mask` (D9.1).
4. **Getters, frozen**:

   ```python
   def get_nlpar_sigma(
       self,
       signal_mask: np.ndarray | None = None,
       saturation_protect: bool = True,
       show_progressbar: bool | None = None,
   ) -> np.ndarray  # float32, navigation shape (row, col)

   def get_nlpar_lambda(
       self,
       target_weight: float = 0.34,
       dthresh: float = 0.0,
       sigma: float | np.ndarray | None = None,
       signal_mask: np.ndarray | None = None,
       saturation_protect: bool = True,
       show_progressbar: bool | None = None,
   ) -> float
   ```

   `get_nlpar_sigma` runs exactly the pass 1 that
   `average_non_local_neighbour_patterns(sigma=None)` runs (same
   kernel, same arguments), so a user can inspect the sigma map and
   pass it back through `sigma=` without recomputation.
   `get_nlpar_lambda` runs pass 1 (or takes `sigma`) and the D5
   optimiser and returns the fitted value; the method with `lam=None`
   calls the same function and logs the value (D5.6).
5. **Return and in-place contract** mirrors `average_neighbour_patterns`
   (`ebsd.py:1018-1019`, `:1095-1122`): `lazy_output=True` with
   `inplace=True` raises `ValueError`; `return_lazy = lazy_output or
   (lazy_output is None and self._lazy)`; `inplace=True` returns
   `None` and replaces the data (eager input: `store(self.data,
   compute=True)` when `dtype_out` equals the data dtype, else
   `self.data = result.compute()` as `downsample` changes the dtype in
   place, `ebsd.py:1195-1223`; lazy input with `lazy_output` `None` or
   `True`: `self.data = result.rechunk(old_chunks)`); `inplace=False`
   returns `EBSD` or `LazyEBSD` with `**self._get_custom_attributes()`
   (`ebsd.py:1111`), computed when not lazy; progress bar as at
   `ebsd.py:1096-1101`. Two cases decided 2026-10-04 (spec review
   C2-F14/C2-F15) that the `average_neighbour_patterns` precedent
   leaves open: (a) a LAZY input with `lazy_output=False` and
   `inplace=True` computes to NumPy and assigns, `self.data =
   result.compute()`, so the signal becomes eager (as HyperSpy's
   `compute()` does; `store` into a dask-array target is never
   attempted); pin: the [B] arm of V7
   `test_inplace_lazy_output_contract` (`isinstance(s.data,
   np.ndarray)`, bitwise equal to the eager run). (b) Eager in-place
   with an unchanged dtype stores into `self.data` while the graph's
   chunks are views of that same buffer (`da.from_array`,
   `_dask.py:159`); on a multi-chunk in-memory signal a halo read
   scheduled after a neighbour's store would read averaged data if the
   scheduler ever interleaved them. Measured 2026-10-04 (spec review
   round 3, C3-R3-F3; critic C3's `scratchpad/critic3_store_race_
   probe.py`, re-run by the fixer): the identical mechanism in
   `average_neighbour_patterns` on a `((47, 8), (47, 28))`-chunked
   in-memory (55, 75 | 60, 60) uint8 signal is bitwise equal to
   `inplace=False` under the synchronous scheduler and under threads
   with 1, 2 and 20 workers (0 differing patterns), as is a toy
   `map_overlap` + `store` into its own source array. It is pinned by
   V7 [A, download]
   `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`
   (`nickel_ebsd_large`, default chunking `((47, 8), (47, 28))`, one
   deterministic `dask.config.set(scheduler="synchronous")` arm and
   one threaded arm, bitwise against `inplace=False`). Fallback if it
   ever differs: eager in-place assigns `self.data = result.compute()`
   for every dtype (the `downsample` precedent), recorded as a dated
   amendment here.
6. **Argument validation** (amended 2026-10-04 with the message
   fragments the tests match, spec review E1-F12/C1-F11/E1-F23):
   `search_radius` is an `int >= 0` or a tuple of them with one entry
   per navigation axis in the order of `self._navigation_shape_rc`
   (row, col); a radius is valid if `isinstance(r, (int, np.integer))
   and not isinstance(r, bool) and r >= 0` (NumPy integer scalars
   accepted, `bool` rejected; spec review E2-F15); anything else
   raises `ValueError` ("search_radius must be a non-negative int" for
   a negative, boolean or non-integer value such as `-1`, `True` or
   `1.5`; "one radius per navigation axis" for a tuple of the wrong
   length). All radii 0 warns ("no averaging is therefore performed",
   mirroring `ebsd.py:1028-1034`) and returns `None`.
   `navigation_dimension == 0` raises `ValueError` ("nothing to
   average") and this check runs FIRST within step (2), before the
   radius broadcast (an `int` radius broadcast to `()` would otherwise
   pass). `lam` given must be `> 0` ("lam must be > 0"); `dthresh >= 0`
   ("dthresh must be >= 0"); `0 < target_weight < 1` ("0 <
   target_weight < 1"); `sigma` a scalar `numbers.Real` (not `bool`)
   that is `> 0` ("sigma must be > 0"), or an array of the navigation
   shape ("navigation shape", D2.6) whose every element is finite and
   `> 0` after the cast to float32, checked with `np.all(np.isfinite(
   sigma)) and np.all(sigma > 0)` ("sigma must be > 0"; spec review
   E2-F9: a zero, negative or NaN element would otherwise reach
   `_nlpar_normalized_distances` as a 0/0). `dtype_out` must be
   `None` or an integer or floating dtype (`np.issubdtype(dt,
   np.integer) or np.issubdtype(dt, np.floating)`), else `ValueError`
   ("dtype_out must be an integer or floating dtype"; D6.2).
   `signal_mask` of the signal shape ("signal shape"), converted
   with `np.asarray(signal_mask, dtype=bool)` after the shape check
   (integer 0/1 masks are accepted, as in `dictionary_indexing`); a
   mask excluding every pixel raises ("excludes every pixel", D9.1).
   The same fragments apply in `get_nlpar_sigma` and
   `get_nlpar_lambda` where the keyword exists. Check order in the
   method: (1) `lazy_output and inplace` -> `ValueError` (D1.5 text);
   (2) this validation, 0-D first; (3) all radii 0 -> warn and return
   `None`; (4) the Stage A guards of D5.7, so the `ValueError`
   contracts are testable in Stage A. Pin: V7
   `test_argument_validation`.
7. **Docstring**: numpydoc; summary "Average patterns with their
   non-local neighbours within a search window, weighted by pattern
   similarity (NLPAR)."; Parameters, Returns, See Also
   (`average_neighbour_patterns`, `get_nlpar_sigma`,
   `get_nlpar_lambda`), Notes with (a) equations (1)-(3) in plain
   text, (b) the paragraph "Differences from PyEBSDIndex and from
   upstream PR #824" whose list is frozen in D12.6, (c) an
   acknowledgement of PyEBSDIndex and the US Naval Research Laboratory
   (D10.4), (d) `:cite:`brewick2019nlpar`` (hakonanes' direction on
   #824; the key exists, `bibliography.bib:11-19`). The keyword map
   of D1.3 goes in the Notes.
8. **Style**: type hints `X | Y` and `bool | None`, never `Optional`
   or `Union` (hakonanes 2026-09-27; rmz-oz 2026-09-30 reported the
   `Union` `NameError` at collection on py3.10 and 3.13).
   `from __future__ import annotations` is already at `ebsd.py:20`
   and is added to `_nlpar.py` so no annotation is evaluated at import
   (the #824 class of `da`/`Union` not being imported). No `print` in
   `src/` or tests; the module logs through
   `_logger = logging.getLogger(__name__)` (precedent `_data.py:40`).
9. **Test modules** (recorded; amended 2026-10-04, spec review
   F1-F1/C1-F1/E1-F3): `tests/test_signals/test_util/test_nlpar.py`
   (kernel flags, `py_func` parity, PyEBSDIndex oracles, depth rule,
   `block_info` wrappers, optimiser) and
   `tests/test_signals/test_ebsd_nlpar.py` (signal-method contracts,
   NumPy transcription oracles, real-data effects). The kernel module
   lives in `pattern/`, but its tests keep the parked plan's location
   so that the single root `tests/test_signals -k nlpar` of every
   gate command, the coverage command and the oldest-matrix recipe
   (D10.5) collects both modules, and so that the approved session
   plan's verification grep (which names this path) holds. The
   drafted alternative `tests/test_pattern/test_nlpar.py` is
   withdrawn. validation.md is the naming authority for test classes
   and test names; plan.md section 6 quotes them verbatim. Shared test
   code (decided 2026-10-04, spec review E2-F1/F2-F3): the repository
   runs pytest with `--import-mode=importlib` (`pyproject.toml:167`)
   and `tests/` has no `__init__.py`, so one test module cannot import
   another (a sibling `from test_nlpar import ...` raises
   `ModuleNotFoundError` at collection); the four synthetic generators
   are therefore plain functions in the root `conftest.py` (the
   `dummy_signal` precedent, `conftest.py:195-205`), each exposed by a
   same-named fixture returning the callable, and both modules take
   them as fixtures; the `pyebsdindex_kernels` warm-up fixture stays
   module-scoped in `test_nlpar.py`, the only module that calls the
   oracle kernels. The one MTP placeholder asserted in BOTH modules,
   `EXP_KERNEL_ULP` (V0 `py_func` parity of the `exp` kernel and V1
   `test_weight_formula_on_injected_distances`), is likewise defined
   once in the root `conftest.py` beside the generators and exposed
   by the fixture `exp_kernel_ulp` (decided 2026-10-04, spec review
   F3-R3-3); every other placeholder lives in the one module that
   asserts it. Pins: V0
   (`KERNEL_NAMES` lists every `@njit` kernel of `_nlpar` via the
   `_njit_kernel_names` helper, `test_spherical_euler.py:72-85`), V7
   (inplace/lazy_output/radius/sigma-argument/0-D/1-D contracts).

### D2 -- Sigma estimation (frozen)

1. **Estimate**, equation (1): for pattern `i` with the up-to-8
   neighbours `j` of its 3 x 3 neighbourhood,
   `sigma_i^2 = min_j [ sum_k (p_ik - p_jk)^2 / (2 n_ij) ]`, where
   the sum runs over the pixel pairs `k` that are unmasked (D9.1) and,
   with `saturation_protect=True`, both below the saturation
   threshold (D3.5), and `n_ij` counts those pairs. The sigma kernel
   reproduces `NLPAR.sigma_numba` (nlpar_cpu.py:752-818) with one
   recorded deviation (spec review C2-F5, 2026-10-04): our `n2`
   starts at `np.float32(0.0)` and grows by `np.float32(1.0)` per kept
   pair (:791-793), where the oracle seeds `n2 = np.float32(1.0e-12)`
   (:785), a guard against a 0/0 that the `d2 > 0` guard (D2.3)
   already excludes; the oracle's `s0 = d2 / np.float32(n2 * 2.0)`
   (:797) rounds the seed away for every `n2 >= 1`, so `sigma` stays
   bitwise, and a zero count is exactly the `n2 == 0` convention of
   D2.5 and D3.4 (the oracle's `nlpar_nb` itself starts at
   `np.float32(0.0)`, :903). Then the minimum over neighbours
   (:798-799), `sigma = sqrt(min)` (:804), float32 throughout (D7).
   Rationale: the paper's estimate; the minimum over neighbours is
   robust at grain boundaries because any one same-grain neighbour
   reveals the noise level, and the division by `2n` accounts for the
   two independent noise realisations in a difference.
2. **Window**: the 3 x 3 neighbourhood, CLIPPED at the map borders
   (`sigma_numba` :770-774: `nn_r_start = max(j - nn, 0)`,
   `nn_r_end = min(j + nn, nrows - 1) + 1`), so border patterns see 3
   or 5 neighbours. Not shifted (contrast D4.2): for a minimum, fewer
   candidates only risk a slightly high estimate at borders, and
   parity with the oracle is bitwise only with clipping. `nn` is
   fixed at 1 (PyEBSDIndex's default, `calcsigma_cpu(nn=1)`, :186;
   not exposed; recorded revisit path: a keyword if measured useful).
3. **Duplicate guard**: a neighbour enters the minimum only if
   `d2 > 0` (D3.3; PyEBSDIndex `d2 >= 1.e-3`, :796, "sometimes EDAX
   collects the same pattern twice"). Identical on integer-valued
   data.
4. **Fallback**: when no neighbour passes the guard (constant map,
   duplicated neighbourhood, every pair excluded), the minimum keeps
   its initial `1e24` and `sigma = 1e12` (:776, :804). Derived
   consequence for pairs with `n_ij > 0`: every neighbour then gets
   `d ~ -sqrt(n/2) < 0`, so weight 1, and the pattern becomes the
   plain mean of its window. On a CONSTANT map this route is real
   only with `saturation_protect=False`: under protection every pixel
   equals the global maximum, every pair has `n_ij = 0` (D3.5) and
   takes the weight-0 route of D3.4, so the pattern is returned
   unchanged (recorded 2026-10-04, spec review C3-R3-F1). Kept for
   parity; V1 pins the constant-map identity in both modes.
5. **Pass 1 output** (amended 2026-10-04, spec review F1-F2/C1-F2/
   E1-F2, adopting plan.md's refinement): the kernel
   `_nlpar_sigma_kernel` returns `sigma` (float32, navigation shape)
   and the RAW per-slot accumulators of the 3 x 3 neighbourhood in a
   FIXED nine-slot layout, slot `= (dj + 1) * 3 + (di + 1)` with `dj`
   the row offset and `di` the column offset in {-1, 0, 1}: `d2`
   (float32, `(..., 9)`), `n2` (float32, `(..., 9)`, the kept-pair
   count) and `valid` (bool, `(..., 9)`, False for out-of-map slots,
   the phantom-free input of D5.2). The NORMALISED distances `d`
   (equation (2) with the neighbours' sigmas) are produced by the
   NumPy driver `_nlpar_normalized_distances(d2, n2, valid, sigma)
   -> d float32 (..., 9)` on the assembled whole-map arrays,
   reproducing the compiled oracle's second loop (`sigma_numba`
   :806-817) operation for operation: `num = d2 - n2 * s2_ij` in
   float32 (`s2_ij = sigma_i^2 + sigma_j^2`, float32), `den =
   float64(s2_ij) * sqrt(float64(2.0) * float64(n2))` in float64
   (numba promotes `2.0 * n2` to float64 there), `d = float32(
   float64(num) / den)`. Conventions: the self slot has `n2 =
   n_kept`, `valid = True` and `d = -inf` (weight exactly 1 through
   `exp(-max(-inf - dthresh, 0)) = 1`); a slot with `n2 == 0` gets
   `d = +inf` (weight 0, D3.4); out-of-map slots keep `valid = False`
   and are never read. PyEBSDIndex differs in three recorded ways
   that V2 masks: it enumerates in-map neighbours COMPACTLY (row
   outer, column inner, :779-802), so `ours_d[valid]` in slot order is
   its slot order; its self slot carries `nout = npix` (the TOTAL
   pixel count, mask-independent, :761, :786) and `dout = -sqrt(npix /
   2)`; a visited pair without a kept pixel keeps `nout = 1e-12 > 0`
   (:785) and a finite tiny negative `dout` (weight 1 in its
   objective). Why the split: pass 1 can then run at depth 1 (D8.2),
   and a user-supplied `sigma` (D1.4, D2.6, D5.6) feeds the lambda
   objective through the same driver without re-running the kernel.
   The drafted alternative (normalisation inside the kernel at depth
   2) is withdrawn; the bitwise `dout` oracle is retained because the
   driver performs the same float32/float64 operations on the same
   values (pinned by V2).
6. **`sigma` argument**: `None` (default) runs pass 1; a float gives a
   constant sigma map (PyEBSDIndex accepts a scalar too, :379-382); an
   array of the navigation shape is used as is (cast to float32);
   another shape raises `ValueError`. `sigma=1e-6` is the V1 identity
   probe (self weight 1, every neighbour underflows to 0).
7. Alternatives rejected: #824's fallback `np.std` of each pattern
   (intensity spread, not noise; also unreachable code, Context);
   EMsoftOO's integer-truncated sigma (Context); a per-pattern sigma
   from a smooth background model (not in the paper).
   Pins: V2 (bitwise vs `sigma_numba` on synthetic generators and
   nickel_ebsd_large raw and background-corrected, masks none and
   circular, protection on and off; fallback MTP ulp pin; clipped
   3 x 3 asserted against a shifted variant), V4 (iid-noise recovery
   band for `sigma_hat / sigma_true`), V1 (sigma = 1e-6 identity;
   constant-map identity), V10 (duplicates; `n2 == 0`).

### D3 -- Normalised distance and weights (frozen)

1. **Distance**, equation (2): for a pair `(i, j)` in the search
   window, over the kept pixel pairs `k` (D9.1, D3.5) with count
   `n_ij`,
   `d_ij = [ sum_k (p_ik - p_jk)^2 - n_ij (sigma_i^2 + sigma_j^2) ]
   / [ (sigma_i^2 + sigma_j^2) sqrt(2 n_ij) ]`,
   i.e. the squared distance minus its noise expectation, in units of
   the noise standard deviation of that difference. `nlpar_nb`
   :901-913: `d2 -= n2 * (sigma0 + sigma1)`;
   `dnorm = (sigma1 + sigma0) * sqrt(2 n2)`; `d2 /= dnorm`.
2. **Weight**, equation (3):
   `w_ij = exp( -max(d_ij - dthresh, 0) / lam^2 )`, `w_ii = 1`, and
   the averaged pattern is `p_i' = sum_j w_ij p_j / sum_j w_ij` over
   the search window (`nlpar_nb` :921-934). The self weight is forced
   to 1 (PyEBSDIndex sets the self distance to `-1e6`, :892-893, so
   that `max(-1e6 - dthresh, 0) = 0`). `dthresh` default `0.0`
   (PyEBSDIndex default, :51): distances below `dthresh` count as
   zero, raising the weights of near-identical neighbours.
3. **Duplicate guard `d2 > 0`** in the sigma minimum (D2.3), replacing
   PyEBSDIndex's absolute `d2 >= 1e-3`. Identical on integer-valued
   data (a sum of squared integer differences is 0 or >= 1) and
   scale-free: on float data in [0, 1] the absolute guard can reject
   every genuine neighbour and degrade the pattern to the D2.4 box
   average. Recorded deviation. Pins: V1 power-of-two scale
   invariance (bitwise: scaling the input by 2^k scales sigma by 2^k
   and leaves `d`, the weights and the normalised output unchanged),
   V10 duplicate-neighbour oracle.
4. **`n2 == 0` policy**: a pair with no comparable pixel (every pair
   masked or saturated) gets `d = +inf`, weight 0; PyEBSDIndex gives
   it weight 1 (`dnorm = 0 <= 1e-8` leads to `d2 = 1e6 * 0`,
   :912-915). No evidence of similarity, no averaging. The oracle's
   other branch is kept: `dnorm <= 1e-8` with `n2 > 0` gives
   `d = 1e6 * n2` (weight underflows to 0). Recorded deviation. Pins:
   V10 `n2 == 0` oracle; V3 saturation arm (per-pair `n2` vs global N).
5. **Saturation protection**: with `saturation_protect=True`
   (default, as PyEBSDIndex :53) a pixel pair is kept only if both
   values are strictly below the threshold, stated ONCE here for all
   three documents (spec review C2-F9/E2-F12, 2026-10-04): sigma
   kernel threshold `np.float64(max_value) * np.float64(0.9961)`,
   compared against the float32 pixels in float64 (the compiled oracle
   unifies its `mxval *= 0.9961` to float64, :763-767); averaging
   kernel threshold `max_value * np.float32(0.999)` in float32
   (:865-869); `saturation_protect=False` sets `np.float64(max_value)
   + np.float64(1.0)` and `max_value + np.float32(1.0)` respectively,
   excluding nothing (:764-765, :866-867). The two factors are the
   module constants `SIGMA_SATURATION_FACTOR = 0.9961` (a Python
   float, it meets a float64 operand) and `AVERAGE_SATURATION_FACTOR =
   np.float32(0.999)`, read by the kernels and pinned by V10
   `test_sigma_fallback_value_is_1e12`. `max_value` is the GLOBAL
   data maximum, computed once per call (one reduction pass for lazy
   signals). PyEBSDIndex takes `np.max(data)` per
   tile (:763, :865), so its result depends on the tiling; our tests
   run the oracle on whole arrays, so parity holds (recorded
   deviation). Alternative rejected: the dtype maximum (255 for
   uint8), which excludes nothing when the camera never reaches full
   scale (nickel_ebsd_large tops at 253) and breaks parity.
   Consequence recorded 2026-10-04 (spec review C3-R3-F1): on a
   CONSTANT map every pixel equals the global maximum, so with
   protection on the strict `<` excludes every pixel, every pair has
   `n_ij = 0`, sigma takes the 1e12 fallback (D2.4), every non-self
   weight is exactly 0 (D3.4) and the pattern is returned unchanged
   (no averaging); PyEBSDIndex gives those pairs weight 1 (`1e6 * 0 =
   0`) and returns the box mean, which on a constant map is the same
   pattern by another route (compiled oracle probe 2026-10-04 on a
   (4, 5 | 6, 6) map at 100: `nout` stays at its `1e-12` seed, sigma
   all `1e12`). With protection off every pair is kept and the D2.4
   box-mean route is real. Pins: V10 global saturation rule and the
   uint16 two-threshold arm (a mutant using one constant for both
   kernels dies), V3 saturation arm, V1 constant-map arms (both
   protection modes).
6. **Weight normalisation**: the weights are divided by their sum
   over the window including the self weight, and the normalised
   weights multiply the float32 patterns in window order (D7.3).
   Pins: V1 weight formula on injected `d` with `dthresh` in
   {0, 0.5}; V3 bitwise parity; V4 (normalised `d` mean and standard
   deviation bands on iid noise, fraction of weights exactly 1,
   noise-reduction vs the weight-derived expectation, monotonicity in
   `lam`); V5 (two-grain: cross-boundary weights exactly 0, derived
   `d ~ 159` at a contrast step of 30 grey levels, sigma 8, N 1024).

### D4 -- Window and edge policy (frozen)

1. **Search window**: the full rectangle of `(2 r_y + 1) x (2 r_x + 1)`
   scan points centred on the pattern, `search_radius` per navigation
   axis (an `int` applies to every axis; a tuple gives one radius per
   axis in (row, col) order). Default 3 (PyEBSDIndex default, :50).
   No circular or weighted mask: the paper and PyEBSDIndex use the
   full square, and a boolean window would add a keyword whose benefit
   is unmeasured (hakonanes' doubt on #824; #824's default "circular"
   window drops the four corners of the 7 x 7, a recorded difference).
2. **Edge policy for the search window: shifted inward.** A pattern
   closer than `r` to a map border keeps a full window by sliding it
   inward (`nlpar_nb` :872-873, :877-878: `winstart = max(i - sr, 0)
   - max(i + sr - (n - 1), 0)`, `winend = min(i + sr, n - 1) +
   max(sr - i, 0) + 1`), so every pattern sees exactly `(2r + 1)`
   candidates per axis. Our kernel `_window_bounds(center, radius, n,
   shift) -> (start, stop)` (signature amended 2026-10-04 to plan.md
   module 2.1's four-argument form, spec review F1-F15/C1-F16/E1-F5)
   returns, for `shift=True`, PyEBSDIndex's expressions `start =
   max(center - radius, 0) - max(center + radius - (n - 1), 0)` and
   `stop = min(center + radius, n - 1) + max(radius - center, 0) + 1`,
   both clamped to `[0, n]`; this equals the closed form `start =
   max(0, min(center - radius, n - 2 radius - 1))`, `stop = start +
   2 radius + 1` whenever `n >= 2 radius + 1`, and the whole axis
   `(0, n)` when `n < 2 radius + 1` (PyEBSDIndex's expressions then
   index out of bounds with no bounds check; quirk not ported).
   `shift=False` returns the CLIPPED window `[max(center - radius, 0),
   min(center + radius, n - 1) + 1)` used by the sigma pass (D2.2).
   Pin: V0 `test_window_bounds_matches_pyebsdindex_expressions_and_
   clamps`. Alternatives
   rejected: (a) clipping: fewer candidates at the border and a band
   of weaker denoising; (b) zero extension with window sums, the
   `average_neighbour_patterns` mechanism (`ebsd.py:1040-1044`): a
   zero pattern is maximally dissimilar and wastes candidates; (c)
   #824's edge replication (`np.pad(mode="edge")`, diff lines
   267-277): the copies of a border pattern have `d = 0` and weight 1
   each, so the border pattern over-weights itself.
3. **Edge policy for the sigma window: clipped** (D2.2). The two
   policies differ on purpose and are pinned separately (V2 clipped
   vs shifted variant; V3 border band vs clamp and zero-extend
   alternatives must differ while the interior is identical).
4. **Tiny maps**: an axis shorter than the window makes the window the
   whole axis (every pattern averages over the full extent in that
   direction); a navigation axis of length 1 is the 1-D case (D9.4);
   both radii 0 is the warned no-op (D1.6). Pins: V7 map smaller than
   the window; V1 `lam = 1e6` equals the shifted-window box mean
   (every weight 1 for finite `d`), computed with a test-local
   reference that implements the shift explicitly.

### D5 -- Lambda optimisation (frozen)

1. **Objective**: `F(lam) = mean_i | tw - 1 / S_i(lam) |` with
   `S_i(lam) = 1 + sum_{j in W3(i), j != i, valid} exp(-max(d_ij -
   dthresh, 0) / lam^2)` (the self term is exactly 1; PyEBSDIndex's
   `+ 1e-12` guard on the weight sum, :109, is not reproduced since
   `S_i >= 1` by construction; recorded 2026-10-04, spec review
   E1-F18), over the scan points `i`, where `W3(i)` is the CLIPPED
   3 x 3 neighbourhood of `i` including `i` itself (`w_ii = 1`) and
   `d_ij` are the normalised distances of pass 1 (D2.5). `1 / S_i` is
   the normalised weight the pattern gives itself in its 3 x 3
   neighbourhood; `tw = target_weight` (default 0.34) asks that a
   pattern keep about a third of the weight. Rationale: this is
   PyEBSDIndex's `loptfunc` (:106-112) with two corrections below;
   the paper leaves lambda to the user. **Summation form and dtypes,
   frozen** (spec review E2-F8, 2026-10-04, so that V6's EXACT
   equality with a test-local `loptfunc` is well defined):
   `_nlpar_lambda_objective(lam, d, valid, dthresh, target_weight) ->
   float` receives `lam` FIRST, as the float64 array of shape `(1,)`
   that `scipy.optimize.minimize` passes (never cast to a Python
   float); `w = np.exp(-np.maximum(d - dthresh, np.float32(0.0)) /
   lam ** 2)` (float64 by array promotion of the float32 `d` with the
   float64 `lam`), `w[~valid] = 0.0` (the self slot is exactly 1.0
   through `d = -inf`; a slot with `n2 == 0` is exactly 0.0 through
   `d = +inf`), `S = w.sum(axis=-1)` over all nine slots in slot
   order, `F = float(np.mean(np.abs(target_weight - 1.0 / S)))`.
   **Stride** (spec review C2-F12): the objective has no stride
   argument; `_nlpar_optimize_lambda(d, valid, target_weight,
   dthresh)` applies `d[::2, ::2]`, `valid[::2, ::2]` BEFORE calling
   the objective when `d.shape[0] * d.shape[1] >= 1e6` (PyEBSDIndex's
   `stride`, :164; pinned by V6 `test_stride_above_1e6_points`).
2. **Phantom-free**: slots outside the map are excluded from `S_i`
   through the presence mask of D2.5. PyEBSDIndex counts them: its
   `dout` keeps `0.0` in the unvisited trailing slots (:756, :800) and
   `exp(0) = 1` enters the sum, so border patterns look more
   self-similar than they are. On nickel_ebsd_large (55 x 75) this is
   776 of 37125 slots (4 corners x 5 + 146 edge points x 3 + 106 edge
   points x 3; arithmetic re-done 2026-10-04) and shifts lambda by
   about +2 % at `tw = 0.34` (seed, D13.3). Recorded deviation,
   measured and documented at the Stage B gate (V6).
3. **Consistent `dthresh`**: the objective uses the averaging kernel's
   `max(d - dthresh, 0)` (:924), not `loptfunc`'s `max(d2, dthresh)`
   (:107), which floors distances instead of shifting them and so
   fits a lambda for a different weight than the one applied. The two
   coincide at `dthresh = 0`. Recorded deviation. (#824 copies
   `loptfunc` and additionally drops `dthresh` from its averaging
   weights, diff lines 338-341.)
4. **Optimiser**: `scipy.optimize.minimize(F, x0=1.0,
   method="Nelder-Mead", bounds=[(1e-3, 10.0)], options={"fatol":
   1e-4})`, PyEBSDIndex's call (:167-169). Bounded Nelder-Mead needs
   SciPy >= 1.7, the declared floor (`pyproject.toml:64`). A result
   within 1 % of either bound, i.e. `lam <= 1.01e-3` or `lam >= 9.9`,
   raises `UserWarning(f"NLPAR lambda optimisation hit the {which}
   bound ({lam:.4f}); the target weight {tw} is not supported by the
   data")` with `which` in {"lower", "upper"} (rule and message made
   explicit 2026-10-04, spec review C1-F19/E1-F24; corrected
   2026-10-04, spec review F2-F1/C2-F3: maps whose weights barely
   depend on lambda, such as every non-self slot at `d = 200`, leave
   bounded Nelder-Mead at its start `x0 = 1.0` because the objective
   is flat there in float64 (measured: `x = 1.000000`, `nit = 10`);
   the bound warning fires only when the data DRIVES the fit to a
   bound, i.e. when the self weight cannot reach the target inside
   `[1e-3, 10]`, as in the two constructed V6 arms). Alternative
   rejected: a
   scalar root find on the monotone `mean(1/S_i) - tw` (cleaner, but
   not what the oracle does; V6's exact equality against a test-local
   `loptfunc` would be lost).
5. **One target instead of PyEBSDIndex's median of three**
   (`target_weights=(0.5, 0.34, 0.25)`, median taken, :94, :179-181).
   Lambda decreases monotonically with `tw`, so the median of the
   three fits is the `0.34` fit; `get_nlpar_lambda(target_weight=...)`
   exposes the whole curve and the tutorial plots it. Default
   `target_weight=0.34` for parity with PyEBSDIndex's `auto_nlpar`.
6. **Reporting**: the INFO record "NLPAR: optimised lambda
   {value:.4f} for target weight {tw} (objective {F:.2e})" is emitted
   INSIDE `_nlpar_optimize_lambda` through `_logger`, so
   `get_nlpar_lambda()` and `lam=None` each emit exactly one record
   per call (spec review E2-F17; the V6 caplog test captures only the
   `lam=None` call inside `caplog.at_level`); no `print`. The
   optimiser uses the same `sigma`,
   `signal_mask` and `saturation_protect` as the averaging that
   follows (PyEBSDIndex recomputes sigma inside `opt_lambda_cpu`,
   :156, with its own arguments).
7. **Stage split** (recorded; extended 2026-10-04, spec review
   C1-F4/F1-F11/E1-F20, and again 2026-10-04, F3-R3-4): Stage A ships
   the frozen signature with three guards raising
   `NotImplementedError`, checked AFTER the argument validation of
   D1.6: `lam is None` -> "lambda optimisation lands in Stage B; pass
   lam=..."; a lazy input signal or `lazy_output=True` -> "lazy NLPAR
   lands in Stage B". `get_nlpar_sigma` raises the lazy message on a
   lazy input (the fourth guard). `get_nlpar_lambda` is present with
   its frozen D1.4 signature as a stub whose body raises the lambda
   message BEFORE any validation (the fifth guard; its
   argument-validation arms in V7 are therefore [B]). Stage B
   deletes all five guards and their pin (V7
   `test_stage_a_guards_raise_not_implemented`, deleted in Stage B).
   The lambda objective and optimiser, `TestLambdaOracle`,
   `TestLambdaMethod` (V6), the real-data effects (V8, V9) and the
   lazy arms of V7 and V10 are Stage B; the V10 `dthresh`-consistency
   test needs the objective and is tagged [B] inside V10.
   Pins: V6 closed-form check on a constructed `d` field (every
   non-self slot at `d = c`, `c` in {2, 5, 12}, `valid` all True:
   `lam(tw) = sqrt(-c / ln((1/tw - 1) / 8))`, derived from `1 / (1 +
   8 e^{-c/lam^2}) = tw`), exact equality with a test-local `loptfunc`
   re-implementation (with the `1e-12` term dropped and the averaging
   form of `dthresh`) on PyEBSDIndex's own `dout` and presence mask,
   end-to-end on nickel_ebsd_large (seeds in D13.3), lambdas ascend
   with decreasing target, bound-hit warning, stride rule, INFO
   logging on a synthetic map, `target_weight=0.5` forwarded through
   `lam=None`; V4 monotonicity of noise reduction in `lam`.

### D6 -- Output dtype policy (frozen)

1. **Default `dtype_out=None`**: the input dtype. Integer inputs:
   `np.rint` of the float32 average, then `np.clip` to the dtype's
   range from `skimage.util.dtype.dtype_range` (imported at
   `ebsd.py:44`), then the cast. Float inputs: the float32 average
   cast to the input float dtype (a float64 input is accumulated in
   float32 and cast up; documented precision note, D7.2).
2. **`dtype_out="float32"`** (or `np.float32`) returns the raw
   average. `dtype_out` must be an integer or floating dtype
   (`np.issubdtype(dt, np.integer) or np.issubdtype(dt,
   np.floating)`), else `ValueError("dtype_out must be an integer or
   floating dtype")` (bool, complex and object are rejected; spec
   review E2-F16, 2026-10-04): integer targets get `rint` and a clip
   to that dtype's range, float targets a cast; no rescaling ever (a
   user who wants another range calls `rescale_intensity` afterwards;
   recorded).
3. **No per-pattern rescale.** Weights summing to 1 give a convex
   combination that stays inside the input range, so nothing needs
   rescaling to fit the dtype; rescaling would destroy the
   cross-pattern intensity comparability NLPAR preserves and tie the
   result to each pattern's extreme pixels. PyEBSDIndex does not
   rescale by default (`rescale=False`, :285, :516-521);
   `average_neighbour_patterns` rescales (`ebsd.py:966-968`,
   `chunk.py:156-163`) because arbitrary window coefficients change
   the range; #824 rescales (diff lines 354-360). Recorded, not
   reproduced.
4. **Round, do not truncate.** A truncating cast shifts the mean by
   -0.5 grey level and biases every downstream intensity statistic.
   PyEBSDIndex's writer path (`flt2int='clip'`, :528) is not relied
   on: parity is asserted on the float32 kernel output (V3), and the
   integer round trip is pinned separately.
   Pins: V7 uint8/uint16/float32/float64 round trip with `rint`
   (bitwise against `np.rint(float32_result).clip(...)`); V3 "zero
   pixels differ by >= 2 grey levels" after rounding against the
   oracle's float32 output rounded the same way; the plan.md mutant
   "floor for rint" dies by the round-trip test.

### D7 -- Kernels and numeric discipline (frozen policy, parity level MTP)

1. **Kernels**: `_window_bounds`, `_nlpar_sigma_kernel` (sigma plus
   the RAW 3 x 3 accumulators `d2`, `n2` and the presence mask
   `valid`; the normalisation is the NumPy driver
   `_nlpar_normalized_distances`, not a kernel, D2.5 as amended
   2026-10-04), `_nlpar_distances_kernel` (window distances for the
   kept region of a chunk), `_nlpar_weights_kernel` (the only kernel
   that calls `exp`), `_nlpar_weighted_sum_kernel`. The implementation
   may split differently, but every `@njit` dispatcher defined in
   `_nlpar.py` must appear in the test's `KERNEL_NAMES`, and the `exp`
   call stays isolated in one kernel so the others are bitwise
   `py_func`-testable.
   Flags: `@njit(cache=True, nogil=True)`; no `parallel=True`, no
   `fastmath` (tech-stack.md:44; the spherical kernel-flag test
   pattern, `test_spherical_euler.py:509-529`). PyEBSDIndex's kernels
   are `parallel=True` (:753, :821); here parallelism is dask over
   chunks with the threaded scheduler, and `nogil` lets the threads
   run.
2. **Accumulation**: patterns enter the kernels as float32 (the
   oracle reads with `convertToFloat=True`, :254-255); `d2` and `n2`
   are float32 accumulators over the kept pixels in ascending flat
   index order (:788-793, :904-909); the weighted sum accumulates
   float32 products in window order after the weights are normalised
   (:929-934). float64 accumulation is rejected: it breaks the bitwise
   oracle (plan.md mutant), and the paper's statistics are not
   precision-limited at float32 for N <= 480 x 480.
3. **Literal discipline**: every constant that meets a float32 operand
   is an explicit `np.float32(...)`; no bare Python literals in mixed
   expressions, so that `py_func` (NumPy promotion, changed by NEP 50
   in NumPy 2) and the compiled kernel (numba promotion) agree under
   NumPy 1.23 (CI oldest pin, `tests.yml:48`) and NumPy 2.4 alike.
   Where the oracle's own promotion is reproduced (`lam2 = 1.0 /
   lam**2`, :847; `np.exp(-1.0 * w * lam2)`, :925; `mxval *= 0.9961`,
   :767) the operand types are written out and recorded in the kernel
   docstring as the parity contract; the Stage A gate records the
   measured sequence that achieves parity. Extended 2026-10-04 (spec
   review E1-F8/E1-F9, measured with numba `nopython_signatures`):
   (a) no `**` on a float32 operand anywhere in a kernel (numba types
   `float32 ** int32` as float32, NumPy as float64): squares are
   written as products, `diff = d0 - d1; d2 += diff * diff`, `s2 =
   sigma * sigma` (numba lowers an integer power 2 to the same
   rounded product, so `nlpar_nb`'s `np.float32(d0 - d1) ** np.int32(
   2)`, :909, is matched bitwise); (b) counters increment by
   `np.float32(1.0)` (numba promotes `n2 += 1.0` to float64,
   measured: the compiled `sigma_numba` counter is `64.000000000001`);
   (c) every intermediate the COMPILED oracle holds in float64 is
   written with explicit `np.float64(...)` casts and cast back to
   float32 on store: the `dout` normalisation denominator (D2.5),
   `lam2` and the exponent (D3.2); (d) the parity target is the
   compiled oracle, whose own `.py_func` is NOT bitwise with it
   (D10.2). `lam` typing: the method casts `lam = float(lam)` and
   `dthresh = np.float32(dthresh)` before any kernel; the weights
   kernel takes `lam` as float64 and computes `lam2 = np.float64(1.0)
   / (lam * lam)`, `w = max(d - dthresh, np.float32(0.0))` in float32,
   `exp(np.float64(-1.0) * np.float64(w) * lam2)` in float64, stored
   float32. The oracle is called with the same Python-float `lam` and
   `np.float32(dthresh)`. PyEBSDIndex's DRIVER rounds `lam` to float32
   before its kernel (:297), so for a given `lam` its driver equals
   ours only for `lam = float(np.float32(lam))` up to that rounding
   (recorded difference, D12.6 item 12). Pin: V0 AST walk (no
   `ast.Pow` inside an `njit` function of `_nlpar.py`).
4. **`py_func` parity**: bitwise for every kernel except
   `_nlpar_weights_kernel`, whose parity is MTP with seed 1 float32
   ulp (numba's `exp` and NumPy's differ by up to 1 ulp; parked plan
   2026-09-11; unverified today), measured and pinned at the Stage A
   gate as `EXP_KERNEL_ULP` (0 or 1; the placeholder is shared with
   V1 through the root `conftest.py`, D1.9). Pin: V0.
5. **PyEBSDIndex parity level** (amended 2026-10-04, spec review
   E1-F1): expected bitwise on float32 output against the COMPILED
   oracle dispatchers for `sigma` and the normalised 3 x 3 distances
   (vs `sigma_numba`) and for the averaged patterns (vs `nlpar_nb`
   with the oracle's sigma injected, and end to end with ours) at
   `search_radius` in {1, 2, 3}, `lam` in {0.7, 2.5}, `dthresh` in
   {0, 0.5}, masks none and circular, protection on and off,
   synthetic maps and nickel_ebsd_large. The oracle's `.py_func` is
   not a bitwise oracle (measured 2026-10-04 on a (7, 7 | 8, 8)
   float32 map: `sigma_numba` sigma bitwise but `dout` not;
   `nlpar_nb` differs on 2586 of 3136 pixels, max 1.07e-4; cause:
   NumPy and numba promote `float32 ** int32` and `2.0 * float32`
   oppositely), so no parity test calls it. If bitwise fails for a
   recorded reason, the fallback is a per-pixel ulp pin (MTP) plus
   "zero pixels differ by >= 2 grey levels" after rounding, cause in
   the ledger. Pins: V2, V3.
6. **No pair memo**: PyEBSDIndex caches pair distances in a typed
   dict per column (`pairdict`, :874, :897-899, :917), which reuses
   only the pairs shared between consecutive window positions in one
   column (about 3 of 48 per the parked plan's estimate) at the cost
   of a dict allocation per column and `nogil`-unfriendly code.
   Recorded; the V11 baseline measures what it costs.
7. **Nothing from `diff_offset`** (:894, a per-slot accumulator that
   also accumulates across `prange` threads), `backsub` (:481-482) or
   `stem_scale` (:472-477, :511-513) is ported (Scope).
8. **Index safety**: every window bound comes from `_window_bounds`
   (D4.2); kernels never compute PyEBSDIndex's raw expressions.
   Pins: V0 (flags, names, `py_func`), V2, V3, V11 (pytest-benchmark
   entry plus `record_property` runtimes, recorded never gated; seeds
   D13.3).

### D8 -- Dask orchestration (frozen)

1. **Array**: `get_dask_array(signal=self, chunk_bytes=8e6,
   rechunk=True)` (`ebsd.py:1056`): navigation axes chunked, signal
   axes whole. For an eager input the same path runs on a one-or-few
   chunk array and is computed at the end (as
   `average_neighbour_patterns` does). For a LAZY input
   `rechunk=True` runs `_reduce_chunks` (`_dask.py:142-151,
   163-195`; recorded 2026-10-04, spec review F3-R3-1/C3-R3-F2): on a
   2-D navigation it KEEPS the user's chunks on the navigation axis
   with the smaller chunksize and re-chunks the other axis with
   `normalize_chunks("auto")` under the 8 MB limit, so the processed
   chunking is in general not the input chunking. Measured (dask
   2026.3.0, uint8 `LazyEBSD`s): `((3, 3, 4), (7, 7, 2))` on
   (10, 16 | 16, 16) -> `((3, 3, 4), (16,))`; `((5, 5), (8, 8))` ->
   `((5, 5), (16,))`; `((1, 9), (2, 14))` -> `((1, 9), (16,))`;
   `((1,) * 10, (1,) * 16)` -> `((1,) * 10, (16,))`; `((26, 26, 3),
   (75,))` on (55, 75 | 60, 60) -> `((26, 26, 3), (40, 35))`; the
   issue-230 tuple on (55, 75 | 6, 6) unchanged. Consequences: the
   method-level lazy pins of V7 vary the ROW chunking only (D8.6);
   column and both-axes chunkings reach the drivers only through the
   Stage A `TestDepthAndHalo` tests, which call `_nlpar_sigma` /
   `_nlpar_average` on `da.from_array(x, chunks=c + (-1, -1))`
   directly; `inplace=True` on a lazy input restores the user's
   chunking afterwards through `rechunk(old_chunks)` (D1.5).
   `rechunk=True` is kept for the `average_neighbour_patterns`
   precedent and the 8 MB chunk bound; the alternative
   `rechunk=False` for lazy inputs (D8.4's `ensure_minimum_chunksize`
   would still normalise thin chunks) is recorded, not adopted.
2. **Pass 1 (sigma)** (amended 2026-10-04, spec review F1-F2/C1-F2/
   E1-F2/E1-F6/E1-F22, adopting plan.md's refinement): depth 1 per
   chunked navigation axis (0 for a single chunk; signal axes 0): the
   clipped 3 x 3 never shifts, so the D8.4 depth rule's second term
   does not apply. Names (defined here once, spec review E3-R3-1):
   `nav_dim` is the navigation dimension (2 after the 1-D routing of
   D9.4) and `nav_chunks = dask_array.chunks[:nav_dim]` the navigation
   chunks of the array the pass runs on (for pass 1 the depth-1
   rechunk `ensure_minimum_chunksize(1, c)` is a no-op, so these are
   the D8.1 chunks; D8.3 rebinds both the array and `nav_chunks`
   after its own rechunk). Route: `overlapped = da.overlap.overlap(
   dask_array, depth=depth1, boundary="none")`, then `da.map_blocks(
   _nlpar_sigma_chunk, overlapped, ..., chunks=nav_chunks + ((4,),
   (9,)), dtype=np.float32, meta=np.empty((0, 0, 4, 9), np.float32))`
   (explicit `chunks=` and `meta=`, so no probe call with
   `block_info=None` is ever made; measured 2026-10-04 on dask
   2026.3.0, plan.md drafting measurements), the wrapper returning
   ONLY the core of its block, packed as float32 `(core_rows,
   core_cols, 4, 9)`: plane 0 `d2`, plane 1 `n2`, plane 2 `valid` as
   1.0/0.0, plane 3 slot 0 `sigma` (slots 1-8 zero). The result is
   computed EAGERLY and unpacked (`_nlpar_unpack_sigma_pass`), then
   `_nlpar_normalized_distances` (D2.5) runs on the whole-map arrays:
   the outputs are small (`(ny, nx, 9)` float32 is 148 kB for
   nickel_ebsd_large, 36 MB for a 1000 x 1000 map) and the optimiser
   needs them in memory. This is `map_overlap(trim=False)` with the
   trim done by the wrapper; chosen because pass 1 changes the signal
   dimensions and `map_overlap`'s `chunks=` would have to be stated
   in overlapped coordinates. The drafted alternative (depth 2 with
   the normalisation inside the kernel and the output ndim kept
   through singleton signal dimensions) is withdrawn.
3. **Pass 2 (average)** (amended 2026-10-04, same route for
   uniformity; `chunks=` corrected 2026-10-04, spec review E3-R3-1):
   first `depth2, chunks_out = _nlpar_depth(dask_array.chunks,
   radius)` and `dask_array = dask_array.rechunk(chunks_out)` (the
   D8.4 rechunk, a no-op when no chunk is thinner than its depth),
   then `nav_chunks = chunks_out[:nav_dim]`; `patterns_ov =
   overlap(dask_array, depth=depth2, boundary="none")` and `sigma_ov
   = overlap(sigma_dask, depth=depth2 on the navigation axes,
   boundary="none")` with `sigma_dask = da.from_array(sigma[...,
   None, None], chunks=nav_chunks + ((1,), (1,)))` (the `window_sums`
   precedent, `ebsd.py:1063-1065`), so both operands receive the same
   halo; then `da.map_blocks(_nlpar_average_chunk, patterns_ov,
   sigma_ov, lam=..., dthresh=..., ..., chunks=chunks_out,
   dtype=dtype_out, meta=np.empty((0,) * ndim, dtype_out))`: the
   POST-rechunk chunks, never the input array's (on the `(1, 1)`
   chunking of V7 at sr 3 the rechunk changes both the block sizes
   and the block count, `ensure_minimum_chunksize(6, (1,) * 16) ==
   (6, 10)` and `(6, (1,) * 10) == (10,)`, measured 2026-10-04, and
   `map_blocks` with the pre-rechunk `chunks=` would raise or
   misplace values), the wrapper returning only the core of its block
   in `dtype_out` (D6 applied inside the wrapper, in NumPy).
4. **Depth rule for pass 2**, per navigation axis with radius `r` and
   length `n`: single chunk -> 0; `2r + 1 >= n` -> rechunk the axis to
   one chunk; else `depth = max(r, 2r + 1 - min(first_chunk,
   last_chunk))`, because a border pattern's inward-shifted window
   reaches `2r` points into the map and the border chunks may be
   shorter than that; then no chunk may be shorter than its depth:
   the driver rechunks with
   `dask.array.overlap.ensure_minimum_chunksize` (present in dask
   2026.3.0; `ensure_minimum_chunksize(4, (26, 26, 3)) == (26, 25,
   4)`, `(6, (26, 26, 3)) == (26, 23, 6)`, `(4, (3, 3, 4)) == (6,
   4)`, `(6, (7, 7, 2)) == (7, 9)`, `(6, (1,) * 16) == (6, 10)` and
   `(6, (1,) * 10) == (10,)`, measured 2026-10-04; present in the CI
   oldest pin dask 2021.8.1, `tests.yml:48`, with a function body
   identical to 2026.3.0's, verified 2026-10-04 against the
   `2021.08.1` tag of `dask/array/overlap.py` (spec review C3-R3-F4;
   `overlap(x, depth, boundary)` is defined there too), so no private
   equivalent is needed; the oldest-matrix recipe of D10.5 pins
   `dask==2021.8.1` as the executable confirmation). This
   closes the pyxem/kikuchipy#230 class ("IndexError in
   average_neighbour_patterns() for LazyEBSD", closed) of irregular
   tiny edge chunks; #824's lazy test with chunks `(3, 3, 4, 3, 4,
   ...)` over 55 rows (diff lines 872-875) is that class, a V7 case.
5. **`block_info`** (amended 2026-10-04, spec review F1-F8/C1-F3/
   E1-F22): both chunk wrappers declare `block_info=None` in their
   signature (dask's calling convention) but assert it is not `None`
   (the explicit `meta=` of D8.2-3 removes the probe call), and derive
   from `block_info[0]["chunk-location"]` against `["num-chunks"]`
   (a) which sides carry a halo: the first/last chunk along an axis
   has none on its outer side (the inward shift of D4.2 applies only
   at true map borders, never at chunk borders) and (b) the kept
   region in block coordinates (PyEBSDIndex's `calclim`, :835-843,
   for free). `array-location` is in overlapped coordinates under
   both `map_overlap` and `map_blocks` on an overlapped array
   (measured 2026-10-04) and is not used. The wrapper computes and
   returns the kept region only; nothing is zero-filled and nothing is
   trimmed afterwards.
6. **Lazy == eager, bitwise** (V7): the result is a deterministic
   function of the data, not of the chunking, the scheduler or the
   thread count (no cross-chunk accumulation anywhere; the global
   maximum is a reduction). Cases through the METHOD (after the
   `_reduce_chunks` re-chunking of D8.1, which keeps the row chunks
   of every small lazy fixture and makes its column axis one chunk;
   amended 2026-10-04, spec review F3-R3-1/C3-R3-F2): single chunk,
   regular rows `(5, 5)`, irregular tiny-edge rows `(3, 3, 4)`, rows
   `(1, 9)`, `(1,) * 10` (thinner than the depth), `(2,) * 5`,
   `(4, 4, 2)`, the explicit `(26, 26, 3)` last-chunk rows on
   nickel_ebsd_large (processed as `((26, 26, 3), (40, 35))`), the
   #230 tuple (unchanged by `_reduce_chunks`). Column and both-axes
   chunkings (`((3, 3, 4), (7, 7, 2))`, `((1, 9), (2, 14))`,
   `((1,) * 10, (1,) * 16)`, `((2,) * 5, (3, 3, 3, 3, 3, 1))`,
   `((4, 4, 2), (4,) * 4)`) are pinned at DRIVER level by the Stage A
   `TestDepthAndHalo` tests, which bypass `get_dask_array`.
7. **Cost, recorded**: a lazy input is read three times (global
   maximum, pass 1, pass 2; `get_nlpar_sigma` reads twice,
   `get_nlpar_lambda` twice); in-memory inputs pay nothing
   noticeable. The docstring says so. Estimated runtimes: parked plan
   seeds in D13.3.
8. **No scheduler manipulation inside the method**: #824 sets
   `num_workers=2` and `scheduler="threads"` by default (diff lines
   632-643); here the user's `dask.config` applies, as everywhere
   else in kikuchipy. Recorded, not reproduced.
9. **Custom attributes** (`static_background`, `detector`, `xmap`)
   are carried through `**self._get_custom_attributes()` (`ebsd.py:
   1111`); V7 asserts them on the `inplace=False` result.
   Pins: V7 (all of the above), V11 (performance baselines).

### D9 -- Masks and 1-D navigation (frozen)

1. **`signal_mask`**: a boolean array of the signal shape (row, col),
   `True` = pixel EXCLUDED (kikuchipy's convention for
   `dictionary_indexing`, `hough_indexing` and the refinement
   methods). It selects the pixels that enter `d2` and `n2` in both
   kernels (`indices = np.flatnonzero(~signal_mask.ravel())`, in the
   oracle's ascending order; PyEBSDIndex's `mask` is 1 = use,
   :232, :370). The weighted SUM averages every pixel, masked or not
   (PyEBSDIndex loops `range(shpdata[1])`, :933-934): masked pixels
   (e.g. outside a circular detector) still benefit from denoising
   and the output stays a complete pattern. #824 averages only the
   unmasked pixels (diff lines 349-352): recorded difference. A mask
   excluding every pixel raises `ValueError` (no evidence to weigh;
   #824 warns and no-ops). Pins: V2/V3 mask arms; V4 mask-polarity
   separation (a flipped mask changes sigma measurably on a map whose
   masked region carries a different noise level); the plan.md mutants
   "mask polarity" and "mask not forwarded to sigma".
2. **No automask**: PyEBSDIndex applies a circular detector mask by
   default (`automask=True`, :55; `makeautomask`, :736-750).
   kikuchipy users pass an explicit mask; the tutorial shows
   `~kp.filters.Window("circular", shape=sig_shape).astype(bool)`
   as the equivalent. Recorded.
3. **Consistency**: the same mask, the same `saturation_protect` and
   the same global maximum feed sigma, the optimiser and the average
   within one call.
4. **1-D navigation**: a `(n|h, w)` signal is processed as `(1, n)`
   internally; `search_radius` is an `int` or a 1-tuple; the result
   is reshaped back; `get_nlpar_sigma` returns shape `(n,)`. Pin: V7
   "1-D scan == the `(1, n)` run, bitwise".
5. **0-D navigation** raises `ValueError`. **Navigation mask**: none
   in v1 (Scope); revisit path: a `navigation_mask` that removes scan
   points from the candidate set and leaves them unaveraged.

### D10 -- Dependencies, licensing and the oracle (frozen)

1. **No new required dependency**: numpy, scipy (>= 1.7 for bounded
   Nelder-Mead, `pyproject.toml:64`), numba, dask, scikit-image
   (`dtype_range`) are all required already. `pyebsdindex` stays an
   OPTIONAL extra (`all` and `doc`, `pyproject.toml:79, 94`; floor
   `>= 0.3.9.2, != 0.3.10` because of
   USNavalResearchLaboratory/PyEBSDIndex#80, closed) and is never
   imported by `_nlpar.py` or by the NLPAR methods in `ebsd.py` (the
   existing Hough-indexing import path is untouched). Without it the
   feature works in full; only the oracle tests skip.
2. **Oracle tests**: `@pytest.mark.skipif(dependency_version[
   "pyebsdindex"] is None, reason="pyebsdindex is not installed")`
   (`_constants.py:31-37`; precedent
   `test_ebsd_hough_indexing.py:35-37`), with `from
   pyebsdindex.nlpar_cpu import NLPAR` inside the gated module or
   fixture. The oracle kernels are called COMPILED on EVERY arm,
   small synthetic maps and nickel_ebsd_large alike, behind the
   module-scoped warm-up fixture `pyebsdindex_kernels` whose JIT time
   is recorded (V3; cold compile measured 2026-10-04: 4.0 s
   `sigma_numba` + 3.2 s `nlpar_nb`, inside the ~90 s budget of
   D13.5). Their `.py_func` is NEVER used as an oracle (amended
   2026-10-04, spec review E1-F1: it is not bitwise with the compiled
   kernels, D7.5). The call recipe (2-D `(nrows * ncols, npix)`
   float32 layout, `col + ncols * row` pattern index, int64 kept
   indices, a FRESH `calclim` array per call, every navigation axis
   `>= 2 sr + 1`) is written once in validation.md's Automated
   section. Both kernels are identical in 0.3.9.2 and 0.3.10.1
   (parked plan, 2026-09-11; (unverified today)), so no version gate
   beyond the floor; the oldest-matrix recipe runs them at 0.3.9.2.
3. **Numba-cache flake rule** (recorded): importing `pyebsdindex`
   redirects `NUMBA_CACHE_DIR` to a shared `~/.pyebsdindex/numbacache`
   (`pyebsdindex/band_detect.py:60-62`, `gnomonic_correction.py:
   52-54`; verified 2026-10-04), defeating per-worker cache isolation
   under xdist. Every gate runs `pytest ... -n 0` first, then `-n 4`,
   and re-runs red tests alone before calling them failures (memory
   note, PyEBSDIndex numba-cache flake).
4. **Licensing and attribution.** `_nlpar.py` is GPL-3.0-or-later like
   `pattern/chunk.py`: the skeleton agent of the `nlpar-stage-tests`
   workflow copies the GPL header verbatim from `chunk.py:1-16` (the
   `licenseheaders` hook of `.pre-commit-config.yaml` is skipped in
   every gate command, `SKIP=licenseheaders`, and is then a no-op;
   spec review C2-F16, 2026-10-04). Below the GPL header it carries the delimited
   third-party block of the `_master_pattern.py:20-57` convention
   (tech-stack.md:45) with: (a) a rationale line "The following notice
   is reproduced from PyEBSDIndex (`pyebsdindex/nlpar_cpu.py`, US
   Naval Research Laboratory), from which the NLPAR kernels of this
   module are derived; the code is in the public domain under 17
   U.S.C. 105, and the notice asks derivative works to state the
   changes made, their date and nature, and to acknowledge NRL as the
   original source."; (b) between `# ####` rules the NRL notice of
   `nlpar_cpu.py:1-21` verbatim (public-domain statement, no-warranty
   disclaimer, derivative-works grant, the request for a change notice
   and acknowledgement, author and date lines); (c) "Changes by the
   kikuchipy developers, <date of the implementation commit>:" with
   the list: rewritten as numba kernels over in-memory NumPy chunks
   with `cache=True, nogil=True` and no `parallel`; per-axis search
   radius; search window clamped when an axis is shorter than the
   window; duplicate guard `d2 > 0`; pairs without comparable pixels
   weighted 0; saturation threshold from the global maximum; lambda
   objective excluding out-of-map neighbours and using the averaging
   weight form; `diff_offset`, `backsub`, `stem_scale`, the pair memo
   and all file I/O removed; integer output rounded to nearest; (d)
   "The US Naval Research Laboratory (David Rowenhorst) is gratefully
   acknowledged as the original source of the NLPAR implementation."
   The method docstring Notes and the CHANGELOG bullet repeat the
   acknowledgement in one sentence. EMsoftOO `mod_NLPAR.f90` (BSD-3,
   header lines 1-12 of the local clone) is named in the module
   docstring as an equation cross-check only; no code is ported, so
   no BSD notice is needed. The paper is cited through
   `:cite:`brewick2019nlpar``; the bibliography needs no edit.
5. **Oldest matrix**: py3.10 / numpy 1.23.0 / numba 0.57 / orix 0.12.1
   / pyebsdindex 0.3.9.2 / dask 2021.8.1 / scikit-image 0.21.0
   (`tests.yml:47-49`). No numba feature newer than 0.57 in the
   kernels. One recorded local run per stage with the ONE recipe
   string used in all three spec documents and in the tech-stack
   amendment (unified 2026-10-04, spec review F1-F6/C1-F5/E1-F21):
   `uv run --isolated --python 3.10 --with "numpy==1.23.0" --with
   "numba==0.57" --with "orix==0.12.1" --with "pyebsdindex==0.3.9.2"
   --with "dask==2021.8.1" --with "scikit-image==0.21.0" pytest
   tests/test_signals -k nlpar` (the parked plan's recipe plus the
   dask pin that makes D8.4's `ensure_minimum_chunksize` import
   checkable locally and the scikit-image pin of the CI oldest line,
   `tests.yml:48`; the test root of D1.9).
   Pins: V2/V3 (skip cleanly without pyebsdindex; the NumPy
   transcription oracles of V1/V4/V5/V10 keep the minimum-requirement
   job meaningful), V0.

### D11 -- Branch, PR and fan-out policy (frozen, user decisions 2026-10-04)

1. **Base**: fork `develop` @ `18d59c07`; branch `feat-NLPAR`; the
   specs tree is already tracked there, so no re-adoption step. This
   supersedes decision 1 of the 2026-09-11 plan (base and PR target
   `feat-spherical-indexing`).
2. **One fork PR** `feat-NLPAR -> develop`, opened after Stage C with
   the PR template, expected number #17 (today: PRs up to #16 exist
   and the fork has issues disabled, so #17 is next unless another PR
   is opened first; confirmed with `gh pr list` at PR time). CHANGELOG
   bullets use the fork PR-link convention
   `` (`#17 <https://github.com/jwestraadt/kikuchipy/pull/17>`_) ``
   (CHANGELOG.rst:29 style). Signed commits (`git commit -s`) with
   the session's attribution trailer, explicit pathspecs, pushed to
   `origin/feat-NLPAR` after each stage; a failing-tests commit is
   never pushed alone (the fork's tests workflow runs on every push,
   `tests.yml:22-29`, so an unpaired red run is noise).
3. **Merge** only on the user's go, as a merge commit, with ubuntu and
   windows CI green. macOS is known red: its job installs `ebsdsim`
   from a git branch for the GPU step (`tests.yml:137-143`) and a
   pre-existing pseudo-symmetry `23 == 22` test fails there (user's
   record; (unverified today)).
4. **Fan-out after the merge** (plan.md section 1, fan-out steps 1-2;
   the session plan's Steps 2-3): merge `develop`
   into `hrebsd-dic` (append-only conflicts; still never merged
   back); replay the NLPAR merge as two clean commits ("Add non-local
   pattern averaging (NLPAR)", "Add NLPAR tutorial") onto a new
   branch `feat-spherical-indexing-nlpar` off `feat-spherical-indexing`
   @ `6723aaf0`, `specs/` stripped, `Staged-from:` trailers, the
   staging `gate.ps1` and equivalence gate, pushed with no PR.
   `feat-spherical-indexing` is never merged into `develop`, nor
   `develop` into it.
5. **Clean-replay rule** (enforced at every stage gate): nothing under
   `src/`, `tests/`, `doc/` or `examples/`, nor the root
   `conftest.py`, `benchmarks/` or `CHANGELOG.rst` (all carried
   verbatim onto the clean branch by the replay, which strips
   `specs/` only; extended 2026-10-04, spec review E3-R3-2), names a
   `specs/` path, a spec file name (`requirements.md`, `plan.md`,
   `validation.md`, `tech-stack.md`) or a D/V number; comments state
   the fact or the measurement itself. Gate (the user's wording "`git
   grep ... develop..` finds nothing new"; executable form, since
   `git grep` takes a tree, not a range, and `develop..` fails with
   "unable to resolve revision", verified 2026-10-04): `git diff
   develop...HEAD -- src tests doc examples benchmarks conftest.py
   CHANGELOG.rst | grep -E "^\+" | grep -n -E
   "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md|
   [DV][0-9]"` prints nothing (the ` [DV][0-9]` alternative catches a
   bare D/V number such as "see D8.4" or "V7"; a hit is a spec
   reference to rewrite as the fact itself).
6. **Convergence rule (take-ours)**: when upstream lands an NLPAR under
   the shared name (#824 or a successor) and the fork merges upstream,
   the fork keeps its own implementation and tests, reconciling only
   keyword names if upstream's differ (a thin alias or rename, with a
   CHANGELOG "Changed" bullet). No review is posted on #824 unless
   the user says so.
7. **Standing constraints**: `stash@{0}` never popped;
   `specs/2026-08-16-constitution/upstream-issue.md` and the parked
   upstream-merge plan never edited (the parked NLPAR plan gets only
   the status header the session plan prescribes, in the spec
   commit); the never-sweep notebooks (tech-stack.md:11) never in a
   diff; pre-commit via `--files <explicit list, never specs/>`; no
   em-dashes in prose; the main `.venv` stays CuPy-free.
   Pins: the definition of done in validation.md (the clean-replay
   grep, `git log origin/feat-NLPAR..feat-NLPAR` empty, CI state
   recorded).

### D12 -- Documentation (frozen)

1. **Tutorial** `doc/tutorials/nlpar.ipynb`: equations and
   acknowledgement; a synthetic two-grain demonstration with the sigma
   map and the preserved boundary next to the Gaussian
   `average_neighbour_patterns` blur (V5's figure); on
   `nickel_ebsd_large(allow_download=True)`: sigma map, lambda-vs-
   target curve, before/after patterns, image quality and average dot
   product maps, background removal plus Hough indexing before and
   after (the `s.inav[::5, ::5]` subset of 165 patterns, navigation
   (11, 15) rows x cols; NOT `extract_grid((11, 15))`, which returns
   an internally inconsistent signal, ledger entry 3); `si_wafer`
   numbers quoted from the
   ledger (no download); parameter guidance, the "Differences" list,
   a link to `hybrid_indexing.ipynb`. Notebook rules
   (tech-stack.md:52): hidden first cell, thumbnail tag, black at 77;
   stored outputs if the Read the Docs estimate exceeds ~2 min.
2. **Registration**: `doc/tutorials/index.rst` "Fundamentals and
   usage" gallery after `pattern_processing` (`index.rst:19-26`);
   `doc/tutorials/run_nbval.sh` `NOTEBOOKS` in its alphabetical slot
   (between `mandm2021_sunday_short_course.ipynb` and
   `pattern_matching.ipynb`); `doc/tutorials/tutorials_sanitize.cfg`
   regexes added only as nbval needs them (9 exist, `regex1`-`regex9`:
   timings regex1/2, tqdm regex4, chunk counts regex9; counted
   2026-10-04).
3. **Gallery example** `examples/pattern_processing/nlpar.py` next to
   `neighbour_pattern_averaging.py`: `nickel_ebsd_small` or the
   large dataset if cached, default parameters, a before/after
   figure; sphinx-gallery renders it (`doc/examples/` is generated).
4. **API reference**: generated from the `EBSD` class (only
   `doc/reference/index.rst` is tracked; the per-method pages under
   `doc/reference/generated/` are build outputs), so the three methods
   appear without a hand edit.
5. **CHANGELOG** `Unreleased -> Added`: one bullet for the API (the
   method and the two getters, the acknowledgement sentence) and one
   for the tutorial, each ending with the fork PR link (D11.2).
   `hybrid_indexing.ipynb` is not edited (Scope).
6. **"Differences from PyEBSDIndex and from upstream PR #824"**
   (docstring Notes; frozen list, numbers from the ledger at Stage B):
   (1) output not rescaled per pattern (PyEBSDIndex `rescale=False`;
   #824 rescales); (2) border windows shift inward (as PyEBSDIndex;
   #824 replicates edge patterns) and axes shorter than the window
   are handled; (3) `dthresh` enters the averaging weights and the
   lambda objective in the same form (PyEBSDIndex uses two forms;
   #824 uses it in the fit only); (4) the lambda objective excludes
   out-of-map neighbours (PyEBSDIndex counts them as weight 1; about
   +2 % on lambda for a 55 x 75 map at target 0.34, ledger value);
   (5) duplicate guard `d2 > 0` (PyEBSDIndex `>= 1e-3`); (6) pairs
   without comparable pixels get weight 0 (PyEBSDIndex 1); (7) the
   saturation threshold uses the global maximum (PyEBSDIndex per
   tile); (8) the search window is the full square and the mask
   affects distances only, every pixel is averaged (as PyEBSDIndex;
   #824 uses a boolean window, "circular" by default, and averages
   unmasked pixels only); (9) own kernels, PyEBSDIndex not needed at
   runtime (#824 needs it for sigma and lambda); (10) integer output
   rounded to nearest and clipped, no dependence on PyEBSDIndex's
   writer; (11) `lam` defaults to `None` (optimised; PyEBSDIndex 0.7,
   #824 0.9); (12) `lam` reaches the kernel as float64 (PyEBSDIndex's
   driver rounds it to float32 first, `nlpar_cpu.py:297`), so for a
   given `lam` the two agree up to that rounding (added 2026-10-04,
   spec review E1-F8). Twelve items; the module-level change notice
   (D10.4) additionally names `diff_offset`, `backsub` and
   `stem_scale`.
   Pins: Stage C validation matrix (clean-kernel execute, nbval,
   `sphinx-build -b html` exit 0, linkcheck of the new links, name and
   spell pass) in validation.md Manual; V8 supplies the tutorial's
   quoted numbers.

### D13 -- Measurement policy (recorded)

1. **MTP**: every tolerance is measured at the owning gate and pinned
   with `pytest.approx(measured, rel=0.05)` or the ~2x margin
   convention (tech-stack.md:51); bitwise pins are stated as such.
   Effects on real data are pinned only when measured as improvements
   (>= 0.5x the measured gain), otherwise "not worse", and the ledger
   says which.
2. **Seeds are seeds**: the 2026-09-11 numbers below were measured with
   a pure-NumPy reference on the drafting machine; they calibrate
   expectations and never become pins without re-measurement.
3. **Seeds carried** (parked plan, 2026-09-11): PyEBSDIndex kernels on
   nickel_ebsd_large at radius 3, single thread: sigma 0.12 s,
   averaging 0.54 s; pure-NumPy reference 6-14 s; lambda at target
   0.34 on nickel_ebsd_large: 1.1164 raw and 2.5246 background-
   corrected with phantoms, 1.1387 and 2.5787 without (an earlier
   reading: 1.098 vs 1.121, +2.05 %); average dot product on the
   corrected map 0.600 -> 0.904 at the optimised lambda (~2.52),
   0.766 at `lam = 0.7`; two-grain cross-boundary `d ~ 159` (step 30,
   sigma 8, N 1024); iid sigma^2 bias band about -1.0..1.4
   `sqrt(2/N)`; runtimes nickel_ebsd_large 0.2-0.4 s with dask
   threads, si_wafer 5-10 s.
4. **Re-measured 2026-10-04** (this machine, read-only): installed
   python 3.13.12, pyebsdindex 0.3.10.1, numba 0.65.1, numpy 2.4.6,
   orix 0.14.2, dask 2026.3.0, kikuchipy 0.14.dev0, pooch 1.9.0;
   `nickel_ebsd_large` cached: `(55, 75 | 60, 60)` rows x cols
   (hyperspy repr `(75, 55|60, 60)`) uint8, min 21, max 253; phantom
   slots 776 / 37125 by arithmetic; `sigma_numba` and `nlpar_nb` flags
   `{nopython: True, fastmath: False, parallel: True}`.
5. **Ledger**: validation.md "Recorded results" is append-only,
   numbered, dated, with the recipe and the machine; every amendment
   to this file is dated in place (the HREBSD precedent). Budget:
   default-suite additions <= ~90 s per worker including the
   PyEBSDIndex warm-up fixture; weekly <= 5 min.
6. **Spec-review measurements (2026-10-04, critics and fixer,
   read-only Python; ledger entry 2 in validation.md)**: PyEBSDIndex
   `sigma_numba` compiled vs `.py_func` on a (7, 7 | 8, 8) float32
   map: sigma bitwise, `dout` NOT bitwise, `nout` bitwise; `nlpar_nb`
   compiled vs `.py_func`: NOT bitwise (2586 of 3136 pixels, max abs
   1.07e-4, same for a float and a float32 `lam`); compiled
   `sigma_numba` counter `n2 += 1.0` is float64 (`64.000000000001`);
   interior self slot `dout = -5.656854 = -sqrt(64 / 2)`, `nout = 64`;
   corner `nout = [64, 64, 64, 64, 0, 0, 0, 0, 0]` (compact
   enumeration); cold compile 4.0 s + 3.2 s. dask 2026.3.0:
   `overlap(..., boundary="none")` + `map_blocks(..., chunks=, meta=)`
   with a core-only wrapper returning `(core_rows, core_cols, 4, 9)`
   computes the right shape, makes no `block_info=None` call, and
   `block_info[0]` carries `array-location`, `chunk-location`,
   `num-chunks`, `shape`; `ensure_minimum_chunksize(4, (26, 26, 3))
   == (26, 25, 4)`, `(6, (26, 26, 3)) == (26, 23, 6)`, `(5, (47, 8))
   == (47, 8)`; depth rule values `(47, 8)`: r3 -> 3, r4 -> 4, r5 ->
   5; `(26, 26, 3)`: r3 -> 4, r4 -> 6. `git grep ... develop..` fails
   ("unable to resolve revision"). `tutorials_sanitize.cfg` has 9
   `[regexN]` sections. `tests.yml`: `os:` matrix at line 43, oldest
   dependencies at line 48.
7. **Spec-review round 2 measurements (2026-10-04, fixer, read-only
   Python with scipy 1.17.1; ledger entry 3)**: bounded Nelder-Mead on
   the D5.1 objective from `x0 = 1.0` with every non-self slot at
   `c = 200` returns `1.000000` (flat objective, `nit` 10), so that
   field is NOT an upper-bound arm; one slot at `c = 1` and seven at
   `+inf` returns `10.0` (upper bound hit); all slots at `c = 1e-4`
   returns `0.0084` (no bound), at `1e-6` and `1e-7` returns `1e-3`
   (lower bound hit; closed form `2.66e-4` at `1e-7`; the bound is hit
   for `c < 1.416e-6` at `tw = 0.34`). A 70/30 two-valued field
   (`c = 2` / `c = 12`) at fixed `lam = 1.5` gives a mean-over-points
   objective of `0.2616` and a median of `0.1068`; its mean-objective
   minimiser from `x0 = 1.0` is the kink `1.1884`, not a value between
   the two closed forms. On the border-masked constant field the
   phantom-free objective with the invalid slots at `d = 0` equals the
   one with them at `+inf` EXACTLY and differs from the
   phantom-counting objective by `0.049` (`lam` 1.0) and `0.042`
   (`lam` 2.0). Sequential float32 accumulation of `k` copies of
   `fl(fl(1/k) * p)`, `p` in 1..254: worst `2 / 3 / 11` float32 ulp at
   `k = 9 / 20 / 49` (critic E2's probe: `3 / 4 / 13`), so the
   constant-map identity is NOT within 2 ulp beyond 9 weights.
   `s.extract_grid((11, 15))` on `nickel_ebsd_large` returns data
   `(13, 10, 60, 60)` with an axes manager claiming `(11, 15)` and an
   `xmap` of shape `(49, 64)` (internally inconsistent; `ebsd.py:
   329-332` reverses `grid_shape` before `grid_indices`), while
   `s.inav[::5, ::5]` returns data `(11, 15, 60, 60)`, 165 patterns,
   `xmap.size == 165` with the orix extent shape `(51, 71)` under step
   slicing (points compared by order, never by shape), detector and
   static background carried. `tests/`, `tests/test_signals/` have no
   `__init__.py` or `conftest.py`; `pyproject.toml:167` sets
   `--import-mode=importlib`.
8. **Spec-review round 3 measurements (2026-10-04, fixer, read-only
   Python `scratchpad/fixer3_probe.py` plus critic C3's two probes
   re-run; ledger entry 4)**: `get_dask_array(signal=<LazyEBSD>,
   chunk_bytes=8e6, rechunk=True)` runs `_reduce_chunks`
   (`_dask.py:142-151, 163-195`), which keeps the chunks of the
   navigation axis with the smaller chunksize and re-chunks the other
   to the 8 MB limit: on (10, 16 | 16, 16) uint8, `((3, 3, 4), (7, 7,
   2))` -> `((3, 3, 4), (16,))`, `((5, 5), (8, 8))` -> `((5, 5),
   (16,))`, `((1, 9), (2, 14))` -> `((1, 9), (16,))`, `((1,) * 10,
   (1,) * 16)` -> `((1,) * 10, (16,))`, `((2,) * 5, (3, 3, 3, 3, 3,
   1))` -> `((2,) * 5, (16,))`, `((4, 4, 2), (4,) * 4)` -> `((4, 4,
   2), (16,))`; on (55, 75 | 60, 60), `((26, 26, 3), (75,))` ->
   `((26, 26, 3), (40, 35))`; the issue-230 tuple on (55, 75 | 6, 6)
   unchanged (D8.1, D8.6). `ensure_minimum_chunksize`: `(4, (3, 3,
   4)) == (6, 4)`, `(6, (7, 7, 2)) == (7, 9)`, `(6, (1,) * 16) == (6,
   10)`, `(6, (1,) * 10) == (10,)`, `(1, (1,) * 10)` unchanged, `(4,
   (26, 26, 3)) == (26, 25, 4)` (D8.3, D8.4); its source at the dask
   `2021.08.1` tag is identical to 2026.3.0's and `overlap(x, depth,
   boundary)` exists there (D8.4). `average_neighbour_patterns(
   inplace=True)` on a `((47, 8), (47, 28))`-chunked in-memory
   (55, 75 | 60, 60) uint8 signal equals `inplace=False` bitwise
   under the synchronous scheduler and threads with 1, 2 and 20
   workers, 0 differing patterns; a toy `map_overlap` + `store` into
   its own source likewise (D1.5 (b)). Compiled `sigma_numba` /
   `nlpar_nb` on a constant (4, 5 | 6, 6) map at 100: protection on,
   `nout` of every neighbour slot stays at the `1e-12` seed (no kept
   pair) and sigma is `1e12` everywhere; protection off, `nout = 36`
   and sigma `1e12` (D2.4, D3.5). `8 exp(-200) = 1.1e-86` is
   representable in float64; `8 exp(-200 / lam^2) < 2^-53` for `lam <
   2.27`, so `1 + 8 w` rounds to 1.0 and the D5.4 objective is flat
   (V6). Line facts: `nlpar_cpu.py:181` is `self.lam = np.median(
   lamopt_values)`; `_data.py` `si_wafer` spans 392-449; the
   `ebsd.py` hunks between `feat-spherical-indexing` and `develop`
   are at 1673 and 1776 (4 insertions, 12 deletions).

## Context

### Algorithm (Brewick, Wright and Rowenhorst 2019)

Patterns `p_i` are vectors of `N` pixel values (after masking, the
`N` unmasked pixels). NLPAR replaces each pattern by a weighted
average of the patterns in a `(2r + 1)^2` search window around it,
with weights that fall off with the pattern DISSIMILARITY measured in
units of the noise, not with the scan distance. Three equations, in
plain text:

(1) Noise estimate: `sigma_i^2 = min over j in N8(i) of
    ||p_i - p_j||^2 / (2 N)`, with `N8(i)` the 3 x 3 neighbours
    (PyEBSDIndex replaces `N` by the per-pair count of unmasked,
    unsaturated pixel pairs `n_ij`, D2.1).
(2) Normalised distance: `d_ij = [ ||p_i - p_j||^2 - N (sigma_i^2 +
    sigma_j^2) ] / [ sqrt(2 N) (sigma_i^2 + sigma_j^2) ]`. For two
    noisy copies of the same pattern the squared distance has
    expectation `N (sigma_i^2 + sigma_j^2)` and standard deviation
    `sqrt(2 N) (sigma_i^2 + sigma_j^2)` (Gaussian noise, large `N`),
    so `d_ij ~ 0 +- 1` within a grain and grows linearly with the
    squared contrast across a boundary (two-grain seed: `d ~ 159`).
(3) Weights: `w_ij = exp( -max(d_ij - dthresh, 0) / lambda^2 )`,
    `w_ii = 1`, `p_i' = sum_j w_ij p_j / sum_j w_ij`.

Lambda sets how many "noise standard deviations" of dissimilarity are
tolerated; PyEBSDIndex chooses it so that the mean self weight in the
3 x 3 neighbourhood hits a target (D5). The output has the same
intensity scale as the input (convex combination; D6).

### PyEBSDIndex quirk catalogue (nlpar_cpu.py, 0.3.10.1, re-verified 2026-10-04)

Ported for parity (by construction; the line cites sit in D2, D3 and
D5): the clipped 3 x 3 sigma window, the `1e24 -> 1e12` fallback and
the normalised `dout` (:752-818); the shifted
`(2 sr + 1)^2` window, the `-1e6` self slot, float32 `d2`/`n2`, the
`dnorm <= 1e-8` branch, `max(d - dthresh, 0)`, `exp(-w / lam^2)`,
normalise-then-accumulate and the two saturation constants
(:820-936); the `loptfunc` metric, Nelder-Mead call and stride
(:94-184).

NOT ported (recorded, each with its D-number):
- Window indices out of bounds when an axis is shorter than `2 sr + 1`
  (no bounds check, :872-878) -- D4.2 clamps.
- Phantom neighbours in the lambda objective (`dout` trailing zeros,
  :756, :800; 776 / 37125 on nickel_ebsd_large) -- D5.2.
- `max(d2, dthresh)` in `loptfunc` (:107) vs `max(d2 - dthresh, 0)` in
  the kernel (:924) -- D5.3.
- `d2 >= 1e-3` duplicate guard (:796) -- D3.3.
- `n2` seeded at `1e-12` in `sigma_numba` (:785; ours starts at 0, the
  seed is rounded away in `s0` for every `n2 >= 1`) -- D2.1/D2.5.
- `n2 == 0 -> d = 0 -> weight 1` (:912-915) -- D3.4.
- Tile-local saturation maximum (:763, :865) -- D3.5.
- `diff_offset` accumulating across `prange` threads (:863, :894),
  `backsub` (:481-482), `stem_scale` (:472-477, :511-513), `automask`
  (:55, :736-750), `rescale` (:391-397, :516-521), the pair memo
  (:874, :897-899) -- Scope, D7.6-7, D9.2.
- `parallel=True` kernels (:753, :821) -- D7.1.
- File-based driver (reads `convertToFloat=True`, :254-255, :470-471;
  writes `flt2int='clip'`, :528) -- parity on float32 arrays only,
  D6.4. PyEBSDIndex reads kikuchipy h5ebsd files (`ebsd_pattern.py`,
  parked plan :105-111; (unverified today)), so a file-based
  end-to-end oracle remains possible for V8's gated arm.
- `print` progress output (:278-280, :407, :178-179) -- D1.8 logging.

### EMsoftOO `mod_NLPAR.f90` cross-check (BSD-3, local clone, re-read 2026-10-04)

Same equations: `getWeightFactors_` computes `exp(-lambda * max(0,
(sum(diff^2) - fps * sigj) / (sfps * sigj)))` with `sigj = sigi +
se(j)` the sum of the two sigma^2 and `sfps = sqrt(2 fps)` (:303-304,
:335-337), self weight `1.0` (:330-331), normalised over the window
(:341); `lambda` is passed as `1 / lambda^2` (:468, :818). Quirks,
cross-check only: `fps`, `sfps`, `sigi`, `sigj` are declared
`integer(kind=ill)` (:301), so sigma^2 sums and `sqrt(2 N)` are
truncated to integers; a `10000.0` sentinel initialises the
neighbour-distance array in the sigma estimate (:224; the parked plan
read it as killing border smoothing); the top block of rows is
handled by a `row = 2*SW+1 - (ipf_ht - jrow)` re-indexing (:635-636;
the parked plan recorded a stale-row bug on the last row); no lambda
optimisation (`lambda = 0.375` default in the DI namelist,
`program_mods/mod_DIfiles.f90:615`; `sw` default (unverified), the
parked plan recorded 3). No code is ported.

### Upstream PR #824: state and maintainer direction (read 2026-10-04)

- Author jorgenasorhaug; commits `b9068b44` "Add non-local pattern
  averaging and corresponding tests" and `d28f0448` "Addressed review
  comments and fixing errors."; files `doc/dev/code_style.rst`,
  `src/kikuchipy/pattern/_nlpar.py` (new, 335 lines),
  `src/kikuchipy/pattern/chunk.py`, `src/kikuchipy/signals/ebsd.py`,
  `tests/test_io/test_kikuchipy_h5ebsd.py`,
  `tests/test_signals/test_ebsd.py`. No tutorial, no CHANGELOG.
- hakonanes (CHANGES_REQUESTED, 2026-09-27): own module
  `pattern/_nlpar.py`; `:cite:` the existing `brewick2019nlpar` key
  and add a citing note to `doc/dev/code_style.rst`; `X | Y` and
  `bool | None` hints; no `print` in tests; doubts that a `Window`
  makes sense as the search window ("An alternative is to just
  support giving the window shape, similar to PyEBSDIndex' search
  radius"); suggested asking drowenhorst-nrl to expose the functions.
  rmz-oz (2026-09-30): collection fails on py3.10 and 3.13 with
  `NameError: name 'Union' is not defined`. The author applied the
  move, cite, hints and print removal on 2026-10-03 and opened
  PyEBSDIndex#83 (open, 0 comments).
- API in `d28f0448`: `average_non_local_neighbour_patterns(window=
  "circular", window_shape=(7, 7), sigma=None, lamda=0.9,
  signal_mask=None, show_progressbar=None, inplace=True,
  lazy_output=None, dask_config_kwargs=None, **kwargs)`; sigma and
  lambda through PyEBSDIndex at runtime; averaging by
  `sliding_window_view` inside `map_overlap`; tests are hard-coded
  answers on the 3 x 3 `dummy_signal`.

Defects in `d28f0448`, RECORDED, NOT REPRODUCED (diff line numbers
refer to the saved `pr824.diff`):
1. Border patterns replicated (`np.pad(mode="edge")` on patterns and
   sigma, 267-277) where PyEBSDIndex shifts the window inward (D4.2).
2. Output min-max rescaled per pattern (354-360); PyEBSDIndex does not
   rescale (D6.3).
3. `dthresh` enters the lambda fit (172) but not the averaging weights
   (338-341), and the fit uses `max(d2, dthresh)` (D5.3).
4. The default "circular" window is cast to bool and the corners are
   dropped (296-301, 506); a `gaussian` window becomes a boolean mask
   of its non-zero coefficients (the test at 746-764 passes
   `std=0.3`), so the float weights are silently ignored (D4.1).
5. `da` is not imported in `_nlpar.py` (235-236) and `InputError` is
   undefined (100): without `from __future__ import annotations` the
   signature raises `NameError` at import on py < 3.14, the class
   rmz-oz hit with `Union` (D1.8).
6. The `np.std` fallback for sigma (118-120) is unreachable
   (`verify_dependency_or_raise` raises rather than returning, 83) and
   invalid anyway (`np.std(..., astype=...)` is not a keyword), though
   the docstring claims PyEBSDIndex is optional (D2.7, D10.1).
7. `print` in the optimiser (225-230).
8. Sigma depth `window // 2 + 1` with `boundary="none"` (590-599) is
   one more than the window needs (wasted halo; harmless).
9. Mask conversion by multiplication: `np.arange(npix) *
   ~signal_mask.flatten()` (104) maps masked pixels to index 0 instead
   of dropping them, and `_optimise_lambda` uses the opposite
   polarity (`np.ones(...)`, 190-195) (D9.1).
10. Sigma estimation materialises the whole dataset in memory
    (`np.asarray(self.data, dtype=float32)`, 88, 541-544), also for
    lazy signals; the averaging ignores saturation while the
    PyEBSDIndex sigma applies it (inconsistent `N`) (D3.5, D8.2).
11. Masked pixels keep their original values while unmasked pixels
    are averaged (349-352); PyEBSDIndex averages all (D9.1).
12. `dask_config_kwargs` defaults to `num_workers=2`,
    `scheduler="threads"` inside the method (632-643) (D8.8).

### Convergence rule

When #824 or a successor lands upstream and the fork merges upstream,
the fork keeps its implementation, tests and tutorial under the shared
names (take-ours on every conflict in `_nlpar.py`, the three methods
and the NLPAR tests), reconciling only keyword names (D11.6); the
docstring's "Differences" paragraph is re-pointed at the merged text.
#824's `code_style.rst` note and `test_kikuchipy_h5ebsd.py` repr fix
arrive with that merge and are not duplicated now.

### Licensing

`_nlpar.py` is GPL-3.0-or-later (kikuchipy default header, like
`chunk.py`) plus the delimited NRL public-domain derivation block of
D10.4 (verbatim notice, change list with date and nature,
acknowledgement); public-domain code may be incorporated into a GPL
work and the notice's requests are honoured in full. Nothing is
imported from kikuchipy's BSD-3 files (`_constants.py` is only read
through `dependency_version` in tests). EMsoftOO `mod_NLPAR.f90` is
BSD-3 (header lines 1-12): equations only, no code, no notice. The
PyEBSDIndex acknowledgement appears in the module block, the method
Notes (D1.7), the CHANGELOG bullet (D12.5) and the tutorial (D12.1);
the paper is cited with `:cite:`brewick2019nlpar`` and, in the
tutorial, `<cite data-cite="brewick2019nlpar">`.

### Constitution pointers

`specs/tech-stack.md`: numba rules :44, licence-block convention :45,
`py_func` testing :49, MTP convention :51, notebook rules :52,
CHANGELOG convention :53, process :70-84; the float64 rule :21 is
EMSphInx-scoped (NLPAR is float32 by parity, D7.2), as HREBSD's D17
scoped it. `specs/roadmap.md` gains a "Feature path: NLPAR" section
after Phase 12 (:118-124), not a Phase 13; `specs/mission.md` a "Fork
feature path: NLPAR" paragraph after :53. The amendment texts are in
plan.md section 0, applied by the main loop in the spec commit. Spec
shape: `hrebsd-dic:specs/2026-09-07-hrebsd-dic/`.

### Sources

Brewick, Wright and Rowenhorst 2019 (`brewick2019nlpar`); PyEBSDIndex
0.3.10.1 `nlpar_cpu.py`; EMsoftOO `mod_NLPAR.f90` and
`mod_DIfiles.f90`; pyxem/kikuchipy#824 and #230; PyEBSDIndex#80 and
#83; `ebsd.py:954-1230`, `chunk.py:130-164`, `_dask.py:198-216`; the
parked plan `specs/_research/plan-nlpar-2026-09-11.md`; the approved
session plan `how-do-run-a-cuddly-possum.md`.
