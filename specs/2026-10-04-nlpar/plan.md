# NLPAR -- `feat-NLPAR`: plan

**Status (2026-10-04):** drafted on `feat-NLPAR` (= fork `develop`
@ 18d59c07, equal to `origin/develop`; `hrebsd-dic` 02a529c0,
`feat-spherical-indexing` 6723aaf0, `upstream/develop` 31666938,
`stash@{0}` present and untouched). Awaiting the user's approval of
this file (the plan gate of tech-stack.md:74); nothing is committed.
The design is the parked plan `specs/_research/plan-nlpar-2026-09-11.md`
(2026-09-11), executed with the user's 2026-10-04 amendments: base
`develop`, one fork PR into `develop`, fan-out to `hrebsd-dic` and to a
clean branch, and the API names of upstream PR pyxem/kikuchipy#824.
Models per the recorded working practice: this spec on Fable 5 (xhigh,
ultracode Workflow); tests, implementation, adversarial review, bug
injection and fixes by Workflow agents with `model: "opus"`, effort
`xhigh`; commits and pushes by the main loop only. (Amended
2026-10-05: the model choice above is history for the work done so
far; going forward the owner's model rule of 2026-10-05 applies:
tests, implementation, review, bug injection and fixes as Workflow
agents on `{model: 'opus', effort: 'medium'}`, spec work on Opus 5.5
xhigh, Fable only as an escalation when repeated errors and an
inconsistency occur. See section 5.) Tests are written
failing before the code they exercise. Three build stages under this
one spec folder; `requirements.md` decisions govern; `validation.md`
holds the V0-V11 oracle suite and the append-only Recorded results
ledger. Spec-stage workflow: three parallel drafters -> three
read-only critics -> fixer with a disposition table (section 10) ->
user approves `plan.md` -> commit 1. Re-submission of the three
documents to review after Stage C is a definition-of-done gate.

Drafting measurements (read-only python, this machine, 2026-10-04;
installed pyebsdindex 0.3.10.1, numba 0.65.1, numpy 2.4.6, orix
0.14.2, dask 2026.3.0): PyEBSDIndex `NLPAR.sigma_numba.py_func` and
`NLPAR.nlpar_nb.py_func` run in pure Python (both carry
`parallel=True, fastmath=False`, `nlpar_cpu.py:753, 821`); `nlpar_nb`
costs ~3.2 us per inner-loop iteration in pure Python (8x8 map of
16x16 patterns, sr 3: 2.59 s for 8.0e5 iterations; 5x5 map of 60x60,
sr 1: 1.62 s for 8.1e5; 7x7 map of 60x60, sr 3: 19.70 s for 8.6e6);
`sigma_numba.py_func` 0.05-0.46 s on the same inputs. Two oracle
quirks measured: an axis shorter than `2 sr + 1` raises `IndexError`
inside `nlpar_nb` (the recorded no-bounds-check quirk), and
`nlpar_nb`'s `calclim` default is a mutable array the kernel writes
into, so a second pure-Python call inherits the first call's limits
(a 3x3 map after an 8x8 map raised `IndexError`): every oracle call
passes `calclim=np.array([0, 0, ncols, nrows])` explicitly. dask
`map_overlap(boundary="none")` with depth 4 on chunks (26, 26, 3)
rechunks to (26, 25, 4) itself (`ensure_minimum_chunksize`; block
sizes 30/33/8 observed) and its `block_info[0]` carries exactly
`array-location` (in overlapped coordinates), `chunk-location`,
`num-chunks` and `shape`.

Spec-review measurements (2026-10-04, read-only python; critic E1's
`scratchpad/critic_numerics_probe.py` re-run by the fixer, plus the
fixer's own dask check; recorded in requirements D13.6 and ledger
entry 2): the PyEBSDIndex oracle's `.py_func` is NOT bitwise with its
compiled kernel (`sigma_numba`: sigma bitwise, `dout` not; `nlpar_nb`:
2586 of 3136 pixels differ, max abs 1.07e-4, for a float and a float32
`lam` alike), because NumPy and numba promote `float32 ** int32` and
`2.0 * float32` oppositely and numba promotes the `n2 += 1.0` counter
to float64; cold compile 4.0 s + 3.2 s. Consequence: every parity test
calls the COMPILED dispatchers (section 7.4). dask 2026.3.0:
`da.overlap.overlap(x, depth, boundary="none")` followed by
`da.map_blocks(f, overlapped, chunks=nav_chunks + ((4,), (9,)),
dtype=float32, meta=np.empty((0, 0, 4, 9), float32))` with a wrapper
that returns only its core computes the right `(10, 12, 4, 9)` result,
makes no `block_info=None` probe call, and `block_info[0]` under
`map_blocks` carries `array-location`, `chunk-location`, `num-chunks`,
`shape`; `ensure_minimum_chunksize(6, (26, 26, 3)) == (26, 23, 6)`,
`(5, (47, 8)) == (47, 8)`; the depth rule gives `(47, 8)`: r3 -> 3,
r4 -> 4; `(26, 26, 3)`: r3 -> 4, r4 -> 6.

Spec-review round 2 measurements (2026-10-04, fixer, read-only
python `scratchpad/fixer2_probe.py`, scipy 1.17.1; requirements D13.7,
ledger entry 3): bounded Nelder-Mead on the D5.1 objective from
`x0 = 1.0` with every non-self slot at `c = 200` stays at `1.000000`
(flat objective in float64, `nit` 10), with four slots at `c = 1` and
four at `c = 250` converges to `1.176` (critic F2's proposed upper arm
does not hit the bound), and with one slot at `c = 1` and seven at
`+inf` returns `10.0` (upper bound hit, the arm adopted); all slots at
`c = 1e-4` return `0.0084`, at `1e-7` return `1e-3` (lower bound hit).
A 70/30 field (`c` 2 / 12) at fixed `lam = 1.5`: mean objective
`0.2616`, median `0.1068`; its mean-objective minimiser from `x0 = 1.0`
is the kink `1.1884`. Phantom-free objective on the border-masked
field: invalid slots at `d = 0` and at `+inf` give EXACTLY the same
value; the phantom-counting objective differs by `0.049` / `0.042`
at `lam` 1.0 / 2.0. Sequential float32 accumulation of `k` copies of
`fl(fl(1/k) p)`: worst `2 / 3 / 11` ulp at `k = 9 / 20 / 49`.
`nickel_ebsd_large.extract_grid((11, 15))` returns data `(13, 10, 60,
60)`, axes manager `(11, 15)`, `xmap` `(49, 64)` (inconsistent);
`s.inav[::5, ::5]` returns data `(11, 15, 60, 60)`, `xmap.size` 165
(orix extent shape `(51, 71)` under step slicing), detector and static
background carried. `tests/` and `tests/test_signals/` have no
`__init__.py` and no `conftest.py`; `pyproject.toml:167`
`--import-mode=importlib`.

Spec-review round 3 measurements (2026-10-04, fixer, read-only
python `scratchpad/fixer3_probe.py` plus critic C3's two probes
re-run; requirements D13.8, ledger entry 4): `get_dask_array(signal=
<LazyEBSD>, chunk_bytes=8e6, rechunk=True)` runs `_reduce_chunks`
(`_dask.py:142-151, 163-195`), which keeps the chunks of the
navigation axis with the smaller chunksize and re-chunks the other
axis to the 8 MB limit: on (10, 16 | 16, 16) uint8 every V7 chunking
keeps its row chunks and gets `(16,)` columns (`((3, 3, 4), (7, 7,
2))` -> `((3, 3, 4), (16,))`, `(1, 1)` -> `((1,) * 10, (16,))`);
on (55, 75 | 60, 60) `((26, 26, 3), (75,))` -> `((26, 26, 3), (40,
35))`; the issue-230 tuple is unchanged. `ensure_minimum_chunksize(4,
(3, 3, 4)) == (6, 4)`, `(6, (7, 7, 2)) == (7, 9)`, `(6, (1,) * 16) ==
(6, 10)`, `(6, (1,) * 10) == (10,)`; the function's source at the
dask `2021.08.1` tag is identical to 2026.3.0's and `overlap(x,
depth, boundary)` exists there. `average_neighbour_patterns(inplace=
True)` on a `((47, 8), (47, 28))`-chunked in-memory (55, 75 | 60, 60)
signal equals `inplace=False` bitwise under the synchronous scheduler
and threads 1/2/20 (0 differing patterns). Compiled `sigma_numba` on
a constant (4, 5 | 6, 6) map at 100: protection on, `nout` stays at
the `1e-12` seed (no kept pair), sigma `1e12` everywhere; protection
off, `nout = 36`, sigma `1e12`. `8 exp(-200 / lam^2) < 2^-53` for
`lam < 2.27` (the flat D5.4 objective is a rounding of `1 + 8 w`, not
an underflow). Line facts: `nlpar_cpu.py:181` assigns `self.lam`;
`si_wafer` spans `_data.py:392-449`; the `ebsd.py` divergence hunks
between `feat-spherical-indexing` and `develop` are at 1673 and 1776.

## 0. Constitution amendments (applied in the spec commit)

The main loop appends the texts below verbatim (Write/Edit tools, LF,
no BOM; `roadmap.md` line 1 unchanged). Nothing else in the three
files changes. `specs/2026-08-16-constitution/upstream-issue.md` and
`specs/_research/plan-upstream-merge-0.13.1.md` are never touched.

### 0.1 `specs/mission.md`, appended after line 53 (end of file)

```
## Fork feature path: NLPAR (recorded 2026-10-04)

Beside the spherical indexing project above, the fork carries NLPAR
(non-local pattern averaging, Brewick, Wright and Rowenhorst,
*Ultramicroscopy* 200 (2019) 50-61, doi 10.1016/j.ultramic.2019.02.013)
as its own feature path on branch `feat-NLPAR`, spec
`specs/2026-10-04-nlpar/`: a numba CPU implementation of the noise
estimate, the lambda optimisation and the weighted non-local average
as `EBSD.average_non_local_neighbour_patterns()`,
`EBSD.get_nlpar_sigma()` and `EBSD.get_nlpar_lambda()`, with a
tutorial `doc/tutorials/nlpar.ipynb` and a gallery example. PyEBSDIndex's
public-domain `nlpar_cpu.py` (US Naval Research Laboratory) is the
numerical oracle and the derivation source, never a runtime
dependency; the module carries the NRL change notice. The method name
and module path follow upstream pull request pyxem/kikuchipy#824 so
that the fork converges with upstream when an NLPAR lands there.
Unlike the spherical phases, this path is built once on fork `develop`
(one PR, `feat-NLPAR -> develop`) and fanned out by a merge into
`hrebsd-dic` and a clean replay onto `feat-spherical-indexing-nlpar`;
`specs/roadmap.md` carries the gates and `specs/tech-stack.md` the
rules.
```

### 0.2 `specs/roadmap.md`, appended after line 124 (end of file)

```
---

# Feature path: NLPAR (branch `feat-NLPAR`; spec `2026-10-04-nlpar`)

Not a Phase 13: NLPAR is not in the EMSphInx dependency chain above.
Base is fork `develop` (18d59c07); one fork PR `feat-NLPAR -> develop`
(expected jwestraadt/kikuchipy#17, confirmed with `gh pr list` at PR
time), merged only on the user's go with ubuntu/windows CI green
(the go was given in advance on 2026-10-04, so the merge proceeds
when CI is green; macOS known red: the ebsdsim step and the pre-existing pseudo-symmetry
`23 == 22` test). A box ticks only when the work is committed on
`feat-NLPAR` (`git log`). Gate list per code stage: spec recorded ->
failing tests committed -> implementation -> adversarial review + bug
injection + fixes -> pre-commit clean -> CHANGELOG entry -> pushed.
Stage C is documentation: it skips the failing-tests gate and keeps the
CHANGELOG gate (it ships a tutorial).

## Stage A -- engine
- [ ] `src/kikuchipy/pattern/_nlpar.py`: five `@njit(cache=True, nogil=True)` kernels (`_window_bounds`, `_nlpar_sigma_kernel`, `_nlpar_distances_kernel`, `_nlpar_weights_kernel`, `_nlpar_weighted_sum_kernel`; no `parallel`, no `fastmath`), chunk wrappers reading `block_info`, the NLPAR depth helper, the eager two-pass driver; NRL change notice in the module header
- [ ] `EBSD.average_non_local_neighbour_patterns()` (eager signals; `lam=None`, lazy input and `lazy_output=True` raise `NotImplementedError` until Stage B) and `EBSD.get_nlpar_sigma()` directly after `average_neighbour_patterns` in `signals/ebsd.py`, then `EBSD.get_nlpar_lambda()` as a stub raising `NotImplementedError` until Stage B (the fifth guard)
- [ ] Tests: `tests/test_signals/test_util/test_nlpar.py` (kernel discipline + `.py_func`, PyEBSDIndex kernel oracles skipif incl. the compiled parity arms on `nickel_ebsd_large` [download], depth/`calclim`, the two pyebsdindex-free multi-chunk driver tests, policy oracles), `tests/test_signals/test_ebsd_nlpar.py` (float64 NumPy reference, identities, iid-noise, two-grain, method contracts incl. the multi-chunk in-place arm on `nickel_ebsd_large`) and the four synthetic generators as fixtures in the root `conftest.py`; every tolerance measured then pinned in `validation.md`
- [ ] Adversarial review (fidelity vs the paper, both PyEBSDIndex kernels and EMsoftOO `mod_NLPAR.f90`; conventions/integration) + bug injection (M1-M22 and S1-S8, plan section 6, Stage A rows) + fixes; coverage 100 % of `_nlpar.py` recorded
- [ ] Gates: `-n 0` then `-n 4` (red tests re-run alone), full suite, `SKIP=licenseheaders` pre-commit on explicit files, oldest-matrix recipe, clean-replay grep; CHANGELOG "Added" bullet with the fork PR link; signed commits pushed (the failing-tests commit never alone)

## Stage B -- optimisation and scale
- [ ] `EBSD.get_nlpar_lambda()` and `lam=None` (phantom-free Nelder-Mead on the pass-1 distances, `target_weight` 0.34, bounds [1e-3, 10], result logged, bound hit warned); lazy input and `lazy_output` through the two `overlap` + `map_blocks` passes (core-only chunk wrappers) with the depth helper and the minimum-chunksize rechunk
- [ ] Pins: eager == lazy bitwise through the method over single, regular, irregular tiny-edge and thinner-than-depth ROW chunkings (a lazy input's column chunking is collapsed by `get_dask_array`'s `_reduce_chunks`; column and both-axes chunkings are pinned at driver level in Stage A) and the explicit (26, 26, 3) last-chunk rows on `nickel_ebsd_large`; issue-230 chunking; 1-D scan == (1, n); scheduler/thread invariance; `nickel_ebsd_large` in the default suite (`allow_download=True`, cached: full-map ADP/IQ gain, the phantom-ratio arm, Hough indexing on the `inav[::5, ::5]` 165-pattern subset; lambda seeds 2026-09-11: 1.1164 / 2.5246 with phantoms, 1.1387 / 2.5787 phantom-free) with the full-map Hough and `si_wafer` weekly; performance baselines recorded, never gated
- [ ] Adversarial review + bug injection (lambda and lazy mutants) + fixes; coverage 100 % of `_nlpar.py` re-recorded
- [ ] Gates as Stage A; CHANGELOG bullet extended for `lam=None`; signed commits pushed

## Stage C -- tutorial
- [ ] `doc/tutorials/nlpar.ipynb` (formulas + acknowledgement; synthetic two-grain demo with sigma map and boundary preservation vs Gaussian `average_neighbour_patterns`; `nickel_ebsd_large` sigma map, lambda-vs-target curve, before/after patterns, IQ/ADP maps, Hough indexing before/after; `si_wafer` numbers quoted from the ledger; parameter guidance; differences from PyEBSDIndex and from upstream #824); `hybrid_indexing.ipynb` untouched, linked
- [ ] Registration: `doc/tutorials/index.rst` after `pattern_processing`, `NOTEBOOKS` entry in `run_nbval.sh`, `tutorials_sanitize.cfg` regexes as needed, stored outputs if > ~2 min on the RTD builder; gallery example `examples/pattern_processing/nlpar.py`
- [ ] Validation matrix + failure-mode review (clean-kernel execute, nbval, html render inspection, linkcheck, name/spell pass) + fixes; `sphinx-build -b html` exit 0
- [ ] CHANGELOG tutorial bullet; signed commit pushed; the three spec documents re-submitted to review (definition of done)

## Fan-out (plan section 1; after the merge)
- [ ] Fork PR `feat-NLPAR -> develop` opened with the PR template (number confirmed; CHANGELOG links rewritten if not #17); roadmap tick commit "Tick NLPAR boxes in roadmap (jwestraadt/kikuchipy#17)"
- [ ] PR merged on the user's go (merge commit; ubuntu/windows CI green); merge sha M recorded here
- [ ] `hrebsd-dic`: `git merge --no-ff develop`, append-type conflicts resolved HREBSD first then NLPAR; `-k "nlpar or hrebsd"` then the full suite green; nbval on `nlpar.ipynb`; pushed; still never merged into `develop`
- [ ] `feat-spherical-indexing-nlpar`: clean replay of M with `pick.ps1`/`gate.ps1` as two commits ("Add non-local pattern averaging (NLPAR)", "Add NLPAR tutorial"; `Staged-from:` trailers), equivalence gate and `specs/` grep clean, worktree suite == baseline + NLPAR tests; pushed, no PR; `feat-spherical-indexing` stays at 6723aaf0
```

### 0.3 `specs/tech-stack.md`, appended after line 84 (end of file)

```
## NLPAR feature path (branch `feat-NLPAR`, spec `specs/2026-10-04-nlpar/`; recorded 2026-10-04)

Everything above applies on `feat-NLPAR` too, with these additions and scopings.

- **Base and branch policy.** `feat-NLPAR` is cut from fork `develop` at 18d59c07; the feature merges back through one fork PR `feat-NLPAR -> develop` (merge commit, on the user's go, ubuntu/windows CI green; macOS red is the known ebsdsim step plus the pre-existing pseudo-symmetry `23 == 22` test). Update rule: merge `develop` into `feat-NLPAR` if needed; **never merge `feat-spherical-indexing` into `develop`**, and `hrebsd-dic` stays never-merged too. The upstream 0.13.1 merge stays parked (`specs/_research/plan-upstream-merge-0.13.1.md`, untouched).
- **Fan-out after the merge.** (1) `hrebsd-dic` receives NLPAR by `git merge --no-ff develop`; every expected conflict is an append (CHANGELOG, `doc/tutorials/index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg`, `bibliography.bib`, the ends of the three constitution files, the `ebsd.py` import block), resolved by keeping both sides, HREBSD first. (2) A new branch `feat-spherical-indexing-nlpar` off `feat-spherical-indexing` (6723aaf0, untouched) receives the merge sha by the staging replay (`C:\Users\westraadt.1\Repos\_staging\pick.ps1`, then `gate.ps1`, in a worktree with `PYTHONPATH=<worktree>\src` and the `kikuchipy.__file__` guard) as two clean commits, feature and tutorial, with `Staged-from: jwestraadt/kikuchipy#17 (<merge sha>)` trailers; `specs/` stripped; no PR; the equivalence gate compares the `+`/`-` lines of `git diff M^1 M -- . ':!specs'` with the replay diff.
- **dtype policy (NLPAR-scoped, like the HREBSD scoping).** The float64 rule under "Numerics" is EMSphInx-scoped. NLPAR kernels accumulate in float32 by design, for bitwise parity with PyEBSDIndex's `nlpar_nb`/`sigma_numba`; float32 literals are cast explicitly, no `**` is applied to a float32 operand (squares are products) and every intermediate the compiled oracle holds in float64 carries an explicit `np.float64` cast, so `.py_func` parity holds under NumPy 1.23 and NumPy 2 alike; the `exp` kernel's `.py_func` parity is measured then pinned (seed 1 float32 ulp), every other kernel bitwise. Output dtype = input dtype by default through `np.rint` + clip to the dtype range (a convex combination never leaves the input range; no per-pattern min-max rescale, unlike `average_neighbour_patterns`); `dtype_out="float32"` returns the raw average.
- **PyEBSDIndex licensing and the derivation notice.** `pyebsdindex/nlpar_cpu.py` is a public-domain work of the US Naval Research Laboratory; derivative works must carry a change notice and acknowledge NRL. `src/kikuchipy/pattern/_nlpar.py` carries kikuchipy's GPL header plus a delimited third-party block (the `_master_pattern.py:20-57` layout) naming the derived functions, the change list and the date; `EBSD.average_non_local_neighbour_patterns` Notes and the CHANGELOG acknowledge PyEBSDIndex and NRL; the method cites `:cite:`brewick2019nlpar`` (`doc/user/bibliography.bib:11`). EMsoftOO `mod_NLPAR.f90` (BSD-3) is an equation cross-check only; no code is ported from it.
- **PyEBSDIndex is optional.** Nothing under `src/` imports `pyebsdindex` for NLPAR, and there is no runtime fallback: sigma and lambda come from the fork's own kernels. Oracle tests are `@pytest.mark.skipif(dependency_version["pyebsdindex"] is None, ...)` (the `tests/test_signals/test_ebsd_hough_indexing.py:35-37` pattern), import `pyebsdindex.nlpar_cpu` inside the test and call the COMPILED `NLPAR.sigma_numba` / `NLPAR.nlpar_nb` dispatchers behind a module-scoped warm-up fixture (their `.py_func` is NOT bitwise with the compiled kernels: NumPy and numba promote `float32 ** int32` and `2.0 * float32` oppositely, measured 2026-10-04; cold compile ~7 s), with a fresh `calclim` array per call and every navigation axis >= `2 sr + 1`, `-n 0` first because the compile writes the shared numba cache; the small synthetic maps and `nickel_ebsd_large` (cached download) run in the default suite, the file-based end-to-end oracle `@pytest.mark.weekly`. The float64 NumPy transcription oracle keeps the minimum-requirement CI job (no pyebsdindex) meaningful. The `pyproject.toml:79` floor `pyebsdindex >= 0.3.9.2, != 0.3.10` is unchanged; 0.3.9.2's two kernels are identical to 0.3.10.1's per the 2026-09-11 reading and every parity arm passes bitwise against `pyebsdindex==0.3.9.2`, so no pyebsdindex version gate beyond the floor (amended 2026-10-05). The one gate is a numba-version gate on the ORACLE side only (amended 2026-10-05): 0.3.9.2's `NLPAR.nlpar_nb` (`parallel=True`) does not compile under numba 0.57.0 and compiles from 0.58.0 on, while `sigma_numba` and the fork's own kernels compile under 0.57, so every test that calls `nlpar_nb` skips when `NLPAR_NB_COMPILES` (`numba >= 0.58.0`, in `test_nlpar.py`) is False and the warm-up fixture then compiles `sigma_numba` only (amended 2026-10-05). The CI oldest job (`numba==0.57`) exercises the sigma oracle and every non-oracle test; the full averaging parity against 0.3.9.2 is recorded from the local numba 0.58.1 run of the next bullet (amended 2026-10-05).
- **Oldest-matrix recipe (local, once per stage, recorded):** `uv run --isolated --python 3.10 --extra tests --with "numpy==1.23.0" --with "numba==0.57" --with "orix==0.12.1" --with "pyebsdindex==0.3.9.2" --with "dask==2021.8.1" --with "scikit-image==0.21.0" pytest tests/test_signals -k nlpar -n 0 -q -p no:cacheprovider` (the dask pin is mandatory: the lazy path relies on `dask.array.overlap.ensure_minimum_chunksize`, whose presence in 2021.8.1 was verified against the tag on 2026-10-04 and this run confirms). No post-0.57 numba features in the kernels. `--extra tests` added 2026-10-05: `--isolated` omits the tests extra. One recorded local run per stage of the same recipe with `--with "numba==0.58.1"` in place of `--with "numba==0.57"` is part of the oldest-matrix gate from now on: it runs the averaging-oracle arms against `pyebsdindex==0.3.9.2` that the numba 0.57 run skips (amended 2026-10-05).
- **Numba-cache flake rule.** `pyebsdindex` redirects `NUMBA_CACHE_DIR` to a shared `~/.pyebsdindex/numbacache` on import, defeating per-worker isolation under xdist. Run `-n 0` first, then `-n 4`; a red test under `-n 4` is re-run alone before it counts as a failure. Two further known flakes, pre-existing and not NLPAR, never block a gate (recorded 2026-10-05): `tests/test_indexing/test_ebsd_refinement.py::TestEBSDRefineOrientationPC::test_refine_orientation_projection_center_local_nlopt` segfaults intermittently (an access violation in the numba refinement objective called by nlopt on a dask thread; 1 of 5 runs alone, 3 of 10 on a develop snapshot), and upstream `tests/test_simulations/test_kikuchi_pattern_simulator.py::TestCalculateMasterPattern::test_shape` passes only through its reruns (`flaky(reruns=5)`).
- **Fixtures.** Test data only from `src/kikuchipy/data/**` (`nickel_ebsd_small`; `nickel_ebsd_large(allow_download=True)` in the default suite once cached, its full-map Hough indexing weekly; `si_wafer(allow_download=True)` weekly) or generated in the test with fixed seeds; the NLPAR synthetic generators are plain functions in the root `conftest.py` exposed as fixtures (pytest runs with `--import-mode=importlib` and `tests/` has no `__init__.py`, so test modules never import each other). No new data files.
- **CHANGELOG.** Fork PR-link convention: `` (`#17 <https://github.com/jwestraadt/kikuchipy/pull/17>`_) ``, the number confirmed with `gh pr list` at PR time and rewritten if it differs.
- **Clean-replay rule (enforced at every stage gate).** Nothing under `src/`, `tests/`, `doc/` or `examples/`, nor the root `conftest.py`, `benchmarks/` or `CHANGELOG.rst` (all carried verbatim by the clean replay, which strips `specs/` only), may name a `specs/` path, a spec file name (`requirements.md`, `plan.md`, `validation.md`, `tech-stack.md`) or a spec D/V number; comments state the fact or the measurement itself. Gate (the executable form of "`git grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md" develop..` finds nothing new in those paths"): `git diff develop...HEAD -- src tests doc examples benchmarks conftest.py CHANGELOG.rst | grep -E "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md| [DV][0-9]"` prints nothing (the ` [DV][0-9]` alternative catches a bare D/V number; a hit is a spec reference to rewrite as the fact itself). The same grep on the replay diff is the clean branch's equivalence gate (section 1). This avoids the `rewrite_specs_refs.py` pass the spherical staging needed.
- **Upstream convergence.** API names follow upstream pyxem/kikuchipy#824 (`pattern/_nlpar.py`, `EBSD.average_non_local_neighbour_patterns`; `window_shape = 2 * search_radius + 1`, `lamda = lam`). When upstream lands an NLPAR under the shared name and the fork merges upstream, the fork keeps its own implementation (take-ours), reconciling keyword names only. #824's defects (border replication, per-pattern rescale, `dthresh` only in the lambda fit, gaussian window cast to bool, undefined names, `np.std` sigma fallback) are recorded in the spec, not reproduced; #824's `doc/dev/code_style.rst` citing note is not duplicated.
- **Type hints and output.** `X | Y` and `bool | None` (no `Optional`/`Union`); no `print()` in `src/` or tests; the optimiser reports through `logging`.
```

### 0.4 `specs/_research/plan-nlpar-2026-09-11.md`, status header (replaces lines 3-10, the `**Status (2026-09-11):**` paragraph; line 1 and everything from `## Context` at line 12 unchanged)

```
**Status (2026-10-04):** SUPERSEDED by `specs/2026-10-04-nlpar/`
(`requirements.md`, `plan.md`, `validation.md`), which carries this
design into execution with three changes decided by Johan on
2026-10-04: the base is fork `develop` (18d59c07), not
`feat-spherical-indexing`, so "Decisions by Johan" item 1 below is
superseded (one fork PR `feat-NLPAR -> develop`, then a merge into
`hrebsd-dic` and a clean replay onto `feat-spherical-indexing-nlpar`;
Step 0 drops out because `develop` tracks `specs/`); the API follows
upstream pyxem/kikuchipy#824 (`src/kikuchipy/pattern/_nlpar.py`,
`EBSD.average_non_local_neighbour_patterns`, with `get_nlpar_sigma`
and `get_nlpar_lambda` as below); and the spec folder is re-dated.
The body below stays as the design record and is not edited.
```

## 1. Branch policy and CI implications (binding, user decision 2026-10-04)

- **Base and target.** `feat-NLPAR` is cut from fork `develop` @
  18d59c07 (verified 2026-10-04 equal to `origin/develop`). All work
  (spec, tests, implementation, fixes, tutorial) commits onto
  `feat-NLPAR` in gate order. ONE fork PR `feat-NLPAR -> develop`,
  opened after Stage C with `gh pr create --repo jwestraadt/kikuchipy
  --base develop` and the PR template; the number is expected to be
  #17 (`gh pr list --state all` on 2026-10-04 shows #16 as the newest,
  merged) and is confirmed at PR time; CHANGELOG bullets carry the
  fork PR-link convention and are rewritten if the number differs.
- **Merge gate.** Merge only on the user's go, as a merge commit, with
  ubuntu and windows CI green. macOS is known red for two pre-existing
  reasons (the ebsdsim step, fixed only by the parked 0.13.1 upstream
  merge, and the pseudo-symmetry `23 == 22` test); both are recorded,
  neither blocks. The fork's tests workflow runs on every push
  (`.github/workflows/tests.yml:22-29`), so pushes of `feat-NLPAR` are
  an EXTRA signal; the local gates of validation.md carry the recorded
  verification burden. **Push policy:** a failing-tests commit is never
  pushed alone; it rides with its stage's implementation commit.
- **Supersession.** This replaces decision 1 of the 2026-09-11 parked
  plan (PR into `feat-spherical-indexing`, specs tree re-adopted on a
  clean base). Consequences that disappear: Step 0 (`.gitignore`,
  pre-commit exclude and `.git/info/exclude` edits), the "merge brings
  the specs tree onto the staging branch" consequence, and the
  cherry-pick-per-commit upstream story for NLPAR; what replaces it is
  the fan-out below. `requirements.md` records the supersession.
- **Fan-out, step 1 (`hrebsd-dic`, after the merge sha M exists):**
  `git switch hrebsd-dic && git merge --no-ff develop -m "Merge develop
  (NLPAR) into hrebsd-dic"`. Expected conflicts are all append-type
  (measured 2026-10-04 in the session plan): `CHANGELOG.rst` Unreleased
  bullets, `doc/tutorials/index.rst`, `run_nbval.sh`,
  `tutorials_sanitize.cfg`, `doc/user/bibliography.bib`, the ends of
  `specs/{roadmap,mission,tech-stack}.md`, and the `ebsd.py` import
  block near line 55 (HREBSD) vs line 91 (NLPAR). Resolve by keeping
  both sides, HREBSD first then NLPAR. Gates: no conflict markers;
  `uv run pytest tests -k "nlpar or hrebsd" -n 0`, then `-n 4`, then
  the full suite; nbval on `nlpar.ipynb`; `SKIP=licenseheaders uvx
  pre-commit run --files <resolved non-specs files>`. Push. The policy
  stands: `hrebsd-dic` is never merged into `develop`.
- **Fan-out, step 2 (clean replay):** `git worktree add
  ../kikuchipy-nlpar-clean -b feat-spherical-indexing-nlpar
  feat-spherical-indexing`; in the worktree, `powershell -File
  C:\Users\westraadt.1\Repos\_staging\pick.ps1 M` (first-parent
  squash cherry-pick `-m 1 -n`, strips `specs/`, reports conflicts
  outside `specs/`). Resolve the 24-commit-gap conflicts (CHANGELOG
  bullets re-applied onto the clean branch's Unreleased section;
  `ebsd.py`/`bibliography.bib` hunks if any; the `ebsd.py` divergence
  is 16 docstring lines in two hunks at 1673 and 1776 (4 insertions,
  12 deletions; `git diff feat-spherical-indexing develop --
  src/kikuchipy/signals/ebsd.py`), disjoint from the NLPAR
  insert at 954-1124). Split the staged result by path into two
  commits: **"Add non-local pattern averaging (NLPAR)"** (`src/`,
  `tests/`, `CHANGELOG.rst`, `doc/user/bibliography.bib` if touched,
  reference docs) and **"Add NLPAR tutorial"**
  (`doc/tutorials/nlpar.ipynb`, `index.rst`, `run_nbval.sh`,
  `tutorials_sanitize.cfg`, `examples/pattern_processing/nlpar.py`).
  Messages in `_staging/msg-17a.txt` and `msg-17b.txt`, each ending
  with `Staged-from: jwestraadt/kikuchipy#17 (M)` and the session's
  attribution trailer (format of `msg-16.txt`). `gate.ps1` before each
  commit (no `specs/` paths or conflict markers; ruff check + format
  with the staging ruff, reformat and record if the versions differ;
  nbformat validation; pre-commit with `SKIP=licenseheaders`).
  Equivalence gate: the `+`/`-` lines of `git diff M^1 M -- . ':!specs'`
  equal those of `git diff feat-spherical-indexing
  feat-spherical-indexing-nlpar` apart from the recorded conflict
  resolutions; the clean-replay grep of section 5 re-run on the replay
  diff, `git diff feat-spherical-indexing feat-spherical-indexing-nlpar
  -- src tests doc examples benchmarks conftest.py CHANGELOG.rst | grep
  -E "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|
  tech-stack\.md| [DV][0-9]"`, prints nothing (the root `conftest.py`,
  `benchmarks/` and `CHANGELOG.rst` travel verbatim with the replay,
  which strips `specs/` only; spec review E3-R3-2). Tests in the
  worktree with `PYTHONPATH=<worktree>\src` and the `kikuchipy.__file__`
  guard: `-k nlpar -n 0`, `-n 4`, the full suite (== the recorded
  clean-branch baseline 4118 passed / 827 skipped plus the NLPAR
  tests), nbval on `nlpar.ipynb`. `git push -u origin
  feat-spherical-indexing-nlpar`; no PR anywhere; remove the worktree;
  `feat-spherical-indexing` stays at 6723aaf0.
- **Commits are signed (`git commit -s`)**, carry the attribution
  trailer the session specifies at commit time, use explicit pathspecs
  only, and never sweep the user's uncommitted notebook edits
  (`spherical_indexing.ipynb`, `hybrid_indexing.ipynb`,
  `load_save_data.ipynb`) or the two protected spec files.


**Advance authorisation (2026-10-04, night).** Johan: "Going to bed,
just go ahead do the plan when ready". The two gates this plan held
for him, approval of this `plan.md` before commit 1 and his go before
the fork PR merges, are therefore satisfied in advance: the spec is
committed on the verification round's verdict (round 3: no blockers,
every major applied) and the PR merges when ubuntu/windows CI is green.
Open questions FOR THE USER (section 7) are recorded, not asked, and
go into the morning summary. Nothing else changes: no upstream contact,
fork-only pushes, `hrebsd-dic` and `feat-spherical-indexing` never
merged into `develop`.

## 2. Stage A -- engine

Deliverables: `src/kikuchipy/pattern/_nlpar.py` (kernels + drivers),
`EBSD.average_non_local_neighbour_patterns()` and
`EBSD.get_nlpar_sigma()` in `src/kikuchipy/signals/ebsd.py`, two test
modules, the CHANGELOG "Added" bullet, roadmap Stage A ticks. Numbered
build modules, in implementation order:

1. **Kernel: `_window_bounds(center: int, radius: int, n: int, shift:
   bool) -> tuple[int, int]`** (`_nlpar.py`). `shift=False` returns the
   clipped window `[max(c - r, 0), min(c + r, n - 1) + 1)` (the 3x3
   sigma window, `nlpar_cpu.py:770-774`). `shift=True` returns
   PyEBSDIndex's shifted-inward search window
   `start = max(c - r, 0) - max(c + r - (n - 1), 0)`,
   `stop = min(c + r, n - 1) + max(r - c, 0) + 1` (`:872-873, 877-878`),
   with `start` and `stop` each clamped to `[0, n]`, so the half-open
   `[start, stop)` lies inside `[0, n)` and equals `(0, n)` when `n <
   2 r + 1` (e.g. `n=2, r=3, c=0`: raw `(-2, 5)` -> `(0, 2)`; `stop`
   is the exclusive bound and may equal `n`, requirements D4.2): the
   bounds check PyEBSDIndex lacks, which
   makes an axis shorter than `2 r + 1` safe (its `nlpar_nb` raises
   `IndexError` there, measured 2026-10-04). Bitwise `.py_func` test.
2. **Kernel: `_nlpar_sigma_kernel(data: float32 (n_rows, n_cols,
   n_pix), mask_indices: int64 (n_kept,), max_value: float32,
   saturation_protect: bool) -> (sigma float32 (n_rows, n_cols), d2
   float32 (n_rows, n_cols, 9), n2 float32 (n_rows, n_cols, 9), valid
   bool (n_rows, n_cols, 9))`.** Port of `sigma_numba` (`:754-804`)
   without the normalisation loop: per point, the clipped 3x3 window;
   per neighbour slot (FIXED layout, slot `= (dj + 1) * 3 + (di + 1)`,
   `valid=False` for out-of-map slots), float32 accumulation of
   `diff = d0 - d1; d2 += diff * diff` and `n2 += np.float32(1.0)` (no
   `**` on a float32 operand, no bare literal: numba promotes both
   differently from NumPy, measured 2026-10-04) over `mask_indices` in
   ascending order, `n2` starting at `np.float32(0.0)` (the oracle
   seeds `np.float32(1.0e-12)` at `:785`, rounded away in `s0` for
   every `n2 >= 1`; recorded deviation, requirements D2.1), counting
   a pixel only when both values are `< threshold` with threshold =
   `np.float64(max_value) * np.float64(0.9961)` (protection on) or
   `np.float64(max_value) + np.float64(1.0)` (off): the float32 pixels
   are compared against this float64 value, as the compiled oracle
   does after numba unifies its `mxval` to float64 (`:763-767`;
   requirements D3.5, the factor is the module constant
   `SIGMA_SATURATION_FACTOR = 0.9961`); the averaging kernel's
   threshold stays float32 (module 4); `sigma_i^2 = min over
   neighbours with d2 > 0 of d2 / (2 n2)` (the `d2 > 0` guard is the
   recorded deviation from `d2 >= 1e-3`, `:796`); no neighbour passes
   -> `sigma = 1e12` (`mind = 1e24` at `:776`, `sqrt` at `:804`).
   Self slot: `d2 = 0`, `n2 = n_kept`,
   `valid = True`. Slot mapping to the oracle: PyEBSDIndex enumerates
   the in-map neighbours compactly (row outer, column inner,
   `:779-802`), so `ours[valid]` taken in slot order is its slot order;
   its self slot carries `nout = npix` (the total pixel count, `:761,
   :786`) and, after normalisation, `dout = -sqrt(npix / 2)`, and a
   visited pair without a kept pixel keeps `nout = 1e-12` (`:785`).
   Bitwise `.py_func` test; V2 oracle on sigma.
3. **Driver: `_nlpar_normalized_distances(d2, n2, valid, sigma) ->
   d float32 (n_rows, n_cols, 9)`** (NumPy, no numba; requirements D2.5
   as amended 2026-10-04). The second loop of `sigma_numba`
   (`:806-817`) on assembled whole-map arrays, reproducing the COMPILED
   oracle's promotion chain: `s2 = sigma * sigma` (float32), `s2_ij =
   s2_i + s2_j` gathered by padded shifts (float32), `num = d2 - n2 *
   s2_ij` (float32), `den = np.float64(s2_ij) * np.sqrt(np.float64(2.0)
   * np.float64(n2))` (numba types `2.0 * n2` as float64 there), `d =
   (np.float64(num) / den).astype(np.float32)`; `+inf` where `n2 == 0`
   (weight 0; the recorded deviation, PyEBSDIndex keeps a finite tiny
   negative value there); no other guard (the oracle has none in this
   loop, and D1.6 enforces every `sigma` element finite and `> 0`, for
   a user array too, so `den > 0` wherever `n2 > 0`); `-inf` at the
   self slot
   (weight exactly 1 through `exp(-0.0)`); `NaN`-free. Phantom slots
   keep `valid=False` and are never read by the objective. Doing this
   on the assembled arrays, not inside the chunk wrapper, lets pass 1
   run at depth 1 and makes a user-supplied `sigma` plug straight into
   the lambda objective (open question 7.2). V2 compares its output
   (not the kernel's) with the oracle's `dout`.
4. **Kernel: `_nlpar_distances_kernel(data, sigma2: float32 (n_rows,
   n_cols), mask_indices, max_value, saturation_protect, radius_rows,
   radius_cols, row_start, row_stop, col_start, col_stop) -> d float32
   (kept_rows, kept_cols, (2 rr + 1) * (2 rc + 1))`.** Port of the
   window scan of `nlpar_nb` (`:870-918`) for the kept region only
   (PyEBSDIndex's `calclim`, `:836-843`), window bounds from
   `_window_bounds(..., shift=True)` per axis, slot `= wr * (2 rc + 1)
   + wc` with `(wr, wc)` the window-local row and column indices from
   `_window_bounds` (row-major over the shifted window); per pair
   `d2`/`n2` in float32 over `mask_indices` (`diff * diff`,
   `np.float32(1.0)` increments, `n2` from `np.float32(0.0)`) with the
   float32 threshold `max_value * np.float32(0.999)` (protection on,
   `AVERAGE_SATURATION_FACTOR`) or `max_value + np.float32(1.0)` (off)
   (`:866-869`); the driver passes `sigma2 = (sigma * sigma).astype(
   np.float32)` (float32 product, bitwise the oracle's scalar
   `sigma[j, i]**2`) and the kernel reads `s0 = sigma2[j, i]`, `s1 =
   sigma2[jn, in]` and never squares; then `d = (d2 - n2 (s0 + s1)) /
   dnorm` with `dnorm = (s1 + s0) * sqrt(np.float32(2.0) * n2)` (all
   float32, `:910-913`) when `dnorm > np.float32(1e-8)`; `d =
   np.float32(1e6) * n2` when `dnorm <= 1e-8` and `n2 > 0` (the
   oracle's branch, `:914-915`, kept); `d = +inf` only when `n2 == 0`
   (recorded deviation from `1e6 * 0 = 0`: a pair without a comparable
   pixel gets weight 0, not 1); the self slot `-inf`; when an axis
   window is shorter than `2 r + 1` (a map smaller than the window) the
   in-map points occupy `wr < len_r`, `wc < len_c` and every slot with
   `wr >= len_r` or `wc >= len_c` holds `+inf` (the weighted-sum kernel
   uses the same mapping). No pair memo (`pairdict`, `:874, 897-899,
   917`, saves ~3/48 of the work and is not ported), no `diff_offset`
   (`:894, 926`). Bitwise `.py_func` test; V3 oracle.
5. **Kernel: `_nlpar_weights_kernel(d: float32 (...), lam: float64,
   dthresh: float32) -> float32 (...)`.** The only `exp` kernel, with
   the operand types fixed (requirements D7.3, spec review E1-F8):
   `lam2 = np.float64(1.0) / (lam * lam)` in float64 (the oracle's
   `1.0 / lam**2`, `:847`, evaluated on the Python-float `lam` the test
   passes to `nlpar_nb` directly); `w = max(d - dthresh, np.float32(
   0.0))` in float32 (`:924`); `e = np.exp(np.float64(-1.0) *
   np.float64(w) * lam2)` in float64, stored float32 (`:925`). The
   method casts `lam = float(lam)` and `dthresh = np.float32(dthresh)`
   before any kernel. `-inf -> 1.0` exactly, `+inf -> 0.0` exactly.
   `.py_func` parity MTP, seed 1 float32 ulp (numba vs CPython `exp`),
   `EXP_KERNEL_ULP` (the one placeholder asserted by both test
   modules, V0 and V1: defined in the root `conftest.py`, fixture
   `exp_kernel_ulp`, requirements D1.9).
6. **Kernel: `_nlpar_weighted_sum_kernel(data, weights, radius_rows,
   radius_cols, row_start, row_stop, col_start, col_stop) -> float32
   (kept_rows, kept_cols, n_pix)`.** Sequential float32 weight sum in
   slot order, `w /= sum`, then `out[q] += data[j, q] * w` in slot
   order over ALL pixels (the signal mask affects distances only, as
   PyEBSDIndex, `:929-934`); window bounds recomputed with
   `_window_bounds(shift=True)`. Bitwise `.py_func` test.
7. **Driver helpers** (`_nlpar.py`, Python; `ValueError` messages
   carry the fragments frozen in requirements D1.6):
   `_nlpar_mask_indices(signal_mask, signal_shape) -> int64[:]`
   (kikuchipy polarity, `True` = excluded; shape check first, then
   `np.asarray(signal_mask, dtype=bool)`; `ValueError` on a shape
   mismatch, "signal shape", or when no pixel is kept, "excludes every
   pixel"); `_nlpar_search_radius(search_radius, nav_dim) ->
   tuple[int, ...]` (int broadcast, tuple length must equal `nav_dim`,
   "one radius per navigation axis"; a radius is valid iff
   `isinstance(r, (int, np.integer)) and not isinstance(r, bool) and
   r >= 0`, else "search_radius must be a non-negative int"; the
   method's `navigation_dimension == 0` check runs before this
   helper); `_nlpar_saturation_max(dask_array) ->
   float32` (one global `max` over the whole map, eager or one dask
   reduction; the recorded deviation from PyEBSDIndex's tile-local
   max); `_nlpar_finalize(out_f32, dtype_out) -> ndarray` (`dtype_out`
   must satisfy `np.issubdtype(dt, np.integer) or np.issubdtype(dt,
   np.floating)`, else `ValueError("dtype_out must be an integer or
   floating dtype")`, checked by the method's validation step before
   any kernel runs; integer dtypes: `np.rint` then clip to `omin, omax
   = dtype_range[np.dtype(dtype_out).type]` (`skimage`'s table is
   keyed by scalar TYPES, the `ebsd.py:1081, 1196-1199` form; the
   table is already imported at `ebsd.py:44`); float dtypes: cast;
   never
   `_rescale_neighbour_averaged_patterns`, `chunk.py:148`, which is
   the per-pattern min-max rescale and carries `fastmath=True`).
8. **Depth helper: `_nlpar_depth(chunks: tuple[tuple[int, ...], ...],
   radius: tuple[int, ...]) -> tuple[dict[int, int], tuple[tuple[int,
   ...], ...]]`**, returning `depth`, a dict keyed by ARRAY axis with 0
   on the signal axes (the form `da.overlap.overlap` takes), and
   `chunks_out`, the full post-rechunk chunks of the array; the chunk
   wrappers (modules 9 and 10) take `depth: tuple[int, int]` for the
   two navigation axes only.
   NLPAR-specific (the generic `_get_chunk_overlap_depth`,
   `_dask.py:198-216`, returns `window.n_neighbours` and is correct
   only for windows that never shift). Per navigation axis `i` with
   radius `r`, chunks `c`, length `n`: one chunk -> depth 0;
   `2 r + 1 >= n` -> rechunk the axis to one chunk, depth 0; else
   `depth = max(r, 2 r + 1 - min(c[0], c[-1]))` (derivation: a kept
   point at distance `k < r` from a true edge has the shifted window
   `[0, 2 r]`, so the edge chunk's core plus halo must reach index
   `2 r`; with every chunk `>= depth` after the rechunk, interior
   chunks never see a shifted point). Then `chunks_out =
   ensure_minimum_chunksize(depth, c)` per chunked axis
   (`dask.array.overlap`; measured 2026-10-04: `(26, 26, 3)` at depth 4
   -> `(26, 25, 4)`; `(26, 26, 3)` at depth 6 -> `(26, 23, 6)`;
   `(3, 3, 4)` at depth 4 -> `(6, 4)`; `(7, 7, 2)` at depth 6 -> `(7,
   9)`; `(1,) * 16` at depth 6 -> `(6, 10)`; `(1,) * 10` at depth 6 ->
   `(10,)`, one chunk; the function and its body are the same in the
   oldest pin dask 2021.8.1, verified against the tag 2026-10-04),
   applied explicitly BEFORE the overlap (`x = x.rechunk(chunks_out)`,
   module 11) so the chunk wrappers know the core sizes and the
   `map_blocks` `chunks=` is stated in the post-rechunk chunks; signal
   axes depth 0. Pass 1 uses depth 1 per chunked
   navigation axis (0 unchunked): the clipped 3x3 never shifts, so the
   rule's second term does not apply and `ensure_minimum_chunksize(1,
   c)` is a no-op.
9. **Chunk wrapper, pass 1: `_nlpar_sigma_chunk(patterns, *,
   mask_indices, max_value, saturation_protect, depth: tuple[int, int],
   block_info=None) -> float32 (core_rows, core_cols, 4, 9)`**
   (requirements D8.2, D8.5
   as amended 2026-10-04). Halo sides from
   `block_info[0]["chunk-location"]` vs `["num-chunks"]` (first/last
   chunk along an axis has no halo on the outer side; `array-location`
   is in overlapped coordinates and is NOT used); core = block minus
   halos; runs module 2 on the whole block and returns ONLY the core,
   packed into one float32 array: plane 0 `d2`, plane 1 `n2`, plane 2
   `valid` as 1.0/0.0, plane 3 slot 0 `sigma` (slots 1-8 zero);
   `_nlpar_unpack_sigma_pass(packed) -> (sigma, d2, n2, valid)` with
   `valid = packed[..., 2, :] > 0.5`. The wrapper asserts `block_info
   is not None`: the explicit `meta=` of module 11 removes the probe
   call (measured 2026-10-04), so there is no zero-returning branch.
10. **Chunk wrapper, pass 2: `_nlpar_average_chunk(patterns,
    sigma_block, *, lam, dthresh, radius, mask_indices, max_value,
    saturation_protect, depth: tuple[int, int], dtype_out,
    block_info=None) ->
    (core_rows, core_cols, h, w) of dtype_out`.** `sigma_block` is the
    second nav-chunked operand (shape `nav + (1, 1)`, the `window_sums`
    pattern of `ebsd.py:1060-1065`), carrying the same halo; halo sides
    as in module 9; modules 4 -> 5 -> 6 -> 7 on the core region, the
    core returned (nothing zero-filled, nothing trimmed afterwards).
    V10's Stage A arm calls this wrapper directly on hand-built haloed
    blocks with hand-built `block_info` dicts.
11. **Drivers: `_nlpar_sigma(dask_array, *, mask_indices, max_value,
    saturation_protect) -> (sigma, d2, n2, valid)`** (pass 1: `depth1`
    = 1 per chunked navigation axis, 0 otherwise (module 8's pass-1
    rule; `ensure_minimum_chunksize(1, c)` is a no-op, so `chunks_out
    == x.chunks` and no rechunk is needed), `overlapped =
    da.overlap.overlap(x, depth=depth1, boundary="none")`,
    `da.map_blocks(_nlpar_sigma_chunk, overlapped, ...,
    chunks=x.chunks[:nav_dim] + ((4,), (9,))` (= `chunks_out[:nav_dim]
    + ((4,), (9,))`)`, dtype=np.float32, meta=np.empty((0,) * nav_dim +
    (4, 9), np.float32))`, `.compute()`, unpack -- pass 1 is always
    eager because the lambda objective and the second operand need the
    whole sigma map; then module 3 on the assembled arrays);
    **`_nlpar_average(dask_array, sigma, *, lam, dthresh, radius,
    mask_indices, max_value, saturation_protect, dtype_out) ->
    da.Array`** (pass 2, lazy graph: `depth2, chunks_out =
    _nlpar_depth(x.chunks, radius)` (module 8), then `x = x.rechunk(
    chunks_out)` EXPLICITLY (`x` is rebound; requirements D8.3, spec
    review E3-R3-1) and `nav_chunks = chunks_out[:nav_dim]`;
    `sigma_dask = da.from_array(sigma[..., None, None], chunks=
    nav_chunks + ((1,), (1,)))`, both operands through `overlap(...,
    depth=depth2 on the navigation axes, boundary="none")`, then
    `da.map_blocks(_nlpar_average_chunk, patterns_ov, sigma_ov, ...,
    chunks=chunks_out, dtype=dtype_out, meta=np.empty((0,) * ndim,
    dtype_out))`: the POST-rechunk chunks, never the input array's,
    since on the V7 `(1, 1)` chunking at sr 3 `ensure_minimum_chunksize(
    6, (1,) * 16) == (6, 10)` changes both the block sizes and the
    block count and the pre-rechunk `chunks=` would raise or misplace
    values). This is `map_overlap(trim=False)` with the trim done
    by the core-only wrappers; chosen over `map_overlap` because pass 1
    changes the signal dimensions and `map_overlap`'s `chunks=` would
    have to be stated in overlapped coordinates (measured 2026-10-04,
    drafting measurements above). 1-D navigation is routed as `(1, n)`
    with radius `(0, r)` and reshaped back; 0-D navigation raises
    `ValueError` in the method.
12. **`EBSD.average_non_local_neighbour_patterns(self, search_radius:
    int | tuple[int, ...] = 3, lam: float | None = None, dthresh: float
    = 0.0, target_weight: float = 0.34, sigma: float | np.ndarray | None
    = None, signal_mask: np.ndarray | None = None, saturation_protect:
    bool = True, dtype_out: str | np.dtype | type | None = None,
    show_progressbar: bool | None = None,
    inplace: bool = True, lazy_output: bool | None = None) -> EBSD |
    LazyEBSD | None`**, inserted directly after
    `average_neighbour_patterns` (`ebsd.py:954-1122`) and before
    `downsample` (`:1124`); imports added beside `ebsd.py:91`. Mirrors
    the existing contract line by line, in this CHECK ORDER
    (requirements D1.6, D5.7): (1) `lazy_output and inplace` ->
    `ValueError("'lazy_output=True' requires 'inplace=False'")`
    (`ebsd.py:1019` text); (2) the D1.6 argument validation with its
    message fragments (`lam = float(lam)`, `dthresh = np.float32(
    dthresh)` casts follow); (3) all radii 0 -> warn and return `None`
    (as the `(1, 1)` window does at `:1028-1034`); (4) **Stage A guards
    (pins deleted in Stage B):** `lam is None` -> `NotImplementedError(
    "lambda optimisation lands in Stage B; pass lam=...")`; `self._lazy
    or lazy_output` -> `NotImplementedError("lazy NLPAR lands in Stage
    B")`. Then `get_dask_array(signal=self, chunk_bytes=8e6,
    rechunk=True)` (`_dask.py:114`; for a lazy input this runs
    `_reduce_chunks`, which keeps the finer-chunked navigation axis
    and re-chunks the other to the 8 MB limit, requirements D8.1, and
    `inplace=True` restores `old_chunks` afterwards); `ProgressBar`
    register/unregister;
    `inplace` + eager input: `self.data = averaged.compute()` whatever
    the dtype (the `downsample` precedent for a dtype change in place,
    `ebsd.py:1195-1223`; amended 2026-10-05, requirements D1.5 (b):
    drafted as `averaged.store(self.data, compute=True)` when
    `np.dtype(dtype_out) == self.data.dtype`, but on dask 2021.8.1 that
    store differed from `inplace=False` in 534 of 4125
    `nickel_ebsd_large` patterns under the synchronous scheduler, so
    the fallback below was taken; one extra map-sized buffer at peak;
    `da.store` into the old buffer would also cast silently on a dtype
    change); `inplace` + lazy input + `return_lazy` -> `self.data
    = averaged.rechunk(old_chunks)` whatever the dtype; `inplace` + lazy
    input + `lazy_output=False` (Stage B) -> `self.data =
    averaged.compute()`, the signal becomes eager (requirements D1.5
    (a); never `store` into a dask target); else `LazyEBSD(averaged,
    **self._get_custom_attributes())` and `compute()` unless lazy;
    `gc.collect()`. The multi-chunk eager in-place path is pinned
    bitwise against `inplace=False` on `nickel_ebsd_large` (V7, D1.5
    (b); a synchronous and a threaded arm; the identical
    `average_neighbour_patterns` mechanism measured bitwise under both
    on 2026-10-04 with dask 2026.3.0); the recorded fallback "if it
    ever differs, every eager in-place path assigns `self.data =
    averaged.compute()`" is in force since 2026-10-05 (dask 2021.8.1
    differs, 534 patterns; `average_neighbour_patterns` 183; amended
    2026-10-05). `sigma`: `None` -> pass 1; float -> constant map;
    array -> shape must equal the navigation shape (rc), else
    `ValueError`. `dtype_out`: `None` -> input dtype through `np.rint` +
    clip; any other integer or floating dtype through module 7 (bool,
    complex and object rejected by the D1.6 validation). Docstring:
    numpydoc,
    formulas (sigma, d, w), the keyword map to #824 (`window_shape = 2 *
    search_radius + 1`, `lamda = lam`, `lam=None` optimises),
    `:cite:`brewick2019nlpar``, the PyEBSDIndex/NRL acknowledgement,
    and the Notes paragraph **"Differences from PyEBSDIndex and from
    upstream PR #824"**: the twelve numbered items frozen in
    requirements D12.6, verbatim (the module-level change notice
    additionally names `diff_offset`, `backsub` and `stem_scale`).
13. **`EBSD.get_nlpar_sigma(self, signal_mask: np.ndarray | None =
    None, saturation_protect: bool = True, show_progressbar: bool | None
    = None) -> np.ndarray`**: float32 of the navigation shape (rc), the
    pass-1 driver; lazy input raises `NotImplementedError` until Stage
    B. Placed after module 12. **`EBSD.get_nlpar_lambda(...)`** with
    the frozen D1.4 signature is placed after it IN STAGE A as a stub
    whose body raises `NotImplementedError("lambda optimisation lands
    in Stage B; pass lam=...")` before any validation (the fifth guard,
    requirements D5.7; pinned by `test_stage_a_guards_raise_not_
    implemented`, audited by V0); its body lands in Stage B (module
    3.2).
14. **Tests**, both failing first against `NotImplementedError` stubs;
    class and test names are validation.md's (the naming authority,
    requirements D1.9): `tests/test_signals/test_util/test_nlpar.py`
    (module level: `KERNEL_NAMES` listing the five kernels,
    `_njit_kernel_names` and `_py_func` helpers copied from
    `tests/test_indexing/test_spherical_euler.py:72-96`, the
    module-scoped `pyebsdindex_kernels` warm-up fixture and the oracle
    call recipe; the four fixture generators of validation.md's
    Automated section are plain functions in the ROOT `conftest.py`,
    each exposed by a same-named fixture returning the callable (the
    `dummy_signal` precedent, `conftest.py:195-205`), together with
    the one MTP placeholder both modules assert, `EXP_KERNEL_ULP`, and
    its fixture `exp_kernel_ulp` (requirements D1.9), because pytest
    runs with `--import-mode=importlib` (`pyproject.toml:167`) and
    `tests/` has no `__init__.py`, so nothing is imported from one test
    module into another (requirements D1.9); Stage A classes
    `TestKernels` (V0, the `test_spherical_euler.py:509-529` pattern,
    `_window_bounds` expressions, the `ast.Pow` walk, the `ast`-based
    pyebsdindex import audit), `TestSigmaOracle` (V2, skipif, compiled
    oracle; the `nickel_ebsd_large` arms [A, download]),
    `TestAveragingOracle` (V3, skipif, compiled oracle; Ni arms [A,
    download]; its file-based end-to-end test
    `test_file_based_pyebsdindex_end_to_end` is [B, weekly] and lands
    with module 3.4),
    `TestDepthAndHalo` (V7 driver half: depth rule, single-chunk rule,
    and the two pyebsdindex-free multi-chunk driver tests
    `test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel`
    and `test_pass_two_driver_on_multichunk_dask_array_equals_single_
    chunk` that drive the real `overlap` + `map_blocks` route in Stage
    A), `TestPolicyOracles` (V10 [A] arms);
    Stage B adds `TestLambdaOracle` (V6 oracle half), the V10 [B] arms
    and `TestPerformance` (V11, `record_property`)) and
    `tests/test_signals/test_ebsd_nlpar.py` (module-level float64
    reference `nlpar_reference(data, search_radius, lam, dthresh,
    sigma=None, signal_mask=None, saturation_protect=True)` written
    from the paper's equations with the clipped 3x3 and the
    shifted-inward search window (the whole axis when `n < 2r + 1`),
    independent of `_nlpar.py`; Stage A classes `TestIdentities` (V1,
    incl. `test_reference_agrees_with_the_method_on_random_maps`
    parametrised over `sr`, `lam`, `dthresh`, `signal_mask`,
    `saturation_protect`, the asymmetric radii `(1, 2)`/`(2, 1)` and a
    1-D scan), `TestNoiseOracle` (V4), `TestTwoGrain` (V5),
    `TestLazyAndContracts` (V7 [A] arms incl. `test_argument_validation`,
    `test_stage_a_guards_raise_not_implemented` and the [A, download]
    `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`),
    `TestSigmaMethod`
    (V7, `get_nlpar_sigma` contracts); Stage B adds `TestLambdaMethod`
    (V6 method half), the V7 [B] arms, `TestRealData` (V8,
    download-gated) and `TestSiWafer` (V9, weekly)). Oracle inputs:
    compiled dispatchers only (section 7.4), every navigation axis `>=
    2 sr + 1` (oracle fixtures `(7, 8 | 12, 12)`), a fresh `calclim`
    per call. Both files live under `tests/test_signals/` so the one
    root `tests/test_signals -k nlpar` of every gate command covers
    them.

Deliberately NOT built in Stage A (or ever, unless re-decided): the
pair memo, `diff_offset`, `backsub`, `stem_scale`, `automask`
(`makeautomask`, `:737-750`; the user passes `signal_mask`), file-based
I/O (`calcnlpar_cpu`'s `fileout`), tile-local saturation maxima, the
`rescale=True` path, a GPU backend, a `Window` argument, a pyebsdindex
runtime fallback, any edit of `hybrid_indexing.ipynb` (its NLPAR
pointer at lines 1704-1705 stays; `nlpar.ipynb` links to it), the
`doc/dev/code_style.rst` note of #824.

Stage A gates: failing tests committed (commit 2) -> implementation +
measured pins (V0-V5 and the V7/V10 [A] arms; V11 is Stage B) ->
adversarial review (section 5 workflow) -> bug injection (section 6
Stage A mutants) -> fixes -> `-n 0`, `-n 4` with red tests re-run
alone, coverage 100 % of `_nlpar.py` (the `coverage run` /
`coverage report` pair of section 5, output recorded; amended
2026-10-05: pytest-cov is not in `.venv`), full suite (`uv run pytest
tests -n 4`), `SKIP=licenseheaders uvx pre-commit run --files <changed
files, never specs/>`, oldest-matrix recipe, clean-replay grep,
never-sweep check (`git diff --name-only` lists none of the three
notebooks) -> CHANGELOG bullet -> roadmap ticks -> commit 3 -> push.

## 3. Stage B -- optimisation and scale

Deliverables: `EBSD.get_nlpar_lambda()`, `lam=None`, lazy input and
`lazy_output`, the eager == lazy pins, real-data pins, performance
baselines, the extended CHANGELOG bullet, roadmap Stage B ticks.

1. **Lambda objective and optimiser** (`_nlpar.py`):
   `_nlpar_lambda_objective(lam, d, valid, dthresh, target_weight) ->
   float` = `mean over points of |target_weight - 1 / S_i|` with `S_i =
   1 + sum over VALID non-self slots of exp(-max(d - dthresh, 0) /
   lam^2)` (self slot weight exactly 1 through `-inf`; PyEBSDIndex's
   `+ 1e-12` on the weight sum, `:109`, is not reproduced, requirements
   D5.1). Form and dtypes frozen in D5.1 (spec review E2-F8): `lam`
   comes FIRST and is the float64 `(1,)` array `scipy.optimize.minimize`
   passes, never cast to a Python float; `w = np.exp(-np.maximum(d -
   dthresh, np.float32(0.0)) / lam ** 2)` (float64 by promotion of the
   float32 `d` with the float64 `lam`); `w[~valid] = 0.0`; `S =
   w.sum(axis=-1)` over all nine slots in slot order (NOT `1 + sum of
   the others`: the two sums are not bitwise); `F = float(np.mean(
   np.abs(target_weight - 1.0 / S)))`. The objective has no stride
   argument. These are the two recorded deviations from `loptfunc`
   (`nlpar_cpu.py:106-112`): phantom-free (PyEBSDIndex counts each
   out-of-map slot as weight 1: 776 of 37125 slots on
   `nickel_ebsd_large`, +2.05 % on lambda at target 0.34, 2026-09-11)
   and `max(d - dthresh, 0)` instead of `max(d, dthresh)` (the
   averaging kernel's form, so `dthresh` means the same thing in both
   places). The statistic over points is the MEAN (`np.mean`, `:111`);
   PyEBSDIndex's `np.median` (`:181`) is over its three target fits,
   not over points (mutant M19). `_nlpar_optimize_lambda(d, valid,
   target_weight, dthresh) -> float`: applies `d[::2, ::2]`,
   `valid[::2, ::2]` BEFORE calling the objective when `d.shape[0] *
   d.shape[1] >= 1e6` (`:164`; spec review C2-F12), then
   `scipy.optimize.minimize(_nlpar_lambda_objective, x0=np.array([1.0]),
   args=(d, valid, dthresh, target_weight), method="Nelder-Mead",
   bounds=[(1e-3, 10.0)], options={"fatol": 1e-4})` (`:168-169`),
   `warnings.warn(f"NLPAR lambda optimisation hit the {which} bound
   ({lam:.4f}); the target weight {tw} is not supported by the data",
   UserWarning)` when `lam <= 1.01e-3` ("lower") or `lam >= 9.9`
   ("upper") (requirements D5.4), and emits the one INFO record of
   D5.6 itself (so `get_nlpar_lambda()` and `lam=None` each log exactly
   once per call, spec review E2-F17); one `target_weight` (PyEBSDIndex's
   median of the fits to 0.5/0.34/0.25 is the 0.34 fit because lambda is
   monotone in the target; recorded).
2. **`EBSD.get_nlpar_lambda(self, target_weight: float = 0.34, dthresh:
   float = 0.0, sigma: float | np.ndarray | None = None, signal_mask:
   np.ndarray | None = None, saturation_protect: bool = True,
   show_progressbar: bool | None = None) -> float`**, after
   `get_nlpar_sigma` (the Stage A stub of module 2.13 receives its
   body); a user `sigma` flows into module 2.3's
   normalisation. `lam=None` in the main method calls the same path
   (pass 1 once, shared with the sigma it needs) and logs the chosen
   lambda; the Stage A `NotImplementedError` pin is deleted.
3. **Lazy path.** Lazy input and `lazy_output=True` through the same
   two-pass driver (pass 1 eager, pass 2 lazy); `inplace=True` on a
   lazy signal assigns the rechunked graph (`rechunk(old_chunks)`,
   `ebsd.py:1107`). The Stage A guards and their pin
   (`test_stage_a_guards_raise_not_implemented`) go; the V7 [B] arms
   of `TestLazyAndContracts` in `test_ebsd_nlpar.py` come (names as in
   validation.md): `test_lazy_equals_eager_<chunking>` parametrised
   over the input navigation chunkings single, `(5, 8)`, `((3, 3, 4),
   (7, 7, 2))`, `((1, 9), (2, 14))`, `(1, 1)`, `(2, 3)` and `(4, 4)`
   (signal axes one chunk) of a `LazyEBSD` built from
   `random_uniform_saturated((10, 16), (16, 16), one_block_only=True)`
   (bitwise against the eager result; the fixture plants saturated
   pixels in one block only so a per-chunk maximum is caught). Through
   the method `get_dask_array(rechunk=True)` runs `_reduce_chunks`,
   which keeps the ROW chunks and makes the 16-column axis one chunk
   (measured 2026-10-04: `((3, 3, 4), (7, 7, 2))` -> `((3, 3, 4),
   (16,))`, `(1, 1)` -> `((1,) * 10, (16,))`; requirements D8.1), so
   these arms pin the row chunkings; the column and both-axes
   chunkings are pinned at driver level by the Stage A
   `TestDepthAndHalo` tests (requirements D8.6).
   `test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3`
   (explicit `(26, 26, 3)` rechunk at sr 3 and 4, processed as
   `((26, 26, 3), (40, 35))` after `_reduce_chunks`, the 3-row last
   chunk still exercising the depth rule, plus the measured default
   `((47, 8), (47, 28))` chunking), `test_issue_230_irregular_
   chunks` (the `((3, 3, 4, ...), (75,), (6,), (6,))` chunking of
   `tests/test_signals/test_ebsd.py:1617-1622`, which `_reduce_chunks`
   leaves unchanged), the lazy 1-D arm of
   `test_one_dimensional_navigation_equals_a_one_row_map`,
   `test_scheduler_and_thread_invariance` (`scheduler="synchronous"`
   vs `"threads"` with `num_workers` 1 and 4, bitwise), the lazy arms
   of `test_inplace_lazy_output_contract` (lazy input keeps its chunks
   in place; `lazy_output=True` returns `LazyEBSD`, the
   `test_lazy_output` shape of `test_ebsd.py:1683-1692`; a lazy input
   with `lazy_output=False, inplace=True` ends with `isinstance(s.data,
   np.ndarray)` and data bitwise equal to the eager run, requirements
   D1.5 (a)), the lazy arm of `test_map_smaller_than_the_window`, and
   the V10 [B] arm of `test_nlpar.py::TestPolicyOracles::
   test_saturation_max_is_global`.
4. **Real data.** V6 end to end: `get_nlpar_lambda()` on
   `nickel_ebsd_large` raw and background-corrected, seeds 2026-09-11
   `1.1164 / 2.5246` with phantoms and `1.1387 / 2.5787` phantom-free
   (which pair belongs to which preprocessing state is re-measured at
   this gate, not assumed), lambdas ascend with decreasing target,
   bound-hit warning on a constructed `d`. V8 default subset: ADP seed
   `0.600 -> 0.904` (auto lambda ~2.52) and `0.766` (lambda 0.7) on the
   background-corrected map, IQ measured; Hough indexing on the
   `s.inav[::5, ::5]` subset (165 patterns, navigation (11, 15) rows x
   cols, `xmap.size` 165 in the same row-major order; NOT
   `extract_grid((11, 15))`, which returns an inconsistent signal,
   round-2 measurements above) before/after (skipif pyebsdindex):
   `pq`, `fit`, `nmatch`, `cm` medians and misorientation to the stored
   orientations (provenance settled at this gate, V8); pin only
   measured improvements (`>= 0.5x` the measured gain), "not worse"
   otherwise, said so in the ledger. The phantom-ratio test runs in the
   default suite ([download]); the compiled-PyEBSDIndex full-map
   parity arms (V2/V3) are Stage A deliverables already green.
   Weekly (`@pytest.mark.weekly`, `pyproject.toml:204`): the full-map
   Hough indexing, the file-based PyEBSDIndex end-to-end oracle
   `TestAveragingOracle::test_file_based_pyebsdindex_end_to_end` ([B,
   weekly]: the class is Stage A, this one test lands here), and V9
   `si_wafer(allow_download=True)` (sigma CV, N_eff, IQ gain; 311 MB
   download, cached). Budget: default-suite additions `<= ~90 s` per
   worker including the ~7 s oracle compile, weekly `<= 5 min`.
5. **Performance baselines (V11, recorded, never gated):**
   `record_property` runtimes of pass 1 and pass 2 on `nickel_ebsd_large`
   at sr 3 single-threaded and with the threaded scheduler, plus one
   `benchmarks/` entry if the directory gains a pattern-processing file
   (otherwise `record_property` only; `benchmarks/indexing/` is the only
   existing tree). Seeds 2026-09-11: PyEBSDIndex compiled kernels 0.12 s
   (sigma) + 0.54 s (average) single-thread; pure-NumPy full map 6-14 s;
   expected ours 0.2-0.4 s threaded; `si_wafer` 5-10 s.
6. Adversarial review + bug injection (section 6 Stage B mutants:
   M17, M18, M19, plus the lazy [B] arms of M10, M11, S1, S3, S4 and
   S8, whose Stage A killers are the multi-chunk driver tests) + fixes;
   coverage
   100 % re-recorded; gates as Stage A; CHANGELOG bullet extended
   ("`lam=None` optimises lambda for a target weight"); roadmap ticks;
   commits 4 and 5; push.

## 4. Stage C -- tutorial

Documentation stage: skips the failing-tests gate, keeps the
Phase-11-style validation matrix + failure-mode review and the
CHANGELOG gate. Deliverables: `doc/tutorials/nlpar.ipynb`, its
registration, the gallery example, the CHANGELOG tutorial bullet,
roadmap Stage C ticks.

1. **`doc/tutorials/nlpar.ipynb` cell outline** (hidden first cell;
   thumbnail tag; black at 77, the `black-jupyter --line-length=77`
   hook; `_ = dask.config.set(num_workers=8)` before any progress-bar
   cell; drift-safe print precision; no bare repr with a memory
   address):
   1. Title + intro: what NLPAR does, the paper citation
      (`brewick2019nlpar`), the PyEBSDIndex/NRL acknowledgement, link
      to `hybrid_indexing.ipynb`'s NLPAR pointer and to the
      `pattern_processing` tutorial.
   2. Formulas: sigma estimate, normalised distance, weights, the
      role of `search_radius`, `lam`, `dthresh`, `target_weight`.
   3. Synthetic two-grain demo (the V5 generator): sigma map, weights
      across the boundary underflow to zero, before/after patterns,
      contrast retained; the same map through
      `average_neighbour_patterns(window="gaussian")` blurs the
      boundary (figure).
   4. `nickel_ebsd_large`: load, `remove_static_background`,
      `remove_dynamic_background`; `get_nlpar_sigma()` map plotted;
      `get_nlpar_lambda()` for targets 0.5/0.34/0.25 and the
      lambda-vs-target curve; `average_non_local_neighbour_patterns(
      inplace=False)`; before/after pattern pair; IQ and ADP maps
      before/after with the ledger numbers printed at 3 decimals.
   5. Hough indexing before/after on the background-corrected map's
      `inav[::5, ::5]` subset (165 patterns; the same points as V8's
      default-suite arm) (`detector.get_indexer`, `hough_indexing`):
      median `pq`/`fit`/`nmatch`/`cm` and the misorientation to the
      shipped xmap.
   6. Parameter guidance: when to raise `search_radius`, what
      `dthresh` does, `sigma`/`lam` reuse across maps, `dtype_out`.
   7. Large data: the lazy call on `si_wafer` shown as code with
      ledger-quoted numbers (not executed on RTD; open question 7.6).
   8. Differences from PyEBSDIndex and from upstream #824 (the module
      12 Notes list, in prose).
2. **Registration:** `doc/tutorials/index.rst` "Fundamentals and
   usage" gallery, a new `nlpar` line after `pattern_processing`
   (`index.rst:24`); `"nlpar.ipynb"` in the `NOTEBOOKS` array of
   `doc/tutorials/run_nbval.sh` (alphabetical slot after
   `mandm2021_sunday_short_course.ipynb`); `tutorials_sanitize.cfg`
   gains a regex only if a new non-deterministic output appears
   (timings are regex1/regex2, tqdm regex4, chunk counts regex9).
3. **Stored outputs rule:** estimate the execution on the ~2-vCPU RTD
   builder; store outputs iff `> ~2 min` there (`uv run --with
   ipykernel jupyter nbconvert --to notebook --execute --inplace
   doc/tutorials/nlpar.ipynb`, then strip the notebook-level
   `metadata.widgets` block). Expected: Hough indexing + two NLPAR
   passes on 4125 patterns run in well under 2 min on the drafting
   machine; the decision is measured and recorded.
4. **Gallery example `examples/pattern_processing/nlpar.py`**, modelled
   on `neighbour_pattern_averaging.py` (same header, `# %%` cells,
   `hs.preferences.General.show_progressbar = False`): load
   `nickel_ebsd_large`, backgrounds removed, one pattern kept, NLPAR
   with `lam=None`, IQ histograms before/after, the 2x2 figure.
   `pyproject.toml:175` ignores `examples/*/*.py` in pytest; Sphinx
   Gallery executes it at docs build.
5. **CHANGELOG:** a second "Added" bullet "Tutorial on non-local
   pattern averaging (NLPAR), ``doc/tutorials/nlpar.ipynb``, and a
   gallery example" with the fork PR link (the #12 precedent).
6. **Validation matrix + failure-mode review** (one reviewer agent):
   clean-kernel execute, nbval locally
   (`./doc/tutorials/run_nbval.sh` restricted to the new notebook),
   `uv run sphinx-build -b html doc doc/_build/html` exit 0, html
   render inspection of the figures, linkcheck of the tutorial's links
   via `output.json`, name/spell pass, the never-sweep check. Fixes;
   roadmap ticks; commit 6; push; the spec re-submission critic
   (section 5) closes the feature.

## 5. Stage-independent recipe

Per-stage sequence (main loop between workflows): `nlpar-stage-tests`
-> commit (failing tests + skeletons; not pushed) ->
`nlpar-stage-implement` -> `nlpar-stage-review` -> `nlpar-stage-mutants`
-> `nlpar-stage-close` -> commit (implementation + pins + fixes +
CHANGELOG + roadmap ticks) -> push both. Stage C runs
`nlpar-stage-implement` (notebook author + measurer), `nlpar-stage-review`
(validation matrix + failure modes) and `nlpar-stage-close` only.
Every workflow (amended 2026-10-05: the clause "all agents `model:
"opus"`, effort `xhigh`" is superseded by the owner's model rule of
2026-10-05: tests, implementation, review, bug injection and fixes
run as Workflow agents on `{model: 'opus', effort: 'medium'}`; spec
work on Opus 5.5 xhigh; Fable only as an escalation when repeated
errors and an inconsistency occur): `<= 15`
agents, read/write only within the explicit file list, no git
commands (the main loop commits and pushes), no notebooks swept, no
em-dashes, `X | Y` hints, no `print()`.

- **`nlpar-stage-tests`** (<= 8 agents): skeleton agent (writes
  `_nlpar.py` stubs that raise `NotImplementedError`, the three EBSD
  method stubs (the `get_nlpar_lambda` stub keeps its
  `NotImplementedError` body through Stage A, requirements D5.7), the
  module header = the GPL header copied verbatim from
  `chunk.py:1-16` plus the NRL block, since the `licenseheaders` hook is
  skipped in every gate; and the four generator functions with their
  fixtures plus the shared `EXP_KERNEL_ULP` placeholder and its
  `exp_kernel_ulp` fixture appended to the root `conftest.py`); one
  writer per test
  module (file list: its module plus read access to `conftest.py`; runs
  only its module, `-n 0`, expects the `NotImplementedError` failures
  and no collection errors); test critic
  (read-only: would a plausibly wrong implementation still pass? does
  every section 6 mutant of this stage have a NAMED killer in the
  files? are the MTP placeholders inventoried in validation.md?);
  fixer applying the critic's findings and writing the ledger entry
  ("Stage X failing-tests gate"); at most two critic rounds.
- **`nlpar-stage-implement`** (<= 7 agents): implementers in dependency
  order, kernels (modules 2.1-2.2, 2.4-2.6) -> drivers (2.3, 2.7-2.11)
  -> EBSD methods (2.12-2.13), each running `pytest tests/test_signals
  -k nlpar -n 0` on its slice; measurer filling every MTP pin with
  date, machine (20-core Windows 11 laptop of the spherical phases),
  recipe/command and the margin convention (`pytest.approx(measured,
  rel=0.05)` or ~2x on bands) in the ledger and in the test constants;
  gate runner (`-n 0`, `-n 4` with red tests re-run alone, coverage
  100 % of `_nlpar.py`, full suite, `SKIP=licenseheaders uvx pre-commit
  run --files <explicit list>`, the oldest-matrix recipe, the
  clean-replay grep) reporting numbers, not adjectives.
- **`nlpar-stage-review`** (<= 7 agents): fidelity reviewer (refutes
  against Brewick 2019's equations, both PyEBSDIndex kernels read line
  by line at `nlpar_cpu.py:752-936` and `opt_lambda_cpu` at `:94-183`,
  and EMsoftOO `mod_NLPAR.f90` as an equation cross-check; every
  recorded deviation either justified or reproduced, none silently in
  between); conventions/integration reviewer (numpydoc, `X | Y` hints,
  no `print`, numba flags, mask polarity, `inplace`/`lazy_output`
  parity with `average_neighbour_patterns`, licence header block,
  CHANGELOG wording, coverage, the clean-replay rule); each finding
  batch is handed to two independent skeptics who try to refute it
  with evidence; a finding dies only if BOTH refute it; the fixer
  applies the survivors, adds any named killer test the review
  demands, and appends the disposition table to this file.
- **`nlpar-stage-mutants`** (<= 4 agents): injector applying the
  stage's section 6 mutants ALONE, one at a time, in the main tree
  (no worktree by default: the editable install's `.pth` points at the
  main `src`; a worktree variant needs `PYTHONPATH` and the
  `kikuchipy.__file__` guard), sequential batches of three, each
  mutant must die by its NAMED killer (`pytest <module>::<Class>::
  <test> -n 0`), then the file is restored and the slice re-run green;
  strengthener turning every survivor into a sharper test with the
  kill verified by re-injection; reporter writing the ledger rows
  (mutant, killer, measured margin where applicable).
- **`nlpar-stage-close`** (<= 4 agents): gate runner (the full gate
  list again on the final tree, incl. `git diff --name-only` showing
  none of `spherical_indexing.ipynb`, `hybrid_indexing.ipynb`,
  `load_save_data.ipynb`, and the parked plans' blobs unchanged);
  ledger/CHANGELOG/roadmap checker (every MTP in validation.md has a
  dated value; the roadmap boxes of the stage are tickable from `git
  log`; the CHANGELOG bullet has the fork link); after Stage C only:
  spec re-submission critic reading the three spec documents against
  the shipped tree and returning dated amendments.

Gate commands (Git Bash; venv is uv-managed):

```
uv run pytest tests/test_signals -k nlpar -n 0
uv run pytest tests/test_signals -k nlpar -n 4        # red tests re-run alone
uv run --no-sync coverage run -m pytest tests/test_signals/test_util/test_nlpar.py tests/test_signals/test_ebsd_nlpar.py -n 0 -q -p no:cacheprovider
uv run --no-sync coverage report -m --include="src/kikuchipy/pattern/_nlpar.py"
uv run pytest tests -n 4                               # full suite
SKIP=licenseheaders uvx pre-commit run --files <explicit changed files, never specs/>
uv run --isolated --python 3.10 --extra tests --with "numpy==1.23.0" --with "numba==0.57" --with "orix==0.12.1" --with "pyebsdindex==0.3.9.2" --with "dask==2021.8.1" --with "scikit-image==0.21.0" pytest tests/test_signals -k nlpar -n 0 -q -p no:cacheprovider
uv run --isolated --python 3.10 --extra tests --with "numpy==1.23.0" --with "numba==0.58.1" --with "orix==0.12.1" --with "pyebsdindex==0.3.9.2" --with "dask==2021.8.1" --with "scikit-image==0.21.0" pytest tests/test_signals -k nlpar -n 0 -q -p no:cacheprovider   # averaging-oracle arms vs 0.3.9.2
git diff develop...HEAD -- src tests doc examples benchmarks conftest.py CHANGELOG.rst | grep -E "^\+" | grep -n -E "specs/|requirements\.md|plan\.md|validation\.md|tech-stack\.md| [DV][0-9]"   # must print nothing
uv run pytest --weekly tests/test_signals -k nlpar     # weekly, local
uv run sphinx-build -b html doc doc/_build/html        # Stage C
```

Amendments of 2026-10-05 (Stage A implementation gate, validation.md
ledger entries 7 and 8):

- Coverage: pytest-cov is not installed in `.venv` (coverage 7.16.0
  is), so the drafted pytest-cov command fails with "unrecognized
  arguments"; the `coverage run` / `coverage report` pair above
  replaces it (set `COVERAGE_FILE` to a scratch path to keep a
  `.coverage` file out of the tree).
- Oldest matrix: `--extra tests` added 2026-10-05: `--isolated` omits
  the tests extra. The numba 0.58.1 line is part of the oldest-matrix
  gate from now on (one recorded run per stage): PyEBSDIndex 0.3.9.2's
  `nlpar_nb` does not compile under numba 0.57, so the first line (the
  CI oldest pins) runs the sigma oracle and every non-oracle test and
  skips every `nlpar_nb` call, and the second line records the full
  averaging parity against 0.3.9.2 (Stage A: 338 passed, 398 skipped
  at numba 0.57; 736 passed, 0 skipped at numba 0.58.1).
- Known flakes beside the numba-cache flake rule (section 0.3; a red
  test under `-n 4` is re-run alone before it counts): two more,
  pre-existing and not NLPAR, never block a gate.
  `tests/test_indexing/test_ebsd_refinement.py::TestEBSDRefineOrientationPC::test_refine_orientation_projection_center_local_nlopt`
  segfaults intermittently (an access violation in the numba
  refinement objective called by nlopt on a dask thread; 1 of 5 runs
  alone, 3 of 10 on a develop snapshot); upstream
  `tests/test_simulations/test_kikuchi_pattern_simulator.py::TestCalculateMasterPattern::test_shape`
  passes only through its reruns (`flaky(reruns=5)`).

## 6. Adversarial review and mutation list

Each mutant is applied alone and must die by the NAMED test (module ::
class :: test; `test_nlpar.py` = `tests/test_signals/test_util/
test_nlpar.py`, `test_ebsd_nlpar.py` = `tests/test_signals/
test_ebsd_nlpar.py`), or is recorded reviewed-only with the reason.
Names are validation.md's (the naming authority, requirements D1.9);
a renamed test renames its killer row in both files in the same
commit, and validation.md's "Mutant killers" table quotes the same
rows. Killers marked (oracle) need pyebsdindex; every mutant below
also has a pyebsdindex-free killer unless stated. Stage: A unless
marked (B); an arm tagged [B] lands with Stage B. The [B]
`test_lazy_equals_eager_<chunking>` arms run through the method,
where `_reduce_chunks` keeps only the ROW chunking of the small lazy
fixture (requirements D8.1); the saturated block, the `(1,) * 10`
rows thinner than the depth and the `(3, 3, 4)` tiny-edge rows
survive it, so those arms still see M11, S1, S4 and S8, while column
and both-axes chunkings are killed at driver level in Stage A by the
two `TestDepthAndHalo` tests. `test_reference_
agrees_with_the_method_on_random_maps` is abbreviated `reference
agreement` after its first mention.

| # | Mutant | Named killer(s) |
|---|---|---|
| M1 | sign of the sigma correction flipped (`d2 + n2 (s0 + s1)`) | `test_ebsd_nlpar.py::TestNoiseOracle::test_normalised_distance_moments` (mean of `d` outside the ~+1.2 band); `test_ebsd_nlpar.py::TestIdentities::test_reference_agrees_with_the_method_on_random_maps`; `test_nlpar.py::TestSigmaOracle::test_sigma_parity_compiled_*` (oracle, the normalised distances) |
| M2 | sigma used unsquared in `d` | `TestNoiseOracle::test_sigma_recovery_median_ratio` (sigma_true 8, never 1); `TestIdentities` reference agreement; `test_nlpar.py::TestAveragingOracle::test_average_parity_compiled_*` (oracle) |
| M3 | `sqrt(n2)` instead of `sqrt(2 n2)` | `TestNoiseOracle::test_normalised_distance_moments` (std of `d` off by sqrt 2); reference agreement; V3 parity (oracle) |
| M4 | `d2 / n2` normalisation | `TestNoiseOracle::test_normalised_distance_moments`; reference agreement; V3 parity (oracle) |
| M5 | self weight 0 or self slot excluded | `TestIdentities::test_injected_tiny_sigma_is_an_identity` (neighbours underflow, sum would be 0); `TestIdentities::test_huge_lambda_is_the_shifted_window_box_mean` |
| M6 | `max(d - dthresh, 0)` clamp dropped | `TestIdentities::test_weight_formula_on_injected_distances` (negative `d` would give weight > 1); `TestNoiseOracle::test_fraction_of_unit_weights`; [B] `test_nlpar.py::TestPolicyOracles::test_dthresh_is_consistent_between_kernel_and_objective` |
| M7 | `exp(-d / lam)` instead of `exp(-d / lam^2)` | `TestIdentities::test_weight_formula_on_injected_distances` (run at `lam != 1`); [B] `test_nlpar.py::TestLambdaOracle::test_closed_form_lambda_on_constructed_distances` |
| M8 | search window clamped at edges instead of shifted inward | `test_nlpar.py::TestAveragingOracle::test_border_band_differs_from_clamp_and_zero_extend` (oracle); `TestIdentities::test_huge_lambda_is_the_shifted_window_box_mean`; reference agreement (the reference shifts) |
| M9 | zero-extended borders with window sums (the `average_neighbour_patterns` edge rule) | same three as M8 |
| M10 (A, lazy arm B) | depth `= r` with a tiny edge chunk | `test_nlpar.py::TestDepthAndHalo::test_depth_helper`; `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk[((3, 3, 4), (7, 7, 2))]` (Stage A, pyebsdindex-free: the 3-row edge chunk plus a 3-row halo cannot reach row 6); [B] `test_ebsd_nlpar.py::TestLazyAndContracts::test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3` |
| M11 (A, lazy arm B) | saturation maximum per chunk instead of global | `test_nlpar.py::TestPolicyOracles::test_saturation_max_is_global` ([A] wrapper arm on two hand-built haloed blocks; [B] lazy two-chunk arm); `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` and `test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel` (Stage A, saturated pixels planted in one block only); [B] `TestLazyAndContracts::test_lazy_equals_eager_<chunking>` (row chunkings through the method; the saturated block survives `_reduce_chunks`) |
| M12 | global pixel count `N` instead of the per-pair `n2` under saturation | `test_nlpar.py::TestAveragingOracle::test_saturation_arm_separates_per_pair_n2_from_global_n` (oracle); `TestIdentities::test_reference_agrees_with_the_method_on_random_maps[saturation_protect=True]` on the saturated fixture (pyebsdindex-free) |
| M13 | `floor` (truncating cast) instead of `rint` | `TestLazyAndContracts::test_dtype_round_trip` (mean shift of -0.5 grey level); reference agreement (`rint` + clip of the reference) |
| M14 | per-pattern min-max rescale of the output | `TestLazyAndContracts::test_dtype_round_trip` (mean preserved) and `test_inplace_with_dtype_out_changes_the_data_dtype`; `TestIdentities::test_constant_map_is_an_identity`; `TestIdentities::test_injected_tiny_sigma_is_an_identity` (bitwise identity) |
| M15 | signal-mask polarity inverted | `TestNoiseOracle::test_mask_polarity_separates`; `test_nlpar.py::TestSigmaOracle::test_sigma_mask_is_forwarded` (oracle) |
| M16 | mask not forwarded to the sigma pass | `test_nlpar.py::TestSigmaOracle::test_sigma_mask_is_forwarded` (oracle); `TestIdentities::test_reference_agrees_with_the_method_on_random_maps[signal_mask=circle]`; `TestSigmaMethod::test_forwards_mask_and_protection` |
| M17 (B) | phantom (out-of-map) slots counted in the lambda objective | `test_nlpar.py::TestLambdaOracle::test_phantom_free_objective_excludes_missing_neighbours` (value-level: at fixed `lam` in {1.0, 2.0} the objective on the border-masked field with the invalid slots at `d = 0.0` equals the `+inf` variant EXACTLY and differs from the phantom-counting objective by more than 1e-3, measured 0.049 / 0.042; the minimiser comparison alone cannot kill, the two minimisers differ by < 1e-4); `TestLambdaOracle::test_phantom_deviation_is_measured_and_pinned` (oracle, download) |
| M18 (B) | objective uses `max(d, dthresh)` | `test_nlpar.py::TestPolicyOracles::test_dthresh_is_consistent_between_kernel_and_objective` (at `dthresh = 0.5`; the forms coincide at 0) |
| M19 (B) | objective statistic: median over points instead of the mean (or the mean over three target fits instead of one 0.34 fit) | `test_nlpar.py::TestLambdaOracle::test_mixed_c_field_distinguishes_mean_from_median` (value-level, no optimiser: on a 70/30 field at fixed `lam = 1.5` the objective equals `np.mean` of the per-point terms EXACTLY, 0.2616, and differs from their `np.median`, 0.1068, by more than 1e-3; an optimiser-based comparison cannot kill, both minimisers from `x0 = 1.0` are the kink 1.1884); the three-fits reading dies by `TestLambdaMethod::test_target_weight_is_forwarded_through_lam_none` |
| M20 | 3x3 sigma window shifted inward instead of clipped | `test_nlpar.py::TestSigmaOracle::test_sigma_window_is_clipped_not_shifted` (oracle arm plus the test-local CLIPPED formula bitwise at every border pixel; the test-local shifted variant asserted `<=` everywhere with strict inequality on >= 25 % of the border pixels, seed 0); reference agreement on the (4, 5) map (the reference clips) |
| M21 | `fastmath=True` on any kernel | `test_nlpar.py::TestKernels::test_kernels_are_compiled_with_cache_and_nogil` |
| M22 | float64 accumulation in the distance or weighted-sum kernel | `test_nlpar.py::TestAveragingOracle::test_average_parity_compiled_*` (oracle, bitwise) -- dies only where pyebsdindex is installed (this machine; the ubuntu/windows `[all]` CI jobs); recorded as such, no pyebsdindex-free killer exists because the float64 result rounds to the same uint8 almost everywhere |

Supplementary mutants carried from the parked plan's seed list (S5
added 2026-10-04, the sigma-statistic reading of the parked "mean for
median" row; M19 keeps the objective-statistic reading; S6-S8 added
2026-10-04, spec review C2-F10, the three remaining seeds "duplicate
guard removed", "saturation threshold on the wrong constant" and
"depth off by one"):

| # | Mutant | Named killer(s) |
|---|---|---|
| S1 | halo region returned with the core (wrapper returns the whole block) | `test_nlpar.py::TestAveragingOracle::test_calclim_from_block_info` (oracle; shape or values differ); `TestDepthAndHalo::test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel` and `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` (Stage A, pyebsdindex-free: `map_blocks` with the stated `chunks=` raises or misplaces values); [B] `TestLazyAndContracts::test_lazy_equals_eager_<chunking>` (row chunkings through the method) |
| S2 | `+inf` padding missing for a map smaller than the window | `TestLazyAndContracts::test_map_smaller_than_the_window`; `test_nlpar.py::TestAveragingOracle::test_small_map_padding_slots_are_inf` |
| S3 | 1-D navigation misrouted or the per-axis radius applied in (col, row) order | `TestIdentities::test_reference_agrees_with_the_method_on_random_maps[(1, 2)]` and `[(2, 1)]` (asserted unequal to each other); `TestLazyAndContracts::test_one_dimensional_navigation_equals_a_one_row_map` (eager arm A, lazy arm [B]) |
| S4 (A, lazy arm B) | lazy path skipping the minimum-chunksize rechunk | `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk[(1, 1)]` and `[((3, 3, 4), (7, 7, 2))]` (Stage A); [B] `TestLazyAndContracts::test_issue_230_irregular_chunks`; `test_lazy_equals_eager_<chunking>[(1, 1)]` (rows `(1,) * 10` through the method, thinner than the depth 6) |
| S5 | sigma^2 from the mean or median of the eight neighbour estimates instead of the minimum | `test_nlpar.py::TestSigmaOracle::test_sigma_parity_compiled_*` (oracle, bitwise); `TestNoiseOracle::test_sigma_recovery_median_ratio` (the mean estimator gives a ratio ~1.00, outside the 0.969-0.978 band); `TestTwoGrain::test_cross_boundary_weights_are_exactly_zero` (a mean over neighbours that include cross-boundary pairs inflates sigma) |
| S6 | duplicate guard dropped (`d2 > 0` removed from the sigma minimum) | `test_nlpar.py::TestSigmaOracle::test_sigma_fallback_and_duplicates` (oracle: the duplicated pair would set `sigma = 0`); `test_nlpar.py::TestPolicyOracles::test_duplicate_neighbour_is_skipped_in_sigma_but_averaged` (pyebsdindex-free arm). NOT `TestIdentities::test_constant_map_is_an_identity`: with the guard dropped a constant map still returns itself in both protection modes (`n2 == 0` under protection; sigma 0 and the kept `dnorm <= 1e-8` branch give weight 0 without it), spec review C3-R3-F1 |
| S7 | saturation constant swapped (0.999 in the sigma kernel or 0.9961 in the averaging kernel) | `test_nlpar.py::TestPolicyOracles::test_uint16_two_threshold_arm` (asserted both ways); `TestSigmaOracle::test_sigma_saturation_threshold_constant` (oracle); `TestPolicyOracles::test_sigma_fallback_value_is_1e12` (the two module constants) |
| S8 (A, lazy arm B) | depth off by one (`r - 1` or `r + 1` with the halo logic unchanged) | `test_nlpar.py::TestDepthAndHalo::test_depth_helper`; `TestDepthAndHalo::test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk` (every chunking); [B] `TestLazyAndContracts::test_lazy_equals_eager_<chunking>` (row chunkings through the method) |

Review scope beyond mutants: the fidelity reviewer also checks the
promotion chains of modules 2.2, 2.4 and 2.5 against the oracle
source (the float32/float64 mixing decides bitwise parity), the
`-inf`/`+inf` slot conventions, and that every "Differences" item in
the module 12 Notes is a measured, recorded deviation.

## 7. Open questions

Decided autonomously (conservative default in force; the measurement
or gate that confirms it):

1. **Test file placement.** DECIDED (spec review 2026-10-04,
   F1-F1/C1-F1/E1-F3): both test modules under `tests/test_signals/`
   (`test_util/test_nlpar.py` and `test_ebsd_nlpar.py`), although the
   module lives in `pattern/`: the one root `tests/test_signals -k
   nlpar` of every gate command covers both, and the approved session
   plan's verification grep names this path. requirements D1.9 records
   it; a `tests/test_pattern/` placement is withdrawn.
2. **Pass-1 depth and normalisation.** DECIDED (spec review
   2026-10-04, F1-F2/C1-F2/E1-F2; requirements D2.5, D7.1 and D8.2
   amended): pass 1 at depth 1 returning raw `d2`/`n2`/`valid`/`sigma`,
   normalisation on the assembled whole-map arrays in NumPy (module
   2.3) reproducing the compiled oracle's promotion chain. The parked
   plan said depth 2 with the normalisation in the kernel; the
   refinement removes the depth-2 halo and lets a user `sigma` feed the
   lambda objective. Confirmed by V2 (sigma bitwise, and the
   normalised distances of module 2.3 bitwise with the oracle's `dout`
   on the neighbour slots) and V6 (objective equals the test-local
   `loptfunc` on PyEBSDIndex's own `dout`, up to the recorded phantom
   and `dthresh` deviations) at the Stage A/B gates.
3. **`search_radius` all zero.** Default: warn and return `None`,
   exactly as `average_neighbour_patterns` does for a `(1, 1)` window
   (`ebsd.py:1028-1034`), even with `inplace=False`. Recorded quirk,
   pinned by `TestIdentities::test_search_radius_zero_warns_and_is_a_
   no_op`.
4. **Default-suite oracle route.** DECIDED (spec review 2026-10-04,
   E1-F1, measurement re-run by the fixer): the compiled
   `NLPAR.sigma_numba` / `NLPAR.nlpar_nb` dispatchers on EVERY parity
   arm, small synthetic maps and `nickel_ebsd_large` alike, behind the
   module-scoped `pyebsdindex_kernels` warm-up fixture; the oracle's
   `.py_func` is never used (not bitwise with the compiled kernel:
   `dout` and `nlpar_nb` differ, drafting measurements above), so the
   earlier `.py_func` sizing rule (~1e6 iterations per call) is
   withdrawn. Default-suite oracle inputs are sized by the warm-up time
   the fixture records (cold compile ~7 s) against the `<= ~90 s`
   per-worker budget; the measurer records both at the Stage A gate.
   Only the file-based end-to-end oracle is weekly.
5. **`ensure_minimum_chunksize` on the oldest CI dask (2021.8.1).**
   DECIDED (spec review 2026-10-04, C3-R3-F4): use
   `dask.array.overlap.ensure_minimum_chunksize` (present in dask
   2026.3.0, measured 2026-10-04), applied explicitly before
   `overlap`. Verified 2026-10-04 against the `2021.08.1` tag of
   `dask/array/overlap.py`: the function exists there with a body
   identical to 2026.3.0's (the `for c in chunks` loop, the `if new >
   size + (size - c)` branch, the `output[-1] += new` tail; tracing
   `(26, 26, 3)` at size 4 gives `(26, 25, 4)` in both), and
   `overlap(x, depth, boundary)` is defined there too. The
   oldest-matrix recipe pins `dask==2021.8.1` (section 5, requirements
   D10.5) as the executable confirmation of the import and of the
   `overlap`/`map_blocks` semantics at the Stage A gate, and the oldest
   CI job checks them again. The private four-line fallback is
   withdrawn.
6. **`si_wafer` in the tutorial.** Default: shown as code with
   ledger-quoted numbers, not executed (311 MB download, ~5-10 s
   compute; RTD budget). The parked plan's "si_wafer lazily" is read as
   "demonstrate the lazy call", which the non-executed cell does.
7. **`dtype_out` values.** DECIDED (spec review 2026-10-04, E2-F16):
   `None` = input dtype; otherwise an integer or floating NumPy dtype
   only (`np.issubdtype(dt, np.integer) or np.issubdtype(dt,
   np.floating)`), integer dtypes through `rint` + clip to that dtype's
   range, float dtypes by cast; bool, complex and object raise
   `ValueError("dtype_out must be an integer or floating dtype")`
   (requirements D6.2, D1.6). Confirmed by
   `TestLazyAndContracts::test_dtype_round_trip`,
   `test_inplace_with_dtype_out_changes_the_data_dtype` and the
   `dtype_out=bool` arm of `test_argument_validation`.
8. **Single `target_weight`.** Default: one target (0.34), no
   median-of-three. Confirmed by
   `TestLambdaOracle::test_lambdas_ascend_with_decreasing_target`
   (monotone, so the median of three fits is the middle fit) and
   `TestLambdaMethod::test_target_weight_is_forwarded_through_lam_none`.
9. **`block_info` usage.** DECIDED (requirements D8.5 amended
   2026-10-04): halo sides from `chunk-location` and `num-chunks` only
   (verified 2026-10-04 that `array-location` is in overlapped
   coordinates under `map_overlap` and under `map_blocks` on an
   overlapped array, and that `chunk-shape`/`dtype` are absent under
   `map_overlap`); the wrappers return the core only and assert
   `block_info is not None` (explicit `meta=`). Confirmed by
   `TestAveragingOracle::test_calclim_from_block_info`.

FOR THE USER (few by design; the 2026-10-04 decisions are not reopened):

10. **Approval scope.** Does the approval of this `plan.md` also cover
    the section 0 constitution texts as written (the main loop applies
    them in commit 1 without a second approval round)? The autonomous-
    mode memory waives the gate for spherical phases only; NLPAR is a
    new feature path, so the default here is: one approval, this file,
    covers section 0.
11. **Fan-out timing.** After the merge on your go, do steps 1 and 2 of
    section 1 proceed immediately in the same session, or each on a
    separate go? Default: immediately, in order, with the gates
    recorded; stop and report at the first red gate.

## 8. Commits

Signed (`git commit -s`), explicit pathspecs, the attribution trailer
the session specifies at commit time (currently `Co-Authored-By: Claude
Fable 5.1 <noreply@anthropic.com>`; amended 2026-10-05: the session
now specifies `Co-Authored-By: Claude Opus 5.5
<noreply@anthropic.com>`, and the trailer always follows the session),
no em-dashes in messages, LF files,
no BOM (`roadmap.md` line 1 checked after every spec-touching commit).

1. **"Add NLPAR spec and constitution amendments"** --
   `specs/2026-10-04-nlpar/{requirements,plan,validation}.md`, the
   section 0 appends to `specs/{mission,roadmap,tech-stack}.md`, the
   status header of `specs/_research/plan-nlpar-2026-09-11.md`. After
   the user approves this plan.
2. **"Add NLPAR Stage A skeletons and failing tests"** --
   `src/kikuchipy/pattern/_nlpar.py` stubs, `ebsd.py` method stubs, the
   two test modules, the generator functions and fixtures (with the
   shared `EXP_KERNEL_ULP` placeholder) appended to
   the root `conftest.py`, `validation.md` ledger entry. Never pushed
   alone.
3. **"Implement NLPAR Stage A: kernels, sigma estimation and
   EBSD.average_non_local_neighbour_patterns"** -- implementation,
   measured pins, review fixes, `CHANGELOG.rst` "Added" bullet with the
   fork PR link (placeholder #17 until confirmed), roadmap Stage A
   ticks, ledger entries, section 10/review disposition rows in this
   file. Push (2 + 3).
4. **"Add NLPAR Stage B failing tests: lambda optimisation and the lazy
   path"** -- never pushed alone.
5. **"Implement NLPAR Stage B: lambda optimisation and the lazy path"**
   -- `get_nlpar_lambda`, `lam=None`, lazy support, pins, performance
   baselines, CHANGELOG bullet extended, roadmap Stage B ticks. Push
   (4 + 5).
6. **"Add NLPAR tutorial notebook"** -- `doc/tutorials/nlpar.ipynb`,
   `index.rst`, `run_nbval.sh`, `tutorials_sanitize.cfg` if touched,
   `examples/pattern_processing/nlpar.py`, CHANGELOG tutorial bullet,
   roadmap Stage C ticks, spec re-review amendments. Push. Then the
   PR is opened.
7. **"Tick NLPAR boxes in roadmap (jwestraadt/kikuchipy#17)"** -- on
   `feat-NLPAR` after the PR number is confirmed (and the CHANGELOG
   links rewritten if it is not #17); ticks "PR opened". The remaining
   fan-out boxes are ticked by one fork-local commit on `develop` after
   the fan-out ("Tick NLPAR fan-out boxes in roadmap"), the
   `0aa3e406`/`18d59c07` precedent for spec-only commits on `develop`.

Between workflows the tree is checked clean apart from the untracked
`AGH__Si_indent_1_512x672.h5oina` and
`specs/_research/plan-upstream-merge-0.13.1.md`.

## 9. Definition of done

- Per stage: failing tests committed first (Stages A, B); implementation;
  adversarial review (two reviewers + two skeptics) and bug injection
  (every section 6 mutant of the stage dead by its named killer or
  recorded reviewed-only) + fixes; `SKIP=licenseheaders` pre-commit
  clean on the explicit file list; coverage 100 % of `_nlpar.py` with the
  command output recorded; `-n 0` then `-n 4` green (red tests re-run
  alone, flakes attributed); full suite green (`uv run pytest tests -n
  4`, recorded); oldest-matrix recipe recorded (both numba lines of
  section 5, amended 2026-10-05); clean-replay grep empty;
  never-sweep check clean; signed commits pushed together; roadmap
  stage boxes ticked.
- Every MTP placeholder in `validation.md` replaced by a dated measured
  value (recipe + machine); every seed quoted from 2026-09-11 either
  confirmed or amended with the 2026-10-04-or-later measurement and the
  date; refutations of frozen decisions recorded in the ledger AND
  amended in `requirements.md` with the same date.
- `EBSD.average_non_local_neighbour_patterns`, `EBSD.get_nlpar_sigma`,
  `EBSD.get_nlpar_lambda` documented (numpydoc, `:cite:`, Notes with the
  differences list), in the API reference; `git grep -n -E "Optional\[|
  Union\[|print\(" -- src/kikuchipy/pattern/_nlpar.py tests/test_signals/
  test_ebsd_nlpar.py tests/test_signals/test_util/test_nlpar.py` finds
  nothing; `src/` never imports `pyebsdindex` for NLPAR (grep recorded).
- Tutorial registered and nbval-wired; `sphinx-build -b html` exit 0;
  gallery example renders; CHANGELOG bullets (feature, tutorial) with the
  confirmed fork PR link.
- Open questions 1-9 confirmed by their named measurements with dated
  records; 10-11 answered by the user.
- The three spec documents re-submitted to review after Stage C with the
  findings folded (disposition rows appended below).
- `git log origin/feat-NLPAR..feat-NLPAR` empty; PR `feat-NLPAR ->
  develop` open with ubuntu/windows CI green (macOS red attributed to the
  two known causes). Merge and fan-out are tracked in the roadmap, not in
  this definition.
- Memory notes updated after the fan-out (`nlpar-parked-plan.md`,
  `feat-spherical-indexing-branch.md`, `hrebsd-dic-project.md`,
  `hrosm-parked-plan.md`), per the session plan.

## 10. Spec-review disposition table (2026-10-04, fixer, round 1)

Three read-only critics (F1, C1, E1) reviewed the three drafts; 67
findings (5 blocker, 32 major, 30 minor). Rules applied: every
blocker and major applied unless it contradicted a user decision or
was factually wrong; minors applied when cheap; validation.md is the
naming authority for test classes and names; the three documents were
re-read for consistency after the edits. Where two critics
recommended opposite resolutions of the same conflict, the row names
the resolution and the reason. Measurements the fixer re-ran are in
the drafting-measurements paragraph above, requirements D13.6 and
validation.md ledger entry 2.

| id | severity | file | disposition (applied / declined: reason) | what changed |
|---|---|---|---|---|
| F1-F1 | blocker | requirements.md | applied | Test path unified to `tests/test_signals/test_util/test_nlpar.py` (session plan grep, parked plan, two of three critics); D1.9 rewritten, D10.5 recipe root `tests/test_signals -k nlpar`; "plan.md updates the grep" sentence deleted |
| F1-F2 | blocker | requirements.md | applied (plan's refinement adopted) | D2.5, D7.1, D8.2 amended 2026-10-04: kernel returns raw `d2`/`n2`/`valid`/`sigma`, `_nlpar_normalized_distances` (NumPy) reproduces the compiled oracle's chain, pass 1 depth 1; validation V2 compares the driver output, V7 header "pass 1 depth 1" |
| F1-F3 | major | plan.md | applied | validation.md names canonical; plan 2.14, 3.3 and both section 6 tables rewritten with `Class::test` names from validation.md; missing killers added to validation (`test_small_map_padding_slots_are_inf`, reference-agreement arms, `test_window_bounds_*`, S1-S5 rows) |
| F1-F4 | major | plan.md | applied | M19 = objective statistic (median over points) Stage B, killer V6 mixed-c; new S5 = sigma minimum replaced by mean/median, Stage A, killers V2 parity + V4 recovery band + V5 cross-boundary; both tables in both files |
| F1-F5 | major | plan.md | applied, with E1-F1's route | One oracle route everywhere: COMPILED dispatchers on every arm (small maps included) behind the warm-up fixture, Ni arms default [download]; the `.py_func` small-map route recommended here is withdrawn because E1-F1 measured it is not bitwise; D10.2, plan 0.3/2.14/7.4, validation Automated/V2/V3 agree |
| F1-F6 | major | plan.md | applied | One recipe string in D10.5, plan 0.3 and section 5, validation Local-gated: adds `--with "dask==2021.8.1" --with "scikit-image==0.21.0"` (C1-F5's addition); 7.5 rewritten |
| F1-F7 | major | validation.md | applied | `(47, 8)` at r=4 -> 4 (rule re-computed by the fixer); added `(26, 26, 3)` at r=4 -> 6 with `ensure_minimum_chunksize(6, ...) == (26, 23, 6)` (measured) |
| F1-F8 | major | requirements.md | applied | D8.5 amended: `chunk-location` vs `num-chunks`; `array-location` not used; core-only wrappers |
| F1-F9 | major | validation.md | applied | V2 compares the eight neighbour slots only (`ours[valid]` in slot order vs compact `dout`, `nout >= 1`), self slot excluded; self-slot convention and PyEBSDIndex's `-sqrt(npix / 2)` recorded in D2.5 and plan 2.2 |
| F1-F10 | major | plan.md | applied (option b) | Lambda objective + optimiser, `TestLambdaOracle`/`TestLambdaMethod` (V6), V8, V9, `TestPerformance` are Stage B in plan 2.14/3.x and validation headers; V10 `test_dthresh_is_consistent_*` tagged [B]; V10 `test_saturation_max_is_global` split into an [A] wrapper arm and a [B] lazy arm; M11 killers name both arms; D5.7 records the split |
| F1-F11 | minor | requirements.md | applied | Scope Stage A and D5.7 name the three Stage A guards and `get_nlpar_sigma`'s lazy guard; check order stated |
| F1-F12 | minor | requirements.md | applied | D12.2: 9 regexes (`regex1`-`regex9`), counted |
| F1-F13 | minor | validation.md | applied | "plan.md section 6 list" |
| F1-F14 | minor | plan.md | applied | Pass-1 wrapper returns `(core_rows, core_cols, 4, 9)`; unpack rule stated |
| F1-F15 | minor | requirements.md | applied | D4.2 uses the four-argument `_window_bounds(center, radius, n, shift)`; min/max form kept as the stated equivalent |
| F1-F16 | minor | validation.md | applied | ledger item 4 `tests.yml:43`; header `:22-49` |
| F1-F17 | minor | validation.md | applied (as C1-F22's broader test) | V0 bullet `test_nlpar_code_never_names_pyebsdindex` (module source plus the three methods' source); D10 row cites it |
| F1-F18 | minor | requirements.md | applied | D7.4: MTP, seed 1 float32 ulp, pinned at the Stage A gate |
| C1-F1 | blocker | requirements.md | applied | as F1-F1 |
| C1-F2 | blocker | plan.md | applied (Option B) | as F1-F2; kernel return tuple stated once in D2.5/D7.1 and cited from V0, V2 and plan 2.2 |
| C1-F3 | major | requirements.md | applied | D8.5 amended as proposed (minus the zero-returning probe branch, removed by explicit `meta=`, E1-F22); V3 `test_calclim_from_block_info` carries the sentence |
| C1-F4 | major | requirements.md | applied | D5.7 extended to the four guards and the stage split; V7 bullet `test_stage_a_guards_raise_not_implemented` [A, deleted in B] |
| C1-F5 | major | plan.md | applied | recipe with dask and scikit-image pins everywhere; 7.5 rewritten; validation unconditional |
| C1-F6 | major | plan.md | applied | section 6 rewritten verbatim from validation.md; S1-S5 rows added to validation's mutant table; "section 6" |
| C1-F7 | major | validation.md | applied | M19/S5 as proposed, M19 bullet states the closed-form behaviour; M19 added to plan 3.6 |
| C1-F8 | major | validation.md | applied in part; "compiled weekly only" declined | One route stated identically in all three documents, but the route is the compiled dispatchers in the DEFAULT suite: E1-F1 measured that the oracle's `.py_func` is not a bitwise oracle, so a `.py_func` default suite would pin nothing; the ~7 s compile fits the budget and the cache side effect is covered by the `-n 0`-first flake rule (now stated in plan 0.3) |
| C1-F9 | major | validation.md | applied | V1 reference agreement parametrised over `dthresh` {0, 0.5}, `signal_mask` {None, circle}, `saturation_protect` {True, False} with "non-default arm differs" assertions; V3 `dthresh=0.5` arm; V6 `test_target_weight_is_forwarded_through_lam_none`; named as M12/M16 killers |
| C1-F10 | major | validation.md | applied | V1 arms `(1, 2)`/`(2, 1)` asserted unequal; V7 1-D test pins `get_nlpar_sigma()` shape `(n,)`; S3 killer |
| C1-F11 | major | validation.md | applied | V7 `test_argument_validation` with the `(kwargs, match)` list; message fragments frozen in D1.6; D1/D2/D9 mapping rows |
| C1-F12 | major | plan.md | applied | plan 2.12 dtype branch (`store` only when the dtype is unchanged); V7 `test_inplace_with_dtype_out_changes_the_data_dtype`; M14 killer |
| C1-F13 | major | validation.md | applied | constant-map bullet: uint8 bitwise for `dtype_out=None`, float32 within 2 ulp (structural), exact identity pinned by the tiny-sigma test |
| C1-F14 | minor | validation.md | applied | executable `git diff develop...HEAD ... \| grep` form in validation and D11.5 |
| C1-F15 | minor | requirements.md | applied | as F1-F12 |
| C1-F16 | minor | requirements.md | applied | as F1-F15 |
| C1-F17 | minor | plan.md | applied | plan 2.12 points at the numbered D12.6 list (now twelve items, E1-F8) |
| C1-F18 | minor | validation.md | applied | DoD names the four memory notes |
| C1-F19 | minor | plan.md | applied | D5.4 rule (`lam <= 1.01e-3` or `>= 9.9`) and one message in all three; V6 matches "upper bound"/"lower bound" (E1-F24's wording) rather than the numeric strings |
| C1-F20 | minor | validation.md | applied | V6 `test_stride_above_1e6_points` (Stage B); D5.1 cites it |
| C1-F21 | minor | validation.md | applied | V6 `test_lam_none_logs_the_optimised_lambda` (Stage B, no download) |
| C1-F22 | minor | validation.md | applied | V0 `test_nlpar_code_never_names_pyebsdindex`; D10 row |
| C1-F23 | minor | validation.md | applied | V1 tiny-sigma rationale names the `d = 1e6 * n2` branch; V10 `test_tiny_dnorm_branch_gives_1e6_n2` |
| E1-F1 | blocker | validation.md | applied (probe re-run by the fixer, confirmed) | compiled dispatchers on every parity arm; `_pyfunc` route and tests deleted, `_compiled_` names; plan 0.3, 2.14, 7.4, D10.2, D7.5 rewritten; ledger entry 2 row; our kernel chain (float32 `d2`, float64 normalisation intermediates, float64 exponent) in D2.5/D7.3/plan 2.3/2.5 |
| E1-F2 | major | requirements.md | applied | dated amendment adopting the plan, as F1-F2 |
| E1-F3 | major | plan.md | applied in substance; recommended path declined | one path everywhere, but `tests/test_signals/test_util/test_nlpar.py`, not `tests/test_pattern/`: the approved session plan's verification grep and the parked plan name it, F1/C1 recommend it, and one root covers both modules |
| E1-F4 | major | plan.md | applied | as F1-F3; plan-only tests added under V0/V1/V3; the names statement opens section 6 |
| E1-F5 | major | requirements.md | applied | D4.2 signature and formulas; V0 `test_window_bounds_matches_pyebsdindex_expressions_and_clamps` |
| E1-F6 | major | plan.md | applied, route adjusted | `(4, 9)` packing with explicit `chunks=`/`meta=`; since `map_overlap`'s `chunks=` would have to be in overlapped coordinates, both passes use `da.overlap.overlap` + `da.map_blocks` with core-only wrappers (measured working 2026-10-04); D8.2/D8.3/D8.5 amended, plan 2.9-2.11 rewritten |
| E1-F7 | major | validation.md | applied | "Oracle call recipe" block in the Automated section (layouts, index arrays, fresh `calclim`, axis `>= 2 sr + 1`), verified against `nlpar_cpu.py:754, :822, :487-502` |
| E1-F8 | major | plan.md | applied | plan 2.5 operand types; D7.3 `lam` typing; D12.6 item 12 (PyEBSDIndex's driver rounds `lam` to float32) |
| E1-F9 | major | requirements.md | applied | D7.3 (a)-(c): no `**` on float32, `np.float32(1.0)` counters, explicit float64 casts; V0 `ast.Pow` walk; plan 2.2/2.4 spell the products |
| E1-F10 | major | validation.md | applied | V2 assertion rewritten (compact order, self slot excluded, `nout >= 1`); slot mapping in plan 2.2 and D2.5 |
| E1-F11 | major | validation.md | applied | V5 `lam` in {0.7, 2.5} Stage A, `lam=None` arm Stage B; V10 saturation test split; every V7 bullet tagged [A]/[B]; check order in plan 2.12 and D1.6 |
| E1-F12 | major | validation.md | applied | as C1-F11, with E1's `(kwargs, match)` list and messages |
| E1-F13 | major | plan.md | applied | as C1-F12 |
| E1-F14 | major | validation.md | applied | as F1-F7 |
| E1-F15 | major | validation.md | applied | "Fixture generators" block in the Automated section (signatures, seeds, ramp base, uint8 clip); oracle fixtures `(7, 8 \| 12, 12)`; V4 `(12, 12 \| 32, 32)` float32; V5 `two_grain(nav_shape=(10, 16), ...)` |
| E1-F16 | major | validation.md | applied | closed form on a constructed `(10, 10, 9)` field with `valid` all True; `test_phantom_free_objective_excludes_missing_neighbours` added as the pyebsdindex-free M17 killer |
| E1-F17 | minor | requirements.md | applied | as C1-F14 |
| E1-F18 | minor | requirements.md | applied | D5.1 `S_i = 1 + sum`, `1e-12` guard not reproduced; V6 transcription drops it |
| E1-F19 | minor | plan.md | applied | plan 2.4 keeps the `1e6 * n2` branch, `+inf` only for `n2 == 0` |
| E1-F20 | minor | plan.md | applied | check order and both guard messages in plan 2.12 and D1.6/D5.7; V7 names the messages |
| E1-F21 | minor | plan.md | applied (path per F1-F1; `-n 0` not added) | one recipe string; `-n 0` left out of the isolated recipe because xdist's presence in `uv run --isolated` is unverified and the recipe never had it |
| E1-F22 | minor | plan.md | applied | wrappers assert `block_info is not None`; explicit `meta=`/`chunks=` on both passes |
| E1-F23 | minor | requirements.md | applied | D1.6: `np.asarray(signal_mask, dtype=bool)` after the shape check |
| E1-F24 | minor | plan.md | applied | as C1-F19 |
| E1-F25 | minor | validation.md | applied | `TestSigmaMethod` tests under V7 (shape, pass-1 equality, mask/protection forwarding) |
| E1-F26 | minor | validation.md | applied | `nlpar_reference` window rule for `n < 2r + 1` stated |

## 10. Spec-review disposition table (2026-10-04, fixer, round 2)

Three read-only critics (F2, C2, E2) reviewed the round-1 documents;
47 findings (1 blocker, 20 major, 26 minor). Rules as in round 1.
Measurements the fixer ran (read-only python,
`scratchpad/fixer2_probe.py`) are in the round-2 measurements
paragraph above, requirements D13.7 and validation.md ledger entry 3.
Two critic proposals were refuted by those measurements and replaced
(F2-F1's upper-bound arm, C2-F2's optimiser-based mean/median arm);
the rows say so.

| id | severity | file | disposition (applied / declined: reason) | what changed |
|---|---|---|---|---|
| F2-F1 | major | validation.md | applied, upper arm redesigned | `test_bound_hit_warns`: lower arm `c = 1e-7` (measured: returns 1e-3); upper arm NOT the proposed 4 x `c = 1` + 4 x `c = 250` mix (measured minimiser 1.176, inside the bounds) but one slot at `c = 1` and seven at `+inf` (self weight `>= 0.5 > 0.34` for every lambda; measured 10.0); D5.4 sentence replaced with the flat-objective explanation and the measured `x = 1.000000` |
| F2-F2 | major | validation.md | applied (measured: data (13, 10), axes (11, 15), xmap (49, 64)) | `extract_grid((11, 15))` replaced by `s.inav[::5, ::5]` in V8, plan 0.2/3.4/4.1.5, requirements Scope/D12.1; ledger entry 1 item 5 corrected in place with a dated note; ledger entry 3 records the quirk and the `inav` facts (`xmap.size` 165, orix extent shape `(51, 71)`, points compared by order) |
| F2-F3 | major | validation.md | applied in substance via E2-F1; the helper module and `tests/test_signals/conftest.py` declined | One sharing mechanism stated in requirements D1.9, plan 0.3/2.14/5/8 and validation Automated: generators as fixtures in the ROOT `conftest.py` (needs no import form at all); `pyebsdindex_kernels` stays module-scoped in `test_nlpar.py` because `test_ebsd_nlpar.py` never calls the oracle kernels (V8's Hough arms use kikuchipy's `hough_indexing`), so no second conftest is needed; the root conftest is collected by every `tests/test_signals -k nlpar` run including the oldest-matrix recipe |
| F2-F4 | minor | requirements.md | applied | Scope and D13.4 write `(55, 75 \| 60, 60)` rows x cols with the hyperspy repr in parentheses |
| F2-F5 | minor | plan.md | applied | roadmap 0.2 Stage B bullet: "two `overlap` + `map_blocks` passes (core-only chunk wrappers)" |
| F2-F6 | minor | requirements.md | applied | "shape unverified today" deleted; cites ledger entry 1 item 5 |
| F2-F7 | minor | validation.md | applied (verified with awk) | ledger entry 1 item 3: `(100)`, `(104)`, `(120)` |
| F2-F8 | minor | requirements.md | applied (verified with sed) | D10.3 `gnomonic_correction.py:52-54` |
| F2-F9 | minor | validation.md | applied (with E2-F3/C2-F13) | `fl(1/n_window)`; constant-map and box-mean tests run at `search_radius=1`, `(1, 2)` and `2` |
| F2-F10 | minor | plan.md | applied | tech-stack bullet: "identical per the 2026-09-11 reading, re-checked by the oldest-matrix run at `pyebsdindex==0.3.9.2`" |
| F2-F11 | minor | requirements.md | applied | D11.4 "plan.md section 1, fan-out steps 1-2; the session plan's Steps 2-3" |
| C2-F1 | major | validation.md | applied | Ni V2/V3 arms tagged [A, download] in validation (headers, DoD), requirements Scope, roadmap 0.2 and plan 3.4; two [A] pyebsdindex-free driver tests added under `TestDepthAndHalo` (`test_pass_one_driver_on_multichunk_dask_array_equals_the_kernel`, `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk`, parametrised over the V7 chunkings on `random_uniform_saturated((10, 16), (16, 16), one_block_only=True)`); M10, M11, S1, S4 rows name them as Stage A killers (M10/M11/S4 relabelled "A, lazy arm B"); plan 3.6 Stage B mutant list adjusted |
| C2-F2 | major | validation.md | applied in substance via E2-F7; the optimiser-based 60/40 design declined | Measured: the mean-objective minimiser from `x0 = 1.0` on an unequal split is the kink 1.1884, not "strictly between", so an optimiser comparison cannot kill M19; the test is value-level at fixed `lam = 1.5` on a 70/30 field (mean 0.2616 vs median 0.1068); M19 rows rewritten in both files |
| C2-F3 | major | validation.md | applied (verified) | as F2-F1's lower arm; `np.float32(1e-7)`, closed form 2.66e-4, threshold `c < 1.42e-6` recorded |
| C2-F4 | major | validation.md | applied | `test_sigma_window_is_clipped_not_shifted`: (a) bitwise vs the test-local clipped formula at every border pixel, (b) shifted `<=` clipped everywhere, (c) strict inequality on `>= 25 %` of border pixels at seed 0; M20 row updated |
| C2-F5 | major | requirements.md | applied (verified `:785` vs `:903`) | D2.1: ours `n2` starts at `np.float32(0.0)`, recorded deviation with the rounding argument; plan 2.2 and 2.4 state the start value; quirk catalogue moves the `1e-12` seed to NOT ported |
| C2-F6 | major | validation.md | applied in part | `REFERENCE_MAX_ABS_GREY` (MTP, seed 1e-3 grey levels) replaces the 1e-4 bound in the reference-agreement test, added to the MTP inventory and the V1 table; the `rint` flip rule rewritten so it depends on that pin; constant-map and box-mean bounds become the structural linear bound of E2-F2 (`n_window` ulp, derivable and probe-confirmed) rather than new MTP names, since D13.1 exempts structural bounds stated as such |
| C2-F7 | minor | validation.md | applied via E2-F11's `block=True` variant; `corner_block` declined | one generator flag suffices: `exact_duplicates(..., block=True)` copies (3, 4) into its eight neighbours |
| C2-F8 | minor | validation.md | applied | `test_mask_polarity_separates` runs the bitwise arm with `saturation_protect=False`; `identical_plus_gaussian(..., sigma_right=None)` added to the generator block |
| C2-F9 | minor | validation.md | applied (with E2-F12) | One threshold statement in D3.5, plan 2.2/2.4, V2 and V10: sigma kernel float64 product with `SIGMA_SATURATION_FACTOR = 0.9961`, averaging kernel float32 product with `AVERAGE_SATURATION_FACTOR = np.float32(0.999)`; V10 `test_sigma_fallback_value_is_1e12` pins both constants |
| C2-F10 | minor | plan.md | applied | S6 (duplicate guard dropped), S7 (saturation constant swapped), S8 (depth off by one) with named killers in both tables |
| C2-F11 | minor | plan.md | applied (verified lines 3-10, `## Context` at 12) | 0.4 heading names the replaced lines |
| C2-F12 | minor | validation.md | applied | stride lives in `_nlpar_optimize_lambda` (D5.1, plan 3.1); `test_stride_above_1e6_points` asserts `_nlpar_optimize_lambda(d) == _nlpar_optimize_lambda(d[::2, ::2])` on (1000, 1001, 9) and `!=` on a (999, 1000, 9) two-valued field with different group ratios on the two grids |
| C2-F13 | minor | validation.md | applied | V1 names `random_uniform_saturated((4, 5), (6, 6), frac=0.0, seed=1)` (frac 0.0 supported); huge-lambda test at `search_radius` 1, `(1, 2)` and 2 |
| C2-F14 | minor | requirements.md | applied | D1.5 (b) records the view-aliasing question and the fallback; V7 [A, download] `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal`; plan 2.12/2.14 |
| C2-F15 | minor | requirements.md | applied | D1.5 (a): lazy input + `lazy_output=False` + `inplace=True` computes and assigns; [B] arm of `test_inplace_lazy_output_contract`; plan 2.12/3.3 |
| C2-F16 | minor | requirements.md | applied | D10.4: the skeleton agent copies the GPL header from `chunk.py:1-16`; plan section 5 skeleton line |
| E2-F1 | blocker | validation.md | applied (verified: `--import-mode=importlib`, no `__init__.py`/`conftest.py` under `tests/`) | Fixture generators as root-`conftest.py` fixtures returning callables; stated in validation Automated, requirements D1.9 and Scope, plan 0.3 (Fixtures bullet), 2.14, 5 and commit 2 |
| E2-F2 | major | validation.md | applied (probe re-run: 2/3/11 ulp at 9/20/49) | constant map: `abs(out - p) <= n_window * np.spacing(np.float32(p))` per pixel, both float32 arms; table row updated |
| E2-F3 | major | validation.md | applied | `search_radius=1` (9 weights, `fl(1/9)`) for the constant-map and box-mean tests plus `(1, 2)` and 2 for the box mean; M8/M9 rows unchanged in name |
| E2-F4 | major | validation.md | applied | `nlpar_reference` header states the per-pair `n_ij`, the two thresholds, the `d2 > 0` guard, the 1e12 fallback and `d = +inf` for `n_ij == 0` |
| E2-F5 | major | validation.md | applied | `test_nlpar_code_never_names_pyebsdindex` is an `ast` import audit plus a docstring-stripped token check on the three method bodies; the module notice and Notes may name PyEBSDIndex |
| E2-F6 | major | validation.md | applied (measured 0.049 / 0.042) | value-level phantom-free assertion at fixed `lam` in {1.0, 2.0}; minimiser-equals-closed-form arm kept; M17 rows updated |
| E2-F7 | major | validation.md | applied (measured 0.2616 vs 0.1068) | value-level mean-vs-median test on a 70/30 field at `lam = 1.5`, no optimiser; M19 rows updated |
| E2-F8 | major | requirements.md | applied | D5.1 and plan 3.1: `_nlpar_lambda_objective(lam, d, valid, dthresh, target_weight)`, `lam` the float64 `(1,)` array, `w.sum(axis=-1)` over nine slots, `float(np.mean(...))`; V6 loptfunc test states the same sum order |
| E2-F9 | major | requirements.md | applied | D1.6 array `sigma` finite and `> 0` element-wise; V7 `test_argument_validation` adds `sigma=np.zeros(nav_shape, np.float32)`; plan 2.3 parenthesis |
| E2-F10 | major | validation.md | applied (arithmetic checked: blocks [0, 5), [1, 8), [4, 9); `rstart` 0, 2, 2) | `test_calclim_from_block_info` fixture `identical_plus_gaussian((9, 8), (12, 12))`, sr 2, row chunks (3, 3, 3), `saturation_protect=False` on both sides, hand-built blocks and `block_info` dicts, `calclim = [0, rstart, 8, 3]`, kept regions concatenated equal the whole-map run |
| E2-F11 | major | validation.md | applied | `exact_duplicates(nav_shape, sig_shape, seed=3, block=False)`, no saturated pixels, `block=True` copies (3, 4) into its eight neighbours; the 1e12 arm runs on `block=True`, the pair arm on the default |
| E2-F12 | minor | plan.md | applied (with C2-F9) | plan 2.2 threshold `np.float64(max_value) * np.float64(0.9961)` / `+ np.float64(1.0)`; averaging kernel float32 |
| E2-F13 | minor | plan.md | applied | plan 2.4: the driver passes `sigma2 = (sigma * sigma).astype(np.float32)`, the kernel never squares |
| E2-F14 | minor | plan.md | applied | plan 2.4 slot `= wr * (2 rc + 1) + wc`; padding rule for `wr >= len_r` or `wc >= len_c`; V3 `test_small_map_padding_slots_are_inf` names the slots |
| E2-F15 | minor | requirements.md | applied | D1.6 radius validity test, `numbers.Real` for scalar `sigma`, 0-D check first; plan 2.7 |
| E2-F16 | minor | requirements.md | applied | D6.2 and D1.6 `dtype_out` validation; plan 2.7 and 7.7; V7 `dtype_out=bool` arm |
| E2-F17 | minor | plan.md | applied | D5.6 and plan 3.1: the INFO record is emitted inside `_nlpar_optimize_lambda` |
| E2-F18 | minor | validation.md | applied | V5 comparison arm is a test-local `scipy.ndimage.correlate` with the normalised (5, 5) Gaussian, no rescale; `average_neighbour_patterns` stays in the tutorial |
| E2-F19 | minor | plan.md | applied | plan 2.8-2.10: `_nlpar_depth -> (dict[int, int] keyed by array axis, full chunks)`; wrappers take `depth: tuple[int, int]` |
| E2-F20 | minor | validation.md | applied | V10 tiny-dnorm oracle arm is a test-local NumPy transcription of `nlpar_cpu.py:901-916` asserted bitwise; no compiled call |

## 10. Spec-review disposition table (2026-10-04, fixer, round 3)

Three read-only critics (F3, C3, E3) reviewed the round-2 documents;
17 findings (0 blocker, 4 major, 13 minor). Rules as in rounds 1-2.
Every finding was verified before application: the fixer re-ran
critic C3's two probes and its own `scratchpad/fixer3_probe.py`
(read-only python; round-3 measurements paragraph above, requirements
D13.8, validation.md ledger entry 4), re-fetched `dask/array/
overlap.py` at the `2021.08.1` tag, and checked the cited lines with
`sed` and `git diff`. All 17 are applied; none contradicted a user
decision or was factually wrong. Where a finding offered two
resolutions the row names the one taken.

| id | severity | file | disposition (applied / declined: reason) | what changed |
|---|---|---|---|---|
| F3-R3-1 | major | validation.md | applied (`rechunk=True` kept, fact recorded; measured) | D8.1 records `_reduce_chunks` (`_dask.py:142-151, 163-195`) with six measured examples and the `rechunk(old_chunks)` restore; D8.6, roadmap 0.2 Stage B, plan 3.3 and the three V7 [B] bullets say the method-level arms pin ROW chunkings (Ni processed as `((26, 26, 3), (40, 35))`, M10 killer stands) and point column/both-axes coverage at the `TestDepthAndHalo` driver tests; the M11/S1/S4/S8 [B] rows in plan 6 and the validation mutant table carry "(row chunkings through the method)" and a note above plan 6 explains why those arms still kill; `rechunk=False` recorded as the rejected alternative |
| F3-R3-2 | minor | validation.md | applied (arithmetic checked: `8 exp(-200 / lam^2) < 2^-53` iff `lam < 2.27`) | V6 `test_bound_hit_warns`: "underflows in float64" replaced by the `1 + 8 w` rounding to 1.0 |
| F3-R3-3 | minor | validation.md | applied (root `conftest.py` option) | `EXP_KERNEL_ULP` defined once in the root `conftest.py` beside the generators, fixture `exp_kernel_ulp`; stated in D1.9, D7.4, plan 2.5/2.14/5/8, validation module table, generator block, MTP inventory ("shared"), V0 and V1 bullets |
| F3-R3-4 | minor | plan.md | applied (fifth-guard option; oracle tagged [B, weekly]) | (a) D5.7 adds the Stage A `get_nlpar_lambda` stub as the fifth guard (raises before validation; its validation arms [B]); Scope, roadmap 0.2 Stage A, plan 2.13/3.2/5, V0 audit note, V7 `test_stage_a_guards_raise_not_implemented` and the DoD updated; (b) `test_file_based_pyebsdindex_end_to_end` tagged [B, weekly] in plan 2.14, plan 3.4 and the validation Local-gated bullet |
| F3-R3-5 | minor | validation.md | applied (verified with `sed` and `git diff`) | ledger entry 1 `:180` -> `:181`; `_data.py:392-449` in requirements Scope and V9; plan 1 "two hunks at 1673 and 1776"; plan 2.12 carries the D1.3 `dtype_out` hint |
| C3-R3-F1 | major | validation.md | applied (compiled-oracle probe re-run: `nout` at the `1e-12` seed, sigma `1e12`) | V1 `test_constant_map_is_an_identity` redesigned: box-mean arms with `saturation_protect=False` (the `n_window * spacing` bound), one protection-on exclusion arm asserting the `n2 == 0` route (bitwise identity, sigma `1e12`, non-self weights exactly 0); dropped from the S6 rows in both files with the reason; D2.4 and D3.5 record the constant-map behaviour under protection (ours unchanged pattern, PyEBSDIndex box mean); V1 table row split |
| C3-R3-F2 | minor | validation.md | applied with F3-R3-1 | V7 header states the `_reduce_chunks` route with the measured chunkings; both-axes claim pointed at `test_pass_two_driver_on_multichunk_dask_array_equals_single_chunk[((1, 9), (2, 14))]` and the driver tests |
| C3-R3-F3 | minor | requirements.md | applied (probe re-run: bitwise under synchronous and threads 1/2/20) | D1.5 (b) "(unverified)" replaced by the measurement; V7 `test_inplace_equals_inplace_false_on_a_multichunk_eager_signal` gets a `scheduler="synchronous"` arm beside the threaded arm (three repeats dropped); plan 2.12 notes both arms; fallback sentence kept |
| C3-R3-F4 | minor | requirements.md | applied (tag source re-fetched, body identical) | D8.4, plan 0.3 oldest-matrix bullet, plan 2.8, plan 7.5 and the validation oldest-matrix bullet say "verified against the dask 2021.08.1 tag"; the private four-line fallback deleted; oldest-matrix run kept as the executable confirmation |
| C3-R3-F5 | minor | validation.md | applied | V0 `test_window_bounds_*`: the `min(2 r + 1, n)` length is stated for `shift=True` only; `shift=False` length `min(c + r, n - 1) - max(c - r, 0) + 1` |
| E3-R3-1 | major | requirements.md | applied (measured `(6, (1,) * 16) == (6, 10)`, `(6, (1,) * 10) == (10,)`) | D8.3: `depth2, chunks_out = _nlpar_depth(...)`, `dask_array = dask_array.rechunk(chunks_out)` before the two `overlap` calls, `nav_chunks = chunks_out[:nav_dim]`, `chunks=chunks_out`; `nav_dim`/`nav_chunks` defined once in D8.2; plan 2.11 writes `x = x.rechunk(chunks_out)` and the post-rechunk `chunks=` for both passes; plan 2.8 adds the measured rechunks |
| E3-R3-2 | major | plan.md | applied | Pathspec `-- src tests doc examples benchmarks conftest.py CHANGELOG.rst` and the ` [DV][0-9]` alternative in plan 5 gate commands, plan 0.3 tech-stack bullet, D11.5 and the validation Local-gated bullet; the same grep on the replay diff added to the section 1 equivalence gate |
| E3-R3-3 | minor | plan.md | applied | plan 2.1: `start`/`stop` clamped to `[0, n]`, `[start, stop)` inside `[0, n)`, `(0, n)` when `n < 2 r + 1`, with the `n=2, r=3, c=0` example |
| E3-R3-4 | minor | validation.md | applied | Oracle call recipe: `dataout` is the full block, zero outside `calclim`, sliced `[rstart:rstart + nrowcalc, cstart:cstart + ncolcalc]` as the driver does (`:507-509`); V3 `test_calclim_from_block_info` compares the sliced region |
| E3-R3-5 | minor | validation.md | applied | V7 `test_argument_validation`: Stage A for `average_non_local_neighbour_patterns` and `get_nlpar_sigma`; every `get_nlpar_lambda` arm [B] |
| E3-R3-6 | minor | plan.md | applied | plan 2.12 `dtype_out: str \| np.dtype \| type \| None = None`; plan 2.7 `omin, omax = dtype_range[np.dtype(dtype_out).type]` |
| E3-R3-7 | minor | validation.md | applied | V7 header: chunkings name the navigation axes, signal axes one chunk (`chunks=c + (-1, -1)`); both driver-test bullets write `da.from_array(x, chunks=c + (-1, -1))` |

## 11. Stage A code-review disposition table (2026-10-05, fixer)

The Stage A code review of `d2b73acd` (fidelity and conventions
reviewers, two sceptics) left 12 findings, all minor; 3 more were
refuted by both sceptics (list below). The fixer verified each
finding before applying it (re-run of the reviewer's
`scratchpad/fidelity/probe4.py`, its own probes under
`scratchpad/fixer_stagea/`; validation.md ledger entry 10) and
edited only its file list: `_nlpar.py`, the three NLPAR methods of
`ebsd.py`, the NLPAR part of `conftest.py` (plus one `import
functools` line in its import block, needed by the moved spy), the
two test modules, `CHANGELOG.rst`, this section, and validation.md
(V4 and V7 dated amendments, ledger entry 10). requirements.md is
outside that list, so every requirements sentence a finding asks
for is proposed here for the main loop, in the "what changed"
column. 11 applied, 1 declined. Final run: the two modules at `-n 0`
751 passed (736 + 15 new), `-n 4` 751 passed, `_nlpar.py` coverage
100.00 % (303 statements), ruff and the clean-replay grep clean.

| id | severity | file | disposition (applied / declined: reason) | what changed |
|---|---|---|---|---|
| F-FID-1 | minor | ebsd.py | applied (code, docstrings, tests); the D1.6 sentence and the plan module 3 note for the main loop | Method: after `_nlpar_mask_indices`, every element of a given sigma must satisfy, in float32, `sigma * sigma > 0` and `np.float32(2 * n_kept) * sigma^2` finite, else `ValueError` "sigma must be > 0 with sigma^2 > 0 and 2 n sigma^2 finite in float32 in every element, n = <n_kept> ..."; the `sigma` parameter states the bound ("about 3e-23 < sigma < 2e17" for 60 x 60 patterns, measured: 2.7e-23 and 2.17e17 pass, 2.6e-23 and 2.18e17 fail). `_nlpar_normalized_distances` docstring states that precondition; its "NaN-free" is now true for every sigma the method lets through (0 / 0 and `-inf / inf` named for the rest), so plan module 3's "den > 0 wherever n2 > 0 ... NaN-free" holds as written once D1.6 carries the check. Tests: `test_argument_validation` arms `sigma=1e19`, `1e-30`, an array with one `1e19` element (fragment "sigma must be > 0 with sigma"); `test_sigma_argument_contract`: 1e18 accepted with finite output, 1e19 rejected, 1e19 accepted with a one-pixel mask (the bound follows `n_kept`; an `n_pix` mutant dies), 1e-30 rejected. Proposed D1.6 addition: "A given `sigma` must also satisfy, element-wise in float32, `sigma * sigma > 0` and `np.float32(2 * n_kept) * (sigma * sigma)` finite, `n_kept` being the pixels left by `signal_mask` ("sigma must be > 0 with sigma"); this check runs after the mask conversion, still within step (2) (amended 2026-10-05, Stage A code review F-FID-1)." Stage B: the same check in `get_nlpar_lambda` |
| F-FID-2 | minor | _nlpar.py | applied | `_nlpar_finalize`: `np.clip(np.rint(out_f32.astype(np.float64)), float(omin), high)`, `high = float(omax)` moved down with `np.nextafter(high, -np.inf)` when it exceeds `omax` (uint64 `2**64 - 2048`, int64 `2**63 - 1024`); Notes say why; 8- and 16-bit outputs bitwise unchanged (`test_dtype_round_trip`, the parity tests). New `TestLazyAndContracts::test_integer_output_keeps_the_maximum_of_32_bit_types[uint32, int32]` (a map at the type maximum returned bitwise) and `test_integer_output_clips_64_bit_types_inside_their_range`; both fail on the pre-fix float32 route and the 64-bit arm on a route without `nextafter` (ledger entry 10 item 1 (d)) |
| F-FID-3 | minor | requirements.md | applied to the docstring; the D7.3 and D12.6 item 12 sentences for the main loop | Re-measured (ledger entry 10 item 1 (b)): 4-5 ulp at lam 0.37, 0.7, 0.9, 1.3 with float32-representable lam, 0 at 0.5 and 1.0 (exact squares), 0 after `rint`. Docstring item 12 now reads: "``lam`` reaches the kernel as float64; PyEBSDIndex's driver passes it as float32 (``nlpar_cpu.py:297``), so its kernel also squares it in float32, and the two differ by a few float32 ulps (up to 5 measured, none after rounding to integers) even for a ``lam`` exactly representable in float32, unless its square is too." Proposed D12.6 item 12 and D7.3: the same sentence, replacing "so for a given `lam` the two agree up to that rounding" and "equals ours only for `lam = float(np.float32(lam))` up to that rounding" (amended 2026-10-05) |
| F-FID-4 | minor | requirements.md | applied to both kernel docstrings and as a test pin; the D3.5 sentence for the main loop | Code unchanged (parity kept). `_nlpar_distances_kernel`: the float32 `max_value + 1` excludes nothing only while `max_value < 2**24`; from `2**24` on the pixels at the maximum drop out of these distances, as in PyEBSDIndex, while the sigma kernel keeps them; `_nlpar_sigma_kernel`: the float64 `max_value + 1` excludes nothing while `max_value < 2**53`. Pin: `TestPolicyOracles::test_uint16_two_threshold_arm` gains a protection-off arm with the maximum at `2**25` (sigma kernel keeps 16 pixels per neighbour slot; distances equal the transcription with the threshold at the maximum, kept counts {15, 16}). Proposed D3.5 addition: "`saturation_protect=False` excludes nothing while the maximum is below `2**24` in the averaging kernel (float32) and `2**53` in the sigma kernel (float64); from `2**24` on the pixels equal to the maximum drop out of the search-window distances while the sigma pass keeps them, so the two passes use different kept sets there, as in PyEBSDIndex (:866-867); recorded, parity kept (amended 2026-10-05)" |
| F-FID-5 | minor | requirements.md | declined here: its only file is requirements.md, outside the fixer's list; no code or docstring needs a change (the code, D2.5 and plan module 4 already say `-inf`) | Proposed quirk-catalogue text: replace "the `-1e6` self slot" with "the self weight forced to exactly 1 (PyEBSDIndex: distance `-1e6`; ours: `-inf`, the same weight)" |
| F-FID-6 | minor | _nlpar.py | applied (with C-CONV-4) | NRL change notice: "Changes by the kikuchipy developers, 2026-10-05:", the date of `ca13e63c` and `d2b73acd`; to be re-checked at the Stage A commit |
| C-CONV-2 | minor | test_ebsd_nlpar.py | applied, with the optional 0-d sigma | `VALIDATION_ARMS` gains 12 arms: `target_weight` 0.0, 1.0 and True; `lam=np.inf`; `dthresh` nan and inf; `sigma=1e39`; the three F-FID-1 range arms; `sigma=np.array(8.0)` accepted and equal to `sigma=8.0` (the method now takes a 0-d array as a scalar, `.item()`); `dtype_out="foo"`. validation.md V7 `test_argument_validation` amended (dated): the method's own `target_weight` arms are [A]; only the `get_nlpar_lambda` arms stay [B] |
| C-CONV-3 | minor | ebsd.py | applied (the code change, not the parity line); the D1.5 sentence for the main loop | `s_out.compute(show_progressbar=False)`: the Dask bar registered by the method covers that computation, so `show_progressbar=False` now draws no bar and `True` one per pass instead of a third HyperSpy bar. `test_show_progressbar_registers_and_unregisters` records the keyword reaching `LazyEBSD.compute`: `[False]` in all four arms of the method (`[None]` before the fix), `[]` for `get_nlpar_sigma`. Proposed D1.5 text: "progress bar as at `ebsd.py:1096-1101`, minus HyperSpy's second bar: the `inplace=False` result is computed with `show_progressbar=False` (amended 2026-10-05, a knowing departure from the precedent)" |
| C-CONV-4 | minor | _nlpar.py | applied (the date); the split of the list deferred to Stage B | Date as F-FID-6. The lambda-objective line stays, since D10.4(c) prescribes the list; at the Stage B implementation commit the notice is re-dated or split ("2026-10-05:" for the engine items, the Stage B date for the lambda objective) |
| C-CONV-5 | minor | ebsd.py | applied | (a) `get_nlpar_lambda` Raises adds "NotImplementedError / Always, for now: the lambda optimisation is not implemented yet." (removed with the guard in Stage B); (b) `data[jn, i_n, q]` in the weighted-sum Notes; (c) method Notes: "An in-memory signal is read once more for the global maximum, and ``inplace=True`` holds one extra copy of the averaged map while it is computed." numpydoc validation clean on the three methods |
| C-CONV-6 | minor | test_ebsd_nlpar.py | applied; the D1.9 sentence for the main loop | Root `conftest.py`: `_circle_mask` with the fixture `circle_mask`, and the fixture `counting_spy(monkeypatch)` returning `spy(module, name) -> list`; both modules' copies deleted; the oracle parity helpers take the mask from their callers; the module-level masks of `VALIDATION_ARMS` became the names "circle" and "circle_int", resolved by `_resolve`. `_require_pin` and `_require_placeholder` left as they are (not part of the fix). Proposed D1.9 addition: "two test helpers, the inscribed-circle mask `circle_mask` and the call-counting `counting_spy`, are shared the same way (amended 2026-10-05)" |
| C-CONV-7 | minor | conftest.py | applied | `EXP_KERNEL_ULP: int = 0` with the comment in past tense and the fixture typed `-> int`; the test_ebsd_nlpar module docstring and constants header rewritten; `\| None` dropped from the eight pinned constants of test_ebsd_nlpar and from `BORDER_BAND_MIN_DIFF`; placeholder comments of test_nlpar in past tense; the `_require_*` guards kept for Stage B; `PYEBSDINDEX_JIT_WARMUP_S` commented as documentation only (no test reads it); the warm-up fixture docstring says the test-suite property is written only by a non-xdist run with `--junit-xml` |

Also done (task items, not findings): the CHANGELOG "Added" bullet
(the conventions reviewer's draft, plus `lazy_output=True` among the
`NotImplementedError` paths as the C-CONV-1 sceptics noted, `#17`
link, PyEBSDIndex and NRL acknowledgement); validation.md V4 (dated
amendment: the `D_STD_BAND` std seed of 2026-09-11, ~1, refuted; the
measured std is 0.83, 0.754-0.888 over 20 seeds, identical to the
compiled `sigma_numba` `dout`) and the matching comment in
`test_ebsd_nlpar.py`.

Refuted by both sceptics (not applied):

- C-CONV-1 (CHANGELOG bullet missing at the pre-review checkpoint):
  the plan schedules the bullet after the review (section 2 Stage A
  gates, section 8 commit 3); it was written now as a task item.
- C-CONV-8 (default-suite addition above the ~90 s per worker):
  recorded, not a gate (D13.5; ledger entry 7 item 6 (d)).
- C-CONV-9 (`inplace=False` drops the axes calibration and shares
  `detector`, `xmap`, `static_background` by reference): the return
  construction D1.5 and D8.9 prescribe, identical to
  `average_neighbour_patterns`.

## 12. Stage B code-review disposition table (2026-10-05, fixer)

The Stage B code review left six surviving findings (three major, all
in the tests, and three minor). The fixer edited only its file list:
`ebsd.py` (the NLPAR method), `test_ebsd_nlpar.py`, `CHANGELOG.rst`,
this section, and validation.md (a dated V8 amendment, the V8 pin
row, ledger entry 15). `_nlpar.py`, `test_nlpar.py` and `conftest.py`
needed no change. 6 applied, 0 declined. Final run of the two modules
at `-n 0`: 816 passed, 5 skipped (812 + 4 new); ruff clean.

| id | severity | file | disposition (applied / declined: reason) | what changed |
|---|---|---|---|---|
| R1 | major | test_ebsd_nlpar.py | applied (test defect, evidence in ledger entries 14 item 6 and 15 item 1) | `test_lambda_forwards_dthresh_sigma_and_protection`: sigma scaled by 0.8 instead of 1.5 (lambda 3.2527 vs 0.8926 at the default; `lam=None` output equals the forwarded-lambda output and differs from the default-lambda output); the comment says that from x 1.1 every numerator is negative and lambda cannot move the output. Assertions unchanged |
| R2 | major | test_ebsd_nlpar.py, validation.md V8 | applied as the reviewer's first option; flagged for the main loop, which may revert it to a code finding | V8 rule amended (dated): a metric NLPAR measurably worsens is pinned as a bounded expected loss at 2x the measured loss, rounded outward. `HOUGH_PQ_GAIN = -3.3` (measured -1.639); the comment records the loss and that fit, cm and misorientation improve. The default suite is green |
| R3 | major | test_ebsd_nlpar.py | applied | The in-place lazy arm adds a `((1, 1, 1, 1), (5,))` input: the `inplace=False` result's chunks differ from it, the in-place result keeps it. The drop-`rechunk(old_chunks)` mutant now dies (ledger entry 15 item 3) |
| R5 | minor | ebsd.py | applied (code, Notes, new test) | `register_pbar` no longer depends on `return_lazy`: the bar covers the eager global maximum, sigma pass and lambda fit of a lazy output. Notes say what the bar covers with a lazy output. New `test_show_progressbar_covers_the_eager_passes_of_a_lazy_output` (4 cases); the reverted-code mutant fails its 2 True cases. Proposed D1.5 sentence for the main loop: "With a lazy output the progress bar, if shown, covers the eager passes (global maximum, sigma, lambda) only (amended 2026-10-05, Stage B review R5)" |
| R6 | minor | ebsd.py | applied | Notes: "A lazy signal is read three times (the global maximum, the sigma pass unless both ``lam`` and ``sigma`` are given, and the averaging)." |
| R8 | minor | CHANGELOG.rst | applied | The "For now ... NotImplementedError" sentence replaced: `lam=None` optimises the weight decay for `target_weight` (default 0.34), `EBSD.get_nlpar_lambda()` returns it, lazy signals are supported (sigma and lambda eager, the averaging lazy unless `lazy_output=False`; `lazy_output=True` from an in-memory signal) |

## 13. Post-implementation spec review (2026-10-06)

The spec re-submission critic (plan section 5, `nlpar-stage-close`,
Opus 5.5), read-only except for this section. Inputs: the three
documents of this folder against `git diff develop...HEAD` (HEAD
a15992db, Stages A and B) plus the uncommitted Stage C working tree
(`nlpar.ipynb`, `examples/pattern_processing/nlpar.py`, `index.rst`,
`run_nbval.sh`, `CHANGELOG.rst`, ledger entry 19). Checked by reading
and by command:

- the method and helper signatures (`ebsd.py`, `_nlpar.py`);
- every `Test*`/`test_*` name in both test modules against the names
  in the three documents. Every spec name exists apart from
  `test_stage_a_guards_raise_not_implemented`, which was deleted on
  purpose. The extra shipped names are the templated
  `_compiled_<generator>` and `<kernel>_py_func_` tests;
- the notebook, cell by cell, with its stored outputs;
- `black --line-length 77 --check` on the notebook and the example
  (both left unchanged);
- ASCII and em-dash scans (clean);
- the clean-replay grep on the Stage C files (clean);
- the quoted numbers against ledger entries 7, 14, 17 and 18. All
  match: lambdas 1.139 / 2.579; ADP 0.600 -> 0.905 and 0.766; IQ
  0.184 -> 0.323; Hough `pq` 80.564 -> 78.926, while fit, cm and
  misorientation improve; `si_wafer` CV 0.603, N_eff 4.32, IQ gain
  1.245; runtimes 1.6 s / 8.5 s.

The method signatures, the defaults, the twelve "Differences" items
and the CHANGELOG bullets match the spec. Every row below is an
amendment dated 2026-10-06. The sections above are not edited, and a
row's amendment text is the reading in force.

| # | location | what the spec says | what shipped | amendment text |
|---|---|---|---|---|
| 13.1 | plan 4.1.3; requirements D12.1; roadmap Stage C box 1; validation Manual (first two items) | The synthetic demo is "the V5 generator" (`two_grain`, grain B = grain A + 30), shown with "the sigma map" of the synthetic map | The notebook builds its own 10 x 16 map of 32 x 32 patterns: grain A is a ramp 40-200, grain B the same ramp reversed, with Gaussian noise 8 and seed 1. Sigma is printed as min / median / max (7.46 / 7.76 / 8.09), not plotted. Lambda comes from `get_nlpar_lambda()` (0.905), not V5's 0.7 / 2.5. Added beyond the spec: the NumPy weight map of boundary pattern (5, 7), in which the grain B weights are exactly 0; a least-squares grain-B fraction per column (NLPAR 0.01 / 1.01 against Gaussian 0.31 / 0.70 at columns 7 / 8); and the noise rms, 8.0 -> 1.7 | The tutorial's two-grain demo is a self-contained map, not the test generator. Grain B is the reversed ramp, so the boundary contrast is visible in the patterns. Its sigma is reported as printed statistics. The nickel sigma map (13.2) meets the sigma-map figure requirement. The figure shows the weight map of one boundary pattern, the noisy / NLPAR / Gaussian 5 x 5 patterns and the grain-B fraction per column |
| 13.2 | plan 4.1.4; validation Manual (items 2 and 3) | Ni: remove the backgrounds, then plot the `get_nlpar_sigma()` map; lambda for targets 0.5 / 0.34 / 0.25, with the curve. The Manual asks for targets 0.1 to 0.9, marks at 0.34 and at the PyEBSDIndex-style value, and sigma maps of both the raw and the corrected map | Raw sigma (median 2.16) and raw lambda (1.139) are printed before background removal. The sigma map is plotted for the corrected map only (median 17.02, colour bar in grey levels). The curve covers targets 0.2, 0.25, 0.3, 0.34, 0.4, 0.5 and 0.6, with 0.25 / 0.34 / 0.5 printed (3.643 / 2.579 / 1.667) and no marks. There is an extra fixed `lam=0.7` arm (IQ 0.253, ADP 0.766) | The Ni section prints the raw sigma and lambda and plots the corrected sigma map only. The lambda curve covers 0.2-0.6, the useful range, with three values printed and no marks. The `lam=0.7` arm is part of the tutorial |
| 13.3 | validation Manual (items 4 and 5) | Before/after patterns at three positions (interior, edge, grain boundary); Hough before/after with IPF maps and `pq`/`cm` histograms | One position, (50, 8), which is also the thumbnail cell. Hough results are a table of medians (`pq`, `fit`, `nmatch`, `cm`) plus the median misorientation to the stored orientations. The `pq` drop is explained with the full-map weekly value, 80.6 -> 79.0 | The shipped form is one before/after pattern pair and a table of medians. IPF maps and histograms are not part of the tutorial |
| 13.4 | validation Manual (item 7) | The parameter guidance covers search radius, lambda, dthresh and saturation | The guidance bullets cover `lam`/`target_weight`, `search_radius`, `dthresh`, reuse of `sigma`/`lam`, `signal_mask` and `dtype_out`. `saturation_protect` is explained in the formulas section only | Saturation handling is documented in the formulas section. The guidance list is as shipped |
| 13.5 | plan 4.1.8; requirements D12.6 heading | Section "Differences from PyEBSDIndex and from upstream #824" | The heading is "Differences from PyEBSDIndex and from an earlier kikuchipy proposal", with #824 linked in the first paragraph. All twelve D12.6 items are present in prose, in a different order | The tutorial heading calls #824 "an earlier kikuchipy proposal". The docstring keeps the D12.6 heading |
| 13.6 | plan 4.3; requirements D12.1 | Execution expected well under 2 min; outputs stored only above ~2 min on the RTD builder; command `uv run --with ipykernel jupyter nbconvert ...` | Outputs are STORED (ledger entry 19). Warm execution takes 16.2 s (16.6 s with two threads). With a cold numba cache and the `nickel_ebsd_large` download, the extrapolated time on ~2 vCPUs is about 2 min or more. With `nbsphinx_execute = "auto"`, RTD does not execute the notebook. The command used was `uv run --no-sync --with ipykernel --with nbconvert jupyter nbconvert --to notebook --execute --inplace doc/tutorials/nlpar.ipynb`, after which the `metadata.widgets` block was stripped | Storing the outputs is the measured decision. The execution command is the `--no-sync --with nbconvert` form |
| 13.7 | plan 4.6; validation Manual (nbval) | nbval runs through `./doc/tutorials/run_nbval.sh`, restricted to the new notebook | nbval is not installed in `.venv`. Ledger entry 19 ran `uv run --no-sync --with nbval pytest --nbval doc/tutorials/nlpar.ipynb --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg -p no:cacheprovider` (18 of 18 passed). `run_nbval.sh` has the `nlpar.ipynb` entry for CI | The local nbval gate is the `uv run --no-sync --with nbval` command. `run_nbval.sh` holds the registration |
| 13.8 | requirements D12.3 | Gallery example: "`nickel_ebsd_small` or the large dataset if cached, default parameters, a before/after figure" | As in plan 4.4: `nickel_ebsd_large(allow_download=True)`, static and dynamic background removed, pattern (50, 8) kept, `lam=None` (the default), IQ histograms and a 2 x 2 figure | D12.3 reads as plan 4.4 does: the large dataset with `allow_download=True` |
| 13.9 | plan 4.6 and section 5 (`nlpar-stage-review`: the fixer "appends the disposition table to this file"); roadmap Stage C box 3; validation Manual and DoD ("the Manual list ticked") | Stage C review dispositions appended to plan.md; html render inspection and the linkcheck via `output.json` recorded; Manual list ticked; one end-user smoke run | The Stage C review dispositions (F1-F7) are in ledger entry 19 only; there is no plan.md table. F1 and F3-F6 are described. F2 is not described anywhere, and F7 only as the cold-cache note. The render inspection and the linkcheck are recorded only as "unchanged from the review run" and "PASS"; the run itself is not recorded. The Manual list is not ticked item by item, and there is no record of a smoke run | Before the Stage C commit, record in the ledger: F2's finding and disposition, the render-inspection and linkcheck evidence, and the Manual list ticks, with 13.1-13.4 as the accepted deviations. Ledger entry 19 stands in for a plan.md Stage C disposition table |
| 13.10 | plan 9; validation DoD ("re-submitted to adversarial review ... fidelity + conventions critics, fixer disposition table appended to plan.md") | Two critics and a fixer table | One read-only spec re-submission critic (plan section 5, `nlpar-stage-close`) that writes the amendment rows of this section | This section is the re-submission record. It meets the definition-of-done item once the main loop folds in or accepts these rows |
| 13.11 | plan 8 (commits 1-7) | Commit 2, then commit 3 ("Implement NLPAR Stage A ...") | Two extra commits sit between them and are already on `origin/feat-NLPAR`: d2b73acd "WIP checkpoint: NLPAR Stage A implementation (pre-review)" and cec8efc9 "WIP checkpoint: NLPAR Stage A review fixes, spec amendments (bug injection parked)". 14a023ef then closes Stage A. Commit 6 (Stage C) had not been made at this review | The history carries the two Stage A checkpoint commits from the parked-session resume. Commit 6 follows this review |
| 13.12 | validation DoD (commit trailer) | `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` | The commits carry the session trailer (`Claude Opus 5.5`), as plan 8's 2026-10-05 amendment says | The trailer always follows the session (plan 8). The Fable string in the validation DoD is history |
| 13.13 | plan 3.4 ("pin only measured improvements ..., 'not worse' otherwise"); validation DoD ("each labelled 'improvement' or 'not worse'") | Two pin classes for the Hough metrics | `HOUGH_PQ_GAIN = -3.3`, a bounded expected loss at 2x the measured -1.639 (V8 amended 2026-10-05, Stage B review R2). It was kept in a15992db, so the option "main loop may revert this" was not taken | The Hough pins have three classes: improvement (>= 0.5x the gain), not worse, and bounded expected loss (2x the measured loss, `pq` only). The R2 decision is final |
| 13.14 | plan 3.3; validation V7 bullet `test_lazy_equals_eager_nickel_ebsd_large_last_chunk_26_26_3` | Processed chunks `((26, 26, 3), (40, 35))` and default chunks `((47, 8), (47, 28))` | The test asserts only the ROW chunks (`default_chunks[0] == (47, 8)`, `processed[0] == (26, 26, 3)`), because the column chunks depend on the dask version (`(25, 25, 25)` on dask 2021.8.1). The depth and bitwise assertions are unchanged (ledger entries 17 h and 18 item 1) | The column chunks quoted for this test are dask 2026.3.0 measurements, not assertions. Only the row chunks are pinned |
| 13.15 | plan 0.3 (tech-stack "Numba-cache flake rule" and "Fixtures" bullets); plan 2.14 (root `conftest.py` contents) | The root `conftest.py` gains the four generators and `EXP_KERNEL_ULP` (and, per section 11, `circle_mask` and `counting_spy`). The cache flake is handled by running `-n 0` first and re-running red tests alone | The root `conftest.py` also records `_WORKER_NUMBA_CACHE_DIR` when a worker starts. Its autouse fixture `_keep_numba_cache_dir_per_worker` restores that directory, with `numba.core.config.reload_config()`, after any test that changed it. The full suite at `-n 4` then ran twice with 4928 passed and 0 failed (ledger entry 18 item 2) | The root `conftest.py` re-asserts each worker's numba cache directory after every test. The rule of running `-n 0` first and re-running red tests alone stays as a fallback. `specs/tech-stack.md` gets the same sentence the next time the constitution is touched |
| 13.16 | plan section 2 module list (2.7-2.11) and 3.2 | Private helpers `_nlpar_mask_indices`, `_nlpar_search_radius`, `_nlpar_saturation_max`, `_nlpar_finalize`, `_nlpar_depth`, the two chunk wrappers, `_nlpar_unpack_sigma_pass`, `_nlpar_sigma` and `_nlpar_average`, then `_nlpar_lambda_objective` and `_nlpar_optimize_lambda` | Also shipped: `_nlpar_check_dthresh`, `_nlpar_check_target_weight`, `_nlpar_check_sigma` and `_nlpar_check_sigma_range` (the D1.6 checks; the last is F-FID-1's float32 range check); `_nlpar_core_bounds(nav_shape, depth, block_info)` (the kept region, shared by both wrappers); `_nlpar_as_map` (two navigation axes with each signal axis in one chunk, which is also the 1-D route); and `_nlpar_lambda(dask_array, sigma, *, ...) -> (lam, sigma)` (the sigma pass plus the optimiser, shared by `get_nlpar_lambda` and `lam=None`) | The module list includes these seven private helpers, with the signatures in `_nlpar.py` |
| 13.17 | requirements D1.5 | Section 12 R5 proposed, for the main loop: "With a lazy output the progress bar, if shown, covers the eager passes (global maximum, sigma, lambda) only" | The code and Notes ship R5 (`register_pbar` no longer depends on `return_lazy`; `test_show_progressbar_covers_the_eager_passes_of_a_lazy_output`). The sentence is not in requirements.md | D1.5 carries the R5 sentence as proposed in section 12 (amended 2026-10-05, applied here by reference) |
| 13.18 | validation V9 `test_si_wafer_sigma_cv` ("a single crystal should have a flat sigma map"); ledger entry 14 item 5 ("whether `SI_SIGMA_CV` should be a robust statistic is for the spec") | A flat sigma map is expected | The plain CV is pinned at 0.634 (measured 0.603). The bulk is flat (robust CV 0.040); 73 of 2500 low-intensity patterns above 2 grey levels set the CV. The tutorial quotes 0.603 and explains the tail | Decided: the plain CV stays the pinned statistic, as in the shipped test. The flat-map expectation holds for the bulk only, and the tutorial documents the tail |
| 13.19 | requirements Scope (non-goal "Edits to `doc/tutorials/hybrid_indexing.ipynb`") | "the new tutorial links to it (D12.5)" | The link requirement is D12.1; D12.5 is the CHANGELOG | Read "(D12.1)" |

### Roadmap Stage C boxes against the tree

The roadmap rule is that a box ticks only when the work is committed
on `feat-NLPAR`. At this review NOTHING of Stage C is committed
(`git log develop..HEAD` ends at a15992db, and the notebook and the
example are untracked), so no Stage C box can be ticked yet. On
content, after commit 6:

- **Box 1 (`nlpar.ipynb` contents):** can be ticked once rows
  13.1-13.5 are accepted (the synthetic sigma map is printed as
  statistics, not drawn as a figure). `hybrid_indexing.ipynb` is
  untouched (no diff) and linked.
- **Box 2 (registration, stored outputs, gallery):** can be ticked:
  - `index.rst` lists `nlpar` after `pattern_processing`;
  - `run_nbval.sh` has the entry in its slot;
  - `tutorials_sanitize.cfg` is rightly untouched, because regex2
    and regex8 cover the PyEBSDIndex speed and PyOpenCL lines;
  - the stored outputs were decided by measurement (13.6);
  - the gallery example is present (13.8).
- **Box 3 (validation matrix, failure-mode review and fixes;
  sphinx-build exit 0):** can be ticked once the records missing in
  row 13.9 are in the ledger (F2, render inspection, linkcheck,
  Manual ticks). Ledger entry 19 (c) records the sphinx-build exit 0.
- **Box 4 (CHANGELOG tutorial bullet; signed commit pushed; spec
  re-submission):** cannot be ticked:
  - the CHANGELOG bullet is only in the working tree (wording as
    plan 4.5, `#17` link, PR number not yet confirmed);
  - the signed commit 6 and its push are still to come;
  - the re-submission is this section, and it is done only when its
    rows are accepted.

### Observations outside spec drift (for the main loop)

- The tutorial calls `si_wafer` "576 MB" (the uncompressed 50 x 50 x
  480 x 480 uint8 array) next to "a larger download". The spec
  records the download as 311 MB zipped, so "576 MB in memory, a
  311 MB download" would be exact.
- Ledger entry 18 ends with the Recorded-results boilerplate
  paragraph ("This section is filled at each stage's ..."), which now
  sits between entries 18 and 19.
