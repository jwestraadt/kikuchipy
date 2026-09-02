# Phase 7 -- `spherical-refinement`: requirements

## Scope

In scope:
- **`_xcorr.py` additions** (private), the Newton refinement of
  `EMSphInx/include/sht/sht_xcorr.hpp` @ 60f3517 that Phase 4 deferred:
  the **`_derivatives`** Numba kernel (`Correlator<Real>::derivatives()`,
  `:889-1119`: direct cross-correlation value, 3-vector Jacobian and
  symmetrised 3x3 Hessian at one ZYZ rotation, with the per-call
  `dTablePre` rebuild `:913`, the Chebyshev multiple-angle recursions
  `:927-984`, the analytic first/second beta-derivative coefficients
  `:1009-1041` -- the formulas Phase 3 pinned in
  `tests/test_indexing/test_spherical_wigner.py::_phase7_derivatives` --
  and the `der=False` value-only branch `:1080-1105`); the module-level
  **`_refine_peak`** (`Correlator<Real>::refinePeak()`, `:442-499`:
  Newton with the 3x3 Cholesky solve, monotone-step rule, saddle
  rejection, the beta ~ 0/pi degeneracy fallbacks, `absEps = eps 2pi/slP`
  stopping, `maxIter = 15`, unrefined-return failure semantics); the
  **`refine_zyz`** entries on `SphericalCrossCorrelator` and
  `NormalizedSphericalCrossCorrelator` (D6; Phase 8 refines its
  pseudo-symmetric candidates through them, `indexer.hpp:243-261`);
  **`correlate(refine=True)` wired** on both correlators (the two
  Phase 4/6 `NotImplementedError`s at `_xcorr.py:1799-1804` and
  `:2078-2083` deleted), the normalised variant dividing the refined
  value by **`_denominator(zyz)`** (`NormalizedCorrelator::refinePeak` +
  `Constants::denominator`, `:1169-1172`, `:1211-1225`); the shared
  `wigner_d_table_factors` triple (eager on a refining indexer, lazy
  on a standalone correlator) and the per-instance
  `d_beta` buffer (D3; the Phase 3 `out=` tripwire runs at
  allocation).
- **`_indexer.py` changes**: `SphericalIndexer(refine=True)` works and
  becomes the **default** (D7; `IndexEBSD`'s `refine=true`,
  `nml.hpp` default); the per-candidate refine sits inside the phase
  loop exactly where `indexImage` puts it (`:230`,
  `correlate(gln, ref)` -- **every** phase's single candidate is
  refined *before* insertion, not only the winner -- D7);
  `memory_per_worker_bytes` and the info message gain the refine
  buffer term (D8); **`SphericalIndexer.refine_patterns`** and the
  chunk worker `_refine_chunk` (the refine-only pipeline, D9).
- **`EBSD.spherical_indexing`**: `refine: bool = True` default (D7;
  a behaviour change against the unreleased Phase 6 default, called
  out in the CHANGELOG `Added` entry -- D11).
- **`EBSD.refine_orientation_spherical(xmap, harmonics, detector, ...)
  -> CrystalMap`** in `signals/ebsd.py` (D9), the refine-only work
  item of `idx.hpp:438-450` (`msk & 0x02`) with the **intended**
  semantics -- the shipped C++ `refineImage` discards its own
  refinement result (D5, a newly recorded EMSphInx defect).
- **Tests** in `tests/test_indexing/test_spherical_xcorr.py` (new
  refine classes), `tests/test_indexing/test_spherical_indexer.py`
  (refine plumbing, memory model), and
  `tests/test_signals/test_ebsd_spherical_indexing.py` (real-data
  refined bounds, the default flip, `refine_orientation_spherical`);
  targeted updates to the Phase 6 tests that pinned coarse values
  through the old default (they pass `refine=False` explicitly, D7).
- **CHANGELOG** entry (`Added`, PR **#9**; no `Changed` entry -- the
  coarse-only default was never released, D11), the constitution
  amendments of plan section 0.

Out of scope: pseudo-symmetry candidates and their refinement loop
(`indexer.hpp:243-261` bodies; Phase 8 -- but `refine_zyz` is shaped
for it now), the window-shift chain rule the C++ itself omits
(`sht_xcorr.hpp:263-264`, documented caveat -- D6), a refinement-only
bandwidth (`derivatives`' `mBW < bw` path: every C++ call site passes
`bw`, and `wigner_d_table_pre` refuses mismatched tables --
`_wigner.py:1519-1526`), exposing the convergence tolerance `eps`
publicly (D9; the C++ default `0.01` of `sht_xcorr.hpp:189` is used
everywhere and `IndexEBSD` exposes no knob), `extractBunge` (Phase 9),
`IndexEBSD.exe` parity runs (Phase 10; they use the default
`emsphinx_compatible=True`), the tutorial (Phase 11), a float32 path.

## Decisions

Every "measured" number comes from the drafting probe -- a faithful
transcription of `derivatives()`/`refinePeak()`/`denominator()` on top
of the **merged** Phase 1-6 modules (`p7_probe*.py`, scratchpad, not
committed; recipes in `validation.md` "Recorded results") -- and is
re-measured with the real implementation. No compiled C++ driver in
this phase: the value path is pinned by two analytic oracles
independent of the port (D2), the derivative formulas by the Phase 3
tables, and the Phase 4 C++ driver already validated the shared
machinery bitwise.

### D1 -- `_derivatives`: kernel, layout, flags (frozen)

- ```
  _derivatives(flm, gln, eu, jac, hes, bandwidth, mirror, n_fold,
               der, d_beta, e_km, w_jkm, b_jkm) -> float
  ```
  `@njit(cache=True, nogil=True, error_model="numpy")`, a faithful
  transcription of `derivatives()` (`:889-1119`) with `mBW == bw`
  always (the C++ parameter is dead freedom -- every call site passes
  `bw`, and the factor-table stride note of `_wigner.py:1519-1526`
  applies). `flm`/`gln` are `(bw, bw)` complex128 `alm[m, l]`;
  `eu = (alpha, beta, gamma)` float64; `jac` `(3,)` and `hes`
  `(3, 3)` float64 caller-owned outputs written only when `der` is
  true (Jacobian `d/dalpha, d/dbeta, d/dgamma` = `wrk[1..3]`,
  Hessian `[[aa, ab, ga], [ab, bb, bg], [ga, bg, gg]]` =
  `wrk[4,7,9,5,8,6]` symmetrised, `:1110-1117`); the return value is
  the cross-correlation at `eu` in the Phase 4 D4 normalisation
  (`4 pi <f_rotated, g>`).
- **Beta handling**: `beta = _euler._wrap_beta(eu[1])` -- the same
  `fmod` wrap into `[-pi, pi]` as the C++ `:895-899` (the
  constitution's every-beta-taking-entry-point rule is the C++'s own
  here); `t = np.cos(beta)`, `negative_beta = copysign(1, beta) < 0`
  (`signbit`, so `-0.0` counts negative), `csc = (1/np.sqrt(1 - t*t))
  * (-1 if negative_beta else 1)` (`:911`). **The pole path must use
  NumPy scalars, not `math.*`** (frozen rule): the compiled
  `_wrap_beta` boxes its float64 return to a Python `float`
  (measured), so a `math.cos`/`math.sqrt` transcription makes the
  `.py_func` twin raise `ZeroDivisionError` at the poles *despite*
  `np.errstate` (measured -- `np.errstate` has no effect on
  Python-float division), which would make the pole branch
  unreachable through `py_func` and the D11 coverage target
  unmeetable. `np.cos`/`np.sqrt` keep the NumPy-scalar chain so the
  `py_func` yields `inf`/NaN under `np.errstate`; measured: the
  compiled results are bitwise identical to the `math.*` forms (worst
  abs diff 0 over 1000 samples). Precedent: the seeded NumPy-scalar
  accumulators of `_preprocessing.py:418-429`, recorded there for
  exactly this reason. **`error_model="numpy"` is load-bearing** in
  the compiled kernel: at `|t| == 1` the `csc` division yields IEEE
  `inf`, the derivative coefficients `:1015-1020` go `inf`, `d1P/d2P`
  go `inf`/NaN and `hes[1, 1]` accumulates **NaN -- which is
  `refinePeak`'s degeneracy detector** (`:461`, `:468`). Numba's
  default model would raise `ZeroDivisionError` instead and break the
  C++ control flow; this is the constitution's third sanctioned
  `error_model="numpy"` kernel (amendment, plan 0.5). The C++ `deg`
  flag (`:909`) is computed and **never used** -- dead code, not
  ported, noted in the docstring.
- **Wigner d table**: `_wigner._wigner_d_table_pre_kernel(bw, t,
  negative_beta, d_beta, e_km, w_jkm, b_jkm)` is called **inside** the
  kernel once per evaluation (the C++ `:913`), on a caller-owned
  `(bw, bw, bw, 2)` buffer (the pre-kernel writes every defined slot
  each call, so reuse across iterations, calls and patterns is
  exact). Table reads are the C++ flat indices re-expressed through
  the array's own strides: `d0P/d0N = d_beta[m, n, j, 0/1]`,
  `d0P_1/d0N_1 = 0 if m >= j else d_beta[m+1, n, j, 0/1]`,
  `d0P_2/d0N_2 = 0 if m+1 >= j else d_beta[m+2, n, j, 0/1]`
  (`:1027-1032`; the
  guards keep every read inside the defined region, so no NaN slot is
  ever read -- and `m+2 <= j <= bw-1` keeps them in bounds).
  Because the guards keep every read defined, the Phase 3 NaN
  tripwire has **no run-time effect on this read path** (the direct
  kernel call bypasses the wrapper's check; review-measured, a
  `1e300` fill leaves refinements bitwise unchanged); the tripwire is
  therefore enforced **once, at allocation**: `d_beta` is built by
  routing the fresh NaN buffer through
  `wigner_d_table_pre(bw, 1.0, False, e_km, w_jkm, b_jkm, out=...)`
  (D3), which validates the factor triple and the `out=` contract at
  the only moment a wrong buffer could enter.
- **Derivative coefficients** (`:1009-1020`, `:1035-1041`): verbatim
  -- `rjm = sqrt((j-m)(j+m+1))`, `coef2_2 = 0 if j == m else
  sqrt((j-m-1)(j+m+2)) rjm` (the `jm == 0` guard keeps the negative
  radicand of the `k == j` slot out, exactly as Phase 3's
  `wigner_d_prime2` records), `d1P = d0P coef1_0PP - d0P_1 rjm`,
  `d1N = d0N coef1_0PN + d0N_1 rjm`, `d2P/d2N` with the three-term
  forms. These are **the formulas
  `test_spherical_wigner.py::_phase7_derivatives` pinned against the
  Phase 3 scalar ports to `1e-12`** (its docstring: "Phase 7 will
  inline"); the implementation transcribes the C++, the tests re-pin
  through the kernel outputs (D2).
- **Loop structure** (`:930-1107`): Chebyshev multiple-angle
  recursions for `exp(i m alpha)` / `exp(i n gamma)` transcribed
  verbatim (seeds `:935-937`, recursion + sign-corrected `sin`
  extraction `:938-950`; measured against direct `sin/cos`: worst
  `2.9e-13` at `alpha = 1.5498` -- adjacent to `pi/2`, the worst case
  -- over `m <= 119`); the `m % n_fold != 0` `continue` *after* the
  recursion update (`:951-952`); `conjMult(expAlpha, expGamma)` and
  the `(-1)^(n+m)` / `(-1)^n` sign prefactors `:996-1000`; `start =
  max(m, n)` with the mirror parity bump `:1003-1004` and `dJ = 2 if
  mirror else 1` (`:922`; the `gMir` branch `:1005` is dead --
  `glnFold == 1`, `gMir == false` exactly as Phase 4 D2 recorded for
  `compute()`); `vp/vc = conjMult(flm[m, j], gln[n, j])` with the
  `(j+m)` parity negation `:1046`; the ten accumulator components
  `:1057-1070` and the **conditional quadrant accumulation**
  `:1072-1078` (`+m+n` always; `+m-n` when `n > 0`; `-m+n` and
  `-m-n` when `m > 0`), implemented with weights `wc = 1 + (m>0 and
  n>0)`, `wp = (n>0) + (m>0)` -- an association change relative to
  the C++'s four sequential `std::transform` adds, permitted because
  no test asserts bitwise against the compiled C++ (CI lesson) and
  the analytic oracles bound the whole evaluation at `1e-13`
  (measured, D2).
- **Validation lesson (measured, blocker-grade)**: with
  `BOUNDSCHECK` off, a shape mismatch between the spectra and the
  kernel's `bandwidth` is **silent garbage, not an error** -- the
  drafting probe fed `(68, 68)` spectra to a `bw` 88 kernel and got
  values of `1e225`-`1e287` and NaN Hessians (out-of-bounds reads).
  Therefore every public entry validates before the kernel:
  `refine_zyz` runs `flm`/`gln` through Phase 4's
  `_validated_spectrum(., bw, name)`, `zyz0` must be shape `(3,)`
  finite (`ValueError` otherwise), and the factor tables/`d_beta`
  are built by the instance itself (never caller-supplied), so their
  shapes are correct by construction.

### D2 -- What `_derivatives` computes: oracles and measured errors (frozen)

- **The oracle fixture is part of the frozen contract**: both oracle
  tests draw their spectra with the `randomPair` coefficient recipe
  of `sht_xcorr.cpp` (`random_alm`: every defined `a[m, l]` slot
  uniform in `[-1, 1]` for real and imaginary part, `m = 0` rows
  real, symmetry flags respected), exactly as the drafting probe did.
  The absolute bounds below are frozen **for this recipe** -- a
  fixture that rescales the spectra rescales every error linearly and
  invalidates them (a review probe with differently normalised
  spectra measured values O(6.7e3) and an FD-hes error of 1.06e-3).
  The tests therefore `record_property` the value and Hessian scales
  so a fixture drift is visible; re-measured scales under the frozen
  recipe: `bw` 16 value fixture max `|value|` 19.4 (self powers
  13-166), `bw` 12 jac/hes fixture max `|value|` 12.7, max `|hes|`
  5.3e2 (seed-dependent, O(1e2)).
- **Value**: identical to Phase 4's D4 closed form. Measured at `bw`
  16 against the **inner-product oracle**
  `<rotate_harmonics(flm, zyz), gln>` (Phase 3 machinery, independent
  of the kernel) over `(n_fold, mirror)` in `{(1,F), (2,T), (3,F),
  (4,T)}` x 6 random rotations: worst abs **6.05e-14**, rel
  **1.81e-13**; `der=True` and `der=False` return the
  same value. Frozen assertion: `abs <= 1e-11` (165x margin), both
  `der` settings, at `bw` 16 and (weekly) 24.
- **Jacobian/Hessian**: measured at `bw` 12 against the **analytic
  triple-sum oracle** built from Phase 3's `wigner_d`,
  `wigner_d_prime`, `wigner_d_prime2` (negative orders via
  `a^l_{-m} = (-1)^m conj(a^l_m)`; `d/dalpha -> i m'`,
  `d/dgamma -> i n'`, `d/dbeta -> d'`): worst abs value
  **5.51e-14**, jac **2.98e-13**, hes **1.71e-12**.
  Frozen assertions: value `abs <= 1e-11`, jac `abs <= 1e-10`, hes
  `abs <= 1e-9` (335-585x margins). Central finite differences
  (`h = 1e-5`, `bw` 16) agree to jac **2.97e-07** / hes **6.66e-04**
  (truncation-limited) -- kept as a second, formula-independent
  oracle with frozen bounds `1e-5` / `1e-2`.
- The `hes[1, 1]`-NaN degeneracy contract: at `beta` exactly `0` or
  `-0.0`, `hes[1, 1]` is NaN and the beta slot of `jac` is NaN while
  `jac[0]`, `jac[2]`, `hes[0, 0]`, `hes[2, 2]` stay finite (measured
  at `bw` 24: `jac = [-0, nan, -0]` at an on-peak start) -- exact by
  definition (`cos(0.0) == 1.0`). At `beta = +-pi` the same NaN
  pattern holds **only because the host libm returns
  `cos(+-pi) == -1.0` exactly** (measured true here, and one ulp
  below `pi` as *input* still gives `-1.0` -- the dependence is on
  the libm's output, not the input; a libm returning `-1 + ulp`
  would give a finite `csc ~ 6.7e7` and a huge finite Hessian
  instead, taking the ordinary Newton path -- the C++ has the
  identical dependence, so this is parity, not a defect). The named
  test pins exact NaN-ness per slot at `beta in {0.0, -0.0}` and
  asserts `isnan(hes[1, 1]) or |hes[1, 1]| > 1e12` at `+-pi` with
  the observed value recorded, plus in all cases that `_refine_peak`
  runs its fallback and returns without raising (D10; the CI lesson
  on float-boundary knife edges).

### D3 -- Buffers: the factor triple and `d_beta` (frozen; measured costs)

- The **factor triple** `(e_km, w_jkm, b_jkm) =
  wigner_d_table_factors(bw)` (the C++ `wigE/wigW/wigB` of
  `Constants`, `:361-370`) is cached on the correlator and handed to
  every `clone()` read-only (the pre-kernel only reads it). Laziness
  alone would not share it in the Phase 6 flow -- the chunk workers
  clone *before* their first refinement, so each clone would build
  its own 5 MB triple. Frozen rule:
  **`SphericalIndexer(refine=True)` builds the triple eagerly at
  construction and passes it to every correlator through the new
  constructor kwarg `wigner_d_factors=`** -- to every per-phase
  `NormalizedSphericalCrossCorrelator` when `normalize=True` **and to
  the shared prototype `SphericalCrossCorrelator` when
  `normalize=False`** (`_indexer.py:526-529`; without the kwarg on
  the prototype every chunk clone would lazily rebuild the triple --
  the exact failure this rule exists to prevent); both classes'
  `clone()` passes a built triple through (shapes/dtype validated
  with `wigner_d_table_pre`'s own checks at the kwarg boundary).
  **`refine_patterns` builds and shares the triple the same way
  before chunking, whatever the constructor's `refine` flag**
  (idempotent; the method always refines, and
  `SphericalIndexer(refine=False).refine_patterns(...)` is a public,
  reachable call which would otherwise hit the per-clone rebuild --
  D9, with an `is`-sharing assertion on this path in validation). A
  standalone correlator builds the triple lazily on its first
  `refine_zyz`.
  Size `8 bw^2 + 16 bw^3` bytes = **5.07 MB** at `bw` 68, shared
  per process.
- The **`d_beta` buffer** (`(bw, bw, bw, 2)` float64, the C++
  per-instance `dBeta`, `:383`) is allocated per correlator instance
  (the normalised class owns it through its inner plain correlator)
  on first use -- a fresh `np.full(..., np.nan)` routed once through
  `wigner_d_table_pre(..., out=)` so the Phase 3 tripwire contract is
  actually checked at the only moment a wrong buffer could enter
  (D1: the per-evaluation call is the raw kernel, which cannot
  check) -- and **reused across
  iterations, calls and patterns**. Measured allocation cost
  `np.full`: **0.50 / 1.03 / 2.27 ms** at `bw` 53 / 68 / 88 -- i.e. a
  fresh buffer per refinement would cost more than the whole
  refinement at `m-3m` flags (0.33 / 0.65 / 2.54 ms) -- so per-call
  allocation is rejected. Per-thread ownership is the Phase 3
  contract verbatim (`_wigner.py:1486-1490`): clones never share
  `d_beta`; `clone()` allocates the clone's own lazily.
- **Memory model** (D8): `memory_per_worker_bytes` gains
  **`+ n_correlators * 16 bw^3`** bytes -- `n_correlators` exactly as
  the Phase 6 model already defines it (`n_phases if normalize else
  1`, `_indexer.py:1085`), because a worker holds one `d_beta` **per
  correlator clone** (`NormalizedSphericalCrossCorrelator.clone`
  gives every per-phase clone its own inner plain correlator,
  `_xcorr.py:2114`, and clones never share `d_beta`); a flat
  `+ 16 bw^3` would understate every multi-phase normalised run by
  `(P-1) * 16 bw^3` and make the 2 GiB warning under-fire exactly on
  the runs that need it. Per correlator the term is 5.03 / 10.9 /
  23.1 MB at `bw` 68 / 88 / 113 (`16 * 88^3 = 10,903,552` B; an
  earlier draft's 11.5 was arithmetic error, re-measured). The term
  applies when the model describes a refining run (`refine=True`, or
  the always-refining `refine_patterns` path -- D8/D9), nothing on a
  coarse-only model (the Phase 6 number is kept; the Phase 6
  model-arithmetic test is updated accordingly, D7). At `bw` 68:
  single phase 49,426,200 + 5,030,912 = **54,457,112 B -> "54 MB"**
  in the info message; two phases normalised 79,169,400 + 2 x
  5,030,912 = **89,231,224 B** (both pinned in validation, so a
  dropped `n_correlators` factor dies by a named test).

### D4 -- `_refine_peak`: the Newton loop (frozen; `:442-499`)

- Module-level Python function (not a kernel: the loop body is two
  njit calls plus scalar logic; measured whole-refinement costs in
  D8 make the Python overhead irrelevant), signature
  `_refine_peak(flm, gln, zyz0, n_fold, mirror, bandwidth,
  side_length, d_beta, e_km, w_jkm, b_jkm, jac, hes, step,
  eps=0.01) -> (zyz, value, converged)`; the correlators wrap it.
- Constants, verbatim: `abs_eps = eps * 2 pi / slP` (`:446`; `eps`
  default **0.01** from `PhaseCorrelator::correlate`, `:189`),
  `eu_eps = sqrt(machine eps)` (`:447`), **`max_iter = 15`**
  (`:448`; the research doc's "25?" is `interpolateMaxima`'s own
  Newton at `:1312`, a different loop -- Phase 4 owns it),
  `prev_mag2 = 2 pi * 3 / slP` (`:450`) -- **a recorded C++ quirk**:
  the comment says "first step better not be more than 1 pixel in
  each direction" but the bound compares `|step|^2` against a
  *linear* quantity; at `slP` 135 it admits a first step of
  `|step| <= 0.374` rad ~ 8 grid cells (a one-cell-per-axis step is
  `mag2 = 3 (2 pi/slP)^2 = 0.0065`). Ported verbatim, documented.
- Iteration, verbatim semantics with exceptions mapped to statuses:
  `value = _derivatives(..., der=True)`; **try-path**: if
  `isnan(hes[1, 1])` -> fallback; else
  `_preprocessing._cholesky_solve_3x3(hes, jac, step)` -- **`hes` is
  passed without a copy, exactly as the C++ passes the live array**
  (`:462`): the decomposition writes only the subdiagonal
  (`a[j, i]`, `j > i` -- `linalg.hpp:425`, `_preprocessing.py:361`),
  the fallback reads only `hes[0, 0]`, `hes[1, 1]` and `hes[0, 1]`
  (`:470`, `:475`) -- all preserved -- and `_derivatives` rewrites
  all nine entries on the next iteration; measured over 15 seeded
  fallback-heavy cases (12 far starts + 3 near-degenerate targets,
  `bw` 24): `(zyz, value, converged, iterations)` bitwise identical
  with and without a copy, so a copy would buy nothing (an earlier
  draft required one with a self-refuting rationale). **The Phase 5
  kernel is the same routine** the C++ calls
  (`solve::cholesky`, `linalg.hpp:354-358` = `decompose:411-431` +
  `backsolve:487-494`, the very lines the Phase 5 header cites):
  status 1 = the sign-mismatch throw `:416` (indefinite -> saddle
  rejection), status 2 = the small-pivot throw `:422`; **imported
  from `_preprocessing`** -- no cycle (`_preprocessing` imports only
  numba/numpy, `_xcorr` -> `_preprocessing` is a new one-way edge),
  one source for a numerically delicate kernel whose
  comparison-direction tests and flags live in Phase 5 (duplication
  rejected); then `mag2 = |step|^2`, `if mag2 >
  prev_mag2` -> fallback else `prev_mag2 = mag2` (`:463-465`;
  NaN comparisons are false, as in C++). **Fallback** (`:466-484`):
  if `isnan(hes[1, 1])` -> the 1x1 sub-problem `step = [jac[0] /
  hes[0, 0], 0, 0]` under `np.errstate` (the C++ divides unguarded;
  `hes[0, 0] == 0` gives `inf`/NaN steps that run to `max_iter` and
  fail, IEEE semantics preserved); else the 2x2 sub-problem: `det =
  hes[0,0] hes[1,1] - hes[0,1]^2`; `if det < eu_eps` -> **total
  failure** (the C++ `:476-479` distinguishes "singular" from
  "converging to saddle" in messages only -- its inner
  `if(det < euEps)` is always true when reached, a recorded
  quirk -- both throw); else `step = [(jac[0] hes[1,1] - jac[1]
  hes[0,1])/det, (jac[1] hes[0,0] - jac[0] hes[0,1])/det, 0]`.
  Apply `eu -= step`; converged when `max(|step|) < abs_eps`
  (`:487-489`); `iter == max_iter` -> total failure (`:490`).
- **Failure semantics** (`:494-498`, frozen): on total failure
  `zyz = zyz0` (the *input* triple, i.e. the interpolated coarse
  result in the `correlate` flow) and the returned value is
  `_derivatives(zyz0, der=False)` -- the **analytic value at the
  coarse point, not the tri-quadratic peak estimate** (faithful; a
  failed refine therefore *changes the score* of a coarse result,
  measured on far-start cases returning values from -29.4 to +9.3).
  `converged=False` is the port's addition for tests and
  `refine_patterns`; `refine_zyz`/`correlate` do not expose it
  (C++ parity -- silent).
- **Value-lag quirk (recorded, faithful)**: on convergence the
  returned value was computed at the `eu` *before* the final
  sub-`abs_eps` step -- second-order small (the step is `< 1 %` of a
  cell at a stationary point) and exactly the C++'s behaviour
  (`peak` is assigned before the step is applied, `:457`, `:487`).
- Measured convergence (synthetic pairs, D10): **2 iterations** in
  every non-degenerate case (72 group cases + 30 symmetry-free), 3
  occasionally near cell edges; real data 8x2 + 1x3 of 9 (small),
  152x2 + 13x3 of 165 (large); zero failures in every
  peak-started refinement. The C++ comment "generally at most 3
  iterations" (`:448`) is confirmed.

### D5 -- Degeneracy behaviour and two newly recorded EMSphInx defects

- **Near-degenerate targets** (measured, `bw` 24, random `flm`):
  a true rotation at `beta = 0` refines from a coarse start 1.20 deg
  away to **0.0 deg**; at `beta = +-1e-3` rad the refined result is
  **5.73e-2 deg** (= the `1e-3` rad beta offset: the 1x1/2x2
  fallbacks freeze the "false DoF", so the residual is the beta
  distance itself -- the C++'s documented near-degeneracy price); at
  `beta = pi` / `pi - 1e-3`: 0.0 / 5.73e-2 deg. Frozen test bound:
  refined `< 0.1` deg for targets within `1e-3` rad of the poles,
  where coarse is ~1.2 deg (the D5 Phase 4 defect zone is *cured* by
  refinement away from the poles: the `beta = 0` target itself is
  exact).
- **Starting exactly on `beta = 0`**: `hes[1, 1]` NaN -> the 1x1
  alpha step; measured: converges (1 iteration on-peak). Starting on
  `beta = +-pi` with the peak at `beta = 0`: converges *along the
  ridge* to the `beta = pi` local maximum 180 deg away -- Newton is
  local, faithful, recorded (not a test failure mode: `correlate`
  starts at the argmax cell).
- **Far starts** (unrelated random start, `bw` 24, re-measured over
  a 40-case seeded sweep -- the first 10 reproduce the original run):
  **36/40 fail** (saddle rejection / non-convergence) and return the
  start unchanged with the analytic value there (including negative
  values); **4/40 converge**, moving 1.9-4.1 deg, and **3 of those 4
  land on a stationary point whose un-normalised value is *below*
  the start's** (worst: moved 3.642 deg, value -19.936 -> -27.293;
  the originally recorded "walks 4.1 deg to a nearby local maximum"
  is one of these decreasers, -27.786 -> -29.353, and is corrected:
  the 1x1/2x2 fallbacks freeze `step[2]` and check only
  `det >= euEps`, so a converged fixed point need not be a maximum).
  **Newton is local**: a start that did not come from the coarse
  pipeline may converge to a spurious stationary point with a lower
  -- even negative -- score, so refinement is score-monotone **only**
  for starts produced by `spherical_indexing`; this caveat goes in
  the `refine_orientation_spherical` docstring (D9) and one
  converged decreaser is pinned by a named test (D10). The failure
  half is the saddle-rejection contract of `:462` working as
  designed.
- **EMSphInx defect 1 (new): `refineImage` discards its refinement.**
  `indexer.hpp:296` calls `refine(res.phase, gln.data(), eu)` and
  drops the returned `Result`; `refine` copies `eu` into a local
  (`:341`) and `eu` is `const` -- so `:297` converts the *unchanged*
  starting orientation back, and `idx.hpp:445` stores `res[0].corr`,
  which the refine-only branch **never assigned**. That stored score
  is **0 or stale, not indeterminate**: `std::vector<Result>
  res(om.size())` value-initialises (zero-initialises `corr` --
  `Result` is an aggregate with no default member initialisers,
  `indexer.hpp:54-64`) and `res` is hoisted *outside* the per-pattern
  loop (`idx.hpp:406-407`), so a pure `msk & 0x02` run stores 0 and a
  mixed 0x01/0x02 batch stores the **previous pattern's**
  `indexImage` score. EMSphInx's refine-only work items
  (`idx.hpp:438-450`) therefore return the input orientation with a
  zero or stale score. Phase 7 implements the
  documented *intent* (refine the stored orientation, store the
  refined score) and records the deviation in the
  `refine_orientation_spherical` docstring `Notes`, the licence
  block and the research addendum (plan 0.6).
- **EMSphInx defect 2 (new): the `prevMag2` unit mix** (D4). Ported
  verbatim; recorded.
- Also recorded: the dead `deg` flag (`:909`), the always-true inner
  `if(det < euEps)` (`:476-478`), the value lag (D4).

### D6 -- `refine_zyz`, `correlate(refine=True)`, the normalised divide (frozen)

- `SphericalCrossCorrelator.refine_zyz(flm, gln, n_fold, mirror,
  zyz0, *, eps=0.01) -> (zyz, score)`: validates as D1, ensures the
  factor triple and `d_beta` exist (D3), calls `_refine_peak`. The
  un-normalised score is the analytic xc value at `zyz` (D4).
- `NormalizedSphericalCrossCorrelator.refine_zyz(gln, zyz0, *,
  eps=0.01) -> (zyz, score)`: the un-normalised `_refine_peak` on the
  stored `flm`, then **`score = value / self._denominator(zyz)`**
  (`:1169-1172`) with `_denominator(zyz)` the port of `:1211-1225`:
  `mrf = _derivatives(flm, mlm, zyz, der=False)`, `mrf2 =
  _derivatives(flm2, mlm, zyz, der=False)`, `s2m = mlm[0, 0].real
  sqrt(4 pi)`, `sqrt(mrf2 - 2 (mrf/s2m) mrf + (mrf/s2m)^2 s2m)`,
  both `derivatives` calls with the **master's** `(n_fold, mirror)`
  flags as the C++ passes `mr, nf`. No guard on the radicand (the
  C++ has none; Phase 4 D8's zero/negative-radicand consequences
  carry over -- for the real Ni data the radicand at the refined
  point is O(1) and positive, measured). The **window chain-rule
  caveat** (`:263-264`) is ported as-is and documented: the Newton
  step maximises the *un-normalised* correlation, so the refined
  *normalised* score can occasionally dip below the coarse one
  (measured: 4 of 165 large-map points, worst **-4.8e-4**; never on
  the small map) and the normalised refined accuracy is
  mask-limited (measured 2.1e-2 deg vs 3e-6 unmasked, D10).
- `correlate(..., refine=True)` on both classes: the Phase 4 flow
  (compute, peak, `interp_peak`) then `refine_zyz` from the
  interpolated triple, returning the refined `(zyz, score)`
  (`:394-400`, `:1140-1159`). The `emsphinx_compatible` keyword
  affects only the *start* (the `x[2]` bug and glide of Phase 4 D5);
  the refinement itself has no compatibility branch. `refine`
  becomes a plain flag and **keeps its `False` default on both
  private correlators** -- a deliberate deviation from the C++'s
  `ref = true` defaults on both `correlate` overloads
  (`sht_xcorr.hpp:189`, `:255`), recorded here: the indexer owns the
  user-facing default (D7), and flipping the private default too
  would silently change every Phase 4 test and consumer that calls
  `correlate` bare. Both `correlate` signatures are pinned in
  validation so the choice cannot drift. The two
  `NotImplementedError`s and their docstring paragraphs are deleted.
- The refined `zyz` is **not wrapped** into the coarse result
  intervals -- Newton may step `beta` across a pole or `alpha/gamma`
  slightly outside the Phase 4 D7 ranges (steps are sub-cell from an
  in-range start; `_derivatives` wraps `beta` internally, and every
  consumer converts through `rotation_from_zyz`). The Phase 4
  range assertion applies to `refine=False` results only (its test
  keeps that keyword explicit).

### D7 -- The default flip and the indexer plumbing (frozen)

- **`SphericalIndexer(refine: bool = True)`** and
  **`EBSD.spherical_indexing(refine: bool = True)`**: the Phase 6
  guard (`_indexer.py:877-881`) is deleted; `IndexEBSD`'s namelist
  default (`refine=true`) is restored now that refinement exists
  (roadmap Phase 7 note; Phase 6 plan decision 8 anticipated the
  flip). **User-visible behaviour change** against the unreleased
  Phase 6 default, stated in the CHANGELOG
  `Added` entry (D11) and both docstrings: a default call now returns
  Newton-refined orientations (small map: median 0.599 -> 0.505 deg
  against the stored xmap) at ~5-11 % more wall time (D8); passing
  `refine=False` restores the Phase 6 coarse path bitwise.
- **Where refinement happens**: inside the per-phase loop of
  `_index_chunk`, i.e. `correlator.correlate(gln, refine=refine,
  emsphinx_compatible=...)` -- the exact `indexImage` wiring
  (`:230` `correlate(p, gln, ref)` -> `xc[p]->correlate(gln, r.qu,
  ref)`, `:326-331`): **every phase's one candidate is refined
  before insertion**, so with `P` phases there are `P` refinements
  per pattern and the top-`n_best` ordering uses refined scores.
  "Refine only the winner" is rejected as a deviation (it would
  reorder near-ties; the C++ refines per candidate). Fill rows are
  never refined (no candidate exists -- bug-injection entry).
- Failure semantics unchanged (Phase 6 D2): `_refine_peak` cannot
  raise in normal operation (its failure path returns the coarse
  triple with the analytic value -- C++ parity); guard (c) still
  catches a non-finite winning score, and the per-pattern
  `except Exception` catch stays. **One consequence is documented**
  (both public docstrings): a failed refinement's analytic value can
  be non-positive (D5), and the zero-seeded insertion rule then
  drops that candidate -- a pattern whose every phase fails that way
  becomes a failed pattern where the coarse path would have kept a
  positive interpolated score. Measured: zero refinement failures on
  every real-data run, so this is a contract statement, not an
  observed regression.
- **Score semantics under the new default are documented** in both
  public docstrings (the exhaustive list of falsified docstring
  blocks is in plan 2.3): a refined score is the analytic
  correlation at the Newton point divided by `denominator(zyz)`
  (`normalize=True`), not the tri-quadratic interpolated peak, so
  refined and coarse scores are **not comparable**; the refined
  normalised score can dip below the coarse one where the window
  chain rule is omitted (D6, measured 4/165); and the measured
  accuracy/score/memory numbers quoted in the docstrings are
  re-stated for the refined default (refined normalised
  0.5143-0.6347, un-normalised 0.2903-0.3592; small-map median
  0.505 / max 0.695 deg -- D10).
- The info message gains a line after `Correlation:`:
  `  Refinement: Newton (on)` / `  Refinement: off`, and the memory
  line prints the D3 model (`"54 MB"` at `bw` 68 with refine on).
- **Phase 6 test updates** (the default flip changes what a
  keywordless call measures): the coarse-value pins
  (scores 0.4963-0.6239, misorientations 0.599/0.838, determinism,
  `n_regions`/mask variants) get an explicit `refine=False`; the
  default-suite headline test asserts the *refined* bounds (D10)
  through the default call; `inspect.signature` default pins flip to
  `True`. Listed exhaustively in plan 2.4 -- **including every
  existing test the change breaks outside the three named modules**
  (the `_xcorr` error-model regression in
  `test_spherical_back_projection.py`, the kernel-name literals, the
  five refusal tests, the clone attribute-set pin, the docstring
  guards) -- no Phase 6 assertion is weakened, each is re-homed.

### D8 -- Performance and memory (recorded; measured on this machine)

- Warm single-thread refine cost (synthetic, converged 2-iteration
  case, includes the in-loop `dTablePre` rebuilds): `bw` 53:
  **1.45 / 0.33 ms** (`n_fold` 1 / `m-3m` flags); `bw` 68:
  **3.00 / 0.65 ms**; `bw` 88: **11.06 / 2.54 ms**. Components at
  `bw` 68: `dTablePre` 0.17 ms, `_derivatives(der=True)` 1.50 / 0.32
  ms, `der=False` 0.38 / 0.17 ms. The normalised `_denominator` adds
  two `der=False` calls (~0.34 ms at `bw` 68 `m-3m`).
- End-to-end per pattern (real data, `bw` 68, warm, 165-pattern run):
  coarse 13.17 ms + refine-and-denominator **1.39 ms** ->
  **ratio 1.11x**; observed ratios across runs 1.05-1.27x. Context,
  not a gate: the compiled C++ ratio is ~1.7x (Phase 4 D11: refine
  2.1 ms on a 6.9 ms coarse at `bw` 68 `m-3m`) -- our refine is
  *relatively* cheaper because the coarse path is slower. Refined
  throughput ~65-70 patterns/s/core at `bw` 68 -- the constitution's
  `>= 2` floor keeps a ~33x margin and is re-asserted with the new
  default (the floor test runs the *default*, now refined, path).
- Memory: D3 -- model `+ n_correlators * 16 bw^3` per worker when
  refining (`refine=True` or the `refine_patterns` path; the
  refine-only info message prints the refined model whatever the
  constructor flag, D9), factor
  triple 5.07 MB per process, `d_beta` reuse mandatory (allocation
  1.03 ms at `bw` 68 would rival the refine itself). No new
  transients (the Newton loop allocates nothing per iteration --
  `hes` goes into the Cholesky solve uncopied, D4).

### D9 -- `EBSD.refine_orientation_spherical` and `refine_patterns` (frozen)

- ```
  EBSD.refine_orientation_spherical(
      xmap: CrystalMap,
      harmonics, detector,
      bandwidth: int = 68,
      navigation_mask: np.ndarray | None = None,
      signal_mask: np.ndarray | None = None,
      normalize: bool = True,
      n_regions: int = 10, gaussian_background: bool = False,
      circular_mask: bool = False, emsphinx_compatible: bool = True,
      chunksize: int | None = None, verbose: int = 1,
  ) -> CrystalMap
  ```
  placed after `spherical_indexing`; `xmap` first mirrors
  `EBSD.refine_orientation`, the rest mirrors `spherical_indexing`
  (same checks, D5 of Phase 6 -- signal shape, the four
  `navigation_mask` checks against the **signal's** navigation
  shape, phases set/unique). No public `eps`
  (out-of-scope list). Builds a `SphericalIndexer(refine=True)` and
  calls `refine_patterns`. A documented `Notes` sentence records the
  two deliberate deviations from the
  `refine_orientation`/`refine_orientation_projection_center`
  convention: the parameter order is `(xmap, harmonics, detector,
  ...)` -- harmonics before detector, mirroring `spherical_indexing`,
  where the siblings take `(xmap, detector, master_pattern, ...)` --
  and there is no `compute`/`rechunk`/`chunk_kwargs` because the
  spherical pipeline is eager, exactly as `spherical_indexing` is
  (`chunksize` is the knob offered).
- **`SphericalIndexer.refine_patterns(patterns, zyz, phase_id, *,
  chunksize=None, progressbar=True) -> dict`** with keys `"zyz"
  (n, 3)`, `"scores" (n,)`, `"iq" (n,)`, `"phase_id" (n,)` (int32,
  echoed input) -- the refine-only work item (`idx.hpp:438-450`)
  with the intended semantics (D5 defect 1). The method **always
  refines**: it first builds and shares the factor triple exactly as
  a `refine=True` construction does, whatever the constructor's
  `refine` flag (D3), and its info message prints the **refined**
  memory model (D8). Per pattern, if
  `phase_id < 0` the input row passes through untouched (not-indexed
  points are never refined -- `is_indexed` is preserved); else
  preprocess -> unproject -> analyze (the Phase 6 guards (a)/(b)
  apply) -> the refinement on **that phase's** correlator
  (`refine(res.phase, ...)`, `:296`): `refine_zyz(gln, zyz_i)` on
  the per-phase normalised correlator when `normalize=True`, and
  the plain `refine_zyz(alm, gln, n_fold, mirror, zyz_i)` on the
  shared prototype with that phase's spectrum triple when
  `normalize=False` (the un-normalised branch is exercised by a
  named `normalize=False, refine=True` real-data test, D10) ->
  refined `zyz` + score, and
  `iq` recomputed (the C++ recomputes it, `:280`, `:304`). **Any
  failure -- guards, exceptions, and also a `_refine_peak`
  non-convergence is NOT one** (it returns the input triple with its
  analytic score, C++ parity) -- leaves the input row unchanged (the
  `idx.hpp:447-449` catch writes nothing). **Chunk alignment**
  (frozen -- `index_patterns` maps a *single* array, so the
  per-pattern rows must be block-aligned explicitly or a mis-aligned
  chunk silently refines from the wrong starts): `zyz` and
  `phase_id` are wrapped as
  `da.from_array(zyz, chunks=(patterns_da.chunks[0], (3,)))` and
  `da.from_array(phase_id, chunks=(patterns_da.chunks[0],))` and
  passed to `map_blocks` as aligned block arguments alongside the
  pattern blocks (corrected 2026-09-02, implementation: `map_blocks`
  cannot express this -- measured, it builds its `argpairs` as
  `tuple(range(a.ndim))[::-1]` and so aligns arrays on their
  *trailing* axes, handing every block the whole `(n, 3)` and `(n,)`
  arrays; the port uses the `dask.array.blockwise` call `map_blocks`
  itself makes, with the explicit indices `"ij"` out of
  `patterns "ikl"`, `zyz "im"`, `phase_id "i"`, and keeps the
  `da.from_array` chunking above verbatim -- see "Recorded results")
  (`_refine_chunk` packs `(nc, 6)` rows:
  `alpha, beta, gamma, score, phase_id, iq`); a bitwise
  chunksize/worker-count invariance test covers this path
  (validation).
- **CrystalMap in, CrystalMap out** (the alignment contract is
  kikuchipy's own, not a shape equality -- **orix returns only
  in-data rows** from `xmap.rotations`/`xmap.phase_id`, and a
  navigation-masked `spherical_indexing` map has full-length
  `is_in_data` but bounding-box-derived `xmap.shape`):
  - Compatibility:
    `_xmap_is_compatible_with_signal(xmap,
    am.navigation_axes[::-1], raise_if_not=True)`
    (`signals/util/_crystal_map.py:28-61`) -- the same `ValueError`
    message as `refine_orientation` plus its step-size warning.
    **A sparse-mask map is refused by this check**: probed on orix
    0.13.0, the Phase 6 large-subset recipe (`mask[::5, ::5] =
    False`) yields `xmap.shape` equal to the in-data bounding box,
    not the navigation shape, so such a map raises (documented; the
    supported route is to refine the full map and mask at refine
    time with `navigation_mask`). Then `phase_id` range: every
    non-negative id must index `harmonics`, else `ValueError`; then
    **phase identity**: for every phase id present among the points
    to refine, `_equal_phase(xmap.phases[id], harmonics[id].phase)`
    (`_crystal_map.py:65`) must hold, else a `ValueError` naming
    both phases and the differing attribute, as
    `refine_orientation` raises (`ebsd.py:3293-3299`) -- ids that
    merely *happen* to be in range must not silently refine against
    the wrong master pattern (a Hough/dictionary map, a re-ordered
    `PhaseList`).
  - Alignment: full-length (`n_all = prod(nav_shape)`) arrays are
    built by scattering through `xmap.is_in_data` --
    `zyz_full[xmap.is_in_data] =
    _euler.rotation_to_zyz(xmap.rotations)` (first column when
    `xmap.rotations_per_point > 1`: only the first is refined, the
    C++ "currently only uses a single result", `idx.hpp:440`),
    `phase_id_full` filled with `-1` then scattered likewise, and
    `is_indexed_full[xmap.is_in_data] = xmap.is_indexed` (the
    `_get_indexed_points_in_data_in_xmap` pattern,
    `_crystal_map.py:111-161`, re-built here because that helper is
    single-phase). The refined set is the full-length boolean
    `points_to_refine = xmap.is_in_data & ~navigation_mask.ravel()
    & is_indexed_full`; `patterns` and the start triples are sliced
    with it and the results scattered back at those positions.
    Rows outside `points_to_refine` keep the input map's values
    (`scores`/`iq` carried from the input `prop` where present,
    `0.0` fill otherwise -- documented).
  The starting triples are
  `_euler.rotation_to_zyz(xmap.rotations)` -- the inverse of
  the Phase 6 output conversion. **Measured round-trip robustness**:
  the stored quaternion hands back the *glide-equivalent* triple
  (`beta` flipped into `[0, pi]` where `correlate` returned
  `beta <= 0`); refining from it converges to the equivalent
  maximum -- refined-vs-refined misorientation **0.0 deg** and
  score difference `<= 2.92e-14` over the nine small-map patterns --
  so `refine_orientation_spherical` on a coarse map **equals
  `spherical_indexing(refine=True)` to tolerance, not bitwise**
  (different beta-sign path through the Wigner tables): frozen
  equivalence assertions `angle < 1e-4 deg`, `|score diff| < 1e-10`.
  The output map carries the refined rotations (one per point),
  `prop["scores"]` refined, `prop["iq"]` recomputed, `phase_id` and
  `is_in_data`/`is_indexed` from the input, `scan_unit` preserved;
  other input props are **not** carried (documented: `n_best`
  columns beyond the first are dropped -- a keep-1 map, like
  `refine_orientation`).
- **The docstring carries the Newton-is-local caveat** (D5): a
  starting orientation that did not come from `spherical_indexing`
  may converge to a spurious stationary point whose score is lower
  than the input's -- the method is score-monotone only for coarse
  maps from the same pipeline; a named test pins the behaviour
  (D10).
- Verbosity mirrors D6 of Phase 6, with the header verb
  parametrised -- `get_info_message(..., refining=True)` prints
  `Refining n orientation(s) in c chunk(s) ...` (never the
  `Indexing ...` line of a refine-only run) and the timing line
  reads `Refinement speed: x.xxxxx patterns/s`, matching
  kikuchipy's refinement wording
  (`_refinement/_refinement.py:111-117`); `verbose=0` silent.

### D10 -- Accuracy: measured values and frozen assertions

- **Synthetic, the `sht_xcorr.cpp` gates** (recipes of Phase 4 D10;
  seeded `default_rng`, deterministic): the C++ criteria are
  `eps = cbrt(float eps) = 4.92e-3 deg` (symmetry-free, `:294`),
  `epsN = 10 eps = 4.92e-2` (**the symmetry-free normalised loop
  only**, `:316-329`), `sqrt(eps) 5 = 0.351 deg` (the eight point
  groups, `:345` -- and the C++ applies this loosened gate to its
  **normalised point-group loop too**, `:371-391`, so a single
  `epsN` over all normalised cases would be a self-imposed
  tightening, not "the C++ criteria verbatim"). Measured refined
  recovery: **symmetry-free** worst **2.958e-06 deg** over `bw in
  {53, 54, 57, 60, 63, 64, 68, 88, 113, 123}` x 3 rotations (30
  cases; the C++ size list `:295-298` thinned -- padded sizes 54,
  57, 60, 64 kept, the adjacent duplicates 55/56/58/59/62 and the
  costly 158 dropped, 63 added (odd `slP` 125, the top of the C++
  point-group loop); every case 2 iterations, zero failures); **eight point groups** (`112, 11m, 2/m, 3, 4, 4/m, 6,
  6/m` at `bw in {53, 60, 63}` x 3) worst **4.518e-06 deg** (72
  cases, symmetry-reduced metric of Phase 4 D4); **normalised wedge**
  (`testNCorr` recipe, `bw in {53, 68}` x `{1, 4/m}` x 3),
  re-measured per subset: the `(n_fold=1, mirror=False)` cases worst
  **1.849e-02 deg**, the `4/m` cases worst **2.133e-02 deg** (the
  overall worst is a `4/m` case) -- mask-limited (the D6 chain-rule
  caveat), with the refined normalised score above the coarse one in
  all 12 cases.
  Frozen assertions: the C++ criteria verbatim, **split as the C++
  splits them** -- symmetry-free `< 4.92e-3` (1660x), point groups
  `< 0.351` (77,600x), normalised wedge `(1, F)` cases `< 4.92e-2`
  (`:316`; margin **2.7x**) and `4/m` cases `< 0.351` (`:345`;
  margin **16.5x**) -- an earlier draft gated every wedge case at
  `epsN` and reported a 2.3x margin, which was this spec's own
  tightening, not the C++'s; the split restores the ported gates
  exactly (mission success criterion 1). All cases seeded and
  deterministic; the per-case values are `record_property`ed.
- **Real data, `bw` 68, default configuration, `pc_average`**
  (coarse rows reproduce Phase 6 exactly):

  | data | coarse med/max | refined med / p90 / p95 / max | score mean | per-point delta |
  |---|---|---|---|---|
  | small (9) | 0.599 / 0.838 | **0.505** / 0.601 / 0.648 / **0.695** | 0.5701 -> **0.5886** | +0.0108..+0.0286, 9/9 up |
  | large 20-pt | 0.499 / 1.350 | **0.478** / 0.815 / 0.981 / **1.115** | 0.5678 -> 0.5813 | min +0.0020, 20/20 up |
  | large 165-pt (weekly) | 0.530 / 1.495 | **0.456** / 0.812 / **0.913** / **1.140** | 0.5684 -> 0.5815 | min **-4.8e-4**, 161/165 up |

  `eps = 0.001` changes nothing (digit-for-digit) -- the residual is
  systematic (mean-PC floor 0.33 + `bw`-limited signal + the window
  caveat), not convergence. At `bw` 88 the small map refines to
  median **0.450** / max **0.549** (recorded row -- more bandwidth,
  smaller residual).
- **`normalize=False, refine=True` (small map, measured)**: refined
  misorientations identical per pattern to the normalised run
  (median **0.505** / max **0.695** -- same coarse cells, and the
  Newton step maximises the *un-normalised* value in both paths);
  un-normalised scores mean 0.3220 -> **0.3324**, coarse
  [0.2799, 0.3533] -> refined [0.2903, 0.3592], per-point deltas
  +0.0059..+0.0164, **9/9 up**, iterations 8x2 + 1x3, zero
  failures. Frozen assertions (named test, validation): per-point
  score increase 9/9, refined mean `pytest.approx(0.332,
  rel=0.05)`, misorientations equal the normalised run's per point
  to `< 1e-4 deg`.
- **The roadmap's small-map `median < 0.5` is amended** (plan 0.1;
  measured 0.505): the bound predicted that refinement reaches the
  mean-PC floor, but the refined residual is
  `sqrt(0.505^2 - 0.33^2) ~ 0.38 deg` of `bw`-68 band-limitation +
  window caveat on top of the floor. Frozen small-map assertions:
  **all nine < 1.0 deg** (roadmap, measured max 0.695, 1.44x),
  **median < 0.75** (measured 0.505, ~1.5x margin -- the pin follows
  the Phase 6 margin convention, which put 1.2 on a measured 0.599;
  an earlier draft's 0.55 left 9 % on a floating-point pipeline
  against the CI lesson on cross-library-tight assertions, and the
  sharp discrimination lives in the per-point score-increase and
  refined-below-coarse-misorientation assertions, not the median
  pin), median and per-pattern values `record_property`;
  **per-point
  refined normalised score strictly above coarse on all nine**
  (measured min delta +0.0108) and **mean delta > 0.005** (2x under
  the measured +0.0185). Large 20-pt (default suite): **median <
  0.6, max < 2.0, all 20 deltas > 0**. Large 165-pt (weekly): the
  roadmap bounds **median < 0.6, p95 < 1.2, max < 2.0** (measured
  0.456 / 0.913 / 1.140 -- all pass) plus **>= 90 % of points with a
  score increase and mean delta > 0.005** (measured 97.6 %,
  +0.0131; the 4 decreasers are the window caveat, recorded).
- Near-degenerate and failure tests: D5 numbers frozen (`< 0.1` deg
  within `1e-3` rad of the poles; far starts return the start; the
  `beta = 0` exact-pole start converges through the 1x1 path).
  **One converged far-start decreaser is pinned by a named test**
  (D5): the seeded `bw` 24 case that moves **3.642 deg** and lands
  at un-normalised value **-27.293** from a start whose analytic
  value is **-19.936** (`record_property` both), killing any future
  "refinement can only raise the score" doc claim; a signal-level
  twin in `TestRefineOrientationSpherical` replaces one small-map
  rotation with a seeded random one and asserts the disjunction --
  the row passes through bitwise (failure path) *or* records a
  score at or below its input -- recording which branch ran.

### D11 -- Determinism, kernels, style (frozen)

- The refined path is deterministic and bitwise reproducible across
  chunking and thread counts (same code path per pattern, no shared
  mutable state -- `d_beta` is per-clone); the Phase 6 determinism
  tests run with the new default and a `refine=False` twin.
- Kernel flags: `_derivatives` is `@njit(cache=True, nogil=True,
  error_model="numpy")` -- the project's **third** sanctioned
  `error_model` kernel (constitution amendment, plan 0.5); the
  kernel-flag tests are updated from "exactly two" to "exactly
  three" and assert the flag on `_derivatives` and its absence on
  any other new function. `.py_func` coverage: the value against the
  compiled kernel to `rel 1e-12` and jac/hes to `abs 1e-9` (not
  bitwise -- FMA contraction differs; CI lesson), driven under
  `np.errstate` with NumPy-scalar inputs. **The pole branch is
  reachable through `py_func` only because D1 keeps the `csc` chain
  in NumPy scalars** (`np.cos`/`np.sqrt`; the compiled `_wrap_beta`
  boxes to a Python `float`, and a `math.*` transcription would
  raise `ZeroDivisionError` there regardless of `np.errstate` --
  measured, D1, with the `_preprocessing.py:418-429` precedent), so
  the `py_func` comparison includes a pole evaluation and the
  >= 95 % / 100 % coverage target needs no exclusion.
- `-n 0` warm-up before `-n 4` (new kernel -> new cache); no
  `scipy.fft` in any new code path (the Newton loop is FFT-free);
  no new `parallel=True`/`fastmath`; `workers=1` inherited.
- Licence: `_xcorr.py`'s delimited EMSphInx notice gains
  `derivatives()` `:889-1119`, `refinePeak()` `:442-499`,
  `NormalizedCorrelator::refinePeak`/`denominator` `:1169-1172`,
  `:1211-1225` and drops them from the not-ported list; the
  modification notice date extends ("changed by Johan Westraadt,
  2026-08, 2026-09"). `_indexer.py`'s notice moves
  `refineImage()`/`refine()` `:277-345` **into the ported list with
  the defect-1 note** ("the shipped refineImage discards the
  refinement result and stores a zero or stale score; this
  port implements the documented intent") and **removes the two
  now-stale bullets**: the "``refine=True`` refuses until it is
  implemented -- Phase 7" entry (`_indexer.py:61-64`) and the
  "refine-only work items ``msk[i] & 0x02``" not-ported entry
  (`:72-73`) -- Phase 7 ports exactly that item as
  `refine_patterns`, so the second bullet is rewritten as a
  *ported-with-documented-deviation* entry, keeping the `_xcorr.py`
  and `_indexer.py` halves of the notice symmetric.
  `signals/ebsd.py` glue stays kikuchipy-header-only.
- CHANGELOG (PR **#9**, fork links): one `Added` entry -- "Newton
  refinement of spherical-indexing orientations on the sphere:
  ``EBSD.refine_orientation_spherical()``, and ``refine=True`` --
  the new default, EMSphInx's own -- in ``EBSD.spherical_indexing()``
  / ``kikuchipy.indexing.SphericalIndexer`` (pass ``refine=False``
  for coarse-only indexing) (`#9 <https://github.com/
  jwestraadt/kikuchipy/pull/9>`_)". **No `Changed` entry**: the
  coarse-only default has never been released -- `EBSD.
  spherical_indexing` itself is an unreleased `Added` entry from
  PR #8 in the same `Unreleased` block (`CHANGELOG.rst:16-28`,
  checked), so a `Changed` entry would announce a change to a
  behaviour no release ever had. If a release ships between the two
  PRs this decision flips back to a `Changed` entry. No public
  docstring names roadmap phases (Phase 6 decision 14 carried
  forward).

## Context

- Algorithm reference: `specs/_research/explore-emsphinx-core-algorithm.md`
  sections 3.6-3.7 (with this spec's corrections: `maxIter` 15, the
  `prevMag2` unit mix, the dead `deg`, the value lag, the refineImage
  defect -- plan 0.6 adds addenda), 2.6, 3.8-3.9; section 8 gotchas
  4, 5, 24 and the new items of plan 0.6.
- C++ read for this spec: `sht_xcorr.hpp:128-139, 155-166, 184-196,
  219-226, 263-265, 361-383, 394-400, 442-499, 889-1119, 1140-1172,
  1182-1225`; `linalg.hpp:354-358, 411-431, 487-494`;
  `indexer.hpp:54-64, 207-345`; `idx.hpp:380-456`;
  `wigner.hpp:575-691, 814-852`; `test/sht/sht_xcorr.cpp:100-395`.
- Phase 1-6 deliverables composed: `_wigner.wigner_d_table_factors` /
  `_wigner_d_table_pre_kernel` (out= tripwire, per-thread buffer),
  the Phase 3 pinned `_phase7_derivatives` helper (copied into the
  new tests, per its docstring), `wigner_d_prime`/`wigner_d_prime2`
  (oracle), `_euler._wrap_beta`/`rotation_from_zyz`/`rotation_to_zyz`
  /`zyz_to_quaternion`, `_preprocessing._cholesky_solve_3x3`,
  `SphericalCrossCorrelator`/`NormalizedSphericalCrossCorrelator`
  (Phase 4 D1-D8; `interp_peak` returns `(zyz, peak, x)`),
  `SphericalBackProjector`/`_preprocess_pattern`, `SphericalIndexer`
  /`_index_chunk`/`EBSD.spherical_indexing` (Phase 6 D1-D9).
- Real data: as Phase 6 (small map + `nickel_ebsd_large` 20-pt
  default / 165-pt weekly, `pooch`-gated).
- CI lessons applied: no bitwise across compilers/libraries (the
  weight-association change of D1; `.py_func` tolerances); no
  knife-edge inputs (the degenerate tests pin `beta = 0` targets --
  an exact grid property -- and `1e-3` rad offsets, not
  float-boundary cases; the `+-pi` pole-slot assertion is weakened
  to NaN-or-huge because its exactness rides on the host libm's
  `cos(+-pi) == -1.0`, D2); the BLAS-guard precedent -- the 3x3 solve
  reuses Phase 5's `_cholesky_solve_3x3`, never `np.linalg.solve`;
  orix used only for `angle_with` metrics; no tight timing bounds
  (the `>= 2` floor only); deterministic seeds; `-n 4` after `-n 0`;
  `error_model="numpy"` only on `_derivatives` (D11 rationale).
