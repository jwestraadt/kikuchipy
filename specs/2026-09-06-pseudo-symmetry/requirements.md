# Phase 8 -- `spherical-pseudo-symmetry`: requirements

Branch `spherical-pseudo-symmetry` (roadmap Phase 8, deferred
2026-09-02, un-deferred after the Phase 11 merge). Spec folder
`specs/2026-09-06-pseudo-symmetry/` (the folder name drops the
`spherical-` prefix of the branch; the roadmap heading is amended to
name the folder explicitly -- plan 0.1; renaming instead is open
question 6 for the user). The phase ports EMSphInx pseudo-symmetry:
operator *prediction* (the `MasterXcorr` program),
pseudo-symmetry-aware *indexing* (the `indexImage()` psym loop), and
the *psymfile* angle-file codec that connects them to `IndexEBSD`.
The same option on the *refinement-only* path is provisionally OUT of
scope pending open question 1 for the user (plan 9.1).

**Drafting caveat (D11):** nothing was executed while drafting -- a
~1h compute job owns this machine's working tree via the editable
install, so no probe scripts, no pytest, no binaries were run. Unlike
prior specs, the decisions below therefore carry **no fresh measured
numbers**; every tolerance is marked "measured-then-pinned" (MTP) and
is filled at the failing-tests/implementation gates per the
tech-stack assertion convention. Numbers that ARE quoted come from
prior phases' recorded measurements or from static line-verified
source reading (both cited). All EMSphInx file:line references are @
60f3517 unless a branch is named.

**Drafted 2026-09-07 with the adversarial review folded in:** the
drafting review (26 findings, two blockers) ran against the drafting
decisions and the two research reports before any spec file existed;
these three documents were then drafted fresh with every finding
applied at drafting time (disposition appendix in plan.md).
Re-submission of the three documents to adversarial review is a
definition-of-done gate (validation.md).

## Scope

In scope:

- **`src/kikuchipy/indexing/_spherical/_pseudo_symmetry.py`** (new):
  the `MasterXcorr` port `find_pseudo_symmetry_operators` (single-
  and two-phase modes, optional correlation volume), the result
  carrier `PseudoSymmetryOperators` (with psymfile `save()`), and
  the psymfile codec `read_emsphinx_psym_file` /
  `write_emsphinx_psym_file` (D2, D3, D6). Public exports per D10.
- **`pseudo_symmetry_ops` in the spherical indexing loop**: the
  `indexer.hpp:241-261` psym loop ported into
  `_spherical/_indexer.py` `_index_chunk`, threaded through
  `SphericalIndexer` and `EBSD.spherical_indexing`, with the winner
  variant reported as crystal-map prop `"pseudo_symmetry_index"`
  mirroring the NCC `refine_orientation` contract (D2, D4).
  Packed-row contract amendment: indexing rows widen 6 -> 7; the
  module constant is split so refine-only rows stay 6 (recorded,
  D4).
- **`MasterPatternHarmonics.rotate`** -- the stale Phase 3 stub
  (`_master_pattern_harmonics.py:2093-2114`) is implemented via
  `_wigner.rotate_harmonics`; Phase 8's synthetic pseudo-symmetric
  masters are its natural first consumer (D7). The pinned
  `NotImplementedError` message test is replaced. Recorded scope
  growth on a Phase 2 class: the roadmap box is amended (plan 0.1).
- **Binary cross-checks (local-gated)**: `MasterXcorr.exe` stdout
  parity for the prediction path; `IndexEBSD.exe` psymfile
  acceptance/error/inertness pins for the codec and for the D1
  baseline evidence (D1, D8). No shipped `.npz` psym reference (D8).
- **Tests**: new `tests/test_indexing/test_spherical_pseudo_symmetry.py`
  plus additions to `test_spherical_indexer.py`,
  `test_spherical_master_pattern_harmonics.py` and
  `tests/test_signals/test_ebsd_spherical_indexing.py` (D9,
  validation.md). Coverage 100 % of touched spherical modules.
- **Docs**: CHANGELOG entry (user-facing -- required), the contracted
  pseudo-symmetry `##` section appended to
  `doc/tutorials/spherical_indexing.ipynb` before "What's next?"
  (tutorial spec `2026-09-03-spherical-indexing-tutorial/
  requirements.md:903-906`), sequenced behind the user's uncommitted
  notebook edits (D10, open question 8 for the user, plan 9.8).
- **Constitution amendments** listed in plan section 0.

Out of scope (confirmed):

- File writers for `pseudo_sym.h5`/`pseudo_sym.xdmf` and the
  `true.svg`/`pseudo.svg` diagrams (the four hard-coded CWD writers
  of `master_xcorr.cpp`, D3): the volume is returned as an ndarray;
  an XDMF/ParaView recipe is a tutorial note only. orix owns
  point-group plotting.
- The **Phase 9 visualisation half** stays separately deferred
  (roadmap:98): sht2png equivalents, `plot_power_spectrum`,
  `.plot()` conveniences, `SphericalBackProjector.plot`, the generic
  xcorr volume plot, Sphinx-Gallery example. **Provisionally (open
  question 3 for the user) the detected-operator stereogram plot
  defers with it**: the roadmap Phase 8 box's "optional volume +
  stereogram" wording is amended per the resolution (plan 0.1);
  Phase 8's own optional extra is then only the returned correlation
  volume ndarray.
- **Provisionally (open question 1 for the user):
  `pseudo_symmetry_ops` on `EBSD.refine_orientation_spherical` /
  `SphericalIndexer.refine_patterns`.** EMSphInx's refine-only work
  item never consults `pSym` (`idx.hpp:438-450`), so this would be a
  kikuchipy-native extension mirroring NCC's `refine_orientation`.
  Until answered: refine-only rows stay width 6 (D4) and the D2
  cross-engine killer runs through the existing NCC
  `refine_orientation` plus `EBSD.spherical_indexing` (D9.3).
- **Multi-phase pseudo-symmetry**: a `ValueError` mirrors EMSphInx's
  single-phase-only throw (`idx.hpp:190-192`) -- per-phase operator
  lists are a possible later extension, not this phase (D5).
- Namelist work: `psym_file` incl. the double-`ipath` quirk shipped
  in Phase 9 (`_namelist.py:702, 833, 858, 919-923, 1007,
  1328-1333`). Phase 8 adds only the angle-file codec (D6).
- A shipped psym regression `.npz` (Phase 10's matrix deliberately
  excludes psym, roadmap:11-12; no shippable pseudo-symmetric input
  exists in the data registry; and the shipped binary cannot produce
  variant ground truth -- D1/D8).
- EMSphInx's ranked `"Scan i"`/`"Match i"` HDF5 output layout and its
  first-map-images-on-every-scan defect (`idx.hpp:347-368`,
  `:350-353`): the crystal-map prop replaces it (recorded deviation,
  D4).
- The generic EMsoft `AngleFile` reader (eu/om/ax/ro/ho/cu types):
  only the `qu` subset pseudo-symmetry accepts (`master.hpp:225`).
- Any upstream (EMSphInx) fix or PR; new runtime dependencies; GPU
  paths; new benchmarks (the per-op refine cost is recorded in
  validation, not benchmarked -- Phase 7 baseline covers the kernel).

## Decisions

### D1 -- IndexEBSD.exe psymfile inertness pin (frozen)

Reference baseline, recorded: at `master` 60f3517 (and upstream
release 87b2387) the shipped CLI's pseudo-symmetry is **silently
inert**: `Indexer::pSym` is only ever declared, reserved,
placeholder-pushed and read (`indexer.hpp:74, 175, 178, 228, 243`),
nothing wires `MasterData::pSm` into it, and `clone()` (`:120`)
rebuilds without it. Setting `psymfile` loads operators and allocates
extra output scans (`idx.hpp:249-250`) but the loop iterates empty
lists. **The reference semantics for this phase are therefore: the
literal loop of `indexer.hpp:228-261` at 60f3517, plus the two-line
wiring that exists only on the user's own `feature/GPU` branch**
(`feature/GPU:include/modality/ebsd/idx.hpp:386` --
`idx->pSym[i] = phases[i].pseudoSym()` -- and
`feature/GPU:include/idx/indexer.hpp:244` -- `clone()` copies `pSym`;
author Johan Westraadt, Mar 2026; scoring logic there is
byte-identical to master). There is **no executable upstream ground
truth** for the indexing psym loop; the loop's port is validated by
kikuchipy-internal oracles (D2, D9), and the *inertness itself* is
pinned by the local-gated binary test below as the executable
evidence for this decision. An optional manual validation against a
locally rebuilt `feature/GPU` CPU binary is open question 2 for the
user (D8.5).

The inertness pin, corrected ground truth (this corrects the research
report -- see Context):

- After the empty psym loop, `indexImage` converts EVERY result row
  zyz -> quaternion (`indexer.hpp:264-269`), so the untouched
  placeholder `{0,0,0,0}` is read as ZYZ (0,0,0) -> the identity
  quaternion (`rotations.hpp:973-989`), multiplied by the identity
  `quNp` (`detector.hpp:455-458`) and conjugated; `ebsdWorkItem`
  casts phase -1 to `uint_fast8_t` = 255 (`idx.hpp:413-418`). The
  validation pin asserts, on the `Scan i/EBSD/Data` datasets
  (i >= 2): **Phase = 255 (uint8), Phi1/Phi/Phi2 = the Bunge Euler
  of the identity, Metric = 0, IQ = 0** -- NOT the raw
  `corr=0/phase=-1/qu=0` placeholder values.
- Never assert on the per-scan `IPF Map`/`XC Map`/`IQ Map` image
  datasets -- the shipped save() bug fills all scans' images from
  Scan 1 (`idx.hpp:350-353`).
- "Scan 1 bitwise-equal with/without psymfile" holds at the
  **dataset level only** (exclude the `EMheader`
  StartTime/StopTime/PatPerS fields, `idx.hpp:318-333`); with one
  operator `padNum` stays 1 so the group name "Scan 1" is unchanged
  (`idx.hpp:348`).
- The with-psymfile namelist must leave `ipath` empty (or use
  CWD-relative paths): `nml.hpp:247` prepends `ipath` to `patFile` a
  SECOND time when `psymfile` is set, and `pSymFile` itself resolves
  relative to the CWD.
- Grounds for the Scan-1-identical claim: `pSym` is only
  declared/reserved/placeholder-pushed/read (`indexer.hpp:74, 175,
  178, 228, 243`); `clone()` (`:120`) rebuilds without it; and the
  larger result-list `n` cannot change the winner slot.

### D2 -- Operator convention and the recorded derivation (frozen)

The single most dangerous convention issue; the C++ author's own
comment concedes the composition "may be wrong here ... q may need to
be conjugated depending on how they are input"
(`indexer.hpp:244-247`). Frozen resolution:

1. **The public kikuchipy convention is the NCC convention**, already
   frozen by `refine_orientation(..., pseudo_symmetry_ops=...)`:
   variants are `[rot] + [op_i * rot]` in orix left-composition
   (`_refinement/_refinement.py:971-975`), every variant refined
   independently, winner by argmax score, reported in prop
   `"pseudo_symmetry_index"` (int32; 0 = the original map point,
   `i` = `ops[i-1]`, 1-based), created only when ops were passed
   (`_refinement.py:125-126, 328-329`; oracle
   `tests/test_indexing/test_ebsd_refinement.py:617-670`). The
   spherical API mirrors every element of this: parameter name, orix
   `Rotation` type with internal `.flatten()`, prop name/dtype/
   semantics, argmax winner.
2. **Derivation reconciling EMSphInx** (line-verified, not yet
   machine-verified -- D11; carried in full HERE because the
   session research reports do not survive): kikuchipy's stored map
   rotation is
   `R = rotation_from_zyz(zyz) = ~Rotation(zyz_to_quaternion(zyz))`
   (sample->crystal; tech-stack Euler bullet, tech-stack.md:28,
   pinned by measurement in Phase 5). EMSphInx's pre-conversion best
   is `q0 = zyz2qu(zyz) = ~R` (crystal->sample), and its variant is
   `qp = q0 * q_file` (`indexer.hpp:242-249`), whose converted map
   rotation is `~qp = (~q_file) * R` (Hamilton algebra:
   `~(q0 * q) = (~q) * R`). So **EMSphInx composes the conjugate of
   the psymfile quaternion where the NCC convention composes the op
   itself**: `op == ~q_file`.
3. **MasterXcorr closes the loop**: it prints
   `q_print = zyz2qu(zyz_peak)` (`master_xcorr.cpp:145`, no
   conjugation, no north-pole correction). The crystal-frame
   invariance at an autocorrelation peak `zyz_peak` is
   `S = rotation_from_zyz(zyz_peak) = ~q_print` (the peak satisfies
   `f((~R_q) n) = f(n)`; crystal-side operators compose on the LEFT
   of the map rotation -- tech-stack Euler bullet). Feeding the
   printed quaternion into the `q0 * q` loop therefore realises the
   variant `(~q_print) * R = S * R` -- exactly the NCC composition
   with `op = S`. **The literal upstream chain is self-consistent**,
   and the port keeps it: no behavioural divergence, only the
   conjugation placed explicitly at the psymfile boundary.
4. **Frozen mappings** (the conjugation lives in exactly two code
   places, the codec and `find_`):
   - `find_pseudo_symmetry_operators` returns
     `op_i = rotation_from_zyz(zyz_peak_i)` -- directly usable as
     `pseudo_symmetry_ops` for BOTH the NCC methods and the
     spherical methods;
   - `write_emsphinx_psym_file(path, ops)` emits rows
     `q_file = (~op).data` (w x y z);
   - `read_emsphinx_psym_file(path)` returns `~Rotation(rows)`.
   Round trip `read(write(ops)) == ops` exactly.
5. **Epistemic status of the WRITE direction, recorded**: the
   shipped binary's psym loop is inert (D1), so the write direction
   is **analytically derived and empirically unverifiable** against
   `IndexEBSD.exe` at 60f3517; all "IndexEBSD parity" language in
   this spec is scoped to inertness + error paths only. (The C++
   author's own doubt, `indexer.hpp:244-247`, is quoted above.) The
   optional `feature/GPU` CPU rebuild that could verify it
   end-to-end is open question 2 for the user.
6. **Why tests can be conjugation-blind, and which are not**
   (recorded): autocorrelation is inversion-symmetric
   (`xc(g) = xc(g^-1)`), so operator SETS from a single master are
   closed under inversion and set-level assertions cannot detect a
   conjugation error. The designated killers use non-involutory ops
   asserted by EXACT index, never inversion-closed pairs without
   index assertions (`test_ebsd_refinement.py:617-670` is the
   pattern -- exact indices `[2, ...]`/`[..., 1, ...]`):
   (a) a **golden literal file-content test**: writing one known
   non-involutory op produces the specific `w x y z` digits of
   `~op`, the digits derived from this section and pinned
   bytes-exact (the writer's number format is frozen in D6 to make
   this well-defined);
   (b) the **NCC cross-engine perturbation oracle** run through a
   psymfile round trip with a single non-involutory op (a ~25 deg
   rotation about a low-symmetry axis: wrong conjugation flips the
   winning index from 1 to 0) -- D9.3;
   (c) the **two-phase mode**, which is NOT inversion-closed, pinned
   via the Phase 6 composed-orientation identity
   `rotation_from_zyz(zyz_b) * O_B == O_A` (measured 0.68/1.07 deg,
   roadmap Phase 6 box) -- D3.9/D8;
   (d) **`MasterXcorr.exe` raw-quaternion parity** (D8.1) asserting
   kikuchipy's returned ops equal the **conjugate** of the printed
   rows (`ops = rotation_from_zyz(zyz_peak) =
   ~zyz_to_quaternion(zyz_peak)` while the binary prints
   unconjugated `zyz2qu`).
7. **Parser facts for the parity tests** (corrects the research
   report -- see Context): the binary prints the intensity at
   `setprecision(4)` but the quaternion via `Quat::to_string(6)`
   (`quaternion.hpp:274, 508-524`) -- fixed precision 6 with a
   leading alignment space per non-negative component. The parity
   parser must tolerate double spaces; the quaternion comparison is
   pinned at ~1e-6 (a 1e-4 tolerance would blunt exactly this
   conjugation killer).

### D3 -- `find_pseudo_symmetry_operators`: the MasterXcorr port fidelity list (frozen)

Signature (public, `kp.indexing`):

```python
def find_pseudo_symmetry_operators(
    harmonics: MasterPatternHarmonics,
    second_harmonics: MasterPatternHarmonics | None = None,
    *,
    bandwidth: int = 88,
    cutoff: float = 0.5,
    exclude_symmetry: bool = True,
    keep_volume: bool = False,
    emsphinx_compatible: bool = True,
) -> PseudoSymmetryOperators
```

Algorithm -- the `programs/master_xcorr.cpp` port (lines 41-281),
step-labelled with its constants kept literal, **each constant with
its stage and metric pinned**:

1. Inputs: operate on DC-removed copies (`remove_dc()`, the
   `master_xcorr.cpp:75-76` `removeDC` on both spectra); the
   caller's objects are never modified. If a harmonics' bandwidth
   differs from `bandwidth`, `resize` a copy (documented note:
   MasterXcorr *computes* at `bw` from the EMsoft h5 -- resize of a
   higher-bandwidth `.sht` is a recorded non-equivalence,
   `_indexer.py:264-273`; the D8 parity tests construct
   at-bandwidth via `from_master_pattern` for exactly this reason).
   Default `bandwidth=88` = the smallest of the C++'s recommended
   list (`master_xcorr.cpp:46`; `2*88-1 = 175 = 5^2*7` is fast).
   The C++ CLI's hard `[53, 313]` clamp is NOT reproduced (CLI
   sanity bound; recorded deviation): input validation reuses the
   module's existing `_BANDWIDTH_LIMITS = (16, 512)` rule
   (`_indexer.py:332`). `cutoff` outside `[0, 1]` raises
   `ValueError` at call time (EMSphInx errors at parse time; same
   outcome, earlier stage recorded). The kikuchipy default
   `cutoff=0.5` is a recorded kikuchipy choice -- open question 4
   for the user; no test depends on the default.
2. Correlator: the **un-normalised** `SphericalCrossCorrelator(bw)`
   (`master_xcorr.cpp:80`), full cube via `compute(flm, gln,
   n_fold, mirror)` using **only the first master's** symmetry
   flags (`master_xcorr.cpp:82`, comment line 50). Single-phase
   mode: `gln = flm` (auto-correlation).
3. `v_max` -- **the seeds are requirements**: every reported
   intensity divides by the `v_max` refined from the exact seed
   below, so 4-decimal intensity parity depends on reproducing
   them. Single-phase (auto) mode: refine (un-normalised
   `refine_zyz`, eps 0.01, the Phase 7 port of `refinePeak`) from
   the near-identity seed -- **the C++ `idxIdent =
   (bw-1)*sl*sl + (bw/2)*sl + bw/2` cell translated to the
   equivalent nearest-to-identity cell on the actual
   `(bwP, slP, slP)` grid** (they coincide whenever
   `slP == 2*bw - 1`, true for every recommended bandwidth).
   Two-file mode: refine from the **sub-pixel interpolated coarse
   argmax** instead (`master_xcorr.cpp:87-91`).
4. Candidate voxels: `xc >= v_max * cutoff * 0.95` (`factor =
   0.95`, the candidate gate for "off grid peaks",
   `master_xcorr.cpp:61, 104`) AND `>=` all 26 neighbours of the
   3x3x3 periodic neighbourhood via `_extract_neighborhood` with
   the `emsphinx_compatible` glide semantics
   (`master_xcorr.cpp:113-140`; ties kept, `>=`). **The scan runs
   over the true `(bwP, slP, slP)` cube returned by `compute()`**
   -- the C++ flat-indexes with `sl = 2*bw - 1` while the
   correlator's true side is `slP = fastSize(2*bw - 1)`
   (`master_xcorr.cpp:86, 108-111, 201` vs
   `sht_xcorr.hpp:376-382`), silently mis-indexed whenever they
   differ; this defect is NOT reproduced (recorded deviation; equal
   for all recommended bandwidths, so parity is unaffected there).
5. Dedup: nearest kept GRID maximum by
   `angle = acos(min(1, |q_i . q_j|))` in degrees; `< 2.0` deg
   keeps only the brighter (`master_xcorr.cpp:148-166`). **The
   metric is the quaternion-dot half-angle -- ~4 deg of
   misorientation -- applied to grid maxima PRE-refinement only**;
   kept literal ("extremely arbitrary" per the C++ comment).
6. Refine each survivor from its UN-interpolated grid euler with
   the un-normalised `refine_zyz`; intensity := refined value /
   `v_max`; re-sort descending; keep `intensity >= cutoff`
   (`master_xcorr.cpp:173-195`). **After refinement there is NO
   second dedup**: two grid maxima refining into the same true peak
   both survive (documented; the D4 duplicate-rows consequence
   follows from this).
7. Returned-list semantics per mode: `exclude_symmetry=False`
   reproduces the raw printed-list semantics (true-symmetry ops
   included, surviving refined duplicates included -- the D8 parity
   setting). `exclude_symmetry=True` (kikuchipy default -- a
   recorded deviation from the C++ stdout list) adopts the
   SVG-stage filters for the returned list: drop every operator
   with `|q . q_sym| > 0.999` against the PROPER rotations of the
   first harmonics' point group, and dedup kept ops against each
   other at the same threshold (`master_xcorr.cpp:232-261`; in C++
   these filters run only for `pseudo.svg`, never stdout). **The
   `0.999` cosine is ~2.56 deg of quaternion-dot half-angle, ~5.1
   deg of misorientation.** Ni (m-3m): `True` returns an empty set;
   `False` returns the identity plus the proper-Oh peaks (D9.1).
8. Conversion (D2): `operators = rotation_from_zyz(refined zyz)`
   per peak, ordered by descending intensity.
9. Two-phase mode (`second_harmonics` given): cross-master
   misorientation peaks; NOT inversion-closed. **Frozen framing of
   the quaternion maps: the peak rotation is the one applied to the
   SECOND master to match the first, per the
   `s2Corr.correlate(p1, p2, ...)` argument order
   (`master_xcorr.cpp:82`), and the cube is folded by the FIRST
   master's mirror/nFold only (`:50, :82`).** `op` means "the
   phase-1-equivalent of a phase-2 orientation is `op * O_2`"
   (derivation in D2.3 applied to `f2 = f1 o op`); exclusion still
   against the FIRST phase's proper rotations (C++ uses
   `mp1.pointGroup()`, `master_xcorr.cpp:232-234`).

Angle-space statement (frozen so neither drafter nor implementer
halves/doubles them): BOTH thresholds above -- the 2-deg dedup and
the 0.999-cosine exclusion -- are **quaternion-dot half-angles**; the
misorientation-angle equivalents are ~4 deg and ~5.1 deg
respectively.

Dropped, recorded: the `[53, 313]` CLI clamp (step 1; D8 oracle runs
still respect it), the four hard-coded CWD writers, and the
`fabs(nFld - nFld)` dead-branch SVG classification
(`master_xcorr.cpp:267`; with the stereogram provisionally deferred
per open question 3, no classifier ships this phase).

Return: `PseudoSymmetryOperators`, a small frozen dataclass --
`operators: Rotation` (NCC convention, descending intensity),
`intensities: np.ndarray` (float64, normalised by `v_max`),
`volume: np.ndarray | None` (`(bwP, slP, slP)` float64, only when
`keep_volume=True`; provisional pending open question 3),
`bandwidth: int`; method `save(filename)` (=
`write_emsphinx_psym_file`, D6; ValueError on an empty operator
set). No `plot()` this phase (provisionally deferred with the Phase
9 visualisation half, open question 3).

### D4 -- Indexing-loop semantics and the result-row contract amendment (frozen)

- Parameter `pseudo_symmetry_ops: Rotation | None = None` (NCC name
  and type; flattened internally), on the `SphericalIndexer` ctor
  (keyword-only group, after `refine` -- EMSphInx holds `pSym` on
  the Indexer, `indexer.hpp:74`) and on `EBSD.spherical_indexing`
  **appended after `emsphinx_compatible`, before `chunksize`** (no
  existing positional call breaks; recorded placement deviation
  from the NCC siblings' mid-signature position).
- Loop semantics, ported from `indexer.hpp:228-261` under the D1
  baseline: per phase, (1) the base candidate honours the run's
  `refine` flag exactly as today; (2) `q0` is THAT PHASE's best
  (post-optional-refine) orientation; (3) for each op, the variant
  map rotation is `op * rotation_from_zyz(zyz_best)`, seeded as its
  zyz and **ALWAYS Newton-refined** via `refine_zyz` regardless of
  `refine` (`indexer.hpp:252` vs 230 -- the asymmetry is preserved;
  the user's CUDA port preserved it too); the seed zyz must equal
  the C++ chain `qu2zyz(q0 * q)` to 1e-14 (unit-pinned); (4) each
  variant carries the pattern's iq and goes through the SAME
  positive-score `upper_bound` insertion (`_insert_candidate`;
  strictly-beats, `score <= 0` never recorded -- machinery already
  ported, provenance comment `_indexer.py:77-81`). With
  `refine=False` the ranked list mixes an interpolated coarse base
  score with analytic variant scores -- EMSphInx's own metric
  inconsistency, preserved and documented in the Notes (the
  existing coarse-vs-analytic non-comparability note,
  `ebsd.py:2143-2152`, is extended).
- Refinement failure inherits Phase 7 semantics: the failed variant
  reports the correlation at its seed (`sht_xcorr.hpp:494-498`); no
  extra flagging/discarding (non-positive scores are dropped by
  insertion anyway).
- **Packed-row amendment, split (recorded)**: indexing rows grow
  6 -> 7 with a new trailing variant-index column; **refine-only
  rows stay 6**. The module constant `_ROW_WIDTH` (=6, used by
  `refine_patterns`' chunk worker at `_indexer.py:731`) is split
  into `_ROW_WIDTH_INDEX = 7` and `_ROW_WIDTH_REFINE = 6` (names
  final at drafting). Indexing row: `(alpha, beta, gamma, score,
  phase_id, iq, psym_index)`; width 7 applies unconditionally on
  the indexing path (one shape through `_map_chunks`/unpacking);
  the prop is created only when ops were passed.
  `refine_patterns`/`refine_orientation_spherical` contracts are
  untouched (provisional on open question 1; a "yes" amends the
  refine contract to 7 instead, with the same 7th-column
  semantics).
- **7th-column semantics**: a float column carrying a small
  integer; **fill value -1.0** on fill rows and failed patterns
  (0 would collide with "original variant won"). Consumers may
  equivalently gate on `phase_id == -1`, but the fill is -1
  regardless. Full fill row: `(0, 0, 0, 0, -1, 0, -1)`.
- **Prop layout (frozen here -- NCC has no `n_best` precedent)**:
  `n_best == 1` -> 1-D int32 prop `"pseudo_symmetry_index"` (exact
  NCC mirror: 0 = base, `i` = `ops[i-1]`, 1-based, -1 on
  fill/failed); `n_best > 1` -> the winner-only 1-D
  `"pseudo_symmetry_index"` **plus** a 2-D int32 per-rank prop
  `"nbest_pseudo_symmetry_index"` beside `nbest_phase_id`
  (`ebsd.py:2343-2353` precedent). EMSphInx never records which op
  won (rank-sorted `"Scan i"` output only, `idx.hpp:406-418`) --
  the props are the recorded, deliberate deviation (exactly what
  the NCC sibling already promises users).
- **Amended fill-row doc**: with P phases and N ops there are up to
  `P*(1+N)` candidates; the frozen sentence "rows >= P of an
  n_best > P request keep the fill" becomes "rows >= P*(1+N) of an
  n_best > P*(1+N) request keep the fill".
- **Duplicate rows documented + pinned**: multiple variants
  routinely Newton-converge into the SAME peak (ops near a true
  symmetry op, strong signal), so `n_best > 1` output can hold
  near-duplicate orientations distinguished only by variant
  index -- EMSphInx behaves identically; the docs say it and a test
  pins it.
- **Tie rule**: the strictly-beats `upper_bound` insertion keeps
  variant 0 (the base orientation) ahead on exact ties, matching
  NCC's argmax-first (`_solvers.py:240`); written down here and
  covered by a mutation-list entry (plan 7.2).
- Frozen wording to amend (targets enumerated): the module doc
  "result contract"/"one candidate per phase" blocks
  (`_indexer.py:166-209`), `index_patterns` docs (`:1509-1512`),
  `EBSD.spherical_indexing` Notes (`ebsd.py:2040-2044`,
  `:2162-2171`). Frozen replacement wording for the
  one-candidate sentence: "one candidate per phase, plus one per
  pseudo-symmetry operator for the single phase when
  `pseudo_symmetry_ops` is given (up to `1 + n_ops` candidates);
  secondary peaks of one phase are still not extracted".
- Memory/timing: each op adds one `refine_zyz` per pattern (Phase 7
  warm baseline 1.45/0.33 ms at bw 53, 3.00/0.65 at 68 (`n_fold`
  1 / m-3m)); no new per-worker buffers (refine buffers reused).
  Measured-then-recorded in the Notes at build (D11).

### D5 -- Multi-phase x ops (frozen)

`ValueError` when `pseudo_symmetry_ops` is given with more than one
phase, mirroring EMSphInx's hard throw (`idx.hpp:190-192`). Frozen
message: `"pseudo_symmetry_ops currently requires a single phase"`
(the C++'s "psuedo-symmetry" typo and exact wording are NOT
reproduced; public messages follow kikuchipy style -- recorded).

### D6 -- psymfile codec and guard semantics (frozen)

`read_emsphinx_psym_file(filename) -> Rotation` and
`write_emsphinx_psym_file(filename, operators: Rotation)` in
`_pseudo_symmetry.py`, named after the `write_emsphinx_patterns`
precedent and the Phase 9 namelist field `psym_file`.

- Format (EMsoft angle file, qu subset -- `master.hpp:220-233`,
  `xtal/vendor/emsoft.hpp:48-145`): first token `qu` (ONLY -- any
  other type token raises ValueError, mirroring the `master.hpp:225`
  throw), second token the count, then `count * 4` numbers,
  whitespace- and/or comma-separated (the C++ ctype treats `,` as
  whitespace), **w x y z scalar-first**, EMsoft pijk=+1 passive
  convention (`constants.hpp:64-66`).
- Read: rows exactly equal to `(1, 0, 0, 0)` are silently skipped --
  EXACT float equality, ported literally (`master.hpp:227-231`;
  near-identity rows are kept and merely re-converge to the global
  peak -- no tolerance added, behaviour documented). Too few / too
  many numbers raise (the C++'s "orientions" typo not reproduced).
  No dedup, no symmetry check on load (parity). **Recorded
  deviation**: orix `Rotation` silently unitizes non-unit rows on
  construction where EMSphInx uses the raw values. Returned ops are
  the **conjugated** quaternions per D2.4 (`~Rotation(rows)`), i.e.
  NCC-convention operators ready for any `pseudo_symmetry_ops=`.
- Write: header `qu`, count, one `w x y z` row per op. **The number
  format is frozen** (full float64 `repr` precision, single-space
  separated, one row per line) so the D2.6a golden bytes-exact test
  is well-defined. (MasterXcorr's 4-decimal stdout is lossier --
  there is NO C++ writer to imitate; `master_xcorr.cpp` prints to
  stdout only.) Rows are `(~op).data` per D2.4, so the file drives
  EMSphInx's `q0 * q` loop to the same variants kikuchipy scores
  with `op * rot`. Zero operators: ValueError.
- Round trip `read(write(ops))` reproduces `ops` exactly (modulo the
  identity-skip rule, documented).

Guards, frozen:

1. A size-0 `Rotation` passed as `pseudo_symmetry_ops` is
   None-equivalent (no prop, no variants) -- stated in the docs.
2. A user-supplied identity op is refined wastefully and can
   spuriously report index > 0 on numerical noise -- the
   identity-skip is CODEC-level only (reader); the indexer does not
   warn (parity with NCC: `_refinement.py:971-975` refines the
   identity too); hazard documented.
3. Ops equivalent to a true symmetry op of the phase guarantee
   near-ties -- `pseudo_symmetry_index` flips between 0 and i on
   noise; hazard documented, no indexer-side filtering (the
   finder's `exclude_symmetry=True` default already removes them at
   the source).
4. An all-identity psymfile -> empty op list -> behaves exactly as
   `ops=None`.
5. `cutoff`/bandwidth validation per D3 step 1.

### D7 -- `MasterPatternHarmonics.rotate` (frozen contract)

The Phase 3 stub (`_master_pattern_harmonics.py:2093-2114`,
`NotImplementedError`) is implemented:
`rotate(rotation: Rotation) -> MasterPatternHarmonics` returns the
harmonics of the master pattern **actively rotated** by `rotation`:
`g(n) = f((~rotation) * n)` -- a feature at direction `n0` moves to
`rotation * n0`. Implementation via the fully tested
`_wigner.rotate_harmonics` (the zyz plumbing must satisfy the frozen
identity `rotate_harmonics(f, zyz)` gives `g(n) = f((~R) n)` with
`R = Rotation(zyz_to_quaternion(zyz))` -- tech-stack Wigner bullet,
tech-stack.md:29); the composition identity
`h.rotate(r1).rotate(r2) == h.rotate(r2 * r1)` and a synthesis
oracle (rotate-then-`to_master_pattern` vs sampling the unrotated
synthesis at rotated directions) pin the direction.

**Symmetry-flag rule, frozen (uniform)**: the returned object's
symmetry flags are neutralized -- its phase is replaced so that
`n_fold == 1` and `has_equatorial_mirror == False` (the flags derive
from the phase, `_master_pattern_harmonics.py:1516-1534`, and a
rotation about a non-z axis falsifies them; rotations about z would
preserve them, but the uniform rule is simpler and safe). Tests that
need folding reassign the phase explicitly.

Removes the `NotImplementedError` whose public message embeds
"Phase 3 (sht-wigner-d)" (`:2093-2114`) -- the decision-6.14
violation (no roadmap phase numbers in public messages) fixed
incidentally. The pinned stub-message test lives in
`specs/2026-08-16-sht-master-spectra-and-file/validation.md:53`; the
recorded pin is updated where it lives (plan 0.5).

`rotate` is public API growth on a Phase 2 class not in the Phase 8
roadmap box: plan 0.1 records the roadmap amendment, and `rotate`
gets its own docstring-convention freeze and tests beyond being a
fixture factory (identity no-op, composition, equivalence vs
`rotate_harmonics`, flag-neutralization, round trip, synthesis
direction). Consumers this phase: the D9.2 synthetic
pseudo-symmetric blends.

### D8 -- Oracle discipline for the binary cross-checks (recorded)

All local-gated on `KIKUCHIPY_EMSPHINX_DIR` (the conftest
`emsphinx_program` fixture), each invocation holding the machine-wide
lock `<tmp>/kikuchipy-emsphinx-program.lock`; binary determinism only
"given a fixed fftw.wisdom" (roadmap:101). **None of these run at
drafting** (D11).

- **Bandwidth discipline**: every `MasterXcorr.exe` oracle run pins
  a bandwidth satisfying `fast_size(2*bw - 1) == 2*bw - 1`, and the
  test states the check (safe: the recommended CLI list 88, 95,
  113, ...; also bw 68 since `135 = 27*5`; UNSAFE e.g. 64 -> 127
  prime, 96 -> 191 prime -- at those the exe mis-indexes its own
  cube, D3.4, and the oracle is garbage). Oracle runs also respect
  the binary's `[53, 313]` clamp even though kikuchipy drops it.
- **CWD isolation**: every exe run gets an isolated working
  directory (the exe hard-writes
  `pseudo_sym.h5`/`.xdmf`/`true.svg`/`pseudo.svg` into the CWD with
  `H5F_ACC_TRUNC`, `master_xcorr.cpp:60, 202, 206, 230, 280`) and
  takes the existing machine-wide binary lock.

1. **`MasterXcorr.exe` parity** (the prediction oracle; gated +
   weekly + pooch): the exe is present in `build/Release` (listed
   2026-09-06). Input: the full EMsoft `ebsd_master_pattern("ni")`
   h5 (0.3 GB, cached locally per tech-stack.md:50 -- MasterXcorr
   reads EMsoft h5, NOT `.sht`, `master_xcorr.cpp:69-70`). Run
   `MasterXcorr <bw> <cutoff> <ni.h5>` at bw 88 in a temp cwd;
   parse the stdout `intensity w x y z` rows per the D2.7 precision
   facts (double-space-tolerant parser). Compare against
   `find_pseudo_symmetry_operators(from_master_pattern(mp,
   bandwidth=88, ...), cutoff=same, exclude_symmetry=False)` -- the
   at-bandwidth construction is the parity route (D3.1).
   Assertions: kikuchipy's returned ops equal the **conjugate** of
   the printed rows at ~1e-6 (D2.6d/D2.7), the intensity vector
   matches within an MTP band, and for Ni both engines find the
   identity + proper-Oh set. This pins the whole un-conjugated
   prediction space.
   **AMENDED 2026-09-07 (Stage A binary measurement, validation.md
   Recorded results)**: the exe's auto-mode `v_max` comes from the
   identity-cell-seeded refine, which STALLS at 74 % of the true
   peak on the Ni h5 at bw 88 (measured `v_max = 0.719309`), so
   printed intensities EXCEED 1 (22 rows at cutoff 0.9: 7 at
   1.3521, 15 at 1.2082 -- proper Oh minus identity minus one
   folded C2'; one further edge row 0.8653 appears at cutoff 0.5).
   The drafted "identity + proper-Oh set with intensities ~1.0"
   expectation is withdrawn for the exe route; the parity test pins
   the measured rows, and the kikuchipy-side values stay MTP at the
   implementation gate (a faithful port of the same seeding is
   expected to reproduce the stall).
2. **Two-phase (cross-master) oracle runs**: the two-file branch is
   the one mode that can expose a conjugation/argument-order error
   the autocorrelation killers cannot (D2.6c). Two runs:
   (i) **mandatory local-gated discriminator** -- the ni h5 passed
   under two different PATH SPELLINGS of the same file (exercises
   the two-file branch: argmax seeding instead of the identity
   cell; no new data needed). **AMENDED 2026-09-07**: the drafted
   "same path twice" run is vacuous -- the auto/two-file branch is
   selected by FILENAME STRING equality (`master_xcorr.cpp:87`), so
   identical spellings are byte-identical to auto mode; two
   spellings of one file route through the two-file branch
   (measured: `v_max = 0.972597` argmax-seeded, top intensities
   1.0000/0.8936/0.6400 -- pinned in the gated test);
   (ii) a **true two-master parity run** against
   `ebsd_master_pattern("al")` (weekly, download-gated, ~0.3 GB
   cached per tech-stack.md:50; skip cleanly without pooch) -- if
   the user prefers not to add the download (open question 7), the
   recorded-gap alternative in validation.md replaces it.
3. **`IndexEBSD.exe` psymfile pins** (gated, cheap; the Phase 9/10
   small-scenario input route reused): (a) **inertness** exactly
   per D1 -- one-op psymfile run exits 0, writes `Scan 1` and
   `Scan 2`; `Scan 2/EBSD/Data` carries Phase = 255, the
   identity-Euler triplet, Metric = 0, IQ = 0; `Scan 1`
   dataset-level equal to the no-psymfile run's `Scan 1` excluding
   the `EMheader` time fields; image maps never asserted;
   `ipath` empty; (b) **error paths** -- an `eu`-type psymfile, a
   count-mismatched psymfile, and a two-master + psymfile namelist
   all fail (exit != 0), pinning the compat claims of D6 and D5
   against the real binary.
4. **No shipped psym `.npz`**: Phase 10's matrix deliberately
   excludes psym; there is no shippable pseudo-symmetric input; and
   the shipped binary cannot produce variant ground truth (D1).
   Parity lives in the gated tests plus the kikuchipy-internal
   oracles.
5. Optional, manual only (open question 2): rebuild `feature/GPU`'s
   CPU `IndexEBSD` and run a psymfile scenario as a live-loop
   oracle (validation.md "Manual"; user's call; not a gate).

### D9 -- Test-data strategy and mechanism tests, de-vacuoused (recorded)

No phase with genuine pseudo-symmetry is shipped or downloadable
(`kp.data` masters: ni, al, si, austenite, ferrite, steels, al6mn,
alpha_almnsi -- no hcp; `_data.py:593-607`). The strategy, per the
roadmap Phase 8 test box, is layered, the limitation documented, and
**no test can pass on an empty return**:

1. **Ni mechanism tests** (in-package `.sht` + small master, default
   suite). **Positive-count pin**: with `exclude_symmetry=False` at
   a stated cutoff, `find_pseudo_symmetry_operators` on the Ni
   master autocorrelation must return the identity plus (close to)
   the 24 proper Oh rotations (Phase 4 measured exactly this cube,
   roadmap.md:66), with a per-op angular tolerance and intensities
   measured-then-pinned (execution-gated placeholders).
   **AMENDED 2026-09-07**: the drafted "intensities ~1.0 with
   identity == 1.0" expectation is withdrawn -- the 2026-09-07
   `MasterXcorr.exe` measurement (validation.md Recorded results)
   shows intensities are normalised by the identity-seeded refine's
   stalled `v_max` (74 % of the true peak on Ni at bw 88), so
   values above 1 are the faithful expectation; the kikuchipy-side
   pins stay MTP at the implementation gate. The
   subset-of-Oh and `exclude_symmetry=True`-empty tests remain but
   are sequenced after the count pin so neither is vacuous. Wrong
   op (30 deg z-rotation, not in Oh) ->
   `pseudo_symmetry_index == 0` everywhere on `nickel_ebsd_small`;
   a true Oh op -> variant ties the base (MTP near-equality,
   winner index in {0, i}; base expected on exact ties --
   `upper_bound` semantics, D4).
2. **Synthetic pseudo-symmetric masters** via D7's `rotate`:
   `g = f + lam * f.rotate(S)` blends (lam < 1) with S a 120 deg
   (3-fold) resp. 60 deg (6-fold) rotation **about +z only**
   (matching the roadmap's wording; z-rotations commute with the
   claimed `C_n^z`/`sigma_h` so inherited flags stay true), AND the
   blend object's flags are neutralized to
   `n_fold=1, mirror=False` anyway (belt and braces: the correlator
   folds the cube by the first master's flags,
   `master_xcorr.cpp:82`, so stale flags would corrupt the very
   cube the test scans). Construction: a test-local helper adding
   coefficient arrays (`f + lam * rotate(f, S)` through the private
   constructor -- no public addition API ships). Tests assert the
   specific expected operator IS found (S or ~S: the
   autocorrelation pair) within refinement tolerance with an MTP
   intensity band, and pass an **explicit cutoff well below the
   expected ~0.5 relative intensity** (e.g. 0.25) so the
   `>= v_max*cutoff*0.95` gate and the `>= cutoff` keep-gate never
   make the flagship test razor-edge flaky (decoupled from the
   default-cutoff open question 4). An off-grid S (a z-angle off
   the euler grid) pins the 0.95 search factor.
3. **The NCC cross-engine perturbation oracle** (the conjugation
   killer, D2.6b), restated for the provisional scope: perturb the
   stored `nickel_ebsd_small` xmap by a non-involutory op, write +
   read the psymfile, and recover the exact 1-based index through
   BOTH `EBSD.refine_orientation` (the existing NCC engine -- an
   independent detector-space implementation; ops already
   supported) and `EBSD.spherical_indexing(...,
   pseudo_symmetry_ops=...)`; the comparison is on
   `pseudo_symmetry_index` and the misorientation of the winners
   (the `test_ebsd_refinement.py:617-670` scheme).
4. **Targeted dedup case**: two synthetic peaks ~3 deg apart in
   quaternion-dot space -- one kept, the brighter wins; the
   mutation list (plan 7.2) swaps the half-angle/misorientation
   conventions to prove the test bites.
5. **A "rescue" indexing scenario** (build-measured): index a
   fixed-seed noisy/band-limited pattern set from a synthetic blend
   master such that the coarse search demonstrably lands in the
   pseudo basin without ops (pinned), and `pseudo_symmetry_ops`
   recovers the true orientation with `pseudo_symmetry_index != 0`
   and a higher winning score. Candidate levers: lam near 1, low
   bandwidth, seeded noise. **If no deterministic construction
   survives measurement, the fallback is recorded**: the
   winner-index path is then pinned at unit level (forced candidate
   sequences through the insertion + variant loop) plus the D9.3
   oracle, and the limitation is documented in validation.md (D11
   gate).
6. **Local hcp/TiAl masters, skip-if-absent** (roadmap box 2): a
   directory of user-supplied EMsoft master h5 files named by a new
   env var **`KIKUCHIPY_LOCAL_MASTERS_DIR`** (the
   `KIKUCHIPY_EMSPHINX_DIR` naming precedent; `KIKUCHIPY_DATA_DIR`
   is the pooch cache redirect, `test_data.py:285`, and is not
   reused for foreign files; the constitution note names the new
   var so future phases reuse it, plan 0.4). Tests glob for
   documented candidate names and `pytest.skip` with a reason
   naming the exact expected filenames. **Which files the user has
   is open question 5**; the tests ship dormant either way.

### D10 -- Placement, exports, licences, CHANGELOG, tutorial (frozen)

- New file `src/kikuchipy/indexing/_spherical/_pseudo_symmetry.py`:
  kikuchipy GPL header + the verbatim CMU/Lenthe third-party block
  (the `_master_pattern.py:20-57` / `_indexer.py` model) naming the
  ported ranges -- `programs/master_xcorr.cpp` lines 41-281,
  `include/idx/master.hpp` lines 220-233, `include/xtal/vendor/
  emsoft.hpp` lines 48-145 (qu subset) -- plus the "Changed by Johan
  Westraadt, 2026-09-<dd>" modification notice and the `LenthePS`
  paper citation (J. Appl. Cryst. 52 (2019) 1157,
  doi 10.1107/S1600576719011233; already in
  `doc/user/bibliography.bib` from Phase 0, so cite only).
- `_indexer.py`: the psym loop lands here; its provenance comment
  (`:74-81`) is rewritten -- the pseudo-symmetry bullet moves from
  the "deliberately not ported" list (now false) to a ported note
  naming `indexer.hpp:241-261` and this spec folder; roadmap phase
  numbers stay out of public docstrings (decision 6.14). The
  third-party block gains the `indexer.hpp:241-261` range.
- Public exports (4 new names, sorted into `indexing/__init__.pyi`
  `__all__` -- 15 names today, `indexing/__init__.pyi:38-54`; the
  numpydoc API reference regenerates from `__all__`):
  `PseudoSymmetryOperators`, `find_pseudo_symmetry_operators`,
  `read_emsphinx_psym_file`, `write_emsphinx_psym_file`.
- Changed files: `_indexer.py`, `_master_pattern_harmonics.py`,
  `signals/ebsd.py`, `indexing/__init__.pyi`, `CHANGELOG.rst`,
  `doc/tutorials/spherical_indexing.ipynb` (+ its nbval/sanitize
  wiring only if the new cells need it), tests. No `_namelist.py`
  change.
- Numba: any new kernel (the local-maxima scan, if not vectorised)
  follows `@njit(cache=True, nogil=True)`, gets a `.py_func` test,
  and must NOT add a fourth `error_model="numpy"` without a
  recorded reason + the kernel-flag test update (tech-stack rule,
  tech-stack.md:44).
- CHANGELOG (required -- user-facing): one `Added` entry naming
  `find_pseudo_symmetry_operators`, the psymfile codec,
  `pseudo_symmetry_ops` on `SphericalIndexer` /
  `EBSD.spherical_indexing`, and `MasterPatternHarmonics.rotate`,
  with the fork PR link (#13).
- Tutorial: the contracted `##` pseudo-symmetry section before
  "What's next?" + stored outputs per the notebook rules. The
  notebook has **uncommitted local user edits** (git status,
  2026-09-06) and tech-stack forbids sweeping them -- the notebook
  commit is sequenced behind the user resolving those edits (open
  question 8 for the user, plan 9.8); if unresolved at PR time, the
  tutorial section ships in an immediate follow-up commit on the
  same branch/PR after the user's go-ahead, and the CHANGELOG entry
  still lands with the code. Plan 0.2 additionally puts the
  notebook on the constitution's never-sweep list.

### D11 -- Tolerances, measurement deferral, and performance (recorded)

A ~1h compute job owned this machine throughout drafting (editable
install; timings contamination-sensitive), so -- deviating from every
prior phase's spec -- **no drafting probes ran**: no tolerance below
is measured yet, no binary was invoked, and the derivations of D2/D7
are line-verified only. Gates: (1) the failing-tests commit carries
placeholder bands marked `# MEASURED-THEN-PINNED`
(`pytest.approx(measured, rel=0.05)` convention), xfail/skip where a
value is required to even run; (2) the implementation gate replaces
every placeholder with a measured value (recorded, dated, in
validation.md "Recorded results" with the recipe); (3) the
adversarial review re-measures the decision-critical ones (the D2
conjugation oracle, the MasterXcorr parity tolerance, the D9.5
rescue scenario). Any decision refuted by measurement is amended in
this file with the dated correction, the Phase 10 precedent.

Performance requirements:

- (a) A recorded **ops-scaling baseline** in validation.md: per-op
  cost ~ one Newton refinement per op per phase per pattern
  (~3.0 ms at bw 68 `n_fold` 1 / ~0.65 ms m-3m per Phase 7's warm
  baseline -- each op adds roughly 25 % of a coarse correlate),
  measured at 0/1/2/4 ops.
- (b) The `>= 2 pat/s/core` hard floor is **scoped to `ops=None`**;
  a psym run at a stated op count (2 ops) is re-measured and
  recorded.
- (c) `BatchEstimate` knows nothing of per-op cost -- recorded
  deviation with chunksize guidance in the Notes; no model change.
- (d) The `EBSD.spherical_indexing` Notes' refined-ratio claims
  (1.05-1.27x) gain a caveat that ops are not covered, plus a new
  measurement entry.

## Context

- Constitution: `specs/mission.md:23` (the deliverable line names
  `find_pseudo_symmetry_operators` + psymfile read/write +
  stereogram plot -- D3/D6 deliver the first two; the stereogram is
  provisionally deferred per open question 3, in which case the
  mission line is amended, plan 0.7); `specs/tech-stack.md` (the
  Euler/rotation contract the D2 derivation stands on,
  tech-stack.md:28-29; the result contract amended per D4; the
  assertions convention); `specs/roadmap.md:91-93` (the Phase 8
  boxes this spec implements; re-scope note :5-13).
- Research: the two 2026-09-06 scratchpad reports
  (`phase8_emsphinx_research.md` -- EMSphInx behaviour, line-cited;
  `phase8_fork_research.md` -- fork constraints). **The scratchpad
  does not survive the session, so every load-bearing citation is
  carried inline in D1-D9 above.** Two errors in the EMSphInx
  report are corrected here: (1) its section 5.2 stated the extra
  scans hold raw `corr=0/phase=-1/qu=0` -- the on-disk values are
  Phase = 255, the identity Bunge Euler, Metric = 0, IQ = 0 (D1);
  (2) it implied 4-decimal stdout quaternions -- the intensity is
  `setprecision(4)` but the quaternion is `Quat::to_string(6)`
  (D2.7). The reports' addenda enter `specs/_research/` per plan
  0.6. Also: `specs/_research/explore-emsphinx-core-algorithm.md`
  section 8 (gotcha checklist -- review gate);
  `explore-emsphinx-programs-and-formats.md:528-551` (MasterXcorr).
- Phase deliverables composed: Phase 7 `refine_zyz` (both
  correlators; the hook shipped for exactly this, roadmap:11),
  Phase 4 cube/`compute`/`_extract_neighborhood`/`index_to_euler`,
  Phase 3 `rotate_harmonics` + `_euler`, Phase 2
  `MasterPatternHarmonics`, Phase 6 indexer + insertion machinery,
  Phase 9 namelist `psym_file`, Phase 9/10 binary-test conventions
  (env var, lock, wisdom caveat).
- NCC mirror source of truth: `_refinement/_refinement.py:971-975,
  125-126, 328-329`, `_solvers.py:239-247`, `ebsd.py:2801-2809`,
  `tests/test_indexing/test_ebsd_refinement.py:617-670`.
- EMSphInx defects catalogued for this phase (reproduce-or-record):
  inert `pSym` wiring (D1); the save() image-map bug -- every
  scan's `IPF/XC/IQ Map` images filled from Scan 1
  (`idx.hpp:350-353`, D1); `nml.hpp:247` double-`ipath` (already
  preserved, Phase 9; its psym face: `pSymFile` never
  `ipath`-prefixed, CWD resolution); `master_xcorr.cpp:267`
  `fabs(nFld - nFld)` dead branch (not implemented this phase --
  the stereogram is deferred, D3); `:44` stale usage string
  (CLI-only, N/A); the `sl` vs `slP` scan (not reproduced, D3.4);
  rank-only output with first-map images on every scan (not
  ported, D4); refine-only `refineImage` discard (already fixed in
  the port, Phase 7).
