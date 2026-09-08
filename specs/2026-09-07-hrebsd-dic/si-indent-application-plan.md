# Plan: Si-indent HREBSD replication tutorial (full-resolution only)

## Context

Johan supplied Winkelmann et al. 2025 (Ultramicroscopy 276, 114180,
"Simulation-based super-resolution EBSD...") and commissioned: "use this file
to create a new tutorial and replicate HREBSD of the full resolution data
only. Leave the super resolution as a follow up." This executes plan open
question 13 of specs/2026-09-07-hrebsd-dic/ (recorded 2026-09-07) with a
sharper scope: apply our HREBSD-DIC chain to the paper's own Zenodo Si-indent
dataset at native 622x512 pattern resolution and compare against the paper's
published results; their simulation-supersampling method ("parameter
super-resolution") is deferred. The data file is LOCAL:
`C:\Users\westraadt.1\Repos\kikuchipy\AGH__Si_indent_1_512x672.h5oina`
(18.9 GB, repo root, untracked). User decisions (2026-09-08): data on this
laptop; deviatoric closure only (strict replication of their isochoric
relative deviatoric maps; stress joins the super-resolution follow-up).

This is the first application of the completed HREBSD-DIC feature (Stages
A-C, commits 8f5b6b11..31e1eb79) to real deformed data, and the first
dataset where a METHOD-LEVEL noise floor can be scored against published
numbers: the paper's Table 1 gives sigma_eps = 0.079 mm/m and sigma_theta =
0.043 mrad at exactly this pattern resolution, measured by a neighbor-pair
estimator (their eqs 17-18) on strain-free row 10 -- directly computable on
our output.

## Verified facts (read-only probes + exploration, 2026-09-08)

FILE (h5py): scan group '1'; Processed Patterns (58500, 512, 622) uint8,
HDF5 chunks (1,512,622), lzf; map 250x234 at 0.2 um; per-point Pattern
Center X/Y + Detector Distance (paper's affine PC calibration; PCx span
0.0021 ~ 1.3 px); Tilt Angle 75.0 deg; detector tilt 6.979 deg from the
Detector Orientation Euler; EBSD/Data/Euler (58500,3) present; Processed
Static Background present; AND the paper's own MapSweeper output at
1/Data Processing/Pattern Matching/Data: Strain (58500, 6) f32 (their
full-res refinement: Refinement Binning=None, Use Calibration=1) + Cross
Correlation Coefficient -- a NUMERICAL comparison target. Their Voigt
order/frame/signs are undocumented: the notebook determines and pins them
honestly (against the record's strain-map PNG panel signs / Fig 5), and
both fields are reference-differenced at the same point before comparison
(the paper's own section 3.5 bias argument).

READER (oxford_h5ebsd): kp.load lazy works (dask, per-pattern chunks);
detector arrives with per-point pc (234,250,3), convention='oxford',
sample_tilt=75.0, tilt=6.979. THREE TRAPS: (a) binning silently unread on
this file (get_binning version-boundary bug, _api.py:228-231) -> set
detector.binning = 2 manually + record the reader bug; (b) xmap returned
EMPTY (reader TODO) though Euler exists -> build the CrystalMap from the
file's Euler + Si phase; (c) px_size stays 1.0 placeholder -> harmless for
hrebsd_dic (per-point-PC route bypasses the beam-scan model,
_geometry.py:236-249), handled per design for hrebsd_pc_shift.

ENGINE: rectangular patterns unit-pinned ((37,61)) but this is the FIRST
end-to-end non-square EBSD.hrebsd_dic run -> add a small rectangular
end-to-end regression test + validation entry first. Lazy input pinned
bitwise. reference='auto' would stream-read 18.6 GB -> pass an explicit
(row, col) = the record's X10Y10 (convention resolved in-notebook).
Capture range (measured): 2.0 deg with default filter_cutoffs=(0.05,None),
4.0 deg + 10x accuracy with (None,None); indent rotations reach ~50 mrad =
2.9 deg -> the cutoffs choice is made by MEASUREMENT in the notebook
(Table-1-style noise floor on row 10 under both settings, stated pick
rule), not assumed. Runtime projection: 6.71 pat/s at 480x480 real data,
x1.38 pixels -> ~3.5-5 pat/s -> full 58500-point map ~3.5-5 h on 8
workers. Memory fine (~25 MB/reference x 8). DISK: only 22 GB free ->
everything stays lazy, rechunk on load, no eager stack, no copies.

CONVENTIONS: local-file gate copied from pseudo_symmetry.ipynb's Ti section
(env-var + filename probe, dependent cells under `if path:` +
nbval-ignore-output, info admonition: Zenodo 14059950, 18.9 GB, CC-BY-4.0,
no auto-download; plus a repo-root fallback probe since the file lives
there now). roadmap.md HREBSD path fully ticked -> NEW section with fresh
gate boxes. validation.md ledger ends at entry 77 -> Si-indent entries
start at 78 under a new dated heading. .gitignore gains *.h5oina BEFORE any
commit. CHANGELOG Added line (fork-only wording). Branch policy unchanged:
hrebsd-dic only, push, no PR, never merge to develop.

## Design additions (source-verified by the design pass)

- `hrebsd_pc_shift`'s per-point-PC route consumes only the detector's own
  PCs in pixels + the homography/reference/converged props; px_size never
  read there -> the placeholder is harmless for the WHOLE chain here.
- `hrebsd_gnd` requires `xmap.scan_unit` in {m, nm, um} + `grain_id` -> the
  tutorial-built CrystalMap carries 0.2 um x/y grids and scan_unit="um".
- Reference resolved EMPIRICALLY: their Cross Correlation Coefficient is
  exactly 1 at their reference (self-correlation) -> read, locate, use;
  the X10Y10 PNG name becomes a cross-check, not an assumption.
- K3 distortion-type map INCLUDED (their eq 10, a cheap invariant of our
  deviatoric strain) for panel-for-panel Fig-5 fidelity.
- Strain output is sample-frame Voigt (11,22,33,23,13,12), TENSOR shears;
  rotation_vector in radians -> mrad conversion in the notebook.

## Steps

1. **Housekeeping commit (first)**: `.gitignore` += `*.h5oina`;
   spec bookkeeping: plan.md OQ13 dated scope note (full-res only,
   super-resolution deferred, 512-rows-x-622-cols correction), roadmap.md
   new section "## Si-indent application (plan open question 13)" with 6
   fresh gate boxes, validation.md new dated heading (entries 78+).
2. **Tests + reader fix commit**: (a) rectangular END-TO-END regression
   test in tests/test_signals/test_ebsd_hrebsd_dic.py (small synthetic
   non-square-pattern map, per-point-PC detector, explicit reference;
   pins: runs through the public method, lazy==eager bitwise, finite
   props, flows through all four analysis functions, an asymmetric-warp
   pin that dies if width/height transpose); (b) FIX the oxford_h5ebsd
   binning-read bug (_api.py:228-231 accepts both 'Camera Mode' and
   'Camera Binning Mode'; two-line fix + dedicated test on a modified
   copy of the shipped asset; contained to the never-merged branch; the
   notebook still carries a set-then-assert binning cell); validation
   entries record both.
3. **Pre-flight script** (scratchpad, never in repo): lazy load + detector
   + xmap build, CCC reference resolution, row-10 A/B of
   filter_cutoffs (0.05,None) vs (None,None) with the paper's eq-17/18
   neighbor-pair estimator, a ~20x20 sub-map run, measured patterns/s.
   De-risks geometry, the first full-scale non-square run, and the
   convention determination BEFORE the one-shot execute.
4. **Author the notebook** `doc/tutorials/hrebsd_si_indent.ipynb` (26
   cells; hidden first cell; gate = KIKUCHIPY_LOCAL_DATA_DIR env var +
   repo-root fallback probe; every dependent cell under `if si_path:` +
   nbval-ignore-output; info admonition with Zenodo/18.9 GB/CC-BY-4.0/no
   auto-download). Sections: load + fixes (rechunk ~32 patterns/chunk,
   binning assert, px_size note) -> CrystalMap from the file's Euler + Si
   phase -> sanity panel -> MEASURED filter_cutoffs decision (pick rule
   stated BEFORE the numbers: the map needs >=3 deg capture so
   (None,None) unless catastrophically noisier, penalty reported) ->
   CCC-based reference -> THE RUN (full 250x234 map, ~3.5-5 h, honest
   non-converged count near crater) -> chain (deviatoric strain,
   rotations mrad, HR-KAM, GND with b=3.84e-10 m, PC-shift residuals
   against their affine calibration) -> Fig-5-style gallery (e11..e33 at
   +-15 mm/m, omega at +-50 mrad, K2-style norm, K3; NaN crater shown,
   thumbnail tag) -> numerical comparison vs THEIR Strain field
   (reference-difference both; convention determined by
   permutation+sign correlation with a stated acceptance threshold and a
   pre-registered invariant-only fallback; difference maps + robust
   statistics) -> noise-floor table vs their Table 1 (ours 1x1 vs their
   3x3, labeled; sigma_eps vs 0.079 mm/m, sigma_theta vs 0.043 mrad; GND
   floor vs the 4-8e12 m^-2 literature scale at 0.2 um; throughput vs
   their ~20 pat/s on 2x RTX4090) -> limitations + super-resolution
   follow-up pointer -> references. Wiring: index.rst nbgallery,
   run_nbval.sh, sanitize cfg only if an ungated cell needs it (design
   goal: none), CHANGELOG Added line.
5. **Adversarial review BEFORE the execute** (theory: every to-be-printed
   derivation vs the paper, both-fields differencing, estimator formula;
   conventions: gate/tags/wiring/no-em-dashes) -> fixer. Reviewing first
   because a post-execute code fix costs a 4-6 h re-run.
6. **Execute once** (venv nbconvert, timeout=-1, ~4-6 h dominated by the
   map cell; overnight-friendly). Post-execute review verifies every
   markdown claim against actual printed numbers (markdown-only fixes
   need no re-execution; a code fix triggers full re-execute, accepted).
7. **Gates + close**: local nbval of the full NOTEBOOKS list (passes
   without the data file by construction); ruff on touched .py; full
   suite untouched-check; validation entries for each gate; roadmap boxes
   ticked; signed commits pushed to origin/hrebsd-dic (third commit:
   notebook + wiring + CHANGELOG + spec closes). NO PR, develop
   untouched. Estimated end-to-end: ~12-17 h, one working day plus an
   overnight execute.

## Verification

- Rectangular regression test + binning-fix test green before the big
  run; full suite otherwise untouched (known flaky simulator test only).
- Pre-flight measured rate replaces the 3.5-5 pat/s projection before the
  one-shot execute is scheduled.
- Executed notebook: convergence counts printed; noise floors reported
  honestly WHATEVER they are (no massaging toward Table 1); the
  their-Strain convention verdict printed as a table or the
  invariant-only fallback engaged and reported as a finding.
- nbval passes with and (by tag construction) without the data file;
  18.9 GB file still untracked AND now ignored; develop (local+remote)
  untouched; all commits signed on hrebsd-dic and pushed.
