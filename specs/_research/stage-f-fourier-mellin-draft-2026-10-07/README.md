# Stage F draft (parked 2026-10-07): Fourier-Mellin rotation initial guess

Status (2026-10-07): REVISED, awaiting splice after the Stage E commit.
All 45 critic findings are dispositioned (20 accepted, 24 already
applied, 1 rejected; plan_section12.md section 12.5). The plan gate
stays PENDING Johan's review. Nothing here is part of the spec yet;
nothing is built.

History: parked mid-review at about 02:50 on 2026-10-07, when Johan
had to power down; the revise step was re-run the same day.

Johan's instructions in force: build Stage E (the GPU backend) to the end,
then finish this Stage F SPEC (splice it into
`specs/2026-09-07-hrebsd-dic/` and commit it) and STOP before any Stage F
failing tests or code. The Stage F plan gate is PENDING his review (his
2026-10-06 waiver no longer covers Stage F).

## What is here

- `requirements_D22.md`, `validation_V10.md`, `plan_section12.md`,
  `roadmap_stageF.md`: the insertable spec blocks (D22, V10 with ledger
  entries numbered from 200 to be renumbered after the Stage E ledger,
  plan section 12, roadmap Stage F). Drafted on Opus 5.5 xhigh from three
  explorations (method and literature, codebase seams, a CPU prototype)
  and revised against the critics; ledger entries now run 200 to 216
  (216 is the spec-review re-measurement of the frozen recipe).
- `critic_findings.md`: the 45 findings of the two adversarial critics
  (14 major, 31 minor, no blocker), dispositioned in the table of
  `plan_section12.md` section 12.5.
- `fm_method_literature.md`: the method report (Ernould et al. 2020, the
  thesis, classic Fourier-Mellin registration, EBSD physics).
- `prototype_fm_lib.py.txt`: the throwaway CPU prototype estimator, kept
  as reference (renamed so no tool collects it).

## Headline measurements (CPU prototype, see validation_V10.md)

- Rotation-only FM seed: every synthetic case from 0 to 30 deg converges to
  the exact-seed answer; the translation-only seed's capture is 2.0 deg.
- Cost: about 66-83 ms per gated pattern on one CPU thread (shared
  machine, about +-30 per cent).
- The 180-degree ambiguity must be resolved by the NEAREST rule; "try both,
  keep the higher ZNCC" wrecked 3 of 32 real Si rim points.

## To resume

1. DONE 2026-10-07: the revise step (every finding verified, applied or
   rejected; disposition table in plan section 12.5).
2. After the Stage E implementation commit, splice the blocks into
   requirements.md, validation.md (renumber the ledger 200-216 to follow
   the last Stage E entry, and every "ledger 2xx" reference with it),
   plan.md and roadmap.md, each at the anchor its block comment names;
   record the plan gate as pending Johan's review; commit.
3. Stop. Stage F tests and implementation wait for Johan.
