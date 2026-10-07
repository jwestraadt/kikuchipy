# Stage F draft (parked 2026-10-07): Fourier-Mellin rotation initial guess

Status: PARKED mid-review at about 02:50 on 2026-10-07, when Johan had to
power down. Nothing here is part of the spec yet; nothing is built.

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
  explorations (method and literature, codebase seams, a CPU prototype).
- `critic_findings.md`: the 45 findings of the two adversarial critics
  (14 major, 31 minor, no blocker), NOT yet dispositioned. The reviser was
  stopped mid-way, so the drafts may already carry some fixes.
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

1. Run the revise step again over the drafts here with
   `critic_findings.md` (verify each finding, apply or reject, append a
   disposition table to plan section 12).
2. After the Stage E implementation commit, splice the blocks into
   requirements.md, validation.md (renumber the ledger), plan.md and
   roadmap.md; record the plan gate as pending Johan's review; commit.
3. Stop. Stage F tests and implementation wait for Johan.
