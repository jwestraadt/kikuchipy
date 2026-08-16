# Phase 0 -- constitution and housekeeping: requirements

## Scope

In scope:
- Repository housekeeping so that all later feature branches start from a clean, current base: fork `develop` == `upstream/develop`, the user's uncommitted notebook edits preserved and never committed, stray `IndexEBSD.nml` (created by the planning explorer running `IndexEBSD.exe -t`) removed.
- The constitution: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, and the archived planning artefacts `specs/_research/` with an index.
- Documentation groundwork every later feature cites: bibliography entries, an extended EMSphInx entry in related projects, contributor credit.
- The legal gate: patent search recorded; upstream issue and maintainer email *drafted* (sending is an outward-facing action that the user triggers).

Out of scope: any `src/` code, tests, notebook cells, CHANGELOG entry. The only user-facing change is that the nine new bibliography entries render on the public Bibliography page immediately (`doc/user/bibliography.rst` uses `:all:`); this is accepted so that every later feature can `:cite:` them.

## Decisions

- Branch: `spherical-indexing-constitution`; PR into fork `develop`.
- `specs/` is committed on the fork by this phase's signed commit (fine for merges: upstream has no `specs/`), stripped from any upstream PR.
- The stash-pop conflict in `doc/tutorials/load_save_data.ipynb` (upstream bumped the kernel `language_info.version` to 3.14.5) was resolved by keeping the user's local value 3.13.12; both notebooks remain unstaged working-tree edits.
- `.pre-commit-config.yaml`: `specs/` added to the top-level `exclude` so the `licenseheaders` hook does not stamp GPL headers onto spec Markdown (fork-only change).
- Contributor entry: name only in `credits` and `.zenodo.json` (ORCID/affiliation can be added by the user).
- Bibliography keys follow the file's `<firstauthor><year><word>` style: `lenthe2019spherical`, `lenthe2019pseudo`, `reinecke2011libpsht`, `schaeffer2013efficient`, `sneeuw1994global`, `huhle2009normalized`, `kostelec2008ffts`, `rosca2010new`, `gutman2008shape`.
- `related_projects.rst`: keep the maintainers' EMSphinxEBSD line, add the source repository and the SHT master-pattern database so the tutorial can point at them.

## Context

- kikuchipy conventions: `doc/dev/using_git.rst` (branch off develop, `-s` sign-off, merge not rebase), `doc/dev/licensing_considerations.rst` (GPL/BSD policy), `.github/PULL_REQUEST_TEMPLATE.md` (credits + `.zenodo.json` reviewer checklist), `doc/dev/maintaining_package_credits.rst`.
- Research notes: `specs/_research/README.md`.
