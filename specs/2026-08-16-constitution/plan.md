# Phase 0 -- constitution and housekeeping: plan

Model: Fable 5 (xhigh, ultracode) -- this is a spec/documentation phase; no
production code.

## 1. Repository housekeeping
1. Delete `IndexEBSD.nml` (untracked artefact of the planning survey).
2. `git remote add upstream https://github.com/pyxem/kikuchipy.git`; `git fetch upstream --tags`.
3. `git stash push` the two locally modified notebooks; `git merge upstream/develop` into `develop`; `git stash pop`; resolve any metadata-only conflict in favour of the local copy; `git reset` so the notebooks stay unstaged; drop the stash.
4. `git push origin develop`; `git switch -c spherical-indexing-constitution`.

## 2. Constitution
1. `specs/_research/` <- the 13 planning artefacts with descriptive names + `README.md` index.
2. `specs/mission.md` (mission, program mapping, out-of-scope list, success criteria, legal status incl. patent search result).
3. `specs/tech-stack.md` (deps, numerics decisions incl. critique corrections, layout/style, tests/docs/data, commands, per-feature process and model assignment).
4. `specs/roadmap.md` (12 phases with checkbox tasks; dependency chain).
5. `specs/2026-08-16-constitution/{requirements,plan,validation}.md` (this folder).

## 3. Documentation groundwork
1. `doc/user/bibliography.bib`: add the nine entries listed in `requirements.md`.
2. `doc/user/related_projects.rst`: extend the EMSphinx line with the source repository and SHT database links.
3. `src/kikuchipy/__init__.py` `credits` and `.zenodo.json`: add "Johan Westraadt".
4. `.pre-commit-config.yaml`: exclude `specs/` from all hooks.

## 4. Legal gate
1. Patent search (Google Patents XHR queries; inventor/assignee/keyword) -- queries, counts and hits archived in `specs/_research/patent-search-2026-08-16.md`, summary in `mission.md`.
2. Draft `specs/2026-08-16-constitution/upstream-issue.md`: GitHub issue for `pyxem/kikuchipy` (references EMSphInx issue #7, describes the port, GPL-only placement, `.sht` BSD-3 codec, patent statement, asks two explicit questions) and the email to `pyxem.team@gmail.com`. **Not sent by the agent** -- the user sends/approves.

## 5. Adversarial review
1. Ultracode workflow: three read-only critics (factual, completeness, executability) refute the constitution against the repo and research notes; findings fixed in place.

## 6. Verification and PR
1. `uv run sphinx-build -b html -D nbsphinx_execute=never doc doc/_build/html_check` exits 0; the nine new keys render in `user/bibliography.html`.
2. `uv run pre-commit run --files doc/user/bibliography.bib doc/user/related_projects.rst src/kikuchipy/__init__.py .zenodo.json .pre-commit-config.yaml` clean.
3. `git add specs doc/user/bibliography.bib doc/user/related_projects.rst src/kikuchipy/__init__.py .zenodo.json .pre-commit-config.yaml`; `git status` shows exactly those staged and the two notebooks unstaged.
4. `git commit -s`; `git push -u origin spherical-indexing-constitution`; `gh pr create --base develop --repo jwestraadt/kikuchipy` with the PR template.
