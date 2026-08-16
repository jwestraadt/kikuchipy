# Phase 0 -- constitution and housekeeping: validation

## Automated
- `git log --oneline -1 develop` == `upstream/develop` head (4ed31813 at the time of writing). After step 6.3 of `plan.md`, `git status --short` shows exactly `specs/`, `doc/user/bibliography.bib`, `doc/user/related_projects.rst`, `src/kikuchipy/__init__.py`, `.zenodo.json`, `.pre-commit-config.yaml` staged and the two notebooks unstaged; after 6.4, `git log develop..HEAD` shows the signed commit(s).
- `uv run python -c "import pathlib, sys; sys.exit(pathlib.Path('IndexEBSD.nml').exists())"` exits 0.
- `uv run python -c "import json; json.load(open('.zenodo.json'))"` succeeds; `uv run python -c "import kikuchipy as kp; assert 'Johan Westraadt' in kp.credits"`.
- Bibliography: `uv run python -c "import pybtex.database as p; db=p.parse_file('doc/user/bibliography.bib'); [db.entries[k] for k in ['lenthe2019spherical','lenthe2019pseudo','reinecke2011libpsht','schaeffer2013efficient','sneeuw1994global','huhle2009normalized','kostelec2008ffts','rosca2010new','gutman2008shape']]"` succeeds (pybtex is a sphinxcontrib-bibtex dependency).
- `uv run pre-commit run --files doc/user/bibliography.bib doc/user/related_projects.rst src/kikuchipy/__init__.py .zenodo.json .pre-commit-config.yaml` passes; `specs/` is excluded from pre-commit (no licence header stamped on the spec Markdown).
- Sphinx: `uv run sphinx-build -b html -D nbsphinx_execute=never doc doc/_build/html_check` exits 0, logs `parsed 41 entries` for the .bib, and `doc/_build/html_check/user/bibliography.html` contains the nine new keys (nothing cites them yet; they render via `:all:`).

## Manual
- `specs/mission.md` program-mapping table covers every EMSphInx program in `EMSphInx/programs/*.cpp` (ebsd_wizard is GUI, index_ebsd, master_xcorr, mp2sht, pattern_repack, sht_wisdom, sht2png, ebsp_dims).
- `specs/tech-stack.md` contains every blocker/major correction from `specs/_research/critique-*.md` (Euler sign, fastSize, dctn type-3, BSD-3 codec, 38 point groups, single PC, no module-global compat flag, version-gated oracles, gather LUT, measured-then-pinned scores).
- `specs/roadmap.md` phases match the approved plan and the dependency chain.
- The upstream issue draft asks (i) whether GPL-derived code is acceptable in `indexing/`, (ii) whether the CMU provisional-patent statement needs resolving before merge, and links EMSphInx issue #7.
- The user's notebook edits survived the sync: `git diff develop -- doc/tutorials/hybrid_indexing.ipynb doc/tutorials/load_save_data.ipynb` still shows the local edits, and the only stash-pop conflict (`language_info.version` 3.14.5 vs 3.13.12) was resolved to the local value; the stash was dropped afterwards, so no other pre-sync reference remains (future phases keep a tag on the stash commit before dropping).

## Definition of done
All boxes of Phase 0 in `specs/roadmap.md` ticked, PR opened into fork `develop`, PR description states that this branch adds no code and that the upcoming modules are GPL-only.
