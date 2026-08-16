# Legal & community-process constraints for porting EMSphInx spherical indexing into kikuchipy

Repo states: kikuchipy local HEAD `49b1c11c` (2026-05-24, branch `develop`, origin = `git@github.com:jwestraadt/kikuchipy.git` — a **fork**, no `upstream` remote configured). EMSphInx local HEAD `60f3517` (2026-03-22, branch `master`, has a local-only commit "Add Visual Studio 2022 CMake build instructions").

---

## 1. LICENSING

### 1.1 EMSphInx license = GPL-2.0-**or-later** (not BSD; no patent clause)

`c:/Users/westraadt.1/Repos/EMSphInx/license.txt` — full text of **GNU GENERAL PUBLIC LICENSE Version 2, June 1991** (lines 1-87). Verbatim line 1-3: `GNU GENERAL PUBLIC LICENSE` / `Version 2, June 1991`. Contains no explicit patent grant; GPLv2 §7 (line 64) is the "liberty or death" clause only: *"if a patent license would not permit royalty-free redistribution of the Program by all those who receive copies directly or indirectly through you, then the only way you could satisfy both it and this License would be to refrain entirely from distribution of the Program."* GPLv2 §9 (line 76) is the "or any later version" mechanism.

Per-file headers are identical in both requested files — `include/sht/square_sht.hpp` lines 1-33 and `include/idx/indexer.hpp` lines 1-33 (byte-identical block). Exact wording:

```
 * Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University *
 * All rights reserved.                                                *
 *                                                                     *
 * Author: William C. Lenthe                                           *
 *                                                                     *
 * This package is free software; you can redistribute it and/or       *
 * modify it under the terms of the GNU General Public License as      *
 * published by the Free Software Foundation; either version 2 of the  *
 * License, or (at your option) any later version.                     *
 ...
 * Interested in a commercial license? Contact:                        *
 *                                                                     *
 * Center for Technology Transfer and Enterprise Creation              *
 * 4615 Forbes Avenue, Suite 302                                       *
 * Pittsburgh, PA 15213                                                *
 *                                                                     *
 * phone. : 412.268.7393                                               *
 * email  : innovation@cmu.edu                                         *
 * website: https://www.cmu.edu/cttec/                                 *
```

Key facts:
- **"either version 2 of the License, or (at your option) any later version"** → SPDX `GPL-2.0-or-later`. This is the decisive clause for the port.
- Copyright holder: **De Graef Group, Carnegie Mellon University** (2019). Author: **William C. Lenthe**.
- Same header appears on the CMake dependency scripts too (`depends/FFTW.cmake` lines 1-33, `depends/HDF5.cmake`, `depends/wxWidgets.cmake`) and on vendored `include/xtal/vendor/tsl.hpp` lines 1-12, which is `Copyright (c) 2019, William C. Lenthe` (personal, not CMU) under the same GPL-2.0-or-later text.
- The presence of a **commercial-license offer** (CMU CTTEC contact block in every header) signals CMU treats this as commercially licensable IP; it is a dual-licensing posture, not a permissive grant.

### 1.2 Patent statement (exact wording)

`c:/Users/westraadt.1/Repos/EMSphInx/ReadMe.md`, section "## Financial Support":

> "The *EMSphInx* code was developed with support from an ONR Vannevar Bush Faculty Fellowship grant, N00014-­16-­1-­2821. **The central indexing algorithm is covered by a provisional patent application.**"

License section of the same ReadMe (verbatim):

> "## License ##
> *EMSphInx* source files are distributed under GNU General Public License v2.0 (GPL2), see the license.txt file for details.
> *EMSphInx* also includes several files from BSD licensed (3-clause) projects (please refer to the individual files for details):
> - include/miniz/miniz.c"

Note the ReadMe body says "GPL2" flatly while the actual file headers say "v2 or any later version" — **the file headers govern** (they are the notices required by GPLv2 §0). The ReadMe's "BSD licensed (3-clause)" claim about miniz is **wrong**: `include/miniz/miniz.c` lines 1-26 carry the **MIT** license (`Copyright 2013-2014 RAD Game Tools and Valve Software` / `Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC` / "Permission is hereby granted, free of charge... subject to the following conditions:"). Harmless (MIT ⊂ GPL-compatible) but worth not repeating the error.

Patent status research: web searches (Google/Bing via WebSearch) found **no granted US patent** attributable to Lenthe/De Graef/CMU on spherical indexing, and no patent number is cited anywhere in the repo (`grep` for "patent" only hits the ReadMe line above). A *provisional* application (filed ~2019) lapses after 12 months unless converted; whether a non-provisional issued is **unresolved and should be checked on Google Patents / USPTO Patent Center** before any public release. This is a **patent** question, entirely orthogonal to copyright — reimplementing from the paper does not avoid it.

### 1.3 kikuchipy license = GPL-3.0-or-later

- `c:/Users/westraadt.1/Repos/kikuchipy/LICENSE` lines 1-2: `GNU GENERAL PUBLIC LICENSE` / `Version 3, 29 June 2007` (full GPLv3, 35815 bytes).
- Per-file header, `src/kikuchipy/indexing/_dictionary_indexing.py` lines 1-18 (verbatim):
```python
#
# Copyright 2019-2026 the kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
#
```
- The header is **auto-inserted** by pre-commit: `.pre-commit-config.yaml` lines 34-38 run `johann-petrak/licenseheaders` v0.8.8 with `args: ["-t", ".license.tmpl", "-cy", "-n", "kikuchipy", "-f"]`. Template = `c:/Users/westraadt.1/Repos/kikuchipy/.license.tmpl` (`Copyright 2019-${years} the ${projectname} developers` …). Note the hook is currently in the CI `skip:` list (`.pre-commit-config.yaml` line 45: `skip: [licenseheaders]`, with `# TODO: Remove skip once (nearly) all files are formatted with the license template`).
- `pyproject.toml` line 13 `license = {file = "LICENSE"}`; classifier line ~29: `"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"`.
- `README.rst` badge target: `https://opensource.org/license/GPL-3.0`.

### 1.4 Compatibility verdict

**GPL-2.0-or-later → GPL-3.0-or-later is legally permitted.** Because every EMSphInx source file grants "version 2 … or (at your option) any later version", a downstream recipient may exercise the GPLv2 §9 option and use the code under **GPLv3**. Combining it with kikuchipy (GPLv3+) produces a GPLv3+ work. This is the standard, uncontroversial upgrade path. (Had EMSphInx been GPL-2.0-**only**, the port would have been *impossible* — GPLv2-only and GPLv3 are mutually incompatible.)

Obligations that attach to the port (GPLv2 §1/§2a and GPLv3 §5):
1. **Preserve the copyright notice** — the ported Python file(s) must carry `Copyright (c) 2019, De Graef Group, Carnegie Mellon University` (author William C. Lenthe) alongside kikuchipy's own header, plus a statement that the code is derived from EMSphInx and relicensed under GPLv3+ per the "or later" option.
2. **Mark modified files** — GPLv2 §2(a): *"You must cause the modified files to carry prominent notices stating that you changed the files and the date of any change."* A translation C++→Python is a modification/"translation is included without limitation in the term 'modification'" (§0, line 30).
3. **No additional restrictions**; source availability is satisfied by kikuchipy's sdist.
4. Academic attribution (separate from law): cite **Lenthe, Singh & De Graef (2019)**, `doi:10.1016/j.ultramic.2019.112841`, and add the entry to `doc/user/bibliography.bib` (currently contains no `lenthe*` entry — verified by grep; nearest entries are `jackson2019dictionary`, `chen2015dictionary`). Also add EMSphInx to `doc/user/related_projects.rst` (currently lists EMsoft but **not** EMSphInx).

**Hard internal constraint — kikuchipy's dual-license policy.** `doc/dev/licensing_considerations.rst` exists on `develop` (https://raw.githubusercontent.com/pyxem/kikuchipy/develop/doc/dev/licensing_considerations.rst, rendered at https://kikuchipy.org/en/latest/dev/licensing_considerations.html) but is **absent from the local clone** (`doc/dev/` locally has 14 files minus this one — the local clone predates it; GitHub `contents` API for `doc/dev?ref=develop` lists `licensing_considerations.rst` at position 9). Its content:

> "kikuchipy as a whole carries the GPLv3+ license, while select components carry a more permissive BSD 3-Clause license."
> "…we cannot import from GPL licensed code in the files and directories that carry the BSD license."
> Reviewers should check "whether the code respects the more permissive BSD license" and "whether contributors are aware they can opt for BSD licensing instead of GPL". Rationale references "a comment on this RosettaSciIO discussion about licensing". Contact: `pyxem.team@gmail.com`.

Consequence: a spherical-indexing module derived from EMSphInx **must be placed in GPLv3+ files and can never be moved to, or imported from, any BSD-3 area** of kikuchipy. This must be stated explicitly in the PR, because the project's default reviewer prompt is to ask contributors to consider BSD. **The local clone must be updated (`git pull upstream develop`) before starting — it is missing this policy file and is ~3 months behind.**

### 1.5 EMSphInx third-party dependencies (`depends/`)

Only three CMake fetch scripts: `depends/FFTW.cmake`, `depends/HDF5.cmake`, `depends/wxWidgets.cmake` (no vendored source trees). Plus in-tree `include/miniz/` (miniz.c/.h/LICENSE/readme.md).

| Dep | Where | Version | License | Needed for a Python port? |
|---|---|---|---|---|
| **FFTW3** | `depends/FFTW.cmake:70` `set(FFTW_URL "http://www.fftw.org/fftw-${FFTW_VER}.tar.gz")`, `:61-62` `set(FFTW_VER "3.3.7")`; used via `include/util/fft.hpp:42` `#include "fftw3.h"` | 3.3.7, built from source by default (`CMakeLists.txt:49` `EMSPHINX_BUILD_FFTW ... ON`) | **GPL-2.0-or-later** (this is why EMSphInx as-distributed is GPL, independent of CMU's choice) | **NO.** Replace with `scipy.fft` (PocketFFT, **BSD-3**) — see §4. |
| **HDF5** | `depends/HDF5.cmake:44-45`, tag `hdf5-1_8_20` | 1.8.20 | HDF5 BSD-style | No — kikuchipy already depends on `h5py >= 2.10`. Needed only to *read* `.sht`/`.h5` master patterns; `.sht` is a custom format (see `include/idx/master.hpp`). |
| **wxWidgets** | `depends/wxWidgets.cmake:50` `GIT_TAG "v3.1.2"` | 3.1.2, GUI only (`CMakeLists.txt:54` `EMSPHINX_BUILD_GUIS ON`) | wxWindows Library Licence (LGPL+exception) | No (GUI only). |
| **miniz** | `include/miniz/` (vendored; `CMakeLists.txt:173` `set(BuildMiniZ OFF)`, fetch block 174-183 disabled, URL `github.com/richgel999/miniz/releases/.../miniz-2.0.8.zip`) | 2.0.8 | **MIT** (header lines 3-25), *not* BSD-3 as ReadMe claims | Only if `.sht` container uses miniz deflate — Python `zlib` covers it. |

FFTW build flags relevant to reproducing numerics: `CMakeLists.txt:46-51` — `EMSPHINX_FFTW_D` (double) ON by default, float/long-double OFF; `EMSPHINX_FFTW_SIMD` ON, `EMSPHINX_FFTW_AVX2` OFF.

**Bottom line on FFTW:** a Python port must NOT link FFTW, and does not need to. `pyfftw` is GPL-3.0 (would be license-compatible but adds a heavy non-wheel-friendly dep); `scipy.fft` is already a kikuchipy dependency (`scipy >= 1.7`) and is BSD-3 PocketFFT. Using scipy.fft removes the only *GPL-forced* dependency in the chain, leaving the copyleft obligation coming solely from CMU's own code.

---

## 2. CONTRIBUTION PROCESS (pyxem/kikuchipy)

### 2.1 Local `.github/` vs GitHub

`ls c:/Users/westraadt.1/Repos/kikuchipy/.github`:
```
dependabot.yml
pull_request_template.md
workflows/  ->  perhaps_make_tagged_release_draft.py, perhaps_make_tagged_release_draft.yml,
                publish.yml, tests.yml, weekly.yml
```
- There is **no** `.github/CONTRIBUTING.md` locally, and `https://raw.githubusercontent.com/pyxem/kikuchipy/develop/.github/CONTRIBUTING.md` returns **HTTP 404**. Contribution guidance lives entirely in `doc/dev/`.
- `https://raw.githubusercontent.com/pyxem/kikuchipy/develop/.github/PULL_REQUEST_TEMPLATE.md` also **404s** — the real path is lowercase `.github/pull_request_template.md`, which matches the local file exactly.
- `README.rst` has **no "Contributing" section** (verified: grep for "contribut" returns nothing; sections are Documentation / Installation / Citing kikuchipy). The GitHub landing page therefore does not carry contribution rules — everything is at https://kikuchipy.org/en/latest/dev/.

### 2.2 PR template — the de-facto checklist (`c:/Users/westraadt.1/Repos/kikuchipy/.github/pull_request_template.md`)

```
1  #### Description of the change
2  <!-- Remember to branch off the develop branch for new features and the main branch for patches. -->
5  #### Progress of the PR
6  - [ ] [Docstrings for all functions](https://numpydoc.readthedocs.io/en/latest/example.html)
7  - [ ] Unit tests with pytest for all lines
8  - [ ] Clean code style by [running black via pre-commit](https://kikuchipy.org/en/latest/dev/code_style.html)
10 #### Minimal example of the bug fix or new feature
19 #### For reviewers
21 - [ ] The PR title is short, concise, and will make sense 1 year later.
22 - [ ] New functions are imported in corresponding `__init__.py`.
23 - [ ] New features, API changes, and deprecations are mentioned in the unreleased
24       section in `CHANGELOG.rst`.
25 - [ ] New contributors are added to `kikuchipy/__init__.py` and `.zenodo.json`.
```

### 2.3 Git workflow (`doc/dev/using_git.rst`)

- Lines 7-8: *"If you want to add a new feature, branch off of the ``develop`` branch, and when you want to fix a bug, branch off of ``main`` instead."* → **spherical indexing = feature = branch off `develop`, PR into `develop`.**
- Line 12: `git switch -c your-awesome-feature-name upstream/develop`
- Line ~22: commits must be **signed off**: `git commit -s -m "..."` — *"The ``-s`` makes sure that you sign your commit with your GitHub-registered email as the author."*
- Lines 31-45: keep branch current by **merging** `develop` in (not rebasing).
- Line 54+: *"make a pull request to kikuchipy's ``develop`` branch for new features and ``main`` branch for bug fixes."*
- Fork required — `doc/dev/setting_up_development_installation.rst` lines 3-19: *"You need a fork of the repository in order to make changes"*, `git clone https://github.com/your-username/kikuchipy.git`, then `git remote add upstream https://github.com/pyxem/kikuchipy.git`. **The local clone has origin=jwestraadt fork but no `upstream` remote** — must be added.
- Dev install: `pip install --editable ".[dev]"` (conda env `kp-dev` recommended).
- No explicit written "one feature per PR" rule exists anywhere in `doc/dev/` or the template; the closest is the reviewer item "The PR title is short, concise, and will make sense 1 year later." Nonetheless the practical norm (see PRs #590, #461) is one feature per PR.

### 2.4 Code style (`doc/dev/code_style.rst`)

PEP 8 + **Black** code style, run via **ruff** through pre-commit (`pre-commit install`, line 13). numpydoc docstrings (line 16), *checked when building the docs*. Comment/docstring lines ≤ **72 characters** (line 19). Imports in three blocks: stdlib / third-party / kikuchipy. **Type hints in signatures only, no types duplicated in docstrings** (line 26 + example). Lazy module imports per **PEP 562** (`lazy_loader.attach_stub` in `src/kikuchipy/__init__.py:41`) — a new `kikuchipy.indexing` symbol must be registered in the corresponding `__init__.pyi` stub, not just `__init__.py`.
Actual hooks (`.pre-commit-config.yaml`): `ruff` + `ruff-format` (rev v0.15.14), `black-jupyter` on `.ipynb` with `--line-length=77`, `licenseheaders` v0.8.8. CI config: `autofix_prs: false`, `autoupdate_schedule: monthly`.

### 2.5 Tests (`doc/dev/running_writing_tests.rst`)

pytest; tests live in a `tests/` directory per module (top-level `tests/` in this layout). `pip install -e ".[tests,coverage]"`; run `pytest --cov`; parallel `pytest -n 4` (xdist); flaky reruns `pytest --reruns 2`; **doctests are run**: `pytest --doctest-modules src` (uses the top-level `conftest.py`, 28 kB). Tests require an internet connection (data module downloads ~15 MB from `pyxem/kikuchipy-data`).
**Numba-specific rules (directly relevant — a fast SHT/xcorr kernel will use numba):**
> "A Numba decorated function ``numba_func()`` is only covered if it is called in the test as ``numba_func.py_func()``."
> "Always test a Numba decorated function calling ``numba_func()`` directly, in addition to ``numba_func.py_func()``, because the machine code function might give different results on different OS with the same Python code. See this issue https://github.com/pyxem/kikuchipy/issues/496".

### 2.6 Performance (`doc/dev/improving_performance.rst`)

> "To keep memory in check, we should use Dask wherever possible. To speed up computations, we should use Numba wherever possible."
> "To check whether a change is an improvement or a regression, **a benchmark should be written**. These are stored in the top directory ``kikuchipy/benchmarks``… run using pytest-benchmark: `pytest --benchmark-only`."
Local `benchmarks/` directory exists at repo root.

### 2.7 Changelog (`CHANGELOG.rst`)

Keep-a-Changelog 1.1.0 + SemVer (lines 9-11). Head of file has an `Unreleased` section with empty `Added / Changed / Removed / Fixed / Deprecated` subsections (lines 17-31). Entry style (from 0.12.0 block, lines 40+): one sentence + PR link, e.g.
```
- Download from conda-forge of ``kikuchipy-base`` without optional dependencies.
- Can now read simulated master patterns from EMsoft's EMEBSDmasterOpenCL.f90 program.
  (`#730 <https://github.com/pyxem/kikuchipy/pull/730>`_)
```
Current version `0.13.dev0` (`src/kikuchipy/__init__.py:39`).

### 2.8 Credits (`doc/dev/maintaining_package_credits.rst`)

> line 4: "Whenever we get a new contributor, **with their consent**, they should be added to the package credits."
> lines 8-9: two sources — `kikuchipy/__init__.py`: List of contributors `__credits__`; `.zenodo.json`: Zenodo entry.
> "the initial commiter is listed first, with the others sorted by line contributions."

In the current tree the variable is actually named `credits` (not `__credits__`) — `src/kikuchipy/__init__.py:22-37`, comment line 22 `# Initial committer first, then sorted by line contributions`, 13 names ending `"Tijmen Vermeij"`. `.zenodo.json` has the same 13 in the same order with ORCIDs/affiliations. **There is no `doc/user/credits` page** — `doc/user/` = `applications.rst, bibliography.bib, bibliography.rst, index.rst, installation.rst, open_datasets.rst, related_projects.rst`. So "credit in doc/user/credits" is not the mechanism; it is `src/kikuchipy/__init__.py` + `.zenodo.json` only.

### 2.9 Code of Conduct

`c:/Users/westraadt.1/Repos/kikuchipy/CODE_OF_CONDUCT.rst` (7582 bytes; also at https://raw.githubusercontent.com/pyxem/kikuchipy/develop/CODE_OF_CONDUCT.rst). Applies to "all spaces managed by the kikuchipy project… mailing lists, issue trackers, wikis, blogs, and any other communication channel". Guidelines: "Be open", "Be empathetic, welcoming, friendly, and patient", etc. (numpy/SciPy-derived CoC). `doc/dev/index.rst` lines 22-23: *"kikuchipy has a Code of conduct that should be honoured by everyone who participates in the kikuchipy community."*

### 2.10 Recommended entry point (from `doc/dev/index.rst` lines 12-18)

> ".. tip:: This guide can look intimidating… **The shortest route to start contributing is to create a GitHub account and explain what you want to do `in an issue <https://github.com/pyxem/kikuchipy/issues/new>`__.**"

Channels: issues https://github.com/pyxem/kikuchipy/issues, discussions https://github.com/pyxem/kikuchipy/discussions, Gitter https://gitter.im/pyxem/kikuchipy. **Open an issue first** — mandatory in spirit for a feature of this size, and doubly so because of the license/patent question (route the licensing question to `pyxem.team@gmail.com` per `licensing_considerations.rst`).

`doc/dev/index.rst` toctree (12 sections): setting_up_development_installation, code_style, using_git, building_writing_documentation, handling_deprecations, running_writing_tests, adding_to_data_module, improving_performance, continuous_integration, maintaining_package_credits, code_of_conduct — **plus `licensing_considerations` on develop** (not in the local, stale copy).

---

## 3. PRIOR ART / EXISTING WORK

### 3.1 kikuchipy issue tracker — nothing exists

- `https://api.github.com/search/issues?q=repo:pyxem/kikuchipy+EMSphInx+OR+"spherical indexing"+OR+"spherical harmonic"` → **`total_count: 0`**.
- `…q=repo:pyxem/kikuchipy in:title,body "spherical" type:issue` → **6 items, none about indexing**: #802 tk error on windows (open, 2026-06-02), #715 test segfault, #566 PyVista/pythreejs, #213 Vector3d spherical coords, #265 rename projection "spherical"→"stereographic", #249 Lambert proj read. Related but off-topic: #536 "Plot spherical master pattern", #518 "Spherical plotting of EBSD master pattern".
- Discussions: `https://github.com/pyxem/kikuchipy/discussions?discussions_q=spherical` → **"There are no matching discussions."**
- Repo-wide grep for `emsphinx|spherical harmonic|.sht` across `*.py *.rst *.toml *.json *.ipynb` → **zero hits**. EMSphInx is not mentioned in `doc/user/related_projects.rst` (which does list EMsoft, PyEBSDIndex, MTEX, DREAM.3D, AstroEBSD, OpenXY, DefDAP, xcdskd, pycotem) nor in `doc/user/bibliography.bib`.

**Conclusion: no issue, no discussion, no roadmap statement, no partial implementation. Greenfield inside kikuchipy.**

### 3.2 The single most important artefact: EMSphInx issue #7

https://github.com/EMsoft-org/EMSphInx/issues/7 — **"What will the upcoming Python bindings entail?"**, author **hakonanes** (Håkon Wiik Ånes, kikuchipy's initial committer/maintainer), created **2020-02-28**, **state: open**, **zero comments** (comments API returns `[]` — unanswered for 6+ years). Body verbatim:

> "Hi all,
> I, as a lot of other people, would be extremely interested in calling EMSphInx from Python. Specifically, I would like to add a spherical indexing method to the `EBSD` class in our `KikuchiPy` package (https://github.com/kikuchipy/kikuchipy). I am therefore curious as to what "Python bindings" listed under upcoming features entails. Basically, will this mean I can download EMSphInx from PyPI or Anaconda Cloud and import it like any other Python package? If not, what will be the necessary steps to use EMSphInx?
> Håkon"

This is decisive framing for a PR: the maintainer **wants** this feature, tried the wrapper route, and got no response. EMSphInx's own ReadMe still lists "Python bindings" under "## What's coming in future versions" — i.e. still not delivered.

### 3.3 No existing Python implementation

- No `pyEMSphInx` on PyPI or GitHub (searched). Python access to the EMsoft ecosystem is only via `pyEMsoft` wrappers for **EMsoft** (Fortran, BSD-3), which do **not** cover EMSphInx (deliberately kept in a separate GPL repo).
- `PyEBSDIndex` (https://pypi.org/project/pyebsdindex) = Hough/Radon indexing only; already a kikuchipy optional dep (`pyebsdindex >= 0.3.9.2` in `[project.optional-dependencies].all`).
- `pyshtools`/SHTOOLS = general SHT library, **no EBSD indexing**.
- Commercial: EDAX **OIM Analysis** ships "Spherical Indexing" (https://edaxblog.com/2022/08/17/spherical-indexing/, https://www.azom.com/article.aspx?ArticleID=24278) — i.e. the algorithm is commercialised, consistent with the CMU patent posture. Master-pattern data: SHT database https://github.com/EMsoft-org/SHTdatabase and https://kilthub.cmu.edu/articles/dataset/Spherical_Harmonic_Transform_Master_Pattern_Library/9974612 (120 structures × 10/15/20/25/30 kV, `.sht`).

### 3.4 Algorithm reference — Lenthe, Singh & De Graef (2019)

Full abstract retrieved from OSTI (https://www.osti.gov/pages/biblio/1575858); paywalled at https://www.sciencedirect.com/science/article/abs/pii/S0304399119301585; DOI `10.1016/j.ultramic.2019.112841` (302-redirects to linkinghub). Citation: **Lenthe, W. C.; Singh, S.; De Graef, M., *Ultramicroscopy* **207** (2019) 112841.** Verbatim abstract:

> "A new approach is proposed for the indexing of electron back-scattered diffraction (EBSD) patterns. The algorithm employs a spherical master EBSD pattern and computes its cross-correlation with a back-projected experimental pattern using the spherical harmonic transform (SHT). This approach is significantly faster than the recent dictionary indexing algorithm, but shares the latter's robustness against noise. The underlying theory is presented, followed by example applications, one on a series of Ni EBSD data sets recorded with decreasing signal-to-noise ratio, the other on a large shot-peened Al data set. The dependence of indexing speed and memory usage on the SHT bandwidth is explored. The speed gains of the new algorithm are achieved by executing real-valued Fast Fourier Transforms, explicitly incorporating crystallographic symmetry in the cross-correlation computation, and using efficient loop ordering to improve the caching behavior. The algorithm produces a cross-correlation array in the zyz Euler space; an orientation refinement procedure is proposed based on analytical derivatives of the Wigner d functions. As a result, the new approach can be applied to any diffraction modality for which the scattered intensity can be represented on a spherical surface."

Second citable paper (per EMSphInx ReadMe): Pseudo-symmetry prediction, `doi:10.1107/S1600576719011233`.

**Bandwidth parameters — authoritative source is the code, not the paper.** `include/modality/ebsd/nml.hpp`:
- line 208: `bw = 68 ;` ← **default bandwidth is 68**
- line 298 comment (verbatim): *"what bandwidth should be used, if 2\*bw-1 is product of small primes it is a good candidate for speed (fft is significant fraction of time): **32,38,41,53,63,68,74,88,95,113,123,158**"*
- line 414 namelist comment: *"! spherical harmonic bandwidth to be used (2\*bw-1 should be a product of small primes for speed)"*
- line 635: `if(bw < 16 || bw > 512) throw std::runtime_error("unreasonable bandwidth (should be [16, 512])");`
- line 76: `int32_t bw ;//spherical harmonic bandwidth to index with`; line 103 `clearIdxPrm()` sets `bw = -1`.
So the recommended set is **{32, 38, 41, 53, 63, 68, 74, 88, 95, 113, 123, 158}** (default **68**), valid range [16, 512], selection rule **2·bw−1 = product of small primes** (e.g. bw=63 → 125 = 5³; bw=68 → 135 = 3³·5; bw=88 → 175 = 5²·7; bw=95 → 189 = 3³·7; bw=113 → 225 = 3²·5²). Other API entry points: `include/idx/indexer.hpp:88,95` (`//@param bw : bandwidth`), `include/idx/master.hpp:163,195`, `include/sht/sht_xcorr.hpp:884` (`mBW` ≤ `bw`), `include/sht/square_sht.hpp:163`.

---

## 4. SHT / FFT LIBRARY LANDSCAPE

(PyPI HTML pages are JS-gated and return "A required part of this site couldn't load" to WebFetch; all data below is from the **PyPI JSON API**.)

### 4.1 pyshtools — https://pypi.org/project/pyshtools/ (JSON: https://pypi.org/pypi/pyshtools/json)

- Version **4.14.1**; summary "SHTOOLS - Spherical Harmonic Tools"; `requires_python >= 3.10`.
- License: **BSD-3-Clause**; classifier `"License :: OSI Approved :: BSD License"`. (Project page: https://shtools.github.io/SHTOOLS/, https://github.com/SHTOOLS/SHTOOLS — "SHTOOLS is open source software and is distributed under the 3-clause BSD license.")
- Classifiers include `"Programming Language :: Fortran"` — the Fortran core **is compiled into the wheels**; no user Fortran toolchain needed.
- **Wheels: `win_amd64` ✔, `manylinux_2_17_x86_64`/`manylinux2014_x86_64` ✔, `macosx_12_0_x86_64` ✔ — for cp39, cp310, cp311, cp312, cp313, cp314**, plus sdist `pyshtools-4.14.1.tar.gz`.
- ⚠ No `win_arm64`; the macOS wheel list reported is x86_64 (arm64 coverage not confirmed from the JSON summary).
- **License-wise this is the easy option** — BSD-3 imposes nothing on kikuchipy. But note it is *not* a kikuchipy dependency today and would be a new (large, compiled) dependency; and pyshtools' Driscoll–Healy/GLQ grids are not the *square Lambert* grid EMSphInx uses (`include/sht/square_sht.hpp`), so it does not drop in for the back-projection step.

### 4.2 shtns — https://pypi.org/project/shtns/ (JSON: https://pypi.org/pypi/shtns/json)

- Version **3.7.5**; `requires_python >= 3.2`.
- License: **CeCILL-2.1** ("CEA CNRS Inria Logiciel Libre License, version 2.1") — a French copyleft licence, GPL-compatible in one direction (CeCILL→GPL is explicitly allowed by CeCILL §5.3.4), but it *would* pull kikuchipy's licensing story into further complexity.
- **No wheels at all** — the only artefact for 3.7.5 is `shtns-3.7.5.tar.gz` (source). **No `win_amd64`.** Requires a C compiler + FFTW at build time. **Effectively unusable as a kikuchipy dependency on Windows.** → Rule out.

### 4.3 Numba + FFT

Numba has **no** built-in `np.fft`/`scipy.fft` support in nopython mode (long-standing numba/numba#5864). Options:
- Do FFTs **outside** `@njit` with `scipy.fft` (PocketFFT, BSD-3) and keep numba for the loop-heavy Wigner-d / SO(3) cross-correlation kernels. **Recommended** — matches kikuchipy's existing idiom.
- `rocket-fft` (https://github.com/styfenschaer/rocket-fft, https://pypi.org/pypi/rocket-fft/json): makes numba aware of `numpy.fft`/`scipy.fft` via PocketFFT. **v0.3.1, BSD-3-Clause, requires_python >= 3.9, wheels for win_amd64 + manylinux_x86_64 + macOS x86_64/arm64 on cp39–cp313.** ⚠ **no cp314 wheels yet**, and kikuchipy's classifiers already declare Python 3.14 support → would be a release-blocking dep. Treat as optional/experimental only.

### 4.4 kikuchipy already does FFT — with scipy.fft (BSD-3), no new dep needed

- `src/kikuchipy/pattern/_pattern.py:22` — `from scipy.fft import fft2, fftshift, ifft2, ifftshift, irfft2, rfft2`; wrappers documented at lines 223, 239-243, 255-257, 276, 289-292, 300-302; `fft2(pattern)` at line 758.
- `src/kikuchipy/filters/fft_barnes.py:24` — `from scipy.fft import irfft2, next_fast_len, rfft2`; used at lines 45, 168, 171.
- No `numpy.fft`, no `pyfftw` anywhere in `src/kikuchipy`.
- `pyproject.toml` dependencies already include `scipy >= 1.7`, `numba >= 0.57`, `numpy >= 1.23.0`, `dask[array] >= 2021.8.1`, `orix >= 0.12.1`, `diffsims >= 0.5.2`, `h5py >= 2.10`, `scikit-image`, `scikit-learn`. `requires-python >= 3.10`; classifiers cover **3.10–3.14**.

**Net: a pure numpy/scipy/numba implementation adds zero new dependencies, zero new licences, and avoids the GPL-FFTW entanglement entirely.** `next_fast_len` (already imported in fft_barnes.py) is the natural way to honour the EMSphInx "2·bw−1 = product of small primes" rule.

---

## 5. ACTION ITEMS / RISKS (ranked)

1. **Patent, not copyright, is the real risk.** Confirm whether the 2019 CMU provisional matured into a granted patent (Google Patents / USPTO Patent Center; inventors Lenthe, Singh, De Graef; assignee Carnegie Mellon). If granted, contact CMU CTTEC (`innovation@cmu.edu`, 412.268.7393) before merging. GPLv2 gives no explicit patent grant; the "or later" clause arguably brings GPLv3 §11's explicit grant into play when the work is conveyed under v3, but this is an interpretation, not a guarantee. **Flag it in the PR; let pyxem maintainers decide (`pyxem.team@gmail.com`).** Note also that a clean-room reimplementation from the published paper does *not* mitigate patent exposure (only copyright exposure).
2. **Copyright is clean.** GPL-2.0-or-later → GPL-3.0-or-later is permitted. Requirements: keep CMU/Lenthe copyright notice in the ported files, add a "changed by … on …" notice (GPLv2 §2a), do not place the code in any BSD-3 area of kikuchipy, cite the Ultramicroscopy paper.
3. **Update the local clone first** (`git remote add upstream https://github.com/pyxem/kikuchipy.git; git pull upstream develop`) — it is missing `doc/dev/licensing_considerations.rst`, the single most relevant policy file.
4. **Open a kikuchipy issue before coding**, referencing EMSphInx issue #7 (hakonanes, 2020, unanswered).
5. **Branch off `develop`**, `git commit -s`, one coherent feature, PR → `develop`, with: numpydoc docstrings, type hints in signatures only, full pytest coverage (plus `.py_func()` variants for numba kernels), a `benchmarks/` entry, `CHANGELOG.rst` "Unreleased → Added" entry with PR link, symbol exported in `__init__.py`/`.pyi` stub, pre-commit clean (ruff/ruff-format), and name added to `src/kikuchipy/__init__.py` `credits` + `.zenodo.json` (with consent — it is your own name here, so simply do it).
6. **Do not depend on FFTW/pyfftw/shtns.** Use `scipy.fft` (already a dep, BSD-3). pyshtools (BSD-3, win_amd64 wheels cp39–cp314) is a viable *optional* accelerator but does not provide the square-Lambert grid EMSphInx uses. rocket-fft (BSD-3) lacks cp314 wheels — do not make it required.
