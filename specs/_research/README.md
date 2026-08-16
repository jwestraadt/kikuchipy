# Research notes for the spherical indexing (EMSphInx port) project

These files are the raw output of the ultracode planning workflow run on
2026-08-16 (13 agents; kikuchipy `develop` @ 49b1c11c, EMSphInx `master`
@ 60f3517). They are **working notes, not normative specification**: where a
note and a feature spec disagree, the spec wins; where two notes disagree, the
critiques win over the plans (they were written to refute them and verified
claims against the source).

| File | What it is |
|---|---|
| `explore-emsphinx-core-algorithm.md` | Line-accurate description of the SHT grids, `DiscreteSHT`, Wigner-d, SO(3) cross-correlation, refinement, detector back-projection, preprocessing, indexer flow and namelist. Start here for any numerical port. |
| `explore-emsphinx-programs-and-formats.md` | Every EMSphInx program (CLI vs GUI), the `.sht` v1.1 binary layout, pattern/vendor file readers, output formats, benchmarks. |
| `explore-emsphinx-xtal-util-vs-orix.md` | Capability map EMSphInx `xtal/`, `util/` → orix / kikuchipy / scipy; Euler conventions; environment versions. |
| `explore-kikuchipy-conventions.md` | Code style, tests, docs, data module, git/PR, optional-dependency and numba conventions extracted from `doc/dev/` and the indexing code. |
| `explore-real-data-and-tests.md` | Datasets, pooch cache, how existing indexing tests/notebooks use real data, EMSphInx binaries usage text, dataset recommendations. |
| `explore-licenses-and-community-process.md` | GPL-2.0-or-later → GPL-3.0-or-later analysis, patent statement, kikuchipy licensing policy, prior art (EMSphInx issue #7), SHT library survey. |
| `plan-*.md` | Three candidate implementation plans (faithful port / kikuchipy-native / incremental delivery). |
| `critique-plan-*.md` | Adversarial reviews of each plan; blockers there (Euler sign, `fastSize`, DCT type, BSD-3 `.sht` codec, single PC) are already folded into `specs/tech-stack.md`. |
| `critique-completeness-and-merged-roadmap.md` | Cross-plan gap list and the merged 12-branch roadmap adopted in `specs/roadmap.md`. |
| `patent-search-2026-08-16.md` | Archived Google Patents queries, counts and hits behind the patent statement in `specs/mission.md`. |

Line numbers refer to the commits above; re-verify against the files before
relying on them in a spec.
