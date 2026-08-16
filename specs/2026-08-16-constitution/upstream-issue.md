# Drafts: upstream issue and maintainer email (NOT yet sent)

Send only after the user has read and approved these. Post the issue from the
user's GitHub account (`gh issue create -R pyxem/kikuchipy`), and send the
email from the user's mail client to `pyxem.team@gmail.com`.

---

## GitHub issue for `pyxem/kikuchipy`

**Title:** Spherical indexing (EMSphInx port, CPU only): licence placement and patent statement — feedback wanted before PRs

**Body:**

Hi maintainers,

I am porting the EMSphInx spherical indexing algorithm (Lenthe, Singh & De Graef, *Ultramicroscopy* 207 (2019) 112841, https://github.com/EMsoft-org/EMSphInx) into kikuchipy as a pure-Python/NumPy/SciPy/Numba implementation of the CPU path (no CUDA), together with notebook-driven equivalents of the EMSphInx command-line programs (`mp2sht`, `IndexEBSD`, `MasterXcorr`, `sht2png`, `PatternRepack`, `EBSPDims`), an `.sht` master-pattern reader for `kp.load()`, and a tutorial notebook. This picks up the long-standing wish in EMsoft-org/EMSphInx#7 (Håkon, 2020). The work is being done as a series of small, independently reviewable feature branches on my fork with spec documents, tests written first against real data (`nickel_ebsd_small/large`, the Ni master pattern), regression tests against `IndexEBSD.exe` output, and no new required dependencies (`scipy.fft` replaces FFTW).

Before I open the first PR I would like your view on two points that `doc/dev/licensing_considerations.rst` asks reviewers to check:

1. **GPL-only placement.** The core algorithm modules will be line-for-line ports of EMSphInx headers, which are licensed *GPL-2.0-or-later* (© 2019 De Graef Group, Carnegie Mellon University, author W. C. Lenthe). I intend to convey them under kikuchipy's GPL-3.0-or-later, keeping the CMU notice verbatim plus the GPLv2 §2(a) "changed by …" line, in a private package `src/kikuchipy/indexing/_spherical/` with public names exported from `kikuchipy.indexing` and methods on `EBSD`/`EBSDMasterPattern`. **These modules cannot be offered under BSD-3** and must never be imported from kikuchipy's BSD-3 files. The `.sht` file codec, by contrast, derives from the separate SHTfile repository (https://github.com/EMsoft-org/SHTfile, BSD-3-Clause) and will carry that notice; the `.sht` reader plugin will import the GPL harmonics code and therefore be GPL too. Is a GPL-only contribution of this kind acceptable in `kikuchipy.indexing`, or would you prefer it under a different location/namespace?

2. **Patent statement.** The EMSphInx README states that "the central indexing algorithm is covered by a provisional patent application" (CMU, 2019). Searching Google Patents on 2026-08-16 (inventor/assignee/keyword queries: Lenthe, Singh, De Graef, Carnegie Mellon, "electron backscatter" + "spherical harmonic"; queries and hits archived in my fork under `specs/_research/patent-search-2026-08-16.md`) surfaced no granted or published patent on this algorithm; a 2019 provisional that was converted would have published by ~2021, so it appears to have lapsed, but I cannot prove a negative. GPLv2 gives no explicit patent grant (GPLv3 §11 does, when conveyed under v3). Do you want this resolved with CMU (CTTEC, innovation@cmu.edu) before a merge, or is a documented note in the module and the changelog sufficient for you?

If both are fine in principle, the planned PR sequence into `develop` is: (1) square-Legendre grid + discrete SHT, (2) master-pattern harmonics + `.sht` I/O, (3) Wigner-d tables, (4) SO(3) cross-correlation, (5) detector back-projection, (6) `EBSD.spherical_indexing`, (7) Newton refinement, (8) pseudo-symmetry (MasterXcorr), (9) visualisation/interop (sht2png, PatternRepack, EBSPDims, namelists), (10) EMSphInx regression data, (11) tutorial. Happy to adjust to whatever makes review easiest for you.

Thanks — Johan

---

## Email to `pyxem.team@gmail.com`

**Subject:** kikuchipy: GPL-derived spherical indexing contribution (EMSphInx port) — licence/patent question

Dear pyxem team,

I have opened pyxem/kikuchipy#<issue-number> describing a port of the EMSphInx spherical indexing algorithm into kikuchipy. `doc/dev/licensing_considerations.rst` suggests emailing you with licensing questions, so in short:

- The ported core is GPL-2.0-or-later (CMU / W. C. Lenthe) and will be conveyed under kikuchipy's GPL-3.0-or-later, notices kept, in GPL-only modules under `kikuchipy.indexing`; it cannot be BSD-licensed and will not be imported from BSD files. The `.sht` codec is BSD-3 (SHTfile) and keeps that notice.
- EMSphInx's README mentions a 2019 CMU provisional patent application on the algorithm; I found no granted or published patent (Google Patents, 2026-08-16). Please tell me whether you want CMU contacted before a merge, or whether a documented note suffices.

Full details and the planned PR sequence are in the issue. Thank you for considering this.

Best regards,
Johan Westraadt
