# Mission: spherical indexing in kikuchipy

kikuchipy gains a pure-Python, **CPU-only** implementation of EMSphInx
spherical indexing (Lenthe, Singh & De Graef, *Ultramicroscopy* 207 (2019)
112841): dynamical master patterns are transformed to spherical-harmonic
coefficients on EMSphInx's square Legendre grid, experimental EBSD patterns are
back-projected onto the sphere through kikuchipy's `EBSDDetector` geometry, and
orientations are found as the maximum of the SO(3) cross-correlation computed
with symmetry-reduced spectra and real 3-D FFTs, optionally refined by Newton's
method on analytic Wigner-d derivatives, with pseudo-symmetry prediction and
handling. Every non-GUI EMSphInx program becomes code-driven functionality that
runs from a notebook like the rest of kikuchipy, and a tutorial notebook
`doc/tutorials/spherical_indexing.ipynb` sits next to the Hough and dictionary
indexing tutorials.

## What we deliver

| EMSphInx program | kikuchipy equivalent |
|---|---|
| `mp2sht` | `EBSDMasterPattern.get_spherical_harmonics()` / `kp.indexing.MasterPatternHarmonics.from_master_pattern()`, `.save()` to `.sht` |
| `.sht` files (SHTdatabase) | `kp.load("*.sht")` → `EBSDMasterPattern` (io plugin `emsphinx_master_pattern`); `MasterPatternHarmonics.from_file()` |
| `IndexEBSD` | `EBSD.spherical_indexing()`, `kp.indexing.SphericalIndexer`, `EBSD.refine_orientation_spherical()` |
| `MasterXcorr` | `kp.indexing.find_pseudo_symmetry_operators()` (+ psymfile read/write, stereogram plot) |
| `sht2png` | `MasterPatternHarmonics.to_master_pattern(...).plot()`, `.plot_power_spectrum()`, `.describe()` |
| `PatternRepack` | `kp.indexing.write_emsphinx_patterns()` |
| `EBSPDims` | scan-grid probe in the `oxford_binary` reader (distinct beam x/y sets + irregular-grid diagnostic) |
| `ShtWisdom` | **not applicable** — `scipy.fft` (pocketfft) needs no planning; we ship `kp.indexing.fast_bandwidths()` instead |
| `EMSphInxEBSD` (GUI) | **out of scope** |
| CUDA/GPU indexer | **out of scope** — CPU multi-threaded path only |

Explicitly out of scope for v1 (documented, revisit later): EMSphInx ROI string
grammar (`roimask`), `Geometry::ecp()`, `.ctf` writer, IPF/XC PNG writers,
per-point projection centres in back-projection (single PC per call in v1), non-zero detector `azimuthal`/`twist` (raises, as EMSphInx does), EMsoft raw `.data` pattern input to the PatternRepack equivalent,
non-EBSD `.sht` modalities (ECP/TKD/Laue raise `NotImplementedError`),
multi-crystal `.sht` files, big-endian `.sht`, `.sht` versions other than 1.1.

## Success criteria

1. Reproduces EMSphInx's own C++ unit tests (`test/sht/{square_sht,wigner,sht_xcorr}.cpp`, `test/xtal/rotations.cpp` for the ZYZ/Bunge relations, `test/util/nml.cpp` for the namelist round trip) to their stated tolerances.
2. Agrees with `IndexEBSD.exe` (EMSphInx `master` @ 60f3517, CPU, `nthread=1`) on kikuchipy's Ni datasets: refined median misorientation < 0.2°, coarse median < 0.5°, scores Pearson r > 0.98; a kikuchipy-written `.sht` and a kikuchipy-repacked pattern file are accepted by the EMSphInx binaries.
3. Zero new required dependencies; follows kikuchipy's numpy/scipy/numba/dask conventions, numpydoc, lazy public API, tests, changelog, credits.
4. GPL-2.0-or-later notices preserved under kikuchipy's GPL-3.0-or-later; BSD-3 SHTfile notice preserved for the `.sht` codec; nothing here imported from kikuchipy's BSD-3 areas.
5. Delivered as small independently mergeable PRs, each with a dated spec folder, tests written first (real data where possible), an adversarial review, and a CHANGELOG entry.

## Legal status (recorded 2026-08-16)

- EMSphInx headers: "GPL v2 or (at your option) any later version", © 2019 De Graef Group, CMU, author W. C. Lenthe → may be conveyed under GPL-3.0-or-later. Required: keep the notice (incl. the CMU CTTEC commercial-licence contact), add the modification notice required by GPLv2 §2(a) / GPLv3 §5(a) ("changed by …, date").
- `.sht` codec: SHTfile repo `https://github.com/EMsoft-org/SHTfile` @ `e49ad6b`, **BSD-3-Clause** — copyright + conditions + disclaimer reproduced verbatim; no GPL relicensing claim.
- FFTW (GPL) is not used; `scipy.fft` (BSD-3) instead. miniz (MIT) not needed.
- Patent: EMSphInx README says "the central indexing algorithm is covered by a provisional patent application". The Google Patents queries run on 2026-08-16 (archived with query strings, counts and hits in `specs/_research/patent-search-2026-08-16.md`) surfaced **no granted or published patent** on this algorithm; the only filings surfaced for inventor Lenthe were 2024 Gatan/EDAX applications (WO2025184557A1, US20260002896A1) on other subjects. A 2019 provisional that was converted would have published by ~2021. This is a negative result from a bounded search, not proof; the residual uncertainty is flagged to the pyxem maintainers (issue + `pyxem.team@gmail.com`) before any upstream merge, and fork work proceeds.
- kikuchipy policy (`doc/dev/licensing_considerations.rst`): GPL-derived code may not be imported from BSD-3 files; every PR states that BSD opt-out is impossible for EMSphInx-derived modules.
