# Phase 2 -- `sht-master-spectra-and-file`: validation

## Automated (default suite; run from Git Bash)

```
uv run pytest tests/test_indexing/test_spherical_sht_file.py tests/test_indexing/test_spherical_symmetry.py tests/test_indexing/test_spherical_master_pattern_harmonics.py tests/test_io/test_emsphinx_master_pattern.py tests/test_signals/test_ebsd_master_pattern.py -k "spherical or sht or emsphinx or SphericalHarmonics" -n 0   # first run: warm the Phase 1 numba cache in one process
uv run pytest tests/test_indexing tests/test_signals tests/test_io -k "spherical or sht or emsphinx" -n 4
uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical src/kikuchipy/signals/ebsd_master_pattern.py
uv run pytest --cov=kikuchipy.indexing._spherical --cov=kikuchipy.io.plugins.emsphinx_master_pattern --cov-report=term-missing tests/test_indexing/test_spherical_sht_file.py tests/test_indexing/test_spherical_symmetry.py tests/test_indexing/test_spherical_master_pattern_harmonics.py tests/test_io/test_emsphinx_master_pattern.py
KIKUCHIPY_EMSPHINX_DIR=/c/Users/westraadt.1/Repos/EMSphInx uv run pytest tests/test_indexing/test_spherical_sht_file.py tests/test_indexing/test_spherical_master_pattern_harmonics.py -k emsphinx_binaries    # local-gated
uv run pre-commit run --files <the new/changed .py files> pyproject.toml .pre-commit-config.yaml
```

Local-gated tests skip (with the reason in the skip message) unless `KIKUCHIPY_EMSPHINX_DIR` points at an EMSphInx checkout with `build/Release/sht2png.exe` (or `sht2png` on POSIX) and `data/Ni {20kV 75.7deg}.sht`. The 25 synthetic `.sht` files come from the session fixture `emsphinx_synthetic_sht_files` (`_dummy_files/emsphinx_sht.py`, D16), not from the package data.

Required assertions (each is a named test):

`.sht` codec (`test_spherical_sht_file.py`)
- CRC-32C: `crc32c(b"") == 0`, `crc32c(b"\x00"*8) == 0xEBE76DE3`, `crc32c(b"123456789") == 0xF28417BE` (differs from standard CRC-32C 0xE3069283 -- the SHTfile variant is asserted, not the standard); chaining `crc32c(b, crc32c(a)) == crc32c(a + b)`; the generated 256-entry table equals the literal table of `sht_file.in.hpp:967-1000` for the eight spot entries `LUT[0] == 0`, `LUT[1] == 0x0A5F4D75`, `LUT[2] == 0x14BE9AEA`, `LUT[128] == 0x1EDC6F41`, `LUT[255] == 0x12A28EAD`, `LUT[16] == 0x15DECED9`, `LUT[64] == 0x11B258E1`, `LUT[192] == 0x0F6E37A0` (first entry of the 25th literal row), `LUT[200] == 0x1B5D3F8D` (first entry of the 26th) and `sum(LUT) == 68719476608` (~~`2**36`~~ **not** `2**36`, which is 68719476736, 128 larger -- corrected 2026-08-16 after re-measuring the generated and the literal table, both 256/256 identical; a plain `assert`, determined 2026-08-16); the CRCs of the two in-package Ni files are `0xE3100CFF` (`ni_small_20kv_bw384.sht`) and `0xEA2875D2` (`ni_20kv_bw384.sht`); `crc32c` accepts `bytes`, `bytearray` and `memoryview` with equal results, and its wall time on the 74 828 B shipped file is recorded (`record_property`; measured 3.9 ms with the plain-Python `tuple` LUT, 51 ms with NumPy scalars -- the former is required, D10).
- Space-group LUTs: length 230 each; values of `space_group_z_rotation` in `{1, 2, 3, 4, 6}`, of `space_group_compression_flags` in `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xA}` and never `0x4 | 0x8`; `sum(rot) == 707`, `sum(cmp) == 948`, histogram `{1: 15, 2: 91, 3: 30, 4: 72, 6: 22}`; spot values `1 -> (1, 0x0)`, `2 -> (1, 0x1)`, `6 -> (1, 0x4)`, `10 -> (1, 0x5)`, `16 -> (2, 0x0)`, `25 -> (2, 0x4)`, `47 -> (2, 0x7)`, `75 -> (4, 0x0)`, `83 -> (4, 0x3)`, `111 -> (2, 0x8)`, `115 -> (2, 0x4)`, `123 -> (4, 0x7)`, `143 -> (3, 0x0)`, `156 -> (3, 0x8)`, `157 -> (3, 0x4)`, `164 -> (3, 0x9)`, `174 -> (3, 0x2)`, `187 -> (3, 0xA)`, `189 -> (3, 0x6)`, `225 -> (4, 0x7)`; exactly 25 distinct `(zRot, cmpFlg)` pairs; `sg` outside `[1, 230]` raises `ValueError`. Cross-check against orix for `sg in 16..230`: `rot == 1 + #proper elements about z`, `bool(cmp & 1) == contains_inversion`, `bool(cmp & 2) == improper 2-fold about z`, `bool(cmp & 0xC) == improper 2-fold with axis in the equator`; and for `sg in 3..15` the recorded **disagreement** is asserted (`rot == 1` while orix says 2 for 3-5 and 10-15; for 6-15 the LUT sets `0x4`, a mirror plane *containing* z (b-unique `m` perpendicular to b), where orix's z-unique `Cs`/`C2h` have the mirror plane *perpendicular* to z, i.e. bit `0x2` -- the two mirror kinds are not the same flag) -- the test docstring names the b-unique vs z-unique cause.
- `num_harmonics(384, 4, 0x7) == 9312` (`96*97`), `num_harmonics(bw, 1, 0) == bw*(bw+1)` (complex, `2 * bw(bw+1)/2`), `num_harmonics(bw, 1, 0x4) == bw*(bw+1)//2` -- these closed forms are the independent pins; and, against the only external oracle for the count, `doub_cnt == num_harmonics(bw, z_rot, flags)` and `len(payload) * 8 == file_size - payload_offset - 4` for **every** `.sht` in the suite (the two shipped Ni files, the shipped EMSphInx file when available, the 25 generated fixtures: 25 distinct flag pairs); `pack_harmonics` output length equals `num_harmonics` for all 25 pairs and `bw in {4, 16, 17}` (self-consistency, recorded as such).
- Pack/unpack: for each of the 25 pairs at `bw in {16, 17}` with a rule-respecting synthetic `alm` (D16 recipe): `unpack(pack(alm)) == alm` bit-exact and `pack(unpack(p)) == p` bit-exact; branch coverage asserted by construction: pair `(1, 0x0)` writes `2 * count` doubles (complex), `(2, 0x4)` writes `count` doubles that equal the real parts, `(2, 0x8)` writes real parts for `m % 4 == 0` rows and imaginary parts for `m % 4 == 2` rows (checked entry by entry against `alm`), `(4, 0x7)` skips rows `m % 4 != 0`, odd `l` and odd `l + m`; flags `0x0C` raise `ValueError` in all three functions; for the three Ni files `pack(unpack(payload)) == payload` bit-exact and `unpack` gives non-zero entries only at `m % 4 == 0`, even `l`, real values.
- Header parse of `ni_small_20kv_bw384.sht`: `magic b"*sht"`, version `(1, 1)`, `software_version == "ve49ad6b"`, `modality == 1`, `beam_energy == pytest.approx(20.1, abs=1e-6)`, `primary_angle == 70`, `secondary_angle == 0`, `doi == "https://doi.org/10.1016/j.ultramic.2019.112841"` (`doi_len 46`), `notes == "created with mp2sht"` (`note_len 19`), `num_xtal == 1`, `sg_eff == 225`, `pijk == 1`, `rot_sense == 112`, `vendor == 1`, `sim_meta_size == 88`, crystal `sg_num 225, sg_set 1, sg_axis 1, sg_cell 1`, origin `(0, 0, 0)`, `lat == approx((0.35236,)*3 + (90,)*3, rel=1e-6)`, `rot == (1, 0, 0, 0)`, `weight == 1`, `num_atoms == 1`, string lengths `(2, 0, 0, 0, 0)`, `formula == "Ni"`, atom `(x, y, z) == (0, 0, 0)`, `occ == 1`, `charge == 0`, `debye_waller == approx(0.0035)`, `atomic_number == 28`; EMsoftED `emsoft_version == "5_0_0_0"`, `sig_start == 70`, `sig_end`/`sig_step` NaN, `omega 0`, `kev approx 20.1`, `e_hist_min 20`, `e_bin_size 1`, `depth_max 100`, `depth_step 1`, `thickness == inf`, `tot_num_el == 2_000_000_000`, `num_sx == 201`, `c1, c2, c3 == 4, 8, 50`, `sig_db_diff == 1`, `d_min approx 0.05`, `num_px == 200`, `lat_grid_type == 1`; harmonics `bandwidth 384, z_rot 4, flags 0x7, doub_cnt 9312`; block offsets `112, 120, 232, 320, 328, 74824`; file size 74 828. Same for `ni_20kv_bw384.sht` with `beam_energy 20.0`, `e_hist_min 5`, `num_sx 501`, `num_px 500`, `kev 20.0`.
- Shipped EMSphInx `data/Ni {20kV 75.7deg}.sht` (env-gated): as above with `beam_energy == approx(20.0)`, `kev == approx(20.0)` (not 20.1: a different master), `primary_angle == approx(75.7, abs=1e-5)`, `sig_start == approx(75.7)`, `e_hist_min 5`, `num_sx 501`, `num_px 500`, CRC `0xF2AF93EF`, `doub_cnt 9312`; the payload unpacks with `a_00 == approx(-3.2555, abs=1e-3)`.
- Byte identity: `sht_file_to_bytes(read_sht(f)) == open(f, "rb").read()` for the two in-package Ni files (and the shipped file when available) -- external oracles; for the 25 generated fixtures the assertion is `write -> read -> write` idempotence (`sht_file_to_bytes(read_sht(sht_file_to_bytes(f))) == sht_file_to_bytes(f)`) plus `md5(file) == PINNED[sg]` from the 25-entry dict recorded below after the one-off `sht2png.exe` acceptance (D16; a writer that produces the accepted bytes cannot drift unnoticed, and a legitimate byte change re-runs `sht2png.exe` and re-pins).
- Dict views: `sht_file_to_bytes(ShtFile.from_dict(f.to_dict())) == sht_file_to_bytes(f)` for the two Ni files (lossless `to_dict`); `f.metadata_dict()` has exactly the keys `header, master_pattern, crystals, simulations, harmonics`, `metadata_dict()["crystals"] == {"crystal_0": {...}}` and `["simulations"] == {"simulation_0": {...}}` (numbered nodes, no lists), no key ends in `_bytes`, and `"packed" not in metadata_dict()["harmonics"]`; a two-crystal synthetic file gives `crystal_0`, `crystal_1`.
- Robustness: flipping one payload byte -> `ValueError` matching "checksum"; `check_crc=False` then parses; magic `b"*SHT"` -> `NotImplementedError` matching "big-endian"; version `(1, 0)` -> `NotImplementedError` matching "1.1"; magic `b"HDF\x89"` -> `ValueError` matching "not an SHT"; a synthetic file with `modality 0x21` (Laue), `vendor 1`, `sim_meta_size 32` and 32 arbitrary bytes round-trips byte-identically with the record kept as `bytes` (EMSphInx itself would refuse it, `sht_file.in.hpp:1605`); a synthetic two-crystal file round-trips; string padding `doi` lengths 0, 1, 7, 8, 9, 46 -> padded 0, 8, 8, 8, 16, 48 with the unpadded length in the field; **raw strings**: a synthetic file whose `notes` bytes are `b"caf\xe9\x00\x00\x00\x01"` (`noteLen 4`, non-UTF-8 byte, non-zero pad) reads with `notes == "caf�"`, `notes_bytes == b"caf\xe9\x00\x00\x00\x01"`, verifies its CRC, and `read -> sht_file_to_bytes` is byte-identical (a decode/re-encode writer would change byte 4 and byte 7 and the CRC); `write` refuses, each with the `File::sanityCheck` wording: `bandwidth > 32767` (ours), `beam_energy < 0` ("negative beam energy is non-physical"), `beam_energy 10001` ("10 MeV beam energy is unrealistic"), `primary_angle 361`, `secondary_angle -361`, non-zero `res_bytes`/`res_bytes2` ("non-zero reserved bytes"/"reserved bytes must be 0"), `modality 5` ("invalid modality flag"), `vendor 2` ("invalid vendor flag"), `sg_eff 0`, `pijk 0`, `rot_sense 98`, `num_xtal 2` with one crystal ("# crystals != crystals size"), `num_xtal 0` (ours: EMSphInx UB), `sim_meta_size 88` with a `None` record ("NULL simulation data for nonzero size"), `sim_meta_size 0` with a record ("non-NULL simulation data for 0 size"), `sim_meta_size 32` with an 88-byte record ("simulation data size doesn't match header size"), an EMsoftED record under `modality 0x21` ("simulation data modality not valid for master pattern modality" / "file modality doesn't match simulation modality"), record vendor 0 under `vendor 1` ("simulation data vendor doesn't match master pattern verndor"), a `doi_bytes` of length 7 for `doi_len 7` ("doi string doesn't match length"), `notes` idem ("noites string doesn't match length"), `doub_cnt != num_harmonics` ("harmonics count doesn't match compression parameters").
- Licence hygiene: `_sht_file.py`'s module-level imports are a subset of `{numpy, dataclasses, struct, math, pathlib, typing, io, os, warnings}` (nothing from `kikuchipy` outside itself); `.pre-commit-config.yaml`'s BSD hook regex matches `src/kikuchipy/indexing/_spherical/_sht_file.py` and the GPL hook's `exclude` matches it (both asserted with `re.search` on the YAML text).
- Local-gated, EMSphInx binaries: `sht2png.exe <fixture> <tmp>/leg.png` exits 0 and prints `effective sg# N` for each of the 25 generated fixtures (their md5s must equal the pinned ones, i.e. the acceptance and the pin refer to the same bytes) and `effective sg# 225` for a `.sht` written by `save()` from the in-package Ni master (`bandwidth=384`, default `emsphinx_compatible=True`); for that file the Legendre PNG (`imageio.v3.imread`, shape `(387, 774)`, uint8) equals our own `SphericalHarmonicTransform(384, "legendre", 387).synthesize(alm)` rescaled with sht2png's rule (`round((v - min_north) * 255 / (max_north - min_north))`, both hemispheres with the **north** min/max, `sht2png.cpp:94-98`) to `max |diff| <= 1` grey level.

Symmetry (`test_spherical_symmetry.py`)
- `set(Z_ROTATION_ORDER_AND_MIRROR) == {g.name for g in orix.quaternion.symmetry._groups} | {"2", "m"}` (40 keys); for every key the value equals the operator oracle (`n_fold == 1 + #{proper elements with |axis_z| == 1 and angle > 0}`, `mirror == any(improper element with |axis_z| == 1 and angle == pi)`) computed on the matching `Symmetry` (`_groups` entry, or `C2`/`Cs` for `"2"`/`"m"`); table spot values `"m-3m" -> (4, True)`, `"-4" -> (2, False)`, `"-6" -> (3, True)`, `"-6m2" -> (3, True)`, `"-3m" -> (3, False)`, `"11m" -> (1, True)`, `"1m1" -> (1, False)`, `"112" -> (2, False)`, `"121" -> (1, False)`, `"2" -> (2, False)`, `"m" -> (1, True)`, `"2/m" -> (2, True)`, `"23" -> (2, False)`, `"m-3" -> (2, True)`.
- `{get_point_group(sg).name for sg in 1..230}` (32 names) is a subset of the keys; `point_group_flags(None) == (1, False)`; `point_group_flags("112/m")` raises `ValueError` listing the known names.
- `space_group_for_point_group`: `"m-3m" -> 221`, `"432" -> 207`, `"2" -> 3`, `"112" -> 3`, `"121" -> 3`, `"m" -> 6`, `"11m" -> 6`, `"2/m" -> 10`, `"32" -> 149`, `"321" -> 150`, `"312" -> 149`, `"3m" -> 156`, `"-3m" -> 162`, `"-6m2" -> 187`, `"-42m" -> 111`, `"1" -> 1`, `"-1" -> 2`; for the 32 `get_point_group` names `get_point_group(result).name == name`; `candidate_space_groups`: `"3m" -> (156, 157)`, `"-3m" -> (162, 164)`, `"-42m" -> (111, 115)`, `"-6m2" -> (187, 189)` (distinct `(zRot, cmpFlg)` pairs, D11), `"m-3m" -> (221,)`, `"32" -> (149,)` (149/150 share `(3, 0x0)`), `"mm2" -> (25,)`.
- `validate_flags` (`SYMMETRY_POWER_TOLERANCE == 1e-8`, relative power with `m > 0` doubled): Ni mp2sht coefficients keep `(4, True)` with `systematic_zero_power <= (1e-20, 1e-20)`; a synthetic 4-fold `alm` with row `m = 2` filled -> `n_fold 2` (largest satisfied divisor) and a `UserWarning` matching "n_fold" and "2"; row `m = 1` filled -> `1`; ~~a 6-fold `alm` with row `m = 3` filled -> `2`, with row `m = 2` filled -> `3`~~ a 6-fold `alm` with row `m = 3` filled -> `3`, with row `m = 2` filled -> `2` (corrected 2026-08-16: the two expected values were swapped; the largest divisor of 6 dividing every non-zero order is 3 for `{0, 3, 6, 12, 18}` and 2 for `{0, 2, 6, 12, 18}`, per the `6 -> 3 -> 2 -> 1` ladder of `plan.md` line 49), with rows `2` and `3` filled -> `1`; odd `(l+m)` filled -> `(..., False)` and a warning matching "equatorial mirror"; boundary: relative power `0.9e-8` passes, `1.1e-8` fails (`<=`; an exact `1e-8` cannot be constructed in floating point and is not asserted).

Harmonics (`test_spherical_master_pattern_harmonics.py`)
- Container: `alm` copied and C-contiguous complex128; non-square, 3-D or non-zero `l < m` input -> `ValueError`; `bandwidth == alm.shape[0]`.
- `_resize_lambert`: constant `c` -> `c * 2*dim**2/new_dim**2` everywhere with `rtol 1e-12` for `(dim, new_dim) in {(13, 21), (21, 13), (401, 547), (1001, 547)}` (the last on a synthetic 1001 image); the DCT mode `cos(pi*k*(n + 1/2)/dim)` outer product for `k = (2, 3)` at `dim 21 -> 31` equals `2*21**2/31**2` times the same mode evaluated on the 31 grid to `atol 1e-12`; `new_dim == dim` returns the input times exactly 1 (early return, `master.hpp:356`; without it a constant would come back doubled, since the factor is `2*dim**2/new_dim**2 == 2` at `new_dim == dim`); negative control (documented in the test, re-measured 2026-08-16): using `scipy.fft.idctn(type=3)` instead of `dctn(type=3)` does **not** give a constant at all -- for the constant `3.7` on `13 -> 21` the result ranges `3.6e-5 .. 6.4e-3` against the correct `2.836`, with a maximum ratio of exactly `1/new_dim**2` (`= 1/441`, because only `X[0, 0]` is non-zero and `idctn` divides that term by `N**2`), not `1/(2*new_dim)**2` -- and DCT-III is not proportional to the inverse DCT-II in general, so the ratio is only constant for the constant-image probe.
- `_to_legendre`: constant image -> constant on both hemispheres (1e-14); the plane `f(X, Y) = 0.3 X - 0.7 Y + 0.1` sampled on the 547 grid is reproduced at every Legendre normal of `dim_leg 387` to 1e-12 (bilinear is exact for bilinear functions); an asymmetric probe (`f = X` vs `f = Y`) is *not* symmetric under transposition (row/column order locked: north pixel `(j, i)` with `i > dim/2`, `j == dim/2` samples the source column direction); the same `(X, Y)` is used for north and south (`_to_legendre(f, g)` and `_to_legendre(g, f)` swap outputs exactly).
- Normalisation: after `_normalize_hemispheres(..., emsphinx_compatible=False)` (opt-in) the weighted mean is 0 (`abs 1e-12`) and the weighted variance 1 (`rel 1e-12`) with the correctly halved weights; with `True` (default) the weighted mean is `-mu` (i.e. the pattern was shifted by `2 mu`, `rel 1e-10`) and the corner weight is `omega_eq / 4`, the other border weights `omega_eq / 2`; ~~the two settings differ by a global factor `sqrt(1 + mu^2/sigma^2)` and a shift (rel 1e-10)~~ the two settings satisfy `plain == compat * (sigma_c / sigma_p) + (2 mu_c - mu_p) / sigma_p` exactly (1e-12) with the scalars each returns, and their *coefficients* agree after `remove_dc` and one global rescaling (rel-L2 < 1e-8 on the Ni master) (corrected 2026-08-16: the two settings weight the corners differently, `omega/4` against `omega/2`, so `mu_c != mu_p` and the short form `(compat + mu/sigma) * sqrt(1 + mu_p^2/sigma_p^2) == plain` is only true to 3.3e-4, max rel 0.30, not 1e-10).
- **`normalize=False` through the public keyword** (`test_normalize_false_reproduces_emsphinx_amplitude`): on the 13 px `master_patterns.h5` (`energy_weights=np.ones(11)`, `bandwidth=4`: `dim_leg 7`, `dim_scaled 10`, crop branch) `from_master_pattern(..., normalize=False).alm` equals `SphericalHarmonicTransform(4, "legendre", 7).analyze(*_to_legendre(north, south, 7))` computed in the test from the energy-weighted hemispheres bit-exactly (same code path, no normalisation step); on a synthetic constant master (`c = 5`, 21 px, both hemispheres, `bandwidth=8`: `dim_leg 11`, `dim_scaled 16`) `normalize=False` gives `a_00 == sqrt(4 pi) * c * 2*21**2/16**2` to `rel 1e-10` and all other coefficients `< 1e-10` (D5's amplitude factor is observable only here) -- while `normalize=True` gives `|a_00| < 1e-10`; and the same 21 px constant master at `bandwidth=12` (`dim_leg 15`, `dim_scaled = round(sqrt(2) * 15) = 21 == dim`, the early-return branch) returns `a_00 == sqrt(4 pi) * c` to `rel 1e-12` (the input is left unchanged, not doubled -- the `2*dim**2/new_dim**2 == 2` factor never runs).
- Integer multi-site guard: an `EBSDMasterPattern` of dtype uint8 whose `original_metadata` carries `CrystalData.Natomtypes 2` and `EBSDMasterNameList.combinesites 0` -> `ValueError` matching "combinesites"; the same with `combinesites 1`, or float32 data, or no such keys, passes; the in-package Ni master (`Natomtypes 1`) passes.
- Energy weights: single-energy master -> `[1.0]`; a temporary HDF5 with `EMData/MCOpenCL/accum_e` of shape `(5, 5, 4)` and counts summing per bin to `[10, 0, 30, 60]`, `Ehistmin 10`, `Ebinsize 1`, energy axis `[10, 11, 12, 13]` -> `[0.1, 0, 0.3, 0.6]`; energy subset `[12, 13]` -> `[1/3, 2/3]`; energy `9` -> `ValueError` (bin out of range); no `accum_e` -> `ValueError` matching "energy_weights"; explicit `[1, 3]` -> `[0.25, 0.75]`; `[-1, 2]` -> `ValueError`; wrong length -> `ValueError`; on the cached full Ni master (weekly) the bin counts equal `[0, 0, 1455, 266084, 5365299, 18896238, 27431857, 34257097, 42689288, 53576717, 68256443, 89645683, 124471882, 191778041, 334039769, 161050771]` (total 1151726624), i.e. weights `counts / 1151726624` (`weights[-1] == approx(0.1398342, rel=1e-6)`, `sum == 1`).
- Pipeline warnings: `from_master_pattern(bandwidth=384)` on the 401 px master warns `UserWarning` matching "384" and "200"; `bandwidth=200` does not warn; `bandwidth=0` and `40000` -> `ValueError`; on the 13 px `master_patterns.h5` (with `energy_weights=np.ones(11)`) `bandwidth=4` runs (dim_leg 7, dim_scaled 10, crop branch) and `bandwidth=8` warns.
- **mp2sht parity** (default suite): `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both").get_spherical_harmonics(bandwidth=384)` (the **default** `emsphinx_compatible=True`, no keyword passed) under `pytest.warns` vs `MasterPatternHarmonics.from_file(ni_small_20kv_bw384.sht)`: `rel-L2 < 1e-6` over all coefficients and after `remove_dc()` on both (measured 2.2e-9 / 5.1e-9); `a_00` agrees to `abs 1e-8` (measured 7e-12); relative power in the entries the file does not store `< 1e-20` (measured 6.4e-30); `max |imag| < 1e-12` (measured 3.4e-16); `beam_energy == approx(20.1)`, `sample_tilt == 70`, `n_fold == 4`, `has_equatorial_mirror`. Opt-in branch, the same call with `emsphinx_compatible=False`: `|a_00| < 1e-3` asserted (measured 6.3e-5), `power_spectrum().sum()` within 1 % of `4 pi` asserted, and the global scale factor vs the file `== approx(1.854, rel=1e-3)` recorded via `record_property`.
- `to_master_pattern`: default `dim == 769`, data shape `(2, 769, 769)` float64, axes `hemisphere/height/width` with offsets `0/-384/-384` (centred, `-(dim//2)`), `projection == "lambert"`, `hemisphere == "both"`, `phase is h.phase`, `original_metadata == h.original_metadata`; **offset convention locked**: at `dim=401` the height/width offset is `-200` while `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both").axes_manager["height"].offset == -201` (kikuchipy's EMsoft reader, `-sy // 2`) -- the one-pixel difference is asserted, not discovered (D13); north hemisphere Pearson `r > 0.99` vs the source uint8 north hemisphere bilinearly upsampled to 769 (`scipy.ndimage.map_coordinates(order=1)` on `mgrid * 400/768`; measured 0.9963); `dim=401` warns `UserWarning` matching "200" and gives `r > 0.95` vs the source (measured 0.9704); `dim=768` or `1` -> `ValueError`; `hemisphere="upper"` shape `(401, 401)` at `dim=401` and equals `data[0]` of `"both"`, `hemisphere="lower"` equals `data[1]` and the signal's `hemisphere == "lower"`, `hemisphere="north"` -> `ValueError` listing `upper, lower, both`; north and south are bit-identical for the Ni file (`north == south` in the source); `EBSDMasterPattern.get_patterns(rotations, detector)` on the `dim 769` result vs the same call on the source master: NCC `> 0.99` for three rotations (`Rotation.from_euler(deg2rad([[10, 20, 30], [120, 45, 60], [200, 80, 300]]))`, `EBSDDetector(shape=(60, 60), pc=(0.42, 0.22, 0.50), sample_tilt=70)`; measured 0.9963-0.9974); the `_sht.py` amendment is exercised (`SphericalHarmonicTransform(384, "lambert", 769)` constructs; `analyze` on it raises `ValueError`).
- `save`/`from_file`: `save(tmp)` then `from_file(tmp)`: `alm` equal to `atol 0` on the kept entries (bit-exact, packing is lossless), `beam_energy`/`sample_tilt` equal to float32 precision, `phase.space_group.number == 225`, lattice `0.35236` (nm, as stored), one atom `Z 28`, `original_metadata` crystal/simulation fields equal (`sig_start 70`, `num_sx 201`, `num_px 200`, `tot_num_el 2e9`, `lat_grid_type 1`, `emsoft_version == "unknown"`), the raw `softwareVersion` field of the written file `== ("kp" + kikuchipy.__version__)[:8].encode().ljust(8, b"\x00")` byte-for-byte (`b"kp0.14.d"` at 0.14.dev0) and, with `kikuchipy.__version__` monkeypatched to `"1.0.0"`, `== b"kp1.0.0\x00"` (NUL, not space, padded); `notes == "created with kikuchipy (normalize=True, emsphinx_compatible=True)"` for the default (parity) instance, `"(normalize=True, emsphinx_compatible=False)"` for the opt-in instance, `"created with kikuchipy"` for `MasterPatternHarmonics(alm, phase=...)` built directly, `doi == ""`; field-by-field equality with `ni_small_20kv_bw384.sht` for every header/crystal/simulation field **except** `software_version`, `doi`, `notes`, `emsoft_version`, crystal `name`, and the payload agrees as in the parity test; `preserve_header=True` on an instance from `from_file`: bytes identical to the source file for the two Ni fixtures, one generated fixture and the non-UTF-8 synthetic file of the codec tests (`open(...).read() ==`), and CRC equal; `preserve_header=True` on an instance from `from_master_pattern` -> `ValueError`; a phase without `space_group` uses the fallback (`Phase(point_group="m-3m")` -> `sg_eff 221`); no `space_group` and no `point_group` -> `ValueError`; **lossy packing** (`strict`, D11): `MasterPatternHarmonics(alm_with_row_m1, phase=Phase(space_group=225))` -> `save` raises `ValueError` matching "symmetry", `save(strict=False)` warns `UserWarning` matching "dropped" and writes a file that unpacks without row 1; **guard agreement**: the Ni coefficients perturbed in row `m = 2` at relative amplitude `1e-6` construct with `n_fold 4` and save silently, at `1e-3` construct with a warning and `n_fold 2` and `save()` raises / `save(strict=False)` warns -- both guards use `SYMMETRY_POWER_TOLERANCE`; **flag-ambiguous fallback**: a `Phase(point_group="3m")` whose coefficients carry the 157 (`cmp 0x4`) storage pattern -> `save` raises `ValueError` matching "156" and "157", and `Phase(space_group=157)` with the same coefficients saves; a `Phase(space_group=3)` (`"2"`, z-unique) with coefficients having an equatorial mirror but no z 2-fold saves fine (`zRot 1`, `cmp 0x0`) while `Phase(space_group=6)` (`cmp 0x4`, real rows) with complex rows raises; atoms at fractional `1/3, 2/3, 1/6, 5/6, 0.25` round-trip to `8, 16, 4, 20, 6` in the file and back to the fractions to float32 precision (`rel 1e-7`); `overwrite=False` on an existing file leaves it unchanged; a filename without suffix gets `.sht`.
- `resize`: `resize(400).resize(384).alm == alm` bit-exact; `resize(100).alm == alm[:100, :100]`; `resize(400).alm[:384, :384] == alm` and zeros elsewhere; `original_metadata` copied with `harmonics.bandwidth` updated; `resize(0)` -> `ValueError`.
- `remove_dc`: only `[0, 0]` changes; original untouched.
- `power_spectrum`: for `alm` with a single unit entry at `(m, l) = (3, 5)` -> `P[5] == 2`, others 0; at `(0, 2)` -> `P[2] == 1`; **Parseval** with an exact quadrature (re-specified: Phase 1's Mazonka weights are `lambert_solid_angles`, Lambert-only, and are only ~1e-3 accurate at low `dim`): a fixed band-limited `alm` (bw 16, `l >= m` entries `uniform(-1, 1)` from `default_rng(0)`, `m = 0` real) synthesized with `SphericalHarmonicTransform(16, "legendre", 67)` and integrated with the per-pixel Gauss-Legendre weights `W = quadrature_weights(67, "legendre")[0][ring_number(67)]` (Phase 1; equator already halved): `sum(P) == 4 pi * (sum(W*north**2) + sum(W*south**2)) / (2 * sum(W))` to **`rel 1e-8`** (measured 2026-08-16 with the Phase 1 code: `6.1e-11` at `dim 67`, `4.7e-8` at `dim 35`, `2.7e-5` at `dim 19` -- the EMSphInx-sized grid, where the pole rings undersample high `m` and the Legendre round trip is itself only 5e-4, Phase 1 -- and `6.8e-14` at `dim 131`; the oracle is the quadrature, independent of the `power_spectrum` formula); on the Ni file `sum(P) == approx(4 pi, rel=1e-2)` (measured rel 8.7e-4).
- `describe()`: contains `"file version 1.1"`, `"software version ve49ad6b"` (file instance) / `"kp"` (built instance), `"EBSD"`, `"20.1"`, `"70"`, `"effective sg# 225"`, `"pijk = 1"`, `"88 bytes"`, `"EMsoft"`, `"sg 225 setting 1"`, `"0.35236"`, `"Ni"`, `"28:"`, `"0.0035"`, `"square lambert"`, `"bandwidth 384"`, `"zRot 4"`, `"cmpFlg 0x7"`, `"9312"`, `"n_fold 4"`, `"equatorial mirror True"`, `"a_00 = -2.985"` and `"DC power fraction 0.71"` (mp2sht file and a default-built instance) / `"DC power fraction 0.00"` (an instance built with `emsphinx_compatible=False`).
- `rotate(Rotation.identity())` -> `NotImplementedError` matching "Phase 3"; `__repr__` equals `"MasterPatternHarmonics: bw = 384, ni (m-3m), 20.1 keV, 70.0 deg"` for the in-package master.
- Symmetry guard on construction: `MasterPatternHarmonics(alm_ni, phase=ni_phase)` keeps `(4, True)`; `MasterPatternHarmonics(alm_ni_with_row_m1, phase=ni_phase)` warns and gives `n_fold == 1`; `alm_ni_with_row_m2` warns and gives `n_fold == 2` (divisor downgrade); `phase=None` -> `(1, False)` silently.
- Hemispheres and types: `"upper"` + centrosymmetric phase gives the same coefficients as `"both"` with `south = north[::-1, ::-1]` (bit-exact on a synthetic asymmetric north with `Phase(point_group="-1")`, and equal to the `"both"` result on Ni to 1e-14); `"lower"` -> `ValueError`; `"upper"` with `Phase(point_group="4mm")` -> `ValueError`; an `ECPMasterPattern` -> `TypeError`; stereographic projection -> `NotImplementedError` (kikuchipy's exception type for that condition, `ebsd_master_pattern.py:356-359`) naming `as_lambert`; a lazy `LazyEBSDMasterPattern` gives identical results.
- Signal method (`test_ebsd_master_pattern.py::TestGetSphericalHarmonics`): `mp.get_spherical_harmonics(bandwidth=32)` returns `MasterPatternHarmonics` with `bandwidth 32`, equal to `MasterPatternHarmonics.from_master_pattern(mp, bandwidth=32)` bit-exact; keywords forwarded (`emsphinx_compatible`, `beam_energy=15` overrides metadata); the docstring example runs (doctest gate).

io plugin (`tests/test_io/test_emsphinx_master_pattern.py`)
- `kp.load(ni_small_20kv_bw384.sht)`: `EBSDMasterPattern`, `data.shape == (2, 769, 769)`, `dtype float64`, `projection == "lambert"`, `hemisphere == "both"`, axes `["hemisphere", "height", "width"]` with offsets `0, -384, -384` and units `"", "px", "px"`, `metadata.General.title == "ni_small_20kv_bw384"`, `metadata.Signal.signal_type == "EBSDMasterPattern"`, `phase.space_group.number == 225`, `phase.point_group.name == "m-3m"`, `phase.name == "Ni"`, `phase.structure.lattice.a == approx(0.35236)`, one atom `element 28`, `original_metadata.header.beam_energy == approx(20.1)`, `.header.primary_angle == 70`, `.harmonics.bandwidth == 384`, `.harmonics.z_rot == 4`, `.crystals.crystal_0.formula == "Ni"`, `.simulations.simulation_0.num_px == 200` (numbered nodes: attribute access works on the `DictionaryTreeBrowser`, which would leave list elements as plain `dict`s), `"packed" not in original_metadata.harmonics`, and `original_metadata.as_dictionary() == MasterPatternHarmonics.from_file(f).original_metadata` (one name, one shape); `kp.load(f, dim=401)` warns and gives `(2, 401, 401)` with height/width offset `-200` (vs `-201` on the EMsoft-read source master, D13); `kp.load(f, hemisphere="upper")` gives `(769, 769)` equal to `data[0]` of the default, `hemisphere="lower"` gives `data[1]` with `hemisphere == "lower"`, `hemisphere="south"` -> `ValueError` listing `upper, lower, both`; `kp.load(f, lazy=True)` returns a `LazyEBSDMasterPattern` whose dask data equals the eager result (the reader itself synthesizes eagerly, documented); `"sht"` is among the extensions of `kikuchipy.io._io.PLUGINS` with `name == "emsphinx_master_pattern"`, `writes is False` and a non-empty `description`; the `file_reader` docstring contains "Not meant to be used directly; use :func:`~kikuchipy.load`."; a synthetic ECP file -> `NotImplementedError` matching "ECP"; two crystals -> `NotImplementedError` matching "numXtal"; `mp.plot()` (Agg) and `mp.get_patterns(...)` (one rotation, `(60, 60)` detector) run; `mp._is_suitable_for_projection()` is `True`.

## Weekly
- `uv run pytest --weekly tests/test_indexing/test_spherical_master_pattern_harmonics.py -k full_master`: `kp.data.ebsd_master_pattern("ni", projection="lambert", hemisphere="both")` (cached; skip if not cached and download not allowed) `.get_spherical_harmonics(bandwidth=384)` (default `emsphinx_compatible=True`) vs `ni_20kv_bw384.sht`: rel-L2 `< 1e-6` all / after `remove_dc` (measured 8.6e-10 / 3.1e-9), `a_00` abs 1e-8, energy weights as recorded, no resolution warning (1001 px carries bw 500), timing recorded.
- The local-gated EMSphInx binary tests are also part of the weekly run on this machine.

## Manual
- Headers: `_master_pattern_harmonics.py`, `_symmetry.py` carry the kikuchipy GPL header + the delimited EMSphInx notice with the D17 line ranges; `_sht_file.py` carries kikuchipy's **BSD-3-Clause** header (`.license_bsd.tmpl`, stamped by the BSD hook after the regex amendment) + the delimited SHTfile BSD-3-Clause notice verbatim (`sht_file.in.hpp:1-33`) with the repo/commit; the io plugin files carry the GPL header (not the BSD template); nothing under `_constants.py`/`ebsdsim_master_pattern/` imports the new code, and `_sht_file.py` imports nothing GPL-derived (also a test).
- Docstrings: numpydoc, types in signatures only, comment lines <= 72 chars, three import blocks, keyword-only arguments, `Raises` sections listing every `ValueError`/`TypeError`/`NotImplementedError` path (incl. `save`'s flag-ambiguous point-group trap and `strict`), the quirk table (`emsphinx_compatible`, with the D7 normalised-correlator statement) in the class `Notes`, `:cite:` keys exist in `doc/user/bibliography.bib`; the `.save` example writes into a temporary directory (the doctest job leaves no file in the CWD).
- `uv run sphinx-build -b html doc doc/_build/html` exits 0 and renders `kikuchipy.indexing.MasterPatternHarmonics` (attributes + methods) and `kikuchipy.io.plugins.emsphinx_master_pattern.file_reader`; `related_projects.rst` sentence present; the three one-sentence CHANGELOG entries present under `Unreleased -> Added` with the fork PR link; the widened `kikuchipy.indexing` package docstring renders.
- Data: the **2** shipped `.sht` files under `src/kikuchipy/data/emsphinx/` (74 828 B each), md5s in `_registry.py`, `Dataset("emsphinx/<file>").fetch_file_path()` works without pooch (`is_in_package`), the generation script documents the exact commands, the `mp2sht.exe`/HDF5-deflate blocker and the one-off `sht2png.exe` acceptance of the 25 generated fixtures whose md5s are recorded below; no `.h5` repack and no synthetic `.sht` is committed; `pyproject.toml` ignores `src/kikuchipy/data/emsphinx/*.py` in the doctest job.
- Coverage of `src/kikuchipy/indexing/_spherical/{_sht_file,_symmetry,_master_pattern_harmonics}.py` and the plugin >= 95 %; `ShtFile.from_dict`, `metadata_dict`, `strict=False`, `hemisphere="lower"` and the raw-bytes writer branch are each hit by a named test (not left to the coverage number).
- Adversarial review findings addressed or explicitly deferred with reason; the bug-injection list of `plan.md` 7.1(c) killed by the suite.
- Known limitations stated in the docs: EBSD-only `.sht`; nm lattice assumption; z-unique monoclinic point groups from orix vs SHTfile's b-unique LUT (guard warns); the four flag-ambiguous point groups need `space_group` for the other setting; `energy_weights` needs the source file or explicit weights; default bandwidth 384 warns on 401 px masters; integer multi-site EMsoft masters are refused (D3); a `.sht` written with the opt-in `emsphinx_compatible=False` carries `a_00 ~ 0` (D7) -- the default `True` writes EMSphInx-identical files; the `.sht` format is not yet listed in the user guide's supported-formats table (`doc/tutorials/load_save_data.ipynb`, Phase 11) although `kp.load` handles it.

## Definition of done
All Phase 2 boxes in `specs/roadmap.md` ticked, default suite green on Windows (this machine) with `-n 4` after a `-n 0` warm-up, the local-gated EMSphInx binary tests run (not skipped) at least once on this machine and green, the weekly full-master parity run once locally and green, doctest gate green, PR opened into fork `develop`; determinations recorded below (parity numbers, `r` values, timings incl. `crc32c`, energy weights, LUT sums, CRCs/md5s of the shipped fixtures, the 25 md5s of the generated fixtures after `sht2png.exe` acceptance). "PR merged" is tracked in the roadmap.

## Recorded results

### 2026-08-16 -- pre-implementation determinations (spec drafting, this machine: Windows 11, uv venv, orix 0.14.2)

Made with scratch NumPy transcriptions on top of the Phase 1 code (`_grid`, `_sht`) and the EMSphInx binaries; they are the evidence behind the frozen tolerances above and must be re-measured and re-recorded by the implementation.

- **`mp2sht.exe`/`sht2png.exe` usage** (no args, exit 1, no files written to the CWD): `usage: mp2sht.exe inputFile outputFile` (`*.h5` -> `*.spx` legacy name), `usage: sht2png.exe inputFile sqLegOut [sterOut]`.
- **`mp2sht.exe` cannot read the in-package Ni master** (gzip): `H5Z_pipeline(): required filter 'deflate' is not registered` (HDF5 1.8.20 built without zlib) -> repacked uncompressed with h5py (1 199 136 B, md5 `b58bece63152a9b5e4c53f5e8899fef7`), then `mp2sht.exe` succeeded in **0.22 s** -> `ni_small_20kv_bw384.sht` (74 828 B, md5 `eef4278b9c48f91f9adbc555f7974d39`, CRC `0xE3100CFF`). On the cached 1001 px/16-energy master `mp2sht.exe` took **0.35 s** -> `ni_20kv_bw384.sht` (74 828 B, md5 `e69da801904a97c812143f0ed78fc769`, CRC `0xEA2875D2`). Shipped `EMSphInx/data/Ni {20kV 75.7deg}.sht`: 74 828 B, CRC `0xF2AF93EF`, `primaryAngle 75.7`, `a_00 = -3.2555`; rel-L2 between the full-master fixture and the shipped file 0.052 (different tilt), between the two fixtures 0.198 (uint8 vs float32 source, 1 vs 16 energies).
- **`sht2png.exe` on `ni_small_20kv_bw384.sht`** (0.22 s): prints `20.1 70`, `file version 1.1`, `written with software version ve49ad6b`, `modality: EBSD`, `beam eng: 20` (int cast of 20.1), `angle 1 : 70`, notes/doi, `master pattern composed from 1 crystals with effective sg# 225`, `rotations are p with pijk = 1`, `simulation data 88 bytes from vendor EMsoft for modality EBSD`, crystal `sg 225 setting 1`, `abc: 0.35236, ...`, `frm: 'Ni'`, `1 atoms: 28: 0 0 0 1 0 0.0035`, `emVers 5_0_0_0`, `sigStart 70`, `sigEnd nan`, `keV 20.1`, `eHistMin 20`, `numSx 201`, `numPx 200`, `latGridType: square lambert`; PNGs `ni_small_leg.png` (118 875 B) and `ni_small_ster.png` (119 750 B) written.
- **CRC-32C**: the literal table equals the table generated by the commented-out loop with polynomial `0x1EDC6F41` (256/256 entries); `LUT[192] = 0x0F6E37A0` (the 25th literal row starts `0x0f6e37a0, 0x05317ad5, ...`, entries 192-199), `LUT[200] = 0x1B5D3F8D`, `sum(LUT) = 68719476608` (~~`= 2**36`~~ corrected 2026-08-16: `2**36` is 68719476736, 128 larger); check values `crc32c(b"") = 0`, `crc32c(b"\x00"*8) = 0xEBE76DE3`, `crc32c(b"123456789") = 0xF28417BE`; all three Ni files verify. Timing on the 74 828 B shipped file (this machine, uv venv): plain-Python `tuple` LUT over `bytes` **3.9 ms**; the same loop with `np.uint32` scalars over `np.frombuffer` **51 ms** (13x slower); identical results.
- **Shipped EMSphInx `data/Ni {20kV 75.7deg}.sht` header** (re-parsed 2026-08-16): `softwareVersion b"ve49ad6b"` (exactly 8 bytes, NUL-terminated source string), `beamEnergy 20.0` (**not** 20.1), `primaryAngle 75.69999694824219`, `keV 20.0`, CRC `0xF2AF93EF` recomputed and matching, `doiLen 46` stored in 48 padded bytes all under the CRC.
- **SHTfile LUTs vs EMSphInx point groups**: `SpaceGroupRot`/`SpaceGroupCmp` reproduced from `SG2PG` + `zRot`/`inversion`/`zMirror`/`mmType` with 0/230 mismatches; vs orix (`get_point_group` operators): 0 mismatches for sg 16-230, and sg 3-15 differ exactly as the b-unique vs z-unique settings predict (`rot` 1 vs 2 for 3-5, 10-15; for 6-15 the LUT sets `0x4`, a mirror plane containing z, where orix's z-unique groups have the mirror perpendicular to z, `0x2`). `sum(rot) = 707`, `sum(cmp) = 948`, `{1: 15, 2: 91, 3: 30, 4: 72, 6: 22}`, 25 distinct pairs with lowest space groups `1, 2, 6, 10, 16, 25, 47, 75, 83, 99, 111, 123, 143, 147, 156, 157, 162, 164, 168, 174, 175, 183, 187, 189, 191`. Point-group names whose space groups span *two* distinct `(zRot, cmpFlg)` pairs (the D11 fallback trap): `3m` (156 `(3, 0x8)` / 157 `(3, 0x4)`), `-3m` (162 `(3, 0x5)` / 164 `(3, 0x9)`), `-42m` (111 `(2, 0x8)` / 115 `(2, 0x4)`), `-6m2` (187 `(3, 0xA)` / 189 `(3, 0x6)`); every other name maps to one pair (`32`: 149 and 150 both `(3, 0x0)`; `mm2`: all `(2, 0x4)`).
- **HyperSpy `DictionaryTreeBrowser` and lists** (hyperspy 2.4.0): `add_dictionary({"crystals": [{"formula": "Ni"}]})` leaves `om.crystals` a `list` of `dict` (`om.crystals[0].formula` -> `AttributeError`); nested dicts (`om.header.beam_energy`, `om.crystals2.crystal_0.formula`) become nodes -> numbered sub-nodes in `metadata_dict()` (D10).
- **EMSphInx correlator default and master DC** (source facts behind D7): `nml.hpp:209` `normed = true`; `idx.hpp:263-284` selects `NormalizedCorrelator` (with `flm2 = SHT(f^2)` and the window `mlm`) when `nml.normed`, `UnNormalizedCorrelator` otherwise; `idx.hpp:181-182` reads each `.sht` and only calls `resize(nml.bw)` -- no `removeDC()`; the only `removeDC` callers are `programs/master_xcorr.cpp:75-76` ("it should already be almost zero from nrm == true in construction").
- **DCT negative control**: constant `3.7` on `13 -> 21` -- `dctn(type=3)` variant `2.8358276643990923 .. 2.8358276643990936` (`== 3.7 * 2*13**2/21**2`), `idctn(type=3)` variant `3.59e-5 .. 6.43e-3` (not constant), max ratio `2.2676e-3 == 1/21**2` (`1/(2*21)**2` would be `5.67e-4`).
- **Parseval oracle for `power_spectrum`** (Phase 1 code, bw 16, `default_rng(0)` uniform coefficients): with `SphericalHarmonicTransform(16, "legendre", dim).synthesize` and `W = quadrature_weights(dim, "legendre")[0][ring_number(dim)]`, `|sum(P) - 4 pi <f^2>_W| / sum(P)` = **2.7e-5** (`dim 19`), **4.7e-8** (`35`), **6.1e-11** (`67`), **6.8e-14** (`131`); with `lambert_solid_angles` (Mazonka) on a Lambert transformer: 5.7e-3 (`dim 33`), 2.9e-3 (65), 1.4e-3 (129), 9.3e-4 (201) -- hence the fine-Legendre-grid oracle at `dim 67`, `rel 1e-8`.
- **Axis offsets**: `nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both").axes_manager["height"].offset == -201.0` (EMsoft reader `-sy // 2`), vs `-(401 // 2) == -200` (ebsdsim reader convention, chosen for D13).
- **EMsoft atom-site sum**: `EMData/EBSDmaster/numset == [1]`, `CrystalData/Natomtypes == 1`, `NMLparameters/EBSDMasterNameList/combinesites == 0` in the in-package Ni file; kikuchipy sums `numset` sites with `data.sum(axis).astype(data.dtype)` (`_emsoft_master_pattern.py:174`), EMSphInx with `std::plus<float>` on `NATIVE_FLOAT` reads (`master.hpp:330-338`).
- **orix names**: `_groups` has the same 38 names on orix 0.12.1 and 0.14.2; `get_point_group(sg)` over 1..230 yields 32 names of which exactly `"2"` and `"m"` are not in `_groups` (`Phase(space_group=3).point_group.name == "2"` is `C2`, a 2-fold about z; `Phase(space_group=6)` gives `"m"` = `Cs`, mirror perpendicular to z). The 40-key table of D9 equals the operator oracle for all 40 names; `Symmetry.get_axis_orders()` counts improper elements (`"2/m"` reports 3 for z) and was rejected as the oracle.
- **mp2sht parity of the scratch pipeline** (`emsphinx_compatible=True`, now the default; bandwidth 384, `dim_leg 387`, `dim_scaled 547`): in-package master rel-L2 **2.2e-9** (all) / **5.1e-9** (after DC removal), `a_00 = -2.984894693337` vs `-2.984894693330`, worst entry `(m, l) = (0, 152)` diff 1.1e-9, structural-zero relative power 6.4e-30, `max |imag| 3.4e-16`; full master rel-L2 **8.6e-10** / **3.1e-9**, `a_00 = -3.30048138900` vs `-3.30048138900`, worst diff 7.6e-10, zero-slot power 4.4e-30. Energy weights of the full master `accum_e (501, 501, 16)` summed over the first two axes: `[0, 0, 1455, 266084, 5365299, 18896238, 27431857, 34257097, 42689288, 53576717, 68256443, 89645683, 124471882, 191778041, 334039769, 161050771]`, total 1151726624 (`EkeVs 5..20`, `Ehistmin 5`, `Ebinsize 1`); the in-package master has `accum_e (201, 201, 1)`, total 16079082, `Ehistmin 20`, `EkeV 20.1`. Timing of the scratch pipeline (excluding HDF5 reads): regrid 0.47 s + SHT construction/analyze 0.22 s (401 px), 0.31 s + 0.14 s (1001 px).
- **`emsphinx_compatible=False`** (opt-in): `a_00 = -6.28e-5` (401 px) / `-5.69e-5` (1001 px); global factor vs the mp2sht coefficients `1.8537` / `2.7405`; rel-L2 after DC removal and rescaling 5.1e-9 / 3.1e-9 (i.e. only scale + DC differ). Parseval on the mp2sht Ni coefficients: `sum_l P_l = 12.5554` vs `4 pi = 12.5664` (rel 8.7e-4), `P_0 = 8.91` (71 % of the power in the DC term under the quirk); dominant degrees after `l = 0`: 44, 40, 50, 36, 60.
- **Lambert reconstruction of the mp2sht coefficients** (direct synthesis, D13): `dim 769/bw 384` r = **0.99633** vs the bilinearly upsampled source, synth 0.10 s; `dim 401/bw 200` r = **0.97043** (north == south exactly), 0.23 s; `dim 201/bw 100` 0.88896; `dim 101/bw 50` 0.62143; Legendre `dim 387` synth min/max -1.699/3.129, north == south. `get_patterns` (three rotations, `(60, 60)` detector, `pc (0.42, 0.22, 0.50)`, tilt 70) NCC vs the source master: **0.99648, 0.99744, 0.99629** at `dim 769`; 0.97787, 0.98196, 0.98116 at `dim 401`.
- **In-package data facts**: `ni_mc_mp_20kv_uint8_gzip_opts9.h5` has `EMData/MCOpenCL/accum_e (201, 201, 1) int32`, `mLPNH (1, 1, 401, 401) uint8`, `EkeV 20.1`, `sig 70`, `numsx 201`, `npx 200`, `totnum_el 2e9`, `multiplier 1`, `SpaceGroupNumber 225`, `SpaceGroupSetting 1`, `LatticeParameters 0.35236 nm`, `AtomData [0, 0, 0, 1, 0.0035]`, `Atomtypes 28`; the loaded phase is `Lattice(a=0.35236, ...)`, `element 28`, `Bisoequiv 0.35`, `Uisoequiv 0.00443`; `original_metadata` keys `BetheList, EBSDMasterNameList, MCCLNameList, CrystalData` (no `EMData`); `tmp_parameters` = folder/filename/extension. `master_patterns.h5` (tests): `mLPNH (1, 11, 13, 13) float32`, `EkeVs 10..20`, **no `accum_e`**, `Natomtypes 2`, partial `MCCLNameList` (no `omega`, `multiplier`) -> no EMsoftED block on save.

(implementation results to be appended below: re-measured parity numbers, `r` values, timings incl. `crc32c`, the 25 md5s of the generated synthetic fixtures after `sht2png.exe` acceptance -- a `dict[int, str]` keyed on space group, copied verbatim into the test -- and the Phase 4 hand-off note on the D7 gate)

### 2026-08-16 -- implementation, step 1: skeleton, failing test suite, shipped fixtures (this machine)

Tests-first pass (`plan.md` tasks 1-6 skeleton + `validation.md` assertions). Only the Phase 1
`_sht.py` lazy-weights amendment is implemented; every other body raises `NotImplementedError`.

- **Shipped fixtures regenerated with the exact commands of `plan.md` 2.3(a),(b)** and they
  reproduce the pre-implementation determinations byte for byte:
  - repack of `ni_mc_mp_20kv_uint8_gzip_opts9.h5` (h5py, no filters): 1 199 136 B, md5
    `b58bece63152a9b5e4c53f5e8899fef7` -- **matches** the recorded value.
  - `mp2sht.exe` on the repack (0.57 s wall incl. process start) ->
    `src/kikuchipy/data/emsphinx/ni_small_20kv_bw384.sht`: 74 828 B, md5
    `eef4278b9c48f91f9adbc555f7974d39`, CRC `0xE3100CFF` -- **both match**.
  - `mp2sht.exe` on the cached `develop/data/ebsd_master_pattern/ni_mc_mp_20kv.h5` (0.32 s) ->
    `src/kikuchipy/data/emsphinx/ni_20kv_bw384.sht`: 74 828 B, md5
    `e69da801904a97c812143f0ed78fc769`, CRC `0xEA2875D2` -- **both match**.
  - Both md5s registered in `src/kikuchipy/data/_registry.py`;
    `Dataset("emsphinx/<file>").fetch_file_path()` resolves in-package without pooch.
- **LUT transcriptions verified against the C++ at transcription time** (parsed out of
  `sht_file.in.hpp` rather than typed): `sum(SpaceGroupRot) = 707`, `sum(SpaceGroupCmp) = 948`,
  `zRot` histogram `{1: 15, 2: 91, 3: 30, 4: 72, 6: 22}`, 25 distinct `(zRot, cmpFlg)` pairs whose
  lowest space groups are `1, 2, 6, 10, 16, 25, 47, 75, 83, 99, 111, 123, 143, 147, 156, 157, 162,
  164, 168, 174, 175, 183, 187, 189, 191` -- all identical to the pre-implementation record. The
  generated CRC table matches the nine pinned literal entries and sums to ~~`2**36`~~ `68719476608` (corrected 2026-08-16). The 40-key
  `Z_ROTATION_ORDER_AND_MIRROR` equals the orix operator oracle for all 40 names (orix 0.14.2).
- **Phase 1 amendment implemented and Phase 1 still green**: `SphericalHarmonicTransform` stores
  `_quadrature_weights = None` and a cached `quadrature_weights` property; `analyze` reads it,
  `synthesize` never does. `tests/test_indexing/{test_spherical_grid,test_spherical_sht,
  test_spherical_fft}.py` -> **348 passed, 507 skipped** (341 before, +7 new
  `TestLazyQuadratureWeights` tests). `SphericalHarmonicTransform(384, "lambert", 769)` now
  constructs and synthesizes (`Y^4_6` to 1e-10 on the rings which carry the order), while
  `analyze` and the property itself raise `ValueError("Insufficient precision ...")`.
- **Failing suite** (`-n 0 -q`, `KIKUCHIPY_EMSPHINX_DIR` unset): 714 collected;
  **627 failed, 79 passed, 8 skipped**, *no* collection or fixture errors. Every failure roots in
  `NotImplementedError`; the only non-`NotImplementedError` assertion messages are the documented
  consequences of a bare stub (`pytest.raises(..., match=...)` on the two codec
  `NotImplementedError` messages, and `pytest.warns` reporting "DID NOT WARN" while handling the
  `NotImplementedError`). The 79 passing tests are exactly the ones asserting the transcribed
  module data (CRC table, the two 230-entry space group tables, the 40-key symmetry table) and the
  licence hygiene of `_sht_file.py`. Under `-n 4`: 627 failed, 428 passed, 516 skipped, 67 s.
- **Gated selections work**: `-k emsphinx_binaries` with `KIKUCHIPY_EMSPHINX_DIR` set selects and
  runs (does not skip) the 6 local-gated tests; `--weekly -k full_master` selects the 2 full-master
  parity tests; without the env var the 6 skip with the reason in the message.
- `pyproject.toml`'s `--ignore-glob=src/kikuchipy/data/emsphinx/*.py` keeps
  `create_emsphinx_sht_fixtures.py` out of `pytest --doctest-modules src/kikuchipy/data`
  (8 doctests collected, none from that file).
- `uv run pre-commit run --files <all 24 new/changed files>`: ruff, ruff-format and both
  `licenseheaders` hooks **pass**; the BSD hook stamps `_sht_file.py` and the GPL hook excludes it.
- **Still to be determined by the implementation**: the 25 md5 sums of the synthetic fixtures
  (`SYNTHETIC_MD5` in `tests/test_indexing/test_spherical_sht_file.py` is an empty dict with a
  loud assertion message until the one-off `sht2png.exe` acceptance of `plan.md` 2.3(c) can run,
  which needs the writer), and every measured tolerance of the pipeline tests.

### 2026-08-16 -- implementation, step 1b: test-quality review fixes (this machine)

Every number below was re-measured here, on top of the Phase 1 code, before the
corresponding test was changed. Three of them contradict sentences written above, which are
struck through in place with the same date.

- **The CRC table sum is `68719476608`, not `2**36`.** `2**36 == 68719476736`, 128 larger.
  Re-generated from `0x1EDC6F41` and compared with the module's `_CRC_TABLE`: 256/256
  entries identical, `sum == 68719476608`. `test_the_table_sum_is_two_to_the_thirty_sixth`
  could never pass and is now `test_the_table_sum_is_the_recorded_value`, which asserts the
  value **and** that it differs from `2**36`. Lines 19, 86 and 125 above corrected.
- **The two 6-fold downgrade cases were swapped.** `_synthetic_alm` is non-zero only at
  `m in {0, 6, 12, 18}`, so filling row `m = 3` leaves the non-zero orders
  `{0, 3, 6, 12, 18}`, whose largest divisor of 6 is **3**, and filling row `m = 2` leaves
  `{0, 2, 6, 12, 18}`, i.e. **2** (measured by enumerating the orders). `plan.md` line 50 and
  `validation.md` line 35 corrected; the other five parametrisations (`4 -> 2`, `4 -> 1`,
  `6 + {2, 3} -> 1`, `3 -> 1`, `2 -> 1`) were right.
- **The per-pixel identity between the two normalisations is false.** With a D7-faithful
  scratch implementation on `default_rng(3)` noise at `dim_legendre 31`:
  `mu_c = 0.4949182367610262`, `mu_p = 0.4949808127642096`, `sigma_c = 0.5753695583161661`,
  `sigma_p = 0.2934128912666275` -- the two settings weight the corners differently
  (`omega/4` against `omega/2`), so the means are not the same number and
  `(compat + mu_c/sigma_c) * sqrt(1 + mu_p**2/sigma_p**2)` differs from `plain` by
  **3.354e-4** absolute, **0.303** relative, against the asserted `rtol = atol = 1e-10`.
  What *is* exact, because both settings are affine in the input:
  `plain == compat * (sigma_c/sigma_p) + (2 mu_c - mu_p)/sigma_p` to **5.6e-16** on both
  hemispheres, which is what the test now asserts (`atol 1e-12`), and it still dies if the
  mean is subtracted once instead of twice under `emsphinx_compatible=True` (the error would
  be `mu_c/sigma_p` ~ 1.7). The recorded 5.1e-9 of line 97 is a *coefficient* level number
  and is now asserted as such by
  `TestMp2shtParity::test_the_two_settings_agree_after_dc_removal_and_rescaling`: a constant
  shift of a function on the sphere lives entirely in `a_00`, so after `remove_dc` the two
  settings differ by the single scalar `sigma_c/sigma_p`; one scalar is fitted from the norm
  ratio (asserted `== approx(1.854, rel=1e-3)`) and the remaining 9311 coefficients carry the
  `rel-L2 < 1e-8`. The same run reproduces the two normalisation assertions the suite already
  made: compatible weighted mean `-0.8601745254118366 == -mu_c/sigma_c` to the last bit,
  corrected weighted mean `1.3e-16` and variance `1.0000000000000000`.
- **The row/column lock was blind to the transposition it names.** On the Legendre grid of
  `dim 31`, `max |X - Y.T| = 1.11e-16` while `max |X - Y| = 1.0`, so
  `allclose(out_x, out_y.T)` holds for a transposed `(X, Y)` in the bilinear step as well.
  The test now asserts the **absolute** mapping, `out_x == sphere_to_square(...)[..., 0]` and
  `out_y == ...[..., 1]` at `atol 1e-12`.
- **Non-vacuous round trip.** `num_harmonics(384, 4, 0x7) == 9312`, asserted as the non-zero
  count of the re-read coefficients before the bit-exact comparison, so a writer which stored
  zeros no longer compares two empty selections.
- **Hemisphere order.** `nickel_ebsd_master_pattern_small(...).data[0]` is bit-identical to
  `data[1]`, so every hemisphere assertion against `data[0]`/`data[1]` of the same Ni call is
  blind to a swap. Both the class and the io plugin now also run on an antisymmetric
  instance/file (a single `alm[0, 1] = 1`, i.e. `Y_1^0 ~ cos(theta)`): `south == -north` and
  the centre pixel of the square Lambert grid, which is the pole (`normals(9, "lambert")[4, 4]
  == [0, 0, 1]`, measured), is positive on the upper and negative on the lower hemisphere.
- **Coverage of the orix cross-check** widened from `range(16, 231, 7)` (31 of 215 space
  groups) to all 215, as line 20 asks; the stride was a pure coverage loss, the full range
  having been run with 0 mismatches.
- **Failing suite after the fixes** (`-n 0 -q`, `KIKUCHIPY_EMSPHINX_DIR` unset): 899
  collected in the four new files plus `TestGetSphericalHarmonics`, **815 failed, 81 passed,
  11 skipped**, no collection or fixture errors. 804 failures are bare `NotImplementedError`;
  the other 11 are the documented consequences of a bare stub (8 "DID NOT WARN", the two
  `match=` misses on `read_sht`'s `NotImplementedError`, and one `in ""` on an exception
  message). The 81 passing are the transcribed-data locks, the licence hygiene, the plugin
  registration, the signature defaults and the Phase 1 amendment. `-k emsphinx_binaries` with
  the env var set selects and runs 8 gated tests (6 before, the two new shipped-file ones
  added). Phase 1 after the change: `test_spherical_{grid,sht,fft}.py` -> **348 passed, 507
  skipped**, unchanged.
- **Still to be determined by the implementation**: unchanged from step 1, plus the two
  numbers this step's new tests will record, `opt_in_rel_l2_after_dc_and_rescale` and
  `opt_in_norm_ratio`.

### 2026-08-16 -- implementation, step 2: symmetry flags (`_symmetry.py`, `plan.md` task 3, this machine)

- **The 40-key table is unchanged and still equals the operator oracle** (orix 0.14.2, all 40
  names; `n_fold == 1 + #{proper g: |axis_z| == 1, angle > 0}`,
  `has_equatorial_mirror == any(improper g: angle == pi, |axis_z| == 1)`), re-measured before
  the functions were written: **0 mismatches**.
- **Eight of the 40 names are never returned by `get_point_group`** over the 230 space groups
  (measured): `112`, `121`, `211`, `m11`, `1m1`, `11m`, `312`, `321`. The first six are the
  axis-specific aliases of the z-unique `"2"`/`"m"` orix returns for space groups 3-5 and 6-9
  (-> 3 and 6, D11); the last two are the trigonal aliases of `"32"`, whose standard settings
  are 149 (P312) and 150 (P321). All eight are tabulated in `AXIS_ALIAS_SPACE_GROUPS`; the
  other 32 names are looked up in orix at first use (`_space_groups_by_name`, `functools.cache`;
  230 `get_point_group` calls measured at **0.44 ms** in total, so no import-time cost is
  taken and orix stays out of the module's import set).
- **`candidate_space_groups` reproduces the D11 trap from the `_sht_file` LUTs**, not from the
  point group table: `3m -> (156, 157)`, `-3m -> (162, 164)`, `-42m -> (111, 115)`,
  `-6m2 -> (187, 189)`, every other name one candidate (`m-3m -> (221,)`, `32 -> (149,)`,
  `mm2 -> (25,)`). Enumerating all 32 returned names against
  `(space_group_z_rotation, space_group_compression_flags)` gives **exactly those four**
  ambiguous names, and the first candidate equals `space_group_for_point_group` in every case.
- **The Ni mp2sht coefficients are clean** (`ni_small_20kv_bw384.sht`, unpacked with a scratch
  parser because the codec's `read_sht` is still a stub: `bw 384`, `zRot 4`, `cmpFlg 0x7`,
  `doubCnt 9312`): `systematic_zero_power(alm, 4, True) == (0.0, 0.0)` **exactly** (not merely
  `<= 1e-20`) and `validate_flags(alm, 4, True) == (4, True, [])`. The same parse gives
  `a_00 = -2.9848946933297453`, i.e. the recorded -2.985 of line 96.
- **The largest-satisfied-divisor downgrade is measured, not asserted from the ladder**: on
  the synthetic 6-fold `alm` of the test (orders `{0, 6, 12, 18}`), filling row `m = 3` gives
  `n_fold 3` and row `m = 2` gives `n_fold 2` (the correction of 2026-08-16 above holds),
  filling both gives 1; `4 + row 2 -> 2`, `4 + row 1 -> 1`, `3 + row 1 -> 1`, `2 + row 1 -> 1`.
  The tolerance boundary behaves as `<=`: a single `m = 2` entry of relative power `0.9e-8`
  keeps `n_fold 4` with no warning, `1.1e-8` downgrades to 2 with one.
- **Edge cases measured**: an all-zero `alm` returns `(0.0, 0.0)` and no downgrade (no
  division by zero); the input array is never modified (`np.abs(alm) ** 2` is a fresh array),
  a read-only and a real-valued `alm` both work; `bandwidth 1` returns `(0.0, 0.0)`.
- **Test status**: `tests/test_indexing/test_spherical_symmetry.py -n 0 -q` -> **182 passed, 2
  failed**, both failures being `read_sht`'s bare `NotImplementedError` in the two Ni
  coefficient tests (`plan.md` task 2, the codec, is not implemented yet); both pass on the
  same coefficients when they are unpacked by the scratch parser, as recorded above.
  `tests/test_indexing/` is otherwise unchanged (Phase 1: 348 passed, 507 skipped).
  `uv run pre-commit run --files src/kikuchipy/indexing/_spherical/{_symmetry,_sht_file}.py`
  passes.
- **Cross-task edit, recorded**: `space_group_z_rotation` and `space_group_compression_flags`
  of `_sht_file.py` (one LUT index each, plus a shared `_check_space_group` for the
  `[1, 230]` `ValueError`) were implemented here, because `candidate_space_groups` is defined
  in terms of them and the two 230-entry tables were already transcribed and verified. The
  rest of the codec is untouched.

### 2026-08-16 -- implementation, step 3: the `.sht` codec (`_sht_file.py`, `plan.md` task 2, this machine)

- **Suite**: `uv run pytest tests/test_indexing/test_spherical_sht_file.py -n 0 -q` -> **530 passed,
  6 skipped** (the skips are the `KIKUCHIPY_EMSPHINX_DIR` gated ones). With
  `KIKUCHIPY_EMSPHINX_DIR=c:/Users/westraadt.1/Repos/EMSphInx`: **536 passed, 0 skipped, 0 failed**,
  i.e. the six local-gated tests (the shipped EMSphInx `data/Ni {20kV 75.7deg}.sht` parse, its DC
  term, its byte-identical rewrite, its count/payload agreement, its bitwise repack, and the
  `sht2png.exe` acceptance of the 25 generated fixtures) all pass.
  `uv run pytest --doctest-modules src/kikuchipy/indexing/_spherical/_sht_file.py` -> 2 passed.
  `uv run pre-commit run --files src/kikuchipy/indexing/_spherical/_sht_file.py
  src/kikuchipy/data/_dummy_files/emsphinx_sht.py
  src/kikuchipy/data/emsphinx/create_emsphinx_sht_fixtures.py
  tests/test_indexing/test_spherical_sht_file.py` passes.

- **`sht2png.exe` acceptance of the 25 synthetic fixtures (the one-off determination of D16 /
  `plan.md` task 2.3(c))**, run 2026-08-16 in a scratch directory with
  `EMSphInx/build/Release/sht2png.exe` (EMSphInx @ 60f3517): **all 25 exit 0**, each stdout ends in
  `master pattern composed from 1 crystals with effective sg# N` with `N` the file's own space
  group, and each wrote a non-degenerate PNG (280-755 B, size varying with the coefficients, so
  the program really unpacked and synthesized rather than emitting a blank image). This is an
  external check of `num_harmonics`/`pack_harmonics` on all 25 distinct `(zRot, cmpFlg)` pairs and
  of the whole byte layout, since EMSphInx re-reads the header, the crystal record and the
  harmonics block and verifies the CRC-32C itself. The 25 md5 sums below are now pinned in
  `SYNTHETIC_MD5` of `tests/test_indexing/test_spherical_sht_file.py`; if the writer legitimately
  changes, re-run the acceptance and re-pin here and there together.

  | sg | md5 | sg | md5 |
  |---|---|---|---|
  | 1 | `25865cf3df7e49438647a6c73e50b8ab` | 157 | `21649433847c4aa1040f860cc2775c6c` |
  | 2 | `1ac5c1930b4e80e41c872276031836d2` | 162 | `79ef7034b716af2c074530343e2185ec` |
  | 6 | `06fc126c96b4f894ec35f2efab1a37e2` | 164 | `17ad8bb7c7c7dd078f8e971b5f1e9fd1` |
  | 10 | `626e2e9b6345688959c68cc97877de20` | 168 | `9a4f8a488fef1cfe3caf937991ae9aea` |
  | 16 | `ae8b1beed0cd4e93d6c7a05f695b5bc5` | 174 | `f36136ea8bb623707de545e06873b378` |
  | 25 | `a0e289fb3e094c69b60047a730a5aba1` | 175 | `06e85c1d584924b2053616b779d9195e` |
  | 47 | `11953424f1178e7e11118c8ca1a7b0e7` | 183 | `3477b79bba9ed985d79d8a3f865e7719` |
  | 75 | `8f133ea4ca405143079634f8b13dc345` | 187 | `00738a7ac24c8cdc819db19009e28a8b` |
  | 83 | `2fbc17b2d35c008e36ec1b00ee5b7cbd` | 189 | `a813922d6a295968ed31ce0749fe73b7` |
  | 99 | `8d910eae70622d11b9767b69b3c91238` | 191 | `2fc91c0642f89084978aae8572824e49` |
  | 111 | `64be105bb1fca491c980bc73e4706a86` | 143 | `efe2e034f792f8149619e4a20560e898` |
  | 123 | `9f2707146f926ab87cc30ccc9d93c4b4` | 147 | `399c71928363f274709ab0188daff6f2` |
  | 156 | `30a1a12e7cbfa01f21452ee31b69d0b4` | | |

  File sizes 284 B (sg 191) to 2 340 B (sg 1), all at bandwidth 16 with the RNG-free fill of D16.

- **`crc32c` timing measured**: 5.1 ms for the 74 828 B `ni_small_20kv_bw384.sht` on this machine
  (the plain-Python `tuple` lookup table over a `bytes` object), against the 3.9 ms recorded during
  spec drafting and the test's 50 ms bound. The bound still separates this implementation from the
  documented-wrong NumPy scalar variant (51 ms).

- **Block offsets confirmed on the shipped files**: `block_offsets` of `ni_small_20kv_bw384.sht`
  gives exactly `{header: 0, master_pattern: 112, crystal_0: 120, simulation_0: 232,
  harmonics: 320, payload: 328, crc: 74824}`, i.e. the measured layout of D10, and
  `sht_file_to_bytes(read_sht(f)) == f.read_bytes()` for all three mp2sht files (the two
  in-package ones and the shipped EMSphInx one, the last behind the env var).

- **Determined during implementation, `metadata_dict()` shape** (D10 left the details open):
  the header node drops the raw padded string bytes and the unpadded string lengths (file
  plumbing) and renames the two reserved byte fields to `reserved` and `reserved2`, because
  `res_bytes`/`res_bytes2` would have tripped the "no key ends in `_bytes`" assertion; the atoms
  of a crystal are **numbered sub-nodes** `{"atom_0": {...}}` for the same reason crystals and
  simulations are, so `original_metadata.crystals.crystal_0.atoms.atom_0.atomic_number` works.
  Verified against hyperspy 2.4.0: `DictionaryTreeBrowser(md).as_dictionary() == md` exactly
  (tuples survive as tuples) and every documented attribute access resolves.

- **Determined, opaque simulation records**: `_record_for_modality`/`_record_vendor` return
  "accepted"/`None` for a record kept as raw `bytes`, so only its length is checked -- the type
  and therefore the modality/vendor support of such a record is unknown to us (the reader
  permissiveness (a) of D10). An `EMsoftED` record keeps EMSphInx's own
  `forModality` (EBSD/ECP/TKD) and vendor EMsoft.

- **Determined, check order in `_sanity_check`**: our `1 <= bandwidth <= 32767` check runs
  *before* `doubCnt != NumHarm(...)`, because `num_harmonics` allocates a `bandwidth**2` mask and
  a bandwidth the int16 field cannot hold would otherwise allocate a 1e9 entry array before
  failing; and our `num_xtal >= 1` check runs where `File::sanityCheck` dereferences
  `mpData.simul.front()`. Every other check keeps EMSphInx's order and wording verbatim,
  including the typo "noites string doesn't match length".

- **Cross-task note**: `space_group_z_rotation`, `space_group_compression_flags` and
  `_check_space_group` were already implemented by the `_symmetry.py` task (recorded in step 2
  above) and are kept unchanged here; the rest of `_sht_file.py` is this step's work.

### 2026-08-16 -- implementation, step 4: `MasterPatternHarmonics`, the signal method and the io plugin (`plan.md` task 4-6, this machine)

Suites (`uv run pytest ... -n 0 -q --tb=short -p no:cacheprovider`, then `-n 4`):
`test_spherical_master_pattern_harmonics.py` **155 passed, 4 skipped** (2 weekly, 2 EMSphInx
binaries); `test_emsphinx_master_pattern.py` **20 passed**; `test_ebsd_master_pattern.py`
**50 passed, 1 skipped**; whole Phase 1+2 set `-n 0` **1287 passed, 518 skipped** and `-n 4`
**1287 passed, 518 skipped in 69 s**; `tests/test_indexing` **1315 passed**; `tests/test_io
tests/test_data` **195 passed** (no regression). `pytest --doctest-modules
src/kikuchipy/indexing/_spherical src/kikuchipy/io/plugins/emsphinx_master_pattern
src/kikuchipy/signals/ebsd_master_pattern.py` **11 passed**. `--weekly -k full_master`
**2 passed** (the full Ni master is cached). `KIKUCHIPY_EMSPHINX_DIR=...` gated set
(`-k emsphinx_binaries`) **8 passed**, i.e. `sht2png.exe` accepts a `.sht` written by `save()`
from the in-package Ni master (`effective sg# 225`, exit 0) and its Legendre PNG matches our own
`SphericalHarmonicTransform(384, "legendre", 387).synthesize(alm)` to `max |diff| <= 1` grey
level. `uv run sphinx-build -b html -D nbsphinx_execute=never doc doc/_build/html_check` exits 0
with **1 warning** (the pre-existing unreachable `pyxem` intersphinx inventory) and renders
`kikuchipy.indexing.MasterPatternHarmonics` (all 11 attributes/methods) and
`kikuchipy.io.plugins.emsphinx_master_pattern.file_reader`. `uv run pre-commit run --files ...`
passes on every touched file.

- **mp2sht parity, measured now** (`record_property`, in-package 401 px uint8 Ni master, default
  `emsphinx_compatible=True`, `bandwidth=384`): rel-L2 over all coefficients **2.208e-09**,
  after `remove_dc()` on both **5.141e-09**, `a_00` **-2.984894693337** (file: -2.9848946933297453,
  i.e. 7e-12 absolute), relative power in the slots the packer discards **1.112e-29**,
  `max |imag| 3.4e-16`. Weekly full master (1001 px, 16 energies): rel-L2 **8.555e-10**, after
  `remove_dc` **3.099e-09**, energy bin total **1151726624**. Every number reproduces the
  pre-implementation determinations of `## Recorded results` above, so the < 1e-6 gate has > 400x
  margin.
- **Opt-in `emsphinx_compatible=False`**: `a_00` **-6.281e-05** (401 px), global amplitude factor
  vs the file **1.8537** (spec: 1.854), rel-L2 after DC removal and one global rescaling
  **5.140e-09**, `power_spectrum().sum()` **12.5277** against `4 pi = 12.5664` (0.31 %).
- **DC ratios** (`power_spectrum`): default `True` on the 401 px master `P_0 / sum(P)` =
  **0.7096** (`sum(P) = 12.5554`), on the cached 1001 px master **0.8680** (`a_00 = -3.300481`);
  opt-in `False` **3.149e-10**, which is the `"DC power fraction 0.00"` of `describe()`.
- **`to_master_pattern`**: Pearson `r` vs the bilinearly upsampled source north hemisphere
  **0.99633** at `dim = 769` and **0.97043** at `dim = 401`; `get_patterns` NCC against the same
  call on the source master **0.99648 / 0.99744 / 0.99629** for the three test rotations.
- **`power_spectrum` Parseval** against the Gauss-Legendre quadrature at `dim 67`: relative error
  **6.065e-11**; on the mp2sht Ni file `sum(P) = 12.5554`.
- **Timings on this machine** (uv venv, Windows 11, warm caches): `from_master_pattern` at
  `bandwidth=384` **247 ms** on the 401 px master and **312 ms** on the cached 1001 px 16 energy
  master; `to_master_pattern()` at `dim 769` **127 ms**; `save()` **12 ms**; `from_file()`
  **11 ms**. The whole Phase 1+2 suite runs in 69 s at `-n 4`.
- **Coverage** (the four suites of `## Automated`): `_master_pattern_harmonics.py` **96.15 %**,
  `_sht_file.py` **97.42 %**, `_symmetry.py` **98.72 %**, `emsphinx_master_pattern/_api.py`
  **100 %** -- all above the 95 % bar; the uncovered lines are defensive `raise`/early-return
  branches.
- **Determined, a constant master under the default normalisation**: EMSphInx subtracts *twice*
  the weighted mean (`master.hpp` lines 572-573), so a constant master normalizes to -1
  everywhere and `a_00` to exactly `-sqrt(4 pi) = -3.5449077`, *not* to 0 -- independent of the
  source amplitude and of D5's DCT factor, which is what the test asserts (the amplitude factor
  is observable only through `normalize=False`). `validation.md` line 43 and the test previously
  claimed `|a_00| < 1e-10` for `normalize=True`; that is impossible for the ported quirk, and
  `emsphinx_compatible=False` cannot give it either, since a constant has zero weighted variance
  and the normalization divides by ~1e-16.
- **Determined, `save()` -> `from_file()` bit-exactness**: the round trip is bit-exact in the
  **real part** of every kept coefficient, and drops the 3.4e-16 imaginary residue of the forward
  transform, because the mirror-y compression of space group 225 stores `alm[i].real()` alone
  (`PackHarm`, `sht_file.in.hpp` line 1727). The relative power dropped is 1.1e-29, far inside
  `SYMMETRY_POWER_TOLERANCE`.
- **Determined, NaN in `metadata_dict()`** (needed by the "one name, one shape" assertion
  `original_metadata.as_dictionary() == MasterPatternHarmonics.from_file(f).original_metadata`):
  the `EMsoftED` record carries `sigEnd`/`sigStep` NaN, and `float("nan") != float("nan")`, so
  two independent parses of one file could never compare equal. `ShtFile.metadata_dict()` now
  maps every NaN to the `math.nan` **singleton** (new private `_singleton_nan`), which makes the
  containers equal through CPython's identity short circuit in `PyObject_RichCompareBool`. The
  byte level reader, writer and `to_dict`/`from_dict` are untouched, so byte identity is
  unaffected (re-verified: the two Ni files still round trip byte-identically).
- **Determined, file size of a kikuchipy-written `.sht`**: 74 836 B against `mp2sht`'s 74 828 B
  for the same master, because the provenance note (65 B -> 72 padded) replaces mp2sht's doi +
  note (46 + 19 -> 48 + 24) and the crystal `name` is written (`"ni"`, 0 -> 8 padded).
- **Determined, axis offsets**: `to_master_pattern(dim=401)` gives height/width offset **-200**
  where `nickel_ebsd_master_pattern_small(projection="lambert")` gives **-201** (kikuchipy's
  EMsoft reader, `-sy // 2`); asserted in both the harmonics and the io test, as D13 requires.
- **Tests touched** (three assertions, each proven impossible against the C++ or against
  floating point; no test logic, no new test): (1)
  `TestContainer::test_the_coefficients_are_copied_and_contiguous` compared a value stored in a
  **64-bit complex** source array against the 128-bit `sqrt(4 pi)` -> compared against
  `np.complex64(SQRT_FOUR_PI)`; (2)
  `TestNormalizeFalse::test_normalize_true_kills_the_constant` -> asserts
  `a_00 == approx(-sqrt(4 pi))`, see the determination above; (3)
  `TestSaveAndFromFile::test_a_round_trip_keeps_the_coefficients_bit_exactly` -> compares the
  `.real` parts, see the determination above.

### 2026-08-16 -- adversarial review pass (fidelity, conventions, bug injection)

Every blocker/major and the cheap clearly-right minors of the three reviews applied to the Phase 2
sources; the two surviving-mutation findings closed with new tests. Measured on this machine.

- **Fidelity review found no defect in any ported numeric or in the `.sht` byte layout.** Its two
  findings are Python-API robustness, not infidelity: `save()` without `preserve_header=True`
  dropped 13 fields the reader had parsed, and `read_sht` let a corrupt `z_rot` reach the
  `order % (z_rot * 2)` of `_row_kind`.
- **Determined, the fields an orix `Phase` cannot hold** (measured by injecting non-default values
  into `ni_small_20kv_bw384.sht`, reading and re-saving without `preserve_header`): before the fix
  `header.secondary_angle` 12.5 -> 0.0, `header.reserved_param` -0.75 -> 0.0, crystal `sg_axis`
  4 -> 1, `sg_cell` 3 -> 1, `origin` (1.5, -2.5, 3.5) -> (0, 0, 0), `rot` (.5, .5, .5, .5) ->
  (1, 0, 0, 0), `weight` 0.25 -> 1.0, `structure_symbol`/`references`/`note` -> `""`, atom
  `charge` -1.5 -> 0.0 and `res_fp` 3.25 -> 0.0. `_crystal_from_phase` now takes the
  `crystals.crystal_0` node of `original_metadata` (and its `atoms.atom_N` sub-nodes) as the
  fallback for exactly those, plus the atom `res` bytes, and `_sht_from_harmonics` seeds
  `secondary_angle`/`reserved_param` from `header`. All 13 now survive the round trip; the raw
  padded string *bytes* still do not, which is what `preserve_header=True` is for, and `save`'s
  `Notes` says so. No numerical impact: `MasterSpectra::read` reads only `sgEff`, `lat`,
  `beamEnergy`, `primaryAngle`, `bw` and the payload. Byte identity of the two shipped files, the
  25 fixtures and the `preserve_header` path is unchanged (re-verified).
- **Determined, a corrupt `z_rot`**: a file claiming `z_rot == 0` with `FLAG_MIRROR_X` raised
  `ZeroDivisionError` from `_row_kind` (the compiled EMSphInx `PackHarm` exits `0xC0000094`,
  integer divide by zero -- not an infidelity, only an unhelpful exception type), and a *negative*
  `z_rot` diverged silently, Python's floored `%` typing rows real where C++' `size_t` modulo
  types them imaginary (e.g. `order=2, z_rot=-1`). `read_sht` now rejects `z_rot < 1` with a
  `ValueError` next to the existing `_flags_to_bools(flags)`; the LUT only ever emits 1-6.
- **Determined, the quirks table did not render**: the committed stub's grid table had a 23 char
  first column and a 25 char first data cell, and split ``SYMMETRY_POWER_TOLERANCE`` from its
  trailing `.` across a row boundary, so docutils rejected the whole table
  ("Malformed table. Right border not aligned or missing") and the built page showed *nothing*
  between the two paragraphs. Re-laid out at 25/10/30 columns; the built HTML now has the table
  with 30 `<td>` cells, and the doc build reports **0** "Malformed table" and **0** "Inline
  emphasis start-string without end-string" warnings.
- **Doc build, honestly**: `uv run sphinx-build -b html -D nbsphinx_execute=never doc
  doc/_build/html_check` exits **0** with **151 warnings** (was 157 before this pass: the
  malformed table plus the five unescaped `*.sht` in the io plugin). Every remaining warning is
  pre-existing and unrelated; the only two naming a Phase 2 page are
  `MasterPatternHarmonics.save.rst: Could not match transformation of 'tempfile'/'pathlib'`, one
  of a class with 110+ siblings elsewhere in the docs.
- **Determined, the two surviving-mutation holes** (60 mutants, 52 killed, 8 survivors, all in
  `_master_pattern_harmonics.py`): (a) every default-suite test that drove a multi-energy master
  through `from_master_pattern` used `energy_weights=np.ones(11)`, which is invariant under a
  reversal *and* equal to a plain mean, so reversing the weights or dropping them entirely was
  invisible; (b) every `save()` used the single-site cubic Ni phase, whose formula sort, `set`,
  unit occupancy, 90/90/90 cell and integer `element` are all degenerate. Two new tests,
  `TestEnergyWeights::test_the_weights_are_applied_by_the_pipeline` (non-uniform
  `np.arange(1., 12.)` against an in-test pipeline) and
  `TestSaveAndFromFile::test_the_crystal_block_of_a_multi_element_hexagonal_phase`
  (Si/O/Si, partial occupancy, 0.49/0.49/0.54/90/90/120). Re-measured: **15/15** targeted
  mutations killed, including all 8 previous survivors.
- **Coverage after the pass** (the four suites of `## Automated`):
  `_master_pattern_harmonics.py` **98.30 %** (was 96.15/96.17 %), `_sht_file.py` **97.42 %**,
  `_symmetry.py` **100.00 %** (was 98.72 %), `emsphinx_master_pattern/_api.py` and
  `__init__.py` **100 %**. Every remaining uncovered line is a defensive `raise`/early return for
  a corrupt or incomplete input.
- **Suites after the pass**: the four Phase 2 files **923 passed, 10 skipped** (was 889/10; +34
  tests); with `tests/test_signals/test_ebsd_master_pattern.py` and the Phase 1 trio
  **1321 passed, 518 skipped** at `-n 0` (46 s) and identically at `-n 4` (57 s);
  `tests/test_indexing tests/test_io tests/test_signals tests/test_data -n 4`
  **1819 passed, 560 skipped** (was 1785/560); doctests **11 passed**; the eight local-gated
  `KIKUCHIPY_EMSPHINX_DIR` binary tests **8 passed** (the writer still produces bytes
  `sht2png.exe` accepts and the 25 pinned md5s are unchanged); `pre-commit` green on all 15
  touched files.
- **Minors fixed**: `describe()` no longer calls `chr()` on the raw signed `rot_sense` byte (shows
  `?`); `save`'s `overwrite` now uses the exact branch shape of `kikuchipy.io._io._save`, so
  `overwrite="yes"` raises whether or not the file exists (it silently wrote before when it did
  not); `from_file`/`save` are annotated `str | Path`, which is what the io plugin passes;
  `_atomic_number(True)` raises instead of returning hydrogen (`bool` is an `int` subclass);
  `_master_pattern_dict` lowercases `hemisphere`, as the sibling `ebsdsim_master_pattern` plugin
  does; `MasterPatternHarmonics.__init__` stores `phase.deepcopy()`, so a master pattern and the
  coefficients made from it can no longer mutate each other's phase.
- **Explicitly not changed**: `to_master_pattern` still hands the signal `self.phase` itself
  (`phase is h.phase` is a named assertion of `## Automated` line 47), so the copy is made at
  construction instead; `SYNTHETIC_MD5` and the three earlier test-assertion corrections stand
  as recorded; `doc/tutorials/load_save_data.ipynb` still does not mention `.sht` (Phase 11, and
  the notebooks are out of scope here); the two >72 char comment lines of
  `create_emsphinx_sht_fixtures.py` are unwrappable shell commands and stay.
- **Not reproduced**: the conventions review saw sporadic, wide-spectrum failures whose values are
  impossible for correct code (axis offset -385 where `-(769 // 2)` must give -384, a literal
  `charge=0.0` reading back as 1.0), always with a second python/`uv` process on the same
  checkout, and never on an idle machine over ~40 repeat runs. Treated as an environmental hazard
  of the shared numba `cache=True` on-disk cache, not a code defect: never run two processes
  against this checkout, and give parallel CI jobs a per-process `NUMBA_CACHE_DIR`.

### 2026-08-17 -- CI portability fix (after PR #5 CI)

- The `ubuntu-latest-py3.10-oldest` job (orix 0.12.1) failed `TestSpaceGroupTables::test_the_tables_agree_with_orix_above_space_group_15[25..46]`: that orix orients the two-fold of the orthorhombic `mm2` groups about x, so its *operators* disagree with the SHT file tables for space groups 25-46 (the tables are right: Pmm2's two-fold is along c, and they are pinned independently by the transcription tests). The two orix-operator oracle tests now skip, self-detected on space group 25, when orix does not put the `mm2` two-fold along z. The name-keyed `_symmetry.py` LUT and its tests are unaffected (they passed on 0.12.1).
- Pre-existing, not ours: the macOS "GPU tests" step fails on upstream develop too (ebsdsim branch missing); the docstring step runs with `continue-on-error`.

