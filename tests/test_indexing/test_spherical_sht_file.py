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

"""Tests of ``kikuchipy.indexing._spherical._sht_file``.

Covers the ".sht codec" assertions of
``specs/2026-08-16-sht-master-spectra-and-file/validation.md``:

- CRC-32C check values, table spot values by true index, the table
  sum and a recorded timing on the 74 828 B shipped file.
- The two 230-entry space group tables: length, value ranges,
  checksums, histogram, spot values, the 25 distinct
  ``(z_rot, flags)`` pairs, and the orix cross check which agrees for
  space groups 16-230 and disagrees for 3-15 (SHT file b-unique vs
  orix z-unique).
- ``num_harmonics`` closed forms and, against the only external
  oracle for the count, ``doub_cnt == num_harmonics(...)`` and the
  payload length for every ``.sht`` file in the suite.
- Pack and unpack round trips for all 25 flag pairs, with each of the
  four storage branches asserted entry by entry.
- Header parse of the two in-package Ni files and, behind
  ``KIKUCHIPY_EMSPHINX_DIR``, of the shipped EMSphInx Ni file.
- Byte identity of ``read -> sht_file_to_bytes`` on the external
  oracles, ``write -> read -> write`` idempotence plus pinned md5
  sums on the 25 generated fixtures, and the lossless ``to_dict``.
- Robustness: a corrupt checksum, big-endian magic, version 1.0, a
  wrong magic, illegal flags, opaque simulation records, two
  crystals, string padding, non-UTF-8 raw bytes and every writer
  sanity check message.
- Licence hygiene: the module imports nothing GPL derived and the
  pre-commit hook regexes cover it.
"""

import ast
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import struct
import time

import numpy as np
import pytest

from kikuchipy.data._data import Dataset
from kikuchipy.indexing._spherical import _sht_file

# Space groups of the 25 synthetic fixtures, one per distinct
# (z_rot, flags) pair (D16)
SYNTHETIC_SPACE_GROUPS = (
    1, 2, 6, 10, 16, 25, 47, 75, 83, 99, 111, 123, 143,
    147, 156, 157, 162, 164, 168, 174, 175, 183, 187, 189, 191,
)  # fmt: skip

# The (z_rot, flags) pair of every one of them
SYNTHETIC_FLAG_PAIRS = {
    1: (1, 0x0),
    2: (1, 0x1),
    6: (1, 0x4),
    10: (1, 0x5),
    16: (2, 0x0),
    25: (2, 0x4),
    47: (2, 0x7),
    75: (4, 0x0),
    83: (4, 0x3),
    99: (4, 0x4),
    111: (2, 0x8),
    123: (4, 0x7),
    143: (3, 0x0),
    147: (3, 0x1),
    156: (3, 0x8),
    157: (3, 0x4),
    162: (3, 0x5),
    164: (3, 0x9),
    168: (6, 0x0),
    174: (3, 0x2),
    175: (6, 0x3),
    183: (6, 0x4),
    187: (3, 0xA),
    189: (3, 0x6),
    191: (6, 0x7),
}

# md5 sums of the 25 synthetic fixtures, pinned after the one-off
# ``sht2png.exe`` acceptance of ``plan.md`` task 2.3(c). They are the
# drift guard the shipped copies would have provided, without the
# bytes: a writer which produces the accepted bytes cannot change
# unnoticed, and a legitimate byte change re-runs ``sht2png.exe`` and
# re-pins these.
SYNTHETIC_MD5 = {}

# The two in-package files, both 74 828 B
NI_SMALL = "emsphinx/ni_small_20kv_bw384.sht"
NI_FULL = "emsphinx/ni_20kv_bw384.sht"

SHT_FILE_SIZE = 74828

REPO_ROOT = Path(__file__).resolve().parents[2]


def _data_path(name: str) -> Path:
    """Return the path of an in-package ``.sht`` file."""
    return Path(Dataset(name).fetch_file_path())


@lru_cache(maxsize=None)
def _read(name: str) -> _sht_file.ShtFile:
    """Return a parsed in-package ``.sht`` file.

    Memoized and called inside the test bodies rather than made a
    fixture, so that an unimplemented stub fails the test instead of
    erroring its setup.
    """
    return _sht_file.read_sht(_data_path(name))


def _emsphinx_dir() -> Path:
    """Return the EMSphInx checkout, skipping if it is not set up.

    The local gated tests need ``KIKUCHIPY_EMSPHINX_DIR`` to point at
    a checkout with the built programs and the shipped Ni file.
    """
    value = os.environ.get("KIKUCHIPY_EMSPHINX_DIR")
    if not value:
        pytest.skip(
            "KIKUCHIPY_EMSPHINX_DIR is not set; set it to an EMSphInx "
            "checkout with build/Release/{mp2sht,sht2png} and "
            "data/'Ni {20kV 75.7deg}.sht' to run this test"
        )
    return Path(value)


def _emsphinx_ni_file() -> Path:
    """Return the shipped EMSphInx Ni file, skipping if missing."""
    fpath = _emsphinx_dir() / "data" / "Ni {20kV 75.7deg}.sht"
    if not fpath.is_file():
        pytest.skip(f"{fpath} not found in the EMSphInx checkout")
    return fpath


def _emsphinx_program(name: str) -> Path:
    """Return an EMSphInx program, skipping if it is not built."""
    directory = _emsphinx_dir() / "build" / "Release"
    for candidate in (directory / f"{name}.exe", directory / name):
        if candidate.is_file():
            return candidate
    pytest.skip(f"{name} not built in {directory}")


def _md5(fpath: Path) -> str:
    """Return the md5 sum of a file."""
    return hashlib.md5(fpath.read_bytes()).hexdigest()


def _rule_respecting_alm(bandwidth: int, z_rot: int, flags: int) -> np.ndarray:
    """Return coefficients the packing keeps in full.

    The recipe of ``requirements.md`` D16, without a random number
    generator so that it is bit reproducible: the real part of a kept
    entry is ``((7 m + 13 l + 3) % 17 - 8) / 8`` and the imaginary
    part ``((5 m + 11 l + 1) % 19 - 9) / 9``, zeroed to match the
    storage type of the row.
    """
    inv = bool(flags & _sht_file.FLAG_INVERSION)
    mirror_z = bool(flags & _sht_file.FLAG_MIRROR_Z)
    mirror_y = bool(flags & _sht_file.FLAG_MIRROR_Y)
    mirror_x = bool(flags & _sht_file.FLAG_MIRROR_X)
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for m in range(bandwidth):
        if z_rot > 1 and m % z_rot != 0:
            continue
        if mirror_y:
            kind = "real"
        elif mirror_x:
            kind = "real" if m % (2 * z_rot) == 0 else "imaginary"
        else:
            kind = "complex"
        for degree in range(m, bandwidth):
            if inv and degree % 2:
                continue
            if mirror_z and (degree + m) % 2:
                continue
            real = ((7 * m + 13 * degree + 3) % 17 - 8) / 8
            imag = ((5 * m + 11 * degree + 1) % 19 - 9) / 9
            if kind == "real" or m == 0:
                alm[m, degree] = complex(real, 0.0)
            elif kind == "imaginary":
                alm[m, degree] = complex(0.0, imag)
            else:
                alm[m, degree] = complex(real, imag)
    return alm


def _minimal_sht_file(
    bandwidth: int = 8, z_rot: int = 1, flags: int = 0
) -> _sht_file.ShtFile:
    """Return a small writeable file.

    Callers which need another field mutate the returned instance,
    which is unambiguous: a ``**kwargs`` dispatch on
    ``hasattr(sht.header, name)`` would silently target the header
    for the names both it and :class:`ShtFile` carry, ``modality``.
    """
    alm = _rule_respecting_alm(bandwidth, z_rot, flags)
    packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
    header = _sht_file.ShtHeader(
        software_version="kp-test",
        modality=_sht_file.MODALITY_EBSD,
        beam_energy=20.0,
        primary_angle=70.0,
    )
    sht = _sht_file.ShtFile(
        header=header,
        num_xtal=1,
        sg_eff=225,
        pijk=1,
        rot_sense=112,
        modality=_sht_file.MODALITY_EBSD,
        vendor=_sht_file.VENDOR_UNKNOWN,
        sim_meta_size=0,
        crystals=[_sht_file.ShtCrystal(sg_num=225)],
        simulations=[None],
        harmonics=_sht_file.ShtHarmonics(
            bandwidth=bandwidth,
            z_rot=z_rot,
            flags=flags,
            doub_cnt=packed.size,
            packed=packed,
        ),
    )
    return sht


class TestCrc32c:
    def test_check_values_are_the_shtfile_variant(self):
        # The standard CRC-32C would give 0xE3069283 for b"123456789",
        # so a switch to the standard polynomial is caught here
        assert _sht_file.crc32c(b"") == 0
        assert _sht_file.crc32c(b"\x00" * 8) == 0xEBE76DE3
        assert _sht_file.crc32c(b"123456789") == 0xF28417BE
        assert _sht_file.crc32c(b"123456789") != 0xE3069283

    def test_the_checksum_chains(self):
        first, second = b"kikuchipy ", b"spherical indexing"
        chained = _sht_file.crc32c(second, _sht_file.crc32c(first))
        assert chained == _sht_file.crc32c(first + second)

    @pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
    def test_every_buffer_type_gives_the_same_value(self, kind):
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        assert _sht_file.crc32c(kind(data)) == _sht_file.crc32c(data)

    @pytest.mark.parametrize(
        "index, value",
        [
            (0, 0x00000000),
            (1, 0x0A5F4D75),
            (2, 0x14BE9AEA),
            (16, 0x15DECED9),
            (64, 0x11B258E1),
            (128, 0x1EDC6F41),
            # First entries of the 25th and 26th literal rows
            (192, 0x0F6E37A0),
            (200, 0x1B5D3F8D),
            (255, 0x12A28EAD),
        ],
    )
    def test_the_generated_table_equals_the_literal_one(self, index, value):
        # The table is generated from the polynomial at import; these
        # spot values come from the literal table of
        # sht_file.in.hpp lines 967-1000, indexed by their true index
        assert len(_sht_file._CRC_TABLE) == 256
        assert _sht_file._CRC_TABLE[index] == value

    def test_the_table_sum_is_the_recorded_value(self):
        # Determined and frozen 2026-08-16. It is *not* 2 ** 36:
        # 2 ** 36 == 68719476736, which is 128 larger. The spec
        # parenthetical was corrected 2026-08-16
        assert sum(_sht_file._CRC_TABLE) == 68719476608
        assert sum(_sht_file._CRC_TABLE) != 2**36

    @pytest.mark.parametrize(
        "name, checksum", [(NI_SMALL, 0xE3100CFF), (NI_FULL, 0xEA2875D2)]
    )
    def test_the_in_package_file_checksums(self, name, checksum):
        data = _data_path(name).read_bytes()
        assert len(data) == SHT_FILE_SIZE
        assert struct.unpack("<I", data[-4:])[0] == checksum
        assert _sht_file.crc32c(data[:-4]) == checksum

    def test_crc32c_timing_is_recorded(self, record_property):
        # The plain-Python tuple lookup table over a bytes object was
        # measured at 3.9 ms for this file; the NumPy scalar variant
        # at 51 ms, 13 times slower, and is not used (D10)
        data = _data_path(NI_SMALL).read_bytes()
        start = time.perf_counter()
        _sht_file.crc32c(data)
        duration = time.perf_counter() - start
        record_property("crc32c_74828_bytes_s", f"{duration:.4f}")
        # 50 ms is between the required 3.9 ms and the 51 ms of the
        # documented-wrong NumPy scalar variant, so the bound fails
        # for it instead of passing for everything
        assert duration < 0.05


class TestSpaceGroupTables:
    def test_both_tables_have_230_entries(self):
        assert len(_sht_file._SPACE_GROUP_ROT) == 230
        assert len(_sht_file._SPACE_GROUP_CMP) == 230

    def test_the_table_checksums(self):
        assert sum(_sht_file._SPACE_GROUP_ROT) == 707
        assert sum(_sht_file._SPACE_GROUP_CMP) == 948

    def test_the_z_rotation_histogram(self):
        counts = {}
        for value in _sht_file._SPACE_GROUP_ROT:
            counts[value] = counts.get(value, 0) + 1
        assert counts == {1: 15, 2: 91, 3: 30, 4: 72, 6: 22}

    def test_the_tables_have_25_distinct_pairs(self):
        pairs = set(zip(_sht_file._SPACE_GROUP_ROT, _sht_file._SPACE_GROUP_CMP))
        assert len(pairs) == 25
        lowest = {}
        for sg, pair in enumerate(
            zip(_sht_file._SPACE_GROUP_ROT, _sht_file._SPACE_GROUP_CMP), start=1
        ):
            lowest.setdefault(pair, sg)
        assert tuple(sorted(lowest.values())) == SYNTHETIC_SPACE_GROUPS

    def test_the_z_rotation_values_are_crystallographic(self):
        for sg in range(1, 231):
            assert _sht_file.space_group_z_rotation(sg) in {1, 2, 3, 4, 6}

    def test_the_compression_flags_are_legal(self):
        for sg in range(1, 231):
            flags = _sht_file.space_group_compression_flags(sg)
            assert flags in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xA}
            # 0x04 | 0x08 is mutually exclusive
            assert flags & 0x0C != 0x0C

    @pytest.mark.parametrize(
        "space_group, z_rot, flags",
        [
            (1, 1, 0x0),
            (2, 1, 0x1),
            (6, 1, 0x4),
            (10, 1, 0x5),
            (16, 2, 0x0),
            (25, 2, 0x4),
            (47, 2, 0x7),
            (75, 4, 0x0),
            (83, 4, 0x3),
            (111, 2, 0x8),
            (115, 2, 0x4),
            (123, 4, 0x7),
            (143, 3, 0x0),
            (156, 3, 0x8),
            (157, 3, 0x4),
            (164, 3, 0x9),
            (174, 3, 0x2),
            (187, 3, 0xA),
            (189, 3, 0x6),
            (225, 4, 0x7),
        ],
    )
    def test_spot_values(self, space_group, z_rot, flags):
        assert _sht_file.space_group_z_rotation(space_group) == z_rot
        assert _sht_file.space_group_compression_flags(space_group) == flags

    @pytest.mark.parametrize("space_group", [0, 231, -1])
    def test_a_space_group_outside_the_range_raises(self, space_group):
        with pytest.raises(ValueError):
            _sht_file.space_group_z_rotation(space_group)
        with pytest.raises(ValueError):
            _sht_file.space_group_compression_flags(space_group)

    @staticmethod
    def _orix_flags(space_group):
        """Return ``(z order, inversion, z mirror, equatorial axis
        mirror)`` of a space group's orix point group.
        """
        from orix.crystal_map import Phase

        symmetry = Phase(space_group=space_group).point_group
        proper = symmetry[~symmetry.improper]
        axis = proper.axis.data
        angle = proper.angle
        along_z = np.isclose(np.abs(axis[:, 2]), 1) & (angle > 1e-6)
        z_order = 1 + int(np.count_nonzero(along_z))
        improper = symmetry[symmetry.improper]
        imp_axis = improper.axis.data
        imp_angle = improper.angle
        two_fold = np.isclose(imp_angle, np.pi)
        z_mirror = bool(np.any(two_fold & np.isclose(np.abs(imp_axis[:, 2]), 1)))
        equator_mirror = bool(np.any(two_fold & np.isclose(np.abs(imp_axis[:, 2]), 0)))
        return z_order, symmetry.contains_inversion, z_mirror, equator_mirror

    @pytest.mark.parametrize("space_group", list(range(16, 231)))
    def test_the_tables_agree_with_orix_above_space_group_15(self, space_group):
        # All 215 space groups, as validation.md line 20 asks for;
        # a stride left 184 of them unchecked
        z_order, inversion, z_mirror, equator_mirror = self._orix_flags(space_group)
        flags = _sht_file.space_group_compression_flags(space_group)
        assert _sht_file.space_group_z_rotation(space_group) == z_order
        assert bool(flags & _sht_file.FLAG_INVERSION) == bool(inversion)
        assert bool(flags & _sht_file.FLAG_MIRROR_Z) == z_mirror
        assert bool(flags & 0x0C) == equator_mirror

    @pytest.mark.parametrize("space_group", list(range(3, 16)))
    def test_the_tables_disagree_with_orix_for_the_monoclinic_groups(self, space_group):
        # Recorded, not discovered: the SHT file tables assume the
        # standard monoclinic setting with unique axis *b* while orix
        # returns z-unique point groups ("2", "m", "2/m") for space
        # groups 3-15. The two-fold is then about y and not about z,
        # so the table says z order 1 where orix says 2; and for 6-15
        # the table sets 0x4, a mirror plane *containing* z, where
        # orix's z-unique groups have a mirror plane *perpendicular*
        # to z, i.e. bit 0x2. The two mirror kinds are not the same
        # flag, so this is a genuine disagreement and not a bug
        z_order, _, z_mirror, _ = self._orix_flags(space_group)
        table_rot = _sht_file.space_group_z_rotation(space_group)
        table_cmp = _sht_file.space_group_compression_flags(space_group)
        assert table_rot == 1
        if space_group in (3, 4, 5) or space_group >= 10:
            assert z_order == 2
        if space_group >= 6:
            assert table_cmp & _sht_file.FLAG_MIRROR_Y
            assert not table_cmp & _sht_file.FLAG_MIRROR_Z
            assert z_mirror


class TestNumHarmonics:
    def test_the_nickel_count(self):
        assert _sht_file.num_harmonics(384, 4, 0x7) == 9312
        assert 9312 == 96 * 97

    @pytest.mark.parametrize("bandwidth", [4, 16, 17, 64, 384])
    def test_the_complex_closed_form(self, bandwidth):
        assert _sht_file.num_harmonics(bandwidth, 1, 0) == bandwidth * (bandwidth + 1)

    @pytest.mark.parametrize("bandwidth", [4, 16, 17, 64, 384])
    def test_the_real_closed_form(self, bandwidth):
        assert (
            _sht_file.num_harmonics(bandwidth, 1, _sht_file.FLAG_MIRROR_Y)
            == bandwidth * (bandwidth + 1) // 2
        )

    @pytest.mark.parametrize("space_group", SYNTHETIC_SPACE_GROUPS)
    @pytest.mark.parametrize("bandwidth", [4, 16, 17])
    def test_pack_length_equals_the_count(self, space_group, bandwidth):
        z_rot, flags = SYNTHETIC_FLAG_PAIRS[space_group]
        alm = _rule_respecting_alm(bandwidth, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        assert packed.size == _sht_file.num_harmonics(bandwidth, z_rot, flags)
        assert packed.dtype == np.float64

    def test_illegal_flags_raise_everywhere(self):
        flags = _sht_file.FLAG_MIRROR_Y | _sht_file.FLAG_MIRROR_X
        alm = np.zeros((8, 8), dtype=np.complex128)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _sht_file.num_harmonics(8, 1, flags)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _sht_file.pack_harmonics(alm, 8, 1, flags)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _sht_file.unpack_harmonics(np.zeros(1), 8, 1, flags)


class TestPackUnpack:
    @pytest.mark.parametrize("space_group", SYNTHETIC_SPACE_GROUPS)
    @pytest.mark.parametrize("bandwidth", [16, 17])
    def test_unpack_of_pack_is_the_identity(self, space_group, bandwidth):
        z_rot, flags = SYNTHETIC_FLAG_PAIRS[space_group]
        alm = _rule_respecting_alm(bandwidth, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        back = _sht_file.unpack_harmonics(packed, bandwidth, z_rot, flags)
        assert np.array_equal(back, alm)

    @pytest.mark.parametrize("space_group", SYNTHETIC_SPACE_GROUPS)
    @pytest.mark.parametrize("bandwidth", [16, 17])
    def test_pack_of_unpack_is_the_identity(self, space_group, bandwidth):
        z_rot, flags = SYNTHETIC_FLAG_PAIRS[space_group]
        count = _sht_file.num_harmonics(bandwidth, z_rot, flags)
        payload = np.arange(1, count + 1, dtype=np.float64) / count
        alm = _sht_file.unpack_harmonics(payload, bandwidth, z_rot, flags)
        again = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        assert np.array_equal(again, payload)

    def test_the_complex_branch_writes_two_doubles_per_entry(self):
        bandwidth, z_rot, flags = 16, 1, 0x0
        alm = _rule_respecting_alm(bandwidth, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        kept = bandwidth * (bandwidth + 1) // 2
        assert packed.size == 2 * kept
        # Interleaved real, imaginary in m-major then l order
        expected = []
        for m in range(bandwidth):
            for degree in range(m, bandwidth):
                expected += [alm[m, degree].real, alm[m, degree].imag]
        assert np.array_equal(packed, np.asarray(expected))

    def test_the_real_branch_writes_the_real_parts(self):
        bandwidth, z_rot, flags = 16, 2, _sht_file.FLAG_MIRROR_Y
        alm = _rule_respecting_alm(bandwidth, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        expected = [
            alm[m, degree].real
            for m in range(0, bandwidth, 2)
            for degree in range(m, bandwidth)
        ]
        assert packed.size == len(expected)
        assert np.array_equal(packed, np.asarray(expected))

    def test_the_mirror_x_branch_alternates_real_and_imaginary(self):
        # Space group 111: z_rot 2, flags 0x8. Rows with
        # m % (2 * z_rot) == 0, i.e. m % 4 == 0, are real; the other
        # kept rows, m % 4 == 2, are imaginary
        bandwidth, z_rot, flags = 16, 2, _sht_file.FLAG_MIRROR_X
        alm = _rule_respecting_alm(bandwidth, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
        expected = []
        for m in range(0, bandwidth, 2):
            for degree in range(m, bandwidth):
                if m % 4 == 0:
                    expected.append(alm[m, degree].real)
                else:
                    expected.append(alm[m, degree].imag)
        assert packed.size == len(expected)
        assert np.array_equal(packed, np.asarray(expected))

    def test_the_nickel_flags_skip_rows_and_parities(self):
        # Space group 225: z_rot 4, flags 0x7 = inversion, mirror z,
        # mirror y. Rows m % 4 != 0 are skipped, odd l is skipped and
        # odd l + m is skipped
        bandwidth, z_rot, flags = 16, 4, 0x7
        count = _sht_file.num_harmonics(bandwidth, z_rot, flags)
        payload = np.arange(1, count + 1, dtype=np.float64)
        alm = _sht_file.unpack_harmonics(payload, bandwidth, z_rot, flags)
        for m in range(bandwidth):
            for degree in range(bandwidth):
                value = alm[m, degree]
                if degree < m or m % 4 != 0 or degree % 2 or (degree + m) % 2:
                    assert value == 0
                else:
                    assert value.imag == 0
                    assert value.real != 0

    @pytest.mark.parametrize("name", [NI_SMALL, NI_FULL])
    def test_the_nickel_payload_round_trips_bitwise(self, name):
        sht = _sht_file.read_sht(_data_path(name))
        harmonics = sht.harmonics
        alm = _sht_file.unpack_harmonics(
            harmonics.packed,
            harmonics.bandwidth,
            harmonics.z_rot,
            harmonics.flags,
        )
        again = _sht_file.pack_harmonics(
            alm, harmonics.bandwidth, harmonics.z_rot, harmonics.flags
        )
        assert np.array_equal(again, harmonics.packed)
        # Only m % 4 == 0, even l, real values
        nonzero = np.nonzero(alm)
        assert np.all(nonzero[0] % 4 == 0)
        assert np.all(nonzero[1] % 2 == 0)
        assert np.all(alm.imag == 0)

    def test_unpack_of_a_wrong_length_payload_raises(self):
        with pytest.raises(ValueError):
            _sht_file.unpack_harmonics(np.zeros(3), 16, 1, 0)

    def test_pack_of_a_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            _sht_file.pack_harmonics(np.zeros((4, 5), dtype=np.complex128), 4, 1, 0)


class TestHeaderParse:
    def test_the_header_fields(self):
        ni_small = _read(NI_SMALL)
        header = ni_small.header
        assert header.magic == b"*sht"
        assert header.file_version == (1, 1)
        assert header.software_version == "ve49ad6b"
        assert header.modality == _sht_file.MODALITY_EBSD
        assert header.beam_energy == pytest.approx(20.1, abs=1e-6)
        assert header.primary_angle == 70
        assert header.secondary_angle == 0
        assert header.doi == "https://doi.org/10.1016/j.ultramic.2019.112841"
        assert header.doi_len == 46
        assert header.notes == "created with mp2sht"
        assert header.note_len == 19
        assert header.res_bytes == (0, 0)
        assert header.res_bytes2 == (0, 0, 0)

    def test_the_master_pattern_data_fields(self):
        ni_small = _read(NI_SMALL)
        assert ni_small.num_xtal == 1
        assert ni_small.sg_eff == 225
        assert ni_small.pijk == 1
        assert ni_small.rot_sense == 112
        assert ni_small.modality == _sht_file.MODALITY_EBSD
        assert ni_small.vendor == _sht_file.VENDOR_EMSOFT
        assert ni_small.sim_meta_size == 88

    def test_the_crystal_fields(self):
        ni_small = _read(NI_SMALL)
        crystal = ni_small.crystals[0]
        assert crystal.sg_num == 225
        assert crystal.sg_set == 1
        assert crystal.sg_axis == 1
        assert crystal.sg_cell == 1
        assert crystal.origin == (0, 0, 0)
        assert crystal.lat == pytest.approx((0.35236,) * 3 + (90.0,) * 3, rel=1e-6)
        assert crystal.rot == (1, 0, 0, 0)
        assert crystal.weight == 1
        assert crystal.num_atoms == 1
        assert (
            crystal.formula_len,
            crystal.material_name_len,
            crystal.structure_symbol_len,
            crystal.references_len,
            crystal.note_len,
        ) == (2, 0, 0, 0, 0)
        assert crystal.formula == "Ni"

    def test_the_atom_fields(self):
        ni_small = _read(NI_SMALL)
        atom = ni_small.crystals[0].atoms[0]
        assert (atom.x, atom.y, atom.z) == (0, 0, 0)
        assert atom.occupancy == 1
        assert atom.charge == 0
        assert atom.debye_waller == pytest.approx(0.0035)
        assert atom.atomic_number == 28

    def test_the_emsoft_simulation_fields(self):
        ni_small = _read(NI_SMALL)
        sim = ni_small.simulations[0]
        assert isinstance(sim, _sht_file.ShtEMsoftSimulation)
        assert sim.emsoft_version == "5_0_0_0"
        assert sim.sig_start == 70
        assert np.isnan(sim.sig_end)
        assert np.isnan(sim.sig_step)
        assert sim.omega == 0
        assert sim.kev == pytest.approx(20.1)
        assert sim.e_hist_min == 20
        assert sim.e_bin_size == 1
        assert sim.depth_max == 100
        assert sim.depth_step == 1
        assert np.isinf(sim.thickness)
        assert sim.tot_num_el == 2_000_000_000
        assert sim.num_sx == 201
        assert (sim.c1, sim.c2, sim.c3) == (4, 8, 50)
        assert sim.sig_db_diff == 1
        assert sim.d_min == pytest.approx(0.05)
        assert sim.num_px == 200
        assert sim.lat_grid_type == _sht_file.LAT_GRID_LAMBERT

    def test_the_harmonics_fields(self):
        ni_small = _read(NI_SMALL)
        harmonics = ni_small.harmonics
        assert harmonics.bandwidth == 384
        assert harmonics.z_rot == 4
        assert harmonics.flags == 0x7
        assert harmonics.doub_cnt == 9312
        assert harmonics.packed.shape == (9312,)
        assert harmonics.packed.dtype == np.float64

    def test_the_block_offsets(self):
        ni_small = _read(NI_SMALL)
        offsets = ni_small.block_offsets
        assert offsets["master_pattern"] == 112
        assert offsets["crystal_0"] == 120
        assert offsets["simulation_0"] == 232
        assert offsets["harmonics"] == 320
        assert offsets["payload"] == 328
        assert offsets["crc"] == 74824

    def test_the_full_master_header(self):
        ni_full = _read(NI_FULL)
        assert ni_full.header.beam_energy == pytest.approx(20.0, abs=1e-6)
        assert ni_full.simulations[0].e_hist_min == 5
        assert ni_full.simulations[0].num_sx == 501
        assert ni_full.simulations[0].num_px == 500
        assert ni_full.simulations[0].kev == pytest.approx(20.0)
        assert ni_full.harmonics.doub_cnt == 9312

    @pytest.mark.parametrize("name", [NI_SMALL, NI_FULL])
    def test_the_payload_length_matches_the_count_and_the_file(self, name):
        fpath = _data_path(name)
        sht = _sht_file.read_sht(fpath)
        harmonics = sht.harmonics
        assert harmonics.doub_cnt == _sht_file.num_harmonics(
            harmonics.bandwidth, harmonics.z_rot, harmonics.flags
        )
        payload_offset = sht.block_offsets["payload"]
        assert len(harmonics.packed) * 8 == fpath.stat().st_size - payload_offset - 4


class TestEmsphinxShippedFile:
    """The shipped ``data/Ni {20kV 75.7deg}.sht`` is a *different*
    master (75.7 degree tilt) and is never redistributed, so these
    tests need ``KIKUCHIPY_EMSPHINX_DIR``.
    """

    def test_emsphinx_binaries_the_shipped_file_parses(self, record_property):
        fpath = _emsphinx_ni_file()
        sht = _sht_file.read_sht(fpath)
        assert fpath.stat().st_size == SHT_FILE_SIZE
        assert sht.header.software_version == "ve49ad6b"
        assert sht.header.beam_energy == pytest.approx(20.0, abs=1e-6)
        assert sht.header.primary_angle == pytest.approx(75.7, abs=1e-5)
        assert sht.header.doi_len == 46
        assert sht.sg_eff == 225
        assert sht.simulations[0].sig_start == pytest.approx(75.7)
        assert sht.simulations[0].kev == pytest.approx(20.0)
        assert sht.simulations[0].e_hist_min == 5
        assert sht.simulations[0].num_sx == 501
        assert sht.simulations[0].num_px == 500
        assert sht.harmonics.doub_cnt == 9312
        assert sht.crc == 0xF2AF93EF
        record_property("emsphinx_shipped_crc", hex(sht.crc))

    def test_emsphinx_binaries_the_shipped_payload_has_the_recorded_dc_term(self):
        sht = _sht_file.read_sht(_emsphinx_ni_file())
        alm = _sht_file.unpack_harmonics(
            sht.harmonics.packed,
            sht.harmonics.bandwidth,
            sht.harmonics.z_rot,
            sht.harmonics.flags,
        )
        assert alm[0, 0].real == pytest.approx(-3.2555, abs=1e-3)

    def test_emsphinx_binaries_the_shipped_file_rewrites_byte_identically(self):
        fpath = _emsphinx_ni_file()
        data = fpath.read_bytes()
        assert _sht_file.sht_file_to_bytes(_sht_file.read_sht(data)) == data

    def test_emsphinx_binaries_the_shipped_count_matches_the_payload(self):
        # validation.md lines 21-22 ask for these on *every* .sht in
        # the suite, the shipped one included
        fpath = _emsphinx_ni_file()
        sht = _sht_file.read_sht(fpath)
        harmonics = sht.harmonics
        assert harmonics.doub_cnt == _sht_file.num_harmonics(
            harmonics.bandwidth, harmonics.z_rot, harmonics.flags
        )
        payload_offset = sht.block_offsets["payload"]
        assert len(harmonics.packed) * 8 == fpath.stat().st_size - payload_offset - 4

    def test_emsphinx_binaries_the_shipped_payload_packs_back_bitwise(self):
        sht = _sht_file.read_sht(_emsphinx_ni_file())
        harmonics = sht.harmonics
        alm = _sht_file.unpack_harmonics(
            harmonics.packed,
            harmonics.bandwidth,
            harmonics.z_rot,
            harmonics.flags,
        )
        again = _sht_file.pack_harmonics(
            alm, harmonics.bandwidth, harmonics.z_rot, harmonics.flags
        )
        assert np.array_equal(again, harmonics.packed)


class TestByteIdentity:
    @pytest.mark.parametrize("name", [NI_SMALL, NI_FULL])
    def test_read_then_serialize_is_byte_identical(self, name):
        # The external oracles: these files were written by
        # EMSphInx' mp2sht, not by us
        data = _data_path(name).read_bytes()
        assert _sht_file.sht_file_to_bytes(_sht_file.read_sht(data)) == data

    @pytest.mark.parametrize("name", [NI_SMALL, NI_FULL])
    def test_to_dict_is_lossless(self, name):
        sht = _sht_file.read_sht(_data_path(name))
        again = _sht_file.ShtFile.from_dict(sht.to_dict())
        # Dataclass equality cannot compare the ndarray payload, so
        # byte identity is the equality here
        assert _sht_file.sht_file_to_bytes(again) == _sht_file.sht_file_to_bytes(sht)

    def test_write_read_write_is_idempotent(self, emsphinx_synthetic_sht_files):
        files = emsphinx_synthetic_sht_files()
        assert len(files) == 25
        for space_group, fpath in sorted(files.items()):
            data = fpath.read_bytes()
            once = _sht_file.sht_file_to_bytes(_sht_file.read_sht(data))
            twice = _sht_file.sht_file_to_bytes(_sht_file.read_sht(once))
            assert once == twice, space_group
            assert once == data, space_group

    def test_the_generated_fixture_md5_sums_are_the_pinned_ones(
        self, emsphinx_synthetic_sht_files, record_property
    ):
        files = emsphinx_synthetic_sht_files()
        measured = {sg: _md5(f) for sg, f in sorted(files.items())}
        record_property("synthetic_sht_md5", repr(measured))
        assert set(measured) == set(SYNTHETIC_SPACE_GROUPS)
        assert SYNTHETIC_MD5, (
            "the 25 md5 sums must be pinned in SYNTHETIC_MD5 after the "
            "one-off sht2png.exe acceptance of plan.md task 2.3(c); "
            f"measured {measured!r}"
        )
        assert measured == SYNTHETIC_MD5

    def test_every_generated_fixture_has_the_expected_flags(
        self, emsphinx_synthetic_sht_files
    ):
        files = emsphinx_synthetic_sht_files()
        for space_group, fpath in sorted(files.items()):
            sht = _sht_file.read_sht(fpath)
            harmonics = sht.harmonics
            assert sht.sg_eff == space_group
            assert (harmonics.z_rot, harmonics.flags) == SYNTHETIC_FLAG_PAIRS[
                space_group
            ]
            assert harmonics.doub_cnt == _sht_file.num_harmonics(
                harmonics.bandwidth, harmonics.z_rot, harmonics.flags
            )
            payload_offset = sht.block_offsets["payload"]
            assert (
                len(harmonics.packed) * 8 == fpath.stat().st_size - payload_offset - 4
            )


class TestDictViews:
    def test_the_metadata_dict_keys(self):
        ni_small = _read(NI_SMALL)
        metadata = ni_small.metadata_dict()
        assert set(metadata) == {
            "header",
            "master_pattern",
            "crystals",
            "simulations",
            "harmonics",
        }

    def test_the_crystals_and_simulations_are_numbered_nodes(self):
        ni_small = _read(NI_SMALL)
        metadata = ni_small.metadata_dict()
        assert set(metadata["crystals"]) == {"crystal_0"}
        assert set(metadata["simulations"]) == {"simulation_0"}
        assert isinstance(metadata["crystals"]["crystal_0"], dict)
        assert metadata["crystals"]["crystal_0"]["formula"] == "Ni"
        assert metadata["simulations"]["simulation_0"]["num_px"] == 200

    def test_the_metadata_dict_carries_no_payload_and_no_raw_bytes(self):
        ni_small = _read(NI_SMALL)
        metadata = ni_small.metadata_dict()
        assert "packed" not in metadata["harmonics"]
        assert metadata["harmonics"]["bandwidth"] == 384
        assert metadata["harmonics"]["z_rot"] == 4
        assert metadata["harmonics"]["doub_cnt"] == 9312

        def walk(node, path=""):
            for key, value in node.items():
                assert not str(key).endswith("_bytes"), f"{path}{key}"
                if isinstance(value, dict):
                    walk(value, f"{path}{key}.")

        walk(metadata)

    def test_a_two_crystal_file_gives_two_numbered_nodes(self, tmp_path):
        sht = _minimal_sht_file()
        sht.num_xtal = 2
        sht.crystals = [
            _sht_file.ShtCrystal(sg_num=225),
            _sht_file.ShtCrystal(sg_num=194),
        ]
        sht.simulations = [None, None]
        fpath = tmp_path / "two.sht"
        _sht_file.write_sht(fpath, sht)
        again = _sht_file.read_sht(fpath)
        assert again.num_xtal == 2
        metadata = again.metadata_dict()
        assert set(metadata["crystals"]) == {"crystal_0", "crystal_1"}
        assert set(metadata["simulations"]) == {"simulation_0", "simulation_1"}


class TestRobustness:
    def test_a_flipped_payload_byte_fails_the_checksum(self):
        data = bytearray(_data_path(NI_SMALL).read_bytes())
        data[1000] ^= 0xFF
        with pytest.raises(ValueError, match="checksum"):
            _sht_file.read_sht(bytes(data))

    def test_check_crc_false_parses_a_corrupt_file(self):
        data = bytearray(_data_path(NI_SMALL).read_bytes())
        data[1000] ^= 0xFF
        sht = _sht_file.read_sht(bytes(data), check_crc=False)
        assert sht.harmonics.bandwidth == 384

    def test_big_endian_magic_raises(self):
        data = bytearray(_data_path(NI_SMALL).read_bytes())
        data[:4] = _sht_file.MAGIC_BE
        with pytest.raises(NotImplementedError, match="big-endian"):
            _sht_file.read_sht(bytes(data), check_crc=False)

    def test_version_one_zero_raises(self):
        data = bytearray(_data_path(NI_SMALL).read_bytes())
        data[5] = 0
        with pytest.raises(NotImplementedError, match="1.1"):
            _sht_file.read_sht(bytes(data), check_crc=False)

    def test_a_wrong_magic_raises(self):
        data = bytearray(_data_path(NI_SMALL).read_bytes())
        data[:4] = b"HDF\x89"
        with pytest.raises(ValueError, match="not an SHT"):
            _sht_file.read_sht(bytes(data), check_crc=False)

    def test_an_opaque_simulation_record_survives_a_round_trip(self, tmp_path):
        # EMSphInx itself refuses this file (sht_file.in.hpp line
        # 1605), while we keep the record as bytes so that the
        # checksum and a byte identical rewrite still work
        record = bytes(range(32))
        sht = _minimal_sht_file()
        sht.header.modality = _sht_file.MODALITY_LAUE
        sht.modality = _sht_file.MODALITY_LAUE
        sht.vendor = _sht_file.VENDOR_EMSOFT
        sht.sim_meta_size = 32
        sht.simulations = [record]
        fpath = tmp_path / "laue.sht"
        _sht_file.write_sht(fpath, sht)
        again = _sht_file.read_sht(fpath)
        assert again.simulations[0] == record
        assert _sht_file.sht_file_to_bytes(again) == fpath.read_bytes()

    @pytest.mark.parametrize(
        "length, padded", [(0, 0), (1, 8), (7, 8), (8, 8), (9, 16), (46, 48)]
    )
    def test_string_padding(self, length, padded, tmp_path):
        assert _sht_file._pad8(length) == padded
        sht = _minimal_sht_file()
        sht.header.doi = "d" * length
        fpath = tmp_path / "doi.sht"
        _sht_file.write_sht(fpath, sht)
        again = _sht_file.read_sht(fpath)
        assert again.header.doi_len == length
        assert len(again.header.doi_bytes) == padded
        assert again.header.doi == "d" * length

    def test_non_utf8_notes_and_a_non_zero_pad_byte_round_trip(self, tmp_path):
        # A writer which decodes and re-encodes would change byte 4
        # (the 0xe9) and byte 7 (the pad) and therefore the checksum
        raw = b"caf\xe9\x00\x00\x00\x01"
        sht = _minimal_sht_file()
        sht.header.notes = "caf�"
        sht.header.note_len = 4
        sht.header.notes_bytes = raw
        fpath = tmp_path / "latin.sht"
        _sht_file.write_sht(fpath, sht)
        data = fpath.read_bytes()
        again = _sht_file.read_sht(data)
        assert again.header.notes == "caf�"
        assert again.header.notes_bytes == raw
        assert _sht_file.sht_file_to_bytes(again) == data


class TestWriterSanityChecks:
    """Every check ``File::sanityCheck`` runs, with its wording."""

    def test_a_bandwidth_above_the_field_raises(self):
        sht = _minimal_sht_file()
        sht.harmonics.bandwidth = 32768
        with pytest.raises(ValueError, match="32767"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_negative_beam_energy_raises(self):
        sht = _minimal_sht_file()
        sht.header.beam_energy = -1
        with pytest.raises(ValueError, match="negative beam energy is non-physical"):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_unrealistic_beam_energy_raises(self):
        sht = _minimal_sht_file()
        sht.header.beam_energy = 10001
        with pytest.raises(ValueError, match="10 MeV beam energy is unrealistic"):
            _sht_file.sht_file_to_bytes(sht)

    @pytest.mark.parametrize(
        "field, value, message",
        [
            ("primary_angle", 361, "primary angle outside [-360,360]"),
            ("secondary_angle", -361, "secondary angle outside [-360,360]"),
        ],
    )
    def test_an_angle_outside_the_range_raises(self, field, value, message):
        sht = _minimal_sht_file()
        setattr(sht.header, field, value)
        with pytest.raises(ValueError, match=re.escape(message)):
            _sht_file.sht_file_to_bytes(sht)

    def test_non_zero_reserved_bytes_raise(self):
        # The two reserved fields have different wordings in
        # FileHeader::sanityCheck, so a plain "reserved bytes" would
        # not tell them apart
        sht = _minimal_sht_file()
        sht.header.res_bytes = (1, 0)
        with pytest.raises(ValueError, match="non-zero reserved bytes"):
            _sht_file.sht_file_to_bytes(sht)

    def test_non_zero_second_reserved_bytes_raise(self):
        sht = _minimal_sht_file()
        sht.header.res_bytes2 = (0, 2, 0)
        with pytest.raises(ValueError, match="reserved bytes must be 0"):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_unknown_modality_raises(self):
        sht = _minimal_sht_file()
        sht.header.modality = 5
        sht.modality = 5
        with pytest.raises(ValueError, match="invalid modality flag"):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_unknown_vendor_raises(self):
        sht = _minimal_sht_file()
        sht.vendor = 2
        with pytest.raises(ValueError, match="invalid vendor flag"):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_effective_space_group_of_zero_raises(self):
        sht = _minimal_sht_file()
        sht.sg_eff = 0
        with pytest.raises(ValueError, match="invalid effective space group number"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_zero_pijk_raises(self):
        sht = _minimal_sht_file()
        sht.pijk = 0
        with pytest.raises(ValueError, match=re.escape("pijk must be +/-1")):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_unknown_rotation_sense_raises(self):
        sht = _minimal_sht_file()
        sht.rot_sense = 98
        with pytest.raises(
            ValueError, match=re.escape("rotation sense must be 'a' or 'p'")
        ):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_crystal_count_mismatch_raises(self):
        sht = _minimal_sht_file()
        sht.num_xtal = 2
        with pytest.raises(ValueError, match=r"# crystals != crystals size"):
            _sht_file.sht_file_to_bytes(sht)

    def test_zero_crystals_raise(self):
        # Ours: EMSphInx dereferences simul.front() unconditionally,
        # which is undefined behaviour for an empty list
        sht = _minimal_sht_file()
        sht.num_xtal = 0
        sht.crystals = []
        sht.simulations = []
        with pytest.raises(ValueError, match="at least one crystal"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_missing_simulation_record_raises(self):
        sht = _minimal_sht_file()
        sht.sim_meta_size = 88
        sht.simulations = [None]
        with pytest.raises(ValueError, match="NULL simulation data for nonzero size"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_record_for_zero_size_raises(self):
        sht = _minimal_sht_file()
        sht.sim_meta_size = 0
        sht.vendor = _sht_file.VENDOR_EMSOFT
        sht.simulations = [_sht_file.ShtEMsoftSimulation()]
        with pytest.raises(ValueError, match="non-NULL simulation data for 0 size"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_record_size_mismatch_raises(self):
        sht = _minimal_sht_file()
        sht.sim_meta_size = 32
        sht.vendor = _sht_file.VENDOR_EMSOFT
        sht.simulations = [_sht_file.ShtEMsoftSimulation()]
        with pytest.raises(
            ValueError, match="simulation data size doesn't match header size"
        ):
            _sht_file.sht_file_to_bytes(sht)

    def test_an_emsoft_record_under_a_laue_modality_raises(self):
        sht = _minimal_sht_file()
        sht.header.modality = _sht_file.MODALITY_LAUE
        sht.modality = _sht_file.MODALITY_LAUE
        sht.vendor = _sht_file.VENDOR_EMSOFT
        sht.sim_meta_size = 88
        sht.simulations = [_sht_file.ShtEMsoftSimulation()]
        # MasterPatternData::sanityCheck runs before the file level
        # cross check, so this is the record-against-block message and
        # not the unrelated "invalid modality flag"
        with pytest.raises(
            ValueError,
            match="simulation data modality not valid for master pattern modality",
        ):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_vendor_mismatch_raises(self):
        sht = _minimal_sht_file()
        sht.vendor = _sht_file.VENDOR_UNKNOWN
        sht.sim_meta_size = 88
        sht.simulations = [_sht_file.ShtEMsoftSimulation()]
        with pytest.raises(
            ValueError,
            match="simulation data vendor doesn't match master pattern verndor",
        ):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_doi_length_mismatch_raises(self):
        sht = _minimal_sht_file()
        sht.header.doi = "1234567"
        sht.header.doi_len = 7
        sht.header.doi_bytes = b"1234567"
        with pytest.raises(ValueError, match="doi string doesn't match length"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_notes_length_mismatch_raises(self):
        # EMSphInx' own typo, quoted so that a grep finds it
        sht = _minimal_sht_file()
        sht.header.notes = "1234567"
        sht.header.note_len = 7
        sht.header.notes_bytes = b"1234567"
        with pytest.raises(ValueError, match="noites string doesn't match length"):
            _sht_file.sht_file_to_bytes(sht)

    def test_a_harmonics_count_mismatch_raises(self):
        sht = _minimal_sht_file()
        sht.harmonics.doub_cnt += 1
        with pytest.raises(
            ValueError,
            match="harmonics count doesn't match compression parameters",
        ):
            _sht_file.sht_file_to_bytes(sht)


class TestLicenceHygiene:
    def test_the_module_imports_nothing_gpl_derived(self):
        # The module ships BSD-3-Clause, so it must not import the
        # GPL derived parts of kikuchipy; those import it
        allowed = {
            "numpy",
            "dataclasses",
            "struct",
            "math",
            "pathlib",
            "typing",
            "io",
            "os",
            "warnings",
        }
        source = Path(_sht_file.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                # A relative import is by definition from kikuchipy,
                # i.e. exactly the GPL leak this test guards, so it
                # must fail rather than go unrecorded
                assert node.level == 0, f"relative import of {node.module!r}"
                assert node.module is not None
                imported.add(node.module.split(".")[0])
        assert imported <= allowed, imported - allowed
        assert "kikuchipy" not in imported

    def test_the_bsd_pre_commit_hook_covers_the_module(self):
        text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        pattern = re.search(r"^\s*files: (\^src/kikuchipy/\(.*\))$", text, re.M)
        assert pattern is not None
        assert re.search(
            pattern.group(1),
            "src/kikuchipy/indexing/_spherical/_sht_file.py",
        )

    def test_the_gpl_pre_commit_hook_excludes_the_module(self):
        text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        pattern = re.search(r"^\s*exclude: (\^src/kikuchipy/\(.*\))$", text, re.M)
        assert pattern is not None
        assert re.search(
            pattern.group(1),
            "src/kikuchipy/indexing/_spherical/_sht_file.py",
        )

    def test_the_module_carries_the_bsd_header_and_the_shtfile_notice(self):
        source = Path(_sht_file.__file__).read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: BSD-3-Clause" in source
        assert "Copyright (c) 2019, De Graef Group" in source
        assert "Author William C. Lenthe" in source
        assert "Redistribution and use in source and binary forms" in source
        assert "https://github.com/EMsoft-org/SHTfile" in source
        assert "e49ad6b" in source


class TestEmsphinxBinaries:
    def test_emsphinx_binaries_sht2png_accepts_every_generated_fixture(
        self, emsphinx_synthetic_sht_files, tmp_path
    ):
        import subprocess

        program = _emsphinx_program("sht2png")
        files = emsphinx_synthetic_sht_files()
        for space_group, fpath in sorted(files.items()):
            out = tmp_path / f"sg{space_group:03d}.png"
            result = subprocess.run(
                [str(program), str(fpath), str(out)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert f"effective sg# {space_group}" in result.stdout
            # The acceptance and the pin must refer to the same bytes
            assert space_group in SYNTHETIC_MD5, (
                "the 25 md5 sums must be pinned in SYNTHETIC_MD5 after "
                "this acceptance run (plan.md task 2.3(c)); measured "
                f"{_md5(fpath)!r} for space group {space_group}"
            )
            assert _md5(fpath) == SYNTHETIC_MD5[space_group]
