#
# Copyright 2026 the kikuchipy developers
#
# SPDX-License-Identifier: BSD-3-Clause
#

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from SHTfile
# (https://github.com/EMsoft-org/SHTfile, commit e49ad6b, vendored by
# EMSphInx commit 60f3517 at
# ``build/_deps/shtfile-src/sht_file.in.hpp``):
# - The CRC-32C lookup table and update loop, ``crc32c()``
#   (lines 940-1005)
# - ``HarmonicsData::NumHarm()`` (lines 1672-1698),
#   ``HarmonicsData::PackHarm()`` (lines 1700-1756) and
#   ``HarmonicsData::UnpackHarm()`` (lines 1758-1831)
# - The ``HarmonicsData::SpaceGroupRot()`` and
#   ``HarmonicsData::SpaceGroupCmp()`` tables (lines 1837-1869)
# - The byte layouts of ``FileHeader`` (lines 138-257),
#   ``AtomData`` (lines 260-317), ``CrystalData`` (lines 320-472),
#   ``MasterPatternData`` (lines 475-539), ``HarmonicsData``
#   (lines 542-637) and ``EMsoftED`` (lines 741-849)
# - The string padding rule of ``FileHeader::setDoi()``/``setNotes()``
#   (lines 1182-1188)
# - The sanity checks of ``FileHeader::sanityCheck()`` (1066-1094),
#   ``MasterPatternData::sanityCheck()`` (1523-1554),
#   ``HarmonicsData::sanityCheck()`` (1625-1628) and
#   ``File::sanityCheck()`` (1970-1980), including their wording
# - The read and write order of ``File::read()``/``File::write()``
#   (lines 1995-2031)
# - The ``initFileEMsoft()``/``addDataEMsoft()`` field mapping
#   (lines 2047-2078, 2139-2230)

# #####################################################################
# Copyright (c) 2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author William C. Lenthe
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Modified by Johan Westraadt, 2026-08: translated to Python/NumPy for
# kikuchipy. BSD-3-Clause, as the original
# #####################################################################

"""Reading and writing of EMSphInx spherical harmonic master pattern
*.sht files, version 1.1 of the SHTfile format.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Notes
-----
This module is BSD-3-Clause licensed, unlike the rest of
:mod:`kikuchipy.indexing._spherical`, because it is derived from the
BSD-3-Clause licensed SHTfile library and not from the GPL licensed
EMSphInx.  It must therefore not import anything GPL derived: only
:mod:`numpy` and the standard library are imported here, and the GPL
modules import this one and never the reverse.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import struct  # noqa: F401  (used by the readers and writers)
from typing import Any

import numpy as np

# --------------------------- Constants ------------------------------ #

MAGIC_LE: bytes = b"*sht"
"""Magic bytes of a little-endian SHT file (``sht_file.in.hpp`` line
1163)."""

MAGIC_BE: bytes = b"*SHT"
"""Magic bytes of a big-endian SHT file, which is not supported."""

VERSION: tuple[int, int] = (1, 1)
"""Only supported SHT file version (``sht_file.in.hpp`` lines 57-58)."""

MODALITY_UNKNOWN: int = 0
MODALITY_EBSD: int = 1
MODALITY_ECP: int = 2
MODALITY_TKD: int = 3
MODALITY_PED: int = 0x11
MODALITY_LAUE: int = 0x21

MODALITY_NAMES: dict[int, str] = {
    MODALITY_UNKNOWN: "Unknown",
    MODALITY_EBSD: "EBSD",
    MODALITY_ECP: "ECP",
    MODALITY_TKD: "TKD",
    MODALITY_PED: "PED",
    MODALITY_LAUE: "Laue",
}
"""Modality flags of ``sht_file.in.hpp`` lines 97-104."""

VENDOR_UNKNOWN: int = 0
VENDOR_EMSOFT: int = 1

VENDOR_NAMES: dict[int, str] = {
    VENDOR_UNKNOWN: "Unknown",
    VENDOR_EMSOFT: "EMsoft",
}
"""Vendor flags of ``sht_file.in.hpp`` lines 106-109."""

FLAG_INVERSION: int = 0x01
"""Compression flag: coefficients with odd degree are systematic
zeros."""

FLAG_MIRROR_Z: int = 0x02
"""Compression flag: coefficients with odd ``l + m`` are systematic
zeros."""

FLAG_MIRROR_Y: int = 0x04
"""Compression flag: every stored coefficient is strictly real
(``Nmm`` type group)."""

FLAG_MIRROR_X: int = 0x08
"""Compression flag: rows with ``m % (2 * z_rot) == 0`` are strictly
real and the other rows strictly imaginary (rotated ``Nmm`` type
group)."""

LAT_GRID_LAMBERT: int = 1
LAT_GRID_LEGENDRE: int = 2

LAT_GRID_NAMES: dict[int, str] = {
    LAT_GRID_LAMBERT: "square lambert",
    LAT_GRID_LEGENDRE: "square legendre",
}
"""Latitude grid types of the EMsoftED simulation block, printed by
``programs/sht2png.cpp`` line 272."""

CRC_POLYNOMIAL: int = 0x1EDC6F41
"""Polynomial the CRC-32C lookup table is generated from
(``sht_file.in.hpp`` line 951).

This is the *normal* representation of the Castagnoli polynomial used
with a *reflected* update loop, so the resulting check values differ
from the standard CRC-32C ones.
"""

EMSOFT_ED_SIZE: int = 88
"""Size in bytes of an ``EMsoftED`` simulation record."""

MAX_BANDWIDTH: int = 32767
"""Largest bandwidth the 16-bit signed field can hold."""


def _build_crc_table() -> tuple[int, ...]:
    """Return the 256-entry CRC-32C lookup table.

    Returns
    -------
    table
        Table of 256 32-bit integers, equal to the literal table of
        ``sht_file.in.hpp`` lines 967-1000.

    Notes
    -----
    Generated by the loop which is commented out in
    ``sht_file.in.hpp`` lines 950-964, from :data:`CRC_POLYNOMIAL`.
    """
    table = []
    for i in range(256):
        value = i
        for _ in range(8):
            odd = value & 1
            value >>= 1
            if odd:
                value ^= CRC_POLYNOMIAL
        table.append(value)
    return tuple(table)


_CRC_TABLE: tuple[int, ...] = _build_crc_table()
"""CRC-32C lookup table, see :func:`_build_crc_table`."""

# The two tables below are copied verbatim from
# ``sht_file.in.hpp`` lines 1837-1869 and assume the standard settings
# of monoclinic unique axis b and orthorhombic axis choice abc, which
# is *not* the z-unique setting orix returns for space groups 3-15
_SPACE_GROUP_ROT: tuple[int, ...] = (
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 3, 3, 3, 3, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4,
    4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
)  # fmt: skip
"""Z rotational order of every space group, ``SpaceGroupRot``."""

_SPACE_GROUP_CMP: tuple[int, ...] = (
    0x0, 0x1, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x5, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x8, 0x8, 0x4,
    0x4, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
    0x7, 0x7, 0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x8, 0x4, 0x8, 0x4, 0x8, 0x8,
    0x5, 0x5, 0x9, 0x9, 0x9, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x3, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4, 0x4,
    0x4, 0x4, 0xA, 0xA, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7,
)  # fmt: skip
"""Compression flags of every space group, ``SpaceGroupCmp``."""


# ---------------------------- Checksum ------------------------------ #


def crc32c(data: bytes | bytearray | memoryview, crc: int = 0) -> int:
    """Return the SHTfile CRC-32C checksum of ``data``.

    Parameters
    ----------
    data
        Bytes to check.
    crc
        Previous checksum value, 0 by default.

    Returns
    -------
    checksum
        Unsigned 32-bit checksum.

    Notes
    -----
    Ported from ``sht_file.in.hpp`` lines 1002-1004.  The update loop
    is reflected while :data:`CRC_POLYNOMIAL` is the normal polynomial
    representation, so the check values differ from the standard
    CRC-32C ones: ``crc32c(b"123456789")`` is ``0xF28417BE`` here and
    ``0xE3069283`` in the standard.

    Chaining works as usual, i.e.
    ``crc32c(b, crc32c(a)) == crc32c(a + b)``.
    """
    raise NotImplementedError


def _pad8(n: int) -> int:
    """Return ``n`` rounded up to a multiple of eight.

    Parameters
    ----------
    n
        Unpadded byte length.

    Returns
    -------
    padded
        Padded byte length, ``n + (8 - n % 8) % 8``.

    Notes
    -----
    The padding rule of ``FileHeader::setDoi()``,
    ``sht_file.in.hpp`` lines 1182-1188.
    """
    raise NotImplementedError


# --------------------------- Data model ----------------------------- #


@dataclass
class ShtAtom:
    """One atom of a :class:`ShtCrystal`, 32 bytes on disk.

    Parameters
    ----------
    x, y, z
        Fractional coordinates in 24ths of the lattice parameters.
    occupancy
        Site occupancy.
    charge
        Ionic charge.
    debye_waller
        Debye-Waller factor in nm^2.
    res_fp
        Reserved floating point value.
    atomic_number
        Atomic number.
    res
        Three reserved bytes.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 260-317.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    occupancy: float = 1.0
    charge: float = 0.0
    debye_waller: float = 0.0
    res_fp: float = 0.0
    atomic_number: int = 0
    res: tuple[int, int, int] = (0, 0, 0)


@dataclass
class ShtCrystal:
    """One crystal of a :class:`ShtFile`, 72 bytes plus atoms and five
    padded strings on disk.

    Parameters
    ----------
    sg_num
        Space group number.
    sg_set
        Space group setting.
    sg_axis
        Space group axis choice.
    sg_cell
        Space group cell choice.
    origin
        Origin shift in 24ths of the lattice parameters.
    lat
        Lattice parameters ``(a, b, c, alpha, beta, gamma)`` with the
        lengths in nm and the angles in degrees.
    rot
        Quaternion ``(w, x, y, z)`` rotating the crystal.
    weight
        Phase fraction.
    num_atoms
        Number of atoms, which must equal ``len(atoms)``.
    atoms
        The atoms.
    formula, material_name, structure_symbol, references, note
        The five strings, decoded from their raw padded bytes with
        ``errors="replace"``.
    formula_len, material_name_len, structure_symbol_len, \
references_len, note_len
        Unpadded byte lengths of the five strings.  ``None`` means
        "encode the text and use its length", which is what instances
        built from scratch use.
    formula_bytes, material_name_bytes, structure_symbol_bytes, \
references_bytes, note_bytes
        Raw *padded* bytes of the five strings exactly as read, or
        ``None`` for instances built from scratch.  They are kept
        because the checksum covers the padding, so re-encoding the
        decoded text would silently change the byte stream of a file
        with non-UTF-8 text or non-zero padding.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 320-472.
    """

    sg_num: int = 1
    sg_set: int = 1
    sg_axis: int = 1
    sg_cell: int = 1
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lat: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        90.0,
        90.0,
        90.0,
    )
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    weight: float = 1.0
    num_atoms: int = 0
    atoms: list[ShtAtom] = field(default_factory=list)
    formula: str = ""
    material_name: str = ""
    structure_symbol: str = ""
    references: str = ""
    note: str = ""
    formula_len: int | None = None
    material_name_len: int | None = None
    structure_symbol_len: int | None = None
    references_len: int | None = None
    note_len: int | None = None
    formula_bytes: bytes | None = None
    material_name_bytes: bytes | None = None
    structure_symbol_bytes: bytes | None = None
    references_bytes: bytes | None = None
    note_bytes: bytes | None = None


@dataclass
class ShtEMsoftSimulation:
    """An ``EMsoftED`` simulation record, 88 bytes on disk.

    Parameters
    ----------
    emsoft_version
        Version string of the EMsoft build which made the master
        pattern, eight bytes on disk.
    sig_start, sig_end, sig_step
        Sample tilt range in degrees.
    omega
        Sample tilt about the y axis in degrees.
    kev
        Beam energy in keV.
    e_hist_min
        Lowest energy bin in keV.
    e_bin_size
        Energy bin width in keV.
    depth_max, depth_step
        Monte Carlo depth range in nm.
    thickness
        Foil thickness in nm.
    tot_num_el
        Number of incident electrons.
    num_sx
        Monte Carlo grid half width.
    res
        Two reserved bytes.
    c1, c2, c3
        Bethe parameters.
    sig_db_diff
        Bethe double diffraction parameter.
    d_min
        Smallest d spacing in nm.
    num_px
        Master pattern grid half width.
    lat_grid_type
        Latitude grid type, see :data:`LAT_GRID_NAMES`.
    res2
        Five reserved bytes.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 741-849.
    """

    emsoft_version: str = "unknown"
    sig_start: float = 0.0
    sig_end: float = math.nan
    sig_step: float = math.nan
    omega: float = 0.0
    kev: float = 0.0
    e_hist_min: float = 0.0
    e_bin_size: float = 0.0
    depth_max: float = 0.0
    depth_step: float = 0.0
    thickness: float = math.inf
    tot_num_el: int = 0
    num_sx: int = 0
    res: tuple[int, int] = (0, 0)
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    sig_db_diff: float = 0.0
    d_min: float = 0.0
    num_px: int = 0
    lat_grid_type: int = LAT_GRID_LAMBERT
    res2: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    emsoft_version_bytes: bytes | None = None


@dataclass
class ShtHeader:
    """The file header, 40 bytes plus two padded strings on disk.

    Parameters
    ----------
    magic
        Magic bytes, :data:`MAGIC_LE` or :data:`MAGIC_BE`.
    file_version
        Major and minor file version.
    res_bytes
        Two reserved bytes.
    software_version
        Eight byte tag of the writing software, NUL padded.
    modality
        Modality flag, see :data:`MODALITY_NAMES`.
    res_bytes2
        Three reserved bytes.
    beam_energy
        Beam energy in keV.
    primary_angle
        Primary angle in degrees, the sample tilt for EBSD.
    secondary_angle
        Secondary angle in degrees.
    reserved_param
        Reserved floating point parameter.
    doi, notes
        The two strings, decoded from their raw padded bytes with
        ``errors="replace"``.
    doi_len, note_len
        Unpadded byte lengths of the two strings.  ``None`` means
        "encode the text and use its length".
    doi_bytes, notes_bytes
        Raw *padded* bytes exactly as read, or ``None``, see
        :class:`ShtCrystal`.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 138-257.
    """

    magic: bytes = MAGIC_LE
    file_version: tuple[int, int] = VERSION
    res_bytes: tuple[int, int] = (0, 0)
    software_version: str = ""
    modality: int = MODALITY_UNKNOWN
    res_bytes2: tuple[int, int, int] = (0, 0, 0)
    beam_energy: float = 0.0
    primary_angle: float = 0.0
    secondary_angle: float = 0.0
    reserved_param: float = 0.0
    doi: str = ""
    notes: str = ""
    doi_len: int | None = None
    note_len: int | None = None
    doi_bytes: bytes | None = None
    notes_bytes: bytes | None = None
    software_version_bytes: bytes | None = None


@dataclass
class ShtHarmonics:
    """The harmonics block, 8 bytes plus the payload on disk.

    Parameters
    ----------
    bandwidth
        Bandwidth of the coefficients.
    z_rot
        Z rotational order the packing skips rows with.
    flags
        Compression flags, see :data:`FLAG_INVERSION` and friends.
    doub_cnt
        Number of doubles in ``packed``, which must equal
        :func:`num_harmonics`.
    packed
        The packed coefficients, of shape ``(doub_cnt,)`` and 64-bit
        floating point data type.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 542-637.
    """

    bandwidth: int = 0
    z_rot: int = 1
    flags: int = 0
    doub_cnt: int = 0
    packed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


@dataclass
class ShtFile:
    """A whole SHT file version 1.1.

    Parameters
    ----------
    header
        The file header.
    num_xtal
        Number of crystals, which must equal ``len(crystals)`` and
        ``len(simulations)``.
    sg_eff
        Effective space group in ``[1, 230]``, the one the harmonic
        symmetry follows.
    pijk
        Rotation convention sign, ``+1`` for EMsoft.
    rot_sense
        Rotation sense, ``112`` (``'p'``, passive) for EMsoft or
        ``97`` (``'a'``, active).
    modality
        Modality flag of the master pattern data block.
    vendor
        Vendor flag, see :data:`VENDOR_NAMES`.
    sim_meta_size
        Byte size of one simulation record, 0 for no records.
    crystals
        One :class:`ShtCrystal` per crystal.
    simulations
        One record per crystal, either a decoded
        :class:`ShtEMsoftSimulation`, the raw ``sim_meta_size`` bytes
        of a record this reader does not decode, or ``None`` when
        ``sim_meta_size`` is 0.
    harmonics
        The harmonics block.
    crc
        The checksum as read, or ``None`` for instances built from
        scratch.

    Notes
    -----
    Layout of ``sht_file.in.hpp`` lines 475-539, written in the order
    of lines 1575-1585.
    """

    header: ShtHeader = field(default_factory=ShtHeader)
    num_xtal: int = 1
    sg_eff: int = 1
    pijk: int = 1
    rot_sense: int = 112
    modality: int = MODALITY_UNKNOWN
    vendor: int = VENDOR_UNKNOWN
    sim_meta_size: int = 0
    crystals: list[ShtCrystal] = field(default_factory=list)
    simulations: list[ShtEMsoftSimulation | bytes | None] = field(default_factory=list)
    harmonics: ShtHarmonics = field(default_factory=ShtHarmonics)
    crc: int | None = None

    @property
    def block_offsets(self) -> dict[str, int]:
        """Return the byte offset of every block in the file.

        Returns
        -------
        offsets
            Offsets keyed on ``"header"``, ``"master_pattern"``,
            ``"crystal_<i>"``, ``"simulation_<i>"``, ``"harmonics"``,
            ``"payload"`` and ``"crc"``.  For the two in-package Ni
            files these are 0, 112, 120, 232, 320, 328 and 74824.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a lossless dictionary of this file.

        Returns
        -------
        dictionary
            Every field, including the packed payload as an
            :class:`numpy.ndarray` and every raw padded string byte
            string.  :meth:`from_dict` is its exact inverse, i.e.
            ``sht_file_to_bytes(ShtFile.from_dict(f.to_dict()))``
            equals ``sht_file_to_bytes(f)``.

        See Also
        --------
        metadata_dict
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> "ShtFile":
        """Return a file from a dictionary made by :meth:`to_dict`.

        Parameters
        ----------
        dictionary
            Dictionary from :meth:`to_dict`.

        Returns
        -------
        sht_file
            The file.
        """
        raise NotImplementedError

    def metadata_dict(self) -> dict[str, Any]:
        """Return a payload free dictionary of plain Python types.

        Returns
        -------
        dictionary
            Dictionary with the keys ``"header"``,
            ``"master_pattern"``, ``"crystals"``, ``"simulations"``
            and ``"harmonics"``.  The crystals and simulations are
            *numbered sub-nodes*, i.e. ``{"crystal_0": {...}}`` and
            ``{"simulation_0": {...}}``, and not lists, because
            HyperSpy's
            :class:`~hyperspy.misc.utils.DictionaryTreeBrowser`
            leaves lists of dictionaries as plain dictionaries.  The
            harmonics node has no ``"packed"`` key and no key ends in
            ``"_bytes"``.

        See Also
        --------
        to_dict
        """
        raise NotImplementedError


# ---------------------------- Packing ------------------------------- #


def num_harmonics(bandwidth: int, z_rot: int, flags: int) -> int:
    """Return the number of doubles a packed payload holds.

    Parameters
    ----------
    bandwidth
        Bandwidth of the coefficients.
    z_rot
        Z rotational order.
    flags
        Compression flags.

    Returns
    -------
    count
        Number of 64-bit floating point values.

    Raises
    ------
    ValueError
        If :data:`FLAG_MIRROR_Y` and :data:`FLAG_MIRROR_X` are both
        set.

    Notes
    -----
    Ported from ``HarmonicsData::NumHarm``, ``sht_file.in.hpp`` lines
    1672-1698.  Closed forms: ``num_harmonics(bw, 1, 0)`` is
    ``bw * (bw + 1)`` and ``num_harmonics(bw, 1, FLAG_MIRROR_Y)`` is
    ``bw * (bw + 1) // 2``.
    """
    raise NotImplementedError


def pack_harmonics(
    alm: np.ndarray, bandwidth: int, z_rot: int, flags: int
) -> np.ndarray:
    """Return the packed payload of harmonic coefficients.

    Parameters
    ----------
    alm
        Coefficients ``alm[m, l]`` of shape ``(bandwidth, bandwidth)``
        and 128-bit complex data type, with ``l < m`` entries zero.
    bandwidth
        Bandwidth of the coefficients.
    z_rot
        Z rotational order.
    flags
        Compression flags.

    Returns
    -------
    packed
        Payload of shape ``(num_harmonics(bandwidth, z_rot, flags),)``
        and 64-bit floating point data type.

    Raises
    ------
    ValueError
        If ``alm`` does not have shape
        ``(bandwidth, bandwidth)``, or if :data:`FLAG_MIRROR_Y` and
        :data:`FLAG_MIRROR_X` are both set.

    Notes
    -----
    Ported from ``HarmonicsData::PackHarm``, ``sht_file.in.hpp`` lines
    1700-1756.  Rows ``m`` with ``z_rot > 1 and m % z_rot != 0`` are
    skipped entirely, and within a row entries with odd ``l`` (when
    :data:`FLAG_INVERSION` is set) or odd ``l + m`` (when
    :data:`FLAG_MIRROR_Z` is set) are skipped.  A row is stored as
    interleaved real and imaginary parts, as real parts only when
    :data:`FLAG_MIRROR_Y` is set, and, when :data:`FLAG_MIRROR_X` is
    set, as real parts for ``m % (2 * z_rot) == 0`` and imaginary
    parts otherwise.  Everything the rules skip is dropped without
    checking, so callers who care about losslessness must check it
    themselves.
    """
    raise NotImplementedError


def unpack_harmonics(
    packed: np.ndarray, bandwidth: int, z_rot: int, flags: int
) -> np.ndarray:
    """Return the harmonic coefficients of a packed payload.

    Parameters
    ----------
    packed
        Payload of shape ``(num_harmonics(bandwidth, z_rot, flags),)``.
    bandwidth
        Bandwidth of the coefficients.
    z_rot
        Z rotational order.
    flags
        Compression flags.

    Returns
    -------
    alm
        Coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type, with
        the ``l < m`` entries and every skipped entry zero.

    Raises
    ------
    ValueError
        If ``packed`` does not have the length
        :func:`num_harmonics` gives, or if :data:`FLAG_MIRROR_Y` and
        :data:`FLAG_MIRROR_X` are both set.

    Notes
    -----
    Ported from ``HarmonicsData::UnpackHarm``, ``sht_file.in.hpp``
    lines 1758-1831, the exact inverse of :func:`pack_harmonics` for
    coefficients which respect the packing rules.
    """
    raise NotImplementedError


# --------------------- Space group lookup tables -------------------- #


def space_group_z_rotation(space_group: int) -> int:
    """Return the z rotational order of a space group.

    Parameters
    ----------
    space_group
        Space group number in ``[1, 230]``.

    Returns
    -------
    z_rot
        One of 1, 2, 3, 4 and 6.

    Raises
    ------
    ValueError
        If ``space_group`` is outside ``[1, 230]``.

    Notes
    -----
    The ``HarmonicsData::SpaceGroupRot`` table of
    ``sht_file.in.hpp`` lines 1837-1849, which assumes the *standard*
    settings of monoclinic unique axis b and orthorhombic axis choice
    abc.  orix instead returns z-unique point groups for space groups
    3-15, so the two disagree there; this table is authoritative for
    the file format and is never derived from the point group.
    """
    raise NotImplementedError


def space_group_compression_flags(space_group: int) -> int:
    """Return the compression flags of a space group.

    Parameters
    ----------
    space_group
        Space group number in ``[1, 230]``.

    Returns
    -------
    flags
        Bit mask of :data:`FLAG_INVERSION`, :data:`FLAG_MIRROR_Z`,
        :data:`FLAG_MIRROR_Y` and :data:`FLAG_MIRROR_X`.

    Raises
    ------
    ValueError
        If ``space_group`` is outside ``[1, 230]``.

    Notes
    -----
    The ``HarmonicsData::SpaceGroupCmp`` table of
    ``sht_file.in.hpp`` lines 1851-1869; see
    :func:`space_group_z_rotation` for the setting caveat.
    """
    raise NotImplementedError


# ------------------------- Reading, writing ------------------------- #


def read_sht(
    source: str | os.PathLike | bytes | bytearray, check_crc: bool = True
) -> ShtFile:
    """Return the contents of an SHT file version 1.1.

    Parameters
    ----------
    source
        Path to the file or its bytes.
    check_crc
        Whether to verify the checksum, ``True`` by default.

    Returns
    -------
    sht_file
        The parsed file.

    Raises
    ------
    ValueError
        If the magic bytes are neither :data:`MAGIC_LE` nor
        :data:`MAGIC_BE`, if the file is truncated, if the
        compression flags are illegal, or if ``check_crc`` is
        ``True`` and the checksum does not match.
    NotImplementedError
        If the file is big-endian (:data:`MAGIC_BE`) or its version
        is not :data:`VERSION`.

    Notes
    -----
    Ported from ``File::read``, ``sht_file.in.hpp`` lines 2010-2031,
    but more permissive in two ways: simulation records are kept as
    raw bytes and only decoded into :class:`ShtEMsoftSimulation` when
    the vendor is EMsoft, the modality is EBSD, ECP or TKD and
    ``sim_meta_size`` is 88 (EMSphInx refuses any other non-zero
    pair, line 1605), and files with more than one crystal or a
    non-EBSD modality are parsed.  Both keep the checksum verifiable
    and a rewrite byte identical.

    EMSphInx does not sanity check on read, only on write, and
    neither do we: the reserved bytes are read as they are.
    """
    raise NotImplementedError


def sht_file_to_bytes(sht_file: ShtFile) -> bytes:
    """Return the bytes of an SHT file version 1.1.

    Parameters
    ----------
    sht_file
        The file to serialize.

    Returns
    -------
    data
        The whole file including the trailing checksum.

    Raises
    ------
    ValueError
        If any check of :func:`_sanity_check` fails.

    Notes
    -----
    Strings are written from their raw padded bytes whenever those
    are present, and are encoded as UTF-8 and NUL padded to a
    multiple of eight otherwise.  Reading a file and serializing it
    again is therefore byte identical for any version 1.1 file, not
    only for ASCII ones with NUL padding.
    """
    raise NotImplementedError


def write_sht(filename: str | os.PathLike, sht_file: ShtFile) -> None:
    """Write an SHT file version 1.1.

    Parameters
    ----------
    filename
        Path to write to, opened in binary mode.
    sht_file
        The file to write.

    Raises
    ------
    ValueError
        If any check of :func:`_sanity_check` fails.

    Notes
    -----
    Ported from ``File::write``, ``sht_file.in.hpp`` lines 1995-2005.
    """
    raise NotImplementedError


def _sanity_check(sht_file: ShtFile) -> None:
    """Raise if a file is not writeable.

    Parameters
    ----------
    sht_file
        The file to check.

    Raises
    ------
    ValueError
        With the wording of the failing EMSphInx check.

    Notes
    -----
    Runs every check ``File::sanityCheck`` runs
    (``sht_file.in.hpp`` lines 1970-1980), i.e. those of
    ``FileHeader::sanityCheck`` (1066-1094),
    ``MasterPatternData::sanityCheck`` (1523-1554) and
    ``HarmonicsData::sanityCheck`` (1625-1628), plus the file level
    modality cross check.  Their wording is quoted verbatim,
    including the typo in "noites string doesn't match length", so
    that a grep of the EMSphInx sources finds the origin of a
    message.

    Two checks are ours, and so is their wording: ``num_xtal >= 1``
    ("at least one crystal is required"; EMSphInx dereferences the
    first simulation record without checking) and
    ``1 <= bandwidth <= 32767`` ("bandwidth must be in [1, 32767]";
    the field is a signed 16-bit integer).

    ``AtomData::sanityCheck`` (line 1209) is deliberately not ported:
    it tests ``z() > 1.0f`` where it means ``occ()``, so it would
    reject any atom with a z coordinate above 1/24, and nothing
    reachable from ``File::write`` calls it.
    """
    raise NotImplementedError


def _read_struct(data: memoryview, offset: int, fmt: str) -> tuple:
    """Return the values of a little-endian struct in ``data``.

    Parameters
    ----------
    data
        Bytes to read from.
    offset
        Byte offset to read at.
    fmt
        Struct format string without the byte order character.

    Returns
    -------
    values
        The unpacked values.

    Raises
    ------
    ValueError
        If the file ends before the struct does.
    """
    raise NotImplementedError


def _write_struct(fmt: str, *values: Any) -> bytes:
    """Return the bytes of a little-endian struct.

    Parameters
    ----------
    fmt
        Struct format string without the byte order character.
    *values
        Values to pack.

    Returns
    -------
    data
        The packed bytes.
    """
    raise NotImplementedError


def _decode_string(data: bytes, length: int) -> str:
    """Return the text of a raw padded string field.

    Parameters
    ----------
    data
        Raw padded bytes.
    length
        Unpadded byte length.

    Returns
    -------
    text
        ``data[:length]`` decoded as UTF-8 with
        ``errors="replace"``.
    """
    raise NotImplementedError


def _encode_string(text: str, raw: bytes | None, length: int | None) -> bytes:
    """Return the raw padded bytes of a string field.

    Parameters
    ----------
    text
        The text.
    raw
        Raw padded bytes as read, or ``None``.
    length
        Unpadded byte length, or ``None``.

    Returns
    -------
    data
        ``raw`` when it is given, else ``text`` encoded as UTF-8 and
        NUL padded to a multiple of eight.
    """
    raise NotImplementedError


def _string_length(text: str, raw: bytes | None, length: int | None) -> int:
    """Return the unpadded byte length of a string field.

    Parameters
    ----------
    text
        The text.
    raw
        Raw padded bytes as read, or ``None``.
    length
        Unpadded byte length, or ``None``.

    Returns
    -------
    unpadded
        ``length`` when it is given, else ``len(text.encode())``.
    """
    raise NotImplementedError


def _path_or_bytes(source: str | os.PathLike | bytes | bytearray) -> bytes:
    """Return the bytes of a path or bytes source.

    Parameters
    ----------
    source
        Path to a file or its bytes.

    Returns
    -------
    data
        The bytes.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()
