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

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import struct
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

HEADER_SIZE: int = 40
"""Size in bytes of the fixed length part of the file header."""

MASTER_PATTERN_DATA_SIZE: int = 8
"""Size in bytes of the fixed length master pattern data block."""

CRYSTAL_SIZE: int = 72
"""Size in bytes of the fixed length part of a crystal record."""

ATOM_SIZE: int = 32
"""Size in bytes of an atom record."""

HARMONICS_SIZE: int = 8
"""Size in bytes of the fixed length part of the harmonics block."""

_EMSOFT_ED_MODALITIES: frozenset = frozenset(
    {MODALITY_EBSD, MODALITY_ECP, MODALITY_TKD}
)
"""Modalities an ``EMsoftED`` record is valid for,
``EMsoftED::forModality``, ``sht_file.in.hpp`` lines 1879-1886."""


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

    Examples
    --------
    >>> from kikuchipy.indexing._spherical._sht_file import crc32c
    >>> hex(crc32c(b"123456789"))
    '0xf28417be'
    """
    table = _CRC_TABLE
    value = ~crc & 0xFFFFFFFF
    for byte in bytes(data):
        value = (value >> 8) ^ table[(value & 0xFF) ^ byte]
    return ~value & 0xFFFFFFFF


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
    n = int(n)
    remainder = n % 8
    if remainder == 0:
        return n
    return n + 8 - remainder


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
    emsoft_version_bytes
        Raw eight bytes of ``emsoft_version`` exactly as read, or
        ``None`` for instances built from scratch.

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
    software_version_bytes
        Raw eight bytes of ``software_version`` exactly as read, or
        ``None`` for instances built from scratch.

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
        header = self.header
        offset = HEADER_SIZE
        offset += len(_encode_string(header.doi, header.doi_bytes, header.doi_len))
        offset += len(_encode_string(header.notes, header.notes_bytes, header.note_len))
        offsets = {"header": 0, "master_pattern": offset}
        offset += MASTER_PATTERN_DATA_SIZE
        for i, crystal in enumerate(self.crystals):
            offsets[f"crystal_{i}"] = offset
            offset += _crystal_size(crystal)
        for i in range(len(self.simulations)):
            offsets[f"simulation_{i}"] = offset
            offset += self.sim_meta_size
        offsets["harmonics"] = offset
        offset += HARMONICS_SIZE
        offsets["payload"] = offset
        offset += 8 * len(self.harmonics.packed)
        offsets["crc"] = offset
        return offsets

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
        harmonics = self.harmonics
        return {
            "header": asdict(self.header),
            "num_xtal": self.num_xtal,
            "sg_eff": self.sg_eff,
            "pijk": self.pijk,
            "rot_sense": self.rot_sense,
            "modality": self.modality,
            "vendor": self.vendor,
            "sim_meta_size": self.sim_meta_size,
            "crystals": [asdict(crystal) for crystal in self.crystals],
            "simulations": [
                asdict(record) if isinstance(record, ShtEMsoftSimulation) else record
                for record in self.simulations
            ],
            "harmonics": {
                "bandwidth": harmonics.bandwidth,
                "z_rot": harmonics.z_rot,
                "flags": harmonics.flags,
                "doub_cnt": harmonics.doub_cnt,
                "packed": np.asarray(harmonics.packed).copy(),
            },
            "crc": self.crc,
        }

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
        crystals = []
        for node in dictionary["crystals"]:
            node = dict(node)
            node["atoms"] = [ShtAtom(**atom) for atom in node.get("atoms", [])]
            crystals.append(ShtCrystal(**node))
        simulations = [
            ShtEMsoftSimulation(**record) if isinstance(record, dict) else record
            for record in dictionary["simulations"]
        ]
        harmonics_node = dict(dictionary["harmonics"])
        harmonics_node["packed"] = np.asarray(
            harmonics_node["packed"], dtype=np.float64
        )
        return cls(
            header=ShtHeader(**dictionary["header"]),
            num_xtal=dictionary["num_xtal"],
            sg_eff=dictionary["sg_eff"],
            pijk=dictionary["pijk"],
            rot_sense=dictionary["rot_sense"],
            modality=dictionary["modality"],
            vendor=dictionary["vendor"],
            sim_meta_size=dictionary["sim_meta_size"],
            crystals=crystals,
            simulations=simulations,
            harmonics=ShtHarmonics(**harmonics_node),
            crc=dictionary["crc"],
        )

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
            leaves lists of dictionaries as plain dictionaries; the
            atoms of a crystal are numbered the same way,
            ``{"atom_0": {...}}``.  The harmonics node has no
            ``"packed"`` key and no key ends in ``"_bytes"``: the two
            reserved byte fields of the header are ``"reserved"`` and
            ``"reserved2"`` here, the raw padded string bytes are
            left out, and so are the unpadded string lengths, which
            are file plumbing.  Every floating point NaN is the
            :data:`math.nan` singleton, so that the dictionaries of
            two parses of one file compare equal: Python's ``==``
            short circuits on identity, which is the only way a
            container holding a NaN can equal another.

        See Also
        --------
        to_dict
        """
        header = self.header
        header_node = {
            "file_version": tuple(header.file_version),
            "software_version": header.software_version,
            "modality": header.modality,
            "reserved": tuple(header.res_bytes),
            "reserved2": tuple(header.res_bytes2),
            "beam_energy": header.beam_energy,
            "primary_angle": header.primary_angle,
            "secondary_angle": header.secondary_angle,
            "reserved_param": header.reserved_param,
            "doi": header.doi,
            "notes": header.notes,
        }
        crystals = {}
        for i, crystal in enumerate(self.crystals):
            node = {
                key: value
                for key, value in asdict(crystal).items()
                if not key.endswith("_bytes") and not key.endswith("_len")
            }
            node["atoms"] = {f"atom_{j}": a for j, a in enumerate(node["atoms"])}
            crystals[f"crystal_{i}"] = node
        simulations = {}
        for i, record in enumerate(self.simulations):
            if isinstance(record, ShtEMsoftSimulation):
                node = {
                    key: value
                    for key, value in asdict(record).items()
                    if not key.endswith("_bytes")
                }
            else:
                node = record
            simulations[f"simulation_{i}"] = node
        harmonics = self.harmonics
        return _singleton_nan(
            {
                "header": header_node,
                "master_pattern": {
                    "num_xtal": self.num_xtal,
                    "sg_eff": self.sg_eff,
                    "pijk": self.pijk,
                    "rot_sense": self.rot_sense,
                    "modality": self.modality,
                    "vendor": self.vendor,
                    "sim_meta_size": self.sim_meta_size,
                },
                "crystals": crystals,
                "simulations": simulations,
                "harmonics": {
                    "bandwidth": harmonics.bandwidth,
                    "z_rot": harmonics.z_rot,
                    "flags": harmonics.flags,
                    "doub_cnt": harmonics.doub_cnt,
                },
            }
        )


def _singleton_nan(value: Any) -> Any:
    """Return a value with every floating point NaN replaced by the
    :data:`math.nan` singleton.

    Parameters
    ----------
    value
        A dictionary, list, tuple or scalar.

    Returns
    -------
    value
        The same structure, with NaNs replaced.  Two containers can
        only compare equal when they hold the *same* NaN object,
        since ``float("nan") != float("nan")``.
    """
    if isinstance(value, dict):
        return {key: _singleton_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_singleton_nan(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_singleton_nan(item) for item in value)
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value


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

    Examples
    --------
    >>> from kikuchipy.indexing._spherical._sht_file import num_harmonics
    >>> num_harmonics(384, 4, 0x7)
    9312
    """
    inversion, mirror_z, mirror_y, mirror_x = _flags_to_bools(flags)
    bandwidth = int(bandwidth)
    if bandwidth < 1:
        return 0
    orders = np.arange(bandwidth).reshape(-1, 1)
    degrees = np.arange(bandwidth).reshape(1, -1)
    keep = degrees >= orders
    if z_rot > 1:
        keep &= orders % z_rot == 0
    if inversion:
        keep &= degrees % 2 == 0
    if mirror_z:
        keep &= (degrees + orders) % 2 == 0
    count = int(np.count_nonzero(keep))
    if not (mirror_x or mirror_y):
        count *= 2
    return count


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
    inversion, mirror_z, mirror_y, mirror_x = _flags_to_bools(flags)
    bandwidth = int(bandwidth)
    alm = np.asarray(alm)
    if alm.shape != (bandwidth, bandwidth):
        raise ValueError(
            f"alm must have shape ({bandwidth}, {bandwidth}), not {alm.shape}"
        )
    parts = []
    for order in range(bandwidth):
        if z_rot > 1 and order % z_rot != 0:
            continue
        keep = _row_mask(order, bandwidth, inversion, mirror_z)
        row = alm[order][keep]
        kind = _row_kind(order, z_rot, mirror_y, mirror_x)
        if kind == 1:
            parts.append(np.real(row).astype(np.float64))
        elif kind == 2:
            parts.append(np.imag(row).astype(np.float64))
        else:
            interleaved = np.empty(2 * row.size, dtype=np.float64)
            interleaved[0::2] = np.real(row)
            interleaved[1::2] = np.imag(row)
            parts.append(interleaved)
    if len(parts) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


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
    inversion, mirror_z, mirror_y, mirror_x = _flags_to_bools(flags)
    bandwidth = int(bandwidth)
    packed = np.asarray(packed, dtype=np.float64).ravel()
    count = num_harmonics(bandwidth, z_rot, flags)
    if packed.size != count:
        raise ValueError(
            f"packed must hold {count} values for bandwidth {bandwidth}, z "
            f"rotational order {z_rot} and flags {flags:#x}, not {packed.size}"
        )
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    start = 0
    for order in range(bandwidth):
        if z_rot > 1 and order % z_rot != 0:
            continue
        keep = _row_mask(order, bandwidth, inversion, mirror_z)
        n_kept = int(np.count_nonzero(keep))
        kind = _row_kind(order, z_rot, mirror_y, mirror_x)
        if kind == 1:
            alm[order, keep] = packed[start : start + n_kept]
            start += n_kept
        elif kind == 2:
            alm[order, keep] = 1j * packed[start : start + n_kept]
            start += n_kept
        else:
            values = packed[start : start + 2 * n_kept]
            alm[order, keep] = values[0::2] + 1j * values[1::2]
            start += 2 * n_kept
    return alm


def _flags_to_bools(flags: int) -> tuple[bool, bool, bool, bool]:
    """Return the four compression flags as booleans.

    Parameters
    ----------
    flags
        Compression flags.

    Returns
    -------
    inversion, mirror_z, mirror_y, mirror_x
        Whether each flag is set.

    Raises
    ------
    ValueError
        If :data:`FLAG_MIRROR_Y` and :data:`FLAG_MIRROR_X` are both
        set (``sht_file.in.hpp`` line 1678).
    """
    flags = int(flags)
    mirror_y = bool(flags & FLAG_MIRROR_Y)
    mirror_x = bool(flags & FLAG_MIRROR_X)
    if mirror_x and mirror_y:
        raise ValueError("compression flags 0x04 and 0x08 are mutually exclusive")
    inversion = bool(flags & FLAG_INVERSION)
    mirror_z = bool(flags & FLAG_MIRROR_Z)
    return inversion, mirror_z, mirror_y, mirror_x


def _row_mask(
    order: int, bandwidth: int, inversion: bool, mirror_z: bool
) -> np.ndarray:
    """Return which degrees of one order the packing keeps.

    Parameters
    ----------
    order
        Order ``m`` of the row.
    bandwidth
        Bandwidth of the coefficients.
    inversion
        Whether odd degrees are systematic zeros.
    mirror_z
        Whether odd ``l + m`` entries are systematic zeros.

    Returns
    -------
    mask
        Boolean mask of shape ``(bandwidth,)``.
    """
    degrees = np.arange(bandwidth)
    mask = degrees >= order
    if inversion:
        mask &= degrees % 2 == 0
    if mirror_z:
        mask &= (degrees + order) % 2 == 0
    return mask


def _row_kind(order: int, z_rot: int, mirror_y: bool, mirror_x: bool) -> int:
    """Return the storage type of one row.

    Parameters
    ----------
    order
        Order ``m`` of the row.
    z_rot
        Z rotational order.
    mirror_y
        Whether :data:`FLAG_MIRROR_Y` is set.
    mirror_x
        Whether :data:`FLAG_MIRROR_X` is set.

    Returns
    -------
    kind
        0 for complex, 1 for strictly real and 2 for strictly
        imaginary, as in ``sht_file.in.hpp`` lines 1721-1732.
    """
    if mirror_y:
        return 1
    if mirror_x:
        return 1 if order % (z_rot * 2) == 0 else 2
    return 0


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
    _check_space_group(space_group)
    return _SPACE_GROUP_ROT[space_group - 1]


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
    _check_space_group(space_group)
    return _SPACE_GROUP_CMP[space_group - 1]


def _check_space_group(space_group: int) -> None:
    """Raise if a space group number is outside ``[1, 230]``.

    Parameters
    ----------
    space_group
        Space group number.

    Raises
    ------
    ValueError
        If ``space_group`` is outside ``[1, 230]``.
    """
    if not 1 <= space_group <= 230:
        raise ValueError(f"Space group {space_group} must be in the range [1, 230]")


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
        compression flags are illegal, if the z rotational order is
        below one, or if ``check_crc`` is ``True`` and the checksum
        does not match.
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
    neither do we: the reserved bytes are read as they are.  The one
    exception is the z rotational order, which :func:`_row_kind`
    divides by: a file claiming zero makes the C++ ``PackHarm`` exit
    on an integer division by zero, and a negative order would type
    rows differently here than there, Python's ``%`` being floored
    where C++' ``size_t`` modulo is not.  Both are rejected.
    """
    data = _path_or_bytes(source)
    view = memoryview(data)

    magic = _read_bytes(view, 0, 4)
    if magic == MAGIC_BE:
        raise NotImplementedError(
            "big-endian SHT files, with the magic bytes b'*SHT', are not supported"
        )
    if magic != MAGIC_LE:
        raise ValueError(f"the magic bytes {magic!r} are not an SHT file's")
    file_version = _read_struct(view, 4, "2b")
    if tuple(file_version) != VERSION:
        raise NotImplementedError(
            "only SHT file version 1.1 is supported, not version "
            f"{file_version[0]}.{file_version[1]}"
        )

    res_bytes = _read_struct(view, 6, "2b")
    software_version_bytes = _read_bytes(view, 8, 8)
    (modality,) = _read_struct(view, 16, "b")
    res_bytes2 = _read_struct(view, 17, "3b")
    beam_energy, primary_angle, secondary_angle, reserved_param = _read_struct(
        view, 20, "4f"
    )
    doi_len, note_len = _read_struct(view, 36, "2h")
    if doi_len < 0 or note_len < 0:
        raise ValueError(
            f"the file header has negative string lengths {doi_len} and {note_len}"
        )
    offset = HEADER_SIZE
    doi_bytes = _read_bytes(view, offset, _pad8(doi_len))
    offset += len(doi_bytes)
    notes_bytes = _read_bytes(view, offset, _pad8(note_len))
    offset += len(notes_bytes)
    header = ShtHeader(
        magic=magic,
        file_version=tuple(file_version),
        res_bytes=tuple(res_bytes),
        software_version=_decode_string(software_version_bytes, 8).rstrip("\x00"),
        modality=modality,
        res_bytes2=tuple(res_bytes2),
        beam_energy=beam_energy,
        primary_angle=primary_angle,
        secondary_angle=secondary_angle,
        reserved_param=reserved_param,
        doi=_decode_string(doi_bytes, doi_len),
        notes=_decode_string(notes_bytes, note_len),
        doi_len=doi_len,
        note_len=note_len,
        doi_bytes=doi_bytes,
        notes_bytes=notes_bytes,
        software_version_bytes=software_version_bytes,
    )

    (
        num_xtal,
        sg_eff,
        pijk,
        rot_sense,
        mp_modality,
        vendor,
        sim_meta_size,
    ) = _read_struct(view, offset, "bBbbbbh")
    offset += MASTER_PATTERN_DATA_SIZE
    if num_xtal < 0:
        raise ValueError(f"the number of crystals {num_xtal} is negative")

    crystals = []
    for _ in range(num_xtal):
        crystal, offset = _read_crystal(view, offset)
        crystals.append(crystal)

    simulations: list = []
    if sim_meta_size == 0:
        simulations = [None] * num_xtal
    else:
        if sim_meta_size < 0:
            raise ValueError(f"the simulation record size {sim_meta_size} is negative")
        decode = (
            vendor == VENDOR_EMSOFT
            and mp_modality in _EMSOFT_ED_MODALITIES
            and sim_meta_size == EMSOFT_ED_SIZE
        )
        for _ in range(num_xtal):
            record = _read_bytes(view, offset, sim_meta_size)
            offset += sim_meta_size
            simulations.append(_read_emsoft_simulation(record) if decode else record)

    bandwidth, z_rot, flags, doub_cnt = _read_struct(view, offset, "hbbi")
    offset += HARMONICS_SIZE
    _flags_to_bools(flags)
    # The order is a signed byte on disk, and _row_kind divides by it
    if z_rot < 1:
        raise ValueError(
            f"the z rotational order {z_rot} of the harmonics block must be "
            "at least one"
        )
    if doub_cnt < 0:
        raise ValueError(f"the harmonics count {doub_cnt} is negative")
    payload = _read_bytes(view, offset, 8 * doub_cnt)
    offset += len(payload)
    harmonics = ShtHarmonics(
        bandwidth=bandwidth,
        z_rot=z_rot,
        flags=flags,
        doub_cnt=doub_cnt,
        packed=np.frombuffer(payload, dtype="<f8").astype(np.float64),
    )

    (crc,) = _read_struct(view, offset, "I")
    if check_crc:
        computed = crc32c(view[:offset])
        if computed != crc:
            raise ValueError(
                "the SHT file is corrupted (incorrect checksum: the file says "
                f"{crc:#010x} while its contents give {computed:#010x})"
            )

    return ShtFile(
        header=header,
        num_xtal=num_xtal,
        sg_eff=sg_eff,
        pijk=pijk,
        rot_sense=rot_sense,
        modality=mp_modality,
        vendor=vendor,
        sim_meta_size=sim_meta_size,
        crystals=crystals,
        simulations=simulations,
        harmonics=harmonics,
        crc=crc,
    )


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
    _sanity_check(sht_file)

    header = sht_file.header
    data = bytearray()
    data += bytes(header.magic)
    data += _write_struct("2b", *header.file_version)
    data += _write_struct("2b", *header.res_bytes)
    data += _encode_fixed_string(
        header.software_version, header.software_version_bytes, 8
    )
    data += _write_struct("b", header.modality)
    data += _write_struct("3b", *header.res_bytes2)
    data += _write_struct(
        "4f",
        header.beam_energy,
        header.primary_angle,
        header.secondary_angle,
        header.reserved_param,
    )
    data += _write_struct(
        "2h",
        _string_length(header.doi, header.doi_bytes, header.doi_len),
        _string_length(header.notes, header.notes_bytes, header.note_len),
    )
    data += _encode_string(header.doi, header.doi_bytes, header.doi_len)
    data += _encode_string(header.notes, header.notes_bytes, header.note_len)

    data += _write_struct(
        "bBbbbbh",
        sht_file.num_xtal,
        sht_file.sg_eff,
        sht_file.pijk,
        sht_file.rot_sense,
        sht_file.modality,
        sht_file.vendor,
        sht_file.sim_meta_size,
    )
    for crystal in sht_file.crystals:
        data += _crystal_to_bytes(crystal)
    if sht_file.sim_meta_size != 0:
        for record in sht_file.simulations:
            data += _simulation_to_bytes(record)

    harmonics = sht_file.harmonics
    data += _write_struct(
        "hbbi",
        harmonics.bandwidth,
        harmonics.z_rot,
        harmonics.flags,
        harmonics.doub_cnt,
    )
    data += np.ascontiguousarray(harmonics.packed, dtype="<f8").tobytes()

    data += _write_struct("I", crc32c(bytes(data)))
    return bytes(data)


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
    data = sht_file_to_bytes(sht_file)
    with open(filename, "wb") as file:
        file.write(data)


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
    ``MasterPatternData::sanityCheck`` (1523-1554),
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
    reachable from ``File::write`` calls it.  Neither
    ``CrystalData::sanityCheck`` nor ``AtomData::sanityCheck`` is
    reached from ``MasterPatternData::sanityCheck``, which does not
    recurse into the crystals.
    """
    # FileHeader::sanityCheck, sht_file.in.hpp lines 1066-1094
    header = sht_file.header
    if tuple(header.file_version) != VERSION:
        raise ValueError("unsupported file version")
    if header.res_bytes[0] != 0 or header.res_bytes[1] != 0:
        raise ValueError("non-zero reserved bytes")
    if header.modality not in MODALITY_NAMES:
        raise ValueError("invalid modality flag")
    if any(value != 0 for value in header.res_bytes2):
        raise ValueError("reserved bytes must be 0")
    doi_len = _string_length(header.doi, header.doi_bytes, header.doi_len)
    if len(_encode_string(header.doi, header.doi_bytes, header.doi_len)) != _pad8(
        doi_len
    ):
        raise ValueError("doi string doesn't match length")
    note_len = _string_length(header.notes, header.notes_bytes, header.note_len)
    if len(_encode_string(header.notes, header.notes_bytes, header.note_len)) != _pad8(
        note_len
    ):
        raise ValueError("noites string doesn't match length")
    if header.beam_energy < 0:
        raise ValueError("negative beam energy is non-physical")
    if header.beam_energy > 10000:
        raise ValueError("10 MeV beam energy is unrealistic")
    if header.primary_angle < -360 or header.primary_angle > 360:
        raise ValueError("primary angle outside [-360,360]")
    if header.secondary_angle < -360 or header.secondary_angle > 360:
        raise ValueError("secondary angle outside [-360,360]")

    # MasterPatternData::sanityCheck, lines 1523-1554
    if sht_file.sg_eff < 1 or sht_file.sg_eff > 230:
        raise ValueError("invalid effective space group number")
    if sht_file.num_xtal != len(sht_file.crystals):
        raise ValueError("# crystals != crystals size")
    if sht_file.num_xtal != len(sht_file.simulations):
        raise ValueError("# crystals != simulation metadata size")
    if sht_file.pijk not in (1, -1):
        raise ValueError("pijk must be +/-1")
    if sht_file.rot_sense not in (97, 112):
        raise ValueError("rotation sense must be 'a' or 'p'")
    if sht_file.modality not in MODALITY_NAMES:
        raise ValueError("invalid modality flag")
    if sht_file.vendor not in VENDOR_NAMES:
        raise ValueError("invalid vendor flag")
    if sht_file.sim_meta_size == 0:
        for record in sht_file.simulations:
            if record is not None:
                raise ValueError("non-NULL simulation data for 0 size")
    else:
        for record in sht_file.simulations:
            if record is None:
                raise ValueError("NULL simulation data for nonzero size")
            if _record_size(record) != sht_file.sim_meta_size:
                raise ValueError("simulation data size doesn't match header size")
            if not _record_for_modality(record, sht_file.modality):
                raise ValueError(
                    "simulation data modality not valid for master pattern modality"
                )
            vendor = _record_vendor(record)
            if vendor is not None and vendor != sht_file.vendor:
                raise ValueError(
                    "simulation data vendor doesn't match master pattern verndor"
                )

    # HarmonicsData::sanityCheck, lines 1625-1628.  Our bandwidth
    # check comes first so that num_harmonics is never called with a
    # bandwidth the signed 16 bit field cannot hold
    harmonics = sht_file.harmonics
    if harmonics.bandwidth < 1 or harmonics.bandwidth > MAX_BANDWIDTH:
        raise ValueError(f"bandwidth must be in [1, {MAX_BANDWIDTH}]")
    if harmonics.doub_cnt != num_harmonics(
        harmonics.bandwidth, harmonics.z_rot, harmonics.flags
    ):
        raise ValueError("harmonics count doesn't match compression parameters")
    if harmonics.doub_cnt != len(harmonics.packed):
        raise ValueError("harmonics count doesn't match size")

    # File::sanityCheck, lines 1976-1979, which dereferences
    # simul.front() without checking that there is one
    if sht_file.num_xtal < 1:
        raise ValueError("at least one crystal is required")
    first = sht_file.simulations[0]
    if first is not None and not _record_for_modality(first, header.modality):
        raise ValueError("file modality doesn't match simulation modality")


def _record_size(record: ShtEMsoftSimulation | bytes) -> int:
    """Return the byte size of a simulation record.

    Parameters
    ----------
    record
        A decoded record or the raw bytes of an opaque one.

    Returns
    -------
    size
        Size in bytes.
    """
    if isinstance(record, ShtEMsoftSimulation):
        return EMSOFT_ED_SIZE
    return len(record)


def _record_for_modality(record: ShtEMsoftSimulation | bytes, modality: int) -> bool:
    """Return whether a simulation record is valid for a modality.

    Parameters
    ----------
    record
        A decoded record or the raw bytes of an opaque one.
    modality
        Modality flag to check against.

    Returns
    -------
    valid
        Whether the record type supports the modality.  An opaque
        record is always accepted, since its type and therefore its
        modality support is unknown to us and only its length can be
        checked.

    Notes
    -----
    ``EMsoftED::forModality``, ``sht_file.in.hpp`` lines 1879-1886.
    """
    if isinstance(record, ShtEMsoftSimulation):
        return modality in _EMSOFT_ED_MODALITIES
    return True


def _record_vendor(record: ShtEMsoftSimulation | bytes) -> int | None:
    """Return the vendor of a simulation record.

    Parameters
    ----------
    record
        A decoded record or the raw bytes of an opaque one.

    Returns
    -------
    vendor
        The vendor flag, or ``None`` for an opaque record whose type
        and therefore vendor is unknown.
    """
    if isinstance(record, ShtEMsoftSimulation):
        return VENDOR_EMSOFT
    return None


def _crystal_size(crystal: ShtCrystal) -> int:
    """Return the byte size of a crystal record.

    Parameters
    ----------
    crystal
        The crystal.

    Returns
    -------
    size
        Size in bytes: the fixed 72, plus 32 per atom, plus the five
        padded strings.
    """
    size = CRYSTAL_SIZE + ATOM_SIZE * len(crystal.atoms)
    for text, raw, length in _crystal_strings(crystal):
        size += len(_encode_string(text, raw, length))
    return size


def _crystal_strings(crystal: ShtCrystal) -> tuple:
    """Return the five strings of a crystal in file order.

    Parameters
    ----------
    crystal
        The crystal.

    Returns
    -------
    strings
        One ``(text, raw padded bytes, unpadded length)`` triplet per
        string, in the order formula, material name, structure
        symbol, references and note.
    """
    return (
        (crystal.formula, crystal.formula_bytes, crystal.formula_len),
        (
            crystal.material_name,
            crystal.material_name_bytes,
            crystal.material_name_len,
        ),
        (
            crystal.structure_symbol,
            crystal.structure_symbol_bytes,
            crystal.structure_symbol_len,
        ),
        (crystal.references, crystal.references_bytes, crystal.references_len),
        (crystal.note, crystal.note_bytes, crystal.note_len),
    )


def _read_crystal(data: memoryview, offset: int) -> tuple:
    """Return one crystal record and the offset just after it.

    Parameters
    ----------
    data
        Bytes of the whole file.
    offset
        Byte offset the record starts at.

    Returns
    -------
    crystal
        The crystal.
    end
        Byte offset just after the record.

    Raises
    ------
    ValueError
        If the file ends before the record does or if the record
        holds a negative count or length.
    """
    sg_num, sg_set, sg_axis, sg_cell = _read_struct(data, offset, "Bbbb")
    origin = _read_struct(data, offset + 4, "3f")
    lat = _read_struct(data, offset + 16, "6f")
    rot = _read_struct(data, offset + 40, "4f")
    (weight,) = _read_struct(data, offset + 56, "f")
    counts = _read_struct(data, offset + 60, "6h")
    num_atoms = counts[0]
    if num_atoms < 0:
        raise ValueError(f"the number of atoms {num_atoms} is negative")
    position = offset + CRYSTAL_SIZE
    atoms = []
    for _ in range(num_atoms):
        atoms.append(_read_atom(data, position))
        position += ATOM_SIZE
    raw_strings = []
    for length in counts[1:]:
        if length < 0:
            raise ValueError(f"a crystal string length {length} is negative")
        raw = _read_bytes(data, position, _pad8(length))
        position += len(raw)
        raw_strings.append(raw)
    crystal = ShtCrystal(
        sg_num=sg_num,
        sg_set=sg_set,
        sg_axis=sg_axis,
        sg_cell=sg_cell,
        origin=tuple(origin),
        lat=tuple(lat),
        rot=tuple(rot),
        weight=weight,
        num_atoms=num_atoms,
        atoms=atoms,
        formula=_decode_string(raw_strings[0], counts[1]),
        material_name=_decode_string(raw_strings[1], counts[2]),
        structure_symbol=_decode_string(raw_strings[2], counts[3]),
        references=_decode_string(raw_strings[3], counts[4]),
        note=_decode_string(raw_strings[4], counts[5]),
        formula_len=counts[1],
        material_name_len=counts[2],
        structure_symbol_len=counts[3],
        references_len=counts[4],
        note_len=counts[5],
        formula_bytes=raw_strings[0],
        material_name_bytes=raw_strings[1],
        structure_symbol_bytes=raw_strings[2],
        references_bytes=raw_strings[3],
        note_bytes=raw_strings[4],
    )
    return crystal, position


def _crystal_to_bytes(crystal: ShtCrystal) -> bytes:
    """Return the bytes of a crystal record.

    Parameters
    ----------
    crystal
        The crystal.

    Returns
    -------
    data
        The record: 72 bytes plus its atoms and padded strings.
    """
    strings = _crystal_strings(crystal)
    data = bytearray()
    data += _write_struct(
        "Bbbb", crystal.sg_num, crystal.sg_set, crystal.sg_axis, crystal.sg_cell
    )
    data += _write_struct("3f", *crystal.origin)
    data += _write_struct("6f", *crystal.lat)
    data += _write_struct("4f", *crystal.rot)
    data += _write_struct("f", crystal.weight)
    data += _write_struct(
        "6h",
        crystal.num_atoms,
        *[_string_length(*triplet) for triplet in strings],
    )
    for atom in crystal.atoms:
        data += _atom_to_bytes(atom)
    for triplet in strings:
        data += _encode_string(*triplet)
    return bytes(data)


def _read_atom(data: memoryview, offset: int) -> ShtAtom:
    """Return one atom record.

    Parameters
    ----------
    data
        Bytes of the whole file.
    offset
        Byte offset the record starts at.

    Returns
    -------
    atom
        The atom.
    """
    values = _read_struct(data, offset, "7f")
    (atomic_number,) = _read_struct(data, offset + 28, "b")
    res = _read_struct(data, offset + 29, "3b")
    return ShtAtom(
        x=values[0],
        y=values[1],
        z=values[2],
        occupancy=values[3],
        charge=values[4],
        debye_waller=values[5],
        res_fp=values[6],
        atomic_number=atomic_number,
        res=tuple(res),
    )


def _atom_to_bytes(atom: ShtAtom) -> bytes:
    """Return the bytes of an atom record.

    Parameters
    ----------
    atom
        The atom.

    Returns
    -------
    data
        The 32 byte record.
    """
    data = _write_struct(
        "7f",
        atom.x,
        atom.y,
        atom.z,
        atom.occupancy,
        atom.charge,
        atom.debye_waller,
        atom.res_fp,
    )
    data += _write_struct("b", atom.atomic_number)
    data += _write_struct("3b", *atom.res)
    return data


def _read_emsoft_simulation(record: bytes) -> ShtEMsoftSimulation:
    """Return a decoded ``EMsoftED`` simulation record.

    Parameters
    ----------
    record
        The 88 raw bytes of the record.

    Returns
    -------
    simulation
        The decoded record.
    """
    view = memoryview(record)
    version_bytes = _read_bytes(view, 0, 8)
    floats = _read_struct(view, 8, "10f")
    (tot_num_el,) = _read_struct(view, 48, "q")
    (num_sx,) = _read_struct(view, 56, "h")
    res = _read_struct(view, 58, "2b")
    bethe = _read_struct(view, 60, "5f")
    (num_px,) = _read_struct(view, 80, "h")
    (lat_grid_type,) = _read_struct(view, 82, "b")
    res2 = _read_struct(view, 83, "5b")
    return ShtEMsoftSimulation(
        emsoft_version=_decode_string(version_bytes, 8).rstrip("\x00"),
        sig_start=floats[0],
        sig_end=floats[1],
        sig_step=floats[2],
        omega=floats[3],
        kev=floats[4],
        e_hist_min=floats[5],
        e_bin_size=floats[6],
        depth_max=floats[7],
        depth_step=floats[8],
        thickness=floats[9],
        tot_num_el=tot_num_el,
        num_sx=num_sx,
        res=tuple(res),
        c1=bethe[0],
        c2=bethe[1],
        c3=bethe[2],
        sig_db_diff=bethe[3],
        d_min=bethe[4],
        num_px=num_px,
        lat_grid_type=lat_grid_type,
        res2=tuple(res2),
        emsoft_version_bytes=version_bytes,
    )


def _simulation_to_bytes(record: ShtEMsoftSimulation | bytes) -> bytes:
    """Return the bytes of a simulation record.

    Parameters
    ----------
    record
        A decoded ``EMsoftED`` record, or the raw bytes of an opaque
        one which are returned unchanged.

    Returns
    -------
    data
        The record.
    """
    if isinstance(record, (bytes, bytearray, memoryview)):
        return bytes(record)
    data = bytearray()
    data += _encode_fixed_string(record.emsoft_version, record.emsoft_version_bytes, 8)
    data += _write_struct(
        "10f",
        record.sig_start,
        record.sig_end,
        record.sig_step,
        record.omega,
        record.kev,
        record.e_hist_min,
        record.e_bin_size,
        record.depth_max,
        record.depth_step,
        record.thickness,
    )
    data += _write_struct("q", record.tot_num_el)
    data += _write_struct("h", record.num_sx)
    data += _write_struct("2b", *record.res)
    data += _write_struct(
        "5f", record.c1, record.c2, record.c3, record.sig_db_diff, record.d_min
    )
    data += _write_struct("h", record.num_px)
    data += _write_struct("b", record.lat_grid_type)
    data += _write_struct("5b", *record.res2)
    return bytes(data)


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
    fmt = "<" + fmt
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise ValueError(
            f"the SHT file ends before its {size} byte field at byte {offset} does"
        )
    return struct.unpack_from(fmt, data, offset)


def _read_bytes(data: memoryview, offset: int, size: int) -> bytes:
    """Return a raw byte range of ``data``.

    Parameters
    ----------
    data
        Bytes to read from.
    offset
        Byte offset to read at.
    size
        Number of bytes to read.

    Returns
    -------
    raw
        The bytes.

    Raises
    ------
    ValueError
        If the file ends before the range does.
    """
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(
            f"the SHT file ends before its {size} bytes at byte {offset} do"
        )
    return bytes(data[offset : offset + size])


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
    return struct.pack("<" + fmt, *values)


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
    return bytes(data[: max(int(length), 0)]).decode("utf-8", errors="replace")


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
    if raw is not None:
        return bytes(raw)
    encoded = text.encode("utf-8")
    return encoded.ljust(_pad8(len(encoded)), b"\x00")


def _encode_fixed_string(text: str, raw: bytes | None, size: int) -> bytes:
    """Return the raw bytes of a fixed width string field.

    Parameters
    ----------
    text
        The text.
    raw
        Raw bytes as read, or ``None``.
    size
        Width of the field in bytes.

    Returns
    -------
    data
        ``raw`` when it is given, else ``text`` encoded as UTF-8,
        either way truncated or NUL padded to ``size`` bytes.
        SHTfile NUL pads these fields
        (``FileHeader::FileHeader``, ``sht_file.in.hpp`` lines
        1053-1063) and never space pads them.
    """
    encoded = bytes(raw) if raw is not None else text.encode("utf-8")
    return encoded[:size].ljust(size, b"\x00")


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
    if length is not None:
        return int(length)
    return len(text.encode("utf-8"))


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
