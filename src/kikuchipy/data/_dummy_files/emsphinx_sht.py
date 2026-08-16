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

"""Creation of small synthetic EMSphInx *.sht files for testing of the
reader and the writer.

One file per distinct ``(z_rot, compression flags)`` pair of the SHT
file space group tables, 25 in total, so that every branch of the
packing is exercised.  They are written at test time by the session
scoped ``emsphinx_synthetic_sht_files`` fixture and never shipped,
since they come from our own writer.
"""

from pathlib import Path

import numpy as np

from kikuchipy.indexing._spherical import _sht_file

SYNTHETIC_SPACE_GROUPS: tuple[int, ...] = (
    1, 2, 6, 10, 16, 25, 47, 75, 83, 99, 111, 123, 143,
    147, 156, 157, 162, 164, 168, 174, 175, 183, 187, 189, 191,
)  # fmt: skip
"""Lowest space group of every distinct ``(z_rot, compression flags)``
pair of ``space_group_z_rotation`` and
``space_group_compression_flags``, 25 in total."""

SYNTHETIC_BANDWIDTH = 16
"""Bandwidth of the synthetic files, small enough to keep them below
2.5 kB."""

SYNTHETIC_SOFTWARE_VERSION = b"kp-fixt\x00"
"""Software version field of the synthetic files.

Fixed rather than taken from ``kikuchipy.__version__``, so that the
pinned md5 sums do not change with every release.
"""

SYNTHETIC_NOTES = "kikuchipy synthetic fixture"
"""Notes string of the synthetic files."""

SYNTHETIC_LATTICE = (0.4, 0.4, 0.4, 90.0, 90.0, 90.0)
"""Cubic 0.4 nm lattice of the one crystal of every synthetic file."""

SYNTHETIC_BEAM_ENERGY = 20.0
"""Beam energy in keV of the synthetic files."""

SYNTHETIC_PRIMARY_ANGLE = 70.0
"""Primary angle in degrees of the synthetic files."""


def synthetic_coefficients(bandwidth: int, z_rot: int, flags: int) -> np.ndarray:
    """Return coefficients which the packing keeps in full.

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
    alm
        Coefficients ``alm[m, l]`` of shape
        ``(bandwidth, bandwidth)`` and 128-bit complex data type,
        non-zero only where the packing stores something.

    Notes
    -----
    No random number generator is involved, so the bytes are
    reproducible across NumPy versions and platforms and the md5 sums
    of the files can be pinned.  The real part of a kept entry is
    ``((7 m + 13 l + 3) % 17 - 8) / 8`` and the imaginary part
    ``((5 m + 11 l + 1) % 19 - 9) / 9``, with the imaginary part
    zeroed on rows the flags store as real and the real part zeroed on
    rows they store as imaginary, so that packing is lossless and each
    of the four storage branches is exercised.
    """
    inversion = bool(flags & _sht_file.FLAG_INVERSION)
    mirror_z = bool(flags & _sht_file.FLAG_MIRROR_Z)
    mirror_y = bool(flags & _sht_file.FLAG_MIRROR_Y)
    mirror_x = bool(flags & _sht_file.FLAG_MIRROR_X)
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for order in range(bandwidth):
        if z_rot > 1 and order % z_rot != 0:
            continue
        if mirror_y:
            kind = "real"
        elif mirror_x:
            kind = "real" if order % (2 * z_rot) == 0 else "imaginary"
        else:
            kind = "complex"
        for degree in range(order, bandwidth):
            if inversion and degree % 2:
                continue
            if mirror_z and (degree + order) % 2:
                continue
            real = ((7 * order + 13 * degree + 3) % 17 - 8) / 8
            imaginary = ((5 * order + 11 * degree + 1) % 19 - 9) / 9
            if kind == "real" or order == 0:
                alm[order, degree] = complex(real, 0.0)
            elif kind == "imaginary":
                alm[order, degree] = complex(0.0, imaginary)
            else:
                alm[order, degree] = complex(real, imaginary)
    return alm


def create_synthetic_sht_files(directory: str | Path) -> dict[int, Path]:
    """Write one synthetic *.sht file per space group of
    :data:`SYNTHETIC_SPACE_GROUPS`.

    Parameters
    ----------
    directory
        Directory to write into, created if it does not exist.

    Returns
    -------
    files
        Path of every written file keyed on its space group, named
        ``sg<NNN>_bw16.sht``.

    Notes
    -----
    Every file is EBSD at 20 keV and 70 degrees, has one crystal with
    the space group as its number, a cubic 0.4 nm lattice and no
    atoms, no simulation record (``sim_meta_size`` 0), vendor
    ``Unknown`` and the coefficients of
    :func:`synthetic_coefficients`.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for space_group in SYNTHETIC_SPACE_GROUPS:
        z_rot = _sht_file.space_group_z_rotation(space_group)
        flags = _sht_file.space_group_compression_flags(space_group)
        alm = synthetic_coefficients(SYNTHETIC_BANDWIDTH, z_rot, flags)
        packed = _sht_file.pack_harmonics(alm, SYNTHETIC_BANDWIDTH, z_rot, flags)
        header = _sht_file.ShtHeader(
            software_version=SYNTHETIC_SOFTWARE_VERSION.decode().rstrip("\x00"),
            software_version_bytes=SYNTHETIC_SOFTWARE_VERSION,
            modality=_sht_file.MODALITY_EBSD,
            beam_energy=SYNTHETIC_BEAM_ENERGY,
            primary_angle=SYNTHETIC_PRIMARY_ANGLE,
            doi="",
            notes=SYNTHETIC_NOTES,
        )
        sht_file = _sht_file.ShtFile(
            header=header,
            num_xtal=1,
            sg_eff=space_group,
            pijk=1,
            rot_sense=112,
            modality=_sht_file.MODALITY_EBSD,
            vendor=_sht_file.VENDOR_UNKNOWN,
            sim_meta_size=0,
            crystals=[_sht_file.ShtCrystal(sg_num=space_group, lat=SYNTHETIC_LATTICE)],
            simulations=[None],
            harmonics=_sht_file.ShtHarmonics(
                bandwidth=SYNTHETIC_BANDWIDTH,
                z_rot=z_rot,
                flags=flags,
                doub_cnt=packed.size,
                packed=packed,
            ),
        )
        fpath = directory / f"sg{space_group:03d}_bw16.sht"
        _sht_file.write_sht(fpath, sht_file)
        files[space_group] = fpath
    return files
