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

from pathlib import Path
from typing import Literal

__all__ = ["file_reader"]

HEMISPHERE = Literal["upper", "lower", "both"]


def file_reader(
    filename: str | Path,
    dim: int | None = None,
    hemisphere: HEMISPHERE = "both",
    lazy: bool = False,
) -> list[dict]:
    """Read an EBSD master pattern from an EMSphInx *.sht file.

    The file holds the master pattern as spherical harmonic
    coefficients, which are synthesized onto a square Lambert grid.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Path to the *.sht file, version 1.1 of the SHTfile format.
    dim
        Odd side length of the square Lambert grid, at least three.
        If not given, ``2 * bandwidth + 1`` is used, the grid whose
        largest bandwidth is exactly the file's, i.e. 769 for the
        usual bandwidth of 384.
    hemisphere
        Which hemisphere to return: "upper", "lower", or "both"
        (default).
    lazy
        Not supported; included for API compatibility. Data are always
        synthesized eagerly.

    Returns
    -------
    signal_dict_list
        Data, axes, metadata, original metadata, projection, phase and
        hemisphere.

    Raises
    ------
    ValueError
        If ``hemisphere`` is not one of "upper", "lower" and "both",
        if ``dim`` is even or smaller than three, if the file is not
        an SHT file, or if its checksum does not match.
    NotImplementedError
        If the file's modality is not EBSD, if it holds more than one
        crystal, if it is big-endian, or if its version is not 1.1.
        Only EBSD master patterns of a single crystal are read; the
        codec parses the other modalities (ECP, TKD, PED and Laue) and
        multi-crystal files, but nothing is built from them.

    Warns
    -----
    UserWarning
        If ``(dim - 1) // 2`` is smaller than the file's bandwidth, in
        which case the coefficients are band limited to that degree.
    """
    raise NotImplementedError
