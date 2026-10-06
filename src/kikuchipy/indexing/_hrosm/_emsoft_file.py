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

"""Private reader of EMsoft's dictionary indexing (dot product) and
EMHROSM HDF5 files, for tests and the reference script.

Arrays EMsoft writes in Fortran order are returned in NumPy's order:
an ``(x, y)`` map as ``(H, W)``, a ``(3, N)`` array as ``(N, 3)``, so
the flat index ``y W + x`` is EMsoft's ``(iy - 1) W + ix`` minus one.
"""

from __future__ import annotations

from pathlib import Path


def read_emsoft_dot_product_file(path: str | Path) -> dict:
    """Return the arrays and namelist values of an EMsoft dot product
    file.

    The map shape is ``(ipf_ht, ipf_wd)``, or ``(ROI[3], ROI[2])``
    when the ROI is not all zeros. ``TopMatchIndices``, padded by
    EMsoft to a multiple of ``numexptsingle`` rows, is sliced to the
    number of map points and kept 1-based. Euler angles are returned
    in radians, unchanged.

    Parameters
    ----------
    path
        Path to the HDF5 file.

    Returns
    -------
    data
        Arrays of ``Scan 1/EBSD/Data``, the namelist values of
        ``NMLparameters/EMDINameList`` as Python scalars, the EMsoft
        version and the map shape.
    """
    raise NotImplementedError


def read_emsoft_hrosm_file(path: str | Path) -> dict:
    """Return the arrays and namelist values of an EMsoft EMHROSM file.

    ``grainROI`` is returned 1-based as ``(x0, y0, w, h)``, as EMsoft
    writes it. ``dilate`` is parsed from the namelist text
    ``NMLfiles/HROSMNML``, since ``NMLparameters/HROSMNameList`` lacks
    it. The optional ``newQuat`` and ``maxGROD`` are None when absent.

    Parameters
    ----------
    path
        Path to the HDF5 file.

    Returns
    -------
    data
        Arrays of ``EMData/HROSM``, the namelist values, the namelist
        text and the EMsoft version.
    """
    raise NotImplementedError


def parse_namelist_text(text: str) -> dict:
    """Return the ``key = value`` pairs of a Fortran namelist text as
    Python values (Fortran logicals as bool, quoted strings without
    quotes, numbers as int or float, several values as a list).

    Parameters
    ----------
    text
        Namelist text; comment lines start with ``!``.

    Returns
    -------
    values
        Values keyed on the namelist keys.
    """
    raise NotImplementedError
