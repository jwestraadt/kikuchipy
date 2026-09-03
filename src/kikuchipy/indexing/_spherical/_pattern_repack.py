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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517):
# - ``binFloat()`` (``programs/pattern_repack.cpp``, lines 50-67) and
#   ``binAvg()`` (lines 76-98), as the binning of
#   :func:`write_emsphinx_patterns`
# - ``flipPat()`` (lines 105-113), as its vertical flip
# - the HDF5 data set layout and write loops (lines 183-208 and
#   210-293), as its file writing
# - ``PatternFile::GetVendor()`` (``include/modality/ebsd/
#   pattern.hpp``, lines 608-637) and the vendor flip table (lines
#   463-471), as the ``Manufacturer`` data set contract
#
# **Deliberate deviations from the ported program**, both documented
# in :func:`write_emsphinx_patterns`:
# - the writer **always writes the root ``Manufacturer`` data set**,
#   which ``pattern_repack.cpp`` omits.  Without it ``IndexEBSD``
#   refuses the file ("<file> doesn't have a Manufacturer string",
#   ``pattern.hpp`` line 623, measured), so the C++ contract is
#   completed rather than copied.
# - ``bin_to_float=True`` with ``binning == 1`` casts to 32-bit
#   float.  The C++ ``main()`` routes ``binning == 1`` through the
#   raw copy (lines 212-218) and never through ``binFloat()``, so
#   that output type is unreachable there; the whole mode is dead
#   code in the shipped binary (``const bool binToFloat = false``,
#   line 116).
#
# The following are deliberately **not** ported here:
# - the ``*.up1``, ``*.up2``, ``*.data`` and ``*.ebsp`` *input* paths
#   of ``PatternFile::Read()`` (the input is a kikuchipy signal;
#   ``kikuchipy.load()`` reads ``*.ebsp``)
# - the stdout report and timings (lines 154-172) and the ``argv``
#   command line interface

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Changed by Johan Westraadt, 2026-09: translated to Python/NumPy/h5py
# for kikuchipy. GPL-2.0-or-later, conveyed under GPL-3.0-or-later
# #####################################################################

"""Writing of EBSD patterns to the repacked HDF5 file EMSphInx's
``IndexEBSD`` reads, the equivalent of its ``PatternRepack`` program.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

The file has exactly two objects:

* the root scalar data set ``Manufacturer``, a variable length
  **ASCII** string.  ``PatternRepack`` writes none, but
  ``PatternFile::GetVendor()`` requires one, and h5py's default
  variable length **UTF-8** string is fatal to the reader with a
  misleading ``H5Dread failed`` error, so the character set is part
  of the contract.
* ``/patterns`` of shape ``(n, height, width)``, contiguous, with
  allocation time ``EARLY`` and **no filters**.  The zero filter rule
  is the functional one: the memory map gate of ``PatternFile::Read``
  (``pattern.hpp`` line 494) requires ``0 != getNfilters()``, which
  no contiguous data set can satisfy, so every HDF5 pattern file is
  read through the buffered branch, which fails on a compressed data
  set.  Contiguous with early allocation is byte layout parity with
  ``PatternRepack``, not a reader requirement.

That buffered branch reads with ``NATIVE_UINT8`` whatever the data
set type is (line 515), so **only unsigned 8-bit files are read
correctly**; other types are written but warned about.  Byte swapped
types are rejected outright by the reader's ``NATIVE_*`` comparison
(lines 476-480), so a non-native input is cast to native order.

The vertical flip is decided by the ``Manufacturer`` string, because
that is what the reader uses (the flip table of ``pattern.hpp`` lines
463-471) and EMSphInx's own convention is origin bottom left.  Flip
and a dividing binning factor commute exactly, so the contract is
stated as the equivalence ``binned(flip(x)) == flip(binned(x))``
rather than as an internal order.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.signals import EBSD, LazyEBSD

__all__ = ["write_emsphinx_patterns"]

# The writer side resolution of ``flip=None``, the inverse of the
# reader's vendor flip table (``pattern.hpp`` lines 463-471).
# ``True`` means **kikuchipy pre-flips** the rows, because the
# reader does not flip this vendor (its ``vendorFlip`` is false);
# ``False`` means the reader flips them itself, so the rows are
# written as they are.  This table is therefore the negation of the
# C++ one: ``vendorFlip`` is true for ``"EDAX"`` and ``"EMsoft"``
# only.  Either way the pattern EMSphInx interpolates is the row
# reversed kikuchipy pattern.
_MANUFACTURER_FLIP: dict[str, bool] = {
    "EDAX": False,
    "EMsoft": False,
    "Oxford": True,
    "Bruker": True,
    "Bruker Nano": True,
    "DREAM.3D": True,
}

# The pixel types of ``ImageSource::Bits`` (``pattern_repack.cpp``
# lines 184-190)
_DTYPES = ("uint8", "uint16", "float32")


def write_emsphinx_patterns(
    filename: str | Path,
    signal: "EBSD | LazyEBSD",
    *,
    manufacturer: str = "EMsoft",
    binning: int = 1,
    bin_to_float: bool = False,
    flip: bool | None = None,
    overwrite: bool | None = None,
) -> None:
    r"""Write EBSD patterns to the repacked HDF5 file EMSphInx's
    ``IndexEBSD`` reads :cite:`lenthe2019spherical`.

    The equivalent of EMSphInx's ``PatternRepack`` program, plus the
    root ``Manufacturer`` data set that program omits and
    ``IndexEBSD`` requires.

    Parameters
    ----------
    filename
        Path to write to.  ``".h5"`` is appended when there is no
        suffix, and the parent directory is created if it does not
        exist.
    signal
        Signal with the patterns to write.  The patterns are written
        in row major navigation order, and a signal without
        navigation dimensions writes one pattern.
    manufacturer
        Vendor string written to the root ``Manufacturer`` data set,
        one of ``"EDAX"``, ``"EMsoft"`` (default), ``"Oxford"``,
        ``"Bruker"``, ``"Bruker Nano"`` and ``"DREAM.3D"``.  These
        are exactly the strings the reader accepts; any other raises.
        It also decides the default vertical flip, see *Notes*.
    binning
        Binning factor, 1 by default, which must divide both signal
        dimensions.  Each output pixel is the mean of a
        ``binning`` x ``binning`` block, rounded half away from zero
        for integer data types.
    bin_to_float
        Whether to write 32-bit floats, ``False`` by default.  Each
        output pixel is then the block **sum**, and ``binning == 1``
        is a plain cast.
    flip
        Whether to reverse the pattern rows before writing.  If not
        given (default), it is resolved from ``manufacturer`` so that
        the pattern EMSphInx ultimately interpolates is the row
        reversed kikuchipy pattern.  Passing it explicitly breaks
        that orientation parity; ``True`` is ``PatternRepack``'s own
        hard coded behaviour.
    overwrite
        Whether to overwrite an existing file.  If not given, the
        user is asked.

    Raises
    ------
    ValueError
        If ``manufacturer`` is not one of the accepted strings; if
        the signal data type is not ``uint8``, ``uint16`` or
        ``float32``; if ``binning`` is smaller than one or does not
        divide both signal dimensions; or if ``overwrite`` is not
        ``None``, ``True`` or ``False``.

    Warns
    -----
    UserWarning
        If the written data type is not ``uint8``.  EMSphInx reads
        HDF5 patterns through a buffered unsigned 8-bit read and
        corrupts every other type (measured: an unsigned 16-bit twin
        of an otherwise correct file indexes 38.9 degrees off).  The
        file is still written, as ``PatternRepack`` writes it.

    See Also
    --------
    kikuchipy.indexing.EMSphInxNamelist

    Notes
    -----
    The written file has two objects: the scalar variable length
    ASCII string ``Manufacturer`` and ``/patterns`` of shape
    ``(n, height, width)``, contiguous, allocated early and without
    filters.  Compression is fatal to the reader, and a byte swapped
    data type is rejected by it, so a non-native input is cast to
    native byte order before writing.

    ``manufacturer`` and the ``vendor`` field of an
    :class:`~kikuchipy.indexing.EMSphInxNamelist` are different
    knobs with different value sets: this one controls the flip of
    the file, that one the interpretation of the pattern centre.  Any
    combination is valid.

    The default ``manufacturer="EMsoft"`` resolves ``flip`` to
    ``False``, which makes ``/patterns`` byte identical to the
    signal data.  Vendors whose reader side flip is false
    (``"Oxford"``, ``"Bruker"``, ``"Bruker Nano"``, ``"DREAM.3D"``)
    are pre-flipped instead.  Both routes index correctly and are
    equivalent but not bitwise identical downstream, since the reader
    flips at interpolation time.

    A lazy signal is streamed chunk by chunk into the pre-allocated
    data set, so the whole map is never held in memory.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> fname = Path(tempfile.mkdtemp()) / "patterns.h5"
    >>> kp.indexing.write_emsphinx_patterns(fname, s, overwrite=True)
    """
    raise NotImplementedError
