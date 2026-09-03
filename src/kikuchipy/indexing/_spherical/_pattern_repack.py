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

import os
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import dask.array as da
import h5py
import numpy as np

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
    from kikuchipy.io._util import _ensure_directory, _overwrite

    if manufacturer not in _MANUFACTURER_FLIP:
        raise ValueError(
            f"The manufacturer {manufacturer!r} is not one of "
            f"{sorted(_MANUFACTURER_FLIP)}, the vendor strings EMSphInx's "
            "pattern reader accepts. Note that these are not the vendors of "
            "an EMSphInx namelist, which are a different set"
        )

    data = signal.data
    dtype = np.dtype(data.dtype)
    if dtype.name not in _DTYPES:
        raise ValueError(
            f"The signal data type {dtype.name!r} is not one of "
            f"{list(_DTYPES)}, the pixel types EMSphInx reads"
        )

    binning = int(binning)
    if binning < 1:
        raise ValueError(f"The binning factor {binning} must be at least one")
    height, width = int(data.shape[-2]), int(data.shape[-1])
    if height % binning != 0 or width % binning != 0:
        raise ValueError(
            f"The binning factor {binning} must divide both signal "
            f"dimensions {(height, width)}"
        )

    if flip is None:
        flip = _MANUFACTURER_FLIP[manufacturer]
    flip = bool(flip)

    native_dtype = dtype.newbyteorder("=")
    written_dtype = np.dtype("float32") if bin_to_float else native_dtype

    filename = str(filename)
    if os.path.splitext(filename)[1] == "":
        filename += ".h5"

    is_file = os.path.isfile(filename)
    if overwrite is None:
        write = _overwrite(filename)
    elif overwrite is True or (overwrite is False and not is_file):
        write = True
    elif overwrite is False and is_file:
        write = False
    else:
        raise ValueError(
            f"overwrite can only be None, True or False, and not {overwrite}"
        )
    if not write:
        return

    # after the decision, so that a refused write neither warns about
    # a file it does not write nor leaves a new directory behind
    if written_dtype != np.uint8:
        warn(
            f"The written data type {written_dtype.name!r} is not 'uint8': "
            "EMSphInx reads HDF5 patterns through a buffered unsigned 8-bit "
            "read whatever the data set type is, and corrupts every other "
            "type (measured: an unsigned 16-bit twin of an otherwise correct "
            "file indexes 38.9 degrees off). The file is still written, as "
            "the ported program writes it",
            UserWarning,
        )
    _ensure_directory(filename)

    n_patterns = int(np.prod(data.shape[:-2]))
    patterns = data.reshape((n_patterns, height, width))
    if flip:
        patterns = patterns[:, ::-1, :]
    if bin_to_float:
        patterns = _bin_float(patterns, binning)
    else:
        patterns = _bin_avg(patterns, binning, native_dtype)

    shape = (n_patterns, height // binning, width // binning)
    with h5py.File(filename, mode="w") as f:
        f.create_dataset(
            "Manufacturer",
            data=manufacturer.encode("ascii"),
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        dataset = _create_patterns_dataset(f, shape, written_dtype)
        if isinstance(patterns, da.Array):
            # streams one chunk at a time into the pre-allocated data
            # set, so the whole map is never held in memory
            da.store(patterns, dataset)
        else:
            dataset[...] = patterns


def _create_patterns_dataset(
    file: h5py.File, shape: tuple[int, int, int], dtype: np.dtype
) -> h5py.Dataset:
    """Return a new contiguous, early allocated and unfiltered
    ``/patterns`` data set, the layout of ``pattern_repack.cpp``
    lines 199-208.

    Early allocation is what lets the C++ program get a raw binary
    offset with ``getOffset()``, and what lets a lazy signal be
    written slab by slab here.  The zero filters are the functional
    part: a compressed data set is fatal to the reader.
    """
    space = h5py.h5s.create_simple(shape)
    properties = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    properties.set_layout(h5py.h5d.CONTIGUOUS)
    properties.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    type_id = h5py.h5t.py_create(dtype, logical=True)
    identifier = h5py.h5d.create(file.id, b"patterns", type_id, space, properties)
    return h5py.Dataset(identifier)


def _bin_sums(patterns, binning: int, accumulate: type):
    """Return the ``binning`` x ``binning`` block sums of a stack of
    patterns, accumulated in ``accumulate``.

    The accumulator is the data type of the summation rather than a
    cast of the input, which is bitwise the same (measured on the
    nickel map at twelve binning factors, in both flip directions
    and for all three input types) and never copies the input: an
    eager (10000, 480, 640) unsigned 8-bit map would otherwise need
    a 24 GB 64-bit float temporary, while the C++ ``binAvg()`` bins
    one pattern at a time.
    """
    n_patterns, height, width = patterns.shape
    reshaped = patterns.reshape(
        n_patterns, height // binning, binning, width // binning, binning
    )
    return reshaped.sum(axis=(2, 4), dtype=accumulate)


def _bin_avg(patterns, binning: int, dtype: np.dtype):
    """Return the block mean of a stack of patterns in its own data
    type, ``binAvg()`` (``pattern_repack.cpp`` lines 76-98).

    The block sum is accumulated in 64-bit floats, divided by
    ``binning ** 2`` and, for integer types only, rounded half away
    from zero, which is ``std::round()``.  The values are
    non-negative, so that rounding is ``floor(x + 0.5)`` and **not**
    :func:`numpy.round`, which is banker's rounding and differs on
    1003 of the 8100 pixels of the small nickel map at binning two.
    """
    if binning == 1:
        # ``copy=False`` because the only conversion this can make is
        # the byte order one, which a native input does not need
        return patterns.astype(dtype, copy=False)
    mean = _bin_sums(patterns, binning, np.float64) / binning**2
    if np.issubdtype(dtype, np.integer):
        mean = np.floor(mean + 0.5)
    return mean.astype(dtype)


def _bin_float(patterns, binning: int):
    """Return the block **sum** of a stack of patterns as 32-bit
    floats, ``binFloat()`` (``pattern_repack.cpp`` lines 50-67).

    ``binning == 1`` is a plain cast, which completes the dead code
    of the C++ ``main()``.  The accumulation order is NumPy's
    pairwise summation, which is the authority here: the mode is
    unreachable in the shipped binary.
    """
    if binning == 1:
        return patterns.astype(np.float32, copy=False)
    return _bin_sums(patterns, binning, np.float32)
