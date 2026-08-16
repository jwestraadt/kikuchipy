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
# - Smallest fast FFT size not smaller than a given size, i.e. the
#   smallest {2, 3, 5, 7, 11, 13}-smooth size (``fastSize()`` in
#   ``include/util/fft.hpp``, lines 438-491)
# - The reasonable bandwidth range [16, 512] of the EBSD indexing
#   name list (``EMsoftNameListHandler`` in
#   ``include/modality/ebsd/nml.hpp``, line 635)

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
# Modified by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy and conveyed under
# GPL-3.0-or-later
# #####################################################################

"""Fast FFT sizes for the spherical harmonic transform and the
spherical cross-correlation.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.
"""

import numpy as np


def fast_size(n: int) -> int:
    """Return the smallest fast FFT size not smaller than ``n``.

    A size is fast when it is a product of the small primes 2, 3, 5,
    7, 11 and 13 ("13-smooth"), which are the transform lengths FFTW
    (and thus EMSphInx) has hard-coded codelets for.

    Parameters
    ----------
    n
        Minimum transform length. Lengths of 16 or less are returned
        unchanged, except zero which returns one, because FFTW
        implements every transform of length 1-16 explicitly.

    Returns
    -------
    size
        Smallest 13-smooth size not smaller than ``max(1, n)``.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Notes
    -----
    This is a verbatim port of EMSphInx' ``fft::fastSize()`` and not
    :func:`scipy.fft.next_fast_len`, because the latter is 11-smooth
    only for real transforms and 7-smooth otherwise, and because the
    zero padding of the spherical cross-correlation must match
    EMSphInx exactly.

    Examples
    --------
    >>> from kikuchipy.indexing._spherical._fft import fast_size
    >>> fast_size(0)
    1
    >>> fast_size(16)
    16
    >>> fast_size(17)
    18
    >>> fast_size(105)
    105
    >>> fast_size(109)
    110
    """
    raise NotImplementedError


def fast_bandwidths(bandwidth_min: int = 16, bandwidth_max: int = 512) -> np.ndarray:
    """Return the bandwidths with a fast cross-correlation transform.

    The spherical cross-correlation of two functions band-limited at
    ``bandwidth`` needs three-dimensional FFTs of side length
    ``2 * bandwidth - 1``. Bandwidths for which this length is already
    a fast size (see :func:`fast_size`) need no zero padding and are
    therefore the fastest to index with.

    Parameters
    ----------
    bandwidth_min
        Smallest bandwidth to consider, inclusive. Default is 16, the
        lower limit EMSphInx accepts in its EBSD name list.
    bandwidth_max
        Largest bandwidth to consider, inclusive. Default is 512, the
        upper limit EMSphInx accepts in its EBSD name list.

    Returns
    -------
    bandwidths
        Sorted 1D array of 64-bit integer bandwidths in the closed
        interval [``bandwidth_min``, ``bandwidth_max``] for which
        ``fast_size(2 * bandwidth - 1) == 2 * bandwidth - 1``.

    Raises
    ------
    ValueError
        If ``bandwidth_min`` is smaller than one or larger than
        ``bandwidth_max``.

    Notes
    -----
    The returned array contains all bandwidths recommended in
    EMSphInx' EBSD name list (``include/modality/ebsd/nml.hpp``, lines
    298 and 415), which are the bandwidths this function is validated
    against.
    """
    raise NotImplementedError
