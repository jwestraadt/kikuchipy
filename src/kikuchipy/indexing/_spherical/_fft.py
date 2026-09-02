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
#   name list (``emsphinx::ebsd::Namelist::sanityCheck()`` in
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
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
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

    The port is verbatim, so it inherits the one deviation of the C++
    from its own docstring: the set-product iteration runs
    ``log2(log2(next power of two))`` rounds and so only forms
    products of at most that many squarings of the small primes. For
    ``n`` in ``range(757, 769)`` this cannot reach
    ``768 = 2 ** 8 * 3`` (nine prime factors) and 770 is returned
    instead. These are the only such sizes below 1101, and 770 is
    still 13-smooth, so the transform stays fast.

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
    if n < 0:
        raise ValueError(f"Minimum transform length {n} cannot be negative")

    # Handle the special/easy cases: FFTW explicitly implements all
    # FFTs from 1 -> 16
    if n <= 16:
        return max(1, n)

    # Start by computing the next power of two up from n
    # https://graphics.stanford.edu/~seander/bithacks.html
    v2n = n - 1
    v2n |= v2n >> 1
    v2n |= v2n >> 2
    v2n |= v2n >> 4
    v2n |= v2n >> 8
    v2n |= v2n >> 16
    v2n = (v2n + 1) & 0xFFFFFFFF  # First power of two >= n

    # Now compute the log_2 of v2n. The mask b[i] selects the bits
    # whose position has bit i set, so OR-ing the flags together spells
    # out the position of the single set bit of a power of two
    # https://graphics.stanford.edu/~seander/bithacks.html
    b = (0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000)
    log2n = int((v2n & b[0]) != 0)
    for i in range(4, 0, -1):
        log2n |= int((v2n & b[i]) != 0) << i

    # Next compute log_2(log_2(v2n)), since we will be squaring in the
    # last step
    max_iter = int((log2n & b[0]) != 0)
    for i in range(4, 0, -1):
        max_iter |= int((log2n & b[i]) != 0) << i

    # Now compute all combinations of 2^i * 3^j * 5^k... for the fast
    # (small) primes. The small primes for FFTW are 2, 3, 5, 7, 11 and
    # 13. i, j, k ... only need to be checked for i < r
    sizes = {2, 3, 5, 7, 11, 13}
    # Our initial guess for the smallest fast size is the next power of
    # two up
    size_min = v2n
    for _ in range(max_iter):  # Loop over required iterations
        sizes_new = set()  # Set to hold new elements to be added
        for i in sorted(sizes):  # Loop over current elements once
            for j in sorted(sizes):  # Loop over current elements twice
                v = i * j  # Compute product of elements
                # Is this element small enough to care about (smaller
                # than our current best)?
                if v < size_min:
                    if v < n:  # Values less than n are prefactors
                        sizes_new.add(v)
                    else:  # Otherwise (>= n) a new best size is found
                        size_min = v
        sizes |= sizes_new  # Add our new prefactors

    return size_min


def fast_bandwidths(bandwidth_min: int = 16, bandwidth_max: int = 512) -> np.ndarray:
    """Return the bandwidths with a fast cross-correlation transform.

    The spherical cross-correlation of two functions band-limited at
    ``bandwidth`` needs three-dimensional FFTs of side length
    ``2 * bandwidth - 1``. Bandwidths for which this length is already
    a fast transform size, i.e. a product of the prime factors 2, 3, 5,
    7, 11 and 13 which the transform library is fastest for, need no
    zero padding and are therefore the fastest to index with.

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

    See Also
    --------
    kikuchipy.indexing.SphericalIndexer
    kikuchipy.signals.EBSD.spherical_indexing

    Notes
    -----
    The returned array contains all bandwidths recommended in
    EMSphInx' EBSD name list (``include/modality/ebsd/nml.hpp``, lines
    298 and 415), which are the bandwidths this function is validated
    against.

    Examples
    --------
    The bandwidths between 50 and 90 which need no zero padding, of
    which 68 is the indexing default:

    >>> import kikuchipy as kp
    >>> kp.indexing.fast_bandwidths(50, 90)
    array([50, 53, 59, 61, 63, 68, 72, 74, 83, 85, 88])
    """
    if bandwidth_min < 1:
        raise ValueError(f"Smallest bandwidth {bandwidth_min} must be at least one")
    if bandwidth_min > bandwidth_max:
        raise ValueError(
            f"Smallest bandwidth {bandwidth_min} cannot be greater than the "
            f"largest bandwidth {bandwidth_max}"
        )

    bandwidths = [
        bw
        for bw in range(bandwidth_min, bandwidth_max + 1)
        if fast_size(2 * bw - 1) == 2 * bw - 1
    ]

    return np.array(bandwidths, dtype=np.int64)
