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
# - ``gaussian::Model<Real>::estimate()`` (``include/util/
#   gaussian.hpp``, lines 140-163) and ``Model<Real>::fit()`` (lines
#   172-231), as :func:`_fit_gaussian_1d_kernel` and
#   :func:`_fit_gaussian_1d`
# - ``gaussian::BckgSub2D<Real>::CircMask()`` (lines 238-251), as
#   :func:`_circular_mask`
# - ``gaussian::BckgSub2D<Real>::fit()`` (lines 257-316), as
#   :func:`_row_col_max_kernel` and :func:`_gaussian_background`
# - the out of place ``gaussian::BckgSub2D<Real>::subtract()`` (lines
#   379-386), as :func:`_remove_gaussian_background`
# - ``solve::cholesky()`` (``include/util/linalg.hpp``, lines
#   354-358), i.e. ``decompose::cholesky()`` (lines 411-431) and
#   ``backsolve::cholesky()`` (lines 487-493) including its ``neg``
#   negation, for the 3x3 normal equations, as
#   :func:`_cholesky_solve_3x3`
# - ``AdaptiveHistogramEqualizer<Real, uint8_t>::setSize()``
#   (``include/util/ahe.hpp``, lines 117-190), as :func:`_ahe_tiles`
# - ``AdaptiveHistogramEqualizer<Real, uint8_t>::computeHist()``
#   (lines 193-222), as :func:`_ahe_cdf_kernel`
# - the out of place ``AdaptiveHistogramEqualizer<Real, uint8_t>::
#   equalize()`` (lines 247-262), as :func:`_ahe_equalize_kernel`
# - ``PatternProcessor<Real>::setSize()`` (``include/modality/ebsd/
#   imprc.hpp``, lines 108-146) and the out of place
#   ``PatternProcessor<Real>::process()`` (lines 166-191), as
#   :func:`_to_uint8` and :func:`_preprocess_pattern`
#
# The following are deliberately **not** ported here:
# - the in place integer variants of ``BckgSub2D<Real>::subtract()``
#   (``include/util/gaussian.hpp``, lines 321-372), used only by the
#   in place ``process(uint8_t*)`` (``imprc.hpp``, lines 151-160),
#   which the indexer never calls, and the in place ``equalize()``
#   (``include/util/ahe.hpp``, lines 227-241)
# - the streaming ``image::adHistEq()`` (``include/util/image.hpp``,
#   lines 285-417), a different algorithm which the pattern pipeline
#   does not use
#
# This module imports **nothing** from SciPy's fast Fourier
# transform package: the discrete cosine image quality of
# ``image::imageQuality()`` lives in ``_back_projection`` alone.

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
# Changed by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""EMSphInx pattern preprocessing: a separable Gaussian background
and a mosaic adaptive histogram equalisation.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**The order is frozen** (``PatternProcessor<Real>::process()``,
``include/modality/ebsd/imprc.hpp`` lines 166-191).
:func:`_preprocess_pattern` takes one of three branches:

1. ``gaussian_background``: fit the separable background, subtract
   it (masked pixels become ``0``), then, if ``n_regions > 0``,
   convert the subtracted float image to unsigned 8-bit and
   equalise it.
2. else ``n_regions > 0``: equalise, taking an unsigned 8-bit input
   **directly** (lines 179-180, no rescale) and converting any other
   input first (lines 181-186).
3. else: the 64-bit float cast (line 188).

The defaults are ``IndexEBSD``'s namelist ones: ``n_regions = 10``,
``gaussian_background = False`` and no mask.

**``circmask`` mapping** (the C++ namelist value, gotcha 21):

.. code-block::

    circmask   good_pixels                      circular_mask
    -1         None                             False
     0         _circular_mask(shape)            True
     r > 0     _circular_mask(shape, r)         False

The last row is the C++'s own asymmetry: a positive radius masks the
histograms but not the projector (``include/idx/idx.hpp`` line 230).
Phase 6 exposes all three through one kikuchipy polarity
``signal_mask``, inverted once at the boundary
(``good_pixels = ~signal_mask``).

**Mapping to kikuchipy's own preprocessing** (no equivalence is
claimed where the maths differs):

.. code-block::

    EMSphInx                     kikuchipy
    ---------------------------  ------------------------------------
    (none)                       EBSD.remove_static_background
    gaussian_background=True     EBSD.remove_dynamic_background
                                 (same role, different maths: a
                                 Gaussian low pass, not a fit)
    _mosaic_ahe(p, n)            EBSD.adaptive_histogram_equalization
                                 (kernel_size=shape // n,
                                 clip_limit=0, nbins=256), equal to
                                 within 1.02 gray levels after a min
                                 max rescale
    _circular_mask(shape)        kp.filters.Window("circular", shape)
                                 for square shapes only
    _to_uint8                    **not** kp.pattern.rescale_intensity
                                 (dtype_out=uint8), which truncates
                                 with astype: [0, 1, 6] gives
                                 [0, 43, 255] here and [0, 42, 255]
                                 there

**Quirks reproduced faithfully**, each measured:

- The stopping rule of ``Model<Real>::fit()`` is
  ``metric = |(ssPrev - ss) / ss|``, and the fit returns when
  ``metric >= metricPrev and metric < 1e-4``.  An exact Gaussian
  with an integer mean gives ``ss == 0`` at every iteration, so
  ``metric`` is NaN, every comparison is false and the fit reports
  non-convergence -- and ``BckgSub2D`` falls back to a flat
  background.  A perfect background centred on a pixel is therefore
  *not* subtracted by ``IndexEBSD``.
- ``_fit_gaussian_1d_kernel`` needs ``error_model="numpy"``, the
  project's second sanctioned use: the flat input divides ``0 / 0``
  in ``estimate()`` too, and both NaNs must propagate to the
  fallback rather than raise.
- The comparisons of :func:`_cholesky_solve_3x3` are written in the
  **C++ direction** (``if signbit(a[i, i]) != neg`` and
  ``if pivot < eps``), never defensively, so a NaN never fails the
  decomposition.  The status a flat input ends on therefore depends
  on the platform's NaN sign propagation (measured: 3 compiled and 2
  interpreted on the development machine, both "not converged" for
  the caller) and is recorded, not asserted.
- The row and column maxima are initialised to ``pattern[0, 0]``
  **even when that pixel is masked out** (lines 259-260).
- The 1-D fits run on the interior ``[1 : w - 1]`` and
  ``[1 : h - 1]``, and the C++ then evaluates the background at the
  fitted mean **without adding the one pixel offset** back, so its
  surface is one pixel left of and above the true one.  That is
  ``emsphinx_compatible=True``; ``False`` adds the offset, which
  drops the error on a synthetic separable Gaussian from 8.16 to
  0.0028 of 180.
- The failed axis falls back to ``(a, b, c) = (w / 2, inf,
  mean(cWrk))``, whose factor is exactly one.
- A uniform image comes out of the equalisation as ``255``
  everywhere, an all-masked tile gets a flat histogram and the
  identity ramp ``(v + 1) 255 / 256``, and **masked out pixels are
  equalised too**: the mask only selects which pixels enter the
  histograms.
- ``_to_uint8`` of a flat buffer would be undefined behaviour in C++
  (``inf * 0``); it returns zeros here, the one documented
  deviation.
- The circle of ``BckgSub2D::CircMask`` is centred on the *integer*
  pixel ``(w // 2, h // 2)`` with radius ``min(w, h) // 2``, half a
  pixel away from the projector's physical circle.  Both are kept.

References
----------
:cite:`lenthe2019spherical`
"""

from numba import njit
import numpy as np

# Nothing is imported here from SciPy's fast Fourier transform
# package, deliberately: the discrete cosine image quality lives in
# ``_back_projection`` alone, which owns the phase's only binding.

# Number of histogram bins of the adaptive histogram equalisation,
# i.e. the number of values an unsigned 8-bit pattern can take
# (``include/util/ahe.hpp``, the ``HistBins`` of line 100)
_N_BINS = 256

# Half width of a mosaic tile in tile units, ``include/util/ahe.hpp``
# line 124: tiles touch but never overlap
_TILE_HALF_WIDTH = 0.5

# Iteration cap and relative stopping threshold of
# ``gaussian::Model<Real>::fit()`` (``include/util/gaussian.hpp``,
# lines 172 and 226)
_MAX_ITERATIONS = 50
_STOP_THRESHOLD = 1e-4

# Initial stopping metric of ``gaussian::Model<Real>::fit()``,
# ``std::numeric_limits<Real>::max()`` (line 176)
_REAL_MAX = float(np.finfo(np.float64).max)

# Machine epsilon of the 64-bit floating point type, the pivot
# threshold of ``decompose::cholesky()``
# (``include/util/linalg.hpp``, line 422)
_EPS = float(np.finfo(np.float64).eps)

# Size of the normal equations of the three parameter Gaussian fit
_N_PARAMETERS = 3

# --------------------------- Numba kernels -------------------------- #


@njit(cache=True, nogil=True)
def _row_col_max_kernel(
    pattern: np.ndarray,
    good: np.ndarray,
    row_max: np.ndarray,
    col_max: np.ndarray,
) -> None:
    """Fill the row and column maxima of the good pixels of a
    pattern.

    Parameters
    ----------
    pattern
        ``(h, w)`` 64-bit float pattern.
    good
        ``(h, w)`` boolean mask, ``True`` = use the pixel.
    row_max
        ``(h,)`` 64-bit float output, written in place.
    col_max
        ``(w,)`` 64-bit float output, written in place.

    Notes
    -----
    The maxima loop of ``gaussian::BckgSub2D<Real>::fit()``
    (``include/util/gaussian.hpp``, lines 259-271).  Both arrays are
    seeded with ``pattern[0, 0]`` **whether or not that pixel is
    masked out** (lines 259-260), which is faithful: a masked
    ``[0, 0]`` that happens to hold the image maximum still shows up
    in every row and column maximum.
    """
    height, width = pattern.shape
    seed = pattern[0, 0]
    for j in range(height):
        row_max[j] = seed
    for i in range(width):
        col_max[i] = seed
    for j in range(height):
        for i in range(width):
            if good[j, i]:
                value = pattern[j, i]
                if value > row_max[j]:
                    row_max[j] = value
                if value > col_max[i]:
                    col_max[i] = value


@njit(cache=True, nogil=True)
def _cholesky_solve_3x3(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> int:
    """Solve a 3x3 symmetric system by Cholesky decomposition and
    return a status.

    Parameters
    ----------
    a
        ``(3, 3)`` 64-bit float matrix, **overwritten** with the
        decomposition in its lower triangle, as the C++ does.
    b
        ``(3,)`` 64-bit float right hand side.
    x
        ``(3,)`` 64-bit float solution, written in place.  Undefined
        when the status is non-zero.

    Returns
    -------
    status
        ``0`` on success, ``1`` when a diagonal entry has a
        different sign from ``a[0, 0]`` and ``2`` when a pivot is
        smaller than the machine epsilon.

    Notes
    -----
    Port of ``solve::cholesky()`` (``include/util/linalg.hpp``,
    lines 354-358), i.e. ``decompose::cholesky()`` (lines 411-431)
    followed by ``backsolve::cholesky()`` (lines 487-493) with its
    ``neg`` negation: a negative definite matrix is negated for the
    decomposition and the solution is negated back.

    **The comparisons are written in the C++ direction**,
    ``if signbit(a[i, i]) != neg`` and ``if pivot < eps``, and never
    as the defensive ``if not (pivot >= eps)``.  Both C++ tests are
    false for NaN, so an all-NaN matrix decomposes "successfully"
    with a NaN solution (status ``0``) and the caller runs its full
    iteration budget, exactly as ``Model<Real>::fit()`` does before
    throwing "failed to converged".  Measured: an all-NaN matrix
    gives ``0``, one with a flipped NaN sign bit gives ``1``,
    ``diag(1e-17, 1, 1)`` gives ``2`` (where a ``< 0`` mutant would
    proceed) and a negative definite matrix gives ``0`` with
    ``A x == b``.
    """
    n = _N_PARAMETERS
    diagonal = np.zeros(n)
    neg = np.signbit(a[0, 0])
    for i in range(n):
        # the C++ direction (line 416): NaN is not unequal to itself
        # here, so an all NaN matrix passes this test
        if np.signbit(a[i, i]) != neg:
            return 1
        for j in range(i, n):
            if neg:
                total = -a[i, j]
            else:
                total = a[i, j]
            for k in range(i):
                total -= a[i, k] * a[j, k]
            if i == j:
                # the C++ direction (line 422): NaN is not < eps
                if total < _EPS:
                    return 2
                diagonal[i] = np.sqrt(total)
            else:
                a[j, i] = total / diagonal[i]
    for i in range(n):
        total = b[i]
        for k in range(i):
            total -= x[k] * a[i, k]
        x[i] = total / diagonal[i]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            x[i] -= a[j, i] * x[j]
        x[i] /= diagonal[i]
    if neg:
        for i in range(n):
            x[i] = -x[i]
    return 0


@njit(cache=True, nogil=True, error_model="numpy")
def _fit_gaussian_1d_kernel(y: np.ndarray, params: np.ndarray) -> int:
    """Fit ``c exp(-(i - a)^2 / b)`` to samples on the integers and
    return a status.

    Parameters
    ----------
    y
        ``(n,)`` 64-bit float samples, at abscissae ``0 .. n - 1``.
    params
        ``(4,)`` 64-bit float output ``(a, b, c, r2)``, written in
        place with the parameters of the **last** step taken, which
        precedes the stopping test.

    Returns
    -------
    status
        ``0`` converged, ``1`` fewer than three samples, ``2`` the
        Cholesky solve failed and ``3`` the iteration cap was
        reached without convergence.

    Notes
    -----
    Port of ``gaussian::Model<Real>::estimate()``
    (``include/util/gaussian.hpp``, lines 140-163) followed by
    ``Model<Real>::fit()`` (lines 172-231) with ``x == NULL``, so
    the abscissae are the indices.  The estimate is the log linear
    regression over the samples with ``y / c > 0``, ``c`` the
    maximum and ``a`` its position; the Gauss-Newton step uses the
    analytic Jacobian of lines 198-206 (``dfda = -2 fxb``,
    ``dfdb = fxb dxb``, ``dfdc = exp(-dx dxb)``) and
    :func:`_cholesky_solve_3x3` on the normal equations.

    The stopping rule is the C++'s verbatim,
    ``metric = |(ssPrev - ss) / ss|`` with a return when
    ``metric >= metricPrev and metric < 1e-4``.  Two divisions by an
    exact zero reach IEEE semantics through ``error_model="numpy"``:
    a flat input gives ``xy == y2 == +0.0`` in the estimate, and an
    exact Gaussian with an integer mean gives ``ss == 0`` at every
    iteration.  Both produce NaN, which propagates to a non-zero
    status and the caller's fallback, as in the C++.  The scalar
    accumulators are seeded from an array element (``y[0] * 0.0``)
    so that the ``py_func`` divides NumPy scalars and yields NaN
    under :func:`numpy.errstate` instead of raising
    ``ZeroDivisionError`` on Python floats.
    """
    n = y.shape[0]
    if n < 3:
        return 1

    # a NumPy scalar seed, so that the ``py_func`` divides NumPy
    # scalars and yields NaN for 0 / 0 under :func:`numpy.errstate`
    zero = y[0] * 0.0

    # ``Model<Real>::estimate()``: the maximum, its position and a
    # log linear regression of ``ln(y / c) == -(x - a)^2 / b``
    i_max = 0
    for i in range(1, n):
        if y[i] > y[i_max]:
            i_max = i
    c = y[i_max]
    a = zero + i_max
    xy = zero
    y2 = zero
    y_bar = zero
    for i in range(n):
        yc = y[i] / c
        y_bar += y[i]
        if yc > 0:
            dx = a - i
            yi = np.log(yc)
            y2 += yi * yi
            xy -= yi * dx * dx
    b = xy / y2
    y_bar = y_bar / n

    ss_tot = zero
    for i in range(n):
        dy = y[i] - y_bar
        ss_tot += dy * dy

    jtj = np.zeros((_N_PARAMETERS, _N_PARAMETERS))
    jtr = np.zeros(_N_PARAMETERS)
    step = np.zeros(_N_PARAMETERS)
    ss = zero
    ss_prev = zero
    metric_prev = zero + _REAL_MAX
    for _ in range(_MAX_ITERATIONS):
        ss = zero
        for i in range(_N_PARAMETERS):
            jtr[i] = 0.0
            for j in range(_N_PARAMETERS):
                jtj[i, j] = 0.0
        for i in range(n):
            dx = a - i
            dxb = dx / b
            dfdc = np.exp(-dx * dxb)
            fx = c * dfdc
            fxb = fx * dxb
            dfda = -fxb * 2
            dfdb = fxb * dxb
            ri = y[i] - fx
            jtj[0, 0] += dfda * dfda
            jtj[0, 1] += dfda * dfdb
            jtj[0, 2] += dfda * dfdc
            jtj[1, 1] += dfdb * dfdb
            jtj[1, 2] += dfdb * dfdc
            jtj[2, 2] += dfdc * dfdc
            jtr[0] += ri * dfda
            jtr[1] += ri * dfdb
            jtr[2] += ri * dfdc
            ss += ri * ri
        jtj[1, 0] = jtj[0, 1]
        jtj[2, 0] = jtj[0, 2]
        jtj[2, 1] = jtj[1, 2]
        if _cholesky_solve_3x3(jtj, jtr, step) != 0:
            params[0] = a
            params[1] = b
            params[2] = c
            params[3] = 1 - ss / ss_tot
            return 2
        a += step[0]
        b += step[1]
        c += step[2]
        metric = np.abs((ss_prev - ss) / ss)
        if metric >= metric_prev and metric < _STOP_THRESHOLD:
            params[0] = a
            params[1] = b
            params[2] = c
            params[3] = 1 - ss / ss_tot
            return 0
        metric_prev = metric
        ss_prev = ss
    params[0] = a
    params[1] = b
    params[2] = c
    params[3] = 1 - ss / ss_tot
    return 3


@njit(cache=True, nogil=True)
def _ahe_cdf_kernel(
    image: np.ndarray,
    good: np.ndarray,
    has_mask: bool,
    tiles: np.ndarray,
    cdfs: np.ndarray,
) -> None:
    """Fill the scaled cumulative histogram of every tile.

    Parameters
    ----------
    image
        ``(h, w)`` unsigned 8-bit image.
    good
        ``(h, w)`` boolean mask, ``True`` = count the pixel.  Read
        only when ``has_mask``.
    has_mask
        Whether ``good`` restricts the histograms.
    tiles
        ``(ny nx, 4)`` 64-bit integer tile bounds
        ``(i_start, i_end, j_start, j_end)`` of :func:`_ahe_tiles`.
    cdfs
        ``(ny nx, 256)`` 64-bit float output, written in place: the
        cumulative histogram of each tile scaled by
        ``255 / cdf[255]``.

    Notes
    -----
    Port of ``AdaptiveHistogramEqualizer<Real, uint8_t>::
    computeHist()`` (``include/util/ahe.hpp``, lines 193-222).  A
    tile with no good pixel gets the flat histogram of one count per
    bin (lines 211-213), which makes its equalisation the identity
    ramp ``(v + 1) 255 / 256`` instead of a division by zero.
    """
    n_tiles = tiles.shape[0]
    hist = np.zeros(_N_BINS, dtype=np.int64)
    for t in range(n_tiles):
        i_start = tiles[t, 0]
        i_end = tiles[t, 1]
        j_start = tiles[t, 2]
        j_end = tiles[t, 3]
        for k in range(_N_BINS):
            hist[k] = 0
        if has_mask:
            for j in range(j_start, j_end):
                for i in range(i_start, i_end):
                    if good[j, i]:
                        hist[image[j, i]] += 1
            largest = 0
            for k in range(_N_BINS):
                if hist[k] > largest:
                    largest = hist[k]
            if largest == 0:
                # no good pixel: a flat histogram, lines 211-213
                for k in range(_N_BINS):
                    hist[k] = 1
        else:
            for j in range(j_start, j_end):
                for i in range(i_start, i_end):
                    hist[image[j, i]] += 1
        total = 0
        for k in range(_N_BINS):
            total += hist[k]
            cdfs[t, k] = total
        # never a division by zero: ``_ahe_tiles`` bounds
        # ``n_regions`` by ``min(h, w)``, so every tile holds at
        # least one pixel, and an all masked out tile took the flat
        # histogram above
        norm = (_N_BINS - 1) / cdfs[t, _N_BINS - 1]
        for k in range(_N_BINS):
            cdfs[t, k] = norm * cdfs[t, k]


@njit(cache=True, nogil=True)
def _ahe_equalize_kernel(
    image: np.ndarray,
    cdfs: np.ndarray,
    nx: int,
    j_l: np.ndarray,
    j_u: np.ndarray,
    j_c: np.ndarray,
    j_f: np.ndarray,
    i_l: np.ndarray,
    i_u: np.ndarray,
    i_c: np.ndarray,
    i_f: np.ndarray,
    out: np.ndarray,
) -> None:
    """Equalise an image by bilinear interpolation between the four
    neighbouring tile histograms.

    Parameters
    ----------
    image
        ``(h, w)`` unsigned 8-bit image.
    cdfs
        ``(ny nx, 256)`` scaled cumulative histograms of
        :func:`_ahe_cdf_kernel`.
    nx
        Number of tiles along the columns, the row stride of
        ``cdfs``.
    j_l, j_u
        ``(h,)`` 64-bit integer lower and upper tile rows of every
        image row.
    j_c, j_f
        ``(h,)`` 64-bit float weights of those two rows, summing to
        one.
    i_l, i_u, i_c, i_f
        The same for the ``(w,)`` image columns.
    out
        ``(h, w)`` 64-bit float output, written in place, in
        ``[0, 255]``.

    Notes
    -----
    Port of the out of place ``AdaptiveHistogramEqualizer<Real,
    uint8_t>::equalize()`` (``include/util/ahe.hpp``, lines
    247-262), the four term sum
    ``cdf[jl, il] jc ic + cdf[jl, iu] jc if + cdf[ju, il] jf ic +
    cdf[ju, iu] jf if``.  **Every** pixel is equalised, masked out
    ones included: the mask only chooses which pixels the histograms
    count.
    """
    height, width = image.shape
    for j in range(height):
        row_lower = j_l[j] * nx
        row_upper = j_u[j] * nx
        close_j = j_c[j]
        far_j = j_f[j]
        for i in range(width):
            value = image[j, i]
            out[j, i] = (
                cdfs[row_lower + i_l[i], value] * close_j * i_c[i]
                + cdfs[row_lower + i_u[i], value] * close_j * i_f[i]
                + cdfs[row_upper + i_l[i], value] * far_j * i_c[i]
                + cdfs[row_upper + i_u[i], value] * far_j * i_f[i]
            )


@njit(cache=True, nogil=True)
def _to_uint8_kernel(buffer: np.ndarray, out: np.ndarray) -> None:
    """Rescale a flat buffer to the unsigned 8-bit range.

    Parameters
    ----------
    buffer
        Flat 64-bit float input.
    out
        Flat unsigned 8-bit output of the same size, written in
        place.

    Notes
    -----
    Port of the min max rescale of ``PatternProcessor<Real>::
    process()`` (``include/modality/ebsd/imprc.hpp``, lines 172-175
    and 182-185): ``factor = 255 / (max - min)`` and the cast
    ``(uint8_t)(factor (v - min) + 0.5)``, which truncates a
    non-negative value, i.e. rounds half away from zero.  kikuchipy's
    :func:`kikuchipy.pattern.rescale_intensity` truncates *without*
    the half, so the two differ by one level on many inputs.

    A flat buffer makes ``factor`` infinite and the C++ cast
    undefined; this kernel writes zeros instead, the module's one
    documented deviation from the C++.
    """
    n = buffer.size
    v_min = buffer[0]
    v_max = buffer[0]
    for i in range(1, n):
        value = buffer[i]
        if value < v_min:
            v_min = value
        if value > v_max:
            v_max = value
    if v_max == v_min:
        # ``255 / 0`` is infinite and the C++ cast undefined
        for i in range(n):
            out[i] = 0
        return
    factor = 255.0 / (v_max - v_min)
    for i in range(n):
        out[i] = np.uint8(factor * (buffer[i] - v_min) + 0.5)


# ------------------------- Circular mask ---------------------------- #


def _circular_mask(shape: tuple[int, int], radius: int | None = None) -> np.ndarray:
    """Return the integer circular mask of EMSphInx.

    Parameters
    ----------
    shape
        ``(h, w)`` of the mask.
    radius
        Radius in pixels.  Defaults to ``min(w, h) // 2``.

    Returns
    -------
    mask
        ``(h, w)`` boolean array which is ``True`` (good) where
        ``(i - w // 2)^2 + (j - h // 2)^2 <= radius^2``.

    Raises
    ------
    ValueError
        If ``radius`` is negative.

    Notes
    -----
    Port of ``gaussian::BckgSub2D<Real>::CircMask()``
    (``include/util/gaussian.hpp``, lines 238-251), whose arithmetic
    is entirely integer: the centre is the *pixel* ``(w // 2,
    h // 2)``, not the physical centre ``((w - 1) / 2, (h - 1) / 2)``
    of the projector's circle, and the two differ by half a pixel.

    Measured identical to
    ``kikuchipy.filters.Window("circular", shape).astype(bool)`` for
    square shapes (2819 good pixels of ``(60, 60)``, 2821 of
    ``(61, 61)``, 29 of ``(7, 7)``) and different for rectangular
    ones, where kikuchipy's radius is the larger half axis (1792
    here against 2531 there on ``(48, 60)``).

    This mask is in **good-pixel** polarity, the C++'s, which is the
    opposite of kikuchipy's ``signal_mask``.
    """
    height, width = int(shape[0]), int(shape[1])
    if radius is None:
        radius = min(width, height) // 2
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"`radius` must be non-negative, but is {radius}")
    rows, columns = np.indices((height, width))
    distance = (columns - width // 2) ** 2 + (rows - height // 2) ** 2
    return distance <= radius * radius


def _good_pixels_or_ones(
    good_pixels: np.ndarray | None, shape: tuple[int, int]
) -> np.ndarray:
    """Return a validated good-pixel mask, all ``True`` for ``None``.

    Parameters
    ----------
    good_pixels
        Optional ``(h, w)`` boolean mask in **good-pixel** polarity,
        or ``None``.
    shape
        ``(h, w)`` the mask must have.

    Returns
    -------
    good
        Contiguous ``(h, w)`` boolean array.

    Raises
    ------
    ValueError
        If ``good_pixels`` is given and is not a boolean array of
        shape ``shape``.
    """
    if good_pixels is None:
        return np.ones(shape, dtype=bool)
    good = np.asarray(good_pixels)
    if good.dtype != np.bool_:
        raise ValueError(
            f"`good_pixels` must be boolean, but has data type {good.dtype}"
        )
    if good.shape != tuple(shape):
        raise ValueError(
            f"`good_pixels` must have shape {tuple(shape)}, but has {good.shape}"
        )
    return np.ascontiguousarray(good)


# ------------------------ Gaussian background ----------------------- #


def _fit_gaussian_1d(
    y: np.ndarray, *, emsphinx_compatible: bool = True
) -> tuple[float, float, float, float, bool]:
    """Fit ``c exp(-(i - a)^2 / b)`` to samples on the integers.

    Parameters
    ----------
    y
        ``(n,)`` array-like of samples, cast to 64-bit float.
    emsphinx_compatible
        **Accepted and ignored**, kept only for symmetry with
        :func:`_gaussian_background`, which is where the flag acts
        (on the surface it evaluates, not on the fit).  ``True`` by
        default; the returned parameters are identical either way.

    Returns
    -------
    a, b, c
        Mean, squared width and amplitude of the fitted Gaussian.
    r2
        Coefficient of determination against the sample mean.
    converged
        Whether :func:`_fit_gaussian_1d_kernel` returned status
        ``0``.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or has fewer than three
        samples.

    Notes
    -----
    Wrapper of :func:`_fit_gaussian_1d_kernel`, whose status it
    turns into ``converged``.  A failed fit returns the parameters
    of the last step, which may be NaN or infinite; the caller is
    expected to use the flat fallback of
    :func:`_gaussian_background` instead.

    Measured on 58 samples: ``(a, b, c) = (25.3, 288, 150)`` and
    ``(40.7, 60.5, 90)`` are recovered exactly in 7 iterations with
    ``r2 == 1``; a ``0.5 sin(x)`` ripple moves the mean by 3e-5 and
    a ``3 sin(x / 2) + 5`` one by 0.012; an *integer* mean gives
    ``ss == 0``, a NaN metric and non-convergence.
    """
    samples = np.ascontiguousarray(y, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"`y` must be one-dimensional, but has shape {samples.shape}")
    if samples.size < _N_PARAMETERS:
        raise ValueError(
            f"`y` must hold at least {_N_PARAMETERS} samples for the "
            f"{_N_PARAMETERS} parameters, but holds {samples.size}"
        )
    parameters = np.zeros(4)
    status = _fit_gaussian_1d_kernel(samples, parameters)
    a, b, c, r2 = (float(value) for value in parameters)
    return a, b, c, r2, status == 0


def _gaussian_background(
    pattern: np.ndarray,
    good_pixels: np.ndarray | None = None,
    *,
    emsphinx_compatible: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return a separable Gaussian background fitted to the row and
    column maxima of a pattern.

    Parameters
    ----------
    pattern
        ``(h, w)`` array-like, cast to 64-bit float.
    good_pixels
        Optional ``(h, w)`` boolean mask in **good-pixel** polarity,
        ``True`` = use the pixel for the maxima.  All pixels are
        used when it is ``None``.
    emsphinx_compatible
        Whether to evaluate the background at the fitted mean
        without adding back the one pixel offset of the interior
        abscissa, ``True`` by default, which is what the C++ does.
        ``False`` adds it, on each axis whose fit converged.

    Returns
    -------
    background
        Fresh ``(h, w)`` 64-bit float outer product
        ``row[:, None] col[None, :]``.
    info
        Dictionary with ``"gx"`` and ``"gy"``, the ``(a, b, c)``
        triples of the column and row fits, ``"c"``, the shared
        amplitude ``max(gx.c, gy.c)``, and ``"converged_x"`` and
        ``"converged_y"``.

    Raises
    ------
    ValueError
        If ``pattern`` is not two-dimensional or is empty, or if
        ``good_pixels`` is given and is not a boolean array of the
        same shape.

    Notes
    -----
    Port of ``gaussian::BckgSub2D<Real>::fit()``
    (``include/util/gaussian.hpp``, lines 257-316).  The 1-D fits
    run on the **interior** ``cWrk[1 : w - 1]`` and
    ``rWrk[1 : h - 1]`` (lines 273 and 280), so the fitted mean is
    one pixel left of and above the true one; the C++ then evaluates
    the surface at that mean anyway (lines 308-315).  A failed axis
    falls back to ``(w / 2, inf, mean(cWrk))`` (lines 274-277 and
    281-284), whose exponential factor is exactly one.

    Measured on a separable Gaussian with
    ``(a_x, a_y, b_x, b_y, c) = (32.4, 27.9, 800, 648, 180)`` inside
    the ``(60, 60)`` circle: the fitted means are 31.400 and 26.900,
    exactly ``a - 1``, and the background error is 8.16 with
    ``emsphinx_compatible=True`` against 0.0028 with ``False``.  The
    row and column *maxima* over-estimate a background under a
    modulated signal, which is the estimator's nature and is
    EMSphInx'.
    """
    image = np.ascontiguousarray(pattern, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"`pattern` must be two-dimensional, but has shape {image.shape}"
        )
    # ``_row_col_max_kernel`` seeds from ``pattern[0, 0]`` and numba
    # compiles without bounds checking, so an empty axis would be an
    # out of bounds read rather than an exception
    if image.size == 0:
        raise ValueError(f"`pattern` must not be empty, but has shape {image.shape}")
    good = _good_pixels_or_ones(good_pixels, image.shape)
    height, width = image.shape

    row_max = np.empty(height, dtype=np.float64)
    col_max = np.empty(width, dtype=np.float64)
    _row_col_max_kernel(image, good, row_max, col_max)

    offset = 0.0 if emsphinx_compatible else 1.0
    parameters = np.zeros(4)
    converged_x = _fit_gaussian_1d_kernel(col_max[1 : width - 1], parameters) == 0
    if converged_x:
        gx = (float(parameters[0]), float(parameters[1]), float(parameters[2]))
        offset_x = offset
    else:
        gx = (width / 2, np.inf, float(col_max.mean()))
        offset_x = 0.0
    converged_y = _fit_gaussian_1d_kernel(row_max[1 : height - 1], parameters) == 0
    if converged_y:
        gy = (float(parameters[0]), float(parameters[1]), float(parameters[2]))
        offset_y = offset
    else:
        gy = (height / 2, np.inf, float(row_max.mean()))
        offset_y = 0.0

    amplitude = max(gx[2], gy[2])
    dx = gx[0] + offset_x - np.arange(width, dtype=np.float64)
    dy = gy[0] + offset_y - np.arange(height, dtype=np.float64)
    column_factor = np.exp(-(dx * dx / gx[1])) * amplitude
    row_factor = np.exp(-(dy * dy / gy[1]))
    background = row_factor[:, np.newaxis] * column_factor[np.newaxis, :]
    info = {
        "gx": gx,
        "gy": gy,
        "c": amplitude,
        "converged_x": converged_x,
        "converged_y": converged_y,
    }
    return background, info


def _remove_gaussian_background(
    pattern: np.ndarray,
    good_pixels: np.ndarray | None = None,
    *,
    emsphinx_compatible: bool = True,
) -> np.ndarray:
    """Return a pattern with its separable Gaussian background
    subtracted.

    Parameters
    ----------
    pattern, good_pixels, emsphinx_compatible
        As in :func:`_gaussian_background`, which this function
        calls to obtain the background.

    Returns
    -------
    subtracted
        Fresh ``(h, w)`` 64-bit float array holding
        ``pattern - background`` on the good pixels and **exactly
        zero** on the masked out ones.

    Raises
    ------
    ValueError
        If ``pattern`` is not two-dimensional or is empty, or if
        ``good_pixels`` is given and is not a boolean array of the
        same shape.

    Notes
    -----
    Port of the out of place ``gaussian::BckgSub2D<Real>::
    subtract()`` (``include/util/gaussian.hpp``, lines 379-386).
    The in place integer variants of lines 321-372 are not ported:
    only the in place ``process(uint8_t*)``, which the indexer never
    calls, uses them.
    """
    image = np.ascontiguousarray(pattern, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"`pattern` must be two-dimensional, but has shape {image.shape}"
        )
    good = _good_pixels_or_ones(good_pixels, image.shape)
    background, _ = _gaussian_background(
        image, good_pixels, emsphinx_compatible=emsphinx_compatible
    )
    return np.where(good, image - background, 0.0)


# ------------------ Mosaic adaptive histogram equalisation ---------- #


def _interpolation_pairs(
    midpoints: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the two neighbouring tiles and weights of every index.

    Parameters
    ----------
    midpoints
        ``(n_regions,)`` 64-bit integer tile midpoints, increasing.
    n
        Number of image rows or columns.

    Returns
    -------
    lower, upper
        ``(n,)`` 64-bit integer tiles bracketing every index.
    close, far
        ``(n,)`` 64-bit float weights of ``lower`` and ``upper``,
        summing to one.

    Notes
    -----
    The ``upper_bound`` search of ``AdaptiveHistogramEqualizer<Real,
    uint8_t>::setSize()`` (``include/util/ahe.hpp``, lines 150-166
    and 171-187), i.e. :func:`numpy.searchsorted` with
    ``side="right"``, with the constant ``0.5 / 0.5`` split before
    the first and after the last midpoint.
    """
    index = np.arange(n)
    bound = np.searchsorted(midpoints, index, side="right")
    lower = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=np.int64)
    close = np.empty(n, dtype=np.float64)
    far = np.empty(n, dtype=np.float64)

    outside = (bound == midpoints.size) | (bound == 0)
    lower[outside] = np.where(bound[outside] == 0, 0, midpoints.size - 1)
    upper[outside] = lower[outside]
    close[outside] = 0.5
    far[outside] = 0.5

    inside = ~outside
    bracket = bound[inside]
    lower[inside] = bracket - 1
    upper[inside] = bracket
    span = (midpoints[bracket] - midpoints[bracket - 1]).astype(np.float64)
    far[inside] = (index[inside] - midpoints[bracket - 1]) / span
    close[inside] = 1.0 - far[inside]
    return lower, upper, close, far


def _ahe_tiles(
    shape: tuple[int, int], n_regions: int
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return the tile bounds and the per-row and per-column
    interpolation pairs of the mosaic equaliser.

    Parameters
    ----------
    shape
        ``(h, w)`` of the image.
    n_regions
        Number of tiles along each axis, at least one and at most
        ``min(h, w)``.

    Returns
    -------
    tiles
        ``(n_regions^2, 4)`` 64-bit integer bounds
        ``(i_start, i_end, j_start, j_end)`` in row major tile
        order.
    j_pairs
        ``(j_l, j_u, j_c, j_f)`` of shape ``(h,)`` each: the lower
        and upper tile rows of every image row (64-bit integer) and
        their two weights (64-bit float).
    i_pairs
        The same four arrays of shape ``(w,)`` for the image
        columns.

    Raises
    ------
    ValueError
        If ``n_regions`` is smaller than one or larger than
        ``min(h, w)``.

    Notes
    -----
    Port of ``AdaptiveHistogramEqualizer<Real, uint8_t>::setSize()``
    (``include/util/ahe.hpp``, lines 117-190) with the mosaic half
    width ``0.5`` of line 124, so tiles touch but never overlap.
    Tile ``i`` spans ``[round(mid - tx / 2), round(mid + tx / 2))``
    clamped to the image, with ``mid = round(tx i + tx / 2)`` and
    ``std::round``, i.e. ``floor(x + 0.5)`` for a non-negative
    argument.

    The interpolation pairs come from ``upper_bound`` on the tile
    midpoints (lines 150-166 and 171-187), i.e.
    ``numpy.searchsorted(mids, index, side="right")``, with the
    constant ``0.5 / 0.5`` split before the first and after the last
    midpoint.  ``lower_bound`` would be wrong: for ``n_regions`` 10
    on 60 pixels the midpoints start at 3, and pixel 3 must
    interpolate between tiles 0 and 1 with ``(l, u, c, f) =
    (0, 1, 1.0, 0.0)``, not take the end branch.

    Measured for ``(60, 60)``: ``n_regions`` 10 gives tiles of
    exactly 6 pixels with midpoints ``3, 9, ..., 57``, and
    ``n_regions`` 7 gives ``[0, 9), [9, 17), [17, 26), [26, 34),
    [34, 43), [43, 51), [51, 60)`` with midpoints ``4, 13, 21, 30,
    39, 47, 56``.
    """
    height, width = int(shape[0]), int(shape[1])
    largest = min(height, width)
    if n_regions < 1 or n_regions > largest:
        raise ValueError(
            f"`n_regions` must be between 1 and {largest}, but is {n_regions}"
        )
    index = np.arange(n_regions, dtype=np.float64)
    tile_width = width / n_regions
    tile_height = height / n_regions

    mid_x = tile_width * index + tile_width / 2
    min_x = np.maximum(mid_x - tile_width * _TILE_HALF_WIDTH, 0.0)
    max_x = np.minimum(mid_x + tile_width * _TILE_HALF_WIDTH, float(width))
    mid_y = tile_height * index + tile_height / 2
    min_y = np.maximum(mid_y - tile_height * _TILE_HALF_WIDTH, 0.0)
    max_y = np.minimum(mid_y + tile_height * _TILE_HALF_WIDTH, float(height))
    # ``std::round`` of a non-negative argument
    min_x = np.floor(min_x + 0.5).astype(np.int64)
    max_x = np.floor(max_x + 0.5).astype(np.int64)
    mid_x = np.floor(mid_x + 0.5).astype(np.int64)
    min_y = np.floor(min_y + 0.5).astype(np.int64)
    max_y = np.floor(max_y + 0.5).astype(np.int64)
    mid_y = np.floor(mid_y + 0.5).astype(np.int64)

    tiles = np.empty((n_regions * n_regions, 4), dtype=np.int64)
    tiles[:, 0] = np.tile(min_x, n_regions)
    tiles[:, 1] = np.tile(max_x, n_regions)
    tiles[:, 2] = np.repeat(min_y, n_regions)
    tiles[:, 3] = np.repeat(max_y, n_regions)

    return (
        tiles,
        _interpolation_pairs(mid_y, height),
        _interpolation_pairs(mid_x, width),
    )


def _mosaic_ahe(
    pattern_uint8: np.ndarray,
    n_regions: int,
    good_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Return an unsigned 8-bit pattern equalised tile by tile.

    Parameters
    ----------
    pattern_uint8
        ``(h, w)`` array-like of unsigned 8-bit values.
    n_regions
        Number of tiles along each axis, at least one and at most
        ``min(h, w)``.
    good_pixels
        Optional ``(h, w)`` boolean mask in **good-pixel** polarity,
        ``True`` = count the pixel in its tile's histogram.  Masked
        out pixels are still equalised.

    Returns
    -------
    equalised
        Fresh ``(h, w)`` 64-bit float array in ``[0, 255]``.

    Raises
    ------
    ValueError
        If ``pattern_uint8`` is not a two-dimensional unsigned
        8-bit array, if ``n_regions`` is out of range or if
        ``good_pixels`` has the wrong shape or data type.

    Notes
    -----
    :func:`_ahe_tiles`, :func:`_ahe_cdf_kernel` and
    :func:`_ahe_equalize_kernel` together, i.e.
    ``AdaptiveHistogramEqualizer<Real, uint8_t>`` with 256 bins.

    A **uniform** image maps to 255 everywhere, since its cumulative
    histogram is a step at its own value; that is faithful and is
    asserted, so that nobody "fixes" it to the identity.  An
    all-masked tile equalises to the identity ramp
    ``(v + 1) 255 / 256``.

    Measured equal to ``255 skimage.exposure.equalize_adapthist(p,
    kernel_size=shape // n_regions, clip_limit=0, nbins=256)`` to
    within 6 gray levels (correlation 1.000000) whenever the tiles
    divide the shape, and to
    :meth:`~kikuchipy.signals.EBSD.adaptive_histogram_equalization`
    with the same arguments to within 1.02 levels after a min max
    rescale.  For tiles which do **not** divide the shape the two
    diverge (max 70 levels at ``n_regions`` 7 on 60 pixels), since
    scikit-image pads to a multiple of its kernel.
    """
    image = np.asarray(pattern_uint8)
    if image.ndim != 2:
        raise ValueError(
            f"`pattern_uint8` must be two-dimensional, but has shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"`pattern_uint8` must be unsigned 8-bit, but has data type {image.dtype}"
        )
    image = np.ascontiguousarray(image)
    good = _good_pixels_or_ones(good_pixels, image.shape)
    tiles, j_pairs, i_pairs = _ahe_tiles(image.shape, n_regions)

    cdfs = np.zeros((n_regions * n_regions, _N_BINS), dtype=np.float64)
    _ahe_cdf_kernel(image, good, good_pixels is not None, tiles, cdfs)
    equalised = np.zeros(image.shape, dtype=np.float64)
    _ahe_equalize_kernel(image, cdfs, n_regions, *j_pairs, *i_pairs, equalised)
    return equalised


def _to_uint8(buffer: np.ndarray) -> np.ndarray:
    """Return a buffer rescaled to the unsigned 8-bit range.

    Parameters
    ----------
    buffer
        Array-like of any shape, cast to 64-bit float.

    Returns
    -------
    rescaled
        Fresh unsigned 8-bit array of the same shape, holding
        ``floor(255 (v - min) / (max - min) + 0.5)``, or zeros when
        the buffer is flat.

    Raises
    ------
    ValueError
        If ``buffer`` is empty.

    Notes
    -----
    Wrapper of :func:`_to_uint8_kernel`.  This is **not**
    :func:`kikuchipy.pattern.rescale_intensity` with
    ``dtype_out=numpy.uint8``, which casts with ``astype`` and so
    truncates toward zero: ``[0, 1, 6]`` gives ``[0, 43, 255]`` here
    and ``[0, 42, 255]`` there.
    """
    values = np.ascontiguousarray(buffer, dtype=np.float64)
    flat = values.reshape(-1)
    # the kernel seeds from ``buffer[0]`` and numba compiles without
    # bounds checking, so an empty buffer would read out of bounds
    if flat.size == 0:
        raise ValueError(f"`buffer` must not be empty, but has shape {values.shape}")
    rescaled = np.empty(flat.size, dtype=np.uint8)
    _to_uint8_kernel(flat, rescaled)
    return rescaled.reshape(values.shape)


# --------------------------- The pipeline --------------------------- #


def _preprocess_pattern(
    pattern: np.ndarray,
    *,
    good_pixels: np.ndarray | None = None,
    gaussian_background: bool = False,
    n_regions: int = 10,
    emsphinx_compatible: bool = True,
) -> np.ndarray:
    """Return a pattern preprocessed the way ``IndexEBSD`` does.

    Parameters
    ----------
    pattern
        ``(h, w)`` array-like of any real data type.
    good_pixels
        Optional ``(h, w)`` boolean mask in **good-pixel** polarity,
        ``True`` = use the pixel.  ``None`` by default, which is
        ``IndexEBSD``'s ``circmask = -1``.
    gaussian_background
        Whether to fit and subtract the separable Gaussian
        background first, ``False`` by default.
    n_regions
        Number of mosaic tiles along each axis, ``10`` by default.
        ``0`` skips the equalisation; otherwise at most
        ``min(h, w)``.
    emsphinx_compatible
        Passed to :func:`_gaussian_background`, ``True`` by default.

    Returns
    -------
    processed
        Fresh ``(h, w)`` 64-bit float array.  The input is never
        modified.

    Raises
    ------
    ValueError
        If ``pattern`` is not two-dimensional, if ``n_regions`` is
        negative or larger than ``min(h, w)``, or if
        ``good_pixels`` has the wrong shape or data type.

    Notes
    -----
    Port of the out of place ``PatternProcessor<Real>::process()``
    (``include/modality/ebsd/imprc.hpp``, lines 166-191).  The order
    of the three branches is frozen, see the module documentation;
    in particular an unsigned 8-bit input goes into the equaliser
    **unrescaled** (lines 179-180) while any other data type is
    converted with :func:`_to_uint8` first, so the two paths differ
    unless the input already spans ``0`` to ``255``.

    Measured on the first ``nickel_ebsd_small`` pattern with a
    Python probe: equalisation only 3.3 ms to ``[0, 255]``,
    background and equalisation 4.3 ms, background only 1.2 ms to
    ``[-35.5, 18.7]``, neither the unsigned 8-bit values
    ``[26, 245]`` as 64-bit float.
    """
    image = np.asarray(pattern)
    if image.ndim != 2:
        raise ValueError(
            f"`pattern` must be two-dimensional, but has shape {image.shape}"
        )
    largest = min(image.shape)
    if n_regions < 0 or n_regions > largest:
        raise ValueError(
            f"`n_regions` must be between 0 and {largest}, but is {n_regions}"
        )
    _good_pixels_or_ones(good_pixels, image.shape)

    if gaussian_background:
        subtracted = _remove_gaussian_background(
            image, good_pixels, emsphinx_compatible=emsphinx_compatible
        )
        if n_regions > 0:
            return _mosaic_ahe(_to_uint8(subtracted), n_regions, good_pixels)
        return subtracted
    if n_regions > 0:
        if image.dtype == np.uint8:
            work = image
        else:
            work = _to_uint8(image)
        return _mosaic_ahe(work, n_regions, good_pixels)
    return image.astype(np.float64)
