# Copyright 2019-2024 The kikuchipy developers
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

# The following notice is reproduced from PyEBSDIndex
# (pyebsdindex/nlpar_cpu.py, US Naval Research Laboratory), from which
# the NLPAR kernels of this module are derived; the code is in the
# public domain under 17 U.S.C. 105, and the notice asks derivative
# works to state the changes made, their date and nature, and to
# acknowledge NRL as the original source.

# #####################################################################
# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 22 May 2024
# #####################################################################

# Changes by the kikuchipy developers, 2026-10-05:
# - Rewritten as Numba kernels over in-memory NumPy chunks with
#   cache=True, nogil=True and no parallel.
# - Per-axis search radius.
# - Search window clamped when an axis is shorter than the window.
# - Duplicate guard d2 > 0.
# - Pairs without comparable pixels weighted 0.
# - Saturation threshold from the global maximum.
# - Lambda objective excluding out-of-map neighbours and using the
#   averaging weight form.
# - diff_offset, backsub, stem_scale, the pair memo and all file I/O
#   removed.
# - Integer output rounded to nearest.
#
# The US Naval Research Laboratory (David Rowenhorst) is gratefully
# acknowledged as the original source of the NLPAR implementation.

"""Private kernels and drivers for non-local pattern averaging (NLPAR)
of EBSD patterns.

NLPAR after Brewick, Wright and Rowenhorst, Ultramicroscopy 200 (2019)
50-61. The Numba kernels are derived from PyEBSDIndex's
``NLPAR.sigma_numba`` and ``NLPAR.nlpar_nb`` (see the notice above);
PyEBSDIndex is never imported here. EMsoftOO's ``mod_NLPAR.f90``
(BSD-3-Clause) served as an independent cross-check of the equations
only; no code is ported from it.

Two passes over a map of patterns, both as Dask ``overlap`` +
``map_blocks`` over navigation-chunked arrays with core-only chunk
wrappers:

1. Sigma pass (halo depth 1): the noise level ``sigma`` of every
   pattern from its clipped 3 x 3 neighbourhood, plus the raw
   per-slot accumulators ``d2``, ``n2`` and the in-map mask ``valid``.
   It is always computed eagerly; the normalised 3 x 3 distances
   follow in NumPy on the assembled whole-map arrays.
2. Averaging pass (halo depth from :func:`_nlpar_depth`): normalised
   distances within the shifted-inward search window, weights, and
   the weighted sum, returned in the requested data type.

Numeric contract shared by every kernel: patterns enter as float32,
accumulators are float32 and updated in ascending pixel order, every
literal that meets a float32 operand is an explicit ``np.float32``,
squares are written as products (no power operator on a float32
operand, since Numba and NumPy promote it differently), and every
intermediate that PyEBSDIndex's compiled kernels hold in float64 is
written with explicit ``np.float64`` casts. The compiled kernels then
agree bitwise with their ``py_func`` (the weights kernel, the only one
calling ``exp``, within a measured float32 ulp tolerance) and with
PyEBSDIndex's compiled kernels.
"""

from __future__ import annotations

import logging
import numbers
import warnings

import dask.array as da
from dask.array.overlap import ensure_minimum_chunksize
from numba import njit
import numpy as np
from scipy.optimize import minimize
from skimage.util.dtype import dtype_range

_logger = logging.getLogger(__name__)

#: Fraction of the global data maximum at or above which a pixel is
#: excluded from the sigma estimate when saturation protection is on.
#: A Python float, since it multiplies the float64 maximum in
#: :func:`_nlpar_sigma_kernel`, as in PyEBSDIndex's compiled
#: ``sigma_numba``.
SIGMA_SATURATION_FACTOR = 0.9961

#: Fraction of the global data maximum at or above which a pixel is
#: excluded from the search-window distances when saturation protection
#: is on. A float32 scalar, since it multiplies the float32 maximum in
#: :func:`_nlpar_distances_kernel`, as in PyEBSDIndex's ``nlpar_nb``.
AVERAGE_SATURATION_FACTOR = np.float32(0.999)


# ------------------------------ Kernels ----------------------------- #


@njit(cache=True, nogil=True)
def _window_bounds(center: int, radius: int, n: int, shift: bool) -> tuple[int, int]:
    """Return the half-open bounds of a window along one map axis.

    Parameters
    ----------
    center
        Index of the window centre along the axis.
    radius
        Window radius along the axis, >= 0.
    n
        Length of the axis.
    shift
        Whether to shift the window inward at the map borders (the
        NLPAR search window) instead of clipping it (the 3 x 3 sigma
        window).

    Returns
    -------
    start
        First index of the window, in ``[0, n]``.
    stop
        Exclusive end index of the window, in ``[0, n]``.

    Notes
    -----
    With ``shift=False`` the window is clipped at the borders,
    ``[max(center - radius, 0), min(center + radius, n - 1) + 1)``,
    so it has ``radius + 1`` points at a corner.

    With ``shift=True`` the bounds are PyEBSDIndex's shifted-inward
    expressions
    ``start = max(center - radius, 0) - max(center + radius - (n - 1), 0)``
    and
    ``stop = min(center + radius, n - 1) + max(radius - center, 0) + 1``,
    each clamped to ``[0, n]``. The window then has ``2 radius + 1``
    points whenever ``n >= 2 radius + 1`` and is the whole axis
    ``(0, n)`` otherwise (``n=2, radius=3, center=0``: the raw
    ``(-2, 5)`` becomes ``(0, 2)``), so the window length is
    ``min(2 radius + 1, n)`` in every case. PyEBSDIndex does not clamp
    and indexes out of bounds on an axis shorter than the window.
    """
    if shift:
        start = max(center - radius, 0) - max(center + radius - (n - 1), 0)
        stop = min(center + radius, n - 1) + max(radius - center, 0) + 1
        # Clamp to the axis, which only an axis shorter than the window
        # needs
        start = min(max(start, 0), n)
        stop = min(max(stop, 0), n)
    else:
        start = max(center - radius, 0)
        stop = min(center + radius, n - 1) + 1
    return start, stop


@njit(cache=True, nogil=True)
def _nlpar_sigma_kernel(
    data: np.ndarray,
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the noise level of every pattern and the raw accumulators
    of its clipped 3 x 3 neighbourhood.

    Parameters
    ----------
    data
        Patterns of shape (n_rows, n_cols, n_pix) as float32.
    mask_indices
        Flat indices of the kept (unmasked) pixels among the ``n_pix``
        pixels, int64 in ascending order, of length ``n_kept``.
    max_value
        Global maximum of the map as float32, the reference of the
        saturation threshold.
    saturation_protect
        Whether a pixel pair is kept only if both values are below
        ``SIGMA_SATURATION_FACTOR`` times ``max_value``. If False, the
        threshold is ``max_value + 1`` in float64, which excludes no
        pair while ``max_value < 2**53``.

    Returns
    -------
    sigma
        Noise level of every pattern, float32 of shape
        (n_rows, n_cols).
    d2
        Sum of squared differences over the kept pixel pairs per
        neighbour slot, float32 of shape (n_rows, n_cols, 9).
    n2
        Number of kept pixel pairs per neighbour slot, float32 of
        shape (n_rows, n_cols, 9).
    valid
        Whether the slot's neighbour lies inside the map, bool of
        shape (n_rows, n_cols, 9).

    Notes
    -----
    Slot ``(dj + 1) * 3 + (di + 1)`` holds the neighbour at row offset
    ``dj`` and column offset ``di``, both in {-1, 0, 1}; slot 4 is the
    pattern itself. The window is clipped at the map borders
    (:func:`_window_bounds` with radius 1 and ``shift=False``), so a
    corner pattern has three neighbours and an edge pattern five;
    out-of-map slots have ``valid=False``.

    Per in-map neighbour, ``d2`` and ``n2`` are float32 accumulators
    starting at ``np.float32(0.0)`` and updated over ``mask_indices``
    in ascending order with ``diff = d0 - d1; d2 += diff * diff`` and
    ``n2 += np.float32(1.0)``, counting a pixel only if both values are
    strictly below the threshold. The threshold is
    ``np.float64(max_value) * np.float64(SIGMA_SATURATION_FACTOR)``
    with protection on and ``np.float64(max_value) + np.float64(1.0)``
    with it off, and the float32 pixels are compared against this
    float64 value, as PyEBSDIndex's compiled ``sigma_numba`` does.

    ``sigma_i^2`` is the minimum, over the neighbours with ``d2 > 0``,
    of ``d2 / (2 n2)``, in float32; ``sigma_i`` is its square root. If
    no neighbour has ``d2 > 0`` (a constant map, exact duplicates,
    every pair excluded), the minimum keeps its seed
    ``np.float32(1e24)`` and ``sigma_i`` is ``1e12``. The self slot
    has ``d2 = 0``, ``n2 = n_kept`` and ``valid=True``.

    Differences from ``sigma_numba``: ``n2`` starts at 0 instead of
    ``1e-12`` (the seed is rounded away for every ``n2 >= 1``, so sigma
    is unchanged); the duplicate guard is ``d2 > 0`` instead of
    ``d2 >= 1e-3`` (identical on integer data); neighbours are stored
    in a fixed nine-slot layout instead of compactly (row outer,
    column inner), so ``valid`` selects PyEBSDIndex's slots in the same
    order; its self slot holds the total pixel count instead of
    ``n_kept``; and the distances are normalised afterwards by
    :func:`_nlpar_normalized_distances`, not here.
    """
    n_rows, n_cols, _ = data.shape
    n_kept = mask_indices.size

    # A float64 threshold compared against float32 pixels
    if saturation_protect:
        threshold = np.float64(max_value) * np.float64(SIGMA_SATURATION_FACTOR)
    else:
        threshold = np.float64(max_value) + np.float64(1.0)

    sigma = np.empty((n_rows, n_cols), dtype=np.float32)
    d2_out = np.zeros((n_rows, n_cols, 9), dtype=np.float32)
    n2_out = np.zeros((n_rows, n_cols, 9), dtype=np.float32)
    valid = np.zeros((n_rows, n_cols, 9), dtype=np.bool_)
    n_kept_f32 = np.float32(n_kept)

    for j in range(n_rows):
        row_start, row_stop = _window_bounds(j, 1, n_rows, False)
        for i in range(n_cols):
            col_start, col_stop = _window_bounds(i, 1, n_cols, False)
            min_s2 = np.float32(1e24)
            for jn in range(row_start, row_stop):
                for i_n in range(col_start, col_stop):
                    slot = (jn - j + 1) * 3 + (i_n - i + 1)
                    valid[j, i, slot] = True
                    if jn == j and i_n == i:
                        n2_out[j, i, slot] = n_kept_f32
                        continue
                    d2 = np.float32(0.0)
                    n2 = np.float32(0.0)
                    for k in range(n_kept):
                        q = mask_indices[k]
                        d0 = data[j, i, q]
                        d1 = data[jn, i_n, q]
                        if d0 < threshold and d1 < threshold:
                            diff = d0 - d1
                            d2 += diff * diff
                            n2 += np.float32(1.0)
                    d2_out[j, i, slot] = d2
                    n2_out[j, i, slot] = n2
                    # Exact duplicates (d2 == 0) never set the minimum
                    if d2 > np.float32(0.0):
                        s2 = d2 / (np.float32(2.0) * n2)
                        if s2 < min_s2:
                            min_s2 = s2
            sigma[j, i] = np.sqrt(min_s2)

    return sigma, d2_out, n2_out, valid


@njit(cache=True, nogil=True)
def _nlpar_distances_kernel(
    data: np.ndarray,
    sigma2: np.ndarray,
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
    radius_rows: int,
    radius_cols: int,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> np.ndarray:
    """Return the normalised distances of every kept pattern to the
    patterns of its search window.

    Parameters
    ----------
    data
        Patterns of a block of shape (n_rows, n_cols, n_pix) as
        float32, including any halo.
    sigma2
        Squared noise level of every pattern of the block, float32 of
        shape (n_rows, n_cols), computed as ``sigma * sigma`` in
        float32 by the caller.
    mask_indices
        Flat indices of the kept (unmasked) pixels, int64 in ascending
        order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether a pixel pair is kept only if both values are below
        ``AVERAGE_SATURATION_FACTOR`` times ``max_value``. If False,
        the threshold is ``max_value + np.float32(1.0)`` in float32,
        which excludes no pair while ``max_value < 2**24``. From
        ``2**24`` on the sum rounds to ``max_value``, so the pixels
        equal to the maximum drop out of these distances, as in
        PyEBSDIndex, while the float64 threshold of
        :func:`_nlpar_sigma_kernel` still keeps them.
    radius_rows, radius_cols
        Search radius along the rows and the columns.
    row_start, row_stop, col_start, col_stop
        Half-open bounds, in block coordinates, of the kept region whose
        patterns are averaged (PyEBSDIndex's ``calclim``).

    Returns
    -------
    d
        Normalised distances, float32 of shape
        (row_stop - row_start, col_stop - col_start, n_slots) with
        ``n_slots = (2 radius_rows + 1) * (2 radius_cols + 1)``.

    Notes
    -----
    For a kept pattern (j, i) the window bounds per axis come from
    :func:`_window_bounds` with ``shift=True`` and the block length,
    and the neighbour at window-local row ``wr`` and column ``wc`` is
    stored in slot ``wr * (2 radius_cols + 1) + wc`` (row-major over the
    shifted window).

    Per pair, ``d2`` and ``n2`` are float32 accumulators over
    ``mask_indices`` exactly as in :func:`_nlpar_sigma_kernel`, with the
    float32 threshold ``max_value * AVERAGE_SATURATION_FACTOR``
    (protection on). With ``s0 = sigma2[j, i]`` and ``s1`` the
    neighbour's ``sigma2`` (never squared here),
    ``dnorm = (s1 + s0) * np.sqrt(np.float32(2.0) * n2)`` and
    ``d = (d2 - n2 * (s0 + s1)) / dnorm``, all in float32, when
    ``dnorm > np.float32(1e-8)``; ``d = np.float32(1e6) * n2`` when
    ``dnorm <= 1e-8`` and ``n2 > 0`` (PyEBSDIndex's branch); and
    ``d = +inf`` (weight 0) when ``n2 == 0``, where PyEBSDIndex gives
    ``1e6 * 0 = 0`` (weight 1). The self slot is ``-inf`` (weight 1).

    When an axis is shorter than ``2 radius + 1``, the window is the
    whole axis, the in-map neighbours occupy ``wr < len_r`` and
    ``wc < len_c`` with ``len_r``, ``len_c`` the in-map window
    lengths, and every other slot is ``+inf``.

    PyEBSDIndex's pair memo and ``diff_offset`` are not ported.
    """
    n_rows, n_cols, _ = data.shape
    n_kept = mask_indices.size
    width_rows = 2 * radius_rows + 1
    width_cols = 2 * radius_cols + 1

    # A float32 threshold compared against float32 pixels
    max32 = np.float32(max_value)
    if saturation_protect:
        threshold = max32 * AVERAGE_SATURATION_FACTOR
    else:
        threshold = max32 + np.float32(1.0)

    # Slots outside an axis shorter than the window keep +inf
    d = np.full(
        (row_stop - row_start, col_stop - col_start, width_rows * width_cols),
        np.inf,
        dtype=np.float32,
    )

    for j in range(row_start, row_stop):
        win_row_start, win_row_stop = _window_bounds(j, radius_rows, n_rows, True)
        for i in range(col_start, col_stop):
            win_col_start, win_col_stop = _window_bounds(i, radius_cols, n_cols, True)
            s0 = sigma2[j, i]
            for jn in range(win_row_start, win_row_stop):
                for i_n in range(win_col_start, win_col_stop):
                    slot = (jn - win_row_start) * width_cols + (i_n - win_col_start)
                    if jn == j and i_n == i:
                        # Weight exactly 1
                        d[j - row_start, i - col_start, slot] = -np.inf
                        continue
                    s1 = sigma2[jn, i_n]
                    d2 = np.float32(0.0)
                    n2 = np.float32(0.0)
                    for k in range(n_kept):
                        q = mask_indices[k]
                        d0 = data[j, i, q]
                        d1 = data[jn, i_n, q]
                        if d0 < threshold and d1 < threshold:
                            diff = d0 - d1
                            d2 += diff * diff
                            n2 += np.float32(1.0)
                    if n2 == np.float32(0.0):
                        # No comparable pixel: weight exactly 0
                        dist = np.float32(np.inf)
                    else:
                        d2 -= n2 * (s0 + s1)
                        dnorm = (s1 + s0) * np.sqrt(np.float32(2.0) * n2)
                        if dnorm > np.float32(1e-8):
                            dist = d2 / dnorm
                        else:
                            dist = np.float32(1e6) * n2
                    d[j - row_start, i - col_start, slot] = dist

    return d


@njit(cache=True, nogil=True)
def _nlpar_weights_kernel(d: np.ndarray, lam: float, dthresh: np.float32) -> np.ndarray:
    """Return the NLPAR weights of normalised distances.

    Parameters
    ----------
    d
        Normalised distances as float32, of shape
        (n_rows, n_cols, n_slots).
    lam
        Weight decay lambda as float64, > 0.
    dthresh
        Distance threshold as float32, >= 0.

    Returns
    -------
    w
        Weights, float32 of the shape of ``d``.

    Notes
    -----
    This is the only kernel calling ``exp``, with the operand types
    fixed to reproduce PyEBSDIndex's compiled ``nlpar_nb``:
    ``lam2 = np.float64(1.0) / (lam * lam)`` in float64,
    ``x = np.maximum(d - dthresh, np.float32(0.0))`` in float32, and
    ``w = exp(np.float64(-1.0) * np.float64(x) * lam2)`` in float64,
    stored as float32. Hence ``d = -inf`` (the self slot) gives a
    weight of exactly 1.0, ``d = +inf`` exactly 0.0, and any
    ``d <= dthresh`` exactly 1.0.

    The compiled kernel and its ``py_func`` may differ by a measured
    number of float32 ulps, since Numba's and NumPy's ``exp`` differ.
    """
    n_rows, n_cols, n_slots = d.shape
    lam64 = np.float64(lam)
    lam2 = np.float64(1.0) / (lam64 * lam64)
    dthresh32 = np.float32(dthresh)
    w = np.empty((n_rows, n_cols, n_slots), dtype=np.float32)
    for j in range(n_rows):
        for i in range(n_cols):
            for s in range(n_slots):
                x = np.maximum(d[j, i, s] - dthresh32, np.float32(0.0))
                e = np.exp(np.float64(-1.0) * np.float64(x) * lam2)
                w[j, i, s] = np.float32(e)
    return w


@njit(cache=True, nogil=True)
def _nlpar_weighted_sum_kernel(
    data: np.ndarray,
    weights: np.ndarray,
    radius_rows: int,
    radius_cols: int,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> np.ndarray:
    """Return the weighted average of every kept pattern over its
    search window.

    Parameters
    ----------
    data
        Patterns of a block of shape (n_rows, n_cols, n_pix) as
        float32, including any halo.
    weights
        Unnormalised weights from :func:`_nlpar_weights_kernel`, float32
        of shape (row_stop - row_start, col_stop - col_start, n_slots),
        in the slot layout of :func:`_nlpar_distances_kernel`.
    radius_rows, radius_cols
        Search radius along the rows and the columns.
    row_start, row_stop, col_start, col_stop
        Half-open bounds of the kept region in block coordinates.

    Returns
    -------
    out
        Averaged patterns, float32 of shape
        (row_stop - row_start, col_stop - col_start, n_pix).

    Notes
    -----
    Per kept pattern the weights are summed sequentially in float32 in
    slot order, starting from ``np.float32(0.0)``, every weight is
    divided by the sum, and ``out[q] += data[jn, i_n, q] * w`` is
    accumulated in float32 in slot order over ALL ``n_pix`` pixels: the
    signal mask affects the distances only, as in PyEBSDIndex. The
    window bounds are recomputed with :func:`_window_bounds` and
    ``shift=True``; slots outside an axis shorter than the window carry
    weight 0 and contribute nothing.
    """
    n_rows, n_cols, n_pix = data.shape
    n_slots = weights.shape[2]
    width_cols = 2 * radius_cols + 1
    out = np.zeros(
        (row_stop - row_start, col_stop - col_start, n_pix), dtype=np.float32
    )

    for j in range(row_start, row_stop):
        win_row_start, win_row_stop = _window_bounds(j, radius_rows, n_rows, True)
        jj = j - row_start
        for i in range(col_start, col_stop):
            win_col_start, win_col_stop = _window_bounds(i, radius_cols, n_cols, True)
            ii = i - col_start
            total = np.float32(0.0)
            for s in range(n_slots):
                total += weights[jj, ii, s]
            for jn in range(win_row_start, win_row_stop):
                for i_n in range(win_col_start, win_col_stop):
                    slot = (jn - win_row_start) * width_cols + (i_n - win_col_start)
                    w = weights[jj, ii, slot] / total
                    for q in range(n_pix):
                        out[jj, ii, q] += data[jn, i_n, q] * w

    return out


# ---------------------- Sigma pass normalisation -------------------- #


def _nlpar_normalized_distances(
    d2: np.ndarray, n2: np.ndarray, valid: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Return the normalised distances of every pattern to its clipped
    3 x 3 neighbours, from the raw accumulators of the sigma pass.

    Parameters
    ----------
    d2, n2
        Sums of squared differences and kept-pair counts per slot,
        float32 of shape (n_rows, n_cols, 9), from
        :func:`_nlpar_sigma_kernel` on the whole map.
    valid
        Whether a slot lies inside the map, bool of shape
        (n_rows, n_cols, 9).
    sigma
        Noise level of every pattern, float32 of shape
        (n_rows, n_cols), from the sigma pass or given by the user;
        every element finite and > 0, with ``sigma * sigma > 0`` and
        ``2 n_kept sigma^2`` finite in float32, as the method checks
        for a given sigma.

    Returns
    -------
    d
        Normalised distances, float32 of shape (n_rows, n_cols, 9), in
        the slot layout of :func:`_nlpar_sigma_kernel`.

    Notes
    -----
    A NumPy function, not a Numba kernel. It reproduces the second loop
    of PyEBSDIndex's compiled ``sigma_numba`` operation for operation:
    ``s2 = sigma * sigma`` (float32), ``s2_ij = s2_i + s2_j`` with the
    neighbour's ``s2_j`` gathered by padded shifts (float32),
    ``num = d2 - n2 * s2_ij`` (float32),
    ``den = np.float64(s2_ij) * np.sqrt(np.float64(2.0) * np.float64(n2))``
    (float64, as Numba types ``2.0 * n2`` there) and
    ``d = (np.float64(num) / den).astype(np.float32)``.

    Conventions: the self slot is ``-inf`` (weight exactly 1); a slot
    with ``n2 == 0`` is ``+inf`` (weight 0), where PyEBSDIndex keeps a
    finite tiny negative value. For a sigma as stated above ``s2_ij``
    and ``n2 * s2_ij`` are finite and ``den > 0`` wherever ``n2 > 0``,
    so the result is NaN-free. A sigma outside that range gives NaN
    here or in :func:`_nlpar_distances_kernel`: 0 / 0 at a duplicate
    neighbour when the square underflows to 0, ``-inf / inf`` when a
    sum or a product of squares overflows. Out-of-map slots keep
    ``valid=False`` and their value is never read.
    """
    d2 = np.asarray(d2, dtype=np.float32)
    n2 = np.asarray(n2, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    n_rows, n_cols = sigma.shape

    s2 = sigma * sigma

    # Squared sigma of the neighbour of every slot, zero outside the
    # map: slot (dj + 1) * 3 + (di + 1) reads the padded map at
    # (j + dj + 1, i + di + 1)
    s2_padded = np.pad(s2, 1)
    s2_neighbour = np.empty((n_rows, n_cols, 9), dtype=np.float32)
    for slot in range(9):
        row_shift, col_shift = divmod(slot, 3)
        s2_neighbour[..., slot] = s2_padded[
            row_shift : row_shift + n_rows, col_shift : col_shift + n_cols
        ]
    s2_ij = s2[..., None] + s2_neighbour

    num = d2 - n2 * s2_ij

    # Out-of-map slots and pairs without a kept pixel (n2 == 0) keep
    # +inf, so no division by zero is evaluated
    d = np.full(d2.shape, np.inf, dtype=np.float32)
    has_pairs = np.asarray(valid, dtype=bool) & (n2 > 0)
    den = s2_ij[has_pairs].astype(np.float64) * np.sqrt(
        np.float64(2.0) * n2[has_pairs].astype(np.float64)
    )
    d[has_pairs] = (num[has_pairs].astype(np.float64) / den).astype(np.float32)
    d[..., 4] = -np.inf

    return d


# ------------------------ Lambda optimisation ----------------------- #


def _nlpar_lambda_objective(
    lam: np.ndarray,
    d: np.ndarray,
    valid: np.ndarray,
    dthresh: float,
    target_weight: float,
) -> float:
    """Return the lambda objective, the mean over scan points of the
    absolute difference between the target weight and the normalised
    weight every pattern gives itself in its clipped 3 x 3
    neighbourhood.

    Parameters
    ----------
    lam
        Smoothing parameter lambda, float64 of shape (1,), as passed by
        :func:`scipy.optimize.minimize`. It is used as an array and
        never cast to a Python float.
    d
        Normalised distances, float32 of shape (n_rows, n_cols, 9),
        from :func:`_nlpar_normalized_distances` (self slot ``-inf``, a
        slot without comparable pixels ``+inf``).
    valid
        Whether a slot lies inside the map, bool of shape
        (n_rows, n_cols, 9).
    dthresh
        Distance threshold subtracted from every distance before the
        weight is computed, as in the averaging weights.
    target_weight
        Normalised weight a pattern should keep for itself, in (0, 1).

    Returns
    -------
    objective
        ``mean_i |target_weight - 1 / S_i|`` as a Python float.

    Notes
    -----
    A NumPy function, not a Numba kernel, driven by
    :func:`scipy.optimize.minimize`. The operations and their order
    are fixed: ``w = np.exp(-np.maximum(d - dthresh, np.float32(0.0))
    / lam ** 2)`` (float64 by promotion of the float32 ``d`` with the
    float64 ``lam``), ``w[~valid] = 0.0``, ``S = w.sum(axis=-1)`` over
    all nine slots in slot order (not ``1 +`` the sum of the non-self
    slots; the two are not bitwise equal) and
    ``float(np.mean(np.abs(target_weight - 1.0 / S)))``. The self slot
    has weight exactly 1 through ``d = -inf``, so ``S >= 1`` and no
    guard term is added to it.

    Two deviations from PyEBSDIndex's ``loptfunc``: out-of-map slots
    are excluded from ``S`` instead of counting with weight 1, and the
    distance enters as ``max(d - dthresh, 0)``, the averaging kernel's
    form, instead of ``max(d, dthresh)``. The two forms coincide at
    ``dthresh = 0``. The statistic over points is the mean.
    """
    w = np.exp(-np.maximum(d - dthresh, np.float32(0.0)) / lam**2)
    w[~valid] = 0.0
    s = w.sum(axis=-1)
    return float(np.mean(np.abs(target_weight - 1.0 / s)))


def _nlpar_optimize_lambda(
    d: np.ndarray,
    valid: np.ndarray,
    target_weight: float,
    dthresh: float,
) -> float:
    """Return the lambda minimising :func:`_nlpar_lambda_objective`.

    Parameters
    ----------
    d
        Normalised distances, float32 of shape (n_rows, n_cols, 9),
        from :func:`_nlpar_normalized_distances`.
    valid
        Whether a slot lies inside the map, bool of shape
        (n_rows, n_cols, 9).
    target_weight
        Normalised weight a pattern should keep for itself, in (0, 1).
    dthresh
        Distance threshold, as in the averaging weights.

    Returns
    -------
    lam
        Optimised lambda as a Python float, inside ``[1e-3, 10]``.

    Warns
    -----
    UserWarning
        If the optimised lambda lies within 1 % of a bound, i.e.
        ``lam <= 1.01e-3`` ("lower") or ``lam >= 9.9`` ("upper"), with
        the message "NLPAR lambda optimisation hit the {which} bound
        ({lam:.4f}); the target weight {target_weight} is not supported
        by the data".

    Notes
    -----
    When ``d.shape[0] * d.shape[1] >= 1e6`` the map is strided as
    ``d[::2, ::2]`` and ``valid[::2, ::2]`` before the objective is
    called, as PyEBSDIndex's ``opt_lambda_cpu`` does. The optimiser is
    ``scipy.optimize.minimize(_nlpar_lambda_objective,
    x0=np.array([1.0]), args=(d, valid, dthresh, target_weight),
    method="Nelder-Mead", bounds=[(1e-3, 10.0)],
    options={"fatol": 1e-4})``, PyEBSDIndex's call, for one target
    weight (PyEBSDIndex's median of the fits to three targets is the
    fit to the middle one, since lambda decreases monotonically with
    the target weight). A map whose weights barely depend on lambda
    leaves the optimiser at its start ``1.0`` without a warning.

    Exactly one INFO record "NLPAR: optimised lambda {lam:.4f} for
    target weight {target_weight} (objective {F:.2e})" is emitted
    through the module logger per call.
    """
    if d.shape[0] * d.shape[1] >= 1e6:
        d = d[::2, ::2]
        valid = valid[::2, ::2]

    result = minimize(
        _nlpar_lambda_objective,
        x0=np.array([1.0]),
        args=(d, valid, dthresh, target_weight),
        method="Nelder-Mead",
        bounds=[(1e-3, 10.0)],
        options={"fatol": 1e-4},
    )
    lam = float(result.x[0])
    objective = float(result.fun)

    which = None
    if lam <= 1.01e-3:
        which = "lower"
    elif lam >= 9.9:
        which = "upper"
    if which is not None:
        warnings.warn(
            f"NLPAR lambda optimisation hit the {which} bound ({lam:.4f}); the "
            f"target weight {target_weight} is not supported by the data",
            UserWarning,
        )

    _logger.info(
        f"NLPAR: optimised lambda {lam:.4f} for target weight {target_weight} "
        f"(objective {objective:.2e})"
    )
    return lam


# -------------------------- Driver helpers -------------------------- #


def _nlpar_mask_indices(
    signal_mask: np.ndarray | None, signal_shape: tuple[int, int]
) -> np.ndarray:
    """Return the flat indices of the pixels that enter the distances.

    Parameters
    ----------
    signal_mask
        Mask of the signal shape (row, column) with ``True`` for a pixel
        to EXCLUDE (kikuchipy's convention; PyEBSDIndex's ``mask`` is
        the opposite). Integer 0/1 masks are accepted. If None, every
        pixel is kept.
    signal_shape
        Signal shape (row, column).

    Returns
    -------
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.

    Raises
    ------
    ValueError
        If the mask does not have the signal shape ("signal shape"),
        checked before the cast with
        ``np.asarray(signal_mask, dtype=bool)``, or if it excludes
        every pixel ("excludes every pixel").
    """
    signal_shape = tuple(int(i) for i in signal_shape)
    if signal_mask is None:
        n_pix = int(np.prod(signal_shape))
        return np.arange(n_pix, dtype=np.int64)

    mask_shape = tuple(np.shape(signal_mask))
    if mask_shape != signal_shape:
        raise ValueError(
            f"signal_mask of shape {mask_shape} must have the signal shape "
            f"{signal_shape}"
        )
    mask = np.asarray(signal_mask, dtype=bool)

    # In ascending order, as the kernels accumulate over them
    mask_indices = np.flatnonzero(~mask.ravel()).astype(np.int64)
    if mask_indices.size == 0:
        raise ValueError(
            "signal_mask excludes every pixel (True means excluded), so no "
            "pixel is left to compare patterns with"
        )
    return mask_indices


def _nlpar_search_radius(
    search_radius: int | tuple[int, ...], nav_dim: int
) -> tuple[int, ...]:
    """Return one validated search radius per navigation axis.

    Parameters
    ----------
    search_radius
        One radius for every navigation axis, or a tuple with one
        radius per navigation axis in (row, column) order.
    nav_dim
        Navigation dimension, >= 1 (the caller rejects 0 first).

    Returns
    -------
    radius
        One int per navigation axis.

    Raises
    ------
    ValueError
        If a tuple's length differs from ``nav_dim`` ("one radius per
        navigation axis"), or if a radius is not a non-negative integer
        ("search_radius must be a non-negative int"). A radius is valid
        if ``isinstance(r, (int, np.integer))`` and
        ``not isinstance(r, bool)`` and ``r >= 0``.
    """
    if isinstance(search_radius, tuple):
        if len(search_radius) != nav_dim:
            raise ValueError(
                f"search_radius {search_radius} must have one radius per navigation "
                f"axis ({nav_dim} axes)"
            )
        radius = search_radius
    else:
        radius = (search_radius,) * nav_dim

    for r in radius:
        is_int = isinstance(r, (int, np.integer)) and not isinstance(r, bool)
        if not is_int or r < 0:
            raise ValueError(
                f"search_radius must be a non-negative int per navigation axis, "
                f"got {search_radius!r}"
            )
    return tuple(int(r) for r in radius)


def _nlpar_check_dthresh(dthresh: float) -> np.float32:
    """Return a validated distance threshold as float32.

    Parameters
    ----------
    dthresh
        Distance threshold, a real number (not a bool), >= 0 and finite
        in float32.

    Returns
    -------
    dthresh
        The threshold as float32.

    Raises
    ------
    ValueError
        If the threshold is invalid ("dthresh must be >= 0 and finite").
    """
    if (
        isinstance(dthresh, (bool, np.bool_))
        or not isinstance(dthresh, numbers.Real)
        or not 0 <= dthresh <= np.finfo(np.float32).max
    ):
        raise ValueError(f"dthresh must be >= 0 and finite, got {dthresh!r}")
    return np.float32(dthresh)


def _nlpar_check_target_weight(target_weight: float) -> float:
    """Return a validated target weight unchanged.

    Parameters
    ----------
    target_weight
        Target weight, a real number (not a bool) in (0, 1).

    Returns
    -------
    target_weight
        The target weight as given.

    Raises
    ------
    ValueError
        If the target weight is invalid ("0 < target_weight < 1").
    """
    if (
        isinstance(target_weight, (bool, np.bool_))
        or not isinstance(target_weight, numbers.Real)
        or not 0 < target_weight < 1
    ):
        raise ValueError(
            f"target_weight must satisfy 0 < target_weight < 1, got {target_weight!r}"
        )
    return target_weight


def _nlpar_check_sigma(
    sigma: float | np.ndarray | None, nav_shape: tuple[int, ...]
) -> np.ndarray | None:
    """Return a given noise level as a float32 map of the navigation
    shape.

    Parameters
    ----------
    sigma
        A real number (not a bool), a 0-d array (taken as a scalar), an
        array of the navigation shape, or None.
    nav_shape
        Navigation shape (row, column), or ``(n,)`` for a 1-D scan.

    Returns
    -------
    sigma
        Float32 map of the navigation shape, or None if not given.

    Raises
    ------
    ValueError
        If a scalar is not finite and > 0 in float32, or an array does
        not have the navigation shape exactly ("navigation shape") or
        has an element that is not finite and > 0 after the cast to
        float32 ("sigma must be > 0").
    """
    if sigma is None:
        return None
    if isinstance(sigma, np.ndarray) and sigma.ndim == 0:
        # A 0-d array is a scalar
        sigma = sigma.item()
    if isinstance(sigma, (bool, np.bool_)):
        raise ValueError(f"sigma must be > 0 and finite, got {sigma!r}")
    if isinstance(sigma, numbers.Real):
        with np.errstate(over="ignore"):
            sigma_value = np.float32(sigma)
        if not (np.isfinite(sigma_value) and sigma_value > 0):
            raise ValueError(f"sigma must be > 0 and finite, got {sigma!r}")
        return np.full(nav_shape, sigma_value, dtype=np.float32)

    # The navigation shape exactly, (n,) for a 1D scan: an array of
    # another shape is never reshaped
    sigma_shape = tuple(np.shape(sigma))
    if sigma_shape != tuple(nav_shape):
        raise ValueError(
            f"sigma of shape {sigma_shape} must be a scalar or have the navigation "
            f"shape {tuple(nav_shape)}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        sigma = np.array(sigma, dtype=np.float32)
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError(
            "sigma must be > 0 and finite in every element (after the cast to float32)"
        )
    return sigma


def _nlpar_check_sigma_range(sigma: np.ndarray, n_kept: int) -> None:
    """Check that a given noise level keeps the float32 distances
    finite.

    Parameters
    ----------
    sigma
        Float32 map from :func:`_nlpar_check_sigma`.
    n_kept
        Number of pixels not excluded by the signal mask.

    Raises
    ------
    ValueError
        Unless ``sigma * sigma > 0`` and ``2 n_kept sigma^2`` is finite
        in float32 in every element ("sigma must be > 0 with sigma").
    """
    # The distances scale n sigma^2 (n kept pixels at most) and divide
    # by sigma^2 in float32: a square that underflows to 0 or a product
    # that overflows gives NaN patterns
    with np.errstate(over="ignore", under="ignore"):
        sigma2 = sigma * sigma
        bound = np.float32(2 * n_kept) * sigma2
    if not (np.all(sigma2 > 0) and np.all(np.isfinite(bound))):
        raise ValueError(
            "sigma must be > 0 with sigma^2 > 0 and 2 n sigma^2 finite in float32 in "
            f"every element, n = {n_kept} being the number of pixels not excluded by "
            "signal_mask"
        )


def _nlpar_saturation_max(dask_array: da.Array | np.ndarray) -> np.float32:
    """Return the global maximum of a map of patterns.

    Parameters
    ----------
    dask_array
        Map of patterns of any real data type.

    Returns
    -------
    max_value
        Maximum over the whole map as float32, computed once (one Dask
        reduction for a Dask array).

    Notes
    -----
    The saturation thresholds of both passes are fractions of this one
    value, so the result does not depend on the chunking. PyEBSDIndex
    takes the maximum per tile instead.
    """
    max_value = dask_array.max()
    if isinstance(max_value, da.Array):
        max_value = max_value.compute()
    # Rounding to float32 is monotonic, so this is also the maximum of
    # the map cast to float32
    return np.float32(max_value)


def _nlpar_finalize(
    out_f32: np.ndarray, dtype_out: str | np.dtype | type
) -> np.ndarray:
    """Return averaged float32 patterns in the output data type.

    Parameters
    ----------
    out_f32
        Averaged patterns as float32.
    dtype_out
        Integer or floating output data type (validated by the caller).

    Returns
    -------
    out
        Patterns in ``dtype_out``.

    Notes
    -----
    Integer types: ``np.rint``, then a clip to
    ``dtype_range[np.dtype(dtype_out).type]`` from
    :mod:`skimage.util.dtype`, then the cast. Floating types: a cast.
    Never a per-pattern rescale: the weights sum to 1, so the average is
    a convex combination inside the input range.

    The rounding and the clip run in float64, where the float32 values
    and their ``rint`` are exact, so 8- and 16-bit outputs are those of
    a float32 route bitwise. In float32 the upper bound of a 32-bit
    type is not representable (``2**32 - 1`` rounds up to ``2**32``),
    and the cast of the clipped value would wrap around. The lower
    bound, 0 or a negative power of two, is exact in float64; the upper
    bound of a 64-bit type is not, and is replaced by the largest
    float64 below it.
    """
    dtype_out = np.dtype(dtype_out)
    if np.issubdtype(dtype_out, np.integer):
        omin, omax = dtype_range[dtype_out.type]
        # Python compares an int with a float exactly
        high = float(omax)
        if high > omax:
            high = float(np.nextafter(high, -np.inf))
        out = np.clip(np.rint(out_f32.astype(np.float64)), float(omin), high)
        return out.astype(dtype_out)
    return out_f32.astype(dtype_out)


def _nlpar_depth(
    chunks: tuple[tuple[int, ...], ...], radius: tuple[int, ...]
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    """Return the halo depth and the chunks of the averaging pass.

    Parameters
    ----------
    chunks
        Chunks of the whole array, navigation axes first and the signal
        axes after them.
    radius
        Search radius per navigation axis.

    Returns
    -------
    depth
        Halo depth per array axis, keyed by axis, 0 on the signal axes
        (the form :func:`dask.array.overlap.overlap` takes).
    chunks_out
        Chunks of the whole array after the rechunk this depth needs.

    Notes
    -----
    Per navigation axis with radius ``r``, chunks ``c`` and length
    ``n``: one chunk gives depth 0; ``2 r + 1 >= n`` rechunks the axis
    to one chunk with depth 0; otherwise
    ``depth = max(r, 2 r + 1 - min(c[0], c[-1]))``, since a pattern
    closer than ``r`` to a true map border has its shifted window
    reaching ``2 r`` points into the map, so the edge chunk's core plus
    its halo must reach that far. The axis chunks are then
    ``dask.array.overlap.ensure_minimum_chunksize(depth, c)``, so no
    chunk is thinner than its depth and interior chunks never hold a
    shifted pattern. Examples: (26, 26, 3) at ``r = 3`` gives depth 4
    and chunks (26, 25, 4); at ``r = 4`` depth 6 and (26, 23, 6);
    (47, 8) at ``r = 4`` depth 4 and unchanged chunks.

    The generic overlap depth of ``average_neighbour_patterns`` is not
    used, since it is correct only for windows that never shift. The
    sigma pass needs depth 1 on every chunked navigation axis and no
    rechunk.
    """
    nav_dim = len(radius)
    depth = {}
    chunks_out = []
    for axis, (c, r) in enumerate(zip(chunks[:nav_dim], radius)):
        c = tuple(int(i) for i in c)
        r = int(r)
        n = sum(c)
        if len(c) == 1:
            depth[axis] = 0
            chunks_out.append(c)
        elif 2 * r + 1 >= n:
            # Every window is the whole axis
            depth[axis] = 0
            chunks_out.append((n,))
        else:
            # The shifted window of a pattern closer than r to a map
            # border spans 2 r + 1 points from that border
            depth_axis = max(r, 2 * r + 1 - min(c[0], c[-1]))
            depth[axis] = depth_axis
            chunks_out.append(tuple(ensure_minimum_chunksize(depth_axis, c)))
    for axis in range(nav_dim, len(chunks)):
        depth[axis] = 0
        chunks_out.append(tuple(int(i) for i in chunks[axis]))
    return depth, tuple(chunks_out)


# -------------------------- Chunk wrappers -------------------------- #


def _nlpar_core_bounds(
    nav_shape: tuple[int, int], depth: tuple[int, int], block_info: dict
) -> tuple[int, int, int, int]:
    """Return the kept region of a haloed block in block coordinates.

    Parameters
    ----------
    nav_shape
        Navigation shape of the haloed block (n_rows, n_cols).
    depth
        Halo depth of the two navigation axes.
    block_info
        Dask's block information of the block.

    Returns
    -------
    row_start, row_stop, col_start, col_stop
        Half-open bounds of the block minus its halos (PyEBSDIndex's
        ``calclim``).

    Notes
    -----
    A block carries a halo of ``depth`` on each side facing another
    chunk, read from ``block_info[0]["chunk-location"]`` against
    ``block_info[0]["num-chunks"]``; the first and the last chunk along
    an axis carry none on their outer side, the map border.
    """
    location = block_info[0]["chunk-location"]
    num_chunks = block_info[0]["num-chunks"]
    bounds = []
    for axis in range(2):
        halo = int(depth[axis])
        before = halo if location[axis] > 0 else 0
        after = halo if location[axis] < num_chunks[axis] - 1 else 0
        bounds.extend([before, int(nav_shape[axis]) - after])
    return bounds[0], bounds[1], bounds[2], bounds[3]


def _nlpar_sigma_chunk(
    patterns: np.ndarray,
    *,
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
    depth: tuple[int, int],
    block_info: dict | None = None,
) -> np.ndarray:
    """Run the sigma pass on one haloed block and return its core,
    packed.

    Parameters
    ----------
    patterns
        Haloed block of shape (n_rows, n_cols, h, w), of any real data
        type; cast to float32 here.
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether to exclude saturated pixel pairs.
    depth
        Halo depth of the two navigation axes (1 for a chunked axis, 0
        otherwise).
    block_info
        Dask's block information. Must not be None: the halo sides are
        read from ``block_info[0]["chunk-location"]`` against
        ``block_info[0]["num-chunks"]``.

    Returns
    -------
    packed
        Float32 of shape (core_rows, core_cols, 4, 9): plane 0 ``d2``,
        plane 1 ``n2``, plane 2 ``valid`` as 1.0/0.0, plane 3 slot 0
        ``sigma`` (slots 1-8 zero). See
        :func:`_nlpar_unpack_sigma_pass`.

    Raises
    ------
    AssertionError
        If ``block_info`` is None.

    Notes
    -----
    The first and the last chunk along an axis carry no halo on their
    outer side; ``array-location`` is in overlapped coordinates and is
    not used. :func:`_nlpar_sigma_kernel` runs on the whole block and
    only the core (the block minus its halos) is returned.
    """
    assert block_info is not None, "block_info is required"
    n_rows, n_cols = patterns.shape[:2]
    n_pix = int(np.prod(patterns.shape[2:]))
    row_start, row_stop, col_start, col_stop = _nlpar_core_bounds(
        (n_rows, n_cols), depth, block_info
    )

    data = np.ascontiguousarray(patterns, dtype=np.float32).reshape(
        n_rows, n_cols, n_pix
    )
    sigma, d2, n2, valid = _nlpar_sigma_kernel(
        data,
        np.ascontiguousarray(mask_indices, dtype=np.int64),
        np.float32(max_value),
        bool(saturation_protect),
    )

    # With a halo of at least one point on every side facing another
    # chunk, the clipped 3 x 3 window of every core pattern lies inside
    # the block and is clipped only at the map borders
    core = (slice(row_start, row_stop), slice(col_start, col_stop))
    packed = np.zeros(
        (row_stop - row_start, col_stop - col_start, 4, 9), dtype=np.float32
    )
    packed[..., 0, :] = d2[core]
    packed[..., 1, :] = n2[core]
    packed[..., 2, :] = valid[core]
    packed[..., 3, 0] = sigma[core]
    return packed


def _nlpar_unpack_sigma_pass(
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the sigma pass results from the packed array of
    :func:`_nlpar_sigma_chunk`.

    Parameters
    ----------
    packed
        Float32 of shape (n_rows, n_cols, 4, 9).

    Returns
    -------
    sigma
        Float32 of shape (n_rows, n_cols), from plane 3, slot 0.
    d2, n2
        Float32 of shape (n_rows, n_cols, 9), from planes 0 and 1.
    valid
        Bool of shape (n_rows, n_cols, 9), ``packed[..., 2, :] > 0.5``.
    """
    packed = np.asarray(packed, dtype=np.float32)
    sigma = np.ascontiguousarray(packed[..., 3, 0])
    d2 = np.ascontiguousarray(packed[..., 0, :])
    n2 = np.ascontiguousarray(packed[..., 1, :])
    valid = packed[..., 2, :] > 0.5
    return sigma, d2, n2, valid


def _nlpar_average_chunk(
    patterns: np.ndarray,
    sigma_block: np.ndarray,
    *,
    lam: float,
    dthresh: np.float32,
    radius: tuple[int, int],
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
    depth: tuple[int, int],
    dtype_out: np.dtype,
    block_info: dict | None = None,
) -> np.ndarray:
    """Average the core of one haloed block of patterns.

    Parameters
    ----------
    patterns
        Haloed block of shape (n_rows, n_cols, h, w), of any real data
        type; cast to float32 here.
    sigma_block
        Noise level of every pattern of the block, of shape
        (n_rows, n_cols, 1, 1) with the same halo.
    lam
        Weight decay lambda as float64, > 0.
    dthresh
        Distance threshold as float32, >= 0.
    radius
        Search radius of the two navigation axes.
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether to exclude saturated pixel pairs.
    depth
        Halo depth of the two navigation axes, from :func:`_nlpar_depth`.
    dtype_out
        Output data type.
    block_info
        Dask's block information. Must not be None: the halo sides are
        read from ``block_info[0]["chunk-location"]`` against
        ``block_info[0]["num-chunks"]``.

    Returns
    -------
    out
        Averaged core of shape (core_rows, core_cols, h, w) in
        ``dtype_out``.

    Raises
    ------
    AssertionError
        If ``block_info`` is None.

    Notes
    -----
    The kept region (PyEBSDIndex's ``calclim``) is the block minus its
    halos, where the first and the last chunk along an axis carry no
    halo on their outer side. On it, :func:`_nlpar_distances_kernel`
    (with ``sigma2 = (sigma * sigma).astype(np.float32)``),
    :func:`_nlpar_weights_kernel`, :func:`_nlpar_weighted_sum_kernel`
    and :func:`_nlpar_finalize` run in turn. Only the core is computed
    and returned; nothing is zero-filled or trimmed afterwards.
    """
    assert block_info is not None, "block_info is required"
    n_rows, n_cols = patterns.shape[:2]
    sig_shape = patterns.shape[2:]
    n_pix = int(np.prod(sig_shape))
    row_start, row_stop, col_start, col_stop = _nlpar_core_bounds(
        (n_rows, n_cols), depth, block_info
    )
    radius_rows, radius_cols = int(radius[0]), int(radius[1])

    data = np.ascontiguousarray(patterns, dtype=np.float32).reshape(
        n_rows, n_cols, n_pix
    )
    sigma = np.asarray(sigma_block, dtype=np.float32).reshape(n_rows, n_cols)
    sigma2 = np.ascontiguousarray(sigma * sigma, dtype=np.float32)

    d = _nlpar_distances_kernel(
        data,
        sigma2,
        np.ascontiguousarray(mask_indices, dtype=np.int64),
        np.float32(max_value),
        bool(saturation_protect),
        radius_rows,
        radius_cols,
        row_start,
        row_stop,
        col_start,
        col_stop,
    )
    weights = _nlpar_weights_kernel(d, float(lam), np.float32(dthresh))
    out = _nlpar_weighted_sum_kernel(
        data,
        weights,
        radius_rows,
        radius_cols,
        row_start,
        row_stop,
        col_start,
        col_stop,
    )
    out = _nlpar_finalize(out, dtype_out)
    return out.reshape((row_stop - row_start, col_stop - col_start) + sig_shape)


# ------------------------------ Drivers ----------------------------- #


def _nlpar_as_map(dask_array: da.Array) -> da.Array:
    """Return a map of patterns with two navigation axes and each signal
    axis in one chunk.

    Parameters
    ----------
    dask_array
        Map of shape (n_rows, n_cols, h, w), or a 1-D scan of shape
        (n, h, w).

    Returns
    -------
    x
        Map of shape (n_rows, n_cols, h, w); a 1-D scan becomes the
        single row (1, n, h, w) with its navigation chunks kept.
    """
    x = dask_array
    if x.ndim == 3:
        x = x[None]
    if any(len(c) > 1 for c in x.chunks[2:]):
        x = x.rechunk(x.chunks[:2] + tuple((s,) for s in x.shape[2:]))
    return x


def _nlpar_sigma(
    dask_array: da.Array,
    *,
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the sigma pass over a navigation-chunked map of patterns.

    Parameters
    ----------
    dask_array
        Map of shape (n_rows, n_cols, h, w), of any real data type. A
        1-D scan, of shape (n, h, w), is processed as the single row
        (1, n, h, w). A signal axis split into several chunks is
        rechunked to one chunk.
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether to exclude saturated pixel pairs.

    Returns
    -------
    sigma
        Float32 of shape (n_rows, n_cols), (1, n) for a 1-D scan.
    d2, n2
        Float32 of shape (n_rows, n_cols, 9), (1, n, 9) for a 1-D
        scan.
    valid
        Bool of shape (n_rows, n_cols, 9), (1, n, 9) for a 1-D scan.

    Notes
    -----
    Halo depth 1 on every chunked navigation axis (0 on an unchunked
    one): the clipped 3 x 3 window never shifts, so no rechunk is
    needed. The route is ``da.overlap.overlap(dask_array, depth,
    boundary="none")`` followed by ``da.map_blocks`` of
    :func:`_nlpar_sigma_chunk` with explicit
    ``chunks=nav_chunks + ((4,), (9,))``, ``dtype=np.float32`` and
    ``meta``, so Dask never probes the wrapper without
    ``block_info``. The result is computed eagerly (the averaging pass
    and the lambda fit need the whole map) and unpacked with
    :func:`_nlpar_unpack_sigma_pass`; it equals
    :func:`_nlpar_sigma_kernel` on the whole map bitwise, whatever the
    chunking. The normalised distances follow from
    :func:`_nlpar_normalized_distances` on these arrays.
    """
    x = _nlpar_as_map(dask_array)
    nav_chunks = x.chunks[:2]

    depth = {axis: (1 if len(c) > 1 else 0) for axis, c in enumerate(nav_chunks)}
    depth.update({2: 0, 3: 0})
    if any(depth.values()):
        overlapped = da.overlap.overlap(x, depth=depth, boundary="none")
    else:
        overlapped = x

    packed = da.map_blocks(
        _nlpar_sigma_chunk,
        overlapped,
        mask_indices=np.ascontiguousarray(mask_indices, dtype=np.int64),
        max_value=np.float32(max_value),
        saturation_protect=bool(saturation_protect),
        depth=(depth[0], depth[1]),
        chunks=nav_chunks + ((4,), (9,)),
        dtype=np.float32,
        meta=np.empty((0, 0, 4, 9), dtype=np.float32),
    )
    return _nlpar_unpack_sigma_pass(packed.compute())


def _nlpar_average(
    dask_array: da.Array,
    sigma: np.ndarray,
    *,
    lam: float,
    dthresh: np.float32,
    radius: tuple[int, ...],
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
    dtype_out: np.dtype,
) -> da.Array:
    """Return the lazy averaging pass over a navigation-chunked map of
    patterns.

    Parameters
    ----------
    dask_array
        Map of shape (n_rows, n_cols, h, w), of any real data type. A
        1-D scan, of shape (n, h, w), is processed as the single row
        (1, n, h, w) with radius 0 along the rows and returned in its
        own shape. A signal axis split into several chunks is
        rechunked to one chunk.
    sigma
        Noise level of every pattern, float32 of shape
        (n_rows, n_cols), (n,) or (1, n) for a 1-D scan, every element
        finite and > 0.
    lam
        Weight decay lambda as float64, > 0.
    dthresh
        Distance threshold as float32, >= 0.
    radius
        Search radius of the two navigation axes; for a 1-D scan,
        ``(r,)`` or ``(0, r)``, only the last entry being used.
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether to exclude saturated pixel pairs.
    dtype_out
        Output data type.

    Returns
    -------
    averaged
        Lazy averaged map of the input shape in ``dtype_out``, chunked
        as ``chunks_out`` of :func:`_nlpar_depth`.

    Notes
    -----
    The array is first rechunked to the ``chunks_out`` of
    :func:`_nlpar_depth`; ``sigma[..., None, None]`` becomes a second
    operand with the same navigation chunks; both go through
    ``da.overlap.overlap`` with the same depth and
    ``boundary="none"``, then ``da.map_blocks`` of
    :func:`_nlpar_average_chunk` with explicit ``chunks=chunks_out``
    (the post-rechunk chunks, never the input's), ``dtype`` and
    ``meta``. The result does not depend on the chunking, the
    scheduler or the number of threads.
    """
    one_dimensional = dask_array.ndim == 3
    x = _nlpar_as_map(dask_array)
    sigma = np.ascontiguousarray(sigma, dtype=np.float32).reshape(x.shape[:2])
    if one_dimensional:
        radius = (0, int(radius[-1]))
    radius = (int(radius[0]), int(radius[1]))
    dtype_out = np.dtype(dtype_out)

    # The explicit rechunk precedes the overlap, so that no chunk is
    # thinner than its halo and the chunk wrapper's core sizes are the
    # post-rechunk chunks
    depth, chunks_out = _nlpar_depth(x.chunks, radius)
    x = x.rechunk(chunks_out)
    nav_chunks = chunks_out[:2]
    sigma_dask = da.from_array(sigma[..., None, None], chunks=nav_chunks + ((1,), (1,)))
    if any(depth.values()):
        patterns_ov = da.overlap.overlap(x, depth=depth, boundary="none")
        sigma_ov = da.overlap.overlap(sigma_dask, depth=depth, boundary="none")
    else:
        patterns_ov = x
        sigma_ov = sigma_dask

    averaged = da.map_blocks(
        _nlpar_average_chunk,
        patterns_ov,
        sigma_ov,
        lam=float(lam),
        dthresh=np.float32(dthresh),
        radius=radius,
        mask_indices=np.ascontiguousarray(mask_indices, dtype=np.int64),
        max_value=np.float32(max_value),
        saturation_protect=bool(saturation_protect),
        depth=(depth[0], depth[1]),
        dtype_out=dtype_out,
        chunks=chunks_out,
        dtype=dtype_out,
        meta=np.empty((0,) * x.ndim, dtype=dtype_out),
    )
    if one_dimensional:
        averaged = averaged[0]
    return averaged


def _nlpar_lambda(
    dask_array: da.Array,
    sigma: np.ndarray | None,
    *,
    mask_indices: np.ndarray,
    max_value: np.float32,
    saturation_protect: bool,
    target_weight: float,
    dthresh: np.float32,
) -> tuple[float, np.ndarray]:
    """Run the sigma pass and return the optimised lambda and the sigma
    map it used.

    Parameters
    ----------
    dask_array
        Map of shape (n_rows, n_cols, h, w), or a 1-D scan of shape
        (n, h, w), of any real data type.
    sigma
        Noise level of every pattern, float32 of the navigation shape,
        or None to use the estimate of the sigma pass.
    mask_indices
        Flat indices of the kept pixels, int64 in ascending order.
    max_value
        Global maximum of the map as float32.
    saturation_protect
        Whether to exclude saturated pixel pairs.
    target_weight
        Normalised weight a pattern should keep for itself, in (0, 1).
    dthresh
        Distance threshold as float32, >= 0.

    Returns
    -------
    lam
        Optimised lambda from :func:`_nlpar_optimize_lambda`.
    sigma
        The given sigma, or the estimate of the sigma pass, float32 of
        shape (n_rows, n_cols), (1, n) for a 1-D scan.

    Notes
    -----
    The sigma pass runs once: its raw accumulators give the normalised
    3 x 3 distances with the given or the estimated sigma, so the
    averaging pass that follows can reuse the returned sigma.
    """
    sigma_pass, d2, n2, valid = _nlpar_sigma(
        dask_array,
        mask_indices=mask_indices,
        max_value=max_value,
        saturation_protect=saturation_protect,
    )
    if sigma is None:
        sigma = sigma_pass
    else:
        sigma = np.asarray(sigma, dtype=np.float32).reshape(sigma_pass.shape)
    d = _nlpar_normalized_distances(d2, n2, valid, sigma)
    lam = _nlpar_optimize_lambda(d, valid, target_weight, dthresh)
    return lam, sigma
