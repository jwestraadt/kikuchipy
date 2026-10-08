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
# functionality in this file is derived and adapted from EMsoftOO:
# - The orientation similarity map getOrientationSimilarityMap, its
#   neighbour bookkeeping and its edge multipliers
#   (EMsoftOOLib/mod_DIsupport.f90)
# - The list intersection count vectormatch (EMsoftOOLib/mod_math.f90)

# #####################################################################
# Copyright (c) 2013-2026, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  - Neither the names of Marc De Graef, Carnegie Mellon University nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################

# Changes by the kikuchipy developers, 2026-10-06: ported from Fortran
# to NumPy; defects reproduced only behind ``emsoft_compatible``.

"""Orientation similarity maps (OSM) of top-match lists, grain aware
and EMsoft compatible.
"""

from __future__ import annotations

import numpy as np

from kikuchipy.indexing._hrosm._kam import (
    _emsoft_edge_multipliers,
    _emsoft_neighbour_sum,
)

# Number of list pairs compared per chunk, bounding the memory of the
# broadcast equality tables
_CHUNK_SIZE = 65536


def _osm_grain_aware(
    simulation_indices: np.ndarray,
    grain_id: np.ndarray,
    reindexed: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return the grain-aware orientation similarity map.

    Per re-indexed grain point, the mean over its 4 nearest neighbours
    that exist, are in the same grain and were re-indexed, of the
    number of indices shared by the first ``n`` indices of both lists
    (integer counts, mean in float64).

    Parameters
    ----------
    simulation_indices
        Top-match indices of shape (H * W, k), flat in raster order,
        ``k >= n``.
    grain_id
        Grain labels of shape (H, W).
    reindexed
        Boolean array of shape (H, W), True where the point was
        re-indexed.
    n
        Number of indices per list to compare.

    Returns
    -------
    osm
        Map of shape (H, W) of float32, range 0 to ``n``; NaN without
        such a neighbour.
    """
    simulation_indices = np.asarray(simulation_indices)
    grain_id = np.asarray(grain_id)
    reindexed = np.asarray(reindexed, dtype=bool)
    H, W = grain_id.shape
    lists = simulation_indices[:, :n].reshape(H, W, n)

    total = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.int64)
    # Horizontal pairs (y, x)-(y, x + 1) and vertical pairs
    # (y, x)-(y + 1, x), each credited to both of its points
    for a_slice, b_slice in [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ]:
        valid = (
            reindexed[a_slice]
            & reindexed[b_slice]
            & (grain_id[a_slice] == grain_id[b_slice])
        )
        if not valid.any():
            continue
        shared = np.zeros(valid.shape, dtype=np.int64)
        shared[valid] = _set_intersection_count(
            lists[a_slice][valid], lists[b_slice][valid]
        )
        for point_slice in (a_slice, b_slice):
            total[point_slice] += shared
            count[point_slice] += valid
    osm = np.full((H, W), np.nan, dtype=np.float64)
    has_neighbour = reindexed & (count > 0)
    osm[has_neighbour] = total[has_neighbour] / count[has_neighbour]
    return osm.astype(np.float32)


def _osm_emsoft(simulation_indices: np.ndarray, H: int, W: int, n: int) -> np.ndarray:
    """Return EMsoft's orientation similarity map, shape (H, W) of
    float32, not divided by ``n``.

    The counts of shared indices among the first ``n`` of each list
    are accumulated in float32 with EMsoft's neighbour bookkeeping (no
    spurious term) and edge multipliers, with ``x -> (x * 4) / 3`` in
    float32.

    Parameters
    ----------
    simulation_indices
        Top-match indices of shape (H * W, k), flat in raster order,
        ``k >= n``.
    H, W
        Number of rows and columns.
    n
        Number of indices per list to compare.
    """
    lists = np.asarray(simulation_indices)[:, :n].reshape(H, W, n)
    # Horizontal pairs, right list against left list, and vertical
    # pairs, upper list against lower list, as EMsoft compares them
    # (the count is not symmetric if a list holds duplicates)
    pair_h = _vectormatch(
        lists[:, 1:].reshape(-1, n), lists[:, :-1].reshape(-1, n)
    ).reshape(H, W - 1)
    pair_v = _vectormatch(lists[:-1].reshape(-1, n), lists[1:].reshape(-1, n)).reshape(
        H - 1, W
    )
    # EMsoft's comparison with its initial list of zeros shares no
    # index (indices are 1-based), so the spurious term is zero
    acc = _emsoft_neighbour_sum(
        pair_h.astype(np.float32),
        pair_v.astype(np.float32),
        np.float32(0),
        H,
        W,
        np.float32,
    )
    acc = _emsoft_edge_multipliers(acc, H, W, _third_float32)
    return acc.reshape(H, W)


def _vectormatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return, per row, the number of entries of ``a`` present in
    ``b``, as EMsoft's ``vectormatch`` counts them.

    Parameters
    ----------
    a, b
        Lists of shape (m, n).

    Returns
    -------
    counts
        Integer counts of shape (m,).
    """
    m = a.shape[0]
    counts = np.zeros(m, dtype=np.int64)
    for start in range(0, m, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, m)
        equal = a[start:stop, :, None] == b[start:stop, None, :]
        counts[start:stop] = equal.any(axis=2).sum(axis=1)
    return counts


def _set_intersection_count(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return, per row, the number of distinct values shared by ``a``
    and ``b``, the size of their set intersection.

    Parameters
    ----------
    a, b
        Lists of shape (m, n).

    Returns
    -------
    counts
        Integer counts of shape (m,).
    """
    m = a.shape[0]
    counts = np.zeros(m, dtype=np.int64)
    for start in range(0, m, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, m)
        a_sorted = np.sort(a[start:stop], axis=1)
        # Count every distinct value of a row once
        first = np.ones(a_sorted.shape, dtype=bool)
        first[:, 1:] = a_sorted[:, 1:] != a_sorted[:, :-1]
        present = (a_sorted[:, :, None] == b[start:stop, None, :]).any(axis=2)
        counts[start:stop] = (present & first).sum(axis=1)
    return counts


def _third_float32(x: np.ndarray) -> np.ndarray:
    """Return ``(x * 4) / 3`` in float32, in the order of EMsoft's
    source text.
    """
    return (x * np.float32(4)) / np.float32(3)
