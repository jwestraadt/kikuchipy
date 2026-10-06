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
    raise NotImplementedError


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
    raise NotImplementedError
