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

"""Tests of the grain-aware and the EMsoft compatible orientation
similarity maps (OSM) on synthetic top-match lists.

The EMsoft compatible map is compared with a literal transcription of
EMsoft's ``getOrientationSimilarityMap`` loop, written in this module
from the EMsoftOO source: a one-dimensional loop over the map points
with EMsoft's neighbour bookkeeping, list intersection counts summed in
float32 and EMsoft's edge multipliers in float32, without a division by
the list length. The grain-aware map is compared with kikuchipy's
legacy ``orientation_similarity_map`` on a single grain and with
analytic lists. Comparisons with EMsoft's own files live in
``test_hrosm_emsoft_regression.py``.
"""

from __future__ import annotations

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrosm._osm import _osm_emsoft, _osm_grain_aware

# ------------------------ Test-local oracles ------------------------ #


def vectormatch(a: np.ndarray, b: np.ndarray) -> int:
    """Return the number of entries of ``a`` present in ``b``, as
    EMsoft's ``vectormatch`` counts them (lists without duplicates).
    """
    count = 0
    for value in a:
        for other in b:
            if value == other:
                count += 1
                break
    return count


def emsoft_third(x):
    """Return ``(x * 4) / 3`` in float32, the order of EMsoft's source
    text.
    """
    return (np.float32(x) * np.float32(4)) / np.float32(3)


def emsoft_osm_loop(top: np.ndarray, n: int, H: int, W: int) -> np.ndarray:
    """Return EMsoft's orientation similarity map by a literal loop.

    ``top`` holds 1-based top-match lists of shape (H * W, k) in raster
    order. The loop runs over the 1-based flat index ``t`` with x
    fastest: ``ii`` is the column and ``jj`` the row, one too large in
    the last column. The previous row's lists are copied into
    ``lstore`` at the start of each row after the first; ``lstore``
    and ``pstore`` start as lists of zeros, which share no index with
    any list. A horizontal pair is credited to both of its points; the
    vertical pair at ``t`` is credited to ``t`` and to ``t - W + 1``.
    Sums are float32, then the edge multipliers are applied in the
    source order, and the result is not divided by ``n``.
    """
    n_points = H * W
    acc = np.zeros(n_points, dtype=np.float32)
    zero = np.zeros(n, dtype=np.int64)
    lstore = [zero] * W
    pstore = [zero] * W
    cp = zero
    for t in range(1, n_points + 1):
        ii = t % W
        if ii == 0:
            ii = W
        jj = t // W + 1
        if ii == 1 and jj > 1:
            lstore = list(pstore)
        if ii == 1:
            cp = top[t - 1, :n]
            pstore[ii - 1] = cp
        else:
            lp = cp
            cp = top[t - 1, :n]
            pstore[ii - 1] = cp
            # vectormatch(lnm, cp, lp): the current list first
            d = np.float32(vectormatch(cp, lp))
            acc[t - 2] = acc[t - 2] + d
            acc[t - 1] = acc[t - 1] + d
        if jj > 1:
            d = np.float32(vectormatch(lstore[ii - 1], cp))
            acc[t - W] = acc[t - W] + d
            acc[t - 1] = acc[t - 1] + d

    acc = acc * np.float32(0.25)
    for i in range(1, W - 1):
        acc[i] = emsoft_third(acc[i])
    for i in range(n_points - W + 1, n_points - 1):
        acc[i] = emsoft_third(acc[i])
    for jj in range(1, H - 1):
        acc[W * jj] = emsoft_third(acc[W * jj])
    for jj in range(2, H):
        acc[W * jj - 1] = emsoft_third(acc[W * jj - 1])
    acc[0] = acc[0] * np.float32(4)
    acc[W - 1] = acc[W - 1] * np.float32(2)
    acc[n_points - 1] = acc[n_points - 1] * np.float32(2)
    acc[n_points - W] = emsoft_third(acc[n_points - W])
    return acc.reshape(H, W)


def crystal_map_with_lists(lists: np.ndarray, shape: tuple[int, int]) -> CrystalMap:
    """Return a nickel crystal map of identity rotations with the
    top-match lists as the property ``"simulation_indices"``.
    """
    coords, n = create_coordinate_arrays(shape, step_sizes=(1, 1))
    return CrystalMap(
        rotations=Rotation.identity((n,)),
        phase_id=np.zeros(n, dtype=np.int32),
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phases=[Phase("ni", point_group="m-3m")], ids=[0]),
        prop={"simulation_indices": lists},
    )


def shape_id(shape: tuple[int, int]) -> str:
    """Return a test ID of a map shape, rows x columns."""
    return f"{shape[0]}x{shape[1]}"


def disjoint_lists(n_points: int, n: int) -> np.ndarray:
    """Return 1-based lists of which no two share an index."""
    return (np.arange(n_points * n).reshape(n_points, n) + 1).astype(np.int32)


# ------------------------- EMsoft compatible ------------------------ #


class TestCompatOSM:
    @pytest.mark.parametrize("n", [10, 5], ids=lambda n: f"n{n}")
    @pytest.mark.parametrize(
        "shape", [(1, 1), (1, 6), (6, 1), (2, 2), (5, 7), (9, 11)], ids=shape_id
    )
    def test_matches_the_loop_transcription_bitwise(self, hrosm_top_lists, shape, n):
        H, W = shape
        top = hrosm_top_lists(shape, n=10, pool=15)
        expected = emsoft_osm_loop(top, n, H, W)
        osm = _osm_emsoft(top, H, W, n)
        assert osm.dtype == np.float32
        assert osm.shape == (H, W)
        assert np.array_equal(osm, expected)

    def test_edge_multiplier_follows_the_source_order(self):
        # Top-edge point (0, 1) shares 2 indices with its left and
        # its right neighbour and gets the vertical pair (0, 0)-(1, 0)
        # with 1 shared index, which EMsoft credits one column to the
        # right: 5 in total, times 0.25 is 1.25, and the edge
        # multiplier (x * 4) / 3 gives 1.6666666 in float32, while the
        # folded x * (4 / 3) gives 1.6666667
        H, W, n = 3, 4, 3
        top = disjoint_lists(H * W, n) + 100
        top[0] = [1, 2, 3]
        top[1] = [1, 2, 4]
        top[2] = [1, 2, 5]
        top[W] = [3, 6, 7]
        source_order = (np.float32(1.25) * np.float32(4)) / np.float32(3)
        folded = np.float32(1.25) * np.float32(4 / 3)
        assert source_order == np.float32(1.6666666)
        assert folded == np.float32(1.6666667)
        assert source_order != folded

        osm = _osm_emsoft(top, H, W, n)
        assert osm[0, 1] == source_order
        assert np.array_equal(osm, emsoft_osm_loop(top, n, H, W))

    def test_pair_order_follows_the_source_with_duplicates(self):
        # vectormatch counts entries of its first list found in the
        # second, which is not symmetric when a list holds duplicates:
        # EMsoft passes the right list first in a horizontal pair and
        # the upper list first in a vertical pair
        assert vectormatch(np.array([1, 1, 2]), np.array([1, 3, 4])) == 2
        assert vectormatch(np.array([1, 3, 4]), np.array([1, 1, 2])) == 1
        n = 3
        top = np.array([[1, 1, 2], [1, 3, 4]], dtype=np.int32)
        osm = _osm_emsoft(top, 1, 2, n)
        assert np.array_equal(osm, emsoft_osm_loop(top, n, 1, 2))
        # One shared entry (two in the reversed order, which doubles
        # the map)
        assert np.array_equal(osm, np.array([[4 / 3, 1.0]], dtype=np.float32))

        top = np.array([[1, 1, 2], [5, 6, 7], [1, 3, 4], [8, 9, 10]], np.int32)
        osm = _osm_emsoft(top, 2, 2, n)
        assert np.array_equal(osm, emsoft_osm_loop(top, n, 2, 2))
        # The vertical pair (0, 0)-(1, 0) counts 2 entries of the
        # upper list (1 in the reversed order); EMsoft credits it to
        # (1, 0) and, one column to the right, to (0, 1)
        expected = np.array([[0.0, 1.0], [2 / 3, 0.0]], dtype=np.float32)
        assert np.array_equal(osm, expected)

    @pytest.mark.parametrize("shape", [(5, 7), (4, 6)], ids=shape_id)
    def test_compat_osm_is_not_divided_by_n(self, shape):
        H, W = shape
        n = 10
        top = np.tile(np.arange(1, n + 1, dtype=np.int32), (H * W, 1))
        osm = _osm_emsoft(top, H, W, n)
        assert osm.dtype == np.float32
        assert np.array_equal(osm, np.full((H, W), 10.0, dtype=np.float32))


# ---------------------------- Grain aware --------------------------- #


class TestGrainAwareOSM:
    @pytest.mark.parametrize("n", [10, 5], ids=lambda n: f"n{n}")
    @pytest.mark.parametrize("shape", [(6, 7), (3, 3), (2, 5)], ids=shape_id)
    def test_equals_the_legacy_function_on_one_grain(self, hrosm_top_lists, shape, n):
        top = hrosm_top_lists(shape, n=10, pool=15, seed=41)
        xmap = crystal_map_with_lists(top, shape)
        legacy = kp.indexing.orientation_similarity_map(xmap, n_best=n)
        grain_id = np.ones(shape, dtype=np.int32)
        reindexed = np.ones(shape, dtype=bool)
        osm = _osm_grain_aware(top, grain_id, reindexed, n)
        assert osm.dtype == np.float32
        assert osm.shape == shape
        assert legacy.shape == shape
        assert np.array_equal(osm, legacy)

    def test_cross_grain_neighbours_are_excluded(self):
        # grain A (columns 0-3) holds the lists 1..10 everywhere and
        # grain B (columns 4-7) 11..20: counting the cross-grain
        # neighbour would lower column 3 and column 4 below 10
        H, W, n = 5, 8, 10
        grain_id = np.ones((H, W), dtype=np.int32)
        grain_id[:, 4:] = 2
        top = np.empty((H * W, n), dtype=np.int32)
        flat_id = grain_id.ravel()
        top[flat_id == 1] = np.arange(1, 11)
        top[flat_id == 2] = np.arange(11, 21)
        osm = _osm_grain_aware(top, grain_id, np.ones((H, W), dtype=bool), n)
        assert np.array_equal(osm, np.full((H, W), 10.0, dtype=np.float32))

    def test_pixel_without_an_in_grain_neighbour_is_nan(self):
        # the centre point is a grain of its own; the point (0, 0) of
        # grain 1 has only neighbours which were not re-indexed
        H, W, n = 3, 4, 5
        grain_id = np.ones((H, W), dtype=np.int32)
        grain_id[1, 1] = 2
        reindexed = np.ones((H, W), dtype=bool)
        reindexed[0, 1] = False
        reindexed[1, 0] = False
        top = np.tile(np.arange(1, n + 1, dtype=np.int32), (H * W, 1))
        osm = _osm_grain_aware(top, grain_id, reindexed, n)
        assert osm.dtype == np.float32
        assert np.isnan(osm[1, 1])
        assert np.isnan(osm[0, 0])
        for point in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3)]:
            assert osm[point] == n, point

    def test_not_reindexed_neighbours_are_excluded(self):
        # one point was not re-indexed and holds a disjoint list:
        # counting it would lower its four neighbours below n
        H, W, n = 3, 4, 5
        top = np.tile(np.arange(1, n + 1, dtype=np.int32), (H * W, 1))
        top[1 * W + 2] = np.arange(101, 101 + n)
        reindexed = np.ones((H, W), dtype=bool)
        reindexed[1, 2] = False
        grain_id = np.ones((H, W), dtype=np.int32)
        osm = _osm_grain_aware(top, grain_id, reindexed, n)
        for point in [(0, 2), (2, 2), (1, 1), (1, 3)]:
            assert osm[point] == n, point

    def test_range_is_zero_to_n(self):
        H, W, n = 4, 5, 6
        grain_id = np.ones((H, W), dtype=np.int32)
        reindexed = np.ones((H, W), dtype=bool)

        disjoint = disjoint_lists(H * W, n)
        osm = _osm_grain_aware(disjoint, grain_id, reindexed, n)
        assert np.array_equal(osm, np.zeros((H, W), dtype=np.float32))

        identical = np.tile(np.arange(1, n + 1, dtype=np.int32), (H * W, 1))
        osm = _osm_grain_aware(identical, grain_id, reindexed, n)
        assert np.array_equal(osm, np.full((H, W), n, dtype=np.float32))

    def test_one_row_map_has_only_horizontal_neighbours(self):
        # a single row has no vertical pair, and no point re-indexed
        # leaves no pair at all
        H, W, n = 1, 4, 5
        grain_id = np.ones((H, W), dtype=np.int32)
        identical = np.tile(np.arange(1, n + 1, dtype=np.int32), (H * W, 1))
        osm = _osm_grain_aware(identical, grain_id, np.ones((H, W), dtype=bool), n)
        assert np.array_equal(osm, np.full((H, W), n, dtype=np.float32))

        osm = _osm_grain_aware(identical, grain_id, np.zeros((H, W), dtype=bool), n)
        assert osm.dtype == np.float32
        assert np.all(np.isnan(osm))
