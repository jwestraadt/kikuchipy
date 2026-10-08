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

from __future__ import annotations

import numpy as np
from orix.crystal_map import CrystalMap, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest
from scipy.ndimage import generic_filter

import kikuchipy as kp
from kikuchipy.indexing._hrosm._osm import _osm_emsoft, _osm_grain_aware


class TestOrientationSimilarityMap:
    def test_orientation_similarity_map(self):
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.tile(np.arange(5), (100, 1))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        assert np.allclose(
            kp.indexing.orientation_similarity_map(xmap), np.full((10, 10), 5)
        )

        assert np.allclose(
            kp.indexing.orientation_similarity_map(xmap, normalize=True),
            np.ones((10, 10)),
        )

    def test_n_best_too_great(self):
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.ones((100, 5))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        with pytest.raises(ValueError, match="n_best 6 cannot be greater than"):
            kp.indexing.orientation_similarity_map(xmap, n_best=6)

    def test_from_n_best(self):
        sim_idx_prop = "simulated_indices"
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={sim_idx_prop: np.ones((100, 5))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        osm = kp.indexing.orientation_similarity_map(
            xmap, simulation_indices_prop=sim_idx_prop, from_n_best=2
        )
        assert osm.shape == (10, 10, 4)


# ----------------- Grain-aware and EMsoft compatible ---------------- #


def _osm_crystal_map(
    simulation_indices: np.ndarray,
    shape: tuple[int, int],
    is_in_data: np.ndarray | None = None,
) -> CrystalMap:
    """Return a crystal map of the given grid shape with the top-match
    lists as the property "simulation_indices", over all points.
    """
    coords, n = create_coordinate_arrays(shape, step_sizes=(1, 1))
    if is_in_data is None:
        is_in_data = np.ones(n, dtype=bool)
    return CrystalMap(
        rotations=Rotation.identity(n),
        x=coords["x"],
        y=coords["y"],
        prop={"simulation_indices": simulation_indices},
        is_in_data=is_in_data,
    )


def _legacy_osm_by_loop(lists: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return the legacy orientation similarity map by an explicit loop:
    per point, the mean over its existing four nearest neighbours of
    the number of shared indices, in float64, stored as float32.
    """
    H, W = shape
    osm = np.zeros(shape, dtype=np.float32)
    for r in range(H):
        for c in range(W):
            counts = []
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    p, q = r * W + c, rr * W + cc
                    counts.append(len(np.intersect1d(lists[p], lists[q])))
            osm[r, c] = np.mean(counts)
    return osm


def _two_grain_id(shape: tuple[int, int] = (5, 8)) -> np.ndarray:
    """Return grain labels of two grains, columns 0-3 label 1 and the
    rest label 2.
    """
    grain_id = np.ones(shape, dtype=np.int32)
    grain_id[:, 4:] = 2
    return grain_id


def _route_keywords(route: str, shape: tuple[int, int]) -> dict:
    """Return the keyword routing to the grain-aware or the EMsoft
    compatible map.
    """
    if route == "grain_id":
        return {"grain_id": _two_grain_id(shape)}
    return {"emsoft_compatible": True}


def _route_oracle(
    route: str, lists: np.ndarray, shape: tuple[int, int], n: int
) -> np.ndarray:
    """Return the array function result the public function must
    reproduce on a dense map.
    """
    if route == "grain_id":
        return _osm_grain_aware(
            lists, _two_grain_id(shape), np.ones(shape, dtype=bool), n
        )
    return _osm_emsoft(lists, shape[0], shape[1], n)


class TestHROSMKeywords:
    def test_defaults_run_the_legacy_path_unchanged(self, hrosm_top_lists):
        defaults = {"grain_id": None, "emsoft_compatible": False}

        # The inputs of the existing tests give identical results
        coords = {"x": np.tile(np.arange(10), 10), "y": np.tile(np.arange(10), 10)}
        xmap1 = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.tile(np.arange(5), (100, 1))},
            **coords,
        )
        for normalize in (False, True):
            np.testing.assert_array_equal(
                kp.indexing.orientation_similarity_map(
                    xmap1, normalize=normalize, **defaults
                ),
                kp.indexing.orientation_similarity_map(xmap1, normalize=normalize),
            )
        xmap2 = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.ones((100, 5))},
            **coords,
        )
        with pytest.raises(ValueError, match="n_best 6 cannot be greater than"):
            kp.indexing.orientation_similarity_map(xmap2, n_best=6, **defaults)
        osm2 = kp.indexing.orientation_similarity_map(xmap2, from_n_best=2, **defaults)
        assert osm2.shape == (10, 10, 4)
        np.testing.assert_array_equal(
            osm2, kp.indexing.orientation_similarity_map(xmap2, from_n_best=2)
        )

        # A (2, 5) map gives the same (2, 5) array bitwise, equal to
        # the explicit loop
        shape = (2, 5)
        lists = hrosm_top_lists(shape) - 1
        xmap3 = _osm_crystal_map(lists, shape)
        osm3 = kp.indexing.orientation_similarity_map(xmap3, **defaults)
        assert osm3.shape == shape
        assert osm3.dtype == np.float32
        np.testing.assert_array_equal(
            osm3, kp.indexing.orientation_similarity_map(xmap3)
        )
        np.testing.assert_array_equal(osm3, _legacy_osm_by_loop(lists, shape))

        # A full 3 x 3 footprint with its centre index still runs on
        # the legacy path
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap4 = _osm_crystal_map(lists, shape)
        footprint = np.ones((3, 3), dtype=bool)

        def eight_neighbours(v: np.ndarray) -> float:
            v = v.astype(int)
            centre = v[4]
            neighbours = v[(v != -1) & (v != centre)]
            counts = [len(np.intersect1d(lists[centre], lists[i])) for i in neighbours]
            return np.nanmean(counts)

        expected = generic_filter(
            np.arange(lists.shape[0]).reshape(shape),
            eight_neighbours,
            footprint=footprint,
            mode="constant",
            cval=-1,
            output=np.float32,
        )
        osm4 = kp.indexing.orientation_similarity_map(
            xmap4, footprint=footprint, center_index=4, **defaults
        )
        np.testing.assert_array_equal(osm4, expected)

        # from_n_best still returns the 3D legacy array
        osm5 = kp.indexing.orientation_similarity_map(xmap4, from_n_best=5, **defaults)
        assert osm5.shape == shape + (6,)
        np.testing.assert_array_equal(
            osm5[:, :, 0], kp.indexing.orientation_similarity_map(xmap4)
        )
        np.testing.assert_array_equal(
            osm5[:, :, -1], kp.indexing.orientation_similarity_map(xmap4, n_best=5)
        )

    def test_grain_id_routes_to_the_grain_aware_map(self, hrosm_top_lists):
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap = _osm_crystal_map(lists, shape)
        grain_id = _two_grain_id(shape)
        osm = kp.indexing.orientation_similarity_map(xmap, grain_id=grain_id)
        assert osm.shape == shape
        assert osm.dtype == np.float32
        expected = _osm_grain_aware(lists, grain_id, np.ones(shape, dtype=bool), 10)
        np.testing.assert_array_equal(osm, expected)
        # Columns 3 and 4 lose their cross-grain neighbour, so the
        # legacy map differs there
        legacy = kp.indexing.orientation_similarity_map(xmap)
        assert not np.array_equal(osm[:, 3:5], legacy[:, 3:5])

    def test_emsoft_compatible_routes_to_the_emsoft_table(self, hrosm_top_lists):
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap = _osm_crystal_map(lists, shape)
        osm = kp.indexing.orientation_similarity_map(xmap, emsoft_compatible=True)
        assert osm.shape == shape
        assert osm.dtype == np.float32
        np.testing.assert_array_equal(osm, _osm_emsoft(lists, 5, 8, 10))
        # EMsoft's edge multipliers change the edges
        legacy = kp.indexing.orientation_similarity_map(xmap)
        assert not np.array_equal(osm, legacy)

    @pytest.mark.parametrize("route", ["grain_id", "emsoft_compatible"])
    @pytest.mark.parametrize(
        "legacy_keyword",
        [
            {"footprint": np.ones((3, 3), dtype=bool)},
            {"center_index": 4},
            {"from_n_best": 5},
        ],
        ids=["footprint", "center_index", "from_n_best"],
    )
    def test_cannot_be_combined_with_footprint_center_index_or_from_n_best(
        self, hrosm_top_lists, route, legacy_keyword
    ):
        shape = (5, 8)
        xmap = _osm_crystal_map(hrosm_top_lists(shape) - 1, shape)
        with pytest.raises(ValueError, match="cannot be combined"):
            kp.indexing.orientation_similarity_map(
                xmap, **legacy_keyword, **_route_keywords(route, shape)
            )

    @pytest.mark.parametrize("route", ["grain_id", "emsoft_compatible"])
    def test_normalize_divides_by_n(self, hrosm_top_lists, route):
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap = _osm_crystal_map(lists, shape)
        keywords = _route_keywords(route, shape)
        osm = kp.indexing.orientation_similarity_map(xmap, normalize=True, **keywords)
        assert osm.dtype == np.float32
        expected = _route_oracle(route, lists, shape, 10) / 10
        np.testing.assert_array_equal(osm, expected)
        osm6 = kp.indexing.orientation_similarity_map(
            xmap, n_best=6, normalize=True, **keywords
        )
        np.testing.assert_array_equal(osm6, _route_oracle(route, lists, shape, 6) / 6)

    @pytest.mark.parametrize("route", ["grain_id", "emsoft_compatible"])
    def test_n_best_sets_n(self, hrosm_top_lists, route):
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap = _osm_crystal_map(lists, shape)
        keywords = _route_keywords(route, shape)
        osm = kp.indexing.orientation_similarity_map(xmap, n_best=3, **keywords)
        np.testing.assert_array_equal(osm, _route_oracle(route, lists, shape, 3))
        assert not np.array_equal(osm, _route_oracle(route, lists, shape, 10))

    def test_absent_points_and_keep_n_one_work_without_squeeze(self):
        # One point not in the data and a squeezed keep_n = 1 property
        shape = (1, 6)
        is_in_data = np.ones(6, dtype=bool)
        is_in_data[2] = False
        values = np.array([3, 3, 0, 3, 4, 4])
        xmap = _osm_crystal_map(values, shape, is_in_data=is_in_data)
        assert xmap.prop["simulation_indices"].shape == (5,)
        grain_id = np.ones(shape, dtype=np.int32)
        osm = kp.indexing.orientation_similarity_map(xmap, grain_id=grain_id)
        assert osm.shape == shape
        assert osm.dtype == np.float32
        expected = np.array([[1, 1, np.nan, 0, 0.5, 1]], dtype=np.float32)
        np.testing.assert_array_equal(osm, expected)
        np.testing.assert_array_equal(
            osm, _osm_grain_aware(values[:, None], grain_id, is_in_data[None], 1)
        )

        # The EMsoft compatible map of a dense one-row map
        dense = _osm_crystal_map(values, shape)
        osm_compat = kp.indexing.orientation_similarity_map(
            dense, emsoft_compatible=True
        )
        assert osm_compat.shape == shape
        assert osm_compat.dtype == np.float32
        np.testing.assert_array_equal(osm_compat, _osm_emsoft(values[:, None], 1, 6, 1))

    def test_new_path_rejects_invalid_combinations(self, hrosm_top_lists):
        shape = (5, 8)
        lists = hrosm_top_lists(shape) - 1
        xmap = _osm_crystal_map(lists, shape)
        with pytest.raises(ValueError, match="grain_id cannot be combined with"):
            kp.indexing.orientation_similarity_map(
                xmap, grain_id=_two_grain_id(shape), emsoft_compatible=True
            )
        for keywords in (
            {"grain_id": _two_grain_id(shape)},
            {"emsoft_compatible": True},
        ):
            with pytest.raises(ValueError, match="n_best 11 cannot be greater"):
                kp.indexing.orientation_similarity_map(xmap, n_best=11, **keywords)

        # A point not in the data has no EMsoft compatible map
        is_in_data = np.ones(lists.shape[0], dtype=bool)
        is_in_data[3] = False
        sparse = _osm_crystal_map(lists, shape, is_in_data=is_in_data)
        with pytest.raises(ValueError, match="emsoft_compatible requires every"):
            kp.indexing.orientation_similarity_map(sparse, emsoft_compatible=True)
