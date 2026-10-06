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

"""Tests of grain segmentation by EMsoft's KAM difference rule, grain
dilation and grain bounding boxes.

The oracles are literal transcriptions, written in this module from
EMsoft's Fortran source (``mod_cluster.f90``), of the iterative region
growing ``grow_region_driver_`` and ``grow_region_``, the dilation
``grain_dilate_`` and the bounding boxes ``getROI_``, plus a brute
force of the correct dilation.
"""

from __future__ import annotations

import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import pytest
from scipy.ndimage import maximum_filter

from kikuchipy.indexing import (
    average_grain_orientations,
    grain_bounding_boxes,
    kernel_average_misorientation_map,
    segment_grains_kam,
)
from kikuchipy.indexing._hrosm._segmentation import _dilate_correct, _dilate_emsoft

# Background KAM of the hand-made maps: a component of background
# points never holds a point at or below the threshold
BACKGROUND = 100.0

# Shapes of the random KAM maps compared with the region growing
# transcription, each with its own seed
GROWTH_SHAPES = [(1, 1), (1, 6), (6, 1), (5, 5), (9, 11), (17, 23)]

# Shapes of the random label maps compared with the dilation
# transcription, each with its own seed
DILATE_SHAPES = [(7, 9), (1, 5), (5, 1), (2, 2)]

# EMsoft's neighbour offsets of grow_region_, in its order, as (dx, dy)
EMSOFT_DX = (0, 0, -1, 1, 1, 1, -1, -1)
EMSOFT_DY = (-1, 1, 0, 0, 1, -1, 1, -1)


# Test-local oracles


def emsoft_grow_regions(kam: np.ndarray, gangle: float) -> tuple[np.ndarray, int]:
    """Return EMsoft's grain labels (H, W) of int32 and its number of
    grains for a float32 KAM map.

    A literal transcription of ``grow_region_driver_`` and
    ``grow_region_`` with 1-based (x, y) indices, followed by the reset
    of -1 to 0. EMsoft's grain count is returned as is, so it is -1
    when no grain is accepted and the last raster point is a lone
    rejected seed.
    """
    kam32 = np.asarray(kam, dtype=np.float32)
    g = np.float32(gangle)
    H, W = kam32.shape
    # grain_id[x, y] and kam_xy[x, y], 1-based, padded at index 0
    grain_id = np.zeros((W + 1, H + 1), dtype=np.int64)
    kam_xy = np.zeros((W + 1, H + 1), dtype=np.float32)
    kam_xy[1:, 1:] = kam32.T
    n_grains = 1

    def grow_region(x_seed: int, y_seed: int) -> None:
        nonlocal n_grains
        if grain_id[x_seed, y_seed] != 0:
            n_grains -= 1
            return
        if kam_xy[x_seed, y_seed] > g:
            n_grains -= 1
            return
        stack = [(x_seed, y_seed)]
        grain_id[x_seed, y_seed] = n_grains
        while len(stack) > 0:
            x, y = stack.pop()
            for i in range(8):
                xn = x + EMSOFT_DX[i]
                yn = y + EMSOFT_DY[i]
                if 1 <= xn <= W and 1 <= yn <= H:
                    if grain_id[xn, yn] == 0:
                        if np.abs(kam_xy[xn, yn] - kam_xy[x, y]) <= g:
                            stack.append((xn, yn))
                            grain_id[xn, yn] = n_grains

    for j in range(1, H + 1):
        for i in range(1, W + 1):
            if grain_id[i, j] == 0:
                grow_region(i, j)
                if np.count_nonzero(grain_id[1:, 1:] == n_grains) == 1:
                    grain_id[i, j] = -1
                else:
                    n_grains += 1
    n_grains -= 1

    grain_id[grain_id == -1] = 0
    return grain_id[1:, 1:].T.astype(np.int32), n_grains


def emsoft_grain_dilate(labels: np.ndarray) -> np.ndarray:
    """Return labels dilated by a literal transcription of EMsoft's
    ``grain_dilate_`` with 1-based (x, y) indices.
    """
    H, W = labels.shape
    im_in = np.zeros((W + 2, H + 2), dtype=np.int64)
    im_in[1 : W + 1, 1 : H + 1] = labels.T
    im_out = im_in.copy()
    for i in range(1, W):
        for j in range(1, H):
            sub = im_in[i : i + 3, j : j + 3]
            m = sub.max()
            if m != 0:
                im_out[i + 1, j + 1] = m
    return im_out[1 : W + 1, 1 : H + 1].T.astype(labels.dtype)


def maximum_filter_dilate(labels: np.ndarray) -> np.ndarray:
    """Return labels dilated by the closed form of EMsoft's dilation
    with a 3 x 3 maximum filter.
    """
    m = maximum_filter(labels, size=3, mode="constant", cval=0)
    out = labels.copy()
    out[1:, 1:] = np.where(m[1:, 1:] != 0, m[1:, 1:], labels[1:, 1:])
    return out


def emsoft_get_roi(labels: np.ndarray, n_grains: int) -> np.ndarray:
    """Return EMsoft's grain boxes (n, 4) of 1-based (x0, y0, w, h) by a
    literal transcription of ``getROI_``.
    """
    H, W = labels.shape
    roi = np.zeros((n_grains, 4), dtype=np.int64)
    for i in range(1, n_grains + 1):
        xmin, xmax, ymin, ymax = W, 1, H, 1
        for ic in range(1, W + 1):
            for ir in range(1, H + 1):
                if labels[ir - 1, ic - 1] == i:
                    xmin = min(xmin, ic)
                    xmax = max(xmax, ic)
                    ymin = min(ymin, ir)
                    ymax = max(ymax, ir)
        roi[i - 1] = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
    return roi


def correct_dilate_reference(
    labels: np.ndarray, present: np.ndarray, phase_id: np.ndarray
) -> np.ndarray:
    """Return labels where every unassigned present point takes the
    largest label among its same-phase 8-neighbours in the input
    labels; assigned labels are kept.
    """
    H, W = labels.shape
    out = labels.copy()
    for y in range(H):
        for x in range(W):
            if labels[y, x] != 0 or not present[y, x]:
                continue
            best = 0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    yn, xn = y + oy, x + ox
                    if (oy, ox) == (0, 0) or not (0 <= yn < H and 0 <= xn < W):
                        continue
                    if phase_id[yn, xn] == phase_id[y, x]:
                        best = max(best, int(labels[yn, xn]))
            out[y, x] = best
    return out


def random_kam(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Return a float32 KAM map uniform in [0, 12]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 12, shape).astype(np.float32)


def interior_mask(truth: np.ndarray) -> np.ndarray:
    """Return the points without a 4-neighbour in another true grain."""
    H, W = truth.shape
    mask = np.ones((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            for oy, ox in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                yn, xn = y + oy, x + ox
                if 0 <= yn < H and 0 <= xn < W and truth[yn, xn] != truth[y, x]:
                    mask[y, x] = False
    return mask


def interior_bijection(
    labels: np.ndarray, truth: np.ndarray, interior: np.ndarray
) -> dict[int, int]:
    """Assert that the interior points of every true grain carry one
    non-zero label, distinct across grains, and return the map from
    true grain to label.
    """
    mapping = {}
    for g in np.unique(truth):
        found = np.unique(labels[interior & (truth == g)])
        assert found.size == 1, (g, found)
        assert found[0] != 0, g
        mapping[int(g)] = int(found[0])
    assert len(set(mapping.values())) == len(mapping)
    return mapping


# Tests


class TestSegmentationRule:
    @pytest.mark.parametrize(
        "shape, seed",
        [(shape, 20 + i) for i, shape in enumerate(GROWTH_SHAPES)],
        ids=[f"{s[0]}x{s[1]}-{20 + i}" for i, s in enumerate(GROWTH_SHAPES)],
    )
    def test_matches_the_flood_fill_transcription(self, shape, seed):
        kam = random_kam(seed, shape)
        labels = segment_grains_kam(kam, threshold=5.0)
        expected, n_grains = emsoft_grow_regions(kam, 5.0)
        assert labels.dtype == np.int32
        assert labels.shape == shape
        assert np.array_equal(labels, expected)
        assert labels.max(initial=0) == max(n_grains, 0)

    def test_diagonal_neighbours_join(self):
        kam = np.full((3, 3), BACKGROUND, dtype=np.float32)
        kam[0, 0] = kam[1, 1] = 1.0
        labels = segment_grains_kam(kam, threshold=5.0)
        expected = np.zeros((3, 3), dtype=np.int32)
        expected[0, 0] = expected[1, 1] = 1
        assert np.array_equal(labels, expected)

    def test_criterion_is_chained_on_kam_differences(self):
        kam = np.array([[1, 5, 9, 13, 17]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.ones((1, 5), dtype=np.int32))

    def test_component_without_a_pixel_at_or_below_threshold_is_unassigned(self):
        kam = np.array([[6, 7, 8]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.zeros((1, 3), dtype=np.int32))

        kam = np.array([[6, 6, 2]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.ones((1, 3), dtype=np.int32))

    def test_seed_at_exactly_the_threshold_is_accepted(self):
        kam = np.array([[5.0, 5.0]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.ones((1, 2), dtype=np.int32))

    def test_singletons_are_unassigned(self):
        kam = np.full((3, 3), BACKGROUND, dtype=np.float32)
        kam[1, 1] = 1.0
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.zeros((3, 3), dtype=np.int32))

    def test_threshold_tie_is_decided_in_float32(self):
        a = np.float32(0.01)
        b = np.float32(5.01)
        # The float32 difference is exactly the threshold, the float64
        # difference of the same values is above it
        assert np.abs(b - a) == np.float32(5.0)
        assert float(b) - float(a) > 5.0

        kam = np.array([[a, b]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.ones((1, 2), dtype=np.int32))

    def test_labels_follow_the_first_qualifying_pixel_in_raster_order(self):
        # The left component's first raster point (0, 0) is above the
        # threshold; its first qualifying point (1, 0) comes after the
        # right component's (0, 2)
        kam = np.array(
            [[6.0, BACKGROUND, 1.0, 1.0], [2.0, BACKGROUND, BACKGROUND, BACKGROUND]],
            dtype=np.float32,
        )
        labels = segment_grains_kam(kam, threshold=5.0)
        expected = np.array([[2, 0, 1, 1], [2, 0, 0, 0]], dtype=np.int32)
        assert np.array_equal(labels, expected)

    def test_nan_kam_is_never_assigned(self):
        kam = np.array([[1, 1, np.nan, 1, 1]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.array([[1, 1, 0, 2, 2]], dtype=np.int32))

        kam = np.array([[np.nan, np.nan], [np.nan, 1.0]], dtype=np.float32)
        labels = segment_grains_kam(kam, threshold=5.0)
        assert np.array_equal(labels, np.zeros((2, 2), dtype=np.int32))

    def test_phases_are_segmented_separately(self):
        kam = np.ones((2, 4), dtype=np.float32)
        phase_id = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        labels = segment_grains_kam(kam, threshold=5.0, phase_id=phase_id)
        expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
        assert np.array_equal(labels, expected)

        kam = np.ones((1, 5), dtype=np.float32)
        phase_id = np.array([[0, 0, -1, 1, 1]])
        labels = segment_grains_kam(kam, threshold=5.0, phase_id=phase_id)
        assert np.array_equal(labels, np.array([[1, 1, 0, 2, 2]], dtype=np.int32))

    def test_rule_is_the_same_in_both_modes(self):
        kam = random_kam(26, (9, 11))
        labels_correct = segment_grains_kam(kam, threshold=5.0)
        labels_compat = segment_grains_kam(kam, threshold=5.0, emsoft_compatible=True)
        assert labels_correct.max() > 0
        assert np.array_equal(labels_correct, labels_compat)


class TestSyntheticGrains:
    @pytest.mark.parametrize("n", [2, 3])
    def test_two_and_three_grain_maps_are_recovered_up_to_permutation(
        self, n, hrosm_grain_xmap
    ):
        xmap, truth = hrosm_grain_xmap(n_grains=n)
        kam = kernel_average_misorientation_map(xmap)
        labels = segment_grains_kam(kam, threshold=5.0)
        interior = interior_mask(truth)
        mapping = interior_bijection(labels, truth, interior)
        assert len(mapping) == n

        # Boundary points are unassigned or carry the label of their own
        # or an adjacent true grain
        H, W = truth.shape
        for y, x in zip(*np.nonzero(~interior)):
            allowed = {0, mapping[int(truth[y, x])]}
            for oy, ox in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                yn, xn = y + oy, x + ox
                if 0 <= yn < H and 0 <= xn < W:
                    allowed.add(mapping[int(truth[yn, xn])])
            assert labels[y, x] in allowed, (y, x)

    def test_boundary_below_four_thresholds_leaks(self, hrosm_grain_xmap):
        xmap, truth = hrosm_grain_xmap(n_grains=2, boundary_angle=15.0)
        kam = kernel_average_misorientation_map(xmap)
        labels = segment_grains_kam(kam, threshold=5.0)
        interior = interior_mask(truth)
        labels_a = np.unique(labels[interior & (truth == 1)])
        labels_b = np.unique(labels[interior & (truth == 2)])
        assert labels_a.size == labels_b.size == 1
        assert labels_a[0] != 0
        assert labels_a[0] == labels_b[0]

    def test_compat_kam_gives_the_same_interior_grains(self, hrosm_grain_xmap):
        xmap, truth = hrosm_grain_xmap(n_grains=2)
        # The generator's rows are identical, and EMsoft's unclipped
        # angle of two identical orientations may be its smallest
        # non-identity symmetry angle (90 deg), so a vertical gradient
        # of 0.1 deg per row is added, which keeps every boundary angle
        H, W = truth.shape
        rows = np.repeat(np.arange(H), W).astype(np.float64)
        half = np.deg2rad(rows * 0.1) / 2
        rx = Rotation(np.column_stack([np.cos(half), np.sin(half), 0 * half, 0 * half]))
        xmap = CrystalMap(
            rotations=rx * xmap.rotations,
            phase_id=xmap.phase_id,
            x=xmap.x,
            y=xmap.y,
            phase_list=xmap.phases,
            scan_unit="um",
        )
        interior = interior_mask(truth)
        labels_correct = segment_grains_kam(
            kernel_average_misorientation_map(xmap), threshold=5.0
        )
        labels_compat = segment_grains_kam(
            kernel_average_misorientation_map(xmap, emsoft_compatible=True),
            threshold=5.0,
        )
        # EMsoft compares the last point of the first row with the
        # identity and adds that angle to the first point and to the
        # last point of the first row
        compat_interior = interior.copy()
        compat_interior[0, 0] = compat_interior[0, W - 1] = False
        mapping_correct = interior_bijection(labels_correct, truth, interior)
        mapping_compat = interior_bijection(labels_compat, truth, compat_interior)
        assert mapping_compat == mapping_correct


class TestDilateAndBoxes:
    @pytest.mark.parametrize(
        "shape, seed",
        [(shape, 30 + i) for i, shape in enumerate(DILATE_SHAPES)],
        ids=[f"{s[0]}x{s[1]}-{30 + i}" for i, s in enumerate(DILATE_SHAPES)],
    )
    def test_compat_dilate_matches_the_emsoft_window_transcription(self, shape, seed):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 5, shape).astype(np.int32)
        expected = emsoft_grain_dilate(labels)
        assert np.array_equal(maximum_filter_dilate(labels), expected)
        dilated = _dilate_emsoft(labels)
        assert np.array_equal(dilated, expected)

        # Through the public function: EMsoft's growth, then its
        # dilation
        kam = rng.uniform(0, 12, shape).astype(np.float32)
        grown, _ = emsoft_grow_regions(kam, 5.0)
        labels_public = segment_grains_kam(
            kam, threshold=5.0, dilate=True, emsoft_compatible=True
        )
        assert labels_public.dtype == np.int32
        assert np.array_equal(labels_public, emsoft_grain_dilate(grown))

    def test_compat_dilate_skips_the_first_row_and_column_and_overwrites(self):
        labels = np.zeros((6, 7), dtype=np.int32)
        # Label 1 touches row 0 and column 0
        labels[0:2, 0:2] = 1
        # Label 3 is right of label 2
        labels[4:6, 3:5] = 2
        labels[4:6, 5:7] = 3
        dilated = _dilate_emsoft(labels)
        assert np.array_equal(dilated, emsoft_grain_dilate(labels))
        # Row 0 and column 0 are unchanged
        assert np.array_equal(dilated[0], labels[0])
        assert np.array_equal(dilated[:, 0], labels[:, 0])
        # Label 1 grows into (2, 1), (2, 2), (1, 2) but not beyond row 0
        # or column 0
        assert dilated[2, 1] == dilated[2, 2] == dilated[1, 2] == 1
        # Label 3 takes the column of label 2 next to it
        assert np.all(dilated[4:6, 4] == 3)
        assert np.all(dilated[4:6, 3] == 2)

    def test_compat_dilate_can_remove_a_grain(self, hrosm_gradient_xmap):
        shape = (5, 5)
        labels = np.full(shape, 2, dtype=np.int32)
        labels[2, 2:4] = 1
        dilated = _dilate_emsoft(labels)
        assert np.array_equal(dilated, emsoft_grain_dilate(labels))
        assert not np.any(dilated == 1)
        assert dilated.max() == 2

        boxes = grain_bounding_boxes(dilated)
        assert boxes.shape == (2, 4)
        assert np.array_equal(boxes[0], [0, 0, 0, 0])
        assert boxes[0, 2] == boxes[0, 3] == 0
        assert np.array_equal(boxes[1], [0, 0, 5, 5])

        xmap = hrosm_gradient_xmap(shape)
        grains = average_grain_orientations(xmap, dilated)
        assert grains.n_grains == 2
        assert grains.kappa[0] == -1.0
        assert not grains.valid[0]
        assert np.allclose(grains.rotation[0].data, Rotation.identity().data)
        assert grains.valid[1]

    def test_correct_dilate_fills_only_unassigned_pixels(self):
        labels = np.zeros((5, 7), dtype=np.int32)
        labels[0:2, 0:2] = 2
        labels[0:2, 3:5] = 3
        labels[1, 2] = 0
        labels[3, 1] = 1
        # An assigned label 2 next to the larger label 3
        labels[1, 5] = 2
        present = np.ones(labels.shape, dtype=bool)
        phase_id = np.zeros(labels.shape, dtype=int)
        dilated = _dilate_correct(labels, present, phase_id)
        assert np.array_equal(
            dilated, correct_dilate_reference(labels, present, phase_id)
        )
        # Assigned labels are unchanged
        assigned = labels != 0
        assert np.array_equal(dilated[assigned], labels[assigned])
        assert dilated[1, 5] == 2
        # An unassigned point next to labels 2 and 3 becomes 3
        assert dilated[1, 2] == 3
        # A point two steps from any label stays 0
        assert dilated[4, 6] == 0
        assert dilated[3, 6] == 0
        # Simultaneous update: (4, 3) is two steps from label 1 via the
        # newly filled (3, 2) and stays 0
        assert dilated[3, 2] == 1
        assert dilated[4, 3] == 0

        # Through the public function
        kam = random_kam(36, (9, 11))
        grown, _ = emsoft_grow_regions(kam, 5.0)
        labels_public = segment_grains_kam(kam, threshold=5.0, dilate=True)
        assert np.array_equal(
            labels_public,
            correct_dilate_reference(
                grown, np.ones(grown.shape, bool), np.zeros(grown.shape, int)
            ),
        )

    def test_correct_dilate_respects_phases(self):
        # The unassigned point (1, 2) of phase 1 ("ni2") has only phase
        # 0 ("ni") labels around it
        labels = np.zeros((3, 4), dtype=np.int32)
        labels[0:3, 0:2] = 1
        phase_id = np.zeros(labels.shape, dtype=int)
        phase_id[:, 2:] = 1
        present = np.ones(labels.shape, dtype=bool)
        dilated = _dilate_correct(labels, present, phase_id)
        assert np.array_equal(dilated, labels)

        # Through the public function
        kam = np.ones((1, 3), dtype=np.float32)
        labels_public = segment_grains_kam(
            kam, threshold=5.0, dilate=True, phase_id=np.array([[0, 0, 1]])
        )
        assert np.array_equal(labels_public, np.array([[1, 1, 0]], dtype=np.int32))

    def test_bounding_boxes_are_zero_based_row_col_height_width(self):
        labels = np.zeros((7, 6), dtype=np.int32)
        # An L-shaped grain at rows 2-5, columns 1-3
        labels[2:6, 1] = 1
        labels[5, 1:4] = 1
        labels[0, 4:6] = 2
        boxes = grain_bounding_boxes(labels)
        assert boxes.dtype == np.int64
        assert boxes.shape == (2, 4)
        assert np.array_equal(boxes[0], [2, 1, 4, 3])
        assert np.array_equal(boxes[1], [0, 4, 1, 2])

    def test_emsoft_grain_roi_converts_to_bounding_boxes(self):
        rng = np.random.default_rng(35)
        labels = rng.integers(0, 5, (7, 9)).astype(np.int32)
        assert np.array_equal(np.unique(labels), np.arange(5))
        roi = emsoft_get_roi(labels, 4)
        x0, y0, w, h = roi.T
        converted = np.column_stack([y0 - 1, x0 - 1, h, w])
        assert np.array_equal(grain_bounding_boxes(labels), converted)
