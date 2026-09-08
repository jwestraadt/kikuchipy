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

"""Tests of ``kikuchipy.indexing._hrebsd._kam``.

The KAM half of oracle V7 of
``specs/2026-09-07-hrebsd-dic/validation.md``, against the frozen
definition of requirements D12: the constant-curvature identity with
its kernel factor, the grain mask, the ``psi_max`` guard, the NaN
rule for points with no valid neighbour, and the milliradian unit.

**The kernel factor is geometry, not error.**  On a rotation field
which varies linearly along ONE grid axis, six of the eight order-1
neighbours differ by ``kappa * step`` and the two along the other
axis by zero, so the kernel mean is ``(6/8) * kappa * step`` exactly.
Asserting ``kappa * step`` would fail a correct implementation by 25
per cent, so every expectation below is computed from the KERNEL
OFFSETS, assembled in this module, and the ``6/8`` is asserted
separately as the closed form it works out to at an interior point.

The rotation field is synthetic throughout and its disorientations
are known by construction: rotations about ONE axis compose
additively, so a pair angle is a difference of angles.  One test
deliberately leaves that regime, with two rotations about different
axes, to pin that the pair angle is the angle of ``R_p R_q^T`` and
not the length of the difference of the two rotation vectors.

**The pair angle must NOT be computed as ``arccos((tr - 1) / 2)``
at this scale** (measured 2026-09-07, Stage B failing-tests review).
``arccos`` has a square-root singularity at the identity, so at the
1e-4 rad the HREBSD rotation field lives at it loses about eight
significant digits: the arccos oracle of an earlier draft returned
0.07499999980334485 mrad at the interior point where the closed form
is 0.075, a relative error of 2.6221e-09.  That made this module's
two tolerance families mutually unsatisfiable -- no implementation
could sit inside ``atol=1e-12`` of the oracle AND inside ``rel=1e-9``
of the ``(6/8) * kappa * step`` closed form at once.  The
well-conditioned form ``arctan2(||skew(M)|| / 2, (tr(M) - 1) / 2)``,
which :func:`pair_angle` uses below, reproduces both closed forms to
3.7e-16 relative, and the same warning belongs to the
implementation.  The commitment of requirements D12 is unchanged:
the pair angle is the rotation angle of ``R_p R_q^T``.

Written failing before the implementation, at the Stage B
failing-tests gate: every test which calls the module failed with
``NotImplementedError`` until HR-KAM landed, and passed unchanged after
it, while the signature and unit freezes passed from the start
(narration corrected to the past tense 2026-09-08, Stage B adversarial
review).
"""

import inspect
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._kam import (
    KAM_UNIT,
    RADIANS_TO_MRAD,
    hrebsd_kam,
    kernel_offsets,
)

# ------------------------- Frozen constants ------------------------- #

# The band of a machine-precision-class identity, FROZEN: everything
# here is analytic and nothing is measured against patterns
ALGEBRA_TOL = 1e-12

# The lattice curvature of the constant-curvature oracle, in radians
# per map step.  1e-4 rad per step is the HREBSD scale: two orders
# above the rotation noise floor and two below a Hough KAM
KAPPA_PER_STEP = 1e-4

# The map the identity is measured on: big enough for interior points
# to have a full kernel at order 2
SHAPE = (5, 6)


# ------------- The plan 3.4 mutation list, mapped ------------------- #
#
#  KAM mean over all pairs, cross-grain included .. TestGrainMask
#  KAM in radians reported as mrad ................ TestUnits


# ----------------------------- Helpers ------------------------------ #


def offsets_here(order):
    """Return the kernel offsets of requirements D12, written out
    here: every point within Chebyshev distance *order* of the
    centre, the centre excluded.  The MTEX-style "all within order"
    convention, not the OIM perimeter-only one."""
    return [
        (drow, dcol)
        for drow in range(-order, order + 1)
        for dcol in range(-order, order + 1)
        if (drow, dcol) != (0, 0)
    ]


def kam_map(rotation_vectors, grain_id=None, navigation_shape=SHAPE):
    """Return a crystal map carrying the two properties HR-KAM
    consumes: the ``rotation_vector`` of the tensor chain and the
    ``grain_id`` of the engine."""
    arrays, size = create_coordinate_arrays(navigation_shape, (1.0, 1.0))
    arrays["rotations"] = Rotation.identity((size,))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    xmap.prop["rotation_vector"] = np.asarray(
        rotation_vectors, dtype=np.float64
    ).reshape(size, 3)
    if grain_id is None:
        grain_id = np.zeros(size, dtype=np.int32)
    xmap.prop["grain_id"] = np.asarray(grain_id, dtype=np.int32).ravel()
    return xmap


def linear_field(navigation_shape=SHAPE, kappa=KAPPA_PER_STEP):
    """Return the constant-curvature rotation field of requirements
    D12: a rotation about ``z`` growing linearly along the COLUMN
    axis, ``omega_3(x1) = kappa * x1`` with ``x1`` in map steps."""
    ny, nx = navigation_shape
    _, cols = np.indices((ny, nx), dtype=np.float64)
    field = np.zeros((ny, nx, 3))
    field[:, :, 2] = kappa * cols
    return field


def row_linear_field(navigation_shape, kappa=KAPPA_PER_STEP):
    """Return the same constant-curvature field varying along the ROW
    axis instead, which is the only way a single-COLUMN map carries
    any misorientation at all."""
    ny, nx = navigation_shape
    rows, _ = np.indices((ny, nx), dtype=np.float64)
    field = np.zeros((ny, nx, 3))
    field[:, :, 2] = kappa * rows
    return field


def rotation_matrix_of(vector):
    """Return the rotation matrix of a rotation vector, Rodrigues'
    formula written out here."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    axis = vector / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def pair_angle(first, second):
    """Return the disorientation of two rotation vectors in
    MILLIRADIANS, as the angle of ``R_p R_q^T`` (requirements D12),
    assembled here from the matrices.

    Computed as ``arctan2`` of the skew part against the trace part
    and NOT as ``arccos((tr - 1) / 2)``: see the module docstring for
    the measurement.  At the 1e-4 rad scale of this module the arccos
    form loses eight digits, which is enough to break the identities
    below against their own closed forms.
    """
    matrix = rotation_matrix_of(first) @ rotation_matrix_of(second).T
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    cosine = np.clip(0.5 * (np.trace(matrix) - 1.0), -1.0, 1.0)
    return float(np.arctan2(0.5 * np.linalg.norm(axis), cosine)) * 1e3


def expected_kam(field, grain_id, order=1, psi_max=None):
    """Return the HR-KAM of a field, assembled here from the KERNEL
    OFFSETS point by point.

    The oracle of this module: the expectation is never
    ``kappa * step``, which would be wrong by the fixed kernel factor,
    and never a second call to the code under test.
    """
    ny, nx = field.shape[:2]
    grain_id = np.asarray(grain_id).reshape(ny, nx)
    out = np.full((ny, nx), np.nan)
    for row in range(ny):
        for col in range(nx):
            if grain_id[row, col] < 0 or not np.all(np.isfinite(field[row, col])):
                continue
            angles = []
            for drow, dcol in offsets_here(order):
                other_row, other_col = row + drow, col + dcol
                if not (0 <= other_row < ny and 0 <= other_col < nx):
                    continue
                if grain_id[other_row, other_col] != grain_id[row, col]:
                    continue
                if not np.all(np.isfinite(field[other_row, other_col])):
                    continue
                angle = pair_angle(field[row, col], field[other_row, other_col])
                if psi_max is not None and angle > psi_max:
                    continue
                angles.append(angle)
            if angles:
                out[row, col] = float(np.mean(angles))
    return out


# ===================== D12 -- the frozen kernel ===================== #


class TestKernel:
    """Every point within Chebyshev distance ``order``, the MTEX-style
    convention.  [D12]"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_the_offsets_are_the_chebyshev_ball(self, order):
        got = kernel_offsets(order)
        assert got.shape == (4 * order * (order + 1), 2)
        assert np.issubdtype(got.dtype, np.integer)
        assert sorted(map(tuple, got.tolist())) == sorted(offsets_here(order))

    def test_order_one_is_eight_neighbours_not_four(self):
        # the perimeter-only convention of OIM would give four here,
        # and the whole ``6/8`` identity of requirements D12 depends
        # on it being eight
        assert kernel_offsets(1).shape[0] == 8

    def test_the_centre_is_excluded(self):
        assert not np.any(np.all(kernel_offsets(2) == 0, axis=1))

    def test_guards(self):
        with pytest.raises(ValueError):
            kernel_offsets(0)
        with pytest.raises(ValueError):
            kernel_offsets(-1)


class TestConstantCurvature:
    """THE identity of requirements D12 and validation V7.  [D12]"""

    def test_the_oracle_itself_reproduces_the_closed_forms(self):
        # A LIBRARY MEASUREMENT of this module's OWN oracle, so it
        # passes today, and the reason it exists is the measurement in
        # the module docstring: an ``arccos`` pair angle puts
        # ``expected_kam`` 2.6221e-09 relative away from the closed
        # form, which is 2600 times the ``rel=1e-9`` the tests below
        # assert against that same closed form while comparing the
        # implementation with the oracle at ``atol=1e-12``.  The two
        # families are then unsatisfiable together.  This guards the
        # well-conditioned form: if it is ever reverted, THIS test
        # fails first and names the cause instead of the failure
        # surfacing as an unexplained implementation error
        expected = expected_kam(linear_field(), np.zeros(SHAPE, dtype=np.int32))
        assert expected[2, 3] == pytest.approx(
            (6 / 8) * KAPPA_PER_STEP * RADIANS_TO_MRAD, rel=1e-12
        )
        assert expected[0, 0] == pytest.approx(
            (2 / 3) * KAPPA_PER_STEP * RADIANS_TO_MRAD, rel=1e-12
        )

    def test_the_kernel_derived_expectation_holds_everywhere(self):
        field = linear_field()
        xmap = kam_map(field)
        got = hrebsd_kam(xmap)
        assert got.shape == SHAPE
        assert got.dtype == np.float64
        expected = expected_kam(field, np.zeros(SHAPE, dtype=np.int32))
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_an_interior_point_is_the_six_eighths_closed_form(self):
        # the closed form of requirements D12, asserted separately
        # from the offsets it comes from: six of the eight order-1
        # neighbours differ by ``kappa * step`` and two by zero
        field = linear_field()
        got = hrebsd_kam(kam_map(field))
        closed_form = (6 / 8) * KAPPA_PER_STEP * RADIANS_TO_MRAD
        assert got[2, 3] == pytest.approx(closed_form, rel=1e-9)
        # and the naive expectation is 25 per cent away, which is why
        # V7 forbids asserting it
        assert not np.isclose(got[2, 3], KAPPA_PER_STEP * RADIANS_TO_MRAD, rtol=1e-3)

    def test_edge_points_average_over_fewer_neighbours(self):
        # the corner sees three neighbours, two of which differ, so
        # its own kernel-derived value is 2/3 of the step and not 6/8
        field = linear_field()
        got = hrebsd_kam(kam_map(field))
        assert got[0, 0] == pytest.approx(
            (2 / 3) * KAPPA_PER_STEP * RADIANS_TO_MRAD, rel=1e-9
        )
        assert np.all(np.isfinite(got))

    def test_order_two_uses_the_bigger_kernel(self):
        field = linear_field()
        xmap = kam_map(field)
        got = hrebsd_kam(xmap, order=2)
        expected = expected_kam(field, np.zeros(SHAPE, dtype=np.int32), order=2)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)
        # an interior point of the order-2 kernel averages
        # ``(20/24) * 2`` half-steps, so the two orders really do
        # differ and one cannot stand in for the other
        assert not np.isclose(got[2, 3], hrebsd_kam(xmap)[2, 3])

    def test_a_uniform_field_has_no_misorientation(self):
        field = np.zeros(SHAPE + (3,))
        field[:, :, 2] = 3e-4
        got = hrebsd_kam(kam_map(field))
        np.testing.assert_allclose(got, np.zeros(SHAPE), rtol=0, atol=ALGEBRA_TOL)

    def test_the_pair_angle_is_the_composed_rotation(self):
        # requirements D12 defines the disorientation as the angle of
        # ``R_p R_q^T``, which is NOT the length of the difference of
        # the two rotation vectors unless the axes are parallel.  Both
        # candidates are computed here: at 0.1 rad about perpendicular
        # axes they sit 141.392 against 141.421 mrad, 0.029 mrad
        # apart, which is ten orders above the comparison band and
        # what a "subtract the rotation vectors" mutant would report
        first = np.array([0.1, 0.0, 0.0])
        second = np.array([0.0, 0.1, 0.0])
        field = np.stack([first, second]).reshape(1, 2, 3)
        composed = pair_angle(first, second)
        naive = float(np.linalg.norm(first - second)) * RADIANS_TO_MRAD
        assert abs(composed - naive) > 1e-2
        got = hrebsd_kam(kam_map(field, navigation_shape=(1, 2)))
        assert got[0, 0] == pytest.approx(composed, rel=1e-9)
        assert got[0, 1] == pytest.approx(composed, rel=1e-9)


class TestShapeContract:
    """The returned map carries the two dimensional navigation shape
    whatever orix reports for a flattened map.  [D12/D15]"""

    def test_orix_flattens_a_one_row_and_a_one_column_map(self):
        # A LIBRARY MEASUREMENT, so it passes today, and the reason
        # the two pins below exist: on the installed orix a crystal
        # map of navigation shape ``(1, n)`` or ``(n, 1)`` has
        # ``xmap.shape == (n,)`` and ``ndim == 1``, so an
        # implementation which returns ``xmap.shape`` gives a ONE
        # dimensional KAM map.  Every other test here indexes
        # ``got[row, col]``, which would then raise ``IndexError``
        # rather than fail an assertion, and the shape contract of the
        # docstring ("the map's navigation shape ``(ny, nx)``") would
        # be required only implicitly.  The row and column grids do
        # carry the distinction, which is what an implementation reads
        row = kam_map(linear_field((1, 4)), navigation_shape=(1, 4))
        assert row.shape == (4,)
        assert row.ndim == 1
        np.testing.assert_array_equal(np.asarray(row.row), [0, 0, 0, 0])
        np.testing.assert_array_equal(np.asarray(row.col), [0, 1, 2, 3])
        column = kam_map(row_linear_field((3, 1)), navigation_shape=(3, 1))
        assert column.shape == (3,)
        np.testing.assert_array_equal(np.asarray(column.row), [0, 1, 2])
        np.testing.assert_array_equal(np.asarray(column.col), [0, 0, 0])

    def test_a_one_row_map_returns_one_row(self):
        field = linear_field((1, 4))
        got = hrebsd_kam(kam_map(field, navigation_shape=(1, 4)))
        assert got.shape == (1, 4)
        expected = expected_kam(field, np.zeros((1, 4), dtype=np.int32))
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_a_one_column_map_returns_one_column(self):
        # the companion of the column case of
        # ``test_hrebsd_segmentation.py``: a map whose points lie in
        # one COLUMN is a column and not a row, which the row and
        # column grids settle.  Both diagnostics have to agree on it,
        # because ``_reference._flatten_labels`` compares the
        # segmentation's shape with the navigation shape EXACTLY
        field = row_linear_field((3, 1))
        got = hrebsd_kam(kam_map(field, navigation_shape=(3, 1)))
        assert got.shape == (3, 1)
        expected = expected_kam(field, np.zeros((3, 1), dtype=np.int32))
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)
        # and it really carries misorientation, so the shape is not
        # asserted on a map of zeros
        assert np.nanmax(got) > 0.0


class TestUnits:
    """Milliradians, frozen.  [D12]"""

    def test_the_map_is_in_milliradians(self):
        field = linear_field()
        got = hrebsd_kam(kam_map(field))
        in_radians = (6 / 8) * KAPPA_PER_STEP
        assert got[2, 3] == pytest.approx(in_radians * 1e3, rel=1e-9)
        # the radians-reported-as-milliradians mutant, excluded by
        # three orders
        assert not np.isclose(got[2, 3], in_radians)
        assert KAM_UNIT == "mrad"
        assert RADIANS_TO_MRAD == 1e3

    def test_psi_max_is_in_milliradians_too(self):
        # the same unit for the guard: at this curvature a step is
        # 0.1 mrad, so a threshold of 0.05 drops every horizontal pair
        # and keeps the vertical ones.  Read as RADIANS the same
        # number would drop nothing, and the map would be unchanged
        field = linear_field()
        xmap = kam_map(field)
        unguarded = hrebsd_kam(xmap)
        guarded = hrebsd_kam(xmap, psi_max=0.05)
        np.testing.assert_allclose(guarded, np.zeros(SHAPE), rtol=0, atol=ALGEBRA_TOL)
        assert not np.allclose(guarded, unguarded)


class TestPsiMaxBoundary:
    """A pair sitting EXACTLY at ``psi_max`` is KEPT.  [D12]

    ADDED 2026-09-08 at the Stage B adversarial review, where the
    ``<=`` -> ``<`` mutant of this comparison SURVIVED the whole
    suite.  It is the KAM analogue of the segmentation threshold side,
    which ``test_hrebsd_segmentation.py::TestThreshold`` does pin, and
    requirements D12 fixes it: *psi_max* "additionally drops pairs
    ABOVE the threshold", so a pair AT it survives.

    The threshold is read back from a ONE PAIR map, where the reported
    KAM is that single pair's angle and nothing is averaged, so the
    boundary is exact rather than a tolerance; the value is
    cross-checked against this module's own oracle before it is used.
    The discrimination is total: the correct form returns the angle,
    the mutant returns NaN.
    """

    @staticmethod
    def one_pair_map(step=KAPPA_PER_STEP):
        """Return a ``(1, 2)`` map whose single pair is the only
        misorientation on it."""
        field = np.zeros((1, 2, 3))
        field[0, 1, 2] = step
        return kam_map(field, navigation_shape=(1, 2)), field

    def test_the_threshold_is_this_modules_own_angle(self):
        xmap, field = self.one_pair_map()
        reported = hrebsd_kam(xmap)
        assert reported.shape == (1, 2)
        angle = pair_angle(field[0, 0], field[0, 1])
        assert angle > 0.0
        np.testing.assert_allclose(reported, np.full((1, 2), angle), atol=ALGEBRA_TOL)

    def test_a_pair_exactly_at_psi_max_is_kept(self):
        xmap, _ = self.one_pair_map()
        unguarded = hrebsd_kam(xmap)
        threshold = float(unguarded[0, 0])
        assert threshold > 0.0
        guarded = hrebsd_kam(xmap, psi_max=threshold)
        # KEPT, so the map is unchanged; the ``<`` mutant drops the
        # only pair there is and returns NaN everywhere
        assert np.array_equal(guarded, unguarded, equal_nan=True)
        assert np.all(np.isfinite(guarded))

    def test_a_pair_just_above_psi_max_is_dropped(self):
        # the complement, without which the test above would pass on a
        # ``psi_max`` that is read and then ignored
        xmap, _ = self.one_pair_map()
        threshold = float(hrebsd_kam(xmap)[0, 0])
        guarded = hrebsd_kam(xmap, psi_max=np.nextafter(threshold, 0.0))
        assert np.all(np.isnan(guarded))


class TestGrainMask:
    """Pairs must share ``grain_id``, always.  [D12]"""

    def test_a_boundary_pair_never_enters_the_mean(self):
        # the left half and the right half are two grains with a large
        # jump between them; a KAM which averaged over all pairs would
        # show that jump at the boundary column
        field = linear_field()
        field[:, 3:, 2] += 5e-2
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[:, 3:] = 1
        got = hrebsd_kam(kam_map(field, grain_id))
        expected = expected_kam(field, grain_id)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)
        # every value stays at the within-grain scale, three orders
        # below the boundary jump
        assert np.nanmax(got) < 1.0
        assert 5e-2 * RADIANS_TO_MRAD > 10.0

    def test_a_point_outside_every_grain_is_nan(self):
        field = linear_field()
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[2, 3] = -1
        got = hrebsd_kam(kam_map(field, grain_id))
        assert np.isnan(got[2, 3])
        # and its neighbours simply drop the pair rather than
        # fabricating a zero
        expected = expected_kam(field, grain_id)
        np.testing.assert_allclose(got[2, 2], expected[2, 2], rtol=0, atol=ALGEBRA_TOL)

    def test_a_one_point_grain_has_no_neighbour_and_is_nan(self):
        field = linear_field()
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[2, 3] = 7
        got = hrebsd_kam(kam_map(field, grain_id))
        assert np.isnan(got[2, 3])
        assert np.isfinite(got[2, 2])


class TestNaNSafety:
    """Non-converged points are dropped, never fabricated.  [D12]"""

    def test_a_nan_point_is_nan_and_drops_its_pairs(self):
        field = linear_field()
        field[1, 1] = np.nan
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        got = hrebsd_kam(kam_map(field, grain_id))
        assert np.isnan(got[1, 1])
        expected = expected_kam(field, grain_id)
        np.testing.assert_allclose(
            got[np.isfinite(expected)],
            expected[np.isfinite(expected)],
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        # the neighbour still has seven other pairs, so it is a value
        # and not a NaN
        assert np.isfinite(got[1, 2])

    def test_an_all_nan_map_is_all_nan(self):
        field = np.full(SHAPE + (3,), np.nan)
        got = hrebsd_kam(kam_map(field))
        assert np.all(np.isnan(got))


class TestContract:
    """Signature, guards and determinism.  [D12/D15]"""

    def test_signature_is_frozen(self):
        parameters = inspect.signature(hrebsd_kam).parameters
        assert list(parameters) == ["xmap", "order", "psi_max"]
        assert parameters["xmap"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["order"].default == 1
        assert parameters["psi_max"].default is None
        for name in ("order", "psi_max"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name

    def test_two_runs_are_bitwise_identical(self):
        xmap = kam_map(linear_field())
        first, second = hrebsd_kam(xmap), hrebsd_kam(xmap)
        assert np.array_equal(first, second, equal_nan=True)

    def test_it_returns_an_array_and_stores_nothing(self):
        # requirements D15.6: the three diagnostics return arrays and
        # never add properties to the map
        xmap = kam_map(linear_field())
        before = set(xmap.prop)
        got = hrebsd_kam(xmap)
        assert isinstance(got, np.ndarray)
        assert set(xmap.prop) == before

    def test_guards(self):
        xmap = kam_map(linear_field())
        stripped = kam_map(linear_field())
        del stripped.prop["rotation_vector"]
        with pytest.raises(ValueError, match="hrebsd_strain_stress"):
            hrebsd_kam(stripped)
        without_grains = kam_map(linear_field())
        del without_grains.prop["grain_id"]
        with pytest.raises(ValueError, match="hrebsd_dic"):
            hrebsd_kam(without_grains)
        with pytest.raises(ValueError, match="order"):
            hrebsd_kam(xmap, order=0)
        with pytest.raises(ValueError, match="psi_max"):
            hrebsd_kam(xmap, psi_max=0.0)

    @pytest.mark.parametrize("order", [1.0, 1.5, "1", True, None])
    def test_a_non_integer_order_is_refused(self, order):
        # ADDED 2026-09-08 (Stage B adversarial review, the coverage
        # gate): the type half of the ``order`` guard had no test, and
        # a float order would otherwise silently build a kernel from
        # ``range`` and fail somewhere less legible.  ``True`` is an
        # integer to Python and is excluded on purpose
        with pytest.raises(ValueError, match="positive integer"):
            hrebsd_kam(kam_map(linear_field()), order=order)

    def test_a_map_without_a_grid_is_named(self):
        # ADDED 2026-09-08 (Stage B adversarial review): orix gives a
        # map whose points share one scan position the shape ``()``,
        # and reading its row grid raises "not enough values to
        # unpack", which names nothing the caller passed
        xmap = CrystalMap(
            rotations=Rotation.identity((2,)),
            phase_id=np.zeros(2, dtype=int),
            phase_list=PhaseList(Phase(name="ni", space_group=225)),
            x=np.zeros(2),
            y=np.zeros(2),
        )
        xmap.prop["rotation_vector"] = np.zeros((2, 3))
        xmap.prop["grain_id"] = np.zeros(2, dtype=np.int32)
        with pytest.raises(ValueError, match="no map grid"):
            hrebsd_kam(xmap)

    def test_a_float_grain_id_carrying_nan_is_unlabelled(self):
        # ADDED 2026-09-08 (Stage B adversarial review).  The
        # documented property is int32, but a float one carrying NaN
        # used to be cast straight to int64, which is undefined and
        # merely HAPPENED to come out negative on this platform.  The
        # NaN point must be unlabelled, and quietly
        field = linear_field()
        grain_id = np.zeros(SHAPE, dtype=np.float64)
        grain_id[2, 3] = np.nan
        xmap = kam_map(field)
        xmap.prop["grain_id"] = grain_id.ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = hrebsd_kam(xmap)
        assert np.isnan(got[2, 3])
        expected = np.zeros(SHAPE, dtype=np.int32)
        expected[2, 3] = -1
        np.testing.assert_allclose(
            got, expected_kam(field, expected), rtol=0, atol=ALGEBRA_TOL
        )

    def test_the_public_name_is_the_module_one(self):
        assert kp.indexing.hrebsd_kam is hrebsd_kam
