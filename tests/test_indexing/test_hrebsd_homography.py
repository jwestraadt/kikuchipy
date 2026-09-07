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

"""Tests of ``kikuchipy.indexing._hrebsd._homography``.

Covers oracle V1 of ``specs/2026-09-07-hrebsd-dic/validation.md``,
the algebraic half: the shape function layout, its group algebra
(compose, invert, project, the corner norm), the closed forms of the
pure cases, and the exact homography to reduced deformation gradient
round trip of requirements D6.

**Every band here is 1e-12 and FROZEN, never measured**: these are
machine-precision-class algebraic identities on 64-bit floats, and a
measured tolerance would hide a wrong formula rather than reveal one.
The warp DIRECTION pin of V1 needs a pattern and lives in
``test_hrebsd_engine.py`` beside the warp-refit oracle.

Written before the implementation exists: every test here calls the
skeleton and therefore fails with ``NotImplementedError`` until the
algebra lands, then passes unchanged.
"""

import numpy as np
import pytest

from kikuchipy.indexing._hrebsd._homography import (
    IDENTITY_HOMOGRAPHY,
    N_HOMOGRAPHY_PARAMETERS,
    compose,
    corner_norm,
    error_norm,
    fe_to_homography,
    homography_parameters,
    homography_to_fe,
    invert,
    project,
    shape_function,
)

# ------------------------- Frozen constants ------------------------- #

# The algebraic identity band, FROZEN (validation V1: "machine
# precision class, no MTP")
ALGEBRA_TOL = 1e-12

# A representative detector distance in binned pixels, 480 px
# patterns with the Bruker ``pcz`` around 0.5
DD = 240.0

# Subregion corners in PC-centred binned pixels, the support of the
# D2.5 convergence norm.  Deliberately NOT symmetric about the
# origin: a PC-centred subregion of a real detector never is, and a
# symmetric box hides sign errors that cancel in pairs
CORNERS = np.array([[-200.0, -100.0], [216.0, -100.0], [216.0, 340.0], [-200.0, 340.0]])


# ----------------------------- Helpers ------------------------------ #


def random_homographies(n=32, seed=0):
    """Return *n* random small homographies of the scale the engine
    actually meets: 2e-2 in the linear terms, a few pixels of
    translation and 1e-4 per pixel of perspective."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n, N_HOMOGRAPHY_PARAMETERS))
    h[:, [0, 1, 3, 4]] = rng.uniform(-2e-2, 2e-2, size=(n, 4))
    h[:, [2, 5]] = rng.uniform(-5.0, 5.0, size=(n, 2))
    h[:, [6, 7]] = rng.uniform(-1e-4, 1e-4, size=(n, 2))
    return h


def random_reduced_fe(n=32, seed=1):
    """Return *n* random reduced deformation gradients, that is
    ``Fe / Fe33``, of a physically plausible size."""
    rng = np.random.default_rng(seed)
    fe = np.eye(3) + rng.uniform(-2e-2, 2e-2, size=(n, 3, 3))
    return fe / fe[:, 2, 2][:, None, None]


def random_pc_rel(n=32, seed=2):
    """Return *n* random relative projection centre offsets in binned
    pixels, the ``PC_target - PC_reference`` of requirements D6."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-20.0, 20.0, size=(n, 2))


def apply_matrix(matrix, x, y):
    """Return the projective image of ``(x, y)`` under an explicit
    3 by 3 matrix, the independent reference of :func:`project`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    numerator_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    numerator_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return numerator_x / denominator, numerator_y / denominator


# ==================== The shape function layout ===================== #


class TestShapeFunction:
    """The frozen textbook layout of requirements D2, the form
    EMsoftOO comments out.  Its inverted-diagonal variant with the
    global sign flip (``mod_DIC.f90:888-934``) is a recorded
    deviation, never reproduced.  [D2]"""

    def test_layout_is_literal(self):
        # the transposed-shape-function mutant of plan 2.5 dies here
        h = np.array([0.11, 0.12, 13.0, 0.21, 0.22, 23.0, 0.31, 0.32])
        expected = np.array(
            [
                [1 + 0.11, 0.12, 13.0],
                [0.21, 1 + 0.22, 23.0],
                [0.31, 0.32, 1.0],
            ]
        )
        got = shape_function(h)
        assert got.shape == (3, 3)
        assert got.dtype == np.float64
        assert np.array_equal(got, expected)

    def test_identity_is_the_zero_vector(self):
        assert len(IDENTITY_HOMOGRAPHY) == N_HOMOGRAPHY_PARAMETERS
        assert N_HOMOGRAPHY_PARAMETERS == 8
        assert np.array_equal(
            shape_function(np.asarray(IDENTITY_HOMOGRAPHY)), np.eye(3)
        )

    def test_parameters_round_trip(self):
        for h in random_homographies(seed=10):
            got = homography_parameters(shape_function(h))
            np.testing.assert_allclose(got, h, rtol=ALGEBRA_TOL, atol=0)

    def test_parameters_renormalize_w33(self):
        # the ``W[2, 2]`` renormalization of D2.3, and its idempotence
        h = random_homographies(n=1, seed=11)[0]
        matrix = shape_function(h)
        for scale in (2.0, -3.5, 1e-6):
            got = homography_parameters(scale * matrix)
            np.testing.assert_allclose(got, h, rtol=ALGEBRA_TOL, atol=0)
        once = homography_parameters(matrix)
        twice = homography_parameters(shape_function(once))
        np.testing.assert_allclose(twice, once, rtol=ALGEBRA_TOL, atol=0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            shape_function(np.zeros(7))
        with pytest.raises(ValueError):
            homography_parameters(np.zeros((2, 2)))

    def test_singular_matrix_raises(self):
        matrix = np.eye(3)
        matrix[2, 2] = 0.0
        with pytest.raises(ValueError):
            homography_parameters(matrix)


# ======================= V1 -- group algebra ======================== #


class TestGroupAlgebra:
    """Compose, invert and project form the projective group the
    IC-GN update walks in.  [D2/V1]"""

    def test_compose_is_the_matrix_product(self):
        # the order matters: the RIGHT factor acts on the coordinates
        # first, which is what makes ``compose(h, invert(dp))`` the
        # inverse-compositional update and ``compose(invert(dp), h)``
        # the mutant of plan 2.5
        a, b = random_homographies(n=2, seed=20)
        expected = shape_function(a) @ shape_function(b)
        expected = expected / expected[2, 2]
        got = shape_function(compose(a, b))
        np.testing.assert_allclose(got, expected, rtol=ALGEBRA_TOL, atol=0)
        # and the two orders are genuinely different here
        other = shape_function(compose(b, a))
        assert not np.allclose(got, other)

    def test_compose_invert_closure(self):
        identity = np.asarray(IDENTITY_HOMOGRAPHY)
        for h in random_homographies(seed=21):
            forward = compose(h, invert(h))
            backward = compose(invert(h), h)
            np.testing.assert_allclose(forward, identity, rtol=0, atol=ALGEBRA_TOL)
            np.testing.assert_allclose(backward, identity, rtol=0, atol=ALGEBRA_TOL)

    def test_compose_is_associative(self):
        a, b, c = random_homographies(n=3, seed=22)
        left = compose(compose(a, b), c)
        right = compose(a, compose(b, c))
        np.testing.assert_allclose(left, right, rtol=ALGEBRA_TOL, atol=0)

    def test_invert_is_an_involution(self):
        for h in random_homographies(n=8, seed=23):
            np.testing.assert_allclose(invert(invert(h)), h, rtol=ALGEBRA_TOL, atol=0)

    def test_project_matches_the_matrix(self):
        h = random_homographies(n=1, seed=24)[0]
        x, y = CORNERS[:, 0], CORNERS[:, 1]
        got_x, got_y = project(h, x, y)
        expected_x, expected_y = apply_matrix(shape_function(h), x, y)
        np.testing.assert_allclose(got_x, expected_x, rtol=ALGEBRA_TOL, atol=0)
        np.testing.assert_allclose(got_y, expected_y, rtol=ALGEBRA_TOL, atol=0)

    def test_project_keeps_the_projective_divide(self):
        # dropping the divide by ``s`` turns the homography into an
        # affinity; with a perspective term of 1e-4 per pixel over a
        # 200 px arm the two differ by several pixels, far above any
        # band in this suite (the mutant of plan 2.5)
        h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        h[6] = 1e-4
        h[7] = -5e-5
        x, y = CORNERS[:, 0], CORNERS[:, 1]
        got_x, got_y = project(h, x, y)
        matrix = shape_function(h)
        affine_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        affine_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        assert np.abs(got_x - affine_x).max() > 1e-3
        assert np.abs(got_y - affine_y).max() > 1e-3

    def test_project_composition_identity(self):
        # projecting through a composition equals projecting twice,
        # the identity the accumulated warp of D2.3 relies on
        a, b = random_homographies(n=2, seed=25)
        x, y = CORNERS[:, 0], CORNERS[:, 1]
        once_x, once_y = project(b, x, y)
        twice_x, twice_y = project(a, once_x, once_y)
        both_x, both_y = project(compose(a, b), x, y)
        np.testing.assert_allclose(both_x, twice_x, rtol=ALGEBRA_TOL, atol=0)
        np.testing.assert_allclose(both_y, twice_y, rtol=ALGEBRA_TOL, atol=0)


class TestCornerNorm:
    """The frozen convergence and recovery metric of D2.5: one
    displacement in binned pixels, unit consistent across all eight
    degrees of freedom.  [D2/V2]"""

    def test_identity_has_zero_norm(self):
        got = corner_norm(np.asarray(IDENTITY_HOMOGRAPHY), CORNERS)
        assert got == 0.0

    def test_pure_translation_is_its_own_length(self):
        dx, dy = 0.3, -0.4
        h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        h[2] = dx
        h[5] = dy
        expected = float(np.hypot(dx, dy))
        got = corner_norm(h, CORNERS)
        assert abs(got - expected) <= ALGEBRA_TOL * expected

    def test_is_the_maximum_over_the_corners(self):
        h = random_homographies(n=1, seed=30)[0]
        x, y = CORNERS[:, 0], CORNERS[:, 1]
        warped_x, warped_y = project(h, x, y)
        expected = float(np.hypot(warped_x - x, warped_y - y).max())
        got = corner_norm(h, CORNERS)
        assert abs(got - expected) <= ALGEBRA_TOL * expected

    def test_error_norm_is_the_error_warp_norm(self):
        # the V2 recovery metric, spelled out: the corner norm of
        # ``W(h_true)**-1 . W(h_fit)``
        h_true, h_fit = random_homographies(n=2, seed=31)
        expected = corner_norm(compose(invert(h_true), h_fit), CORNERS)
        got = error_norm(h_fit, h_true, CORNERS)
        assert abs(got - expected) <= ALGEBRA_TOL * max(expected, 1.0)

    def test_error_norm_is_zero_for_an_exact_fit(self):
        h = random_homographies(n=1, seed=32)[0]
        assert error_norm(h, h, CORNERS) <= ALGEBRA_TOL


# ============ V1 -- homography to deformation gradient ============== #


class TestHomographyToFe:
    """The exact conversion of requirements D6 in PC-centred binned
    pixels, and its inverse.  [D6/V1]"""

    def test_formula_is_literal(self):
        # every scaling of the frozen formula pinned by hand: the
        # ``Fe13 not divided by DD`` and ``Fe31 not multiplied by DD``
        # mutants of plan 2.5 die here
        h = np.array([0.011, 0.012, 3.0, 0.021, 0.022, -4.0, 1e-4, -2e-4])
        pc_rel = np.array([12.0, -7.0])
        beta0 = 1.0 - h[6] * pc_rel[0] - h[7] * pc_rel[1]
        expected = (
            np.array(
                [
                    [1 + h[0], h[1], h[2] / DD],
                    [h[3], 1 + h[4], h[5] / DD],
                    [DD * h[6], DD * h[7], beta0],
                ]
            )
            / beta0
        )
        got = homography_to_fe(h, pc_rel, DD)
        assert got.shape == (3, 3)
        np.testing.assert_allclose(got, expected, rtol=ALGEBRA_TOL, atol=0)

    def test_reduced_tensor_has_unit_fe33(self):
        for h in random_homographies(n=8, seed=40):
            fe = homography_to_fe(h, np.zeros(2), DD)
            assert abs(fe[2, 2] - 1.0) <= ALGEBRA_TOL

    def test_beta0_is_one_at_the_frame_origin(self):
        # the reference's own projection centre IS the origin (D1.3),
        # so its conversion has no cross terms at all
        h = random_homographies(n=1, seed=41)[0]
        fe = homography_to_fe(h, np.zeros(2), DD)
        assert abs(fe[0, 0] - (1 + h[0])) <= ALGEBRA_TOL
        assert abs(fe[0, 2] - h[2] / DD) <= ALGEBRA_TOL
        assert abs(fe[2, 0] - DD * h[6]) <= ALGEBRA_TOL

    @pytest.mark.parametrize("with_pc_rel", [False, True])
    def test_h_fe_round_trip(self, with_pc_rel):
        # validation V1: exact to 1e-12 in BOTH directions, with and
        # without the corner-origin cross terms
        h_all = random_homographies(seed=42)
        pc_all = random_pc_rel(n=h_all.shape[0], seed=43)
        for i, h in enumerate(h_all):
            pc_rel = pc_all[i] if with_pc_rel else np.zeros(2)
            fe = homography_to_fe(h, pc_rel, DD)
            back = fe_to_homography(fe, pc_rel, DD)
            np.testing.assert_allclose(back, h, rtol=ALGEBRA_TOL, atol=0)

    @pytest.mark.parametrize("with_pc_rel", [False, True])
    def test_fe_h_round_trip(self, with_pc_rel):
        fe_all = random_reduced_fe(seed=44)
        pc_all = random_pc_rel(n=fe_all.shape[0], seed=45)
        for i, fe in enumerate(fe_all):
            pc_rel = pc_all[i] if with_pc_rel else np.zeros(2)
            h = fe_to_homography(fe, pc_rel, DD)
            back = homography_to_fe(h, pc_rel, DD)
            np.testing.assert_allclose(back, fe, rtol=ALGEBRA_TOL, atol=0)

    def test_fe_to_homography_reduces_first(self):
        # any scaling of the tensor is the same reduced tensor
        fe = random_reduced_fe(n=1, seed=46)[0]
        base = fe_to_homography(fe, np.zeros(2), DD)
        for scale in (2.0, -0.25):
            got = fe_to_homography(scale * fe, np.zeros(2), DD)
            np.testing.assert_allclose(got, base, rtol=ALGEBRA_TOL, atol=0)

    def test_invalid_inputs_raise(self):
        h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        with pytest.raises(ValueError):
            homography_to_fe(h, np.zeros(2), 0.0)
        with pytest.raises(ValueError):
            fe_to_homography(np.eye(2), np.zeros(2), DD)
        singular = np.eye(3)
        singular[2, 2] = 0.0
        with pytest.raises(ValueError):
            fe_to_homography(singular, np.zeros(2), DD)


class TestPureCasesClosedForm:
    """Pure translation, pure in-plane rotation, pure dilation and a
    pure out-of-plane tilt against their hand derived closed forms.
    These are the analytic halves of validation V1 and V4; their
    pattern halves live in ``test_hrebsd_engine.py``.  [D2/D6/V1/V4]"""

    def test_pure_translation(self):
        dx, dy = 2.5, -1.25
        h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        h[2] = dx
        h[5] = dy
        expected = np.array([[1.0, 0.0, dx / DD], [0.0, 1.0, dy / DD], [0.0, 0.0, 1.0]])
        got = homography_to_fe(h, np.zeros(2), DD)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_pure_in_plane_rotation(self):
        # a rotation about the detector normal is its own reduced
        # tensor, so the homography is exactly the rotation minus the
        # identity in the linear block and zero everywhere else
        theta = np.deg2rad(1.5)
        cos, sin = np.cos(theta), np.sin(theta)
        fe = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        expected = np.array([cos - 1, -sin, 0.0, sin, cos - 1, 0.0, 0.0, 0.0])
        got = fe_to_homography(fe, np.zeros(2), DD)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_pure_isotropic_dilation(self):
        # an in-plane dilation with an unchanged normal direction
        alpha = 1.002
        fe = np.diag([alpha, alpha, 1.0])
        expected = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        expected[0] = alpha - 1
        expected[4] = alpha - 1
        got = fe_to_homography(fe, np.zeros(2), DD)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_pure_out_of_plane_tilt(self):
        # the leading terms of validation V4: a small rotation about
        # the detector x axis puts a translation of about
        # ``-omega * DD`` into h23 and a perspective term of about
        # ``omega / DD`` into h32, with NOTHING in h13 or h31
        omega = np.deg2rad(0.5)
        cos, sin = np.cos(omega), np.sin(omega)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
        fe = rotation / rotation[2, 2]
        got = fe_to_homography(fe, np.zeros(2), DD)
        tan = sin / cos
        expected = np.array([1 / cos - 1, 0.0, 0.0, 0.0, 0.0, -DD * tan, 0.0, tan / DD])
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)
        # and the leading order the docstring quotes
        assert got[5] == pytest.approx(-omega * DD, rel=1e-4)
        assert got[7] == pytest.approx(omega / DD, rel=1e-4)
