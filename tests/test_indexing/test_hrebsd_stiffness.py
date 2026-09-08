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

"""Tests of ``kikuchipy.indexing._hrebsd._stiffness``.

Covers requirements D9.4 and D9.5 of
``specs/2026-09-07-hrebsd-dic/``: the frozen stiffness convention
(6 by 6 Voigt, GPa, crystal frame, order ``(11, 22, 33, 23, 13,
12)``, engineering shears in the Hooke product), the
``voigt_stiffness`` builder against literature nickel and titanium
constants, the Voigt to fourth-order round trip, and the crystal to
sample rotation.

THE pin of this module is the 22.5 degree ``C16'`` sign
(requirements D9.5).  The two pins a first draft would reach for --
invariance under a symmetry operation and the 45 degree textbook
forms -- are both EVEN in the rotation angle and pass with the
rotation matrix and its transpose alike, so neither can fix which way
the input map's orientation enters the rotation of ``C``.  The
``C16'`` closed form is ODD in the angle::

    C16'(theta) = (1/4) * sin(4*theta) * (C12 + 2*C44 - C11)

for a rotation of ``theta`` about the crystal ``z`` axis, and this
module re-derives it independently (from the fourth-order rotation
written out here in plain numpy) rather than importing it, so a
coordinated error in the module under test cannot make the pin
agree with itself.

Written before the implementation exists: every test which calls the
module fails with ``NotImplementedError`` until the stiffness lands,
then passes unchanged.  The two library measurements marked as such
pass today, which is what makes them freezes rather than behaviour
tests.
"""

import inspect

import numpy as np
from orix.quaternion import Rotation
import pytest

from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd._stiffness import (
    SHEAR_COMPONENTS,
    SUPPORTED_SYMMETRIES,
    VOIGT_INDICES,
    VOIGT_SIZE,
    hooke_product,
    rotate_stiffness,
    tensor_to_voigt,
    voigt_stiffness,
    voigt_to_tensor,
)

# ------------------------- Frozen constants ------------------------- #

# The algebraic identity band of this feature, FROZEN and not
# measured, as in the Stage A sibling modules
ALGEBRA_TOL = 1e-12

# Single-crystal elastic constants in GPa at room temperature, from
# Simmons and Wang, Single Crystal Elastic Constants and Calculated
# Aggregate Properties (1971).  They are TEST DATA, never shipped:
# requirements D9.4 freezes that kikuchipy holds no elastic constant
# database and that the stiffness is supplied per phase by the user
NICKEL_CUBIC = {"c11": 246.5, "c12": 147.3, "c44": 124.7}
TITANIUM_HEXAGONAL = {
    "c11": 162.4,
    "c12": 92.0,
    "c13": 69.0,
    "c33": 180.7,
    "c44": 46.7,
}


# ----------------------------- Helpers ------------------------------ #


def rotation_about(axis, angle_deg):
    """Return the ``(3, 3)`` matrix of a rotation, written out here
    with Rodrigues' formula so that no orix or kikuchipy convention
    enters the oracle."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(angle_deg)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def cubic_matrix(c11, c12, c44):
    """Return the cubic Voigt matrix, assembled here entry by entry
    rather than through the builder under test."""
    c = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            c[i, j] = c11 if i == j else c12
    for i in range(3, 6):
        c[i, i] = c44
    return c


def hexagonal_matrix(c11, c12, c13, c33, c44):
    """Return the hexagonal Voigt matrix, assembled here.  ``C66`` is
    DERIVED, ``(C11 - C12)/2``, and is never an input."""
    c = np.zeros((6, 6))
    c[0, 0] = c[1, 1] = c11
    c[2, 2] = c33
    c[0, 1] = c[1, 0] = c12
    c[0, 2] = c[2, 0] = c13
    c[1, 2] = c[2, 1] = c13
    c[3, 3] = c[4, 4] = c44
    c[5, 5] = 0.5 * (c11 - c12)
    return c


def voigt_to_tensor_here(c):
    """Return the fourth-order form of a Voigt matrix, expanded here
    with the minor symmetries so that the round trip below judges the
    module against something which does not share its code."""
    tensor = np.zeros((3, 3, 3, 3))
    for p, (i, j) in enumerate(VOIGT_INDICES):
        for q, (k, m) in enumerate(VOIGT_INDICES):
            value = c[p, q]
            for a, b in ((i, j), (j, i)):
                for e, f in ((k, m), (m, k)):
                    tensor[a, b, e, f] = value
    return tensor


def rotate_here(c, g):
    """Return the sample-frame Voigt matrix of a crystal-frame one,
    the fourth-order rotation of requirements D9.5 written out here::

        C_sample_ijkl = g_ai g_bj g_ck g_dl C_crystal_abcd

    with ``g`` the orientation in the kikuchipy sense,
    ``v_crystal = g @ v_sample``.  The oracle of this module, and the
    reason the sign pin below means anything: it is assembled from
    the definition, not from ``_stiffness``.
    """
    tensor = voigt_to_tensor_here(c)
    rotated = np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, tensor)
    return np.array(
        [[rotated[i, j, k, m] for (k, m) in VOIGT_INDICES] for (i, j) in VOIGT_INDICES]
    )


def c16_closed_form(theta_deg, c11, c12, c44):
    """Return the hand-derived ``C16'`` of a cubic crystal rotated by
    *theta_deg* about ``z`` (requirements D9.5).  ODD in the angle,
    which is what makes it transpose-sensitive."""
    theta = np.deg2rad(theta_deg)
    return 0.25 * np.sin(4.0 * theta) * (c12 + 2.0 * c44 - c11)


# =============== D9.4 -- the builder and the convention ============= #


class TestVoigtStiffness:
    """The convenience builder of requirements D9.4, against
    literature constants assembled here.  [D9.4]"""

    def test_cubic_matches_the_literature_nickel_matrix(self):
        got = voigt_stiffness("cubic", **NICKEL_CUBIC)
        expected = cubic_matrix(**NICKEL_CUBIC)
        assert got.shape == (VOIGT_SIZE, VOIGT_SIZE)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_hexagonal_matches_the_literature_titanium_matrix(self):
        got = voigt_stiffness("hexagonal", **TITANIUM_HEXAGONAL)
        expected = hexagonal_matrix(**TITANIUM_HEXAGONAL)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)
        # the derived entry, stated literally: a hexagonal C66 which
        # was taken as an input instead would be whatever the caller
        # passed, and nothing else in the matrix would notice
        assert got[5, 5] == pytest.approx(
            0.5 * (TITANIUM_HEXAGONAL["c11"] - TITANIUM_HEXAGONAL["c12"])
        )

    def test_cubic_c66_is_c44_and_not_the_hexagonal_combination(self):
        # the copy-paste killer between the two branches
        got = voigt_stiffness("cubic", **NICKEL_CUBIC)
        assert got[5, 5] == pytest.approx(NICKEL_CUBIC["c44"])
        hexagonal_like = 0.5 * (NICKEL_CUBIC["c11"] - NICKEL_CUBIC["c12"])
        assert not np.isclose(got[5, 5], hexagonal_like)

    @pytest.mark.parametrize(
        "symmetry,constants",
        [("cubic", NICKEL_CUBIC), ("hexagonal", TITANIUM_HEXAGONAL)],
    )
    def test_the_matrix_is_symmetric_and_positive_definite(self, symmetry, constants):
        got = voigt_stiffness(symmetry, **constants)
        np.testing.assert_allclose(got, got.T, rtol=0, atol=ALGEBRA_TOL)
        # a stiffness which is not positive definite is not a
        # stiffness; this catches a mistyped literature constant as
        # well as a mis-assembled matrix
        assert np.all(np.linalg.eigvalsh(got) > 0)

    def test_zero_entries_are_really_zero(self):
        # the shear block never couples to the normal block in either
        # symmetry, and the two shear axes never couple to each other
        for symmetry, constants in (
            ("cubic", NICKEL_CUBIC),
            ("hexagonal", TITANIUM_HEXAGONAL),
        ):
            got = voigt_stiffness(symmetry, **constants)
            assert np.all(got[:3, 3:] == 0.0), symmetry
            assert np.all(got[3:, :3] == 0.0), symmetry
            off_diagonal_shear = got[3:, 3:] - np.diag(np.diag(got[3:, 3:]))
            assert np.all(off_diagonal_shear == 0.0), symmetry

    def test_signature_is_frozen(self):
        parameters = inspect.signature(voigt_stiffness).parameters
        assert list(parameters) == ["symmetry", "c11", "c12", "c44", "c13", "c33"]
        assert parameters["symmetry"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in ("c11", "c12", "c44", "c13", "c33"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        for name in ("c11", "c12", "c44"):
            assert parameters[name].default is inspect.Parameter.empty, name
        for name in ("c13", "c33"):
            assert parameters[name].default is None, name

    def test_supported_symmetries_are_frozen(self):
        assert SUPPORTED_SYMMETRIES == ("cubic", "hexagonal")

    def test_guards(self):
        with pytest.raises(ValueError, match="cubic"):
            voigt_stiffness("tetragonal", **NICKEL_CUBIC)
        # hexagonal needs its two extra constants, each named
        with pytest.raises(ValueError, match="c13"):
            voigt_stiffness("hexagonal", c11=1.0, c12=0.5, c44=0.25, c33=2.0)
        with pytest.raises(ValueError, match="c33"):
            voigt_stiffness("hexagonal", c11=1.0, c12=0.5, c44=0.25, c13=0.4)
        # and cubic refuses them rather than ignoring them, which is
        # how a caller learns that a cubic C13 is not independent
        with pytest.raises(ValueError, match="c13"):
            voigt_stiffness("cubic", c13=0.4, **NICKEL_CUBIC)


class TestVoigtTensorRoundTrip:
    """The Voigt to fourth-order expansion and back.  [D9.5]"""

    def test_round_trip_is_exact(self):
        c = voigt_stiffness("hexagonal", **TITANIUM_HEXAGONAL)
        np.testing.assert_allclose(
            tensor_to_voigt(voigt_to_tensor(c)), c, rtol=0, atol=ALGEBRA_TOL
        )

    def test_expansion_matches_the_hand_built_one(self):
        c = voigt_stiffness("hexagonal", **TITANIUM_HEXAGONAL)
        np.testing.assert_allclose(
            voigt_to_tensor(c), voigt_to_tensor_here(c), rtol=0, atol=ALGEBRA_TOL
        )

    def test_the_voigt_order_is_the_frozen_one(self):
        # a permutation of the last three Voigt slots -- (23, 13, 12)
        # against, say, (12, 13, 23) -- survives every symmetric test
        # in this module, so the order is pinned literally here
        assert VOIGT_INDICES == ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        assert SHEAR_COMPONENTS == (3, 4, 5)
        c = np.arange(36, dtype=np.float64).reshape(6, 6)
        c = 0.5 * (c + c.T)
        tensor = voigt_to_tensor(c)
        for p, (i, j) in enumerate(VOIGT_INDICES):
            for q, (k, m) in enumerate(VOIGT_INDICES):
                assert tensor[i, j, k, m] == pytest.approx(c[p, q]), (p, q)

    def test_the_minor_and_major_symmetries_hold(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        tensor = voigt_to_tensor(c)
        np.testing.assert_allclose(
            tensor, np.swapaxes(tensor, 0, 1), rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            tensor, np.swapaxes(tensor, 2, 3), rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            tensor,
            np.transpose(tensor, (2, 3, 0, 1)),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_guards(self):
        with pytest.raises(ValueError):
            voigt_to_tensor(np.zeros((5, 6)))
        with pytest.raises(ValueError):
            tensor_to_voigt(np.zeros((3, 3, 3)))


# ========== D9.5 -- the crystal to sample rotation direction ======== #


class TestRotationDirection:
    """The Bond rotation and THE transpose pin.  [D9.5]"""

    def test_the_matrix_is_the_kikuchipy_orientation(self):
        # A LIBRARY MEASUREMENT, so it passes before the
        # implementation lands.  ``rotate_stiffness`` takes the
        # point's orientation ``g`` with ``v_crystal = g @ v_sample``,
        # and this pins that ``Rotation.to_matrix`` -- what a crystal
        # map's rotations give -- IS that matrix, derived from
        # ``rotate_vector`` so that no orix convention is assumed
        rotation = Rotation.from_axes_angles((1.0, 2.0, 3.0), np.deg2rad(37.0))
        quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
        sample_to_crystal = rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T
        np.testing.assert_allclose(
            rotation.to_matrix().squeeze(),
            sample_to_crystal,
            rtol=0,
            atol=1e-12,
        )

    def test_identity_changes_nothing(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        np.testing.assert_allclose(
            rotate_stiffness(c, np.eye(3)), c, rtol=0, atol=ALGEBRA_TOL
        )

    @pytest.mark.parametrize(
        "axis,angle", [((0.0, 0.0, 1.0), 90.0), ((1.0, 1.0, 1.0), 120.0)]
    )
    def test_a_cubic_symmetry_operation_is_an_invariance(self, axis, angle):
        # necessary but NOT sufficient: this passes with the
        # transposed convention too, which is exactly why the sign pin
        # below exists
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        np.testing.assert_allclose(
            rotate_stiffness(c, rotation_about(axis, angle)),
            c,
            rtol=0,
            atol=1e-10,
        )

    def test_forty_five_degrees_reproduces_the_textbook_form(self):
        # the classic pins of requirements D9.5, and also transpose
        # blind: ``sin(4 * 45 deg) = 0`` and every entry below is even
        # in the angle
        c11, c12, c44 = (NICKEL_CUBIC[k] for k in ("c11", "c12", "c44"))
        rotated = rotate_stiffness(
            voigt_stiffness("cubic", **NICKEL_CUBIC),
            rotation_about((0.0, 0.0, 1.0), 45.0),
        )
        assert rotated[0, 0] == pytest.approx(0.5 * (c11 + c12 + 2 * c44))
        assert rotated[1, 1] == pytest.approx(0.5 * (c11 + c12 + 2 * c44))
        assert rotated[0, 1] == pytest.approx(0.5 * (c11 + c12 - 2 * c44))
        assert rotated[5, 5] == pytest.approx(0.5 * (c11 - c12))
        # a rotation about z leaves the z axis alone
        assert rotated[2, 2] == pytest.approx(c11)
        assert rotated[0, 2] == pytest.approx(c12)
        assert rotated[3, 3] == pytest.approx(c44)

    @pytest.mark.parametrize("angle", [22.5, -22.5, 10.0])
    def test_the_c16_sign_pins_the_transpose(self, angle):
        # THE pin of requirements D9.5.  ``C16'`` is ODD in the
        # rotation angle, so the transposed convention returns its
        # negation, and the closed form is re-derived here from the
        # fourth-order definition (``rotate_here``) as well as written
        # out in ``c16_closed_form``: three independent statements of
        # the same number
        c11, c12, c44 = (NICKEL_CUBIC[k] for k in ("c11", "c12", "c44"))
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        g = rotation_about((0.0, 0.0, 1.0), angle)
        expected = c16_closed_form(angle, c11, c12, c44)
        got = rotate_stiffness(c, g)
        assert got[0, 5] == pytest.approx(expected, abs=1e-9)
        np.testing.assert_allclose(got, rotate_here(c, g), rtol=0, atol=1e-9)
        # the transposed convention, asserted to FAIL: this is the
        # plan section 3.4 "Bond rotation transposed" mutant, and the
        # 22.5 degree magnitude is 37.55 GPa, a quarter of the
        # anisotropy factor and nothing like a rounding difference
        assert rotate_stiffness(c, g.T)[0, 5] == pytest.approx(-expected, abs=1e-9)
        assert abs(expected) > 1.0
        assert not np.isclose(rotate_stiffness(c, g.T)[0, 5], expected)

    def test_the_tensor_norm_is_invariant(self):
        # a rotation is a change of basis: the fourth-order Frobenius
        # norm cannot move, whichever way the matrix goes.  It catches
        # an einsum which contracts the wrong pair of indices, which
        # neither the sign pin nor the textbook forms see
        c = voigt_stiffness("hexagonal", **TITANIUM_HEXAGONAL)
        g = rotation_about((1.0, 2.0, 3.0), 37.0)
        before = np.linalg.norm(voigt_to_tensor_here(c))
        after = np.linalg.norm(voigt_to_tensor_here(rotate_stiffness(c, g)))
        assert after == pytest.approx(before, rel=1e-10)

    def test_a_stack_of_orientations_matches_the_loop(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        matrices = np.stack(
            [
                rotation_about((0.0, 0.0, 1.0), 22.5),
                rotation_about((1.0, 2.0, 3.0), 37.0),
                np.eye(3),
            ]
        )
        stacked = rotate_stiffness(c, matrices)
        assert stacked.shape == (3, VOIGT_SIZE, VOIGT_SIZE)
        for i, matrix in enumerate(matrices):
            np.testing.assert_allclose(
                stacked[i], rotate_stiffness(c, matrix), rtol=0, atol=ALGEBRA_TOL
            )

    def test_guards(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        with pytest.raises(ValueError):
            rotate_stiffness(np.zeros((3, 3)), np.eye(3))
        with pytest.raises(ValueError):
            rotate_stiffness(c, np.eye(2))


# ============ D9.4 -- the engineering shear factor of two =========== #


class TestHookeProduct:
    """The one place the engineering shear convention lives.
    [D9.4/D10]"""

    def test_a_pure_shear_carries_the_factor_of_two(self):
        # THE killer of the plan section 3.4 mutants "engineering
        # shear factor dropped" and "doubled": with TENSOR shears in
        # and the engineering convention inside, a tensor shear
        # ``e12`` gives ``sigma12 = 2 * C44 * e12``
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        strain = np.zeros(VOIGT_SIZE)
        strain[5] = 1.5e-3
        stress = hooke_product(c, strain)
        assert stress[5] == pytest.approx(2 * NICKEL_CUBIC["c44"] * strain[5])
        # the two ways of getting it wrong, both excluded by name
        assert not np.isclose(stress[5], NICKEL_CUBIC["c44"] * strain[5])
        assert not np.isclose(stress[5], 4 * NICKEL_CUBIC["c44"] * strain[5])
        # and a pure shear produces no normal stress in a cubic
        # crystal on its own axes
        assert np.allclose(stress[:3], 0.0, atol=ALGEBRA_TOL)

    def test_a_uniaxial_strain_gives_the_textbook_normal_stresses(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        strain = np.zeros(VOIGT_SIZE)
        strain[0] = 1e-3
        stress = hooke_product(c, strain)
        assert stress[0] == pytest.approx(NICKEL_CUBIC["c11"] * strain[0])
        assert stress[1] == pytest.approx(NICKEL_CUBIC["c12"] * strain[0])
        assert stress[2] == pytest.approx(NICKEL_CUBIC["c12"] * strain[0])
        assert np.allclose(stress[3:], 0.0, atol=ALGEBRA_TOL)

    def test_it_is_linear_and_stacks(self):
        c = voigt_stiffness("hexagonal", **TITANIUM_HEXAGONAL)
        rng = np.random.default_rng(3)
        strains = rng.uniform(-1e-3, 1e-3, size=(4, VOIGT_SIZE))
        stacked = hooke_product(c, strains)
        assert stacked.shape == (4, VOIGT_SIZE)
        for i, strain in enumerate(strains):
            np.testing.assert_allclose(
                stacked[i], hooke_product(c, strain), rtol=0, atol=ALGEBRA_TOL
            )
        np.testing.assert_allclose(
            hooke_product(c, 2.0 * strains[0]),
            2.0 * hooke_product(c, strains[0]),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_a_stack_of_stiffnesses_pairs_point_by_point(self):
        # the per-point stiffness the chain builds from the per-point
        # orientation
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        matrices = np.stack([np.eye(3), rotation_about((0.0, 0.0, 1.0), 45.0)])
        stiffnesses = rotate_stiffness(c, matrices)
        strain = np.zeros((2, VOIGT_SIZE))
        strain[:, 0] = 1e-3
        stress = hooke_product(stiffnesses, strain)
        assert stress[0, 0] == pytest.approx(NICKEL_CUBIC["c11"] * 1e-3)
        c11, c12, c44 = (NICKEL_CUBIC[k] for k in ("c11", "c12", "c44"))
        assert stress[1, 0] == pytest.approx(0.5 * (c11 + c12 + 2 * c44) * 1e-3)

    def test_guards(self):
        c = voigt_stiffness("cubic", **NICKEL_CUBIC)
        with pytest.raises(ValueError):
            hooke_product(c, np.zeros(5))
        with pytest.raises(ValueError):
            hooke_product(np.zeros((5, 5)), np.zeros(VOIGT_SIZE))
        with pytest.raises(ValueError):
            hooke_product(np.zeros((2, 6, 6)), np.zeros((3, VOIGT_SIZE)))
