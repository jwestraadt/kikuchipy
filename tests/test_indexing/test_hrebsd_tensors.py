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

"""Tests of ``kikuchipy.indexing._hrebsd._tensors``.

Covers requirements D7 to D10 and D15.5 to D15.6 of
``specs/2026-09-07-hrebsd-dic/`` and the TENSOR half of oracle V3 of
its ``validation.md``:

- the frame chain, ``beta_s = M^T beta_det M``, and the detector
  frame ``M`` actually is (see the frame note below);
- the polar decomposition and the strain measures of D8, on a pure
  stretch, a pure rotation and a composed case where the right and
  left stretches differ;
- the ninth degree of freedom closure of D9 against an independently
  built tensor which carries ``sigma33 = 0`` by construction, where
  ``e33`` comes back EXACTLY -- the V3 tensor half;
- the deviatoric fallback and the two closures disagreeing;
- the derived stress maps of D10 on hand-built tensors;
- the small-strain fast-path measurement harness of D8, which is a
  HARNESS and not an enabling: nothing public takes that path until
  its band and its enabling angle are measured and recorded;
- the Stage B properties of D15.6 on the returned crystal map, and
  the frozen signature of ``hrebsd_strain_stress``.

**The detector frame, made explicit (and a correction requirements
D7 needs).**  D7 writes the rotation with
``R = detector.sample_to_detector.to_matrix()``.  That matrix maps
the sample frame into kikuchipy's GNOMONIC detector frame, whose
``y`` points up, while the ``Fe`` the engine measures lives in the
numpy array frame of D1.1, whose ``y`` points DOWN: the two differ by
``DETECTOR_Y_FLIP``, the same flip the Stage A half-pixel measurement
recorded for the coordinates (``test_hrebsd_engine.py``,
``TestPcCentredFrame``, and ``impose_detector_frame_fe`` there, which
imposes every V3 and V4 deformation through exactly this composed
matrix).  ``TestFrameChain`` below MEASURES the relation from the
detector geometry again, here, so the correction is pinned rather
than asserted; dropping the flip mirrors ``beta13``, ``beta23``,
``beta31`` and ``beta32`` and flips two components of every reported
rotation vector.

Written failing before the implementation, at the Stage B
failing-tests gate: every test which calls the module failed with
``NotImplementedError`` until the tensor chain landed, and passed
unchanged after it, while the frame measurement and the frozen
signature pins passed from the start (narration corrected to the past
tense 2026-09-08, Stage B adversarial review).
"""

import inspect

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._stiffness import (
    VOIGT_INDICES,
    VOIGT_SIZE,
    hooke_product,
)
from kikuchipy.indexing._hrebsd._tensors import (
    BETA_PROP_SIZE,
    DETECTOR_Y_FLIP,
    PRINCIPAL_PROP_SIZE,
    ROTATION_VECTOR_PROP_SIZE,
    STAGE_B_PROP_NAMES,
    STRAIN_PROP_SIZE,
    SUPPORTED_CLOSURES,
    SUPPORTED_STRAIN_MEASURES,
    beta_to_sample_frame,
    close_beta,
    hrebsd_strain_stress,
    hydrostatic_stress,
    polar_decomposition,
    principal_stresses,
    rotation_vector_from_matrix,
    sample_to_detector_matrix,
    small_strain_split,
    strain_from_stretch,
    tensor_chain,
    tensor_to_voigt_vector,
    von_mises_stress,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
)

# ------------------------- Frozen constants ------------------------- #

# The algebraic identity band of this feature, FROZEN, as in the
# Stage A sibling modules: every identity below is machine precision
# class by construction and nothing here is measured against patterns
ALGEBRA_TOL = 1e-12

# The band of an identity which passes through a 3 by 3 linear solve
# or a singular value decomposition, FROZEN and loosened from
# ALGEBRA_TOL only by those solvers' own conditioning
SOLVER_TOL = 1e-10

# A detector which is NOT at the pattern centre and NOT square, so
# that a row-column swap anywhere in the frame chain shows
SHAPE = (60, 80)
PC = (0.4210, 0.5794, 0.5049)

# A generic crystal orientation, deliberately away from every
# symmetry position: at a symmetric one the crystal to sample
# stiffness rotation is unexercised (the companion of the 22.5 degree
# C16' pin of ``test_hrebsd_stiffness.py``)
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0

# Single-crystal elastic constants of nickel in GPa (Simmons and
# Wang 1971), TEST DATA only: requirements D9.4 freezes that no
# elastic constant database ships with kikuchipy
NICKEL_CUBIC = {"c11": 246.5, "c12": 147.3, "c44": 124.7}

# The strain scale HREBSD is built for
STRAIN_SCALE = 1e-3

# The equality criterion requirements D8 puts on the small-strain
# fast path, FROZEN there: the path may only be enabled where it
# agrees with the polar one to this many strain units
FAST_PATH_CRITERION = 1e-6


# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders (requirements D19).  Each is replaced by a
# dated measured value at the Stage B implementation gate, with the
# recipe recorded in validation.md "Recorded results".

# MTP [D8/D9, V3 tensor half]: the worst per-component strain error of
# the chain run on a REDUCED deformation gradient, against the tensor
# it was built from.  It cannot be zero and the reason is structural,
# not numerical: the stored ``Fe`` is reduced by ``Fe33``, which
# scales every off-diagonal by ``1/(1 + beta33)``, so the chain
# recovers the imposed strain to second order in beta -- about
# 1e-3 * 1e-3 = 1e-6 at the strain scale of this module.  A measured
# value far ABOVE 1e-5 means the closure or the frame is wrong rather
# than the reduction, and that is the finding.
# MEASURING RECIPE: ``TestChain::test_strain_recovery_at_a_generic_
# orientation`` below; record the worst component over both closures
# and both strain measures
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md Recorded results, Stage B implementation gate entry 39).
# MEASURED worst 1.1654500102918543e-06 over all four arms which
# consume it -- strain 1.1655e-06, e33 9.4132e-07, beta 9.5697e-07,
# rotation vector 8.4533e-08 -- pinned at 2x that worst.  It is the
# reduction's own second order, as the note above predicts, and it
# reproduces the failing-tests-gate reference chain (entry 30) to every
# digit.  It still kills what it exists to kill by two orders: the Bond
# rotation transposed measures 2.9630e-04 here and the frame rotation
# untransposed 8.3352e-04
REDUCED_CLOSURE_TOL = 2.4e-06

# MTP [D9.2/D10, V3]: the residual ``sigma33`` in GPa of the
# traction-free closure, run through the whole chain.  Zero by
# construction in the linear closure and second order through the
# reduction and the Biot strain, so the scale to expect is
# ``|C| * 1e-6`` = a few times 1e-4 GPa.
# MEASURING RECIPE: ``TestChain::test_sigma33_is_the_closure_self_
# check``
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md entry 39).  MEASURED 3.0638726905057867e-04 GPa against
# a stress scale of 0.2420 GPa on the same three points, pinned at 2x.
# RECORDED, not a killer of anything: the Bond-transposed mutant
# measures 3.0914e-04 GPa here, 0.9 per cent away, so this band must
# never be quoted as its killer (plan 3.4 (a), validation entry 31)
SIGMA33_TOL = 6.2e-04

# MTP [D8]: the worst strain difference between the polar path and
# the small-strain fast path over the rotation sweep, and the angle
# at which it crosses the 1e-6 requirements D8 asks for.  The fast
# path is NOT enabled by this measurement: requirements D8 asks for
# the band and the enabling angle to be RECORDED first, and until
# they are, every public result goes through the polar path, which
# ``test_the_public_path_is_the_polar_one`` pins.
# MEASURING RECIPE: ``TestSmallStrainFastPath`` below; record the
# error at each swept angle and the largest angle still inside 1e-6
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md entry 39).  Sweep errors, all reproduced bitwise
# between runs and under randomized test order: 9.1400e-08 (0.01 deg),
# 5.6564e-07 (0.05), 1.6043e-06 (0.1), 3.7330e-05 (0.5), 1.4790e-04
# (1.0), 5.8890e-04 (2.0).  ``SMALL_STRAIN_EQUALITY_TOL`` is 2x the
# worst of those, 5.889002203349758e-04 at 2.0 deg.
#
# ``SMALL_STRAIN_ENABLE_ANGLE_DEG`` is a LOWER bound (``assert_at_
# least``), so its 2x margin goes the other way: the BISECTED crossing
# of the 1e-6 criterion measures 0.07794410136352782 deg and the pin is
# half of it.  The fast path is STILL NOT ENABLED by this measurement
# -- requirements D8 asks for the band and the angle to be recorded
# first, and ``test_the_public_path_is_the_polar_one`` pins that every
# public result goes through the polar path
SMALL_STRAIN_EQUALITY_TOL = 1.2e-03
SMALL_STRAIN_ENABLE_ANGLE_DEG = 0.039


# ------------- The plan 3.4 mutation list, mapped ------------------- #
#
# The Stage B mutants of plan section 3.4 which this module owns, each
# with the test that kills it.  The segmentation, KAM and PC-shift
# mutants are mapped in their own modules.
#
#  engineering shear dropped or doubled . stiffness: test_a_pure_shear
#                                         _carries_the_factor_of_two
#  Bond rotation transposed ............. stiffness: test_the_c16_sign
#                                         _pins_the_transpose, plus
#                                         TestChain::test_strain
#                                         _recovery_at_a_generic
#                                         _orientation here (MEASURED
#                                         worst strain error 2.9630e-04
#                                         transposed against 1.1655e-06
#                                         correct) and, through
#                                         PATTERNS, test_hrebsd
#                                         _deformed_master.py.  NOT
#                                         test_sigma33_is_the_closure
#                                         _self_check, which is blind
#                                         to it (3.0914e-04 GPa
#                                         transposed against 3.0639e-04
#                                         correct)
#  D6.2 correction re-applied ........... test_hrebsd_pc_shift.py::
#                                         TestPhantomThroughTheChain
#  closure rows swapped ................. TestClosure::test_traction
#                                         _free_recovers_the_built_in
#                                         _tensor (the shear terms of
#                                         the first row are nonzero
#                                         only at a generic C)
#  b7/b8 misassigned .................... the same test (1.4711e-05
#                                         against SOLVER_TOL), and the
#                                         second arm of test_b7_and_b8
#                                         _are_not_interchangeable.
#                                         Its FIRST arm is blind to the
#                                         mutant -- see the note there
#  shear terms of the traction-free ..... the same test (2.7692e-05),
#  right-hand side dropped               plus test_the_shear_terms_of
#  (not on the plan 3.4 list)            _the_right_hand_side_are_used
#  R versus R^T in the frame rotation ... TestFrameChain::test_the
#                                         _rotation_uses_the_transpose,
#                                         and independently TestChain
#                                         ::test_strain_recovery_at_a
#                                         _generic_orientation
#                                         (8.3352e-04 against
#                                         1.1655e-06).  Plan 3.4 says
#                                         it "dies by V4"; V4's Stage B
#                                         split is a pure-algebra
#                                         harness here, so these two
#                                         are the killers
#  DETECTOR_Y_FLIP dropped from M ....... TestFrameChain::test_the
#  (not on the plan 3.4 list; the         _detector_frame_is_the_y_down
#  Stage B analogue of Stage A's          _one and test_sample_to
#  D1.1 correction)                       _detector_matrix_carries_the
#                                         _flip
#  Biot computed as V - I ............... TestPolarDecomposition::test
#                                         _composed_case_uses_the_right
#                                         _stretch
#  small-strain path taken publicly ..... TestSmallStrainFastPath::test
#                                         _the_public_path_is_the_polar
#                                         _one (6.09e-04 separation on
#                                         the chain's own closed beta)
#  principal stresses ascending ......... TestDerivedMaps::test
#                                         _principal_stresses_descend


# ----------------------------- Helpers ------------------------------ #


def make_detector(shape=SHAPE, pc=PC, binning=8):
    """Return a detector with one Bruker projection centre."""
    return kp.detectors.EBSDDetector(
        shape=shape,
        binning=binning,
        px_size=70.0,
        pc=np.atleast_2d(np.asarray(pc, dtype=np.float64)),
        sample_tilt=70.0,
        tilt=0.0,
    )


def spec_frame_matrix(detector):
    """Return ``M`` with ``v_detector = M @ v_sample`` in the spec's
    y-DOWN detector frame, assembled here from the detector's own
    rotation and the measured flip rather than taken from
    ``_tensors``."""
    matrix = detector.sample_to_detector.to_matrix().squeeze()
    return np.diag([1.0, -1.0, 1.0]) @ matrix


def rotation_about(axis, angle_deg):
    """Return the ``(3, 3)`` rotation matrix, from Rodrigues' formula
    written out here."""
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
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def skew_of(vector):
    """Return the antisymmetric tensor whose dual vector is
    ``(w32, w13, w21)``, the convention requirements D8 reports the
    rotation vector in."""
    w1, w2, w3 = vector
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def voigt_to_tensor_here(c):
    """Return the fourth-order form of a Voigt matrix, expanded here
    so that this module's stiffness rotation shares no code with
    ``_stiffness``."""
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
    ``C_sample_ijkl = g_ai g_bj g_ck g_dl C_crystal_abcd`` with ``g``
    the orientation, ``v_crystal = g @ v_sample`` (requirements
    D9.5)."""
    tensor = voigt_to_tensor_here(c)
    rotated = np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, tensor)
    return np.array(
        [[rotated[i, j, k, m] for (k, m) in VOIGT_INDICES] for (i, j) in VOIGT_INDICES]
    )


def voigt_vector_here(tensor):
    """Return the six Voigt components of a symmetric tensor, with
    TENSOR shears, assembled here."""
    symmetric = 0.5 * (tensor + tensor.T)
    return np.array([symmetric[i, j] for (i, j) in VOIGT_INDICES])


def generic_orientation_matrix():
    """Return the generic orientation ``g`` of this module,
    ``v_crystal = g @ v_sample``."""
    return rotation_about(ORIENTATION_AXIS, ORIENTATION_ANGLE_DEG)


def generic_sample_stiffness():
    """Return the nickel stiffness rotated into the sample frame at
    the generic orientation, built with this module's own rotation.

    A GENERIC orientation matters here for a reason beyond the
    stiffness pin: an unrotated cubic matrix has ``C3312``, ``C3313``
    and ``C3323`` identically zero, so the shear terms of the
    traction-free right-hand side would drop out of the closure and
    a mutant which forgets them would survive every assertion.
    """
    return rotate_here(
        np.array(
            [[c11c12(i, j) for j in range(VOIGT_SIZE)] for i in range(VOIGT_SIZE)]
        ),
        generic_orientation_matrix(),
    )


def c11c12(i, j):
    """Return one entry of the unrotated cubic nickel matrix."""
    c11, c12, c44 = (NICKEL_CUBIC[k] for k in ("c11", "c12", "c44"))
    if i < 3 and j < 3:
        return c11 if i == j else c12
    return c44 if i == j else 0.0


def traction_free_beta(stiffness_sample, seed=0):
    """Return a sample-frame ``beta`` whose symmetric part carries
    ``sigma33 = 0`` BY CONSTRUCTION, and the strain it was built
    from.

    ``e33`` is solved from the Hooke product written out here::

        sigma33 = C31 e11 + C32 e22 + C33 e33
                  + C34 (2 e23) + C35 (2 e13) + C36 (2 e12)

    so that the closure of requirements D9.2 has an exact answer to
    find, and a skew part is added, since a closure which quietly
    symmetrizes its input would otherwise pass.
    """
    rng = np.random.default_rng(seed)
    e11, e22 = rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=2)
    e23, e13, e12 = rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=3)
    c = np.asarray(stiffness_sample, dtype=np.float64)
    e33 = (
        -(
            c[2, 0] * e11
            + c[2, 1] * e22
            + c[2, 3] * 2 * e23
            + c[2, 4] * 2 * e13
            + c[2, 5] * 2 * e12
        )
        / c[2, 2]
    )
    strain = np.array(
        [[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]], dtype=np.float64
    )
    rotation = skew_of(rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=3))
    return strain + rotation, strain


def reduced_fe(beta_sample, matrix):
    """Return the REDUCED detector-frame tensor a strain-free engine
    would store for a sample-frame *beta_sample*.

    The engine measures in the detector frame and stores
    ``Fe / Fe33``, which is where the isotropic degree of freedom is
    lost: this is the input the chain has to close.
    """
    f_detector = matrix @ (np.eye(3) + beta_sample) @ matrix.T
    return f_detector / f_detector[2, 2]


def crystal_map_with(fe, navigation_shape=(2, 3), orientation=None, phase_ids=None):
    """Return a crystal map carrying an ``Fe`` property, the input
    :func:`hrebsd_strain_stress` takes."""
    arrays, size = create_coordinate_arrays(navigation_shape, (1.0, 1.0))
    if orientation is None:
        rotation = Rotation.from_axes_angles(
            ORIENTATION_AXIS, np.deg2rad(ORIENTATION_ANGLE_DEG)
        )
    else:
        rotation = orientation
    arrays["rotations"] = Rotation(np.tile(rotation.data.ravel(), (size, 1)))
    if phase_ids is None:
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    else:
        arrays["phase_id"] = np.asarray(phase_ids, dtype=int)
        arrays["phase_list"] = PhaseList(
            [Phase(name="ni", space_group=225), Phase(name="ti", space_group=194)]
        )
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    fe = np.asarray(fe, dtype=np.float64).reshape(-1, BETA_PROP_SIZE)
    if fe.shape[0] == 1:
        fe = np.tile(fe, (size, 1))
    xmap.prop["Fe"] = fe
    return xmap


def assert_within(measured, bound, name: str) -> None:
    """Assert ``measured <= bound``, failing loudly and informatively
    while *bound* is an unfilled measured-then-pinned placeholder.
    The Stage A helper, repeated here so that each test module stands
    on its own (pytest imports test modules by path, so they cannot
    import one another)."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled MEASURED-THEN-PINNED placeholder "
            f"(requirements D19); measured {measured!r}. Fill it with "
            "a dated value and record the recipe in validation.md"
        )
    assert measured <= bound, f"{name}: {measured} > {bound}"


def assert_at_least(measured, bound, name: str) -> None:
    """Assert ``measured >= bound``, the direction a RANGE is pinned
    in: the small-strain fast path is valid AT LEAST up to the
    recorded angle, so a smaller measurement is the regression."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled MEASURED-THEN-PINNED placeholder "
            f"(requirements D19); measured {measured!r}. Fill it with "
            "a dated value and record the recipe in validation.md"
        )
    assert measured >= bound, f"{name}: {measured} < {bound}"


# ================= D7 -- the frame chain and its M ================== #


class TestFrameChain:
    """Which detector frame ``Fe`` lives in, and which way the
    rotation to the sample frame goes.  [D7]"""

    def test_the_detector_frame_is_the_y_down_one(self):
        # A LIBRARY MEASUREMENT which needs no ``_hrebsd`` code, so it
        # passes today.  It repeats the Stage A half-pixel measurement
        # for one purpose: to show that the frame the engine's
        # coordinates -- and therefore its ``Fe`` -- live in is
        # kikuchipy's detector rotation composed with a y flip, which
        # is what ``sample_to_detector_matrix`` has to return
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        nrows, ncols = SHAPE
        pcx, pcy, pcz = np.atleast_2d(detector.pc_flattened)[0]
        pc_px = np.array([pcx * ncols, pcy * nrows, pcz * nrows])
        cosines = _get_direction_cosines_from_detector(detector)
        rays = cosines @ matrix.T
        rays = rays / rays[:, 2:3] * pc_px[2]
        rows, cols = np.indices((nrows, ncols), dtype=np.float64)
        assert np.abs(rays[:, 0] - (cols.ravel() + 0.5 - pc_px[0])).max() < 1e-9
        assert np.abs(rays[:, 1] - (rows.ravel() + 0.5 - pc_px[1])).max() < 1e-9
        assert np.all(rays[:, 2] > 0)
        # and the flip is a reflection, not a rotation, which is why
        # it cannot be absorbed into the detector's own rotation
        assert np.linalg.det(np.diag([1.0, -1.0, 1.0])) == pytest.approx(-1.0)

    def test_the_module_constant_is_that_flip(self):
        np.testing.assert_allclose(
            DETECTOR_Y_FLIP, np.diag([1.0, -1.0, 1.0]), rtol=0, atol=0
        )

    def test_sample_to_detector_matrix_carries_the_flip(self):
        detector = make_detector()
        got = sample_to_detector_matrix(detector)
        np.testing.assert_allclose(
            got, spec_frame_matrix(detector), rtol=0, atol=ALGEBRA_TOL
        )
        # the unflipped matrix is a DIFFERENT matrix, and the two
        # differ in exactly the rows a lattice rotation about the
        # detector x and y axes lives in
        unflipped = detector.sample_to_detector.to_matrix().squeeze()
        assert not np.allclose(got, unflipped)

    def test_the_rotation_uses_the_transpose(self):
        # ``beta_s = M^T beta_det M``: the plan section 3.4 "R versus
        # R^T" mutant, killed on an ASYMMETRIC tensor, since a
        # symmetric one is blind to it whenever M is close to its own
        # transpose
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        beta_sample = (
            np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]])
            * STRAIN_SCALE
        )
        beta_detector = matrix @ beta_sample @ matrix.T
        np.testing.assert_allclose(
            beta_to_sample_frame(beta_detector, matrix),
            beta_sample,
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        # the other side of the conjugation gives something else
        assert not np.allclose(
            beta_to_sample_frame(beta_detector, matrix.T), beta_sample
        )

    def test_the_rotation_stacks(self):
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        rng = np.random.default_rng(11)
        beta = rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=(4, 3, 3))
        stacked = beta_to_sample_frame(beta, matrix)
        assert stacked.shape == (4, 3, 3)
        for i in range(4):
            np.testing.assert_allclose(
                stacked[i],
                beta_to_sample_frame(beta[i], matrix),
                rtol=0,
                atol=ALGEBRA_TOL,
            )

    def test_guards(self):
        with pytest.raises(ValueError):
            beta_to_sample_frame(np.zeros((2, 2)), np.eye(3))
        with pytest.raises(ValueError):
            beta_to_sample_frame(np.zeros((3, 3)), np.eye(2))


# ============ D8 -- the polar split and the strain measures ========= #


class TestPolarDecomposition:
    """Pure stretch, pure rotation and a composed case.  [D8]"""

    def test_pure_stretch_gives_the_identity_rotation(self):
        stretch = (
            np.eye(3)
            + np.array([[2.0, 0.5, -0.3], [0.5, -1.0, 0.2], [-0.3, 0.2, 0.7]])
            * STRAIN_SCALE
        )
        rotation, recovered = polar_decomposition(stretch)
        np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=SOLVER_TOL)
        np.testing.assert_allclose(recovered, stretch, rtol=0, atol=SOLVER_TOL)
        np.testing.assert_allclose(
            strain_from_stretch(recovered),
            stretch - np.eye(3),
            rtol=0,
            atol=SOLVER_TOL,
        )
        np.testing.assert_allclose(
            rotation_vector_from_matrix(rotation), np.zeros(3), rtol=0, atol=SOLVER_TOL
        )

    @pytest.mark.parametrize("angle", [0.05, 1.0, 5.0])
    def test_pure_rotation_leaks_no_strain(self, angle):
        # the analytic half of validation V4's strain-leakage arm: a
        # pure rotation has NO strain at any angle through the polar
        # path, which is the whole reason it is the public path
        axis = np.array([0.3, -0.5, 0.81])
        matrix = rotation_about(axis, angle)
        rotation, stretch = polar_decomposition(matrix)
        np.testing.assert_allclose(stretch, np.eye(3), rtol=0, atol=SOLVER_TOL)
        np.testing.assert_allclose(
            strain_from_stretch(stretch), np.zeros((3, 3)), rtol=0, atol=SOLVER_TOL
        )
        expected = axis / np.linalg.norm(axis) * np.deg2rad(angle)
        np.testing.assert_allclose(
            rotation_vector_from_matrix(rotation), expected, rtol=0, atol=SOLVER_TOL
        )

    def test_composed_case_uses_the_right_stretch(self):
        # THE killer of the plan section 3.4 "Biot computed as V - I"
        # mutant: with a rotation and a stretch which do not commute,
        # the left stretch ``V = R U R^T`` is a different tensor, and
        # requirements D8 reports ``U - I``
        stretch = (
            np.eye(3)
            + np.array([[3.0, 1.0, 0.0], [1.0, -2.0, 0.5], [0.0, 0.5, 1.0]])
            * STRAIN_SCALE
        )
        matrix = rotation_about((0.2, 0.9, -0.4), 2.0)
        f = matrix @ stretch
        rotation, recovered = polar_decomposition(f)
        np.testing.assert_allclose(rotation, matrix, rtol=0, atol=SOLVER_TOL)
        np.testing.assert_allclose(recovered, stretch, rtol=0, atol=SOLVER_TOL)
        left = matrix @ stretch @ matrix.T
        # the two stretches really do differ here, by far more than
        # the band above, so the assertion discriminates
        assert np.abs(left - stretch).max() > 1e-5
        np.testing.assert_allclose(
            strain_from_stretch(recovered),
            stretch - np.eye(3),
            rtol=0,
            atol=SOLVER_TOL,
        )
        assert not np.allclose(
            strain_from_stretch(recovered), left - np.eye(3), atol=1e-6
        )

    def test_green_lagrange_is_the_documented_alternative(self):
        stretch = (
            np.eye(3)
            + np.array([[3.0, 1.0, 0.0], [1.0, -2.0, 0.5], [0.0, 0.5, 1.0]])
            * STRAIN_SCALE
        )
        matrix = rotation_about((0.2, 0.9, -0.4), 2.0)
        f = matrix @ stretch
        _, recovered = polar_decomposition(f)
        expected = 0.5 * (f.T @ f - np.eye(3))
        np.testing.assert_allclose(
            strain_from_stretch(recovered, "green-lagrange"),
            expected,
            rtol=0,
            atol=SOLVER_TOL,
        )
        # and it differs from Biot at second order, so the default is
        # not silently the other one
        assert not np.allclose(expected, stretch - np.eye(3), atol=1e-8)

    def test_the_decomposition_stacks(self):
        rng = np.random.default_rng(5)
        f = np.eye(3) + rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=(4, 3, 3))
        rotations, stretches = polar_decomposition(f)
        assert rotations.shape == (4, 3, 3)
        assert stretches.shape == (4, 3, 3)
        for i in range(4):
            np.testing.assert_allclose(
                rotations[i] @ stretches[i], f[i], rtol=0, atol=SOLVER_TOL
            )
            np.testing.assert_allclose(
                stretches[i], stretches[i].T, rtol=0, atol=SOLVER_TOL
            )
            assert np.linalg.det(rotations[i]) == pytest.approx(1.0, abs=SOLVER_TOL)

    def test_supported_strain_measures_are_frozen(self):
        assert SUPPORTED_STRAIN_MEASURES == ("biot", "green-lagrange")

    def test_guards(self):
        with pytest.raises(ValueError):
            polar_decomposition(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="biot"):
            strain_from_stretch(np.eye(3), "hencky")
        with pytest.raises(ValueError):
            rotation_vector_from_matrix(np.zeros((2, 2)))


class TestSmallStrainFastPath:
    """The requirements D8 fast path, as a MEASUREMENT HARNESS.  It
    is not enabled by anything public until its band and its enabling
    angle are recorded.  [D8]"""

    def test_a_stack_is_split_point_by_point(self):
        # ADDED 2026-09-08 (Stage B adversarial review, the coverage
        # gate): the stacked return of this private fast path was never
        # reached, so a stacked call could have returned the single
        # case's shape and nothing would have noticed
        rng = np.random.default_rng(11)
        stack = rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=(4, 3, 3))
        strain, rotation = small_strain_split(stack)
        assert strain.shape == (4, 3, 3)
        assert rotation.shape == (4, 3, 3)
        for point in range(4):
            one_strain, one_rotation = small_strain_split(stack[point])
            np.testing.assert_allclose(
                strain[point], one_strain, rtol=0, atol=ALGEBRA_TOL
            )
            np.testing.assert_allclose(
                rotation[point], one_rotation, rtol=0, atol=ALGEBRA_TOL
            )

    def test_the_split_is_literally_symmetric_and_antisymmetric(self):
        rng = np.random.default_rng(7)
        beta = rng.uniform(-STRAIN_SCALE, STRAIN_SCALE, size=(3, 3))
        strain, rotation = small_strain_split(beta)
        np.testing.assert_allclose(
            strain, 0.5 * (beta + beta.T), rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            rotation, 0.5 * (beta - beta.T), rtol=0, atol=ALGEBRA_TOL
        )

    @staticmethod
    def equality_error(angle, stretch):
        """Return the worst strain difference between the two paths
        at one rotation angle."""
        f = rotation_about((0.2, 0.9, -0.4), angle) @ stretch
        _, recovered = polar_decomposition(f)
        polar_strain = strain_from_stretch(recovered)
        small_strain, _ = small_strain_split(f - np.eye(3))
        return float(np.abs(polar_strain - small_strain).max())

    def test_equality_with_the_polar_path_is_measured(self):
        # the harness of requirements D8: the difference is second
        # order in the rotation, so it is swept and recorded rather
        # than guessed.
        #
        # The enabling angle is BISECTED and not read off the sweep
        # grid (corrected 2026-09-07 at the Stage B failing-tests
        # review).  The drafted arm recorded the largest SWEPT angle
        # whose error was still inside the 1e-6 of requirements D8,
        # which is an artefact of the grid rather than a measurement:
        # the errors on this sweep are 9.140e-08, 5.656e-07,
        # 1.604e-06, 3.733e-05, 1.479e-04 and 5.889e-04, so the grid
        # would record exactly 0.05 deg while the criterion is
        # actually crossed near 0.0779 deg.  Requirements D8 asks for
        # the angle the fast path stays valid UP TO, so the crossing
        # is what belongs in the record
        stretch = (
            np.eye(3)
            + np.array([[2.0, 0.7, -0.4], [0.7, -1.5, 0.3], [-0.4, 0.3, 1.1]])
            * STRAIN_SCALE
        )
        worst = {
            angle: self.equality_error(angle, stretch)
            for angle in (0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
        }
        assert_within(
            max(worst.values()), SMALL_STRAIN_EQUALITY_TOL, "SMALL_STRAIN_EQUALITY_TOL"
        )
        inside = [a for a, error in worst.items() if error <= FAST_PATH_CRITERION]
        outside = [a for a, error in worst.items() if error > FAST_PATH_CRITERION]
        # the sweep has to BRACKET the crossing, or there is nothing
        # to bisect and the recorded angle would again be a grid point
        assert inside and outside
        low, high = max(inside), min(outside)
        assert low < high
        for _ in range(40):
            middle = 0.5 * (low + high)
            if self.equality_error(middle, stretch) <= FAST_PATH_CRITERION:
                low = middle
            else:
                high = middle
        assert high - low < 1e-6
        assert_at_least(
            low, SMALL_STRAIN_ENABLE_ANGLE_DEG, "SMALL_STRAIN_ENABLE_ANGLE_DEG"
        )

    def test_the_public_path_is_the_polar_one(self):
        # WHICH SPLIT the public result took, decided on the chain's
        # OWN reported ``beta``, which is the closed sample-frame
        # tensor the split consumes (pinned by
        # ``TestChain::test_the_beta_property_is_the_closed_sample
        # _frame_tensor``).  Both candidate answers are then computed
        # from that same tensor, so the comparison sees the SPLIT
        # alone and nothing upstream of it.
        #
        # CORRECTED 2026-09-07 at the Stage B failing-tests review.
        # The drafted arm asserted the reported strain of a two degree
        # pure rotation to be below a tenth of ``sym(R - I)``, on the
        # premise that "the polar path leaks none".  That premise is
        # true of the SPLIT (which
        # ``TestPolarDecomposition::test_pure_rotation_leaks_no_strain``
        # pins at 6.7e-16) and false of the CHAIN, and no conformant
        # implementation could satisfy it: MEASURED on this module's
        # own detector, the reduction by ``Fe33`` is an isotropic
        # dilatation of 7.13e-05 at two degrees -- already above the
        # 6.09e-05 bound on its own -- and the ninth-degree-of-freedom
        # closure then adds ``-tr(beta)/3``, so the chain reports
        # 4.0614e-04 under the deviatoric closure (exactly ``2/3`` of
        # ``sym(R - I)``, which is algebra: the closure turns
        # ``diag(c-1, c-1, 0)`` into ``diag(-(1-c)/3, -(1-c)/3,
        # 2(1-c)/3)``) and 2.9104e-04 under the traction-free one.
        # That floor is a property of the reduction plus the closure,
        # not of the split, so it cannot discriminate between them
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        beta_sample = rotation_about((0.0, 0.0, 1.0), 2.0) - np.eye(3)
        fe = reduced_fe(beta_sample, matrix)
        xmap = crystal_map_with(fe.ravel(), navigation_shape=(1, 2))
        result = hrebsd_strain_stress(xmap, detector)
        strain = np.asarray(result.prop["strain"])[0]
        closed = np.asarray(result.prop["beta"])[0].reshape(3, 3)
        _, stretch = polar_decomposition(np.eye(3) + closed)
        polar_strain = tensor_to_voigt_vector(strain_from_stretch(stretch))
        fast_strain = tensor_to_voigt_vector(small_strain_split(closed)[0])
        # the two candidates really do differ here, by 6.09e-04, so
        # the assertion below discriminates
        separation = float(np.abs(polar_strain - fast_strain).max())
        assert separation > 1e-4
        # and the reported strain IS the polar one, to the band an
        # identity of the same numbers is asserted in
        np.testing.assert_allclose(strain, polar_strain, rtol=0, atol=SOLVER_TOL)
        # and it is NOT the fast path's answer
        assert float(np.abs(strain - fast_strain).max()) > 1e-4


# ================ D9 -- the ninth degree of freedom ================= #


class TestClosure:
    """The traction-free solve against an independently built tensor,
    the V3 tensor half, plus the deviatoric fallback.  [D9]"""

    @pytest.mark.parametrize("gauge", [0.0, -2.3e-3, 4.1e-3])
    def test_traction_free_recovers_the_built_in_tensor(self, gauge):
        # THE V3 tensor half.  A tensor is built here whose symmetric
        # part carries ``sigma33 = 0`` exactly, then the isotropic
        # part the projection cannot see is destroyed by adding
        # ``gauge * I``, and the closure has to put back exactly what
        # was removed -- so ``e33`` comes back EXACTLY, not to a band.
        # The gauge is swept because the reduction by ``Fe33`` leaves
        # an arbitrary isotropic offset, not the zero one
        stiffness = generic_sample_stiffness()
        beta_true, strain_true = traction_free_beta(stiffness)
        measured = beta_true + gauge * np.eye(3)
        closed = close_beta(
            measured, stiffness_sample=stiffness, closure="traction_free"
        )
        np.testing.assert_allclose(closed, beta_true, rtol=0, atol=SOLVER_TOL)
        assert closed[2, 2] == pytest.approx(strain_true[2, 2], abs=SOLVER_TOL)
        # and the self-check: the closed strain really is traction
        # free, measured with a Hooke product written out here
        stress = hooke_product(stiffness, voigt_vector_here(closed))
        scale = float(np.abs(stress).max())
        assert abs(stress[2]) < 1e-9 * max(scale, 1.0)

    def test_the_shear_terms_of_the_right_hand_side_are_used(self):
        # the closure's first row carries ``C3312``, ``C3313`` and
        # ``C3323``, which are identically zero for an UNROTATED cubic
        # matrix: a mutant which drops the shear terms survives every
        # test built on one.  Here the generic orientation makes them
        # large, and the same tensor closed against the unrotated
        # matrix gives a measurably different e33
        stiffness = generic_sample_stiffness()
        assert np.abs(stiffness[2, 3:]).max() > 1.0
        beta_true, _ = traction_free_beta(stiffness)
        measured = beta_true - beta_true[2, 2] * np.eye(3)
        closed = close_beta(measured, stiffness_sample=stiffness)
        unrotated = np.array(
            [[c11c12(i, j) for j in range(VOIGT_SIZE)] for i in range(VOIGT_SIZE)]
        )
        other = close_beta(measured, stiffness_sample=unrotated)
        assert np.abs(closed[2, 2] - other[2, 2]) > 1e-6

    def test_b7_and_b8_are_not_interchangeable(self):
        # What this test says, restated 2026-09-07 at the Stage B
        # failing-tests review: the closure reads ``b7`` and ``b8``
        # through DIFFERENT rows, so its answer is not symmetric under
        # exchanging them.  It is NOT the killer of the plan section
        # 3.4 "b7/b8 misassigned" mutant, which the drafted comment
        # claimed: MEASURED by injection, the swapped implementation
        # gives the identical 1.4711e-05 separation below (the two
        # answers are merely exchanged), so the first assertion passes
        # either way.  That mutant dies at
        # ``test_traction_free_recovers_the_built_in_tensor``, by
        # 1.4711e-05 against SOLVER_TOL = 1e-10, and at the second
        # arm here.
        stiffness = generic_sample_stiffness()
        beta_true, _ = traction_free_beta(stiffness)
        measured = beta_true - beta_true[2, 2] * np.eye(3)
        swapped = measured.copy()
        swapped[0, 0], swapped[1, 1] = measured[1, 1], measured[0, 0]
        assert abs(measured[0, 0] - measured[1, 1]) > 1e-4
        first = close_beta(measured, stiffness_sample=stiffness)
        second = close_beta(swapped, stiffness_sample=stiffness)
        assert abs(first[2, 2] - second[2, 2]) > 1e-6
        # and THE discriminating arm: each closed diagonal entry
        # reproduces the one it was built from, which a swap breaks by
        # 1.4711e-05 in e11 and e22 while leaving the separation above
        # untouched
        assert first[0, 0] == pytest.approx(beta_true[0, 0], abs=SOLVER_TOL)
        assert first[1, 1] == pytest.approx(beta_true[1, 1], abs=SOLVER_TOL)
        assert abs(beta_true[0, 0] - beta_true[1, 1]) > 1e-4

    def test_the_off_diagonals_are_untouched(self):
        # closing adds a multiple of the identity and nothing else
        stiffness = generic_sample_stiffness()
        beta_true, _ = traction_free_beta(stiffness, seed=2)
        closed = close_beta(beta_true, stiffness_sample=stiffness)
        difference = closed - beta_true
        np.testing.assert_allclose(
            difference - np.trace(difference) / 3 * np.eye(3),
            np.zeros((3, 3)),
            rtol=0,
            atol=SOLVER_TOL,
        )

    def test_deviatoric_is_traceless(self):
        stiffness = generic_sample_stiffness()
        beta_true, _ = traction_free_beta(stiffness, seed=3)
        closed = close_beta(beta_true, closure="deviatoric")
        assert np.trace(closed) == pytest.approx(0.0, abs=SOLVER_TOL)
        # and it is the same tensor up to the isotropic part
        difference = closed - beta_true
        np.testing.assert_allclose(
            difference - np.trace(difference) / 3 * np.eye(3),
            np.zeros((3, 3)),
            rtol=0,
            atol=SOLVER_TOL,
        )

    def test_auto_follows_the_stiffness(self):
        stiffness = generic_sample_stiffness()
        beta_true, _ = traction_free_beta(stiffness, seed=4)
        measured = beta_true - beta_true[2, 2] * np.eye(3)
        np.testing.assert_allclose(
            close_beta(measured, stiffness_sample=stiffness, closure="auto"),
            close_beta(measured, stiffness_sample=stiffness, closure="traction_free"),
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        np.testing.assert_allclose(
            close_beta(measured, closure="auto"),
            close_beta(measured, closure="deviatoric"),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_the_two_closures_differ_on_a_non_deviatoric_tensor(self):
        # the fallback-always mutant: on a tensor whose traction-free
        # closure is NOT traceless the two answers must differ
        stiffness = generic_sample_stiffness()
        beta_true, _ = traction_free_beta(stiffness, seed=5)
        measured = beta_true - beta_true[2, 2] * np.eye(3)
        traction_free = close_beta(
            measured, stiffness_sample=stiffness, closure="traction_free"
        )
        deviatoric = close_beta(measured, closure="deviatoric")
        assert abs(np.trace(traction_free)) > 1e-5
        assert abs(traction_free[2, 2] - deviatoric[2, 2]) > 1e-5

    def test_the_closure_stacks(self):
        stiffness = generic_sample_stiffness()
        betas = np.stack(
            [traction_free_beta(stiffness, seed=seed)[0] for seed in range(4)]
        )
        stacked = close_beta(betas, stiffness_sample=stiffness)
        assert stacked.shape == (4, 3, 3)
        for i in range(4):
            np.testing.assert_allclose(
                stacked[i],
                close_beta(betas[i], stiffness_sample=stiffness),
                rtol=0,
                atol=SOLVER_TOL,
            )

    def test_supported_closures_are_frozen(self):
        assert SUPPORTED_CLOSURES == ("auto", "traction_free", "deviatoric")

    def test_guards(self):
        stiffness = generic_sample_stiffness()
        beta = np.zeros((3, 3))
        with pytest.raises(ValueError, match="deviatoric"):
            close_beta(beta, stiffness_sample=stiffness, closure="plane_stress")
        # traction free without a stiffness names the missing argument
        # rather than falling back silently
        with pytest.raises(ValueError, match="stiffness"):
            close_beta(beta, closure="traction_free")


# ==================== D10 -- the derived maps ======================= #


class TestDerivedMaps:
    """Identities on hand-built stress tensors.  [D10]"""

    def test_uniaxial_stress(self):
        stress = np.zeros(VOIGT_SIZE)
        stress[0] = 1.5
        assert von_mises_stress(stress) == pytest.approx(1.5, abs=SOLVER_TOL)
        assert hydrostatic_stress(stress) == pytest.approx(0.5, abs=SOLVER_TOL)
        np.testing.assert_allclose(
            principal_stresses(stress), [1.5, 0.0, 0.0], rtol=0, atol=SOLVER_TOL
        )

    def test_pure_shear_stress(self):
        stress = np.zeros(VOIGT_SIZE)
        stress[5] = 0.7
        assert von_mises_stress(stress) == pytest.approx(
            np.sqrt(3) * 0.7, abs=SOLVER_TOL
        )
        assert hydrostatic_stress(stress) == pytest.approx(0.0, abs=SOLVER_TOL)
        np.testing.assert_allclose(
            principal_stresses(stress), [0.7, 0.0, -0.7], rtol=0, atol=SOLVER_TOL
        )

    def test_von_mises_ignores_the_hydrostatic_part(self):
        rng = np.random.default_rng(13)
        stress = rng.uniform(-1.0, 1.0, size=VOIGT_SIZE)
        shifted = stress.copy()
        shifted[:3] += 0.9
        assert von_mises_stress(shifted) == pytest.approx(
            von_mises_stress(stress), abs=SOLVER_TOL
        )
        assert hydrostatic_stress(shifted) == pytest.approx(
            hydrostatic_stress(stress) + 0.9, abs=SOLVER_TOL
        )

    def test_principal_stresses_descend(self):
        # the plan section 3.4 "principal stresses ascending" mutant,
        # which is what ``numpy.linalg.eigvalsh`` returns unreversed
        stress = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0])
        principal = principal_stresses(stress)
        assert principal[0] >= principal[1] >= principal[2]
        np.testing.assert_allclose(principal, [2.0, 0.5, -1.0], rtol=0, atol=SOLVER_TOL)
        # and the sum is the trace, whatever the order
        assert principal.sum() == pytest.approx(stress[:3].sum(), abs=SOLVER_TOL)

    def test_they_stack(self):
        rng = np.random.default_rng(17)
        stress = rng.uniform(-1.0, 1.0, size=(5, VOIGT_SIZE))
        assert von_mises_stress(stress).shape == (5,)
        assert hydrostatic_stress(stress).shape == (5,)
        assert principal_stresses(stress).shape == (5, PRINCIPAL_PROP_SIZE)
        for i in range(5):
            assert von_mises_stress(stress)[i] == pytest.approx(
                von_mises_stress(stress[i]), abs=SOLVER_TOL
            )

    def test_the_voigt_vector_carries_tensor_shears(self):
        tensor = np.array(
            [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]], dtype=np.float64
        )
        np.testing.assert_allclose(
            tensor_to_voigt_vector(tensor),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_the_voigt_vector_takes_the_symmetric_part(self):
        # ADDED 2026-09-08 at the Stage B adversarial review, where
        # dropping the symmetrization SURVIVED the whole suite: the
        # only caller feeds this function ``strain_from_stretch``,
        # whose worst asymmetry is 3.4e-17, so on the chain's own path
        # the difference is identically zero.  The docstring
        # nevertheless promises that the symmetric part is taken, and
        # requirements D15.6 makes this vector a PUBLIC property with
        # TENSOR shears, so a second consumer -- the Stage C
        # antisymmetry fix, which works on a deliberately
        # NON-symmetric displacement gradient -- would silently read
        # every shear twice over
        tensor = np.array(
            [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64
        )
        assert not np.allclose(tensor, tensor.T)
        np.testing.assert_allclose(
            tensor_to_voigt_vector(tensor),
            [1.0, 4.0, 6.0, 2.5, 1.5, 1.0],
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        # the unsymmetrized reading, a factor of two out on every
        # shear, is excluded
        assert not np.allclose(
            tensor_to_voigt_vector(tensor), [1.0, 4.0, 6.0, 5.0, 3.0, 2.0]
        )
        # and a stack takes it point by point
        stacked = tensor_to_voigt_vector(np.stack([tensor, tensor.T]))
        assert stacked.shape == (2, VOIGT_SIZE)
        np.testing.assert_allclose(stacked[0], stacked[1], rtol=0, atol=ALGEBRA_TOL)

    def test_guards(self):
        with pytest.raises(ValueError):
            von_mises_stress(np.zeros(5))
        with pytest.raises(ValueError):
            hydrostatic_stress(np.zeros(5))
        with pytest.raises(ValueError):
            principal_stresses(np.zeros(5))
        with pytest.raises(ValueError):
            tensor_to_voigt_vector(np.zeros((2, 2)))


# ============== The chain and the public function =================== #


class TestChain:
    """``tensor_chain`` and ``hrebsd_strain_stress`` end to end on
    analytically built deformation gradients.  [D7-D10/D15]"""

    @staticmethod
    def imposed_map(navigation_shape=(1, 2), seed=0):
        """Return the detector, the imposed sample-frame tensors and
        a crystal map carrying their reduced detector-frame form."""
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        stiffness = generic_sample_stiffness()
        size = int(np.prod(navigation_shape))
        betas, strains = [], []
        for point in range(size):
            beta, strain = traction_free_beta(stiffness, seed=seed + point)
            betas.append(beta)
            strains.append(strain)
        fe = np.stack([reduced_fe(beta, matrix) for beta in betas])
        xmap = crystal_map_with(
            fe.reshape(size, BETA_PROP_SIZE), navigation_shape=navigation_shape
        )
        return detector, np.stack(betas), np.stack(strains), xmap

    def test_strain_recovery_at_a_generic_orientation(self):
        # the V3 tensor half through the WHOLE chain, at a generic
        # orientation so that the crystal to sample stiffness rotation
        # is exercised: transposing it changes the sample-frame
        # ``C3311``, ``C3322`` and the three shear terms and so
        # changes the recovered e33
        detector, _, strains, xmap = self.imposed_map(navigation_shape=(1, 3))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        strain = np.asarray(result.prop["strain"])
        worst = 0.0
        for point, imposed in enumerate(strains):
            expected = voigt_vector_here(imposed)
            worst = max(worst, float(np.abs(strain[point] - expected).max()))
        # the imposed strains are of the 1e-3 scale, so a recovery
        # good to 1e-6 is three orders below what is measured
        assert np.abs(strains).max() > 1e-4
        assert_within(worst, REDUCED_CLOSURE_TOL, "REDUCED_CLOSURE_TOL")

    def test_e33_is_the_closure_derived_one(self):
        # ``e33`` is not measured anywhere: it is what the closure
        # supplies, and the tensor was built with the value the
        # traction-free assumption implies
        detector, _, strains, xmap = self.imposed_map()
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        strain = np.asarray(result.prop["strain"])
        for point, imposed in enumerate(strains):
            assert_within(
                abs(strain[point, 2] - imposed[2, 2]),
                REDUCED_CLOSURE_TOL,
                "REDUCED_CLOSURE_TOL (e33)",
            )

    def test_sigma33_is_the_closure_self_check(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(1, 3))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        stress = np.asarray(result.prop["stress"])
        # the stress really is of the GPa scale, so a near-zero
        # ``sigma33`` is a statement and not an artefact of everything
        # being small
        assert np.abs(stress).max() > 1e-2
        assert_within(float(np.abs(stress[:, 2]).max()), SIGMA33_TOL, "SIGMA33_TOL")

    def test_the_deviatoric_route_is_different_and_carries_no_stress(self):
        detector, _, _, xmap = self.imposed_map()
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        traction_free = np.asarray(
            hrebsd_strain_stress(xmap, detector, stiffness=stiffness).prop["strain"]
        )
        deviatoric = np.asarray(hrebsd_strain_stress(xmap, detector).prop["strain"])
        assert np.abs(traction_free[:, 2] - deviatoric[:, 2]).max() > 1e-5
        # without a stiffness there is no stress, and the properties
        # are still all present, filled with NaN
        result = hrebsd_strain_stress(xmap, detector)
        for name in (
            "stress",
            "stress_von_mises",
            "stress_hydrostatic",
            "stress_principal",
        ):
            assert np.all(np.isnan(np.asarray(result.prop[name]))), name
        assert np.all(np.isfinite(np.asarray(result.prop["strain"])))

    def test_the_derived_stress_maps_follow_the_stress(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(1, 3))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        stress = np.asarray(result.prop["stress"])
        np.testing.assert_allclose(
            np.asarray(result.prop["stress_von_mises"]),
            von_mises_stress(stress),
            rtol=0,
            atol=SOLVER_TOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.prop["stress_hydrostatic"]),
            hydrostatic_stress(stress),
            rtol=0,
            atol=SOLVER_TOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.prop["stress_principal"]),
            principal_stresses(stress),
            rtol=0,
            atol=SOLVER_TOL,
        )

    def test_the_stress_is_the_hooke_product_of_the_reported_strain(self):
        # the stress is not computed from some other strain than the
        # one reported, and the engineering factor lives in exactly
        # one place
        detector, _, _, xmap = self.imposed_map()
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        rotated = rotate_here(stiffness, generic_orientation_matrix())
        expected = hooke_product(rotated, np.asarray(result.prop["strain"]))
        np.testing.assert_allclose(
            np.asarray(result.prop["stress"]), expected, rtol=0, atol=SOLVER_TOL
        )

    def test_the_beta_property_is_the_closed_sample_frame_tensor(self):
        detector, betas, _, xmap = self.imposed_map()
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        beta = np.asarray(result.prop["beta"]).reshape(-1, 3, 3)
        # row-major flattening, and the off-diagonals are the imposed
        # ones to the reduction's own second order
        for point, imposed in enumerate(betas):
            assert_within(
                float(np.abs(beta[point] - imposed).max()),
                REDUCED_CLOSURE_TOL,
                "REDUCED_CLOSURE_TOL (beta)",
            )

    def test_the_rotation_vector_is_the_imposed_rotation(self):
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        axis, angle = np.array([0.1, -0.3, 0.95]), 0.4
        beta_sample = rotation_about(axis, angle) - np.eye(3)
        fe = reduced_fe(beta_sample, matrix)
        xmap = crystal_map_with(fe.ravel(), navigation_shape=(1, 2))
        result = hrebsd_strain_stress(xmap, detector)
        expected = axis / np.linalg.norm(axis) * np.deg2rad(angle)
        got = np.asarray(result.prop["rotation_vector"])[0]
        assert_within(
            float(np.abs(got - expected).max()),
            REDUCED_CLOSURE_TOL,
            "REDUCED_CLOSURE_TOL (rotation vector)",
        )
        # in RADIANS, not degrees or milliradians
        assert np.abs(got).max() < 1e-2

    def test_property_names_shapes_and_dtypes(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(2, 3))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        assert isinstance(result, CrystalMap)
        assert set(STAGE_B_PROP_NAMES) <= set(result.prop)
        size = 6
        expected = {
            "strain": (size, STRAIN_PROP_SIZE),
            "rotation_vector": (size, ROTATION_VECTOR_PROP_SIZE),
            "beta": (size, BETA_PROP_SIZE),
            "stress": (size, VOIGT_SIZE),
            "stress_von_mises": (size,),
            "stress_hydrostatic": (size,),
            "stress_principal": (size, PRINCIPAL_PROP_SIZE),
        }
        for name, shape in expected.items():
            array = np.asarray(result.prop[name])
            assert array.shape == shape, name
            assert array.dtype == np.float64, name

    def test_stage_b_property_names_are_frozen(self):
        assert STAGE_B_PROP_NAMES == (
            "strain",
            "rotation_vector",
            "beta",
            "stress",
            "stress_von_mises",
            "stress_hydrostatic",
            "stress_principal",
        )

    def test_the_input_map_is_not_modified(self):
        detector, _, _, xmap = self.imposed_map()
        before = set(xmap.prop)
        result = hrebsd_strain_stress(xmap, detector)
        assert set(xmap.prop) == before
        assert result is not xmap
        # and the orientations and phases come through untouched
        # (requirements D7: this feature never modifies an
        # orientation)
        np.testing.assert_allclose(
            result.rotations.data, xmap.rotations.data, rtol=0, atol=0
        )
        assert result.phases.names == xmap.phases.names
        assert result.scan_unit == xmap.scan_unit

    def test_the_documented_reshape_route(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(2, 3))
        result = hrebsd_strain_stress(xmap, detector)
        strain = np.asarray(result.prop["strain"]).reshape(2, 3, STRAIN_PROP_SIZE)
        assert strain.shape == (2, 3, STRAIN_PROP_SIZE)
        np.testing.assert_allclose(
            strain[1, 2], np.asarray(result.prop["strain"])[5], rtol=0, atol=0
        )

    def test_nan_rows_stay_nan(self):
        # requirements D2.6: a point which did not converge, failed or
        # was masked out carries NaN in every derived property, and is
        # never zeroed
        detector, _, _, xmap = self.imposed_map(navigation_shape=(1, 3))
        fe = np.asarray(xmap.prop["Fe"]).copy()
        fe[1] = np.nan
        xmap.prop["Fe"] = fe
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        for name in STAGE_B_PROP_NAMES:
            array = np.asarray(result.prop[name])
            assert np.all(np.isnan(array[1])), name
            assert np.all(np.isfinite(array[0])), name
            assert np.all(np.isfinite(array[2])), name

    def test_two_runs_are_bitwise_identical(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(2, 3))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        first = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        second = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        for name in STAGE_B_PROP_NAMES:
            assert np.array_equal(
                np.asarray(first.prop[name]),
                np.asarray(second.prop[name]),
                equal_nan=True,
            ), name

    def test_the_chain_and_the_public_function_agree(self):
        # plan section 3.2: ONE chain, shared, so the public function
        # is a thin wrapper over it and Stage C's GND path can call
        # the same thing
        detector, _, _, xmap = self.imposed_map(navigation_shape=(1, 2))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        # the SAME matrices the public function reads, so that the
        # bitwise claim below is about the chain and not about two
        # constructions of one orientation.  CORRECTED 2026-09-07 at
        # the Stage B failing-tests review: the drafted arm built them
        # with this module's own Rodrigues helper, which differs from
        # ``Rotation.to_matrix`` by one ulp (MEASURED: maxdiff
        # 2.7755575615628914e-17), and that ulp reaches the output --
        # ``stress`` differs by up to 5.55e-17 and ``beta`` by 1.08e-19
        # through the stiffness rotation, which ``atol=0`` rejects.
        # Substituting costs no independence: that the crystal map's
        # rotations ARE the ``v_crystal = g @ v_sample`` matrix the
        # rotation of C takes is pinned separately, and against
        # ``rotate_vector`` rather than against itself, by
        # ``test_hrebsd_stiffness.py::TestRotationDirection::test_the
        # _matrix_is_the_kikuchipy_orientation``
        matrices = xmap.rotations.to_matrix()
        # ... and it really is this module's generic orientation
        np.testing.assert_allclose(
            matrices[0], generic_orientation_matrix(), rtol=0, atol=ALGEBRA_TOL
        )
        direct = tensor_chain(
            np.asarray(xmap.prop["Fe"]), matrices, detector, stiffness=stiffness
        )
        result = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        for name in STAGE_B_PROP_NAMES:
            np.testing.assert_allclose(
                direct[name], np.asarray(result.prop[name]), rtol=0, atol=0
            )

    def test_the_chain_takes_both_fe_layouts(self):
        detector, _, _, xmap = self.imposed_map(navigation_shape=(1, 2))
        matrices = np.stack([generic_orientation_matrix()] * xmap.size)
        flat = np.asarray(xmap.prop["Fe"])
        first = tensor_chain(flat, matrices, detector)
        second = tensor_chain(flat.reshape(-1, 3, 3), matrices, detector)
        for name in STAGE_B_PROP_NAMES:
            np.testing.assert_allclose(first[name], second[name], rtol=0, atol=0)

    def test_signature_is_frozen(self):
        parameters = inspect.signature(hrebsd_strain_stress).parameters
        assert list(parameters) == [
            "xmap",
            "detector",
            "stiffness",
            "closure",
            "strain_measure",
        ]
        for name in ("xmap", "detector"):
            assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                name
            )
        assert parameters["stiffness"].default is None
        assert parameters["closure"].default == "auto"
        assert parameters["strain_measure"].default == "biot"
        for name in ("stiffness", "closure", "strain_measure"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name

    def test_the_private_chain_names_its_own_shapes(self):
        # ADDED 2026-09-08 (Stage B adversarial review, the coverage
        # gate): both shape guards of ``tensor_chain`` were unreached.
        # They belong to the PRIVATE chain, whose contract the public
        # function no longer breaks -- which is exactly why they are
        # tested here, directly, rather than through it
        detector = make_detector()
        fe = np.tile(np.eye(3).ravel(), (3, 1))
        with pytest.raises(ValueError, match=r"orientation_matrices"):
            tensor_chain(fe, np.tile(np.eye(3), (2, 1, 1)), detector)
        with pytest.raises(ValueError, match=r"\(n, 9\)"):
            tensor_chain(np.zeros((3, 4)), np.tile(np.eye(3), (3, 1, 1)), detector)
        # and the two accepted layouts of the stored property agree
        by_rows = tensor_chain(fe, np.tile(np.eye(3), (3, 1, 1)), detector)
        by_matrices = tensor_chain(
            fe.reshape(3, 3, 3), np.tile(np.eye(3), (3, 1, 1)), detector
        )
        np.testing.assert_array_equal(by_rows["strain"], by_matrices["strain"])

    def test_guards(self):
        detector, _, _, xmap = self.imposed_map()
        # a map which never went through the engine names the method
        # that fills the property
        arrays, size = create_coordinate_arrays((1, 2), (1.0, 1.0))
        arrays["rotations"] = Rotation.identity((size,))
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        with pytest.raises(ValueError, match="hrebsd_dic"):
            hrebsd_strain_stress(CrystalMap(**arrays), detector)
        with pytest.raises(ValueError, match="closure"):
            hrebsd_strain_stress(xmap, detector, closure="plane_stress")
        with pytest.raises(ValueError, match="strain_measure"):
            hrebsd_strain_stress(xmap, detector, strain_measure="hencky")
        with pytest.raises(ValueError, match="stiffness"):
            hrebsd_strain_stress(xmap, detector, closure="traction_free")
        with pytest.raises(ValueError):
            hrebsd_strain_stress(xmap, detector, stiffness=np.zeros((3, 3)))

    def test_a_map_with_several_rotations_per_point_takes_the_best(self):
        # ADDED 2026-09-08 (Stage B adversarial review).  A map from
        # dictionary indexing with ``n_best > 1`` carries ``(n, k)``
        # rotations, which ``segment_grains`` already reads at their
        # best entry; this function used to raise instead, with a
        # message naming ``orientation_matrices``, an argument of the
        # private chain that the public caller never passed.  The two
        # now agree, and the best rotation is the map's own
        detector, _, _, single = self.imposed_map(navigation_shape=(1, 2))
        best = single.rotations
        arrays, size = create_coordinate_arrays((1, 2), (1.0, 1.0))
        stacked = np.stack([best.data, Rotation.identity((2,)).data], axis=1)
        arrays["rotations"] = Rotation(stacked)
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        several = CrystalMap(**arrays)
        several.prop["Fe"] = np.asarray(single.prop["Fe"])
        assert several.rotations.to_matrix().shape == (2, 2, 3, 3)
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        got = hrebsd_strain_stress(several, detector, stiffness=stiffness)
        expected = hrebsd_strain_stress(single, detector, stiffness=stiffness)
        for name in ("strain", "stress", "beta", "rotation_vector"):
            np.testing.assert_array_equal(
                np.asarray(got.prop[name]),
                np.asarray(expected.prop[name]),
                err_msg=name,
            )
        # and the second rotation really is a different one, so the
        # test would fail on a chain that took the last entry
        assert not np.allclose(stacked[:, 0], stacked[:, 1])

    def test_a_map_of_only_non_converged_points_gives_the_whole_nan_set(self):
        # ADDED 2026-09-08 (Stage B adversarial review, the coverage
        # gate): the early return of requirements D2.6, which no test
        # reached.  Every key of the frozen property set must still be
        # there, at the full length, so that a downstream reader never
        # meets a missing key on a map that happened to fail entirely
        detector = make_detector()
        xmap = crystal_map_with(
            np.full((6, BETA_PROP_SIZE), np.nan), navigation_shape=(2, 3)
        )
        result = hrebsd_strain_stress(xmap, detector)
        for name in STAGE_B_PROP_NAMES:
            value = np.asarray(result.prop[name])
            assert value.shape[0] == 6, name
            assert np.all(np.isnan(value)), name

    def test_the_single_phase_stress_guard(self):
        # requirements D9.6: version one takes ONE stiffness, so a
        # multi-phase map is refused on the stress path with a message
        # naming the limitation
        detector = make_detector()
        matrix = spec_frame_matrix(detector)
        fe = np.tile(reduced_fe(np.zeros((3, 3)), matrix).ravel(), (4, 1))
        xmap = crystal_map_with(fe, navigation_shape=(2, 2), phase_ids=[0, 0, 1, 1])
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        with pytest.raises(ValueError, match="phase"):
            hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        # without a stiffness there is no stress and no limitation
        result = hrebsd_strain_stress(xmap, detector)
        assert np.all(np.isfinite(np.asarray(result.prop["strain"])))


# ====================== Exports and the API ========================= #


class TestExports:
    """The five Stage B public names of requirements D15.3.  Pure
    introspection, so these pass today: they are the freeze.
    [D15.3]"""

    @pytest.mark.parametrize(
        "name",
        [
            "hrebsd_kam",
            "hrebsd_pc_shift",
            "hrebsd_strain_stress",
            "segment_grains",
            "voigt_stiffness",
        ],
    )
    def test_the_name_resolves_through_the_lazy_loader(self, name):
        assert hasattr(kp.indexing, name)
        assert name in kp.indexing.__all__

    def test_all_is_sorted(self):
        assert list(kp.indexing.__all__) == sorted(kp.indexing.__all__)

    def test_hrebsd_gnd_is_exported_by_stage_c(self):
        # SUPERSEDES ``test_hrebsd_gnd_is_not_exported_yet``, deleted
        # 2026-09-08 at the Stage C failing-tests gate.  That pin said
        # the sixth frozen name of requirements D15.3 must NOT appear
        # in the reference before its function exists, which was right
        # while Stage C was unwritten and is refuted by Stage C itself:
        # ``_gnd.py`` ships with the name registered, so the promise
        # the old pin guarded against is now a delivery.  The positive
        # replacement keeps the freeze -- all six names of D15.3 are
        # exported and ``__all__`` stays sorted, which the two tests
        # above assert -- and the sixth name's own contract lives in
        # ``test_hrebsd_gnd.py``
        assert "hrebsd_gnd" in kp.indexing.__all__
        assert hasattr(kp.indexing, "hrebsd_gnd")

    def test_the_public_function_is_the_module_one(self):
        assert kp.indexing.hrebsd_strain_stress is hrebsd_strain_stress

    def test_every_public_docstring_names_its_frames_and_units(self):
        docstring = hrebsd_strain_stress.__doc__
        lowered = docstring.lower()
        for phrase in ("sample frame", "gpa", "voigt", "radians"):
            assert phrase in lowered, phrase
        # the two conventions which are emphasized on purpose, since
        # both are factor-of-two or interpretation traps
        assert "TENSOR shears" in docstring
        # the closure-derived caveat of requirements D10 is
        # documented, not implied
        assert "CLOSURE DERIVED" in docstring
        assert "reshape" in docstring
