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

"""The TENSOR half of oracle V3 of
``specs/2026-09-07-hrebsd-dic/validation.md``: strain recovery from
DEFORMED-MASTER PATTERNS.

ADDED 2026-09-07 at the Stage B failing-tests adversarial review.
validation V3 names ``test_deformed_master_strain_recovery`` as a
Stage B deliverable and plan section 3.4 names it as a co-killer of
the "Bond rotation transposed" mutant, and the drafted Stage B commit
shipped neither it nor any other pattern-level tensor oracle: the
whole Stage B suite was analytic.  This module supplies it.

**What only this file can see.**  ``test_hrebsd_tensors.py`` builds
its reduced deformation gradients by algebra, so the frame chain it
exercises is the one the test itself wrote down.  Here the tensor is
carried through real projected patterns and the Stage A engine, so
the D7 frame chain, the D6 conversion, the D9.5 crystal to sample
stiffness rotation and the D9.2 closure all have to agree at once
with a deformation imposed OUTSIDE the feature, in the crystal frame,
on the shipped nickel Lambert master.

**File-layout deviation, recorded.**  validation V3 places this test
in ``test_hrebsd_engine.py`` beside the Stage A halves it shares a
projection helper with.  It is delivered here instead so that the
Stage A regression file stays free of Stage B failures; the helpers
it needs are duplicated below rather than imported, because pytest
imports test modules by path and they cannot import one another.  The
duplication is deliberate and small: the projection helper, the frame
matrices and the detector.

**The stiffness is built here**, from the nickel constants and this
module's own fourth-order rotation, so that the expectation does not
rest on :func:`~kikuchipy.indexing.voigt_stiffness`; that builder has
its own pins in ``test_hrebsd_stiffness.py``.  One test compares the
two, so the public route is still the one measured.

Written failing before the implementation, at the Stage B
failing-tests gate: every test which calls the Stage B modules failed
with ``NotImplementedError`` until the tensor chain landed, and passed
unchanged after it (narration corrected to the past tense 2026-09-08,
Stage B adversarial review).
"""

import functools

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd._engine import run_hrebsd_dic
from kikuchipy.indexing._hrebsd._stiffness import VOIGT_INDICES, VOIGT_SIZE
from kikuchipy.indexing._hrebsd._tensors import (
    BETA_PROP_SIZE,
    DETECTOR_Y_FLIP,
    hrebsd_strain_stress,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# The oracle detector, as in ``test_hrebsd_engine.py``: every
# precision claim of this feature is tied to 480 px
SHAPE_480 = (480, 480)
PC_480 = (0.4210, 0.5794, 0.5049)

# The generic crystal orientation of the Stage B suite, deliberately
# away from every symmetry position: at a symmetric one the crystal to
# sample stiffness rotation is unexercised, which is exactly what the
# "Bond rotation transposed" mutant hides behind (requirements D9.5)
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0

# Single-crystal elastic constants of nickel in GPa (Simmons and
# Wang 1971), TEST DATA only: requirements D9.4 freezes that no
# elastic constant database ships with kikuchipy
NICKEL_CUBIC = {"c11": 246.5, "c12": 147.3, "c44": 124.7}

# The strain scale HREBSD is built for
STRAIN_SCALE = 1e-3

# The band of a machine-precision-class algebraic identity, FROZEN
ALGEBRA_TOL = 1e-12


# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders (requirements D19).  Each is replaced by a
# dated measured value at the Stage B implementation gate, with the
# recipe recorded in validation.md "Recorded results".

# MTP [D8/D9/D10, V3 tensor half]: the worst per-component strain
# error of the WHOLE chain -- patterns, engine, frame, closure, split
# -- against the strain imposed on the master.  It is bounded from
# below by the Stage A homography accuracy, whose ``Fe`` band is
# DEFORMED_MASTER_FE_TOL = 3.1e-5, so a value in the 1e-5 class is the
# expectation and the drafting seed of validation V3 is <= 2e-4 per
# component at 480 px.  A value far above that means a frame, a
# closure or the stiffness rotation is wrong rather than the DIC.
# MEASURING RECIPE: ``TestDeformedMaster::test_deformed_master_strain
# _recovery`` below; record the worst component over both imposed
# cases
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md Recorded results, Stage B implementation gate entry 39).
# MEASURED worst 9.404545683540204e-06 over the three arms which
# consume it -- strain 9.4045e-06, e33 8.3928e-06, rotation vector
# 4.2064e-06 -- pinned at 2x that worst.  It lands in the 1e-5 class
# the note above predicts from the Stage A ``DEFORMED_MASTER_FE_TOL``
# of 3.1e-5, and the drafting seed of validation V3 (2e-4 per
# component) is 21x looser than achievable.  It kills the pattern-level
# mutant plan 3.4 names: the Bond rotation transposed measures
# 3.0068e-04 here, 16x the pin
DEFORMED_MASTER_STRAIN_TOL = 1.9e-05

# MTP [D9.2/D10, V3]: the residual ``sigma33`` in GPa of the
# traction-free closure through the same route.  Zero by construction
# in the linear closure, so what is measured is the DIC error times
# the stiffness, that is ``|C| * DEFORMED_MASTER_STRAIN_TOL``, a few
# times 1e-3 GPa at the seed above.
# MEASURING RECIPE: ``TestDeformedMaster::test_sigma33_is_zero_through
# _the_patterns``
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md entry 39).  MEASURED 1.8337937617562972e-04 GPa against
# a stress scale of 0.2420 GPa through the same patterns, pinned at 2x.
# The drafted "a few times 1e-3 GPa" estimate above rested on the 2e-4
# strain seed; the measured strain is 21x better, and this number
# follows it
DEFORMED_MASTER_SIGMA33_TOL = 3.7e-04


# ------------- The plan 3.4 mutation list, mapped ------------------- #
#
#  Bond rotation transposed ..... test_deformed_master_strain_recovery
#                                 here, the PATTERN-level co-killer
#                                 plan 3.4 names, beside the D9.5
#                                 22.5 degree C16' sign pin of
#                                 test_hrebsd_stiffness.py and the
#                                 analytic TestChain::test_strain
#                                 _recovery_at_a_generic_orientation.
#                                 NOT the sigma33 self-check, which is
#                                 blind to it (MEASURED 3.0914e-04 GPa
#                                 transposed against 3.0639e-04
#                                 correct on the analytic route)
#  R versus R^T in the frame .... the same test: the conjugation is
#  rotation                       carried through real patterns here,
#                                 which is the arm plan 3.4 expected
#                                 from V4's Stage B split


# ----------------------------- Helpers ------------------------------ #


@functools.lru_cache(maxsize=1)
def ni_master_arrays():
    """Return the shipped Ni Lambert master as
    ``(upper, lower, npx, npy, scale)`` in 64-bit floats."""
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    upper, lower = master._get_master_pattern_arrays_from_energy()
    npx, npy = master.axes_manager.signal_shape
    return (
        np.asarray(upper, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        int(npx),
        int(npy),
        float((npx - 1) / 2),
    )


def make_detector(shape=SHAPE_480, pc=PC_480, binning=1):
    """Return a detector with one Bruker projection centre, or with
    one per map point when *pc* is already an array of them."""
    pc = np.asarray(pc, dtype=np.float64)
    if pc.ndim == 1:
        pc = pc[None, :]
    return kp.detectors.EBSDDetector(
        shape=shape,
        binning=binning,
        px_size=70.0,
        pc=pc,
        sample_tilt=70.0,
        tilt=0.0,
    )


def generic_rotation():
    """Return the generic crystal orientation of this module."""
    return Rotation.from_axes_angles(
        ORIENTATION_AXIS, np.deg2rad(ORIENTATION_ANGLE_DEG)
    )


def sample_to_crystal_matrix(rotation):
    """Return the matrix ``g`` with ``v_crystal = g @ v_sample``,
    derived from ``rotate_vector`` itself so that no orix convention
    is assumed anywhere in this module."""
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    return rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T


def sample_to_spec_detector_matrix(detector):
    """Return the matrix ``M`` with ``v_detector = M @ v_sample`` in
    the SPEC's y-down detector frame (requirements D7, corrected):
    the detector's own rotation composed with the measured y flip."""
    return DETECTOR_Y_FLIP @ detector.sample_to_detector.to_matrix().squeeze()


def crystal_to_spec_detector_matrix(detector, rotation):
    """Return the matrix with ``v_detector = R @ v_crystal``, the D7
    frame chain a deformation gradient is imposed through."""
    return (
        sample_to_spec_detector_matrix(detector) @ sample_to_crystal_matrix(rotation).T
    )


def _bilinear(master, nii, nij, niip, nijp, di, dj, dim, djm):
    """Vectorized twin of ``_get_pixel_from_master_pattern``."""
    return (
        master[nii, nij] * dim * djm
        + master[niip, nij] * di * djm
        + master[nii, nijp] * dim * dj
        + master[niip, nijp] * di * dj
    )


def project_pattern(detector, rotation, deformation=None, pc_index=None):
    """Return one pattern projected from the Ni Lambert master with
    an imposed deformation gradient.

    Reimplements ``EBSDMasterPattern.get_patterns``' geometry with
    EMsoft's own ``applyDeformation`` mechanism (validation V3):
    obtain the detector direction cosines, rotate them to the crystal
    frame, left-multiply ``F**-1``, renormalize, interpolate the
    master.
    """
    upper, lower, npx, npy, scale = ni_master_arrays()
    cosines = _get_direction_cosines_from_detector(detector)
    if cosines.ndim == 3:
        cosines = cosines[pc_index]
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    vectors = rotate_vector(quaternion, np.ascontiguousarray(cosines))
    if deformation is not None:
        vectors = vectors @ np.linalg.inv(deformation).T
        vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, None]
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    parameters = _get_lambert_interpolation_parameters(vectors, npx, npy, scale)
    north = _bilinear(upper, *parameters)
    south = _bilinear(lower, *parameters)
    pattern = np.where(vectors[:, 2] >= 0, north, south)
    return pattern.reshape(detector.shape)


def impose_detector_frame_fe(detector, rotation, fe_detector):
    """Return the crystal-frame deformation gradient which imposes
    *fe_detector* in the spec's detector frame."""
    chain = crystal_to_spec_detector_matrix(detector, rotation)
    return chain.T @ np.asarray(fe_detector, dtype=np.float64) @ chain


def cubic_entry(i, j):
    """Return one entry of the unrotated cubic nickel Voigt matrix."""
    c11, c12, c44 = (NICKEL_CUBIC[k] for k in ("c11", "c12", "c44"))
    if i < 3 and j < 3:
        return c11 if i == j else c12
    return c44 if i == j else 0.0


def crystal_stiffness_here():
    """Return the nickel Voigt matrix in the CRYSTAL frame, built
    here rather than with ``voigt_stiffness``."""
    return np.array(
        [[cubic_entry(i, j) for j in range(VOIGT_SIZE)] for i in range(VOIGT_SIZE)]
    )


def voigt_to_tensor_here(c):
    """Return the fourth-order form of a Voigt matrix, expanded
    here."""
    tensor = np.zeros((3, 3, 3, 3))
    for p, (i, j) in enumerate(VOIGT_INDICES):
        for q, (k, m) in enumerate(VOIGT_INDICES):
            value = c[p, q]
            for a, b in ((i, j), (j, i)):
                for e, f in ((k, m), (m, k)):
                    tensor[a, b, e, f] = value
    return tensor


def rotate_here(c, g):
    """Return the SAMPLE-frame Voigt matrix of a crystal-frame one,
    ``C_sample_ijkl = g_ai g_bj g_ck g_dl C_crystal_abcd`` with ``g``
    the orientation ``v_crystal = g @ v_sample`` (requirements
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


def skew_of(vector):
    """Return the antisymmetric tensor whose dual vector is
    ``(w32, w13, w21)``."""
    w1, w2, w3 = vector
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def traction_free_beta(stiffness_sample, seed=0):
    """Return a sample-frame ``beta`` whose symmetric part carries
    ``sigma33 = 0`` BY CONSTRUCTION, and the strain it was built
    from.

    ``e33`` is solved from the Hooke product written out here, so
    that the closure of requirements D9.2 has an EXACT answer to find
    rather than a band to land in.
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


def assert_within(measured, bound, name: str) -> None:
    """Assert ``measured <= bound``, failing loudly and informatively
    while *bound* is an unfilled measured-then-pinned placeholder."""
    if bound is None:
        raise AssertionError(
            f"{name} is an unfilled MEASURED-THEN-PINNED placeholder "
            f"(requirements D19); measured {measured!r}. Fill it with "
            "a dated value and record the recipe in validation.md"
        )
    assert measured <= bound, f"{name}: {measured} > {bound}"


@functools.lru_cache(maxsize=1)
def imposed_run():
    """Return the engine result of the imposed-strain map, cached
    because the three 480 px projections plus the fits cost a second.

    The map is one reference plus two deformed points, every point
    sharing ONE projection centre so that the beam-scan phantom of
    requirements D6.2 is exactly the identity and what is measured is
    the tensor chain alone.

    Returns
    -------
    detector, strains, xmap
        The single-projection-centre detector the chain is given, the
        two imposed SAMPLE-frame strain tensors, and the crystal map
        carrying the recovered ``Fe`` property and the generic
        orientation.
    """
    detector = make_detector()
    rotation = generic_rotation()
    matrix = sample_to_spec_detector_matrix(detector)
    stiffness_sample = rotate_here(
        crystal_stiffness_here(), sample_to_crystal_matrix(rotation)
    )
    patterns = [project_pattern(detector, rotation)]
    strains = []
    for seed in (0, 1):
        beta, strain = traction_free_beta(stiffness_sample, seed=seed)
        strains.append(strain)
        f_detector = matrix @ (np.eye(3) + beta) @ matrix.T
        reduced = f_detector / f_detector[2, 2]
        patterns.append(
            project_pattern(
                detector,
                rotation,
                impose_detector_frame_fe(detector, rotation, reduced),
            )
        )
    per_point = make_detector(pc=np.tile(PC_480, (1, len(patterns), 1)))
    properties = run_hrebsd_dic(
        np.stack(patterns),
        (1, len(patterns)),
        per_point,
        reference=(0, 0),
        verbose=0,
    )
    assert np.all(properties["converged"])
    arrays, size = create_coordinate_arrays((1, len(patterns)), (1.0, 1.0))
    arrays["rotations"] = Rotation(np.tile(rotation.data.ravel(), (size, 1)))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    xmap.prop["Fe"] = np.asarray(properties["Fe"], dtype=np.float64).reshape(
        size, BETA_PROP_SIZE
    )
    return detector, np.stack(strains), xmap


# ============ V3 -- strain recovery through the patterns ============ #


class TestDeformedMaster:
    """Patterns projected from the shipped Ni Lambert master with an
    imposed deformation, carried through the Stage A engine and then
    through the Stage B tensor chain.  [D7-D10/V3]"""

    def test_the_imposed_strain_is_traction_free_by_construction(self):
        # A LIBRARY MEASUREMENT of the oracle itself, so it passes
        # today: the tensor the patterns are built from really does
        # carry sigma33 = 0 in the sample frame, and really is of the
        # 1e-3 scale.  Without this the recovery tests below could
        # pass on a chain which returns zeros
        rotation = generic_rotation()
        stiffness_sample = rotate_here(
            crystal_stiffness_here(), sample_to_crystal_matrix(rotation)
        )
        for seed in (0, 1):
            _, strain = traction_free_beta(stiffness_sample, seed=seed)
            voigt = voigt_vector_here(strain)
            engineering = voigt.copy()
            engineering[3:] *= 2.0
            stress = stiffness_sample @ engineering
            assert abs(stress[2]) < 1e-12 * max(float(np.abs(stress).max()), 1.0)
            assert np.abs(strain).max() > 1e-4
        # and the generic orientation really does rotate the
        # stiffness: an unrotated cubic matrix has C3312, C3313 and
        # C3323 identically zero, which would leave the shear terms of
        # the closure and the Bond rotation direction unexercised
        assert np.abs(stiffness_sample[2, 3:]).max() > 1.0

    def test_the_local_stiffness_is_the_public_one(self):
        # the oracle above is built with this module's own matrix, so
        # that it does not rest on the function under test; this pins
        # the two together, and it is the only place the public
        # builder enters this file
        np.testing.assert_allclose(
            kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC),
            crystal_stiffness_here(),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_deformed_master_strain_recovery(self):
        # THE tensor half of validation V3.  Everything between the
        # imposed crystal-frame deformation and the reported strain
        # has to be right at once: the D6 conversion, the D7 frame
        # chain with its y flip and its transpose, the D9.5 crystal to
        # sample rotation of the stiffness at a GENERIC orientation,
        # and the D9.2 traction-free closure.  Transposing the Bond
        # rotation changes the sample-frame C3311, C3322 and the three
        # shear terms, so it changes the recovered e33 (MEASURED on
        # the analytic route: worst strain error 2.9630e-04 against
        # 1.1655e-06 correct)
        detector, strains, xmap = imposed_run()
        result = hrebsd_strain_stress(
            xmap, detector, stiffness=crystal_stiffness_here()
        )
        strain = np.asarray(result.prop["strain"])
        worst = 0.0
        for point, imposed in enumerate(strains, start=1):
            worst = max(
                worst, float(np.abs(strain[point] - voigt_vector_here(imposed)).max())
            )
        assert np.abs(strains).max() > 1e-4
        assert_within(worst, DEFORMED_MASTER_STRAIN_TOL, "DEFORMED_MASTER_STRAIN_TOL")

    def test_e33_is_the_built_in_closure_value(self):
        # THE closure killer of validation V3: ``e33`` is not measured
        # anywhere -- the projection cannot see the isotropic part --
        # so what comes back is what the traction-free assumption
        # supplies, and the tensor was built with exactly that value
        detector, strains, xmap = imposed_run()
        result = hrebsd_strain_stress(
            xmap, detector, stiffness=crystal_stiffness_here()
        )
        strain = np.asarray(result.prop["strain"])
        for point, imposed in enumerate(strains, start=1):
            assert_within(
                abs(strain[point, 2] - imposed[2, 2]),
                DEFORMED_MASTER_STRAIN_TOL,
                "DEFORMED_MASTER_STRAIN_TOL (e33)",
            )
            # and e33 is not incidentally zero, so the assertion says
            # something
            assert abs(imposed[2, 2]) > 1e-5

    def test_sigma33_is_zero_through_the_patterns(self):
        # the self-check of requirements D10, which is NOT stored and
        # is asserted here instead
        detector, _, xmap = imposed_run()
        result = hrebsd_strain_stress(
            xmap, detector, stiffness=crystal_stiffness_here()
        )
        stress = np.asarray(result.prop["stress"])[1:]
        # the stress really is of the GPa scale, so a near-zero
        # sigma33 is a statement and not an artefact of smallness
        assert np.abs(stress).max() > 1e-2
        assert_within(
            float(np.abs(stress[:, 2]).max()),
            DEFORMED_MASTER_SIGMA33_TOL,
            "DEFORMED_MASTER_SIGMA33_TOL",
        )

    def test_the_reference_point_carries_no_strain(self):
        # point zero IS the reference pattern, so its homography is
        # the identity and its strain is zero to the closure's own
        # arithmetic: the free contrast arm of the recovery above
        detector, _, xmap = imposed_run()
        result = hrebsd_strain_stress(
            xmap, detector, stiffness=crystal_stiffness_here()
        )
        strain = np.asarray(result.prop["strain"])
        assert float(np.abs(strain[0]).max()) < 1e-9
        assert float(np.abs(strain[1:]).max()) > 1e-4

    def test_the_two_closures_differ_through_the_patterns(self):
        # validation V3's ``test_traction_free_vs_deviatoric_differ``
        # on the pattern route: the imposed tensors are not traceless,
        # so the fallback-always mutant shows here too
        detector, _, xmap = imposed_run()
        traction_free = np.asarray(
            hrebsd_strain_stress(
                xmap, detector, stiffness=crystal_stiffness_here()
            ).prop["strain"]
        )
        deviatoric = np.asarray(hrebsd_strain_stress(xmap, detector).prop["strain"])
        assert float(np.abs(traction_free[1:, 2] - deviatoric[1:, 2]).max()) > 1e-5

    def test_the_rotation_comes_back_in_the_sample_frame(self):
        # the imposed skew part is a sample-frame lattice rotation, so
        # the reported rotation vector is its dual vector: this is the
        # pattern-level half of the D7 frame pin, where dropping the
        # y flip would flip two of its three components
        detector, _, xmap = imposed_run()
        rotation = generic_rotation()
        stiffness_sample = rotate_here(
            crystal_stiffness_here(), sample_to_crystal_matrix(rotation)
        )
        result = hrebsd_strain_stress(
            xmap, detector, stiffness=crystal_stiffness_here()
        )
        got = np.asarray(result.prop["rotation_vector"])
        for point, seed in enumerate((0, 1), start=1):
            beta, strain = traction_free_beta(stiffness_sample, seed=seed)
            skew = 0.5 * (beta - beta.T)
            expected = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            assert float(np.abs(expected).max()) > 1e-4
            assert_within(
                float(np.abs(got[point] - expected).max()),
                DEFORMED_MASTER_STRAIN_TOL,
                "DEFORMED_MASTER_STRAIN_TOL (rotation vector)",
            )
