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

"""Tests of ``kikuchipy.indexing._hrebsd._geometry``.

Covers the unit system of requirements D1 and the analytic half of
oracle V6 of ``specs/2026-09-07-hrebsd-dic/validation.md``: the
projection centre to binned pixel conversion, the per-point
projection centre routes of D6.1, the closed-form beam-scan phantom
of D6.2 and its removal before the conversion to a deformation
gradient.  The PATTERN half of V6, which pins the D6.3 signs against
synthesized per-point-PC patterns, lives in ``test_hrebsd_engine.py``
where the projection helper is.

Every band here is the frozen 1e-12 of the algebraic identities; the
signs and the composition side of the phantom are DRAFTED from D6.2
and are pinned by the pattern oracle, so a refutation amends
requirements D6.3 and this module together, with a date.

Written before the implementation exists: every test here calls the
skeleton and therefore fails with ``NotImplementedError`` until the
geometry lands, then passes unchanged.
"""

import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._geometry import (
    correct_homography,
    fe_from_homography,
    pc_to_pixels,
    per_point_pc_pixels,
    phantom_homography,
)
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    compose,
    fe_to_homography,
    homography_to_fe,
    invert,
    shape_function,
)

# ------------------------- Frozen constants ------------------------- #

# The algebraic identity band, FROZEN (no measurement, see the module
# docstring)
ALGEBRA_TOL = 1e-12

# A deliberately NON-SQUARE detector: with nrows == ncols the row and
# column scalings of D1.2 are indistinguishable, and the
# ``pcy``/``pcz`` swap mutant of plan 2.5 survives
SHAPE = (40, 60)

# The Bruker projection centre of the reference pattern
PC_REFERENCE = (0.4210, 0.7794, 0.5049)


# ----------------------------- Helpers ------------------------------ #


def make_detector(pc=PC_REFERENCE, shape=SHAPE, binning=8, **kwargs):
    """Return an EBSD detector with the given Bruker projection
    centre or centres, in the fork's own convention."""
    return kp.detectors.EBSDDetector(
        shape=shape,
        binning=binning,
        px_size=70.0,
        pc=np.atleast_2d(np.asarray(pc, dtype=np.float64)),
        sample_tilt=70.0,
        tilt=0.0,
        **kwargs,
    )


def reference_and_target_pc(delta=(3.0, -2.0, 1.5)):
    """Return two projection centres in binned pixels which differ by
    a plain beam-scan step: a shift in both in-plane directions and a
    change of detector distance."""
    pc_reference = np.array([120.0, 180.0, 240.0])
    return pc_reference, pc_reference + np.asarray(delta, dtype=np.float64)


# ================== D1 -- projection centre units =================== #


class TestPcToPixels:
    """``PCx_px = pcx * ncols``, ``PCy_px = pcy * nrows`` and
    ``DD_px = pcz * nrows`` from the stored Bruker fractions, with no
    sign flip anywhere.  [D1.2]"""

    def test_conversion_is_literal(self):
        pc = np.array([0.25, 0.30, 0.50])
        nrows, ncols = SHAPE
        expected = np.array([[0.25 * ncols, 0.30 * nrows, 0.50 * nrows]])
        got = pc_to_pixels(pc, SHAPE)
        assert got.shape == (1, 3)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_no_emsoft_y_flip(self):
        # EMsoftOO's ``patcenty = 0.5 - ypc``
        # (``mod_HREBSDDIC.f90:917``) is an EMsoft convention artifact
        # and must never appear: Bruker's y runs from the top, as the
        # numpy array frame does
        pc = np.array([0.5, 0.30, 0.50])
        nrows = SHAPE[0]
        got = pc_to_pixels(pc, SHAPE)[0]
        assert got[1] == pytest.approx(0.30 * nrows, abs=ALGEBRA_TOL)
        assert got[1] != pytest.approx((0.5 - 0.30) * nrows, abs=1e-6)

    def test_many_projection_centres(self):
        pc = np.array([[0.4, 0.5, 0.6], [0.41, 0.51, 0.61]])
        got = pc_to_pixels(pc, SHAPE)
        assert got.shape == (2, 3)
        np.testing.assert_allclose(
            got[:, 2], pc[:, 2] * SHAPE[0], rtol=0, atol=ALGEBRA_TOL
        )

    def test_only_the_binned_shape_enters(self):
        # the ``DD in unbinned pixels`` mutant of plan 2.5: the
        # conversion knows nothing of ``binning`` or ``px_size``, the
        # shape it is handed is already the binned one
        pc = np.array([0.4, 0.5, 0.6])
        first = pc_to_pixels(pc, SHAPE)
        second = pc_to_pixels(pc, SHAPE)
        assert np.array_equal(first, second)
        halved = pc_to_pixels(pc, (SHAPE[0] // 2, SHAPE[1] // 2))
        assert halved[0, 2] == pytest.approx(first[0, 2] / 2)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            pc_to_pixels(np.array([0.4, 0.5]), SHAPE)


class TestPerPointProjectionCentres:
    """Per-point projection centres are first class: consumed from
    the detector when it has them, derived from the beam-scan model
    when it does not.  [D6.1]"""

    def test_per_point_detector_is_consumed_directly(self):
        navigation_shape = (3, 4)
        rng = np.random.default_rng(0)
        pc = np.array(PC_REFERENCE) + rng.uniform(
            -1e-3, 1e-3, size=navigation_shape + (3,)
        )
        detector = kp.detectors.EBSDDetector(
            shape=SHAPE,
            binning=8,
            px_size=70.0,
            pc=pc,
            sample_tilt=70.0,
            tilt=0.0,
        )
        got = per_point_pc_pixels(detector, navigation_shape, (1.5, 1.5))
        expected = pc_to_pixels(detector.pc_flattened, SHAPE)
        assert got.shape == (12, 3)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_single_pc_uses_extrapolate_pc(self):
        # the internal beam-scan model IS
        # ``EBSDDetector.extrapolate_pc``, never a hand built one:
        # nothing about the geometry is hardcoded (the EMsoftOO 70
        # degree literal of ``mod_HREBSDDIC.f90:925`` is a recorded
        # deviation)
        navigation_shape = (3, 4)
        step_sizes = (1.5, 2.5)
        detector = make_detector()
        got = per_point_pc_pixels(
            detector, navigation_shape, step_sizes, reference_index=0
        )
        expected_detector = detector.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=navigation_shape,
            step_sizes=step_sizes,
        )
        expected = pc_to_pixels(expected_detector.pc_flattened, SHAPE)
        assert got.shape == (12, 3)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_single_pc_is_anchored_at_the_reference(self):
        # the extrapolation is anchored at the REFERENCE pattern's
        # scan position, so that point keeps the detector's own
        # projection centre exactly
        navigation_shape = (3, 4)
        detector = make_detector()
        reference_index = 6  # row 1, column 2
        got = per_point_pc_pixels(
            detector,
            navigation_shape,
            (1.5, 1.5),
            reference_index=reference_index,
        )
        expected = pc_to_pixels(np.asarray(PC_REFERENCE), SHAPE)[0]
        # the frozen algebraic band, not a guessed one: the beam-scan
        # model's offsets are ``(pc_indices_mean - index) * step``
        # (``_ebsd_detector.py:1411-1413``), which is EXACTLY zero at
        # the anchor, so the anchor keeps the detector's own
        # projection centre to machine precision
        np.testing.assert_allclose(
            got[reference_index], expected, rtol=0, atol=ALGEBRA_TOL
        )

    def test_wrong_navigation_size_raises(self):
        detector = kp.detectors.EBSDDetector(
            shape=SHAPE,
            binning=8,
            px_size=70.0,
            pc=np.tile(PC_REFERENCE, (5, 1)),
            sample_tilt=70.0,
        )
        with pytest.raises(ValueError):
            per_point_pc_pixels(detector, (3, 4), (1.5, 1.5))


# ============= V6 (analytic) -- the beam-scan phantom =============== #


class TestPhantomHomography:
    """The closed form of requirements D6.2 in the spec's own
    reference-PC-centred frame: a pure projection centre change maps
    ``xi' = alpha_s * xi + gamma`` with ``gamma`` the projection
    centre difference in binned pixels EXACTLY.

    The orientation of the detector distance ratio and the
    composition side below are the DRAFTED ones; the pattern oracle
    of validation V6 pins them and requirements D6.3 records the
    verdict with its date.  [D6.2/D6.3]"""

    def test_closed_form_is_literal(self):
        pc_reference, pc_target = reference_and_target_pc()
        alpha_s = pc_target[2] / pc_reference[2]  # DRAFTED orientation
        gamma = pc_target[:2] - pc_reference[:2]
        expected = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        expected[0] = alpha_s - 1.0
        expected[4] = alpha_s - 1.0
        expected[2] = gamma[0]
        expected[5] = gamma[1]
        got = phantom_homography(pc_reference, pc_target)
        assert got.shape == (N_HOMOGRAPHY_PARAMETERS,)
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_no_emsoftoo_patcent_term(self):
        # EMsoftOO's ``gamma_i = delta_i + (delta_i - patcent_i) *
        # (alpha_s - 1)`` (``mod_HREBSDDIC.f90:955-970``) is written
        # for its absolute-PC coordinates; the extra term vanishes
        # only when the projection centre is the coordinate origin,
        # which is exactly this frame.  With a 1.5 px detector
        # distance change over a 240 px distance the transplanted
        # formula differs from the frozen one by about 0.75 px, far
        # above any band here
        pc_reference, pc_target = reference_and_target_pc()
        got = phantom_homography(pc_reference, pc_target)
        gamma = pc_target[:2] - pc_reference[:2]
        np.testing.assert_allclose(got[[2, 5]], gamma, rtol=0, atol=ALGEBRA_TOL)

    def test_is_the_identity_for_an_unchanged_pc(self):
        pc_reference, _ = reference_and_target_pc()
        got = phantom_homography(pc_reference, pc_reference)
        np.testing.assert_allclose(
            got, np.zeros(N_HOMOGRAPHY_PARAMETERS), rtol=0, atol=ALGEBRA_TOL
        )

    def test_is_an_affinity(self):
        # a pure projection centre change induces NO perspective
        # terms and no shear at all
        pc_reference, pc_target = reference_and_target_pc()
        got = phantom_homography(pc_reference, pc_target)
        assert got[1] == 0.0
        assert got[3] == 0.0
        assert got[6] == 0.0
        assert got[7] == 0.0
        matrix = shape_function(got)
        assert matrix[0, 0] == pytest.approx(matrix[1, 1], abs=ALGEBRA_TOL)


class TestCorrectHomography:
    """The correction removes the phantom by composition, BEFORE the
    conversion.  EMsoftOO computes the corrected homographies and
    then converts the uncorrected ones
    (``mod_HREBSDDIC.f90:978-982``), a recorded deviation.
    [D6.2]"""

    def test_removes_a_pure_phantom(self):
        pc_reference, pc_target = reference_and_target_pc()
        phantom = phantom_homography(pc_reference, pc_target)
        got = correct_homography(phantom, pc_reference, pc_target)
        np.testing.assert_allclose(
            got, np.zeros(N_HOMOGRAPHY_PARAMETERS), rtol=0, atol=ALGEBRA_TOL
        )

    def test_is_a_no_op_for_an_unchanged_pc(self):
        pc_reference, _ = reference_and_target_pc()
        rng = np.random.default_rng(3)
        h = rng.uniform(-1e-2, 1e-2, size=N_HOMOGRAPHY_PARAMETERS)
        got = correct_homography(h, pc_reference, pc_reference)
        np.testing.assert_allclose(got, h, rtol=0, atol=ALGEBRA_TOL)

    def test_composition_side_is_the_drafted_one(self):
        # DRAFTED ``W_corr = W_phantom**-1 . W`` (D6.2); the flipped
        # side is a genuinely different homography and dies at the
        # V6 pattern oracle if the draft is wrong
        pc_reference, pc_target = reference_and_target_pc()
        rng = np.random.default_rng(4)
        h = rng.uniform(-1e-2, 1e-2, size=N_HOMOGRAPHY_PARAMETERS)
        phantom = phantom_homography(pc_reference, pc_target)
        expected = compose(invert(phantom), h)
        got = correct_homography(h, pc_reference, pc_target)
        np.testing.assert_allclose(got, expected, rtol=ALGEBRA_TOL, atol=0)
        flipped = compose(h, invert(phantom))
        assert not np.allclose(got, flipped)


class TestFeFromHomography:
    """Correct, then convert with the target's own projection centre
    expressed in the reference-centred frame.  [D6.2/D6.3]"""

    def test_pure_phantom_gives_the_identity(self):
        # the analytic twin of validation V6's
        # ``test_phantom_corrected``: a pattern that differs from its
        # reference ONLY by the beam-scan geometry carries no strain
        pc_reference, pc_target = reference_and_target_pc()
        phantom = phantom_homography(pc_reference, pc_target)
        got = fe_from_homography(phantom, pc_reference, pc_target)
        np.testing.assert_allclose(got, np.eye(3), rtol=0, atol=ALGEBRA_TOL)

    def test_uncorrected_phantom_is_not_the_identity(self):
        # with the private switch off the phantom shows up as a
        # sizeable spurious deformation, which is what makes the
        # correction worth having and kills a no-op correction
        pc_reference, pc_target = reference_and_target_pc()
        phantom = phantom_homography(pc_reference, pc_target)
        got = fe_from_homography(phantom, pc_reference, pc_target, correct=False)
        assert np.abs(got - np.eye(3)).max() > 1e-3

    def test_conversion_uses_the_relative_target_pc(self):
        # the D6.2 rule spelled out: the conversion sees
        # ``PC_rel = PC_target - PC_reference`` and the TARGET's own
        # detector distance
        pc_reference, pc_target = reference_and_target_pc()
        rng = np.random.default_rng(5)
        h = rng.uniform(-1e-3, 1e-3, size=N_HOMOGRAPHY_PARAMETERS)
        expected = homography_to_fe(h, pc_target[:2] - pc_reference[:2], pc_target[2])
        got = fe_from_homography(h, pc_reference, pc_target, correct=False)
        np.testing.assert_allclose(got, expected, rtol=ALGEBRA_TOL, atol=0)

    def test_correction_precedes_conversion(self):
        # the ``correction applied after Fe conversion`` mutant of
        # plan 2.5: a homography built as the phantom composed with a
        # true deformation must convert back to exactly that
        # deformation.  Note what this does and does not pin: the
        # measured homography here is built with the SAME composition
        # the corrector inverts, so this is an order-of-operations
        # check, not a physical one.  The physical pin is the pattern
        # oracle of validation V6
        # (``test_hrebsd_engine.py::TestPcShiftPhantom``), whose map
        # was re-pinned 2026-09-07 to move the projection centre by
        # 11.4 px, 5.4 px and 8e-3 in the detector distance ratio,
        # far above the fit floor, precisely so that it can carry
        # that burden
        pc_reference, pc_target = reference_and_target_pc()
        pc_rel = pc_target[:2] - pc_reference[:2]
        fe_true = np.array(
            [[1.001, 0.0004, 0.0], [-0.0004, 0.999, 0.0], [0.0, 0.0, 1.0]]
        )
        h_true = fe_to_homography(fe_true, pc_rel, pc_target[2])
        phantom = phantom_homography(pc_reference, pc_target)
        h_measured = compose(phantom, h_true)
        got = fe_from_homography(h_measured, pc_reference, pc_target)
        np.testing.assert_allclose(got, fe_true, rtol=0, atol=ALGEBRA_TOL)
