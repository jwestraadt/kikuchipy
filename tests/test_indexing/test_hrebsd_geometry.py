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
    STEP_SIZE_UNIT_FACTORS,
    correct_homography,
    fe_from_homography,
    pc_to_pixels,
    per_point_pc_pixels,
    phantom_homography,
    step_sizes_in_micrometres,
)
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    compose,
    homography_parameters,
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


def raw_fit_of(fe, pc_reference, pc_target):
    """Return the homography a target of deformation *fe* really
    fits, derived HERE from the ray model rather than taken from
    ``_geometry``.

    In the D1.3 reference-PC-centred frame a pixel ``xi`` of the
    REFERENCE carries the ray ``(xi_x, xi_y, DD_ref)`` while the same
    pixel of the TARGET carries ``(xi_x - dx, xi_y - dy, DD_tar)``
    with ``d = PC_target - PC_reference``.  Deforming the first by
    ``fe`` and re-intersecting the target screen therefore gives::

        W = T(d) . diag(1, 1, 1/DD_tar) . fe . diag(1, 1, DD_ref)

    whose ``fe = I`` case is exactly the closed-form phantom of
    requirements D6.2, which is what ties this construction to the
    signs validation V6 pinned against synthesized PATTERNS.
    """
    fe = np.asarray(fe, dtype=np.float64)
    delta = (
        np.asarray(pc_target, dtype=np.float64)[:2]
        - np.asarray(pc_reference, dtype=np.float64)[:2]
    )
    translation = np.array(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]]
    )
    return homography_parameters(
        translation
        @ np.diag([1.0, 1.0, 1.0 / pc_target[2]])
        @ fe
        @ np.diag([1.0, 1.0, pc_reference[2]])
    )


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

    # A GENERIC reduced deformation gradient: every off-diagonal
    # nonzero, and in particular the perspective row ``Fe31``/``Fe32``
    # and the out-of-plane column ``Fe13``/``Fe23``.  REWRITTEN
    # 2026-09-07 at the Stage A adversarial review, which found the
    # drafted tensor blind twice over: with ``Fe31 = Fe32 = 0`` every
    # homography below has ``beta0 = 1`` and the conversion collapses
    # to a conjugation, which is a group homomorphism, so correcting
    # BEFORE and correcting AFTER the conversion agree to 1.4e-19 and
    # the plan-2.5 ``correction applied after Fe conversion`` mutant
    # lived; and with ``Fe13 = Fe23 = 0`` too the homography carries
    # no detector distance at all, so neither the frame nor the
    # distance of the conversion was pinned either
    GENERIC_FE = np.array(
        [
            [1.0012, 6.0e-4, 9.0e-4],
            [-4.0e-4, 0.9988, -7.0e-4],
            [1.0e-3, -8.0e-4, 1.0],
        ]
    )

    def test_correction_precedes_conversion(self):
        # the ``correction applied after Fe conversion`` mutant of
        # plan 2.5, on a homography built the way the ENGINE really
        # measures one: the raw fit of a target which differs from
        # its reference by the beam-scan geometry AND by a real
        # deformation.  ``raw_fit_of`` derives it from the ray model
        # in this module, so the expectation shares no arithmetic
        # with ``_geometry``
        pc_reference, pc_target = reference_and_target_pc()
        fe_true = self.GENERIC_FE
        h_measured = raw_fit_of(fe_true, pc_reference, pc_target)
        got = fe_from_homography(h_measured, pc_reference, pc_target)
        np.testing.assert_allclose(got, fe_true, rtol=0, atol=ALGEBRA_TOL)

    def test_the_phantom_is_the_undeformed_raw_fit(self):
        # the tie between ``raw_fit_of`` above and the closed form
        # validation V6 pinned against synthesized patterns: with no
        # deformation the ray model IS the D6.2 phantom, which is
        # what makes the ray model usable as an oracle here
        pc_reference, pc_target = reference_and_target_pc()
        np.testing.assert_allclose(
            raw_fit_of(np.eye(3), pc_reference, pc_target),
            phantom_homography(pc_reference, pc_target),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_corrected_homography_lives_in_the_reference_frame(self):
        # THE frame pin of requirements D6.2, corrected 2026-09-07 at
        # the Stage A adversarial review.  Removing the phantom
        # cancels both the projection centre offset and the detector
        # distance ratio, leaving the pure conjugation
        # ``diag(1, 1, DD_ref)**-1 . Fe . diag(1, 1, DD_ref)``, so the
        # exact conversion of a CORRECTED homography sees
        # ``pc_rel = (0, 0)`` and ``dd = DD_reference``
        pc_reference, pc_target = reference_and_target_pc()
        fe_true = self.GENERIC_FE
        h_measured = raw_fit_of(fe_true, pc_reference, pc_target)
        corrected = correct_homography(h_measured, pc_reference, pc_target)
        distance = np.diag([1.0, 1.0, pc_reference[2]])
        np.testing.assert_allclose(
            shape_function(corrected),
            np.linalg.inv(distance) @ fe_true @ distance,
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        np.testing.assert_allclose(
            homography_to_fe(corrected, np.zeros(2), pc_reference[2]),
            fe_true,
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_target_pc_route_is_refuted_by_measurement(self):
        # the arm which makes the pin above a discriminator: the
        # route requirements D6.2 called "first-order equivalent"
        # until 2026-09-07 -- convert the CORRECTED homography with
        # ``pc_rel = PC_t - PC_ref`` and ``dd = DD_target`` -- injects
        # a spurious isotropic strain into Fe11/Fe22 which is orders
        # above this module's own algebraic band, and which no oracle
        # with ``Fe = I`` or a shared projection centre can see
        pc_reference, pc_target = reference_and_target_pc()
        fe_true = self.GENERIC_FE
        h_measured = raw_fit_of(fe_true, pc_reference, pc_target)
        corrected = correct_homography(h_measured, pc_reference, pc_target)
        old_route = homography_to_fe(
            corrected, pc_target[:2] - pc_reference[:2], pc_target[2]
        )
        assert np.abs(old_route - fe_true).max() > 1e3 * ALGEBRA_TOL
        # and it is invisible at ``Fe = I``, which is why validation
        # V6's corrected-phantom arm passed with it in place
        phantom = phantom_homography(pc_reference, pc_target)
        np.testing.assert_allclose(
            fe_from_homography(phantom, pc_reference, pc_target),
            np.eye(3),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_correcting_after_the_conversion_is_now_identical(self):
        # RECORDED, not a bug: once the conversion happens in the
        # reference frame it is the conjugation by
        # ``diag(1, 1, DD_ref)``, a group HOMOMORPHISM, so removing
        # the phantom in homography space and removing it in Fe space
        # give the same tensor exactly.  The plan-2.5 "correction
        # applied after Fe conversion" mutant is therefore an
        # EQUIVALENT mutant of the corrected design rather than a
        # live defect, and this test states that with numbers so that
        # the equivalence is re-checked whenever the conversion moves
        pc_reference, pc_target = reference_and_target_pc()
        h_measured = raw_fit_of(self.GENERIC_FE, pc_reference, pc_target)
        phantom = phantom_homography(pc_reference, pc_target)
        origin = np.zeros(2)
        after = np.linalg.inv(
            homography_to_fe(phantom, origin, pc_reference[2])
        ) @ homography_to_fe(h_measured, origin, pc_reference[2])
        after = after / after[2, 2]
        before = fe_from_homography(h_measured, pc_reference, pc_target)
        np.testing.assert_allclose(after, before, rtol=0, atol=ALGEBRA_TOL)

    def test_uncorrected_route_keeps_the_target_pc(self):
        # the diagnostic path is unchanged: validation V6's
        # uncorrected arm reads the target's own projection centre
        pc_reference, pc_target = reference_and_target_pc()
        h = raw_fit_of(self.GENERIC_FE, pc_reference, pc_target)
        np.testing.assert_allclose(
            fe_from_homography(h, pc_reference, pc_target, correct=False),
            homography_to_fe(h, pc_target[:2] - pc_reference[:2], pc_target[2]),
            rtol=ALGEBRA_TOL,
            atol=0,
        )


# ============ D14.5 precedent -- scan step size units =============== #


class TestStepSizeUnits:
    """The navigation axes' units are READ and converted into the
    micrometres of ``EBSDDetector.px_size``, never guessed: the
    beam-scan model of D6.1 divides every step by
    ``px_size * binning``, so a nanometre scan described as
    micrometres moves every derived projection centre by a factor of
    a thousand.  Requirements D14.5 sets the precedent.  [D6.1]"""

    @pytest.mark.parametrize(
        "unit, factor",
        [("um", 1.0), ("nm", 1e-3), ("mm", 1e3), ("m", 1e6), ("µm", 1.0)],
    )
    def test_known_units_convert(self, unit, factor):
        got = step_sizes_in_micrometres((2.0, 4.0), (unit, unit))
        assert got == pytest.approx((2.0 * factor, 4.0 * factor), rel=1e-12)

    def test_case_and_whitespace_are_tolerated(self):
        assert step_sizes_in_micrometres((1.0,), (" NM ",)) == pytest.approx((1e-3,))

    @pytest.mark.parametrize("unit", ["px", "", "<undefined>", "arb. units"])
    def test_unknown_unit_raises_naming_the_axis(self, unit):
        with pytest.raises(ValueError, match="navigation axis 1"):
            step_sizes_in_micrometres((1.0, 1.0), ("um", unit))

    def test_the_factor_table_is_the_d14_5_set(self):
        # the same units D14.5 pins for ``CrystalMap.scan_unit``, in
        # the micrometre base ``px_size`` uses
        assert STEP_SIZE_UNIT_FACTORS["m"] == 1e6
        assert STEP_SIZE_UNIT_FACTORS["nm"] == 1e-3
        assert STEP_SIZE_UNIT_FACTORS["um"] == 1.0
        assert "px" not in STEP_SIZE_UNIT_FACTORS

    def test_a_nanometre_scan_moves_the_projection_centres(self):
        # the reason the conversion exists, in binned pixels: the
        # same scan described in nanometres and in micrometres must
        # not give the same projection centre map
        detector = make_detector()
        as_um = per_point_pc_pixels(detector, (3, 3), (0.4, 0.4))
        as_nm = per_point_pc_pixels(
            detector, (3, 3), step_sizes_in_micrometres((400.0, 400.0), ("nm", "nm"))
        )
        np.testing.assert_allclose(as_um, as_nm, rtol=0, atol=ALGEBRA_TOL)
        unconverted = per_point_pc_pixels(detector, (3, 3), (400.0, 400.0))
        assert np.abs(unconverted - as_um).max() > 1.0
