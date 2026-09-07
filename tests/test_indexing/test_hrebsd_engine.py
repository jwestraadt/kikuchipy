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

"""Tests of ``kikuchipy.indexing._hrebsd._engine``,
``._preprocessing`` and ``._reference``.

Covers, from ``specs/2026-09-07-hrebsd-dic/validation.md``:

- **V2**, the synthetic warp-refit oracle: a reference warped by a
  known homography with the INDEPENDENT warper
  ``skimage.transform.ProjectiveTransform`` plus ``warp`` (a
  test-only import, requirements D18) and refitted, measured with
  the corner-displacement error-warp metric in binned pixels; the
  intensity-scale-invariance test that kills a mismatched
  Hessian/gradient normalization; the convergence metadata and NaN
  contract; the seed-required case; the dtype A/B harness; the
  determinism pin; the border and dead-band test.
- **V1**'s warp DIRECTION pin, which needs a pattern and therefore
  lives here rather than in ``test_hrebsd_homography.py``.
- **V3**, the engine half of the deformed-master oracle: patterns
  projected from the shipped Ni Lambert master with an imposed
  deformation gradient, whose expected homography is EXACT.
- **V4**, the pure-rotation analytic cases and the detector-frame
  axis and sign pins of D1.5 and D7.
- **V6**, the pattern half of the PC-shift phantom oracle, which
  pins the D6.3 signs.
- **D4**'s preprocessing, **D5**'s initial guess, **D11**'s Stage A
  reference modes, **D15.7**'s ``get_map_data`` verification task,
  **D16**'s information message and **D18**'s import audit.

V3, V4 and V6's pattern half were drafted for their own files; the
consolidation into this one module is a recorded file-layout
deviation (validation.md, the V3 heading and the 2026-09-07 entry of
"Recorded results"): all three need the same projection helper, the
same cached 480 px reference and the same engine fixtures.

**The half-pixel finding, MEASURED here and now folded into the
spec (2026-09-07).** Requirements D1.1 put the coordinate origin at
"the upper-left pixel center" with ``x`` the column INDEX, and D1.2
takes ``PCx_px = pcx * ncols`` from the Bruker fraction, which is
measured from the detector EDGE. Those two together made
``xi_x = col - PCx_px``, half a pixel away from kikuchipy's own
projection geometry, which this module MEASURED to be exactly
``xi_x = col + 0.5 - PCx_px`` and
``xi_y = row + 0.5 - PCy_px`` after mapping
``EBSDDetector.sample_to_detector`` into the spec's y-down frame (see
``test_pc_centred_frame_matches_kikuchipy_geometry``, which needs no
``_hrebsd`` code and passes today). The oracles below are built in
kikuchipy's frame, because a synthetic pattern has to be projected
with SOME geometry and only the kikuchipy one keeps the engine
self-consistent with ``fit_pc``, ``extrapolate_pc`` and every stored
projection centre. Requirements D1.1 and D1.3 now carry the dated
correction and validation.md records the measurement, so the frozen
spec and this contract agree.

Written before the implementation exists: except where a test is
marked as a pure library verification task, every test here calls the
skeleton and therefore fails with ``NotImplementedError`` until the
engine lands, then passes unchanged.
"""

import functools
import inspect
import pathlib

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest
from scipy.ndimage import shift as ndi_shift

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing import _hrebsd
from kikuchipy.indexing._hrebsd._engine import (
    FE_PROP_SIZE,
    HOMOGRAPHY_PROP_SIZE,
    STAGE_A_PROP_NAMES,
    ReferenceState,
    fit_pattern,
    get_info_message,
    initial_guess,
    run_hrebsd_dic,
)
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    fe_to_homography,
)
from kikuchipy.indexing._hrebsd._interpolation import (
    evaluate,
    gradient_planes,
    spline_coefficients,
)
from kikuchipy.indexing._hrebsd._preprocessing import (
    band_pass_transfer_function,
    hann_window,
    preprocess,
    subregion_bounds,
    subregion_mask,
    zero_mean_normalize,
)
from kikuchipy.indexing._hrebsd._reference import AUTO_REFERENCE, resolve_reference
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# The oracle detector size.  Every precision claim of this feature is
# tied to 480 px or more; the 60 px arm below is qualitative only
SHAPE_480 = (480, 480)
SHAPE_60 = (60, 60)

# A projection centre which is NOT at the pattern centre, so that a
# PC-centred coordinate frame is distinguishable from a
# pattern-centred one (EMsoftOO warps about (0.5, 0.5), a recorded
# deviation)
PC_480 = (0.4210, 0.5794, 0.5049)

# A generic crystal orientation, deliberately away from any symmetry
# position: a symmetric one leaves whole rotation conventions
# unexercised
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0

# The V6 phantom map, RE-PINNED 2026-09-07 at the adversarial review.
# ``EBSDDetector.extrapolate_pc`` builds both the ``pcy`` and the
# ``pcz`` offsets from the ROW index alone
# (``_ebsd_detector.py:1412-1413``), so on the drafted one-row map
# with 1.5 unit steps ``alpha_s`` is exactly 1, ``gamma_y`` exactly 0
# and ``gamma_x`` 0.02 to 0.04 px, at or below the cross-interpolator
# systematic: the oracle was structurally incapable of pinning the
# D6.3 signs it is the designated killer for.  MEASURED offsets of
# the map below, ``(gamma_x, gamma_y, alpha_s - 1)`` at the far
# corner: (-11.43 px, -5.37 px, 8.06e-3), every one far above the DIC
# floor and inside the 24 px border budget of D4.4
PHANTOM_NAVIGATION_SHAPE = (2, 3)
PHANTOM_STEP_SIZES = (400.0, 400.0)

# The spec's detector frame is x right, y DOWN (the numpy array
# frame) while kikuchipy's gnomonic y is up, so the two are related
# by this flip.  MEASURED exactly (1.8e-14 px over a whole 40 by 60
# detector) by ``test_pc_centred_frame_matches_kikuchipy_geometry``
DETECTOR_Y_FLIP = np.diag([1.0, -1.0, 1.0])

# The algebraic identity band, FROZEN, as in the sibling modules
ALGEBRA_TOL = 1e-12

# The band of the per-reference precompute against the same algebra
# assembled here in plain numpy.  FROZEN, not measured: it is a
# machine-precision-class identity, loosened from ALGEBRA_TOL only
# because the Hessian is a sum over about 2e5 subregion pixels, whose
# summation order is an implementation detail (accumulated relative
# error of order sqrt(n) * eps, about 1e-13)
PRECOMPUTE_TOL = 1e-10

# The D17 acceptance criterion, quoted literally from requirements:
# 32-bit coefficient storage is pinned only if the homography
# recovery degrades by less than ten per cent of the 64-bit error
DTYPE_F32_MAX_DEGRADATION = 0.10

# MEASURED 2026-09-07 (failing-tests gate) [D15.7] on the venv orix
# 0.14.2: ``CrystalMap.get_map_data`` on an ``(n, k)`` property
# raises ``TypeError`` ("NumPy boolean array indexing assignment
# requires a 0 or 1-dimensional input"), so the DOCUMENTED retrieval
# route for the tensor properties is the reshape one and
# ``get_map_data`` is mentioned only for scalar properties.  Recorded
# in validation.md "Recorded results"; re-measured against the orix
# 0.12.1 floor by the local oldest-matrix run of plan section 1
GET_MAP_DATA_2D_PROP_OUTCOME = "TypeError"

# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders and drafting seeds (requirements D19).  Each
# is replaced by a dated measured value at the Stage A implementation
# gate, with the recipe recorded in validation.md "Recorded results".

# MTP [D2/D3/D5, V2]: the corner-displacement error-warp norm in
# binned pixels of the 480 px warp-refit batch.  DRAFTING NOTE, kept
# as a note and NOT as a live number (emptied 2026-09-07 at the
# adversarial review): the cross-interpolator systematic between the
# numba bicubic kernel and skimage's independent warper is expected at
# 0.01 to 0.05 px, so the drafted seed was 0.1 px with the project's
# ~2x margin convention.  A live seed 2 to 10 times above the expected
# achievable error is an acceptance gate that a mediocre
# implementation passes silently, and requirements D19 asks for a
# MEASURED value here, so the placeholder now fails loudly instead.
# MEASURING RECIPE: ``test_warp_refit_small_h_480`` below; record the
# worst error over the whole random batch and both seeds
WARP_REFIT_TOL_480 = None

# MTP [D2, V2]: the same metric on the 60 px shipped Ni patterns, a
# qualitative regime with no precision claim attached.  The batch is
# SCALED to the pattern (``scale=SHAPE_60[0]/SHAPE_480[0]``): the
# 480 px design budget of 5 px translations sits far outside a 60 px
# pattern's 3 px border, so the drafted unscaled call would have
# measured the mirror boundary rather than the engine.
# MEASURING RECIPE: ``test_warp_refit_60px`` below
WARP_REFIT_TOL_60 = None

# MTP [D2, V2]: the agreement of two fits which differ only by a
# GENERIC (non power of two) intensity factor.  Powers of two are
# asserted BITWISE and need no constant.
# MEASURING RECIPE: ``test_intensity_scale_invariance`` below
INTENSITY_SCALE_GENERIC_TOL = None

# MTP [D2-D7, V3]: the error-warp norm of the deformed-master
# recovery, where the expected homography is EXACT by construction.
# MEASURING RECIPE: ``test_deformed_master_homography_recovery``
DEFORMED_MASTER_H_TOL = None

# MTP [D6/D15.6, V3]: the worst absolute entry of
# ``Fe_recovered - Fe_imposed`` for the reduced detector-frame tensor
# THROUGH THE ENGINE, which is the only place the homography to Fe
# wiring (the detector distance in binned pixels, the pcy and pcz
# roles, row-major flattening, reference versus target) is exercised
# end to end; the algebra itself is pinned literally in
# ``test_hrebsd_homography.py``.
# MEASURING RECIPE: ``test_deformed_master_fe_through_the_engine``
DEFORMED_MASTER_FE_TOL = None

# MTP [D1/D2/D7, V4]: the recovered rotation angle error in radians
# of the pure-rotation cases.  DRAFTING NOTE, not a live number
# (emptied 2026-09-07 at the adversarial review, with
# WARP_REFIT_TOL_480 and for the same reason): the drafted seed was
# 1e-5 rad.
# MEASURING RECIPE: ``test_in_plane_rotation`` below
ROTATION_TOL_RAD = None

# MTP [D6, V6]: the agreement of the fitted homographies with the
# closed-form beam-scan phantom, uncorrected, in binned pixels.  THE
# sign pin of D6.3: the passing sign set is recorded in requirements
# D6 with its date.  The map this is measured on is the ``(2, 3)``
# grid with 400.0 unit steps of ``phantom_map`` below, whose offsets
# were MEASURED at 11.43 px, 5.37 px and 8.1e-3 in
# ``(gamma_x, gamma_y, alpha_s - 1)``.
# MEASURING RECIPE: ``test_phantom_uncorrected`` below
PC_PHANTOM_TOL = None

# MTP [D6, V6]: the residual deformation of the CORRECTED phantom,
# which must sit at the interpolation floor.
# MEASURING RECIPE: ``test_phantom_corrected`` below
PC_PHANTOM_FE_TOL = None

# MTP [D13/D15.6, V6]: the agreement of the RAW stored translations
# with the closed-form phantom translations, in binned pixels.
# MEASURING RECIPE: ``test_raw_homography_is_stored_uncorrected``
PC_PHANTOM_TRANSLATION_TOL = None

# MTP [D6.1, V6]: the agreement of the two per-point projection
# centre routes, an ENGINE result and not an algebraic identity, so
# it carries a measured band rather than the drafted guess of 1e-9.
# MEASURING RECIPE: ``test_single_pc_equals_per_point_pc`` below
PC_ROUTE_EQUIVALENCE_TOL = None

# MTP [D2.3, V2]: the error-warp norm in binned pixels between the
# engine run for a FIXED number of iterations from an explicit seed
# and the same iterations assembled by hand from ``_interpolation``
# and ``_preprocessing`` alone.  Both sides share the algebra
# exactly, so the expected agreement is the linear solver's own
# conditioning (about 1e-8 px on this 8 by 8 Hessian), and the band
# must be pinned FAR below the mutant separations MEASURED
# 2026-09-07 with scipy stand-ins for the not-yet-written kernel:
# composing the update on the wrong side 2.5e-3 px at one iteration
# and 8e-5 px at two, a flipped perspective sign 4e-3 px, warping
# the warped target 1.7e-3 px at two iterations (0 at one, where the
# two schemes coincide by construction), and solving ``H dp = +g``
# or swapping the gradients about 0.8 px.  A band of 1e-6 px would
# leave two orders of margin against the tightest of them; if the
# measured agreement is not far below 1e-4 px, one of the D2.3
# deviations is not implemented and that is the finding.
# MEASURING RECIPE: ``test_iterations_match_the_hand_built_update``
UPDATE_RULE_TOL = None

# MTP [D4.4, V2]: how far a defect planted in the EXCLUDED border
# moves the fit, in binned pixels.  It cannot be zero and the drafted
# bitwise claim was impossible: the D4.1 band-pass and the D3 spline
# prefilter are whole-pattern operations, so a planted defect reaches
# every kept pixel (measured 2026-09-07: max 0.055 intensity units of
# 242 through the band-pass alone, median 3.2e-5).
# MEASURING RECIPE: ``test_border_keeps_a_planted_defect_out``
BORDER_LEAK_TOL = None

# MTP [D4.4, V2]: the same for the dead-band cross.
# MEASURING RECIPE: ``test_dead_band_keeps_a_planted_cross_out``
DEAD_BAND_LEAK_TOL = None


# ----------------------------- Helpers ------------------------------ #


# ---------------- The plan 2.5 mutation list, mapped --------------- #
#
# Every mutant of plan section 2.5 with the test that kills it, so
# that the review gate reads one list rather than the whole suite.
# Updated 2026-09-07 at the adversarial review, which found six of
# them alive.
#
#  1 transpose the shape function ....... homography: layout_is_literal
#  2 drop the projective divide ......... homography: project_keeps_the
#                                         _projective_divide
#  3 compose the update as W(dp)^-1 . W . TestReferenceStateAndUpdate::
#                                         test_iterations_match_the_hand
#                                         _built_update (both arms share
#                                         the same fixed point, so no
#                                         accuracy band can separate
#                                         them: this is the only killer)
#  4 skip the W33 renormalization ....... homography: parameters_
#                                         renormalize_w33; at engine
#                                         level the update test above
#  5 swap gx/gy in GJ .................. test_reference_state_precompute
#  6 flip a GJ perspective-term sign .... test_reference_state_precompute
#                                         (a sign flip leaves GJ^T r = 0
#                                         untouched, so convergence is
#                                         not a killer)
#  7 Hessian from target gradients ...... test_reference_state_precompute
#                                         (the forward-additive scheme
#                                         converges to the same optimum)
#  8 solve H dp = +g .................... test_iterations_match_the_hand
#                                         _built_update, plus the
#                                         converged arms of V2
#  9 warp the reference, not the target . test_direction_pinned_once
# 10 re-warp the warped target .......... the two-iteration arm of
#                                         test_iterations_match_the_hand
#                                         _built_update (D2.3 puts the
#                                         difference at 1e-4 to 1e-6,
#                                         far below WARP_REFIT_TOL_480)
# 11 drop the ZMN mean, or use stdev .... test_zero_mean_normalize
# 12 mismatch the H and g normalization . test_intensity_scale_invariance
#                                         and the update test above
# 13 seed (dx, dy) transposed ........... test_seed_recovers_a_known_shift
# 14 border applied to one side only .... test_subregion_bounds_removes
#                                         _every_edge
# 15 PC-centred coordinates with +PC .... test_reference_state_uses_that
#                                         _frame (requirements D1.1 and
#                                         D1.3 amended 2026-09-07 with
#                                         the measured half-pixel term)
# 16 DD in unbinned pixels .............. geometry: only_the_binned_shape
#                                         _enters; at engine level
#                                         test_deformed_master_fe_through
#                                         _the_engine, whose out-of-plane
#                                         case is the one that carries DD
# 17 pcy/pcz swapped .................... geometry: conversion_is_literal
#                                         on a non-square detector; at
#                                         engine level the same Fe test
# 18 correction applied after conversion  geometry: correction_precedes
#                                         _conversion (algebraic) and
#                                         test_phantom_corrected (the
#                                         physical pin)
# 19 correction sign flipped ............ test_phantom_uncorrected on the
#                                         re-pinned phantom map, whose
#                                         drafted geometry could not see
#                                         it at all
# 20 Fe13 not divided by DD ............. homography: formula_is_literal;
#                                         at engine level the Fe test
# 21 Fe31 not multiplied by DD .......... the same pair
# 22 non-converged zeroed, not NaN ...... test_unreachable_threshold_exits
#                                         _at_the_cap and test_non
#                                         _converged_gets_nan_in_the
#                                         _derived_property
# 23 grain order not restored ........... test_multiple_grains_are
#                                         _restored_to_map_order (nothing
#                                         else in the suite calls the
#                                         engine with more than one grain)


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


def assert_close_to_scale(got, expected, name, tolerance=PRECOMPUTE_TOL):
    """Assert ``got`` equals ``expected`` to a SCALE RELATIVE band,
    ``atol = tolerance * max|expected|``.

    The metric of every machine-precision-class array comparison in
    this module: a gradient, a steepest-descent column and a zero-mean
    intensity all pass through zero, where a plain relative band is
    meaningless.
    """
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert got.shape == expected.shape, name
    scale = float(np.abs(expected).max())
    error = float(np.abs(got - expected).max())
    assert error <= tolerance * max(scale, 1.0), f"{name}: {error} of {scale}"


def matrix_of(h):
    """Return the ``(3, 3)`` shape function of *h*, built here rather
    than taken from the code under test: the whole point of the
    metrics below is that neither the oracle DATA nor the oracle
    METRIC may share an error with the module they judge."""
    h = np.asarray(h, dtype=np.float64)
    return np.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )


def parameters_of(matrix):
    """Return the eight parameters of a ``(3, 3)`` shape function,
    renormalized by ``W[2, 2]`` as requirements D2.3 demands."""
    matrix = np.asarray(matrix, dtype=np.float64) / matrix[2, 2]
    return np.array(
        [
            matrix[0, 0] - 1.0,
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 0],
            matrix[1, 1] - 1.0,
            matrix[1, 2],
            matrix[2, 0],
            matrix[2, 1],
        ]
    )


def project_with(matrix, x, y):
    """Return the projective image of ``(x, y)``, divide included."""
    s = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / s,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / s,
    )


def matrix_corner_norm(matrix, corners):
    """Return the D2.5 norm of a ``(3, 3)`` warp: the largest
    displacement it induces over the four subregion corners, in binned
    pixels."""
    x, y = corners[:, 0], corners[:, 1]
    warped_x, warped_y = project_with(matrix, x, y)
    return float(np.hypot(warped_x - x, warped_y - y).max())


def warp_norm(h, corners):
    """Return the D2.5 corner norm of ``W(h)``, computed here in plain
    numpy.  The independent twin of ``_homography.corner_norm``, which
    stays pinned by ``test_hrebsd_homography.py`` alone."""
    return matrix_corner_norm(matrix_of(h), corners)


def recovery_error(h_fit, h_true, corners):
    """Return the V2 recovery metric: the corner norm of the error
    warp ``W(h_true)**-1 . W(h_fit)`` in binned pixels.

    The independent twin of ``_homography.error_norm``.  Every oracle
    in this module measures with THIS function, so that a coordinated
    error inside the algebra module cannot leave the pattern-level
    oracles self-consistent and green.
    """
    matrix = np.linalg.inv(matrix_of(h_true)) @ matrix_of(h_fit)
    return matrix_corner_norm(matrix, corners)


def inverse_recovery_error(h_fit, h_true, corners):
    """Return the recovery error of *h_fit* against the INVERSE of
    *h_true*, which is the backward composition of the direction pin.
    ``W(h_true**-1)**-1 = W(h_true)``, so no inversion is needed."""
    return matrix_corner_norm(matrix_of(h_true) @ matrix_of(h_fit), corners)


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
    """Return the matrix ``R`` with ``v_crystal = R @ v_sample``,
    derived from ``rotate_vector`` itself so that no orix convention
    is assumed anywhere in this module."""
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    return rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T


def sample_to_spec_detector_matrix(detector):
    """Return the matrix ``M`` with ``v_detector = M @ v_sample`` in
    the SPEC's detector frame (x right, y down, z toward the screen),
    built from ``EBSDDetector.sample_to_detector`` and the measured
    y flip.  Nothing is hardcoded: the EMsoftOO 70 degree literal is
    a recorded deviation."""
    matrix = detector.sample_to_detector.to_matrix().squeeze()
    return DETECTOR_Y_FLIP @ matrix


def crystal_to_spec_detector_matrix(detector, rotation):
    """Return the matrix ``R`` with ``v_detector = R @ v_crystal``,
    the D7 frame chain a deformation gradient is imposed through."""
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
    master.  Because the engine's geometric model and this simulation
    are the same first-order model, the expected homography is EXACT.

    Parameters
    ----------
    detector
        Detector with one projection centre, or with one per map
        point together with *pc_index*.
    rotation
        The crystal orientation, an :class:`~orix.quaternion.Rotation`
        of size one.
    deformation
        The ``(3, 3)`` deformation gradient in the CRYSTAL frame, or
        ``None`` for the undeformed pattern.
    pc_index
        Flat index into a multi-projection-centre detector.
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


def pc_pixels_of(detector, index=0):
    """Return ``(PCx_px, PCy_px, DD_px)`` of one projection centre,
    computed here rather than through the code under test."""
    nrows, ncols = detector.shape
    pcx, pcy, pcz = np.atleast_2d(detector.pc_flattened)[index]
    return np.array([pcx * ncols, pcy * nrows, pcz * nrows])


def pc_centred_coordinates(shape, pc_px):
    """Return the PC-centred pixel coordinate grids ``(x, y)`` in
    kikuchipy's own geometry, MEASURED to be
    ``col + 0.5 - PCx_px`` and ``row + 0.5 - PCy_px`` (see the module
    docstring's half-pixel finding)."""
    nrows, ncols = shape
    rows, cols = np.indices((nrows, ncols), dtype=np.float64)
    return cols + 0.5 - pc_px[0], rows + 0.5 - pc_px[1]


def subregion_corners(shape, pc_px, border=0.05):
    """Return the ``(4, 2)`` PC-centred corners of the subregion, the
    support of the D2.5 convergence and recovery norms."""
    nrows, ncols = shape
    margin_row = int(round(border * nrows))
    margin_col = int(round(border * ncols))
    x0 = margin_col + 0.5 - pc_px[0]
    x1 = ncols - 1 - margin_col + 0.5 - pc_px[0]
    y0 = margin_row + 0.5 - pc_px[1]
    y1 = nrows - 1 - margin_row + 0.5 - pc_px[1]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def warp_with_skimage(image, h, pc_px):
    """Return *image* warped by ``W(h)`` with an INDEPENDENT warper.

    ``skimage.transform`` is a test-oracle-only import for this
    feature (requirements D18): no ``_hrebsd`` module may import it.
    The homography acts on PC-centred coordinates, so it is
    conjugated into the array frame first.  The shape function matrix
    is built inline rather than taken from the code under test, so
    that this oracle stays independent of it.
    """
    from skimage.transform import ProjectiveTransform, warp

    # a fresh writable copy: the cached oracle reference is read-only
    # and skimage's Cython warper refuses a read-only buffer
    image = np.array(image, dtype=np.float64)
    matrix = np.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )
    translation = np.eye(3)
    translation[0, 2] = pc_px[0] - 0.5
    translation[1, 2] = pc_px[1] - 0.5
    array_matrix = translation @ matrix @ np.linalg.inv(translation)
    transform = ProjectiveTransform(matrix=np.linalg.inv(array_matrix))
    return warp(image, transform, order=3, mode="reflect", preserve_range=True)


def random_small_homographies(n=8, dd=240.0, seed=0, scale=1.0):
    """Return random small homographies of the V2 design budget:
    translations to 5 px, rotations to 1 degree and strains to
    2e-3.  Built with plain numpy so the oracle's own setup never
    depends on the code under test.

    *scale* multiplies everything that carries a length: the
    translations and the two angles.  It exists because the budget is
    sized for 480 px patterns, where D4.4's ``border=0.05`` leaves
    24 px of margin, while the same 5 px translation at 60 px would
    push every subregion sample past the 3 px border into the mirror
    region and measure the boundary instead of the engine.  The
    strains are dimensionless and are not scaled.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.deg2rad(1.0), np.deg2rad(1.0), size=n) * scale
    h = np.zeros((n, N_HOMOGRAPHY_PARAMETERS))
    h[:, 0] = rng.uniform(-2e-3, 2e-3, size=n)
    h[:, 4] = rng.uniform(-2e-3, 2e-3, size=n)
    h[:, 1] = -np.sin(theta) + rng.uniform(-1e-3, 1e-3, size=n) * scale
    h[:, 3] = np.sin(theta) + rng.uniform(-1e-3, 1e-3, size=n) * scale
    h[:, 2] = rng.uniform(-5.0, 5.0, size=n) * scale
    h[:, 5] = rng.uniform(-5.0, 5.0, size=n) * scale
    h[:, 6] = rng.uniform(-np.deg2rad(0.5), np.deg2rad(0.5), size=n) * scale / dd
    h[:, 7] = rng.uniform(-np.deg2rad(0.5), np.deg2rad(0.5), size=n) * scale / dd
    return h


# The 60 px arm's share of the 480 px design budget, so that the
# largest translation stays inside that pattern's own 3 px border
SCALE_60 = SHAPE_60[0] / SHAPE_480[0]


def closed_form_phantom(pc_reference, pc_target):
    """Return the beam-scan phantom homography of requirements D6.2,
    written out here rather than imported from ``_geometry``.

    In the reference-PC-centred pixel frame a pure projection centre
    change maps ``xi' = alpha_s * xi + gamma`` with
    ``gamma = PC_target - PC_reference`` in binned pixels EXACTLY and
    ``alpha_s`` the detector distance ratio; the drafted orientation
    of that ratio and the signs are what validation V6 pins, so the
    expectation may not come from the module that would carry the
    error.  EMsoftOO's ``(delta - patcent)(alpha_s - 1)`` term
    (``mod_HREBSDDIC.f90:955-970``) belongs to its absolute-PC
    coordinates and vanishes in this frame.
    """
    alpha_s = pc_target[2] / pc_reference[2]
    h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    h[0] = alpha_s - 1.0
    h[4] = alpha_s - 1.0
    h[2] = pc_target[0] - pc_reference[0]
    h[5] = pc_target[1] - pc_reference[1]
    return h


def hand_built_precompute(reference, pc_px, *, border=0.05, cutoffs=(0.05, None)):
    """Return the D2.1 per-reference precompute, assembled here.

    Built from ``_preprocessing`` and ``_interpolation`` ONLY, which
    have their own oracles (validation V0 and the D4 tests above), so
    that the engine's own assembly of the steepest-descent images and
    the Hessian is judged by something that does not share its code.
    Requirements D2.1, literally::

        GJ = (gx*x, gx*y, gx, gy*x, gy*y, gy,
              -(gx*x**2 + gy*x*y), -(gx*x*y + gy*y**2))
        H  = (2/ref_norm**2) * sum_px GJ GJ^T
    """
    shape = reference.shape
    mask = subregion_mask(shape, border=border)
    transfer = band_pass_transfer_function(shape, cutoffs)
    preprocessed = preprocess(reference, transfer_function=transfer)
    x_grid, y_grid = pc_centred_coordinates(shape, pc_px)
    xi_x = x_grid[mask]
    xi_y = y_grid[mask]
    centred = preprocessed[mask] - preprocessed[mask].mean()
    reference_norm = float(np.linalg.norm(centred))
    coefficients = spline_coefficients(preprocessed, dtype=np.float64)
    plane_x, plane_y = gradient_planes(coefficients)
    gx = plane_x[mask]
    gy = plane_y[mask]
    steepest_descent = np.stack(
        [
            gx * xi_x,
            gx * xi_y,
            gx,
            gy * xi_x,
            gy * xi_y,
            gy,
            -(gx * xi_x**2 + gy * xi_x * xi_y),
            -(gx * xi_x * xi_y + gy * xi_y**2),
        ],
        axis=1,
    )
    return {
        "mask": mask,
        "transfer": transfer,
        "xi_x": xi_x,
        "xi_y": xi_y,
        "reference": centred / reference_norm,
        "reference_norm": reference_norm,
        "steepest_descent": steepest_descent,
        "hessian": (2.0 / reference_norm**2) * (steepest_descent.T @ steepest_descent),
    }


def hand_built_icgn(reference, target, pc_px, h0, n_iterations, **kwargs):
    """Return the homography after *n_iterations* of an independent
    accumulated-W IC-GN, the loop of requirements D2.3.

    The ORIGINAL target is re-warped by the ACCUMULATED warp every
    iteration and interpolated exactly once, which is the frozen
    deviation from EMsoftOO's progressive warp of the warped target
    (``mod_DIC.f90:684-747``).  The update is
    ``W <- W . W(dp)**-1`` with ``dp`` solving ``H dp = -g`` and the
    matched D2.1/D2.3 scales ``2/ref_norm**2`` and ``2/ref_norm``.
    """
    state = hand_built_precompute(reference, pc_px, **kwargs)
    xi_x, xi_y = state["xi_x"], state["xi_y"]
    coefficients = spline_coefficients(
        preprocess(target, transfer_function=state["transfer"]), dtype=np.float64
    )
    matrix = matrix_of(h0)
    for _ in range(n_iterations):
        warped_x, warped_y = project_with(matrix, xi_x, xi_y)
        values = evaluate(
            coefficients,
            np.ascontiguousarray(warped_x + pc_px[0] - 0.5),
            np.ascontiguousarray(warped_y + pc_px[1] - 0.5),
        )
        centred = values - values.mean()
        residuals = state["reference"] - centred / np.linalg.norm(centred)
        gradient = (2.0 / state["reference_norm"]) * (
            state["steepest_descent"].T @ residuals
        )
        step = np.linalg.solve(state["hessian"], -gradient)
        matrix = matrix @ np.linalg.inv(matrix_of(step))
        matrix = matrix / matrix[2, 2]
    return parameters_of(matrix)


def deformation_cases():
    """Return the imposed detector-frame deformation gradients of the
    V3 oracle: a pure strain, a pure shear, a pure rotation and a
    mixed case, all at the 1e-3 scale HREBSD is built for."""
    strain = np.eye(3)
    strain[0, 0] += 1.5e-3
    strain[1, 1] -= 8e-4
    shear = np.eye(3)
    shear[0, 1] = 6e-4
    shear[1, 0] = 6e-4
    theta = np.deg2rad(0.7)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return [
        ("pure_strain", strain),
        ("pure_shear", shear),
        ("pure_rotation", rotation),
        ("mixed", rotation @ strain @ shear),
    ]


@functools.lru_cache(maxsize=2)
def oracle_reference(shape=SHAPE_480):
    """Return the undeformed 480 px projected reference pattern and
    its detector, cached because the projection costs about 45 ms."""
    detector = make_detector(shape=shape)
    pattern = project_pattern(detector, generic_rotation())
    pattern.flags.writeable = False
    return pattern, detector


def make_state(reference, pc_px, **kwargs):
    """Return a :class:`ReferenceState` built with the engine's own
    default preprocessing chain."""
    border = kwargs.pop("border", 0.05)
    cutoffs = kwargs.pop("filter_cutoffs", (0.05, None))
    use_window = kwargs.pop("window", False)
    dead_band = kwargs.pop("dead_band", None)
    shape = reference.shape
    mask = subregion_mask(shape, border=border, dead_band=dead_band)
    return ReferenceState(
        reference,
        mask,
        pc_px,
        transfer_function=band_pass_transfer_function(shape, cutoffs),
        window=hann_window(shape) if use_window else None,
        **kwargs,
    )


def one_point_xmap(navigation_shape, step_sizes=(1.0, 1.0)):
    """Return a single-phase crystal map of identity orientations."""
    arrays, size = create_coordinate_arrays(navigation_shape, step_sizes)
    arrays["rotations"] = Rotation.identity((size,))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    return xmap


# ============ The frame the whole feature is measured in ============ #


class TestPcCentredFrame:
    """The relation between the spec's PC-centred pixel frame and
    kikuchipy's own projection geometry.  A pure library measurement
    which needs no ``_hrebsd`` code and therefore passes before the
    implementation lands; it exists to make the half-pixel question
    of the module docstring explicit and reviewable.  [D1]"""

    @pytest.mark.parametrize("shape", [(40, 60), SHAPE_60])
    def test_pc_centred_frame_matches_kikuchipy_geometry(self, shape):
        detector = make_detector(shape=shape, binning=8)
        pc_px = pc_pixels_of(detector)
        cosines = _get_direction_cosines_from_detector(detector)
        rays = cosines @ sample_to_spec_detector_matrix(detector).T
        rays = rays / rays[:, 2:3] * pc_px[2]
        nrows, ncols = shape
        rows, cols = np.indices((nrows, ncols), dtype=np.float64)
        # the measured relation, exact to 2e-14 px: the PIXEL CENTRE
        # convention, half a pixel away from a literal reading of
        # D1.1 plus D1.2
        assert np.abs(rays[:, 0] - (cols.ravel() + 0.5 - pc_px[0])).max() < 1e-9
        assert np.abs(rays[:, 1] - (rows.ravel() + 0.5 - pc_px[1])).max() < 1e-9
        # the z axis points from the sample toward the screen, so the
        # ray of D1.5 is ``(xi_x, xi_y, DD_px)``
        assert np.all(rays[:, 2] > 0)

    def test_reference_state_uses_that_frame(self):
        # the engine's own coordinates must be the ones the detector
        # geometry implies, or every projection centre derived
        # quantity carries a systematic half-pixel error
        reference, detector = oracle_reference(SHAPE_60)
        pc_px = pc_pixels_of(detector)
        state = make_state(reference, pc_px)
        mask = subregion_mask(SHAPE_60, border=0.05)
        expected_x, expected_y = pc_centred_coordinates(SHAPE_60, pc_px)
        np.testing.assert_allclose(
            state.xi_x, expected_x[mask], rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            state.xi_y, expected_y[mask], rtol=0, atol=ALGEBRA_TOL
        )


# ============ D2.1 and D2.3 -- the loop, assembled by hand ========== #


class TestReferenceStateAndUpdate:
    """The per-reference precompute and the IC-GN update itself,
    against an independent assembly of requirements D2.1 and D2.3
    built from ``_preprocessing`` and ``_interpolation`` alone.

    These exist because the drafted suite asserted NOTHING about
    either: a swapped ``gx``/``gy`` in ``GJ`` and a flipped
    perspective-term sign both leave the zero-gradient fixed point
    ``GJ^T r = 0`` untouched and only change the descent direction,
    and a Hessian built from the TARGET gradients is the
    forward-additive scheme, which converges to the same optimum.  So
    ``assert result["converged"]`` is not a reliable killer for any of
    the three, and no accuracy band separates them either.
    [D2.1/D2.3]"""

    def test_reference_state_precompute(self):
        reference, detector = oracle_reference(SHAPE_60)
        pc_px = pc_pixels_of(detector)
        # 64-bit storage, so that the comparison is against the
        # algebra and not against the D17 provisional 32-bit
        # coefficient storage
        state = make_state(reference, pc_px, coefficient_dtype=np.float64)
        expected = hand_built_precompute(reference, pc_px)
        n_pixels = expected["steepest_descent"].shape[0]
        assert state.reference_norm == pytest.approx(
            expected["reference_norm"], rel=PRECOMPUTE_TOL
        )
        # every band below is scale relative, ``atol = tol * max|ref|``
        # over the quantity compared, because a gradient and a
        # zero-mean intensity both pass through zero and a plain
        # relative comparison would blow up there.  The eight
        # steepest-descent columns are compared one at a time: their
        # scales span the subregion half width squared
        assert_close_to_scale(
            np.asarray(state.reference), expected["reference"], "reference"
        )
        got = np.asarray(state.steepest_descent)
        assert got.shape == (n_pixels, N_HOMOGRAPHY_PARAMETERS)
        for column in range(N_HOMOGRAPHY_PARAMETERS):
            assert_close_to_scale(
                got[:, column],
                expected["steepest_descent"][:, column],
                f"steepest_descent[:, {column}]",
            )
        hessian = np.asarray(state.hessian)
        assert hessian.shape == (
            N_HOMOGRAPHY_PARAMETERS,
            N_HOMOGRAPHY_PARAMETERS,
        )
        assert_close_to_scale(hessian, expected["hessian"], "hessian")

    def test_steepest_descent_mutants_are_distinguishable(self):
        # the three mutants named above, spelled out: each is a
        # genuinely different array on this reference, so the
        # equality above is a real discriminator and not an identity
        # that holds for all of them
        reference, detector = oracle_reference(SHAPE_60)
        pc_px = pc_pixels_of(detector)
        expected = hand_built_precompute(reference, pc_px)
        got = expected["steepest_descent"]
        swapped = got[:, [3, 4, 5, 0, 1, 2, 6, 7]]
        assert not np.allclose(got, swapped)
        flipped = got.copy()
        flipped[:, 6:] *= -1.0
        assert not np.allclose(got, flipped)
        # and the Hessian scale of D2.1 is not cosmetic either: the
        # unscaled ``sum GJ GJ^T`` differs from it by the factor the
        # matched D2.3 pairing needs
        unscaled = got.T @ got
        assert not np.allclose(expected["hessian"], unscaled)

    @pytest.mark.parametrize("n_iterations", [1, 2])
    def test_iterations_match_the_hand_built_update(self, n_iterations):
        # THE engine-level pin of the D2.3 update: the composition
        # side ``W <- W . W(dp)**-1`` (whose mutant shares the same
        # fixed point and is therefore invisible to every accuracy
        # band in this suite), the sign of ``H dp = -g``, the
        # ``W[2, 2]`` renormalization, and -- in the two-iteration
        # arm -- the accumulated-W re-warp of the ORIGINAL target,
        # whose warp-of-warp mutant D2.3 puts at 1e-4 to 1e-6, three
        # orders below WARP_REFIT_TOL_480
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=17, scale=0.5)[0]
        target = warp_with_skimage(reference, h_true, pc_px)
        # an explicit seed inside the basin but off the answer, so
        # that the increment is not degenerate and the seed is not
        # part of what is compared
        h0 = h_true.copy()
        h0[2] += 0.3
        h0[5] -= 0.25
        state = make_state(reference, pc_px, coefficient_dtype=np.float64)
        result = fit_pattern(
            state,
            target,
            h0=h0,
            max_iterations=n_iterations,
            min_step=1e-12,
        )
        assert result["num_iterations"] == n_iterations
        expected = hand_built_icgn(reference, target, pc_px, h0, n_iterations)
        assert_within(
            recovery_error(result["h"], expected, corners),
            UPDATE_RULE_TOL,
            f"UPDATE_RULE_TOL ({n_iterations} iterations)",
        )
        # the update really moved, so the comparison above is not
        # trivially satisfied by an engine that does nothing
        assert recovery_error(result["h"], h0, corners) > 1e-3


# ==================== D4 -- in-engine preprocessing ================= #


class TestPreprocessing:
    """The band-pass, the optional window, the subregion and the
    zero-mean normalization.  [D4]"""

    # a NON-SQUARE shape as well: requirements D4.1 states the
    # cut-offs as fractions of the pattern WIDTH, and with
    # ``nrows == ncols`` a ``shape[0]`` versus ``shape[1]`` mutant is
    # invisible.  The sibling modules use non-square shapes for
    # exactly this reason (``test_hrebsd_interpolation.py`` SHAPES,
    # ``test_hrebsd_geometry.py`` SHAPE)
    @pytest.mark.parametrize("shape", [SHAPE_60, (37, 61)])
    def test_band_pass_default_is_high_pass_only(self, shape):
        # the frozen default ``(0.05, None)`` is EMsoftOO's
        # ``hipassw`` default and switches the low-pass half off
        got = band_pass_transfer_function(shape, (0.05, None))
        expected = kp.filters.highpass_fft_filter(shape, cutoff=0.05 * shape[1])
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    @pytest.mark.parametrize("shape", [SHAPE_60, (37, 61)])
    def test_band_pass_is_the_product_of_the_two_halves(self, shape):
        got = band_pass_transfer_function(shape, (0.05, 0.4))
        expected = kp.filters.highpass_fft_filter(
            shape, cutoff=0.05 * shape[1]
        ) * kp.filters.lowpass_fft_filter(shape, cutoff=0.4 * shape[1])
        np.testing.assert_allclose(got, expected, rtol=0, atol=ALGEBRA_TOL)

    def test_band_pass_none_switches_filtering_off(self):
        assert band_pass_transfer_function(SHAPE_60, (None, None)) is None

    def test_band_pass_validation(self):
        with pytest.raises(ValueError):
            band_pass_transfer_function(SHAPE_60, (0.05,))
        with pytest.raises(ValueError):
            band_pass_transfer_function(SHAPE_60, (-0.05, None))

    def test_subregion_bounds_removes_every_edge(self):
        # the ``border applied to one side only`` mutant of plan 2.5
        nrows, ncols = 200, 100
        row0, row1, col0, col1 = subregion_bounds((nrows, ncols), border=0.05)
        assert row0 == 10
        assert row1 == nrows - 10
        assert col0 == 5
        assert col1 == ncols - 5

    def test_subregion_mask_polarity_and_shape(self):
        # frozen internal convention: ``True`` means the pixel is
        # USED, the opposite of the user facing navigation mask
        shape = (200, 100)
        mask = subregion_mask(shape, border=0.05)
        assert mask.dtype == np.bool_
        assert mask.shape == shape
        assert mask[100, 50]
        assert not mask[0, 50]
        assert not mask[100, 0]
        assert not mask[-1, 50]
        assert not mask[100, -1]
        assert mask.sum() == (200 - 20) * (100 - 10)

    def test_subregion_mask_dead_band(self):
        # a cross of dead camera columns and rows, EMsoftOO's
        # ``cross(4)`` semantics
        shape = (100, 100)
        mask = subregion_mask(shape, border=0.05, dead_band=(48, 52, 30, 33))
        assert not mask[:, 48:52].any()
        assert not mask[30:33, :].any()
        assert mask[60, 60]

    def test_subregion_validation(self):
        with pytest.raises(ValueError):
            subregion_mask(SHAPE_60, border=0.5)
        with pytest.raises(ValueError):
            subregion_mask(SHAPE_60, border=-0.01)
        with pytest.raises(ValueError):
            subregion_mask(SHAPE_60, dead_band=(1, 2, 3))

    def test_preprocess_is_affine_in_the_intensities(self):
        # the D4.2 pin: NO adaptive histogram equalization anywhere in
        # the DIC chain, because a nonlinear locally varying
        # intensity map violates the affine model ZNSSD assumes.  A
        # linear chain commutes with a scale exactly
        reference, _ = oracle_reference(SHAPE_60)
        transfer = band_pass_transfer_function(SHAPE_60, (0.05, None))
        once = preprocess(reference, transfer_function=transfer)
        scaled = preprocess(4.0 * reference, transfer_function=transfer)
        np.testing.assert_allclose(scaled, 4.0 * once, rtol=1e-12, atol=0)

    def test_preprocess_returns_float64(self):
        reference, _ = oracle_reference(SHAPE_60)
        got = preprocess(reference.astype(np.uint8))
        assert got.dtype == np.float64
        assert got.shape == SHAPE_60

    def test_zero_mean_normalize(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=257)
        normalized, norm = zero_mean_normalize(values)
        centred = values - values.mean()
        assert norm == pytest.approx(float(np.linalg.norm(centred)), rel=ALGEBRA_TOL)
        np.testing.assert_allclose(normalized, centred / norm, rtol=ALGEBRA_TOL, atol=0)
        assert abs(float(normalized.mean())) < 1e-14
        assert float(np.linalg.norm(normalized)) == pytest.approx(1.0)

    def test_zero_mean_normalize_refuses_a_constant(self):
        with pytest.raises(ValueError):
            zero_mean_normalize(np.full(16, 3.0))


# ====================== D5 -- the initial guess ===================== #


class TestInitialGuess:
    """The upsampled phase cross-correlation seed.  [D5]"""

    def test_seed_recovers_a_known_shift(self):
        # the ``(dx, dy) transposed`` mutant of plan 2.5 dies here:
        # a shift of +3 rows and -2 columns must land in h23 and h13
        # respectively, not the other way round
        reference, _ = oracle_reference(SHAPE_60)
        target = ndi_shift(reference, (3.0, -2.0), order=3, mode="reflect")
        got = initial_guess(reference, target)
        assert got.shape == (N_HOMOGRAPHY_PARAMETERS,)
        assert got[2] == pytest.approx(-2.0, abs=0.1)
        assert got[5] == pytest.approx(3.0, abs=0.1)
        # nothing but the two translations is seeded
        assert np.array_equal(got[[0, 1, 3, 4, 6, 7]], np.zeros(6))

    def test_upsample_factor_default_is_sixteen(self):
        parameters = inspect.signature(initial_guess).parameters
        assert parameters["upsample_factor"].default == 16
        assert parameters["upsample_factor"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_subpixel_shift(self):
        reference, _ = oracle_reference(SHAPE_60)
        target = ndi_shift(reference, (0.75, 0.0), order=3, mode="reflect")
        got = initial_guess(reference, target, upsample_factor=16)
        # 1/16 px seed precision, so a 0.75 px shift is representable
        assert got[5] == pytest.approx(0.75, abs=1.0 / 16)


# ============= V1 (pattern half) -- the warp direction ============== #


class TestWarpDirection:
    """The fitted homography maps REFERENCE coordinates to TARGET
    coordinates.  Tested in both compositions once and then frozen,
    which is what makes every later sign in this feature unambiguous.
    EMsoftOO's inverted shape function with its global ``h <- -h`` is
    a recorded deviation.  [D2/V1]"""

    def test_direction_pinned_once(self):
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=100)[0]
        # ``target(W xi) = reference(xi)``, the frozen direction
        target = warp_with_skimage(reference, h_true, pc_px)
        state = make_state(reference, pc_px)
        result = fit_pattern(state, target)
        assert result["converged"]
        forward = recovery_error(result["h"], h_true, corners)
        # the opposite direction is the same fit compared with the
        # INVERSE homography, and it must be much worse: the two
        # differ by twice the imposed warp, several pixels here
        backward = inverse_recovery_error(result["h"], h_true, corners)
        assert forward < backward
        assert_within(forward, WARP_REFIT_TOL_480, "WARP_REFIT_TOL_480")
        # the 1.0 px floor is not a guess: this seed's imposed warp
        # displaces the subregion corners by several pixels (measured
        # 8.4 px worst for seed 100), so a fit of the wrong direction
        # sits at about twice that, a decade above this floor
        assert warp_norm(h_true, corners) > 1.0
        assert backward > 1.0


# ================= V2 -- the synthetic warp-refit =================== #


class TestWarpRefit:
    """The core oracle: warp a reference by a known homography with
    an independent warper, refit, compare with the corner
    displacement of the error warp.  [D2/D3/D5/V2]"""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_warp_refit_small_h_480(self, seed):
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        state = make_state(reference, pc_px)
        worst = 0.0
        for h_true in random_small_homographies(n=6, dd=pc_px[2], seed=seed):
            target = warp_with_skimage(reference, h_true, pc_px)
            result = fit_pattern(state, target)
            assert result["converged"], h_true
            worst = max(worst, recovery_error(result["h"], h_true, corners))
        assert_within(worst, WARP_REFIT_TOL_480, "WARP_REFIT_TOL_480")

    def test_warp_refit_60px(self):
        # the shipped 60 px Ni patterns: a qualitative regime, whose
        # band is recorded and never quoted as a precision claim.
        # The batch is SCALED to the pattern (see SCALE_60): at 60 px
        # the D4.4 border is 3 px, so the unscaled 480 px budget of
        # 5 px translations would put every subregion sample outside
        # the frame and measure the mirror boundary instead
        signal = kp.data.nickel_ebsd_small()
        signal.remove_static_background(show_progressbar=False)
        reference = np.asarray(signal.data[0, 0], dtype=np.float64)
        detector = make_detector(shape=SHAPE_60, binning=8)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_60, pc_px)
        state = make_state(reference, pc_px)
        worst = 0.0
        for h_true in random_small_homographies(
            n=3, dd=pc_px[2], seed=5, scale=SCALE_60
        ):
            target = warp_with_skimage(reference, h_true, pc_px)
            result = fit_pattern(state, target)
            worst = max(worst, recovery_error(result["h"], h_true, corners))
        assert_within(worst, WARP_REFIT_TOL_60, "WARP_REFIT_TOL_60")

    def test_intensity_scale_invariance(self):
        # THE killer of a mismatched D2.1/D2.3 Hessian and gradient
        # normalization: an unmatched pairing scales the step by
        # ``1/s**2`` under ``ref -> s*ref`` and can satisfy the exit
        # criterion at the seed.  Powers of two are EXACT through the
        # linear preprocessing, the zero-mean normalization and the
        # matched pairing, so the fitted parameters must be BITWISE
        # identical
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=7)[0]
        target = warp_with_skimage(reference, h_true, pc_px)
        base = fit_pattern(make_state(reference, pc_px), target)["h"]
        for scale_reference, scale_target in [
            (2.0**-10, 1.0),
            (1.0, 2.0**10),
            (2.0**10, 2.0**-10),
        ]:
            scaled = fit_pattern(
                make_state(scale_reference * reference, pc_px),
                scale_target * target,
            )["h"]
            assert np.array_equal(scaled, base), (
                scale_reference,
                scale_target,
            )
        # one generic factor, which differs only in the last bits
        generic = fit_pattern(make_state(3.7 * reference, pc_px), 0.31 * target)["h"]
        assert_within(
            recovery_error(generic, base, corners),
            INTENSITY_SCALE_GENERIC_TOL,
            "INTENSITY_SCALE_GENERIC_TOL",
        )

    def test_convergence_metadata(self):
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=11)[0]
        target = warp_with_skimage(reference, h_true, pc_px)
        state = make_state(reference, pc_px)
        result = fit_pattern(state, target)
        assert set(result) == {
            "h",
            "residual",
            "num_iterations",
            "norm_dp",
            "converged",
        }
        # ``is True`` deliberately, not ``bool(...) is True``: the
        # ``fit_pattern`` docstring pins ``"converged"`` as a PYTHON
        # bool, and the crystal map property it feeds is assembled
        # from these, so a stray ``np.bool_`` is a contract break
        assert result["converged"] is True
        assert 0 < result["num_iterations"] < 50
        assert result["norm_dp"] < 1e-3
        assert np.isfinite(result["residual"])

    def test_unreachable_threshold_exits_at_the_cap(self):
        # non-converged points keep their LAST ITERATE and say so;
        # they are never zeroed (a recorded deviation from
        # ``mod_HREBSDDIC.f90:868-870``)
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=12)[0]
        target = warp_with_skimage(reference, h_true, pc_px)
        state = make_state(reference, pc_px)
        result = fit_pattern(state, target, min_step=1e-12, max_iterations=8)
        assert result["converged"] is False
        assert result["num_iterations"] == 8
        assert np.all(np.isfinite(result["h"]))
        assert not np.array_equal(result["h"], np.zeros(N_HOMOGRAPHY_PARAMETERS))

    def test_seed_required_for_large_translation(self):
        # pins D5's reason for existing, and kills a dropped-seed
        # mutant: EMsoftOO's identity-only start fails here.
        # The negative arm is the one validation V2 asks for
        # LITERALLY ("a +-15 px translation case fails from identity
        # seeding and succeeds with the phase-XC seed"), so it stays
        # a gate rather than a recorded measurement; the 1.0 px floor
        # below is a decade under the 21 px corner displacement the
        # imposed warp itself induces, which the first assertion
        # states rather than assumes
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = np.zeros(N_HOMOGRAPHY_PARAMETERS)
        h_true[2] = 15.0
        h_true[5] = -15.0
        target = warp_with_skimage(reference, h_true, pc_px)
        state = make_state(reference, pc_px)
        assert warp_norm(h_true, corners) > 20.0
        seeded = fit_pattern(state, target)
        assert seeded["converged"]
        assert_within(
            recovery_error(seeded["h"], h_true, corners),
            WARP_REFIT_TOL_480,
            "WARP_REFIT_TOL_480",
        )
        identity_seeded = fit_pattern(
            state, target, h0=np.zeros(N_HOMOGRAPHY_PARAMETERS)
        )
        failed = (
            not identity_seeded["converged"]
            or recovery_error(identity_seeded["h"], h_true, corners) > 1.0
        )
        assert failed

    def test_border_keeps_a_planted_defect_out(self):
        # D4.4: a feature living only in the excluded border must
        # barely reach the fit.  NOT bitwise, and the drafted
        # ``np.array_equal`` claim was impossible for any conformant
        # implementation: D4.1's band-pass and D3's spline prefilter
        # are BOTH whole-pattern operations, so a planted defect
        # reaches every kept pixel (measured 2026-09-07: max 0.055
        # intensity units of 242 through the band-pass alone, median
        # 3.2e-5).  What the border buys is measured instead, against
        # the same strips planted where the border does not cover
        # them
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=13, scale=0.25)[0]
        # the imposed warp must keep every kept pixel's warped
        # bicubic support (2 px) clear of the planted 8 px strips,
        # which the 24 px border leaves 16 px of room for
        assert warp_norm(h_true, corners) < 14.0
        target = warp_with_skimage(reference, h_true, pc_px)
        planted = target.copy()
        planted[:8, :] = planted.max() * 4.0
        planted[:, -8:] = 0.0
        # the SAME strips planted INSIDE the subregion instead, the
        # contrast arm: same state, same preprocessing, only the
        # location differs, so it needs no tolerance at all
        inside = target.copy()
        inside[100:108, :] = inside.max() * 4.0
        inside[:, 300:308] = 0.0
        state = make_state(reference, pc_px, border=0.05)
        clean = fit_pattern(state, target)["h"]
        leak = recovery_error(fit_pattern(state, planted)["h"], clean, corners)
        direct = recovery_error(fit_pattern(state, inside)["h"], clean, corners)
        assert leak < direct
        assert_within(leak, BORDER_LEAK_TOL, "BORDER_LEAK_TOL")

    def test_dead_band_keeps_a_planted_cross_out(self):
        # the same for the dead-pixel cross, with the SAME correction
        # to the drafted bitwise claim.  The declared band is wider
        # than the planted cross by eight pixels on every side and
        # the imposed warp displaces no kept pixel by more than two,
        # so no kept pixel's warped bicubic support (which reaches
        # 2 px) can sample the planted values.  What is left to
        # measure is the leak through the whole-pattern
        # preprocessing, which is what the band pins.  The drafted
        # arm planted INSIDE the support of kept pixels and then
        # demanded a bitwise identical fit, which is doubly
        # impossible
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        h_true = random_small_homographies(n=1, dd=pc_px[2], seed=14, scale=0.25)[0]
        assert warp_norm(h_true, corners) < 2.0
        target = warp_with_skimage(reference, h_true, pc_px)
        planted = target.copy()
        planted[:, 238:242] = planted.max() * 4.0
        planted[100:103, :] = 0.0
        # the SAME cross planted where the band does NOT cover it,
        # the contrast arm, which needs no tolerance
        inside = target.copy()
        inside[:, 300:304] = inside.max() * 4.0
        inside[200:203, :] = 0.0
        state = make_state(reference, pc_px, dead_band=(230, 250, 92, 111))
        clean = fit_pattern(state, target)["h"]
        leak = recovery_error(fit_pattern(state, planted)["h"], clean, corners)
        direct = recovery_error(fit_pattern(state, inside)["h"], clean, corners)
        assert leak < direct
        assert_within(leak, DEAD_BAND_LEAK_TOL, "DEAD_BAND_LEAK_TOL")

    def test_dtype_ab_harness(self):
        # the D17 measurement harness: 32-bit coefficient storage is
        # pinned only if it degrades the recovery by less than ten per
        # cent of the 64-bit error.  The verdict and the numbers are
        # recorded in validation.md at the implementation gate
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        state64 = make_state(reference, pc_px, coefficient_dtype=np.float64)
        state32 = make_state(reference, pc_px, coefficient_dtype=np.float32)
        worst64 = 0.0
        worst32 = 0.0
        for h_true in random_small_homographies(n=4, dd=pc_px[2], seed=15):
            target = warp_with_skimage(reference, h_true, pc_px)
            worst64 = max(
                worst64,
                recovery_error(fit_pattern(state64, target)["h"], h_true, corners),
            )
            worst32 = max(
                worst32,
                recovery_error(fit_pattern(state32, target)["h"], h_true, corners),
            )
        assert worst32 <= (1.0 + DTYPE_F32_MAX_DEGRADATION) * worst64, (
            f"32-bit storage degrades the recovery from {worst64} to "
            f"{worst32} px, beyond the D17 criterion; re-pin the "
            "default to 64-bit with a dated record"
        )


# ============ V3 -- the deformed-master end-to-end oracle =========== #


class TestDeformedMaster:
    """Patterns projected from the shipped Ni Lambert master with an
    imposed deformation gradient.  Because the engine's geometric
    model and this simulation are the same first-order model, the
    expected homography is EXACT rather than approximate.
    [D2-D7/V3]"""

    @pytest.mark.parametrize("name,fe_detector", deformation_cases())
    def test_deformed_master_homography_recovery(self, name, fe_detector):
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        rotation = generic_rotation()
        reduced = np.asarray(fe_detector) / fe_detector[2, 2]
        target = project_pattern(
            detector,
            rotation,
            impose_detector_frame_fe(detector, rotation, reduced),
        )
        expected = fe_to_homography(reduced, np.zeros(2), pc_px[2])
        state = make_state(reference, pc_px)
        result = fit_pattern(state, target)
        assert result["converged"], name
        assert_within(
            recovery_error(result["h"], expected, corners),
            DEFORMED_MASTER_H_TOL,
            f"DEFORMED_MASTER_H_TOL ({name})",
        )

    def test_deformed_master_fe_through_the_engine(self):
        # THE end-to-end pin of the D15.6 ``Fe`` property, which no
        # other test in this suite supplies: every other ``Fe``
        # assertion here is NaN, the identity, or one route against
        # another, and all three survive a shared mutation.  This one
        # compares an engine result with a tensor built OUTSIDE the
        # feature, so the detector distance in binned pixels, the
        # pcy and pcz roles, the row-major flattening and which
        # pattern is reference all have to be right at once.  The
        # cases are the ASYMMETRIC ones: a symmetric tensor cannot
        # see a transposition.  An OUT-OF-PLANE tilt is included on
        # purpose: every case of ``deformation_cases`` is in-plane,
        # so ``Fe13``, ``Fe23``, ``Fe31`` and ``Fe32`` are all zero
        # there and the detector distance would cancel out of the
        # comparison, leaving the ``DD in unbinned pixels``, ``Fe13
        # not divided by DD`` and ``Fe31 not multiplied by DD``
        # mutants of plan 2.5 alive at engine level
        reference, detector = oracle_reference(SHAPE_480)
        rotation = generic_rotation()
        omega = np.deg2rad(1.0)
        cos, sin = np.cos(omega), np.sin(omega)
        cases = [
            (name, fe)
            for name, fe in deformation_cases()
            if name in ("pure_rotation", "mixed")
        ]
        cases.append(
            (
                "out_of_plane_tilt",
                np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]) / cos,
            )
        )
        patterns = [reference]
        expected = []
        for _, fe_detector in cases:
            reduced = np.asarray(fe_detector) / fe_detector[2, 2]
            expected.append(reduced)
            patterns.append(
                project_pattern(
                    detector,
                    rotation,
                    impose_detector_frame_fe(detector, rotation, reduced),
                )
            )
        # every point shares one projection centre, so the beam-scan
        # phantom is exactly the identity and what is measured is the
        # conversion alone
        per_point = kp.detectors.EBSDDetector(
            shape=SHAPE_480,
            binning=1,
            px_size=70.0,
            pc=np.tile(PC_480, (1, len(patterns), 1)),
            sample_tilt=70.0,
            tilt=0.0,
        )
        properties = run_hrebsd_dic(
            np.stack(patterns),
            (1, len(patterns)),
            per_point,
            reference=(0, 0),
            verbose=0,
        )
        assert np.all(properties["converged"])
        worst = 0.0
        for i, reduced in enumerate(expected, start=1):
            got = properties["Fe"][i].reshape(3, 3)
            worst = max(worst, float(np.abs(got - reduced).max()))
        # the out-of-plane case really does carry the detector
        # distance: its ``Fe23`` is ``-tan(omega)``, which a DD
        # expressed in unbinned pixels or a missing division would
        # miss by tens of per cent
        assert abs(expected[-1][1, 2]) > 1e-2
        assert_within(worst, DEFORMED_MASTER_FE_TOL, "DEFORMED_MASTER_FE_TOL")

    def test_undeformed_pattern_gives_the_identity(self):
        # the free sanity arm: a target which IS the reference must
        # come back as the identity homography
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        state = make_state(reference, pc_px)
        result = fit_pattern(state, reference.copy())
        assert result["converged"]
        # measured with the frozen D2.5 convergence norm itself, so
        # the bound is the frozen ``min_step`` rather than a guess
        assert warp_norm(result["h"], corners) < 1e-3


# ================= V4 -- pure rotations and frames ================== #


class TestPureRotations:
    """The analytic rotation cases and THE detector-frame axis and
    sign pin of requirements D1.5 and D7: a rotation imposed about a
    known SAMPLE frame axis must come back about that axis, mapped
    through ``EBSDDetector.sample_to_detector``.  [D1/D2/D7/V4]"""

    @pytest.mark.parametrize("angle_deg", [0.1, 1.0, 5.0])
    def test_in_plane_rotation(self, angle_deg):
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        theta = np.deg2rad(angle_deg)
        cos, sin = np.cos(theta), np.sin(theta)
        # a rotation about the detector NORMAL is its own reduced
        # tensor, so no polar decomposition is needed in Stage A
        fe_detector = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        rotation = generic_rotation()
        target = project_pattern(
            detector,
            rotation,
            impose_detector_frame_fe(detector, rotation, fe_detector),
        )
        expected = fe_to_homography(fe_detector, np.zeros(2), pc_px[2])
        result = fit_pattern(make_state(reference, pc_px), target)
        assert result["converged"]
        # the recovered angle, read straight off the linear block
        recovered = np.arctan2(result["h"][3], 1.0 + result["h"][0])
        assert_within(abs(recovered - theta), ROTATION_TOL_RAD, "ROTATION_TOL_RAD")
        assert_within(
            recovery_error(result["h"], expected, corners),
            DEFORMED_MASTER_H_TOL,
            "DEFORMED_MASTER_H_TOL (in-plane rotation)",
        )

    # the contrast the ratio arms below demand is derived, not
    # guessed: at ``angle_deg`` the exact conversion puts
    # ``|h23| = DD * tan(omega)`` into the tilted pair, which is
    # 0.85 px at 0.2 degrees and 4.2 px at 1.0, while the orthogonal
    # pair carries only the fit's own systematic, expected at 0.01 to
    # 0.05 px.  A factor of 2 at 0.2 degrees therefore still leaves a
    # margin of about 8 against that systematic, and a factor of 10
    # at 1.0 degrees leaves about the same
    @pytest.mark.parametrize("angle_deg,contrast", [(0.2, 2.0), (1.0, 10.0)])
    def test_out_of_plane_tilt(self, angle_deg, contrast):
        # THE frame pin: the leading terms land in h23 and h32 as the
        # exact conversion predicts, with the signs of D1.5 rather
        # than a hand built tilt matrix
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        dd = pc_px[2]
        omega = np.deg2rad(angle_deg)
        cos, sin = np.cos(omega), np.sin(omega)
        fe_detector = (
            np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]) / cos
        )
        rotation = generic_rotation()
        target = project_pattern(
            detector,
            rotation,
            impose_detector_frame_fe(detector, rotation, fe_detector),
        )
        result = fit_pattern(make_state(reference, pc_px), target)
        assert result["converged"]
        h = result["h"]
        tan = sin / cos
        # the SIGNS, which need no tolerance at all, and the exact
        # magnitudes through the frozen error-warp metric below
        assert np.sign(h[5]) == np.sign(-dd * tan)
        assert np.sign(h[7]) == np.sign(tan / dd)
        # and much more in the tilted pair than in the orthogonal
        # one, which is what a transposed frame or a swapped axis
        # would fill instead
        assert abs(h[5]) > contrast * abs(h[2])
        assert abs(h[7]) > contrast * abs(h[6])
        expected = fe_to_homography(fe_detector, np.zeros(2), dd)
        corners = subregion_corners(SHAPE_480, pc_px)
        assert_within(
            recovery_error(h, expected, corners),
            DEFORMED_MASTER_H_TOL,
            "DEFORMED_MASTER_H_TOL (out-of-plane tilt)",
        )

    def test_sample_frame_axis_is_pinned(self):
        # a rotation imposed about a known SAMPLE frame axis: the
        # detector-frame rotation the engine recovers must be about
        # that axis carried through ``sample_to_detector``, with the
        # measured y flip.  The ``R`` versus ``R**T`` mutant dies here
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        omega = np.deg2rad(0.8)
        axis_sample = np.array([0.0, 0.0, 1.0])
        chain = sample_to_spec_detector_matrix(detector)
        axis_detector = chain @ axis_sample
        skew = np.array(
            [
                [0.0, -axis_detector[2], axis_detector[1]],
                [axis_detector[2], 0.0, -axis_detector[0]],
                [-axis_detector[1], axis_detector[0], 0.0],
            ]
        )
        fe_detector = (
            np.eye(3) + np.sin(omega) * skew + (1 - np.cos(omega)) * (skew @ skew)
        )
        fe_detector = fe_detector / fe_detector[2, 2]
        rotation = generic_rotation()
        target = project_pattern(
            detector,
            rotation,
            impose_detector_frame_fe(detector, rotation, fe_detector),
        )
        result = fit_pattern(make_state(reference, pc_px), target)
        assert result["converged"]
        expected = fe_to_homography(fe_detector, np.zeros(2), pc_px[2])
        corners = subregion_corners(SHAPE_480, pc_px)
        assert_within(
            recovery_error(result["h"], expected, corners),
            DEFORMED_MASTER_H_TOL,
            "DEFORMED_MASTER_H_TOL (sample frame axis)",
        )

    @pytest.mark.weekly
    def test_rotation_sweep_capture_range(self):
        # RECORDED, not gated (validation V4): the largest angle the
        # translation-only seed still converges from bounds the
        # regime version one is valid in and sizes a future
        # Fourier-Mellin stage.  The recorded value goes into the
        # ``hrebsd_dic`` docstring Notes
        reference, detector = oracle_reference(SHAPE_480)
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_480, pc_px)
        state = make_state(reference, pc_px)
        rotation = generic_rotation()
        largest = 0.0
        for angle_deg in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            theta = np.deg2rad(angle_deg)
            cos, sin = np.cos(theta), np.sin(theta)
            fe_detector = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
            target = project_pattern(
                detector,
                rotation,
                impose_detector_frame_fe(detector, rotation, fe_detector),
            )
            result = fit_pattern(state, target)
            expected = fe_to_homography(fe_detector, np.zeros(2), pc_px[2])
            # a plain 1.0 px criterion, not an MTP band: this sweep
            # is RECORDED and never gated, and what it measures is
            # whether the translation-only seed still lands in the
            # basin at all, not how accurate the fit then is
            if (
                result["converged"]
                and recovery_error(result["h"], expected, corners) < 1.0
            ):
                largest = angle_deg
            else:
                break
        # the sweep must at least reach the smallest angle, otherwise
        # the seed is broken rather than merely limited
        assert largest >= 0.5


# ============== V6 (pattern half) -- the PC-shift phantom =========== #


class TestPcShiftPhantom:
    """Identity-deformation patterns synthesized on a per-point
    projection centre grid.  THE sign pin of requirements D6.3: the
    passing sign set is recorded there with its date.  [D6/V6]"""

    @staticmethod
    def phantom_map(
        navigation_shape=PHANTOM_NAVIGATION_SHAPE,
        step_sizes=PHANTOM_STEP_SIZES,
    ):
        """Return patterns, the per-point detector and the flat
        reference index of a strain-free map whose only variation is
        the beam-scan projection centre."""
        single = make_detector(shape=SHAPE_480)
        per_point = single.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=navigation_shape,
            step_sizes=step_sizes,
        )
        rotation = generic_rotation()
        patterns = np.stack(
            [
                project_pattern(per_point, rotation, pc_index=i)
                for i in range(int(np.prod(navigation_shape)))
            ]
        )
        return patterns, per_point, 0

    @staticmethod
    def phantom_pc_pixels(detector, n_points):
        """Return the per-point projection centres in binned pixels,
        computed here rather than through ``_geometry``: this oracle
        judges that module and may not borrow its arithmetic."""
        return np.stack([pc_pixels_of(detector, i) for i in range(n_points)])

    def test_phantom_map_moves_the_geometry_measurably(self):
        # the guard on the oracle itself, which is what the drafted
        # one-row map failed: unless the projection centre really
        # moves, the assertions below cannot separate a right sign
        # from a wrong one.  Both in-plane offsets must clear the
        # 0.01 to 0.05 px cross-interpolator systematic by orders,
        # the detector distance ratio must actually differ from one,
        # and the largest displacement must stay inside the 24 px
        # border of D4.4
        _, detector, reference_index = self.phantom_map()
        n_points = int(np.prod(PHANTOM_NAVIGATION_SHAPE))
        pc_px = self.phantom_pc_pixels(detector, n_points)
        delta = pc_px - pc_px[reference_index]
        assert np.abs(delta[:, 0]).max() > 5.0
        assert np.abs(delta[:, 1]).max() > 2.0
        assert np.abs(delta[:, 2] / pc_px[reference_index, 2]).max() > 1e-3
        corners = subregion_corners(SHAPE_480, pc_px[reference_index])
        for i in range(n_points):
            phantom = closed_form_phantom(pc_px[reference_index], pc_px[i])
            assert warp_norm(phantom, corners) < 24.0

    def test_phantom_uncorrected(self):
        # THE sign pin of D6.3.  The expectation is the D6.2 closed
        # form written out in this module (``closed_form_phantom``),
        # never ``_geometry.phantom_homography``: the oracle may not
        # share an error with what it judges
        patterns, detector, reference_index = self.phantom_map()
        properties = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            detector,
            reference=(0, 0),
            verbose=0,
            correct_pc_shift=False,
        )
        pc_px = self.phantom_pc_pixels(detector, patterns.shape[0])
        corners = subregion_corners(SHAPE_480, pc_px[reference_index])
        assert np.all(properties["converged"])
        worst = 0.0
        for i in range(1, patterns.shape[0]):
            expected = closed_form_phantom(pc_px[reference_index], pc_px[i])
            worst = max(
                worst,
                recovery_error(properties["homography"][i], expected, corners),
            )
        assert_within(worst, PC_PHANTOM_TOL, "PC_PHANTOM_TOL")

    def test_phantom_corrected(self):
        # with the correction on, a strain-free wafer of shifting
        # projection centres carries NO deformation at all
        patterns, detector, _ = self.phantom_map()
        properties = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            detector,
            reference=(0, 0),
            verbose=0,
            correct_pc_shift=True,
        )
        fe = properties["Fe"].reshape(-1, 3, 3)
        worst = float(np.abs(fe - np.eye(3)).max())
        # the contrast arm, which is what makes the band above a
        # discriminator rather than a formality: with the correction
        # OFF the same map carries a large spurious deformation, of
        # order ``alpha_s - 1`` (8e-3) and ``gamma / DD`` (5e-2).
        # This needs no tolerance, only the comparison
        uncorrected = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            detector,
            reference=(0, 0),
            verbose=0,
            correct_pc_shift=False,
        )["Fe"].reshape(-1, 3, 3)
        spurious = float(np.abs(uncorrected - np.eye(3)).max())
        assert spurious > 1e-3
        assert worst < 0.1 * spurious
        assert_within(worst, PC_PHANTOM_FE_TOL, "PC_PHANTOM_FE_TOL")

    def test_raw_homography_is_stored_uncorrected(self):
        # D15.6: the stored ``homography`` is the RAW fit, so that a
        # projection centre analysis reads the real beam-scan
        # translations; only the ``Fe`` path is corrected.  The
        # drafted arm asked only for translations above 1e-3 px,
        # which a wrongly stored CORRECTED homography passes on fit
        # noise alone; this one pins them to the closed form and adds
        # the negative arm
        patterns, detector, reference_index = self.phantom_map()
        properties = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            detector,
            reference=(0, 0),
            verbose=0,
        )
        pc_px = self.phantom_pc_pixels(detector, patterns.shape[0])
        worst = 0.0
        for i in range(1, patterns.shape[0]):
            expected = closed_form_phantom(pc_px[reference_index], pc_px[i])
            got = properties["homography"][i][[2, 5]]
            worst = max(worst, float(np.abs(got - expected[[2, 5]]).max()))
            # a CORRECTED homography would carry translations of
            # zero plus fit noise, orders below the several pixels
            # of beam-scan translation this map imposes
            assert np.abs(got).max() > 0.5 * np.abs(expected[[2, 5]]).max()
        assert_within(worst, PC_PHANTOM_TRANSLATION_TOL, "PC_PHANTOM_TRANSLATION_TOL")

    def test_single_pc_equals_per_point_pc(self):
        # the D6.1 equivalence: deriving the projection centres
        # internally from a single-PC detector gives the same answer
        # as handing the engine the explicit per-point detector built
        # from the same model
        patterns, per_point, _ = self.phantom_map()
        single = make_detector(shape=SHAPE_480)
        explicit = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            per_point,
            reference=(0, 0),
            verbose=0,
        )
        internal = run_hrebsd_dic(
            patterns,
            PHANTOM_NAVIGATION_SHAPE,
            single,
            reference=(0, 0),
            step_sizes=PHANTOM_STEP_SIZES,
            verbose=0,
        )
        assert_within(
            float(np.abs(internal["Fe"] - explicit["Fe"]).max()),
            PC_ROUTE_EQUIVALENCE_TOL,
            "PC_ROUTE_EQUIVALENCE_TOL",
        )


# ============ D11 -- Stage A reference resolution modes ============= #


class TestReferenceResolution:
    """Explicit modes only in Stage A; ``"auto"`` names Stage B.
    [D11]"""

    def test_auto_raises_naming_stage_b(self):
        # the Stage A pin, replaced when the segmentation lands.  The
        # message must be actionable: it names the argument and the
        # stage, so a user knows to pass an explicit reference
        with pytest.raises(NotImplementedError, match="Stage B"):
            resolve_reference(AUTO_REFERENCE, None, (3, 3))

    def test_tuple_reference_is_one_implicit_grain(self):
        grain_id, reference_index = resolve_reference((1, 2), None, (3, 4))
        assert grain_id.shape == (12,)
        assert reference_index.shape == (12,)
        assert grain_id.dtype == np.int32
        assert reference_index.dtype == np.int32
        assert np.array_equal(grain_id, np.zeros(12, dtype=np.int32))
        assert np.array_equal(reference_index, np.full(12, 6, dtype=np.int32))

    def test_index_array_with_grain_labels(self):
        labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1]])
        grain_id, reference_index = resolve_reference(np.array([0, 3]), labels, (3, 4))
        assert np.array_equal(grain_id, labels.ravel().astype(np.int32))
        expected = np.where(labels.ravel() == 0, 0, 3).astype(np.int32)
        assert np.array_equal(reference_index, expected)

    def test_unlabelled_points_get_minus_one(self):
        labels = np.array([[0, 0, -1], [0, -1, -1], [0, 0, 0]])
        grain_id, reference_index = resolve_reference(np.array([0]), labels, (3, 3))
        assert np.array_equal(grain_id == -1, labels.ravel() == -1)
        assert np.all(reference_index[labels.ravel() == -1] == -1)

    def test_validation(self):
        # every ValueError condition the ``resolve_reference``
        # docstring lists, in its order.  The last two were missing
        # from the drafted arm
        with pytest.raises(ValueError):
            resolve_reference("nonsense", None, (3, 3))
        with pytest.raises(ValueError):
            resolve_reference((9, 9), None, (3, 3))
        with pytest.raises(ValueError):
            resolve_reference(np.array([0, 1]), None, (3, 3))
        # one index per grain label, no more and no fewer: two
        # labels, three indices
        labels = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]])
        with pytest.raises(ValueError):
            resolve_reference(np.array([0, 2, 5]), labels, (3, 3))
        # and a grain map which is not of the navigation shape
        with pytest.raises(ValueError):
            resolve_reference(np.array([0, 2]), labels, (3, 4))


# =========== D16 -- orchestration, determinism, messages ============ #


class TestOrchestration:
    """Determinism, grain order restoration, masking and the
    information message.  [D15/D16]"""

    @staticmethod
    def small_map():
        """Return three 60 px patterns and their per-point detector,
        the cheap orchestration fixture.  The imposed warps are
        SCALED to the pattern (see ``SCALE_60``), so that no
        subregion sample leaves the 3 px border of a 60 px
        pattern."""
        detector = make_detector(shape=SHAPE_60, binning=8)
        pc_px = pc_pixels_of(detector)
        reference = project_pattern(detector, generic_rotation())
        targets = [
            warp_with_skimage(reference, h, pc_px)
            for h in random_small_homographies(
                n=2, dd=pc_px[2], seed=21, scale=SCALE_60
            )
        ]
        patterns = np.stack([reference] + targets)
        per_point = kp.detectors.EBSDDetector(
            shape=SHAPE_60,
            binning=8,
            px_size=70.0,
            pc=np.tile(PC_480, (1, 3, 1)),
            sample_tilt=70.0,
        )
        return patterns, per_point

    @staticmethod
    def two_grain_map():
        """Return a 2 by 3 map of 60 px patterns in TWO grains, each
        point carrying a DIFFERENT known homography against its own
        grain's reference.

        Grain 0 is the points ``[0, 1, 3]`` with reference 0, grain 1
        the points ``[2, 4, 5]`` with reference 2, and the two
        references are projections of two different crystal
        orientations.  Nothing else in this suite calls the engine
        with more than one grain: every other call passes a
        ``(row, col)`` reference, which requirements D11.3 defines as
        ONE implicit grain, so the D16 grain-by-grain ordering
        contract would otherwise never be exercised.
        """
        detector = make_detector(shape=SHAPE_60, binning=8)
        pc_px = pc_pixels_of(detector)
        first = project_pattern(detector, generic_rotation())
        second = project_pattern(
            detector, Rotation.from_axes_angles((3.0, -1.0, 2.0), np.deg2rad(52.0))
        )
        warps = random_small_homographies(n=4, dd=pc_px[2], seed=23, scale=SCALE_60)
        labels = np.array([[0, 0, 1], [0, 1, 1]])
        reference_indices = np.array([0, 2])
        patterns = np.stack(
            [
                first,
                warp_with_skimage(first, warps[0], pc_px),
                second,
                warp_with_skimage(first, warps[1], pc_px),
                warp_with_skimage(second, warps[2], pc_px),
                warp_with_skimage(second, warps[3], pc_px),
            ]
        )
        imposed = np.stack(
            [
                np.zeros(N_HOMOGRAPHY_PARAMETERS),
                warps[0],
                np.zeros(N_HOMOGRAPHY_PARAMETERS),
                warps[1],
                warps[2],
                warps[3],
            ]
        )
        per_point = kp.detectors.EBSDDetector(
            shape=SHAPE_60,
            binning=8,
            px_size=70.0,
            pc=np.tile(PC_480, (2, 3, 1)),
            sample_tilt=70.0,
        )
        return patterns, per_point, labels, reference_indices, imposed

    def test_multiple_grains_are_restored_to_map_order(self):
        # THE killer of the plan-2.5 ``grain order not restored``
        # mutant.  The engine MAY correlate grain by grain so that a
        # chunk shares one precomputed reference state, and D16 then
        # requires it to put the results back in map order.  Every
        # point here carries a different imposed homography, so a
        # permuted result is visible without any tolerance: each
        # point's fit must be closer to ITS OWN imposed warp than to
        # any other point's
        patterns, detector, labels, reference_indices, imposed = self.two_grain_map()
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_60, pc_px)
        properties = run_hrebsd_dic(
            patterns,
            (2, 3),
            detector,
            reference=reference_indices,
            grain_labels=labels,
            verbose=0,
        )
        assert np.array_equal(properties["grain_id"], labels.ravel().astype(np.int32))
        expected_reference = np.where(labels.ravel() == 0, 0, 2).astype(np.int32)
        assert np.array_equal(properties["reference_index"], expected_reference)
        assert np.all(properties["converged"])
        # each grain's reference correlates with itself
        for i in reference_indices:
            assert warp_norm(properties["homography"][i], corners) < 1e-3
        for i in (1, 3, 4, 5):
            own = recovery_error(properties["homography"][i], imposed[i], corners)
            others = [
                recovery_error(properties["homography"][i], imposed[j], corners)
                for j in range(patterns.shape[0])
                if j != i
            ]
            assert own < min(others), i
        # and the same run under a different chunking, which is the
        # second half of the D16 determinism contract: the drafted
        # bitwise test repeated one call with one chunk size on a
        # single-grain map and could see neither
        chunked = run_hrebsd_dic(
            patterns,
            (2, 3),
            detector,
            reference=reference_indices,
            grain_labels=labels,
            chunksize=1,
            verbose=0,
        )
        for name in STAGE_A_PROP_NAMES:
            left, right = properties[name], chunked[name]
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name

    def test_two_runs_bitwise(self):
        patterns, detector = self.small_map()
        first = run_hrebsd_dic(patterns, (1, 3), detector, reference=(0, 0), verbose=0)
        second = run_hrebsd_dic(patterns, (1, 3), detector, reference=(0, 0), verbose=0)
        assert set(first) == set(STAGE_A_PROP_NAMES)
        for name in STAGE_A_PROP_NAMES:
            left, right = first[name], second[name]
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name

    def test_property_shapes_and_dtypes(self):
        patterns, detector = self.small_map()
        properties = run_hrebsd_dic(
            patterns, (1, 3), detector, reference=(0, 0), verbose=0
        )
        assert properties["homography"].shape == (3, HOMOGRAPHY_PROP_SIZE)
        assert properties["Fe"].shape == (3, FE_PROP_SIZE)
        assert properties["homography"].dtype == np.float64
        assert properties["Fe"].dtype == np.float64
        assert properties["residual"].shape == (3,)
        assert properties["num_iterations"].dtype == np.int32
        assert properties["norm_dp"].dtype == np.float64
        assert properties["converged"].dtype == np.bool_
        assert properties["grain_id"].dtype == np.int32
        assert properties["reference_index"].dtype == np.int32

    def test_reference_point_is_its_own_identity(self):
        # the reference correlates with itself, so its homography is
        # the identity and its reference index points at itself
        patterns, detector = self.small_map()
        pc_px = pc_pixels_of(detector)
        corners = subregion_corners(SHAPE_60, pc_px)
        properties = run_hrebsd_dic(
            patterns, (1, 3), detector, reference=(0, 0), verbose=0
        )
        assert np.all(properties["reference_index"] == 0)
        # measured with the frozen convergence norm itself, so the
        # bound is the frozen ``min_step`` and not a guessed number
        assert warp_norm(properties["homography"][0], corners) < 1e-3

    def test_navigation_mask_polarity(self):
        # kikuchipy polarity: only ``False`` entries are correlated,
        # and a masked point carries NaN rather than a stale value
        patterns, detector = self.small_map()
        mask = np.array([[False, True, False]])
        properties = run_hrebsd_dic(
            patterns,
            (1, 3),
            detector,
            reference=(0, 0),
            navigation_mask=mask,
            verbose=0,
        )
        assert np.all(np.isnan(properties["homography"][1]))
        assert np.isnan(properties["residual"][1])
        assert not properties["converged"][1]
        assert np.all(np.isfinite(properties["homography"][2]))

    def test_navigation_mask_validation(self):
        patterns, detector = self.small_map()
        for bad in [
            np.ones((1, 3), dtype=int),
            np.ones((1, 3), dtype=bool),
            np.zeros((2, 3), dtype=bool),
        ]:
            with pytest.raises(ValueError):
                run_hrebsd_dic(
                    patterns,
                    (1, 3),
                    detector,
                    reference=(0, 0),
                    navigation_mask=bad,
                    verbose=0,
                )

    def test_failed_pattern_gives_nan(self):
        # a constant pattern has no variance, so it is marked failed
        # with NaN properties, following the spherical result
        # contract; it never raises and never kills the run
        patterns, detector = self.small_map()
        patterns = patterns.copy()
        patterns[2] = 7.0
        properties = run_hrebsd_dic(
            patterns, (1, 3), detector, reference=(0, 0), verbose=0
        )
        assert np.all(np.isnan(properties["homography"][2]))
        assert np.all(np.isnan(properties["Fe"][2]))
        assert not properties["converged"][2]
        assert np.all(np.isfinite(properties["homography"][1]))

    def test_non_converged_gets_nan_in_the_derived_property(self):
        # the second half of the D2.6 contract, which no drafted test
        # asserted: a non-converged point keeps its last iterate in
        # ``homography`` and is NEVER zeroed, but every DERIVED
        # property downstream is NaN.  The NaN arms of the suite all
        # sat on the FAILED-pattern path instead, so a mutant which
        # happily fills ``Fe`` for non-converged points survived
        patterns, detector = self.small_map()
        properties = run_hrebsd_dic(
            patterns,
            (1, 3),
            detector,
            reference=(0, 0),
            max_iterations=1,
            min_step=1e-12,
            verbose=0,
        )
        # the two targets, not the reference: a reference correlated
        # with itself may legitimately meet even an unreachable
        # threshold at the first iterate
        assert not properties["converged"][1:].any()
        assert np.all(properties["num_iterations"][1:] == 1)
        homography = properties["homography"][1:]
        assert np.all(np.isfinite(homography))
        assert np.any(homography != 0.0)
        assert np.all(np.isnan(properties["Fe"][1:]))

    def test_run_defaults_are_frozen(self):
        parameters = inspect.signature(run_hrebsd_dic).parameters
        expected = {
            "reference": "auto",
            "grain_labels": None,
            "misorientation_threshold": 5.0,
            "filter_cutoffs": (0.05, None),
            "window": False,
            "border": 0.05,
            "dead_band": None,
            "interpolation": "bicubic",
            "upsample_factor": 16,
            "max_iterations": 50,
            "min_step": 1e-3,
            "step_scale": 1.0,
            "navigation_mask": None,
            "chunksize": None,
            "verbose": 1,
            "correct_pc_shift": True,
        }
        for name, default in expected.items():
            assert parameters[name].default == default, name
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    def test_fit_pattern_defaults_are_frozen(self):
        parameters = inspect.signature(fit_pattern).parameters
        assert parameters["h0"].default is None
        assert parameters["upsample_factor"].default == 16
        assert parameters["max_iterations"].default == 50
        assert parameters["min_step"].default == 1e-3
        assert parameters["step_scale"].default == 1.0
        assert parameters["interpolation"].default == "bicubic"

    def test_info_message_carries_the_memory_note(self):
        # D16: the information message states the resident memory of
        # the per-grain precompute, the
        # ``SphericalIndexer.get_info_message`` precedent
        message = get_info_message(2500, SHAPE_480, 3, chunksize=64)
        assert "2500" in message
        assert "MB" in message
        assert "480" in message
        assert "64" in message

    def test_auto_reference_raises_through_the_engine(self):
        patterns, detector = self.small_map()
        with pytest.raises(NotImplementedError, match="Stage B"):
            run_hrebsd_dic(patterns, (1, 3), detector, verbose=0)


# ============= D15.7 -- the get_map_data verification =============== #


class TestGetMapData2dProp:
    """The frozen Stage A verification task of requirements D15.7.  A
    pure orix measurement which needs no ``_hrebsd`` code, so it
    passes before the implementation lands; its outcome is recorded
    in validation.md and re-measured against the orix 0.12.1 floor by
    the local oldest-matrix run.  [D15.7]"""

    @staticmethod
    def map_with_tensor_prop(navigation_shape=(3, 4)):
        xmap = one_point_xmap(navigation_shape)
        size = xmap.size
        xmap.prop["Fe"] = np.arange(size * FE_PROP_SIZE, dtype=np.float64).reshape(
            size, FE_PROP_SIZE
        )
        xmap.prop["residual"] = np.arange(size, dtype=np.float64)
        return xmap

    def test_documented_reshape_route(self):
        # the conservative default the docstrings and the tutorial
        # use, whatever ``get_map_data`` does
        ny, nx = 3, 4
        xmap = self.map_with_tensor_prop((ny, nx))
        reshaped = xmap.prop["Fe"].reshape(ny, nx, FE_PROP_SIZE)
        assert reshaped.shape == (ny, nx, FE_PROP_SIZE)
        assert np.array_equal(reshaped[1, 2], xmap.prop["Fe"][1 * nx + 2])

    def test_get_map_data_on_a_two_dimensional_prop(self):
        xmap = self.map_with_tensor_prop()
        try:
            result = xmap.get_map_data("Fe")
        except Exception as error:  # noqa: BLE001 - the pin itself
            outcome = type(error).__name__
        else:
            outcome = str(tuple(result.shape))
        assert outcome == GET_MAP_DATA_2D_PROP_OUTCOME, (
            "the D15.7 pin moved on this orix; re-measure, record the "
            "new outcome in validation.md and re-pin"
        )

    def test_get_map_data_on_a_scalar_prop(self):
        # the documented route for scalar properties, which is what
        # the docstrings mention ``get_map_data`` for
        xmap = self.map_with_tensor_prop()
        assert xmap.get_map_data("residual").shape == (3, 4)


# ==================== D18 -- the import audit ======================= #


class TestImportAudit:
    """Requirements D18, asserted rather than left to the reviewer's
    eye: ``skimage.transform`` is a TEST-ORACLE-ONLY import for this
    feature (the independent warper of validation V2 above) and no
    ``_hrebsd`` module may import it, sympy stays banned, and no new
    required dependency arrives with this feature.  A pure source
    measurement, so it passes before the implementation lands and
    keeps passing only while the rule holds.  [D18]"""

    @staticmethod
    def module_sources():
        directory = pathlib.Path(_hrebsd.__file__).parent
        paths = sorted(directory.glob("*.py"))
        assert paths, "no _hrebsd modules found"
        return [(path.name, path.read_text(encoding="utf-8")) for path in paths]

    @pytest.mark.parametrize("banned", ["skimage.transform", "sympy"])
    def test_no_banned_import(self, banned):
        for name, source in self.module_sources():
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    assert banned not in stripped, f"{name}: {stripped}"

    def test_no_new_required_dependency(self):
        # everything the engine imports is already required:
        # numpy/scipy/numba/dask, scikit-image and orix
        allowed = (
            "numpy",
            "scipy",
            "numba",
            "dask",
            "skimage",
            "orix",
            "kikuchipy",
            "warnings",
            "sys",
            "time",
            "math",
        )
        for name, source in self.module_sources():
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or (
                    stripped.startswith("from ") and " import " in stripped
                ):
                    module = stripped.split()[1].lstrip(".")
                    if not module:  # a purely relative import
                        continue
                    top = module.split(".")[0]
                    assert top in allowed, f"{name}: {stripped}"
