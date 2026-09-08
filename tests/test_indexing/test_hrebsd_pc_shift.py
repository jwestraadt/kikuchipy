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

"""Tests of ``kikuchipy.indexing._hrebsd._pc_shift``, and the Stage B
half of oracle V6.

Covers requirements D13 of ``specs/2026-09-07-hrebsd-dic/``: the
frozen dictionary of named maps, the measured translations read from
the RAW stored homography, the beam-scan model built from the
detector's per-point projection centres, the residual between them
and its miscalibration signature.

It also carries THE Stage B half of validation V6, the
double-correction killer: the corrected phantom fed through
``hrebsd_strain_stress`` must show NO strain.  The Stage A oracle
(``test_hrebsd_engine.py::TestPcShiftPhantom``) already pins that the
engine stores an ``Fe`` of the identity for a strain-free map of
moving projection centres, to 1.7e-05; what NO Stage A test can see
is whether the Stage B chain removes that same phantom a SECOND time,
because the engine's own correction is upstream of it.  So the
phantom is rebuilt here analytically -- the same ``(2, 3)`` map with
400.0 unit steps V6 pins, the same closed form -- and the properties a
correct engine would store are handed to the chain directly.  The
expectation is then EXACT rather than at the interpolation floor, and
the magnitude a second correction would inject is computed alongside
it from ``_homography``, so the band is shown to discriminate.

Written before the implementation exists: every test which calls the
module fails with ``NotImplementedError`` until the analysis lands,
then passes unchanged.  The signature and key freezes pass today.
"""

import inspect

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    homography_to_fe,
    invert,
)
from kikuchipy.indexing._hrebsd._pc_shift import (
    PC_SHIFT_KEYS,
    TRANSLATION_SLOTS,
    beam_scan_model,
    hrebsd_pc_shift,
)
from kikuchipy.indexing._hrebsd._tensors import (
    BETA_PROP_SIZE,
    hrebsd_strain_stress,
)

# ------------------------- Frozen constants ------------------------- #

# The band of a machine-precision-class identity, FROZEN: every
# expectation here is analytic, with no pattern and so no
# interpolation floor anywhere
ALGEBRA_TOL = 1e-12

# The validation V6 phantom geometry, RE-PINNED at the Stage A
# adversarial review and reused here unchanged: a ``(2, 3)`` map with
# 400.0 unit steps on the 480 px oracle detector.  ``extrapolate_pc``
# builds both the ``pcy`` and the ``pcz`` offsets from the ROW index
# alone, so a single-row map would have ``alpha_s = 1`` and
# ``gamma_y = 0`` identically and could pin neither
SHAPE_480 = (480, 480)
PC_480 = (0.4210, 0.5794, 0.5049)
PHANTOM_NAVIGATION_SHAPE = (2, 3)
PHANTOM_STEP_SIZES = (400.0, 400.0)


# ------------- The plan 3.4 mutation list, mapped ------------------- #
#
#  the D6.2 correction re-applied in the Stage B chain ....
#      TestPhantomThroughTheChain::test_the_corrected_phantom_carries
#      _no_strain


# ----------------------------- Helpers ------------------------------ #


def phantom_detector():
    """Return the per-point detector of the V6 phantom map, built
    with kikuchipy's own beam-scan model."""
    single = kp.detectors.EBSDDetector(
        shape=SHAPE_480,
        binning=1,
        px_size=70.0,
        pc=np.atleast_2d(np.asarray(PC_480, dtype=np.float64)),
        sample_tilt=70.0,
        tilt=0.0,
    )
    return single.extrapolate_pc(
        pc_indices=[0, 0],
        navigation_shape=PHANTOM_NAVIGATION_SHAPE,
        step_sizes=PHANTOM_STEP_SIZES,
    )


def pc_pixels_of(detector):
    """Return every projection centre in binned pixels, computed here
    rather than through ``_geometry``: this module judges the model
    built on it and may not borrow its arithmetic."""
    nrows, ncols = detector.shape
    pc = np.atleast_2d(np.asarray(detector.pc_flattened, dtype=np.float64))
    return np.stack([pc[:, 0] * ncols, pc[:, 1] * nrows, pc[:, 2] * nrows], axis=1)


def closed_form_phantom(pc_reference, pc_target):
    """Return the beam-scan phantom homography of requirements D6.2,
    written out here: ``gamma = PC_target - PC_reference`` in binned
    pixels exactly and ``alpha_s`` the detector distance ratio."""
    alpha_s = pc_target[2] / pc_reference[2]
    h = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    h[0] = alpha_s - 1.0
    h[4] = alpha_s - 1.0
    h[2] = pc_target[0] - pc_reference[0]
    h[5] = pc_target[1] - pc_reference[1]
    return h


def phantom_homographies(pc_px, reference_index=0):
    """Return the raw homography a strain-free engine run on the
    phantom map would store, one row per point."""
    return np.stack(
        [
            closed_form_phantom(pc_px[reference_index], pc_px[point])
            for point in range(pc_px.shape[0])
        ]
    )


def phantom_map(homography, reference_index=0, fe=None):
    """Return a crystal map carrying the Stage A properties the
    analysis reads."""
    arrays, size = create_coordinate_arrays(PHANTOM_NAVIGATION_SHAPE, (1.0, 1.0))
    arrays["rotations"] = Rotation.identity((size,))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    xmap.prop["homography"] = np.asarray(homography, dtype=np.float64).reshape(
        size, N_HOMOGRAPHY_PARAMETERS
    )
    xmap.prop["reference_index"] = np.full(size, int(reference_index), dtype=np.int32)
    xmap.prop["grain_id"] = np.zeros(size, dtype=np.int32)
    if fe is not None:
        xmap.prop["Fe"] = np.asarray(fe, dtype=np.float64).reshape(size, BETA_PROP_SIZE)
    return xmap


# =============== The oracle guard, before anything else ============= #


class TestThePhantomMoves:
    """The guard on the oracle itself, the one the drafted single-row
    V6 map failed: unless the projection centre really moves, nothing
    below can separate a right model from a wrong one.  A LIBRARY
    MEASUREMENT, so it passes today.  [D6.2/D13]"""

    def test_the_map_moves_the_geometry_measurably(self):
        pc_px = pc_pixels_of(phantom_detector())
        delta = pc_px - pc_px[0]
        assert np.abs(delta[:, 0]).max() > 5.0
        assert np.abs(delta[:, 1]).max() > 2.0
        assert np.abs(delta[:, 2] / pc_px[0, 2]).max() > 1e-3


# ================= D13 -- the frozen return contract ================ #


class TestReturnContract:
    """Seven named maps, never a crystal map.  [D13]"""

    def test_the_keys_are_frozen(self):
        assert PC_SHIFT_KEYS == (
            "translation_x",
            "translation_y",
            "translation_x_model",
            "translation_y_model",
            "scaling_model",
            "residual_x",
            "residual_y",
        )
        # the two translation slots of the eight homography
        # parameters, ``h13`` and ``h23``
        assert TRANSLATION_SLOTS == (2, 5)

    def test_the_maps_have_the_navigation_shape(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        xmap = phantom_map(phantom_homographies(pc_px))
        got = hrebsd_pc_shift(xmap, detector)
        assert isinstance(got, dict)
        assert tuple(got) == PC_SHIFT_KEYS
        for key in PC_SHIFT_KEYS:
            assert got[key].shape == PHANTOM_NAVIGATION_SHAPE, key
            assert got[key].dtype == np.float64, key

    def test_it_stores_nothing_on_the_map(self):
        detector = phantom_detector()
        xmap = phantom_map(phantom_homographies(pc_pixels_of(detector)))
        before = set(xmap.prop)
        hrebsd_pc_shift(xmap, detector)
        assert set(xmap.prop) == before

    def test_signature_is_frozen(self):
        parameters = inspect.signature(hrebsd_pc_shift).parameters
        assert list(parameters) == ["xmap", "detector"]
        for name in parameters:
            assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                name
            )
            assert parameters[name].default is inspect.Parameter.empty, name

    def test_the_public_name_is_the_module_one(self):
        assert kp.indexing.hrebsd_pc_shift is hrebsd_pc_shift


# =========== V6 -- the phantom, measured against the model ========== #


class TestPhantom:
    """A strain-free map whose only variation is the beam scan: the
    measurement and the model agree and the residual vanishes.
    [D13/V6]"""

    def test_the_measurement_is_the_raw_translation(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        homography = phantom_homographies(pc_px)
        got = hrebsd_pc_shift(phantom_map(homography), detector)
        np.testing.assert_allclose(
            got["translation_x"].ravel(),
            homography[:, 2],
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        np.testing.assert_allclose(
            got["translation_y"].ravel(),
            homography[:, 5],
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_the_model_is_the_closed_form(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        got = hrebsd_pc_shift(phantom_map(phantom_homographies(pc_px)), detector)
        expected = np.stack([closed_form_phantom(pc_px[0], pc) for pc in pc_px])
        np.testing.assert_allclose(
            got["translation_x_model"].ravel(), expected[:, 2], rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            got["translation_y_model"].ravel(), expected[:, 5], rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            got["scaling_model"].ravel(),
            pc_px[:, 2] / pc_px[0, 2],
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        # the reference point models nothing at all, which is the one
        # value a sign error cannot hide in
        assert got["scaling_model"].ravel()[0] == pytest.approx(1.0, abs=ALGEBRA_TOL)

    def test_the_residual_vanishes(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        got = hrebsd_pc_shift(phantom_map(phantom_homographies(pc_px)), detector)
        np.testing.assert_allclose(
            got["residual_x"], np.zeros(PHANTOM_NAVIGATION_SHAPE), atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            got["residual_y"], np.zeros(PHANTOM_NAVIGATION_SHAPE), atol=ALGEBRA_TOL
        )
        # against translations of several pixels, so the vanishing is
        # a statement and not an artefact of a static map
        assert np.abs(got["translation_x_model"]).max() > 5.0

    def test_a_miscalibrated_projection_centre_shows_as_a_nonzero_mean(self):
        # the documented signature of requirements D13: the residuals
        # of a miscalibrated projection centre do not average away
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        homography = phantom_homographies(pc_px)
        offset = np.array([0.37, -0.21])
        homography[:, 2] += offset[0]
        homography[:, 5] += offset[1]
        got = hrebsd_pc_shift(phantom_map(homography), detector)
        np.testing.assert_allclose(
            got["residual_x"],
            np.full(PHANTOM_NAVIGATION_SHAPE, offset[0]),
            atol=ALGEBRA_TOL,
        )
        np.testing.assert_allclose(
            got["residual_y"],
            np.full(PHANTOM_NAVIGATION_SHAPE, offset[1]),
            atol=ALGEBRA_TOL,
        )
        assert abs(float(np.nanmean(got["residual_x"]))) > 0.1

    def test_a_corrected_homography_would_be_blind(self):
        # requirements D15.6 stores the RAW fit for exactly this
        # reason.  Handed a CORRECTED homography -- zero translation,
        # which is what the beam-scan removal leaves on this map --
        # the residuals become the whole model, several pixels of it,
        # and the diagnostic would report a miscalibration on a
        # perfectly calibrated detector
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        corrected = np.zeros((pc_px.shape[0], N_HOMOGRAPHY_PARAMETERS))
        got = hrebsd_pc_shift(phantom_map(corrected), detector)
        np.testing.assert_allclose(
            got["residual_x"].ravel(),
            -np.stack([closed_form_phantom(pc_px[0], pc) for pc in pc_px])[:, 2],
            rtol=0,
            atol=ALGEBRA_TOL,
        )
        assert np.abs(got["residual_x"]).max() > 5.0

    def test_another_reference_point_moves_the_whole_model(self):
        # every translation is measured against THAT point's
        # reference, so choosing another one shifts the model and
        # leaves the residual at zero
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        homography = phantom_homographies(pc_px, reference_index=4)
        got = hrebsd_pc_shift(phantom_map(homography, reference_index=4), detector)
        np.testing.assert_allclose(
            got["residual_x"], np.zeros(PHANTOM_NAVIGATION_SHAPE), atol=ALGEBRA_TOL
        )
        assert got["scaling_model"].ravel()[4] == pytest.approx(1.0, abs=ALGEBRA_TOL)
        assert not np.isclose(got["scaling_model"].ravel()[0], 1.0)


class TestBeamScanModel:
    """The model helper on its own.  [D13/D6.2]"""

    def test_it_is_the_closed_form_per_point(self):
        pc_px = pc_pixels_of(phantom_detector())
        reference_index = np.zeros(pc_px.shape[0], dtype=np.int32)
        got = beam_scan_model(pc_px, reference_index)
        assert got.shape == (pc_px.shape[0], 3)
        assert got.dtype == np.float64
        for point in range(pc_px.shape[0]):
            expected = closed_form_phantom(pc_px[0], pc_px[point])
            assert got[point, 0] == pytest.approx(expected[2], abs=ALGEBRA_TOL)
            assert got[point, 1] == pytest.approx(expected[5], abs=ALGEBRA_TOL)
            assert got[point, 2] == pytest.approx(
                pc_px[point, 2] / pc_px[0, 2], abs=ALGEBRA_TOL
            )

    def test_a_negative_reference_index_is_nan(self):
        pc_px = pc_pixels_of(phantom_detector())
        reference_index = np.zeros(pc_px.shape[0], dtype=np.int32)
        reference_index[2] = -1
        got = beam_scan_model(pc_px, reference_index)
        assert np.all(np.isnan(got[2]))
        assert np.all(np.isfinite(got[1]))

    def test_guards(self):
        pc_px = pc_pixels_of(phantom_detector())
        with pytest.raises(ValueError):
            beam_scan_model(pc_px[:, :2], np.zeros(pc_px.shape[0], dtype=np.int32))
        with pytest.raises(ValueError):
            beam_scan_model(pc_px, np.zeros(2, dtype=np.int32))
        with pytest.raises(ValueError):
            beam_scan_model(pc_px, np.full(pc_px.shape[0], 99, dtype=np.int32))


class TestFailedPoints:
    """NaN in, NaN out: nothing is fabricated for a point which never
    converged.  [D2.6/D13]"""

    def test_a_nan_homography_gives_nan_everywhere(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        homography = phantom_homographies(pc_px)
        homography[3] = np.nan
        got = hrebsd_pc_shift(phantom_map(homography), detector)
        for key in PC_SHIFT_KEYS:
            assert np.isnan(got[key].ravel()[3]), key
            assert np.isfinite(got[key].ravel()[1]), key

    def test_a_point_without_a_reference_gives_nan(self):
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        xmap = phantom_map(phantom_homographies(pc_px))
        reference_index = np.asarray(xmap.prop["reference_index"]).copy()
        reference_index[5] = -1
        xmap.prop["reference_index"] = reference_index
        got = hrebsd_pc_shift(xmap, detector)
        for key in PC_SHIFT_KEYS:
            assert np.isnan(got[key].ravel()[5]), key


class TestGuards:
    """Every guard names what is missing.  [D13]"""

    def test_a_map_which_never_went_through_the_engine(self):
        detector = phantom_detector()
        arrays, size = create_coordinate_arrays(PHANTOM_NAVIGATION_SHAPE, (1.0, 1.0))
        arrays["rotations"] = Rotation.identity((size,))
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
        with pytest.raises(ValueError, match="hrebsd_dic"):
            hrebsd_pc_shift(CrystalMap(**arrays), detector)

    def test_a_single_projection_centre_detector_is_refused(self):
        # requirements D6.1's rule, applied here: the beam-scan model
        # needs the scan steps in the unit of ``px_size``, and this
        # function has no way to read a unit, so it refuses rather
        # than guessing -- and names the way out
        detector = kp.detectors.EBSDDetector(
            shape=SHAPE_480,
            binning=1,
            px_size=70.0,
            pc=np.atleast_2d(np.asarray(PC_480, dtype=np.float64)),
            sample_tilt=70.0,
        )
        xmap = phantom_map(phantom_homographies(pc_pixels_of(phantom_detector())))
        with pytest.raises(ValueError, match="extrapolate_pc"):
            hrebsd_pc_shift(xmap, detector)

    def test_a_detector_of_the_wrong_size(self):
        xmap = phantom_map(phantom_homographies(pc_pixels_of(phantom_detector())))
        detector = kp.detectors.EBSDDetector(
            shape=SHAPE_480,
            binning=1,
            px_size=70.0,
            pc=np.tile(PC_480, (2, 4, 1)),
            sample_tilt=70.0,
        )
        with pytest.raises(ValueError):
            hrebsd_pc_shift(xmap, detector)

    def test_two_runs_are_bitwise_identical(self):
        detector = phantom_detector()
        xmap = phantom_map(phantom_homographies(pc_pixels_of(detector)))
        first = hrebsd_pc_shift(xmap, detector)
        second = hrebsd_pc_shift(xmap, detector)
        for key in PC_SHIFT_KEYS:
            assert np.array_equal(first[key], second[key], equal_nan=True), key


# ====== V6 Stage B -- the double-correction killer, through ========= #
# ================== ``hrebsd_strain_stress`` ======================== #


class TestPhantomThroughTheChain:
    """The corrected phantom fed through the tensor chain shows NO
    strain.  [D6.2/D8/V6]"""

    def test_the_corrected_phantom_carries_no_strain(self):
        # THE plan section 3.4 mutant "the D6.2 correction re-applied
        # in the Stage B chain".  The engine corrects the beam-scan
        # phantom ONCE, before the conversion, so what it stores for a
        # strain-free map of moving projection centres is the identity
        # (Stage A pins that at 1.7e-05); a chain which removed the
        # phantom a second time would turn that identity into an
        # isotropic strain of order ``alpha_s - 1``
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        size = pc_px.shape[0]
        fe = np.tile(np.eye(3).ravel(), (size, 1))
        xmap = phantom_map(phantom_homographies(pc_px), fe=fe)
        result = hrebsd_strain_stress(xmap, detector)
        strain = np.asarray(result.prop["strain"])
        beta = np.asarray(result.prop["beta"])
        assert np.abs(strain).max() < ALGEBRA_TOL
        assert np.abs(beta).max() < ALGEBRA_TOL

    def test_the_band_above_discriminates(self):
        # what a second correction would inject, computed here from
        # the Stage A algebra: removing the phantom from an already
        # corrected homography leaves ``W_phantom**-1``, whose
        # deformation gradient is the identity scaled by ``1/alpha_s``
        # plus a translation over the detector distance.  A LIBRARY
        # MEASUREMENT of the oracle's own contrast, so it passes today
        pc_px = pc_pixels_of(phantom_detector())
        worst = 0.0
        for point in range(pc_px.shape[0]):
            phantom = closed_form_phantom(pc_px[0], pc_px[point])
            doubled = homography_to_fe(invert(phantom), np.zeros(2), pc_px[0, 2])
            worst = max(worst, float(np.abs(doubled - np.eye(3)).max()))
        # eight parts in a thousand, ten orders above the band the
        # test above asserts
        assert worst > 1e-3

    def test_a_real_deformation_still_comes_through(self):
        # the complement: the chain must not be blind to strain on the
        # same geometry, or the test above would pass on a chain which
        # returns zeros
        detector = phantom_detector()
        pc_px = pc_pixels_of(detector)
        size = pc_px.shape[0]
        fe = np.eye(3)
        fe[0, 0] += 1.5e-3
        fe[1, 1] -= 8e-4
        xmap = phantom_map(
            phantom_homographies(pc_px), fe=np.tile(fe.ravel(), (size, 1))
        )
        result = hrebsd_strain_stress(xmap, detector)
        strain = np.asarray(result.prop["strain"])
        assert np.abs(strain).max() > 1e-4
