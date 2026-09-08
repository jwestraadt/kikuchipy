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

"""Tests of ``kikuchipy.signals.EBSD.hrebsd_dic``.

Covers the signal-method suite of
``specs/2026-09-07-hrebsd-dic/validation.md``: the frozen signature
and defaults of requirements D15.4, the crystal map contract of
D15.6 (property names, shapes, data types, and the input
orientations left untouched), both navigation-mask polarities, lazy
input, the frozen default ``reference="auto"`` of D11.3, the argument
guards, the information message of D16 and the documented retrieval
route for the two dimensional tensor properties of D15.7.

The signature and docstring pins are pure introspection and passed
before the implementation landed, which is what makes them a freeze;
every other test called the skeleton and so failed with
``NotImplementedError`` until the engine landed, and passed unchanged
after it (narration corrected to the past tense 2026-09-08, Stage B
adversarial review).
"""

import inspect

import numpy as np
from orix.crystal_map import CrystalMap, create_coordinate_arrays
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._engine import (
    FE_PROP_SIZE,
    HOMOGRAPHY_PROP_SIZE,
    STAGE_A_PROP_NAMES,
)

# ------------------------- Frozen constants ------------------------- #

# The ordered parameter list of requirements D15.4, with the kind and
# the literal default of each.  ``xmap`` is an argument like the
# refinement methods take it, never ``self.xmap`` implicitly, and
# everything after ``detector`` is keyword only
FROZEN_SIGNATURE = [
    ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("xmap", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    (
        "detector",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.empty,
    ),
    ("reference", inspect.Parameter.KEYWORD_ONLY, "auto"),
    ("grain_labels", inspect.Parameter.KEYWORD_ONLY, None),
    ("misorientation_threshold", inspect.Parameter.KEYWORD_ONLY, 5.0),
    ("filter_cutoffs", inspect.Parameter.KEYWORD_ONLY, (0.05, None)),
    ("window", inspect.Parameter.KEYWORD_ONLY, False),
    ("border", inspect.Parameter.KEYWORD_ONLY, 0.05),
    ("dead_band", inspect.Parameter.KEYWORD_ONLY, None),
    ("interpolation", inspect.Parameter.KEYWORD_ONLY, "bicubic"),
    ("upsample_factor", inspect.Parameter.KEYWORD_ONLY, 16),
    ("max_iterations", inspect.Parameter.KEYWORD_ONLY, 50),
    ("min_step", inspect.Parameter.KEYWORD_ONLY, 1e-3),
    ("step_scale", inspect.Parameter.KEYWORD_ONLY, 1.0),
    ("navigation_mask", inspect.Parameter.KEYWORD_ONLY, None),
    ("chunksize", inspect.Parameter.KEYWORD_ONLY, None),
    ("verbose", inspect.Parameter.KEYWORD_ONLY, 1),
]

# The property data types of D15.6
PROP_DTYPES = {
    "homography": np.float64,
    "Fe": np.float64,
    "residual": np.float64,
    "num_iterations": np.int32,
    "norm_dp": np.float64,
    "converged": np.bool_,
    "grain_id": np.int32,
    "reference_index": np.int32,
}


# ----------------------------- Helpers ------------------------------ #


def ni_signal():
    """Return the nine shipped ``nickel_ebsd_small`` patterns with
    their static background removed, a 3 by 3 map of 60 px
    patterns."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    return signal


def ni_inputs():
    """Return ``(signal, xmap, detector)`` of the shipped Ni data,
    with a single-projection-centre detector."""
    signal = ni_signal()
    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average
    return signal, signal.xmap.deepcopy(), detector


def run(signal, xmap, detector, **kwargs):
    """Call the method with the one non-default this whole suite
    needs, an explicit reference.

    The frozen default ``"auto"`` has its own class,
    :class:`TestAutoReference`; naming a reference here keeps every
    other test measuring what it is about rather than the segmentation
    (clause corrected 2026-09-08, Stage B adversarial review: it used
    to call the default a Stage B thing, which stopped being true when
    Stage B landed).
    """
    kwargs.setdefault("reference", (1, 1))
    kwargs.setdefault("verbose", 0)
    return signal.hrebsd_dic(xmap, detector, **kwargs)


# ===================== The frozen public surface ==================== #


class TestFrozenSignature:
    """Requirements D15.4, asserted literally.  Pure introspection, so
    this passes before the implementation lands: it is a freeze, not
    a behaviour test.  [D15]"""

    def test_signature_is_frozen(self):
        parameters = inspect.signature(kp.signals.EBSD.hrebsd_dic).parameters
        assert list(parameters) == [name for name, _, _ in FROZEN_SIGNATURE]
        for name, kind, default in FROZEN_SIGNATURE:
            assert parameters[name].kind is kind, name
            assert parameters[name].default == default, name

    def test_method_name_is_the_frozen_one(self):
        # ``hrebsd`` names the technique and ``_dic`` the algorithm,
        # which leaves ``hrebsd_xcorr`` free for a future classic
        # cross-correlation variant (D15.2)
        assert hasattr(kp.signals.EBSD, "hrebsd_dic")
        assert not hasattr(kp.signals.EBSD, "hrebsd")
        assert not hasattr(kp.signals.EBSD, "refine_strain")

    def test_lazy_signal_inherits_the_method(self):
        assert kp.signals.LazyEBSD.hrebsd_dic is kp.signals.EBSD.hrebsd_dic

    def test_docstring_carries_the_documented_limitations(self):
        # the scope decisions are documented rather than silently
        # absorbed: optical distortion is out of version one, the
        # measurement is relative to a reference, and the tensor
        # properties are retrieved by reshaping
        docstring = kp.signals.EBSD.hrebsd_dic.__doc__
        assert docstring is not None
        for phrase in [
            "distortion",
            "relative",
            "reshape",
            "binned",
            "never zeroed",
        ]:
            assert phrase in docstring, phrase

    def test_docstring_lists_every_stage_a_property(self):
        docstring = kp.signals.EBSD.hrebsd_dic.__doc__
        for name in STAGE_A_PROP_NAMES:
            assert name in docstring, name


# ==================== The crystal map contract ====================== #


class TestResultContract:
    """The returned crystal map of requirements D15.6.  [D15]"""

    def test_returns_a_crystal_map_with_the_stage_a_props(self):
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector)
        assert isinstance(result, CrystalMap)
        assert result.shape == (3, 3)
        assert set(STAGE_A_PROP_NAMES) <= set(result.prop)

    def test_property_shapes_and_dtypes(self):
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector)
        size = 9
        assert result.prop["homography"].shape == (size, HOMOGRAPHY_PROP_SIZE)
        assert result.prop["Fe"].shape == (size, FE_PROP_SIZE)
        for name, dtype in PROP_DTYPES.items():
            array = np.asarray(result.prop[name])
            assert array.dtype == dtype, name
            assert array.shape[0] == size, name

    def test_orientations_and_phases_are_unchanged(self):
        # the feature never modifies orientations: the high angular
        # resolution rotation lives in the Stage B properties (D7)
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector)
        assert np.array_equal(result.rotations.data, xmap.rotations.data)
        assert result.phases.names == xmap.phases.names

    def test_tensor_props_reshape_route(self):
        # the documented retrieval route of D15.7, which the
        # docstrings and the tutorial use
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector)
        reshaped = result.prop["Fe"].reshape(3, 3, FE_PROP_SIZE)
        assert reshaped.shape == (3, 3, FE_PROP_SIZE)
        assert np.array_equal(reshaped[1, 2], result.prop["Fe"][5])

    def test_scan_unit_survives(self):
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector)
        assert result.scan_unit == xmap.scan_unit

    def test_two_runs_are_bitwise_identical(self):
        # D16 determinism through the public method, chunking
        # included
        signal, xmap, detector = ni_inputs()
        first = run(signal, xmap, detector, chunksize=2)
        second = run(signal, xmap, detector, chunksize=2)
        for name in STAGE_A_PROP_NAMES:
            left = np.asarray(first.prop[name])
            right = np.asarray(second.prop[name])
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name


# ========================= Masks and guards ========================= #


class TestNavigationMask:
    """kikuchipy polarity: only patterns equal to ``False`` are
    correlated.  Both polarities are exercised, so a mask which is
    not forwarded is distinguishable from a flipped one.  [D15.4]"""

    def test_polarity(self):
        signal, xmap, detector = ni_inputs()
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[2, 2] = True
        result = run(signal, xmap, detector, navigation_mask=mask)
        homography = np.asarray(result.prop["homography"])
        assert np.all(np.isnan(homography[0]))
        assert np.all(np.isnan(homography[8]))
        assert np.all(np.isfinite(homography[4]))
        assert not result.prop["converged"][0]

    def test_flipped_mask_masks_the_complement(self):
        signal, xmap, detector = ni_inputs()
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        flipped = ~mask
        flipped[1, 1] = False  # the reference must stay unmasked
        result = run(signal, xmap, detector, navigation_mask=flipped)
        homography = np.asarray(result.prop["homography"])
        assert np.all(np.isfinite(homography[0]))
        assert np.all(np.isnan(homography[2]))

    @pytest.mark.parametrize(
        "bad",
        [
            [[False] * 3] * 3,
            np.zeros((3, 3), dtype=int),
            np.zeros((2, 3), dtype=bool),
            np.ones((3, 3), dtype=bool),
        ],
    )
    def test_validation(self, bad):
        # the frozen order of ``spherical_indexing``: is-array, data
        # type, shape, all-``True``
        signal, xmap, detector = ni_inputs()
        with pytest.raises(ValueError):
            run(signal, xmap, detector, navigation_mask=bad)


class TestGuards:
    """Every guard names the offending argument.  [D15]"""

    # ``test_auto_reference_names_stage_b`` stood here and was DELETED
    # 2026-09-08 at the Stage B implementation gate, exactly as
    # ``TestAutoReference`` below and validation.md's Stage B
    # unit-suite block said it would be: it pinned the Stage A
    # ``NotImplementedError`` of ``reference="auto"``, which the
    # segmentation of requirements D11 replaces with real behaviour

    def test_detector_shape_mismatch(self):
        signal, xmap, _ = ni_inputs()
        detector = kp.detectors.EBSDDetector(shape=(59, 60), pc=[0.4, 0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            run(signal, xmap, detector)

    def test_crystal_map_shape_mismatch(self):
        signal, _, detector = ni_inputs()
        other = kp.data.nickel_ebsd_small().xmap.deepcopy()
        with pytest.raises(ValueError, match="shape"):
            run(signal, other[0:2, 0:2], detector)

    def test_detector_navigation_size_mismatch(self):
        signal, xmap, _ = ni_inputs()
        detector = kp.detectors.EBSDDetector(
            shape=(60, 60),
            binning=8,
            px_size=70.0,
            pc=np.tile([0.4, 0.5, 0.5], (5, 1)),
            sample_tilt=70.0,
        )
        with pytest.raises(ValueError):
            run(signal, xmap, detector)

    def test_unknown_interpolation(self):
        signal, xmap, detector = ni_inputs()
        with pytest.raises(ValueError, match="bicubic"):
            run(signal, xmap, detector, interpolation="bilinear")

    def test_reference_outside_the_map(self):
        signal, xmap, detector = ni_inputs()
        with pytest.raises(ValueError):
            run(signal, xmap, detector, reference=(5, 5))


# ====================== Lazy input and verbosity ==================== #


class TestLazyAndVerbose:
    """Lazy signals give an eager crystal map, and the information
    message carries the D16 memory note.  [D15/D16]"""

    def test_lazy_input_gives_an_eager_crystal_map(self):
        signal, xmap, detector = ni_inputs()
        lazy = signal.as_lazy()
        result = run(lazy, xmap, detector)
        assert isinstance(result, CrystalMap)
        assert isinstance(np.asarray(result.prop["residual"]), np.ndarray)

    def test_lazy_and_eager_agree_bitwise(self):
        signal, xmap, detector = ni_inputs()
        eager = run(signal, xmap, detector)
        lazy = run(signal.as_lazy(), xmap, detector)
        for name in STAGE_A_PROP_NAMES:
            left = np.asarray(eager.prop[name])
            right = np.asarray(lazy.prop[name])
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name

    def test_verbose_prints_the_information_message(self, capsys):
        signal, xmap, detector = ni_inputs()
        run(signal, xmap, detector, verbose=1)
        captured = capsys.readouterr()
        assert "MB" in captured.out
        assert "60" in captured.out

    def test_verbose_zero_is_silent(self, capsys):
        signal, xmap, detector = ni_inputs()
        run(signal, xmap, detector, verbose=0)
        captured = capsys.readouterr()
        assert captured.out == ""


# ========================= Failure semantics ======================== #


class TestFailureSemantics:
    """Non-converged and failed points keep NaN properties and are
    never zeroed, a recorded deviation from
    ``mod_HREBSDDIC.f90:868-870``.  [D2.6]"""

    def test_failed_pattern_gets_nan(self):
        signal, xmap, detector = ni_inputs()
        signal = signal.deepcopy()
        signal.data[0, 0] = 7  # a constant pattern has no variance
        result = run(signal, xmap, detector)
        assert np.all(np.isnan(np.asarray(result.prop["homography"])[0]))
        assert np.all(np.isnan(np.asarray(result.prop["Fe"])[0]))
        assert not result.prop["converged"][0]
        # and one bad pattern never kills the run
        assert np.all(np.isfinite(np.asarray(result.prop["homography"])[4]))

    def test_non_converged_keeps_its_last_iterate(self):
        signal, xmap, detector = ni_inputs()
        result = run(signal, xmap, detector, max_iterations=1, min_step=1e-12)
        homography = np.asarray(result.prop["homography"])
        converged = np.asarray(result.prop["converged"])
        assert not converged.any()
        # never zeroed: the last iterate survives
        assert np.any(homography[np.isfinite(homography)] != 0.0)
        assert np.all(np.asarray(result.prop["num_iterations"]) == 1)
        # and the OTHER half of the D2.6 contract, which no drafted
        # test asserted: a non-converged point gets NaN in every
        # DERIVED property downstream, which in Stage A is ``Fe``.
        # The suite's NaN arms all sat on the failed-pattern path
        # instead, so a mutant which fills ``Fe`` for non-converged
        # points survived them all
        assert np.all(np.isnan(np.asarray(result.prop["Fe"])))

    def test_warns_when_a_pattern_fails(self):
        signal, xmap, detector = ni_inputs()
        signal = signal.deepcopy()
        signal.data[0, 0] = 7
        with pytest.warns(UserWarning):
            run(signal, xmap, detector)


# ================= Navigation dimensions and units ================== #


class TestNavigationDimensions:
    """The documented one-dimensional support and the navigation
    dimension guard, neither of which the drafted suite executed
    through the public method.  [D15.4]"""

    @staticmethod
    def line_map(xmap):
        """Return a genuinely ONE dimensional crystal map of the first
        three points, which slicing a 2-D map cannot give."""
        arrays, size = create_coordinate_arrays((3,), (1.5,))
        arrays["rotations"] = xmap.rotations[:3]
        arrays["phase_id"] = np.zeros(size, dtype=int)
        arrays["phase_list"] = xmap.phases.deepcopy()
        line = CrystalMap(**arrays)
        line.scan_unit = xmap.scan_unit
        return line

    def test_one_dimensional_navigation(self):
        # a 1-D scan is a single map row to the engine, ``(1, nx)``
        signal, xmap, detector = ni_inputs()
        line = signal.inav[:, 0]
        line_map = self.line_map(xmap)
        result = line.hrebsd_dic(line_map, detector, reference=(0, 1), verbose=0)
        assert isinstance(result, CrystalMap)
        assert result.shape == (3,)
        for name in STAGE_A_PROP_NAMES:
            assert np.asarray(result.prop[name]).shape[0] == 3
        assert result.prop["homography"].shape == (3, HOMOGRAPHY_PROP_SIZE)
        # the reference is the point it was asked for, which is only
        # right if the engine really saw ``(1, 3)``
        assert np.all(np.asarray(result.prop["reference_index"]) == 1)

    def test_zero_dimensional_navigation_raises(self):
        signal, xmap, detector = ni_inputs()
        single = signal.inav[0, 0]
        with pytest.raises(ValueError, match="one or two dimensions"):
            single.hrebsd_dic(xmap[0, 0].deepcopy(), detector, verbose=0)

    @staticmethod
    def column_map(xmap):
        """Return a genuinely one dimensional crystal map whose three
        points lie in a COLUMN, which orix reports with the same shape
        ``(3,)`` as a row and tells apart only by its grids."""
        line = CrystalMap(
            rotations=xmap.rotations[:3],
            phase_id=np.zeros(3, dtype=int),
            phase_list=xmap.phases.deepcopy(),
            x=np.zeros(3),
            y=np.arange(3.0) * 1.5,
        )
        line.scan_unit = xmap.scan_unit
        return line

    def test_a_one_dimensional_column_scan(self):
        # ADDED 2026-09-08 at the Stage B adversarial review.  The
        # engine used to be told a one dimensional scan is a single
        # map ROW whatever the crystal map said, so a column line scan
        # passed the shape guard and then died inside
        # ``resolve_reference`` on the FROZEN DEFAULT
        # ``reference="auto"``, with a message naming a function the
        # caller never called.  Requirements D11.1(b) settles it: the
        # grids decide which of the two a flattened map is
        signal, xmap, detector = ni_inputs()
        line = signal.inav[0, :]
        column_map = self.column_map(xmap)
        assert column_map.shape == (3,)
        np.testing.assert_array_equal(column_map.col, [0, 0, 0])
        result = line.hrebsd_dic(column_map, detector, verbose=0)
        assert result.shape == (3,)
        for name in STAGE_A_PROP_NAMES:
            assert np.asarray(result.prop[name]).shape[0] == 3
        # every point is measured, and the reference is one of them
        assert np.all(np.asarray(result.prop["grain_id"]) >= 0)
        assert np.all(np.asarray(result.prop["reference_index"]) >= 0)

    def test_a_one_dimensional_map_which_is_neither_is_already_refused(self):
        # the reason no separate guard is needed for the grids: a map
        # whose points walk diagonally spans a two dimensional grid,
        # which orix reports as a two dimensional SHAPE, and the shape
        # check above rejects it before any of this
        signal, xmap, detector = ni_inputs()
        line = signal.inav[0, :]
        diagonal = CrystalMap(
            rotations=xmap.rotations[:3],
            phase_id=np.zeros(3, dtype=int),
            phase_list=xmap.phases.deepcopy(),
            x=np.arange(3.0) * 1.5,
            y=np.arange(3.0) * 1.5,
        )
        assert diagonal.shape == (3, 3)
        with pytest.raises(ValueError, match="must be identical"):
            line.hrebsd_dic(diagonal, detector, verbose=0)


class TestScanStepUnits:
    """The beam-scan projection centre model consumes the navigation
    axes' ``scale`` beside the detector's ``px_size``, so the two must
    share a unit.  The axes' ``units`` are READ and converted, and an
    axis whose unit cannot be read raises, exactly as requirements
    D14.5 pins for ``CrystalMap.scan_unit``.  Nothing in the drafted
    suite touched this path.  [D6.1/D14.5]"""

    @staticmethod
    def with_units(unit, scale):
        signal, xmap, detector = ni_inputs()
        signal = signal.deepcopy()
        for axis in signal.axes_manager.navigation_axes:
            axis.units = unit
            axis.scale = scale
        return signal, xmap, detector

    def test_the_shipped_micrometre_scan_is_unchanged(self):
        signal, xmap, detector = ni_inputs()
        units = {str(a.units) for a in signal.axes_manager.navigation_axes}
        assert units == {"um"}
        result = run(signal, xmap, detector)
        assert np.all(np.isfinite(np.asarray(result.prop["Fe"])[4]))

    def test_a_nanometre_scan_converts_to_the_micrometre_result(self):
        # the same physical scan described two ways must give the same
        # projection centre map and therefore the same tensors
        as_um = run(*self.with_units("um", 1.5))
        as_nm = run(*self.with_units("nm", 1500.0))
        for name in ("homography", "Fe"):
            left = np.asarray(as_um.prop[name])
            right = np.asarray(as_nm.prop[name])
            assert np.array_equal(left, right, equal_nan=True), name

    def test_an_unreadable_unit_raises_naming_the_axis(self):
        signal, xmap, detector = self.with_units("px", 1.0)
        with pytest.raises(ValueError, match="navigation axis 0"):
            run(signal, xmap, detector)

    def test_a_per_point_detector_needs_no_unit(self):
        # the steps are not used at all when the detector already
        # carries one projection centre per map point, so refusing
        # there would be gratuitous
        signal, xmap, _ = self.with_units("px", 1.0)
        detector = signal.detector.deepcopy()
        detector.pc = np.tile(detector.pc_average, (3, 3, 1))
        result = run(signal, xmap, detector)
        assert np.all(np.isfinite(np.asarray(result.prop["homography"])[4]))


class TestGrainLabels:
    """``grain_labels`` never reached the engine through the public
    method before 2026-09-07, so the reshape at the signal layer and
    both D11.3 combinations were untested here.  [D11.3]"""

    @staticmethod
    def labels():
        return np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.int32)

    def test_labels_reach_the_engine_with_a_tuple_reference(self):
        signal, xmap, detector = ni_inputs()
        labels = self.labels()
        result = run(signal, xmap, detector, grain_labels=labels)
        assert np.array_equal(np.asarray(result.prop["grain_id"]), labels.ravel())
        # one global reference serves every label (D11.3)
        assert np.all(np.asarray(result.prop["reference_index"]) == 4)

    def test_labels_with_a_per_grain_index_array(self):
        signal, xmap, detector = ni_inputs()
        labels = self.labels()
        result = signal.hrebsd_dic(
            xmap,
            detector,
            reference=np.array([0, 2]),
            grain_labels=labels,
            verbose=0,
        )
        expected = np.where(labels.ravel() == 0, 0, 2)
        assert np.array_equal(np.asarray(result.prop["reference_index"]), expected)
        assert np.array_equal(np.asarray(result.prop["grain_id"]), labels.ravel())

    def test_an_index_outside_its_grain_raises(self):
        signal, xmap, detector = ni_inputs()
        with pytest.raises(ValueError, match="grain label"):
            signal.hrebsd_dic(
                xmap,
                detector,
                reference=np.array([2, 0]),
                grain_labels=self.labels(),
                verbose=0,
            )


# ============ D11.1-D11.3 -- the ``reference="auto"`` mode ========== #


class TestAutoReference:
    """``reference="auto"`` through the public method (Stage B).

    ADDED at the Stage B failing-tests gate.  ``"auto"`` is the
    DEFAULT of the frozen signature, so until Stage B every call of
    this method without an explicit reference raised, and the whole
    suite above passes one.  These tests exercise the default itself:
    the map is segmented into grains (requirements D11.1) unless a
    grain map is supplied, and each grain's reference is its highest
    image quality pattern (D11.2), which is then paired exactly as an
    explicit per-grain index array would be (D11.3).

    ``TestGuards::test_auto_reference_names_stage_b`` was the Stage A
    pin of this mode and is REPLACED by this class; the
    implementation gate DELETED it on 2026-09-08 (validation.md:
    "``reference='auto'`` NotImplementedError pin in Stage A,
    replaced in Stage B").

    The expectations are built here from
    :func:`kikuchipy.pattern.get_image_quality` on the raw patterns,
    the kernel D11.2 freezes the selection on.  [D11]
    """

    @staticmethod
    def quality(signal):
        """Return the image quality of every pattern of the map, in
        map order, computed with kikuchipy's own public kernel."""
        patterns = np.asarray(signal.data, dtype=np.float64).reshape(9, 60, 60)
        return np.array([kp.pattern.get_image_quality(pattern) for pattern in patterns])

    @staticmethod
    def per_grain_references(signal, labels):
        """Return the argmax-quality flat index of every grain and the
        per-point reference index those pair to, assembled here."""
        quality = TestAutoReference.quality(signal)
        flat = np.asarray(labels).ravel()
        indices = np.array(
            [
                np.flatnonzero(flat == label)[np.argmax(quality[flat == label])]
                for label in np.unique(flat[flat >= 0])
            ]
        )
        per_point = np.full(flat.size, -1, dtype=np.int32)
        for position, label in enumerate(np.unique(flat[flat >= 0])):
            per_point[flat == label] = indices[position]
        return indices, per_point

    def test_auto_is_the_default_and_segments_the_map(self):
        # CORRECTED 2026-09-08 at the Stage B implementation gate.
        # The drafted arm was
        # ``test_auto_is_the_default_and_gives_one_grain_here`` and
        # asserted ONE grain, "by any sensible threshold".  MEASURED
        # on the shipped map and REFUTED: the symmetry-reduced
        # misorientation between column 0 and column 1 of
        # ``nickel_ebsd_small``'s crystal map is 34.1111 degrees, and
        # every within-column pair sits at 0.16 to 0.92 degrees.  The
        # number is measured three independent ways which agree to
        # 1e-4 degrees -- orix's ``Orientation.angle_with``, a
        # hand-built minimisation of ``angle(g_i^T S g_j)`` over the
        # 24 proper cubic rotations, and a quaternion dot
        # maximisation over ``Oh`` -- so the shipped map holds TWO
        # grains at the FROZEN 5.0 degree default of requirements
        # D11.1, and one grain here could only come from ignoring
        # that default.  What the arm is FOR is unchanged and is all
        # asserted below: ``"auto"`` is the default, every point gets
        # a grain, each grain's reference is its own highest image
        # quality pattern, and a reference correlates with itself
        signal, xmap, detector = ni_inputs()
        result = signal.hrebsd_dic(xmap, detector, verbose=0)
        assert isinstance(result, CrystalMap)
        labels = np.asarray(kp.indexing.segment_grains(xmap))
        grain_id = np.asarray(result.prop["grain_id"])
        assert np.array_equal(grain_id, labels.ravel())
        # the measured segmentation of this map, written out so that a
        # change in the shipped data fails here and names itself
        assert np.array_equal(labels, [[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        indices, expected = self.per_grain_references(signal, labels)
        assert np.array_equal(np.asarray(result.prop["reference_index"]), expected)
        # and every reference correlates with itself, so its own
        # deformation gradient is the identity
        for best in indices.tolist():
            fe = np.asarray(result.prop["Fe"])[int(best)].reshape(3, 3)
            assert np.abs(fe - np.eye(3)).max() < 1e-6

    def test_auto_agrees_with_the_equivalent_explicit_reference(self):
        # "auto" chooses the grains and their references and changes
        # nothing else, so the run must be BITWISE the run which names
        # them.  CORRECTED 2026-09-08 with the arm above: the
        # equivalent explicit run is the per-grain index array of
        # requirements D11.3 paired with the segmentation's own grain
        # map, not one global ``(row, col)``, since the shipped map
        # holds two grains
        signal, xmap, detector = ni_inputs()
        labels = np.asarray(kp.indexing.segment_grains(xmap))
        indices, _ = self.per_grain_references(signal, labels)
        automatic = signal.hrebsd_dic(xmap, detector, verbose=0)
        explicit = signal.hrebsd_dic(
            xmap, detector, reference=indices, grain_labels=labels, verbose=0
        )
        for name in STAGE_A_PROP_NAMES:
            left = np.asarray(automatic.prop[name])
            right = np.asarray(explicit.prop[name])
            assert np.array_equal(left, right, equal_nan=True), name

    def test_auto_with_a_grain_map_selects_one_reference_per_grain(self):
        # requirements D11.3: with *grain_labels* given, "auto" skips
        # the segmentation and selects inside the supplied grains --
        # which is what separates it from the ``(row, col)`` mode,
        # where ONE global reference serves every label
        signal, xmap, detector = ni_inputs()
        labels = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int32)
        result = signal.hrebsd_dic(xmap, detector, grain_labels=labels, verbose=0)
        quality = self.quality(signal)
        expected = np.where(
            labels.ravel() == 0,
            int(np.argmax(np.where(labels.ravel() == 0, quality, -np.inf))),
            int(np.argmax(np.where(labels.ravel() == 1, quality, -np.inf))),
        )
        assert np.array_equal(np.asarray(result.prop["grain_id"]), labels.ravel())
        assert np.array_equal(np.asarray(result.prop["reference_index"]), expected)
        # two grains really do get two different references
        assert len(set(expected.tolist())) == 2

    def test_the_misorientation_threshold_now_reaches_the_segmentation(self):
        # in Stage A this argument was threaded through and read by
        # nothing.  Below the map's own point-to-point misorientation
        # every point becomes its own grain, and every point is then
        # its own reference
        signal, xmap, detector = ni_inputs()
        result = signal.hrebsd_dic(
            xmap, detector, misorientation_threshold=1e-6, verbose=0
        )
        grain_id = np.asarray(result.prop["grain_id"])
        assert np.unique(grain_id).size > 1
        assert np.array_equal(np.asarray(result.prop["reference_index"]), np.arange(9))

    def test_the_documented_behaviour_is_the_real_one(self):
        # the docstring promised a later release for this mode and
        # said the threshold has NO effect; both statements have to go
        # when the behaviour arrives, and a docstring which still
        # carries them is a documentation defect this gate catches
        signal, xmap, detector = ni_inputs()
        result = signal.hrebsd_dic(xmap, detector, verbose=0)
        assert np.all(np.asarray(result.prop["grain_id"]) >= 0)
        docstring = kp.signals.EBSD.hrebsd_dic.__doc__
        assert "NO effect" not in docstring
        assert "not implemented yet" not in docstring
        assert "segment" in docstring

    def test_auto_is_deterministic(self):
        signal, xmap, detector = ni_inputs()
        first = signal.hrebsd_dic(xmap, detector, verbose=0)
        second = signal.hrebsd_dic(xmap, detector, verbose=0)
        for name in STAGE_A_PROP_NAMES:
            assert np.array_equal(
                np.asarray(first.prop[name]),
                np.asarray(second.prop[name]),
                equal_nan=True,
            ), name

    def test_auto_survives_a_navigation_mask(self):
        # a masked point takes no part in the run, and its grain is
        # still segmented and still measured: masking is a correlation
        # decision, not a segmentation one.
        #
        # COMMENT CORRECTED 2026-09-08 (Stage B adversarial review).
        # It used to add "and the reference is still chosen over the
        # whole grain", which was true of the code and WRONG: a masked
        # out pattern could become the reference every strain in its
        # grain was measured against, with its own homography NaN. The
        # assertions here are unchanged; the new behaviour is pinned by
        # ``test_a_masked_out_pattern_is_never_the_reference`` below
        signal, xmap, detector = ni_inputs()
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        result = signal.hrebsd_dic(xmap, detector, navigation_mask=mask, verbose=0)
        assert np.all(np.isnan(np.asarray(result.prop["homography"])[0]))
        assert np.all(np.isfinite(np.asarray(result.prop["homography"])[4]))

    def test_a_masked_out_pattern_is_never_the_reference(self):
        # ADDED 2026-09-08 at the Stage B adversarial review, which
        # MEASURED the defect on this very map: with the best pattern
        # of a grain masked out, ``"auto"`` still chose it, so every
        # point of that grain was measured against a pattern the caller
        # had excluded -- and that pattern's own ``homography`` came
        # back NaN with ``converged=False``
        signal, xmap, detector = ni_inputs()
        labels = np.asarray(kp.indexing.segment_grains(xmap))
        _, unmasked = self.per_grain_references(signal, labels)
        best = int(unmasked[4])
        mask = np.zeros((3, 3), dtype=bool)
        mask.ravel()[best] = True
        result = signal.hrebsd_dic(xmap, detector, navigation_mask=mask, verbose=0)
        reference_index = np.asarray(result.prop["reference_index"])
        assert np.all(reference_index != best)
        # it is the second best of the SAME grain, computed here
        quality = self.quality(signal)
        inside = np.flatnonzero(labels.ravel() == labels.ravel()[best])
        allowed = inside[inside != best]
        assert reference_index[4] == allowed[int(np.argmax(quality[allowed]))]
        # the other grain, which the mask does not touch, keeps its own
        assert reference_index[0] == unmasked[0]
        # and the grain identifiers are the unmasked ones: only the
        # choice of reference changed
        assert np.array_equal(np.asarray(result.prop["grain_id"]), labels.ravel())
        # the reference now really is a fitted point
        assert np.all(np.isfinite(np.asarray(result.prop["Fe"])[reference_index[4]]))
