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

ADDED 2026-09-08 for the Si-indent application (plan open question
13): :class:`TestRectangularEndToEnd`, the first NON-SQUARE pattern
shape through the public method and through the four analysis
functions after it.  Every end-to-end call above uses the 60 by 60
shipped nickel patterns, so before that class the whole signal layer
was square-only and the rectangular pins lived at unit level alone
(``(37, 61)`` in ``test_hrebsd_preprocessing``-style arms of
``test_hrebsd_engine.py``, ``test_hrebsd_interpolation.py`` and
``test_hrebsd_geometry.py``).  The Si-indent data set is 512 rows by
622 columns, so the whole application runs on a shape this suite had
never exercised end to end.
"""

import functools
import inspect

import numpy as np
from orix.crystal_map import (
    CrystalMap,
    Phase,
    PhaseList,
    create_coordinate_arrays,
)
from orix.quaternion import Rotation
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


# ========= The rectangular end-to-end regression (Si-indent) ======== #
#
# ADDED 2026-09-08, plan open question 13 step 2(a).  See the module
# docstring: this is the FIRST non-square pattern shape through
# ``EBSD.hrebsd_dic`` and through the four analysis functions.

# The pattern shape, deliberately NON-SQUARE and deliberately WIDER
# than tall, as every Oxford detector is and as the Si-indent file's
# 512 by 622 is.  61 rows by 101 columns is small enough to fit six
# patterns in a few milliseconds and big enough that the frozen
# ``border=0.05`` leaves 3 rows and 5 columns of margin, so no
# subregion sample of the warps below reaches the pattern edge and
# the fit measures the engine rather than the boundary policy
RECT_SHAPE = (61, 101)
RECT_NAVIGATION_SHAPE = (2, 3)
RECT_SIZE = RECT_NAVIGATION_SHAPE[0] * RECT_NAVIGATION_SHAPE[1]

# A projection centre away from the pattern centre in BOTH axes, so
# that the PC-centred frame is distinguishable from a pattern-centred
# one, and away from 0.5 so that ``pcx * ncols`` and ``pcx * nrows``
# are far apart (the whole point of this class)
RECT_PC = (0.4210, 0.5794, 0.5049)
RECT_BINNING = 8
RECT_STEP_UM = 0.2

# The reference, named explicitly.  ``reference="auto"`` has its own
# class above; here a fixed ``(row, col)`` keeps every number below
# about the rectangular shape.  It is NOT ``(0, 0)``: the beam-scan
# anchor of ``_geometry.per_point_pc_pixels`` is the first grain
# reference, so an anchor at the map origin would be indistinguishable
# from an ignored one
RECT_REFERENCE = (0, 1)
RECT_REFERENCE_INDEX = RECT_REFERENCE[0] * RECT_NAVIGATION_SHAPE[1] + RECT_REFERENCE[1]

# Where the texture is cut out of the shipped 401 by 401 stereographic
# nickel master pattern.  OFF CENTRE on purpose: the master is
# four-fold symmetric about its centre, and a centred crop would carry
# a texture which is itself nearly invariant under a row/column swap,
# which is precisely the invariance this class must not have.  Both
# corners sit 95 px or less from the centre of a disc of radius about
# 200, so the crop holds no part of the black outside
RECT_CROP = (150, 120)

# The imposed homographies, one per map point, in the D1.3 PC-centred
# binned-pixel frame of the reference.  Point
# ``RECT_REFERENCE_INDEX`` is the identity because it IS the
# reference.  Every other point carries a DIFFERENT warp, so a result
# permuted by the D16 grain-by-grain ordering is visible without any
# tolerance.  Sizes: linear parts at the 1e-2 scale, which is ten
# times the HREBSD regime and is chosen so the origin-conjugation
# below is far above the fit floor; translations at 1 px, which the
# 3-row/5-column margin absorbs.  Index 3 also carries the two
# perspective parameters, so all eight degrees of freedom are moved
RECT_WARPS = np.array(
    [
        [8.0e-3, 5.0e-3, -0.62, -5.0e-3, 6.0e-3, 0.55, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.5e-2, -1.2e-2, 1.00, 1.2e-2, -1.0e-2, -0.80, 0.0, 0.0],
        [-6.0e-3, 2.0e-3, 0.44, 3.0e-3, 9.0e-3, 0.70, 3.0e-5, -2.0e-5],
        [4.0e-3, -7.0e-3, -0.91, 7.0e-3, -3.0e-3, 0.35, 0.0, 0.0],
        [-1.0e-2, 8.0e-4, 0.77, -9.0e-4, 1.1e-2, -0.62, 0.0, 0.0],
    ]
)

# THE asymmetric case of pin (v).  Its linear block is anisotropic
# (``h11 = +1.5e-2`` against ``h22 = -1.0e-2``) and its off-diagonal
# pair is antisymmetric, so ``(A - I) d`` has a large component along
# BOTH axes and no coincidental cancellation is possible
RECT_ASYMMETRIC = 2

# Per-point projection centre offsets added to RECT_PC, of the same
# 1e-3-fraction class as the Si-indent file's own affine PC
# calibration (its PCx spans about 1.3 px).  They make the D6.2
# beam-scan phantom NON-TRIVIAL, which is what puts the ``gamma =
# PC_target - PC_reference`` conversion -- ``pcx * ncols`` against
# ``pcy * nrows`` -- on the ``Fe`` path of a rectangular detector
RECT_PC_OFFSETS = np.array(
    [
        [[0.0, 0.0, 0.0], [2.0e-3, -1.5e-3, 8.0e-4], [4.0e-3, -3.0e-3, 1.6e-3]],
        [[1.0e-3, 2.5e-3, -6.0e-4], [3.0e-3, 1.0e-3, 0.0], [5.0e-3, -5.0e-4, 2.4e-3]],
    ]
)

# MEASURED-THEN-PINNED [D2/D6, plan OQ13 step 2(a)], the corner
# displacement of the error warp ``W(h_true)**-1 . W(h_fit)`` over the
# subregion corners, in binned pixels, worst over the five warped
# points.  MEASURED 2026-09-08 on this fixture: 0.043956 px, the same
# regime as ``test_hrebsd_engine.py``'s 60 px arm (0.06507 px measured,
# ``WARP_REFIT_TOL_60 = 0.13``) and about four times its 480 px arm, as
# a pattern this small should be.  PINNED at about 2x the measurement,
# the convention of the engine module's bands
RECT_WARP_TOL = 0.09

# MEASURED-THEN-PINNED [D6/D15.6], the worst absolute entry of
# ``Fe_recovered - Fe_expected``, where the expectation is assembled in
# this module from the imposed homography and the D6.2 phantom.
# MEASURED 2026-09-08: 7.1949e-04, worst over the six points.  PINNED
# at about 2x
RECT_FE_TOL = 1.5e-3


# ----------------- Independent rectangular helpers ------------------ #
#
# Assembled here in plain numpy rather than imported from
# ``_hrebsd``: an oracle which shares an axis-order error with the
# module it judges cannot see that error.  The twins of these live in
# ``test_hrebsd_engine.py``; the two files are deliberately not
# coupled, since ``tests/`` is not an importable package.


def rect_matrix_of(h):
    """Return the ``(3, 3)`` shape function of the eight parameters."""
    h = np.asarray(h, dtype=np.float64)
    return np.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )


def rect_parameters_of(matrix):
    """Return the eight parameters of a shape function, renormalized
    by ``W[2, 2]`` as requirements D2.3 demands."""
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


def rect_corner_norm(matrix, corners):
    """Return the D2.5 norm of a warp: the largest displacement it
    induces over the four subregion corners, in binned pixels."""
    x, y = corners[:, 0], corners[:, 1]
    s = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    warped_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / s
    warped_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / s
    return float(np.hypot(warped_x - x, warped_y - y).max())


def rect_recovery_error(h_fit, h_true, corners):
    """Return the corner norm of the error warp
    ``W(h_true)**-1 . W(h_fit)``, the V2 recovery metric."""
    matrix = np.linalg.inv(rect_matrix_of(h_true)) @ rect_matrix_of(h_fit)
    return rect_corner_norm(matrix, corners)


def rect_pc_pixels(pc, shape, transpose=False):
    """Return ``(PCx_px, PCy_px, DD_px)`` of every projection centre.

    Requirements D1.2, written out here: ``PCx`` is a fraction of the
    number of COLUMNS while ``PCy`` and the detector distance are
    fractions of the number of ROWS.  With *transpose* the two are
    swapped, which is the single mutant this class exists to kill and
    which on a square detector is the identity.
    """
    pc = np.asarray(pc, dtype=np.float64).reshape(-1, 3)
    nrows, ncols = shape
    if transpose:
        nrows, ncols = ncols, nrows
    return np.column_stack([pc[:, 0] * ncols, pc[:, 1] * nrows, pc[:, 2] * nrows])


def rect_subregion_corners(shape, pc_px, border=0.05):
    """Return the ``(4, 2)`` PC-centred corners of the D4.4
    subregion, the support the recovery norm is measured over."""
    nrows, ncols = shape
    margin_row = int(round(border * nrows))
    margin_col = int(round(border * ncols))
    x0 = margin_col + 0.5 - pc_px[0]
    x1 = ncols - 1 - margin_col + 0.5 - pc_px[0]
    y0 = margin_row + 0.5 - pc_px[1]
    y1 = nrows - 1 - margin_row + 0.5 - pc_px[1]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def rect_warp_with_skimage(image, h, pc_px):
    """Return *image* warped by ``W(h)`` with an INDEPENDENT warper.

    ``skimage.transform`` is a test-oracle-only import for this
    feature (requirements D18): no ``_hrebsd`` module may import it.
    The homography acts on PC-centred coordinates, so it is conjugated
    into the array frame first, with the same half-pixel origin
    ``PC_px - 0.5`` the engine's own frame was MEASURED to use.
    """
    from skimage.transform import ProjectiveTransform, warp

    image = np.array(image, dtype=np.float64)
    translation = np.eye(3)
    translation[0, 2] = pc_px[0] - 0.5
    translation[1, 2] = pc_px[1] - 0.5
    array_matrix = translation @ rect_matrix_of(h) @ np.linalg.inv(translation)
    transform = ProjectiveTransform(matrix=np.linalg.inv(array_matrix))
    return warp(image, transform, order=3, mode="reflect", preserve_range=True)


def rect_expected_fe(h, pc_reference, pc_target):
    """Return the reduced detector-frame ``Fe`` of a raw fitted *h*,
    assembled here from requirements D6.2 and D6.

    Removes the beam-scan phantom
    ``W_phantom = [[a, 0, gx], [0, a, gy], [0, 0, 1]]`` with
    ``a = DD_target / DD_reference`` and ``gamma = PC_target -
    PC_reference``, composed as ``W_phantom**-1 . W``, then converts
    the corrected homography at ``pc_rel = (0, 0)`` and
    ``dd = DD_reference``, which is where the corrected one lives.
    Both halves read ``DD_px = pcz * nrows``, so a row/column swap
    anywhere in the conversion moves this expectation.
    """
    alpha = pc_target[2] / pc_reference[2]
    phantom = np.array(
        [
            [alpha, 0.0, pc_target[0] - pc_reference[0]],
            [0.0, alpha, pc_target[1] - pc_reference[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    corrected = rect_parameters_of(np.linalg.inv(phantom) @ rect_matrix_of(h))
    dd = pc_reference[2]
    return np.array(
        [
            [1 + corrected[0], corrected[1], corrected[2] / dd],
            [corrected[3], 1 + corrected[4], corrected[5] / dd],
            [dd * corrected[6], dd * corrected[7], 1.0],
        ]
    )


@functools.lru_cache(maxsize=1)
def rect_texture():
    """Return the rectangular reference pattern: an off-centre crop of
    the shipped stereographic nickel master, scaled to ``[0, 1]``.

    A real Kikuchi texture rather than a procedural one, for the
    reason the engine module projects its oracles from this same
    master: the band-pass of D4.1 and the bicubic interpolator of D3
    are both tuned on broadband band-contrast content, and a narrow
    band synthetic texture measures the interpolator's own aliasing
    instead of the fit.  Read-only and cached: the master is loaded
    from disk.
    """
    master = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
    nrows, ncols = RECT_SHAPE
    row0, col0 = RECT_CROP
    crop = np.asarray(
        master.data[row0 : row0 + nrows, col0 : col0 + ncols], dtype=np.float64
    )
    crop = crop - crop.min()
    crop = crop / crop.max()
    crop.flags.writeable = False
    return crop


@functools.lru_cache(maxsize=1)
def rect_patterns():
    """Return the ``(6, 61, 101)`` pattern stack, each point a copy of
    the reference warped by its own entry of ``RECT_WARPS``."""
    pc_px = rect_pc_pixels(
        np.asarray(RECT_PC) + RECT_PC_OFFSETS.reshape(-1, 3), RECT_SHAPE
    )[RECT_REFERENCE_INDEX]
    texture = rect_texture()
    stack = np.stack([rect_warp_with_skimage(texture, h, pc_px) for h in RECT_WARPS])
    stack.flags.writeable = False
    return stack


def rect_inputs():
    """Return ``(signal, xmap, detector)`` of the rectangular map, with
    ONE PROJECTION CENTRE PER MAP POINT as the Si-indent file has."""
    signal = kp.signals.EBSD(
        np.array(rect_patterns()).reshape(*RECT_NAVIGATION_SHAPE, *RECT_SHAPE)
    )
    detector = kp.detectors.EBSDDetector(
        shape=RECT_SHAPE,
        binning=RECT_BINNING,
        px_size=70.0,
        pc=np.asarray(RECT_PC) + RECT_PC_OFFSETS,
        sample_tilt=70.0,
        tilt=0.0,
    )
    arrays, size = create_coordinate_arrays(
        RECT_NAVIGATION_SHAPE, (RECT_STEP_UM, RECT_STEP_UM)
    )
    arrays["rotations"] = Rotation.identity((size,))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="si", space_group=227))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    return signal, xmap, detector


def rect_run(signal, xmap, detector, **kwargs):
    """Call the public method on the rectangular map."""
    kwargs.setdefault("reference", RECT_REFERENCE)
    kwargs.setdefault("verbose", 0)
    return signal.hrebsd_dic(xmap, detector, **kwargs)


class TestRectangularEndToEnd:
    """A NON-SQUARE map through ``EBSD.hrebsd_dic`` and the four
    analysis functions, the regression the Si-indent application rests
    on.

    Everything above this point runs on 60 by 60 patterns, where every
    ``nrows`` versus ``ncols`` mutant of requirements D1.2, D4.1 and
    D4.4 is the identity.  The Si-indent data set is 512 rows by 622
    columns, so the entire application would run through code paths
    this suite had exercised only at unit level.  [D15/D16, plan OQ13]
    """

    @staticmethod
    def reference_pc_pixels(transpose=False):
        """Return the reference point's ``(PCx_px, PCy_px, DD_px)``,
        computed here."""
        return rect_pc_pixels(
            np.asarray(RECT_PC) + RECT_PC_OFFSETS.reshape(-1, 3),
            RECT_SHAPE,
            transpose=transpose,
        )[RECT_REFERENCE_INDEX]

    @staticmethod
    def corners():
        return rect_subregion_corners(
            RECT_SHAPE, TestRectangularEndToEnd.reference_pc_pixels()
        )

    # ---------------- (i) the run and the prop set ---------------- #

    def test_the_shape_really_is_rectangular(self):
        # the premise of the whole class, asserted so that a future
        # edit which squares the fixture fails HERE and says why
        assert RECT_SHAPE[0] != RECT_SHAPE[1]
        signal, _, detector = rect_inputs()
        assert signal.axes_manager.signal_shape[::-1] == RECT_SHAPE
        assert detector.shape == RECT_SHAPE
        assert detector.pc.shape == (*RECT_NAVIGATION_SHAPE, 3)

    def test_the_full_prop_set_with_the_right_shapes(self):
        signal, xmap, detector = rect_inputs()
        result = rect_run(signal, xmap, detector)
        assert isinstance(result, CrystalMap)
        assert result.shape == RECT_NAVIGATION_SHAPE
        assert set(STAGE_A_PROP_NAMES) <= set(result.prop)
        assert result.prop["homography"].shape == (RECT_SIZE, HOMOGRAPHY_PROP_SIZE)
        assert result.prop["Fe"].shape == (RECT_SIZE, FE_PROP_SIZE)
        for name, dtype in PROP_DTYPES.items():
            array = np.asarray(result.prop[name])
            assert array.dtype == dtype, name
            assert array.shape[0] == RECT_SIZE, name
        # the documented reshape route of D15.7 on a rectangular map
        reshaped = result.prop["Fe"].reshape(*RECT_NAVIGATION_SHAPE, FE_PROP_SIZE)
        assert np.array_equal(reshaped[1, 2], result.prop["Fe"][5])
        # the input orientations are untouched (D7)
        assert np.array_equal(result.rotations.data, xmap.rotations.data)

    def test_a_transposed_detector_shape_is_refused(self):
        # the cheapest rectangular pin there is, and one a square
        # fixture cannot make: a detector whose shape is the pattern's
        # transpose must not be silently accepted
        signal, xmap, _ = rect_inputs()
        detector = kp.detectors.EBSDDetector(
            shape=RECT_SHAPE[::-1],
            binning=RECT_BINNING,
            px_size=70.0,
            pc=np.asarray(RECT_PC) + RECT_PC_OFFSETS,
            sample_tilt=70.0,
        )
        with pytest.raises(ValueError, match="shape"):
            rect_run(signal, xmap, detector)

    # -------------- (iii) and (v) the recovered warps ------------- #

    def test_every_point_recovers_its_own_imposed_warp(self):
        signal, xmap, detector = rect_inputs()
        result = rect_run(signal, xmap, detector)
        homography = np.asarray(result.prop["homography"])
        converged = np.asarray(result.prop["converged"])
        assert np.all(converged)
        assert np.all(np.isfinite(homography))
        assert np.all(np.isfinite(np.asarray(result.prop["Fe"])))
        assert np.all(np.isfinite(np.asarray(result.prop["residual"])))
        assert np.all(np.isfinite(np.asarray(result.prop["norm_dp"])))
        corners = self.corners()
        # the reference correlates with itself
        assert (
            rect_corner_norm(rect_matrix_of(homography[RECT_REFERENCE_INDEX]), corners)
            < 1e-6
        )
        for index in range(RECT_SIZE):
            own = rect_recovery_error(homography[index], RECT_WARPS[index], corners)
            assert own < RECT_WARP_TOL, index
            # and no point's fit is closer to another point's imposed
            # warp than to its own, which is what makes a permuted
            # result visible with no tolerance at all (D16)
            others = [
                rect_recovery_error(homography[index], RECT_WARPS[other], corners)
                for other in range(RECT_SIZE)
                if other != index
            ]
            assert own < min(others), index

    def test_the_asymmetric_warp_dies_if_the_shape_is_transposed(self):
        """Pin (v): the recovered homography of one deliberately
        asymmetric warp, and the number a row/column swap would give.

        HOW A TRANSPOSE IS NUMERICALLY VISIBLE HERE, stated so the
        pin is reviewable rather than merely tight.  The fit lives in
        the reference-PC-centred frame of requirements D1.3, whose
        origin is ``(PCx_px, PCy_px) = (pcx * ncols, pcy * nrows)``.
        On a square detector those two are the same number and a swap
        changes nothing; here ``ncols - nrows`` is 40, so the swapped
        origin sits ``d = (+16.92, -23.12)`` binned pixels away from
        the true one (computed below, not quoted).  A homography
        fitted about a DIFFERENT origin is the conjugate
        ``T(d) . W . T(-d)``, which leaves the linear block ``A``
        alone and moves the translation by ``-(A - I) d``.  So a pure
        translation would be origin invariant and could not see the
        swap at all, which is why the warp used here has an
        ANISOTROPIC linear block: ``(A - I) d`` then has a large
        component along both axes.

        MEASURED 2026-09-08 on this fixture: the swap moves ``h13`` by
        0.5312 px and ``h23`` by 0.4342 px, while the fit recovers
        them to 0.0025 px and 0.0005 px, so the two hypotheses are
        separated by more than two orders of magnitude.  The same
        comparison in the corner norm is 0.0203 px against 0.6900 px.
        """
        signal, xmap, detector = rect_inputs()
        result = rect_run(signal, xmap, detector)
        fit = np.asarray(result.prop["homography"])[RECT_ASYMMETRIC]
        imposed = RECT_WARPS[RECT_ASYMMETRIC]
        corners = self.corners()

        # the homography the SAME data would give about the swapped
        # origin, assembled here by conjugation
        true_origin = self.reference_pc_pixels()[:2]
        swapped_origin = self.reference_pc_pixels(transpose=True)[:2]
        offset = true_origin - swapped_origin
        forward = np.eye(3)
        forward[0, 2], forward[1, 2] = offset
        backward = np.eye(3)
        backward[0, 2], backward[1, 2] = -offset
        transposed = rect_parameters_of(forward @ rect_matrix_of(imposed) @ backward)

        # the swap really does move the translations, and by far more
        # than the linear block it leaves alone
        assert abs(offset[0]) > 10.0 and abs(offset[1]) > 10.0
        assert np.allclose(transposed[[0, 1, 3, 4]], imposed[[0, 1, 3, 4]])
        for slot in (2, 5):
            measured = abs(fit[slot] - imposed[slot])
            separation = abs(transposed[slot] - imposed[slot])
            assert measured < 0.02, slot
            assert separation > 0.3, slot
            assert separation > 10.0 * measured, slot

        own = rect_recovery_error(fit, imposed, corners)
        swapped = rect_recovery_error(fit, transposed, corners)
        assert own < RECT_WARP_TOL
        assert swapped > 5.0 * RECT_WARP_TOL
        assert swapped > 10.0 * own

    def test_fe_matches_the_hand_built_d6_conversion(self):
        """The ``Fe`` half of the same pin.

        ``Fe`` divides the fitted translations by ``DD_px = pcz *
        nrows`` and multiplies the perspective pair by it, and the
        D6.2 phantom removed first carries ``gamma = PC_target -
        PC_reference`` in the same mixed units.  A row/column swap
        there rescales ``Fe13`` and ``Fe23`` by ``ncols / nrows``,
        which is 1.66 here and exactly one on a square detector.

        MEASURED 2026-09-08: the recovered ``Fe`` of the asymmetric
        point sits 2.12e-04 from the expectation built with the right
        axes and 1.03e-02 from the one built with them swapped, a
        factor of 48.
        """
        signal, xmap, detector = rect_inputs()
        result = rect_run(signal, xmap, detector)
        fe = np.asarray(result.prop["Fe"])
        pc_px = rect_pc_pixels(
            np.asarray(RECT_PC) + RECT_PC_OFFSETS.reshape(-1, 3), RECT_SHAPE
        )
        reference_pc = pc_px[RECT_REFERENCE_INDEX]
        for index in range(RECT_SIZE):
            expected = rect_expected_fe(RECT_WARPS[index], reference_pc, pc_px[index])
            error = np.abs(fe[index].reshape(3, 3) - expected).max()
            assert error < RECT_FE_TOL, index
        # the reference's own tensor is the identity
        np.testing.assert_allclose(
            fe[RECT_REFERENCE_INDEX].reshape(3, 3), np.eye(3), atol=1e-9
        )

        # and the swapped-axes expectation is far away
        pc_px_swapped = rect_pc_pixels(
            np.asarray(RECT_PC) + RECT_PC_OFFSETS.reshape(-1, 3),
            RECT_SHAPE,
            transpose=True,
        )
        swapped = rect_expected_fe(
            RECT_WARPS[RECT_ASYMMETRIC],
            pc_px_swapped[RECT_REFERENCE_INDEX],
            pc_px_swapped[RECT_ASYMMETRIC],
        )
        distance = np.abs(fe[RECT_ASYMMETRIC].reshape(3, 3) - swapped).max()
        assert distance > 5.0 * RECT_FE_TOL

    # ------------------- (ii) lazy equals eager ------------------- #

    def test_lazy_input_agrees_with_eager_bitwise(self):
        # the Si-indent run is lazy from the first line, the data set
        # being 18.9 GB, so the rectangular route has to be pinned on
        # the lazy path too
        signal, xmap, detector = rect_inputs()
        eager = rect_run(signal, xmap, detector)
        lazy = rect_run(signal.as_lazy(), xmap, detector)
        assert isinstance(lazy, CrystalMap)
        for name in STAGE_A_PROP_NAMES:
            left = np.asarray(eager.prop[name])
            right = np.asarray(lazy.prop[name])
            assert isinstance(right, np.ndarray), name
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name

    # ------------- (iv) the four analysis functions --------------- #

    def test_the_whole_analysis_chain_runs_on_a_rectangular_map(self):
        # deviatoric closure, which is the Si-indent scope: no
        # stiffness is passed anywhere
        signal, xmap, detector = rect_inputs()
        result = rect_run(signal, xmap, detector)

        stage_b = kp.indexing.hrebsd_strain_stress(result, detector)
        assert stage_b.shape == RECT_NAVIGATION_SHAPE
        assert np.asarray(stage_b.prop["strain"]).shape == (RECT_SIZE, 6)
        assert np.asarray(stage_b.prop["rotation_vector"]).shape == (RECT_SIZE, 3)
        assert np.asarray(stage_b.prop["beta"]).shape == (RECT_SIZE, 9)
        assert np.all(np.isfinite(np.asarray(stage_b.prop["strain"])))
        assert np.all(np.isfinite(np.asarray(stage_b.prop["rotation_vector"])))
        # the deviatoric closure was taken, so no stress exists
        assert np.all(np.isnan(np.asarray(stage_b.prop["stress"])))
        # the reference point measures nothing against itself
        np.testing.assert_allclose(
            np.asarray(stage_b.prop["strain"])[RECT_REFERENCE_INDEX], 0.0, atol=1e-9
        )

        kam = kp.indexing.hrebsd_kam(stage_b)
        assert kam.shape == RECT_NAVIGATION_SHAPE
        assert kam.dtype == np.float64
        assert np.all(np.isfinite(kam))
        assert np.all(kam >= 0)

        gnd = kp.indexing.hrebsd_gnd(stage_b, detector, 3.84e-10)
        assert gnd.shape == RECT_NAVIGATION_SHAPE
        assert gnd.dtype == np.float64
        assert np.all(np.isfinite(gnd))
        assert np.all(gnd > 0)

        maps = kp.indexing.hrebsd_pc_shift(result, detector)
        assert set(maps) == {
            "translation_x",
            "translation_y",
            "translation_x_model",
            "translation_y_model",
            "scaling_model",
            "residual_x",
            "residual_y",
        }
        for name, array in maps.items():
            assert array.shape == RECT_NAVIGATION_SHAPE, name
            assert array.dtype == np.float64, name
            assert np.all(np.isfinite(array)), name
        # the measured translations ARE the raw homography's, read
        # back through the map grid of a rectangular NON-SQUARE
        # navigation shape, and the model half reads the same
        # per-point projection centres in the same mixed units
        homography = np.asarray(result.prop["homography"]).reshape(
            *RECT_NAVIGATION_SHAPE, HOMOGRAPHY_PROP_SIZE
        )
        np.testing.assert_array_equal(maps["translation_x"], homography[..., 2])
        np.testing.assert_array_equal(maps["translation_y"], homography[..., 5])
        np.testing.assert_allclose(
            maps["residual_x"], maps["translation_x"] - maps["translation_x_model"]
        )
