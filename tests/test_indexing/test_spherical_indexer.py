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

"""Tests of ``kikuchipy.indexing._spherical._indexer``.

Covers the assertions of
``specs/2026-09-02-spherical-indexing-ebsd/validation.md`` which
belong to the module rather than to the signal method:

- Construction and guards: the frozen signature defaults, the
  ``refine`` refusal and its order against the projector, the
  ``[16, 512]`` bandwidth rule, the harmonics against detector
  ``sample_tilt`` binding, the guards which propagate from the
  back-projector, the shared geometry check, the resize warning
  asymmetry, the attribute table, the shared Wigner table, both
  ``repr`` forms and the immutability of a built indexer.
- ``_batch_estimate``: the three large-map pins, the load balancing
  rule and the recorded ``max(1, .)`` clamp.
- ``index_patterns`` and ``_map_chunks``: the input guards, the three
  monkeypatched failure paths (a non-positive score, a constant
  processed pattern and a non-finite score), the unguarded NaN pixel,
  the returned contract and the truthfulness of the graph metadata.
- The memory model, its ``normalize`` factor and the 2 GiB warning.
- The public exports, the ``fast_bandwidths`` publication pass and
  the absence of roadmap phase numbers in public docstrings.
- The inherited kernel flag regression.
"""

import functools
import importlib
import inspect
import pkgutil
import re
import warnings

import dask
import dask.array as da
import numpy as np
from orix.crystal_map import Phase
import pytest

import kikuchipy as kp
from kikuchipy.detectors import EBSDDetector
from kikuchipy.indexing import _spherical
from kikuchipy.indexing._spherical import _back_projection, _fft, _indexer
from kikuchipy.indexing._spherical._back_projection import SphericalBackProjector
from kikuchipy.indexing._spherical._indexer import (
    SphericalIndexer,
    _batch_estimate,
    _map_chunks,
)
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._preprocessing import _circular_mask
from kikuchipy.indexing._spherical._xcorr import (
    NormalizedSphericalCrossCorrelator,
)

# The bandwidth of the real data tests and the Euler cube it gives
NI_BANDWIDTH = 68
NI_SIDE_LENGTH = 135
NI_HALF_SIDE_LENGTH = 68
NI_HALF_CELL_DEGREES = 1.3333

# Measured sphere window of the Ni detector at ``bw`` 68, without and
# with the inscribed circle (Phase 5 D3)
WINDOW_POINTS = 1317
CIRCLE_WINDOW_POINTS = 1117

# The frozen ``[16, 512]`` rule of ``IndexEBSD``'s name list
BANDWIDTH_LIMITS = (16, 512)

# ``memory_per_worker_bytes`` at ``bw`` 68 with one normalized phase:
# 135^2 x 68 x 24 + 135^3 x 8
MEMORY_MODEL_BW68 = 49_426_200

# What one more normalized phase costs, ``slP^2 x bwP x 24``
MEMORY_PER_EXTRA_PHASE = NI_SIDE_LENGTH**2 * NI_HALF_SIDE_LENGTH * 24

# ``BatchEstimate`` pins for a large map on eight workers (D4)
BATCH_PINS = {53: 34, 68: 15, 88: 6}

# ``fast_bandwidths(16, 128)``, a superset of the values ``nml.hpp``
# suggests in this range (D9)
FAST_BANDWIDTHS_16_128 = [
    17, 18, 20, 23, 25, 28, 32, 33, 38, 39, 41, 46, 50, 53, 59, 61,
    63, 68, 72, 74, 83, 85, 88, 95, 98, 113, 116, 122, 123,
]  # fmt: skip

# The two kernels of the package which need the IEEE error model
NUMPY_ERROR_MODEL_KERNELS = {"_interpolate_maxima", "_fit_gaussian_1d_kernel"}


# ----------------------------- Helpers ------------------------------ #


@functools.lru_cache(maxsize=1)
def ni_patterns():
    """Return the nine background corrected ``nickel_ebsd_small``
    patterns as a read-only ``(9, 60, 60)`` unsigned 8-bit array.

    The array is made read-only because it is handed straight to the
    indexer, so an in-place write there would corrupt every later
    test instead of failing where it happens.
    """
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    data = signal.data.reshape((-1, 60, 60))
    data.flags.writeable = False
    return data


def ni_detector():
    """Return a fresh Ni detector with one projection centre, the
    average of the nine stored ones.
    """
    detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    detector.pc = detector.pc_average
    return detector


@functools.lru_cache(maxsize=4)
def ni_harmonics(bandwidth):
    """Return the harmonics of the shipped Ni master pattern built
    **directly** at ``bandwidth``, cached because the transform costs
    about a second.

    Direct construction and ``resize()`` from another bandwidth do
    not agree, so every pinned value in this suite assumes this one.
    """
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


def ni_indexer(**kwargs):
    """Return an indexer of the Ni harmonics and detector at ``bw``
    68, with the default configuration unless overridden.
    """
    return SphericalIndexer(ni_harmonics(NI_BANDWIDTH), ni_detector(), **kwargs)


def scrambled_harmonics(seed=42):
    """Return a decoy phase with the same power spectrum as the Ni
    master but a different function.

    Every coefficient keeps its magnitude and the ``m = 0`` row stays
    real, so the coefficients are those of a real spherical function,
    which a *phase* scramble would not be.
    """
    harmonics = ni_harmonics(NI_BANDWIDTH)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], harmonics.alm.shape)
    return MasterPatternHarmonics(
        harmonics.alm * signs, phase=Phase("scrambled", point_group="1")
    )


def user_warnings(record):
    """Return the ``UserWarning`` entries of a recorded warning list.

    Building harmonics deep copies a phase, which emits an unrelated
    :class:`DeprecationWarning` from ``diffpy.structure``, so an
    "error on any warning" assertion would fail for the wrong reason.
    """
    return [w for w in record if issubclass(w.category, UserWarning)]


def raise_assertion(*args, **kwargs):
    """Sentinel standing in for an expensive constructor."""
    raise AssertionError("the sentinel was reached")


def spherical_modules():
    """Return every submodule of ``kikuchipy.indexing._spherical``."""
    return [
        importlib.import_module(f"{_spherical.__name__}.{info.name}")
        for info in pkgutil.iter_modules(_spherical.__path__)
    ]


def njit_kernels(module):
    """Return the module's own Numba kernels, by name."""
    return {
        name: value
        for name, value in vars(module).items()
        if type(value).__name__ == "CPUDispatcher"
        and getattr(value, "py_func", None) is not None
        and value.py_func.__module__ == module.__name__
    }


def public_docstrings(module):
    """Return the docstrings of the module's exported names and their
    public methods, keyed on a readable name.

    Only names in :data:`kikuchipy.indexing.__all__` count as public:
    ``fast_size`` has no leading underscore but is not exported, so
    it is documentation of a private helper.
    """
    exported = set(kp.indexing.__all__)
    docstrings = {}
    for name, obj in vars(module).items():
        if name not in exported:
            continue
        if obj.__doc__:
            docstrings[f"{module.__name__}.{name}"] = obj.__doc__
        if not inspect.isclass(obj):
            continue
        for member_name, member in vars(obj).items():
            if member_name.startswith("_"):
                continue
            doc = getattr(member, "__doc__", None)
            if doc:
                docstrings[f"{module.__name__}.{name}.{member_name}"] = doc
    return docstrings


# -------------------- Construction and guards (D1) ------------------- #


class TestSphericalIndexerConstruction:
    def test_signature_defaults_are_the_indexebsd_name_list(self):
        # pinned so that a drive-by "improvement" of a default is
        # caught: these are ``IndexEBSD``'s own name list defaults
        parameters = inspect.signature(SphericalIndexer.__init__).parameters
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        assert defaults == {
            "bandwidth": 68,
            "normalize": True,
            "refine": False,
            "signal_mask": None,
            "n_regions": 10,
            "gaussian_background": False,
            "circular_mask": False,
            "emsphinx_compatible": True,
        }
        # ``harmonics`` and ``detector`` are positional, the rest
        # keyword only
        assert [
            name
            for name, parameter in parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ] == ["self", "harmonics", "detector"]

    def test_refine_true_is_refused(self):
        with pytest.raises(NotImplementedError, match="refine=True") as info:
            ni_indexer(refine=True)
        message = str(info.value)
        assert "not implemented" in message
        # decision: public messages carry no roadmap phase numbers
        for phase in ("Phase 6", "Phase 7", "spherical-refinement"):
            assert phase not in message

    def test_refine_is_refused_before_the_projector_is_built(self, monkeypatch):
        # pins the guard order with no wall clock: if the projector
        # were built first the sentinel would fire instead
        monkeypatch.setattr(SphericalBackProjector, "__init__", raise_assertion)
        with pytest.raises(NotImplementedError, match="refine=True"):
            ni_indexer(refine=True)

    @pytest.mark.parametrize("bandwidth", [8, 15, 513, 600])
    def test_unreasonable_bandwidth_is_refused(self, bandwidth):
        # the projector's own guard is only ``bandwidth >= 1``, so
        # without this rule ``bandwidth=8`` silently builds a grid
        # with a 12 degree half cell
        with pytest.raises(ValueError, match="unreasonable bandwidth") as info:
            ni_indexer(bandwidth=bandwidth)
        assert "[16, 512]" in str(info.value)

    def test_the_smallest_allowed_bandwidth_constructs_and_indexes(self):
        indexer = ni_indexer(bandwidth=BANDWIDTH_LIMITS[0])
        assert indexer.bandwidth == BANDWIDTH_LIMITS[0]
        results = indexer.index_patterns(ni_patterns()[:1], progressbar=False)
        assert results["zyz"].shape == (1, 1, 3)

    def test_the_largest_allowed_bandwidth_passes_the_range_guard(
        self, monkeypatch, record_property
    ):
        # a real ``bw`` 512 kit is tens of gigabytes, so the boundary
        # is asserted with the projector sentinel: reaching it means
        # the range guard let 512 through.  The deviation from the
        # spec's "constructs" is recorded with its arithmetic
        record_property(
            "bandwidth_512_deviation",
            "not constructed: slP 1024, bwP 513 give correlator cubes "
            "1024^2 x 513 x 24 = 12.91 GB, an interpolation cube "
            "1024^3 x 8 = 8.59 GB (21.5 GB of model) and a 512^3 x 8 = "
            "1.07 GB Wigner table; the range guard is asserted with the "
            "projector sentinel instead",
        )
        monkeypatch.setattr(SphericalBackProjector, "__init__", raise_assertion)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AssertionError, match="sentinel"):
                ni_indexer(bandwidth=BANDWIDTH_LIMITS[1])

    def test_the_sample_tilt_of_the_harmonics_binds_the_detector(self, record_property):
        # EMSphInx builds the geometry *from* the master, so a
        # mismatch is impossible there; here the tilt comes from the
        # detector and must be checked
        record_property(
            "tilt_mismatch_rationale",
            "harmonics 70 deg vs detector 65 deg indexes at median "
            "4.680 / max 5.099 deg from the stored xmap with *higher* "
            "scores (0.5053-0.6474 vs 0.4963-0.6239), so no score "
            "check could catch a dropped binding",
        )
        detector = ni_detector()
        detector.sample_tilt = 65.0
        with pytest.raises(ValueError, match="sample_tilt") as info:
            SphericalIndexer(ni_harmonics(NI_BANDWIDTH), detector)
        message = str(info.value)
        assert "70" in message
        assert "65" in message
        assert "EBSDDetector" in message

    def test_a_harmonics_without_a_sample_tilt_skips_the_binding(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        tiltless = MasterPatternHarmonics(harmonics.alm, phase=harmonics.phase)
        assert tiltless.sample_tilt is None
        detector = ni_detector()
        detector.sample_tilt = 65.0
        indexer = SphericalIndexer(tiltless, detector)
        assert indexer.projector.detector.sample_tilt == 65.0

    def test_the_multi_projection_centre_message_propagates(self):
        # the indexer adds no duplicate geometry checks: the
        # projector's message is the one to maintain
        detector = kp.data.nickel_ebsd_small().detector
        with pytest.raises(ValueError, match="pc_average") as info:
            SphericalIndexer(ni_harmonics(NI_BANDWIDTH), detector)
        assert "deepcopy" in str(info.value)

    def test_the_azimuthal_guard_propagates(self):
        detector = ni_detector()
        detector.azimuthal = 5
        with pytest.raises(ValueError, match="azimuthal"):
            SphericalIndexer(ni_harmonics(NI_BANDWIDTH), detector)

    def test_an_empty_harmonics_sequence_is_refused(self):
        with pytest.raises(ValueError):
            SphericalIndexer([], ni_detector())

    def test_a_non_harmonics_entry_is_refused(self):
        # the message must *name the offending index*, which a bare
        # ``match="1"`` would not pin: any message containing the
        # character passes that
        with pytest.raises(TypeError, match=r"index 1|harmonics\[1\]"):
            SphericalIndexer([ni_harmonics(NI_BANDWIDTH), 3], ni_detector())

    @pytest.mark.parametrize("n_regions", [-1, 61])
    def test_n_regions_is_validated_at_construction(self, n_regions):
        # failing here beats failing inside a dask worker
        with pytest.raises(ValueError):
            ni_indexer(n_regions=n_regions)

    def test_two_phases_must_share_the_sample_tilt(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        other = MasterPatternHarmonics(
            harmonics.alm,
            phase=Phase("other", point_group="1"),
            sample_tilt=60.0,
            beam_energy=harmonics.beam_energy,
        )
        with pytest.raises(ValueError) as info:
            SphericalIndexer([harmonics, other], ni_detector())
        message = str(info.value)
        assert "70" in message
        assert "60" in message

    def test_two_phases_must_share_the_beam_energy(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        other = MasterPatternHarmonics(
            harmonics.alm,
            phase=Phase("other", point_group="1"),
            sample_tilt=harmonics.sample_tilt,
            beam_energy=15.0,
        )
        with pytest.raises(ValueError) as info:
            SphericalIndexer([harmonics, other], ni_detector())
        message = str(info.value)
        assert "20.1" in message
        assert "15" in message

    def test_a_phase_without_geometry_skips_the_shared_check(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        other = MasterPatternHarmonics(
            harmonics.alm, phase=Phase("other", point_group="1")
        )
        indexer = SphericalIndexer([harmonics, other], ni_detector())
        assert indexer.n_phases == 2

    def test_truncating_a_larger_harmonics_emits_no_warning(self):
        # never ``pytest.warns(None)`` or ``simplefilter("error")``:
        # the deep copy of the phase emits an unrelated
        # ``DeprecationWarning`` from diffpy
        harmonics = ni_harmonics(120)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            indexer = SphericalIndexer(harmonics, ni_detector(), bandwidth=NI_BANDWIDTH)
        assert user_warnings(record) == []
        assert indexer.phases[0].bandwidth == NI_BANDWIDTH

    def test_zero_padding_a_smaller_harmonics_warns(self):
        # EMSphInx pads silently; the warning is our addition, since
        # padding buys a finer Euler grid but no new signal
        with pytest.warns(UserWarning) as record:
            indexer = SphericalIndexer(
                ni_harmonics(53), ni_detector(), bandwidth=NI_BANDWIDTH
            )
        messages = " ".join(str(w.message) for w in record)
        assert "53" in messages
        assert "68" in messages
        assert indexer.phases[0].bandwidth == NI_BANDWIDTH
        # and the correlate path still runs
        results = indexer.index_patterns(ni_patterns()[:1], progressbar=False)
        assert np.isfinite(results["scores"]).all()

    def test_attributes(self):
        indexer = ni_indexer()
        assert indexer.n_phases == 1
        assert indexer.bandwidth == NI_BANDWIDTH
        assert indexer.normalize is True
        assert isinstance(indexer.projector, SphericalBackProjector)
        assert indexer.projector.bandwidth == NI_BANDWIDTH
        assert indexer.projector.n_points == WINDOW_POINTS
        assert indexer.side_length == NI_SIDE_LENGTH
        assert indexer.half_cell_degrees == pytest.approx(
            NI_HALF_CELL_DEGREES, abs=1e-3
        )
        assert indexer.n_regions == 10
        assert indexer.gaussian_background is False
        assert indexer.circular_mask is False
        assert indexer.emsphinx_compatible is True
        assert indexer.signal_mask is None
        # neither a signal mask nor the circle: no histogram mask,
        # ``IndexEBSD``'s ``circmask = -1``
        assert indexer.good_pixels is None

    def test_the_attributes_follow_the_arguments(self):
        # the table above pins the *defaults*, so an attribute frozen
        # at its default value is invisible there; these are
        # documented public attributes which a caller reads back
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:32, 25:40] = True
        indexer = ni_indexer(
            signal_mask=mask,
            circular_mask=True,
            n_regions=4,
            gaussian_background=True,
            emsphinx_compatible=False,
            normalize=False,
        )
        assert np.array_equal(indexer.signal_mask, mask)
        assert indexer.circular_mask is True
        assert indexer.n_regions == 4
        assert indexer.gaussian_background is True
        assert indexer.emsphinx_compatible is False
        assert indexer.normalize is False

    def test_the_circle_reaches_both_the_window_and_the_histogram(self):
        indexer = ni_indexer(circular_mask=True)
        assert indexer.projector.n_points == CIRCLE_WINDOW_POINTS
        assert indexer.good_pixels is not None
        assert indexer.good_pixels.dtype == np.bool_

    def test_a_signal_mask_and_the_circle_intersect(self):
        # the two terms of ``good_pixels`` are reduced to the ones
        # present, and both present means their intersection.  No
        # other test of the suite gives both, so a mutant which keeps
        # only one of the two terms survives without this one
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:32, 25:40] = True
        indexer = ni_indexer(signal_mask=mask, circular_mask=True)
        circle = _circular_mask((60, 60))
        assert np.array_equal(indexer.good_pixels, ~mask & circle)
        # strictly fewer good pixels than either term alone
        assert indexer.good_pixels.sum() < circle.sum()
        assert indexer.good_pixels.sum() < (~mask).sum()

    def test_the_wigner_table_is_shared_between_correlators(self):
        indexer = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()], ni_detector()
        )
        first, second = indexer.correlators
        assert first.correlator.wigner_d_half_pi is second.correlator.wigner_d_half_pi

    def test_repr(self):
        text = repr(ni_indexer())
        assert "1 phase" in text
        assert "ni" in text
        assert "bw = 68" in text
        assert "normalized" in text

    def test_repr_of_a_phase_less_harmonics(self):
        # no ``catch_warnings`` wrapper: the phase-less construction
        # emits nothing (measured, empty record list), so a wrapper
        # would only hide a future regression -- as the sibling
        # ``test_a_phase_less_harmonics_is_refused`` of the signal
        # file already shows by not having one
        harmonics = ni_harmonics(NI_BANDWIDTH)
        anonymous = MasterPatternHarmonics(harmonics.alm)
        assert anonymous.phase is None
        indexer = SphericalIndexer(anonymous, ni_detector())
        assert "?" in repr(indexer)

    def test_the_indexer_is_isolated_from_the_callers_detector(self):
        detector = ni_detector()
        indexer = SphericalIndexer(ni_harmonics(NI_BANDWIDTH), detector)
        before = indexer.projector.detector.pc.copy()
        first = indexer.index_patterns(ni_patterns(), progressbar=False)

        detector.pc = (0.1, 0.2, 0.3)

        assert np.array_equal(indexer.projector.detector.pc, before)
        second = indexer.index_patterns(ni_patterns(), progressbar=False)
        for key in ("zyz", "scores", "phase_id", "iq"):
            assert np.array_equal(first[key], second[key])


# -------------------------- _batch_estimate ------------------------- #


class TestBatchEstimate:
    @pytest.mark.parametrize("bandwidth", sorted(BATCH_PINS))
    def test_large_map_pins(self, bandwidth):
        # the ``k = 1e-8`` model with the C++'s truncation towards
        # zero, never ``round``
        assert _batch_estimate(bandwidth, 8, 100_000) == BATCH_PINS[bandwidth]

    def test_the_load_balancing_rule_parallelises_small_maps(self):
        # nine patterns on four workers give nine chunks, where a
        # byte based chunking would give one
        assert _batch_estimate(NI_BANDWIDTH, 4, 9) == 1
        assert _batch_estimate(NI_BANDWIDTH, 1, 9) == BATCH_PINS[NI_BANDWIDTH]

    def test_zero_patterns_hits_the_recorded_clamp(self, record_property):
        # the verbatim ``nt^2`` branch returns ``ceil(0 / 16) = 0``,
        # which the C++ never meets: the ``max(1, .)`` clamp is our
        # one line deviation, recorded
        record_property("batch_estimate_zero_deviation", "verbatim 0, clamped 1")
        assert _batch_estimate(NI_BANDWIDTH, 4, 0) == 1


# ---------------------- index_patterns, chunking -------------------- #


class TestIndexPatterns:
    def test_the_result_contract(self):
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert set(results) == {"zyz", "scores", "phase_id", "iq"}
        assert results["zyz"].shape == (9, 1, 3)
        assert results["zyz"].dtype == np.float64
        assert results["scores"].shape == (9, 1)
        assert results["scores"].dtype == np.float64
        assert results["phase_id"].shape == (9, 1)
        assert results["phase_id"].dtype == np.int32
        assert results["iq"].shape == (9,)
        assert results["iq"].dtype == np.float64

    @pytest.mark.parametrize("n_best", [0, -1])
    def test_n_best_below_one_is_refused(self, n_best):
        indexer = ni_indexer()
        with pytest.raises(ValueError):
            indexer.index_patterns(ni_patterns(), n_best=n_best, progressbar=False)

    @pytest.mark.parametrize("chunksize", [0, -1])
    def test_an_explicit_chunksize_below_one_is_refused(self, chunksize):
        # nothing rejected it before, and ``da.from_array`` fails
        # obscurely on a chunk of zero
        indexer = ni_indexer()
        with pytest.raises(ValueError):
            indexer.index_patterns(
                ni_patterns(), chunksize=chunksize, progressbar=False
            )

    def test_patterns_of_the_wrong_detector_shape_are_refused(self):
        indexer = ni_indexer()
        with pytest.raises(ValueError) as info:
            indexer.index_patterns(np.zeros((2, 60, 59), np.uint8), progressbar=False)
        message = str(info.value)
        assert "(60, 59)" in message
        assert "(60, 60)" in message

    def test_a_non_positive_score_is_never_recorded(self, monkeypatch):
        # the C++ seeds every row with ``corr 0`` / ``phase -1`` and
        # ``upper_bound`` places a candidate with ``corr <= 0`` after
        # every fill row, so it is dropped.  An ``argsort`` top-n
        # port would write phase 0 with score -1 here
        def negative(self, gln, **kwargs):
            return np.array([0.5, 0.5, 0.5]), -1.0

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", negative)
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert np.array_equal(results["phase_id"], np.full((9, 1), -1, np.int32))
        assert np.array_equal(results["zyz"], np.zeros((9, 1, 3)))
        assert np.array_equal(results["scores"], np.zeros((9, 1)))

    def test_a_score_which_ties_the_fill_is_never_recorded(self, monkeypatch):
        # the other half of ``upper_bound``: it inserts only where
        # the candidate **strictly** beats a row, so a candidate
        # which merely ties the fill score of zero is dropped too.
        # The negative score above does not separate ``upper_bound``
        # from ``lower_bound``, since a ``>=`` port drops ``-1`` as
        # well and records ``0`` with phase 0 only here
        def zero(self, gln, **kwargs):
            return np.array([0.5, 0.5, 0.5]), 0.0

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", zero)
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert np.array_equal(results["phase_id"], np.full((9, 1), -1, np.int32))
        assert np.array_equal(results["zyz"], np.zeros((9, 1, 3)))
        assert np.array_equal(results["scores"], np.zeros((9, 1)))

    def test_an_equal_score_ranks_after_the_earlier_phase(self, monkeypatch):
        # ``std::upper_bound`` places a tie *after* the rows it
        # equals, so the earlier phase keeps the better row; a
        # ``lower_bound`` port swaps the two
        def tied(self, gln, **kwargs):
            return np.array([0.1, 0.2, 0.3]), 0.5

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", tied)
        indexer = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()], ni_detector()
        )
        results = indexer.index_patterns(ni_patterns(), n_best=2, progressbar=False)
        assert np.array_equal(results["phase_id"], np.tile([0, 1], (9, 1)))
        assert np.array_equal(results["scores"], np.full((9, 2), 0.5))

    def test_a_later_phase_displaces_an_earlier_one(self):
        # the only phase order in the suite which makes the top-n
        # shift do any work: with the decoy first, the true phase
        # wins from the *second* slot and must push the decoy's
        # candidate down to row 1.  Everywhere else the winner is
        # already phase 0, where the shift runs over identical fill
        # rows and its direction cannot be seen
        indexer = SphericalIndexer(
            [scrambled_harmonics(), ni_harmonics(NI_BANDWIDTH)], ni_detector()
        )
        results = indexer.index_patterns(ni_patterns(), n_best=2, progressbar=False)
        assert np.array_equal(results["phase_id"], np.tile([1, 0], (9, 1)))
        # row 1 holds the decoy's real candidate, not the fill a
        # wrong way shift would leave behind
        assert (results["scores"][:, 1] > 0).all()
        assert (np.diff(results["scores"], axis=1) < 0).all()
        assert (results["zyz"][:, 1] != 0).any()

    def test_a_dropped_candidate_does_not_fail_the_pattern(self, monkeypatch):
        # the drop branch must *return*: a second phase whose non
        # positive score is thrown away has to leave the first
        # phase's winning row alone.  An off by one bound writes past
        # the end of the list instead, the per-pattern ``except``
        # swallows the ``IndexError`` and the whole point is failed
        # -- which the single phase tests above cannot see, because
        # there the dropped candidate is the only one there is
        def by_phase(self, gln, **kwargs):
            if self.n_fold == 1:
                return np.array([0.5, 0.5, 0.5]), -1.0
            return np.array([0.1, 0.2, 0.3]), 0.75

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", by_phase)
        indexer = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()], ni_detector()
        )
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert np.array_equal(results["phase_id"], np.zeros((9, 1), np.int32))
        assert np.array_equal(results["scores"], np.full((9, 1), 0.75))

    def test_a_constant_processed_pattern_is_failed(self, monkeypatch):
        # guard (b) is unreachable from a raw input, since every
        # constant raw pattern is caught by guard (a) first
        def constant(pattern, **kwargs):
            return np.full((60, 60), 1.0)

        monkeypatch.setattr(_indexer, "_preprocess_pattern", constant)
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert np.array_equal(results["phase_id"], np.full((9, 1), -1, np.int32))
        assert np.array_equal(results["zyz"], np.zeros((9, 1, 3)))
        assert np.array_equal(results["scores"], np.zeros((9, 1)))
        assert np.array_equal(results["iq"], np.zeros(9))

    def test_a_constant_processed_pattern_never_reaches_the_projector(
        self, monkeypatch
    ):
        # guard (b) is a **short circuit**, not only a fill value:
        # the back-projection would answer a constant with its window
        # mask and the correlation would then score that mask.  The
        # sibling above cannot see the guard at all, because the
        # measured -2.64 of the mask is dropped by the insertion rule
        # anyway and leaves exactly the same fill, so it passes with
        # the guard deleted
        def constant(pattern, **kwargs):
            return np.full((60, 60), 1.0)

        calls = []
        original = SphericalBackProjector.unproject

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_indexer, "_preprocess_pattern", constant)
        monkeypatch.setattr(SphericalBackProjector, "unproject", spy)
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert calls == []
        assert np.array_equal(results["phase_id"], np.full((9, 1), -1, np.int32))

    def test_a_non_finite_score_is_failed(self, monkeypatch):
        def infinite(self, gln, **kwargs):
            return np.array([0.5, 0.5, 0.5]), np.inf

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", infinite)
        indexer = ni_indexer()
        results = indexer.index_patterns(ni_patterns(), progressbar=False)
        assert np.array_equal(results["phase_id"], np.full((9, 1), -1, np.int32))
        assert np.array_equal(results["scores"], np.zeros((9, 1)))

    def test_only_the_winning_row_is_checked_for_finiteness(self, monkeypatch):
        # guard (c) is scoped to the **winning** score and angles, as
        # the contract says.  A losing candidate whose interpolated
        # peak came back with a non-finite angle -- which the numpy
        # error model of ``_interpolate_maxima`` allows -- must not
        # fail a point the winning phase indexed fine.  A guard which
        # scans the whole result block fails it instead
        def by_phase(self, gln, **kwargs):
            if self.n_fold == 1:
                return np.array([np.nan, 0.0, 0.0]), 0.1
            return np.array([0.1, 0.2, 0.3]), 0.9

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", by_phase)
        indexer = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()], ni_detector()
        )
        results = indexer.index_patterns(ni_patterns(), n_best=2, progressbar=False)
        assert np.array_equal(results["phase_id"], np.tile([0, 1], (9, 1)))
        assert (results["scores"][:, 0] == 0.9).all()
        assert np.isnan(results["zyz"][:, 1, 0]).all()

    def test_a_flat_pattern_in_a_stack_carries_the_fill(self):
        patterns = np.array(ni_patterns())
        patterns[4] = 37
        indexer = ni_indexer()
        results = indexer.index_patterns(patterns, progressbar=False)
        assert results["phase_id"][4, 0] == -1
        assert np.array_equal(results["zyz"][4], np.zeros((1, 3)))
        assert results["scores"][4, 0] == 0.0
        assert results["iq"][4] == 0.0
        # and the other eight are untouched
        assert (results["phase_id"][[0, 1, 2, 3, 5, 6, 7, 8]] == 0).all()

    def test_a_failed_pattern_is_warned_about(self):
        # a failed pattern is never re-raised on, so without the
        # count a run whose every pattern failed returns in silence
        patterns = np.array(ni_patterns())
        patterns[4] = 37
        indexer = ni_indexer()
        with pytest.warns(UserWarning, match="1 of 9 pattern"):
            results = indexer.index_patterns(patterns, progressbar=False)
        assert results["phase_id"][4, 0] == -1

    def test_a_run_without_failures_is_silent(self):
        indexer = ni_indexer()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            indexer.index_patterns(ni_patterns(), progressbar=False)
        assert not [w for w in record if "could not be indexed" in str(w.message)]

    def test_the_guards_are_caught_per_pattern(self, monkeypatch):
        # the zero variance guard sits *inside* the per-pattern
        # catch, as ``ebsdWorkItem`` wraps the whole of
        # ``indexImage()``: a guard which raises must fail its own
        # pattern, never the chunk
        real_ptp = np.ptp

        def exploding_ptp(array, *args, **kwargs):
            flat = getattr(array, "dtype", None) == np.uint8 and bool(
                np.asarray(array == 37).all()
            )
            if flat:
                raise RuntimeError("the guard raised")
            return real_ptp(array, *args, **kwargs)

        patterns = np.array(ni_patterns())
        patterns[4] = 37
        monkeypatch.setattr(_indexer.np, "ptp", exploding_ptp)
        indexer = ni_indexer()
        with pytest.warns(UserWarning, match="1 of 9 pattern"):
            results = indexer.index_patterns(patterns, chunksize=9, progressbar=False)
        assert results["phase_id"][4, 0] == -1
        # the eight others of the one chunk survive the raise
        assert (results["phase_id"][[0, 1, 2, 3, 5, 6, 7, 8]] == 0).all()

    def test_a_nan_pixel_is_documented_as_unguarded(self, record_property):
        # measured: the unsigned 8-bit conversion swallows it and the
        # pattern indexes near normally (score 0.605 against 0.624
        # clean), so only the *score* path is guarded.  Platform
        # dependent, hence no assertion on the values
        record_property("nan_pixel", "no exception, measured score 0.605")
        patterns = np.array(ni_patterns()[:1], dtype=np.float64)
        patterns[0, 5, 5] = np.nan
        indexer = ni_indexer()
        with np.errstate(invalid="ignore"):
            results = indexer.index_patterns(patterns, progressbar=False)
        assert results["scores"].shape == (1, 1)

    def test_the_graph_metadata_is_truthful(self):
        # without an explicit ``chunks=`` the graph declares the
        # shape ``(9, 1, 1)`` -- measured -- which computes correctly
        # but lies to anything slicing before the compute
        indexer = ni_indexer()
        patterns = da.from_array(np.array(ni_patterns()), chunks=(4, -1, -1))
        results = _map_chunks(patterns, indexer, 2)
        assert results.shape == (9, 2, 6)
        assert results.chunks[0] == patterns.chunks[0]
        assert results.dtype == np.float64

    def test_the_info_message_counts_the_chunks(self):
        indexer = ni_indexer()
        assert "5 chunk(s)" in indexer.get_info_message(9, 2)

    def test_the_default_chunksize_follows_batch_estimate(self):
        # asserted with an explicit worker count, never the
        # machine's real CPU count
        indexer = ni_indexer()
        with dask.config.set(num_workers=4):
            message = indexer.get_info_message(9, None)
        assert _batch_estimate(NI_BANDWIDTH, 4, 9) == 1
        assert "9 chunk(s)" in message
        assert "1 pattern(s)" in message

    def test_the_info_message_describes_the_phase(self):
        # the D6 phase line is the name, the point group and the two
        # symmetry flags, which the memory model test does not reach
        # past the name
        message = ni_indexer().get_info_message(9, 1)
        assert "Phase(s): ni (m-3m; 4-fold, mirror)" in message

    @pytest.mark.parametrize("lazy", [False, True])
    def test_an_explicit_chunksize_reaches_the_graph(self, monkeypatch, lazy):
        # the results are bitwise identical across chunk sizes by
        # design, which is exactly what hides an ``index_patterns``
        # that quietly re-estimates the caller's chunk size from
        # every other test -- while the information message would go
        # on reporting the size which was asked for and the user's
        # only lever on the per-worker memory would be dead
        original = _indexer._index_chunk
        sizes = []

        def spy(patterns_block, indexer, n_best):
            sizes.append(int(patterns_block.shape[0]))
            return original(patterns_block, indexer, n_best)

        monkeypatch.setattr(_indexer, "_index_chunk", spy)
        patterns = np.array(ni_patterns())
        if lazy:
            patterns = da.from_array(patterns, chunks=(9, -1, -1))
        indexer = ni_indexer()
        # an explicit worker count, so that the estimate a mutant
        # would fall back to is a known one and never the machine's
        with dask.config.set(num_workers=4):
            indexer.index_patterns(patterns, chunksize=4, progressbar=False)
        # dask calls the function once on an empty block to build the
        # graph's meta, which is not a chunk of patterns
        assert sorted(size for size in sizes if size > 0) == [1, 4, 4]


# ------------------------- Memory model (D8) ------------------------ #


class TestMemoryModel:
    def test_the_model_arithmetic(self):
        # 135^2 x 68 x 24 + 135^3 x 8 bytes
        assert ni_indexer().memory_per_worker_bytes == MEMORY_MODEL_BW68

    def test_an_un_normalised_second_phase_is_free(self):
        # one shared scratch correlator serves every phase when
        # ``normalize=False``, so the phase count drops out
        one = ni_indexer(normalize=False)
        two = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()],
            ni_detector(),
            normalize=False,
        )
        assert two.memory_per_worker_bytes == one.memory_per_worker_bytes

    def test_a_normalized_second_phase_costs_one_cube(self):
        one = ni_indexer()
        two = SphericalIndexer(
            [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()], ni_detector()
        )
        delta = two.memory_per_worker_bytes - one.memory_per_worker_bytes
        assert delta == MEMORY_PER_EXTRA_PHASE

    def test_many_workers_warn(self):
        # 64 x 49.4 MB = 3.16 GB, over the 2 GiB threshold; the
        # threads still index the nine patterns fine
        indexer = ni_indexer()
        with dask.config.set(num_workers=64):
            with pytest.warns(UserWarning, match="2 GiB"):
                indexer.index_patterns(ni_patterns(), progressbar=False)


# --------------------- Exports and docs (D9) ------------------------ #


class TestExports:
    @pytest.mark.parametrize(
        "name", ["SphericalIndexer", "SphericalBackProjector", "fast_bandwidths"]
    )
    def test_the_name_resolves_through_the_lazy_loader(self, name):
        assert hasattr(kp.indexing, name)
        assert name in kp.indexing.__all__

    def test_all_is_sorted(self):
        assert list(kp.indexing.__all__) == sorted(kp.indexing.__all__)

    def test_fast_bandwidths_pin(self):
        returned = list(_fft.fast_bandwidths(16, 128))
        assert returned == FAST_BANDWIDTHS_16_128
        # the values ``nml.hpp`` suggests in this range, checked
        # against what the function returned and never against the
        # literal above, which would compare two literals
        for suggested in (53, 63, 68, 74, 88, 95, 113, 122, 123):
            assert suggested in returned

    def test_the_fast_bandwidths_docstring_is_published(self):
        doc = _fft.fast_bandwidths.__doc__
        assert "Examples" in doc
        assert "See Also" in doc
        assert "SphericalIndexer" in doc
        # ``fast_size`` stays private, so the cross-reference would
        # dangle in the public reference
        assert ":func:`fast_size`" not in doc

    def test_no_new_public_docstring_links_a_private_name(self):
        # a Sphinx role pointing at a private name renders as an
        # unresolved literal, since only ``__all__`` gets a stub.
        # ``MasterPatternHarmonics`` is excluded: its four private
        # links predate this module and are not part of this pass
        role = re.compile(r":(?:func|class|meth|attr|mod):`~?([\w.]+)`")
        docstrings = {}
        for module in (_indexer, _back_projection, _fft):
            docstrings.update(public_docstrings(module))
        docstrings["EBSD.spherical_indexing"] = (
            kp.signals.EBSD.spherical_indexing.__doc__
        )
        for name, doc in docstrings.items():
            if "MasterPatternHarmonics" in name:
                continue
            for target in role.findall(doc):
                private = [p for p in target.split(".") if p.startswith("_")]
                assert not private, f"{name} links the private {target}"

    def test_no_public_docstring_names_a_roadmap_phase(self):
        docstrings = {}
        for module in (_indexer, _back_projection, _fft):
            docstrings.update(public_docstrings(module))
        docstrings["EBSD.spherical_indexing"] = (
            kp.signals.EBSD.spherical_indexing.__doc__
        )
        assert len(docstrings) > 3
        for name, doc in docstrings.items():
            for phase in ("Phase 5", "Phase 6", "Phase 7"):
                assert phase not in doc, f"{name} names {phase}"


# --------------------- Kernel flag regression (D10) ----------------- #


class TestKernelFlags:
    def test_the_package_has_exactly_two_numpy_error_model_kernels(self):
        found = set()
        for module in spherical_modules():
            for name, kernel in njit_kernels(module).items():
                if kernel.targetoptions.get("error_model") == "numpy":
                    found.add(name)
        assert found == NUMPY_ERROR_MODEL_KERNELS

    def test_the_indexer_defines_no_kernels(self):
        assert njit_kernels(_indexer) == {}

    def test_the_indexer_makes_no_new_transform_call(self):
        # the recording test on the back-projector's transform still
        # covers every call this pipeline makes
        assert "scipy.fft" not in inspect.getsource(_indexer)


# ------------------------- Detector helpers ------------------------- #


def test_the_detector_helper_matches_the_spec():
    # a guard on the fixtures themselves, so a changed data set is
    # not mistaken for a changed indexer
    detector = ni_detector()
    assert isinstance(detector, EBSDDetector)
    assert detector.shape == (60, 60)
    assert detector.navigation_size == 1
    assert detector.sample_tilt == 70.0
    assert detector.tilt == 0.0
    assert np.allclose(
        detector.pc.squeeze(), (0.42513885, 0.21336699, 0.50070692), atol=1e-8
    )
    harmonics = ni_harmonics(NI_BANDWIDTH)
    assert harmonics.sample_tilt == 70.0
    assert harmonics.beam_energy == 20.1
    assert harmonics.phase.name == "ni"
    assert harmonics.n_fold == 4
    assert harmonics.has_equatorial_mirror is True
