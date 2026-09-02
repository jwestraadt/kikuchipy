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

"""Tests of ``kikuchipy.signals.EBSD.spherical_indexing``.

Covers the assertions of
``specs/2026-09-02-spherical-indexing-ebsd/validation.md`` which
belong to the signal method:

- Real data: the nine ``nickel_ebsd_small`` patterns against the
  stored crystal map at ``bw`` 68 and 53 (88 weekly), the pinned
  scores and image qualities, and the crystal map structure.
- The optional preprocessing paths, each with pinned values rather
  than a "differs from the default" tautology: ``n_regions=0``, the
  inscribed circle and the Gaussian background at both
  ``emsphinx_compatible`` settings.
- ``n_best`` and its one candidate per phase fill semantics.
- Multi-phase: the sign-scrambled discrimination phase, the rotated
  copy kept as a degeneracy control with the composed orientation
  identity, and the phase bookkeeping orix leaves behind.
- Per-pattern failure injection, both the zero variance guard with
  its recorded deviation and the exception arm which keeps one bad
  pattern from killing a run.
- Both mask polarities, with the pinned masked values which separate
  a not-forwarded mask from a flipped one.
- Lazy input, chunking, worker counts, data types and determinism.
- The information message and the verbosity switch.
- The throughput floor, the only loose timing assertion, and the
  recorded per-stage timings, worker scaling and per-worker memory.
- The ``nickel_ebsd_large`` subsets, which skip without ``pooch``.
"""

import functools
import inspect
import time
import tracemalloc

import dask
import numpy as np
from orix.crystal_map import CrystalMap, Phase, create_coordinate_arrays
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy.indexing._spherical import _indexer
from kikuchipy.indexing._spherical._euler import rotation_from_zyz
from kikuchipy.indexing._spherical._indexer import SphericalIndexer, _index_chunk
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._wigner import rotate_harmonics
from kikuchipy.indexing._spherical._xcorr import (
    NormalizedSphericalCrossCorrelator,
)

# The bandwidth of the real data tests
NI_BANDWIDTH = 68

# Measured sphere window of the Ni detector with the inscribed circle
CIRCLE_WINDOW_POINTS = 1117

# The roadmap's binding coarse bounds, degrees
ROADMAP_MEDIAN = 1.5
ROADMAP_LOOSE = 3.0

# Measured-then-pinned tighteners at ``bw`` 68, degrees (measured
# median 0.599 / max 0.838, so 2.0x and 2.4x of margin)
PINNED_MEDIAN = 1.2
PINNED_MAX = 2.0

# Measured scores and image qualities of the nine patterns at ``bw``
# 68 in the default configuration, ``pytest.approx(., rel=0.05)``
SCORES_BW68 = {"min": 0.4963, "max": 0.6239, "mean": 0.5701}
IQ_BW68 = {"min": 0.1727, "max": 0.2036}

# The same, un-normalised
SCORES_BW68_PLAIN = {"min": 0.2799, "max": 0.3533}

# ``n_regions=0``: the image quality band which separates a
# preprocessing-not-forwarded mutant (measured 0.2890-0.3269)
IQ_NO_EQUALISATION = (0.25, 0.40)

# ``circular_mask=True`` (measured)
SCORES_CIRCLE = {"min": 0.4915, "max": 0.6390}
IQ_CIRCLE = {"min": 0.1920, "max": 0.2224}

# ``gaussian_background=True`` with ``emsphinx_compatible=True``
SCORES_GAUSSIAN = {"min": 0.4942, "max": 0.6101}

# The two ``emsphinx_compatible`` settings differ by a measured
# 1.83e-3 on the Gaussian background path: real but far below the
# five per cent pins, so the non-identity needs its own bar
COMPATIBLE_SCORE_BAR = 1e-4

# The Phase 5 ``rDen`` block and the values it gives (measured).  The
# score pins kill a mask which never reached the preprocessing
# (0.4963-0.6239, 8-11 % off), the image quality pins kill a flipped
# ``good_pixels`` (0.2866-0.3199, about 60 % off) -- which lands
# *inside* the score pins, so they are load bearing, not decorative
SIGNAL_MASK_BLOCK = (slice(20, 32), slice(25, 40))
SCORES_SIGNAL_MASK = {"min": 0.4461, "max": 0.5762, "mean": 0.5307}
IQ_SIGNAL_MASK = {"min": 0.1740, "max": 0.2028}

# Multi-phase: the sign-scrambled decoy loses at every point by a
# measured 0.2970-0.4151, with its own scores at 0.1993-0.2194
SCRAMBLE_SEED = 42
SCRAMBLE_GAP = 0.1

# The rotated copy is degenerate by design: measured gaps -0.0151 to
# +0.0090, and the composed orientation lands at median 0.676 / max
# 1.074 degrees while the three wrong compositions sit at 24.3-28.7
ROTATION_ZYZ = (0.9, 0.7, -0.4)
DEGENERATE_GAP = 0.05
COMPOSED_TOLERANCE = 2.5
WRONG_COMPOSITION_FLOOR = 10.0

# ``nickel_ebsd_large`` subsets (measured 0.499 / 1.350 on 20 points
# and 0.530 / p95 1.082 / max 1.495 on 165)
LARGE_SCORES = (0.40, 0.70)
LARGE_MAX = 3.0
WEEKLY_P95 = 2.5
WEEKLY_MAX = 3.5

# The constitution's hard floor, patterns per second per core
THROUGHPUT_FLOOR = 2

# Loose per-worker memory bound at ``bw`` 68 (measured peak 44.9 MB)
PEAK_BOUND_BYTES = 200e6


# ----------------------------- Helpers ------------------------------ #


@functools.lru_cache(maxsize=1)
def _ni_signal_cached():
    """Return the background corrected ``nickel_ebsd_small`` signal,
    built once because the two background steps cost a tenth of a
    second each.
    """
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    return signal


def ni_signal():
    """Return a fresh deep copy of the background corrected signal,
    so a test may modify its patterns.
    """
    return _ni_signal_cached().deepcopy()


def ni_detector():
    """Return a fresh Ni detector with one projection centre, the
    average of the nine stored ones.
    """
    detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    detector.pc = detector.pc_average
    return detector


@functools.lru_cache(maxsize=8)
def ni_harmonics(bandwidth):
    """Return the harmonics of the shipped Ni master pattern built
    **directly** at ``bandwidth``, cached.

    Every pinned value assumes direct construction: resizing from a
    larger bandwidth gives coefficients which differ by up to about
    one hundred per cent on individual significant terms.
    """
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


def scrambled_harmonics(seed=SCRAMBLE_SEED):
    """Return the sign-scrambled decoy phase.

    Every ``|a_lm|`` is preserved exactly, so the power spectrum is
    genuinely the same, and the ``m = 0`` row stays real, so the
    coefficients are still those of a real spherical function -- a
    *phase* scramble is not, and its ``squared_harmonics`` would not
    match its own spectrum.
    """
    harmonics = ni_harmonics(NI_BANDWIDTH)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], harmonics.alm.shape)
    return MasterPatternHarmonics(
        harmonics.alm * signs, phase=Phase("scrambled", point_group="1")
    )


def rotated_harmonics(zyz=ROTATION_ZYZ):
    """Return a copy of the Ni master rotated by ``zyz``, the
    degeneracy control phase.
    """
    harmonics = ni_harmonics(NI_BANDWIDTH)
    alm = rotate_harmonics(harmonics.alm, np.asarray(zyz, dtype=np.float64))
    return MasterPatternHarmonics(alm, phase=Phase("rotated", point_group="1"))


def misorientation(rotations, reference):
    """Return the symmetry reduced misorientation in degrees between
    two rotation sets of the same shape, under m-3m.
    """
    angles = Orientation(rotations.data, Oh).angle_with(
        Orientation(reference.data, Oh), degrees=True
    )
    return np.asarray(angles, dtype=np.float64).ravel()


def index_default(signal=None, **kwargs):
    """Return the crystal map of the default call, i.e. the Ni
    harmonics and detector with every parameter at its default.
    """
    if signal is None:
        signal = ni_signal()
    kwargs.setdefault("verbose", 0)
    harmonics = kwargs.pop("harmonics", ni_harmonics(NI_BANDWIDTH))
    detector = kwargs.pop("detector", ni_detector())
    return signal.spherical_indexing(harmonics, detector, **kwargs)


def stored_rotations(signal=None):
    """Return the stored orientations of the small map."""
    if signal is None:
        signal = _ni_signal_cached()
    return signal.xmap.rotations


def record_angles(record_property, tag, angles):
    """Record the per-point misorientations, their median and their
    maximum, so a run's numbers survive in the report.
    """
    record_property(f"{tag}_per_point", ", ".join(f"{a:.3f}" for a in angles))
    record_property(f"{tag}_median", f"{np.median(angles):.4f}")
    record_property(f"{tag}_max", f"{angles.max():.4f}")


def assert_coarse_bounds(angles):
    """Assert the roadmap's binding coarse bounds and the
    measured-then-pinned tighteners.

    Arithmetic: the mean projection centre error floor of the small
    map is a median of 0.33 and a maximum of 0.54 degrees, and the
    interpolated coarse grid contributes 0.34 / 0.72 at this
    bandwidth, so their quadrature sum predicts about 0.47 and 0.9
    degrees.  Measured end to end: 0.599 and 0.838.
    """
    assert np.median(angles) < ROADMAP_MEDIAN
    assert int((angles < ROADMAP_LOOSE).sum()) >= angles.size - 1
    assert angles.max() < PINNED_MAX
    assert np.median(angles) < PINNED_MEDIAN


def identity_like(rotations):
    """Return the identity rotation of the same shape."""
    data = np.zeros_like(rotations.data)
    data[..., 0] = 1.0
    return Rotation(data)


# ------------------------ Signature (D5) ---------------------------- #


class TestSignature:
    def test_defaults_are_frozen(self):
        parameters = inspect.signature(kp.signals.EBSD.spherical_indexing).parameters
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        assert defaults == {
            "bandwidth": 68,
            "n_best": 1,
            "navigation_mask": None,
            "signal_mask": None,
            "normalize": True,
            "refine": False,
            "n_regions": 10,
            "gaussian_background": False,
            "circular_mask": False,
            "emsphinx_compatible": True,
            "chunksize": None,
            "verbose": 1,
        }
        # ``n_best`` keeps its name rather than dictionary indexing's
        # ``keep_n``: a row here is one candidate *per phase*
        assert "keep_n" not in parameters

    def test_the_method_sits_next_to_dictionary_indexing(self):
        names = [
            name
            for name in vars(kp.signals.EBSD)
            if name in ("dictionary_indexing", "spherical_indexing")
        ]
        assert names == ["dictionary_indexing", "spherical_indexing"]


# ---------------------- Real data, small map (D7) ------------------- #


class TestNickelSmall:
    def test_the_default_call_meets_the_coarse_bounds(self, record_property):
        xmap = index_default()
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_normalized", angles)
        assert_coarse_bounds(angles)

    def test_the_scores_and_image_qualities_are_pinned(self, record_property):
        xmap = index_default()
        scores = xmap.scores
        iq = xmap.iq
        record_property(
            "small_bw68_scores",
            f"{scores.min():.4f}-{scores.max():.4f} mean {scores.mean():.4f}",
        )
        record_property("small_bw68_iq", f"{iq.min():.4f}-{iq.max():.4f}")
        assert scores.min() == pytest.approx(SCORES_BW68["min"], rel=0.05)
        assert scores.max() == pytest.approx(SCORES_BW68["max"], rel=0.05)
        assert scores.mean() == pytest.approx(SCORES_BW68["mean"], rel=0.05)
        # the image quality is that of the *processed* pattern: a
        # raw-pattern mutant reads 0.766-0.779 and dies here, an
        # equalisation-skipping one 0.2890-0.3269
        assert iq.min() == pytest.approx(IQ_BW68["min"], rel=0.05)
        assert iq.max() == pytest.approx(IQ_BW68["max"], rel=0.05)

    def test_the_un_normalised_correlation(self, record_property):
        xmap = index_default(normalize=False)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_plain", angles)
        assert_coarse_bounds(angles)
        # a distinct range, so an indexer which normalizes anyway
        # dies here
        assert xmap.scores.min() == pytest.approx(SCORES_BW68_PLAIN["min"], rel=0.05)
        assert xmap.scores.max() == pytest.approx(SCORES_BW68_PLAIN["max"], rel=0.05)

    def test_bandwidth_53(self, record_property):
        xmap = index_default(harmonics=ni_harmonics(53), bandwidth=53)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw53", angles)
        # 1.7 half cells of margin: measured median 0.747 / max 0.991
        assert (angles < ROADMAP_LOOSE).all()

    @pytest.mark.weekly
    def test_bandwidth_88(self, record_property):
        xmap = index_default(harmonics=ni_harmonics(88), bandwidth=88)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw88", angles)
        assert (angles < ROADMAP_LOOSE).all()

    def test_the_crystal_map_structure(self):
        xmap = index_default()
        assert isinstance(xmap, CrystalMap)
        assert xmap.shape == (3, 3)
        assert xmap.scan_unit == "um"
        assert xmap.phases.names == ["ni"]
        assert xmap.phases[0].space_group.short_name == "Fm-3m"
        # the ``n_best == 1`` squeeze
        assert xmap.rotations.shape == (9,)
        assert xmap.scores.shape == (9,)
        assert xmap.scores.dtype == np.float64
        assert xmap.iq.shape == (9,)
        assert xmap.iq.dtype == np.float64
        assert (xmap.phase_id == 0).all()
        assert xmap.is_indexed.all()
        # the property is only added when there is more than one
        # candidate per point
        assert "nbest_phase_id" not in xmap.prop

    def test_the_coordinates(self):
        xmap = index_default()
        expected, _ = create_coordinate_arrays((3, 3), (1.5, 1.5))
        assert np.allclose(xmap.x, expected["x"])
        assert np.allclose(xmap.y, expected["y"])


# ------------------- Optional preprocessing (D7) -------------------- #


class TestPreprocessingPaths:
    def test_without_the_histogram_equalisation(self, record_property):
        xmap = index_default(n_regions=0)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_nregions0", angles)
        record_property(
            "small_bw68_nregions0_iq",
            f"{xmap.iq.min():.4f}-{xmap.iq.max():.4f}",
        )
        assert_coarse_bounds(angles)
        # a band distinct from the ``n_regions=10`` one, so a
        # preprocessing-not-forwarded mutant dies here
        assert xmap.iq.min() > IQ_NO_EQUALISATION[0]
        assert xmap.iq.max() < IQ_NO_EQUALISATION[1]

    def test_the_inscribed_circle(self, record_property):
        indexer = SphericalIndexer(
            ni_harmonics(NI_BANDWIDTH), ni_detector(), circular_mask=True
        )
        # the circle cuts the sphere window down from 1317 points
        assert indexer.projector.n_points == CIRCLE_WINDOW_POINTS

        xmap = index_default(circular_mask=True)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_circle", angles)
        assert_coarse_bounds(angles)
        assert xmap.scores.min() == pytest.approx(SCORES_CIRCLE["min"], rel=0.05)
        assert xmap.scores.max() == pytest.approx(SCORES_CIRCLE["max"], rel=0.05)
        # the circle reaches the histogram as well, which is what
        # lifts the image quality out of the unmasked band
        assert xmap.iq.min() == pytest.approx(IQ_CIRCLE["min"], rel=0.05)
        assert xmap.iq.max() == pytest.approx(IQ_CIRCLE["max"], rel=0.05)

    def test_the_gaussian_background(self, record_property):
        xmap = index_default(gaussian_background=True)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_gaussian", angles)
        assert_coarse_bounds(angles)
        assert xmap.scores.min() == pytest.approx(SCORES_GAUSSIAN["min"], rel=0.05)
        assert xmap.scores.max() == pytest.approx(SCORES_GAUSSIAN["max"], rel=0.05)

    def test_emsphinx_compatible_changes_the_gaussian_background(self, record_property):
        # the only path on which the flag has a numeric effect, and
        # the effect is real but far below the five per cent pins,
        # so the mutant needs this non-identity assertion
        compatible = index_default(gaussian_background=True)
        corrected = index_default(gaussian_background=True, emsphinx_compatible=False)
        angles = misorientation(corrected.rotations, stored_rotations())
        assert_coarse_bounds(angles)
        difference = np.abs(compatible.scores - corrected.scores).max()
        record_property("gaussian_compat_score_difference", f"{difference:.3e}")
        assert difference > COMPATIBLE_SCORE_BAR

    def test_the_preprocessing_keywords_are_forwarded(self, monkeypatch):
        original = _indexer._preprocess_pattern
        calls = []

        def spy(pattern, **kwargs):
            calls.append(kwargs)
            return original(pattern, **kwargs)

        monkeypatch.setattr(_indexer, "_preprocess_pattern", spy)
        index_default(gaussian_background=True, n_regions=4, emsphinx_compatible=False)
        assert len(calls) == 9
        for kwargs in calls:
            assert kwargs["gaussian_background"] is True
            assert kwargs["n_regions"] == 4
            assert kwargs["emsphinx_compatible"] is False

    def test_emsphinx_compatible_reaches_the_correlator(self, monkeypatch):
        original = NormalizedSphericalCrossCorrelator.correlate
        calls = []

        def spy(self, gln, **kwargs):
            calls.append(kwargs)
            return original(self, gln, **kwargs)

        monkeypatch.setattr(NormalizedSphericalCrossCorrelator, "correlate", spy)
        index_default(emsphinx_compatible=False)
        assert len(calls) == 9
        for kwargs in calls:
            assert kwargs["emsphinx_compatible"] is False


# ------------------------- n_best (D3) ------------------------------ #


class TestNBest:
    def test_three_candidates_of_one_phase(self):
        signal = ni_signal()
        one = index_default(signal=signal)
        three = index_default(signal=signal, n_best=3)

        assert three.rotations.shape == (9, 3)
        assert three.scores.shape == (9, 3)
        assert three.prop["nbest_phase_id"].shape == (9, 3)
        assert three.prop["nbest_phase_id"].dtype == np.int32

        # row 0 is the single candidate run, bitwise
        assert np.array_equal(three.rotations[:, 0].data, one.rotations.data)
        assert np.array_equal(three.scores[:, 0], one.scores)

        # the image quality belongs to the *pattern*, not to the
        # candidate: one column whatever ``n_best`` is, and the same
        # values as the single candidate run.  An ``(9, 3)`` iq is a
        # legal orix property, so without this the mis-packed column
        # mutant survives
        assert three.iq.shape == (9,)
        assert np.array_equal(three.iq, one.iq)

        # rows 1 and 2 keep the fill: one candidate per phase
        assert (three.scores[:, 1:] == 0.0).all()
        assert (three.prop["nbest_phase_id"][:, 0] == 0).all()
        assert (three.prop["nbest_phase_id"][:, 1:] == -1).all()
        rest = three.rotations[:, 1:]
        assert (rest.angle_with(identity_like(rest), degrees=True) == 0).all()

        # and the scores never increase along the candidate axis
        assert (np.diff(three.scores, axis=1) <= 0).all()

    def test_one_candidate_squeezes(self):
        xmap = index_default(n_best=1)
        assert xmap.rotations.shape == (9,)
        assert xmap.scores.shape == (9,)

    def test_zero_candidates_are_refused(self):
        with pytest.raises(ValueError):
            index_default(n_best=0)


# ---------------------- Multi-phase (D3) ---------------------------- #


class TestMultiPhase:
    def test_the_sign_scrambled_decoy_loses_everywhere(self, record_property):
        signal = ni_signal()
        harmonics = [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()]
        single = index_default(signal=signal)
        both = index_default(signal=signal, harmonics=harmonics, n_best=2)

        assert (both.phase_id == 0).all()
        nbest = both.prop["nbest_phase_id"]
        assert (nbest[:, 0] == 0).all()
        assert (nbest[:, 1] == 1).all()

        gaps = both.scores[:, 0] - both.scores[:, 1]
        record_property("scramble_gaps", f"{gaps.min():.4f}-{gaps.max():.4f}")
        record_property(
            "scramble_decoy_scores",
            f"{both.scores[:, 1].min():.4f}-{both.scores[:, 1].max():.4f}",
        )
        assert (gaps > SCRAMBLE_GAP).all()

        # the winning rows are the single phase run, bitwise
        assert np.array_equal(both.rotations[:, 0].data, single.rotations.data)
        assert np.array_equal(both.scores[:, 0], single.scores)

        # orix deletes a phase whose identifier never occurs, so the
        # losing phase is absent from the map; the configured list
        # survives on the indexer and in the property above
        assert both.phases.names == ["ni"]
        indexer = SphericalIndexer(harmonics, ni_detector())
        assert [p.phase.name for p in indexer.phases] == ["ni", "scrambled"]

    def test_a_rotated_copy_is_degenerate(self, record_property):
        # the peak of the cross-correlation does not change when the
        # reference is rotated, so this phase is *not* asserted to
        # lose: it is a control on the bookkeeping instead
        signal = ni_signal()
        true = index_default(signal=signal)
        rotated = index_default(signal=signal, harmonics=rotated_harmonics())
        gaps = true.scores - rotated.scores
        record_property("rotated_gaps", f"{gaps.min():.4f}-{gaps.max():.4f}")
        assert np.abs(gaps).max() < DEGENERATE_GAP

    def test_the_composed_orientation_identity(self, record_property):
        # an independent pin of the rotation convention and of the
        # per-phase bookkeeping, which does not use the stored map
        signal = ni_signal()
        ori_a = index_default(signal=signal).rotations
        ori_b = index_default(signal=signal, harmonics=rotated_harmonics()).rotations
        rotation = rotation_from_zyz(np.asarray(ROTATION_ZYZ, dtype=np.float64))

        right = misorientation(rotation * ori_b, ori_a)
        record_angles(record_property, "composed_orientation", right)
        assert (right < COMPOSED_TOLERANCE).all()

        wrong = {
            "conjugated": misorientation(~rotation * ori_b, ori_a),
            "reversed": misorientation(ori_b * rotation, ori_a),
            "reversed_conjugated": misorientation(ori_b * ~rotation, ori_a),
        }
        for tag, angles in wrong.items():
            record_property(f"composed_wrong_{tag}", f"{np.median(angles):.3f}")
            assert np.median(angles) > WRONG_COMPOSITION_FLOOR

    def test_two_phases_are_ordered_by_score(self):
        xmap = index_default(
            harmonics=[ni_harmonics(NI_BANDWIDTH), scrambled_harmonics()],
            n_best=2,
        )
        assert (np.diff(xmap.scores, axis=1) <= 0).all()

    def test_a_phase_less_harmonics_is_refused(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        anonymous = MasterPatternHarmonics(harmonics.alm)
        with pytest.raises(ValueError, match="phase"):
            index_default(harmonics=anonymous)
        # the indexer itself accepts it: only the phase list needs a
        # phase
        indexer = SphericalIndexer(anonymous, ni_detector())
        results = indexer.index_patterns(
            ni_signal().data.reshape((-1, 60, 60))[:1], progressbar=False
        )
        assert results["phase_id"][0, 0] == 0

    def test_duplicate_phase_names_are_refused(self):
        harmonics = ni_harmonics(NI_BANDWIDTH)
        twin = MasterPatternHarmonics(
            harmonics.alm,
            phase=harmonics.phase,
            sample_tilt=harmonics.sample_tilt,
            beam_energy=harmonics.beam_energy,
        )
        with pytest.raises(ValueError):
            index_default(harmonics=[harmonics, twin])


# -------------------- Failure injection (D2) ------------------------ #


class TestFailureInjection:
    @pytest.mark.parametrize("value", [37, 0])
    def test_a_zero_variance_pattern_is_failed(self, value, record_property):
        record_property(
            "deviation",
            "through EMSphInx' own path a constant 37 pattern indexes "
            "at score 0.2301 off the O(1e-13) equalisation ripple with "
            "a garbage orientation -- the recorded deviation; a "
            "constant float with n_regions=0 correlates the window "
            "mask at -2.64, which EMSphInx' own positive-score "
            "insertion rule then drops, so that case is parity, "
            "failed earlier",
        )
        clean = index_default()

        signal = ni_signal()
        signal.data[1, 1] = np.full((60, 60), value, np.uint8)
        broken = index_default(signal=signal)

        # the flat point fails with the exact contract values
        assert not broken.is_indexed[4]
        assert broken.phase_id[4] == -1
        assert broken.scores[4] == 0.0
        assert broken.iq[4] == 0.0
        failed = broken.rotations[4]
        assert (failed.angle_with(identity_like(failed), degrees=True) == 0).all()

        # and the other eight are bitwise what they were
        keep = np.array([0, 1, 2, 3, 5, 6, 7, 8])
        assert np.array_equal(broken.rotations[keep].data, clean.rotations[keep].data)
        assert np.array_equal(broken.scores[keep], clean.scores[keep])
        assert np.array_equal(broken.iq[keep], clean.iq[keep])

    def test_a_raising_pattern_fails_alone(self, monkeypatch, record_property):
        # failure case (d), the exception arm: the per-pattern body is
        # wrapped, so one pattern which raises fails by itself.  The
        # mutant which drops the ``try``/``except`` takes the whole
        # run down and dies here; nothing else reaches this arm, since
        # the other injections *return* a value
        record_property(
            "exception_arm",
            "an exception raised for one pattern fails that point only, "
            "with the same fill as the guards",
        )
        signal = ni_signal()
        clean = index_default(signal=signal)

        # the injection keys on the pattern's own bytes rather than on
        # a call counter, so it does not depend on the chunking or on
        # the order the threads take the patterns in
        target = np.array(signal.data[1, 1])
        matching = [
            index
            for index, pattern in enumerate(signal.data.reshape((-1, 60, 60)))
            if np.array_equal(pattern, target)
        ]
        assert matching == [4]

        original = _indexer._preprocess_pattern

        def exploding(pattern, **kwargs):
            if np.array_equal(pattern, target):
                raise RuntimeError("injected per-pattern failure")
            return original(pattern, **kwargs)

        monkeypatch.setattr(_indexer, "_preprocess_pattern", exploding)
        broken = index_default(signal=signal)

        assert not broken.is_indexed[4]
        assert broken.phase_id[4] == -1
        assert broken.scores[4] == 0.0
        assert broken.iq[4] == 0.0
        failed = broken.rotations[4]
        assert (failed.angle_with(identity_like(failed), degrees=True) == 0).all()

        # and the eight which did not raise are bitwise the clean run
        keep = np.array([0, 1, 2, 3, 5, 6, 7, 8])
        assert np.array_equal(broken.rotations[keep].data, clean.rotations[keep].data)
        assert np.array_equal(broken.scores[keep], clean.scores[keep])
        assert np.array_equal(broken.iq[keep], clean.iq[keep])


# ---------------------------- Masks (D5) ---------------------------- #


class TestMasks:
    def test_the_navigation_mask_polarity(self, capsys):
        signal = ni_signal()
        unmasked = index_default(signal=signal)

        mask = np.ones((3, 3), dtype=bool)
        mask.ravel()[[0, 4, 8]] = False
        masked = index_default(signal=signal, navigation_mask=mask, verbose=1)
        message = capsys.readouterr().out
        assert "3 pattern(s)" in message

        # only the ``False`` entries are indexed
        assert masked.is_in_data.sum() == 3
        assert np.array_equal(masked.is_in_data, ~mask.ravel())
        assert np.array_equal(masked.rotations.data, unmasked.rotations[[0, 4, 8]].data)
        assert np.array_equal(masked.scores, unmasked.scores[[0, 4, 8]])
        assert np.array_equal(masked.iq, unmasked.iq[[0, 4, 8]])

        # the masked out points carry the deterministic fill, which
        # is why orix prepends a not-indexed phase
        assert masked.phases.names[0] == "not_indexed"
        masked.is_in_data = np.ones(9, dtype=bool)
        skipped = np.array([1, 2, 3, 5, 6, 7])
        assert (masked.phase_id[skipped] == -1).all()
        assert (masked.scores[skipped] == 0.0).all()
        assert (masked.iq[skipped] == 0.0).all()
        filled = masked.rotations[skipped]
        assert (filled.angle_with(identity_like(filled), degrees=True) == 0).all()

    def test_an_all_true_navigation_mask_is_refused(self):
        with pytest.raises(ValueError, match="at least one value equal to"):
            index_default(navigation_mask=np.ones((3, 3), dtype=bool))

    def test_a_navigation_mask_of_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="navigation"):
            index_default(navigation_mask=np.zeros((2, 3), dtype=bool))

    def test_a_navigation_mask_which_is_not_an_array_is_refused(self):
        with pytest.raises(ValueError, match="NumPy array"):
            index_default(navigation_mask=[[False] * 3] * 3)

    def test_a_non_boolean_navigation_mask_is_refused(self):
        # the flow computes ``~navigation_mask``, and the bitwise NOT
        # of an integer 0/1 mask is truthy everywhere, which would
        # silently index every point.
        #
        # **The dtype check must run before the all-``True`` check**:
        # ``np.ones((3, 3), dtype=int).all()`` is ``True``, so an
        # integer mask of ones reaches the all-``True`` branch first
        # and raises "at least one value equal to `False`" instead of
        # this message.  The order which satisfies all four mask tests
        # is: is-ndarray (a list has no ``dtype``), dtype, shape,
        # all-``True``
        with pytest.raises(
            ValueError, match="The navigation mask must be a boolean array"
        ):
            index_default(navigation_mask=np.ones((3, 3), dtype=int))

    def test_the_signal_mask_values_are_pinned(self, record_property):
        mask = np.zeros((60, 60), dtype=bool)
        mask[SIGNAL_MASK_BLOCK] = True
        xmap = index_default(signal_mask=mask)
        angles = misorientation(xmap.rotations, stored_rotations())
        record_angles(record_property, "small_bw68_signal_mask", angles)
        record_property(
            "small_bw68_signal_mask_scores",
            f"{xmap.scores.min():.4f}-{xmap.scores.max():.4f} "
            f"mean {xmap.scores.mean():.4f}",
        )
        assert_coarse_bounds(angles)
        # a mask which never reached the preprocessing reads
        # 0.4963-0.6239 and dies on these
        assert xmap.scores.min() == pytest.approx(SCORES_SIGNAL_MASK["min"], rel=0.05)
        assert xmap.scores.max() == pytest.approx(SCORES_SIGNAL_MASK["max"], rel=0.05)
        assert xmap.scores.mean() == pytest.approx(SCORES_SIGNAL_MASK["mean"], rel=0.05)
        # a flipped ``good_pixels`` lands inside the score pins but
        # reads 0.2866-0.3199 here, so these are load bearing
        assert xmap.iq.min() == pytest.approx(IQ_SIGNAL_MASK["min"], rel=0.05)
        assert xmap.iq.max() == pytest.approx(IQ_SIGNAL_MASK["max"], rel=0.05)

    def test_the_signal_mask_reaches_the_projector(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[SIGNAL_MASK_BLOCK] = True
        indexer = SphericalIndexer(
            ni_harmonics(NI_BANDWIDTH), ni_detector(), signal_mask=mask
        )
        assert indexer.projector.signal_mask is not None
        assert np.array_equal(indexer.projector.signal_mask, mask)
        assert indexer.good_pixels is not None
        assert np.array_equal(indexer.good_pixels, ~mask)

    @pytest.mark.parametrize(
        "mask",
        [np.zeros((60, 59), dtype=bool), np.zeros((60, 60), dtype=int)],
    )
    def test_a_bad_signal_mask_is_refused_by_the_projector(self, mask):
        with pytest.raises(ValueError):
            index_default(signal_mask=mask)


# ------------------------- Signal guards (D5) ----------------------- #


class TestSignalGuards:
    def test_a_detector_of_another_shape_is_refused(self):
        detector = ni_detector()
        detector.shape = (60, 59)
        with pytest.raises(ValueError) as info:
            index_default(detector=detector)
        message = str(info.value)
        assert "(60, 59)" in message
        assert "(60, 60)" in message

    def test_refine_is_refused(self):
        with pytest.raises(NotImplementedError, match="refine=True"):
            index_default(refine=True)


# ------------- Lazy input, chunking and determinism (D4) ------------ #


class TestLazyAndDeterminism:
    def test_a_lazy_signal_gives_the_same_map(self):
        signal = ni_signal()
        eager = index_default(signal=signal)
        lazy = index_default(signal=signal.as_lazy())
        assert np.array_equal(lazy.rotations.data, eager.rotations.data)
        assert np.array_equal(lazy.scores, eager.scores)
        assert np.array_equal(lazy.iq, eager.iq)
        assert np.array_equal(lazy.phase_id, eager.phase_id)

    # never the reference's own 9, which would only re-test run to run
    # determinism; 1 and 4 against 9 give the three pairs by
    # transitivity
    @pytest.mark.parametrize("chunksize", [1, 4])
    def test_the_chunk_size_does_not_change_the_result(self, chunksize):
        signal = ni_signal()
        reference = index_default(signal=signal, chunksize=9)
        other = index_default(signal=signal, chunksize=chunksize)
        assert np.array_equal(other.rotations.data, reference.rotations.data)
        assert np.array_equal(other.scores, reference.scores)
        assert np.array_equal(other.iq, reference.iq)

    def test_the_worker_count_does_not_change_the_result(self):
        signal = ni_signal()
        with dask.config.set(num_workers=1):
            one = index_default(signal=signal, chunksize=1)
        with dask.config.set(num_workers=4):
            four = index_default(signal=signal, chunksize=1)
        assert np.array_equal(four.rotations.data, one.rotations.data)
        assert np.array_equal(four.scores, one.scores)
        assert np.array_equal(four.iq, one.iq)

    def test_an_explicit_chunk_size_is_honoured(self, capsys):
        index_default(chunksize=2, verbose=1)
        assert "5 chunk(s)" in capsys.readouterr().out


class TestDtypes:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_the_input_data_type_does_not_change_the_result(self, dtype):
        signal = ni_signal()
        reference = index_default(signal=signal)
        converted = ni_signal()
        converted.data = converted.data.astype(dtype)
        other = index_default(signal=converted)
        assert np.array_equal(other.rotations.data, reference.rotations.data)
        assert np.array_equal(other.scores, reference.scores)
        assert np.array_equal(other.iq, reference.iq)
        # and the output types never follow the input's: literals,
        # never the reference's own dtypes, which would compare the
        # implementation to itself.  ``phase_id`` is only pinned to an
        # integer kind because orix casts it with ``.astype(int)``,
        # i.e. to the platform's default integer (measured int64 here
        # on orix 0.14.2); the int32 pin of the contract lives on the
        # ``nbest_phase_id`` property and on ``index_patterns``
        assert other.scores.dtype == np.float64
        assert other.iq.dtype == np.float64
        assert other.phase_id.dtype.kind == "i"


# ---------------- Information message and verbosity (D6) ------------ #


class TestVerbose:
    def test_verbose_zero_is_silent(self, capsys):
        index_default(verbose=0)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_verbose_one_prints_the_message_and_the_speed(self, capsys):
        index_default(verbose=1)
        out = capsys.readouterr().out
        for part in (
            "Spherical indexing information:",
            "ni",
            "68",
            "pattern(s)",
            "chunk",
            "Indexing speed:",
        ):
            assert part in out

    def test_the_message_prints_the_memory_model(self):
        # the *model*, not the measured peak: the first draft of the
        # template printed 45 MB, the measurement, and no test could
        # have caught the divergence
        indexer = SphericalIndexer(ni_harmonics(NI_BANDWIDTH), ni_detector())
        message = indexer.get_info_message(9, 1)
        assert "Estimated memory per worker: 49 MB" in message
        assert "Spherical indexing information:" in message
        assert "Phase(s): ni" in message
        assert "Correlation: normalized" in message


# ------------------ Performance and memory (D8) --------------------- #


def pipeline_stages(indexer, correlator, pattern, buffers):
    """Return the wall time of every stage of one pattern and the
    pattern's spectrum.

    The four stages are the frozen ones of the module documentation --
    preprocess, back-project, analyse, correlate -- walked here with
    the indexer's own collaborators, because ``_index_chunk`` reports
    no per-stage split.  Only the D8 records and the resident memory
    measurement use this: no assertion depends on its products, which
    the real data tests pin through the method instead.
    """
    times = {}

    start = time.perf_counter()
    processed = _indexer._preprocess_pattern(
        pattern,
        good_pixels=indexer.good_pixels,
        gaussian_background=indexer.gaussian_background,
        n_regions=indexer.n_regions,
        emsphinx_compatible=indexer.emsphinx_compatible,
    )
    times["preprocess"] = time.perf_counter() - start

    start = time.perf_counter()
    north, south, _ = indexer.projector.unproject(
        processed, out=buffers, return_image_quality=True
    )
    times["unproject"] = time.perf_counter() - start

    start = time.perf_counter()
    gln = indexer.projector.sht.analyze(north, south)
    times["analyze"] = time.perf_counter() - start

    start = time.perf_counter()
    correlator.correlate(gln, emsphinx_compatible=indexer.emsphinx_compatible)
    times["correlate"] = time.perf_counter() - start

    return times, gln


class TestPerformance:
    def test_the_throughput_floor(self, record_property):
        # the single loose timing assertion of the suite: one thread,
        # warm, the nine patterns through ``index_patterns``
        indexer = SphericalIndexer(ni_harmonics(NI_BANDWIDTH), ni_detector())
        patterns = ni_signal().data.reshape((-1, 60, 60))
        with dask.config.set(num_workers=1):
            indexer.index_patterns(patterns, chunksize=9, progressbar=False)
            start = time.perf_counter()
            indexer.index_patterns(patterns, chunksize=9, progressbar=False)
            elapsed = time.perf_counter() - start
        per_second = 9 / elapsed
        record_property("patterns_per_second_per_core_bw68", f"{per_second:.1f}")
        assert per_second >= THROUGHPUT_FLOOR

    def test_the_four_worker_throughput_is_recorded(self, record_property):
        # recorded, never asserted: the scaling of this machine is not
        # a contract, and the floor above is the one timing bound
        indexer = SphericalIndexer(ni_harmonics(NI_BANDWIDTH), ni_detector())
        patterns = ni_signal().data.reshape((-1, 60, 60))
        with dask.config.set(num_workers=4):
            indexer.index_patterns(patterns, chunksize=1, progressbar=False)
            start = time.perf_counter()
            results = indexer.index_patterns(patterns, chunksize=1, progressbar=False)
            elapsed = time.perf_counter() - start
        record_property("patterns_per_second_four_workers_bw68", f"{9 / elapsed:.1f}")
        assert results["scores"].shape == (9, 1)

    @staticmethod
    def _stage_sweep(bandwidth):
        """Return the milliseconds per pattern of every stage and the
        patterns per second of the whole pipeline at ``bandwidth``,
        best of three sweeps of the nine patterns on one thread.
        """
        indexer = SphericalIndexer(
            ni_harmonics(bandwidth), ni_detector(), bandwidth=bandwidth
        )
        correlator = indexer.correlators[0].clone()
        dim = indexer.projector.dim
        buffers = (np.zeros((dim, dim)), np.zeros((dim, dim)))
        patterns = ni_signal().data.reshape((-1, 60, 60))

        best = None
        for _ in range(3):
            totals = {}
            for pattern in patterns:
                times, _ = pipeline_stages(indexer, correlator, pattern, buffers)
                for stage, seconds in times.items():
                    totals[stage] = totals.get(stage, 0.0) + seconds
            total = sum(totals.values())
            # the first sweep pays the lazily compiled paths
            if best is None or total < best[1]:
                best = (totals, total)

        totals, total = best
        n_patterns = patterns.shape[0]
        milliseconds = {
            stage: 1e3 * seconds / n_patterns for stage, seconds in totals.items()
        }
        return milliseconds, n_patterns / total

    @staticmethod
    def _record_stages(record_property, bandwidth, milliseconds, per_second):
        record_property(
            f"stage_ms_per_pattern_bw{bandwidth}",
            ", ".join(f"{stage} {ms:.2f}" for stage, ms in milliseconds.items())
            + f", total {sum(milliseconds.values()):.2f}",
        )
        record_property(
            f"pipeline_patterns_per_second_bw{bandwidth}", f"{per_second:.1f}"
        )

    @pytest.mark.parametrize("bandwidth", [53, 68])
    def test_the_stage_timings_are_recorded(self, bandwidth, record_property):
        # a determination, not a bound (measured at ``bw`` 68: 0.22 /
        # 0.21 / 0.46-0.49 / 11.8-12.5 ms per pattern, so the
        # correlation is about 93 % of the total)
        milliseconds, per_second = self._stage_sweep(bandwidth)
        self._record_stages(record_property, bandwidth, milliseconds, per_second)
        assert set(milliseconds) == {
            "preprocess",
            "unproject",
            "analyze",
            "correlate",
        }

    @pytest.mark.weekly
    def test_the_stage_timings_at_the_large_bandwidth(self, record_property):
        milliseconds, per_second = self._stage_sweep(88)
        self._record_stages(record_property, 88, milliseconds, per_second)
        assert set(milliseconds) == {
            "preprocess",
            "unproject",
            "analyze",
            "correlate",
        }

    @staticmethod
    def _chunk_memory(bandwidth):
        """Return the model, the transient peak of one ``_index_chunk``
        call and the resident cost of one chunk kit at ``bandwidth``.

        :func:`tracemalloc.get_traced_memory`'s ``current`` right after
        ``_index_chunk`` returns is **not** the resident cost: the kit
        is local to that call, so only the small packed result array
        survives it and ``current`` reads 0.0 MB.  The resident
        numbers are therefore measured with a kit of the documented
        composition -- one correlator clone per phase plus the zeroed
        north and south pair -- held alive, once after the clone and
        once after a correlation has allocated its cube.  Both
        measurements are wrapped in ``try``/``finally``, so a raise
        never leaves :mod:`tracemalloc` running for the rest of the
        session and perturbing the other rows.
        """
        indexer = SphericalIndexer(
            ni_harmonics(bandwidth), ni_detector(), bandwidth=bandwidth
        )
        patterns = ni_signal().data.reshape((-1, 60, 60))[:1]
        dim = indexer.projector.dim

        # warm every lazily compiled path first, so the measurement
        # sees allocations and not compilation
        _index_chunk(patterns, indexer, 1)
        tracemalloc.start()
        try:
            _index_chunk(patterns, indexer, 1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        # a spectrum to correlate, built outside the traced region
        _, gln = pipeline_stages(
            indexer,
            indexer.correlators[0].clone(),
            patterns[0],
            (np.zeros((dim, dim)), np.zeros((dim, dim))),
        )
        tracemalloc.start()
        try:
            kit = [correlator.clone() for correlator in indexer.correlators]
            kit += [np.zeros((dim, dim)), np.zeros((dim, dim))]
            resident = tracemalloc.get_traced_memory()[0]
            kit[0].correlate(gln, emsphinx_compatible=indexer.emsphinx_compatible)
            after_correlate = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        del kit

        return indexer.memory_per_worker_bytes, peak, resident, after_correlate

    @staticmethod
    def _record_memory(record_property, bandwidth, model, peak, resident, correlated):
        record_property(
            f"chunk_memory_bw{bandwidth}",
            f"kit resident {resident / 1e6:.1f} MB, after correlate "
            f"{correlated / 1e6:.1f} MB, peak {peak / 1e6:.1f} MB, model "
            f"{model / 1e6:.1f} MB, model/peak {model / peak:.2f}",
        )

    @pytest.mark.parametrize("bandwidth", [63, 68])
    def test_the_memory_model_matches_the_measurement(self, bandwidth, record_property):
        model, peak, resident, correlated = self._chunk_memory(bandwidth)
        self._record_memory(
            record_property, bandwidth, model, peak, resident, correlated
        )
        # the model sits about ten per cent above the peak (measured
        # 8.9-10.2 % over the five bandwidths of the table), so this
        # loose factor of two catches a structurally wrong model
        # without pinning the allocator
        assert 0.5 < model / peak < 2

    @pytest.mark.weekly
    @pytest.mark.parametrize("bandwidth", [88, 113])
    def test_the_memory_model_at_the_large_bandwidths(self, bandwidth, record_property):
        model, peak, resident, correlated = self._chunk_memory(bandwidth)
        self._record_memory(
            record_property, bandwidth, model, peak, resident, correlated
        )
        assert 0.5 < model / peak < 2

    def test_the_per_chunk_peak_stays_small(self):
        # a loose bound with 4.5x of margin on the measured 44.9 MB
        peak = self._chunk_memory(NI_BANDWIDTH)[1]
        assert peak < PEAK_BOUND_BYTES


# ------------------- nickel_ebsd_large subsets (D7) ----------------- #


class TestNickelLargeSubset:
    @staticmethod
    def _large_subset(step, record_property, tag):
        pytest.importorskip("pooch")
        signal = kp.data.nickel_ebsd_large(allow_download=True)
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        detector = signal.detector.deepcopy()
        detector.pc = detector.pc_average

        nav_shape = signal.axes_manager.navigation_shape[::-1]
        mask = np.ones(nav_shape, dtype=bool)
        mask[::step, ::step] = False
        keep = np.flatnonzero(~mask.ravel())

        xmap = signal.spherical_indexing(
            ni_harmonics(NI_BANDWIDTH),
            detector,
            navigation_mask=mask,
            verbose=0,
        )
        angles = misorientation(xmap.rotations, signal.xmap.rotations[keep])
        record_angles(record_property, tag, angles)
        record_property(
            f"{tag}_scores",
            f"{xmap.scores.min():.4f}-{xmap.scores.max():.4f}",
        )
        return xmap, angles

    def test_the_twenty_point_subset(self, record_property):
        xmap, angles = self._large_subset(15, record_property, "large_20pt")
        assert angles.size == 20
        assert np.median(angles) < ROADMAP_MEDIAN
        assert angles.max() < LARGE_MAX
        assert xmap.scores.min() > LARGE_SCORES[0]
        assert xmap.scores.max() < LARGE_SCORES[1]

    @pytest.mark.weekly
    def test_the_one_hundred_and_sixty_five_point_subset(self, record_property):
        _, angles = self._large_subset(5, record_property, "large_165pt")
        assert angles.size == 165
        record_property("large_165pt_above_two_degrees", str(int((angles > 2).sum())))
        assert np.median(angles) < ROADMAP_MEDIAN
        assert np.percentile(angles, 95) < WEEKLY_P95
        assert angles.max() < WEEKLY_MAX
