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

"""Oracle V5 of ``specs/2026-09-07-hrebsd-dic/validation.md``: the
Si-wafer noise-floor benchmark.

ADDED 2026-09-07 at the Stage B failing-tests adversarial review.
Plan section 3.3 makes V5 a Stage B deliverable -- "first executed
here" -- and it owns plan open question 10, the D4.1 band-pass
default re-pin; the drafted Stage B commit shipped none of it.  This
module supplies the four tests validation V5 names plus the three
sweep harnesses of plan open questions 2 to 4 and 10.

**This is the feature's only REAL-DATA gate.**  Every other oracle in
this suite is synthetic, with an answer known exactly by
construction, and therefore says nothing about how the engine behaves
on a real detector's noise, drift and optical distortion.  A
nominally strain-free single crystal is the one measurement that
does: whatever it reports IS the noise floor, and the honest headline
numbers of this feature come from here.

**WHAT THIS DATASET CAN AND CANNOT SUPPORT, MEASURED 2026-09-08 at
the Stage B implementation gate** (validation.md Recorded results,
Stage B implementation gate entries 43 and 44; requirements D4.1 and
D13 amended with the same date).  The sentence above -- "whatever it
reports IS the noise floor" -- holds only where the correlation has
diffraction signal to lock onto, and on ``kp.data.si_wafer()`` it does
not.  Three measurements, none of them ambiguous:

* The ENGINE is not at fault.  A synthetic warp of a REAL wafer
  pattern, refitted through the Stage A V2 oracle, comes back to
  0.033 px worst (raw) and 0.052 px (map-average subtracted), against
  0.0073 px on the Stage A synthetic reference and a pinned
  ``WARP_REFIT_TOL_480`` of 0.025 px.  All twelve cases converge.
* The PAIRS are.  Two DIFFERENT wafer patterns, band-passed at the
  frozen ``(0.05, None)``, correlate at ZNCC 0.842 (adjacent), 0.773
  (five steps) and 0.749 (opposite corners of an 11 by 11 block at
  the map centre) -- but subtract the map
  average and the same pairs fall to 0.486, 0.253 and 0.191, while a
  synthetic warp of one pattern against itself stays at 0.978.  What
  survives the band-pass and does NOT move with the beam pins the
  correlation at zero shift: ``phase_cross_correlation`` returns
  EXACTLY 0.0 px for every pair tested, corners included, and the
  fitted translations come back at a median of 0.026 px (largest
  1.88 px) where the beam-scan model puts 20 px.  The drift is real
  and independently measured: kikuchipy's own Hough projection-centre
  fit of THIS map (doc/tutorials/pc_fit_plane.ipynb) reports a PCx
  standard deviation of 0.0241 over a grid spanning it, 11.6 detector
  pixels, and a deviation from its own fitted plane of only 0.0021.
* Subtracting the map average is NOT the remedy on a single-crystal
  scan, because there the map average IS the Kikuchi pattern: doing it
  removes the signal, and 11 of 100 fits then converge instead of 87.

So the floors recorded below are this DATASET's floors, 1.2e-02 strain
against a 1e-4 to 2e-4 literature class, and they must never be quoted
as the method's precision.  The dataset was acquired for
projection-centre calibration, which reads band POSITIONS and is
untroubled by a fixed-pattern component; high angular resolution DIC
reads band SHAPES and is not.  Plan open question 13's Si-indent
dataset is the real-data application the plan defers to after Stage C,
and it is where a method-level floor can be measured.

**It is [download] gated and skips cleanly.**  ``kp.data.si_wafer()``
is a pooch collection member (311 MB zipped, about 581 MB unzipped,
CC BY 4.0), and neither pooch nor the network is a required
dependency (tech-stack: pooch is optional, in the ``tests`` extra).
Nothing here downloads anything: the tests skip with a message naming
the one command that populates the cache, so that the download is a
deliberate act of whoever discharges the gate and never a surprise in
somebody's test run.

**Recorded, mostly not gated.**  The floors are MEASURED-THEN-PINNED:
they are recorded at the Stage B implementation gate in
validation.md, with the machine and the recipe, and only then do the
placeholders below become regression bands.  The sweeps are recorded
and NOT gated at all, the precedent of validation V4's
``test_rotation_sweep_capture_range``: a sweep's job is to produce
the table that a dated default re-pin rests on, not to fail.

Written failing before the implementation, at the Stage B
failing-tests gate: every test which calls the Stage B modules failed
with ``NotImplementedError`` once the data was cached, and skipped
before that (narration corrected to the past tense 2026-09-08, Stage B
adversarial review).
"""

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hrebsd._kam import hrebsd_kam
from kikuchipy.indexing._hrebsd._pc_shift import hrebsd_pc_shift
from kikuchipy.indexing._hrebsd._tensors import hrebsd_strain_stress

# ------------------------- Frozen constants ------------------------- #

# The default-suite subset of validation's Local-gated section: every
# fifth point of the 50 by 50 map, so 100 patterns rather than 2500
SMOKE_STEP = 5

# Silicon single-crystal elastic constants in GPa (Hopcroft, Nix and
# Kenny 2010), TEST DATA only: requirements D9.4 freezes that no
# elastic constant database ships with kikuchipy
SILICON_CUBIC = {"c11": 165.7, "c12": 63.9, "c44": 79.6}

# The detector pixel size of the NORDIF UF-420 the wafer was acquired
# on, in micrometres, unbinned at the full (480, 480) resolution.
#
# ADDED 2026-09-08 at the Stage B implementation gate, and the reason
# is a MEASUREMENT.  ``kp.data.si_wafer()`` ships a detector carrying
# kikuchipy's PLACEHOLDER ``px_size = 1.0``, which requirements D6.1
# records (2026-09-07, validation Recorded results entries 20 and 26)
# as a placeholder and explicitly as THE CALLER'S RESPONSIBILITY --
# documented rather than guarded, because a placeholder 1.0 cannot be
# told apart from a genuine 1 um pixel.  This module is that caller and
# did not discharge it: with 1.0 and the wafer's 40 um scan step the
# beam-scan model puts the projection centre 1800 px across and the
# detector distance 616 px through on a 480 px detector, and the
# measured strain floor comes back at 1.033, which FAILS this module's
# own frozen order check ``strain_floor < 100 * LITERATURE_STRAIN
# _FLOOR``.  The number itself is kikuchipy's own, for this very
# dataset: doc/tutorials/pc_fit_plane.ipynb, which loads
# ``kp.data.si_wafer()``, states "The NORDIF UF-420 (unbinned) detector
# pixel size is about 90 um" and derives the expected 2000 / 90 = 22 px
# projection-centre shift across the scan.  MEASURED with it: 20.0 px
# of modelled PCx drift and a strain floor of 1.2e-02.
UF420_PX_SIZE = 90.0

# The literature context of the strain floor at 480 px, RECORDED and
# never presumed: 1e-4 to 2e-4 class (Wilkinson, Meaden and Dingley
# 2006; Ruggles 2018).  It is quoted in the skip-free assertions only
# as an ORDER check, so that a floor which comes back at 1e-1 or at
# 1e-12 is caught as a wiring error rather than recorded as a result
LITERATURE_STRAIN_FLOOR = 2e-4


# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders (requirements D19).  Each is replaced by a
# dated measured value at the Stage B implementation gate, with the
# recipe and the machine recorded in validation.md "Recorded
# results", and only then guards regressions.

# MTP [D2-D9, V5]: the worst per-component standard deviation of the
# strain over the nominally strain-free wafer, default knobs, on the
# smoke sub-grid.  Literature context is the 1e-4 to 2e-4 class at
# 480 px -- recorded, not presumed.
# MEASURING RECIPE: ``TestNoiseFloor::test_si_noise_floor_default
# _route`` below
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation.md Recorded results, Stage B implementation gate entry
# 43).  MEASURED 1.2168e-02 on the smoke sub-grid and 1.2208e-02 on
# the full 50 by 50 map -- the two agree to 0.3 per cent, so the
# hundred-point sub-grid is not the noisy estimate the full-map arm
# was added to guard against -- pinned at 2x the larger.
#
# READ THE MODULE DOCSTRING BEFORE QUOTING THIS.  It is two orders
# ABOVE the 1e-4 to 2e-4 literature class, and it is a property of
# THIS DATASET and not of the method: the engine recovers a synthetic
# warp of a real wafer pattern to 0.033 px (entry 44), while two
# different wafer patterns band-passed at the frozen cutoffs correlate
# at only 0.842 adjacent and 0.749 ten steps apart, and what survives
# the band-pass does not move with the beam.
SI_STRAIN_FLOOR = 2.5e-02

# MTP [D8, V5]: the worst per-component standard deviation of the
# rotation vector in radians over the same map and the same run.
# MEASURING RECIPE: the same test
#
# PINNED 2026-09-08 as above.  MEASURED 1.2006e-02 rad, pinned at 2x.
# The same caveat applies: the HR rotation floor of a working
# measurement is the 5e-5 to 1e-4 rad class this module's KAM note
# quotes, and 1.2e-02 rad is 120x that.
SI_ROTATION_FLOOR = 2.5e-02

# MTP [D12, V5]: the median HR-KAM of the same map in milliradians.
# The theory expectation is ``sigma_omega * sqrt(2 / N)``, the
# 0.05 to 0.1 mrad class for an order-1 kernel at this rotation
# floor.
# MEASURING RECIPE: ``TestKamFloor::test_si_kam_floor``
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation entry 43).  MEASURED 5.2402 mrad, pinned at 2x.  The
# theory expectation quoted above -- ``sigma_omega * sqrt(2 / N)``,
# the 0.05 to 0.1 mrad class -- is for a rotation floor of 5e-5 to
# 1e-4 rad; at the 1.2e-02 rad this dataset actually delivers the
# same identity predicts 1.2e-02 * sqrt(2/8) = 6.0e-03 rad = 6.0 mrad,
# which the measurement reproduces to 13 per cent.  So the KAM is
# consistent with its own rotation floor and carries no separate
# defect; it is high for the same reason the rotation floor is.
# Recorded: at 2x the measurement this band sits just ABOVE the
# test's own frozen ``floor < 10.0`` order check, so on this dataset
# it is that assertion, not this pin, which catches a regression
# first.  The pin keeps the 2x convention rather than being tightened
# to beat it, because a pin is a record of a measurement.
SI_KAM_FLOOR = 1.05e01

# MTP [D13, V5]: the worst absolute MEAN of the PC-shift residual
# maps in pixels, with the per-point projection centres taken from
# ``extrapolate_pc``.  Requirements D13 says a nonzero mean is the
# miscalibration signature, so what is pinned is how close to zero a
# consistent geometry gets.
# MEASURING RECIPE: ``TestPcShiftPlane::test_si_pc_shift_plane``
#
# PINNED 2026-09-08 (Stage B implementation gate, machine A;
# validation entry 43; requirements D13 amended with the same date).
# MEASURED 9.9739 px in ``residual_x`` and 9.3568 px in
# ``residual_y``, pinned at 2x the larger.
#
# RE-MEASURED AND RE-PINNED 2026-09-08 (Stage B adversarial review;
# validation entry 49).  Those two numbers were a MIXTURE: 13 of these
# 100 points did not converge, and ``hrebsd_pc_shift`` was reading the
# finiteness of the ``homography`` alone, which requirements D2.6
# deliberately keeps finite for exactly such a point.  The 13 abandoned
# fits contributed a mean of 3.1365 px and pulled the reported mean
# DOWN by 10.2 per cent.  With the convergence flag read, the same run
# gives 10.9956 px in ``residual_x`` and 9.8403 px in ``residual_y``
# over the 87 points that are measurements, so the band moves to 2x
# 10.9956.  Nothing about the wafer changed; what changed is that the
# number now describes only the fits that finished.
#
# The MTP note above says what is pinned is "how close to zero a
# consistent geometry gets", and on this dataset the answer is NOT
# close: the modelled drift is 20.0 px and the converged fits measure
# a median translation of 0.045 px (0.026 px over the contaminated
# set), so the residual is essentially the whole model.  The diagnostic is CORRECT and is reporting exactly the
# D13 signature it was built to report, a geometry inconsistent with
# the measurement; the inconsistency is on the measurement side (see
# the module docstring).  This band is therefore a regression guard
# on THIS dataset's behaviour and is not evidence about how close to
# zero a well-conditioned map gets.
SI_PC_RESIDUAL_MEAN_TOL = 2.2e01


# ----------------------------- Helpers ------------------------------ #


def si_wafer_signal():
    """Return the cached Si-wafer signal, or SKIP.

    Nothing here downloads: a missing cache is a skip whose message
    names the single command which fills it, so the 311 MB transfer is
    always a deliberate act.
    """
    pytest.importorskip("pooch", reason="the [download] gate of validation V5")
    from kikuchipy.data._data import Dataset

    dataset = Dataset("si_wafer/Pattern.dat", collection_name="ebsd_si_wafer.zip")
    if not dataset.file_path.exists():
        pytest.skip(
            "the Si wafer dataset of validation V5 is not in the local cache; "
            "fetch it once with kp.data.si_wafer(allow_download=True, lazy=True) "
            "before discharging the Stage B noise-floor gate"
        )
    return kp.data.si_wafer(lazy=True)


def smoke_subset(signal, step=SMOKE_STEP):
    """Return the ``[::step, ::step]`` sub-grid of the wafer as an
    eager signal, the default-suite subset of validation's
    Local-gated section."""
    subset = signal.inav[::step, ::step]
    subset.compute()
    return subset


def per_point_detector(signal):
    """Return the wafer detector with ONE projection centre per map
    point, built with kikuchipy's own beam-scan model.

    Validation V5's PC-shift arm asks for exactly this: the residuals
    are near-zero mean when the per-point projection centres come from
    :meth:`~kikuchipy.detectors.EBSDDetector.extrapolate_pc`, and the
    documented nonzero-mean signature is what a single fixed centre
    produces instead.

    The pixel size is set here rather than taken from the shipped
    detector; see ``UF420_PX_SIZE`` for the measurement that made it
    necessary (2026-09-08, Stage B implementation gate).
    """
    detector = signal.detector.deepcopy()
    detector.px_size = UF420_PX_SIZE
    return detector.extrapolate_pc(
        pc_indices=[0, 0],
        navigation_shape=signal.axes_manager.navigation_shape[::-1],
        step_sizes=step_sizes_of(signal),
    )


def single_crystal_xmap(signal):
    """Return the input crystal map of the wafer: one silicon phase,
    one orientation everywhere.

    The wafer is a single crystal, so its Hough orientations carry no
    information the HREBSD measurement uses: requirements D7 says
    orientations enter ONLY through the crystal to sample stiffness
    rotation, and the default route of this module takes the
    DEVIATORIC closure and no stiffness, which is orientation
    independent. Indexing the wafer first would therefore change none
    of the recorded floors, and a nominal orientation keeps the
    benchmark free of a dictionary-indexing dependency. The stress
    arm, which does depend on the orientation, is a separate weekly
    measurement and says so.
    """
    navigation_shape = signal.axes_manager.navigation_shape[::-1]
    arrays, size = create_coordinate_arrays(navigation_shape, step_sizes_of(signal))
    arrays["rotations"] = Rotation.identity((size,))
    arrays["phase_id"] = np.zeros(size, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="si", space_group=227))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    return xmap


def step_sizes_of(signal):
    """Return the ``(row, column)`` scan step sizes of a signal."""
    axes = signal.axes_manager.navigation_axes
    return (float(abs(axes[1].scale)), float(abs(axes[0].scale)))


def strain_stress_of(signal, stiffness=None, **kwargs):
    """Return the Stage B crystal map of a wafer subset, run through
    the default route: ``reference="auto"``, the per-point projection
    centres of the beam-scan model, and -- unless a *stiffness* is
    given -- the deviatoric closure of requirements D9.3."""
    detector = per_point_detector(signal)
    xmap = signal.hrebsd_dic(single_crystal_xmap(signal), detector, verbose=0, **kwargs)
    return detector, xmap, hrebsd_strain_stress(xmap, detector, stiffness=stiffness)


def worst_component_std(values):
    """Return the largest per-component standard deviation of an
    ``(n, k)`` property, NaN points excluded."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.all(np.isfinite(array), axis=1)]
    assert finite.shape[0] > 0.5 * array.shape[0], (
        "over half the map failed to converge"
    )
    return float(np.nanmax(np.std(finite, axis=0)))


def sweep_floor(values):
    """Return ``(worst per-component std, number of points it used)``
    of an ``(n, k)`` property, ``(nan, 0)`` when nothing converged.

    ADDED 2026-09-08 at the Stage B implementation gate.  The sweeps
    below cannot use :func:`worst_component_std`, which REFUSES a map
    where over half the points failed: that refusal is right for a
    floor which gets PINNED and wrong for a sweep ARM, whose job is to
    record what a knob setting does even when the answer is "it stops
    converging".  MEASURED on this dataset, that is not hypothetical:
    ``min_step = 1e-4`` leaves 26 of 100 points converged and the
    unfiltered ``filter_cutoffs=(None, None)`` arm fewer still, so the
    drafted sweeps aborted inside the helper before reaching their own
    assertions -- which contradicts both this class's "RECORDED, NOT
    GATED" contract and validation V5's line that the sweeps are
    "measurement harnesses ... results recorded below and defaults
    re-pinned only with dated amendments".  The precedent is validation
    V4's ``test_rotation_sweep_capture_range``, which records a sweep
    most of whose arms fail and asserts only that the easiest one
    worked.  So each sweep now asserts on its FROZEN DEFAULT arm --
    the number a re-pin decision would compare against -- and records
    the rest, convergence count included.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.all(np.isfinite(array), axis=1)]
    if finite.shape[0] == 0:
        return float("nan"), 0
    return float(np.nanmax(np.std(finite, axis=0))), int(finite.shape[0])


def assert_default_arm_recorded(entry, signal):
    """Assert one :func:`sweep_floor` entry -- the FROZEN DEFAULT arm
    of a sweep -- really produced a number over most of the map.

    This is what a recording harness may gate on (2026-09-08): a sweep
    whose own default arm produced nothing has recorded nothing, and
    the re-pin decision would have no baseline to compare against.
    """
    floor, points = entry
    assert np.isfinite(floor) and floor > 0.0, (
        f"the frozen default arm of this sweep recorded {floor!r}"
    )
    assert points > 0.5 * signal.axes_manager.navigation_size, (
        f"the frozen default arm converged on only {points} points"
    )


def assert_the_sweep_varied_something(floors):
    """Assert a sweep's arms are not all the same number.

    Added 2026-09-08 with :func:`sweep_floor`: relaxing "every arm is
    finite" to "the default arm is finite" would otherwise leave a
    sweep that silently ignores its knob passing, which is the one
    thing a measurement harness must never do.
    """
    values = {round(floor, 12) for floor, _ in floors.values() if np.isfinite(floor)}
    assert len(values) > 1, f"every arm of this sweep recorded the same floor: {values}"


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


# =================== V5 -- the strain noise floor =================== #


class TestNoiseFloor:
    """A nominally strain-free single crystal: whatever comes back IS
    the floor.  [D2-D9/V5]"""

    def test_si_noise_floor_default_route(self):
        # THE headline honesty number of this feature.  Default knobs,
        # ``reference="auto"`` (the Stage B default), the per-point
        # projection centres of the beam-scan model, on the smoke
        # sub-grid
        subset = smoke_subset(si_wafer_signal())
        _, _, result = strain_stress_of(subset)
        strain_floor = worst_component_std(result.prop["strain"])
        rotation_floor = worst_component_std(result.prop["rotation_vector"])
        # the ORDER check, which catches a wiring error rather than
        # recording it: a strain-free wafer cannot be at per-cent
        # strain, and a floor of exactly zero means nothing was
        # measured
        assert 0.0 < strain_floor < 100 * LITERATURE_STRAIN_FLOOR
        assert_within(strain_floor, SI_STRAIN_FLOOR, "SI_STRAIN_FLOOR")
        assert_within(rotation_floor, SI_ROTATION_FLOOR, "SI_ROTATION_FLOOR")

    @pytest.mark.weekly
    def test_si_stress_floor_with_the_silicon_stiffness(self):
        # the traction-free arm, weekly and RECORDED, NOT GATED, and
        # separate from the floors above because it is the one
        # measurement here which DOES depend on the wafer's
        # orientation, which this module deliberately does not index
        # (see ``single_crystal_xmap``). What it records is the SCALE
        # of the stress a nominally strain-free crystal reports, never
        # an absolute stress
        subset = smoke_subset(si_wafer_signal())
        stiffness = kp.indexing.voigt_stiffness("cubic", **SILICON_CUBIC)
        _, _, result = strain_stress_of(subset, stiffness=stiffness)
        floor = worst_component_std(result.prop["stress"])
        assert np.isfinite(floor)
        assert floor > 0.0

    @pytest.mark.weekly
    def test_si_noise_floor_full_map(self):
        # the full 50 by 50 map, weekly: the smoke sub-grid above is
        # a hundred points and its standard deviations are noisy in
        # their own right, so the RECORDED floor is this one
        signal = si_wafer_signal()
        signal.compute()
        _, _, result = strain_stress_of(signal)
        assert_within(
            worst_component_std(result.prop["strain"]),
            SI_STRAIN_FLOOR,
            "SI_STRAIN_FLOOR (full map)",
        )


class TestKamFloor:
    """The HR-KAM noise floor of requirements D12.  [D12/V5]"""

    def test_si_kam_floor(self):
        subset = smoke_subset(si_wafer_signal())
        _, _, result = strain_stress_of(subset)
        kam = hrebsd_kam(result)
        floor = float(np.nanmedian(kam))
        # milliradians, and a strain-free wafer's KAM is small but not
        # zero: the unit mutant would put this three orders out
        assert 0.0 < floor < 10.0
        assert_within(floor, SI_KAM_FLOOR, "SI_KAM_FLOOR")

    @pytest.mark.weekly
    def test_kam_order_sweep(self):
        # plan open question 4: the KAM kernel order default.
        # RECORDED, NOT GATED (the validation V4 capture-range
        # precedent): the noise-versus-resolution trade is the table a
        # dated re-pin would rest on, and ``order=1`` stays in force
        # unless it is refuted
        subset = smoke_subset(si_wafer_signal())
        _, _, result = strain_stress_of(subset)
        floors = {
            order: float(np.nanmedian(hrebsd_kam(result, order=order)))
            for order in (1, 2, 3)
        }
        assert all(np.isfinite(value) for value in floors.values())
        # the orders really do differ, or the sweep records nothing
        assert len(set(floors.values())) == len(floors)


class TestPcShiftPlane:
    """The PC-shift residuals of requirements D13 on real data.
    [D13/V5]"""

    def test_si_pc_shift_plane(self):
        subset = smoke_subset(si_wafer_signal())
        detector, xmap, _ = strain_stress_of(subset)
        maps = hrebsd_pc_shift(xmap, detector)
        worst = 0.0
        for name in ("residual_x", "residual_y"):
            worst = max(worst, abs(float(np.nanmean(maps[name]))))
        assert_within(worst, SI_PC_RESIDUAL_MEAN_TOL, "SI_PC_RESIDUAL_MEAN_TOL")
        # and the documented miscalibration signature: a deliberately
        # wrong beam-scan geometry moves the residual mean away from
        # zero by far more than the consistent geometry does.
        #
        # CORRECTED 2026-09-08 at the Stage B implementation gate.  The
        # drafted arm miscalibrated by ``wrong.pc[..., 0] += 0.02``, a
        # UNIFORM offset, and that is a provable no-op: ``beam_scan
        # _model`` is built from ``PC_target - PC_reference`` (the D6.2
        # closed form), from which a constant cancels identically.
        # MEASURED on this map: the drafted perturbation moves the
        # residual mean from 9.97390698866289 to 9.973906988662913, a
        # relative 2.3e-15, so the ``> 10 *`` assertion could never
        # hold and the arm was unsatisfiable by any conformant
        # implementation.  What a miscalibrated beam-scan geometry
        # really gets wrong is the scan-step to pixel-size RATIO, which
        # scales every modelled difference: MEASURED at ``px_size / 20``
        # the modelled PCx drift goes from 20.0 px to 400.0 px and the
        # residual mean to 199.97 px, a factor of 20.05.  The analytic
        # pin of the same signature, where the measured translations
        # rather than the geometry carry the offset, stays in
        # ``test_hrebsd_pc_shift.py::test_a_miscalibrated_projection
        # _centre_shows_as_a_nonzero_mean``
        wrong = subset.detector.deepcopy()
        wrong.px_size = UF420_PX_SIZE / 20.0
        wrong = wrong.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=subset.axes_manager.navigation_shape[::-1],
            step_sizes=step_sizes_of(subset),
        )
        shifted = hrebsd_pc_shift(xmap, wrong)
        assert abs(float(np.nanmean(shifted["residual_x"]))) > 10 * max(worst, 1e-6)


class TestDefaultSweeps:
    """The measurement harnesses of plan open questions 2, 3 and 10.
    RECORDED, NOT GATED: each produces the table a dated default
    re-pin would rest on, and the frozen defaults stay in force until
    one of them refutes them.  [D4/V5]"""

    @pytest.mark.weekly
    def test_preprocessing_default_sweep(self):
        # plan open question 10, which V5 OWNS: the D4.1 band-pass
        # cutoffs cannot be settled on noise-free synthetic oracles,
        # where a high-pass can only remove signal (validation V4's
        # capture-range record says so with a date).  Here noise is
        # real, so the comparison means something
        subset = smoke_subset(si_wafer_signal())
        floors = {}
        for cutoffs in ((0.05, None), (None, None), (0.05, 0.4), (None, 0.4)):
            _, _, result = strain_stress_of(subset, filter_cutoffs=cutoffs)
            floors[cutoffs] = sweep_floor(result.prop["strain"])
        for cutoffs in ((False, "no window"), (True, "Hann window")):
            _, _, result = strain_stress_of(subset, window=cutoffs[0])
            floors[cutoffs[1]] = sweep_floor(result.prop["strain"])
        # the frozen default is one of the arms, so the table always
        # contains the number the re-pin decision compares against
        assert (0.05, None) in floors
        assert_default_arm_recorded(floors[(0.05, None)], subset)
        assert_the_sweep_varied_something(floors)

    @pytest.mark.weekly
    def test_border_sweep(self):
        # plan open question 3: the D4.4 border default, whose knee is
        # the noise floor against the border fraction
        subset = smoke_subset(si_wafer_signal())
        floors = {}
        for border in (0.0, 0.05, 0.1, 0.2):
            _, _, result = strain_stress_of(subset, border=border)
            floors[border] = sweep_floor(result.prop["strain"])
        assert 0.05 in floors
        assert_default_arm_recorded(floors[0.05], subset)
        assert_the_sweep_varied_something(floors)

    @pytest.mark.weekly
    def test_convergence_threshold_sweep(self):
        # plan open question 2: the D2.5 and D2.6 convergence
        # defaults, as an error-versus-threshold curve on real data
        subset = smoke_subset(si_wafer_signal())
        floors = {}
        for min_step in (1e-2, 1e-3, 1e-4):
            _, _, result = strain_stress_of(subset, min_step=min_step)
            floors[min_step] = sweep_floor(result.prop["strain"])
        assert 1e-3 in floors
        assert_default_arm_recorded(floors[1e-3], subset)
        assert_the_sweep_varied_something(floors)
