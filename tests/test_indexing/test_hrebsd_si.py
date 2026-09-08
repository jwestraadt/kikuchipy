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

Written before the implementation exists: every test which calls the
Stage B modules fails with ``NotImplementedError`` once the data is
cached, and skips before that.
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
SI_STRAIN_FLOOR = None

# MTP [D8, V5]: the worst per-component standard deviation of the
# rotation vector in radians over the same map and the same run.
# MEASURING RECIPE: the same test
SI_ROTATION_FLOOR = None

# MTP [D12, V5]: the median HR-KAM of the same map in milliradians.
# The theory expectation is ``sigma_omega * sqrt(2 / N)``, the
# 0.05 to 0.1 mrad class for an order-1 kernel at this rotation
# floor.
# MEASURING RECIPE: ``TestKamFloor::test_si_kam_floor``
SI_KAM_FLOOR = None

# MTP [D13, V5]: the worst absolute MEAN of the PC-shift residual
# maps in pixels, with the per-point projection centres taken from
# ``extrapolate_pc``.  Requirements D13 says a nonzero mean is the
# miscalibration signature, so what is pinned is how close to zero a
# consistent geometry gets.
# MEASURING RECIPE: ``TestPcShiftPlane::test_si_pc_shift_plane``
SI_PC_RESIDUAL_MEAN_TOL = None


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
    """
    detector = signal.detector.deepcopy()
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
        # wrong projection centre moves the residual mean away from
        # zero by far more than the consistent geometry does
        wrong = detector.deepcopy()
        wrong.pc[..., 0] += 0.02
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
            floors[cutoffs] = worst_component_std(result.prop["strain"])
        for cutoffs in ((False, "no window"), (True, "Hann window")):
            _, _, result = strain_stress_of(subset, window=cutoffs[0])
            floors[cutoffs[1]] = worst_component_std(result.prop["strain"])
        assert all(np.isfinite(value) for value in floors.values())
        # the frozen default is one of the arms, so the table always
        # contains the number the re-pin decision compares against
        assert (0.05, None) in floors

    @pytest.mark.weekly
    def test_border_sweep(self):
        # plan open question 3: the D4.4 border default, whose knee is
        # the noise floor against the border fraction
        subset = smoke_subset(si_wafer_signal())
        floors = {}
        for border in (0.0, 0.05, 0.1, 0.2):
            _, _, result = strain_stress_of(subset, border=border)
            floors[border] = worst_component_std(result.prop["strain"])
        assert all(np.isfinite(value) for value in floors.values())
        assert 0.05 in floors

    @pytest.mark.weekly
    def test_convergence_threshold_sweep(self):
        # plan open question 2: the D2.5 and D2.6 convergence
        # defaults, as an error-versus-threshold curve on real data
        subset = smoke_subset(si_wafer_signal())
        floors = {}
        for min_step in (1e-2, 1e-3, 1e-4):
            _, _, result = strain_stress_of(subset, min_step=min_step)
            floors[min_step] = worst_component_std(result.prop["strain"])
        assert all(np.isfinite(value) for value in floors.values())
        assert 1e-3 in floors
