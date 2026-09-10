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

"""Tests of the neighbour-seeded propagation of requirements D20
(Stage D), the oracles of ``specs/2026-09-07-hrebsd-dic/
validation.md`` V8.

Covers:

- **V8(a)**, the RESCUE oracle: a deformed-master map carrying a
  smooth in-plane rotation ramp whose far points lie outside the
  phase-cross-correlation-seeded capture range measured in
  requirements D5, while every neighbour step stays well inside it.
  The default path must FAIL those points -- the premise is asserted,
  not assumed -- and the seeded path must recover the imposed field.
- **V8(b)**, the D20.6 equivalence band: wherever the independent
  default-path fit converges, the seeded run's answer agrees within
  ``SEED_EQUIVALENCE_TOL``.
- **V8(c)**, the D20.3 determinism pins: two seeded runs bitwise, two
  chunk sizes bitwise, lazy equal to eager bitwise, on a map whose
  cascade rounds each carry TWO points with DIFFERENT seeds, so that
  intra-round scheduling and chunking have something to get wrong.
- **V8(d)**, the D20.4 isolation pins: a two-grain map with an
  orientation discontinuity constructed so that a seed which crossed
  the boundary would demonstrably corrupt the second grain, and the
  mask pins.
- **V8(e)**, the D20.5 ``seed_round`` encoding and the D20.2
  ``PASS1_CAP`` semantics, including the rescue pass.
- The D20.2 PC-transport bound, a library measurement which records
  the justification for copying a neighbour's RAW homography without
  transporting it between projection centre frames.
- The D20.1 default-off pin at engine level: ``seed_from_neighbors=
  False`` reproduces a FROZEN PRE-STAGE-D result and emits exactly
  the pre-Stage-D property set, with no ``seed_round`` in it.

"Bitwise" throughout this module means :func:`numpy.array_equal` with
``equal_nan=True``, the Stage A convention of
``test_hrebsd_engine.py::test_two_runs_bitwise``: it treats ``-0.0``
and ``+0.0`` as equal and all NaN payloads as equal, so it is value
identity rather than byte identity.

Written failing before the implementation, at the Stage D
failing-tests gate: every test which calls the seeded path fails with
``NotImplementedError`` from the Stage D skeleton, the seed-choice
unit tests fail with an assertion naming the private names the gate
freezes, and the groups which do neither pass today and are freezes
or premises rather than behaviour tests.  The tally at the gate is
``59 failed, 12 passed``, the 59 being 21 ``NotImplementedError`` and
38 assertions naming the two frozen private names, and the 12 listed
so that it is checkable rather than described:

- :class:`TestDefaultOffIsBitwiseUnchanged`, all four -- the
  signature freeze, the pre-Stage-D pin, the inert-keyword check and
  the D20.5 absence rule
- :class:`TestPcTransportBound`, all three
- the three premise tests of :class:`TestRescueOracle`:
  ``test_the_ramp_is_the_construction_it_claims_to_be``,
  ``test_the_ramp_stays_inside_the_pattern`` and
  ``test_the_default_path_fails_the_far_points``.  The last DOES run
  the engine and is the premise half of V8(a), which is what keeps
  the rescue oracle from being vacuous, so it is named here rather
  than lumped in with the freezes
- the two fixture-construction premises
  :class:`TestSimultaneousSeeding`
  ``::test_the_two_rows_never_see_each_other`` and
  :class:`TestGrainIsolation`
  ``::test_the_two_grains_impose_visibly_different_fields``

**The oracle helpers are deliberately duplicated** from
``test_hrebsd_engine.py`` rather than imported: ``tests/`` is not an
importable package, and an oracle which shares an error with the
module it judges cannot see that error.  The projection helper is the
V3 deformed-master one (EMsoft's own ``applyDeformation`` mechanism),
so the expected homography of every point below is EXACT by
construction rather than approximate.

**Two private names are FROZEN by this gate** and are the only
implementation surface it constrains beyond the public keyword:
``_engine.NEIGHBOR_OFFSETS``, the frozen tie order of requirements
D20.2, and ``_engine.choose_seed_indices``, the pure seed-choice
function the round orchestration is built on.  The tie rule cannot be
pinned through the public method at all -- two neighbours with
BITWISE equal residuals also have bitwise equal homographies on every
construction this gate could build, so which of them wins is
invisible in the result -- and the lowest-residual rule, the grain
gate and the mask gate are all sharper as a unit test of that
function than as a pattern-level oracle.

Freezing a helper is only worth anything if the orchestration
actually goes through it, so a THIRD thing is frozen with them:
:class:`TestSeedChoiceIsTheSeamTheCascadeUses` spies on
``_engine.choose_seed_indices`` as a module attribute during a real
seeded run and requires at least one call per cascade round.  Without
that, an implementation could define a conformant helper, satisfy
every unit test below, and then seed by different logic -- which is
exactly how the "seed from the highest residual", "tie order
shuffled" and "seed from an unconverged neighbour" mutants of plan
10.3 would survive, since MEASURED (2026-09-09) every seeding event
on every fixture in this module has exactly ONE same-grain converged
candidate, so no map-level property can see which candidate won.
"""

import functools
import inspect
import itertools
import warnings

import dask.array as da
import numpy as np
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd import _engine
from kikuchipy.indexing._hrebsd._engine import STAGE_A_PROP_NAMES, run_hrebsd_dic
from kikuchipy.indexing._hrebsd._homography import (
    N_HOMOGRAPHY_PARAMETERS,
    fe_to_homography,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# The oracle detector size.  Every capture-range number requirements
# D5 records is measured at 480 px, and this module's premise is one
# of those numbers, so it may not be measured at any other size
SHAPE = (480, 480)

# A projection centre off the pattern centre, so that a PC-centred
# frame stays distinguishable from a pattern-centred one, but LESS off
# centre than ``test_hrebsd_engine.py``'s ``PC_480``.  MEASURED reason,
# not a preference: the ramp below reaches 4.8 degrees of in-plane
# rotation, which displaces a subregion corner at radius r by
# ``r * sin(theta)``.  At ``PC_480 = (0.4210, 0.5794)`` the farthest
# subregion corner sits 358.5 px from the projection centre and the
# largest angle whose warp stays inside the pattern at all is 3.84
# degrees (recorded in ``test_hrebsd_engine.py::TestPureRotations``),
# so a 4.8 degree case there would measure the mirror boundary rather
# than the engine.  Here the radius is 311.6 px and the rotated
# subregion is inside the pattern at every angle of the ramp, VERIFIED
# by ``test_the_ramp_stays_inside_the_pattern`` below rather than
# asserted here
PC = (0.49, 0.51, 0.5049)

# The imposed in-plane rotation ramp of V8(a), in degrees, one entry
# per column of the map's first row.  The step is 0.8 degrees, well
# inside the 2.0 degree phase-cross-correlation capture range of
# requirements D5, and the far end is 4.8 degrees, well outside it.
# MEASURED on this fixture 2026-09-09 (the numbers the premise test
# re-asserts at run time): from the phase cross-correlation seed the
# 0.8 and 1.6 degree points converge in 5 and 10 iterations to 0.0019
# and 0.0056 px, while 2.4, 3.2, 4.0 and 4.8 degrees exhaust a 200
# iteration budget at 47.7, 48.7, 42.9 and 105.2 px from the imposed
# homography.  From the PREVIOUS ramp point's homography every one of
# them converges in 5 iterations to 0.014 px or better
RAMP_ANGLES = (0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8)
RAMP_STEP_DEG = 0.8

# The iteration budget of every run in this module.  It must exceed
# the 113 iterations the rescue point below needs (measured), and the
# far ramp points then exhaust it on the default path
MAX_ITERATIONS = 200

# The map of V8(a) and V8(e): row 0 carries the ramp, row 1 is masked
# out entirely, and row 2 carries three isolated points whose whole
# eight-neighbourhood is masked, so that each of them can reach only
# the rescue pass and never a cascade round
RAMP_NAVIGATION_SHAPE = (3, 7)
RAMP_SIZE = RAMP_NAVIGATION_SHAPE[0] * RAMP_NAVIGATION_SHAPE[1]

# Flat map indices of the three isolated points of row 2.
# ``UNFITTABLE`` is an 8.0 degree in-plane rotation, far outside every
# capture range and outside the pattern budget too, which MEASURED
# exhausts the whole budget without converging; ``RESCUE`` is a 9.0
# degree out-of-plane tilt, which MEASURED converges from the phase
# cross-correlation seed in 113 iterations, i.e. it is caught by any
# pass-1 cap below that and by the full budget; ``FAILED`` is a
# constant pattern, the D2.6 no-variance failure contract
RAMP_UNFITTABLE_INDEX = 15
RAMP_RESCUE_INDEX = 17
RAMP_FAILED_INDEX = 19
RAMP_UNFITTABLE_ANGLE = 8.0
RAMP_RESCUE_TILT = 9.0

# The two-grain map of V8(d): grain 0 is columns 0 and 1, grain 1 is
# columns 2, 3 and 4, and the hard point of grain 1 sits at column 2,
# in the eight-neighbourhood of grain 0's column 1
ISOLATION_NAVIGATION_SHAPE = (1, 5)
ISOLATION_LABELS = np.array([[0, 0, 1, 1, 1]], dtype=np.int32)
ISOLATION_REFERENCES = np.array([0, 4])
ISOLATION_CROSS_INDEX = 1
ISOLATION_HARD_INDEX = 2
ISOLATION_SAME_INDEX = 3

# MEASURED 2026-09-09 and RECORDED, not a live quantity: what the
# grain gate of requirements D20.4 actually buys on this map, obtained
# by simulating the crossed seed with the low-level fit directly
# (``fit_pattern`` against grain 1's own reference, seeded with the
# EXACT homography of the neighbour named, budget 200).  Recipe in
# validation.md "Recorded results".
#
#   seeded from column 3, the same-grain neighbour ... CONVERGED in
#     19 iterations, 0.100195 px from the imposed field (0.27 per
#     cent of the 36.4788 px it imposes)
#   seeded from column 1, the cross-grain neighbour .. did NOT
#     converge, exhausted the 200 iteration budget 30.8866 px away
#     (84.67 per cent of the imposed displacement)
#
# So the grain gate is worth a factor of 308 in accuracy AND the
# difference between a converged and a non-converged point.  The
# assertion below compares the run's own hard-point error against
# this recorded number rather than against the DISTANCE BETWEEN TWO
# IMPOSED FIELDS, which is a static property of the fixture and says
# nothing about what a crossed seed would do
ISOLATION_CROSSED_SEED_ERROR = 30.8866

# The pre-Stage-D pin map of D20.1: the first four ramp points, the
# cheapest map on which the default path does something interesting
# (three convergences at 1, 5 and 10 iterations and one point which
# exhausts the budget)
PIN_NAVIGATION_SHAPE = (1, 4)

# The determinism map of V8(c) and the simultaneous-seeding map which
# closes plan 10.3's "per-point h0 array off by one in flat order":
# TWO opposite-signed in-plane rotation ramps on rows 0 and 2 with row
# 1 masked out, so the rows are never in each other's
# eight-neighbourhood and each cascade round carries TWO points whose
# seeds differ by twice the ramp angle.
#
# Why this map and not the ramp map of V8(a): MEASURED 2026-09-09,
# every cascade round on the (3, 7) ramp map, on the (1, 5) isolation
# map and on the four-point pin map has exactly ONE member, so a
# permutation of a per-round seed array is unobservable there and a
# chunk-size pin only ever varies the chunking of pass 1.  Here round
# 1 is {(0, 3), (2, 3)} and round 2 is {(0, 4), (2, 4)}, and the two
# members of each carry seeds 3.2 degrees apart
TWO_RAMP_ANGLES = (0.0, 0.8, 1.6, 2.4, 3.2)
TWO_RAMP_NAVIGATION_SHAPE = (3, 5)
TWO_RAMP_SIZE = TWO_RAMP_NAVIGATION_SHAPE[0] * TWO_RAMP_NAVIGATION_SHAPE[1]

# The columns of each ramp row which pass 1 converges and which the
# cascade converts, MEASURED 2026-09-09 at a pass-1 cap of 50: columns
# 0, 1 and 2 converge in 1, 5 and 10 iterations on row 0 and in 1, 5
# and 7 on row 2, while columns 3 and 4 exhaust the cap on both rows
TWO_RAMP_PASS1_COLUMNS = (0, 1, 2)
TWO_RAMP_CASCADE_COLUMNS = (3, 4)

# The ``seed_round`` encoding of requirements D20.5, written out
SEED_ROUND_PASS1 = 0
SEED_ROUND_RESCUE = -2
SEED_ROUND_NONE = -1
SEED_ROUND_PROP = "seed_round"

# The property set a SEEDED run emits: the Stage A set of D15.6 plus
# the one new int32 property of D20.5.  Assembled here and not
# imported, because the absence pin below is exactly that the engine's
# own ``STAGE_A_PROP_NAMES`` does NOT grow a ``seed_round`` entry
SEEDED_PROP_NAMES = STAGE_A_PROP_NAMES + (SEED_ROUND_PROP,)

# The frozen neighbour offset order of requirements D20.2, which
# breaks a residual tie.  Row-major reading order of the
# eight-neighbourhood, the centre skipped
FROZEN_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# The fraction of its own imposed corner displacement a recovered
# homography may sit away from the imposed one before it counts as a
# DIFFERENT optimum rather than the same one.  FROZEN, not measured:
# it separates two populations which sit about 200x apart on these
# fixtures.  MEASURED 2026-09-09, and CORRECTED on that date after
# the adversarial review found two of the five denominators quoted
# here were not this fixture's (the imposed corner norms are 4.3500,
# 8.6999, 13.0493, 17.3980, 21.7460 and 26.0928 px along the ramp,
# 91.2533 px at the rescue point and 36.4788 px at the isolation hard
# point):
#
#   recovered field   0.044 to 0.43 per cent of the imposed
#                     displacement -- 0.0019 px of 4.3500, 0.3888 px
#                     of 91.2533, 0.1002 px of 36.4788
#   wrong optimum     85 to 403 per cent of it -- 30.887 px of
#                     36.4788, 105.239 px of 26.0928
#
# so anything between about 1e-2 and 3e-1 says the same thing.  It is
# a discriminator, never a precision claim: the precision bands are
# the measured-then-pinned ones below.  The tightest user of it is
# the rescue point, at 0.43 per cent, i.e. 2.35x of headroom
SAME_OPTIMUM_FRACTION = 0.01

# ------------- The FROZEN pre-Stage-D default-path result ----------- #
#
# Requirements D20.1 asks for "an explicit default-off bitwise pin
# against a PRE-STAGE-D RESULT".  The arrays below are that result,
# not a second live call: MEASURED 2026-09-09 by running
# ``run_hrebsd_dic`` from commit cec39de4 -- the last commit before
# the Stage D skeleton, whose ``_engine.py`` has no
# ``seed_from_neighbors`` in it at all -- on the four-point pin map at
# ``max_iterations=200``, and VERIFIED on that date to be BITWISE what
# today's working tree gives on the ``False`` path, property by
# property.
#
# Why literals and not a live comparison: a pin which calls the
# CURRENT default path twice is ``f(x) == f(x)`` and cannot fail,
# whatever a later Stage D commit does to that path.  This was
# demonstrated at the 2026-09-09 adversarial review with two injected
# default-path regressions, both of which the whole hrebsd suite and
# the earlier form of this pin passed; both die here.  The freshness
# objection is answered by the commit id above, not by re-running the
# same code twice.
#
# REFRESH RULE: these numbers change only on a DELIBERATE change to
# the default path, and such a change must be recorded in
# validation.md "Recorded results" with its own date and reason.  A
# failure here is a Stage D regression until proven otherwise.
PRE_STAGE_D_NUM_ITERATIONS = np.array([1, 5, 10, 200], dtype=np.int32)
PRE_STAGE_D_CONVERGED = np.array([True, True, True, False])
PRE_STAGE_D_RESIDUAL = np.array(
    [
        1.7752252190879082e-16,
        8.0396229820182334e-04,
        8.7140529693886999e-04,
        1.3544898885699341e00,
    ]
)
PRE_STAGE_D_HOMOGRAPHY = np.array(
    [
        [
            2.8124169659804465e-12,
            7.7263846565462982e-13,
            -5.5839979058452432e-11,
            -1.8389308374422834e-12,
            2.5450752616507089e-12,
            -2.3605975561908617e-10,
            1.7033843029450640e-15,
            -1.2765008070562455e-14,
        ],
        [
            -9.3367806158761724e-05,
            -1.3960010227122189e-02,
            -1.9868284876050455e-04,
            1.3962833035760562e-02,
            -9.7452531258257480e-05,
            8.9780865628809827e-05,
            -1.8810392422214518e-08,
            7.1503485865651986e-09,
        ],
        [
            -3.9032561031437929e-04,
            -2.7920041114921718e-02,
            -5.2351655554776558e-04,
            2.7925112444333482e-02,
            -3.9022684414868891e-04,
            3.0497015936517808e-05,
            -7.4025979165113979e-08,
            6.6318700769855435e-09,
        ],
        [
            3.0433358869287996e-02,
            3.8845791240189817e-02,
            -1.1068489150088306e01,
            4.0191110870032180e-02,
            3.4276158189613604e-02,
            -5.3022638876779804e-01,
            -5.1667134427688983e-05,
            2.1708191560492978e-04,
        ],
    ]
)

# MTP [D20.1]: the corner-displacement band in binned pixels in which
# the default path must reproduce the frozen homography above.
# MEASURED 2026-09-09, all four numbers on the pin map:
#   two live default runs .................... 5.6843e-14 px, which is
#     the metric's OWN floor (``inv(W) @ W`` is not exactly the
#     identity in float64); the two runs are bitwise equal
#   ``upsample_factor=4`` instead of 16, the
#     subtlest demonstrated regression (a new
#     Stage D call site which forgets to
#     forward the knob) ...................... 1.9614e-06 px
#   ``upsample_factor=8`` ..................... 3.1815e-06 px
#   ``min_step=1e-2`` ......................... 5.6042e-06 px
# PINNED at 1e-9 px: four orders above the metric's floor and three
# orders below the subtlest demonstrated regression.  It is a
# float-noise band, never a precision claim
PRE_STAGE_D_PIN_TOL = 1e-9

# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #

# MTP [D20.2/V8(a)]: the corner-displacement error-warp norm in binned
# pixels of the SEEDED fits of the rotation ramp against their EXACT
# imposed homographies, worst over the ramp.
# MEASURING RECIPE: ``TestRescueOracle::test_the_seeded_path_recovers
# _the_whole_ramp`` below; record the worst over row 0 of the ramp map
# and fill this with a dated value at the Stage D implementation gate,
# with the recipe in validation.md "Recorded results".
# DRAFTING NOTE, deliberately NOT a live number (the
# ``WARP_REFIT_TOL_480`` precedent of ``test_hrebsd_engine.py``): the
# cascade was SIMULATED at this gate by seeding each ramp point with
# its predecessor's EXACT homography, which measured 0.0143 px worst
# over the six warped points, so the pin is expected near 0.03.  A
# live seed 2 to 10 times above the achievable error is an acceptance
# gate a mediocre implementation passes silently, so this fails loudly
# instead.
# MEASURED 2026-09-09 at the Stage D implementation gate on the live
# cascade, both fixtures which carry this pin:
#   ``TestRescueOracle::test_the_seeded_path_recovers
#     _the_whole_ramp``, worst over the seven ramp
#     points ................................. 0.0142852057 px
#   ``TestSimultaneousSeeding::test_each_point
#     _of_a_round_gets_its_own_seed``, worst over
#     the ten unmasked two-ramp points ....... 0.0142852057 px
# The two agree to every digit because the worst point of both maps is
# the same construction, a 3.2 degree point fitted from a seed 0.8
# degrees away.  The live cascade lands exactly on the 0.0143 px the
# simulation predicted, so the cap-truncated pass-1 iterate a real seed
# carries costs nothing against the simulation's exact homography.
# PINNED at 0.03 px, 2.1x the measurement, which is the drafting note's
# own expectation.  This is an accuracy claim, not a float-noise band:
# the same fits are about 3500x closer to their own imposed field than
# the default path's far-basin answers
SEED_RESCUE_TOL = 0.03

# MTP [D20.6/V8(b)]: the corner-displacement disagreement in binned
# pixels between the default-path answer and the seeded answer, worst
# over the points BOTH paths converge WITHIN THE RUN'S OWN ITERATION
# BUDGET.
# MEASURING RECIPE: ``TestEquivalence::test_both_paths_agree_where
# _both_converge`` below.
# DRAFTING NOTE: the rescue point's route was simulated at this gate
# (fit at a cap of 30, 50, 80 or 100 iterations, then re-fitted at the
# full budget from that last iterate) and lands on the default path's
# answer to 5.7e-14 px in all four, which is float noise, so this pin
# is expected to be a machine-precision band rather than a tolerance.
#
# THE BUDGET CLAUSE IS LOAD BEARING, and D20.6 was narrowed to carry
# it after a MEASUREMENT on 2026-09-09 refuted the unqualified form
# ("wherever the independent fit converges, the two agree").  On this
# very ramp, at ``max_iterations=2000``, the default path CONVERGES
# the 2.4, 3.2, 4.0 and 4.8 degree points in 291, 281, 320 and 728
# iterations -- to 50.84, 50.81, 49.90 and 108.04 px from the imposed
# field, i.e. to a DIFFERENT optimum, while the seeded path reaches
# each in 5 iterations at 0.0096, 0.0143, 0.0119 and 0.0138 px.  The
# two paths then disagree by about 50 px and the unqualified claim is
# simply false.  At the ``MAX_ITERATIONS = 200`` this module runs,
# those four points do NOT converge on the default path, so they are
# outside the ``both`` mask and the band below is a machine-precision
# one.  The correction is recorded in requirements D20.6 and in
# validation.md V8(b), and the ``both`` set is pinned literally in the
# test so that a budget change fails loudly instead of silently
# widening the claim.
# MEASURED 2026-09-09 at the Stage D implementation gate, on the live
# cascade rather than the simulation, on BOTH populations D20.6 names
# for this constant -- the oracle map and the Si sub-map:
#   this ramp, worst over the four points of
#     ``EQUIVALENCE_BOTH_CONVERGED`` (the three
#     easy columns and the isolated rescue
#     point) ................................. 5.6843e-14 px
#   the REAL Si-indent rim, entry 80's rows
#     125:145 columns 115:135 at 500
#     iterations, worst over the 373 points
#     both paths converge .................... 2.3437e-13 px
#   the same file's far field, rows 20:40
#     columns 20:40, worst over 400 ........... 2.3437e-13 px
# Every one of them is the metric's OWN floor and not a disagreement:
# the seeded and the default homography of each of those points are
# bitwise equal, so what is measured is ``inv(W) @ W`` in float64.  The
# Si floor is 4.1x the ramp's because the metric is evaluated on a
# bigger support (the D4.4 corners of the 512 by 622 Si detector reach
# 494 px from its off-centre projection centre against the oracle's
# 312) and on warps carrying more rotation.
# PINNED at 5e-13 px, 2.1x the worst of the two populations, which
# leaves 8.8x on the ramp the test actually evaluates.  A
# machine-precision band, never a tolerance, and the width costs
# nothing: the subtlest regression ``PRE_STAGE_D_PIN_TOL`` could
# demonstrate above is 1.96e-06 px, four million times this pin.  The
# claim it guards is a strong one -- on the Si rim the seeded path
# reaches all 373 points through a cap-truncated pass 1 and a rescue
# re-fit, a completely different route from the default path's single
# fit, and lands on the same float64 bits at every one of them
SEED_EQUIVALENCE_TOL = 5e-13

# The points of the ramp map on which BOTH paths converge at
# ``MAX_ITERATIONS``, pinned so that the equivalence band above cannot
# quietly start describing a different population.  MEASURED
# 2026-09-09: the default path converges columns 0, 1 and 2 of the
# ramp (1, 5 and 10 iterations) and the isolated rescue point (113
# iterations), and nothing else
EQUIVALENCE_BOTH_CONVERGED = {0, 1, 2, 17}

# MEASURED-THEN-PINNED and MEASURED at this gate [D20.2]: the
# corner-displacement norm in binned pixels of the projection-centre
# phantom between two NEIGHBOURING map points of the Si-indent
# geometry, which is exactly the error a seed acquires when it is
# copied from a neighbour without being transported into that
# neighbour's projection centre frame.
# MEASURING RECIPE: ``TestPcTransportBound`` below, on the geometry
# validation.md ledger entry 80 records for
# ``AGH__Si_indent_1_512x672.h5oina``: 512 rows by 622 columns at
# binning 2, a 234 by 250 map, and projection centre spans of 1.31,
# 1.14 and 0.47 binned px over the whole map.
# MEASURED 2026-09-09 on that data set's REAL projection centre, read
# from the file on that date and converted to the kikuchipy (Bruker)
# frame the engine works in: 0.009727 px worst over a row step, a
# column step and a diagonal step and over all three compositions
# with a typical converged rim homography.  PINNED at just under 2x.
# CORRECTED on 2026-09-09 after the adversarial review found the
# earlier form of this test invented a central projection centre and
# then advertised the number as a measurement on the Si geometry; the
# three other readings of the file's fractions the test now sweeps
# measure 0.008578, 0.009171 and 0.009727 px, so the verdict is
# unchanged and the provenance is now real
PC_TRANSPORT_BOUND = 0.019

# The same bound as a fraction of the capture range it has to be
# negligible against, which is what D20.2's justification actually
# claims.  MEASURED 2026-09-09 on the same real projection centre:
# 5.637e-04 of the 17.255 px of corner motion a 2.0 degree in-plane
# rotation induces on that geometry, the largest rotation
# requirements D5 records as recoverable from the phase
# cross-correlation seed.  PINNED at about 3x; the four readings
# swept below span 4.972e-04 to 6.519e-04
PC_TRANSPORT_BASIN_FRACTION = 1.6e-3


# ----------------------------- Helpers ------------------------------ #
#
# Assembled here in plain numpy.  The twins of these live in
# ``test_hrebsd_engine.py`` and the two files are deliberately not
# coupled: ``tests/`` is not an importable package, and an oracle
# which shares an error with the module it judges cannot see it.


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


def matrix_of(h):
    """Return the ``(3, 3)`` shape function of the eight parameters,
    built here rather than taken from the code under test."""
    h = np.asarray(h, dtype=np.float64)
    return np.array(
        [
            [1.0 + h[0], h[1], h[2]],
            [h[3], 1.0 + h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )


def corner_norm(matrix, corners):
    """Return the D2.5 norm of a warp: the largest displacement it
    induces over the four subregion corners, in binned pixels."""
    x, y = corners[:, 0], corners[:, 1]
    scale = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    warped_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / scale
    warped_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / scale
    return float(np.hypot(warped_x - x, warped_y - y).max())


def recovery_error(h_fit, h_true, corners):
    """Return the V2 recovery metric: the corner norm of the error
    warp ``W(h_true)**-1 . W(h_fit)`` in binned pixels."""
    return corner_norm(np.linalg.inv(matrix_of(h_true)) @ matrix_of(h_fit), corners)


def subregion_corners(shape, pc_px, border=0.05):
    """Return the ``(4, 2)`` PC-centred corners of the D4.4 subregion,
    the support of the D2.5 norms."""
    nrows, ncols = shape
    margin_row = int(round(border * nrows))
    margin_col = int(round(border * ncols))
    x0 = margin_col + 0.5 - pc_px[0]
    x1 = ncols - 1 - margin_col + 0.5 - pc_px[0]
    y0 = margin_row + 0.5 - pc_px[1]
    y1 = nrows - 1 - margin_row + 0.5 - pc_px[1]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def pc_pixels_of(detector, index=0):
    """Return ``(PCx_px, PCy_px, DD_px)`` of one projection centre,
    requirements D1.2 written out here."""
    nrows, ncols = detector.shape
    pcx, pcy, pcz = np.atleast_2d(detector.pc_flattened)[index]
    return np.array([pcx * ncols, pcy * nrows, pcz * nrows])


def make_detector(navigation_shape=None):
    """Return the oracle detector, with one projection centre per map
    point when a navigation shape is given.

    Every point carries the SAME projection centre, so the D6.2
    beam-scan phantom is exactly the identity and nothing below
    measures it; the phantom's own oracle is validation V6.
    """
    pc = np.asarray(PC, dtype=np.float64)
    if navigation_shape is None:
        pc = pc[None, :]
    else:
        pc = np.tile(pc, (*navigation_shape, 1))
    return kp.detectors.EBSDDetector(
        shape=SHAPE,
        binning=1,
        px_size=70.0,
        pc=pc,
        sample_tilt=70.0,
        tilt=0.0,
    )


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


def sample_to_crystal_matrix(rotation):
    """Return ``R`` with ``v_crystal = R @ v_sample``, derived from
    ``rotate_vector`` itself so that no orix convention is assumed."""
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    return rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T


def crystal_to_detector_matrix(detector, rotation):
    """Return ``R`` with ``v_detector = R @ v_crystal`` in the spec's
    y-DOWN detector frame, the D7 chain a deformation is imposed
    through.  Nothing is hardcoded: the rotation comes from the
    detector and the flip is the one measured in D1.5."""
    matrix = detector.sample_to_detector.to_matrix().squeeze()
    return (np.diag([1.0, -1.0, 1.0]) @ matrix) @ sample_to_crystal_matrix(rotation).T


def _bilinear(master, nii, nij, niip, nijp, di, dj, dim, djm):
    """Vectorized twin of ``_get_pixel_from_master_pattern``."""
    return (
        master[nii, nij] * dim * djm
        + master[niip, nij] * di * djm
        + master[nii, nijp] * dim * dj
        + master[niip, nijp] * di * dj
    )


def project_pattern(detector, rotation, deformation=None):
    """Return one pattern projected from the Ni Lambert master with an
    imposed CRYSTAL frame deformation gradient, the V3 mechanism.

    Because the engine's geometric model and this simulation are the
    same first-order model, the expected homography is EXACT.
    """
    upper, lower, npx, npy, scale = ni_master_arrays()
    cosines = _get_direction_cosines_from_detector(detector)
    if cosines.ndim == 3:
        cosines = cosines[0]
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


def in_plane_fe(angle_deg):
    """Return the detector-frame tensor of an in-plane rotation, which
    is its own reduced tensor."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def out_of_plane_fe(angle_deg):
    """Return the REDUCED detector-frame tensor of an out-of-plane
    tilt about the detector x axis."""
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]]) / cos


def orientation_a():
    """Return the generic crystal orientation of the first grain,
    deliberately away from any symmetry position."""
    return Rotation.from_axes_angles((1.0, 2.0, 3.0), np.deg2rad(37.0))


def orientation_b():
    """Return the crystal orientation of the second grain, a large
    misorientation away from the first."""
    return Rotation.from_axes_angles((3.0, -1.0, 2.0), np.deg2rad(52.0))


def deformed_pattern(detector, rotation, fe_detector):
    """Return ``(pattern, h_exact)`` of one imposed reduced
    detector-frame deformation gradient."""
    reduced = np.asarray(fe_detector, dtype=np.float64)
    reduced = reduced / reduced[2, 2]
    chain = crystal_to_detector_matrix(detector, rotation)
    pattern = project_pattern(detector, rotation, chain.T @ reduced @ chain)
    pc_px = pc_pixels_of(detector)
    return pattern, fe_to_homography(reduced, np.zeros(2), pc_px[2])


def run_map(patterns, navigation_shape, reference, *, seeded, **kwargs):
    """Return the engine's properties for one map.

    The user warning about non-converged points is expected on every
    map here -- that is what the premise of V8(a) says -- and is
    silenced rather than asserted, since ``TestFailureSemantics`` of
    the signal suite owns it.
    """
    kwargs.setdefault("max_iterations", MAX_ITERATIONS)
    kwargs.setdefault("verbose", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run_hrebsd_dic(
            patterns,
            navigation_shape,
            make_detector(navigation_shape),
            reference=reference,
            seed_from_neighbors=seeded,
            **kwargs,
        )


# --------------------------- The fixtures --------------------------- #


@functools.lru_cache(maxsize=1)
def oracle_reference():
    """Return the undeformed 480 px reference pattern of grain 0."""
    pattern = project_pattern(make_detector(), orientation_a())
    pattern.flags.writeable = False
    return pattern


@functools.lru_cache(maxsize=1)
def ramp_map():
    """Return the V8(a)/V8(e) map: ``(patterns, mask, exact, corners)``.

    Row 0 is the rotation ramp against the reference at ``(0, 0)``.
    Row 1 is masked out entirely, so that every ramp point has masked
    neighbours and the cascade has to walk ALONG the row.  Row 2 holds
    the three isolated points, each with its whole neighbourhood
    masked: the unfittable one, the rescue one and the failed one.
    """
    detector = make_detector()
    reference = oracle_reference()
    row0, exact = [], []
    for angle in RAMP_ANGLES:
        if angle == 0.0:
            row0.append(reference)
            exact.append(np.zeros(N_HOMOGRAPHY_PARAMETERS))
        else:
            pattern, h = deformed_pattern(detector, orientation_a(), in_plane_fe(angle))
            row0.append(pattern)
            exact.append(h)
    unfittable, h_unfittable = deformed_pattern(
        detector, orientation_a(), in_plane_fe(RAMP_UNFITTABLE_ANGLE)
    )
    rescue, h_rescue = deformed_pattern(
        detector, orientation_a(), out_of_plane_fe(RAMP_RESCUE_TILT)
    )
    # a constant pattern has no variance at all, the D2.6 failure
    # contract, and is what pins ``seed_round`` on a point which is
    # neither masked nor fittable
    failed = np.full(SHAPE, 7.0)

    row2 = [reference, unfittable, reference, rescue, reference, failed, reference]
    patterns = np.stack(row0 + [reference] * 7 + row2)
    patterns.flags.writeable = False

    mask = np.zeros(RAMP_NAVIGATION_SHAPE, dtype=bool)
    mask[1] = True
    mask[2] = True
    for column in (1, 3, 5):
        mask[2, column] = False
    mask.flags.writeable = False

    exact = (
        exact
        + [np.zeros(N_HOMOGRAPHY_PARAMETERS)] * 7
        + [
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            h_unfittable,
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            h_rescue,
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
            np.zeros(N_HOMOGRAPHY_PARAMETERS),
        ]
    )
    corners = subregion_corners(SHAPE, pc_pixels_of(detector))
    return patterns, mask, np.stack(exact), corners


@functools.lru_cache(maxsize=1)
def ramp_default():
    """Return the DEFAULT path's properties on the ramp map, which is
    the premise half of V8(a) and the reference half of V8(b)."""
    patterns, mask, _, _ = ramp_map()
    return run_map(
        patterns, RAMP_NAVIGATION_SHAPE, (0, 0), seeded=False, navigation_mask=mask
    )


@functools.lru_cache(maxsize=1)
def ramp_seeded():
    """Return the SEEDED path's properties on the ramp map.

    Cached like its default-path twin.  While the Stage D skeleton
    raises, every caller gets the exception instead:
    :func:`functools.lru_cache` does not cache exceptions, so the
    fixture is built once and the raise is re-raised per test.
    """
    patterns, mask, _, _ = ramp_map()
    return run_map(
        patterns,
        RAMP_NAVIGATION_SHAPE,
        (0, 0),
        seeded=True,
        navigation_mask=mask,
    )


@functools.lru_cache(maxsize=1)
def isolation_map():
    """Return the V8(d) two-grain map: ``(patterns, exact, corners)``.

    Five 480 px patterns in a row.  Columns 0 and 1 are grain 0 with
    its reference at column 0; columns 2, 3 and 4 are grain 1 with its
    reference at column 4.  Column 2 is grain 1's HARD point, which
    the phase cross-correlation seed MEASURED cannot fit, and it sits
    in the eight-neighbourhood of grain 0's column 1.

    The construction is what makes a crossed seed VISIBLE, and both
    halves of it are measured.  (a) Column 1 is grain 0's easy point,
    whose residual MEASURED 8.04e-04, while column 3 is grain 1's own
    easy point, an out-of-plane tilt whose residual MEASURED 1.02e-02:
    a 12.75-fold ordering, so the LOWEST-residual neighbour of the
    hard point is the one across the grain boundary, and an
    implementation which forgot the grain gate would take it.  (b)
    From column 3's homography the hard point MEASURED converges in 19
    iterations to 0.100 px of the imposed field; from column 1's it
    exhausts a 200 iteration budget 30.9 px away, which is 85 per cent
    of the imposed displacement itself.
    """
    detector = make_detector()
    reference_a = oracle_reference()
    reference_b = project_pattern(detector, orientation_b())
    cross, h_cross = deformed_pattern(detector, orientation_a(), in_plane_fe(0.8))
    tilt = out_of_plane_fe(3.0)
    same, h_same = deformed_pattern(detector, orientation_b(), tilt)
    hard, h_hard = deformed_pattern(detector, orientation_b(), in_plane_fe(3.2) @ tilt)
    patterns = np.stack([reference_a, cross, hard, same, reference_b])
    patterns.flags.writeable = False
    zero = np.zeros(N_HOMOGRAPHY_PARAMETERS)
    exact = np.stack([zero, h_cross, h_hard, h_same, zero])
    corners = subregion_corners(SHAPE, pc_pixels_of(detector))
    return patterns, exact, corners


@functools.lru_cache(maxsize=1)
def isolation_seeded():
    """Return the SEEDED path's properties on the two-grain map."""
    patterns, _, _ = isolation_map()
    return run_map(
        patterns,
        ISOLATION_NAVIGATION_SHAPE,
        ISOLATION_REFERENCES,
        seeded=True,
        grain_labels=ISOLATION_LABELS,
    )


@functools.lru_cache(maxsize=1)
def pin_map():
    """Return the four cheapest ramp points, the D20.1 pin fixture.

    The last of them is outside the phase cross-correlation capture
    range, so the default path exhausts its budget there and the pin
    covers a non-converged point as well as three converged ones.
    """
    detector = make_detector()
    patterns = [oracle_reference()]
    for angle in RAMP_ANGLES[1:4]:
        patterns.append(
            deformed_pattern(detector, orientation_a(), in_plane_fe(angle))[0]
        )
    stack = np.stack(patterns)
    stack.flags.writeable = False
    return stack


@functools.lru_cache(maxsize=1)
def two_ramp_map():
    """Return the V8(c) determinism and simultaneous-seeding fixture:
    ``(patterns, mask, exact, corners)``.

    Rows 0 and 2 carry the SAME in-plane rotation ramp with OPPOSITE
    signs and row 1 is masked out entirely, so the two rows are never
    in each other's eight-neighbourhood and never see each other's
    results.  Each row's columns 0, 1 and 2 are inside the phase
    cross-correlation capture range and its columns 3 and 4 are not,
    so cascade round 1 is ``{(0, 3), (2, 3)}`` and round 2 is
    ``{(0, 4), (2, 4)}`` -- TWO points per round, carrying seeds 3.2
    degrees apart.

    That is the whole point of the map, and it is what the (3, 7)
    ramp map, the (1, 5) isolation map and the four-point pin map
    cannot do: MEASURED 2026-09-09, every cascade round on all three
    of those has exactly one member, so a per-round seed array which
    was permuted in flat order would be unobservable and a chunk-size
    pin would only ever vary the chunking of pass 1.  Here a swap is
    loud: MEASURED on this fixture, fitting row 0's column 3 from row
    2's column 2 (a seed 4.0 degrees away instead of 0.8) exhausts a
    200 iteration budget 47.6 px from the imposed field, 365 per cent
    of the 13.05 px it imposes, against 0.0096 px in 5 iterations
    from its own row's seed.
    """
    detector = make_detector()
    reference = oracle_reference()
    rows, exact_rows = [], []
    for sign in (1.0, -1.0):
        row, exact = [reference], [np.zeros(N_HOMOGRAPHY_PARAMETERS)]
        for angle in TWO_RAMP_ANGLES[1:]:
            pattern, h = deformed_pattern(
                detector, orientation_a(), in_plane_fe(sign * angle)
            )
            row.append(pattern)
            exact.append(h)
        rows.append(row)
        exact_rows.append(exact)
    columns = TWO_RAMP_NAVIGATION_SHAPE[1]
    patterns = np.stack(rows[0] + [reference] * columns + rows[1])
    patterns.flags.writeable = False

    mask = np.zeros(TWO_RAMP_NAVIGATION_SHAPE, dtype=bool)
    mask[1] = True
    mask.flags.writeable = False

    exact = np.stack(
        exact_rows[0] + [np.zeros(N_HOMOGRAPHY_PARAMETERS)] * columns + exact_rows[1]
    )
    corners = subregion_corners(SHAPE, pc_pixels_of(detector))
    return patterns, mask, exact, corners


def two_ramp_flat(row, column):
    """Return the flat map index of one two-ramp point."""
    return row * TWO_RAMP_NAVIGATION_SHAPE[1] + column


def run_two_ramp(**kwargs):
    """Return the SEEDED properties on the two-ramp map.

    Deliberately NOT cached: every caller of it is a determinism pin,
    and a pin which compares one cached dictionary with itself is the
    tautology this module's D20.1 pin was corrected for.
    """
    patterns, mask, _, _ = two_ramp_map()
    return run_map(
        patterns,
        TWO_RAMP_NAVIGATION_SHAPE,
        (0, 0),
        seeded=True,
        navigation_mask=mask,
        **kwargs,
    )


def seed_selection():
    """Return the frozen seed-choice helper of requirements D20.2.

    FROZEN by the Stage D failing-tests gate: the round orchestration
    computes, from results of STRICTLY EARLIER rounds only, which
    already converged neighbour seeds each candidate.  That choice is
    a pure function of arrays and is pinned as one, since the tie rule
    it implements is invisible in the public result (see the module
    docstring).

    Signature::

        choose_seed_indices(
            converged, residual, grain_id, navigation_shape, *,
            navigation_mask=None,
        ) -> np.ndarray

    every array flat in map order and the result the flat index of the
    chosen seed of each point, ``-1`` where no seed exists.
    """
    function = getattr(_engine, "choose_seed_indices", None)
    if function is None:
        raise AssertionError(
            "Stage D: `kikuchipy.indexing._hrebsd._engine.choose_seed_indices` does "
            "not exist yet (requirements D20.2). The failing-tests gate freezes the "
            "name, the argument order and the tie rule this module pins"
        )
    return function


def frozen_offsets():
    """Return the engine's declared neighbour offset order."""
    offsets = getattr(_engine, "NEIGHBOR_OFFSETS", None)
    if offsets is None:
        raise AssertionError(
            "Stage D: `kikuchipy.indexing._hrebsd._engine.NEIGHBOR_OFFSETS` does not "
            "exist yet (requirements D20.2), so the frozen tie order is declared "
            "nowhere"
        )
    return tuple(tuple(int(i) for i in offset) for offset in offsets)


# ============ D20.1 -- the default-off pin and the freeze =========== #


class TestDefaultOffIsBitwiseUnchanged:
    """The whole of requirements D20.1 at engine level: the new
    keyword's ``False`` default leaves the pre-Stage-D path alone, to
    the number and to the property name.  Pure regression, so it
    PASSES today.

    CORRECTED 2026-09-09 (adversarial review).  This class used to
    compare two LIVE calls -- one passing ``seed_from_neighbors=
    False`` and one omitting the keyword -- and assert they agreed.
    Both take the identical code path, so that is ``f(x) == f(x)``: it
    cannot fail, whatever a later Stage D commit does to the default
    path, and it was demonstrated not to fail under two injected
    default-path regressions which the entire hrebsd suite also
    passed.  The comparison is now against a FROZEN PRE-STAGE-D
    RESULT, measured on commit cec39de4, which is what D20.1 asks for
    and what makes a later Stage D commit provably harmless to the
    default path.  The two-live-calls check is kept as well, since
    the keyword must ALSO be inert, but it is no longer the pin.
    [D20.1/D20.5]
    """

    @staticmethod
    def default_run(**kwargs):
        return run_map(pin_map(), PIN_NAVIGATION_SHAPE, (0, 0), seeded=False, **kwargs)

    def test_the_keyword_exists_with_its_frozen_default(self):
        parameter = inspect.signature(run_hrebsd_dic).parameters["seed_from_neighbors"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False

    def test_the_default_path_still_gives_the_pre_stage_d_result(self):
        # THE pin of requirements D20.1, and the only test in this
        # module which can see a Stage D regression of the DEFAULT
        # path.  The expected arrays are frozen literals measured on
        # the last pre-Stage-D commit (see the block above), not a
        # second call to the code under test
        _, _, _, corners = ramp_map()
        properties = self.default_run()
        assert set(properties) == set(STAGE_A_PROP_NAMES)
        # the integer and boolean bookkeeping, EXACTLY.  This alone
        # kills every regression which changes the iteration count of
        # any point -- a capped fit, a moved convergence threshold, a
        # moved subregion border
        np.testing.assert_array_equal(
            np.asarray(properties["num_iterations"]), PRE_STAGE_D_NUM_ITERATIONS
        )
        np.testing.assert_array_equal(
            np.asarray(properties["converged"]), PRE_STAGE_D_CONVERGED
        )
        # and the answer itself, in the D2.5 corner-displacement
        # metric so that the reference point's float noise around zero
        # is measured on the scale it matters at rather than
        # relatively.  This is what catches a regression which leaves
        # the iteration counts alone, such as a new Stage D call site
        # which forgets to forward ``upsample_factor``
        homography = np.asarray(properties["homography"])
        worst = max(
            recovery_error(homography[index], PRE_STAGE_D_HOMOGRAPHY[index], corners)
            for index in range(PRE_STAGE_D_HOMOGRAPHY.shape[0])
        )
        assert_within(worst, PRE_STAGE_D_PIN_TOL, "PRE_STAGE_D_PIN_TOL")
        # the residual is stored, not derived on the way out, so it
        # gets its own comparison.  ``atol`` exempts the reference
        # point, whose 1.78e-16 is float noise and not a quantity
        np.testing.assert_allclose(
            np.asarray(properties["residual"]),
            PRE_STAGE_D_RESIDUAL,
            rtol=1e-11,
            atol=1e-15,
        )

    def test_explicit_false_is_the_call_without_the_keyword(self):
        # the keyword is inert, which is a different statement from
        # the pin above and worth its own line: passing the frozen
        # default explicitly may not change anything either
        without = self.default_run()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            explicit = run_hrebsd_dic(
                pin_map(),
                PIN_NAVIGATION_SHAPE,
                make_detector(PIN_NAVIGATION_SHAPE),
                reference=(0, 0),
                max_iterations=MAX_ITERATIONS,
                verbose=0,
            )
        assert set(without) == set(explicit)
        for name in STAGE_A_PROP_NAMES:
            left = np.asarray(without[name])
            right = np.asarray(explicit[name])
            if np.issubdtype(left.dtype, np.floating):
                assert np.array_equal(left, right, equal_nan=True), name
            else:
                assert np.array_equal(left, right), name

    def test_the_default_path_emits_no_seed_round(self):
        # requirements D20.5's absence rule, which is the half a
        # positive test cannot reach: with the keyword off the engine
        # emits EXACTLY the pre-Stage-D property set
        properties = self.default_run()
        assert set(properties) == set(STAGE_A_PROP_NAMES)
        assert SEED_ROUND_PROP not in properties
        # and the engine's own Stage A list never grows the entry
        assert SEED_ROUND_PROP not in STAGE_A_PROP_NAMES


# ================= V8(a) -- the rescue oracle ======================= #


class TestRescueOracle:
    """A smooth in-plane rotation ramp whose far points lie outside
    the capture range of the translation-only seed while every
    neighbour step stays well inside it.  [D20.2/V8(a)]"""

    def test_the_ramp_is_the_construction_it_claims_to_be(self):
        # the premise of the whole class, asserted rather than
        # described: every STEP is inside the 2.0 degree capture range
        # requirements D5 measures for the phase cross-correlation
        # seed, and the far END is beyond 4 degrees, twice outside it
        steps = np.diff(RAMP_ANGLES)
        assert np.all(steps < 1.0)
        assert np.allclose(steps, RAMP_STEP_DEG)
        assert RAMP_ANGLES[-1] > 4.0

    def test_the_ramp_stays_inside_the_pattern(self):
        # the geometric budget of D4.4, computed rather than trusted:
        # an in-plane rotation about the projection centre carries the
        # subregion corners around it, and if any of them leaves the
        # pattern the fit measures the mirror boundary of D3 instead
        # of the engine.  This is why this module moves the projection
        # centre in from ``test_hrebsd_engine.py``'s
        nrows, ncols = SHAPE
        pc_px = pc_pixels_of(make_detector())
        corners = subregion_corners(SHAPE, pc_px)
        x_low, x_high = 0.5 - pc_px[0], ncols - 0.5 - pc_px[0]
        y_low, y_high = 0.5 - pc_px[1], nrows - 0.5 - pc_px[1]
        # the two-ramp map's NEGATIVE angles are swept as well: the
        # projection centre is off the pattern centre, so a sign flip
        # is not a symmetry of this budget and may not be assumed
        angles = set(RAMP_ANGLES) | {
            sign * a for sign in (1, -1) for a in TWO_RAMP_ANGLES
        }
        for angle in sorted(angles):
            matrix = in_plane_fe(angle)
            x, y = corners[:, 0], corners[:, 1]
            warped_x = matrix[0, 0] * x + matrix[0, 1] * y
            warped_y = matrix[1, 0] * x + matrix[1, 1] * y
            assert warped_x.min() > x_low and warped_x.max() < x_high, angle
            assert warped_y.min() > y_low and warped_y.max() < y_high, angle

    def test_the_default_path_fails_the_far_points(self):
        # THE premise of V8(a), measured at run time and not quoted:
        # without seeding the far half of the ramp is not recovered at
        # all.  Both halves are asserted, so a fixture which drifted
        # into an easy ramp fails HERE and says so
        _, _, exact, corners = ramp_map()
        properties = ramp_default()
        homography = np.asarray(properties["homography"])
        for column, angle in enumerate(RAMP_ANGLES):
            error = recovery_error(homography[column], exact[column], corners)
            imposed = corner_norm(matrix_of(exact[column]), corners)
            if angle <= 1.6:
                assert properties["converged"][column], angle
                assert error < SAME_OPTIMUM_FRACTION * max(imposed, 1.0), angle
            else:
                assert error > 1.0, angle
                assert error > SAME_OPTIMUM_FRACTION * imposed, angle
        # and the point which the whole capture-range record is about:
        # beyond 4 degrees the default path is nowhere near the truth
        assert recovery_error(homography[6], exact[6], corners) > corner_norm(
            matrix_of(exact[6]), corners
        )

    def test_the_seeded_path_recovers_the_whole_ramp(self):
        _, _, exact, corners = ramp_map()
        properties = ramp_seeded()
        homography = np.asarray(properties["homography"])
        converged = np.asarray(properties["converged"])
        worst = 0.0
        for column in range(len(RAMP_ANGLES)):
            assert converged[column], column
            error = recovery_error(homography[column], exact[column], corners)
            worst = max(worst, error)
            # and no point's fit is closer to another point's imposed
            # warp than to its own, which makes a seed taken from the
            # wrong neighbour visible with no tolerance at all
            others = [
                recovery_error(homography[column], exact[other], corners)
                for other in range(len(RAMP_ANGLES))
                if other != column
            ]
            assert error < min(others), column
        assert_within(worst, SEED_RESCUE_TOL, "SEED_RESCUE_TOL")

    def test_the_reference_is_round_zero_by_construction(self):
        properties = ramp_seeded()
        assert properties["converged"][0]
        assert int(np.asarray(properties[SEED_ROUND_PROP])[0]) == SEED_ROUND_PASS1


# ================ V8(b) -- the D20.6 equivalence band =============== #


class TestEquivalence:
    """Seeding may change HOW the optimum is reached and never WHICH
    optimum it is, ON THE POINTS THE DEFAULT PATH CONVERGES WITHIN THE
    RUN'S OWN ITERATION BUDGET (requirements D20.6, narrowed by a
    dated correction on 2026-09-09).

    The unqualified form of D20.6 is FALSE and was measured false on
    this very fixture: at ``max_iterations=2000`` the default path
    converges the four far ramp points, in 291 to 728 iterations, to
    about 50 px from the imposed field, while the seeded path reaches
    each in 5 iterations to better than 0.015 px.  A seed moving a
    point out of a spurious far basin into the correct one is the
    whole benefit of D20, and calling it an equivalence violation
    would be backwards; what the requirement can honestly claim is
    the narrowed statement above.  [D20.6/V8(b)]
    """

    def test_both_paths_agree_where_both_converge(self):
        _, _, _, corners = ramp_map()
        default = ramp_default()
        seeded = ramp_seeded()
        both = np.asarray(default["converged"]) & np.asarray(seeded["converged"])
        assert both.any()
        # the ramp's easy points, which both paths reach in pass one,
        # AND the rescue point, which the default path reaches in 113
        # iterations and the seeded path only after its pass-1 budget
        # runs out: a genuinely different route to the same answer
        assert both[0] and both[1] and both[2]
        assert both[RAMP_RESCUE_INDEX]
        # and the population is pinned EXACTLY, because
        # ``MAX_ITERATIONS = 200`` is load bearing for what this test
        # is allowed to claim: raise it past about 291 and the four
        # far ramp points join ``both`` at a different optimum, and
        # the band below becomes a 50 px disagreement rather than a
        # machine-precision one.  A budget change must fail HERE and
        # say so, not silently widen the claim
        assert set(np.flatnonzero(both).tolist()) == EQUIVALENCE_BOTH_CONVERGED, (
            "the both-converged population moved; MAX_ITERATIONS is load bearing for "
            "the narrowed D20.6 claim, see the SEED_EQUIVALENCE_TOL block"
        )
        worst = 0.0
        for index in np.flatnonzero(both):
            worst = max(
                worst,
                recovery_error(
                    np.asarray(seeded["homography"])[index],
                    np.asarray(default["homography"])[index],
                    corners,
                ),
            )
        assert_within(worst, SEED_EQUIVALENCE_TOL, "SEED_EQUIVALENCE_TOL")

    def test_seeding_never_loses_a_point_the_default_path_had(self):
        # the other direction of the same decision: the cascade may
        # only ADD converged points
        default = np.asarray(ramp_default()["converged"])
        seeded = np.asarray(ramp_seeded()["converged"])
        assert np.all(seeded[default])
        assert seeded.sum() > default.sum()


# ================= V8(c) -- the D20.3 determinism pins ============== #


def assert_bitwise(first, second):
    """Assert two seeded property dictionaries agree, property by
    property, in the Stage A sense of "bitwise": value equality with
    ``equal_nan``, which is what ``test_hrebsd_engine.py``'s
    ``test_two_runs_bitwise`` and its chunked twin use."""
    assert set(first) == set(second)
    for name in SEEDED_PROP_NAMES:
        left = np.asarray(first[name])
        right = np.asarray(second[name])
        assert left.dtype == right.dtype, name
        if np.issubdtype(left.dtype, np.floating):
            assert np.array_equal(left, right, equal_nan=True), name
        else:
            assert np.array_equal(left, right), name


class TestDeterminism:
    """The seed of every fit is a pure function of results from
    strictly earlier phases and of the frozen tie rules, so nothing
    depends on intra-round scheduling or on chunking.

    MOVED 2026-09-09 (adversarial review) from the four-point map to
    the two-ramp map, because "nothing depends on intra-round
    scheduling" is only exercised when a round HAS more than one
    member: on the four-point map every cascade round had exactly one
    point and therefore exactly one chunk at either chunk size, so
    the chunking these pins varied was pass 1's alone.  Each round
    here carries two points with seeds 3.2 degrees apart, and the two
    chunk sizes below put them in one chunk and in separate chunks.
    [D20.3/V8(c)]
    """

    run = staticmethod(run_two_ramp)
    assert_bitwise = staticmethod(assert_bitwise)

    def test_the_map_really_exercises_multi_point_cascade_rounds(self):
        # a determinism pin on a map where every point converges in
        # pass one would pin the default path over again, and one
        # where every round has a single member cannot see a
        # scheduling bug at all, so both premises are asserted
        properties = self.run()
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        converged = np.asarray(properties["converged"])
        _, mask, _, _ = two_ramp_map()
        flat = np.asarray(mask).ravel()
        assert np.all(converged[~flat])
        for row in (0, 2):
            for column in TWO_RAMP_PASS1_COLUMNS:
                assert int(seed_round[two_ramp_flat(row, column)]) == SEED_ROUND_PASS1
        # round 1 is columns 3 of both rows and round 2 is columns 4.
        # Asserted as an EQUALITY of the whole round membership, not as
        # a count of the list this loop just built: "the round has two
        # members" has to be a statement about the RUN, or it is the
        # tautology this module's D20.1 pin was corrected for
        for round_index, column in enumerate(TWO_RAMP_CASCADE_COLUMNS, start=1):
            members = {two_ramp_flat(row, column) for row in (0, 2)}
            assert set(np.flatnonzero(seed_round == round_index).tolist()) == members

    def test_two_seeded_runs_are_bitwise_identical(self):
        self.assert_bitwise(self.run(), self.run())

    def test_chunksize_invariance_is_bitwise(self):
        # one chunk for the whole map against one chunk per point, so
        # the two members of every cascade round share a chunk in the
        # first and never do in the second
        self.assert_bitwise(self.run(chunksize=1), self.run(chunksize=TWO_RAMP_SIZE))

    def test_lazy_equals_eager_bitwise(self):
        eager = self.run()
        patterns, mask, _, _ = two_ramp_map()
        lazy = run_map(
            da.from_array(np.asarray(patterns), chunks=(1, -1, -1)),
            TWO_RAMP_NAVIGATION_SHAPE,
            (0, 0),
            seeded=True,
            navigation_mask=mask,
        )
        self.assert_bitwise(eager, lazy)


class TestSimultaneousSeeding:
    """Two points fitted in the SAME cascade round, from DIFFERENT
    seeds, land on their own answers and not on each other's.

    ADDED 2026-09-09 (adversarial review) to close the one mutant of
    plan 10.3 which had no designed killer: "per-point h0 array off by
    one in flat order".  Every other fixture in this module fits at
    most one point per round, so a permutation of a per-round seed
    array is unobservable on them -- there is nothing to permute.

    The kill is measured, not hoped for.  On this fixture, fitting a
    point from the OTHER row's seed (4.0 degrees away instead of 0.8)
    exhausts the 200 iteration budget without converging, 47.6 and
    49.5 px from the imposed field in round 1 and 47.4 and 52.8 px in
    round 2, i.e. 272 to 379 per cent of what those points impose,
    against 0.0078 to 0.0143 px in 5 iterations from the right seed.
    So ``converged``, the own-versus-other comparison and the
    ``SAME_OPTIMUM_FRACTION`` band each kill it on their own.
    [D20.2/D20.3]
    """

    def test_each_point_of_a_round_gets_its_own_seed(self):
        patterns, mask, exact, corners = two_ramp_map()
        properties = run_two_ramp()
        homography = np.asarray(properties["homography"])
        converged = np.asarray(properties["converged"])
        flat = np.asarray(mask).ravel()
        worst = 0.0
        for index in np.flatnonzero(~flat):
            assert converged[index], index
            own = recovery_error(homography[index], exact[index], corners)
            imposed = corner_norm(matrix_of(exact[index]), corners)
            worst = max(worst, own)
            if imposed == 0.0:
                # the two undeformed points, which have no ramp of
                # their own to be closer to
                assert own < SAME_OPTIMUM_FRACTION, index
                continue
            # its own imposed field beats every other point's, with
            # no tolerance at all: a seed taken from the other row
            # cannot survive this line
            others = [
                recovery_error(homography[index], exact[other], corners)
                for other in np.flatnonzero(~flat)
                if other != index
            ]
            assert own < min(others), index
            assert own < SAME_OPTIMUM_FRACTION * imposed, index
        assert_within(worst, SEED_RESCUE_TOL, "SEED_RESCUE_TOL (two-ramp map)")

    def test_the_two_rows_never_see_each_other(self):
        # the construction which makes the two seeds of a round
        # genuinely independent: row 1 is masked, so rows 0 and 2 are
        # two rows apart and never in each other's
        # eight-neighbourhood.  Asserted rather than described, since
        # a fixture which lost the mask would quietly turn this class
        # into a second copy of the ramp oracle
        _, mask, exact, corners = two_ramp_map()
        assert np.all(np.asarray(mask)[1])
        assert not np.asarray(mask)[[0, 2]].any()
        # and the two rows really do impose different fields
        for column in TWO_RAMP_CASCADE_COLUMNS:
            top = exact[two_ramp_flat(0, column)]
            bottom = exact[two_ramp_flat(2, column)]
            gap = recovery_error(top, bottom, corners)
            assert gap > corner_norm(matrix_of(top), corners)


# ================== V8(d) -- the D20.4 isolation pins =============== #


class TestGrainIsolation:
    """A seed never crosses a ``grain_id`` boundary, on a map built so
    that a crossed seed corrupts the second grain visibly.
    [D20.4/V8(d)]"""

    seeded = staticmethod(isolation_seeded)

    def test_the_lowest_residual_neighbour_is_the_one_across_the_boundary(self):
        # THE premise which makes this map a test of isolation rather
        # than of luck: the hard point's cheapest neighbour by the
        # D20.2 rule sits in the OTHER grain, so an implementation
        # which dropped the grain gate would take it
        properties = self.seeded()
        residual = np.asarray(properties["residual"])
        converged = np.asarray(properties["converged"])
        assert converged[ISOLATION_CROSS_INDEX] and converged[ISOLATION_SAME_INDEX]
        assert residual[ISOLATION_CROSS_INDEX] < residual[ISOLATION_SAME_INDEX]
        grain_id = np.asarray(properties["grain_id"])
        assert grain_id[ISOLATION_CROSS_INDEX] != grain_id[ISOLATION_HARD_INDEX]
        assert grain_id[ISOLATION_SAME_INDEX] == grain_id[ISOLATION_HARD_INDEX]

    def test_the_hard_point_is_seeded_from_its_own_grain(self):
        _, exact, corners = isolation_map()
        properties = self.seeded()
        homography = np.asarray(properties["homography"])
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        assert np.asarray(properties["converged"])[ISOLATION_HARD_INDEX]
        # it really did go through the cascade rather than pass one,
        # which is what makes the seed load bearing here
        assert seed_round[ISOLATION_HARD_INDEX] >= 1
        imposed = exact[ISOLATION_HARD_INDEX]
        error = recovery_error(homography[ISOLATION_HARD_INDEX], imposed, corners)
        size = corner_norm(matrix_of(imposed), corners)
        assert error < SAME_OPTIMUM_FRACTION * size
        # and the run's answer is compared against what a CROSSED seed
        # actually gives, which is a RECORDED simulation (see
        # ``ISOLATION_CROSSED_SEED_ERROR``) and not something this
        # line measures: 30.8866 px, an unconverged point 84.67 per
        # cent of the imposed displacement from the truth.  Two orders
        # separate the two hypotheses
        assert error < 0.01 * ISOLATION_CROSSED_SEED_ERROR
        assert ISOLATION_CROSSED_SEED_ERROR > 0.5 * size

    def test_the_two_grains_impose_visibly_different_fields(self):
        # a static property of the FIXTURE, split out from the test
        # above on 2026-09-09 because it used to be asserted there
        # under a comment claiming it measured what a crossed seed
        # does.  It does not: it measures how far apart the two
        # imposed fields are, which is what makes the map a candidate
        # for the crossed-seed simulation, not evidence of its result
        _, exact, corners = isolation_map()
        imposed = exact[ISOLATION_HARD_INDEX]
        size = corner_norm(matrix_of(imposed), corners)
        gap = recovery_error(exact[ISOLATION_CROSS_INDEX], imposed, corners)
        assert gap > 0.5 * size

    def test_every_second_grain_point_matches_its_own_imposed_field(self):
        _, exact, corners = isolation_map()
        properties = self.seeded()
        homography = np.asarray(properties["homography"])
        for index in (ISOLATION_HARD_INDEX, ISOLATION_SAME_INDEX):
            own = recovery_error(homography[index], exact[index], corners)
            others = [
                recovery_error(homography[index], exact[other], corners)
                for other in range(exact.shape[0])
                if other != index
            ]
            assert own < min(others), index

    def test_each_reference_still_correlates_with_itself(self):
        _, _, corners = isolation_map()
        properties = self.seeded()
        homography = np.asarray(properties["homography"])
        for index in ISOLATION_REFERENCES.tolist():
            assert corner_norm(matrix_of(homography[index]), corners) < 1e-3
            assert int(np.asarray(properties[SEED_ROUND_PROP])[index]) == (
                SEED_ROUND_PASS1
            )


class TestMaskIsolation:
    """A masked point is neither fitted nor seeded nor a seed, and its
    unmasked neighbours still find seeds elsewhere.  [D20.4]"""

    def test_masked_points_are_untouched_and_carry_minus_one(self):
        _, mask, _, _ = ramp_map()
        properties = ramp_seeded()
        flat = np.asarray(mask).ravel()
        homography = np.asarray(properties["homography"])
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        assert flat.sum() == 11
        assert np.all(np.isnan(homography[flat]))
        assert np.all(np.isnan(np.asarray(properties["Fe"])[flat]))
        assert not np.asarray(properties["converged"])[flat].any()
        # no fit at all: not a capped one, not a rescued one
        assert np.all(np.asarray(properties["num_iterations"])[flat] == 0)
        # and no participation in the encoding beyond the -1 of D20.5
        assert np.all(seed_round[flat] == SEED_ROUND_NONE)

    def test_a_neighbour_of_masked_points_still_seeds_from_elsewhere(self):
        # every ramp point of row 0 has three MASKED neighbours in row
        # 1, so a cascade which stopped at the first masked
        # neighbour, or which counted one as converged, would never
        # walk along the row.  It does: the far ramp points are
        # reached, and their rounds increase along the row
        properties = ramp_seeded()
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        assert np.all(np.asarray(properties["converged"])[:7])
        assert np.all(np.diff(seed_round[3:7]) == 1)


# ============ V8(e) -- the D20.5 encoding and PASS1_CAP ============= #


class TestSeedRoundEncoding:
    """The one new property of requirements D20.5, pinned VALUE BY
    VALUE on a map constructed to produce each of them.

    The ``EXPECTED`` array below holds for a WINDOW of ``PASS1_CAP``
    values, and the window is recorded here so that the
    implementation gate can pin the constant knowing what it may not
    cross.  It follows from two measurements re-confirmed 2026-09-09:
    the 1.6 degree ramp point converges from its own phase
    cross-correlation seed in 10 iterations, and the isolated rescue
    point in 113.

      cap <= 9 ..... the 1.6 degree point drops out of pass 1, so
                     the whole ramp shifts by one round and
                     ``EXPECTED`` becomes ``[0, 0, 1, 2, 3, 4, 5]``
      10 <= cap <= 112 ... ``EXPECTED`` exactly as written
      cap >= 113 ... the rescue point converges IN PASS 1, so
                     ``seed_round[17]`` becomes 0 rather than -2 and
                     the rescue pass stops being exercised at all

    The drafting candidate of 50 sits comfortably inside.  [D20.5]
    """

    # The expected property, written out.  Row 0: the three points
    # inside the capture range converge in pass one and the four
    # outside it are converted one per cascade round, walking along
    # the row.  Row 1: masked.  Row 2: the isolated unfittable point,
    # the isolated rescue point and the isolated failed point, with
    # masked points between them
    EXPECTED = np.array(
        [0, 0, 0, 1, 2, 3, 4]
        + [SEED_ROUND_NONE] * 7
        + [
            SEED_ROUND_NONE,
            SEED_ROUND_NONE,
            SEED_ROUND_NONE,
            SEED_ROUND_RESCUE,
            SEED_ROUND_NONE,
            SEED_ROUND_NONE,
            SEED_ROUND_NONE,
        ],
        dtype=np.int32,
    )

    def test_the_property_is_there_with_its_frozen_dtype(self):
        properties = ramp_seeded()
        assert set(properties) == set(SEEDED_PROP_NAMES)
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        assert seed_round.dtype == np.int32
        assert seed_round.shape == (RAMP_SIZE,)

    def test_every_encoded_value_is_the_constructed_one(self):
        properties = ramp_seeded()
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        np.testing.assert_array_equal(seed_round, self.EXPECTED)
        # the map really does exercise all five kinds of entry
        assert set(np.unique(seed_round).tolist()) == {-2, -1, 0, 1, 2, 3, 4}

    def test_the_encoding_agrees_with_convergence_everywhere(self):
        # the invariant behind the values: a point carries -1 exactly
        # when it is masked or did not converge, and any other entry
        # exactly when it did
        _, mask, _, _ = ramp_map()
        properties = ramp_seeded()
        seed_round = np.asarray(properties[SEED_ROUND_PROP])
        converged = np.asarray(properties["converged"])
        flat = np.asarray(mask).ravel()
        np.testing.assert_array_equal(seed_round == SEED_ROUND_NONE, ~converged | flat)
        assert np.all(seed_round[converged] >= SEED_ROUND_RESCUE)
        assert np.all(seed_round[converged] != SEED_ROUND_NONE)


class TestPass1CapSemantics:
    """Requirements D20.2's last sentence: no point is ever left with
    a cap-truncated pass-1 iterate as its final answer.  [D20.2]"""

    def test_the_rescue_pass_catches_the_isolated_slow_point(self):
        # THE pin that the pass-1 cap exists AND that the rescue pass
        # undoes it.  This point's whole neighbourhood is masked, so
        # no cascade round can ever reach it; MEASURED, it converges
        # from its own phase cross-correlation seed in 113 iterations
        # of a 200 iteration budget, so a pass 1 capped anywhere below
        # 113 must leave it unconverged and the rescue pass must then
        # convert it.  If the implementation gate pins ``PASS1_CAP``
        # at or above 113 -- far above the drafting candidate 50 and
        # above anything the far-field median of 5 to 14 iterations
        # justifies -- this fixture is the thing to re-tune, and it
        # fails HERE and says so rather than passing vacuously
        _, _, exact, corners = ramp_map()
        properties = ramp_seeded()
        index = RAMP_RESCUE_INDEX
        assert np.asarray(properties["converged"])[index]
        assert int(np.asarray(properties[SEED_ROUND_PROP])[index]) == SEED_ROUND_RESCUE
        assert int(np.asarray(properties["num_iterations"])[index]) > 0
        # and it converged to the IMPOSED field, not to some other
        # optimum the truncated iterate happened to sit near.
        #
        # HEADROOM WARNING, measured 2026-09-09: this is the TIGHTEST
        # assertion in the module.  The point recovers to 0.388833 px
        # of the 91.2533 px it imposes, which is 0.426 per cent
        # against the 1 per cent bound, i.e. 2.35x -- next to 23x for
        # the easiest ramp point and 3.6x for the isolation hard
        # point.  The looseness is the out-of-plane tilt's own
        # projective conditioning, not a defect, but a future tweak of
        # ``RAMP_RESCUE_TILT`` has very little slack to spend
        imposed = exact[index]
        error = recovery_error(
            np.asarray(properties["homography"])[index], imposed, corners
        )
        assert error < SAME_OPTIMUM_FRACTION * corner_norm(matrix_of(imposed), corners)

    def test_no_final_answer_is_a_cap_truncated_pass_one_iterate(self):
        # the unfittable point never converges, from any seed, and its
        # bookkeeping must still show the FULL budget: a point left
        # with its pass-1 iterate would show the cap instead.  The
        # comparison is ``>=`` so that it reads the same whether the
        # implementation stores the last fit's own count or the sum
        # over the fits of that point
        properties = ramp_seeded()
        num_iterations = np.asarray(properties["num_iterations"])
        converged = np.asarray(properties["converged"])
        _, mask, _, _ = ramp_map()
        flat = np.asarray(mask).ravel()
        index = RAMP_UNFITTABLE_INDEX
        assert not converged[index]
        assert num_iterations[index] >= MAX_ITERATIONS
        # and the statement over the whole map: an unmasked point
        # which neither converged nor failed outright has been fitted
        # at the full budget
        for point in np.flatnonzero(~converged & ~flat):
            if point == RAMP_FAILED_INDEX:
                # the D2.6 no-variance contract, which never iterates
                assert num_iterations[point] == 0
                assert np.all(np.isnan(np.asarray(properties["homography"])[point]))
                continue
            assert num_iterations[point] >= MAX_ITERATIONS, point

    def test_a_failed_pattern_still_carries_the_stage_a_contract(self):
        # the seeded path may not quietly change the D2.6 failure
        # contract: NaN properties, no iterations, ``converged=False``
        # and, now, ``seed_round=-1``
        properties = ramp_seeded()
        index = RAMP_FAILED_INDEX
        assert np.all(np.isnan(np.asarray(properties["homography"])[index]))
        assert np.isnan(np.asarray(properties["residual"])[index])
        assert not np.asarray(properties["converged"])[index]
        assert int(np.asarray(properties[SEED_ROUND_PROP])[index]) == SEED_ROUND_NONE


class TestVerboseProgress:
    """The Stage D progress messages, which the seeded path alone
    prints and only at ``verbose >= 1``.  Not a frozen oracle: the
    counts they carry are pinned by the fixtures elsewhere, so this
    asserts only that a run exercising both phases emits both the
    per-round cascade line and the rescue-pass line.  [D20.2]"""

    def test_the_seeded_run_prints_the_cascade_and_rescue_lines(self, capsys):
        # the ramp map walks the cascade along row 0 one point per
        # round (each far point is seeded from its inner neighbour once
        # that neighbour has converged) and rescues the isolated slow
        # point at RAMP_RESCUE_INDEX, whose whole neighbourhood is
        # masked, so a verbose seeded run must print BOTH messages
        patterns, mask, _, _ = ramp_map()
        run_map(
            patterns,
            RAMP_NAVIGATION_SHAPE,
            (0, 0),
            seeded=True,
            navigation_mask=mask,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert "Cascade round 1:" in out
        assert "Rescue pass:" in out


# ========== D20.2 -- the seed choice, the tie and the gates ========= #


class TestSeedChoice:
    """The frozen seed-choice rule of requirements D20.2 as a unit
    test of the pure function it is: the LOWEST-residual converged
    same-grain unmasked neighbour, ties broken by the frozen offset
    order.

    Why here rather than through the public method: a residual tie has
    to be EXACT to be a tie, and the only way to make two neighbours'
    residuals bitwise equal is to give them bitwise equal patterns,
    which makes their homographies bitwise equal too -- so which of
    them won is invisible in every property the run returns.  The
    lowest-residual rule and both gates are sharper here as well,
    since a constructed array says exactly what a pattern fixture only
    implies.  [D20.2/D20.4]
    """

    SHAPE_3 = (3, 3)

    @staticmethod
    def arrays(shape, converged, residual, grain_id=None):
        """Return the three flat inputs of the choice function."""
        size = shape[0] * shape[1]
        if grain_id is None:
            grain_id = np.zeros(size, dtype=np.int32)
        return (
            np.asarray(converged, dtype=bool).reshape(size),
            np.asarray(residual, dtype=np.float64).reshape(size),
            np.asarray(grain_id, dtype=np.int32).reshape(size),
        )

    def test_the_offset_order_is_the_frozen_one(self):
        assert frozen_offsets() == FROZEN_NEIGHBOR_OFFSETS

    def test_the_lowest_residual_converged_neighbour_wins(self):
        # three converged neighbours of the centre with clearly
        # separated residuals; the cheapest one is the seed, which is
        # the whole reliability-guided idea and the one thing the
        # "seed from the highest residual" mutant reverses
        converged = np.zeros(9, dtype=bool)
        converged[[0, 2, 6]] = True
        residual = np.full(9, np.nan)
        residual[0], residual[2], residual[6] = 0.5, 0.1, 0.9
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual), self.SHAPE_3
        )
        assert int(chosen[4]) == 2

    # A tie's two candidates, as flat indices of the 3 by 3 map whose
    # centre is index 4, and the index the FROZEN order picks.
    #
    # RECORDED 2026-09-09: the frozen order is row-major reading
    # order, and the eight offsets map to the flat deltas -W-1, -W,
    # -W+1, -1, +1, W-1, W, W+1, which is ASCENDING for every map
    # width W >= 2.  So on a complete eight-neighbourhood the frozen
    # order picks the same winner as a plain flat-index argmin, and a
    # tie case can only discriminate it from a REORDERED traversal,
    # never from an index sort.  What the freeze buys is that the
    # traversal is not reordered; the ``NEIGHBOR_OFFSETS`` pin above
    # is what states the order itself.
    #
    # The three hand-picked pairs are the discriminating ones.  The
    # first two separate the frozen order from the column-major one
    # ((-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)),
    # the plausible shuffle: there ``(0, -1)`` at flat 3 would beat
    # both ``(-1, 0)`` at flat 1 and ``(-1, 1)`` at flat 2.  The third
    # separates it from a reversed order
    TIE_CASES = [((1, 3), 1), ((2, 3), 2), ((1, 8), 1)]

    # and the completeness sweep: all 28 pairs of the eight
    # neighbours, added 2026-09-09 so that the tie rule is pinned over
    # its whole domain and not on three samples of it
    NEIGHBOR_FLAT_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
    ALL_TIE_CASES = [
        (pair, min(pair)) for pair in itertools.combinations(NEIGHBOR_FLAT_INDICES, 2)
    ]

    @pytest.mark.parametrize("candidates,winner", TIE_CASES)
    def test_an_exact_residual_tie_is_broken_by_the_frozen_order(
        self, candidates, winner
    ):
        # the two candidates carry the SAME float, bit for bit, so
        # nothing but the offset order can decide between them
        converged = np.zeros(9, dtype=bool)
        converged[list(candidates)] = True
        residual = np.full(9, np.nan)
        residual[list(candidates)] = 0.25
        assert residual[candidates[0]] == residual[candidates[1]]
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual), self.SHAPE_3
        )
        assert int(chosen[4]) == winner

    @pytest.mark.parametrize("candidates,winner", ALL_TIE_CASES)
    def test_every_tie_pair_is_broken_the_same_way(self, candidates, winner):
        assert len(self.ALL_TIE_CASES) == 28
        converged = np.zeros(9, dtype=bool)
        converged[list(candidates)] = True
        residual = np.full(9, np.nan)
        residual[list(candidates)] = 0.25
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual), self.SHAPE_3
        )
        assert int(chosen[4]) == winner

    def test_an_unconverged_neighbour_is_never_a_seed(self):
        converged = np.zeros(9, dtype=bool)
        converged[0] = True
        residual = np.full(9, 0.4)
        residual[2] = 1e-9  # the cheapest, and not converged
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual), self.SHAPE_3
        )
        assert int(chosen[4]) == 0

    def test_no_seed_crosses_a_grain_boundary(self):
        # the cheapest neighbour is in the other grain, exactly as the
        # two-grain pattern fixture is built, and must be refused
        converged = np.zeros(9, dtype=bool)
        converged[[1, 7]] = True
        residual = np.full(9, np.nan)
        residual[1], residual[7] = 1e-6, 0.5
        grain_id = np.zeros(9, dtype=np.int32)
        grain_id[[0, 1, 2]] = 1
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual, grain_id), self.SHAPE_3
        )
        assert int(chosen[4]) == 7
        # and with no same-grain converged neighbour there is no seed
        converged = np.zeros(9, dtype=bool)
        converged[1] = True
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual, grain_id), self.SHAPE_3
        )
        assert int(chosen[4]) == -1

    def test_a_masked_point_neither_seeds_nor_is_seeded(self):
        converged = np.zeros(9, dtype=bool)
        converged[[1, 7]] = True
        residual = np.full(9, np.nan)
        residual[1], residual[7] = 1e-6, 0.5
        mask = np.zeros(self.SHAPE_3, dtype=bool)
        mask[0, 1] = True  # the cheapest neighbour, masked out
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual),
            self.SHAPE_3,
            navigation_mask=mask,
        )
        assert int(chosen[4]) == 7
        mask[1, 1] = True  # the candidate itself, masked out
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual),
            self.SHAPE_3,
            navigation_mask=mask,
        )
        assert int(chosen[4]) == -1

    def test_the_map_edge_is_not_wrapped(self):
        # a corner point has three neighbours and not eight, and a
        # flat-index arithmetic which forgot the row boundary would
        # hand it one from the opposite edge
        converged = np.zeros(9, dtype=bool)
        converged[3] = True
        residual = np.full(9, np.nan)
        residual[3] = 0.2
        chosen = seed_selection()(
            *self.arrays(self.SHAPE_3, converged, residual), self.SHAPE_3
        )
        assert int(chosen[2]) == -1
        assert int(chosen[0]) == 3


class TestSeedChoiceIsTheSeamTheCascadeUses:
    """The round orchestration really goes through the frozen helper
    :class:`TestSeedChoice` pins.

    ADDED 2026-09-09 (adversarial review).  Without it the freeze is
    decorative: an implementation could define a conformant
    ``choose_seed_indices``, pass every unit test above, and then seed
    by entirely different logic inside the cascade.  Three of the ten
    mutants of plan 10.3 -- "seed from the HIGHEST-residual
    neighbour", "tie order shuffled" and "seed from an unconverged
    neighbour" -- are killed by :class:`TestSeedChoice` ALONE, because
    MEASURED on every fixture in this module every seeding event has
    exactly ONE same-grain converged candidate, so which candidate won
    is invisible in any property a seeded run returns.  Those three
    mutants therefore rest entirely on this seam.

    What this FREEZES beyond the helper's own contract: the cascade
    must call it through the module-global name, so that it is
    interceptable here.  That is a real constraint on the
    implementation and is stated so the implementation gate can meet
    it deliberately rather than by accident.  [D20.2/D20.3]
    """

    def test_the_orchestration_calls_the_frozen_helper_every_round(self, monkeypatch):
        original = seed_selection()
        calls = []

        @functools.wraps(original)
        def spy(converged, residual, grain_id, navigation_shape, **kwargs):
            calls.append(
                (
                    np.array(converged),
                    np.array(residual),
                    np.array(grain_id),
                    tuple(navigation_shape),
                    dict(kwargs),
                )
            )
            return original(converged, residual, grain_id, navigation_shape, **kwargs)

        monkeypatch.setattr(_engine, "choose_seed_indices", spy)
        properties = run_two_ramp()

        # two cascade rounds fire on this map, and a third call is
        # expected for the round which converts nothing and ends the
        # cascade, so ``>=`` rather than ``==``
        assert len(calls) >= len(TWO_RAMP_CASCADE_COLUMNS), (
            "the cascade did not route through `_engine.choose_seed_indices`; the "
            "seed-choice unit tests above then pin a function nothing uses"
        )
        for converged, residual, grain_id, shape, _ in calls:
            # whole-map arrays in flat map order, which is the
            # helper's frozen contract
            assert shape == TWO_RAMP_NAVIGATION_SHAPE
            assert converged.shape == (TWO_RAMP_SIZE,)
            assert residual.shape == (TWO_RAMP_SIZE,)
            assert grain_id.shape == (TWO_RAMP_SIZE,)
            assert converged.dtype == bool
        # the calls really did see a growing frontier, which is what
        # "results of STRICTLY EARLIER rounds only" means: no call may
        # already know about a point a later round converts
        counts = [int(call[0].sum()) for call in calls]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]
        # and the run itself still came out right, so this is not
        # measuring a broken monkeypatched run
        assert np.all(
            np.asarray(properties["converged"])[~np.asarray(two_ramp_map()[1]).ravel()]
        )


# ============== D20.2 -- the PC-transport justification ============= #


class TestPcTransportBound:
    """Requirements D20.2 copies a neighbour's RAW homography as the
    seed WITHOUT transporting it between projection centre frames, and
    asks the build to MEASURE the bound that justifies it.

    A pure library measurement on the geometry validation.md's ledger
    entry 80 records for the Si-indent data set, so it needs no
    ``_hrebsd`` code and passes before the cascade lands.

    CORRECTED 2026-09-09 (adversarial review).  This class used to
    invent a central projection centre, ``(0.5 * ncols, 0.5 * nrows,
    0.65 * nrows)``, while advertising the result as a measurement on
    the Si geometry.  It now uses the REAL one, READ from
    ``AGH__Si_indent_1_512x672.h5oina`` on 2026-09-09 and converted
    into the kikuchipy (Bruker) frame with the fork's own Oxford
    conversion written out below.  The verdict did not change -- the
    worst case moved from 0.009478 to 0.009727 px -- but the
    provenance is now what the docstring says it is, and the three
    other defensible readings of the file's fractions are swept as a
    sensitivity check rather than assumed away.  [D20.2]
    """

    # ledger entry 80: 512 rows by 622 columns at binning 2, a 234 by
    # 250 map at 0.2 um, projection centre spans of 1.31, 1.14 and
    # 0.47 BINNED pixels over the whole map
    SI_SHAPE = (512, 622)
    SI_MAP_SHAPE = (234, 250)
    SI_PC_SPANS = (1.31, 1.14, 0.47)

    # The file's own ``/1/EBSD/Data`` means, READ 2026-09-09: Pattern
    # Center X 0.526845 (span 0.002110), Pattern Center Y 0.678911
    # (span 0.001827), Detector Distance 0.610303 (span 0.000759).
    # Oxford normalises ALL THREE by the pattern WIDTH, which is why
    # the three spans above times 622 give the ledger's 1.31, 1.14 and
    # 0.47 binned px -- the arithmetic that reconciles the file with
    # the ledger, and the reason the ledger's numbers are already in
    # the units this test works in
    SI_PC_OXFORD = (0.526845, 0.678911, 0.610303)

    @classmethod
    def pc_pixels(cls):
        """Return the REAL projection centre in binned pixels, in the
        kikuchipy (Bruker) frame the engine works in.

        The Oxford-to-Bruker conversion of
        :class:`~kikuchipy.detectors.EBSDDetector` written out rather
        than called: ``PCy_bruker = 1 - PCy_oxford * aspect_ratio``
        and ``PCz_bruker = PCz_oxford * aspect_ratio``, with the
        aspect ratio ``ncols / nrows``.  An oracle which shares an
        error with the code it judges cannot see that error, and this
        one has to be readable next to the numbers it produces.
        """
        nrows, ncols = cls.SI_SHAPE
        pcx, pcy, pcz = cls.SI_PC_OXFORD
        aspect = ncols / nrows
        return np.array(
            [pcx * ncols, (1.0 - pcy * aspect) * nrows, (pcz * aspect) * nrows]
        )

    @classmethod
    def steps(cls):
        """Return the worst per-step projection centre drift, which is
        each whole span divided by the number of steps along its own
        axis."""
        map_rows, map_cols = cls.SI_MAP_SHAPE
        span_x, span_y, span_dd = cls.SI_PC_SPANS
        return (
            span_x / (map_cols - 1),
            span_y / (map_rows - 1),
            span_dd / (map_rows - 1),
        )

    @classmethod
    def geometry(cls, pc_px=None):
        """Return ``(corners, dd_px, per-step drifts)`` of that map."""
        if pc_px is None:
            pc_px = cls.pc_pixels()
        pc_px = np.asarray(pc_px, dtype=np.float64)
        return subregion_corners(cls.SI_SHAPE, pc_px), float(pc_px[2]), cls.steps()

    @staticmethod
    def phantom(step_x, step_y, step_dd, dd):
        """Return the D6.2 beam-scan phantom between two neighbouring
        points, written out here rather than imported: it is the whole
        difference between a transported seed and a copied one."""
        alpha = (dd + step_dd) / dd
        return np.array([[alpha, 0.0, step_x], [0.0, alpha, step_y], [0.0, 0.0, 1.0]])

    @classmethod
    def worst_transport(cls, pc_px=None):
        """Return ``(worst phantom norm, that norm over the capture
        range)`` in binned pixels for one projection centre."""
        corners, dd, (step_x, step_y, step_dd) = cls.geometry(pc_px)
        # a typical converged homography of the deformed zone: ledger
        # entry 80's south rim patch, 11.9 px of translation and
        # 42 mrad of lattice rotation.  The bound is taken over the
        # phantom itself and over both conjugations of it by that
        # deformation, since which side a seed's own deformation sits
        # on is an implementation detail of the composition
        theta = 42.08e-3
        typical = np.array(
            [
                [np.cos(theta), -np.sin(theta), 11.9],
                [np.sin(theta), np.cos(theta), 11.9],
                [0.0, 0.0, 1.0],
            ]
        )
        worst = 0.0
        for dx, dy in ((step_x, 0.0), (0.0, step_y), (step_x, step_y)):
            phantom = cls.phantom(dx, dy, step_dd, dd)
            inverse = np.linalg.inv(typical)
            for candidate in (
                phantom,
                inverse @ phantom @ typical,
                typical @ phantom @ inverse,
            ):
                worst = max(worst, corner_norm(candidate, corners))
        # the statement the decision actually rests on: that bound is
        # negligible against the capture range a seed has to land
        # inside, the 2.0 degree in-plane rotation requirements D5
        # measures for the phase cross-correlation seed
        basin = corner_norm(in_plane_fe(2.0), corners)
        return worst, worst / basin

    def test_the_transport_a_copied_seed_skips_is_bounded(self):
        worst, fraction = self.worst_transport()
        assert_within(worst, PC_TRANSPORT_BOUND, "PC_TRANSPORT_BOUND")
        assert_within(
            fraction, PC_TRANSPORT_BASIN_FRACTION, "PC_TRANSPORT_BASIN_FRACTION"
        )

    def test_the_bound_survives_every_reading_of_the_recorded_fractions(self):
        # The sensitivity check the corrected provenance calls for.
        # Oxford's three fractions are all normalised by the pattern
        # WIDTH and its PCy runs from the opposite edge to kikuchipy's,
        # so a reader coming to the ledger cold has four defensible
        # ways to turn them into pixels.  All four are swept, and the
        # decision D20.2 rests on holds under every one -- MEASURED
        # 2026-09-09: 0.009727, 0.008578, 0.009727 and 0.009171 px,
        # and 5.637e-04, 4.972e-04, 5.637e-04 and 6.014e-04 of the
        # capture range.  The invented central stand-in this class
        # used to use is swept with them, at 0.009478 px and
        # 7.508e-04, so the correction can be seen not to have moved
        # the verdict
        nrows, ncols = self.SI_SHAPE
        pcx, pcy, pcz = self.SI_PC_OXFORD
        aspect = ncols / nrows
        readings = {
            "kikuchipy Bruker frame": self.pc_pixels(),
            "raw Oxford, PCy from the top": np.array(
                [pcx * ncols, pcy * ncols, pcz * ncols]
            ),
            "raw Oxford, PCy from the bottom": np.array(
                [pcx * ncols, nrows - pcy * ncols, pcz * ncols]
            ),
            "fractions of the pattern HEIGHT": np.array(
                [pcx * ncols, pcy * nrows, pcz * nrows]
            ),
            "the superseded central stand-in": np.array(
                [0.5 * ncols, 0.5 * nrows, 0.65 * nrows]
            ),
        }
        for label, pc_px in readings.items():
            worst, fraction = self.worst_transport(pc_px)
            assert_within(worst, PC_TRANSPORT_BOUND, f"PC_TRANSPORT_BOUND ({label})")
            assert_within(
                fraction,
                PC_TRANSPORT_BASIN_FRACTION,
                f"PC_TRANSPORT_BASIN_FRACTION ({label})",
            )
        assert aspect == pytest.approx(1.2148, rel=1e-3)

    def test_the_recorded_geometry_is_the_file_s_own(self):
        # the geometry this bound is measured on, asserted so that a
        # future edit of the numbers fails here and names itself
        _, _, (step_x, step_y, step_dd) = self.geometry()
        assert step_x == pytest.approx(5.26e-3, rel=0.01)
        assert step_y == pytest.approx(4.89e-3, rel=0.01)
        assert step_dd == pytest.approx(2.02e-3, rel=0.01)
        # and the reconciliation of the file's fractions with the
        # ledger's binned pixels: Oxford normalises all three by the
        # pattern WIDTH, so each recorded span is the file's span
        # times ``ncols``.  READ from the file 2026-09-09: spans of
        # 0.002110, 0.001827 and 0.000759
        _, ncols = self.SI_SHAPE
        for span, oxford_span in zip(self.SI_PC_SPANS, (0.002110, 0.001827, 0.000759)):
            assert span == pytest.approx(oxford_span * ncols, rel=0.01)
        # the real projection centre, in the frame the engine uses
        pc_px = self.pc_pixels()
        assert pc_px[0] == pytest.approx(327.70, rel=1e-3)
        assert pc_px[1] == pytest.approx(89.72, rel=1e-3)
        assert pc_px[2] == pytest.approx(379.61, rel=1e-3)


# ------------- The plan 10.3 mutation list, mapped ------------------ #
#
#  Plan section 10 item 3 lists ten mutants for the Stage D bug
#  injection pass.  Each line names the test which KILLS it and, where
#  the kill is partial or rests on something other than the test named
#  after the mutant, says so: a mutation map whose killer is wrong is
#  worse than none.  Every entry was VERIFIED 2026-09-09, either by
#  simulating the mutant with the low-level fit or by construction of
#  the fixture.
#
#  seed from the HIGHEST-residual
#    neighbour ............................ TestSeedChoice
#                                           ::test_the_lowest_residual
#                                           _converged_neighbour_wins,
#                                           a UNIT kill, plus
#                                           TestSeedChoiceIsTheSeam
#                                           TheCascadeUses for the
#                                           routing.  There is no
#                                           map-level killer and there
#                                           cannot be one on these
#                                           fixtures: MEASURED, every
#                                           seeding event on every map
#                                           here has exactly ONE
#                                           same-grain converged
#                                           candidate (ramp 3 from 2,
#                                           4 from 3, 5 from 4, 6 from
#                                           5; isolation 2 from 3;
#                                           two-ramp column 3 from
#                                           column 2 and column 4 from
#                                           column 3 in each row), so
#                                           which candidate won is
#                                           invisible in every
#                                           property a run returns
#  tie order shuffled ..................... TestSeedChoice
#                                           ::test_the_offset_order_is
#                                           _the_frozen_one, which
#                                           pins NEIGHBOR_OFFSETS
#                                           itself, plus the three
#                                           discriminating TIE_CASES
#                                           and the 28-pair sweep.
#                                           PARTIAL BY CONSTRUCTION:
#                                           the frozen order equals
#                                           ascending flat index for
#                                           every map width >= 2, so
#                                           no tie case separates it
#                                           from a flat-index argmin.
#                                           The constant pin is what
#                                           carries this one
#  seed across grain_id ................... TestSeedChoice::test_no
#                                           _seed_crosses_a_grain
#                                           _boundary (unit) and
#                                           TestGrainIsolation::test
#                                           _the_hard_point_is_seeded
#                                           _from_its_own_grain (map).
#                                           SIMULATED: with the gate
#                                           removed the hard point
#                                           does not converge and sits
#                                           30.887 px out, 84.7 per
#                                           cent of its imposed
#                                           displacement, against
#                                           0.100 px converged in 19
#                                           iterations with it
#  use same-round results (a race) ........ TestSeedRoundEncoding
#                                           ::test_every_encoded_value
#                                           _is_the_constructed_one.
#                                           VERIFIED: each ramp point
#                                           converges from its
#                                           predecessor in 5
#                                           iterations, so an
#                                           implementation which saw
#                                           its own round results
#                                           would convert columns 3
#                                           to 6 all in round 1 and
#                                           the encoding would read
#                                           0, 0, 0, 1, 1, 1, 1.  Also
#                                           caught by TestSeedChoiceIs
#                                           TheSeamTheCascadeUses,
#                                           whose frontier counts must
#                                           grow monotonically
#  ignore PASS1_CAP ....................... TestPass1CapSemantics
#                                           ::test_the_rescue_pass
#                                           _catches_the_isolated_slow
#                                           _point plus the EXPECTED
#                                           array.  VERIFIED: the
#                                           rescue point converges
#                                           from its own phase
#                                           cross-correlation seed in
#                                           exactly 113 iterations, so
#                                           with no cap seed_round[17]
#                                           is 0 rather than -2
#  skip the rescue pass ................... the same test: with no
#                                           rescue pass the isolated
#                                           slow point stays
#                                           unconverged, so
#                                           seed_round[17] is -1 and
#                                           converged[17] is False
#  mislabel seed_round .................... TestSeedRoundEncoding, all
#                                           21 entries pinned
#                                           literally with all seven
#                                           distinct values present,
#                                           plus ::test_the_encoding
#                                           _agrees_with_convergence
#                                           _everywhere
#  seed from an unconverged neighbour ..... TestSeedChoice::test_an
#                                           _unconverged_neighbour_is
#                                           _never_a_seed, a UNIT kill
#                                           routed by TestSeedChoiceIs
#                                           TheSeamTheCascadeUses.  A
#                                           secondary map-level kill
#                                           through the EXPECTED array
#                                           is plausible but is NOT
#                                           claimed here, because it
#                                           was not verified
#  drop the mask check .................... TestMaskIsolation::test
#                                           _masked_points_are
#                                           _untouched_and_carry_minus
#                                           _one on the CANDIDATE
#                                           side.  On the SEED side
#                                           the gate is unobservable
#                                           at map level and this is
#                                           recorded rather than
#                                           papered over: a masked
#                                           point is never converged,
#                                           so the converged gate
#                                           already excludes it, and
#                                           TestSeedChoice::test_a
#                                           _masked_point_neither
#                                           _seeds_nor_is_seeded
#                                           reaches it only by
#                                           constructing converged
#                                           True at a masked index, a
#                                           state the engine cannot
#                                           produce
#  per-point h0 array off by one in
#    flat order ........................... TestSimultaneousSeeding
#                                           ::test_each_point_of_a
#                                           _round_gets_its_own_seed,
#                                           the class added for this
#                                           mutant on 2026-09-09.
#                                           SIMULATED on its fixture:
#                                           a swap leaves both points
#                                           of round 1 unconverged at
#                                           47.6 and 49.5 px, 365 and
#                                           379 per cent of what they
#                                           impose, against 0.0096 and
#                                           0.0078 px converged in 5
#                                           iterations.  Also reached
#                                           by TestDeterminism, whose
#                                           two chunk sizes put the
#                                           two members of a round in
#                                           one chunk and in separate
#                                           chunks
