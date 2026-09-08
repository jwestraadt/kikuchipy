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

"""The GND half of oracle V7 of
``specs/2026-09-07-hrebsd-dic/validation.md``, against the frozen
scalar GND decision of requirements D14.

**THE ORACLE IS THE MATH, never another code.**  Requirements D14.1
records that the sign and index conventions of the Nye tensor differ
across papers and codes, and freezes THIS one, to be validated against
a synthetic field whose dislocation content is known by construction.
So nothing below is compared with OpenXY or EMsoftOO output; every
expectation is derived here from

.. code-block::

    alpha_ij = eps_jkl d(beta_il)/d(x_k)

and the derivation is written out in the helpers and their comments.

**How the expectation is derived, in full.**  A surface map carries
in-plane derivatives only, so ``k`` runs over 1 and 2 and the ``k = 3``
slot is IDENTICALLY ABSENT.  Expanding the Levi-Civita symbol for each
``j`` and striking the two terms that would need a ``d/dx3``:

.. code-block::

    j = 1:  eps_123 = +1, eps_132 = -1
            alpha_i1 = +d(beta_i3)/dx2 - d(beta_i2)/dx3
                                         ^ struck (D14.2)
    j = 2:  eps_231 = +1, eps_213 = -1
            alpha_i2 = +d(beta_i1)/dx3 - d(beta_i3)/dx1
                        ^ struck (D14.2)
    j = 3:  eps_312 = +1, eps_321 = -1
            alpha_i3 = +d(beta_i2)/dx1 - d(beta_i1)/dx2
                        both in-plane: EXACT (D14.1)

which is exactly the three-line form requirements D14.2 states.  This
module carries the expansion TWICE, once written out by hand
(:func:`alpha_by_hand`) and once contracted from the Levi-Civita
symbol itself (:func:`alpha_from_the_definition`), and a library
measurement pins the two against each other, so a transposed curl
index -- the first mutant of plan section 4.3 -- cannot hide in a
shared helper.

**The SIGN, pinned physically and related to the literature.**  Two
transcriptions of one formula cannot see a GLOBAL sign, and every
shipped estimator sums MODULI, so the sign would otherwise be free.
``TestNyeConvention.test_a_constant_lattice_curvature_gives_the_frozen
_signed_alpha`` therefore builds validation V7's own named recipe --
the rotation field ``omega_3(x1) = kappa * x1`` -- and pins the SIGNED
answer, which is what discharges requirements D14.2's "global sign
pinned by V7".  MEASURED AND RECORDED HERE, 2026-09-08: the frozen
D14.1 convention is exactly MINUS the classical Nye/Pantleon tensor
``alpha_ij = kappa_ji - delta_ij kappa_kk``, on that field and on
random in-plane curvature fields alike.  Nothing shipped sees it,
since the estimators take moduli, but a reader comparing the signed
``nye_tensor`` output with a paper must know it, so the relation is
asserted rather than left as prose.

**The field is linear, so the oracle is a constant.**  Every analytic
case here is
``beta(x1, x2) = B0 + P * x1 + Q * x2`` with constant 3 by 3
coefficient matrices, whose Nye tensor is the CONSTANT

.. code-block::

    alpha_i1 = +Q[i, 3]
    alpha_i2 = -P[i, 3]
    alpha_i3 = +P[i, 2] - Q[i, 1]

everywhere on the map.  Central differences of a linear field are
exact, and so are the one-sided differences requirements D14.5 uses at
the map edges, so a conformant implementation reproduces the constant
to ROUNDING.  The frozen bound validation V7 states is ``< 1 %``
(:data:`ORACLE_REL_TOL`), and that is what the V7-named assertions
use; the separate machine-precision arms exist because the exactness
is a property of the frozen difference scheme and is worth pinning
rather than leaving as slack.

**Assumption tiers, asserted per tier.**  The three ``alpha_i3`` of
requirements D14.1 are exact and the six ``alpha_i1``/``alpha_i2`` of
D14.2 hold only under the d/dx3-neglect, so each of the nine has its
own single-component case, and the estimator consumption sets are
pinned BOTH ways: an entry inside a set moves that estimator, and an
entry outside it leaves that estimator at exactly zero.  Per V7 the
structural claim is NOT that an entry is absent from every estimator
-- ``"a9"`` legitimately consumes all nine of its own construction --
but that no ``d/dx3`` term is ever fabricated, which
:class:`TestNoFabricatedDerivative` asserts on a field where any
value-based stand-in for such a term would move the answer by orders
of magnitude.

**File-layout note.**  The pattern-level oracle at the bottom
duplicates the master-pattern projection helpers of
``test_hrebsd_deformed_master.py`` rather than importing them: pytest
imports test modules by path and they cannot import one another. The
duplication is the same deliberate one that module records.

Written failing before the implementation, at the Stage C
failing-tests gate: every test which calls
``kikuchipy.indexing._hrebsd._gnd`` fails with ``NotImplementedError``
against the skeleton, while the signature, constant and library
measurements pass from the start.
"""

import functools
import inspect
import re
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.indexing._hrebsd._engine import run_hrebsd_dic
from kikuchipy.indexing._hrebsd._gnd import (
    ANTISYMMETRY_REPLACEMENTS,
    ESTIMATOR_COMPONENTS,
    ESTIMATOR_PREFACTORS,
    REQUIRED_PROP_NAMES,
    SCAN_UNIT_TO_METERS,
    SUPPORTED_ESTIMATORS,
    UNLABELLED,
    enforce_beta_antisymmetry,
    gnd_density,
    hrebsd_gnd,
    in_plane_gradients,
    log10_density,
    nye_tensor,
    scan_step_meters,
)
from kikuchipy.indexing._hrebsd._kam import UNLABELLED as KAM_UNLABELLED
from kikuchipy.indexing._hrebsd._reference import UNLABELLED as REFERENCE_UNLABELLED
from kikuchipy.indexing._hrebsd._segmentation import UNINDEXED
from kikuchipy.indexing._hrebsd._stiffness import rotate_stiffness
from kikuchipy.indexing._hrebsd._tensors import (
    BETA_PROP_SIZE,
    DETECTOR_Y_FLIP,
    STAGE_B_PROP_NAMES,
    close_beta,
    hrebsd_strain_stress,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
)

# ------------------------- Frozen constants ------------------------- #

# The band of a machine-precision-class algebraic identity, FROZEN.
# Central and one-sided differences of a LINEAR field are both exact,
# so every analytic case here reaches it
ALGEBRA_TOL = 1e-12

# THE frozen bound of validation V7: the implemented alpha equals the
# analytic constant alpha to better than one per cent.  It is the
# contract; the arms above assert the exactness the frozen difference
# scheme actually delivers on a linear field
ORACLE_REL_TOL = 1e-2

# The map the analytic oracles live on: big enough that an interior
# point has a full central-difference stencil in both directions and
# that a grain boundary can sit away from the edges
SHAPE = (5, 6)

# Map steps in micrometres, deliberately ANISOTROPIC so that a
# swapped-axis or one-step-for-both implementation is visible.  100 nm
# is the step requirements D14.6 quotes the noise floor scale at
STEP_X1_UM = 0.1
STEP_X2_UM = 0.2
STEP_X1_M = STEP_X1_UM * 1e-6
STEP_X2_M = STEP_X2_UM * 1e-6

# The Nye tensor scale of the analytic cases, in m^-1: a distortion
# change of 1e-4, the HREBSD scale, over the 1e-7 m step above.  With
# the nickel Burgers vector below it puts the "a3" density of a single
# entry at 1.2e13 m^-2, the 4e12 to 8e12 class requirements D14.6
# quotes
ALPHA_SCALE = 1e3

# The CONSTANT part of the analytic beta fields, five thousand times
# the per-step change.  It contributes nothing to a curl and is here
# to be ignored: see TestNoFabricatedDerivative
OFFSET_SCALE = 0.5

# Burgers vector lengths in METRES: nickel a/2<110> at a = 3.524 A and
# silicon a/2<110> at a = 5.431 A.  TEST DATA only, and in metres
# because requirements D14.5 freezes the unit and refuses a default
BURGERS_NI = 2.49e-10
BURGERS_SI = 3.84e-10

# The oracle detector of the whole feature; every precision claim is
# tied to 480 px (requirements Scope)
SHAPE_480 = (480, 480)
PC_480 = (0.4210, 0.5794, 0.5049)

# The generic crystal orientation of the Stage B and C suites,
# deliberately away from every symmetry position
ORIENTATION_AXIS = (1.0, 2.0, 3.0)
ORIENTATION_ANGLE_DEG = 37.0

# Single-crystal elastic constants of nickel in GPa (Simmons and
# Wang 1971), TEST DATA only: requirements D9.4 freezes that no
# elastic constant database ships with kikuchipy
NICKEL_CUBIC = {"c11": 246.5, "c12": 147.3, "c44": 124.7}


# ------------------ MEASURED-THEN-PINNED (MTP) ---------------------- #
#
# Unfilled placeholders (requirements D19).  Each is replaced by a
# dated measured value at the Stage C implementation gate, with the
# recipe and the machine recorded in validation.md "Recorded results",
# and only then guards regressions.

# MTP [D14, V7]: the worst RELATIVE error of the recovered Nye tensor
# when the analytic curvature field is imposed through DEFORMED-MASTER
# PATTERNS and carried through the whole chain -- projection, IC-GN,
# the D6 conversion, the D7 frame, the D9 closure and the D14
# gradients -- rather than handed to the module as algebra.
#
# It is bounded from below by the Stage A homography accuracy: the
# gradient reads a DIFFERENCE of two neighbouring Fe tensors, so the
# pinned DEFORMED_MASTER_FE_TOL of 3.1e-5 divided by the per-step
# distortion change this oracle imposes (5e-3) puts the expectation in
# the sub-per-cent class, and the analytic half of V7 is exact by
# comparison.  A value far above that means the gradient, the frame or
# the estimator is wrong rather than the DIC.
# MEASURING RECIPE: ``TestEndToEndCurvature::test_gnd_end_to_end
# _curvature`` below; record the worst relative error over every
# finite map point and over the three estimators
GND_E2E_TOL = None


# ------------- The plan 4.3 mutation list, mapped ------------------- #
#
#  Plan section 4.3 is the STAGE C list (plan.md, "## 4. Stage C",
#  item 3); section 3.3 is Stage B's and is not what this module maps.
#  Each line names the test which KILLS the mutant, which is not
#  always the test named after it -- where the two differ the
#  difference is spelled out, because a mutation map whose killer is
#  wrong is worse than none.
#
#  curl index convention transposed ....... TestNyeConvention
#                                           ::test_a_single_entry_field
#                                           _gives_that_entry_alone,
#                                           on the six off-diagonal
#                                           (i, j).  The three
#                                           diagonal cases are
#                                           transpose-invariant and
#                                           ride on those six.  The
#                                           two module oracles are
#                                           kept apart so neither can
#                                           move alone
#  alpha's global sign flipped in the
#    IMPLEMENTATION alone ................. the single-entry cases,
#                                           which compare against a
#                                           SIGNED expectation
#  the D14.1 convention itself replaced
#    by the classical Nye sign, oracle
#    and implementation together .......... TestNyeConvention
#                                           ::test_the_frozen
#                                           _convention_is_minus_the
#                                           _classical_nye_tensor,
#                                           which is the only place
#                                           an INDEPENDENT formula
#                                           (kappa_ji - delta_ij
#                                           kappa_kk) meets the
#                                           eps_jkl contraction, plus
#                                           ::test_a_constant_lattice
#                                           _curvature_gives_the
#                                           _frozen_signed_alpha for
#                                           V7's own named recipe.
#                                           Every estimator sums
#                                           moduli, so nothing
#                                           downstream can see either
#  prefactor swapped between estimators ... TestPrefactors
#                                           ::test_the_three
#                                           _prefactors_are_the
#                                           _literal_ratios and
#                                           ::test_the_density_is_the
#                                           _literal_formula.  NOT the
#                                           comparison-band test
#                                           beside them, which the
#                                           swap survives: 30/10 <->
#                                           30/14 still separates the
#                                           three estimators by more
#                                           than the band it checks
#  antisymmetry replacing beta13/23 by
#    -beta31/32 (backwards) ............... TestAntisymmetry
#                                           ::test_the_replacement_is
#                                           _beta31_from_minus_beta13,
#                                           through its
#                                           every-other-entry-
#                                           untouched loop
#  the antisymmetry fix applied AFTER the
#    ninth-degree closure ................. TestAntisymmetry
#                                           ::test_the_fix_is_applied
#                                           _before_the_closure.  It
#                                           needs a STIFFNESS: the
#                                           deviatoric closure adds a
#                                           multiple of the identity
#                                           and commutes with a fix
#                                           that touches
#                                           off-diagonals only, while
#                                           the traction-free solve
#                                           reads beta31 and beta32
#                                           and does not
#  gradients in pixels not metres ......... TestScanUnit
#                                           ::test_the_gradient_is_per
#                                           _metre_and_not_per_pixel,
#                                           by its exact per-column
#                                           step arms
#  b in nm not m .......................... TestBurgersVector.  It is
#                                           a CALLER input, so the
#                                           arithmetic arm only
#                                           quantifies it; the guards
#                                           are the absolute-scale
#                                           order check here and the
#                                           Si band in
#                                           test_hrebsd_si.py
#  log10 of signed values unguarded ....... TestLog10Guard
#
#  and the V7 structural claims:
#  a fabricated d/dx3 term ................ TestNoFabricatedDerivative
#                                           ::test_a_uniform
#                                           _distortion_has_no
#                                           _dislocation_content and
#                                           ::test_a_constant_offset
#                                           _leaves_alpha_unchanged
#                                           _to_rounding
#  an estimator consuming the wrong set ... TestConsumptionSets
#  the antisymmetry fix in the wrong frame  TestAntisymmetry
#  the antisymmetry fix off the GND path .. TestAntisymmetry
#  a point whose own beta is non-finite
#    reported as a finite density ......... TestNaNSafety, at the
#                                           nye_tensor level and
#                                           through hrebsd_gnd
#                                           (requirements D2.6)
#  a step read off a ragged or descending
#    coordinate array ..................... TestScanUnit
#                                           ::test_an_irregular_grid
#                                           _still_gives_a_positive
#                                           _step_of_the_right_class


# --------------------- Oracle: the definition ----------------------- #


def levi_civita():
    """Return the Levi-Civita symbol ``eps[i, j, k]``, built here from
    its definition so that no index convention is imported."""
    eps = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, k] = 1.0
        eps[i, k, j] = -1.0
    return eps


def alpha_from_the_definition(p, q):
    """Return the constant Nye tensor of the linear beta field
    ``beta(x1, x2) = B0 + P x1 + Q x2``, contracted straight from
    ``alpha_ij = eps_jkl d(beta_il)/d(x_k)`` (requirements D14.1).

    The whole assumption tier of requirements D14.2 sits in ONE line
    below: the ``k = 3`` plane of the derivative array is filled with
    zeros and is never given a value.  That is what "no ``d/dx3``
    derivative term is ever fabricated" means structurally, and it is
    why this helper and not a hand expansion is the primary oracle.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # derivative[k, i, l] = d(beta_il)/d(x_k)
    derivative = np.zeros((3, 3, 3))
    derivative[0] = p
    derivative[1] = q
    # derivative[2] stays ZERO: a surface map has no x3 neighbour, so
    # the slot is absent, not estimated (requirements D14.2)
    return np.einsum("jkl,kil->ij", levi_civita(), derivative)


def alpha_by_hand(p, q):
    """Return the same constant Nye tensor from the three-line
    expansion of the module docstring, written out by hand.

    Kept beside :func:`alpha_from_the_definition` on purpose: if a
    transposed curl index is ever introduced into one of them, the
    library measurement of :class:`TestNyeConvention` fails and names
    the disagreement instead of both oracles moving together.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    alpha = np.zeros((3, 3))
    # alpha_i1 = +d(beta_i3)/dx2, the d/dx3 half struck (D14.2)
    alpha[:, 0] = q[:, 2]
    # alpha_i2 = -d(beta_i3)/dx1, the d/dx3 half struck (D14.2)
    alpha[:, 1] = -p[:, 2]
    # alpha_i3 = d(beta_i2)/dx1 - d(beta_i1)/dx2, EXACT (D14.1)
    alpha[:, 2] = p[:, 1] - q[:, 0]
    return alpha


def rotation_field_coefficients(curvature):
    """Return ``(P, Q)`` of the beta field of a constant in-plane
    LATTICE CURVATURE, validation V7's own named recipe.

    *curvature* is ``kappa_ij = d(omega_i)/d(x_j)`` with the third
    column zero, since a surface map has no ``x3`` neighbour.  A small
    rotation field ``omega(x)`` displaces by ``u = omega x x``, so

    .. code-block::

        beta_il = d(u_i)/d(x_l) = eps_ijl omega_j

    and the coefficient matrices follow by differentiating that along
    each in-plane axis.  Built from the Levi-Civita symbol itself, so
    the physics enters here and no index convention is imported.
    """
    curvature = np.asarray(curvature, dtype=np.float64)
    eps = levi_civita()
    p = np.einsum("ijl,j->il", eps, curvature[:, 0])
    q = np.einsum("ijl,j->il", eps, curvature[:, 1])
    return p, q


def classical_nye_tensor(curvature):
    """Return the CLASSICAL Nye/Pantleon tensor of a lattice curvature,
    ``alpha_ij = kappa_ji - delta_ij kappa_kk``.

    Here to be CONTRASTED with the frozen D14.1 convention, never to
    define it: requirements D14.1 freezes its own convention and
    validates it against the derivation, never against another code or
    paper.  The two differ by a global sign, which
    :meth:`TestNyeConvention.test_a_constant_lattice_curvature_gives
    _the_frozen_signed_alpha` measures and records.
    """
    curvature = np.asarray(curvature, dtype=np.float64)
    return curvature.T - np.trace(curvature) * np.eye(3)


def density_by_hand(alpha, burgers_vector_length, estimator):
    """Return the scalar density of requirements D14.4, assembled from
    the literal prefactor and the literal consumption set rather than
    from the module's own tables."""
    literal_prefactor = {"a3": 30 / 10, "a5": 30 / 14, "a9": 30 / 20}[estimator]
    literal_components = {
        "a3": [(0, 2), (1, 2), (2, 2)],
        "a5": [(0, 2), (1, 2), (2, 2), (0, 1), (1, 0)],
        "a9": [(i, j) for i in range(3) for j in range(3)],
    }[estimator]
    alpha = np.asarray(alpha, dtype=np.float64)
    total = sum(np.abs(alpha[..., i, j]) for (i, j) in literal_components)
    return literal_prefactor * total / burgers_vector_length


# ------------------------ Field construction ------------------------ #


def single_entry_coefficients(i, j, value=ALPHA_SCALE):
    """Return ``(P, Q)`` of a linear beta field whose Nye tensor has
    EXACTLY ONE nonzero entry, ``alpha[i, j] = value``.

    Inverting the three lines of the expansion, one tier at a time:

    - ``alpha_i1 = +d(beta_i3)/dx2`` so ``Q[i, 3] = value`` alone,
      which leaves ``alpha_i2 = -P[i, 3] = 0``;
    - ``alpha_i2 = -d(beta_i3)/dx1`` so ``P[i, 3] = -value`` alone,
      which leaves ``alpha_i1 = Q[i, 3] = 0``;
    - ``alpha_i3 = +d(beta_i2)/dx1 - d(beta_i1)/dx2`` so
      ``P[i, 2] = value`` alone, and ``beta_i2`` enters no other
      entry.

    Every case is a single off-diagonal or diagonal slot of one
    coefficient matrix, so the two tiers are exercised separately and
    an entry which leaks into a second slot of ``alpha`` is caught by
    the caller.
    """
    p = np.zeros((3, 3))
    q = np.zeros((3, 3))
    if j == 0:
        q[i, 2] = value
    elif j == 1:
        p[i, 2] = -value
    else:
        p[i, 1] = value
    return p, q


def linear_beta_field(
    p,
    q,
    b0=None,
    shape=SHAPE,
    step_x1=STEP_X1_M,
    step_x2=STEP_X2_M,
):
    """Return the map-shaped field ``beta(x1, x2) = B0 + P x1 + Q x2``
    with ``x1`` the COLUMN axis and ``x2`` the ROW axis, in METRES.

    The axis identification is requirements D12's, the one the
    constant-curvature HR-KAM identity already uses: ``x1`` grows with
    the column index and ``x2`` with the row index.
    """
    ny, nx = shape
    rows, cols = np.indices((ny, nx))
    field = np.zeros((ny, nx, 3, 3), dtype=np.float64)
    if b0 is not None:
        field += np.asarray(b0, dtype=np.float64)
    field = field + cols[..., None, None] * step_x1 * np.asarray(p, dtype=np.float64)
    field = field + rows[..., None, None] * step_x2 * np.asarray(q, dtype=np.float64)
    return field


# The detector-frame coefficient matrices of the through-the-function
# oracles.  Three constraints, each for a stated reason:
#
# 1. entry [3, 3] is zero, so the stored Fe = I + beta has Fe33 = 1
#    exactly, which is what requirements D15.6 says the REDUCED tensor
#    carries;
# 2. the trace is zero, so the deviatoric closure of requirements D9.3
#    is an exact no-op and what the chain reports IS the field built
#    here, with no closure arithmetic in between;
# 3. the [3, 1] and [3, 2] entries are already the negatives of [1, 3]
#    and [2, 3], so the antisymmetry fix of requirements D14.3 is an
#    exact no-op too and the toggle cannot silently change these
#    oracles.  TestAntisymmetry deliberately breaks this third one.
P_DETECTOR = ALPHA_SCALE * np.array(
    [
        [1.0, 0.7, -0.4],
        [-0.6, -1.0, 0.9],
        [0.4, -0.9, 0.0],
    ]
)
Q_DETECTOR = ALPHA_SCALE * np.array(
    [
        [-0.5, 1.1, 0.8],
        [0.3, 0.5, -0.2],
        [-0.8, 0.2, 0.0],
    ]
)
B0_DETECTOR = OFFSET_SCALE * np.array(
    [
        [0.2, -0.3, 0.5],
        [0.1, -0.2, -0.4],
        [-0.5, 0.4, 0.0],
    ]
)

# The coefficient matrices of the PURE-``nye_tensor`` cases, which
# never go through :func:`gnd_map` and are therefore NOT bound by the
# first constraint above.  Their ``[3, 3]`` entries are NONZERO, and
# that is the whole point (added 2026-09-08, Stage C adversarial
# review): with ``P[3, 3] = Q[3, 3] = 0`` the inversion
# ``alpha_31 = +Q[3, 3]`` and ``alpha_32 = -P[3, 3]`` makes those two
# entries STRUCTURALLY zero, so a generic field built on the detector
# pair moves seven of the nine and not nine.  Seven of nine is exactly
# the wrong number for a case whose stated job is the whole tensor at
# once, and the two it misses are two of the six d/dx3-neglect entries
# only "a9" consumes.  The 31/32 slots stay the negatives of the 13/23
# ones, so the antisymmetry constraint is kept; the traceless one is
# deliberately BROKEN by the new entry and does not matter here,
# because no closure is on this path at all
P_GENERIC = P_DETECTOR.copy()
Q_GENERIC = Q_DETECTOR.copy()
P_GENERIC[2, 2] = 0.6 * ALPHA_SCALE
Q_GENERIC[2, 2] = -0.5 * ALPHA_SCALE
B0_GENERIC = B0_DETECTOR.copy()
B0_GENERIC[2, 2] = 0.3 * OFFSET_SCALE


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
    """Return the matrix ``g`` with ``v_crystal = g @ v_sample``,
    derived from ``rotate_vector`` itself so that no orix convention
    is assumed anywhere in this module."""
    quaternion = np.asarray(rotation.data, dtype=np.float64).ravel()
    return rotate_vector(quaternion, np.ascontiguousarray(np.eye(3))).T


def sample_to_spec_detector_matrix(detector):
    """Return the matrix ``M`` with ``v_detector = M @ v_sample`` in
    the SPEC's y-down detector frame (requirements D7, corrected)."""
    return DETECTOR_Y_FLIP @ detector.sample_to_detector.to_matrix().squeeze()


def to_sample_frame(tensor, matrix):
    """Return ``M^T A M``, the frame rotation of requirements D7,
    written out here."""
    return matrix.T @ np.asarray(tensor, dtype=np.float64) @ matrix


def deviatoric(tensor):
    """Return the traceless part of a tensor, the closure of
    requirements D9.3 written out here."""
    tensor = np.asarray(tensor, dtype=np.float64)
    return tensor - np.trace(tensor) / 3.0 * np.eye(3)


def antisymmetry_by_hand(tensor):
    """Return a tensor with ``beta31 <- -beta13`` and
    ``beta32 <- -beta23``, the fix of requirements D14.3 written out
    here rather than taken from the module under test."""
    fixed = np.array(tensor, dtype=np.float64, copy=True)
    fixed[2, 0] = -fixed[0, 2]
    fixed[2, 1] = -fixed[1, 2]
    return fixed


def expected_alpha_through_the_chain(p, q, detector, fix="detector"):
    """Return the constant sample-frame Nye tensor a detector-frame
    linear field produces through the whole D14 route, assembled here.

    The chain is linear in the field, so it acts on the coefficient
    matrices one at a time and no finite difference enters this
    oracle at all: the antisymmetry fix (a slot replacement), the D7
    frame rotation (a congruence) and the D9.3 deviatoric closure (a
    trace removal) are each applied to ``P`` and ``Q``, and the
    analytic curl of the result is the answer.

    *fix* selects the ORDER, which is the whole point of requirements
    D14.3: ``"detector"`` is the frozen one, ``"sample"`` is the
    mutant that fixes after rotating, and ``"none"`` is the toggle
    off.
    """
    matrix = sample_to_spec_detector_matrix(detector)
    transformed = []
    for coefficients in (p, q):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if fix == "detector":
            coefficients = antisymmetry_by_hand(coefficients)
        coefficients = to_sample_frame(coefficients, matrix)
        if fix == "sample":
            coefficients = antisymmetry_by_hand(coefficients)
        transformed.append(deviatoric(coefficients))
    return alpha_from_the_definition(*transformed)


def gnd_map(
    beta_detector,
    *,
    grain_id=None,
    scan_unit="um",
    step_sizes=(STEP_X2_UM, STEP_X1_UM),
    phase_id=None,
    phase_list=None,
):
    """Return a crystal map carrying the two properties the GND path
    consumes: the reduced detector-frame ``Fe`` of the engine and the
    ``grain_id`` of its reference resolution.

    *beta_detector* is the DETECTOR-frame displacement gradient field
    of shape ``(ny, nx, 3, 3)``; the stored property is ``I + beta``,
    which is the reduced ``Fe`` of requirements D15.6 exactly when
    ``beta33`` is zero, as every field built here has.

    *step_sizes* is orix's ``(row, column)`` order, so the defaults
    are the anisotropic ``(x2, x1)`` steps of this module.
    """
    beta_detector = np.asarray(beta_detector, dtype=np.float64)
    ny, nx = beta_detector.shape[:2]
    arrays, size = create_coordinate_arrays((ny, nx), step_sizes)
    rotation = generic_rotation()
    arrays["rotations"] = Rotation(np.tile(rotation.data.ravel(), (size, 1)))
    arrays["phase_id"] = (
        np.zeros(size, dtype=int) if phase_id is None else np.asarray(phase_id).ravel()
    )
    arrays["phase_list"] = (
        PhaseList(Phase(name="ni", space_group=225))
        if phase_list is None
        else phase_list
    )
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = scan_unit
    xmap.prop["Fe"] = (np.eye(3) + beta_detector).reshape(size, BETA_PROP_SIZE)
    if grain_id is None:
        grain_id = np.zeros((ny, nx), dtype=np.int32)
    xmap.prop["grain_id"] = np.asarray(grain_id, dtype=np.int32).ravel()
    return xmap


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


# ============ V7 -- the Nye convention and its two tiers ============ #


class TestNyeConvention:
    """``alpha_ij = eps_jkl d(beta_il)/dx_k``, validated against the
    DERIVATION.  [D14.1/D14.2/V7]"""

    def test_the_two_oracles_of_this_module_agree(self):
        # A LIBRARY MEASUREMENT of this module's own arithmetic, so it
        # passes today.  One oracle contracts the Levi-Civita symbol
        # and the other writes the three lines out; they are kept
        # apart so that a transposed curl index cannot move both at
        # once, and this is where the disagreement would be named
        rng = np.random.default_rng(0)
        for _ in range(5):
            p = rng.normal(size=(3, 3))
            q = rng.normal(size=(3, 3))
            np.testing.assert_allclose(
                alpha_from_the_definition(p, q),
                alpha_by_hand(p, q),
                rtol=0,
                atol=ALGEBRA_TOL,
            )

    def test_the_definition_oracle_never_reads_an_x3_derivative(self):
        # A LIBRARY MEASUREMENT of the structural claim: the oracle
        # takes only two coefficient matrices, so there is no third
        # argument a d/dx3 term could come from, and the transpose of
        # the curl is visible -- alpha is NOT symmetric in general
        p = np.arange(9.0).reshape(3, 3)
        q = np.arange(9.0, 18.0).reshape(3, 3)
        alpha = alpha_from_the_definition(p, q)
        assert not np.allclose(alpha, alpha.T)
        # and the transposed-index mutant really is a different tensor
        assert not np.allclose(alpha, alpha_from_the_definition(q, p))

    @pytest.mark.parametrize("i", [0, 1, 2])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_a_single_entry_field_gives_that_entry_alone(self, i, j):
        # The per-tier single-component cases validation V7 asks for:
        # the three EXACT alpha_i3 of requirements D14.1 at j = 2, and
        # the six d/dx3-neglect alpha_i1/alpha_i2 of D14.2 at j = 0
        # and j = 1.  Each field is built to move ONE entry, and the
        # oracle confirms it before the implementation is asked
        p, q = single_entry_coefficients(i, j)
        expected = alpha_from_the_definition(p, q)
        assert expected[i, j] == pytest.approx(ALPHA_SCALE, rel=1e-15)
        assert np.count_nonzero(expected) == 1
        got = nye_tensor(linear_beta_field(p, q), STEP_X1_M, STEP_X2_M)
        assert got.shape == SHAPE + (3, 3)
        assert got.dtype == np.float64
        # THE frozen V7 bound, on the ONE entry that moves.  Split from
        # the eight that do not (2026-09-08, Stage C adversarial
        # review): a flat ``atol`` of ORACLE_REL_TOL * ALPHA_SCALE is
        # 10 m^-1, and applying that to entries whose expectation is
        # 0.0 would tolerate ten orders of leakage into them and gut
        # the single-component claim.  These fields have exactly-zero
        # coefficient columns, so a conformant implementation
        # differences them to EXACTLY 0.0
        np.testing.assert_allclose(
            got[..., i, j],
            np.full(SHAPE, expected[i, j]),
            rtol=ORACLE_REL_TOL,
            atol=0,
        )
        elsewhere = np.ones((3, 3), dtype=bool)
        elsewhere[i, j] = False
        np.testing.assert_allclose(
            got[..., elsewhere],
            np.zeros(SHAPE + (8,)),
            rtol=0,
            atol=ALGEBRA_TOL,
        )

    def test_the_detector_pair_leaves_alpha_31_and_alpha_32_at_zero(self):
        # A LIBRARY MEASUREMENT, and the reason P_GENERIC exists.  The
        # first constraint on the detector pair pins P[3, 3] and
        # Q[3, 3] to zero so that the stored Fe33 is 1, and the
        # inversion alpha_31 = +Q[3, 3], alpha_32 = -P[3, 3] then makes
        # those two entries structurally zero -- SEVEN of the nine
        # move, not nine.  Recorded here so that the generic case below
        # cannot silently drift back onto the detector pair
        detector_alpha = alpha_from_the_definition(P_DETECTOR, Q_DETECTOR)
        assert P_DETECTOR[2, 2] == 0.0 and Q_DETECTOR[2, 2] == 0.0
        assert detector_alpha[2, 0] == 0.0
        assert detector_alpha[2, 1] == 0.0
        assert np.count_nonzero(detector_alpha) == 7
        # and the generic pair, which no gnd_map ever sees, moves all
        # nine for exactly the same reason read backwards
        assert P_GENERIC[2, 2] != 0.0 and Q_GENERIC[2, 2] != 0.0
        assert np.count_nonzero(alpha_from_the_definition(P_GENERIC, Q_GENERIC)) == 9

    def test_a_generic_field_reproduces_the_constant_exactly(self):
        # the whole nine entries at once, and at machine precision:
        # requirements D14.5 freezes central differences with one-sided
        # differences at the edges, and BOTH are exact on a linear
        # field, so there is no discretization error to leave slack for
        got = nye_tensor(
            linear_beta_field(P_GENERIC, Q_GENERIC, B0_GENERIC),
            STEP_X1_M,
            STEP_X2_M,
        )
        expected = alpha_from_the_definition(P_GENERIC, Q_GENERIC)
        assert np.count_nonzero(expected) == 9
        np.testing.assert_allclose(
            got,
            np.broadcast_to(expected, got.shape),
            rtol=1e-10,
            atol=ALGEBRA_TOL * ALPHA_SCALE,
        )

    def test_a_constant_lattice_curvature_gives_the_frozen_signed_alpha(self):
        # VALIDATION V7's own named recipe, built at last: a constant
        # lattice curvature omega_3(x1) = kappa * x1, that is a pure
        # rotation field with no elastic strain in it at all, so the
        # d/dx3-neglect of requirements D14.2 is exact rather than
        # assumed.  This is the ONLY signed pin in the module -- the
        # two oracles above are two transcriptions of one formula and
        # cannot see a global sign, and every estimator sums moduli --
        # so it is what discharges requirements D14.2's "global sign
        # pinned by V7"
        kappa = ALPHA_SCALE
        curvature = np.zeros((3, 3))
        curvature[2, 0] = kappa
        p, q = rotation_field_coefficients(curvature)
        # the field really is the rotation one: beta is antisymmetric,
        # so its symmetric part, the elastic strain, is exactly zero
        np.testing.assert_array_equal(p, -p.T)
        np.testing.assert_array_equal(q, np.zeros((3, 3)))
        expected = alpha_from_the_definition(p, q)
        # a tilt wall of edge dislocations with b along x1 and line
        # along x3 puts its whole content in alpha_13, and the FROZEN
        # convention gives it the MINUS sign
        assert expected[0, 2] == -kappa
        assert np.count_nonzero(expected) == 1
        got = nye_tensor(linear_beta_field(p, q), STEP_X1_M, STEP_X2_M)
        np.testing.assert_allclose(
            got[..., 0, 2], np.full(SHAPE, -kappa), rtol=ORACLE_REL_TOL, atol=0
        )
        # and the density is blind to the sign, which is why the pin
        # has to live at this level
        density = gnd_density(got, BURGERS_NI, "a3")
        np.testing.assert_allclose(
            density,
            np.full(SHAPE, (30 / 10) * kappa / BURGERS_NI),
            rtol=ORACLE_REL_TOL,
            atol=0,
        )

    def test_the_frozen_convention_is_minus_the_classical_nye_tensor(self):
        # A LIBRARY MEASUREMENT and a RECORDED FACT, 2026-09-08.  The
        # classical Nye/Pantleon relation alpha_ij = kappa_ji -
        # delta_ij kappa_kk is not what requirements D14.1 freezes, and
        # the difference is a global sign.  It changes nothing shipped,
        # since gnd_density sums moduli, but a reader comparing the
        # SIGNED nye_tensor output with a paper needs it in writing --
        # and if a future implementation ever flips to the classical
        # sign, this is where the change is named rather than passing
        # silently through every modulus
        rng = np.random.default_rng(19)
        for _ in range(5):
            curvature = np.zeros((3, 3))
            curvature[:, :2] = ALPHA_SCALE * rng.normal(size=(3, 2))
            frozen = alpha_from_the_definition(*rotation_field_coefficients(curvature))
            classical = classical_nye_tensor(curvature)
            np.testing.assert_allclose(
                frozen, -classical, rtol=0, atol=ALGEBRA_TOL * ALPHA_SCALE
            )
            assert not np.allclose(frozen, classical)

    def test_the_map_edges_carry_the_same_constant(self):
        # the one-sided arm of requirements D14.5, stated separately:
        # a constant-curvature field has NO edge artefact, so an
        # implementation which leaves the border NaN, or which wraps
        # around, is caught here rather than by an aggregate
        got = nye_tensor(linear_beta_field(P_GENERIC, Q_GENERIC), STEP_X1_M, STEP_X2_M)
        assert np.all(np.isfinite(got))
        expected = alpha_from_the_definition(P_GENERIC, Q_GENERIC)
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            np.testing.assert_allclose(
                got[corner], expected, rtol=1e-10, atol=ALGEBRA_TOL * ALPHA_SCALE
            )


class TestInPlaneGradients:
    """Two in-plane derivatives and no third one.  [D14.2/D14.5]"""

    def test_the_two_derivatives_are_the_coefficient_matrices(self):
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR)
        d_dx1, d_dx2 = in_plane_gradients(field, STEP_X1_M, STEP_X2_M)
        for got, expected in ((d_dx1, P_DETECTOR), (d_dx2, Q_DETECTOR)):
            assert got.shape == field.shape
            np.testing.assert_allclose(
                got,
                np.broadcast_to(expected, got.shape),
                rtol=1e-10,
                atol=ALGEBRA_TOL * ALPHA_SCALE,
            )

    def test_only_two_derivatives_are_returned(self):
        # the structural half of requirements D14.2: there is no third
        # return value, so no caller can be handed a d/dx3 array
        got = in_plane_gradients(
            linear_beta_field(P_DETECTOR, Q_DETECTOR), STEP_X1_M, STEP_X2_M
        )
        assert isinstance(got, tuple)
        assert len(got) == 2

    def test_the_two_steps_are_not_interchangeable(self):
        # the anisotropic map of this module: swapping the two steps
        # scales each derivative by the ratio, which is 2 here, so a
        # one-step-for-both or a swapped-axis implementation cannot
        # pass both this and the test above
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR)
        d_dx1, d_dx2 = in_plane_gradients(field, STEP_X1_M, STEP_X2_M)
        swapped_1, swapped_2 = in_plane_gradients(field, STEP_X2_M, STEP_X1_M)
        assert STEP_X2_M / STEP_X1_M == pytest.approx(2.0)
        np.testing.assert_allclose(
            swapped_1, d_dx1 * (STEP_X1_M / STEP_X2_M), rtol=1e-10, atol=0
        )
        np.testing.assert_allclose(
            swapped_2, d_dx2 * (STEP_X2_M / STEP_X1_M), rtol=1e-10, atol=0
        )


class TestNoFabricatedDerivative:
    """No ``d/dx3`` term is ever fabricated.  [D14.2/V7]

    The structural assertion validation V7 names.  A surface map has
    no third neighbour, so the only way a ``d/dx3`` term could appear
    is as a stand-in built from the beta VALUES themselves.  Every
    test here feeds a field where such a stand-in would move the
    answer by orders of magnitude and asserts that nothing moves.
    """

    def test_a_uniform_distortion_has_no_dislocation_content(self):
        # THE sharpest form: a crystal under a large, perfectly
        # UNIFORM elastic distortion contains no geometrically
        # necessary dislocations at all.  The field is 0.5 in every
        # entry-scale, five thousand times the per-step change of the
        # cases above, so a value-based d/dx3 stand-in would report a
        # density near 1e16 m^-2 instead of exactly zero
        field = linear_beta_field(np.zeros((3, 3)), np.zeros((3, 3)), B0_DETECTOR)
        alpha = nye_tensor(field, STEP_X1_M, STEP_X2_M)
        assert np.all(np.isfinite(alpha))
        np.testing.assert_array_equal(alpha, np.zeros_like(alpha))
        for estimator in SUPPORTED_ESTIMATORS:
            density = gnd_density(alpha, BURGERS_NI, estimator)
            np.testing.assert_array_equal(density, np.zeros_like(density))

    def test_a_constant_offset_leaves_alpha_unchanged_to_rounding(self):
        # the same claim in its general form: alpha depends on the
        # field ONLY through its two in-plane derivatives, so two
        # fields with the same P and Q and different constants give
        # the same answer.
        #
        # NOT bitwise, and an earlier draft of this test asserted
        # exactly that (corrected 2026-09-08, Stage C adversarial
        # review).  ``(c + a) - (c + b)`` need not round to the same
        # double as ``a - b``, so NO conformant implementation can
        # satisfy a bitwise claim here -- numpy.gradient on these two
        # fields is not bitwise equal either, which is a property of
        # IEEE-754 and not of any curl.  MEASURED at this gate with a
        # reference implementation of the frozen chain: 1.8e-10 m^-1
        # absolute on the B0 arm and 3.9e-10 RELATIVE on the arm whose
        # offset is a thousand times larger, against entries of order
        # 1e3 m^-1.  The bound below is 1e-8 relative, still seven
        # orders inside the frozen V7 per-cent contract, while a
        # value-based d/dx3 stand-in would move the answer by ORDERS
        # OF MAGNITUDE -- the offsets here are five thousand and five
        # million times the per-step change
        without = nye_tensor(
            linear_beta_field(P_DETECTOR, Q_DETECTOR), STEP_X1_M, STEP_X2_M
        )
        with_offset = nye_tensor(
            linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR),
            STEP_X1_M,
            STEP_X2_M,
        )
        huge = nye_tensor(
            linear_beta_field(P_DETECTOR, Q_DETECTOR, 1e3 * B0_DETECTOR),
            STEP_X1_M,
            STEP_X2_M,
        )
        for got in (with_offset, huge):
            np.testing.assert_allclose(
                got, without, rtol=1e-8, atol=ALGEBRA_TOL * ALPHA_SCALE
            )

    def test_the_beta_i3_column_moves_only_its_own_two_entries(self):
        # the tier boundary, stated as a claim about which slots are
        # read: the beta_i3 column feeds alpha_i1 and alpha_i2 (the
        # d/dx3-neglect tier) and NOTHING else, while the beta_i1 and
        # beta_i2 columns feed alpha_i3 (the exact tier) and nothing
        # else.  A curl which fabricated a third derivative would mix
        # them.
        #
        # B0_DETECTOR is passed for a reason (added 2026-09-08, Stage C
        # adversarial review): the two struck terms are
        # -d(beta_i2)/dx3 in alpha_i1 and +d(beta_i1)/dx3 in alpha_i2,
        # so a value-based stand-in for either is proportional to the
        # beta_i2 or the beta_i1 column.  With P alone in its third
        # column and no offset those two columns are IDENTICALLY ZERO
        # everywhere, and a fabricating implementation would pass this
        # test with nothing to fabricate from.  The offset gives it
        # something, at 0.5 per entry against a per-step change of
        # 1e-4
        p = np.zeros((3, 3))
        p[:, 2] = ALPHA_SCALE
        field = linear_beta_field(p, np.zeros((3, 3)), B0_DETECTOR)
        assert np.abs(field[..., :, :2]).min() > 0.0
        alpha = nye_tensor(field, STEP_X1_M, STEP_X2_M)
        np.testing.assert_allclose(
            alpha[..., 0], np.zeros(SHAPE + (3,)), rtol=0, atol=ALGEBRA_TOL
        )
        np.testing.assert_allclose(
            alpha[..., 2], np.zeros(SHAPE + (3,)), rtol=0, atol=ALGEBRA_TOL
        )
        assert np.abs(alpha[..., 1]).min() > 0.5 * ALPHA_SCALE


# ============== D14.4 -- estimators, sets and factors =============== #


class TestPrefactors:
    """30/10, 30/14 and 30/20, literally.  [D14.4/V7]"""

    def test_the_three_prefactors_are_the_literal_ratios(self):
        assert ESTIMATOR_PREFACTORS["a3"] == 30 / 10
        assert ESTIMATOR_PREFACTORS["a5"] == 30 / 14
        assert ESTIMATOR_PREFACTORS["a9"] == 30 / 20
        assert set(ESTIMATOR_PREFACTORS) == set(SUPPORTED_ESTIMATORS)
        assert SUPPORTED_ESTIMATORS == ("a3", "a5", "a9")

    def test_the_three_prefactors_are_pairwise_distinct(self):
        # without this a swap between two of them would be invisible
        # to every test that follows
        values = [ESTIMATOR_PREFACTORS[name] for name in SUPPORTED_ESTIMATORS]
        assert len(set(values)) == 3

    @pytest.mark.parametrize("estimator", ["a3", "a5", "a9"])
    def test_the_density_is_the_literal_formula(self, estimator):
        # a hand-built alpha, and an expectation assembled from the
        # literal ratio and the literal index set of requirements
        # D14.4 rather than from the module's own tables
        rng = np.random.default_rng(3)
        alpha = ALPHA_SCALE * rng.normal(size=(4, 3, 3))
        got = gnd_density(alpha, BURGERS_NI, estimator)
        expected = density_by_hand(alpha, BURGERS_NI, estimator)
        assert got.shape == (4,)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)

    def test_the_three_prefactors_are_far_apart_on_one_alpha(self):
        # WHAT THIS MEASURES, stated honestly (renamed 2026-09-08,
        # Stage C adversarial review; it was called
        # ``test_a_swapped_prefactor_would_be_visible``): the three
        # prefactors are far enough apart that a swap cannot hide
        # inside the comparison band.  It does NOT kill the plan 4.3
        # swapped-prefactor mutant, and the old name claimed it did --
        # under 30/10 <-> 30/14 the ratio below is 1.4, still past the
        # 0.25 gate, and the two per-term densities still separate.
        # The killer is ``test_the_three_prefactors_are_the_literal
        # _ratios`` above, which pins each constant literally
        alpha = np.full((3, 3), ALPHA_SCALE)
        densities = {
            name: gnd_density(alpha, BURGERS_NI, name) for name in SUPPORTED_ESTIMATORS
        }
        swapped = ESTIMATOR_PREFACTORS["a5"] / ESTIMATOR_PREFACTORS["a3"]
        assert abs(swapped - 1.0) > 0.25
        assert not np.isclose(densities["a3"] / 3, densities["a5"] / 5)


class TestConsumptionSets:
    """Each estimator consumes EXACTLY its documented set.  [D14.4/V7]

    Pinned in both directions on the nine single-entry fields: an
    entry inside a set moves that estimator by the full prefactor, and
    an entry outside it leaves that estimator at EXACTLY zero.  Per
    validation V7 the claim is not that any entry is absent from every
    estimator -- ``"a9"`` legitimately takes all nine of its own
    d/dx3-neglect construction -- but that no set is wider or narrower
    than requirements D14.4 documents.
    """

    def test_the_sets_are_the_documented_ones(self):
        assert set(ESTIMATOR_COMPONENTS["a3"]) == {(0, 2), (1, 2), (2, 2)}
        assert set(ESTIMATOR_COMPONENTS["a5"]) == {
            (0, 2),
            (1, 2),
            (2, 2),
            (0, 1),
            (1, 0),
        }
        assert set(ESTIMATOR_COMPONENTS["a9"]) == {
            (i, j) for i in range(3) for j in range(3)
        }
        # the names say how many terms are summed, and the prefactor
        # denominators 10, 14 and 20 are twice the term counts, which
        # is the OpenXY L1 extrapolation these constants come from
        for name, count in (("a3", 3), ("a5", 5), ("a9", 9)):
            assert len(ESTIMATOR_COMPONENTS[name]) == count
            assert len(set(ESTIMATOR_COMPONENTS[name])) == count

    def test_a3_takes_only_the_three_exact_entries(self):
        # requirements D14.1: alpha_i3 are the only entries a surface
        # map determines without an assumption, and "a3" is the
        # assumption-free estimator
        assert set(ESTIMATOR_COMPONENTS["a3"]) == {(i, 2) for i in range(3)}

    def test_a5_adds_exactly_alpha_12_and_alpha_21(self):
        extra = set(ESTIMATOR_COMPONENTS["a5"]) - set(ESTIMATOR_COMPONENTS["a3"])
        assert extra == {(0, 1), (1, 0)}

    def test_a9_adds_exactly_the_remaining_six(self):
        extra = set(ESTIMATOR_COMPONENTS["a9"]) - set(ESTIMATOR_COMPONENTS["a3"])
        assert len(extra) == 6
        assert all(j != 2 for (_, j) in extra)

    @pytest.mark.parametrize("i", [0, 1, 2])
    @pytest.mark.parametrize("j", [0, 1, 2])
    @pytest.mark.parametrize("estimator", ["a3", "a5", "a9"])
    def test_one_entry_moves_exactly_the_estimators_that_take_it(self, i, j, estimator):
        p, q = single_entry_coefficients(i, j)
        alpha = nye_tensor(linear_beta_field(p, q), STEP_X1_M, STEP_X2_M)
        density = gnd_density(alpha, BURGERS_NI, estimator)
        assert density.shape == SHAPE
        if (i, j) in ESTIMATOR_COMPONENTS[estimator]:
            expected = ESTIMATOR_PREFACTORS[estimator] * ALPHA_SCALE / BURGERS_NI
            np.testing.assert_allclose(
                density,
                np.full(SHAPE, expected),
                rtol=ORACLE_REL_TOL,
                atol=0,
            )
        else:
            # EXACTLY zero, not merely small: an entry outside the
            # documented set contributes nothing at all
            np.testing.assert_array_equal(density, np.zeros(SHAPE))

    def test_the_default_estimator_is_a5(self):
        parameters = inspect.signature(gnd_density).parameters
        assert parameters["estimator"].default == "a5"
        assert inspect.signature(hrebsd_gnd).parameters["estimator"].default == "a5"

    def test_an_unknown_estimator_is_refused(self):
        alpha = np.zeros((3, 3))
        with pytest.raises(ValueError, match="estimator"):
            gnd_density(alpha, BURGERS_NI, "a7")


class TestBurgersVector:
    """``burgers_vector_length`` is in METRES, required.  [D14.4/D14.5]"""

    def test_the_density_scales_as_one_over_b(self):
        alpha = ALPHA_SCALE * np.ones((3, 3))
        first = gnd_density(alpha, BURGERS_NI, "a5")
        second = gnd_density(alpha, 2 * BURGERS_NI, "a5")
        assert first == pytest.approx(2 * second, rel=1e-12)

    def test_a_burgers_vector_in_nanometres_is_nine_orders_out(self):
        # the plan 4.3 mutant "b in nm not m", quantified.  Nothing in
        # the arithmetic can catch it, which is exactly why
        # requirements D14.5 gives the argument no default and states
        # the unit in the name of the failure mode
        alpha = ALPHA_SCALE * np.ones((3, 3))
        in_metres = gnd_density(alpha, BURGERS_NI, "a5")
        in_nanometres = gnd_density(alpha, BURGERS_NI * 1e9, "a5")
        assert in_metres / in_nanometres == pytest.approx(1e9, rel=1e-9)
        assert not np.isclose(in_metres, in_nanometres, rtol=1e6)

    def test_the_absolute_density_is_the_literature_scale(self):
        # requirements D14.6 quotes rho_noise ~ sigma_beta/(b*step),
        # about 4e12 to 8e12 m^-2 at sigma_beta = 1e-4, b = 0.25 nm and
        # a 100 nm step.  This module's ALPHA_SCALE is exactly that
        # 1e-4 over that 1e-7 m, so the single-entry "a3" density lands
        # WITHIN AN ORDER of that class -- (30/10) * 1e3 / 2.49e-10 =
        # 1.2e13 m^-2, one and a half to three times the top of the
        # quoted band, because the estimator's own prefactor and its
        # three-term sum are not in the noise-floor identity (wording
        # corrected 2026-09-08, Stage C adversarial review: it said
        # "lands in that class").  The assertion is the order check it
        # always was, and that is the sanity the absolute value of
        # every number here rests on
        p, q = single_entry_coefficients(0, 2)
        alpha = nye_tensor(linear_beta_field(p, q), STEP_X1_M, STEP_X2_M)
        density = float(np.mean(gnd_density(alpha, BURGERS_NI, "a3")))
        assert 1e12 < density < 1e14

    @pytest.mark.parametrize("bad", [0.0, -1e-10, np.nan, np.inf])
    def test_a_non_positive_or_non_finite_b_is_refused(self, bad):
        with pytest.raises(ValueError, match="burgers_vector_length"):
            gnd_density(np.zeros((3, 3)), bad, "a5")

    def test_it_has_no_default(self):
        parameters = inspect.signature(hrebsd_gnd).parameters
        assert parameters["burgers_vector_length"].default is inspect.Parameter.empty
        assert (
            parameters["burgers_vector_length"].kind
            is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )


# ================ D14.5 -- steps, units and metres ================== #


class TestScanUnit:
    """Gradients per METRE, from ``scan_unit``.  [D14.5]"""

    def test_the_conversion_table_is_the_documented_one(self):
        assert SCAN_UNIT_TO_METERS["m"] == 1.0
        assert SCAN_UNIT_TO_METERS["nm"] == 1e-9
        assert SCAN_UNIT_TO_METERS["um"] == 1e-6
        # kikuchipy writes the scan unit from the signal axes and
        # either micro sign can arrive, U+00B5 or U+03BC
        assert SCAN_UNIT_TO_METERS["µm"] == 1e-6
        assert SCAN_UNIT_TO_METERS["μm"] == 1e-6
        # and "px", orix's own default, is deliberately absent
        assert "px" not in SCAN_UNIT_TO_METERS

    def test_the_steps_come_back_in_metres_and_in_the_right_order(self):
        # THE axis pin: the map is deliberately anisotropic, so a
        # row-for-column swap changes both numbers.  x1 is the COLUMN
        # axis and x2 the ROW axis, requirements D12's identification
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        step_x1, step_x2 = scan_step_meters(xmap)
        assert step_x1 == pytest.approx(STEP_X1_M, rel=1e-12)
        assert step_x2 == pytest.approx(STEP_X2_M, rel=1e-12)
        assert step_x2 == pytest.approx(2 * step_x1, rel=1e-12)

    @pytest.mark.parametrize(
        "unit, factor", [("m", 1.0), ("nm", 1e-9), ("um", 1e-6), ("µm", 1e-6)]
    )
    def test_every_accepted_unit_converts(self, unit, factor):
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR), scan_unit=unit)
        step_x1, step_x2 = scan_step_meters(xmap)
        assert step_x1 == pytest.approx(STEP_X1_UM * factor, rel=1e-12)
        assert step_x2 == pytest.approx(STEP_X2_UM * factor, rel=1e-12)

    @pytest.mark.parametrize("unit", ["px", "mm", "A", "", "microns"])
    def test_an_unknown_unit_is_refused_by_name(self, unit):
        # requirements D14.5, amended at the spec review: an earlier
        # draft silently assumed micrometres, which is the exact
        # unit-guessing the burgers_vector_length rule forbids.  orix's
        # OWN default is "px", so this guard fires on a map nobody
        # thought about rather than on an exotic one
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR), scan_unit=unit)
        with pytest.raises(ValueError, match="scan_unit"):
            scan_step_meters(xmap)
        with pytest.raises(ValueError, match="scan_unit"):
            hrebsd_gnd(xmap, make_detector(), BURGERS_NI)

    def test_a_missing_unit_is_refused(self):
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        xmap.scan_unit = None
        with pytest.raises(ValueError, match="scan_unit"):
            scan_step_meters(xmap)

    def test_the_gradient_is_per_metre_and_not_per_pixel(self):
        # the plan 4.3 mutant "gradients in pixels not metres",
        # quantified: the same numeric map described in micrometres
        # and in nanometres differs by exactly a thousand, and an
        # implementation working in map INDICES would return the same
        # number for both
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR)
        detector = make_detector()
        in_um = hrebsd_gnd(gnd_map(field, scan_unit="um"), detector, BURGERS_NI)
        in_nm = hrebsd_gnd(gnd_map(field, scan_unit="nm"), detector, BURGERS_NI)
        np.testing.assert_allclose(in_nm, 1e3 * in_um, rtol=1e-10, atol=0)
        # and the absolute value is the metre one.  PER COLUMN, and
        # step-aware (corrected 2026-09-08, Stage C adversarial
        # review): an earlier draft compared the two maxima and
        # asserted the single factor 1 / step_x1, which a correct
        # implementation MISSES by 24 per cent, because the steps here
        # are deliberately anisotropic.  alpha_i1 = +d(beta_i3)/dx2
        # carries step_x2 alone and alpha_i2 = -d(beta_i3)/dx1 carries
        # step_x1 alone, so those two columns rescale EXACTLY; the
        # third mixes both steps and no single factor relates it, nor
        # the two maxima, whose argmax entries are not even the same
        generic = linear_beta_field(P_GENERIC, Q_GENERIC)
        alpha = nye_tensor(generic, STEP_X1_M, STEP_X2_M)
        per_index = nye_tensor(generic, 1.0, 1.0)
        np.testing.assert_allclose(
            alpha[..., 0], per_index[..., 0] / STEP_X2_M, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            alpha[..., 1], per_index[..., 1] / STEP_X1_M, rtol=1e-12, atol=0
        )
        # on an ISOTROPIC-step field the whole tensor does rescale by
        # the one factor, mixed column included, which is the arm the
        # ratio above was reaching for
        isotropic = linear_beta_field(
            P_GENERIC, Q_GENERIC, step_x1=STEP_X1_M, step_x2=STEP_X1_M
        )
        np.testing.assert_allclose(
            nye_tensor(isotropic, STEP_X1_M, STEP_X1_M),
            nye_tensor(isotropic, 1.0, 1.0) / STEP_X1_M,
            rtol=1e-12,
            atol=0,
        )

    def test_an_irregular_grid_still_gives_a_positive_step_of_the_right_class(self):
        # the grid nobody plans for.  Every other map in this module is
        # uniform and ASCENDING, and neither requirements D14.5 nor the
        # skeleton says what a descending or a ragged coordinate array
        # does, so what is pinned here is only what the frozen text
        # does imply: a STEP is a positive length.  A negative one
        # would flip the sign of every gradient and so of alpha, and a
        # zero or a NaN one on a map that plainly has two columns would
        # silently NaN the whole density
        descending = gnd_map(
            linear_beta_field(P_DETECTOR, Q_DETECTOR),
            step_sizes=(-STEP_X2_UM, -STEP_X1_UM),
        )
        step_x1, step_x2 = scan_step_meters(descending)
        assert step_x1 == pytest.approx(STEP_X1_M, rel=1e-9)
        assert step_x2 == pytest.approx(STEP_X2_M, rel=1e-9)
        # a RAGGED one, where the spacing is not constant along a row,
        # is a LIBRARY MEASUREMENT and NOT a claim about this module:
        # orix cannot represent it at all.  It infers the grid from the
        # smallest spacing, so twelve points on four unevenly spaced
        # columns come back as a three by SEVEN map with five of every
        # seven positions empty, and its own row accessor then raises
        # on the length mismatch.  A ragged coordinate array therefore
        # never reaches this module through the documented route --
        # EBSD.hrebsd_dic builds its map from the signal's uniform
        # navigation axes -- so nothing here fabricates a step for one,
        # and there is no behaviour of ours left to pin
        columns = np.array([0.0, 1.0, 3.0, 6.0]) * STEP_X1_UM
        ragged = CrystalMap(
            rotations=Rotation.identity((12,)),
            phase_id=np.zeros(12, dtype=int),
            phase_list=PhaseList(Phase(name="ni", space_group=225)),
            x=np.tile(columns, 3),
            y=np.repeat(np.arange(3.0) * STEP_X2_UM, 4),
        )
        assert tuple(ragged.shape) == (3, 7)
        assert ragged.size == 12 < int(np.prod(ragged.shape))


# ============= D14.3 -- the enforced antisymmetry fix =============== #


class TestAntisymmetry:
    """In the DETECTOR frame, and on the GND path only.  [D14.3/V7]"""

    @staticmethod
    def asymmetric_field():
        """Return a detector-frame field whose ``beta31``/``beta32``
        are NOT the negatives of ``beta13``/``beta23``, so the fix
        does something, while the trace and ``beta33`` stay zero so
        the reduction and the closure stay no-ops.

        Only the two replaced slots are changed, and their values were
        CHOSEN so that the three quantities the tests below compare --
        the fix in the detector frame, the fix in the sample frame and
        no fix at all -- separate by at least fifty per cent on every
        one of the three estimators, that is by fifty times the
        comparison band. The self-check inside
        :meth:`test_the_fix_is_applied_in_the_detector_frame` asserts
        the separation rather than trusting this note.
        """
        p = P_DETECTOR.copy()
        q = Q_DETECTOR.copy()
        b0 = B0_DETECTOR.copy()
        p[2, 0] = -2.9 * ALPHA_SCALE
        p[2, 1] = -2.4 * ALPHA_SCALE
        q[2, 0] = 0.3 * ALPHA_SCALE
        q[2, 1] = 3.0 * ALPHA_SCALE
        b0[2, 0] = -1.1 * OFFSET_SCALE
        b0[2, 1] = 0.8 * OFFSET_SCALE
        return p, q, b0

    def test_the_replacement_is_beta31_from_minus_beta13(self):
        rng = np.random.default_rng(7)
        beta = rng.normal(size=(3, 3))
        before = beta.copy()
        got = enforce_beta_antisymmetry(beta)
        assert got[2, 0] == -before[0, 2]
        assert got[2, 1] == -before[1, 2]
        # every other entry untouched, and the input not modified in
        # place: the caller's "Fe" property is a public array
        for i in range(3):
            for j in range(3):
                if (i, j) not in {(2, 0), (2, 1)}:
                    assert got[i, j] == before[i, j]
        np.testing.assert_array_equal(beta, before)
        assert ANTISYMMETRY_REPLACEMENTS == (((2, 0), (0, 2)), ((2, 1), (1, 2)))

    def test_the_backwards_replacement_is_a_different_operator(self):
        # the plan 4.3 mutant "antisymmetry replacing beta13/23 by
        # -beta31/32 (backwards)": it keeps the NOISY pair and
        # discards the quiet one, and it is a genuinely different
        # tensor whenever the two pairs differ
        rng = np.random.default_rng(11)
        beta = rng.normal(size=(3, 3))
        forwards = enforce_beta_antisymmetry(beta)
        backwards = beta.copy()
        backwards[0, 2] = -backwards[2, 0]
        backwards[1, 2] = -backwards[2, 1]
        assert not np.allclose(forwards, backwards)

    def test_it_is_idempotent_and_a_no_op_on_an_antisymmetric_pair(self):
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR)
        stacked = field.reshape(-1, 3, 3)
        once = enforce_beta_antisymmetry(stacked)
        # the module's own coefficient matrices satisfy the relation
        # already, which is why the through-the-function oracles below
        # are blind to the toggle by construction
        np.testing.assert_allclose(once, stacked, rtol=0, atol=ALGEBRA_TOL)
        np.testing.assert_allclose(
            enforce_beta_antisymmetry(once), once, rtol=0, atol=0
        )

    def test_the_fix_is_applied_in_the_detector_frame(self):
        # THE order-of-operations pin of requirements D14.3.  The frame
        # matrix M is not diagonal, so replacing two entries and then
        # conjugating is a different operator from conjugating and then
        # replacing two entries: the oracle computes BOTH and the
        # implementation has to match the first and miss the second
        p, q, b0 = self.asymmetric_field()
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        got = hrebsd_gnd(xmap, detector, BURGERS_NI)
        in_detector_frame = expected_alpha_through_the_chain(
            p, q, detector, fix="detector"
        )
        in_sample_frame = expected_alpha_through_the_chain(p, q, detector, fix="sample")
        expected = density_by_hand(in_detector_frame, BURGERS_NI, "a5")
        mutant = density_by_hand(in_sample_frame, BURGERS_NI, "a5")
        # the two orders really do disagree on this field, and by far
        # more than the band the implementation is compared in, so the
        # assertion below is a discrimination and not a coincidence
        assert abs(expected - mutant) > 10 * ORACLE_REL_TOL * expected
        np.testing.assert_allclose(
            got, np.full(SHAPE, expected), rtol=ORACLE_REL_TOL, atol=0
        )
        assert not np.allclose(got, np.full(SHAPE, mutant), rtol=ORACLE_REL_TOL)

    def test_the_fix_is_applied_before_the_closure(self):
        # the SECOND order-of-operations mutant, and it needs a
        # stiffness to be visible at all (added 2026-09-08, Stage C
        # adversarial review, which listed it as uncovered).  The
        # deviatoric closure of requirements D9.3 adds a multiple of
        # the identity, so it commutes exactly with a fix that writes
        # only the off-diagonal 31 and 32 slots, and on every
        # deviatoric field in this module the two orders are the SAME
        # OPERATOR.  The traction-free solve of requirements D9.2 does
        # not commute: its right-hand side reads (beta13 + beta31) and
        # (beta23 + beta32), which the fix sends to exactly zero, so
        # fixing first changes the isotropic part the closure adds
        p, q, b0 = self.asymmetric_field()
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        stiffness_sample = rotate_stiffness(
            stiffness, xmap.rotations.to_matrix().reshape(-1, 3, 3)
        )[0]
        matrix = sample_to_spec_detector_matrix(detector)

        def alpha_in_order(order):
            # the chain is linear in the field and close_beta is
            # linear too, so it acts on the coefficient matrices one
            # at a time and no finite difference enters this oracle
            transformed = []
            for coefficients in (p, q):
                coefficients = np.asarray(coefficients, dtype=np.float64)
                if order == "before":
                    coefficients = antisymmetry_by_hand(coefficients)
                coefficients = to_sample_frame(coefficients, matrix)
                coefficients = close_beta(
                    coefficients,
                    stiffness_sample=stiffness_sample,
                    closure="traction_free",
                )
                if order == "after":
                    coefficients = antisymmetry_by_hand(coefficients)
                transformed.append(coefficients)
            return alpha_from_the_definition(*transformed)

        expected = density_by_hand(alpha_in_order("before"), BURGERS_NI, "a5")
        mutant = density_by_hand(alpha_in_order("after"), BURGERS_NI, "a5")
        # the two orders separate by far more than the band, so the
        # assertion is a discrimination and not a coincidence
        assert abs(expected - mutant) > 10 * ORACLE_REL_TOL * expected
        got = hrebsd_gnd(xmap, detector, BURGERS_NI, stiffness=stiffness)
        np.testing.assert_allclose(
            got, np.full(SHAPE, expected), rtol=ORACLE_REL_TOL, atol=0
        )
        assert not np.allclose(got, np.full(SHAPE, mutant), rtol=ORACLE_REL_TOL)

    def test_the_fix_and_the_deviatoric_closure_commute(self):
        # A LIBRARY MEASUREMENT recording WHY the test above needs a
        # stiffness: with the deviatoric closure the two orders are
        # the same operator on any field at all, so no deviatoric case
        # in this module can discriminate them and none pretends to
        rng = np.random.default_rng(23)
        for _ in range(5):
            tensor = rng.normal(size=(3, 3))
            np.testing.assert_allclose(
                deviatoric(antisymmetry_by_hand(tensor)),
                antisymmetry_by_hand(deviatoric(tensor)),
                rtol=0,
                atol=ALGEBRA_TOL,
            )

    def test_the_toggle_changes_the_gnd_density(self):
        p, q, b0 = self.asymmetric_field()
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        on = hrebsd_gnd(xmap, detector, BURGERS_NI, enforce_antisymmetry=True)
        off = hrebsd_gnd(xmap, detector, BURGERS_NI, enforce_antisymmetry=False)
        assert not np.allclose(on, off)
        unfixed = expected_alpha_through_the_chain(p, q, detector, fix="none")
        np.testing.assert_allclose(
            off,
            np.full(SHAPE, density_by_hand(unfixed, BURGERS_NI, "a5")),
            rtol=ORACLE_REL_TOL,
            atol=0,
        )

    def test_it_is_on_by_default(self):
        parameters = inspect.signature(hrebsd_gnd).parameters
        assert parameters["enforce_antisymmetry"].default is True
        p, q, b0 = self.asymmetric_field()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        detector = make_detector()
        np.testing.assert_array_equal(
            hrebsd_gnd(xmap, detector, BURGERS_NI),
            hrebsd_gnd(xmap, detector, BURGERS_NI, enforce_antisymmetry=True),
        )

    def test_the_strain_and_stress_path_is_bitwise_unchanged(self):
        # "Applied on the GND path ONLY, never to stress" (requirements
        # D14.3, frozen).  The Stage B chain is run before and after
        # both toggles of the GND path and has to return the SAME BITS,
        # and the stored Fe property has to come back untouched
        p, q, b0 = self.asymmetric_field()
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        fe_before = np.array(xmap.prop["Fe"], copy=True)
        before = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        on = hrebsd_gnd(xmap, detector, BURGERS_NI, enforce_antisymmetry=True)
        off = hrebsd_gnd(xmap, detector, BURGERS_NI, enforce_antisymmetry=False)
        after = hrebsd_strain_stress(xmap, detector, stiffness=stiffness)
        for name in STAGE_B_PROP_NAMES:
            assert np.array_equal(
                np.asarray(before.prop[name]),
                np.asarray(after.prop[name]),
                equal_nan=True,
            ), name
        # the input map is not mutated either, which is what would let
        # the fix leak onto the stress path
        np.testing.assert_array_equal(np.asarray(xmap.prop["Fe"]), fe_before)
        # and the toggle really did something, so this is not a
        # statement about two identical runs
        assert not np.allclose(on, off)

    def test_the_fix_changes_no_strain_even_where_it_changes_the_gnd(self):
        # the same claim from the other side: the sample-frame beta the
        # Stage B chain reports is built WITHOUT the fix, so it differs
        # from the one the GND path differentiates whenever the field
        # is not already antisymmetric.  A shared, fixed beta would
        # show up here as a changed strain
        p, q, b0 = self.asymmetric_field()
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(p, q, b0))
        reported = np.asarray(hrebsd_strain_stress(xmap, detector).prop["beta"])
        matrix = sample_to_spec_detector_matrix(detector)
        raw = np.stack(
            [
                deviatoric(to_sample_frame(tensor, matrix))
                for tensor in linear_beta_field(p, q, b0).reshape(-1, 3, 3)
            ]
        ).reshape(-1, BETA_PROP_SIZE)
        np.testing.assert_allclose(reported, raw, rtol=1e-9, atol=ALGEBRA_TOL)


# =============== D14.6 -- the logarithmic plotting map ============== #


class TestLog10Guard:
    """``log10`` of a signed or zero density is guarded.  [D14.6]"""

    def test_a_positive_density_is_the_plain_logarithm(self):
        density = np.array([1.0, 1e12, 4e12, 8e12, 1e16])
        np.testing.assert_allclose(
            log10_density(density), np.log10(density), rtol=0, atol=ALGEBRA_TOL
        )

    def test_zero_negative_and_nan_all_become_nan_without_warning(self):
        # the plan 4.3 mutant "log10 of signed values unguarded": a
        # bare numpy.log10 returns -inf at zero and NaN at a negative
        # value, and WARNS for both, which on a map of a strain-free
        # region is every point
        density = np.array([0.0, -1.0, -1e12, np.nan, 1e12])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = log10_density(density)
        assert np.all(np.isnan(got[:4]))
        assert got[4] == pytest.approx(12.0, rel=1e-12)
        assert not np.any(np.isinf(got))

    def test_the_shape_and_dtype_are_preserved(self):
        got = log10_density(np.ones((3, 4)))
        assert got.shape == (3, 4)
        assert got.dtype == np.float64

    def test_a_density_is_never_negative(self):
        # the guard exists because a density CAN be zero, never
        # because it can be negative: it is a sum of moduli
        rng = np.random.default_rng(5)
        alpha = ALPHA_SCALE * rng.normal(size=(20, 3, 3))
        for estimator in SUPPORTED_ESTIMATORS:
            assert np.all(gnd_density(alpha, BURGERS_NI, estimator) >= 0.0)


# ================= D14.5 -- the NaN safety contract ================= #


class TestNaNSafety:
    """Nothing is ever fabricated at a boundary, an edge or a hole.
    [D14.5/V7]"""

    @staticmethod
    def default_field():
        return linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR)

    def test_a_grain_boundary_refuses_the_pairs_that_cross_it(self):
        # the left three columns and the right three are two grains.
        # A central pair straddling the boundary is refused, so the two
        # columns either side are NaN, while a column with a full
        # same-grain stencil keeps the constant
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[:, 3:] = 1
        got = hrebsd_gnd(
            gnd_map(self.default_field(), grain_id=grain_id),
            make_detector(),
            BURGERS_NI,
        )
        assert np.all(np.isnan(got[:, 2]))
        assert np.all(np.isnan(got[:, 3]))
        # and the far columns still have their one-sided or central
        # pairs inside one grain
        assert np.all(np.isfinite(got[:, 0]))
        assert np.all(np.isfinite(got[:, 5]))

    def test_a_point_outside_every_grain_is_nan_and_poisons_no_value(self):
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[2, 3] = UNLABELLED
        got = hrebsd_gnd(
            gnd_map(self.default_field(), grain_id=grain_id),
            make_detector(),
            BURGERS_NI,
        )
        assert np.isnan(got[2, 3])
        # its two neighbours along each axis lose the pair they need
        assert np.isnan(got[2, 2])
        assert np.isnan(got[2, 4])
        assert np.isnan(got[1, 3])
        assert np.isnan(got[3, 3])
        # a point two steps away still has its own full stencil
        assert np.isfinite(got[0, 0])

    def test_a_one_point_grain_is_nan(self):
        grain_id = np.zeros(SHAPE, dtype=np.int32)
        grain_id[2, 3] = 7
        got = hrebsd_gnd(
            gnd_map(self.default_field(), grain_id=grain_id),
            make_detector(),
            BURGERS_NI,
        )
        assert np.isnan(got[2, 3])

    def test_a_non_converged_point_is_nan_and_is_never_zeroed(self):
        # requirements D2.6, quoted: a non-converged, failed or masked
        # point "get[s] NaN in every derived prop downstream; they are
        # NEVER zeroed".  A zero here would read as a dislocation-free
        # point.
        #
        # The SELF rule is what this pins, and it is not the pair rule
        # of requirements D14.5 (recorded 2026-09-08, Stage C
        # adversarial review): a central-difference pair never reads
        # its own centre, so a pair-only implementation leaves point
        # (1, 1) with the finite pairs (1, 0)/(1, 2) and (0, 1)/(2, 1)
        # and reports the full constant density there.  D2.6 forbids
        # that, and the arm at the nye_tensor level below pins where
        # it has to be enforced rather than leaving it to the caller
        field = self.default_field()
        field[1, 1] = np.nan
        got = hrebsd_gnd(gnd_map(field), make_detector(), BURGERS_NI)
        assert np.isnan(got[1, 1])
        assert np.isnan(got[1, 0])
        assert np.isnan(got[0, 1])
        assert np.isfinite(got[4, 5])
        assert not np.any(got[np.isfinite(got)] == 0.0)

    def test_a_non_finite_point_is_nan_in_alpha_itself(self):
        # the same rule one level down, so that an implementation
        # cannot satisfy it by a special case in hrebsd_gnd alone: the
        # whole tensor at the point is NaN, not only the entries whose
        # PAIR happens to read the hole.  Two steps away in both
        # directions the stencil is untouched and the constant
        # survives, which is what makes this a rule about ONE point and
        # not a mask that spreads
        field = self.default_field()
        field[1, 1] = np.nan
        alpha = nye_tensor(field, STEP_X1_M, STEP_X2_M)
        assert np.all(np.isnan(alpha[1, 1]))
        # its neighbours lose only the derivative whose pair reaches
        # the hole, which is the PAIR rule of requirements D14.5 doing
        # its own separate job: at (1, 0) the one-sided d/dx1 pair
        # reads it and every d/dx2 entry survives
        assert np.all(np.isnan(alpha[1, 0][:, 1:]))
        assert np.all(np.isfinite(alpha[1, 0][:, 0]))
        expected = alpha_from_the_definition(P_DETECTOR, Q_DETECTOR)
        for point in ((3, 3), (4, 5)):
            np.testing.assert_allclose(
                alpha[point], expected, rtol=1e-10, atol=ALGEBRA_TOL * ALPHA_SCALE
            )

    def test_the_map_edges_are_one_sided_and_finite(self):
        got = hrebsd_gnd(gnd_map(self.default_field()), make_detector(), BURGERS_NI)
        assert np.all(np.isfinite(got))
        # a constant-curvature field has no edge artefact, so every
        # point carries the same density
        np.testing.assert_allclose(got, got[2, 3], rtol=ORACLE_REL_TOL, atol=0)

    def test_an_all_nan_map_is_all_nan(self):
        field = np.full(SHAPE + (3, 3), np.nan)
        got = hrebsd_gnd(gnd_map(field), make_detector(), BURGERS_NI)
        assert np.all(np.isnan(got))

    def test_a_one_column_map_has_no_x1_derivative(self):
        # the geometry has to be refused rather than guessed: with one
        # column there is no x1 neighbour, so alpha_i2 and alpha_i3 are
        # NaN while alpha_i1, which needs only x2, survives.  Every
        # estimator reads at least one NaN entry, so the DENSITY is NaN
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, shape=(4, 1))
        xmap = gnd_map(field)
        step_x1, step_x2 = scan_step_meters(xmap)
        assert np.isnan(step_x1)
        assert step_x2 == pytest.approx(STEP_X2_M, rel=1e-12)
        alpha = nye_tensor(field, step_x1, step_x2)
        assert np.all(np.isfinite(alpha[..., 0]))
        assert np.all(np.isnan(alpha[..., 1]))
        assert np.all(np.isnan(alpha[..., 2]))
        got = hrebsd_gnd(xmap, make_detector(), BURGERS_NI)
        assert got.shape == (4, 1)
        assert np.all(np.isnan(got))

    def test_a_one_row_map_has_no_x2_derivative(self):
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, shape=(1, 4))
        xmap = gnd_map(field)
        step_x1, step_x2 = scan_step_meters(xmap)
        assert step_x1 == pytest.approx(STEP_X1_M, rel=1e-12)
        assert np.isnan(step_x2)
        got = hrebsd_gnd(xmap, make_detector(), BURGERS_NI)
        assert got.shape == (1, 4)
        assert np.all(np.isnan(got))


# ============ D14 -- the private helpers' own guards =============== #


class TestHelperGuards:
    """Every documented ``ValueError`` of the four private helpers, and
    their ``grain`` keyword directly.  [D14/D15]

    ADDED 2026-09-08 at the Stage C adversarial review, which found six
    documented raises with no test at all and the ``grain`` argument of
    the two gradient helpers exercised only through
    :func:`hrebsd_gnd`.  The stage recipe asks for 100 per cent
    coverage of the stage's modules, and a raise nobody calls is a
    branch nobody has read.
    """

    def test_enforce_beta_antisymmetry_refuses_a_non_3x3_tensor(self):
        for bad in (np.zeros((3, 2)), np.zeros((4, 4)), np.zeros(3), np.zeros((2, 9))):
            with pytest.raises(ValueError, match="beta_detector"):
                enforce_beta_antisymmetry(bad)

    def test_enforce_beta_antisymmetry_takes_a_stack_or_a_single(self):
        single = enforce_beta_antisymmetry(np.zeros((3, 3)))
        assert single.shape == (3, 3)
        stacked = enforce_beta_antisymmetry(np.zeros((7, 3, 3)))
        assert stacked.shape == (7, 3, 3)
        assert stacked.dtype == np.float64

    def test_in_plane_gradients_refuses_a_one_axis_field(self):
        with pytest.raises(ValueError, match="field"):
            in_plane_gradients(np.zeros(6), STEP_X1_M, STEP_X2_M)

    def test_in_plane_gradients_refuses_a_mismatched_grain_array(self):
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR)
        for bad in (np.zeros((4, 6), dtype=np.int32), np.zeros(SHAPE[0] * SHAPE[1])):
            with pytest.raises(ValueError, match="grain"):
                in_plane_gradients(field, STEP_X1_M, STEP_X2_M, grain=bad)

    def test_in_plane_gradients_takes_its_grain_argument_directly(self):
        # the keyword itself, not through hrebsd_gnd: a two-grain map
        # refuses the pairs that cross the boundary and keeps the
        # coefficient matrices everywhere else
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR)
        grain = np.zeros(SHAPE, dtype=np.int32)
        grain[:, 3:] = 1
        d_dx1, d_dx2 = in_plane_gradients(field, STEP_X1_M, STEP_X2_M, grain=grain)
        assert np.all(np.isnan(d_dx1[:, 2]))
        assert np.all(np.isnan(d_dx1[:, 3]))
        # the ROW derivative never crosses the column boundary, so it
        # is untouched: a grain rule which refused whole points rather
        # than pairs would fail here
        np.testing.assert_allclose(
            d_dx2,
            np.broadcast_to(Q_DETECTOR, d_dx2.shape),
            rtol=1e-10,
            atol=ALGEBRA_TOL * ALPHA_SCALE,
        )
        # and a point outside every grain has NaN in BOTH
        unlabelled = np.zeros(SHAPE, dtype=np.int32)
        unlabelled[2, 3] = UNLABELLED
        first, second = in_plane_gradients(
            field, STEP_X1_M, STEP_X2_M, grain=unlabelled
        )
        assert np.all(np.isnan(first[2, 3]))
        assert np.all(np.isnan(second[2, 3]))

    def test_nye_tensor_refuses_a_field_that_is_not_ny_nx_3_3(self):
        for bad in (
            np.zeros(SHAPE + (9,)),
            np.zeros(SHAPE + (3, 2)),
            np.zeros((3, 3)),
            np.zeros(SHAPE + (2, 3, 3)),
        ):
            with pytest.raises(ValueError, match="beta"):
                nye_tensor(bad, STEP_X1_M, STEP_X2_M)

    def test_nye_tensor_refuses_a_mismatched_grain_array(self):
        with pytest.raises(ValueError, match="grain"):
            nye_tensor(
                linear_beta_field(P_DETECTOR, Q_DETECTOR),
                STEP_X1_M,
                STEP_X2_M,
                grain=np.zeros((2, 2), dtype=np.int32),
            )

    def test_nye_tensor_takes_its_grain_argument_directly(self):
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR)
        grain = np.zeros(SHAPE, dtype=np.int32)
        grain[:, 3:] = 1
        alpha = nye_tensor(field, STEP_X1_M, STEP_X2_M, grain=grain)
        assert np.all(np.isnan(alpha[:, 2, :, 1]))
        assert np.all(np.isnan(alpha[:, 3, :, 1]))
        # alpha_i1 needs only the row derivative, which no column
        # boundary can cross, so it survives with the constant
        expected = alpha_from_the_definition(P_DETECTOR, Q_DETECTOR)
        np.testing.assert_allclose(
            alpha[..., 0],
            np.broadcast_to(expected[:, 0], SHAPE + (3,)),
            rtol=1e-10,
            atol=ALGEBRA_TOL * ALPHA_SCALE,
        )

    def test_gnd_density_refuses_a_non_3x3_alpha(self):
        for bad in (np.zeros((3, 2)), np.zeros(9), np.zeros(SHAPE + (9,))):
            with pytest.raises(ValueError, match="alpha"):
                gnd_density(bad, BURGERS_NI, "a5")


# =================== D14/D15 -- the public contract ================= #


class TestContract:
    """Signature, guards, shape and determinism.  [D14/D15]"""

    def test_signature_is_frozen(self):
        parameters = inspect.signature(hrebsd_gnd).parameters
        assert list(parameters) == [
            "xmap",
            "detector",
            "burgers_vector_length",
            "stiffness",
            "estimator",
            "enforce_antisymmetry",
        ]
        for name in ("xmap", "detector", "burgers_vector_length"):
            assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
                name
            )
            assert parameters[name].default is inspect.Parameter.empty, name
        for name in ("stiffness", "estimator", "enforce_antisymmetry"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameters["stiffness"].default is None
        assert parameters["estimator"].default == "a5"
        assert parameters["enforce_antisymmetry"].default is True

    def test_the_required_properties_are_the_documented_two(self):
        assert REQUIRED_PROP_NAMES == ("Fe", "grain_id")

    def test_the_unlabelled_sentinel_agrees_with_every_other_copy(self):
        # four private modules now spell the same sentinel, and only
        # two of them were pinned against each other (added 2026-09-08,
        # Stage C adversarial review).  This module's copy decides
        # whether a point outside every grain is refused, so a drift
        # here would silently fabricate gradients ACROSS unindexed
        # points rather than raising anything
        assert UNLABELLED == REFERENCE_UNLABELLED == KAM_UNLABELLED == UNINDEXED == -1

    def test_it_returns_an_array_and_stores_nothing(self):
        # requirements D15.6: the three diagnostics return arrays and
        # never add properties to the map
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        before = set(xmap.prop)
        got = hrebsd_gnd(xmap, make_detector(), BURGERS_NI)
        assert isinstance(got, np.ndarray)
        assert got.shape == SHAPE
        assert got.dtype == np.float64
        assert set(xmap.prop) == before

    def test_two_runs_are_bitwise_identical(self):
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR, B0_DETECTOR))
        detector = make_detector()
        first = hrebsd_gnd(xmap, detector, BURGERS_NI)
        second = hrebsd_gnd(xmap, detector, BURGERS_NI)
        assert np.array_equal(first, second, equal_nan=True)

    def test_the_three_estimators_all_run_and_differ(self):
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        detector = make_detector()
        values = {
            name: float(
                np.nanmean(hrebsd_gnd(xmap, detector, BURGERS_NI, estimator=name))
            )
            for name in SUPPORTED_ESTIMATORS
        }
        assert all(np.isfinite(value) and value > 0 for value in values.values())
        assert len(set(values.values())) == 3

    def test_a_stiffness_selects_the_traction_free_closure(self):
        # the two closures of requirements D9 differ only in the
        # ISOTROPIC part they add, and the curl feels that part
        # through beta11, beta22 and beta33 -- but only when the
        # measured field has a TRACE to close in the first place.  The
        # module's own coefficients are traceless on purpose, so a
        # field with a large trace gradient is built here, which
        # separates the two closures by about thirteen per cent
        p = P_DETECTOR.copy()
        q = Q_DETECTOR.copy()
        p[0, 0] += 3 * ALPHA_SCALE
        q[1, 1] -= 4 * ALPHA_SCALE
        xmap = gnd_map(linear_beta_field(p, q))
        detector = make_detector()
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        deviatoric_density = hrebsd_gnd(xmap, detector, BURGERS_NI)
        traction_free = hrebsd_gnd(xmap, detector, BURGERS_NI, stiffness=stiffness)
        assert np.all(np.isfinite(traction_free))
        assert np.all(np.isfinite(deviatoric_density))
        relative = np.abs(traction_free - deviatoric_density) / deviatoric_density
        assert relative.min() > 10 * ORACLE_REL_TOL

    def test_guards(self):
        detector = make_detector()
        xmap = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        without_fe = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        del without_fe.prop["Fe"]
        with pytest.raises(ValueError, match="hrebsd_dic"):
            hrebsd_gnd(without_fe, detector, BURGERS_NI)
        without_grains = gnd_map(linear_beta_field(P_DETECTOR, Q_DETECTOR))
        del without_grains.prop["grain_id"]
        with pytest.raises(ValueError, match="hrebsd_dic"):
            hrebsd_gnd(without_grains, detector, BURGERS_NI)
        with pytest.raises(ValueError, match="estimator"):
            hrebsd_gnd(xmap, detector, BURGERS_NI, estimator="a4")
        with pytest.raises(ValueError, match="burgers_vector_length"):
            hrebsd_gnd(xmap, detector, 0.0)
        with pytest.raises(ValueError, match="stiffness"):
            hrebsd_gnd(xmap, detector, BURGERS_NI, stiffness=np.eye(3))

    def test_a_map_without_a_grid_is_named(self):
        # the shared map_grids guard the three grid-shaped functions
        # read their grids through
        xmap = CrystalMap(
            rotations=Rotation.identity((2,)),
            phase_id=np.zeros(2, dtype=int),
            phase_list=PhaseList(Phase(name="ni", space_group=225)),
            x=np.zeros(2),
            y=np.zeros(2),
        )
        xmap.scan_unit = "um"
        xmap.prop["Fe"] = np.tile(np.eye(3).ravel(), (2, 1))
        xmap.prop["grain_id"] = np.zeros(2, dtype=np.int32)
        with pytest.raises(ValueError, match="no map grid"):
            hrebsd_gnd(xmap, make_detector(), BURGERS_NI)
        # and directly, since that is where the raise is documented
        # (added 2026-09-08, Stage C adversarial review: it was reached
        # only through the public entry point)
        with pytest.raises(ValueError, match="no map grid"):
            scan_step_meters(xmap)

    def test_a_multi_phase_map_with_a_stiffness_is_refused(self):
        # the D9.6 guard of the shared chain, reached through this
        # entry point too
        field = linear_beta_field(P_DETECTOR, Q_DETECTOR)
        size = SHAPE[0] * SHAPE[1]
        phase_id = np.zeros(size, dtype=int)
        phase_id[: size // 2] = 1
        xmap = gnd_map(
            field,
            phase_id=phase_id,
            phase_list=PhaseList(
                [Phase(name="ni", space_group=225), Phase(name="fe", space_group=229)]
            ),
        )
        stiffness = kp.indexing.voigt_stiffness("cubic", **NICKEL_CUBIC)
        with pytest.raises(ValueError, match="phase"):
            hrebsd_gnd(xmap, make_detector(), BURGERS_NI, stiffness=stiffness)

    def test_the_public_name_is_the_module_one(self):
        assert kp.indexing.hrebsd_gnd is hrebsd_gnd
        assert "hrebsd_gnd" in kp.indexing.__all__
        assert list(kp.indexing.__all__) == sorted(kp.indexing.__all__)

    def test_the_public_docstring_links_no_private_name(self):
        # the conventions rule the spherical suite already applies to
        # its own exported names: a Sphinx role pointing at a private
        # name renders as an unresolved literal, since only the names
        # in ``__all__`` get a stub.  It is asserted HERE because this
        # module's public function documents a whole private table of
        # estimators and a private logarithm helper, and naming either
        # in a role is the easy mistake
        role = re.compile(r":(?:func|class|meth|attr|mod|data):`~?([\w.]+)`")
        for target in role.findall(hrebsd_gnd.__doc__):
            private = [part for part in target.split(".") if part.startswith("_")]
            assert not private, f"hrebsd_gnd links the private {target}"

    def test_the_public_docstring_names_its_units_and_assumptions(self):
        docstring = hrebsd_gnd.__doc__
        lowered = docstring.lower()
        for phrase in ("m^-2", "detector frame", "sum of moduli"):
            assert phrase in lowered, phrase
        # the two traps requirements D14.5 gives the argument no
        # default for, and the tier the six extra entries rest on
        assert "METRES" in docstring
        assert "scan_unit" in docstring
        assert "lower bound" in lowered


# ========= V7 -- the curvature carried through the patterns ========= #


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


def _bilinear(master, nii, nij, niip, nijp, di, dj, dim, djm):
    """Vectorized twin of ``_get_pixel_from_master_pattern``."""
    return (
        master[nii, nij] * dim * djm
        + master[niip, nij] * di * djm
        + master[nii, nijp] * dim * dj
        + master[niip, nijp] * di * dj
    )


def project_pattern(detector, rotation, deformation=None, pc_index=None):
    """Return one pattern projected from the Ni Lambert master with an
    imposed deformation gradient (validation V3's recipe, duplicated
    here; see the module docstring's file-layout note)."""
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
    chain = (
        sample_to_spec_detector_matrix(detector) @ sample_to_crystal_matrix(rotation).T
    )
    return chain.T @ np.asarray(fe_detector, dtype=np.float64) @ chain


# The pattern-level oracle's own map and field.  Three by three is the
# smallest grid with an interior point carrying a full central-
# difference stencil in both directions, and every point is projected
# and fitted at 480 px, so it is kept small on purpose and cached.
E2E_SHAPE = (3, 3)
E2E_STEP_UM = 1.0
E2E_STEP_M = E2E_STEP_UM * 1e-6

# The per-step distortion change, in the 1e-3 class HREBSD is built
# for and well inside the 2 degree capture range requirements D5
# records: over the whole map the far corner reaches about 2e-2, that
# is 1.1 degrees.  It is deliberately fifty times the Stage A Fe band
# of 3.1e-5, so that the DIFFERENCE the gradient reads carries signal
E2E_PER_STEP = 5e-3

# The detector-frame coefficient matrices of the imposed field, in
# per-metre units.  Same three constraints as the analytic ones:
# beta33 zero (so Fe33 = 1 and the engine's reduction is exact),
# traceless (so the deviatoric closure is a no-op) and already
# antisymmetric in the 31/32 slots (so the D14.3 fix is a no-op and
# what is measured here is the curvature, not the fix)
E2E_P = (E2E_PER_STEP / E2E_STEP_M) * np.array(
    [
        [0.6, 0.5, -0.3],
        [-0.4, -0.6, 0.7],
        [0.3, -0.7, 0.0],
    ]
)
E2E_Q = (E2E_PER_STEP / E2E_STEP_M) * np.array(
    [
        [-0.4, 0.8, 0.6],
        [0.2, 0.4, -0.5],
        [-0.6, 0.5, 0.0],
    ]
)


@functools.lru_cache(maxsize=1)
def curvature_run():
    """Return the engine result of the imposed-curvature map.

    Nine patterns are projected from the shipped Ni Lambert master
    with the detector-frame field ``beta(x1, x2) = P x1 + Q x2``
    imposed on each, every point sharing ONE projection centre so that
    the beam-scan phantom of requirements D6.2 is exactly the identity
    and what is measured is the curvature alone.  Point ``(0, 0)`` is
    the reference and carries ``beta = 0`` by construction, which is
    what makes the field a RELATIVE one, as HREBSD measures it; a
    uniform gradient has the same Nye tensor either way.

    Returns
    -------
    detector, xmap
        The single-projection-centre detector the chain is given, and
        the crystal map carrying the recovered ``Fe`` and
        ``grain_id`` properties on the ``E2E_STEP_UM`` grid.
    """
    detector = make_detector()
    rotation = generic_rotation()
    field = linear_beta_field(
        E2E_P,
        E2E_Q,
        shape=E2E_SHAPE,
        step_x1=E2E_STEP_M,
        step_x2=E2E_STEP_M,
    )
    patterns = []
    for beta in field.reshape(-1, 3, 3):
        fe_detector = np.eye(3) + beta
        # beta33 is zero by construction, so the reduced tensor of
        # requirements D15.6 IS this one and the engine has nothing to
        # divide out
        assert fe_detector[2, 2] == 1.0
        patterns.append(
            project_pattern(
                detector,
                rotation,
                impose_detector_frame_fe(detector, rotation, fe_detector),
            )
        )
    size = len(patterns)
    per_point = make_detector(pc=np.tile(PC_480, (1, size, 1)))
    properties = run_hrebsd_dic(
        np.stack(patterns),
        E2E_SHAPE,
        per_point,
        reference=(0, 0),
        verbose=0,
    )
    assert np.all(properties["converged"])
    arrays, count = create_coordinate_arrays(E2E_SHAPE, (E2E_STEP_UM, E2E_STEP_UM))
    arrays["rotations"] = Rotation(np.tile(rotation.data.ravel(), (count, 1)))
    arrays["phase_id"] = np.zeros(count, dtype=int)
    arrays["phase_list"] = PhaseList(Phase(name="ni", space_group=225))
    xmap = CrystalMap(**arrays)
    xmap.scan_unit = "um"
    xmap.prop["Fe"] = np.asarray(properties["Fe"], dtype=np.float64).reshape(
        count, BETA_PROP_SIZE
    )
    xmap.prop["grain_id"] = np.asarray(properties["grain_id"]).ravel().astype(np.int32)
    return detector, xmap


class TestEndToEndCurvature:
    """The analytic curvature imposed through DEFORMED-MASTER PATTERNS
    and recovered through the whole chain.  [D14/V7]"""

    def test_the_imposed_field_is_what_this_module_thinks_it_is(self):
        # A LIBRARY MEASUREMENT of the oracle itself, so it passes
        # today: the imposed detector-frame field really is traceless,
        # really has beta33 = 0, really already satisfies the D14.3
        # antisymmetry, and really carries a curvature of the intended
        # scale.  Without this the recovery below could pass on a
        # chain returning zeros
        for coefficients in (E2E_P, E2E_Q):
            assert coefficients[2, 2] == 0.0
            assert (
                abs(np.trace(coefficients)) < ALGEBRA_TOL * np.abs(coefficients).max()
            )
            assert coefficients[2, 0] == -coefficients[0, 2]
            assert coefficients[2, 1] == -coefficients[1, 2]
        field = linear_beta_field(
            E2E_P, E2E_Q, shape=E2E_SHAPE, step_x1=E2E_STEP_M, step_x2=E2E_STEP_M
        )
        # the largest distortion is at the far corner, two steps along
        # each axis, and it has to stay inside the 2 degree capture
        # range requirements D5 records while staying far above the
        # Stage A Fe band of 3.1e-5 that the gradient has to beat
        corner = 2 * (E2E_P + E2E_Q) * E2E_STEP_M
        assert np.abs(field).max() == pytest.approx(np.abs(corner).max(), rel=1e-9)
        assert 5e-3 < np.abs(field).max() < 3e-2
        assert np.rad2deg(np.abs(field).max()) < 2.0
        assert E2E_PER_STEP > 100 * 3.1e-5
        expected = expected_alpha_through_the_chain(
            E2E_P, E2E_Q, make_detector(), fix="detector"
        )
        assert np.count_nonzero(np.abs(expected) > 1.0) == 9

    def test_gnd_end_to_end_curvature(self):
        # THE Stage C half of validation V7: the curvature field
        # imposed through patterns, recovered within GND_E2E_TOL of the
        # analytic density.  Everything between the imposed
        # crystal-frame deformation and the reported density has to be
        # right at once -- the D6 conversion, the D7 frame, the D9
        # closure, the D14.1/D14.2 curl, the D14.5 metre steps and the
        # D14.4 estimator
        detector, xmap = curvature_run()
        alpha = expected_alpha_through_the_chain(E2E_P, E2E_Q, detector, fix="detector")
        worst = 0.0
        for estimator in SUPPORTED_ESTIMATORS:
            expected = density_by_hand(alpha, BURGERS_SI, estimator)
            assert expected > 0.0
            got = hrebsd_gnd(xmap, detector, BURGERS_SI, estimator=estimator)
            assert got.shape == E2E_SHAPE
            assert np.all(np.isfinite(got))
            worst = max(worst, float(np.abs(got - expected).max() / expected))
        assert_within(worst, GND_E2E_TOL, "GND_E2E_TOL")

    def test_the_recovered_density_matches_both_contrast_routes(self):
        # TWO contrasts, and the comment now says which is which
        # (corrected 2026-09-08, Stage C adversarial review: the arm
        # below was presented as having "no finite difference in it"
        # and as therefore independent of the module, while it calls
        # nye_tensor and gnd_density on BOTH sides).
        #
        # 1. the SHARED-HELPER route.  The analytically imposed field
        #    is pushed through this module's own curl and estimator.
        #    What it isolates is the ENGINE -- the recovered Fe against
        #    the imposed Fe -- with the GND math held fixed, and any
        #    bug shared by the two sides cancels here by construction.
        # 2. the INDEPENDENT route, expected_alpha_through_the_chain
        #    plus density_by_hand: neither calls the module and neither
        #    takes a finite difference, so this arm IS a statement
        #    about the curl.  It carries the frozen V7 per-cent bound,
        #    and while GND_E2E_TOL is an unfilled placeholder it is the
        #    only effective end-to-end numeric bound in the file
        detector, xmap = curvature_run()
        through_patterns = hrebsd_gnd(xmap, detector, BURGERS_SI)
        independent = density_by_hand(
            expected_alpha_through_the_chain(E2E_P, E2E_Q, detector, fix="detector"),
            BURGERS_SI,
            "a5",
        )
        np.testing.assert_allclose(
            through_patterns,
            np.full(E2E_SHAPE, independent),
            rtol=ORACLE_REL_TOL,
            atol=0,
        )
        by_algebra = gnd_density(
            nye_tensor(
                np.stack(
                    [
                        deviatoric(
                            to_sample_frame(
                                antisymmetry_by_hand(tensor),
                                sample_to_spec_detector_matrix(detector),
                            )
                        )
                        for tensor in linear_beta_field(
                            E2E_P,
                            E2E_Q,
                            shape=E2E_SHAPE,
                            step_x1=E2E_STEP_M,
                            step_x2=E2E_STEP_M,
                        ).reshape(-1, 3, 3)
                    ]
                ).reshape(E2E_SHAPE + (3, 3)),
                E2E_STEP_M,
                E2E_STEP_M,
            ),
            BURGERS_SI,
            "a5",
        )
        # the D14.3 fix is applied on this side too, so the two routes
        # are the same operator rather than differing by it.  It is an
        # exact no-op on E2E_P and E2E_Q, which are already
        # antisymmetric in the 31 and 32 slots, and writing it out
        # keeps that a property of the DATA and not of the comparison
        assert np.all(np.isfinite(by_algebra))
        np.testing.assert_allclose(
            through_patterns, by_algebra, rtol=ORACLE_REL_TOL, atol=0
        )
