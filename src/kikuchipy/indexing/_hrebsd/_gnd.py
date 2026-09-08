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

"""Scalar geometrically necessary dislocation (GND) densities.

The Nye tensor of the measured elastic distortion field and the three
scalar L1 estimators built on it (requirements D14).  Everything here
is frozen against the DERIVATION and against the constant-curvature
oracle of validation V7, never against another code's sign.

**The convention, frozen (requirements D14.1).**  ``alpha =
curl(beta_e)`` in the index form

.. code-block::

    alpha_ij = eps_jkl d(beta_il)/d(x_k)

with ``x1`` the map COLUMN axis and ``x2`` the map ROW axis, the same
identification requirements D12 makes for the HR-KAM identity.  A
surface map carries in-plane derivatives only, so ``k`` runs over 1
and 2 and the ``k = 3`` slot is IDENTICALLY ABSENT -- never estimated,
never stood in for by a beta value.  Expanding the two surviving terms
of each ``j``:

.. code-block::

    alpha_i1 = +d(beta_i3)/dx2   - d(beta_i2)/dx3   <- dropped
    alpha_i2 = -d(beta_i3)/dx1   + d(beta_i1)/dx3   <- dropped
    alpha_i3 = +d(beta_i2)/dx1   - d(beta_i1)/dx2

**Assumption tiers (requirements D14.2).**  The three ``alpha_i3`` are
EXACT: both their terms are in-plane.  The six ``alpha_i1`` and
``alpha_i2`` survive only under the d/dx3-neglect of OpenXY's beta
route (``DislocationDensityCalculate.m:239, 324-336``, where only the
in-plane derivative slots are ever filled), and they are what the
``"a5"`` and ``"a9"`` estimators add.  Pantleon's curvature route,
which supplies six knowns rather than nine, is recorded context in
requirements D14.2 and is NOT built: the D14.4 prefactors are
calibrated for this construction.

**Enforced antisymmetry (requirements D14.3).**  ``beta31`` and
``beta32`` are about 9.6 times noisier than ``beta13`` and ``beta23``
for typical geometry and dominate the GND noise, so the default
replaces them by ``-beta13`` and ``-beta23``.  The replacement is
defined in the DETECTOR frame, which is why this module recomputes the
tensor chain from the stored ``Fe`` rather than reading the
``"beta"`` property of a Stage B map: rotating first and fixing
afterwards is a different operator whenever the frame matrix is not
diagonal.  The fix lives on THIS path only and never touches a strain
or a stress (frozen).

It is NOT a filter, and calling it a noise reduction alone would
misdescribe it (stated 2026-09-08, Stage C fix gate).  The replacement
makes the lower left of the detector-frame tensor exactly minus its
upper right, so the two detector-frame out-of-plane elastic shear
strains ``eps13 = (beta13 + beta31) / 2`` and ``eps23`` become
IDENTICALLY ZERO: a measured elastic shear is traded away for a lower
noise operator.  Measured on the constant-field tutorial oracle at
this gate, both fall from 3.5e-04 and 5.5e-04 to 0.0 exactly, the
sample-frame field moves by up to 9.7e-04 against its own 1.4e-03
amplitude, and the ``"a5"`` density of that analytically imposed field
rises to 2.3972e12 m^-2, 8.0 per cent above the 2.2192e12 m^-2 of its
own Nye content.  The default is kept on the Ruggles 2020
geometric-noise argument, and the trade is now documented rather than
implied.

**Units (requirements D14.5, D14.6).**  Gradients are per METRE, with
the map step converted from
:attr:`~orix.crystal_map.CrystalMap.scan_unit`; a missing or
unrecognized unit, ``"px"`` included, is a ``ValueError`` and never a
guess.  ``burgers_vector_length`` is in METRES and has no default, for
the same reason.  The returned density is in ``m^-2`` and its expected
noise floor scale is ``rho_noise ~ sigma_beta / (b * step)``, about
4e12 to 8e12 m^-2 at ``sigma_beta = 1e-4``, ``b = 0.25 nm`` and a
100 nm step (Jiang, Britton and Wilkinson 2013; Ernould's thesis).

**The one measured floor, and how to read it (requirements D14.6;
recorded 2026-09-08, validation entry 67).**  On
:func:`~kikuchipy.data.si_wafer`, every fifth point of both axes so a
200 um step, ``b = 3.84e-10`` m and the default ``"a5"``, the median
density over the 66 finite points of 100 is **1.1237e11 m^-2**.  That
is 36 to 71 times BELOW the literature class above, and reading it as
a better floor would be exactly backwards: the class is quoted at a
100 nm step, this one is measured at a step two thousand times longer,
and a curvature is a distortion divided by a distance.  The same
identity fed that dataset's own rotation floor of 1.2006e-02 rad
predicts 1.5633e11 m^-2, so the measurement is consistent with being
noise and nothing else.  It is THAT DATASET's floor, acquired for
projection-centre calibration rather than for HR-EBSD, and never the
method's.

References
----------
Nye, Acta Metall. 1 (1953) 153; Pantleon, Scripta Mater. 58 (2008)
994; Ruggles et al., Ultramicroscopy 210 (2020) 112927; Jiang, Britton
and Wilkinson, Ultramicroscopy 125 (2013) 1.  OpenXY's
``DislocationDensityCalculate.m`` is the equation-level
cross-reference for the estimator prefactors and their consumption
sets; no code is ported.
"""

import numpy as np

from kikuchipy.indexing._hrebsd._segmentation import map_grids
from kikuchipy.indexing._hrebsd._tensors import (
    _as_stack,
    _fe_as_matrices,
    best_orientation_matrices,
    single_phase_stiffness_guard,
    tensor_chain,
)

# The scalar estimators of requirements D14.4, in the order their
# consumption sets grow
SUPPORTED_ESTIMATORS: tuple[str, ...] = ("a3", "a5", "a9")

# The OpenXY L1-extrapolation constants of requirements D14.4, written
# as the literal ratios they are documented as.  They are an ESTIMATE
# and not a measurement, and each is meaningful only with the matching
# consumption set below: swapping two of them is the plan section 4.3
# mutant that ``test_hrebsd_gnd.py::TestPrefactors`` kills
ESTIMATOR_PREFACTORS: dict[str, float] = {
    "a3": 30.0 / 10.0,
    "a5": 30.0 / 14.0,
    "a9": 30.0 / 20.0,
}

# Which entries of the Nye tensor each estimator sums the modulus of,
# as zero-based ``(i, j)`` index pairs (requirements D14.4).  The SET
# is what is frozen; the order only fixes the summation order:
#
# - ``"a3"`` takes the three EXACT ``alpha_i3`` of requirements D14.1
#   and nothing else,
# - ``"a5"`` adds exactly ``alpha_12`` and ``alpha_21``, the two
#   d/dx3-neglect entries whose neglect Pantleon's analysis
#   independently supports,
# - ``"a9"`` takes all nine of the same construction.
ESTIMATOR_COMPONENTS: dict[str, tuple[tuple[int, int], ...]] = {
    "a3": ((0, 2), (1, 2), (2, 2)),
    "a5": ((0, 2), (1, 2), (2, 2), (0, 1), (1, 0)),
    "a9": tuple((i, j) for i in range(3) for j in range(3)),
}

# The map step units requirements D14.5 accepts and their conversion
# to metres.  Both micro signs are here, U+00B5 and U+03BC, because
# kikuchipy writes the scan unit from the signal axes and either can
# arrive.  Anything else -- ``"px"``, ``"mm"``, an empty string, None
# -- raises rather than being guessed
SCAN_UNIT_TO_METERS: dict[str, float] = {
    "m": 1.0,
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
}

# The properties a map must carry to have a GND density: the reduced
# detector-frame deformation gradients of the engine, which the chain
# is recomputed from, and the grain identifiers, without which a
# gradient across a grain boundary could not be refused
REQUIRED_PROP_NAMES: tuple[str, ...] = ("Fe", "grain_id")

# The replacements the enforced antisymmetry of requirements D14.3
# makes, as ``(target, source)`` index pairs: ``beta31 <- -beta13`` and
# ``beta32 <- -beta23``.  The direction is not symmetric and the
# reversed form is the plan section 4.3 mutant
ANTISYMMETRY_REPLACEMENTS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((2, 0), (0, 2)),
    ((2, 1), (1, 2)),
)

# Entries of one stored deformation gradient or displacement gradient
BETA_PROP_SIZE: int = 9

# The grain identifier of a point outside every grain
UNLABELLED: int = -1


def scan_step_meters(xmap) -> tuple[float, float]:
    """Return the map steps along ``x1`` and ``x2`` in METRES.

    The step sizes of requirements D14.5, read from the crystal map's
    own coordinate arrays and converted with its
    :attr:`~orix.crystal_map.CrystalMap.scan_unit`.  ``x1`` is the
    COLUMN axis and ``x2`` the ROW axis, the identification
    requirements D12 uses for the constant-curvature identity.

    Parameters
    ----------
    xmap
        :class:`~orix.crystal_map.CrystalMap` whose points lie on a
        scan grid.

    Returns
    -------
    step_x1, step_x2
        The column-axis and row-axis steps in metres. An axis the map
        does not span, a single column or a single row, reports
        ``nan``: no derivative along it exists, so every quantity that
        would need one is ``nan`` anyway, and a fabricated step would
        hide that.

    Raises
    ------
    ValueError
        If the map has no grid, which is what orix reports as the
        shape ``()``; or if
        :attr:`~orix.crystal_map.CrystalMap.scan_unit` is missing or
        is not one of :data:`SCAN_UNIT_TO_METERS`. The unmapped
        default ``"px"`` lands there on purpose (requirements D14.5):
        a pixel is not a length, and guessing micrometres is exactly
        the unit-guessing the ``burgers_vector_length`` rule forbids.
    """
    # The two dimensional shape comes from the grids and never from
    # ``CrystalMap.shape``, which orix flattens to ``(n,)`` for a map
    # one point wide or one point tall; ``map_grids`` is also where a
    # map with no grid at all is named
    rows, cols = map_grids(xmap)
    unit = getattr(xmap, "scan_unit", None)
    if unit not in SCAN_UNIT_TO_METERS:
        raise ValueError(
            f"the scan_unit of xmap is {unit!r}, which is not one of "
            f"{sorted(SCAN_UNIT_TO_METERS)}: a gradient per metre needs a length, "
            "and orix's own default 'px' is a pixel count rather than one, so it "
            "is refused rather than guessed (requirements D14.5)"
        )
    factor = SCAN_UNIT_TO_METERS[unit]
    # orix reads its step off the two smallest unique coordinates, so
    # it is already positive on a DESCENDING grid; the modulus states
    # that a step is a length rather than relying on that
    step_x1 = abs(float(xmap.dx)) * factor if int(cols.max()) > 0 else np.nan
    step_x2 = abs(float(xmap.dy)) * factor if int(rows.max()) > 0 else np.nan
    return step_x1, step_x2


def enforce_beta_antisymmetry(beta_detector: np.ndarray) -> np.ndarray:
    """Return a detector-frame tensor with its lower row replaced.

    The noise fix of requirements D14.3: ``beta31 <- -beta13`` and
    ``beta32 <- -beta23``, in the DETECTOR frame and on the GND path
    only.  The two replaced entries are about 9.6 times noisier than
    the two they are replaced by for typical geometry, and the fix
    reduces the GND noise-operator 2-norm by about 2.8 times.

    Parameters
    ----------
    beta_detector
        Array of shape ``(3, 3)`` or ``(n, 3, 3)`` in the detector
        frame, that is ``Fe - I`` before any rotation.

    Returns
    -------
    beta_fixed
        A NEW array of the same shape and 64-bit float data type; the
        input is never modified in place, because the caller's ``Fe``
        property is a public array.

    Raises
    ------
    ValueError
        If *beta_detector* does not have two trailing axes of length
        three.

    Notes
    -----
    What the replacement DOES, beside lowering the noise: it makes the
    lower left of the tensor exactly minus its upper right, so the two
    detector-frame out-of-plane elastic shear strains
    ``eps13 = (beta13 + beta31) / 2`` and ``eps23`` are identically
    zero afterwards. A measured elastic shear is traded for a quieter
    noise operator, so this is not a filter (recorded 2026-09-08,
    Stage C fix gate).

    The direction matters and is not symmetric: replacing ``beta13``
    and ``beta23`` by ``-beta31`` and ``-beta32`` instead keeps the
    noisy pair and discards the quiet one, which is the backwards
    mutant of plan section 4.3.

    A full projection onto a pure rotation, keeping only the polar
    ``R``, is explicitly NOT offered: Ruggles 2020 shows that
    discarding the elastic-strain derivatives corrupts the GND
    identification (requirements D14.3).
    """
    stacked, single = _as_stack(beta_detector, "beta_detector")
    # ``_as_stack`` hands back a VIEW when the caller's array is already
    # 64-bit float, so the copy is what keeps the public ``Fe`` property
    # of the caller's map untouched; the sources are read from the
    # original for the same reason
    fixed = stacked.copy()
    for (target_i, target_j), (source_i, source_j) in ANTISYMMETRY_REPLACEMENTS:
        fixed[:, target_i, target_j] = -stacked[:, source_i, source_j]
    return fixed[0] if single else fixed


def in_plane_gradients(
    field: np.ndarray,
    step_x1: float,
    step_x2: float,
    *,
    grain: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two in-plane derivatives of a map-shaped field.

    Central differences on the map grid, one-sided at the map EDGES
    only (requirements D14.5).  There is no third derivative and no
    third return value: the ``d/dx3`` slot of the curl is absent by
    construction, never estimated.

    Parameters
    ----------
    field
        Array of shape ``(ny, nx, ...)``, the quantity to
        differentiate, with the map grid on the two leading axes.
    step_x1
        Step along the COLUMN axis in metres. May be ``nan`` for a map
        one column wide, in which case every ``d/dx1`` is ``nan``.
    step_x2
        Step along the ROW axis in metres, ``nan`` for a map one row
        tall.
    grain
        Grain identifiers of shape ``(ny, nx)``, with
        :data:`UNLABELLED` outside every grain, or ``None`` for one
        grain everywhere.

    Returns
    -------
    d_dx1, d_dx2
        Arrays of the shape of *field* and 64-bit float data type.

    Raises
    ------
    ValueError
        If *field* has fewer than two axes, or if *grain* is given and
        does not have the map shape of *field*.

    Notes
    -----
    The NaN rule of requirements D14.5 is frozen here, and it never
    fabricates a value:

    - a point outside every grain has ``nan`` in both derivatives;
    - a point whose OWN value is non-finite has ``nan`` in both
      derivatives, whatever its neighbours hold. This is the SELF
      rule, and it is requirements D2.6 rather than D14.5: a
      non-converged, failed or masked point "get[s] NaN in every
      derived prop downstream" and is never zeroed. It has to be
      stated because a central-difference pair never reads its own
      centre, so the pair rule below would leave such a point holding
      the full density of its neighbourhood (clarified 2026-09-08,
      Stage C adversarial review; ``test_hrebsd_gnd.py::TestNaNSafety
      ::test_a_non_finite_point_is_nan_in_alpha_itself`` pins it at
      this level and not only at :func:`hrebsd_gnd`);
    - an INTERIOR point uses its central pair, and gets ``nan`` when
      that pair crosses a grain boundary or touches a non-finite
      value. It does NOT silently fall back on a one-sided
      difference, which would be a different estimator reported under
      the same name;
    - a point on the map EDGE has no central pair and uses the one
      one-sided pair it has, under the same two conditions.

    The pair rule and the self rule are SEPARATE, and a neighbour of a
    hole loses only the derivative whose pair reaches it: at a point
    one step to the left of a non-finite one the ``d/dx1`` pair is
    ``nan`` while every ``d/dx2`` entry survives.

    Both rules are read PER MAP POINT and not per trailing component: a
    map point holding a single non-finite entry anywhere in its
    trailing block is non-finite as a point, so the whole block of both
    its derivatives is ``nan``. That is requirements D2.6 read the
    conservative way round -- a point which did not converge carries
    NaN in every derived quantity -- and on the documented input it
    changes nothing, because the engine writes a whole ``Fe`` row of
    NaN or none of it.
    """
    array = np.asarray(field, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(
            "field must carry the map grid on its two leading axes, that is shape "
            f"(ny, nx, ...), but has shape {array.shape}"
        )
    map_shape = array.shape[:2]
    if grain is None:
        # one grain everywhere, so no pair ever crosses a boundary
        labels = np.zeros(map_shape, dtype=np.int64)
    else:
        labels = np.asarray(grain)
        if labels.shape != map_shape:
            raise ValueError(
                "grain must hold one identifier per map point, that is shape "
                f"{map_shape} for this field, but has shape {labels.shape}"
            )
        # A NaN in a FLOAT identifier array casts to an undefined
        # integer, so it becomes the unlabelled sentinel first rather
        # than being left to the platform, as the HR-KAM path does
        labels = np.where(np.isfinite(labels), labels, UNLABELLED).astype(np.int64)
    finite = np.isfinite(array)
    for _ in range(array.ndim - 2):
        finite = finite.all(axis=-1)
    # THE self rule of requirements D2.6 and the labelling half of the
    # requirements D14.5 pair rule, in one map-shaped mask: a point
    # which is not usable is refused as a centre AND as a neighbour
    usable = finite & (labels >= 0)
    return (
        _axis_derivative(array, step_x1, 1, usable, labels),
        _axis_derivative(array, step_x2, 0, usable, labels),
    )


def _axis_derivative(
    field: np.ndarray,
    step: float,
    axis: int,
    usable: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return one in-plane derivative of a map-shaped field.

    Central differences in the interior and one-sided differences at
    the two map edges, with the NaN rule of requirements D14.5 applied
    to every stencil: a stencil is used only where every point it
    reads, the CENTRE included, is usable and carries the grain
    identifier of the centre.

    Parameters
    ----------
    field
        Array of shape ``(ny, nx, ...)`` and 64-bit float data type.
    step
        Step along *axis* in metres, possibly ``nan``.
    axis
        0 for the ROW axis ``x2``, 1 for the COLUMN axis ``x1``.
    usable
        Map-shaped mask of the points a stencil may read.
    labels
        Map-shaped grain identifiers.

    Returns
    -------
    derivative
        Array of the shape of *field* and 64-bit float data type.

    Notes
    -----
    The centre is part of every grain comparison, which is what makes a
    ONE-POINT grain ``nan``: its two neighbours share a grain with each
    other but not with it, and a pair-only comparison would hand it the
    full density of the grain surrounding it.

    An axis the map does not span carries no stencil at all and stays
    ``nan``; no one-sided difference is invented for it.
    """
    derivative = np.full(field.shape, np.nan, dtype=np.float64)
    length = field.shape[axis]
    if length < 2:
        return derivative
    # views, so writing through ``out`` writes into ``derivative``
    work = np.moveaxis(field, axis, 0)
    out = np.moveaxis(derivative, axis, 0)
    use = np.moveaxis(usable, axis, 0)
    label = np.moveaxis(labels, axis, 0)
    trailing = (1,) * (field.ndim - 2)

    def expand(mask: np.ndarray) -> np.ndarray:
        return mask.reshape(mask.shape + trailing)

    # a non-finite ``step`` or a non-finite neighbour is arithmetic and
    # not an error, and neither is ever reported: the mask has already
    # decided which stencils are read
    with np.errstate(invalid="ignore", divide="ignore"):
        if length > 2:
            valid = (
                use[1:-1]
                & use[2:]
                & use[:-2]
                & (label[1:-1] == label[2:])
                & (label[1:-1] == label[:-2])
            )
            central = (work[2:] - work[:-2]) / (2.0 * step)
            out[1:-1] = np.where(expand(valid), central, np.nan)
        # the two map EDGES, each with the one one-sided pair it has.
        # An interior point NEVER falls back on one of these: a
        # one-sided difference reported under the same name would be a
        # different estimator (requirements D14.5)
        for edge, first, second in ((0, 0, 1), (length - 1, length - 2, length - 1)):
            valid = use[first] & use[second] & (label[first] == label[second])
            one_sided = (work[second] - work[first]) / step
            out[edge] = np.where(expand(valid), one_sided, np.nan)
    return derivative


def nye_tensor(
    beta: np.ndarray,
    step_x1: float,
    step_x2: float,
    *,
    grain: np.ndarray | None = None,
) -> np.ndarray:
    """Return the Nye dislocation density tensor of a beta field.

    The convention of requirements D14.1 with the d/dx3-neglect
    extension of D14.2, written out in the module docstring:

    .. code-block::

        alpha_i1 = +d(beta_i3)/dx2
        alpha_i2 = -d(beta_i3)/dx1
        alpha_i3 = +d(beta_i2)/dx1 - d(beta_i1)/dx2

    The first two lines are the surviving in-plane halves of the exact
    curl and hold only where elastic-strain gradients along ``x3``
    vanish; the third is EXACT on a surface map.

    Parameters
    ----------
    beta
        Elastic distortion field of shape ``(ny, nx, 3, 3)``, in the
        SAMPLE frame, closed in its ninth degree of freedom.
    step_x1
        Step along the COLUMN axis in metres.
    step_x2
        Step along the ROW axis in metres.
    grain
        Grain identifiers of shape ``(ny, nx)`` or ``None``, passed
        through to :func:`in_plane_gradients`.

    Returns
    -------
    alpha
        Array of shape ``(ny, nx, 3, 3)`` and 64-bit float data type
        in ``m^-1``, with ``nan`` wherever the derivative it needs is
        ``nan``.

    Raises
    ------
    ValueError
        If *beta* is not ``(ny, nx, 3, 3)``, or if *grain* does not
        match its map shape.

    Notes
    -----
    Only the in-plane derivative slots of *beta* are ever read, which
    is a structural claim: adding a CONSTANT to every entry of *beta*
    leaves ``alpha`` unchanged to ROUNDING, and
    ``test_hrebsd_gnd.py::TestNoFabricatedDerivative`` asserts that on
    a field where any value-based stand-in for a ``d/dx3`` term would
    move the answer by orders of magnitude.

    Not BITWISE unchanged, and an earlier draft of this docstring said
    so (corrected 2026-09-08, Stage C adversarial review):
    ``(c + a) - (c + b)`` need not round to the same double as
    ``a - b``, so no correct implementation can be bitwise invariant
    under an offset, and :func:`numpy.gradient` is not either.  The
    invariance is real; its band is machine precision, not zero.
    """
    field = np.asarray(beta, dtype=np.float64)
    if field.ndim != 4 or field.shape[-2:] != (3, 3):
        raise ValueError(
            "beta must be the map-shaped elastic distortion field of shape "
            f"(ny, nx, 3, 3), but has shape {field.shape}"
        )
    d_dx1, d_dx2 = in_plane_gradients(field, step_x1, step_x2, grain=grain)
    alpha = np.empty(field.shape, dtype=np.float64)
    # Only the third COLUMN of beta reaches the first two columns of
    # alpha and only its first two reach the third, which is the tier
    # boundary of requirements D14.2 written as an assignment: no slot
    # here reads a beta VALUE, so no d/dx3 term can be fabricated
    alpha[..., 0] = d_dx2[..., 2]
    alpha[..., 1] = -d_dx1[..., 2]
    alpha[..., 2] = d_dx1[..., 1] - d_dx2[..., 0]
    return alpha


def gnd_density(
    alpha: np.ndarray,
    burgers_vector_length: float,
    estimator: str = "a5",
) -> np.ndarray:
    """Return the scalar GND density of a Nye tensor field.

    The L1 estimators of requirements D14.4::

        rho = prefactor * sum over the consumption set of |alpha_ij|
              / burgers_vector_length

    with the prefactor of :data:`ESTIMATOR_PREFACTORS` and the set of
    :data:`ESTIMATOR_COMPONENTS`.  The prefactors are OpenXY's L1
    extrapolation constants, an ESTIMATE and not a measurement, and
    each is meaningful only with its own set.

    Parameters
    ----------
    alpha
        Nye tensor of shape ``(..., 3, 3)`` in ``m^-1``.
    burgers_vector_length
        Burgers vector length in METRES.
    estimator
        One of :data:`SUPPORTED_ESTIMATORS`. Default is ``"a5"``.

    Returns
    -------
    density
        Array of the leading shape of *alpha* and 64-bit float data
        type in ``m^-2``, ``nan`` wherever a consumed entry is
        ``nan``. It is a sum of moduli and is therefore never
        negative.

    Raises
    ------
    ValueError
        If *alpha* does not have two trailing axes of length three; if
        *estimator* is not one of :data:`SUPPORTED_ESTIMATORS`; or if
        *burgers_vector_length* is not a positive finite number of
        METRES, which the message says in full: a Burgers vector given
        in nanometres puts every density out by nine orders and is
        indistinguishable from a correct one by inspection.
    """
    tensor = np.asarray(alpha, dtype=np.float64)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError(
            "alpha must have two trailing axes of length three, that is shape "
            f"(3, 3) or (..., 3, 3), but has shape {tensor.shape}"
        )
    if estimator not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"estimator {estimator!r} is not one of {list(SUPPORTED_ESTIMATORS)}"
        )
    length = _burgers_vector_length_in_meters(burgers_vector_length)
    components = ESTIMATOR_COMPONENTS[estimator]
    # the prefactor and the set are read TOGETHER from the two frozen
    # tables: each constant is an L1 extrapolation calibrated for its
    # own consumption set and means nothing beside another one
    total = np.abs(tensor[..., components[0][0], components[0][1]])
    for i, j in components[1:]:
        total = total + np.abs(tensor[..., i, j])
    return ESTIMATOR_PREFACTORS[estimator] * total / length


def _burgers_vector_length_in_meters(burgers_vector_length: float) -> float:
    """Return a Burgers vector length after checking it is one.

    Parameters
    ----------
    burgers_vector_length
        The caller's value, in METRES.

    Returns
    -------
    length
        The same number as a 64-bit float.

    Raises
    ------
    ValueError
        If it is not a positive finite number. The message states the
        UNIT, because nothing downstream can catch a value handed over
        in nanometres: the density scales as ``1 / b``, so such a value
        is wrong by nine orders and looks entirely reasonable in the
        source (requirements D14.5).
    """
    length = float(burgers_vector_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(
            f"burgers_vector_length {burgers_vector_length!r} must be a positive "
            "finite length in METRES, for instance 2.49e-10 for nickel; the density "
            "scales as 1 / b, so a value given in nanometres is wrong by nine "
            "orders of magnitude"
        )
    return length


def log10_density(density: np.ndarray) -> np.ndarray:
    """Return the base-ten logarithm of a GND density map, guarded.

    The plotting recipe of requirements D14.6 as executable code, so
    that the guard has a place where it is TESTED rather than only
    described in prose.  This module is private, so the tutorial
    cannot import it and writes the same expression out inline
    (corrected 2026-09-08, Stage C fix gate: an earlier draft said the
    helper existed so that the tutorial need not repeat it, which it
    is not free to do).  A density is a sum of moduli and so is never
    negative, but it IS exactly zero on a strain-free synthetic map,
    where a bare :func:`numpy.log10` returns ``-inf`` and warns; and a
    caller may hand this any array at all.

    Parameters
    ----------
    density
        Array of any shape, in ``m^-2``.

    Returns
    -------
    log10
        Array of the same shape and 64-bit float data type: the
        base-ten logarithm where *density* is strictly positive and
        ``nan`` everywhere else, that is at zero, at a negative value
        and at ``nan``. No warning is emitted for any input.
    """
    values = np.asarray(density, dtype=np.float64)
    positive = values > 0.0
    # the logarithm never SEES a non-positive value, which is how this
    # stays warning free rather than by suppressing a warning: a bare
    # log10 returns -inf at zero and warns, and on a curvature-free map
    # that is every point
    return np.where(positive, np.log10(np.where(positive, values, 1.0)), np.nan)


def hrebsd_gnd(
    xmap,
    detector,
    burgers_vector_length: float,
    *,
    stiffness: np.ndarray | None = None,
    estimator: str = "a5",
    enforce_antisymmetry: bool = True,
) -> np.ndarray:
    """Return the scalar geometrically necessary dislocation density.

    The Nye tensor of the measured elastic distortion field, reduced
    to one scalar per map point by an L1 estimator: the density of
    dislocations that the lattice curvature REQUIRES, as opposed to
    the statistically stored ones a curvature measurement cannot see.

    Parameters
    ----------
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map returned by
        :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`, carrying its
        ``"Fe"`` and ``"grain_id"`` properties and a
        :attr:`~orix.crystal_map.CrystalMap.scan_unit` of ``"m"``,
        ``"nm"`` or ``"um"``.
    detector : ~kikuchipy.detectors.EBSDDetector
        The detector the patterns were measured with, supplying the
        detector to sample frame chain. Nothing is hardcoded.
    burgers_vector_length : float
        Burgers vector length in METRES, for instance ``2.49e-10`` for
        nickel and ``3.84e-10`` for silicon. Required, with no
        default: the density scales as ``1 / b``, so a value given in
        nanometres is wrong by nine orders and looks perfectly
        reasonable in the source. Deriving it from the phase
        structure is a recorded version two nicety.
    stiffness : numpy.ndarray, optional
        Elastic stiffness as a 6 by 6 Voigt matrix in GPa in the
        CRYSTAL frame, as
        :func:`~kikuchipy.indexing.hrebsd_strain_stress` takes it. It
        selects the traction-free closure of the ninth degree of
        freedom; without it the deviatoric closure is used. No stress
        is computed here either way.
    estimator : str, optional
        Which entries of the Nye tensor the density sums, and with
        which prefactor (requirements D14.4). ``"a3"`` (prefactor
        30/10) takes only the three EXACT ``alpha_i3``; ``"a5"``
        (default, 30/14) adds ``alpha_12`` and ``alpha_21``, the two
        entries whose neglect of the out-of-plane derivatives
        Pantleon's analysis independently supports
        :cite:`pantleon2008resolving`; ``"a9"`` (30/20) takes all
        nine. The six entries beyond ``alpha_i3`` rest on neglecting
        the out-of-plane derivatives, so ``"a3"`` is the
        assumption-free choice and the other two trade assumptions
        for sensitivity.
    enforce_antisymmetry : bool, optional
        Whether to replace ``beta31`` and ``beta32`` by ``-beta13``
        and ``-beta23`` in the DETECTOR frame before rotating
        (requirements D14.3). Default is True: those two entries are
        about 9.6 times noisier than the pair replacing them for
        typical geometry and otherwise dominate the result
        :cite:`ruggles2020correlating`. It applies to this function
        alone and never changes a strain or a stress.

        It is a TRADE and not a filter. The replacement makes the
        lower left of the detector-frame tensor exactly minus its
        upper right, so the detector-frame out-of-plane elastic shear
        strains ``eps13`` and ``eps23`` become identically zero. On
        an analytically imposed constant-curvature field it moves the
        ``"a5"`` density 8.0 per cent away from the field's own Nye
        content, and on :func:`~kikuchipy.data.si_wafer` it RAISES
        the measured floor rather than lowering it, 1.1237e11 against
        7.7823e10 m^-2, because that dataset's floor is set by a
        band-pass-surviving fixed pattern component and not by
        ``beta31`` and ``beta32`` noise. The default rests on the
        geometric-noise argument above, which those two measurements
        do not test (measured 2026-09-08, validation entries 67 and
        74; requirements D14.3).

    Returns
    -------
    density
        Array of the map's navigation shape ``(ny, nx)`` and 64-bit
        float data type in ``m^-2``. Points whose gradient cannot be
        formed are ``nan``: a point outside every grain, a point in a
        one-point grain, a point whose ``Fe`` is ``nan`` because the
        fit did not converge or was masked out, and an interior point
        whose central-difference pair crosses a grain boundary or
        touches such a point. Nothing is ever zero-filled. The shape
        comes from the map's row and column grids, so a map of one row
        returns ``(1, nx)``, as
        :func:`~kikuchipy.indexing.hrebsd_kam` does.

    Raises
    ------
    ValueError
        If *xmap* does not carry both ``"Fe"`` and ``"grain_id"``,
        naming :meth:`~kikuchipy.signals.EBSD.hrebsd_dic`; if it has
        no map grid, which is what orix reports as the shape ``()``;
        if its :attr:`~orix.crystal_map.CrystalMap.scan_unit` is
        missing or unrecognized, the ``"px"`` default included; if
        *burgers_vector_length* is not a positive finite number; if
        *estimator* is not one of ``"a3"``, ``"a5"`` or ``"a9"``; if
        *stiffness* is not a 6 by 6 matrix; or if a *stiffness* is
        given for a map holding more than one phase.

    Notes
    -----
    Frames and units. The density is in ``m^-2`` and the map step is
    converted to metres from the crystal map's own ``scan_unit``, so a
    map still carrying orix's ``"px"`` default is refused rather than
    assumed to be micrometres.

    The tensor chain is recomputed here from the stored
    detector-frame ``"Fe"``, through the same private chain
    :func:`~kikuchipy.indexing.hrebsd_strain_stress` uses, rather than
    reading a ``"beta"`` property: the antisymmetry fix is DEFINED in
    the detector frame, and applying it after the rotation is a
    different operator.

    What this measures. Only the lattice curvature the map RESOLVES
    contributes, so the density depends on the step size and is a
    lower bound on the true dislocation content; the estimators
    themselves are L1 lower bounds on the density consistent with the
    measured Nye tensor. The expected noise floor scale is
    ``rho_noise ~ sigma_beta / (b * step)``, about 4e12 to 8e12
    ``m^-2`` at ``sigma_beta = 1e-4``, ``b = 0.25 nm`` and a 100 nm
    step :cite:`jiang2013measurement`.

    The one floor kikuchipy has MEASURED, and it must be read beside
    that scale rather than against it. On
    :func:`~kikuchipy.data.si_wafer`, every fifth point of both axes
    so a 200 um step, ``b = 3.84e-10`` m and the default ``"a5"``,
    the median over the 66 finite points of 100 is 1.1237e11
    ``m^-2``. That is 36 to 71 times BELOW the literature class, and
    it is not a better floor: the class is quoted at a 100 nm step,
    this one at a step two thousand times longer, and a curvature is
    a distortion divided by a distance. The identity above, fed that
    dataset's own rotation floor of 1.2006e-02 rad and its own step,
    predicts 1.5633e11 ``m^-2``, so the number is consistent with
    being noise and nothing else. It is that dataset's floor, from a
    scan acquired for projection-centre calibration rather than for
    HR-EBSD, and never the method's (measured 2026-09-08, validation
    entry 67).

    A per-slip-system L1 split, which would report a density per
    dislocation type, needs a slip-system catalog and is deferred to a
    future version; version one ships these scalar estimators.

    Maps are plotted on a base-ten logarithmic scale, and the
    logarithm has to be guarded: a density is never negative, being a
    sum of moduli, but it IS exactly zero on a curvature-free map,
    where a bare :func:`numpy.log10` warns and returns ``-inf``. Take
    it as ``numpy.log10(numpy.where(rho > 0, rho, numpy.nan))``, which
    is what this feature's tutorial does.

    See Also
    --------
    kikuchipy.signals.EBSD.hrebsd_dic
    kikuchipy.indexing.hrebsd_strain_stress
    kikuchipy.indexing.hrebsd_kam
    """
    missing = [name for name in REQUIRED_PROP_NAMES if name not in xmap.prop]
    if missing:
        raise ValueError(
            f"xmap does not carry the {missing} property this function needs, so no "
            "gradient can be formed; pass the crystal map EBSD.hrebsd_dic returned"
        )
    if estimator not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"estimator {estimator!r} is not one of {list(SUPPORTED_ESTIMATORS)}"
        )
    length = _burgers_vector_length_in_meters(burgers_vector_length)
    step_x1, step_x2 = scan_step_meters(xmap)
    single_phase_stiffness_guard(xmap, stiffness)

    # The chain is recomputed from the stored DETECTOR-frame Fe rather
    # than read off a "beta" property, because the antisymmetry fix is
    # DEFINED in that frame and applying it after the rotation is a
    # different operator whenever the frame matrix is not diagonal
    # (requirements D14.3).  Everything after the fix is the ONE shared
    # chain of plan section 3.2, not a second copy of it
    beta_detector = _fe_as_matrices(np.asarray(xmap.prop["Fe"])) - np.eye(3)
    if enforce_antisymmetry:
        beta_detector = enforce_beta_antisymmetry(beta_detector)
    properties = tensor_chain(
        np.eye(3) + beta_detector,
        best_orientation_matrices(xmap),
        detector,
        stiffness=stiffness,
    )

    rows, cols = map_grids(xmap)
    ny = int(rows.max()) + 1
    nx = int(cols.max()) + 1
    size = rows.size
    # A grid position the map has no point at keeps NaN and the
    # unlabelled sentinel, so it is refused as a stencil neighbour in
    # exactly the way a non-converged point is
    field = np.full((ny, nx, 3, 3), np.nan, dtype=np.float64)
    field[rows, cols] = np.asarray(properties["beta"], dtype=np.float64).reshape(
        size, 3, 3
    )
    grain = np.full((ny, nx), UNLABELLED, dtype=np.int64)
    identifiers = np.asarray(xmap.prop["grain_id"]).ravel()
    grain[rows, cols] = np.where(
        np.isfinite(identifiers), identifiers, UNLABELLED
    ).astype(np.int64)

    alpha = nye_tensor(field, step_x1, step_x2, grain=grain)
    return gnd_density(alpha, length, estimator)
