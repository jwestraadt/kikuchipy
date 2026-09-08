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

**Units (requirements D14.5, D14.6).**  Gradients are per METRE, with
the map step converted from
:attr:`~orix.crystal_map.CrystalMap.scan_unit`; a missing or
unrecognized unit, ``"px"`` included, is a ``ValueError`` and never a
guess.  ``burgers_vector_length`` is in METRES and has no default, for
the same reason.  The returned density is in ``m^-2`` and its expected
noise floor scale is ``rho_noise ~ sigma_beta / (b * step)``, about
4e12 to 8e12 m^-2 at ``sigma_beta = 1e-4``, ``b = 0.25 nm`` and a
100 nm step (Jiang, Britton and Wilkinson 2013; Ernould's thesis).

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
    raise NotImplementedError("Stage C is not implemented yet")


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
    The direction matters and is not symmetric: replacing ``beta13``
    and ``beta23`` by ``-beta31`` and ``-beta32`` instead keeps the
    noisy pair and discards the quiet one, which is the backwards
    mutant of plan section 4.3.

    A full projection onto a pure rotation, keeping only the polar
    ``R``, is explicitly NOT offered: Ruggles 2020 shows that
    discarding the elastic-strain derivatives corrupts the GND
    identification (requirements D14.3).
    """
    raise NotImplementedError("Stage C is not implemented yet")


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
    """
    raise NotImplementedError("Stage C is not implemented yet")


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
    raise NotImplementedError("Stage C is not implemented yet")


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
    raise NotImplementedError("Stage C is not implemented yet")


def log10_density(density: np.ndarray) -> np.ndarray:
    """Return the base-ten logarithm of a GND density map, guarded.

    The plotting recipe of requirements D14.6, in one place so that
    the tutorial does not repeat it.  A density is a sum of moduli and
    so is never negative, but it IS exactly zero on a strain-free
    synthetic map, where a bare :func:`numpy.log10` returns ``-inf``
    and warns; and a caller may hand this any array at all.

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
    raise NotImplementedError("Stage C is not implemented yet")


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
        (default, 30/14) adds ``alpha_12`` and ``alpha_21``; ``"a9"``
        (30/20) takes all nine. The six entries beyond ``alpha_i3``
        rest on neglecting the out-of-plane derivatives, so ``"a3"``
        is the assumption-free choice and the other two trade
        assumptions for sensitivity.
    enforce_antisymmetry : bool, optional
        Whether to replace ``beta31`` and ``beta32`` by ``-beta13``
        and ``-beta23`` in the DETECTOR frame before rotating
        (requirements D14.3). Default is True: those two entries are
        about 9.6 times noisier than the pair replacing them for
        typical geometry and otherwise dominate the result. It applies
        to this function alone and never changes a strain or a stress.

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
    step.

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
    raise NotImplementedError("Stage C is not implemented yet")
