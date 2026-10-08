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

"""The Fourier-Mellin rotation initial guess of the HREBSD IC-GN
engine: the per-run polar look-up table, the angle about the detector
normal from the reused target spectra, the de-rotation, the
translation and the partial starting row, the acceptance, and the
pre-fit ``CrystalMap`` gate.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Requirements D22 (``specs/2026-09-07-hrebsd-dic/requirements.md``)
govern; the names below are frozen by D22.3 to D22.7, and the
argument lists beyond them are fixed by the failing-tests commit and
recorded in the docstring of
``tests/test_indexing/test_hrebsd_fourier_mellin.py``.

The seed stage is ONE xp-agnostic code path, run under numpy on the
CPU route and in the default suite and under cupy on the device
(D22.10, D21.14.3).  Import direction (D22.6): this module imports
neither ``_engine`` nor ``_batched`` at module scope (``_batched`` may
be imported inside a function), and never cupy at module scope.

Call-time seams (D22.6): :func:`fourier_mellin_rows` reaches
:func:`fourier_mellin_angles`, :func:`fourier_mellin_peak`,
:func:`fourier_mellin_derotate`, :func:`fourier_mellin_translate` and
:func:`fourier_mellin_criteria` through this module's globals at call
time, never through a ``from ... import name`` binding, a default
argument or a local alias taken before a loop.

SKELETON (Stage F failing-tests gate, 2026-10-08): every function
body raises ``NotImplementedError``; the constants and the containers
are the frozen ones.
"""

import numpy as np

_NOT_IMPLEMENTED = "Stage F: not implemented yet"

# ---------------------- Frozen constants (D22.3, D22.7) -------------- #

# The polar sampling of D22.3.3: FM_N_THETA angles over [0, pi), radii
# from FM_RHO_MIN to FM_RHO_MAX cycles/px in steps of 1 / min(sr, sc)
FM_N_THETA: int = 360
FM_RHO_MIN: float = 0.02
FM_RHO_MAX: float = 0.40

# The search window of D22.3.6 and D22.3.7, in degrees
FM_SEARCH_DEG: float = 30.0

# The pre-fit gate threshold of D22.7 on the twist about the detector
# normal, in degrees (RECORDED DEFAULT, measured then pinned)
FM_GATE_DEG: float = 1.5

# The twist below which a fitted non-reference point counts as "no
# rotation" for the D22.7 warning, in degrees (an input condition, not
# a float equality)
FM_ZERO_TWIST_DEG: float = 1e-9

# The accepted values of ``fourier_mellin`` (D22.1), case-sensitive
FOURIER_MELLIN_VALUES: tuple[str, ...] = ("off", "auto", "always")

# The ``SeedBatch.extras`` input key and the two ``SeedBatch.outputs``
# keys of D22.6
FOURIER_MELLIN_ROUTE_KEY: str = "fourier_mellin_route"
FOURIER_MELLIN_ANGLE_KEY: str = "fourier_mellin_angle"
FOURIER_MELLIN_APPLIED_KEY: str = "fourier_mellin_applied"

# The route codes of D22.6: not routed, routed with the acceptance,
# forced (the retry)
ROUTE_NONE: int = 0
ROUTE_ACCEPT: int = 1
ROUTE_FORCED: int = 2

# The two properties of D22.9, written on runs with ``fourier_mellin``
# other than ``"off"`` only; the engine re-exports this tuple
FOURIER_MELLIN_PROP_NAMES: tuple[str, ...] = (
    "fourier_mellin_seed",
    "fourier_mellin_angle",
)

# The seed codes of the ``fourier_mellin_seed`` property (D22.9)
FOURIER_MELLIN_SEED_TRANSLATION: int = 0
FOURIER_MELLIN_SEED_FIRST_PASS: int = 1
FOURIER_MELLIN_SEED_RETRY: int = 2
FOURIER_MELLIN_SEED_NONE: int = -1

# The frozen message literals of D22.1 and D22.11 and the frozen
# warning literal of D22.7 (no stage letters in public text)
FOURIER_MELLIN_VALUE_MESSAGE: str = (
    "fourier_mellin {fourier_mellin!r} not in the list of supported values "
    "['off', 'auto', 'always']"
)
FOURIER_MELLIN_NEIGHBORS_MESSAGE: str = (
    "seed_from_neighbors=True cannot be combined with "
    "fourier_mellin={fourier_mellin!r}; use one seeding strategy"
)
FOURIER_MELLIN_XMAP_MESSAGE: str = (
    "fourier_mellin='auto' requires the input CrystalMap; pass "
    "fourier_mellin='always' to seed every point without it"
)
FOURIER_MELLIN_NO_TWIST_WARNING: str = (
    "fourier_mellin='auto' found no rotation about the detector normal in "
    "the crystal map (every point matches its reference orientation); no "
    "point is routed before the fit and only the post-fit retry applies"
)


# ------------------------------ Containers --------------------------- #


class FourierMellinState:
    """The per-reference Fourier-Mellin state of requirements D22.6,
    attached to the reference's ``SeedState.fourier_mellin``.

    Parameters
    ----------
    lut
        ``(indices, weights)`` of :func:`fourier_mellin_lut`, the run's
        shared look-up table in the namespace of the batch.
    reference_profile
        ``(n_theta,)`` float64 zero-mean unit-norm radial-mean profile
        of the reference spectrum, in ``ctx.xp``.
    box
        The bounding-box resident of D22.4.1: a ``ReferenceResident``
        whose subregion is every pixel of the D5 bounding box.
    resident
        A REFERENCE to the grain's existing subregion
        ``ReferenceResident`` (on the device the very object the
        lockstep uses), never a copy.
    n_theta, rho_min, rho_max, search_deg
        The recipe constants the state was built with.
    """

    def __init__(
        self,
        lut,
        reference_profile,
        box,
        resident,
        n_theta: int = FM_N_THETA,
        rho_min: float = FM_RHO_MIN,
        rho_max: float = FM_RHO_MAX,
        search_deg: float = FM_SEARCH_DEG,
    ) -> None:
        self.lut = lut
        self.reference_profile = reference_profile
        self.box = box
        self.resident = resident
        self.n_theta = int(n_theta)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.search_deg = float(search_deg)


# ------------------------- Angle (D22.3) ----------------------------- #


def fourier_mellin_lut(sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-run polar look-up table of requirements D22.3.3
    for an ``(sr, sc)`` spectrum: ``(indices, weights)``, the
    ``(n_theta, n_rho, 4)`` int64 flat indices into the unshifted
    spectrum and the ``(n_theta, n_rho, 4)`` float64 bilinear weights,
    sampled in PHYSICAL frequency (never bin units on both axes).
    Built once per run on the host and cached by crop shape."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_hann_stencil(xp, spectra):
    """Return the edge-treated spectra of requirements D22.3.1: the
    periodic Hann window applied EXACTLY in the frequency domain,
    separably per axis, ``X_w[k] = X[k] / 2 - (X[k - 1] + X[k + 1]) /
    4`` with circular indices, on ``(P, sr, sc)`` *spectra* promoted
    to complex128 (D22.12).  A NEW array: *spectra* is never modified
    in place (D22.6)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_profiles(xp, spectra, lut):
    """Return the ``(P, n_theta)`` float64 zero-mean unit-norm
    radial-mean profiles of requirements D22.3.2 and D22.3.4:
    ``log1p`` of the magnitude of the edge-treated *spectra*, gathered
    through the look-up table *lut* and averaged over radius (a gather
    and a fixed-shape sum, no scatter and no atomics)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_peak(xp, correlation, search_deg):
    """Return ``(theta_deg, peak)``, each ``(P,)`` float64, of the
    ``(P, n_theta)`` circular *correlation* (requirements D22.3.6):
    the argmax over the output indices whose lag lies within
    *search_deg*, ties to the LOWEST output index, a parabolic sub-bin
    offset from the raw circular neighbours when the curvature is
    negative (else 0), and ``theta_hat = (lag + delta) * 180 /
    n_theta``.  The peak is a diagnostic only."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_angles(ctx, target_spectra, fm_state):
    """Return ``(theta_deg, peak)``, each ``(P,)`` float64 in
    ``ctx.xp``: the Fourier-Mellin angle about the detector normal of
    every slot from the REUSED *target_spectra* of ``seed_spectra``
    against ``fm_state.reference_profile`` (requirements D22.3, D22.6),
    through :func:`fourier_mellin_peak` at ``fm_state.search_deg``.
    Positive is a rotation about +z, ``h21 = +sin(theta_hat)``."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


# --------------- De-rotation, translation, row (D22.4) --------------- #


def fourier_mellin_derotate(ctx, batch, fm_state, theta_deg):
    """Return ``(crops, ok)``: the ``(P, sr, sc)`` float64 targets of
    *batch* de-rotated by ``R(theta_deg)`` about the grain reference
    projection centre over every pixel of the D5 bounding box,
    ``D(xi) = target(R xi)``, through ``ctx.kernels.gather`` on
    ``fm_state.box``, and the ``(P,)`` bool conjunction of the
    coordinate flags (requirements D22.4.1)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_translate(ctx, crops, seed_state):
    """Return ``rows_t``, the ``(P, 8)`` float64 translation rows the
    Stage E phase cross-correlation seed returns for the de-rotated
    *crops* against ``seed_state.reference_spectrum`` (requirements
    D22.4.2); reached through a function-scope import of ``_batched``."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_partial_row(xp, theta_deg, rows_t):
    """Return the ``(P, 8)`` float64 PARTIAL starting rows of
    requirements D22.4.3, ``W0 = R(theta) T(t)``, i.e. ``(c - 1, -s, c
    tx - s ty, s, c - 1, s tx + c ty, 0, 0)`` with ``t = (rows_t[:, 2],
    rows_t[:, 5])`` and *theta_deg* in degrees."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


# ------------------------- Acceptance (D22.5) ------------------------ #


def fourier_mellin_criteria(ctx, batch, fm_state, rows):
    """Return the ``(P,)`` float64 D2.7 criterion of every slot of
    *batch* AT the starting *rows*, through ``ctx.kernels.gather`` on
    the grain's subregion resident ``fm_state.resident`` with the
    initial shifts K of the lockstep and ``ctx.kernels.final_criterion``
    (requirements D22.5)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_rows(ctx, batch, target_spectra, seed_state, h_t, route):
    """Return ``(rows, angle, applied)``: the whole Fourier-Mellin
    branch on one sub-batch at the FULL shape P (requirements D22.6),
    ``(P, 8)`` float64, ``(P,)`` float64 and ``(P,)`` bool in
    ``ctx.xp``.  *h_t* are the Stage E translation rows and *route*
    the ``(P,)`` int8 host route flags (0 not routed, 1 routed with
    the acceptance of D22.5, 2 forced); the rows are *h_t* except
    where D22.5 keeps the FM row (route 1) or the FM row is forced and
    valid (route 2), and a failed estimate returns *h_t* with
    ``applied`` False."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def build_fourier_mellin_state(ctx, state, resident):
    """Return the :class:`FourierMellinState` of one
    ``ReferenceState`` *state* in the namespace of *ctx*, holding a
    reference to the grain's subregion *resident* (requirements D22.6),
    or ``None`` when the reference profile is non-finite or of zero
    norm (that grain then runs as with ``"off"``)."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


# --------------------------- The gate (D22.7) ------------------------ #


def twist_about_detector_normal(xmap, detector, point_index, reference_index):
    """Return the ``(n,)`` float64 twist in degrees about the detector
    normal of the symmetry-reduced misorientation of every point in
    *point_index* (flat map indices) from its GRAIN reference in
    *reference_index* (flat map indices, one per point), NaN where it
    cannot be computed (requirements D22.7): ``R_s = g_t^T S^T g_r``
    maximising the trace over the reference phase's proper point group,
    ``R_det = M R_s M^T`` with ``M = sample_to_detector_matrix``, and
    ``atan2(R_det[1, 0] - R_det[0, 1], R_det[0, 0] + R_det[1, 1])``."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def fourier_mellin_routes(twist_deg, gate_deg):
    """Return the ``(n,)`` int8 pre-fit routes of requirements D22.7:
    1 if and only if ``|twist| >= gate_deg`` or the twist is NaN (the
    gate fails open), else 0."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
