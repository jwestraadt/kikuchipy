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

"""

import warnings

import numpy as np

from kikuchipy.indexing._hrebsd._tensors import (
    best_orientation_matrices,
    sample_to_detector_matrix,
)

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


# The read-only host look-up tables, one per crop shape (D22.3.3).  A
# plain dictionary: the D21.13 import audit allows no ``functools``
_HOST_LUTS: dict = {}


def _host_lut(sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the read-only host look-up table of :func:`fourier_mellin_lut`,
    built once per crop shape (D22.3.3)."""
    key = (int(sr), int(sc))
    if key not in _HOST_LUTS:
        _HOST_LUTS[key] = _build_host_lut(*key)
    return _HOST_LUTS[key]


def _build_host_lut(sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the host look-up table of :func:`_host_lut`."""
    m = min(sr, sc)
    n_rho = int(np.floor((FM_RHO_MAX - FM_RHO_MIN) * m + 1e-9)) + 1
    theta = np.arange(FM_N_THETA, dtype=np.float64) * np.pi / FM_N_THETA
    rho = FM_RHO_MIN + np.arange(n_rho, dtype=np.float64) / m
    # PHYSICAL frequency: (fx sc, fy sr) in (column, row) bin units of
    # the unshifted spectrum, never bin units on both axes
    u = rho[None, :] * np.cos(theta)[:, None] * sc
    v = rho[None, :] * np.sin(theta)[:, None] * sr
    c0 = np.floor(u)
    r0 = np.floor(v)
    du = u - c0
    dv = v - r0
    c0 = c0.astype(np.int64)
    r0 = r0.astype(np.int64)
    indices = np.empty((*u.shape, 4), dtype=np.int64)
    weights = np.empty((*u.shape, 4), dtype=np.float64)
    corners = (
        (0, 0, (1 - dv) * (1 - du)),
        (0, 1, (1 - dv) * du),
        (1, 0, dv * (1 - du)),
        (1, 1, dv * du),
    )
    for k, (a, b, weight) in enumerate(corners):
        indices[..., k] = ((r0 + a) % sr) * sc + (c0 + b) % sc
        weights[..., k] = weight
    indices.flags.writeable = False
    weights.flags.writeable = False
    return indices, weights


def fourier_mellin_lut(sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-run polar look-up table of requirements D22.3.3
    for an ``(sr, sc)`` spectrum: ``(indices, weights)``, the
    ``(n_theta, n_rho, 4)`` int64 flat indices into the unshifted
    spectrum and the ``(n_theta, n_rho, 4)`` float64 bilinear weights,
    sampled in PHYSICAL frequency (never bin units on both axes).
    Built once per run on the host and cached by crop shape.

    The ``n_theta = FM_N_THETA`` angles are ``k pi / n_theta`` over
    ``[0, pi)``; with ``m = min(sr, sc)`` the ``n_rho = floor((FM_RHO_MAX
    - FM_RHO_MIN) m + 1e-9) + 1`` radii are ``FM_RHO_MIN + j / m``
    cycles/px; each sample ``rho (cos theta, sin theta)`` is read at
    the fractional bin coordinates ``(fx sc, fy sr)`` (column, row) with
    bilinear weights over its four neighbouring bins, indices modulo
    ``(sr, sc)``.  The arrays are read-only HOST arrays shared by every
    caller.
    """
    return _host_lut(int(sr), int(sc))


def fourier_mellin_hann_stencil(xp, spectra):
    """Return the edge-treated spectra of requirements D22.3.1: the
    periodic Hann window applied EXACTLY in the frequency domain,
    separably per axis, ``X_w[k] = X[k] / 2 - (X[k - 1] + X[k + 1]) /
    4`` with circular indices, on ``(P, sr, sc)`` *spectra* promoted
    to complex128 (D22.12).  A NEW array: *spectra* is never modified
    in place (D22.6)."""
    # ``astype`` always copies: the reused target spectra stay untouched
    windowed = xp.asarray(spectra).astype(xp.complex128)
    for axis in (-1, -2):
        windowed = (
            windowed / 2
            - (xp.roll(windowed, 1, axis=axis) + xp.roll(windowed, -1, axis=axis)) / 4
        )
    return windowed


def fourier_mellin_profiles(xp, spectra, lut):
    """Return the ``(P, n_theta)`` float64 zero-mean unit-norm
    radial-mean profiles of requirements D22.3.2 and D22.3.4:
    ``log1p`` of the magnitude of the edge-treated *spectra*, gathered
    through the look-up table *lut* and averaged over radius (a gather
    and a fixed-shape sum, no scatter and no atomics).

    A profile with a zero or non-finite norm is NaN throughout (the
    estimate's failure, D22.5)."""
    indices, weights = lut
    windowed = fourier_mellin_hann_stencil(xp, spectra)
    n_slots = int(windowed.shape[0])
    with np.errstate(all="ignore"):
        magnitude = xp.log1p(xp.abs(windowed).reshape(n_slots, -1))
        del windowed
        samples = (magnitude[:, indices] * weights[None]).sum(axis=-1)
        profiles = samples.mean(axis=-1)
        profiles = profiles - profiles.mean(axis=1, keepdims=True)
        norm = xp.sqrt((profiles * profiles).sum(axis=1, keepdims=True))
        profiles = profiles / norm
        good = xp.isfinite(norm) & (norm > 0)
        profiles = xp.where(good, profiles, xp.nan)
    return xp.ascontiguousarray(profiles, dtype=xp.float64)


def fourier_mellin_peak(xp, correlation, search_deg):
    """Return ``(theta_deg, peak)``, each ``(P,)`` float64, of the
    ``(P, n_theta)`` circular *correlation* (requirements D22.3.6):
    the argmax over the output indices whose lag lies within
    *search_deg*, ties to the LOWEST output index, a parabolic sub-bin
    offset from the raw circular neighbours when the curvature is
    negative (else 0), and ``theta_hat = (lag + delta) * 180 /
    n_theta``.  The peak is a diagnostic only.

    Output index ``k`` carries the lag ``k`` for ``k < n / 2`` and ``k -
    n`` otherwise.  A row that is not finite throughout gives a NaN
    angle (the estimate's failure, D22.5)."""
    correlation = xp.asarray(correlation, dtype=xp.float64)
    n_slots, n = (int(i) for i in correlation.shape)
    index = xp.arange(n)
    lags = xp.where(index < n // 2, index, index - n)
    inside = xp.abs(lags).astype(xp.float64) * 180.0 / n <= float(search_deg)
    masked = xp.where(inside[None], correlation, -xp.inf)
    # ``argmax`` returns the FIRST maximum, i.e. the lowest output index
    k = xp.argmax(masked, axis=1)
    slots = xp.arange(n_slots)
    centre = correlation[slots, k]
    left = correlation[slots, (k - 1) % n]
    right = correlation[slots, (k + 1) % n]
    with np.errstate(all="ignore"):
        denominator = left - 2.0 * centre + right
        negative = denominator < 0
        safe = xp.where(negative, denominator, 1.0)
        delta = xp.where(negative, (left - right) / (2.0 * safe), 0.0)
        theta = (lags[k].astype(xp.float64) + delta) * 180.0 / n
    finite = xp.isfinite(correlation).all(axis=1)
    theta = xp.where(finite, theta, xp.nan)
    return (
        xp.ascontiguousarray(theta, dtype=xp.float64),
        xp.ascontiguousarray(centre, dtype=xp.float64),
    )


def _circular_correlation(ctx, reference_profile, profiles):
    """Return the ``(P, n_theta)`` circular ZNCC ``c[k] = sum_j p_ref[j]
    p_tgt[j + k]`` by length-``n_theta`` FFTs (D22.3.5)."""
    xp = ctx.xp
    with np.errstate(all="ignore"):
        reference = xp.conj(ctx.fft.fft(reference_profile))
        product = reference[None] * ctx.fft.fft(profiles, axis=1)
        correlation = xp.real(ctx.fft.ifft(product, axis=1))
    return xp.ascontiguousarray(correlation, dtype=xp.float64)


def fourier_mellin_angles(ctx, target_spectra, fm_state):
    """Return ``(theta_deg, peak)``, each ``(P,)`` float64 in
    ``ctx.xp``: the Fourier-Mellin angle about the detector normal of
    every slot from the REUSED *target_spectra* of ``seed_spectra``
    against ``fm_state.reference_profile`` (requirements D22.3, D22.6),
    through :func:`fourier_mellin_peak` at ``fm_state.search_deg``.
    Positive is a rotation about +z, ``h21 = +sin(theta_hat)``."""
    profiles = fourier_mellin_profiles(ctx.xp, target_spectra, fm_state.lut)
    correlation = _circular_correlation(ctx, fm_state.reference_profile, profiles)
    return fourier_mellin_peak(ctx.xp, correlation, fm_state.search_deg)


# --------------- De-rotation, translation, row (D22.4) --------------- #


def _rotation_matrices(xp, theta_deg):
    """Return the ``(P, 3, 3)`` float64 rotations ``R(theta_deg)`` about
    the PC-centred origin (the D1.3 frame, a rotation about +z)."""
    theta = xp.deg2rad(xp.asarray(theta_deg, dtype=xp.float64))
    cos = xp.cos(theta)
    sin = xp.sin(theta)
    matrices = xp.zeros((int(theta.shape[0]), 3, 3), dtype=xp.float64)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 1] = -sin
    matrices[:, 1, 0] = sin
    matrices[:, 1, 1] = cos
    matrices[:, 2, 2] = 1.0
    return matrices


def fourier_mellin_derotate(ctx, batch, fm_state, theta_deg):
    """Return ``(crops, ok)``: the ``(P, sr, sc)`` float64 targets of
    *batch* de-rotated by ``R(theta_deg)`` about the grain reference
    projection centre over every pixel of the D5 bounding box,
    ``D(xi) = target(R xi)``, through ``ctx.kernels.gather`` on
    ``fm_state.box``, and the ``(P,)`` bool conjunction of the
    coordinate flags (requirements D22.4.1)."""
    xp = ctx.xp
    box = fm_state.box
    matrices = _rotation_matrices(xp, theta_deg)
    values, ok = ctx.kernels.gather(box, batch.coefficients, matrices)
    n_slots = int(matrices.shape[0])
    crops = xp.asarray(values, dtype=xp.float64).reshape(n_slots, *box.box_shape)
    return xp.ascontiguousarray(crops), xp.asarray(ok, dtype=bool)


def fourier_mellin_translate(ctx, crops, seed_state):
    """Return ``rows_t``, the ``(P, 8)`` float64 translation rows the
    Stage E phase cross-correlation seed returns for the de-rotated
    *crops* against ``seed_state.reference_spectrum`` (requirements
    D22.4.2); reached through a function-scope import of ``_batched``.

    The crops are made zero-mean unit-norm and transformed at
    ``seed_state.precision`` exactly as ``seed_spectra`` transforms a
    target crop, then the Stage E rows are computed by the Stage E code
    itself, unchanged."""
    from kikuchipy.indexing._hrebsd import _batched

    xp = ctx.xp
    dtype = np.dtype(seed_state.precision)
    with np.errstate(all="ignore"):
        zmn = _batched._zero_mean_normalize_crops(
            xp, xp.asarray(crops, dtype=xp.float64)
        )
        spectra = ctx.fft.fft2(zmn.astype(dtype), axes=(1, 2))
    del zmn
    spectra = xp.ascontiguousarray(spectra.astype(dtype, copy=False))
    # the Stage E body ``seed_homographies`` runs for ``h_T``, reached
    # directly so the seam's patch point sees ONE call per sub-batch
    rows_t = _batched._translation_rows(ctx, spectra, seed_state)
    return xp.ascontiguousarray(rows_t, dtype=xp.float64)


def fourier_mellin_partial_row(xp, theta_deg, rows_t):
    """Return the ``(P, 8)`` float64 PARTIAL starting rows of
    requirements D22.4.3, ``W0 = R(theta) T(t)``, i.e. ``(c - 1, -s, c
    tx - s ty, s, c - 1, s tx + c ty, 0, 0)`` with ``t = (rows_t[:, 2],
    rows_t[:, 5])`` and *theta_deg* in degrees."""
    theta = xp.deg2rad(xp.asarray(theta_deg, dtype=xp.float64))
    rows_t = xp.asarray(rows_t, dtype=xp.float64)
    cos = xp.cos(theta)
    sin = xp.sin(theta)
    tx = rows_t[:, 2]
    ty = rows_t[:, 5]
    rows = xp.zeros((int(theta.shape[0]), 8), dtype=xp.float64)
    with np.errstate(all="ignore"):
        rows[:, 0] = cos - 1
        rows[:, 1] = -sin
        rows[:, 2] = cos * tx - sin * ty
        rows[:, 3] = sin
        rows[:, 4] = cos - 1
        rows[:, 5] = sin * tx + cos * ty
    return rows


# ------------------------- Acceptance (D22.5) ------------------------ #


def fourier_mellin_criteria(ctx, batch, fm_state, rows):
    """Return the ``(P,)`` float64 D2.7 criterion of every slot of
    *batch* AT the starting *rows*, through ``ctx.kernels.gather`` on
    the grain's subregion resident ``fm_state.resident`` with the
    initial shifts K of the lockstep and ``ctx.kernels.final_criterion``
    (requirements D22.5).

    NaN where a row's warped coordinates are not finite (the gather's
    flag) or the warped values have a zero or non-finite centred norm.
    """
    from kikuchipy.indexing._hrebsd import _batched

    xp = ctx.xp
    resident = fm_state.resident
    shifts = _batched.initial_shifts(ctx, resident, batch)
    matrices = _batched.matrices_from_parameters(xp, xp.asarray(rows, dtype=xp.float64))
    values, ok = ctx.kernels.gather(resident, batch.coefficients, matrices)
    criterion = ctx.kernels.final_criterion(resident, values, shifts)
    criterion = xp.asarray(criterion, dtype=xp.float64)
    return xp.where(xp.asarray(ok, dtype=bool), criterion, xp.nan)


def _finite_rows(xp, rows):
    """Return ``(P,)`` bool, True where a row is finite throughout."""
    return xp.isfinite(rows).reshape(int(rows.shape[0]), -1).all(axis=1)


def fourier_mellin_rows(ctx, batch, target_spectra, seed_state, h_t, route):
    """Return ``(rows, angle, applied)``: the whole Fourier-Mellin
    branch on one sub-batch at the FULL shape P (requirements D22.6),
    ``(P, 8)`` float64, ``(P,)`` float64 and ``(P,)`` bool in
    ``ctx.xp``.  *h_t* are the Stage E translation rows and *route*
    the ``(P,)`` int8 host route flags (0 not routed, 1 routed with
    the acceptance of D22.5, 2 forced); the rows are *h_t* except
    where D22.5 keeps the FM row (route 1) or the FM row is forced and
    valid (route 2), and a failed estimate returns *h_t* with
    ``applied`` False.

    Every slot is computed (masked, never compacted), so that no
    routed slot's bits depend on which other slots are routed
    (D21.7.3).  ``angle`` is ``theta_hat`` on route-1 and route-2 slots
    and NaN on route-0 slots (the D22.6 outputs rule).  The FM stage
    never introduces a non-finite row and never raises for a per-slot
    failure: an estimate fails on a non-finite angle, a de-rotated crop
    with a false coordinate flag or a non-finite pixel, or a non-finite
    translation or FM row; a non-finite *h_t* is never rescued.  The
    criterion of D22.5 is evaluated (twice, once per candidate row, at
    the full P) only when a slot carries route 1.  The stages are
    reached through this module's globals at call time (D22.6).
    """
    xp = ctx.xp
    fm_state = seed_state.fourier_mellin
    route_host = np.asarray(route, dtype=np.int8).ravel()
    route_device = xp.asarray(route_host)
    h_t = xp.asarray(h_t, dtype=xp.float64)
    n_slots = int(h_t.shape[0])

    theta, _ = fourier_mellin_angles(ctx, target_spectra, fm_state)
    theta = xp.asarray(theta, dtype=xp.float64)
    crops, ok = fourier_mellin_derotate(ctx, batch, fm_state, theta)
    crops = xp.asarray(crops, dtype=xp.float64)
    rows_t = xp.asarray(
        fourier_mellin_translate(ctx, crops, seed_state), dtype=xp.float64
    )
    h_fm = xp.asarray(fourier_mellin_partial_row(xp, theta, rows_t), dtype=xp.float64)

    valid = (
        xp.isfinite(theta)
        & xp.asarray(ok, dtype=bool)
        & _finite_rows(xp, crops)
        & xp.isfinite(rows_t[:, 2])
        & xp.isfinite(rows_t[:, 5])
        & _finite_rows(xp, h_fm)
        & _finite_rows(xp, h_t)
    )
    forced = route_device == ROUTE_FORCED
    accept = route_device == ROUTE_ACCEPT
    if bool((route_host == ROUTE_ACCEPT).any()):
        criterion_t = xp.asarray(
            fourier_mellin_criteria(ctx, batch, fm_state, h_t), dtype=xp.float64
        )
        criterion_fm = xp.asarray(
            fourier_mellin_criteria(ctx, batch, fm_state, h_fm), dtype=xp.float64
        )
        # STRICTLY lower: ties and every comparison with NaN keep h_T
        with np.errstate(all="ignore"):
            wins = xp.isfinite(criterion_fm) & (criterion_fm < criterion_t)
    else:
        wins = xp.zeros(n_slots, dtype=bool)
    applied = valid & (forced | (accept & wins))
    with np.errstate(all="ignore"):
        rows = xp.where(applied[:, None], h_fm, h_t)
    angle = xp.where(route_device != ROUTE_NONE, theta, xp.nan)
    return (
        xp.ascontiguousarray(rows, dtype=xp.float64),
        xp.ascontiguousarray(angle, dtype=xp.float64),
        xp.ascontiguousarray(applied, dtype=bool),
    )


def _box_resident(ctx, state):
    """Return the bounding-box resident of requirements D22.4.1: a
    ``ReferenceResident`` whose subregion is EVERY pixel of the D5
    bounding box of *state* in row-major order (dead-band pixels
    included), carrying the fields ``kernels.gather`` reads at the
    precision of ``ctx.kernels``, plus ``box_shape``, ``(sr, sc)``."""
    from kikuchipy.indexing._hrebsd import _batched

    xp = ctx.xp
    precision = ctx.kernels.precision
    pixel_dtype = np.float32 if precision == "mixed" else np.float64
    r0, r1, c0, c1 = (int(i) for i in state.bounds)
    rows, columns = np.mgrid[r0:r1, c0:c1]
    rows = rows.ravel()
    columns = columns.ravel()
    pcx, pcy = (float(i) for i in np.asarray(state.pc_pixels, dtype=np.float64)[:2])
    xi_x = columns + 0.5 - pcx
    xi_y = rows + 0.5 - pcy
    shape = tuple(int(i) for i in state.shape)

    def upload(array, dtype):
        return xp.asarray(np.ascontiguousarray(np.asarray(array).astype(dtype)))

    box = _batched.ReferenceResident()
    box.precision = precision
    box.shape = shape
    box.box_shape = (r1 - r0, c1 - c0)
    box.n_pixels = int(rows.size)
    box.xi_x = upload(xi_x, pixel_dtype)
    box.xi_y = upload(xi_y, pixel_dtype)
    box.mask_index = upload(np.ravel_multi_index((rows, columns), shape), np.int32)
    if precision == "mixed":
        box.columns = upload(columns, np.int32)
        box.rows = upload(rows, np.int32)
    else:
        box.columns = None
        box.rows = None
    box.offset = (pcx - 0.5, pcy - 0.5)
    box.reference = None
    box.gradient_x = None
    box.gradient_y = None
    box.weights = None
    box.transfer_function = None
    return box


# The attribute of a ``SeedContext`` holding its uploaded look-up
# tables, one upload per context and crop shape (D22.3.3: uploaded once
# per device session, released with the context)
_CONTEXT_LUTS_ATTRIBUTE: str = "_fourier_mellin_luts"


def _context_lut(ctx, sr: int, sc: int):
    """Return the look-up table of ``(sr, sc)`` in the namespace of
    *ctx*, uploaded once per context (cached on the context itself, so
    it lives and dies with it).  A context that refused the attribute
    would fail loudly here rather than re-upload per call (2026-10-08
    implementation review, RC-F-CONV-6)."""
    per_context = getattr(ctx, _CONTEXT_LUTS_ATTRIBUTE, None)
    if per_context is None:
        per_context = {}
        setattr(ctx, _CONTEXT_LUTS_ATTRIBUTE, per_context)
    key = (int(sr), int(sc))
    if key not in per_context:
        indices, weights = fourier_mellin_lut(sr, sc)
        per_context[key] = (
            ctx.xp.asarray(indices, dtype=ctx.xp.int64),
            ctx.xp.asarray(weights, dtype=ctx.xp.float64),
        )
    return per_context[key]


def build_fourier_mellin_state(ctx, state, resident):
    """Return the :class:`FourierMellinState` of one
    ``ReferenceState`` *state* in the namespace of *ctx*, holding a
    reference to the grain's subregion *resident* (requirements D22.6),
    or ``None`` when the reference profile is non-finite or of zero
    norm (that grain then runs as with ``"off"``).

    The reference profile is computed in complex128 and float64 from
    the reference's zero-mean unit-norm D5 crop (the transform
    ``build_seed_state`` makes at its default precision) whatever the
    seed precision (D22.12)."""
    from kikuchipy.indexing._hrebsd import _batched

    xp = ctx.xp
    r0, r1, c0, c1 = (int(i) for i in state.bounds)
    lut = _context_lut(ctx, r1 - r0, c1 - c0)
    crop = xp.asarray(np.ascontiguousarray(state.reference_subregion), dtype=xp.float64)
    with np.errstate(all="ignore"):
        zmn = _batched._zero_mean_normalize_crops(xp, crop[None])
        spectrum = ctx.fft.fft2(zmn.astype(xp.complex128), axes=(1, 2))
    del crop, zmn
    profile = fourier_mellin_profiles(xp, spectrum, lut)[0]
    if not bool(xp.isfinite(profile).all()):
        return None
    box = _box_resident(ctx, state)
    return FourierMellinState(
        lut, xp.ascontiguousarray(profile, dtype=xp.float64), box, resident
    )


# --------------------------- The gate (D22.7) ------------------------ #


def twist_about_detector_normal(
    xmap, detector, point_index, reference_index, *, navigation_shape=None
):
    """Return the ``(n,)`` float64 twist in degrees about the detector
    normal of the symmetry-reduced misorientation of every point in
    *point_index* (flat map indices) from its GRAIN reference in
    *reference_index* (flat map indices, one per point), NaN where it
    cannot be computed (requirements D22.7): ``R_s = g_t^T S^T g_r``
    maximising the trace over the reference phase's proper point group,
    ``R_det = M R_s M^T`` with ``M = sample_to_detector_matrix``, and
    ``atan2(R_det[1, 0] - R_det[0, 1], R_det[0, 0] + R_det[1, 1])``.

    NaN covers a point or a reference that is not indexed, a
    non-finite rotation and a point of a phase other than its
    reference's (the gate fails open).  A phase without a point group
    uses no symmetry and warns (the ``segment_grains`` precedent).
    The swing-twist split's twist is exact and independent of the
    out-of-plane swing; matrices are used throughout, never a rotation
    vector (``det M = -1``).

    Flat map indices are matched to map points exactly as
    ``segment_grains`` matches them, through the map's own row and
    column grid (D22.7 step 1): the flat index ``i`` is the grid
    position ``divmod(i, nx)`` of *navigation_shape* ``(ny, nx)`` (the
    map grid's own shape when ``None``), and a position outside the
    grid or holding no map point gives NaN, so a sliced map fails open
    (2026-10-08 implementation review, RF-F3)."""
    from kikuchipy.indexing._hrebsd._segmentation import _map_grid

    point_index = np.asarray(point_index, dtype=np.int64).ravel()
    reference_index = np.asarray(reference_index, dtype=np.int64).ravel()
    n = int(point_index.size)
    twist = np.full(n, np.nan, dtype=np.float64)
    if len(tuple(xmap.shape)) == 0:
        # orix's shape of a one-point map, which has no row grid
        # (``map_grids`` names it); its one point is flat index 0
        grid_shape = (1, int(np.asarray(xmap.phase_id).size))
        grid = np.arange(grid_shape[1], dtype=np.int64).reshape(grid_shape)
    else:
        grid, grid_shape = _map_grid(xmap)
    if navigation_shape is None:
        navigation_shape = grid_shape
    ny, nx = (int(i) for i in navigation_shape)

    def map_point(flat_index):
        row, col = np.divmod(flat_index, nx)
        inside = (
            (flat_index >= 0) & (row < min(ny, grid.shape[0])) & (col < grid.shape[1])
        )
        point = np.full(flat_index.shape, -1, dtype=np.int64)
        point[inside] = grid[row[inside], col[inside]]
        return point

    point_index = map_point(point_index)
    reference_index = map_point(reference_index)
    described = (point_index >= 0) & (reference_index >= 0)
    # unusable positions read point 0 and are masked out below
    point_index = np.where(described, point_index, 0)
    reference_index = np.where(described, reference_index, 0)
    size = int(np.asarray(xmap.phase_id).size)
    phase_id = np.asarray(xmap.phase_id).ravel().astype(np.int64)
    indexed = np.asarray(xmap.is_indexed).ravel().astype(bool)
    g = best_orientation_matrices(xmap).reshape(size, 3, 3)
    finite = np.isfinite(g).all(axis=(1, 2))
    m = sample_to_detector_matrix(detector)

    usable = (
        described
        & indexed[point_index]
        & indexed[reference_index]
        & finite[point_index]
        & finite[reference_index]
        & (phase_id[point_index] == phase_id[reference_index])
    )
    for identifier in np.unique(phase_id[reference_index[usable]]):
        selection = np.flatnonzero(usable & (phase_id[reference_index] == identifier))
        phase = xmap.phases[int(identifier)]
        point_group = phase.point_group
        if point_group is None:
            warnings.warn(
                f"phase {phase.name!r} carries no point group, so the twist about "
                "the detector normal is measured from the RAW misorientation, "
                "NOT symmetry-reduced. Give the phase a space group or a point "
                "group to reduce it",
                UserWarning,
            )
            operators = np.eye(3)[None]
        else:
            operators = np.asarray(
                point_group.proper_subgroup.to_matrix(), dtype=np.float64
            ).reshape(-1, 3, 3)
        g_t = g[point_index[selection]]
        g_r = g[reference_index[selection]]
        # trace(g_t^T S^T g_r) = sum(S * (g_r g_t^T)) elementwise
        outer = g_r @ np.swapaxes(g_t, 1, 2)
        traces = np.einsum("sij,nij->ns", operators, outer)
        best = np.argmax(traces, axis=1)
        s = operators[best]
        r_s = np.swapaxes(g_t, 1, 2) @ np.swapaxes(s, 1, 2) @ g_r
        r_det = m @ r_s @ m.T
        twist[selection] = np.rad2deg(
            np.arctan2(r_det[:, 1, 0] - r_det[:, 0, 1], r_det[:, 0, 0] + r_det[:, 1, 1])
        )
    return twist


def fourier_mellin_routes(twist_deg, gate_deg):
    """Return the ``(n,)`` int8 pre-fit routes of requirements D22.7:
    1 if and only if ``|twist| >= gate_deg`` or the twist is NaN (the
    gate fails open), else 0."""
    twist = np.asarray(twist_deg, dtype=np.float64).ravel()
    with np.errstate(invalid="ignore"):
        routed = np.isnan(twist) | (np.abs(twist) >= float(gate_deg))
    return routed.astype(np.int8)
