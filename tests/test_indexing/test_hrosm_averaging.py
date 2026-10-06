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

"""Tests of the grain averaging, the grain reference orientation
deviation (GROD), the grain table and the map grid of the HROSM tools.

The oracles are a test-local exact sampler of von Mises-Fisher and
Watson densities on the unit quaternions (orix'
``Rotation.random_vonmises`` is a Watson density on the global NumPy
random state and too slow at high concentrations, so it is used for one
cross-check only), orix' symmetry-aware ``angle_with``, and a
test-local transcription of EMsoft's expectation maximisation
(``EMforDS_``, ``Estep_``, ``Mstep_``, ``getQandL_``, ``logCp_`` and
the concentration table of ``mod_dirstats.f90``), vectorised over the
samples and the operators but keeping EMsoft's loop order over the
initial guesses and iterations, the operator side of each step and the
stale ``Q`` and ``L`` carried over when the density underflows.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest
from scipy.special import i0, i1, ive

from kikuchipy.indexing import (
    GrainTable,
    average_grain_orientations,
    grain_reference_orientation_deviation_map,
    kernel_average_misorientation_map,
)
from kikuchipy.indexing._hrosm._averaging import (
    _apply_kappa_gate,
    _center_pixel,
    _coverage_warning_message,
)
from kikuchipy.indexing._hrosm._directional_statistics import (
    C2,
    C2W,
    C,
    _em_emsoft,
    _final_representative,
    _kappa_from_y,
    _log_cp,
)
from kikuchipy.indexing._hrosm._grains import _broadcast_grain_props, _map_grid

# Recovery band of the mean orientation: this factor times 2 /
# sqrt(8 alpha N) radians; measured, then pinned
RECOVERY_ANGLE_FACTOR = 4.0
# Band [1 / b, b] of the estimated over the sampled concentration for
# 300 and 30 samples; measured, then pinned
KAPPA_RATIO_BAND_N300 = 1.25
KAPPA_RATIO_BAND_N30 = 1.6
# Largest ratio of the concentration estimated on samples scrambled on
# the side the model does not handle over the one estimated on the same
# samples scrambled on the model's side; measured, then pinned
WRONG_SIDE_MAX_KAPPA_RATIO = 0.2
# Seed of the unscrambled Watson samples at kappa 1e4 of which one
# initial guess underflows at its second iteration in EMsoft's
# expectation maximisation; the first seed from 80 up that does,
# measured, then pinned
WATSON_UNDERFLOW_SEED = 80

# Mean orientation of the synthetic grains, Euler angles in degrees
MU_EULER_DEG = (37.0, 51.0, 113.0)
# Proper rotations of m-3m in orix' order
OPERATORS = Oh.proper_subgroup.data
# Initial guesses of the fast expectation maximisation arms
N_EM_FAST = 3
# Smallest scalar part of the variants S_j * mu (equal to that of mu *
# S_j) whose operators scramble von Mises-Fisher samples for EMsoft's
# expectation maximisation only. The von Mises-Fisher density is not
# antipodally symmetric, and EMsoft's mixture has one component per
# operator: samples around a variant whose scalar part is near 0 or
# negative change sign when the scalar part is made non-negative, land
# near -S_j * mu, which is no centre of that model, and get no
# responsibility. The correct mixture also has a component per negated
# operator, so its samples use every operator and random signs.
VMF_MIN_VARIANT_SCALAR = 0.2


# ----------------------------- Helpers ------------------------------


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of quaternions broadcast over the
    leading axes.
    """
    a0, a1, a2, a3 = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    b0, b1, b2, b3 = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 + a2 * b0 + a3 * b1 - a1 * b3,
            a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1,
        ],
        axis=-1,
    )


def _conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def _axis_angle(axis, angle_deg) -> Rotation:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(np.atleast_1d(np.asarray(angle_deg, dtype=np.float64))) / 2
    data = np.zeros(half.shape + (4,))
    data[..., 0] = np.cos(half)
    data[..., 1:] = np.sin(half)[..., None] * axis
    return Rotation(data)


def _mu() -> Rotation:
    return Rotation.from_euler(np.deg2rad(MU_EULER_DEG))


def _crystal_map(
    rotations: Rotation | np.ndarray,
    shape: tuple[int, ...],
    is_in_data: np.ndarray | None = None,
) -> CrystalMap:
    """Return a nickel (m-3m) crystal map of step 1 um built with
    ``create_coordinate_arrays``.
    """
    if not isinstance(rotations, Rotation):
        rotations = Rotation(np.asarray(rotations, dtype=np.float64))
    coords, n = create_coordinate_arrays(shape, step_sizes=(1,) * len(shape))
    if is_in_data is None:
        is_in_data = np.ones(n, dtype=bool)
    return CrystalMap(
        rotations=rotations,
        phase_id=np.zeros(n, dtype=np.int32),
        x=coords.get("x"),
        y=coords.get("y"),
        phase_list=PhaseList(phases=[Phase("ni", point_group="m-3m")], ids=[0]),
        is_in_data=np.asarray(is_in_data, dtype=bool).ravel(),
        scan_unit="um",
    )


def _one_grain_row(rotations: Rotation) -> tuple[CrystalMap, np.ndarray]:
    """Return a (1, n) map of the rotations and labels of one grain."""
    n = rotations.size
    xmap = _crystal_map(Rotation(rotations.data.reshape(n, 4)), (1, n))
    return xmap, np.ones((1, n), dtype=np.int32)


def _sample_s3(
    mu: Rotation, kappa: float, n: int, kind: str, rng: np.random.Generator
) -> Rotation:
    """Return ``n`` exact samples of a von Mises-Fisher (``kind="vmf"``)
    or Watson (``kind="watson"``) density on the unit quaternions
    around ``mu``.

    The half angle is drawn by the inverse cumulative density on a grid
    of ``2**16 + 1`` points over [0, pi / 2] with the log density
    ``f(cos t) - f(1) + 2 log(sin t)``, ``f(t) = kappa t`` (von
    Mises-Fisher) or ``kappa t**2`` (Watson); the axis is uniform. The
    von Mises-Fisher mass beyond pi / 2 (below ``exp(-kappa)``) is
    dropped. The samples are ``mu * x`` (orix product).
    """
    theta = np.linspace(0, np.pi / 2, 2**16 + 1)
    t = np.cos(theta)
    f = kappa * t if kind == "vmf" else kappa * t**2
    with np.errstate(divide="ignore"):
        log_density = f - kappa + 2 * np.log(np.sin(theta))
    density = np.exp(log_density)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    half = np.interp(rng.random(n), cdf, theta)
    axis = rng.standard_normal((n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    data = np.column_stack([np.cos(half), np.sin(half)[:, None] * axis])
    return mu * Rotation(data)


def _operator_indices(sign_safe: bool = False) -> np.ndarray:
    """Return the indices of the proper m-3m operators that scramble
    samples: every operator, or, if ``sign_safe``, those whose variant
    ``S_j * mu`` has a scalar part of at least
    ``VMF_MIN_VARIANT_SCALAR``, so that no scrambled von Mises-Fisher
    sample changes sign when its scalar part is made non-negative.
    """
    if not sign_safe:
        return np.arange(OPERATORS.shape[0])
    scalar = _qmul(OPERATORS, _mu().data.reshape(1, 4))[:, 0]
    return np.flatnonzero(scalar >= VMF_MIN_VARIANT_SCALAR)


def _draw_operators(
    n: int, rng: np.random.Generator, sign_safe: bool = False
) -> np.ndarray:
    """Return ``n`` random operator indices drawn uniformly from
    ``_operator_indices(sign_safe)``.
    """
    indices = _operator_indices(sign_safe)
    return indices[rng.integers(indices.size, size=n)]


def _scramble(
    rotations: Rotation,
    side: str,
    rng: np.random.Generator,
    sign_safe: bool = False,
) -> tuple[Rotation, np.ndarray]:
    """Return the rotations multiplied by random proper m-3m operators
    from the left (``S_j * x``) or the right (``x * S_j``), and the
    operator indices. Operators are drawn from
    ``_operator_indices(sign_safe)``.
    """
    j = _draw_operators(rotations.size, rng, sign_safe)
    ops = OPERATORS[j]
    x = rotations.data.reshape(-1, 4)
    data = _qmul(ops, x) if side == "left" else _qmul(x, ops)
    return Rotation(data), j


def _flip_signs(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return the quaternions of shape (n, 4), each negated with
    probability 1/2 (one uniform draw per quaternion).
    """
    q = np.asarray(q, dtype=np.float64)
    negate = rng.random(q.shape[0]) < 0.5
    return np.where(negate[:, None], -q, q)


def _with_positive_scalar(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)


def _left_error_rad(rotation: Rotation, mu: Rotation) -> float:
    """Return the symmetry-aware angle in radians from orix."""
    a = Orientation(rotation.data.reshape(-1, 4), symmetry=Oh)
    b = Orientation(mu.data.reshape(-1, 4), symmetry=Oh)
    return float(np.asarray(a.angle_with(b)).ravel()[0])


def _right_class_error_rad(q: np.ndarray, mu: Rotation) -> float:
    """Return the angle in radians from ``q`` to the nearest member of
    the class ``{mu * S_j}``.
    """
    members = _qmul(mu.data.reshape(1, 4), OPERATORS)
    d = np.abs(members @ np.asarray(q, dtype=np.float64).reshape(4)).max()
    return float(2 * np.arccos(min(d, 1.0)))


def _band_rad(alpha: float, n: int) -> float:
    return RECOVERY_ANGLE_FACTOR * 2 / np.sqrt(8 * alpha * n)


def _kappa_band(n: int) -> float:
    return KAPPA_RATIO_BAND_N300 if n >= 300 else KAPPA_RATIO_BAND_N30


def _sampled_kappa(kind: str, alpha: float) -> float:
    return 8 * alpha if kind == "vmf" else 4 * alpha


def _orix_angles_deg(pixels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return orix' m-3m disorientation angles in degrees of each pixel
    quaternion to one reference quaternion.
    """
    a = Orientation(np.asarray(pixels).reshape(-1, 4), symmetry=Oh)
    b = Orientation(np.asarray(reference).reshape(1, 4), symmetry=Oh)
    return np.asarray(a.angle_with(b, degrees=True), dtype=np.float64).ravel()


def _assert_within_one_float32_ulp(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected32 = np.asarray(expected, dtype=np.float64).astype(np.float32)
    assert actual.shape == expected32.shape
    nan = np.isnan(expected32)
    assert np.array_equal(np.isnan(actual), nan)
    diff = np.abs(actual[~nan].astype(np.float64) - expected32[~nan])
    assert np.all(diff <= np.spacing(np.abs(expected32[~nan])).astype(np.float64))


def _emsoft_eq(euler: np.ndarray) -> np.ndarray:
    """Return EMsoft's ``eq_`` of Bunge Euler angles in radians."""
    euler = np.asarray(euler, dtype=np.float64)
    phi1, phi, phi2 = np.moveaxis(euler, -1, 0)
    c_phi = np.cos(phi * 0.5)
    s_phi = np.sin(phi * 0.5)
    cm = np.cos((phi1 - phi2) * 0.5)
    sm = np.sin((phi1 - phi2) * 0.5)
    cp = np.cos((phi1 + phi2) * 0.5)
    sp = np.sin((phi1 + phi2) * 0.5)
    q = np.stack([c_phi * cp, -(s_phi * cm), -(s_phi * sm), -(c_phi * sp)], axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def _compat_pixel_quaternion(xmap: CrystalMap, flat: int) -> np.ndarray:
    """Return EMsoft's quaternion of a map point: ``eq_`` of its Euler
    angles rounded through float32.
    """
    euler = xmap.rotations.to_euler()[flat].astype(np.float32).astype(np.float64)
    return _emsoft_eq(euler)


def _standalone_numbers(message: str, number: str) -> list[str]:
    """Return the occurrences of the integer ``number``, alone or with
    trailing zero decimals, not part of another number.
    """
    return re.findall(rf"(?<![\d.]){number}(?:\.0+)?(?![\d.])", message)


def _numbers_near(message: str, value: float, tol: float = 0.006) -> list[str]:
    """Return the decimal numbers in the message within ``tol`` of
    ``value``.
    """
    numbers = re.findall(r"\d+\.\d+", message)
    return [m for m in numbers if abs(float(m) - value) < tol]


def _user_warnings(record) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def _ball_warnings(record) -> list[str]:
    return [m for m in _user_warnings(record) if "misorientation ball" in m]


def _assert_tables_equal(a: GrainTable, b: GrainTable):
    np.testing.assert_array_equal(a.n_pixels, b.n_pixels)
    np.testing.assert_array_equal(a.bounding_box, b.bounding_box)
    np.testing.assert_array_equal(a.rotation.data, b.rotation.data)
    np.testing.assert_array_equal(a.phase_id, b.phase_id)
    np.testing.assert_array_equal(a.kappa, b.kappa)
    np.testing.assert_array_equal(a.max_grod, b.max_grod)
    np.testing.assert_array_equal(a.valid, b.valid)


# -------------- EMsoft expectation maximisation oracle --------------


def _emsoft_quatmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return EMsoft's quaternion product in its term order."""
    a0, a1, a2, a3 = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    b0, b1, b2, b3 = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            (a0 * b0 - a1 * b1) - (a2 * b2 + a3 * b3),
            (a0 * b1 + a1 * b0) + (a2 * b3 - a3 * b2),
            (a0 * b2 + a2 * b0) + (a3 * b1 - a1 * b3),
            (a0 * b3 + a3 * b0) + (a1 * b2 - a2 * b1),
        ],
        axis=-1,
    )


def _dot4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the four-term dot product summed left to right."""
    return ((a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]) + (
        a[..., 3] * b[..., 3]
    )


@functools.lru_cache(maxsize=None)
def _kappa_table(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return EMsoft's concentration table ``xAp`` and the mean
    resultant length (von Mises-Fisher) or top eigenvalue (Watson)
    ``yAp``, from Bessel function ratios of scaled Bessel functions.
    """
    i = np.arange(1, 35001)
    x = 0.001 + (i - 1).astype(np.float64) * 0.001
    if kind == "vmf":
        y = ive(2, x) / ive(1, x)
    else:
        y1 = ive(1, x * 0.5)
        y2 = ive(0, x * 0.5)
        y = y1 / (y2 - y1) / x
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


def _kappa_oracle(y: float, kind: str) -> float:
    """Return EMsoft's concentration of ``y`` (``Mstep_``)."""
    if y >= 0.94:
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "vmf":
                return float(
                    (15.0 - 3.0 * y + np.sqrt(15.0 + 90.0 * y + 39.0 * y * y))
                    / (16.0 * (1.0 - y))
                )
            return float(
                (5.0 * y - 11.0 - np.sqrt(39.0 - 12.0 * y + 9.0 * y**2))
                / (8.0 * (y - 1.0))
            )
    x_ap, y_ap = _kappa_table(kind)
    p = int(np.argmin(np.abs(y - y_ap)))
    if p == 0:
        p = 1
    return float(x_ap[p])


def _log_cp_oracle(kappa: float, kind: str) -> float:
    """Return EMsoft's ``logCp_``."""
    if kind == "vmf":
        if kappa > 30.0:
            lcp = kappa**4.5 / (
                -105.0 + 8.0 * kappa * (-15.0 + 16.0 * kappa * (-3.0 + 8.0 * kappa))
            )
            return C2 - kappa + np.log(lcp)
        return C + np.log(kappa / i1(kappa))
    if kappa > 20.0:
        lcp = kappa**4.5 / (
            525.0 + 4.0 * kappa * (45.0 + 8.0 * kappa * (3.0 + 4.0 * kappa))
        )
        return C2W - kappa + np.log(lcp)
    return -kappa * 0.5 - np.log(i0(kappa * 0.5) - i1(kappa * 0.5))


def _emsoft_density(centres, x, kappa, c, kind):
    """Return EMsoft's ``Density_`` of every sample (rows) for every
    centre (columns).
    """
    t = _dot4(centres[None, :, :], x[:, None, :])
    f = t if kind == "vmf" else t**2
    return np.exp(c + kappa * f)


def _emsoft_em(x, operators, kind, n_em, n_iter, rng, trace=None):
    """Return ``(mu, kappa, log_likelihood, n_iterations, best_init)``
    of EMsoft's expectation maximisation, transcribed from ``EMforDS_``.

    The E-step centres are ``mu * S_j`` and the M-step vectors ``x_n *
    conj(S_j)`` (operators from the right); ``getQandL_`` uses ``S_j *
    mu`` (from the left), the exponential of the log normalising
    constant for the von Mises-Fisher density, and keeps the previous
    ``Q`` and ``L`` (0.0 before the first iteration of the first
    initial guess, then carried over iterations and initial guesses)
    when the smallest density is not positive. If ``trace`` is a list,
    one list per initial guess of ``(Q, L, underflow)`` per iteration
    is appended.
    """
    x = np.asarray(x, dtype=np.float64)
    operators = np.asarray(operators, dtype=np.float64)
    n = x.shape[0]
    n_ops = operators.shape[0]
    v = _emsoft_quatmult(x[:, None, :], _conj(operators)[None, :, :])

    mu_all = np.zeros((n_em, 4))
    kappa_all = np.zeros(n_em)
    l_all = np.zeros(n_em)
    n_iterations = np.zeros(n_em, dtype=np.int64)
    q_i = 0.0
    l_i = 0.0
    with np.errstate(all="ignore"):
        for init in range(n_em):
            mu = rng.standard_normal(4)
            mu = mu / np.sqrt(np.sum(mu**2))
            if mu[0] < 0:
                mu = -mu
            kappa = 30.0
            q_hist = np.zeros(n_iter)
            init_trace = []
            for i in range(n_iter):
                # E-step
                c = _log_cp_oracle(kappa, kind)
                centres = _emsoft_quatmult(mu[None, :], operators)
                r = _emsoft_density(centres, x, kappa, c, kind)
                r_denom = 1.0 / np.sum(r, axis=1)
                r = r * r_denom[:, None]
                # M-step
                if kind == "vmf":
                    gamma = np.sum(r[..., None] * v, axis=(0, 1))
                    n_gamma = np.sqrt(np.sum(gamma**2))
                    mu_new = gamma / n_gamma
                    y = n_gamma / n
                else:
                    t_scatt = np.einsum("nm,nmi,nmj->ij", r, v, v) / n
                    _, vectors = np.linalg.eigh(t_scatt)
                    mu_new = vectors[:, 3]
                    y = float(mu_new @ (t_scatt @ mu_new))
                kappa_new = _kappa_oracle(float(y), kind)
                # Q and L
                old_q, old_l = q_i, l_i
                c = _log_cp_oracle(kappa_new, kind)
                if kind == "vmf":
                    c = np.exp(c)
                centres = _emsoft_quatmult(operators, mu_new[None, :])
                phi = _emsoft_density(centres, x, kappa_new, c, kind) / n_ops
                underflow = not np.min(phi) > 0.0
                if not underflow:
                    l_i = float(np.sum(np.log(np.sum(phi, axis=1))))
                    q_i = float(np.sum(r * np.log(phi)))
                else:
                    l_i, q_i = old_l, old_q
                q_hist[i] = q_i
                init_trace.append((q_i, l_i, underflow))

                mu_all[init] = mu_new
                kappa_all[init] = kappa_new
                l_all[init] = l_i
                mu = mu_new
                kappa = kappa_new
                n_iterations[init] = i + 1
                if i >= 1 and abs(q_hist[i] - q_hist[i - 1]) < 0.01:
                    break
            if trace is not None:
                trace.append(init_trace)

    if np.all(np.isnan(l_all)):
        best = 0
    else:
        best = int(np.nanargmax(l_all))
    mu = mu_all[best].copy()
    if mu[0] < 0:
        mu = -mu
    # The variant mu * S_i with the largest absolute scalar part, the
    # first on ties, with a non-negative scalar part
    variants = _emsoft_quatmult(mu[None, :], operators)
    k = int(np.argmax(np.abs(variants[:, 0])))
    mu_final = variants[k]
    if mu_final[0] < 0:
        mu_final = -mu_final
    return mu_final, float(kappa_all[best]), l_all, n_iterations, best


def _assert_em_result_equals(result, expected):
    mu, kappa, log_likelihood, n_iterations, best = expected
    np.testing.assert_array_equal(np.asarray(result.mu), mu)
    np.testing.assert_array_equal(np.asarray(result.kappa), kappa)
    np.testing.assert_array_equal(np.asarray(result.log_likelihood), log_likelihood)
    np.testing.assert_array_equal(np.asarray(result.n_iterations), n_iterations)
    assert result.best_init == best


def _compat_samples(alpha: float, n: int, kind: str) -> np.ndarray:
    """Return LEFT-scrambled samples of ``default_rng(80)`` with a
    non-negative scalar part, shape (n, 4).
    """
    rng = np.random.default_rng(80)
    samples = _sample_s3(_mu(), _sampled_kappa(kind, alpha), n, kind, rng)
    scrambled, _ = _scramble(samples, "left", rng)
    return _with_positive_scalar(scrambled.data.reshape(-1, 4))


def _recovery_params() -> list:
    """Return the recovery arms: the default arms run in every suite,
    the rest of the grid weekly.
    """
    default = {
        ("mean", 100, 300),
        ("vmf", 100, 300),
        ("watson", 100, 300),
        ("watson", 1000, 30),
        ("mean", 1000, 30),
    }
    params = []
    for method in ("mean", "vmf", "watson"):
        for alpha in (20, 100, 1000):
            for n in (30, 300):
                marks = () if (method, alpha, n) in default else pytest.mark.weekly
                params.append(
                    pytest.param(
                        method, alpha, n, id=f"{method}-alpha{alpha}-n{n}", marks=marks
                    )
                )
    return params


def _compat_em_params() -> list:
    """Return the arms compared with the transcription: three initial
    guesses in every suite, 25 weekly.
    """
    cases = {"alpha20-n30": (20, 30), "alpha1000-n50": (1000, 50)}
    params = []
    for kind in ("vmf", "watson"):
        for case, (alpha, n) in cases.items():
            params.append(pytest.param(kind, alpha, n, N_EM_FAST, id=f"{kind}-{case}"))
            params.append(
                pytest.param(
                    kind,
                    alpha,
                    n,
                    25,
                    id=f"{kind}-{case}-n_em25",
                    marks=pytest.mark.weekly,
                )
            )
    return params


# ------------------------------ Tests -------------------------------


class TestCenterPixel:
    def test_compat_center_is_the_box_centre_rounded_up(self, hrosm_gradient_xmap):
        xmap = hrosm_gradient_xmap((4, 5))
        grain_id = np.full((4, 5), 2, dtype=np.int32)
        grain_id[:3, :4] = 1
        # 1-based x0 = 1, w = 4, y0 = 1, h = 3 picks the 1-based (x, y)
        # = (3, 2), the 0-based (row, column) = (1, 2)
        box = np.array([0, 0, 3, 4], dtype=np.int64)
        center = _center_pixel(grain_id, 1, box, True)
        assert tuple(int(v) for v in center) == (1, 2)

        table = average_grain_orientations(
            xmap, grain_id, method="center", emsoft_compatible=True
        )
        expected = _compat_pixel_quaternion(xmap, 1 * 5 + 2)
        np.testing.assert_array_equal(table.rotation.data[0], expected)
        assert table.kappa[0] == 1.0
        assert table.valid[0]

    def test_compat_center_may_lie_outside_the_grain(self, hrosm_gradient_xmap):
        xmap = hrosm_gradient_xmap((5, 5))
        grain_id = np.ones((5, 5), dtype=np.int32)
        grain_id[1:4, 1:4] = 2
        box = np.array([0, 0, 5, 5], dtype=np.int64)

        compat = tuple(int(v) for v in _center_pixel(grain_id, 1, box, True))
        assert compat == (2, 2)
        assert grain_id[compat] == 2

        correct = tuple(int(v) for v in _center_pixel(grain_id, 1, box, False))
        assert grain_id[correct] == 1
        # Centroid (2, 2); four ring points at distance 2, the lowest
        # flat index wins
        assert correct == (0, 2)

        table = average_grain_orientations(
            xmap, grain_id, method="center", emsoft_compatible=True
        )
        expected = _compat_pixel_quaternion(xmap, 2 * 5 + 2)
        np.testing.assert_array_equal(table.rotation.data[0], expected)

    @pytest.mark.parametrize(
        "pixels, expected",
        [
            # A 1 x 2 grain: the centroid ties both, the first wins
            ([(3, 2), (3, 3)], (3, 2)),
            # A 2 x 2 block: four ties, the first wins
            ([(1, 1), (1, 2), (2, 1), (2, 2)], (1, 1)),
            # An L-shaped grain at rows 2-5, columns 1-3: centroid (4,
            # 1.5), nearest point (4, 1)
            ([(2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3)], (4, 1)),
            # A plus sign: its centre
            ([(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)], (2, 2)),
        ],
        ids=["pair", "block", "l-shape", "plus"],
    )
    def test_correct_center_is_the_grain_pixel_nearest_the_centroid(
        self, pixels, expected
    ):
        grain_id = np.zeros((7, 6), dtype=np.int32)
        rows, cols = np.array(pixels).T
        grain_id[rows, cols] = 1
        box = np.array(
            [rows.min(), cols.min(), np.ptp(rows) + 1, np.ptp(cols) + 1],
            dtype=np.int64,
        )

        # Test-local brute force: raster order, first minimum
        points = np.argwhere(grain_id == 1)
        centroid = points.mean(axis=0)
        d2 = np.sum((points - centroid) ** 2, axis=1)
        assert tuple(points[np.argmin(d2)]) == expected

        center = _center_pixel(grain_id, 1, box, False)
        assert tuple(int(v) for v in center) == expected

    @pytest.mark.parametrize("emsoft_compatible", [False, True])
    def test_center_kappa_is_one_and_never_gated(
        self, hrosm_grain_xmap, emsoft_compatible
    ):
        xmap, truth = hrosm_grain_xmap((6, 8))
        for method in ("center", "mean"):
            table = average_grain_orientations(
                xmap,
                truth,
                method=method,
                min_kappa=1e9,
                emsoft_compatible=emsoft_compatible,
            )
            assert np.all(table.valid)
            assert np.all(table.kappa != -1.0)
            if method == "center":
                assert np.all(table.kappa == 1.0)


class TestRecovery:
    def test_sampler_matches_orix_random_vonmises(self):
        state = np.random.get_state()
        try:
            np.random.seed(0)
            theirs = Rotation.random_vonmises(300, alpha=20.0)
        finally:
            np.random.set_state(state)
        ours = _sample_s3(
            Rotation.identity(), 80.0, 3000, "watson", np.random.default_rng(70)
        )
        cos_theirs = np.cos(theirs.angle)
        cos_ours = np.cos(ours.angle)
        standard_error = cos_theirs.std(ddof=1) / np.sqrt(cos_theirs.size)
        assert abs(cos_ours.mean() - cos_theirs.mean()) <= 4 * standard_error

    @pytest.mark.parametrize("method, alpha, n", _recovery_params())
    def test_recovers_left_scrambled_variants(self, method, alpha, n):
        rng = np.random.default_rng(70)
        kind = "vmf" if method == "vmf" else "watson"
        kappa = _sampled_kappa(kind, alpha)
        mu = _mu()
        samples = _sample_s3(mu, kappa, n, kind, rng)
        scrambled, _ = _scramble(samples, "left", rng)
        if kind == "vmf":
            # q and -q are one orientation: von Mises-Fisher samples get
            # random signs, so both signs occur
            flipped = _flip_signs(scrambled.data.reshape(-1, 4), rng)
            assert 0 < np.mean(flipped[:, 0] < 0) < 1
            scrambled = Rotation(flipped)
        xmap, grain_id = _one_grain_row(scrambled)

        table = average_grain_orientations(
            xmap,
            grain_id,
            method=method,
            max_angle=90.0,
            n_em=25,
            n_iter=40,
            seed=0,
        )
        assert table.valid[0]
        assert _left_error_rad(table.rotation[0], mu) <= _band_rad(alpha, n)
        if method != "mean":
            band = _kappa_band(n)
            assert 1 / band <= table.kappa[0] / kappa <= band

    @pytest.mark.parametrize("method", ["mean", "vmf", "watson"])
    def test_right_scrambled_variants_are_not_recovered(self, method):
        rng = np.random.default_rng(70)
        kind = "vmf" if method == "vmf" else "watson"
        samples = _sample_s3(_mu(), _sampled_kappa(kind, 100), 300, kind, rng)
        j = _draw_operators(300, rng)
        x = samples.data.reshape(-1, 4)
        left = _qmul(OPERATORS[j], x)
        right = _qmul(x, OPERATORS[j])
        if kind == "vmf":
            # The same random signs on both sides
            negate = rng.random(300) < 0.5
            left = np.where(negate[:, None], -left, left)
            right = np.where(negate[:, None], -right, right)
        left = Rotation(left)
        right = Rotation(right)

        kappas = {}
        for side, rotations in (("left", left), ("right", right)):
            xmap, grain_id = _one_grain_row(rotations)
            table = average_grain_orientations(
                xmap,
                grain_id,
                method=method,
                max_angle=90.0,
                n_em=25,
                n_iter=40,
                min_kappa=0.0,
                seed=0,
            )
            kappas[side] = table.kappa[0]
        assert kappas["left"] > 0
        assert kappas["right"] <= WRONG_SIDE_MAX_KAPPA_RATIO * kappas["left"]

    def test_vmf_treats_q_and_minus_q_as_one_orientation(self):
        rng = np.random.default_rng(70)
        kappa = _sampled_kappa("vmf", 100)
        mu = _mu()
        samples = _sample_s3(mu, kappa, 300, "vmf", rng)
        scrambled, _ = _scramble(samples, "left", rng)
        x = scrambled.data.reshape(-1, 4)
        flipped = _flip_signs(x, rng)
        changed = np.any(flipped != x, axis=1)
        assert 0.3 < np.mean(changed) < 0.7

        tables = []
        for data in (x, flipped):
            xmap, grain_id = _one_grain_row(Rotation(data))
            tables.append(
                average_grain_orientations(
                    xmap,
                    grain_id,
                    method="vmf",
                    max_angle=90.0,
                    n_em=25,
                    n_iter=40,
                    min_kappa=0.0,
                    seed=0,
                )
            )
        plain, signed = tables
        assert plain.valid[0]
        assert signed.valid[0]
        # The same mean up to sign (the angle from the chord, accurate
        # near 0, unlike the arc cosine of the dot product) and the same
        # concentration
        a = plain.rotation.data[0]
        b = signed.rotation.data[0]
        b = b if a @ b >= 0 else -b
        angle = 4 * np.arcsin(min(np.linalg.norm(a - b) / 2, 1.0))
        assert angle <= 1e-8
        assert signed.kappa[0] == pytest.approx(plain.kappa[0], rel=1e-8, abs=0)
        # Both recover the sampled mean and concentration
        assert _left_error_rad(plain.rotation[0], mu) <= _band_rad(100, 300)
        band = KAPPA_RATIO_BAND_N300
        assert 1 / band <= plain.kappa[0] / kappa <= band

    def test_watson_is_antipodally_symmetric(self, record_property):
        rng = np.random.default_rng(70)
        mu = _axis_angle((1, 2, 3), 179.8)
        samples = _sample_s3(mu, 400.0, 300, "watson", rng)
        # About half of the samples change sign when the scalar part is
        # made non-negative
        negative = np.mean(samples.data.reshape(-1, 4)[:, 0] < 0)
        assert 0.3 < negative < 0.7
        xmap, grain_id = _one_grain_row(samples)

        watson = average_grain_orientations(
            xmap, grain_id, method="watson", max_angle=90.0, seed=0
        )
        assert watson.valid[0]
        assert _left_error_rad(watson.rotation[0], mu) <= _band_rad(100, 300)
        band = KAPPA_RATIO_BAND_N300
        assert 1 / band <= watson.kappa[0] / 400.0 <= band

        vmf = average_grain_orientations(
            xmap, grain_id, method="vmf", max_angle=90.0, min_kappa=0.0, seed=0
        )
        record_property(
            "vmf_error_deg", float(np.rad2deg(_left_error_rad(vmf.rotation[0], mu)))
        )

    def test_mean_draws_nothing(self):
        rng = np.random.default_rng(70)
        samples = _sample_s3(_mu(), 400.0, 50, "watson", rng)
        scrambled, _ = _scramble(samples, "left", rng)
        xmap, grain_id = _one_grain_row(scrambled)

        tables = [
            average_grain_orientations(
                xmap, grain_id, method="mean", max_angle=90.0, seed=seed
            )
            for seed in (0, 1, None)
        ]
        _assert_tables_equal(tables[0], tables[1])
        _assert_tables_equal(tables[0], tables[2])

        generator = np.random.default_rng(5)
        state = generator.bit_generator.state
        table = average_grain_orientations(
            xmap, grain_id, method="mean", max_angle=90.0, seed=generator
        )
        _assert_tables_equal(tables[0], table)
        assert generator.bit_generator.state == state

    def test_one_generator_is_consumed_across_grains_in_label_order(
        self, hrosm_grain_xmap
    ):
        xmap, truth = hrosm_grain_xmap((6, 8))
        kwargs = {"method": "watson", "n_em": N_EM_FAST, "n_iter": 40}
        both = average_grain_orientations(xmap, truth, seed=0, **kwargs)

        only_second = (truth == 2).astype(np.int32)
        rng = np.random.default_rng(0)
        rng.standard_normal((N_EM_FAST, 4))
        second = average_grain_orientations(xmap, only_second, seed=rng, **kwargs)
        np.testing.assert_array_equal(both.rotation.data[1], second.rotation.data[0])
        assert both.kappa[1] == second.kappa[0]

        # A fresh generator for the second grain gives other draws
        fresh = average_grain_orientations(xmap, only_second, seed=0, **kwargs)
        assert not (
            np.array_equal(fresh.rotation.data[0], both.rotation.data[1])
            and fresh.kappa[0] == both.kappa[1]
        )


class TestGROD:
    @pytest.mark.parametrize("emsoft_compatible", [False, True])
    @pytest.mark.parametrize("method", ["mean", "center", "vmf", "watson"])
    def test_max_grod_is_the_largest_angle_to_the_grain_reference(
        self, hrosm_grain_xmap, method, emsoft_compatible
    ):
        xmap, truth = hrosm_grain_xmap((6, 8))
        table = average_grain_orientations(
            xmap,
            truth,
            method=method,
            n_em=N_EM_FAST,
            seed=0,
            emsoft_compatible=emsoft_compatible,
        )
        assert table.max_grod.dtype == np.float32
        pixels = xmap.rotations.data.reshape(-1, 4)
        labels = truth.ravel()
        for g in (1, 2):
            if not table.valid[g - 1]:
                assert np.isnan(table.max_grod[g - 1])
                continue
            reference = table.rotation.data[g - 1]
            angles = _orix_angles_deg(pixels[labels == g], reference)
            _assert_within_one_float32_ulp(table.max_grod[g - 1], angles.max())

    @staticmethod
    def _row_grains(spreads_deg) -> tuple[CrystalMap, np.ndarray]:
        """Return a map of one 1 x 3 grain per row, the outer points
        rotated by -spread and +spread degrees about z from the middle.
        """
        n = len(spreads_deg)
        data = np.zeros((n, 3, 4))
        for k, spread in enumerate(spreads_deg):
            g = Rotation.from_euler(np.deg2rad((10.0 + 7.0 * k, 20.0, 30.0)))
            for col, angle in enumerate((-spread, 0.0, spread)):
                data[k, col] = (_axis_angle((0, 0, 1), angle) * g).data.reshape(4)
        xmap = _crystal_map(data.reshape(-1, 4), (n, 3))
        grain_id = np.repeat(np.arange(1, n + 1, dtype=np.int32), 3).reshape(n, 3)
        return xmap, grain_id

    def test_warning_lists_the_largest_ten_in_descending_max_grod(self):
        spreads = [3.0 + 0.1 * k for k in range(1, 13)]
        xmap, grain_id = self._row_grains(spreads)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            table = average_grain_orientations(
                xmap, grain_id, method="center", max_angle=3
            )
        _assert_within_one_float32_ulp(table.max_grod, spreads)

        messages = _user_warnings(record)
        assert len(messages) == 1
        message = messages[0]
        assert "misorientation ball" in message
        assert "max GROD" in message
        # 12 appears as the count and as the label of grain 12; 3 as
        # max_angle, as the label of grain 3 and as each of the ten
        # n_pixels
        assert len(_standalone_numbers(message, "12")) >= 2
        assert len(_standalone_numbers(message, "3")) >= 12
        # The standalone warning is the helper's message without the
        # spacing clause
        assert message == _coverage_warning_message(
            np.arange(1, 13), table.max_grod, table.n_pixels, 3
        )

        # The max GROD values in the message, matched to their grains,
        # first occurrences in descending order, grains 1 and 2 absent
        found = []
        for value in re.findall(r"\d+\.\d+", message):
            k = int(round((float(value) - 3.0) * 10))
            if 1 <= k <= 12 and abs(float(value) - spreads[k - 1]) < 0.006:
                if k not in found:
                    found.append(k)
        assert found == list(range(12, 2, -1))
        # Each entry lists the label, its max GROD and 3 points
        for k in range(12, 2, -1):
            pattern = rf"\b{k}\b\D+?(\d+\.\d+)\D+?\b3\b"
            matches = [
                m
                for m in re.finditer(pattern, message)
                if abs(float(m.group(1)) - spreads[k - 1]) < 0.006
            ]
            assert matches, k

    def test_coverage_warning_message_contract(self):
        # 14 grains: three not above max_angle 2.75 (equality does not
        # count), eleven above, two pairs of them tied; labels 101 to
        # 114 and n_pixels 201 to 214 collide with no other number
        labels = np.arange(101, 115)
        max_grod = np.array(
            [1.0, 2.0, 2.75, 3.37, 4.37, 3.37, 3.11, 3.62, 3.62]
            + [2.91, 3.83, 3.05, 4.09, 2.97],
            dtype=np.float32,
        )
        n_pixels = np.arange(201, 215)
        max_angle = 2.75

        above = max_grod.astype(np.float64) > max_angle
        assert np.count_nonzero(above) == 11
        # Descending max GROD, ties by ascending label, the first ten
        order = sorted(np.flatnonzero(above), key=lambda i: (-max_grod[i], labels[i]))
        listed = [int(labels[i]) for i in order[:10]]
        assert listed[:2] == [105, 113]
        assert listed.index(104) < listed.index(106)
        assert listed.index(108) < listed.index(109)
        absent = [int(labels[i]) for i in order[10:]] + [101, 102, 103]

        message = _coverage_warning_message(labels, max_grod, n_pixels, max_angle)
        assert isinstance(message, str)
        assert "misorientation ball" in message
        assert "max GROD" in message
        # The count of grains concerned and max_angle
        assert _standalone_numbers(message, "11")
        assert _numbers_near(message, 2.75, 1e-9)
        # The largest max GROD and the hint rounded up to 0.5 degrees
        assert _numbers_near(message, 4.37)
        assert re.search(r"max_angle\s*>=\s*4\.50*(?![\d.])", message)
        # The entries: label, max GROD and n_pixels, in the order above
        positions = []
        for label in listed:
            i = int(np.flatnonzero(labels == label)[0])
            pattern = rf"(?<![\d.]){label}(?![\d.])\D+?(\d+\.\d+)\D+?(\d+)"
            matches = [
                m
                for m in re.finditer(pattern, message)
                if abs(float(m.group(1)) - float(max_grod[i])) < 0.006
                and int(m.group(2)) == n_pixels[i]
            ]
            assert matches, label
            positions.append(matches[0].start())
        assert positions == sorted(positions)
        for label in absent:
            assert not re.search(rf"(?<![\d.]){label}(?![\d.])", message), label

        # The spacing clause only when the spacing is given
        assert not _numbers_near(message, 0.4321)
        with_spacing = _coverage_warning_message(
            labels, max_grod, n_pixels, max_angle, spacing=0.4321
        )
        assert with_spacing != message
        assert _numbers_near(with_spacing, 0.4321)

        # None unless some max GROD is strictly above max_angle,
        # compared in float64
        first = slice(0, 3)
        assert (
            _coverage_warning_message(
                labels[first], max_grod[first], n_pixels[first], 2.75
            )
            is None
        )
        below = float(np.nextafter(max_grod[2], np.float32(0)))
        message_below = _coverage_warning_message(
            labels[first], max_grod[first], n_pixels[first], below
        )
        assert isinstance(message_below, str)

    def test_no_warning_within_max_angle(self):
        xmap, grain_id = self._row_grains([1.0])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            table = average_grain_orientations(
                xmap, grain_id, method="center", max_angle=3
            )
        assert _ball_warnings(record) == []
        _assert_within_one_float32_ulp(table.max_grod, [1.0])

    def test_warning_is_strict_at_max_angle(self):
        g = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0)))
        data = np.stack([g.data.reshape(4), (_axis_angle((0, 0, 1), 2.0) * g).data[0]])
        xmap = _crystal_map(data, (1, 2))
        grain_id = np.ones((1, 2), dtype=np.int32)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            table = average_grain_orientations(
                xmap, grain_id, method="center", max_angle=90.0
            )
        assert _ball_warnings(record) == []
        m = table.max_grod.max()
        assert m.dtype == np.float32
        assert m > 0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            at = average_grain_orientations(
                xmap, grain_id, method="center", max_angle=float(m)
            )
        assert _ball_warnings(record) == []
        np.testing.assert_array_equal(at.max_grod, table.max_grod)

        below = float(np.nextafter(m, np.float32(0)))
        assert below < float(m)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            average_grain_orientations(xmap, grain_id, method="center", max_angle=below)
        assert len(_ball_warnings(record)) == 1

    @staticmethod
    def _below_one_rotation() -> np.ndarray:
        """Return the first rotation of float32 Euler triples of
        ``default_rng(13)`` whose float64 self dot product, the largest
        ``|<S_j o, o>|`` over the proper m-3m operators, is below 1.
        """
        rng = np.random.default_rng(13)
        scale = np.array([2 * np.pi, np.pi, 2 * np.pi])
        eu32 = (rng.random((1000, 3)) * scale).astype(np.float32)
        q = Rotation.from_euler(eu32.astype(np.float64)).data
        variants = _qmul(OPERATORS[None, :, :], q[:, None, :])
        # Summed by einsum: about 5 % of these rotations round below 1
        # and a third above
        self_dot = np.abs(np.einsum("njk,nk->nj", variants, q)).max(axis=1)
        below = np.flatnonzero(self_dot < 1)
        assert below.size > 0
        return q[below[0]]

    def test_correct_center_pixel_has_zero_grod(self):
        below = self._below_one_rotation()
        tilted = _qmul(_axis_angle((0, 0, 1), 0.5).data.reshape(4), below)
        other = Rotation.from_euler(np.deg2rad((80.0, 40.0, 10.0))).data.reshape(4)
        other_tilted = _qmul(_axis_angle((0, 0, 1), 0.3).data.reshape(4), other)

        data = np.zeros((3, 5, 4))
        data[:, :3] = tilted
        data[1, 1] = below
        data[:, 3:] = other_tilted
        data[1, 3] = other
        xmap = _crystal_map(data.reshape(-1, 4), (3, 5))
        grain_id = np.ones((3, 5), dtype=np.int32)
        grain_id[:, 3:] = 2

        table = average_grain_orientations(xmap, grain_id, method="center")
        np.testing.assert_array_equal(
            _with_positive_scalar(table.rotation.data[0]), _with_positive_scalar(below)
        )
        grod = grain_reference_orientation_deviation_map(xmap, grain_id, table)
        assert grod.dtype == np.float32
        assert grod.shape == (3, 5)
        # Grain 1: the centre of the 3 x 3 block; grain 2: the 3 x 2
        # block's centroid ties (1, 3) and (1, 4), the first wins
        assert grod[1, 1] == 0.0
        assert grod[1, 3] == 0.0
        assert np.all(grod[grain_id == 1][np.arange(9) != 4] > 0.4)

    def test_compat_center_grod_is_measured_from_the_box_centre_pixel(
        self, hrosm_gradient_xmap
    ):
        xmap = hrosm_gradient_xmap((5, 5))
        grain_id = np.ones((5, 5), dtype=np.int32)
        grain_id[1:4, 1:4] = 2

        table = average_grain_orientations(
            xmap, grain_id, method="center", emsoft_compatible=True
        )
        grod = grain_reference_orientation_deviation_map(xmap, grain_id, table)

        pixels = xmap.rotations.data.reshape(-1, 4)
        ring = grain_id.ravel() == 1
        box_centre = _compat_pixel_quaternion(xmap, 2 * 5 + 2)
        expected = _orix_angles_deg(pixels[ring], box_centre)
        _assert_within_one_float32_ulp(grod.ravel()[ring], expected)

        # The centroid-nearest ring point (0, 2) gives another map
        nearest = _orix_angles_deg(pixels[ring], pixels[0 * 5 + 2])
        assert np.max(np.abs(nearest - expected)) > 0.1

    @pytest.mark.parametrize("method", ["mean", "center"])
    def test_variant_scrambled_pixels_have_the_same_grod(
        self, hrosm_gradient_xmap, method
    ):
        grain_id = np.ones((4, 5), dtype=np.int32)
        grain_id[:, 3:] = 2
        maps = []
        for scramble_seed in (None, 11):
            xmap = hrosm_gradient_xmap((4, 5), scramble_seed=scramble_seed)
            table = average_grain_orientations(xmap, grain_id, method=method)
            maps.append(
                grain_reference_orientation_deviation_map(xmap, grain_id, table)
            )
        assert np.all(np.isfinite(maps[0]))
        assert np.nanmax(maps[0]) < 5
        _assert_within_one_float32_ulp(maps[1], maps[0])

    def test_deviation_map_and_max_grod_agree_bitwise(self, hrosm_gradient_xmap):
        # Absent points (row, column) (0, 3), (2, 0) and (2, 6)
        absent = (3, 14, 20)
        xmap = hrosm_gradient_xmap((6, 7), absent=absent)
        grain_id = np.zeros((6, 7), dtype=np.int32)
        grain_id[:, 0:2] = 1
        grain_id[:, 2:4] = 2
        grain_id[:, 4:6] = 3

        table = average_grain_orientations(xmap, grain_id, method="mean")
        grod = grain_reference_orientation_deviation_map(xmap, grain_id, table)
        assert grod.dtype == np.float32
        assert grod.shape == (6, 7)
        for g in (1, 2, 3):
            assert np.nanmax(grod[grain_id == g]) == table.max_grod[g - 1]
            assert np.nanmax(grod[grain_id == g]).dtype == np.float32
        assert np.all(np.isnan(grod[grain_id == 0]))
        assert np.all(np.isnan(grod.ravel()[list(absent)]))
        present_in_grains = (grain_id > 0).ravel()
        present_in_grains[list(absent)] = False
        assert np.all(np.isfinite(grod.ravel()[present_in_grains]))

        # An invalid grain is NaN; the rest of the map is unchanged
        invalid = dataclasses.replace(table, valid=np.array([True, False, True]))
        grod_invalid = grain_reference_orientation_deviation_map(
            xmap, grain_id, invalid
        )
        assert np.all(np.isnan(grod_invalid[grain_id == 2]))
        keep = grain_id != 2
        np.testing.assert_array_equal(grod_invalid[keep], grod[keep])

        # All grains rejected by the concentration gate
        gated = average_grain_orientations(
            xmap, grain_id, method="vmf", n_em=N_EM_FAST, min_kappa=1e9, seed=0
        )
        assert not np.any(gated.valid)
        assert np.all(np.isnan(gated.max_grod))
        grod_gated = grain_reference_orientation_deviation_map(xmap, grain_id, gated)
        assert np.all(np.isnan(grod_gated))


class TestGrainTable:
    def test_fields_dtypes_and_label_order(self, hrosm_gradient_xmap):
        xmap = hrosm_gradient_xmap((6, 7))
        grain_id = np.zeros((6, 7), dtype=np.int32)
        grain_id[:, 0:1] = 3
        grain_id[:, 1:3] = 1
        grain_id[:, 3:7] = 2

        table = average_grain_orientations(xmap, grain_id, method="mean")
        assert isinstance(table, GrainTable)
        assert table.n_grains == 3
        assert table.method == "mean"

        assert table.n_pixels.dtype == np.int64
        assert table.n_pixels.shape == (3,)
        np.testing.assert_array_equal(table.n_pixels, [12, 24, 6])

        assert table.bounding_box.dtype == np.int64
        assert table.bounding_box.shape == (3, 4)
        np.testing.assert_array_equal(
            table.bounding_box, [[0, 1, 6, 2], [0, 3, 6, 4], [0, 0, 6, 1]]
        )

        assert isinstance(table.rotation, Rotation)
        assert table.rotation.shape == (3,)
        assert table.phase_id.dtype == np.int32
        np.testing.assert_array_equal(table.phase_id, [0, 0, 0])
        assert table.kappa.dtype == np.float64
        assert table.kappa.shape == (3,)
        assert table.max_grod.dtype == np.float32
        assert table.max_grod.shape == (3,)
        assert table.valid.dtype == np.bool_
        assert table.valid.shape == (3,)

        # Label order: grain 3 (column 0) has the smallest spread
        assert table.max_grod[2] < table.max_grod[1]

        with pytest.raises(dataclasses.FrozenInstanceError):
            table.method = "center"

    def test_invalid_grains_are_identity_with_nan_max_grod(self, hrosm_grain_xmap):
        xmap, truth = hrosm_grain_xmap((6, 8))
        kept = average_grain_orientations(
            xmap, truth, method="vmf", n_em=N_EM_FAST, min_kappa=0.0, seed=0
        )
        np.testing.assert_array_equal(kept.valid, kept.kappa != -1)
        assert np.all(kept.valid)

        rejected = average_grain_orientations(
            xmap, truth, method="vmf", n_em=N_EM_FAST, min_kappa=1e9, seed=0
        )
        np.testing.assert_array_equal(rejected.valid, rejected.kappa != -1)
        assert not np.any(rejected.valid)
        assert np.all(rejected.kappa == -1.0)
        assert np.all(np.isnan(rejected.max_grod))
        np.testing.assert_array_equal(
            rejected.rotation.data, np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        )

    def test_center_kappa_is_one(self, hrosm_grain_xmap):
        xmap, truth = hrosm_grain_xmap((6, 8))
        table = average_grain_orientations(xmap, truth, method="center")
        assert np.all(table.kappa == 1.0)
        assert np.all(table.valid)

    def test_from_crystal_map_round_trip(self):
        grain_id = np.array([[1, 1, 3, 3], [1, 0, 3, 4], [0, 0, 4, 4]], dtype=np.int32)
        r1 = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0))).data.reshape(4)
        r4 = Rotation.from_euler(np.deg2rad((50.0, 60.0, 70.0))).data.reshape(4)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        # Label 2 has vanished (no point); label 3 is invalid
        table = GrainTable(
            n_pixels=np.array([3, 0, 3, 3], dtype=np.int64),
            bounding_box=np.array(
                [[0, 0, 2, 2], [0, 0, 0, 0], [0, 2, 2, 2], [1, 2, 2, 2]],
                dtype=np.int64,
            ),
            rotation=Rotation(np.stack([r1, identity, identity, r4])),
            phase_id=np.array([0, -1, 0, 0], dtype=np.int32),
            kappa=np.array([123.25, -1.0, -1.0, 1.0]),
            max_grod=np.array([0.75, np.nan, np.nan, 0.5], dtype=np.float32),
            valid=np.array([True, False, False, True]),
            method="watson",
        )
        props = _broadcast_grain_props(table, grain_id.ravel())
        assert props["grain_orientation"].dtype == np.float64
        assert props["grain_orientation"].shape == (12, 4)
        assert props["grain_kappa"].dtype == np.float64
        assert props["grain_kappa"].shape == (12,)
        assert props["grain_max_grod"].dtype == np.float32
        assert props["grain_max_grod"].shape == (12,)
        outside = grain_id.ravel() == 0
        assert np.all(np.isnan(props["grain_orientation"][outside]))
        assert np.all(np.isnan(props["grain_kappa"][outside]))
        assert np.all(np.isnan(props["grain_max_grod"][outside]))

        coords, n = create_coordinate_arrays((3, 4), step_sizes=(1, 1))
        xmap = CrystalMap(
            rotations=Rotation.identity((n,)),
            phase_id=np.zeros(n, dtype=np.int32),
            x=coords["x"],
            y=coords["y"],
            phase_list=PhaseList(phases=[Phase("ni", point_group="m-3m")], ids=[0]),
            prop={"grain_id": grain_id.ravel(), **props},
            scan_unit="um",
        )
        rebuilt = GrainTable.from_crystal_map(xmap)
        assert rebuilt.method is None
        assert rebuilt.n_grains == grain_id.max() == 4
        assert rebuilt.n_pixels.dtype == np.int64
        assert rebuilt.bounding_box.dtype == np.int64
        assert rebuilt.phase_id.dtype == np.int32
        assert rebuilt.kappa.dtype == np.float64
        assert rebuilt.max_grod.dtype == np.float32
        assert rebuilt.valid.dtype == np.bool_
        _assert_tables_equal(rebuilt, table)

    @staticmethod
    def _identity_map(shape, is_in_data=None) -> CrystalMap:
        n = int(np.prod(shape))
        return _crystal_map(Rotation.identity((n,)), shape, is_in_data=is_in_data)

    @pytest.mark.parametrize(
        "shape, orix_shape",
        [((1, 6), (6,)), ((6, 1), (6,)), ((1, 1), ()), ((2, 5), (2, 5))],
    )
    def test_map_grid_restores_one_row_one_column_and_one_point_maps(
        self, shape, orix_shape
    ):
        xmap = self._identity_map(shape)
        assert xmap.shape == orix_shape
        grid, grid_shape = _map_grid(xmap)
        assert tuple(grid_shape) == shape
        assert grid.shape == shape
        assert grid.dtype == np.int64
        n = int(np.prod(shape))
        np.testing.assert_array_equal(grid, np.arange(n).reshape(shape))

    @pytest.mark.parametrize(
        "absent, orix_shape",
        [
            ((0, 1, 2, 3), (2, 4)),
            ((3, 7, 11), (3, 3)),
            (tuple(i for i in range(12) if i != 5), (1, 1)),
        ],
        ids=["first-row", "last-column", "one-point"],
    )
    def test_map_grid_spans_points_not_in_the_data(self, absent, orix_shape):
        is_in_data = np.ones(12, dtype=bool)
        is_in_data[list(absent)] = False
        xmap = self._identity_map((3, 4), is_in_data=is_in_data)
        assert xmap.shape == orix_shape
        grid, grid_shape = _map_grid(xmap)
        assert tuple(grid_shape) == (3, 4)
        expected = np.full(12, -1, dtype=np.int64)
        expected[is_in_data] = np.arange(is_in_data.sum())
        np.testing.assert_array_equal(grid, expected.reshape(3, 4))
        if len(absent) == 11:
            assert grid[1, 1] == 0

    @pytest.mark.parametrize("shape, absent", [((1, 5), 0), ((5, 1), 4)])
    def test_map_grid_spans_points_not_in_the_data_of_one_row_or_column(
        self, shape, absent
    ):
        is_in_data = np.ones(5, dtype=bool)
        is_in_data[absent] = False
        xmap = self._identity_map(shape, is_in_data=is_in_data)
        grid, grid_shape = _map_grid(xmap)
        assert tuple(grid_shape) == shape
        expected = np.full(5, -1, dtype=np.int64)
        expected[is_in_data] = np.arange(4)
        np.testing.assert_array_equal(grid, expected.reshape(shape))

    def test_map_grid_of_a_sliced_map_keeps_the_original_grid(self):
        xmap = self._identity_map((3, 4))
        sliced = xmap[1:3, 1:3]
        grid, grid_shape = _map_grid(sliced)
        assert tuple(grid_shape) == (3, 4)
        expected = np.full((3, 4), -1, dtype=np.int64)
        expected[1:3, 1:3] = np.arange(4).reshape(2, 2)
        np.testing.assert_array_equal(grid, expected)

    def test_map_grid_keeps_absent_rows_in_the_kam_map(self, hrosm_gradient_xmap):
        full = hrosm_gradient_xmap((3, 4))
        partial = hrosm_gradient_xmap((3, 4), absent=(0, 1, 2, 3))
        kam_full = kernel_average_misorientation_map(full)
        kam = kernel_average_misorientation_map(partial)
        assert kam.shape == (3, 4)
        assert np.all(np.isnan(kam[0]))
        # (2, 1) has no absent neighbour
        assert kam[2, 1] == kam_full[2, 1]
        assert np.isfinite(kam[2, 1])


class TestCompatEM:
    @pytest.mark.parametrize("kind, alpha, n, n_em", _compat_em_params())
    def test_matches_the_seeded_transcription(self, kind, alpha, n, n_em):
        x = _compat_samples(alpha, n, kind)
        result = _em_emsoft(x, OPERATORS, kind, n_em, 40, np.random.default_rng(0))
        expected = _emsoft_em(x, OPERATORS, kind, n_em, 40, np.random.default_rng(0))
        _assert_em_result_equals(result, expected)
        assert np.asarray(result.mu).dtype == np.float64
        assert np.asarray(result.log_likelihood).shape == (n_em,)
        assert np.asarray(result.n_iterations).dtype == np.int64

    @pytest.mark.parametrize("kind", ["vmf", "watson"])
    def test_compat_model_is_right_sided(self, kind):
        rng = np.random.default_rng(70)
        kappa = _sampled_kappa(kind, 100)
        mu = _mu()
        samples = _sample_s3(mu, kappa, 300, kind, rng)
        # EMsoft's von Mises-Fisher mixture has no component per negated
        # operator, so its samples use the sign-safe operator subset;
        # the scalar part of mu * S_j equals that of S_j * mu, so the
        # subset keeps the signs on both sides
        j = _draw_operators(300, rng, sign_safe=kind == "vmf")
        x = samples.data.reshape(-1, 4)
        right = Rotation(_qmul(x, OPERATORS[j]))
        left = Rotation(_qmul(OPERATORS[j], x))

        tables = {}
        for side, rotations in (("right", right), ("left", left)):
            xmap, grain_id = _one_grain_row(rotations)
            tables[side] = average_grain_orientations(
                xmap,
                grain_id,
                method=kind,
                max_angle=90.0,
                n_em=N_EM_FAST,
                min_kappa=0.0,
                seed=0,
                emsoft_compatible=True,
            )
        recovered = tables["right"].rotation.data[0]
        assert _right_class_error_rad(recovered, mu) <= _band_rad(100, 300)
        band = KAPPA_RATIO_BAND_N300
        assert 1 / band <= tables["right"].kappa[0] / kappa <= band
        assert (
            tables["left"].kappa[0]
            <= WRONG_SIDE_MAX_KAPPA_RATIO * tables["right"].kappa[0]
        )

    def test_vmf_at_realistic_kappa_runs_every_iteration_once_q_is_not_finite(
        self, record_property
    ):
        rng = np.random.default_rng(80)
        samples = _sample_s3(_mu(), 1e4, 50, "vmf", rng)
        x = _with_positive_scalar(samples.data.reshape(-1, 4))

        trace = []
        expected = _emsoft_em(
            x, OPERATORS, "vmf", N_EM_FAST, 40, np.random.default_rng(0), trace
        )
        result = _em_emsoft(
            x, OPERATORS, "vmf", N_EM_FAST, 40, np.random.default_rng(0)
        )

        # Once the concentration is realistic, exp(logCp) underflows to
        # 0 and exp(kappa t) overflows, so Q is not finite and the
        # convergence test never passes again: every initial guess
        # whose Q is not finite at some iteration runs every iteration.
        # An initial guess whose Q stays finite may stop early.
        not_finite = [
            init
            for init, init_trace in enumerate(trace)
            if not np.all(np.isfinite([q for q, _, _ in init_trace]))
        ]
        assert not_finite
        for init in not_finite:
            assert len(trace[init]) == 40
            assert result.n_iterations[init] == 40
        # The best initial guess is the first maximum of the log
        # likelihood, NaN ignored, so it need not be the first guess
        log_likelihood = np.asarray(expected[2])
        assert result.best_init == int(np.nanargmax(log_likelihood))
        _assert_em_result_equals(result, expected)

        record_property("log_likelihood", [float(v) for v in result.log_likelihood])
        record_property("n_iterations", [int(v) for v in result.n_iterations])
        record_property("best_init", int(result.best_init))

    def test_watson_underflow_keeps_the_previous_q_and_exits_at_the_second_iteration(
        self, record_property
    ):
        rng = np.random.default_rng(WATSON_UNDERFLOW_SEED)
        samples = _sample_s3(_mu(), 1e4, 50, "watson", rng)
        x = _with_positive_scalar(samples.data.reshape(-1, 4))

        trace = []
        expected = _emsoft_em(
            x, OPERATORS, "watson", N_EM_FAST, 40, np.random.default_rng(0), trace
        )
        underflowing = [
            init
            for init, init_trace in enumerate(trace)
            if len(init_trace) >= 2 and init_trace[1][2]
        ]
        assert underflowing, "no initial guess underflows at its second iteration"
        for init in underflowing:
            (q1, l1, _), (q2, l2, _) = trace[init][:2]
            assert q2 == q1
            assert l2 == l1
            assert len(trace[init]) == 2

        result = _em_emsoft(
            x, OPERATORS, "watson", N_EM_FAST, 40, np.random.default_rng(0)
        )
        for init in underflowing:
            assert result.n_iterations[init] == 2
            assert result.log_likelihood[init] == trace[init][0][1]
        _assert_em_result_equals(result, expected)

        record_property("log_likelihood", [float(v) for v in result.log_likelihood])
        record_property("n_iterations", [int(v) for v in result.n_iterations])
        record_property("best_init", int(result.best_init))

    def test_final_representative_side(self):
        mu = _mu().data.reshape(4)
        compat = _final_representative(mu, OPERATORS, True)
        correct = _final_representative(mu, OPERATORS, False)

        right = _qmul(mu[None, :], OPERATORS)
        left = _qmul(OPERATORS, mu[None, :])
        k_right = int(np.argmax(np.abs(right[:, 0])))
        k_left = int(np.argmax(np.abs(left[:, 0])))
        np.testing.assert_allclose(
            compat, _with_positive_scalar(right[k_right]), atol=1e-15, rtol=0
        )
        np.testing.assert_allclose(
            correct, _with_positive_scalar(left[k_left]), atol=1e-15, rtol=0
        )
        assert compat[0] >= 0
        assert correct[0] >= 0
        assert np.max(np.abs(compat - correct)) > 1e-3

    @pytest.mark.parametrize("kind", ["vmf", "watson"])
    @pytest.mark.parametrize(
        "emsoft_compatible", [False, True], ids=["correct", "compat"]
    )
    def test_kappa_gate_is_strict(self, emsoft_compatible, kind):
        rng = np.random.default_rng(70)
        samples = _sample_s3(_mu(), _sampled_kappa(kind, 100), 50, kind, rng)
        side = "right" if emsoft_compatible else "left"
        scrambled, _ = _scramble(samples, side, rng)
        xmap, grain_id = _one_grain_row(scrambled)
        kwargs = {
            "method": kind,
            "max_angle": 90.0,
            "n_em": N_EM_FAST,
            "seed": 0,
            "emsoft_compatible": emsoft_compatible,
        }

        first = average_grain_orientations(xmap, grain_id, min_kappa=0.0, **kwargs)
        kappa_hat = float(first.kappa[0])
        assert first.valid[0]
        assert np.isfinite(kappa_hat)
        assert kappa_hat > 0

        at = average_grain_orientations(xmap, grain_id, min_kappa=kappa_hat, **kwargs)
        assert not at.valid[0]
        assert at.kappa[0] == -1.0
        np.testing.assert_array_equal(at.rotation.data[0], [1.0, 0.0, 0.0, 0.0])
        assert np.isnan(at.max_grod[0])

        below = float(np.nextafter(kappa_hat, 0))
        kept = average_grain_orientations(xmap, grain_id, min_kappa=below, **kwargs)
        assert kept.valid[0]
        assert kept.kappa[0] == kappa_hat

    def test_nan_kappa_is_rejected_and_inf_is_kept(self):
        assert not _apply_kappa_gate(np.nan, 5.0)
        assert _apply_kappa_gate(np.inf, 5.0)
        assert not _apply_kappa_gate(5.0, 5.0)
        assert _apply_kappa_gate(float(np.nextafter(5.0, 6.0)), 5.0)

        g = Rotation.from_euler(np.deg2rad(MU_EULER_DEG))
        identical = Rotation(np.repeat(g.data.reshape(1, 4), 5, axis=0))
        xmap, grain_id = _one_grain_row(identical)
        table = average_grain_orientations(
            xmap, grain_id, method="vmf", n_em=N_EM_FAST, seed=0
        )
        assert table.kappa[0] == np.inf
        assert table.valid[0]

    @pytest.mark.parametrize("kind", ["vmf", "watson"])
    def test_kappa_lookup_and_closed_forms(self, kind):
        ys = np.linspace(0.05, 0.99, 95)
        actual = [_kappa_from_y(float(y), kind) for y in ys]
        expected = [_kappa_oracle(float(y), kind) for y in ys]
        np.testing.assert_array_equal(actual, expected)
        assert any(y >= 0.94 for y in ys)
        assert any(y < 0.94 for y in ys)

        # The table's operation order matters
        x_ap, _ = _kappa_table(kind)
        assert np.count_nonzero(x_ap != 0.001 * np.arange(1, 35001)) == 11504

    @pytest.mark.parametrize(
        "kind, kappas",
        [
            ("vmf", [0.5, 5.0, 29.9, 30.0, 30.1, 100.0, 1e4]),
            ("watson", [0.5, 5.0, 19.9, 20.0, 20.1, 100.0, 1e4]),
        ],
    )
    def test_log_normaliser_branches(self, kind, kappas):
        assert C == -3.675754132818690967
        assert C2 == 4.1746562059854348688
        assert C2W == 5.4243952068443172530
        for kappa in kappas:
            expected = _log_cp_oracle(kappa, kind)
            assert _log_cp(kappa, kind) == pytest.approx(expected, rel=1e-12)
