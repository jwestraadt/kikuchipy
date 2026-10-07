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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMsoftOO:
# - The expectation maximisation of a von Mises-Fisher or Watson
#   mixture over the symmetry variants EMforDS_, its E- and M-steps,
#   getQandL, the concentration from the mean resultant length (the
#   kappa tables and their asymptotic forms) and the log normalising
#   constant logCp (EMsoftOOLib/mod_dirstats.f90)
# - The choice of the final representative among the symmetry
#   variants (EMsoftOOLib/mod_dirstats.f90)

# #####################################################################
# Copyright (c) 2014-2026, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  - Neither the names of Marc De Graef, Carnegie Mellon University nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################

# Changes by the kikuchipy developers, 2026-10-06: ported from Fortran
# to NumPy; defects reproduced only behind ``emsoft_compatible``.

"""Von Mises-Fisher and Watson mixture estimates of the mean
orientation of a grain over the symmetry variants, in a correct mode
and in an EMsoft compatible mode.
"""

from __future__ import annotations

import functools
from typing import Literal, NamedTuple

import numpy as np
from scipy.special import i0, i1, ive

# Constants of EMsoft's log normalising constant logCp: C for the von
# Mises-Fisher density at small kappa, C2 at large kappa, and C2W for
# the Watson density at large kappa
C = -3.675754132818690967
C2 = 4.1746562059854348688
C2W = 5.4243952068443172530

# Initial concentration of every initial guess
_KAPPA_INIT = 30.0
# Convergence tolerance of the expected complete log likelihood Q
_Q_TOLERANCE = 0.01


class EMResult(NamedTuple):
    """Result of one expectation maximisation over several initial
    guesses.

    Attributes
    ----------
    mu
        Final mean quaternion of shape (4,) of float64 with a
        non-negative scalar part, the chosen symmetry representative.
    kappa
        Concentration of the best initial guess.
    log_likelihood
        Final log likelihood of every initial guess, shape (n_em,) of
        float64.
    n_iterations
        Number of iterations run per initial guess, shape (n_em,) of
        int64.
    best_init
        0-based index of the chosen initial guess.
    """

    mu: np.ndarray
    kappa: float
    log_likelihood: np.ndarray
    n_iterations: np.ndarray
    best_init: int


@functools.lru_cache(maxsize=None)
def _kappa_table(kind: Literal["vmf", "watson"]) -> tuple[np.ndarray, np.ndarray]:
    """Return EMsoft's concentration table ``xAp`` and its mean
    resultant length (von Mises-Fisher) or top eigenvalue (Watson)
    ``yAp``, read-only.
    """
    i = np.arange(1, 35001)
    # EMsoft's operation order, which differs from 0.001 * i
    x = 0.001 + (i - 1).astype(np.float64) * 0.001
    # Ratios of exponentially scaled Bessel functions, the scaling
    # cancels in each ratio
    if kind == "vmf":
        y = ive(2, x) / ive(1, x)
    else:
        y1 = ive(1, x * 0.5)
        y2 = ive(0, x * 0.5)
        y = y1 / (y2 - y1) / x
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


def _kappa_from_y(y: float, kind: Literal["vmf", "watson"]) -> float:
    """Return the concentration of a mean resultant length ``y`` (von
    Mises-Fisher) or top eigenvalue ``y`` (Watson).

    For ``y >= 0.94`` EMsoft's asymptotic form is used, else the
    nearest entry of EMsoft's table over ``kappa = 0.001 + (i - 1) *
    0.001``, ``i = 1, ..., 35000``, first minimum, index 1 replaced by
    2; the table's Bessel function ratios come from
    :func:`scipy.special.ive`.
    """
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


def _log_cp(kappa: float, kind: Literal["vmf", "watson"]) -> float:
    """Return EMsoft's log normalising constant of the von Mises-Fisher
    or Watson density on the unit quaternions, using :data:`C`,
    :data:`C2` and :data:`C2W`.
    """
    with np.errstate(all="ignore"):
        if kind == "vmf":
            if kappa > 30.0:
                lcp = kappa**4.5 / (
                    -105.0 + 8.0 * kappa * (-15.0 + 16.0 * kappa * (-3.0 + 8.0 * kappa))
                )
                return float(C2 - kappa + np.log(lcp))
            return float(C + np.log(kappa / i1(kappa)))
        if kappa > 20.0:
            lcp = kappa**4.5 / (
                525.0 + 4.0 * kappa * (45.0 + 8.0 * kappa * (3.0 + 4.0 * kappa))
            )
            return float(C2W - kappa + np.log(lcp))
        return float(-kappa * 0.5 - np.log(i0(kappa * 0.5) - i1(kappa * 0.5)))


def _conj(q: np.ndarray) -> np.ndarray:
    """Return the conjugates of quaternions of shape (..., 4)."""
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def _quatmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a * b`` in EMsoft's term order,
    broadcast over the leading axes.
    """
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
    """Return the four-term dot product summed left to right, as
    EMsoft's density does.
    """
    return ((a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]) + (
        a[..., 3] * b[..., 3]
    )


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Return one initial mean: four standard normal draws, normalised,
    with a non-negative scalar part.
    """
    mu = rng.standard_normal(4)
    mu = mu / np.sqrt(np.sum(mu**2))
    if mu[0] < 0:
        mu = -mu
    return mu


def _em_correct(
    x: np.ndarray,
    operators: np.ndarray,
    kind: Literal["vmf", "watson"],
    n_em: int,
    n_iter: int,
    rng: np.random.Generator,
) -> EMResult:
    """Return the correct expectation maximisation estimate.

    The symmetry operators act from the left, the E-step is computed
    in log space with a log-sum-exp normalisation, the Watson M-step
    takes the top eigenvector of :func:`numpy.linalg.eigh`, and the
    best initial guess has the largest finite log likelihood.

    A quaternion and its negative are one orientation. The Watson
    density is antipodally symmetric and its mixture runs over the
    operators. The von Mises-Fisher density is not, so its mixture
    runs over the operators followed by their negatives, in the same
    order (twice as many components), in the E-step, the M-step and
    the log likelihood, which subtracts the log of that number of
    components. Points of a grain in either sign are then aligned.

    Parameters
    ----------
    x
        Unit quaternions of the grain's points, shape (n, 4) of
        float64, in raster order.
    operators
        Proper point group operators of shape (m, 4).
    kind
        "vmf" or "watson".
    n_em
        Number of initial guesses.
    n_iter
        Largest number of iterations per initial guess.
    rng
        Random number generator drawing the initial guesses.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 4)
    operators = np.asarray(operators, dtype=np.float64).reshape(-1, 4)
    if kind == "vmf":
        components = np.concatenate([operators, -operators])
    else:
        components = operators
    n = x.shape[0]
    log_n_components = np.log(components.shape[0])

    # v[n, j] = conj(S_j) x_n, so that <S_j mu, x_n> = <mu, v[n, j]>
    v = _quatmult(_conj(components)[None, :, :], x[:, None, :])

    def profile(mu: np.ndarray) -> np.ndarray:
        t = v @ mu
        return t if kind == "vmf" else t * t

    def relative_log_density(f: np.ndarray, kappa: float) -> np.ndarray:
        # kappa (f - max_j f) without inf * 0 when kappa is infinite
        s = f - f.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            a = kappa * s
        return np.where(s == 0, 0.0, a)

    mu_all = np.zeros((n_em, 4))
    kappa_all = np.zeros(n_em)
    l_all = np.full(n_em, np.nan)
    n_iterations = np.zeros(n_em, dtype=np.int64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for init in range(n_em):
            mu = _random_unit_quaternion(rng)
            kappa = _KAPPA_INIT
            q_prev = np.nan
            for i in range(n_iter):
                # E-step: responsibilities normalised over the
                # components in log space
                a = relative_log_density(profile(mu), kappa)
                r = np.exp(a)
                r /= r.sum(axis=1, keepdims=True)

                # M-step
                if kind == "vmf":
                    gamma = np.einsum("nm,nmi->i", r, v)
                    norm = np.sqrt(np.sum(gamma**2))
                    mu_new = gamma / norm
                    y = norm / n
                else:
                    scatter = np.einsum("nm,nmi,nmj->ij", r, v, v) / n
                    values, vectors = np.linalg.eigh(scatter)
                    mu_new = vectors[:, -1]
                    y = values[-1]
                # A mean resultant length or eigenvalue above 1 is
                # rounding only
                y = min(float(y), 1.0)
                kappa_new = _kappa_from_y(y, kind)
                if mu_new[0] < 0:
                    mu_new = -mu_new

                # Expected complete log likelihood Q and log likelihood
                # L of the new parameters
                f = profile(mu_new)
                log_cp = _log_cp(kappa_new, kind)
                f_max = f.max(axis=1)
                a = relative_log_density(f, kappa_new)
                log_phi_max = log_cp + kappa_new * f_max - log_n_components
                log_phi = log_phi_max[:, None] + a
                l_value = float(np.sum(log_phi_max + np.log(np.exp(a).sum(axis=1))))
                q_value = float(np.sum(np.where(r > 0, r * log_phi, 0.0)))

                mu, kappa = mu_new, kappa_new
                mu_all[init] = mu
                kappa_all[init] = kappa
                l_all[init] = l_value
                n_iterations[init] = i + 1
                if i >= 1 and abs(q_value - q_prev) < _Q_TOLERANCE:
                    break
                q_prev = q_value

    finite = np.isfinite(l_all)
    best = int(np.argmax(np.where(finite, l_all, -np.inf))) if finite.any() else 0
    mu = _final_representative(mu_all[best], operators, False)
    return EMResult(
        mu=mu,
        kappa=float(kappa_all[best]),
        log_likelihood=l_all,
        n_iterations=n_iterations,
        best_init=best,
    )


def _emsoft_density(
    centres: np.ndarray,
    x: np.ndarray,
    kappa: float,
    c: float,
    kind: Literal["vmf", "watson"],
) -> np.ndarray:
    """Return EMsoft's density of every sample (rows) for every centre
    (columns), ``exp(c + kappa f(t))``.
    """
    t = _dot4(centres[None, :, :], x[:, None, :])
    f = t if kind == "vmf" else t**2
    return np.exp(c + kappa * f)


def _em_emsoft(
    x: np.ndarray,
    operators: np.ndarray,
    kind: Literal["vmf", "watson"],
    n_em: int,
    n_iter: int,
    rng: np.random.Generator,
) -> EMResult:
    """Return the expectation maximisation estimate as EMsoft computes
    it.

    The E- and M-steps apply the operators from the right, the log
    likelihood from the left; the previous ``Q`` and ``L``, starting at
    0.0 and carried across iterations and initial guesses, are kept
    when the smallest density is not positive; the mean is not sign
    fixed between iterations; the best initial guess is the first
    maximum of the log likelihood, ignoring NaN, the first guess if all
    are NaN. Parameters as :func:`_em_correct`.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 4)
    operators = np.asarray(operators, dtype=np.float64).reshape(-1, 4)
    n = x.shape[0]
    n_ops = operators.shape[0]
    # M-step vectors x_n * conj(S_j)
    v = _quatmult(x[:, None, :], _conj(operators)[None, :, :])

    mu_all = np.zeros((n_em, 4))
    kappa_all = np.zeros(n_em)
    l_all = np.zeros(n_em)
    n_iterations = np.zeros(n_em, dtype=np.int64)
    # EMsoft declares Q and L once, so the previous values carry over
    # iterations and initial guesses
    q_i = 0.0
    l_i = 0.0
    with np.errstate(all="ignore"):
        for init in range(n_em):
            mu = _random_unit_quaternion(rng)
            kappa = _KAPPA_INIT
            q_hist = np.zeros(n_iter)
            for i in range(n_iter):
                # E-step with the centres mu * S_j
                c = _log_cp(kappa, kind)
                centres = _quatmult(mu[None, :], operators)
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
                kappa_new = _kappa_from_y(float(y), kind)

                # Q and L with the centres S_j * mu; the von
                # Mises-Fisher density takes exp(logCp) as its constant
                old_q, old_l = q_i, l_i
                c = _log_cp(kappa_new, kind)
                if kind == "vmf":
                    c = np.exp(c)
                centres = _quatmult(operators, mu_new[None, :])
                phi = _emsoft_density(centres, x, kappa_new, c, kind) / n_ops
                if np.min(phi) > 0.0:
                    l_i = float(np.sum(np.log(np.sum(phi, axis=1))))
                    q_i = float(np.sum(r * np.log(phi)))
                else:
                    l_i, q_i = old_l, old_q
                q_hist[i] = q_i

                mu_all[init] = mu_new
                kappa_all[init] = kappa_new
                l_all[init] = l_i
                mu = mu_new
                kappa = kappa_new
                n_iterations[init] = i + 1
                if i >= 1 and abs(q_hist[i] - q_hist[i - 1]) < _Q_TOLERANCE:
                    break

    best = 0 if np.all(np.isnan(l_all)) else int(np.nanargmax(l_all))
    mu = mu_all[best].copy()
    if mu[0] < 0:
        mu = -mu
    mu = _final_representative(mu, operators, True)
    return EMResult(
        mu=mu,
        kappa=float(kappa_all[best]),
        log_likelihood=l_all,
        n_iterations=n_iterations,
        best_init=best,
    )


def _final_representative(
    mu: np.ndarray, operators: np.ndarray, emsoft_compatible: bool
) -> np.ndarray:
    """Return the symmetry variant of ``mu`` with the largest absolute
    scalar part, with a non-negative scalar part.

    The variants are ``S_i * mu`` (correct) or ``mu * S_i`` (EMsoft
    compatible), the first in operator order on ties. EMsoft returns
    its variant through a Rodrigues vector and back, which is
    reproduced in the EMsoft compatible mode.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(1, 4)
    operators = np.asarray(operators, dtype=np.float64).reshape(-1, 4)
    if emsoft_compatible:
        variants = _quatmult(mu, operators)
    else:
        variants = _quatmult(operators, mu)
    k = int(np.argmax(np.abs(variants[:, 0])))
    out = variants[k].copy()
    if out[0] < 0:
        out = -out
    if emsoft_compatible:
        out = _emsoft_rodrigues_round_trip(out)
    return out


def _emsoft_rodrigues_round_trip(q: np.ndarray) -> np.ndarray:
    """Return a quaternion with q0 >= 0 converted to a Rodrigues
    vector and back as EMsoft does in float64 (quaternion to Rodrigues,
    Rodrigues to axis-angle, axis-angle to quaternion, normalised).
    """
    q = np.asarray(q, dtype=np.float64)
    thr = 1e-10
    eps = 1e-12
    v = q[1:].copy()
    # Quaternion to Rodrigues (axis, tangent of half the angle)
    if q[0] < thr:
        axis = v
        t = np.inf
    else:
        s = np.sqrt(np.sum(v * v))
        if s < thr:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = v / s
        t = np.tan(np.arccos(q[0]))
    # Rodrigues to axis-angle (a near-zero tangent gives angle 0 about
    # z, which the next step turns into the identity)
    if abs(t - 0.0) < eps:
        axis = np.array([0.0, 0.0, 1.0])
        angle = 0.0
    elif t == np.inf:
        angle = np.pi
    else:
        angle = 2.0 * np.arctan(t)
        axis = axis * (1.0 / np.sqrt(np.sum(axis * axis)))
    # Axis-angle to quaternion, normalised
    if abs(angle - 0.0) < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    c = np.cos(angle * 0.5)
    s = np.sin(angle * 0.5)
    out = np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])
    return out / np.sqrt(np.sum(out**2))
