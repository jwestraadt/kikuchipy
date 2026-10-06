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

from typing import Literal, NamedTuple

import numpy as np

# Constants of EMsoft's log normalising constant logCp: C for the von
# Mises-Fisher density at small kappa, C2 at large kappa, and C2W for
# the Watson density at large kappa
C = -3.675754132818690967
C2 = 4.1746562059854348688
C2W = 5.4243952068443172530


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


def _kappa_from_y(y: float, kind: Literal["vmf", "watson"]) -> float:
    """Return the concentration of a mean resultant length ``y`` (von
    Mises-Fisher) or top eigenvalue ``y`` (Watson).

    For ``y >= 0.94`` EMsoft's asymptotic form is used, else the
    nearest entry of EMsoft's table over ``kappa = 0.001 + (i - 1) *
    0.001``, ``i = 1, ..., 35000``, first minimum, index 1 replaced by
    2; the table's Bessel function ratios come from
    :func:`scipy.special.ive`.
    """
    raise NotImplementedError


def _log_cp(kappa: float, kind: Literal["vmf", "watson"]) -> float:
    """Return EMsoft's log normalising constant of the von Mises-Fisher
    or Watson density on the unit quaternions, using :data:`C`,
    :data:`C2` and :data:`C2W`.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def _final_representative(
    mu: np.ndarray, operators: np.ndarray, emsoft_compatible: bool
) -> np.ndarray:
    """Return the symmetry variant of ``mu`` with the largest absolute
    scalar part, with a non-negative scalar part.

    The variants are ``S_i * mu`` (correct) or ``mu * S_i`` (EMsoft
    compatible), the first in operator order on ties.
    """
    raise NotImplementedError
