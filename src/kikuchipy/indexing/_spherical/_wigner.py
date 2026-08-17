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
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517), all from
# ``include/sht/wigner.hpp``:
# - The recursion coefficients ``u_jkm_0/1/2``, ``v_jkm``, ``w_jkm``,
#   ``a_jkm_0/1/2``, ``a_jkm_0_pre/1_pre/2_pre``, ``b_jkm``,
#   ``u_km_0/1/2``, ``a_km_0/1/2`` and ``e_km`` (lines 204-286)
# - ``d(j, k, m, t, nB)``, the Wigner (lowercase) d function (lines
#   298-371)
# - ``d(j, k, m)``, the same at beta = pi/2 (lines 380-409)
# - ``dSign(j, k, m)`` (lines 416-425)
# - ``D(j, k, m, eu)``, the Wigner (uppercase) D function (lines
#   436-439)
# - ``dTable(jMax, t, nB, table)`` (lines 452-559)
# - ``dTablePre(jMax, t, nB, table, pE, pW, pB)`` (lines 575-671)
# - ``dTablePreBuild(jMax, pE, pW, pB)`` (lines 678-691)
# - ``dTable(jMax, table, trans)``, the pi/2 table (lines 699-761)
# - ``rotateHarmonics(bw, alm, blm, zyz)`` (lines 769-799)
# - ``dPrime(j, k, m, t, nB)`` (lines 814-822)
# - ``dPrime2(j, k, m, t, nB)`` (lines 837-852)
#
# The wrap of beta into [-pi, pi] which ``rotate_harmonics()`` and
# ``wigner_D()`` apply is from ``Correlator::derivatives()``
# (``include/sht/sht_xcorr.hpp``, lines 895-899) and is a deliberate
# deviation from ``rotateHarmonics()`` and ``D()``, which do not wrap.

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Changed by Johan Westraadt, 2026-08: translated to
# Python/NumPy/Numba for kikuchipy. GPL-2.0-or-later, conveyed
# under GPL-3.0-or-later
# #####################################################################

"""Wigner (lowercase) d functions, their tables and the rotation of
spherical harmonic coefficients.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**Convention.** The reduced (lowercase) Wigner d function is the one
of Fukushima 2016 equation 1 :cite:`fukushima2016wigner`, which is
also EMSphInx' (``include/sht/wigner.hpp``, line 48),

.. code-block::

    d^j_{k,m}(beta) = sqrt((j+k)! (j-k)! / ((j+m)! (j-m)!))
                      * cos(beta/2)^(k+m) * sin(beta/2)^(k-m)
                      * P^(k-m, k+m)_(j-k)(cos beta)

with ``P`` the Jacobi polynomial, ``j`` the degree, ``k`` the first
order and ``m`` the second order (Fukushima's letters, used
throughout this module; the transform modules keep ``l``/``m``).
This equals Mathematica's ``WignerD[{j, k, m}, beta]`` and is **the
transpose of the Wikipedia/Varshalovich small d**,

.. code-block::

    d^here_{k,m}(beta) = d^Wikipedia_{m,k}(beta)
                       = d^Wikipedia_{k,m}(-beta)

e.g. ``d^1_{1,0}(pi/2) = +1/sqrt(2)`` and
``d^1_{0,1}(pi/2) = -1/sqrt(2)`` here, while Wikipedia's
``d^1_{1,0}(beta) = -sin(beta)/sqrt(2)``. It is NaN when
``j < max(|k|, |m|)``.

The uppercase Wigner D function of passive ZYZ Euler angles
``zyz = (alpha, beta, gamma)`` is

.. code-block::

    D^j_{k,m}(zyz) = d^j_{k,m}(beta) exp(i (m alpha + k gamma))

which is Mathematica's ``WignerD[{j, k, m}, gamma, beta, alpha]``,
and the harmonic coefficients of a rotated function are

.. code-block::

    b^l_m = sum_(n=-l)^(l) a^l_n D^l_{m,n}(zyz)

Only ``n >= 0`` is stored (:mod:`kikuchipy.indexing._spherical._sht`
convention ``a^l_{-n} = (-1)^n conj(a^l_n)``), and the ``-n`` term
uses ``d^l_{m,-n}(beta) = (-1)^(l+m) d^l_{m,n}(pi - beta)``, which is
why slot 1 of the beta table carries the ``(-1)^(l+m+n)`` sign in
:func:`rotate_harmonics`.

**Direction.** With ``R = Rotation(zyz_to_quaternion(zyz))`` and
``g = synthesize(rotate_harmonics(analyze(f), zyz))``,

.. code-block::

    g(n) = f((~R) * n)      equivalently      g(R * n) = f(n)

i.e. the features of ``f`` move by orix' active vector rotation
``R * v``, and ``(~R).to_matrix() == R.to_matrix().T ==
Rz(alpha) Ry(beta) Rz(gamma)``. Rotations compose with the **later
rotation on the left**:

.. code-block::

    rotate_harmonics(rotate_harmonics(alm, z1), z2)
        == rotate_harmonics(alm, quaternion_to_zyz((Q2 * Q1).data))

**Table layouts** (frozen; Phase 4's correlator and Phase 7's Newton
refinement index them exactly as the C++ does):

``wigner_d_table``, ``wigner_d_table_pre``
    ``(bw, bw, bw, 2)`` float64 with ``table[k, m, j, 0] =
    d^j_{k,m}(beta)`` and ``table[k, m, j, 1] = d^j_{k,m}(pi - beta)``,
    i.e. the flat C++ index ``(k bw^2 + m bw + j) * 2 + s``.
``wigner_d_table_factors``
    ``e_km`` of shape ``(bw, bw)`` for ``m <= k``, ``w_jkm`` and
    ``b_jkm`` of shape ``(bw, bw, bw)`` indexed ``[k, m, i]`` for
    ``m <= k`` and ``i >= k + 2``.
``wigner_d_half_pi_table``
    ``(bw, bw, bw)`` float64 with ``table[k, m, j] =
    d^j_{k,m}(pi/2)`` when ``transpose=False`` and
    ``table[m, k, j] = d^j_{k,m}(pi/2)`` when ``transpose=True``, the
    latter being what the correlator allocates.

**Undefined slots are NaN.** Every table is allocated NaN filled and
the kernels write only the defined slots ``j >= max(k, m)``, a
deliberate deviation from the uninitialized memory of the C++, so
that a stray read poisons the result loudly. On the ``out=`` path of
:func:`wigner_d_table_pre` the NaN slots are caller owned; the buffer
is not refilled and two representative slots are checked instead.

**Beta is wrapped.** :func:`rotate_harmonics` and :func:`wigner_D`
wrap ``beta`` into [-pi, pi] with
:func:`kikuchipy.indexing._spherical._euler.wrap_beta` before taking
``cos`` and ``signbit``, which ``rotateHarmonics()`` and ``D()`` do
not; outside that interval the sign bit no longer identifies the
branch and the C++ is wrong by up to 2.2. The scalar functions and
:func:`wigner_d_table` take ``(cos_beta, negative_beta)`` as the C++
does and cannot wrap, so their callers must.

**The derivatives have an open beta domain.**
:func:`wigner_d_prime` and :func:`wigner_d_prime2` divide by
``sin(beta)`` and raise ``ZeroDivisionError`` at
``|cos_beta| == 1``, in the compiled kernel as in its ``py_func``,
where the C++ returns an IEEE infinity or NaN. Phase 7's Newton
refinement must keep ``beta`` off ``0`` and ``pi``.

**Memory.** A beta table is ``2 bw^3`` doubles and the ``pi/2`` table
``bw^3``: 5.0 / 10.9 / 23.1 / 63 MB at ``bw`` 68 / 88 / 113 / 158,
and 906 MB at ``bw`` 384. A :func:`wigner_d_table_pre` caller holds
``w_jkm``, ``b_jkm`` and the table simultaneously, i.e. ``4 bw^3``
doubles (10.1 / 21.8 / 46.2 MB at ``bw`` 68 / 88 / 113). This is why
:func:`rotate_harmonics` is a test and visualization tool at master
pattern bandwidths, never a per pattern operation.

**Known limitation.** Fukushima's extended exponent ("X-number")
arithmetic is not implemented, so the recursion seed
``cos(beta/2)^(k+m) sin(beta/2)^(k-m)`` underflows for large
``k + m`` at large ``beta`` and the whole ``(k, m)`` column is
returned as exactly ``0.0``. The largest true value lost is about
1e-139 at ``bw`` 512 for ``beta >= 2.5``, and nothing is lost at
``bw <= 128`` for ``beta <= 3.0``, which is irrelevant to every
consumer. It is pinned by a named test rather than hidden
:cite:`lenthe2019spherical`.
"""

import math

from numba import njit
import numpy as np

from kikuchipy.indexing._spherical._euler import _wrap_beta

# Value of every undefined table slot and of every scalar call whose
# degree is smaller than the largest absolute order
_NAN = math.nan

# --------------------- Recursion coefficients ----------------------- #


@njit(cache=True, nogil=True)
def _u_jkm_0(j: int, k: int, m: int, tc: float) -> float:
    """Return the recursion coefficient ``u_{j,k,m}`` for beta < pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    tc
        ``1 - cos(beta)``.

    Returns
    -------
    u
        ``-tc * ((j - 1) * j) - (k * m - (j - 1) * j)``, Fukushima
        equation 13 (``include/sht/wigner.hpp``, line 210).
    """
    return -tc * ((j - 1) * j) - (k * m - (j - 1) * j)


@njit(cache=True, nogil=True)
def _u_jkm_1(j: int, k: int, m: int) -> int:
    """Return the recursion coefficient ``u_{j,k,m}`` at beta = pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``. ``j`` is unused
        and kept for signature symmetry with the other two branches.

    Returns
    -------
    u
        ``-k * m`` in 64-bit integer arithmetic
        (``include/sht/wigner.hpp``, line 211).
    """
    return -k * m


@njit(cache=True, nogil=True)
def _u_jkm_2(j: int, k: int, m: int, t: float) -> float:
    """Return the recursion coefficient ``u_{j,k,m}`` for beta > pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    t
        ``cos(beta)``.

    Returns
    -------
    u
        ``t * ((j - 1) * j) - k * m``
        (``include/sht/wigner.hpp``, line 212).
    """
    return t * ((j - 1) * j) - k * m


@njit(cache=True, nogil=True)
def _v_jkm(j: int, k: int, m: int) -> float:
    """Return the recursion coefficient ``v_{j,k,m}``.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.

    Returns
    -------
    v
        ``sqrt((j+k-1)(j-k-1)(j+m-1)(j-m-1)) * j``, Fukushima
        equation 14 (``include/sht/wigner.hpp``, line 219). The
        product inside the square root is formed in 64-bit integer
        arithmetic before the cast, as the C++ ``Real(...)`` does.
    """
    product = (j + k - 1) * (j - k - 1) * (j + m - 1) * (j - m - 1)
    return math.sqrt(float(product)) * j


@njit(cache=True, nogil=True)
def _w_jkm(j: int, k: int, m: int) -> float:
    """Return the recursion coefficient ``w_{j,k,m}``.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.

    Returns
    -------
    w
        ``1 / (sqrt((j+k)(j-k)(j+m)(j-m)) * (j - 1))``, Fukushima
        equation 15 (``include/sht/wigner.hpp``, line 227).

    Notes
    -----
    This is the coefficient most susceptible to integer overflow:
    the product reaches ``512^4 = 6.9e10`` at ``bw`` 512, which
    overflows 32-bit integers from ``k`` about 215 onwards. 64-bit
    integers are good to a degree of about 55 000.
    """
    product = (j + k) * (j - k) * (j + m) * (j - m)
    return 1.0 / (math.sqrt(float(product)) * (j - 1))


@njit(cache=True, nogil=True)
def _a_jkm_0(j: int, k: int, m: int, tc: float) -> float:
    """Return the recursion coefficient ``a_{j,k,m}`` for beta < pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    tc
        ``1 - cos(beta)``.

    Returns
    -------
    a
        ``w_jkm(j, k, m) * (u_jkm_0(j, k, m, tc) * (2 * j - 1))``,
        Fukushima equation 11 (``include/sht/wigner.hpp``, line 244).
        The association is the C++ one and must not be changed, see
        :func:`_b_jkm`.
    """
    return _w_jkm(j, k, m) * (_u_jkm_0(j, k, m, tc) * (2 * j - 1))


@njit(cache=True, nogil=True)
def _a_jkm_1(j: int, k: int, m: int) -> float:
    """Return the recursion coefficient ``a_{j,k,m}`` at beta = pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.

    Returns
    -------
    a
        ``w_jkm(j, k, m) * (u_jkm_1(j, k, m) * (2 * j - 1))``
        (``include/sht/wigner.hpp``, line 245).
    """
    return _w_jkm(j, k, m) * (_u_jkm_1(j, k, m) * (2 * j - 1))


@njit(cache=True, nogil=True)
def _a_jkm_2(j: int, k: int, m: int, t: float) -> float:
    """Return the recursion coefficient ``a_{j,k,m}`` for beta > pi/2.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    t
        ``cos(beta)``.

    Returns
    -------
    a
        ``w_jkm(j, k, m) * (u_jkm_2(j, k, m, t) * (2 * j - 1))``
        (``include/sht/wigner.hpp``, line 246).
    """
    return _w_jkm(j, k, m) * (_u_jkm_2(j, k, m, t) * (2 * j - 1))


@njit(cache=True, nogil=True)
def _a_jkm_0_pre(w: float, j: int, k: int, m: int, tc: float) -> float:
    """Return ``a_{j,k,m}`` for beta < pi/2 from a precomputed ``w``.

    Parameters
    ----------
    w
        ``w_jkm(j, k, m)``, read from the table of
        :func:`wigner_d_table_factors`.
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    tc
        ``1 - cos(beta)``.

    Returns
    -------
    a
        ``w * (u_jkm_0(j, k, m, tc) * (2 * j - 1))``
        (``include/sht/wigner.hpp``, line 258).
    """
    return w * (_u_jkm_0(j, k, m, tc) * (2 * j - 1))


@njit(cache=True, nogil=True)
def _a_jkm_2_pre(w: float, j: int, k: int, m: int, t: float) -> float:
    """Return ``a_{j,k,m}`` for beta > pi/2 from a precomputed ``w``.

    Parameters
    ----------
    w
        ``w_jkm(j, k, m)``, read from the table of
        :func:`wigner_d_table_factors`.
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.
    t
        ``cos(beta)``.

    Returns
    -------
    a
        ``w * (u_jkm_2(j, k, m, t) * (2 * j - 1))``
        (``include/sht/wigner.hpp``, line 260).

    Notes
    -----
    ``a_jkm_1_pre`` (line 259) is not ported: the table kernels group
    ``beta == pi/2`` with ``beta < pi/2`` through
    ``isType0 = not signbit(t)`` (line 467), which is bitwise
    identical because ``u_jkm_0(j, k, m, tc=1) == -k * m ==
    u_jkm_1(j, k, m)`` exactly in floating point, so the type 1
    branch never reaches a precomputed coefficient.
    """
    return w * (_u_jkm_2(j, k, m, t) * (2 * j - 1))


@njit(cache=True, nogil=True)
def _b_jkm(j: int, k: int, m: int) -> float:
    """Return the recursion coefficient ``b_{j,k,m}``.

    Parameters
    ----------
    j, k, m
        Degree and the two orders of ``d^j_{k,m}``.

    Returns
    -------
    b
        ``w_jkm(j, k, m) * v_jkm(j, k, m)``, Fukushima equation 12
        (``include/sht/wigner.hpp``, line 268).

    Notes
    -----
    The association matters: ``w * (sqrt(...) * j)`` is the C++ form
    and ``(w * sqrt(...)) * j`` differs by 1.7e-16, which is enough
    to break the bitwise equality of the tables and the scalar
    function that the tests assert.
    """
    return _w_jkm(j, k, m) * _v_jkm(j, k, m)


@njit(cache=True, nogil=True)
def _u_km_0(k: int, m: int, tc: float) -> float:
    """Return the seed coefficient ``u_{k,m}`` for beta < pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``.
    tc
        ``1 - cos(beta)``.

    Returns
    -------
    u
        ``-tc * (k + 1) - (m - 1 - k)``, Fukushima equation 23
        (``include/sht/wigner.hpp``, line 276).
    """
    return -tc * (k + 1) - (m - 1 - k)


@njit(cache=True, nogil=True)
def _u_km_1(k: int, m: int) -> float:
    """Return the seed coefficient ``u_{k,m}`` at beta = pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``. ``k`` is unused and kept for
        signature symmetry with the other two branches.

    Returns
    -------
    u
        ``-m`` (``include/sht/wigner.hpp``, line 277).
    """
    return float(-m)


@njit(cache=True, nogil=True)
def _u_km_2(k: int, m: int, t: float) -> float:
    """Return the seed coefficient ``u_{k,m}`` for beta > pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``.
    t
        ``cos(beta)``.

    Returns
    -------
    u
        ``t * (k + 1) - m`` (``include/sht/wigner.hpp``, line 278).
    """
    return t * (k + 1) - m


@njit(cache=True, nogil=True)
def _a_km_0(k: int, m: int, tc: float) -> float:
    """Return the seed coefficient ``a_{k,m}`` for beta < pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``.
    tc
        ``1 - cos(beta)``.

    Returns
    -------
    a
        ``sqrt((2k+1) / ((k+m+1)(k-m+1))) * u_km_0(k, m, tc)``,
        Fukushima equation 22 (``include/sht/wigner.hpp``, line 287).
    """
    scale = math.sqrt(float(2 * k + 1) / ((k + m + 1) * (k - m + 1)))
    return scale * _u_km_0(k, m, tc)


@njit(cache=True, nogil=True)
def _a_km_1(k: int, m: int) -> float:
    """Return the seed coefficient ``a_{k,m}`` at beta = pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``.

    Returns
    -------
    a
        ``sqrt((2k+1) / ((k+m+1)(k-m+1))) * u_km_1(k, m)``
        (``include/sht/wigner.hpp``, line 288).
    """
    scale = math.sqrt(float(2 * k + 1) / ((k + m + 1) * (k - m + 1)))
    return scale * _u_km_1(k, m)


@njit(cache=True, nogil=True)
def _a_km_2(k: int, m: int, t: float) -> float:
    """Return the seed coefficient ``a_{k,m}`` for beta > pi/2.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``.
    t
        ``cos(beta)``.

    Returns
    -------
    a
        ``sqrt((2k+1) / ((k+m+1)(k-m+1))) * u_km_2(k, m, t)``
        (``include/sht/wigner.hpp``, line 289).
    """
    scale = math.sqrt(float(2 * k + 1) / ((k + m + 1) * (k - m + 1)))
    return scale * _u_km_2(k, m, t)


@njit(cache=True, nogil=True)
def _e_km(k: int, m: int) -> float:
    """Return the seed coefficient ``e_{k,m}``.

    Parameters
    ----------
    k, m
        The two orders of ``d^k_{k,m}``, with ``0 <= m <= k``.

    Returns
    -------
    e
        ``sqrt((2k)! / ((k+m)! (k-m)!))``, accumulated as the product
        of ``sqrt(l (2l-1) / (2 (l+m) (l-m))) * 2`` over
        ``l = m+1 ... k``, Fukushima equation 21
        (``include/sht/wigner.hpp``, lines 296-300). It satisfies
        ``d^k_{k,m}(pi/2) = 2^-k e_km`` exactly, since the power of
        two is exact.
    """
    e_im = 1.0  # e_mm
    for i in range(m + 1, k + 1):
        e_im *= math.sqrt(float(i * (2 * i - 1)) / (2 * (i + m) * (i - m))) * 2
    return e_im  # e_km


# ------------------------ Scalar d and D ---------------------------- #


@njit(cache=True, nogil=True)
def _wigner_d_core(j: int, k: int, m: int, t: float) -> float:
    """Return ``d^j_{k,m}(beta)`` for the reduced index range.

    Parameters
    ----------
    j, k, m
        Degree and the two orders, required to satisfy
        ``0 <= m <= k <= j``. Only ``j < k`` is checked, which
        returns NaN.
    t
        ``cos(beta)`` for a non-negative ``beta``.

    Returns
    -------
    d
        ``d^j_{k,m}(beta)``.

    Notes
    -----
    The non-recursive body of ``d()`` (``include/sht/wigner.hpp``,
    lines 317-370): the seed ``d^k_{k,m} = c2^(k+m) s2^(k-m) e_km``
    with ``c2 = sqrt((1+t)/2)`` and ``s2 = sqrt((1-t)/2)``, then
    ``d^(k+1)_{k,m} = d^k_{k,m} a_km``, then the three term
    recursion in degree, each with the three way branch on the sign
    of ``t`` (type 0 for ``t > 0``, type 1 for ``t == 0``, type 2 for
    ``t < 0``).

    The seed is where the missing extended exponent arithmetic bites:
    the two powers are formed before the multiplication by ``e_km``
    (which reaches about 1e71), so the product underflows to exactly
    ``0.0`` for large ``k + m`` at large ``beta``.
    """
    if j < k:
        return _NAN

    # determine if beta is < (0), > (2), or = (1) to pi/2
    tc = 1.0 - t

    # powers of cos/sin of beta / 2, always positive since 0 <= beta
    # <= pi at this point
    c2 = math.sqrt((1.0 + t) / 2)
    s2 = math.sqrt((1.0 - t) / 2)
    cn = c2 ** float(k + m)  # equation 20 for n = k + m
    sn = s2 ** float(k - m)  # equation 20 for n = k - m

    # first term of the three term recursion, equation 18
    d_kkm = cn * sn * _e_km(k, m)
    if j == k:
        return d_kkm

    # second term of the three term recursion, equation 19
    if t > 0:
        a_km = _a_km_0(k, m, tc)
    elif t < 0:
        a_km = _a_km_2(k, m, t)
    else:
        a_km = _a_km_1(k, m)
    d_k1km = d_kkm * a_km
    if j == k + 1:
        return d_k1km

    # recursively compute by degree to j, equation 10
    d_ikm = d_k1km
    d_i2km = d_kkm
    d_i1km = d_k1km
    for i in range(k + 2, j + 1):
        if t > 0:
            a_jkm = _a_jkm_0(i, k, m, tc)
        elif t < 0:
            a_jkm = _a_jkm_2(i, k, m, t)
        else:
            a_jkm = _a_jkm_1(i, k, m)
        d_ikm = a_jkm * d_i1km - _b_jkm(i, k, m) * d_i2km
        d_i2km = d_i1km
        d_i1km = d_ikm
    return d_ikm


@njit(cache=True, nogil=True)
def wigner_d(j: int, k: int, m: int, cos_beta: float, negative_beta: bool) -> float:
    """Return the Wigner (lowercase) d function ``d^j_{k,m}(beta)``.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.
    cos_beta
        ``cos(beta)``, which must be in [-1, 1]. ``beta`` itself must
        be in [-pi, pi], which callers ensure with
        :func:`kikuchipy.indexing._spherical._euler.wrap_beta`.
    negative_beta
        Whether ``beta`` is negative, i.e. ``signbit(beta)``.

    Returns
    -------
    d
        ``d^j_{k,m}(beta)`` in the Fukushima/Mathematica convention
        of the module docstring, or NaN when
        ``j < max(|k|, |m|)``.

    Notes
    -----
    Port of ``d(j, k, m, t, nB)`` (``include/sht/wigner.hpp``, lines
    298-371). The symmetry reduction of lines 300-315 to
    ``0 <= m <= k <= j`` is written iteratively rather than
    recursively (Numba's self recursion support is fragile), which
    accumulates the signs of Fukushima equations 5-9

    .. code-block::

        d^j_{ k, m}(-beta) =            d^j_{m,k}(     beta)
        d^j_{-k,-m}( beta) = (-1)^(k-m) d^j_{k,m}(     beta)
        d^j_{ k,-m}( beta) = (-1)^(j+k) d^j_{k,m}(pi - beta)
        d^j_{-k, m}( beta) = (-1)^(j+m) d^j_{k,m}(pi - beta)
        d^j_{ m, k}( beta) = (-1)^(k-m) d^j_{k,m}(     beta)

    before calling :func:`_wigner_d_core`.

    Examples
    --------
    >>> from kikuchipy.indexing._spherical._wigner import wigner_d
    >>> round(wigner_d(1, 1, 0, 0.0, False), 12)
    0.707106781187
    >>> round(wigner_d(1, 0, 1, 0.0, False), 12)
    -0.707106781187
    >>> wigner_d(1, 2, 0, 0.5, False)
    nan
    """
    # reduce to 0 <= m <= k <= j and beta >= 0 with equations 5-9,
    # accumulating the signs instead of recursing
    t = cos_beta
    sign = 1.0
    if negative_beta:  # equation 5
        k, m = m, k
    if k < 0 and m < 0:  # equation 6
        if (k - m) % 2 != 0:
            sign = -sign
        k = -k
        m = -m
    elif m < 0:  # equation 7
        if (j + k) % 2 != 0:
            sign = -sign
        m = -m
        t = -t
    elif k < 0:  # equation 8
        if (j + m) % 2 != 0:
            sign = -sign
        k = -k
        t = -t
    if k < m:  # equation 9
        if (k - m) % 2 != 0:
            sign = -sign
        k, m = m, k
    return _wigner_d_core(j, k, m, t) * sign


@njit(cache=True, nogil=True)
def wigner_d_half_pi(j: int, k: int, m: int) -> float:
    """Return ``d^j_{k,m}(pi/2)``.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.

    Returns
    -------
    d
        ``d^j_{k,m}(pi/2)``, or NaN when ``j < max(|k|, |m|)``.

    Notes
    -----
    Port of the ``beta = pi/2`` overload ``d(j, k, m)``
    (``include/sht/wigner.hpp``, lines 380-409), which seeds the
    recursion with the closed form ``2^-k e_km`` instead of the
    powers of ``cos(beta/2)`` and ``sin(beta/2)``. It therefore
    agrees with ``wigner_d(j, k, m, 0.0, False)`` only to rounding
    (7.2e-16 over a ``bw`` 15 table), not bitwise.
    """
    # reduce to 0 <= m <= k <= j, where pi - beta == beta collapses
    # equations 7 and 8 onto the same value
    sign = 1.0
    if k < 0 and m < 0:
        if (k - m) % 2 != 0:
            sign = -sign
        k = -k
        m = -m
    elif m < 0:
        if (j + k + 2 * m) % 2 != 0:
            sign = -sign
        m = -m
    elif k < 0:
        if (j + 2 * k + 3 * m) % 2 != 0:
            sign = -sign
        k = -k
    if k < m:
        if (k - m) % 2 != 0:
            sign = -sign
        k, m = m, k

    if j < k:
        return _NAN

    # first two terms of the three term recursion, equations 18-19
    d_kkm = 2.0 ** float(-k) * _e_km(k, m)
    if j == k:
        return d_kkm * sign
    d_k1km = d_kkm * _a_km_1(k, m)
    if j == k + 1:
        return d_k1km * sign

    # recursively compute by degree to j, equation 10
    d_ikm = d_k1km
    d_i2km = d_kkm
    d_i1km = d_k1km
    for i in range(k + 2, j + 1):
        d_ikm = _a_jkm_1(i, k, m) * d_i1km - _b_jkm(i, k, m) * d_i2km
        d_i2km = d_i1km
        d_i1km = d_ikm
    return d_ikm * sign


@njit(cache=True, nogil=True)
def wigner_d_sign(j: int, k: int, m: int) -> int:
    """Return the sign relating ``d^j_{k,m}`` to ``d^j_{|k|,|m|}``.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.

    Returns
    -------
    sign
        ``+1`` or ``-1`` such that
        ``wigner_d_sign(j, k, m) * d^j_{|k|,|m|}(pi/2) ==
        d^j_{k,m}(pi/2)``.

    Notes
    -----
    Port of ``dSign()`` (``include/sht/wigner.hpp``, lines 416-425).
    The relation holds at ``pi/2`` only, where ``pi - beta == beta``
    collapses equations 7 and 8 onto the same table slot.
    """
    if k < 0 and m < 0:  # equation 6
        return 1 if (k - m) % 2 == 0 else -1
    elif m < 0:  # equation 7
        return 1 if (j + k) % 2 == 0 else -1
    elif k < 0:  # equation 8
        return 1 if (j + m) % 2 == 0 else -1
    return 1


@njit(cache=True, nogil=True)
def wigner_D(j: int, k: int, m: int, zyz: np.ndarray) -> complex:
    """Return the Wigner (uppercase) D function ``D^j_{k,m}(zyz)``.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.
    zyz
        Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in radians
        in an array of shape ``(3,)`` and 64-bit floating point data
        type. A 32-bit array is accepted by Numba, which compiles a
        separate specialization for it, but costs about 3e-8 of
        accuracy: this kernel has no Python wrapper to widen it.

    Returns
    -------
    value
        ``d^j_{k,m}(beta) exp(i (m alpha + k gamma))``, i.e.
        Mathematica's ``WignerD[{j, k, m}, gamma, beta, alpha]``, or
        NaN when ``j < max(|k|, |m|)``.

    Notes
    -----
    Port of ``D()`` (``include/sht/wigner.hpp``, lines 436-439), with
    the deliberate addition of the ``[-pi, pi]`` wrap of ``beta``
    (``include/sht/sht_xcorr.hpp``, lines 895-899) before ``cos`` and
    ``signbit`` are taken, so that the result is periodic in ``beta``
    as the mathematics requires.

    The real and imaginary parts are scaled separately because the
    C++ ``std::complex<Real> * Real`` of line 438 scales
    componentwise. Written as ``complex(...) * d`` instead, both
    Numba and CPython promote ``d`` to a complex number and use the
    four multiplication form, which turns the ``-0.0`` of a slot
    with ``sin(total) == 0.0`` and ``d < 0`` into ``+0.0`` (23 of
    9464 sampled slots, all with ``k == m == 0``).
    """
    beta = _wrap_beta(zyz[1])
    total = zyz[0] * m + zyz[2] * k
    d = wigner_d(j, k, m, math.cos(beta), math.copysign(1.0, beta) < 0.0)
    # componentwise, as C++ scales a complex by a real
    return complex(math.cos(total) * d, math.sin(total) * d)


# --------------------------- Table kernels -------------------------- #


@njit(cache=True, nogil=True)
def _wigner_d_table_kernel(
    bandwidth: int, t: float, negative_beta: bool, table: np.ndarray
) -> None:
    """Fill a Wigner d table for one beta, in place.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree.
    t
        ``cos(beta)``.
    negative_beta
        Whether ``beta`` is negative.
    table
        NaN filled array of shape
        ``(bandwidth, bandwidth, bandwidth, 2)``, 64-bit floating
        point and C-contiguous, written in place on the defined slots
        ``j >= max(k, m)`` only.

    Notes
    -----
    Port of ``dTable()`` (``include/sht/wigner.hpp``, lines 452-559).
    The ``+t`` and ``-t`` recursions run simultaneously into slots 0
    and 1, and each ``(k, m)`` pair also writes the swapped
    ``(m, k)`` pair through equation 9, whose sign
    ``(-1)^(k-m)`` is swapped with the identity sign when
    ``negative_beta`` is true (line 499).

    The branch selection is ``isType0 = not signbit(t)`` (line 467),
    which groups ``beta == pi/2`` with ``beta < pi/2``; it then picks
    the ``a_km``/``a_jkm`` pair and the two arguments
    ``t0 = tc if isType0 else t`` and
    ``tN = -t if isType0 else 1 + t``. Integer powers of
    ``cos(beta/2)`` and ``sin(beta/2)`` are tabulated once
    (``2 * bandwidth`` each) and read at ``k + m`` and ``k - m``,
    with the roles of the two exchanged for the ``-t`` slot.
    """
    # which branch is needed, grouping beta == pi/2 with beta < pi/2,
    # and the cosine and sine of the half angle
    is_type_0 = not (math.copysign(1.0, t) < 0.0)
    tc = 1.0 - t
    tc_n = 1.0 + t  # tc for -t
    c2 = math.sqrt(tc_n / 2)
    s2 = math.sqrt(tc / 2)
    t0 = tc if is_type_0 else t
    t_n = -t if is_type_0 else tc_n

    # integer powers of c2 and s2, equation 20
    pc2 = np.empty(bandwidth * 2)
    ps2 = np.empty(bandwidth * 2)
    for i in range(bandwidth * 2):
        pc2[i] = c2 ** float(i)
        ps2[i] = s2 ** float(i)

    for k in range(bandwidth):
        for m in range(k + 1):
            # sign change for swapping k and m, equation 9
            sign = 1.0
            sign_n = 1.0 if (k - m) % 2 == 0 else -1.0
            if negative_beta:
                sign, sign_n = sign_n, sign

            cn = pc2[k + m]
            sn = ps2[k - m]
            cn_n = ps2[k + m]
            sn_n = pc2[k - m]

            # first term of the three term recursion, equation 18
            ekm = _e_km(k, m)
            d_kkm = cn * sn * ekm
            d_kkm_n = cn_n * sn_n * ekm
            table[k, m, k, 0] = d_kkm * sign
            table[m, k, k, 0] = d_kkm * sign_n
            table[k, m, k, 1] = d_kkm_n * sign
            table[m, k, k, 1] = d_kkm_n * sign_n

            if k + 1 < bandwidth:
                # second term of the recursion, equation 19
                if is_type_0:
                    a_km = _a_km_0(k, m, t0)
                    a_km_n = _a_km_2(k, m, t_n)
                else:
                    a_km = _a_km_2(k, m, t0)
                    a_km_n = _a_km_0(k, m, t_n)
                d_k1km = d_kkm * a_km
                d_k1km_n = d_kkm_n * a_km_n
                table[k, m, k + 1, 0] = d_k1km * sign
                table[m, k, k + 1, 0] = d_k1km * sign_n
                table[k, m, k + 1, 1] = d_k1km_n * sign
                table[m, k, k + 1, 1] = d_k1km_n * sign_n

                if k + 2 < bandwidth:
                    # recursively compute by degree, equation 10
                    d_i2km = d_kkm
                    d_i1km = d_k1km
                    d_i2km_n = d_kkm_n
                    d_i1km_n = d_k1km_n
                    for i in range(k + 2, bandwidth):
                        if is_type_0:
                            a_jkm = _a_jkm_0(i, k, m, t0)
                            a_jkm_n = _a_jkm_2(i, k, m, t_n)
                        else:
                            a_jkm = _a_jkm_2(i, k, m, t0)
                            a_jkm_n = _a_jkm_0(i, k, m, t_n)
                        b_ikm = _b_jkm(i, k, m)
                        d_ikm = a_jkm * d_i1km - b_ikm * d_i2km
                        d_i2km = d_i1km
                        d_i1km = d_ikm
                        d_ikm_n = a_jkm_n * d_i1km_n - b_ikm * d_i2km_n
                        d_i2km_n = d_i1km_n
                        d_i1km_n = d_ikm_n
                        table[k, m, i, 0] = d_i1km * sign
                        table[m, k, i, 0] = d_i1km * sign_n
                        table[k, m, i, 1] = d_i1km_n * sign
                        table[m, k, i, 1] = d_i1km_n * sign_n


@njit(cache=True, nogil=True)
def _wigner_d_table_factors_kernel(
    bandwidth: int, e_km: np.ndarray, w_jkm: np.ndarray, b_jkm: np.ndarray
) -> None:
    """Fill the precomputed coefficient tables, in place.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree.
    e_km
        NaN filled array of shape ``(bandwidth, bandwidth)`` and
        64-bit floating point data type, written at ``[k, m]`` for
        ``m <= k``.
    w_jkm, b_jkm
        NaN filled arrays of shape
        ``(bandwidth, bandwidth, bandwidth)`` and 64-bit floating
        point data type, written at ``[k, m, i]`` for ``m <= k`` and
        ``i >= k + 2``.

    Notes
    -----
    Port of ``dTablePreBuild()`` (``include/sht/wigner.hpp``, lines
    678-691).
    """
    for k in range(bandwidth):
        for m in range(k + 1):
            e_km[k, m] = _e_km(k, m)
            for i in range(k + 2, bandwidth):
                w_jkm[k, m, i] = _w_jkm(i, k, m)
                b_jkm[k, m, i] = _b_jkm(i, k, m)


@njit(cache=True, nogil=True)
def _wigner_d_table_pre_kernel(
    bandwidth: int,
    t: float,
    negative_beta: bool,
    table: np.ndarray,
    e_km: np.ndarray,
    w_jkm: np.ndarray,
    b_jkm: np.ndarray,
) -> None:
    """Fill a Wigner d table from precomputed factors, in place.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree.
    t
        ``cos(beta)``.
    negative_beta
        Whether ``beta`` is negative.
    table
        Array of shape ``(bandwidth, bandwidth, bandwidth, 2)``,
        64-bit floating point and C-contiguous, whose undefined slots
        are already NaN. Only the defined slots are written.
    e_km, w_jkm, b_jkm
        The three tables of :func:`wigner_d_table_factors`.

    Notes
    -----
    Port of ``dTablePre()`` (``include/sht/wigner.hpp``, lines
    575-671). It is the same recursion as
    :func:`_wigner_d_table_kernel` with ``e_km``, ``w_jkm`` and
    ``b_jkm`` read from the tables instead of recomputed, and gives
    bitwise identical results.
    """
    # which branch is needed, grouping beta == pi/2 with beta < pi/2,
    # and the cosine and sine of the half angle
    is_type_0 = not (math.copysign(1.0, t) < 0.0)
    tc = 1.0 - t
    tc_n = 1.0 + t  # tc for -t
    c2 = math.sqrt(tc_n / 2)
    s2 = math.sqrt(tc / 2)
    t0 = tc if is_type_0 else t
    t_n = -t if is_type_0 else tc_n

    # integer powers of c2 and s2, equation 20
    pc2 = np.empty(bandwidth * 2)
    ps2 = np.empty(bandwidth * 2)
    for i in range(bandwidth * 2):
        pc2[i] = c2 ** float(i)
        ps2[i] = s2 ** float(i)

    for k in range(bandwidth):
        for m in range(k + 1):
            # sign change for swapping k and m, equation 9
            sign = 1.0
            sign_n = 1.0 if (k - m) % 2 == 0 else -1.0
            if negative_beta:
                sign, sign_n = sign_n, sign

            cn = pc2[k + m]
            sn = ps2[k - m]
            cn_n = ps2[k + m]
            sn_n = pc2[k - m]

            # first term of the three term recursion, equation 18
            ekm = e_km[k, m]
            d_kkm = cn * sn * ekm
            d_kkm_n = cn_n * sn_n * ekm
            table[k, m, k, 0] = d_kkm * sign
            table[m, k, k, 0] = d_kkm * sign_n
            table[k, m, k, 1] = d_kkm_n * sign
            table[m, k, k, 1] = d_kkm_n * sign_n

            if k + 1 < bandwidth:
                # second term of the recursion, equation 19
                if is_type_0:
                    a_km = _a_km_0(k, m, t0)
                    a_km_n = _a_km_2(k, m, t_n)
                else:
                    a_km = _a_km_2(k, m, t0)
                    a_km_n = _a_km_0(k, m, t_n)
                d_k1km = d_kkm * a_km
                d_k1km_n = d_kkm_n * a_km_n
                table[k, m, k + 1, 0] = d_k1km * sign
                table[m, k, k + 1, 0] = d_k1km * sign_n
                table[k, m, k + 1, 1] = d_k1km_n * sign
                table[m, k, k + 1, 1] = d_k1km_n * sign_n

                if k + 2 < bandwidth:
                    # recursively compute by degree, equation 10
                    d_i2km = d_kkm
                    d_i1km = d_k1km
                    d_i2km_n = d_kkm_n
                    d_i1km_n = d_k1km_n
                    for i in range(k + 2, bandwidth):
                        w = w_jkm[k, m, i]
                        b_ikm = b_jkm[k, m, i]
                        if is_type_0:
                            a_jkm = _a_jkm_0_pre(w, i, k, m, t0)
                            a_jkm_n = _a_jkm_2_pre(w, i, k, m, t_n)
                        else:
                            a_jkm = _a_jkm_2_pre(w, i, k, m, t0)
                            a_jkm_n = _a_jkm_0_pre(w, i, k, m, t_n)
                        d_ikm = a_jkm * d_i1km - b_ikm * d_i2km
                        d_i2km = d_i1km
                        d_i1km = d_ikm
                        d_ikm_n = a_jkm_n * d_i1km_n - b_ikm * d_i2km_n
                        d_i2km_n = d_i1km_n
                        d_i1km_n = d_ikm_n
                        table[k, m, i, 0] = d_i1km * sign
                        table[m, k, i, 0] = d_i1km * sign_n
                        table[k, m, i, 1] = d_i1km_n * sign
                        table[m, k, i, 1] = d_i1km_n * sign_n


@njit(cache=True, nogil=True)
def _wigner_d_half_pi_table_kernel(
    bandwidth: int, table: np.ndarray, transpose: bool
) -> None:
    """Fill a Wigner d table at beta = pi/2, in place.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree.
    table
        NaN filled array of shape
        ``(bandwidth, bandwidth, bandwidth)``, 64-bit floating point
        and C-contiguous, written in place on the defined slots only.
    transpose
        Whether to write ``d^j_{k,m}(pi/2)`` at ``[m, k, j]`` instead
        of ``[k, m, j]``.

    Notes
    -----
    Port of the ``pi/2`` overload ``dTable(jMax, table, trans)``
    (``include/sht/wigner.hpp``, lines 699-761), which seeds with the
    closed form ``2^-k e_km`` and fills the mirror slot with the
    ``(-1)^(k-m)`` sign of equation 9.
    """
    # compute d^j_{k,m} for k >= m >= 0 and fill the mirror slot with
    # the (-1)^(k-m) sign of equation 9
    for k in range(bandwidth):
        for m in range(k + 1):
            km = (k - m) % 2 == 0

            # d^k_{k,m} from the closed form, equation 18
            d_kkm = 2.0 ** float(-k) * _e_km(k, m)
            d_kmk = d_kkm if km else -d_kkm
            if transpose:
                table[m, k, k] = d_kkm
                table[k, m, k] = d_kmk
            else:
                table[k, m, k] = d_kkm
                table[m, k, k] = d_kmk
            if k + 1 == bandwidth:
                continue

            # d^{k+1}_{k,m} from the recursion, equation 19
            d_k1km = d_kkm * _a_km_1(k, m)
            d_k1mk = d_k1km if km else -d_k1km
            if transpose:
                table[m, k, k + 1] = d_k1km
                table[k, m, k + 1] = d_k1mk
            else:
                table[k, m, k + 1] = d_k1km
                table[m, k, k + 1] = d_k1mk
            if k + 2 == bandwidth:
                continue

            # higher degrees from the three term recursion, eq 10
            d_j1km = d_k1km
            d_j2km = d_kkm
            for j in range(k + 2, bandwidth):
                d_jkm = _a_jkm_1(j, k, m) * d_j1km - _b_jkm(j, k, m) * d_j2km
                d_jmk = d_jkm if km else -d_jkm
                if transpose:
                    table[m, k, j] = d_jkm
                    table[k, m, j] = d_jmk
                else:
                    table[k, m, j] = d_jkm
                    table[m, k, j] = d_jmk
                d_j2km = d_j1km
                d_j1km = d_jkm


@njit(cache=True, nogil=True)
def _rotate_harmonics_kernel(
    alm: np.ndarray,
    alpha: float,
    gamma: float,
    table: np.ndarray,
    out: np.ndarray,
) -> None:
    """Accumulate the rotated harmonic coefficients, in place.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[n, l]`` of shape ``(bw, bw)`` and
        128-bit complex data type, C-contiguous.
    alpha, gamma
        The outer ZYZ Euler angles in radians.
    table
        Wigner d table of shape ``(bw, bw, bw, 2)`` for the middle
        angle, as returned by :func:`wigner_d_table`.
    out
        Zeroed array of shape ``(bw, bw)`` and 128-bit complex data
        type to accumulate ``blm[m, l]`` into. It **must not alias**
        ``alm``: the C++ accumulates into a ``almRot`` temporary
        "in case alm == blm" (line 772) and copies at the end, while
        this kernel accumulates into ``out`` directly, so an aliased
        buffer is read after it has been written. The one caller,
        :func:`rotate_harmonics`, passes a fresh zeroed array.

    Notes
    -----
    Port of ``rotateHarmonics()`` (``include/sht/wigner.hpp``, lines
    778-797). The two complex exponentials are named by content
    here: ``exp_m_gamma = exp(i m gamma)`` is the C++ ``expAlpha``
    (line 780) and ``exp_n_alpha = exp(i n alpha)`` is its
    ``expGamma`` (line 782); the mathematics is the ``D`` of the
    module docstring either way.

    The ``+n`` term reads ``table[m, n, j, 0]`` and the ``-n`` term
    ``table[m, n, j, 1]`` with the sign ``(-1)^(j+m+n)`` (line 794),
    which comes from combining
    ``a^l_{-n} = (-1)^n conj(a^l_n)`` with
    ``d^l_{m,-n}(beta) = (-1)^(l+m) d^l_{m,n}(pi - beta)``. The first
    two table indices are ``(m, n)`` and never ``(n, m)``: the
    transposed read is conjugation by the two fold about z and
    survives every group theoretic identity, so it is caught only by
    the tests which pin ``D`` itself.
    """
    bandwidth = alm.shape[0]
    for m in range(bandwidth):
        exp_m_gamma = complex(math.cos(gamma * m), math.sin(gamma * m))
        for n in range(bandwidth):
            exp_n_alpha = complex(math.cos(alpha * n), math.sin(alpha * n))
            for j in range(max(m, n), bandwidth):
                a_n = alm[n, j] * exp_n_alpha
                rr = exp_m_gamma.real * a_n.real
                ri = exp_m_gamma.real * a_n.imag
                ir = exp_m_gamma.imag * a_n.real
                ii = exp_m_gamma.imag * a_n.imag

                # exp_m_gamma * a_n and exp_m_gamma * conj(a_n), the
                # +n and -n terms of the sum
                vp = complex(rr - ii, ir + ri)
                vc = complex(rr + ii, ir - ri)

                dmn0 = table[m, n, j, 0]  # d^j_{m,n}(     beta)
                dmn1 = table[m, n, j, 1]  # d^j_{m,n}(pi - beta)
                out[m, j] += vp * dmn0
                if n > 0:
                    sign = 1.0 if (j + m + n) % 2 == 0 else -1.0
                    out[m, j] += vc * (dmn1 * sign)


# ------------------------- Table wrappers --------------------------- #


def wigner_d_table(bandwidth: int, cos_beta: float, negative_beta: bool) -> np.ndarray:
    """Return a table of ``d^j_{k,m}`` at beta and pi - beta.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree, which must be at least one.
    cos_beta
        ``cos(beta)``, which must be in [-1, 1].
    negative_beta
        Whether ``beta`` is negative, i.e. ``signbit(beta)``.

    Returns
    -------
    table
        C-contiguous array of shape
        ``(bandwidth, bandwidth, bandwidth, 2)`` and 64-bit floating
        point data type with ``table[k, m, j, 0] = d^j_{k,m}(beta)``
        and ``table[k, m, j, 1] = d^j_{k,m}(pi - beta)``, and NaN on
        the undefined slots ``j < max(k, m)``.

    Raises
    ------
    ValueError
        If ``bandwidth`` is smaller than one or ``|cos_beta| > 1``.

    Notes
    -----
    Port of ``dTable()`` (``include/sht/wigner.hpp``, lines 452-559).
    Only non-negative ``k`` and ``m`` are stored; negative orders
    follow from

    .. code-block::

        d^j_{-k,-m}( beta) = (-1)^(k-m) d^j_{k,m}(     beta)
        d^j_{ k,-m}( beta) = (-1)^(j+k) d^j_{k,m}(pi - beta)
        d^j_{-k, m}( beta) = (-1)^(j+m) d^j_{k,m}(pi - beta)

    and ``negative_beta`` swaps the ``(k, m)`` and ``(m, k)`` slots,
    i.e. ``table(nB)[k, m, j, s] == table(not nB)[m, k, j, s]``.

    The table is ``2 bandwidth^3`` doubles, and the NaN fill of the
    allocation costs about 0.65 times the recursion itself (0.90 ms
    of 1.39 ms at ``bw`` 68). Use :func:`wigner_d_table_pre` with
    ``out=`` to reuse a buffer when a table is needed repeatedly.
    """
    if bandwidth < 1:
        raise ValueError("`bandwidth` must be at least one")
    if not -1.0 <= cos_beta <= 1.0:
        raise ValueError("`cos_beta` must be in [-1, 1]")
    table = np.full((bandwidth, bandwidth, bandwidth, 2), np.nan)
    _wigner_d_table_kernel(bandwidth, float(cos_beta), bool(negative_beta), table)
    return table


def wigner_d_table_factors(
    bandwidth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the beta independent coefficient tables.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree, which must be at least one.

    Returns
    -------
    e_km
        Array of shape ``(bandwidth, bandwidth)`` and 64-bit floating
        point data type with ``e_km[k, m] = e_{k,m}`` for ``m <= k``
        and NaN elsewhere.
    w_jkm
        Array of shape ``(bandwidth, bandwidth, bandwidth)`` and
        64-bit floating point data type with
        ``w_jkm[k, m, i] = w_{i,k,m}`` for ``m <= k`` and
        ``i >= k + 2``, NaN elsewhere.
    b_jkm
        The same for ``b_{i,k,m}``.

    Raises
    ------
    ValueError
        If ``bandwidth`` is smaller than one.

    Notes
    -----
    Port of ``dTablePreBuild()`` (``include/sht/wigner.hpp``, lines
    678-691). The tables are not memoized here; Phase 4's correlator
    holds them in its own constants, as
    ``include/sht/sht_xcorr.hpp`` lines 360-370 do.
    """
    if bandwidth < 1:
        raise ValueError("`bandwidth` must be at least one")
    e_km = np.full((bandwidth, bandwidth), np.nan)
    w_jkm = np.full((bandwidth, bandwidth, bandwidth), np.nan)
    b_jkm = np.full((bandwidth, bandwidth, bandwidth), np.nan)
    _wigner_d_table_factors_kernel(bandwidth, e_km, w_jkm, b_jkm)
    return e_km, w_jkm, b_jkm


def wigner_d_table_pre(
    bandwidth: int,
    cos_beta: float,
    negative_beta: bool,
    e_km: np.ndarray,
    w_jkm: np.ndarray,
    b_jkm: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return a Wigner d table built from precomputed factors.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree, which must be at least one.
    cos_beta
        ``cos(beta)``, which must be in [-1, 1].
    negative_beta
        Whether ``beta`` is negative, i.e. ``signbit(beta)``.
    e_km, w_jkm, b_jkm
        The three tables of :func:`wigner_d_table_factors` built for
        the same ``bandwidth``. Their shapes and data type are
        checked, since the kernel reads them without bounds checking
        (Numba's ``BOUNDSCHECK`` is off) and an undersized table is
        therefore silently out of bounds rather than an error.
    out
        Buffer to write into. If not given, a NaN filled array is
        allocated. **All undefined slots of a given buffer must
        already be NaN**, which they are when it came from a previous
        call or from ``np.full(..., np.nan)``, because the kernel
        never writes them. The check is a tripwire on two
        representative slots rather than a full scan, so an
        ``np.empty()`` buffer is rejected but a partially poisoned
        one is not. The buffer is caller owned **per thread**: every
        kernel is ``nogil=True``, so two threads writing one buffer
        interleave and leave a table which matches neither beta. One
        buffer per Newton refinement is the intended re-use, one
        buffer for a thread pool is not.

    Returns
    -------
    table
        ``out`` if given, else a new array; the layout is that of
        :func:`wigner_d_table` and the values are bitwise equal to
        it.

    Raises
    ------
    ValueError
        If ``bandwidth`` is smaller than one, ``|cos_beta| > 1``, any
        of ``e_km``, ``w_jkm`` and ``b_jkm`` was not built for
        ``bandwidth`` or is not 64-bit floating point, or ``out`` has
        the wrong shape or data type, is not C-contiguous, or has a
        non-NaN value in one of the two representative undefined
        slots ``out[bandwidth - 1, 0, 0, 0]`` and
        ``out[0, bandwidth - 1, 0, 1]`` (checked for
        ``bandwidth >= 2``, since ``bandwidth == 1`` has no undefined
        slot).

    Notes
    -----
    Port of ``dTablePre()`` (``include/sht/wigner.hpp``, lines
    575-671). Reusing ``out`` is what Phase 7's Newton refinement
    does, one call per iteration, which is why the NaN fill is not
    repeated.

    The C++ forms its own flat index ``k jMax^2 + m jMax + i`` into
    the factor tables, so a ``jMax`` smaller than the bandwidth they
    were built for reads them at the wrong stride
    (``include/sht/sht_xcorr.hpp`` line 913 passes ``mBW`` to tables
    built at ``bw`` on line 369; every call site currently has
    ``mBW == bw``). This port indexes ``w_jkm[k, m, i]`` through the
    arrays' own strides and so would not reproduce that read, which
    is why the tables must match ``bandwidth`` here.
    """
    if bandwidth < 1:
        raise ValueError("`bandwidth` must be at least one")
    if not -1.0 <= cos_beta <= 1.0:
        raise ValueError("`cos_beta` must be in [-1, 1]")
    # the kernel reads the factors without bounds checking, so an
    # undersized table is out of bounds rather than an error
    for name, factor, factor_shape in (
        ("e_km", e_km, (bandwidth, bandwidth)),
        ("w_jkm", w_jkm, (bandwidth,) * 3),
        ("b_jkm", b_jkm, (bandwidth,) * 3),
    ):
        if (
            getattr(factor, "shape", None) != factor_shape
            or getattr(factor, "dtype", None) != np.float64
        ):
            raise ValueError(
                f"`{name}` must have shape {factor_shape} and a 64-bit floating "
                "point data type, i.e. come from "
                "`wigner_d_table_factors(bandwidth)`"
            )
    shape = (bandwidth, bandwidth, bandwidth, 2)
    if out is None:
        table = np.full(shape, np.nan)
    else:
        table = out
        if table.shape != shape:
            raise ValueError(f"`out` must have shape {shape}")
        if table.dtype != np.float64:
            raise ValueError("`out` must have a 64-bit floating point data type")
        if not table.flags.c_contiguous:
            raise ValueError("`out` must be C-contiguous")
        # tripwire on two slots the kernel never writes, which are
        # undefined for every bandwidth of at least two
        if bandwidth >= 2 and not (
            np.isnan(table[bandwidth - 1, 0, 0, 0])
            and np.isnan(table[0, bandwidth - 1, 0, 1])
        ):
            raise ValueError("all undefined slots of `out` must already be NaN")
    _wigner_d_table_pre_kernel(
        bandwidth, float(cos_beta), bool(negative_beta), table, e_km, w_jkm, b_jkm
    )
    return table


def wigner_d_half_pi_table(bandwidth: int, transpose: bool) -> np.ndarray:
    """Return a table of ``d^j_{k,m}(pi/2)``.

    Parameters
    ----------
    bandwidth
        Exclusive maximum degree, which must be at least one.
    transpose
        Whether to store ``d^j_{k,m}(pi/2)`` at ``[m, k, j]``
        (``True``) or at ``[k, m, j]`` (``False``). There is no
        default: callers state which layout they want. The correlator
        of Phase 4 allocates the transposed one
        (``include/sht/sht_xcorr.hpp``, line 368).

    Returns
    -------
    table
        C-contiguous array of shape
        ``(bandwidth, bandwidth, bandwidth)`` and 64-bit floating
        point data type, NaN on the undefined slots
        ``j < max(k, m)``.

    Raises
    ------
    ValueError
        If ``bandwidth`` is smaller than one.

    Notes
    -----
    Port of ``dTable(jMax, table, trans)``
    (``include/sht/wigner.hpp``, lines 699-761). The two layouts are
    exact transposes of each other, and within one layout
    ``table[k, m, j] == (-1)^(k-m) table[m, k, j]``.
    """
    if bandwidth < 1:
        raise ValueError("`bandwidth` must be at least one")
    table = np.full((bandwidth, bandwidth, bandwidth), np.nan)
    _wigner_d_half_pi_table_kernel(bandwidth, table, bool(transpose))
    return table


def rotate_harmonics(alm: np.ndarray, zyz: np.ndarray) -> np.ndarray:
    """Return the harmonic coefficients of a rotated function.

    Parameters
    ----------
    alm
        Harmonic coefficients ``alm[m, l]`` of a real function in an
        array-like of shape ``(bw, bw)``, the layout of
        :class:`kikuchipy.indexing._spherical._sht.
        SphericalHarmonicTransform`. It is cast to 128-bit complex
        and made C-contiguous. Only the upper triangle ``l >= m`` is
        read.
    zyz
        Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in radians
        in an array-like of shape ``(3,)``. ``beta`` is wrapped into
        [-pi, pi] first, so the result is periodic.

    Returns
    -------
    blm
        New array of the same shape and 128-bit complex data type
        with ``blm[m, l] = sum_(n=-l)^(l) a^l_n D^l_{m,n}(zyz)``,
        and exactly zero below the diagonal. The input is never
        written to.

    Raises
    ------
    ValueError
        If ``alm`` is not two-dimensional and square, or if ``zyz``
        does not have three elements.

    Notes
    -----
    Port of ``rotateHarmonics()`` (``include/sht/wigner.hpp``, lines
    769-799), with the ``beta`` wrap added. The synthesized function
    satisfies ``g(n) = f((~R) * n)`` with
    ``R = Rotation(zyz_to_quaternion(zyz))``, see the module
    docstring.

    This allocates a full ``2 bw^3`` double Wigner d table, i.e. 5.0
    MB at ``bw`` 68 and 906 MB at ``bw`` 384, so it is a test and
    visualization tool at master pattern bandwidths and never a per
    pattern operation.

    Examples
    --------
    >>> import numpy as np
    >>> from kikuchipy.indexing._spherical._wigner import (
    ...     rotate_harmonics,
    ... )
    >>> alm = np.zeros((3, 3), dtype=np.complex128)
    >>> alm[0, 0] = 1  # the rotation invariant l = 0 coefficient
    >>> blm = rotate_harmonics(alm, [0.3, 0.7, -1.2])
    >>> blm.shape
    (3, 3)
    >>> round(float(blm[0, 0].real), 12)
    1.0
    """
    coefficients = np.ascontiguousarray(alm, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError("`alm` must be two-dimensional and square")
    angles = np.asarray(zyz, dtype=np.float64)
    if angles.shape != (3,):
        raise ValueError("`zyz` must have three elements")

    beta = _wrap_beta(float(angles[1]))
    table = wigner_d_table(
        coefficients.shape[0], math.cos(beta), math.copysign(1.0, beta) < 0.0
    )
    blm = np.zeros_like(coefficients)
    _rotate_harmonics_kernel(
        coefficients, float(angles[0]), float(angles[2]), table, blm
    )
    return blm


# --------------------------- Derivatives ---------------------------- #


@njit(cache=True, nogil=True)
def wigner_d_prime(
    j: int, k: int, m: int, cos_beta: float, negative_beta: bool
) -> float:
    """Return the first derivative of ``d^j_{k,m}`` in beta.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.
    cos_beta
        ``cos(beta)``, which must be in (-1, 1). The end points are
        excluded: ``csc(beta)`` is infinite at ``beta = 0`` and
        ``beta = pi``, see Raises.
    negative_beta
        Whether ``beta`` is negative, i.e. ``signbit(beta)``.

    Returns
    -------
    d_prime
        ``(d/dbeta) d^j_{k,m}(beta)``, i.e. Mathematica's
        ``D[WignerD[{j, k, m}, beta], beta]``, or NaN when
        ``j < max(|k|, |m|)``.

    Raises
    ------
    ZeroDivisionError
        If ``|cos_beta| == 1``, i.e. at ``beta = 0`` or
        ``beta = pi``, where the ``csc`` prefactor divides by zero.
        The C++ divides in C and returns the IEEE ``+-inf`` or NaN
        there, while Numba's default ``error_model="python"`` raises,
        as the ``py_func`` does. Changing the error model would alter
        the provenance of every other value, so the domain is
        documented rather than widened; a Newton refinement which can
        land on a pole must keep ``beta`` off it.

    Notes
    -----
    Port of ``dPrime()`` (``include/sht/wigner.hpp``, lines
    814-822). The NaN for undefined indices is produced by an
    explicit guard rather than by ``sqrt()`` of a negative number, a
    recorded deviation with an identical result: Numba's
    ``math.sqrt`` returns NaN for a negative argument while Python's
    raises ``ValueError``, so the compiled kernel and its
    ``py_func`` would otherwise disagree.
    """
    if j < max(abs(k), abs(m)):
        return _NAN

    # prefactor, the same for all j, k and m at a given beta
    t = cos_beta
    csc = 1.0 / math.sqrt(1.0 - t * t) * (-1.0 if negative_beta else 1.0)

    d0_term = wigner_d(j, k, m, t, negative_beta) * (t * k - m) * csc
    if j == k:
        d1_term = 0.0
    else:
        d1_term = wigner_d(j, k + 1, m, t, negative_beta) * math.sqrt(
            float((j - k) * (j + k + 1))
        )
    return d0_term - d1_term


@njit(cache=True, nogil=True)
def wigner_d_prime2(
    j: int, k: int, m: int, cos_beta: float, negative_beta: bool
) -> float:
    """Return the second derivative of ``d^j_{k,m}`` in beta.

    Parameters
    ----------
    j
        Degree.
    k
        First order.
    m
        Second order.
    cos_beta
        ``cos(beta)``, which must be in (-1, 1). The end points are
        excluded: ``csc(beta)`` is infinite at ``beta = 0`` and
        ``beta = pi``, see Raises.
    negative_beta
        Whether ``beta`` is negative, i.e. ``signbit(beta)``.

    Returns
    -------
    d_prime2
        ``(d/dbeta)^2 d^j_{k,m}(beta)``, i.e. Mathematica's
        ``D[WignerD[{j, k, m}, beta], {beta, 2}]``, or NaN when
        ``j < max(|k|, |m|)``.

    Raises
    ------
    ZeroDivisionError
        If ``|cos_beta| == 1``, i.e. at ``beta = 0`` or
        ``beta = pi``, where the ``csc`` prefactor divides by zero.
        The C++ divides in C and returns the IEEE ``+-inf`` or NaN
        there, while Numba's default ``error_model="python"`` raises,
        as the ``py_func`` does. Changing the error model would alter
        the provenance of every other value, so the domain is
        documented rather than widened; a Newton refinement which can
        land on a pole must keep ``beta`` off it.

    Notes
    -----
    Port of ``dPrime2()`` (``include/sht/wigner.hpp``, lines
    837-852), with the same explicit NaN guard as
    :func:`wigner_d_prime` and one further recorded deviation: the
    C++ evaluates ``d2Coef = rjk * sqrt((j-k-1)(j+k+2))``
    unconditionally at line 845 and then discards it in the
    ``k + 1 >= j`` ternary at line 851, but that radicand is negative
    on **every** defined slot with ``k == j`` (it is ``-(j+k+2)``,
    i.e. -2, -4, -6, -8 at ``(0,0,0)``, ``(1,1,0)``, ``(2,2,1)``,
    ``(3,3,3)``). The port therefore evaluates the product only
    inside the branch that uses it, with the association of lines
    845 and 851 preserved, so the value is bitwise the C++ one
    wherever the C++ value is defined and the ``py_func`` does not
    raise.

    Phase 7's Newton refinement will not call this function; it
    inlines the table based formulas of
    ``include/sht/sht_xcorr.hpp`` lines 1009-1041, which the Phase 3
    tests pin against this one.
    """
    if j < max(abs(k), abs(m)):
        return _NAN

    # prefactor, the same for all j, k and m at a given beta
    t = cos_beta
    csc = 1.0 / math.sqrt(1.0 - t * t) * (-1.0 if negative_beta else 1.0)

    rjk = math.sqrt(float((j - k) * (j + k + 1)))
    d0_coefficient = (t * t * k * k + t * m * (1 - 2 * k) + (m * m - k)) * csc * csc
    d1_coefficient = rjk * (t * (1 + 2 * k) - 2 * m) * csc

    d0_term = wigner_d(j, k, m, t, negative_beta) * d0_coefficient
    if k >= j:
        d1_term = 0.0
    else:
        d1_term = wigner_d(j, k + 1, m, t, negative_beta) * d1_coefficient
    if k + 1 >= j:
        # the C++ forms d2Coef unconditionally, whose radicand is
        # negative on every defined slot with k == j
        d2_term = 0.0
    else:
        d2_coefficient = rjk * math.sqrt(float((j - k - 1) * (j + k + 2)))
        d2_term = wigner_d(j, k + 2, m, t, negative_beta) * d2_coefficient
    return d0_term - d1_term + d2_term
