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
# ``include/sht/sht_xcorr.hpp`` unless stated otherwise:
# - ``Correlator<Real>::Correlator()`` and its size members (lines
#   372-383 and 155-168) and ``Correlator<Real>::Constants`` (lines
#   361-370), as :class:`SphericalCrossCorrelator`
# - ``Correlator<Real>::compute()`` (lines 658-858), as
#   :func:`_xcorr_spectrum` and :meth:`SphericalCrossCorrelator.
#   compute`
# - ``detail::conjMult()`` (lines 1234-1243), inlined in
#   :func:`_xcorr_spectrum`
# - ``fft::SepRealFFT3D<double>::inverse()``
#   (``include/util/fft.hpp``, lines 671-678, with ``howmany = vH``
#   of line 630), re-expressed with :mod:`scipy.fft` in
#   :func:`_inverse_fft`
# - ``Correlator<Real>::findPeak()`` (lines 862-876), as
#   :func:`_find_peak`, and the fused normalize/argmax pass of
#   ``NormalizedCorrelator<Real>::correlate()`` (lines 1146-1154),
#   as :func:`_scale_and_find_peak`
# - ``Correlator<Real>::extractNeighborhood<1>()`` (lines 505-544)
#   and ``detail::extractInds()`` (lines 1249-1255), as
#   :func:`_extract_neighborhood`
# - ``detail::interpolateMaxima()`` (lines 1261-1366), as
#   :func:`_interpolate_maxima`
# - ``Correlator<Real>::interpPeak()`` (lines 406-432), as
#   :meth:`SphericalCrossCorrelator.interp_peak`
# - ``Correlator<Real>::correlate()`` (lines 394-400), as
#   :meth:`SphericalCrossCorrelator.correlate`
# - ``Correlator<Real>::indexEuler()`` (lines 580-590) and
#   ``Correlator<Real>::eulerIndex()`` (lines 549-575), as
#   :func:`index_to_euler` and :func:`euler_to_index`
# - ``NormalizedCorrelator<Real>`` (lines 237-270), its
#   ``Constants`` (lines 1182-1204) and its ``correlate()`` (lines
#   1140-1159), as :class:`NormalizedSphericalCrossCorrelator`
#
# The following are deliberately **not** ported here:
# - ``Correlator<Real>::refinePeak()`` (lines 442-499) and
#   ``Correlator<Real>::derivatives()`` (lines 889-1119), the real
#   space Newton refinement, and
#   ``NormalizedCorrelator<Real>::Constants::denominator()`` (lines
#   1211-1225), which it needs.  ``refine=True`` raises
#   ``NotImplementedError`` until they arrive
# - ``Correlator<Real>::extractBunge()`` (lines 594-649), which has
#   no consumer in kikuchipy yet and whose ZYZ to Bunge offsets are
#   the reversed ``zyz2eu()`` ones
#
# The wrap of ``beta`` and the reduction of ``alpha`` and ``gamma``
# which :func:`euler_to_index` applies before the C++ formulas are a
# deliberate addition to ``eulerIndex()``, which has no wrap and no
# caller in EMSphInx.

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

"""SO(3) cross-correlation of two spherical functions from their
harmonic coefficients, and its peak.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

**What is computed.** For the harmonic coefficients ``flm`` and
``gln`` of two real functions ``f`` and ``g`` on the unit sphere,
:meth:`SphericalCrossCorrelator.compute` fills the cross-correlation
on a uniform grid of passive ZYZ Euler angles,

.. code-block::

    xc[k, n, m] = sum_l sum_{m'=-l}^(l) sum_{n'=-l}^(l)
                      f^l_{m'} conj(g^l_{n'})
                      D^l_{n',m'}(zyz(k, n, m))

with :func:`kikuchipy.indexing._spherical._wigner.wigner_D` and the
negative orders ``a^l_(-m) = (-1)^m conj(a^l_m)``.  Equivalently, and
this is the interpretation,

.. code-block::

    xc[k, n, m] = <rotate_harmonics(flm, zyz(k, n, m)), gln>

with the real inner product ``sum_l [Re(a^l_0 conj(b^l_0))
+ 2 sum_(m>0) Re(a^l_m conj(b^l_m))]``, i.e. ``4 pi`` times the mean
of the product of the two real functions.  The correlator therefore
returns the ``zyz`` for which ``gln == rotate_harmonics(flm, zyz)``,
and ``correlate(flm, rotate_harmonics(flm, zyz0))`` peaks at ``zyz0``
with the peak value equal to the total power ``<f, f>``.  An on-grid
``zyz0`` is recovered exactly, since the autocorrelation identity
``C(Q) = C(Q^-1)`` makes the two axis neighbours of the peak cell
equal in each direction and the Newton step from ``x = 0`` vanishes.

The boundary orientation of a result is
``_euler.rotation_from_zyz(zyz)``, i.e. sample to crystal, which is
**provisional until Phase 5** as Phase 3 recorded.

**Grid and layouts** (frozen).  With ``bw`` the bandwidth,

.. code-block::

    sl  = 2 bw - 1                    side_length_unpadded
    slP = _fft.fast_size(sl)          side_length
    bwP = slP // 2 + 1                half_side_length

and never :func:`scipy.fft.next_fast_len`, which disagrees for most
recommended bandwidths (``next_fast_len(105, real=True) == 108`` but
``fast_size(105) == 105``).  The Euler grid is

.. code-block::

    alpha = 2 pi m / slP - pi / 2
    beta  = 2 pi k / slP - pi
    gamma = 2 pi n / slP - pi / 2

``fxc`` is ``(slP, slP, bwP)`` 128-bit complex indexed ``[k, n, m]``,
half-complex along the last (alpha) axis, and ``xc`` is
``(bwP, slP, slP)`` 64-bit float indexed ``[k, n, m]``, i.e. beta
slowest, gamma middle and alpha fastest, holding the stored half
``beta in [-pi, 0]`` only.  Both are the flat C++ orders, so the flat
argmax index is the C++ ``findPeak()`` index.  The other half of the
cube follows from the glide ``xc(a + pi, -b, g + pi) = xc(a, b, g)``.

**Two reproduced C++ defects** are behind the single keyword
``emsphinx_compatible``, ``True`` by default:

- ``extractNeighborhood<1>`` (lines 505-544) shifts alpha and gamma
  **per slot index** rather than per glided plane (lines 530-531),
  and its even-``slP`` shift ``x < bwP ? x + bwP - 1 : x - bwP`` is
  ``+slP/2`` going up but ``-slP/2 - 1`` coming down, so at
  ``x = bwP - 1`` it produces ``slP``, one past the axis: such a slot
  reads the next row's or the next beta slice's first element.  At
  ``k0 = bwP - 1`` a few of those offsets fall past the end of
  ``xc``; they are undefined behaviour in the C++ and are clamped to
  the last element here.  ``emsphinx_compatible=False`` uses the
  per-plane glide instead, which is exact for even ``slP`` and a half
  cell approximation for odd ``slP`` (no on-grid glide exists there).
- ``interpPeak`` (line 421) bounds the interpolated step with
  ``max(|x[0]|, max(|x[1]|, |x[0]|))``, so an alpha over-step is
  never caught.  ``emsphinx_compatible=False`` uses
  ``max(|x[0]|, |x[1]|, |x[2]|)``.

Consequently the returned ``alpha`` is unbounded in principle with
the default ``True``, while ``beta`` and ``gamma`` stay in
``[-pi - cell, cell]`` and ``[-pi/2 - cell, 3 pi/2]`` with
``cell = 2 pi / slP``.

A **third** C++ defect is reproduced unconditionally, since it moves
no orientation: the ``vPeak`` expression of ``interpolateMaxima()``
(lines 1354-1364) evaluates its three mixed cubic coefficients at
the wrong monomials, so the returned ``score`` is the fitted
tri-quadratic only up to those terms, see :func:`_interpolate_maxima`.

**Scores.** :meth:`SphericalCrossCorrelator.correlate` returns the
interpolated peak of the raw sum above, i.e. an **un-normalised**
metric equal to the total power for a perfect match, which scales
with both spectra and is comparable only within one geometry.
:meth:`NormalizedSphericalCrossCorrelator.correlate` returns
``xc * rDen`` at the peak, the Huhle semi-normalised metric of the
C++, which is **not** divided by the standard deviation of the
pattern function and is therefore not a bounded normalised
cross-correlation either.

**Accuracy.** The coarse (interpolated, unrefined) result is within
half a grid cell (``180 / slP`` degrees) of the true rotation, except
within one cell of ``beta = 0`` or ``beta = pi`` and for masked
patterns, where it is within one cell.  Phase 7's Newton refinement
supersedes this.

**Memory and threads.** A correlator owns ``fxc``
(``16 slP^2 bwP`` bytes), the ``pi/2`` Wigner table (``8 bw^3``) and
the two small scratch buffers, and rebinds ``xc`` (``8 slP^2 bwP``)
to a fresh array on every :meth:`SphericalCrossCorrelator.compute`.
Measured with :mod:`tracemalloc` at ``bw`` 68: a recurring
``correlate`` peaks at 62 MB (72 MB normalised), of which the table
(2.5 MB), ``r_den`` (9.9 MB) and the spectra are shared by
:meth:`SphericalCrossCorrelator.clone`, so each additional thread
costs ``2 (fxc + xc)`` = 59 MB.  A correlator is **not thread-safe**:
give every thread its own ``clone()``.

**Speed.** Single thread on the development machine, ``correlate``
end to end: 12 / 26 / 100 / 242 ms at ``bw`` 53 / 68 / 88 / 113
without symmetry and 7 / 14-20 / 44 / 93 ms with ``m-3m``, i.e.
1.4-2.7 times the compiled C++.  pocketfft has no radix-13 codelet,
so an ``slP`` with a factor 13 costs 1.06-1.18 times as much in the
inverse transform (``slP`` 117 against 120), about 4-11 % of a
``correlate`` at ``bw`` 57-59; this is recorded rather than warned
about, since ``bw`` 59 is a recommended
:func:`kikuchipy.indexing._spherical._fft.fast_bandwidths` value.

References
----------
:cite:`lenthe2019spherical`, :cite:`huhle2009normalized`
"""

import math

import numpy as np

# TODO: The implementer adds the two remaining import blocks, which
# are omitted here because a body which only raises uses none of
# them:
#     from numba import njit
#     from scipy.fft import ifft, irfft
#
#     from kikuchipy.indexing._spherical import _euler, _fft, _wigner
#
# The two SciPy transforms must be bound in this namespace, as
# Phases 1-3 bind theirs, so that the FFT call recording test can
# patch ``_xcorr.ifft`` and ``_xcorr.irfft``.

# Machine epsilon of the 64-bit floating point type
_EPS = float(np.finfo(np.float64).eps)

# Convergence criterion and iteration cap of the Newton loop of
# ``detail::interpolateMaxima()``
# (``include/sht/sht_xcorr.hpp``, lines 1338 and 1348)
_NEWTON_EPS = math.sqrt(_EPS)
_NEWTON_MAX_ITERATIONS = 25

# Half and full width of the extracted neighbourhood, i.e. the C++
# ``extractNeighborhood<N>`` with ``N = 1`` (lines 505-544)
_NEIGHBORHOOD_HALF_WIDTH = 1
_NEIGHBORHOOD_WIDTH = 2 * _NEIGHBORHOOD_HALF_WIDTH + 1

# sqrt(4 pi), the factor of ``s2m`` in
# ``NormalizedCorrelator<Real>::Constants`` (line 1197).  With the
# normalization of :mod:`kikuchipy.indexing._spherical._sht` this
# makes ``s2m`` the solid angle of a binary window
_SQRT_FOUR_PI = math.sqrt(4 * math.pi)

# Full turn
_TWO_PI = 2 * math.pi

# --------------------------- Numba kernels -------------------------- #


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
def _xcorr_spectrum(
    flm: np.ndarray,
    gln: np.ndarray,
    table: np.ndarray,
    n_fold: int,
    mirror: bool,
    fxc: np.ndarray,
    fm: np.ndarray,
    gn: np.ndarray,
) -> None:
    """Fill the half-complex spectrum of the cross-correlation.

    Parameters
    ----------
    flm, gln
        Harmonic coefficients ``a[m, l]`` of the two real functions,
        both C-contiguous ``(bw, bw)`` and 128-bit complex.  The
        rows ``m % n_fold != 0`` of ``flm`` and, when ``mirror``,
        its entries with ``l + m`` odd are **assumed** zero and are
        skipped, as in EMSphInx.
    table
        Transposed ``pi/2`` Wigner d table
        ``table[m, k, j] = d^j_{k,m}(pi/2)`` of shape
        ``(bw, bw, bw)``, i.e.
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_half_pi_table` with ``transpose=True``.  Its NaN
        slots ``j < max(k, m)`` are never read.
    n_fold
        Order of the rotational symmetry of ``flm`` about z, at
        least one.
    mirror
        Whether ``flm`` has an equatorial mirror plane.
    fxc
        ``(slP, slP, bwP)`` 128-bit complex output buffer indexed
        ``[k, n, m]``, written in place.  Every slot the kernel does
        not write is a column ``m in [bw, bwP)`` of a row ``n < bw``,
        which nothing ever writes, so a zero filled buffer may be
        reused across calls with different ``n_fold`` and ``mirror``.
    fm
        ``(bw, bw)`` 128-bit complex scratch buffer.
    gn
        ``(bw,)`` 128-bit complex scratch buffer.

    Notes
    -----
    Faithful transcription of the symmetry reduced loop of
    ``Correlator<Real>::compute()``
    (``include/sht/sht_xcorr.hpp``, lines 699-854), with
    ``detail::conjMult()`` (lines 1234-1243) inlined.  The pattern
    side never carries symmetry (``glnFold = 1``, ``gMir = false``),
    so the systemic-zero column mask of lines 709-726 reduces to
    ``m % n_fold == 0`` and the ``nonZero0``/``nonZero1`` parity
    split is dead.

    Each ``(k, n, m)`` writes up to four slots (lines 796-817):
    ``fxc[k, n, m] = v``, ``fxc[slP-k, slP-n, m] = vnc``,
    ``fxc[slP-k, n, m] = +-v`` and ``fxc[k, slP-n, m] = +-vnc``, the
    sign being positive when ``(m + n)`` is even.  Systemic zeros
    (lines 818-829), zero rows (lines 831-837) and the zero pad
    slices ``k in [bw, slP - bw]`` (line 854, inclusive at both
    ends) are written, not assumed.

    The negation of ``vnc`` at the end of the mirror branch (line
    783) is dead code: line 774 has just made ``start + m`` even, so
    ``toggle`` of line 776 is always true.  It is transcribed for
    fidelity and excluded from coverage.

    Complex arithmetic only, so the compiled kernel and its
    ``py_func`` agree bitwise on the development machine.
    """
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
def _find_peak(xc: np.ndarray) -> int:
    """Return the flat index of the largest cross-correlation.

    Parameters
    ----------
    xc
        Cross-correlation cube of shape ``(bwP, slP, slP)``, read
        through its flat view.

    Returns
    -------
    index
        Flat index of the first strict maximum in C order.

    Notes
    -----
    Port of ``Correlator<Real>::findPeak()``
    (``include/sht/sht_xcorr.hpp``, lines 862-876).  The comparison
    is a strict ``>`` seeded with ``xc.flat[0]``, so among tied
    maxima the **smallest** flat index wins and NaN slots are
    skipped, neither of which :func:`numpy.argmax` does.
    """
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
def _scale_and_find_peak(xc: np.ndarray, r_den: np.ndarray) -> int:
    """Scale the cross-correlation in place and return the flat
    index of its largest value.

    Parameters
    ----------
    xc
        Cross-correlation cube of shape ``(bwP, slP, slP)``,
        multiplied by ``r_den`` element by element in place.
    r_den
        Reciprocal Huhle denominator of the same shape.

    Returns
    -------
    index
        Flat index of the first strict maximum of the scaled cube in
        C order.

    Notes
    -----
    Port of the fused normalize and argmax pass of
    ``NormalizedCorrelator<Real>::correlate()``
    (``include/sht/sht_xcorr.hpp``, lines 1146-1154), which needs no
    temporary.  The tie and NaN semantics are those of
    :func:`_find_peak`.
    """
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True). It is left undecorated here because
# Numba cannot compile a body which only raises.
def _extract_neighborhood(
    xc_flat: np.ndarray,
    slp: int,
    bwp: int,
    k0: int,
    n0: int,
    m0: int,
    emsphinx_compatible: bool,
    nh: np.ndarray,
) -> None:
    """Extract the 3 x 3 x 3 neighbourhood around a grid point.

    Parameters
    ----------
    xc_flat
        Flat 64-bit float view of the ``(bwP, slP, slP)``
        cross-correlation cube, i.e. ``xc.reshape(-1)``.  The flat
        view is required because the faithful glide of the C++ reads
        one past a row or one past a beta slice.
    slp
        Padded side length ``slP``.
    bwp
        Half side length ``bwP = slP // 2 + 1``.
    k0, n0, m0
        Centre of the neighbourhood, ``0 <= k0 < bwP`` and
        ``0 <= n0, m0 < slP``.
    emsphinx_compatible
        Whether to reproduce the two C++ glide defects.  When
        ``False`` the per-plane glide is used instead, which is
        exact for even ``slP`` and a half cell approximation for odd
        ``slP``.
    nh
        ``(3, 3, 3)`` 64-bit float output buffer, written in place.

    Notes
    -----
    Port of ``Correlator<Real>::extractNeighborhood<1>()``
    (``include/sht/sht_xcorr.hpp``, lines 505-544) with
    ``detail::extractInds()`` (lines 1249-1255).  The index arrays
    wrap periodically in all three directions, and the slots whose
    ``k`` lands in the unstored half ``k >= bwP`` are brought back
    by the glide plane (lines 527-533), which is reachable only when
    ``k0`` is ``0`` or ``bwP - 1``.

    With ``emsphinx_compatible=True`` the glide is the C++ one, with
    both of its defects: the alpha shift (line 530) and the gamma
    shift (line 531) are applied to the slot **index** ``i`` and so
    hit that slot of all three planes, and for even ``slP`` the
    shift is ``+slP/2`` for ``x < bwP`` but ``-slP/2 - 1`` for
    ``x >= bwP``, which at ``x = bwP - 1`` gives ``slP``, one past
    the axis.  Such a slot reads the first element of the next row
    (an alpha slot) or of the next beta slice (a gamma slot).

    **Every flat offset is clamped to ``xc_flat.size - 1``**, since
    at ``k0 = bwP - 1`` and even ``slP`` the shifted gamma slot
    combined with the centre ``k`` slot reaches past the end of the
    cube.  The reachable set is exactly ``(bwP - 1, bwP - 2, m0)``
    for every ``m0`` (three slots each) and
    ``(bwP - 1, n0, bwP - 2)`` for
    ``n0 in {0, bwP - 3, slP - 1}`` (one slot each), i.e.
    ``slP + 3`` centres per even ``slP`` and none for odd ``slP``.
    Those reads are undefined behaviour in the C++, so the clamped
    values are never compared with it.

    With ``emsphinx_compatible=False`` every slot of a glided plane
    reads ``xc[slP - k, (n + s) % slP, (m + s) % slP]`` with
    ``s = slP // 2``.  For even ``slP`` this is the exact on-grid
    glide; for odd ``slP`` no on-grid glide exists and the shift is
    a half cell approximation.
    """
    raise NotImplementedError


# TODO: The implementer decorates this kernel with
# @njit(cache=True, nogil=True, error_model="numpy"). It is left
# undecorated here because Numba cannot compile a body which only
# raises.
def _interpolate_maxima(p: np.ndarray, x: np.ndarray) -> float:
    """Return the value of the tri-quadratic fitted to a
    neighbourhood at its maximum, and write the maximum's position.

    Parameters
    ----------
    p
        ``(3, 3, 3)`` 64-bit float neighbourhood indexed
        ``[k, n, m] = [z, y, x]``.
    x
        ``(3,)`` 64-bit float buffer which receives the sub-pixel
        position ``(dk, dn, dm)`` of the maximum relative to the
        centre of ``p``, written in place.

    Returns
    -------
    peak
        Value of the C++ ``vPeak`` expression at ``x``, which is
        the fitted tri-quadratic except for the three terms of the
        defect below.

    Notes
    -----
    Port of ``detail::interpolateMaxima()``
    (``include/sht/sht_xcorr.hpp``, lines 1261-1366): the 27
    coefficients ``a_{kji}`` of ``f(x, y, z) = a_{kji} x^i y^j z^k``,
    then at most 25 Newton steps with the analytic 3 x 3 Hessian and
    a convergence criterion of ``sqrt(eps)`` on the largest step
    component.  When the loop runs out of iterations, ``x`` is reset
    to zero (line 1350) and the fit is evaluated there.

    **A third reproduced C++ defect**: the returned value is the
    literal ``vPeak`` expression of lines 1354-1364, in which the
    three mixed cubic terms are evaluated at the wrong monomials --
    ``a112 * xy * x[2]``, ``a211 * yz * x[0]`` and
    ``a121 * zx * x[1]`` all reduce to ``z y x`` instead of
    ``z y x^2``, ``z^2 y x`` and ``z y^2 x`` (the C++ names are
    ``x = (z, y, x)``, so ``xx = z^2``, ``zz = x^2``, ``xy = z y``
    and ``yz = y x``).  The Hessian (lines 1318-1323) and the
    gradient (lines 1335-1337) use the correct monomials, so ``x``
    is the stationary point of the full fit and only the returned
    value is affected: measured 1.2e-7 to 6.1e-7 low on random
    27-coefficient blocks with unit curvature and a peak of 5, and
    exactly zero whenever the three coefficients vanish, as they do
    for a separable quadratic.

    The kernel is compiled with ``error_model="numpy"``, the one
    deviation from the project's default Numba flags.  The C++
    divides the six cofactors by the Hessian determinant without a
    guard (lines 1319-1325) and relies on IEEE semantics: on a flat
    neighbourhood, where the determinant is zero, the steps are NaN,
    the criterion never passes and the reset returns the centre
    value.  Numba's default error model raises ``ZeroDivisionError``
    there instead, and flat neighbourhoods are reachable in
    production (a masked pattern with an empty region, an all-zero
    ``gln``, a rotation invariant spectrum).
    """
    raise NotImplementedError


# ------------------------- Inverse transform ------------------------ #


def _inverse_fft(fxc: np.ndarray, n_fold: int) -> np.ndarray:
    """Return the stored half of the real cross-correlation cube.

    Parameters
    ----------
    fxc
        ``(slP, slP, bwP)`` 128-bit complex half-complex spectrum
        indexed ``[k, n, m]``, as filled by
        :func:`_xcorr_spectrum`.  It is not written to.
    n_fold
        Order of the rotational symmetry of the first function about
        z.  Only the alpha planes ``m % n_fold == 0`` are
        transformed, the others being the systemic zeros the kernel
        wrote, which makes the skipping exact.

    Returns
    -------
    xc
        Fresh ``(bwP, slP, slP)`` 64-bit float C-contiguous array
        indexed ``[k, n, m]``.

    Notes
    -----
    Re-expression of ``fft::SepRealFFT3D<double>::inverse()``
    (``include/util/fft.hpp``, lines 671-678) with :mod:`scipy.fft`:
    an unnormalised backward transform along ``k`` for every ``n``,
    the same along ``n`` for the first ``bwP`` values of ``k`` only
    (``howmany = vH``, line 630), and a half-complex to real
    transform along ``m``.  ``norm="forward"`` makes both complex
    transforms unnormalised, i.e. FFTW's convention, and
    ``workers=1`` is the project rule.

    Equal to
    ``scipy.fft.irfftn(fxc, s=(slP,) * 3, axes=(0, 1, 2),
    norm="forward")[:bwP]``, which is 1.4 to 2.6 times slower and
    materialises the whole ``slP^3`` cube, so it stays a test
    oracle.  The separable path allocates one ``fxc`` sized
    transient, recorded as an optimisation target.
    """
    raise NotImplementedError


# ------------------------- Euler grid <-> index --------------------- #


def index_to_euler(knm: tuple[int, int, int], side_length: int) -> np.ndarray:
    """Return the ZYZ Euler angles of a grid point.

    Parameters
    ----------
    knm
        Grid indices ``(k, n, m)``, i.e. beta, gamma and alpha.
    side_length
        Padded side length ``slP`` of the Euler cube.

    Returns
    -------
    zyz
        Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in radians
        in a new ``(3,)`` 64-bit float array, with
        ``alpha = 2 pi m / slP - pi / 2``,
        ``beta = 2 pi k / slP - pi`` and
        ``gamma = 2 pi n / slP - pi / 2``.

    Notes
    -----
    Port of ``Correlator<Real>::indexEuler()``
    (``include/sht/sht_xcorr.hpp``, lines 580-590).  The stored
    ``k in [0, bwP)`` covers ``beta in [-pi, 0]``: for odd ``slP``
    only up to ``-pi / slP``, so ``beta = 0`` falls between the last
    stored slice and its glide image, while for even ``slP`` it is
    slice ``bwP - 1``.
    """
    raise NotImplementedError


def euler_to_index(zyz: np.ndarray, side_length: int) -> tuple[int, int, int]:
    """Return the grid point closest to a rotation.

    Parameters
    ----------
    zyz
        Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in radians
        in an array-like of shape ``(3,)``.  They need not be
        wrapped.
    side_length
        Padded side length ``slP`` of the Euler cube.

    Returns
    -------
    knm
        Grid indices ``(k, n, m)`` with ``0 <= k < bwP`` and
        ``0 <= n, m < slP``.

    Raises
    ------
    ValueError
        If ``zyz`` does not have three elements.

    Notes
    -----
    Port of ``Correlator<Real>::eulerIndex()``
    (``include/sht/sht_xcorr.hpp``, lines 549-575), with three
    documented additions.  First, ``beta`` is wrapped into
    ``[-pi, pi]`` with
    :func:`kikuchipy.indexing._spherical._euler.wrap_beta` and
    ``alpha`` and ``gamma`` are reduced into the grid's own
    ``[-pi/2, 3 pi/2)``, which the C++ does not do and which its
    unsigned arithmetic cannot survive: an unwrapped
    ``beta = pi + 0.1`` at ``slP`` 135 gives ``k = -2``.  Second,
    the rounding is ``floor(x + 0.5)``, i.e. C++ ``std::round()``,
    and not Python's banker's rounding.  Third, ``k`` is clamped to
    ``bwP - 1``: for odd ``slP`` and ``beta`` a multiple of
    ``2 pi``, the fractional ``k`` is exactly ``slP / 2``, which the
    strict ``>`` of the glide leaves alone and which rounds to
    ``bwP``, one slice outside the stored half.

    ``eulerIndex()`` has no caller in EMSphInx.
    """
    raise NotImplementedError


# --------------------------- Correlators ---------------------------- #


class SphericalCrossCorrelator:
    """Cross-correlation of two spherical functions over SO(3).

    Parameters
    ----------
    bandwidth
        Bandwidth, i.e. the exclusive maximum harmonic degree, at
        least one.
    wigner_d_half_pi
        Transposed ``pi/2`` Wigner d table
        ``table[m, k, j] = d^j_{k,m}(pi/2)`` of shape
        ``(bw, bw, bw)`` to share with another correlator, as
        :meth:`clone` does.  If not given, one is built with
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_half_pi_table`.

    Attributes
    ----------
    bandwidth : int
        Exclusive maximum harmonic degree.
    side_length_unpadded : int
        ``2 bw - 1``, the side length before zero padding.
    side_length : int
        Zero padded side length ``slP`` of the Euler cube, the
        smallest fast FFT size not smaller than
        :attr:`side_length_unpadded`.
    half_side_length : int
        ``bwP = slP // 2 + 1``, the number of stored beta slices.
    wigner_d_half_pi : numpy.ndarray
        The transposed ``pi/2`` Wigner d table, read only and
        shareable between instances.
    fxc : numpy.ndarray
        ``(slP, slP, bwP)`` 128-bit complex spectrum buffer, owned
        by this instance and reused by every :meth:`compute`.
    xc : numpy.ndarray or None
        The cross-correlation cube of the last :meth:`compute`, a
        fresh ``(bwP, slP, slP)`` 64-bit float array per call, and
        ``None`` before the first one.

    Raises
    ------
    ValueError
        If ``bandwidth`` is smaller than one, or if
        ``wigner_d_half_pi`` is not a C-contiguous 64-bit float
        array of shape ``(bw, bw, bw)`` whose undefined slots are
        NaN and whose slot ``[1, 0, 1]`` is negative, i.e. the
        transposed and not the untransposed layout.

    Notes
    -----
    Port of ``Correlator<Real>`` (``include/sht/sht_xcorr.hpp``,
    lines 155-168, 361-383, 394-432, 505-544, 549-590, 658-876).
    See the module documentation for what is computed, the array
    layouts, the two ``emsphinx_compatible`` quirks, the score
    semantics and the measured memory and speed.

    An instance is **not thread-safe**, since :meth:`compute`
    overwrites :attr:`fxc` and the two scratch buffers of
    :func:`_xcorr_spectrum`, ``_fm`` of shape ``(bw, bw)`` and
    ``_gn`` of shape ``(bw,)``, both 128-bit complex.  Use one
    :meth:`clone` per thread; clones share the Wigner table and
    allocate the rest.

    :meth:`compute` returns a fresh caller-owned array on every call
    and rebinds :attr:`xc` to it, so a result kept by the caller
    stays valid across later calls.  The C++ writes into a buffer
    the caller passes instead; an ``out=`` parameter is not offered
    because :mod:`scipy.fft` has no ``out=`` and it would only add a
    copy.

    ``refine=True`` raises ``NotImplementedError`` until Phase 7
    (``spherical-refinement``) ports ``refinePeak()`` and
    ``derivatives()``.

    Examples
    --------
    An on-grid rotation is recovered exactly, with a peak equal to
    the total power of the spectrum:

    >>> import numpy as np
    >>> from kikuchipy.indexing._spherical._wigner import (
    ...     rotate_harmonics,
    ... )
    >>> from kikuchipy.indexing._spherical._xcorr import (
    ...     SphericalCrossCorrelator,
    ...     euler_to_index,
    ...     index_to_euler,
    ... )
    >>> correlator = SphericalCrossCorrelator(8)
    >>> correlator
    SphericalCrossCorrelator: bw = 8, side_length = 15
    (unpadded 15), half 8
    >>> rng = np.random.default_rng(0)
    >>> flm = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    >>> flm = np.triu(flm)
    >>> flm[0] = flm[0].real
    >>> zyz0 = index_to_euler((3, 5, 7), correlator.side_length)
    >>> gln = rotate_harmonics(flm, zyz0)
    >>> zyz, score = correlator.correlate(flm, gln, 1, False)
    >>> euler_to_index(zyz, correlator.side_length)
    (3, 5, 7)
    >>> power = float(
    ...     (flm[0] * flm[0].conjugate()).real.sum()
    ...     + 2 * (flm[1:] * flm[1:].conjugate()).real.sum()
    ... )
    >>> round(score / power, 6)
    1.0
    """

    def __init__(
        self, bandwidth: int, *, wigner_d_half_pi: np.ndarray | None = None
    ) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the bandwidth and the three side
        lengths, e.g. ``"SphericalCrossCorrelator: bw = 68,
        side_length = 135 (unpadded 135), half 68"``.
        """
        raise NotImplementedError

    def compute(
        self,
        flm: np.ndarray,
        gln: np.ndarray,
        n_fold: int,
        mirror: bool,
    ) -> np.ndarray:
        """Return the cross-correlation of two spherical functions
        on the Euler grid.

        Parameters
        ----------
        flm, gln
            Harmonic coefficients ``a[m, l]`` of the two real
            functions in array-likes of shape ``(bw, bw)``, cast to
            128-bit complex and made C-contiguous.  The rows
            ``m % n_fold != 0`` of ``flm`` and, when ``mirror``, its
            entries with ``l + m`` odd are **assumed** zero and are
            skipped, as in EMSphInx.
        n_fold
            Order of the rotational symmetry of ``flm`` about z, at
            least one.  Only ``flm`` may carry symmetry: the pattern
            side never does.
        mirror
            Whether ``flm`` has an equatorial mirror plane.

        Returns
        -------
        xc
            Fresh caller-owned ``(bwP, slP, slP)`` 64-bit float
            array of the stored half ``beta in [-pi, 0]``, to which
            :attr:`xc` is rebound.

        Raises
        ------
        ValueError
            If ``flm`` or ``gln`` does not have shape
            ``(bw, bw)``, if ``n_fold`` is a :class:`bool` or
            smaller than one, or if ``mirror`` is not a
            :class:`bool`.  The argument order follows
            :func:`kikuchipy.indexing._spherical._symmetry.
            point_group_flags` while the C++ ``compute()`` takes the
            two the other way round, so a swapped call fails loudly.

        Notes
        -----
        Port of ``Correlator<Real>::compute()``
        (``include/sht/sht_xcorr.hpp``, lines 658-858), i.e.
        :func:`_xcorr_spectrum` followed by :func:`_inverse_fft`.
        """
        raise NotImplementedError

    def interp_peak(
        self, index: int, emsphinx_compatible: bool = True
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return the sub-pixel maximum of the last
        cross-correlation near a grid point.

        Parameters
        ----------
        index
            Flat index into :attr:`xc`, which should be at or near a
            local maximum, e.g. the return of :func:`_find_peak`.
        emsphinx_compatible
            Whether to reproduce the two C++ defects, ``True`` by
            default: the neighbourhood glide of
            :func:`_extract_neighborhood` and the bounds check of
            ``interpPeak()``, which tests ``|x[0]|`` twice and
            ``|x[2]|`` never, so that an alpha over-step is not
            caught.

        Returns
        -------
        zyz
            Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in
            radians in a new ``(3,)`` 64-bit float array.
        peak
            Value of the fitted tri-quadratic at the maximum, or the
            value at the centre of the neighbourhood when the step
            was rejected.
        x
            Sub-pixel offset ``(dk, dn, dm)`` of the maximum from
            the centre of the neighbourhood, in cells, in a new
            ``(3,)`` 64-bit float array.  Exactly zero when the step
            was rejected.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called yet.

        Notes
        -----
        Port of ``Correlator<Real>::interpPeak()``
        (``include/sht/sht_xcorr.hpp``, lines 406-432).  A step
        larger than one cell is rejected and replaced by the centre,
        but only ``x[0]`` and ``x[1]`` enter that check with
        ``emsphinx_compatible=True`` (line 421).
        """
        raise NotImplementedError

    def correlate(
        self,
        flm: np.ndarray,
        gln: np.ndarray,
        n_fold: int,
        mirror: bool,
        *,
        refine: bool = False,
        emsphinx_compatible: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return the rotation of the largest cross-correlation
        between two spherical functions, and its value.

        Parameters
        ----------
        flm, gln
            Harmonic coefficients ``a[m, l]`` of the two real
            functions, see :meth:`compute`.
        n_fold
            Order of the rotational symmetry of ``flm`` about z.
        mirror
            Whether ``flm`` has an equatorial mirror plane.
        refine
            Whether to refine the interpolated peak in real space,
            ``False`` by default.
        emsphinx_compatible
            Whether to reproduce the two C++ defects, ``True`` by
            default, see :meth:`interp_peak`.

        Returns
        -------
        zyz
            Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in
            radians of the peak, in a new ``(3,)`` 64-bit float
            array.  ``beta`` lies in ``[-pi - cell, cell]`` and
            ``gamma`` in ``[-pi/2 - cell, 3 pi/2]`` with
            ``cell = 2 pi / slP``; ``alpha`` lies in the same
            interval as ``gamma`` when ``emsphinx_compatible`` is
            ``False`` and is only finite when it is ``True``.
        score
            Interpolated peak of the un-normalised cross-correlation
            of the module documentation, equal to the total power
            ``<f, f>`` for a perfect match.  It scales with both
            spectra and is comparable only within one geometry.

        Raises
        ------
        NotImplementedError
            If ``refine`` is ``True``.  The real space refinement
            arrives in Phase 7 (``spherical-refinement``).
        ValueError
            See :meth:`compute`.

        Notes
        -----
        Port of ``Correlator<Real>::correlate()``
        (``include/sht/sht_xcorr.hpp``, lines 394-400), i.e.
        :meth:`compute`, :func:`_find_peak` and
        :meth:`interp_peak`.
        """
        raise NotImplementedError

    def clone(self) -> "SphericalCrossCorrelator":
        """Return a new correlator sharing this one's Wigner table.

        Returns
        -------
        correlator
            New instance with the same bandwidth, the **same**
            :attr:`wigner_d_half_pi` and its own :attr:`fxc` and
            ``_fm``/``_gn`` scratch buffers, ready for use in
            another thread.

        Notes
        -----
        Port of ``Correlator<Real>::clone()``, which shares the
        read-only ``xcLut`` and copies the rest.
        """
        raise NotImplementedError


class NormalizedSphericalCrossCorrelator:
    """Normalized cross-correlation of a reference function with a
    masked pattern over SO(3).

    Parameters
    ----------
    bandwidth
        Bandwidth, i.e. the exclusive maximum harmonic degree, at
        least one.
    flm
        Harmonic coefficients ``a[m, l]`` of the reference function
        in an array-like of shape ``(bw, bw)``.
    flm2
        Harmonic coefficients of the **square** of the reference
        function, of the same shape.
    n_fold
        Order of the rotational symmetry of ``flm`` about z, at
        least one.
    mirror
        Whether ``flm`` has an equatorial mirror plane.
    mlm
        Harmonic coefficients of the window function, which is
        assumed to be a binary mask, of the same shape.
    wigner_d_half_pi
        Transposed ``pi/2`` Wigner d table to share, see
        :class:`SphericalCrossCorrelator`.

    Attributes
    ----------
    correlator : SphericalCrossCorrelator
        The un-normalised correlator this one owns.
    flm, flm2, mlm : numpy.ndarray
        Copies of the three spectra, all ``(bw, bw)`` 128-bit
        complex.
    n_fold : int
        Order of the rotational symmetry of ``flm`` about z.
    mirror : bool
        Whether ``flm`` has an equatorial mirror plane.
    r_den : numpy.ndarray
        ``(bwP, slP, slP)`` 64-bit float reciprocal of the Huhle
        denominator, computed once by the constructor and read only.

    Raises
    ------
    ValueError
        See :class:`SphericalCrossCorrelator`, plus any of the three
        spectra not having shape ``(bw, bw)``.

    Notes
    -----
    Port of ``NormalizedCorrelator<Real>``
    (``include/sht/sht_xcorr.hpp``, lines 237-270, 1140-1204)
    :cite:`huhle2009normalized`.  The constructor computes
    ``mrf = compute(flm, mlm)`` and ``mrf2 = compute(flm2, mlm)``
    with the **reference's** flags, as the C++ does (lines 1191 and
    1194) even though the mask has no symmetry, takes
    ``s2m = mlm[0, 0].real sqrt(4 pi)`` (line 1197, the solid angle
    of a binary window) and stores

    .. code-block::

        rDen = 1 / sqrt(mrf2 - 2 fWbar mrf + fWbar^2 s2m)
        fWbar = mrf / s2m

    which is Huhle equations 8 and 9 (line 1200), evaluated in place
    on the two cubes so that no third temporary is needed.

    The radicand is unguarded, as in the C++.  It is
    ``s2m Var_window(f) >= 0`` mathematically, so a non-positive
    value is a caller error: a **negative** radicand gives a NaN
    slot, which loses every comparison of :func:`_scale_and_find_peak`
    and is therefore harmless, while a radicand of exactly **zero**
    gives ``rDen = +inf``, and ``xc * inf`` is ``+inf`` for a
    positive ``xc``, which *wins* the argmax and yields a garbage
    peak.  A zero radicand means the reference is constant over the
    window.  A ``flm2`` which is not the transform of the square of
    the reference produces NaNs.

    The score is the C++ metric ``xc * rDen`` at the interpolated
    peak and is **not divided by the standard deviation of the
    pattern function** (the C++ says so at lines 255 and 1138, and
    its own ``Indexer::sum2`` is written and never read), so it is
    not a bounded normalized cross-correlation and is comparable
    only within a fixed geometry.  Measured for the shipped Ni
    master at ``bw`` 68: 3.71-3.93 with the
    ``emsphinx_compatible=True`` master normalization and 1.37-1.47
    with ``False``, a factor of about 2.6 from the master's DC term,
    while the argmax stays within the coarse tolerance in both.

    An instance is **not thread-safe**; use one :meth:`clone` per
    thread.
    """

    def __init__(
        self,
        bandwidth: int,
        flm: np.ndarray,
        flm2: np.ndarray,
        n_fold: int,
        mirror: bool,
        mlm: np.ndarray,
        *,
        wigner_d_half_pi: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the bandwidth, the three side
        lengths and the flags, e.g.
        ``"NormalizedSphericalCrossCorrelator: bw = 68,
        side_length = 135 (unpadded 135), half 68, n_fold = 4,
        mirror = True"``.
        """
        raise NotImplementedError

    @property
    def bandwidth(self) -> int:
        """Return the exclusive maximum harmonic degree."""
        raise NotImplementedError

    @property
    def side_length(self) -> int:
        """Return the zero padded side length ``slP`` of the Euler
        cube.
        """
        raise NotImplementedError

    @property
    def half_side_length(self) -> int:
        """Return the number of stored beta slices
        ``bwP = slP // 2 + 1``.
        """
        raise NotImplementedError

    def correlate(
        self,
        gln: np.ndarray,
        *,
        refine: bool = False,
        emsphinx_compatible: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return the rotation of the largest normalized
        cross-correlation between the reference and a pattern, and
        its value.

        Parameters
        ----------
        gln
            Harmonic coefficients ``a[m, l]`` of the masked pattern
            in an array-like of shape ``(bw, bw)``.
        refine
            Whether to refine the interpolated peak in real space,
            ``False`` by default.
        emsphinx_compatible
            Whether to reproduce the two C++ defects, ``True`` by
            default, see
            :meth:`SphericalCrossCorrelator.interp_peak`.

        Returns
        -------
        zyz
            Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in
            radians of the peak, in a new ``(3,)`` 64-bit float
            array, with the ranges of
            :meth:`SphericalCrossCorrelator.correlate`.
        score
            Interpolated peak of the **normalized**
            cross-correlation ``xc * rDen``, which still needs to be
            divided by the standard deviation of the pattern
            function, see the class ``Notes``.

        Raises
        ------
        NotImplementedError
            If ``refine`` is ``True``.  The real space refinement
            arrives in Phase 7 (``spherical-refinement``).
        ValueError
            If ``gln`` does not have shape ``(bw, bw)``.

        Notes
        -----
        Port of ``NormalizedCorrelator<Real>::correlate()``
        (``include/sht/sht_xcorr.hpp``, lines 1140-1159): one
        :meth:`SphericalCrossCorrelator.compute` with the
        reference's flags, then the fused in-place scaling and
        argmax of :func:`_scale_and_find_peak`, then
        :meth:`SphericalCrossCorrelator.interp_peak` on the
        **scaled** cube, as the C++ interpolates the normalized
        values (line 1157).
        """
        raise NotImplementedError

    def clone(self) -> "NormalizedSphericalCrossCorrelator":
        """Return a new correlator sharing this one's spectra,
        denominator and Wigner table.

        Returns
        -------
        correlator
            New instance sharing :attr:`flm`, :attr:`flm2`,
            :attr:`mlm`, :attr:`r_den` and the Wigner table, all
            read only, with its own spectrum and scratch buffers.

        Notes
        -----
        Port of ``NormalizedCorrelator<Real>::clone()``, which
        shares the read-only ``ncLut`` and ``xcLut``.
        """
        raise NotImplementedError
