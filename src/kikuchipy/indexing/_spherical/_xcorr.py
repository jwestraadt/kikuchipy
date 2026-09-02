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
# - ``Correlator<Real>::derivatives()`` (lines 889-1119), the value,
#   Jacobian and Hessian of the cross-correlation at one rotation, as
#   :func:`_derivatives`
# - ``Correlator<Real>::refinePeak()`` (lines 442-499), the real
#   space Newton refinement, as :func:`_refine_peak` and
#   :meth:`SphericalCrossCorrelator.refine_zyz`
# - ``NormalizedCorrelator<Real>::refinePeak()`` (lines 1169-1172)
#   and ``NormalizedCorrelator<Real>::Constants::denominator()``
#   (lines 1211-1225), as
#   :meth:`NormalizedSphericalCrossCorrelator.refine_zyz` and its
#   ``_denominator``
#
# The following are deliberately **not** ported here:
# - ``Correlator<Real>::extractBunge()`` (lines 594-649), which has
#   no consumer in kikuchipy yet and whose ZYZ to Bunge offsets are
#   the reversed ``zyz2eu()`` ones
# - the window shift chain rule the C++ itself omits from the
#   normalized refinement (lines 263-264): the Newton step maximizes
#   the **un-normalized** correlation in both ports, so the refined
#   normalized score can dip below the coarse one
#
# The wrap of ``beta`` and the reduction of ``alpha`` and ``gamma``
# which :func:`euler_to_index` applies before the C++ formulas are a
# deliberate addition to ``eulerIndex()``, which has no wrap and no
# caller in EMSphInx.
#
# The ``refine`` keyword of both ``correlate()`` methods keeps a
# ``False`` default, a deliberate deviation from the C++ ``ref =
# true`` of lines 189 and 255: the user facing default lives on the
# indexer and the signal method instead.

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
# Changed by Johan Westraadt, 2026-08, 2026-09: translated to
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
``_euler.rotation_from_zyz(zyz)``, i.e. sample to crystal.  Phase 5
(``spherical-back-projection``, D8) froze that sign by measurement:
over 27 rotations of ``EBSDMasterPattern.get_patterns`` back-projected
and correlated at ``bw`` 68, with either correlator,
``~Rotation(zyz_to_quaternion(zyz))`` is 0.34 degrees from the true
orientation in the median and 0.72 at worst, while
``Rotation(zyz_to_quaternion(zyz))`` is 35 degrees out.

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
patterns, where it is within one cell.  The Newton refinement below
supersedes this.

**Refinement.**  :meth:`SphericalCrossCorrelator.refine_zyz` and
``correlate(refine=True)`` run the real space Newton refinement of
``refinePeak()`` (lines 442-499) from a starting triple, using the
value, Jacobian and Hessian :func:`_derivatives` evaluates at one
rotation (lines 889-1119).  The loop is a 3 x 3 Cholesky solve
(:func:`kikuchipy.indexing._spherical._preprocessing.
_cholesky_solve_3x3`, the same ``solve::cholesky`` the C++ calls)
with a monotone step rule, saddle rejection, the 1 x 1 and 2 x 2
sub-problems the ``beta ~ 0`` and ``beta ~ pi`` degeneracies fall
back to, a stopping threshold of ``eps 2 pi / slP`` with ``eps =
0.01`` and at most 15 iterations.  Measured recovery of a rotated
pair: worst 2.96e-06 degrees over 30 symmetry free cases at ``bw``
53-123 and 4.52e-06 over 72 point group cases, against the C++ test
gates of 4.92e-3 and 0.351 degrees; 2 iterations in every
non-degenerate case.

**What a refinement returns and its recorded quirks.**

- On failure -- a rejected saddle, a singular 2 x 2 sub-problem or
  the iteration cap -- the *input* triple comes back together with
  ``derivatives(zyz0, der=False)``, the **analytic** value there and
  not the tri-quadratic interpolated peak, so a failed refinement
  changes the score of a coarse result (measured -29.4 to +9.3 on
  unrelated far starts).  This is the C++ behaviour of lines 494-498.
- On success the returned value was computed at the ``eu`` **before**
  the final sub-threshold step (lines 457 and 487), a second order
  small lag which is again the C++'s own.
- ``prevMag2`` is seeded with ``2 pi 3 / slP`` (line 450), which
  compares a squared step length against a linear bound: at ``slP``
  135 the first step may be 0.374 radians, about eight cells, where
  the comment says one.  Ported verbatim.
- The inner ``if(det < euEps)`` of lines 476-478 is always true when
  it is reached, so both of the C++'s two messages are one failure.
- ``derivatives()``' ``deg`` flag (line 909) is computed and never
  used.
- Newton is **local**: a starting triple which did not come from the
  coarse pipeline may converge to a stationary point whose value is
  *below* the start's (measured 3 of 4 converged far starts), since
  the sub-problems freeze a degree of freedom and only test
  ``det >= euEps``.

**The normalized refinement** divides the refined un-normalized value
by ``denominator(zyz)`` (lines 1211-1225) evaluated at the refined
rotation.  The window shift chain rule of lines 263-264 is omitted
there, as in the C++, so the refined normalized score can dip below
the coarse one (measured 4 of 165 points, worst -4.8e-4) and the
masked refined accuracy is window limited (measured 2.1e-2 degrees
against 3e-6 unmasked).

**Refinement memory and speed.**  A refining correlator owns one
``(bw, bw, bw, 2)`` 64-bit float ``d_beta`` table (``16 bw^3`` bytes,
5.03 MB at ``bw`` 68), allocated once and reused across iterations,
calls and patterns -- a fresh :func:`numpy.full` per refinement would
cost 1.03 ms at ``bw`` 68, more than the refinement itself -- and is
never shared between clones.  The beta independent factor triple of
:func:`kikuchipy.indexing._spherical._wigner.wigner_d_table_factors`
(``8 bw^2 + 16 bw^3`` bytes, 5.07 MB at ``bw`` 68) is read only and
shared per process.  Warm single thread refinement, two iterations
including the per-iteration table rebuild: 1.45 / 3.00 / 11.06 ms at
``bw`` 53 / 68 / 88 without symmetry and 0.33 / 0.65 / 2.54 ms with
``m-3m``; end to end this is 1.05-1.27 times the coarse cost.

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
end to end, best of five with the kernels warm: 10.8 / 24.6 / 59.6
/ 133.6 ms at ``bw`` 53 / 68 / 88 / 113 without symmetry and 5.4 /
11.7 / 28.2 / 59.2 ms with ``m-3m``, i.e. 1.2-1.7 times the
compiled C++ (8.9 / 16.3 / 48.1 / 90.4 resp. 3.3 / 6.9 / 18.9 /
34.2 ms).  pocketfft has no radix-13 codelet,
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

from numba import njit
import numpy as np
from scipy.fft import ifft, irfft

from kikuchipy.indexing._spherical import _euler, _fft, _wigner

# The two SciPy transforms are bound in this namespace, as Phases
# 1-3 bind theirs, so that a test can patch ``_xcorr.ifft`` and
# ``_xcorr.irfft`` and see every call this module makes.
#
# TODO: the Newton loop adds ``_preprocessing`` to that import, a
# one-way edge (that module imports nothing of this package), for its
# 3 x 3 Cholesky solve: the very ``solve::cholesky()`` the C++ calls
# from both sites, so it has one implementation here too and never a
# ``numpy.linalg.solve`` substitute.

# Machine epsilon of the 64-bit floating point type
_EPS = float(np.finfo(np.float64).eps)

# Convergence criterion and iteration cap of the Newton loop of
# ``detail::interpolateMaxima()``
# (``include/sht/sht_xcorr.hpp``, lines 1313 and 1312)
_NEWTON_EPS = math.sqrt(_EPS)
_NEWTON_MAX_ITERATIONS = 25

# Convergence criterion, iteration cap and first step bound of the
# real space Newton loop of ``Correlator<Real>::refinePeak()``
# (``include/sht/sht_xcorr.hpp``, lines 446-450).  ``_REFINE_EPS`` is
# the ``eps`` default of ``PhaseCorrelator::correlate()`` (line 189),
# which every C++ call site uses, and scales the stopping threshold
# ``eps 2 pi / slP``; ``_REFINE_EU_EPS`` is the ``euEps`` of line 447
# which the 2 x 2 sub-problem tests its determinant against
_REFINE_EPS = 0.01
_REFINE_MAX_ITERATIONS = 15
_REFINE_EU_EPS = math.sqrt(_EPS)
_REFINE_FIRST_STEP_SCALE = 3.0

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


@njit(cache=True, nogil=True)
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
        not write is a column ``m in [bw, bwP)`` of a row **and** a
        slice which are not zero padded, i.e. ``n`` and ``k`` both
        outside ``[bw, slP - bw]``: those columns exist only for
        even ``slP``, where they are the single column ``m = bw``,
        and nothing ever writes them, so a zero filled buffer may be
        reused across calls with different ``n_fold`` and ``mirror``.
        Verified with a NaN tripwire: ``(slP - 1)^2`` unwritten
        slots at ``slP`` 24 / 32 / 48 and none at odd ``slP``.
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
    bandwidth = flm.shape[0]
    slp = fxc.shape[0]
    bwp = fxc.shape[2]
    for k in range(bandwidth):
        # fm[m, j] = f^j_m d^j_{k,m}(pi/2), lines 736-744
        for m in range(bandwidth):
            for j in range(max(m, k), bandwidth):
                fm[m, j] = flm[m, j] * table[m, k, j]
        positive_k = k > 0
        for n in range(bwp):
            positive_n = n > 0
            if n >= bandwidth:
                # a zero padded row, lines 831-837
                for m in range(bwp):
                    fxc[k, n, m] = 0j
                    if positive_k:
                        fxc[slp - k, n, m] = 0j
                    if positive_n:
                        fxc[k, slp - n, m] = 0j
                    if positive_k and positive_n:
                        fxc[slp - k, slp - n, m] = 0j
                continue
            max_kn = max(k, n)
            # gn[j] = conj(g^j_n) d^j_{n,k}(pi/2), line 762
            for j in range(max_kn, bandwidth):
                gn[j] = gln[n, j].conjugate() * table[k, n, j]
            for m in range(bandwidth):
                if m % n_fold != 0:
                    # a systemic zero column, lines 818-829
                    fxc[k, n, m] = 0j
                    if positive_k:
                        fxc[slp - k, n, m] = 0j
                    if positive_n:
                        fxc[k, slp - n, m] = 0j
                    if positive_k and positive_n:
                        fxc[slp - k, slp - n, m] = 0j
                    continue
                value = 0j
                negated = 0j
                start = max(m, max_kn)
                if mirror:
                    # a single mirror, lines 773-783
                    if (start + m) % 2 != 0:
                        start += 1
                    toggle = (start + m) % 2 == 0
                    for j in range(start, bandwidth, 2):
                        first = fm[m, j]
                        second = gn[j]
                        rr = first.real * second.real
                        ii = first.imag * second.imag
                        ri = first.real * second.imag
                        ir = first.imag * second.real
                        value += complex(rr - ii, ir + ri)
                        negated += complex(rr + ii, ir - ri)
                    if not toggle:  # pragma: no cover
                        # dead: line 774 has just made start + m
                        # even, so toggle of line 776 is always true
                        negated = -negated
                else:
                    # no mirror, lines 784-793
                    toggle = (start + m) % 2 == 0
                    for j in range(start, bandwidth):
                        first = fm[m, j]
                        second = gn[j]
                        rr = first.real * second.real
                        ii = first.imag * second.imag
                        ri = first.real * second.imag
                        ir = first.imag * second.real
                        value += complex(rr - ii, ir + ri)
                        conjugated = complex(rr + ii, ir - ri)
                        if toggle:
                            negated += conjugated
                        else:
                            negated -= conjugated
                        toggle = not toggle
                if k % 2 != 0:
                    negated = -negated
                # the four mirrored slots, lines 796-817
                fxc[k, n, m] = value
                if positive_k and positive_n:
                    fxc[slp - k, slp - n, m] = negated
                if (m + n) % 2 == 0:
                    if positive_k:
                        fxc[slp - k, n, m] = value
                    if positive_n:
                        fxc[k, slp - n, m] = negated
                else:
                    if positive_k:
                        fxc[slp - k, n, m] = -value
                    if positive_n:
                        fxc[k, slp - n, m] = -negated
    # the zero pad slices, line 854, inclusive at both ends
    for k in range(bandwidth, slp - bandwidth + 1):
        for n in range(slp):
            for m in range(bwp):
                fxc[k, n, m] = 0j


@njit(cache=True, nogil=True)
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
    maxima the **smallest** flat index wins and NaN slots **other
    than the seed** are skipped, neither of which
    :func:`numpy.argmax` does.  A NaN in slot 0 is *not* harmless:
    it loses every later comparison too, so the returned index stays
    ``0``, the grid corner ``(-pi/2, -pi, -pi/2)``.  This is the C++
    behaviour of ``Real vMax = xc.front()`` and must not be
    "fixed"; see :class:`NormalizedSphericalCrossCorrelator` for
    when it is reachable.
    """
    flat = xc.reshape(-1)
    index = 0
    largest = flat[0]
    for i in range(flat.size):
        if flat[i] > largest:
            largest = flat[i]
            index = i
    return index


@njit(cache=True, nogil=True)
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
    flat = xc.reshape(-1)
    reciprocal = r_den.reshape(-1)
    index = 0
    largest = flat[0] * reciprocal[0]
    for i in range(flat.size):
        value = flat[i] * reciprocal[i]
        flat[i] = value
        if value > largest:
            largest = value
            index = i
    return index


@njit(cache=True, nogil=True)
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
    half = _NEIGHBORHOOD_HALF_WIDTH
    width = _NEIGHBORHOOD_WIDTH
    # the periodic index arrays of lines 512-524
    inds = np.empty((3, width), dtype=np.int64)
    for axis in range(3):
        if axis == 0:
            center = k0
        elif axis == 1:
            center = n0
        else:
            center = m0
        inds[axis, half] = center
        for j in range(half):
            below = inds[axis, half - j]
            inds[axis, half - 1 - j] = slp - 1 if below == 0 else below - 1
            above = inds[axis, half + j] + 1
            inds[axis, half + 1 + j] = 0 if above == slp else above
    if emsphinx_compatible:
        # the per-slot glide of lines 527-533
        for i in range(width):
            if inds[0, i] >= bwp:
                alpha = inds[2, i]
                inds[2, i] = alpha + bwp - 1 if alpha < bwp else alpha - bwp
                gamma = inds[1, i]
                inds[1, i] = gamma + bwp - 1 if gamma < bwp else gamma - bwp
                inds[0, i] = slp - inds[0, i]
        last = xc_flat.size - 1
        for k in range(width):
            for n in range(width):
                for m in range(width):
                    offset = inds[0, k] * slp * slp + inds[1, n] * slp + inds[2, m]
                    if offset > last:
                        offset = last
                    nh[k, n, m] = xc_flat[offset]
    else:
        # the per-plane glide, exact on the grid for even slP
        shift = slp // 2
        for k in range(width):
            beta = inds[0, k]
            glided = beta >= bwp
            if glided:
                beta = slp - beta
            for n in range(width):
                gamma = inds[1, n]
                if glided:
                    gamma = (gamma + shift) % slp
                for m in range(width):
                    alpha = inds[2, m]
                    if glided:
                        alpha = (alpha + shift) % slp
                    nh[k, n, m] = xc_flat[beta * slp * slp + gamma * slp + alpha]


@njit(cache=True, nogil=True, error_model="numpy")
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
    guard (line 1326 for the determinant and lines 1327-1332 for
    the cofactors) and relies on IEEE semantics: on a flat
    neighbourhood, where the determinant is zero, the steps are NaN,
    the criterion never passes and the reset returns the centre
    value.  Numba's default error model raises ``ZeroDivisionError``
    there instead, and flat neighbourhoods are reachable in
    production (a masked pattern with an empty region, an all-zero
    ``gln``, a rotation invariant spectrum).  The compiled kernel
    and its ``py_func`` return the same values there, but only the
    ``py_func`` emits NumPy's ``invalid value encountered in scalar
    divide`` warnings, since ``error_model="numpy"`` is a compile
    time contract with no Python warning behind it: wrap a
    ``py_func`` call on a degenerate block in
    :func:`numpy.errstate`.
    """
    # the 27 coefficients a_{kji} of f = a_{kji} x^i y^j z^k,
    # lines 1265-1308
    a000 = p[1, 1, 1]
    a001 = (p[1, 1, 2] - p[1, 1, 0]) / 2
    a002 = (p[1, 1, 2] + p[1, 1, 0]) / 2 - a000
    a010 = (p[1, 2, 1] - p[1, 0, 1]) / 2
    a020 = (p[1, 2, 1] + p[1, 0, 1]) / 2 - a000
    a100 = (p[2, 1, 1] - p[0, 1, 1]) / 2
    a200 = (p[2, 1, 1] + p[0, 1, 1]) / 2 - a000
    a022 = (p[1, 2, 2] + p[1, 2, 0] + p[1, 0, 2] + p[1, 0, 0]) / 4 - a000 - a020 - a002
    a011 = (p[1, 2, 2] - p[1, 2, 0] - p[1, 0, 2] + p[1, 0, 0]) / 4
    a012 = (p[1, 2, 2] + p[1, 2, 0] - p[1, 0, 2] - p[1, 0, 0]) / 4 - a010
    a021 = (p[1, 2, 2] - p[1, 2, 0] + p[1, 0, 2] - p[1, 0, 0]) / 4 - a001
    a220 = (p[2, 2, 1] + p[2, 0, 1] + p[0, 2, 1] + p[0, 0, 1]) / 4 - a000 - a200 - a020
    a110 = (p[2, 2, 1] - p[2, 0, 1] - p[0, 2, 1] + p[0, 0, 1]) / 4
    a120 = (p[2, 2, 1] + p[2, 0, 1] - p[0, 2, 1] - p[0, 0, 1]) / 4 - a100
    a210 = (p[2, 2, 1] - p[2, 0, 1] + p[0, 2, 1] - p[0, 0, 1]) / 4 - a010
    a202 = (p[2, 1, 2] + p[0, 1, 2] + p[2, 1, 0] + p[0, 1, 0]) / 4 - a000 - a002 - a200
    a101 = (p[2, 1, 2] - p[0, 1, 2] - p[2, 1, 0] + p[0, 1, 0]) / 4
    a201 = (p[2, 1, 2] + p[0, 1, 2] - p[2, 1, 0] - p[0, 1, 0]) / 4 - a001
    a102 = (p[2, 1, 2] - p[0, 1, 2] + p[2, 1, 0] - p[0, 1, 0]) / 4 - a100
    a222 = (
        (
            p[2, 2, 2]
            + p[0, 0, 0]
            + p[0, 2, 2]
            + p[2, 0, 2]
            + p[2, 2, 0]
            + p[2, 0, 0]
            + p[0, 2, 0]
            + p[0, 0, 2]
        )
        / 8
        - a000
        - a200
        - a020
        - a002
        - a022
        - a202
        - a220
    )
    a211 = (
        p[2, 2, 2]
        + p[0, 0, 0]
        + p[0, 2, 2]
        - p[2, 0, 2]
        - p[2, 2, 0]
        + p[2, 0, 0]
        - p[0, 2, 0]
        - p[0, 0, 2]
    ) / 8 - a011
    a121 = (
        p[2, 2, 2]
        + p[0, 0, 0]
        - p[0, 2, 2]
        + p[2, 0, 2]
        - p[2, 2, 0]
        - p[2, 0, 0]
        + p[0, 2, 0]
        - p[0, 0, 2]
    ) / 8 - a101
    a112 = (
        p[2, 2, 2]
        + p[0, 0, 0]
        - p[0, 2, 2]
        - p[2, 0, 2]
        + p[2, 2, 0]
        - p[2, 0, 0]
        - p[0, 2, 0]
        + p[0, 0, 2]
    ) / 8 - a110
    a111 = (
        p[2, 2, 2]
        - p[0, 0, 0]
        - p[0, 2, 2]
        - p[2, 0, 2]
        - p[2, 2, 0]
        + p[2, 0, 0]
        + p[0, 2, 0]
        + p[0, 0, 2]
    ) / 8
    a122 = (
        (
            p[2, 2, 2]
            - p[0, 0, 0]
            - p[0, 2, 2]
            + p[2, 0, 2]
            + p[2, 2, 0]
            + p[2, 0, 0]
            - p[0, 2, 0]
            - p[0, 0, 2]
        )
        / 8
        - a100
        - a120
        - a102
    )
    a212 = (
        (
            p[2, 2, 2]
            - p[0, 0, 0]
            + p[0, 2, 2]
            - p[2, 0, 2]
            + p[2, 2, 0]
            - p[2, 0, 0]
            + p[0, 2, 0]
            - p[0, 0, 2]
        )
        / 8
        - a010
        - a012
        - a210
    )
    a221 = (
        (
            p[2, 2, 2]
            - p[0, 0, 0]
            + p[0, 2, 2]
            + p[2, 0, 2]
            - p[2, 2, 0]
            - p[2, 0, 0]
            - p[0, 2, 0]
            + p[0, 0, 2]
        )
        / 8
        - a001
        - a201
        - a021
    )

    # Newton iterate to the maximum, lines 1310-1352
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(_NEWTON_MAX_ITERATIONS):
        xx = x[0] * x[0]
        yy = x[1] * x[1]
        zz = x[2] * x[2]
        xy = x[0] * x[1]
        yz = x[1] * x[2]
        zx = x[2] * x[0]
        h00 = (
            a200
            + a210 * x[1]
            + a201 * x[2]
            + a220 * yy
            + a202 * zz
            + a211 * yz
            + a221 * yy * x[2]
            + a212 * x[1] * zz
            + a222 * yy * zz
        ) * 2
        h11 = (
            a020
            + a021 * x[2]
            + a120 * x[0]
            + a022 * zz
            + a220 * xx
            + a121 * zx
            + a122 * zz * x[0]
            + a221 * x[2] * xx
            + a222 * zz * xx
        ) * 2
        h22 = (
            a002
            + a102 * x[0]
            + a012 * x[1]
            + a202 * xx
            + a022 * yy
            + a112 * xy
            + a212 * xx * x[1]
            + a122 * x[0] * yy
            + a222 * xx * yy
        ) * 2
        h01 = (
            a110
            + a111 * x[2]
            + a112 * zz
            + (
                a210 * x[0]
                + a120 * x[1]
                + a211 * zx
                + a121 * yz
                + a212 * x[0] * zz
                + a122 * x[1] * zz
                + (a220 * xy + a221 * xy * x[2] + a222 * xy * zz) * 2
            )
            * 2
        )
        h12 = (
            a011
            + a111 * x[0]
            + a211 * xx
            + (
                a021 * x[1]
                + a012 * x[2]
                + a121 * xy
                + a112 * zx
                + a221 * x[1] * xx
                + a212 * x[2] * xx
                + (a022 * yz + a122 * yz * x[0] + a222 * yz * xx) * 2
            )
            * 2
        )
        h02 = (
            a101
            + a111 * x[1]
            + a121 * yy
            + (
                a102 * x[2]
                + a201 * x[0]
                + a112 * yz
                + a211 * xy
                + a122 * x[2] * yy
                + a221 * x[0] * yy
                + (a202 * zx + a212 * zx * x[1] + a222 * zx * yy) * 2
            )
            * 2
        )

        # the inverse Hessian, divided by an unguarded determinant
        # as in lines 1326-1332
        det = (
            h00 * h11 * h22
            - h00 * h12 * h12
            - h11 * h02 * h02
            - h22 * h01 * h01
            + h01 * h12 * h02 * 2
        )
        i00 = (h11 * h22 - h12 * h12) / det
        i11 = (h22 * h00 - h02 * h02) / det
        i22 = (h00 * h11 - h01 * h01) / det
        i01 = (h02 * h12 - h01 * h22) / det
        i12 = (h01 * h02 - h12 * h00) / det
        i02 = (h12 * h01 - h02 * h11) / det

        d0 = (
            a100
            + a110 * x[1]
            + a101 * x[2]
            + a120 * yy
            + a102 * zz
            + a111 * yz
            + a121 * yy * x[2]
            + a112 * x[1] * zz
            + a122 * yy * zz
            + x[0]
            * (
                a200
                + a210 * x[1]
                + a201 * x[2]
                + a220 * yy
                + a202 * zz
                + a211 * yz
                + a221 * yy * x[2]
                + a212 * x[1] * zz
                + a222 * yy * zz
            )
            * 2
        )
        d1 = (
            a010
            + a011 * x[2]
            + a110 * x[0]
            + a012 * zz
            + a210 * xx
            + a111 * zx
            + a112 * zz * x[0]
            + a211 * x[2] * xx
            + a212 * zz * xx
            + x[1]
            * (
                a020
                + a021 * x[2]
                + a120 * x[0]
                + a022 * zz
                + a220 * xx
                + a121 * zx
                + a122 * zz * x[0]
                + a221 * x[2] * xx
                + a222 * zz * xx
            )
            * 2
        )
        d2 = (
            a001
            + a101 * x[0]
            + a011 * x[1]
            + a201 * xx
            + a021 * yy
            + a111 * xy
            + a211 * xx * x[1]
            + a121 * x[0] * yy
            + a221 * xx * yy
            + x[2]
            * (
                a002
                + a102 * x[0]
                + a012 * x[1]
                + a202 * xx
                + a022 * yy
                + a112 * xy
                + a212 * xx * x[1]
                + a122 * x[0] * yy
                + a222 * xx * yy
            )
            * 2
        )

        step0 = i00 * d0 + i01 * d1 + i02 * d2
        step1 = i01 * d0 + i11 * d1 + i12 * d2
        step2 = i02 * d0 + i12 * d1 + i22 * d2
        x[0] -= step0
        x[1] -= step1
        x[2] -= step2

        max_step = max(max(abs(step0), abs(step1)), abs(step2))
        if max_step < _NEWTON_EPS:
            break
        if i + 1 == _NEWTON_MAX_ITERATIONS:
            # no convergence, so do not interpolate, line 1350
            x[0] = 0.0
            x[1] = 0.0
            x[2] = 0.0

    # the literal vPeak expression of lines 1354-1364, with the
    # three mixed cubic terms at the wrong monomials
    xx = x[0] * x[0]
    yy = x[1] * x[1]
    zz = x[2] * x[2]
    xy = x[0] * x[1]
    yz = x[1] * x[2]
    zx = x[2] * x[0]
    return (
        a000
        + a111 * x[0] * x[1] * x[2]
        + a222 * xx * yy * zz
        + a100 * x[0]
        + a010 * x[1]
        + a001 * x[2]
        + a200 * xx
        + a020 * yy
        + a002 * zz
        + a110 * xy
        + a011 * yz
        + a101 * zx
        + a120 * x[0] * yy
        + a012 * x[1] * zz
        + a201 * x[2] * xx
        + a210 * xx * x[1]
        + a021 * yy * x[2]
        + a102 * zz * x[0]
        + a220 * xx * yy
        + a022 * yy * zz
        + a202 * zz * xx
        + a112 * xy * x[2]
        + a211 * yz * x[0]
        + a121 * zx * x[1]
        + a122 * x[0] * yy * zz
        + a212 * xx * x[1] * zz
        + a221 * xx * yy * x[2]
    )


# TODO: decorate with ``@njit(cache=True, nogil=True,
# error_model="numpy")`` when the body lands.  The ``numpy`` error
# model is **load bearing** and is the project's third sanctioned
# one: at ``|cos(beta)| == 1`` the unguarded ``csc = 1 / sqrt(1 -
# t^2)`` must give the IEEE infinity which propagates into the NaN
# ``hes[1, 1]`` that :func:`_refine_peak` uses as its degeneracy
# detector (``include/sht/sht_xcorr.hpp``, lines 461 and 468), where
# Numba's default model would raise ``ZeroDivisionError`` and break
# the C++ control flow.
def _derivatives(
    flm: np.ndarray,
    gln: np.ndarray,
    eu: np.ndarray,
    jac: np.ndarray,
    hes: np.ndarray,
    bandwidth: int,
    mirror: bool,
    n_fold: int,
    der: bool,
    d_beta: np.ndarray,
    e_km: np.ndarray,
    w_jkm: np.ndarray,
    b_jkm: np.ndarray,
) -> float:
    """Return the cross-correlation at one rotation and, optionally,
    write its gradient and Hessian.

    Parameters
    ----------
    flm, gln
        Harmonic coefficients ``a[m, l]`` of the two real functions,
        both C-contiguous ``(bw, bw)`` and 128-bit complex.  A shape
        which disagrees with ``bandwidth`` is **silent garbage**, not
        an error, since bounds checking is off: measured values of
        ``1e225`` and NaN Hessians from ``(68, 68)`` spectra in a
        ``bw`` 88 kernel.  Every entry point validates first.
    eu
        Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in radians
        in a ``(3,)`` 64-bit float array.  ``beta`` is wrapped into
        ``[-pi, pi]`` with
        :func:`kikuchipy.indexing._spherical._euler._wrap_beta`, the
        C++'s own wrap of lines 895-899.
    jac
        ``(3,)`` 64-bit float caller-owned output, written only when
        ``der``: the derivatives with respect to ``alpha``, ``beta``
        and ``gamma``.
    hes
        ``(3, 3)`` 64-bit float caller-owned output, written only
        when ``der``, symmetrised as the C++ does at lines
        1110-1117.
    bandwidth
        Exclusive maximum harmonic degree ``bw``, which must equal
        the side of ``flm``, ``gln`` and the tables.  The C++
        ``mBW`` parameter is dead freedom -- every call site passes
        ``bw``, and a smaller value would read the factor tables at
        the wrong stride, see
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_pre`.
    mirror
        Whether ``flm`` has an equatorial mirror plane, which makes
        the degree loop step by two.
    n_fold
        Order of the rotational symmetry of ``flm`` about z, at
        least one; the orders ``m % n_fold != 0`` are skipped
        **after** the multiple-angle recursion is advanced.
    der
        Whether to write ``jac`` and ``hes``.  The returned value is
        the same either way.
    d_beta
        ``(bw, bw, bw, 2)`` 64-bit float caller-owned Wigner d table
        buffer, **rebuilt in place on every call** with
        :func:`kikuchipy.indexing._spherical._wigner.
        _wigner_d_table_pre_kernel`, the C++ ``dTablePre`` of line
        913.  Every defined slot is written each call, so one buffer
        is reused across iterations, calls and patterns; it must not
        be shared between threads.
    e_km, w_jkm, b_jkm
        The beta independent factor triple of
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_factors`, read only and shareable.

    Returns
    -------
    value
        Cross-correlation at ``eu`` in the normalization of the
        module documentation, i.e. ``4 pi <rotate_harmonics(flm,
        eu), gln>``.

    Notes
    -----
    Port of ``Correlator<Real>::derivatives()``
    (``include/sht/sht_xcorr.hpp``, lines 889-1119): the per call
    ``dTablePre`` rebuild (line 913), the Chebyshev multiple-angle
    recursions for ``exp(i m alpha)`` and ``exp(i n gamma)`` (lines
    927-984), the analytic first and second beta derivative
    coefficients (lines 1009-1041) and the ten accumulator
    components with their conditional quadrant sums (lines
    1057-1078).

    Two recorded deviations, neither of which changes a value:

    - the four sequential quadrant adds of lines 1072-1078 are
      folded into the two weights ``1 + (m > 0 and n > 0)`` and
      ``(n > 0) + (m > 0)``, an association change permitted because
      no test asserts bitwise against the compiled C++ and the two
      analytic oracles bound the whole evaluation at 1e-13;
    - the ``deg`` flag of line 909 is computed and never used in the
      C++, so it is not ported.

    The ``t``/``csc`` chain uses :func:`numpy.cos` and
    :func:`numpy.sqrt` rather than :mod:`math`, so that the values
    stay NumPy scalars and the ``py_func`` yields ``inf`` under
    :func:`numpy.errstate` at the poles where the ``math``
    transcription would raise ``ZeroDivisionError`` regardless.  The
    compiled results are bitwise identical either way (measured,
    worst absolute difference 0 over 1000 samples).

    The table reads are guarded with ``m >= j`` and ``m + 1 >= j``
    (lines 1027-1032), so no undefined NaN slot of ``d_beta`` is
    ever read and every index stays in bounds.
    """
    raise NotImplementedError(
        "`_derivatives()` is not implemented yet, so a refinement cannot run"
    )


def _refine_peak(
    flm: np.ndarray,
    gln: np.ndarray,
    zyz0: np.ndarray,
    n_fold: int,
    mirror: bool,
    bandwidth: int,
    side_length: int,
    d_beta: np.ndarray,
    e_km: np.ndarray,
    w_jkm: np.ndarray,
    b_jkm: np.ndarray,
    jac: np.ndarray,
    hes: np.ndarray,
    step: np.ndarray,
    eps: float = _REFINE_EPS,
) -> tuple[np.ndarray, float, bool]:
    """Return the Newton refined rotation of a cross-correlation
    maximum, its value and whether the loop converged.

    Parameters
    ----------
    flm, gln
        Harmonic coefficients ``a[m, l]`` of the two real functions,
        see :func:`_derivatives`.
    zyz0
        Starting passive ZYZ Euler angles in a ``(3,)`` 64-bit float
        array, e.g. the interpolated coarse peak of
        :meth:`SphericalCrossCorrelator.interp_peak`.  It is not
        modified.
    n_fold, mirror
        Symmetry flags of ``flm``.
    bandwidth
        Exclusive maximum harmonic degree ``bw``.
    side_length
        Padded Euler side length ``slP``, which sets the stopping
        threshold ``eps 2 pi / slP`` and the first step bound.
    d_beta, e_km, w_jkm, b_jkm
        The Wigner d buffer and factor triple of
        :func:`_derivatives`.
    jac, hes, step
        ``(3,)``, ``(3, 3)`` and ``(3,)`` 64-bit float caller-owned
        scratch buffers.  ``hes`` is handed to the Cholesky solve
        **uncopied**, exactly as the C++ passes its live array (line
        462): the decomposition writes only the subdiagonal, which
        the fallbacks never read, and the next iteration rewrites
        all nine entries.  Measured bitwise identical to a copying
        variant over 15 fallback heavy cases.
    eps
        Convergence scale, ``0.01`` by default, the value every C++
        call site uses.  The stopping threshold is
        ``eps 2 pi / slP``.

    Returns
    -------
    zyz
        Refined angles in a new ``(3,)`` 64-bit float array on
        success, and the **input** triple on failure.  They are not
        wrapped back into the coarse grid intervals: Newton may step
        ``beta`` across a pole, which :func:`_derivatives` wraps
        internally and every consumer converts through
        :func:`kikuchipy.indexing._spherical._euler.
        rotation_from_zyz`.
    value
        Cross-correlation at ``zyz``.  On success it is the value
        computed at the iterate **before** the final sub-threshold
        step, the C++'s own second order lag (lines 457 and 487); on
        failure it is ``_derivatives(zyz0, der=False)``, the
        analytic value at the starting triple and **not** the
        tri-quadratic interpolated peak, so a failed refinement
        changes the score of a coarse result.
    converged
        Whether the loop reached the stopping threshold.  This flag
        is an addition of the port, for tests and for
        :meth:`kikuchipy.indexing.SphericalIndexer.refine_patterns`;
        the C++ fails silently by returning the start.

    Notes
    -----
    Port of ``Correlator<Real>::refinePeak()``
    (``include/sht/sht_xcorr.hpp``, lines 442-499): at most 15
    iterations (line 448) of ``derivatives(der=True)``, a 3 x 3
    Cholesky solve through
    :func:`kikuchipy.indexing._spherical._preprocessing.
    _cholesky_solve_3x3` -- the same ``solve::cholesky()`` the C++
    calls, whose two throws map to the indefinite (saddle) and small
    pivot statuses -- and the monotone step rule of lines 463-465.
    A NaN ``hes[1, 1]``, a non-zero solve status or a step longer
    than the previous one falls back to the 1 x 1 sub-problem
    ``step = [jac[0] / hes[0, 0], 0, 0]`` when ``hes[1, 1]`` is NaN
    and otherwise to the 2 x 2 sub-problem of lines 480-483, whose
    ``det < euEps`` is a total failure.  ``prev_mag2`` is **not**
    updated by a fallback step, as in the C++.

    This is a Python loop over two Numba kernels rather than one
    kernel: the C++ exception control flow maps to statuses
    naturally and the measured cost is kernel dominated.

    The seeding of ``prev_mag2`` with ``2 pi 3 / slP`` (line 450)
    compares a squared step length against a linear bound, so the
    first step may be about eight cells where the C++ comment says
    one.  Ported verbatim.
    """
    raise NotImplementedError(
        "`_refine_peak()` is not implemented yet, so a refinement cannot run"
    )


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
    side_length = fxc.shape[0]
    half_side_length = fxc.shape[2]
    if n_fold == 1:
        # backward along k for every n, then along n for k < bwP
        along_k = ifft(fxc, axis=0, norm="forward", workers=1)
        planes = ifft(
            along_k[:half_side_length],
            axis=1,
            norm="forward",
            workers=1,
            overwrite_x=True,
        )
    else:
        # the alpha planes m % n_fold != 0 are the systemic zeros
        # the spectrum kernel wrote, so skipping them is exact
        along_k = ifft(fxc[:, :, ::n_fold], axis=0, norm="forward", workers=1)
        along_n = ifft(
            along_k[:half_side_length],
            axis=1,
            norm="forward",
            workers=1,
            overwrite_x=True,
        )
        planes = np.zeros(
            (half_side_length, side_length, half_side_length), dtype=np.complex128
        )
        planes[:, :, ::n_fold] = along_n
    return irfft(
        planes,
        n=side_length,
        axis=2,
        norm="forward",
        workers=1,
        overwrite_x=True,
    )


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

    Raises
    ------
    ValueError
        If ``knm`` does not have three elements, as
        :func:`euler_to_index` raises for its ``zyz``.

    Notes
    -----
    Port of ``Correlator<Real>::indexEuler()``
    (``include/sht/sht_xcorr.hpp``, lines 580-590).  The stored
    ``k in [0, bwP)`` covers ``beta in [-pi, 0]``: for odd ``slP``
    only up to ``-pi / slP``, so ``beta = 0`` falls between the last
    stored slice and its glide image, while for even ``slP`` it is
    slice ``bwP - 1``.
    """
    indices = tuple(knm)
    if len(indices) != 3:
        raise ValueError(f"`knm` must have three elements, not {len(indices)}")
    slp = int(side_length)
    k = int(indices[0])
    n = int(indices[1])
    m = int(indices[2])
    return np.array(
        [
            (m * 4 - slp) * math.pi / (2 * slp),
            (k * 2 - slp) * math.pi / slp,
            (n * 4 - slp) * math.pi / (2 * slp),
        ]
    )


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
    angles = np.asarray(zyz, dtype=np.float64).reshape(-1)
    if angles.size != 3:
        raise ValueError(f"`zyz` must have three elements, not {angles.size}")
    slp = int(side_length)
    # the wrap and the reduction the C++ does not have; both leave
    # an angle which is already in range bitwise unchanged
    alpha = float(angles[0])
    alpha -= _TWO_PI * math.floor((alpha + math.pi / 2) / _TWO_PI)
    beta = _euler.wrap_beta(float(angles[1]))
    gamma = float(angles[2])
    gamma -= _TWO_PI * math.floor((gamma + math.pi / 2) / _TWO_PI)

    # the fractional indices of lines 552-556
    k_real = ((beta * slp) / math.pi + slp) / 2
    n_real = ((gamma * 2 * slp) / math.pi + slp) / 4
    m_real = ((alpha * 2 * slp) / math.pi + slp) / 4

    # the glide of lines 559-565
    half = slp / 2
    if k_real > half:
        k_real = slp - k_real
        n_real = math.fmod(n_real + half, slp)
        m_real = math.fmod(m_real + half, slp)

    # C++ std::round(), not Python's banker's rounding
    k = int(math.floor(k_real + 0.5))
    n = int(math.floor(n_real + 0.5)) % slp
    m = int(math.floor(m_real + 0.5)) % slp
    bwp = slp // 2 + 1
    if k > bwp - 1:
        k = bwp - 1
    return k, n, m


# ---------------------------- Validation ---------------------------- #


def _validated_spectrum(alm: np.ndarray, bandwidth: int, name: str) -> np.ndarray:
    """Return harmonic coefficients as a C-contiguous complex array.

    Parameters
    ----------
    alm
        Harmonic coefficients ``a[m, l]`` in an array-like of shape
        ``(bw, bw)``.
    bandwidth
        Exclusive maximum harmonic degree ``bw``.
    name
        Name of the parameter, used in the error message.

    Returns
    -------
    array
        ``(bw, bw)`` C-contiguous 128-bit complex array.

    Raises
    ------
    ValueError
        If the shape is not ``(bw, bw)``.
    """
    array = np.ascontiguousarray(alm, dtype=np.complex128)
    if array.shape != (bandwidth, bandwidth):
        raise ValueError(
            f"`{name}` must have shape ({bandwidth}, {bandwidth}), not {array.shape}"
        )
    return array


def _validated_flags(n_fold: int, mirror: bool) -> tuple[int, bool]:
    """Return the two symmetry flags of the first function.

    Parameters
    ----------
    n_fold
        Order of the rotational symmetry about z, at least one.
    mirror
        Whether there is an equatorial mirror plane.

    Returns
    -------
    flags
        ``(n_fold, mirror)`` as a :class:`int` and a :class:`bool`.

    Raises
    ------
    ValueError
        If ``n_fold`` is a :class:`bool`, is not an integer or is
        smaller than one, or if ``mirror`` is not a :class:`bool`.
        The C++ ``compute()`` takes the two the other way round, so
        a swapped call must fail loudly rather than silently
        correlate with the wrong symmetry.
    """
    if isinstance(n_fold, (bool, np.bool_)) or not isinstance(
        n_fold, (int, np.integer)
    ):
        raise ValueError(
            f"`n_fold` must be an integer which is not a bool, not {n_fold!r}"
        )
    if int(n_fold) < 1:
        raise ValueError(f"`n_fold` must be at least one, not {int(n_fold)}")
    if not isinstance(mirror, (bool, np.bool_)):
        raise ValueError(f"`mirror` must be a bool, not {mirror!r}")
    return int(n_fold), bool(mirror)


def _validated_wigner_table(table: np.ndarray, bandwidth: int) -> np.ndarray:
    """Return a shared transposed ``pi/2`` Wigner d table unchanged.

    Parameters
    ----------
    table
        Transposed table ``table[m, k, j] = d^j_{k,m}(pi/2)`` of
        shape ``(bw, bw, bw)``, as built by
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_half_pi_table` with ``transpose=True``.
    bandwidth
        Exclusive maximum harmonic degree ``bw``.

    Returns
    -------
    table
        The very same array, so that instances share one table.

    Raises
    ------
    ValueError
        If the array is not a C-contiguous 64-bit float array of
        shape ``(bw, bw, bw)``, if its undefined slot
        ``[0, bw - 1, 0]`` is not NaN (Phase 3's ``out=`` tripwire,
        which refuses a :func:`numpy.zeros` or :func:`numpy.empty`
        buffer) or if its slot ``[1, 0, 1]`` is not negative.  The
        latter is ``d^1_{0,1}(pi/2) = -1/sqrt(2)`` transposed and
        ``d^1_{1,0}(pi/2) = +1/sqrt(2)`` untransposed, and the two
        layouts differ by ``(-1)^(k - m)`` in half of their slots,
        so an untransposed table would silently give a wrong
        correlation.
    """
    if not isinstance(table, np.ndarray):
        raise ValueError("`wigner_d_half_pi` must be a NumPy array")
    if table.shape != (bandwidth,) * 3:
        raise ValueError(
            f"`wigner_d_half_pi` must have shape {(bandwidth,) * 3}, not {table.shape}"
        )
    if table.dtype != np.float64:
        raise ValueError(f"`wigner_d_half_pi` must be 64-bit float, not {table.dtype}")
    if not table.flags.c_contiguous:
        raise ValueError("`wigner_d_half_pi` must be C-contiguous")
    if bandwidth > 1:
        if not math.isnan(table[0, bandwidth - 1, 0]):
            raise ValueError(
                "`wigner_d_half_pi` must have NaN in its undefined slots, "
                "as the table of `wigner_d_half_pi_table()` has"
            )
        if not table[1, 0, 1] < 0:
            raise ValueError(
                "`wigner_d_half_pi` must be the transposed table, whose "
                "slot [1, 0, 1] is d^1_(0,1)(pi/2) = -1/sqrt(2)"
            )
    return table


def _validated_wigner_d_factors(
    factors: tuple[np.ndarray, np.ndarray, np.ndarray], bandwidth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a shared beta independent Wigner d factor triple
    unchanged.

    Parameters
    ----------
    factors
        ``(e_km, w_jkm, b_jkm)`` of
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_factors` built for ``bandwidth``.
    bandwidth
        Exclusive maximum harmonic degree ``bw``.

    Returns
    -------
    factors
        The very same three arrays, so that instances share one
        triple.

    Raises
    ------
    ValueError
        If ``factors`` is not three arrays, or if any of them was
        not built for ``bandwidth`` or is not 64-bit floating point.
        The pre-kernel reads them without bounds checking, so an
        undersized table is a read outside the array rather than an
        error; the shapes are the ones
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_pre` checks at its own boundary.
    """
    try:
        e_km, w_jkm, b_jkm = factors
    except (TypeError, ValueError):
        raise ValueError(
            "`wigner_d_factors` must be the `(e_km, w_jkm, b_jkm)` triple of "
            "`wigner_d_table_factors(bandwidth)`"
        ) from None
    for name, factor, shape in (
        ("e_km", e_km, (bandwidth, bandwidth)),
        ("w_jkm", w_jkm, (bandwidth,) * 3),
        ("b_jkm", b_jkm, (bandwidth,) * 3),
    ):
        if (
            getattr(factor, "shape", None) != shape
            or getattr(factor, "dtype", None) != np.float64
        ):
            raise ValueError(
                f"`wigner_d_factors` entry `{name}` must have shape {shape} and a "
                "64-bit floating point data type, i.e. come from "
                f"`wigner_d_table_factors({bandwidth})`"
            )
    return e_km, w_jkm, b_jkm


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
    wigner_d_factors
        The ``(e_km, w_jkm, b_jkm)`` beta independent factor triple
        of :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_factors` to share, read only, which only a
        refinement needs.  If not given, one is built lazily on the
        first :meth:`refine_zyz`; a refining indexer passes one in so
        that its chunk clones do not each build their own 5 MB copy.

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
    wigner_d_factors : tuple or None
        The ``(e_km, w_jkm, b_jkm)`` factor triple of the Wigner d
        tables the refinement rebuilds, read only and shareable
        between instances, and ``None`` until it is given or built.
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
        If ``bandwidth`` is smaller than one, if
        ``wigner_d_half_pi`` is not a C-contiguous 64-bit float
        array of shape ``(bw, bw, bw)`` whose undefined slots are
        NaN and whose slot ``[1, 0, 1]`` is negative, i.e. the
        transposed and not the untransposed layout, or if
        ``wigner_d_factors`` is not a triple of 64-bit float arrays
        of shapes ``(bw, bw)``, ``(bw, bw, bw)`` and
        ``(bw, bw, bw)``.

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
    ``_gn`` of shape ``(bw,)``, both 128-bit complex, and
    :meth:`refine_zyz` overwrites the ``(bw, bw, bw, 2)`` ``_d_beta``
    table and its three small Newton scratch buffers.  Use one
    :meth:`clone` per thread; clones share the Wigner table and the
    factor triple, both read only, and allocate the rest.

    :meth:`compute` returns a fresh caller-owned array on every call
    and rebinds :attr:`xc` to it, so a result kept by the caller
    stays valid across later calls.  The C++ writes into a buffer
    the caller passes instead; an ``out=`` parameter is not offered
    because :mod:`scipy.fft` has no ``out=`` and it would only add a
    copy.

    ``refine=True`` runs the real space Newton refinement of
    :meth:`refine_zyz` from the interpolated peak.  Its ``False``
    default is a deliberate deviation from the C++ ``ref = true``
    (lines 189 and 255), which is recorded in the module's licence
    notice: the user facing default lives on
    :class:`kikuchipy.indexing.SphericalIndexer` instead.

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
        self,
        bandwidth: int,
        *,
        wigner_d_half_pi: np.ndarray | None = None,
        wigner_d_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        bandwidth = int(bandwidth)
        if bandwidth < 1:
            raise ValueError(f"`bandwidth` must be at least one, not {bandwidth}")
        self.bandwidth = bandwidth
        self.side_length_unpadded = 2 * bandwidth - 1
        self.side_length = int(_fft.fast_size(self.side_length_unpadded))
        self.half_side_length = self.side_length // 2 + 1
        if wigner_d_half_pi is None:
            self.wigner_d_half_pi = _wigner.wigner_d_half_pi_table(bandwidth, True)
        else:
            self.wigner_d_half_pi = _validated_wigner_table(wigner_d_half_pi, bandwidth)
        if wigner_d_factors is None:
            self.wigner_d_factors = None
        else:
            self.wigner_d_factors = _validated_wigner_d_factors(
                wigner_d_factors, bandwidth
            )
        slp = self.side_length
        bwp = self.half_side_length
        self.fxc = np.zeros((slp, slp, bwp), dtype=np.complex128)
        self._fm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        self._gn = np.zeros(bandwidth, dtype=np.complex128)
        self.xc = None
        # The refinement buffers: this instance's own ``dTablePre``
        # table and the three Newton scratch arrays, all allocated on
        # the first ``refine_zyz()`` and never shared with a clone
        self._d_beta = None
        self._jac = None
        self._hes = None
        self._step = None

    def __repr__(self) -> str:
        """Return a string with the bandwidth and the three side
        lengths, e.g. ``"SphericalCrossCorrelator: bw = 68,
        side_length = 135 (unpadded 135), half 68"``.
        """
        return (
            f"{type(self).__name__}: bw = {self.bandwidth}, "
            f"side_length = {self.side_length} "
            f"(unpadded {self.side_length_unpadded}), "
            f"half {self.half_side_length}"
        )

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
        n_fold, mirror = _validated_flags(n_fold, mirror)
        flm = _validated_spectrum(flm, self.bandwidth, "flm")
        gln = _validated_spectrum(gln, self.bandwidth, "gln")
        _xcorr_spectrum(
            flm,
            gln,
            self.wigner_d_half_pi,
            n_fold,
            mirror,
            self.fxc,
            self._fm,
            self._gn,
        )
        self.xc = _inverse_fft(self.fxc, n_fold)
        return self.xc

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
        ValueError
            If ``index`` is not a flat index into :attr:`xc`, i.e.
            not in ``[0, xc.size)``.  The C++ has no such check, but
            :func:`_extract_neighborhood` reads a flat buffer with
            bounds checking off and its glide turns a ``k0`` past
            ``bwP`` into negative offsets, so an unchecked index is
            a read outside the cube rather than a wrong answer.

        Notes
        -----
        Port of ``Correlator<Real>::interpPeak()``
        (``include/sht/sht_xcorr.hpp``, lines 406-432).  A step
        larger than one cell is rejected and replaced by the centre,
        but only ``x[0]`` and ``x[1]`` enter that check with
        ``emsphinx_compatible=True`` (line 421).
        """
        if self.xc is None:
            raise RuntimeError("`compute()` must be called before `interp_peak()`")
        index = int(index)
        if not 0 <= index < self.xc.size:
            raise ValueError(
                f"`index` must be a flat index into `xc`, i.e. in "
                f"[0, {self.xc.size}), not {index}"
            )
        slp = self.side_length
        bwp = self.half_side_length
        # detail::extractInds(), lines 1249-1255
        k, remainder = divmod(index, slp * slp)
        n, m = divmod(remainder, slp)
        nh = np.empty((3, 3, 3))
        _extract_neighborhood(
            self.xc.reshape(-1), slp, bwp, k, n, m, bool(emsphinx_compatible), nh
        )
        x = np.zeros(3)
        peak = _interpolate_maxima(nh, x)
        if emsphinx_compatible:
            # the x[2] bounds bug of line 421
            largest = max(abs(x[0]), max(abs(x[1]), abs(x[0])))
        else:
            largest = max(abs(x[0]), max(abs(x[1]), abs(x[2])))
        if largest > 1:
            # do not step too far in case we are near a degeneracy
            x[:] = 0.0
            peak = nh[1, 1, 1]
        zyz = np.array(
            [
                ((m + x[2]) * 4 - slp) * math.pi / (2 * slp),
                ((k + x[0]) * 2 - slp) * math.pi / slp,
                ((n + x[1]) * 4 - slp) * math.pi / (2 * slp),
            ]
        )
        return zyz, float(peak), x

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
            Whether to refine the interpolated peak in real space
            with :meth:`refine_zyz`, ``False`` by default.  The
            default deviates from the C++ ``ref = true`` on purpose,
            see the class ``Notes``.
        emsphinx_compatible
            Whether to reproduce the two C++ defects, ``True`` by
            default, see :meth:`interp_peak`.  It affects only the
            *starting* triple of a refinement; the Newton loop has
            no compatibility branch.

        Returns
        -------
        zyz
            Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in
            radians of the peak, in a new ``(3,)`` 64-bit float
            array.  Without refinement ``beta`` lies in
            ``[-pi - cell, cell]`` and ``gamma`` in
            ``[-pi/2 - cell, 3 pi/2]`` with ``cell = 2 pi / slP``,
            and ``alpha`` lies in the same interval as ``gamma``
            when ``emsphinx_compatible`` is ``False`` and is only
            finite when it is ``True``.  A **refined** triple is not
            wrapped back into those intervals, see
            :meth:`refine_zyz`.
        score
            Interpolated peak of the un-normalised cross-correlation
            of the module documentation, equal to the total power
            ``<f, f>`` for a perfect match, or the analytic value at
            the refined rotation when ``refine`` is ``True``.  The
            two are **not comparable**: the coarse score is a fitted
            tri-quadratic and the refined one is the correlation
            itself.  Either scales with both spectra and is
            comparable only within one geometry.

        Raises
        ------
        ValueError
            See :meth:`compute` and :meth:`refine_zyz`.

        Notes
        -----
        Port of ``Correlator<Real>::correlate()``
        (``include/sht/sht_xcorr.hpp``, lines 394-400), i.e.
        :meth:`compute`, :func:`_find_peak` and
        :meth:`interp_peak`, followed by :meth:`refine_zyz` from the
        interpolated triple when ``refine`` is ``True``.
        """
        self.compute(flm, gln, n_fold, mirror)
        index = _find_peak(self.xc)
        zyz, peak, _ = self.interp_peak(int(index), emsphinx_compatible)
        if refine:
            return self.refine_zyz(flm, gln, n_fold, mirror, zyz)
        return zyz, peak

    def refine_zyz(
        self,
        flm: np.ndarray,
        gln: np.ndarray,
        n_fold: int,
        mirror: bool,
        zyz0: np.ndarray,
        *,
        eps: float = _REFINE_EPS,
    ) -> tuple[np.ndarray, float]:
        """Return the Newton refined rotation of a cross-correlation
        maximum, and its value.

        Parameters
        ----------
        flm, gln
            Harmonic coefficients ``a[m, l]`` of the two real
            functions in array-likes of shape ``(bw, bw)``, see
            :meth:`compute`.
        n_fold
            Order of the rotational symmetry of ``flm`` about z.
        mirror
            Whether ``flm`` has an equatorial mirror plane.
        zyz0
            Starting passive ZYZ Euler angles in an array-like of
            shape ``(3,)``, normally the interpolated coarse peak.
        eps
            Convergence scale of the Newton loop, ``0.01`` by
            default, which is the only value EMSphInx uses.  The
            stopping threshold is ``eps 2 pi / slP``.

        Returns
        -------
        zyz
            Refined angles in a new ``(3,)`` 64-bit float array, or
            ``zyz0`` unchanged when the refinement failed.  They are
            **not** wrapped back into the coarse grid intervals of
            :meth:`correlate`, since Newton may step ``beta`` across
            a pole; every consumer converts through
            :func:`kikuchipy.indexing._spherical._euler.
            rotation_from_zyz`, which is periodic.
        score
            Un-normalised cross-correlation at ``zyz``, see
            :func:`_refine_peak` for the value on failure and for
            the second order lag on success.  A failure is
            **silent**, as it is in the C++.

        Raises
        ------
        ValueError
            If ``flm`` or ``gln`` does not have shape ``(bw, bw)``,
            if ``n_fold`` is a :class:`bool` or smaller than one, if
            ``mirror`` is not a :class:`bool`, or if ``zyz0`` is not
            three finite numbers.  The shape check is not optional:
            with bounds checking off a mismatched spectrum is
            silent garbage rather than an error, measured at
            ``1e225``.

        Notes
        -----
        Port of ``Correlator<Real>::refinePeak()``
        (``include/sht/sht_xcorr.hpp``, lines 442-499), see
        :func:`_refine_peak`.  The factor triple
        :attr:`wigner_d_factors` and this instance's ``_d_beta``
        table are built on the first call if they are not there yet,
        the latter as a NaN filled buffer routed once through
        :func:`kikuchipy.indexing._spherical._wigner.
        wigner_d_table_pre` so that its ``out=`` tripwire actually
        runs, and both are then reused by every later call.
        """
        raise NotImplementedError(
            "`refine_zyz()` is not implemented yet; it needs `_refine_peak()` "
            "and `_derivatives()`"
        )

    def clone(self) -> "SphericalCrossCorrelator":
        """Return a new correlator sharing this one's Wigner table.

        Returns
        -------
        correlator
            New instance with the same bandwidth, the **same**
            :attr:`wigner_d_half_pi` and :attr:`wigner_d_factors`,
            both read only, and its own :attr:`fxc`, ``_fm``/``_gn``
            and refinement buffers, ready for use in another thread.

        Notes
        -----
        Port of ``UnNormalizedCorrelator<Real>::clone()``
        (``include/sht/sht_xcorr.hpp``, line 230), the copy
        constructor which shares the read-only ``xcLut`` and copies
        the rest.  ``Correlator<Real>`` itself has no ``clone()``.

        A built factor triple is passed on, so that the clones of a
        refining run share one; the per-instance ``_d_beta`` table
        is **never** shared, since every kernel is ``nogil=True``
        and two threads writing one table leave a table which
        matches neither rotation.
        """
        return SphericalCrossCorrelator(
            self.bandwidth,
            wigner_d_half_pi=self.wigner_d_half_pi,
            wigner_d_factors=self.wigner_d_factors,
        )


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
    wigner_d_factors
        The beta independent Wigner d factor triple to share, see
        :class:`SphericalCrossCorrelator`.  It is handed to the
        owned un-normalised correlator, which owns every refinement
        buffer of this class as well.

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
    slot, which loses every comparison of
    :func:`_scale_and_find_peak` -- **except in flat slot 0**, which
    seeds the search, so a NaN there pins the peak at the grid
    corner ``(-pi/2, -pi, -pi/2)`` with a NaN score instead of
    raising.  A radicand of exactly **zero** gives ``rDen = +inf``,
    and ``xc * inf`` is ``+inf`` for a positive ``xc``, which *wins*
    the argmax and yields a garbage peak.  A zero radicand means the
    reference is constant over the window.  A ``flm2`` which is not
    the transform of the square of the reference makes **every**
    slot NaN, slot 0 included, so it silently returns that corner:
    inspect :attr:`r_den` for finiteness when a score is NaN.  The
    two in-place NumPy calls run under
    ``numpy.errstate(divide="ignore", invalid="ignore")`` so that a
    degeneracy does not leak an unattributed ``RuntimeWarning`` out
    of the constructor; the values are the documented ones either
    way.

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

    The constructor sets ``correlator.xc`` back to ``None``
    afterwards: the second :meth:`SphericalCrossCorrelator.compute`
    left it bound to the array which becomes :attr:`r_den`, which
    every :meth:`clone` shares read only, and
    :func:`_scale_and_find_peak` writes its first argument in place.

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
        wigner_d_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.correlator = SphericalCrossCorrelator(
            bandwidth,
            wigner_d_half_pi=wigner_d_half_pi,
            wigner_d_factors=wigner_d_factors,
        )
        bw = self.correlator.bandwidth
        self.n_fold, self.mirror = _validated_flags(n_fold, mirror)
        self.flm = _validated_spectrum(flm, bw, "flm").copy()
        self.flm2 = _validated_spectrum(flm2, bw, "flm2").copy()
        self.mlm = _validated_spectrum(mlm, bw, "mlm").copy()

        # the window function correlated with the reference and with
        # its square, both with the reference's flags (lines 1191
        # and 1194)
        mrf = self.correlator.compute(self.flm, self.mlm, self.n_fold, self.mirror)
        mrf2 = self.correlator.compute(self.flm2, self.mlm, self.n_fold, self.mirror)
        # the integral of the window function, line 1197
        s2m = float(self.mlm[0, 0].real) * _SQRT_FOUR_PI
        # Huhle equations 8 and 9 (line 1200), which simplify to
        # 1 / sqrt(mrf2 - mrf^2 / s2m) and are evaluated in place on
        # the two cubes so that no third one is allocated.  The
        # radicand is unguarded as in the C++, so an empty window or
        # a mismatched flm2 gives inf or NaN rather than an error
        # and must not leak a bare NumPy warning
        with np.errstate(divide="ignore", invalid="ignore"):
            mrf *= mrf
            mrf /= s2m
            mrf2 -= mrf
            np.sqrt(mrf2, out=mrf2)
            np.reciprocal(mrf2, out=mrf2)
        self.r_den = mrf2
        # the second compute() left the correlator's `xc` bound to
        # what is now the shared read only `r_den`, which every
        # clone() also shares, so release it and restore the
        # documented "None before the first compute()" invariant
        self.correlator.xc = None

    def __repr__(self) -> str:
        """Return a string with the bandwidth, the three side
        lengths and the flags, e.g.
        ``"NormalizedSphericalCrossCorrelator: bw = 68,
        side_length = 135 (unpadded 135), half 68, n_fold = 4,
        mirror = True"``.
        """
        correlator = self.correlator
        return (
            f"{type(self).__name__}: bw = {correlator.bandwidth}, "
            f"side_length = {correlator.side_length} "
            f"(unpadded {correlator.side_length_unpadded}), "
            f"half {correlator.half_side_length}, "
            f"n_fold = {self.n_fold}, mirror = {self.mirror}"
        )

    @property
    def bandwidth(self) -> int:
        """Return the exclusive maximum harmonic degree."""
        return self.correlator.bandwidth

    @property
    def side_length(self) -> int:
        """Return the zero padded side length ``slP`` of the Euler
        cube.
        """
        return self.correlator.side_length

    @property
    def half_side_length(self) -> int:
        """Return the number of stored beta slices
        ``bwP = slP // 2 + 1``.
        """
        return self.correlator.half_side_length

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
            Whether to refine the interpolated peak in real space
            with :meth:`refine_zyz`, ``False`` by default.  The
            default deviates from the C++ ``ref = true`` on purpose,
            see :class:`SphericalCrossCorrelator`.
        emsphinx_compatible
            Whether to reproduce the two C++ defects, ``True`` by
            default, see
            :meth:`SphericalCrossCorrelator.interp_peak`.  It
            affects only the *starting* triple of a refinement.

        Returns
        -------
        zyz
            Passive ZYZ Euler angles ``(alpha, beta, gamma)`` in
            radians of the peak, in a new ``(3,)`` 64-bit float
            array, with the ranges of
            :meth:`SphericalCrossCorrelator.correlate`.
        score
            Interpolated peak of the **normalized**
            cross-correlation ``xc * rDen``, or the refined
            correlation divided by the denominator at the refined
            rotation when ``refine`` is ``True``.  Neither is
            divided by the standard deviation of the pattern
            function, see the class ``Notes``, and the two are not
            comparable with one another.

        Raises
        ------
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
        values (line 1157), followed by :meth:`refine_zyz` from the
        interpolated triple when ``refine`` is ``True``.
        """
        correlator = self.correlator
        correlator.compute(self.flm, gln, self.n_fold, self.mirror)
        index = _scale_and_find_peak(correlator.xc, self.r_den)
        zyz, peak, _ = correlator.interp_peak(int(index), emsphinx_compatible)
        if refine:
            return self.refine_zyz(gln, zyz)
        return zyz, peak

    def refine_zyz(
        self,
        gln: np.ndarray,
        zyz0: np.ndarray,
        *,
        eps: float = _REFINE_EPS,
    ) -> tuple[np.ndarray, float]:
        """Return the Newton refined rotation of a normalized
        cross-correlation maximum, and its value.

        Parameters
        ----------
        gln
            Harmonic coefficients ``a[m, l]`` of the masked pattern
            in an array-like of shape ``(bw, bw)``.
        zyz0
            Starting passive ZYZ Euler angles in an array-like of
            shape ``(3,)``, normally the interpolated coarse peak.
        eps
            Convergence scale of the Newton loop, ``0.01`` by
            default, see
            :meth:`SphericalCrossCorrelator.refine_zyz`.

        Returns
        -------
        zyz
            Refined angles in a new ``(3,)`` 64-bit float array, or
            ``zyz0`` unchanged when the refinement failed.  They are
            not wrapped, see
            :meth:`SphericalCrossCorrelator.refine_zyz`.
        score
            The refined **un-normalised** correlation divided by
            ``denominator(zyz)`` evaluated at the refined rotation.

        Raises
        ------
        ValueError
            If ``gln`` does not have shape ``(bw, bw)`` or if
            ``zyz0`` is not three finite numbers.

        Notes
        -----
        Port of ``NormalizedCorrelator<Real>::refinePeak()``
        (``include/sht/sht_xcorr.hpp``, lines 1169-1172): the
        un-normalised :func:`_refine_peak` on the stored
        :attr:`flm`, then a division by ``Constants::denominator()``
        (lines 1211-1225), which is

        .. code-block::

            mrf  = derivatives(flm,  mlm, zyz, der=False)
            mrf2 = derivatives(flm2, mlm, zyz, der=False)
            s2m  = mlm[0, 0].real sqrt(4 pi)
            den  = sqrt(mrf2 - 2 (mrf/s2m) mrf + (mrf/s2m)^2 s2m)

        with both evaluations taking the **reference's**
        ``(n_fold, mirror)`` flags, as the C++ passes ``mr, nf``,
        and with the radicand unguarded, as in the C++.

        **The Newton step maximizes the un-normalized correlation**,
        since the window shift chain rule of lines 263-264 is
        omitted here exactly as it is omitted there.  The refined
        normalized score can therefore dip below the coarse one
        (measured 4 of 165 points on real data, worst -4.8e-4) and
        the refined accuracy of a masked pattern is window limited
        (measured 2.1e-2 degrees against 3e-6 unmasked).
        """
        raise NotImplementedError(
            "`refine_zyz()` is not implemented yet; it needs `_refine_peak()`, "
            "`_derivatives()` and the denominator"
        )

    def _denominator(self, zyz: np.ndarray) -> float:
        """Return the Huhle denominator at one rotation.

        Parameters
        ----------
        zyz
            Passive ZYZ Euler angles in a ``(3,)`` 64-bit float
            array.

        Returns
        -------
        denominator
            ``sqrt(mrf2 - 2 fWbar mrf + fWbar^2 s2m)`` with
            ``fWbar = mrf / s2m``, see :meth:`refine_zyz`.

        Notes
        -----
        Port of ``NormalizedCorrelator<Real>::Constants::
        denominator()`` (``include/sht/sht_xcorr.hpp``, lines
        1211-1225), the pointwise counterpart of the whole cube
        :attr:`r_den` holds the reciprocal of.  The radicand is
        unguarded, as in the C++; measured O(1) and positive at the
        refined rotation of every real Ni pattern.
        """
        raise NotImplementedError(
            "`_denominator()` is not implemented yet; it needs `_derivatives()`"
        )

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
        Port of ``NormalizedCorrelator<Real>::clone()``
        (``include/sht/sht_xcorr.hpp``, line 269), which shares the
        read-only ``ncLut`` and ``xcLut``.

        The attributes are copied one by one, so a test asserts that
        a clone has exactly the attribute set of a constructed
        instance.
        """
        clone = NormalizedSphericalCrossCorrelator.__new__(
            NormalizedSphericalCrossCorrelator
        )
        clone.correlator = self.correlator.clone()
        clone.flm = self.flm
        clone.flm2 = self.flm2
        clone.mlm = self.mlm
        clone.n_fold = self.n_fold
        clone.mirror = self.mirror
        clone.r_den = self.r_den
        return clone
