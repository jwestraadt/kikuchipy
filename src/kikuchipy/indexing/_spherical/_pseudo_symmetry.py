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
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517):
# - ``programs/master_xcorr.cpp`` (lines 41-281), the pseudo-symmetry
#   prediction program, as :func:`find_pseudo_symmetry_operators` and
#   the private helpers :func:`_local_maxima`,
#   :func:`_dedup_keep_brighter` and :func:`_exclude_true_ops`.
#   Deliberate deviations, each recorded in the docstrings: the local
#   maxima scan runs over the true ``(bwP, slP, slP)`` cube returned
#   by the correlator rather than flat-indexing with ``sl = 2 bw - 1``
#   (the C++ mis-indexes its own cube whenever
#   ``fastSize(2 bw - 1) != 2 bw - 1``); the hard CLI bandwidth clamp
#   ``[53, 313]`` is replaced by the module's ``[16, 512]`` rule; the
#   four hard-coded CWD writers (``pseudo_sym.h5``, ``.xdmf``,
#   ``true.svg``, ``pseudo.svg``) are not ported, the correlation
#   volume being returned as an array instead; and
#   ``exclude_symmetry=True`` applies the C++'s SVG-stage
#   true-symmetry filters (lines 232-261) to the *returned* list,
#   which the C++ stdout list never filters
# - ``include/idx/master.hpp`` (lines 220-233),
#   ``MasterData::addPseudoSym(std::string)``, as
#   :func:`read_emsphinx_psym_file`: the ``qu``-only rule, the exact
#   identity row skip and the count validation
# - ``include/xtal/vendor/emsoft.hpp`` (lines 48-145), the EMsoft
#   angle file grammar (``qu`` subset only), as the psymfile codec's
#   tokenising: whitespace- and/or comma-separated numbers,
#   scalar-first ``w x y z``
#
# The psymfile quaternion convention is the one derived and frozen in
# the Phase 8 spec (D2): a psymfile row is the CONJUGATE of the
# public (NCC-convention) operator, so the conjugation lives in
# exactly two places, this codec and
# :func:`find_pseudo_symmetry_operators`'s conversion of refined
# peaks.

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
# Changed by Johan Westraadt, 2026-09: translated to Python/NumPy for
# kikuchipy, with the deviations stated above. GPL-2.0-or-later,
# conveyed under GPL-3.0-or-later
# #####################################################################

"""Pseudo-symmetry operator prediction and the EMSphInx psymfile
codec :cite:`lenthe2019pseudo`.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Function to EMSphInx source mapping:

- :func:`find_pseudo_symmetry_operators` --
  ``programs/master_xcorr.cpp`` lines 41-281 (the ``MasterXcorr``
  program)
- :func:`read_emsphinx_psym_file` -- ``include/idx/master.hpp`` lines
  220-233 with the angle-file grammar of
  ``include/xtal/vendor/emsoft.hpp`` lines 48-145 (``qu`` subset)
- :func:`write_emsphinx_psym_file` -- the inverse of the reader; the
  C++ has no writer (``MasterXcorr`` prints to stdout only), so the
  number format here is kikuchipy's frozen one: full 64-bit float
  ``repr`` precision, single-space separated, one ``w x y z`` row per
  line

**The operator convention** (frozen): public operators are in the NCC
convention of :meth:`kikuchipy.signals.EBSD.refine_orientation`, i.e.
a variant of a map rotation ``rot`` is ``op * rot`` in orix
left-composition.  A psymfile row is the **conjugate** ``~op``, which
is what EMSphInx' ``q0 * q`` indexing loop consumes to realise the
same variant, and :func:`read_emsphinx_psym_file` conjugates back on
read.  Round trip ``read(write(ops)) == ops`` exactly.

References
----------
:cite:`lenthe2019pseudo`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from orix.quaternion import Rotation

from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)

# How far below the requested cutoff the local maxima scan searches,
# to allow for off-grid peaks (``master_xcorr.cpp`` line 61)
_CANDIDATE_FACTOR = 0.95

# Grid maxima closer than this are merged, keeping the brighter.  The
# metric is the quaternion-dot half-angle
# ``acos(min(1, |q_i . q_j|))`` in degrees, i.e. about 4 degrees of
# misorientation (``master_xcorr.cpp`` lines 148-166)
_DEDUP_HALF_ANGLE_DEG = 2.0

# ``exclude_symmetry`` drops operators whose absolute quaternion dot
# with a proper rotation of the first phase's point group exceeds
# this, about 2.56 degrees of quaternion-dot half-angle
# (``master_xcorr.cpp`` lines 237, 246, 257)
_EXCLUDE_COSINE = 0.999

# The identity quaternion row the reader skips on exact float
# equality (``master.hpp`` lines 227-231)
_IDENTITY_ROW = (1.0, 0.0, 0.0, 0.0)

# The rotation type tokens the EMsoft angle file grammar knows; only
# ``qu`` is accepted for pseudo-symmetry (``master.hpp`` line 225)
_ANGLE_FILE_TYPES = ("eu", "om", "ax", "ro", "qu", "ho", "cu")


@dataclass(frozen=True)
class PseudoSymmetryOperators:
    """Pseudo-symmetry operators detected on a master pattern's
    rotational cross-correlation, ordered by descending intensity.

    Returned by
    :func:`~kikuchipy.indexing.find_pseudo_symmetry_operators`.

    Attributes
    ----------
    operators
        Operators as an :class:`~orix.quaternion.Rotation`, in the
        NCC convention of
        :meth:`kikuchipy.signals.EBSD.refine_orientation`: a variant
        of a map rotation ``rot`` is ``op * rot``.  Directly usable
        as ``pseudo_symmetry_ops`` of both the NCC refinement methods
        and the spherical indexing methods.  Ordered by descending
        intensity.
    intensities
        Cross-correlation intensity of every operator in a 64-bit
        float array of the operators' shape, normalized by the
        refined global maximum, so the identity peak of an
        autocorrelation is one.
    volume
        The full ``(bwP, slP, slP)`` 64-bit float cross-correlation
        cube when the operators were found with ``keep_volume=True``,
        and ``None`` otherwise.
    bandwidth
        Bandwidth the correlation was computed at.
    """

    operators: Rotation
    intensities: np.ndarray
    volume: np.ndarray | None
    bandwidth: int

    def save(self, filename: str | Path) -> None:
        """Write the operators to an EMSphInx psymfile.

        Parameters
        ----------
        filename
            Path to write to.

        Raises
        ------
        ValueError
            If the operator set is empty.

        Notes
        -----
        Equivalent to
        :func:`~kikuchipy.indexing.write_emsphinx_psym_file` on
        :attr:`operators`; the written rows are the conjugated
        quaternions EMSphInx' indexing loop consumes.
        """
        write_emsphinx_psym_file(filename, self.operators)


def read_emsphinx_psym_file(filename: str | Path) -> Rotation:
    """Read pseudo-symmetry operators from an EMSphInx psymfile.

    Parameters
    ----------
    filename
        Path of an EMsoft quaternion angle file: the token ``qu``, a
        count, then ``count`` scalar-first ``w x y z`` rows separated
        by whitespace and/or commas.

    Returns
    -------
    operators
        Operators in the NCC convention (the file rows conjugated),
        directly usable as ``pseudo_symmetry_ops``.  Rows exactly
        equal to the identity ``(1, 0, 0, 0)`` are silently skipped,
        by exact float equality as in EMSphInx, so the result may be
        empty.

    Raises
    ------
    ValueError
        If the type token is not ``qu`` (EMSphInx: "only quaternion
        angle files are supported"), or if the file holds fewer or
        more numbers than the count promises.

    Notes
    -----
    Port of ``MasterData::addPseudoSym(std::string)``
    (``include/idx/master.hpp``, lines 220-233) over the EMsoft angle
    file grammar (``include/xtal/vendor/emsoft.hpp``, lines 48-145).
    Near-identity rows are kept, as in EMSphInx; they merely
    re-converge to the global peak when used.  One recorded
    deviation: non-unit rows are silently unitized by the orix
    :class:`~orix.quaternion.Rotation` constructor, where EMSphInx
    uses the raw values.
    """
    raise NotImplementedError


def write_emsphinx_psym_file(filename: str | Path, operators: Rotation) -> None:
    """Write pseudo-symmetry operators to an EMSphInx psymfile.

    Parameters
    ----------
    filename
        Path to write to.
    operators
        Operators in the NCC convention, e.g. the ``operators`` of
        :class:`~kikuchipy.indexing.PseudoSymmetryOperators`.
        Flattened before writing.

    Raises
    ------
    ValueError
        If ``operators`` is empty.

    Notes
    -----
    The file is an EMsoft quaternion angle file: the ``qu`` token,
    the count, then one ``w x y z`` row per operator.  Each row is
    the **conjugate** ``(~op).data``, which is what EMSphInx'
    ``q0 * q`` indexing loop consumes to realise the variant
    kikuchipy scores as ``op * rot``.  The number format is frozen:
    full 64-bit float ``repr`` precision, single-space separated, one
    row per line, ``\\n`` line ends.  There is no C++ writer to
    imitate; ``MasterXcorr`` prints to stdout only, at a lossier
    precision.

    Round trip: ``read_emsphinx_psym_file`` of the written file
    reproduces ``operators`` exactly (the identity-skip rule of the
    reader cannot trigger, since an exact-identity operator would
    have been refused here as pure waste is not: it is written as
    given).
    """
    raise NotImplementedError


def find_pseudo_symmetry_operators(
    harmonics: MasterPatternHarmonics,
    second_harmonics: MasterPatternHarmonics | None = None,
    *,
    bandwidth: int = 88,
    cutoff: float = 0.5,
    exclude_symmetry: bool = True,
    keep_volume: bool = False,
    emsphinx_compatible: bool = True,
) -> PseudoSymmetryOperators:
    """Predict pseudo-symmetry operators of a master pattern from
    its rotational cross-correlation :cite:`lenthe2019pseudo`.

    This is the kikuchipy equivalent of EMSphInx' ``MasterXcorr``
    program.

    Parameters
    ----------
    harmonics
        Harmonics of the master pattern.  A copy has its DC term
        removed, and is resized when its bandwidth differs from
        ``bandwidth``; the caller's object is never modified.
    second_harmonics
        Harmonics of a second master pattern for the two-phase
        (cross-master) mode, in which the peaks are misorientations
        relating the two phases: an operator means the phase-1
        equivalent of a phase-2 orientation is ``op * rot_2``.  If
        not given (default), the autocorrelation of ``harmonics`` is
        searched.  Only the first master's symmetry flags fold the
        correlation grid, as in the C++.
    bandwidth
        Bandwidth to correlate at, between 16 and 512.  Default is
        88, the smallest of the C++'s recommended values
        (``2 * 88 - 1 = 175`` is a product of small primes).  The
        C++'s hard CLI clamp ``[53, 313]`` is not reproduced.
    cutoff
        Relative intensity in ``[0, 1]`` below which peaks are
        dropped, 0.5 by default.  Intensities are normalized by the
        refined global maximum.
    exclude_symmetry
        Whether to drop operators equivalent to a true proper
        rotation of the first phase's point group, and to dedup the
        kept operators, both at an absolute quaternion dot of 0.999
        (about 2.56 degrees of quaternion-dot half-angle).  Default
        is ``True``.  ``False`` reproduces the raw C++ stdout list,
        true-symmetry operators and surviving refined duplicates
        included, which is the binary parity setting.
    keep_volume
        Whether to return the full correlation cube, ``False`` by
        default.
    emsphinx_compatible
        Whether the peak neighbourhood extraction reproduces the C++
        glide defects, ``True`` by default, see
        :class:`~kikuchipy.indexing.SphericalIndexer`.

    Returns
    -------
    result
        The detected operators with their normalized intensities,
        descending, see
        :class:`~kikuchipy.indexing.PseudoSymmetryOperators`.

    Raises
    ------
    ValueError
        If ``bandwidth`` is outside ``[16, 512]``, or if ``cutoff``
        is outside ``[0, 1]``.

    Notes
    -----
    Port of ``programs/master_xcorr.cpp`` (lines 41-281): DC removal
    on copies of both spectra, one full un-normalised correlation
    cube folded by the first master's symmetry flags, the reference
    maximum refined from the near-identity grid cell
    (autocorrelation) or from the interpolated coarse argmax
    (two-phase mode), a 26-neighbour local maxima scan at
    ``>= v_max * cutoff * 0.95``, a 2 degree quaternion-dot
    half-angle dedup of grid maxima keeping the brighter, Newton
    refinement of each survivor from its un-interpolated grid Euler,
    normalization by the reference maximum, and the ``>= cutoff``
    keep gate.  After refinement there is no second dedup, so two
    grid maxima refining into the same true peak both survive when
    ``exclude_symmetry`` is ``False``.

    The returned operators are
    ``rotation_from_zyz(refined peak)`` per peak -- the NCC
    convention, see the module documentation.

    An autocorrelation is inversion symmetric, so its operator set
    is closed under inversion: the identity peak and inverse pairs
    are expected, and ``exclude_symmetry=True`` on a phase with no
    genuine pseudo-symmetry (e.g. Ni, m-3m) returns an empty set.
    """
    raise NotImplementedError


def _local_maxima(
    xc: np.ndarray,
    threshold: float,
    emsphinx_compatible: bool,
) -> np.ndarray:
    """Return the flat indices of the 26-neighbour local maxima of a
    correlation cube which reach a threshold.

    Parameters
    ----------
    xc
        The ``(bwP, slP, slP)`` 64-bit float cross-correlation cube
        of
        :meth:`kikuchipy.indexing._spherical._xcorr.
        SphericalCrossCorrelator.compute`.
    threshold
        Smallest value a voxel must reach, i.e.
        ``v_max * cutoff * 0.95``.
    emsphinx_compatible
        Whether the periodic 3 x 3 x 3 neighbourhood uses the C++
        glide defects of
        :func:`kikuchipy.indexing._spherical._xcorr.
        _extract_neighborhood`.

    Returns
    -------
    indices
        Flat indices into ``xc`` of every voxel which reaches
        ``threshold`` and is ``>=`` all 26 neighbours of its
        periodic neighbourhood (ties kept, as in the C++).

    Notes
    -----
    Port of the scan of ``master_xcorr.cpp`` lines 94-140, with the
    recorded deviation that it runs over the true cube shape rather
    than flat-indexing with ``sl = 2 bw - 1``.
    """
    raise NotImplementedError


def _dedup_keep_brighter(
    quaternions: np.ndarray,
    intensities: np.ndarray,
) -> np.ndarray:
    """Return a boolean keep mask merging near-duplicate grid maxima.

    Parameters
    ----------
    quaternions
        ``(n, 4)`` unit quaternions of the grid maxima, in the order
        they are considered.
    intensities
        ``(n,)`` intensities of the maxima.

    Returns
    -------
    keep
        ``(n,)`` boolean mask.  A maximum whose quaternion-dot
        half-angle ``acos(min(1, |q_i . q_j|))`` to a nearer kept
        maximum is below 2 degrees is merged with it, only the
        brighter of the two surviving.

    Notes
    -----
    Port of ``master_xcorr.cpp`` lines 148-166 ("this is extremely
    arbitrary"), applied to grid maxima **before** refinement only.
    """
    raise NotImplementedError


def _exclude_true_ops(
    operators: Rotation,
    intensities: np.ndarray,
    point_group,
) -> np.ndarray:
    """Return a boolean keep mask dropping true-symmetry operators
    and deduplicating the kept ones.

    Parameters
    ----------
    operators
        Candidate operators, descending in intensity.
    intensities
        Their intensities.
    point_group
        :class:`orix.quaternion.symmetry.Symmetry` of the first
        phase, whose **proper** rotations are the exclusion set.

    Returns
    -------
    keep
        Boolean mask: an operator whose absolute quaternion dot with
        any proper rotation of ``point_group`` exceeds 0.999 is
        dropped, and a kept operator within the same threshold of an
        earlier kept one is dropped as a duplicate.

    Notes
    -----
    Port of the SVG-stage filters of ``master_xcorr.cpp`` lines
    232-261, applied to the *returned* list under
    ``exclude_symmetry=True`` -- a recorded deviation, since the C++
    runs them only for ``pseudo.svg`` and never filters its stdout.
    """
    raise NotImplementedError
