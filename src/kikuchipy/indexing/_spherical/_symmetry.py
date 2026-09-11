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
# ``include/xtal/symmetry.hpp``:
# - ``PointGroup::zRot()`` (lines 1576-1594), the z rotational order
#   of a point group
# - ``PointGroup::zMirror()`` (lines 1471-1519), whether a point group
#   has a mirror plane perpendicular to z
# - ``PointGroup::rotationGroup()`` (lines 1179-1226), which the two
#   above are evaluated through

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
# Modified by Johan Westraadt, 2026-08: translated to Python/NumPy for
# kikuchipy. GPL-2.0-or-later, conveyed under GPL-3.0-or-later
# #####################################################################

"""Symmetry of spherical harmonic coefficients of a master pattern.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from kikuchipy.indexing._spherical import _sht_file

if TYPE_CHECKING:  # pragma: no cover
    from orix.quaternion import Symmetry

# ------------------------- Lookup tables ---------------------------- #

Z_ROTATION_ORDER_AND_MIRROR: dict[str, tuple[int, bool]] = {
    # 1 fold, no equatorial mirror
    "1": (1, False),
    "-1": (1, False),
    "211": (1, False),
    "121": (1, False),
    "m11": (1, False),
    "1m1": (1, False),
    # 1 fold, equatorial mirror
    "11m": (1, True),
    "m": (1, True),
    # 2 fold, no equatorial mirror
    "112": (2, False),
    "2": (2, False),
    "222": (2, False),
    "mm2": (2, False),
    "-4": (2, False),
    "-42m": (2, False),
    "23": (2, False),
    "-43m": (2, False),
    # 2 fold, equatorial mirror
    "2/m": (2, True),
    "mmm": (2, True),
    "m-3": (2, True),
    # 3 fold, no equatorial mirror
    "3": (3, False),
    "-3": (3, False),
    "321": (3, False),
    "312": (3, False),
    "32": (3, False),
    "3m": (3, False),
    "-3m": (3, False),
    # 3 fold, equatorial mirror
    "-6": (3, True),
    "-6m2": (3, True),
    # 4 fold, no equatorial mirror
    "4": (4, False),
    "422": (4, False),
    "4mm": (4, False),
    "432": (4, False),
    # 4 fold, equatorial mirror
    "4/m": (4, True),
    "4/mmm": (4, True),
    "m-3m": (4, True),
    # 6 fold, no equatorial mirror
    "6": (6, False),
    "622": (6, False),
    "6mm": (6, False),
    # 6 fold, equatorial mirror
    "6/m": (6, True),
    "6/mmm": (6, True),
}
"""Z rotational order and equatorial mirror of every point group name
orix can return.

The 38 names of :data:`orix.quaternion.symmetry._groups` plus ``"2"``
and ``"m"``, the names of
:data:`~orix.quaternion.symmetry.C2` and
:data:`~orix.quaternion.symmetry.Cs`, which
:meth:`~orix.crystal_map.Phase.point_group` returns for space groups
3-5 and 6-9 and which are not in ``_groups``.

The values are EMSphInx' ``PointGroup::zRot()`` and
``PointGroup::zMirror()`` (``xtal/symmetry.hpp`` lines 1576-1594 and
1471-1519) evaluated on the orix operators: ``n_fold`` is one plus the
number of proper elements with an axis along z and a non-zero angle,
and ``has_equatorial_mirror`` is whether any improper element is a two
fold about z, i.e. a mirror plane perpendicular to z.
"""

AXIS_ALIAS_SPACE_GROUPS: dict[str, int] = {
    # Monoclinic: orix returns the z-unique "2" and "m" for space
    # groups 3-5 and 6-9, so these axis specific aliases are never
    # returned and get the space group of their z-unique twin
    "112": 3,
    "121": 3,
    "211": 3,
    "11m": 6,
    "1m1": 6,
    "m11": 6,
    # Trigonal: orix returns "32" for space groups 149-155, of which
    # 149 is P312 and 150 is P321
    "312": 149,
    "321": 150,
}
"""Space group of the eight point group names
:func:`orix.quaternion.symmetry.get_point_group` never returns.

The other 32 names of :data:`Z_ROTATION_ORDER_AND_MIRROR` are exactly
the names it does return, so their space groups are looked up in orix
itself (:func:`_space_groups_by_name`) instead of being tabulated
here.
"""

_DIVISOR_LADDER: dict[int, tuple[int, ...]] = {
    1: (),
    2: (1,),
    3: (1,),
    4: (2, 1),
    6: (3, 2, 1),
}
"""Proper divisors of every z rotational order in decreasing order,
i.e. the order in which :func:`validate_flags` tries them.
"""

SYMMETRY_POWER_TOLERANCE: float = 1e-8
"""Largest relative power allowed in coefficients a symmetry says are
systematic zeros.

Shared by the construction guard of
:func:`validate_flags` and the losslessness guard of
:meth:`~kikuchipy.indexing.MasterPatternHarmonics.save`, so that the
two cannot disagree.  A relative power of ``1e-8`` is a relative
amplitude of about ``1e-4``.  Comparisons are ``<=``.
"""


# --------------------------- Private helpers ------------------------ #


def _check_name(name: str) -> None:
    """Raise if a point group name is unknown.

    Parameters
    ----------
    name
        Point group name.

    Raises
    ------
    ValueError
        If ``name`` is not a key of
        :data:`Z_ROTATION_ORDER_AND_MIRROR`.  The message lists the
        known names.
    """
    if name not in Z_ROTATION_ORDER_AND_MIRROR:
        known = ", ".join(sorted(Z_ROTATION_ORDER_AND_MIRROR))
        raise ValueError(
            f"Point group name {name!r} is unknown, it must be one of: {known}"
        )


@cache
def _space_groups_by_name() -> dict[str, tuple[int, ...]]:
    """Return the space groups of every point group name orix
    returns.

    Returns
    -------
    space_groups
        The space group numbers, in increasing order, of every name
        :func:`orix.quaternion.symmetry.get_point_group` returns for
        the 230 space groups, e.g. ``(221, ..., 230)`` for
        ``"m-3m"``.  32 of the 40 names of
        :data:`Z_ROTATION_ORDER_AND_MIRROR` are keys, the other eight
        being the aliases of :data:`AXIS_ALIAS_SPACE_GROUPS`.

    Notes
    -----
    Computed from orix once, on first use.  orix is imported here and
    not at the top of the module, so that importing this module costs
    no more than importing NumPy.  The returned dictionary is cached
    and must not be modified.
    """
    from orix.quaternion.symmetry import get_point_group

    space_groups: dict[str, list[int]] = {}
    for number in range(1, 231):
        space_groups.setdefault(get_point_group(number).name, []).append(number)
    return {name: tuple(numbers) for name, numbers in space_groups.items()}


# ---------------------------- Functions ----------------------------- #


def point_group_flags(name_or_symmetry: str | Symmetry | None) -> tuple[int, bool]:
    """Return the z rotational order and equatorial mirror of a point
    group.

    Parameters
    ----------
    name_or_symmetry
        Point group name, an :class:`orix.quaternion.Symmetry`, or
        ``None``.

    Returns
    -------
    n_fold
        Z rotational order, one of 1, 2, 3, 4 and 6.
    has_equatorial_mirror
        Whether the point group has a mirror plane perpendicular to
        z.

    Raises
    ------
    ValueError
        If the name is not a key of
        :data:`Z_ROTATION_ORDER_AND_MIRROR`.  The message lists the
        known names.

    Notes
    -----
    ``None``, which is what
    :attr:`orix.crystal_map.Phase.point_group` is for a phase without
    a space group, gives the safe ``(1, False)``.
    """
    if name_or_symmetry is None:
        return Z_ROTATION_ORDER_AND_MIRROR["1"]
    name = getattr(name_or_symmetry, "name", name_or_symmetry)
    _check_name(name)
    return Z_ROTATION_ORDER_AND_MIRROR[name]


def space_group_for_point_group(name: str) -> int:
    """Return the lowest space group of a point group name.

    Parameters
    ----------
    name
        Point group name, a key of
        :data:`Z_ROTATION_ORDER_AND_MIRROR`.

    Returns
    -------
    space_group
        Lowest space group number whose
        :func:`orix.quaternion.symmetry.get_point_group` has this
        name, e.g.
        221 for ``"m-3m"`` and 3 for ``"2"``, ``"112"`` and
        ``"121"``.

    Raises
    ------
    ValueError
        If the name is unknown.

    Notes
    -----
    The axis specific aliases which
    :func:`orix.quaternion.symmetry.get_point_group` never returns
    (``"112"``, ``"121"``, ``"211"``, ``"m11"``, ``"1m1"``,
    ``"11m"``) map to the space group of their z-unique twin, i.e. 3
    and 6, and the two trigonal aliases (``"312"``, ``"321"``) to
    their standard settings 149 (P312) and 150 (P321).  They are
    tabulated in :data:`AXIS_ALIAS_SPACE_GROUPS`; every other name is
    looked up in orix.
    """
    _check_name(name)
    space_groups = _space_groups_by_name().get(name)
    if space_groups is None:
        return AXIS_ALIAS_SPACE_GROUPS[name]
    return space_groups[0]


def candidate_space_groups(name: str) -> tuple[int, ...]:
    """Return the space groups of a point group name with distinct
    file compression parameters.

    Parameters
    ----------
    name
        Point group name.

    Returns
    -------
    space_groups
        Lowest space group of every distinct
        ``(z_rot, compression flags)`` pair the name's space groups
        give, in increasing order.  Four names have two:
        ``"3m"`` gives ``(156, 157)``, ``"-3m"`` ``(162, 164)``,
        ``"-42m"`` ``(111, 115)`` and ``"-6m2"`` ``(187, 189)``,
        where the mirror plane contains z through x in one setting
        and through y in the other, so the packed rows are real in
        one and imaginary in the other.  Every other name gives one.

    Raises
    ------
    ValueError
        If the name is unknown.

    See Also
    --------
    kikuchipy.indexing._spherical._sht_file.space_group_z_rotation
    kikuchipy.indexing._spherical._sht_file.space_group_compression_flags

    Notes
    -----
    The first candidate is always the space group of
    :func:`space_group_for_point_group`, so a caller which only has a
    point group can use that one and offer the others, e.g. when the
    packing of the coefficients turns out to be lossy.  The eight
    aliases of :data:`AXIS_ALIAS_SPACE_GROUPS` give their one
    tabulated space group.
    """
    _check_name(name)
    space_groups = _space_groups_by_name().get(name)
    if space_groups is None:
        return (AXIS_ALIAS_SPACE_GROUPS[name],)
    lowest: dict[tuple[int, int], int] = {}
    for number in space_groups:
        pair = (
            _sht_file.space_group_z_rotation(number),
            _sht_file.space_group_compression_flags(number),
        )
        lowest.setdefault(pair, number)
    return tuple(sorted(lowest.values()))


def systematic_zero_power(
    alm: np.ndarray, n_fold: int, has_equatorial_mirror: bool
) -> tuple[float, float]:
    """Return the relative power in the coefficients a symmetry says
    are zero.

    Parameters
    ----------
    alm
        Coefficients ``alm[m, l]`` of shape ``(bw, bw)``.
    n_fold
        Z rotational order.
    has_equatorial_mirror
        Whether there is a mirror plane perpendicular to z.

    Returns
    -------
    rotation_power
        Power in rows with ``m % n_fold != 0`` divided by the total
        power, 0 when ``n_fold`` is 1.
    mirror_power
        Power in entries with odd ``l + m`` divided by the total
        power, 0 when ``has_equatorial_mirror`` is ``False``.

    Notes
    -----
    Power is the :meth:`power_spectrum` weighting, i.e.
    ``|alm[m, l]| ** 2`` with the ``m > 0`` entries counted twice,
    because only the non-negative orders are stored.  The same
    quantity and the same tolerance
    (:data:`SYMMETRY_POWER_TOLERANCE`) guard both construction and
    saving, so that the two cannot disagree.
    """
    power = np.abs(alm) ** 2
    # Only the non-negative orders are stored, so every m > 0 row
    # stands for itself and its m < 0 twin
    power[1:] *= 2
    total = power.sum()
    if total == 0:
        return 0.0, 0.0

    rotation_power = 0.0
    orders = np.arange(alm.shape[0])
    if n_fold > 1:
        rotation_power = float(power[orders % n_fold != 0].sum() / total)

    mirror_power = 0.0
    if has_equatorial_mirror:
        degrees = np.arange(alm.shape[1])
        odd = (orders[:, np.newaxis] + degrees[np.newaxis, :]) % 2 == 1
        mirror_power = float(power[odd].sum() / total)

    return rotation_power, mirror_power


def validate_flags(
    alm: np.ndarray, n_fold: int, has_equatorial_mirror: bool
) -> tuple[int, bool, list[str]]:
    """Return symmetry flags the coefficients actually satisfy.

    Parameters
    ----------
    alm
        Coefficients ``alm[m, l]`` of shape ``(bw, bw)``.
    n_fold
        Z rotational order claimed by the point group.
    has_equatorial_mirror
        Equatorial mirror claimed by the point group.

    Returns
    -------
    n_fold
        The claimed order when the systematic zeros hold within
        :data:`SYMMETRY_POWER_TOLERANCE`, else the *largest divisor*
        of the claimed order which does hold, tried in the order
        ``6 -> 3 -> 2 -> 1``, ``4 -> 2 -> 1``, ``3 -> 1`` and
        ``2 -> 1``.  Downgrading to the largest satisfied divisor
        keeps as much of the plane skipping of the spherical
        correlation as the coefficients allow.
    has_equatorial_mirror
        The claimed mirror, or ``False`` when the odd ``l + m``
        entries carry more than the tolerance.
    warnings
        One message per downgrade, empty when nothing changed.

    Notes
    -----
    EMSphInx has no such guard.  It is needed because orix maps the
    monoclinic space groups 3-15 to *z-unique* point groups
    (``"2"``, ``"m"``, ``"2/m"``) while EMSphInx and the SHT file
    format assume unique axis *b*, so a b-unique monoclinic master
    would otherwise carry a wrong two fold and mirror into the
    correlator with no error at all.
    """
    rotation_power, mirror_power = systematic_zero_power(
        alm, n_fold, has_equatorial_mirror
    )
    warnings: list[str] = []

    validated_n_fold = n_fold
    if n_fold > 1 and rotation_power > SYMMETRY_POWER_TOLERANCE:
        # The ladder ends in 1, whose set of systematic zeros is
        # empty, so a divisor is always found
        for divisor in _DIVISOR_LADDER[n_fold]:
            divisor_power, _ = systematic_zero_power(alm, divisor, False)
            if divisor_power <= SYMMETRY_POWER_TOLERANCE:
                validated_n_fold = divisor
                break
        warnings.append(
            f"The coefficients carry a relative power of {rotation_power:.3e} "
            f"in the orders m % {n_fold} != 0, which the {n_fold} fold "
            "rotation about z of the point group says are zero (tolerance "
            f"{SYMMETRY_POWER_TOLERANCE}); n_fold is downgraded to "
            f"{validated_n_fold}, the largest divisor of {n_fold} the "
            "coefficients satisfy"
        )

    validated_mirror = bool(has_equatorial_mirror)
    if validated_mirror and mirror_power > SYMMETRY_POWER_TOLERANCE:
        validated_mirror = False
        warnings.append(
            f"The coefficients carry a relative power of {mirror_power:.3e} "
            "in the entries with odd l + m, which the equatorial mirror "
            "plane of the point group says are zero (tolerance "
            f"{SYMMETRY_POWER_TOLERANCE}); has_equatorial_mirror is set to "
            "False"
        )

    return validated_n_fold, validated_mirror, warnings
