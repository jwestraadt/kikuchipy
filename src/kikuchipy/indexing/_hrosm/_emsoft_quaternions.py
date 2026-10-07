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
# - The quaternion symmetry operator table SYM_Qsymop and its
#   selection per rotational point group QSym_Init_
#   (EMsoftOOLib/mod_quaternions.f90)
# - The rotational point group table PGrot
#   (EMsoftOOLib/mod_symmetry.f90)
# - The Euler angle to quaternion conversion eq_
#   (EMsoftOOLib/mod_rotations.f90)
# - The term order of the quaternion product quatmult
#   (EMsoftOOLib/mod_quaternions.f90)
# - The symmetry double loop of the disorientation angle
#   getDisorientation_ (EMsoftOOLib/mod_so3.f90)

# #####################################################################
# Copyright (c) 2013-2026, Marc De Graef Research Group/Carnegie Mellon
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

"""Literal EMsoft quaternion tables and float64 arithmetic used by the
EMsoft compatible paths of the HROSM tools.

Every function here reproduces EMsoft's operation order, so that the
compatible paths agree with EMsoft's own output bit for bit where
EMsoft is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from orix.quaternion import Symmetry

# EMsoft's double precision constants of the operator table: sqrt(2)/2,
# sqrt(3)/2 and 1/2, written with EMsoft's digits
SQ22 = 0.7071067811865475244
SQ32 = 0.8660254037844386467
HALF = 0.5

# EMsoft's quaternion symmetry operators (q0, q1, q2, q3), q0 the scalar
# part, in EMsoft's column order; row i is EMsoft's column i + 1. Only
# the 35 crystallographic columns are kept: the icosahedral and the
# quasicrystal columns 36-152 belong to no crystallographic point
# group.
SYM_Qsymop = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # 1: identity
        [0.0, 1.0, 0.0, 0.0],  # 2: 180@[100]
        [0.0, 0.0, 1.0, 0.0],  # 3: 180@[010]
        [0.0, 0.0, 0.0, 1.0],  # 4: 180@[001]
        [SQ22, SQ22, 0.0, 0.0],  # 5: 90@[100]
        [SQ22, 0.0, SQ22, 0.0],  # 6: 90@[010]
        [SQ22, 0.0, 0.0, SQ22],  # 7: 90@[001]
        [SQ22, -SQ22, 0.0, 0.0],  # 8: 270@[100]
        [SQ22, 0.0, -SQ22, 0.0],  # 9: 270@[010]
        [SQ22, 0.0, 0.0, -SQ22],  # 10: 270@[001]
        [0.0, SQ22, SQ22, 0.0],  # 11: 180@[110]
        [0.0, -SQ22, SQ22, 0.0],  # 12: 180@[-110]
        [0.0, 0.0, SQ22, SQ22],  # 13: 180@[011]
        [0.0, 0.0, -SQ22, SQ22],  # 14: 180@[0-11]
        [0.0, SQ22, 0.0, SQ22],  # 15: 180@[101]
        [0.0, -SQ22, 0.0, SQ22],  # 16: 180@[-101]
        [HALF, HALF, HALF, HALF],  # 17: 120@[111]
        [HALF, -HALF, -HALF, -HALF],  # 18: 120@[-1-1-1]
        [HALF, HALF, -HALF, HALF],  # 19: 120@[1-11]
        [HALF, -HALF, HALF, -HALF],  # 20: 120@[-11-1]
        [HALF, -HALF, HALF, HALF],  # 21: 120@[-111]
        [HALF, HALF, -HALF, -HALF],  # 22: 120@[1-1-1]
        [HALF, -HALF, -HALF, HALF],  # 23: 120@[-1-11]
        [HALF, HALF, HALF, -HALF],  # 24: 120@[11-1]
        [SQ32, 0.0, 0.0, HALF],  # 25: 60@[001], hexagonal start
        [HALF, 0.0, 0.0, SQ32],  # 26: 120@[001]
        [0.0, 0.0, 0.0, 1.0],  # 27: 180@[001], duplicate of 4
        [-HALF, 0.0, 0.0, SQ32],  # 28: 240@[001]
        [-SQ32, 0.0, 0.0, HALF],  # 29: 300@[001]
        [0.0, 1.0, 0.0, 0.0],  # 30: 180@[100]
        [0.0, SQ32, HALF, 0.0],  # 31: 180 about a two-fold axis
        [0.0, HALF, SQ32, 0.0],  # 32: 180 about a two-fold axis
        [0.0, 0.0, 1.0, 0.0],  # 33: 180@[010]
        [0.0, -HALF, SQ32, 0.0],  # 34: 180 about a two-fold axis
        [0.0, -SQ32, HALF, 0.0],  # 35: 180 about a two-fold axis
    ],
    dtype=np.float64,
)

# EMsoft's rotational point group number per point group number 1-41
# (the 32 crystallographic point groups in International Tables order,
# then EMsoft's special settings); entry i is point group i + 1
PGROT = (
    1, 1, 3, 1, 3, 6, 3, 6, 9, 3, 9, 12, 9, 6, 12, 16, 16,
    18, 16, 18, 21, 16, 21, 24, 21, 18, 24, 28, 28, 30,
    28, 30, 33, 34, 35, 36, 37, 16, 37, 40, 37,
)  # fmt: skip

# The 1-based columns of SYM_Qsymop that EMsoft's QSym_Init_ copies,
# in its order, per rotational point group number. Only the groups
# built from the 35 crystallographic columns are listed.
_QSYM_INIT_COLUMNS: dict[int, tuple[int, ...]] = {
    1: (1,),
    3: (1, 3),
    6: (1, 2, 3, 4),
    9: (1, 4, 7, 10),
    12: (1, 4, 7, 10, 2, 3, 11, 12),
    16: (1, 26, 28),
    18: (1, 26, 28, 30, 32, 34),
    21: (1, 25, 26, 27, 28, 29),
    24: (1, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 27),
    28: (1, 2, 3, 4, 17, 18, 19, 20, 21, 22, 23, 24),
    30: (
        1, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24,
        2, 3, 4, 11, 12, 13, 14, 15, 16,
    ),
    37: (1, 26, 28, 31, 33, 35),
    39: (1, 26, 28, 31, 33, 35),
    40: (1, 11, 12, 4),
}  # fmt: skip

# The names of EMsoft's point groups 1-32 in International Tables
# order, as orix names them
_EMSOFT_POINT_GROUP_NAMES = (
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m",
    "422", "4mm", "-42m", "4/mmm", "3", "-3", "32", "3m", "-3m", "6",
    "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432",
    "-43m", "m-3m",
)  # fmt: skip


def emsoft_symmetry_operators(pgnum: int) -> np.ndarray:
    """Return EMsoft's quaternion symmetry operators of a point group.

    Parameters
    ----------
    pgnum
        EMsoft point group number, 1-32.

    Returns
    -------
    operators
        Operators as an array of shape (n, 4) of float64, in the order
        EMsoft's ``QSym_Init_`` copies them from its operator table for
        the point group's rotational group.

    Raises
    ------
    ValueError
        If ``pgnum`` is not an EMsoft crystallographic point group
        number.
    """
    if (
        isinstance(pgnum, bool)
        or not isinstance(pgnum, (int, np.integer))
        or not 1 <= pgnum <= len(_EMSOFT_POINT_GROUP_NAMES)
    ):
        raise ValueError(
            f"EMsoft point group number {pgnum!r} must be an integer in "
            f"[1, {len(_EMSOFT_POINT_GROUP_NAMES)}]"
        )
    prot = PGROT[int(pgnum) - 1]
    columns = np.asarray(_QSYM_INIT_COLUMNS[prot], dtype=np.int64)
    return SYM_Qsymop[columns - 1].copy()


def emsoft_point_group_number(point_group: Symmetry) -> int:
    """Return EMsoft's number of an orix point group.

    Parameters
    ----------
    point_group
        orix point group, looked up by its name.

    Returns
    -------
    pgnum
        EMsoft point group number, 1-32.

    Raises
    ------
    ValueError
        If the point group has no EMsoft number, or if the set of
        EMsoft's operators of that number, with each operator taken
        together with its negative, differs from the point group's
        proper subgroup (message contains "point group").
    """
    name = getattr(point_group, "name", None)
    if name not in _EMSOFT_POINT_GROUP_NAMES:
        raise ValueError(f"The point group {name!r} has no EMsoft number")
    pgnum = _EMSOFT_POINT_GROUP_NAMES.index(name) + 1

    # Compare the operators as sets of orientations: q and -q are the
    # same rotation
    emsoft_ops = emsoft_symmetry_operators(pgnum)
    orix_ops = np.asarray(point_group.proper_subgroup.data, dtype=np.float64)
    orix_ops = orix_ops.reshape(-1, 4)
    atol = 1e-12

    def contained(a: np.ndarray, b: np.ndarray) -> bool:
        diff = np.minimum(
            np.abs(a[:, None] - b[None]).max(axis=-1),
            np.abs(a[:, None] + b[None]).max(axis=-1),
        )
        return bool(np.all(diff.min(axis=1) <= atol))

    if emsoft_ops.shape[0] != orix_ops.shape[0] or not (
        contained(emsoft_ops, orix_ops) and contained(orix_ops, emsoft_ops)
    ):
        raise ValueError(
            f"EMsoft's symmetry operators of point group {name!r} differ "
            "from orix' proper subgroup of that point group"
        )
    return pgnum


def emsoft_euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Return quaternions from Bunge Euler angles in EMsoft's operation
    order.

    The half angles of ``Phi``, ``phi1 - phi2`` and ``phi1 + phi2``
    give ``(cos(Phi/2) cos((phi1 + phi2)/2), -sin(Phi/2)
    cos((phi1 - phi2)/2), -sin(Phi/2) sin((phi1 - phi2)/2), -cos(Phi/2)
    sin((phi1 + phi2)/2))``, negated where the scalar part is negative.

    Parameters
    ----------
    euler
        Euler angles (phi1, Phi, phi2) in radians, of shape (..., 3).
        They are computed in float64.

    Returns
    -------
    quaternions
        Quaternions of shape (..., 4) of float64 with a non-negative
        scalar part.
    """
    e = np.asarray(euler, dtype=np.float64)
    c_phi = np.cos(e[..., 1] * 0.5)
    s_phi = np.sin(e[..., 1] * 0.5)
    cm = np.cos((e[..., 0] - e[..., 2]) * 0.5)
    sm = np.sin((e[..., 0] - e[..., 2]) * 0.5)
    cp = np.cos((e[..., 0] + e[..., 2]) * 0.5)
    sp = np.sin((e[..., 0] + e[..., 2]) * 0.5)
    q = np.stack([c_phi * cp, -s_phi * cm, -s_phi * sm, -c_phi * sp], axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def emsoft_quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a * b`` in EMsoft's term order.

    Component 0 is ``(a0 b0 - a1 b1) - (a2 b2 + a3 b3)``, 1 ``(a0 b1 +
    a1 b0) + (a2 b3 - a3 b2)``, 2 ``(a0 b2 + a2 b0) + (a3 b1 - a1 b3)``
    and 3 ``(a0 b3 + a3 b0) + (a1 b2 - a2 b1)``.

    Parameters
    ----------
    a, b
        Quaternions of shape (..., 4) of float64, broadcast against
        each other.

    Returns
    -------
    product
        Quaternions of the broadcast shape (..., 4) of float64.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            (a0 * b0 - a1 * b1) - (a2 * b2 + a3 * b3),
            (a0 * b1 + a1 * b0) + (a2 * b3 - a3 * b2),
            (a0 * b2 + a2 * b0) + (a3 * b1 - a1 * b3),
            (a0 * b3 + a3 * b0) + (a1 * b2 - a2 * b1),
        ],
        axis=-1,
    )


def emsoft_disorientation_angle(
    a: np.ndarray, b: np.ndarray, operators: np.ndarray
) -> np.ndarray:
    """Return EMsoft's disorientation angle between pairs of
    quaternions.

    For every operator ``j``, then every operator ``k``, in the given
    order, the angle ``2 arccos(x)`` with ``x = |scalar((S_j a) *
    conj(S_k b))|`` is computed without clipping ``x``; the running
    minimum starts at 1000 and is replaced only by a smaller angle, so
    an angle of NaN (``x`` above 1 by rounding) is never selected.

    Parameters
    ----------
    a, b
        Quaternions of shape (n, 4) of float64.
    operators
        Symmetry operators of shape (m, 4) of float64 in EMsoft's
        order, see :func:`emsoft_symmetry_operators`.

    Returns
    -------
    angle
        Disorientation angles in radians of shape (n,) of float64.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    operators = np.asarray(operators, dtype=np.float64).reshape(-1, 4)
    n = a.shape[0]
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    angle_min = np.full(n, 1000.0)
    # Pairs in chunks, so that the operator products stay in cache
    chunk = 8192
    with np.errstate(invalid="ignore"):
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            # The operators applied to both quaternions once, the
            # second one conjugated, as contiguous components of shape
            # (m, 4, chunk); a sign flip of either factor changes no
            # absolute scalar part
            sa = emsoft_quaternion_multiply(operators[:, None], a[None, start:stop])
            sa = np.ascontiguousarray(np.moveaxis(sa, -1, 1))
            sb = emsoft_quaternion_multiply(operators[:, None], b[None, start:stop])
            sb = np.ascontiguousarray(np.moveaxis(sb * conj, -1, 1))
            ac = angle_min[start:stop]
            for m0, m1, m2, m3 in sa:
                for c0, c1, c2, c3 in sb:
                    # The scalar part of the product in EMsoft's term
                    # order
                    x = np.abs((m0 * c0 - m1 * c1) - (m2 * c2 + m3 * c3))
                    angle = 2.0 * np.arccos(x)
                    # NaN compares False, so it is never selected
                    ac = np.where(angle < ac, angle, ac)
            angle_min[start:stop] = ac
    return angle_min
