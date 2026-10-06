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

"""Tests of EMsoft's quaternion ingredients and of the kernel average
misorientation (KAM) map in the correct and the EMsoft compatible
modes.

The oracle of the compatible mode is a literal transcription, written
in this module from EMsoft's Fortran source, of ``getKAMMap``
(``mod_DIsupport.f90``), ``getDisorientation_`` without ``fix1``
(``mod_so3.f90``), ``eq_`` (``mod_rotations.f90``) and ``quatmult``
(``mod_quaternions.f90``). The oracles of the correct mode are
analytic gradient fields and orix' ``Orientation.angle_with``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Orientation, Rotation, Symmetry
from orix.quaternion.symmetry import O, Oh
import pytest

import kikuchipy as kp
from kikuchipy.indexing import kernel_average_misorientation_map
from kikuchipy.indexing._hrosm import _emsoft_quaternions
from kikuchipy.indexing._hrosm._emsoft_quaternions import (
    emsoft_euler_to_quaternion,
    emsoft_point_group_number,
    emsoft_quaternion_multiply,
    emsoft_symmetry_operators,
)
from kikuchipy.indexing._hrosm._kam import _dot_to_angle

# Radians to degrees in float64, EMsoft's rtod
RTOD = 180 / np.pi

# EMsoft's double precision constants of its operator table
SQ22 = 0.7071067811865475244
SQ32 = 0.8660254037844386467
HALF = 0.5

# EMsoft's quaternion symmetry operators of the cubic groups, 1-based
# column number of SYM_Qsymop -> (q0, q1, q2, q3), transcribed from
# mod_quaternions.f90
EMSOFT_CUBIC_COLUMNS = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0, 0.0),
    3: (0.0, 0.0, 1.0, 0.0),
    4: (0.0, 0.0, 0.0, 1.0),
    5: (SQ22, SQ22, 0.0, 0.0),
    6: (SQ22, 0.0, SQ22, 0.0),
    7: (SQ22, 0.0, 0.0, SQ22),
    8: (SQ22, -SQ22, 0.0, 0.0),
    9: (SQ22, 0.0, -SQ22, 0.0),
    10: (SQ22, 0.0, 0.0, -SQ22),
    11: (0.0, SQ22, SQ22, 0.0),
    12: (0.0, -SQ22, SQ22, 0.0),
    13: (0.0, 0.0, SQ22, SQ22),
    14: (0.0, 0.0, -SQ22, SQ22),
    15: (0.0, SQ22, 0.0, SQ22),
    16: (0.0, -SQ22, 0.0, SQ22),
    17: (HALF, HALF, HALF, HALF),
    18: (HALF, -HALF, -HALF, -HALF),
    19: (HALF, HALF, -HALF, HALF),
    20: (HALF, -HALF, HALF, -HALF),
    21: (HALF, -HALF, HALF, HALF),
    22: (HALF, HALF, -HALF, -HALF),
    23: (HALF, -HALF, -HALF, HALF),
    24: (HALF, HALF, HALF, -HALF),
}

# The order in which EMsoft's QSym_Init_ copies the columns for the
# rotational group 432 (point groups 30 and 32)
EMSOFT_432_COLUMN_ORDER = (
    (1,) + tuple(range(5, 11)) + tuple(range(17, 25)) + (2, 3, 4)
    + tuple(range(11, 17))
)  # fmt: skip

EMSOFT_432_OPERATORS = np.array(
    [EMSOFT_CUBIC_COLUMNS[i] for i in EMSOFT_432_COLUMN_ORDER], dtype=np.float64
)

# EMsoft's running minimum of the disorientation angle starts here
EMSOFT_START_ANGLE = 1000.0

# Shapes of the compatible KAM maps compared bit for bit with the loop
# transcription
COMPAT_SHAPES = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (4, 7), (7, 9)]

# Relative tolerance of compatible KAM values of fields whose pair
# angles are all equal in float64: the compatible mode reads float32
# Euler angles, which moves each pair angle by up to a few 1e-6 deg
# (largest relative deviation measured with phi = 0.3 deg: 4.2e-6 on
# the (4, 5) and (6, 3) fields from (10, 20, 30) deg, 1.5e-5 on the
# (4, 5) field from (40, 50, 60) deg; about twice that)
CONSTANT_PAIR_RTOL = 3e-5

# Largest absolute difference per component between EMsoft's and
# orix' quaternion products of unit quaternions (measured on 1000
# random pairs: two float64 machine epsilons)
QUATMULT_ORIX_ATOL = 2 * np.finfo(np.float64).eps


# Test-local oracles


def emsoft_eq(euler: np.ndarray) -> np.ndarray:
    """Return EMsoft's ``eq_`` of Euler angles (..., 3) in float64."""
    e = np.asarray(euler, dtype=np.float64)
    c_phi = np.cos(e[..., 1] * 0.5)
    s_phi = np.sin(e[..., 1] * 0.5)
    cm = np.cos((e[..., 0] - e[..., 2]) * 0.5)
    sm = np.sin((e[..., 0] - e[..., 2]) * 0.5)
    cp = np.cos((e[..., 0] + e[..., 2]) * 0.5)
    sp = np.sin((e[..., 0] + e[..., 2]) * 0.5)
    q = np.stack([c_phi * cp, -s_phi * cm, -s_phi * sm, -c_phi * sp], axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def emsoft_quatmult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return EMsoft's ``quatmult`` of quaternions (..., 4) in its term
    order (``epsijk = +1``).
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


def emsoft_quat_pos(q: np.ndarray) -> np.ndarray:
    """Return quaternions negated where the scalar part is negative."""
    return np.where(q[..., :1] < 0, -q, q)


def emsoft_dis(euler_a: np.ndarray, euler_b: np.ndarray, clip: bool = False) -> float:
    """Return EMsoft's disorientation angle in radians between two
    float32 Euler triplets: ``getDisorientation_`` without ``fix1``,
    the unclipped ``2 acos`` and the running minimum from 1000.

    ``clip=True`` clips the cosines to [0, 1] instead, which EMsoft
    does not do; used only to show that a fixture is sensitive to it.
    """
    mu = emsoft_quat_pos(emsoft_eq(np.float64(euler_a)))
    qu = emsoft_quat_pos(emsoft_eq(np.float64(euler_b)))
    ops = EMSOFT_432_OPERATORS
    mus = emsoft_quat_pos(emsoft_quatmult(ops, mu))
    qus = emsoft_quat_pos(emsoft_quatmult(ops, qu))
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    lo, hi = (0.0, 1.0) if clip else (-np.inf, np.inf)
    ac = EMSOFT_START_ANGLE
    with np.errstate(invalid="ignore"):
        for j in range(ops.shape[0]):
            p = emsoft_quat_pos(emsoft_quatmult(mus[j], qus * conj))
            angles = 2.0 * np.arccos(np.clip(p[:, 0], lo, hi))
            p2 = emsoft_quat_pos(emsoft_quatmult(qus, mus[j] * conj))
            angles2 = 2.0 * np.arccos(np.clip(p2[:, 0], lo, hi))
            for k in range(ops.shape[0]):
                if angles[k] < ac:
                    ac = angles[k]
                if angles2[k] < ac:
                    ac = angles2[k]
    return float(ac)


def emsoft_kam_loop(
    euler32: np.ndarray, clip: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return EMsoft's KAM map of float32 Euler angles (H, W, 3) in
    radians, as the float32 radians ``sngl(localkam)`` and the float32
    degrees ``float32(float64(kam) * rtod)``, both of shape (H, W).
    ``clip`` is passed on to :func:`emsoft_dis`.

    A literal transcription of ``getKAMMap`` with 1-based indices.
    """
    H, W = euler32.shape[:2]
    n = H * W
    e = np.concatenate([np.zeros((1, 3), np.float32), euler32.reshape(n, 3)])
    zero = np.zeros(3, dtype=np.float32)
    localkam = np.zeros(n + 1, dtype=np.float64)
    lstore = [zero] * (W + 1)
    pstore = [zero] * (W + 1)
    cp = zero
    for t in range(1, n + 1):
        ii = t % W
        if ii == 0:
            ii = W
        jj = t // W + 1
        if ii == 1 and jj > 1:
            lstore = list(pstore)
        if ii == 1:
            cp = e[t]
            pstore[ii] = cp
        else:
            lp = cp
            cp = e[t]
            pstore[ii] = cp
            d = emsoft_dis(lp, cp, clip)
            localkam[t - 1] = localkam[t - 1] + d
            localkam[t] = localkam[t] + d
        if jj > 1:
            d = emsoft_dis(lstore[ii], cp, clip)
            localkam[t - W + 1] = localkam[t - W + 1] + d
            localkam[t] = localkam[t] + d

    def third(x):
        return (x * 4.0) / 3.0

    localkam = localkam * 0.25
    localkam[2:W] = third(localkam[2:W])
    localkam[n - W + 2 : n] = third(localkam[n - W + 2 : n])
    for jj in range(1, H - 1):
        localkam[W * jj + 1] = third(localkam[W * jj + 1])
    for jj in range(2, H):
        localkam[W * jj] = third(localkam[W * jj])
    localkam[1] = localkam[1] * 4.0
    localkam[W] = localkam[W] * 2.0
    localkam[n] = localkam[n] * 2.0
    localkam[n - W + 1] = third(localkam[n - W + 1])

    kam_rad = localkam[1:].astype(np.float32).reshape(H, W)
    kam_deg = (kam_rad.astype(np.float64) * RTOD).astype(np.float32)
    return kam_rad, kam_deg


def random_euler32(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return ``n`` random float32 Euler triplets, ``phi1`` and ``phi2``
    uniform in [0, 2 pi) and ``Phi`` uniform in [0, pi].
    """
    phi1 = rng.uniform(0, 2 * np.pi, n)
    Phi = rng.uniform(0, np.pi, n)
    phi2 = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([phi1, Phi, phi2]).astype(np.float32)


def crystal_map(
    rotations: Rotation,
    shape: tuple[int, int],
    phase_id: np.ndarray | None = None,
    is_in_data: np.ndarray | None = None,
) -> CrystalMap:
    """Return a nickel crystal map (m-3m) of step 1 with the rotations
    in raster order, and a second nickel phase "ni2" for ID 1.
    """
    coords, n = create_coordinate_arrays(shape, step_sizes=(1, 1))
    if phase_id is None:
        phase_id = np.zeros(n, dtype=np.int32)
    phase_id = np.asarray(phase_id, dtype=np.int32).ravel()
    phases = [Phase("ni", point_group="m-3m")]
    ids = [0]
    if np.any(phase_id == 1):
        phases.append(Phase("ni2", point_group="m-3m"))
        ids.append(1)
    if is_in_data is None:
        is_in_data = np.ones(n, dtype=bool)
    return CrystalMap(
        rotations=rotations,
        phase_id=phase_id,
        x=coords.get("x"),
        y=coords.get("y"),
        phase_list=PhaseList(phases=phases, ids=ids),
        is_in_data=is_in_data,
        scan_unit="um",
    )


def crystal_map_from_euler32(euler32: np.ndarray) -> CrystalMap:
    """Return a crystal map of float32 Euler angles (H, W, 3), built
    with ``Rotation.from_euler`` of the angles promoted to float64.
    """
    H, W = euler32.shape[:2]
    rot = Rotation.from_euler(euler32.reshape(-1, 3).astype(np.float64))
    return crystal_map(rot, (H, W))


def euler32_of_map(xmap: CrystalMap, shape: tuple[int, int]) -> np.ndarray:
    """Return the float32 Euler angles (H, W, 3) of the first rotation
    per map point, EMsoft compatible input route.
    """
    rot = xmap.rotations
    if rot.ndim > 1:
        rot = rot[:, 0]
    return rot.to_euler().astype(np.float32).reshape(shape + (3,))


def assert_euler_round_trip(euler32: np.ndarray) -> None:
    """Assert that Euler angles survive orix' Euler conversion and back
    in float32, the input route of the compatible mode.
    """
    flat = euler32.reshape(-1, 3)
    back = Rotation.from_euler(flat.astype(np.float64)).to_euler()
    assert np.array_equal(back.astype(np.float32), flat)


def rotate_about_axis(axis, angle_deg) -> Rotation:
    """Return rotations about one axis by angles in degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(np.atleast_1d(np.asarray(angle_deg, dtype=np.float64))) / 2
    data = np.zeros(half.shape + (4,))
    data[..., 0] = np.cos(half)
    data[..., 1:] = np.sin(half)[..., None] * axis
    return Rotation(data)


def near_identity_euler32() -> np.ndarray:
    """Return float32 Euler angles of a (6, 8) map of (10, 20, 30) deg
    perturbed by at most 0.01 deg per point (``default_rng(13)``).
    """
    rng = np.random.default_rng(13)
    n = 6 * 8
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = np.deg2rad(rng.uniform(0, 0.01, n)) / 2
    data = np.column_stack([np.cos(half), np.sin(half)[:, None] * axes])
    g0 = Rotation.from_euler(np.deg2rad([10.0, 20.0, 30.0]))
    rot = Rotation(data) * g0
    return rot.to_euler().astype(np.float32).reshape(6, 8, 3)


def duplicated_euler32() -> tuple[np.ndarray, np.ndarray]:
    """Return float32 Euler angles of a (4, 6) map whose every second
    column copies its left neighbour and whose rows 1 and 3 copy the
    row above, and the triplet placed at (0, 0).

    The triplet at (0, 0) is the first one drawn from
    ``default_rng(15)`` after the map whose EMsoft disorientation angle
    to itself is not 0 but a symmetry angle: every near-identity cosine
    rounds above 1, so the unclipped ``acos`` gives NaN there and a
    non-identity combination wins. (A cosine rounding below 1 gives
    only about 4e-8 rad, which the float32 KAM may not resolve.)
    """
    rng = np.random.default_rng(15)
    euler = random_euler32(rng, 24).reshape(4, 6, 3)
    special = None
    for _ in range(10_000):
        candidate = random_euler32(rng, 1)[0]
        if emsoft_dis(candidate, candidate) > 1.0:
            special = candidate
            break
    assert special is not None
    euler[0, 0] = special
    euler[:, 1::2] = euler[:, 0::2]
    euler[1] = euler[0]
    euler[3] = euler[2]
    return euler, special


def disorientation_to_identity_deg(rotation: Rotation) -> float:
    """Return the m-3m disorientation angle in degrees of one rotation
    to the identity, from orix.
    """
    ori = Orientation(rotation.data.reshape(1, 4), Oh)
    identity = Orientation.identity((1,))
    identity.symmetry = Oh
    return float(np.rad2deg(ori.angle_with(identity)[0]))


def correct_expected(mean_rad: np.ndarray) -> np.ndarray:
    """Return the float32 degrees of float64 mean angles in radians as
    the correct mode rounds them: through float32 radians.
    """
    mean32 = np.asarray(mean_rad, dtype=np.float64).astype(np.float32)
    return (mean32.astype(np.float64) * RTOD).astype(np.float32)


def assert_within_one_float32_ulp(kam: np.ndarray, expected: np.ndarray) -> None:
    """Assert that two float32 arrays have NaN at the same points and
    differ by at most one float32 spacing of the expected values
    elsewhere.
    """
    assert kam.dtype == np.float32
    assert kam.shape == expected.shape
    expected = np.asarray(expected, dtype=np.float32)
    assert np.array_equal(np.isnan(kam), np.isnan(expected))
    finite = ~np.isnan(expected)
    diff = np.abs(kam[finite].astype(np.float64) - expected[finite])
    tol = np.spacing(np.abs(expected[finite])).astype(np.float64)
    assert np.all(diff <= tol), (diff - tol).max()


def gradient_mean_rad(
    shape: tuple[int, int],
    delta_x: float,
    delta_y: float,
    present: np.ndarray | None = None,
    phase_id: np.ndarray | None = None,
) -> np.ndarray:
    """Return the analytic correct KAM in float64 radians of a gradient
    map: the mean of ``delta_x`` (horizontal) and ``delta_y`` (vertical)
    over the existing, present, same-phase 4-neighbours; NaN without
    one or where not present.
    """
    H, W = shape
    if present is None:
        present = np.ones(shape, dtype=bool)
    if phase_id is None:
        phase_id = np.zeros(shape, dtype=int)
    present = np.asarray(present).reshape(shape)
    phase_id = np.asarray(phase_id).reshape(shape)
    out = np.full(shape, np.nan)
    dx, dy = np.deg2rad(delta_x), np.deg2rad(delta_y)
    for y in range(H):
        for x in range(W):
            if not present[y, x]:
                continue
            values = []
            for oy, ox, value in ((0, -1, dx), (0, 1, dx), (-1, 0, dy), (1, 0, dy)):
                yn, xn = y + oy, x + ox
                if 0 <= yn < H and 0 <= xn < W and present[yn, xn]:
                    if phase_id[yn, xn] == phase_id[y, x]:
                        values.append(value)
            if values:
                out[y, x] = np.mean(values)
    return out


def self_dot(rotation_data: np.ndarray) -> float:
    """Return the float64 ``max_j |<S_j o, o>|`` over m-3m's proper
    operators.
    """
    o = Rotation(rotation_data.reshape(1, 4))
    ops = Rotation(Oh.proper_subgroup.data)
    products = (ops * o).data.reshape(-1, 4)
    return float(np.abs(products @ rotation_data.reshape(4)).max())


# Tests


class TestEMsoftQuaternions:
    def test_operator_table_for_m3m_is_emsoft_order(self):
        operators = emsoft_symmetry_operators(32)
        assert operators.shape == (24, 4)
        assert operators.dtype == np.float64
        assert np.array_equal(operators[0], [1.0, 0.0, 0.0, 0.0])
        assert np.array_equal(operators, EMSOFT_432_OPERATORS)

        # As a set up to sign equal to orix' proper subgroup of m-3m
        orix_ops = Oh.proper_subgroup.data
        assert orix_ops.shape == (24, 4)
        for q in operators:
            dist = np.minimum(
                np.abs(orix_ops - q).max(axis=1), np.abs(orix_ops + q).max(axis=1)
            )
            assert dist.min() <= 1e-15
        for q in orix_ops:
            dist = np.minimum(
                np.abs(operators - q).max(axis=1), np.abs(operators + q).max(axis=1)
            )
            assert dist.min() <= 1e-15

    def test_eq_matches_orix_from_euler(self):
        rng = np.random.default_rng(14)
        euler64 = random_euler32(rng, 1000).astype(np.float64)
        q = emsoft_euler_to_quaternion(euler64)
        assert q.shape == (1000, 4)
        assert q.dtype == np.float64
        assert np.all(q[:, 0] >= 0)
        assert np.allclose(q, Rotation.from_euler(euler64).data, atol=1e-15, rtol=0)
        # The transcription of eq_ in this module agrees bit for bit
        assert np.array_equal(q, emsoft_eq(euler64))

        q1 = emsoft_euler_to_quaternion(np.deg2rad([10.0, 20.0, 30.0]))
        assert np.allclose(
            q1.ravel(), [0.92542, -0.17101, 0.03015, -0.33682], atol=5e-6, rtol=0
        )

    def test_euler_round_trip_reproduces_the_shipped_float32_angles(self):
        directory = Path(kp.data.__file__).parent / "emsoft_hrosm"
        with np.load(
            directory / "regression_hrosm_large_refined.npz", allow_pickle=False
        ) as npz:
            arrays = {key: npz[key] for key in ("EulerAngles", "RefinedEulerAngles")}
        for name, euler32 in arrays.items():
            assert euler32.dtype == np.float32, name
            assert euler32.shape == (4125, 3), name
            back = Rotation.from_euler(euler32.astype(np.float64)).to_euler()
            back32 = back.astype(np.float32)
            assert np.array_equal(back32, euler32), name
            assert np.array_equal(
                emsoft_euler_to_quaternion(back32.astype(np.float64)),
                emsoft_euler_to_quaternion(euler32.astype(np.float64)),
            ), name

    def test_point_group_number_rejects_unmapped_or_mismatched_groups(
        self, monkeypatch
    ):
        assert emsoft_point_group_number(Oh) == 32
        assert emsoft_point_group_number(O) == 30

        renamed = Symmetry(Oh.data)
        renamed.name = "not-an-emsoft-group"
        with pytest.raises(ValueError, match="point group"):
            emsoft_point_group_number(renamed)

        # EMsoft's operator table with one operator of m-3m dropped
        columns = _emsoft_quaternions._QSYM_INIT_COLUMNS
        monkeypatch.setitem(columns, 30, columns[30][:-1])
        assert emsoft_symmetry_operators(32).shape == (23, 4)
        with pytest.raises(ValueError, match="point group"):
            emsoft_point_group_number(Oh)

    def test_quatmult_term_order_matches_orix_product(self):
        rng = np.random.default_rng(16)
        a = rng.normal(size=(1000, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.normal(size=(1000, 4))
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        ab = emsoft_quaternion_multiply(a, b)
        assert ab.shape == (1000, 4)
        assert ab.dtype == np.float64
        # EMsoft's term order, bit for bit
        assert np.array_equal(ab, emsoft_quatmult(a, b))
        # Close to orix' product, whose term order differs
        ab_orix = (Rotation(a) * Rotation(b)).data
        assert np.allclose(ab, ab_orix, atol=QUATMULT_ORIX_ATOL, rtol=0)

        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.array_equal(emsoft_quaternion_multiply(i, j), k)


class TestCompatKAM:
    @pytest.mark.parametrize(
        "shape", COMPAT_SHAPES + ["near-identity"], ids=lambda s: str(s)
    )
    def test_matches_the_loop_transcription_bitwise(self, shape):
        if shape == "near-identity":
            euler32 = near_identity_euler32()
        else:
            rng = np.random.default_rng(10)
            n = int(np.prod(shape))
            euler32 = random_euler32(rng, n).reshape(shape + (3,))
        assert_euler_round_trip(euler32)
        H, W = euler32.shape[:2]
        xmap = crystal_map_from_euler32(euler32)

        kam = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
        _, expected = emsoft_kam_loop(euler32)
        assert kam.dtype == np.float32
        assert kam.shape == (H, W)
        assert np.array_equal(kam, expected, equal_nan=True)

    def test_matches_the_loop_transcription_on_duplicated_orientations(self):
        euler32, special = duplicated_euler32()
        assert_euler_round_trip(euler32)

        # The map holds a duplicated neighbour pair whose unclipped
        # EMsoft angle is not 0, and clipping the cosines changes the
        # map
        assert np.array_equal(euler32[0, 1], euler32[0, 0])
        assert emsoft_dis(special, special) != 0
        assert emsoft_dis(special, special, clip=True) == 0
        _, expected = emsoft_kam_loop(euler32)
        _, expected_clipped = emsoft_kam_loop(euler32, clip=True)
        assert not np.array_equal(expected, expected_clipped)

        xmap = crystal_map_from_euler32(euler32)
        kam = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
        assert np.array_equal(kam, expected)

    @pytest.mark.parametrize("shape", [(4, 5), (6, 3)])
    def test_constant_pair_angle_field(self, shape, hrosm_constant_pair_xmap):
        phi = 0.3
        H, W = shape
        xmap = hrosm_constant_pair_xmap(shape, phi=phi)
        s = disorientation_to_identity_deg(xmap.rotations[W - 1])

        kam = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
        assert kam.dtype == np.float32
        expected = np.full(shape, phi)
        expected[0, 0] = phi + s
        expected[0, W - 1] = phi + s / 2
        assert np.allclose(kam, expected, rtol=CONSTANT_PAIR_RTOL, atol=0)

    def test_vertical_pair_is_credited_one_column_right(self, hrosm_constant_pair_xmap):
        shape = (5, 5)
        xmap = hrosm_constant_pair_xmap(shape, phi=0.3)
        data = xmap.rotations.data.copy()
        flat = 2 * 5 + 1
        data[flat] = (rotate_about_axis((0, 0, 1), 3.0) * Rotation(data[flat])).data
        xmap2 = crystal_map(Rotation(data), shape)

        changed = {}
        for compat in (True, False):
            kam1 = kernel_average_misorientation_map(xmap, emsoft_compatible=compat)
            kam2 = kernel_average_misorientation_map(xmap2, emsoft_compatible=compat)
            ys, xs = np.nonzero(kam1 != kam2)
            changed[compat] = set(zip(ys.tolist(), xs.tolist()))
        assert changed[True] == {(2, 0), (2, 1), (2, 2), (1, 2), (3, 1)}
        assert changed[False] == {(2, 0), (2, 1), (2, 2), (1, 1), (3, 1)}

    def test_first_row_last_column_is_compared_with_the_identity(
        self, hrosm_constant_pair_xmap
    ):
        phi = 0.3
        shape = (4, 5)
        W = shape[1]
        kams = []
        s = []
        for euler0 in [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)]:
            xmap = hrosm_constant_pair_xmap(shape, phi=phi, euler0=euler0)
            s.append(disorientation_to_identity_deg(xmap.rotations[W - 1]))
            kams.append(kernel_average_misorientation_map(xmap, emsoft_compatible=True))
        ds = s[1] - s[0]
        assert abs(ds) > 1
        assert np.isclose(
            float(kams[1][0, 0]) - float(kams[0][0, 0]),
            ds,
            rtol=0,
            atol=CONSTANT_PAIR_RTOL * (phi + max(s)) * 2,
        )
        assert np.isclose(
            float(kams[1][0, W - 1]) - float(kams[0][0, W - 1]),
            ds / 2,
            rtol=0,
            atol=CONSTANT_PAIR_RTOL * (phi + max(s)) * 2,
        )
        others = np.ones(shape, dtype=bool)
        others[0, 0] = others[0, W - 1] = False
        for kam in kams:
            assert np.allclose(kam[others], phi, rtol=CONSTANT_PAIR_RTOL, atol=0)

    def test_degrees_output_rounds_through_float32_radians(self):
        rng = np.random.default_rng(10)
        euler32 = random_euler32(rng, 7 * 9).reshape(7, 9, 3)
        xmap = crystal_map_from_euler32(euler32)

        kam_deg = kernel_average_misorientation_map(xmap, emsoft_compatible=True)
        kam_rad = kernel_average_misorientation_map(
            xmap, degrees=False, emsoft_compatible=True
        )
        assert kam_deg.dtype == np.float32
        assert kam_rad.dtype == np.float32
        assert kam_deg.shape == kam_rad.shape == (7, 9)
        assert np.array_equal(
            kam_deg, (kam_rad.astype(np.float64) * RTOD).astype(np.float32)
        )

        # The map is sensitive to the order of rounding: a float32
        # product with rtod differs on some points
        rad, _ = emsoft_kam_loop(euler32)
        assert np.array_equal(kam_rad, rad)
        float32_product = rad * np.float32(RTOD)
        assert np.any(float32_product != kam_deg)

    def test_rejects_absent_points_and_several_phases(self, hrosm_gradient_xmap):
        xmap_absent = hrosm_gradient_xmap((4, 5), absent=(3,))
        with pytest.raises(
            ValueError, match="emsoft_compatible requires every map point"
        ):
            kernel_average_misorientation_map(xmap_absent, emsoft_compatible=True)

        phase_id = np.zeros((4, 5), dtype=int)
        phase_id[:, 3:] = 1
        xmap_phases = hrosm_gradient_xmap((4, 5), phase_id=phase_id)
        with pytest.raises(ValueError, match="one phase"):
            kernel_average_misorientation_map(xmap_phases, emsoft_compatible=True)


class TestCorrectKAM:
    @pytest.mark.parametrize("shape", [(6, 7), (1, 5), (5, 1), (2, 2), (1, 1)])
    def test_two_axis_gradient_field_is_exact(self, shape, hrosm_gradient_xmap):
        dx, dy = 0.4, 0.1
        xmap = hrosm_gradient_xmap(shape, delta_x=dx, delta_y=dy)
        kam = kernel_average_misorientation_map(xmap)
        mean_rad = gradient_mean_rad(shape, dx, dy)
        assert_within_one_float32_ulp(kam, correct_expected(mean_rad))

        # The named cases of the analytic field (degrees)
        H, W = shape
        expected_deg = correct_expected(mean_rad)
        if shape == (6, 7):
            for (y, x), value in [
                ((2, 3), (2 * dx + 2 * dy) / 4),
                ((0, 3), (2 * dx + dy) / 3),
                ((H - 1, 3), (2 * dx + dy) / 3),
                ((2, 0), (dx + 2 * dy) / 3),
                ((2, W - 1), (dx + 2 * dy) / 3),
                ((0, 0), (dx + dy) / 2),
                ((H - 1, W - 1), (dx + dy) / 2),
            ]:
                assert np.isclose(expected_deg[y, x], value, rtol=1e-6)
        elif shape == (1, 5):
            assert np.allclose(expected_deg, dx, rtol=1e-6)
        elif shape == (5, 1):
            assert np.allclose(expected_deg, dy, rtol=1e-6)
        elif shape == (1, 1):
            assert np.isnan(kam[0, 0])

    def test_matches_orix_angle_with_on_scrambled_variants(self, hrosm_gradient_xmap):
        shape = (5, 6)
        H, W = shape
        xmap = hrosm_gradient_xmap(shape, scramble_seed=11)
        rng = np.random.default_rng(12)
        n = H * W
        axes = rng.normal(size=(n, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        half = np.deg2rad(rng.uniform(0, 2, n)) / 2
        small = Rotation(np.column_stack([np.cos(half), np.sin(half)[:, None] * axes]))
        rot = small * xmap.rotations
        xmap2 = crystal_map(rot, shape)

        ori = Orientation(rot.data, Oh)
        mean_rad = np.full(shape, np.nan)
        for y in range(H):
            for x in range(W):
                angles = []
                for oy, ox in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    yn, xn = y + oy, x + ox
                    if 0 <= yn < H and 0 <= xn < W:
                        a = ori[y * W + x]
                        b = ori[yn * W + xn]
                        angles.append(float(a.angle_with(b)[0]))
                mean_rad[y, x] = np.mean(angles)
        # Without symmetry, scrambled neighbours are far apart
        unreduced = Rotation(rot.data[:-1]).angle_with(Rotation(rot.data[1:]))
        assert np.max(unreduced) > np.deg2rad(90)

        kam = kernel_average_misorientation_map(xmap2)
        assert_within_one_float32_ulp(kam, correct_expected(mean_rad))

    def test_absent_neighbours_are_excluded_and_isolated_pixels_are_nan(
        self, hrosm_gradient_xmap
    ):
        shape = (4, 5)
        dx, dy = 0.4, 0.1
        absent = (1, 5, 13)
        xmap = hrosm_gradient_xmap(shape, delta_x=dx, delta_y=dy, absent=absent)
        kam = kernel_average_misorientation_map(xmap)
        assert kam.shape == shape

        flat = kam.ravel()
        assert np.all(np.isnan(flat[list(absent)]))
        assert np.isnan(kam[0, 0])
        present = np.ones(shape[0] * shape[1], dtype=bool)
        present[list(absent)] = False
        mean_rad = gradient_mean_rad(shape, dx, dy, present=present)
        assert np.isclose(mean_rad[1, 1], np.deg2rad((dx + dy) / 2))
        assert_within_one_float32_ulp(kam, correct_expected(mean_rad))

    def test_pairs_across_phases_are_skipped(self, hrosm_gradient_xmap):
        shape = (4, 6)
        dx, dy = 0.4, 0.1
        phase_id = np.zeros(shape, dtype=int)
        phase_id[:, 3:] = 1
        xmap = hrosm_gradient_xmap(shape, delta_x=dx, delta_y=dy, phase_id=phase_id)
        kam = kernel_average_misorientation_map(xmap)

        mean_rad = gradient_mean_rad(shape, dx, dy, phase_id=phase_id)
        # Columns 2 and 3 use three (interior rows) or two (edge rows)
        # neighbours
        for x in (2, 3):
            assert np.isclose(mean_rad[1, x], np.deg2rad((dx + 2 * dy) / 3))
            assert np.isclose(mean_rad[0, x], np.deg2rad((dx + dy) / 2))
        assert_within_one_float32_ulp(kam, correct_expected(mean_rad))

    def test_first_rotation_is_used_with_several_rotations_per_point(
        self, hrosm_gradient_xmap
    ):
        shape = (4, 5)
        kam1 = kernel_average_misorientation_map(hrosm_gradient_xmap(shape))
        xmap3 = hrosm_gradient_xmap(shape, rotations_per_point=3)
        kam3 = kernel_average_misorientation_map(xmap3)
        assert np.array_equal(kam1, kam3, equal_nan=True)
        assert np.all(np.isfinite(kam1))

    def test_degrees_false_returns_float32_radians(self, hrosm_gradient_xmap):
        shape = (6, 7)
        dx, dy = 0.4, 0.1
        xmap = hrosm_gradient_xmap(shape, delta_x=dx, delta_y=dy)
        kam_rad = kernel_average_misorientation_map(xmap, degrees=False)
        kam_deg = kernel_average_misorientation_map(xmap)
        assert kam_rad.dtype == np.float32
        assert kam_rad.shape == shape

        mean_rad = gradient_mean_rad(shape, dx, dy)
        assert_within_one_float32_ulp(kam_rad, mean_rad.astype(np.float32))
        assert_within_one_float32_ulp(
            kam_deg, (kam_rad.astype(np.float64) * RTOD).astype(np.float32)
        )

    def test_dot_to_angle_snaps_within_four_eps_of_one(self):
        # Exact on the dot products themselves, whatever order the
        # implementation sums the four products of a dot product in
        eps = np.finfo(np.float64).eps
        d = np.array([1.0, 1 - eps, 1 - 4 * eps, 1 + 2 * eps, 1 - 5 * eps, 0.5])
        angle = _dot_to_angle(d)
        assert angle.dtype == np.float64
        assert angle.shape == d.shape
        # At or above 1 - 4 eps: exactly 0, never NaN. A clip to [0, 1]
        # gives a positive angle at 1 - eps, a plain arccos NaN above 1
        assert np.array_equal(angle[:4], np.zeros(4))
        # Below the snap: the plain 2 arccos
        assert angle[4] > 0
        assert angle[4] == 2 * np.arccos(1 - 5 * eps)
        assert angle[5] == 2 * np.arccos(0.5)

    def test_identical_neighbours_give_zero_not_nan(self):
        rng = np.random.default_rng(13)
        n = 1000
        euler32 = np.column_stack(
            [
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0, np.pi, n),
                rng.uniform(0, 2 * np.pi, n),
            ]
        ).astype(np.float32)
        rot = Rotation.from_euler(euler32.astype(np.float64)).data
        below = above = None
        for q in rot:
            d = self_dot(q)
            if below is None and d < 1:
                below = q
            if above is None and d > 1:
                above = q
            if below is not None and above is not None:
                break
        assert below is not None and self_dot(below) < 1
        assert above is not None and self_dot(above) > 1

        shape = (3, 4)
        for q in (below, above):
            xmap = crystal_map(Rotation(np.tile(q, (12, 1))), shape)
            kam = kernel_average_misorientation_map(xmap)
            assert np.all(np.isfinite(kam))
            assert np.all(kam == 0.0)

        # The map of duplicated orientations: finite everywhere, exactly
        # 0 where every neighbour is identical
        euler_dup, _ = duplicated_euler32()
        xmap = crystal_map_from_euler32(euler_dup)
        kam = kernel_average_misorientation_map(xmap)
        H, W = euler_dup.shape[:2]
        assert np.all(np.isfinite(kam))
        all_identical = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                same = []
                for oy, ox in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    yn, xn = y + oy, x + ox
                    if 0 <= yn < H and 0 <= xn < W:
                        same.append(np.array_equal(euler_dup[yn, xn], euler_dup[y, x]))
                all_identical[y, x] = all(same)
        assert all_identical.any()
        assert np.all(kam[all_identical] == 0.0)
