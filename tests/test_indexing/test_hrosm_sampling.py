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

"""Tests of the misorientation ball sampling of the HROSM tools.

The oracles are the cubochoric grid of EMsoft's filled misorientation
cube written out here, orix' own cubochoric to homochoric conversion,
the Rodrigues composition formula of EMsoft's misorientation sampling,
an all-pairs brute force of the nearest-neighbour angle, the ball that
EMsoft's ``EMsampleRFZ`` wrote (shipped for 6 steps, and run by the
binary for 6 and 20 steps when ``KIKUCHIPY_EMSOFT_BIN`` is set).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from orix.quaternion import Rotation
from orix.quaternion._conversions import cu2ho
import pytest

import kikuchipy as kp
from kikuchipy.indexing import misorientation_ball, misorientation_ball_spacing
from kikuchipy.indexing._hrosm._sampling import (
    _cubochoric_grid,
    _cubochoric_to_homochoric,
)

# Mean nearest-neighbour angle in degrees of the ball at 5 degrees and
# 20 steps (the defaults), 10 steps and 2 steps; measured 2026-10-06
BALL_SPACING_DEFAULT_DEG = 0.159176
BALL_SPACING_N10_DEG = 0.317654
BALL_SPACING_N2_DEG = 1.601304

# Centre of the balls compared with EMsoft: Rodrigues vector along the
# unit axis (1, 2, 3) / sqrt(14) with magnitude tan(omega / 2) = 0.1
CENTER_AXIS = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
CENTER_TAN_HALF_ANGLE = 0.1

# Half edge of the cubochoric cube
CUBE_HALF_EDGE = 0.5 * np.pi ** (2 / 3)

# Text precision of EMsoft's orientation files (9 decimals)
EMSOFT_TEXT_TOL = 6e-10


# ------------------------ Test-local oracles ------------------------


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


def _angle_deg(q: np.ndarray) -> np.ndarray:
    """Return rotation angles in degrees, accurate for small angles."""
    q = np.asarray(q, dtype=np.float64)
    s = np.linalg.norm(q[..., 1:], axis=-1)
    return np.rad2deg(2 * np.arctan2(s, np.abs(q[..., 0])))


def _misorientation_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation angle in radians of ``a * conj(b)``,
    elementwise, without crystal symmetry.
    """
    return np.deg2rad(_angle_deg(_qmul(a, _conj(b))))


def _grid_indices(n_steps: int) -> np.ndarray:
    """Return the integer grid ``(i, j, k)`` from ``-N`` to ``N``, the
    first index varying slowest, shape ((2N + 1)**3, 3).
    """
    idx = np.arange(-n_steps, n_steps + 1)
    grid = np.meshgrid(idx, idx, idx, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def _half_edge(max_angle_deg: float) -> float:
    """Return the cube half edge of a ball whose surface lies at
    ``max_angle_deg``.
    """
    w = np.deg2rad(max_angle_deg)
    return 0.5 * (np.pi * (w - np.sin(w))) ** (1 / 3)


def _raw_grid_rotations(max_angle_deg: float, n_steps: int) -> np.ndarray:
    """Return the unshifted grid rotations ``Q_v`` through orix'
    cubochoric to homochoric conversion, shape ((2N + 1)**3, 4).
    """
    dx = _half_edge(max_angle_deg) / n_steps
    cu = _grid_indices(n_steps) * dx
    return Rotation.from_homochoric(cu2ho(cu)).data


def _center_rotation() -> Rotation:
    half = np.arctan(CENTER_TAN_HALF_ANGLE)
    data = np.r_[np.cos(half), np.sin(half) * CENTER_AXIS]
    return Rotation(data)


def _shell(n_steps: int) -> np.ndarray:
    return np.abs(_grid_indices(n_steps)).max(axis=1)


def _expected_shell_counts(n_steps: int) -> list[int]:
    return [1] + [24 * m**2 + 2 for m in range(1, n_steps + 1)]


def _brute_force_spacing_deg(q: np.ndarray) -> float:
    """Return the mean, over all rotations, of the angle in degrees to
    the nearest other rotation, from all pairs.
    """
    m = _qmul(q[:, None, :], _conj(q)[None, :, :])
    angle = _angle_deg(m)
    np.fill_diagonal(angle, np.inf)
    return float(angle.min(axis=1).mean())


def _rodrigues_four_vectors(q: np.ndarray) -> np.ndarray:
    """Return (unit axis, tan(omega / 2)) per quaternion with q0 >= 0;
    the axis is undefined for a zero rotation (left as NaN).
    """
    q = np.where(q[:, :1] < 0, -q, q)
    s = np.linalg.norm(q[:, 1:], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = q[:, 1:] / s[:, None]
    return np.column_stack([axis, s / q[:, 0]])


def _emsoft_float32_rodrigues_round_trip(q: np.ndarray) -> np.ndarray:
    """Return quaternions stored as EMsoft's float32 Rodrigues
    four-vector (unit axis, tan(omega / 2)), ``(0, 0, 1, 0)`` for a zero
    rotation, and converted back in float64 by EMsoft's Rodrigues to
    axis-angle (angle ``2 atan(t)``, axis renormalised by the reciprocal
    norm), axis-angle to quaternion and normalisation.
    """
    q = np.where(q[:, :1] < 0, -q, q)
    s = np.linalg.norm(q[:, 1:], axis=1)
    zero = s == 0
    rod = np.zeros((q.shape[0], 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        rod[:, :3] = q[:, 1:] / s[:, None]
        rod[:, 3] = s / q[:, 0]
    rod[zero] = (0.0, 0.0, 1.0, 0.0)
    rod = rod.astype(np.float32).astype(np.float64)

    out = np.zeros_like(rod)
    t = rod[:, 3]
    is_zero = t == 0
    angle = 2.0 * np.arctan(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = 1.0 / np.sqrt(np.sum(rod[:, :3] * rod[:, :3], axis=1))
    axis = rod[:, :3] * inv[:, None]
    c = np.cos(angle * 0.5)
    sn = np.sin(angle * 0.5)
    out[:, 0] = c
    out[:, 1:] = axis * sn[:, None]
    out[is_zero] = (1.0, 0.0, 0.0, 0.0)
    norm = np.sqrt(np.sum(out**2, axis=1))
    return out / norm[:, None]


def _with_positive_scalar(q: np.ndarray) -> np.ndarray:
    return np.where(q[:, :1] < 0, -q, q)


def _read_emsoft_orientation_file(fpath: Path) -> np.ndarray:
    """Return the rows of an EMsoft orientation text file (first line
    the representation, second line the count).
    """
    lines = Path(fpath).read_text().splitlines()
    count = int(lines[1])
    data = np.loadtxt(lines[2:], dtype=np.float64, ndmin=2)
    assert data.shape[0] == count
    return data


# ------------------------------ Tests -------------------------------


class TestMisorientationBall:
    @pytest.mark.parametrize("n_steps", [1, 2, 6, 20])
    def test_count_order_and_shells(self, n_steps):
        n = 2 * n_steps + 1
        ball = misorientation_ball(max_angle=5.0, n_steps=n_steps)
        assert isinstance(ball, Rotation)
        assert ball.shape == (n**3,)

        # Grid order: the first index varies slowest, the last fastest
        grid = _cubochoric_grid(np.deg2rad(5.0), n_steps)
        assert grid.shape == (n**3, 3)
        assert grid.dtype == np.float64
        dx = _half_edge(5.0) / n_steps
        idx = _grid_indices(n_steps)
        np.testing.assert_allclose(grid, idx * dx, rtol=1e-14, atol=1e-18)
        i, j, k = 1, -1, n_steps
        flat = (i + n_steps) * n**2 + (j + n_steps) * n + (k + n_steps)
        np.testing.assert_allclose(grid[flat], np.array([i, j, k]) * dx, rtol=1e-14)

        # Shells of constant angle with 24 m**2 + 2 points
        shell = _shell(n_steps)
        counts = np.bincount(shell, minlength=n_steps + 1)
        assert counts.tolist() == _expected_shell_counts(n_steps)
        if n_steps == 6:
            assert counts.tolist() == [1, 26, 98, 218, 386, 602, 866]
        if n_steps == 20:
            assert counts[-1] == 9602

        angle = _angle_deg(ball.data)
        assert np.all(angle[shell == 0] <= 1e-12)
        shell_angle = np.array([angle[shell == m].mean() for m in range(n_steps + 1)])
        assert np.all(np.diff(shell_angle) > 0)
        for m in range(1, n_steps + 1):
            values = angle[shell == m]
            assert values.max() - values.min() <= 1e-10

    @pytest.mark.parametrize(
        "max_angle, n_steps", [(5.0, 20), (1.0, 6), (10.0, 6), (30.0, 6)]
    )
    def test_outer_shell_is_exactly_max_angle(self, max_angle, n_steps):
        ball = misorientation_ball(max_angle=max_angle, n_steps=n_steps)
        angle = _angle_deg(ball.data)
        shell = _shell(n_steps)
        assert abs(angle.max() - max_angle) <= 1e-9
        assert np.all(np.abs(angle[shell == n_steps] - max_angle) <= 1e-9)
        for m in range(1, n_steps + 1):
            values = angle[shell == m]
            assert values.max() - values.min() <= 1e-10
            # w_m - sin(w_m) = (m / N)**3 (w - sin(w))
            w = np.deg2rad(max_angle)
            w_m = np.deg2rad(values.mean())
            assert w_m - np.sin(w_m) == pytest.approx(
                (m / n_steps) ** 3 * (w - np.sin(w)), rel=1e-8
            )

        grid = _cubochoric_grid(np.deg2rad(max_angle), n_steps)
        edge = _half_edge(max_angle)
        assert grid.max() == pytest.approx(edge, rel=1e-15)
        assert grid.min() == pytest.approx(-edge, rel=1e-15)

        if max_angle == 5.0 and n_steps == 20:
            assert angle[shell == 1].mean() == pytest.approx(0.24997, abs=1e-5)
            assert grid.max() == pytest.approx(0.035163745447257796, rel=1e-15)
            dx = grid[1, 2] - grid[0, 2]
            assert dx == pytest.approx(0.0017581872723628899, rel=1e-12)
            assert grid[(2 * 20 + 1) ** 3 // 2 + 1, 2] == pytest.approx(
                0.0017581872723628899, rel=1e-15
            )

    def test_cube_to_ball_matches_orix_cu2ho(self):
        n_steps = 20
        grid = _cubochoric_grid(np.deg2rad(5.0), n_steps)
        ho = _cubochoric_to_homochoric(grid)
        assert ho.shape == grid.shape
        np.testing.assert_allclose(ho, cu2ho(grid), atol=1e-12, rtol=0)

        # Random cube points plus the origin, face centres, edges,
        # corners and the pyramid boundaries |x| = |y| (and the other
        # pairs of coordinates); the cube surface is taken two floats
        # inside the half edge, so that no point lies outside the cube
        # by rounding
        rng = np.random.default_rng(60)
        a = np.nextafter(np.nextafter(CUBE_HALF_EDGE, 0), 0)
        special = [np.zeros(3)]
        for axis in range(3):
            for sign in (-1, 1):
                p = np.zeros(3)
                p[axis] = sign * a
                special.append(p)
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                special += [
                    np.array([s1 * a, s2 * a, 0.0]),
                    np.array([s1 * a, 0.0, s2 * a]),
                    np.array([0.0, s1 * a, s2 * a]),
                ]
                for s3 in (-1, 1):
                    special.append(np.array([s1 * a, s2 * a, s3 * a]))
        special = np.array(special)
        n_boundary = 300
        t = rng.uniform(-a, a, n_boundary)
        s = rng.uniform(-1, 1, n_boundary) * np.abs(t)
        sign = rng.choice([-1.0, 1.0], n_boundary)
        third = n_boundary // 3
        boundary = np.empty((n_boundary, 3))
        boundary[:third] = np.column_stack([t, sign * t, s])[:third]
        boundary[third : 2 * third] = np.column_stack([t, s, sign * t])[
            third : 2 * third
        ]
        boundary[2 * third :] = np.column_stack([s, t, sign * t])[2 * third :]
        n_random = 1000 - special.shape[0] - n_boundary
        random = rng.uniform(-a, a, (n_random, 3))
        cu = np.concatenate([special, boundary, random])
        assert cu.shape == (1000, 3)
        np.testing.assert_allclose(
            _cubochoric_to_homochoric(cu), cu2ho(cu), atol=1e-12, rtol=0
        )

        # The ball is the conjugate of the grid rotations, in order
        ball = misorientation_ball(max_angle=5.0, n_steps=n_steps)
        expected = _conj(Rotation.from_homochoric(ho).data)
        np.testing.assert_allclose(ball.data, expected, atol=1e-12, rtol=0)

    def test_composition_is_conj_ball_times_center(self):
        n_steps = 6
        center = _center_rotation()
        q0 = center.data.reshape(4)
        raw = _raw_grid_rotations(5.0, n_steps)
        ball = misorientation_ball(center, max_angle=5.0, n_steps=n_steps)
        assert ball.shape == (raw.shape[0],)

        expected = _qmul(_conj(raw), q0)
        np.testing.assert_allclose(ball.data, expected, atol=1e-15, rtol=0)

        # EMsoft's Rodrigues composition (-v + rho0 + rho0 x v) /
        # (1 + v . rho0) of the grid vector v with the centre rho0
        v = raw[:, 1:] / raw[:, :1]
        rho0 = q0[1:] / q0[0]
        moved = (-v + rho0 + np.cross(rho0, v)) / (1 + v @ rho0)[:, None]
        rod = ball.data[:, 1:] / ball.data[:, :1]
        np.testing.assert_allclose(rod, moved, atol=1e-12, rtol=0)

        # The other composition sides are measurably different
        for wrong in (_qmul(q0, _conj(raw)), _qmul(raw, q0)):
            assert _misorientation_rad(wrong, expected).max() > 1e-3

    def test_center_none_is_the_identity(self):
        a = misorientation_ball(max_angle=5.0, n_steps=2)
        b = misorientation_ball(Rotation.identity(), max_angle=5.0, n_steps=2)
        np.testing.assert_array_equal(a.data, b.data)
        expected = _conj(_raw_grid_rotations(5.0, 2))
        np.testing.assert_allclose(a.data, expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("centered", [False, True], ids=["identity", "center"])
    def test_compat_storage_is_float32_rodrigues(self, centered):
        center = _center_rotation() if centered else None
        n_steps = 6 if centered else 2
        correct = misorientation_ball(center, max_angle=5.0, n_steps=n_steps)
        compat = misorientation_ball(
            center, max_angle=5.0, n_steps=n_steps, emsoft_compatible=True
        )
        assert compat.shape == correct.shape
        expected = _emsoft_float32_rodrigues_round_trip(correct.data)
        np.testing.assert_allclose(
            _with_positive_scalar(compat.data), expected, atol=1e-15, rtol=0
        )
        if not centered:
            # The zero rotation is stored as (0, 0, 1, 0), the identity
            middle = compat.size // 2
            np.testing.assert_allclose(
                _with_positive_scalar(compat.data[middle : middle + 1]),
                [[1.0, 0.0, 0.0, 0.0]],
                atol=1e-15,
            )
        assert not np.array_equal(compat.data, correct.data)
        assert _misorientation_rad(compat.data, correct.data).max() <= 2e-7

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_angle": 0}, "max_angle"),
            ({"max_angle": -1}, "max_angle"),
            ({"max_angle": 180}, "max_angle"),
            ({"n_steps": 0}, "n_steps"),
            ({"n_steps": True}, "n_steps"),
        ],
    )
    def test_argument_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            misorientation_ball(**kwargs)

    def test_argument_validation_rejects_several_centers(self):
        center = Rotation.identity((2,))
        with pytest.raises(ValueError, match="center"):
            misorientation_ball(center, max_angle=5.0, n_steps=1)


class TestBallSpacing:
    @pytest.mark.parametrize("n_steps", [1, 2, 3])
    def test_spacing_equals_brute_force_nearest_neighbour(self, n_steps):
        spacing = misorientation_ball_spacing(5.0, n_steps)
        ball = misorientation_ball(max_angle=5.0, n_steps=n_steps)
        expected = _brute_force_spacing_deg(ball.data)
        assert abs(spacing - expected) <= 1e-12
        if n_steps == 2:
            assert expected == pytest.approx(BALL_SPACING_N2_DEG, rel=1e-4)

    def test_spacing_is_invariant_under_the_grain_composition(self):
        rng = np.random.default_rng(61)
        data = rng.standard_normal(4)
        data /= np.linalg.norm(data)
        center = Rotation(data)
        ball = misorientation_ball(center, max_angle=5.0, n_steps=2)
        expected = _brute_force_spacing_deg(ball.data)
        assert abs(misorientation_ball_spacing(5.0, 2) - expected) <= 1e-9

    def test_default_spacing_pin(self):
        default = misorientation_ball_spacing()
        assert isinstance(default, float)
        assert default == pytest.approx(BALL_SPACING_DEFAULT_DEG, rel=1e-4)
        n10 = misorientation_ball_spacing(5, 10)
        assert isinstance(n10, float)
        assert n10 == pytest.approx(BALL_SPACING_N10_DEG, rel=1e-4)
        n2 = misorientation_ball_spacing(5, 2)
        assert isinstance(n2, float)
        assert n2 == pytest.approx(BALL_SPACING_N2_DEG, rel=1e-4)
        # Smaller than the radial step between shells
        assert default < 5.0 / 20

    def test_spacing_decreases_with_n_steps_and_grows_with_max_angle(self):
        by_steps = [misorientation_ball_spacing(5.0, n) for n in (2, 3, 4, 6)]
        assert np.all(np.diff(by_steps) < 0)
        assert min(by_steps) > 0
        by_angle = [misorientation_ball_spacing(a, 4) for a in (1.0, 2.0, 5.0, 10.0)]
        assert np.all(np.diff(by_angle) > 0)
        assert min(by_angle) > 0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_angle": 0}, "max_angle"),
            ({"max_angle": -1}, "max_angle"),
            ({"max_angle": 180}, "max_angle"),
            ({"n_steps": 0}, "n_steps"),
            ({"n_steps": True}, "n_steps"),
        ],
    )
    def test_spacing_argument_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            misorientation_ball_spacing(**kwargs)


class TestEMsampleRFZ:
    def test_shipped_n6_ball_matches_in_order(self):
        fpath = (
            Path(kp.data.__file__).parent
            / "emsoft_hrosm"
            / "regression_hrosm_ball_n6.npz"
        )
        with np.load(fpath, allow_pickle=False) as reference:
            qu = reference["qu"]
            ro = reference["ro"]
        n_steps = 6
        assert qu.shape == ((2 * n_steps + 1) ** 3, 4)
        assert ro.shape == qu.shape

        ball = misorientation_ball(_center_rotation(), max_angle=5.0, n_steps=n_steps)
        ours = _with_positive_scalar(ball.data)
        np.testing.assert_allclose(ours, qu, atol=EMSOFT_TEXT_TOL, rtol=0)

        self._assert_unshifted_grid_matches(ro, n_steps)

    @staticmethod
    def _assert_unshifted_grid_matches(ro: np.ndarray, n_steps: int):
        grid = _cubochoric_grid(np.deg2rad(5.0), n_steps)
        raw = Rotation.from_homochoric(_cubochoric_to_homochoric(grid)).data
        ours = _rodrigues_four_vectors(raw)
        zero = np.linalg.norm(raw[:, 1:], axis=1) == 0
        assert zero.sum() == 1
        np.testing.assert_allclose(ours[~zero], ro[~zero], atol=EMSOFT_TEXT_TOL, rtol=0)
        np.testing.assert_allclose(
            ours[zero, 3], ro[zero, 3], atol=EMSOFT_TEXT_TOL, rtol=0
        )

    @pytest.mark.parametrize("n_steps", [6, 20])
    def test_binary_n20_matches_in_order(self, emsoft_program, n_steps):
        run_dir, relative = emsoft_program.new_run_dir()
        axis = ", ".join(f"{v:.16e}" for v in CENTER_AXIS)
        names = {
            key: f"{relative}/ball_n{n_steps}_{key}.txt" for key in ("qu", "eu", "ro")
        }
        namelist = "\n".join(
            [
                " &RFZlist",
                " samplemode = 'MIS',",
                " pgnum = 32,",
                f" nsteps = {n_steps},",
                " maxmisor = 5.0,",
                f" rodrigues = {axis}, {CENTER_TAN_HALF_ANGLE:.16e},",
                f" quoutname = '{names['qu']}',",
                f" euoutname = '{names['eu']}',",
                f" rooutname = '{names['ro']}',",
                " /",
                "",
            ]
        )
        result = emsoft_program("EMsampleRFZ", namelist, run_dir, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr

        data_root = emsoft_program.data_root
        qu = _read_emsoft_orientation_file(data_root / names["qu"])
        eu = _read_emsoft_orientation_file(data_root / names["eu"])
        ro = _read_emsoft_orientation_file(data_root / names["ro"])
        n = (2 * n_steps + 1) ** 3
        assert qu.shape == (n, 4)
        assert eu.shape == (n, 3)
        assert ro.shape == (n, 4)

        ball = misorientation_ball(_center_rotation(), max_angle=5.0, n_steps=n_steps)
        ours = _with_positive_scalar(ball.data)
        np.testing.assert_allclose(ours, qu, atol=EMSOFT_TEXT_TOL, rtol=0)
        self._assert_unshifted_grid_matches(ro, n_steps)

        from_eu = Rotation.from_euler(np.deg2rad(eu)).data
        assert _misorientation_rad(from_eu, ours).max() <= 2e-8
