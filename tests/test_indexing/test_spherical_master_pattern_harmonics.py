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

"""Tests of
``kikuchipy.indexing._spherical._master_pattern_harmonics``.

Covers the "Harmonics" assertions of
``specs/2026-08-16-sht-master-spectra-and-file/validation.md``:

- The container: a copied, C-contiguous, square and lower triangle
  free coefficient array.
- ``_resize_lambert`` against closed form oracles for the pad and the
  crop branch, including the negative control that
  ``scipy.fft.idctn`` does *not* give a constant.
- ``_to_legendre``: bilinear interpolation is exact for bilinear
  functions, the row and column order is locked with an asymmetric
  probe, and both hemispheres sample the same square position.
- The two normalisations of ``emsphinx_compatible``, and
  ``normalize=False`` through the public keyword, the one path on
  which the discrete cosine transform amplitude factor is observable.
- The integer multi-site guard and the energy weights, synthetic and
  real.
- **mp2sht parity** against the shipped ``ni_small_20kv_bw384.sht``
  at rel-L2 < 1e-6, and its weekly twin on the cached full master.
- ``to_master_pattern`` including the deliberate one pixel axis
  offset difference against kikuchipy's EMsoft reader.
- ``save``/``from_file`` round trips, ``preserve_header`` byte
  identity, the losslessness guard in both ``strict`` settings and
  the flag-ambiguous point group trap.
- ``resize``, ``remove_dc``, ``power_spectrum`` (with a Parseval
  oracle independent of its formula), ``describe``, ``rotate`` and
  ``__repr__``.
"""

import dataclasses
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import warnings

import h5py
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.indexing._spherical import (
    _grid,
    _sht,
    _sht_file,
    _symmetry,
)
from kikuchipy.indexing._spherical import (
    _master_pattern_harmonics as _mph,
)

MasterPatternHarmonics = _mph.MasterPatternHarmonics

NI_SMALL = "emsphinx/ni_small_20kv_bw384.sht"
NI_FULL = "emsphinx/ni_20kv_bw384.sht"

SQRT_FOUR_PI = np.sqrt(4 * np.pi)


def _data_path(name: str) -> Path:
    """Return the path of an in-package ``.sht`` file."""
    return Path(Dataset(name).fetch_file_path())


def _emsphinx_program(name: str) -> Path:
    """Return an EMSphInx program, skipping if it is not available."""
    value = os.environ.get("KIKUCHIPY_EMSPHINX_DIR")
    if not value:
        pytest.skip(
            "KIKUCHIPY_EMSPHINX_DIR is not set; set it to an EMSphInx "
            "checkout with build/Release/{mp2sht,sht2png} to run this test"
        )
    directory = Path(value) / "build" / "Release"
    for candidate in (directory / f"{name}.exe", directory / name):
        if candidate.is_file():
            return candidate
    pytest.skip(f"{name} not built in {directory}")


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Return the relative L2 difference of two coefficient arrays."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _file_harmonics(name: str) -> np.ndarray:
    """Return the coefficients stored in an in-package ``.sht``
    file, unpacked with the codec alone.
    """
    sht = _sht_file.read_sht(_data_path(name))
    return _sht_file.unpack_harmonics(
        sht.harmonics.packed,
        sht.harmonics.bandwidth,
        sht.harmonics.z_rot,
        sht.harmonics.flags,
    )


def _ni_master() -> "kp.signals.EBSDMasterPattern":
    """Return the in-package 401 px Ni master pattern, both
    hemispheres, square Lambert.
    """
    return kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )


def _constant_master(
    side: int = 21, value: float = 5.0
) -> "kp.signals.EBSDMasterPattern":
    """Return a synthetic constant master pattern of both
    hemispheres.
    """
    data = np.full((2, side, side), value, dtype=np.float32)
    return kp.signals.EBSDMasterPattern(
        data,
        projection="lambert",
        hemisphere="both",
        phase=Phase(name="c", space_group=225),
    )


def _two_energy_master() -> "kp.signals.EBSDMasterPattern":
    """Return a synthetic master pattern with two energies.

    The axes are set through ``axes_manager._axes``, which is in
    array order: the energy axis is array axis 1, while
    ``axes_manager[1]`` is the *hemisphere* axis, because the
    navigation axes come in natural, i.e. reversed, order. It happens
    not to matter here, both navigation axes being of size two, but
    naming the wrong axis "energy" would hide a reader which picks
    the energy axis by name.
    """
    data = np.ones((2, 2, 21, 21), dtype=np.float32)
    master = kp.signals.EBSDMasterPattern(
        data,
        projection="lambert",
        hemisphere="both",
        phase=Phase(name="x", space_group=225),
    )
    axes = master.axes_manager._axes
    assert master.axes_manager[1] is axes[0]
    axes[0].name = "hemisphere"
    axes[1].name = "energy"
    axes[1].units = "keV"
    axes[1].scale = 1.0
    axes[1].offset = 19.0
    axes[2].name = "height"
    axes[3].name = "width"
    return master


@lru_cache(maxsize=None)
def _ni_harmonics_from_file(name: str = NI_SMALL) -> "MasterPatternHarmonics":
    """Return the coefficients of an in-package ``.sht`` file.

    Memoized and called inside the test bodies rather than made a
    fixture, so that an unimplemented stub fails the test instead of
    erroring its setup.
    """
    return MasterPatternHarmonics.from_file(_data_path(name))


@lru_cache(maxsize=None)
def _ni_harmonics_bw384(emsphinx_compatible: bool = True) -> "MasterPatternHarmonics":
    """Return the coefficients of the in-package Ni master at the
    ``mp2sht`` bandwidth.

    The resolution warning is silenced here and asserted by
    ``TestPipelineWarningsAndErrors``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _ni_master().get_spherical_harmonics(
            bandwidth=384, emsphinx_compatible=emsphinx_compatible
        )


def _antisymmetric_harmonics() -> "MasterPatternHarmonics":
    """Return coefficients whose synthesis tells the two hemispheres
    apart.

    A single zonal entry of odd degree, ``Y_1^0``, which is
    proportional to ``cos(theta)``: positive at the north pole,
    negative at the south one and antisymmetric about the equator.
    Nickel cannot serve here, since its north and south hemispheres
    are bit-identical, so every Ni assertion against ``data[0]`` and
    ``data[1]`` of the same call survives a hemisphere swap.
    """
    alm = np.zeros((4, 4), dtype=np.complex128)
    alm[0, 1] = 1.0
    return MasterPatternHarmonics(
        alm,
        phase=Phase(name="x", point_group="-1"),
        beam_energy=20.0,
        sample_tilt=70.0,
    )


def _legendre_alm(bandwidth: int = 16) -> np.ndarray:
    """Return a fixed band limited spectrum of a real function."""
    rng = np.random.default_rng(0)
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for order in range(bandwidth):
        for degree in range(order, bandwidth):
            if order == 0:
                alm[order, degree] = rng.uniform(-1, 1)
            else:
                alm[order, degree] = rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)
    return alm


# The five fields validation.md line 48 exempts from the field by
# field equality of a written file against the mp2sht one
_DELIBERATE_DIFFERENCES = frozenset(
    {"software_version", "doi", "notes", "emsoft_version", "material_name"}
)


def _comparable_fields(instance) -> list[str]:
    """Return the field names of a codec dataclass a faithful writer
    must reproduce.

    Everything except the five deliberate differences and the raw
    ``*_bytes`` and ``*_len`` twins, which are this codec's way of
    keeping the padding of a file it read and not fields of the C++
    format.
    """
    names = []
    for field in dataclasses.fields(instance):
        if field.name in _DELIBERATE_DIFFERENCES:
            continue
        if field.name.endswith("_bytes") or field.name.endswith("_len"):
            continue
        names.append(field.name)
    return names


def _assert_same_field(ours, theirs, path: str) -> None:
    """Assert two codec field values are equal, recursing into
    dataclasses and sequences.
    """
    if isinstance(ours, float) or isinstance(theirs, float):
        assert ours == pytest.approx(theirs, rel=1e-6, nan_ok=True), path
    elif dataclasses.is_dataclass(ours):
        assert dataclasses.is_dataclass(theirs), path
        for name in _comparable_fields(ours):
            _assert_same_field(
                getattr(ours, name), getattr(theirs, name), f"{path}.{name}"
            )
    elif isinstance(ours, (tuple, list)):
        assert len(ours) == len(theirs), path
        for i, (a, b) in enumerate(zip(ours, theirs)):
            _assert_same_field(a, b, f"{path}[{i}]")
    else:
        assert ours == theirs, path


class TestContainer:
    def test_the_coefficients_are_copied_and_contiguous(self):
        alm = np.zeros((8, 8), dtype=np.complex64, order="F")
        alm[0, 0] = SQRT_FOUR_PI
        harmonics = MasterPatternHarmonics(alm)
        assert harmonics.alm.dtype == np.complex128
        assert harmonics.alm.flags.c_contiguous
        assert harmonics.alm is not alm
        alm[0, 0] = 0
        assert harmonics.alm[0, 0] == SQRT_FOUR_PI

    def test_the_bandwidth_is_the_side_length(self):
        assert MasterPatternHarmonics(np.zeros((17, 17), np.complex128)).bandwidth == 17

    @pytest.mark.parametrize("shape", [(4, 5), (4,), (2, 2, 2)])
    def test_a_non_square_or_three_dimensional_array_raises(self, shape):
        with pytest.raises(ValueError):
            MasterPatternHarmonics(np.zeros(shape, dtype=np.complex128))

    def test_a_non_zero_lower_triangle_raises(self):
        # The packer relies on this padding being zero
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[3, 1] = 1
        with pytest.raises(ValueError):
            MasterPatternHarmonics(alm)

    def test_a_bandwidth_below_one_raises(self):
        with pytest.raises(ValueError):
            MasterPatternHarmonics(np.zeros((0, 0), dtype=np.complex128))


class TestResizeLambert:
    @pytest.mark.parametrize(
        "dim, new_dim", [(13, 21), (21, 13), (401, 547), (1001, 547)]
    )
    def test_a_constant_scales_by_the_ported_factor(self, dim, new_dim):
        # The 0.5 / new_dim ** 2 factor of master.hpp lines 365-367 is
        # amplitude preserving only when new_dim ** 2 == 2 dim ** 2; it
        # cancels in the normalisation and is ported verbatim
        value = 3.7
        image = np.full((dim, dim), value, dtype=np.float64)
        out = _mph._resize_lambert(image, new_dim)
        expected = value * 2 * dim**2 / new_dim**2
        assert out.shape == (new_dim, new_dim)
        assert np.allclose(out, expected, rtol=1e-12, atol=0)

    def test_a_low_order_mode_upsamples_to_the_same_mode(self):
        dim, new_dim = 21, 31
        kx, ky = 2, 3
        n = np.arange(dim)
        mode = np.outer(
            np.cos(np.pi * ky * (n + 0.5) / dim),
            np.cos(np.pi * kx * (n + 0.5) / dim),
        )
        out = _mph._resize_lambert(mode, new_dim)
        n2 = np.arange(new_dim)
        expected = (2 * dim**2 / new_dim**2) * np.outer(
            np.cos(np.pi * ky * (n2 + 0.5) / new_dim),
            np.cos(np.pi * kx * (n2 + 0.5) / new_dim),
        )
        assert np.allclose(out, expected, rtol=0, atol=1e-12)

    def test_an_equal_side_returns_the_input_unchanged(self):
        # The early return of master.hpp line 356. Without it a
        # constant would come back doubled, since the factor is
        # 2 dim ** 2 / new_dim ** 2 == 2 at new_dim == dim
        image = np.full((21, 21), 3.7, dtype=np.float64)
        out = _mph._resize_lambert(image, 21)
        assert np.array_equal(out, image)

    def test_the_idctn_negative_control(self):
        # Re-measured 2026-08-16: scipy.fft.idctn(type=3) does not
        # give a constant at all, so a swap to it is caught. Its
        # maximum ratio to the correct result is exactly
        # 1 / new_dim ** 2, not 1 / (2 new_dim) ** 2, because only
        # X[0, 0] is non-zero and idctn divides that term by N ** 2
        from scipy.fft import dctn, idctn

        dim, new_dim, value = 13, 21, 3.7
        image = np.full((dim, dim), value, dtype=np.float64)
        spectrum = dctn(image, type=2, workers=1)
        padded = np.zeros((new_dim, new_dim), dtype=np.float64)
        padded[:dim, :dim] = spectrum[:dim, :dim]
        wrong = idctn(padded, type=3, workers=1) * 0.5 / new_dim**2
        correct = _mph._resize_lambert(image, new_dim)
        assert np.allclose(correct, value * 2 * dim**2 / new_dim**2, rtol=1e-12)
        assert np.ptp(wrong) > 1e-3
        assert (wrong.max() / correct.max()) == pytest.approx(1 / new_dim**2, rel=1e-6)


class TestToLegendre:
    def test_a_constant_image_stays_constant(self):
        dim, dim_legendre = 31, 11
        north = np.full((dim, dim), 2.5)
        south = np.full((dim, dim), -1.25)
        out_north, out_south = _mph._to_legendre(north, south, dim_legendre)
        assert out_north.shape == (dim_legendre, dim_legendre)
        assert np.allclose(out_north, 2.5, rtol=0, atol=1e-14)
        assert np.allclose(out_south, -1.25, rtol=0, atol=1e-14)

    def test_a_plane_is_reproduced_exactly(self):
        # Bilinear interpolation is exact for bilinear functions, so
        # a plane in the square coordinates comes back exactly
        dim, dim_legendre = 547, 387
        grid = np.arange(dim) / (dim - 1)
        plane = 0.3 * grid[np.newaxis, :] - 0.7 * grid[:, np.newaxis] + 0.1
        out_north, _ = _mph._to_legendre(plane, plane, dim_legendre)
        normals = _grid.legendre_normals(dim_legendre)
        square = _grid.sphere_to_square(normals.reshape(-1, 3))
        expected = (0.3 * square[:, 0] - 0.7 * square[:, 1] + 0.1).reshape(
            dim_legendre, dim_legendre
        )
        assert np.allclose(out_north, expected, rtol=0, atol=1e-12)

    def test_the_row_and_column_order_is_locked(self):
        # Against the *absolute* mapping: f = X must come back as the
        # X column of sphere_to_square and f = Y as its Y column. The
        # square Legendre grid has max |X - Y.T| = 1.1e-16 (measured
        # 2026-08-16), so asserting out_x == out_y.T merely restates
        # that and holds just as well for a transposed (X, Y) in the
        # bilinear step, the injection of plan.md line 68
        dim, dim_legendre = 101, 31
        grid = np.arange(dim) / (dim - 1)
        image_x = np.broadcast_to(grid[np.newaxis, :], (dim, dim)).copy()
        image_y = np.broadcast_to(grid[:, np.newaxis], (dim, dim)).copy()
        out_x, _ = _mph._to_legendre(image_x, image_x, dim_legendre)
        out_y, _ = _mph._to_legendre(image_y, image_y, dim_legendre)
        square = _grid.sphere_to_square(
            _grid.legendre_normals(dim_legendre).reshape(-1, 3)
        ).reshape(dim_legendre, dim_legendre, 2)
        assert np.allclose(out_x, square[..., 0], rtol=0, atol=1e-12)
        assert np.allclose(out_y, square[..., 1], rtol=0, atol=1e-12)
        assert not np.allclose(out_x, out_y)

    def test_the_two_hemispheres_use_the_same_square_position(self):
        dim, dim_legendre = 101, 31
        rng = np.random.default_rng(0)
        first = rng.uniform(size=(dim, dim))
        second = rng.uniform(size=(dim, dim))
        a_north, a_south = _mph._to_legendre(first, second, dim_legendre)
        b_north, b_south = _mph._to_legendre(second, first, dim_legendre)
        assert np.array_equal(a_north, b_south)
        assert np.array_equal(a_south, b_north)


class TestNormalizeHemispheres:
    @staticmethod
    def _weights(dim_legendre: int, corner_factor: float) -> np.ndarray:
        """Return the per-pixel weights with the borders halved and
        the corners scaled by ``corner_factor``.
        """
        rings = _grid.ring_number(dim_legendre)
        weights = _grid.ring_solid_angles(dim_legendre, "legendre")[rings]
        weights[0] /= 2
        weights[-1] /= 2
        weights[:, 0] /= 2
        weights[:, -1] /= 2
        if corner_factor != 0.25:
            for i in (0, -1):
                for j in (0, -1):
                    weights[i, j] *= corner_factor / 0.25
        return weights

    def test_the_corrected_setting_gives_zero_mean_and_unit_variance(self):
        dim_legendre = 31
        rng = np.random.default_rng(1)
        north = rng.uniform(size=(dim_legendre, dim_legendre))
        south = rng.uniform(size=(dim_legendre, dim_legendre))
        out_north, out_south, _, _ = _mph._normalize_hemispheres(
            north, south, emsphinx_compatible=False
        )
        weights = self._weights(dim_legendre, 0.5)
        total = weights.sum()
        mean = (np.sum(weights * out_north) + np.sum(weights * out_south)) / (2 * total)
        variance = (np.sum(weights * out_north**2) + np.sum(weights * out_south**2)) / (
            2 * total
        )
        assert mean == pytest.approx(0, abs=1e-12)
        assert variance == pytest.approx(1, rel=1e-12)

    def test_the_compatible_setting_subtracts_twice_the_mean(self):
        dim_legendre = 31
        rng = np.random.default_rng(2)
        north = rng.uniform(size=(dim_legendre, dim_legendre))
        south = rng.uniform(size=(dim_legendre, dim_legendre))
        out_north, out_south, mean, std = _mph._normalize_hemispheres(
            north, south, emsphinx_compatible=True
        )
        weights = self._weights(dim_legendre, 0.25)
        total = weights.sum()
        weighted_mean = (np.sum(weights * out_north) + np.sum(weights * out_south)) / (
            2 * total
        )
        # The pattern was shifted by 2 mu, so the weighted mean of the
        # result is -mu / sigma and not zero
        assert weighted_mean == pytest.approx(-mean / std, rel=1e-10)

    def test_the_two_settings_differ_by_a_global_factor_and_a_shift(self):
        # Both settings are affine in the input, so
        # plain == compat * (std / std2) + (2 mu - mu2) / std2
        # exactly, with the scalars each setting returns. The identity
        # is written that way and not as
        # (compat + mu / std) * sqrt(1 + mu2 ** 2 / std2 ** 2) because
        # the two settings weight the corners differently (omega / 4
        # against omega / 2, D7), so mu != mu2 and the shorter form is
        # only true to 3.3e-4 here, not to 1e-10 (measured 2026-08-16;
        # validation.md line 41 corrected). Subtracting the mean once
        # instead of twice under emsphinx_compatible=True breaks this
        dim_legendre = 31
        rng = np.random.default_rng(3)
        north = rng.uniform(size=(dim_legendre, dim_legendre))
        south = rng.uniform(size=(dim_legendre, dim_legendre))
        compat_north, compat_south, mean, std = _mph._normalize_hemispheres(
            north, south, emsphinx_compatible=True
        )
        plain_north, plain_south, mean2, std2 = _mph._normalize_hemispheres(
            north, south, emsphinx_compatible=False
        )
        shift = (2 * mean - mean2) / std2
        assert np.allclose(
            compat_north * (std / std2) + shift, plain_north, rtol=0, atol=1e-12
        )
        assert np.allclose(
            compat_south * (std / std2) + shift, plain_south, rtol=0, atol=1e-12
        )
        assert mean != mean2


class TestEnergyWeights:
    def test_a_single_energy_master_gives_one(self):
        weights = _mph._energy_weights(_ni_master(), None)
        assert np.array_equal(weights, np.array([1.0]))

    def test_synthetic_accumulated_energies(self, tmp_path):
        fpath = tmp_path / "mc.h5"
        counts = np.zeros((5, 5, 4), dtype=np.int32)
        counts[0, 0] = [10, 0, 30, 60]
        with h5py.File(fpath, "w") as f:
            f.create_dataset("EMData/MCOpenCL/accum_e", data=counts)
        weights = _mph._accum_e_weights(
            str(fpath), np.array([10.0, 11.0, 12.0, 13.0]), 10.0, 1.0
        )
        assert np.allclose(weights, [0.1, 0.0, 0.3, 0.6], rtol=0, atol=1e-15)

    def test_an_energy_subset_is_renormalized(self, tmp_path):
        # EMSphInx refuses a bin count mismatch (master.hpp line 319);
        # we renormalise over the loaded bins, a recorded deviation
        fpath = tmp_path / "mc.h5"
        counts = np.zeros((5, 5, 4), dtype=np.int32)
        counts[0, 0] = [10, 0, 30, 60]
        with h5py.File(fpath, "w") as f:
            f.create_dataset("EMData/MCOpenCL/accum_e", data=counts)
        weights = _mph._accum_e_weights(str(fpath), np.array([12.0, 13.0]), 10.0, 1.0)
        assert np.allclose(weights, [1 / 3, 2 / 3], rtol=1e-12, atol=0)

    def test_an_energy_outside_the_histogram_raises(self, tmp_path):
        fpath = tmp_path / "mc.h5"
        counts = np.zeros((5, 5, 4), dtype=np.int32)
        counts[0, 0] = [10, 0, 30, 60]
        with h5py.File(fpath, "w") as f:
            f.create_dataset("EMData/MCOpenCL/accum_e", data=counts)
        with pytest.raises(ValueError, match="energy_weights"):
            _mph._accum_e_weights(str(fpath), np.array([9.0]), 10.0, 1.0)

    def test_a_missing_dataset_raises(self, tmp_path):
        fpath = tmp_path / "empty.h5"
        with h5py.File(fpath, "w") as f:
            f.create_group("EMData")
        with pytest.raises(ValueError, match="energy_weights"):
            _mph._accum_e_weights(str(fpath), np.array([20.0]), 20.0, 1.0)

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="energy_weights"):
            _mph._accum_e_weights(
                str(tmp_path / "nope.h5"), np.array([20.0]), 20.0, 1.0
            )

    def test_explicit_weights_are_normalized(self):
        master = _ni_master()
        weights = _mph._energy_weights(master, [1])
        assert np.allclose(weights, [1.0])

    def test_negative_explicit_weights_raise(self, emsoft_ebsd_master_pattern_file):
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        with pytest.raises(ValueError):
            _mph._energy_weights(master, [-1] + [2] * 10)

    def test_explicit_weights_of_the_wrong_length_raise(self):
        with pytest.raises(ValueError):
            _mph._energy_weights(_ni_master(), [1, 3])

    def test_a_two_element_weight_set_is_normalized(self):
        master = _two_energy_master()
        weights = _mph._energy_weights(master, [1, 3])
        assert np.allclose(weights, [0.25, 0.75], rtol=0, atol=1e-15)

    def test_negative_weights_on_a_two_energy_master_raise(self):
        with pytest.raises(ValueError):
            _mph._energy_weights(_two_energy_master(), [-1, 2])

    def test_a_multi_energy_master_without_accum_e_raises(
        self, emsoft_ebsd_master_pattern_file
    ):
        # master_patterns.h5 has 11 energies and no accum_e
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        with pytest.raises(ValueError, match="energy_weights"):
            MasterPatternHarmonics.from_master_pattern(master, bandwidth=4)

    def test_a_multi_energy_master_with_explicit_weights_runs(
        self, emsoft_ebsd_master_pattern_file
    ):
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master, bandwidth=4, energy_weights=np.ones(11)
        )
        assert harmonics.bandwidth == 4


class TestIntegerMultiSiteGuard:
    @staticmethod
    def _master(dtype, natomtypes, combinesites):
        data = np.ones((2, 21, 21), dtype=dtype)
        master = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="both",
            phase=Phase(name="x", space_group=225),
        )
        metadata = {"CrystalData": {"Natomtypes": natomtypes}}
        if combinesites is not None:
            metadata["EBSDMasterNameList"] = {"combinesites": combinesites}
        metadata["MCCLNameList"] = {"EkeV": 20.0, "sig": 70.0}
        master.original_metadata.add_dictionary(metadata)
        return master

    def test_an_integer_two_site_master_raises(self):
        master = self._master(np.uint8, 2, 0)
        with pytest.raises(ValueError, match="combinesites"):
            MasterPatternHarmonics.from_master_pattern(master, bandwidth=4)

    def test_combinesites_true_passes(self):
        master = self._master(np.uint8, 2, 1)
        assert (
            MasterPatternHarmonics.from_master_pattern(master, bandwidth=4).bandwidth
            == 4
        )

    def test_a_float_two_site_master_passes(self):
        master = self._master(np.float32, 2, 0)
        assert (
            MasterPatternHarmonics.from_master_pattern(master, bandwidth=4).bandwidth
            == 4
        )

    def test_a_master_without_the_keys_passes(self):
        data = np.ones((2, 21, 21), dtype=np.uint8)
        master = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="both",
            phase=Phase(name="x", space_group=225),
        )
        assert (
            MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=4, beam_energy=20.0, sample_tilt=70.0
            ).bandwidth
            == 4
        )

    def test_the_in_package_nickel_master_passes(self):
        # Natomtypes 1, so the file dtype sum is exact
        assert (
            MasterPatternHarmonics.from_master_pattern(
                _ni_master(), bandwidth=4
            ).bandwidth
            == 4
        )


class TestPipelineWarningsAndErrors:
    def test_the_default_bandwidth_warns_on_a_401_px_master(self):
        with pytest.warns(UserWarning, match="384"):
            MasterPatternHarmonics.from_master_pattern(_ni_master(), bandwidth=384)

    def test_the_warning_names_the_resolution_limit(self):
        with pytest.warns(UserWarning) as record:
            MasterPatternHarmonics.from_master_pattern(_ni_master(), bandwidth=384)
        message = " ".join(str(w.message) for w in record)
        assert "384" in message
        assert "200" in message

    def test_a_bandwidth_within_the_resolution_does_not_warn(self, recwarn):
        MasterPatternHarmonics.from_master_pattern(_ni_master(), bandwidth=200)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize("bandwidth", [0, -1, 40000])
    def test_an_impossible_bandwidth_raises(self, bandwidth):
        with pytest.raises(ValueError):
            MasterPatternHarmonics.from_master_pattern(
                _ni_master(), bandwidth=bandwidth
            )

    def test_the_thirteen_pixel_master_runs_at_bandwidth_four(
        self, emsoft_ebsd_master_pattern_file
    ):
        # dim_leg 7, dim_scaled 10: the crop branch of _resize_lambert
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master, bandwidth=4, energy_weights=np.ones(11)
        )
        assert harmonics.bandwidth == 4

    def test_the_thirteen_pixel_master_warns_at_bandwidth_eight(
        self, emsoft_ebsd_master_pattern_file
    ):
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        with pytest.warns(UserWarning):
            MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=8, energy_weights=np.ones(11)
            )


class TestNormalizeFalse:
    def test_it_equals_an_in_test_analyze_of_the_regridded_hemispheres(
        self, emsoft_ebsd_master_pattern_file
    ):
        master = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            hemisphere="both",
        )
        bandwidth = 4
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master,
            bandwidth=bandwidth,
            energy_weights=np.ones(11),
            normalize=False,
        )
        weights = np.ones(11) / 11
        data = np.asarray(master.data, dtype=np.float64)
        north = np.tensordot(weights, data[0], axes=(0, 0))
        south = np.tensordot(weights, data[1], axes=(0, 0))
        dim_legendre = _grid.default_dim(bandwidth, "legendre")
        dim_scaled = int(round(np.sqrt(2) * dim_legendre))
        north = _mph._resize_lambert(north, dim_scaled)
        south = _mph._resize_lambert(south, dim_scaled)
        north, south = _mph._to_legendre(north, south, dim_legendre)
        expected = _sht.SphericalHarmonicTransform(
            bandwidth, "legendre", dim_legendre
        ).analyze(north, south)
        assert np.array_equal(harmonics.alm, expected)

    def test_a_constant_master_shows_the_amplitude_factor(self):
        # bandwidth 8: dim_leg 11, dim_scaled 16, so the factor
        # 2 * 21 ** 2 / 16 ** 2 is observable end to end. This is the
        # only path on which it is
        value, side, bandwidth = 5.0, 21, 8
        master = _constant_master(side, value)
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master,
            bandwidth=bandwidth,
            normalize=False,
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        dim_scaled = int(round(np.sqrt(2) * _grid.default_dim(bandwidth, "legendre")))
        assert dim_scaled == 16
        expected = SQRT_FOUR_PI * value * 2 * side**2 / dim_scaled**2
        assert harmonics.alm[0, 0].real == pytest.approx(expected, rel=1e-10)
        rest = harmonics.alm.copy()
        rest[0, 0] = 0
        assert np.abs(rest).max() < 1e-10

    def test_normalize_true_kills_the_constant(self):
        master = _constant_master(21, 5.0)
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master,
            bandwidth=8,
            normalize=True,
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        assert abs(harmonics.alm[0, 0]) < 1e-10

    def test_the_equal_side_early_return_leaves_the_amplitude_alone(self):
        # bandwidth 12: dim_leg 15, dim_scaled round(sqrt(2) * 15) = 21
        # == dim, so the early return runs and the 2x factor never does
        value, side, bandwidth = 5.0, 21, 12
        master = _constant_master(side, value)
        dim_scaled = int(round(np.sqrt(2) * _grid.default_dim(bandwidth, "legendre")))
        assert dim_scaled == side
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master,
            bandwidth=bandwidth,
            normalize=False,
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        assert harmonics.alm[0, 0].real == pytest.approx(
            SQRT_FOUR_PI * value, rel=1e-12
        )


class TestMp2shtParity:
    """The one external oracle of the whole pipeline."""

    def test_the_coefficients_agree(self, record_property):
        ours, theirs = _ni_harmonics_bw384(), _ni_harmonics_from_file()
        error = _relative_l2(ours.alm, theirs.alm)
        record_property("mp2sht_parity_rel_l2", f"{error:.3e}")
        assert error < 1e-6

    def test_they_agree_after_the_dc_term_is_removed(self, record_property):
        ours, theirs = _ni_harmonics_bw384(), _ni_harmonics_from_file()
        error = _relative_l2(ours.remove_dc().alm, theirs.remove_dc().alm)
        record_property("mp2sht_parity_rel_l2_no_dc", f"{error:.3e}")
        assert error < 1e-6

    def test_the_dc_term_agrees(self, record_property):
        ours, theirs = _ni_harmonics_bw384(), _ni_harmonics_from_file()
        record_property("mp2sht_a00", f"{ours.alm[0, 0].real:.12f}")
        assert ours.alm[0, 0].real == pytest.approx(theirs.alm[0, 0].real, abs=1e-8)
        # The 2x mean quirk of the default emsphinx_compatible=True
        assert ours.alm[0, 0].real == pytest.approx(-2.985, abs=1e-2)

    def test_the_slots_the_file_discards_carry_no_power(self, record_property):
        ours = _ni_harmonics_bw384()
        rotation, mirror = _symmetry.systematic_zero_power(ours.alm, 4, True)
        record_property("mp2sht_structural_zero_power", f"{rotation:.3e}")
        assert rotation < 1e-20
        assert mirror < 1e-20

    def test_the_coefficients_are_real(self):
        ours = _ni_harmonics_bw384()
        assert np.abs(ours.alm.imag).max() < 1e-12

    def test_the_metadata_agrees(self):
        ours = _ni_harmonics_bw384()
        assert ours.beam_energy == pytest.approx(20.1)
        assert ours.sample_tilt == 70
        assert ours.n_fold == 4
        assert ours.has_equatorial_mirror

    def test_the_opt_in_setting_has_almost_no_dc_term(self, record_property):
        harmonics = _ni_harmonics_bw384(emsphinx_compatible=False)
        record_property("opt_in_a00", f"{harmonics.alm[0, 0].real:.3e}")
        assert abs(harmonics.alm[0, 0].real) < 1e-3

    def test_the_opt_in_setting_satisfies_parseval(self):
        harmonics = _ni_harmonics_bw384(emsphinx_compatible=False)
        total = harmonics.power_spectrum().sum()
        assert total == pytest.approx(4 * np.pi, rel=1e-2)

    def test_the_global_factor_between_the_settings_is_recorded(self, record_property):
        theirs = _ni_harmonics_from_file()
        harmonics = _ni_harmonics_bw384(emsphinx_compatible=False)
        mask = np.abs(theirs.alm) > 1e-3
        mask[0, 0] = False
        factor = float(
            np.median(np.abs(harmonics.alm[mask]) / np.abs(theirs.alm[mask]))
        )
        record_property("emsphinx_compatible_false_global_factor", f"{factor:.4f}")
        assert factor == pytest.approx(1.854, rel=1e-3)

    def test_the_two_settings_agree_after_dc_removal_and_rescaling(
        self, record_property
    ):
        # The one place the "differ only by a global factor and the DC
        # term" statement of D7 is exactly true: a constant shift of
        # the function on the sphere lives entirely in a_00, so after
        # remove_dc the two settings differ by the single scalar
        # std_compat / std_plain whatever the corner weights are.
        # One scalar is fitted, the remaining 9311 coefficients are
        # the assertion (recorded 5.1e-9 pre-implementation)
        ours = _ni_harmonics_bw384(emsphinx_compatible=False).remove_dc().alm
        theirs = _ni_harmonics_from_file().remove_dc().alm
        factor = float(np.linalg.norm(ours) / np.linalg.norm(theirs))
        error = _relative_l2(ours / factor, theirs)
        record_property("opt_in_rel_l2_after_dc_and_rescale", f"{error:.3e}")
        record_property("opt_in_norm_ratio", f"{factor:.4f}")
        assert factor == pytest.approx(1.854, rel=1e-3)
        assert error < 1e-8

    @pytest.mark.weekly
    def test_full_master_parity(self, record_property, recwarn):
        try:
            master = kp.data.ebsd_master_pattern(
                "ni", projection="lambert", hemisphere="both", allow_download=False
            )
        except (ValueError, FileNotFoundError) as error:  # pragma: no cover
            pytest.skip(f"the full Ni master is not cached: {error}")
        ours = master.get_spherical_harmonics(bandwidth=384)
        theirs = MasterPatternHarmonics.from_file(_data_path(NI_FULL))
        error_all = _relative_l2(ours.alm, theirs.alm)
        error_dc = _relative_l2(ours.remove_dc().alm, theirs.remove_dc().alm)
        record_property("full_master_rel_l2", f"{error_all:.3e}")
        record_property("full_master_rel_l2_no_dc", f"{error_dc:.3e}")
        assert error_all < 1e-6
        assert error_dc < 1e-6
        assert ours.alm[0, 0].real == pytest.approx(theirs.alm[0, 0].real, abs=1e-8)
        # A 1001 px master carries bandwidth 500, so no warning
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.weekly
    def test_full_master_energy_weights(self, record_property):
        try:
            master = kp.data.ebsd_master_pattern(
                "ni", projection="lambert", hemisphere="both", allow_download=False
            )
        except (ValueError, FileNotFoundError) as error:  # pragma: no cover
            pytest.skip(f"the full Ni master is not cached: {error}")
        weights = _mph._energy_weights(master, None)
        counts = np.array(
            [
                0,
                0,
                1455,
                266084,
                5365299,
                18896238,
                27431857,
                34257097,
                42689288,
                53576717,
                68256443,
                89645683,
                124471882,
                191778041,
                334039769,
                161050771,
            ],
            dtype=np.int64,
        )
        assert weights.size == 16
        assert weights.sum() == pytest.approx(1.0, rel=1e-12)
        assert np.allclose(weights, counts / counts.sum(), rtol=1e-12, atol=0)
        assert weights[-1] == pytest.approx(0.1398342, rel=1e-6)
        record_property("full_master_energy_total", str(int(counts.sum())))


class TestToMasterPattern:
    def test_the_default_grid_and_axes(self):
        harmonics = _ni_harmonics_from_file()
        master = harmonics.to_master_pattern()
        assert master.data.shape == (2, 769, 769)
        assert master.data.dtype == np.float64
        assert master.projection == "lambert"
        assert master.hemisphere == "both"
        names = [ax.name for ax in master.axes_manager._axes]
        assert names == ["hemisphere", "height", "width"]
        offsets = [ax.offset for ax in master.axes_manager._axes]
        assert offsets == [0, -384, -384]
        assert master.axes_manager["height"].units == "px"
        assert master.phase is harmonics.phase
        assert master.original_metadata.as_dictionary() == (harmonics.original_metadata)

    def test_the_axis_offset_convention_is_one_pixel_off_the_emsoft_reader(self):
        harmonics = _ni_harmonics_from_file()
        # Deliberate and recorded: we centre on the middle pixel,
        # -(dim // 2), as the ebsdsim reader does, while kikuchipy's
        # EMsoft reader writes -dim // 2. Nothing computes with these
        with pytest.warns(UserWarning):
            master = harmonics.to_master_pattern(dim=401)
        assert master.axes_manager["height"].offset == -200
        assert _ni_master().axes_manager["height"].offset == -201

    def test_the_north_hemisphere_matches_the_source(self, record_property):
        harmonics = _ni_harmonics_from_file()
        from scipy.ndimage import map_coordinates

        master = harmonics.to_master_pattern()
        source = np.asarray(_ni_master().data[0], dtype=np.float64)
        grid = np.mgrid[0:769, 0:769] * (400 / 768)
        upsampled = map_coordinates(source, grid, order=1)
        r = float(np.corrcoef(master.data[0].ravel(), upsampled.ravel())[0, 1])
        record_property("to_master_pattern_r_769", f"{r:.5f}")
        assert r > 0.99

    def test_a_smaller_grid_warns_and_still_correlates(self, record_property):
        harmonics = _ni_harmonics_from_file()
        with pytest.warns(UserWarning, match="200"):
            master = harmonics.to_master_pattern(dim=401)
        source = np.asarray(_ni_master().data[0], dtype=np.float64)
        r = float(np.corrcoef(master.data[0].ravel(), source.ravel())[0, 1])
        record_property("to_master_pattern_r_401", f"{r:.5f}")
        assert r > 0.95

    @pytest.mark.parametrize("dim", [768, 1])
    def test_an_invalid_grid_raises(self, dim):
        harmonics = _ni_harmonics_from_file()
        with pytest.raises(ValueError):
            harmonics.to_master_pattern(dim=dim)

    def test_the_upper_hemisphere(self):
        harmonics = _ni_harmonics_from_file()
        with pytest.warns(UserWarning):
            both = harmonics.to_master_pattern(dim=401)
        with pytest.warns(UserWarning):
            upper = harmonics.to_master_pattern(dim=401, hemisphere="upper")
        assert upper.data.shape == (401, 401)
        assert np.array_equal(upper.data, both.data[0])
        assert upper.hemisphere == "upper"

    def test_the_lower_hemisphere(self):
        harmonics = _ni_harmonics_from_file()
        with pytest.warns(UserWarning):
            both = harmonics.to_master_pattern(dim=401)
        with pytest.warns(UserWarning):
            lower = harmonics.to_master_pattern(dim=401, hemisphere="lower")
        assert np.array_equal(lower.data, both.data[1])
        assert lower.hemisphere == "lower"

    def test_an_unknown_hemisphere_raises_and_lists_the_options(self):
        harmonics = _ni_harmonics_from_file()
        with pytest.raises(ValueError) as info:
            harmonics.to_master_pattern(hemisphere="north")
        message = str(info.value)
        for option in ("upper", "lower", "both"):
            assert option in message

    def test_the_two_hemispheres_are_identical_for_nickel(self):
        harmonics = _ni_harmonics_from_file()
        master = harmonics.to_master_pattern()
        assert np.array_equal(master.data[0], master.data[1])

    def test_the_hemisphere_order_is_absolute(self):
        # The Ni assertions above are blind to a hemisphere swap,
        # since north == south there. Y_1^0 is not
        harmonics = _antisymmetric_harmonics()
        master = harmonics.to_master_pattern()
        assert master.data.shape == (2, 9, 9)
        north, south = master.data
        assert not np.array_equal(north, south)
        assert np.allclose(south, -north, rtol=0, atol=1e-12)
        # The centre pixel of a square Lambert grid is the pole, and
        # Y_1^0 is positive at the north one
        assert north[4, 4] > 0
        assert south[4, 4] < 0

    def test_the_hemisphere_keyword_picks_the_right_pole(self):
        harmonics = _antisymmetric_harmonics()
        upper = harmonics.to_master_pattern(hemisphere="upper")
        lower = harmonics.to_master_pattern(hemisphere="lower")
        assert upper.data.shape == (9, 9)
        assert upper.data[4, 4] > 0
        assert lower.data[4, 4] < 0
        assert np.allclose(lower.data, -upper.data, rtol=0, atol=1e-12)

    def test_get_patterns_agrees_with_the_source_master(self, record_property):
        harmonics = _ni_harmonics_from_file()
        rotations = Rotation.from_euler(
            np.deg2rad([[10, 20, 30], [120, 45, 60], [200, 80, 300]])
        )
        detector = kp.detectors.EBSDDetector(
            shape=(60, 60), pc=(0.42, 0.22, 0.50), sample_tilt=70
        )
        ours = harmonics.to_master_pattern().get_patterns(
            rotations, detector, compute=True, show_progressbar=False
        )
        theirs = _ni_master().get_patterns(
            rotations, detector, compute=True, show_progressbar=False
        )
        for i in range(3):
            a = np.asarray(ours.data[i], dtype=np.float64).ravel()
            b = np.asarray(theirs.data[i], dtype=np.float64).ravel()
            ncc = float(np.corrcoef(a, b)[0, 1])
            record_property(f"get_patterns_ncc_{i}", f"{ncc:.5f}")
            assert ncc > 0.99

    def test_the_sht_amendment_is_exercised(self):
        # to_master_pattern needs a Lambert transformer at dim 769,
        # whose quadrature weights cannot be solved; only analyze
        # needs them, so the constructor and synthesize work
        transform = _sht.SphericalHarmonicTransform(384, "lambert", 769)
        assert transform.dim == 769
        with pytest.raises(ValueError):
            transform.analyze(np.ones((769, 769)), np.ones((769, 769)))


class TestSaveAndFromFile:
    def test_a_round_trip_keeps_the_coefficients_bit_exactly(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        fpath = tmp_path / "ni.sht"
        from_master.save(fpath)
        again = MasterPatternHarmonics.from_file(fpath)
        kept = again.alm != 0
        # Without this a writer which stored zeros would compare two
        # empty selections and pass
        assert np.count_nonzero(kept) == _sht_file.num_harmonics(384, 4, 0x7)
        assert np.count_nonzero(kept) == 9312
        assert np.array_equal(again.alm[kept], from_master.alm[kept])

    def test_a_round_trip_keeps_the_metadata(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        fpath = tmp_path / "ni.sht"
        from_master.save(fpath)
        again = MasterPatternHarmonics.from_file(fpath)
        assert again.beam_energy == pytest.approx(20.1, abs=1e-5)
        assert again.sample_tilt == pytest.approx(70.0, abs=1e-5)
        assert again.phase.space_group.number == 225
        assert again.phase.structure.lattice.a == pytest.approx(0.35236, rel=1e-6)
        assert len(again.phase.structure) == 1
        assert again.phase.structure[0].element in (28, "28", "Ni")
        simulation = again.original_metadata["simulations"]["simulation_0"]
        assert simulation["sig_start"] == pytest.approx(70)
        assert simulation["num_sx"] == 201
        assert simulation["num_px"] == 200
        assert simulation["tot_num_el"] == 2_000_000_000
        assert simulation["lat_grid_type"] == 1
        assert simulation["emsoft_version"] == "unknown"

    def test_the_software_version_is_a_nul_padded_kikuchipy_tag(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        fpath = tmp_path / "ni.sht"
        from_master.save(fpath)
        expected = ("kp" + kp.__version__)[:8].encode().ljust(8, b"\x00")
        assert fpath.read_bytes()[8:16] == expected

    def test_a_short_version_is_nul_padded_and_not_space_padded(
        self, tmp_path, monkeypatch
    ):
        from_master = _ni_harmonics_bw384()
        monkeypatch.setattr(kp, "__version__", "1.0.0")
        fpath = tmp_path / "ni.sht"
        from_master.save(fpath)
        assert fpath.read_bytes()[8:16] == b"kp1.0.0\x00"

    @pytest.mark.parametrize(
        "emsphinx_compatible, expected",
        [
            (True, "created with kikuchipy (normalize=True, emsphinx_compatible=True)"),
            (
                False,
                "created with kikuchipy (normalize=True, emsphinx_compatible=False)",
            ),
        ],
    )
    def test_the_provenance_note(self, tmp_path, emsphinx_compatible, expected):
        harmonics = MasterPatternHarmonics.from_master_pattern(
            _ni_master(),
            bandwidth=32,
            emsphinx_compatible=emsphinx_compatible,
        )
        fpath = tmp_path / "ni.sht"
        harmonics.save(fpath)
        sht = _sht_file.read_sht(fpath)
        assert sht.header.notes == expected
        assert sht.header.doi == ""

    def test_a_directly_built_instance_gets_the_plain_note(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = SQRT_FOUR_PI
        harmonics = MasterPatternHarmonics(
            alm,
            phase=Phase(name="ni", space_group=225),
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        fpath = tmp_path / "flat.sht"
        harmonics.save(fpath)
        assert _sht_file.read_sht(fpath).header.notes == "created with kikuchipy"

    def test_the_written_fields_equal_the_mp2sht_ones(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        fpath = tmp_path / "ni.sht"
        from_master.save(fpath)
        ours = _sht_file.read_sht(fpath)
        theirs = _sht_file.read_sht(_data_path(NI_SMALL))
        assert ours.header.modality == theirs.header.modality
        assert ours.header.beam_energy == pytest.approx(
            theirs.header.beam_energy, abs=1e-5
        )
        assert ours.header.primary_angle == pytest.approx(
            theirs.header.primary_angle, abs=1e-5
        )
        assert ours.sg_eff == theirs.sg_eff
        assert ours.pijk == theirs.pijk
        assert ours.rot_sense == theirs.rot_sense
        assert ours.vendor == theirs.vendor
        assert ours.sim_meta_size == theirs.sim_meta_size
        assert ours.crystals[0].lat == pytest.approx(theirs.crystals[0].lat, rel=1e-6)
        assert ours.crystals[0].formula == theirs.crystals[0].formula
        assert (
            ours.crystals[0].atoms[0].atomic_number
            == theirs.crystals[0].atoms[0].atomic_number
        )
        assert ours.harmonics.bandwidth == theirs.harmonics.bandwidth
        assert ours.harmonics.z_rot == theirs.harmonics.z_rot
        assert ours.harmonics.flags == theirs.harmonics.flags
        assert ours.harmonics.doub_cnt == theirs.harmonics.doub_cnt
        # Every remaining header, crystal, atom and simulation field,
        # not the dozen picked above
        _assert_same_field(ours.header, theirs.header, "header")
        _assert_same_field(ours.crystals[0], theirs.crystals[0], "crystals[0]")
        _assert_same_field(ours.simulations[0], theirs.simulations[0], "simulations[0]")
        # All five deliberate differences, none of them assumed
        assert ours.header.software_version != theirs.header.software_version
        assert ours.header.doi != theirs.header.doi
        assert ours.header.notes != theirs.header.notes
        assert ours.simulations[0].emsoft_version != (
            theirs.simulations[0].emsoft_version
        )
        assert ours.crystals[0].material_name != theirs.crystals[0].material_name

    @pytest.mark.parametrize("name", [NI_SMALL, NI_FULL])
    def test_preserve_header_is_byte_identical(self, name, tmp_path):
        source = _data_path(name)
        harmonics = MasterPatternHarmonics.from_file(source)
        fpath = tmp_path / "again.sht"
        harmonics.save(fpath, preserve_header=True)
        assert fpath.read_bytes() == source.read_bytes()

    def test_preserve_header_on_a_generated_fixture(
        self, emsphinx_synthetic_sht_files, tmp_path
    ):
        files = emsphinx_synthetic_sht_files()
        source = files[225] if 225 in files else files[123]
        harmonics = MasterPatternHarmonics.from_file(source)
        fpath = tmp_path / "again.sht"
        harmonics.save(fpath, preserve_header=True)
        assert fpath.read_bytes() == source.read_bytes()

    def test_preserve_header_keeps_non_utf8_bytes(self, tmp_path):
        # The raw padded bytes branch of the writer
        raw = b"caf\xe9\x00\x00\x00\x01"
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        packed = _sht_file.pack_harmonics(alm, 8, 1, 0)
        sht = _sht_file.ShtFile(
            header=_sht_file.ShtHeader(
                software_version="kp-test",
                modality=_sht_file.MODALITY_EBSD,
                beam_energy=20.0,
                primary_angle=70.0,
                notes="caf�",
                note_len=4,
                notes_bytes=raw,
            ),
            num_xtal=1,
            sg_eff=1,
            modality=_sht_file.MODALITY_EBSD,
            crystals=[_sht_file.ShtCrystal(sg_num=1)],
            simulations=[None],
            harmonics=_sht_file.ShtHarmonics(
                bandwidth=8, z_rot=1, flags=0, doub_cnt=packed.size, packed=packed
            ),
        )
        source = tmp_path / "latin.sht"
        _sht_file.write_sht(source, sht)
        harmonics = MasterPatternHarmonics.from_file(source)
        fpath = tmp_path / "again.sht"
        harmonics.save(fpath, preserve_header=True)
        assert fpath.read_bytes() == source.read_bytes()

    def test_preserve_header_on_a_non_file_instance_raises(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        with pytest.raises(ValueError):
            from_master.save(tmp_path / "x.sht", preserve_header=True)

    def test_a_point_group_only_phase_uses_the_fallback(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        harmonics = MasterPatternHarmonics(
            alm,
            phase=Phase(point_group="m-3m"),
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        fpath = tmp_path / "fallback.sht"
        harmonics.save(fpath)
        assert _sht_file.read_sht(fpath).sg_eff == 221

    def test_a_phase_without_any_symmetry_raises(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        harmonics = MasterPatternHarmonics(alm, beam_energy=20.0, sample_tilt=70.0)
        with pytest.raises(ValueError):
            harmonics.save(tmp_path / "nope.sht")

    def test_lossy_packing_raises_with_strict(self, tmp_path):
        alm = np.zeros((16, 16), dtype=np.complex128)
        alm[0, 0] = 1
        for degree in range(1, 16):
            alm[1, degree] = 0.5
        harmonics = MasterPatternHarmonics(alm, phase=Phase(space_group=225))
        with pytest.raises(ValueError, match="symmetry"):
            harmonics.save(tmp_path / "lossy.sht")

    def test_lossy_packing_warns_and_drops_without_strict(self, tmp_path):
        alm = np.zeros((16, 16), dtype=np.complex128)
        alm[0, 0] = 1
        for degree in range(1, 16):
            alm[1, degree] = 0.5
        harmonics = MasterPatternHarmonics(alm, phase=Phase(space_group=225))
        fpath = tmp_path / "lossy.sht"
        with pytest.warns(UserWarning, match="dropped"):
            harmonics.save(fpath, strict=False)
        again = MasterPatternHarmonics.from_file(fpath)
        assert np.all(again.alm[1] == 0)

    @pytest.mark.parametrize(
        "amplitude, expected_n_fold, saves",
        [(1e-6, 4, True), (1e-3, 2, False)],
    )
    def test_the_two_guards_agree(self, amplitude, expected_n_fold, saves, tmp_path):
        # Both guards use SYMMETRY_POWER_TOLERANCE on the same
        # quantity, so there is no window in which construction passes
        # silently and save() hard-fails
        alm = _file_harmonics(NI_SMALL)
        scale = np.abs(alm).max() * amplitude
        perturbed = alm.copy()
        for degree in range(2, 384, 2):
            perturbed[2, degree] = scale
        phase = Phase(name="ni", space_group=225)
        fpath = tmp_path / "perturbed.sht"
        if saves:
            harmonics = MasterPatternHarmonics(perturbed, phase=phase)
            assert harmonics.n_fold == expected_n_fold
            harmonics.save(fpath)
            assert fpath.is_file()
        else:
            with pytest.warns(UserWarning):
                harmonics = MasterPatternHarmonics(perturbed, phase=phase)
            assert harmonics.n_fold == expected_n_fold
            with pytest.raises(ValueError):
                harmonics.save(fpath)
            with pytest.warns(UserWarning):
                harmonics.save(fpath, strict=False)

    def test_the_flag_ambiguous_point_group_trap(self, tmp_path):
        # "3m" maps to space group 156 (3, 0x8) by the lowest-space
        # group fallback, but 157 is (3, 0x4). Coefficients in the 157
        # storage pattern therefore hit the losslessness guard, whose
        # message names both candidates
        bandwidth = 16
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        for m in range(0, bandwidth, 3):
            for degree in range(m, bandwidth):
                alm[m, degree] = complex(0.5, 0.0)
        harmonics = MasterPatternHarmonics(
            alm, phase=Phase(point_group="3m"), beam_energy=20.0, sample_tilt=70.0
        )
        with pytest.raises(ValueError) as info:
            harmonics.save(tmp_path / "trap.sht")
        message = str(info.value)
        assert "156" in message
        assert "157" in message

        ok = MasterPatternHarmonics(
            alm,
            phase=Phase(space_group=157),
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        ok.save(tmp_path / "ok.sht")

    def test_a_z_unique_monoclinic_phase_saves(self, tmp_path):
        # Space group 3 is (1, 0x0) in the SHT file tables, so nothing
        # is dropped whatever the coefficients look like
        bandwidth = 12
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        for m in range(bandwidth):
            for degree in range(m, bandwidth):
                if (degree + m) % 2 == 0:
                    alm[m, degree] = complex(0.3, 0.0 if m == 0 else 0.2)
        harmonics = MasterPatternHarmonics(
            alm, phase=Phase(space_group=3), beam_energy=20.0, sample_tilt=70.0
        )
        harmonics.save(tmp_path / "monoclinic.sht")

    def test_a_b_unique_monoclinic_phase_with_complex_rows_raises(self, tmp_path):
        # Space group 6 is (1, 0x4), i.e. every row is stored real
        bandwidth = 12
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        for m in range(1, bandwidth):
            for degree in range(m, bandwidth):
                alm[m, degree] = complex(0.3, 0.4)
        harmonics = MasterPatternHarmonics(
            alm, phase=Phase(space_group=6), beam_energy=20.0, sample_tilt=70.0
        )
        with pytest.raises(ValueError):
            harmonics.save(tmp_path / "b_unique.sht")

    def test_fractional_atom_coordinates_round_trip(self, tmp_path):
        from diffpy.structure import Atom, Lattice, Structure

        fractions = [1 / 3, 2 / 3, 1 / 6, 5 / 6, 0.25]
        expected = [8, 16, 4, 20, 6]
        atoms = [Atom(atype="Ni", xyz=(f, 0, 0)) for f in fractions]
        structure = Structure(lattice=Lattice(0.4, 0.4, 0.4, 90, 90, 90), atoms=atoms)
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        harmonics = MasterPatternHarmonics(
            alm,
            phase=Phase(name="ni", space_group=1, structure=structure),
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        fpath = tmp_path / "atoms.sht"
        harmonics.save(fpath)
        sht = _sht_file.read_sht(fpath)
        assert [atom.x for atom in sht.crystals[0].atoms] == pytest.approx(
            expected, rel=1e-6
        )
        again = MasterPatternHarmonics.from_file(fpath)
        assert [atom.xyz[0] for atom in again.phase.structure] == pytest.approx(
            fractions, rel=1e-7
        )

    def test_overwrite_false_leaves_the_file_alone(self, tmp_path):
        from_master = _ni_harmonics_bw384()
        fpath = tmp_path / "ni.sht"
        fpath.write_bytes(b"not an sht file")
        from_master.save(fpath, overwrite=False)
        assert fpath.read_bytes() == b"not an sht file"

    def test_a_filename_without_a_suffix_gets_one(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        harmonics = MasterPatternHarmonics(
            alm,
            phase=Phase(name="ni", space_group=225),
            beam_energy=20.0,
            sample_tilt=70.0,
        )
        harmonics.save(tmp_path / "noext")
        assert (tmp_path / "noext.sht").is_file()


class TestResizeRemoveDcAndPower:
    def test_growing_then_shrinking_is_the_identity(self):
        harmonics = _ni_harmonics_from_file()
        again = harmonics.resize(400).resize(384)
        assert np.array_equal(again.alm, harmonics.alm)

    def test_shrinking_is_a_block_slice(self):
        harmonics = _ni_harmonics_from_file()
        assert np.array_equal(harmonics.resize(100).alm, harmonics.alm[:100, :100])

    def test_growing_zero_pads(self):
        harmonics = _ni_harmonics_from_file()
        grown = harmonics.resize(400)
        assert grown.bandwidth == 400
        assert np.array_equal(grown.alm[:384, :384], harmonics.alm)
        assert np.all(grown.alm[384:, :] == 0)
        assert np.all(grown.alm[:, 384:] == 0)

    def test_the_metadata_bandwidth_is_updated(self):
        harmonics = _ni_harmonics_from_file()
        grown = harmonics.resize(400)
        assert grown.original_metadata["harmonics"]["bandwidth"] == 400
        assert harmonics.original_metadata["harmonics"]["bandwidth"] == 384

    def test_a_bandwidth_below_one_raises(self):
        harmonics = _ni_harmonics_from_file()
        with pytest.raises(ValueError):
            harmonics.resize(0)

    def test_remove_dc_only_changes_the_constant_term(self):
        harmonics = _ni_harmonics_from_file()
        before = harmonics.alm.copy()
        without = harmonics.remove_dc()
        assert without.alm[0, 0] == 0
        assert np.array_equal(without.alm[1:], harmonics.alm[1:])
        assert np.array_equal(without.alm[0, 1:], harmonics.alm[0, 1:])
        # The original is untouched
        assert np.array_equal(harmonics.alm, before)

    def test_the_power_of_a_single_entry(self):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[3, 5] = 1
        power = MasterPatternHarmonics(alm).power_spectrum()
        assert power.shape == (8,)
        assert power[5] == pytest.approx(2)
        assert np.count_nonzero(power) == 1

    def test_the_power_of_a_zonal_entry(self):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 2] = 1
        power = MasterPatternHarmonics(alm).power_spectrum()
        assert power[2] == pytest.approx(1)
        assert np.count_nonzero(power) == 1

    def test_parseval_against_an_exact_quadrature(self, record_property):
        # The oracle is the Gauss-Legendre quadrature, independent of
        # the power_spectrum formula. Phase 1's Mazonka weights are
        # Lambert only and about 1e-3 accurate, so a fine Legendre
        # grid is used instead: measured 6.1e-11 at dim 67
        bandwidth, dim = 16, 67
        alm = _legendre_alm(bandwidth)
        transform = _sht.SphericalHarmonicTransform(bandwidth, "legendre", dim)
        north, south = transform.synthesize(alm)
        weights = _grid.quadrature_weights(dim, "legendre")[0][_grid.ring_number(dim)]
        integral = (np.sum(weights * north**2) + np.sum(weights * south**2)) / (
            2 * np.sum(weights)
        )
        power = MasterPatternHarmonics(alm).power_spectrum().sum()
        error = abs(power - 4 * np.pi * integral) / power
        record_property("parseval_rel_error_dim_67", f"{error:.3e}")
        assert error < 1e-8

    def test_parseval_on_the_nickel_file(self, record_property):
        harmonics = _ni_harmonics_from_file()
        total = harmonics.power_spectrum().sum()
        record_property("ni_power_sum", f"{total:.4f}")
        assert total == pytest.approx(4 * np.pi, rel=1e-2)


class TestDescribeAndRepr:
    @pytest.mark.parametrize(
        "substring",
        [
            "file version 1.1",
            "software version ve49ad6b",
            "EBSD",
            "20.1",
            "70",
            "effective sg# 225",
            "pijk = 1",
            "88 bytes",
            "EMsoft",
            "sg 225 setting 1",
            "0.35236",
            "Ni",
            "28:",
            "0.0035",
            "square lambert",
            "bandwidth 384",
            "zRot 4",
            "cmpFlg 0x7",
            "9312",
            "n_fold 4",
            "equatorial mirror True",
            "a_00 = -2.985",
            "DC power fraction 0.71",
        ],
    )
    def test_describe_of_a_file_instance(self, substring):
        harmonics = _ni_harmonics_from_file()
        assert substring in harmonics.describe()

    def test_describe_of_a_built_instance(self):
        with pytest.warns(UserWarning):
            built = _ni_master().get_spherical_harmonics(bandwidth=384)
        text = built.describe()
        assert "kp" in text
        assert "a_00 = -2.985" in text
        assert "DC power fraction 0.71" in text

    def test_describe_of_the_opt_in_instance(self):
        with pytest.warns(UserWarning):
            built = _ni_master().get_spherical_harmonics(
                bandwidth=384, emsphinx_compatible=False
            )
        assert "DC power fraction 0.00" in built.describe()

    def test_the_repr(self):
        with pytest.warns(UserWarning):
            built = _ni_master().get_spherical_harmonics(bandwidth=384)
        assert repr(built) == (
            "MasterPatternHarmonics: bw = 384, ni (m-3m), 20.1 keV, 70.0 deg"
        )

    def test_the_repr_of_an_empty_instance(self):
        text = repr(MasterPatternHarmonics(np.zeros((4, 4), np.complex128)))
        assert "bw = 4" in text
        assert "None" in text


class TestRotate:
    def test_rotate_defers_to_phase_three(self):
        harmonics = MasterPatternHarmonics(np.zeros((4, 4), np.complex128))
        with pytest.raises(NotImplementedError, match="Phase 3"):
            harmonics.rotate(Rotation.identity())


class TestSymmetryGuardOnConstruction:
    def test_the_nickel_coefficients_keep_their_flags(self):
        ni_alm = _file_harmonics(NI_SMALL)
        harmonics = MasterPatternHarmonics(
            ni_alm, phase=Phase(name="ni", space_group=225)
        )
        assert harmonics.n_fold == 4
        assert harmonics.has_equatorial_mirror

    def test_a_filled_first_order_row_downgrades_to_one(self):
        ni_alm = _file_harmonics(NI_SMALL)
        alm = ni_alm.copy()
        scale = np.abs(alm).max()
        for degree in range(1, 384, 2):
            alm[1, degree] = scale
        with pytest.warns(UserWarning):
            harmonics = MasterPatternHarmonics(
                alm, phase=Phase(name="ni", space_group=225)
            )
        assert harmonics.n_fold == 1

    def test_a_filled_second_order_row_downgrades_to_two(self):
        ni_alm = _file_harmonics(NI_SMALL)
        alm = ni_alm.copy()
        scale = np.abs(alm).max()
        for degree in range(2, 384, 2):
            alm[2, degree] = scale
        with pytest.warns(UserWarning):
            harmonics = MasterPatternHarmonics(
                alm, phase=Phase(name="ni", space_group=225)
            )
        assert harmonics.n_fold == 2

    def test_no_phase_is_the_safe_default(self, recwarn):
        harmonics = MasterPatternHarmonics(_file_harmonics(NI_SMALL))
        assert harmonics.n_fold == 1
        assert harmonics.has_equatorial_mirror is False
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


class TestHemispheresAndTypes:
    def test_upper_only_with_a_centrosymmetric_phase(self):
        rng = np.random.default_rng(0)
        north = rng.uniform(size=(21, 21)).astype(np.float32)
        upper = kp.signals.EBSDMasterPattern(
            north,
            projection="lambert",
            hemisphere="upper",
            phase=Phase(name="x", point_group="-1"),
        )
        both = kp.signals.EBSDMasterPattern(
            np.stack([north, north[::-1, ::-1]]),
            projection="lambert",
            hemisphere="both",
            phase=Phase(name="x", point_group="-1"),
        )
        a = MasterPatternHarmonics.from_master_pattern(
            upper, bandwidth=8, beam_energy=20.0, sample_tilt=70.0
        )
        b = MasterPatternHarmonics.from_master_pattern(
            both, bandwidth=8, beam_energy=20.0, sample_tilt=70.0
        )
        assert np.array_equal(a.alm, b.alm)

    def test_upper_only_nickel_equals_both(self):
        upper = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="upper"
        )
        a = MasterPatternHarmonics.from_master_pattern(upper, bandwidth=32)
        b = MasterPatternHarmonics.from_master_pattern(_ni_master(), bandwidth=32)
        assert np.allclose(a.alm, b.alm, rtol=0, atol=1e-14)

    def test_lower_only_raises(self):
        data = np.ones((21, 21), dtype=np.float32)
        lower = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="lower",
            phase=Phase(name="x", point_group="m-3m"),
        )
        with pytest.raises(ValueError):
            MasterPatternHarmonics.from_master_pattern(
                lower, bandwidth=4, beam_energy=20.0, sample_tilt=70.0
            )

    def test_upper_only_without_inversion_raises(self):
        data = np.ones((21, 21), dtype=np.float32)
        upper = kp.signals.EBSDMasterPattern(
            data,
            projection="lambert",
            hemisphere="upper",
            phase=Phase(name="x", point_group="4mm"),
        )
        with pytest.raises(ValueError):
            MasterPatternHarmonics.from_master_pattern(
                upper, bandwidth=4, beam_energy=20.0, sample_tilt=70.0
            )

    def test_an_ecp_master_pattern_raises_a_type_error(self):
        ecp = kp.signals.ECPMasterPattern(
            np.ones((2, 21, 21), dtype=np.float32),
            projection="lambert",
            hemisphere="both",
            phase=Phase(name="x", point_group="m-3m"),
        )
        with pytest.raises(TypeError):
            MasterPatternHarmonics.from_master_pattern(ecp, bandwidth=4)

    def test_a_stereographic_master_pattern_raises(self):
        master = kp.data.nickel_ebsd_master_pattern_small(hemisphere="both")
        assert master.projection == "stereographic"
        with pytest.raises(NotImplementedError) as info:
            MasterPatternHarmonics.from_master_pattern(master, bandwidth=4)
        message = str(info.value)
        assert "square Lambert projection" in message
        assert "as_lambert" in message

    def test_a_lazy_master_pattern_gives_the_same_result(self):
        eager = _ni_master()
        lazy = eager.as_lazy()
        a = MasterPatternHarmonics.from_master_pattern(eager, bandwidth=16)
        b = MasterPatternHarmonics.from_master_pattern(lazy, bandwidth=16)
        assert np.array_equal(a.alm, b.alm)


class TestBeamEnergyAndSampleTilt:
    def test_they_come_from_the_metadata(self):
        harmonics = MasterPatternHarmonics.from_master_pattern(
            _ni_master(), bandwidth=8
        )
        assert harmonics.beam_energy == pytest.approx(20.1)
        assert harmonics.sample_tilt == pytest.approx(70.0)

    def test_the_keywords_override_the_metadata(self):
        harmonics = MasterPatternHarmonics.from_master_pattern(
            _ni_master(), bandwidth=8, beam_energy=15.0, sample_tilt=60.0
        )
        assert harmonics.beam_energy == 15.0
        assert harmonics.sample_tilt == 60.0

    def test_a_missing_beam_energy_raises(self):
        master = _constant_master()
        with pytest.raises(ValueError, match="beam_energy"):
            MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=4, sample_tilt=70.0
            )

    def test_a_missing_sample_tilt_raises(self):
        master = _constant_master()
        with pytest.raises(ValueError, match="sample_tilt"):
            MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=4, beam_energy=20.0
            )


class TestFromFileErrors:
    def test_a_non_ebsd_modality_raises(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        packed = _sht_file.pack_harmonics(alm, 8, 1, 0)
        sht = _sht_file.ShtFile(
            header=_sht_file.ShtHeader(
                software_version="kp-test",
                modality=_sht_file.MODALITY_ECP,
                beam_energy=20.0,
                primary_angle=70.0,
            ),
            num_xtal=1,
            sg_eff=225,
            modality=_sht_file.MODALITY_ECP,
            crystals=[_sht_file.ShtCrystal(sg_num=225)],
            simulations=[None],
            harmonics=_sht_file.ShtHarmonics(
                bandwidth=8, z_rot=1, flags=0, doub_cnt=packed.size, packed=packed
            ),
        )
        fpath = tmp_path / "ecp.sht"
        _sht_file.write_sht(fpath, sht)
        with pytest.raises(NotImplementedError, match="ECP"):
            MasterPatternHarmonics.from_file(fpath)

    def test_more_than_one_crystal_raises(self, tmp_path):
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        packed = _sht_file.pack_harmonics(alm, 8, 1, 0)
        sht = _sht_file.ShtFile(
            header=_sht_file.ShtHeader(
                software_version="kp-test",
                modality=_sht_file.MODALITY_EBSD,
                beam_energy=20.0,
                primary_angle=70.0,
            ),
            num_xtal=2,
            sg_eff=225,
            modality=_sht_file.MODALITY_EBSD,
            crystals=[
                _sht_file.ShtCrystal(sg_num=225),
                _sht_file.ShtCrystal(sg_num=194),
            ],
            simulations=[None, None],
            harmonics=_sht_file.ShtHarmonics(
                bandwidth=8, z_rot=1, flags=0, doub_cnt=packed.size, packed=packed
            ),
        )
        fpath = tmp_path / "two.sht"
        _sht_file.write_sht(fpath, sht)
        with pytest.raises(NotImplementedError, match="numXtal"):
            MasterPatternHarmonics.from_file(fpath)


class TestEmsphinxBinaries:
    def test_emsphinx_binaries_sht2png_accepts_a_written_file(self, tmp_path):
        program = _emsphinx_program("sht2png")
        with pytest.warns(UserWarning):
            harmonics = _ni_master().get_spherical_harmonics(bandwidth=384)
        fpath = tmp_path / "ours.sht"
        harmonics.save(fpath)
        result = subprocess.run(
            [str(program), str(fpath), str(tmp_path / "leg.png")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "effective sg# 225" in result.stdout

    def test_emsphinx_binaries_the_legendre_png_matches_our_synthesis(self, tmp_path):
        import imageio.v3 as iio

        program = _emsphinx_program("sht2png")
        with pytest.warns(UserWarning):
            harmonics = _ni_master().get_spherical_harmonics(bandwidth=384)
        fpath = tmp_path / "ours.sht"
        harmonics.save(fpath)
        png = tmp_path / "leg.png"
        result = subprocess.run(
            [str(program), str(fpath), str(png)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        image = iio.imread(png)
        assert image.shape == (387, 774)
        assert image.dtype == np.uint8
        north, south = _sht.SphericalHarmonicTransform(384, "legendre", 387).synthesize(
            harmonics.alm
        )
        # sht2png scales both hemispheres with the north min and max
        low, high = north.min(), north.max()
        ours = np.concatenate(
            [
                np.round((north - low) * 255 / (high - low)),
                np.round((south - low) * 255 / (high - low)),
            ],
            axis=1,
        )
        assert np.abs(ours - image.astype(np.float64)).max() <= 1
