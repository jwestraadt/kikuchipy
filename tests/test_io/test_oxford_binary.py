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

import os

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.io.plugins.oxford_binary import get_scan_info
from kikuchipy.io.plugins.oxford_binary._api import OxfordBinaryFileReader


class TestOxfordBinaryReader:
    def test_load(self, oxford_binary_path):
        """Load into memory."""
        s = kp.load(oxford_binary_path / "patterns.ebsp")
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.EBSD)
        assert np.allclose(s.data, s2.data)

    def test_load_lazy(self, oxford_binary_path):
        """Load lazily."""
        s = kp.load(oxford_binary_path / "patterns.ebsp", lazy=True)
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.LazyEBSD)
        assert isinstance(s.data, da.Array)
        s.compute()
        assert np.allclose(s.data, s2.data)

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, True, True)],
        indirect=["oxford_binary_file"],
    )
    def test_compressed_patterns_raises(self, oxford_binary_file):
        """Ensure explanatory error message is raised when a file we
        cannot read is tried to be read from.
        """
        with pytest.raises(NotImplementedError, match="Cannot read compressed"):
            _ = kp.load(oxford_binary_file.name)

    @pytest.mark.parametrize(
        "oxford_binary_file, dtype",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), np.uint8),
            (((2, 3), (60, 60), np.uint16, 2, False, True), np.uint16),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_dtype(self, oxford_binary_file, dtype):
        """Ensure both uint8 and uint16 patterns can be read."""
        s = kp.load(oxford_binary_file.name)
        assert np.issubdtype(s.data.dtype, dtype)

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, False, False)],
        indirect=["oxford_binary_file"],
    )
    def test_not_all_patterns_present(self, oxford_binary_file):
        """Ensure files with only non-indexed patterns can be read."""
        s = kp.load(oxford_binary_file.name)
        assert s.axes_manager.navigation_shape == (5,)
        # (2, 2) is missing
        assert np.allclose(s.original_metadata.beam_y, [0, 1, 1, 1, 0])
        assert np.allclose(s.original_metadata.beam_x, [2, 0, 1, 2, 0])

    @pytest.mark.parametrize(
        "oxford_binary_file, ver, desired_nav_shape",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), 2, (2, 3)),
            (((2, 3), (60, 60), np.uint16, 1, False, True), 1, (2, 3)),
            (((2, 3), (60, 60), np.uint8, 0, False, True), 0, (6,)),
            (((2, 3), (60, 60), np.uint8, 4, False, True), 4, (2, 3)),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_versions(self, oxford_binary_file, ver, desired_nav_shape):
        """Ensure that versions 0, 1 and > 1 can be read."""
        s = kp.load(oxford_binary_file.name)
        assert s._navigation_shape_rc == desired_nav_shape
        if ver > 0:
            assert s.original_metadata.has_item("beam_x")
            assert s.original_metadata.has_item("beam_y")

    @pytest.mark.parametrize(
        "oxford_binary_file, n_patterns",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), 6),
            (((3, 4), (62, 73), np.uint8, 2, False, True), 12),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_guess_number_of_patterns(self, oxford_binary_file, n_patterns):
        """Ensure that the function guessing the number of patterns in
        the file works.
        """
        with open(oxford_binary_file.name, mode="rb") as f:
            fox = OxfordBinaryFileReader(f)
            assert fox.n_patterns == n_patterns

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [
            ((2, 3), (50, 50), np.uint8, 5, False, True),
            ((2, 3), (50, 50), np.uint8, 6, False, True),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_version_5(self, oxford_binary_file):
        with open(oxford_binary_file.name, mode="rb") as f:
            fox = OxfordBinaryFileReader(f)
            assert fox.n_patterns == 6

    @pytest.mark.parametrize(
        "oxford_binary_file, file_size",
        [
            (((2, 3), (50, 50), np.uint8, 5, False, True), 15309),
            (((2, 3), (50, 50), np.uint8, 4, False, True), 15261),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_estimated_file_size(self, oxford_binary_file, file_size):
        with open(oxford_binary_file.name, mode="rb") as f:
            fox = OxfordBinaryFileReader(f)
            assert fox.get_estimated_file_size() == file_size
            assert os.path.getsize(oxford_binary_file.name) == file_size


# ------------------- The scan grid probe (D3) ----------------------- #

# The measured ``EBSPDims.exe`` report of the in-package
# ``patterns.ebsp``, which holds the ``nickel_ebsd_small`` map: nine
# 60 x 60 unsigned 8-bit patterns of 3600 bytes each, on a regular
# grid of three x and three y coordinates one and a half micron
# apart.
IN_PACKAGE_SCAN_INFO = {
    "n_patterns": 9,
    "n_patterns_present": 9,
    "all_patterns_present": True,
    "signal_shape": (60, 60),
    "pattern_bytes": 3600,
    "total_bytes": 32400,
    "version": 2,
    "is_regular_grid": True,
}

SCAN_INFO_KEYS = {
    "n_patterns",
    "n_patterns_present",
    "all_patterns_present",
    "signal_shape",
    "dtype",
    "pattern_bytes",
    "total_bytes",
    "version",
    "beam_x",
    "beam_y",
    "is_regular_grid",
}

# The staggered synthetic file ``EBSPDims.exe`` reported as
# ``found 6 x and 2 y coordinates``, ``X: 0 0.5 1 1.5 2 2.5`` and
# ``Y: 0 1``
STAGGERED_X = [0.0, 1.0, 2.0, 0.5, 1.5, 2.5]
STAGGERED_Y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def write_dummy_ebsp(
    fpath, beam_x, beam_y, signal_shape=(60, 60), dtype=np.uint8, version=2
):
    """Write a dummy .ebsp file with the given beam positions.

    Modelled on
    ``src/kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py``
    and module local on purpose: the ``oxford_binary_file`` fixture of
    the root ``conftest.py`` derives the beam positions from the
    navigation indices and cannot express a staggered grid or two
    coordinates a thousandth of a nanometre apart (plan 3.2).

    Every pattern is present and written in file order.
    """
    beam_x = np.asarray(beam_x, dtype=np.float64)
    beam_y = np.asarray(beam_y, dtype=np.float64)
    n_patterns = beam_x.size
    sr, sc = signal_shape
    n_pixels = sr * sc
    n_bytes = n_pixels if np.dtype(dtype) == np.uint8 else 2 * n_pixels
    pattern_header_size = 16
    pattern_footer_size = 18

    with open(fpath, mode="wb") as f:
        np.array(-version, dtype=np.int64).tofile(f)
        starts = np.arange(n_patterns, dtype=np.int64)
        starts *= pattern_header_size + n_bytes + pattern_footer_size
        starts += 8 + n_patterns * 8
        starts.tofile(f)

        header = np.array([0, sr, sc, n_bytes], dtype=np.int32)
        data = np.arange(n_patterns * n_pixels, dtype=dtype)
        data = data.reshape((n_patterns, sr, sc))
        for i in range(n_patterns):
            header.tofile(f)
            data[i].tofile(f)
            np.array(1, dtype=bool).tofile(f)
            np.array(beam_x[i], dtype=np.float64).tofile(f)
            np.array(1, dtype=bool).tofile(f)
            np.array(beam_y[i], dtype=np.float64).tofile(f)
    return fpath


class TestGetScanInfo:
    """The ``EBSPDims`` equivalent probe of the .ebsp reader."""

    def test_get_scan_info_in_package_file(self, oxford_binary_path):
        info = get_scan_info(oxford_binary_path / "patterns.ebsp")
        for key, value in IN_PACKAGE_SCAN_INFO.items():
            assert info[key] == value, key
        assert info["dtype"] == np.uint8
        assert np.array_equal(info["beam_x"], [0.0, 1.5, 3.0])
        assert np.array_equal(info["beam_y"], [0.0, 1.5, 3.0])

    def test_get_scan_info_key_set(self, oxford_binary_path):
        # plain Python scalars, not NumPy ones: the reader's own
        # ``signal_shape`` is a tuple of ``numpy.int32`` and its
        # ``n_patterns`` a ``numpy.int64``, neither an ``int``
        # instance, so the cast is part of the contract
        info = get_scan_info(oxford_binary_path / "patterns.ebsp")
        assert set(info) == SCAN_INFO_KEYS
        assert isinstance(info["signal_shape"], tuple)
        assert all(isinstance(value, int) for value in info["signal_shape"])
        for key in (
            "n_patterns",
            "n_patterns_present",
            "pattern_bytes",
            "total_bytes",
            "version",
        ):
            assert isinstance(info[key], int), key
        assert isinstance(info["all_patterns_present"], bool)
        assert isinstance(info["is_regular_grid"], bool)
        assert info["beam_x"].dtype == np.float64

    def test_get_scan_info_takes_a_string(self, oxford_binary_path):
        fpath = oxford_binary_path / "patterns.ebsp"
        assert get_scan_info(str(fpath))["n_patterns"] == 9

    def test_get_scan_info_irregular(self, tmp_path):
        # the staggered rows ``EBSPDims.exe`` reported as six x and
        # two y coordinates, which is exactly its irregularity test
        fpath = write_dummy_ebsp(tmp_path / "staggered.ebsp", STAGGERED_X, STAGGERED_Y)
        info = get_scan_info(fpath)
        assert info["n_patterns_present"] == 6
        assert np.array_equal(info["beam_x"], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        assert np.array_equal(info["beam_y"], [0.0, 1.0])
        assert len(info["beam_x"]) * len(info["beam_y"]) != 6
        assert info["is_regular_grid"] is False

    def test_get_scan_info_exact_value_distinctness(self, tmp_path):
        # the C++ collects the coordinates in a ``std::set<double>``,
        # so there is no tolerance: these are two coordinates
        beam_x = [0.0, 1.0, 1.0 + 1e-12, 2.0]
        fpath = write_dummy_ebsp(tmp_path / "near.ebsp", beam_x, [0.0, 0.0, 1.0, 1.0])
        info = get_scan_info(fpath)
        assert len(info["beam_x"]) == 4
        assert 1.0 in info["beam_x"]
        assert 1.0 + 1e-12 in info["beam_x"]
        assert info["is_regular_grid"] is False

    def test_get_scan_info_regular_synthetic(self, tmp_path):
        fpath = write_dummy_ebsp(
            tmp_path / "regular.ebsp",
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        info = get_scan_info(fpath)
        assert info["is_regular_grid"] is True
        assert len(info["beam_x"]) == 3
        assert len(info["beam_y"]) == 2

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 0, False, True)],
        indirect=["oxford_binary_file"],
    )
    def test_get_scan_info_version0_has_no_beams(self, oxford_binary_file):
        info = get_scan_info(oxford_binary_file.name)
        assert info["version"] == 0
        assert info["beam_x"] is None
        assert info["beam_y"] is None
        assert info["is_regular_grid"] is False
        assert info["n_patterns_present"] == 6

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, False, False)],
        indirect=["oxford_binary_file"],
    )
    def test_get_scan_info_not_all_present(self, oxford_binary_file):
        # the regularity test counts the patterns which are actually
        # there, not the header slots: ``EBSPDims`` cannot open such
        # a file at all, so this extension is ours (open question 10)
        info = get_scan_info(oxford_binary_file.name)
        assert info["n_patterns"] == 6
        assert info["n_patterns_present"] == 5
        assert info["all_patterns_present"] is False
        assert np.array_equal(info["beam_x"], [0.0, 1.0, 2.0])
        assert np.array_equal(info["beam_y"], [0.0, 1.0])
        assert info["total_bytes"] == 5 * 3600
        assert info["is_regular_grid"] is False

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint16, 2, False, True)],
        indirect=["oxford_binary_file"],
    )
    def test_get_scan_info_uint16(self, oxford_binary_file):
        info = get_scan_info(oxford_binary_file.name)
        assert info["dtype"] == np.uint16
        assert info["pattern_bytes"] == 2 * 3600
        assert info["total_bytes"] == 6 * 2 * 3600
        assert info["is_regular_grid"] is True

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, True, True)],
        indirect=["oxford_binary_file"],
    )
    def test_get_scan_info_compressed_raises(self, oxford_binary_file):
        # inherited from the reader, whose C++ analogue throws too
        with pytest.raises(NotImplementedError, match="Cannot read compressed"):
            get_scan_info(oxford_binary_file.name)
