# Copyright 2019-2026 The kikuchipy developers
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

import shutil

import h5py
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.io.plugins.oxford_h5ebsd._api import get_binning


class TestOxfordH5EBSD:
    @pytest.mark.parametrize(
        "oxford_h5ebsd_file", ["7.0", "6.0"], indirect=["oxford_h5ebsd_file"]
    )
    def test_load_oxford_h5ebsd(
        self, oxford_h5ebsd_file, ni_small_axes_manager, assert_dictionary_func
    ):
        s = kp.load(oxford_h5ebsd_file)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary_func(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        assert s.metadata.Acquisition_instrument.SEM.beam_energy == 20

        s2 = kp.data.nickel_ebsd_small()
        s2.remove_static_background()
        assert np.allclose(s.data, s2.data)
        assert np.allclose(s.static_background, s2.static_background)

        # Detector
        det = s.detector
        assert det.pc.shape == (3, 3, 3)
        assert det.binning == 17.0
        assert np.isclose(det.sample_tilt, 69.9, atol=0.1)
        assert np.isclose(det.tilt, 1.5)

    def test_load_unprocessed_patterns(self, oxford_h5ebsd_file):
        s1 = kp.load(oxford_h5ebsd_file)
        s2 = kp.load(oxford_h5ebsd_file, processed=False)
        assert np.allclose(s1.data + 1, s2.data)

    @pytest.mark.parametrize(
        ["version", "header_group", "binning"],
        [
            ("5.0", {"Camera Binning Mode": "8x8 (168x128 px)"}, 8),
            ("6.0", {"Camera Binning Mode": "Resolution (1244x1024 px)"}, 1),
            ("6.0", {"Camera Binning Mode": "Speed 3 (156x88 px)"}, 8),
            ("7.0", {"Camera Mode": "Sensitivity (622x512 px)"}, 2),
        ],
    )
    def test_get_binning(self, version, header_group, binning):
        assert get_binning(header_group, version) == binning


class TestCameraModeDatasetName:
    """The H5OINA format version is not a reliable guide to which of
    the two camera mode dataset names a file carries.

    ADDED 2026-09-08.  ``get_binning`` used to pick ONE name from the
    format version, "Camera Mode" at 7.0 and above and "Camera
    Binning Mode" below, and silently return ``None`` when the file
    carried the other one.  AZtec 3.2.0.0 writes format 7.0 files
    whose header holds "Camera Binning Mode", which is how the bug was
    found: the Si-indent data set of Winkelmann et al. 2025 (Zenodo
    14059950) is such a file, and its detector arrived with the
    default ``binning=1`` instead of the 2 its "Speed 1 (622x512 px)"
    camera mode states.  Both names are now accepted, the
    version-appropriate one first.
    """

    @pytest.mark.parametrize(
        ["version", "header_group", "binning"],
        [
            # the crossed cases, neither of which parsed before.  The
            # first is the Si-indent file's own header, transcribed
            # from it on 2026-09-08
            ("7.0", {"Camera Binning Mode": "Speed 1 (622x512 px)"}, 2),
            ("7.0", {"Camera Binning Mode": "Speed 3 (156x88 px)"}, 8),
            ("5.0", {"Camera Mode": "Resolution (1244x1024 px)"}, 1),
            # and a file carrying BOTH takes the version-appropriate
            # one, so the fallback never silently overrides a correct
            # reading
            (
                "7.0",
                {
                    "Camera Mode": "4x4 (311x256 px)",
                    "Camera Binning Mode": "8x8 (168x128 px)",
                },
                4,
            ),
            (
                "6.0",
                {
                    "Camera Mode": "4x4 (311x256 px)",
                    "Camera Binning Mode": "8x8 (168x128 px)",
                },
                8,
            ),
        ],
    )
    def test_either_dataset_name_parses(self, version, header_group, binning):
        assert get_binning(header_group, version) == binning

    @pytest.mark.parametrize("version", ["5.0", "7.0"])
    def test_neither_name_still_returns_none(self, version):
        # the unchanged half of the contract: an unreadable binning is
        # left unset rather than guessed
        assert get_binning({"Beam Voltage": 20}, version) is None

    def test_a_format_7_file_with_the_old_dataset_name(
        self, oxford_h5ebsd_file, tmp_path
    ):
        """The bug end to end, on a copy of the shipped test asset.

        The asset is a format 7.0 file carrying "Camera Mode"; here it
        is copied and the dataset RENAMED to "Camera Binning Mode",
        which is exactly what AZtec 3.2.0.0 writes.  Before the fix
        the detector came back with ``binning=1``.
        """
        path = tmp_path / "camera_binning_mode.h5oina"
        shutil.copy(oxford_h5ebsd_file, path)
        with h5py.File(path, mode="r+") as f:
            assert f["Format Version"][()] == b"7.0"
            header = f["1/EBSD/Header"]
            assert "Camera Mode" in header
            header.move("Camera Mode", "Camera Binning Mode")
            assert "Camera Mode" not in header

        s = kp.load(path)
        # the same 17 the unrenamed asset gives, so the rename is the
        # only difference and the reader is insensitive to it
        assert s.detector.binning == 17.0
        assert s.detector.binning == kp.load(oxford_h5ebsd_file).detector.binning
