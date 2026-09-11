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

"""Tests for the EMSphInx *.sht master pattern reader.

Covers the "io plugin" assertions of
``specs/2026-08-16-sht-master-spectra-and-file/validation.md``.
"""

from pathlib import Path

import dask.array as da
import matplotlib
import numpy as np
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.indexing import MasterPatternHarmonics
from kikuchipy.indexing._spherical import _sht_file
from kikuchipy.io._io import PLUGINS
from kikuchipy.io.plugins.emsphinx_master_pattern import file_reader

matplotlib.use("Agg")

NI_SMALL = "emsphinx/ni_small_20kv_bw384.sht"


@pytest.fixture(scope="module")
def ni_sht_file() -> Path:
    """Return the path of the in-package mp2sht Ni ``.sht`` file."""
    return Path(Dataset(NI_SMALL).fetch_file_path())


def _write_sht(
    fpath: Path,
    modality: int,
    num_xtal: int = 1,
    alm: np.ndarray | None = None,
    space_group: int = 225,
) -> Path:
    """Write a small synthetic ``.sht`` file with our own codec."""
    if alm is None:
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1.0
    packed = _sht_file.pack_harmonics(alm, 8, 1, 0)
    crystals = [_sht_file.ShtCrystal(sg_num=space_group) for _ in range(num_xtal)]
    sht = _sht_file.ShtFile(
        header=_sht_file.ShtHeader(
            software_version="kp-test",
            modality=modality,
            beam_energy=20.0,
            primary_angle=70.0,
        ),
        num_xtal=num_xtal,
        sg_eff=space_group,
        pijk=1,
        rot_sense=112,
        modality=modality,
        vendor=_sht_file.VENDOR_UNKNOWN,
        sim_meta_size=0,
        crystals=crystals,
        simulations=[None] * num_xtal,
        harmonics=_sht_file.ShtHarmonics(
            bandwidth=8, z_rot=1, flags=0, doub_cnt=packed.size, packed=packed
        ),
    )
    _sht_file.write_sht(fpath, sht)
    return fpath


class TestPluginRegistration:
    def test_the_plugin_is_discovered(self):
        matches = [
            plugin for plugin in PLUGINS if plugin["name"] == "emsphinx_master_pattern"
        ]
        assert len(matches) == 1
        plugin = matches[0]
        assert "sht" in plugin["file_extensions"]
        assert plugin["writes"] is False
        assert plugin["manufacturer"] == "emsphinx"
        assert plugin["description"].strip()

    def test_the_reader_docstring_has_the_boilerplate(self):
        assert (
            "Not meant to be used directly; use :func:`~kikuchipy.load`."
            in file_reader.__doc__
        )
        assert "signal_dict_list" in file_reader.__doc__


class TestDefaultLoad:
    def test_the_signal_and_the_data(self, ni_sht_file):
        master = kp.load(ni_sht_file)
        assert isinstance(master, kp.signals.EBSDMasterPattern)
        assert master.data.shape == (2, 769, 769)
        assert master.data.dtype == np.float64
        assert master.projection == "lambert"
        assert master.hemisphere == "both"

    def test_the_axes(self, ni_sht_file):
        master = kp.load(ni_sht_file)
        axes = master.axes_manager._axes
        assert [ax.name for ax in axes] == ["hemisphere", "height", "width"]
        assert [ax.offset for ax in axes] == [0, -384, -384]
        assert [ax.units for ax in axes] == ["", "px", "px"]

    def test_the_metadata(self, ni_sht_file):
        master = kp.load(ni_sht_file)
        assert master.metadata.General.title == "ni_small_20kv_bw384"
        assert master.metadata.Signal.signal_type == "EBSDMasterPattern"

    def test_the_phase(self, ni_sht_file):
        master = kp.load(ni_sht_file)
        assert master.phase.space_group.number == 225
        assert master.phase.point_group.name == "m-3m"
        assert master.phase.name == "Ni"
        assert master.phase.structure.lattice.a == pytest.approx(0.35236, rel=1e-6)
        assert len(master.phase.structure) == 1
        assert master.phase.structure[0].element in (28, "28", "Ni")

    def test_the_original_metadata_nodes(self, ni_sht_file):
        # Numbered sub-nodes, not lists: hyperspy's
        # DictionaryTreeBrowser leaves list elements as plain dicts
        master = kp.load(ni_sht_file)
        original = master.original_metadata
        assert original.header.beam_energy == pytest.approx(20.1)
        assert original.header.primary_angle == 70
        assert original.harmonics.bandwidth == 384
        assert original.harmonics.z_rot == 4
        assert original.crystals.crystal_0.formula == "Ni"
        assert original.simulations.simulation_0.num_px == 200
        assert "packed" not in original.harmonics.as_dictionary()

    def test_the_original_metadata_equals_the_class_attribute(self, ni_sht_file):
        # One name, one shape, across the class and the signal
        master = kp.load(ni_sht_file)
        harmonics = MasterPatternHarmonics.from_file(ni_sht_file)
        assert master.original_metadata.as_dictionary() == (harmonics.original_metadata)

    def test_it_is_suitable_for_projection(self, ni_sht_file):
        assert kp.load(ni_sht_file)._is_suitable_for_projection()


class TestReaderArguments:
    def test_a_smaller_grid_warns_and_shrinks(self, ni_sht_file):
        with pytest.warns(UserWarning):
            master = kp.load(ni_sht_file, dim=401)
        assert master.data.shape == (2, 401, 401)
        # The centred convention, one pixel off kikuchipy's EMsoft
        # reader, which gives -201 for the same side length
        assert master.axes_manager["height"].offset == -200
        source = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        assert source.axes_manager["height"].offset == -201

    def test_the_upper_hemisphere(self, ni_sht_file):
        both = kp.load(ni_sht_file)
        upper = kp.load(ni_sht_file, hemisphere="upper")
        assert upper.data.shape == (769, 769)
        assert np.array_equal(upper.data, both.data[0])
        assert upper.hemisphere == "upper"

    def test_the_lower_hemisphere(self, ni_sht_file):
        both = kp.load(ni_sht_file)
        lower = kp.load(ni_sht_file, hemisphere="lower")
        assert lower.data.shape == (769, 769)
        assert np.array_equal(lower.data, both.data[1])
        assert lower.hemisphere == "lower"

    def test_an_unknown_hemisphere_raises_and_lists_the_options(self, ni_sht_file):
        with pytest.raises(ValueError) as info:
            kp.load(ni_sht_file, hemisphere="south")
        message = str(info.value)
        for option in ("upper", "lower", "both"):
            assert option in message

    def test_lazy_is_accepted(self, ni_sht_file):
        # The reader synthesizes eagerly; kp.load still wraps the
        # array, so a lazy signal comes back
        lazy = kp.load(ni_sht_file, lazy=True)
        assert isinstance(lazy, kp.signals.LazyEBSDMasterPattern)
        assert isinstance(lazy.data, da.Array)
        eager = kp.load(ni_sht_file)
        assert np.array_equal(lazy.data.compute(), eager.data)


def _write_antisymmetric_sht(fpath: Path) -> Path:
    """Write a file whose synthesis tells the hemispheres apart.

    One zonal coefficient of odd degree, ``Y_1^0``, so the south
    hemisphere is the negative of the north one. The Ni fixture has
    north == south, so comparing ``hemisphere="upper"`` against
    ``data[0]`` of the same file cannot see a swap. Space group 1,
    whose ``(z_rot, flags)`` pair is ``(1, 0x0)``, keeps every entry
    and needs no symmetry.
    """
    alm = np.zeros((8, 8), dtype=np.complex128)
    alm[0, 1] = 1.0
    return _write_sht(fpath, _sht_file.MODALITY_EBSD, alm=alm, space_group=1)


class TestHemisphereOrder:
    """A hemisphere swap in the reader is invisible on Nickel."""

    def test_the_hemispheres_are_not_swapped(self, tmp_path):
        fpath = _write_antisymmetric_sht(tmp_path / "odd.sht")
        master = kp.load(fpath)
        assert master.data.shape == (2, 17, 17)
        north, south = master.data
        assert not np.array_equal(north, south)
        assert np.allclose(south, -north, rtol=0, atol=1e-12)
        # The centre pixel of a square Lambert grid is the pole, and
        # Y_1^0 is positive at the north one
        assert north[8, 8] > 0
        assert south[8, 8] < 0

    def test_the_hemisphere_keyword_picks_the_right_pole(self, tmp_path):
        fpath = _write_antisymmetric_sht(tmp_path / "odd.sht")
        upper = kp.load(fpath, hemisphere="upper")
        lower = kp.load(fpath, hemisphere="lower")
        assert upper.data[8, 8] > 0
        assert lower.data[8, 8] < 0
        assert np.allclose(lower.data, -upper.data, rtol=0, atol=1e-12)


class TestUnsupportedFiles:
    def test_a_non_ebsd_modality_raises(self, tmp_path):
        fpath = _write_sht(tmp_path / "ecp.sht", _sht_file.MODALITY_ECP)
        with pytest.raises(NotImplementedError, match="ECP"):
            kp.load(fpath)

    def test_more_than_one_crystal_raises(self, tmp_path):
        fpath = _write_sht(tmp_path / "two.sht", _sht_file.MODALITY_EBSD, num_xtal=2)
        with pytest.raises(NotImplementedError, match="numXtal"):
            kp.load(fpath)


class TestSmoke:
    def test_plot(self, ni_sht_file):
        import matplotlib.pyplot as plt

        with pytest.warns(UserWarning):
            master = kp.load(ni_sht_file, dim=101)
        master.plot()
        plt.close("all")

    def test_get_patterns(self, ni_sht_file):
        master = kp.load(ni_sht_file)
        detector = kp.detectors.EBSDDetector(
            shape=(60, 60), pc=(0.42, 0.22, 0.50), sample_tilt=70
        )
        patterns = master.get_patterns(
            Rotation.identity((1,)),
            detector,
            compute=True,
            show_progressbar=False,
        )
        assert patterns.data.shape == (1, 60, 60)
        assert np.isfinite(patterns.data).all()
