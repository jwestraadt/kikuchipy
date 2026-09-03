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

"""Tests of ``kikuchipy.indexing._spherical._pattern_repack``.

Covers the ``test_spherical_pattern_repack.py`` assertions of
``specs/2026-09-02-sht-interop/validation.md``:

- The HDF5 contract: contiguous layout, early allocation, zero
  filters, no chunks, a defined data offset, the data set shape and
  type, and the scalar variable length **ASCII** ``Manufacturer``
  data set at the file root.
- The data: byte equality with the signal on the default route, row
  reversal under ``flip=True``, the frozen manufacturer flip table,
  the row major navigation order for two, one and zero dimensional
  navigation, the native byte order cast and lazy equals eager.
- The guards: the manufacturer whitelist, the data type whitelist,
  the non-divisor binning error, the non 8-bit warning and the
  ``overwrite``, suffix and directory conventions.
- Binning: the ``binAvg`` block mean rounded half away from zero and
  accumulated in 64-bit floats, and the ``binFloat`` 32-bit float
  block sum with its ``binning == 1`` cast.
- Behind ``KIKUCHIPY_EMSPHINX_DIR``: the acid test which indexes a
  kikuchipy written repack with ``IndexEBSD.exe`` and its negative
  controls, bitwise parity with ``PatternRepack.exe`` at binning one
  and two, and ``EBSPDims.exe`` against ``get_scan_info``.
"""

from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import subprocess
import tracemalloc
import warnings

import dask.array as da
import h5py
import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.indexing._spherical._namelist import EMSphInxNamelist
from kikuchipy.indexing._spherical._pattern_repack import write_emsphinx_patterns
from kikuchipy.io.plugins.oxford_binary import get_scan_info

# ------------------------- Frozen constants ------------------------- #

# The reader's vendor flip table (``pattern.hpp`` lines 463-471) as
# the writer must resolve ``flip=None``: ``True`` means the rows are
# written reversed, because the reader does not reverse them itself.
# Both routes were measured to index correctly, and the wrong pairing
# to a median of about 39.6 degrees (D1, D7).
MANUFACTURER_FLIP = {
    "EDAX": False,
    "EMsoft": False,
    "Oxford": True,
    "Bruker": True,
    "Bruker Nano": True,
    "DREAM.3D": True,
}

# Strings which are *not* ``Manufacturer`` values.  ``tsl`` and
# ``TSL`` are namelist ``vendor`` values, a different whitelist;
# ``IndexEBSD.exe`` exits 1 with ``unknown EBSD vendor: <s>`` for any
# of them (measured with ``"kikuchipy"``).
UNKNOWN_MANUFACTURERS = ["kikuchipy", "tsl", "TSL", "Bruker nano", ""]

# Part of the warning every write of a data type other than
# ``uint8`` must emit: EMSphInx reads HDF5 patterns through a
# buffered ``NATIVE_UINT8`` read and corrupts everything else
# (measured: 38.9 degree median garbage from a uint16 twin, D2)
NON_UINT8_WARNING = "8-bit"

# The in-package ``.ebsp`` is the ``nickel_ebsd_small`` map, and
# ``PatternRepack.exe`` repacks it to a (9, 60, 60) uint8 data set of
# 34544 B at data offset 2144, rows flipped (D2)
PATTERN_REPACK_BIN1_SIZE = 34544
PATTERN_REPACK_BIN2_SIZE = 10244

# Acid test bands on the canonical default route (``Manufacturer``
# EMsoft, rows unflipped, namelist vendor Bruker with ``pc_average``,
# ``nthread=1 batchsize=1``, ``bw`` 68).  Measured refined against
# the stored crystal map: median 0.7245 and maximum 0.9479 degrees,
# scores mean 0.6283; the bands carry the Phase 6 margin convention
# of about 1.7x (D7).
ACID_MEDIAN_DEG = 1.2
ACID_MAX_DEG = 1.6
ACID_SCORES_MEAN = 0.628

# A wrong flip pairing was measured at a median of 39.6 degrees, so
# ten is a discrimination threshold, not a pinned value (D7)
WRONG_FLIP_MEDIAN_DEG = 10.0

# Peak fraction of the whole map a lazy write may allocate.  A
# materialising implementation needs the full map, a chunk streaming
# one one slab.  Measured on this fixture with three plausible
# writers: ``da.store`` on the default threaded scheduler 0.160, the
# same synchronous 0.144, an explicit ``arr.blocks`` loop 0.281, and
# a full materialisation 1.017 -- so 0.5 has at least 1.7x margin
# over the slowest streaming route, is not thread count sensitive
# and still kills the materialising mutant (D9).
LAZY_PEAK_FRACTION = 0.5

NI_SHT = "emsphinx/ni_small_20kv_bw384.sht"


# ----------------------------- Helpers ------------------------------ #


def ni_signal():
    """Return a fresh ``nickel_ebsd_small`` signal, (3, 3) unsigned
    8-bit patterns of 60 x 60 pixels.
    """
    return kp.data.nickel_ebsd_small()


def ni_background_removed():
    """Return the small map with both backgrounds removed, the input
    of the acid test.
    """
    signal = ni_signal()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    return signal


def ebsp_path():
    """Return the path of the in-package ``patterns.ebsp``, the same
    nine patterns as ``nickel_ebsd_small``.
    """
    return Path(kp.data.__file__).parent / "oxford_binary" / "patterns.ebsp"


def signal_of(data):
    """Return an EBSD signal wrapping ``data`` without copying it."""
    return kp.signals.EBSD(data)


def patterns_of(fpath):
    """Return the ``/patterns`` data set of a written file."""
    with h5py.File(fpath, mode="r") as f:
        return np.asarray(f["patterns"])


def written(tmp_path, signal, name="patterns.h5", **kwargs):
    """Write ``signal`` into ``tmp_path`` and return the file path."""
    fpath = Path(tmp_path) / name
    kwargs.setdefault("overwrite", True)
    write_emsphinx_patterns(fpath, signal, **kwargs)
    return fpath


def block_sums(data, binning, accumulate=np.float64):
    """Return the ``binning`` x ``binning`` block sums of a stack of
    patterns, accumulated in ``accumulate``.

    The oracle both binning modes are pinned against: ``binAvg``
    accumulates in 64-bit floats, divides by ``binning ** 2`` and
    rounds half away from zero for integer types, while ``binFloat``
    accumulates in 32-bit floats and writes the sum as it is.  The
    C++ ``binFloat`` sums each input row with ``std::accumulate``,
    which may differ from NumPy's pairwise summation in the last
    place; the mode is dead code in the shipped binary, so this
    oracle is the authority (D1).
    """
    n, h, w = data.shape
    reshaped = data.astype(accumulate).reshape(
        n, h // binning, binning, w // binning, binning
    )
    return reshaped.sum(axis=(2, 4), dtype=accumulate)


def bin_avg(data, binning):
    """Return the frozen ``binAvg`` oracle: the block mean rounded
    half away from zero, back in the input data type.

    ``numpy.round`` is banker's rounding and is **not** this: the two
    differ on 1003 of the 8100 pixels of the nickel map at binning
    two, and ``PatternRepack.exe`` agrees with this one bitwise.
    """
    mean = block_sums(data, binning) / binning**2
    if np.issubdtype(data.dtype, np.integer):
        mean = np.floor(mean + 0.5)
    return mean.astype(data.dtype)


def read_ang_scores(array):
    """Return the cross correlation column of a read ``.ang``."""
    return array[:, 6]


def ang_rotations(array):
    """Return the orientations of a read ``.ang``, whose first three
    columns are Bunge Euler angles in radians.
    """
    return Rotation.from_euler(array[:, :3])


def misorientation(rotations, reference):
    """Return the m-3m reduced misorientation in degrees between two
    rotation sets of the same shape.
    """
    angles = Orientation(rotations.data, Oh).angle_with(
        Orientation(reference.data, Oh), degrees=True
    )
    return np.asarray(angles, dtype=np.float64).ravel()


def acid_namelist(detector, **kwargs):
    """Return the acid test namelist: the canonical route with the
    deterministic single thread configuration.
    """
    parameters = dict(
        pattern_file="patterns.h5",
        master_files=["ni.sht"],
        detector=detector,
        scan_shape=(3, 3),
        scan_steps=(1.5, 1.5),
        data_file="out.h5",
        vendor_file="out.ang",
        n_thread=1,
        batch_size=1,
        bandwidth=68,
        normalize=True,
        refine=True,
        n_regions=10,
        gaussian_background=False,
        circular_mask=False,
    )
    parameters.update(kwargs)
    return EMSphInxNamelist.from_kwargs(**parameters)


def build_acid_files(tmp_path, flip=None, manufacturer="EMsoft", **namelist_kwargs):
    """Write the three input files of the acid test into ``tmp_path``
    and return the namelist file name.
    """
    signal = ni_background_removed()
    write_emsphinx_patterns(
        tmp_path / "patterns.h5",
        signal,
        manufacturer=manufacturer,
        flip=flip,
        overwrite=True,
    )
    shutil.copy(Path(Dataset(NI_SHT).fetch_file_path()), tmp_path / "ni.sht")
    detector = signal.detector.deepcopy()
    namelist = acid_namelist(detector, **namelist_kwargs)
    namelist.write(tmp_path / "index.nml", overwrite=True)
    return "index.nml"


def run_index_ebsd(program, tmp_path, namelist="index.nml"):
    """Run ``IndexEBSD`` in ``tmp_path`` and return the result."""
    return subprocess.run(
        [str(program), namelist],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


@contextmanager
def no_user_warning():
    """Turn a ``UserWarning`` into an error for the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


# --------------------- The written file (D1, D2) -------------------- #


class TestWriteEmsphinxPatterns:
    def test_layout_is_contiguous_alloc_early(self, tmp_path):
        # zero filters is the functional guard: the reader's memory
        # map gate is dead code, so every HDF5 file is read through
        # the buffered branch, which fails on a compressed data set
        # (measured).  Contiguous with early allocation is byte
        # layout parity with ``PatternRepack.exe`` (D2).
        fpath = written(tmp_path, ni_signal())
        with h5py.File(fpath, mode="r") as f:
            dataset = f["patterns"]
            plist = dataset.id.get_create_plist()
            assert plist.get_layout() == h5py.h5d.CONTIGUOUS
            assert plist.get_alloc_time() == h5py.h5d.ALLOC_TIME_EARLY
            assert plist.get_nfilters() == 0
            assert dataset.chunks is None
            assert dataset.compression is None
            assert dataset.shape == (9, 60, 60)
            assert dataset.dtype == np.uint8
            offset = dataset.id.get_offset()
            assert isinstance(offset, int)
            assert offset > 0

    def test_the_file_holds_exactly_two_objects(self, tmp_path):
        fpath = written(tmp_path, ni_signal())
        with h5py.File(fpath, mode="r") as f:
            assert sorted(f.keys()) == ["Manufacturer", "patterns"]

    def test_default_route_dataset_equals_signal_bytes(self, tmp_path):
        # the default ``manufacturer="EMsoft"`` resolves ``flip`` to
        # ``False``, which is what makes this the cheapest write and
        # the canonical route of the acid test and Phase 10
        signal = ni_signal()
        fpath = written(tmp_path, signal)
        expected = signal.data.reshape((9, 60, 60))
        assert np.array_equal(patterns_of(fpath), expected)

    def test_flip_true_reverses_rows(self, tmp_path):
        # rows, not columns: ``flipPat`` swaps whole rows
        signal = ni_signal()
        fpath = written(tmp_path, signal, flip=True)
        data = signal.data.reshape((9, 60, 60))
        assert np.array_equal(patterns_of(fpath), data[:, ::-1, :])
        assert not np.array_equal(patterns_of(fpath), data[..., ::-1])

    @pytest.mark.parametrize("manufacturer", sorted(MANUFACTURER_FLIP))
    def test_manufacturer_auto_flip_table(self, tmp_path, manufacturer):
        signal = ni_signal()
        fpath = written(tmp_path, signal, manufacturer=manufacturer)
        data = signal.data.reshape((9, 60, 60))
        expected = data[:, ::-1, :] if MANUFACTURER_FLIP[manufacturer] else data
        assert np.array_equal(patterns_of(fpath), expected)

    @pytest.mark.parametrize("manufacturer", sorted(MANUFACTURER_FLIP))
    def test_manufacturer_is_a_root_scalar_vlen_dataset(self, tmp_path, manufacturer):
        # the character set is load bearing: h5py's default variable
        # length UTF-8 string makes ``IndexEBSD.exe`` exit 1 with a
        # misleading ``H5Dread failed`` (measured, D1)
        fpath = written(tmp_path, ni_signal(), manufacturer=manufacturer)
        with h5py.File(fpath, mode="r") as f:
            assert "Manufacturer" in f
            dataset = f["Manufacturer"]
            assert isinstance(dataset, h5py.Dataset)
            assert dataset.shape == ()
            assert dataset.id.get_space().get_simple_extent_type() == h5py.h5s.SCALAR
            info = h5py.check_string_dtype(dataset.dtype)
            assert info is not None
            assert info.length is None
            assert info.encoding == "ascii"
            assert dataset.id.get_type().get_cset() == h5py.h5t.CSET_ASCII
            value = dataset[()]
            assert value.decode("ascii") == manufacturer
            assert not f.attrs

    @pytest.mark.parametrize("manufacturer", UNKNOWN_MANUFACTURERS)
    def test_unknown_manufacturer_raises(self, tmp_path, manufacturer):
        with pytest.raises(ValueError, match="anufacturer"):
            written(tmp_path, ni_signal(), manufacturer=manufacturer)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float64, np.uint32])
    def test_bad_dtype_raises(self, tmp_path, dtype):
        signal = signal_of(np.zeros((2, 4, 4), dtype=dtype))
        with pytest.raises(ValueError, match="uint8"):
            written(tmp_path, signal)

    @pytest.mark.parametrize("dtype", [np.uint16, np.float32])
    def test_non_uint8_write_warns(self, tmp_path, dtype):
        # the C++ program writes these happily, so the port writes
        # them too; only the warning is ours (D2, open question 5)
        signal = signal_of(np.ones((2, 4, 4), dtype=dtype))
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal)
        assert patterns_of(fpath).dtype == dtype

    def test_uint8_write_does_not_warn(self, tmp_path):
        signal = signal_of(np.ones((2, 4, 4), dtype=np.uint8))
        with no_user_warning():
            written(tmp_path, signal)

    @pytest.mark.parametrize("dtype", [">u2", ">f4"])
    def test_byteswapped_input_writes_native(self, tmp_path, dtype):
        # ``PatternFile::Read`` compares against ``NATIVE_*`` types,
        # so a big endian data set exits 1 with "only uint8, uint16,
        # and float hdf patterns are supported" (measured, D1)
        data = (np.arange(32, dtype=np.float64).reshape((2, 4, 4)) + 1).astype(dtype)
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal_of(data))
        with h5py.File(fpath, mode="r") as f:
            written_dtype = f["patterns"].dtype
        assert written_dtype.byteorder in ("=", "|")
        assert written_dtype == np.dtype(dtype).newbyteorder("=")
        assert np.array_equal(patterns_of(fpath), data.astype(written_dtype))

    @pytest.mark.parametrize("binning", [0, -1, 7, 8, 11])
    def test_binning_must_divide(self, tmp_path, binning):
        with pytest.raises(ValueError, match="[Bb]inning"):
            written(tmp_path, ni_signal(), binning=binning)

    def test_navigation_orders(self, tmp_path):
        # a column major flatten permutes rows 1 to 8 here
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape((2, 3, 4, 5))
        fpath = written(tmp_path, signal_of(data), name="two_dimensional.h5")
        assert np.array_equal(patterns_of(fpath), data.reshape((6, 4, 5)))

        one_dimensional = data[0]
        fpath = written(tmp_path, signal_of(one_dimensional), name="one.h5")
        assert np.array_equal(patterns_of(fpath), one_dimensional)

        single = data[0, 0]
        fpath = written(tmp_path, signal_of(single), name="zero.h5")
        assert patterns_of(fpath).shape == (1, 4, 5)
        assert np.array_equal(patterns_of(fpath)[0], single)

    def test_lazy_write_equals_eager(self, tmp_path):
        signal = ni_signal()
        eager = written(tmp_path, signal, name="eager.h5")
        lazy_signal = kp.signals.LazyEBSD(
            da.from_array(signal.data, chunks=(2, 1, 60, 60))
        )
        lazy = written(tmp_path, lazy_signal, name="lazy.h5")
        assert np.array_equal(patterns_of(eager), patterns_of(lazy))

    def test_lazy_write_does_not_materialise(self, tmp_path, record_property):
        # byte equality cannot see a materialising implementation
        shape = (64, 32, 60, 60)
        chunks = (8, 32, 60, 60)
        signal = kp.signals.LazyEBSD(da.zeros(shape, dtype=np.uint8, chunks=chunks))
        full_bytes = int(np.prod(shape))
        tracemalloc.start()
        try:
            written(tmp_path, signal, name="lazy_big.h5")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        record_property("lazy_write_peak_bytes", peak)
        record_property("lazy_write_full_bytes", full_bytes)
        assert peak < LAZY_PEAK_FRACTION * full_bytes

    # ------------------ The writer conventions (D1) ----------------- #

    def test_suffix_defaulting(self, tmp_path):
        write_emsphinx_patterns(tmp_path / "patterns", ni_signal(), overwrite=True)
        assert (tmp_path / "patterns.h5").is_file()
        assert not (tmp_path / "patterns").is_file()

    def test_parent_directory_is_created(self, tmp_path):
        fpath = tmp_path / "a" / "b" / "patterns.h5"
        write_emsphinx_patterns(fpath, ni_signal(), overwrite=True)
        assert fpath.is_file()

    def test_overwrite_false_leaves_file(self, tmp_path):
        fpath = written(tmp_path, ni_signal())
        before = fpath.read_bytes()
        other = signal_of(np.zeros((2, 4, 4), dtype=np.uint8))
        write_emsphinx_patterns(fpath, other, overwrite=False)
        assert fpath.read_bytes() == before

    def test_overwrite_true_replaces(self, tmp_path):
        fpath = written(tmp_path, ni_signal())
        other = signal_of(np.zeros((2, 4, 4), dtype=np.uint8))
        write_emsphinx_patterns(fpath, other, overwrite=True)
        assert patterns_of(fpath).shape == (2, 4, 4)

    def test_overwrite_false_writes_a_new_file(self, tmp_path):
        fpath = tmp_path / "new.h5"
        write_emsphinx_patterns(fpath, ni_signal(), overwrite=False)
        assert fpath.is_file()

    def test_overwrite_none_asks_and_does_not_overwrite(self, tmp_path, monkeypatch):
        # ``_get_input_bool`` warns and keeps the file when the
        # terminal cannot be read.  The unanswerable prompt is forced
        # rather than left to pytest's captured standard input, which
        # would block forever under ``pytest -s``
        def no_terminal(*args, **kwargs):
            raise OSError("no raw input")

        monkeypatch.setattr("builtins.input", no_terminal)
        fpath = written(tmp_path, ni_signal())
        before = fpath.read_bytes()
        with pytest.warns(UserWarning, match="raw input"):
            write_emsphinx_patterns(fpath, ni_signal(), overwrite=None)
        assert fpath.read_bytes() == before

    def test_bad_overwrite_raises(self, tmp_path):
        fpath = written(tmp_path, ni_signal())
        with pytest.raises(ValueError, match="overwrite"):
            write_emsphinx_patterns(fpath, ni_signal(), overwrite="yes")


# --------------------------- Binning (D1) --------------------------- #


class TestBinning:
    def test_binavg_rounds_half_away_from_zero(self, tmp_path):
        # two blocks with exact half means, 0.5 and 2.5: rounding
        # half away from zero gives 1 and 3, banker's rounding 0 and
        # 2, so ``numpy.round`` dies here
        data = np.array(
            [[[1, 1, 3, 3], [0, 0, 2, 2]]],
            dtype=np.uint8,
        )
        fpath = written(tmp_path, signal_of(data), binning=2)
        assert np.array_equal(patterns_of(fpath), np.array([[[1, 3]]], np.uint8))
        assert np.array_equal(patterns_of(fpath), bin_avg(data, 2))

    def test_binavg_of_the_nickel_map_equals_the_oracle(self, tmp_path):
        # the recipe measured bitwise equal to ``PatternRepack.exe``
        # at binning two, where banker's rounding differs on 1003 of
        # the 8100 pixels (D2)
        signal = ni_signal()
        data = signal.data.reshape((9, 60, 60))
        fpath = written(tmp_path, signal, binning=2)
        binned = patterns_of(fpath)
        assert binned.shape == (9, 30, 30)
        assert binned.dtype == np.uint8
        assert np.array_equal(binned, bin_avg(data, 2))
        banker = np.round(block_sums(data, 2) / 4).astype(np.uint8)
        assert int((binned != banker).sum()) == 1003

    @pytest.mark.parametrize("binning", [2, 3, 4, 5, 6])
    def test_binavg_and_flip_commute(self, tmp_path, binning):
        # measured bitwise for binning two to six on the nickel
        # fixture, which is why the contract is the equivalence and
        # not an internal order (D1)
        signal = ni_signal()
        data = signal.data.reshape((9, 60, 60))
        fpath = written(tmp_path, signal, binning=binning, flip=True)
        assert np.array_equal(patterns_of(fpath), bin_avg(data, binning)[:, ::-1, :])

    def test_binavg_does_not_overflow_the_input_dtype(self, tmp_path):
        # every block sums to more than 255 (1020 and 1010), which an
        # accumulation in the input data type would wrap.  Only that
        # mutant dies here: both sums are exact in 32-bit floats and
        # even in ``uint16``, so the width of the *floating point*
        # accumulator is unobservable for ``uint8`` input at any
        # binning a legal pattern shape allows.  The C++
        # ``std::vector<double>`` is pinned by the float input test
        # below instead.
        data = np.array([[[255, 255, 255, 251], [255, 255, 250, 250]]], np.uint8)
        fpath = written(tmp_path, signal_of(data), binning=2)
        assert np.array_equal(patterns_of(fpath), np.array([[[255, 252]]], np.uint8))
        assert int(block_sums(data, 2).max()) == 1020

    def test_binavg_accumulates_in_float64(self, tmp_path):
        # the block is ``2**24, 1, 1, 2``: in 64-bit floats it sums to
        # 16777220 and the mean is exactly 4194305, while *every*
        # 32-bit float accumulation of it loses the odd bits and gives
        # 4194304.5 (16777218, pairwise or left to right alike).  The
        # mean is exact in 32-bit floats either way, so the written
        # value alone separates the two accumulators (D1).
        data = np.array([[[2.0**24, 1.0], [1.0, 2.0]]], dtype=np.float32)
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal_of(data), binning=2)
        binned = patterns_of(fpath)
        assert binned.dtype == np.float32
        assert np.array_equal(binned, np.array([[[4194305.0]]], np.float32))
        assert np.array_equal(binned, bin_avg(data, 2))
        assert not np.array_equal(
            binned, (block_sums(data, 2, np.float32) / 4).astype(np.float32)
        )

    def test_binavg_binning_one_copies(self, tmp_path):
        signal = ni_signal()
        fpath = written(tmp_path, signal, binning=1)
        assert np.array_equal(patterns_of(fpath), signal.data.reshape((9, 60, 60)))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_binfloat_sums_in_float32(self, tmp_path, dtype):
        # ``binFloat`` writes the block **sum**, not the mean, and
        # always in 32-bit floats.  The warning is keyed on the
        # **written** data type, so a ``uint8`` input warns here too:
        # what the reader corrupts is the file, not the signal.
        data = (np.arange(2 * 4 * 6, dtype=np.float64).reshape((2, 4, 6)) + 1).astype(
            dtype
        )
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal_of(data), binning=2, bin_to_float=True)
        binned = patterns_of(fpath)
        assert binned.dtype == np.float32
        assert binned.shape == (2, 2, 3)
        assert np.array_equal(binned, block_sums(data, 2, np.float32))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_binfloat_binning_one_casts_to_float32(self, tmp_path, dtype):
        # a deliberate completion of dead code: the C++ ``main``
        # never routes ``binning == 1`` through ``binFloat`` (D1)
        data = (np.arange(2 * 4 * 6, dtype=np.float64).reshape((2, 4, 6)) + 1).astype(
            dtype
        )
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal_of(data), bin_to_float=True)
        binned = patterns_of(fpath)
        assert binned.dtype == np.float32
        assert np.array_equal(binned, data.astype(np.float32))

    def test_binfloat_of_a_float_input_uses_numpy_summation(self, tmp_path):
        # the mode is unreachable in the shipped binary, so the NumPy
        # pairwise summation of the oracle is the authority (D1)
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=(3, 8, 8)).astype(np.float32)
        with pytest.warns(UserWarning, match=NON_UINT8_WARNING):
            fpath = written(tmp_path, signal_of(data), binning=4, bin_to_float=True)
        assert np.array_equal(patterns_of(fpath), block_sums(data, 4, np.float32))

    def test_bin_to_float_binning_must_divide(self, tmp_path):
        with pytest.raises(ValueError, match="[Bb]inning"):
            written(tmp_path, ni_signal(), binning=7, bin_to_float=True)


# ------------- The EMSphInx binaries (D7, D9, local only) ----------- #


class TestAgainstEmsphinxBinaries:
    """Every test here needs ``KIKUCHIPY_EMSPHINX_DIR`` and runs the
    program with ``cwd=tmp_path``: ``IndexEBSD -t`` writes to a hard
    coded relative path and namelist paths resolve against the
    process working directory (D9).
    """

    def test_emsphinx_binaries_index_ebsd_accepts_kikuchipy_repack(
        self, emsphinx_program, read_ang, tmp_path, record_property
    ):
        program = emsphinx_program("IndexEBSD")
        namelist = build_acid_files(tmp_path)
        result = run_index_ebsd(program, tmp_path, namelist)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Vertical Flip: true" in result.stdout

        array = read_ang(tmp_path / "out.ang")
        assert array.shape == (9, 8)
        angles = misorientation(ang_rotations(array), ni_signal().xmap.rotations)
        scores = read_ang_scores(array)
        record_property("acid_per_point", ", ".join(f"{a:.4f}" for a in angles))
        record_property("acid_median", f"{np.median(angles):.4f}")
        record_property("acid_max", f"{angles.max():.4f}")
        record_property("acid_scores_mean", f"{scores.mean():.4f}")
        record_property("acid_stdout", result.stdout)
        assert np.median(angles) < ACID_MEDIAN_DEG
        assert angles.max() < ACID_MAX_DEG
        assert scores.mean() == pytest.approx(ACID_SCORES_MEAN, rel=0.05)

    def test_emsphinx_binaries_wrong_flip_pairing_is_discriminated(
        self, emsphinx_program, read_ang, tmp_path, record_property
    ):
        # the same run with the flip forced wrong still exits 0, so
        # the contract is only observable in the orientations
        program = emsphinx_program("IndexEBSD")
        namelist = build_acid_files(tmp_path, flip=True)
        result = run_index_ebsd(program, tmp_path, namelist)
        assert result.returncode == 0, result.stdout + result.stderr
        array = read_ang(tmp_path / "out.ang")
        angles = misorientation(ang_rotations(array), ni_signal().xmap.rotations)
        record_property("wrong_flip_median", f"{np.median(angles):.4f}")
        assert np.median(angles) > WRONG_FLIP_MEDIAN_DEG

    def test_emsphinx_binaries_index_ebsd_rejects_missing_manufacturer(
        self, emsphinx_program, tmp_path
    ):
        program = emsphinx_program("IndexEBSD")
        namelist = build_acid_files(tmp_path)
        with h5py.File(tmp_path / "patterns.h5", mode="a") as f:
            del f["Manufacturer"]
        result = run_index_ebsd(program, tmp_path, namelist)
        assert result.returncode != 0
        assert "doesn't have a Manufacturer string" in result.stdout + result.stderr

    def test_emsphinx_binaries_index_ebsd_rejects_unknown_manufacturer(
        self, emsphinx_program, tmp_path
    ):
        program = emsphinx_program("IndexEBSD")
        namelist = build_acid_files(tmp_path)
        with h5py.File(tmp_path / "patterns.h5", mode="a") as f:
            dtype = f["Manufacturer"].dtype
            del f["Manufacturer"]
            f.create_dataset("Manufacturer", data="kikuchipy", dtype=dtype)
        result = run_index_ebsd(program, tmp_path, namelist)
        assert result.returncode != 0
        assert "unknown EBSD vendor" in result.stdout + result.stderr

    @pytest.mark.parametrize("vendor", ["EMsoft", "EDAX", "Oxford", "tsl"])
    def test_emsphinx_binaries_vendor_namelists_are_equivalent(
        self, emsphinx_program, read_ang, tmp_path, vendor
    ):
        # the four conversions describe the same geometry, so the
        # Euler columns were measured bitwise equal (D6)
        program = emsphinx_program("IndexEBSD")
        namelist = build_acid_files(tmp_path)
        assert run_index_ebsd(program, tmp_path, namelist).returncode == 0
        reference = read_ang(tmp_path / "out.ang")[:, :3]

        namelist = build_acid_files(tmp_path, vendor=vendor)
        result = run_index_ebsd(program, tmp_path, namelist)
        assert result.returncode == 0, result.stdout + result.stderr
        assert np.array_equal(read_ang(tmp_path / "out.ang")[:, :3], reference)

    @pytest.mark.parametrize("binning", [1, 2])
    def test_emsphinx_binaries_pattern_repack_binary_parity(
        self, emsphinx_program, tmp_path, binning
    ):
        # ``PatternRepack.exe`` hard codes ``flip = true`` and writes
        # no ``Manufacturer``, so only ``/patterns`` is compared
        program = emsphinx_program("PatternRepack")
        result = subprocess.run(
            [str(program), str(ebsp_path()), "repacked.h5", str(binning)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        theirs = patterns_of(tmp_path / "repacked.h5")
        ours = patterns_of(written(tmp_path, ni_signal(), flip=True, binning=binning))
        assert theirs.dtype == ours.dtype
        assert np.array_equal(theirs, ours)
        expected_size = (
            PATTERN_REPACK_BIN1_SIZE if binning == 1 else PATTERN_REPACK_BIN2_SIZE
        )
        assert (tmp_path / "repacked.h5").stat().st_size == expected_size

    def test_emsphinx_binaries_ebsp_dims_binary_parity(
        self, emsphinx_program, tmp_path
    ):
        program = emsphinx_program("EBSPDims")
        result = subprocess.run(
            [str(program), str(ebsp_path())],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        info = get_scan_info(ebsp_path())
        found = re.search(r"found (\d+) patterns", result.stdout)
        coordinates = re.search(r"found (\d+) x and (\d+) y coordinates", result.stdout)
        assert found is not None and coordinates is not None
        assert int(found.group(1)) == info["n_patterns"]
        assert int(coordinates.group(1)) == len(info["beam_x"])
        assert int(coordinates.group(2)) == len(info["beam_y"])
        assert f"bytes : {info['pattern_bytes']}" in result.stdout
        # the coordinate lists are printed only for an irregular grid
        assert info["is_regular_grid"]
        assert "\nX:" not in result.stdout
