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

# ------------------- Explanation of file location ------------------- #
# Why is this file located in the top directory and not in tests/?
# Because if it was, running "pytest --doctest-modules src" wouldn't
# discover this file.
#
# An unwanted side-effect of this is that test files cannot import
# anything from the conftest file.

import os
from pathlib import Path
import tempfile

# ---------------- Per-worker Numba cache under pytest-xdist ---------- #
# Numba writes cache=True kernels (ours, and orix's dynamic gufuncs
# such as ``qu_conj_gufunc``) to a shared on-disk cache. When several
# xdist workers compile the same kernel at the same time on a fresh
# machine (CI), they race on the cache index and a worker may load a
# half-written kernel and crash with an access violation. Giving each
# worker its own cache directory removes the race; it has no effect
# without xdist, and must happen before numba is imported.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER and "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(
        Path(tempfile.gettempdir()) / "kikuchipy-numba-cache" / _XDIST_WORKER
    )
# The directory chosen above, restored after every test by the autouse
# fixture ``_keep_numba_cache_dir_per_worker`` below
_WORKER_NUMBA_CACHE_DIR = os.environ.get("NUMBA_CACHE_DIR") if _XDIST_WORKER else None

from contextlib import contextmanager
import functools
import hashlib
from io import TextIOWrapper
import json
from numbers import Number
import subprocess
import threading
import time
from typing import Callable, Generator, Literal

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
import h5py
import hyperspy.api as hs
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy._constants import dependency_version
from kikuchipy.data._data import get_fetching_pooch
from kikuchipy.data._dummy_files.bruker_h5ebsd import (
    create_dummy_bruker_h5ebsd_file,
    create_dummy_bruker_h5ebsd_nonrectangular_roi_file,
    create_dummy_bruker_h5ebsd_roi_file,
)
from kikuchipy.data._dummy_files.oxford_h5ebsd import create_dummy_oxford_h5ebsd_file
from kikuchipy.draw._vtk import system_supports_plotting
from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group

DATA_PATH = Path(kp.data.__file__).parent.resolve()


# ---------------------- Control test selection ---------------------- #


def pytest_addoption(parser):
    # Flags for optional markers. Markers requiring something (installed
    # dependency, e.g.) should not have a flag to still run them.
    parser.addoption(
        "--gpu",
        action="store_true",
        help="Run tests that should run only when wgpu has a GPU available",
    )
    parser.addoption(
        "--weekly",
        action="store_true",
        help="Run tests that should run only weekly",
    )


MARKERS = [
    "gpu",
    "weekly",
]


def pytest_runtest_setup(item):
    """Skip certain tests when flag is missing:
    https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_runtest_setup.

    To run tests marked by this marker *only*, say, `gpu`, do
    `pytest -m gpu --gpu`.
    """
    for marker in MARKERS:
        marker_str = f"--{marker}"
        if marker in item.keywords and not item.config.getoption(
            marker_str, default=False
        ):
            pytest.skip(f"Needs {marker_str} flag to run")


# ----------------------------- PyVista ------------------------------ #


if dependency_version["pyvista"] is not None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.interactive = False


@pytest.fixture(autouse=True)
def _keep_numba_cache_dir_per_worker():
    """Re-assert the worker's Numba cache directory after each test.

    Importing several PyEBSDIndex modules (``band_detect``,
    ``tripletvote`` and others) sets ``NUMBA_CACHE_DIR`` to one
    directory shared by every process on the machine, which re-opens
    the cache race the block at the top of this file closes: kernels
    compiled later in the same worker (ours and orix's gufuncs) are then
    written to, and loaded from, the shared directory while other
    workers do the same. Restoring the worker's directory after each
    test keeps later compilations in the worker's own cache.
    """
    yield
    expected = _WORKER_NUMBA_CACHE_DIR
    if expected is not None and os.environ.get("NUMBA_CACHE_DIR") != expected:
        os.environ["NUMBA_CACHE_DIR"] = expected
        from numba.core import config as numba_config

        numba_config.reload_config()


@pytest.fixture(autouse=False)
def skipif_no_vtk_support():
    if not system_supports_plotting():
        pytest.skip("System does not support VTK plotting")


# --------------------------- pytest hooks --------------------------- #


def pytest_sessionstart(session):
    _ = kp.data.nickel_ebsd_large(allow_download=True)
    _ = kp.data.si_ebsd_moving_screen(0, allow_download=True)
    _ = kp.data.si_ebsd_moving_screen(5, allow_download=True)
    _ = kp.data.si_ebsd_moving_screen(10, allow_download=True)
    plt.rcParams.update({"backend": "agg", "figure.max_open_warning": False})


# ---------------------- pytest doctest-modules ---------------------- #


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    # Temporarily turn off interactive plotting with Matplotlib
    plt.ioff()

    # Temporarily suppress HyperSpy's progressbar
    hs.preferences.General.show_progressbar = False

    # Temporary directory for saving files in
    temporary_directory = tempfile.TemporaryDirectory()
    original_directory = os.getcwd()
    os.chdir(temporary_directory.name)
    yield

    # Teardown
    os.chdir(original_directory)


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace) -> None:
    doctest_namespace["DATA_PATH"] = DATA_PATH / "kikuchipy_h5ebsd"


# ----------------------------- Fixtures ----------------------------- #


@pytest.fixture
def assert_dictionary_func() -> Callable:
    def assert_dictionary(dict1: dict, dict2: dict) -> None:
        """Assert that two dictionaries are (almost) equal.

        Used to compare signal's axes managers or metadata in tests.
        """
        for key in dict2.keys():
            if isinstance(dict2[key], dict):
                assert_dictionary(dict1[key], dict2[key])
            else:
                if isinstance(dict2[key], list) or isinstance(dict1[key], list):
                    dict2[key] = np.array(dict2[key])
                    dict1[key] = np.array(dict1[key])
                if isinstance(dict2[key], (np.ndarray, Number)):
                    assert np.allclose(dict1[key], dict2[key])
                else:
                    assert dict1[key] == dict2[key]

    return assert_dictionary


@pytest.fixture
def dummy_signal(
    dummy_background: np.ndarray,
) -> Generator[kp.signals.EBSD, None, None]:
    """Dummy signal of shape <3, 3|3, 3>. If this is changed, all
    tests using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """
    nav_shape = (3, 3)
    nav_size = int(np.prod(nav_shape))
    sig_shape = (3, 3)

    # fmt: off
    dummy_array = np.array(
        [
            5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2,
            9, 3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8,
            6, 0, 4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0,
            2, 9, 2, 9, 4, 3, 6, 5, 6, 2, 5, 9
        ],
        dtype=np.uint8
    ).reshape(nav_shape + sig_shape)
    # fmt: on

    # Initialize and set static background attribute
    s = kp.signals.EBSD(dummy_array, static_background=dummy_background)

    # Axes manager
    s.axes_manager.navigation_axes[1].name = "x"
    s.axes_manager.navigation_axes[0].name = "y"

    # Crystal map
    phase_list = PhaseList([Phase("a", space_group=225), Phase("b", space_group=227)])
    y, x = np.indices(nav_shape)
    s.xmap = CrystalMap(
        rotations=Rotation.identity((nav_size,)),
        # fmt: off
        phase_id=np.array(
            [
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, 0],
            ]
        ).ravel(),
        # fmt: on
        phase_list=phase_list,
        x=x.ravel(),
        y=y.ravel(),
    )
    pc = np.arange(np.prod(nav_shape) * 3).reshape(nav_shape + (3,))
    pc = pc.astype(float) / pc.max()
    s.detector = kp.detectors.EBSDDetector(shape=sig_shape, pc=pc)

    yield s


@pytest.fixture
def dummy_background() -> Generator[np.ndarray, None, None]:
    """Dummy static background image for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """
    yield np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture
def ebsd_with_axes_and_random_data(request) -> Generator[kp.signals.EBSD, None, None]:
    """EBSD signal with minimally defined axes and random data.

    Parameters expected in `request`
    -------------------------------
    navigation_shape : tuple
    signal_shape : tuple
    lazy : bool
    dtype : numpy.dtype
    """
    nav_shape, sig_shape, lazy, dtype = getattr(
        request, "param", [(3, 3), (3, 3), False, np.float32]
    )

    nav_ndim = len(nav_shape)
    sig_ndim = len(sig_shape)
    data_shape = nav_shape + sig_shape
    data_size = int(np.prod(data_shape))
    axes = []
    if nav_ndim == 1:
        axes.append({"name": "x", "size": nav_shape[0], "scale": 1})
    if nav_ndim == 2:
        axes.append({"name": "y", "size": nav_shape[0], "scale": 1})
        axes.append({"name": "x", "size": nav_shape[1], "scale": 1})
    if sig_ndim == 2:
        axes.append({"name": "dy", "size": sig_shape[0], "scale": 1})
        axes.append({"name": "dx", "size": sig_shape[1], "scale": 1})
    if np.issubdtype(dtype, np.integer):
        kw = {"low": 1, "high": 255, "size": data_size}
    else:
        kw = {"low": 0.1, "high": 1, "size": data_size}
    if lazy:
        data = da.random.uniform(**kw).reshape(data_shape).astype(dtype)
        s = kp.signals.LazyEBSD(data, axes=axes)
    else:
        data = np.random.uniform(**kw).reshape(data_shape).astype(dtype)
        s = kp.signals.EBSD(data, axes=axes)
    yield s


@pytest.fixture
def nickel_structure() -> Generator[Structure, None, None]:
    """A diffpy.structure with a Nickel crystal structure."""
    yield Structure(
        atoms=[Atom("Ni", [0, 0, 0])],
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    )


@pytest.fixture
def nickel_phase(nickel_structure) -> Generator[Phase, None, None]:
    yield Phase(name="ni", structure=nickel_structure, space_group=225)


@pytest.fixture
def pc1() -> Generator[list[float], None, None]:
    """One projection center (PC) in TSL convention."""
    yield [0.4210, 0.7794, 0.5049]


@pytest.fixture
def detector(request, pc1) -> Generator[kp.detectors.EBSDDetector, None, None]:
    """An EBSD detector of a given shape with a number of PCs given by
    a navigation shape.
    """
    nav_shape, sig_shape = getattr(request, "param", [(1,), (60, 60)])
    yield kp.detectors.EBSDDetector(
        shape=sig_shape,
        binning=8,
        px_size=70,
        pc=np.ones(nav_shape + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )


@pytest.fixture
def rotations() -> Generator[Rotation, None, None]:
    yield Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture
def get_single_phase_xmap(rotations) -> Generator[Callable, None, None]:
    def _get_single_phase_xmap(
        nav_shape,
        rotations_per_point=5,
        prop_names=("scores", "simulation_indices"),
        name="a",
        space_group=225,
        phase_id=0,
        step_sizes=None,
    ):
        d, map_size = create_coordinate_arrays(shape=nav_shape, step_sizes=step_sizes)
        rot_idx = np.random.choice(
            np.arange(rotations.size), map_size * rotations_per_point
        )
        data_shape = (map_size,)
        if rotations_per_point > 1:
            data_shape += (rotations_per_point,)
        d["rotations"] = rotations[rot_idx].reshape(*data_shape)
        d["phase_id"] = np.ones(map_size) * phase_id
        d["phase_list"] = PhaseList(Phase(name=name, space_group=space_group))
        # Scores and simulation indices
        d["prop"] = {
            prop_names[0]: np.ones(data_shape, dtype=np.float32),
            prop_names[1]: np.arange(np.prod(data_shape)).reshape(data_shape),
        }
        return CrystalMap(**d)

    yield _get_single_phase_xmap


# ---------------------------- IO fixtures --------------------------- #


@pytest.fixture
def save_path_hdf5(request, tmpdir) -> Generator[Path, None, None]:
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    ext = getattr(request, "param", "h5")
    yield Path(tmpdir / f"patterns.{ext}")


@pytest.fixture()
def save_path_nordif(tmpdir) -> Generator[Path, None, None]:
    yield Path(tmpdir / "nordif/save_temp.dat")


@pytest.fixture
def ni_small_axes_manager() -> Generator[dict, None, None]:
    """Axes manager for :func:`kikuchipy.data.nickel_ebsd_small`."""
    names = ["y", "x", "dy", "dx"]
    scales = [1.5, 1.5, 1, 1]
    sizes = [3, 3, 60, 60]
    navigates = [True, True, False, False]
    axes_manager = {}
    for i in range(len(names)):
        axes_manager[f"axis-{i}"] = {
            "_type": "UniformDataAxis",
            "name": names[i],
            "units": "um",
            "navigate": navigates[i],
            "is_binned": False,
            "size": sizes[i],
            "scale": scales[i],
            "offset": 0.0,
        }
    yield axes_manager


@pytest.fixture
def ebsd_directory(tmpdir, request) -> Generator[Path, None, None]:
    """Temporary directory with EBSD files as .tif, .png or .bmp files.

    Parameters expected in `request`
    -------------------------------
    xy_pattern : str
    nav_shape : tuple of ints
    """
    s = kp.data.nickel_ebsd_small()
    s.unfold_navigation_space()

    xy_pattern, nav_shape = getattr(request, "param", ("_x{}y{}.tif", (3, 3)))
    y, x = np.indices(nav_shape)
    x = x.ravel()
    y = y.ravel()
    for i in range(s.axes_manager.navigation_size):
        fname = str(tmpdir / ("pattern" + xy_pattern.format(x[i], y[i])))
        iio.imwrite(fname, s.data[i])

    yield tmpdir


# ------------------------ kikuchipy formats ------------------------- #


@pytest.fixture
def kikuchipy_h5ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "kikuchipy_h5ebsd"


@pytest.fixture
def nickel_ebsd_large_h5ebsd_renamed() -> Generator[Path, None, None]:
    marshall = get_fetching_pooch()
    f1 = Path(marshall.path) / "data/nickel_ebsd_large/patterns.h5"
    f2 = f1.rename(f1.with_suffix(".bak"))
    yield f2
    f2.rename(f1)


# --------------------------- EDAX formats --------------------------- #


@pytest.fixture
def edax_binary_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "edax_binary"


@pytest.fixture
def edax_binary_file(tmpdir, request) -> Generator[TextIOWrapper, None, None]:
    """Create a dummy EDAX binary UP1/2 file.

    The creation of dummy UP1/2 files is explained in more detail in
    kikuchipy/data/edax_binary/create_dummy_edax_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    up_version : int
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : str
    version : int
    is_hex : bool
    """
    # Unpack parameters
    up_ver, (ny, nx), (sy, sx), dtype, ver, is_hex = getattr(
        request, "param", (1, (2, 3), (60, 60), "uint8", 2, False)
    )

    if up_ver == 1:
        fname = tmpdir.join("dummy_edax_file.up1")
        f = open(fname, mode="w")

        # File header: 16 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(f)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 16], "uint32").tofile(f)

        # Patterns
        np.ones(ny * nx * sy * sx, dtype).tofile(f)
    else:  # up_ver == 2
        fname = tmpdir.join("dummy_edax_file.up2")
        f = open(fname, mode="w")

        # File header: 42 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(f)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 42], "uint32").tofile(f)
        # 1 byte with any "extra patterns" (?)
        np.array([1], "uint8").tofile(f)
        # 8 bytes with the map width and height (same as square)
        np.array([nx, ny], "uint32").tofile(f)
        # 1 byte to say whether the grid is hexagonal
        np.array([int(is_hex)], "uint8").tofile(f)
        # 16 bytes with the horizontal and vertical step sizes
        np.array([np.pi, np.pi / 2], "float64").tofile(f)

        # Patterns
        np.ones((ny * nx + ny // 2) * sy * sx, dtype).tofile(f)

    f.close()

    yield f


@pytest.fixture
def edax_h5ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "edax_h5ebsd"


# -------------------- Oxford Instruments formats -------------------- #


@pytest.fixture
def oxford_binary_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "oxford_binary"


@pytest.fixture
def oxford_binary_file(tmpdir, request) -> Generator[TextIOWrapper, None, None]:
    """Create a dummy Oxford Instruments' binary .ebsp file.

    The creation of a dummy .ebsp file is explained in more detail in
    kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : numpy.dtype
    version : int
    compressed : bool
    all_present : bool
    """
    # Unpack parameters
    (nr, nc), (sr, sc), dtype, ver, compressed, all_present = getattr(
        request, "param", ((2, 3), (60, 60), np.uint8, 2, False, True)
    )

    fname = tmpdir.join("dummy_oxford_file.ebsp")
    f = open(fname, mode="w")

    if ver > 0:
        np.array(-ver, dtype=np.int64).tofile(f)

    pattern_header_size = 16
    if ver == 0:
        pattern_footer_size = 0
    elif ver == 1:
        pattern_footer_size = 16
    else:
        pattern_footer_size = 18

    n_patterns = nr * nc
    n_pixels = sr * sc

    if np.issubdtype(dtype, np.uint8):
        n_bytes = n_pixels
    else:
        n_bytes = 2 * n_pixels

    pattern_starts = np.arange(n_patterns, dtype=np.int64)
    pattern_starts *= pattern_header_size + n_bytes + pattern_footer_size
    pattern_starts += n_patterns * 8
    if ver in [1, 2, 3]:
        pattern_starts += 8
    elif ver > 3:
        np.array(0, dtype=np.uint8).tofile(f)
        pattern_starts += 9

    pattern_starts = np.roll(pattern_starts, shift=1)
    if not all_present:
        pattern_starts[0] = 0
    pattern_starts.tofile(f)
    new_order = np.roll(np.arange(n_patterns), shift=-1)

    pattern_header = np.array([compressed, sr, sc, n_bytes], dtype=np.int32)
    data = np.arange(n_patterns * n_pixels, dtype=dtype).reshape((nr, nc, sr, sc))

    if not all_present:
        new_order = new_order[1:]

    for i in new_order:
        r, c = np.unravel_index(i, (nr, nc))
        if ver > 4:
            extra_pattern_header = np.array([c, r], dtype=np.int32)
            extra_pattern_header.tofile(f)
        pattern_header.tofile(f)
        data[r, c].tofile(f)
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_x
        if ver > 0:
            np.array(c, dtype=np.float64).tofile(f)  # beam_x
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_y
        if ver > 0:
            np.array(r, dtype=np.float64).tofile(f)  # beam_y

    f.close()

    yield f


@pytest.fixture
def oxford_h5ebsd_file(tmpdir, request) -> Generator[Path, None, None]:
    """Yield the file path to a temporary H5OINA file.

    Parameters expected in `request`
    -------------------------------
    version : "6.0" or "7.0"
    """
    version: Literal["6.0", "7.0"] = getattr(request, "param", "7.0")
    fpath = tmpdir / "patterns.h5oina"
    create_dummy_oxford_h5ebsd_file(fpath, version=version)
    yield fpath


# -------------------------- ebsdsim formats ------------------------- #


@pytest.fixture(scope="session")
def ebsdsim_master_pattern_file(tmp_path_factory) -> Generator[Path, None, None]:
    """Minimal ebsdsim Ni master pattern .npz file, created once per
    session.
    """
    from kikuchipy.data._dummy_files.ebsdsim_master_pattern_npz import (
        create_small_ebsdsim_npz_file,
    )

    fpath = tmp_path_factory.mktemp("ebsdsim") / "ni_master_pattern.npz"
    create_small_ebsdsim_npz_file(fpath)
    yield fpath


# ------------------------- EMSphInx formats ------------------------- #

# Seconds to wait for the EMSphInx program lock, and the age at which
# one is assumed to belong to a killed run and taken over
_EMSPHINX_LOCK_TIMEOUT = 600.0
_EMSPHINX_LOCK_STALE = 900.0


@pytest.fixture
def emsphinx_dir() -> Generator[Path, None, None]:
    """Yield the EMSphInx checkout, skipping if it is not set up.

    The locally gated tests need ``KIKUCHIPY_EMSPHINX_DIR`` to point
    at a checkout with the built programs and the shipped Ni file.
    """
    value = os.environ.get("KIKUCHIPY_EMSPHINX_DIR")
    if not value:
        pytest.skip(
            "KIKUCHIPY_EMSPHINX_DIR is not set; set it to an EMSphInx "
            "checkout with build/Release/{mp2sht,sht2png,IndexEBSD,"
            "PatternRepack,EBSPDims} and data/'Ni {20kV 75.7deg}.sht' "
            "to run this test"
        )
    yield Path(value)


@contextmanager
def _emsphinx_program_lock() -> Generator[None, None, None]:
    """Hold a lock shared by every process running an EMSphInx
    program.

    The programs read FFTW wisdom from one machine wide file in a
    global constructor and write it back in a global destructor
    (``include/util/fft.hpp`` lines 320-372, ``C:\\ProgramData\\
    fftw.wisdom`` here, 384 kB).  Two of them at once therefore race
    on that file, and a program which imports a half written one
    fast fails: measured under ``pytest -n 4``, ``IndexEBSD.exe``
    exits with 3221226505 (``STATUS_STACK_BUFFER_OVERRUN``) and an
    empty standard output and error, in roughly one run in two.
    Serialising the programs removes the race; it costs nothing,
    since they take a few seconds in total.

    A stdlib exclusive create is used rather than a lock library, so
    that the test suite gains no dependency, and a lock older than
    ``_EMSPHINX_LOCK_STALE`` seconds is taken over, so that a killed
    run cannot block the next one.
    """
    path = Path(tempfile.gettempdir()) / "kikuchipy-emsphinx-program.lock"
    deadline = time.monotonic() + _EMSPHINX_LOCK_TIMEOUT
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > _EMSPHINX_LOCK_STALE
            except OSError:  # it went away between the two calls
                continue
            if stale:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Waited {_EMSPHINX_LOCK_TIMEOUT} s for the EMSphInx "
                    f"program lock {str(path)!r}. Delete it if no test is "
                    "running an EMSphInx program"
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


@pytest.fixture
def emsphinx_program(emsphinx_dir) -> Generator[Callable, None, None]:
    """Yield a callable returning the path of a built EMSphInx
    program, skipping if the checkout or the program is missing.

    Every invocation of a returned program must pass ``cwd=tmp_path``
    to :func:`subprocess.run`: ``IndexEBSD -t`` writes to a hard
    coded relative path and namelist paths resolve against the
    process working directory.

    The fixture holds :func:`_emsphinx_program_lock` for the whole
    test, so that no two of these tests run a program at the same
    time under ``pytest -n``.
    """

    def program(name: str) -> Path:
        directory = emsphinx_dir / "build" / "Release"
        for candidate in (directory / f"{name}.exe", directory / name):
            if candidate.is_file():
                return candidate
        pytest.skip(f"{name} not built in {directory}")

    with _emsphinx_program_lock():
        yield program


@pytest.fixture
def read_ang() -> Generator[Callable, None, None]:
    """Yield a callable reading an EMSphInx written *.ang file into a
    plain array of its eight columns.

    ``orix.io.load`` reads the file but warns about the column
    layout and names the quality columns ``unknown1``/``unknown2``,
    while the assertions index the array.
    """

    def read(filename) -> np.ndarray:
        return np.loadtxt(filename, comments="#")

    yield read


@pytest.fixture(scope="session")
def emsphinx_synthetic_sht_files(tmp_path_factory) -> Generator[Callable, None, None]:
    """Return a callable giving the 25 synthetic EMSphInx *.sht files,
    one per distinct ``(z_rot, compression flags)`` pair, keyed on
    space group.

    The files are written by our own writer
    (``kikuchipy.data._dummy_files.emsphinx_sht``) into a session
    scoped temporary directory on the *first call* and cached, not by
    the fixture itself, for two reasons: only a handful of tests need
    all 25, and a failure inside the writer then shows up as a test
    failure rather than as a fixture error.
    """
    directory = tmp_path_factory.mktemp("emsphinx_sht")
    cache: dict[int, Path] = {}

    def files() -> dict[int, Path]:
        if not cache:
            from kikuchipy.data._dummy_files.emsphinx_sht import (
                create_synthetic_sht_files,
            )

            cache.update(create_synthetic_sht_files(directory))
        return cache

    yield files


# -------------------------- EMsoft formats -------------------------- #


@pytest.fixture
def emsoft_ebsd_master_pattern_file() -> Generator[Path, None, None]:
    yield DATA_PATH / "emsoft_ebsd_master_pattern/master_patterns.h5"


@pytest.fixture
def emsoft_ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "emsoft_ebsd"


@pytest.fixture
def emsoft_ebsd_file(emsoft_ebsd_path) -> Generator[Path, None, None]:
    yield emsoft_ebsd_path / "EBSD_TEST_Ni.h5"


@pytest.fixture
def emsoft_ebsd_master_pattern_metadata() -> Generator[dict, None, None]:
    fname = "master_patterns.h5"
    yield {
        "General": {"original_filename": fname, "title": fname.split(".")[0]},
        "Signal": {"signal_type": "EBSDMasterPattern"},
    }


@pytest.fixture
def emsoft_ebsd_master_pattern_axes_manager(request) -> Generator[dict, None, None]:
    axes = getattr(request, "param", ["hemisphere", "energy", "height", "width"])
    am = {
        "hemisphere": {
            "name": "hemisphere",
            "scale": 1,
            "offset": 0,
            "size": 2,
            "units": "",
            "navigate": True,
        },
        "energy": {
            "name": "energy",
            "scale": 1,
            "offset": 10.0,
            "size": 11,
            "units": "keV",
            "navigate": True,
        },
        "height": {
            "name": "height",
            "scale": 1,
            "offset": -7.0,
            "size": 13,
            "units": "px",
            "navigate": False,
        },
        "width": {
            "name": "width",
            "scale": 1,
            "offset": -7.0,
            "size": 13,
            "units": "px",
            "navigate": False,
        },
    }
    d = {}
    for i, a in enumerate(axes):
        d["axis-" + str(i)] = am[a]
    yield d


@pytest.fixture
def emsoft_ecp_master_pattern_file(tmpdir) -> Generator[Path, None, None]:
    """Dummy EMsoft ECP master pattern file."""

    fpath = tmpdir / "ecp_master_pattern.h5"
    f = h5py.File(fpath, mode="w")

    npx = 6
    signal_shape = (npx * 2 + 1,) * 2
    energies = np.linspace(10, 20, 11, dtype=np.float32)
    data_shape = (len(energies),) + signal_shape

    mp_lam_upper = np.ones((1,) + data_shape, dtype=np.float32) * energies.reshape(
        (1, 11, 1, 1)
    )
    mp_lam_lower = mp_lam_upper
    circle = kp.filters.Window(shape=signal_shape).astype(np.float32)
    mp_sph_upper = mp_lam_upper.squeeze() * circle
    mp_sph_lower = mp_sph_upper

    data = {
        "CrystalData": {
            "AtomData": np.array(
                [[0.1587, 0], [0.6587, 0], [0, 0.25], [1, 1], [0.005, 0.005]],
                dtype=np.float32,
            ),
            "Atomtypes": np.array([13, 29], dtype=np.int32),
            "CrystalSystem": 2,
            "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
            "Natomtypes": 2,
            "Source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
            "SpaceGroupNumber": 140,
            "SpaceGroupSetting": 1,
        },
        "EMData": {
            "ECPmaster": {
                "EkeV": np.linspace(10, 20, 11, dtype=np.float32),
                "mLPNH": mp_lam_upper,  # mLPSH written below
                "masterSPNH": mp_sph_upper,
                "masterSPSH": mp_sph_lower,
                "numset": 1,
            }
        },
        "NMLparameters": {
            "ECPMasterNameList": {"dmin": 0.05, "npx": npx},
            "MCCLNameList": {
                "Ebinsize": energies[1] - energies[0],
                "Ehistmin": np.min(energies),
                "EkeV": np.max(energies),
                "MCmode": "CSDA",
                "dataname": "crystal_data/al2cu/al2cu_mc_mp_20kv.h5",
                "depthmax": 100.0,
                "depthstep": 1.0,
                "mode": "bse1",
                "numsx": npx,
                "sigend": 10.0,
                "sigstart": 0.0,
                "sigstep": 2.0,
                "totnum_el": 2000000000,
            },
            "BetheList": {"c1": 4.0, "c2": 8.0, "c3": 50.0, "sgdbdiff": 1.0},
        },
        "EMheader": {
            "ECPmaster": {"ProgramName": np.array([b"EMECPmaster.f90"], dtype="S15")},
        },
    }

    _dict2hdf5group(dictionary=data, group=f["/"])

    # One chunked data set
    f["EMData/ECPmaster"].create_dataset("mLPSH", data=mp_lam_lower, chunks=True)

    # One byte string with latin-1 stuff
    creation_time = b"12:30:13.559 PM\xf0\x14\x1e\xc8\xbcU"
    f["CrystalData"].create_dataset("CreationTime", data=creation_time)

    f.close()

    yield fpath


@pytest.fixture
def emsoft_tkd_master_pattern_file(tmpdir) -> Generator[Path, None, None]:
    fpath = tmpdir / "tkd_master_pattern.h5"
    f = h5py.File(fpath, mode="w")

    npx = 6
    signal_shape = (npx * 2 + 1,) * 2
    energies = np.linspace(10, 20, 11, dtype=np.float32)
    data_shape = (len(energies),) + signal_shape

    mp_lam_upper = np.ones((1,) + data_shape, dtype=np.float32) * energies.reshape(
        (1, 11, 1, 1)
    )
    mp_lam_lower = mp_lam_upper
    circle = kp.filters.Window(shape=signal_shape).astype(np.float32)
    mp_sph_upper = mp_lam_upper.squeeze() * circle
    mp_sph_lower = mp_sph_upper

    data = {
        "CrystalData": {
            "AtomData": np.array(
                [[0.1587, 0], [0.6587, 0], [0, 0.25], [1, 1], [0.005, 0.005]],
                dtype=np.float32,
            ),
            "Atomtypes": np.array([13, 29], dtype=np.int32),
            "CrystalSystem": 2,
            "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
            "Natomtypes": 2,
            "Source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
            "SpaceGroupNumber": 140,
            "SpaceGroupSetting": 1,
        },
        "EMData": {
            "TKDmaster": {
                "BetheParameters": np.array([4, 8, 50, 1], dtype=np.float32),
                "EkeVs": np.linspace(10, 20, 11, dtype=np.float32),
                "mLPNH": mp_lam_upper,
                "masterSPNH": mp_sph_upper,
                "masterSPSH": mp_sph_lower,
                "numEbins": len(energies),
                "numset": 1,
            }
        },
        "NMLparameters": {
            "TKDMasterNameList": {"dmin": 0.05, "npx": npx},
            "MCCLfoilNameList": {
                "Ebinsize": energies[1] - energies[0],
                "Ehistmin": np.min(energies),
                "EkeV": np.max(energies),
                "MCmode": "CSDA",
                "dataname": "crystal_data/al2cu/al2cu_mc_mp_20kv.h5",
                "depthmax": 100.0,
                "depthstep": 1.0,
                "numsx": npx,
                "sig": -20.0,
                "totnum_el": 2000000000,
            },
            "BetheList": {"c1": 4.0, "c2": 8.0, "c3": 50.0, "sgdbdiff": 1.0},
        },
        "EMheader": {
            "TKDmaster": {"ProgramName": np.array([b"EMTKDmaster.f90"], dtype="S15")},
        },
    }

    _dict2hdf5group(dictionary=data, group=f["/"])

    # One chunked data set
    f["EMData/TKDmaster"].create_dataset("mLPSH", data=mp_lam_lower, chunks=True)

    # One byte string with latin-1 stuff
    creation_time = b"12:30:13.559 PM\xf0\x14\x1e\xc8\xbcU"
    f["CrystalData"].create_dataset("CreationTime", data=creation_time)

    f.close()

    yield fpath


# -------------------------- NORDIF formats -------------------------- #


@pytest.fixture
def nordif_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "nordif"


@pytest.fixture
def nordif_renamed_calibration_pattern(
    nordif_path: Path,
) -> Generator[Path, None, None]:
    fname = "Background calibration pattern.bmp"
    f1 = nordif_path / fname
    f2 = f1.rename(f1.with_suffix(".bak"))
    yield f2
    f2.rename(f1)


# -------------------------- Bruker formats -------------------------- #


@pytest.fixture
def bruker_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "bruker_h5ebsd"


@pytest.fixture
def bruker_h5ebsd_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with no region of interest."""
    fpath = tmpdir / "patterns.h5"
    create_dummy_bruker_h5ebsd_file(fpath)
    yield fpath


@pytest.fixture
def bruker_h5ebsd_roi_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with rectangular region of interest (and SEM
    group under EBSD group).
    """
    path = tmpdir / "patterns_roi.h5"
    create_dummy_bruker_h5ebsd_roi_file(path)
    yield path


@pytest.fixture
def bruker_h5ebsd_nonrectangular_roi_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with non-rectangular region of interest (and
    SEM group under EBSD group).
    """
    path = tmpdir / "patterns_roi_nonrectangular.h5"
    create_dummy_bruker_h5ebsd_nonrectangular_roi_file(path)
    yield path


# ------------------------------- NLPAR ------------------------------ #
# Synthetic maps of patterns and test helpers shared by the NLPAR kernel
# tests and the NLPAR EBSD method tests. Test modules cannot import from
# each other or from this file (pytest runs with --import-mode=importlib
# and tests/ has no __init__.py), so every generator or helper is a
# plain function exposed by a fixture of the same name without the
# leading underscore, which returns the callable. Every generator is
# deterministic: it draws from its own seeded numpy.random.Generator
# only.

# Measured, then pinned: the largest difference, in float32 ulps,
# allowed between the compiled NLPAR weights kernel (the only NLPAR
# kernel calling exp) and its pure Python function, and between that
# kernel and the closed-form weight exp(-max(d - dthresh, 0) / lam^2).
# The drafting seed was 1 ulp (Numba's exp and NumPy's were expected to
# differ by at most that).
# Pinned 2026-10-05 at the measured ulp count, 0: compiled vs pure
# Python and kernel vs closed form on every test input, and on 2e5
# (pure Python) and 2e6 (closed form) random distances in [-5, 60]
# per (lam, dthresh) for lam in {0.5, 0.7, 1.0, 2.5}
# and dthresh in {0, 0.5} (the kernel rounds a float64 exp to
# float32). Measured on a 20-core Intel Raptor Lake laptop, Windows 11,
# numba 0.65.1, numpy 2.4.6.
EXP_KERNEL_ULP: int = 0


def _circle_mask(sig_shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask of the signal shape which is True outside
    the circle inscribed in it (kikuchipy polarity: True means
    excluded).
    """
    h, w = sig_shape
    rows, cols = np.ogrid[:h, :w]
    r2 = (rows - (h - 1) / 2) ** 2 + (cols - (w - 1) / 2) ** 2
    return r2 > (min(h, w) / 2) ** 2


def _nlpar_ramp(sig_shape: tuple[int, int]) -> np.ndarray:
    """Return the noise-free base pattern of the NLPAR generators: a
    smooth float32 ramp from 40 to 200 of the signal shape.
    """
    h, w = sig_shape
    return np.linspace(40.0, 200.0, h * w, dtype=np.float32).reshape(h, w)


def _nlpar_cast(x: np.ndarray, dtype: type | np.dtype | str) -> np.ndarray:
    """Return float32 data in a data type, rounded to nearest and
    clipped to the range of an integer data type.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(x), info.min, info.max).astype(dtype)
    return x.astype(dtype)


def _identical_plus_gaussian(
    nav_shape: tuple[int, ...],
    sig_shape: tuple[int, int],
    sigma: float = 8.0,
    dtype: type | np.dtype | str = np.float32,
    seed: int = 0,
    sigma_right: float | None = None,
) -> np.ndarray:
    """Return a map of one smooth pattern plus independent Gaussian
    noise.

    Parameters
    ----------
    nav_shape
        Navigation shape, e.g. (12, 12) or (7,).
    sig_shape
        Signal shape (h, w).
    sigma
        Standard deviation of the noise.
    dtype
        Data type of the map. Integer types are rounded to nearest and
        clipped to their range (uint8: ``np.clip(np.rint(x), 0, 255)``).
    seed
        Seed of :func:`numpy.random.default_rng`.
    sigma_right
        If given, the standard deviation of the noise on the right
        half of the columns, ``[w // 2, w)``, of every pattern.

    Returns
    -------
    data
        Array of shape ``nav_shape + sig_shape``.

    Notes
    -----
    The base is ``np.linspace(40.0, 200.0, h * w, dtype=np.float32)``
    reshaped to (h, w), the same for every pattern. The noise is
    ``default_rng(seed).normal(0, scale, nav_shape + sig_shape)`` cast
    to float32, with ``scale`` the scalar ``sigma`` or, with
    ``sigma_right``, the per-column scale. The left half of the columns
    is therefore the same with and without ``sigma_right``.
    """
    nav_shape = tuple(nav_shape)
    sig_shape = tuple(sig_shape)
    if sigma_right is None:
        scale = sigma
    else:
        w = sig_shape[1]
        scale = np.full(w, sigma, dtype=np.float64)
        scale[w // 2 :] = sigma_right
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, nav_shape + sig_shape).astype(np.float32)
    data = _nlpar_ramp(sig_shape) + noise
    return _nlpar_cast(data, dtype)


def _two_grain(
    nav_shape: tuple[int, int] = (10, 16),
    sig_shape: tuple[int, int] = (32, 32),
    delta: float = 30.0,
    sigma: float = 8.0,
    dtype: type | np.dtype | str = np.float32,
    seed: int = 1,
) -> np.ndarray:
    """Return a map of two grains separated by a vertical boundary,
    plus independent Gaussian noise.

    Parameters
    ----------
    nav_shape
        Navigation shape (rows, columns).
    sig_shape
        Signal shape (h, w).
    delta
        Intensity added to every pixel of the grain B pattern.
    sigma
        Standard deviation of the noise.
    dtype
        Data type of the map, as in :func:`_identical_plus_gaussian`.
    seed
        Seed of :func:`numpy.random.default_rng`.

    Returns
    -------
    data
        Array of shape ``nav_shape + sig_shape``.

    Notes
    -----
    Grain A, the left ``n_cols // 2`` navigation columns, has the ramp
    base of :func:`_identical_plus_gaussian`; grain B, the remaining
    columns ``[n_cols // 2, n_cols)``, has that base plus
    ``np.float32(delta)`` (in float32). The noise is
    ``default_rng(seed).normal(0, sigma, nav_shape + sig_shape)`` cast
    to float32 and added to the grain's base.
    """
    nav_shape = tuple(nav_shape)
    sig_shape = tuple(sig_shape)
    base = _nlpar_ramp(sig_shape)
    bases = np.broadcast_to(base, nav_shape + sig_shape).copy()
    bases[:, nav_shape[1] // 2 :] = base + np.float32(delta)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, nav_shape + sig_shape).astype(np.float32)
    data = bases + noise
    return _nlpar_cast(data, dtype)


def _random_uniform_saturated(
    nav_shape: tuple[int, ...],
    sig_shape: tuple[int, int],
    frac: float = 0.05,
    seed: int = 2,
    one_block_only: bool = False,
) -> np.ndarray:
    """Return a uint8 map of uniform random patterns with a fraction of
    saturated pixels.

    Parameters
    ----------
    nav_shape
        Navigation shape, e.g. (4, 5), (10, 16) or (7,).
    sig_shape
        Signal shape (h, w).
    frac
        Fraction of the pixels of a pattern set to 255. The number per
        pattern is ``round(frac * h * w)``, which must be less than
        ``h * w``. 0 plants none.
    seed
        Seed of :func:`numpy.random.default_rng`.
    one_block_only
        Whether to plant the saturated pixels only in the patterns of
        the first (5, 8) navigation block (the first 5 patterns of a 1D
        scan), leaving every other pattern unsaturated.

    Returns
    -------
    data
        Array of shape ``nav_shape + sig_shape`` and data type uint8.

    Raises
    ------
    ValueError
        If ``frac`` would saturate every pixel of a pattern.

    Notes
    -----
    The patterns are ``rng.integers(20, 240, nav_shape + sig_shape,
    dtype=np.uint8)``, so every unsaturated value is in [20, 239]. The
    saturated positions of every pattern are the first
    ``round(frac * h * w)`` indices of a stable argsort of
    ``rng.random(nav_shape + (h * w,))``, drawn after the patterns, so
    they are distinct within a pattern and the unsaturated values do
    not depend on ``frac`` or ``one_block_only``.
    """
    nav_shape = tuple(nav_shape)
    sig_shape = tuple(sig_shape)
    n_pixels = int(np.prod(sig_shape))
    n_saturated = round(frac * n_pixels)
    if not 0 <= n_saturated < n_pixels:
        raise ValueError(
            f"frac={frac} gives {n_saturated} saturated pixels of {n_pixels} per "
            "pattern; at least one pixel must stay unsaturated"
        )
    rng = np.random.default_rng(seed)
    data = rng.integers(20, 240, nav_shape + sig_shape, dtype=np.uint8)
    if n_saturated == 0:
        return data
    keys = rng.random(nav_shape + (n_pixels,))
    positions = np.argsort(keys, axis=-1, kind="stable")[..., :n_saturated]
    saturated = data.reshape(nav_shape + (n_pixels,)).copy()
    np.put_along_axis(saturated, positions, np.uint8(255), axis=-1)
    saturated = saturated.reshape(data.shape)
    if one_block_only:
        block = tuple(slice(0, size) for size in (5, 8)[: len(nav_shape)])
        data[block] = saturated[block]
        return data
    return saturated


def _exact_duplicates(
    nav_shape: tuple[int, int],
    sig_shape: tuple[int, int],
    seed: int = 3,
    block: bool = False,
) -> np.ndarray:
    """Return a uint8 map of uniform random patterns in which some
    patterns are exact copies of a neighbour.

    Parameters
    ----------
    nav_shape
        Navigation shape (rows, columns), at least (6, 6).
    sig_shape
        Signal shape (h, w).
    seed
        Seed of :func:`numpy.random.default_rng`.
    block
        Whether to also copy pattern (3, 4) into its eight 3 x 3
        neighbours, so that (3, 4) has no neighbour with a non-zero
        distance while each of those neighbours still has neighbours
        outside the block.

    Returns
    -------
    data
        Array of shape ``nav_shape + sig_shape`` and data type uint8.

    Raises
    ------
    ValueError
        If the navigation shape is not 2D or smaller than (6, 6).

    Notes
    -----
    The patterns are ``rng.integers(20, 240, nav_shape + sig_shape,
    dtype=np.uint8)``, without saturated pixels. Then pattern (1, 1) is
    copied to (1, 2), (3, 4) to (4, 4) and (5, 2) to (5, 3), in that
    order, before the optional block copy.
    """
    nav_shape = tuple(nav_shape)
    sig_shape = tuple(sig_shape)
    if len(nav_shape) != 2 or nav_shape[0] < 6 or nav_shape[1] < 6:
        raise ValueError(f"nav_shape {nav_shape} must be 2D and at least (6, 6)")
    rng = np.random.default_rng(seed)
    data = rng.integers(20, 240, nav_shape + sig_shape, dtype=np.uint8)
    for source, target in [((1, 1), (1, 2)), ((3, 4), (4, 4)), ((5, 2), (5, 3))]:
        data[target] = data[source]
    if block:
        for row in (2, 3, 4):
            for col in (3, 4, 5):
                data[row, col] = data[3, 4]
    return data


@pytest.fixture
def identical_plus_gaussian() -> Callable:
    """Return the generator :func:`_identical_plus_gaussian` of maps of
    one smooth pattern plus Gaussian noise.
    """
    return _identical_plus_gaussian


@pytest.fixture
def two_grain() -> Callable:
    """Return the generator :func:`_two_grain` of two-grain maps."""
    return _two_grain


@pytest.fixture
def random_uniform_saturated() -> Callable:
    """Return the generator :func:`_random_uniform_saturated` of uint8
    random maps with saturated pixels.
    """
    return _random_uniform_saturated


@pytest.fixture
def exact_duplicates() -> Callable:
    """Return the generator :func:`_exact_duplicates` of uint8 random
    maps with exactly duplicated patterns.
    """
    return _exact_duplicates


@pytest.fixture
def exp_kernel_ulp() -> int:
    """Return the measured, then pinned float32 ulp tolerance of the
    NLPAR weights kernel, ``EXP_KERNEL_ULP``.
    """
    return EXP_KERNEL_ULP


@pytest.fixture
def circle_mask() -> Callable:
    """Return the helper :func:`_circle_mask` of the inscribed-circle
    signal mask (True outside the circle).
    """
    return _circle_mask


@pytest.fixture
def counting_spy(monkeypatch) -> Callable:
    """Return a function ``spy(module, name)`` that replaces
    ``module.name`` by a wrapper recording every call and delegating to
    the original, and returns the list of recorded calls.

    Each call records its ``block_info`` keyword (None if absent).
    ``functools.wraps`` keeps the original signature visible to
    :func:`inspect.signature`, so Dask still passes ``block_info`` to a
    wrapped chunk function. The NLPAR drivers look the chunk wrappers
    up as module globals at call time, so the spy sees every block. The
    replacement is undone at teardown.
    """

    def spy(module, name: str) -> list:
        original = getattr(module, name)
        calls = []

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            calls.append(kwargs.get("block_info"))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)
        return calls

    return spy


# ------------------------------- HROSM ------------------------------ #
# Synthetic crystal maps, top-match lists and EMsoft-layout files shared
# by the HROSM tests, and the two opt-in EMsoft gates. As for NLPAR,
# every generator is a plain function exposed by a fixture of the same
# name without the leading underscore, which returns the callable.
# Every generator is deterministic: it draws from its own seeded
# numpy.random.Generator only.

# Seconds to wait for the EMsoft program lock, the age of a lock file
# assumed to belong to a killed run (taken over), and the interval at
# which the holder refreshes the lock file's modification time. A
# reference run holds the lock for up to half an hour, so the holder
# keeps the file young while it runs.
_EMSOFT_LOCK_TIMEOUT = 3600.0
_EMSOFT_LOCK_STALE = 300.0
_EMSOFT_LOCK_HEARTBEAT = 30.0
_EMSOFT_LOCK_NAME = "kikuchipy-emsoft-program.lock"

# The EMsoftOO programs and libraries the bin gate needs
_EMSOFT_PROGRAMS = (
    "EMDI.exe",
    "EMFitOrientation.exe",
    "EMHROSM.exe",
    "EMgetOSM.exe",
    "EMsampleRFZ.exe",
)
_EMSOFT_LIBRARIES = ("EMsoftOOLib.dll", "EMOpenCLLib.dll")

# md5 sums of the EMsoft data files already checked in this session
_EMSOFT_MD5_CACHE: dict[Path, str] = {}


@contextmanager
def _emsoft_program_lock(
    path: Path | str | None = None,
    *,
    timeout: float | None = None,
    stale: float | None = None,
    heartbeat: float | None = None,
) -> Generator[Path, None, None]:
    """Hold a lock shared by every process running an EMsoft program.

    The programs share EMsoft's temporary directory (the dictionary
    scratch file) and one GPU, so two of them must never run at once.
    The lock is an exclusive create of ``kikuchipy-emsoft-program.lock``
    in the temporary directory (``path`` is only for the lock's own
    test). While it is held, a daemon thread refreshes the file's
    modification time every ``heartbeat`` seconds, so a lock file older
    than ``stale`` seconds belongs to a killed holder and is taken over;
    no process ID test is made, since probing a process on Windows
    with ``os.kill(pid, 0)`` terminates it. The reference generation
    script uses the same file and protocol.

    ``timeout``, ``stale`` and ``heartbeat`` default to
    ``_EMSOFT_LOCK_TIMEOUT``, ``_EMSOFT_LOCK_STALE`` and
    ``_EMSOFT_LOCK_HEARTBEAT``, read at call time.
    """
    if path is None:
        path = Path(tempfile.gettempdir()) / _EMSOFT_LOCK_NAME
    path = Path(path)
    timeout = _EMSOFT_LOCK_TIMEOUT if timeout is None else float(timeout)
    stale = _EMSOFT_LOCK_STALE if stale is None else float(stale)
    heartbeat = _EMSOFT_LOCK_HEARTBEAT if heartbeat is None else float(heartbeat)

    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:  # it went away between the two calls
                continue
            if age > stale:
                try:
                    path.unlink(missing_ok=True)
                except OSError:  # still open by its holder on Windows
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Waited {timeout} s for the EMsoft program lock "
                    f"{str(path)!r}. Delete it if no process is running an "
                    "EMsoft program"
                )
            time.sleep(0.05)

    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(heartbeat):
            try:
                os.utime(path)
            except OSError:
                pass

    thread = threading.Thread(
        target=refresh, name="kikuchipy-emsoft-lock-heartbeat", daemon=True
    )
    thread.start()
    try:
        yield path
    finally:
        stop.set()
        thread.join()
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _emsoft_config() -> dict:
    """Return EMsoft's configuration ``~/.config/EMsoft/
    EMsoftConfig.json``, or an empty dict if it cannot be read.
    """
    fpath = Path.home() / ".config" / "EMsoft" / "EMsoftConfig.json"
    try:
        return json.loads(fpath.read_text())
    except (OSError, ValueError):
        return {}


def _md5_of_file(fpath: Path) -> str:
    """Return the md5 sum of a file, read in chunks."""
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk)
    return md5.hexdigest()


@pytest.fixture
def emsoft_program_lock() -> Callable:
    """Return the context manager :func:`_emsoft_program_lock`, for
    tests which run EMsoft programs outside :func:`emsoft_program` (the
    reference regeneration) and for the lock's own test.
    """
    return _emsoft_program_lock


@pytest.fixture
def emsoft_bin_dir() -> Generator[Path, None, None]:
    """Yield the directory of the EMsoftOO programs, skipping if it is
    not set up.

    The bin gated tests need ``KIKUCHIPY_EMSOFT_BIN`` to name a
    directory with the five programs and two libraries, EMsoft's
    configuration file with an existing ``EMdatapathname``, and an
    OpenCL runtime for the GPU programs.
    """
    value = os.environ.get("KIKUCHIPY_EMSOFT_BIN")
    if not value:
        pytest.skip(
            "KIKUCHIPY_EMSOFT_BIN is not set; set it to an EMsoftOO Bin "
            f"directory with {', '.join(_EMSOFT_PROGRAMS + _EMSOFT_LIBRARIES)} "
            "to run this test"
        )
    directory = Path(value)
    for name in _EMSOFT_PROGRAMS + _EMSOFT_LIBRARIES:
        if not (directory / name).is_file():
            pytest.skip(f"{name} is missing from KIKUCHIPY_EMSOFT_BIN {directory}")
    data_path = _emsoft_config().get("EMdatapathname")
    if not data_path or not Path(data_path).is_dir():
        pytest.skip(
            "EMdatapathname of ~/.config/EMsoft/EMsoftConfig.json is not set "
            "or does not exist"
        )
    import ctypes.util

    if ctypes.util.find_library("OpenCL") is None:
        pytest.skip("No OpenCL runtime found; EMDI and EMHROSM need a GPU")
    yield directory


@pytest.fixture
def emsoft_program(emsoft_bin_dir) -> Generator[Callable, None, None]:
    """Yield a callable running one EMsoft program under the program
    lock and returning its :class:`subprocess.CompletedProcess`.

    ``run(name, namelist, run_dir, timeout=None)`` writes the namelist
    text to ``<run_dir>/<name>.nml`` and runs ``<bin>/<name>.exe
    <name>.nml`` with the working directory ``run_dir``, capturing
    standard output and error as text. The programs prepend
    ``EMdatapathname`` to every namelist path, so every path in the
    namelist must start with the relative run directory.

    ``run.new_run_dir()`` creates and returns a new run directory
    ``<EMdatapathname>/kikuchipy_hrosm/<YYYYmmdd-HHMMSS>/`` as the pair
    (absolute path, path relative to ``EMdatapathname`` with forward
    slashes); ``run.data_root`` is ``EMdatapathname``.
    """
    data_root = Path(_emsoft_config()["EMdatapathname"])

    def run(
        name: str, namelist: str, run_dir: Path, timeout: float | None = None
    ) -> subprocess.CompletedProcess:
        run_dir = Path(run_dir)
        nml = run_dir / f"{name}.nml"
        nml.write_text(namelist)
        with _emsoft_program_lock():
            return subprocess.run(
                [str(emsoft_bin_dir / f"{name}.exe"), nml.name],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

    def new_run_dir() -> tuple[Path, str]:
        while True:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            relative = f"kikuchipy_hrosm/{stamp}"
            directory = data_root / "kikuchipy_hrosm" / stamp
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                time.sleep(1.0)
                continue
            return directory, relative

    run.new_run_dir = new_run_dir
    run.data_root = data_root
    yield run


@pytest.fixture
def emsoft_data_file() -> Callable:
    """Return a callable ``get(relpath, md5)`` giving the path of an
    EMsoft data file below ``KIKUCHIPY_EMSOFT_DATA``, skipping if the
    variable is not set or the file is missing.

    The file's md5 sum is asserted to equal ``md5``; it is computed
    once per file and session.
    """

    def get(relpath: str, md5: str) -> Path:
        root = os.environ.get("KIKUCHIPY_EMSOFT_DATA")
        if not root:
            pytest.skip(
                "KIKUCHIPY_EMSOFT_DATA is not set; set it to the EMsoft data "
                f"root holding {relpath} to run this test"
            )
        fpath = (Path(root) / relpath).resolve()
        if not fpath.is_file():
            pytest.skip(f"{relpath} is missing from KIKUCHIPY_EMSOFT_DATA {root}")
        if fpath not in _EMSOFT_MD5_CACHE:
            _EMSOFT_MD5_CACHE[fpath] = _md5_of_file(fpath)
        assert _EMSOFT_MD5_CACHE[fpath] == md5, f"{fpath} has another md5"
        return fpath

    return get


def _hrosm_axis_angle(axis, angle_deg) -> Rotation:
    """Return rotations about one axis by angles in degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(np.atleast_1d(np.asarray(angle_deg, dtype=np.float64))) / 2
    data = np.zeros(half.shape + (4,))
    data[..., 0] = np.cos(half)
    data[..., 1:] = np.sin(half)[..., None] * axis
    return Rotation(data)


def _hrosm_repeat(rotation: Rotation, n: int) -> Rotation:
    """Return one rotation repeated ``n`` times."""
    return Rotation(np.repeat(rotation.data.reshape(1, 4), n, axis=0))


def _hrosm_crystal_map(
    rotations: Rotation,
    shape: tuple[int, int],
    phase_id: np.ndarray | None = None,
    is_in_data: np.ndarray | None = None,
) -> CrystalMap:
    """Return a crystal map of nickel (m-3m) of step 1 um, with a second
    nickel phase "ni2" (ID 1) if ``phase_id`` has ones.
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
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phases=phases, ids=ids),
        is_in_data=is_in_data,
        scan_unit="um",
    )


def _hrosm_gradient_xmap(
    shape: tuple[int, int] = (6, 7),
    delta_x: float = 0.4,
    delta_y: float = 0.1,
    euler0: tuple[float, float, float] = (10.0, 20.0, 30.0),
    scramble_seed: int | None = None,
    absent: tuple[int, ...] | list[int] = (),
    phase_id: np.ndarray | None = None,
    rotations_per_point: int = 1,
) -> CrystalMap:
    """Return a nickel crystal map with a constant orientation gradient.

    The rotation at ``(y, x)`` is ``Rz(x * delta_x) * Rx(y * delta_y) *
    g0``, ``g0`` from the Euler angles ``euler0``; all angles in
    degrees. Each horizontal 4-neighbour pair is disoriented by exactly
    ``delta_x`` and each vertical one by ``delta_y``, for angles below
    45 degrees.

    ``scramble_seed`` left-multiplies every point by a random operator
    of m-3m's proper subgroup (``default_rng(scramble_seed)``);
    ``absent`` (flat indices) are not in the data; ``phase_id`` (one ID
    per point, 0 or 1, of the map shape or flat) puts the points with
    ID 1 in a second nickel phase "ni2"; ``rotations_per_point > 1``
    appends seeded random rotations after the first one per point.
    """
    shape = tuple(shape)
    n_rows, n_cols = shape
    n = n_rows * n_cols
    rows, cols = np.divmod(np.arange(n), n_cols)
    g0 = Rotation.from_euler(np.deg2rad(euler0))
    rot = (
        _hrosm_axis_angle((0, 0, 1), cols * delta_x)
        * _hrosm_axis_angle((1, 0, 0), rows * delta_y)
        * _hrosm_repeat(g0, n)
    )
    if scramble_seed is not None:
        rng = np.random.default_rng(scramble_seed)
        operators = Oh.proper_subgroup.data
        rot = Rotation(operators[rng.integers(operators.shape[0], size=n)]) * rot
    if rotations_per_point > 1:
        rng_extra = np.random.default_rng(1)
        extra = rng_extra.normal(size=(n, rotations_per_point - 1, 4))
        extra /= np.linalg.norm(extra, axis=-1, keepdims=True)
        rot = Rotation(np.concatenate([rot.data[:, None], extra], axis=1))
    is_in_data = np.ones(n, dtype=bool)
    is_in_data[list(absent)] = False
    return _hrosm_crystal_map(rot, shape, phase_id=phase_id, is_in_data=is_in_data)


def _hrosm_constant_pair_xmap(
    shape: tuple[int, int] = (4, 5),
    phi: float = 0.3,
    euler0: tuple[float, float, float] = (10.0, 20.0, 30.0),
) -> CrystalMap:
    """Return a nickel crystal map with the rotation ``Rz((x + y) *
    phi) * g0`` at ``(y, x)``, so that every 4-neighbour pair is
    disoriented by exactly ``phi`` degrees.
    """
    shape = tuple(shape)
    n_rows, n_cols = shape
    n = n_rows * n_cols
    rows, cols = np.divmod(np.arange(n), n_cols)
    g0 = Rotation.from_euler(np.deg2rad(euler0))
    rot = _hrosm_axis_angle((0, 0, 1), (cols + rows) * phi) * _hrosm_repeat(g0, n)
    return _hrosm_crystal_map(rot, shape)


def _hrosm_disorientation_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the m-3m disorientation angles in degrees between every
    quaternion of ``a`` (n, 4) and every one of ``b`` (m, 4), shape (n,
    m), with the symmetry operators applied from the left.
    """
    a = a[:, None, :]
    b_conj = b * np.array([1.0, -1.0, -1.0, -1.0])
    b_conj = b_conj[None, :, :]
    r = np.empty(np.broadcast_shapes(a.shape, b_conj.shape))
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b_conj, -1, 0)
    r[..., 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    r[..., 1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    r[..., 2] = a0 * b2 + a2 * b0 + a3 * b1 - a1 * b3
    r[..., 3] = a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1
    operators = Oh.proper_subgroup.data
    d = np.abs(r @ operators.T).max(axis=-1)
    return np.rad2deg(2 * np.arccos(np.clip(d, 0, 1)))


def _hrosm_grain_xmap(
    shape: tuple[int, int] = (12, 16),
    n_grains: int = 2,
    gradient: float = 0.2,
    boundary_angle: float = 30.0,
) -> tuple[CrystalMap, np.ndarray]:
    """Return a nickel crystal map of two or three grains and the true
    labels.

    Grain A (label 1) is the columns ``x < W // 2`` with ``g_A = Rz(x *
    gradient) * g0``, ``g0`` from the Euler angles (10, 20, 30); grain
    B (label 2) the other columns with ``R[111](boundary_angle) *
    g_A``; with ``n_grains=3``, the rows ``y >= H // 2`` of the right
    half are grain C (label 3) with ``R[100](45) * g_A``. Angles in
    degrees.

    Returns the map and the labels of shape (H, W) of int32. Every pair
    of distinct grains is asserted to be disoriented by more than 20
    degrees unless ``boundary_angle`` is below 20.
    """
    if n_grains not in (2, 3):
        raise ValueError(f"n_grains {n_grains} must be 2 or 3")
    shape = tuple(shape)
    n_rows, n_cols = shape
    n = n_rows * n_cols
    rows, cols = np.divmod(np.arange(n), n_cols)
    g0 = Rotation.from_euler(np.deg2rad((10.0, 20.0, 30.0)))
    g_a = _hrosm_axis_angle((0, 0, 1), cols * gradient) * _hrosm_repeat(g0, n)
    g_b = _hrosm_repeat(_hrosm_axis_angle((1, 1, 1), boundary_angle), n) * g_a
    g_c = _hrosm_repeat(_hrosm_axis_angle((1, 0, 0), 45.0), n) * g_a

    truth = np.ones(n, dtype=np.int32)
    truth[cols >= n_cols // 2] = 2
    if n_grains == 3:
        truth[(cols >= n_cols // 2) & (rows >= n_rows // 2)] = 3
    data = g_a.data.copy()
    data[truth == 2] = g_b.data[truth == 2]
    data[truth == 3] = g_c.data[truth == 3]

    if boundary_angle >= 20:
        for label_a in range(1, n_grains + 1):
            for label_b in range(label_a + 1, n_grains + 1):
                angles = _hrosm_disorientation_deg(
                    data[truth == label_a], data[truth == label_b]
                )
                assert angles.min() > 20, (label_a, label_b, angles.min())

    xmap = _hrosm_crystal_map(Rotation(data), shape)
    return xmap, truth.reshape(shape)


def _hrosm_top_lists(
    shape: tuple[int, int], n: int = 10, pool: int = 15, seed: int = 40
) -> np.ndarray:
    """Return 1-based top-match lists without duplicates within a row,
    shape (H * W, n) of int32: per point, in raster order,
    ``rng.permutation(pool)[:n] + 1`` of one ``default_rng(seed)``.
    """
    n_points = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    lists = [rng.permutation(pool)[:n] + 1 for _ in range(n_points)]
    return np.asarray(lists, dtype=np.int32).reshape(n_points, n)


@functools.lru_cache(maxsize=1)
def _hrosm_master_pattern() -> kp.signals.EBSDMasterPattern:
    """Return the small nickel master pattern in the Lambert projection
    of both hemispheres, loaded once per session.
    """
    return kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )


def _hrosm_synthetic_signal(
    xmap: CrystalMap,
    sig_shape: tuple[int, int] = (32, 32),
    pc: tuple[float, float, float] = (0.42, 0.22, 0.50),
    noise: float = 0.0,
    seed: int = 50,
) -> tuple[kp.signals.EBSD, kp.detectors.EBSDDetector, kp.signals.EBSDMasterPattern]:
    """Return an EBSD signal simulated from the small nickel master
    pattern at 20 kV for the rotations of a crystal map, with the
    detector and the master pattern.

    The signal has the map's 2D grid as navigation shape: points not in
    the data get the pattern of the identity rotation. With ``noise >
    0``, Gaussian noise of standard deviation ``noise * (max - min)``
    of the patterns, from ``default_rng(seed)``, is added in float32.
    """
    from kikuchipy.indexing._hrosm._grains import _map_grid

    mp = _hrosm_master_pattern()
    det = kp.detectors.EBSDDetector(sig_shape, pc=pc, sample_tilt=70)
    _, grid_shape = _map_grid(xmap)
    rotations = xmap.rotations
    if rotations.ndim > 1:
        rotations = rotations[:, 0]
    data = np.zeros((xmap.is_in_data.size, 4))
    data[:, 0] = 1
    data[xmap.is_in_data] = rotations.data
    s = mp.get_patterns(
        Rotation(data).reshape(*grid_shape), det, energy=20, compute=True
    )
    if noise > 0:
        rng = np.random.default_rng(seed)
        scale = noise * float(s.data.max() - s.data.min())
        s.data += rng.normal(0, scale, size=s.data.shape).astype(np.float32)
    return s, det, mp


# Namelist keys of EMsoft's dictionary indexing and EMHROSM files, and
# the ones stored as float32 (the others as int32 or strings)
_EMSOFT_DI_NAMELIST_KEYS = (
    "nnk",
    "nosm",
    "ipf_wd",
    "ipf_ht",
    "ROI",
    "numdictsingle",
    "numexptsingle",
    "xpc",
    "ypc",
    "L",
    "delta",
    "thetac",
    "energymin",
    "energymax",
)
_EMSOFT_HROSM_NAMELIST_KEYS = (
    "gangle",
    "misorang",
    "nsamples",
    "nosm",
    "orav",
    "numEM",
    "numIter",
    "maxRAMmem",
    "dpfile",
    "OSMfile",
    "OSMtiff",
    "IPFmap",
)
_EMSOFT_FLOAT_KEYS = (
    "xpc",
    "ypc",
    "L",
    "delta",
    "thetac",
    "energymin",
    "energymax",
    "gangle",
    "misorang",
    "maxRAMmem",
)


def _emsoft_strings(values) -> np.ndarray:
    """Return strings as EMsoft stores them, variable-length ASCII."""
    values = [values] if isinstance(values, (str, bytes)) else list(values)
    values = [v.encode("ascii") if isinstance(v, str) else v for v in values]
    return np.array(values, dtype=h5py.string_dtype("ascii"))


def _emsoft_namelist_value(key: str, value) -> np.ndarray:
    """Return a namelist value as EMsoft stores it."""
    if isinstance(value, (str, bytes)):
        return _emsoft_strings(value)
    dtype = np.float32 if key in _EMSOFT_FLOAT_KEYS else np.int32
    return np.atleast_1d(np.asarray(value, dtype=dtype))


def _fortran_value(value) -> str:
    """Return a value as Fortran namelist text."""
    if isinstance(value, (bool, np.bool_)):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, str):
        return f"'{value}'"
    return " ".join(str(v) for v in np.atleast_1d(value))


def _write_emsoft_layout_file(path: Path | str, kind: str, arrays: dict) -> Path:
    """Write a minimal HDF5 file in the layout of EMsoft's dictionary
    indexing (``kind="dot_product"``) or EMHROSM (``kind="hrosm"``)
    output and return its path.

    ``arrays`` holds NumPy arrays in NumPy's order (maps as (H, W),
    Euler angles as (N, 3)), which is how h5py sees EMsoft's Fortran
    ``(x, y)`` and ``(3, N)`` arrays, and namelist values. Recognised
    special keys: ``"version"`` (EMsoft version string, default
    "6_0_20260411_0"), ``"namelist_text"`` (list of lines of the
    namelist file stored under ``NMLfiles``) and, for ``"hrosm"``,
    ``"dilate"`` (bool, written only to the namelist text, as EMsoft
    does).

    Dot product files: namelist keys go to
    ``NMLparameters/EMDINameList``, all other arrays to ``Scan
    1/EBSD/Data``; ``TopMatchIndices`` and ``TopDotProductList`` are
    padded with zero rows to a multiple of ``numexptsingle`` (default
    the number of rows). HROSM files: namelist keys go to
    ``NMLparameters/HROSMNameList``, all other arrays to
    ``EMData/HROSM``, and ``NMLfiles/HROSMNML`` holds the namelist
    text with ``dilate``. Strings are variable-length ASCII, read back
    as object arrays of bytes; scalars are stored with shape (1,).
    """
    if kind not in ("dot_product", "hrosm"):
        raise ValueError(f"kind {kind!r} must be 'dot_product' or 'hrosm'")
    path = Path(path)
    arrays = dict(arrays)
    version = arrays.pop("version", "6_0_20260411_0")
    namelist_text = arrays.pop("namelist_text", None)
    with h5py.File(path, "w") as f:
        if kind == "dot_product":
            f.create_dataset("EMheader/Version", data=_emsoft_strings(version))
            n_single = arrays.get("numexptsingle")
            for key, value in arrays.items():
                if key in _EMSOFT_DI_NAMELIST_KEYS:
                    f.create_dataset(
                        f"NMLparameters/EMDINameList/{key}",
                        data=_emsoft_namelist_value(key, value),
                    )
                    continue
                value = np.asarray(value)
                if key in ("TopMatchIndices", "TopDotProductList"):
                    multiple = int(n_single) if n_single is not None else len(value)
                    n_pad = -len(value) % multiple
                    padding = np.zeros((n_pad,) + value.shape[1:], value.dtype)
                    value = np.concatenate([value, padding])
                f.create_dataset(f"Scan 1/EBSD/Data/{key}", data=np.atleast_1d(value))
            if namelist_text is not None:
                f.create_dataset(
                    "NMLfiles/DictionaryIndexingNML",
                    data=_emsoft_strings(namelist_text),
                )
        else:
            f.create_dataset("EMheader/HROSM/Version", data=_emsoft_strings(version))
            dilate = bool(arrays.pop("dilate", False))
            namelist = {}
            for key, value in arrays.items():
                if key in _EMSOFT_HROSM_NAMELIST_KEYS:
                    namelist[key] = value
                    f.create_dataset(
                        f"NMLparameters/HROSMNameList/{key}",
                        data=_emsoft_namelist_value(key, value),
                    )
                    continue
                f.create_dataset(
                    f"EMData/HROSM/{key}", data=np.atleast_1d(np.asarray(value))
                )
            if namelist_text is None:
                namelist["dilate"] = dilate
                namelist_text = (
                    [" &HROSMdata", "! The line above must not be changed"]
                    + [f" {k} = {_fortran_value(v)}," for k, v in namelist.items()]
                    + [" /"]
                )
            f.create_dataset("NMLfiles/HROSMNML", data=_emsoft_strings(namelist_text))
    return path


@pytest.fixture
def hrosm_gradient_xmap() -> Callable:
    """Return the generator :func:`_hrosm_gradient_xmap` of crystal maps
    with a constant orientation gradient.
    """
    return _hrosm_gradient_xmap


@pytest.fixture
def hrosm_constant_pair_xmap() -> Callable:
    """Return the generator :func:`_hrosm_constant_pair_xmap` of crystal
    maps with one disorientation angle between all neighbours.
    """
    return _hrosm_constant_pair_xmap


@pytest.fixture
def hrosm_grain_xmap() -> Callable:
    """Return the generator :func:`_hrosm_grain_xmap` of two- and
    three-grain crystal maps with their true labels.
    """
    return _hrosm_grain_xmap


@pytest.fixture
def hrosm_top_lists() -> Callable:
    """Return the generator :func:`_hrosm_top_lists` of 1-based
    top-match lists.
    """
    return _hrosm_top_lists


@pytest.fixture
def write_emsoft_layout_file() -> Callable:
    """Return the writer :func:`_write_emsoft_layout_file` of minimal
    EMsoft-layout dot product and HROSM files.
    """
    return _write_emsoft_layout_file


@pytest.fixture
def hrosm_synthetic_signal() -> Callable:
    """Return the generator :func:`_hrosm_synthetic_signal` of EBSD
    signals simulated for a crystal map's rotations.
    """
    return _hrosm_synthetic_signal
