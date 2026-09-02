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

from __future__ import annotations

import copy
import datetime
import gc
import logging
import os
from pathlib import Path
from time import sleep, time
from typing import TYPE_CHECKING, Iterable, Sequence
import warnings

import dask
import dask.array as da
from dask.diagnostics.progress import ProgressBar
from h5py import File
import hyperspy.api as hs
from hyperspy.axes import AxesManager
from hyperspy.roi import BaseInteractiveROI
from hyperspy.signals import Signal2D
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from scipy.ndimage import correlate, gaussian_filter
from skimage.util.dtype import dtype_range

from kikuchipy._constants import pyopencl_context_available, verify_dependency_or_raise
from kikuchipy._utils.hyperspy_utils import LearningResults
from kikuchipy.detectors._ebsd_detector import EBSDDetector
from kikuchipy.filters.fft_barnes import _fft_filter, _fft_filter_setup
from kikuchipy.filters.window import Window
from kikuchipy.indexing._dictionary_indexing import _dictionary_indexing
from kikuchipy.indexing._hough_indexing import (
    _hough_indexing,
    _indexer_is_compatible_with_kikuchipy,
    _optimize_pc,
    _phase_lists_are_compatible,
)
from kikuchipy.indexing._refinement._refinement import (
    _refine_orientation,
    _refine_orientation_pc,
    _refine_pc,
)
from kikuchipy.indexing._spherical._euler import (
    rotation_from_zyz as _rotation_from_zyz,
)
from kikuchipy.indexing._spherical._euler import (
    rotation_to_zyz as _rotation_to_zyz,
)
from kikuchipy.indexing._spherical._indexer import SphericalIndexer
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    NormalizedCrossCorrelationMetric,
)
from kikuchipy.indexing.similarity_metrics._normalized_dot_product import (
    NormalizedDotProductMetric,
)
from kikuchipy.indexing.similarity_metrics._similarity_metric import SimilarityMetric
from kikuchipy.io._io import _save
from kikuchipy.pattern._pattern import (
    _downsample2d,
    _dynamic_background_frequency_space_setup,
    _get_image_quality,
    _remove_dynamic_background,
    _remove_static_background_divide,
    _remove_static_background_subtract,
    fft_filter,
    fft_frequency_vectors,
)
from kikuchipy.pattern.chunk import _average_neighbour_patterns, get_dynamic_background
from kikuchipy.pattern.chunk import fft_filter as fft_filter_chunk
from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D, LazyKikuchipySignal2D
from kikuchipy.signals.util._crystal_map import (
    _equal_phase,
    _get_indexed_points_in_data_in_xmap,
    _xmap_is_compatible_with_signal,
)
from kikuchipy.signals.util._dask import (
    _get_chunk_overlap_depth,
    _rechunk_learning_results,
    _update_learning_results,
    get_chunking,
    get_dask_array,
)
from kikuchipy.signals.util._detector import _detector_is_compatible_with_signal
from kikuchipy.signals.util._map_helper import (
    _get_average_dot_product_map,
    _get_neighbour_dot_product_matrices,
)
from kikuchipy.signals.util._overwrite_hyperspy_methods import (
    get_parameters,
    insert_doc_disclaimer,
)
from kikuchipy.signals.util.array_tools import grid_indices
from kikuchipy.signals.virtual_bse_image import VirtualBSEImage

if TYPE_CHECKING:  # pragma: no cover
    try:
        from pyebsdindex.ebsd_index import EBSDIndexer
    except ImportError:
        pass

    from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern

_logger = logging.getLogger(__name__)


class EBSD(KikuchipySignal2D):
    """Scan of Electron Backscatter Diffraction (EBSD) patterns.

    This class extends HyperSpy's Signal2D class for EBSD patterns. Some
    of the docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal2d.Signal2D` for the list of
    inherited attributes and methods.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    detector : :class:`~kikuchipy.detectors.EBSDDetector`, optional
        Detector describing the EBSD detector-sample geometry. If not
        given, this is a default detector.
    static_background : numpy.ndarray or dask.array.Array, optional
        Static background pattern. If not given, this is None.
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map containing the phases, unit cell rotations and
        auxiliary properties of the EBSD dataset. If not given, this is
        None.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.

    See Also
    --------
    kikuchipy.data.nickel_ebsd_small :
        An EBSD signal with ``(3, 3)`` experimental nickel patterns.
    kikuchipy.data.nickel_ebsd_large :
        An EBSD signal with ``(55, 75)`` experimental nickel patterns.
    kikuchipy.data.si_ebsd_moving_screen :
        An EBSD signal with experimental silicon patterns.

    Examples
    --------
    Load one of the example datasets and inspect some properties

    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
    >>> s.detector
    EBSDDetector
      shape (Ny, Nx):     (60, 60)
      pc (PCx, PCy, PCz): (0.425, 0.213, 0.501)
      sample_tilt:        70.0 deg
      tilt:               0.0 deg
      azimuthal:          0.0 deg
      twist:              0.0 deg
      binning:            8
      px_size:            1.0 um
    >>> s.static_background
    array([[84, 87, 90, ..., 27, 29, 30],
           [87, 90, 93, ..., 27, 28, 30],
           [92, 94, 97, ..., 39, 28, 29],
           ...,
           [80, 82, 84, ..., 36, 30, 26],
           [79, 80, 82, ..., 28, 26, 26],
           [76, 78, 80, ..., 26, 26, 25]], shape=(60, 60), dtype=uint8)
    >>> s.xmap
    Phase  Orientations  Name  Space group  Point group  Proper point group     Color
        0    9 (100.0%)    ni        Fm-3m         m-3m                 432  tab:blue
    Properties: scores, z
    Scan unit: px
    """

    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _custom_attributes = ["detector", "static_background", "xmap"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._detector = kwargs.get(
            "detector",
            EBSDDetector(
                shape=self._signal_shape_rc,
                px_size=self.axes_manager.signal_axes[0].scale,
            ),
        )
        self._static_background = kwargs.get("static_background")
        self._xmap = kwargs.get("xmap")

    # ---------------------- Custom attributes ----------------------- #

    @property
    def detector(self) -> EBSDDetector:
        """Return or set the detector describing the EBSD
        detector-sample geometry.

        Parameters
        ----------
        value : EBSDDetector
            EBSD detector.
        """
        return self._detector

    @detector.setter
    def detector(self, value: EBSDDetector) -> None:
        if _detector_is_compatible_with_signal(
            detector=value,
            nav_shape=self._navigation_shape_rc,
            sig_shape=self._signal_shape_rc,
            raise_if_not=True,
        ):
            self._detector = value

    @property
    def xmap(self) -> CrystalMap | None:
        """Return or set the crystal map containing the phases, unit
        cell rotations and auxiliary properties of the EBSD dataset.

        Parameters
        ----------
        value : ~orix.crystal_map.CrystalMap
            Crystal map with the same shape as the signal navigation
            shape.
        """
        return self._xmap

    @xmap.setter
    def xmap(self, value: CrystalMap) -> None:
        if _xmap_is_compatible_with_signal(
            value, self.axes_manager.navigation_axes[::-1], raise_if_not=True
        ):
            self._xmap = value

    @property
    def static_background(self) -> np.ndarray | da.Array | None:
        """Return or set the static background pattern.

        Parameters
        ----------
        value : ~numpy.ndarray or ~dask.array.Array
            Static background pattern with the same (signal) shape and
            data type as the EBSD signal.
        """
        return self._static_background

    @static_background.setter
    def static_background(self, value: np.ndarray | da.Array) -> None:
        if value.dtype != self.data.dtype:
            warnings.warn("Background pattern has different data type from patterns")
        if value.shape != self._signal_shape_rc:
            warnings.warn("Background pattern has different shape from patterns")
        self._static_background = value

    # ------------------------ Custom methods ------------------------ #

    def extract_grid(
        self, grid_shape: tuple[int, int] | int, return_indices: bool = False
    ) -> EBSD | LazyEBSD | tuple[EBSD | LazyEBSD, np.ndarray]:
        """Return a new signal with patterns from positions in a grid of
        shape ``grid_shape`` evenly spaced in navigation space.

        Parameters
        ----------
        grid_shape
            Tuple of integers or just an integer signifying the number
            of grid indices in each dimension. If 2D, the shape is
            (n columns, n rows).
        return_indices
            Whether to return the indices of the extracted patterns into
            :attr:`data` as an array of shape ``(2,) + grid_shape``.
            Default is ``False``.

        Returns
        -------
        new
            New signal with patterns from indices in a grid
            corresponding to ``grid_shape``. Attributes :attr:`xmap`,
            :attr:`static_background` and :attr:`detector` are deep
            copied.
        indices
            Indices of the extracted patterns into :attr:`data`,
            returned if ``return_indices=True``.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_large(lazy=True)
        >>> s
        <LazyEBSD, title: patterns Scan 1, dimensions: (75, 55|60, 60)>
        >>> s2 = s.extract_grid((5, 4))
        >>> s2
        <LazyEBSD, title: patterns Scan 1, dimensions: (5, 4|60, 60)>
        """
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,)

        nav_shape = self.axes_manager.navigation_shape
        if len(grid_shape) != len(nav_shape) or any(
            [g > n for g, n in zip(grid_shape, nav_shape)]
        ):
            raise ValueError(
                f"grid_shape {grid_shape} must be compatible with navigation shape "
                f"{nav_shape}"
            )

        # NumPy order (rows, columns)
        grid_shape = grid_shape[::-1]
        nav_shape = nav_shape[::-1]

        idx, spacing = grid_indices(grid_shape, nav_shape, return_spacing=True)
        idx_tuple = tuple(idx)

        # Data
        if self._lazy:
            data_new = self.data.vindex[idx_tuple]
        else:
            data_new = self.data[idx_tuple]

        # Crystal map
        if self.xmap is not None:
            mask = np.zeros(nav_shape, dtype=bool)
            mask[idx_tuple] = True
            mask = mask.ravel()
            xmap_new = self.xmap[mask].deepcopy()
        else:
            xmap_new = None

        # EBSD detector
        detector_new = self.detector.deepcopy()
        if detector_new.navigation_shape == nav_shape:
            detector_new.pc = detector_new.pc[idx_tuple]
        elif detector_new.navigation_shape != (1,):
            detector_new.pc = [0.5, 0.5, 0.5]

        # Static background
        bg_new = self.static_background
        if bg_new is not None:
            bg_new = bg_new.copy()

        # Axes manager
        am = self.axes_manager.deepcopy()
        nav_idx = am.navigation_indices_in_array
        for i, size, spacing_i in zip(nav_idx, grid_shape, spacing):
            am[i].size = size
            am[i].scale = spacing_i * am[i].scale
        am_list = [a for a in am.as_dictionary().values()]

        scan_dict = {
            "data": data_new,
            "xmap": xmap_new,
            "detector": detector_new,
            "static_background": bg_new,
            "axes": am_list,
            "metadata": self.metadata.as_dictionary(),
            "original_metadata": self.original_metadata.as_dictionary(),
        }

        if self._lazy:
            new = LazyEBSD(**scan_dict)
        else:
            new = EBSD(**scan_dict)

        out = new
        if return_indices:
            out = (out, idx)

        return out

    def set_scan_calibration(
        self, step_x: int | float = 1.0, step_y: int | float = 1.0
    ) -> None:
        """Set the step size in microns.

        Parameters
        ----------
        step_x
            Scan step size in um per pixel in horizontal direction.
        step_y
            Scan step size in um per pixel in vertical direction.

        See Also
        --------
        set_detector_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['x'].scale
        1.5
        >>> s.set_scan_calibration(step_x=2)  # Microns
        >>> s.axes_manager['x'].scale
        2.0
        """
        # TODO: Update crystal map step size too
        x, y = self.axes_manager.navigation_axes
        x.name, y.name = ("x", "y")
        x.scale, y.scale = (step_x, step_y)
        x.units, y.units = ["um"] * 2

    def set_detector_calibration(self, delta: int | float) -> None:
        """Set detector pixel size in microns. The offset is set to the
        the detector center.

        Parameters
        ----------
        delta
            Detector pixel size in microns.

        See Also
        --------
        set_scan_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['dx'].scale  # Default value
        1.0
        >>> s.set_detector_calibration(delta=70.)
        >>> s.axes_manager['dx'].scale
        70.0
        """
        # TODO: Update detector pixel size too
        center = delta * np.array(self.axes_manager.signal_shape) / 2
        dx, dy = self.axes_manager.signal_axes
        dx.units, dy.units = ["um"] * 2
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -center

    def remove_static_background(
        self,
        operation: str = "subtract",
        static_bg: np.ndarray | da.Array | None = None,
        scale_bg: bool = False,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        """Remove the static background.

        The removal is performed by subtracting or dividing by a static
        background pattern. Resulting pattern intensities are rescaled
        loosing relative intensities and stretched to fill the available
        grey levels in the patterns' data type range.

        Parameters
        ----------
        operation
            Whether to ``"subtract"`` (default) or ``"divide"`` by the
            static background pattern.
        static_bg
            Static background pattern. If not given, the background is
            obtained from the ``EBSD.static_background`` property.
        scale_bg
            Whether to scale the static background pattern to each
            individual pattern's data range before removal. Default is
            ``False``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Background corrected signal, returned if ``inplace=False``.
            Whether it is lazy is determined from ``lazy_output``.

        See Also
        --------
        remove_dynamic_background

        Examples
        --------
        It is assumed that a static background pattern of the same shape
        and data type (e.g. 8-bit unsigned integer, ``uint8``) as the
        patterns is available in signal metadata:

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.static_background
        array([[84, 87, 90, ..., 27, 29, 30],
               [87, 90, 93, ..., 27, 28, 30],
               [92, 94, 97, ..., 39, 28, 29],
               ...,
               [80, 82, 84, ..., 36, 30, 26],
               [79, 80, 82, ..., 28, 26, 26],
               [76, 78, 80, ..., 26, 26, 25]], shape=(60, 60), dtype=uint8)

        The static background can be removed by subtracting or dividing
        this background from each pattern:

        >>> s.remove_static_background(operation="divide")

        If the ``static_background`` property is ``None``, this must be
        passed in the ``static_bg`` parameter as a ``numpy`` or ``dask``
        array.
        """
        if lazy_output and inplace:
            raise ValueError("'lazy_output=True' requires 'inplace=False'")

        dtype = np.float32  # During processing
        dtype_out = self.data.dtype.type
        omin, omax = dtype_range[dtype_out]

        # Get background pattern
        if static_bg is None:
            static_bg = self.static_background
            try:
                if not isinstance(static_bg, (np.ndarray, da.Array)):
                    raise ValueError
            except (AttributeError, ValueError):
                raise ValueError("`EBSD.static_background` is not a valid array")
        if isinstance(static_bg, da.Array):
            static_bg = static_bg.compute()
        if dtype_out != static_bg.dtype:
            raise ValueError(
                f"Static background dtype_out {static_bg.dtype} is not the same as "
                f"pattern dtype_out {dtype_out}"
            )
        pat_shape = self._signal_shape_rc  # xy -> ij
        bg_shape = static_bg.shape
        if bg_shape != pat_shape:
            raise ValueError(
                f"Signal {pat_shape} and static background {bg_shape} shapes are not "
                "the same"
            )
        static_bg = static_bg.astype(dtype)

        # Remove background and rescale to input data type
        if operation == "subtract":
            operation_func = _remove_static_background_subtract
        else:
            operation_func = _remove_static_background_divide

        map_kw = {
            "show_progressbar": show_progressbar,
            "output_dtype": dtype_out,
            "static_bg": static_bg,
            "dtype_out": dtype_out,
            "omin": omin,
            "omax": omax,
            "scale_bg": scale_bg,
        }
        attrs = self._get_custom_attributes()
        if inplace:
            self.map(operation_func, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(
                operation_func, inplace=False, lazy_output=lazy_output, **map_kw
            )
            s_out._set_custom_attributes(attrs)
            return s_out

    def remove_dynamic_background(
        self,
        operation: str = "subtract",
        filter_domain: str = "frequency",
        std: int | float | None = None,
        truncate: int | float = 4.0,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
        **kwargs,
    ) -> EBSD | LazyEBSD | None:
        """Remove the dynamic background.

        The removal is performed by subtracting or dividing by a
        Gaussian blurred version of each pattern. Resulting pattern
        intensities are rescaled to fill the input patterns' data type
        range individually.

        Parameters
        ----------
        operation
            Whether to ``"subtract"`` (default) or ``"divide"`` by the
            dynamic background pattern.
        filter_domain
            Whether to obtain the dynamic background by applying a
            Gaussian convolution filter in the ``"frequency"`` (default)
            or ``"spatial"`` domain.
        std
            Standard deviation of the Gaussian window. If None
            (default), it is set to width/8.
        truncate
            Truncate the Gaussian window at this many standard
            deviations. Default is ``4.0``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.
        **kwargs
            Keyword arguments passed to the Gaussian blurring function
            determined from ``filter_domain``.

        Returns
        -------
        s_out
            Background corrected signal, returned if ``inplace=False``.
            Whether it is lazy is determined from ``lazy_output``.

        See Also
        --------
        remove_static_background,
        get_dynamic_background,
        kikuchipy.pattern.remove_dynamic_background,
        kikuchipy.pattern.get_dynamic_background

        Examples
        --------
        Remove the static and dynamic background

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.remove_static_background()
        >>> s.remove_dynamic_background(operation="divide", std=5)
        """
        if lazy_output and inplace:
            raise ValueError("'lazy_output=True' requires 'inplace=False'")

        if std is None:
            std = self.axes_manager.signal_shape[0] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            # FFT filter setup for Connelly Barnes' algorithm
            filter_func = _fft_filter
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self._signal_shape_rc,
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}")

        map_func = _remove_dynamic_background

        dtype_out = self.data.dtype.type
        omin, omax = dtype_range[dtype_out]

        map_kw = {
            "show_progressbar": show_progressbar,
            "output_dtype": dtype_out,
            "filter_func": filter_func,
            "operation": operation,
            "dtype_out": dtype_out,
            "omin": omin,
            "omax": omax,
            **kwargs,
        }
        attrs = self._get_custom_attributes()
        if inplace:
            self.map(map_func, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(map_func, inplace=False, lazy_output=lazy_output, **map_kw)
            s_out._set_custom_attributes(attrs)
            return s_out

    def get_dynamic_background(
        self,
        filter_domain: str = "frequency",
        std: int | float | None = None,
        truncate: int | float = 4.0,
        dtype_out: str | np.dtype | type | None = None,
        show_progressbar: bool | None = None,
        lazy_output: bool | None = None,
        **kwargs,
    ) -> EBSD | LazyEBSD:
        """Return the dynamic background per pattern in a new signal.

        Parameters
        ----------
        filter_domain
            Whether to apply a Gaussian convolution filter in the
            ``"frequency"`` (default) or ``"spatial"`` domain.
        std
            Standard deviation of the Gaussian window. If not given, it
            is set to width/8.
        truncate
            Truncate the Gaussian filter at this many standard
            deviations. Default is ``4.0``.
        dtype_out
            Data type of the background patterns. If not given, it is
            set to the same data type as the input pattern.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal.
        **kwargs
            Keyword arguments passed to the Gaussian blurring function
            determined from ``filter_domain``.

        Returns
        -------
        s_out
            Signal with the large scale variations across the detector.
            Whether it is lazy is determined from ``lazy_output``.
        """
        if std is None:
            std = self.axes_manager.signal_shape[-1] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            filter_func = _fft_filter
            # FFT filter setup for Connelly Barnes' algorithm
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self._signal_shape_rc,
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}")

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)
        dask_array = get_dask_array(self, dtype=dtype_out)

        background_patterns = dask_array.map_blocks(
            func=get_dynamic_background,
            filter_func=filter_func,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        attrs = self._get_custom_attributes()
        if lazy_output or (lazy_output is None and self._lazy):
            s_out = LazyEBSD(background_patterns, **attrs)
        else:
            background_return = np.empty(
                shape=background_patterns.shape, dtype=dtype_out
            )

            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            background_patterns.store(background_return, compute=True)
            s_out = EBSD(background_return, **attrs)

            try:
                pbar.unregister()
            except KeyError:
                pass

        return s_out

    def fft_filter(
        self,
        transfer_function: np.ndarray | Window,
        function_domain: str,
        shift: bool = False,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        """Filter patterns in the frequency domain.

        Patterns are transformed via the Fast Fourier Transform (FFT) to
        the frequency domain, where their spectrum is multiplied by the
        ``transfer_function``, and the filtered spectrum is subsequently
        transformed to the spatial domain via the inverse FFT (IFFT).
        Filtered patterns are rescaled to input data type range.

        Note that if ``function_domain`` is ``"spatial"``, only real
        valued FFT and IFFT is used.

        Parameters
        ----------
        transfer_function
            Filter to apply to patterns. This can either be a transfer
            function in the frequency domain of pattern shape or a
            kernel in the spatial domain. What is passed is determined
            from ``function_domain``.
        function_domain
            Options are ``"frequency"`` and ``"spatial"``, indicating,
            respectively, whether the filter function passed to
            ``filter_function`` is a transfer function in the frequency
            domain or a kernel in the spatial domain.
        shift
            Whether to shift the zero-frequency component to the center.
            Default is ``False``. This is only used when
            ``function_domain="frequency"``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Filtered signal, returned if ``inplace=False``. Whether it
            is lazy is determined from ``lazy_output``.

        See Also
        --------
        kikuchipy.filters.Window

        Examples
        --------
        Applying a Gaussian low pass filter with a cutoff frequency of
        20:

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> pattern_shape = s.axes_manager.signal_shape[::-1]
        >>> w = kp.filters.Window(
        ...     "lowpass", cutoff=20, shape=pattern_shape
        ... )
        >>> s.fft_filter(
        ...     transfer_function=w,
        ...     function_domain="frequency",
        ...     shift=True,
        ... )
        """
        if lazy_output and inplace:
            raise ValueError("'lazy_output=True' requires 'inplace=False'")

        dtype_out = self.data.dtype.type
        dtype = np.float32
        dask_array = get_dask_array(signal=self, dtype=dtype)

        kwargs = {}
        if function_domain == "frequency":
            filter_func = fft_filter
            kwargs["shift"] = shift
        elif function_domain == "spatial":
            filter_func = _fft_filter  # Barnes
            kwargs["window_shape"] = transfer_function.shape

            # FFT filter setup
            (
                kwargs["fft_shape"],
                transfer_function,  # Padded
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _fft_filter_setup(
                image_shape=self._signal_shape_rc,
                window=transfer_function,
            )
        else:
            function_domains = ["frequency", "spatial"]
            raise ValueError(f"{function_domain} must be either of {function_domains}")

        filtered_patterns = dask_array.map_blocks(
            func=fft_filter_chunk,
            filter_func=filter_func,
            transfer_function=transfer_function,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        return_lazy = lazy_output or (lazy_output is None and self._lazy)
        register_pbar = show_progressbar or (
            show_progressbar is None and hs.preferences.General.show_progressbar
        )
        if not return_lazy and register_pbar:
            pbar = ProgressBar()
            pbar.register()

        if inplace:
            if not return_lazy:
                filtered_patterns.store(self.data, compute=True)
            else:
                self.data = filtered_patterns
            s_out = None
        else:
            s_out = LazyEBSD(filtered_patterns, **self._get_custom_attributes())
            if not return_lazy:
                s_out.compute()

        if not return_lazy and register_pbar:
            pbar.unregister()

        if s_out:
            return s_out

    def average_neighbour_patterns(
        self,
        window: str | np.ndarray | da.Array | Window = "circular",
        window_shape: tuple[int, ...] = (3, 3),
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
        **kwargs,
    ) -> EBSD | LazyEBSD | None:
        """Average patterns with its neighbours within a window.

        The amount of averaging is specified by the window coefficients.
        All patterns are averaged with the same window. Map borders are
        extended with zeros. Resulting pattern intensities are rescaled
        to fill the input patterns' data type range individually.

        Averaging is accomplished by correlating the window with the
        extended array of patterns using
        :func:`scipy.ndimage.correlate`.

        Parameters
        ----------
        window
            Name of averaging window or an array. Available types are
            listed in :func:`scipy.signal.windows.get_window`, in
            addition to a ``"circular"`` window (default) filled with
            ones in which corner coefficients are set to zero. A window
            element is considered to be in a corner if its radial
            distance to the origin (window center) is shorter or equal
            to the half width of the window's longest axis. A 1D or 2D
            :class:`~numpy.ndarray`, :class:`~dask.array.Array` or
            :class:`~kikuchipy.filters.Window` can also be passed.
        window_shape
            Shape of averaging window. Not used if a custom window or
            :class:`~kikuchipy.filters.Window` is passed to ``window``.
            This can be either 1D or 2D, and can be asymmetrical.
            Default is ``(3, 3)``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.
        **kwargs
            Keyword arguments passed to the available window type listed
            in :func:`~scipy.signal.windows.get_window`. If not given,
            the default values of that particular window are used.

        Returns
        -------
        s_out
            Averaged signal, returned if ``inplace=False``. Whether it
            is lazy is determined from ``lazy_output``.

        See Also
        --------
        kikuchipy.filters.Window, scipy.signal.windows.get_window,
        scipy.ndimage.correlate
        """
        if lazy_output and inplace:
            raise ValueError("'lazy_output=True' requires 'inplace=False'")

        if isinstance(window, Window) and window.is_valid:
            averaging_window = copy.copy(window)
        else:
            averaging_window = Window(window=window, shape=window_shape, **kwargs)

        nav_shape = self._navigation_shape_rc
        window_shape = averaging_window.shape
        if window_shape in [(1,), (1, 1)]:
            # Do nothing if a window of shape (1,) or (1, 1) is passed
            warnings.warn(
                f"A window of shape {window_shape} was passed, no averaging is "
                "therefore performed"
            )
            return
        elif len(nav_shape) > len(window_shape):
            averaging_window = averaging_window.reshape(window_shape + (1,))

        # Get sum of window data for each pattern, to normalize with
        # after correlation
        window_sums = correlate(
            input=np.ones(nav_shape, dtype=int),
            weights=averaging_window,
            mode="constant",
        )

        # Add signal dimensions to window array to enable its use with
        # Dask's map_overlap()
        sig_dim = self.axes_manager.signal_dimension
        averaging_window = averaging_window.reshape(
            averaging_window.shape + (1,) * sig_dim
        )

        # Create dask array of signal patterns and do processing on this
        if self._lazy:
            old_chunks = self.data.chunks
        dask_array = get_dask_array(signal=self, chunk_bytes=8e6, rechunk=True)

        # Add signal dimensions to array be able to use with Dask's
        # map_overlap()
        nav_dim = self.axes_manager.navigation_dimension
        for i in range(sig_dim):
            window_sums = np.expand_dims(window_sums, axis=window_sums.ndim)
        window_sums = da.from_array(
            window_sums, chunks=dask_array.chunks[:nav_dim] + (1,) * sig_dim
        )

        # Create overlap between chunks to enable correlation with the
        # window using Dask's map_overlap()
        window_dim = averaging_window.ndim
        overlap_depth = {}
        for i in range(nav_dim):
            if i < window_dim and dask_array.chunks[i][0] < dask_array.shape[i]:
                overlap_depth[i] = (window_shape[i] // 2) + 1
            else:
                overlap_depth[i] = 1
        overlap_depth.update(
            {i: 0 for i in self.axes_manager.signal_indices_in_array[::-1]}
        )

        dtype_out = self.data.dtype
        omin, omax = dtype_range[dtype_out.type]
        averaged_patterns = da.overlap.map_overlap(
            _average_neighbour_patterns,
            dask_array,
            window_sums,
            window=averaging_window,
            dtype_out=dtype_out,
            omin=omin,
            omax=omax,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )

        return_lazy = lazy_output or (lazy_output is None and self._lazy)
        register_pbar = show_progressbar or (
            show_progressbar is None and hs.preferences.General.show_progressbar
        )
        if not return_lazy and register_pbar:
            pbar = ProgressBar()
            pbar.register()

        if inplace:
            if not return_lazy:
                averaged_patterns.store(self.data, compute=True)
            else:
                averaged_patterns = averaged_patterns.rechunk(old_chunks)
                self.data = averaged_patterns
            s_out = None
        else:
            s_out = LazyEBSD(averaged_patterns, **self._get_custom_attributes())
            if not return_lazy:
                s_out.compute()

        # Don't sink
        gc.collect()

        if not return_lazy and register_pbar:
            pbar.unregister()

        if s_out:
            return s_out

    def downsample(
        self,
        factor: int,
        dtype_out: str | None = None,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        r"""Downsample the pattern shape by an integer factor and
        rescale intensities to fill the data type range.

        Parameters
        ----------
        factor
            Integer binning factor to downsample by. Must be a divisor
            of the initial pattern shape :math:`(s_y, s_x)`. The new
            pattern shape given by the ``factor`` :math:`k` is
            :math:`(s_y / k, s_x / k)`.
        dtype_out
            Name of the data type of the new patterns overwriting
            :attr:`data`. Contrast between patterns is lost. If not
            given, patterns maintain their data type and. Patterns are
            rescaled to fill the data type range.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Downsampled signal, returned if ``inplace=False``. Whether
            it is lazy is determined from ``lazy_output``.

        See Also
        --------
        rebin, crop

        Notes
        -----
        This method differs from :meth:`rebin` in that intensities are
        rescaled after binning in order to maintain the data type. If
        rescaling is undesirable, use :meth:`rebin` instead.
        """
        if lazy_output and inplace:
            raise ValueError("'lazy_output=True' requires 'inplace=False'")

        if not isinstance(factor, int) or factor <= 1:
            raise ValueError(f"Binning factor {factor} must be an integer > 1")
        else:
            factor = int(factor)

        sig_shape_old = self.axes_manager.signal_shape
        rest = np.mod(sig_shape_old, factor)
        if not all(rest == 0):
            raise ValueError(
                f"Binning factor {factor} must be a divisor of the initial pattern "
                f"shape {sig_shape_old}, but {tuple(rest)} pixels remain.\n"
                "You might try to crop away these pixels first using EBSD.crop()."
            )
        sig_shape_new = np.array(sig_shape_old) // factor
        sig_shape_new = tuple(map(int, sig_shape_new))
        sig_shape_new = sig_shape_new[::-1]

        if dtype_out is not None:
            dtype_out = np.dtype(dtype_out).type
        else:
            dtype_out = self.data.dtype.type
        omin, omax = dtype_range[dtype_out]

        attrs = self._get_custom_attributes()

        static_bg = attrs["static_background"]
        if static_bg is not None:
            if isinstance(static_bg, da.Array):
                static_bg = static_bg.compute()
            static_bg_new = _downsample2d(static_bg, factor, omin, omax, dtype_out)
            attrs["static_background"] = static_bg_new

        detector: EBSDDetector = attrs["detector"]
        detector.shape = sig_shape_new
        detector.binning *= factor

        map_kw = {
            "show_progressbar": show_progressbar,
            "output_dtype": dtype_out,
            "factor": factor,
            "omin": omin,
            "omax": omax,
            "dtype_out": dtype_out,
        }
        if inplace:
            self.map(_downsample2d, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(
                _downsample2d, inplace=False, lazy_output=lazy_output, **map_kw
            )
            s_out._set_custom_attributes(attrs)
            return s_out

    def get_neighbour_dot_product_matrices(
        self,
        window: Window | None = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: str | np.dtype | type = "float32",
        show_progressbar: bool | None = None,
    ) -> np.ndarray | da.Array:
        """Get an array with dot products of a pattern and its
        neighbours within a window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the dot products with. If not given, the four
            nearest neighbours are used. Must have the same number of
            dimensions as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is ``True``.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            ``zero_mean=True``. Default is ``True``.
        dtype_out
            Data type of the output map. Default is ``"float32"``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        dp_matrices
            Dot products between a pattern and its nearest neighbours.
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        dtype_out = np.dtype(dtype_out)

        dp_matrices = dask_array.map_overlap(
            _get_neighbour_dot_product_matrices,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array[::-1],
            new_axis=tuple(np.arange(window.ndim) + nav_dim),
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )

        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            dp_matrices = dp_matrices.compute()

            try:
                pbar.unregister()
            except KeyError:
                pass

        return dp_matrices

    def get_image_quality(
        self,
        normalize: bool = True,
        show_progressbar: bool | None = None,
    ) -> np.ndarray | da.Array:
        """Compute the image quality map of patterns in an EBSD scan.

        The image quality :math:`Q` is calculated based on the procedure
        defined by Krieger Lassen :cite:`lassen1994automated`.

        Parameters
        ----------
        normalize
            Whether to normalize patterns to a mean of zero and standard
            deviation of 1 before calculating the image quality. Default
            is ``True``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        image_quality_map
            Image quality map of same shape as navigation axes. This is
            a Dask array if the signal is lazy.

        See Also
        --------
        kikuchipy.pattern.get_image_quality

        Examples
        --------
        Load an example dataset, remove the static and dynamic
        background and compute :math:`Q`

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s
        <EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
        >>> s.remove_static_background()
        >>> s.remove_dynamic_background()
        >>> iq = s.get_image_quality()
        >>> iq
        array([[0.19935645, 0.16657268, 0.18802597],
               [0.19040637, 0.16169308, 0.17834103],
               [0.19411428, 0.16031112, 0.18414427]], dtype=float32)
        """
        # Calculate frequency vectors
        sx, sy = self.axes_manager.signal_shape
        frequency_vectors = fft_frequency_vectors((sy, sx))
        inertia_max = np.sum(frequency_vectors) / (sy * sx)

        image_quality_map = self.map(
            _get_image_quality,
            show_progressbar=show_progressbar,
            inplace=False,
            output_dtype=np.float32,
            normalize=normalize,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
        )

        return image_quality_map.data

    def get_average_neighbour_dot_product_map(
        self,
        window: Window | None = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: str | np.dtype | type = "float32",
        dp_matrices: np.ndarray | None = None,
        show_progressbar: bool | None = None,
    ) -> np.ndarray | da.Array:
        """Get a map of the average dot product between patterns and
        their neighbours within an averaging window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the average with. If not given, the four nearest
            neighbours are used. Must have the same number of dimensions
            as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is ``True``.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            ``zero_mean=True``. Default is ``True``.
        dtype_out
            Data type of the output map. Default is ``"float32"``.
        dp_matrices
            Optional pre-calculated dot product matrices, by default
            ``None``. If an array is passed, the average dot product map
            is calculated from this array. The ``dp_matrices`` array can
            be obtained from :meth:`get_neighbour_dot_product_matrices`.
            Its shape must correspond to the signal's navigation shape
            and the window's shape.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        adp
            Average dot product map.
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        if dp_matrices is not None:
            nan_slices = [slice(None) for _ in range(nav_dim)]
            nan_slices += [slice(i, i + 1) for i in window.origin]
            dp_matrices2 = dp_matrices.copy()
            dp_matrices2[tuple(nan_slices)] = np.nan
            if nav_dim == 1:
                mean_axis = 1
            else:  # == 2
                mean_axis = (2, 3)
            return np.nanmean(dp_matrices2, axis=mean_axis)

        # Create dask array of signal data and do processing on this
        dask_array = get_dask_array(signal=self)

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        dtype_out = np.dtype(dtype_out)

        adp = dask_array.map_overlap(
            _get_average_dot_product_map,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )
        chunks = get_chunking(
            data_shape=self._navigation_shape_rc,
            nav_dim=nav_dim,
            sig_dim=0,
            dtype=dtype_out,
        )
        adp = adp.rechunk(chunks=chunks)

        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            adp = adp.compute()

            try:
                pbar.unregister()
            except KeyError:
                pass

        return adp

    def plot_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Iterable[int] | Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        """Plot an interactive virtual backscatter electron (VBSE)
        image formed from intensities within a specified and adjustable
        region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.plot_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If not given, the first two navigation axes are used.
        **kwargs:
            Keyword arguments passed to the ``plot()`` method of the
            virtual image.

        See Also
        --------
        get_virtual_bse_intensity

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s.plot_virtual_bse_intensity(rect_roi)
        """
        # Plot signal if necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the ROI
        sliced_signal = roi.interactive(self, axes=self.axes_manager.signal_axes)

        # Create an output signal for the virtual backscatter electron
        # calculation
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Virtual backscatter electron intensity"

        # Create the interactive signal
        hs.interactive(
            f=sliced_signal.nansum,
            axis=sliced_signal.axes_manager.signal_axes,
            event=roi.events.changed,
            recompute_out_event=None,
            out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    def get_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Iterable[int] | Iterable[str] | None = None,
    ) -> VirtualBSEImage:
        """Get a virtual backscatter electron (VBSE) image formed from
        intensities within a region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.get_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If not given, the first two navigation axes are used.

        Returns
        -------
        virtual_image
            VBSE image formed from detector intensities within an ROI
            on the detector.

        See Also
        --------
        plot_virtual_bse_intensity

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s = kp.data.nickel_ebsd_small()
        >>> vbse_image = s.get_virtual_bse_intensity(rect_roi)
        """
        vbse = roi(self, axes=self.axes_manager.signal_axes)
        vbse_sum = self._get_sum_signal(vbse, out_signal_axes)
        vbse_sum.metadata.General.title = "Virtual backscatter electron image"
        vbse_sum.set_signal_type("VirtualBSEImage")
        return vbse_sum

    def hough_indexing(
        self,
        phase_list: PhaseList,
        indexer: "EBSDIndexer",
        chunksize: int = 528,
        verbose: int = 1,
        return_index_data: bool = False,
        return_band_data: bool = False,
    ) -> (
        CrystalMap
        | tuple[CrystalMap, np.ndarray]
        | tuple[CrystalMap, np.ndarray, np.ndarray]
    ):
        """Index patterns by Hough indexing using :mod:`pyebsdindex`.

        See :class:`~pyebsdindex.ebsd_index.EBSDIndexer` and
        :meth:`~pyebsdindex.ebsd_index.EBSDIndexer.index_pats` for
        details.

        Currently, PyEBSDIndex only supports indexing with a single
        mean projection center (PC).

        Parameters
        ----------
        phase_list
            List of phases. The list must correspond to the phase list
            in the passed.
        indexer
            PyEBSDIndex EBSD indexer instance of which the
            :meth:`~pyebsdindex.ebsd_index.EBSDIndexer.index_pats`
            method is called. Its `phaselist` must be compatible with
            the given ``phase_list``, and the ``indexer.vendor`` must be
            ``"KIKUCHIPY"``. An indexer can be obtained with
            :meth:`~kikuchipy.detectors.EBSDDetector.get_indexer`.
        chunksize
            Number of patterns to index at a time. Default is the
            minimum of 528 or the number of patterns in the signal.
            Increasing the chunksize may give faster indexing but
            increases memory use.
        verbose
            Which information to print from PyEBSDIndex. Options are
            0 - no output, 1 - timings (default), 2 - timings and the
            Hough transform of the first pattern with detected bands
            highlighted.
        return_index_data
            Whether to return the index data array returned from
            ``EBSDIndexer.index_pats()`` in addition to the resulting
            crystal map. Default is ``False``.
        return_band_data
            Whether to return the band data array returned from
            ``EBSDIndexer.index_pats()``. Default is ``False``.

        Returns
        -------
        xmap
            Crystal map with indexing results.
        index_data
            Array returned from ``EBSDIndexer.index_pats()``, returned
            if ``return_index_data=True``.
        band_data
            Array returned from ``EBSDIndexer.index_pats()``, returned
            if ``return_band_data=True``.

        Notes
        -----
        Requires :mod:`pyebsdindex` to be installed. See
        :ref:`dependencies` for further details.

        This wrapper of PyEBSDIndex is meant for convenience more than
        speed. It uses the GPU if :mod:`pyopencl` is installed, but only
        uses a single thread. If you need the fastest indexing, refer to
        the PyEBSDIndex documentation for multi-threading and more.
        """
        verify_dependency_or_raise("pyebsdindex", "Hough indexing")

        if self._lazy and not pyopencl_context_available:  # pragma: no cover
            raise ValueError(
                "Hough indexing of lazy signals must use PyOpenCL, which must be able "
                "to create a context. See https://documen.tician.de/pyopencl/misc.html "
                "for details"
            )

        am = self.axes_manager
        nav_shape = am.navigation_shape[::-1]
        nav_size = int(np.prod(nav_shape))
        sig_shape = am.signal_shape[::-1]
        step_sizes = tuple([a.scale for a in am.navigation_axes[::-1]])

        # Check indexer (but not the reflectors)
        _ = _indexer_is_compatible_with_kikuchipy(
            indexer, sig_shape, nav_size, raise_if_not=True
        )
        _ = _phase_lists_are_compatible(phase_list, indexer, raise_if_not=True)

        # Prepare patterns
        chunksize = min(chunksize, max(am.navigation_size, 1))
        patterns = self.data.reshape((-1,) + sig_shape)
        if self._lazy:  # pragma: no cover
            patterns = patterns.rechunk({0: chunksize, 1: -1, 2: -1})

        xmap, index_data, band_data = _hough_indexing(
            patterns=patterns,
            phase_list=phase_list,
            nav_shape=nav_shape,
            step_sizes=step_sizes,
            indexer=indexer,
            chunksize=chunksize,
            verbose=verbose,
        )

        xmap.scan_unit = _get_navigation_axes_unit(am)

        if return_index_data and return_band_data:
            return xmap, index_data, band_data
        elif return_index_data and not return_band_data:
            return xmap, index_data
        elif not return_index_data and return_band_data:
            return xmap, band_data
        else:
            return xmap

    def hough_indexing_optimize_pc(
        self,
        pc0: np.ndarray | list | tuple,
        indexer: "EBSDIndexer",
        batch: bool = False,
        method: str = "Nelder-Mead",
        **kwargs,
    ) -> "EBSDDetector":
        """Return a detector with one projection center (PC) per
        pattern optimized using Hough indexing from :mod:`pyebsdindex`.

        See :class:`~pyebsdindex.ebsd_index.EBSDIndexer` and
        :mod:`~pyebsdindex.pcopt` for details.

        Parameters
        ----------
        pc0
            A single initial guess of PC for all patterns in Bruker's
            convention, (PCx, PCy, PCz).
        indexer
            PyEBSDIndex EBSD indexer instance to pass on to the
            optimization function. An indexer can be obtained with
            :meth:`~kikuchipy.detectors.EBSDDetector.get_indexer`.
        batch
            Whether the fit for the patterns should be optimized using
            the cumulative fit for all patterns (``False``, default), or
            if an optimization is run for each pattern individually.
        method
            Which optimization method to use, either ``"Nelder-Mead"``
            from SciPy (default) or ``"PSO"`` (particle swarm).
        **kwargs
            Keyword arguments passed on to PyEBSDIndex' optimization
            method (depending on the chosen ``method``).

        Returns
        -------
        new_detector
            EBSD detector with one PC if ``batch=False`` or one PC per
            pattern if ``batch=True``. The detector attributes are
            extracted from ``indexer.sampleTilt`` etc.

        Notes
        -----
        Requires :mod:`pyebsdindex` to be installed. See
        :ref:`dependencies` for further details.
        """
        verify_dependency_or_raise(
            "pyebsdindex", "Projection center optimization with Hough indexing"
        )

        if self._lazy and not pyopencl_context_available:  # pragma: no cover
            raise ValueError(
                "Hough indexing of lazy signals must use PyOpenCL, which must be able "
                "to create a context. See https://documen.tician.de/pyopencl/misc.html "
                "for details"
            )

        pc0 = np.asarray(pc0)
        if pc0.size != 3:
            raise ValueError("`pc0` must be of size 3")
        pc0 = list(pc0.squeeze())

        supported_methods = ["nelder-mead", "pso"]
        method = method.lower()
        if method not in supported_methods:
            raise ValueError(
                f"`method` '{method}' must be one of the supported methods "
                f"{supported_methods}"
            )

        am = self.axes_manager
        nav_shape = am.navigation_shape[::-1]
        nav_size = int(np.prod(nav_shape))
        sig_shape = am.signal_shape[::-1]

        # Check indexer
        _ = _indexer_is_compatible_with_kikuchipy(
            indexer, sig_shape, nav_size, check_pc=False, raise_if_not=True
        )

        # Prepare patterns
        patterns = self.data.reshape((-1,) + sig_shape)
        if self._lazy:  # pragma: no cover
            patterns = patterns.rechunk({0: "auto", 1: -1, 2: -1})

        pc = _optimize_pc(
            pc0=pc0,
            patterns=patterns,
            indexer=indexer,
            batch=batch,
            method=method,
            **kwargs,
        )

        if batch:
            pc = pc.reshape(nav_shape + (3,))

        new_detector = EBSDDetector(
            shape=sig_shape,
            pc=pc,
            sample_tilt=indexer.sampleTilt,
            tilt=indexer.camElev,
        )

        return new_detector

    def dictionary_indexing(
        self,
        dictionary: EBSD,
        metric: SimilarityMetric | str = "ncc",
        keep_n: int = 20,
        n_per_iteration: int | None = None,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        rechunk: bool = False,
        dtype: str | np.dtype | type | None = None,
    ) -> CrystalMap:
        """Index patterns by matching each pattern to a dictionary of
        simulated patterns of known orientations
        :cite:`chen2015dictionary,jackson2019dictionary`.

        Parameters
        ----------
        dictionary
            One EBSD signal with dictionary patterns. The signal must
            have a 1D navigation axis, an :attr:`xmap` property with
            crystal orientations set, and equal detector shape.
        metric
            Similarity metric, by default ``"ncc"`` (normalized
            cross-correlation). ``"ndp"`` (normalized dot product) is
            also available. A valid user-defined similarity metric
            may be used instead. The metric must be a class implementing
            the :class:`~kikuchipy.indexing.SimilarityMetric` abstract
            class methods. See
            :class:`~kikuchipy.indexing.NormalizedCrossCorrelationMetric`
            and :class:`~kikuchipy.indexing.NormalizedDotProductMetric`
            for examples.
        keep_n
            Number of best matches to keep, by default 20 or the number
            of dictionary patterns if fewer than 20 are available.
        n_per_iteration
            Number of dictionary patterns to compare to all experimental
            patterns in each indexing iteration. If not given, and the
            dictionary is a ``LazyEBSD`` signal, it is equal to the
            chunk size of the first pattern array axis, while if if is
            an ``EBSD`` signal, it is set equal to the number of
            dictionary patterns, yielding only one iteration. This
            parameter can be increased to use less memory during
            indexing, but this will increase the computation time.
        navigation_mask
            A boolean mask equal to the signal's navigation (map) shape,
            where only patterns equal to ``False`` are indexed. This can
            be used by ``metric`` in
            :meth:`~kikuchipy.indexing.SimilarityMetric.prepare_experimental`.
            If not given, all patterns are indexed.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape, where only pixels equal to ``False`` are matched.
            This can be used by ``metric`` in
            :meth:`~kikuchipy.indexing.SimilarityMetric.prepare_experimental`.
            If not given, all pixels are used.
        rechunk
            Whether ``metric`` is allowed to rechunk experimental and
            dictionary patterns before matching. Default is ``False``.
            Rechunking usually makes indexing faster, but uses more
            memory. If a custom ``metric`` is passed, whatever
            :attr:`~kikuchipy.indexing.SimilarityMetric.rechunk` is set
            to will be used.
        dtype
            Which data type ``metric`` shall cast the patterns to before
            matching. If not given, ``"float32"`` will be used unless a
            custom ``metric`` is passed and it has set the
            :attr:`~kikuchipy.indexing.SimilarityMetric.dtype`, which
            will then be used instead. ``"float32"`` and ``"float64"``
            are allowed for the available ``"ncc"`` and ``"ndp"``
            metrics.

        Returns
        -------
        xmap
            A crystal map with ``keep_n`` rotations per point with the
            sorted best matching orientations in the dictionary. The
            corresponding best scores and indices into the dictionary
            are stored in the ``xmap.prop`` dictionary as ``"scores"``
            and ``"simulation_indices"``.

        See Also
        --------
        refine_orientation
        refine_projection_center
        refine_orientation_projection_center
        kikuchipy.indexing.SimilarityMetric
        kikuchipy.indexing.NormalizedCrossCorrelationMetric
        kikuchipy.indexing.NormalizedDotProductMetric
        kikuchipy.indexing.merge_crystal_maps :
            Merge multiple single phase crystal maps into one multi
            phase map.
        kikuchipy.indexing.orientation_similarity_map :
            Calculate an orientation similarity map.
        """
        am_exp = self.axes_manager
        am_dict = dictionary.axes_manager
        dict_size = am_dict.navigation_size

        if n_per_iteration is None:
            if isinstance(dictionary.data, da.Array):
                n_per_iteration = dictionary.data.chunksize[0]
            else:
                n_per_iteration = dict_size

        nav_shape_exp = am_exp.navigation_shape[::-1]
        if navigation_mask is not None:
            if navigation_mask.shape != nav_shape_exp:
                raise ValueError(
                    f"The navigation mask shape {navigation_mask.shape} and the "
                    f"signal's navigation shape {nav_shape_exp} must be identical"
                )
            elif navigation_mask.all():
                raise ValueError(
                    "The navigation mask must allow for indexing of at least one "
                    "pattern (at least one value equal to `False`)"
                )
            elif not isinstance(navigation_mask, np.ndarray):
                raise ValueError("The navigation mask must be a NumPy array")

        if signal_mask is not None:
            if not isinstance(signal_mask, np.ndarray):
                raise ValueError("The signal mask must be a NumPy array")

        sig_shape_exp = am_exp.signal_shape[::-1]
        sig_shape_dict = am_dict.signal_shape[::-1]
        if sig_shape_exp != sig_shape_dict:
            raise ValueError(
                f"Experimental {sig_shape_exp} and dictionary {sig_shape_dict} signal "
                "shapes must be identical"
            )

        dict_xmap = dictionary.xmap
        if dict_xmap is None or dict_xmap.shape != (dict_size,):
            raise ValueError(
                "Dictionary signal must have a non-empty `EBSD.xmap` attribute of equal"
                " size as the number of dictionary patterns, and both the signal and "
                "crystal map must have only one navigation dimension"
            )

        metric = self._prepare_metric(
            metric, navigation_mask, signal_mask, dtype, rechunk, dict_size
        )

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            xmap = _dictionary_indexing(
                experimental=self.data,
                experimental_nav_shape=am_exp.navigation_shape[::-1],
                dictionary=dictionary.data,
                step_sizes=tuple(a.scale for a in am_exp.navigation_axes[::-1]),
                dictionary_xmap=dictionary.xmap,
                keep_n=keep_n,
                n_per_iteration=n_per_iteration,
                metric=metric,
            )

        xmap.scan_unit = _get_navigation_axes_unit(am_exp)

        return xmap

    def spherical_indexing(
        self,
        harmonics: MasterPatternHarmonics | Sequence[MasterPatternHarmonics],
        detector: EBSDDetector,
        bandwidth: int = 68,
        n_best: int = 1,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        normalize: bool = True,
        refine: bool = True,
        n_regions: int = 10,
        gaussian_background: bool = False,
        circular_mask: bool = False,
        emsphinx_compatible: bool = True,
        chunksize: int | None = None,
        verbose: int = 1,
    ) -> CrystalMap:
        """Index patterns by spherical cross-correlation with one or
        more master patterns :cite:`lenthe2019spherical`.

        Each pattern is back-projected onto a square grid on the unit
        sphere, transformed to spherical harmonics and cross-correlated
        with every master pattern over all rotations. The best rotation
        of each phase is a candidate, and the best candidate is the
        indexed orientation.

        Parameters
        ----------
        harmonics
            One :class:`~kikuchipy.indexing.MasterPatternHarmonics` or
            a sequence of them, one per phase. Each must have its
            ``phase`` set, and the phase names must be unique.
        detector
            EBSD detector with exactly one projection center (PC),
            describing the detector-sample geometry of all patterns.
            Get one PC for a detector with several with
            ``detector = detector.deepcopy()`` followed by
            ``detector.pc = detector.pc_average``.
        bandwidth
            Bandwidth, i.e. the exclusive maximum harmonic degree, to
            index at, between 16 and 512. Default is 68. A bandwidth
            from :func:`~kikuchipy.indexing.fast_bandwidths` needs no
            zero padding and is therefore faster.
        n_best
            Number of candidates to keep per pattern. Default is 1.
            Each phase contributes exactly one candidate, so rows
            beyond the number of phases are filled with an invalid
            phase, see the ``Notes``.
        navigation_mask
            A boolean mask equal to the signal's navigation (map)
            shape, where only patterns equal to ``False`` are indexed.
            If not given, all patterns are indexed.
        signal_mask
            A boolean mask equal to the detector shape, where only
            pixels equal to ``False`` are used. If not given, all
            pixels are used.
        normalize
            Whether to use the normalized cross-correlation, which
            divides by a rotation dependent denominator computed from
            the sphere window. Default is ``True``.
        refine
            Whether to refine every phase's best orientation of the
            coarse Euler grid with Newton's method on the sphere.
            Default is ``True``, EMSphInx' own default. Pass
            ``False`` for the coarse grid orientations alone, which
            costs 5-21 % less time (measured refined to coarse
            ratios 1.05-1.27x) and gives scores which are not
            comparable with the refined ones, see the ``Notes``.
        n_regions
            Number of tiles along each detector axis of the adaptive
            histogram equalization applied to every pattern. Default
            is 10. ``0`` skips the equalization.
        gaussian_background
            Whether to fit and subtract a Gaussian background from
            every pattern first. Default is ``False``.
        circular_mask
            Whether to use only the largest circle inscribed in the
            detector. Default is ``False``.
        emsphinx_compatible
            Whether to reproduce EMSphInx' Gaussian fit off-by-one and
            the two defects of its peak interpolation. Default is
            ``True``, which EMSphInx parity requires, see the
            ``Notes``.
        chunksize
            Number of patterns to index per chunk. If not given, it is
            estimated from the bandwidth, the number of patterns and
            the number of Dask workers.
        verbose
            Which information to print. Options are 0 - no output,
            1 - information, progress bar and timing (default).

        Returns
        -------
        xmap
            A crystal map with the best orientation of each pattern,
            or the ``n_best`` best, and the properties ``"scores"``
            and ``"iq"``, the correlation and the image quality. With
            ``n_best`` greater than one, the property
            ``"nbest_phase_id"`` holds the phase of every candidate.

        Raises
        ------
        ValueError
            If the detector shape and the signal shape differ; if the
            signal has no navigation axis or more than two; if
            ``navigation_mask`` is not a boolean NumPy array of the
            navigation shape with at least one ``False`` entry; if a
            master pattern has no phase or two share a name; or for
            any error of
            :class:`~kikuchipy.indexing.SphericalIndexer`.

        Warns
        -----
        UserWarning
            If at least one pattern could not be indexed, since a
            failed pattern is never raised on, or if all Dask workers
            together are estimated to need more than 2 GiB.

        See Also
        --------
        dictionary_indexing
        hough_indexing
        refine_orientation
        refine_orientation_spherical
        kikuchipy.indexing.SphericalIndexer
        kikuchipy.indexing.MasterPatternHarmonics
        kikuchipy.indexing.fast_bandwidths

        Notes
        -----
        This is a port of EMSphInx' ``IndexEBSD`` program
        :cite:`lenthe2019spherical`, and a call with no optional
        parameters reproduces its name list defaults.

        **The scores are not normalized cross-correlations.** The
        ``"scores"`` property is the correlation of the winning
        rotation, which is never divided by the standard deviation of
        the pattern, so it is unbounded, comparable only within a
        fixed detector geometry and bandwidth, and **cannot** be used
        with
        :func:`~kikuchipy.indexing.orientation_similarity_map`.
        Measured for the nine shipped Ni patterns at a bandwidth of
        68 with the refined default: 0.51-0.63 normalized and
        0.29-0.36 un-normalized; with ``refine=False``: 0.50-0.62 and
        0.28-0.35.

        **A refined score is not a coarse score.** With ``refine``,
        the score is the *analytic* correlation at the refined
        rotation, divided by the rotation dependent window
        denominator when ``normalize`` is ``True``, while the coarse
        score is a tri-quadratic fitted to the grid around the
        maximum. The two are therefore not comparable with one
        another, and the refined normalized score can even dip
        slightly below the coarse one, because the Newton step
        maximizes the un-normalized correlation exactly as it does in
        EMSphInx (measured 4 of 165 points, worst -4.8e-4).

        A refinement adds no failure case of its own -- a Newton loop
        which does not converge returns the coarse orientation with
        the analytic correlation there -- but that correlation can be
        non-positive, and a candidate with a non-positive score is
        never recorded, so a pattern whose every phase failed that
        way becomes a failed pattern where the coarse path would have
        kept it. No such failure was observed on any real data run.

        **One candidate per phase.** ``n_best`` counts phases, not
        peaks: every phase contributes its single best rotation, so a
        row beyond the number of phases keeps the invalid phase
        ``-1``, the identity rotation and a score of ``0``. Only
        phases which win at least one point appear in
        ``xmap.phases``; the losing ones stay in
        ``SphericalIndexer.phases`` and in the ``"nbest_phase_id"``
        property. A rotated copy of a structure is indistinguishable
        from it by design, since the correlation peak does not change
        when the reference is rotated.

        **Failed patterns** keep the invalid phase ``-1``, the
        identity rotation, a score of ``0`` and an image quality of
        ``0``, so ``xmap.is_indexed`` is ``False`` exactly there. A
        pattern fails when it has no variance, when the preprocessing
        leaves it constant, when its best score or rotation is not
        finite, when it raises, or when no phase scored above zero.

        **The orientations are Newton refined** with the default
        ``refine=True``. Measured against the stored orientations of
        the shipped Ni data set at a bandwidth of 68: a median of
        0.51 and a maximum of 0.70 degrees, against 0.60 and 0.84 for
        the coarse orientations of ``refine=False``. The residual is
        systematic rather than a convergence limit: it is the mean
        projection center floor of that data set plus the
        band-limitation of the bandwidth, and a larger bandwidth
        shrinks it (median 0.45 at 88).

        **With ``refine=False`` the orientations are coarse.** They
        land within about half a grid cell of the Euler grid,
        ``180 / side_length`` degrees, which is 1.33 degrees at the
        default bandwidth and is
        ``SphericalIndexer.half_cell_degrees``.

        **Memory.** Each Dask worker holds one correlation kit, about
        21, 45, 98 and 207 MB at bandwidths 53, 68, 88 and 113 with
        one phase, and one more correlation cube per additional phase
        when ``normalize`` is ``True``. Refinement adds one Wigner
        table per correlator per worker, 5.0, 10.9 and 23.1 MB at
        bandwidths 68, 88 and 113, plus one 5.1 MB table shared by
        the whole process. The model these estimates come from is
        :attr:`~kikuchipy.indexing.SphericalIndexer.memory_per_worker_bytes`,
        and a warning is raised when all workers together would need
        more than 2 GiB.

        **EMSphInx parity** needs ``emsphinx_compatible=True`` both
        here and in
        :meth:`~kikuchipy.indexing.MasterPatternHarmonics.from_master_pattern`,
        since the master pattern normalization quirk is frozen into
        the coefficients when they are built.
        """
        am = self.axes_manager
        nav_shape = am.navigation_shape[::-1]
        sig_shape = am.signal_shape[::-1]

        if sig_shape != detector.shape:
            raise ValueError(
                f"The detector shape {detector.shape} and the signal's pattern "
                f"shape {sig_shape} must be identical"
            )

        # A crystal map is one- or two-dimensional. Without this a
        # navigation-less signal builds one from orix' default (5, 10)
        # coordinate arrays, so the returned map holds 50 coordinates
        # for its one point and raises on `xmap.shape`; a
        # three-dimensional one raises inside orix, but only after the
        # whole map has been indexed
        if len(nav_shape) not in (1, 2):
            raise ValueError(
                f"The signal's navigation shape {nav_shape} must have one or "
                "two dimensions, since a crystal map is one- or "
                "two-dimensional"
            )

        # Own checks first, in the frozen order: is-array, data type,
        # shape, all-`True`. The data type check must precede the
        # all-`True` one, since an integer mask of ones is all `True`
        # and would raise the wrong message; and it must follow the
        # is-array one, since a list has no data type
        if navigation_mask is not None:
            if not isinstance(navigation_mask, np.ndarray):
                raise ValueError("The navigation mask must be a NumPy array")
            elif navigation_mask.dtype != np.bool_:
                raise ValueError("The navigation mask must be a boolean array")
            elif navigation_mask.shape != nav_shape:
                raise ValueError(
                    f"The navigation mask shape {navigation_mask.shape} and the "
                    f"signal's navigation shape {nav_shape} must be identical"
                )
            elif navigation_mask.all():
                raise ValueError(
                    "The navigation mask must allow for indexing of at least one "
                    "pattern (at least one value equal to `False`)"
                )

        if isinstance(harmonics, MasterPatternHarmonics):
            harmonics_list = [harmonics]
        else:
            harmonics_list = list(harmonics)
        phase_names = []
        for i, harmonics_i in enumerate(harmonics_list):
            phase = getattr(harmonics_i, "phase", None)
            if phase is None:
                raise ValueError(
                    f"The master pattern harmonics at index {i} must have their "
                    "`MasterPatternHarmonics.phase` set, since the crystal map's "
                    "phase list is built from it"
                )
            phase_names.append(phase.name)
        if len(set(phase_names)) != len(phase_names):
            raise ValueError(
                f"The master pattern phase names {phase_names} must be unique"
            )

        # The materialized list, never the argument: a one-shot
        # iterator is exhausted by the checks above, and the indexer
        # would then refuse it as empty
        indexer = SphericalIndexer(
            harmonics_list,
            detector,
            bandwidth=bandwidth,
            normalize=normalize,
            refine=refine,
            signal_mask=signal_mask,
            n_regions=n_regions,
            gaussian_background=gaussian_background,
            circular_mask=circular_mask,
            emsphinx_compatible=emsphinx_compatible,
        )

        patterns = self.data.reshape((-1,) + sig_shape)
        n_all = int(np.prod(nav_shape))
        if navigation_mask is None:
            keep = np.ones(n_all, dtype=bool)
        else:
            keep = ~navigation_mask.ravel()
            patterns = patterns[keep]
        n_kept = int(keep.sum())

        if verbose >= 1:
            print(indexer.get_info_message(n_kept, chunksize))

        time_start = time()
        res = indexer.index_patterns(
            patterns, n_best=n_best, chunksize=chunksize, progressbar=verbose >= 1
        )
        total_time = time() - time_start

        if verbose >= 1:
            # Without this pause a part of the progress bar is
            # displayed below this print, as in dictionary indexing
            sleep(0.2)
            print(f"  Indexing speed: {n_kept / total_time:.5f} patterns/s")

        # Fill values of the result contract on every point which was
        # not indexed, deterministic where dictionary indexing leaves
        # uninitialized memory
        rotations = Rotation.identity((n_all, n_best))
        rotations[keep] = _rotation_from_zyz(res["zyz"]).data
        scores = np.zeros((n_all, n_best), dtype=np.float64)
        scores[keep] = res["scores"]
        phase_id = np.full((n_all, n_best), -1, dtype=np.int32)
        phase_id[keep] = res["phase_id"]
        iq = np.zeros(n_all, dtype=np.float64)
        iq[keep] = res["iq"]

        # ``scores`` first, as dictionary and Hough indexing list it
        prop = {}
        if n_best == 1:
            rotations = rotations.flatten()
            prop["scores"] = scores.squeeze(axis=1)
        else:
            prop["scores"] = scores
            # A crystal map holds one phase per point, so the phase of
            # every candidate goes into its own property
            prop["nbest_phase_id"] = phase_id
        prop["iq"] = iq

        step_sizes = tuple(a.scale for a in am.navigation_axes[::-1])
        xmap_kw, _ = create_coordinate_arrays(nav_shape, step_sizes)
        xmap_kw["rotations"] = rotations
        xmap_kw["phase_id"] = phase_id[:, 0]
        xmap_kw["prop"] = prop
        if navigation_mask is not None:
            xmap_kw["is_in_data"] = keep
        # orix deletes a phase whose identifier never occurs in
        # `phase_id`, so a losing phase is absent from `xmap.phases`;
        # the configured list stays on `SphericalIndexer.phases`
        phase_list = PhaseList([p.phase.deepcopy() for p in indexer.phases])
        xmap = CrystalMap(phase_list=phase_list, **xmap_kw)
        xmap.scan_unit = _get_navigation_axes_unit(am)

        return xmap

    def refine_orientation_spherical(
        self,
        xmap: CrystalMap,
        harmonics: MasterPatternHarmonics | Sequence[MasterPatternHarmonics],
        detector: EBSDDetector,
        bandwidth: int = 68,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        normalize: bool = True,
        n_regions: int = 10,
        gaussian_background: bool = False,
        circular_mask: bool = False,
        emsphinx_compatible: bool = True,
        chunksize: int | None = None,
        verbose: int = 1,
    ) -> CrystalMap:
        """Refine orientations by Newton's method on the sphere,
        without re-scanning the coarse rotation grid
        :cite:`lenthe2019spherical`.

        Every pattern is back-projected onto the unit sphere and
        transformed to spherical harmonics exactly as in
        :meth:`spherical_indexing`, and its stored orientation is then
        refined against the master pattern of its own phase.

        Parameters
        ----------
        xmap
            Crystal map with the orientations to refine, e.g. from
            :meth:`spherical_indexing`. It must have the same shape
            as the signal's navigation shape, and the phase of every
            point to refine must be the phase of the master pattern
            its phase index points at.
        harmonics
            One :class:`~kikuchipy.indexing.MasterPatternHarmonics` or
            a sequence of them, indexed by the crystal map's phase
            indices. Each must have its ``phase`` set.
        detector
            EBSD detector with exactly one projection center (PC),
            describing the detector-sample geometry of all patterns.
        bandwidth
            Bandwidth, i.e. the exclusive maximum harmonic degree, to
            refine at, between 16 and 512. Default is 68.
        navigation_mask
            A boolean mask equal to the signal's navigation (map)
            shape, where only orientations equal to ``False`` are
            refined. If not given, all indexed points in the map are
            refined. A point which is masked out here keeps its input
            row.
        signal_mask
            A boolean mask equal to the detector shape, where only
            pixels equal to ``False`` are used. If not given, all
            pixels are used.
        normalize
            Whether to use the normalized cross-correlation, which
            divides by a rotation dependent denominator computed from
            the sphere window. Default is ``True``.
        n_regions
            Number of tiles along each detector axis of the adaptive
            histogram equalization applied to every pattern. Default
            is 10. ``0`` skips the equalization.
        gaussian_background
            Whether to fit and subtract a Gaussian background from
            every pattern first. Default is ``False``.
        circular_mask
            Whether to use only the largest circle inscribed in the
            detector. Default is ``False``.
        emsphinx_compatible
            Whether to reproduce EMSphInx' Gaussian fit off-by-one.
            Default is ``True``.
        chunksize
            Number of patterns to refine per chunk. If not given, it
            is estimated from the bandwidth, the number of patterns
            and the number of Dask workers.
        verbose
            Which information to print. Options are 0 - no output,
            1 - information, progress bar and timing (default).

        Returns
        -------
        xmap_refined
            A crystal map with one refined orientation per point and
            the properties ``"scores"`` and ``"iq"``, the correlation
            at the refined orientation and the image quality. The
            phases, the phase of every point and which points are in
            the data are those of ``xmap``.

        Raises
        ------
        ValueError
            If the detector shape and the signal shape differ; if the
            signal has no navigation axis or more than two; if the
            crystal map shape and the signal's navigation shape
            differ; if ``navigation_mask`` is not a boolean NumPy
            array of the navigation shape with at least one ``False``
            entry; if a phase index of a point to refine does not
            index ``harmonics``; if the phase of a point to refine
            differs from the phase of that master pattern; if a
            master pattern has no phase or two share a name; or for
            any error of
            :class:`~kikuchipy.indexing.SphericalIndexer`.

        Warns
        -----
        UserWarning
            If all Dask workers together are estimated to need more
            than 2 GiB, or if any point to refine could not be
            refined and keeps its input row.

        See Also
        --------
        spherical_indexing
        refine_orientation
        kikuchipy.indexing.SphericalIndexer

        Notes
        -----
        This is the refinement-only work item of EMSphInx' EBSD
        driver, with one deliberate deviation: the shipped C++
        discards the result of its own refinement call and stores a
        score its refinement-only branch never assigned, so it hands
        back the *unrefined* orientation with a zero or stale score.
        This method implements the documented intent instead -- it
        refines the stored orientation and stores the refined score.

        **Refinement is local.** Newton's method converges to the
        nearest stationary point of the correlation, so this method
        is score-monotone only for orientations which came from
        :meth:`spherical_indexing` with the same configuration. An
        orientation from another source -- Hough indexing, dictionary
        indexing, another geometry -- may converge to a stationary
        point whose score is *lower* than the input's, or fail to
        converge and keep its input orientation with the correlation
        there.

        Refining a coarse map from :meth:`spherical_indexing` equals
        ``spherical_indexing(refine=True)`` to tolerance rather than
        bitwise, because the stored quaternion hands back the
        equivalent Euler triple with the opposite ``beta`` sign, whose
        refinement walks the mirrored Wigner table path. Measured over
        the nine shipped Ni patterns: a misorientation of 0.0 degrees
        and a score difference of at most 2.9e-14.

        **A crystal map whose points are not a full rectangle is
        refused.** A map from a navigation-masked
        :meth:`spherical_indexing` call has a shape derived from the
        bounding box of the points which are in the data, so it fails
        the shape check every refinement method of kikuchipy applies.
        Index the full map instead and mask at refinement time with
        this method's own ``navigation_mask``.

        **A point which is not refined keeps its input row**: a point
        which is not indexed, one masked out by ``navigation_mask``,
        and one whose pattern fails a guard or raises. Only the
        rotation, ``"scores"`` and ``"iq"`` are carried over from the
        input map, so a map with more than one orientation per point
        comes back with the first one alone, as
        :meth:`refine_orientation` returns one.

        Two conventions deviate from
        :meth:`refine_orientation` and
        :meth:`refine_orientation_projection_center` on purpose: the
        parameters read ``(xmap, harmonics, detector, ...)``, mirroring
        :meth:`spherical_indexing`, where the siblings read
        ``(xmap, detector, master_pattern, ...)``; and there is no
        ``compute``, ``rechunk`` or ``chunk_kwargs``, because the
        spherical pipeline is eager, again as
        :meth:`spherical_indexing` is. ``chunksize`` is the knob
        offered instead.
        """
        am = self.axes_manager
        nav_shape = am.navigation_shape[::-1]
        sig_shape = am.signal_shape[::-1]

        if sig_shape != detector.shape:
            raise ValueError(
                f"The detector shape {detector.shape} and the signal's pattern "
                f"shape {sig_shape} must be identical"
            )

        if len(nav_shape) not in (1, 2):
            raise ValueError(
                f"The signal's navigation shape {nav_shape} must have one or "
                "two dimensions, since a crystal map is one- or "
                "two-dimensional"
            )

        # The same check and message every kikuchipy refinement
        # applies, which also refuses a map whose points in the data
        # are not the full navigation grid, see the `Notes`
        _xmap_is_compatible_with_signal(
            xmap, am.navigation_axes[::-1], raise_if_not=True
        )

        # Own checks in the frozen order of `spherical_indexing`
        if navigation_mask is not None:
            if not isinstance(navigation_mask, np.ndarray):
                raise ValueError("The navigation mask must be a NumPy array")
            elif navigation_mask.dtype != np.bool_:
                raise ValueError("The navigation mask must be a boolean array")
            elif navigation_mask.shape != nav_shape:
                raise ValueError(
                    f"The navigation mask shape {navigation_mask.shape} and the "
                    f"signal's navigation shape {nav_shape} must be identical"
                )
            elif navigation_mask.all():
                raise ValueError(
                    "The navigation mask must allow for refinement of at least "
                    "one orientation (at least one value equal to `False`)"
                )

        if isinstance(harmonics, MasterPatternHarmonics):
            harmonics_list = [harmonics]
        else:
            harmonics_list = list(harmonics)
        phase_names = []
        for i, harmonics_i in enumerate(harmonics_list):
            phase = getattr(harmonics_i, "phase", None)
            if phase is None:
                raise ValueError(
                    f"The master pattern harmonics at index {i} must have their "
                    "`MasterPatternHarmonics.phase` set, since the crystal map's "
                    "phase list is built from it"
                )
            phase_names.append(phase.name)
        if len(set(phase_names)) != len(phase_names):
            raise ValueError(
                f"The master pattern phase names {phase_names} must be unique"
            )

        # Full length rows scattered through `is_in_data`: orix hands
        # back the points in the data alone, so the map's own arrays
        # are shorter than the navigation grid whenever a point is
        # out of it
        n_all = int(np.prod(nav_shape))
        in_data = np.asarray(xmap.is_in_data, dtype=bool).ravel()
        rotations_in_data = xmap.rotations
        if rotations_in_data.ndim > 1:
            # only the first candidate is refined, as the C++ refine
            # only work item "currently only uses a single result"
            rotations_in_data = rotations_in_data[:, 0]
        rotation_data = np.zeros((n_all, 4), dtype=np.float64)
        rotation_data[:, 0] = 1.0
        rotation_data[in_data] = rotations_in_data.data
        zyz_full = np.zeros((n_all, 3), dtype=np.float64)
        zyz_full[in_data] = _rotation_to_zyz(rotations_in_data)
        phase_id_full = np.full(n_all, -1, dtype=np.int32)
        phase_id_full[in_data] = xmap.phase_id
        is_indexed_full = np.zeros(n_all, dtype=bool)
        is_indexed_full[in_data] = xmap.is_indexed

        points_to_refine = in_data & is_indexed_full
        if navigation_mask is not None:
            points_to_refine = points_to_refine & ~navigation_mask.ravel()

        # A phase index which does not name a master pattern, and one
        # which names the wrong one: an index which merely happens to
        # be in range must not silently refine against another phase
        ids = np.unique(phase_id_full[points_to_refine])
        for phase_id_i in ids.tolist():
            if phase_id_i >= len(harmonics_list):
                raise ValueError(
                    f"The phase index {phase_id_i} of the points to refine must "
                    f"index the {len(harmonics_list)} master pattern(s) given, "
                    "i.e. be smaller than that"
                )
            xmap_phase = xmap.phases[phase_id_i]
            mp_phase = harmonics_list[phase_id_i].phase
            equal_phases, are_different = _equal_phase(mp_phase, xmap_phase)
            if not equal_phases:
                raise ValueError(
                    f"Master pattern phase '{mp_phase.name}' and phase of points "
                    f"to refine in crystal map '{xmap_phase.name}' must be the "
                    f"same, but have different {are_different}"
                )

        indexer = SphericalIndexer(
            harmonics_list,
            detector,
            bandwidth=bandwidth,
            normalize=normalize,
            refine=True,
            signal_mask=signal_mask,
            n_regions=n_regions,
            gaussian_background=gaussian_background,
            circular_mask=circular_mask,
            emsphinx_compatible=emsphinx_compatible,
        )

        patterns = self.data.reshape((-1,) + sig_shape)[points_to_refine]
        n_kept = int(points_to_refine.sum())

        if verbose >= 1:
            print(indexer.get_info_message(n_kept, chunksize, refining=True))

        time_start = time()
        res = indexer.refine_patterns(
            patterns,
            zyz_full[points_to_refine],
            phase_id_full[points_to_refine],
            chunksize=chunksize,
            progressbar=verbose >= 1,
        )
        total_time = time() - time_start

        if verbose >= 1:
            # Without this pause a part of the progress bar is
            # displayed below this print, as in dictionary indexing
            sleep(0.2)
            print(f"  Refinement speed: {n_kept / total_time:.5f} patterns/s")

        # A point which was not refined keeps its input row, so the
        # scattering starts from the input map's own values.  A
        # pattern which failed a guard or raised carries its starting
        # angles back with a score and an image quality of exactly
        # zero, which is how the refinement reports that it wrote
        # nothing.  This re-derives the flag from the two columns
        # rather than reading one: a refined row whose score *and*
        # image quality are both exactly zero would be read as "not
        # written", which needs the back-projection to answer a
        # non-constant pattern with an all-zero spectrum and the
        # correlation to land on a float64 zero at the same point
        was_refined = np.zeros(n_all, dtype=bool)
        was_refined[points_to_refine] = (res["scores"] != 0.0) | (res["iq"] != 0.0)
        rotation_data[was_refined] = _rotation_from_zyz(
            res["zyz"][was_refined[points_to_refine]]
        ).data
        scores = np.zeros(n_all, dtype=np.float64)
        iq = np.zeros(n_all, dtype=np.float64)
        for name, array in (("scores", scores), ("iq", iq)):
            if name in xmap.prop:
                # never ``prop.get()``: only orix' own ``__getitem__``
                # applies ``is_in_data``, so ``get()`` returns the
                # full length column of a partially in-data map
                stored = np.asarray(xmap.prop[name], dtype=np.float64)
                if stored.ndim > 1:
                    stored = stored[:, 0]
                array[in_data] = stored
        scores[was_refined] = res["scores"][was_refined[points_to_refine]]
        iq[was_refined] = res["iq"][was_refined[points_to_refine]]

        step_sizes = tuple(a.scale for a in am.navigation_axes[::-1])
        xmap_kw, _ = create_coordinate_arrays(nav_shape, step_sizes)
        xmap_kw["rotations"] = Rotation(rotation_data)
        xmap_kw["phase_id"] = phase_id_full
        # `scores` first, as dictionary and Hough indexing list it
        xmap_kw["prop"] = {"scores": scores, "iq": iq}
        if not in_data.all():
            xmap_kw["is_in_data"] = in_data
        xmap_refined = CrystalMap(phase_list=xmap.phases.deepcopy(), **xmap_kw)
        xmap_refined.scan_unit = xmap.scan_unit

        return xmap_refined

    def refine_orientation(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: int | float,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        pseudo_symmetry_ops: Rotation | None = None,
        method: str | None = "minimize",
        method_kwargs: dict | None = None,
        trust_region: tuple | list | np.ndarray | None = None,
        initial_step: float | None = None,
        rtol: float = 1e-4,
        maxeval: int | None = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: dict | None = None,
    ) -> CrystalMap | da.Array:
        r"""Refine orientations by searching orientation space around
        the best indexed solution using fixed projection centers.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The similarity metric used is the normalized
        cross-correlation (NCC). The orientation, represented by a
        Euler angle triplet (:math:`\phi_1`, :math:`\Phi`,
        :math:`\phi_2`) relative to the EDAX TSL sample reference frame
        RD-TD-ND, is optimized during refinement, while the
        sample-detector geometry, represented by the three projection
        center (PC) parameters (PCx, PCy, PCz) in the Bruker convention,
        is fixed.

        A subset of the optimization methods in *SciPy* and *NLopt* are
        available:

        - Local optimization:
            - :func:`~scipy.optimize.minimize` (includes Nelder-Mead,
              Powell etc.).
            - Nelder-Mead via `nlopt.LN_NELDERMEAD
              <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#nelder-mead-simplex>`_
        - Global optimization:
            - :func:`~scipy.optimize.differential_evolution`
            - :func:`~scipy.optimize.dual_annealing`
            - :func:`~scipy.optimize.basinhopping`
            - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Crystal map with points to refine. Only the points in the
            data (see :class:`~orix.crystal_map.CrystalMap`) are
            refined. If a ``navigation_mask`` is given, points equal to
            points in the data and points equal to ``False`` in this
            mask are refined.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        navigation_mask
            A boolean mask of points in the crystal map to refine (equal
            to ``False``, i.e. points to *mask out* are ``True``). The
            mask must be of equal shape to the signal's navigation
            shape. If not given, all points in the crystal map data are
            refined.
        signal_mask
            A boolean mask of detector pixels to use in refinement
            (equal to ``False``, i.e. pixels to *mask out* are
            ``True``). The mask must be of equal shape to the signal's
            signal shape. If not given, all pixels are used.
        pseudo_symmetry_ops
            Pseudo-symmetry operators as rotations. If given, each
            map point will be refined using the map orientation and the
            orientation after applying each operator. The chosen
            solution is the one with the highest score. E.g. if two
            operators are given, each map point is refined three times.
            If given, the returned crystal map will have a property
            array with the operator index giving the best score, with 0
            meaning the original map point gave the best score.
        method
            Name of the :mod:`scipy.optimize` or *NLopt* optimization
            method, among ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, ``"shgo"`` and
            ``"ln_neldermead"`` (from *NLopt*). Default is
            ``"minimize"``, which by default performs local optimization
            with the Nelder-Mead method, unless another ``"minimize"``
            method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm from *SciPy*, pass
            ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``. Not used if
            ``method="LN_NELDERMEAD"``.
        trust_region
            List of three +/- angular deviations in degrees used to
            determine the bound constraints on the three Euler angles
            per navigation point, e.g. ``[2, 2, 2]``. Not passed to
            *SciPy* ``method`` if it does not support bounds. The
            definition ranges of the Euler angles are
            :math:`\phi_1 \in [0, 360]`, :math:`\Phi \in [0, 180]` and
            :math:`\phi_2 \in [0, 360]` in radians.
        initial_step
            A single initial step size for all Euler angle, in degrees.
            Only used if ``method="LN_NELDERMEAD"``. If not given, this
            is not set for the *NLopt* optimizer.
        rtol
            Stop optimization of a pattern when the difference in NCC
            score between two iterations is below this value (relative
            tolerance). Default is ``1e-4``. Only used if
            ``method="LN_NELDERMEAD"``.
        maxeval
            Stop optimization of a pattern when the number of function
            evaluations exceeds this value, e.g. ``100``. Only used if
            ``method="LN_NELDERMEAD"``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            If ``compute=True``, a crystal map with refined
            orientations, NCC scores in a ``"scores"`` property, the
            number of function evaluations in a ``"num_evals"``
            property and which pseudo-symmetry operator gave the best
            score if ``pseudo_symmetry_ops`` is given is returned.. If
            ``compute=False``, a dask array of navigation size + (5,)
            (or (6,) if ``pseudo_symmetry_ops`` is passed) is returned, to be
            computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_results`.
            Each navigation point in the data has the optimized score,
            the number of function evaluations, the three Euler angles
            in radians and potentially the pseudo-symmetry operator
            index in element 0, 1, 2, 3, 4 and 5, respectively.

        See Also
        --------
        scipy.optimize, refine_orientation_spherical,
        refine_projection_center, refine_orientation_projection_center

        Notes
        -----
        *NLopt* is for now an optional dependency, see
        :ref:`dependencies` for details. Be aware that *NLopt*
        does not fail gracefully. If continued use of *NLopt* proves
        stable enough, its implementation of the Nelder-Mead algorithm
        might become the default.
        """
        points_to_refine = self._check_refinement_parameters(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            rechunk=rechunk,
            chunk_kwargs=chunk_kwargs,
        )
        return _refine_orientation(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            trust_region=trust_region,
            rtol=rtol,
            pseudo_symmetry_ops=pseudo_symmetry_ops,
            method=method,
            method_kwargs=method_kwargs,
            initial_step=initial_step,
            maxeval=maxeval,
            compute=compute,
            navigation_mask=navigation_mask,
        )

    def refine_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: int | float,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        method: str | None = "minimize",
        method_kwargs: dict | None = None,
        trust_region: tuple | list | np.ndarray | None = None,
        initial_step: float | None = None,
        rtol: float = 1e-4,
        maxeval: int | None = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, EBSDDetector, np.ndarray] | da.Array:
        """Refine projection centers by searching the parameter space
        using fixed orientations.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The similarity metric used is the normalized
        cross-correlation (NCC). The sample-detector geometry,
        represented by the three projection center (PC) parameters
        (PCx, PCy, PCz) in the Bruker convention, is updated during
        refinement, while the orientations, defined relative to the EDAX
        TSL sample reference frame RD-TD-ND, are fixed.

        A subset of the optimization methods in *SciPy* and *NLopt* are
        available:

        - Local optimization:
            - :func:`~scipy.optimize.minimize` (includes Nelder-Mead,
              Powell etc.).
            - Nelder-Mead via `nlopt.LN_NELDERMEAD
              <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#nelder-mead-simplex>`_
        - Global optimization:
            - :func:`~scipy.optimize.differential_evolution`
            - :func:`~scipy.optimize.dual_annealing`
            - :func:`~scipy.optimize.basinhopping`
            - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Crystal map with points to use in refinement. Only the
            points in the data
            (see :class:`~orix.crystal_map.CrystalMap`) are used. If a
            ``navigation_mask`` is given, points equal to points in the
            data and points equal to ``False`` in this mask are used.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
            Which PCs are refined depend on ``xmap`` and
            ``navigation_mask``.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        navigation_mask
            A boolean mask of points in the crystal map to use in
            refinement (equal to ``False``, i.e. points to *mask out*
            are ``True``). The mask must be of equal shape to the
            signal's navigation shape. If not given, all points in
            the crystal map data are used.
        signal_mask
            A boolean mask of detector pixels to use in refinement
            (equal to ``False``, i.e. pixels to *mask out* are
            ``True``). The mask must be of equal shape to the signal's
            signal shape. If not given, all pixels are used.
        method
            Name of the :mod:`scipy.optimize` or *NLopt* optimization
            method, among ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, ``"shgo"`` and
            ``"ln_neldermead"`` (from *NLopt*). Default is
            ``"minimize"``, which by default performs local optimization
            with the Nelder-Mead method, unless another ``"minimize"``
            method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm from *SciPy*, pass
            ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``. Not used if
            ``method="LN_NELDERMEAD"``.
        trust_region
            List of three +/- deviations in the range [0, 1] used to
            determine the bounds constraints on the PC parameters per
            navigation point, e.g. ``[0.05, 0.05, 0.05]``. Not passed to
            *SciPy* ``method`` if it does not support bounds. The
            definition range of the PC parameters are assumed to be
            [-2, 2].
        initial_step
            A single initial step size for all PC parameters in the
            range [0, 1]. Only used if ``method="LN_NELDERMEAD"``.
        rtol
            Stop optimization of a pattern when the difference in NCC
            score between two iterations is below this value (relative
            tolerance). Default is ``1e-4``. Only used if
            ``method="LN_NELDERMEAD"``.
        maxeval
            Stop optimization of a pattern when the number of function
            evaluations exceeds this value, e.g. ``100``. Only used if
            ``method="LN_NELDERMEAD"``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            New similarity metrics, a new EBSD detector instance with
            the refined PCs and the number of function evaluations if
            ``compute=True``. If ``compute=False``, a dask array of
            navigation size + (5,) is returned, to be computed later.
            See
            :func:`~kikuchipy.indexing.compute_refine_projection_center_results`.
            Each navigation point has the optimized score, the three
            PC parameters in the Bruker convention and the number of
            function evaluations in element 0, 1, 2, 3 and 4,
            respectively.

        See Also
        --------
        scipy.optimize, refine_orientation,
        refine_orientation_projection_center

        Notes
        -----
        If the crystal map to refine contains points marked as not
        indexed, the returned detector might not have a 2D navigation
        shape.

        *NLopt* is for now an optional dependency, see
        :ref:`dependencies` for details. Be aware that *NLopt*
        does not fail gracefully. If continued use of *NLopt* proves
        stable enough, its implementation of the Nelder-Mead algorithm
        might become the default.
        """
        points_to_refine = self._check_refinement_parameters(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            rechunk=rechunk,
            chunk_kwargs=chunk_kwargs,
        )
        return _refine_pc(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            maxeval=maxeval,
            compute=compute,
            navigation_mask=navigation_mask,
        )

    def refine_orientation_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: int | float,
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
        pseudo_symmetry_ops: Rotation | None = None,
        method: str | None = "minimize",
        method_kwargs: dict | None = None,
        trust_region: tuple | list | np.ndarray | None = None,
        initial_step: tuple | list | np.ndarray | None = None,
        rtol: float | None = 1e-4,
        maxeval: int | None = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: dict | None = None,
    ) -> tuple[CrystalMap, EBSDDetector] | da.Array:
        r"""Refine orientations and projection centers simultaneously by
        searching the orientation and PC parameter space.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The only supported similarity metric is the normalized
        cross-correlation (NCC). The orientation, represented by a
        Euler angle triplet (:math:`\phi_1`, :math:`\Phi`,
        :math:`\phi_2`) relative to the EDAX TSL sample reference frame
        RD-TD-ND, is optimized during refinement, while the
        sample-detector geometry, represented by the three projection
        center (PC) parameters (PCx, PCy, PCz) in the Bruker convention,
        is fixed.

        A subset of the optimization methods in *SciPy* and *NLopt* are
        available:

        - Local optimization:
            - :func:`~scipy.optimize.minimize` (includes Nelder-Mead,
              Powell etc.).
            - Nelder-Mead via `nlopt.LN_NELDERMEAD
              <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#nelder-mead-simplex>`_
        - Global optimization:
            - :func:`~scipy.optimize.differential_evolution`
            - :func:`~scipy.optimize.dual_annealing`
            - :func:`~scipy.optimize.basinhopping`
            - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Crystal map with points to refine. Only the points in the
            data (see :class:`~orix.crystal_map.CrystalMap`) are
            refined. If a ``navigation_mask`` is given, points equal to
            points in the data and points equal to ``False`` in this
            mask are refined.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
            Which PCs are refined depend on ``xmap`` and
            ``navigation_mask``.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        navigation_mask
            A boolean mask of points in the crystal map to refine (equal
            to ``False``, i.e. points to *mask out* are ``True``). The
            mask must be of equal shape to the signal's navigation
            shape. If not given, all points in the crystal map data are
            refined.
        signal_mask
            A boolean mask of detector pixels to use in refinement
            (equal to ``False``, i.e. pixels to *mask out* are
            ``True``). The mask must be of equal shape to the signal's
            signal shape. If not given, all pixels are used.
        pseudo_symmetry_ops
            Pseudo-symmetry operators as rotations. If given, each
            map point will be refined using the map orientation and the
            orientation after applying each operator. The chosen
            solution is the one with the highest score. E.g. if two
            operators are given, each map point is refined three times.
            If given, the returned crystal map will have a property
            array with the operator index giving the best score, with 0
            meaning the original map point gave the best score.
        method
            Name of the :mod:`scipy.optimize` or *NLopt* optimization
            method, among ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, ``"shgo"`` and
            ``"ln_neldermead"`` (from *NLopt*). Default is
            ``"minimize"``, which by default performs local optimization
            with the Nelder-Mead method, unless another ``"minimize"``
            method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm from *SciPy*, pass
            ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``. Not used if
            ``method="LN_NELDERMEAD"``.
        trust_region
            List of three +/- angular deviations in degrees as bound
            constraints on the three Euler angles and three +/-
            deviations in the range [0, 1] as bound constraints on the
            PC parameters, e.g. ``[2, 2, 2, 0.05, 0.05, 0.05]``. Not
            passed to *SciPy* ``method`` if it does not support bounds.
            The definition ranges of the Euler angles are
            :math:`\phi_1 \in [0, 360]`, :math:`\Phi \in [0, 180]` and
            :math:`\phi_2 \in [0, 360]` in radians, while the definition
            range of the PC parameters are assumed to be [-2, 2].
        initial_step
            A list of two initial step sizes to use, one in degrees for
            all Euler angles and one in the range [0, 1] for all PC
            parameters. Only used if ``method="LN_NELDERMEAD"``.
        rtol
            Stop optimization of a pattern when the difference in NCC
            score between two iterations is below this value (relative
            tolerance). Only used if ``method="LN_NELDERMEAD"``. If not
            given, this is set to ``1e-4``.
        maxeval
            Stop optimization of a pattern when the number of function
            evaluations exceeds this value, e.g. ``100``. Only used if
            ``method="LN_NELDERMEAD"``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            If ``compute=True``, a crystal map with refined
            orientations, NCC scores in a ``"scores"`` property, the
            number of function evaluations in a ``"num_evals"``
            property and which pseudo-symmetry operator gave the best
            score if ``pseudo_symmetry_ops`` is given is returned, as
            well as a new EBSD detector with the refined PCs. If
            ``compute=False``, a dask array of navigation size + (8,)
            (or (9,) if ``pseudo_symmetry_ops`` is passed) is returned,
            to be computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_projection_center_results`.
            Each navigation point in the data has the score, the number
            of function evaluations, the three Euler angles in radians,
            the three PC parameters and potentially the pseudo-symmetry
            operator index in element 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9,
            respectively.

        See Also
        --------
        scipy.optimize, refine_orientation, refine_projection_center

        Notes
        -----
        If the crystal map to refine contains points marked as not
        indexed, the returned detector might not have a 2D navigation
        shape.

        The method attempts to refine the orientations and projection
        center at the same time for each map point. The optimization
        landscape is sloppy :cite:`pang2020global`, where the
        orientation and PC can make up for each other. Thus, it is
        possible that the parameters that yield the highest similarity
        are incorrect. As always, it is left to the user to ensure that
        the output is reasonable.

        *NLopt* is for now an optional dependency, see
        :ref:`dependencies` for details. Be aware that *NLopt*
        does not fail gracefully. If continued use of *NLopt* proves
        stable enough, its implementation of the Nelder-Mead algorithm
        might become the default.
        """
        points_to_refine = self._check_refinement_parameters(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            navigation_mask=navigation_mask,
            signal_mask=signal_mask,
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            rechunk=rechunk,
            chunk_kwargs=chunk_kwargs,
        )
        return _refine_orientation_pc(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            points_to_refine=points_to_refine,
            signal_mask=signal_mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            pseudo_symmetry_ops=pseudo_symmetry_ops,
            maxeval=maxeval,
            compute=compute,
            navigation_mask=navigation_mask,
        )

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #

    def save(
        self,
        filename: Path | str | None = None,
        overwrite: bool | None = None,
        extension: str | None = None,
        **kwargs,
    ) -> None:
        """Write the signal to file in the specified format.

        If no extension is given, the signal is written to a file in
        kikuchipy's h5ebsd format.

        Parameters
        ----------
        filename
            If not given and 'tmp_parameters.filename' and
            'tmp_parameters.folder' are defined in signal metadata, the
            filename and path will be taken from there.
        overwrite
            If not given and the file exists, it will query the user. If
            True (False) it (does not) overwrite the file if it exists.
        extension
            Extension of the file that defines the file format. Options
            are:

            * h5, hdf5, or h5ebsd: kikuchipy's specification of the
              h5ebsd format
            * dat: NORDIF's binary format
            * hspy: HyperSpy's HDF5 format
            * zspy: HyperSpy's zarr format

            Each format accepts different parameters.

            If not given, the extension is determined from the following
            list in this order: i) the filename, ii)
            'tmp_parameters.extension' or iii) 'h5' (kikuchipy's h5ebsd
            format).
        **kwargs
            Keyword arguments passed to the corresponding writer.

        See Also
        --------
        kikuchipy.io.plugins

        Notes
        -----
        This method is a modified version of HyperSpy's function
        :meth:`hyperspy.signal.BaseSignal.save`.
        """
        if filename is not None:
            fname = Path(filename)
        else:
            tmp_params = self.tmp_parameters
            if tmp_params.has_item("filename") and tmp_params.has_item("folder"):
                fname = Path(tmp_params.folder) / tmp_params.filename
                extension = tmp_params.extension if not extension else extension
            elif self.metadata.has_item("General.original_filename"):
                fname = Path(self.metadata.General.original_filename)
            else:
                raise ValueError("filename not given")
        if extension is not None:
            ext = extension
        else:
            ext = fname.suffix[1:]
        if ext.lower() in ["hspy", "zspy"]:
            super().save(filename, overwrite=overwrite, **kwargs)
            return
        _save(fname, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(
        self,
        components: int | list[int] | None = None,
        dtype_out: str | np.dtype | type = "float32",
    ) -> EBSD | LazyEBSD:
        """Get the model signal generated with the selected number of
        principal components from a decomposition.

        Calls HyperSpy's
        :meth:`~hyperspy.learn.mva.MVA.get_decomposition_model`.
        Learning results are preconditioned before this call, doing the
        following:

        1. Set :class:`numpy.dtype` to desired ``dtype_out``.
        2. Remove unwanted components.
        3. Rechunk to suitable chunks if :class:`~dask.array.Array`.

        Parameters
        ----------
        components
            If not given, rebuilds the signal from all components.
            If ``int``, rebuilds signal from ``components`` in range
            0-given ``int``. If list of ``ints``, rebuilds signal from
            only ``components`` in given list.
        dtype_out
            Data type to cast learning results to (default is
            ``"float32``). Note that HyperSpy casts to ``"float64"``.

        Returns
        -------
        s_model
            Model signal.
        """
        # Keep original results to revert back after updating
        factors_orig = self.learning_results.factors.copy()
        loadings_orig = self.learning_results.loadings.copy()

        # Change data type, keep desired components and rechunk if lazy
        dtype_out = np.dtype(dtype_out)
        (
            self.learning_results.factors,
            self.learning_results.loadings,
        ) = _update_learning_results(
            learning_results=self.learning_results,
            dtype_out=dtype_out,
            components=components,
        )

        # Call HyperSpy's function
        s_model = super().get_decomposition_model()

        # Revert learning results to original results
        self.learning_results.factors = factors_orig
        self.learning_results.loadings = loadings_orig

        # Remove learning results from model signal
        s_model.learning_results = LearningResults()

        return s_model

    @insert_doc_disclaimer(cls=Signal2D, meth=Signal2D.crop)
    def crop(self, *args, **kwargs):
        # This method is called by crop_signal(), so attributes are
        # handled correctly by that method as well

        old_shape = self.data.shape

        super().crop(*args, **kwargs)

        # Get input parameters
        params = get_parameters(super().crop, ["start", "end"], args, kwargs)

        # End if not all parameters of interest could be found or if
        # there is nothing to do
        if params is None or all(p is None for p in params.values()):
            return

        # Determine which data axis changed
        new_shape = self.data.shape
        diff_data = np.array(old_shape) - np.array(new_shape)
        if not diff_data.any():
            return
        idx_data = np.atleast_1d(diff_data).nonzero()[0][0]

        am = self.axes_manager
        nav_ndim = am.navigation_dimension

        # Update attributes
        attrs = self._get_custom_attributes()
        if idx_data in am.signal_indices_in_array:
            # Slice static background and update detector shape...
            sig_slices = 2 * [slice(None)]
            sig_slices[idx_data - nav_ndim] = slice(params["start"], params["end"])
            sig_slices = tuple(sig_slices)
            # TODO: Update PC values
            attrs = _update_custom_attributes(
                attrs, sig_slices=sig_slices, new_sig_shape=am.signal_shape[::-1]
            )
        else:
            # ... or slice crystal map and detector PC values
            nav_slices = nav_ndim * [slice(None)]
            nav_slices[idx_data] = slice(params["start"], params["end"])
            nav_slices = tuple(nav_slices)
            attrs = _update_custom_attributes(attrs, nav_slices=nav_slices)

        self._set_custom_attributes(attrs)

    @insert_doc_disclaimer(cls=Signal2D, meth=Signal2D.rebin)
    def rebin(self, *args, **kwargs):
        data_shape_old = self.data.shape
        am_old = self.axes_manager
        nav_shape_old = am_old.navigation_shape[::-1]
        sig_shape_old = am_old.signal_shape[::-1]

        new = super().rebin(*args, **kwargs)

        if new is None:
            return
        elif new.data.shape == data_shape_old:
            return new

        # Get input parameters
        params = get_parameters(
            super().rebin, ["new_shape", "scale", "crop", "dtype"], args, kwargs
        )

        am_new = new.axes_manager
        nav_shape_new = am_new.navigation_shape[::-1]
        sig_shape_new = am_new.signal_shape[::-1]

        attrs = self._get_custom_attributes(make_deepcopy=True)

        # Update static background
        static_bg = attrs["static_background"]
        if sig_shape_new != sig_shape_old and static_bg is not None:
            sig_idx = am_old.signal_indices_in_array[::-1]
            if params["new_shape"] is not None:
                params["new_shape"] = [params["new_shape"][i] for i in sig_idx]
            else:
                params["scale"] = [params["scale"][i] for i in sig_idx]
            s_static_bg = hs.signals.Signal2D(static_bg)
            s_static_bg2 = s_static_bg.rebin(**params)
            static_bg2 = s_static_bg2.data
            static_bg2 = static_bg2.astype(new.data.dtype)
            attrs["static_background"] = static_bg2

        # Update detector shape and binning factor
        detector: EBSDDetector = attrs["detector"]
        detector.shape = sig_shape_new
        factors = np.array(sig_shape_old) / np.array(sig_shape_new)
        binning2 = detector.binning * factors
        if binning2[0] == binning2[1] and np.allclose(binning2, binning2.round(0)):
            detector.binning = float(binning2[0])
        else:
            detector.binning = 1.0

        if nav_shape_new != nav_shape_old:
            attrs["xmap"] = None
            detector.pc = np.full(nav_shape_new + (3,), 0.5)
            attrs["static_background"] = None

        new._set_custom_attributes(attrs)

        return new

    def _slicer(self, *args, **kwargs):
        # This method is called by inav and isig via FancySlicing

        # Get input parameters
        params = get_parameters(
            super()._slicer, ["slices", "isNavigation"], args, kwargs
        )

        # Get slices prior to call
        all_slices = self._get_array_slices(params["slices"], params["isNavigation"])
        nav_ndim = self.axes_manager.navigation_dimension
        if params["isNavigation"]:
            slices = all_slices[:nav_ndim]
        else:
            slices = all_slices[nav_ndim:]

        props = self._get_custom_attributes(make_deepcopy=True)

        new = super()._slicer(*args, **kwargs)

        if not isinstance(new, EBSD):
            # Can be Signal1D when slicing the signal shape into a 1D
            # array
            return new

        # Update attributes
        if new is None:  # pragma: no cover
            new_nav_shape = self._navigation_shape_rc
            new_sig_shape = self._signal_shape_rc
        else:
            new_nav_shape = new._navigation_shape_rc
            new_sig_shape = new._signal_shape_rc
        if params["isNavigation"]:
            props = _update_custom_attributes(
                props, nav_slices=slices, new_nav_shape=new_nav_shape
            )
        else:
            props = _update_custom_attributes(
                props, sig_slices=slices, new_sig_shape=new_sig_shape
            )

        if new is None:  # pragma: no cover
            self._set_custom_attributes(props)
        else:
            new._set_custom_attributes(props)

        return new

    # ------------------------ Private methods ----------------------- #

    def _check_refinement_parameters(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        navigation_mask: np.ndarray | None = None,
        signal_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Check compatibility of refinement parameters with refinement.

        No checks of the parameters should be necessary after this
        function runs successfully.

        Checks of the refinement algorithm and parameters used in this
        algorithm are done in the refinement setup.

        Parameters
        ----------
        xmap
            Crystal map with rotation(s) to refine. Its shape must be
            equal to the signal's navigation shape, and the points to
            refine must contain only one phase, equal to the master
            pattern phase. Only points which are ``True`` in the map's
            ``is_in_data`` 1D array are refined.
        detector
            Detector with projection center(s) (PCs) to use in
            refinement. Its navigation shape must be equal to the
            signal's navigation shape, or it must only contain one PC.
        master_pattern
            Must be in the Lambert projection and have a phase equal to
            the crystal map phase in the points to refine.
        navigation_mask
            Navigation mask of points to refine, equal to ``False``. Its
            shape must be equal to the signal's navigation shape. If
            given, this mask is combined with the crystal map's
            ``is_in_data`` 1D array.
        signal_mask
            Signal mask of detector pixels to use in refinement, equal
            to ``False``. Its shape must be equal to the signal's signal
            shape.

        Returns
        -------
        points_to_refine
            1D mask of points to refine in the crystal map.

        Raises
        ------
        ValueError
            If refinement parameters are not compatible with the signal
            or if they are not suitable for refinement.
        """
        am = self.axes_manager
        nav_shape = am.navigation_shape[::-1]
        sig_shape = am.signal_shape[::-1]

        _ = _detector_is_compatible_with_signal(
            detector=detector,
            nav_shape=nav_shape,
            sig_shape=sig_shape,
            raise_if_not=True,
        )
        if signal_mask is not None and sig_shape != signal_mask.shape:
            raise ValueError(
                f"Signal mask shape {signal_mask.shape} and signal's signal shape "
                f"{sig_shape} must be the same shape"
            )

        _ = _xmap_is_compatible_with_signal(
            xmap=xmap, navigation_axes=am.navigation_axes[::-1], raise_if_not=True
        )

        # Checks navigation mask shape and whether there is only one
        # phase ID in points to refine
        points_to_refine, _, phase_id, _ = _get_indexed_points_in_data_in_xmap(
            xmap, navigation_mask
        )

        master_pattern._is_suitable_for_projection(raise_if_not=True)

        xmap_phase = xmap.phases_in_data[phase_id]
        mp_phase = master_pattern.phase
        equal_phases, are_different = _equal_phase(mp_phase, xmap_phase)
        if not equal_phases:
            raise ValueError(
                f"Master pattern phase '{mp_phase.name}' and phase of points to refine "
                f"in crystal map '{xmap_phase.name}' must be the same, but have "
                f"different {are_different}"
            )

        return points_to_refine

    def _prepare_patterns_for_refinement(
        self,
        points_to_refine: np.ndarray,
        signal_mask: np.ndarray | None,
        rechunk: bool,
        chunk_kwargs: dict | None = None,
    ) -> tuple[da.Array, np.ndarray]:
        """Prepare pattern array and mask for refinement.

        Parameters
        ----------
        points_to_refine
            1D mask of points (patterns) to use in refinement.
        signal_mask
            A boolean mask equal to the signal's signal shape, where
            only pixels equal to ``False`` are used in refimement. If
            not given, all pixels are used.
        rechunk
            Whether to allow rechunking of Dask array produced from the
            signal patterns if the array has only one chunk.
        chunk_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.signals.util._dask.get_chunking`.

        Returns
        -------
        patterns
            3D Dask array (last axis is for potential pseudo-symmetry
            operators in the rotations array).
        signal_mask
            1D NumPy array with points to use in refinement equal to
            ``True``.
        """
        # Could cast pattern array to float32 here already, but
        # have found that this gives different results than doing it
        # in the Numba accelerated function preparing each pattern for
        # refinement...
        patterns = get_dask_array(signal=self)

        # Flatten dimensions for masking
        am = self.axes_manager
        patterns = da.atleast_3d(patterns)
        patterns = patterns.reshape((am.navigation_size, am.signal_size))

        if not points_to_refine.all():
            patterns = patterns[points_to_refine, :]

        if signal_mask is None:
            signal_mask = np.ones(self.axes_manager.signal_size, dtype=bool)
        else:
            signal_mask = ~signal_mask.ravel()
            patterns = patterns[:, signal_mask]

        if patterns.shape[0] == patterns.chunksize[0] and not rechunk:
            pass
        else:
            if chunk_kwargs is None:
                chunk_kwargs = {}
            if "chunk_shape" not in chunk_kwargs:
                chunk_shape = patterns.chunksize[0]
                if chunk_shape == patterns.shape[0]:
                    chunk_shape = 64
                chunk_kwargs["chunk_shape"] = chunk_shape
            chunks = get_chunking(
                data_shape=patterns.shape,
                nav_dim=1,
                sig_dim=1,
                dtype="float32",
                **chunk_kwargs,
            )
            patterns = patterns.rechunk(chunks)

        # Add axis for pseudo-symmetry operators in rotations array
        patterns = patterns[:, np.newaxis, :]

        return patterns, signal_mask

    def _prepare_metric(
        self,
        metric: SimilarityMetric | str,
        navigation_mask: np.ndarray | None,
        signal_mask: np.ndarray | None,
        dtype: str | np.dtype | type | None,
        rechunk: bool,
        n_dictionary_patterns: int,
    ) -> SimilarityMetric:
        metrics = {
            "ncc": NormalizedCrossCorrelationMetric,
            "ndp": NormalizedDotProductMetric,
        }
        if isinstance(metric, str) and metric in metrics:
            metric_class = metrics[metric]
            metric = metric_class()
            metric.rechunk = rechunk

        if not isinstance(metric, SimilarityMetric):
            raise ValueError(
                f"'{metric}' must be either of {metrics.keys()} or a custom metric "
                "class inheriting from SimilarityMetric. See "
                "kikuchipy.indexing.SimilarityMetric"
            )

        metric.n_experimental_patterns = max(self.axes_manager.navigation_size, 1)
        metric.n_dictionary_patterns = max(n_dictionary_patterns, 1)

        if navigation_mask is not None:
            metric.navigation_mask = navigation_mask

        if signal_mask is not None:
            metric.signal_mask = signal_mask

        if dtype is not None:
            metric.dtype = dtype

        metric.raise_error_if_invalid()

        return metric

    @staticmethod
    def _get_sum_signal(
        signal, out_signal_axes: list | None = None
    ) -> hs.signals.Signal2D:
        out = signal.nansum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2))
            )
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError(
                "The length of 'out_signal_axes' cannot be longer than the navigation "
                "dimension of the signal"
            )
        out.set_signal_type("")
        return out.transpose(out_signal_axes)

    # --- Inherited methods from KikuchipySignal2D (possibly from
    # Signal2D or BaseSignal) overwritten here for documentation
    # purposes

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: tuple[int, int] | tuple[float, float] | None = None,
        out_range: tuple[int, int] | tuple[float, float] | None = None,
        dtype_out: (
            str | np.dtype | type | tuple[int, int] | tuple[float, float] | None
        ) = None,
        percentiles: tuple[int, int] | tuple[float, float] | None = None,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        return super().rescale_intensity(
            relative,
            in_range,
            out_range,
            dtype_out,
            percentiles,
            show_progressbar,
            inplace,
            lazy_output,
        )

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: str | np.dtype | type | None = None,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        return super().normalize_intensity(
            num_std,
            divide_by_square_root,
            dtype_out,
            show_progressbar,
            inplace,
            lazy_output,
        )

    def adaptive_histogram_equalization(
        self,
        kernel_size: tuple[int, int] | list[int] | None = None,
        clip_limit: int | float = 0.0,
        nbins: int = 128,
        show_progressbar: bool | None = None,
        inplace: bool = True,
        lazy_output: bool | None = None,
    ) -> EBSD | LazyEBSD | None:
        return super().adaptive_histogram_equalization(
            kernel_size,
            clip_limit,
            nbins,
            show_progressbar,
            inplace,
            lazy_output,
        )

    def as_lazy(self, *args, **kwargs) -> LazyEBSD:
        return super().as_lazy(*args, **kwargs)

    def change_dtype(self, *args, **kwargs) -> None:
        super().change_dtype(*args, **kwargs)
        if isinstance(self, EBSD) and isinstance(
            self._static_background, (np.ndarray, da.Array)
        ):
            self._static_background = self._static_background.astype(self.data.dtype)

    @insert_doc_disclaimer(cls=Signal2D, meth=Signal2D.deepcopy)
    def deepcopy(self) -> EBSD:
        return super().deepcopy()


class LazyEBSD(LazyKikuchipySignal2D, EBSD):
    """Lazy implementation of :class:`~kikuchipy.signals.EBSD`.

    See the documentation of ``EBSD`` for attributes and methods.

    This class extends HyperSpy's
    :class:`~hyperspy.signals.LazySignal2D` class for EBSD patterns. See
    the documentation of that class for how to create this signal and
    the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)

    def get_decomposition_model_write(
        self,
        components: int | list[int] | None = None,
        dtype_learn: str | np.dtype | type = "float32",
        mbytes_chunk: int = 100,
        dir_out: str | None = None,
        fname_out: str | None = None,
    ) -> None:
        """Write the model signal generated from the selected number of
        principal components directly to an ``.hspy`` file.

        The model signal intensities are rescaled to the original
        signals' data type range, keeping relative intensities.

        Parameters
        ----------
        components
            If not given, rebuilds the signal from all ``components``.
            If ``int``, rebuilds signal from ``components`` in range
            0-given ``int``. If list of ``int``, rebuilds signal from
            only ``components`` in given list.
        dtype_learn
            Data type to set learning results to (default is
            ``"float32"``) before multiplication.
        mbytes_chunk
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        dir_out
            Directory to place output signal in.
        fname_out
            Name of output signal file.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal cannot sometimes be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using
        :func:`dask.array.matmul`, out of core.
        """
        dtype_learn = np.dtype(dtype_learn)

        # Change data type, keep desired components and rechunk if lazy
        factors, loadings = _update_learning_results(
            self.learning_results, components=components, dtype_out=dtype_learn
        )

        # Write learning results to HDF5 file
        if dir_out is None:
            try:
                dir_out = self.original_metadata.General.original_filepath
            except AttributeError:
                raise AttributeError("Output directory has to be specified")

        t_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        file_learn = os.path.join(dir_out, "learn_" + t_str + ".h5")
        with File(file_learn, mode="w") as f:
            f.create_dataset(name="factors", data=factors)
            f.create_dataset(name="loadings", data=loadings)

        # Matrix multiplication
        with File(file_learn) as f:
            # Read learning results from HDF5 file
            chunks = _rechunk_learning_results(
                factors=factors, loadings=loadings, mbytes_chunk=mbytes_chunk
            )
            factors = da.from_array(f["factors"], chunks=chunks[0])
            loadings = da.from_array(f["loadings"], chunks=chunks[1])

            # Perform matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose

            # Create new signal from result of matrix multiplication
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = res.reshape(s_model.data.shape)
            s_model.data = s_model.data.rechunk(chunks=(1, 1, -1, -1))

            # Rescale intensities
            s_model.rescale_intensity(dtype_out=self.data.dtype, relative=True)

            # Write signal to file
            if fname_out is None:
                fname_out = "model_" + t_str
            file_model = os.path.join(dir_out, fname_out)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink


def _update_custom_attributes(
    attributes: dict,
    nav_slices: slice | tuple | None = None,
    sig_slices: slice | tuple | None = None,
    new_nav_shape: tuple[int, ...] | None = None,
    new_sig_shape: tuple[int, int] | None = None,
) -> dict:
    """Update dictionary of custom attributes after slicing the signal
    data.

    Parameters
    ----------
    attributes
        Dictionary of attribute keys ``"xmap"``, ``"static_background"``
        and ``"detector"``.
    nav_slices
        Slice or tuple of slices or ints or a combination. If not given,
        navigation dimensions of attributes are not updated.
    sig_slices
        Slice or tuple of slices or ints or a combination. If not given,
        signal dimensions of attributes are not updated.
    """
    if sig_slices is not None:
        try:
            static_bg = attributes["static_background"]
            attributes["static_background"] = static_bg[sig_slices]
        except TypeError:
            _logger.debug("Could not slice EBSD.static_background attribute array")

        # Make slices into extent (top, bottom, left, right)
        extent = [
            sig_slices[0].start,
            sig_slices[0].stop,
            sig_slices[1].start,
            sig_slices[1].stop,
        ]
        for i, new in enumerate([0, new_sig_shape[0], 0, new_sig_shape[1]]):
            if extent[i] is None:
                extent[i] = new

        det = attributes["detector"]
        try:
            attributes["detector"] = det.crop(extent)
        except ValueError:
            _logger.debug(
                "Could not crop EBSD.detector attribute, setting PC array to [0.5, 0.5,"
                " 0.5]"
            )
            attributes["detector"] = EBSDDetector(
                shape=new_sig_shape,
                pc=[0.5, 0.5, 0.5],
                sample_tilt=det.sample_tilt,
                tilt=det.tilt,
                azimuthal=det.azimuthal,
                px_size=det.px_size,
                binning=det.binning,
            )

    if nav_slices is not None:
        try:
            xmap = attributes["xmap"]
            xmap = xmap[nav_slices]

            # Remove singleton dimension to make a 1D crystal map
            if new_nav_shape is not None and len(new_nav_shape) < len(xmap.shape):
                if xmap.shape[0] == 1:
                    xmap._y = None
                else:
                    xmap._x = None
            attributes["xmap"] = xmap
        except (IndexError, TypeError, ValueError):
            _logger.debug("Could not slice EBSD.xmap attribute, setting it to None")

        if attributes["detector"].navigation_shape != (1,):
            pc = attributes["detector"].pc[nav_slices]
            if pc.size == 0:
                _logger.debug(
                    "Could not slice EBSD.detector.pc attribute array, setting it to "
                    "[0.5, 0.5, 0.5]"
                )
                pc = [0.5, 0.5, 0.5]
            attributes["detector"].pc = pc

    return attributes


def _get_navigation_axes_unit(axes_manager: AxesManager) -> str:
    nav_shape = axes_manager.navigation_shape[::-1]
    scan_unit = "px"
    if len(nav_shape) > 0:  # Navigation shape can be (1,)
        scan_unit_hs = str(axes_manager.navigation_axes[0].units)
        if scan_unit_hs != "<undefined>":
            scan_unit = scan_unit_hs
    return scan_unit
