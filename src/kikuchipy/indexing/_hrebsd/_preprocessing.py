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

"""In-engine preprocessing of the HREBSD-DIC engine.

The band-pass filter, the subregion and the zero-mean normalization
are applied identically to the grain reference and to every target
(requirements D4), and the optional Hann window built here is applied
by the engine as a residual weight in the one reference frame both
share.  The user's own
:meth:`~kikuchipy.signals.EBSD.remove_static_background` and
:meth:`~kikuchipy.signals.EBSD.remove_dynamic_background` stay their
choice and are never implied here.

Two recorded deviations from EMsoftOO's shared dictionary-indexing
preprocessing (``mod_HREBSDDIC.f90:689-733``): no adaptive histogram
equalization, since a nonlinear locally varying intensity map
violates the affine intensity model the ZNSSD criterion assumes
(D4.2); and no re-normalization of a warped pattern between
iterations, since the affine normalization is inherent to ZNSSD
(D4.5).

References
----------
Ernould et al., AIEP 223 (2022) Ch. 2; Wilkinson, Meaden and Dingley,
Ultramicroscopy 106 (2006) 307 (the band-pass transfer functions
kikuchipy already carries in :mod:`kikuchipy.filters`).
"""

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from kikuchipy.filters.window import highpass_fft_filter, lowpass_fft_filter


def band_pass_transfer_function(
    shape: tuple[int, int], cutoffs: tuple
) -> np.ndarray | None:
    """Return the FFT domain band-pass transfer function.

    The elementwise product of the kikuchipy high-pass and low-pass
    Gaussian transfer functions
    (:func:`kikuchipy.filters.highpass_fft_filter` and
    :func:`kikuchipy.filters.lowpass_fft_filter`), requirements D4.1.

    Parameters
    ----------
    shape
        Pattern shape ``(nrows, ncols)``.
    cutoffs
        ``(high_pass, low_pass)`` cut-off frequencies as fractions of
        the pattern width, either of which may be ``None`` to switch
        that half off. The frozen default of the engine is
        ``(0.05, None)``, EMsoftOO's ``hipassw`` default.

    Returns
    -------
    transfer_function
        Array of *shape* and 64-bit float data type, or ``None`` when
        both cut-offs are ``None``, in which case no filtering is
        done at all.

    Raises
    ------
    ValueError
        If *cutoffs* does not hold exactly two entries, or if a
        cut-off is not a positive fraction below one.
    """
    cutoffs = tuple(cutoffs)
    if len(cutoffs) != 2:
        raise ValueError(
            f"cutoffs must hold exactly two entries (high_pass, low_pass), not "
            f"{cutoffs!r}"
        )
    for cutoff in cutoffs:
        if cutoff is None:
            continue
        if not 0 < float(cutoff) < 1:
            raise ValueError(
                f"every cut-off in cutoffs must be a fraction of the pattern "
                f"width in (0, 1), not {cutoff!r}"
            )
    high_pass, low_pass = cutoffs
    # The cut-offs are fractions of the pattern WIDTH, so the number of
    # COLUMNS scales both of them (requirements D4.1)
    width = shape[1]
    transfer_function = None
    if high_pass is not None:
        transfer_function = highpass_fft_filter(shape, cutoff=high_pass * width)
    if low_pass is not None:
        low = lowpass_fft_filter(shape, cutoff=low_pass * width)
        if transfer_function is None:
            transfer_function = low
        else:
            transfer_function = transfer_function * low
    return transfer_function


def hann_window(
    shape: tuple[int, int], *, bounds: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """Return a separable Hann window over the subregion.

    Optional (``window=False`` is the frozen default of D4.3): neither
    Ernould's chain nor EMsoftOO's DIC path windows the subregion, and
    the knob exists only because the classic cross-correlation HREBSD
    literature windows its regions of interest.

    Parameters
    ----------
    shape
        Pattern shape ``(nrows, ncols)``.
    bounds
        ``(row_start, row_stop, col_start, col_stop)`` of the
        subregion the window is built OVER, from
        :func:`subregion_bounds`. The window is zero outside it. If
        not given, the window spans the whole pattern.

    Returns
    -------
    window
        Array of *shape* and 64-bit float data type.

    Notes
    -----
    Requirements D4.3 asks for a window over the SUBREGION, which is
    what *bounds* builds, and the engine applies the result as a
    per-pixel WEIGHT on the ZNSSD residual in the REFERENCE frame,
    never as a taper multiplied into a pattern before it is warped
    (see :class:`~kikuchipy.indexing._hrebsd._engine.ReferenceState`).
    A pre-warp taper travels with the target and breaks the affine
    intensity model ZNSSD assumes; measured at the Stage A review, it
    degraded a two pixel translation fit by a factor of nineteen.
    """
    nrows, ncols = shape
    if bounds is None:
        return np.outer(np.hanning(nrows), np.hanning(ncols)).astype(np.float64)
    row_start, row_stop, col_start, col_stop = (int(i) for i in bounds)
    window = np.zeros(shape, dtype=np.float64)
    window[row_start:row_stop, col_start:col_stop] = np.outer(
        np.hanning(row_stop - row_start), np.hanning(col_stop - col_start)
    )
    return window


def subregion_bounds(
    shape: tuple[int, int], *, border: float = 0.05
) -> tuple[int, int, int, int]:
    """Return the subregion box left after removing the border.

    Parameters
    ----------
    shape
        Pattern shape ``(nrows, ncols)``.
    border
        Border removed from every edge as a fraction of the pattern
        side. Default is 0.05, applied to all four edges alike.

    Returns
    -------
    bounds
        ``(row_start, row_stop, col_start, col_stop)``, the usual
        half-open Python slice bounds.

    Raises
    ------
    ValueError
        If *border* is negative or leaves no rows or columns.
    """
    if border < 0:
        raise ValueError(f"border must not be negative, got {border}")
    nrows, ncols = shape
    # A fraction of the pattern SIDE, so each axis loses its own count
    # of pixels from BOTH of its edges
    row_border = int(np.rint(border * nrows))
    col_border = int(np.rint(border * ncols))
    bounds = (row_border, nrows - row_border, col_border, ncols - col_border)
    if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        raise ValueError(
            f"border {border} leaves no subregion of a pattern of shape {shape}"
        )
    return bounds


def subregion_mask(
    shape: tuple[int, int],
    *,
    border: float = 0.05,
    dead_band: tuple | None = None,
) -> np.ndarray:
    """Return the boolean subregion of a pattern.

    One global subregion per run (Ernould style global DIC; there is
    no multi-subset grid in this feature, requirements D4.4).

    Parameters
    ----------
    shape
        Pattern shape ``(nrows, ncols)``.
    border
        Border removed from every edge as a fraction of the pattern
        side. Default is 0.05.
    dead_band
        ``(x0, x1, y0, y1)`` of a cross of dead camera pixel columns
        ``x0:x1`` and rows ``y0:y1`` to exclude, in binned pixels,
        matching EMsoftOO's ``cross(4)``
        (``mod_DIC.f90:524-559``). ``None`` by default, which
        excludes nothing.

    Returns
    -------
    mask
        Boolean array of *shape* which is ``True`` for the pixels
        that BELONG TO the subregion, the opposite polarity of the
        user facing ``navigation_mask``. Stated here because it is a
        frozen internal convention pinned by a test.

    Raises
    ------
    ValueError
        If *border* leaves no pixels, or if *dead_band* does not hold
        exactly four integers within the pattern.
    """
    nrows, ncols = shape
    row_start, row_stop, col_start, col_stop = subregion_bounds(shape, border=border)
    mask = np.zeros(shape, dtype=bool)
    mask[row_start:row_stop, col_start:col_stop] = True
    if dead_band is not None:
        dead_band = tuple(dead_band)
        if len(dead_band) != 4:
            raise ValueError(
                f"dead_band must hold exactly four entries (x0, x1, y0, y1), not "
                f"{dead_band!r}"
            )
        x0, x1, y0, y1 = (int(i) for i in dead_band)
        if not (0 <= x0 <= x1 <= ncols and 0 <= y0 <= y1 <= nrows):
            raise ValueError(
                f"dead_band {dead_band} must hold two ordered column bounds within "
                f"{ncols} and two ordered row bounds within {nrows}"
            )
        # A cross of dead camera columns and rows, EMsoftOO's
        # ``cross(4)`` semantics (``mod_DIC.f90:524-559``)
        mask[:, x0:x1] = False
        mask[y0:y1, :] = False
    return mask


def preprocess(
    pattern: np.ndarray,
    *,
    transfer_function: np.ndarray | None = None,
) -> np.ndarray:
    """Return one preprocessed pattern.

    The band-pass filter is applied in the FFT domain.  Every
    :mod:`scipy.fft` call passes ``workers=1`` so that dask threads
    never oversubscribe the machine (the constitution's numba and FFT
    rules).

    Parameters
    ----------
    pattern
        Pattern of shape ``(nrows, ncols)`` and any real data type.
    transfer_function
        Band-pass transfer function from
        :func:`band_pass_transfer_function`, or ``None`` for no
        filtering.

    Returns
    -------
    preprocessed
        Array of the shape of *pattern* and 64-bit float data type.

    Notes
    -----
    The optional D4.3 window is NOT applied here (corrected
    2026-09-07, Stage A adversarial review).  It is a weight on the
    ZNSSD residual in the reference frame, applied by
    :class:`~kikuchipy.indexing._hrebsd._engine.ReferenceState` after
    the target is warped; multiplying it into a pattern here would
    make it travel with the target.
    """
    # A copy, always: the engine holds on to the result and must never
    # alias, let alone mutate, the caller's pattern buffer
    preprocessed = np.array(pattern, dtype=np.float64)
    if transfer_function is not None:
        # The kikuchipy transfer functions are built about the CENTRE
        # of the spectrum (``distance_to_origin`` puts its origin at
        # ``shape // 2``), so the spectrum is shifted before the
        # product and shifted back after it
        spectrum = fftshift(fft2(preprocessed, workers=1))
        spectrum = spectrum * transfer_function
        preprocessed = np.real(ifft2(ifftshift(spectrum), workers=1))
    return np.ascontiguousarray(preprocessed, dtype=np.float64)


def zero_mean_normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the zero-mean unit-norm vector and its norm.

    The ZNSSD normalization of requirements D2.1: the mean is
    subtracted and the result divided by the vector 2-norm of the
    centred values, which matches EMsoftOO's ``applyZMN_``
    (``mod_DIC.f90:583-624``).  The norm is returned because the
    IC-GN Hessian and gradient carry the matched ``2/norm**2`` and
    ``2/norm`` scales (D2.3): a scale on the gradient alone does not
    cancel in ``H**-1 g``.

    Parameters
    ----------
    values
        1D array of subregion intensities.

    Returns
    -------
    normalized
        ``(values - values.mean()) / norm`` as 64-bit floats.
    norm
        The 2-norm of the centred values.

    Raises
    ------
    ValueError
        If the centred values have a zero norm, the constant pattern
        case which the engine marks as a failed pattern.
    """
    values = np.asarray(values, dtype=np.float64)
    centred = values - values.mean()
    norm = float(np.linalg.norm(centred))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(
            "the zero-mean values have a zero or non-finite 2-norm, so the "
            "pattern carries no contrast to correlate"
        )
    return centred / norm, norm
