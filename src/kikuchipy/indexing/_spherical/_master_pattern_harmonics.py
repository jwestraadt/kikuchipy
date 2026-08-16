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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMSphInx
# (https://github.com/EMsoft-org/EMSphInx, commit 60f3517):
# - ``include/idx/master.hpp``: ``MasterPattern::read()`` (lines
#   242-347, the energy weights and the atom site sum),
#   ``MasterPattern::resize()`` (354-374),
#   ``MasterPattern::toLegendre()`` (381-416),
#   ``MasterPattern::makeInvSym()`` (521-527),
#   ``MasterSpectra::MasterSpectra()`` (550-595),
#   ``MasterSpectra::resize()`` (601-614),
#   ``MasterSpectra::read()`` (619-640) and
#   ``MasterSpectra::removeDC()`` (line 200)
# - ``include/util/image.hpp``: ``BiPix::interpolate()`` and
#   ``BiPix::bilinearCoeff()`` (lines 513-553) and
#   ``Rescaler::scale()`` (582-618)
# - ``programs/mp2sht.cpp`` (lines 53-130), the file field mapping and
#   the chemical formula string
# - ``programs/sht2png.cpp`` (lines 179-274), the order of the header
#   dump reproduced by ``describe()``

# #####################################################################
# Copyright (c) 2019-2019, De Graef Group, Carnegie Mellon University
# All rights reserved.
#
# Author: William C. Lenthe
#
# This package is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, check the Free Software Foundation
# website: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
#
#
# Interested in a commercial license? Contact:
#
# Center for Technology Transfer and Enterprise Creation
# 4615 Forbes Avenue, Suite 302
# Pittsburgh, PA 15213
#
# phone. : 412.268.7393
# email  : innovation@cmu.edu
# website: https://www.cmu.edu/cttec/
#
# Modified by Johan Westraadt, 2026-08: translated to Python/NumPy for
# kikuchipy. GPL-2.0-or-later, conveyed under GPL-3.0-or-later
# #####################################################################

"""Spherical harmonic coefficients of an EBSD master pattern, the
equivalent of EMSphInx' ``MasterSpectra`` and its ``*.sht`` files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from kikuchipy.indexing._spherical import _sht_file

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import Phase
    from orix.quaternion import Rotation

    from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern

__all__ = ["MasterPatternHarmonics"]

HEMISPHERES = ("upper", "lower", "both")
"""Accepted hemisphere selections, as in the ``ebsdsim_master_pattern``
plugin."""

DEFAULT_BANDWIDTH = 384
"""Default bandwidth, that of EMSphInx' ``mp2sht``
(``programs/mp2sht.cpp`` line 53)."""

_ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)  # fmt: skip
"""Chemical symbols of the elements, index ``Z - 1``."""


# --------------------------- Private helpers ------------------------ #


def _resize_lambert(image: np.ndarray, new_dim: int) -> np.ndarray:
    """Return a square Lambert image rescaled with the discrete cosine
    transform.

    Parameters
    ----------
    image
        Square image of shape ``(dim, dim)``.
    new_dim
        Side length of the returned image.

    Returns
    -------
    rescaled
        Image of shape ``(new_dim, new_dim)`` and 64-bit floating
        point data type.  ``image`` itself is returned unchanged when
        ``new_dim == dim``.

    Notes
    -----
    Ported from ``MasterPattern::resize`` (``idx/master.hpp`` lines
    354-374) and ``image::Rescaler::scale``
    (``util/image.hpp`` lines 582-618): an unnormalized DCT-II
    (:func:`scipy.fft.dctn` with ``type=2``, the FFTW ``REDFT10``),
    truncation or zero padding of the low frequency corner, an
    unnormalized DCT-III (``type=3``, ``REDFT01``, never
    :func:`scipy.fft.idctn`) and the scale factor
    ``0.5 / new_dim ** 2``.

    That factor preserves the amplitude only when
    ``new_dim ** 2 == 2 * dim ** 2``: a constant ``c`` comes back as
    ``c * 2 * dim ** 2 / new_dim ** 2``.  It cancels in the
    normalization of :func:`_normalize_hemispheres` and is ported
    verbatim rather than "fixed", so that ``normalize=False``
    reproduces EMSphInx' amplitude and not the source's.
    """
    raise NotImplementedError


def _to_legendre(
    north: np.ndarray, south: np.ndarray, dim_legendre: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return two square Lambert hemispheres sampled on a square
    Legendre grid.

    Parameters
    ----------
    north, south
        Square Lambert hemispheres of shape ``(dim, dim)``.
    dim_legendre
        Side length of the square Legendre grid.

    Returns
    -------
    north_legendre, south_legendre
        Hemispheres of shape ``(dim_legendre, dim_legendre)`` and
        64-bit floating point data type.

    Notes
    -----
    Ported from ``MasterPattern::toLegendre`` (``idx/master.hpp``
    lines 390-408) and ``image::BiPix``
    (``util/image.hpp`` lines 513-553).  For every Legendre normal
    ``v`` of the *northern* hemisphere the square Lambert coordinate
    ``(X, Y) = sphere_to_square(v)`` is used for both hemispheres,
    since the transform uses ``|z|``.  The sample is bilinear with
    ``x = X * (dim - 1)``, ``y = Y * (dim - 1)``, floored corner
    indices clamped to ``dim - 1`` and the row index taken from
    ``Y``.
    """
    raise NotImplementedError


def _normalize_hemispheres(
    north: np.ndarray, south: np.ndarray, emsphinx_compatible: bool = True
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return two square Legendre hemispheres with zero solid angle
    weighted mean and unit weighted standard deviation.

    Parameters
    ----------
    north, south
        Hemispheres of shape ``(dim_legendre, dim_legendre)``.
    emsphinx_compatible
        Whether to reproduce EMSphInx' two quirks, ``True`` by
        default.

    Returns
    -------
    north, south
        The normalized hemispheres.
    mean
        The weighted mean over both hemispheres.  With
        ``emsphinx_compatible=True`` it is *twice* this value which
        is subtracted, so the weighted mean of the returned
        hemispheres is ``-mean / std`` and not zero.
    std
        The weighted standard deviation which was divided out.

    Notes
    -----
    Ported from ``MasterSpectra::MasterSpectra`` (``idx/master.hpp``
    lines 557-584).  The weights are the Legendre ring solid angles
    at every pixel with the four border lines halved, so that the
    corners, which lie on two borders, end up quartered.

    With ``emsphinx_compatible=True`` (the default) the mean is
    divided by ``sum(w)`` although it sums over both hemispheres, so
    twice the mean is subtracted and ``a_00`` ends up at
    ``-sqrt(4 pi) mu / sigma``, about ``-2.985`` for the in-package Ni
    master; the corners stay quartered.  This is what ``mp2sht``
    writes, so it reproduces every existing ``*.sht`` file.

    With ``emsphinx_compatible=False`` the mean is divided by
    ``2 * sum(w)`` and the corners are halved once, which is the
    corrected normalization: ``a_00`` is then about ``-6e-5``.  The
    two settings differ only by a global factor
    ``sqrt(1 + mu ** 2 / sigma ** 2)`` and the DC term.
    """
    raise NotImplementedError


def _energy_weights(
    master_pattern: EBSDMasterPattern, energy_weights: np.ndarray | None
) -> np.ndarray:
    """Return the energy weights of a master pattern.

    Parameters
    ----------
    master_pattern
        The master pattern.
    energy_weights
        Weights given by the user, or ``None``.

    Returns
    -------
    weights
        Weights of shape ``(n_energies,)`` and 64-bit floating point
        data type, summing to one.

    Raises
    ------
    ValueError
        If the weights are negative, sum to zero or have the wrong
        length; or, when they must be read from the source file, if
        that file or its ``EMData/MCOpenCL/accum_e`` data set is
        missing or an energy is outside the histogram.  Every message
        names ``energy_weights``.

    Notes
    -----
    Ported from ``MasterPattern::read`` (``idx/master.hpp`` lines
    274-278): ``accum_e`` is summed over every axis but the last and
    the bins of the loaded energies are selected and normalized.  The
    source file is found through ``tmp_parameters``, which
    :func:`kikuchipy.load` sets.

    EMSphInx refuses a file whose bin count does not match the master
    pattern's (line 319), while we renormalize over the loaded bins,
    so a master pattern loaded with ``energy=(15, 20)`` works.
    """
    raise NotImplementedError


def _accum_e_weights(
    filename: str, energies: np.ndarray, e_hist_min: float, e_bin_size: float
) -> np.ndarray:
    """Return energy weights read from a Monte Carlo simulation.

    Parameters
    ----------
    filename
        Path to the EMsoft HDF5 file.
    energies
        Energies of the master pattern's energy axis in keV.
    e_hist_min
        Lowest energy bin in keV.
    e_bin_size
        Energy bin width in keV.

    Returns
    -------
    weights
        Weights of shape ``(len(energies),)``, summing to one.

    Raises
    ------
    ValueError
        If the file or the ``EMData/MCOpenCL/accum_e`` data set is
        missing, or if an energy maps to a bin outside the
        histogram.  Every message names ``energy_weights``.
    """
    raise NotImplementedError


def _check_atom_sum(master_pattern: EBSDMasterPattern) -> None:
    """Raise if a master pattern's atom sites were summed in an
    integer data type.

    Parameters
    ----------
    master_pattern
        The master pattern.

    Raises
    ------
    ValueError
        If the data are integer, ``CrystalData.Natomtypes`` is above
        one and ``EBSDMasterNameList.combinesites`` is missing or
        false.  The message names ``combinesites``.

    Notes
    -----
    kikuchipy's EMsoft reader sums the atom sites in the *file* data
    type (``io/plugins/_emsoft_master_pattern.py`` line 174) whereas
    EMSphInx reads them as ``NATIVE_FLOAT`` and accumulates in 32-bit
    floating point (``idx/master.hpp`` lines 330-338).  The two agree
    for a single site or a floating point source, but an unsigned
    integer sum of several sites wraps modulo the data type, so the
    parity claim would be void.
    """
    raise NotImplementedError


def _metadata_from_master_pattern(
    master_pattern: EBSDMasterPattern,
    beam_energy: float,
    sample_tilt: float,
    bandwidth: int,
    normalize: bool,
    emsphinx_compatible: bool,
) -> dict[str, Any]:
    """Return the file metadata of a master pattern.

    Parameters
    ----------
    master_pattern
        The master pattern.
    beam_energy
        Beam energy in keV.
    sample_tilt
        Sample tilt in degrees.
    bandwidth
        Bandwidth of the coefficients.
    normalize, emsphinx_compatible
        Recorded in the ``header.notes`` provenance string.

    Returns
    -------
    original_metadata
        Dictionary of the shape
        :meth:`~kikuchipy.indexing._spherical._sht_file.ShtFile.metadata_dict`
        gives, so that :meth:`MasterPatternHarmonics.save` writes the
        same crystal and simulation blocks as ``mp2sht``.

    Notes
    -----
    The EMsoftED simulation block is only built when every key of the
    mapping is present in the signal's ``original_metadata``, which
    the in-package Ni master has and the 13 px
    ``master_patterns.h5`` does not.
    """
    raise NotImplementedError


def _phase_from_sht(sht_file: _sht_file.ShtFile) -> Phase:
    """Return the phase of a parsed SHT file.

    Parameters
    ----------
    sht_file
        The parsed file.

    Returns
    -------
    phase
        Phase with the *effective* space group (which is what
        EMSphInx takes the point group from, ``idx/master.hpp`` line
        628), the lattice as stored (nm) and one atom per record.
    """
    raise NotImplementedError


def _sht_from_harmonics(
    harmonics: MasterPatternHarmonics,
    doi: str | None,
    notes: str | None,
    preserve_header: bool,
) -> _sht_file.ShtFile:
    """Return a writeable SHT file from harmonic coefficients.

    Parameters
    ----------
    harmonics
        The coefficients and their metadata.
    doi
        DOI string, or ``None`` to use the stored one.
    notes
        Notes string, or ``None`` to use the stored one.
    preserve_header
        Whether to keep every non-harmonics field of the file the
        coefficients were read from.

    Returns
    -------
    sht_file
        The file.

    Raises
    ------
    ValueError
        If ``preserve_header`` is ``True`` and the coefficients did
        not come from a file, or if the phase has neither a space
        group nor a point group.
    """
    raise NotImplementedError


def _packing_loss(alm: np.ndarray, bandwidth: int, z_rot: int, flags: int) -> float:
    """Return the relative power the packing would drop.

    Parameters
    ----------
    alm
        Coefficients ``alm[m, l]`` of shape ``(bw, bw)``.
    bandwidth
        Bandwidth of the coefficients.
    z_rot
        Z rotational order.
    flags
        Compression flags.

    Returns
    -------
    loss
        ``sum |alm - unpack(pack(alm))| ** 2 / sum |alm| ** 2`` with
        the ``m > 0`` entries counted twice, the same quantity
        :func:`~kikuchipy.indexing._spherical._symmetry.systematic_zero_power`
        uses so that the construction and saving guards agree.
    """
    raise NotImplementedError


def _master_pattern_dict(
    harmonics: MasterPatternHarmonics, dim: int | None, hemisphere: str
) -> dict[str, Any]:
    """Return the signal dictionary of a synthesized master pattern.

    Parameters
    ----------
    harmonics
        The coefficients.
    dim
        Side length of the square Lambert grid, or ``None`` for
        ``2 * bandwidth + 1``.
    hemisphere
        One of :data:`HEMISPHERES`.

    Returns
    -------
    signal_dict
        Dictionary with the keys ``"data"``, ``"axes"``,
        ``"metadata"``, ``"original_metadata"``, ``"projection"``,
        ``"phase"`` and ``"hemisphere"``, shared by
        :meth:`MasterPatternHarmonics.to_master_pattern` and the
        ``emsphinx_master_pattern`` io plugin.

    Raises
    ------
    ValueError
        If ``hemisphere`` is not one of :data:`HEMISPHERES` or
        ``dim`` is even or smaller than three.

    Warns
    -----
    UserWarning
        If ``(dim - 1) // 2`` is smaller than the bandwidth, in which
        case the coefficients are band limited to that degree.
    """
    raise NotImplementedError


# ---------------------------- Public class -------------------------- #


class MasterPatternHarmonics:
    r"""Spherical harmonic coefficients of an EBSD master pattern.

    This is the kikuchipy equivalent of EMSphInx'
    ``MasterSpectra`` :cite:`lenthe2019spherical` and of the
    ``*.sht`` files its ``mp2sht`` program writes and its
    ``IndexEBSD`` program indexes with.

    Parameters
    ----------
    alm
        Coefficients ``alm[m, l]`` of shape ``(bandwidth,
        bandwidth)``, m-major and l-minor, with the ``l < m`` entries
        zero.  Copied and cast to 128-bit complex.
    phase
        Phase of the master pattern, which gives the symmetry flags
        and the crystal block of a written file.
    beam_energy
        Beam energy in keV.
    sample_tilt
        Sample tilt in degrees.
    original_metadata
        Header, crystal, simulation and harmonics fields of the file
        the coefficients came from or will be written to, of the
        shape
        :meth:`~kikuchipy.indexing._spherical._sht_file.ShtFile.metadata_dict`
        gives.  A plain dictionary whose content equals the
        ``original_metadata`` of the signal
        :meth:`to_master_pattern` returns.

    Attributes
    ----------
    alm : numpy.ndarray
        The coefficients, C-contiguous and never modified in place.
    phase : orix.crystal_map.Phase
        The phase.
    beam_energy : float
        Beam energy in keV.
    sample_tilt : float
        Sample tilt in degrees.
    original_metadata : dict
        The file fields.

    Raises
    ------
    ValueError
        If ``alm`` is not square and two dimensional, if any
        ``l < m`` entry is non-zero, or if the bandwidth is smaller
        than one.

    Warns
    -----
    UserWarning
        If the coefficients do not have the symmetry the phase's
        point group claims, see
        :func:`~kikuchipy.indexing._spherical._symmetry.validate_flags`.

    See Also
    --------
    kikuchipy.signals.EBSDMasterPattern.get_spherical_harmonics

    Notes
    -----
    Coefficients follow the convention of
    :func:`scipy.special.sph_harm_y`, Condon-Shortley phase
    included, and are fully normalized, so a function equal to one
    everywhere has ``alm[0, 0] == sqrt(4 * pi)``.  Only the
    non-negative orders are stored; order ``-m`` is
    ``(-1) ** m * conj(alm[m, l])``.

    Ported EMSphInx quirks, each behind one keyword:

    +-----------------------+----------+-----------------------------+
    | keyword               | default  | what it does                |
    +=======================+==========+=============================+
    | ``emsphinx_compatible`` | ``True`` | Reproduces EMSphInx'      |
    |                       |          | normalization: the         |
    |                       |          | weighted mean is divided   |
    |                       |          | by the weight sum of *one* |
    |                       |          | hemisphere although it     |
    |                       |          | sums over both, so twice   |
    |                       |          | the mean is subtracted,    |
    |                       |          | and the grid corners are   |
    |                       |          | quartered.  ``a_00`` ends  |
    |                       |          | up at about ``-2.985`` for |
    |                       |          | the Ni master, 71 % of the |
    |                       |          | total power.  ``False`` is |
    |                       |          | the corrected             |
    |                       |          | normalization with         |
    |                       |          | ``a_00`` about ``-6e-5``;  |
    |                       |          | the two differ only by a   |
    |                       |          | global factor and the DC   |
    |                       |          | term.                      |
    +-----------------------+----------+-----------------------------+
    | ``strict``            | ``True`` | :meth:`save` refuses to    |
    |                       |          | write when the file's      |
    |                       |          | compression would drop     |
    |                       |          | more relative power than   |
    |                       |          | ``SYMMETRY_POWER_TOLERANCE``|
    |                       |          | . ``False`` warns and      |
    |                       |          | drops, which is what       |
    |                       |          | EMSphInx does silently.    |
    +-----------------------+----------+-----------------------------+

    The default ``emsphinx_compatible=True`` is parity first: every
    ``*.sht`` file in existence was written that way and EMSphInx'
    ``IndexEBSD`` never removes the DC term.  That matters because
    EMSphInx' *normalized* correlator (its namelist default) divides
    the numerator by a rotation dependent denominator, so the
    master's DC term can shift the best orientation, while the
    un-normalized correlator is unaffected by it.

    References
    ----------
    :cite:`lenthe2019spherical`
    """

    def __init__(
        self,
        alm: np.ndarray,
        *,
        phase: Phase | None = None,
        beam_energy: float | None = None,
        sample_tilt: float | None = None,
        original_metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    # ------------------------- Properties -------------------------- #

    @property
    def bandwidth(self) -> int:
        """Return the bandwidth, i.e. ``alm.shape[0]``."""
        raise NotImplementedError

    @property
    def n_fold(self) -> int:
        """Return the z rotational order of the coefficients, one of
        1, 2, 3, 4 and 6.

        EMSphInx' ``MasterSpectra::nFold()``.
        """
        raise NotImplementedError

    @property
    def has_equatorial_mirror(self) -> bool:
        """Return whether the coefficients have a mirror plane
        perpendicular to z.

        EMSphInx' ``MasterSpectra::mirror()``.
        """
        raise NotImplementedError

    # --------------------- Construction and IO --------------------- #

    @classmethod
    def from_master_pattern(
        cls,
        master_pattern: EBSDMasterPattern,
        *,
        bandwidth: int = DEFAULT_BANDWIDTH,
        energy_weights: np.ndarray | None = None,
        normalize: bool = True,
        emsphinx_compatible: bool = True,
        beam_energy: float | None = None,
        sample_tilt: float | None = None,
    ) -> MasterPatternHarmonics:
        """Return the harmonic coefficients of a master pattern.

        Parameters
        ----------
        master_pattern
            Master pattern in the square Lambert projection with both
            hemispheres, or with the upper hemisphere alone when the
            point group contains inversion.
        bandwidth
            Bandwidth, i.e. the exclusive maximum harmonic degree,
            384 by default as in EMSphInx' ``mp2sht``.
        energy_weights
            Weight of every energy of the signal.  If not given and
            the signal has an energy axis, they are read from the
            Monte Carlo results ``EMData/MCOpenCL/accum_e`` of the
            source file.  Normalized to sum to one.
        normalize
            Whether to give the master pattern zero solid angle
            weighted mean and unit weighted standard deviation before
            the transform, ``True`` by default.
        emsphinx_compatible
            Whether to reproduce EMSphInx' normalization quirks,
            ``True`` by default, see the class ``Notes``.
        beam_energy
            Beam energy in keV.  If not given,
            ``original_metadata.MCCLNameList.EkeV`` is used.
        sample_tilt
            Sample tilt in degrees.  If not given,
            ``original_metadata.MCCLNameList.sig`` is used.

        Returns
        -------
        harmonics
            The coefficients.

        Raises
        ------
        TypeError
            If ``master_pattern`` is not an
            :class:`~kikuchipy.signals.EBSDMasterPattern`.
        NotImplementedError
            If the master pattern is not in the square Lambert
            projection.
        ValueError
            If the bandwidth is outside ``[1, 32767]``; if the
            hemisphere is ``"lower"``, or ``"upper"`` for a point
            group without inversion; if the atom sites were summed in
            an integer data type, see :func:`_check_atom_sum`; if the
            energy weights are unusable, see :func:`_energy_weights`;
            or if the beam energy or sample tilt is given neither as
            a keyword nor in the metadata.

        Warns
        -----
        UserWarning
            If ``bandwidth`` exceeds ``(dim - 1) // 2``, the largest
            degree a square Lambert master pattern of side ``dim``
            carries.

        Notes
        -----
        A line by line port of EMSphInx' ``mp2sht``, i.e.
        ``MasterPattern::read`` followed by the ``MasterSpectra``
        constructor: energy weighting, a discrete cosine transform
        rescale to ``round(sqrt(2) * dim_legendre)``, bilinear
        sampling at the Legendre normals, the weighted normalization
        and the transform at ``dim_legendre = default_dim(bandwidth,
        "legendre")``.  No DC term is removed; use
        :meth:`remove_dc` for that.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(
        ...     projection="lambert", hemisphere="both"
        ... )
        >>> h = kp.indexing.MasterPatternHarmonics.from_master_pattern(
        ...     mp, bandwidth=32
        ... )
        >>> h.bandwidth
        32
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename: str) -> MasterPatternHarmonics:
        """Return the harmonic coefficients stored in an EMSphInx
        ``*.sht`` file.

        Parameters
        ----------
        filename
            Path to the file.

        Returns
        -------
        harmonics
            The coefficients.

        Raises
        ------
        NotImplementedError
            If the file's modality is not EBSD or it holds more than
            one crystal, and for the codec's own unsupported cases
            (big-endian files and versions other than 1.1).
        ValueError
            If the file is not an SHT file or its checksum does not
            match.

        Notes
        -----
        Ported from ``MasterSpectra::read`` (``idx/master.hpp`` lines
        619-640), extended with everything else the header carries.
        The parsed file is kept privately, so that
        ``save(preserve_header=True)`` can write it back byte
        identically.
        """
        raise NotImplementedError

    def save(
        self,
        filename: str,
        *,
        doi: str | None = None,
        notes: str | None = None,
        preserve_header: bool = False,
        strict: bool = True,
        overwrite: bool | None = None,
    ) -> None:
        """Write the coefficients to an EMSphInx ``*.sht`` file.

        Parameters
        ----------
        filename
            Path to write to.  ``".sht"`` is appended when there is
            no suffix.
        doi
            DOI string.  If not given, the stored one is used.
        notes
            Notes string.  If not given, the stored provenance note
            is used, e.g. ``"created with kikuchipy (normalize=True,
            emsphinx_compatible=True)"``.
        preserve_header
            Whether to write every non-harmonics field exactly as it
            was read, ``False`` by default.  Only for coefficients
            made by :meth:`from_file`, for which
            ``from_file(f).save(g, preserve_header=True)`` gives a
            file byte identical to ``f``.
        strict
            Whether to refuse to write when the file's compression
            would drop coefficients, ``True`` by default.
        overwrite
            Whether to overwrite an existing file.  If not given, the
            user is asked.

        Raises
        ------
        ValueError
            If the phase has neither a space group nor a point group;
            if ``preserve_header`` is ``True`` for coefficients not
            read from a file; if ``strict`` is ``True`` and the
            compression would drop more relative power than
            ``SYMMETRY_POWER_TOLERANCE``, in which case the message
            names the two candidate space groups when the point group
            has two (``"3m"``: 156 or 157, ``"-3m"``: 162 or 164,
            ``"-42m"``: 111 or 115, ``"-6m2"``: 187 or 189); or if
            any file sanity check fails.

        Warns
        -----
        UserWarning
            If ``strict`` is ``False`` and coefficients are dropped.

        Notes
        -----
        The field mapping of ``programs/mp2sht.cpp`` lines 58-130,
        with three deliberate differences: the software version is a
        NUL padded kikuchipy tag, the EMsoft version is
        ``"unknown"`` because kikuchipy's reader keeps no
        ``EMheader``, and the crystal name and a provenance note are
        written.

        The lattice parameters are written *as stored* on the phase.
        kikuchipy's EMsoft readers keep them in nm, which is what the
        file format wants, while the ``ebsdsim`` reader gives
        angstrom.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(
        ...     projection="lambert", hemisphere="both"
        ... )
        >>> h = mp.get_spherical_harmonics(bandwidth=32)
        >>> fname = Path(tempfile.mkdtemp()) / "ni_bw32.sht"
        >>> h.save(fname)
        """
        raise NotImplementedError

    def to_master_pattern(
        self, dim: int | None = None, hemisphere: str = "both"
    ) -> EBSDMasterPattern:
        """Return a master pattern synthesized on a square Lambert
        grid.

        Parameters
        ----------
        dim
            Odd side length of the grid, at least three.  If not
            given, ``2 * bandwidth + 1`` is used, the grid whose
            largest bandwidth is exactly :attr:`bandwidth`.
        hemisphere
            One of ``"upper"``, ``"lower"`` and ``"both"``
            (default).

        Returns
        -------
        master_pattern
            Master pattern of 64-bit floating point data type in the
            square Lambert projection, with the ``height`` and
            ``width`` axes centred on the middle pixel, i.e. offset
            ``-(dim // 2)``.

        Raises
        ------
        ValueError
            If ``hemisphere`` is unknown or ``dim`` is even or
            smaller than three.

        Warns
        -----
        UserWarning
            If ``(dim - 1) // 2`` is smaller than :attr:`bandwidth`,
            in which case the coefficients are band limited.

        Notes
        -----
        This replaces EMSphInx' ``MasterPattern::toLambert``
        (``idx/master.hpp`` lines 423-473), which synthesizes on a
        Legendre grid and then picks the nearest of four bounding
        Legendre points.  Sampling the band limited function on the
        Lambert nodes directly is strictly better and is exactly what
        the inverse transform does.

        The axis offsets are centred, ``-(dim // 2)``, as in the
        ``ebsdsim_master_pattern`` plugin, and are therefore one
        pixel off kikuchipy's EMsoft reader, which writes
        ``-dim // 2``.  Nothing computes with these offsets.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(
        ...     projection="lambert", hemisphere="both"
        ... )
        >>> h = mp.get_spherical_harmonics(bandwidth=32)
        >>> mp2 = h.to_master_pattern()
        >>> mp2.data.shape
        (2, 65, 65)
        """
        raise NotImplementedError

    # -------------------------- Manipulation ----------------------- #

    def resize(self, bandwidth: int) -> MasterPatternHarmonics:
        """Return new coefficients at another bandwidth.

        Parameters
        ----------
        bandwidth
            New bandwidth, at least one.

        Returns
        -------
        harmonics
            New instance whose coefficients are the upper left block
            of these when the bandwidth shrinks and these zero padded
            when it grows.

        Raises
        ------
        ValueError
            If ``bandwidth`` is smaller than one.

        Notes
        -----
        Ported from ``MasterSpectra::resize`` (``idx/master.hpp``
        lines 601-614), which is what ``IndexEBSD`` does with the
        bandwidth of its namelist.
        """
        raise NotImplementedError

    def remove_dc(self) -> MasterPatternHarmonics:
        """Return new coefficients with a zero constant term.

        Returns
        -------
        harmonics
            New instance with ``alm[0, 0] == 0``.

        Notes
        -----
        Ported from ``MasterSpectra::removeDC`` (``idx/master.hpp``
        line 200).  EMSphInx' ``IndexEBSD`` never calls it, so a
        ``*.sht`` file written by ``mp2sht`` is indexed with its full
        DC term.
        """
        raise NotImplementedError

    def rotate(self, rotation: Rotation) -> MasterPatternHarmonics:
        """Return coefficients of the rotated master pattern.

        Parameters
        ----------
        rotation
            The rotation.

        Returns
        -------
        harmonics
            New instance.

        Raises
        ------
        NotImplementedError
            Always, until the Wigner-d tables arrive.
        """
        raise NotImplementedError

    # -------------------------- Inspection ------------------------- #

    def power_spectrum(self) -> np.ndarray:
        """Return the power of every harmonic degree.

        Returns
        -------
        power
            ``P[l] = |a_l0| ** 2 + 2 sum_{m = 1}^{l} |a_lm| ** 2`` in
            an array of shape ``(bandwidth,)`` and 64-bit floating
            point data type.

        Notes
        -----
        The ``m > 0`` entries count twice because only the
        non-negative orders are stored.  For a function of unit
        variance the sum over all degrees is ``4 pi`` up to the
        quadrature error.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(
        ...     projection="lambert", hemisphere="both"
        ... )
        >>> h = mp.get_spherical_harmonics(bandwidth=32)
        >>> h.power_spectrum().shape
        (32,)
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Return a multi-line description of the file fields.

        Returns
        -------
        description
            Every field EMSphInx' ``sht2png`` prints
            (``programs/sht2png.cpp`` lines 179-274), in its order,
            followed by a harmonics summary with the bandwidth, the
            compression parameters, the symmetry flags, ``a_00`` and
            the fraction of the total power in the constant term.

        Notes
        -----
        Not a byte copy of ``sht2png``'s output, which is a C++
        stream format, but every field it prints appears.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string with the bandwidth, phase, beam energy and
        sample tilt, e.g. ``"MasterPatternHarmonics: bw = 384, ni
        (m-3m), 20.1 keV, 70.0 deg"``.
        """
        raise NotImplementedError
