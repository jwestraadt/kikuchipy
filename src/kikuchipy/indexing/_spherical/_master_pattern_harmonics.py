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

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from scipy.fft import dctn

from kikuchipy.indexing._spherical import _grid, _sht, _sht_file, _symmetry

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
    image = np.asarray(image, dtype=np.float64)
    dim = image.shape[0]
    if new_dim == dim:
        return image
    spectrum = dctn(image, type=2, workers=1)
    kept = min(dim, new_dim)
    padded = np.zeros((new_dim, new_dim), dtype=np.float64)
    padded[:kept, :kept] = spectrum[:kept, :kept]
    rescaled = dctn(padded, type=3, workers=1)
    rescaled *= 0.5 / new_dim**2
    return rescaled


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
    north = np.asarray(north, dtype=np.float64)
    south = np.asarray(south, dtype=np.float64)
    height, width = north.shape
    normals = _grid.legendre_normals(dim_legendre)
    square = _grid.sphere_to_square(normals.reshape(-1, 3))
    x = square[:, 0] * (width - 1)
    y = square[:, 1] * (height - 1)
    # The C++ cast to size_t truncates towards zero, and no
    # coordinate is negative, so it is a floor
    index_x0 = np.minimum(np.floor(x).astype(np.int64), width - 1)
    index_y0 = np.minimum(np.floor(y).astype(np.int64), height - 1)
    index_x1 = np.minimum(index_x0 + 1, width - 1)
    index_y1 = np.minimum(index_y0 + 1, height - 1)
    weight_x1 = x - index_x0
    weight_y1 = y - index_y0
    weight_x0 = 1.0 - weight_x1
    weight_y0 = 1.0 - weight_y1
    corner_00 = weight_y0 * weight_x0
    corner_01 = weight_y0 * weight_x1
    corner_10 = weight_y1 * weight_x0
    corner_11 = weight_y1 * weight_x1

    sampled = []
    for image in (north, south):
        values = (
            image[index_y0, index_x0] * corner_00
            + image[index_y0, index_x1] * corner_01
            + image[index_y1, index_x0] * corner_10
            + image[index_y1, index_x1] * corner_11
        )
        sampled.append(values.reshape(dim_legendre, dim_legendre))
    return sampled[0], sampled[1]


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
    dim = north.shape[0]
    weights = _grid.ring_solid_angles(dim, "legendre")[_grid.ring_number(dim)]
    # The equatorial ring straddles the equator and the four border
    # lines are shared with the neighbouring hemisphere, so they are
    # halved; the corners lie on two borders and are halved twice
    weights[0] /= 2
    weights[-1] /= 2
    weights[:, 0] /= 2
    weights[:, -1] /= 2
    if not emsphinx_compatible:
        for j in (0, -1):
            for i in (0, -1):
                weights[j, i] *= 2
    total = weights.sum()

    north = np.array(north, dtype=np.float64)
    south = np.array(south, dtype=np.float64)
    mean = (np.sum(weights * north) + np.sum(weights * south)) / (2 * total)
    # EMSphInx divides the sum over both hemispheres by the weight
    # sum of one of them, so twice the mean is subtracted
    shift = 2 * mean if emsphinx_compatible else mean
    north -= shift
    south -= shift
    std = np.sqrt(
        (np.sum(weights * north**2) + np.sum(weights * south**2)) / (2 * total)
    )
    north /= std
    south /= std
    return north, south, float(mean), float(std)


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
    axes = master_pattern.axes_manager
    names = [axis.name for axis in axes.navigation_axes]
    has_energy = "energy" in names
    n_energies = axes["energy"].size if has_energy else 1

    if energy_weights is not None:
        weights = np.asarray(energy_weights, dtype=np.float64).ravel()
        if weights.size != n_energies:
            raise ValueError(
                f"energy_weights must hold {n_energies} values, one per "
                f"energy of the master pattern, not {weights.size}"
            )
        if np.any(weights < 0):
            raise ValueError("energy_weights cannot hold a negative weight")
        total = weights.sum()
        if total == 0:
            raise ValueError("energy_weights cannot sum to zero")
        return weights / total

    if not has_energy:
        return np.ones(1, dtype=np.float64)

    metadata = master_pattern.original_metadata
    e_hist_min = metadata.get_item("MCCLNameList.Ehistmin", None)
    e_bin_size = metadata.get_item("MCCLNameList.Ebinsize", None)
    if e_hist_min is None or e_bin_size is None:
        raise ValueError(
            "Cannot find the energy histogram of the master pattern in "
            "'MCCLNameList.Ehistmin' and 'MCCLNameList.Ebinsize' of its "
            "original metadata; pass energy_weights instead"
        )
    parameters = master_pattern.tmp_parameters
    stem = parameters.get_item("filename", None)
    if stem is None:
        raise ValueError(
            "Cannot find the file the master pattern was read from; pass "
            "energy_weights instead"
        )
    filename = os.path.join(
        parameters.get_item("folder", ""),
        f"{stem}.{parameters.get_item('extension', '')}",
    )
    return _accum_e_weights(
        filename, axes["energy"].axis, float(e_hist_min), float(e_bin_size)
    )


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
    import h5py

    path = Path(filename)
    if not path.is_file():
        raise ValueError(
            f"Cannot read the Monte Carlo results from {str(path)!r}, which "
            "does not exist; pass energy_weights instead"
        )
    dataset = "EMData/MCOpenCL/accum_e"
    with h5py.File(path, "r") as file:
        if dataset not in file:
            raise ValueError(
                f"The file {str(path)!r} has no {dataset!r} data set to get "
                "the energy weights from; pass energy_weights instead"
            )
        counts = np.asarray(file[dataset][()], dtype=np.int64)
    # The last axis is the energy one, the others are the exit
    # positions of the Monte Carlo simulation
    counts = counts.reshape(-1, counts.shape[-1]).sum(axis=0)

    energies = np.asarray(energies, dtype=np.float64)
    bins = np.rint((energies - e_hist_min) / e_bin_size).astype(np.int64)
    if np.any(bins < 0) or np.any(bins >= counts.size):
        raise ValueError(
            f"The energies {energies.tolist()} keV fall outside the "
            f"{counts.size} energy bins of the Monte Carlo results; pass "
            "energy_weights instead"
        )
    weights = counts[bins].astype(np.float64)
    total = weights.sum()
    if total == 0:
        raise ValueError(
            "The Monte Carlo results have no electrons in the energy bins "
            "of the master pattern; pass energy_weights instead"
        )
    return weights / total


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
    if master_pattern.data.dtype.kind not in "iu":
        return
    metadata = master_pattern.original_metadata
    n_atom_types = metadata.get_item("CrystalData.Natomtypes", None)
    if n_atom_types is None or int(n_atom_types) <= 1:
        return
    combine_sites = metadata.get_item("EBSDMasterNameList.combinesites", None)
    if combine_sites is not None and bool(int(combine_sites)):
        return
    raise ValueError(
        f"The master pattern holds {int(n_atom_types)} atom sites which "
        "EMsoft did not combine ('combinesites' is false or absent), and "
        f"kikuchipy's reader has summed them in the {master_pattern.data.dtype} "
        "data type, which wraps on overflow. Simulate the master pattern "
        "with 'combinesites = .TRUE.' or read it as floating point"
    )


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
    phase = master_pattern.phase
    metadata = master_pattern.original_metadata
    space_group = _effective_space_group(phase)
    z_rot = _sht_file.space_group_z_rotation(space_group)
    flags = _sht_file.space_group_compression_flags(space_group)
    simulation = _simulation_from_master_pattern(metadata)
    crystal = _crystal_from_phase(
        phase,
        space_group,
        {"sg_set": int(metadata.get_item("CrystalData.SpaceGroupSetting", 1))},
    )
    sht_file = _sht_file.ShtFile(
        header=_sht_file.ShtHeader(
            software_version=_software_version(),
            modality=_sht_file.MODALITY_EBSD,
            beam_energy=beam_energy,
            primary_angle=sample_tilt,
            doi="",
            notes=(
                "created with kikuchipy "
                f"(normalize={normalize}, "
                f"emsphinx_compatible={emsphinx_compatible})"
            ),
        ),
        num_xtal=1,
        sg_eff=space_group,
        modality=_sht_file.MODALITY_EBSD,
        vendor=_vendor_for(simulation),
        sim_meta_size=_meta_size_for(simulation),
        crystals=[crystal],
        simulations=[simulation],
        harmonics=_sht_file.ShtHarmonics(
            bandwidth=bandwidth,
            z_rot=z_rot,
            flags=flags,
            doub_cnt=_sht_file.num_harmonics(bandwidth, z_rot, flags),
        ),
    )
    return sht_file.metadata_dict()


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
    from diffpy.structure import Atom, Lattice, Structure
    from orix.crystal_map import Phase

    crystal = sht_file.crystals[0]
    atoms = [
        Atom(
            atype=atom.atomic_number,
            xyz=(atom.x / 24, atom.y / 24, atom.z / 24),
            occupancy=atom.occupancy,
            Uisoequiv=atom.debye_waller * 1e2 / (8 * math.pi**2),
        )
        for atom in crystal.atoms
    ]
    return Phase(
        name=crystal.material_name or crystal.formula,
        space_group=sht_file.sg_eff,
        structure=Structure(lattice=Lattice(*crystal.lat), atoms=atoms),
    )


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
    alm = harmonics.alm
    bandwidth = harmonics.bandwidth

    if preserve_header:
        stored = harmonics._sht_file
        if stored is None:
            raise ValueError(
                "preserve_header=True only works for coefficients read "
                "from a file with MasterPatternHarmonics.from_file()"
            )
        sht_file = deepcopy(stored)
        z_rot = sht_file.harmonics.z_rot
        flags = sht_file.harmonics.flags
        if doi is not None:
            sht_file.header.doi = doi
            sht_file.header.doi_len = None
            sht_file.header.doi_bytes = None
        if notes is not None:
            sht_file.header.notes = notes
            sht_file.header.note_len = None
            sht_file.header.notes_bytes = None
    else:
        phase = harmonics.phase
        space_group = _effective_space_group(phase, required=True)
        z_rot = _sht_file.space_group_z_rotation(space_group)
        flags = _sht_file.space_group_compression_flags(space_group)
        metadata = harmonics.original_metadata
        header_node = metadata.get("header", {})
        crystal_node = metadata.get("crystals", {}).get("crystal_0", {})
        simulation = _simulation_from_metadata(metadata)
        if doi is None:
            doi = header_node.get("doi", "")
        if notes is None:
            notes = header_node.get("notes", "created with kikuchipy")
        sht_file = _sht_file.ShtFile(
            header=_sht_file.ShtHeader(
                software_version=_software_version(),
                modality=_sht_file.MODALITY_EBSD,
                beam_energy=float(harmonics.beam_energy or 0.0),
                primary_angle=float(harmonics.sample_tilt or 0.0),
                secondary_angle=float(header_node.get("secondary_angle", 0.0)),
                reserved_param=float(header_node.get("reserved_param", 0.0)),
                doi=doi,
                notes=notes,
            ),
            num_xtal=1,
            sg_eff=space_group,
            modality=_sht_file.MODALITY_EBSD,
            vendor=_vendor_for(simulation),
            sim_meta_size=_meta_size_for(simulation),
            crystals=[_crystal_from_phase(phase, space_group, crystal_node)],
            simulations=[simulation],
        )

    packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
    sht_file.harmonics = _sht_file.ShtHarmonics(
        bandwidth=bandwidth,
        z_rot=z_rot,
        flags=flags,
        doub_cnt=int(packed.size),
        packed=packed,
    )
    return sht_file


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
    packed = _sht_file.pack_harmonics(alm, bandwidth, z_rot, flags)
    unpacked = _sht_file.unpack_harmonics(packed, bandwidth, z_rot, flags)
    dropped = np.abs(alm - unpacked) ** 2
    total = np.abs(alm) ** 2
    # Only the non-negative orders are stored, so every m > 0 row
    # stands for itself and its m < 0 twin
    dropped[1:] *= 2
    total[1:] *= 2
    total_power = total.sum()
    if total_power == 0:
        return 0.0
    return float(dropped.sum() / total_power)


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
    hemi = hemisphere.lower()
    if hemi not in HEMISPHERES:
        options = ", ".join(HEMISPHERES)
        raise ValueError(f"Hemisphere must be one of {options}, not {hemisphere!r}")
    hemisphere = hemi
    bandwidth = harmonics.bandwidth
    if dim is None:
        dim = 2 * bandwidth + 1
    dim = int(dim)
    if dim < 3 or dim % 2 == 0:
        raise ValueError(f"Grid side length {dim} must be odd and at least three")

    band_limit = (dim - 1) // 2
    if band_limit < bandwidth:
        warn(
            f"A square Lambert grid of side length {dim} carries harmonics "
            f"up to degree {band_limit}, so the coefficients of bandwidth "
            f"{bandwidth} are band limited to {band_limit}",
            UserWarning,
        )
    else:
        band_limit = bandwidth
    transform = _sht.SphericalHarmonicTransform(band_limit, "lambert", dim)
    north, south = transform.synthesize(harmonics.alm[:band_limit, :band_limit])

    if hemisphere == "both":
        data = np.stack([north, south])
    elif hemisphere == "upper":
        data = north
    else:
        data = south

    axes = []
    index = 0
    if hemisphere == "both":
        axes.append(
            {
                "size": 2,
                "index_in_array": 0,
                "name": "hemisphere",
                "scale": 1.0,
                "offset": 0.0,
                "units": "",
            }
        )
        index = 1
    for i, name in enumerate(("height", "width")):
        axes.append(
            {
                "size": dim,
                "index_in_array": index + i,
                "name": name,
                "scale": 1.0,
                # Centred on the middle pixel, as in the ebsdsim
                # plugin and one pixel off kikuchipy's EMsoft reader
                "offset": -(dim // 2),
                "units": "px",
            }
        )

    phase = harmonics.phase
    signal_dict = {
        "data": data,
        "axes": axes,
        "metadata": {
            "Signal": {"signal_type": "EBSDMasterPattern", "record_by": "image"},
            "General": {"title": "" if phase is None else phase.name},
        },
        "original_metadata": harmonics.original_metadata,
        "projection": "lambert",
        "hemisphere": hemisphere,
    }
    if phase is not None:
        signal_dict["phase"] = phase
    return signal_dict


def _software_version() -> str:
    """Return the eight character software tag of a written file.

    Returns
    -------
    tag
        ``"kp"`` followed by :data:`kikuchipy.__version__`, truncated
        to eight characters, e.g. ``"kp0.14.d"``.  The writer NUL pads
        it, as SHTfile does.
    """
    import kikuchipy

    return ("kp" + kikuchipy.__version__)[:8]


def _atomic_number(element: Any) -> int:
    """Return the atomic number of an element.

    Parameters
    ----------
    element
        Atomic number or chemical symbol, e.g. 28 or ``"Ni"``.

    Returns
    -------
    atomic_number
        The atomic number.

    Raises
    ------
    ValueError
        If the symbol is unknown.
    """
    if isinstance(element, (int, np.integer)) and not isinstance(element, bool):
        return int(element)
    text = str(element).strip()
    if text.isdigit():
        return int(text)
    # Strip an ionic charge such as the "2+" of "Ni2+"
    symbol = "".join(character for character in text if character.isalpha())
    for i, known in enumerate(_ELEMENT_SYMBOLS):
        if known.lower() == symbol.lower():
            return i + 1
    raise ValueError(f"Cannot determine the atomic number of element {element!r}")


def _effective_space_group(phase: Phase | None, required: bool = False) -> int:
    """Return the effective space group of a phase.

    Parameters
    ----------
    phase
        The phase, or ``None``.
    required
        Whether to raise when the phase has neither a space group nor
        a point group, ``False`` by default, in which case 1 is
        returned.

    Returns
    -------
    space_group
        The phase's space group number, else the lowest space group of
        its point group, see
        :func:`~kikuchipy.indexing._spherical._symmetry.space_group_for_point_group`.

    Raises
    ------
    ValueError
        If ``required`` is ``True`` and the phase has neither.
    """
    if phase is not None:
        if phase.space_group is not None:
            return int(phase.space_group.number)
        if phase.point_group is not None:
            return _symmetry.space_group_for_point_group(phase.point_group.name)
    if required:
        raise ValueError(
            "The phase must have a space group or a point group to write an "
            "EMSphInx *.sht file; set `phase.space_group` or "
            "`phase.point_group`"
        )
    return 1


def _crystal_from_phase(
    phase: Phase | None, space_group: int, stored: dict[str, Any] | None = None
) -> _sht_file.ShtCrystal:
    """Return the crystal block of a phase.

    Parameters
    ----------
    phase
        The phase, or ``None`` for an empty crystal.
    space_group
        Effective space group number, written as the crystal's own.
    stored
        The ``"crystal_0"`` node of a file metadata dictionary, or
        ``None``.  Every field an :class:`~orix.crystal_map.Phase`
        cannot hold, i.e. the space group setting, axis and cell
        choice, the origin shift, the crystal rotation and weight,
        the structure symbol, references and note strings and the
        atom charge, reserved value and reserved bytes, is taken from
        it, so that a file read with
        :meth:`MasterPatternHarmonics.from_file` and written back
        keeps them.

    Returns
    -------
    crystal
        The crystal.

    Notes
    -----
    Ported from ``File::addDataEMsoft`` (``sht_file.in.hpp`` lines
    2139-2230) and ``programs/mp2sht.cpp`` lines 102-115: the lattice
    parameters are written *as stored* on the phase, in nm for
    kikuchipy's EMsoft readers, the fractional atom coordinates are
    given in 24ths with the four exact sixths special cased in 32-bit
    floating point, and the formula is the concatenation of the
    unique element symbols in ascending atomic number.
    """
    stored = {} if stored is None else stored
    stored_atoms = stored.get("atoms", {})
    atoms = []
    atomic_numbers = []
    if phase is not None and phase.structure is not None:
        for i, atom in enumerate(phase.structure):
            atomic_number = _atomic_number(atom.element)
            atomic_numbers.append(atomic_number)
            coordinates = [_coordinate_in_24ths(value) for value in atom.xyz]
            node = stored_atoms.get(f"atom_{i}", {})
            atoms.append(
                _sht_file.ShtAtom(
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    occupancy=float(atom.occupancy),
                    charge=float(node.get("charge", 0.0)),
                    debye_waller=float(atom.Bisoequiv) / 100,
                    res_fp=float(node.get("res_fp", 0.0)),
                    atomic_number=atomic_number,
                    res=tuple(int(value) for value in node.get("res", (0, 0, 0))),
                )
            )
    formula = "".join(
        _ELEMENT_SYMBOLS[number - 1] for number in sorted(set(atomic_numbers))
    )
    if phase is not None and phase.structure is not None:
        lattice = phase.structure.lattice
        lat = (
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
        )
    else:  # pragma: no cover
        lat = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    return _sht_file.ShtCrystal(
        sg_num=space_group,
        sg_set=int(stored.get("sg_set", 1)),
        sg_axis=int(stored.get("sg_axis", 1)),
        sg_cell=int(stored.get("sg_cell", 1)),
        origin=tuple(float(value) for value in stored.get("origin", (0.0,) * 3)),
        lat=tuple(float(value) for value in lat),
        rot=tuple(float(value) for value in stored.get("rot", (1.0, 0.0, 0.0, 0.0))),
        weight=float(stored.get("weight", 1.0)),
        num_atoms=len(atoms),
        atoms=atoms,
        formula=formula,
        material_name="" if phase is None else str(phase.name),
        structure_symbol=str(stored.get("structure_symbol", "")),
        references=str(stored.get("references", "")),
        note=str(stored.get("note", "")),
    )


def _coordinate_in_24ths(value: float) -> float:
    """Return a fractional coordinate in 24ths of a lattice parameter.

    Parameters
    ----------
    value
        Fractional coordinate.

    Returns
    -------
    coordinate
        ``value`` brought into [0, 1) and multiplied by 24, with the
        four exact sixths ``1/6``, ``1/3``, ``2/3`` and ``5/6`` given
        their exact images 4, 8, 16 and 20.

    Notes
    -----
    Ported from ``File::addDataEMsoft`` (``sht_file.in.hpp`` lines
    2160-2169), which does every step in 32-bit floating point.
    """
    x = np.float32(value)
    x = np.fmod(x, np.float32(1))
    if x < 0:
        x = x + np.float32(1)
    one = np.float32(1)
    for numerator, denominator, image in (
        (1, 6, 4.0),
        (1, 3, 8.0),
        (2, 3, 16.0),
        (5, 6, 20.0),
    ):
        if x == np.float32(numerator) * one / np.float32(denominator):
            return image
    return float(x * np.float32(24))


def _vendor_for(simulation: _sht_file.ShtEMsoftSimulation | None) -> int:
    """Return the vendor flag of a file with or without a simulation
    record.

    Parameters
    ----------
    simulation
        The record, or ``None``.

    Returns
    -------
    vendor
        :data:`~kikuchipy.indexing._spherical._sht_file.VENDOR_EMSOFT`
        when there is a record, else the unknown vendor.
    """
    if simulation is None:
        return _sht_file.VENDOR_UNKNOWN
    return _sht_file.VENDOR_EMSOFT


def _meta_size_for(simulation: _sht_file.ShtEMsoftSimulation | None) -> int:
    """Return the simulation record size of a file with or without a
    record.

    Parameters
    ----------
    simulation
        The record, or ``None``.

    Returns
    -------
    size
        88 bytes when there is a record, else 0.
    """
    if simulation is None:
        return 0
    return _sht_file.EMSOFT_ED_SIZE


_EMSOFT_ED_KEYS: tuple[tuple[str, str], ...] = (
    ("sig_start", "MCCLNameList.sig"),
    ("omega", "MCCLNameList.omega"),
    ("kev", "MCCLNameList.EkeV"),
    ("e_hist_min", "MCCLNameList.Ehistmin"),
    ("e_bin_size", "MCCLNameList.Ebinsize"),
    ("depth_max", "MCCLNameList.depthmax"),
    ("depth_step", "MCCLNameList.depthstep"),
    ("c1", "BetheList.c1"),
    ("c2", "BetheList.c2"),
    ("c3", "BetheList.c3"),
    ("sig_db_diff", "BetheList.sgdbdiff"),
    ("d_min", "EBSDMasterNameList.dmin"),
)
"""Floating point fields of an ``EMsoftED`` simulation record and the
original metadata nodes ``mp2sht`` reads them from
(``programs/mp2sht.cpp`` lines 80-100)."""

_EMSOFT_ED_INT_KEYS: tuple[tuple[str, str], ...] = (
    ("num_sx", "MCCLNameList.numsx"),
    ("num_px", "EBSDMasterNameList.npx"),
)
"""Integer fields of an ``EMsoftED`` record and their original
metadata nodes."""


def _simulation_from_master_pattern(
    metadata: Any,
) -> _sht_file.ShtEMsoftSimulation | None:
    """Return the simulation record of a master pattern's original
    metadata.

    Parameters
    ----------
    metadata
        The signal's ``original_metadata``.

    Returns
    -------
    simulation
        The record, or ``None`` when any field is missing, which is
        the case for master patterns without the full EMsoft name
        lists.

    Notes
    -----
    The field mapping of ``programs/mp2sht.cpp`` lines 80-100, with
    the EMsoft version left as ``"unknown"`` because kikuchipy's
    reader keeps no ``EMheader`` group.
    """
    values = {}
    for field, path in _EMSOFT_ED_KEYS:
        value = metadata.get_item(path, None)
        if value is None:
            return None
        values[field] = float(value)
    for field, path in _EMSOFT_ED_INT_KEYS:
        value = metadata.get_item(path, None)
        if value is None:
            return None
        values[field] = int(value)
    n_electrons = metadata.get_item("MCCLNameList.totnum_el", None)
    multiplier = metadata.get_item("MCCLNameList.multiplier", None)
    if n_electrons is None or multiplier is None:
        return None
    return _sht_file.ShtEMsoftSimulation(
        emsoft_version="unknown",
        sig_end=math.nan,
        sig_step=math.nan,
        thickness=math.inf,
        tot_num_el=int(n_electrons) * int(multiplier),
        lat_grid_type=_sht_file.LAT_GRID_LAMBERT,
        **values,
    )


def _simulation_from_metadata(
    metadata: dict[str, Any],
) -> _sht_file.ShtEMsoftSimulation | None:
    """Return the simulation record stored in a file metadata
    dictionary.

    Parameters
    ----------
    metadata
        Dictionary of the shape
        :meth:`~kikuchipy.indexing._spherical._sht_file.ShtFile.metadata_dict`
        gives.

    Returns
    -------
    simulation
        The record, or ``None`` when the dictionary has none or holds
        one this reader does not decode.
    """
    node = metadata.get("simulations", {}).get("simulation_0", None)
    if not isinstance(node, dict):
        return None
    return _sht_file.ShtEMsoftSimulation(**node)


def _format_number(value: Any) -> str:
    """Return a compact string of a number.

    Parameters
    ----------
    value
        The number.

    Returns
    -------
    text
        Integers in full, e.g. ``"2000000000"``, and everything else
        in the general format, e.g. ``"20.1"`` for the 32-bit
        floating point beam energy 20.100000381469727.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return f"{float(value):g}"


def _describe_crystal(crystal: dict[str, Any]) -> list[str]:
    """Return the lines describing one crystal.

    Parameters
    ----------
    crystal
        One ``crystal_<i>`` node of
        :meth:`~kikuchipy.indexing._spherical._sht_file.ShtFile.metadata_dict`.

    Returns
    -------
    lines
        The lines, in the order of ``programs/sht2png.cpp`` lines
        222-241.
    """
    lat = crystal.get("lat", (0,) * 6)
    origin = crystal.get("origin", (0.0, 0.0, 0.0))
    rot = crystal.get("rot", (1.0, 0.0, 0.0, 0.0))
    atoms = crystal.get("atoms", {})
    lines = [
        f"    sg {crystal.get('sg_num', 1)} setting {crystal.get('sg_set', 1)}",
        f"        axis / cell choice: {crystal.get('sg_axis', 1)} / "
        f"{crystal.get('sg_cell', 1)}",
        "        additional origin shift: "
        + ", ".join(_format_number(value) for value in origin),
        "        abc: " + ", ".join(_format_number(value) for value in lat[:3]),
        "        abg: " + ", ".join(_format_number(value) for value in lat[3:]),
        "        rot: " + ", ".join(_format_number(value) for value in rot),
        f"        wgt: {_format_number(crystal.get('weight', 1.0))}",
        f"        frm: `{crystal.get('formula', '')}'",
        f"        nam: `{crystal.get('material_name', '')}'",
        f"        sym: `{crystal.get('structure_symbol', '')}'",
        f"        ref: `{crystal.get('references', '')}'",
        f"        not: `{crystal.get('note', '')}'",
        f"        {len(atoms)} atoms:",
    ]
    for atom in atoms.values():
        values = " ".join(
            _format_number(atom.get(key, 0.0) / divisor)
            for key, divisor in (
                ("x", 24),
                ("y", 24),
                ("z", 24),
                ("occupancy", 1),
                ("charge", 1),
                ("debye_waller", 1),
            )
        )
        lines.append(f"            {atom.get('atomic_number', 0)}: {values}")
    return lines


def _describe_simulation(simulation: Any) -> list[str]:
    """Return the lines describing one simulation record.

    Parameters
    ----------
    simulation
        One ``simulation_<i>`` node of
        :meth:`~kikuchipy.indexing._spherical._sht_file.ShtFile.metadata_dict`,
        or ``None``.

    Returns
    -------
    lines
        The lines, in the order of ``programs/sht2png.cpp`` lines
        246-273, empty when there is no decoded record.
    """
    if not isinstance(simulation, dict):
        return []
    grid = simulation.get("lat_grid_type", _sht_file.LAT_GRID_LAMBERT)
    lines = [f"    emVers   : {simulation.get('emsoft_version', '')}"]
    for label, key in (
        ("sigStart ", "sig_start"),
        ("sigEnd   ", "sig_end"),
        ("sigStep  ", "sig_step"),
        ("omega    ", "omega"),
        ("keV      ", "kev"),
        ("eHistMin ", "e_hist_min"),
        ("eBinSize ", "e_bin_size"),
        ("depthMax ", "depth_max"),
        ("depthStep", "depth_step"),
        ("thickness", "thickness"),
        ("totNumEl ", "tot_num_el"),
        ("numSx    ", "num_sx"),
        ("c1       ", "c1"),
        ("c2       ", "c2"),
        ("c3       ", "c3"),
        ("sigDbDiff", "sig_db_diff"),
        ("dMin     ", "d_min"),
        ("numPx    ", "num_px"),
    ):
        lines.append(f"    {label}: {_format_number(simulation.get(key, 0))}")
    name = _sht_file.LAT_GRID_NAMES.get(grid, "unknown")
    lines.append(f"    latGridType: {name}")
    return lines


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

    +-------------------------+----------+------------------------------+
    | keyword                 | default  | what it does                 |
    +=========================+==========+==============================+
    | ``emsphinx_compatible`` | ``True`` | Reproduces EMSphInx'         |
    |                         |          | normalization: the weighted  |
    |                         |          | mean is divided by the       |
    |                         |          | weight sum of *one*          |
    |                         |          | hemisphere although it sums  |
    |                         |          | over both, so twice the mean |
    |                         |          | is subtracted, and the grid  |
    |                         |          | corners are quartered.       |
    |                         |          | ``a_00`` ends up at about    |
    |                         |          | ``-2.985`` for the Ni        |
    |                         |          | master, 71 % of the total    |
    |                         |          | power.  ``False`` is the     |
    |                         |          | corrected normalization with |
    |                         |          | ``a_00`` about ``-6e-5``;    |
    |                         |          | the two differ only by a     |
    |                         |          | global factor and the DC     |
    |                         |          | term.                        |
    +-------------------------+----------+------------------------------+
    | ``strict``              | ``True`` | :meth:`save` refuses to      |
    |                         |          | write when the file's        |
    |                         |          | compression would drop more  |
    |                         |          | relative power than          |
    |                         |          | ``SYMMETRY_POWER_TOLERANCE`` |
    |                         |          | (1e-8).  ``False`` warns and |
    |                         |          | drops, which is what         |
    |                         |          | EMSphInx does silently.      |
    +-------------------------+----------+------------------------------+

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
        alm = np.asarray(alm)
        if alm.ndim != 2 or alm.shape[0] != alm.shape[1]:
            raise ValueError(
                f"Coefficient shape {alm.shape} must be square and two "
                "dimensional, i.e. (bandwidth, bandwidth)"
            )
        if alm.shape[0] < 1:
            raise ValueError("The bandwidth must be at least one")
        alm = np.array(alm, dtype=np.complex128, order="C")
        # The packer relies on this padding being zero
        if np.any(np.tril(alm, -1) != 0):
            raise ValueError(
                "Coefficients of a degree below their order, i.e. the "
                "lower triangle of alm, must all be zero"
            )

        self.alm = alm
        # A copy, as kikuchipy's readers do, so that the source
        # master pattern and these coefficients cannot mutate each
        # other's phase
        self.phase = None if phase is None else phase.deepcopy()
        self.beam_energy = None if beam_energy is None else float(beam_energy)
        self.sample_tilt = None if sample_tilt is None else float(sample_tilt)
        self.original_metadata = {} if original_metadata is None else original_metadata
        self._sht_file: _sht_file.ShtFile | None = None

        point_group = None if self.phase is None else self.phase.point_group
        n_fold, mirror = _symmetry.point_group_flags(point_group)
        n_fold, mirror, messages = _symmetry.validate_flags(alm, n_fold, mirror)
        for message in messages:
            warn(message, UserWarning)
        self._n_fold = n_fold
        self._has_equatorial_mirror = mirror

    @property
    def bandwidth(self) -> int:
        """Return the bandwidth, i.e. ``alm.shape[0]``."""
        return int(self.alm.shape[0])

    @property
    def n_fold(self) -> int:
        """Return the z rotational order of the coefficients, one of
        1, 2, 3, 4 and 6.

        EMSphInx' ``MasterSpectra::nFold()``.
        """
        return self._n_fold

    @property
    def has_equatorial_mirror(self) -> bool:
        """Return whether the coefficients have a mirror plane
        perpendicular to z.

        EMSphInx' ``MasterSpectra::mirror()``.
        """
        return self._has_equatorial_mirror

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
        from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern

        if not isinstance(master_pattern, EBSDMasterPattern):
            raise TypeError(
                "master_pattern must be an EBSDMasterPattern, not a "
                f"{type(master_pattern).__name__}"
            )
        bandwidth = int(bandwidth)
        if bandwidth < 1 or bandwidth > _sht_file.MAX_BANDWIDTH:
            raise ValueError(
                f"Bandwidth {bandwidth} must be in the closed interval "
                f"[1, {_sht_file.MAX_BANDWIDTH}]"
            )
        if master_pattern.projection != "lambert":
            raise NotImplementedError(
                "Master pattern must be in the square Lambert projection; "
                "use `as_lambert()`"
            )

        phase = master_pattern.phase
        point_group = None if phase is None else phase.point_group
        hemisphere = master_pattern.hemisphere
        if hemisphere not in ("both", "upper"):
            raise ValueError(
                f"Master pattern hemisphere {hemisphere!r} must be 'both', "
                "or 'upper' for a phase whose point group contains "
                "inversion"
            )
        if hemisphere == "upper" and (
            point_group is None or not point_group.contains_inversion
        ):
            name = None if point_group is None else point_group.name
            raise ValueError(
                "Only the upper hemisphere of the master pattern is "
                f"available and its point group {name!r} does not contain "
                "inversion, so the lower hemisphere cannot be built from it"
            )

        _check_atom_sum(master_pattern)
        weights = _energy_weights(master_pattern, energy_weights)

        metadata = master_pattern.original_metadata
        if beam_energy is None:
            beam_energy = metadata.get_item("MCCLNameList.EkeV", None)
            if beam_energy is None:
                raise ValueError(
                    "Cannot find the beam energy in 'MCCLNameList.EkeV' of "
                    "the master pattern's original metadata; pass "
                    "beam_energy"
                )
        if sample_tilt is None:
            sample_tilt = metadata.get_item("MCCLNameList.sig", None)
            if sample_tilt is None:
                raise ValueError(
                    "Cannot find the sample tilt in 'MCCLNameList.sig' of "
                    "the master pattern's original metadata; pass "
                    "sample_tilt"
                )
        beam_energy = float(beam_energy)
        sample_tilt = float(sample_tilt)

        data = np.asarray(master_pattern.data, dtype=np.float64)
        if hemisphere == "both":
            north, south = data[0], data[1]
        else:
            north, south = data, None
        names = [axis.name for axis in master_pattern.axes_manager.navigation_axes]
        if "energy" in names:
            north = np.tensordot(weights, north, axes=(0, 0))
            if south is not None:
                south = np.tensordot(weights, south, axes=(0, 0))
        if south is None:
            # EMSphInx' makeInvSym: f(x, y, -z) == f(-x, -y, z), and
            # both hemispheres are stored at the square coordinates
            # of (x, y, |z|)
            south = np.ascontiguousarray(north[::-1, ::-1])

        dim = north.shape[0]
        band_limit = (dim - 1) // 2
        if bandwidth > band_limit:
            warn(
                f"The requested bandwidth {bandwidth} exceeds {band_limit}, "
                "the largest harmonic degree a square Lambert master "
                f"pattern of side length {dim} carries; the EMSphInx "
                "bandwidth 384 exists for parity with its *.sht files, "
                "never for an accuracy claim",
                UserWarning,
            )

        dim_legendre = _grid.default_dim(bandwidth, "legendre")
        dim_scaled = int(round(math.sqrt(2) * dim_legendre))
        north = _resize_lambert(north, dim_scaled)
        south = _resize_lambert(south, dim_scaled)
        north, south = _to_legendre(north, south, dim_legendre)
        if normalize:
            north, south, _, _ = _normalize_hemispheres(
                north, south, emsphinx_compatible
            )
        # No DC term is removed, EMSphInx' master.hpp line 594 being
        # commented out; remove_dc() is explicit
        alm = _sht.SphericalHarmonicTransform(
            bandwidth, "legendre", dim_legendre
        ).analyze(north, south)

        return cls(
            alm,
            phase=phase,
            beam_energy=beam_energy,
            sample_tilt=sample_tilt,
            original_metadata=_metadata_from_master_pattern(
                master_pattern,
                beam_energy,
                sample_tilt,
                bandwidth,
                normalize,
                emsphinx_compatible,
            ),
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> MasterPatternHarmonics:
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
        sht_file = _sht_file.read_sht(filename)
        modality = sht_file.header.modality
        if modality != _sht_file.MODALITY_EBSD:
            name = _sht_file.MODALITY_NAMES.get(modality, "unknown")
            raise NotImplementedError(
                f"Only EBSD master patterns are read, not {name} ones "
                f"(modality {modality})"
            )
        if sht_file.num_xtal != 1:
            raise NotImplementedError(
                "Only files holding one crystal are read, not this one "
                f"with numXtal {sht_file.num_xtal}"
            )
        harmonics = sht_file.harmonics
        alm = _sht_file.unpack_harmonics(
            harmonics.packed,
            harmonics.bandwidth,
            harmonics.z_rot,
            harmonics.flags,
        )
        instance = cls(
            alm,
            phase=_phase_from_sht(sht_file),
            beam_energy=float(sht_file.header.beam_energy),
            sample_tilt=float(sht_file.header.primary_angle),
            original_metadata=sht_file.metadata_dict(),
        )
        instance._sht_file = sht_file
        return instance

    def save(
        self,
        filename: str | Path,
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

        Without ``preserve_header`` the file is rebuilt from the
        phase, which cannot hold every field the format has.  Those
        fields, i.e. the secondary angle and reserved parameter of
        the header, the space group setting, axis and cell choice,
        the origin shift, rotation and weight of the crystal, its
        structure symbol, references and note strings and the charge,
        reserved value and reserved bytes of each atom, are carried
        over from :attr:`original_metadata` when the coefficients
        were read from a file, and are otherwise the format's
        defaults.  Raw string bytes are *not* preserved, so a file
        whose text is not valid UTF-8 or whose padding is non-zero
        needs ``preserve_header=True`` to come back byte identical.

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
        from kikuchipy.io._util import _ensure_directory, _overwrite

        filename = str(filename)
        if os.path.splitext(filename)[1] == "":
            filename += ".sht"

        sht_file = _sht_from_harmonics(self, doi, notes, preserve_header)
        harmonics = sht_file.harmonics
        loss = _packing_loss(self.alm, self.bandwidth, harmonics.z_rot, harmonics.flags)
        if loss > _symmetry.SYMMETRY_POWER_TOLERANCE:
            message = (
                f"A relative power of {loss:.3e} of the coefficients is "
                "dropped by the compression of effective space group "
                f"{sht_file.sg_eff}, more than the tolerance "
                f"{_symmetry.SYMMETRY_POWER_TOLERANCE}: the coefficients do "
                "not have the symmetry that space group implies in the SHT "
                "file's standard setting, which is unique axis b for the "
                "monoclinic groups"
            )
            candidates = ()
            point_group = None if self.phase is None else self.phase.point_group
            if point_group is not None:
                candidates = _symmetry.candidate_space_groups(point_group.name)
            if len(candidates) > 1:
                listed = " or ".join(str(number) for number in candidates)
                message += (
                    f". The point group {point_group.name!r} maps to space "
                    f"groups with different compression: set "
                    f"`phase.space_group` to {listed}"
                )
            if strict:
                raise ValueError(
                    message + ". Pass strict=False to write the file anyway"
                )
            warn(message + ". They are dropped", UserWarning)

        _ensure_directory(filename)
        is_file = os.path.isfile(filename)
        if overwrite is None:
            write = _overwrite(filename)
        elif overwrite is True or (overwrite is False and not is_file):
            write = True
        elif overwrite is False and is_file:
            write = False
        else:
            raise ValueError(
                f"overwrite can only be None, True or False, and not {overwrite}"
            )
        if write:
            _sht_file.write_sht(filename, sht_file)

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
        from kikuchipy.io._io import _dict2signal

        signal_dict = _master_pattern_dict(self, dim, hemisphere)
        return _dict2signal(signal_dict, lazy=False)

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
        bandwidth = int(bandwidth)
        if bandwidth < 1:
            raise ValueError(f"Bandwidth {bandwidth} must be at least one")
        alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
        kept = min(bandwidth, self.bandwidth)
        alm[:kept, :kept] = self.alm[:kept, :kept]
        metadata = deepcopy(self.original_metadata)
        node = metadata.get("harmonics", None)
        if node is not None:
            node["bandwidth"] = bandwidth
            node["doub_cnt"] = _sht_file.num_harmonics(
                bandwidth, node.get("z_rot", 1), node.get("flags", 0)
            )
        return type(self)(
            alm,
            phase=self.phase,
            beam_energy=self.beam_energy,
            sample_tilt=self.sample_tilt,
            original_metadata=metadata,
        )

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
        alm = self.alm.copy()
        alm[0, 0] = 0
        return type(self)(
            alm,
            phase=self.phase,
            beam_energy=self.beam_energy,
            sample_tilt=self.sample_tilt,
            original_metadata=deepcopy(self.original_metadata),
        )

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
        raise NotImplementedError(
            "Rotating spherical harmonic coefficients requires the Wigner-d "
            "tables of Phase 3 (sht-wigner-d)"
        )

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
        power = np.abs(self.alm) ** 2
        # Only the non-negative orders are stored, so every m > 0 row
        # stands for itself and its m < 0 twin
        power[1:] *= 2
        return power.sum(axis=0)

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
        metadata = self.original_metadata
        header = metadata.get("header", {})
        master = metadata.get("master_pattern", {})
        harmonics = metadata.get("harmonics", {})
        version = tuple(header.get("file_version", _sht_file.VERSION))
        modality = header.get("modality", _sht_file.MODALITY_EBSD)
        vendor = master.get("vendor", _sht_file.VENDOR_UNKNOWN)
        # The rotation sense is a signed byte on disk and is not
        # sanity checked on read, so it need not be a code point
        rot_sense = master.get("rot_sense", 112)
        sense = chr(rot_sense) if 0 <= rot_sense < 0x110000 else "?"
        lines = [
            f"file version {version[0]}.{version[1]}",
            f"software version {header.get('software_version', '')}",
            f"modality: {_sht_file.MODALITY_NAMES.get(modality, 'invalid')}",
            f"beam energy: {_format_number(header.get('beam_energy', 0.0))} keV",
            f"angle 1: {_format_number(header.get('primary_angle', 0.0))} deg",
            f"angle 2: {_format_number(header.get('secondary_angle', 0.0))} deg",
            f"reserved: {_format_number(header.get('reserved_param', 0.0))}",
            f"notes: `{header.get('notes', '')}'",
            f"doi: `{header.get('doi', '')}'",
            "",
            f"master pattern composed from {master.get('num_xtal', 1)} "
            f"crystals with effective sg# {master.get('sg_eff', 1)}",
            f"rotations are {sense} with pijk = {master.get('pijk', 1)}",
            f"simulation data {master.get('sim_meta_size', 0)} bytes from "
            f"vendor {_sht_file.VENDOR_NAMES.get(vendor, 'invalid')} for "
            f"modality {_sht_file.MODALITY_NAMES.get(modality, 'invalid')}",
        ]
        crystals = metadata.get("crystals", {})
        simulations = metadata.get("simulations", {})
        for i, crystal in enumerate(crystals.values()):
            lines += _describe_crystal(crystal)
            lines += _describe_simulation(simulations.get(f"simulation_{i}"))

        power = self.power_spectrum()
        total = float(power.sum())
        fraction = 0.0 if total == 0 else float(power[0]) / total
        z_rot = harmonics.get("z_rot", 1)
        flags = harmonics.get("flags", 0)
        lines += [
            "",
            "harmonics",
            f"    bandwidth {self.bandwidth}",
            f"    zRot {z_rot}",
            f"    cmpFlg {flags:#x}",
            "    doubCnt "
            + str(
                harmonics.get(
                    "doub_cnt",
                    _sht_file.num_harmonics(self.bandwidth, z_rot, flags),
                )
            ),
            f"    n_fold {self.n_fold}",
            f"    equatorial mirror {self.has_equatorial_mirror}",
            f"    a_00 = {self.alm[0, 0].real:.3f} (DC power fraction {fraction:.2f})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a string with the bandwidth, phase, beam energy and
        sample tilt, e.g. ``"MasterPatternHarmonics: bw = 384, ni
        (m-3m), 20.1 keV, 70.0 deg"``.
        """
        phase = self.phase
        name = None if phase is None else phase.name
        point_group = None if phase is None else phase.point_group
        group_name = None if point_group is None else point_group.name
        energy = "None" if self.beam_energy is None else f"{self.beam_energy:.1f}"
        tilt = "None" if self.sample_tilt is None else f"{self.sample_tilt:.1f}"
        return (
            f"{type(self).__name__}: bw = {self.bandwidth}, {name} "
            f"({group_name}), {energy} keV, {tilt} deg"
        )
