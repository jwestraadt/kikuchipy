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

"""Private tools for spherical harmonic (spherical indexing)
transforms on square grids on the unit sphere.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

Submodules
----------
``_back_projection``
    Back-projection of detector patterns onto the square Legendre
    grid through kikuchipy's detector geometry.
``_euler``
    ZYZ Euler angle conversions and the beta wrap.
``_fft``
    Fast FFT sizes and the bandwidths that use them.
``_grid``
    Square Lambert and square Legendre grids on the unit sphere.
``_indexer``
    The spherical indexer: per-pattern pipeline, multi-phase top-n
    bookkeeping, dask chunking.
``_master_pattern_harmonics``
    Spherical harmonic coefficients of an EBSD master pattern,
    read from and written to EMSphInx *.sht files.
``_namelist``
    The EMSphInx ``IndexEBSD`` namelist file: its grammar, its EBSD
    fields and the conversions to kikuchipy's arguments.
``_pattern_repack``
    Writing of EBSD patterns to the HDF5 file EMSphInx's
    ``IndexEBSD`` reads.
``_preprocessing``
    EMSphInx pattern preprocessing: Gaussian background, mosaic
    adaptive histogram equalisation.
``_sht``
    Discrete spherical harmonic transform on those grids.
``_sht_file``
    The EMSphInx *.sht file format, version 1.1 of SHTfile.  This
    module is BSD-3-Clause licensed, unlike the rest of this
    package.
``_symmetry``
    Symmetry of the coefficients of a master pattern.
``_wigner``
    Wigner (lowercase) d functions, tables and harmonic rotation.
``_xcorr``
    SO(3) cross-correlation of two spherical functions from their
    harmonic coefficients, and its peak.

Nothing is imported here on purpose: each submodule is imported
directly by the code that needs it, keeping the import cost of
:mod:`kikuchipy.indexing` unchanged.
"""
