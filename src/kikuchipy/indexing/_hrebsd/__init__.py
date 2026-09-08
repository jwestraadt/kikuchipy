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

"""Private tools for high angular resolution EBSD (HREBSD) by
inverse-compositional Gauss-Newton digital image correlation
(IC-GN DIC) of a homography shape function.

This module and documentation is only relevant for kikuchipy
developers, not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with
    no warning.

The conventions of every submodule are re-derived from the published
equations of Ernould et al. (Acta Mater. 191 (2020) 131-148; AIEP 223
(2022) Ch. 2) and Ruggles et al. (Ultramicroscopy 195 (2018) 85;
210 (2020) 112927), and are pinned by the synthetic oracles of
``specs/2026-09-07-hrebsd-dic/validation.md``.  EMsoftOO's
``EMHREBSDDIC`` (``mod_DIC.f90``, ``mod_HREBSDDIC.f90``, BSD-3) and
OpenXY (GPL-2.0) are equation-level cross-references only: no code is
ported from either, and their recorded quirks (inverted-diagonal
shape function with a global sign flip, hardcoded 70 degree sample
tilt, mixed pixel/normalized units, ``Fe`` built from uncorrected
homographies, zeroed non-converged points, progressive warp-of-warp)
are deviations recorded in the spec, never reproduced.

Submodules
----------
``_engine``
    Per-reference precompute, the IC-GN loop, the phase
    cross-correlation initial guess and the map orchestration behind
    :meth:`kikuchipy.signals.EBSD.hrebsd_dic`.
``_geometry``
    Projection centres in binned pixels, per-point PC/DD, the
    beam-scan phantom homography and its removal before the
    homography to ``Fe`` conversion.
``_homography``
    The 8 degree-of-freedom homography shape function, its group
    algebra (compose, invert, project, corner norm) and the exact
    homography to elastic deformation gradient conversion.
``_interpolation``
    Cubic B-spline coefficients and the hand written numba
    evaluation and analytic gradient kernels.
``_kam``
    The frozen high angular resolution kernel average
    misorientation behind :func:`kikuchipy.indexing.hrebsd_kam`.
``_pc_shift``
    Measured against modelled beam-scan projection centre shifts,
    the diagnostic behind
    :func:`kikuchipy.indexing.hrebsd_pc_shift`.
``_preprocessing``
    In-engine band-pass filtering, the optional Hann window, the
    subregion (border plus dead band) and zero-mean normalization.
``_reference``
    Resolution of the ``reference`` argument into a per-point grain
    identifier and reference pattern index.
``_segmentation``
    Grain segmentation by neighbour misorientation and the per-grain
    image-quality reference selection behind
    :func:`kikuchipy.indexing.segment_grains` and
    ``reference="auto"``.
``_stiffness``
    The Voigt stiffness convention, its builder
    :func:`kikuchipy.indexing.voigt_stiffness`, the crystal to sample
    rotation and the Hooke product.
``_tensors``
    The one tensor chain -- frame, ninth degree of freedom closure,
    polar split, stress and derived maps -- behind
    :func:`kikuchipy.indexing.hrebsd_strain_stress`.

Nothing is imported here on purpose: each submodule is imported
directly by the code that needs it, keeping the import cost of
:mod:`kikuchipy.indexing` unchanged.
"""
