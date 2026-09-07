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

"""The homography shape function and its algebra.

The 8 degree-of-freedom homography
``h = (h11, h12, h13, h21, h22, h23, h31, h32)`` is the textbook
shape function of requirements D2::

    W(h) = [ 1 + h11   h12      h13 ]
           [ h21       1 + h22  h23 ]
           [ h31       h32      1   ]

acting projectively on the PC-centred pixel coordinates of D1:
``(x', y', s) = W . (xi_x, xi_y, 1)`` and ``xi' = (x'/s, y'/s)``.

**The fitted homography maps reference coordinates to target
coordinates.**  The converged solution satisfies
``target(xi') ~= reference(xi)`` after zero-mean normalization, which
is the direction the ``homography <-> Fe`` derivation assumes, so
there is no post-fit sign flip anywhere in this package.  EMsoftOO's
inverted-diagonal shape function with its global ``h <- -h``
(``mod_DIC.f90:888-934``, ``mod_HREBSDDIC.f90:902``) is a recorded
deviation, never reproduced.  The direction is pinned in both
compositions once by validation V1 and V2, then frozen.

Every coordinate here is a BINNED DETECTOR PIXEL, PC-centred on the
grain reference's projection centre (requirements D1.3).  There is no
normalized [0, 1] coordinate anywhere, another recorded deviation
from EMsoftOO (``mod_DIC.f90:188-201``): pixels make Ernould's
0.001 px convergence criterion native and remove the unit trap.

References
----------
Ernould et al., Acta Mater. 191 (2020) 131-148; Ernould et al., AIEP
223 (2022) Ch. 2; Pan, Meas. Sci. Technol. 29 (2018) 082001.
"""

import numpy as np

# The number of free parameters of the shape function, frozen
N_HOMOGRAPHY_PARAMETERS: int = 8

# The identity homography, the zero vector of the parameterization
IDENTITY_HOMOGRAPHY: tuple[float, ...] = (0.0,) * N_HOMOGRAPHY_PARAMETERS


def shape_function(h: np.ndarray) -> np.ndarray:
    """Return the 3 by 3 shape function matrix of *h*.

    Parameters
    ----------
    h
        Homography parameters of shape ``(8,)``.

    Returns
    -------
    matrix
        Array of shape ``(3, 3)`` and 64-bit float data type, with
        ``matrix[2, 2]`` exactly one.

    Raises
    ------
    ValueError
        If *h* does not hold exactly
        :data:`N_HOMOGRAPHY_PARAMETERS` entries.
    """
    raise NotImplementedError


def homography_parameters(matrix: np.ndarray) -> np.ndarray:
    """Return the 8 parameters of a shape function matrix.

    The inverse of :func:`shape_function`, including the ``W[2, 2]``
    renormalization the IC-GN update applies after every composition
    (requirements D2.3).

    Parameters
    ----------
    matrix
        Array of shape ``(3, 3)``. It is divided by ``matrix[2, 2]``
        first, so any nonzero scaling of a homography gives the same
        parameters.

    Returns
    -------
    h
        Array of shape ``(8,)`` and 64-bit float data type.

    Raises
    ------
    ValueError
        If *matrix* is not 3 by 3, or if ``matrix[2, 2]`` is zero.
    """
    raise NotImplementedError


def compose(h_left: np.ndarray, h_right: np.ndarray) -> np.ndarray:
    """Return the parameters of ``W(h_left) . W(h_right)``.

    Matrix order, not a commuting operation: the right factor acts on
    the coordinates first.  The inverse-compositional update of
    requirements D2.3 is ``compose(h, invert(dp))``, never
    ``compose(invert(dp), h)``.

    Parameters
    ----------
    h_left
        Homography parameters of shape ``(8,)``.
    h_right
        Homography parameters of shape ``(8,)``.

    Returns
    -------
    h
        Renormalized parameters of shape ``(8,)``.
    """
    raise NotImplementedError


def invert(h: np.ndarray) -> np.ndarray:
    """Return the parameters of ``W(h)**-1``.

    Parameters
    ----------
    h
        Homography parameters of shape ``(8,)``.

    Returns
    -------
    h_inverse
        Renormalized parameters of shape ``(8,)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``W(h)`` is singular.
    """
    raise NotImplementedError


def project(
    h: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the projective image of ``(x, y)`` under ``W(h)``.

    Parameters
    ----------
    h
        Homography parameters of shape ``(8,)``.
    x
        Column coordinates in PC-centred binned pixels, of any shape.
    y
        Row coordinates of the shape of *x*.

    Returns
    -------
    x_warped
        ``x'/s`` of the shape of *x* and 64-bit float data type.
    y_warped
        ``y'/s`` of the shape of *x*.

    Notes
    -----
    The projective divide by ``s = h31*x + h32*y + 1`` is what makes
    this a homography rather than an affinity, and dropping it is one
    of the mutants the oracles kill.
    """
    raise NotImplementedError


def corner_norm(h: np.ndarray, corners: np.ndarray) -> float:
    """Return the maximum corner displacement induced by ``W(h)``.

    The frozen convergence norm of requirements D2.5:
    ``max_corners |proj(W(h), xi_c) - xi_c|_2`` in binned pixels.  It
    is the quantity both published criteria approximate, and being a
    length it is the only unit-consistent scalar over all eight
    degrees of freedom, which mix dimensionless, pixel and inverse
    pixel units.

    Parameters
    ----------
    h
        Homography parameters of shape ``(8,)``.
    corners
        Array of shape ``(4, 2)`` of the subregion corner coordinates
        ``(x, y)`` in PC-centred binned pixels.

    Returns
    -------
    norm
        The maximum displacement in binned pixels.
    """
    raise NotImplementedError


def error_norm(h_fit: np.ndarray, h_true: np.ndarray, corners: np.ndarray) -> float:
    """Return the corner norm of the error warp between two fits.

    The recovery metric of validation V2:
    ``corner_norm(compose(invert(h_true), h_fit), corners)``, one
    number in binned pixels.  A raw ``max|h_fit - h_true|`` is not
    used anywhere because it mixes units across the eight degrees of
    freedom.

    Parameters
    ----------
    h_fit
        Fitted homography parameters of shape ``(8,)``.
    h_true
        Imposed homography parameters of shape ``(8,)``.
    corners
        Array of shape ``(4, 2)`` of subregion corners ``(x, y)``.

    Returns
    -------
    norm
        The error warp's maximum corner displacement in binned
        pixels, zero for an exact recovery.
    """
    raise NotImplementedError


def homography_to_fe(h: np.ndarray, pc_rel: np.ndarray, dd: float) -> np.ndarray:
    """Return the reduced elastic deformation gradient of *h*.

    The exact conversion of requirements D6 in PC-centred binned
    pixels::

        beta0 = 1 - h31*PCx_rel - h32*PCy_rel
        Fe = (1/beta0) * [ 1 + h11    h12       h13/DD ]
                         [ h21        1 + h22   h23/DD ]
                         [ DD*h31     DD*h32    beta0  ]

    Parameters
    ----------
    h
        Homography parameters of shape ``(8,)``.
    pc_rel
        ``(PCx, PCy)`` of the converting pattern RELATIVE to the
        frame origin, in binned pixels, which is
        ``PC_target - PC_reference``. It is ``(0, 0)`` for every
        single-PC run and for the grain reference itself.
    dd
        Detector distance ``DD_px = pcz * nrows`` in binned pixels.

    Returns
    -------
    fe
        Array of shape ``(3, 3)`` and 64-bit float data type, the
        REDUCED tensor ``Fe_hat = Fe / Fe33``, which has ``Fe[2, 2]``
        equal to one by construction. The ninth degree of freedom is
        unobservable from a single projection and its closure is a
        Stage B decision (requirements D9).

    Raises
    ------
    ValueError
        If *dd* is not positive, or if ``beta0`` is zero.

    Notes
    -----
    The two dimensional analogue lives in EMsoftOO's
    ``homography2Fe_`` (``mod_DIC.f90:970-1040``) in corner-origin
    coordinates; the equations agree, the coordinate origin does not.
    """
    raise NotImplementedError


def fe_to_homography(fe: np.ndarray, pc_rel: np.ndarray, dd: float) -> np.ndarray:
    """Return the homography of a reduced deformation gradient.

    The exact inverse of :func:`homography_to_fe`, needed by every
    synthetic oracle: the round trip is exact to 1e-12 in both
    directions (validation V1, a machine-precision class identity
    with no measured tolerance).

    Parameters
    ----------
    fe
        Array of shape ``(3, 3)``. It is divided by ``fe[2, 2]``
        first, so any scaling of the tensor gives the same
        homography.
    pc_rel
        ``(PCx, PCy)`` of the converting pattern relative to the
        frame origin, in binned pixels.
    dd
        Detector distance in binned pixels.

    Returns
    -------
    h
        Homography parameters of shape ``(8,)`` and 64-bit float data
        type.

    Raises
    ------
    ValueError
        If *fe* is not 3 by 3, if *dd* is not positive, or if
        ``fe[2, 2]`` is zero.
    """
    raise NotImplementedError
