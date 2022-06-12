# Copyright 2019-2022 The kikuchipy developers
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

from typing import Optional, Tuple, Union

import dask.array as da
from dask.diagnostics import ProgressBar
from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.pyplot as plt
import numpy as np
from orix import projections
from orix.crystal_map import Phase
from orix.plot._util import Arrow3D
from orix.quaternion import Rotation
from orix.vector import Miller, Vector3d

from kikuchipy.detectors import EBSDDetector
from kikuchipy.simulations._kikuchi_pattern_simulation import (
    GeometricalKikuchiPatternSimulation,
)
from kikuchipy.simulations._kikuchi_pattern_features import (
    KikuchiPatternLine,
    KikuchiPatternZoneAxis,
)
from kikuchipy.signals import EBSDMasterPattern


class KikuchiPatternSimulator:
    """Setup and calculation of Kikuchi pattern simulations.

    Parameters
    ----------
    reflectors : ReciprocalLatticeVector
        Reflectors to use in the simulation, flattened to one navigation
        dimension.

    """

    def __init__(self, reflectors: ReciprocalLatticeVector):
        self._reflectors = reflectors.deepcopy().flatten()

    @property
    def reflectors(self) -> ReciprocalLatticeVector:
        """Reflectors to use in the simulation."""
        return self._reflectors

    @property
    def phase(self) -> Phase:
        """Phase with unit cell and symmetry."""
        return self._reflectors.phase

    def __repr__(self) -> str:
        """String representation."""
        ref = self.reflectors
        name = self.__class__.__name__
        shape = ref.shape
        symmetry = self.phase.point_group.name
        data = np.array_str(ref.coordinates, precision=0, suppress_small=True)
        phase_name = self.phase.name
        return f"{name} {shape}, {phase_name} ({symmetry})\n" f"{data}"

    def calculate_master_pattern(
        self,
        half_size: Optional[int] = 500,
        hemisphere: Optional[str] = "north",
        scaling: Optional[str] = "linear",
    ) -> EBSDMasterPattern:
        r"""Calculate a kinematical master pattern in the stereographic
        projection.

        Requires that the :attr:`reflectors` have structure factors
        (:attr:`~diffsims.crystallography.ReciprocalLatticeVector.structure_factor)
        and Bragg angles
        (:attr:`~diffsims.crystallography.ReciprocalLatticeVector.theta`)
        calculated.

        Parameters
        ----------
        half_size
            Number of pixels along the x-direction of the square master
            pattern. Default is ``500``. The full size will be
            ``2 * half_size + 1``, given a master pattern of shape
            (1001, 1001) for the default value.
        hemisphere
            Which hemisphere(s) to calculate. Options are ``"north"``
            (default), ``"south"`` or ``"both"``.
        scaling
            Intensity scaling of the band kinematical intensities,
            either ``"linear"`` (default), :math:`|F|`, ``"square"``,
            :math:`|F|^2`, or ``"None"``, giving all bands an intensity
            of ``1``.

        Returns
        -------
        master_pattern
            Kinematical master pattern in the stereographic projection.
        """
        self._raise_if_no_theta()
        self._raise_if_no_structure_factor()

        size = int(2 * half_size + 1)

        # Which hemisphere(s) to calculate
        if hemisphere == "both":
            poles = [-1, 1]
        elif hemisphere == "north":
            poles = [-1]
        elif hemisphere == "south":
            poles = [1]
        else:
            raise ValueError(
                "Unknown `hemisphere`, valid options are 'north', 'south' or 'both'"
            )

        if scaling == "linear":
            intensity = abs(self.reflectors.structure_factor)
        elif scaling == "square":
            factor = self.reflectors.structure_factor
            intensity = abs(factor * factor.conjugate())
        elif scaling is None:
            intensity = np.ones(self.reflectors.size)
        else:
            raise ValueError(
                "Unknown `scaling`, valid options are 'linear', 'square' or None"
            )

        # Get Dask arrays of reflector information
        intensity = da.from_array(intensity).astype(np.float64)
        theta1 = da.from_array((np.pi / 2) - self.reflectors.theta)
        theta2 = da.from_array(np.pi / 2)
        xyz_ref = da.from_array(Vector3d(self.reflectors).unit.data)

        # Stereographic coordinates (X, Y) for a square encompassing the
        # stereographic projection and outside (in the corners)
        arr = np.linspace(-1, 1, size)
        x, y = np.meshgrid(arr, arr)

        # Loop over hemisphere(s)
        master_pattern_da = []
        for i in range(len(poles)):
            # 3D coordinates (x, y, z) of unit vectors on the
            # half-sphere
            stereo2sphere = projections.InverseStereographicProjection(poles[i])
            v_hemi = stereo2sphere.xy2vector(x.ravel(), y.ravel()).flatten()
            xyz_hemi = da.from_array(v_hemi.data.squeeze(), chunks=(half_size, -1))

            # Angles between vectors and Kikuchi lines
            dp = da.einsum("ij,kj->ik", xyz_hemi, xyz_ref)
            angles = da.arccos(dp)

            # Exclude Kikuchi bands with:
            #   1. Dot products lower than a threshold
            #   2. Angles outside a vector
            mask1 = da.absolute(dp) <= 1e-7
            mask2 = da.logical_and(
                da.logical_and(angles >= theta1, angles <= theta2), ~mask1
            )

            # Generate master pattern on this hemisphere, chunk by chunk
            pattern = da.map_blocks(
                _get_pattern, intensity, mask1, mask2, drop_axis=1, dtype=np.float64
            )
            master_pattern_da.append(pattern)

        if hemisphere == "both":
            master_pattern = np.zeros((2, size * size))
            shape_out = (2, size, size)
        else:
            master_pattern_da = master_pattern_da[0]
            master_pattern = np.zeros((size * size,))
            shape_out = (size, size)

        with ProgressBar():
            da.store(master_pattern_da, master_pattern)

        master_pattern = master_pattern.reshape(shape_out)

        return EBSDMasterPattern(
            master_pattern,
            phase=self.phase,
            hemisphere=hemisphere,
            projection="stereographic",
            mode="kinematical",
        )

    def on_detector(
        self, detector: EBSDDetector, rotations: Rotation
    ) -> GeometricalKikuchiPatternSimulation:
        """Project Kikuchi lines and zone axes onto a detector, one per
        crystal orientation.

        Parameters
        ----------
        detector
            EBSD detector describing the detectors view of the sample.
        rotations
            Crystal orientations assumed to be expressed with respect to
            the default EDAX TSL sample reference frame RD-TD-ND in the
            Bunge convention.

        Returns
        -------
        simulations
        """
        lattice = self.phase.structure.lattice
        ref = self._reflectors
        hkl = ref.hkl
        #        hkl = ref.hkl.round().astype(int)

        # Transformation from detector reference frame CSd to sample
        # reference frame CSs
        total_tilt = np.deg2rad(detector.sample_tilt - 90 - detector.tilt)
        u_s_bruker = Rotation.from_axes_angles((-1, 0, 0), total_tilt)
        u_s = Rotation.from_axes_angles((0, 0, -1), -np.pi / 2) * u_s_bruker
        u_s = u_s.to_matrix().squeeze()
        # Transformation from CSs to cartesian crystal reference frame
        # CSc
        u_o = rotations.to_matrix()
        # Transform rotations once
        u_o = da.from_array(u_o)
        u_s = da.from_array(u_s)
        u_os = da.matmul(u_o, u_s)
        # Transformation from CSc to reciprocal crystal reference frame
        # CSk*
        u_astar = lattice.recbase.T
        u_astar = da.from_array(u_astar)

        # Combine transformations
        u_kstar = da.matmul(u_astar, u_os)

        # Transform {hkl} from CSk* to CSd
        hkl_da = da.from_array(hkl)
        hkl_d = da.matmul(hkl_da, u_kstar)

        # Get vectors that are in some pattern
        nav_axes = (0, 1)[: rotations.ndim]
        hkl_is_upper, hkl_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=hkl_d[..., 2], navigation_axes=nav_axes
        )
        hkl_in_pattern = hkl_is_upper[..., hkl_in_a_pattern]
        hkl_d = hkl_d[..., hkl_in_a_pattern, :]
        hkl_in_a_pattern = hkl_in_a_pattern.compute()

        # Visible reflectors
        visible_reflectors = self._reflectors[hkl_in_a_pattern]
        hkl = hkl[hkl_in_a_pattern]

        # Max. gnomonic radius to consider
        max_r_gnomonic = detector.r_max[0]

        # Zone axes <uvw> from {hkl}
        uvw = np.cross(hkl[:, None, :], hkl)
        uvw = uvw[~np.isclose(uvw, 0).all(axis=-1)]
        uvw = np.unique(uvw, axis=0)

        # Reduce an index triplet to smallest integer
        uvw = Miller(uvw=uvw, phase=self.phase)
        uvw = uvw.round().unique()
        uvw = uvw.coordinates

        # Transformation from CSc to direct crystal reference frame CSk
        u_a = lattice.base
        u_a = da.from_array(u_a)

        # Combine transformations
        u_k = da.matmul(u_a, u_os)

        # Transform direct lattice vectors from CSk to CSd
        uvw_da = da.from_array(uvw)
        uvw_d = da.matmul(uvw_da, u_k)

        # Get vectors that are in some pattern
        uvw_is_upper, uvw_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=uvw_d[..., 2], navigation_axes=nav_axes
        )
        uvw_in_pattern = uvw_is_upper[..., uvw_in_a_pattern]
        uvw_d = uvw_d[..., uvw_in_a_pattern, :]
        uvw_in_a_pattern.compute()

        # Visible zone axes
        uvw = uvw[uvw_in_a_pattern]

        with ProgressBar(minimum=1):
            hkl_d, hkl_in_pattern, uvw_d, uvw_in_pattern = da.compute(
                [hkl_d, hkl_in_pattern, uvw_d, uvw_in_pattern]
            )[0]

        lines = KikuchiPatternLine(
            hkl=Miller(hkl=hkl, phase=self.phase),
            hkl_detector=Vector3d(hkl_d),
            in_pattern=hkl_in_pattern,
            max_r_gnomonic=max_r_gnomonic,
        )

        zone_axes = KikuchiPatternZoneAxis(
            uvw=Miller(uvw=uvw, phase=self.phase),
            uvw_detector=Vector3d(uvw_d),
            in_pattern=uvw_in_pattern,
            max_r_gnomonic=max_r_gnomonic,
        )

        simulation = GeometricalKikuchiPatternSimulation(
            detector=detector,
            rotations=rotations,
            reflectors=visible_reflectors,
            lines=lines,
            zone_axes=zone_axes,
        )

        return simulation

    def plot(
        self,
        projection: Optional[str] = "stereographic",
        mode: Optional[str] = "lines",
        hemisphere: Optional[str] = "north",
        figure: Optional[plt.Figure] = None,
        return_figure: bool = False,
        backend: str = "matplotlib",
    ) -> Union[plt.Figure, "pyvista.Plotter"]:
        """Plot reflectors as lines or bands in the stereographic or
        spherical projection.

        Parameters
        ----------
        projection
            Either ``"stereographic"`` (default) or ``"spherical"``.
        mode
            Either ``"lines"`` (default) or ``"bands"``. The latter
            option requires that :attr:`reflectors` have Bragg angles
            (:attr:`~diffsims.crystallography.ReciprocalLatticeVector.theta`)
            calculated.
        hemisphere
            Which hemisphere to plot when
            ``projection="stereographic"``. Options are ``"north"``
            (default), ``"south"`` or ``"both"``. Ignored if ``figure``
            is given.
        figure
            An existing :class:`~matplotlib.figure.Figure` to add the
            reflectors to. If not given, a new figure is created.
        return_figure
            Whether to return the figure. Default is ``False``. This is
            a :class:`~matplotlib.figure.Figure` if
            ``backend=="matplotlib"`` or a :class:`~pyvista.Plotter` if
            ``backend=="pyvista"``.
        backend
            Which plotting library to use when
            ``projection="spherical"``, either ``"matplotlib"``
            (default) or ``"pyvista"``. The latter option requires that
            PyVista is installed.

        Returns
        -------
        figure
            If ``return_figure=True``, a
            :class:`~matplotlib.figure.Figure` or a
            :class:`~pyvista.Plotter` is returned.
        """
        ref = self._reflectors

        allowed_modes = ["lines", "bands"]
        if mode not in allowed_modes:
            raise ValueError(f"`mode` must be either of {allowed_modes}")
        if mode == "bands" and ref.theta[0] is None:
            raise ValueError(
                "Requires that reflectors have Bragg angles calculated with "
                "`self.reflectors.calculate_theta()`."
            )

        f_hkl = abs(ref.structure_factor)
        color = f_hkl / f_hkl.max()
        color = abs(color - color.min() - color.max())
        color = np.full((ref.size, 3), color[:, np.newaxis])

        if projection == "stereographic":
            kwargs = dict(color=color, linewidth=0.5)
            if mode == "lines":
                if figure is not None:
                    ref.draw_circle(figure=figure, **kwargs)
                else:
                    figure = ref.draw_circle(
                        hemisphere=hemisphere, return_figure=True, **kwargs
                    )
            else:  # bands
                v = Vector3d(ref)
                theta = ref.theta
                if figure is not None:
                    v.draw_circle(
                        opening_angle=np.pi / 2 - theta, figure=figure, **kwargs
                    )
                else:
                    figure = v.draw_circle(
                        opening_angle=np.pi / 2 - theta,
                        hemisphere=hemisphere,
                        return_figure=True,
                        **kwargs,
                    )
                v.draw_circle(opening_angle=np.pi / 2 + theta, figure=figure, **kwargs)
        elif projection == "spherical":
            figure = _plot_spherical(mode, ref, color, backend, figure)
        else:
            raise ValueError(
                "Unknown `projection`, valid options are 'stereographic' and "
                "'spherical'"
            )

        if return_figure:
            return figure

    def _raise_if_no_theta(self):
        if self.reflectors.theta[0] is None:
            raise ValueError(
                "Reflectors have no Bragg angles. Calculate with "
                "`diffsims.crystallography.ReciprocalLatticeVector.calculate_theta()`."
            )

    def _raise_if_no_structure_factor(self):
        if self.reflectors.structure_factor[0] is None:
            raise ValueError(
                "Reflectors have no structure factors. Calculate with "
                "`diffsims.crystallography.ReciprocalLatticeVector."
                "calculate_structure_factor()`."
            )


def _get_pattern(
    intensity: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
) -> np.ndarray:
    """Generate part of a master pattern by summing intensities from
    reflectors.

    Used in :meth:`calculate_master_pattern`.

    Parameters
    ----------
    intensity
        Reflector intensities.
    mask1, mask2
        Boolean arrays with ``True`` for reflectors to include the
        intensity from in each pattern coordinate.

    Returns
    -------
    part
        Master pattern part.
    """
    intensity_part = np.full(mask1.shape, intensity)
    part = 0.5 * np.sum(intensity_part, where=mask1, axis=1)
    part += np.sum(intensity_part, where=mask2, axis=1)
    return part


def _get_coordinates_in_upper_hemisphere(
    z_coordinates: Union[da.Array, np.ndarray], navigation_axes: tuple
) -> Tuple[da.Array, da.Array]:
    """Return two boolean arrays with True if a coordinate is in the
    upper hemisphere and if it is in the upper hemisphere in some
    pattern, respectively.
    """
    upper_hemisphere = da.atleast_2d(z_coordinates) > 0
    in_a_pattern = da.sum(upper_hemisphere, axis=navigation_axes) != 0
    return upper_hemisphere, in_a_pattern


def _plot_spherical(
    mode: str, ref: ReciprocalLatticeVector, color: np.ndarray, backend: str, figure
):
    v = Vector3d(ref).unit

    steps = 101
    if mode == "lines":
        circles = v.get_circle(steps=steps).data
    else:  # bands
        theta = ref.theta
        circles = (
            v.get_circle(opening_angle=np.pi / 2 - theta, steps=steps).data,
            v.get_circle(opening_angle=np.pi / 2 + theta, steps=steps).data,
        )

    colors = ["r", "g", "b"]
    labels = ["e1", "e2", "e3"]

    if backend == "matplotlib":
        if figure is not None:
            ax = figure.axes[0]
        else:
            figure, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        if mode == "lines":
            for i, ci in enumerate(circles):
                ax.plot3D(*ci.T, c=color[i])
        else:  # bands
            for i, (c1i, c2i) in enumerate(zip(*circles)):
                ax.plot3D(*c1i.T, c=color[i])
                ax.plot3D(*c2i.T, c=color[i])
        ax.axis("off")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.azim = -90
        ax.elev = 90
        arrow_kwargs = dict(mutation_scale=20, arrowstyle="-|>", linewidth=1)
        ha = ["left", "center", "center"]
        va = ["center", "bottom", "bottom"]
        for i in range(3):
            data = np.zeros((3, 2))
            data[i, 0] = 1
            data[i, 1] = 1.5
            arrow = Arrow3D(
                *data, color=colors[i], label=labels[i] + " axis", **arrow_kwargs
            )
            ax.add_artist(arrow)
            ax.text3D(
                *data[:, 1],
                s=labels[i],
                color=colors[i],
                label=labels[i] + " label",
                ha=ha[i],
                va=va[i],
            )
    else:  # pyvista
        import pyvista as pv

        if figure is None:
            show = True
            figure = pv.Plotter()
            figure.add_axes(
                color="k", xlabel=labels[0], ylabel=labels[1], zlabel=labels[2]
            )
            sphere = pv.Sphere(radius=0.99)
            figure.add_mesh(sphere, color="w", lighting=False)
            figure.disable_shadows()
            figure.set_background("w")
            figure.set_viewup((0, 1, 0))
        else:
            show = False
            # Assume that the existing plot has a sphere with a radius of 1
            v = Vector3d(circles)
            circles = Vector3d.from_polar(v.azimuth, v.polar, radial=1.01).data

        if mode == "lines":
            circles_shape = circles.shape[:-1]
            circles = circles.reshape((-1, 3))
            lines = np.arange(circles.shape[0]).reshape(circles_shape)
            color = color[:, 0]
        else:  # bands
            circles = np.vstack(circles)
            circles_shape = circles.shape[:-1]
            circles = circles.reshape((-1, 3))
            lines = np.arange(circles.shape[0]).reshape(circles_shape)
            color = np.tile(color[:, 0], 2)

        # Create mesh from vertices (3D coordinates) and line
        # connectivity arrays
        lines = np.insert(lines, 0, steps, axis=1)
        lines = lines.ravel()
        mesh = pv.PolyData(circles, lines=lines)
        figure.add_mesh(
            mesh,
            scalars=color,
            cmap="gray",
            scalar_bar_args=dict(title=r"$F_{hkl}$"),
        )

        if show:
            figure.show()

    return figure
