# Copyright 2019-2024 The kikuchipy developers
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

from ._hough_indexing import xmap_from_hough_indexing_data
from ._hrosm._averaging import (
    average_grain_orientations,
    grain_reference_orientation_deviation_map,
)
from ._hrosm._grains import GrainTable
from ._hrosm._kam import kernel_average_misorientation_map
from ._hrosm._sampling import misorientation_ball, misorientation_ball_spacing
from ._hrosm._segmentation import grain_bounding_boxes, segment_grains_kam
from ._merge_crystal_maps import merge_crystal_maps
from ._orientation_similarity_map import orientation_similarity_map
from ._refinement._refinement import (
    compute_refine_orientation_projection_center_results,
    compute_refine_orientation_results,
    compute_refine_projection_center_results,
)
from ._spherical._back_projection import SphericalBackProjector
from ._spherical._fft import fast_bandwidths
from ._spherical._indexer import SphericalIndexer
from ._spherical._master_pattern_harmonics import MasterPatternHarmonics
from ._spherical._namelist import EMSphInxNamelist
from ._spherical._pattern_repack import write_emsphinx_patterns
from ._spherical._pseudo_symmetry import (
    PseudoSymmetryOperators,
    find_pseudo_symmetry_operators,
    read_emsphinx_psym_file,
    write_emsphinx_psym_file,
)
from .similarity_metrics._normalized_cross_correlation import (
    NormalizedCrossCorrelationMetric,
)
from .similarity_metrics._normalized_dot_product import NormalizedDotProductMetric
from .similarity_metrics._similarity_metric import SimilarityMetric

__all__ = [
    "EMSphInxNamelist",
    "GrainTable",
    "MasterPatternHarmonics",
    "NormalizedCrossCorrelationMetric",
    "NormalizedDotProductMetric",
    "PseudoSymmetryOperators",
    "SimilarityMetric",
    "SphericalBackProjector",
    "SphericalIndexer",
    "average_grain_orientations",
    "compute_refine_orientation_projection_center_results",
    "compute_refine_orientation_results",
    "compute_refine_projection_center_results",
    "fast_bandwidths",
    "find_pseudo_symmetry_operators",
    "grain_bounding_boxes",
    "grain_reference_orientation_deviation_map",
    "kernel_average_misorientation_map",
    "merge_crystal_maps",
    "misorientation_ball",
    "misorientation_ball_spacing",
    "orientation_similarity_map",
    "read_emsphinx_psym_file",
    "segment_grains_kam",
    "write_emsphinx_patterns",
    "write_emsphinx_psym_file",
    "xmap_from_hough_indexing_data",
]
