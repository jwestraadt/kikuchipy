#!/bin/bash
##
## Copyright 2019-2026 the kikuchipy developers
##
## This file is part of kikuchipy.
##
## kikuchipy is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## kikuchipy is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
##


# This script must be run from the kikuchipy/ top directory:
#   $ chmod u+x ./doc/tutorials/run_nbval.sh
#   $ ./doc/tutorials/run_nbval.sh

# hrebsd_si_indent.ipynb gates on a local 18.9 GB file that is not
# shipped, and finds it in the working directory, its parent or its
# grandparent when KIKUCHIPY_LOCAL_DATA_DIR is unset. This script runs
# from the repository root, where a local copy of that file may well
# sit, and nbval executes every cell it is given, so without this the
# gate would start a multi hour correlation. Naming a directory that
# does not hold the file switches the tutorial off, which is what the
# notebook documents.
export KIKUCHIPY_LOCAL_DATA_DIR="${KIKUCHIPY_LOCAL_DATA_DIR:-$(mktemp -d)}"

# List notebooks that nbval should run
declare -a NOTEBOOKS=(\
  "hough_indexing.ipynb"\
  "hrebsd_dic.ipynb"\
  "hrebsd_si_indent.ipynb"\
  "hybrid_indexing.ipynb"\
  "mandm2021_sunday_short_course.ipynb"\
  "pattern_matching.ipynb"\
  "pc_extrapolate_plane.ipynb"\
  "pc_fit_plane.ipynb"\
  "pseudo_symmetry.ipynb"\
  "spherical_indexing.ipynb"\
)

# Append relative path to notebook names
for i in "${!NOTEBOOKS[@]}"; do
  NOTEBOOKS[i]=doc/tutorials/"${NOTEBOOKS[i]}"
done

# Test with nbval
pytest -v --nbval "${NOTEBOOKS[@]}" --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg
