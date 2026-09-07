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

"""Tests of ``kikuchipy.indexing._spherical._pseudo_symmetry``.

Covers the codec, prediction, result-object, export and binary
sections of ``specs/2026-09-06-pseudo-symmetry/validation.md``:

- ``TestPsymFileCodec``: the EMsoft ``qu`` angle file grammar, the
  exact-identity skip, the frozen writer format, and the D2
  conjugation contract asserted two ways, including the golden
  bytes-exact literal (D2.6a).
- ``TestFindPseudoSymmetryOperators``: the D3 frozen-signature and
  no-input-mutation pins, the D9.1 Ni positive-count pin and its
  subset/empty companions, cutoff and volume semantics (including
  the ``_local_maxima`` scan-shape killer at a bandwidth where
  ``fast_size(2 bw - 1) != 2 bw - 1``, D3.4), the D6 guards, the
  synthetic 3-fold/6-fold blends, the off-grid 0.95-factor pin, the
  dedup metric discriminator (D9.4) and the two-phase rotated-copy
  direction oracle (D2.6c).
- ``TestPseudoSymmetryOperatorsObject``: ``save`` round trip and the
  empty-set refusal.
- ``TestExports``: the four public names of D10.
- ``TestLocalMasters``: the ``KIKUCHIPY_LOCAL_MASTERS_DIR``
  skip-if-absent mechanism tests (D9.6), dormant until open
  question 9.5 supplies filenames.
- ``TestIndexEBSDPsymFile`` (gated): the D1 inertness pin and the
  D5/D6 error paths against the shipped binary.
- ``TestMasterXcorrParity`` (gated): the D8.1 stdout parity, the
  D8.2i two-file-branch discriminator and the D8.2ii weekly
  two-master run.

Values marked ``MEASURED-THEN-PINNED`` follow requirements D11:
placeholders carry a ``FIXME-pin`` marker and are replaced by dated
measured values at the implementation gate.  Binary-side numbers
measured on 2026-09-07 (MasterXcorr.exe 60f3517, bw 88, EMsoft Ni
h5) are recorded in validation.md "Recorded results" and quoted
where they inform a pin.
"""

import functools
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import warnings

import h5py
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.quaternion.symmetry import O as PROPER_OH
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.indexing import EMSphInxNamelist, write_emsphinx_patterns
from kikuchipy.indexing._spherical import _pseudo_symmetry
from kikuchipy.indexing._spherical._fft import fast_size
from kikuchipy.indexing._spherical._indexer import SphericalIndexer
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._pseudo_symmetry import (
    PseudoSymmetryOperators,
    find_pseudo_symmetry_operators,
    read_emsphinx_psym_file,
    write_emsphinx_psym_file,
)

# ------------------------- Frozen constants ------------------------- #

# The golden non-involutory operator of the D2.6a bytes-exact test:
# 25 degrees about the low-symmetry axis [1, 2, 3] (its square is a
# 50 degree rotation, so no inversion pair can mask a conjugation
# error).  The expected file body is the conjugate ``(~op).data`` in
# the frozen writer format, derived from the D2 chain and computed
# with orix on 2026-09-07 (recorded in validation.md).
GOLDEN_AXIS = (1.0, 2.0, 3.0)
GOLDEN_ANGLE_DEG = 25.0
GOLDEN_ROW = (
    "0.9762960071199334 -0.057845920020143056 -0.11569184004028611 -0.17353776006042917"
)
GOLDEN_FILE_BODY = f"qu\n1\n{GOLDEN_ROW}\n"

# In-package Ni ``*.sht`` files
NI_SMALL_SHT = "emsphinx/ni_small_20kv_bw384.sht"

# Bandwidth of the pure-Python Ni mechanism tests.  It satisfies the
# D8 discipline ``fast_size(2 bw - 1) == 2 bw - 1`` (135 = 27 x 5)
NI_BANDWIDTH = 68

# Bandwidth of the synthetic blend tests.  MEASURED CORRECTION at the
# implementation gate (2026-09-07, recorded in validation.md): the
# drafted 53 is unusable for the neutral-flag blends -- at odd
# ``slP`` the identity cell sits at the stored beta edge and the D3.3
# identity-cell-seeded reference refinement Newton-steps off the
# gimbal-degenerate edge into a negative stationary value (measured
# ``v_max = -0.256673`` at bw 53 against a true peak of 2.428933),
# poisoning every normalized intensity.  At an EVEN ``slP`` the
# translated identity cell is exactly the identity, the reference
# refine returns the true peak, and the blend intensities match the
# D9.2 "expected ~0.5" narrative (measured 0.5039 at weight 0.7).
# 60 gives ``fast_size(119) = 120`` (even, 2^3 x 3 x 5); note it is
# therefore a second non-coincident bandwidth beside
# ``NON_COINCIDENT_BANDWIDTH``
BLEND_BANDWIDTH = 60

# The scan-shape killer bandwidth (D3.4 recorded deviation):
# 2 x 64 - 1 = 127 is prime, so ``fast_size(127) == 128 != 127`` and
# the correlator's true cube shape ``(65, 128, 128)`` differs from
# the flat ``sl = 2 bw - 1`` indexing the C++ mis-scans with -- the
# planted-peak unit test below is the direct slP-vs-sl mutant killer
# of plan 7.2 (since the implementation-gate blend correction,
# ``BLEND_BANDWIDTH`` is a second non-coincident regime, so the
# mutant now also corrupts every blend test; the Ni routes at 68 and
# 88 stay coincident)
NON_COINCIDENT_BANDWIDTH = 64

# The stated cutoff of the D9.1 Ni pins.  Binary-measured guidance
# (bw 88, h5 route, 2026-09-07): at cutoff 0.9 the raw list holds
# exactly 22 proper-Oh rows and at 0.5 one extra displaced edge row
NI_CUTOFF = 0.9

# MEASURED-THEN-PINNED (2026-09-07, implementation gate): the D9.1
# positive-count pin.  Binary-measured 22 at bw 88 on the h5 route
# and kikuchipy-measured 22 on this .sht/bw-68 route (recorded in
# validation.md): 24 proper Oh rotations minus the identity (whose
# cell sits at the stored beta edge; the identity-seeded reference
# refinement stalls below the true peaks) minus one C2' merged by
# the folding.
NI_OPS_COUNT = 22

# MEASURED-THEN-PINNED (2026-09-07): per-operator angular tolerance
# to the nearest proper Oh rotation, degrees.  Measured maximum
# 1.9948 on the .sht/bw-68 route (a single displaced near-C2'
# glide-edge row at intensity 0.949; the bulk sit far closer);
# pinned at 1.25x
NI_OH_ANGLE_TOL_DEG = 2.5

# MEASURED-THEN-PINNED (2026-09-07): the largest normalized
# intensity, measured 1.495790 on the .sht/bw-68 route (ABOVE one:
# the reference maximum is the identity-cell-seeded refinement,
# which stalls below the true peaks -- this refutes the drafted
# "intensities ~1.0" expectation, recorded in validation.md;
# binary-measured 1.3521 at bw 88 on the h5 route)
NI_TOP_INTENSITY = 1.4958

# The synthetic blends: weight and explicit cutoff far below the
# expected relative intensity (D9.2 decouples this from the default
# cutoff open question)
BLEND_WEIGHT = 0.7
BLEND_CUTOFF = 0.25

# MEASURED-THEN-PINNED (2026-09-07): angular tolerance of a
# recovered blend operator to the constructed one, degrees.
# Measured 0.1748 for the on-sum-grid 120/60 degree operators and
# 0.4221 for the off-grid one at bw 60; pinned at ~2x the worst
BLEND_ANGLE_TOL_DEG = 0.85

# MEASURED-THEN-PINNED (2026-09-07): relative-intensity band of a
# recovered blend operator, ``lam / (1 + lam^2)``-ish for weight
# ``lam`` (0.4698 for 0.7).  Measured 0.5039 at bw 60; pinned at
# rel 0.05
BLEND_INTENSITY_BOUNDS = (0.48, 0.53)

# An off-grid z angle for the 0.95 search-factor pin: not a
# multiple of the alpha/gamma sum cell 360 / 120 = 3 deg nor of its
# half at ``BLEND_BANDWIDTH`` (100.7 / 3 = 33.57)
OFF_GRID_ANGLE_DEG = 100.7

# MEASURED-THEN-PINNED (2026-09-07): the discriminating cutoff of
# the off-grid 0.95-factor pin.  Measured at bw 60: the off-grid
# operator's brightest grid voxel reads 0.4800 of ``v_max`` while
# its refined intensity is 0.5510, so with 0.49 the candidate gate
# ``>= v_max * 0.49 * 0.95 = 0.4655 v_max`` admits the voxel and
# the keep gate ``0.5510 >= 0.49`` retains it, while a port which
# drops the 0.95 factor gates at ``0.49 v_max > 0.4800 v_max`` and
# loses the operator
OFF_GRID_CUTOFF = 0.49

# The two-phase rotated-copy oracle (D2.6c): tolerance per the
# Phase 6 margin convention on the composed-orientation identity
# (measured 0.68/1.07 deg there); the direction itself was
# machine-verified 2026-09-07 (peak at the constructed zyz exactly,
# the inverse 97.9 deg away)
TWO_PHASE_ZYZ = (0.9, 0.7, -0.4)
TWO_PHASE_TOL_DEG = 2.5
TWO_PHASE_WRONG_FLOOR_DEG = 10.0

# MasterXcorr.exe parity (D8.1), binary-measured 2026-09-07 at
# bw 88, cutoff 0.9, EMsoft Ni h5: 22 rows, intensities 1.3521 (7
# rows) and 1.2082 (15 rows), reference maximum 0.719309.  The
# two-file branch (distinct path spellings, D8.2i) re-seeds the
# reference from the coarse argmax: 0.972597, intensities 1.0000
# and 0.8936
MASTERXCORR_BANDWIDTH = 88
MASTERXCORR_CUTOFF = 0.9
MASTERXCORR_ROW_COUNT = 22
MASTERXCORR_VMAX_AUTO = 0.719309
MASTERXCORR_VMAX_TWO_FILE = 0.972597
# The printed quaternion is ``Quat::to_string(6)``, fixed precision
# six (D2.7): half an ulp is 5e-7, doubled for the comparison so
# the conjugation killer keeps its teeth (1e-4 would blunt it)
MASTERXCORR_QUAT_ATOL = 2e-6
# MEASURED-THEN-PINNED (2026-09-07, implementation gate; validated
# again at the review and fix gates): kikuchipy-vs-binary relative
# intensity band.  Fix-stage measurement on the parity route (bw 88,
# cutoff 0.9, Ni h5, 22 bijectively matched rows, recorded in
# validation.md): max relative deviation 2.852e-5 (mean 8.9e-6), so
# 0.05 stands as measured-sufficient with ~1750x margin -- it is
# deliberately NOT tightened to that scale because the same band
# also guards the never-yet-executed Al two-master weekly route
MASTERXCORR_INTENSITY_RTOL = 0.05

# Candidate file names of the local hcp/TiAl EMsoft masters (D9.6).
# PROVISIONAL until open question 9.5 supplies the user's actual
# filenames; the skip reason names them so the dormant tests are
# self-documenting
LOCAL_MASTERS_ENV = "KIKUCHIPY_LOCAL_MASTERS_DIR"
LOCAL_MASTER_CANDIDATES = (
    "ti_alpha_mc_mp_20kv.h5",
    "mg_mc_mp_20kv.h5",
    "ti64_alpha_mc_mp_20kv.h5",
    "tial_gamma_mc_mp_20kv.h5",
)


# ----------------------------- Helpers ------------------------------ #


@functools.lru_cache(maxsize=4)
def ni_harmonics(bandwidth):
    """Return the harmonics of the shipped small Ni master built
    directly at ``bandwidth``, cached."""
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


@functools.lru_cache(maxsize=1)
def ni_sht_harmonics():
    """Return the in-package bw 384 ``*.sht`` coefficients, the D9.1
    input (resized inside ``find_pseudo_symmetry_operators``)."""
    return MasterPatternHarmonics.from_file(Dataset(NI_SMALL_SHT).fetch_file_path())


def z_rotation(degrees):
    """Return a rotation about +z by ``degrees``."""
    return Rotation.from_axes_angles([0, 0, 1], np.deg2rad(degrees))


def golden_operator():
    """Return the frozen golden non-involutory operator."""
    return Rotation.from_axes_angles(GOLDEN_AXIS, np.deg2rad(GOLDEN_ANGLE_DEG))


def blended_harmonics(s_rotation, weight=BLEND_WEIGHT, bandwidth=BLEND_BANDWIDTH):
    """Return a synthetic pseudo-symmetric blend
    ``f + weight * f.rotate(S)`` with neutral symmetry flags.

    The blend rotations of this suite are about +z only (D9.2), so
    the inherited 4-fold/mirror flags would still be true, but the
    flags are neutralized anyway: the correlator folds the cube by
    the first master's flags, so stale flags would corrupt the very
    cube the tests scan.  The coefficients are added through the
    private constructor; no public addition API ships.
    """
    harmonics = ni_harmonics(bandwidth)
    rotated = harmonics.rotate(s_rotation)
    alm = harmonics.alm + weight * rotated.alm
    return MasterPatternHarmonics(alm, phase=Phase("blend", point_group="1"))


def min_angle_to(operators, target):
    """Return the smallest rotation angle in degrees between each of
    ``operators`` and the single rotation ``target``."""
    return np.rad2deg((operators.flatten() * ~target.flatten()).angle).min()


def angles_to_proper_oh(operators):
    """Return each operator's angle in degrees to the nearest of the
    24 proper Oh rotations."""
    ops = operators.flatten()
    proper = Rotation(PROPER_OH.data)
    angles = np.rad2deg((ops.outer(~proper)).angle)
    return angles.reshape(ops.size, proper.size).min(axis=1)


def write_text(path, text):
    """Write ``text`` with ``\\n`` line ends, byte exactly."""
    with open(path, "w", newline="\n") as file:
        file.write(text)
    return path


def parse_masterxcorr_stdout(text):
    """Return the printed ``(intensity, w, x, y, z)`` rows of a
    ``MasterXcorr`` stdout, and the ``maximum intensity`` value.

    The parser is double-space tolerant (D2.7): ``Quat::to_string``
    left-pads every non-negative component with an alignment space,
    so rows are split on any whitespace run.
    """
    rows = []
    v_max = None
    for line in text.splitlines():
        if line.startswith("maximum intensity:"):
            v_max = float(line.split(":")[1])
            continue
        tokens = line.split()
        if len(tokens) != 5:
            continue
        try:
            rows.append([float(token) for token in tokens])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float64), v_max


# --------------------------- Codec (D6/D2) -------------------------- #


class TestPsymFileCodec:
    """The psymfile codec of D6 with the D2.4 conjugation at both
    ends.  [D2, D6]"""

    def test_read_qu_file(self, tmp_path):
        # comma- and whitespace-separated tokens mix freely, and a
        # row may span lines: the grammar is token based
        # (emsoft.hpp lines 110-141).  [D6]
        path = write_text(
            tmp_path / "ops.txt",
            "qu\n2\n"
            "0.7071067811865476, 0.7071067811865476, 0.0, 0.0\n"
            "0.5 0.5\n0.5, 0.5\n",
        )
        ops = read_emsphinx_psym_file(path)
        assert ops.size == 2
        rows = np.array(
            [
                [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        # the D2.4 contract: returned operators are the CONJUGATED
        # file rows
        expected = (~Rotation(rows)).data
        assert np.allclose(ops.data, expected, atol=1e-15)

    @pytest.mark.parametrize("token", ["eu", "om", "ax", "ro", "ho", "cu"])
    def test_read_rejects_non_qu(self, tmp_path, token):
        # mirrors the master.hpp line 225 throw for every other
        # angle-file type token.  [D6]
        path = write_text(tmp_path / "ops.txt", f"{token}\n1\n0.1 0.2 0.3 0.4\n")
        with pytest.raises(ValueError, match="quaternion"):
            read_emsphinx_psym_file(path)

    def test_read_skips_exact_identity(self, tmp_path):
        # the exact float equality skip of master.hpp lines 227-231,
        # ported literally.  [D6]
        path = write_text(
            tmp_path / "ops.txt",
            "qu\n3\n0.5 0.5 0.5 0.5\n1.0 0.0 0.0 0.0\n0.5 -0.5 -0.5 -0.5\n",
        )
        ops = read_emsphinx_psym_file(path)
        assert ops.size == 2

    def test_read_keeps_near_identity(self, tmp_path):
        # ``1.0000001 0 0 0`` is NOT the exact identity and is kept
        # (near-identity operators merely re-converge to the global
        # peak).  The orix constructor unitizes the non-unit row, a
        # recorded deviation from EMSphInx's raw values (D6).  [D6]
        path = write_text(tmp_path / "ops.txt", "qu\n1\n1.0000001, 0.0, 0.0, 0.0\n")
        ops = read_emsphinx_psym_file(path)
        assert ops.size == 1
        assert np.allclose(ops.data, [[1.0, 0.0, 0.0, 0.0]], atol=1e-6)

    @pytest.mark.parametrize(
        "body",
        [
            # too few numbers for the count
            "qu\n3\n1 0 0 0\n0.5 0.5 0.5 0.5\n",
            # too many numbers for the count
            "qu\n1\n0.5 0.5 0.5 0.5\n0.5 -0.5 0.5 -0.5\n",
        ],
    )
    def test_read_count_mismatch_raises(self, tmp_path, body):
        # emsoft.hpp lines 140-141 (the typo'd "orientions" messages
        # are not reproduced).  [D6]
        path = write_text(tmp_path / "ops.txt", body)
        with pytest.raises(ValueError):
            read_emsphinx_psym_file(path)

    def test_write_format(self, tmp_path):
        # the frozen writer format: ``qu`` token, count, one full
        # 64-bit-repr ``w x y z`` row per line, single spaces; rows
        # equal ``(~ops).data`` to 1e-15 -- the conjugation unit
        # pin.  [D6, D2.4]
        ops = Rotation(
            np.array(
                [
                    [
                        0.9762960071199334,
                        0.057845920020143056,
                        0.11569184004028611,
                        0.17353776006042917,
                    ],
                    [0.5, -0.5, 0.5, -0.5],
                ]
            )
        )
        path = tmp_path / "ops.txt"
        write_emsphinx_psym_file(path, ops)
        lines = path.read_text().splitlines()
        assert lines[0] == "qu"
        assert int(lines[1]) == 2
        assert len(lines) == 4
        written = np.array(
            [[float(token) for token in line.split(" ")] for line in lines[2:]]
        )
        assert written.shape == (2, 4)
        assert np.allclose(written, (~ops).data, atol=1e-15)

    def test_write_golden_bytes(self, tmp_path):
        # THE D2.6a conjugation killer: one known non-involutory
        # operator produces the specific ``w x y z`` digits of
        # ``~op``, pinned bytes exact against the literal derived
        # from the D2 chain (recorded 2026-09-07).  An unconjugated
        # writer flips three signs and dies here.  [D2.6a, D6]
        path = tmp_path / "golden.txt"
        write_emsphinx_psym_file(path, golden_operator())
        assert path.read_bytes() == GOLDEN_FILE_BODY.encode()

    def test_write_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_emsphinx_psym_file(tmp_path / "ops.txt", Rotation.empty())

    def test_round_trip_identity(self, tmp_path):
        # ``read(write(ops)) == ops`` exactly: the two conjugations
        # cancel bitwise, and full-repr precision loses nothing.
        # [D6]
        ops = Rotation(
            np.array(
                [
                    [
                        0.9762960071199334,
                        -0.057845920020143056,
                        0.11569184004028611,
                        -0.17353776006042917,
                    ],
                    [0.5, 0.5, -0.5, 0.5],
                    [0.8, 0.6, 0.0, 0.0],
                ]
            )
        )
        path = tmp_path / "ops.txt"
        write_emsphinx_psym_file(path, ops)
        returned = read_emsphinx_psym_file(path)
        assert np.array_equal(returned.data, ops.data)


# ------------------------ Prediction (D3/D9) ------------------------ #


class TestFindPseudoSymmetryOperators:
    """The ``MasterXcorr`` port of D3 on shipped and synthetic
    inputs.  [D3, D9]"""

    def test_signature_defaults_are_the_d3_frozen_ones(self):
        # the suite's freeze convention (the ctor and signal-method
        # defaults dicts) extended to the new public function:
        # bandwidth 88, exclude_symmetry True, keep_volume False,
        # emsphinx_compatible True are all D3-frozen.  ``cutoff`` is
        # deliberately NOT pinned: its 0.5 default is a recorded
        # kikuchipy choice under open question 9.4, and no test may
        # depend on it.  [D3]
        parameters = inspect.signature(find_pseudo_symmetry_operators).parameters
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        assert "cutoff" in defaults
        del defaults["cutoff"]
        assert defaults == {
            "second_harmonics": None,
            "bandwidth": 88,
            "exclude_symmetry": True,
            "keep_volume": False,
            "emsphinx_compatible": True,
        }
        # ``harmonics`` and ``second_harmonics`` are positional, the
        # rest keyword only
        assert [
            name
            for name, parameter in parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        ] == [
            "bandwidth",
            "cutoff",
            "exclude_symmetry",
            "keep_volume",
            "emsphinx_compatible",
        ]

    def test_callers_harmonics_are_never_modified(self):
        # the frozen D3 step 1: the finder operates on DC-removed
        # COPIES; the caller's objects are never modified.  The
        # natural mistranslation -- ``removeDC`` in place, exactly
        # what the C++ does to its own local objects -- is invisible
        # to every value test because DC removal is idempotent, while
        # it would silently corrupt the caller's object (and this
        # suite's lru-cached shared harmonics).  Both the auto and
        # the two-phase modes are pinned.  [D3]
        harmonics = ni_harmonics(BLEND_BANDWIDTH)
        before = harmonics.alm.copy()
        find_pseudo_symmetry_operators(
            harmonics,
            bandwidth=BLEND_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert np.array_equal(harmonics.alm, before)

        second = MasterPatternHarmonics(
            harmonics.alm.copy(), phase=Phase("copy", point_group="1")
        )
        before_second = second.alm.copy()
        find_pseudo_symmetry_operators(
            harmonics,
            second,
            bandwidth=BLEND_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert np.array_equal(harmonics.alm, before)
        assert np.array_equal(second.alm, before_second)

    def test_ni_proper_oh_count(self, record_property):
        # the D9.1 positive-count pin, which no other Ni test may
        # run without: the raw list at the stated cutoff holds the
        # proper-Oh autocorrelation peaks, so nothing below can pass
        # on an empty return.  Binary-measured guidance (bw 88, h5
        # route, 2026-09-07): 22 rows -- the identity's cell sits at
        # the stored beta edge and is NOT returned, and intensities
        # sit ABOVE one (1.3521/1.2082) because the reference
        # maximum is the identity-cell-seeded refinement (0.719309),
        # refuting the drafted "identity + ~1.0" expectation.
        # [D9.1, D3]
        result = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        record_property("ni_ops_count", int(result.operators.size))
        record_property("ni_intensities", result.intensities.tolist())
        # MEASURED-THEN-PINNED count, re-measured on this
        # .sht/bw-68 route at the implementation gate (2026-09-07):
        # 22, agreeing with the binary's bw-88 h5 count
        assert result.operators.size == NI_OPS_COUNT
        angles = angles_to_proper_oh(result.operators)
        assert (angles < NI_OH_ANGLE_TOL_DEG).all()
        # descending intensities, the top one at the measured band
        assert (np.diff(result.intensities) <= 0).all()
        assert result.intensities[0] == pytest.approx(NI_TOP_INTENSITY, rel=0.05)
        assert (result.intensities >= NI_CUTOFF).all()

    def test_ni_ops_subset_of_oh(self):
        # sequenced after the count pin: non-empty is asserted here
        # too, so the subset claim cannot pass vacuously.  [D9.1]
        result = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert result.operators.size > 0
        assert (angles_to_proper_oh(result.operators) < NI_OH_ANGLE_TOL_DEG).all()

    def test_ni_exclude_symmetry_empty(self):
        # Ni (m-3m) has no genuine pseudo-symmetry: every raw peak
        # is a true proper rotation, so ``exclude_symmetry=True``
        # returns an empty set -- asserted only after the raw list
        # is seen non-empty.  [D3.7, D9.1]
        raw = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert raw.operators.size > 0
        excluded = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=True,
        )
        assert excluded.operators.size == 0

    def test_cutoff_filters(self):
        # a lower cutoff keeps a superset, and every reported
        # intensity respects the keep gate ``>= cutoff`` (the 0.95
        # factor widens only the SEARCH, never the report).  [D3]
        low = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=0.5,
            exclude_symmetry=False,
        )
        high = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=NI_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert low.operators.size >= high.operators.size
        assert (low.intensities >= 0.5).all()
        assert (high.intensities >= NI_CUTOFF).all()

    @pytest.mark.parametrize("cutoff", [-0.1, 1.1])
    def test_cutoff_out_of_range_raises(self, cutoff):
        # EMSphInx errors at CLI parse time; kikuchipy at call time,
        # same outcome (D3.1 recorded).  [D3]
        with pytest.raises(ValueError):
            find_pseudo_symmetry_operators(
                ni_sht_harmonics(), bandwidth=NI_BANDWIDTH, cutoff=cutoff
            )

    @pytest.mark.parametrize("bandwidth", [8, 600])
    def test_bandwidth_out_of_range_raises(self, bandwidth):
        # the module's [16, 512] rule replaces the C++ CLI clamp
        # [53, 313] (D3.1 recorded deviation).  [D3]
        with pytest.raises(ValueError):
            find_pseudo_symmetry_operators(ni_sht_harmonics(), bandwidth=bandwidth)

    def test_volume_shape_and_optionality(self):
        # the correlation cube is the one optional extra of this
        # module (the stereogram is deferred): the true
        # ``(bwP, slP, slP)`` shape, and ``None`` unless asked for.
        # [D3]
        harmonics = ni_harmonics(BLEND_BANDWIDTH)
        without = find_pseudo_symmetry_operators(
            harmonics,
            bandwidth=BLEND_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert without.volume is None
        with_volume = find_pseudo_symmetry_operators(
            harmonics,
            bandwidth=BLEND_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
            keep_volume=True,
        )
        side = fast_size(2 * BLEND_BANDWIDTH - 1)
        assert with_volume.volume is not None
        assert with_volume.volume.shape == (side // 2 + 1, side, side)
        assert with_volume.volume.dtype == np.float64
        assert with_volume.bandwidth == BLEND_BANDWIDTH

    def test_local_maxima_finds_a_planted_off_fast_grid_peak(self):
        # THE slP-vs-sl killer named by plan 7.2: at bandwidth 64 the
        # true cube side is ``fast_size(127) = 128``, so a port which
        # reintroduces the C++'s flat ``sl = 2 bw - 1`` scan
        # (``master_xcorr.cpp`` lines 108-111) reads every voxel
        # through a wrong stride and cannot return the planted peak's
        # flat index.  Every other bandwidth in this suite satisfies
        # ``fast_size(2 bw - 1) == 2 bw - 1``, where the two scans
        # coincide and the D3.4 recorded deviation is invisible.  The
        # peak is interior, so both glide semantics must agree and
        # the ``emsphinx_compatible`` flag is threaded through both
        # ways.  [D3.4]
        side = fast_size(2 * NON_COINCIDENT_BANDWIDTH - 1)
        assert 2 * NON_COINCIDENT_BANDWIDTH - 1 == 127
        assert side == 128
        shape = (side // 2 + 1, side, side)
        rng = np.random.default_rng(19)
        xc = rng.uniform(0.0, 0.2, shape)
        peak = (30, 70, 90)
        xc[peak] = 1.0
        # a bright shoulder above the threshold but beside the peak:
        # it must be rejected as a non-maximum, not merely thresholded
        xc[30, 70, 91] = 0.9
        expected = int(np.ravel_multi_index(peak, shape))
        for compatible in (True, False):
            indices = _pseudo_symmetry._local_maxima(
                xc, threshold=0.6, emsphinx_compatible=compatible
            )
            assert np.asarray(indices).ravel().tolist() == [expected]

    def test_local_maxima_agree_with_a_brute_force_reference(self):
        # the review-gate mutation killer (P11, recorded in
        # validation.md 2026-09-07): a scan which never compares one
        # plane of the 3 x 3 x 3 neighbourhood (e.g. the mutant
        # ``neighborhood[:2]``, dropping the k+1 plane) survived the
        # whole prior suite -- its spurious keeps refine into
        # already-found peaks on every routed cube.  Here one probe
        # pair is planted per neighbour offset on an otherwise
        # sub-threshold random cube: the probe voxel (0.9) is beaten
        # ONLY by its single planted neighbour (1.0), so a scan
        # which skips ANY of the 26 comparisons wrongly keeps the
        # corresponding probe.  The kept set must equal a
        # brute-force 26-neighbour reference computed in-test (by
        # fixture construction, exactly the 26 planted maxima).
        # Every candidate is interior, so both glide semantics agree
        # and the flag is threaded through both ways.  [D3.4]
        shape = (13, 24, 24)
        rng = np.random.default_rng(11)
        xc = rng.uniform(0.0, 0.2, shape)
        offsets = [
            (dk, dn, dm)
            for dk in (-1, 0, 1)
            for dn in (-1, 0, 1)
            for dm in (-1, 0, 1)
            if (dk, dn, dm) != (0, 0, 0)
        ]
        # probe-pair bases on a step-4 interior grid: members deviate
        # at most one voxel from their base, so voxels of different
        # pairs are at least two apart and never neighbours
        bases = [
            (k, n, m)
            for k in range(2, shape[0] - 2, 4)
            for n in range(2, shape[1] - 2, 4)
            for m in range(2, shape[2] - 2, 4)
        ]
        assert len(bases) >= len(offsets)
        expected = []
        for base, offset in zip(bases, offsets):
            neighbor = tuple(b + d for b, d in zip(base, offset))
            xc[base] = 0.9
            xc[neighbor] = 1.0
            expected.append(int(np.ravel_multi_index(neighbor, shape)))
        threshold = 0.5
        brute = []
        for k, n, m in zip(*np.nonzero(xc >= threshold)):
            block = xc[k - 1 : k + 2, n - 1 : n + 2, m - 1 : m + 2]
            assert block.shape == (3, 3, 3)  # interior by construction
            if xc[k, n, m] >= block.max():
                brute.append(int(np.ravel_multi_index((k, n, m), shape)))
        assert sorted(brute) == sorted(expected)
        for compatible in (True, False):
            indices = _pseudo_symmetry._local_maxima(
                xc, threshold=threshold, emsphinx_compatible=compatible
            )
            assert np.asarray(indices).ravel().tolist() == sorted(expected)

    def test_volume_shape_at_a_non_coincident_bandwidth(self):
        # the volume-shape pin repeated where it discriminates: at
        # bandwidth 64 the true ``(65, 128, 128)`` shape refutes any
        # port which sizes the cube (or its scan) with
        # ``sl = 2 bw - 1 = 127``.  [D3, D3.4]
        side = fast_size(2 * NON_COINCIDENT_BANDWIDTH - 1)
        assert side != 2 * NON_COINCIDENT_BANDWIDTH - 1
        result = find_pseudo_symmetry_operators(
            ni_harmonics(NON_COINCIDENT_BANDWIDTH),
            bandwidth=NON_COINCIDENT_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
            keep_volume=True,
        )
        assert result.volume is not None
        assert result.volume.shape == (side // 2 + 1, side, side)
        assert result.operators.size > 0

    def test_guards_empty_ops_are_none_equivalent(self, tmp_path):
        # D6 guards 1 and 4: an all-identity psymfile reads to a
        # size-0 rotation, and a size-0 rotation handed to the
        # indexer behaves exactly as ``None`` -- identical results,
        # no variant column key.  [D6]
        path = write_text(tmp_path / "ops.txt", "qu\n2\n1 0 0 0\n1.0, 0.0, 0.0, 0.0\n")
        empty = read_emsphinx_psym_file(path)
        assert empty.size == 0

        detector = kp.data.nickel_ebsd_small().detector.deepcopy()
        detector.pc = detector.pc_average
        signal = kp.data.nickel_ebsd_small()
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        patterns = signal.data.reshape((-1, 60, 60))[:1]

        harmonics = ni_harmonics(NI_BANDWIDTH)
        with_empty = SphericalIndexer(
            harmonics, detector, refine=False, pseudo_symmetry_ops=empty
        ).index_patterns(patterns, progressbar=False)
        with_none = SphericalIndexer(
            harmonics, detector, refine=False, pseudo_symmetry_ops=None
        ).index_patterns(patterns, progressbar=False)
        assert set(with_empty) == set(with_none)
        for key in with_none:
            assert np.array_equal(with_empty[key], with_none[key])

    def test_synthetic_three_fold_recovered(self):
        # the flagship D9.2 blend: a 120 degree z operator baked
        # into ``f + lam f.rotate(S)`` is recovered, together with
        # its autocorrelation inverse, at an explicit cutoff far
        # below the expected relative intensity.  [D3, D7, D9.2]
        s = z_rotation(120.0)
        blend = blended_harmonics(s)
        assert blend.n_fold == 1
        assert blend.has_equatorial_mirror is False
        result = find_pseudo_symmetry_operators(
            blend,
            bandwidth=BLEND_BANDWIDTH,
            cutoff=BLEND_CUTOFF,
            exclude_symmetry=True,
        )
        assert result.operators.size > 0
        assert min_angle_to(result.operators, s) < BLEND_ANGLE_TOL_DEG
        assert min_angle_to(result.operators, ~s) < BLEND_ANGLE_TOL_DEG
        low, high = BLEND_INTENSITY_BOUNDS
        matched = result.intensities[
            np.rad2deg((result.operators.flatten() * ~s).angle) < BLEND_ANGLE_TOL_DEG
        ]
        assert ((matched >= low) & (matched <= high)).all()

    def test_synthetic_six_fold_recovered(self):
        # the 60 degree companion of the roadmap box.  [D9.2]
        s = z_rotation(60.0)
        result = find_pseudo_symmetry_operators(
            blended_harmonics(s),
            bandwidth=BLEND_BANDWIDTH,
            cutoff=BLEND_CUTOFF,
            exclude_symmetry=True,
        )
        assert result.operators.size > 0
        assert min_angle_to(result.operators, s) < BLEND_ANGLE_TOL_DEG
        assert min_angle_to(result.operators, ~s) < BLEND_ANGLE_TOL_DEG

    def test_off_grid_operator_found(self):
        # an operator whose z angle sits between grid nodes leaves
        # its brightest voxel BELOW the true peak value; the
        # ``v_max * cutoff * 0.95`` candidate gate (the 0.95
        # "factor" of master_xcorr.cpp line 61) exists exactly so
        # such peaks survive the scan.  A port which drops the
        # factor loses this operator at ``OFF_GRID_CUTOFF``, which
        # rides the peak's measured grid value (0.4800 < 0.49 <=
        # 0.4800 / 0.95; measured 2026-09-07, the implementation
        # gate).  [D3.4]
        s = z_rotation(OFF_GRID_ANGLE_DEG)
        result = find_pseudo_symmetry_operators(
            blended_harmonics(s),
            bandwidth=BLEND_BANDWIDTH,
            cutoff=OFF_GRID_CUTOFF,
            exclude_symmetry=True,
        )
        assert result.operators.size > 0
        assert min_angle_to(result.operators, s) < BLEND_ANGLE_TOL_DEG

    def test_dedup_three_deg_pair(self):
        # the D9.4 metric discriminator, at unit level on the dedup
        # helper.  The 2 degree threshold is a QUATERNION-DOT
        # HALF-ANGLE (~4 degrees of misorientation, D3's frozen
        # angle-space statement).  A pair 3 degrees apart in
        # MISORIENTATION (1.5 in half-angle) merges under the
        # correct reading and survives under the swapped one; a
        # pair 6 degrees apart in misorientation (3 in half-angle)
        # survives under the correct reading and would merge under
        # a doubled threshold.  Both geometries together kill every
        # convention-swap mutant of plan 7.2; the mirrored intensity
        # ordering below kills the keep-the-last-seen one, which the
        # brighter-second cases cannot see.  The end-to-end blend
        # construction of this pair is deferred to the
        # implementation-gate measurement (D9.4 feasibility).
        # [D3.5, D9.4]
        def z_pair(separation_deg):
            qu = np.stack(
                [
                    z_rotation(40.0).data.ravel(),
                    z_rotation(40.0 + separation_deg).data.ravel(),
                ]
            )
            return qu

        intensities = np.array([0.7, 0.9])

        # 3 deg misorientation = 1.5 deg half-angle: inside the
        # 2 deg threshold, merged, the brighter (second) kept
        keep = _pseudo_symmetry._dedup_keep_brighter(z_pair(3.0), intensities)
        assert keep.tolist() == [False, True]

        # the mirrored ordering, brighter FIRST: completes the
        # discriminator -- a keep-the-last-seen mutant produces the
        # same mask as keep-the-brighter above and dies only here
        keep = _pseudo_symmetry._dedup_keep_brighter(z_pair(3.0), np.array([0.9, 0.7]))
        assert keep.tolist() == [True, False]

        # 6 deg misorientation = 3 deg half-angle: outside the
        # threshold, both kept
        keep = _pseudo_symmetry._dedup_keep_brighter(z_pair(6.0), intensities)
        assert keep.tolist() == [True, True]

    def test_two_phase_rotated_copy(self):
        # the pure-Python conjugation-sensitive oracle (D2.6c): for
        # ``h2 = h.rotate(S)`` the cross-master operator is ``~S``
        # -- "the phase-1 equivalent of a phase-2 orientation is
        # ``op * O_2``", the Phase 6 composed-orientation identity.
        # Direction machine-verified 2026-09-07 on the merged
        # correlator: the peak lands on the constructed operator
        # exactly, its inverse 97.9 degrees away, so the assertion
        # is on the EXACT direction and a conjugation error cannot
        # hide (two-phase sets are not inversion closed).
        # [D2.6c, D3.9]
        harmonics = ni_harmonics(BLEND_BANDWIDTH)
        from kikuchipy.indexing._spherical._euler import zyz_to_quaternion

        s = Rotation(zyz_to_quaternion(np.asarray(TWO_PHASE_ZYZ)))
        result = find_pseudo_symmetry_operators(
            harmonics,
            harmonics.rotate(s),
            bandwidth=BLEND_BANDWIDTH,
            cutoff=0.5,
            exclude_symmetry=False,
        )
        assert result.operators.size > 0
        assert min_angle_to(result.operators, ~s) < TWO_PHASE_TOL_DEG
        assert min_angle_to(result.operators, s) > TWO_PHASE_WRONG_FLOOR_DEG

    def test_bandwidth_resize_path(self):
        # a stored bw 384 ``*.sht`` resized down inside the finder:
        # structural assertions only -- resize is a recorded
        # non-equivalence with direct construction, so no value of
        # the other tests is pinned on this route.  [D3.1]
        result = find_pseudo_symmetry_operators(
            ni_sht_harmonics(),
            bandwidth=BLEND_BANDWIDTH,
            cutoff=NI_CUTOFF,
            exclude_symmetry=False,
        )
        assert result.bandwidth == BLEND_BANDWIDTH
        assert result.operators.size > 0
        assert np.isfinite(result.intensities).all()


# ------------------------- Result object (D3) ----------------------- #


class TestPseudoSymmetryOperatorsObject:
    """The ``PseudoSymmetryOperators`` carrier.  [D3]"""

    def test_save_round_trip(self, tmp_path):
        ops = Rotation(
            np.array(
                [
                    [
                        0.9762960071199334,
                        -0.057845920020143056,
                        -0.11569184004028611,
                        -0.17353776006042917,
                    ],
                    [0.5, 0.5, 0.5, 0.5],
                ]
            )
        )
        result = PseudoSymmetryOperators(ops, np.array([0.9, 0.8]), None, NI_BANDWIDTH)
        saved = tmp_path / "saved.txt"
        result.save(saved)
        assert np.array_equal(read_emsphinx_psym_file(saved).data, ops.data)
        # ``save`` IS the codec writer, byte for byte
        written = tmp_path / "written.txt"
        write_emsphinx_psym_file(written, ops)
        assert saved.read_bytes() == written.read_bytes()

    def test_save_empty_raises(self, tmp_path):
        result = PseudoSymmetryOperators(
            Rotation.empty(), np.empty(0), None, NI_BANDWIDTH
        )
        with pytest.raises(ValueError):
            result.save(tmp_path / "empty.txt")


# --------------------------- Exports (D10) -------------------------- #


class TestExports:
    """The four public names of D10 and their docstring hygiene."""

    @pytest.mark.parametrize(
        "name",
        [
            "PseudoSymmetryOperators",
            "find_pseudo_symmetry_operators",
            "read_emsphinx_psym_file",
            "write_emsphinx_psym_file",
        ],
    )
    def test_the_name_resolves_through_the_lazy_loader(self, name):
        assert hasattr(kp.indexing, name)
        assert name in kp.indexing.__all__

    def test_all_is_sorted(self):
        assert list(kp.indexing.__all__) == sorted(kp.indexing.__all__)

    def test_no_public_docstring_names_a_roadmap_phase(self):
        # decision 6.14: roadmap phase numbers live in provenance
        # comments only
        docstrings = [
            _pseudo_symmetry.__doc__,
            PseudoSymmetryOperators.__doc__,
            PseudoSymmetryOperators.save.__doc__,
            find_pseudo_symmetry_operators.__doc__,
            read_emsphinx_psym_file.__doc__,
            write_emsphinx_psym_file.__doc__,
        ]
        for doc in docstrings:
            assert doc
            for phase in ("Phase 3", "Phase 7", "Phase 8", "Phase 9"):
                assert phase not in doc


# --------------- Local masters (KIKUCHIPY_LOCAL_MASTERS_DIR) -------- #


class TestLocalMasters:
    """Skip-if-absent mechanism tests on user-supplied EMsoft
    masters with genuine pseudo-symmetry (D9.6).  Dormant until open
    question 9.5 supplies the actual filenames; the candidate names
    below are provisional and the skip reason states them."""

    def _local_master_path(self):
        value = os.environ.get(LOCAL_MASTERS_ENV)
        if not value:
            pytest.skip(
                f"{LOCAL_MASTERS_ENV} is not set; set it to a directory "
                "holding an EMsoft master pattern h5 named one of "
                f"{', '.join(LOCAL_MASTER_CANDIDATES)} to run this test"
            )
        directory = Path(value)
        for name in LOCAL_MASTER_CANDIDATES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        pytest.skip(
            f"none of {', '.join(LOCAL_MASTER_CANDIDATES)} found in {directory}"
        )

    def test_local_master_finds_operators(self, record_property):
        # a genuinely pseudo-symmetric phase returns a non-empty
        # set at ``exclude_symmetry=True``.  Intensity/angle pins
        # are MEASURED-THEN-PINNED on the first execution
        # (FIXME-pin: record via ``record_property`` output).
        # [D9.6, roadmap box 2]
        path = self._local_master_path()
        master = kp.load(path, projection="lambert", hemisphere="both")
        harmonics = MasterPatternHarmonics.from_master_pattern(
            master, bandwidth=NI_BANDWIDTH
        )
        result = find_pseudo_symmetry_operators(
            harmonics,
            bandwidth=NI_BANDWIDTH,
            cutoff=0.5,
            exclude_symmetry=True,
        )
        record_property("local_master", str(path.name))
        record_property("local_ops_count", int(result.operators.size))
        record_property("local_intensities", result.intensities.tolist())
        assert result.operators.size > 0
        assert (result.intensities >= 0.5).all()


# ------------------ IndexEBSD psymfile pins (gated) ----------------- #

# HDF5 data set names never compared between the two Scan 1 groups:
# the EMheader time fields differ between runs by design, and the
# three per-scan image maps carry the shipped save() bug (every
# scan's images are the first map's, idx.hpp lines 350-353), so
# they are never asserted at all (D1)
_SCAN1_EXCLUDED_NAMES = frozenset(
    {"StartTime", "StopTime", "PatPerS", "IPF Map", "XC Map", "IQ Map"}
)


def _collect_datasets(group, prefix=""):
    """Return ``{relative path: array}`` of every data set under an
    HDF5 group, excluding the D1 exclusion names."""
    arrays = {}
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            arrays.update(_collect_datasets(item, path))
        elif name not in _SCAN1_EXCLUDED_NAMES:
            arrays[path] = np.asarray(item[()])
    return arrays


@functools.lru_cache(maxsize=1)
def _small_signal_and_detector():
    """Return the background corrected small signal and its single
    projection centre detector, the Phase 10 canonical inputs."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average
    return signal, detector


def _write_small_run(run_dir, psym_file="", n_masters=1):
    """Write the Phase 10 canonical small-scenario inputs into
    ``run_dir`` and return the namelist file name.

    ``ipath`` stays empty and every path is CWD relative: with a
    set ``psymfile`` the C++ prepends ``ipath`` to ``patFile`` a
    SECOND time and never prefixes ``pSymFile`` (nml.hpp line 247),
    so only an empty ``ipath`` keeps the run well formed (D1).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    signal, detector = _small_signal_and_detector()
    write_emsphinx_patterns(run_dir / "patterns.h5", signal, overwrite=True)
    master = Dataset(NI_SMALL_SHT).fetch_file_path()
    master_names = []
    for i in range(n_masters):
        name = f"ni{i}.sht"
        shutil.copy(master, run_dir / name)
        master_names.append(name)
    namelist = EMSphInxNamelist.from_kwargs(
        pattern_file="patterns.h5",
        master_files=master_names,
        detector=detector,
        scan_shape=(3, 3),
        scan_steps=(1.5, 1.5),
        data_file="out.h5",
        vendor_file="out.ang",
        vendor="Bruker",
        delta=500.0,
        n_thread=1,
        batch_size=1,
        bandwidth=NI_BANDWIDTH,
        normalize=True,
        refine=True,
        n_regions=10,
        gaussian_background=False,
        circular_mask=False,
    )
    namelist.psym_file = psym_file
    namelist.write(run_dir / "index.nml", overwrite=True)
    return "index.nml"


def _run_index_ebsd(program, run_dir):
    """Run ``IndexEBSD`` on the namelist in ``run_dir``."""
    return subprocess.run(
        [str(program("IndexEBSD")), "index.nml"],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )


class TestIndexEBSDPsymFile:
    """The D1 inertness pin and the D5/D6 error paths against the
    shipped ``IndexEBSD.exe`` at 60f3517.  [D1, D5, D6, D8.3]"""

    def test_psymfile_is_inert_at_60f3517(self, emsphinx_program, tmp_path):
        # the executable evidence of the D1 baseline: the shipped
        # CLI loads a psymfile and allocates the extra scan, but
        # never wires the operators into the Indexer, so Scan 1 is
        # data-set-level identical and Scan 2 holds the converted
        # placeholder (phase 255, identity orientation, zero metric
        # and image quality) -- NOT the raw corr=0/phase=-1/qu=0
        # values.  [D1, D8.3]
        plain_dir = tmp_path / "plain"
        _write_small_run(plain_dir)
        result = _run_index_ebsd(emsphinx_program, plain_dir)
        assert result.returncode == 0, (
            f"IndexEBSD without psymfile failed ({result.returncode}): "
            f"{result.stdout} {result.stderr} -- a corrupt machine-wide "
            "fftw.wisdom is suspect number one"
        )

        psym_dir = tmp_path / "psym"
        psym_dir.mkdir()
        write_text(psym_dir / "ops.txt", GOLDEN_FILE_BODY)
        _write_small_run(psym_dir, psym_file="ops.txt")
        result = _run_index_ebsd(emsphinx_program, psym_dir)
        assert result.returncode == 0, (
            f"IndexEBSD with psymfile failed ({result.returncode}): "
            f"{result.stdout} {result.stderr} -- a corrupt machine-wide "
            "fftw.wisdom is suspect number one"
        )

        with (
            h5py.File(plain_dir / "out.h5", "r") as plain,
            h5py.File(psym_dir / "out.h5", "r") as psym,
        ):
            plain_scans = [k for k in plain if k.startswith("Scan")]
            psym_scans = [k for k in psym if k.startswith("Scan")]
            assert plain_scans == ["Scan 1"]
            # one operator: one extra scan, and ``padNum`` stays 1
            # so the group names are unpadded
            assert psym_scans == ["Scan 1", "Scan 2"]

            ours = _collect_datasets(plain["Scan 1"])
            theirs = _collect_datasets(psym["Scan 1"])
            assert set(ours) == set(theirs)
            for path in ours:
                assert np.array_equal(ours[path], theirs[path]), (
                    f"Scan 1/{path} differs between the psymfile and "
                    "no-psymfile runs, refuting the D1 inertness baseline"
                )

            data = psym["Scan 2/EBSD/Data"]
            phase = np.asarray(data["Phase"])
            assert phase.dtype == np.uint8
            assert (phase == 255).all()
            assert (np.asarray(data["Metric"]) == 0).all()
            assert (np.asarray(data["IQ"]) == 0).all()
            # the placeholder row is converted like every result:
            # ZYZ (0, 0, 0) -> the identity -> its Bunge Euler
            euler = np.stack(
                [
                    np.asarray(data["Phi1"], dtype=np.float64),
                    np.asarray(data["Phi"], dtype=np.float64),
                    np.asarray(data["Phi2"], dtype=np.float64),
                ],
                axis=-1,
            )
            identity_angle = np.rad2deg(Rotation.from_euler(euler).angle)
            assert (identity_angle < 1e-4).all()

    def test_eu_type_psymfile_rejected(self, emsphinx_program, tmp_path):
        # only ``qu`` angle files are accepted for pseudo-symmetry
        # (master.hpp line 225).  The thrown message is pinned so the
        # run fails FOR the psym reason, not for some unrelated one
        # (``index_ebsd.cpp`` line 193 prints ``e.what()`` to
        # stdout).  [D6]
        tmp_path.mkdir(exist_ok=True)
        write_text(tmp_path / "ops.txt", "eu\n1\n10.0 20.0 30.0\n")
        _write_small_run(tmp_path, psym_file="ops.txt")
        result = _run_index_ebsd(emsphinx_program, tmp_path)
        assert result.returncode != 0
        assert "only quaternion angle files are supported" in (
            result.stdout + result.stderr
        )

    def test_psymfile_count_mismatch_rejected(self, emsphinx_program, tmp_path):
        # too few numbers for the count (emsoft.hpp line 140; the
        # "orientions" typo is the binary's own).  [D6]
        write_text(tmp_path / "ops.txt", "qu\n3\n1 0 0 0\n")
        _write_small_run(tmp_path, psym_file="ops.txt")
        result = _run_index_ebsd(emsphinx_program, tmp_path)
        assert result.returncode != 0
        assert "not enough orientions in angle file" in (result.stdout + result.stderr)

    def test_two_phase_with_psymfile_rejected(self, emsphinx_program, tmp_path):
        # "psuedo-symmetry files currently only supported for
        # single phase indexing" (idx.hpp lines 190-192), the D5
        # single-phase rule's upstream anchor.  The message pin
        # matters most here: a two-master run can fail for reasons
        # which have nothing to do with the psymfile.  [D5]
        write_text(tmp_path / "ops.txt", GOLDEN_FILE_BODY)
        _write_small_run(tmp_path, psym_file="ops.txt", n_masters=2)
        result = _run_index_ebsd(emsphinx_program, tmp_path)
        assert result.returncode != 0
        assert (
            "psuedo-symmetry files currently only supported for single phase indexing"
            in (result.stdout + result.stderr)
        )


# ------------------- MasterXcorr parity (gated) --------------------- #


def _full_ni_h5_path():
    """Return the cached full EMsoft Ni master h5, skipping when it
    is not obtainable (no pooch, or not in the local cache).

    The ``Dataset`` is constructed OUTSIDE the try: a registry typo
    raises ``KeyError`` loudly instead of becoming a permanent
    silent skip.  Only the can't-obtain cases convert to a skip:
    ``ImportError`` (no pooch) and ``ValueError`` (not cached, or a
    cache hash mismatch needing a re-download -- the skip reason
    carries the message either way).
    """
    dataset = Dataset("ebsd_master_pattern/ni_mc_mp_20kv.h5")
    try:
        return Path(dataset.fetch_file_path())
    except (ImportError, ValueError) as error:  # pragma: no cover
        pytest.skip(f"the full Ni master h5 is not obtainable: {error}")


def _run_masterxcorr(program, cwd, arguments):
    """Run ``MasterXcorr`` in an isolated CWD (it hard-writes four
    files there) and return its parsed stdout."""
    result = subprocess.run(
        [str(program("MasterXcorr")), *[str(a) for a in arguments]],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"MasterXcorr failed ({result.returncode}): {result.stdout} "
        f"{result.stderr} -- a corrupt machine-wide fftw.wisdom is "
        "suspect number one"
    )
    return parse_masterxcorr_stdout(result.stdout)


class TestMasterXcorrParity:
    """Prediction parity against ``MasterXcorr.exe`` (D8.1/D8.2).
    Oracle bandwidths satisfy ``fast_size(2 bw - 1) == 2 bw - 1``
    and the binary's [53, 313] clamp (D8)."""

    def _assert_bandwidth_discipline(self):
        assert fast_size(2 * MASTERXCORR_BANDWIDTH - 1) == (
            2 * MASTERXCORR_BANDWIDTH - 1
        )
        assert 53 <= MASTERXCORR_BANDWIDTH <= 313

    @pytest.mark.weekly
    def test_masterxcorr_stdout_parity(
        self, emsphinx_program, tmp_path, record_property
    ):
        # the prediction oracle (D8.1): kikuchipy's returned
        # operators are the CONJUGATE of the printed rows at the
        # stdout's own precision, the intensities match, and both
        # engines return the measured proper-Oh set (22 rows at
        # cutoff 0.9, identity absent -- binary-measured
        # 2026-09-07).  At-bandwidth ``from_master_pattern``
        # construction is the parity route (D3.1).
        # [D8.1, D2.6d, D2.7, D3]
        self._assert_bandwidth_discipline()
        ni_h5 = _full_ni_h5_path()
        theirs, v_max = _run_masterxcorr(
            emsphinx_program,
            tmp_path,
            [MASTERXCORR_BANDWIDTH, MASTERXCORR_CUTOFF, ni_h5],
        )
        record_property("masterxcorr_v_max", v_max)
        assert v_max == pytest.approx(MASTERXCORR_VMAX_AUTO, rel=0.05)
        assert theirs.shape[0] == MASTERXCORR_ROW_COUNT

        master = kp.data.ebsd_master_pattern(
            "ni", projection="lambert", hemisphere="both", allow_download=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            harmonics = MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=MASTERXCORR_BANDWIDTH
            )
        ours = find_pseudo_symmetry_operators(
            harmonics,
            bandwidth=MASTERXCORR_BANDWIDTH,
            cutoff=MASTERXCORR_CUTOFF,
            exclude_symmetry=False,
        )
        assert ours.operators.size == theirs.shape[0]

        printed = Rotation(theirs[:, 1:])
        # match each printed row to our nearest operator, then
        # assert the CONJUGATE relation component-wise: ours must
        # equal ``~printed`` within the stdout precision.  For the
        # inversion-closed Oh set this is conjugation-blind (stated
        # honestly, D2.6); the two-phase discriminators carry the
        # direction burden.  The matching must be a BIJECTION: with
        # count parity alone, a duplicated match would mask a
        # missing peak
        our_data = ours.operators.flatten().data
        matched = set()
        for i in range(printed.size):
            conjugated = (~printed[i]).data.ravel()
            deviation = np.minimum(
                np.abs(our_data - conjugated).max(axis=-1),
                np.abs(our_data + conjugated).max(axis=-1),
            )
            best = int(np.argmin(deviation))
            assert deviation[best] < MASTERXCORR_QUAT_ATOL
            assert best not in matched, (
                f"printed row {i} matched kikuchipy operator {best}, "
                "which an earlier printed row already claimed"
            )
            matched.add(best)
            # the MEASURED-THEN-PINNED intensity band (measured max
            # relative deviation 2.852e-5 on this route, 2026-09-07;
            # see the constant)
            assert ours.intensities[best] == pytest.approx(
                theirs[i, 0], rel=MASTERXCORR_INTENSITY_RTOL
            )
        # both engines return proper-Oh rotations only
        assert (angles_to_proper_oh(printed) < NI_OH_ANGLE_TOL_DEG).all()
        assert (angles_to_proper_oh(ours.operators) < NI_OH_ANGLE_TOL_DEG).all()

    def test_two_file_branch_same_master(self, emsphinx_program, tmp_path):
        # the mandatory two-phase-mode discriminator (D8.2i) with
        # no new data: the same master h5 handed over TWICE.  The
        # C++ takes the auto-correlation branch on FILENAME STRING
        # equality (master_xcorr.cpp line 87) -- measured
        # 2026-09-07: the literal same path twice is byte-identical
        # to the single-file run -- so the two spellings below name
        # the same file with different strings, which flips the
        # reference maximum from the identity-cell seed (0.719309)
        # to the coarse-argmax seed (0.972597) and rescales every
        # intensity (1.0000/0.8936 against 1.3521/1.2082).
        # kikuchipy's two-file mode is entered by passing
        # ``second_harmonics``, here the same object.
        # [D8.2i, D2.6c, D3.3]
        self._assert_bandwidth_discipline()
        ni_h5 = _full_ni_h5_path()
        spelling_one = str(ni_h5)
        spelling_two = str(ni_h5).replace("\\", "/")
        assert spelling_one != spelling_two
        theirs, v_max = _run_masterxcorr(
            emsphinx_program,
            tmp_path,
            [MASTERXCORR_BANDWIDTH, 0.5, spelling_one, spelling_two],
        )
        # the argmax-seeded reference maximum: the two-file branch
        # ran (binary-measured 2026-09-07)
        assert v_max == pytest.approx(MASTERXCORR_VMAX_TWO_FILE, rel=0.05)
        assert theirs[0, 0] == pytest.approx(1.0, abs=0.01)

        master = kp.data.ebsd_master_pattern(
            "ni", projection="lambert", hemisphere="both", allow_download=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            harmonics = MasterPatternHarmonics.from_master_pattern(
                master, bandwidth=MASTERXCORR_BANDWIDTH
            )
        ours = find_pseudo_symmetry_operators(
            harmonics,
            harmonics,
            bandwidth=MASTERXCORR_BANDWIDTH,
            cutoff=0.5,
            exclude_symmetry=False,
        )
        # two-file normalization: the brightest peak is the
        # reference itself
        assert ours.intensities[0] == pytest.approx(1.0, rel=MASTERXCORR_INTENSITY_RTOL)
        assert ours.operators.size == theirs.shape[0]

    @pytest.mark.weekly
    def test_two_master_parity_al(self, emsphinx_program, tmp_path, record_property):
        # the true two-master run (D8.2ii, open question 9.7):
        # cross-master peaks between Ni and Al against the binary.
        # HONESTY NOTE (review-corrected 2026-09-07): Ni and Al are
        # BOTH fcc m-3m in the same EMsoft setting, so the
        # cross-master peak set is (near-)identity composed with
        # proper Oh -- a group, hence inversion closed, hence as
        # conjugation-blind as the Ni autocorrelation parity above.
        # This run pins two-master MODE mechanics (count,
        # intensities, argmax-seeded v_max) against the binary; NO
        # binary test anywhere compares kikuchipy's quaternion
        # DIRECTION -- that burden rests wholly on the pure-Python
        # rotated-copy oracle (``test_two_phase_rotated_copy``,
        # D2.6c, machine-verified: inverse 97.9 deg away) and the D2
        # derivation, a recorded oracle gap in validation.md.  Skips
        # cleanly when the ~0.3 GB Al master is not obtainable.
        # [D8.2ii, D2.6c]
        self._assert_bandwidth_discipline()
        ni_h5 = _full_ni_h5_path()
        al_dataset = Dataset("ebsd_master_pattern/al_mc_mp_20kv.h5")
        try:
            al_h5 = Path(al_dataset.fetch_file_path())
        except (ImportError, ValueError) as error:  # pragma: no cover
            pytest.skip(f"the full Al master h5 is not obtainable: {error}")
        theirs, v_max = _run_masterxcorr(
            emsphinx_program,
            tmp_path,
            [MASTERXCORR_BANDWIDTH, 0.5, ni_h5, al_h5],
        )
        record_property("two_master_v_max", v_max)
        record_property("two_master_rows", theirs.tolist())

        ni_master = kp.data.ebsd_master_pattern(
            "ni", projection="lambert", hemisphere="both", allow_download=True
        )
        al_master = kp.data.ebsd_master_pattern(
            "al", projection="lambert", hemisphere="both", allow_download=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ni_h = MasterPatternHarmonics.from_master_pattern(
                ni_master, bandwidth=MASTERXCORR_BANDWIDTH
            )
            al_h = MasterPatternHarmonics.from_master_pattern(
                al_master, bandwidth=MASTERXCORR_BANDWIDTH
            )
        ours = find_pseudo_symmetry_operators(
            ni_h,
            al_h,
            bandwidth=MASTERXCORR_BANDWIDTH,
            cutoff=0.5,
            exclude_symmetry=False,
        )
        assert ours.operators.size == theirs.shape[0]
        printed = Rotation(theirs[:, 1:])
        our_data = ours.operators.flatten().data
        matched = set()
        for i in range(printed.size):
            conjugated = (~printed[i]).data.ravel()
            deviation = np.minimum(
                np.abs(our_data - conjugated).max(axis=-1),
                np.abs(our_data + conjugated).max(axis=-1),
            )
            best = int(np.argmin(deviation))
            assert deviation[best] < MASTERXCORR_QUAT_ATOL
            assert best not in matched, (
                f"printed row {i} matched kikuchipy operator {best}, "
                "which an earlier printed row already claimed"
            )
            matched.add(best)
            assert ours.intensities[best] == pytest.approx(
                theirs[i, 0], rel=MASTERXCORR_INTENSITY_RTOL
            )
