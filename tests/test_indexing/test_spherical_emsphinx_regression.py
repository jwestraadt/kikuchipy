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

"""Regression of ``EBSD.spherical_indexing`` against EMSphInx'
``IndexEBSD`` program.

Covers every named assertion of
``specs/2026-09-03-spherical-indexing-emsphinx-regression/
validation.md``.  The eight shipped
``src/kikuchipy/data/emsphinx/regression_*.npz`` files hold what
``IndexEBSD.exe`` at commit ``60f3517`` wrote for eight configurations
of kikuchipy's own inputs, together with the provenance needed to
reproduce those inputs, so the comparison needs no binary:

- ``TestReferenceFiles``: the registry, the scenario set completeness,
  the frozen key set and data types, the provenance pins and the file
  size budget.
- ``TestReferenceIntegrity``: frozen bytes against frozen bytes.
- ``TestRoutePins``: the name list and the pattern centre recomputed
  from the declared route with kikuchipy's own conversion code.
- ``TestOursVsTheirsSmall`` and ``TestOursVsTheirsLarge``: **the parity
  surface**, orientations, scores and image quality of our engine
  against the references on identical inputs, plus the stretch
  emulation which decomposes the residual.
- ``TestRegenerateReferences``: behind ``KIKUCHIPY_EMSPHINX_DIR``, a
  rerun of the generation script reproduces the shipped files
  bitwise.

The comparison inputs are reconstructed rather than read from a
binary artefact: on the canonical route the repacked pattern file is
byte identical to ``signal.data.reshape(-1, h, w)``, so indexing the
background corrected signal *is* indexing the file the program read.
"""

import functools
import hashlib
from pathlib import Path
import time

import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset
from kikuchipy.data._registry import _registry_hashes
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._namelist import EMSphInxNamelist

# ------------------------- Frozen constants ------------------------- #

# The EMSphInx commit the references were generated from.  This is a
# **deliberate duplicate** of the literal in
# ``kikuchipy.data.emsphinx.create_emsphinx_reference``: the value
# stored in every reference file is probed from the checkout at
# generation time, so a stale pin in the script would store itself and
# agree with itself.  ``test_provenance_pins`` has power only because
# this literal is maintained here, independently of the script's.
EMSPHINX_COMMIT = "60f351741036c63a59a6061a7ac2fca4f60f2c64"

NI_SHT = "emsphinx/ni_small_20kv_bw384.sht"

# The one master pattern, sample tilt and bandwidth of every scenario
SAMPLE_TILT = 70.0
BANDWIDTH = 68

# Every array a reference file holds, which ``test_frozen_keys_and_
# dtypes`` pins exactly: the six result arrays of the data file's
# ``Scan 1/EBSD/Data`` group and the provenance the tests recompute
# against
RESULT_KEYS = ("phi1", "phi", "phi2", "metric", "iq", "phase")
PROVENANCE_KEYS = (
    "emsphinx_commit",
    "bw",
    "normed",
    "refine",
    "nregions",
    "gausbckg",
    "delta",
    "vendor",
    "route",
    "dataset",
    "scan_shape",
    "scan_steps",
    "sample_tilt",
    "pc",
    "namelist",
    "master_sht",
    "master_md5",
    "patterns_md5",
    "preprocessing",
    "subset_slice",
    "emsphinx_compatible",
    "manufacturer",
    "flip",
    "kikuchipy_version",
)

# The scenario matrix, frozen here as the test module's own table.
# Together with the script's :data:`SCENARIOS`, the registry keys and
# the directory glob it forms the quad equality of
# ``test_scenario_set_is_complete``, so a scenario added or renamed
# without regenerating and registering fails on CI, with no binaries.
SCENARIO_TABLE = {
    "small_coarse_nr10": {
        "dataset": "nickel_ebsd_small",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": False,
        "nregions": 10,
        "gausbckg": False,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "small_refined_nr10": {
        "dataset": "nickel_ebsd_small",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 10,
        "gausbckg": False,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "small_refined_nr0": {
        "dataset": "nickel_ebsd_small",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 0,
        "gausbckg": False,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "small_refined_nr7": {
        "dataset": "nickel_ebsd_small",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 7,
        "gausbckg": False,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "small_refined_nr10_gb": {
        "dataset": "nickel_ebsd_small",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 10,
        "gausbckg": True,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "small_refined_emsoft_d500": {
        "dataset": "nickel_ebsd_small",
        "vendor": "EMsoft",
        "delta": 500.0,
        "refine": True,
        "nregions": 10,
        "gausbckg": False,
        "scan_shape": (3, 3),
        "scan_steps": (1.5, 1.5),
        "subset_slice": "",
    },
    "large20_refined_nr10": {
        "dataset": "nickel_ebsd_large_20pt",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 10,
        "gausbckg": False,
        "scan_shape": (4, 5),
        "scan_steps": (22.5, 22.5),
        "subset_slice": "::15,::15",
    },
    "large165_refined_nr10": {
        "dataset": "nickel_ebsd_large_165pt",
        "vendor": "Bruker",
        "delta": 500.0,
        "refine": True,
        "nregions": 10,
        "gausbckg": False,
        "scan_shape": (11, 15),
        "scan_steps": (7.5, 7.5),
        "subset_slice": "::5,::5",
    },
}

SMALL_SCENARIOS = [
    name
    for name, entry in SCENARIO_TABLE.items()
    if entry["dataset"] == "nickel_ebsd_small"
]
ANCHOR = "small_refined_nr10"

# The indexing arguments each reference must be compared under,
# frozen as literals.  ``scenario_kwargs`` derives the same dictionary
# from the file's provenance, and ``test_scenario_kwargs_derive_from_
# provenance`` pins the two against one another: this is the
# **structural** killer of a harness which ignores the provenance,
# because a wrong ``refine`` flag has no band signature (measured:
# indexing coarse against the refined reference gives a median of
# 0.431 and a maximum of 0.668 degrees, inside every band, and the
# coarse and refined image quality are identical by construction).
KWARGS_TABLE = {
    "small_coarse_nr10": {
        "bandwidth": 68,
        "normalize": True,
        "refine": False,
        "n_regions": 10,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "small_refined_nr10": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 10,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "small_refined_nr0": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 0,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "small_refined_nr7": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 7,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "small_refined_nr10_gb": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 10,
        "gaussian_background": True,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "small_refined_emsoft_d500": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 10,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "large20_refined_nr10": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 10,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
    "large165_refined_nr10": {
        "bandwidth": 68,
        "normalize": True,
        "refine": True,
        "n_regions": 10,
        "gaussian_background": False,
        "circular_mask": False,
        "emsphinx_compatible": True,
    },
}

# The file names ``from_kwargs`` writes into the name list, the same
# three the generation script uses inside its run directory
PATTERN_FILE = "patterns.h5"
MASTER_FILE = "ni.sht"
DATA_FILE = "out.h5"
VENDOR_FILE = "out.ang"

# --------------------------- The bands ------------------------------ #
#
# Measured 2026-09-03 against the shipped references on this machine
# and pinned with the Phase 6 margin convention of about 1.7 to 2.1
# times the measured worst value.

# Coarse scenario.  The maximum carries the outlier convention rather
# than a plain bound: one correlation grid cell at ``bw`` 68 is
# 360 / fast_size(135) = 2.667 degrees, which is larger than the whole
# band, so a single argmax landing one cell over on another platform
# is a legitimate near tie.  Eight of the nine points must stay under
# the tight maximum and every point under the single cell ceiling.
COARSE_MEDIAN_DEG = 1.0  # measured 0.510
COARSE_TIGHT_MAX_DEG = 1.25  # measured 0.622
COARSE_TIGHT_COUNT = 8  # of nine
# the measured maximum plus one grid cell is 0.622 + 2.667 = 3.29,
# rounded up to the round number below
COARSE_CEILING_DEG = 4.0

# Refined scenarios keep plain maxima: their margins are at least
# twice the measurement and Newton refinement re-converges a near tie
REFINED_MEDIAN_DEG = 0.7  # measured 0.310 to 0.341
REFINED_MAX_DEG = 0.75  # measured 0.335 to 0.367

# The large subsets
LARGE20_MAX_DEG = 0.8  # measured 0.373
LARGE165_P95_DEG = 0.9  # measured 0.441
LARGE165_MAX_DEG = 1.0  # measured 0.487

# Scores.  Both metrics are normalized but differ by about two per
# cent systematically, so this is a correlation plus a band and never
# an equality
SCORE_R_SMALL = 0.85  # measured 0.935 to 0.969
SCORE_R_LARGE20 = 0.90  # measured 0.973
SCORE_R_LARGE165 = 0.88  # measured 0.944
SCORE_MEAN_DIFF = 0.03  # measured 0.0088 to 0.0139
SCORE_MAX_DIFF = 0.07  # measured 0.0226 to 0.0364

# Image quality, the preprocessing discriminator.  The ladder:
# measured parity on this machine is at most 1.4e-8; one flipped
# uint8 gray level in one pixel moves that pattern's image quality by
# up to 5.2e-5 through the equalisation histogram cascade, and the
# numba ``fastmath`` background removal leaves a handful of pixels
# within a few units in the last place of a truncation boundary, so
# the cross platform drift budget is about 2e-4.  The smallest *real*
# signature the band must still catch is 1.5e-2 (the Gaussian
# background scenario against the default one), so 1e-3 sits about
# five times above the drift budget and at least fifteen times below
# the smallest kill.
IQ_MAX_DIFF = 1e-3

# The item 31 stretch emulation, the evidence the mission's original
# gate rests on: emulating EMSphInx' un-ported ``bilinearCoeff``
# sampling stretch in pattern centre space collapses the residual
# (measured median 0.094 against the unmodified 0.340)
STRETCH_MEDIAN_DEG = 0.2

# Reference integrity: the EMsoft and Bruker pattern centre routes
# describe the same geometry and differ only in the sixth decimal of
# the ``.6g`` rounded name list values (measured 5.83e-5 degrees)
EMSOFT_VS_BRUKER_DEG = 1e-3

# The constitution's per file rule for in-package data
FILE_BUDGET_BYTES = 100_000


# ----------------------------- Helpers ------------------------------ #


def reference_directory():
    """Return the directory the shipped references live in.

    Anchored at the **installed** package, not at ``src``: the
    ``build-install-wheel`` job runs the suite from the installed
    wheel, where ``src`` does not exist.
    """
    return Path(kp.data.__file__).parent / "emsphinx"


def registry_keys():
    """Return the registry keys of the shipped references."""
    return {
        key
        for key in _registry_hashes
        if key.startswith("emsphinx/regression_") and key.endswith(".npz")
    }


def reference_name(key_or_path):
    """Return the scenario name of a registry key or file name."""
    return Path(str(key_or_path)).name[len("regression_") : -len(".npz")]


@functools.lru_cache(maxsize=None)
def load_reference(name):
    """Return one shipped reference as a plain dictionary of read only
    arrays, fetched through the data module so that its registry md5
    is verified on the way in.

    ``allow_pickle=False``: the provenance is stored as fixed width
    unicode arrays and never as pickled objects.  The ``NpzFile`` is
    read out and **closed** rather than cached open: this cache lives
    for the whole session, and an open handle keeps the shipped file
    locked on Windows.  The arrays are made read only because every
    caller now shares one object instead of getting a fresh read.
    """
    fpath = Dataset(f"emsphinx/regression_{name}.npz").fetch_file_path()
    with np.load(fpath, allow_pickle=False) as reference:
        arrays = {key: reference[key] for key in reference.files}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def reference_path(name):
    """Return the path of one shipped reference."""
    return Path(Dataset(f"emsphinx/regression_{name}.npz").fetch_file_path())


def scenario_kwargs(ref):
    """Return the indexing arguments a reference must be compared
    under, derived from its own provenance.

    Pinned against :data:`KWARGS_TABLE` by
    ``test_scenario_kwargs_derive_from_provenance``.
    """
    return {
        "bandwidth": int(ref["bw"]),
        "normalize": bool(ref["normed"]),
        "refine": bool(ref["refine"]),
        "n_regions": int(ref["nregions"]),
        "gaussian_background": bool(ref["gausbckg"]),
        "circular_mask": False,
        "emsphinx_compatible": bool(ref["emsphinx_compatible"]),
    }


def frozen_namelist(name, detector):
    """Return the name list the declared route builds for a scenario,
    from this module's frozen table and a detector.

    The detector carries the **unrounded** ``pc_average``, which is
    what the generation script passed; ``to_string`` is what rounds it
    to six significant digits.
    """
    entry = SCENARIO_TABLE[name]
    indexing = {
        key: value
        for key, value in KWARGS_TABLE[name].items()
        if key != "emsphinx_compatible"
    }
    return EMSphInxNamelist.from_kwargs(
        pattern_file=PATTERN_FILE,
        master_files=[MASTER_FILE],
        detector=detector,
        scan_shape=entry["scan_shape"],
        scan_steps=entry["scan_steps"],
        data_file=DATA_FILE,
        vendor_file=VENDOR_FILE,
        vendor=entry["vendor"],
        delta=entry["delta"],
        n_thread=1,
        batch_size=1,
        **indexing,
    )


def their_orientations(ref):
    """Return the reference orientations under m-3m.

    The stored angles are EMSphInx' ``qu2eu`` Bunge triples in
    **radians**, so ``from_euler`` takes its defaults.  Both sides of
    every comparison are :class:`~orix.quaternion.Orientation` with
    the symmetry: a bare ``Rotation`` on this side raises an
    ``AttributeError`` inside ``angle_with`` (orix 0.14.2), and a
    comparison without symmetry on both sides is degree scale wrong.
    """
    euler = np.stack([ref["phi1"], ref["phi"], ref["phi2"]], axis=1).astype(np.float64)
    return Orientation(Rotation.from_euler(euler).data, Oh)


def misorientation(rotations, ref):
    """Return the m-3m reduced misorientation in degrees between our
    rotations and a reference's, one value per scan point.
    """
    angles = Orientation(rotations.data, Oh).angle_with(
        their_orientations(ref), degrees=True
    )
    return np.asarray(angles, dtype=np.float64).ravel()


def stretched_pc(pc, shape):
    """Return the pattern centre which emulates EMSphInx' un-ported
    ``bilinearCoeff`` detector sampling stretch.

    The stretch is ``x = X (w - 1)`` where kikuchipy uses pixel
    centres; its exact equivalent in pattern centre space is
    ``pc' = (0.5/w + pc_x (w-1)/w, 0.5/h + pc_y (h-1)/h,
    pc_z (w-1)/w)``.
    """
    height, width = shape
    return np.array(
        [
            0.5 / width + pc[0] * (width - 1) / width,
            0.5 / height + pc[1] * (height - 1) / height,
            pc[2] * (width - 1) / width,
        ]
    )


def record_comparison(record_property, tag, angles, ours, ref):
    """Record the measured misorientation, score and image quality
    statistics of one comparison.
    """
    scores = np.asarray(ours.prop["scores"], dtype=np.float64).ravel()
    metric = ref["metric"].astype(np.float64)
    iq = np.asarray(ours.prop["iq"], dtype=np.float64).ravel()
    record_property(f"{tag}_median", f"{np.median(angles):.4f}")
    record_property(f"{tag}_p95", f"{np.percentile(angles, 95):.4f}")
    record_property(f"{tag}_max", f"{angles.max():.4f}")
    record_property(f"{tag}_score_r", f"{pearson(scores, metric):.4f}")
    record_property(
        f"{tag}_score_diff",
        f"mean {np.abs(scores - metric).mean():.5f} "
        f"max {np.abs(scores - metric).max():.5f}",
    )
    record_property(
        f"{tag}_iq_max_diff", f"{np.abs(iq - ref['iq'].astype(np.float64)).max():.3e}"
    )


def pearson(one, two):
    """Return the Pearson correlation coefficient of two arrays."""
    return float(np.corrcoef(np.asarray(one), np.asarray(two))[0, 1])


def assert_scores(ours, ref, r_band):
    """Assert the score correlation and difference bands.

    Never an equality: the two metrics are both normalized but differ
    by about two per cent systematically (measured mean absolute
    difference 0.0088 to 0.0139 over a range of about 0.1).
    """
    scores = np.asarray(ours.prop["scores"], dtype=np.float64).ravel()
    metric = ref["metric"].astype(np.float64)
    difference = np.abs(scores - metric)
    assert pearson(scores, metric) > r_band
    assert difference.mean() < SCORE_MEAN_DIFF
    assert difference.max() < SCORE_MAX_DIFF


def patterns_md5(signal):
    """Return the md5 of the ``uint8`` pattern block a signal repacks
    to.

    On the canonical route ``write_emsphinx_patterns`` writes
    ``signal.data.reshape(-1, h, w)`` verbatim, which the generation
    script asserts, so this is the md5 of the ``/patterns`` data set
    ``IndexEBSD`` read and is directly comparable with the reference's
    ``patterns_md5``.
    """
    height, width = signal.axes_manager.signal_shape[::-1]
    block = np.ascontiguousarray(np.asarray(signal.data).reshape(-1, height, width))
    return hashlib.md5(block.tobytes()).hexdigest()


def assert_iq(ours, ref, signal=None):
    """Assert the image quality near equality band.

    The image quality is computed from the *processed* pattern, so it
    is the sharpest cross engine probe of the preprocessing -- and it
    is chained to the exact ``uint8`` pattern bytes, which is why the
    message carries both ``patterns_md5`` values: a platform whose
    background removal rounds differently fails here as "your patterns
    differ from the reference machine's, here is the sum to compare",
    not as an unexplained miss.

    The two md5 sums are deliberately **not** asserted equal: byte
    equality of the preprocessed patterns is far stricter than the
    1e-3 band this band was chosen to be (D6), and a platform which
    rounds one gray level differently must still pass.
    """
    if signal is None:
        signal = small_signal()
    iq = np.asarray(ours.prop["iq"], dtype=np.float64).ravel()
    worst = float(np.abs(iq - ref["iq"].astype(np.float64)).max())
    ours_md5 = patterns_md5(signal)
    assert worst < IQ_MAX_DIFF, (
        f"the image quality differs by {worst:.3e}, beyond the "
        f"{IQ_MAX_DIFF:.0e} band. The reference was indexed from patterns "
        f"with md5 {str(ref['patterns_md5'])} and these patterns are "
        f"{ours_md5}; check that {str(ref['preprocessing'])} gives the same "
        "bytes here"
    )


@functools.lru_cache(maxsize=1)
def small_signal():
    """Return the background corrected ``nickel_ebsd_small`` signal.

    On the canonical route the repacked pattern file is byte identical
    to ``signal.data.reshape(-1, h, w)``, so this *is* the input the
    program indexed.
    """
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    return signal


@functools.lru_cache(maxsize=2)
def large_signal(step):
    """Return a background corrected ``nickel_ebsd_large`` subset and
    the full map's detector.

    The background is removed on the **full** map and the subset taken
    afterwards, and the detector keeps the full map's ``pc_average``,
    which is what the references were generated with.
    """
    pytest.importorskip("pooch")
    signal = kp.data.nickel_ebsd_large(allow_download=True)
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average
    return signal.inav[::step, ::step], detector


def detector_at(reference_pc, detector=None):
    """Return a detector with one projection centre, that of a
    reference.

    ``ref["pc"]`` is the ``.6g`` rounded triple the program actually
    read back from its name list, so both engines see the same
    geometry; the unrounded ``pc_average`` differs by about 3e-4
    pixels, which is far below every band.
    """
    if detector is None:
        detector = small_signal().detector
    detector = detector.deepcopy()
    detector.pc = np.asarray(reference_pc, dtype=np.float64)
    return detector


_RUNS = {}


def index_small(name, harmonics, pc=None):
    """Return the crystal map of one small map scenario, cached per
    module so that each configuration is indexed once.

    The harmonics are part of the key and are kept alive by the cache
    entry, so that ``id`` cannot be recycled: a scenario indexed
    against a different set -- a second phase, or spectra built at 68
    directly rather than resized from 384 -- must not silently receive
    the map of the session fixture, whose bands would then mean
    nothing while still passing.
    """
    key = (name, id(harmonics), None if pc is None else tuple(np.asarray(pc).ravel()))
    if key not in _RUNS:
        ref = load_reference(name)
        # the structural pin at the point of use: a mutant which hard
        # codes the arguments here rather than in ``scenario_kwargs``
        # survives ``test_scenario_kwargs_derive_from_provenance``,
        # and the refine flag has no band signature
        kwargs = scenario_kwargs(ref)
        assert kwargs == KWARGS_TABLE[name]
        detector = detector_at(ref["pc"] if pc is None else pc)
        _RUNS[key] = (
            harmonics,
            small_signal().spherical_indexing(harmonics, detector, verbose=0, **kwargs),
        )
    return _RUNS[key][1]


def md5_of_file(fpath):
    """Return the md5 sum of a file."""
    return hashlib.md5(Path(fpath).read_bytes()).hexdigest()


def regeneration_message(name, regenerated, shipped):
    """Return the failure message of a regeneration mismatch.

    Suspect number one is the FFTW wisdom: the programs plan with
    ``FFTW_PATIENT`` and import and export one machine wide wisdom
    file, and different plans round differently, so the reference
    bytes are a function of its state.  The per array difference is a
    diagnostic and never an acceptance tolerance -- the contract is
    bitwise.
    """
    lines = [
        f"the regenerated {name} is not bitwise identical to the shipped reference.",
        "Suspect #1 is the machine's FFTW wisdom: IndexEBSD plans with "
        "FFTW_PATIENT and imports and exports the shared fftw.wisdom at "
        "start and exit, so a changed wisdom can change the last bits.",
        "Per-array differences (diagnostic only, the contract is bitwise):",
    ]
    # both are closed again before returning: this runs inside a
    # failing assertion, and a handle left open on a file under
    # ``tmp_path`` blocks its removal on Windows
    with (
        np.load(regenerated, allow_pickle=False) as theirs,
        np.load(shipped, allow_pickle=False) as ours,
    ):
        for key in sorted(set(theirs.files) | set(ours.files)):
            if key not in theirs.files or key not in ours.files:
                lines.append(f"  {key}: only in one of the two files")
                continue
            one, two = theirs[key], ours[key]
            if one.dtype != two.dtype or one.shape != two.shape:
                lines.append(
                    f"  {key}: {one.dtype}{one.shape} vs {two.dtype}{two.shape}"
                )
            elif not np.array_equal(one, two):
                lines.append(f"  {key}: differs, {one!r} vs {two!r}")
    return "\n".join(lines)


def assert_regenerated_bitwise(written):
    """Assert every regenerated file equals its shipped twin."""
    for name, fpath in written.items():
        shipped = reference_path(name)
        if md5_of_file(fpath) != md5_of_file(shipped):
            raise AssertionError(regeneration_message(name, fpath, shipped))


@pytest.fixture(scope="session")
def ni_harmonics():
    """Return the in-package Ni master pattern harmonics.

    Read from the shipped ``.sht`` at its stored bandwidth of 384 and
    resized by the indexer to the indexing bandwidth, which is what
    ``IndexEBSD`` itself does; building the spectra at 68 directly is
    a different, non-parity route.

    Session scoped because the read and resize is the dominant fixed
    cost of the module: under ``pytest -n`` each worker pays it once.
    """
    return MasterPatternHarmonics.from_file(Dataset(NI_SHT).fetch_file_path())


# ==================== The reference files (D2) ====================== #


class TestReferenceFiles:
    def test_registry_lists_every_reference(self):
        # the glob is anchored at the installed package, not at
        # ``src``, which the build-install-wheel job does not have
        found = {path.name for path in reference_directory().glob("regression_*.npz")}
        keys = registry_keys()
        assert {Path(key).name for key in keys} == found
        assert len(found) == len(SCENARIO_TABLE)
        for key in keys:
            dataset = Dataset(key)
            assert dataset.is_in_package
            assert dataset.has_correct_hash, key

    def test_scenario_set_is_complete(self):
        # the quad equality: the generation script's own table, the
        # registry, the data directory and this module's frozen table.
        # Importing the script is what its import safety buys -- it
        # looks up no environment variable and probes no binary
        from kikuchipy.data.emsphinx import create_emsphinx_reference

        script = {scenario.name for scenario in create_emsphinx_reference.SCENARIOS}
        registry = {reference_name(key) for key in registry_keys()}
        directory = {
            reference_name(path)
            for path in reference_directory().glob("regression_*.npz")
        }
        frozen = set(SCENARIO_TABLE)
        assert script == registry == directory == frozen

    @pytest.mark.parametrize("name", sorted(SCENARIO_TABLE))
    def test_references_load_without_pickle(self, name):
        # no object arrays: pickled provenance would be both a
        # security surface and a version dependent byte stream
        with np.load(reference_path(name), allow_pickle=False) as reference:
            assert len(reference.files) > 0

    @pytest.mark.parametrize("name", sorted(SCENARIO_TABLE))
    def test_frozen_keys_and_dtypes(self, name):
        ref = load_reference(name)
        assert set(ref) == set(RESULT_KEYS + PROVENANCE_KEYS)

        rows, columns = SCENARIO_TABLE[name]["scan_shape"]
        for key in ("phi1", "phi", "phi2", "metric", "iq"):
            assert ref[key].dtype == np.float32, key
            assert ref[key].shape == (rows * columns,), key
        # the data file stores the phase as an unsigned 8-bit integer
        # and not as a float, which the Phase 9 record had wrong
        assert ref["phase"].dtype == np.uint8
        assert ref["phase"].shape == (rows * columns,)

        assert ref["pc"].dtype == np.float64
        assert ref["pc"].shape == (3,)
        # both are (rows, columns) pairs, so the shape is pinned as
        # well: a scalar or a three element scan shape would otherwise
        # only fail through ``test_provenance_pins``
        assert ref["scan_shape"].dtype == np.int64
        assert ref["scan_shape"].shape == (2,)
        assert ref["scan_steps"].dtype == np.float64
        assert ref["scan_steps"].shape == (2,)
        for key in ("bw", "nregions"):
            assert ref[key].dtype == np.int64, key
        for key in ("normed", "refine", "gausbckg", "emsphinx_compatible", "flip"):
            assert ref[key].dtype == np.bool_, key
        for key in ("delta", "sample_tilt"):
            assert ref[key].dtype == np.float64, key
        for key in (
            "emsphinx_commit",
            "vendor",
            "route",
            "dataset",
            "namelist",
            "master_sht",
            "master_md5",
            "patterns_md5",
            "preprocessing",
            "subset_slice",
            "manufacturer",
            "kikuchipy_version",
        ):
            assert ref[key].dtype.kind == "U", key
            assert ref[key].shape == (), key

    @pytest.mark.parametrize("name", sorted(SCENARIO_TABLE))
    def test_provenance_pins(self, name):
        ref = load_reference(name)
        entry = SCENARIO_TABLE[name]
        # the stored commit is probed from the checkout at generation
        # time, so this check has power only because the literal above
        # is maintained independently of the script's
        assert str(ref["emsphinx_commit"]) == EMSPHINX_COMMIT
        assert int(ref["bw"]) == BANDWIDTH
        assert bool(ref["normed"]) is True
        assert float(ref["sample_tilt"]) == SAMPLE_TILT
        assert np.array_equal(ref["phase"], np.zeros_like(ref["phase"]))

        assert bool(ref["refine"]) is entry["refine"]
        assert int(ref["nregions"]) == entry["nregions"]
        assert bool(ref["gausbckg"]) is entry["gausbckg"]
        assert str(ref["vendor"]) == entry["vendor"]
        assert float(ref["delta"]) == entry["delta"]
        assert str(ref["dataset"]) == entry["dataset"]
        assert str(ref["subset_slice"]) == entry["subset_slice"]
        assert tuple(ref["scan_shape"]) == entry["scan_shape"]
        assert tuple(ref["scan_steps"]) == entry["scan_steps"]

        # the canonical route and the master, which a sibling .sht or
        # a changed writer default would otherwise replace silently
        assert str(ref["manufacturer"]) == "EMsoft"
        assert bool(ref["flip"]) is False
        assert bool(ref["emsphinx_compatible"]) is True
        assert str(ref["master_sht"]) == Path(NI_SHT).name
        assert (
            f"md5:{str(ref['master_md5'])}"
            == _registry_hashes[f"emsphinx/{str(ref['master_sht'])}"]
        )

    def test_each_file_within_budget(self, record_property):
        # the constitution's rule is per file; the total is recorded
        sizes = {
            name: reference_path(name).stat().st_size for name in sorted(SCENARIO_TABLE)
        }
        record_property(
            "reference_bytes", ", ".join(f"{k} {v}" for k, v in sizes.items())
        )
        record_property("reference_bytes_total", sum(sizes.values()))
        for name, size in sizes.items():
            assert size < FILE_BUDGET_BYTES, name


# ============ Reference integrity, frozen bytes only (D6) =========== #


class TestReferenceIntegrity:
    """Guards which compare one shipped reference with another.

    They exercise **no kikuchipy code** and can only fire at
    regeneration or if the shipped data is edited, so they are not the
    parity surface: that is ``TestOursVsTheirs*`` plus the route pins.
    What they buy is that a generation which mixed two runs up, or
    wrote one run into two files, dies on CI without any binary.
    """

    def test_emsoft_route_is_close_but_distinct_from_bruker(self, record_property):
        one = load_reference("small_refined_emsoft_d500")
        two = load_reference(ANCHOR)
        angles = np.asarray(
            their_orientations(one).angle_with(their_orientations(two), degrees=True),
            dtype=np.float64,
        ).ravel()
        record_property("emsoft_vs_bruker_max_deg", f"{angles.max():.3e}")
        assert angles.max() < EMSOFT_VS_BRUKER_DEG
        # ... but not the same run: the two routes round the pattern
        # centre differently in the sixth decimal
        assert not np.array_equal(one["metric"], two["metric"])
        assert not np.array_equal(one["namelist"], two["namelist"])

    def test_coarse_and_refined_share_the_preprocessing(self):
        # refinement does not touch the pattern the image quality is
        # computed from, so the two are bitwise equal
        coarse = load_reference("small_coarse_nr10")
        refined = load_reference(ANCHOR)
        assert np.array_equal(coarse["iq"], refined["iq"])
        assert not np.array_equal(coarse["phi1"], refined["phi1"])

    def test_refined_metric_exceeds_coarse_per_point(self, record_property):
        # strict positivity only: an upper bound with seven per cent
        # headroom on the measured worst (+0.0187) would violate the
        # margin convention and pin nothing meaningful
        coarse = load_reference("small_coarse_nr10")
        refined = load_reference(ANCHOR)
        deltas = refined["metric"].astype(np.float64) - coarse["metric"].astype(
            np.float64
        )
        record_property(
            "refined_minus_coarse_metric",
            f"min {deltas.min():+.5f} max {deltas.max():+.5f}",
        )
        assert (deltas > 0).all()

    @pytest.mark.parametrize(
        "other", ["small_refined_nr0", "small_refined_nr7", "small_refined_nr10_gb"]
    )
    def test_preprocessing_scenarios_are_distinct(self, other, record_property):
        # the image quality is the preprocessing signature: the four
        # scenarios sit at measured ranges 0.173-0.204 (default),
        # 0.289-0.327 (no equalisation), 0.186-0.221 (the seven region
        # remainder path) and 0.187-0.216 (Gaussian background)
        anchor = load_reference(ANCHOR)
        reference = load_reference(other)
        record_property(
            f"iq_range_{other}",
            f"{reference['iq'].min():.5f}-{reference['iq'].max():.5f}",
        )
        assert not np.array_equal(anchor["iq"], reference["iq"])


# =============== The route, recomputed in kikuchipy (D6) ============ #


class TestRoutePins:
    @pytest.mark.parametrize("name", SMALL_SCENARIOS)
    def test_namelist_matches_the_declared_route(self, name):
        # one string comparison pinning the vendor, delta, pattern
        # centre, bandwidth, normed, refine, nregions, gausbckg,
        # circmask, thetac, patdims, scandims and every file name: a
        # reference generated on the wrong vendor route, or a drifted
        # kwargs table, dies here
        ref = load_reference(name)
        detector = small_signal().detector.deepcopy()
        detector.pc = detector.pc_average
        assert frozen_namelist(name, detector).to_string() == str(ref["namelist"])

    @pytest.mark.parametrize("name", SMALL_SCENARIOS)
    def test_pc_matches_the_stored_namelist(self, name):
        # exact float64 equality, which is safe **only** because
        # ``to_string`` quantised the pattern centre at six
        # significant digits before this round trip.  Do not loosen it
        # to an approximate comparison: the point is that ``ref["pc"]``
        # is what the program read, not what kikuchipy averaged.
        #
        # This is a regression guard on the round trip and **not** an
        # independent oracle for the stored value: it recomputes the
        # very expression the generation script used to produce
        # ``ref["pc"]``, so a wrongly generated pattern centre would
        # agree with itself here.  The independent oracle is
        # ``test_namelist_matches_the_declared_route``, which rebuilds
        # the whole name list from kikuchipy's own detector
        ref = load_reference(name)
        namelist = EMSphInxNamelist.from_string(str(ref["namelist"]))
        pc = namelist.to_detector(sample_tilt=SAMPLE_TILT).pc.ravel()
        assert np.array_equal(pc, ref["pc"])


# ============ Ours against theirs, the small map (D5, D6) =========== #


class TestOursVsTheirsSmall:
    @pytest.mark.parametrize("name", sorted(SCENARIO_TABLE))
    def test_scenario_kwargs_derive_from_provenance(self, name):
        # the structural killer of a harness which hard codes its
        # arguments: the refine flag has no band signature
        assert scenario_kwargs(load_reference(name)) == KWARGS_TABLE[name]

    def test_sample_tilt_binding(self, ni_harmonics):
        # the Phase 6 binding guard's regression suite face: harmonics
        # for a 70 degree tilt indexed against a detector set to 65
        # land 4.68 degrees off at *higher* scores, so no score based
        # check could catch a mismatch
        ref = load_reference(ANCHOR)
        detector = detector_at(ref["pc"])
        assert detector.sample_tilt == SAMPLE_TILT
        assert ni_harmonics.sample_tilt == SAMPLE_TILT
        assert float(ref["sample_tilt"]) == SAMPLE_TILT

    @pytest.mark.parametrize("name", SMALL_SCENARIOS)
    def test_orientations_agree(self, name, ni_harmonics, record_property):
        ref = load_reference(name)
        ours = index_small(name, ni_harmonics)
        angles = misorientation(ours.rotations, ref)
        record_comparison(record_property, name, angles, ours, ref)
        record_property(
            f"{name}_per_point", ", ".join(f"{angle:.3f}" for angle in angles)
        )
        assert angles.size == 9
        if name == "small_coarse_nr10":
            assert np.median(angles) < COARSE_MEDIAN_DEG
            # the outlier convention: one correlation grid cell is
            # 2.667 degrees at this bandwidth, so a lone neighbouring
            # cell argmax is a legitimate near tie rather than a
            # regression
            assert int((angles < COARSE_TIGHT_MAX_DEG).sum()) >= COARSE_TIGHT_COUNT
            assert angles.max() < COARSE_CEILING_DEG
        else:
            assert np.median(angles) < REFINED_MEDIAN_DEG
            assert angles.max() < REFINED_MAX_DEG

    @pytest.mark.parametrize("name", SMALL_SCENARIOS)
    def test_scores_correlate(self, name, ni_harmonics):
        assert_scores(
            index_small(name, ni_harmonics), load_reference(name), SCORE_R_SMALL
        )

    @pytest.mark.parametrize("name", SMALL_SCENARIOS)
    def test_iq_is_float32_equal(self, name, ni_harmonics):
        assert_iq(index_small(name, ni_harmonics), load_reference(name))

    def test_stretch_emulation_collapses_the_residual(
        self, ni_harmonics, record_property
    ):
        # the evidence pin of the mission's re-anchored criterion 2:
        # emulating EMSphInx' deliberately un-ported ``bilinearCoeff``
        # stretch in pattern centre space collapses the disagreement
        # to its interpolation and Newton floor, which is under the
        # original 0.2 degree gate.  kikuchipy keeps its own
        # convention: the same emulation *worsens* agreement with the
        # stored crystal map (0.365 to 0.584 degrees on the large map)
        ref = load_reference(ANCHOR)
        signal = small_signal()
        shape = signal.axes_manager.signal_shape[::-1]
        baseline = misorientation(index_small(ANCHOR, ni_harmonics).rotations, ref)
        emulated = misorientation(
            index_small(
                ANCHOR, ni_harmonics, pc=stretched_pc(ref["pc"], shape)
            ).rotations,
            ref,
        )
        record_property("stretch_baseline_median", f"{np.median(baseline):.4f}")
        record_property("stretch_emulated_median", f"{np.median(emulated):.4f}")
        record_property("stretch_emulated_max", f"{emulated.max():.4f}")
        assert np.median(emulated) < STRETCH_MEDIAN_DEG
        assert np.median(emulated) < np.median(baseline)


# ============ Ours against theirs, the large map (D5, D6) =========== #


class TestOursVsTheirsLarge:
    @staticmethod
    def _compare(name, step, harmonics, record_property):
        """Return the misorientations of one large subset scenario,
        its crystal map, its reference, the subset signal and the full
        map's detector, and record every statistic.

        The signal comes back for the image quality message's pattern
        md5 and the detector for the route pins, both of which would
        otherwise rebuild the cached subset for nothing.
        """
        ref = load_reference(name)
        signal, full_detector = large_signal(step)
        detector = detector_at(ref["pc"], full_detector)
        # the same structural pin ``index_small`` carries: the
        # arguments come from the provenance and must equal the frozen
        # table, since the refine flag has no band signature
        kwargs = scenario_kwargs(ref)
        assert kwargs == KWARGS_TABLE[name]
        ours = signal.spherical_indexing(harmonics, detector, verbose=0, **kwargs)
        angles = misorientation(ours.rotations, ref)
        record_comparison(record_property, name, angles, ours, ref)
        return angles, ours, ref, signal, full_detector

    @staticmethod
    def _assert_route_pins(name, detector):
        """Assert the name list and pattern centre pins of one large
        scenario.

        They recompute here rather than in ``TestRoutePins`` since
        they need the downloaded map's detector, and inside each
        scenario's own test rather than in one shared test, so that
        the default suite does not pay the 165 point map's background
        removal (measured 0.70 s) for a pin whose parity test is
        weekly.
        """
        ref = load_reference(name)
        assert frozen_namelist(name, detector).to_string() == str(ref["namelist"])
        namelist = EMSphInxNamelist.from_string(str(ref["namelist"]))
        pc = namelist.to_detector(sample_tilt=SAMPLE_TILT).pc.ravel()
        assert np.array_equal(pc, ref["pc"])

    def test_the_twenty_point_scenario(self, ni_harmonics, record_property):
        name = "large20_refined_nr10"
        angles, ours, ref, signal, detector = self._compare(
            name, 15, ni_harmonics, record_property
        )
        assert angles.size == 20
        assert np.median(angles) < REFINED_MEDIAN_DEG
        assert angles.max() < LARGE20_MAX_DEG
        assert_scores(ours, ref, SCORE_R_LARGE20)
        assert_iq(ours, ref, signal)
        self._assert_route_pins(name, detector)

    @pytest.mark.weekly
    def test_the_165_point_scenario(self, ni_harmonics, record_property):
        name = "large165_refined_nr10"
        angles, ours, ref, signal, detector = self._compare(
            name, 5, ni_harmonics, record_property
        )
        assert angles.size == 165
        assert np.median(angles) < REFINED_MEDIAN_DEG
        assert np.percentile(angles, 95) < LARGE165_P95_DEG
        assert angles.max() < LARGE165_MAX_DEG
        assert_scores(ours, ref, SCORE_R_LARGE165)
        assert_iq(ours, ref, signal)
        self._assert_route_pins(name, detector)


# ========= Regeneration, the EMSphInx binaries (D8, local) ========== #


class TestRegenerateReferences:
    """The one gated surface Phase 10 adds.

    Rerunning the generation script must reproduce the shipped files
    **bitwise**, which is simultaneously the strongest theirs-on-ours
    acceptance -- the program still accepts and identically indexes
    our files -- and the guard that the shipped bytes match the pinned
    commit.  Phase 9's acid test and its negative controls own the
    rest of that surface.
    """

    def test_the_mismatch_message_names_the_wisdom_and_the_arrays(self, tmp_path):
        # this one needs no binary: the diagnostic the two gated tests
        # print is exercised on CI against a stand-in which is not the
        # shipped file, so a real mismatch cannot fail with a broken
        # message.  The stand-in drops one array, changes one value
        # and widens one data type, which walks every branch of the
        # per-array difference
        shipped = load_reference("small_refined_nr0")
        arrays = {key: shipped[key] for key in shipped if key != "phase"}
        arrays["nregions"] = np.int64(4)
        arrays["iq"] = shipped["iq"].astype(np.float64)
        fpath = tmp_path / "regression_small_refined_nr0.npz"
        np.savez(fpath, **arrays)

        with pytest.raises(AssertionError) as error:
            assert_regenerated_bitwise({"small_refined_nr0": fpath})
        message = str(error.value)
        assert "fftw.wisdom" in message
        assert "the contract is bitwise" in message
        assert "nregions: differs" in message
        assert "iq: float64(9,) vs float32(9,)" in message
        assert "phase: only in one of the two files" in message
        # the arrays which agree are not listed
        assert "emsphinx_commit" not in message

    def test_regenerated_references_are_bitwise(
        self, emsphinx_program, tmp_path, record_property
    ):
        # the fixture holds the machine wide program lock for the
        # whole test and its callable carries the "not built" skip, so
        # the program must be resolved through it.  ``main`` does not
        # take that lock again when it is handed a program: doing so
        # from this process would dead lock against the fixture
        program = emsphinx_program("IndexEBSD")
        from kikuchipy.data.emsphinx import create_emsphinx_reference

        start = time.monotonic()
        written = create_emsphinx_reference.main(
            tmp_path, program=program, scenarios=SMALL_SCENARIOS
        )
        record_property("regeneration_seconds", f"{time.monotonic() - start:.2f}")
        assert sorted(written) == sorted(SMALL_SCENARIOS)
        assert_regenerated_bitwise(written)

    @pytest.mark.weekly
    def test_regenerated_references_are_bitwise_all(
        self, emsphinx_program, tmp_path, record_property
    ):
        # the full sweep, which adds pooch, the full map background
        # removal and about two seconds of indexing
        pytest.importorskip("pooch")
        program = emsphinx_program("IndexEBSD")
        from kikuchipy.data.emsphinx import create_emsphinx_reference

        start = time.monotonic()
        written = create_emsphinx_reference.main(tmp_path, program=program)
        record_property("regeneration_all_seconds", f"{time.monotonic() - start:.2f}")
        assert sorted(written) == sorted(SCENARIO_TABLE)
        assert_regenerated_bitwise(written)
