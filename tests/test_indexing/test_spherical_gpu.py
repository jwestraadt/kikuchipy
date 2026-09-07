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

"""Tests of the optional CuPy GPU backend of spherical indexing
(``kikuchipy.indexing._spherical._gpu`` and the ``backend`` plumbing
of ``SphericalIndexer``/``EBSD.spherical_indexing``).

Covers ``specs/2026-09-07-spherical-gpu/validation.md``:

- **Default suite** (no GPU, runs on CI): the backend switch and the
  CPU-default bitwise guard [D1/D5.1], the three-stage availability
  gate with monkeypatched failures [D6], the xp-agnostic pipeline
  core under NumPy -- the CI-side correctness anchor [D2/D3/D11.1],
  the neighborhood-offsets and peak-epilogue parity units [D2], the
  padding/exclusion batch assembly [D7.3-4], the VRAM model and the
  ``gpu_memory_per_batch_bytes`` helper [D8], and the gating-fixture
  decision logic (kill switch, structural xdist skip) [D10].
- **Locally gated suite** (``cupy_gpu`` fixture): the parity oracle
  against the CPU backend (IQ bitwise, refined-to-refined bands,
  score scale) [D4], determinism per backend [D5], pseudo-symmetry
  and multi-phase dual-backend runs [D9/D2], OOM handling and
  device-error propagation [D8.4/D7.7], and the D12 throughput
  floor.

**Gating conventions** (D10): the ``cupy_gpu`` fixture is an
availability probe (three-stage gate), NOT an env-var opt-in; it
skips unconditionally when ``KIKUCHIPY_NO_GPU_TESTS`` is set (the
kill switch) and whenever ``PYTEST_XDIST_WORKER`` is set (the
structural ``-n 0`` rule), so the default ``-n 4`` suite command
stays GPU-silent even on cupy-capable machines.  The wgpu ``gpu``
marker of ``conftest.py`` is deliberately NOT reused.  **GitHub CI
green is zero evidence for the GPU path**: every gated check is a
local definition-of-done gate.

Values marked ``FIXME-pin`` are MEASURED-THEN-PINNED placeholders
(requirements D3/D4/D5/D8/D12): they are replaced by dated measured
values at the implementation gate and recorded in validation.md
"Recorded results".  Every GPU pin is machine-specific (NVIDIA RTX
2000 Ada Generation Laptop GPU, 8 GB, cc 8.9, driver 595.71, CuPy
14.2.0, Windows 11) and says so where it is pinned.
"""

import functools
import inspect
import os
import re
import subprocess
import sys
import types
import warnings

import dask
import dask.array as da
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh
import pytest

import kikuchipy as kp
from kikuchipy.indexing._spherical import _gpu, _indexer
from kikuchipy.indexing._spherical._euler import rotation_from_zyz
from kikuchipy.indexing._spherical._indexer import SphericalIndexer
from kikuchipy.indexing._spherical._master_pattern_harmonics import (
    MasterPatternHarmonics,
)
from kikuchipy.indexing._spherical._preprocessing import _preprocess_pattern
from kikuchipy.indexing._spherical._wigner import rotate_harmonics
from kikuchipy.indexing._spherical._xcorr import (
    SphericalCrossCorrelator,
    _extract_neighborhood,
    _find_peak,
    _scale_and_find_peak,
    euler_to_index,
    index_to_euler,
)

# ------------------------- Frozen constants ------------------------- #

# The bandwidth of the real-data parity tests, the project's default
NI_BANDWIDTH = 68

# The two synthetic oracle bandwidths of the numpy-xp pipeline tests:
# bw 16 -> slP 32 (EVEN, bwP 17 > bw -- the even-slP guard case of
# D2: quadrant rows n in [bw, bwP) and the m = bw column) and
# bw 20 -> slP 39 (odd, bwP 20 == bw)
XP_BANDWIDTH_EVEN = 16
XP_BANDWIDTH_ODD = 20

# The real-data numpy-xp case (validation: "one real Ni bw-32 case");
# fast_size(63) = 63, odd slP
XP_BANDWIDTH_NI = 32

# ``nickel_ebsd_large`` subset steps, the Phase 6/10 conventions:
# ``inav[::15, ::15]`` = 20 points, ``inav[::5, ::5]`` = 165 points
LARGE_STEP_20PT = 15
LARGE_STEP_165PT = 5

# ------------- MEASURED-THEN-PINNED placeholders (MTP) -------------- #
#
# FIXME-pin: every value below is a drafting-stage placeholder from
# the a-priori reasoning of requirements D3/D4/D5 and is replaced by
# a dated measured value (with the ~2x margin convention) at the
# implementation gate; the measurement is recorded in validation.md
# "Recorded results".  GPU pins are machine-specific (RTX 2000 Ada
# 8 GB, driver 595.71, CuPy 14.2.0).

# FIXME-pin [D11.1/D3]: relative band of the numpy-xp float32 cube
# against the float64 CPU cube, as atol = scale * max|cube|
XP_CUBE_ATOL_SCALE = 1e-5

# FIXME-pin [D4.2]: exact winner flip COUNTS on the small sets (a
# fraction is unmeasurable at N = 9/20, review-corrected); expected 0
REFINED_FLIP_COUNT_SMALL = 0
REFINED_FLIP_COUNT_20PT = 0

# FIXME-pin [D4.2, weekly]: the flip-RATE pin lives on the 165-pt run
REFINED_AGREEMENT_RATE_165PT = 0.99

# FIXME-pin [D4.4]: refined-to-refined misorientation bands, degrees
# (expected median << 0.05; a-priori ceiling half a grid cell,
# 1.33 deg at bw 68; sanity anchor: far inside the 0.31-0.34 deg
# CPU-vs-EMSphInx band)
REFINED_MISO_MEDIAN_DEG = 0.05
REFINED_MISO_MAX_DEG = 1.34

# FIXME-pin [D4.5]: score bands (same scale by construction -- the
# anti-EMSphInx-defect requirement; their CPU/CUDA scales diverged
# ~7x)
SCORE_PEARSON_MIN = 0.9999
REFINED_SCORE_REL_DIFF = 1e-4
COARSE_SCORE_REL_DIFF = 1e-5

# FIXME-pin [D4.3]: coarse argmax-cell agreement (recorded, not the
# gate -- D4.4 is)
COARSE_CELL_AGREEMENT_MIN = 0.9

# FIXME-pin [D9]: pseudo-symmetry winner-index flip count on the
# 9-pattern set with a true 90-deg-z operator (a true op guarantees
# near-ties, so the measured count may be pinned above zero)
PSYM_INDEX_FLIP_COUNT = 0

# FIXME-pin [D8.2]: g(bw) calibration bands, bytes.  The bw-68 band
# brackets the ~50 MB drafting anchor (measured <= 1.4 GB pool
# high-water at B=32 -> real working set ~40-44 MB/pattern); the
# bw-88 band brackets the ~110 MB scaling ESTIMATE (no separable
# pipeline was measured at bw 88 while drafting)
VRAM_G68_BOUNDS = (35_000_000, 65_000_000)
VRAM_G88_BOUNDS = (70_000_000, 150_000_000)

# FIXME-pin [D8.3]: per-phase resident band at bw 68 (~11 MB
# drafting: A/A2 c64 + table f32 + r_den f32)
VRAM_RESIDENT68_BOUNDS = (6_000_000, 20_000_000)

# MEASURED 2026-09-07 (failing-tests gate, review-corrected)
# [D12.1]: the idle-machine 8-worker CPU baseline on the
# ``nickel_ebsd_large`` route at bw 68 (m-3m, refine=True, default
# settings, chunksize model 15/chunk), best of 3 full-map runs after
# a 256-pattern warm-up.  The first recorded window (183.8 / 154.2 /
# 155.5 pat/s, warm-up 2.39 s) FAILED an idle-machine consistency
# re-run and was ~22 % too soft (not thermally/load clean); two
# independent re-runs of the identical recipe on the now-dedicated
# idle machine measured 236.0 / 231.6 / 228.3 (warm-up 1.35 s,
# test-critic session) and 221.9 / 223.8 / 209.1 (warm-up 1.48 s,
# fix-application session), each with the identical score band
# 0.2289-0.6903 and 0 failed rows of 4125.  The floor pins the
# HIGHEST honest idle measurement (the hardest floor); the pin is
# re-confirmed in the same session/thermal state as the GPU
# measurement at the implementation gate (D12 convention).
# Machine-specific: the RTX 2000 Ada machine's 20-core laptop CPU,
# Windows 11, python 3.13.12, dedicated idle window (open question
# 9.5 answered).  Recipe + command recorded in validation.md
# "Recorded results".  This is the LEFT side of the D12 floor: the
# GPU run must reach at least this.
CPU_BASELINE_8_WORKERS_PAT_S = 236.0


# --------------------- The gated-suite fixture ---------------------- #


def _cupy_gpu_skip_reason() -> "str | None":
    """Return why the gated GPU suite must skip, or ``None`` to run.

    The decision order is frozen (D10.3-4) and the first two checks
    come BEFORE the availability probe, so neither the kill switch
    nor an xdist worker ever initialises CUDA:

    1. ``KIKUCHIPY_NO_GPU_TESTS`` set (to any non-empty value): the
       manual kill switch, e.g. for a concurrent job owning the GPU.
    2. ``PYTEST_XDIST_WORKER`` set: the structural ``-n 0`` rule --
       under any ``-n N`` run the gated suite skips by construction,
       so the default ``-n 4`` command stays GPU-silent everywhere.
    3. The three-stage availability gate of ``_gpu`` (import, device,
       cuFFT): its actionable message becomes the skip reason.
    """
    if os.environ.get("KIKUCHIPY_NO_GPU_TESTS"):
        return (
            "KIKUCHIPY_NO_GPU_TESTS is set, the manual GPU-test kill "
            "switch; unset it to run the locally gated GPU tests"
        )
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        return (
            "GPU tests run only at -n 0 (PYTEST_XDIST_WORKER is set, "
            "the structural xdist skip); run 'uv run pytest "
            "tests/test_indexing/test_spherical_gpu.py -n 0' to "
            "exercise them"
        )
    try:
        _gpu._verify_gpu_or_raise()
    except NotImplementedError:
        return (
            "the three-stage CuPy gate of _gpu.py is not implemented "
            "yet (failing-tests stage of specs/2026-09-07-spherical-"
            "gpu); the gated suite arms at the implementation gate"
        )
    except Exception as error:
        # The gate's own frozen message is the instruction-bearing
        # skip reason, one per failing stage (D10.2)
        return str(error)
    return None


@pytest.fixture
def cupy_gpu():
    """Yield the imported :mod:`cupy` when the GPU backend is usable
    here, else skip with an instruction-bearing reason (D10).

    An availability probe, not an env-var opt-in: unlike the
    ``emsphinx_dir`` convention there is nothing to point at.
    """
    reason = _cupy_gpu_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    import cupy

    return cupy


# ----------------------------- Helpers ------------------------------ #


@functools.lru_cache(maxsize=1)
def ni_patterns():
    """Return the nine background corrected ``nickel_ebsd_small``
    patterns as a read-only ``(9, 60, 60)`` unsigned 8-bit array."""
    signal = kp.data.nickel_ebsd_small()
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    data = signal.data.reshape((-1, 60, 60))
    data.flags.writeable = False
    return data


def ni_detector():
    """Return a fresh Ni detector with one projection centre."""
    detector = kp.data.nickel_ebsd_small().detector.deepcopy()
    detector.pc = detector.pc_average
    return detector


@functools.lru_cache(maxsize=4)
def ni_harmonics(bandwidth):
    """Return the Ni master harmonics built directly at
    ``bandwidth``, cached (the transform costs about a second)."""
    master = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    return MasterPatternHarmonics.from_master_pattern(master, bandwidth=bandwidth)


def ni_indexer(**kwargs):
    """Return an indexer of the Ni harmonics and detector at ``bw``
    68 with the given overrides.  Unlike the sibling test modules,
    ``refine`` keeps the class default: the parity oracle of this
    suite is refined-to-refined (D4.4)."""
    bandwidth = kwargs.pop("bandwidth", NI_BANDWIDTH)
    return SphericalIndexer(ni_harmonics(bandwidth), ni_detector(), **kwargs)


def scrambled_harmonics(point_group, seed=42):
    """Return a sign-scrambled copy of the Ni master declared under
    ``point_group``.

    The scramble keeps every coefficient magnitude, the ``m = 0`` row
    real and the zero structure, so the coefficients stay those of a
    real spherical function and still satisfy the m-3m symmetry
    validation; re-declaring the copy under ``"1"`` (n_fold 1, no
    mirror) is the mixed-symmetry case that kills stale ``fxc``
    columns across phases (D2 freshness rule) -- the same-symmetry
    scramble shares zero structure and cannot.
    """
    harmonics = ni_harmonics(NI_BANDWIDTH)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], harmonics.alm.shape)
    return MasterPatternHarmonics(
        harmonics.alm * signs, phase=Phase("scrambled", point_group=point_group)
    )


@functools.lru_cache(maxsize=2)
def large_signal(step):
    """Return a background corrected ``nickel_ebsd_large`` subset and
    the full map's ``pc_average`` detector (the Phase 6/10 route)."""
    pytest.importorskip("pooch")
    signal = kp.data.nickel_ebsd_large(allow_download=True)
    signal.remove_static_background(show_progressbar=False)
    signal.remove_dynamic_background(show_progressbar=False)
    detector = signal.detector.deepcopy()
    detector.pc = detector.pc_average
    return signal.inav[::step, ::step], detector


def large_patterns(step):
    """Return the subset's patterns as a ``(n, 60, 60)`` array."""
    signal, _ = large_signal(step)
    return signal.data.reshape((-1, 60, 60))


def large_indexer(**kwargs):
    """Return an indexer on the large map's detector at ``bw`` 68."""
    _, detector = large_signal(kwargs.pop("step"))
    return SphericalIndexer(ni_harmonics(NI_BANDWIDTH), detector, **kwargs)


def random_spectrum(bandwidth, n_fold=1, mirror=False, seed=0):
    """Return a random ``(bw, bw)`` harmonic spectrum ``a[m, l]`` of
    a real function carrying the requested symmetry: upper triangle
    ``l >= m`` only, real ``m = 0`` row, rows ``m % n_fold != 0``
    zero, and ``l + m`` odd entries zero under ``mirror``."""
    rng = np.random.default_rng(seed)
    alm = rng.normal(size=(bandwidth, bandwidth)) + 1j * rng.normal(
        size=(bandwidth, bandwidth)
    )
    alm = np.triu(alm)
    alm[0] = alm[0].real
    if n_fold > 1:
        alm[(np.arange(bandwidth) % n_fold) != 0, :] = 0
    if mirror:
        m = np.arange(bandwidth)[:, None]
        ell = np.arange(bandwidth)[None, :]
        alm[(m + ell) % 2 == 1] = 0
    return np.ascontiguousarray(alm, dtype=np.complex128)


def gln_batch_for(flm, side_length, cells, noise_seed=7):
    """Return a ``(B, bw, bw)`` batch of pattern spectra: rotations
    of ``flm`` onto the given grid ``cells`` plus a pinch of noise,
    so every cube has one sharp, unambiguous peak."""
    rng = np.random.default_rng(noise_seed)
    batch = []
    for k, n, m in cells:
        zyz = index_to_euler((k, n, m), side_length)
        gln = rotate_harmonics(flm, zyz)
        noise = random_spectrum(flm.shape[0], seed=int(rng.integers(1 << 30)))
        batch.append(gln + 1e-6 * noise)
    return np.ascontiguousarray(np.stack(batch), dtype=np.complex128)


def run_xp_pipeline(flm, gln_batch, n_fold, mirror, r_den=None, table=None):
    """Run device stages 4-6 of the xp-agnostic core under NumPy and
    return ``(xc_batch, indices, values)`` -- the D11.1 seam every
    numpy-xp oracle test drives.
    """
    bandwidth = int(flm.shape[0])
    correlator = SphericalCrossCorrelator(bandwidth)
    slp = correlator.side_length
    bwp = correlator.half_side_length
    if table is None:
        table = _gpu._sanitized_table(correlator.wigner_d_half_pi)
    a, a2 = _gpu._build_a_tables(np, flm, table)
    g, g2 = _gpu._build_g_batch(np, gln_batch, table, bandwidth)
    batch_size = int(gln_batch.shape[0])
    # Deliberately DIRTY (NaN-seeded), never pre-zeroed: the frozen D2
    # freshness rule says ``_spectrum_batch`` re-zeroes (or fully
    # writes across all m columns) the batch buffer on every per-phase
    # call, so a correct implementation must produce identical cubes
    # from a garbage buffer -- and the stale-columns-across-phases
    # mutant (an implementation leaning on the caller's zeroing) dies
    # here on CI at every parametrisation instead of only in the gated
    # ``test_multiphase_mixed_symmetry_dual_backend`` run
    # (review-strengthened)
    fxc = np.full(
        (batch_size, slp, slp, bwp), complex(np.nan, np.nan), dtype=np.complex64
    )
    _gpu._spectrum_batch(np, g, g2, a, a2, fxc, n_fold, mirror)
    xc = _gpu._inverse_fft_batch(np, np.fft, fxc, n_fold)
    indices, values = _gpu._scale_argmax_batch(np, xc, r_den)
    return xc, indices, values


def cpu_reference_cubes(flm, gln_batch, n_fold, mirror, r_den=None):
    """Return the CPU oracle's ``(cubes, indices)`` for a batch: one
    float64 ``compute`` (+ optional in-place ``r_den`` scale) and
    peak per pattern, through the frozen ``_xcorr`` path."""
    correlator = SphericalCrossCorrelator(int(flm.shape[0]))
    cubes, indices = [], []
    for gln in gln_batch:
        xc = correlator.compute(flm, gln, n_fold, mirror).copy()
        if r_den is not None:
            index = _scale_and_find_peak(xc, r_den)
        else:
            index = _find_peak(xc)
        cubes.append(xc)
        indices.append(int(index))
    return cubes, indices


def misorientation_deg(zyz_a, zyz_b):
    """Return the m-3m reduced misorientation angles in degrees
    between two ``(n, 3)`` ZYZ stacks (the Phase 10 vocabulary)."""
    a = Orientation(rotation_from_zyz(np.asarray(zyz_a).reshape(-1, 3)).data, Oh)
    b = Orientation(rotation_from_zyz(np.asarray(zyz_b).reshape(-1, 3)).data, Oh)
    return np.asarray(a.angle_with(b, degrees=True), dtype=np.float64).ravel()


def pearson(x, y):
    """Return the Pearson correlation of two flat arrays."""
    return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])


def rel_diff(x, y):
    """Return the elementwise relative |difference| of two arrays."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    scale = np.maximum(np.abs(x), np.abs(y))
    scale[scale == 0] = 1.0
    return np.abs(x - y) / scale


def dual_backend_results(patterns, cupy_gpu=None, indexer_kwargs=None, **kwargs):
    """Return ``(cpu, gpu)`` result dictionaries of the same
    ``index_patterns`` call under both backends."""
    indexer_kwargs = dict(indexer_kwargs or {})
    step = indexer_kwargs.pop("step", None)
    results = {}
    for backend in ("cpu", "gpu"):
        if step is None:
            indexer = ni_indexer(backend=backend, **indexer_kwargs)
        else:
            indexer = large_indexer(step=step, backend=backend, **indexer_kwargs)
        results[backend] = indexer.index_patterns(patterns, progressbar=False, **kwargs)
    return results["cpu"], results["gpu"]


def assert_refined_parity(
    cpu, gpu, flip_count, miso_median_deg, miso_max_deg, score_rel_diff
):
    """The shared D4 refined-to-refined parity block: IQ bitwise,
    winner flip count, misorientation and score bands (all
    FIXME-pin placeholders until the implementation gate)."""
    # D4.1: IQ bitwise -- the free wiring probe
    assert np.array_equal(cpu["iq"], gpu["iq"])
    # D4.2: winner agreement as an exact COUNT at small N.  The
    # phase_id comparison alone is VACUOUS in the single-phase
    # no-psym callers (phase_id is constant 0 on both backends), so
    # the count is also measured on the coarse WINNER CELL: the
    # refined zyz rounded back to its grid cell (the
    # ``test_coarse_cell_agreement`` vocabulary) -- a GPU argmax
    # landing on a different near-tied cell counts as a flip even
    # when Newton reconverges inside the miso band
    # (review-strengthened).  Both counts share the FIXME-pin; the
    # implementation gate splits the constant if they measure apart
    flips = int((cpu["phase_id"][:, 0] != gpu["phase_id"][:, 0]).sum())
    assert flips == flip_count
    side = SphericalCrossCorrelator(NI_BANDWIDTH).side_length
    cells_cpu = [euler_to_index(zyz, side) for zyz in cpu["zyz"][:, 0]]
    cells_gpu = [euler_to_index(zyz, side) for zyz in gpu["zyz"][:, 0]]
    cell_flips = int(sum(a != b for a, b in zip(cells_cpu, cells_gpu)))
    assert cell_flips == flip_count
    # D4.4: refined-to-refined misorientation bands
    angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
    assert float(np.median(angles)) <= miso_median_deg
    assert float(angles.max()) <= miso_max_deg
    # D4.5: same score scale
    assert pearson(cpu["scores"], gpu["scores"]) >= SCORE_PEARSON_MIN
    assert float(rel_diff(cpu["scores"], gpu["scores"]).max()) <= score_rel_diff


def make_fake_cupy(version="14.2.0", device_count=1, device_error=None, fft_error=None):
    """Return a fake :mod:`cupy` stand-in for the gate seams: enough
    ``cuda.runtime``/``fft``/array surface for the three probes, with
    plantable failures."""

    class FakeCudaRuntimeError(Exception):
        pass

    def get_device_count():
        if device_error is not None:
            raise FakeCudaRuntimeError(device_error)
        return device_count

    class _FftNamespace:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                if fft_error is not None:
                    raise fft_error
                return np.asarray(args[0]) if args else np.zeros(1)

            return call

    fake = types.SimpleNamespace()
    fake.__version__ = version
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            getDeviceCount=get_device_count,
            CUDARuntimeError=FakeCudaRuntimeError,
        )
    )
    fake.fft = _FftNamespace()
    for name in ("asarray", "array", "ascontiguousarray"):
        setattr(fake, name, np.asarray)
    for name in ("ones", "zeros", "arange", "empty"):
        setattr(fake, name, getattr(np, name))
    return fake


@pytest.fixture
def fresh_gate(monkeypatch):
    """Reset the gate's cached verdict for one test."""
    monkeypatch.setattr(_gpu, "_gate_result", None)


# =============== Backend switch and CPU-default guard =============== #


class TestBackendSwitch:
    """The explicit per-call backend keyword and the bitwise
    protection of the CPU reference path.  [D1, D5.1]"""

    @pytest.mark.parametrize("bad", ["tpu", "GPU", "auto", "cuda", ""])
    def test_backend_validation_raises(self, bad):
        # the SUPPORTED_OPTIMIZATION_METHODS message shape, naming
        # the accepted set.  [D1]
        with pytest.raises(ValueError, match="not in the list of supported") as info:
            ni_indexer(backend=bad)
        message = str(info.value)
        assert "'cpu'" in message
        assert "'gpu'" in message
        assert repr(bad) in message

    def test_backend_validation_raises_via_signal_method(self):
        # the keyword threads through ``EBSD.spherical_indexing``
        # into the ctor guard.  [D1]
        signal = kp.data.nickel_ebsd_small()
        with pytest.raises(ValueError, match="not in the list of supported"):
            signal.spherical_indexing(
                ni_harmonics(NI_BANDWIDTH),
                ni_detector(),
                backend="tpu",
                verbose=0,
            )

    def test_backend_attribute_is_the_string_only(self):
        # no device state ever lives on the indexer (D7.2): the
        # attribute is the plain string
        indexer = ni_indexer(refine=False)
        assert indexer.backend == "cpu"

    def test_gate_fires_at_ctor_for_gpu_only(self, monkeypatch):
        # fail fast at construction (D1.3), and a "cpu" construction
        # never touches the gate (so never cupy)
        sentinel = ImportError("gate sentinel")

        def raising_gate():
            raise sentinel

        monkeypatch.setattr(_indexer, "_verify_gpu_or_raise", raising_gate)
        with pytest.raises(ImportError, match="gate sentinel"):
            ni_indexer(backend="gpu")

        def forbidden_gate():
            raise AssertionError("the gate must not fire for backend='cpu'")

        monkeypatch.setattr(_indexer, "_verify_gpu_or_raise", forbidden_gate)
        indexer = ni_indexer(backend="cpu", refine=False)
        assert indexer.backend == "cpu"

    def test_backend_cpu_is_bitwise_default(self):
        # the D1/D5.1 guard: shared-code edits must never perturb
        # the reference path -- ``backend="cpu"`` is bitwise equal
        # to a call without the keyword on ``nickel_ebsd_small``
        explicit = ni_indexer(refine=False, backend="cpu")
        implicit = ni_indexer(refine=False)
        results_explicit = explicit.index_patterns(ni_patterns(), progressbar=False)
        results_implicit = implicit.index_patterns(ni_patterns(), progressbar=False)
        assert set(results_explicit) == set(results_implicit)
        for key, value in results_implicit.items():
            assert np.array_equal(results_explicit[key], value), key

    def test_indexer_signature_placement(self):
        # keyword-only, right after ``pseudo_symmetry_ops`` in the
        # existing keyword-only group (D1)
        parameters = inspect.signature(SphericalIndexer.__init__).parameters
        names = list(parameters)
        position = names.index("backend")
        assert names[position - 1] == "pseudo_symmetry_ops"
        assert parameters["backend"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["backend"].default == "cpu"

    def test_signal_method_signature_placement(self):
        # plain positional-or-keyword append after
        # ``pseudo_symmetry_ops``, before ``chunksize`` (the method
        # has no bare ``*``; review-corrected D1, the Phase 8
        # placement precedent)
        parameters = inspect.signature(kp.signals.EBSD.spherical_indexing).parameters
        names = list(parameters)
        position = names.index("backend")
        assert names[position - 1] == "pseudo_symmetry_ops"
        assert names[position + 1] == "chunksize"
        assert parameters["backend"].default == "cpu"


# ---------------------- Availability gate (D6) ---------------------- #


class TestAvailabilityGate:
    """The three-stage gate, its frozen actionable messages, the
    version floor, the Windows DLL shim and the module-scope import
    ban -- all with cupy faked or absent.  [D6]"""

    def test_import_stage_message(self, monkeypatch, fresh_gate):
        # stage (a): cupy not importable -> ImportError naming the
        # wheel family remedies (frozen wording, D6.2a)
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.raises(ImportError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert "backend='gpu'" in message
        assert "cupy" in message
        assert "pip install cupy-cuda12x" in message
        assert "cupy-cuda11x" in message

    def test_gate_version_floor(self, monkeypatch, fresh_gate):
        # stage (a) version floor: cupy < 13 -> actionable
        # ImportError naming the found version (D6.2)
        monkeypatch.setitem(sys.modules, "cupy", make_fake_cupy(version="12.6.0"))
        with pytest.raises(ImportError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert "cupy >= 13" in message
        assert "12.6.0" in message

    def test_device_stage_message(self, monkeypatch, fresh_gate):
        # stage (b): no device -> the message names drivers.  Both
        # failure shapes: a zero count and a CUDARuntimeError
        for fake in (
            make_fake_cupy(device_count=0),
            make_fake_cupy(device_error="no CUDA-capable device"),
        ):
            monkeypatch.setattr(_gpu, "_gate_result", None)
            monkeypatch.setitem(sys.modules, "cupy", fake)
            with pytest.raises(RuntimeError) as info:
                _gpu._verify_gpu_or_raise()
            message = str(info.value)
            assert "CUDA device" in message
            assert "driver" in message

    def test_cufft_stage_message(self, monkeypatch, fresh_gate):
        # stage (c): import + device probe succeed while the first
        # FFT dies (the probe-measured Windows failure mode) -> the
        # message names BOTH remedies: the CUDA Toolkit and the
        # nvidia wheels
        fake = make_fake_cupy(
            fft_error=ImportError("DLL load failed while importing cufft")
        )
        monkeypatch.setitem(sys.modules, "cupy", fake)
        with pytest.raises(RuntimeError) as info:
            _gpu._verify_gpu_or_raise()
        message = str(info.value)
        assert "cuFFT" in message
        assert "CUDA Toolkit" in message
        assert "nvidia-cufft-cu12" in message

    def test_gate_stage_order_and_caching(self, monkeypatch, fresh_gate):
        # the frozen order -- import, device, shim, probe (the
        # swapped-order mutant of plan 7.2) -- and the D6.1 caching:
        # a second call re-probes nothing
        calls = []
        fake = make_fake_cupy()

        monkeypatch.setattr(
            _gpu, "_import_cupy", lambda: calls.append("import") or fake
        )
        monkeypatch.setattr(
            _gpu, "_device_count", lambda cupy: calls.append("device") or 1
        )
        monkeypatch.setattr(
            _gpu,
            "_add_nvidia_dll_directories",
            lambda: calls.append("shim"),
        )
        monkeypatch.setattr(_gpu, "_probe_cufft", lambda cupy: calls.append("probe"))
        _gpu._verify_gpu_or_raise()
        assert calls == ["import", "device", "shim", "probe"]
        _gpu._verify_gpu_or_raise()
        assert calls == ["import", "device", "shim", "probe"]

    def test_gate_failure_is_cached(self, monkeypatch, fresh_gate):
        # the failing half of the D6.1 cache (review-added: a
        # success-only cache would pass the passing-verdict test
        # above): the first failing evaluation's exception is cached
        # and re-raised on the second call with NO re-probe
        calls = []

        def failing_import():
            calls.append("import")
            raise ImportError("cached-failure sentinel")

        monkeypatch.setattr(_gpu, "_import_cupy", failing_import)
        with pytest.raises(ImportError, match="cached-failure sentinel"):
            _gpu._verify_gpu_or_raise()
        assert calls == ["import"]
        with pytest.raises(ImportError, match="cached-failure sentinel"):
            _gpu._verify_gpu_or_raise()
        assert calls == ["import"]

    def test_dll_shim_is_silent(self, monkeypatch, capsys):
        # D6.5: silent no-op off Windows and with no nvidia
        # namespace package -- no raise, no log spam
        monkeypatch.setattr(os, "name", "posix")
        assert _gpu._add_nvidia_dll_directories() is None

        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setitem(sys.modules, "nvidia", None)
        assert _gpu._add_nvidia_dll_directories() is None
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_dll_shim_registration_failure_is_silent(
        self, monkeypatch, tmp_path, capsys
    ):
        # the THIRD silent contract of D6.5 (review-added: only the
        # non-Windows and missing-nvidia branches were tested): on
        # Windows, with the nvidia namespace package present, a
        # failing ``os.add_dll_directory`` registration is swallowed
        # -- no raise, no output; the stage-(c) gate message then
        # carries the manual remedy
        (tmp_path / "cufft" / "bin").mkdir(parents=True)
        fake_nvidia = types.ModuleType("nvidia")
        fake_nvidia.__path__ = [str(tmp_path)]
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)

        def failing_add_dll_directory(path):
            raise OSError(f"registration refused: {path}")

        monkeypatch.setattr(
            os, "add_dll_directory", failing_add_dll_directory, raising=False
        )
        assert _gpu._add_nvidia_dll_directories() is None
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_no_module_scope_cupy_import(self):
        # D6.1: never at module scope.  Source-level: no top-level
        # cupy import in ``_gpu.py``; process-level: a fresh
        # interpreter importing the module (and kikuchipy) ends with
        # cupy absent from ``sys.modules`` -- a subprocess, so the
        # check cannot be poisoned by a gated test having imported
        # cupy earlier in this process
        source = inspect.getsource(_gpu)
        assert not re.search(r"^(import cupy|from cupy)", source, re.MULTILINE)
        script = (
            "import sys; "
            "import kikuchipy.indexing._spherical._gpu; "
            "assert 'cupy' not in sys.modules, 'cupy imported at module scope'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr


# ------------- xp-agnostic pipeline core under NumPy ---------------- #


class TestPipelineNumpyXp:
    """Stages 4-6 of the device pipeline run with ``xp=numpy``
    against the frozen ``_xcorr`` CPU path -- the CI-side
    correctness anchor.  [D2, D3, D11.1]"""

    @pytest.mark.parametrize("bandwidth", [XP_BANDWIDTH_EVEN, XP_BANDWIDTH_ODD])
    @pytest.mark.parametrize(
        "n_fold,mirror", [(1, False), (1, True), (4, False), (4, True)]
    )
    def test_pipeline_numpy_xp_matches_cpu(self, bandwidth, n_fold, mirror):
        # both the un-normalized (r_den None) and normalized (r_den
        # given) stage-6 branches, at odd AND even slP.  n_fold=1 is
        # the named killer of the ``(-1)**j``-only A2 seed (odd-m
        # columns flip, invisible at even n_fold); bw 16 (slP 32,
        # bwP 17) exercises the even-slP n >= bw guard and the
        # m = bw column; the raw-table NaN mutant is self-revealing
        # (an all-NaN cube).  [D2/plan 7.2]
        flm = random_spectrum(bandwidth, n_fold, mirror, seed=bandwidth)
        correlator = SphericalCrossCorrelator(bandwidth)
        slp = correlator.side_length
        bwp = correlator.half_side_length
        cells = [(2, 5, 7), (bwp - 2, 1, slp - 2), (5, slp // 2, 3)]
        gln_batch = gln_batch_for(flm, slp, cells)

        rng = np.random.default_rng(11)
        r_den64 = 1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=(bwp, slp, slp))
        for r_den in (None, r_den64):
            cubes, indices = cpu_reference_cubes(flm, gln_batch, n_fold, mirror, r_den)
            r_den32 = None if r_den is None else r_den.astype(np.float32)
            xc, xp_indices, xp_values = run_xp_pipeline(
                flm, gln_batch, n_fold, mirror, r_den32
            )
            assert xc.shape == (len(cells), bwp, slp, slp)
            assert xc.dtype == np.float32
            for b, (cube, index) in enumerate(zip(cubes, indices)):
                # cube values within the f32 band -- the mutant
                # killer (every sign/conj/slice/norm mutant perturbs
                # the cube asymmetrically).  FIXME-pin
                # (XP_CUBE_ATOL_SCALE): measured then pinned
                atol = XP_CUBE_ATOL_SCALE * float(np.abs(cube).max())
                np.testing.assert_allclose(xc[b], cube, rtol=0, atol=atol)
                # identical argmax cell, with one documented escape:
                # at even slP a z-fold or mirror symmetry of flm
                # duplicates cube values EXACTLY on the grid (e.g.
                # slP 32 with n_fold 4 repeats alpha every 8 cells),
                # so f64 and f32 rounding may break the exact tie at
                # different copies -- then the cells must still be
                # value-tied at the maximum.  A genuinely wrong
                # argmax lands below the band and fails
                if int(xp_indices[b]) != index:
                    flat = cube.reshape(-1)
                    assert float(flat[int(xp_indices[b])]) == pytest.approx(
                        float(flat[index]), abs=atol
                    )
                assert float(xp_values[b]) == pytest.approx(
                    float(cube.reshape(-1)[index]), abs=atol
                )

    def test_pipeline_numpy_xp_matches_cpu_real_ni_bw32(self):
        # the real-data anchor: one Ni pattern through the real host
        # stages 0-3, then stages 4-6 under xp=numpy against BOTH
        # correlators -- the normalized one with its real r_den and
        # the plain one (D2 stage-6 covers both normalize settings)
        indexer = ni_indexer(bandwidth=XP_BANDWIDTH_NI, refine=False)
        ncc = indexer.correlators[0]
        projector = indexer.projector
        dim = projector.dim
        buffers = (np.zeros((dim, dim)), np.zeros((dim, dim)))
        processed = _preprocess_pattern(
            ni_patterns()[0],
            good_pixels=None,
            gaussian_background=False,
            n_regions=10,
            emsphinx_compatible=True,
        )
        north, south, _ = projector.unproject(
            processed, out=buffers, return_image_quality=True
        )
        gln = projector.sht.analyze(north, south)
        gln_batch = np.ascontiguousarray(gln[None], dtype=np.complex128)

        inner = ncc.correlator
        xc_cpu = inner.compute(ncc.flm, gln, ncc.n_fold, ncc.mirror).copy()
        index_plain = _find_peak(xc_cpu)
        xc_scaled = xc_cpu.copy()
        index_norm = _scale_and_find_peak(xc_scaled, ncc.r_den)

        table = _gpu._sanitized_table(inner.wigner_d_half_pi)
        # un-normalized branch
        xc, indices, _ = run_xp_pipeline(
            ncc.flm, gln_batch, ncc.n_fold, ncc.mirror, None, table=table
        )
        assert int(indices[0]) == int(index_plain)
        atol = XP_CUBE_ATOL_SCALE * float(np.abs(xc_cpu).max())  # FIXME-pin
        np.testing.assert_allclose(xc[0], xc_cpu, rtol=0, atol=atol)
        # normalized branch, the REAL reciprocal denominator
        _, indices_norm, _ = run_xp_pipeline(
            ncc.flm,
            gln_batch,
            ncc.n_fold,
            ncc.mirror,
            ncc.r_den.astype(np.float32),
            table=table,
        )
        assert int(indices_norm[0]) == int(index_norm)

    def test_sanitized_table_zeroes_only_nans(self):
        # D2 sanitization: NaN slots (j < max(k, m), the validated
        # contract) become 0.0 and every defined slot is untouched
        # bitwise; the input is not mutated
        correlator = SphericalCrossCorrelator(XP_BANDWIDTH_ODD)
        table = correlator.wigner_d_half_pi
        before = table.copy()
        sanitized = _gpu._sanitized_table(table)
        assert sanitized is not table
        assert np.array_equal(table, before, equal_nan=True)
        nan_slots = np.isnan(table)
        assert nan_slots.any()
        assert np.all(sanitized[nan_slots] == 0.0)
        assert np.array_equal(sanitized[~nan_slots], table[~nan_slots])

    def test_argmax_first_occurrence_tie(self):
        # the strict-> first-max parity of ``_find_peak``, under
        # xp=numpy ONLY -- this proves nothing about ``cp.argmax``;
        # the cupy twin is ``test_argmax_tie_gpu`` in the gated
        # suite (review-noted).  [D4/D2]
        xc = np.zeros((2, 3, 5, 5), dtype=np.float32)
        xc[0].reshape(-1)[5] = 7.0
        xc[0].reshape(-1)[50] = 7.0  # planted exact tie
        # cube 1 is all-equal: every slot ties, index 0 must win
        indices, values = _gpu._scale_argmax_batch(np, xc, None)
        assert int(indices[0]) == 5
        assert int(indices[1]) == 0
        assert float(values[0]) == 7.0

    @pytest.mark.parametrize("slp,bwp", [(39, 20), (32, 17)])
    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_neighborhood_offsets_parity(self, slp, bwp, emsphinx_compatible):
        # the host offsets helper vs ``_extract_neighborhood`` on
        # random cubes: odd AND even slP, BOTH compat settings,
        # edge/wrap centres enumerated (incl. the even-slP one-past
        # clamp at k0 = bwP - 1).  [D2]
        rng = np.random.default_rng(slp)
        cube = rng.normal(size=(bwp, slp, slp))
        flat = cube.reshape(-1)
        centers = [
            (k0, n0, m0)
            for k0 in (0, 1, bwp - 2, bwp - 1)
            for n0 in (0, 1, slp // 2, slp - 1)
            for m0 in (0, slp // 2, slp - 1)
        ]
        centers += [
            tuple(map(int, (rng.integers(bwp), rng.integers(slp), rng.integers(slp))))
            for _ in range(20)
        ]
        nh = np.empty((3, 3, 3))
        for k0, n0, m0 in centers:
            _extract_neighborhood(flat, slp, bwp, k0, n0, m0, emsphinx_compatible, nh)
            flat_index = (k0 * slp + n0) * slp + m0
            offsets = _gpu._neighborhood_offsets(
                flat_index, bwp, slp, emsphinx_compatible
            )
            assert offsets.shape == (27,)
            gathered = flat[offsets].reshape(3, 3, 3)
            assert np.array_equal(gathered, nh), (k0, n0, m0)

    @pytest.mark.parametrize("slp,bwp", [(39, 20), (32, 17)])
    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_gather_neighborhoods_parity(self, slp, bwp, emsphinx_compatible):
        # the batched device gather itself vs per-cube
        # ``_extract_neighborhood`` under xp=numpy -- review-added:
        # no other test CALLS ``_gather_neighborhoods`` (the offsets
        # test above gathers with raw fancy indexing), yet D11.1
        # claims the gather is default-suite-covered.  The realistic
        # mutant is the batched base-offset arithmetic:
        # ``_neighborhood_offsets`` returns PER-CUBE flat offsets, so
        # the ``(B, 27)`` gather must add ``b * cube_size`` (or take
        # per-cube flat views); the cubes are DISTINCT random draws,
        # so a wrong-base gather reads another cube's values and
        # fails here on CI instead of only in gated end-to-end
        # misorientation bands.  [D2/D11.1]
        rng = np.random.default_rng(slp + int(emsphinx_compatible))
        centers = [
            (0, 0, 0),
            (0, 1, slp - 1),
            (1, slp // 2, 0),
            (bwp - 2, 0, slp - 1),
            (bwp - 1, slp - 1, slp // 2),
            (bwp - 1, slp - 1, slp - 1),  # incl. the even-slP clamp
        ]
        centers += [
            tuple(map(int, (rng.integers(bwp), rng.integers(slp), rng.integers(slp))))
            for _ in range(6)
        ]
        batch = rng.normal(size=(len(centers), bwp, slp, slp)).astype(np.float32)
        offsets = np.stack(
            [
                _gpu._neighborhood_offsets(
                    (k0 * slp + n0) * slp + m0, bwp, slp, emsphinx_compatible
                )
                for k0, n0, m0 in centers
            ]
        )
        assert offsets.shape == (len(centers), 27)
        gathered = np.asarray(_gpu._gather_neighborhoods(np, batch, offsets))
        assert gathered.shape == (len(centers), 27)
        nh = np.empty((3, 3, 3))
        for b, (k0, n0, m0) in enumerate(centers):
            _extract_neighborhood(
                np.ascontiguousarray(batch[b], dtype=np.float64).reshape(-1),
                slp,
                bwp,
                k0,
                n0,
                m0,
                emsphinx_compatible,
                nh,
            )
            assert np.array_equal(
                gathered[b].astype(np.float64).reshape(3, 3, 3), nh
            ), (b, k0, n0, m0)

    @pytest.mark.parametrize("bandwidth", [XP_BANDWIDTH_EVEN, XP_BANDWIDTH_ODD])
    def test_peak_epilogue_parity(self, bandwidth):
        # the extracted neighborhood-fed epilogue vs ``interp_peak``
        # directly on random cubes, both compat settings -- bitwise
        # (D2 stage-7: flat-index decomposition,
        # ``_interpolate_maxima``, the |x[0]|-twice bounds bug, the
        # step-rejection reset, the zyz grid formula)
        correlator = SphericalCrossCorrelator(bandwidth)
        slp = correlator.side_length
        bwp = correlator.half_side_length
        rng = np.random.default_rng(bandwidth)
        correlator.xc = np.ascontiguousarray(rng.normal(size=(bwp, slp, slp)))
        flat = correlator.xc.reshape(-1)
        indices = [int(rng.integers(correlator.xc.size)) for _ in range(10)]
        indices += [0, correlator.xc.size - 1]
        nh = np.empty((3, 3, 3))
        for emsphinx_compatible in (True, False):
            for index in indices:
                k0, remainder = divmod(index, slp * slp)
                n0, m0 = divmod(remainder, slp)
                _extract_neighborhood(
                    flat, slp, bwp, k0, n0, m0, emsphinx_compatible, nh
                )
                zyz_a, peak_a, x_a = correlator.interp_peak(index, emsphinx_compatible)
                zyz_b, peak_b, x_b = correlator._interp_peak_from_neighborhood(
                    index, nh.copy(), emsphinx_compatible
                )
                assert np.array_equal(zyz_a, zyz_b)
                assert peak_a == peak_b
                assert np.array_equal(x_a, x_b)

    def test_peak_epilogue_step_rejection_branches(self):
        # planted step-rejection cases so the |x[0]|-twice bounds
        # branch is HIT, not masked (end-to-end runs mask it: step
        # rejection is rare and Newton reconverges).  A separable
        # quadratic with its maximum 1.6 cells away along alpha
        # (x[2]) is accepted under ``emsphinx_compatible=True`` (the
        # alpha over-step is never caught) and rejected under
        # ``False``; one 1.5 cells away along beta (x[0]) is
        # rejected under BOTH.  [D2 stage-7]
        bandwidth = XP_BANDWIDTH_ODD
        correlator = SphericalCrossCorrelator(bandwidth)
        slp = correlator.side_length
        bwp = correlator.half_side_length
        rng = np.random.default_rng(3)
        base = 0.01 * rng.normal(size=(bwp, slp, slp))

        def planted_cube(offsets, curvatures):
            cube = np.ascontiguousarray(base.copy())
            k0, n0, m0 = bwp // 2, slp // 2, slp // 2
            for dk in (-1, 0, 1):
                for dn in (-1, 0, 1):
                    for dm in (-1, 0, 1):
                        cube[k0 + dk, n0 + dn, m0 + dm] = 10.0 - (
                            curvatures[0] * (dk - offsets[0]) ** 2
                            + curvatures[1] * (dn - offsets[1]) ** 2
                            + curvatures[2] * (dm - offsets[2]) ** 2
                        )
            return cube, (k0 * slp + n0) * slp + m0

        nh = np.empty((3, 3, 3))

        # alpha over-step: |x[2]| = 1.6 > 1, |x[0]|, |x[1]| small
        cube, index = planted_cube((0.2, 0.1, 1.6), (1.0, 1.0, 0.05))
        correlator.xc = cube
        zyz_t, peak_t, x_t = correlator.interp_peak(index, True)
        zyz_f, peak_f, x_f = correlator.interp_peak(index, False)
        # guard the plant itself: accepted under True, rejected
        # under False
        assert abs(x_t[2]) > 1.0
        assert np.array_equal(x_f, np.zeros(3))
        k0, remainder = divmod(index, slp * slp)
        n0, m0 = divmod(remainder, slp)
        for emsphinx_compatible, expected in (
            (True, (zyz_t, peak_t, x_t)),
            (False, (zyz_f, peak_f, x_f)),
        ):
            _extract_neighborhood(
                cube.reshape(-1), slp, bwp, k0, n0, m0, emsphinx_compatible, nh
            )
            zyz, peak, x = correlator._interp_peak_from_neighborhood(
                index, nh.copy(), emsphinx_compatible
            )
            assert np.array_equal(zyz, expected[0])
            assert peak == expected[1]
            assert np.array_equal(x, expected[2])

        # beta over-step: |x[0]| = 1.5 > 1 is rejected under BOTH
        cube, index = planted_cube((1.5, 0.1, 0.2), (0.05, 1.0, 1.0))
        correlator.xc = cube
        for emsphinx_compatible in (True, False):
            zyz_a, peak_a, x_a = correlator.interp_peak(index, emsphinx_compatible)
            assert np.array_equal(x_a, np.zeros(3))
            k0, remainder = divmod(index, slp * slp)
            n0, m0 = divmod(remainder, slp)
            _extract_neighborhood(
                cube.reshape(-1), slp, bwp, k0, n0, m0, emsphinx_compatible, nh
            )
            zyz_b, peak_b, x_b = correlator._interp_peak_from_neighborhood(
                index, nh.copy(), emsphinx_compatible
            )
            assert np.array_equal(zyz_a, zyz_b)
            assert peak_a == peak_b
            assert np.array_equal(x_a, x_b)

    def test_padding_and_stripping(self):
        # D7.3: the last partial batch is zero-padded to B, padded
        # slots are computed and DISCARDED -- a planted peak in a
        # pad slot never reaches any result row
        stack = np.arange(20, dtype=np.float64).reshape(5, 4)
        padded, slots = _gpu._pad_batch(np, stack, 8)
        assert padded.shape == (8, 4)
        assert np.array_equal(padded[:5], stack)
        assert np.all(padded[5:] == 0)
        assert np.array_equal(np.asarray(slots), np.arange(5))

        batch_values = np.zeros((8, 2))
        batch_values[:5] = np.arange(10).reshape(5, 2)
        batch_values[5:] = 1e9  # the planted peak in the pad slots
        values, written = _gpu._strip_padding(np, batch_values, slots, 5)
        assert values.shape == (5, 2)
        assert np.all(values < 1e9)
        assert np.array_equal(values, batch_values[:5])
        assert np.all(np.asarray(written))

        with pytest.raises(ValueError):
            _gpu._pad_batch(np, stack, 4)

    def test_failed_patterns_excluded(self):
        # D7.4: failed patterns (ptp == 0 guards, masked points) are
        # excluded from the device batch -- their slots padded --
        # and their rows stay on the CPU fill-row path (the written
        # mask hands them back untouched)
        stack = np.arange(20, dtype=np.float64).reshape(5, 4)
        keep = np.array([True, False, True, True, False])
        padded, slots = _gpu._pad_batch(np, stack, 4, keep)
        assert padded.shape == (4, 4)
        assert np.array_equal(np.asarray(slots), np.array([0, 2, 3]))
        assert np.array_equal(padded[:3], stack[[0, 2, 3]])
        assert np.all(padded[3:] == 0)

        batch_values = np.full((4, 2), -1.0)
        batch_values[:3] = np.arange(6).reshape(3, 2)
        batch_values[3:] = 1e9
        values, written = _gpu._strip_padding(np, batch_values, slots, 5)
        assert np.array_equal(np.asarray(written), keep)
        assert np.all(values[~keep] == 0)
        assert np.array_equal(values[keep], batch_values[:3])


# --------------------- Batch model and helpers ---------------------- #


class TestBatchModel:
    """The VRAM model, the default batch chooser and the
    ``gpu_memory_per_batch_bytes`` helper -- pure model math, no
    device query.  [D8]"""

    def test_vram_model_and_default_batch(self):
        g68 = _gpu._gpu_memory_per_pattern_bytes(68)
        g88 = _gpu._gpu_memory_per_pattern_bytes(88)
        # FIXME-pin (D8.2 calibration): drafting bands around the
        # ~50 MB measured anchor at bw 68 and the ~110 MB scaling
        # ESTIMATE at bw 88; calibrated from measured pool
        # high-water marks at the implementation gate
        assert VRAM_G68_BOUNDS[0] <= g68 <= VRAM_G68_BOUNDS[1]
        assert VRAM_G88_BOUNDS[0] <= g88 <= VRAM_G88_BOUNDS[1]
        assert g88 > g68

        # the chooser: clamp [1, 64] and the half-free-VRAM headroom
        # rule, self-consistent with the model
        assert _gpu._default_batch_size(68, 0) == 1
        assert _gpu._default_batch_size(68, 10**15) == 64
        free = 7_450_000_000  # the bench-start free VRAM, recorded
        expected = min(64, max(1, int(0.5 * free // g68)))
        assert _gpu._default_batch_size(68, free) == expected
        assert _gpu._default_batch_size(88, free) == min(
            64, max(1, int(0.5 * free // g88))
        )

    def test_vram_resident_term(self):
        with_r_den = _gpu._gpu_resident_bytes_per_phase(68, True)
        without = _gpu._gpu_resident_bytes_per_phase(68, False)
        # ``r_den`` is resident only when normalizing (D2 stage-6)
        assert with_r_den > without
        # FIXME-pin: the ~11 MB drafting band at bw 68
        assert VRAM_RESIDENT68_BOUNDS[0] <= with_r_den <= VRAM_RESIDENT68_BOUNDS[1]
        assert _gpu._gpu_resident_bytes_per_phase(88, True) > with_r_den

    def test_gpu_memory_helper(self):
        # D8.3: a METHOD taking ``batch_size`` (the review
        # clarifier; ``memory_per_worker_bytes`` is an argument-less
        # property, this is parameterized model math), equal to the
        # batch term plus the per-phase resident term, monotone in B
        indexer = ni_indexer(refine=False)
        attribute = inspect.getattr_static(
            SphericalIndexer, "gpu_memory_per_batch_bytes"
        )
        assert not isinstance(attribute, property)
        parameters = inspect.signature(
            SphericalIndexer.gpu_memory_per_batch_bytes
        ).parameters
        assert list(parameters) == ["self", "batch_size"]

        expected = 32 * _gpu._gpu_memory_per_pattern_bytes(
            NI_BANDWIDTH
        ) + indexer.n_phases * _gpu._gpu_resident_bytes_per_phase(NI_BANDWIDTH, True)
        assert indexer.gpu_memory_per_batch_bytes(32) == expected
        assert indexer.gpu_memory_per_batch_bytes(
            64
        ) > indexer.gpu_memory_per_batch_bytes(32)

    def test_gpu_memory_helper_makes_no_device_query(self):
        # pure model math: the call must not import cupy (the
        # free-VRAM query lives in ``index_patterns``).  Guarded
        # against the gated suite having imported cupy earlier in
        # this -n 0 process
        already = "cupy" in sys.modules
        ni_indexer(refine=False).gpu_memory_per_batch_bytes(8)
        if not already:
            assert "cupy" not in sys.modules


# ------------------ Gating-fixture decision logic ------------------- #


class TestFixtureGating:
    """The ``cupy_gpu`` fixture's frozen decision order: kill
    switch, structural xdist skip, then (and only then) the
    availability probe.  [D10.3-4]"""

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("KIKUCHIPY_NO_GPU_TESTS", "1")
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

        def forbidden():
            raise AssertionError("the probe must not run under the kill switch")

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", forbidden)
        reason = _cupy_gpu_skip_reason()
        assert reason is not None
        assert "KIKUCHIPY_NO_GPU_TESTS" in reason

    def test_xdist_worker_skip(self, monkeypatch):
        # the structural -n 0 rule (review-added D10.4): under any
        # xdist worker the fixture skips BEFORE probing, so a
        # default ``-n 4`` run never initialises CUDA
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")

        def forbidden():
            raise AssertionError("the probe must not run under xdist")

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", forbidden)
        reason = _cupy_gpu_skip_reason()
        assert reason is not None
        assert "-n 0" in reason

    def test_gate_failure_reason_is_instruction_bearing(self, monkeypatch):
        # a failing gate stage's actionable message IS the skip
        # reason (D10.2)
        monkeypatch.delenv("KIKUCHIPY_NO_GPU_TESTS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        frozen = "install remedy goes here"

        def failing():
            raise ImportError(frozen)

        monkeypatch.setattr(_gpu, "_verify_gpu_or_raise", failing)
        assert _cupy_gpu_skip_reason() == frozen

    def test_wgpu_marker_not_reused(self):
        # D10.1: the wgpu/Metal ``gpu`` marker keeps its documented
        # meaning and its macOS CI ``--gpu`` flag semantics; no test
        # in this file may claim it (the regex does not match its
        # own escaped spelling in this source)
        source = inspect.getsource(sys.modules[__name__])
        assert re.search(r"@pytest\.mark\.gpu\b", source) is None


# ================== Locally gated GPU suite (D10) =================== #
#
# Everything below needs the ``cupy_gpu`` fixture: it runs ONLY at
# ``-n 0`` on a machine where the three-stage gate passes, and every
# pinned number is machine-specific (RTX 2000 Ada 8 GB, driver
# 595.71, CuPy 14.2.0, Windows 11).


class TestGatedGate:
    def test_gate_passes_here(self, cupy_gpu, record_property):
        # the three stages green on this machine; versions recorded
        _gpu._verify_gpu_or_raise()
        count = cupy_gpu.cuda.runtime.getDeviceCount()
        assert count >= 1
        properties = cupy_gpu.cuda.runtime.getDeviceProperties(0)
        record_property("device", properties["name"].decode())
        record_property("cupy", cupy_gpu.__version__)
        record_property("cuda_runtime", cupy_gpu.cuda.runtime.runtimeGetVersion())


class TestGatedParity:
    """The D4 parity oracle: the CPU backend is the reference for
    every GPU output.  All bands FIXME-pin until measured."""

    def test_iq_bitwise_vs_cpu(self, cupy_gpu):
        # D4.1: preprocessing and the DCT IQ never leave the host --
        # any IQ difference means corrupted routing
        cpu, gpu = dual_backend_results(ni_patterns(), indexer_kwargs={"refine": False})
        assert np.array_equal(cpu["iq"], gpu["iq"])

    def test_refined_parity_small(self, cupy_gpu, record_property):
        cpu, gpu = dual_backend_results(ni_patterns())
        assert_refined_parity(
            cpu,
            gpu,
            REFINED_FLIP_COUNT_SMALL,
            REFINED_MISO_MEDIAN_DEG,
            REFINED_MISO_MAX_DEG,
            REFINED_SCORE_REL_DIFF,
        )
        angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
        record_property("miso_median_deg", float(np.median(angles)))
        record_property("miso_max_deg", float(angles.max()))

    def test_refined_parity_large_20pt(self, cupy_gpu, record_property):
        cpu, gpu = dual_backend_results(
            large_patterns(LARGE_STEP_20PT),
            indexer_kwargs={"step": LARGE_STEP_20PT},
        )
        assert_refined_parity(
            cpu,
            gpu,
            REFINED_FLIP_COUNT_20PT,
            REFINED_MISO_MEDIAN_DEG,
            REFINED_MISO_MAX_DEG,
            REFINED_SCORE_REL_DIFF,
        )
        angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
        record_property("miso_median_deg", float(np.median(angles)))
        record_property("miso_max_deg", float(angles.max()))

    @pytest.mark.weekly
    def test_refined_parity_large_165pt(self, cupy_gpu, record_property):
        # the flip-RATE pin lives here: N = 165 resolves a fraction
        # (review-corrected D4.2)
        cpu, gpu = dual_backend_results(
            large_patterns(LARGE_STEP_165PT),
            indexer_kwargs={"step": LARGE_STEP_165PT},
        )
        assert np.array_equal(cpu["iq"], gpu["iq"])
        angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
        agreement = float(np.mean(angles <= REFINED_MISO_MAX_DEG))
        record_property("agreement_rate", agreement)
        record_property("miso_median_deg", float(np.median(angles)))
        assert agreement >= REFINED_AGREEMENT_RATE_165PT  # FIXME-pin
        assert float(np.median(angles)) <= REFINED_MISO_MEDIAN_DEG
        assert pearson(cpu["scores"], gpu["scores"]) >= SCORE_PEARSON_MIN

    def test_coarse_cell_agreement(self, cupy_gpu, record_property):
        # D4.3: measured and RECORDED -- the honest gate is the
        # refined-to-refined oracle, not this.  The coarse
        # interpolated zyz rounds back to its cell, so equal cells
        # give equal indices except at sub-cell boundary flips
        cpu, gpu = dual_backend_results(ni_patterns(), indexer_kwargs={"refine": False})
        side = ni_indexer(refine=False).side_length
        cells_cpu = [euler_to_index(z, side) for z in cpu["zyz"][:, 0]]
        cells_gpu = [euler_to_index(z, side) for z in gpu["zyz"][:, 0]]
        agreement = float(np.mean([a == b for a, b in zip(cells_cpu, cells_gpu)]))
        record_property("coarse_cell_agreement", agreement)
        assert agreement >= COARSE_CELL_AGREEMENT_MIN  # FIXME-pin (recorded)

    def test_refine_false_coarse_scores(self, cupy_gpu):
        # D3.4/D4.5: coarse interpolated scores off the f32 cube
        cpu, gpu = dual_backend_results(ni_patterns(), indexer_kwargs={"refine": False})
        assert pearson(cpu["scores"], gpu["scores"]) >= SCORE_PEARSON_MIN
        assert float(rel_diff(cpu["scores"], gpu["scores"]).max()) <= (
            COARSE_SCORE_REL_DIFF
        )

    def test_unnormalised_gpu_parity(self, cupy_gpu, monkeypatch):
        # review-added (D2 stage-6/D9): ``normalize=False`` is
        # supported -- no ``r_den`` resident, the multiply skipped
        # (the stage-6 seam must see ``r_den is None``), plain
        # ``_find_peak`` parity
        seen_r_den = []
        original = _gpu._scale_argmax_batch

        def spy(xp, xc, r_den):
            seen_r_den.append(r_den)
            return original(xp, xc, r_den)

        monkeypatch.setattr(_gpu, "_scale_argmax_batch", spy)
        cpu, gpu = dual_backend_results(
            ni_patterns(), indexer_kwargs={"normalize": False}
        )
        assert seen_r_den, "the GPU path must route through _scale_argmax_batch"
        assert all(r is None for r in seen_r_den)
        assert_refined_parity(
            cpu,
            gpu,
            REFINED_FLIP_COUNT_SMALL,
            REFINED_MISO_MEDIAN_DEG,
            REFINED_MISO_MAX_DEG,
            REFINED_SCORE_REL_DIFF,
        )

    def test_argmax_tie_gpu(self, cupy_gpu):
        # review-added: the cupy twin of the numpy tie test --
        # ``cp.argmax`` first-occurrence pinned on device
        # (probe-confirmed at drafting review; guards regression and
        # portability).  [D4/D2]
        cp = cupy_gpu
        xc = cp.zeros((2, 3, 5, 5), dtype=cp.float32)
        flat = xc.reshape(2, -1)
        flat[0, 5] = 7.0
        flat[0, 50] = 7.0  # planted exact f32 tie
        indices, values = _gpu._scale_argmax_batch(cp, xc, None)
        assert int(indices[0]) == 5
        assert int(indices[1]) == 0  # the all-equal cube
        assert float(values[0]) == 7.0

    @pytest.mark.parametrize("emsphinx_compatible", [True, False])
    def test_emsphinx_compatible_toggle_parity(self, cupy_gpu, emsphinx_compatible):
        # D2 compat neutrality: the device stages carry no compat
        # branch, so parity holds under BOTH settings
        cpu, gpu = dual_backend_results(
            ni_patterns(),
            indexer_kwargs={"emsphinx_compatible": emsphinx_compatible},
        )
        assert_refined_parity(
            cpu,
            gpu,
            REFINED_FLIP_COUNT_SMALL,
            REFINED_MISO_MEDIAN_DEG,
            REFINED_MISO_MAX_DEG,
            REFINED_SCORE_REL_DIFF,
        )

    def test_failed_and_masked_patterns_gpu(self, cupy_gpu):
        # D4.6/D7.4: a failed (ptp == 0) pattern takes the identical
        # fill-row path under both backends and never enters the
        # device batch
        patterns = np.array(ni_patterns())
        patterns[0] = 47  # constant -> guard (a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cpu, gpu = dual_backend_results(patterns)
        for results in (cpu, gpu):
            assert np.array_equal(results["zyz"][0, 0], np.zeros(3))
            assert results["scores"][0, 0] == 0.0
            assert results["phase_id"][0, 0] == -1
            assert results["iq"][0] == 0.0
        assert np.array_equal(cpu["phase_id"], gpu["phase_id"])
        angles = misorientation_deg(cpu["zyz"][1:, 0], gpu["zyz"][1:, 0])
        assert float(angles.max()) <= REFINED_MISO_MAX_DEG


class TestGatedPseudoSymmetryAndPhases:
    def test_psym_dual_backend(self, cupy_gpu):
        # D9: the Phase 8 variant loop is host code -- supported
        # UNCHANGED under backend="gpu".  A true 90-deg-z Oh op with
        # n_best=2 on the un-normalized path, mirroring
        # ``test_variants_on_the_un_normalised_path``
        ops = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(90))
        cpu, gpu = dual_backend_results(
            ni_patterns(),
            indexer_kwargs={"normalize": False, "pseudo_symmetry_ops": ops},
            n_best=2,
        )
        assert "pseudo_symmetry_index" in cpu
        assert "pseudo_symmetry_index" in gpu
        # FIXME-pin (PSYM_INDEX_FLIP_COUNT): a true op guarantees
        # near-ties, so the measured flip count may pin above zero
        flips = int(
            (cpu["pseudo_symmetry_index"] != gpu["pseudo_symmetry_index"]).sum()
        )
        assert flips <= PSYM_INDEX_FLIP_COUNT
        angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
        assert float(np.median(angles)) <= REFINED_MISO_MEDIAN_DEG
        assert float(rel_diff(cpu["scores"], gpu["scores"]).max()) <= (
            REFINED_SCORE_REL_DIFF
        )

    def test_multiphase_dual_backend(self, cupy_gpu):
        # same-symmetry sign-scrambled second phase: winner phase
        # parity per pattern kills cross-phase table/r_den/n_fold
        # mix-ups.  [D2/plan 7.2]
        phases = [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics("m-3m")]
        results = {}
        for backend in ("cpu", "gpu"):
            indexer = SphericalIndexer(phases, ni_detector(), backend=backend)
            results[backend] = indexer.index_patterns(ni_patterns(), progressbar=False)
        cpu, gpu = results["cpu"], results["gpu"]
        assert np.array_equal(cpu["phase_id"], gpu["phase_id"])
        assert np.all(cpu["phase_id"][:, 0] == 0)  # the true phase wins
        assert float(rel_diff(cpu["scores"], gpu["scores"]).max()) <= (
            REFINED_SCORE_REL_DIFF
        )

    def test_multiphase_mixed_symmetry_dual_backend(self, cupy_gpu):
        # review-added (D2 freshness rule): the second phase carries
        # DIFFERENT declared symmetry (n_fold 1, no mirror), so a
        # ``fxc`` batch buffer not re-zeroed between phases keeps
        # stale ``m % 4 != 0`` columns -- which the same-symmetry
        # scramble above shares and cannot catch
        phases = [ni_harmonics(NI_BANDWIDTH), scrambled_harmonics("1")]
        results = {}
        for backend in ("cpu", "gpu"):
            indexer = SphericalIndexer(phases, ni_detector(), backend=backend)
            results[backend] = indexer.index_patterns(ni_patterns(), progressbar=False)
        cpu, gpu = results["cpu"], results["gpu"]
        assert np.array_equal(cpu["phase_id"], gpu["phase_id"])
        angles = misorientation_deg(cpu["zyz"][:, 0], gpu["zyz"][:, 0])
        assert float(angles.max()) <= REFINED_MISO_MAX_DEG
        assert float(rel_diff(cpu["scores"], gpu["scores"]).max()) <= (
            REFINED_SCORE_REL_DIFF
        )


class TestGatedDeterminism:
    def test_run_to_run_bitwise(self, cupy_gpu):
        # D5.2: fixed device, driver and batch size -> bitwise
        # deterministic run to run (no atomics-based reductions in
        # the device code; library argmax/matmul/cuFFT only).
        # Machine-specific: RTX 2000 Ada, driver 595.71, CuPy 14.2.0
        first, second = (
            ni_indexer(backend="gpu").index_patterns(
                ni_patterns(), chunksize=8, progressbar=False
            )
            for _ in range(2)
        )
        for key, value in first.items():
            assert np.array_equal(value, second[key]), key

    def test_run_to_run_bitwise_4_workers(self, cupy_gpu):
        # the D7 device-lock stress named by validation.md's D7 row
        # and plan 7.2 ("device lock dropped: killed by a 4-worker
        # gated stress run") -- review-added: it was claimed but not
        # implemented, and the plain test above creates at most 2
        # concurrent chunks.  Four concurrent DASK workers (the
        # relevant axis -- the structural xdist skip makes pytest
        # ``-n`` irrelevant here) contend over five chunks (9
        # patterns at chunksize 2) for the lock-guarded device
        # section; a dropped or broken device lock bites here if it
        # bites anywhere, surfacing as run-to-run divergence
        with dask.config.set(scheduler="threads", num_workers=4):
            first, second = (
                ni_indexer(backend="gpu").index_patterns(
                    ni_patterns(), chunksize=2, progressbar=False
                )
                for _ in range(2)
            )
        for key, value in first.items():
            assert np.array_equal(value, second[key]), key

    def test_batch_size_invariance(self, cupy_gpu, record_property):
        # D5.3: MEASURED across B in {8, 32, default}.  Written as
        # bitwise; if the implementation-gate measurement refutes
        # bitwise, the pin is relaxed to the measured tolerance and
        # the deviation recorded in validation.md (FIXME-pin)
        results = [
            ni_indexer(backend="gpu").index_patterns(
                ni_patterns(), chunksize=chunksize, progressbar=False
            )
            for chunksize in (8, 32, None)
        ]
        for other in results[1:]:
            for key, value in results[0].items():
                bitwise = np.array_equal(value, other[key])
                record_property(f"bitwise_{key}", bitwise)
                assert bitwise, key


class TestGatedRobustness:
    """OOM handling, device-error propagation, laziness and the
    verbose surface.  The monkeypatched seams (``_GpuSession``,
    ``_spectrum_batch``, ``_inverse_fft_batch``) are the plan-named
    functions the implementation MUST route through."""

    def test_oom_halving_at_session_build(self, cupy_gpu, monkeypatch):
        # D8.4 window (a): OOM at session build halves B, frees the
        # pool and retries; the run completes
        cp = cupy_gpu
        built = []
        original = _gpu._GpuSession

        class FlakySession(original):
            def __init__(self, indexer, batch_size):
                built.append(batch_size)
                if len(built) == 1:
                    raise cp.cuda.memory.OutOfMemoryError(0, 0)
                super().__init__(indexer, batch_size)

        monkeypatch.setattr(_gpu, "_GpuSession", FlakySession)
        monkeypatch.setattr(_indexer, "_GpuSession", FlakySession, raising=False)
        indexer = ni_indexer(backend="gpu")
        results = indexer.index_patterns(ni_patterns(), chunksize=8, progressbar=False)
        assert len(built) >= 2
        assert built[1] == built[0] // 2
        assert results["zyz"].shape == (9, 1, 3)
        # the halved run must also be CORRECT, not merely complete
        # (review-strengthened to match the mid-compute twin): no
        # fill rows, and inside the D4 band of a CPU reference -- a
        # retry path completing with corrupted or zero rows fails
        assert np.all(results["phase_id"][:, 0] != -1)
        reference = ni_indexer(backend="cpu").index_patterns(
            ni_patterns(), progressbar=False
        )
        angles = misorientation_deg(reference["zyz"][:, 0], results["zyz"][:, 0])
        assert float(angles.max()) <= REFINED_MISO_MAX_DEG

    def test_oom_halving_mid_compute(self, cupy_gpu, monkeypatch):
        # D8.4 window (b): a mid-compute OOM aborts the compute;
        # ``index_patterns`` disposes the session, rebuilds session
        # + graph at B/2 and re-runs the WHOLE map -- results come
        # entirely from the final B (never mixed batch sizes)
        cp = cupy_gpu
        calls = []
        original = _gpu._inverse_fft_batch

        def flaky(xp, fft_ns, fxc, n_fold):
            calls.append(True)
            if len(calls) == 1:
                raise cp.cuda.memory.OutOfMemoryError(0, 0)
            return original(xp, fft_ns, fxc, n_fold)

        monkeypatch.setattr(_gpu, "_inverse_fft_batch", flaky)
        indexer = ni_indexer(backend="gpu")
        results = indexer.index_patterns(ni_patterns(), chunksize=8, progressbar=False)
        assert len(calls) >= 2
        reference = ni_indexer(backend="cpu").index_patterns(
            ni_patterns(), progressbar=False
        )
        angles = misorientation_deg(reference["zyz"][:, 0], results["zyz"][:, 0])
        assert float(angles.max()) <= REFINED_MISO_MAX_DEG

    def test_oom_below_batch_one_raises_memory_error(self, cupy_gpu, monkeypatch):
        # D8.4 floor: at B = 1 the OOM re-raises as ``MemoryError``
        # naming the bandwidth, the model bytes, the free VRAM and
        # the remedies -- NEVER a silent fallback to the CPU backend
        cp = cupy_gpu

        class AlwaysOom(_gpu._GpuSession):
            def __init__(self, indexer, batch_size):
                raise cp.cuda.memory.OutOfMemoryError(0, 0)

        monkeypatch.setattr(_gpu, "_GpuSession", AlwaysOom)
        monkeypatch.setattr(_indexer, "_GpuSession", AlwaysOom, raising=False)
        indexer = ni_indexer(backend="gpu")
        with pytest.raises(MemoryError) as info:
            indexer.index_patterns(ni_patterns(), chunksize=8, progressbar=False)
        message = str(info.value)
        # the frozen D8.4 content -- the bandwidth, the model bytes,
        # the free VRAM and BOTH remedies (review-tightened: two bare
        # substrings pinned almost nothing, and a bare "68" could
        # match inside a byte count)
        assert re.search(r"(?<!\d)68(?!\d)", message)  # the bandwidth
        assert re.search(
            r"\d[\d,_.]*\s*(GiB|GB|MiB|MB|KiB|kB|KB|bytes?)\b", message
        )  # a model-bytes / free-VRAM figure with its unit
        assert "VRAM" in message
        assert "bandwidth" in message.lower()  # the smaller-bandwidth remedy
        assert re.search(r"backend\s*=\s*['\"]cpu['\"]", message)  # the cpu remedy

    def test_device_error_propagates(self, cupy_gpu, monkeypatch):
        # review-added D7.7/D14.8: a non-OOM device-stage exception
        # FAILS the run -- never swallowed into per-pattern fill
        # rows (a whole chunk of silent fill rows is the forbidden
        # failure mode)
        def exploding(xp, g, g2, a, a2, fxc, n_fold, mirror):
            raise RuntimeError("planted device failure")

        monkeypatch.setattr(_gpu, "_spectrum_batch", exploding)
        indexer = ni_indexer(backend="gpu")
        with pytest.raises(RuntimeError, match="planted device failure"):
            indexer.index_patterns(ni_patterns(), chunksize=8, progressbar=False)

    def test_lazy_input_eager_output(self, cupy_gpu, monkeypatch):
        # D7.5: lazy dask input computes eagerly to NumPy, with
        # ``scheduler="threads"`` forced on the compute so a global
        # process/distributed scheduler can never move device
        # handles across processes
        captured = {}
        original = da.Array.compute

        def spying_compute(self, **kwargs):
            captured.update(kwargs)
            return original(self, **kwargs)

        monkeypatch.setattr(da.Array, "compute", spying_compute)
        lazy = da.from_array(np.array(ni_patterns()), chunks=(3, -1, -1))
        indexer = ni_indexer(backend="gpu")
        results = indexer.index_patterns(lazy, chunksize=8, progressbar=False)
        assert captured.get("scheduler") == "threads"
        for key in ("zyz", "scores", "phase_id", "iq"):
            assert isinstance(results[key], np.ndarray)

    def test_default_batch_size_wiring(self, cupy_gpu, monkeypatch):
        # D8.1/D8.2 wiring (review-added): a ``chunksize=None`` GPU
        # run must derive B from ``_default_batch_size`` fed by a
        # real free-VRAM query, and that return must BECOME the
        # session batch size -- a mutant leaving the GPU default on
        # the CPU ``_batch_estimate`` heuristic (15/chunk at bw 68)
        # passes every other committed test and only degrades
        # throughput quietly.  The explicit-chunksize override path
        # is already exercised by the chunksize=8 tests
        chosen = []
        original_default = _gpu._default_batch_size

        def spying_default(bandwidth, free_bytes):
            value = original_default(bandwidth, free_bytes)
            chosen.append((bandwidth, free_bytes, value))
            return value

        monkeypatch.setattr(_gpu, "_default_batch_size", spying_default)
        monkeypatch.setattr(
            _indexer, "_default_batch_size", spying_default, raising=False
        )
        built = []
        original_session = _gpu._GpuSession

        class SpyingSession(original_session):
            def __init__(self, indexer, batch_size):
                built.append(batch_size)
                super().__init__(indexer, batch_size)

        monkeypatch.setattr(_gpu, "_GpuSession", SpyingSession)
        monkeypatch.setattr(_indexer, "_GpuSession", SpyingSession, raising=False)
        indexer = ni_indexer(backend="gpu")
        indexer.index_patterns(ni_patterns(), progressbar=False)
        assert chosen, "chunksize=None must consult _default_batch_size"
        assert chosen[-1][0] == NI_BANDWIDTH
        assert chosen[-1][1] > 0  # a real free-VRAM query fed the model
        assert built, "the run must build its session through _GpuSession"
        assert built[0] == chosen[-1][2]  # the model's B IS the batch size

    def test_verbose_gpu_info_line(self, cupy_gpu):
        # D8.3: the information message names the device, the VRAM,
        # the batch size and the model bytes (exact wording frozen at
        # the implementation gate; content pins review-tightened --
        # the bare 'gpu'/'VRAM' substrings asserted almost nothing of
        # what validation.md claims this test covers)
        indexer = ni_indexer(backend="gpu")
        message = indexer.get_info_message(9, chunksize=8)
        assert "gpu" in message.lower()
        assert "VRAM" in message
        device_name = cupy_gpu.cuda.runtime.getDeviceProperties(0)["name"].decode()
        assert device_name in message  # the device name
        assert re.search(r"(?<!\d)8(?!\d)", message)  # the batch size B
        assert re.search(
            r"\d[\d,_.]*\s*(GiB|GB|MiB|MB|KiB|kB|KB|bytes?)\b", message
        )  # the model bytes figure with its unit

    def test_verbose_warns_above_free_vram(self, cupy_gpu):
        # the D8.3 warning branch (review-added: it was entirely
        # untested and would land as an unexplained miss in the D11
        # 100 %-coverage gate): an explicit chunksize whose model
        # bytes exceed the free VRAM makes the info message carry a
        # warning naming the excess -- the 2 GiB host-memory
        # warning's device counterpart.  chunksize=100_000 models
        # ~5 PB at bw 68, above any card
        indexer = ni_indexer(backend="gpu")
        message = indexer.get_info_message(9, chunksize=100_000)
        assert "VRAM" in message
        assert "exceed" in message.lower()

    def test_indexer_holds_no_device_state(self, cupy_gpu):
        # D7.2: the session is per-call; after a run the indexer
        # still holds only host objects (nothing device-resident can
        # be captured in a graph or pickled)
        indexer = ni_indexer(backend="gpu")
        indexer.index_patterns(ni_patterns(), chunksize=8, progressbar=False)
        for name, value in vars(indexer).items():
            assert not type(value).__module__.startswith("cupy"), name
        assert indexer.backend == "gpu"

    def test_signal_method_gpu_smoke(self, cupy_gpu):
        # ``EBSD.spherical_indexing(backend="gpu")`` end to end: a
        # crystal map whose rotations sit inside the D4 band of the
        # CPU map's
        signal = kp.data.nickel_ebsd_small()
        signal.remove_static_background(show_progressbar=False)
        signal.remove_dynamic_background(show_progressbar=False)
        maps = {}
        for backend in ("cpu", "gpu"):
            maps[backend] = signal.spherical_indexing(
                ni_harmonics(NI_BANDWIDTH),
                ni_detector(),
                backend=backend,
                verbose=0,
            )
        angles = np.asarray(
            Orientation(maps["cpu"].rotations.data, Oh).angle_with(
                Orientation(maps["gpu"].rotations.data, Oh), degrees=True
            )
        ).ravel()
        assert float(angles.max()) <= REFINED_MISO_MAX_DEG
        assert np.array_equal(maps["cpu"].iq, maps["gpu"].iq)


class TestGatedThroughput:
    def test_throughput_floor(self, cupy_gpu, record_property):
        # D12.1, the go/no-go: end-to-end backend="gpu" refined
        # throughput on the nickel_ebsd_large route at bw 68 >= the
        # idle-machine 8-worker CPU baseline recorded in
        # validation.md (the floor's left side).  Machine-specific:
        # RTX 2000 Ada 8 GB vs the same machine's 20-core CPU
        if CPU_BASELINE_8_WORKERS_PAT_S is None:
            pytest.skip(
                "the idle-machine 8-worker CPU baseline is not recorded "
                "yet; measure it per validation.md (open question 9.5) "
                "and set CPU_BASELINE_8_WORKERS_PAT_S"
            )
        from time import perf_counter

        patterns = large_patterns(1)
        indexer = large_indexer(step=1, backend="gpu")
        # warm-up on a slice (kernels, plans, pools)
        indexer.index_patterns(patterns[:64], progressbar=False)
        best = 0.0
        for _ in range(3):
            start = perf_counter()
            indexer.index_patterns(patterns, progressbar=False)
            best = max(best, patterns.shape[0] / (perf_counter() - start))
        record_property("gpu_pat_per_s_best_of_3", best)
        record_property("cpu_baseline_pat_per_s", CPU_BASELINE_8_WORKERS_PAT_S)
        assert best >= CPU_BASELINE_8_WORKERS_PAT_S
