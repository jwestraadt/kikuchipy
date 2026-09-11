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

"""Tests of ``kikuchipy.indexing._spherical._symmetry``.

Covers the "Symmetry" assertions of
``specs/2026-08-16-sht-master-spectra-and-file/validation.md``:

- The 40 keys of ``Z_ROTATION_ORDER_AND_MIRROR`` are exactly the 38
  ``orix.quaternion.symmetry._groups`` names plus the ``"2"`` and
  ``"m"`` aliases ``get_point_group`` returns, and every value equals
  an operator oracle computed on the matching ``Symmetry``, not on
  ``get_axis_orders``, which counts improper elements too.
- ``point_group_flags`` on ``None`` and on an unknown name.
- ``space_group_for_point_group`` and ``candidate_space_groups``,
  including the four names whose space groups carry two distinct
  ``(z_rot, flags)`` pairs.
- ``validate_flags``: the Ni coefficients keep ``(4, True)``, and
  synthetic violations downgrade to the *largest satisfied divisor*
  with a warning, with the boundary of
  ``SYMMETRY_POWER_TOLERANCE`` asserted from both sides.
"""

from pathlib import Path

import numpy as np
from orix.crystal_map import Phase
from orix.quaternion.symmetry import C2, Cs, _groups, get_point_group
import pytest

from kikuchipy.data._data import Dataset
from kikuchipy.indexing._spherical import _sht_file, _symmetry

# The names ``get_point_group`` returns over the 230 space groups
GET_POINT_GROUP_NAMES = sorted({get_point_group(sg).name for sg in range(1, 231)})

# The two z-unique aliases which are not in ``_groups``
ALIASES = {"2": C2, "m": Cs}


def _nickel_file() -> Path:
    """Return the path of the in-package mp2sht Ni ``.sht`` file."""
    return Path(Dataset("emsphinx/ni_small_20kv_bw384.sht").fetch_file_path())


def _operator_flags(symmetry) -> tuple[int, bool]:
    """Return ``(n_fold, has_equatorial_mirror)`` computed from the
    operators of an orix ``Symmetry``.

    ``n_fold`` is one plus the number of *proper* elements whose axis
    is along z and whose angle is non-zero; the mirror is whether any
    *improper* element is a two fold about z.
    ``Symmetry.get_axis_orders()`` is deliberately not the oracle: it
    counts improper elements too and reports 3 for the z axis of
    ``"2/m"``.
    """
    proper = symmetry[~symmetry.improper]
    axis = proper.axis.data
    angle = proper.angle
    along_z = np.isclose(np.abs(axis[:, 2]), 1) & (angle > 1e-6)
    n_fold = 1 + int(np.count_nonzero(along_z))
    improper = symmetry[symmetry.improper]
    imp_axis = improper.axis.data
    imp_angle = improper.angle
    mirror = bool(
        np.any(np.isclose(imp_angle, np.pi) & np.isclose(np.abs(imp_axis[:, 2]), 1))
    )
    return n_fold, mirror


def _symmetry_for_name(name: str):
    """Return the ``Symmetry`` of a table key."""
    if name in ALIASES:
        return ALIASES[name]
    for group in _groups:
        if group.name == name:
            return group
    raise AssertionError(f"no orix symmetry named {name!r}")


def _synthetic_alm(bandwidth: int = 24) -> np.ndarray:
    """Return coefficients with a 6-fold axis and an equatorial
    mirror, i.e. non-zero only at ``m % 6 == 0`` and even ``l + m``.
    """
    alm = np.zeros((bandwidth, bandwidth), dtype=np.complex128)
    for m in range(0, bandwidth, 6):
        for degree in range(m, bandwidth):
            if (degree + m) % 2:
                continue
            alm[m, degree] = complex(1.0 / (1 + degree), 0.0 if m == 0 else 0.5)
    return alm


def _fill_row(alm: np.ndarray, order: int, amplitude: float) -> np.ndarray:
    """Return a copy of ``alm`` with one order row filled."""
    out = alm.copy()
    for degree in range(order, out.shape[0]):
        if (degree + order) % 2:
            continue
        out[order, degree] = complex(amplitude, 0.0)
    return out


class TestTable:
    def test_the_key_set_is_the_orix_names_plus_the_two_aliases(self):
        expected = {group.name for group in _groups} | {"2", "m"}
        assert set(_symmetry.Z_ROTATION_ORDER_AND_MIRROR) == expected
        assert len(_symmetry.Z_ROTATION_ORDER_AND_MIRROR) == 40
        assert len(_groups) == 38

    @pytest.mark.parametrize(
        "name", sorted({group.name for group in _groups} | {"2", "m"})
    )
    def test_every_value_equals_the_operator_oracle(self, name):
        symmetry = _symmetry_for_name(name)
        if name == "mm2" and _operator_flags(_symmetry_for_name("mm2"))[0] != 2:
            # orix 0.12.1 (the CI "oldest" job) orients the two-fold
            # of mm2 about x, so its operators give (1, True) where the
            # table -- and orix >= 0.13 -- give (2, False); the table's
            # value is pinned separately in TestTable spot values
            pytest.skip("this orix version orients the mm2 two-fold about x")
        assert _symmetry.Z_ROTATION_ORDER_AND_MIRROR[name] == _operator_flags(symmetry)

    @pytest.mark.parametrize(
        "name, value",
        [
            ("m-3m", (4, True)),
            ("mm2", (2, False)),
            ("-4", (2, False)),
            ("-6", (3, True)),
            ("-6m2", (3, True)),
            ("-3m", (3, False)),
            ("11m", (1, True)),
            ("1m1", (1, False)),
            ("112", (2, False)),
            ("121", (1, False)),
            ("2", (2, False)),
            ("m", (1, True)),
            ("2/m", (2, True)),
            ("23", (2, False)),
            ("m-3", (2, True)),
        ],
    )
    def test_spot_values(self, name, value):
        assert _symmetry.Z_ROTATION_ORDER_AND_MIRROR[name] == value

    def test_every_get_point_group_name_is_a_key(self):
        assert len(GET_POINT_GROUP_NAMES) == 32
        assert set(GET_POINT_GROUP_NAMES) <= set(_symmetry.Z_ROTATION_ORDER_AND_MIRROR)

    def test_the_two_aliases_are_not_in_the_orix_group_list(self):
        # ``Phase(space_group=3).point_group.name`` is "2" (C2, a two
        # fold about z) and ``Phase(space_group=6)`` gives "m" (Cs, a
        # mirror perpendicular to z); neither string is in ``_groups``
        assert Phase(space_group=3).point_group.name == "2"
        assert Phase(space_group=6).point_group.name == "m"
        assert {"2", "m"} & {group.name for group in _groups} == set()

    def test_the_power_tolerance(self):
        assert _symmetry.SYMMETRY_POWER_TOLERANCE == 1e-8


class TestPointGroupFlags:
    @pytest.mark.parametrize(
        "name", sorted({group.name for group in _groups} | {"2", "m"})
    )
    def test_the_name_gives_the_table_value(self, name):
        assert (
            _symmetry.point_group_flags(name)
            == (_symmetry.Z_ROTATION_ORDER_AND_MIRROR[name])
        )

    def test_a_symmetry_object_works_too(self):
        assert _symmetry.point_group_flags(Phase(space_group=225).point_group) == (
            4,
            True,
        )

    def test_none_gives_the_safe_default(self):
        assert _symmetry.point_group_flags(None) == (1, False)

    def test_an_unknown_name_raises_and_lists_the_known_ones(self):
        with pytest.raises(ValueError) as info:
            _symmetry.point_group_flags("112/m")
        message = str(info.value)
        assert "112/m" in message
        assert "m-3m" in message


class TestSpaceGroupForPointGroup:
    @pytest.mark.parametrize(
        "name, space_group",
        [
            ("m-3m", 221),
            ("432", 207),
            ("2", 3),
            ("112", 3),
            ("121", 3),
            ("m", 6),
            ("11m", 6),
            ("2/m", 10),
            ("32", 149),
            ("321", 150),
            ("312", 149),
            ("3m", 156),
            ("-3m", 162),
            ("-6m2", 187),
            ("-42m", 111),
            ("1", 1),
            ("-1", 2),
        ],
    )
    def test_spot_values(self, name, space_group):
        assert _symmetry.space_group_for_point_group(name) == space_group

    @pytest.mark.parametrize("name", GET_POINT_GROUP_NAMES)
    def test_the_result_maps_back_to_the_name(self, name):
        space_group = _symmetry.space_group_for_point_group(name)
        assert 1 <= space_group <= 230
        assert get_point_group(space_group).name == name

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError):
            _symmetry.space_group_for_point_group("112/m")


class TestCandidateSpaceGroups:
    @pytest.mark.parametrize(
        "name, candidates",
        [
            # The four flag-ambiguous names: the mirror plane contains
            # z through x in one setting and through y in the other,
            # so the packed rows are real in one and imaginary in the
            # other
            ("3m", (156, 157)),
            ("-3m", (162, 164)),
            ("-42m", (111, 115)),
            ("-6m2", (187, 189)),
            # Unambiguous, although 149 and 150 are two space groups
            ("m-3m", (221,)),
            ("32", (149,)),
            ("mm2", (25,)),
        ],
    )
    def test_the_candidates(self, name, candidates):
        assert _symmetry.candidate_space_groups(name) == candidates

    @pytest.mark.parametrize("name", ["3m", "-3m", "-42m", "-6m2", "m-3m", "32", "mm2"])
    def test_the_candidates_carry_distinct_flag_pairs(self, name):
        candidates = _symmetry.candidate_space_groups(name)
        pairs = {
            (
                _sht_file.space_group_z_rotation(sg),
                _sht_file.space_group_compression_flags(sg),
            )
            for sg in candidates
        }
        assert len(pairs) == len(candidates)

    def test_the_first_candidate_is_the_fallback(self):
        for name in ("3m", "-3m", "-42m", "-6m2", "m-3m"):
            assert _symmetry.candidate_space_groups(name)[
                0
            ] == _symmetry.space_group_for_point_group(name)

    @pytest.mark.parametrize(
        "name, space_group", sorted(_symmetry.AXIS_ALIAS_SPACE_GROUPS.items())
    )
    def test_the_eight_aliases_give_their_tabulated_space_group(
        self, name, space_group
    ):
        # orix never returns these names, so _space_groups_by_name
        # has no entry for them and the table is the only source
        assert _symmetry.candidate_space_groups(name) == (space_group,)
        assert _symmetry.space_group_for_point_group(name) == space_group

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError):
            _symmetry.candidate_space_groups("112/m")


class TestSystematicZeroPower:
    def test_a_clean_six_fold_has_no_power_in_the_zeros(self):
        alm = _synthetic_alm()
        rotation, mirror = _symmetry.systematic_zero_power(alm, 6, True)
        assert rotation == 0
        assert mirror == 0

    def test_a_one_fold_reports_zero_rotation_power(self):
        alm = _synthetic_alm()
        rotation, _ = _symmetry.systematic_zero_power(alm, 1, False)
        assert rotation == 0

    def test_no_mirror_reports_zero_mirror_power(self):
        alm = _synthetic_alm()
        _, mirror = _symmetry.systematic_zero_power(alm, 6, False)
        assert mirror == 0

    def test_the_nickel_coefficients_are_clean(self):
        sht = _sht_file.read_sht(_nickel_file())
        alm = _sht_file.unpack_harmonics(
            sht.harmonics.packed,
            sht.harmonics.bandwidth,
            sht.harmonics.z_rot,
            sht.harmonics.flags,
        )
        rotation, mirror = _symmetry.systematic_zero_power(alm, 4, True)
        assert rotation <= 1e-20
        assert mirror <= 1e-20

    def test_the_m_greater_than_zero_entries_count_twice(self):
        # A single entry at m = 1 against a single entry at m = 0 of
        # the same magnitude: the relative power of the m = 1 one is
        # 2 / 3, not 1 / 2
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1
        alm[1, 1] = 1
        rotation, _ = _symmetry.systematic_zero_power(alm, 2, False)
        assert rotation == pytest.approx(2 / 3, rel=1e-12)


class TestValidateFlags:
    def test_the_nickel_coefficients_keep_their_flags(self):
        sht = _sht_file.read_sht(_nickel_file())
        alm = _sht_file.unpack_harmonics(
            sht.harmonics.packed,
            sht.harmonics.bandwidth,
            sht.harmonics.z_rot,
            sht.harmonics.flags,
        )
        n_fold, mirror, warnings = _symmetry.validate_flags(alm, 4, True)
        assert (n_fold, mirror) == (4, True)
        assert warnings == []

    def test_a_clean_spectrum_is_unchanged(self):
        alm = _synthetic_alm()
        assert _symmetry.validate_flags(alm, 6, True)[:2] == (6, True)

    @pytest.mark.parametrize(
        "claimed, filled_orders, expected",
        [
            # Largest satisfied divisor, not a collapse to one.
            # ``_synthetic_alm`` is non-zero only on m in {0, 6, 12,
            # 18}, so filling row m = 3 leaves the non-zero orders
            # {0, 3, 6, 12, 18}, whose largest divisor of 6 is 3, and
            # filling row m = 2 leaves {0, 2, 6, 12, 18}, i.e. 2.
            # ``plan.md`` line 50 and ``validation.md`` line 35 had
            # the two the other way round and were corrected
            # 2026-08-16
            (4, (2,), 2),
            (4, (1,), 1),
            (6, (3,), 3),
            (6, (2,), 2),
            (6, (2, 3), 1),
            (3, (1,), 1),
            (2, (1,), 1),
        ],
    )
    def test_the_divisor_downgrade(self, claimed, filled_orders, expected):
        alm = _synthetic_alm()
        for order in filled_orders:
            alm = _fill_row(alm, order, 0.5)
        n_fold, _, warnings = _symmetry.validate_flags(alm, claimed, True)
        assert n_fold == expected
        assert warnings
        assert any("n_fold" in message for message in warnings)
        assert any(str(expected) in message for message in warnings)

    def test_a_broken_mirror_is_dropped(self):
        alm = _synthetic_alm()
        alm[0, 1] = 0.5
        n_fold, mirror, warnings = _symmetry.validate_flags(alm, 6, True)
        assert n_fold == 6
        assert mirror is False
        assert any("equatorial mirror" in message for message in warnings)

    @pytest.mark.parametrize(
        "relative_power, passes", [(0.9e-8, True), (1.1e-8, False)]
    )
    def test_the_tolerance_boundary(self, relative_power, passes):
        # ``<=``: an exact 1e-8 cannot be constructed in floating
        # point and is not asserted
        alm = np.zeros((8, 8), dtype=np.complex128)
        alm[0, 0] = 1.0
        # A single m = 2 entry of power 2 |a| ** 2 against 1
        alm[2, 2] = np.sqrt(relative_power / 2)
        n_fold, _, warnings = _symmetry.validate_flags(alm, 4, False)
        if passes:
            assert n_fold == 4
            assert warnings == []
        else:
            assert n_fold == 2
            assert warnings
