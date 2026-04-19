import numpy as np
import pandas as pd

from wcp.tournament.bracket import R32_SLOTS, resolve_r32_pairs
from wcp.tournament.draw import ALL_TEAMS, GROUPS, POSITIONS


def test_groups_have_48_distinct_teams():
    assert len(ALL_TEAMS) == 48
    assert len(set(ALL_TEAMS)) == 48


def test_positions_index_correct():
    assert POSITIONS["A1"] == "Mexico"
    assert POSITIONS["L4"] == "Panama"
    assert POSITIONS["I1"] == "France"


def test_annex_c_all_495_covered():
    from wcp.tournament.bracket import _load_annex_c
    table = _load_annex_c()
    assert len(table) == 495


def test_resolve_r32_smoke():
    """Given 8 qualifying groups, we get 16 well-formed pairs."""
    pairs = resolve_r32_pairs({"B", "D", "E", "F", "I", "J", "K", "L"})
    assert len(pairs) == 16
    # Every position code is either 1X, 2X, or 3X for some group letter
    for a, b in pairs:
        for code in (a, b):
            assert code[0] in {"1", "2", "3"}
            assert code[1] in "ABCDEFGHIJKL"
    # No team advances twice — collect all codes and check for dupes.
    codes = [c for pair in pairs for c in pair]
    assert len(codes) == len(set(codes))


def test_resolve_r32_matches_fifa_option_67():
    """Sanity check against the real 2026 tournament matchups."""
    pairs = resolve_r32_pairs({"B", "D", "E", "F", "I", "J", "K", "L"})
    # From FIFA / Wikipedia knockout page:
    expected = {
        ("1A", "3E"), ("1B", "3J"), ("1D", "3B"), ("1E", "3D"),
        ("1G", "3I"), ("1I", "3F"), ("1K", "3L"), ("1L", "3K"),
        ("2A", "2B"), ("1F", "2C"), ("1C", "2F"), ("2E", "2I"),
        ("2K", "2L"), ("1H", "2J"), ("1J", "2H"), ("2D", "2G"),
    }
    assert set(pairs) == expected
