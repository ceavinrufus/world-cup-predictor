import pandas as pd

from wcp.data.elo import (
    DEFAULT_RATING,
    _expected,
    _g_factor,
    _k_factor,
    compute_elo,
)


def test_expected_score_symmetric():
    # Neutral, equal ratings → 50/50
    assert abs(_expected(1500, 1500, neutral=True) - 0.5) < 1e-9


def test_home_advantage_matters():
    # Same ratings, home not neutral → home > 50%
    p = _expected(1500, 1500, neutral=False)
    assert p > 0.55


def test_g_factor_bands():
    assert _g_factor(0) == 1.0
    assert _g_factor(1) == 1.0
    assert _g_factor(2) == 1.5
    assert _g_factor(3) == (11 + 3) / 8
    assert _g_factor(5) == (11 + 5) / 8


def test_k_factor_hierarchy():
    assert _k_factor("FIFA World Cup") > _k_factor("UEFA Euro")
    assert _k_factor("UEFA Euro") > _k_factor("Friendly")
    assert _k_factor("FIFA World Cup qualification") == 40.0


def test_compute_elo_on_toy_dataset():
    """Two teams, three matches — sanity check the update rule."""
    m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-01"]).date,
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "home_score": [3, 0, 2],
            "away_score": [0, 2, 1],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, True, True],
        }
    )
    state = compute_elo(m)
    # A won 3-0, lost 0-2, won 2-1 → A should be clearly ahead
    assert state.get("A") > state.get("B")
    # Neither team should be at the default any more
    assert state.get("A") != DEFAULT_RATING
    assert state.get("B") != DEFAULT_RATING
