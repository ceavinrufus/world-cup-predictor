import numpy as np
import pandas as pd
from datetime import date

from wcp.data.elo import EloState, compute_elo
from wcp.model.blend import BlendedParams, elo_outcome_probs
from wcp.model.dixon_coles import fit_dixon_coles


def _toy_matches():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]).date,
            "home_team": ["Brazil", "Argentina", "Brazil"],
            "away_team": ["Argentina", "Brazil", "Argentina"],
            "home_score": [2, 1, 3],
            "away_score": [1, 2, 0],
            "tournament": ["Friendly"] * 3,
            "neutral": [True] * 3,
        }
    )


def test_elo_outcome_probs_sum_to_one():
    st = EloState.blank()
    st.ratings["Brazil"] = 2000.0
    st.ratings["Argentina"] = 1900.0
    ph, pd_, pa = elo_outcome_probs(st, "Brazil", "Argentina", neutral=True)
    assert abs(ph + pd_ + pa - 1.0) < 1e-9
    # Higher-rated team should be favoured
    assert ph > pa


def test_elo_equal_ratings_are_symmetric():
    st = EloState.blank()
    st.ratings["A"] = st.ratings["B"] = 1500.0
    ph, pd_, pa = elo_outcome_probs(st, "A", "B", neutral=True)
    assert abs(ph - pa) < 1e-9


def test_blended_probs_sum_to_one():
    m = _toy_matches()
    dc = fit_dixon_coles(m, min_matches_per_team=1)
    st = compute_elo(m)
    blend = BlendedParams(dc=dc, elo=st, lam=0.4)
    ph, pd_, pa = blend.outcome_probs("Brazil", "Argentina", neutral=True)
    assert abs(ph + pd_ + pa - 1.0) < 1e-9


def test_blended_score_matrix_sums_to_one():
    m = _toy_matches()
    dc = fit_dixon_coles(m, min_matches_per_team=1)
    st = compute_elo(m)
    blend = BlendedParams(dc=dc, elo=st, lam=0.4)
    mat = blend.score_matrix("Brazil", "Argentina", neutral=True)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_blend_lambda_endpoints():
    m = _toy_matches()
    dc = fit_dixon_coles(m, min_matches_per_team=1)
    st = compute_elo(m)
    # λ=0 → pure DC
    pure_dc = BlendedParams(dc=dc, elo=st, lam=0.0)
    dc_probs = pure_dc.outcome_probs("Brazil", "Argentina", neutral=True)
    # λ=1 → pure Elo
    pure_elo = BlendedParams(dc=dc, elo=st, lam=1.0)
    elo_probs = pure_elo.outcome_probs("Brazil", "Argentina", neutral=True)
    # They should differ
    assert dc_probs != elo_probs
