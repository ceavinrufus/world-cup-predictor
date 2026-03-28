from datetime import date

import numpy as np
import pandas as pd

from wcp.model.dixon_coles import (
    fit_dixon_coles,
    outcome_probs,
    score_matrix,
    _tau,
)


def _toy_matches(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthesise matches from a known ground-truth model.

    Ground truth: 4 teams with attack strengths [+0.3, +0.1, -0.1, -0.3]
    and identical defence.  Home advantage γ=0.25.  Random pairings.
    """
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    attack = {"A": 0.3, "B": 0.1, "C": -0.1, "D": -0.3}
    defence = {t: 0.0 for t in teams}
    ha = 0.25

    rows = []
    base = date(2020, 1, 1)
    for k in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = np.exp(attack[h] - defence[a] + ha) + 0.6
        mu = np.exp(attack[a] - defence[h]) + 0.6
        rows.append(
            {
                "date": base.replace(year=2020 + k // 200),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(lam)),
                "away_score": int(rng.poisson(mu)),
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def test_tau_pointwise():
    x = np.array([0, 0, 1, 1, 2])
    y = np.array([0, 1, 0, 1, 2])
    lam = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    t = _tau(x, y, lam, mu, rho=0.1)
    assert abs(t[0] - 0.9) < 1e-9
    assert abs(t[1] - 1.1) < 1e-9
    assert abs(t[2] - 1.1) < 1e-9
    assert abs(t[3] - 0.9) < 1e-9
    assert abs(t[4] - 1.0) < 1e-9  # untouched outside 2x2 block


def test_fit_recovers_ordering_from_synthetic():
    """Optimizer should recover the correct team strength ordering."""
    m = _toy_matches(n=800, seed=1)
    p = fit_dixon_coles(m, min_matches_per_team=1, max_iter=2000)
    idx = p.team_index()
    # Attack ranking should be A > B > C > D
    ranked = sorted(p.teams, key=lambda t: -p.attack[idx[t]])
    assert ranked == ["A", "B", "C", "D"]
    # Home advantage should be roughly positive
    assert p.home_advantage > 0.0


def test_score_matrix_rows_sum_to_one():
    m = _toy_matches(seed=2)
    p = fit_dixon_coles(m, min_matches_per_team=1)
    mat = score_matrix(p, "A", "B", neutral=True)
    assert abs(mat.sum() - 1.0) < 1e-9


def test_outcome_probs_sum_to_one():
    m = _toy_matches(seed=3)
    p = fit_dixon_coles(m, min_matches_per_team=1)
    ph, pdw, pa = outcome_probs(p, "A", "D", neutral=True)
    assert abs(ph + pdw + pa - 1.0) < 1e-9
    # A is strictly stronger than D → home should be favoured
    assert ph > pa
