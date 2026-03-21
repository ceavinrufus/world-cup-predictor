"""World Football Elo rating computation.

We compute Elo from scratch over the martj42 match dataset rather than
scraping eloratings.net.  Advantages:

* **Reproducible** — same inputs always yield the same ratings.
* **Cutoff-safe** — the calculation is bounded by ``TRAINING_CUTOFF``
  because it consumes :func:`wcp.data.results.load_results`, which is
  itself cutoff-aware.
* **No brittle scrapes.**

Formula (World Football Elo, https://www.eloratings.net/about):

    R'      = R + K · G · (W - We)

    W       = 1 win / 0.5 draw / 0 loss

    We      = 1 / (10^(-dr / 400) + 1)          # expected result
    dr      = R_home - R_away + 100·(home_flag) # 100-pt home bonus

    K       depends on match importance
    G       goal-difference multiplier

K weights (World Football Elo convention):

    World Cup finals            60
    Continental championship    50
    WC qualifier / continental
      qualifier                 40
    All other tournaments       30
    Friendly                    20

Goal-difference multiplier:

    G = 1                          if |gd| ≤ 1
    G = 1.5                        if |gd| == 2
    G = (11 + |gd|) / 8            if |gd| ≥ 3
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from wcp.config import PROCESSED_DIR, TRAINING_CUTOFF
from wcp.data.results import load_results
from wcp.data.teams import canonical

log = logging.getLogger(__name__)

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0

_K_BY_TOURNAMENT: dict[str, float] = {
    "FIFA World Cup": 60.0,
    "Confederations Cup": 50.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "Copa America": 50.0,
    "African Cup of Nations": 50.0,
    "Africa Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "CONCACAF Championship": 50.0,
    "Gold Cup": 50.0,
    "OFC Nations Cup": 50.0,
    "UEFA Nations League": 40.0,
    "CONCACAF Nations League": 40.0,
    "Friendly": 20.0,
}


def _k_factor(tournament: str) -> float:
    """K by importance — qualifiers and misc tournaments default to 30/40."""
    if tournament in _K_BY_TOURNAMENT:
        return _K_BY_TOURNAMENT[tournament]
    t = tournament.lower()
    if "qualification" in t or "qualifier" in t or "qualifying" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _g_factor(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _expected(r_home: float, r_away: float, neutral: bool) -> float:
    dr = r_home - r_away
    if not neutral:
        dr += HOME_ADVANTAGE
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


@dataclass(slots=True)
class EloState:
    """Container so we can snapshot ratings at arbitrary dates."""

    ratings: dict[str, float]
    last_update: dict[str, date]

    @classmethod
    def blank(cls) -> "EloState":
        return cls(ratings={}, last_update={})

    def get(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_RATING)

    def as_series(self) -> pd.Series:
        return pd.Series(self.ratings, name="elo").sort_values(ascending=False)


def compute_elo(
    matches: pd.DataFrame | None = None,
    *,
    up_to: date | None = None,
) -> EloState:
    """Roll Elo forward through ``matches`` (chronologically).

    Parameters
    ----------
    matches:
        Optional pre-loaded dataframe.  If omitted, we pull from
        :func:`wcp.data.results.load_results` with the cutoff applied.
    up_to:
        Only include matches on or before this date.  Defaults to
        :data:`wcp.config.TRAINING_CUTOFF`.
    """
    if matches is None:
        matches = load_results(apply_cutoff=True)

    limit = up_to or TRAINING_CUTOFF
    matches = matches[matches["date"] <= limit].sort_values("date")

    state = EloState.blank()

    for row in matches.itertuples(index=False):
        home = canonical(row.home_team)
        away = canonical(row.away_team)
        neutral = bool(row.neutral) if not pd.isna(row.neutral) else False

        r_h = state.get(home)
        r_a = state.get(away)

        we_h = _expected(r_h, r_a, neutral)
        we_a = 1.0 - we_h

        gd = int(row.home_score) - int(row.away_score)
        if gd > 0:
            w_h, w_a = 1.0, 0.0
        elif gd < 0:
            w_h, w_a = 0.0, 1.0
        else:
            w_h = w_a = 0.5

        k = _k_factor(row.tournament)
        g = _g_factor(gd)

        state.ratings[home] = r_h + k * g * (w_h - we_h)
        state.ratings[away] = r_a + k * g * (w_a - we_a)
        state.last_update[home] = row.date
        state.last_update[away] = row.date

    return state


def elo_table(state: EloState, min_matches: int = 0) -> pd.DataFrame:
    """Return a sorted table of Elo ratings."""
    df = pd.DataFrame(
        {
            "team": list(state.ratings.keys()),
            "elo": list(state.ratings.values()),
            "last_match": [state.last_update.get(t) for t in state.ratings],
        }
    )
    return df.sort_values("elo", ascending=False).reset_index(drop=True)


def save_elo_table(state: EloState, path: str | None = None) -> str:
    """Persist an Elo snapshot to parquet."""
    df = elo_table(state)
    out = path or str(PROCESSED_DIR / "elo_ratings.parquet")
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    st = compute_elo()
    tbl = elo_table(st)
    print(f"teams rated: {len(tbl):,}")
    print("top 20:")
    print(tbl.head(20).to_string(index=False))
    save_elo_table(st)
    print(f"saved → {PROCESSED_DIR / 'elo_ratings.parquet'}")
