"""Backtest: fit on data before some date, score on the held-out slice."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from wcp.data.teams import canonical
from wcp.model.dixon_coles import DixonColesParams, fit_dixon_coles, outcome_probs
from wcp.model.metrics import score_predictions

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    metrics: dict[str, float]


def _tournament_probs(
    params: DixonColesParams,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Score a batch of matches with a fitted model.

    Skips matches with a team the model has never seen (would raise).
    """
    known = set(params.teams)
    out_rows = []
    for row in matches.itertuples(index=False):
        h = canonical(row.home_team)
        a = canonical(row.away_team)
        if h not in known or a not in known:
            continue
        neutral = bool(row.neutral) if not pd.isna(row.neutral) else False
        ph, pd_, pa = outcome_probs(params, h, a, neutral=neutral)
        out_rows.append(
            {
                "date": row.date,
                "home_team": h,
                "away_team": a,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "neutral": neutral,
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
            }
        )
    return pd.DataFrame(out_rows)


def backtest_window(
    all_matches: pd.DataFrame,
    train_end: date,
    test_end: date,
    *,
    half_life_days: float = 180.0,
    min_matches_per_team: int = 10,
    train_lookback_years: int = 6,
) -> BacktestResult:
    """Fit on matches in (train_end - lookback, train_end], evaluate on
    (train_end, test_end].
    """
    train_start = train_end - timedelta(days=int(365.25 * train_lookback_years))

    train = all_matches[
        (all_matches["date"] > train_start) & (all_matches["date"] <= train_end)
    ]
    test = all_matches[
        (all_matches["date"] > train_end) & (all_matches["date"] <= test_end)
    ]

    log.info(
        "backtest %s → %s | train n=%d | test n=%d",
        train_end,
        test_end,
        len(train),
        len(test),
    )

    params = fit_dixon_coles(
        train,
        ref_date=train_end,
        half_life_days=half_life_days,
        min_matches_per_team=min_matches_per_team,
    )
    preds = _tournament_probs(params, test)
    metrics = score_predictions(preds)

    return BacktestResult(
        train_end=train_end,
        test_start=train_end + timedelta(days=1),
        test_end=test_end,
        n_train=len(train),
        n_test=metrics["n"],
        metrics=metrics,
    )


def rolling_backtest(
    all_matches: pd.DataFrame,
    windows: Iterable[tuple[date, date]],
    *,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Run backtest across multiple (train_end, test_end) tuples."""
    rows = []
    for train_end, test_end in tqdm(list(windows), desc="backtest"):
        r = backtest_window(
            all_matches,
            train_end,
            test_end,
            half_life_days=half_life_days,
        )
        rows.append(
            {
                "train_end": r.train_end,
                "test_end": r.test_end,
                "n_train": r.n_train,
                "n_test": r.n_test,
                **r.metrics,
            }
        )
    return pd.DataFrame(rows)
