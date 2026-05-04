"""Grid-search the blend hyperparameters against historical WC data.

Two knobs:

* ``lam`` — weight on Elo in the linear pool (0 = pure DC, 1 = pure Elo).
* ``half_life_days`` — time-decay half-life for the DC likelihood.

We fit on data before each World Cup opener, evaluate on that WC's
matches, and average log-loss across WC 2018 and 2022 (2010 and 2014
predate a decent chunk of the dataset's tournament coding — skipped).
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from wcp.data.elo import compute_elo
from wcp.data.results import load_results
from wcp.data.teams import canonical
from wcp.model.blend import BlendedParams
from wcp.model.dixon_coles import fit_dixon_coles
from wcp.model.metrics import score_predictions

log = logging.getLogger(__name__)

# Each entry: label, cutoff (inclusive train end), test window
WORLD_CUPS: list[tuple[str, date, date, date]] = [
    ("WC2018", date(2018, 6, 13), date(2018, 6, 14), date(2018, 7, 15)),
    ("WC2022", date(2022, 11, 19), date(2022, 11, 20), date(2022, 12, 18)),
]

DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
DEFAULT_HALF_LIVES: tuple[float, ...] = (90.0, 180.0, 365.0, 730.0)


def _score(
    params_like,
    test: pd.DataFrame,
    known_teams: set[str],
) -> dict[str, float]:
    rows = []
    for row in test.itertuples(index=False):
        h = canonical(row.home_team)
        a = canonical(row.away_team)
        if h not in known_teams or a not in known_teams:
            continue
        neutral = bool(row.neutral) if not pd.isna(row.neutral) else False
        ph, pd_, pa = params_like.outcome_probs(h, a, neutral=neutral)
        rows.append(
            {
                "home_score": row.home_score,
                "away_score": row.away_score,
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
            }
        )
    if not rows:
        return {"n": 0, "log_loss": float("nan"), "brier": float("nan"), "accuracy": float("nan")}
    return score_predictions(pd.DataFrame(rows))


def _fit_one_wc(train_end: date, half_life: float, lookback_years: int = 8):
    """Return (dc_params, elo_state) fit only on pre-``train_end`` data."""
    m = load_results(apply_cutoff=False)
    train_start = train_end - timedelta(days=int(365.25 * lookback_years))
    train = m[(m["date"] > train_start) & (m["date"] <= train_end)]
    dc = fit_dixon_coles(
        train,
        ref_date=train_end,
        half_life_days=half_life,
        min_matches_per_team=8,
    )
    elo = compute_elo(m, up_to=train_end)
    return dc, elo


class _DCWrapper:
    """Match the .outcome_probs signature so pure-DC can go through the same loop."""

    def __init__(self, dc):
        self.dc = dc

    def outcome_probs(self, home, away, *, neutral=False):
        from wcp.model.dixon_coles import outcome_probs
        return outcome_probs(self.dc, home, away, neutral=neutral)


def sweep(
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
) -> pd.DataFrame:
    all_matches = load_results(apply_cutoff=False)
    rows = []
    for hl in half_lives:
        # Fit each WC once per half-life, then re-use for all lambdas.
        fits = {}
        for label, train_end, ts, te in WORLD_CUPS:
            log.info("fitting %s @ half_life=%.0f", label, hl)
            dc, elo = _fit_one_wc(train_end, hl)
            test = all_matches[
                (all_matches["date"] >= ts) & (all_matches["date"] <= te)
            ]
            fits[label] = (dc, elo, test)

        for lam in lambdas:
            per_wc = {}
            for label, (dc, elo, test) in fits.items():
                if lam <= 0.0:
                    predictor = _DCWrapper(dc)
                else:
                    predictor = BlendedParams(dc=dc, elo=elo, lam=lam)
                known = set(dc.teams)
                metrics = _score(predictor, test, known)
                per_wc[label] = metrics
            row = {
                "half_life_days": hl,
                "lambda": lam,
                "avg_log_loss": np.mean(
                    [m["log_loss"] for m in per_wc.values()]
                ),
                "avg_brier": np.mean(
                    [m["brier"] for m in per_wc.values()]
                ),
                "avg_accuracy": np.mean(
                    [m["accuracy"] for m in per_wc.values()]
                ),
                "n_total": sum(m["n"] for m in per_wc.values()),
            }
            for label, m in per_wc.items():
                row[f"{label}_log_loss"] = m["log_loss"]
                row[f"{label}_n"] = m["n"]
            rows.append(row)
            log.info(
                "hl=%.0f λ=%.2f | avg_log_loss=%.4f | avg_brier=%.4f",
                hl, lam, row["avg_log_loss"], row["avg_brier"],
            )
    return pd.DataFrame(rows).sort_values("avg_log_loss").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/blend_sweep.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = sweep()
    df.to_csv(args.output, index=False)
    print(f"\nsaved → {args.output}")
    print("\nTop 5 configs by avg_log_loss:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
