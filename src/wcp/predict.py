"""Generate final World Cup 2026 predictions.

Fits the blended model on all pre-cutoff data, runs 100k Monte Carlo
simulations of the tournament, and writes:

* ``outputs/predictions_summary.json`` — champion / stage-reached
  probabilities per team.
* ``outputs/group_standings.json`` — per-group finishing probabilities.
* ``outputs/match_predictions.json`` — pre-tournament match 1×2 for
  every group-stage fixture we can compute (with team pairings from
  the draw's default order).

Usage:
    python -m wcp.predict            # defaults: 100k sims, best-known blend
    python -m wcp.predict --n 10000  # faster smoke run

Everything is deterministic given ``--seed``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from wcp.config import (
    DEFAULT_BLEND_LAMBDA,
    DEFAULT_N_SIMULATIONS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TIME_DECAY_HALF_LIFE_DAYS,
    TRAINING_CUTOFF,
)
from wcp.data.elo import compute_elo, elo_table
from wcp.data.results import load_results
from wcp.model.blend import BlendedParams
from wcp.model.dixon_coles import fit_dixon_coles
from wcp.tournament.draw import ALL_TEAMS, GROUPS
from wcp.tournament.simulate import simulate_tournament

log = logging.getLogger(__name__)


def _group_fixture_predictions(params: BlendedParams) -> list[dict]:
    """Compute pre-tournament 1×2 probabilities for every group match.

    The default schedule pairs positions as (1,2), (3,4), (1,3), (2,4),
    (1,4), (2,3) per group (matchdays 1-3).
    """
    fixtures = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    out = []
    for letter, teams in GROUPS.items():
        for matchday, (i, j) in enumerate(fixtures, start=1):
            home, away = teams[i], teams[j]
            ph, pd_, pa = params.outcome_probs(home, away, neutral=True)
            lam, mu = params.rates(home, away, neutral=True)
            out.append(
                {
                    "group": letter,
                    "matchday": matchday,
                    "home_team": home,
                    "away_team": away,
                    "lambda_home": round(lam, 3),
                    "lambda_away": round(mu, 3),
                    "p_home_win": round(ph, 4),
                    "p_draw": round(pd_, 4),
                    "p_away_win": round(pa, 4),
                }
            )
    return out


def _human_summary(probs: pd.DataFrame, top: int = 15) -> str:
    lines = ["\n=== Champion probability (top {top}) ===".format(top=top)]
    lines.append(
        probs[["team", "p_champion", "p_final", "p_semi_finals", "p_quarter_finals"]]
        .head(top)
        .to_string(index=False)
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N_SIMULATIONS,
                    help="Number of Monte Carlo simulations")
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_BLEND_LAMBDA,
                    help="Elo blend weight (0 = pure DC, 1 = pure Elo)")
    ap.add_argument("--half-life", type=float, default=TIME_DECAY_HALF_LIFE_DAYS,
                    help="Time-decay half-life in days for DC likelihood")
    ap.add_argument("--min-matches", type=int, default=8,
                    help="Min matches per team to include in DC fit")
    ap.add_argument("--lookback-years", type=int, default=6)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument(
        "--generated-at",
        default=None,
        help="ISO timestamp to record in output metadata (default: now)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    log.info("Loading match data (cutoff %s)", TRAINING_CUTOFF)
    all_matches = load_results(apply_cutoff=True)
    from datetime import timedelta
    train_start = TRAINING_CUTOFF - timedelta(days=int(365.25 * args.lookback_years))
    train = all_matches[all_matches["date"] > train_start]
    log.info("DC fit on %d matches (%d-year lookback)", len(train), args.lookback_years)

    t0 = time.time()
    dc = fit_dixon_coles(
        train,
        ref_date=TRAINING_CUTOFF,
        half_life_days=args.half_life,
        min_matches_per_team=args.min_matches,
    )
    log.info("DC fit in %.1fs (%d teams)", time.time() - t0, len(dc.teams))

    log.info("Computing Elo through cutoff")
    elo = compute_elo(all_matches, up_to=TRAINING_CUTOFF)

    missing = [t for t in ALL_TEAMS if t not in set(dc.teams)]
    if missing:
        log.warning("WC teams missing from DC fit: %s", missing)
        log.warning("Consider lowering --min-matches")

    predictor = BlendedParams(dc=dc, elo=elo, lam=args.lam)

    log.info("Running %d Monte Carlo simulations", args.n)
    t0 = time.time()
    tally = simulate_tournament(predictor, n_sims=args.n, seed=args.seed, progress=True)
    log.info("Sim done in %.1fs", time.time() - t0)

    probs = tally.probabilities()
    group_probs = tally.group_probabilities()
    matches = _group_fixture_predictions(predictor)

    meta = {
        "generated_at": args.generated_at or datetime.now().isoformat(timespec="seconds"),
        "training_cutoff": str(TRAINING_CUTOFF),
        "n_simulations": args.n,
        "seed": args.seed,
        "blend_lambda": args.lam,
        "half_life_days": args.half_life,
        "min_matches_per_team": args.min_matches,
        "lookback_years": args.lookback_years,
        "n_training_matches": int(len(train)),
        "n_teams_in_dc_fit": len(dc.teams),
    }

    out_summary = OUTPUT_DIR / "predictions_summary.json"
    out_groups = OUTPUT_DIR / "group_standings.json"
    out_matches = OUTPUT_DIR / "match_predictions.json"
    out_elo = OUTPUT_DIR / "elo_snapshot.json"

    with open(out_summary, "w") as f:
        json.dump(
            {
                "meta": meta,
                "team_probabilities": probs.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    with open(out_groups, "w") as f:
        json.dump(
            {"meta": meta, "groups": group_probs.to_dict(orient="records")},
            f,
            indent=2,
        )
    with open(out_matches, "w") as f:
        json.dump({"meta": meta, "matches": matches}, f, indent=2)
    with open(out_elo, "w") as f:
        json.dump(
            {
                "meta": meta,
                "ratings": elo_table(elo).to_dict(orient="records"),
            },
            f,
            indent=2,
            default=str,
        )

    log.info("Saved:")
    log.info("  %s", out_summary)
    log.info("  %s", out_groups)
    log.info("  %s", out_matches)
    log.info("  %s", out_elo)

    print(_human_summary(probs, top=15))
    print("\n=== Group advancement (top 2) ===")
    # Show favourites per group
    for g in sorted(GROUPS):
        gs = group_probs[group_probs["group"] == g].sort_values(
            "p_advance", ascending=False
        )
        picks = gs.head(2)
        line = f"Group {g}:"
        for _, r in picks.iterrows():
            line += f"  {r['team']} ({r['p_advance']*100:5.1f}%)"
        print(line)


if __name__ == "__main__":
    main()
