# world-cup-predictor

Pre-tournament probability forecasts for the FIFA World Cup 2026, built
on a Dixon–Coles bivariate Poisson model blended with a World Football
Elo prior, then Monte Carlo–simulated end-to-end.

Everything the model uses is dated on or before **10 June 2026** — the
day before the tournament opener in Mexico City. Nothing from the
tournament itself informs the predictions.

## Methodology

### Model

Match scores are modelled as a bivariate Poisson à la Dixon & Coles
(1997). For a match between home team *i* and away team *j*:

    X ~ Poisson(λ),   λ = exp(α_i − β_j + γ)         # home advantage γ
    Y ~ Poisson(μ),   μ = exp(α_j − β_i)

The joint PMF is corrected on the low-score block (0-0, 0-1, 1-0, 1-1)
with a scalar ρ to fix the Poisson model's chronic under-prediction of
draws.

Attack (α) and defence (β) parameters are fit by maximum likelihood with
a **time-weighted** likelihood — each observed match is weighted
`w(t) = 2^(-Δdays / half_life)`. That way old friendlies exist in the
gradient but don't dominate it.

The fitted DC model gets blended with a **World Football Elo prior**
computed from the same match history (all matches back to 1872, standard
K-factor and G-factor by tournament type). The final match probability
is a linear pool

    P_blend = (1 − λ) · P_DC + λ · P_Elo

with `λ` and the DC half-life tuned on backtest (see below).

### Tournament simulator

Given a fitted model, each simulation runs the full 2026 format:

1. All 72 group matches sampled from the joint score distribution.
2. FIFA tiebreakers applied cascading through points → GD → GF →
   H2H points → H2H GD → H2H GF → drawing of lots.
3. Twelve group winners, twelve runners-up, and the best eight of the
   twelve third-placed teams advance to the Round of 32.
4. R32 pairings are looked up in **Annex C** of the tournament
   regulations (all 495 possible combinations encoded).
5. R32 → R16 → QF → SF → Final via single elimination. Knockout draws
   resolve via a coin flip biased ~52/48 by the model's pre-match
   probability, standing in for an explicit shootout model.

Aggregating over 100k simulations gives calibrated stage-of-elimination
probabilities per team.

## Data

* [`martj42/international_results`](https://github.com/martj42/international_results)
  — every men's international since 1872. ~49k matches through cutoff.
* Elo ratings computed in-repo from the same match history — no external
  scraping, always reproducible.

The training cutoff (`wcp.config.TRAINING_CUTOFF = 2026-06-10`) is the
single source of truth. Every data loader routes through it.

## Backtest

Fits are re-run per training window (no peeking): fit on data prior to
each tournament opener, predict on that WC's matches, average log-loss
across WC 2018 and WC 2022. Random baseline: log(3) ≈ 1.099.

_(Backtest table added in a follow-up commit.)_

## Layout

    src/wcp/
      config.py            Cutoff, hyper-params, host countries
      data/
        results.py         Match ingestion (martj42)
        elo.py             World Football Elo replay
        teams.py           Name normalization
      model/
        dixon_coles.py     Dixon-Coles MLE fit + score matrix
        blend.py           DC + Elo linear pool
        metrics.py         Log-loss, Brier, calibration
        backtest.py        Rolling backtest harness
        tune.py            Grid search over lambda / half-life
      tournament/
        draw.py            Group draw
        standings.py       FIFA tiebreakers
        bracket.py         Annex C + R32 slot resolution
        simulate.py        Monte Carlo engine
      predict.py           CLI: fit + simulate → JSON

## Usage

    pip install -r requirements.txt

    # Fit the blended model and run 100k simulations.
    PYTHONPATH=src python -m wcp.predict

    # Faster smoke run.
    PYTHONPATH=src python -m wcp.predict --n 5000

    # Sweep hyper-params against historical WCs.
    PYTHONPATH=src python -m wcp.model.tune

Outputs land in `outputs/`:

* `predictions_summary.json` — stage-of-elimination probabilities per team
* `group_standings.json` — per-group finish-position probabilities
* `match_predictions.json` — 1×2 for every group-stage fixture
* `elo_snapshot.json` — Elo ratings at cutoff

## Caveats

* No public xG for internationals → the model sees only goals. Fine for
  aggregate strength but weak for one-off matchups where one side had
  many big chances but the score didn't reflect it.
* Squad injuries / lineup availability are unmodelled. If a key player
  is out for the tournament, the model doesn't know.
* Manager tournament experience, travel legs, altitude and climate
  effects are unmodelled. These are small effects that we'll layer in
  if the base model's calibration holds.
* Elo ratings absorb long-run form but move slowly — a team that's had
  a genuine step-change in the last six months is under-rated.

## References

* Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football
  Scores and Inefficiencies in the Football Betting Market.* Applied
  Statistics 46(2), 265-280.
* World Football Elo Ratings, https://www.eloratings.net/about
* FIFA World Cup 2026 Regulations (May 2025 revision) — group format,
  tiebreakers, and the 495-row Annex C.
