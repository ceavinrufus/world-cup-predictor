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

    half_life  lambda   avg log-loss   WC2018    WC2022    n
       730       0.4      0.9872       0.9536    1.0208   152
       730       0.3      0.9873       0.9515    1.0231   152
       730       0.5      0.9881       0.9568    1.0195   152
       365       0.5      0.9906       0.9573    1.0239   152
       180       0.5      0.9921       —         —        152
        90       0.7      0.9925       —         —        152

Pure Dixon-Coles at hl=90 gets 1.0515; pure Elo gets 1.0108; the
blend beats both. Long half-life + moderate Elo weight (λ=0.4) wins,
which makes sense — Elo already captures short-term form through
K-factors, so DC's contribution is the goal-distribution shape and
home advantage.

Full sweep in `outputs/blend_sweep.csv`.

## Top-line results

25k Monte Carlo simulations, model frozen at 2026-06-10 (tournament
opener 2026-06-11 in Mexico City).  Host advantage is applied: full
γ for host home matches in the group stage, half-γ in knockouts.

**Champion probability, top 10:**

| Team       |    P |
|------------|-----:|
| Argentina  | 17.6% |
| Spain      | 15.6% |
| England    |  8.0% |
| France     |  7.8% |
| Brazil     |  6.7% |
| Portugal   |  5.3% |
| Colombia   |  4.9% |
| Germany    |  3.6% |
| Ecuador    |  3.4% |
| Morocco    |  3.0% |

**Group top-2 picks (probability to advance):**

| Group | 1st | 2nd |
|-------|-----|-----|
| A | Mexico (84%) | South Korea (53%) |
| B | Switzerland (89%) | Canada (87%) |
| C | Brazil (87%) | Morocco (75%) |
| D | Türkiye (56%) | United States (50%) |
| E | Germany (83%) | Ecuador (77%) |
| F | Netherlands (79%) | Japan (76%) |
| G | Belgium (82%) | Iran (60%) |
| H | Spain (97%) | Uruguay (79%) |
| I | France (83%) | Norway (63%) |
| J | Argentina (93%) | Austria (50%) |
| K | Portugal (82%) | Colombia (80%) |
| L | England (92%) | Croatia (78%) |

**Modal bracket.**  The single most-probable pairing in each slot,
picked independently per round.  These are not jointly the *most-
likely path* through the tournament — they're each round's most-
likely fixture given all upstream uncertainty.

Round of 32 (all 16 matches in `outputs/modal_bracket.json`), a few
of the most-locked-in:

- M75: Netherlands vs Morocco (19.9%)
- M76: Brazil vs Japan (20.9%)
- M84: Spain vs Austria (29.5%)
- M86: Argentina vs Uruguay (43.8%)

Round of 16:

- M4: Mexico vs England (15.9%)
- M7: Switzerland vs Argentina (17.0%)
- M2: Morocco vs Brazil (13.1%)

Quarter-finals:

- Germany vs Brazil (7.2%)
- France vs England (12.0%)
- Belgium vs Spain (9.6%)
- Argentina vs Portugal (10.2%)

Semi-finals:

- Brazil vs France (5.9%)
- Spain vs Argentina (11.0%)

**Final: France vs Argentina (3.8% for this exact pairing).**

Full outputs: `outputs/predictions_summary.json`,
`outputs/group_standings.json`, `outputs/match_predictions.json`,
`outputs/modal_bracket.json`.

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
* Host advantage is applied at the country level, not the venue level.
  Mexico's altitude in Mexico City vs sea-level Miami is unmodelled;
  every host home match gets the same γ.
* Manager tournament experience, travel legs, humidity effects are
  unmodelled. Small effects individually; we may layer some in if the
  base model's calibration holds.
* Elo ratings absorb long-run form but move slowly — a team that's had
  a genuine step-change in the last six months is under-rated.

## References

* Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football
  Scores and Inefficiencies in the Football Betting Market.* Applied
  Statistics 46(2), 265-280.
* World Football Elo Ratings, https://www.eloratings.net/about
* FIFA World Cup 2026 Regulations (May 2025 revision) — group format,
  tiebreakers, and the 495-row Annex C.
