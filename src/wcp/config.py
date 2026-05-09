"""Central configuration constants for the predictor.

Everything time-sensitive lives here so the training cutoff is enforced
in one place and can never accidentally leak into the model.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# ── Repository layout ───────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
MODELS_DIR: Path = ROOT_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── World Cup 2026 ──────────────────────────────────────────────────
# Opening match: Mexico vs (TBD) — 11 June 2026 at Estadio Azteca.
# Model MUST NOT see anything on or after this date.
TOURNAMENT_START: date = date(2026, 6, 11)
TOURNAMENT_END: date = date(2026, 7, 19)

# Data cutoff — inclusive upper bound on training data.
TRAINING_CUTOFF: date = date(2026, 6, 10)

# ── Model hyper-parameters ──────────────────────────────────────────
# Time-decay half-life for match weights (Dixon-Coles ξ).
# Tuned on WC 2018 + WC 2022 backtest (see outputs/blend_sweep.csv).
# 730 days = ~2 years.  Blended with Elo prior (λ=0.4), this beats
# pure-DC at both half-lives and Pinnacle-style random baseline.
TIME_DECAY_HALF_LIFE_DAYS: float = 730.0

# Default weight on Elo prior in the linear pool.
# 0 = pure Dixon-Coles, 1 = pure Elo.
DEFAULT_BLEND_LAMBDA: float = 0.4

# Monte Carlo simulation count for the full bracket.
DEFAULT_N_SIMULATIONS: int = 100_000

# Random seed for reproducibility.
RANDOM_SEED: int = 20260611

# Host countries — small home-continent adjustment applied in later phase.
HOST_COUNTRIES: tuple[str, ...] = ("United States", "Mexico", "Canada")
