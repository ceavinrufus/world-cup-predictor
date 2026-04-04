"""Dixon-Coles bivariate Poisson model with time-weighted MLE.

Given a match between home team i and away team j, the number of goals
scored is:

    X ~ Poisson(λ),    Y ~ Poisson(μ)

with team-specific attack (α) and defence (β) parameters:

    log(λ) = α_i - β_j + γ          # γ = home advantage
    log(μ) = α_j - β_i

The joint PMF is the product with the Dixon-Coles low-score correction
τ(x, y; λ, μ, ρ) applied to (0,0), (0,1), (1,0), (1,1) — this fixes the
Poisson model's tendency to under-predict draws and 1-0 / 0-1 results.

    τ(0,0) = 1 - λ·μ·ρ
    τ(0,1) = 1 + λ·ρ
    τ(1,0) = 1 + μ·ρ
    τ(1,1) = 1 - ρ

The log-likelihood is time-weighted with an exponential decay so recent
matches count more:

    L = Σ_m  w(t_m) · [ log P(X=x_m, Y=y_m; θ) ]

with w(t) = exp(-ξ · Δdays).

We fit by minimising -L via scipy.optimize.minimize (L-BFGS-B).  The
attack params are constrained sum-to-zero to remove the additive
degeneracy in α (else α_i + c, β_i + c for all i is equivalent).

Reference: Dixon & Coles (1997), *Modelling Association Football Scores
and Inefficiencies in the Football Betting Market*, Applied Statistics
46(2), 265-280.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from wcp.config import TIME_DECAY_HALF_LIFE_DAYS, TRAINING_CUTOFF
from wcp.data.teams import canonical

log = logging.getLogger(__name__)


# ── Dixon-Coles low-score correction ────────────────────────────────

def _tau(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Correction factor τ(x, y) for low scores.  Vectorised."""
    out = np.ones_like(lam, dtype=np.float64)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


def _log_poisson_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log P(K=k | λ) for k a non-negative int array."""
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _decay_weights(
    match_dates: pd.Series, ref_date: date, half_life_days: float
) -> np.ndarray:
    """Exponential decay weights: w(t) = 2^(-Δdays / half_life)."""
    delta = np.array(
        [(ref_date - d).days for d in match_dates], dtype=np.float64
    )
    # Guard against negative deltas (dates > ref_date) — should not happen.
    delta = np.maximum(delta, 0.0)
    return np.power(2.0, -delta / half_life_days)


# ── Model container ─────────────────────────────────────────────────

@dataclass
class DixonColesParams:
    teams: list[str]
    attack: np.ndarray          # α_i
    defence: np.ndarray         # β_i   (higher = worse defence)
    home_advantage: float       # γ
    rho: float                  # DC low-score correction
    fitted_at: date = field(default_factory=lambda: TRAINING_CUTOFF)
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS

    def team_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def rates(
        self, home: str, away: str, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Return (λ, μ) — expected goals for home / away."""
        idx = self.team_index()
        i = idx.get(canonical(home))
        j = idx.get(canonical(away))
        if i is None or j is None:
            raise KeyError(
                f"Unknown team in fitted model: home={home!r} away={away!r}"
            )
        ha = 0.0 if neutral else self.home_advantage
        lam = math.exp(self.attack[i] - self.defence[j] + ha)
        mu = math.exp(self.attack[j] - self.defence[i])
        return lam, mu


# ── Fitting ─────────────────────────────────────────────────────────

def _pack(
    attack: np.ndarray, defence: np.ndarray, ha: float, rho: float
) -> np.ndarray:
    # Drop the last attack param — reconstructed via sum-to-zero.
    return np.concatenate([attack[:-1], defence, [ha, rho]])


def _unpack(
    theta: np.ndarray, n_teams: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    a_free = theta[: n_teams - 1]
    attack = np.concatenate([a_free, [-a_free.sum()]])   # sum-to-zero
    defence = theta[n_teams - 1 : 2 * n_teams - 1]
    ha = float(theta[-2])
    rho = float(theta[-1])
    return attack, defence, ha, rho


def _neg_log_likelihood(
    theta: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    neutral: np.ndarray,
    weights: np.ndarray,
) -> float:
    attack, defence, ha, rho = _unpack(theta, n_teams)

    log_lam = attack[home_idx] - defence[away_idx] + (~neutral) * ha
    log_mu = attack[away_idx] - defence[home_idx]

    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    log_p = _log_poisson_pmf(x, lam) + _log_poisson_pmf(y, mu)
    tau = _tau(x, y, lam, mu, rho)
    # Guard: τ can go non-positive for pathological ρ; keep it > eps.
    tau = np.where(tau <= 0.0, 1e-10, tau)
    log_p = log_p + np.log(tau)

    return float(-(weights * log_p).sum())


def fit_dixon_coles(
    matches: pd.DataFrame,
    *,
    ref_date: date | None = None,
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
    min_matches_per_team: int = 5,
    max_iter: int = 5000,
    verbose: bool = False,
) -> DixonColesParams:
    """Fit the Dixon-Coles model on ``matches``.

    Parameters
    ----------
    matches:
        DataFrame with columns date, home_team, away_team, home_score,
        away_score, neutral (bool-ish).
    ref_date:
        Anchor for time-decay weights.  Defaults to the training cutoff.
    half_life_days:
        Half-life of the exponential decay in days.
    min_matches_per_team:
        Teams with fewer appearances are dropped (they hurt convergence
        and their fits are noise anyway).
    """
    ref = ref_date or TRAINING_CUTOFF

    m = matches.copy()
    m["home_team"] = m["home_team"].map(canonical)
    m["away_team"] = m["away_team"].map(canonical)

    # Filter thin teams
    counts = pd.concat([m["home_team"], m["away_team"]]).value_counts()
    keep = set(counts[counts >= min_matches_per_team].index)
    m = m[m["home_team"].isin(keep) & m["away_team"].isin(keep)].copy()
    log.info(
        "fitting on %d matches | %d teams (min %d apps)",
        len(m), len(keep), min_matches_per_team,
    )

    teams = sorted(keep)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = m["home_team"].map(idx).to_numpy(dtype=np.int64)
    away_idx = m["away_team"].map(idx).to_numpy(dtype=np.int64)
    x = m["home_score"].to_numpy(dtype=np.int64)
    y = m["away_score"].to_numpy(dtype=np.int64)
    neutral = m["neutral"].fillna(False).astype(bool).to_numpy()

    weights = _decay_weights(m["date"], ref, half_life_days)
    weights = weights / weights.mean()  # normalise for interpretability

    # Initial guess: small random attack/defence, γ=0.3, ρ=-0.1
    rng = np.random.default_rng(42)
    attack0 = rng.normal(0.0, 0.05, size=n)
    defence0 = rng.normal(0.0, 0.05, size=n)
    theta0 = _pack(attack0, defence0, 0.3, -0.1)

    # Bound ρ to (-0.2, 0.2) — outside this τ frequently goes negative.
    bounds = (
        [(-3, 3)] * (n - 1)           # attack (free)
        + [(-3, 3)] * n               # defence
        + [(-0.5, 1.5)]               # home advantage γ
        + [(-0.2, 0.2)]               # rho
    )

    res = minimize(
        _neg_log_likelihood,
        theta0,
        args=(n, home_idx, away_idx, x, y, neutral, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "maxfun": max_iter * 100},
    )

    if not res.success:
        log.warning("optimizer did not fully converge: %s", res.message)

    attack, defence, ha, rho = _unpack(res.x, n)
    log.info(
        "fit done | γ=%.3f  ρ=%.3f  -logL=%.1f  iters=%s",
        ha, rho, res.fun, res.nit,
    )

    return DixonColesParams(
        teams=teams,
        attack=attack,
        defence=defence,
        home_advantage=ha,
        rho=rho,
        fitted_at=ref,
        half_life_days=half_life_days,
    )


# ── Match outcome probabilities ─────────────────────────────────────

def score_matrix(
    params: DixonColesParams,
    home: str,
    away: str,
    *,
    neutral: bool = False,
    max_goals: int = 10,
) -> np.ndarray:
    """P(X=x, Y=y) matrix up to ``max_goals``.  Renormalised."""
    lam, mu = params.rates(home, away, neutral=neutral)
    xs = np.arange(max_goals + 1)
    px = np.exp(_log_poisson_pmf(xs, np.full(max_goals + 1, lam)))
    py = np.exp(_log_poisson_pmf(xs, np.full(max_goals + 1, mu)))
    p = np.outer(px, py)  # rows = home goals, cols = away goals

    # DC correction on the 2×2 low-score block
    p[0, 0] *= 1.0 - lam * mu * params.rho
    p[0, 1] *= 1.0 + lam * params.rho
    p[1, 0] *= 1.0 + mu * params.rho
    p[1, 1] *= 1.0 - params.rho

    p = np.clip(p, 0.0, None)
    return p / p.sum()


def outcome_probs(
    params: DixonColesParams,
    home: str,
    away: str,
    *,
    neutral: bool = False,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win)."""
    p = score_matrix(params, home, away, neutral=neutral, max_goals=max_goals)
    return (
        float(np.tril(p, -1).sum()),
        float(np.trace(p)),
        float(np.triu(p, 1).sum()),
    )
