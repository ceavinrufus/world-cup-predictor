"""Model blending: combine Dixon-Coles predictions with an Elo-only baseline.

The idea: DC is expressive but fits only what its training window can
see, and it's noisy for teams with few recent matches.  Elo integrates
match evidence back to 1872 with the standard World Football decay so
long-run strength survives.  A linear pool of the two gets us the best
of both — the Poisson bites when it has real signal, the Elo damps out
when it doesn't.

    P_blend(outcome) = (1 - λ) · P_DC(outcome) + λ · P_Elo(outcome)

The Elo probability is derived from the rating difference via the
standard logistic:

    P(home) = 1 / (10^(-Δ/400) + 1),    Δ = R_home - R_away

We convert this two-way probability into a three-way (H, D, A) by
allocating a chunk to Draw.  Empirically for internationals, draws
happen ~24% of the time, and are slightly more likely for even matchups.
We use a simple Rue-Salvesen style model:

    P_draw = P_draw_base · (1 - |P_home_2way - 0.5|)

Then rescale H and A proportionally so H + D + A = 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wcp.data.elo import EloState, DEFAULT_RATING, HOME_ADVANTAGE
from wcp.data.teams import canonical
from wcp.model.dixon_coles import DixonColesParams, outcome_probs

log = logging.getLogger(__name__)


def elo_outcome_probs(
    elo: EloState,
    home: str,
    away: str,
    *,
    neutral: bool = False,
    draw_base: float = 0.28,
) -> tuple[float, float, float]:
    """Return (P_home, P_draw, P_away) from a pure-Elo model."""
    r_h = elo.get(canonical(home))
    r_a = elo.get(canonical(away))
    dr = r_h - r_a + (0.0 if neutral else HOME_ADVANTAGE)

    p_home_2way = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
    p_draw = draw_base * (1.0 - 2.0 * abs(p_home_2way - 0.5))
    p_home = (1.0 - p_draw) * p_home_2way
    p_away = (1.0 - p_draw) * (1.0 - p_home_2way)
    return p_home, p_draw, p_away


@dataclass(slots=True)
class BlendedParams:
    """A blended prediction wrapper.  Not a fitted model itself — just
    routing between DC and Elo.
    """

    dc: DixonColesParams
    elo: EloState
    lam: float = 0.35   # weight on Elo (0 = pure DC, 1 = pure Elo)

    def outcome_probs(
        self, home: str, away: str, *, neutral: bool = False
    ) -> tuple[float, float, float]:
        ph_dc, pd_dc, pa_dc = outcome_probs(self.dc, home, away, neutral=neutral)
        ph_e, pd_e, pa_e = elo_outcome_probs(
            self.elo, home, away, neutral=neutral
        )
        ph = (1 - self.lam) * ph_dc + self.lam * ph_e
        pd = (1 - self.lam) * pd_dc + self.lam * pd_e
        pa = (1 - self.lam) * pa_dc + self.lam * pa_e
        return ph, pd, pa

    def score_matrix(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        max_goals: int = 10,
    ) -> np.ndarray:
        """Return a blended score matrix.

        Reshaping a three-way probability back to a goals grid is
        ambiguous, so we compute the DC matrix normally and only adjust
        its marginals to match the blended (H, D, A) probabilities.
        """
        from wcp.model.dixon_coles import score_matrix
        mat = score_matrix(self.dc, home, away, neutral=neutral, max_goals=max_goals)
        ph_dc = float(np.tril(mat, -1).sum())
        pd_dc = float(np.trace(mat))
        pa_dc = float(np.triu(mat, 1).sum())

        ph_b, pd_b, pa_b = self.outcome_probs(home, away, neutral=neutral)

        # Multiply each outcome region by the ratio of blended : DC prob
        # then renormalise.  Preserves the *shape* of goal distributions
        # within each outcome (home-win margins, draw scores, etc.).
        eps = 1e-12
        m = mat.copy()
        m[np.tril_indices_from(m, -1)] *= ph_b / max(ph_dc, eps)
        m[np.diag_indices_from(m)]     *= pd_b / max(pd_dc, eps)
        m[np.triu_indices_from(m, 1)]  *= pa_b / max(pa_dc, eps)
        return m / m.sum()

    def rates(
        self, home: str, away: str, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Return the DC's expected-goals rates unchanged.

        The simulator uses this only for logging in a few places;
        keeping the DC values is fine because the actual sampling uses
        our score matrix override.
        """
        return self.dc.rates(home, away, neutral=neutral)

    @property
    def teams(self) -> list[str]:
        return self.dc.teams
