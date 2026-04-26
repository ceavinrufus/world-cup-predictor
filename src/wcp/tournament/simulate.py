"""Full tournament simulator.

Given a fitted Dixon-Coles model + the WC 2026 group draw, run Monte
Carlo simulations of the entire tournament.  Each simulation:

1. Simulates all group matches by sampling from the joint DC score
   distribution.  All group matches are treated as neutral-venue for
   this baseline model; a host bump can be added in a later phase.
2. Ranks each group per FIFA tiebreakers → identifies winners /
   runners-up / third-placed teams.
3. Ranks the 12 third-placed teams cross-group; takes the top 8.
4. Looks up the R32 pairings via Annex C.
5. Plays R32 → R16 → QF → SF → Final, using single-elimination.
   Draws in knockouts trigger a penalty-shootout coin-flip (with a
   small skill-weighting toward the stronger team).
6. Records final standings.

Aggregating over many simulations yields calibrated probabilities.
"""
from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from wcp.config import DEFAULT_N_SIMULATIONS, RANDOM_SEED
from wcp.model.dixon_coles import DixonColesParams, score_matrix as _dc_score_matrix
from wcp.tournament.bracket import R32_SLOTS, resolve_r32_pairs
from wcp.tournament.draw import GROUPS, POSITIONS
from wcp.tournament.standings import GroupResult, TeamRecord, rank_group

log = logging.getLogger(__name__)


# ── Prediction backend protocol ─────────────────────────────────────
#
# Everything the simulator needs is a way to get P(X=x, Y=y) for a
# match.  Both raw DixonColesParams and the blended variant satisfy
# this — we just dispatch through _get_matrix.

def _get_matrix(params, home: str, away: str, *, neutral: bool, max_goals: int):
    """Return a (max_goals+1)² score matrix from either DC or blended params."""
    fn = getattr(params, "score_matrix", None)
    if fn is not None:
        return fn(home, away, neutral=neutral, max_goals=max_goals)
    return _dc_score_matrix(
        params, home, away, neutral=neutral, max_goals=max_goals
    )


# ── Match sampling ──────────────────────────────────────────────────

def _sample_score(
    params,
    home: str,
    away: str,
    rng: np.random.Generator,
    *,
    neutral: bool = True,
    max_goals: int = 10,
) -> tuple[int, int]:
    """Sample a score (x, y) from the model's joint distribution."""
    matrix = _get_matrix(params, home, away, neutral=neutral, max_goals=max_goals)
    flat = matrix.ravel()
    idx = rng.choice(flat.size, p=flat)
    x, y = divmod(int(idx), max_goals + 1)
    return x, y


def _sample_knockout(
    params,
    home: str,
    away: str,
    rng: np.random.Generator,
    *,
    neutral: bool = True,
    max_goals: int = 10,
    shootout_bias: float = 0.06,
) -> str:
    """Play a knockout match.  Returns the winner's name.

    A tie in 90 minutes goes to a coin-flip biased slightly by the
    model's home-win vs away-win probability ratio — captures the
    intuition that the stronger team is a *slight* favourite in
    shootouts (empirically ~52/48 rather than 50/50 when there's a
    clear quality gap).
    """
    x, y = _sample_score(params, home, away, rng, neutral=neutral, max_goals=max_goals)
    if x > y:
        return home
    if y > x:
        return away
    # Tied — extra time then shootout.  Nudge the coin toward whichever
    # side has higher pre-match win probability.
    matrix = _get_matrix(params, home, away, neutral=neutral, max_goals=max_goals)
    p_home = float(np.tril(matrix, -1).sum())
    p_away = float(np.triu(matrix, 1).sum())
    total = p_home + p_away
    if total <= 0.0:
        return home if rng.random() < 0.5 else away
    p_h_shootout = 0.5 + shootout_bias * (p_home - p_away) / total
    return home if rng.random() < p_h_shootout else away


# ── Group stage ─────────────────────────────────────────────────────

_GROUP_FIXTURES: list[tuple[int, int]] = [
    (0, 1), (2, 3),   # matchday 1
    (0, 2), (1, 3),   # matchday 2
    (0, 3), (1, 2),   # matchday 3
]


def _simulate_group(
    letter: str,
    teams: list[str],
    params,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> GroupResult:
    records = {t: TeamRecord(team=t) for t in teams}
    matches: list[tuple[str, str, int, int]] = []
    for i, j in _GROUP_FIXTURES:
        home, away = teams[i], teams[j]
        x, y = _sample_score(params, home, away, np_rng, neutral=True)
        records[home].apply(x, y)
        records[away].apply(y, x)
        matches.append((home, away, x, y))
    standings = rank_group(records, matches, py_rng)
    return GroupResult(group=letter, standings=standings, matches=matches)


# ── Third-place ranking across groups ───────────────────────────────

def _rank_third_place(
    all_groups: dict[str, GroupResult],
    py_rng: random.Random,
) -> list[tuple[str, TeamRecord]]:
    """Return ordered (group_letter, third_place_record) — best first."""
    thirds = [(g, gr.standings[2]) for g, gr in all_groups.items()]
    thirds.sort(
        key=lambda x: (
            -x[1].points,
            -x[1].gd,
            -x[1].gf,
            py_rng.random(),  # ties broken by lots
        )
    )
    return thirds


# ── Knockout tree ───────────────────────────────────────────────────

def _play_bracket(
    r32_teams: list[tuple[str, str]],   # 16 pairs
    params,
    np_rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Play the entire knockout tree.  Returns a dict of stage → advancers.

    ``r32_teams`` is a list of 16 (team_a, team_b) tuples in R32 match order.
    """
    stages: dict[str, list[str]] = {}

    def play_round(pairs: list[tuple[str, str]]) -> list[str]:
        return [
            _sample_knockout(params, a, b, np_rng, neutral=True)
            for a, b in pairs
        ]

    r16_teams = play_round(r32_teams)
    stages["r32_winners"] = r16_teams

    # R16 pairs: consecutive (winner of Mi, winner of M(i+1)) — the
    # bracket is a straight elimination tree.
    r16_pairs = list(zip(r16_teams[0::2], r16_teams[1::2]))
    qf_teams = play_round(r16_pairs)
    stages["r16_winners"] = qf_teams

    qf_pairs = list(zip(qf_teams[0::2], qf_teams[1::2]))
    sf_teams = play_round(qf_pairs)
    stages["qf_winners"] = sf_teams

    sf_pairs = list(zip(sf_teams[0::2], sf_teams[1::2]))
    finalists = play_round(sf_pairs)
    stages["sf_winners"] = finalists

    champion = _sample_knockout(params, finalists[0], finalists[1], np_rng, neutral=True)
    stages["champion"] = [champion]
    return stages


# ── Full-tournament simulation ──────────────────────────────────────

@dataclass
class SimulationTally:
    """Aggregated Monte Carlo outputs across N simulations."""

    n_sims: int = 0
    group_position: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    stage_reached: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    _STAGES = (
        "group_stage",
        "round_of_32",
        "round_of_16",
        "quarter_finals",
        "semi_finals",
        "final",
        "champion",
    )

    def record(
        self,
        group_results: dict[str, GroupResult],
        advancing_thirds: list[str],
        bracket: dict[str, list[str]],
    ) -> None:
        self.n_sims += 1

        # Group positions
        for gr in group_results.values():
            for pos, rec in enumerate(gr.standings, start=1):
                self.group_position[rec.team][pos] += 1

        # Everyone reaches group stage
        for gr in group_results.values():
            for rec in gr.standings:
                self.stage_reached[rec.team]["group_stage"] += 1

        # R32 = top-2 from each group + 8 advancing thirds
        r32_teams = [
            gr.standings[0].team for gr in group_results.values()
        ] + [
            gr.standings[1].team for gr in group_results.values()
        ] + list(advancing_thirds)
        for t in r32_teams:
            self.stage_reached[t]["round_of_32"] += 1

        for t in bracket["r32_winners"]:
            self.stage_reached[t]["round_of_16"] += 1
        for t in bracket["r16_winners"]:
            self.stage_reached[t]["quarter_finals"] += 1
        for t in bracket["qf_winners"]:
            self.stage_reached[t]["semi_finals"] += 1
        for t in bracket["sf_winners"]:
            self.stage_reached[t]["final"] += 1
        for t in bracket["champion"]:
            self.stage_reached[t]["champion"] += 1

    def probabilities(self) -> pd.DataFrame:
        rows = []
        for team, counter in sorted(self.stage_reached.items()):
            row = {"team": team}
            for stage in self._STAGES:
                row[f"p_{stage}"] = counter.get(stage, 0) / self.n_sims
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True)
        )

    def group_probabilities(self) -> pd.DataFrame:
        rows = []
        for g, teams in GROUPS.items():
            for team in teams:
                cnt = self.group_position.get(team, Counter())
                rows.append(
                    {
                        "group": g,
                        "team": team,
                        "p_first": cnt.get(1, 0) / self.n_sims,
                        "p_second": cnt.get(2, 0) / self.n_sims,
                        "p_third": cnt.get(3, 0) / self.n_sims,
                        "p_fourth": cnt.get(4, 0) / self.n_sims,
                        "p_advance": (
                            (cnt.get(1, 0) + cnt.get(2, 0)) / self.n_sims
                        ),
                    }
                )
        return pd.DataFrame(rows)


def _resolve_position_codes(
    r32_pairs: list[tuple[str, str]],
    group_results: dict[str, GroupResult],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Convert position codes ('1A', '3E', '2B') to team names.

    Returns (list of (team_a, team_b) pairs, list of advancing-third teams).
    """
    resolved: list[tuple[str, str]] = []
    third_teams: list[str] = []
    for a_code, b_code in r32_pairs:
        team_a = _pos_to_team(a_code, group_results)
        team_b = _pos_to_team(b_code, group_results)
        if b_code.startswith("3"):
            third_teams.append(team_b)
        if a_code.startswith("3"):  # shouldn't happen per R32_SLOTS but safe
            third_teams.append(team_a)
        resolved.append((team_a, team_b))
    return resolved, third_teams


def _pos_to_team(code: str, group_results: dict[str, GroupResult]) -> str:
    """'1A' → group A winner, '2C' → group C runner-up, etc."""
    pos = int(code[0])
    letter = code[1]
    return group_results[letter].standings[pos - 1].team


def simulate_tournament(
    params,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    *,
    seed: int = RANDOM_SEED,
    progress: bool = False,
) -> SimulationTally:
    """Run ``n_sims`` full-tournament Monte Carlo simulations.

    ``params`` can be either a :class:`DixonColesParams` or a
    :class:`~wcp.model.blend.BlendedParams` — the simulator uses
    duck-typing on ``.score_matrix(home, away, neutral, max_goals)``.
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    tally = SimulationTally()

    iterator: Iterator[int] = range(n_sims)  # type: ignore[assignment]
    if progress:
        from tqdm import tqdm  # lazy import
        iterator = tqdm(iterator, desc="simulating", total=n_sims)

    for _ in iterator:
        # 1. Groups
        gres: dict[str, GroupResult] = {}
        for letter, teams in GROUPS.items():
            gres[letter] = _simulate_group(letter, teams, params, np_rng, py_rng)

        # 2. Rank third-placed teams — take top 8
        thirds_ranked = _rank_third_place(gres, py_rng)
        top_thirds = thirds_ranked[:8]
        qualifying_third_groups = {g for g, _ in top_thirds}

        # 3. Build R32 pairs from Annex C
        r32_slot_pairs = resolve_r32_pairs(qualifying_third_groups)
        r32_teams, advancing_thirds = _resolve_position_codes(r32_slot_pairs, gres)

        # 4. Play knockouts
        bracket = _play_bracket(r32_teams, params, np_rng)

        tally.record(gres, advancing_thirds, bracket)

    return tally
