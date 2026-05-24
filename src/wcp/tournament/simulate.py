"""Full tournament simulator.

Given a fitted Dixon-Coles model + the WC 2026 group draw, run Monte
Carlo simulations of the entire tournament.  Each simulation:

1. Simulates all group matches by sampling from the joint DC score
   distribution.  Host countries (USA, Mexico, Canada) get the model's
   home-advantage γ when they are the ``home_team`` slot in the fixture;
   every other match is played on a neutral venue.
2. Ranks each group per FIFA tiebreakers → identifies winners /
   runners-up / third-placed teams.
3. Ranks the 12 third-placed teams cross-group; takes the top 8.
4. Looks up the R32 pairings via Annex C.
5. Plays R32 → R16 → QF → SF → Final, using single-elimination.  Hosts
   still enjoy a *partial* home advantage — the schedule places host
   knockout matches in their own country when possible.
6. Records final standings AND per-match, per-round matchup counts so
   we can export the modal bracket (the most-probable path through the
   tournament).

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

from wcp.config import DEFAULT_N_SIMULATIONS, HOST_COUNTRIES, RANDOM_SEED
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


# ── Host-advantage helper ───────────────────────────────────────────

_HOSTS = frozenset(HOST_COUNTRIES)


def _is_host_home(home: str, away: str) -> bool:
    """True when ``home`` is a host and ``away`` isn't.

    A host-vs-host meeting resolves to neutral: they're both at home.
    """
    return home in _HOSTS and away not in _HOSTS


def _knockout_neutral(home: str, away: str) -> bool:
    """Neutral flag for a knockout match.

    Once the bracket starts, matches happen at pre-scheduled venues.
    FIFA loaded the R16/QF/SF/F stadia into the USA where hosts get a
    natural bump but visitors are in their own continent (Mexico plays
    the R16 in Mexico City, though — that's a real home game).

    Approximation: treat a knockout match as home for a host team
    (i.e. NOT neutral) with roughly half the effect a group match has.
    We implement that by using the half-γ score matrix returned by
    :func:`_get_matrix_partial_ha` below.
    """
    return not _is_host_home(home, away)


def _get_matrix_partial_ha(
    params,
    home: str,
    away: str,
    *,
    max_goals: int,
    ha_scale: float,
) -> np.ndarray:
    """Score matrix with home advantage scaled by ``ha_scale``.

    ha_scale=0 → neutral, ha_scale=1 → full home venue.
    We compute the two endpoints and linearly interpolate their
    outcome-marginal probabilities, keeping DC's goal-distribution
    shape.
    """
    neutral_mat = _get_matrix(params, home, away, neutral=True, max_goals=max_goals)
    if ha_scale <= 0.0:
        return neutral_mat
    home_mat = _get_matrix(params, home, away, neutral=False, max_goals=max_goals)
    if ha_scale >= 1.0:
        return home_mat
    mix = (1 - ha_scale) * neutral_mat + ha_scale * home_mat
    return mix / mix.sum()


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
    ha_scale: float = 0.5,
    max_goals: int = 10,
    shootout_bias: float = 0.06,
) -> str:
    """Play a knockout match.  Returns the winner's name.

    Applies host-advantage at ``ha_scale`` strength (default 0.5 = half
    of a full home-venue effect) when ``home`` is a host and ``away``
    isn't.  Everyone else plays neutral.
    """
    scale = ha_scale if _is_host_home(home, away) else 0.0
    matrix = _get_matrix_partial_ha(
        params, home, away, max_goals=max_goals, ha_scale=scale
    )
    flat = matrix.ravel()
    idx = rng.choice(flat.size, p=flat)
    x, y = divmod(int(idx), max_goals + 1)

    if x > y:
        return home
    if y > x:
        return away
    # Tied — nudge coin toward higher pre-match win prob.
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
        # Host countries get full γ when they're the home slot in their
        # group's fixture list.  Everything else is neutral.
        neutral = not _is_host_home(home, away)
        x, y = _sample_score(params, home, away, np_rng, neutral=neutral)
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
            _sample_knockout(params, a, b, np_rng)
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

    champion = _sample_knockout(params, finalists[0], finalists[1], np_rng)
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
    # For modal-bracket export: how often each team occupies each slot.
    slot_occupants: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Per-slot pairing counts — helpful for the modal bracket
    # (which two teams most often meet in QF2, say?).
    slot_pairings: dict[str, Counter[tuple[str, str]]] = field(
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
        r32_teams: list[tuple[str, str]],
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
        r32_advance = [
            gr.standings[0].team for gr in group_results.values()
        ] + [
            gr.standings[1].team for gr in group_results.values()
        ] + list(advancing_thirds)
        for t in r32_advance:
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

        # Slot occupancy for the modal bracket.
        # R32 slots numbered 1..16 (matches M73-M88).
        for i, (a, b) in enumerate(r32_teams, start=1):
            self.slot_pairings[f"R32_M{72 + i}"][(a, b)] += 1
            self.slot_occupants[f"R32_M{72 + i}_home"][a] += 1
            self.slot_occupants[f"R32_M{72 + i}_away"][b] += 1
        # R16 pairs
        r16_pairs = list(zip(bracket["r32_winners"][0::2], bracket["r32_winners"][1::2]))
        for i, (a, b) in enumerate(r16_pairs, start=1):
            self.slot_pairings[f"R16_M{i}"][(a, b)] += 1
        qf_pairs = list(zip(bracket["r16_winners"][0::2], bracket["r16_winners"][1::2]))
        for i, (a, b) in enumerate(qf_pairs, start=1):
            self.slot_pairings[f"QF_M{i}"][(a, b)] += 1
        sf_pairs = list(zip(bracket["qf_winners"][0::2], bracket["qf_winners"][1::2]))
        for i, (a, b) in enumerate(sf_pairs, start=1):
            self.slot_pairings[f"SF_M{i}"][(a, b)] += 1
        finalists = bracket["sf_winners"]
        self.slot_pairings["FINAL"][(finalists[0], finalists[1])] += 1

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

    def modal_bracket(self) -> dict[str, list[dict]]:
        """Return the *most-likely* bracket walk.

        For each slot, pick the modal pair — the pairing that came up
        most often across sims.  Note the picks aren't jointly modal —
        each round is picked independently, so the "modal R16 match"
        may not use the "modal R32 winners".  That's the standard way
        prediction sites present it.
        """
        def _stage(prefix: str, n_matches: int) -> list[dict]:
            out = []
            for i in range(1, n_matches + 1):
                key = f"{prefix}_M{i}"
                counter = self.slot_pairings.get(key, Counter())
                if not counter:
                    continue
                (a, b), count = counter.most_common(1)[0]
                out.append(
                    {
                        "match": key,
                        "home": a,
                        "away": b,
                        "p_this_pairing": count / self.n_sims,
                    }
                )
            return out

        r32 = []
        for i in range(1, 17):
            key = f"R32_M{72 + i}"
            counter = self.slot_pairings.get(key, Counter())
            if not counter:
                continue
            (a, b), count = counter.most_common(1)[0]
            r32.append(
                {
                    "match": key,
                    "home": a,
                    "away": b,
                    "p_this_pairing": count / self.n_sims,
                }
            )

        r16 = _stage("R16", 8)
        qf = _stage("QF", 4)
        sf = _stage("SF", 2)

        # Final
        final_pair_counter = self.slot_pairings.get("FINAL", Counter())
        final_entry = []
        if final_pair_counter:
            (a, b), count = final_pair_counter.most_common(1)[0]
            final_entry.append(
                {
                    "match": "FINAL",
                    "home": a,
                    "away": b,
                    "p_this_pairing": count / self.n_sims,
                }
            )

        return {
            "round_of_32": r32,
            "round_of_16": r16,
            "quarter_finals": qf,
            "semi_finals": sf,
            "final": final_entry,
        }


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

    Host countries (USA, Mexico, Canada) are given the model's home-
    advantage γ for group-stage matches where they are the home side,
    and half-γ in knockout matches against non-hosts.
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

        tally.record(gres, advancing_thirds, bracket, r32_teams)

    return tally
