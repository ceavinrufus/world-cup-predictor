"""Group-stage tiebreakers per FIFA 2026 regulations.

The full 2026 regulation cascade for ranking teams within a group:

1. Points (win=3, draw=1, loss=0)
2. Goal difference across all group matches
3. Goals scored across all group matches
4. Points earned in head-to-head matches between the tied teams
5. Goal difference in head-to-head matches
6. Goals scored in head-to-head matches
7. Fair-play conduct points (yellow/red card penalties)
8. Drawing of lots by FIFA

We implement rules 1-6, then fall back to a stable random-lots tiebreaker
seeded per simulation for determinism.  Fair-play conduct isn't
observable at model time, so we skip step 7.  In our sims that difference
is invisible — lots vs fair-play both act as a small stochastic residual.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

# ── Group standings container ───────────────────────────────────────

@dataclass
class TeamRecord:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return 3 * self.won + self.drawn

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def apply(self, my_goals: int, opp_goals: int) -> None:
        self.played += 1
        self.gf += my_goals
        self.ga += opp_goals
        if my_goals > opp_goals:
            self.won += 1
        elif my_goals == opp_goals:
            self.drawn += 1
        else:
            self.lost += 1


@dataclass
class GroupResult:
    """Result of a simulated group — teams in final standings order."""

    group: str
    standings: list[TeamRecord]
    matches: list[tuple[str, str, int, int]] = field(default_factory=list)

    def ranked_teams(self) -> list[str]:
        return [r.team for r in self.standings]


# ── Ranking ─────────────────────────────────────────────────────────

def _head_to_head(
    teams: Iterable[str],
    matches: list[tuple[str, str, int, int]],
) -> dict[str, TeamRecord]:
    tset = set(teams)
    h2h = {t: TeamRecord(team=t) for t in tset}
    for home, away, hg, ag in matches:
        if home in tset and away in tset:
            h2h[home].apply(hg, ag)
            h2h[away].apply(ag, hg)
    return h2h


def rank_group(
    records: dict[str, TeamRecord],
    matches: list[tuple[str, str, int, int]],
    rng: random.Random,
) -> list[TeamRecord]:
    """Return group records sorted best-to-worst per FIFA tiebreakers."""
    # Primary sort: points → GD → GF
    teams = list(records.values())
    teams.sort(
        key=lambda r: (-r.points, -r.gd, -r.gf),
    )

    # Now resolve ties across the {points, GD, GF} equivalence classes.
    final: list[TeamRecord] = []
    i = 0
    while i < len(teams):
        j = i + 1
        while (
            j < len(teams)
            and teams[j].points == teams[i].points
            and teams[j].gd == teams[i].gd
            and teams[j].gf == teams[i].gf
        ):
            j += 1
        tied = teams[i:j]
        if len(tied) == 1:
            final.append(tied[0])
        else:
            # H2H mini-league
            h2h = _head_to_head([t.team for t in tied], matches)
            tied.sort(
                key=lambda r: (
                    -h2h[r.team].points,
                    -h2h[r.team].gd,
                    -h2h[r.team].gf,
                    rng.random(),  # 7/8 fair-play + lots → stochastic tiebreak
                ),
            )
            final.extend(tied)
        i = j
    return final
