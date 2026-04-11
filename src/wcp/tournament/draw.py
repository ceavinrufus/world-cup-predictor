"""FIFA World Cup 2026 group draw.

Result of the 5 December 2025 draw at the Kennedy Center, Washington D.C.
Encoded manually — the source of truth is FIFA's published bracket.

Team names are in :func:`wcp.data.teams.canonical` form so they join
cleanly against the results dataset and the fitted model's team list.
"""
from __future__ import annotations

# Position (A1, A2, …) → team.  Positions matter because the knockout
# bracket is determined by them (A1 vs 2C etc.).
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# All 48 teams, in draw order.
ALL_TEAMS: list[str] = [t for teams in GROUPS.values() for t in teams]

# Position code (e.g. "A1") → team.
POSITIONS: dict[str, str] = {
    f"{g}{i + 1}": t for g, teams in GROUPS.items() for i, t in enumerate(teams)
}

# Host countries — used later for a small home-continent bonus.
HOSTS = ("United States", "Mexico", "Canada")


def team_group(team: str) -> str:
    """Return the group letter for ``team``.  Raises if not in the draw."""
    for g, teams in GROUPS.items():
        if team in teams:
            return g
    raise KeyError(team)
