"""Round-of-32 bracket construction for the 48-team format.

The tricky bit of the 2026 format is that the top two teams from each of
the 12 groups (24) plus the 8 best third-placed teams (32 total) advance,
and the specific R32 matchups depend on *which* of the 12 groups' third-
placed teams qualify.  FIFA published 495 pre-set combinations
(C(12, 8) = 495) in Annex C of the tournament regulations; we ingested
the full table from Wikipedia's rendered template.

Given a simulated tournament we:

1. Collect the 12 group winners (1A, 1B, …) and runners-up (2A, 2B, …).
2. Rank all 12 third-placed teams by (points, GD, GF, GS) — same
   tiebreakers FIFA uses across groups.
3. Take the top 8 → look up Annex C by which groups they came from.
4. Build the R32 pairings per FIFA's fixed match numbers 73-88.

The bracket structure downstream (R16, QF, SF, F) is a fixed tree; the
Wikipedia knockout page confirms winners of the standard slots meet in
predictable positions.

The R32 pairings hard-coded here are FIFA's schedule for matches 73-88
(from the WC26 knockout stage page).  For matchups that involve a third
place, we substitute the actual third-placed team via the Annex C
mapping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

from wcp.config import PROCESSED_DIR

# ── FIFA R32 fixed matchups (matches 73-88) ─────────────────────────
# Each is (slot_a, slot_b).  Slots are either winner "1X", runner-up
# "2X", or "3?" — a third-place spot that gets resolved via Annex C.
# Order matches FIFA's schedule for matches 73 to 88 exactly.
R32_SLOTS: list[tuple[str, str]] = [
    ("2A", "2B"),     # M73
    ("1E", "3?"),     # M74  1E vs Third-place A/B/C/D/F
    ("1F", "2C"),     # M75
    ("1C", "2F"),     # M76
    ("1I", "3?"),     # M77  1I vs Third-place C/D/F/G/H
    ("2E", "2I"),     # M78
    ("1A", "3?"),     # M79  1A vs Third-place C/E/F/H/I
    ("1L", "3?"),     # M80  1L vs Third-place E/H/I/J/K
    ("1D", "3?"),     # M81  1D vs Third-place B/E/F/I/J
    ("1G", "3?"),     # M82  1G vs Third-place A/E/H/I/J
    ("2K", "2L"),     # M83
    ("1H", "2J"),     # M84
    ("1B", "3?"),     # M85  1B vs Third-place E/F/G/I/J
    ("1J", "2H"),     # M86
    ("1K", "3?"),     # M87  1K vs Third-place D/E/I/J/L
    ("2D", "2G"),     # M88
]

# Which R32 slots host a third-placed opponent, in Annex-C-slot order.
# Annex C returns a mapping like {"1A": "3E", ...} — we substitute the
# right-hand side of these slot pairs by looking up the LHS.
_THIRD_HOSTS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


# ── Annex C loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_annex_c() -> dict[frozenset[str], dict[str, str]]:
    """Return {frozenset('B','D','E','F','I','J','K','L'): {'1A': '3E', …}}."""
    path = PROCESSED_DIR / "third_place_annex_c.json"
    with open(path) as f:
        raw = json.load(f)
    out: dict[frozenset[str], dict[str, str]] = {}
    for _, entry in raw.items():
        key = frozenset(entry["qualifying"])
        out[key] = entry["opponents"]
    return out


def annex_c_lookup(qualifying_groups: Iterable[str]) -> dict[str, str]:
    """Return the winner-slot → third-place-slot mapping for a given
    set of 8 groups whose third-placed team qualified.

    Example: annex_c_lookup({"B","D","E","F","I","J","K","L"})
             → {"1A": "3E", "1B": "3J", "1D": "3B", …}
    """
    key = frozenset(qualifying_groups)
    if len(key) != 8:
        raise ValueError(
            f"Expected 8 distinct group letters, got {len(key)}: {key}"
        )
    table = _load_annex_c()
    if key not in table:
        raise KeyError(
            f"No Annex C entry for qualifying groups {sorted(key)} "
            "(should be impossible)"
        )
    return table[key]


# ── R32 bracket resolution ──────────────────────────────────────────

def resolve_r32_pairs(
    qualifying_third_groups: Iterable[str],
) -> list[tuple[str, str]]:
    """Return the 16 R32 pairings as pairs of position-code strings
    (e.g. ("1A", "3E")).

    Downstream code maps the position codes to actual teams using the
    group-standings from the simulated tournament.
    """
    mapping = annex_c_lookup(qualifying_third_groups)
    resolved: list[tuple[str, str]] = []
    for a, b in R32_SLOTS:
        if b == "3?":
            if a not in mapping:
                raise KeyError(
                    f"R32 host slot {a!r} not in Annex C mapping — "
                    "check that Annex C covers all R32 winner slots"
                )
            resolved.append((a, mapping[a]))
        else:
            resolved.append((a, b))
    return resolved


# ── R16 → SF bracket topology (fixed) ───────────────────────────────
# Straight elimination from R32: pair (M73, M74) → R16 M89, etc.
# We generate this generically at simulation time — the tree is just
# consecutive pairs.
