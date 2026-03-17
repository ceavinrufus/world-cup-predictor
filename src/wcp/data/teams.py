"""Team-name normalization.

Different data sources disagree wildly on national-team names:

* martj42:            "United States", "South Korea", "Republic of Ireland"
* eloratings.net:     "USA", "Korea South", "Ireland"
* FIFA official:      "USA", "Korea Republic", "Republic of Ireland"
* Transfermarkt:      "United States", "Korea, South", "Ireland"

We define a single canonical name per FIFA member and route everything through
:func:`canonical`.  Add aliases as they surface — the tests exercise the
common failure modes.

Canonical form matches the martj42 dataset (our primary corpus), so raw
loads pass through untouched.
"""
from __future__ import annotations

from typing import Iterable

# Canonical name → set of known aliases (lowercased).  Canonical name is
# the martj42 spelling.
_ALIASES: dict[str, set[str]] = {
    "United States": {"usa", "us", "u.s.", "u.s.a.", "united states of america"},
    "South Korea": {"korea south", "korea republic", "korea, south", "republic of korea", "kor"},
    "North Korea": {"korea north", "korea dpr", "dpr korea", "korea, north"},
    "Ivory Coast": {"cote d'ivoire", "côte d'ivoire", "cote divoire"},
    "Cape Verde": {"cabo verde", "cape verde islands"},
    "DR Congo": {"congo dr", "democratic republic of congo",
                  "democratic republic of the congo", "congo-kinshasa", "zaire"},
    "Republic of Ireland": {"ireland", "eire", "irl"},
    "Czech Republic": {"czechia"},
    "Bosnia and Herzegovina": {"bosnia", "bosnia-herzegovina", "bosnia & herzegovina"},
    "North Macedonia": {"macedonia", "fyr macedonia", "f.y.r. macedonia"},
    "Curaçao": {"curacao"},
    "São Tomé and Príncipe": {"sao tome and principe", "sao tome & principe"},
    "Saint Kitts and Nevis": {"st kitts and nevis", "st. kitts and nevis",
                               "saint kitts & nevis"},
    "Saint Vincent and the Grenadines": {"st vincent and the grenadines",
                                          "saint vincent & the grenadines"},
    "Saint Lucia": {"st lucia", "st. lucia"},
    "Trinidad and Tobago": {"trinidad & tobago"},
    "Antigua and Barbuda": {"antigua & barbuda"},
    "United Arab Emirates": {"uae", "u.a.e."},
    "Wales": set(),
    "Scotland": set(),
    "Northern Ireland": {"n. ireland", "n ireland"},
    "England": set(),
    "Iran": {"ir iran", "islamic republic of iran"},
    "Russia": {"russian federation", "ussr", "soviet union"},
    "Germany": {"west germany", "east germany", "german democratic republic",
                 "federal republic of germany", "fr germany"},
    "Serbia": {"yugoslavia", "serbia and montenegro", "fr yugoslavia"},
    "Timor-Leste": {"east timor"},
    "Eswatini": {"swaziland"},
    "Türkiye": {"turkey"},
    "Kyrgyzstan": {"kyrgyz republic"},
    "Palestine": {"palestinian territories", "state of palestine"},
    "Chinese Taipei": {"taiwan"},
    "China PR": {"china", "prc", "china, pr"},
    "Hong Kong": {"hong kong, china", "hong kong sar"},
    "Macau": {"macao"},
    "Vietnam": {"viet nam"},
    "Brunei": {"brunei darussalam"},
    "Myanmar": {"burma"},
}

# Build reverse lookup: alias-lower → canonical
_REVERSE: dict[str, str] = {}
for canon, aliases in _ALIASES.items():
    _REVERSE[canon.lower()] = canon
    for a in aliases:
        _REVERSE[a.lower()] = canon


def canonical(name: str) -> str:
    """Return the canonical martj42-style spelling of ``name``.

    Unknown names are returned unchanged (with whitespace stripped) so
    the function is safe to sprinkle everywhere — worst case, no-op.
    """
    if name is None:
        return name  # type: ignore[return-value]
    key = str(name).strip()
    return _REVERSE.get(key.lower(), key)


def canonical_many(names: Iterable[str]) -> list[str]:
    return [canonical(n) for n in names]
