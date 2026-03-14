"""Fetch and cache raw international match results.

Source: `martj42/international_results` on GitHub — a canonical, weekly-updated
CSV of every men's international since 1872.  Free, permissive, and ubiquitous
in academic work.

We download once and re-use.  All filtering (tournament cutoff, competition
scope, etc.) happens downstream so the raw cache stays a faithful mirror.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from wcp.config import RAW_DIR, TRAINING_CUTOFF

log = logging.getLogger(__name__)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/shootouts.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)

RESULTS_CSV: Path = RAW_DIR / "results.csv"
SHOOTOUTS_CSV: Path = RAW_DIR / "shootouts.csv"
GOALSCORERS_CSV: Path = RAW_DIR / "goalscorers.csv"


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.debug("cache hit: %s", dest.name)
        return dest
    log.info("fetching %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def fetch_all(force: bool = False) -> None:
    """Download all three CSVs into ``data/raw/``."""
    _download(RESULTS_URL, RESULTS_CSV, force=force)
    _download(SHOOTOUTS_URL, SHOOTOUTS_CSV, force=force)
    _download(GOALSCORERS_URL, GOALSCORERS_CSV, force=force)


def load_results(apply_cutoff: bool = True) -> pd.DataFrame:
    """Return the full results dataframe.

    Parameters
    ----------
    apply_cutoff:
        If True (default), drop matches on or after
        :data:`wcp.config.TRAINING_CUTOFF`.  This is the safety net that keeps
        the World Cup tournament itself out of the training set — nothing
        the model touches should be able to see it.
    """
    if not RESULTS_CSV.exists():
        fetch_all()

    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Drop rows with missing scores (future/unplayed matches).
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if apply_cutoff:
        before = len(df)
        df = df[df["date"] <= TRAINING_CUTOFF].copy()
        log.info(
            "cutoff %s: %d → %d matches (%d dropped)",
            TRAINING_CUTOFF,
            before,
            len(df),
            before - len(df),
        )

    return df.reset_index(drop=True)


def load_shootouts() -> pd.DataFrame:
    if not SHOOTOUTS_CSV.exists():
        fetch_all()
    df = pd.read_csv(SHOOTOUTS_CSV, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[df["date"] <= TRAINING_CUTOFF].reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_all(force=True)
    r = load_results()
    print(f"results: {len(r):,} rows | {r['date'].min()} → {r['date'].max()}")
    s = load_shootouts()
    print(f"shootouts: {len(s):,} rows")
