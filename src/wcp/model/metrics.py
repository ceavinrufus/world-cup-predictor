"""Probability calibration and skill metrics.

Two things I care about:

* **Log-loss** — the strictly proper scoring rule.  Lower = better.
  Guessing 1/3 for every 3-way match gives log(3) ≈ 1.099, so anything
  below that beats random.  Pinnacle closing lines hover around 0.95.

* **Brier score** — MSE against one-hot outcome.  ~0.19 = decent,
  <0.185 = strong.

* **Calibration curve** — bin predictions by decile, compare mean
  predicted probability vs empirical frequency.  A perfect model
  lies on y = x.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _clip(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-class log-loss.

    Parameters
    ----------
    y_true:
        Integer class indices, shape (N,).
    y_pred:
        Predicted probabilities, shape (N, K).
    """
    y_pred = _clip(np.asarray(y_pred, dtype=np.float64))
    idx = np.arange(len(y_true))
    return float(-np.log(y_pred[idx, y_true]).mean())


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-class Brier score (average squared distance to one-hot)."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n, k = y_pred.shape
    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(n), y_true] = 1.0
    return float(((y_pred - one_hot) ** 2).sum(axis=1).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred.argmax(axis=1) == y_true).mean())


@dataclass
class CalibrationBin:
    low: float
    high: float
    n: int
    mean_pred: float
    freq: float


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """One-vs-rest calibration across all classes flattened together.

    Every (match, class) pair contributes one (predicted_prob,
    was_actual_class ∈ {0, 1}) point.  Bins those and returns per-bin
    stats.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n, k = y_pred.shape

    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(n), y_true] = 1.0

    p = y_pred.reshape(-1)
    y = one_hot.reshape(-1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            bins.append(CalibrationBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        bins.append(
            CalibrationBin(
                low=float(lo),
                high=float(hi),
                n=int(mask.sum()),
                mean_pred=float(p[mask].mean()),
                freq=float(y[mask].mean()),
            )
        )
    return bins


def outcome_class(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def score_predictions(
    df: pd.DataFrame,
    prob_cols: tuple[str, str, str] = ("p_home", "p_draw", "p_away"),
) -> dict[str, float]:
    """Convenience: given a DataFrame with home_score/away_score and
    predicted (p_home, p_draw, p_away), return the summary metrics.
    """
    df = df.dropna(subset=list(prob_cols) + ["home_score", "away_score"]).copy()
    y_true = np.array(
        [outcome_class(int(h), int(a)) for h, a in zip(df["home_score"], df["away_score"])]
    )
    y_pred = df[list(prob_cols)].to_numpy(dtype=np.float64)
    return {
        "n": int(len(df)),
        "log_loss": log_loss(y_true, y_pred),
        "brier": brier_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }
