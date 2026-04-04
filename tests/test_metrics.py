import numpy as np

from wcp.model.metrics import (
    brier_score,
    calibration_curve,
    log_loss,
    outcome_class,
)


def test_log_loss_perfect():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    # log(1) = 0 but we clip; check it's ~0
    assert log_loss(y, p) < 1e-10


def test_log_loss_uniform():
    y = np.array([0, 1, 2])
    p = np.full((3, 3), 1 / 3)
    assert abs(log_loss(y, p) - np.log(3)) < 1e-6


def test_brier_uniform_three_class():
    y = np.array([0, 1, 2])
    p = np.full((3, 3), 1 / 3)
    # Per-sample brier = (2/3)^2 + (1/3)^2 + (1/3)^2 = 6/9
    assert abs(brier_score(y, p) - 6 / 9) < 1e-9


def test_outcome_class():
    assert outcome_class(2, 1) == 0
    assert outcome_class(1, 1) == 1
    assert outcome_class(0, 3) == 2


def test_calibration_curve_perfect_model():
    """A perfectly-calibrated confident model on binary problem."""
    y = np.array([0, 0, 1, 1] * 25)
    # For class 0 predict 1.0 when true, 0.0 otherwise; class 1 mirror.
    p = np.zeros((len(y), 2))
    p[y == 0] = [1.0, 0.0]
    p[y == 1] = [0.0, 1.0]
    bins = calibration_curve(y, p, n_bins=5)
    # All mass in the 0-0.2 and 0.8-1.0 bins; freq should match pred
    for b in bins:
        if b.n > 0:
            assert abs(b.mean_pred - b.freq) < 1e-9
