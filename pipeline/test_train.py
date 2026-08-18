import pytest

from train import compute_loss_weights


def test_skewed_distribution_gets_capped_at_max_ratio():
    counts = {
        "touchdown": {0: 9600, 1: 400},
        "yards_gained": {i: 1000 for i in range(7)},
    }
    weights = compute_loss_weights(counts, max_ratio=10.0)

    ratio = max(weights.values()) / min(weights.values())
    assert ratio == pytest.approx(10.0, rel=1e-6)


def test_uniform_distribution_gives_weights_near_one():
    counts = {
        "touchdown": {0: 500, 1: 500},
        "turnover": {0: 500, 1: 500},
    }
    weights = compute_loss_weights(counts, max_ratio=10.0)

    assert weights["touchdown"] == pytest.approx(1.0, abs=1e-6)
    assert weights["turnover"] == pytest.approx(1.0, abs=1e-6)


def test_weights_always_have_mean_one():
    counts = {
        "touchdown": {0: 9600, 1: 400},
        "yards_gained": {i: 1000 for i in range(7)},
        "turnover": {0: 9800, 1: 200},
    }
    weights = compute_loss_weights(counts, max_ratio=10.0)

    mean_weight = sum(weights.values()) / len(weights)
    assert mean_weight == pytest.approx(1.0, abs=1e-6)


def test_degenerate_distribution_raises_clear_error():
    counts = {
        "touchdown": {0: 100},
        "yards_gained": {i: 1000 for i in range(7)},
    }
    with pytest.raises(ValueError, match="touchdown"):
        compute_loss_weights(counts)