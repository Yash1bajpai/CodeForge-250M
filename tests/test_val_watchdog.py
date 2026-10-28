import math
import pytest
from training.utils import val_trend_diverged

# Real step 166-185 validation losses from the 28 Oct 2026 extension run. The
# old rule (4 strictly rising points) stopped it at 185 although loss fell 0.17.
REAL = [6.2606, 6.2625, 6.2288, 6.2206, 6.2063, 6.1941, 6.1869, 6.1550, 6.1478, 6.1464,
        6.1429, 6.1275, 6.1306, 6.1139, 6.0857, 6.0945, 6.0613, 6.0621, 6.0847, 6.0858]


@pytest.mark.parametrize("half", [2, 5])
def test_real_noisy_but_falling_run_is_not_stopped(half):
    for n in range(1, len(REAL) + 1):
        assert not val_trend_diverged(REAL[:n], half=half)


@pytest.mark.parametrize("half, expected", [(5, 10), (2, 4)])
def test_clear_rise_is_stopped(half, expected):
    # +0.3 per five evaluations; test the actual first stop, not just an endpoint.
    history = [3.0 + .06 * i for i in range(30)]
    assert first_stop(history, half) == expected


def test_needs_enough_points():
    assert not val_trend_diverged([3.0, 3.5, 4.0, 4.5])


def test_small_rise_below_margin_is_ignored():
    assert not val_trend_diverged([3.0] * 5 + [3.04] * 5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", [0, 4, 5, 9])
def test_nonfinite_is_stopped_immediately_even_before_warmup(bad, position):
    history = [3.0] * 10
    history[position] = bad
    assert val_trend_diverged(history[:position + 1])
    assert val_trend_diverged(history)


def first_stop(history, half):
    return next((n for n in range(1, len(history) + 1)
                 if val_trend_diverged(history[:n], half=half)), None)


@pytest.mark.parametrize("half, expected", [(5, 18), (2, 15)])
def test_sustained_slow_rise_eventually_stops(half, expected):
    # Adjacent means differ by only .02/.008, forever below the .05 margin.
    history = [3.0 + .004 * i for i in range(200)]
    assert first_stop(history, half) == expected


@pytest.mark.parametrize("half, expected", [(5, 35), (2, 34)])
def test_slow_reversal_after_improvement_stops(half, expected):
    history = [4.0] * 10 + [3.0] * 10 + [3.0 + .004 * i for i in range(1, 101)]
    assert first_stop(history, half) == expected


@pytest.mark.parametrize("half", [2, 5])
def test_noisy_flat_series_never_stops(half):
    history = [3.0 + noise for noise in [0, .02, -.02, .01, -.01] * 80]
    assert first_stop(history, half) is None
