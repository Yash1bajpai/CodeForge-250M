import math
from training.utils import val_trend_diverged

# Real step 166-185 validation losses from the 28 Oct 2026 extension run. The
# old rule (4 strictly rising points) stopped it at 185 although loss fell 0.17.
REAL = [6.2606, 6.2625, 6.2288, 6.2206, 6.2063, 6.1941, 6.1869, 6.1550, 6.1478, 6.1464,
        6.1429, 6.1275, 6.1306, 6.1139, 6.0857, 6.0945, 6.0613, 6.0621, 6.0847, 6.0858]


def test_real_noisy_but_falling_run_is_not_stopped():
    for n in range(1, len(REAL) + 1):
        assert not val_trend_diverged(REAL[:n])


def test_clear_rise_is_stopped():
    assert val_trend_diverged([3.0] * 5 + [3.2] * 5)


def test_needs_enough_points():
    assert not val_trend_diverged([3.0, 3.5, 4.0, 4.5])


def test_small_rise_below_margin_is_ignored():
    assert not val_trend_diverged([3.0] * 5 + [3.04] * 5)


def test_nan_is_stopped():
    assert val_trend_diverged([3.0] * 9 + [math.nan])
