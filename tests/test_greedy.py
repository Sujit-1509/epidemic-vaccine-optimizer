import pytest

from core.greedy import greedy_01, greedy_fractional
from core.knapsack_dp import knapsack_dp
from core.models import items_from_network


def test_greedy_01_classic(classic_items):
    res = greedy_01(classic_items, 50)
    assert res.chosen == ["a", "b"]
    assert res.value == 160          # optimal is 220 -> greedy is not optimal


def test_fractional_is_optimal_relaxation(classic_items):
    res = greedy_fractional(classic_items, 50)
    assert res.value == pytest.approx(240)
    assert res.fractions["c"] == pytest.approx(2 / 3)
    assert res.doses_used == pytest.approx(50)


def test_greedy_never_beats_dp(small_city):
    items = items_from_network(small_city)
    for cap in range(0, 70, 5):
        g, d, f = greedy_01(items, cap), knapsack_dp(items, cap), greedy_fractional(items, cap)
        assert g.value <= d.value + 1e-9 <= f.value + 1e-6
        assert g.doses_used <= cap
