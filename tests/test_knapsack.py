import random

from conftest import brute_force_knapsack
from core.knapsack_dp import knapsack_dp
from core.models import Item


def test_classic(classic_items):
    res = knapsack_dp(classic_items, 50)
    assert res.value == 220
    assert sorted(res.chosen) == ["b", "c"]
    assert res.doses_used == 50


def test_zero_capacity(classic_items):
    res = knapsack_dp(classic_items, 0)
    assert res.value == 0 and res.chosen == []


def test_matches_brute_force_random():
    rng = random.Random(1)
    for _ in range(30):
        items = [Item(str(i), rng.randint(1, 15), rng.randint(1, 100)) for i in range(rng.randint(1, 10))]
        cap = rng.randint(0, 50)
        res = knapsack_dp(items, cap)
        assert res.value == brute_force_knapsack(items, cap)
        w = {i.id: i.weight for i in items}
        assert sum(w[c] for c in res.chosen) <= cap
