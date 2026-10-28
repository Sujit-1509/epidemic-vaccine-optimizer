import random

from core.backtracking import backtracking_allocate
from core.knapsack_dp import knapsack_dp
from core.models import Item, items_from_network


def test_equals_dp_without_constraints():
    rng = random.Random(3)
    for _ in range(30):
        items = [Item(str(i), rng.randint(1, 15), rng.randint(1, 100)) for i in range(rng.randint(1, 12))]
        cap = rng.randint(0, 60)
        assert backtracking_allocate(items, cap).value == knapsack_dp(items, cap).value


def test_must_include_respected(classic_items):
    res = backtracking_allocate(classic_items, 50, must_include={"a"})
    assert "a" in res.chosen
    assert res.value == 180          # a + c (40 doses) beats a + b (30 doses)


def test_max_areas(classic_items):
    res = backtracking_allocate(classic_items, 60, max_areas=1)
    assert res.chosen == ["c"] and res.value == 120


def test_infeasible(classic_items):
    res = backtracking_allocate(classic_items, 20, must_include={"c"})
    assert not res.feasible


def test_pruning_happens(maharashtra):
    items = items_from_network(maharashtra)
    res = backtracking_allocate(items, 600)
    assert res.stats["nodes_explored"] < 2 ** (len(items) + 1)
    assert res.stats["pruned_bound"] > 0


def test_high_risk_constraint(small_city):
    items = items_from_network(small_city)
    must = {a.id for a in small_city.demand_areas() if a.risk >= 0.7}
    res = backtracking_allocate(items, 40, must_include=must)
    assert res.feasible and must <= set(res.chosen) and res.doses_used <= 40
