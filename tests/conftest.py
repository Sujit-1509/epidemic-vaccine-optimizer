import sys
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.models import Area, Item, Network, Road  # noqa: E402


@pytest.fixture
def small_city():
    return Network.from_json(ROOT / "data" / "small_city.json")


@pytest.fixture
def maharashtra():
    return Network.from_json(ROOT / "data" / "maharashtra_districts.json")


@pytest.fixture
def line_graph():
    """S - A - B - C   plus isolated D."""
    areas = {i: Area(i, i, 1000, 0.5, 1) for i in "SABCD"}
    roads = [Road("S", "A", 1, 5), Road("A", "B", 1, 5), Road("B", "C", 1, 5)]
    return Network(areas, roads, ["S"])


def brute_force_knapsack(items, capacity):
    best = 0.0
    for k in range(len(items) + 1):
        for combo in combinations(items, k):
            if sum(i.weight for i in combo) <= capacity:
                best = max(best, sum(i.value for i in combo))
    return best


@pytest.fixture
def classic_items():
    # textbook example where 0/1 greedy is not optimal
    return [Item("a", 10, 60), Item("b", 20, 100), Item("c", 30, 120)]
