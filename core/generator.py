"""Seeded random network generator, used for benchmarks, tests and the large dataset."""
from __future__ import annotations

import math
import random

from .models import Area, Network, Road


def random_network(n: int, seed: int = 42, extra_edge_ratio: float = 0.6,
                   n_warehouses: int = 1) -> Network:
    """Build a connected network of n areas.

    A random spanning tree guarantees connectivity, then extra roads are added
    between nearby areas so the graph looks like a road map.
    """
    rng = random.Random(seed)
    areas: dict[str, Area] = {}
    for i in range(n):
        aid = f"A{i}"
        areas[aid] = Area(
            id=aid,
            name=f"Area {i}",
            population=rng.randint(5_000, 200_000),
            risk=round(rng.uniform(0.05, 0.95), 2),
            doses_required=rng.randint(2, 30),
            x=round(rng.uniform(0, 100), 1),
            y=round(rng.uniform(0, 100), 1),
        )

    ids = list(areas)
    roads: list[Road] = []
    seen: set[tuple[str, str]] = set()

    def add(u: str, v: str) -> None:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return
        seen.add(key)
        a, b = areas[u], areas[v]
        dist = round(math.hypot(a.x - b.x, a.y - b.y), 1)
        roads.append(Road(u, v, dist, rng.randint(5, 40)))

    # random spanning tree
    for i in range(1, n):
        add(ids[i], ids[rng.randrange(i)])
    # extra roads to the nearest neighbours
    for _ in range(int(n * extra_edge_ratio)):
        u = rng.choice(ids)
        nearest = sorted(ids, key=lambda v: math.hypot(areas[u].x - areas[v].x,
                                                       areas[u].y - areas[v].y))[1:4]
        if nearest:
            add(u, rng.choice(nearest))

    warehouses = ids[:max(0, min(n_warehouses, n))]
    for w in warehouses:
        areas[w].name = f"Warehouse {w}"
    return Network(areas, roads, warehouses, name=f"Random network (n={n})")
