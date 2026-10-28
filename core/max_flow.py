"""Maximum vaccine flow with the Ford-Fulkerson method.

Augmenting paths are found with BFS (the Edmonds-Karp variant), giving
O(V * E^2) time regardless of capacity values.

Flow network built from the area graph:
    SUPER_SOURCE -> each warehouse      capacity = warehouse stock (or unlimited)
    road u - v (undirected)             capacity = road capacity in both directions
    each demand area -> SUPER_SINK      capacity = doses the area requires
The max flow value is the largest number of doses/day that can actually be
delivered. The min cut shows the bottleneck roads (Max-Flow Min-Cut theorem).
"""
from __future__ import annotations

from collections import defaultdict, deque

from .models import Network

SOURCE, SINK = "__SOURCE__", "__SINK__"


def edmonds_karp(capacity: dict[str, dict[str, float]], s: str, t: str) -> dict:
    """Generic max flow on a directed capacity map {u: {v: cap}}."""
    cap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for u, nbrs in capacity.items():
        for v, c in nbrs.items():
            cap[u][v] += c
            cap[v][u] += 0          # make sure the reverse residual edge exists
    flow: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total, paths = 0.0, []

    def residual(u: str, v: str) -> float:
        return cap[u][v] - flow[u][v]

    while True:
        parent = {s: None}
        queue = deque([s])
        while queue and t not in parent:            # BFS for shortest augmenting path
            u = queue.popleft()
            for v in cap[u]:
                if v not in parent and residual(u, v) > 1e-12:
                    parent[v] = u
                    queue.append(v)
        if t not in parent:
            break
        path, v = [], t
        while v is not None:
            path.append(v)
            v = parent[v]
        path.reverse()
        bottleneck = min(residual(path[k], path[k + 1]) for k in range(len(path) - 1))
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck                # skew symmetry
        total += bottleneck
        paths.append((path, bottleneck))

    # vertices still reachable in the residual graph form the source side of the min cut
    reach, stack = {s}, [s]
    while stack:
        u = stack.pop()
        for v in cap[u]:
            if v not in reach and residual(u, v) > 1e-12:
                reach.add(v)
                stack.append(v)
    cut = [(u, v) for u in reach for v, c in capacity.get(u, {}).items()
           if v not in reach and c > 0]
    return {"value": total, "flow": flow, "paths": paths, "source_side": reach, "cut": cut}


def vaccine_max_flow(net: Network, warehouse_stock: dict[str, float] | None = None) -> dict:
    capacity: dict[str, dict[str, float]] = defaultdict(dict)
    for w in net.warehouses:
        capacity[SOURCE][w] = (warehouse_stock or {}).get(w, float("inf"))
    for r in net.roads:
        capacity[r.u][r.v] = capacity[r.u].get(r.v, 0) + r.capacity
        capacity[r.v][r.u] = capacity[r.v].get(r.u, 0) + r.capacity
    for a in net.demand_areas():
        if a.doses_required > 0:
            capacity[a.id][SINK] = a.doses_required

    res = edmonds_karp(capacity, SOURCE, SINK)
    flow = res["flow"]

    road_flow = []
    for r in net.roads:
        f = flow[r.u][r.v]          # positive: u -> v, negative: v -> u
        if f >= 0:
            road_flow.append({"from": r.u, "to": r.v, "flow": f, "capacity": r.capacity})
        else:
            road_flow.append({"from": r.v, "to": r.u, "flow": -f, "capacity": r.capacity})

    delivered = {a.id: flow[a.id][SINK] for a in net.demand_areas() if a.doses_required > 0}
    unmet = {a: net.areas[a].doses_required - d for a, d in delivered.items()
             if net.areas[a].doses_required - d > 1e-9}
    cut_roads = [(u, v) for u, v in res["cut"] if SOURCE not in (u, v) and SINK not in (u, v)]
    total_demand = sum(a.doses_required for a in net.demand_areas())
    return {"value": res["value"], "total_demand": total_demand, "road_flow": road_flow,
            "delivered": delivered, "unmet": unmet, "cut_roads": cut_roads,
            "augmenting_paths": [([p for p in path if p not in (SOURCE, SINK)], b)
                                 for path, b in res["paths"]]}
