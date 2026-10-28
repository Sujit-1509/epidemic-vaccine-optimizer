"""Performance comparison: time every algorithm on growing random inputs.

Run from the project folder:  python -m core.benchmark
"""
from __future__ import annotations

import pandas as pd

from .backtracking import backtracking_allocate
from .divide_conquer import merge_sort
from .generator import random_network
from .greedy import greedy_01, greedy_fractional
from .knapsack_dp import knapsack_dp
from .max_flow import vaccine_max_flow
from .models import items_from_network
from .np_complete import EXACT_LIMIT, approx_vertex_cover, edges_of, exact_vertex_cover
from .timing import timed
from .traversal import bfs_levels, dfs_components

COMPLEXITY = {
    "BFS": "O(V + E)",
    "DFS (components)": "O(V + E)",
    "Greedy 0/1": "O(n log n)",
    "Greedy fractional": "O(n log n)",
    "Knapsack DP": "O(n · W)",
    "Backtracking (B&B)": "O(2^n) worst case",
    "Max flow (Edmonds-Karp)": "O(V · E²)",
    "Merge sort (D&C)": "O(n log n)",
    "Vertex cover exact": "O(2^n · E)",
    "Vertex cover 2-approx": "O(V + E)",
}

DEFAULT_SIZES = [10, 20, 40, 80, 160]
BACKTRACK_LIMIT = 40      # backtracking gets slow beyond this many areas
EXACT_VC_LIMIT = 18       # brute force vertex cover beyond this takes too long for a demo


def _best_of(fn, *args, repeats: int = 3):
    best_ms, result = float("inf"), None
    for _ in range(repeats):
        result, ms = timed(fn, *args)
        best_ms = min(best_ms, ms)
    return result, best_ms


def run_benchmark(sizes: list[int] | None = None, supply_ratio: float = 0.4,
                  seed: int = 0, repeats: int = 3) -> pd.DataFrame:
    """Return a long-format table: n, algorithm, time_ms, value (solution quality if any)."""
    rows = []
    for n in sizes or DEFAULT_SIZES:
        net = random_network(n, seed=seed + n)
        items = items_from_network(net)
        supply = int(sum(i.weight for i in items) * supply_ratio)
        start = net.warehouses[0]

        def add(name, fn, *args, value=None):
            res, ms = _best_of(fn, *args, repeats=repeats)
            v = value(res) if value else None
            rows.append({"n": n, "algorithm": name, "time_ms": ms, "value": v})

        add("BFS", bfs_levels, net, start)
        add("DFS (components)", dfs_components, net)
        add("Greedy 0/1", greedy_01, items, supply, value=lambda r: r.value)
        add("Greedy fractional", greedy_fractional, items, supply, value=lambda r: r.value)
        add("Knapsack DP", knapsack_dp, items, supply, value=lambda r: r.value)
        if n <= BACKTRACK_LIMIT:
            add("Backtracking (B&B)", backtracking_allocate, items, supply, value=lambda r: r.value)
        add("Max flow (Edmonds-Karp)", vaccine_max_flow, net, value=lambda r: r["value"])
        add("Merge sort (D&C)", merge_sort, net.area_list(), lambda a: a.priority)
        edges = edges_of(net)
        if n <= min(EXACT_VC_LIMIT, EXACT_LIMIT):
            add("Vertex cover exact", exact_vertex_cover, list(net.areas), edges,
                value=lambda r: r["size"])
        add("Vertex cover 2-approx", approx_vertex_cover, edges, value=lambda r: r["size"])
    return pd.DataFrame(rows)


def vertex_cover_scaling(sizes: list[int] | None = None, seed: int = 0) -> pd.DataFrame:
    """Exact vs approximate vertex cover on small graphs - shows exponential blow-up."""
    rows = []
    for n in sizes or list(range(4, EXACT_VC_LIMIT + 1, 2)):
        net = random_network(n, seed=seed + n, extra_edge_ratio=0.3)
        edges = edges_of(net)
        ex, ms_ex = timed(exact_vertex_cover, list(net.areas), edges)
        ap, ms_ap = timed(approx_vertex_cover, edges)
        rows.append({"n": n, "exact_ms": ms_ex, "approx_ms": ms_ap,
                     "exact_size": ex["size"], "approx_size": ap["size"],
                     "subsets_checked": ex["subsets_checked"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_benchmark()
    pd.set_option("display.width", 160)
    print("Time (ms) by input size\n")
    print(df.pivot(index="algorithm", columns="n", values="time_ms").round(3))
    print("\nSolution value (allocation algorithms, same supply)\n")
    alloc = df[df.algorithm.isin(["Greedy 0/1", "Knapsack DP", "Backtracking (B&B)", "Greedy fractional"])]
    print(alloc.pivot(index="algorithm", columns="n", values="value").round(0))
    print("\nExact vs approximate vertex cover\n")
    print(vertex_cover_scaling().round(3).to_string(index=False))
