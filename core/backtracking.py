"""Constrained vaccine allocation by Backtracking with Branch-and-Bound.

Explores the include/exclude decision tree for every area while enforcing:
  * total doses <= supply
  * every area in `must_include` is served (e.g. all areas with risk >= 0.7)
  * at most `max_areas` areas are served (e.g. limited vaccination teams)

Pruning:
  * feasibility - stop a branch as soon as a constraint is broken
  * bound       - stop if even the fractional-greedy upper bound of the
                  remaining areas cannot beat the best solution found so far
Worst case O(2^n); pruning usually cuts this down drastically.
"""
from __future__ import annotations

from .models import AllocationResult, Item


def backtracking_allocate(items: list[Item], capacity: int,
                          must_include: set[str] | None = None,
                          max_areas: int | None = None) -> AllocationResult:
    must = set(must_include or ())
    # best-ratio first makes the bound tight early on
    items = sorted(items, key=lambda it: it.value / it.weight, reverse=True)
    n = len(items)
    limit = max_areas if max_areas is not None else n
    # how many must-include areas remain from position i onward
    must_after = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        must_after[i] = must_after[i + 1] + (items[i].id in must)

    best = {"value": -1.0, "set": None}
    stats = {"nodes_explored": 0, "pruned_infeasible": 0, "pruned_bound": 0}

    def upper_bound(i: int, room: int, value: float) -> float:
        """Fractional knapsack over items[i:] - an optimistic estimate."""
        for it in items[i:]:
            if it.weight <= room:
                room -= it.weight
                value += it.value
            else:
                return value + it.value * room / it.weight
        return value

    chosen: list[str] = []

    def solve(i: int, room: int, value: float) -> None:
        stats["nodes_explored"] += 1
        if i == n:
            if value > best["value"]:
                best["value"], best["set"] = value, list(chosen)
            return
        if len(chosen) + must_after[i] > limit:     # cannot fit remaining must-include areas
            stats["pruned_infeasible"] += 1
            return
        if upper_bound(i, room, value) <= best["value"]:
            stats["pruned_bound"] += 1
            return
        it = items[i]
        # branch 1: include
        if it.weight <= room and len(chosen) < limit:
            chosen.append(it.id)
            solve(i + 1, room - it.weight, value + it.value)
            chosen.pop()
        elif it.id in must:
            stats["pruned_infeasible"] += 1
            return
        # branch 2: exclude (not allowed for must-include areas)
        if it.id not in must:
            solve(i + 1, room, value)

    solve(0, int(capacity), 0.0)

    if best["set"] is None:
        return AllocationResult("Backtracking (constrained)", [], 0, 0.0, feasible=False, stats=stats)
    weight = {it.id: it.weight for it in items}
    picked = sorted(best["set"])
    return AllocationResult(
        method="Backtracking (constrained)", chosen=picked,
        doses_used=sum(weight[c] for c in picked), value=best["value"],
        fractions={c: 1.0 for c in picked}, stats=stats)
