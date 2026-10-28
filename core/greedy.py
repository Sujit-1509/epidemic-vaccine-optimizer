"""Greedy vaccine allocation.

Rule: always serve the area with the highest priority per dose first.
  * 0/1 greedy  - an area is either fully vaccinated or skipped. Fast, NOT always optimal.
  * fractional  - an area may be partly vaccinated. Provably optimal for that relaxed problem.
Time: O(n log n) for the sort, O(n) for the scan.
"""
from __future__ import annotations

from .models import AllocationResult, Item


def _by_ratio(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda it: it.value / it.weight, reverse=True)


def greedy_01(items: list[Item], capacity: int) -> AllocationResult:
    remaining = capacity
    chosen, value, steps = [], 0.0, 0
    for it in _by_ratio(items):
        steps += 1
        if it.weight <= remaining:
            chosen.append(it.id)
            remaining -= it.weight
            value += it.value
    return AllocationResult(
        method="Greedy (0/1)", chosen=chosen, doses_used=capacity - remaining,
        value=value, fractions={c: 1.0 for c in chosen}, stats={"items_scanned": steps})


def greedy_fractional(items: list[Item], capacity: int) -> AllocationResult:
    remaining = float(capacity)
    fractions: dict[str, float] = {}
    value = 0.0
    for it in _by_ratio(items):
        if remaining <= 0:
            break
        take = min(1.0, remaining / it.weight)
        fractions[it.id] = take
        remaining -= take * it.weight
        value += take * it.value
    return AllocationResult(
        method="Greedy (fractional)", chosen=list(fractions), doses_used=capacity - remaining,
        value=value, fractions=fractions,
        stats={"partial_areas": sum(1 for f in fractions.values() if f < 1.0)})
