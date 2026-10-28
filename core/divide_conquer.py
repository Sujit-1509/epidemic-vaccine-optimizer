"""Divide and Conquer.

1. merge_sort       - rank areas by priority. T(n) = 2T(n/2) + O(n) = O(n log n)
2. regional_split   - split the region into west/east halves recursively and
                      combine demand totals and the highest-risk area of each half.
                      T(n) = 2T(n/2) + O(1) after an initial O(n log n) sort.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from .models import Area, Network

T = TypeVar("T")


def merge_sort(items: list[T], key: Callable[[T], float] = lambda x: x,
               reverse: bool = False) -> tuple[list[T], int]:
    """Return (sorted copy, number of comparisons)."""
    comparisons = 0

    def before(a: T, b: T) -> bool:
        return key(a) >= key(b) if reverse else key(a) <= key(b)

    def sort(lst: list[T]) -> list[T]:
        nonlocal comparisons
        if len(lst) <= 1:                      # base case
            return lst
        mid = len(lst) // 2                    # divide
        left, right = sort(lst[:mid]), sort(lst[mid:])   # conquer
        merged, i, j = [], 0, 0                # combine
        while i < len(left) and j < len(right):
            comparisons += 1
            if before(left[i], right[j]):
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    return sort(list(items)), comparisons


def rank_areas(net: Network) -> tuple[list[Area], int]:
    """Demand areas ranked by priority (population x risk), highest first."""
    return merge_sort(net.demand_areas(), key=lambda a: a.priority, reverse=True)


def regional_split(areas: list[Area], max_depth: int = 3) -> dict:
    """Recursively divide areas by longitude (x) and combine regional statistics.

    Returns a tree: {label, areas, total_doses, total_population, riskiest, children}
    """
    ordered, _ = merge_sort(areas, key=lambda a: a.x)

    def solve(lst: list[Area], depth: int, label: str) -> dict:
        if len(lst) == 1 or depth == max_depth:
            # base case: compute directly
            return {"label": label, "areas": [a.id for a in lst],
                    "total_doses": sum(a.doses_required for a in lst),
                    "total_population": sum(a.population for a in lst),
                    "riskiest": max(lst, key=lambda a: a.risk).id, "children": []}
        mid = len(lst) // 2
        left = solve(lst[:mid], depth + 1, label + "-W")
        right = solve(lst[mid:], depth + 1, label + "-E")
        # combine step: O(1)
        rl, rr = left["riskiest"], right["riskiest"]
        by_id = {a.id: a for a in lst}
        return {"label": label, "areas": left["areas"] + right["areas"],
                "total_doses": left["total_doses"] + right["total_doses"],
                "total_population": left["total_population"] + right["total_population"],
                "riskiest": rl if by_id[rl].risk >= by_id[rr].risk else rr,
                "children": [left, right]}

    if not ordered:
        return {"label": "Region", "areas": [], "total_doses": 0,
                "total_population": 0, "riskiest": None, "children": []}
    return solve(ordered, 0, "Region")


def flatten_tree(node: dict, depth: int = 0) -> list[dict]:
    """Tree -> list of rows (for tables / charts)."""
    rows = [{"depth": depth, **{k: v for k, v in node.items() if k != "children"}}]
    for c in node["children"]:
        rows.extend(flatten_tree(c, depth + 1))
    return rows
