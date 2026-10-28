"""0/1 Knapsack by Dynamic Programming.

dp[i][w] = best total priority using the first i areas with at most w doses.

    dp[i][w] = dp[i-1][w]                                    (skip area i)
             = max(dp[i-1][w], dp[i-1][w - wt_i] + val_i)    (if wt_i <= w)

Time O(n * W), space O(n * W). This is *pseudo-polynomial*: it is polynomial
in the numeric value W, not in the number of bits needed to write W, which is
why it does not contradict 0/1 Knapsack being NP-complete.
"""
from __future__ import annotations

from .models import AllocationResult, Item


def knapsack_dp(items: list[Item], capacity: int, keep_table: bool = False) -> AllocationResult:
    n, W = len(items), int(capacity)
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wt, val = items[i - 1].weight, items[i - 1].value
        prev, row = dp[i - 1], dp[i]
        for w in range(W + 1):
            row[w] = prev[w]
            if wt <= w and prev[w - wt] + val > row[w]:
                row[w] = prev[w - wt] + val

    # walk back through the table to find which areas were chosen
    chosen, w = [], W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(items[i - 1].id)
            w -= items[i - 1].weight
    chosen.reverse()

    stats = {"table_cells": (n + 1) * (W + 1)}
    if keep_table:
        stats["table"] = dp
    return AllocationResult(
        method="Dynamic Programming (0/1 Knapsack)", chosen=chosen,
        doses_used=W - w, value=dp[n][W], fractions={c: 1.0 for c in chosen}, stats=stats)
