"""NP-Completeness demonstration: placing vaccination checkpoints.

Problem: choose the fewest areas for checkpoints so that every road has a
checkpoint at one of its ends (so no travel goes unscreened). This is the
VERTEX COVER problem, one of Karp's 21 NP-complete problems.

  * exact_vertex_cover   - tries all subsets by size: O(2^n * E). Correct but explodes.
  * approx_vertex_cover  - repeatedly take an uncovered road and add BOTH ends.
                           O(V + E), and the answer is at most 2x the optimum
                           (the chosen roads form a matching; the optimum must
                           contain at least one end of each of them).
"""
from __future__ import annotations

from itertools import combinations

from .models import Network

EXACT_LIMIT = 24   # refuse brute force above this many vertices


def edges_of(net: Network) -> list[tuple[str, str]]:
    return [(r.u, r.v) for r in net.roads if r.u != r.v]


def is_vertex_cover(edges: list[tuple[str, str]], cover: set[str]) -> bool:
    return all(u in cover or v in cover for u, v in edges)


def exact_vertex_cover(nodes: list[str], edges: list[tuple[str, str]]) -> dict:
    # only vertices that touch an edge can matter
    useful = sorted({x for e in edges for x in e})
    if len(useful) > EXACT_LIMIT:
        raise ValueError(f"Brute force refused for {len(useful)} vertices (limit {EXACT_LIMIT})")
    checked = 0
    for k in range(len(useful) + 1):
        for subset in combinations(useful, k):
            checked += 1
            s = set(subset)
            if is_vertex_cover(edges, s):
                return {"cover": sorted(s), "size": k, "subsets_checked": checked}
    return {"cover": [], "size": 0, "subsets_checked": checked}


def approx_vertex_cover(edges: list[tuple[str, str]]) -> dict:
    cover: set[str] = set()
    matching = []
    for u, v in edges:
        if u not in cover and v not in cover:
            cover.update((u, v))
            matching.append((u, v))
    return {"cover": sorted(cover), "size": len(cover), "matching": matching}


EXPLANATION = """
**P** – problems solvable in polynomial time (e.g. BFS, DFS, max flow, sorting).

**NP** – problems whose *solution can be verified* in polynomial time.

**NP-Complete** – the hardest problems in NP: every NP problem reduces to them in
polynomial time. No polynomial algorithm is known for any of them; finding one
would prove P = NP.

**Where they appear in this project**
- *0/1 Knapsack* (vaccine allocation, decision version) is NP-complete. Our DP is
  O(n·W) – **pseudo-polynomial**, fast only while the supply W is a small number.
- *Constrained allocation* via backtracking is exponential in the worst case.
- *Vaccination checkpoint placement* = **Vertex Cover**, NP-complete. Below, exact
  brute force is compared with a fast 2-approximation.

**Practical response:** use exact methods for small instances, and greedy or
approximation algorithms with proven quality bounds for large ones.
"""
