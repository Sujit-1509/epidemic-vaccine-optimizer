"""Graph traversal: BFS for spread waves, DFS for clusters and reachability.

Both run in O(V + E) time and O(V) extra space.
"""
from __future__ import annotations

from collections import deque

from .models import Network


def bfs_levels(net: Network, start: str) -> dict:
    """Breadth-First Search from the outbreak area.

    level[v] = minimum number of roads between `start` and v, i.e. the
    "wave" in which the infection could reach v if it spreads one hop per step.
    """
    if start not in net.areas:
        raise KeyError(start)
    level = {start: 0}
    parent: dict[str, str | None] = {start: None}
    order = []
    queue = deque([start])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in net.adj[u]:
            if v not in level:          # first visit = shortest hop distance
                level[v] = level[u] + 1
                parent[v] = u
                queue.append(v)

    waves: list[list[str]] = [[] for _ in range(max(level.values()) + 1)]
    for v in order:
        waves[level[v]].append(v)
    unreachable = [a for a in net.areas if a not in level]
    return {"order": order, "level": level, "parent": parent,
            "waves": waves, "unreachable": unreachable}


def bfs_path(parent: dict[str, str | None], target: str) -> list[str]:
    """Rebuild the shortest-hop path from the BFS start to `target`."""
    if target not in parent:
        return []
    path = []
    node: str | None = target
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def dfs_order(net: Network, start: str) -> list[str]:
    """Depth-First Search visit order from `start` (iterative, explicit stack)."""
    visited: set[str] = set()
    order = []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        # push neighbours in reverse so the smallest-index neighbour is explored first
        for v in reversed(net.adj[u]):
            if v not in visited:
                stack.append(v)
    return order


def dfs_components(net: Network) -> list[list[str]]:
    """Connected components (isolated clusters of areas) using repeated DFS."""
    seen: set[str] = set()
    components = []
    for a in net.areas:
        if a not in seen:
            comp = dfs_order(net, a)
            seen.update(comp)
            components.append(comp)
    return components


def dfs_reachable(net: Network, sources: list[str]) -> set[str]:
    """All areas reachable from any of the sources (e.g. warehouses)."""
    reached: set[str] = set()
    for s in sources:
        if s not in reached:
            reached.update(dfs_order(net, s))
    return reached
