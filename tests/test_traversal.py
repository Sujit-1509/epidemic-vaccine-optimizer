from core.traversal import bfs_levels, bfs_path, dfs_components, dfs_order, dfs_reachable


def test_bfs_levels_line(line_graph):
    res = bfs_levels(line_graph, "S")
    assert res["level"] == {"S": 0, "A": 1, "B": 2, "C": 3}
    assert res["waves"] == [["S"], ["A"], ["B"], ["C"]]
    assert res["unreachable"] == ["D"]
    assert bfs_path(res["parent"], "C") == ["S", "A", "B", "C"]


def test_bfs_small_city(small_city):
    res = bfs_levels(small_city, "WH")
    assert set(res["waves"][1]) == {"A", "C", "E"}
    assert res["level"]["D"] == 2
    assert set(res["unreachable"]) == {"G", "H"}


def test_dfs_order_visits_all_reachable(line_graph):
    assert dfs_order(line_graph, "S") == ["S", "A", "B", "C"]


def test_dfs_components(small_city):
    comps = sorted(sorted(c) for c in dfs_components(small_city))
    assert comps == [sorted(["WH", "A", "B", "C", "D", "E", "F"]), ["G", "H"]]


def test_dfs_reachable(small_city):
    assert dfs_reachable(small_city, ["WH"]) == {"WH", "A", "B", "C", "D", "E", "F"}
