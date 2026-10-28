import pytest

from core.max_flow import edmonds_karp, vaccine_max_flow


def test_clrs_example():
    # classic textbook network (CLRS Fig. 26.1): max flow is 23
    cap = {"s": {"v1": 16, "v2": 13}, "v1": {"v3": 12}, "v2": {"v1": 4, "v4": 14},
           "v3": {"v2": 9, "t": 20}, "v4": {"v3": 7, "t": 4}}
    res = edmonds_karp(cap, "s", "t")
    assert res["value"] == 23
    cut_capacity = sum(cap[u][v] for u, v in res["cut"])
    assert cut_capacity == 23        # max-flow = min-cut


def test_flow_conservation(maharashtra):
    res = vaccine_max_flow(maharashtra)
    assert res["value"] == pytest.approx(sum(res["delivered"].values()))
    assert res["value"] <= res["total_demand"]
    for rf in res["road_flow"]:
        assert rf["flow"] <= rf["capacity"] + 1e-9


def test_disconnected_areas_unserved(small_city):
    res = vaccine_max_flow(small_city)
    assert res["delivered"]["G"] == 0 and res["delivered"]["H"] == 0
    assert "G" in res["unmet"] and "H" in res["unmet"]


def test_warehouse_stock_limits_flow(small_city):
    res = vaccine_max_flow(small_city, {"WH": 10})
    assert res["value"] == pytest.approx(10)
