import pytest

from core.generator import random_network
from core.np_complete import approx_vertex_cover, edges_of, exact_vertex_cover, is_vertex_cover


def test_star_graph():
    edges = [("c", x) for x in "abde"]
    assert exact_vertex_cover(list("cabde"), edges)["size"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_approx_within_factor_two(seed):
    net = random_network(12, seed=seed)
    edges = edges_of(net)
    exact = exact_vertex_cover(list(net.areas), edges)
    approx = approx_vertex_cover(edges)
    assert is_vertex_cover(edges, set(exact["cover"]))
    assert is_vertex_cover(edges, set(approx["cover"]))
    assert exact["size"] <= approx["size"] <= 2 * exact["size"]


def test_brute_force_refuses_large():
    net = random_network(40, seed=1)
    with pytest.raises(ValueError):
        exact_vertex_cover(list(net.areas), edges_of(net))
