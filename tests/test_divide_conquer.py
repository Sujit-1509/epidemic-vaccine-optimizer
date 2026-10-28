import random

from core.divide_conquer import flatten_tree, merge_sort, rank_areas, regional_split


def test_merge_sort_matches_sorted():
    rng = random.Random(5)
    for n in [0, 1, 2, 7, 100]:
        data = [rng.randint(-50, 50) for _ in range(n)]
        out, _ = merge_sort(data)
        assert out == sorted(data)
        assert merge_sort(data, reverse=True)[0] == sorted(data, reverse=True)


def test_rank_areas(maharashtra):
    ranked, _ = rank_areas(maharashtra)
    pr = [a.priority for a in ranked]
    assert pr == sorted(pr, reverse=True)
    assert ranked[0].id == "MUM"


def test_regional_split_totals(maharashtra):
    areas = maharashtra.demand_areas()
    tree = regional_split(areas)
    assert tree["total_doses"] == sum(a.doses_required for a in areas)
    assert tree["riskiest"] == max(areas, key=lambda a: a.risk).id
    for row in flatten_tree(tree):
        assert row["depth"] <= 3
