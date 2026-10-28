"""Data model for the epidemic network.

An *area* (village, ward, district) is a node in an undirected graph.
A *road* connects two areas and has a transport capacity (doses/day).
Warehouses are areas that hold the vaccine stock (flow sources).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Area:
    id: str
    name: str
    population: int
    risk: float            # infection rate / risk score in [0, 1]
    doses_required: int    # in units of 1,000 doses
    x: float = 0.0         # coordinates used only for drawing
    y: float = 0.0

    @property
    def priority(self) -> float:
        """Expected infections prevented if this area is vaccinated."""
        return self.population * self.risk


@dataclass
class Road:
    u: str
    v: str
    distance_km: float
    capacity: int          # doses/day (units of 1,000) the road can carry


@dataclass
class Network:
    areas: dict[str, Area]
    roads: list[Road]
    warehouses: list[str] = field(default_factory=list)
    name: str = "Network"

    def __post_init__(self) -> None:
        self.adj: dict[str, list[str]] = {a: [] for a in self.areas}
        for r in self.roads:
            if r.u not in self.areas or r.v not in self.areas:
                raise ValueError(f"Road {r.u}-{r.v} references an unknown area")
            self.adj[r.u].append(r.v)
            self.adj[r.v].append(r.u)
        for w in self.warehouses:
            if w not in self.areas:
                raise ValueError(f"Warehouse {w} is not an area")

    # ---- convenience -------------------------------------------------
    def area_list(self) -> list[Area]:
        return list(self.areas.values())

    def demand_areas(self) -> list[Area]:
        """Areas that need vaccines (warehouses excluded)."""
        return [a for a in self.areas.values() if a.id not in self.warehouses]

    def __len__(self) -> int:
        return len(self.areas)

    # ---- JSON I/O ----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        areas = {a["id"]: Area(**a) for a in data["areas"]}
        roads = [Road(**r) for r in data["roads"]]
        return cls(areas, roads, data.get("warehouses", []), data.get("name", "Network"))

    @classmethod
    def from_json(cls, path: str | Path) -> "Network":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "warehouses": self.warehouses,
            "areas": [asdict(a) for a in self.areas.values()],
            "roads": [asdict(r) for r in self.roads],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class Item:
    """One knapsack item: an area that can be vaccinated."""
    id: str
    weight: int      # doses required (units of 1,000)
    value: float     # priority = population x risk


def items_from_network(net: Network) -> list[Item]:
    return [Item(a.id, a.doses_required, a.priority)
            for a in net.demand_areas() if a.doses_required > 0]


@dataclass
class AllocationResult:
    method: str
    chosen: list[str]                 # ids of areas that receive vaccines
    doses_used: float
    value: float                      # total priority covered
    feasible: bool = True
    fractions: dict[str, float] = field(default_factory=dict)  # id -> share vaccinated (1.0 = full)
    stats: dict = field(default_factory=dict)                   # algorithm-specific counters
