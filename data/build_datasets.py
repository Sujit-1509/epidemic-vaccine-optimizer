"""Builds the three sample datasets in this folder. Run: python -m data.build_datasets"""
from pathlib import Path

from core.generator import random_network
from core.models import Area, Network, Road

HERE = Path(__file__).parent


def small_city() -> Network:
    A = Area
    areas = [
        A("WH", "Central Medical Store", 20_000, 0.10, 0, 50, 50),
        A("A", "Old Town", 85_000, 0.80, 12, 35, 70),
        A("B", "Riverside", 60_000, 0.65, 9, 20, 85),
        A("C", "Market Hub", 120_000, 0.70, 18, 60, 75),
        A("D", "Hillview", 30_000, 0.30, 5, 45, 95),
        A("E", "Industrial Zone", 70_000, 0.55, 10, 75, 35),
        A("F", "University Area", 45_000, 0.45, 7, 85, 60),
        A("G", "Outer Village", 15_000, 0.90, 3, 10, 20),
        A("H", "Island Colony", 8_000, 0.75, 2, 25, 10),
    ]
    R = Road
    roads = [
        R("WH", "A", 3, 15), R("WH", "C", 4, 20), R("WH", "E", 6, 10),
        R("A", "B", 2, 8), R("A", "C", 3, 10), R("C", "D", 5, 6),
        R("C", "F", 4, 8), R("E", "F", 3, 6), R("B", "D", 4, 5),
        R("G", "H", 2, 4),  # G-H cluster is cut off (flooded road) -> DFS demo
    ]
    return Network({a.id: a for a in areas}, roads, ["WH"], "Small City (9 areas)")


def maharashtra() -> Network:
    # Populations: approx. Census 2011. Risk scores are hypothetical (for demo only).
    # doses_required = priority-group campaign size in units of 1,000 doses.
    # x, y = longitude, latitude.
    A = Area
    areas = [
        A("PUN", "Pune (Warehouse)", 9_430_000, 0.40, 0, 73.86, 18.52),
        A("NAG", "Nagpur (Warehouse)", 4_650_000, 0.35, 0, 79.09, 21.15),
        A("MUM", "Mumbai", 12_440_000, 0.85, 250, 72.88, 19.08),
        A("THA", "Thane", 11_060_000, 0.75, 220, 72.97, 19.20),
        A("NAS", "Nashik", 6_110_000, 0.55, 120, 73.79, 20.00),
        A("AUR", "Aurangabad", 3_700_000, 0.60, 75, 75.34, 19.88),
        A("SOL", "Solapur", 4_320_000, 0.45, 85, 75.91, 17.66),
        A("KOL", "Kolhapur", 3_880_000, 0.35, 80, 74.24, 16.70),
        A("AMR", "Amravati", 2_890_000, 0.50, 60, 77.75, 20.93),
        A("NAN", "Nanded", 3_360_000, 0.40, 65, 77.32, 19.15),
        A("SAT", "Satara", 3_000_000, 0.30, 60, 74.00, 17.68),
        A("AHM", "Ahmednagar", 4_540_000, 0.50, 90, 74.74, 19.09),
        A("JAL", "Jalgaon", 4_230_000, 0.45, 85, 75.56, 21.00),
        A("LAT", "Latur", 2_450_000, 0.55, 50, 76.56, 18.40),
        A("RAT", "Ratnagiri", 1_620_000, 0.25, 30, 73.30, 16.99),
        A("AKO", "Akola", 1_810_000, 0.60, 35, 77.00, 20.70),
    ]
    R = Road
    roads = [
        R("MUM", "THA", 25, 120), R("MUM", "PUN", 150, 100), R("THA", "NAS", 150, 80),
        R("PUN", "NAS", 210, 60), R("PUN", "AHM", 120, 70), R("PUN", "SAT", 110, 80),
        R("SAT", "KOL", 125, 70), R("KOL", "RAT", 130, 30), R("MUM", "RAT", 330, 40),
        R("PUN", "SOL", 250, 70), R("SOL", "LAT", 110, 40), R("LAT", "NAN", 140, 40),
        R("AHM", "AUR", 115, 60), R("NAS", "AUR", 180, 50), R("AUR", "JAL", 160, 50),
        R("AUR", "NAN", 270, 40), R("JAL", "AKO", 230, 40), R("AKO", "AMR", 100, 60),
        R("AMR", "NAG", 155, 80), R("AHM", "SOL", 230, 40), R("NAS", "JAL", 250, 40),
        R("AKO", "NAN", 250, 30),
    ]
    return Network({a.id: a for a in areas}, roads, ["PUN", "NAG"], "Maharashtra Districts (16)")


def large_region() -> Network:
    net = random_network(40, seed=7, n_warehouses=2)
    net.name = "Large Region (40 areas, generated)"
    return net


if __name__ == "__main__":
    small_city().to_json(HERE / "small_city.json")
    maharashtra().to_json(HERE / "maharashtra_districts.json")
    large_region().to_json(HERE / "large_region.json")
    print("Datasets written to", HERE)
