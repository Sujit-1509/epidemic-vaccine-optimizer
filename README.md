# Epidemic Containment & Vaccine Distribution Optimizer

A Design and Analysis of Algorithms (DAA) project. Areas and roads are modelled as a graph.
Different algorithms are used to study how an epidemic spreads and to plan vaccine allocation
and distribution under limited supply. Every algorithm is written by hand in `core/`; no
algorithm libraries are used.

> Educational decision-support model. Results need validation before any real public-health use.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard at http://localhost:8501
python -m pytest tests -p no:randomly   # unit tests
python -m core.benchmark      # timing table in the terminal
```

## Structure

| File | Algorithm | Question it answers | Complexity |
|---|---|---|---|
| `core/traversal.py` | BFS | How many road hops is each area from the outbreak (spread waves)? | O(V+E) |
| | DFS | Which clusters are cut off? Which areas can a warehouse reach? | O(V+E) |
| `core/greedy.py` | Greedy (0/1, fractional) | Quick allocation by priority per dose | O(n log n) |
| `core/knapsack_dp.py` | 0/1 Knapsack DP | Best whole-area allocation under the supply limit | O(n·W) |
| `core/backtracking.py` | Backtracking + branch & bound | Best allocation that must also cover high-risk areas or use at most k teams | O(2ⁿ) worst |
| `core/max_flow.py` | Ford–Fulkerson (Edmonds–Karp) | Max doses/day through road capacities, and the bottleneck roads (min cut) | O(V·E²) |
| `core/divide_conquer.py` | Merge sort, recursive regional split | Priority ranking; demand per sub-region | O(n log n) |
| `core/np_complete.py` | Vertex Cover: brute force vs 2-approximation | Fewest vaccination checkpoints covering every road (NP-complete) | O(2ⁿ) vs O(V+E) |
| `core/benchmark.py` | — | Runtime and solution quality as input size grows | — |

Other files: `core/models.py` (Area, Road, Network), `core/generator.py` (random networks),
`data/` (sample datasets; `python -m data.build_datasets` regenerates them), `tests/`.

## Data model

- **Area**: population, risk (0–1), doses required (units of 1,000), x/y position.
- **Road**: distance and transport capacity (thousand doses/day).
- **Priority** of an area = population × risk (expected infections prevented).
- **Warehouses** are the source nodes for max flow.

## Datasets

- `small_city.json`: 9 wards. Includes a cut-off cluster (G–H) so the DFS and max-flow sections have something to flag.
- `maharashtra_districts.json`: 16 districts with approximate Census 2011 populations. Risk scores are made up. Warehouses are in Pune and Nagpur.
- `large_region.json`: 40 generated areas, for stress testing.

To use your own network, upload a JSON file in the same format from the dashboard sidebar.
