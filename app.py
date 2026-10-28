"""Epidemic Containment & Vaccine Distribution Optimizer - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.backtracking import backtracking_allocate
from core.benchmark import COMPLEXITY, run_benchmark, vertex_cover_scaling
from core.divide_conquer import flatten_tree, rank_areas, regional_split
from core.greedy import greedy_01, greedy_fractional
from core.knapsack_dp import knapsack_dp
from core.max_flow import vaccine_max_flow
from core.models import Network, items_from_network
from core.np_complete import EXPLANATION, approx_vertex_cover, edges_of, exact_vertex_cover
from core.timing import timed
from core.traversal import bfs_levels, bfs_path, dfs_components, dfs_order, dfs_reachable

DATA_DIR = Path(__file__).parent / "data"
BACKTRACK_MAX_AREAS = 40
EXACT_VC_MAX = 20
WAVE_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1

st.set_page_config(page_title="Epidemic & Vaccine Optimizer", page_icon="💉", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data
def cached_vc_scaling() -> pd.DataFrame:
    return vertex_cover_scaling()


@st.cache_data
def load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def network_figure(net: Network, node_color: dict[str, str] | None = None,
                   node_label: dict[str, str] | None = None,
                   edge_style: dict[tuple[str, str], tuple[str, float]] | None = None,
                   edge_text: dict[tuple[str, str], str] | None = None,
                   height: int = 520) -> go.Figure:
    """Draw the area graph. Nodes sized by population; warehouses drawn as squares."""
    fig = go.Figure()
    edge_style = edge_style or {}
    groups: dict[tuple[str, float], tuple[list, list]] = {}
    mid_x, mid_y, mid_t = [], [], []
    for r in net.roads:
        a, b = net.areas[r.u], net.areas[r.v]
        style = edge_style.get((r.u, r.v)) or edge_style.get((r.v, r.u)) or ("#b0b0b0", 1.5)
        xs, ys = groups.setdefault(style, ([], []))
        xs += [a.x, b.x, None]
        ys += [a.y, b.y, None]
        mid_x.append((a.x + b.x) / 2)
        mid_y.append((a.y + b.y) / 2)
        txt = (edge_text or {}).get((r.u, r.v)) or (edge_text or {}).get((r.v, r.u))
        mid_t.append(txt or f"{r.u}–{r.v}: {r.distance_km} km, cap {r.capacity}")
    for (color, width), (xs, ys) in groups.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                 line=dict(color=color, width=width), showlegend=False))
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", hoverinfo="text", text=mid_t,
                             marker=dict(size=6, color="rgba(0,0,0,0)"), showlegend=False))

    max_pop = max(a.population for a in net.areas.values()) or 1
    ids = list(net.areas)
    colors = [(node_color or {}).get(i, "#888") for i in ids]
    labels = [(node_label or {}).get(i, i) for i in ids]
    fig.add_trace(go.Scatter(
        x=[net.areas[i].x for i in ids], y=[net.areas[i].y for i in ids],
        mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=[14 + 26 * math.sqrt(net.areas[i].population / max_pop) for i in ids],
                    color=colors, line=dict(color="#333", width=1),
                    symbol=["square" if i in net.warehouses else "circle" for i in ids]),
        hovertext=[f"<b>{net.areas[i].name}</b> ({i})<br>Population: {net.areas[i].population:,}"
                   f"<br>Risk: {net.areas[i].risk:.2f}<br>Doses needed: {net.areas[i].doses_required}k"
                   for i in ids],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def risk_color(risk: float) -> str:
    return px.colors.sample_colorscale("YlOrRd", [0.15 + 0.85 * risk])[0]


def legend(items: dict[str, str]) -> None:
    html = " ".join(f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
                    f"border:1px solid #333;margin:0 4px 0 12px;vertical-align:middle'></span>{t}"
                    for t, c in items.items())
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("💉 Controls")
files = sorted(DATA_DIR.glob("*.json"))
names = [f.name for f in files]
choice = st.sidebar.selectbox("Dataset", names + ["Upload JSON…"],
                              index=names.index("maharashtra_districts.json")
                              if "maharashtra_districts.json" in names else 0)
if choice == "Upload JSON…":
    up = st.sidebar.file_uploader("Network JSON", type="json")
    if not up:
        st.info("Upload a JSON file with `areas`, `roads` and `warehouses` (same format as files in `data/`).")
        st.stop()
    raw = json.load(up)
else:
    raw = load_dataset(str(DATA_DIR / choice))
try:
    net = Network.from_dict(raw)
except (KeyError, TypeError, ValueError) as e:
    st.error(f"Invalid network file: {e}")
    st.stop()

demand_areas = net.demand_areas()
total_demand = sum(a.doses_required for a in demand_areas)
supply = st.sidebar.slider("Vaccine supply (thousand doses)", 0, max(total_demand, 1),
                           int(total_demand * 0.5), help="Total doses available this round")
default_outbreak = max(demand_areas, key=lambda a: a.risk).id if demand_areas else next(iter(net.areas))
outbreak = st.sidebar.selectbox("Outbreak origin", list(net.areas),
                                index=list(net.areas).index(default_outbreak),
                                format_func=lambda i: f"{net.areas[i].name} ({i})")
st.sidebar.markdown("**Backtracking constraints**")
use_must = st.sidebar.checkbox("Must serve all high-risk areas", value=True)
risk_thr = st.sidebar.slider("High-risk threshold", 0.0, 1.0, 0.7, 0.05, disabled=not use_must)
max_areas = st.sidebar.number_input("Max areas served (vaccination teams, 0 = no limit)",
                                    0, len(net), 0)
st.sidebar.caption("Doses and road capacities are in units of 1,000. "
                   "Priority = population × risk.")

# ---------------------------------------------------------------- header
st.title("Epidemic Containment & Vaccine Distribution Optimizer")
st.caption(f"Dataset: **{net.name}** · Design and Analysis of Algorithms project · "
           "Educational decision-support model — not for real public-health decisions.")

SECTIONS = (["🗺️ Overview", "🦠 Spread (BFS/DFS)", "💉 Allocation", "🚚 Distribution (Max Flow)",
                "✂️ Divide & Conquer", "🧩 NP-Completeness", "📈 Performance"])
# a radio keeps the chosen section across reruns (st.tabs jumps back to the first tab)
section = st.radio("Section", SECTIONS, horizontal=True, label_visibility="collapsed", key="section")
st.divider()

# ---------------------------------------------------------------- 1. overview
if section == SECTIONS[0]:
    c = st.columns(5)
    c[0].metric("Areas", len(net))
    c[1].metric("Roads", len(net.roads))
    c[2].metric("Warehouses", len(net.warehouses))
    c[3].metric("Population", f"{sum(a.population for a in net.areas.values()):,}")
    c[4].metric("Total demand (k doses)", f"{total_demand:,}")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(network_figure(net, {i: ("#4c78a8" if i in net.warehouses else risk_color(a.risk))
                                             for i, a in net.areas.items()}), width="stretch")
        st.caption("Colour = infection risk (yellow → red). Size = population. Squares = warehouses. "
                   "Hover over nodes and road midpoints for details.")
    with right:
        df = pd.DataFrame([{"id": a.id, "name": a.name, "population": a.population, "risk": a.risk,
                            "doses (k)": a.doses_required, "priority": round(a.priority),
                            "warehouse": a.id in net.warehouses} for a in net.areas.values()])
        st.dataframe(df, hide_index=True, width="stretch", height=480)

# ---------------------------------------------------------------- 2. spread
if section == SECTIONS[1]:
    st.subheader(f"BFS — infection spread waves from {net.areas[outbreak].name}")
    bfs, bfs_ms = timed(bfs_levels, net, outbreak)
    wave_col = {v: WAVE_COLORS[lvl % len(WAVE_COLORS)] for v, lvl in bfs["level"].items()}
    for v in bfs["unreachable"]:
        wave_col[v] = "#dddddd"
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(network_figure(net, wave_col, {v: f"{v} (w{l})" for v, l in bfs["level"].items()}),
                        width="stretch")
        legend({f"Wave {k}": WAVE_COLORS[k % len(WAVE_COLORS)] for k in range(len(bfs["waves"]))}
               | ({"Unreachable": "#dddddd"} if bfs["unreachable"] else {}))
    with right:
        st.metric("BFS time", f"{bfs_ms:.3f} ms")
        st.dataframe(pd.DataFrame([{"wave (hops)": k, "areas": ", ".join(w),
                                    "population at risk": sum(net.areas[v].population for v in w)}
                                   for k, w in enumerate(bfs["waves"])]),
                     hide_index=True, width="stretch")
        if bfs["unreachable"]:
            st.info("Not reachable from the outbreak (no road link): " + ", ".join(bfs["unreachable"]))
        target = st.selectbox("Shortest spread path to", [v for v in net.areas if v in bfs["level"]],
                              index=len(bfs["order"]) - 1)
        st.write(" → ".join(bfs_path(bfs["parent"], target)))

    st.divider()
    st.subheader("DFS — isolated clusters and warehouse reachability")
    comps, dfs_ms = timed(dfs_components, net)
    reach = dfs_reachable(net, net.warehouses)
    left, right = st.columns([3, 2])
    with left:
        comp_col = {}
        for k, comp in enumerate(comps):
            for v in comp:
                comp_col[v] = WAVE_COLORS[k % len(WAVE_COLORS)]
        st.plotly_chart(network_figure(net, comp_col, height=420), width="stretch")
        legend({f"Cluster {k + 1}": WAVE_COLORS[k % len(WAVE_COLORS)] for k in range(len(comps))})
    with right:
        st.metric("Connected clusters", len(comps), help=f"DFS time {dfs_ms:.3f} ms")
        for k, comp in enumerate(comps, 1):
            st.write(f"**Cluster {k}** ({len(comp)} areas): {', '.join(comp)}")
        cut_off = [a.id for a in demand_areas if a.id not in reach]
        if cut_off:
            st.error(f"No warehouse can reach: {', '.join(cut_off)} — these areas need an alternative "
                     "supply route (e.g. airlift) or a new warehouse.")
        else:
            st.success("Every area is reachable from at least one warehouse.")
        if net.warehouses:
            st.caption(f"DFS visit order from {net.warehouses[0]}: "
                       + " → ".join(dfs_order(net, net.warehouses[0])))

# ---------------------------------------------------------------- 3. allocation
if section == SECTIONS[2]:
    st.subheader(f"Vaccine allocation with supply = {supply}k doses (demand {total_demand}k)")
    items = items_from_network(net)
    total_priority = sum(i.value for i in items) or 1
    must = {a.id for a in demand_areas if a.risk >= risk_thr} if use_must else set()
    results = []
    for fn, args in [(greedy_01, (items, supply)), (greedy_fractional, (items, supply)),
                     (knapsack_dp, (items, supply))]:
        results.append(timed(fn, *args))
    if len(items) <= BACKTRACK_MAX_AREAS:
        results.append(timed(backtracking_allocate, items, supply, must, max_areas or None))
    else:
        st.warning(f"Backtracking skipped: more than {BACKTRACK_MAX_AREAS} areas (exponential search).")

    rows = []
    for res, ms in results:
        rows.append({"method": res.method, "feasible": res.feasible,
                     "areas served": len(res.chosen), "doses used (k)": round(res.doses_used, 1),
                     "priority covered": round(res.value),
                     "% of total priority": round(100 * res.value / total_priority, 1),
                     "time (ms)": round(ms, 3),
                     "stats": ", ".join(f"{k}={v}" for k, v in res.stats.items() if k != "table")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if use_must:
        st.caption(f"Backtracking must serve areas with risk ≥ {risk_thr}: {', '.join(sorted(must)) or 'none'}"
                   + (f"; at most {max_areas} areas." if max_areas else "."))
    if results and not results[-1][0].feasible:
        st.error("Backtracking found no allocation that satisfies the constraints — "
                 "increase the supply, lower the threshold, or raise the area limit.")

    left, right = st.columns([2, 3])
    with left:
        fig = px.bar(pd.DataFrame(rows), x="method", y="priority covered", color="method",
                     text="% of total priority")
        fig.update_layout(showlegend=False, height=420, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Fractional greedy is an upper bound (it may vaccinate part of an area). "
                   "DP is optimal for whole areas; 0/1 greedy can fall short; backtracking is optimal "
                   "under the extra constraints.")
    with right:
        pick = st.selectbox("Show on map", [r[0].method for r in results])
        res = next(r[0] for r in results if r[0].method == pick)
        col = {}
        for a in net.areas.values():
            f = res.fractions.get(a.id, 0)
            col[a.id] = ("#4c78a8" if a.id in net.warehouses else
                         "#2ca02c" if f >= 0.999 else "#98df8a" if f > 0 else "#d62728")
        st.plotly_chart(network_figure(net, col, height=420), width="stretch")
        legend({"Fully vaccinated": "#2ca02c", "Partly vaccinated": "#98df8a",
                "Not served": "#d62728", "Warehouse": "#4c78a8"})

    if len(items) <= 10 and supply <= 60:
        with st.expander("DP table (rows = areas considered, columns = doses available)"):
            dp_res = knapsack_dp(items, supply, keep_table=True)
            table = pd.DataFrame(dp_res.stats["table"], index=["—"] + [i.id for i in items]).round(0)
            st.dataframe(table, width="stretch")

# ---------------------------------------------------------------- 4. max flow
if section == SECTIONS[3]:
    st.subheader("Ford–Fulkerson (Edmonds–Karp) — maximum doses/day through the road network")
    unlimited = st.checkbox("Unlimited warehouse stock", value=False,
                            help="Otherwise the sidebar supply is split equally across warehouses")
    stock = None if unlimited else {w: supply / max(len(net.warehouses), 1) for w in net.warehouses}
    if not net.warehouses:
        st.warning("This network has no warehouses.")
    else:
        mf, mf_ms = timed(vaccine_max_flow, net, stock)
        c = st.columns(4)
        c[0].metric("Max flow", f"{mf['value']:.0f}k")
        c[1].metric("Total demand", f"{mf['total_demand']}k")
        c[2].metric("Demand met", f"{100 * mf['value'] / max(mf['total_demand'], 1):.1f}%")
        c[3].metric("Time", f"{mf_ms:.3f} ms")
        cut = {tuple(sorted(e)) for e in mf["cut_roads"]}
        estyle, etext = {}, {}
        max_cap = max((r.capacity for r in net.roads), default=1)
        for rf in mf["road_flow"]:
            key = tuple(sorted((rf["from"], rf["to"])))
            w = 1 + 7 * rf["flow"] / max_cap
            color = "#d62728" if key in cut else ("#1f77b4" if rf["flow"] > 0 else "#cccccc")
            estyle[key] = (color, round(w, 1))
            etext[key] = f"{rf['from']} → {rf['to']}: {rf['flow']:.0f}/{rf['capacity']}"
        left, right = st.columns([3, 2])
        with left:
            col = {a.id: ("#4c78a8" if a.id in net.warehouses else
                          "#2ca02c" if a.id not in mf["unmet"] else
                          "#ff7f0e" if mf["delivered"].get(a.id, 0) > 0 else "#8c564b")
                   for a in net.areas.values()}
            st.plotly_chart(network_figure(net, col, edge_style=estyle, edge_text=etext),
                            width="stretch")
            legend({"Flow on road (width = amount)": "#1f77b4", "Min-cut bottleneck road": "#d62728",
                    "Demand fully met": "#2ca02c", "Partly met": "#ff7f0e", "Not reached": "#8c564b"})
        with right:
            dd = pd.DataFrame([{"area": a, "demand": net.areas[a].doses_required, "delivered": round(d, 1)}
                               for a, d in mf["delivered"].items()])
            fig = go.Figure([go.Bar(name="Demand", x=dd.area, y=dd.demand, marker_color="#cccccc"),
                             go.Bar(name="Delivered", x=dd.area, y=dd.delivered, marker_color="#2ca02c")])
            fig.update_layout(barmode="overlay", height=300, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            if mf["cut_roads"]:
                st.write("**Bottleneck roads (min cut):** " + ", ".join(f"{u}–{v}" for u, v in mf["cut_roads"]))
            else:
                st.write("**Min cut:** limited by warehouse stock or area demand, not roads.")
        with st.expander(f"Augmenting paths found ({len(mf['augmenting_paths'])})"):
            st.dataframe(pd.DataFrame([{"step": k + 1, "path": " → ".join(p), "bottleneck": b}
                                       for k, (p, b) in enumerate(mf["augmenting_paths"])]),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. divide & conquer
if section == SECTIONS[4]:
    st.subheader("Merge sort — priority ranking of areas")
    (ranked, comps_n), ms = timed(rank_areas, net)
    n = len(ranked)
    c = st.columns(3)
    c[0].metric("Comparisons", comps_n)
    c[1].metric("n·log₂n", f"{n * math.log2(n):.0f}" if n > 1 else "0")
    c[2].metric("Time", f"{ms:.3f} ms")
    st.dataframe(pd.DataFrame([{"rank": k + 1, "id": a.id, "name": a.name, "priority": round(a.priority),
                                "risk": a.risk, "doses (k)": a.doses_required}
                               for k, a in enumerate(ranked)]), hide_index=True, width="stretch")
    st.divider()
    st.subheader("Recursive regional split (west/east) — demand per sub-region")
    depth = st.slider("Recursion depth", 1, 5, 3)
    tree = regional_split(demand_areas, max_depth=depth)
    if tree["areas"]:
        # recursion-tree layout: leaves spread left→right, parents centred above their children
        pos: dict[str, tuple[float, int]] = {}
        edge_x, edge_y, next_leaf = [], [], [0]

        def place(node: dict, d: int) -> float:
            if node["children"]:
                x = sum(place(ch, d + 1) for ch in node["children"]) / len(node["children"])
            else:
                x = next_leaf[0]; next_leaf[0] += 1
            pos[node["label"]] = (x, d)
            for ch in node["children"]:
                cx, cd = pos[ch["label"]]
                edge_x.extend([x, cx, None]); edge_y.extend([-d, -cd, None])
            return x

        place(tree, 0)
        rows = flatten_tree(tree)
        fig = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#999"),
                                   hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[pos[r["label"]][0] for r in rows], y=[-r["depth"] for r in rows], mode="markers+text",
            marker=dict(size=[18 + 30 * r["total_doses"] / max(tree["total_doses"], 1) for r in rows],
                        color=[risk_color(net.areas[r["riskiest"]].risk) for r in rows],
                        line=dict(color="#333", width=1)),
            text=[f"{r['total_doses']}k" if r["depth"] else f"Region {r['total_doses']}k" for r in rows],
            textposition="top center",
            hovertext=[f"<b>{r['label']}</b><br>Areas: {', '.join(r['areas'])}<br>"
                       f"Demand: {r['total_doses']}k doses<br>Riskiest: {r['riskiest']}" for r in rows],
            hoverinfo="text"))
        fig.update_layout(height=420, showlegend=False, margin=dict(t=30, l=10, r=10, b=10),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each node is one recursive call: size = doses needed, colour = risk of the "
                   "riskiest area inside it. Hover for the areas in each sub-region.")
        st.dataframe(pd.DataFrame(flatten_tree(tree)).assign(areas=lambda d: d.areas.apply(", ".join)),
                     hide_index=True, width="stretch")
        st.caption("Divide: split areas at the median longitude. Conquer: solve each half recursively. "
                   "Combine: add the demand totals and keep the riskier of the two halves' riskiest areas.")

# ---------------------------------------------------------------- 6. NP-completeness
if section == SECTIONS[5]:
    st.markdown(EXPLANATION)
    st.subheader("Vaccination checkpoints = Vertex Cover on this network")
    edges = edges_of(net)
    approx, ap_ms = timed(approx_vertex_cover, edges)
    useful = {x for e in edges for x in e}
    exact = None
    if len(useful) <= EXACT_VC_MAX:
        exact, ex_ms = timed(exact_vertex_cover, list(net.areas), edges)
    c = st.columns(2)
    c[0].metric("2-approximation", f"{approx['size']} checkpoints", f"{ap_ms:.3f} ms", delta_color="off")
    if exact:
        c[1].metric("Exact (brute force)", f"{exact['size']} checkpoints",
                    f"{ex_ms:.1f} ms · {exact['subsets_checked']:,} subsets", delta_color="off")
    else:
        c[1].metric("Exact (brute force)", "skipped",
                    f"{len(useful)} vertices → up to 2^{len(useful)} subsets", delta_color="off")
    show = st.radio("Show cover", ["2-approximation"] + (["Exact"] if exact else []), horizontal=True)
    cover = set((exact if show == "Exact" else approx)["cover"])
    st.plotly_chart(network_figure(net, {v: ("#9467bd" if v in cover else "#dddddd") for v in net.areas},
                                   height=420), width="stretch")
    legend({"Checkpoint": "#9467bd", "No checkpoint": "#dddddd"})

    st.subheader("Exponential blow-up: exact vs approximation as n grows")
    vc = cached_vc_scaling()
    fig = go.Figure([go.Scatter(x=vc.n, y=vc.exact_ms, name="Exact (brute force)", mode="lines+markers"),
                     go.Scatter(x=vc.n, y=vc.approx_ms, name="2-approximation", mode="lines+markers")])
    fig.update_layout(yaxis_type="log", yaxis_title="time (ms, log scale)", xaxis_title="areas (n)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(vc.round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 7. performance
if section == SECTIONS[6]:
    st.subheader("Algorithm performance on random networks of growing size")
    sizes = st.multiselect("Input sizes (number of areas)", [10, 20, 40, 80, 160, 320],
                           default=[10, 20, 40, 80, 160])
    ratio = st.slider("Supply as fraction of demand", 0.1, 0.9, 0.4, 0.1)
    if st.button("Run benchmark", type="primary") and sizes:
        with st.spinner("Timing algorithms…"):
            st.session_state["bench"] = run_benchmark(sorted(sizes), supply_ratio=ratio)
    bench = st.session_state.get("bench")
    if bench is None:
        st.info("Press **Run benchmark** to time every algorithm.")
    else:
        fig = px.line(bench, x="n", y="time_ms", color="algorithm", markers=True, log_y=True,
                      labels={"time_ms": "time (ms, log scale)", "n": "areas (n)"})
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Time (ms)**")
            st.dataframe(bench.pivot(index="algorithm", columns="n", values="time_ms").round(3),
                         width="stretch")
        with right:
            st.markdown("**Theoretical complexity**")
            st.dataframe(pd.DataFrame(COMPLEXITY.items(), columns=["algorithm", "complexity"]),
                         hide_index=True, width="stretch")
        alloc = bench[bench.algorithm.isin(["Greedy 0/1", "Knapsack DP", "Backtracking (B&B)"])].dropna()
        dp_val = alloc[alloc.algorithm == "Knapsack DP"].set_index("n")["value"]
        alloc = alloc.assign(quality=lambda d: 100 * d.value / d.n.map(dp_val))
        st.markdown("**Solution quality (% of optimal DP value)**")
        fig = px.bar(alloc, x="n", y="quality", color="algorithm", barmode="group",
                     labels={"quality": "% of optimum", "n": "areas (n)"})
        fig.update_yaxes(range=[min(90, alloc.quality.min() - 1), 100.5])
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
