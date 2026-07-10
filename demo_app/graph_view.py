"""
Interactive knowledge-graph view.

The graph is written during ingest by NetworkXStorage to
``<workdir>/graph_chunk_entity_relation.graphml`` and can be read back with a plain
``networkx.read_graphml``. Node ids are the UPPERCASE entity names *including* literal
surrounding double quotes (e.g. ``"MELATONIN"``); node attrs are ``entity_type``,
``description``, ``source_id``. Edge attrs are ``weight``, ``description``,
``source_id``, ``order``. The graph is undirected.
"""
import os
import html as _html
from collections import Counter

GRAPH_FIELD_SEP = "<SEP>"

# Stable colors for the common entity types the prompt emits; others get a fallback.
_TYPE_COLORS = {
    "PERSON": "#4f9dff",
    "ORGANIZATION": "#ff8c42",
    "GEO": "#33c06f",
    "LOCATION": "#33c06f",
    "EVENT": "#c678dd",
    "CONCEPT": "#e5c07b",
    "TECHNOLOGY": "#56b6c2",
    "PRODUCT": "#e06c75",
    "UNKNOWN": "#9aa0a6",
}
_FALLBACK_PALETTE = [
    "#4f9dff", "#ff8c42", "#33c06f", "#c678dd", "#e5c07b",
    "#56b6c2", "#e06c75", "#d19a66", "#98c379", "#61afef",
]


def _clean(s) -> str:
    if s is None:
        return ""
    return str(s).strip().strip('"')


def load_graph(graphml_path: str):
    """Return an undirected networkx graph, or None if the file is missing/empty."""
    if not os.path.exists(graphml_path):
        return None
    import networkx as nx
    try:
        g = nx.read_graphml(graphml_path)
    except Exception:
        return None
    return g if g.number_of_nodes() else None


def graph_stats(g) -> dict:
    types = Counter(_clean(data.get("entity_type", "UNKNOWN")) for _, data in g.nodes(data=True))
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "types": dict(types.most_common()),
    }


def entity_types(g) -> list:
    return sorted({_clean(data.get("entity_type", "UNKNOWN")) for _, data in g.nodes(data=True)})


def _color_for(etype: str, type_index: dict) -> str:
    if etype in _TYPE_COLORS:
        return _TYPE_COLORS[etype]
    if etype not in type_index:
        type_index[etype] = _FALLBACK_PALETTE[len(type_index) % len(_FALLBACK_PALETTE)]
    return type_index[etype]


def build_html(g, selected_types=None, min_degree: int = 0, physics: bool = True,
               max_nodes: int = 400, height: int = 650) -> str:
    """Render an interactive vis.js network as a self-contained HTML string."""
    from pyvis.network import Network

    # Filter nodes by type + degree, then cap to the highest-degree ``max_nodes``.
    degrees = dict(g.degree())
    keep = []
    for node, data in g.nodes(data=True):
        etype = _clean(data.get("entity_type", "UNKNOWN"))
        if selected_types and etype not in selected_types:
            continue
        if degrees.get(node, 0) < min_degree:
            continue
        keep.append(node)
    keep = sorted(keep, key=lambda n: degrees.get(n, 0), reverse=True)[:max_nodes]
    keep_set = set(keep)

    net = Network(
        height=f"{height}px", width="100%", bgcolor="#111418", font_color="#e6e6e6",
        directed=False, cdn_resources="in_line", notebook=False,
    )
    net.toggle_physics(physics)

    type_index = {}
    for node in keep:
        data = g.nodes[node]
        etype = _clean(data.get("entity_type", "UNKNOWN"))
        label = _clean(node)
        desc = _clean(data.get("description", "")).replace(GRAPH_FIELD_SEP, " • ")
        deg = degrees.get(node, 0)
        title = f"{label}  [{etype}]\n{desc}" if desc else f"{label}  [{etype}]"
        net.add_node(
            node, label=label, title=title,
            color=_color_for(etype, type_index),
            size=10 + min(deg * 3, 40),
        )

    for u, v, data in g.edges(data=True):
        if u not in keep_set or v not in keep_set:
            continue
        try:
            weight = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        edesc = _clean(data.get("description", "")).replace(GRAPH_FIELD_SEP, " • ")
        net.add_edge(u, v, value=weight, width=min(1 + weight, 8), title=edesc)

    net.set_options("""
    {
      "interaction": {"hover": true, "tooltipDelay": 120, "navigationButtons": true},
      "physics": {"stabilization": {"iterations": 150}, "barnesHut": {"gravitationalConstant": -12000, "springLength": 130}}
    }
    """)

    try:
        return net.generate_html(notebook=False)
    except TypeError:
        return net.generate_html()
    except Exception:
        # Last resort: write to a temp file and read it back.
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "ebr_graph.html")
        net.save_graph(tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            return f.read()


def legend_html(stats: dict) -> str:
    """Small HTML legend of entity types + counts, colored to match the graph."""
    type_index = {}
    rows = []
    for etype, count in stats.get("types", {}).items():
        color = _color_for(etype, type_index)
        rows.append(
            f'<span style="display:inline-block;margin:2px 8px 2px 0;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:4px;"></span>{_html.escape(etype)} ({count})</span>'
        )
    return "".join(rows)
