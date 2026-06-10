from __future__ import annotations

from typing import Any

from .registry import EntityRegistry, PROVISIONAL_PATTERN


async def build_unified_graph(
    storage,
    registry_payload: dict[str, Any],
    alignment_segments: dict[str, dict[str, Any]],
    *,
    clear: bool = True,
) -> dict[str, Any]:
    registry = EntityRegistry(registry_payload)
    if clear and hasattr(storage, "clear"):
        await storage.clear()

    for entity_id, node in registry.entities.items():
        if PROVISIONAL_PATTERN.match(entity_id):
            raise ValueError(f"Provisional node leaked into graph: {entity_id}")
        source_ids = sorted(
            {
                provenance.segment_id
                for provenance in node.provenance
                if provenance.segment_id
            }
        )
        await storage.upsert_node(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": node.entity_type,
                "canonical_name": node.canonical_name,
                "description": node.canonical_name,
                "aliases": [alias.to_dict() for alias in node.aliases],
                "sources": node.sources,
                "first_seen": node.first_seen if node.first_seen is not None else -1.0,
                "last_seen": node.last_seen if node.last_seen is not None else -1.0,
                "confidence": node.confidence,
                "attributes": node.attributes,
                "provenance": [item.to_dict() for item in node.provenance],
                "source_id": "<SEP>".join(source_ids),
            },
        )

    edge_count = 0
    for segment_id, segment in alignment_segments.items():
        for edge in segment.get("edges", []):
            source = str(edge["source_id"])
            target = str(edge["target_id"])
            if source not in registry.entities or target not in registry.entities:
                raise ValueError(
                    f"Edge {edge.get('edge_id')} references missing node: {source} -> {target}"
                )
            await storage.upsert_edge(
                source,
                target,
                {
                    **edge,
                    "weight": float(edge.get("confidence", 0.0)),
                    "description": str(edge.get("description") or edge["predicate"]),
                },
            )
            edge_count += 1
    return {
        "nodes": len(registry.entities),
        "edges": edge_count,
    }


async def validate_unified_graph(storage) -> dict[str, Any]:
    graph = storage._graph
    provisional_nodes = [
        node for node in graph.nodes if PROVISIONAL_PATTERN.match(str(node))
    ]
    dangling_edges = [
        {"source": source, "target": target, "edge_id": str(key)}
        for source, target, key in graph.edges(keys=True)
        if source not in graph or target not in graph
    ]
    missing_provenance = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        provenance = data.get("provenance")
        if not provenance or provenance in {"[]", ""}:
            missing_provenance.append(
                {"source": source, "target": target, "edge_id": str(key)}
            )
    return {
        "valid": not provisional_nodes and not dangling_edges and not missing_provenance,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "provisional_nodes": provisional_nodes,
        "dangling_edges": dangling_edges,
        "edges_missing_provenance": missing_provenance,
    }

