from __future__ import annotations

import logging
from typing import Any

from .registry import EntityRegistry, PROVISIONAL_PATTERN

logger = logging.getLogger(__name__)


async def build_unified_graph(
    storage,
    registry_payload: dict[str, Any],
    alignment_segments: dict[str, dict[str, Any]],
    *,
    clear: bool = True,
    entity_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = EntityRegistry(registry_payload)
    entity_memory = entity_memory or {}
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
        mem = entity_memory.get(entity_id, {})
        descriptions = mem.get("accumulated_descriptions", [])
        if descriptions:
            description = f"{node.canonical_name}: " + "; ".join(descriptions[-5:])
        else:
            description = node.canonical_name
        await storage.upsert_node(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": node.entity_type,
                "canonical_name": node.canonical_name,
                "description": description,
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
    dropped_edges = 0
    for segment_id, segment in alignment_segments.items():
        for edge in segment.get("edges", []):
            source = str(edge["source_id"])
            target = str(edge["target_id"])
            if source not in registry.entities or target not in registry.entities:
                # Drop edges whose endpoints never registered as nodes. This is
                # invalid data (an entity referenced in a relationship but filtered
                # as noise / not carried forward when memory or gleaning is off),
                # not a recoverable case — log it transparently and count it rather
                # than aborting the whole graph build over a handful of bad edges.
                logger.warning(
                    "Dropping dangling edge %s: missing node %s -> %s",
                    edge.get("edge_id"), source, target,
                )
                dropped_edges += 1
                continue
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
    if dropped_edges:
        logger.warning(
            "build_unified_graph: dropped %d dangling edge(s) referencing missing nodes",
            dropped_edges,
        )
    return {
        "nodes": len(registry.entities),
        "edges": edge_count,
        "dropped_edges": dropped_edges,
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
    total_nodes = graph.number_of_nodes()
    isolated_count = sum(1 for n in graph.nodes if graph.degree(n) == 0)
    isolated_ratio = isolated_count / max(total_nodes, 1)
    if isolated_ratio > 0.5:
        logger.warning(
            "validate_unified_graph: %.1f%% isolated nodes (%d/%d)",
            isolated_ratio * 100, isolated_count, total_nodes,
        )
    return {
        "valid": not provisional_nodes and not dangling_edges and not missing_provenance,
        "nodes": total_nodes,
        "edges": graph.number_of_edges(),
        "provisional_nodes": provisional_nodes,
        "dangling_edges": dangling_edges,
        "edges_missing_provenance": missing_provenance,
        "isolated_nodes": isolated_count,
        "isolated_ratio": round(isolated_ratio, 4),
    }

