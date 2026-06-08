from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import networkx as nx


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def _result_score(result: dict[str, Any]) -> float:
    if "similarity" in result:
        return max(0.0, min(float(result["similarity"]), 1.0))
    if "distance" in result:
        return max(0.0, min(1.0 - float(result["distance"]), 1.0))
    metrics = result.get("__metrics__")
    if metrics is not None:
        return max(0.0, min(1.0 - float(metrics), 1.0))
    return 0.0


def _edge_relevance(data: dict[str, Any], query_tokens: set[str]) -> float:
    predicate_tokens = _tokens(
        str(data.get("predicate", "")) + " " + str(data.get("description", ""))
    )
    relation_match = (
        len(predicate_tokens & query_tokens) / max(len(predicate_tokens), 1)
        if predicate_tokens
        else 0.0
    )
    confidence = float(data.get("confidence", data.get("weight", 0.5)) or 0.5)
    return max(1e-4, 0.65 * confidence + 0.35 * relation_match)


def _collapsed_graph(graph: nx.MultiDiGraph, query: str) -> nx.DiGraph:
    query_tokens = _tokens(query)
    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        relevance = _edge_relevance(data, query_tokens)
        previous = collapsed.get_edge_data(source, target, {}).get("weight", 0.0)
        collapsed.add_edge(source, target, weight=max(previous, relevance))
        # Add a weaker reverse transition so evidence can be discovered from either end.
        reverse = collapsed.get_edge_data(target, source, {}).get("weight", 0.0)
        collapsed.add_edge(target, source, weight=max(reverse, relevance * 0.35))
    return collapsed


def _best_occurrence(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    query_tokens: set[str],
) -> dict[str, Any] | None:
    candidates = []
    for left, right in ((source, target), (target, source)):
        occurrences = graph.get_edge_data(left, right) or {}
        for edge_id, data in occurrences.items():
            candidates.append(
                {
                    "edge_id": str(edge_id),
                    **data,
                    "source_node": left,
                    "target_node": right,
                    "_score": _edge_relevance(data, query_tokens),
                }
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["_score"])


def _path_packet(
    graph: nx.MultiDiGraph,
    path: list[str],
    seed_scores: dict[str, float],
    pagerank: dict[str, float],
    query_tokens: set[str],
) -> dict[str, Any] | None:
    edges = []
    edge_scores = []
    times = []
    provenance_segments = []
    for source, target in zip(path, path[1:]):
        occurrence = _best_occurrence(graph, source, target, query_tokens)
        if occurrence is None:
            return None
        edge_scores.append(float(occurrence.pop("_score")))
        edges.append(occurrence)
        start = float(occurrence.get("start", -1.0) or -1.0)
        end = float(occurrence.get("end", -1.0) or -1.0)
        if start >= 0:
            times.append((start, end))
        segment_id = (
            occurrence.get("storage_id")
            or occurrence.get("segment_id")
            or occurrence.get("source_id")
        )
        if segment_id and str(segment_id) not in provenance_segments:
            provenance_segments.append(str(segment_id))
    temporal_coherence = 1.0
    if len(times) > 1:
        gaps = [
            max(0.0, times[index + 1][0] - times[index][1])
            for index in range(len(times) - 1)
        ]
        temporal_coherence = math.exp(-sum(gaps) / max(60.0 * len(gaps), 1.0))
    geometric_edge = (
        math.prod(max(score, 1e-4) for score in edge_scores)
        ** (1.0 / max(len(edge_scores), 1))
    )
    seed_relevance = max(seed_scores.get(node, 0.0) for node in path)
    propagation = sum(pagerank.get(node, 0.0) for node in path) / len(path)
    length_penalty = 1.0 / (1.0 + 0.20 * max(len(path) - 2, 0))
    score = (
        0.35 * seed_relevance
        + 0.30 * geometric_edge
        + 0.20 * propagation
        + 0.15 * temporal_coherence
    ) * length_penalty
    return {
        "packet_id": "GRAPH_" + "_".join(path),
        "nodes": path,
        "edges": edges,
        "provenance_segments": provenance_segments[:2],
        "time_span": (
            [min(item[0] for item in times), max(item[1] for item in times)]
            if times
            else None
        ),
        "score": score,
    }


async def retrieve_graph_packets(
    query: str,
    entities_vdb,
    graph_storage,
    *,
    top_k: int = 4,
    seed_k: int = 4,
    restart_probability: float = 0.15,
    max_path_length: int = 2,
    fallback_path_length: int = 3,
) -> list[dict[str, Any]]:
    if not hasattr(graph_storage, "_graph"):
        return []
    graph = graph_storage._graph
    if not isinstance(graph, nx.MultiDiGraph) or graph.number_of_nodes() == 0:
        return []

    entity_results = await entities_vdb.query(query, top_k=seed_k)
    seed_scores = {
        str(result.get("entity_name", result.get("id"))): _result_score(result)
        for result in entity_results
        if str(result.get("entity_name", result.get("id"))) in graph
    }
    if not seed_scores:
        return []
    total = sum(seed_scores.values()) or float(len(seed_scores))
    personalization = {
        node: score / total for node, score in seed_scores.items()
    }
    collapsed = _collapsed_graph(graph, query)
    pagerank = nx.pagerank(
        collapsed,
        alpha=1.0 - restart_probability,
        personalization=personalization,
        weight="weight",
    )
    ranked_targets = [
        node
        for node, _ in sorted(
            pagerank.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if node not in seed_scores
    ][: max(12, top_k * 4)]
    query_tokens = _tokens(query)
    packets = []
    for seed in seed_scores:
        for target in ranked_targets:
            try:
                path = nx.shortest_path(collapsed, seed, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            edge_count = len(path) - 1
            if edge_count > max_path_length:
                continue
            packet = _path_packet(
                graph,
                path,
                seed_scores,
                pagerank,
                query_tokens,
            )
            if packet:
                packets.append(packet)
    if not packets and fallback_path_length > max_path_length:
        for seed in seed_scores:
            for target in ranked_targets:
                try:
                    path = nx.shortest_path(collapsed, seed, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(path) - 1 > fallback_path_length:
                    continue
                packet = _path_packet(
                    graph,
                    path,
                    seed_scores,
                    pagerank,
                    query_tokens,
                )
                if packet:
                    packets.append(packet)

    deduplicated = {}
    for packet in packets:
        signature = tuple(packet["nodes"])
        if (
            signature not in deduplicated
            or packet["score"] > deduplicated[signature]["score"]
        ):
            deduplicated[signature] = packet
    return sorted(
        deduplicated.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]
