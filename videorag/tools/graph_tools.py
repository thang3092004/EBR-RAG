"""
videorag/tools/graph_tools.py
=============================
Author's original graph retrieval logic wrapped for EBR-RAG.
"""
from typing import Any, List, Optional
from ..debate.evidence_types import EvidenceItem
from .formatters import (
    _score_from_result,
    make_graph_packet_evidence,
    make_segment_evidence,
    make_entity_evidence,
)
from .._op import _refine_entity_retrieval_query, _find_most_related_segments_from_entities
from .._utils import encode_string_by_tiktoken, logger
import asyncio


def _get_segment_payload(video_segments, segment_id: str) -> dict | None:
    if video_segments is None:
        return None
    video_name, _, segment_index = str(segment_id).rpartition("_")
    video_data = getattr(video_segments, "_data", {}).get(video_name, {})
    payload = video_data.get(segment_index, {}) if isinstance(video_data, dict) else {}
    if not isinstance(payload, dict):
        return None
    return {
        **payload,
        "video_name": video_name,
        "segment_index": segment_index,
    }


async def _search_unified_graph(
    query: str,
    entities_vdb,
    kg,
    video_segments,
    top_k: int,
    global_config: dict,
) -> List[EvidenceItem]:
    from .._unified_graph.retrieval import retrieve_graph_packets

    packets = await retrieve_graph_packets(
        query,
        entities_vdb,
        kg,
        top_k=top_k,
        seed_k=int(global_config.get("graph_seed_k", 4)),
        restart_probability=float(
            global_config.get("graph_restart_probability", 0.15)
        ),
        max_path_length=int(global_config.get("graph_max_path_length", 2)),
        fallback_path_length=int(
            global_config.get("graph_fallback_path_length", 3)
        ),
    )
    token_cap = int(global_config.get("graph_context_token_cap", 1800))
    used_tokens = 0
    evidence = []
    for packet in packets:
        payloads = [
            payload
            for segment_id in packet.get("provenance_segments", [])
            if (payload := _get_segment_payload(video_segments, segment_id))
        ]
        item = make_graph_packet_evidence(packet, payloads)
        token_count = len(encode_string_by_tiktoken(item.snippet))
        if evidence and used_tokens + token_count > token_cap:
            continue
        if token_count > token_cap:
            item.snippet = item.snippet[: max(400, token_cap * 3)]
            token_count = len(encode_string_by_tiktoken(item.snippet))
        evidence.append(item)
        used_tokens += token_count
    return evidence

async def search_graph_evidence(
    query: str,
    stores: dict,
    top_k: int = 8,
    global_config: Optional[dict] = None,
    query_param=None,
) -> List[EvidenceItem]:
    """Retrieve evidence from the Original/TM Graph using author's logic."""
    entities_vdb = stores.get("entities_vdb")
    kg = stores.get("knowledge_graph")
    text_chunks_db = stores.get("text_chunks")
    video_segments = stores.get("video_segments")

    if entities_vdb is None or kg is None:
        return []

    if getattr(kg, "_graph", None) is not None and kg._graph.is_multigraph():
        return await _search_unified_graph(
            query,
            entities_vdb,
            kg,
            video_segments,
            top_k,
            global_config or {},
        )

    # 1. Refine query for entities
    entity_query = query
    if global_config and query_param:
        try:
            entity_query = await _refine_entity_retrieval_query(query, query_param, global_config)
        except Exception:
            pass

    # 2. Search entities in VDB
    entity_results = await entities_vdb.query(entity_query, top_k=top_k)
    if not entity_results:
        return []

    # 3. Get node data and find related segments (Author's logic)
    raw_nodes = await asyncio.gather(
        *[kg.get_node(r.get("entity_name", r.get("id"))) for r in entity_results]
    )
    
    node_datas = []
    for res, node in zip(entity_results, raw_nodes):
        if node:
            # Ensure entity_name is present in the dict for _find_most_related_segments_from_entities
            node_copy = dict(node)
            node_copy["entity_name"] = res.get("entity_name", res.get("id"))
            node_datas.append(node_copy)
    
    # We'll return both entities and segments as evidence
    evidence: List[EvidenceItem] = []
    
    # Add entities
    for node in node_datas:
        # For the formatter, we need a 'result' dict that has the score/id
        # We find the original result match
        res_match = next((r for r in entity_results if r.get("entity_name", r.get("id")) == node["entity_name"]), {})
        evidence.append(make_entity_evidence(res_match, node))

    # Add segments (using _find_most_related_segments_from_entities)
    # This matches the 'skimming' behaviour in NaiveRAG
    try:
        related_seg_ids = await _find_most_related_segments_from_entities(
            top_k, 
            node_datas, 
            text_chunks_db, 
            kg
        )
        
        for seg_id in related_seg_ids:
            video_name = "_".join(str(seg_id).split("_")[:-1])
            seg_idx = str(seg_id).split("_")[-1]
            
            segment_payload = {}
            if video_segments:
                video_data = getattr(video_segments, "_data", {}).get(video_name, {})
                segment_payload = video_data.get(seg_idx, {}) if isinstance(video_data, dict) else {}
                if isinstance(segment_payload, dict):
                    segment_payload = {**segment_payload, "video_name": video_name, "segment_index": seg_idx}
            
            seed_score = max(
                (_score_from_result(result) for result in entity_results),
                default=0.0,
            )
            evidence.append(
                make_segment_evidence(
                    {"id": seg_id, "similarity": seed_score * 0.8},
                    segment_payload,
                )
            )
    except Exception as e:
        logger.warning(f"[search_graph_evidence] segment extraction failed: {e}")

    return evidence
