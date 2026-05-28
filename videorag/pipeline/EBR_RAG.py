"""
EBR-RAG Pipeline Orchestrator — full rewrite.

EBR_RAG_answer() is the top-level entry point called from VideoRAG.aquery()
when param.mode == "EBR_RAG".

Improvements over previous version:
- threads global_config (LLM + retrieval functions) into retrieval tools
- reads max_rounds and ebr_top_k from QueryParam
- dispatch_tool passes global_config + query_param for LLM query rewriting
- returns richer output: confidence, rounds_run, tool_calls_made
"""
import asyncio
import os
import re
from dataclasses import asdict
from typing import Any, List

from openai import AsyncOpenAI

from ..tools.text_tools import search_text_evidence
from ..tools.vision_tools import search_visual_segment
from ..tools.graph_tools import search_graph_evidence
from ..tools.schemas import ALL_TOOLS
from ..debate.debate_manager import run_debate
from ..debate.state import DebateConfig, DebateState
from ..debate.evidence_types import EvidenceItem
from ..agents.roles import ROLE_CONFIGS
from .._llm import get_openai_async_client_instance
from .._utils import logger


def _extract_mcq_options(query: str) -> List[dict]:
    """Helper to extract options from queries like '(A) apple (B) banana'."""
    options = []
    # Matches patterns like (A) text or A) text
    pattern = r'(?:\(?([A-D])[\)\.])\s*([^\(\n]+)'
    matches = re.findall(pattern, query)
    for label, text in matches:
        options.append({
            "id": f"Option {label}",
            "claim": f"{label}) {text.strip()}",
            "reasoning": f"Official option {label} from the multiple-choice query."
        })
    return options


def _dedup_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[str] = set()
    deduped: list[EvidenceItem] = []
    for ev in items:
        if ev.id in seen:
            continue
        seen.add(ev.id)
        deduped.append(ev)
    return deduped


async def EBR_RAG_answer(vrag, query: str, param) -> dict:
    """
    Iterative Evidence-Verification RAG (EBR-RAG pipeline).

    Args:
        vrag:   VideoRAG instance (provides all storage handles + global_config)
        query:  User question
        param:  QueryParam — controls ebr_top_k, max_rounds

    Returns:
        {answer, rationale, confidence, citations, transcript,
         evidence, rounds_run, tool_calls_made}
    """
    # global_config is the full asdict(vrag) dict — contains LLM functions, retrieval params
    global_config: dict = asdict(vrag) if hasattr(vrag, "__dataclass_fields__") else {}

    # --- Build store handles & config ---
    stores: dict[str, Any] = {
        "chunks_vdb":               getattr(vrag, "chunks_vdb", None),
        "text_chunks":              getattr(vrag, "text_chunks", None),
        "entities_vdb":             getattr(vrag, "entities_vdb", None),
        "knowledge_graph":          getattr(vrag, "chunk_entity_relation_graph", None),
        "video_segment_feature_vdb": getattr(vrag, "video_segment_feature_vdb", None),
        "video_segments":           getattr(vrag, "video_segments", None),
        "video_path_db":            getattr(vrag, "video_path_db", None),
        # VLM for re-captioning
        "caption_model":            getattr(vrag, "caption_model", None),
        "caption_tokenizer":        getattr(vrag, "caption_tokenizer", None),
        # Embedding func exposed so tools can embed the query if needed
        "text_embedding_func":      getattr(getattr(vrag, "embedding_func", None), "func", None)
                                    or getattr(vrag, "embedding_func", None),
    }

    # -------------------------------------------------------------------------
    # RE-CAPTIONING HELPER (Query-Aware Refinement)
    # -------------------------------------------------------------------------
    async def refine_segment_evidence(evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
        """Performs query-aware re-captioning for visual segments (Matching Baseline)."""
        # Filter for unrefined segments
        target_evs = [ev for ev in evidence_list if ev.type == "segment" and not ev.metadata.get("refined")]
        if not target_evs:
            return evidence_list
        
        c_model = stores.get("caption_model")
        c_tok = stores.get("caption_tokenizer")
        if not c_model or not c_tok:
            return evidence_list

        from .._videoutil.caption import retrieved_segment_caption
        from .._op import _extract_keywords_query
        
        id_map = {ev.id: ev for ev in target_evs}
        clean_ids = list(id_map.keys())

        # Extract keywords for query-aware focus
        keywords = await _extract_keywords_query(query, param, global_config)
        logger.info(f"[EBR_RAG] Refining {len(clean_ids)} segments with keywords: {keywords!r}")
        
        try:
            refined = retrieved_segment_caption(
                c_model, c_tok, keywords, clean_ids,
                stores["video_path_db"], stores["video_segments"],
                num_sampled_frames=global_config.get('fine_num_frames_per_segment', 15)
            )
            for cid, caption_content in refined.items():
                if cid in id_map:
                    id_map[cid].snippet = caption_content
                    id_map[cid].metadata["refined"] = True
        except Exception as e:
            logger.error(f"[EBR_RAG] Refinement failed: {e}")
            
        return evidence_list

    # -------------------------------------------------------------------------
    # UNIFIED EVIDENCE BUDGETING (Matching Baseline + Debate Bonus)
    # -------------------------------------------------------------------------
    max_rounds: int = getattr(param, "max_rounds", 3)
    base_top_k: int = 8  # Unified Top-K matching baseline
    
    # Stage 1 Retrieval Targets
    text_k, graph_k, visual_k = base_top_k, base_top_k, base_top_k
    global_config["retrieval_topk_chunks"] = text_k

    # --- Stage 1: Initial retrieval ---
    init_evidence = []
    
    # Text evidence
    logger.info(f"[EBR_RAG] Stage 1a — Text retrieval top_k={text_k}")
    text_ev = await search_text_evidence(query, stores, top_k=text_k, entity_boost=True,
                                         global_config=global_config, query_param=param)
    
    # Graph evidence
    logger.info(f"[EBR_RAG] Stage 1b — Graph retrieval top_k={graph_k}")
    graph_ev = await search_graph_evidence(query, stores, top_k=graph_k, 
                                          global_config=global_config, query_param=param)

    # Visual evidence (Baseline ImageBind)
    logger.info(f"[EBR_RAG] Stage 1c — Visual retrieval (Baseline) top_k={visual_k}")
    visual_ev = await search_visual_segment(query, stores, top_k=visual_k,
                                           global_config=global_config, query_param=param)

    init_evidence = _dedup_evidence(text_ev + graph_ev + visual_ev)

    # --- INITIAL TRUNCATION (Pre-Refinement Optimization) ---
    # We limit to the initial budget (24) BEFORE calling the expensive VLM
    initial_budget_limit = text_k + graph_k + visual_k # 24
    if len(init_evidence) > initial_budget_limit:
        logger.info(f"[EBR_RAG] Truncating pool {len(init_evidence)} -> {initial_budget_limit} BEFORE refinement.")
        init_evidence = init_evidence[:initial_budget_limit]

    # --- REFINEMENT (Stage 1) ---
    # Only re-caption the items that actually made the cut
    init_evidence = await refine_segment_evidence(init_evidence)

    # --- FINAL UNIVERSAL CAP ---
    # 24 (Initial) + 9 (Debate: 3 rounds * 3 items) = 33
    universal_cap = initial_budget_limit + (max_rounds * 3)
    if len(init_evidence) > universal_cap:
        init_evidence = init_evidence[:universal_cap]

    # --- Build LLM client ---
    model_name: str = getattr(getattr(vrag, "llm", None), "best_model_name", "gpt-4o-mini")
    try:
        llm_client: AsyncOpenAI = get_openai_async_client_instance()
    except TypeError:
        llm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )

    debate_cfg = DebateConfig(
        model=model_name,
        max_rounds=max_rounds,
        tool_top_k=1, # Defender retrieves 1 item per call
        critique_see_evidence=getattr(param, "debate_critique_see_evidence", False),
        defender_disable_tools=getattr(param, "debate_defender_disable_tools", False),
        single_hypothesis=getattr(param, "debate_single_hypothesis", False),
    )

    # --- Tool dispatcher ---
    async def dispatch_tool(name: str, args: dict, evidence_pool: list[EvidenceItem]) -> dict:
        q: str = args.get("query", query)
        # Unified limit: 1 item per call during debate
        tk = 1
        
        call_config = dict(global_config)
        call_config["retrieval_topk_chunks"] = tk
        evs = []

        if name == "search_text_evidence":
            evs = await search_text_evidence(q, stores, top_k=tk, global_config=call_config, query_param=param)

        elif name == "search_visual_segment":
            evs = await search_visual_segment(q, stores, top_k=tk, global_config=call_config, query_param=param)

        elif name == "search_graph_evidence":
            evs = await search_graph_evidence(q, stores, top_k=tk, global_config=call_config, query_param=param)

        else:
            return {"error": f"unknown tool: {name}"}

        # Refine only the newly found segment
        evs = await refine_segment_evidence(evs)
        return {"evidence": evs}

    # --- Detect MCQ ---
    mcq_options = _extract_mcq_options(query)
    is_mcq = len(mcq_options) > 0
    if is_mcq:
        logger.info(f"[EBR_RAG] MCQ detected with {len(mcq_options)} options")

    # --- Stage 2-4: Multi-agent debate ---
    logger.info(f"[EBR_RAG] Starting debate: max_rounds={max_rounds}, budget={len(init_evidence)} initial, cap={universal_cap}")
    verdict, state = await run_debate(
        query=query,
        initial_evidence=init_evidence,
        llm_client=llm_client,
        tools=ALL_TOOLS,
        dispatch_tool=dispatch_tool,
        cfg=debate_cfg,
        role_cfgs=ROLE_CONFIGS,
        is_mcq=is_mcq,
        forced_hypotheses=mcq_options if is_mcq else None,
        wo_reference=param.wo_reference
    )
    # Ensure verdict is a dict (LLM might return a raw string for open-ended questions - baseline style)
    if isinstance(verdict, str):
        logger.info(f"[EBR_RAG] Verdict is a raw text string (baseline open-ended style). Wrapping.")
        verdict = {"answer": verdict, "rationale": "Synthesized from debate.", "confidence": 1.0}

    logger.info(f"[EBR_RAG] Debate complete — "
                f"rounds={state.rounds_run}, tool_calls={state.tool_calls_made}, "
                f"evidence_pool={len(state.evidence)}, "
                f"confidence={verdict.get('confidence', 'N/A')}")

    return {
        "answer":          verdict.get("answer", ""),
        "rationale":       verdict.get("rationale", ""),
        "confidence":      verdict.get("confidence", 0.0),
        "citations":       verdict.get("citations", []),
        "transcript":      state.transcript,
        "evidence":        [ev.to_dict() for ev in state.evidence],
        "rounds_run":      state.rounds_run,
        "tool_calls_made": state.tool_calls_made,
    }
