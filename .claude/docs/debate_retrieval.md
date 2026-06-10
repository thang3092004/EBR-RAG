# EBR-RAG Debate & Retrieval

## Query Entry Point

`videorag/pipeline/EBR_RAG.py` — `EBR_RAG_answer(vrag, query, param)`

Called from `VideoRAG.aquery()` when `param.mode == "EBR_RAG"`.  
Returns a dict: `{answer, rationale, confidence, citations, transcript, evidence, rounds_run, tool_calls_made}`.

## QueryParam Fields (controls all EBR behavior)

`videorag/base.py` — `@dataclass class QueryParam`

| Field | Default | Purpose |
|---|---|---|
| `mode` | `"global"` | Must be `"EBR_RAG"` to trigger the debate pipeline |
| `initial_text_k` | `4` | Text retrieval initial top-k |
| `initial_graph_k` | `4` | Graph retrieval initial top-k |
| `initial_visual_k` | `4` | Visual retrieval initial top-k |
| `max_evidence` | `16` | Hard cap on evidence pool size throughout debate |
| `max_rounds` | `2` | Number of Critique → Defender cycles |
| `max_tool_calls_per_round` | `2` | Max tool calls Defender may make per round |
| `max_total_tool_calls` | `4` | Hard cap on total Defender tool calls |
| `graph_context_token_cap` | `1800` | Max tokens for graph context string |
| `debate_critique_see_evidence` | `False` | A7 ablation: set True to expose evidence to Critique |
| `debate_defender_disable_tools` | `False` | A6-adjacent: disable Defender tool use |
| `return_detailed` | `False` | Set True to get full metadata dict instead of plain string |
| `wo_reference` | `True` | Judge prompt mode (without reference answer) |

## Stage 1: 3-Channel Initial Retrieval

Three channels run in parallel:

```python
text_ev   = await search_text_evidence(query, stores, top_k=text_k, ...)
graph_ev  = await search_graph_evidence(query, stores, top_k=graph_k, ...)
visual_ev = await search_visual_segment(query, stores, top_k=visual_k, ...)
```

### Text Channel (`videorag/tools/text_tools.py`)
- Dense retrieval from `chunks_vdb` (NanoVectorDB, text-embedding-3-small)
- Also retrieves entity seeds from `entities_vdb` for graph expansion
- Returns `EvidenceItem` objects with `type="text"`

### Graph Channel (`videorag/tools/graph_tools.py`)
- Seeds: top entity matches from `entities_vdb`
- Scores edges by confidence × predicate-query token overlap
- Personalized PageRank on the collapsed `DiGraph` (multi-edges merged, reverse edge weight × 0.35)
- BFS paths up to `graph_max_path_length=2` hops; fallback to 3 hops
- Scores graph packets by seed relevance + edge relevance + propagation + temporal coherence
- Context capped at `graph_context_token_cap=1800` tokens
- Returns `EvidenceItem` objects with `type="graph"` and `provenance_path`

### Visual Channel (`videorag/tools/vision_tools.py`)
- ImageBind text-to-video embedding retrieval from `video_segment_feature_vdb`
- Returns `EvidenceItem` objects with `type="segment"` (raw captions at this stage)

## Stage 1b: Query-Aware Re-Captioning (Refinement)

After initial retrieval, visual segment evidence is re-captioned with MiniCPM-V using 15 frames and extracted query keywords (`_extract_keywords_query`). Only items that made the cut (within `initial_budget_limit`) are re-captioned. This is the expensive VLM step.

## Fusion: RRF + Lexical MMR

`videorag/tools/fusion.py` — `reciprocal_rank_fusion(channels, limit, rrf_constant=60, mmr_lambda=0.75)`

1. Compute RRF score for each evidence item across all channels: `1 / (60 + rank)`.
2. Apply lexical MMR (marginal relevance): `score = lambda * relevance - (1-lambda) * max_similarity_to_selected`.
3. Similarity is Jaccard on token bags (`id + snippet + provenance_path`).
4. Items are greedily selected by MMR score until `limit` is reached.
5. Final pool is truncated to `max_evidence=16`.

## Stage 2–4: The Debate Loop

`videorag/debate/debate_manager.py` — `run_debate()`

### Role Configs (`videorag/agents/roles.py`)
Each role can have a distinct model, temperature, and max_tokens.
Default all use `gpt-4o-mini`. Configure in `ROLE_CONFIGS` dict.

### Generator (Stage 2)
- Sees: query + full initial evidence pool
- Uses: `GENERATOR_PROMPT_OPEN` or `GENERATOR_PROMPT_MCQ`
- Produces: initial draft answer (plain text or MCQ analysis)
- Single LLM call, no tools

### Critique (Stage 3a, each round)
- Sees: query + current draft + debate history (last 10 messages)
- **Default**: does NOT see evidence pool (blinded) — this is the key ablation flag `A7`
- Uses: `CRITIQUE_PROMPT_OPEN` or `CRITIQUE_PROMPT_MCQ`
- Produces: list of flaws, gaps, unsupported claims
- Single LLM call, no tools

### Defender (Stage 3b, each round)
- Sees: query + current draft + critique + full evidence pool
- Has access to tool calls: `search_text_evidence`, `search_visual_segment`, `search_graph_evidence`
- Each tool call retrieves 1 item at a time; new items are appended to `state.evidence` if not already present and pool is below `max_evidence`
- Tool loop runs while `calls_made < max_tool_calls_per_round AND total_calls < max_total_tool_calls AND pool_size < max_evidence`
- Produces: updated draft with `"Updated Draft"` section extracted as next round's draft
- `A6 ablation`: `max_rounds=0` skips all Critique/Defender rounds entirely

### Judge (Stage 4)
- Sees: query + compact transcript (last 50 messages) + full evidence pool (up to 50 items)
- Open-ended: returns raw natural language answer directly
- MCQ: returns structured JSON `{Answer, Explanation, Confidence}` with up to 3 retries to parse valid JSON
- Citations are cross-validated against `state.evidence` IDs

## Evidence Pool Invariants

- Pool starts with initial RRF/MMR result (≤ 16 items).
- Pool is **append-only** during debate (no removal).
- Deduplication by `EvidenceItem.id` (no duplicate IDs).
- Hard cap: `max_evidence=16` — Defender cannot add items beyond this.
- `EvidenceItem.validated=True` is set by Judge citation validation.

## MCQ Detection

`_extract_mcq_options(query)` scans for patterns like `(A) ... (B) ...`.
If found, `is_mcq=True` switches all prompts to MCQ variants and the Judge returns structured JSON.

## LLM Client

`get_openai_async_client_instance()` from `videorag/_llm.py`.  
Falls back to `AsyncOpenAI(api_key=env, base_url=env)` if that raises.  
All debate `_chat()` calls use `tenacity` retry: 12 attempts, exponential backoff 4–60s on `RateLimitError` or `APIConnectionError`.
