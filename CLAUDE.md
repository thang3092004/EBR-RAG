# EBR-RAG — Evidence-Based Retrieval RAG for Long-Video QA

## Architecture

```
Video files ──► UnifiedIngestPipeline (9 stages, resumable)
                      │
              ├─ MiniCPM-V-2_6-int4 visual captioning (concise, no transcript)
              ├─ GPT-4o-mini entity + relationship extraction (GraphRAG-style)
              ├─ Progressive entity memory across batches + cross-video dedup
              │
              Unified MultiDiGraph (NetworkX GraphML)
              + NanoVectorDB (text chunks, entities, visual segments)
                      │
              EBR_RAG_answer() ──► 3-channel retrieval + RRF/MMR fusion
                      │
              Debate: Generator → Critique (blinded) → Defender (tools) → Judge
                      │
                Final answer (open-ended or MCQ)
```

## Pipeline Stages (9 stages)

```
probe → asr → shot_detection → segmentation → frame_selection
  → alignment_caption (MiniCPM caption + GPT-4o-mini extraction + gleaning)
  → graph_build → embedding_index → validation
```

- **No YOLO tracking** — removed entirely, entity extraction via LLM
- **No spaCy NER** — removed, GPT-4o-mini extracts from both captions and transcripts
- **Captions preserved on restart** via `preserve_on_restart=("captions.json",)`

## Key Entry Points

| Task | File | Function / Class |
|---|---|---|
| Ingest videos | `videorag/videorag.py` | `VideoRAG.insert_video()` |
| Query (EBR-RAG) | `videorag/pipeline/EBR_RAG.py` | `EBR_RAG_answer()` |
| Query (baseline) | `videorag/videorag.py` | `VideoRAG.aquery(mode="videorag")` |
| Unified V3 pipeline | `videorag/pipeline/unified_ingest.py` | `UnifiedIngestPipeline` |
| Ablation config | `videorag/ablation.py` | `INGESTION_PROFILES`, `QUERY_SCENARIOS` |
| Run ablation matrix | `reproduce/run_ablation_matrix.py` | CLI script |
| Ground truth gen | `reproduce/generate_ground_truth.py` | GPT-4o reference answers |

## Quick Commands

```bash
# Baseline must run FIRST (separate command, avoids CUDA fork issue)
python reproduce/run_ablation_matrix.py ingest \
  --collections 0 6 11 --ingestion-profiles video_rag_baseline

# Then unified profiles (baseline skipped, full_framework runs first)
python reproduce/run_ablation_matrix.py ingest \
  --collections 0 6 11 \
  --ingestion-profiles full_framework no_adaptive_segmentation \
  no_entity_memory no_gleaning no_crossmodal_alignment

# Query evaluation
python reproduce/run_ablation_matrix.py query --collections 0 6 11

# Resume after crash: re-run same command (completed stages skipped)
```

## Critical Config Flags

| Flag | Default | Purpose |
|---|---|---|
| `use_unified_graph` | `False` | **Must be True** for V3 pipeline |
| `entity_source` | `"caption"` | `"caption"` = LLM extraction (V3) |
| `crossmodal_batch_size` | `8` | Segments per GPT-4o-mini batch |
| `entity_memory_max_context` | `25` | Max entities in prompt memory |
| `extraction_gleaning_rounds` | `1` | Gleaning retry rounds (0 = off) |
| `disable_crossmodal_alignment` | `False` | A5: skip GPT extraction entirely |
| `segmentation_strategy` | `"adaptive"` | A2: set to `"fixed"` |

## Ablation Matrix (A0–A8)

### Ingestion profiles (5 profiles, each creates separate artifacts)

| ID | Profile | Changes from full |
|---|---|---|
| A0 | `video_rag_baseline` | Original VideoRAG (no graph, no debate) |
| A1 | `full_framework` | All features enabled (default) |
| A2 | `no_adaptive_segmentation` | Fixed 30s segments |
| A3 | `no_entity_memory` | `entity_memory_max_context=0` |
| A4 | `no_gleaning` | `extraction_gleaning_rounds=0` |
| A5 | `no_crossmodal_alignment` | Skip GPT extraction entirely |

### Query scenarios (9 scenarios, A6-A8 share A1 artifacts)

| ID | Scenario | Artifacts | Changes |
|---|---|---|---|
| A6 | `no_debate_refinement` | A1 | `max_rounds=0` |
| A7 | `critique_sees_evidence` | A1 | Critique sees evidence (not blinded) |
| A8 | `defender_no_tools` | A1 | Defender cannot use retrieval tools |

## Debate Architecture (DRAG/AceMAD-inspired)

- **Generator**: Produces initial draft from evidence pool
- **Critique** (BLINDED): No evidence access, uses general knowledge to find gaps
  - Independent sketch → gap analysis → severity labels
  - Early stopping when no significant flaws detected
- **Defender**: Has evidence + retrieval tools, addresses each critique point
  - `search_text_evidence`, `search_visual_segment`, `search_graph_evidence`
  - `tool_top_k=3`, `max_tool_calls_per_round=3`
- **Judge**: Synthesizes final answer, discards debunked claims

## Source of Truth Rules

- **Active benchmark**: `longervideos/dataset.json`, collections 0, 6, 11 (8 videos, 53 questions)
- **CG-Bench** files exist but are **not used**
- Never change config flags between controlled ablations except the one being measured
- Baseline VideoRAG code in `videorag/videorag.py` (use_unified_graph=False path) must match original paper
- YOLO tracking code preserved in branch `yolo-tracking-legacy` for reference
