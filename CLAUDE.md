# EBR-RAG — Unified Multimodal Entity Graph + Evidence-Based Debate for Long-Video QA

## Source of Truth Rules (read before touching anything)

- **Active framework**: Unified Multimodal Graph V2 (`use_unified_graph=True`). TM-Graph is legacy.
- **Active benchmark**: `longervideos/dataset.json`, collections 0, 6, 11 for ablation (8 videos, 53 questions).
- **CG-Bench** files exist in the repo but are **not used**. Never feed CG-Bench into ingestion or evaluation.
- Never mix TM-Graph results with Unified V2 results in the thesis.
- Never change config flags between controlled ablations except the one flag being measured.
- `pipeline_strict=True` must fail loudly on missing models/dependencies — no silent fallbacks.
- Never lose `provenance`, `timestamp`, `modality`, or `global_entity_id` from any artifact.
- Do not restore deleted test files; add targeted tests for new changes only.

## Architecture at a Glance

```
Video files ──► UnifiedIngestPipeline (11 stages, resumable)
                      │
              Unified MultiDiGraph (NetworkX GraphML)
              + NanoVectorDB (text chunks, entities, visual segments)
                      │
              EBR_RAG_answer() ──► 3-channel retrieval + RRF/MMR fusion
                      │
              Debate: Generator → Critique → Defender (tools) → Judge
                      │
                Final answer (open-ended or MCQ)
```

→ Deep details: [`.claude/docs/architecture.md`](.claude/docs/architecture.md)

## Key Entry Points

| Task | File | Function / Class |
|---|---|---|
| Ingest videos | `videorag/videorag.py` | `VideoRAG.insert_video()` |
| Query (EBR-RAG) | `videorag/pipeline/EBR_RAG.py` | `EBR_RAG_answer()` |
| Query (baseline) | `videorag/videorag.py` | `VideoRAG.aquery(mode="videorag")` |
| Unified V2 pipeline | `videorag/pipeline/unified_ingest.py` | `UnifiedIngestPipeline` |
| Ablation config | `videorag/ablation.py` | `INGESTION_PROFILES`, `QUERY_SCENARIOS` |
| Run full ablation matrix | `reproduce/run_ablation_matrix.py` | CLI script |
| Run baseline | `reproduce/run_baseline_naive.py` | CLI script |

## Quick Commands

```bash
# Full ablation on collections 0, 6, 11
python reproduce/run_ablation_matrix.py --collections 0 6 11

# Baseline only
python reproduce/run_baseline_naive.py --collections 0 6 11

# Resume a stuck pipeline for one video (re-run from a failed stage)
# Set restart_stage= in UnifiedIngestPipeline or pass --restart-stage flag
```

## VideoRAG Config: Critical Flags

| Flag | Default | Purpose |
|---|---|---|
| `use_unified_graph` | `False` | **Must be True** for Unified V2 |
| `pipeline_strict` | `False` | Fail on missing models, no fallbacks |
| `pipeline_resume` | `True` | Skip completed stages on re-run |
| `segmentation_strategy` | `"adaptive"` | A2 ablation sets this to `"fixed"` |
| `disable_transcript_memory` | `False` | A3 ablation sets True |
| `disable_visual_identity_linking` | `False` | A4 ablation sets True |
| `disable_crossmodal_alignment` | `False` | A5 ablation sets True |
| `ablation_profile` | `"full_framework"` | Identifies which artifact set to use |

→ Full flag reference: [`.claude/docs/architecture.md`](.claude/docs/architecture.md)

## Ablation Matrix (A0–A7)

| ID | Scenario key | Mode | Shares artifacts with |
|---|---|---|---|
| A0 | `video_rag_baseline` | `videorag` | separate ingest |
| A1 | `full_framework` | `EBR_RAG` | — |
| A2 | `no_adaptive_segmentation` | `EBR_RAG` | own ingest |
| A3 | `no_transcript_memory` | `EBR_RAG` | own ingest |
| A4 | `no_visual_identity_linking` | `EBR_RAG` | own ingest |
| A5 | `no_crossmodal_alignment` | `EBR_RAG` | own ingest |
| A6 | `no_debate_refinement` | `EBR_RAG` | **A1** |
| A7 | `critique_sees_evidence` | `EBR_RAG` | **A1** |

→ Full protocol: [`.claude/docs/ablation_matrix.md`](.claude/docs/ablation_matrix.md)

## Where to Find Things

| Topic | Doc |
|---|---|
| 11-stage ingestion pipeline, stage resume, artifacts | [`.claude/docs/ingestion_pipeline.md`](.claude/docs/ingestion_pipeline.md) |
| Retrieval (text/graph/visual), RRF, debate roles | [`.claude/docs/debate_retrieval.md`](.claude/docs/debate_retrieval.md) |
| Storage classes, graph schema, entity registry | [`.claude/docs/storage_layer.md`](.claude/docs/storage_layer.md) |
| Full ablation protocol and run scripts | [`.claude/docs/ablation_matrix.md`](.claude/docs/ablation_matrix.md) |
| System architecture, model stack, data flow | [`.claude/docs/architecture.md`](.claude/docs/architecture.md) |
