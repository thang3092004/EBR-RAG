# Unified V2 Ingestion Pipeline

## Overview

Entry: `VideoRAG.insert_video(video_path_list)` with `use_unified_graph=True`  
Orchestrator: `videorag/pipeline/unified_ingest.py` — `UnifiedIngestPipeline`  
Stage definitions: `videorag/pipeline/unified_stages.py` — `UNIFIED_STAGE_DEFINITIONS`  
Resume/checkpoint engine: `videorag/pipeline/stage_runner.py` — `StageRunner`

## 11 Stages in Order

```
probe → asr → shot_detection → segmentation → tracking_base
     → frame_selection → text_entities → alignment_caption
     → graph_build → embedding_index → validation
```

Dependencies are explicit in `UNIFIED_STAGE_DEFINITIONS`. If a stage's config hash changes, `StageRunner` automatically invalidates that stage and all downstream descendants.

### Stage Details

| Stage | Key Input | Key Output | Notes |
|---|---|---|---|
| `probe` | video file | `probe.json` (duration, fps, resolution, has_audio) | Uses FFprobe + OpenCV; strict mode raises on unreadable file |
| `asr` | probe.json | `asr.json` (words with timestamps, utterances) | `faster-distil-whisper-large-v3`; no-audio video gets empty transcript, not an error unless strict |
| `shot_detection` | probe.json | `shots.json` (boundaries, motion samples) | HSV histogram diff + grayscale motion + PySceneDetect |
| `segmentation` | asr.json + shots.json | `segments.json` (list of segments with start/end/index) | `adaptive` = dynamic programming ~24s target; `fixed` = 30s fixed (A2 ablation). Adds `storage_id = f"{video_id}_{index}"` |
| `tracking_base` | probe.json + segments.json | `observations.json`, `tracklets.json`, `visual_entities.json`, `registry.json` | YOLOv8n detect + BoT-SORT track at ~3 FPS; OpenCLIP embeds each tracklet crop; disjoint-set union links cross-tracklet |
| `frame_selection` | segments + shots + observations | `frame_selections.json`, per-segment `segments/{id}.json` | 2-6 frames per segment; scores diversity, new entity coverage, temporal-bin coverage, image quality |
| `text_entities` | asr.json + segments.json | `text_entities.json`, `text_memory.json` | spaCy NER + noun chunks + pronoun/event rules; discourse memory resolves coreference; resumable via `checkpoint.json` |
| `alignment_caption` | frame_selections + text_entities + tracking_base | `alignment.json` (per-segment captions, edges, registry) | Phase 1: MiniCPM-V captions frames with entity-ID overlays. Phase 2: OpenCLIP text-visual candidate matching → MiniCPM decides merges. Merges global registry. |
| `graph_build` | alignment.json | graph written to `chunk_entity_relation_v2.graphml` | `build_unified_graph()` writes nodes + edges to `UnifiedNetworkXStorage` |
| `embedding_index` | graph_build + alignment + segments | VDB updated, chunks indexed | Extracts video clips, chunks segment text (~1200 tokens), upserts to `chunks_vdb`, `entities_vdb`, `video_segment_feature_vdb` |
| `validation` | graph + alignment | `validation_report.json` | `validate_unified_graph()` checks node/edge provenance; fails pipeline if `valid=False` |

## Working Directory Layout

```
<working_dir>/
├── pipeline_v2/
│   ├── global_registry.json           # Cross-video entity registry (atomic writes)
│   └── <video_id>/
│       ├── manifest.json              # Stage status, config hashes, metrics
│       ├── progress.json              # Live progress (last write ≤ 0.5s ago)
│       ├── events.jsonl               # Append-only event log
│       ├── run_report.json            # Final run summary
│       ├── pipeline.log               # Per-video file logger
│       ├── probe/probe.json
│       ├── asr/asr.json
│       ├── shot_detection/shots.json
│       ├── segmentation/segments.json
│       ├── tracking_base/
│       │   ├── observations.json
│       │   ├── tracklets.json
│       │   ├── visual_entities.json
│       │   └── registry.json
│       ├── frame_selection/
│       │   ├── frame_selections.json
│       │   └── frames/               # Extracted frame images
│       ├── text_entities/
│       │   ├── text_entities.json
│       │   ├── text_memory.json
│       │   └── checkpoint.json       # Resume pointer (next_index + extractor state)
│       ├── alignment_caption/
│       │   └── alignment.json
│       ├── graph_build/graph_stats.json
│       ├── embedding_index/           # Index metrics
│       └── validation/validation_report.json
├── video_segments_v2.json             # KV: video_id → segment content map
├── video_path_v2.json
├── text_chunks_v2.json
├── llm_response_cache_v2.json
├── chunks_v2.json / chunks_v2/        # NanoVectorDB text chunks
├── entities_v2.json / entities_v2/   # NanoVectorDB entities
├── video_segment_feature_v2/          # NanoVectorDB ImageBind vectors
└── chunk_entity_relation_v2.graphml   # Unified directed graph
```

## Resume / Invalidation Logic (`StageRunner`)

- Each stage has a `config_hash` computed from its declared `config_keys` values.
- On startup, `_prepare_manifest()` compares stored vs. current hashes.
- If a hash differs, `_invalidate()` wipes that stage's output directory and marks all downstream stages as `pending`.
- `should_run(stage)` returns `False` if `manifest["stages"][stage]["status"] == "done"`.
- `restart_stage="asr"` explicitly invalidates ASR and all descendants.
- `force=True` wipes everything and starts fresh.
- Interrupted stages leave `status="running"` in the manifest — on resume they re-run from scratch (idempotent within a stage).
- `text_entities` stage is internally resumable via `checkpoint.json` (saves `next_index` + extractor memory state).

## Global Registry Merge (`_merge_global_registry`)

Called after `alignment_caption` completes for each video. It:
1. Loads `global_registry.json` (shared across all videos).
2. Removes all aliases and provenance that belonged to the just-processed `video_id`.
3. Re-inserts entities from the fresh `alignment.json["registry"]`.
4. Writes back atomically.

This makes re-ingestion of a single video safe — stale entities from previous runs of the same video are cleaned up.

## Strict Mode

`pipeline_strict=True` turns all soft warnings into hard errors:
- Missing local MiniCPM model directory → `FileNotFoundError`
- No audio stream → `RuntimeError`
- OpenCLIP embeddings missing for any tracklet → `RuntimeError`
- Graph validation failure → `RuntimeError`

Use `pipeline_strict=False` for development; always use `True` for controlled ablation runs.
