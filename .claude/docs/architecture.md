# Architecture — EBR-RAG Unified Multimodal Graph

## Two Generations in the Repo

| Generation | Code location | Status |
|---|---|---|
| TM-Graph (legacy) | `videorag/_op.py` + `use_tm_graph=True` | Internship report only. Do not present results as Unified V2. |
| Unified Multimodal Graph V2 | `videorag/pipeline/unified_ingest.py` + `use_unified_graph=True` | **Current, thesis source of truth** |

The `VideoRAG` dataclass selects the generation via two flags:
- `use_unified_graph=True` → Unified V2 (triggers `UnifiedIngestPipeline`)  
- `use_tm_graph=True, use_unified_graph=False` → TM-Graph (legacy, entity extraction uses `extract_entities_tm`)

## Module Map

```
videorag/
├── videorag.py               # VideoRAG dataclass — all config, insert_video(), aquery()
├── base.py                   # Abstract base classes: QueryParam, BaseVectorStorage,
│                             #   BaseKVStorage, BaseGraphStorage, StorageNameSpace
├── ablation.py               # INGESTION_PROFILES, QUERY_SCENARIOS dicts
├── _llm.py                   # LLMConfig, openai_config, client factory
├── _op.py                    # Legacy entity extraction, chunking, baseline query
├── _utils.py                 # Hashing, async helpers, logger
├── prompt.py                 # Legacy LLM prompt templates
│
├── pipeline/
│   ├── unified_ingest.py     # UnifiedIngestPipeline orchestrator (11 stages)
│   ├── unified_stages.py     # UNIFIED_STAGE_DEFINITIONS (stage deps + config keys)
│   ├── stage_runner.py       # StageRunner: resume, invalidation, manifest, events
│   └── EBR_RAG.py            # EBR_RAG_answer(): retrieval + debate entry point
│
├── _videoutil/
│   ├── media_probe.py        # probe_video() — FFprobe + OpenCV checks
│   ├── asr_v2.py             # transcribe_full_video(), assign_words_to_segments()
│   ├── shot_detection.py     # detect_shots_and_motion() — HSV + PySceneDetect
│   ├── smart_segment.py      # smart_segment() + fixed_segment()
│   ├── frame_selector.py     # select_segment_frames() — diversity-aware
│   ├── caption.py            # MiniCPM-V caption helpers
│   ├── feature.py            # ImageBind encoding
│   └── split.py, asr.py      # Legacy baseline split/ASR (non-Unified)
│
├── _entity_anchor/
│   ├── tracker_v2.py         # YOLOv8 + BoT-SORT tracking
│   ├── linker.py             # Cross-tracklet identity linking
│   ├── appearance.py         # OpenCLIP crop embeddings
│   ├── text_entities.py      # TextEntityExtractor (spaCy + rules)
│   ├── text_memory.py        # Discourse memory: short-term + long-term + events
│   ├── memory.py             # Entity memory for baseline entity anchoring
│   ├── overlay.py            # Draw bounding boxes with global entity IDs
│   └── pipeline.py, schema.py
│
├── _unified_graph/
│   ├── schema.py             # EntityNode, EntityAlias, ProvenanceRecord, EdgeData
│   ├── registry.py           # EntityRegistry: allocate IDs, ensure_entity, merge
│   ├── alignment.py          # MiniCPMAligner, align_all_segments()
│   ├── correspondence.py     # OpenCLIPTextEncoder for text-visual candidate matching
│   ├── builder.py            # build_unified_graph(), validate_unified_graph()
│   └── retrieval.py          # Personalized PageRank graph retrieval
│
├── _storage/
│   ├── kv_json.py            # JsonKVStorage
│   ├── vdb_nanovectordb.py   # NanoVectorDBStorage (text/entity), NanoVectorDBVideoSegmentStorage
│   ├── gdb_networkx.py       # NetworkXStorage (legacy undirected)
│   ├── gdb_unified_networkx.py # UnifiedNetworkXStorage (directed MultiDiGraph)
│   └── gdb_neo4j.py          # Neo4j optional backend
│
├── debate/
│   ├── debate_manager.py     # run_debate(): Generator → Critique → Defender → Judge
│   ├── state.py              # DebateConfig, DebateState dataclasses
│   └── evidence_types.py     # EvidenceItem dataclass
│
├── tools/
│   ├── text_tools.py         # search_text_evidence()
│   ├── vision_tools.py       # search_visual_segment()
│   ├── graph_tools.py        # search_graph_evidence()
│   ├── fusion.py             # reciprocal_rank_fusion() + lexical MMR
│   ├── schemas.py            # ALL_TOOLS (OpenAI tool-call schemas)
│   └── formatters.py         # Evidence formatting helpers
│
└── agents/
    ├── agents_prompts.py     # GENERATOR/CRITIQUE/DEFENDER/JUDGE prompt strings
    └── roles.py              # RoleConfig (model, temperature, max_tokens per role)
```

## Model Stack

| Role | Model / Library |
|---|---|
| LLM (all roles) | `gpt-4o-mini` via OpenAI API |
| Text embedding | `text-embedding-3-small` (1536-dim) |
| ASR | `faster-distil-whisper-large-v3` (faster-whisper) |
| Caption + alignment VLM | `MiniCPM-V-2_6-int4` (local, CUDA) |
| Visual retrieval | ImageBind-Huge (1024-dim) — model at `.checkpoints/imagebind_huge.pth` |
| Object detector | YOLOv8n |
| Multi-object tracker | BoT-SORT |
| Appearance similarity | OpenCLIP `ViT-B-32 laion2b_s34b_b79k` |
| NLP / NER | `spacy en_core_web_trf` |
| Video processing | FFmpeg, FFprobe, OpenCV, PySceneDetect, MoviePy |

## Storage Backends

| Store | Class | File(s) | Content |
|---|---|---|---|
| Segment content KV | `JsonKVStorage` | `video_segments_v2.json` | caption + transcript + entity_memory per segment |
| Video path KV | `JsonKVStorage` | `video_path_v2.json` | video_id → absolute path |
| Text chunks KV | `JsonKVStorage` | `text_chunks_v2.json` | chunked segment text |
| LLM cache | `JsonKVStorage` | `llm_response_cache_v2.json` | prompt hash → response |
| Text/entity VDB | `NanoVectorDBStorage` | `chunks_v2/`, `entities_v2/` | text-embedding-3-small vectors |
| Visual VDB | `NanoVectorDBVideoSegmentStorage` | `video_segment_feature_v2/` | ImageBind vectors |
| Unified graph | `UnifiedNetworkXStorage` | `chunk_entity_relation_v2.graphml` | directed MultiDiGraph |
| Global entity registry | JSON (atomic write) | `pipeline_v2/global_registry.json` | cross-video entity IDs |

All JSON KV stores share a `namespace` suffix that changes per generation (`_v2`, `_tm`, `""`).

## VideoRAG Dataclass — Full Config Reference

`videorag/videorag.py` — `@dataclass class VideoRAG`

Key groupings:
- **Video split (baseline)**: `video_segment_length=30`, `rough_num_frames_per_segment=5`, `fine_num_frames_per_segment=15`
- **Unified V2 ASR**: `asr_model`, `asr_device`, `asr_vad_filter`, `asr_beam_size`
- **Segmentation**: `segmentation_strategy` (`"adaptive"`/`"fixed"`), `segment_target_seconds=24`, `segment_min_seconds=8`, `segment_max_seconds=45`
- **Tracking**: `entity_tracking_model`, `entity_tracking_fps=3.0`, `visual_merge_threshold=0.85`
- **Frame selection**: `frame_min=2`, `frame_max=6`, `frame_duplicate_threshold=0.94`
- **Text entity/memory**: `spacy_model`, `text_alias_similarity_threshold=0.88`, `disable_transcript_memory`
- **Alignment/correspondence**: `correspondence_similarity_threshold=0.28`, `disable_crossmodal_alignment`
- **Graph retrieval**: `graph_seed_k=4`, `graph_restart_probability=0.15`, `graph_max_path_length=2`, `graph_context_token_cap=1800`
- **Ablation flags**: `ablation_profile`, `disable_transcript_memory`, `disable_visual_identity_linking`, `disable_crossmodal_alignment`
- **EBR query** (via `QueryParam`): `initial_text_k=4`, `initial_graph_k=4`, `initial_visual_k=4`, `max_evidence=16`, `max_rounds=2`
