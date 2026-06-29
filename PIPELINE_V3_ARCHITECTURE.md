# Unified Multimodal Graph V3 — Pipeline Architecture & Changelog

## Pipeline Overview

```
Video file
  │
  ├─► [1] probe           — FFprobe metadata (duration, fps, resolution)
  │
  ├─► [2] asr             — Whisper ASR → timestamped words
  │
  ├─► [3] shot_detection   — Shot boundaries + motion analysis
  │
  ├─► [4] segmentation     — Adaptive segmentation (shot-boundary-aware)
  │                          Output: segments with has_shot_at_start field
  │
  ├─► [5] tracking_base    — SKIP (<1s, empty output when entity_source="caption")
  │
  ├─► [6] frame_selection  — Diverse frame selection (temporal + visual diversity)
  │                          Clean frames, no overlay text or bounding boxes
  │
  ├─► [7] text_entities    — SKIP (<1s when entity_source="caption")
  │
  ├─► [8] alignment_caption:
  │       8a: MiniCPM-V-2_6-int4 visual captioning (parallel, GPU)
  │           → captions.json (preserved on restart)
  │           Plain text descriptions with appearance, position, actions
  │       8b: GPT-4o-mini entity + relationship extraction (sequential, API)
  │           GraphRAG/LightRAG-style unified extraction in 1 call per batch
  │           Input: captions + raw transcripts + entity_memory
  │           Output: entities + relationships + memory_updates
  │           → registry + edges
  │
  ├─► [9] graph_build      — Build MultiDiGraph from registry + edges
  │                          Entity descriptions enriched from entity_memory
  │
  ├─► [10] embedding_index — Embed chunks (caption + original transcript) + entities
  │
  └─► [11] validation      — Check graph integrity, warn on >50% isolated nodes
```

## Key Design Decisions

### Entity Extraction: LLM-based (GraphRAG-style), not NER

All leading GraphRAG systems (Microsoft GraphRAG, LightRAG, KGGen) use LLMs to extract
entities and relationships in a single call. We follow this approach:

- GPT-4o-mini receives visual captions + audio transcripts per batch
- Extracts entities from BOTH sources simultaneously
- Resolves cross-modal identity (visual "large male lion" = transcript "alpha male")
- No spaCy/NER pre-processing, no V_/T_ prefix system
- Entity deduplication via accumulated entity_memory across batches

### Visual Perception: MiniCPM captioning only, no YOLO

YOLO+BoT-SORT tracking was removed because it produced massive noise:
- Hallucinated objects (banana, pizza, surfboard in wildlife videos)
- 70% singleton entities, 83% appeared in only 1 segment
- 15+ hours GPU time per video with minimal useful output

MiniCPM-V-2_6-int4 captioning produces rich descriptions with appearance,
position, and action information — sufficient for LLM entity extraction.

### Graph Storage: MultiDiGraph

Uses `nx.MultiDiGraph` via `UnifiedNetworkXStorage` — directed edges with
multiple edge occurrences between the same entity pair preserved.

## Prompt Design

System prompt establishes role as knowledge graph builder.

Per-batch prompt includes:
1. **Known entities** (accumulated memory, filtered by relevance)
2. **Segments** with visual caption + transcript per segment

GPT-4o-mini tasks:
1. **Entity extraction** — extract from both captions and transcripts
2. **Relationship extraction** — directed relationships with predicates
3. **Memory update** — update descriptions for known entities

### Entity Naming Rules (enforced in prompt + post-processing)
- SINGULAR form (LION not LIONS, DOLPHIN not DOLPHINS)
- Check known entities before creating new ones
- Merge synonyms (HIPPO = HIPPOPOTAMUS, ORCA = KILLER WHALE)
- Named entities preferred (SCARFACE not MONKEY)
- Role-based names for unnamed people (NARRATOR, RESEARCHER, DIVER)
- No meta-entities (the scene, the atmosphere, BBC EARTH)

### Predicate Normalization
- Present tense base form (chase not chases, inhabit not inhabits)
- Synonym merging (share_habitat = shares_space_with = shares_environment_with)
- Post-processing via PREDICATE_NORMALIZE lookup table

## Entity Memory System

Progressive memory across batches:
```json
{
  "ANIMAL_001": {
    "canonical_name": "LION",
    "entity_type": "animal",
    "accumulated_descriptions": ["Adult male, mane, dominant", "Hunting buffalo"],
    "relationships": ["hunt BUFFALO (SEG_00045)", "confront HYENA (SEG_00052)"],
    "segments_seen": ["SEG_00040", "SEG_00045", "SEG_00052"]
  }
}
```

Memory filtering (4-layer priority, max 25 entities per prompt):
1. 1-hop relational neighbors
2. Entities seen in recent segments (shot-boundary-aware window)
3. Globally frequent entities (seen in 5+ segments)

Display format shows canonical_name as key (not internal IDs) so GPT-4o-mini
can recognize and reuse existing entities.

## Chunk Content Format

```
Entity Memory:
- ANIMAL_001 (LION)
- ANIMAL_002 (HYENA)

Caption:
A large male lion walks from the left toward a group of hyenas...

Transcript:
The alpha male approaches the scavengers cautiously as the narrator describes...
```

Sections separated by double newlines for clean embedding boundaries.
Transcript uses original ASR text (real names, no entity IDs).

## Problems Fixed (V2 → V3)

### B1 — Caption coverage 0% [CRITICAL, FIXED]
**Was:** MiniCPM couldn't OCR entity IDs drawn on frames → captions empty.
**Fix:** Removed all frame overlay (bbox + text). MiniCPM receives clean frames.
Captioning produces plain text descriptions, no structured JSON.

### B2 — Graph extremely sparse [CRITICAL, FIXED]
**Was:** 5958 nodes, 97 edges, 96% isolated nodes.
**Fix:** GraphRAG-style LLM extraction produces meaningful entities (189 nodes)
with high connectivity (262 edges, 18% isolated, 39 components).

### B3 — Cross-segment entity merge = 0 [CRITICAL, FIXED]
**Was:** OpenCLIP cosine gate too strict, 0 merges.
**Fix:** Removed OpenCLIP gate entirely. Entity deduplication handled by
GPT-4o-mini via entity_memory + name normalization post-processing.

### B4 — Chunks embed anonymous text [HIGH, FIXED]
**Was:** Chunks contained "OBJECT_352 gather together on OBJECT_353".
**Fix:** Chunks embed original transcript with real names. Entity memory
section lists entity IDs with canonical names for context.

### B5 — Frame paths not portable [HIGH, FIXED]
**Was:** Absolute paths in frame_selections.json.
**Fix:** Captions checkpoint preserved on stage restart. Frame paths
relative to workdir.

### B6 — YOLO hallucination [CRITICAL, FIXED]
**Was:** YOLO detected banana, pizza, surfboard in wildlife videos.
8443 tracklets → 4725 entities, 70% singletons.
**Fix:** Removed YOLO entirely. Entity extraction from captions by GPT-4o-mini.
No computer vision object detection in pipeline.

### Entity duplicates (DOLPHIN/DOLPHINS, HIPPO/HIPPOPOTAMUS) [HIGH, FIXED]
**Was:** 14+ duplicate entity groups, alias_to_global completely empty.
**Fix:** Prompt enforces singular form + synonym checking. Post-processing
ENTITY_NAME_NORMALIZE table handles 25+ known mappings. Zero duplicates
in output.

### Predicate inconsistency [HIGH, FIXED]
**Was:** 98 unique predicates with near-duplicates (chases/chase, inhabit/inhabits).
**Fix:** Prompt enforces present tense base form. PREDICATE_NORMALIZE table
handles 30+ verb lemmatizations.

### Chunk formatting [MEDIUM, FIXED]
**Was:** Missing newlines between transcript and caption sections in 98% of chunks.
**Fix:** Sections separated by `\n\n` via explicit parts joining.

### PERSON conflation [MEDIUM, FIXED]
**Was:** All unnamed people merged into single PERSON_001 entity.
**Fix:** Prompt instructs role-based naming (NARRATOR, RESEARCHER, DIVER).

### Noise entities [MEDIUM, FIXED]
**Was:** spaCy extracted "the scene", "a sense", "the atmosphere" from captions.
**Fix:** LLM extraction doesn't produce meta-entities. ENTITY_NOISE filter
catches remaining cases (BBC EARTH, SUBMARINE, etc.).

### Silent exception swallowing [HIGH, FIXED]
**Was:** GPT-4o-mini API errors silently produced empty batches.
**Fix:** Exceptions logged with batch number and video ID. pipeline_strict
mode re-raises. Dropped relationships logged to relationship_drops.jsonl.

### Graph validation incomplete [MEDIUM, FIXED]
**Was:** 97% isolated nodes but validation reported valid=true.
**Fix:** validate_unified_graph warns when >50% isolated, reports
isolated_count and isolated_ratio in output.

### Captions lost on restart [MEDIUM, FIXED]
**Was:** --restart-stage alignment_caption deleted captions.json.
**Fix:** preserve_on_restart=("captions.json",) in stage definition.
Stage runner saves and restores preserved files during invalidation.

## Performance (Collection 0, 1 video ~3h)

| Stage | Time | Notes |
|---|---|---|
| probe | <1s | |
| asr | ~3m | Whisper |
| shot_detection | ~11m | |
| segmentation | ~10s | |
| tracking_base | <1s | Skipped (caption mode) |
| frame_selection | ~23m | |
| text_entities | <1s | Skipped (caption mode) |
| alignment_caption | ~3h | MiniCPM (cached) + GPT-4o-mini |
| graph_build | <1s | |
| embedding_index | ~1h | |
| validation | <1s | |

## Output Quality (Collection 0)

| Metric | Value |
|---|---|
| Segments | 585 |
| Entities | 189 |
| Edges | 262 |
| Isolated nodes | 18% |
| Connected components | 39 |
| Duplicate entity names | 0 |
| Dropped relationships | 9 |
| Entity types | 127 animal, 28 object, 25 location, 9 person |
| Top entities | LION(30), FISH(26), BIRD(14), DOLPHIN(12), POLAR BEAR(12) |

## Config Flags

| Flag | Default | Purpose |
|---|---|---|
| entity_source | "caption" | "caption" = LLM extraction, "yolo" = legacy YOLO pipeline |
| crossmodal_batch_size | 8 | Segments per GPT-4o-mini batch |
| entity_memory_max_context | 25 | Max entities shown in prompt memory |
| caption_visual_batch_size | 2 | MiniCPM batch size (VRAM dependent) |
| disable_crossmodal_alignment | False | Skip GPT-4o-mini extraction entirely |
