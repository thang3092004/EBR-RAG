# Storage Layer, Graph Schema & Entity Registry

## Storage Class Hierarchy

```
StorageNameSpace (base.py)
├── BaseKVStorage[T]
│   └── JsonKVStorage (_storage/kv_json.py)
├── BaseVectorStorage
│   ├── NanoVectorDBStorage (_storage/vdb_nanovectordb.py)         # text + entities
│   └── NanoVectorDBVideoSegmentStorage (_storage/vdb_nanovectordb.py)  # ImageBind visual
└── BaseGraphStorage
    ├── NetworkXStorage (_storage/gdb_networkx.py)                 # legacy undirected
    ├── UnifiedNetworkXStorage (_storage/gdb_unified_networkx.py)  # Unified V2 directed MultiDiGraph
    └── NetworkXNeo4jStorage (_storage/gdb_neo4j.py)              # optional, not default
```

## Unified Graph Schema (`videorag/_unified_graph/schema.py`)

### EntityNode
```python
@dataclass
class EntityNode:
    entity_id: str          # e.g. "PERSON_001", "ANIMAL_003"
    entity_type: str        # normalized type: "person", "animal", "object", "concept", ...
    canonical_name: str
    aliases: list[EntityAlias]
    sources: list[str]      # ["text", "visual", "merged"]
    first_seen: float | None  # absolute timestamp (seconds)
    last_seen: float | None
    confidence: float
    attributes: dict[str, Any]
    provenance: list[ProvenanceRecord]
```

### EntityAlias
```python
@dataclass
class EntityAlias:
    alias_id: str           # "{video_id}::{local_id}" for global registry entries
    source: str             # "text", "visual", "alignment"
    label: str              # human-readable name
    confidence: float
    segment_id: str | None
```

### ProvenanceRecord
```python
@dataclass
class ProvenanceRecord:
    video_id: str
    segment_id: str
    modality: str           # "text", "visual", "crossmodal"
    timestamp: float | None
    confidence: float
```

### Graph Node / Edge Storage (NetworkX attributes)

Node attributes stored in graph:
- `entity_type`, `canonical_name`, `confidence`, `first_seen`, `last_seen`, `provenance` (JSON string), `aliases` (JSON string)

Edge attributes (MultiDiGraph — multiple edges allowed per pair):
- `predicate`: relation label (e.g. "attacks", "appears_in", "is_located_at")
- `direction`: "forward" or "backward"
- `confidence`: float
- `timestamp`: float (absolute seconds)
- `segment_id`: source segment
- `modality`: "text", "visual", or "crossmodal"
- `provenance`: JSON string (list of ProvenanceRecord dicts)
- `source_id`, `target_id`: entity IDs

### GLOBAL_PREFIXES

Entity type → ID prefix mapping (from `schema.py`):

| Type | Prefix |
|---|---|
| `person` | `PERSON` |
| `animal` | `ANIMAL` |
| `object` | `OBJECT` |
| `location` | `LOC` |
| `organization` | `ORG` |
| `concept` | `CONCEPT` |
| `event` | `EVENT` |
| `time` | `TIME` |
| _(other)_ | `ENTITY` |

IDs are zero-padded 3-digit counters: `PERSON_001`, `ANIMAL_003`, etc.

## EntityRegistry (`videorag/_unified_graph/registry.py`)

The registry lives in memory during ingestion, serialized to `registry.json` per stage and merged into `global_registry.json` after alignment.

Key operations:
- `allocate(entity_type)` → new ID (increments counter for type)
- `ensure_entity(type, name, ...)` → returns existing ID if name already known, else allocates new
- `add_alias(alias_id, entity_id, ...)` → registers an alias mapping
- `add_provenance(entity_id, record)` → appends provenance
- `normalize_alias(value)` → strips titles (Mr/Dr/The/A), punctuation, lowercases — used for name matching

The `alias_to_global` and `name_to_global` dicts enable fast lookup. IDs starting with `V_` or `T_` prefixes are provisional (visual tracklet or text provisional) and may be rejected by graph validation.

## NanoVectorDB

Two storage classes backed by NanoVectorDB:

### NanoVectorDBStorage (text chunks + entities)
- `upsert(data: dict[str, dict])`: uses `content` field for embedding, `entity_name` meta field for entities
- `query(query_str, top_k)`: returns list of `{id, score, ...meta}` dicts
- Embedding via `text-embedding-3-small` (1536-dim)

### NanoVectorDBVideoSegmentStorage (visual)
- `upsert(video_id, segment_index2name, video_format)`: reads segment clips, encodes with ImageBind-Huge (1024-dim)
- `query(query_str, top_k)`: ImageBind encodes the text query, retrieves by cosine similarity
- Segment IDs are `{video_id}_{segment_index}`

## JsonKVStorage

Simple JSON file backed dict with atomic writes. Supports:
- `upsert(data: dict)`, `get_by_id(id)`, `get_by_ids(ids, fields)`, `filter_keys(ids)` (returns keys not yet present), `drop()`
- `_data` dict is the in-memory state; `index_done_callback()` flushes to `{working_dir}/{namespace}.json`
- Used for: `video_segments`, `video_path`, `text_chunks`, `llm_response_cache`

## Segment Content Format

Each segment entry in `video_segments_v2.json`:
```json
{
  "content": "Entity Memory:\n- PERSON_001\nCaption:\n...\nTranscript:\n...",
  "time": "24.000-48.000",
  "transcript": "...",
  "caption": "...",
  "entity_memory": "Entity Memory:\n- PERSON_001",
  "frame_times": [24.1, 30.5, 42.0],
  "segment_id": "video_stem_0",
  "storage_id": "video_stem_0"
}
```

The `content` field is what gets chunked (~1200 tokens) and embedded for text retrieval.

## Text Chunk Format

Each chunk in `text_chunks_v2.json`:
```json
{
  "tokens": 847,
  "content": "...",
  "video_segment_id": ["video_stem_0", "video_stem_1"],
  "chunk_order_index": 0
}
```

Key: MD5 hash of content. Used by `filter_keys()` to skip already-indexed chunks.

## EvidenceItem (`videorag/debate/evidence_types.py`)

```python
@dataclass
class EvidenceItem:
    id: str             # unique key from storage
    type: str           # "text", "graph", "segment"
    score: float        # retrieval score
    snippet: str        # text content (may be updated by re-captioning)
    source: str         # storage namespace / channel name
    video_name: str | None
    time_range: str | None   # "24.0-48.0"
    metadata: dict      # retrieval_channels, rrf_score, refined flag
    provenance_path: str | None  # graph path string
    validated: bool     # set True by Judge citation cross-check
```
