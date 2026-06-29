# Fix Agent Prompt — EBR-RAG Pipeline Logical Errors

> Dành cho agent thực thi trên GPU server.
> Đọc toàn bộ trước khi sửa bất kỳ dòng code nào.
> Tất cả các fix dưới đây là BẮT BUỘC để pipeline hoạt động đúng.

---

## Context

EBR-RAG là hệ thống Long-Video QA với Unified Multimodal Graph V2.
Pipeline ingestion gồm 11 stage: `probe → asr → shot_detection → segmentation → tracking_base → frame_selection → text_entities → alignment_caption → graph → index → validation`.

Đây là danh sách **lỗi logic** được phát hiện qua review toàn bộ source code. Chúng chưa được fix.
Các lỗi được liệt kê theo thứ tự ưu tiên từ cao xuống thấp.

---

## Fix 1 — CRITICAL: bbox_position() nhận pixel coords, cần normalized

### Vấn đề

`tracker_v2.py` → `_process_chunk()` lưu bbox từ YOLO:
```python
bbox = [float(value) for value in xyxy[box_index].tolist()]
```

`xyxy` là `boxes.xyxy` từ Ultralytics — trả về **absolute pixel coordinates**
(e.g., `[100.0, 50.0, 320.0, 400.0]` trên frame 640×360).

Sau đó trong `link_visual_tracklets()` (cùng file):
```python
obs_bbox = list(observation["bbox"])
obs_position = bbox_position(obs_bbox)
```

`bbox_position()` trong `schema.py`:
```python
cx = (x1 + x2) / 2
h_zone = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
```

Với pixel value `cx = 320`, điều kiện `cx < 0.33` luôn sai, `cx > 0.66` luôn đúng
→ **toàn bộ entity bị gán `h_zone = "right"`**, bất kể vị trí thực trên màn hình.

Hệ quả: `dominant_position` trong `visual_entities.json` luôn sai
→ GPT-4o-mini nhận vị trí sai khi merge visual ↔ transcript entity.

### Fix

**File: `videorag/_entity_anchor/tracker_v2.py`**

Trong hàm `_process_chunk()`, sau dòng tính `bbox`, thêm normalization:

```python
# Tìm đoạn này (khoảng line 113):
bbox = [float(value) for value in xyxy[box_index].tolist()]

# Sửa thành — normalize về [0, 1] dùng frame dimensions:
h_frame, w_frame = frame.shape[:2]
raw_bbox = [float(value) for value in xyxy[box_index].tolist()]
bbox = [
    raw_bbox[0] / w_frame,  # x1
    raw_bbox[1] / h_frame,  # y1
    raw_bbox[2] / w_frame,  # x2
    raw_bbox[3] / h_frame,  # y2
]
```

Sau đó cũng fix crop extraction (dùng raw_bbox để crop ảnh, vì crop cần pixel coords):

```python
# Dòng 134-143 dùng bbox để crop, cần dùng raw_bbox (pixel):
x1, y1, x2, y2 = [int(round(v)) for v in raw_bbox]
crop = frame[
    max(0, y1): max(0, y2),
    max(0, x1): max(0, x2),
]
```

`_appearance_histogram()` cũng nhận pixel bbox — truyền `raw_bbox` vào đó:
```python
# Tìm dòng gọi _appearance_histogram:
"appearance": _appearance_histogram(frame, bbox),
# Sửa thành:
"appearance": _appearance_histogram(frame, raw_bbox),
```

**Lưu ý:** `_appearance_histogram()` đã xử lý clipping đúng với pixel coords — không cần sửa bên trong hàm đó.

---

## Fix 2 — CRITICAL: _stage_alignment() phải đọc ASR và truyền transcripts

### Vấn đề

`unified_ingest.py` → `_stage_alignment()` không đọc `asr.json`.
Kết quả: `align_all_segments()` → `run_crossmodal_merge()` → `_build_merge_prompt()`
không có raw transcript text của từng segment.

GPT-4o-mini thấy prompt như sau:
```
SEGMENT SEG_001 [0.0s - 12.3s]:
Visual caption: A zebra walks toward...
Visual entities present: ANIMAL_001
```

Không có dòng `Transcript: "the female moves toward the watering hole..."` → GPT-4o-mini
không biết tên thật trong transcript → không thể match V_ ↔ T_ entities.

### Fix

**Bước 1 — File: `videorag/pipeline/unified_ingest.py`**

Trong `_stage_alignment()`, thêm đọc ASR và tính transcripts per segment:

```python
def _stage_alignment(self, context, runner: StageRunner, video_id: str) -> dict[str, Any]:
    segments = read_json(runner.output("segmentation", "segments.json"))
    selections = read_json(runner.output("frame_selection", "frame_selections.json"))
    text_results = read_json(runner.output("text_entities", "text_entities.json"))
    observations = read_json(runner.output("tracking_base", "observations.json"))
    visual_entities_list = read_json(runner.output("tracking_base", "visual_entities.json"), [])
    registry = EntityRegistry(read_json(runner.output("tracking_base", "registry.json")))

    # === ADD THIS BLOCK ===
    asr = read_json(runner.output("asr", "asr.json"))
    by_segment = assign_words_to_segments(asr["words"], segments)
    transcripts: dict[str, str] = {
        str(seg["segment_id"]): " ".join(
            str(w["text"])
            for w in sorted(by_segment.get(str(seg["segment_id"]), []), key=lambda w: float(w["start"]))
        )
        for seg in segments
    }
    # === END ADD ===

    # ... rest of the function unchanged until align_all_segments() call ...

    result = align_all_segments(
        video_id,
        segments,
        selections,
        text_results,
        observations,
        registry,
        self.config,
        context.path("segments"),
        progress=progress,
        aligner=captioner,
        visual_entities=visual_entities_list,
        loop=self.loop,
        transcripts=transcripts,    # ADD this argument
    )
```

**Bước 2 — File: `videorag/_unified_graph/alignment.py`**

Thêm `transcripts` parameter vào `align_all_segments()` và truyền xuống:

```python
def align_all_segments(
    video_id: str,
    segments: list[dict[str, Any]],
    frame_selections: dict[str, dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    initial_registry: EntityRegistry,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    progress=None,
    aligner: MiniCPMCaptioner | None = None,
    visual_entities: list[dict[str, Any]] | None = None,
    loop=None,
    transcripts: dict[str, str] | None = None,   # ADD
    correspondence_encoder=None,  # legacy ignored
) -> dict[str, Any]:
    ...
    result = run_crossmodal_merge(
        video_id,
        segments,
        captions,
        visual_entities or [],
        text_results,
        observations,
        initial_registry,
        config,
        checkpoint_path,
        loop=loop,
        progress=progress,
        transcripts=transcripts or {},    # ADD
    )
```

**Bước 3 — File: `videorag/_unified_graph/alignment.py`**

Thêm `transcripts` parameter vào `run_crossmodal_merge()` và truyền xuống:

```python
def run_crossmodal_merge(
    video_id: str,
    segments: list[dict[str, Any]],
    captions: dict[str, str],
    visual_entities: list[dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    initial_registry: EntityRegistry,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    loop=None,
    progress=None,
    transcripts: dict[str, str] | None = None,    # ADD
) -> dict[str, Any]:
    ...
    # Khi gọi _build_merge_prompt:
    prompt = _build_merge_prompt(
        batch, captions, visual_entities, text_results,
        observations, entity_memory,
        transcripts=transcripts or {},    # ADD
    )
```

---

## Fix 3 — CRITICAL: 4 GAPs trong _build_merge_prompt()

### Vấn đề

`alignment.py` → `_build_merge_prompt()` thiếu 4 thông tin quan trọng:

**GAP 1:** Per-segment block không có `Transcript:` (đã fix bằng Fix 2 ở trên)

**GAP 2:** Per-segment visual entities chỉ liệt kê IDs, không có position:
```
Visual entities present: ANIMAL_001, ANIMAL_002
```
GPT-4o-mini không biết ANIMAL_001 đang ở vị trí nào trong segment này.

**GAP 3:** Batch-level transcript entities thiếu thông tin:
```
- T_ANIMAL_003: "lion" (animal)
```
Thiếu: `aliases`, text xuất hiện thực tế trong transcript, timestamp.

**GAP 4:** Transcript entities liệt kê ở level batch, không per-segment.
GPT-4o-mini không biết T_ANIMAL_003 xuất hiện ở segment nào
→ không thể match đúng khi có nhiều segments.

### Fix

**File: `videorag/_unified_graph/alignment.py`**

Sửa toàn bộ hàm `_build_merge_prompt()`. Thay thế implementation hiện tại bằng:

```python
def _build_merge_prompt(
    batch_segments: list[dict[str, Any]],
    captions: dict[str, str],
    visual_entities: list[dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    entity_memory: dict[str, Any],
    *,
    transcripts: dict[str, str] | None = None,    # NEW parameter
) -> str:
    transcripts = transcripts or {}
    entity_memory_json = json.dumps(entity_memory, ensure_ascii=False) if entity_memory else "{}"

    # ---- Visual entities section (batch-level) ----
    # Build lookup: entity_id → dominant_position
    ve_position: dict[str, str] = {
        ve["entity_id"]: ve.get("dominant_position") or "unknown"
        for ve in visual_entities
    }

    # Collect visual entity IDs that appear in this batch
    batch_visual_ids: set[str] = set()
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        for obs in observations:
            if str(obs.get("segment_id")) == seg_id:
                batch_visual_ids.add(str(obs["entity_id"]))

    v_lines = []
    for ve in visual_entities:
        eid = ve["entity_id"]
        if eid not in batch_visual_ids:
            continue
        pos = ve_position.get(eid, "unknown")
        v_lines.append(
            f"- {eid}: {ve['entity_type']} ({ve['label']}), "
            f"dominant position across video: {pos}"
        )
    visual_entities_str = "\n".join(v_lines) if v_lines else "(none)"

    # ---- Transcript entities section (batch-level) ----
    # Build deduplicated map: text_id → mention record
    t_seen: dict[str, dict[str, Any]] = {}
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        for mention in text_results.get(seg_id, {}).get("mentions", []):
            tid = mention["text_id"]
            if tid not in t_seen:
                t_seen[tid] = mention

    t_lines = []
    for tid, mention in t_seen.items():
        # surface_text: first occurrence text or label
        surface = (
            mention["occurrences"][0]["text"]
            if mention.get("occurrences")
            else mention["label"]
        )
        # aliases
        aliases = mention.get("aliases", [])
        alias_str = (", ".join(f'"{a}"' for a in aliases[:3])) if aliases else "none"
        # earliest timestamp
        start = mention.get("start", 0.0)
        t_lines.append(
            f"- {tid}: label=\"{mention['label']}\", type={mention['entity_type']}, "
            f"surface_text=\"{surface}\", aliases=[{alias_str}], first_seen={start:.1f}s"
        )
    transcript_entities_str = "\n".join(t_lines) if t_lines else "(none)"

    # ---- Segments section (per-segment detail) ----
    segments_data_parts = []
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        caption = captions.get(seg_id, "")
        raw_transcript = transcripts.get(seg_id, "")

        # Per-segment visual entities with positions
        seg_obs_by_entity: dict[str, list[dict[str, Any]]] = {}
        for obs in observations:
            if str(obs.get("segment_id")) == seg_id:
                eid = str(obs["entity_id"])
                seg_obs_by_entity.setdefault(eid, []).append(obs)

        seg_v_parts = []
        for eid, obs_list in sorted(seg_obs_by_entity.items()):
            # Use most common position from observations in this specific segment
            from collections import Counter
            from .schema import bbox_position
            pos_counts: Counter = Counter()
            for obs in obs_list:
                p = bbox_position(obs.get("bbox"))
                if p:
                    pos_counts[p] += 1
            seg_pos = pos_counts.most_common(1)[0][0] if pos_counts else ve_position.get(eid, "unknown")
            ve_info = next((ve for ve in visual_entities if ve["entity_id"] == eid), {})
            seg_v_parts.append(
                f"{eid} ({ve_info.get('label', '?')}, pos_in_segment: {seg_pos})"
            )
        seg_v_str = ", ".join(seg_v_parts) if seg_v_parts else "(none)"

        # Per-segment transcript entities
        seg_t_parts = []
        for mention in text_results.get(seg_id, {}).get("mentions", []):
            surface = (
                mention["occurrences"][0]["text"]
                if mention.get("occurrences")
                else mention["label"]
            )
            seg_t_parts.append(
                f"{mention['text_id']} (\"{surface}\", {mention['entity_type']})"
            )
        seg_t_str = ", ".join(seg_t_parts) if seg_t_parts else "(none)"

        segments_data_parts.append(
            f"SEGMENT {seg_id} [{seg['start']:.1f}s - {seg['end']:.1f}s]:\n"
            f"Visual caption: {caption}\n"
            f"Transcript: {raw_transcript}\n"
            f"Visual entities in this segment: {seg_v_str}\n"
            f"Transcript entities in this segment: {seg_t_str}"
        )
    segments_data_str = "\n\n".join(segments_data_parts)

    return CROSSMODAL_MERGE_PROMPT.format(
        entity_memory_json=entity_memory_json,
        visual_entities_with_positions=visual_entities_str,
        transcript_entities_with_names=transcript_entities_str,
        segments_data=segments_data_str,
    )
```

**Lưu ý quan trọng:**
- Import `Counter` từ `collections` đã có sẵn ở đầu file
- Import `bbox_position` từ `.schema` — cần thêm vào import block ở đầu file nếu chưa có:
  ```python
  from .schema import EdgeOccurrence, ProvenanceRecord, bbox_position
  ```
- Hàm này bây giờ nhận thêm `transcripts` keyword argument — đảm bảo signature match với các chỗ gọi (đã xử lý ở Fix 2)

---

## Fix 4 — HIGH: Entity descriptions không được lưu vào graph nodes

### Vấn đề

`run_crossmodal_merge()` tích lũy `entity_memory` với:
```python
{
    "ANIMAL_001": {
        "canonical_name": "lion",
        "accumulated_descriptions": [
            "adult male, chasing prey (seg5)",
            "resting under tree, left frame (seg12)",
        ],
        "relationships": ["chased ANIMAL_003 (SEG_005)"],
        "segments_seen": ["SEG_005", "SEG_012"],
    }
}
```

Nhưng `build_unified_graph()` trong `builder.py` chỉ dùng:
```python
"description": node.canonical_name,  # chỉ "lion", không có gì khác
```

Toàn bộ accumulated descriptions bị mất. Graph retrieval không có semantic content.

### Fix

**File: `videorag/pipeline/unified_ingest.py`**

Trong `_stage_alignment()`, sau khi `align_all_segments()` trả về, cập nhật entity descriptions trong registry:

```python
result = align_all_segments(...)

# === ADD: enrich registry entity descriptions from accumulated entity_memory ===
entity_memory = result.get("entity_memory", {})
registry_data = result.get("registry", {})
# Build a lookup: entity_id → canonical description
for entity in registry_data.get("entities", []):
    eid = entity["entity_id"]
    mem = entity_memory.get(eid)
    if mem and mem.get("accumulated_descriptions"):
        # Join last 5 descriptions into a rich description
        rich_desc = "; ".join(mem["accumulated_descriptions"][-5:])
        entity["description"] = rich_desc
    else:
        entity.setdefault("description", entity.get("canonical_name", eid))
# === END ADD ===

context.write_json("alignment.json", result)
```

**File: `videorag/_unified_graph/builder.py`**

Trong `build_unified_graph()`, dùng `entity.get("description")` nếu có:

```python
# Tìm dòng này:
"description": node.canonical_name,

# Sửa thành:
"description": node.attributes.get("description") or node.canonical_name,
```

**Và trong `ensure_entity()` của `registry.py`**, khi load entity từ payload, đảm bảo `description` field được lưu vào `attributes`:

Thực ra cách đơn giản hơn: trong `build_unified_graph()`, thêm parameter `entity_memory`:

```python
async def build_unified_graph(
    storage,
    registry_payload: dict[str, Any],
    alignment_segments: dict[str, dict[str, Any]],
    *,
    clear: bool = True,
    entity_memory: dict[str, Any] | None = None,  # ADD
) -> dict[str, Any]:
    registry = EntityRegistry(registry_payload)
    entity_memory = entity_memory or {}
    ...
    for entity_id, node in registry.entities.items():
        ...
        # Build rich description
        mem = entity_memory.get(entity_id, {})
        descriptions = mem.get("accumulated_descriptions", [])
        if descriptions:
            description = f"{node.canonical_name}: " + "; ".join(descriptions[-5:])
        else:
            description = node.canonical_name

        await storage.upsert_node(
            entity_id,
            {
                ...
                "description": description,    # was: node.canonical_name
                ...
            },
        )
```

Và trong `_stage_graph()` của `unified_ingest.py`, truyền `entity_memory`:

```python
alignment = read_json(runner.output("alignment_caption", "alignment.json"))
...
graph_stats = self._await(
    build_unified_graph(
        self.vrag.chunk_entity_relation_graph,
        alignment["registry"],
        alignment["segments"],
        clear=False,
        entity_memory=alignment.get("entity_memory", {}),   # ADD
    )
)
```

---

## Fix 5 — LOW: segments_seen không được cập nhật ở memory_updates loop

### Vấn đề

`alignment.py` → `run_crossmodal_merge()`. Khi GPT-4o-mini trả về `memory_updates`
(entity description update) mà không trả về `relationships` tương ứng, `segments_seen`
của entity đó không được cập nhật.

### Fix

**File: `videorag/_unified_graph/alignment.py`**

Trong loop xử lý `memory_updates`, thêm segments_seen update:

```python
for update in validated["memory_updates"]:
    eid = update["entity_id"]
    canonical = registry.resolve(eid) or eid
    entry = entity_memory.setdefault(canonical, {
        "canonical_name": "",
        "accumulated_descriptions": [],
        "relationships": [],
        "segments_seen": [],
    })
    node = registry.entities.get(canonical)
    if node:
        entry["canonical_name"] = node.canonical_name
    entry["accumulated_descriptions"].append(update["new_description"])
    entry["accumulated_descriptions"] = entry["accumulated_descriptions"][-10:]

    # ADD: track which segments this entity was seen in
    for seg in batch:
        seg_id = str(seg["segment_id"])
        # Check if entity appears in this segment (visual or transcript)
        seg_has_entity = any(
            str(obs.get("entity_id")) == canonical
            for obs in observations
            if str(obs.get("segment_id")) == seg_id
        ) or any(
            m["text_id"] == eid or (registry.resolve(m["text_id"]) == canonical)
            for m in text_results.get(seg_id, {}).get("mentions", [])
        )
        if seg_has_entity and seg_id not in entry["segments_seen"]:
            entry["segments_seen"].append(seg_id)
```

---

## Thứ tự thực hiện

```
Fix 1 → tracker_v2.py (bbox normalization)
Fix 2 → unified_ingest.py + alignment.py (ASR transcripts)
Fix 3 → alignment.py (rebuild _build_merge_prompt)
Fix 4 → builder.py + unified_ingest.py (entity descriptions)
Fix 5 → alignment.py (segments_seen, minor)
```

Làm theo thứ tự này vì Fix 3 phụ thuộc vào signature thay đổi ở Fix 2.

---

## Kiểm tra sau khi fix

Sau khi sửa xong, verify bằng unit test nhanh:

```python
# test bbox_position
from videorag._unified_graph.schema import bbox_position
assert bbox_position([0.1, 0.1, 0.4, 0.4]) == "top-left, medium"   # normalized
assert bbox_position([0.5, 0.5, 0.9, 0.9]) == "bottom-right, large"

# test _build_merge_prompt có transcript
from videorag._unified_graph.alignment import _build_merge_prompt
batch = [{"segment_id": "SEG_001", "start": 0.0, "end": 10.0}]
result = _build_merge_prompt(
    batch, {"SEG_001": "a zebra walks"},
    [], {}, [], {},
    transcripts={"SEG_001": "the herd moves slowly"}
)
assert "Transcript: the herd moves slowly" in result
assert "Transcript entities in this segment:" in result
```

---

## Files cần sửa (tóm tắt)

| File | Fixes |
|------|-------|
| `videorag/_entity_anchor/tracker_v2.py` | Fix 1: normalize bbox trước khi lưu |
| `videorag/pipeline/unified_ingest.py` | Fix 2 + Fix 4: đọc ASR, enrichment descriptions |
| `videorag/_unified_graph/alignment.py` | Fix 2 + Fix 3 + Fix 5: transcripts param, rebuild _build_merge_prompt |
| `videorag/_unified_graph/builder.py` | Fix 4: entity_memory param, rich description |

---

## Không sửa gì khác

Chỉ sửa đúng những gì được liệt kê. Đặc biệt:
- Không thay đổi `VISUAL_CAPTION_PROMPT` (MiniCPM không nhận transcript, đây là đúng theo design)
- Không thay đổi `CROSSMODAL_MERGE_PROMPT` template
- Không thay đổi logic checkpoint/resume trong `run_crossmodal_merge()`
- Không thay đổi `link_visual_tracklets()` threshold (B3 là vấn đề riêng, không sửa ở đây)
