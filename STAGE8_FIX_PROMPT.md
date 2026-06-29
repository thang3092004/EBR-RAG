# Stage 8b Fix Agent Prompt

**File duy nhất cần sửa:** `videorag/_unified_graph/alignment.py`

Có 3 fix độc lập, thực hiện lần lượt. Đọc kỹ từng phần trước khi viết code.

---

## FIX 1 — canonical_name của visual entity không được cập nhật sau khi match

### Vấn đề

Khi GPT-4o-mini match thành công `PERSON_001 (visual) ↔ T_PERSON_001 = "Roger Federer"`,
code gọi `registry.add_alias(T_PERSON_001, PERSON_001, label="Roger Federer")`.
Nhưng `PERSON_001.canonical_name` vẫn là `"person"` (YOLO class label).

Entity_memory ở các batch sau sẽ thấy:
```json
{"PERSON_001": {"canonical_name": "person", "accumulated_descriptions": [...]}}
```
GPT-4o-mini không biết `PERSON_001 = Roger Federer` trừ khi nó tự ghi tên vào
accumulated_descriptions.

### Vị trí code

Trong hàm `run_crossmodal_merge()`, vòng lặp xử lý `validated["matches"]` nằm
quanh **dòng 733–753**.

Hiện tại vòng lặp này chỉ gọi `registry.add_alias(...)`. Sau lời gọi đó, thêm:

```python
# Update canonical_name if it is still the generic YOLO class label
GENERIC_VISUAL_LABELS = {
    "person", "animal", "object", "vehicle", "plant",
    "food", "sports", "outdoor", "indoor",
}
node = registry.entities.get(match["visual_id"])
if node and mention and mention.get("label"):
    if (
        node.canonical_name in GENERIC_VISUAL_LABELS
        or node.canonical_name == node.entity_id
    ):
        node.canonical_name = mention["label"]

# Propagate into entity_memory immediately so next batch sees the proper name
if match["visual_id"] in entity_memory and mention and mention.get("label"):
    entity_memory[match["visual_id"]]["canonical_name"] = mention["label"]
```

### Cross-batch memory match (thêm vào cùng vòng lặp, sau khối trên)

Khi `match["text_id"]` là global ID từ registry (không tìm thấy trong text_results
của batch hiện tại vì entity đó ở batch khác), handle riêng:

```python
if mention is None and match["text_id"] in registry.entities:
    memory_node = registry.entities[match["text_id"]]
    vis_node = registry.entities.get(match["visual_id"])
    if (
        vis_node
        and memory_node.canonical_name
        and memory_node.canonical_name not in GENERIC_VISUAL_LABELS
        and memory_node.canonical_name != vis_node.entity_id
    ):
        if (
            vis_node.canonical_name in GENERIC_VISUAL_LABELS
            or vis_node.canonical_name == vis_node.entity_id
        ):
            vis_node.canonical_name = memory_node.canonical_name
        if match["visual_id"] in entity_memory:
            entity_memory[match["visual_id"]]["canonical_name"] = (
                memory_node.canonical_name
            )
```

---

## FIX 2 — Cross-batch matches bị reject bởi _validate_merge_result()

### Vấn đề

Khi GPT-4o-mini thấy trong entity_memory có `PERSON_002: {canonical_name: "Roger Federer"}`
(entity từ batch trước) và muốn match `PERSON_001 (visual)` với `PERSON_002`,
match bị reject vì `_validate_merge_result()` chỉ chấp nhận `text_id` nằm trong
`batch_text_ids` (tập T_ IDs của batch hiện tại).

### Fix `_validate_merge_result()`

Thêm parameter `registry_entity_ids=None`:

```python
def _validate_merge_result(
    raw: dict[str, Any],
    visual_ids: set[str],
    text_ids: set[str],
    registry_entity_ids: set[str] | None = None,
) -> dict[str, Any]:
    registry_entity_ids = registry_entity_ids or set()
    matches = []
    for match in raw.get("matches", []):
        vid = str(match.get("visual_id", "")).strip()
        tid = match.get("text_id")
        if tid is not None:
            tid = str(tid).strip()
        if vid in visual_ids:
            if tid is None or tid in text_ids or tid in registry_entity_ids:
                matches.append({
                    "visual_id": vid,
                    "text_id": tid,
                    "reason": str(match.get("reason", "")),
                })
    # ... phần relationships và memory_updates giữ nguyên
```

### Fix call site

Tại dòng gọi `_validate_merge_result(...)` trong `run_crossmodal_merge()`, truyền thêm:

```python
validated = _validate_merge_result(
    raw_result,
    batch_visual_ids,
    batch_text_ids,
    registry_entity_ids=set(registry.entities.keys()),
)
```

---

## FIX 3 — Entity memory relevance filtering (context dilution)

### Vấn đề

Sau 50+ segment, `entity_memory` có thể chứa 60–80 entity, tất cả được dump nguyên
thành JSON ở đầu prompt. Entity từ 90 phút trước nằm cạnh entity vừa xuất hiện 30
giây trước — equal weight. GPT-4o-mini attend ít hơn vào phần per-segment content
ở cuối prompt.

### Logic 4 lớp priority

```
Layer 1 (luôn giữ):  entity đang xuất hiện trong batch hiện tại (visual + transcript)
Layer 2 (giữ nếu còn chỗ):  1-hop neighbors — entity có edge trực tiếp với Layer 1
Layer 3 (giữ nếu còn chỗ):  entity xuất hiện trong ~20 segment gần nhất (short-term)
Layer 4 (giữ nếu còn chỗ):  entity xuất hiện trong ≥ 5 segment khác nhau (global main characters)
Drop:  tất cả còn lại
Cap tổng: 25 entity (configurable qua config["entity_memory_max_context"])
```

Lý do Layer 2 quan trọng: nếu `ANIMAL_001` đang trong batch và memory của nó có
`"chased_by ANIMAL_003 (SEG_007)"`, GPT-4o-mini cần biết `ANIMAL_003` để không
extract lại edge đã có và để hiểu context quan hệ đúng.

Relationship strings hiện có format: `"predicate ENTITY_ID (SEG_xxx)"`.
Parse bằng `parts = rel_str.split(); entity_id = parts[1]` nếu `len(parts) >= 2`.

### Thêm helper function `_filter_entity_memory()`

Thêm hàm này vào file, đặt trước `_build_merge_prompt()`:

```python
def _filter_entity_memory(
    entity_memory: dict[str, Any],
    active_entity_ids: set[str],
    recent_segment_ids: set[str],
    *,
    global_threshold: int = 5,
    max_entities: int = 25,
) -> dict[str, Any]:
    """
    Filter entity_memory to the most relevant entities for the current batch.
    Uses 4-layer priority to avoid context dilution in GPT-4o-mini prompt.

    Layer 1: entities active in current batch (always included)
    Layer 2: 1-hop relational neighbors of active entities
    Layer 3: entities seen in recent segments (short-term continuity)
    Layer 4: globally frequent entities (main characters of the video)
    """
    keep: dict[str, int] = {}  # entity_id -> priority (lower = higher priority)

    # Layer 1: active entities in current batch
    for eid in active_entity_ids:
        if eid in entity_memory:
            keep[eid] = 1

    # Layer 2: 1-hop relational neighbors of Layer 1 entities
    for eid in list(keep.keys()):
        for rel_str in entity_memory.get(eid, {}).get("relationships", []):
            parts = rel_str.split()
            if len(parts) >= 2:
                # Format: "predicate ENTITY_ID (SEG_xxx)"
                neighbor_id = parts[1].strip("(),.")
                if neighbor_id in entity_memory and neighbor_id not in keep:
                    keep[neighbor_id] = 2

    # Layer 3: entities seen in recent segments
    for eid, data in entity_memory.items():
        if eid not in keep:
            if any(s in recent_segment_ids for s in data.get("segments_seen", [])):
                keep[eid] = 3

    # Layer 4: globally frequent entities
    for eid, data in entity_memory.items():
        if eid not in keep:
            if len(data.get("segments_seen", [])) >= global_threshold:
                keep[eid] = 4

    # Sort by priority, cap at max_entities
    sorted_eids = sorted(keep.keys(), key=lambda e: keep[e])[:max_entities]
    return {eid: entity_memory[eid] for eid in sorted_eids if eid in entity_memory}
```

### Sửa `_build_merge_prompt()` để nhận filtered memory

Thêm 2 parameter mới vào signature của `_build_merge_prompt()`:

```python
def _build_merge_prompt(
    batch_segments: list[dict[str, Any]],
    captions: dict[str, str],
    visual_entities: list[dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    entity_memory: dict[str, Any],
    *,
    transcripts: dict[str, str] | None = None,
    recent_segment_ids: set[str] | None = None,   # NEW
    max_memory_entities: int = 25,                  # NEW
) -> str:
```

Thay dòng hiện tại:
```python
entity_memory_json = json.dumps(entity_memory, ensure_ascii=False) if entity_memory else "{}"
```

Bằng:
```python
# Collect active entity IDs (visual + transcript) for this batch
_active_ids: set[str] = set()
for _seg in batch_segments:
    _sid = str(_seg["segment_id"])
    for _obs in observations:
        if str(_obs.get("segment_id")) == _sid:
            _active_ids.add(str(_obs["entity_id"]))
    for _mention in text_results.get(_sid, {}).get("mentions", []):
        _active_ids.add(_mention["text_id"])

filtered_memory = _filter_entity_memory(
    entity_memory,
    _active_ids,
    recent_segment_ids or set(),
    max_entities=max_memory_entities,
)
entity_memory_json = json.dumps(filtered_memory, ensure_ascii=False) if filtered_memory else "{}"
```

### Sửa call site trong `run_crossmodal_merge()`

Tại dòng gọi `_build_merge_prompt(...)`, thêm 2 argument mới:

```python
# Tính recent_segment_ids: 20 segment trước batch hiện tại
all_segment_ids = [str(s["segment_id"]) for s in segments]
_first_seg_id = str(batch[0]["segment_id"]) if batch else ""
_first_idx = all_segment_ids.index(_first_seg_id) if _first_seg_id in all_segment_ids else 0
_recent_start = max(0, _first_idx - 20)
_recent_segment_ids = set(all_segment_ids[_recent_start:_first_idx])

prompt = _build_merge_prompt(
    batch, captions, visual_entities, text_results,
    observations, entity_memory,
    transcripts=transcripts or {},
    recent_segment_ids=_recent_segment_ids,                              # NEW
    max_memory_entities=int(config.get("entity_memory_max_context", 25)),  # NEW
)
```

---

## Những thứ KHÔNG được thay đổi

- Signature của `run_crossmodal_merge()` và `align_all_segments()` — không thêm parameter
- Logic checkpoint/resume (`merge_state.json`)
- Logic xử lý `disable_crossmodal_alignment`
- Relationship extraction và edge creation logic
- Bất kỳ file nào khác ngoài `alignment.py`
- Không xóa test nào, không thêm test mới

---

## Thứ tự thực hiện

1. Thêm `GENERIC_VISUAL_LABELS` constant ở đầu file (sau imports)
2. Thêm `_filter_entity_memory()` function trước `_build_merge_prompt()`
3. Sửa `_validate_merge_result()` — thêm parameter `registry_entity_ids`
4. Sửa `_build_merge_prompt()` — thêm 2 parameter + logic filter
5. Sửa `run_crossmodal_merge()` — Fix 1 (canonical_name), Fix 2 (call site validation), Fix 3 (call site prompt)

Sau khi xong, chạy:
```bash
python -c "from videorag._unified_graph.alignment import _filter_entity_memory, _validate_merge_result, _build_merge_prompt; print('imports OK')"
```
để verify không có syntax error.
