# Pipeline Redesign Spec — Unified Multimodal Graph V2

> Tài liệu này mô tả: (1) các lỗi nghiêm trọng trong pipeline hiện tại, (2) nguyên nhân gốc rễ,
> (3) thiết kế mới đề xuất với đặc tả đủ chi tiết để implement.
>
> Dành cho agent thực thi trên GPU server. Đọc toàn bộ trước khi chạm vào code.

---

## Phần 1 — Tổng quan pipeline hiện tại

Pipeline `UnifiedIngestPipeline` chạy 11 stage theo thứ tự, resumable:

```
probe → asr → shot_detection → segmentation → tracking_base
    → frame_selection → text_entities → alignment_caption
    → graph → index → validation
```

Các stage liên quan đến redesign:

| Stage | File chính | Việc làm |
|---|---|---|
| `tracking_base` | `unified_ingest.py:_stage_tracking` | YOLO+BoT-SORT → tracklets → EntityRegistry (V_ IDs) |
| `frame_selection` | `unified_ingest.py:_stage_frames` | Chọn diverse frames, **vẽ overlay** lên ảnh |
| `text_entities` | `unified_ingest.py:_stage_text` | Extract T_ entities từ transcript (sequential+memory) |
| `alignment_caption` | `unified_ingest.py:_stage_alignment` | MiniCPM caption + cross-modal merge (BROKEN) |
| `graph` | `unified_ingest.py:_stage_graph` | Build graph từ output của alignment_caption |
| `index` | `unified_ingest.py:_stage_index` | Embed vào chunks_vdb, entities_vdb |

---

## Phần 2 — Các lỗi hiện tại và nguyên nhân

### B1 — Caption coverage ≈ 0% [CRITICAL]

**Triệu chứng:** Trên 3 collections (0, 6, 11), chỉ 0.2–5.5% segments có caption thật.
Chunk content gần như luôn là `"Caption:\n\nTranscript:\n..."` (caption trống).

**Nguyên nhân — chuỗi 3 lớp:**

**Lớp 1:** `frame_selector.py` hàm `_draw_overlay()` vẽ entity ID dạng text
("ANIMAL_001", "PERSON_003") lên ảnh frame trước khi lưu, với `font_scale=0.5` —
rất nhỏ trên ảnh 1280×720 background phức tạp.

**Lớp 2:** MiniCPM-V-2_6-int4 không thể OCR text nhỏ đó. Test thực nghiệm trên 5
frames thật từ pipeline:

| Model | Precision | Recall |
|---|---|---|
| MiniCPM-V-2_6-int4 | **0.000** | **0.000** |
| GPT-4o-mini | 0.905 | 0.891 |

MiniCPM hallucinate ID sai (ví dụ "ANIMAL_024" thay vì "ANIMAL_001").

**Lớp 3:** `alignment.py` hàm `_validate_visual_analysis()` filter strict — chỉ giữ
`entity_descriptions` có `entity_id` nằm trong allowed set. MiniCPM hallucinate ID
sai → tất cả bị filter → `entity_descriptions = []`, `visual_caption = ""`.

**Hậu quả:** Ablation A5 (`no_crossmodal_alignment`) ≈ A1 (`full_framework`) vì A1
cũng không có caption. Delta đo được = noise.

---

### B2 — Graph cực thưa, PPR vô dụng [CRITICAL]

**Triệu chứng:**

| Collection | Nodes | Edges | Isolated (~) | % reachable via PPR |
|---|---|---|---|---|
| 0 | 4 551 | 97 | ~4 357 | ~4% |
| 6 | 6 901 | 70 | ~6 761 | ~2% |
| 11 | 19 052 | 209 | ~18 634 | ~2% |

Tất cả edges có `modalities: ["visual"]` — không có transcript edge hay cross-modal edge.

**Nguyên nhân:** Hệ quả trực tiếp của B1. Không có caption → không extract được
entity pairs từ visual → không có visual relationship edges ngoài những gì được
cache từ lần chạy gốc.

---

### B3 — Cross-segment entity merge = 0 [CRITICAL]

**Triệu chứng:**

| Collection | Candidates | Gate pass | Accepted merges | Gate pass rate |
|---|---|---|---|---|
| 0 | 1 004 | 3 | 1 | 0.3% |
| 6 | 2 687 | 20 | 0 | 0.7% |
| 11 | 9 209 | 98 | 0 | 1.1% |

Hầu hết entity nodes có `first_seen == last_seen`, không có cross-segment links.

**Nguyên nhân:** OpenCLIP cosine gate quá strict + CLIP frame embeddings không đủ
discriminative cho generic YOLO classes (nhiều "zebra" khác nhau có embedding tương tự).

**Hậu quả:** Ablation A4 (`no_visual_identity_linking`) ≈ A1 vì A1 cũng = 0 merges.

---

### B4 — chunks_vdb embed text ẩn danh [HIGH]

**Triệu chứng:** `kv_store_text_chunks_v2.json` lưu content dạng:
`"OBJECT_352 gather together on OBJECT_353"` thay vì text gốc.
Query người dùng dùng tên thật ("synchrosqueezing", "Giselle Fernandez") không hit
được chunk nào vì từ đó không tồn tại trong embedded text.

**Nguyên nhân:** `unified_ingest.py` `_segment_storage_payload()` build chunk content
từ `rewritten_transcript` (đã entity-replace) và pass thẳng vào `chunks_vdb.upsert()`.

---

### B5 — Frames không được lưu portable [HIGH]

**Triệu chứng:** `alignment.py` `_load_frame_images()` dùng absolute paths từ
`frame_selections.json`. Khi re-run từ workdir khác, paths không valid → trả về `[]`
→ skip MiniCPM → caption trống.

---

### B6 — Reference resolution thấp (6–22%) [MEDIUM]

| Collection | Rate |
|---|---|
| 0 | 21.5% |
| 6 | 7.7% |
| 11 | 6.3% |

Đặc biệt lecture (Col 6) — "it", "this function", "the transform" gần như không
được resolve về antecedent. Ảnh hưởng đến chất lượng transcript entities.

---

## Phần 3 — Thiết kế mới đề xuất

### Nguyên tắc cốt lõi

1. **MiniCPM làm đúng 1 việc: visual perception.** Không OCR, không entity ID tracking.
   Output là plain-text caption mô tả cảnh, events, và entities với appearance+position.

2. **Tách V_ và T_ entity lists riêng biệt đến khi merge.** Visual entities từ YOLO
   dùng provisional prefix `V_`. Transcript entities từ NLP dùng provisional prefix `T_`.

3. **Sau merge: IDs phẳng, không prefix.** `V_ANIMAL_001` + `T_ANIMAL_003` → `ANIMAL_003`.
   T_ side làm canonical vì có tên thật từ transcript.

4. **GPT-4o-mini xử lý merge và graph building.** Đọc text descriptions, không nhận image.
   Giống baseline VideoRAG nhưng có thêm visual caption làm input.

5. **Progressive memory cho GPT-4o-mini stages.** Entity list tích lũy qua các batches.
   Batch đầu không có memory. Từ batch 2 trở đi có memory từ các batches trước.

6. **Transcript gốc được preserve.** Không rewrite IDs vào transcript. Embed transcript
   gốc vào chunks_vdb.

7. **MiniCPM chạy parallel như baseline.** Không phụ thuộc EntityRegistry, không cần
   sequential.

---

### Stage 5 — `tracking_base` (giữ nguyên logic, thêm position data)

**Thay đổi:** Khi build EntityRegistry từ YOLO tracklets, tính thêm relative position
từ bounding box cho mỗi observation và lưu vào provenance.

**Position calculation từ bbox `[x1, y1, x2, y2]`** (normalized 0–1):
```python
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
area = (x2 - x1) * (y2 - y1)

h_zone = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
v_zone = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
size = "small" if area < 0.05 else ("large" if area > 0.20 else "medium")

position_str = f"{v_zone}-{h_zone}, {size}"
# Ví dụ: "middle-left, large"
```

**Output schema thêm vào mỗi visual entity:**
```json
{
  "entity_id": "V_ANIMAL_001",
  "entity_type": "animal",
  "canonical_name": "zebra",
  "dominant_position": "middle-left, large",
  "position_history": [
    {"segment_id": "SEG_003", "position": "middle-left, large", "time": 42.5},
    {"segment_id": "SEG_005", "position": "bottom-center, medium", "time": 91.2}
  ]
}
```

**File output:** `tracking_base/registry.json` và `tracking_base/visual_entities.json`
(đã có, chỉ thêm field `dominant_position` và `position_history`).

---

### Stage 6 — `frame_selection` (bỏ overlay text)

**Thay đổi:** Trong `frame_selector.py` hàm `_draw_overlay()`:
- **GIỮ:** vẽ bounding box màu để debug
- **BỎ:** vẽ text entity ID lên ảnh (dòng `cv2.putText(...)` với label_text)

Frames lưu xuống không có text entity ID nào. MiniCPM sẽ nhận ảnh sạch.

**Không thay đổi gì khác ở stage này.**

---

### Stage 7 — `text_entities` (giữ nguyên, đây là T_ extraction)

Stage này đã làm đúng: extract T_ entities từ transcript gốc với progressive memory
qua `TextEntityExtractor`. Output `text_entities.json` chứa T_ provisional entities
với real names từ transcript.

**Không thay đổi.**

---

### Stage 8 — `alignment_caption` (REDESIGN HOÀN TOÀN)

Stage này được split thành 2 sub-steps:

#### Sub-step 8a — MiniCPM Visual Captioning (parallel subprocess)

Chạy trong separate process như baseline VideoRAG gốc (multiprocessing.Process).
Không phụ thuộc EntityRegistry.

**Input mỗi segment:**
- Frames đã chọn từ `frame_selection` (ảnh sạch, không có overlay text)
- Transcript gốc của segment (raw text, không có entity ID)

**Prompt MiniCPM:**
```
The transcript of this video segment:
{transcript}

Provide a detailed description of what is happening in this video segment in English.

Your description must include:
1. All events and actions occurring in the scene
2. The overall context and atmosphere
3. For each visible person or distinct object, describe:
   - Appearance (clothing color, size, distinguishing features)
   - Position in frame (e.g. "standing on the left", "in the foreground center")
   - What they are doing

Be specific about spatial relationships between entities and their positions.
```

**Output mỗi segment:** plain-text caption string. Không có entity ID. Không có
structured JSON. Giống hệt baseline VideoRAG output format.

**Lưu output:** `alignment_caption/segments/{segment_id}_caption.txt`
hoặc dict `{segment_id: caption_text}` trong `captions.json`.

#### Sub-step 8b — Cross-modal Entity Merge (GPT-4o-mini, sequential với memory)

Chạy **sau** khi 8a hoàn tất. Sequential, xử lý từng batch segments.

**Input:**
1. MiniCPM captions từ 8a
2. Visual entity list từ `tracking_base/visual_entities.json` (V_ IDs + positions)
3. Transcript entity list từ `text_entities/text_entities.json` (T_ IDs + real names)
4. **Entity memory tích lũy** từ các batches trước (trống ở batch đầu tiên)

**Batch size:** 5–10 segments (đủ để GPT-4o-mini có context, không quá dài).

**Entity memory format** (tích lũy dần):
```json
{
  "ANIMAL_001": {
    "canonical_name": "the female zebra",
    "entity_type": "animal",
    "visual_source": "V_ANIMAL_001",
    "transcript_source": "T_ANIMAL_003",
    "accumulated_descriptions": [
      "drinking water near the riverbank (seg3)",
      "fleeing from a predator at high speed (seg7)"
    ],
    "relationships": [
      "chased by ANIMAL_015 (predator, seg7)"
    ],
    "segments_seen": ["SEG_003", "SEG_007"]
  }
}
```

**Prompt GPT-4o-mini cho mỗi batch:**
```
You are building an entity knowledge graph from a video.

=== KNOWN ENTITIES (accumulated so far) ===
{entity_memory_json}

=== VISUAL ENTITIES FROM OBJECT TRACKING (this batch) ===
{visual_entities_with_positions}
Example format:
- V_ANIMAL_001: animal (zebra), dominant position: middle-left, large
- V_PERSON_002: person, dominant position: top-right, small

=== TRANSCRIPT ENTITIES (this batch) ===
{transcript_entities_with_names}
Example format:
- T_ANIMAL_003: "the female zebra" (mentioned at 42s, 91s)
- T_PERSON_001: "David Attenborough" (mentioned at 15s)

=== SEGMENTS TO PROCESS ===
For each segment below, you are given:
- The original transcript (real names, no IDs)
- The visual caption from MiniCPM (describes appearance and positions)

{segments_data}
Each segment formatted as:
SEGMENT {segment_id} [{start}s - {end}s]:
Transcript: {original_transcript}
Visual caption: {minicpm_caption}
Visual entities present: {v_entities_in_segment}

=== YOUR TASKS ===

**Task 1 — Entity matching:**
For each visual entity (V_xxx) that appears in a segment, match it to a transcript
entity (T_xxx) or known entity from memory, using:
- Appearance description in MiniCPM caption vs canonical_name
- Position in frame (V_ entity position vs caption description like "on the left")
- Context from transcript

Return matches as:
{"V_ANIMAL_001": "T_ANIMAL_003", "V_PERSON_002": null}
(null = no transcript match found)

**Task 2 — Relationship extraction:**
Extract entity relationships and events from each segment.
Use canonical IDs where known (from matches + memory), provisional IDs otherwise.
Format: {"source": "ANIMAL_001", "predicate": "chased_by", "target": "ANIMAL_015",
         "segment_id": "SEG_007", "description": "predator chases prey at high speed"}

**Task 3 — Memory update:**
For each known/matched entity, provide updated description for this batch.
Format: {"entity_id": "ANIMAL_001", "new_description": "fleeing from predator (seg7)"}

Return valid JSON with keys: "matches", "relationships", "memory_updates"
```

**Post-processing sau mỗi batch:**
1. Áp dụng matches: `V_ANIMAL_001` → canonical ID = T_ side nếu matched, V_ promoted nếu không
2. Update entity memory với `new_description` entries
3. Pass entity memory sang batch tiếp theo

**Output cuối stage 8:** `alignment_caption/alignment.json` chứa:
```json
{
  "registry": { ... },      // merged EntityRegistry (no V_/T_ prefixes)
  "segments": {
    "SEG_003": {
      "caption": "...",           // MiniCPM plain text caption
      "transcript": "...",        // transcript gốc (KHÔNG rewrite IDs)
      "edges": [ ... ],           // relationships từ GPT-4o-mini
      "entity_ids": ["ANIMAL_001", "ANIMAL_002"]  // entities xuất hiện
    }
  },
  "edge_count": 245,
  "entity_memory": { ... }   // final accumulated memory
}
```

---

### Stage 9 — `graph` (giữ logic, input thay đổi)

`build_unified_graph()` nhận `alignment["registry"]` và `alignment["segments"]`
như hiện tại. Không thay đổi graph building logic.

**Thay đổi duy nhất:** edges trong `alignment["segments"]` giờ có cả
visual edges và transcript edges (từ GPT-4o-mini merge), thay vì chỉ visual.

---

### Stage 10 — `index` (fix chunks_vdb embedding)

**Thay đổi:** Trong `_segment_storage_payload()`, field `"transcript"` trong chunk
content phải là **transcript gốc**, không phải `rewritten_transcript`.

```python
# HIỆN TẠI (sai):
"transcript": aligned.get("rewritten_transcript", "")

# MỚI (đúng):
"transcript": aligned.get("transcript", "")  # transcript gốc từ ASR
```

Chunk content format mới:
```
Caption:
{minicpm_plain_text_caption}
Transcript:
{original_transcript_no_entity_ids}
```

`entity_memory` field giữ lại để list entities xuất hiện, nhưng không embed IDs
vào transcript text.

---

## Phần 4 — Sơ đồ data flow mới

```
Video
  │
  ├─► YOLO+BoT-SORT ──────────────────► V_ entity list
  │       (tracking_base)                + positions (bbox)
  │
  ├─► Whisper ASR ────────────────────► words + timestamps
  │       (asr)
  │
  ├─► Frame selection ────────────────► frames (NO overlay text)
  │       (frame_selection)
  │
  ├─► Transcript entity extraction ───► T_ entity list
  │       (text_entities, sequential)    + real names
  │       GPT-4o-mini + memory
  │
  └─► MiniCPM captioning ─────────────► plain text captions
          (alignment_caption sub-step 8a)  (appearance + position + events)
          parallel subprocess, no entity IDs

              ▼
  Cross-modal merge (sub-step 8b)
  GPT-4o-mini, sequential batches, accumulative memory
  Input: V_ list + T_ list + captions + entity memory
  Output:
    - V_ANIMAL_001 ↔ T_ANIMAL_003 → ANIMAL_003 (T_ canonical)
    - V_PERSON_002 (no match) → PERSON_001 (promoted)
    - Relationship edges: ANIMAL_003 —[chased_by]→ ANIMAL_015

              ▼
  Graph building (graph)
  EntityRegistry (flat IDs, no prefix) + edges

              ▼
  Index (index)
  chunks_vdb ← original transcript (real names, no OBJECT_xxx)
  entities_vdb ← canonical names
```

---

## Phần 5 — File cần sửa

| File | Thay đổi |
|---|---|
| `videorag/_videoutil/frame_selector.py` | Bỏ `cv2.putText()` entity ID text trong `_draw_overlay()` |
| `videorag/_unified_graph/alignment.py` | Rewrite: bỏ `_validate_visual_analysis` strict filter; MiniCPM prompt mới; split thành 8a (MiniCPM) và 8b (GPT-4o-mini merge) |
| `videorag/pipeline/unified_ingest.py` | `_stage_alignment()`: gọi 8a rồi 8b; `_segment_storage_payload()`: dùng `transcript` gốc thay vì `rewritten_transcript` |
| `videorag/_unified_graph/schema.py` | Thêm `dominant_position` và `position_history` vào `EntityNode`/`ProvenanceRecord` |
| `videorag/_unified_graph/builder.py` | Giữ nguyên (đọc từ registry mới) |

**Không thay đổi:**
- `tracking_base` stage logic (chỉ thêm position fields)
- `text_entities` stage (`TextEntityExtractor` + sequential memory đã đúng)
- `graph` stage (`build_unified_graph`)
- `validation` stage
- Query pipeline (`EBR_RAG.py`, debate loop)

---

## Phần 6 — Các điểm cần chú ý khi implement

**1. MiniCPM subprocess pattern** — giữ đúng pattern của baseline:
```python
process_segment_caption = multiprocessing.Process(
    target=segment_caption,   # hàm mới trong caption.py
    args=(video_name, video_path, segment_index2name, transcripts,
          segment_times_info, captions, error_queue)
)
```
`segment_caption()` không nhận entity memory — chỉ nhận frames + transcript.
Output vào shared dict `captions`.

**2. Batch sequential cho sub-step 8b** — chạy trong main process sau khi subprocess 8a join:
```python
process_segment_caption.join()
# Sau đó:
alignment_result = run_crossmodal_merge_batched(
    captions,           # từ MiniCPM
    visual_entities,    # từ tracking_base
    text_entities,      # từ text_entities stage
    batch_size=8,
)
```

**3. Entity ID resolution cho entities_vdb** — `entities_vdb` embed canonical_name
(đã có trong registry). Không thay đổi.

**4. `entity_memory` field trong chunk content** — giữ lại để list entity IDs xuất hiện
trong segment (hữu ích cho LLM biết context), nhưng KHÔNG đưa IDs vào transcript text.
Format:
```
Entity Memory:
- ANIMAL_001 (the female zebra)
- ANIMAL_002 (elephant herd)

Caption:
A zebra drinks water while elephants approach from the right...

Transcript:
The female zebra needs to stay hydrated during migration...
```

**5. Ablation flags** — `disable_crossmodal_alignment=True` (A5 ablation) skip sub-step
8b hoàn toàn, chỉ giữ MiniCPM captions. `disable_visual_identity_linking=True` (A4
ablation) skip OpenCLIP cross-segment merge trong tracking_base.

**6. Checkpoint/resume** — sub-step 8a dùng caption checkpoint đã có
(`_caption_checkpoints/`). Sub-step 8b cần checkpoint riêng lưu `entity_memory` state
và `next_batch_index` để resume nếu crash giữa chừng.

---

## Phần 7 — Ablation validity sau fix

| Ablation | Validity sau fix |
|---|---|
| A0 — VideoRAG baseline | ✅ Không ảnh hưởng |
| A1 — full_framework | ✅ Hoạt động đúng sau fix B1+B3+B4 |
| A1 vs A4 — visual identity linking | ✅ Có thể đo được (B3 fixed) |
| A1 vs A5 — crossmodal alignment | ✅ Có thể đo được (B1 fixed) |
| A1 vs A6 — debate refinement | ✅ Đã valid, không đổi |
| A1 vs A7 — critique blindness | ✅ Đã valid, không đổi |
