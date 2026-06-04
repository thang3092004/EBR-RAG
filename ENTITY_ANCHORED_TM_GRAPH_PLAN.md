# Entity-Anchored TM Graph Plan

## 1. Muc tieu

Huong moi khong tiep tuc lam `temporal edges`. Chu `Temporal` trong TM Graph
duoc hieu la **temporary memory dua vao context khi trich xuat graph**, khong
phai canh thoi gian trong graph.

Muc tieu cua nhanh nay:

- Tao global entity IDs truoc khi caption va truoc khi graph extraction.
- Giam node mo ho nhu `he`, `she`, `man`, `woman`, `person`, `object`.
- Bien TM Graph thanh entity-centric memory thay vi text-name memory.
- Giu huong train-free: dung pretrained models nhu observation extractors, con
  dong gop nam o constrained entity linking va cach dua entity memory vao graph.

## 2. Kien truc de xuat

```text
Long video
  -> segment / shot metadata
  -> visual detection + OCR + optional audio ASR/diarization
  -> local tracklets
  -> constrained global entity linking
  -> entity memory packets per segment
  -> ID-overlaid frames / ID-aware captioning
  -> Entity-Anchored TM Graph extraction
  -> retrieval + reasoning / debate
```

## 3. Module moi nen them

Tao package moi:

```text
videorag/_entity_anchor/
  __init__.py
  schema.py
  detector.py
  tracker.py
  linker.py
  memory.py
  overlay.py
  audio.py
```

Vai tro:

- `schema.py`: dataclass/schema cho `EntityObservation`, `Tracklet`,
  `GlobalEntity`, `EntityMemoryPacket`.
- `detector.py`: wrapper cho YOLO/RT-DETR/GroundingDINO/SAM2 tuy setup.
- `tracker.py`: tao local tracklets bang ByteTrack/BoT-SORT/OC-SORT.
- `linker.py`: merge tracklets thanh global IDs bang similarity + constraints.
- `memory.py`: tao memory ngan gon cho tung segment.
- `overlay.py`: ve bbox + entity ID len frame truoc khi dua vao VLM.
- `audio.py`: optional ASR, speaker diarization, active speaker linking.

## 4. Schema can luu

Vi du:

```json
{
  "entity_id": "PERSON_001",
  "type": "person",
  "tracklets": ["seg_0003_track_02", "seg_0004_track_01"],
  "observations": [
    {
      "segment_id": "video_001_segment_003",
      "time": [90.0, 118.5],
      "bbox": [121, 44, 382, 511],
      "confidence": 0.91
    }
  ],
  "aliases": ["woman in white shirt", "protagonist"],
  "attributes": ["white shirt", "near table"],
  "linked_speakers": [
    {"speaker_id": "SPEAKER_001", "confidence": 0.78}
  ],
  "evidence_chunks": ["chunk_003", "chunk_004"]
}
```

Nen luu vao:

```text
<working_dir>/entity_anchor/entities.json
<working_dir>/entity_anchor/tracklets.json
<working_dir>/entity_anchor/segment_memory.json
```

## 5. Thuat toan global entity linking

Moi tracklet `i` co feature:

```text
visual embedding: CLIP/SigLIP/DINOv2 crop embedding
person ReID embedding: OSNet/torchreid neu la person
face embedding: InsightFace/ArcFace neu co face ro
audio speaker embedding: ECAPA/pyannote neu co lien ket active speaker
color histogram
time span
bbox trajectory
OCR text neu la screen element
```

Score de xuat:

```text
S(i, j) =
  w_visual * cos(visual_i, visual_j)
+ w_reid   * cos(reid_i, reid_j)
+ w_face   * cos(face_i, face_j)
+ w_audio  * cos(speaker_i, speaker_j)
+ w_time   * temporal_compatibility(i, j)
+ w_motion * motion_compatibility(i, j)
- lambda_conflict * conflict(i, j)
```

Constraints:

- Same frame, khac bbox ro rang: cannot-link.
- Khac entity type: cannot-link.
- Cung continuous tracklet: must-link.
- Face mismatch ro: cannot-link manh.
- Speaker chi merge vao person khi active-speaker confidence cao.
- Clothes/color chi la weak evidence, khong du de merge mot minh.

Baseline train-free:

1. Merge trong cung segment bang tracker output.
2. Merge giua cac segment gan nhau bang Hungarian matching.
3. Merge global bang constrained agglomerative clustering hoac DBSCAN/HDBSCAN
   tren affinity matrix.

## 6. Tich hop vao code hien tai

Vi frame caption hien tai lay kha thua, khong nen dung sample caption de track.
Nen them mot branch rieng cho tracking:

```text
entity_tracking_fps = 3
entity_tracking_resolution = 640
entity_tracking_vid_stride = auto
```

Hook de xuat trong `VideoRAG.insert_video()`:

1. Sau khi co video path va segment metadata.
2. Chay entity anchoring tren original video hoac cached segment videos.
3. Tao `segment_memory`.
4. Khi caption moi segment, dua vao:
   - ID-overlaid frames, hoac
   - memory text voi entity IDs, tot nhat la ca hai.
5. Khi goi `extract_entities_tm`, dua entity memory vao prompt va ep node name
   uu tien `PERSON_001`, `OBJECT_014`, `SCREEN_001`, `SPEAKER_001`.

## 7. TM Graph sau khi co entity IDs

TM Graph khong can chinh de giai quyet identity bang text nua. Vai tro moi:

- Nho semantic aliases cua tung entity.
- Nho attributes va actions da xuat hien truoc do.
- Nho OCR/audio links.
- Nho evidence chunks de retrieval.
- Giam duplicate nodes khi sinh graph tu cac chunk sau.

Vi du context moi:

```text
Active entity memory:
PERSON_001: woman in white shirt, protagonist, previously held OBJECT_014.
PERSON_002: man in black jacket, talked to PERSON_001 near the table.
OBJECT_014: red phone, last seen on the table.
SPEAKER_001: off-screen narrator, explains rules in this segment.
```

Graph extraction mong muon:

```text
("PERSON_001", "hands", "OBJECT_014")
("PERSON_002", "looks_at", "OBJECT_014")
("SPEAKER_001", "mentions", "RULE_003")
```

## 8. Ke hoach thuc hien

### Phase 1: Train-free MVP

- Them config flags:
  - `enable_entity_anchoring`
  - `entity_tracking_fps`
  - `entity_tracking_model`
  - `entity_tracking_conf`
  - `entity_memory_top_k`
- Tao schema va storage JSON.
- Dung YOLO + ByteTrack/BoT-SORT cho person/object/animal tracklets.
- Tao global IDs don gian bang IoU/time/ReID/CLIP similarity.
- Tao ID-overlaid frames cho caption.
- Sua caption prompt de dung IDs.
- Sua TM extraction prompt de uu tien entity IDs.

### Phase 2: Entity memory and retrieval

- Tao `segment_memory.json`.
- Index aliases/attributes cua entity vao vector DB.
- Khi query co "woman", "man", "protagonist", map ve entity candidates truoc.
- Retrieval lay ca graph chunks va entity evidence chunks.

### Phase 3: Audio optional

- Them Whisper/faster-whisper transcript.
- Them speaker diarization bang pyannote/WhisperX neu setup duoc.
- Tao `SPEAKER_ID`.
- Chi merge `SPEAKER_ID` voi `PERSON_ID` neu active speaker confidence cao.
- Off-screen narrator giu thanh speaker entity rieng.

### Phase 4: Stronger linking, van train-free

- Them cannot-link/must-link constraints.
- Them constrained agglomerative clustering.
- Them conflict report:
  - same-frame ID conflict
  - duplicate entity names
  - ambiguous pronoun/entity nodes

### Phase 5: Optional small training neu co thoi gian

Chi nen lam neu MVP da chay:

- Train mot matcher nho `same_entity(tracklet_i, tracklet_j)`.
- Positive tu same continuous track.
- Negative tu same-frame different boxes.
- Hard negatives tu tracklets nhin giong nhau nhung cannot-link.

Day la optional, khong phai core cua huong hien tai.

## 9. Ablation de bao ve contribution

Chay cac setting:

```text
VideoRAG baseline
TM Graph text-only hien tai
TM Graph + local tracking IDs
TM Graph + global constrained entity linking
TM Graph + entity-aware caption
TM Graph + entity-aware caption + audio/OCR evidence
```

Metric nen bao cao:

- QA accuracy.
- Retrieval recall@k neu co evidence labels.
- So node mo ho: `he`, `she`, `it`, `man`, `woman`, `person`.
- Duplicate entity node rate.
- Same-frame ID conflict rate.
- Manual entity consistency tren mot subset nho.

## 10. Diem can noi voi giao vien

Thong diep chinh:

> Cac pretrained models chi la observation extractors. Dong gop cua do an la
> mot train-free entity anchoring module va cach tich hop no vao Temporary
> Memory Graph de tao graph on dinh hon cho long-video QA.

Khong nen noi:

> Em chi them YOLO/SAM/Whisper vao pipeline.

Nen noi:

> Em them mot tang dinh danh thuc the truoc khi sinh graph. Tang nay dung
> tracking, embedding similarity, va hard constraints de tao global entity IDs.
> Sau do TM Graph duoc neo theo entity IDs, giup giam ambiguity va duplicate
> nodes khi trich xuat graph tu video dai.
