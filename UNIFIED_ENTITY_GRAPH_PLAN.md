# Unified Entity Graph Architecture

## 1. Muc tieu

Pipeline `unified-graph-v2-correspondence` tao mot knowledge graph thong nhat tu
hai kenh evidence:

- Visual entities tu frame video.
- Text entities tu transcript.

Hai kenh duoc xu ly doc lap bang cac model chuyen dung, sau do MiniCPM chi lam
phan can suy luan da duoc rang buoc:

- Can chinh visual entity voi text entity.
- Viet caption bang global entity ID.
- Sinh relation giua cac global entity.

Pipeline khong con:

- Phan loai `visual-rich`, `speech-rich`, hay modality routing.
- OCR.
- Speaker diarization.
- Active-speaker detection.
- Voice identity.
- Tu dong tao speaker/claim node.
- Deep-processing branch theo tung modality.

Moi segment deu di qua cung mot duong xu ly.

## 2. So do tong the

```text
Input video
    |
    v
[1] Media probe
    |
    +--------------------------+
    |                          |
    v                          v
[2] Full-video ASR        [3] Shot/motion analysis
    |                          |
    +------------+-------------+
                 v
        [4] Smart segmentation
                 |
          +------+------+
          |             |
          v             v
[5] Visual tracking  [7] Transcript entities
          |
          v
[6] Diverse frame selection
          |             |
          +------+------+
                 v
[8] MiniCPM visual facts + train-free correspondence + constrained alignment
                 |
                 v
[9] Unified global-ID graph
                 |
                 v
[10] Text/entity/video embedding indexes
                 |
                 v
[11] Graph validation
```

Stage graph trong code:

```text
probe
-> asr
-> shot_detection
-> segmentation
-> tracking_base
-> frame_selection
-> text_entities
-> alignment_caption
-> graph_build
-> embedding_index
-> validation
```

## 3. Stage 1: Media probe

### Muc dich

Lay metadata on dinh cua video de cac stage sau co chung mot he toa do thoi
gian.

### Cong cu

- Chinh: `ffprobe`.
- Fallback: OpenCV `VideoCapture`.

### Input

- Duong dan video.

### Output

```json
{
  "duration": 612.4,
  "fps": 29.97,
  "frame_count": 18354,
  "width": 1920,
  "height": 1080,
  "has_audio": true,
  "video_codec": "vp9",
  "audio_codec": "opus"
}
```

Metadata nay cung duoc dung nhu input fingerprint. Neu file doi size hoac
modification time, pipeline invalidates stage va cac stage phu thuoc.

## 4. Stage 2: Full-video ASR

### Model

Mac dinh:

```text
Systran/faster-distil-whisper-large-v3
```

Runtime:

- Framework: `faster-whisper`.
- GPU: `float16`.
- CPU fallback: `int8`.
- Beam size: `5`.
- VAD filter: bat.
- Word timestamps: bat.

### Ky thuat

Video duoc transcript mot lan tren toan bo timeline. Pipeline khong cat audio
thanh segment truoc khi ASR, vi cat truoc co the lam mat cau, mat tu, hoac tao
timestamp khong lien tuc.

Moi word luu:

```json
{
  "start": 31.24,
  "end": 31.61,
  "text": "John",
  "probability": 0.94,
  "segment_index": 12
}
```

Whisper utterance cung duoc luu de debug, nhung smart segmentation va text
entity extraction dung word-level timestamps.

Neu video khong co audio stream, stage tra ve danh sach word rong; pipeline
van tiep tuc theo kenh visual.

## 5. Stage 3: Shot va motion analysis

### Cong cu

- PySceneDetect `ContentDetector`.
- OpenCV fallback va bo sung metric.

### Ky thuat shot detection

OpenCV sample mac dinh `2 FPS`. Moi frame duoc resize ve `160x90`, chuyen HSV,
sau do tinh histogram `24x16`.

Khoang cach shot:

```text
shot_score = BhattacharyyaDistance(hist_t-1, hist_t)
```

Fallback danh dau shot khi:

```text
shot_score >= 0.38
```

Neu PySceneDetect san sang, `ContentDetector(threshold=27)` quyet dinh scene
boundary; OpenCV metric van duoc giu de cung cap score.

### Metric khac

```text
motion_score = mean(abs(gray_t - gray_t-1)) / 255
blur_score   = variance(Laplacian(gray_t))
```

`blur_score` duoc dung cho frame selection. Motion metric duoc luu cho debug
va phan tich video, khong con dung de chia visual-rich/speech-rich.

## 6. Stage 4: Smart segmentation

### Muc tieu

Chon boundary dung ca visual va transcript ma khong cat giua mot word.

Thong so mac dinh:

```text
target = 24 giay
minimum = 8 giay
maximum = 45 giay
context padding = 1.5 giay
```

### Boundary candidates

Candidate duoc tao tu:

- Dau va cuoi video.
- Shot boundary.
- Cuoi moi word.
- Khoang pause giua hai word.
- Dau cau co `.`, `?`, `!`, `:`, `;`.
- Moc forced gan target duration.
- Moc hard-limit gan maximum duration.

Score cua boundary:

```text
B(t) =
    0.40 * shot_score
  + 0.35 * pause_score
  + 0.15 * sentence_end
  + 0.10 * forced_boundary
```

Neu boundary nam ben trong mot word:

```text
speech_split(t) = 1
```

### Dynamic programming

Voi segment tu boundary `i` den `j`:

```text
length_cost = ((duration(i,j) - target) / target)^2

transition_cost =
    length_cost
  + short_final_penalty
  - B(j)
  + 4.0 * speech_split(j)
```

Dynamic programming tim chuoi boundary co tong cost nho nhat, dong thoi ep
segment nam trong `[minimum, maximum]` tru segment cuoi.

### Context padding

Moi segment luu hai cap timestamp:

```text
start, end
context_start = start - 1.5s
context_end   = end + 1.5s
```

`start/end` la thoi gian chinh de gan transcript va graph edge.
`context_start/context_end` chi dung khi chon frame, giup giu visual context
quanh boundary.

## 7. Stage 5: Visual entity extraction va tracking

### Models

- Detector: Ultralytics YOLOv8n, file `yolov8n.pt`.
- Tracker: BoT-SORT, config `botsort.yaml`.
- Appearance embedding: OpenCLIP `ViT-B-32`.
- OpenCLIP weights: `laion2b_s34b_b79k`.

### Runtime

```text
tracking FPS = 3
YOLO confidence = 0.25
YOLO IoU = 0.5
input size = 640
checkpoint chunk = 60 giay
```

Video duoc xu ly theo chunk 60 giay de resume. Tracker state reset dau moi
chunk, sau do global linker noi cac tracklet lai.

### Observation schema

Moi detection da track tao:

```json
{
  "segment_id": "SEG_00003",
  "tracklet_id": "V_PERSON_00001_4",
  "local_track_id": "chunk00001:4",
  "frame_index": 2871,
  "time": 95.80,
  "bbox": [121.0, 44.0, 382.0, 511.0],
  "confidence": 0.91,
  "label": "person",
  "entity_type": "person"
}
```

Entity type mapping:

- YOLO `person` -> `person`.
- COCO animal labels -> `animal`.
- Cac class con lai -> `object`.

### Tracklet appearance

Moi tracklet giu representative crop co detection confidence cao nhat.

Hai feature appearance duoc tinh:

1. HSV histogram tren crop.
2. OpenCLIP normalized image embedding.

### Cross-chunk visual linking

Chi xet merge hai tracklet khi:

- Cung entity type.
- Khoang thoi gian khong overlap.

Day la hard cannot-link quan trong: hai bbox xuat hien dong thoi khong the la
cung mot physical entity.

Similarity:

```text
S_visual(i,j) =
    0.45 * cosine(OpenCLIP_i, OpenCLIP_j)
  + 0.20 * same_detector_class
  + 0.15 * motion_compatibility
  + 0.10 * exp(-time_gap / 300)
  + 0.10 * cosine(HSV_i, HSV_j)
```

Trong do:

```text
motion_compatibility =
    0                                      if time_gap > 3s
    exp(-center_distance / bbox_scale)     otherwise
```

Merge khi:

```text
S_visual >= 0.85
```

Connected components duoc tao bang disjoint-set union. Moi component duoc cap
global ID:

```text
PERSON_001
OBJECT_002
ANIMAL_001
```

Tracklet ID nhu `V_PERSON_00001_4` duoc giu nhu alias. Do do MiniCPM nhan
global visual ID tren frame, khong phai tu tu suy ra identity visual.

### Provenance

Moi visual observation duoc luu vao node:

```json
{
  "source": "visual",
  "video_id": "video_01",
  "segment_id": "SEG_00003",
  "start": 95.8,
  "end": 95.8,
  "frame_time": 95.8,
  "bbox": [121.0, 44.0, 382.0, 511.0],
  "confidence": 0.91
}
```

## 8. Stage 6: Diverse frame selection

### Cong cu

Chi dung OpenCV va visual tracking output; khong dung LLM.

### Candidate timestamps

Moi segment tao candidate tu:

- `context_start`.
- Midpoint.
- `context_end`.
- Moi moc 1 giay.
- Shot boundaries.
- First/last observation cua moi visual entity.

### Quality filtering

Do net:

```text
quality = variance(Laplacian(gray_frame))
```

Neu co it nhat 5 candidate, bo 20% frame mo nhat.

### Duplicate detection

Frame duoc bieu dien boi normalized HSV histogram `24x16`.

```text
similarity(i,j) = cosine(hist_i, hist_j)
```

Frame moi bi bo neu:

```text
max_similarity_with_selected >= 0.94
```

### Greedy marginal gain

Moi candidate co score:

```text
gain(frame) =
    0.35 * visual_diversity
  + 0.30 * new_entity_coverage
  + 0.20 * new_temporal_bin
  + 0.15 * normalized_sharpness
```

Trong do:

```text
visual_diversity = 1 - max cosine similarity voi frame da chon
new_entity_coverage = so entity moi / tong entity trong segment
new_temporal_bin = 1 neu frame phu mot khoang thoi gian chua duoc phu
```

Pipeline chon toi thieu 2, toi da 6 frame. Sau khi da dat minimum, dung neu:

```text
best_gain < 0.05
```

Vi vay 6 chi la upper bound; pipeline khong co gang lay du 6 frame neu cac
frame con lai trung lap hoac khong them evidence.

### ID overlay

Truoc khi luu, moi frame duoc ve:

- Bounding box mau do.
- Global visual entity ID, vi du `PERSON_001`.

Day la image input truc tiep cua MiniCPM.

## 9. Stage 7: Transcript entity extraction

### Model

Mac dinh:

```text
spaCy en_core_web_trf
```

Fallback:

```text
en_core_web_sm
-> spacy.blank("en")
-> regex proper-noun fallback neu spaCy khong ton tai
```

### Transcript spans

Words trong segment duoc noi thanh mot span cho den khi gap giua hai word lon
hon `0.8s`.

Khong gom theo speaker va khong tao speaker ID.

### Entity extraction

SpaCy NER mapping:

```text
PERSON      -> person
ORG         -> organization
GPE/LOC/FAC -> location
EVENT       -> event
PRODUCT     -> object
WORK_OF_ART -> object
NORP/LAW/LANGUAGE -> concept
```

Noun chunks bo sung object/concept entities voi confidence thap hon.

Pronoun va generic noun khong con bi bo. Chung duoc chia thanh hai truong hop:

- Gioi thieu indefinite nhu `a man`, `a woman`, `someone`, `something`:
  tao provisional entity moi voi confidence thap.
- Referring expression nhu `he`, `she`, `it`, `they`, `the man`, `the object`:
  dua vao discourse-memory resolver.

### Text entity identity

Moi mention duoc normalize:

- Lowercase.
- Bo article/title dau chuoi.
- Bo punctuation.
- Collapse whitespace.

Voi mention moi va mention cu cung type:

```text
lexical = SequenceMatcher(normalized_new, normalized_old)
semantic = cosine(spaCy_vector_new, spaCy_vector_old)
score = max(lexical, semantic)
```

Tai su dung text ID khi:

```text
best_score >= 0.88
and best_score - second_best_score >= 0.05
```

Neu khong, cap provisional ID moi:

```text
T_PERSON_001
T_OBJECT_001
T_LOCATION_001
T_EVENT_001
T_CONCEPT_001
```

Transcript duoc rewrite:

```text
[31.24s -> 34.90s]: [T_PERSON_001] holds [T_OBJECT_001].
```

Stage nay khong dung generative LLM.

### Short-term transcript memory

Mac dinh giu:

```text
32 mention occurrences gan nhat
12 lightweight transcript events gan nhat
12 active entities gan nhat trong prompt context
```

Moi mention occurrence luu:

- Text ID.
- Entity type va canonical label.
- Timestamp va segment.
- Grammatical role: subject, object, possessor, oblique.
- Day la explicit mention hay reference.

Neu spaCy parser san sang, moi transcript span con luu root predicate va danh
sach entity IDs trong span. Vi du:

```json
{
  "predicate": "hold",
  "entity_ids": ["T_PERSON_001", "T_OBJECT_003"],
  "text": "[T_PERSON_001] holds [T_OBJECT_003]."
}
```

### Long-term transcript memory

Moi text entity co mot state:

```json
{
  "text_id": "T_PERSON_001",
  "canonical_label": "John",
  "aliases": ["John", "John Smith"],
  "first_seen": 12.4,
  "last_seen": 95.8,
  "mention_count": 8,
  "last_role": "subject",
  "descriptors": ["male"],
  "recent_predicates": ["arrive", "hold"]
}
```

Entity duoc xep hang trong long-term memory:

```text
memory_rank =
    0.45 * recency
  + 0.35 * mention_frequency
  + 0.20 * subject_ratio
```

Mac dinh toi da 40 entity state duoc dua vao long-term context.

### Coreference scoring

Voi moi reference, pipeline chi xet entity co type, number va descriptor khong
mau thuan.

```text
S_coref(reference, entity) =
    0.30 * recency
  + 0.15 * segment_continuity
  + 0.20 * entity_salience
  + 0.15 * grammatical_role_continuity
  + 0.10 * type_compatibility
  + 0.10 * descriptor_compatibility
```

Trong do:

```text
recency = exp(-time_gap / 120s)
segment_continuity = exp(-segment_gap / 3)
```

Reference duoc resolve khi:

```text
best_score >= 0.66
and best_score - second_best_score >= 0.10
```

Vi du:

```text
Segment 1: A man arrived.
           -> T_PERSON_001

Segment 2: He sat down.
           -> T_PERSON_001 sat down.
```

Neu co hai candidate gan nhau:

```text
John met David. He left.
```

pipeline khong doan bua:

```xml
<unresolved-ref id="R_000001">He</unresolved-ref>
```

Reference metadata van giu top-3 candidates va score components de MiniCPM co
them visual context ma quyet dinh relation. Reference unresolved khong tu tao
graph node.

`I`, `you`, `we` va cac bien the khong tu resolve vi pipeline khong co speaker
identity. Chung luon duoc giu nhu unresolved references.

### Resume

Sau moi segment, extractor checkpoint:

- Text-ID counters.
- Alias index.
- spaCy alias vectors.
- Short-term mentions/events.
- Long-term entity states.
- Reference counter.

Do do resume giua text stage khong lam mat memory hoac cap lai entity ID.

## 10. Stage 8: Cross-modal alignment va graph extraction

### Model

```text
MiniCPM-V-2_6-int4
```

Runtime:

- `transformers==4.40.0`.
- `torch.bfloat16`.
- Device: CUDA.
- Attention: SDPA.
- `max_slice_nums=2`.
- `max_new_tokens=450`.

MiniCPM la generative model duy nhat trong graph-ingest core.

### Pha A: visual-only fact extraction

MiniCPM duoc goi voi frame va visual global IDs, nhung khong nhan transcript.
Output chi gom visual caption, entity descriptions va visual edges. Visual
edges hop le duoc giu lai ngay ca khi visual va transcript khong co
correspondence.

### Candidate generation truoc correspondence gate

Pipeline khong dua moi cap visual-text vao model.

Deterministic alias resolution:

- Neu text label da resolve duoc trong registry va khong phai generic noun,
  map truc tiep ve global ID.

Candidate whitelist:

- Cung entity type.
- Hoac cap `object/concept`.
- Toi da 4 visual candidates cho moi text mention.
- Toi da 12 candidate cho mot segment.

Day moi chi la type-compatible raw whitelist, chua phai merge candidate cuoi.

### Transcript propositions

spaCy transcript memory da luu moi span nhu mot lightweight proposition gom
root predicate, entity IDs, timestamp va rewritten text. Khong goi them LLM de
tao proposition.

### Train-free correspondence gate

Visual fact text va transcript proposition text duoc dua qua OpenCLIP
`ViT-B-32` text encoder. Timestamp va entity ID duoc thay bang token trung
tinh truoc khi embed.

```text
S(f, p) = cosine(OpenCLIP_text(f), OpenCLIP_text(p))
```

Voi moi cap `(visual_id, text_id)`, score la similarity lon nhat trong cac
visual fact chua `visual_id` va transcript proposition chua `text_id`.

```text
score >= 0.28
and mutual best match theo ca visual_id va text_id
and visual-side margin >= 0.04
and text-side margin >= 0.04
```

`0.28` va `0.04` la tham so cau hinh, khong phai confidence do MiniCPM tu
khai. Bao cao thuc nghiem can co sensitivity analysis tren sau collection
benchmark da chon. Gate fail-closed khi thieu evidence, score thap hoac
ambiguous.

### Pha B: constrained MiniCPM alignment

Moi segment gom:

1. 2-6 frame da ve bbox va global visual ID.
2. Visible visual entities:

```json
{
  "entity_id": "PERSON_001",
  "entity_type": "person",
  "label": "person",
  "first_seen": 90.1,
  "last_seen": 112.4,
  "confidence": 0.88
}
```

3. Transcript entities.
4. Rewritten transcript.
5. Visual-only facts.
6. Transcript propositions.
7. Merge candidates da qua correspondence gate, kem score va paired evidence.
8. 8 recent graph events.
9. Toi da 20 entity state gan day, trong cua so 120 giay.
10. Toi da 3 recent relations cho moi entity.
11. Short-term va long-term transcript discourse memory.
12. Current resolved/unresolved references cung top candidate scores.

### Output MiniCPM

```json
{
  "caption": "PERSON_001 holds OBJECT_003.",
  "entity_merges": [
    {
      "visual_id": "PERSON_001",
      "text_id": "T_PERSON_002",
      "reason": "The named person matches the boxed person."
    }
  ],
  "edges": [
    {
      "source": "PERSON_001",
      "predicate": "holds",
      "target": "OBJECT_003",
      "confidence": 0.91,
      "modalities": ["visual"]
    }
  ]
}
```

### Guardrails

- MiniCPM khong duoc invent ID.
- Merge ngoai candidate whitelist bi loai.
- Raw candidate khong qua correspondence gate khong duoc dua cho MiniCPM.
- Mot person visible va mot person name cung segment khong du de merge.
- Mutual-best va ambiguity margin ngan merge khong ro rang.
- Confidence bi clamp vao `[0,1]`.
- Modality chi nhan `visual` va `transcript`.
- Self-edge bi loai.
- Edge co source/target khong resolve duoc trong registry bi loai.
- Neu output khong parse duoc, MiniCPM duoc goi them mot lan de repair JSON.

### Globalization

Text entity da merge:

```text
T_PERSON_002 -> PERSON_001
```

Text entity khong merge se duoc cap global node moi:

```text
T_LOCATION_001 -> LOCATION_001
T_CONCEPT_001  -> CONCEPT_001
```

Graph cuoi khong co node bat dau bang `V_` hoac `T_`.

## 11. Graph memory giua cac segment

### Short-term memory

Mac dinh giu 8 event gan nhat:

```json
{
  "source": "PERSON_001",
  "predicate": "holds",
  "target": "OBJECT_003",
  "segment_id": "SEG_00004"
}
```

### Structured long-term state

Moi source entity giu toi da 5 relation gan nhat:

```json
{
  "predicate": "holds",
  "target": "OBJECT_003",
  "segment_id": "SEG_00004",
  "time": [118.0, 139.5]
}
```

Day la graph-side structured memory, bo sung cho transcript discourse memory
o Stage 7. Ca hai deu khong phai LLM-generated story summary, nen giam token,
de resume va de debug hon.

Khi mot text alias da merge o segment truoc, segment sau thay:

```text
PERSON_001
```

thay vi `T_PERSON_002`.

## 12. Stage 9: Unified graph storage

### Storage

```text
NetworkX MultiDiGraph
```

Implementation:

```text
UnifiedNetworkXStorage
```

Graph la directed multigraph. Cung mot cap node co the co nhieu edge occurrence
o nhieu segment hoac thoi diem khac nhau.

### Node schema

```json
{
  "entity_id": "PERSON_001",
  "entity_type": "person",
  "canonical_name": "John",
  "aliases": [
    {"alias_id": "V_PERSON_00001_4", "source": "visual"},
    {"alias_id": "T_PERSON_002", "source": "transcript", "label": "John"}
  ],
  "sources": ["visual", "transcript"],
  "first_seen": 31.24,
  "last_seen": 139.50,
  "confidence": 0.91,
  "attributes": {},
  "provenance": []
}
```

### Edge occurrence schema

```json
{
  "edge_id": "EDGE_video_0000042",
  "source_id": "PERSON_001",
  "target_id": "OBJECT_003",
  "predicate": "holds",
  "start": 118.0,
  "end": 139.5,
  "segment_id": "SEG_00004",
  "confidence": 0.91,
  "modalities": ["visual"],
  "provenance": [
    {
      "source": "minicpm_alignment",
      "video_id": "video",
      "segment_id": "SEG_00004",
      "start": 118.0,
      "end": 139.5
    }
  ]
}
```

Thoi gian nam tren:

- Node provenance va `first_seen/last_seen`.
- Tung edge occurrence.

Node khong bi tach theo segment.

## 13. Stage 10: Embedding indexes

Stage nay khong tao graph moi. No tao index de retrieval sau ingest.

### Text chunk index

Noi dung segment gom:

- Entity memory.
- MiniCPM caption.
- Rewritten transcript.

Embedding mac dinh:

```text
OpenAI text-embedding-3-small
dimension = 1536
```

Storage:

```text
NanoVectorDB
```

### Entity index

Moi entity duoc embed tu:

```text
global ID + canonical name + entity type + alias labels
```

Model va storage giong text chunk index:

```text
text-embedding-3-small + NanoVectorDB
```

### Video segment index

Moi segment duoc cat lai thanh clip MP4 bang FFmpeg.

Model:

```text
ImageBind-Huge
dimension = 1024
```

Khi ingest:

- Encode clip bang ImageBind vision encoder.

Khi query:

- Encode query text bang ImageBind text encoder.
- Tim clip gan nhat trong cung cross-modal space.

ImageBind chi phuc vu visual retrieval, khong phuc vu entity merge.

## 14. Stage 11: Validation

Pipeline fail neu:

- Co provisional node `V_*` hoac `T_*` lot vao graph.
- Edge tham chieu node khong ton tai.
- Edge khong co provenance.
- Co segment khong co alignment result.

Validation report luu node count, edge count va danh sach loi.

## 15. Resume, progress va reproducibility

Moi stage co:

- Dependency list.
- Config keys rieng.
- Config hash.
- Status: pending/running/done/failed/interrupted.
- Output folder rieng.
- Metrics.

Neu config cua mot stage doi, stage do va toan bo descendants bi invalidated.

Tracking checkpoint moi 60 giay.
Frame selection checkpoint moi segment.
MiniCPM alignment checkpoint moi segment.

File theo doi:

```text
<workdir>/pipeline_v2/<video_id>/
  manifest.json
  progress.json
  events.jsonl
  pipeline.log
  run_report.json
```

`progress.json` luu:

- Stage dang chay.
- Completed/total.
- Elapsed time.
- ETA.
- Model va tham so chinh.
- Stage metrics.

## 16. Model inventory

| Giai doan | Model/cong cu | Vai tro |
|---|---|---|
| Probe | FFprobe, OpenCV fallback | Metadata video |
| ASR | faster-distil-whisper-large-v3 | Transcript va word timestamps |
| Shot | PySceneDetect + OpenCV | Scene boundaries, motion, blur |
| Segmentation | Dynamic programming | Boundary optimization |
| Detection | YOLOv8n | Person/object/animal detection |
| Tracking | BoT-SORT | Local identity trong chunk |
| Appearance | OpenCLIP ViT-B-32 | Cross-tracklet visual similarity |
| Frame selection | OpenCV HSV + Laplacian | Diversity, quality, entity coverage |
| Text entities | spaCy en_core_web_trf | NER, noun chunks, text vectors |
| Visual facts | MiniCPM-V-2_6-int4 | Visual-only descriptions and relations |
| Correspondence | OpenCLIP ViT-B-32 text encoder | Fact/proposition cosine gate |
| Alignment/graph | MiniCPM-V-2_6-int4 | Constrained merge, caption, relation proposal |
| Graph storage | NetworkX MultiDiGraph | Global nodes va edge occurrences |
| Text/entity index | text-embedding-3-small | Dense text/entity retrieval |
| Video index | ImageBind-Huge | Text-to-video segment retrieval |
| Vector storage | NanoVectorDB | Persist vector indexes |

## 17. Phan nao dung generative model

Trong graph ingest:

- Whisper la ASR model, khong phai generative reasoning LLM.
- YOLO, BoT-SORT, OpenCLIP, spaCy, ImageBind la pretrained specialist models.
- MiniCPM la generative VLM duy nhat de xuat visual facts, caption, merge va
  relations; deterministic code moi validate va ghi graph.
- GPT khong tham gia caption hoac graph extraction.
- OpenAI API chi duoc dung cho embedding index neu giu `openai_config`.

## 18. Gioi han hien tai

1. Visual identity linking dua nhieu vao OpenCLIP va detector class. Nguoi thay
   quan ao, goc nhin rat khac hoac bien mat lau co the bi tach ID.
2. OpenCLIP la generic image embedding, khong phai person-ReID model chuyen
   dung.
3. YOLOv8n chi phat hien class trong detector vocabulary. Cac object nho hoac
   domain-specific co the bi bo sot.
4. `en_core_web_trf` toi uu cho tieng Anh. Transcript ngon ngu khac can spaCy
   model tuong ung.
5. Coreference resolver la train-free heuristic, khong phai mot end-to-end
   neural coreference model. No co the de unresolved nhieu hon mong muon, nhung
   duoc thiet ke de tranh merge sai khi co nhieu antecedent gan nhau.
6. OpenCLIP text similarity khong phai identity proof. No chi loai cac cap
   khong co semantic correspondence ro rang.
7. MiniCPM relation extraction van co kha nang sai semantic, du da co
   correspondence gate, whitelist va ID validation.
8. Graph ingest khong biet ai dang noi. Relation kieu `says` chi nen ton tai neu
   transcript noi ro chu the trong noi dung, khong duoc suy tu voice.
9. Global registry hien tai dam bao ID khong bi trung va tong hop graph cua
   nhieu video, nhung khong tu merge identity giua hai video. Cross-video
   entity linking la mot bai toan rieng chua duoc trien khai.

## 19. Nguyen tac thiet ke

Kien truc nay dung model hoc san de quan sat, nhung dua phan lon quyet dinh
pipeline ve cac buoc train-free va kiem tra duoc:

- Dynamic programming cho segmentation.
- Hard constraints va weighted similarity cho visual linking.
- Greedy submodular-style gain cho frame selection.
- Lexical/vector threshold cho text aliasing.
- Candidate whitelist cho cross-modal merge.
- Schema validation cho graph output.
- Provenance va timestamp tren moi evidence.

MiniCPM duoc dat o cuoi chuoi, khi context da duoc thu gon va entity candidates
da bi rang buoc, thay vi bat model tu doc video va tu suy ra toan bo graph.
