# Ke hoach ablation cho Unified Multimodal Graph

## 1. Cau hoi nghien cuu

Ablation nay chi danh gia cac cai tien trong **graph construction**:

```text
visual tracking
+ transcript entity extraction/memory
+ visual-text correspondence
+ global entity alignment
+ relation extraction
+ graph memory/provenance
```

Retrieval va debate phai giu co dinh:

```text
text retrieval
+ graph retrieval
+ ImageBind visual retrieval
-> RRF/MMR fusion
-> generator
-> blinded critique
-> tool-enabled defender
-> judge
```

Khong thay doi top-k, evidence cap, prompt, model, temperature, debate rounds
hoac critique visibility giua cac graph variants.

## 2. Nguyen tac cong bang

Moi scenario dung cung:

```text
initial_text_k = 4
initial_graph_k = 4
initial_visual_k = 4
max_evidence = 16
max_rounds = 2
max_tool_calls_per_round = 2
max_total_tool_calls = 4
graph_context_token_cap = 1800
debate_critique_see_evidence = false
debate_defender_disable_tools = false
debate_single_hypothesis = false
```

Dong thoi co dinh:

- Cung 6 collections va cung danh sach questions.
- Cung source videos.
- Cung segmentation cho cac scenario khong ablate segmentation.
- Cung selected frames neu scenario khong ablate frame selection.
- Cung ASR transcript.
- Cung text chunks, text index va visual index cua G1 trong controlled graph
  ablations.
- Cung embedding model va index parameters.
- Cung MiniCPM checkpoint.
- Cung answer/debate models.
- Cung evaluator va evaluation prompts.

Moi graph variant phai co workdir rieng. Khong ghi de graph/index cua scenario
khac.

### Pham vi ket luan

Can tach hai loai so sanh:

```text
System-level baseline:
  old ingestion + old graph artifact
  vs
  unified ingestion + unified graph artifact

Controlled graph ablation:
  same ASR/segments/frames/detections/transcript
  + only one graph-construction component changes
```

`Original graph baseline` co the thay doi segmentation, caption, text chunks
va visual index so voi pipeline moi. Vi vay no la moc so sanh end-to-end, khong
duoc dung mot minh de ket luan cai thien den tu entity alignment.

Ket luan nhan qua cho tung module phai dua tren G1-G7, noi upstream artifacts
duoc giu giong nhau.

### Artifact isolation

`alignment_caption` hien tao caption va rewritten transcript, sau do cac noi
dung nay co the di vao text chunks/index. Neu rebuild toan bo index cho moi
scenario, ta se thay ca text retrieval du retrieval code khong doi.

Vi vay, trong controlled graph ablations:

```text
fixed from G1:
  text_chunks_v2
  chunks_v2
  video_segment_feature_v2
  source segment metadata used by text/visual retrieval

scenario-specific:
  unified graph namespace
  graph entity index
  alignment audit artifacts
```

Graph variants duoc build trong staging workdir rieng, sau do query runner nap
text/visual stores cua G1 va graph stores cua scenario. Neu code hien tai chua
ho tro mixed-store loading, phai them lop experiment loader; khong duoc copy
de graph variant ghi de index G1.

## 3. So do thuc nghiem

```text
                          shared source video
                                  |
                probe -> ASR -> shots -> segmentation
                                  |
                  +---------------+---------------+
                  |                               |
          tracking + frame selection      transcript entities
                  |                               |
                  +---------------+---------------+
                                  |
                    shared pre-alignment artifacts
                                  |
       +----------+----------+----------+----------+----------+
       |          |          |          |          |          |
      G1         G2         G3         G4         G5         G6
     full    no text     no visual   no gate    no merge   no align
             memory       linking                           memory
       |          |          |          |          |          |
       +----------+----------+----------+----------+----------+
                                  |
                    graph build + fixed indexes
                                  |
                fixed retrieval + fixed blinded debate
                                  |
                              evaluation
```

G3 re nhanh som hon tai `tracking_base`. G2 re nhanh tai `text_entities`.
G4-G6 dung chung moi artifact truoc `alignment_caption`.

## 4. Ma tran chinh

Day la ma tran toi thieu de dua vao bang ket qua chinh.

| ID | Scenario | Thanh phan bi bo | Cau hoi duoc tra loi |
|---|---|---|---|
| G0 | Original graph baseline | Toan bo unified graph construction | He thong moi co tot hon he thong cu khong? |
| G1 | Full unified graph | Khong | Ket qua day du cua phuong phap de xuat |
| G2 | No transcript memory | Short/long-term memory va coreference | Memory transcript dong gop bao nhieu? |
| G3 | No visual identity linking | Cross-tracklet identity linking | Global visual identity co can thiet cho video dai? |
| G4 | No correspondence gate | OpenCLIP threshold, mutual-best, margin | Mathematical gate co giam merge sai khong? |
| G5 | No cross-modal merge | Tat ca visual-text entity merges | Alignment hai kenh co tao gia tri hay chi tao noise? |
| G6 | No alignment memory | Recent graph events/entity state trong MiniCPM prompt | Memory lien segment cua alignment co can thiet? |
| G7 | Entity-only graph | Relation edges | Loi ich den tu entity index hay tu graph relations? |
| Q0 | Full minus graph retrieval | Graph retrieval channel tai query-time | Graph channel thuc su dong gop bao nhieu vao answer? |

`Q0` la diagnostic query-time control, khong phai graph-construction ablation
chinh. No duoc bao cao rieng.

## 5. Dinh nghia tung scenario

### G0: Original graph baseline

Dung graph construction truoc unified pipeline:

```text
original/TM graph artifact
+ retrieval/debate hien tai
```

Muc dich:

- So sanh end-to-end phuong phap de xuat voi graph baseline.
- Khong dung de phan ra dong gop cua tung module moi.

Can than:

- Retrieval/debate phai la cung implementation voi G1.
- Khong so sanh G0 chay query pipeline cu voi G1 chay query pipeline moi.
- Neu G0 khong the dung chung segmentation/text/visual indexes, danh dau ro
  `system-level baseline` trong bang va khong mo ta la controlled ablation.

### G1: Full unified graph

```text
visual cross-tracklet identity
+ transcript memory/coreference
+ MiniCPM visual-only facts
+ OpenCLIP correspondence gate
+ constrained MiniCPM merge/relation proposal
+ global registry
+ temporal provenance
```

Day la reference scenario cho moi ablation.

### G2: No transcript memory

Tat:

```text
short-term mention memory
long-term entity state
pronoun/generic-reference resolution
```

Van giu:

- spaCy NER.
- spaCy noun chunks.
- Exact normalized-name matching xuyen video.
- Visual processing va correspondence gate.

Tat semantic alias lookup, short/long-term mention state va pronoun/generic
reference resolution. Pronoun/generic references duoc giu unresolved, khong
bi xoa.

Gia thuyet:

- Giam duplicate text entities qua cac segment.
- Giam recall cho documentary dung nhieu `he`, `she`, `the company`, `the
  object`.
- Co the tang precision nhe do resolve bao thu hon.

### G3: No visual identity linking

Tat cross-tracklet linking:

```text
OpenCLIP appearance
+ HSV appearance
+ motion/temporal cross-tracklet merge
```

Moi tracker tracklet duoc cap mot global visual ID rieng. YOLO + BoT-SORT trong
local chunk van giu nguyen.

Gia thuyet:

- Mot nguoi/vat xuat hien lai sau scene cut se bi tach node.
- Graph fragmentation tang.
- Multi-hop retrieval qua entity xuyen video giam.

Khong nen tat YOLO/BoT-SORT trong scenario nay, vi khi do thay doi ca detection
va identity cung luc.

### G4: No correspondence gate

Bo:

```text
OpenCLIP fact-proposition similarity threshold
mutual-best constraint
ambiguity margin
```

MiniCPM nhan raw type-compatible candidates:

```text
person-person
object-object
object-concept
```

De tranh prompt phinh theo O(V x T), ca G1 va G4 dung cung
`max_alignment_candidates` moi segment. G4 chi bo threshold, margin va
mutual-best; candidate van phai thoa type va temporal co-occurrence constraints.

Van giu visual-only fact pass va MiniCPM merge prompt.

Gia thuyet:

- Merge recall co the tang.
- Merge precision giam, dac biet talking-head documentary.
- Jeff Bezos visible co nguy co bi merge voi nguoi duoc nhac trong speech.

Day la ablation truc tiep nhat de bao ve contribution cua correspondence gate.

### G5: No cross-modal merge

Khong cho phep:

```text
visual entity + transcript entity -> same global node
```

Moi entity van duoc globalize:

```text
visual PERSON -> PERSON_x
text PERSON -> PERSON_y
```

Visual-only, transcript-only nodes va relation edges van duoc giu.
Cross-modal relation proposal van duoc phep noi hai node khac nhau; chi thao
tac identity merge bi cam. Nhu vay G5 chi do gia tri cua entity unification,
khong dong thoi ablate relation extraction.

Gia thuyet:

- Merge precision toi da theo cach bao thu.
- Graph bi tach thanh hai subgraph modality-specific.
- Cau hoi can noi visual participant voi spoken entity se kem hon.
- Talking-head questions co the it bi anh huong.

Scenario nay khac G4:

```text
G4: cho MiniCPM merge ma khong co mathematical gate.
G5: cam merge hoan toan.
```

### G6: No alignment memory

MiniCPM chi nhan evidence cua segment hien tai:

```text
current frames
current visual facts
current transcript propositions
gated candidates
```

Khong dua vao prompt:

```text
recent graph events
long-term relation state
recent entity state
transcript discourse context
```

Registry van ton tai de ID da xac nhan khong bi cap lai. G2 moi la scenario tat
transcript coreference state; G6 chi tat memory trong alignment prompt.

Gia thuyet:

- Quan he va merge o scene lien tuc kem on dinh.
- It bi historical context lam lech o cac topic change.

### G7: Entity-only graph

Giu:

- Unified global nodes.
- Alias.
- Source modalities.
- Timestamp va provenance tren node.
- Entity vector index.

Bo:

- Tat ca semantic relation edges.

Graph retrieval chi con seed/entity evidence, khong co relational path.

Gia thuyet:

- Neu G1 vuot G7, relation edges va graph traversal co gia tri.
- Neu G1 gan G7, improvement chu yeu den tu entity resolution/indexing.

### Q0: Full minus graph retrieval

Dung artifact G1, nhung query pipeline chi retrieve:

```text
text + visual
```

Khong goi `search_graph_evidence` trong initial retrieval va defender.

Moi tham so debate con lai giu nguyen. De cong bang context budget, khong tang
text/visual top-k de bu phan graph bi bo.

Muc dich:

- Do marginal contribution cua graph channel.
- Khong dung de ket luan module construction nao tot.

## 6. Ma tran phu neu con ngan sach

Khong nhat thiet dua tat ca vao bang chinh.

| ID | Scenario | Muc dich |
|---|---|---|
| S1 | Transcript-only graph | Do upper/lower bound cua kenh transcript |
| S2 | Visual-only graph | Do upper/lower bound cua kenh visual |
| S3 | Deterministic auto-merge | Gate pass la merge, khong MiniCPM phan xu |
| S4 | Single-pass early fusion | So sanh late/conditional fusion voi prompt chung mot lan |
| S5 | No temporal provenance in retrieval | Do loi ich cua timestamp/provenance |
| S6 | Fixed 30-second segments | Kiem tra smart segmentation |
| S7 | Uniform 5 frames | Kiem tra adaptive frame selection |
| S8 | Threshold sensitivity | Kiem tra gate threshold/margin |
| S9 | No cross-modal relation edges | Tach dong gop cua cross-modal relations khoi identity merge |

### S1: Transcript-only graph

Chi giu transcript entities va transcript-supported edges. Visual retrieval
channel tai query-time van giu, vi muc tieu la ablate graph artifact chu khong
ablate retrieval architecture.

### S2: Visual-only graph

Chi giu visual entities va visual-supported edges. Text retrieval channel van
giu.

Cap S1/S2 cho biet graph modality nao dong gop nhieu theo tung collection.

### S3: Deterministic auto-merge

Moi candidate qua correspondence gate duoc merge truc tiep, khong hoi MiniCPM.

So sanh:

```text
G1 - S3 = gia tri cua VLM identity adjudication sau mathematical gate
```

### S4: Single-pass early fusion

MiniCPM nhan frames, transcript va raw candidate trong mot call, khong co
visual-only pass/correspondence gate.

Day la baseline truc tiep cho conditional late fusion, nhung no trung lap mot
phan voi G4. Chi chay neu can bao ve kien truc hai pha.

### S5: No temporal provenance in retrieval

Khong xoa timestamp khoi storage. Chi khong dua time/provenance vao graph
evidence packet.

Khong nen xoa provenance vat ly vi validation va debug can no.

### S6: Fixed 30-second segments

Dung fixed non-overlapping 30-second windows thay smart segmentation.

Scenario nay rat dat vi invalidates gan nhu toan pipeline. Bao cao rieng nhu
segmentation study, khong tron vao bang graph-component chinh.

### S7: Uniform 5 frames

Lay 5 frame cach deu cho moi segment, khong quality/diversity/entity-coverage
scoring.

Scenario nay danh gia input evidence cho graph construction, khong phai graph
representation truc tiep.

### S8: Threshold sensitivity

Khong goi cac diem la model training. Chay grid nho:

```text
similarity threshold: 0.24, 0.28, 0.32
margin:               0.02, 0.04, 0.06
```

Khong can chay full Cartesian product luc dau. Chay:

```text
(0.24, 0.04)
(0.28, 0.04) default
(0.32, 0.04)
(0.28, 0.02)
(0.28, 0.06)
```

Dung development subset de chon mot setting truoc khi danh gia final test
collections. Khong chon threshold tren cung tap dung de bao cao ket qua cuoi.

### S9: No cross-modal relation edges

Van cho phep visual-text aliases merge ve mot global entity. Chi loai cac edge
duoc VLM tao boi evidence ket hop hai kenh; visual-supported va
transcript-supported edges van giu.

S9 khac G5:

```text
G5: ablate identity merge, van giu relation extraction.
S9: giu identity merge, ablate cross-modal relation edges.
```

## 7. Kich ban khong nen dung

### Khong ablate retrieve/debate trong bang graph construction

Khong tron cac scenario sau vao cung bang:

```text
no debate
critique sees evidence
defender without tools
different top-k
different evidence cap
different debate rounds
```

Day la debate/retrieval ablations cua project cu, khong giai thich graph moi.

### Khong dung random merge lam baseline chinh

Random merge chi la sanity check, khong phai doi thu co y nghia. No co the pha
graph mot cach nhan tao va khong tra loi cau hoi khoa hoc cu the.

### Khong tat nhieu module trong mot scenario

Vi du `no memory + no gate + no visual linking` khong cho biet module nao gay
thay doi.

## 8. Metrics

Can bao cao ca downstream QA va intrinsic graph quality.

### 8.1 Downstream answer metrics

Neu benchmark la multiple choice:

```text
Accuracy
Macro accuracy theo collection/category
Paired correctness delta voi G1
```

Neu co open-ended answers:

```text
LLM-as-judge win/tie/loss
Comprehensiveness
Trustworthiness
Depth
Density
```

Evaluator phai:

- Blind scenario names.
- Randomize answer order.
- Dung cung model/prompt.
- Chay cung reference answer.

### 8.2 Retrieval metrics

Neu benchmark co temporal evidence labels:

```text
Evidence Recall@K
Temporal IoU
MRR
nDCG
```

Neu khong co evidence labels:

```text
answer-cited evidence coverage
validated citation rate
retrieval channel contribution
unique relevant segments in final evidence pool
```

### 8.3 Intrinsic graph metrics

```text
number of nodes
number of edges
connected components
largest component ratio
average degree
duplicate entity rate
cross-modal merge count
merge acceptance rate
unresolved reference rate
visual identity fragmentation
edge modality distribution
edge provenance coverage
invalid/dangling edge count
```

### 8.4 Manual audit subset

Lay stratified sample, vi du 20-30 merge decisions moi collection:

```text
merge precision
merge recall neu annotate du candidate
relation precision
pronoun resolution accuracy
identity fragmentation
```

Sample phai gom:

- Talking-head.
- Lecture/slides.
- Documentary/B-roll.
- Multi-person scene.
- Reappearing person/object.

### 8.5 Efficiency

```text
ingest wall-clock time
GPU-hours
peak VRAM
graph size on disk
index size
query latency
LLM/VLM calls per video
```

## 9. Phan tich thong ke

Moi question phai duoc paired across scenarios.

Khuyen nghi:

```text
MCQ accuracy:
  McNemar test G1 vs moi ablation
  bootstrap 95% CI cho accuracy delta

Open-ended judge score:
  paired bootstrap/permutation test
  report mean delta va 95% CI

Manual merge precision:
  Wilson confidence interval
```

Do debate model co temperature khac 0, chay it nhat 3 answer-generation repeats
cho bang chinh neu ngan sach cho phep. Graph ingestion co the chay mot lan moi
scenario neu MiniCPM decoding deterministic; neu khong, ghi seed/decoding va
kiem tra variance tren subset.

## 10. Thu tu uu tien theo ngan sach

### Tier 1: bat buoc

```text
G0 Original baseline
G1 Full
G2 No transcript memory
G4 No correspondence gate
G5 No cross-modal merge
G7 Entity-only graph
Q0 No graph retrieval
```

Day la 7 graph artifacts/query controls co signal khoa hoc ro nhat.

### Tier 2: nen co

```text
G3 No visual identity linking
G6 No alignment memory
S1 Transcript-only graph
S2 Visual-only graph
```

### Tier 3: sensitivity/architecture

```text
S3 Deterministic auto-merge
S4 Single-pass early fusion
S6 Fixed segmentation
S7 Uniform frames
S8 Threshold sensitivity
```

## 11. Cau hinh ablation de xuat

Cac flag duoi day la experiment contract. Chung chua duoc coi la da implement
chi vi co trong tai lieu:

```text
graph_ablation_profile = full
disable_text_memory = false
disable_visual_identity_linking = false
disable_correspondence_gate = false
disable_cross_modal_merge = false
disable_alignment_memory = false
entity_only_graph = false
disable_cross_modal_relation_edges = false
max_alignment_candidates = <same fixed value for all scenarios>
```

Moi profile chi duoc doi mot flag:

| Profile | Flag thay doi |
|---|---|
| `full` | Khong |
| `no_text_memory` | `disable_text_memory=true` |
| `no_visual_identity` | `disable_visual_identity_linking=true` |
| `no_correspondence_gate` | `disable_correspondence_gate=true` |
| `no_cross_modal_merge` | `disable_cross_modal_merge=true` |
| `no_alignment_memory` | `disable_alignment_memory=true` |
| `entity_only_graph` | `entity_only_graph=true` |
| `no_cross_modal_edges` | `disable_cross_modal_relation_edges=true` |

Moi flag phai duoc them vao `config_keys` cua stage som nhat ma no anh huong.
Neu khong, resume fingerprint co the tai nham artifact cua scenario khac.

```text
text_entities:
  disable_text_memory

tracking_base:
  disable_visual_identity_linking

alignment_caption:
  disable_correspondence_gate
  disable_cross_modal_merge
  disable_alignment_memory
  max_alignment_candidates

graph_build:
  entity_only_graph
  disable_cross_modal_relation_edges
```

Manifest moi run phai luu:

```text
git commit
dirty-worktree marker
scenario/profile
full resolved config
config hash
model checkpoint names
input video hash
stage artifact hashes
random seeds/decoding parameters
```

## 12. Chi phi va tai su dung artifact

Khong phai scenario nao cung can chay lai toan pipeline.

| Scenario | Stage bat dau chay lai | Co the tai su dung |
|---|---|---|
| G2 no transcript memory | `text_entities` | probe, ASR, shots, segments, tracking, frames |
| G3 no visual identity | `tracking_base` | probe, ASR, shots, segments |
| G4 no correspondence gate | `alignment_caption` | moi artifact truoc alignment |
| G5 no cross-modal merge | `alignment_caption` | moi artifact truoc alignment |
| G6 no alignment memory | `alignment_caption` | moi artifact truoc alignment |
| G7 entity-only graph | post-process/full alignment artifact | nodes, aliases, provenance |
| Q0 no graph retrieval | query only | toan bo G1 workdir |
| S1/S2 modality-only graph | post-process G1 neu provenance/modalities du | G1 registry/alignment |
| S9 no cross-modal edges | `graph_build`/post-process | G1 registry/alignment |
| S6 fixed segments | `segmentation` | probe, ASR, shots |
| S7 uniform frames | `frame_selection` | probe, ASR, shots, segments, tracking |

De tranh copy video/model cache, co the:

- Dung chung source videos va HF cache.
- Dung workdir rieng cho moi scenario.
- Copy/hard-link immutable upstream stage artifacts neu script ablation co
  fingerprint va khong ghi nguoc.

Khong dung symbolic link cho artifact ma stage co the xoa khi invalidation.

## 13. Giao thuc chay

### Buoc 1: smoke test

Chay mot video ngan hoac mot phan cua collection 19 cho moi profile de kiem tra:

```text
resume/fingerprint dung
khong ghi de artifact
graph validation pass
Q0 that su khong dung graph
retrieval/debate config giong nhau
```

Smoke test chi kiem tra ky thuat, khong dung de chon threshold.

### Buoc 2: pilot co kiem soat

Chay G1, G2, G4 va G5 tren collections 6, 11, 19. Muc dich la:

```text
uoc luong runtime/VRAM
kiem tra metric co thu duoc
tao manual audit form
phat hien scenario bi vo hieu do flag khong tac dong
```

Khong thay threshold dua tren final QA accuracy. Neu can tune threshold, dung
development videos/questions tach khoi sau collections danh gia.

### Buoc 3: bang chinh tren 6 collections

Chay cung question list cho:

```text
G0, G1, G2, G3, G4, G5, G7, Q0
```

G6 co the dua vao bang chinh neu pilot cho thay alignment memory la mot module
doc lap, khong trung tac dong voi G2.

### Buoc 4: diagnostic

Chi sau khi bang chinh hoan tat moi chay S1-S9 theo ngan sach. Khong thay doi
tap scenario chinh dua tren ket qua test vua xem.

### So lan lap

```text
Ingestion:
  1 lan/profile neu deterministic decoding.
  3 lan tren mot subset neu can do ingestion variance.

Answer/debate:
  3 repeats/question/scenario neu ngan sach cho phep.

Manual audit:
  sample stratified va blind scenario label.
```

Moi answer phai luu ca final answer, evidence IDs, retrieval scores, tool calls,
debate trace, latency va token usage. Resume chi skip khi record da day du va
config hash trung khop.

## 14. Bang ket qua de xuat

### Bang chinh

| Scenario | Accuracy/Win rate | Evidence Recall | Merge precision | Nodes | Edges | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|
| G0 Original | | | | | | |
| G1 Full | | | | | | |
| G2 w/o transcript memory | | | | | | |
| G3 w/o visual identity | | | | | | |
| G4 w/o correspondence gate | | | | | | |
| G5 w/o cross-modal merge | | | | | | |
| G6 w/o alignment memory | | | | | | |
| G7 entity-only graph | | | | | | |

### Bang diagnostic retrieval

| Scenario | Accuracy/Win rate | Text evidence | Graph evidence | Visual evidence |
|---|---:|---:|---:|---:|
| G1 Full retrieval | | | | |
| Q0 w/o graph channel | | | 0 | |

### Bang theo category

| Scenario | Lecture | Talking-head | Documentary | Event/video montage | Overall |
|---|---:|---:|---:|---:|---:|
| G1 | | | | | |
| G2 | | | | | |
| G4 | | | | | |
| G5 | | | | | |

## 15. Ket luan mong doi theo gia thuyet

```text
G1 > G0:
  unified construction tot hon graph baseline.

G1 > G2:
  transcript memory giup documentary/coreference.

G1 > G3:
  cross-segment visual identity giup long-video graph.

G1 > G4 ve trustworthiness/merge precision:
  correspondence gate ngan false merge.

G1 > G5:
  cross-modal identity alignment tao gia tri that.

G1 > G6:
  alignment memory giup continuity.

G1 > G7:
  relation edges va graph traversal co dong gop ngoai entity indexing.

G1 > Q0:
  graph retrieval channel co gia tri downstream.
```

Neu ket qua khong theo gia thuyet, khong nen gop scenario hoac dieu chinh
threshold sau khi xem test results. Can phan tich theo category va intrinsic
metrics de tim nguyen nhan.
