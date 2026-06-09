# Minimal Ablation Protocol for Collections 0, 6, and 11

## 1. Muc tieu

Ma tran nay dung so scenario toi thieu de tra loi ba cau hoi:

1. Full Framework co tot hon VideoRAG baseline khong?
2. Moi khoi ky thuat lon cua graph construction co dong gop khong?
3. Debate refinement va blinded critique co dong gop khong?

Chi co 8 query scenarios, nhung chi can 6 ingestion artifacts. Hai debate
ablations dung lai toan bo artifact cua Full Framework.

## 2. Collections

| ID | Collection | Dataset type | Kieu evidence noi bat |
|---|---|---|---|
| 0 | `fights-in-animal-kingdom` | documentary | Hanh dong dong vat, visual evidence manh |
| 6 | `daubechies-wavelet-lecture` | lecture | Bai giang ky thuat, transcript va concept evidence manh |
| 11 | `primetime-emmy-awards` | entertainment | San khau, nhieu nguoi, visual va speech deu quan trong |

Ba collection co ba dataset category khac nhau:

```text
0: documentary
6: lecture
11: entertainment
```

Day la tap test phu hop de kiem tra:

```text
visual-heavy nature documentary
speech/slide/concept-heavy technical lecture
mixed-modal awards entertainment
```

Collection 0 co 1 video va 11 questions; collection 6 co 4 video va 25
questions; collection 11 co 3 video va 17 questions. Toan bo tap 0, 6, 11 co
8 video va 53 questions, giu chi phi ingest va ablation o muc hop ly.

Visual identity linking co the co tac dong nho tren lecture neu video chu yeu
la slides/talking-head. Day khong lam scenario A4 mat gia tri: A4 can duoc
phan tich ca overall va theo collection; signal visual identity du kien manh
hon o collections 0 va 11.

## 3. Tam query scenarios

| ID | Scenario code | Artifact | Thanh phan duoc do |
|---|---|---|---|
| A0 | `video_rag_baseline` | VideoRAG legacy | Full system vs original baseline |
| A1 | `full_framework` | Unified Full | Reference method |
| A2 | `no_adaptive_segmentation` | Unified fixed 30s | Adaptive multimodal segmentation |
| A3 | `no_transcript_memory` | Unified no text memory | Transcript discourse memory/coreference |
| A4 | `no_visual_identity_linking` | Unified no cross-tracklet linking | Long-video visual identity |
| A5 | `no_crossmodal_alignment` | Unified no gate/merge | Cross-modal entity alignment |
| A6 | `no_debate_refinement` | Reuse A1 | Critique/Defender refinement rounds |
| A7 | `critique_sees_evidence` | Reuse A1 | Value of blinded critique |

### A0: VideoRAG Baseline

Dung ingestion va query mode VideoRAG goc:

```text
fixed 30-second segments
legacy caption/transcript ingestion
legacy graph extraction
VideoRAG retrieval and answer generation
no EBR debate
```

Day la system-level baseline, khong phai controlled component ablation.

### A1: Full Framework

```text
adaptive segmentation
transcript memory/coreference
YOLO + BoT-SORT + cross-tracklet identity linking
OpenCLIP correspondence gate
MiniCPM entity alignment and graph relations
text + ImageBind feature + unified graph retrieval
blinded critique + tool-enabled Defender
```

### A2: No Adaptive Segmentation

Thay adaptive segmentation bang fixed non-overlapping 30-second windows. Moi
thanh phan phia sau van chay nhu Full.

### A3: No Transcript Memory

Tat:

```text
short-term discourse memory
long-term entity state
pronoun/generic-reference resolution
semantic alias matching across segments
```

Van giu exact normalized-name identity. Vi du, hai lan xuat hien chinh xac
`ColPali` van dung cung text entity ID.

### A4: No Visual Identity Linking

Giu YOLO va BoT-SORT tracklets. Tat OpenCLIP/appearance/motion/temporal linking
giua cac tracklet, nen moi tracklet duoc globalize thanh entity rieng.

### A5: No Cross-modal Alignment

Tat ca khoi visual-text identity alignment bi tat:

```text
OpenCLIP correspondence gate
MiniCPM visual-text merge
exact-name deterministic visual-text merge
cross-modal relation bridge without a valid merge
```

Visual-only va transcript-only entities/edges van duoc giu.

### A6: No Debate Refinement

Dung chinh text, ImageBind feature va unified graph artifact cua A1.

```text
Generator -> Judge
```

Khong co Critique/Defender rounds. Ten nay chinh xac hon `No Debate`, vi
Generator va Judge van ton tai.

### A7: Critique Sees Evidence

Giong A1, nhung Critique duoc xem evidence pool. Scenario nay do gia tri cua
viec giu Critique doc lap khoi retrieval evidence.

## 4. Fixed query configuration

Moi unified scenario dung:

```text
initial_text_k = 4
initial_graph_k = 4
initial_visual_k = 4
max_evidence = 16
max_rounds = 2
max_tool_calls_per_round = 2
max_total_tool_calls = 4
graph_context_token_cap = 1800
```

A6 la ngoai le duy nhat voi `max_rounds=0`.

A7 la ngoai le duy nhat voi `debate_critique_see_evidence=true`.

Khong thay top-k, model, prompt, evidence cap hoac tool budget giua cac
controlled scenarios.

## 5. Metrics

### 5.1 Full Framework vs baseline

Bao cao:

```text
QA answer score / win rate
accuracy neu co answer labels
paired bootstrap 95% confidence interval
ingest time
query latency
GPU-hours
```

### 5.2 Adaptive segmentation

So sanh A1 voi A2:

```text
QA score
Evidence Recall@K neu co temporal evidence annotation
boundary cuts inside ASR words
segment duration distribution
retrieved relevant segment coverage
```

### 5.3 Transcript memory

So sanh A1 voi A3:

```text
pronoun resolution accuracy tren manual audit subset
resolved/unresolved reference count
duplicate text entity rate
QA score tren transcript-heavy questions
```

### 5.4 Visual identity

So sanh A1 voi A4:

```text
visual identity fragmentation
tracklets per global visual entity
duplicate visual entity rate
QA score tren visual/reappearing-entity questions
```

### 5.5 Cross-modal alignment

So sanh A1 voi A5:

```text
cross-modal merge count
manual merge precision
manual merge recall tren annotated candidate subset
false merge rate
cross-modal question QA score
```

### 5.6 Debate

So sanh:

```text
A1 vs A6:
  QA score
  answer correction rate after refinement
  evidence/tool-call count

A1 vs A7:
  QA score
  hallucination/error detection rate
  correction acceptance rate
```

## 6. Artifact layout

Runner tao:

```text
<work-root>/
  0-fights-in-animal-kingdom/
    video_rag_baseline/
    full_framework/
    no_adaptive_segmentation/
    no_transcript_memory/
    no_visual_identity_linking/
    no_crossmodal_alignment/
  6-daubechies-wavelet-lecture/
    ...
  11-primetime-emmy-awards/
    ...
```

Query outputs:

```text
<output-root>/
  <collection>/
    <scenario>/
      answer_<question_id>.md
      result_<question_id>.json
  ablation_status.json
```

Moi unified workdir co:

```text
ablation_ingest_result.json
ablation_artifact_report.json
pipeline_v2/<video_id>/manifest.json
pipeline_v2/<video_id>/progress.json
pipeline_v2/<video_id>/run_report.json
```

## 7. Server preparation

Vi du repo tai `/workspace/EBR-RAG`, data va workdir tai persistent volume
`/workspace/ebr-data`.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git build-essential python3.10-venv

cd /workspace/EBR-RAG

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<70"

python -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-unified-v2.txt
python -m spacy download en_core_web_trf
python -m pip install yt-dlp

python -m pip install --no-deps \
  git+https://github.com/facebookresearch/pytorchvideo.git
python -m pip install --no-deps \
  git+https://github.com/facebookresearch/ImageBind.git

export OPENAI_API_KEY="YOUR_KEY"

python scripts/check_unified_environment.py \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --spacy-model en_core_web_trf \
  --strict
```

Chi chay ingestion khi preflight tra ve `status: ready`. MiniCPM co the duoc
tai truoc theo [SERVER_UNIFIED_V2.md](SERVER_UNIFIED_V2.md); neu khong co thu
muc local, code se tai model Hugging Face o lan chay dau.

Chuan bi danh sach URL:

```bash
cd /workspace/EBR-RAG/longervideos
python prepare_data.py
```

Tai ba collection:

```bash
for collection in \
  0-fights-in-animal-kingdom \
  6-daubechies-wavelet-lecture \
  11-primetime-emmy-awards
do
  mkdir -p "$collection/videos"
  yt-dlp \
    -o "%(id)s.%(ext)s" \
    -S "res:720" \
    -a "$collection/videos.txt" \
    -P "$collection/videos"
done
```

## 8. Run ingestion

```bash
cd /workspace/EBR-RAG
source .venv/bin/activate

python reproduce/run_ablation_matrix.py ingest \
  --collections 0 6 11 \
  --dataset /workspace/EBR-RAG/longervideos/dataset.json \
  --video-root /workspace/EBR-RAG/longervideos \
  --work-root /workspace/ebr-data/ablation-workdirs \
  --output-root /workspace/ebr-data/ablation-results
```

Lenh tren tao 18 collection-artifacts:

```text
3 collections x 6 ingestion profiles
```

Runner mac dinh tai su dung artifact an toan tu Full bang hard-link neu
filesystem ho tro, fallback sang copy:

| Profile | Tai su dung tu Full | Chay lai tu |
|---|---|---|
| no adaptive segmentation | probe, ASR, shot detection | segmentation |
| no transcript memory | probe den frame selection | text entities |
| no visual identity linking | probe, ASR, shots, segmentation, text entities | tracking |
| no cross-modal alignment | moi stage truoc alignment | alignment |

Khong copy `global_registry.json`, graph hoac indexes giua profiles.

Dung `--no-reuse-full-artifacts` neu muon moi profile chay doc lap tu dau.

Chay lai cung lenh de resume. Khong them `--force`.

Neu mot video loi nhung muon tiep tuc:

```bash
... --continue-on-error
```

Neu can chay lai tu mot stage:

```bash
... --restart-stage alignment_caption
```

Chi dung `--force` khi muon rebuild toan bo moi artifact.

## 9. Run queries

```bash
python reproduce/run_ablation_matrix.py query \
  --collections 0 6 11 \
  --dataset /workspace/EBR-RAG/longervideos/dataset.json \
  --video-root /workspace/EBR-RAG/longervideos \
  --work-root /workspace/ebr-data/ablation-workdirs \
  --output-root /workspace/ebr-data/ablation-results
```

Runner nap MiniCPM mot lan va chia se model giua cac scenario trong cung
process. A6 va A7 doc cung Full artifact voi A1.

Chay lai cung lenh de resume. Mot result chi duoc skip khi:

```text
status = complete
answer is non-empty
config hash matches
```

## 10. Run selected scenarios

Smoke test Full:

```bash
python reproduce/run_ablation_matrix.py ingest \
  --collections 6 \
  --ingestion-profiles full_framework \
  --work-root /workspace/ebr-data/ablation-workdirs

python reproduce/run_ablation_matrix.py query \
  --collections 6 \
  --query-scenarios full_framework \
  --work-root /workspace/ebr-data/ablation-workdirs \
  --output-root /workspace/ebr-data/ablation-results
```

Chi chay debate ablations sau khi Full da ingest:

```bash
python reproduce/run_ablation_matrix.py query \
  --collections 0 6 11 \
  --query-scenarios \
    full_framework \
    no_debate_refinement \
    critique_sees_evidence \
  --work-root /workspace/ebr-data/ablation-workdirs \
  --output-root /workspace/ebr-data/ablation-results
```

## 11. Status

```bash
python reproduce/run_ablation_matrix.py status \
  --collections 0 6 11 \
  --work-root /workspace/ebr-data/ablation-workdirs \
  --output-root /workspace/ebr-data/ablation-results

cat /workspace/ebr-data/ablation-results/ablation_status.json
```

Theo doi mot video dang ingest:

```bash
python scripts/show_unified_progress.py \
  --workdir /workspace/ebr-data/ablation-workdirs/6-daubechies-wavelet-lecture/full_framework \
  --watch
```

## 12. Reporting rule

Bang chinh nen gom A0-A7. Moi component claim phai co:

1. Downstream QA metric.
2. Intrinsic metric cua component.
3. Runtime/cost metric.
4. Paired statistical comparison voi A1.

Khong chon threshold hoac thay prompt sau khi da xem ket qua final tren
collections 0, 6, 11.

## 13. Blind pairwise evaluation

Sau khi query hoan tat, so sanh Full voi tung baseline/ablation bang evaluator
blind. Scenario name khong duoc dua vao prompt va vi tri Answer A/B duoc doi
deterministic de giam position bias.

```bash
python reproduce/evaluate_minimal_ablation.py \
  --collections 0 6 11 \
  --dataset /workspace/EBR-RAG/longervideos/dataset.json \
  --answers-root /workspace/ebr-data/ablation-results \
  --output /workspace/ebr-data/ablation-results/pairwise_judgments.jsonl \
  --summary /workspace/ebr-data/ablation-results/pairwise_summary.json \
  --concurrency 8 \
  --repeats 1
```

Evaluator resume theo `comparison_id`; chay lai cung lenh chi xu ly cac pair
con thieu. Neu ngan sach cho phep, dung `--repeats 3` va bao cao paired
bootstrap confidence intervals tu cac judgment.

`pairwise_summary.json` bao cao:

```text
Full win/tie/loss
Full win rate
Full non-loss rate
mean criterion score cua Full
mean criterion score cua baseline/ablation
```
