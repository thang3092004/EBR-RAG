# Huong dan thue server va chay Unified Graph Ingestion

Tai lieu nay la runbook chinh de:

1. Chon GPU server.
2. Dua code, model va video len server.
3. Cai moi truong dung phien ban.
4. Chay preflight va smoke test.
5. Chay ingestion cho tung collection.
6. Theo doi, resume, xu ly loi va tai ket qua ve.

Tap ablation hien tai dung collections `0`, `6`, `11`, tuong ung documentary,
lecture va entertainment. Lenh tai data, ingest, query va evaluate chinh xac
nam trong [MINIMAL_ABLATION_0_6_11.md](MINIMAL_ABLATION_0_6_11.md). Cac vi du
`collection-01` ben duoi la mau tong quat cho ingestion mot collection.

Pipeline hien tai:

```text
probe
-> ASR
-> shot detection
-> smart segmentation
-> YOLO + BoT-SORT tracking
-> OpenCLIP visual identity
-> frame selection
-> spaCy transcript entities + discourse memory
-> MiniCPM visual-only facts
-> OpenCLIP correspondence gate
-> MiniCPM constrained alignment/relation proposal
-> deterministic graph validation/build
-> text/entity/ImageBind indexes
-> final validation
```

## 1. Cau hinh server nen thue

### Cau hinh khuyen nghi

```text
OS: Ubuntu 22.04
GPU: 1 GPU, 40-48 GB VRAM
CPU: it nhat 8 vCPU, khuyen nghi 16 vCPU
RAM: it nhat 64 GB
Persistent SSD: it nhat 300 GB
CUDA driver: tuong thich CUDA 12.1
Python: 3.10 hoac 3.11
```

Lua chon GPU hop ly:

```text
A40 48 GB       tot ve chi phi
RTX A6000 48 GB tot ve chi phi
A100 40 GB      nhanh va on dinh
A100 80 GB      rat thoai mai, nhung khong bat buoc
L40/L40S 48 GB  tot neu gia hop ly
```

GPU 24 GB nhu RTX 4090/A5000 co the chay, nhung de gap OOM hon o MiniCPM hoac
ImageBind. Khong nen thue GPU 16 GB cho full collection. H100 la du thua neu
gia cao hon nhieu.

Pipeline chay cac model lon tuan tu va giai phong VRAM giua cac stage. Khong
can nhieu GPU. Mot GPU VRAM lon huu ich hon nhieu GPU nho.

### Uoc luong dung luong dia

Dung luong persistent volume nen tinh:

```text
source videos
+ MiniCPM model
+ Whisper/OpenCLIP/ImageBind caches
+ pipeline checkpoints va graph indexes
+ 30-50% du phong
```

Neu chua biet tong dung luong, bat dau voi 300-500 GB persistent SSD. Khong dat
code, model, video hoac workdir tren temporary/container disk neu nha cung cap
co the xoa no khi stop/terminate instance.

Voi RunPod, nen dat moi thu can giu trong `/workspace` hoac network volume.
Tai lieu chinh thuc cua RunPod phan biet container volume, disk volume va
network volume:

- https://docs.runpod.io/pods
- https://docs.runpod.io/pods/manage-pods

### Lua chon nha cung cap

Co the dung RunPod, Vast.ai, Lambda Cloud hoac nha cung cap GPU khac. Tieu chi:

- Co SSH.
- Co persistent volume.
- Co the chon Ubuntu/PyTorch image.
- GPU co it nhat 24 GB VRAM, tot nhat 40-48 GB.
- SSD du lon va co the tang dung luong.
- Khong bi gioi han thoi gian session ngan.

RunPod co huong dan SSH chinh thuc:

- https://docs.runpod.io/pods/configuration/use-ssh
- https://docs.runpod.io/pods/connect-to-a-pod

Lambda Cloud cong bo cau hinh va gia hien tai tai:

- https://lambda.ai/service/gpu-cloud/pricing

Khong nen chon chi dua tren gia GPU/gio. Persistent storage, toc do tai model,
egress va kha nang resume cung anh huong tong chi phi.

## 2. Truoc khi thue server

Dam bao code moi nhat da duoc commit va push len branch can chay:

```bash
git status
git branch --show-current
git log -1 --oneline
```

Ghi lai:

```text
repository URL
branch name
commit hash
3 collections se chay: 0, 6, 11
duong dan video trong moi collection
OpenAI-compatible embedding endpoint va API key
```

Model va data lon dang nam trong `.gitignore`, nen `git clone` se khong mang
theo:

```text
MiniCPM-V-2_6-int4/
faster-distil-whisper-large-v3/
longervideos/* video files
workdirs/
```

Can tai model tren server hoac upload rieng.

## 3. Tao server

Neu dung RunPod:

1. Chon official PyTorch template.
2. Chon GPU 40-48 GB neu co.
3. Tao persistent volume va mount vao `/workspace`.
4. Dat container disk toi thieu 30-50 GB cho OS/package cache.
5. Them SSH public key.
6. Bat public IP/full SSH neu muon dung `scp` hoac `rsync`.

Tu Windows PowerShell, tao SSH key neu chua co:

```powershell
ssh-keygen -t ed25519 -C "your-email@example.com"
Get-Content $HOME\.ssh\id_ed25519.pub
```

Them public key vao tai khoan nha cung cap. Sau do connect bang lenh ho cung
cap, vi du:

```powershell
ssh root@SERVER_IP -p SSH_PORT -i $HOME\.ssh\id_ed25519
```

Ngay sau khi connect:

```bash
nvidia-smi
df -h
free -h
uname -a
```

Khong tiep tuc neu:

- `nvidia-smi` khong thay GPU.
- Persistent volume khong duoc mount.
- Dung luong con lai qua it.
- GPU co VRAM thap hon cau hinh da thue.

## 4. Chuan bi thu muc persistent

Vi du voi `/workspace`:

```bash
mkdir -p /workspace/ebr
mkdir -p /workspace/data/collections
mkdir -p /workspace/workdirs
mkdir -p /workspace/hf
mkdir -p /workspace/logs
cd /workspace/ebr
```

Dat Hugging Face va Torch caches tren persistent volume:

```bash
export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub
export TORCH_HOME=/workspace/hf/torch
export XDG_CACHE_HOME=/workspace/hf/xdg
```

Them vao shell profile de session moi van dung:

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/hf
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub
export TORCH_HOME=/workspace/hf/torch
export XDG_CACHE_HOME=/workspace/hf/xdg
EOF
```

Neu mount point khong phai `/workspace`, thay tat ca duong dan trong tai lieu
bang persistent mount cua server.

## 5. Clone dung branch va commit

```bash
cd /workspace/ebr
git clone https://github.com/thang3092004/EBR-RAG.git
cd EBR-RAG
git fetch --all --prune
git switch entity-anchored-tm-graph
git pull --ff-only
git log -1 --oneline
```

So sanh commit hash voi commit da ghi lai tren may local.

Neu repository private:

```bash
git clone git@github.com:thang3092004/EBR-RAG.git
```

Khong chay tren mot checkout co thay doi local khong ro nguon:

```bash
git status --short
```

Ket qua nen rong.

## 6. Cai system packages

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  git \
  git-lfs \
  tmux \
  rsync \
  curl \
  wget \
  unzip \
  build-essential \
  python3.10 \
  python3.10-venv \
  python3.10-dev
```

Kiem tra:

```bash
python3.10 --version
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
```

Neu image chi co Python 3.11, co the dung `python3.11`; environment checker
chap nhan ca 3.10 va 3.11.

## 7. Tao Python environment

```bash
cd /workspace/ebr/EBR-RAG
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<70"
```

Cai PyTorch CUDA 12.1 dung phien ban da test:

```bash
pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Cai requirements:

```bash
pip install -r requirements-unified-v2.txt
```

Cai spaCy English transformer pipeline:

```bash
python -m spacy download en_core_web_trf
```

Cai PyTorchVideo va ImageBind:

```bash
pip install --no-deps \
  git+https://github.com/facebookresearch/pytorchvideo.git

pip install --no-deps \
  git+https://github.com/facebookresearch/ImageBind.git
```

Kiem tra dependency conflict:

```bash
pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

Phai thay:

```text
torch 2.4.1+cu121
CUDA available = True
transformers 4.40.0
```

Neu `pip install` sau do lam thay doi Torch hoac Transformers, cai lai dung
phien ban pin o tren.

## 8. Tai model

### MiniCPM

Khuyen nghi tai truoc de smoke test khong bi dung giua chung:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openbmb/MiniCPM-V-2_6-int4",
    local_dir="/workspace/ebr/EBR-RAG/MiniCPM-V-2_6-int4",
)
PY
```

Kiem tra:

```bash
du -sh /workspace/ebr/EBR-RAG/MiniCPM-V-2_6-int4
```

Neu khong tai local, code se fallback sang
`openbmb/MiniCPM-V-2_6-int4` va tai luc chay.

### Cac model con lai

Mac dinh cac model sau tu tai o lan chay dau:

```text
Systran/faster-distil-whisper-large-v3
yolov8n.pt
OpenCLIP ViT-B-32 laion2b_s34b_b79k
ImageBind-Huge pretrained weights
```

Vi vay lan smoke test dau co the mat nhieu thoi gian cho download. Kiem tra
server co Internet va cache dang nam tren persistent volume.

## 9. Dua video len server

Cau truc du lieu khuyen nghi:

```text
/workspace/data/collections/
  collection-01/
    videos/
      video-a.webm
      video-b.mp4
  collection-02/
    videos/
  ...
```

Moi collection dung mot workdir rieng:

```text
/workspace/workdirs/
  collection-01-v2/
  collection-02-v2/
```

Khong dung chung mot workdir cho hai collection neu muon danh gia doc lap.

### Upload bang rsync

Tu may Linux/WSL:

```bash
rsync -ah --info=progress2 \
  -e "ssh -p SSH_PORT -i ~/.ssh/id_ed25519" \
  /local/path/collection-01/ \
  root@SERVER_IP:/workspace/data/collections/collection-01/
```

### Upload bang scp tu Windows PowerShell

```powershell
scp -P SSH_PORT -i $HOME\.ssh\id_ed25519 -r `
  "C:\path\to\collection-01" `
  root@SERVER_IP:/workspace/data/collections/
```

Voi data lon, `rsync` tot hon `scp` vi co the tiep tuc file dang copy.

Sau khi upload:

```bash
find /workspace/data/collections -type f \
  \( -iname "*.mp4" -o -iname "*.webm" -o -iname "*.mkv" \
     -o -iname "*.mov" -o -iname "*.avi" \) \
  -printf "%p %s bytes\n"

du -sh /workspace/data/collections/*
```

Hai video trong cung mot lenh ingestion khong duoc co cung filename stem. Vi
du `a/video.mp4` va `b/video.webm` deu tao `video_id=video`, nen CLI se tu choi
de tranh ghi de.

## 10. Cau hinh embedding API

Ingestion khong dung GPT de tao graph. Tuy nhien stage `embedding_index` can
OpenAI-compatible embeddings cho text chunks va entity index.

Tao file secret ngoai Git ma khong ghi key vao shell history:

```bash
read -rsp "OPENAI_API_KEY: " EBR_API_KEY
echo
printf 'export OPENAI_API_KEY=%q\n' "$EBR_API_KEY" \
  > /workspace/ebr/ebr.env
printf 'export OPENAI_BASE_URL=%q\n' 'https://api.openai.com/v1' \
  >> /workspace/ebr/ebr.env
unset EBR_API_KEY

chmod 600 /workspace/ebr/ebr.env
source /workspace/ebr/ebr.env
```

Neu dung OpenAI-compatible endpoint khac, thay `OPENAI_BASE_URL`. Endpoint phai
ho tro model embedding dang duoc cau hinh, mac dinh:

```text
text-embedding-3-small
```

Khong commit API key vao repository, `.env`, log hoac tmux command history.

## 11. Preflight bat buoc

Activate environment va load secrets:

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate
source /workspace/ebr/ebr.env
```

Chay:

```bash
set -o pipefail

python scripts/check_unified_environment.py \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --spacy-model en_core_web_trf \
  --strict | tee /workspace/logs/preflight.json
```

Chi tiep tuc neu command exit code la `0` va report co:

```json
{
  "status": "ready",
  "required_failures": []
}
```

Kiem tra them:

```bash
nvidia-smi
df -h /workspace
pip check
```

Neu preflight bao:

```text
transformers
```

cai lai:

```bash
pip install --force-reinstall "transformers==4.40.0"
```

Neu bao `spacy_model`:

```bash
python -m spacy download en_core_web_trf
```

Neu bao `embedding_api`, load lai:

```bash
source /workspace/ebr/ebr.env
```

## 12. Smoke test mot video

Khong chay ca collection ngay. Chon mot video dai vua phai, co ca hinh va
transcript. Video dau tien cua collection 11 la smoke test phu hop:

```text
dYX809pLH00.webm
```

Tao tmux:

```bash
tmux new -s ebr-smoke
```

Trong tmux:

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate
source /workspace/ebr/ebr.env
set -o pipefail

python scripts/ingest_unified_v2.py \
  --video /workspace/ebr/EBR-RAG/longervideos/11-primetime-emmy-awards/videos/dYX809pLH00.webm \
  --workdir /workspace/workdirs/smoke-primetime-emmy \
  --caption-model ./MiniCPM-V-2_6-int4 \
  2>&1 | tee /workspace/logs/smoke-primetime-emmy.log
```

Detach tmux:

```text
Ctrl-b, sau do nhan d
```

Attach lai:

```bash
tmux attach -t ebr-smoke
```

Theo doi GPU tu terminal khac:

```bash
watch -n 2 nvidia-smi
```

Theo doi pipeline:

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate

python scripts/show_unified_progress.py \
  --workdir /workspace/workdirs/smoke-jeff-bezos \
  --watch
```

## 13. Xac nhan smoke test thanh cong

Lay video ID tu filename:

```text
DcWqzZ3I2cY.webm -> DcWqzZ3I2cY
```

Kiem tra:

```bash
cat /workspace/workdirs/smoke-jeff-bezos/pipeline_v2/\
DcWqzZ3I2cY/validation/validation_report.json

cat /workspace/workdirs/smoke-jeff-bezos/pipeline_v2/\
DcWqzZ3I2cY/run_report.json
```

Smoke test chi dat khi:

```text
run_report.status = complete
validation_report.valid = true
missing_aligned_segments = []
provisional_nodes = []
dangling_edges = []
edges_missing_provenance = []
```

Kiem tra artifact:

```bash
ls -lh /workspace/workdirs/smoke-jeff-bezos/
ls -lh /workspace/workdirs/smoke-jeff-bezos/pipeline_v2/DcWqzZ3I2cY/
```

Kiem tra graph:

```bash
ls -lh /workspace/workdirs/smoke-jeff-bezos/\
graph_chunk_entity_relation_v2.graphml
```

Kiem tra mot so segment alignment:

```bash
find /workspace/workdirs/smoke-jeff-bezos/pipeline_v2/\
DcWqzZ3I2cY/alignment_caption/segments \
  -name "*.json" | head
```

Moi segment JSON co:

```text
visual_analysis
transcript_propositions
raw_candidate_merges
candidate_merges
correspondence
accepted_merges
edges
```

Talking-head segment co speech khong lien quan hinh anh thuong nen co:

```json
{
  "correspondence": {
    "status": "independent"
  }
}
```

va khong merge ten nguoi trong transcript vao nguoi visible neu khong co bridge
evidence.

## 14. Chay mot collection

Sau khi smoke test dat:

```bash
tmux new -s ebr-c01
```

Trong tmux:

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate
source /workspace/ebr/ebr.env
set -o pipefail

python scripts/ingest_unified_v2.py \
  --video-dir /workspace/data/collections/collection-01/videos \
  --workdir /workspace/workdirs/collection-01-v2 \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --continue-on-error \
  2>&1 | tee /workspace/logs/collection-01-v2.log
```

`--continue-on-error`:

- Ghi video loi vao final report.
- Tiep tuc cac video con lai.
- Ket thuc command voi exit code khac 0 neu co bat ky video nao loi.
- Khong bien partial failure thanh thanh cong gia.

Khong chay hai process ingestion cung ghi vao cung mot workdir. Global registry,
GraphML va vector DB la shared artifacts cua collection, chua co inter-process
lock.

## 15. Chay nhieu collection

An toan nhat la chay tuan tu. Day la mau tong quat; voi ablation `0,6,11`,
dung runner va lenh trong `MINIMAL_ABLATION_0_6_11.md`.

```bash
set -o pipefail

for name in collection-01 collection-02 collection-03 \
            collection-04 collection-05 collection-06
do
  echo "===== START $name $(date -Is) ====="

  python scripts/ingest_unified_v2.py \
    --video-dir "/workspace/data/collections/$name/videos" \
    --workdir "/workspace/workdirs/${name}-v2" \
    --caption-model ./MiniCPM-V-2_6-int4 \
    --continue-on-error \
    2>&1 | tee "/workspace/logs/${name}-v2.log"

  code=${PIPESTATUS[0]}
  echo "===== END $name exit=$code $(date -Is) ====="
done
```

Lenh tren tiep tuc collection tiep theo ke ca collection truoc co video loi.
Sau khi chay xong, phai doc log va validation report, khong chi nhin loop ket
thuc.

Khong nen chay nhieu collection song song tren mot GPU. Cac process se tranh VRAM,
RAM, disk I/O va ghi nham shared cache.

## 16. Resume sau khi SSH rot, server reboot hoac process dung

Neu chi SSH rot nhung tmux con:

```bash
tmux ls
tmux attach -t ebr-c01
```

Neu process da dung, chay lai dung cung command va cung workdir:

```bash
python scripts/ingest_unified_v2.py \
  --video-dir /workspace/data/collections/collection-01/videos \
  --workdir /workspace/workdirs/collection-01-v2 \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --continue-on-error
```

Resume mac dinh se skip stage da `done`. Stage dang `running`, `failed` hoac
`interrupted` se chay lai.

Khong them `--force` khi chi muon resume.

## 17. Restart mot stage

Restart alignment va moi stage phu thuoc:

```bash
python scripts/ingest_unified_v2.py \
  --video /workspace/data/collections/collection-01/videos/VIDEO.webm \
  --workdir /workspace/workdirs/collection-01-v2 \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --restart-stage alignment_caption
```

Stage names hop le:

```text
probe
asr
shot_detection
segmentation
tracking_base
frame_selection
text_entities
alignment_caption
graph_build
embedding_index
validation
```

Vi du:

```text
Sua transcript memory:
  --restart-stage text_entities

Sua frame selection:
  --restart-stage frame_selection

Chi index bi loi:
  --restart-stage embedding_index
```

`--force` xoa checkpoint pipeline cua video va chay lai toan bo:

```bash
python scripts/ingest_unified_v2.py \
  --video /path/to/VIDEO.webm \
  --workdir /workspace/workdirs/collection-01-v2 \
  --force
```

Chi dung khi thuc su muon re-ingest tu dau.

## 18. Monitoring va doc trang thai

Theo doi tat ca video trong mot workdir:

```bash
python scripts/show_unified_progress.py \
  --workdir /workspace/workdirs/collection-01-v2 \
  --watch \
  --interval 3
```

Theo doi log:

```bash
tail -f /workspace/logs/collection-01-v2.log
```

Theo doi mot video:

```bash
tail -f /workspace/workdirs/collection-01-v2/pipeline_v2/\
VIDEO_ID/pipeline.log
```

Theo doi disk:

```bash
watch -n 30 'df -h /workspace; du -sh /workspace/workdirs/* 2>/dev/null'
```

Theo doi GPU:

```bash
watch -n 2 nvidia-smi
```

Khong kill process chi vi GPU utilization ve 0 trong thoi gian ngan. ASR, spaCy,
FFmpeg, graph storage va API embedding co cac khoang CPU/network-heavy.

## 19. Cau truc output

Moi collection workdir gom:

```text
<workdir>/
  graph_chunk_entity_relation_v2.graphml
  kv_store_video_path_v2.json
  kv_store_video_segments_v2.json
  kv_store_text_chunks_v2.json
  vdb_entities_v2.json
  vdb_chunks_v2.json
  vdb_video_segment_feature_v2.json
  pipeline_v2/
    global_registry.json
    <video_id>/
      manifest.json
      progress.json
      events.jsonl
      pipeline.log
      run_report.json
      probe/
      asr/
      shot_detection/
      segmentation/
      tracking_base/
      frame_selection/
      text_entities/
      alignment_caption/
        alignment.json
        segments/
      graph_build/
      embedding_index/
      validation/
        validation_report.json
```

File quan trong nhat:

```text
graph_chunk_entity_relation_v2.graphml
pipeline_v2/global_registry.json
pipeline_v2/<video_id>/run_report.json
pipeline_v2/<video_id>/validation/validation_report.json
pipeline_v2/<video_id>/alignment_caption/alignment.json
```

## 20. Kiem tra ca collection

Liet ke video khong complete:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/workdirs/collection-01-v2/pipeline_v2")
for report_path in sorted(root.glob("*/run_report.json")):
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete":
        print(report_path.parent.name, report.get("status"))
PY
```

Liet ke validation fail:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/workdirs/collection-01-v2/pipeline_v2")
failed = []
for path in sorted(root.glob("*/validation/validation_report.json")):
    report = json.loads(path.read_text())
    if not report.get("valid"):
        failed.append((path.parts[-3], report))

print("failed:", len(failed))
for video_id, report in failed:
    print(video_id, report)
PY
```

Kiem tra graph ton tai va khong rong:

```bash
python - <<'PY'
import networkx as nx

path = "/workspace/workdirs/collection-01-v2/graph_chunk_entity_relation_v2.graphml"
graph = nx.read_graphml(path)
print("nodes", graph.number_of_nodes())
print("edges", graph.number_of_edges())
assert graph.number_of_nodes() > 0
PY
```

## 21. Xu ly loi thuong gap

### CUDA out of memory

1. Xac nhan khong co process ingestion khac:

```bash
nvidia-smi
ps aux | grep ingest_unified_v2
```

2. Resume lai command sau khi process cu da dung.
3. Neu van OOM, thue GPU VRAM lon hon.
4. Khong chay nhieu collection song song.

### Transformers sai phien ban

```bash
pip install --force-reinstall "transformers==4.40.0"
python -c "import transformers; print(transformers.__version__)"
```

### FFmpeg/FFprobe missing

```bash
apt-get update
apt-get install -y ffmpeg
```

### spaCy model missing

```bash
python -m spacy download en_core_web_trf
```

### OpenCLIP/ImageBind download loi

Kiem tra Internet, cache path va disk:

```bash
df -h /workspace
echo "$HF_HOME"
```

Sau do resume. Stage failed se chay lai.

### Embedding API loi

```bash
source /workspace/ebr/ebr.env
python scripts/check_unified_environment.py \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --spacy-model en_core_web_trf \
  --strict
```

Neu API rate limit, doi mot luc va resume `embedding_index`.

### Disk full

Kiem tra:

```bash
df -h
du -h --max-depth=2 /workspace | sort -h | tail -n 30
```

Khong xoa workdir dang chay. Co the xoa package cache hoac model cache khong
con dung, hoac tang persistent volume.

### Mot video corrupt

Voi `--continue-on-error`, batch se chay tiep. Xem:

```bash
cat <workdir>/pipeline_v2/<video_id>/run_report.json
tail -n 100 <workdir>/pipeline_v2/<video_id>/pipeline.log
```

Thu `ffprobe`:

```bash
ffprobe -v error -show_format -show_streams /path/to/video
```

## 22. Backup va tai ket qua ve

Khong terminate server truoc khi backup:

```bash
cd /workspace/workdirs
tar -czf /workspace/collection-01-v2-results.tar.gz collection-01-v2
sha256sum /workspace/collection-01-v2-results.tar.gz
```

Tai ve bang `scp`:

```powershell
scp -P SSH_PORT -i $HOME\.ssh\id_ed25519 `
  root@SERVER_IP:/workspace/collection-01-v2-results.tar.gz `
  .
```

Hoac `rsync`:

```bash
rsync -ah --info=progress2 \
  -e "ssh -p SSH_PORT -i ~/.ssh/id_ed25519" \
  root@SERVER_IP:/workspace/workdirs/collection-01-v2/ \
  ./collection-01-v2/
```

Xac minh checksum sau khi tai:

```bash
sha256sum collection-01-v2-results.tar.gz
```

Backup toi thieu:

```text
GraphML
KV stores
vector DB JSON files
pipeline_v2/global_registry.json
run reports
validation reports
alignment outputs
logs
```

## 23. Dung hoac terminate server

Truoc khi stop:

```bash
tmux ls
ps aux | grep ingest_unified_v2
df -h /workspace
```

Chi stop khi ingestion da dung hoac da duoc interrupt co chu dich. `Ctrl-C`
se de stage o trang thai interrupted va co the resume.

Truoc khi terminate:

1. Xac nhan ket qua da backup.
2. Xac nhan checksum.
3. Xac nhan model/data/workdir nao nam tren persistent volume.
4. Xac nhan chinh sach nha cung cap: stop va terminate co the co hanh vi storage
   khac nhau.

RunPod luu y container disk co the bi xoa, trong khi persistent/network volume
co vong doi rieng. Doc tai lieu cua pod dang thue truoc khi terminate:

- https://docs.runpod.io/pods/manage-pods

## 24. Checklist copy-paste truoc full run

```text
[ ] Code dung branch va commit
[ ] GPU dung model, VRAM du
[ ] Persistent disk du 300-500 GB hoac da tinh theo data
[ ] HF/Torch caches nam tren persistent volume
[ ] Video da upload day du
[ ] Moi collection co workdir rieng
[ ] Khong co duplicate filename stem trong mot batch
[ ] OPENAI_API_KEY va OPENAI_BASE_URL da load
[ ] Preflight status=ready
[ ] pip check thanh cong
[ ] Smoke test validation=true
[ ] Tmux dang chay
[ ] Monitoring command da thu
[ ] Biet cach resume
[ ] Biet cach backup truoc terminate
```

## 25. Lenh ngan gon sau khi da setup xong

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate
source /workspace/ebr/ebr.env
set -o pipefail

python scripts/check_unified_environment.py \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --spacy-model en_core_web_trf \
  --strict
```

Mo tmux:

```bash
tmux new -s ebr-run
```

Trong tmux:

```bash
cd /workspace/ebr/EBR-RAG
source .venv/bin/activate
source /workspace/ebr/ebr.env
set -o pipefail

python scripts/ingest_unified_v2.py \
  --video-dir /workspace/data/collections/COLLECTION/videos \
  --workdir /workspace/workdirs/COLLECTION-v2 \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --continue-on-error \
  2>&1 | tee /workspace/logs/COLLECTION-v2.log
```
