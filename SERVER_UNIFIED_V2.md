# Unified Multimodal Graph V2: Server Guide

## Recommended server

- Ubuntu 22.04 and Python 3.10.
- NVIDIA GPU with at least 24 GB VRAM; 48 GB is preferable for smoother
  MiniCPM, pyannote, and ImageBind stage transitions.
- At least 64 GB RAM and enough SSD space for the source videos, models, and
  temporary segment clips.
- CUDA 12.1 compatible driver.

The pipeline runs heavy models sequentially and releases GPU memory between
stages. It does not require all models to fit in VRAM at once.

## Installation

```bash
git clone <your-repository-url> EBR-RAG
cd EBR-RAG
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-unified-v2.txt
python -m spacy download en_core_web_trf

pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
```

MiniCPM-V-2_6-int4 is pinned to `transformers==4.40.0`, matching its
[official model card](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4).

Install `ffmpeg`:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Keep the MiniCPM directory at `./MiniCPM-V-2_6-int4`, or pass another local
path through `--caption-model`. Caption and cross-modal alignment always use
MiniCPM in V2.

For pyannote, accept the model terms on Hugging Face and export:

```bash
export HF_TOKEN=...
```

The current text and entity vector indexes use the configured VideoRAG
embedding function:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

This API is used for embeddings and later debate. Ingest caption/alignment does
not call GPT.

Before renting a long GPU session, validate the environment:

```bash
python scripts/check_unified_environment.py \
  --caption-model ./MiniCPM-V-2_6-int4 \
  --require-diarization \
  --strict
```

This imports both PyTorch and torchvision, so it catches incompatible wheel
pairs before the ingest starts.

## First run

Run inside `tmux` so an SSH disconnect does not stop the job:

```bash
tmux new -s ebr-ingest
source .venv/bin/activate

python scripts/ingest_unified_v2.py \
  --video /data/videos/DcWqzZ3I2cY.webm \
  --workdir /data/workdirs/19-jeff-bezos-v2
```

For a collection:

```bash
python scripts/ingest_unified_v2.py \
  --video-dir /data/collections/19-jeff-bezos/videos \
  --workdir /data/workdirs/19-jeff-bezos-v2
```

Default important settings:

```text
ASR: faster-distil-whisper-large-v3
caption/alignment: MiniCPM-V-2_6-int4
tracking: YOLOv8n + BoT-SORT, 3 FPS
deep visual tracking: 6 FPS on visual-rich segments
appearance: OpenCLIP ViT-B-32
diarization: pyannote speaker-diarization-3.1
visual retrieval: ImageBind-Huge
segment target/min/max: 24/8/45 seconds
frames: 2-6 per segment
```

## Resume and restart

Resume is on by default. Re-run the same command after interruption:

```bash
python scripts/ingest_unified_v2.py \
  --video /data/videos/DcWqzZ3I2cY.webm \
  --workdir /data/workdirs/19-jeff-bezos-v2
```

Restart one stage and every dependent stage:

```bash
python scripts/ingest_unified_v2.py \
  --video /data/videos/DcWqzZ3I2cY.webm \
  --workdir /data/workdirs/19-jeff-bezos-v2 \
  --restart-stage alignment_caption
```

Re-run the complete video:

```bash
python scripts/ingest_unified_v2.py \
  --video /data/videos/DcWqzZ3I2cY.webm \
  --workdir /data/workdirs/19-jeff-bezos-v2 \
  --force
```

Changing a stage-specific configuration automatically invalidates that stage
and its downstream dependencies.

## Monitoring

The main command shows video and stage progress bars. From another terminal:

```bash
python scripts/show_unified_progress.py \
  --workdir /data/workdirs/19-jeff-bezos-v2 \
  --watch
```

Per-video files are stored under:

```text
<workdir>/pipeline_v2/<video_id>/
  manifest.json
  progress.json
  events.jsonl
  pipeline.log
  run_report.json
  validation/validation_report.json
```

`manifest.json` records stage status and configuration hashes.
`run_report.json` records models, parameters, timings, and stage metrics.

## Optional components

OCR is optional. If PaddlePaddle is not installed, the run report records
`paddleocr_not_installed` and processing continues.

To enable OCR, first install the PaddlePaddle GPU wheel matching the server's
CUDA version, then:

```bash
pip install -r requirements-unified-v2-ocr.txt
python scripts/check_unified_environment.py --require-ocr --strict
```

TalkNet is disabled by default because its official repository is not a stable
Python package. Enable it through an external wrapper:

```bash
python scripts/ingest_unified_v2.py ... \
  --enable-talknet \
  --talknet-command \
  "python /opt/talknet/run_adapter.py --video {video} --tracks {tracks} --output {output}"
```

The wrapper must write:

```json
{
  "links": [
    {
      "speaker_id": "SPEAKER_001",
      "person_id": "V_PERSON_00001_2",
      "score": 0.91,
      "overlap_seconds": 4.2
    }
  ]
}
```

Links are accepted only above the configured confidence, margin, and overlap
thresholds. Missing TalkNet never causes the pipeline to guess a speaker.
