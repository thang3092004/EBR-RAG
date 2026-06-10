#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${1:-$ROOT/workdirs/full-framework}"
OUTPUT_ROOT="${2:-$ROOT/reproduce/full-framework-results}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1

cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; environment setup is incomplete." >&2
  exit 1
fi

mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT"

echo "Strict work root: $WORK_ROOT"
echo "Results root:     $OUTPUT_ROOT"
echo "Monitor command:"
echo "  $ROOT/.venv/bin/python $ROOT/scripts/show_unified_progress.py --workdir $WORK_ROOT --recursive --watch --interval 2"

.venv/bin/python scripts/verify_minimal_ablation_data.py

.venv/bin/python scripts/check_unified_environment.py \
  --caption-model "$ROOT/MiniCPM-V-2_6-int4" \
  --asr-model "$ROOT/faster-distil-whisper-large-v3" \
  --tracking-model "$ROOT/yolov8n.pt" \
  --imagebind-checkpoint "$ROOT/.checkpoints/imagebind_huge.pth" \
  --spacy-model en_core_web_trf \
  --strict-pipeline \
  --strict

.venv/bin/python reproduce/run_ablation_matrix.py ingest \
  --collections 0 6 11 \
  --ingestion-profiles full_framework \
  --dataset "$ROOT/longervideos/dataset.json" \
  --video-root "$ROOT/longervideos" \
  --work-root "$WORK_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --caption-model "$ROOT/MiniCPM-V-2_6-int4" \
  --asr-model "$ROOT/faster-distil-whisper-large-v3" \
  --tracking-model "$ROOT/yolov8n.pt" \
  --tracking-fps 0.5 \
  --spacy-model en_core_web_trf \
  --strict-pipeline \
  --no-reuse-full-artifacts
