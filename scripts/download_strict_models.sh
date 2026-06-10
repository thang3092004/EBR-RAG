#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"

cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; create and install the environment first." >&2
  exit 1
fi

git lfs install

if [[ -d MiniCPM-V-2_6-int4/.git ]]; then
  git -C MiniCPM-V-2_6-int4 lfs pull
else
  git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4
fi

if [[ -d faster-distil-whisper-large-v3/.git ]]; then
  git -C faster-distil-whisper-large-v3 lfs pull
else
  git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3
fi

mkdir -p .checkpoints
wget -c \
  https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth \
  -O .checkpoints/imagebind_huge.pth

.venv/bin/python -c \
  'from ultralytics import YOLO; YOLO("yolov8n.pt"); print("YOLO ready")'

.venv/bin/python -c \
  'import open_clip; model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"); print("OpenCLIP ready")'

echo "All strict model assets are downloaded."
