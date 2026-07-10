#!/usr/bin/env bash
# ==============================================================================
# EBR-RAG interactive demo — one-time setup.
#
# Installs the full ML stack + demo deps, downloads the 3 model checkpoints
# (MiniCPM-V, faster-whisper, ImageBind) PER THE README, and installs yt-dlp.
#
# Run from the REPO ROOT, inside your GPU python env (e.g. /venv/main or the
# `videorag` conda env). Checkpoints land in the repo root because the code
# resolves them relative to the CWD.
#
#   cd /root/EBR-RAG
#   source /venv/main/bin/activate    # or: conda activate videorag
#   bash demo_app/setup.sh
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
echo "==> Repo root: $ROOT_DIR"
echo "==> Using python: $(command -v python)  |  pip: $(command -v pip)"

# ── 1. Core ML stack (py3.12 / cu121 — keeps the existing torch 2.4.1) ────────
echo "==> Installing core requirements (requirements.txt)…"
pip install -r requirements.txt

# Video stack the README lists but requirements.txt omits:
echo "==> Installing eva-decord + pytorchvideo + ImageBind (patched, --no-deps)…"
pip install eva-decord==0.6.1
pip install --no-deps "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d"
pip install --no-deps "git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3"

echo "==> Installing demo deps (streamlit, pyvis, yt-dlp, pandas)…"
pip install -r demo_app/requirements-demo.txt

# ── 2. Model checkpoints (PER README) ─────────────────────────────────────────
echo "==> Ensuring git-lfs…"
git lfs install

if [ ! -d "$ROOT_DIR/MiniCPM-V-2_6-int4" ]; then
  echo "==> Downloading MiniCPM-V-2_6-int4 (caption VLM)…"
  git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4
else
  echo "==> MiniCPM-V-2_6-int4 already present — skipping."
fi

if [ ! -d "$ROOT_DIR/faster-distil-whisper-large-v3" ]; then
  echo "==> Downloading faster-distil-whisper-large-v3 (ASR)…"
  git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3
else
  echo "==> faster-distil-whisper-large-v3 already present — skipping."
fi

mkdir -p "$ROOT_DIR/.checkpoints"
if [ ! -f "$ROOT_DIR/.checkpoints/imagebind_huge.pth" ]; then
  echo "==> Downloading ImageBind checkpoint (imagebind_huge.pth, ~4.5GB)…"
  wget -O "$ROOT_DIR/.checkpoints/imagebind_huge.pth" \
    https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
else
  echo "==> imagebind_huge.pth already present — skipping."
fi

# ── 3. .env ───────────────────────────────────────────────────────────────────
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "OPENAI_API_KEY=" > "$ROOT_DIR/.env"
  echo "==> Created .env (empty OPENAI_API_KEY — paste your key)."
fi

echo ""
echo "=============================================================="
echo " Setup xong. Tiếp theo:"
echo "   1) Dán OpenAI key vào  $ROOT_DIR/.env   (OPENAI_API_KEY=sk-...)"
echo "   2) Từ thư mục gốc repo, chạy:"
echo "        streamlit run demo_app/app.py"
echo "   3) Mở tab ① Ingest, dán URL video (mặc định TED-Ed), bấm Tải & Ingest."
echo "=============================================================="
