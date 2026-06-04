"""
ingest_test_video.py
====================
Script ingest video test_video với GPT-4o Vision thay thế MiniCPM-V.

Thay vì load MiniCPM-V cục bộ (cần GPU + 8GB VRAM), script này:
- Monkey-patches hàm segment_caption để dùng GPT-4o Vision API
- Chạy đầy đủ pipeline: split → whisper → gpt4o-caption → imagebind → TM graph
"""

import os
import sys
import base64
import logging

# Reconfigure stdout/stderr to use UTF-8 encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import warnings
import multiprocessing
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── USE GPT-4o Vision captioning by setting environment variable ──────────────────
os.environ["USE_GPT4O_CAPTION"] = "True"
os.environ.setdefault("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

# ─── INGEST ───────────────────────────────────────────────────────────────────

from videorag._llm import (
    LLMConfig, openai_embedding, gpt_4o_mini_complete
)
from videorag.videorag import VideoRAG

llm_config = LLMConfig(
    embedding_func_raw=openai_embedding,
    embedding_model_name="text-embedding-3-small",
    embedding_dim=1536,
    embedding_max_token_size=8192,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,

    best_model_func_raw=gpt_4o_mini_complete,
    best_model_name="gpt-4o-mini",
    best_model_max_token_size=32768,
    best_model_max_async=16,

    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)

VIDEO_DIR = os.environ.get("VIDEORAG_TEST_VIDEO_DIR", "./longervideos/test_video/videos")
WORKDIR = os.environ.get("VIDEORAG_TEST_WORKDIR", "./longervideos/videorag-workdir/test_video")

def main():
    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.endswith((".mp4", ".avi", ".mkv", ".webm"))
    ])
    if not video_files:
        print(f"❌ Không tìm thấy video nào trong {VIDEO_DIR}")
        sys.exit(1)

    video_paths = [os.path.join(VIDEO_DIR, f) for f in video_files]
    print(f"✅ Tìm thấy {len(video_paths)} video(s): {video_files}")

    # use_tm_graph=True để ingest với TM Graph
    vrag = VideoRAG(
        llm=llm_config,
        working_dir=WORKDIR,
        use_tm_graph=True,
    )

    print("\n🚀 Bắt đầu ingest test_video với GPT-4o Vision captions + TM Graph...\n")
    vrag.insert_video(video_path_list=video_paths)
    print("\n✅ HOÀN TẤT ingest test_video!\n")
    print(f"   Working dir: {WORKDIR}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
