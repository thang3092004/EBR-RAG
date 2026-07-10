"""
Central config for the EBR-RAG interactive demo.

Paths are resolved relative to the REPO ROOT (the parent of this demo_app/ folder),
because the VideoRAG code loads model checkpoints with paths relative to the current
working directory (e.g. ``./MiniCPM-V-2_6-int4`` and ``./.checkpoints/imagebind_huge.pth``).
=> Always launch the app from the repo root:  ``streamlit run demo_app/app.py``
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))          # .../EBR-RAG/demo_app
REPO_ROOT = os.path.dirname(DEMO_DIR)                           # .../EBR-RAG

VIDEO_DIR = os.path.join(DEMO_DIR, "videos")                    # downloaded mp4s
WORKDIR_ROOT = os.path.join(DEMO_DIR, "workdir")               # one VideoRAG working_dir per video

# ── Demo video ────────────────────────────────────────────────────────────────
# TED-Ed — "What staying up all night does to your brain" (~5 min). Short & self-contained.
DEMO_VIDEO_URL = "https://www.youtube.com/watch?v=idrbwnWLJ7w"

# ── Model checkpoints (must exist in REPO ROOT after running setup.sh) ─────────
CHECKPOINTS = {
    "MiniCPM-V (caption)": os.path.join(REPO_ROOT, "MiniCPM-V-2_6-int4"),
    "faster-whisper (ASR)": os.path.join(REPO_ROOT, "faster-distil-whisper-large-v3"),
    "ImageBind (visual)": os.path.join(REPO_ROOT, ".checkpoints", "imagebind_huge.pth"),
}


def workdir_for(video_name: str) -> str:
    """Return (and create) the VideoRAG working_dir for a given video basename."""
    path = os.path.join(WORKDIR_ROOT, video_name)
    os.makedirs(path, exist_ok=True)
    return path


def graphml_path(workdir: str) -> str:
    """Path to the knowledge-graph file written by NetworkXStorage during ingest."""
    return os.path.join(workdir, "graph_chunk_entity_relation.graphml")


def get_llm_config():
    """Repo-default OpenAI config: gpt-4o-mini + text-embedding-3-small.

    Imported lazily so that importing this module never triggers the heavy
    videorag dependency chain (torch, transformers, ...).
    """
    from videorag._llm import openai_config
    return openai_config


def openai_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())
