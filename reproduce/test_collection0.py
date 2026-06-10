"""
Smoke-test for EBR-RAG Unified V2 pipeline on collection 0 (fights-in-animal-kingdom).

Usage:
    python reproduce/test_collection0.py
    python reproduce/test_collection0.py --questions 0 1 2
    python reproduce/test_collection0.py --ingest-only
    python reproduce/test_collection0.py --query-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env if present
_env = ROOT / ".env"
if _env.is_file():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


DATASET_PATH = ROOT / "longervideos" / "dataset.json"
VIDEO_DIR    = ROOT / "longervideos" / "0-fights-in-animal-kingdom" / "videos"
WORK_DIR     = ROOT / "longervideos" / "0-fights-in-animal-kingdom" / "full_framework"
VIDEO_EXTS   = (".mp4", ".webm", ".mkv", ".mov", ".avi")


def _load_collection0() -> dict:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = data.get("0")
    if not rows:
        raise KeyError("Collection 0 not found in dataset.json")
    item = dict(rows[0])
    item["collection_id"] = "0"
    item["slug"] = "0-fights-in-animal-kingdom"
    return item


def _find_video(video_id: str) -> str:
    matches = sorted(
        p for p in VIDEO_DIR.glob(f"{video_id}.*")
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not matches:
        raise FileNotFoundError(
            f"Video {video_id!r} not found in {VIDEO_DIR}.\n"
            "Run yt-dlp to download it first:\n"
            f"  yt-dlp -o '{VIDEO_DIR}/{video_id}.%(ext)s' "
            f"https://www.youtube.com/watch?v={video_id}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple files match {video_id}: {matches}")
    return str(matches[0].resolve())


def _youtube_id(url: str) -> str:
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    values = parse_qs(parsed.query).get("v", [])
    if values:
        return values[0]
    raise ValueError(f"Cannot extract YouTube ID from {url!r}")


def _build_vrag() -> "VideoRAG":
    from videorag import VideoRAG
    from videorag._llm import openai_config

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return VideoRAG(
        llm=openai_config,
        working_dir=str(WORK_DIR),
        use_unified_graph=True,
        use_tm_graph=False,
        enable_entity_anchoring=False,
        pipeline_strict=False,
        pipeline_continue_on_error=True,
    )


def run_ingest(collection: dict, args) -> None:
    video_urls = collection.get("video_url", [])
    video_paths = []
    for url in video_urls:
        vid_id = _youtube_id(url)
        video_paths.append(_find_video(vid_id))

    print(f"\n=== INGESTING {len(video_paths)} VIDEO(S) ===")
    for p in video_paths:
        print(f"  {p}")

    vrag = _build_vrag()
    reports = vrag.insert_video(
        video_paths,
        resume=not args.no_resume,
    )
    print("\n=== INGEST COMPLETE ===")
    for r in reports:
        status = r.get("status", "?")
        vid    = r.get("video_id", "?")
        stages = r.get("stages_completed", [])
        print(f"  [{status}] {vid}  stages: {stages}")
    return vrag


async def run_queries(collection: dict, question_indices: list[int], args) -> None:
    from videorag import QueryParam

    all_questions = collection.get("questions", [])
    if not all_questions:
        print("No questions found in collection 0.")
        return

    if question_indices:
        questions = [all_questions[i] for i in question_indices if i < len(all_questions)]
    else:
        questions = all_questions[:3]

    vrag = _build_vrag()

    param = QueryParam(
        mode="EBR_RAG",
        initial_text_k=4,
        initial_graph_k=4,
        initial_visual_k=4,
        max_evidence=16,
        max_rounds=2,
        max_tool_calls_per_round=2,
        max_total_tool_calls=4,
        graph_context_token_cap=1800,
        debate_critique_see_evidence=False,
        debate_defender_disable_tools=False,
        return_detailed=True,
        wo_reference=True,
    )

    print(f"\n=== QUERYING {len(questions)} QUESTION(S) ===\n")
    for i, q_text in enumerate(questions):
        question = q_text if isinstance(q_text, str) else q_text.get("question", str(q_text))
        print(f"[Q{i}] {question}")
        print("-" * 60)
        try:
            response = await vrag.aquery(question, param=param)
            if isinstance(response, dict):
                answer    = response.get("answer", "").strip()
                rationale = response.get("rationale", "").strip()
                confidence = response.get("confidence", "?")
                rounds    = response.get("rounds_run", "?")
                tools     = response.get("tool_calls_made", "?")
                n_ev      = len(response.get("evidence", []))
                print(f"Answer:     {answer}")
                print(f"Confidence: {confidence}")
                print(f"Rounds:     {rounds}  |  Tool calls: {tools}  |  Evidence items: {n_ev}")
                if rationale:
                    print(f"Rationale:  {rationale[:200]}...")
            else:
                print(f"Answer: {str(response)}")
        except Exception:
            print("ERROR during query:")
            traceback.print_exc()
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test collection-0 pipeline")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--query-only",  action="store_true")
    parser.add_argument("--no-resume",   action="store_true")
    parser.add_argument(
        "--questions", nargs="*", type=int, default=None,
        help="0-based indices of questions to run (default: 0 1 2)",
    )
    args = parser.parse_args()

    collection = _load_collection0()
    print(f"Collection: {collection['slug']} | Type: {collection.get('type','?')} | "
          f"Questions: {len(collection.get('questions', []))}")

    if not args.query_only:
        run_ingest(collection, args)

    if not args.ingest_only:
        q_indices = args.questions if args.questions is not None else [0, 1, 2]
        asyncio.run(run_queries(collection, q_indices, args))


if __name__ == "__main__":
    main()
