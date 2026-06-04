import argparse
import json
import math
import multiprocessing
import os
import sys
import shutil
import importlib.util
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from videorag._llm import openai_4o_mini_config
from videorag.videorag import VideoRAG


def _default_video_dir(chunk_json: Path) -> Path:
    stem = chunk_json.stem
    for suffix in ("_mini_subset", "_subset"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return chunk_json.parent / stem


def _load_video_jobs(chunk_json: Path, video_dir: Path):
    with open(chunk_json, "r", encoding="utf-8-sig") as f:
        rows = json.load(f)

    seen = set()
    jobs = []
    for row in rows:
        uid = row["video_uid"]
        if uid in seen:
            continue
        seen.add(uid)
        path = video_dir / f"{uid}.mp4"
        jobs.append(
            {
                "uid": uid,
                "path": path,
                "duration": float(row.get("duration") or 0),
            }
        )
    return rows, jobs


def _usage_value(usage: dict, key: str) -> int:
    return int(usage.get(key) or 0)


def _usage_cost_usd(model: str, endpoint: str, usage: dict) -> float:
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")

    if model == "gpt-4o-mini":
        return (prompt_tokens / 1_000_000 * 0.15) + (completion_tokens / 1_000_000 * 0.60)
    if model == "text-embedding-3-small" or endpoint == "embeddings":
        return total_tokens / 1_000_000 * 0.02
    return 0.0


def _summarize_usage(usage_log: Path):
    if not usage_log.exists():
        print(f"usage_log_missing={usage_log}")
        return

    buckets = {}
    with open(usage_log, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            model = record.get("model", "")
            endpoint = record.get("endpoint", "")
            usage = record.get("usage") or {}
            key = (endpoint, model)
            bucket = buckets.setdefault(
                key,
                {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usd": 0.0},
            )
            bucket["requests"] += 1
            bucket["prompt_tokens"] += _usage_value(usage, "prompt_tokens")
            bucket["completion_tokens"] += _usage_value(usage, "completion_tokens")
            bucket["total_tokens"] += _usage_value(usage, "total_tokens")
            bucket["usd"] += _usage_cost_usd(model, endpoint, usage)

    print("\nOpenAI usage summary")
    total_usd = 0.0
    for (endpoint, model), bucket in sorted(buckets.items()):
        total_usd += bucket["usd"]
        print(
            f"- endpoint={endpoint} model={model} requests={bucket['requests']} "
            f"prompt_tokens={bucket['prompt_tokens']} completion_tokens={bucket['completion_tokens']} "
            f"total_tokens={bucket['total_tokens']} usd={bucket['usd']:.4f}"
        )
    print(f"total_usd={total_usd:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Ingest one CG-Bench chunk with entity anchoring.")
    parser.add_argument("--chunk-json", default="cg-bench/video_chunk_01_mini_subset.json")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--workdir", default="cg-bench/videorag-workdir/video_chunk_01_mini_subset_gpt4o_mini")
    parser.add_argument(
        "--caption-backend",
        choices=("openai", "minicpm"),
        default="openai",
        help="Use OpenAI vision API or local MiniCPM-V for segment captioning.",
    )
    parser.add_argument("--caption-model", default="gpt-4o-mini")
    parser.add_argument("--segment-length", type=int, default=30)
    parser.add_argument(
        "--entity-anchoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run local entity tracking before captioning so captions can use grounded IDs.",
    )
    parser.add_argument(
        "--entity-anchor-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail before captioning if local tracking dependencies are unavailable.",
    )
    parser.add_argument("--entity-tracking-model", default="yolov8n.pt")
    parser.add_argument("--entity-tracking-fps", type=float, default=3.0)
    parser.add_argument("--fresh", action="store_true", help="Delete the workdir before ingesting.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    chunk_json = Path(args.chunk_json)
    video_dir = Path(args.video_dir) if args.video_dir else _default_video_dir(chunk_json)
    workdir = Path(args.workdir)
    usage_log = workdir / "openai_usage.jsonl"

    rows, jobs = _load_video_jobs(chunk_json, video_dir)
    missing = [str(job["path"]) for job in jobs if not job["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing video files:\n" + "\n".join(missing))

    total_duration = sum(job["duration"] for job in jobs)
    expected_segments = sum(math.ceil(job["duration"] / args.segment_length) for job in jobs)
    print(f"chunk_json={chunk_json}")
    print(f"video_dir={video_dir}")
    print(f"workdir={workdir}")
    print(f"qa_count={len(rows)}")
    print(f"unique_videos={len(jobs)}")
    print(f"total_duration_seconds={total_duration:.2f}")
    print(f"expected_segments_approx={expected_segments}")
    print(f"openai_usage_log={usage_log}")
    print(f"caption_backend={args.caption_backend}")
    print(f"caption_model={args.caption_model if args.caption_backend == 'openai' else 'MiniCPM-V-2_6-int4'}")
    print(f"entity_anchoring={args.entity_anchoring}")
    print(f"entity_tracking_model={args.entity_tracking_model}")
    print(f"entity_tracking_fps={args.entity_tracking_fps}")
    print(f"entity_anchor_strict={args.entity_anchor_strict}")
    for job in jobs:
        print(f"- {job['uid']} duration={job['duration']:.2f}s path={job['path']}")

    if args.dry_run:
        return

    if args.fresh and workdir.exists():
        shutil.rmtree(workdir)

    if args.entity_anchoring and args.entity_anchor_strict:
        missing = [
            package
            for package in ("ultralytics", "cv2")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise RuntimeError(
                "Strict entity anchoring is enabled, but missing packages: "
                + ", ".join(missing)
                + ". Install them before running captioning."
            )

    if args.caption_backend == "openai":
        os.environ["USE_GPT4O_CAPTION"] = "True"
        os.environ["OPENAI_CAPTION_MODEL"] = args.caption_model
    else:
        os.environ.pop("USE_GPT4O_CAPTION", None)
        os.environ.pop("OPENAI_CAPTION_MODEL", None)
    os.environ["OPENAI_USAGE_LOG"] = str(usage_log)

    vrag = VideoRAG(
        llm=openai_4o_mini_config,
        working_dir=str(workdir),
        use_tm_graph=True,
        enable_entity_anchoring=args.entity_anchoring,
        entity_anchor_strict=args.entity_anchor_strict,
        entity_tracking_model=args.entity_tracking_model,
        entity_tracking_fps=args.entity_tracking_fps,
    )
    vrag.insert_video(video_path_list=[str(job["path"]) for job in jobs])
    print(f"usage_log={usage_log}")
    _summarize_usage(usage_log)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
