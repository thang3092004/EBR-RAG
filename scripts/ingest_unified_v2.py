from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _videos(args) -> list[str]:
    paths = [Path(value) for value in args.video]
    if args.video_dir:
        video_dir = Path(args.video_dir)
        for extension in ("*.mp4", "*.webm", "*.mkv", "*.mov", "*.avi"):
            paths.extend(sorted(video_dir.glob(extension)))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.expanduser().resolve())
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    missing = [path for path in unique if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing video files:\n" + "\n".join(missing))
    by_video_id = {}
    for path in unique:
        video_id = Path(path).stem
        previous = by_video_id.get(video_id)
        if previous and previous != path:
            raise ValueError(
                "Video filename collision: both paths map to workspace ID "
                f"{video_id!r}:\n{previous}\n{path}"
            )
        by_video_id[video_id] = path
    return unique


def _gpu_report() -> dict:
    try:
        import torch

        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        return {"cuda_available": False, "reason": "torch_not_installed"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe Unified Multimodal Graph V2 ingestion."
    )
    parser.add_argument("--video", action="append", default=[], help="Video path; repeatable.")
    parser.add_argument("--video-dir", help="Ingest all supported videos in this directory.")
    parser.add_argument("--workdir", required=True, help="Collection work directory.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore completed stages.")
    parser.add_argument("--restart-stage", help="Re-run this stage and all downstream stages.")
    parser.add_argument("--force", action="store_true", help="Re-run every stage for each video.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed video and continue with the remaining collection.",
    )
    parser.add_argument("--asr-model", default="Systran/faster-distil-whisper-large-v3")
    parser.add_argument("--tracking-model", default="yolov8n.pt")
    parser.add_argument("--tracking-fps", type=float, default=3.0)
    parser.add_argument("--caption-model", default="./MiniCPM-V-2_6-int4")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument(
        "--profile",
        default="full_framework",
        choices=(
            "full_framework",
            "no_adaptive_segmentation",
            "no_transcript_memory",
            "no_visual_identity_linking",
            "no_crossmodal_alignment",
        ),
        help="Controlled ingestion ablation profile.",
    )
    parser.add_argument("--keep-segment-cache", action="store_true")
    args = parser.parse_args()

    videos = _videos(args)
    if not videos:
        parser.error("Provide at least one --video or --video-dir.")

    from videorag._llm import openai_config
    from videorag.ablation import ingestion_profile_overrides
    from videorag.pipeline.stage_runner import PIPELINE_VERSION
    from videorag.videorag import VideoRAG

    profile_overrides = ingestion_profile_overrides(args.profile)

    settings = {
        "pipeline": PIPELINE_VERSION,
        "workdir": str(Path(args.workdir).resolve()),
        "videos": videos,
        "resume": not args.no_resume,
        "profile": args.profile,
        "restart_stage": args.restart_stage,
        "force": args.force,
        "continue_on_error": args.continue_on_error,
        "models": {
            "caption_alignment": "MiniCPM-V-2_6-int4",
            "caption_model_path": args.caption_model,
            "asr": args.asr_model,
            "tracking": args.tracking_model,
            "tracker": "BoT-SORT",
            "appearance": "OpenCLIP ViT-B-32",
            "correspondence": "OpenCLIP ViT-B-32 text encoder",
            "text_entities": args.spacy_model,
            "visual_retrieval": "ImageBind-Huge",
        },
        "gpu": _gpu_report(),
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2))

    vrag = VideoRAG(
        llm=openai_config,
        working_dir=args.workdir,
        use_unified_graph=True,
        use_tm_graph=False,
        enable_entity_anchoring=False,
        asr_model=args.asr_model,
        entity_tracking_model=args.tracking_model,
        entity_tracking_fps=args.tracking_fps,
        caption_model_path=args.caption_model,
        spacy_model=args.spacy_model,
        keep_segment_cache=args.keep_segment_cache,
        pipeline_continue_on_error=args.continue_on_error,
        **profile_overrides,
    )
    reports = vrag.insert_video(
        videos,
        resume=not args.no_resume,
        restart_stage=args.restart_stage,
        force=args.force,
    )
    failed = [
        report
        for report in reports
        if report.get("status") not in {"complete", "completed"}
    ]
    status = "complete" if not failed else "partial_failure"
    print(
        json.dumps(
            {"status": status, "failed_videos": len(failed), "reports": reports},
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
