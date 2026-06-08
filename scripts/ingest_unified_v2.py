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
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--enable-talknet", action="store_true")
    parser.add_argument(
        "--talknet-command",
        default="",
        help="Command template with {video}, {tracks}, and {output} placeholders.",
    )
    parser.add_argument("--asr-model", default="Systran/faster-distil-whisper-large-v3")
    parser.add_argument("--tracking-model", default="yolov8n.pt")
    parser.add_argument("--tracking-fps", type=float, default=3.0)
    parser.add_argument("--deep-tracking-fps", type=float, default=6.0)
    parser.add_argument("--caption-model", default="./MiniCPM-V-2_6-int4")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--keep-segment-cache", action="store_true")
    args = parser.parse_args()

    videos = _videos(args)
    if not videos:
        parser.error("Provide at least one --video or --video-dir.")

    from videorag._llm import openai_config
    from videorag.videorag import VideoRAG

    settings = {
        "pipeline": "unified-graph-v2",
        "workdir": str(Path(args.workdir).resolve()),
        "videos": videos,
        "resume": not args.no_resume,
        "restart_stage": args.restart_stage,
        "force": args.force,
        "models": {
            "caption_alignment": "MiniCPM-V-2_6-int4",
            "caption_model_path": args.caption_model,
            "asr": args.asr_model,
            "tracking": args.tracking_model,
            "tracker": "BoT-SORT",
            "appearance": "OpenCLIP ViT-B-32",
            "diarization": (
                "disabled"
                if args.disable_diarization
                else "pyannote/speaker-diarization-3.1"
            ),
            "active_speaker": "TalkNet" if args.enable_talknet else "disabled",
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
        entity_tracking_deep_fps=args.deep_tracking_fps,
        caption_model_path=args.caption_model,
        spacy_model=args.spacy_model,
        enable_ocr=not args.disable_ocr,
        enable_diarization=not args.disable_diarization,
        enable_talknet=args.enable_talknet,
        talknet_command=args.talknet_command,
        keep_segment_cache=args.keep_segment_cache,
    )
    reports = vrag.insert_video(
        videos,
        resume=not args.no_resume,
        restart_stage=args.restart_stage,
        force=args.force,
    )
    print(json.dumps({"status": "complete", "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

