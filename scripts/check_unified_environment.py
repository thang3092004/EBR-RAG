from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _package(
    name: str,
    import_name: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"ok": False, "reason": "not_installed"}

    module_name = import_name or name.replace("-", "_")
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {
            "ok": False,
            "version": version,
            "reason": f"import_failed: {type(exc).__name__}: {exc}",
        }
    if expected_version and version != expected_version:
        return {
            "ok": False,
            "version": version,
            "reason": f"expected_version: {expected_version}",
        }
    return {"ok": True, "version": version}


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    report: dict[str, Any] = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        report["gpus"].append(
            {
                "name": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    if not report["cuda_available"]:
        report["ok"] = False
        report["reason"] = "CUDA is unavailable."
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a server before running Unified Graph V2."
    )
    parser.add_argument(
        "--caption-model",
        default="./MiniCPM-V-2_6-int4",
        help="Local MiniCPM model directory; a missing path means HF fallback.",
    )
    parser.add_argument(
        "--require-diarization",
        action="store_true",
        help="Treat pyannote and HF_TOKEN as required.",
    )
    parser.add_argument(
        "--require-ocr",
        action="store_true",
        help="Treat PaddleOCR as required.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required check fails.",
    )
    args = parser.parse_args()

    caption_path = Path(args.caption_model).expanduser()
    checks: dict[str, Any] = {
        "python": {
            "ok": sys.version_info[:2] in {(3, 10), (3, 11)},
            "version": sys.version.split()[0],
            "recommended": "3.10 or 3.11",
        },
        "ffmpeg": {"ok": shutil.which("ffmpeg") is not None},
        "ffprobe": {"ok": shutil.which("ffprobe") is not None},
        "torch": _torch_report(),
        "torchvision": _package("torchvision"),
        "transformers": _package(
            "transformers",
            expected_version="4.40.0",
        ),
        "faster_whisper": _package("faster-whisper", "faster_whisper"),
        "ultralytics": _package("ultralytics"),
        "spacy": _package("spacy"),
        "open_clip": _package("open-clip-torch", "open_clip"),
        "pyannote": _package("pyannote.audio", "pyannote.audio"),
        "paddleocr": _package("paddleocr"),
        "caption_model": {
            "ok": caption_path.is_dir(),
            "path": str(caption_path.resolve()),
            "fallback": "openbmb/MiniCPM-V-2_6-int4",
        },
        "tokens": {
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "huggingface": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            ),
        },
    }

    required = [
        "python",
        "ffmpeg",
        "ffprobe",
        "torch",
        "torchvision",
        "transformers",
        "faster_whisper",
        "ultralytics",
        "spacy",
        "open_clip",
    ]
    if args.require_diarization:
        required.append("pyannote")
    if args.require_ocr:
        required.append("paddleocr")

    failures = [name for name in required if not checks[name].get("ok")]
    if not checks["tokens"]["openai"]:
        failures.append("OPENAI_API_KEY")
    if args.require_diarization and not checks["tokens"]["huggingface"]:
        failures.append("HF_TOKEN")

    report = {
        "status": "ready" if not failures else "not_ready",
        "required_failures": failures,
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
