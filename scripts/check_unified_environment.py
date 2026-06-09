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


def _spacy_model_report(model_name: str) -> dict[str, Any]:
    try:
        import spacy

        nlp = spacy.load(model_name)
        return {
            "ok": True,
            "model": model_name,
            "pipeline": list(nlp.pipe_names),
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model_name,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _embedding_report() -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ).strip()
    return {
        "ok": bool(api_key),
        "provider": "openai_config",
        "base_url": base_url,
        "reason": None if api_key else "OPENAI_API_KEY is not set.",
    }


def _print_json(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


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
        "--strict",
        action="store_true",
        help="Exit non-zero when a required check fails.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_trf",
        help="spaCy pipeline used for transcript entities.",
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
        "spacy_model": _spacy_model_report(args.spacy_model),
        "open_clip": _package("open-clip-torch", "open_clip"),
        "moviepy": _package("moviepy"),
        "nano_vectordb": _package("nano-vectordb", "nano_vectordb"),
        "imagebind": _package("imagebind"),
        "pytorchvideo": _package("pytorchvideo"),
        "openai": _package("openai"),
        "embedding_api": _embedding_report(),
        "caption_model": {
            "ok": caption_path.is_dir(),
            "path": str(caption_path.resolve()),
            "fallback": "openbmb/MiniCPM-V-2_6-int4",
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
        "spacy_model",
        "open_clip",
        "moviepy",
        "nano_vectordb",
        "imagebind",
        "pytorchvideo",
        "openai",
        "embedding_api",
    ]
    failures = [name for name in required if not checks[name].get("ok")]

    report = {
        "status": "ready" if not failures else "not_ready",
        "required_failures": failures,
        "checks": checks,
    }
    _print_json(report)
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
