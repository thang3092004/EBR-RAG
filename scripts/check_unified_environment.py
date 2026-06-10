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


def _embedding_report(
    *,
    check_connectivity: bool = False,
    model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ).strip()
    report = {
        "ok": bool(api_key),
        "provider": "openai_config",
        "base_url": base_url,
        "model": model,
        "reason": None if api_key else "OPENAI_API_KEY is not set.",
    }
    if not api_key or not check_connectivity:
        return report
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.embeddings.create(
            model=model,
            input=["Unified V2 strict preflight"],
            encoding_format="float",
        )
        dimensions = len(response.data[0].embedding)
        report.update(
            {
                "ok": dimensions == 1536,
                "dimensions": dimensions,
                "reason": (
                    None
                    if dimensions == 1536
                    else f"Expected 1536 embedding dimensions, got {dimensions}."
                ),
            }
        )
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return report


def _local_caption_model_report(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    required_files = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_files if not (path / name).is_file()]
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    weight_bytes = sum(weight.stat().st_size for weight in weights)
    ok = (
        path.is_dir()
        and not missing
        and bool(weights)
        and weight_bytes >= 5_000_000_000
    )
    return {
        "ok": ok,
        "path": str(resolved),
        "missing_files": missing,
        "weight_files": len(weights),
        "weight_bytes": weight_bytes,
        "minimum_weight_bytes": 5_000_000_000,
    }


def _local_asr_model_report(path: Path) -> dict[str, Any]:
    required_files = ("config.json", "model.bin", "tokenizer.json")
    missing = [name for name in required_files if not (path / name).is_file()]
    model_path = path / "model.bin"
    model_bytes = model_path.stat().st_size if model_path.is_file() else 0
    return {
        "ok": path.is_dir() and not missing and model_bytes >= 1_000_000_000,
        "path": str(path.resolve()),
        "missing_files": missing,
        "model_bytes": model_bytes,
        "minimum_model_bytes": 1_000_000_000,
    }


def _imagebind_checkpoint_report(
    path: Path,
    *,
    load_weights: bool,
) -> dict[str, Any]:
    report = _local_file_report(path, minimum_bytes=4_000_000_000)
    report["loaded"] = False
    if not report["ok"] or not load_weights:
        return report
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise RuntimeError("checkpoint did not contain a non-empty state dict")
        report["loaded"] = True
        report["state_entries"] = len(state)
        del state
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return report


def _local_file_report(path: Path, *, minimum_bytes: int = 1) -> dict[str, Any]:
    size = path.stat().st_size if path.is_file() else 0
    return {
        "ok": path.is_file() and size >= minimum_bytes,
        "path": str(path.resolve()),
        "size_bytes": size,
        "minimum_bytes": minimum_bytes,
    }


def _openclip_weights_report(
    model_name: str,
    pretrained: str,
    *,
    load_weights: bool,
) -> dict[str, Any]:
    report = {
        "ok": True,
        "model": model_name,
        "pretrained": pretrained,
        "loaded": False,
    }
    if not load_weights:
        return report
    try:
        import gc
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device="cpu",
        )
        report["loaded"] = True
        del model
        gc.collect()
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return report


def _print_json(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        description="Validate a server before running Unified Graph V2."
    )
    parser.add_argument(
        "--caption-model",
        default="./MiniCPM-V-2_6-int4",
        help="Local MiniCPM model directory.",
    )
    parser.add_argument(
        "--asr-model",
        default="./faster-distil-whisper-large-v3",
    )
    parser.add_argument(
        "--tracking-model",
        default="./yolov8n.pt",
    )
    parser.add_argument(
        "--imagebind-checkpoint",
        default="./.checkpoints/imagebind_huge.pth",
    )
    parser.add_argument("--openclip-model", default="ViT-B-32")
    parser.add_argument(
        "--openclip-pretrained",
        default="laion2b_s34b_b79k",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required check fails.",
    )
    parser.add_argument(
        "--strict-pipeline",
        action="store_true",
        help="Require local models and reject runtime fallback prerequisites.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_trf",
        help="spaCy pipeline used for transcript entities.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
    )
    args = parser.parse_args()

    caption_path = Path(args.caption_model).expanduser()
    asr_path = Path(args.asr_model).expanduser()
    tracking_path = Path(args.tracking_model).expanduser()
    imagebind_path = Path(args.imagebind_checkpoint).expanduser()
    checks: dict[str, Any] = {
        "python": {
            "ok": sys.version_info[:2] in {(3, 10), (3, 11)},
            "version": sys.version.split()[0],
            "recommended": "3.10 or 3.11",
        },
        "ffmpeg": {"ok": shutil.which("ffmpeg") is not None},
        "ffprobe": {"ok": shutil.which("ffprobe") is not None},
        "opencv": _package("opencv-python-headless", "cv2"),
        "scenedetect": _package("scenedetect"),
        "torch": _torch_report(),
        "torchvision": _package("torchvision"),
        "transformers": _package(
            "transformers",
            expected_version="4.40.0",
        ),
        "faster_whisper": _package("faster-whisper", "faster_whisper"),
        "ultralytics": _package("ultralytics"),
        "lap": _package("lap", expected_version="0.5.13"),
        "spacy": _package("spacy"),
        "spacy_model": _spacy_model_report(args.spacy_model),
        "open_clip": _package("open-clip-torch", "open_clip"),
        "openclip_weights": _openclip_weights_report(
            args.openclip_model,
            args.openclip_pretrained,
            load_weights=args.strict_pipeline,
        ),
        "moviepy": _package("moviepy"),
        "nano_vectordb": _package("nano-vectordb", "nano_vectordb"),
        "imagebind": _package("imagebind"),
        "pytorchvideo": _package("pytorchvideo"),
        "openai": _package("openai"),
        "embedding_api": _embedding_report(
            check_connectivity=args.strict_pipeline,
            model=args.embedding_model,
        ),
        "caption_model": _local_caption_model_report(caption_path),
        "asr_model": _local_asr_model_report(asr_path),
        "tracking_model": _local_file_report(
            tracking_path,
            minimum_bytes=1_000_000,
        ),
        "imagebind_checkpoint": _imagebind_checkpoint_report(
            imagebind_path,
            load_weights=args.strict_pipeline,
        ),
    }

    required = [
        "python",
        "ffmpeg",
        "ffprobe",
        "opencv",
        "scenedetect",
        "torch",
        "torchvision",
        "transformers",
        "faster_whisper",
        "ultralytics",
        "lap",
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
    if args.strict_pipeline:
        required.extend(
            (
                "caption_model",
                "asr_model",
                "tracking_model",
                "imagebind_checkpoint",
                "openclip_weights",
            )
        )
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
