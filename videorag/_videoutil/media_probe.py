from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _fraction(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / max(float(denominator), 1e-12)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def probe_video(video_path: str) -> dict[str, Any]:
    """Return stable media metadata, preferring ffprobe and falling back to OpenCV."""
    path = str(Path(video_path).resolve())
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        path,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            {},
        )
        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"),
            {},
        )
        duration = _fraction(
            video_stream.get("duration") or payload.get("format", {}).get("duration")
        )
        fps = _fraction(
            video_stream.get("avg_frame_rate")
            or video_stream.get("r_frame_rate"),
            30.0,
        )
        return {
            "path": path,
            "duration": duration,
            "fps": fps,
            "frame_count": int(video_stream.get("nb_frames") or round(duration * fps)),
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "has_audio": bool(audio_stream),
            "video_codec": video_stream.get("codec_name"),
            "audio_codec": audio_stream.get("codec_name"),
            "probe_backend": "ffprobe",
        }
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video probing requires ffprobe or opencv-python."
        ) from exc

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration = frame_count / max(fps, 1e-6)
    return {
        "path": path,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "has_audio": None,
        "video_codec": None,
        "audio_codec": None,
        "probe_backend": "opencv",
    }

