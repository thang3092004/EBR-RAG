from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "longervideos"
EXPECTED = {
    "0-fights-in-animal-kingdom": ("lyd5Q77qHKA",),
    "6-daubechies-wavelet-lecture": (
        "RkGK0MloK0E",
        "1s9zZ6ERAko",
        "RNqXpdwd9AA",
        "2zaJZ_F7Xrk",
    ),
    "11-primetime-emmy-awards": (
        "dYX809pLH00",
        "5gWAZV8KoEw",
        "0udZzgn1UcM",
    ),
}


def _probe(path: Path) -> tuple[float, set[str], str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    stream_types = {
        str(stream.get("codec_type"))
        for stream in payload.get("streams", [])
    }
    video_codec = next(
        (
            str(stream.get("codec_name"))
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        "",
    )
    return duration, stream_types, video_codec


def _verify_opencv_decode(path: Path, duration: float) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        for ratio in (0.0, 0.5, 0.95):
            capture.set(cv2.CAP_PROP_POS_MSEC, duration * ratio * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"OpenCV could not decode {path} at {ratio:.0%} duration"
                )
    finally:
        capture.release()


def main() -> None:
    verified = 0
    for collection, video_ids in EXPECTED.items():
        video_dir = VIDEO_ROOT / collection / "videos"
        for video_id in video_ids:
            matches = sorted(
                path
                for path in video_dir.iterdir()
                if path.is_file() and path.stem == video_id
            ) if video_dir.is_dir() else []
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one media file for {video_id} in "
                    f"{video_dir}, found {len(matches)}."
                )
            path = matches[0]
            duration, stream_types, video_codec = _probe(path)
            missing = {"audio", "video"} - stream_types
            if duration <= 0 or missing:
                raise RuntimeError(
                    f"Invalid media {path}: duration={duration}, "
                    f"missing_streams={sorted(missing)}"
                )
            if video_codec != "h264":
                raise RuntimeError(
                    f"Strict pipeline requires H.264 input for OpenCV stages; "
                    f"{path} uses {video_codec or 'unknown'}."
                )
            _verify_opencv_decode(path, duration)
            verified += 1
            print(
                f"OK {collection}/{path.name}: "
                f"duration={duration:.1f}s audio+video"
            )
    print(f"Verified {verified} strict-ablation videos.")


if __name__ == "__main__":
    main()
