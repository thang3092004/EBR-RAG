from __future__ import annotations

import math
from typing import Any

import numpy as np


def _histogram(frame, cv2) -> np.ndarray:
    resized = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram.astype(np.float32)


def _opencv_shots(
    video_path: str,
    duration: float,
    sample_fps: float,
    threshold: float,
    progress=None,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Shot detection fallback requires opencv-python.") from exc

    capture = cv2.VideoCapture(video_path)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(source_fps / max(sample_fps, 0.1))))
    frame_index = 0
    previous_histogram = None
    previous_gray = None
    previous_edge_density = None
    boundaries: list[dict[str, float]] = []
    samples: list[dict[str, float]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        timestamp = frame_index / max(source_fps, 1e-6)
        histogram = _histogram(frame, cv2)
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))
        shot_score = 0.0
        motion_score = 0.0
        if previous_histogram is not None:
            shot_score = float(
                cv2.compareHist(
                    previous_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
        if previous_gray is not None:
            motion_score = float(
                np.mean(cv2.absdiff(previous_gray, gray)) / 255.0
            )
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        text_change_score = (
            abs(edge_density - previous_edge_density)
            if previous_edge_density is not None
            else 0.0
        )
        sample = {
            "time": timestamp,
            "shot_score": shot_score,
            "motion_score": motion_score,
            "blur_score": blur_score,
            "edge_density": edge_density,
            "text_change_score": text_change_score,
        }
        samples.append(sample)
        if shot_score >= threshold:
            boundaries.append({"time": timestamp, "score": min(shot_score, 1.0)})
        previous_histogram = histogram
        previous_gray = gray
        previous_edge_density = edge_density
        frame_index += 1
        if progress is not None:
            progress.set(min(timestamp, duration), shots=len(boundaries))
    capture.release()
    return {
        "backend": "opencv_histogram",
        "boundaries": boundaries,
        "samples": samples,
    }


def detect_shots_and_motion(
    video_path: str,
    duration: float,
    config: dict[str, Any],
    *,
    progress=None,
) -> dict[str, Any]:
    """Detect scene boundaries and collect deterministic motion/quality samples."""
    threshold = float(config.get("shot_detector_threshold", 0.38))
    sample_fps = float(config.get("motion_sample_fps", 2.0))

    result = _opencv_shots(
        video_path,
        duration,
        sample_fps,
        threshold,
        progress=progress,
    )

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(open_video(video_path), show_progress=False)
        scene_list = scene_manager.get_scene_list()
        boundaries = []
        for scene_index, (_, end) in enumerate(scene_list[:-1]):
            timestamp = float(end.get_seconds())
            nearest = min(
                result["samples"],
                key=lambda item: abs(float(item["time"]) - timestamp),
                default={"shot_score": 1.0},
            )
            boundaries.append(
                {
                    "time": timestamp,
                    "score": max(float(nearest.get("shot_score", 0.0)), 0.5),
                    "scene_index": scene_index,
                }
            )
        result["backend"] = "pyscenedetect+opencv_metrics"
        result["boundaries"] = boundaries
    except (ImportError, RuntimeError, OSError, ValueError):
        pass

    result["duration"] = duration
    result["sample_fps"] = sample_fps
    return result
