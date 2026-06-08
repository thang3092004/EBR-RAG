from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_frame(capture, cv2, time_s: float):
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def _frame_vector(frame, cv2) -> np.ndarray:
    resized = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    vector = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _candidate_times(
    segment: dict[str, Any],
    shot_data: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[float]:
    start = float(segment["context_start"])
    end = float(segment["context_end"])
    times = {start, (start + end) / 2.0, max(start, end - 1e-3)}
    now = math.ceil(start)
    while now < end:
        times.add(float(now))
        now += 1
    for boundary in shot_data.get("boundaries", []):
        time_s = float(boundary["time"])
        if start <= time_s <= end:
            times.add(time_s)
    track_times: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        time_s = float(observation.get("time", 0.0))
        if start <= time_s <= end:
            track_id = str(
                observation.get("entity_id")
                or observation.get("tracklet_id")
                or observation.get("local_track_id")
            )
            track_times[track_id].append(time_s)
    for values in track_times.values():
        times.add(min(values))
        times.add(max(values))
    return sorted(time_s for time_s in times if start <= time_s <= end)


def _visible_entities(
    time_s: float,
    observations: list[dict[str, Any]],
    tolerance: float = 0.35,
) -> set[str]:
    return {
        str(
            observation.get("entity_id")
            or observation.get("tracklet_id")
            or observation.get("local_track_id")
        )
        for observation in observations
        if abs(float(observation.get("time", 0.0)) - time_s) <= tolerance
    }


def _draw_overlay(frame, time_s: float, observations: list[dict[str, Any]], cv2):
    output = frame.copy()
    candidates = [
        observation
        for observation in observations
        if abs(float(observation.get("time", 0.0)) - time_s) <= 0.35
    ]
    for observation in candidates:
        bbox = observation.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        label = str(
            observation.get("entity_id")
            or observation.get("tracklet_id")
            or observation.get("local_track_id")
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def select_segment_frames(
    video_path: str,
    segment: dict[str, Any],
    profile: dict[str, Any],
    shot_data: dict[str, Any],
    observations: list[dict[str, Any]],
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Frame selection requires opencv-python.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    candidates = []
    for time_s in _candidate_times(segment, shot_data, observations):
        frame = _read_frame(capture, cv2, time_s)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates.append(
            {
                "time": time_s,
                "frame": frame,
                "vector": _frame_vector(frame, cv2),
                "quality": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "entities": _visible_entities(time_s, observations),
            }
        )
    capture.release()
    if not candidates:
        return {"frames": [], "embedding_backend": "opencv_histogram"}

    qualities = np.asarray([item["quality"] for item in candidates], dtype=np.float32)
    blur_cutoff = float(np.percentile(qualities, 20)) if len(candidates) >= 5 else -1.0
    filtered = [item for item in candidates if item["quality"] >= blur_cutoff]
    if len(filtered) < min(2, len(candidates)):
        filtered = sorted(candidates, key=lambda item: item["quality"], reverse=True)[:2]

    minimum = int(config.get("frame_min", 2))
    mode_max = {
        "visual_rich": int(config.get("frame_max", 6)),
        "balanced": min(4, int(config.get("frame_max", 6))),
        "speech_rich": max(2, min(3, int(config.get("frame_max", 6)))),
        "low_information": 2,
    }
    maximum = max(minimum, mode_max.get(profile.get("mode"), 4))
    duplicate_threshold = float(config.get("frame_duplicate_threshold", 0.94))
    gain_threshold = float(config.get("frame_marginal_gain_threshold", 0.05))
    all_entities = set().union(*(item["entities"] for item in filtered))
    start = float(segment["context_start"])
    end = float(segment["context_end"])
    bins = max(maximum, 1)

    selected: list[dict[str, Any]] = []
    covered_entities: set[str] = set()
    covered_bins: set[int] = set()
    while filtered and len(selected) < maximum:
        best = None
        best_gain = -1.0
        for item in filtered:
            if selected:
                max_similarity = max(
                    _cosine(item["vector"], existing["vector"])
                    for existing in selected
                )
                if max_similarity >= duplicate_threshold:
                    continue
                visual_gain = 1.0 - max_similarity
            else:
                visual_gain = 1.0
            new_entities = item["entities"] - covered_entities
            entity_gain = len(new_entities) / max(len(all_entities), 1)
            ratio = (item["time"] - start) / max(end - start, 1e-6)
            bin_index = min(bins - 1, max(0, int(ratio * bins)))
            temporal_gain = 1.0 if bin_index not in covered_bins else 0.0
            quality_gain = item["quality"] / max(float(np.max(qualities)), 1e-6)
            gain = (
                0.35 * visual_gain
                + 0.30 * entity_gain
                + 0.20 * temporal_gain
                + 0.15 * quality_gain
            )
            if gain > best_gain:
                best = item
                best_gain = gain
        if best is None:
            if len(selected) < minimum and filtered:
                remaining = list(filtered)
                if not remaining:
                    break
                best = max(
                    remaining,
                    key=lambda item: min(
                        (
                            abs(item["time"] - chosen["time"])
                            for chosen in selected
                        ),
                        default=float("inf"),
                    ),
                )
                best_gain = 0.0
            else:
                break
        if len(selected) >= minimum and best_gain < gain_threshold:
            break
        selected.append(best)
        covered_entities.update(best["entities"])
        ratio = (best["time"] - start) / max(end - start, 1e-6)
        covered_bins.add(min(bins - 1, max(0, int(ratio * bins))))
        filtered.remove(best)

    selected.sort(key=lambda item: item["time"])
    serialized = []
    for index, item in enumerate(selected):
        filename = f"{segment['segment_id']}_{index:02d}_{item['time']:.3f}.jpg"
        frame_path = output_path / filename
        overlay = _draw_overlay(item["frame"], item["time"], observations, cv2)
        cv2.imwrite(str(frame_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        serialized.append(
            {
                "time": item["time"],
                "path": str(frame_path),
                "quality": item["quality"],
                "entities": sorted(item["entities"]),
            }
        )
    return {
        "frames": serialized,
        "embedding_backend": "opencv_histogram",
        "candidate_count": len(candidates),
        "blur_cutoff": blur_cutoff,
        "covered_entities": sorted(covered_entities),
    }


def run_ocr(
    frame_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not config.get("enable_ocr", True):
        return {"available": False, "reason": "disabled", "items": []}
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return {
            "available": False,
            "reason": "paddleocr_not_installed",
            "items": [],
        }

    ocr = PaddleOCR(use_angle_cls=True, lang=str(config.get("ocr_language", "en")))
    items = []
    for frame in frame_records:
        result = ocr.ocr(frame["path"], cls=True)
        texts = []
        for page in result or []:
            for line in page or []:
                if len(line) < 2:
                    continue
                text, confidence = line[1]
                texts.append({"text": str(text), "confidence": float(confidence)})
        items.append({"time": frame["time"], "texts": texts})
    return {"available": True, "backend": "PaddleOCR", "items": items}
