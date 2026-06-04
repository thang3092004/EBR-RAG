from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._utils import logger
from .schema import EntityObservation, normalize_entity_type


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _segment_for_time(time_s: float, segment_times_info: dict[str, dict]) -> str:
    for segment_id, info in segment_times_info.items():
        timestamp = info.get("timestamp") if isinstance(info, dict) else None
        if timestamp is None:
            continue
        start, end = float(timestamp[0]), float(timestamp[1])
        if start <= time_s < end or math.isclose(time_s, end):
            return str(segment_id)
    return ""


def _appearance_histogram(image: np.ndarray, bbox: list[float], bins: int = 4) -> list[float] | None:
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hist_parts = []
    for channel in range(min(3, crop.shape[2])):
        hist, _ = np.histogram(crop[:, :, channel], bins=bins, range=(0, 256))
        hist_parts.append(hist.astype(np.float32))
    hist_vec = np.concatenate(hist_parts)
    norm = np.linalg.norm(hist_vec)
    if norm == 0:
        return None
    return (hist_vec / norm).astype(float).tolist()


def _video_fps(video_path: str) -> float:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        with VideoFileClip(video_path) as video:
            fps = _safe_float(getattr(video, "fps", 0.0), 0.0)
        return fps if fps > 0 else 30.0
    except Exception:
        logger.warning("Could not read video FPS with moviepy; falling back to 30 FPS for entity anchoring.")
        return 30.0


def run_ultralytics_tracking(
    video_name: str,
    video_path: str,
    segment_times_info: dict[str, dict],
    global_config: dict,
) -> list[EntityObservation]:
    """Run YOLO tracking when ultralytics is available.

    The dependency is intentionally optional. This keeps the existing VideoRAG
    ingestion path usable on machines that have not installed tracker models yet.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:
        if global_config.get("entity_anchor_strict", False):
            raise RuntimeError("Entity anchoring requires `ultralytics`.") from exc
        logger.warning(
            "Entity anchoring is enabled, but `ultralytics` is not installed. "
            "Skipping visual tracking for this video."
        )
        return []

    fps = _video_fps(video_path)
    target_fps = max(_safe_float(global_config.get("entity_tracking_fps", 3.0), 3.0), 0.1)
    configured_stride = int(global_config.get("entity_tracking_vid_stride") or 0)
    vid_stride = configured_stride if configured_stride > 0 else max(1, int(round(fps / target_fps)))

    model_name = global_config.get("entity_tracking_model", "yolov8n.pt")
    tracker_name = global_config.get("entity_tracking_tracker", "botsort.yaml")
    conf = _safe_float(global_config.get("entity_tracking_conf", 0.25), 0.25)
    iou = _safe_float(global_config.get("entity_tracking_iou", 0.5), 0.5)
    imgsz = int(global_config.get("entity_tracking_imgsz", 640))

    logger.info(
        f"[Entity Anchoring] Tracking {video_name} with {model_name}, "
        f"tracker={tracker_name}, stride={vid_stride}, fps={fps:.2f}"
    )

    model = YOLO(model_name)
    observations: list[EntityObservation] = []
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker_name,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        vid_stride=vid_stride,
        verbose=False,
    )

    class_names = getattr(model, "names", {}) or {}
    for output_index, result in enumerate(results):
        frame_index = output_index * vid_stride
        time_s = frame_index / fps
        segment_id = _segment_for_time(time_s, segment_times_info)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        ids = _to_numpy(getattr(boxes, "id", None))
        cls_ids = _to_numpy(getattr(boxes, "cls", None))
        confs = _to_numpy(getattr(boxes, "conf", None))
        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        if ids.size == 0 or xyxy.size == 0:
            continue

        image = getattr(result, "orig_img", None)
        for box_index, raw_track_id in enumerate(ids.astype(int).tolist()):
            class_id = int(cls_ids[box_index]) if box_index < len(cls_ids) else -1
            label = str(class_names.get(class_id, class_id))
            entity_type = normalize_entity_type(label)
            bbox = [float(x) for x in xyxy[box_index].tolist()]
            local_track_id = str(raw_track_id)
            tracklet_id = f"{video_name}:{entity_type}:{local_track_id}"
            confidence = float(confs[box_index]) if box_index < len(confs) else 0.0
            appearance = _appearance_histogram(image, bbox)
            observations.append(
                EntityObservation(
                    video_name=video_name,
                    segment_id=str(segment_id),
                    tracklet_id=tracklet_id,
                    local_track_id=local_track_id,
                    frame_index=frame_index,
                    time=float(time_s),
                    bbox=bbox,
                    confidence=confidence,
                    label=label,
                    entity_type=entity_type,
                    appearance=appearance,
                )
            )

    logger.info(f"[Entity Anchoring] Collected {len(observations)} observations for {video_name}.")
    return observations
