from __future__ import annotations

import gc
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline.stage_runner import atomic_write_json, read_json
from .schema import normalize_entity_type


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _appearance_histogram(image: np.ndarray, bbox: list[float]) -> list[float] | None:
    try:
        import cv2
    except ImportError:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    vector = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
    return vector.tolist()


def _segment_for_time(time_s: float, segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        if float(segment["start"]) <= time_s <= float(segment["end"]):
            return str(segment["segment_id"])
    return ""


def _process_chunk(
    model,
    capture,
    chunk_index: int,
    start: float,
    end: float,
    fps: float,
    target_fps: float,
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    crop_dir: Path,
    progress=None,
) -> list[dict[str, Any]]:
    import cv2

    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps))
    stride = max(1, int(round(fps / max(target_fps, 0.1))))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    observations = []
    best_crops: dict[str, tuple[float, np.ndarray]] = {}
    class_names = getattr(model, "names", {}) or {}
    frame_index = start_frame

    # Reset tracker state at each resumable chunk. Global linking joins chunks later.
    if hasattr(model, "predictor"):
        model.predictor = None

    while frame_index <= end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % stride:
            frame_index += 1
            continue
        timestamp = frame_index / max(fps, 1e-6)
        outputs = model.track(
            source=frame,
            persist=True,
            tracker=str(config.get("entity_tracking_tracker", "botsort.yaml")),
            conf=float(config.get("entity_tracking_conf", 0.25)),
            iou=float(config.get("entity_tracking_iou", 0.5)),
            imgsz=int(config.get("entity_tracking_imgsz", 640)),
            verbose=False,
        )
        result = outputs[0] if outputs else None
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            ids = _to_numpy(getattr(boxes, "id", None))
            classes = _to_numpy(getattr(boxes, "cls", None))
            confidences = _to_numpy(getattr(boxes, "conf", None))
            xyxy = _to_numpy(getattr(boxes, "xyxy", None))
            if ids.size and xyxy.size:
                for box_index, raw_track_id in enumerate(ids.astype(int).tolist()):
                    class_id = (
                        int(classes[box_index]) if box_index < len(classes) else -1
                    )
                    label = str(class_names.get(class_id, class_id))
                    entity_type = normalize_entity_type(label)
                    bbox = [float(value) for value in xyxy[box_index].tolist()]
                    confidence = (
                        float(confidences[box_index])
                        if box_index < len(confidences)
                        else 0.0
                    )
                    local_track_id = f"chunk{chunk_index:05d}:{raw_track_id}"
                    tracklet_id = f"V_{entity_type.upper()}_{chunk_index:05d}_{raw_track_id}"
                    observation = {
                        "segment_id": _segment_for_time(timestamp, segments),
                        "tracklet_id": tracklet_id,
                        "local_track_id": local_track_id,
                        "frame_index": frame_index,
                        "time": timestamp,
                        "bbox": bbox,
                        "confidence": confidence,
                        "label": label,
                        "entity_type": entity_type,
                        "appearance": _appearance_histogram(frame, bbox),
                    }
                    observations.append(observation)
                    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                    crop = frame[
                        max(0, y1) : max(0, y2),
                        max(0, x1) : max(0, x2),
                    ]
                    if crop.size and (
                        tracklet_id not in best_crops
                        or confidence > best_crops[tracklet_id][0]
                    ):
                        best_crops[tracklet_id] = (confidence, crop.copy())
        frame_index += 1
        if progress is not None:
            progress.set(min(timestamp, end))

    for tracklet_id, (_, crop) in best_crops.items():
        crop_path = crop_dir / f"{tracklet_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        for observation in observations:
            if observation["tracklet_id"] == tracklet_id:
                observation["representative_crop"] = str(crop_path)
    return observations


def run_chunked_tracking(
    video_path: str,
    probe: dict[str, Any],
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    progress=None,
) -> list[dict[str, Any]]:
    if config.get("pipeline_strict", False):
        try:
            import lap
        except ImportError as exc:
            raise RuntimeError(
                "Strict pipeline requires lap==0.5.13 before tracking; "
                "runtime dependency installation is disabled."
            ) from exc
        if str(getattr(lap, "__version__", "")) != "0.5.13":
            raise RuntimeError(
                "Strict pipeline requires lap==0.5.13 before tracking."
            )
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Unified visual tracking requires opencv-python and ultralytics."
        ) from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    crop_dir = checkpoint_path / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    duration = float(probe["duration"])
    fps = float(probe.get("fps") or 30.0)
    target_fps = float(config.get("entity_tracking_fps", 3.0))
    chunk_seconds = float(config.get("tracking_chunk_seconds", 60.0))
    chunk_count = max(1, int(math.ceil(duration / chunk_seconds)))
    model_path = str(config.get("entity_tracking_model", "yolov8n.pt"))
    if config.get("pipeline_strict", False) and not Path(model_path).is_file():
        raise FileNotFoundError(
            "Strict pipeline requires a local YOLO weights file: "
            f"{Path(model_path).resolve()}"
        )
    model = YOLO(model_path)
    capture = cv2.VideoCapture(video_path)
    all_observations = []
    for chunk_index in range(chunk_count):
        chunk_file = checkpoint_path / f"chunk_{chunk_index:05d}.json"
        existing = read_json(chunk_file)
        start = chunk_index * chunk_seconds
        end = min(duration, (chunk_index + 1) * chunk_seconds)
        if isinstance(existing, list):
            all_observations.extend(existing)
            if progress is not None:
                progress.set(end, resumed_chunks=chunk_index + 1)
            continue
        chunk_observations = _process_chunk(
            model,
            capture,
            chunk_index,
            start,
            end,
            fps,
            target_fps,
            segments,
            config,
            crop_dir,
            progress=progress,
        )
        atomic_write_json(chunk_file, chunk_observations)
        all_observations.extend(chunk_observations)
    capture.release()
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return sorted(all_observations, key=lambda item: float(item["time"]))


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    return float(
        np.dot(left_array, right_array)
        / max(np.linalg.norm(left_array) * np.linalg.norm(right_array), 1e-12)
    )


def _mean(vectors: list[list[float] | None]) -> list[float] | None:
    arrays = [np.asarray(vector, dtype=np.float32) for vector in vectors if vector]
    if not arrays:
        return None
    result = np.mean(np.stack(arrays), axis=0)
    result /= max(float(np.linalg.norm(result)), 1e-12)
    return result.tolist()


def build_visual_tracklets(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["tracklet_id"])].append(observation)
    tracklets = []
    for tracklet_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: float(item["time"]))
        label = Counter(str(item["label"]) for item in ordered).most_common(1)[0][0]
        entity_type = Counter(
            str(item["entity_type"]) for item in ordered
        ).most_common(1)[0][0]
        tracklets.append(
            {
                "tracklet_id": tracklet_id,
                "entity_type": entity_type,
                "label": label,
                "start": float(ordered[0]["time"]),
                "end": float(ordered[-1]["time"]),
                "first_bbox": ordered[0]["bbox"],
                "last_bbox": ordered[-1]["bbox"],
                "appearance": _mean([item.get("appearance") for item in ordered]),
                "representative_crop": next(
                    (
                        item.get("representative_crop")
                        for item in ordered
                        if item.get("representative_crop")
                    ),
                    None,
                ),
                "observations": ordered,
            }
        )
    return sorted(tracklets, key=lambda item: (item["start"], item["tracklet_id"]))


class _DisjointSet:
    def __init__(self, keys: list[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not (
        float(left["end"]) < float(right["start"])
        or float(right["end"]) < float(left["start"])
    )


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _motion_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    gap = max(float(right["start"]) - float(left["end"]), 0.0)
    if gap > 3.0:
        return 0.0
    distance = math.dist(_center(left["last_bbox"]), _center(right["first_bbox"]))
    scale = max(
        math.sqrt(
            max(
                1.0,
                (left["last_bbox"][2] - left["last_bbox"][0])
                * (left["last_bbox"][3] - left["last_bbox"][1]),
            )
        ),
        1.0,
    )
    return math.exp(-distance / scale)


def link_visual_tracklets(
    tracklets: list[dict[str, Any]],
    registry,
    video_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config.get("visual_merge_threshold", 0.85))
    disjoint = _DisjointSet([item["tracklet_id"] for item in tracklets])
    if not bool(config.get("disable_visual_identity_linking", False)):
        for index, left in enumerate(tracklets):
            for right in tracklets[index + 1 :]:
                if left["entity_type"] != right["entity_type"]:
                    continue
                if _overlap(left, right):
                    continue
                gap = max(float(right["start"]) - float(left["end"]), 0.0)
                clip_score = _cosine(
                    left.get("clip_embedding"),
                    right.get("clip_embedding"),
                )
                appearance_score = _cosine(
                    left.get("appearance"),
                    right.get("appearance"),
                )
                class_score = 1.0 if left["label"] == right["label"] else 0.0
                temporal_score = math.exp(-gap / 300.0)
                score = (
                    0.45 * clip_score
                    + 0.20 * class_score
                    + 0.15 * _motion_score(left, right)
                    + 0.10 * temporal_score
                    + 0.10 * appearance_score
                )
                if score >= threshold:
                    disjoint.union(left["tracklet_id"], right["tracklet_id"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tracklet in tracklets:
        grouped[disjoint.find(tracklet["tracklet_id"])].append(tracklet)

    from .._unified_graph.schema import ProvenanceRecord

    visual_entities = []
    for _, group in sorted(
        grouped.items(),
        key=lambda item: min(float(tracklet["start"]) for tracklet in item[1]),
    ):
        entity_type = Counter(
            tracklet["entity_type"] for tracklet in group
        ).most_common(1)[0][0]
        label = Counter(tracklet["label"] for tracklet in group).most_common(1)[0][0]
        confidence = float(
            np.mean(
                [
                    observation["confidence"]
                    for tracklet in group
                    for observation in tracklet["observations"]
                ]
            )
        )
        global_id = registry.ensure_entity(
            entity_type,
            label,
            source="visual",
            confidence=confidence,
            attributes={"detector_label": label},
        )
        for tracklet in group:
            registry.add_alias(
                tracklet["tracklet_id"],
                global_id,
                source="visual",
                label=label,
                confidence=confidence,
            )
            for observation in tracklet["observations"]:
                observation["entity_id"] = global_id
                registry.add_provenance(
                    global_id,
                    ProvenanceRecord(
                        source="visual",
                        video_id=video_id,
                        segment_id=str(observation["segment_id"]),
                        start=float(observation["time"]),
                        end=float(observation["time"]),
                        frame_time=float(observation["time"]),
                        bbox=list(observation["bbox"]),
                        confidence=float(observation["confidence"]),
                    ),
                )
        visual_entities.append(
            {
                "entity_id": global_id,
                "entity_type": entity_type,
                "label": label,
                "tracklets": [tracklet["tracklet_id"] for tracklet in group],
                "confidence": confidence,
            }
        )
    return visual_entities
