from __future__ import annotations

from collections import Counter

from .schema import GlobalEntity


def _segment_interval(info: dict) -> tuple[float, float]:
    timestamp = info.get("timestamp") if isinstance(info, dict) else None
    if timestamp is None:
        return 0.0, 0.0
    return float(timestamp[0]), float(timestamp[1])


def _position_name(bbox: list[float], frame_width: float = 1280.0) -> str:
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    ratio = center_x / max(frame_width, 1.0)
    if ratio < 0.33:
        return "left"
    if ratio > 0.66:
        return "right"
    return "center"


def _entity_line(entity: GlobalEntity, segment_id: str, start: float, end: float) -> str:
    observations = [
        obs for obs in entity.observations
        if obs.segment_id == str(segment_id) and start <= obs.time <= end
    ]
    if not observations:
        return ""

    visible_start = min(obs.time for obs in observations)
    visible_end = max(obs.time for obs in observations)
    positions = Counter(_position_name(obs.bbox) for obs in observations)
    dominant_position = positions.most_common(1)[0][0] if positions else "unknown"
    label_text = entity.label or entity.entity_type
    return (
        f"- {entity.entity_id}: {entity.entity_type} ({label_text}), "
        f"visible {visible_start:.1f}-{visible_end:.1f}s in this segment, "
        f"{len(observations)} detections, mostly {dominant_position}."
    )


def build_segment_memory(
    entities: list[GlobalEntity],
    segment_times_info: dict[str, dict],
    top_k: int = 12,
) -> dict[str, str]:
    segment_memory: dict[str, str] = {}
    for segment_id, info in segment_times_info.items():
        start, end = _segment_interval(info)
        candidate_lines = []
        for entity in entities:
            line = _entity_line(entity, str(segment_id), start, end)
            if line:
                count = sum(
                    1
                    for obs in entity.observations
                    if obs.segment_id == str(segment_id) and start <= obs.time <= end
                )
                candidate_lines.append((count, line))

        candidate_lines.sort(key=lambda item: item[0], reverse=True)
        lines = [line for _, line in candidate_lines[:top_k]]
        if lines:
            segment_memory[str(segment_id)] = (
                "Entity Memory (use these IDs exactly when describing visible entities):\n"
                + "\n".join(lines)
            )
        else:
            segment_memory[str(segment_id)] = ""
    return segment_memory
