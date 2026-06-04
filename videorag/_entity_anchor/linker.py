from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .schema import ENTITY_PREFIX, EntityObservation, GlobalEntity, Tracklet


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _mean_vector(vectors: list[list[float] | None]) -> list[float] | None:
    clean_vectors = [np.asarray(v, dtype=np.float32) for v in vectors if v]
    if not clean_vectors:
        return None
    mean_vec = np.mean(np.stack(clean_vectors, axis=0), axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    return (mean_vec / norm).astype(float).tolist()


def build_tracklets(observations: list[EntityObservation]) -> list[Tracklet]:
    grouped: dict[str, list[EntityObservation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.tracklet_id].append(obs)

    tracklets: list[Tracklet] = []
    for tracklet_id, obs_list in grouped.items():
        obs_list = sorted(obs_list, key=lambda obs: obs.time)
        label = Counter(obs.label for obs in obs_list).most_common(1)[0][0]
        entity_type = Counter(obs.entity_type for obs in obs_list).most_common(1)[0][0]
        tracklets.append(
            Tracklet(
                tracklet_id=tracklet_id,
                entity_type=entity_type,
                label=label,
                observations=obs_list,
                appearance=_mean_vector([obs.appearance for obs in obs_list]),
            )
        )
    return sorted(tracklets, key=lambda tr: (tr.start_time, tr.tracklet_id))


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_scale(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))))


def _time_gap(left: Tracklet, right: Tracklet) -> float:
    if left.end_time < right.start_time:
        return right.start_time - left.end_time
    if right.end_time < left.start_time:
        return left.start_time - right.end_time
    return 0.0


def _overlaps_in_time(left: Tracklet, right: Tracklet, tolerance: float = 0.05) -> bool:
    return not (left.end_time + tolerance < right.start_time or right.end_time + tolerance < left.start_time)


def _motion_compatibility(left: Tracklet, right: Tracklet) -> float:
    if left.end_time <= right.start_time:
        first, second = left, right
    else:
        first, second = right, left

    c1 = _center(first.last_bbox)
    c2 = _center(second.first_bbox)
    distance = math.dist(c1, c2)
    scale = max(_bbox_scale(first.last_bbox), _bbox_scale(second.first_bbox))
    return math.exp(-distance / max(scale, 1.0))


def _link_score(left: Tracklet, right: Tracklet, max_time_gap: float) -> float:
    gap = _time_gap(left, right)
    appearance_score = _cosine(left.appearance, right.appearance)
    time_score = math.exp(-gap / max(max_time_gap, 0.1))
    motion_score = _motion_compatibility(left, right)
    return 0.60 * appearance_score + 0.25 * time_score + 0.15 * motion_score


def link_tracklets(tracklets: list[Tracklet], global_config: dict) -> list[GlobalEntity]:
    """Merge local tracklets into global entity IDs with train-free constraints."""
    if not tracklets:
        return []

    max_time_gap = float(global_config.get("entity_linking_max_time_gap", 3.0))
    threshold = float(global_config.get("entity_linking_similarity_threshold", 0.82))
    dsu = _DisjointSet([tracklet.tracklet_id for tracklet in tracklets])

    for idx, left in enumerate(tracklets):
        for right in tracklets[idx + 1 :]:
            if left.entity_type != right.entity_type:
                continue
            if _overlaps_in_time(left, right):
                continue
            if _time_gap(left, right) > max_time_gap:
                continue
            if _link_score(left, right, max_time_gap) >= threshold:
                dsu.union(left.tracklet_id, right.tracklet_id)

    grouped: dict[str, list[Tracklet]] = defaultdict(list)
    for tracklet in tracklets:
        grouped[dsu.find(tracklet.tracklet_id)].append(tracklet)

    counters: dict[str, int] = defaultdict(int)
    entities: list[GlobalEntity] = []
    for _, group in sorted(grouped.items(), key=lambda item: min(tr.start_time for tr in item[1])):
        entity_type = Counter(tr.entity_type for tr in group).most_common(1)[0][0]
        label = Counter(tr.label for tr in group).most_common(1)[0][0]
        counters[entity_type] += 1
        prefix = ENTITY_PREFIX.get(entity_type, entity_type.upper())
        entity_id = f"{prefix}_{counters[entity_type]:03d}"
        observations = sorted(
            [obs for tracklet in group for obs in tracklet.observations],
            key=lambda obs: obs.time,
        )
        for obs in observations:
            obs.tracklet_id = entity_id
        confidence = float(np.mean([obs.confidence for obs in observations])) if observations else 0.0
        entities.append(
            GlobalEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                label=label,
                tracklets=[tracklet.tracklet_id for tracklet in group],
                observations=observations,
                aliases=[label] if label else [],
                attributes=[f"detector_label:{label}"] if label else [],
                confidence=confidence,
            )
        )
    return entities
