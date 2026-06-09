from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


ANIMAL_LABELS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

ENTITY_PREFIX = {
    "person": "PERSON",
    "object": "OBJECT",
    "animal": "ANIMAL",
    "screen_element": "SCREEN",
}


def normalize_entity_type(label: str) -> str:
    """Map detector labels to the entity groups used by the graph layer."""
    clean_label = (label or "").strip().lower().replace("_", " ")
    if clean_label == "person":
        return "person"
    if clean_label in ANIMAL_LABELS:
        return "animal"
    return "object"


@dataclass
class EntityObservation:
    video_name: str
    segment_id: str
    tracklet_id: str
    local_track_id: str
    frame_index: int
    time: float
    bbox: list[float]
    confidence: float
    label: str
    entity_type: str
    appearance: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tracklet:
    tracklet_id: str
    entity_type: str
    label: str
    observations: list[EntityObservation] = field(default_factory=list)
    appearance: Optional[list[float]] = None

    @property
    def start_time(self) -> float:
        return min((obs.time for obs in self.observations), default=0.0)

    @property
    def end_time(self) -> float:
        return max((obs.time for obs in self.observations), default=0.0)

    @property
    def first_bbox(self) -> list[float]:
        if not self.observations:
            return [0.0, 0.0, 0.0, 0.0]
        return min(self.observations, key=lambda obs: obs.time).bbox

    @property
    def last_bbox(self) -> list[float]:
        if not self.observations:
            return [0.0, 0.0, 0.0, 0.0]
        return max(self.observations, key=lambda obs: obs.time).bbox

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time
        data["end_time"] = self.end_time
        data["first_bbox"] = self.first_bbox
        data["last_bbox"] = self.last_bbox
        return data


@dataclass
class GlobalEntity:
    entity_id: str
    entity_type: str
    label: str
    tracklets: list[str]
    observations: list[EntityObservation]
    aliases: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def start_time(self) -> float:
        return min((obs.time for obs in self.observations), default=0.0)

    @property
    def end_time(self) -> float:
        return max((obs.time for obs in self.observations), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time
        data["end_time"] = self.end_time
        return data


@dataclass
class EntityAnchorResult:
    video_name: str
    entities: list[GlobalEntity]
    tracklets: list[Tracklet]
    segment_memory: dict[str, str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "entities": [entity.to_dict() for entity in self.entities],
            "tracklets": [tracklet.to_dict() for tracklet in self.tracklets],
            "segment_memory": self.segment_memory,
            "diagnostics": self.diagnostics,
        }
