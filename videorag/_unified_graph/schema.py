from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GLOBAL_PREFIXES = {
    "person": "PERSON",
    "object": "OBJECT",
    "animal": "ANIMAL",
    "location": "LOCATION",
    "organization": "ORGANIZATION",
    "concept": "CONCEPT",
    "event": "EVENT",
    "screen_element": "SCREEN_ELEMENT",
    "unknown": "ENTITY",
}


def normalize_entity_type(value: str | None) -> str:
    clean = str(value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "per": "person",
        "person": "person",
        "org": "organization",
        "gpe": "location",
        "loc": "location",
        "facility": "location",
        "product": "object",
        "work_of_art": "object",
        "nor": "concept",
        "date": "concept",
        "time": "concept",
    }
    return aliases.get(clean, clean if clean in GLOBAL_PREFIXES else "unknown")


def bbox_position(bbox: list[float] | None) -> str | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    h_zone = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    v_zone = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
    size = "small" if area < 0.05 else ("large" if area > 0.20 else "medium")
    return f"{v_zone}-{h_zone}, {size}"


@dataclass
class ProvenanceRecord:
    source: str
    video_id: str
    segment_id: str
    start: float
    end: float
    frame_time: float | None = None
    bbox: list[float] | None = None
    position: str | None = None
    text: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityAlias:
    alias_id: str
    source: str
    label: str = ""
    confidence: float = 0.0
    segment_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityNode:
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[EntityAlias] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None
    confidence: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aliases"] = [alias.to_dict() for alias in self.aliases]
        data["provenance"] = [item.to_dict() for item in self.provenance]
        return data


@dataclass
class EdgeOccurrence:
    edge_id: str
    source_id: str
    target_id: str
    predicate: str
    start: float
    end: float
    segment_id: str
    confidence: float
    modalities: list[str]
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = [item.to_dict() for item in self.provenance]
        return data
