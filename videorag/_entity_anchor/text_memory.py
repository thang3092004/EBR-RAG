from __future__ import annotations

import math
from collections import Counter
from typing import Any


FIRST_SECOND_PERSON = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

REFERENCE_SPECS = {
    "he": ({"person"}, "singular", "male"),
    "him": ({"person"}, "singular", "male"),
    "his": ({"person"}, "singular", "male"),
    "himself": ({"person"}, "singular", "male"),
    "she": ({"person"}, "singular", "female"),
    "her": ({"person"}, "singular", "female"),
    "hers": ({"person"}, "singular", "female"),
    "herself": ({"person"}, "singular", "female"),
    "they": ({"person", "organization"}, "unknown", None),
    "them": ({"person", "organization"}, "unknown", None),
    "their": ({"person", "organization"}, "unknown", None),
    "theirs": ({"person", "organization"}, "unknown", None),
    "themselves": ({"person", "organization"}, "plural", None),
    "it": ({"object", "organization", "concept", "event", "animal"}, "singular", None),
    "its": ({"object", "organization", "concept", "event", "animal"}, "singular", None),
    "itself": ({"object", "organization", "concept", "event", "animal"}, "singular", None),
    "this": ({"object", "concept", "event"}, "singular", None),
    "that": ({"object", "concept", "event"}, "singular", None),
    "these": ({"object", "concept", "event"}, "plural", None),
    "those": ({"object", "concept", "event"}, "plural", None),
    "who": ({"person"}, "unknown", None),
    "whom": ({"person"}, "unknown", None),
    "whose": ({"person"}, "unknown", None),
    "which": ({"object", "organization", "concept", "event", "animal"}, "unknown", None),
}

GENERIC_REFERENCE_SPECS = {
    "person": ({"person"}, "singular", None),
    "people": ({"person"}, "plural", None),
    "man": ({"person"}, "singular", "male"),
    "woman": ({"person"}, "singular", "female"),
    "someone": ({"person"}, "singular", None),
    "somebody": ({"person"}, "singular", None),
    "thing": ({"object", "concept", "event"}, "singular", None),
    "something": ({"object", "concept", "event"}, "singular", None),
    "anything": ({"object", "concept", "event"}, "singular", None),
}


def reference_spec(text: str, head: str | None = None) -> dict[str, Any] | None:
    normalized = text.strip().lower()
    key = (head or normalized).strip().lower()
    if normalized in FIRST_SECOND_PERSON or key in FIRST_SECOND_PERSON:
        return {
            "expected_types": ["person"],
            "number": "unknown",
            "descriptor": None,
            "resolvable": False,
        }
    values = REFERENCE_SPECS.get(normalized) or REFERENCE_SPECS.get(key)
    if values is None:
        values = GENERIC_REFERENCE_SPECS.get(key)
    if values is None:
        return None
    expected_types, number, descriptor = values
    return {
        "expected_types": sorted(expected_types),
        "number": number,
        "descriptor": descriptor,
        "resolvable": True,
    }


def is_indefinite_introduction(text: str, head: str | None = None) -> bool:
    normalized = " ".join(text.strip().lower().split())
    key = (head or normalized).strip().lower()
    if key not in GENERIC_REFERENCE_SPECS:
        return False
    return normalized.startswith(
        ("a ", "an ", "another ", "some ", "one ")
    ) or normalized in {"someone", "somebody", "something"}


def _role_weight(role: str) -> float:
    return {
        "subject": 1.0,
        "possessor": 0.85,
        "object": 0.75,
        "oblique": 0.55,
        "other": 0.40,
        "unknown": 0.35,
    }.get(role, 0.35)


class TextDiscourseMemory:
    def __init__(
        self,
        config: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ):
        self.config = config
        self.entities: dict[str, dict[str, Any]] = {}
        self.recent_mentions: list[dict[str, Any]] = []
        self.recent_events: list[dict[str, Any]] = []
        self.reference_counter = 0
        if payload:
            self.entities = {
                str(key): dict(value)
                for key, value in payload.get("entities", {}).items()
            }
            self.recent_mentions = list(payload.get("recent_mentions", []))
            self.recent_events = list(payload.get("recent_events", []))
            self.reference_counter = int(payload.get("reference_counter", 0))

    def new_reference_id(self) -> str:
        self.reference_counter += 1
        return f"R_{self.reference_counter:06d}"

    def observe_entity(
        self,
        *,
        text_id: str,
        entity_type: str,
        label: str,
        start: float,
        end: float,
        segment_id: str,
        segment_index: int,
        role: str,
        number: str = "unknown",
        descriptor: str | None = None,
        generic: bool = False,
        confidence: float = 0.0,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        state = self.entities.setdefault(
            text_id,
            {
                "text_id": text_id,
                "entity_type": entity_type,
                "canonical_label": label,
                "aliases": [],
                "first_seen": start,
                "last_seen": end,
                "last_segment_id": segment_id,
                "last_segment_index": segment_index,
                "mention_count": 0,
                "role_counts": {},
                "last_role": role,
                "number": number,
                "descriptors": [],
                "generic": bool(generic),
                "confidence": float(confidence),
                "recent_predicates": [],
            },
        )
        if label and label not in state["aliases"] and not reference_text:
            state["aliases"].append(label)
        state["first_seen"] = min(float(state["first_seen"]), float(start))
        state["last_seen"] = max(float(state["last_seen"]), float(end))
        state["last_segment_id"] = segment_id
        state["last_segment_index"] = int(segment_index)
        state["mention_count"] = int(state.get("mention_count", 0)) + 1
        state["last_role"] = role
        roles = Counter(state.get("role_counts", {}))
        roles[role] += 1
        state["role_counts"] = dict(roles)
        if state.get("number", "unknown") == "unknown" and number != "unknown":
            state["number"] = number
        if descriptor and descriptor not in state["descriptors"]:
            state["descriptors"].append(descriptor)
        state["generic"] = bool(state.get("generic", False) and generic)
        state["confidence"] = max(
            float(state.get("confidence", 0.0)),
            float(confidence),
        )
        mention = {
            "text_id": text_id,
            "entity_type": entity_type,
            "label": label,
            "reference_text": reference_text,
            "start": float(start),
            "end": float(end),
            "segment_id": segment_id,
            "segment_index": int(segment_index),
            "role": role,
        }
        self.recent_mentions.append(mention)
        limit = int(self.config.get("text_memory_short_term_mentions", 32))
        self.recent_mentions = self.recent_mentions[-max(limit, 1) :]
        return state

    def _candidate_score(
        self,
        state: dict[str, Any],
        reference: dict[str, Any],
    ) -> tuple[float, dict[str, float]] | None:
        expected_types = set(reference.get("expected_types", []))
        entity_type = str(state.get("entity_type", "unknown"))
        if expected_types and entity_type not in expected_types:
            return None

        descriptor = reference.get("descriptor")
        descriptors = set(state.get("descriptors", []))
        if (
            descriptor in {"male", "female"}
            and descriptors & {"male", "female"}
            and descriptor not in descriptors
        ):
            return None

        reference_number = str(reference.get("number", "unknown"))
        entity_number = str(state.get("number", "unknown"))
        if (
            reference_number != "unknown"
            and entity_number != "unknown"
            and reference_number != entity_number
        ):
            return None

        time_gap = max(
            0.0,
            float(reference["start"]) - float(state.get("last_seen", 0.0)),
        )
        recency_seconds = float(
            self.config.get("text_memory_recency_seconds", 120.0)
        )
        recency = math.exp(-time_gap / max(recency_seconds, 1e-6))
        segment_gap = max(
            0,
            int(reference["segment_index"])
            - int(state.get("last_segment_index", 0)),
        )
        segment_continuity = math.exp(-segment_gap / 3.0)
        mention_count = int(state.get("mention_count", 0))
        salience = min(
            1.0,
            math.log1p(mention_count) / math.log(8.0),
        )
        reference_role = str(reference.get("role", "unknown"))
        previous_role = str(state.get("last_role", "unknown"))
        if reference_role == previous_role:
            role_continuity = 1.0
        elif reference_role == "subject" and previous_role == "subject":
            role_continuity = 1.0
        else:
            role_continuity = (
                _role_weight(previous_role) + _role_weight(reference_role)
            ) / 2.0
        type_compatibility = 1.0
        descriptor_compatibility = (
            1.0 if descriptor and descriptor in descriptors else 0.5
        )
        score = (
            0.30 * recency
            + 0.15 * segment_continuity
            + 0.20 * salience
            + 0.15 * role_continuity
            + 0.10 * type_compatibility
            + 0.10 * descriptor_compatibility
        )
        return score, {
            "recency": recency,
            "segment_continuity": segment_continuity,
            "salience": salience,
            "role_continuity": role_continuity,
            "type_compatibility": type_compatibility,
            "descriptor_compatibility": descriptor_compatibility,
        }

    def resolve_reference(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        reference_id = self.new_reference_id()
        scored = []
        if reference.get("resolvable", True):
            for text_id, state in self.entities.items():
                result = self._candidate_score(state, reference)
                if result is None:
                    continue
                score, components = result
                scored.append(
                    {
                        "text_id": text_id,
                        "label": state.get("canonical_label", text_id),
                        "entity_type": state.get("entity_type", "unknown"),
                        "score": score,
                        "components": components,
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        threshold = float(
            self.config.get("text_reference_resolution_threshold", 0.66)
        )
        margin = float(self.config.get("text_reference_margin", 0.10))
        best = scored[0] if scored else None
        second_score = scored[1]["score"] if len(scored) > 1 else 0.0
        resolved = bool(
            best
            and best["score"] >= threshold
            and (
                len(scored) == 1
                or best["score"] - second_score >= margin
            )
        )
        return {
            "reference_id": reference_id,
            "text": reference["text"],
            "start": float(reference["start"]),
            "end": float(reference["end"]),
            "role": reference.get("role", "unknown"),
            "expected_types": list(reference.get("expected_types", [])),
            "number": reference.get("number", "unknown"),
            "descriptor": reference.get("descriptor"),
            "status": "resolved" if resolved else "unresolved",
            "resolved_text_id": best["text_id"] if resolved else None,
            "confidence": float(best["score"]) if resolved else 0.0,
            "candidates": scored[:3],
        }

    def add_event(self, event: dict[str, Any]) -> None:
        self.recent_events.append(dict(event))
        limit = int(self.config.get("text_memory_recent_events", 12))
        self.recent_events = self.recent_events[-max(limit, 1) :]
        predicate = str(event.get("predicate", "")).strip()
        if not predicate:
            return
        for text_id in event.get("entity_ids", []):
            state = self.entities.get(str(text_id))
            if state is None:
                continue
            predicates = list(state.get("recent_predicates", []))
            predicates.append(predicate)
            state["recent_predicates"] = predicates[-5:]

    def prompt_context(
        self,
        *,
        current_time: float,
    ) -> dict[str, Any]:
        recent_ids = []
        for mention in reversed(self.recent_mentions):
            text_id = str(mention["text_id"])
            if text_id not in recent_ids:
                recent_ids.append(text_id)
        short_entities = [
            self._compact_entity(self.entities[text_id], current_time)
            for text_id in recent_ids[:12]
            if text_id in self.entities
        ]
        ranked = sorted(
            self.entities.values(),
            key=lambda state: (
                self._memory_rank(state, current_time),
                float(state.get("last_seen", 0.0)),
            ),
            reverse=True,
        )
        long_limit = int(self.config.get("text_memory_long_term_entities", 40))
        return {
            "short_term": {
                "recent_mentions": self.recent_mentions[-12:],
                "recent_events": self.recent_events[-8:],
                "active_entities": short_entities,
            },
            "long_term": {
                "entities": [
                    self._compact_entity(state, current_time)
                    for state in ranked[:max(long_limit, 1)]
                ]
            },
        }

    def _memory_rank(
        self,
        state: dict[str, Any],
        current_time: float,
    ) -> float:
        gap = max(0.0, current_time - float(state.get("last_seen", 0.0)))
        recency = math.exp(-gap / 300.0)
        frequency = min(
            1.0,
            math.log1p(int(state.get("mention_count", 0))) / math.log(12.0),
        )
        subject_ratio = (
            int(state.get("role_counts", {}).get("subject", 0))
            / max(int(state.get("mention_count", 0)), 1)
        )
        return 0.45 * recency + 0.35 * frequency + 0.20 * subject_ratio

    def _compact_entity(
        self,
        state: dict[str, Any],
        current_time: float,
    ) -> dict[str, Any]:
        return {
            "text_id": state["text_id"],
            "entity_type": state["entity_type"],
            "canonical_label": state["canonical_label"],
            "aliases": list(state.get("aliases", []))[-6:],
            "first_seen": state.get("first_seen"),
            "last_seen": state.get("last_seen"),
            "mention_count": state.get("mention_count", 0),
            "last_role": state.get("last_role", "unknown"),
            "descriptors": list(state.get("descriptors", [])),
            "recent_predicates": list(state.get("recent_predicates", [])),
            "memory_rank": self._memory_rank(state, current_time),
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "recent_mentions": self.recent_mentions,
            "recent_events": self.recent_events,
            "reference_counter": self.reference_counter,
        }
