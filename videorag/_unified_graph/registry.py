from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schema import (
    GLOBAL_PREFIXES,
    EntityAlias,
    EntityNode,
    ProvenanceRecord,
    normalize_entity_type,
)


PROVISIONAL_PATTERN = re.compile(r"^(?:V_|T_)[A-Z_]+_\d+$")


def normalize_alias(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^(?:the|a|an|mr\.?|mrs\.?|ms\.?|dr\.?)\s+", "", value)
    value = re.sub(r"[^\w\s-]", " ", value)
    return " ".join(value.split())


class EntityRegistry:
    def __init__(self, payload: dict[str, Any] | None = None):
        self.entities: dict[str, EntityNode] = {}
        self.alias_to_global: dict[str, str] = {}
        self.name_to_global: dict[str, str] = {}
        self.counters: dict[str, int] = defaultdict(int)
        if payload:
            self._load(payload)

    def _load(self, payload: dict[str, Any]) -> None:
        self.counters.update(
            {str(key): int(value) for key, value in payload.get("counters", {}).items()}
        )
        for raw in payload.get("entities", []):
            aliases = [EntityAlias(**alias) for alias in raw.get("aliases", [])]
            provenance = [
                ProvenanceRecord(**item) for item in raw.get("provenance", [])
            ]
            node = EntityNode(
                entity_id=raw["entity_id"],
                entity_type=raw["entity_type"],
                canonical_name=raw.get("canonical_name", raw["entity_id"]),
                aliases=aliases,
                sources=list(raw.get("sources", [])),
                first_seen=raw.get("first_seen"),
                last_seen=raw.get("last_seen"),
                confidence=float(raw.get("confidence", 0.0)),
                attributes=dict(raw.get("attributes", {})),
                provenance=provenance,
            )
            self.entities[node.entity_id] = node
            self.name_to_global[normalize_alias(node.canonical_name)] = node.entity_id
            for alias in aliases:
                self.alias_to_global[alias.alias_id] = node.entity_id
                if alias.label:
                    self.name_to_global.setdefault(
                        normalize_alias(alias.label), node.entity_id
                    )

    def allocate(self, entity_type: str) -> str:
        normalized = normalize_entity_type(entity_type)
        self.counters[normalized] += 1
        prefix = GLOBAL_PREFIXES.get(normalized, "ENTITY")
        return f"{prefix}_{self.counters[normalized]:03d}"

    def ensure_entity(
        self,
        entity_type: str,
        canonical_name: str,
        *,
        entity_id: str | None = None,
        source: str | None = None,
        confidence: float = 0.0,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        normalized_type = normalize_entity_type(entity_type)
        if entity_id is None:
            entity_id = self.allocate(normalized_type)
        if PROVISIONAL_PATTERN.match(entity_id):
            raise ValueError(f"Provisional ID cannot be a global entity ID: {entity_id}")
        if entity_id not in self.entities:
            prefix = GLOBAL_PREFIXES.get(normalized_type, "ENTITY")
            match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", entity_id)
            if match:
                self.counters[normalized_type] = max(
                    self.counters[normalized_type],
                    int(match.group(1)),
                )
            self.entities[entity_id] = EntityNode(
                entity_id=entity_id,
                entity_type=normalized_type,
                canonical_name=canonical_name or entity_id,
                sources=[source] if source else [],
                confidence=float(confidence),
                attributes=dict(attributes or {}),
            )
        else:
            node = self.entities[entity_id]
            if source and source not in node.sources:
                node.sources.append(source)
            node.confidence = max(node.confidence, float(confidence))
            node.attributes.update(attributes or {})
            if (
                canonical_name
                and node.canonical_name == node.entity_id
                and canonical_name != entity_id
            ):
                node.canonical_name = canonical_name
        if canonical_name:
            self.name_to_global.setdefault(normalize_alias(canonical_name), entity_id)
        return entity_id

    def add_alias(
        self,
        alias_id: str,
        global_id: str,
        *,
        source: str,
        label: str = "",
        confidence: float = 0.0,
        segment_id: str | None = None,
    ) -> None:
        if global_id not in self.entities:
            raise KeyError(f"Unknown global entity: {global_id}")
        existing = self.alias_to_global.get(alias_id)
        if existing and existing != global_id:
            raise ValueError(
                f"Alias {alias_id} already belongs to {existing}, cannot assign to {global_id}"
            )
        self.alias_to_global[alias_id] = global_id
        node = self.entities[global_id]
        for alias in node.aliases:
            if alias.alias_id == alias_id:
                alias.confidence = max(alias.confidence, float(confidence))
                if segment_id and segment_id not in alias.segment_ids:
                    alias.segment_ids.append(segment_id)
                return
        node.aliases.append(
            EntityAlias(
                alias_id=alias_id,
                source=source,
                label=label,
                confidence=float(confidence),
                segment_ids=[segment_id] if segment_id else [],
            )
        )
        if source not in node.sources:
            node.sources.append(source)
        if label:
            normalized_label = normalize_alias(label)
            existing_name = self.name_to_global.get(normalized_label)
            if existing_name in (None, global_id):
                self.name_to_global[normalized_label] = global_id

    def resolve(self, alias_or_name: str) -> str | None:
        if alias_or_name in self.entities:
            return alias_or_name
        if alias_or_name in self.alias_to_global:
            return self.alias_to_global[alias_or_name]
        return self.name_to_global.get(normalize_alias(alias_or_name))

    def add_provenance(self, global_id: str, provenance: ProvenanceRecord) -> None:
        node = self.entities[global_id]
        node.provenance.append(provenance)
        if node.first_seen is None or provenance.start < node.first_seen:
            node.first_seen = provenance.start
        if node.last_seen is None or provenance.end > node.last_seen:
            node.last_seen = provenance.end
        if provenance.source not in node.sources:
            node.sources.append(provenance.source)

    def merge_alias(self, alias_id: str, global_id: str, **kwargs: Any) -> str:
        self.add_alias(alias_id, global_id, **kwargs)
        return global_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "alias_to_global": dict(self.alias_to_global),
            "entities": [
                self.entities[key].to_dict() for key in sorted(self.entities)
            ],
        }
