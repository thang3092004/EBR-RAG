import pytest

from videorag._unified_graph.registry import EntityRegistry
from videorag._unified_graph.schema import ProvenanceRecord


def test_registry_keeps_aliases_as_metadata():
    registry = EntityRegistry()
    person = registry.ensure_entity("person", "heroine", source="visual")
    registry.add_alias(
        "V_PERSON_001",
        person,
        source="visual",
        label="woman",
    )
    registry.add_alias(
        "T_PERSON_002",
        person,
        source="transcript",
        label="John",
    )
    registry.add_provenance(
        person,
        ProvenanceRecord(
            source="visual",
            video_id="v",
            segment_id="SEG_00001",
            start=1.0,
            end=2.0,
        ),
    )
    assert registry.resolve("V_PERSON_001") == person
    assert registry.resolve("T_PERSON_002") == person
    assert list(registry.entities) == [person]
    assert registry.entities[person].first_seen == 1.0


def test_alias_cannot_be_assigned_to_two_entities():
    registry = EntityRegistry()
    first = registry.ensure_entity("person", "first")
    second = registry.ensure_entity("person", "second")
    registry.add_alias("T_PERSON_001", first, source="transcript")
    with pytest.raises(ValueError):
        registry.add_alias("T_PERSON_001", second, source="transcript")

