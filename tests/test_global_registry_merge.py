from videorag._unified_graph.registry import EntityRegistry
from videorag._unified_graph.schema import ProvenanceRecord
from videorag.pipeline.stage_runner import atomic_write_json, read_json
from videorag.pipeline.unified_ingest import _merge_global_registry


def _entity_payload(video_id, label, entity_id):
    registry = EntityRegistry()
    entity_id = registry.ensure_entity(
        "concept",
        label,
        entity_id=entity_id,
        source="transcript",
    )
    registry.add_alias(
        "T_CONCEPT_001",
        entity_id,
        source="transcript",
        label=label,
    )
    registry.add_provenance(
        entity_id,
        ProvenanceRecord(
            source="transcript",
            video_id=video_id,
            segment_id="SEG_00000",
            start=0.0,
            end=1.0,
            text=label,
        ),
    )
    return registry.to_dict()


def test_global_registry_replaces_only_the_rerun_video(tmp_path):
    root = tmp_path / "pipeline_v2"
    root.mkdir()
    atomic_write_json(root / "global_registry.json", {})

    _merge_global_registry(
        tmp_path,
        "video_a",
        _entity_payload("video_a", "A", "CONCEPT_001"),
    )
    _merge_global_registry(
        tmp_path,
        "video_b",
        _entity_payload("video_b", "B", "CONCEPT_002"),
    )
    before = read_json(root / "global_registry.json")
    assert len(before["entities"]) == 2

    removed = _merge_global_registry(
        tmp_path,
        "video_a",
        _entity_payload("video_a", "A rerun", "CONCEPT_003"),
    )
    after = read_json(root / "global_registry.json")

    assert removed
    assert len(after["entities"]) == 2
    assert any(
        item["canonical_name"] == "B"
        for item in after["entities"]
    )
