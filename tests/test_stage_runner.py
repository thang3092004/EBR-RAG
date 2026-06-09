from pathlib import Path

from videorag.pipeline.stage_runner import (
    PIPELINE_VERSION,
    StageDefinition,
    StageRunner,
    atomic_write_json,
)
from videorag.pipeline.unified_stages import UNIFIED_STAGE_DEFINITIONS


def _stages():
    return [
        StageDefinition("first", config_keys=("x",)),
        StageDefinition("second", dependencies=("first",)),
    ]


def test_stage_runner_resumes_and_invalidates_downstream(tmp_path):
    calls = []
    runner = StageRunner(tmp_path, "video", {"x": 1}, _stages())
    with runner.run_scope():
        runner.run_stage("first", lambda context: calls.append("first") or {"ok": 1})
        runner.run_stage("second", lambda context: calls.append("second") or {"ok": 1})
    assert calls == ["first", "second"]

    calls.clear()
    resumed = StageRunner(tmp_path, "video", {"x": 1}, _stages())
    with resumed.run_scope():
        resumed.run_stage("first", lambda context: calls.append("first"))
        resumed.run_stage("second", lambda context: calls.append("second"))
    assert calls == []

    changed = StageRunner(tmp_path, "video", {"x": 2}, _stages())
    with changed.run_scope():
        changed.run_stage("first", lambda context: calls.append("first"))
        changed.run_stage("second", lambda context: calls.append("second"))
    assert calls == ["first", "second"]


def test_restart_stage_removes_stage_outputs(tmp_path):
    runner = StageRunner(tmp_path, "video", {"x": 1}, _stages())
    with runner.run_scope():
        runner.run_stage(
            "first",
            lambda context: context.write_json("artifact.json", {"old": True}) or {},
        )
        runner.run_stage("second", lambda context: {})

    restarted = StageRunner(
        tmp_path,
        "video",
        {"x": 1},
        _stages(),
        restart_stage="first",
    )
    assert not (restarted.video_dir / "first" / "artifact.json").exists()
    restarted.close()


def test_input_fingerprint_invalidates_probe_and_descendants(tmp_path):
    stages = [
        StageDefinition("probe", config_keys=("input_size_bytes",)),
        StageDefinition("next", dependencies=("probe",)),
    ]
    calls = []
    first = StageRunner(tmp_path, "video", {"input_size_bytes": 10}, stages)
    with first.run_scope():
        first.run_stage("probe", lambda context: calls.append("probe"))
        first.run_stage("next", lambda context: calls.append("next"))

    calls.clear()
    changed = StageRunner(tmp_path, "video", {"input_size_bytes": 11}, stages)
    with changed.run_scope():
        changed.run_stage("probe", lambda context: calls.append("probe"))
        changed.run_stage("next", lambda context: calls.append("next"))

    assert calls == ["probe", "next"]


def test_unified_stage_graph_is_entity_only():
    stage_names = {
        definition.name for definition in UNIFIED_STAGE_DEFINITIONS
    }

    assert "frame_selection" in stage_names
    assert "text_entities" in stage_names
    assert "alignment_caption" in stage_names
    assert "modality_profile" not in stage_names
    assert "deep_processing" not in stage_names
    assert "speaker_linking" not in stage_names


def test_ablation_flags_are_part_of_stage_fingerprints():
    by_name = {
        definition.name: definition
        for definition in UNIFIED_STAGE_DEFINITIONS
    }

    assert "segmentation_strategy" in by_name["segmentation"].config_keys
    assert (
        "disable_visual_identity_linking"
        in by_name["tracking_base"].config_keys
    )
    assert (
        "disable_transcript_memory"
        in by_name["text_entities"].config_keys
    )
    assert (
        "disable_crossmodal_alignment"
        in by_name["alignment_caption"].config_keys
    )


def test_version_migration_removes_obsolete_stage_outputs(tmp_path):
    video_dir = tmp_path / "pipeline_v2" / "video"
    for stage_name in (
        "modality_profile",
        "deep_processing",
        "speaker_linking",
    ):
        stage_dir = video_dir / stage_name
        stage_dir.mkdir(parents=True)
        (stage_dir / "artifact.json").write_text("{}", encoding="utf-8")
    atomic_write_json(
        video_dir / "manifest.json",
        {
            "pipeline_version": "unified-graph-v2",
            "video_id": "video",
            "stages": {},
        },
    )

    runner = StageRunner(tmp_path, "video", {}, _stages())

    assert runner.manifest["pipeline_version"] == PIPELINE_VERSION
    assert not (video_dir / "modality_profile").exists()
    assert not (video_dir / "deep_processing").exists()
    assert not (video_dir / "speaker_linking").exists()
    runner.close()
