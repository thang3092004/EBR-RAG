from pathlib import Path

from videorag.pipeline.stage_runner import StageDefinition, StageRunner


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
