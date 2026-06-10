from pathlib import Path

from reproduce.run_ablation_matrix import (
    REUSABLE_FULL_STAGES,
    _seed_profile_from_full,
)
from videorag.pipeline.stage_runner import (
    PIPELINE_VERSION,
    atomic_write_json,
    read_json,
)


def test_no_transcript_memory_reuses_only_safe_full_stages(tmp_path):
    full = tmp_path / "full"
    target = tmp_path / "target"
    video_id = "video-a"
    source_video = full / "pipeline_v2" / video_id
    stages = REUSABLE_FULL_STAGES["no_transcript_memory"]
    states = {}
    for stage in stages:
        stage_dir = source_video / stage
        stage_dir.mkdir(parents=True)
        (stage_dir / "artifact.json").write_text(
            f'{{"stage": "{stage}"}}',
            encoding="utf-8",
        )
        states[stage] = {
            "status": "done",
            "config_hash": f"hash-{stage}",
        }
    atomic_write_json(
        source_video / "manifest.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "video_id": video_id,
            "stages": states,
        },
    )

    report = _seed_profile_from_full(
        full,
        target,
        [str(Path("videos") / f"{video_id}.webm")],
        "no_transcript_memory",
    )

    assert report["seeded_videos"] == 1
    target_video = target / "pipeline_v2" / video_id
    assert (target_video / "frame_selection" / "artifact.json").exists()
    assert not (target_video / "text_entities").exists()
    manifest = read_json(target_video / "manifest.json")
    assert set(manifest["stages"]) == set(stages)
    assert manifest["seeded_from_profile"] == "full_framework"


def test_profile_seed_rejects_stale_pipeline_version(tmp_path):
    full = tmp_path / "full"
    target = tmp_path / "target"
    video_id = "video-a"
    source_video = full / "pipeline_v2" / video_id
    stages = REUSABLE_FULL_STAGES["no_crossmodal_alignment"]
    states = {}
    for stage in stages:
        (source_video / stage).mkdir(parents=True)
        states[stage] = {"status": "done", "config_hash": f"hash-{stage}"}
    atomic_write_json(
        source_video / "manifest.json",
        {
            "pipeline_version": "stale-version",
            "video_id": video_id,
            "stages": states,
        },
    )

    report = _seed_profile_from_full(
        full,
        target,
        [str(Path("videos") / f"{video_id}.webm")],
        "no_crossmodal_alignment",
    )

    assert report["seeded_videos"] == 0
    assert report["unavailable_videos"] == [video_id]
    assert not (target / "pipeline_v2" / video_id / "manifest.json").exists()


def test_strict_profile_resumes_its_own_completed_stages(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import reproduce.run_ablation_matrix as ablation

    collection = {"slug": "collection"}
    workdir = tmp_path / "collection" / "full_framework"
    workdir.mkdir(parents=True)
    (workdir / "existing-artifact.json").write_text("{}", encoding="utf-8")
    calls = {}

    class FakeVideoRAG:
        def insert_video(self, videos, *, resume, restart_stage, force):
            calls.update(
                videos=videos,
                resume=resume,
                restart_stage=restart_stage,
                force=force,
            )
            return [{"status": "complete"}]

    monkeypatch.setattr(
        ablation,
        "_new_vrag",
        lambda *_args, **_kwargs: FakeVideoRAG(),
    )
    monkeypatch.setattr(
        ablation,
        "_collect_artifact_report",
        lambda *_args, **_kwargs: {},
    )
    args = SimpleNamespace(
        work_root=tmp_path,
        strict_pipeline=True,
        reuse_full_artifacts=False,
        force=False,
        no_resume=False,
        restart_stage=None,
    )

    result = ablation._ingest_profile(
        collection,
        ["video.mp4"],
        "full_framework",
        args,
    )

    assert result["status"] == "complete"
    assert calls["resume"] is True
