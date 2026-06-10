from argparse import Namespace

import pytest

from scripts.ingest_unified_v2 import _require_empty_workdir, _videos
from videorag._entity_anchor import text_entities
from videorag._videoutil.smart_segment import smart_segment
from videorag._videoutil.media_probe import _fraction
from videorag.pipeline.unified_ingest import UnifiedIngestPipeline


def test_video_inputs_reject_duplicate_workspace_ids(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = left / "same.webm"
    second = right / "same.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(ValueError, match="filename collision"):
        _videos(
            Namespace(
                video=[str(first), str(second)],
                video_dir=None,
            )
        )


def test_strict_workdir_must_be_empty(tmp_path):
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="no artifact can be reused"):
        _require_empty_workdir(str(tmp_path))


def test_media_probe_fraction_parses_ffprobe_values():
    assert _fraction("25/1") == 25.0
    assert _fraction("10794.901") == 10794.901


def test_collection_can_continue_after_one_video_failure(tmp_path):
    class _Vrag:
        working_dir = str(tmp_path)

    pipeline = UnifiedIngestPipeline(_Vrag())
    pipeline.config["pipeline_continue_on_error"] = True
    calls = []

    def run_video(path):
        calls.append(path)
        if path == "bad.webm":
            raise RuntimeError("broken video")
        return {"video_id": path, "status": "complete"}

    pipeline.run_video = run_video
    reports = pipeline.run(["bad.webm", "good.webm"])

    assert calls == ["bad.webm", "good.webm"]
    assert reports[0]["status"] == "failed"
    assert reports[1]["status"] == "complete"


def test_strict_text_entities_reject_model_fallback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "spacy", None)

    with pytest.raises(RuntimeError, match="requires spaCy"):
        text_entities._load_nlp({"pipeline_strict": True})


def test_strict_segmentation_rejects_fixed_window_fallback(monkeypatch):
    monkeypatch.setattr(
        "videorag._videoutil.smart_segment.build_boundary_candidates",
        lambda *args, **kwargs: [
            type(
                "Boundary",
                (),
                {"time": 0.0, "score": 0.0, "speech_split": 0.0},
            )(),
            type(
                "Boundary",
                (),
                {"time": 100.0, "score": 0.0, "speech_split": 0.0},
            )(),
        ],
    )

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        smart_segment(
            100.0,
            [],
            [],
            {
                "segment_min_seconds": 8.0,
                "segment_target_seconds": 24.0,
                "segment_max_seconds": 45.0,
                "pipeline_strict": True,
            },
        )
