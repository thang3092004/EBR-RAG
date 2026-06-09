from argparse import Namespace

import pytest

from scripts.ingest_unified_v2 import _videos
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
