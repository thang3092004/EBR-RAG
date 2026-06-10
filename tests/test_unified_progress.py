import json

from scripts.show_unified_progress import (
    _duration,
    _load_workdir_rows,
    _render_table,
)


def test_progress_table_shows_percent_eta_and_details():
    rendered = _render_table(
        [
            {
                "video_id": "video-a",
                "stage": "alignment_caption",
                "completed": 3,
                "total": 12,
                "unit": "seg",
                "elapsed_seconds": 65,
                "eta_seconds": 195,
                "details": {"edges": 8},
            }
        ]
    )

    assert "video-a" in rendered
    assert "25.0%" in rendered
    assert "01:05" in rendered
    assert "03:15" in rendered
    assert "edges=8" in rendered


def test_duration_formats_hours():
    assert _duration(3661) == "1:01:01"


def test_recursive_progress_labels_collection_and_profile(tmp_path):
    pipeline = tmp_path / "collection" / "full_framework" / "pipeline_v2"
    video = pipeline / "video-a"
    video.mkdir(parents=True)
    (video / "progress.json").write_text(
        json.dumps({"video_id": "video-a", "stage": "asr"}),
        encoding="utf-8",
    )

    rows = _load_workdir_rows(tmp_path, recursive=True)

    assert rows[0]["video_id"] == "collection/full_framework/video-a"
