from videorag._videoutil.smart_segment import fixed_segment, smart_segment


def test_smart_segmentation_uses_safe_boundaries():
    words = [
        {"start": 0.0, "end": 4.0, "text": "Introduction."},
        {"start": 18.0, "end": 23.5, "text": "A long sentence ends here."},
        {"start": 24.2, "end": 30.0, "text": "The next thought."},
        {"start": 47.0, "end": 51.0, "text": "Conclusion."},
    ]
    segments = smart_segment(
        60.0,
        words,
        [{"time": 20.0, "score": 1.0}, {"time": 40.0, "score": 1.0}],
        {
            "segment_target_seconds": 24.0,
            "segment_min_seconds": 8.0,
            "segment_max_seconds": 45.0,
        },
    )
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 60.0
    assert all(segment["duration"] <= 45.0 for segment in segments)
    for segment in segments[:-1]:
        boundary = segment["end"]
        assert not any(word["start"] < boundary < word["end"] for word in words)


def test_short_tail_is_kept_without_exceeding_maximum():
    segments = smart_segment(49.0, [], [], {})
    assert segments[-1]["end"] == 49.0
    assert max(segment["duration"] for segment in segments) <= 45.0


def test_fixed_segmentation_uses_non_overlapping_30_second_windows():
    segments = fixed_segment(65.0, length=30.0, context=1.5)

    assert [(item["start"], item["end"]) for item in segments] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 65.0),
    ]
    assert segments[1]["context_start"] == 28.5
    assert segments[1]["context_end"] == 61.5
