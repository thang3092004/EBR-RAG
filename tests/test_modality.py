from videorag._videoutil.modality import profile_modalities


def test_modality_profile_is_segment_specific():
    segments = [
        {"segment_id": "SEG_00000", "start": 0.0, "end": 10.0},
        {"segment_id": "SEG_00001", "start": 10.0, "end": 20.0},
    ]
    words = [
        {
            "start": 10.0 + index * 0.3,
            "end": 10.2 + index * 0.3,
            "text": f"technical{index}",
            "probability": 0.95,
        }
        for index in range(25)
    ]
    shots = {
        "boundaries": [{"time": 2.0}, {"time": 4.0}, {"time": 6.0}],
        "samples": [
            {"time": 2.0, "motion_score": 0.8},
            {"time": 4.0, "motion_score": 0.9},
            {"time": 12.0, "motion_score": 0.01},
        ],
    }
    observations = [
        {
            "time": float(index),
            "tracklet_id": f"V_OBJECT_{index}",
        }
        for index in range(1, 8)
    ]
    profiles = profile_modalities(segments, words, shots, observations, {})
    assert profiles[0]["visual_score"] > profiles[1]["visual_score"]
    assert profiles[1]["speech_score"] > profiles[0]["speech_score"]
    assert profiles[0]["mode"] != profiles[1]["mode"]

