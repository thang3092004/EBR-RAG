from videorag._entity_anchor.tracker_v2 import link_visual_tracklets
from videorag._unified_graph.registry import EntityRegistry


def _tracklet(tracklet_id, start, end, x):
    observation = {
        "segment_id": "SEG_00000",
        "time": start,
        "bbox": [x, 0.0, x + 10.0, 10.0],
        "confidence": 0.9,
        "entity_type": "person",
        "label": "person",
    }
    return {
        "tracklet_id": tracklet_id,
        "entity_type": "person",
        "label": "person",
        "start": start,
        "end": end,
        "first_bbox": observation["bbox"],
        "last_bbox": observation["bbox"],
        "clip_embedding": [1.0, 0.0],
        "appearance": [1.0, 0.0],
        "observations": [observation],
    }


def test_visual_identity_ablation_keeps_tracklets_separate():
    tracklets = [
        _tracklet("track-a", 0.0, 1.0, 0.0),
        _tracklet("track-b", 2.0, 3.0, 0.0),
    ]
    linked = link_visual_tracklets(
        tracklets,
        EntityRegistry(),
        "video",
        {"visual_merge_threshold": 0.85},
    )
    separated = link_visual_tracklets(
        [
            _tracklet("track-a", 0.0, 1.0, 0.0),
            _tracklet("track-b", 2.0, 3.0, 0.0),
        ],
        EntityRegistry(),
        "video",
        {
            "visual_merge_threshold": 0.85,
            "disable_visual_identity_linking": True,
        },
    )

    assert len(linked) == 1
    assert len(separated) == 2
