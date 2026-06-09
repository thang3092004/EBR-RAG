from videorag._unified_graph.alignment import align_all_segments
from videorag._unified_graph.registry import EntityRegistry


class _Aligner:
    def align(self, prompt, frame_paths):
        if "Analyze only the supplied video frames" in prompt:
            return {
                "visual_caption": "PERSON_001 is visible.",
                "entity_descriptions": [
                    {
                        "entity_id": "PERSON_001",
                        "description": "Jeff Bezos is seated.",
                    }
                ],
                "visual_edges": [],
            }
        return {
            "caption": "PERSON_001 is speaking.",
            "entity_merges": [
                {
                    "visual_id": "PERSON_001",
                    "text_id": "T_PERSON_001",
                    "reason": "Same name.",
                }
            ],
            "edges": [],
        }

    def close(self):
        pass


def test_no_crossmodal_alignment_rejects_all_visual_text_merges(tmp_path):
    registry = EntityRegistry()
    registry.ensure_entity(
        "person",
        "Jeff Bezos",
        entity_id="PERSON_001",
        source="visual",
    )
    result = align_all_segments(
        "video",
        [
            {
                "segment_id": "SEG_00000",
                "index": 0,
                "start": 0.0,
                "end": 10.0,
                "storage_id": "video_0",
            }
        ],
        {"SEG_00000": {"frames": [{"path": "unused.jpg"}]}},
        {
            "SEG_00000": {
                "mentions": [
                    {
                        "text_id": "T_PERSON_001",
                        "entity_type": "person",
                        "label": "Jeff Bezos",
                        "confidence": 0.9,
                        "start": 0.0,
                        "end": 1.0,
                    }
                ],
                "references": [],
                "memory_context": {},
                "rewritten_transcript": "[T_PERSON_001] speaks.",
            }
        },
        [
            {
                "segment_id": "SEG_00000",
                "entity_id": "PERSON_001",
                "entity_type": "person",
                "label": "person",
                "time": 1.0,
                "confidence": 0.9,
            }
        ],
        registry,
        {"disable_crossmodal_alignment": True},
        tmp_path,
        aligner=_Aligner(),
    )

    segment = result["segments"]["SEG_00000"]
    assert segment["accepted_merges"] == []
    assert segment["correspondence"]["status"] == "disabled_by_ablation"
    person_nodes = [
        entity
        for entity in result["registry"]["entities"]
        if entity["entity_type"] == "person"
    ]
    assert len(person_nodes) == 2
