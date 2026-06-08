from videorag._unified_graph.alignment import _validate_alignment


def test_alignment_rejects_merge_outside_candidate_whitelist():
    result = _validate_alignment(
        {
            "caption": "test",
            "entity_merges": [
                {"visual_id": "PERSON_001", "text_id": "T_PERSON_001"},
                {"visual_id": "PERSON_999", "text_id": "T_PERSON_001"},
            ],
            "edges": [],
        },
        [
            {
                "visual_id": "PERSON_001",
                "text_id": "T_PERSON_001",
            }
        ],
    )
    assert result["entity_merges"] == [
        {
            "visual_id": "PERSON_001",
            "text_id": "T_PERSON_001",
            "reason": "",
        }
    ]


def test_alignment_keeps_only_supported_modalities():
    result = _validate_alignment(
        {
            "caption": "test",
            "entity_merges": [],
            "edges": [
                {
                    "source": "PERSON_001",
                    "predicate": "speaks",
                    "target": "CLAIM_001",
                    "confidence": 0.8,
                    "modalities": ["transcript", "invented"],
                }
            ],
        },
        [],
    )

    assert result["edges"][0]["modalities"] == ["transcript"]
