import numpy as np

from videorag._unified_graph.correspondence import gate_merge_candidates


def _encoder(mapping):
    def encode(texts):
        return np.asarray([mapping[text] for text in texts], dtype=np.float32)

    return encode


def test_gate_accepts_mutual_distinct_correspondence():
    candidates = [
        {"visual_id": "PERSON_001", "text_id": "T_PERSON_001"},
        {"visual_id": "PERSON_002", "text_id": "T_PERSON_001"},
    ]
    facts = [
        {
            "fact_id": "VF_001",
            "text": "PERSON_001 holds a rocket",
            "entity_ids": ["PERSON_001"],
        },
        {
            "fact_id": "VF_002",
            "text": "PERSON_002 sits at a desk",
            "entity_ids": ["PERSON_002"],
        },
    ]
    propositions = [
        {
            "proposition_id": "TP_001",
            "text": "T_PERSON_001 holds the rocket",
            "entity_ids": ["T_PERSON_001"],
        }
    ]
    encode = _encoder(
        {
            "entity holds a rocket": [1.0, 0.0],
            "entity sits at a desk": [0.0, 1.0],
            "entity holds the rocket": [0.99, 0.01],
        }
    )

    accepted, report = gate_merge_candidates(
        candidates,
        facts,
        propositions,
        encode,
        threshold=0.5,
        margin=0.1,
    )

    assert [(item["visual_id"], item["text_id"]) for item in accepted] == [
        ("PERSON_001", "T_PERSON_001")
    ]
    assert report["status"] == "matched"


def test_gate_rejects_ambiguous_visual_candidates():
    candidates = [
        {"visual_id": "PERSON_001", "text_id": "T_PERSON_001"},
        {"visual_id": "PERSON_002", "text_id": "T_PERSON_001"},
    ]
    facts = [
        {
            "fact_id": "VF_001",
            "text": "PERSON_001 is seated",
            "entity_ids": ["PERSON_001"],
        },
        {
            "fact_id": "VF_002",
            "text": "PERSON_002 is seated",
            "entity_ids": ["PERSON_002"],
        },
    ]
    propositions = [
        {
            "proposition_id": "TP_001",
            "text": "T_PERSON_001 is seated",
            "entity_ids": ["T_PERSON_001"],
        }
    ]

    accepted, report = gate_merge_candidates(
        candidates,
        facts,
        propositions,
        lambda texts: np.asarray([[1.0, 0.0]] * len(texts)),
        threshold=0.5,
        margin=0.1,
    )

    assert accepted == []
    assert report["status"] == "independent"


def test_gate_rejects_unrelated_modal_facts():
    candidates = [
        {"visual_id": "PERSON_001", "text_id": "T_PERSON_001"},
    ]
    facts = [
        {
            "fact_id": "VF_001",
            "text": "PERSON_001 sits in a studio",
            "entity_ids": ["PERSON_001"],
        }
    ]
    propositions = [
        {
            "proposition_id": "TP_001",
            "text": "T_PERSON_001 discusses orbital mechanics",
            "entity_ids": ["T_PERSON_001"],
        }
    ]
    encode = _encoder(
        {
            "entity sits in a studio": [1.0, 0.0],
            "entity discusses orbital mechanics": [0.0, 1.0],
        }
    )

    accepted, report = gate_merge_candidates(
        candidates,
        facts,
        propositions,
        encode,
        threshold=0.5,
        margin=0.1,
    )

    assert accepted == []
    assert report["pairs"][0]["passed"] is False
