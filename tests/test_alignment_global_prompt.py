import numpy as np

from videorag._unified_graph.alignment import align_all_segments
from videorag._unified_graph.registry import EntityRegistry


class _FakeAligner:
    def __init__(self):
        self.prompts = []

    def align(self, prompt, frame_paths):
        self.prompts.append(prompt)
        if prompt.startswith("Analyze only"):
            return {
                "visual_caption": "PERSON_001 speaks.",
                "entity_descriptions": [
                    {
                        "entity_id": "PERSON_001",
                        "description": "A person speaks.",
                    }
                ],
                "visual_edges": [],
            }
        if "T_PERSON_001" in prompt:
            return {
                "caption": "PERSON_001 appears.",
                "entity_merges": [
                    {
                        "visual_id": "PERSON_001",
                        "text_id": "T_PERSON_001",
                        "reason": "same person",
                    }
                ],
                "edges": [],
            }
        return {
            "caption": "PERSON_001 continues.",
            "entity_merges": [],
            "edges": [],
        }


class _FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_known_text_alias_is_global_in_next_segment_prompt(tmp_path):
    registry = EntityRegistry()
    registry.ensure_entity(
        "person",
        "person",
        entity_id="PERSON_001",
        source="visual",
    )
    segments = [
        {
            "segment_id": f"SEG_{index:05d}",
            "index": index,
            "start": float(index),
            "end": float(index + 1),
            "storage_id": f"video_{index}",
        }
        for index in range(2)
    ]
    text_results = {
        segment["segment_id"]: {
            "mentions": [
                {
                    "text_id": "T_PERSON_001",
                    "entity_type": "person",
                    "label": "John",
                    "confidence": 0.9,
                    "start": segment["start"],
                    "end": segment["end"],
                }
            ],
            "rewritten_transcript": "[T_PERSON_001] speaks.",
            "memory_context": {
                "short_term": {
                    "active_entities": [
                        {
                            "text_id": "T_PERSON_001",
                            "canonical_label": "John",
                        }
                    ]
                }
            },
            "references": [
                {
                    "reference_id": "R_000001",
                    "text": "he",
                    "status": "resolved",
                    "resolved_text_id": "T_PERSON_001",
                }
            ],
        }
        for segment in segments
    }
    observations = [
        {
            "segment_id": segment["segment_id"],
            "entity_id": "PERSON_001",
            "entity_type": "person",
            "label": "person",
            "time": segment["start"],
            "confidence": 0.9,
        }
        for segment in segments
    ]
    aligner = _FakeAligner()

    align_all_segments(
        "video",
        segments,
        {
            segment["segment_id"]: {
                "frames": [{"path": "unused.jpg"}],
            }
            for segment in segments
        },
        text_results,
        observations,
        registry,
        {},
        tmp_path,
        aligner=aligner,
        correspondence_encoder=_FakeEncoder(),
    )

    second_prompt = aligner.prompts[3]
    assert "[T_PERSON_001]" not in second_prompt
    assert "PERSON_001 speaks." in second_prompt
    assert '"text_id": "PERSON_001"' in second_prompt
    assert '"resolved_text_id": "PERSON_001"' in second_prompt
