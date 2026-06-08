from videorag.debate.evidence_types import EvidenceItem
from videorag.tools.fusion import reciprocal_rank_fusion


def _evidence(evidence_id, source, snippet):
    return EvidenceItem(
        id=evidence_id,
        type="text",
        score=0.5,
        snippet=snippet,
        source=source,
    )


def test_fusion_deduplicates_and_respects_limit():
    shared = _evidence("same", "text", "shared evidence")
    result = reciprocal_rank_fusion(
        {
            "text": [shared, _evidence("text", "text", "lecture transcript")],
            "graph": [shared, _evidence("graph", "graph", "PERSON holds OBJECT")],
            "visual": [_evidence("visual", "visual", "a person on stage")],
        },
        limit=3,
    )
    assert len(result) == 3
    assert len({item.id for item in result}) == 3
    same = next(item for item in result if item.id == "same")
    assert set(same.metadata["retrieval_channels"]) == {"text", "graph"}

