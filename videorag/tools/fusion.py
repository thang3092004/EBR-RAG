from __future__ import annotations

import re
from collections import defaultdict

from ..debate.evidence_types import EvidenceItem


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokens(item: EvidenceItem) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(
            f"{item.id} {item.snippet or ''} {item.provenance_path or ''}"
        )
    }


def _similarity(left: EvidenceItem, right: EvidenceItem) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def reciprocal_rank_fusion(
    channels: dict[str, list[EvidenceItem]],
    *,
    limit: int,
    rrf_constant: int = 60,
    mmr_lambda: float = 0.75,
) -> list[EvidenceItem]:
    by_id: dict[str, EvidenceItem] = {}
    rrf_scores: dict[str, float] = defaultdict(float)
    retrieval_channels: dict[str, list[str]] = defaultdict(list)
    for channel, items in channels.items():
        for rank, item in enumerate(items, start=1):
            by_id.setdefault(item.id, item)
            rrf_scores[item.id] += 1.0 / (rrf_constant + rank)
            retrieval_channels[item.id].append(channel)

    for evidence_id, item in by_id.items():
        item.metadata["retrieval_channels"] = retrieval_channels[evidence_id]
        item.metadata["rrf_score"] = rrf_scores[evidence_id]

    remaining = list(by_id.values())
    selected: list[EvidenceItem] = []
    max_rrf = max(rrf_scores.values(), default=1.0)
    while remaining and len(selected) < limit:
        best = None
        best_score = float("-inf")
        for item in remaining:
            relevance = rrf_scores[item.id] / max_rrf
            redundancy = (
                max(_similarity(item, chosen) for chosen in selected)
                if selected
                else 0.0
            )
            score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if score > best_score:
                best = item
                best_score = score
        selected.append(best)
        remaining.remove(best)
    return selected
