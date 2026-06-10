from __future__ import annotations

import gc
import re
from collections import defaultdict
from typing import Any, Callable

import numpy as np


ENTITY_ID_PATTERN = re.compile(
    r"\b(?:T_|V_)?(?:PERSON|OBJECT|ANIMAL|LOCATION|ORGANIZATION|CONCEPT|"
    r"EVENT|SCREEN_ELEMENT|ENTITY)(?:_[A-Z]+)*_\d+\b"
)
TIMESTAMP_PATTERN = re.compile(
    r"\[\d+(?:\.\d+)?s\s*->\s*\d+(?:\.\d+)?s\]:\s*"
)


def _semantic_text(value: str) -> str:
    value = TIMESTAMP_PATTERN.sub("", value)
    value = ENTITY_ID_PATTERN.sub("entity", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def transcript_propositions(
    text_result: dict[str, Any],
    segment_id: str,
) -> list[dict[str, Any]]:
    memory = text_result.get("memory_after", {}).get("short_term", {})
    events = [
        event
        for event in memory.get("recent_events", [])
        if str(event.get("segment_id")) == segment_id
        and str(event.get("text", "")).strip()
    ]
    if events:
        return [
            {
                "proposition_id": f"TP_{index:03d}",
                "text": str(event["text"]),
                "entity_ids": list(
                    dict.fromkeys(
                        str(entity_id)
                        for entity_id in event.get("entity_ids", [])
                    )
                ),
                "predicate": str(event.get("predicate", "")),
                "start": event.get("start"),
                "end": event.get("end"),
            }
            for index, event in enumerate(events, start=1)
        ]

    transcript = str(text_result.get("rewritten_transcript", "")).strip()
    entity_ids = [
        str(mention["text_id"])
        for mention in text_result.get("mentions", [])
    ]
    if not transcript or not entity_ids:
        return []
    return [
        {
            "proposition_id": "TP_001",
            "text": transcript,
            "entity_ids": list(dict.fromkeys(entity_ids)),
            "predicate": "",
            "start": None,
            "end": None,
        }
    ]


def visual_facts(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    for item in analysis.get("entity_descriptions", []):
        entity_id = str(item.get("entity_id", "")).strip()
        description = str(item.get("description", "")).strip()
        if entity_id and description:
            facts.append(
                {
                    "fact_id": f"VF_{len(facts) + 1:03d}",
                    "text": f"{entity_id}: {description}",
                    "entity_ids": [entity_id],
                    "kind": "entity_description",
                }
            )
    for edge in analysis.get("visual_edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        predicate = str(edge.get("predicate", "")).strip().replace("_", " ")
        if source and target and predicate:
            facts.append(
                {
                    "fact_id": f"VF_{len(facts) + 1:03d}",
                    "text": f"{source} {predicate} {target}",
                    "entity_ids": [source, target],
                    "kind": "visual_relation",
                }
            )
    return facts


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Correspondence embeddings must be a 2D array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def gate_merge_candidates(
    candidates: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
    encode_texts: Callable[[list[str]], np.ndarray],
    *,
    threshold: float = 0.28,
    margin: float = 0.04,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates or not facts or not propositions:
        return [], {
            "status": "no_correspondence_evidence",
            "threshold": threshold,
            "margin": margin,
            "pairs": [],
        }

    embedding_texts = [
        _semantic_text(fact["text"]) for fact in facts
    ] + [
        _semantic_text(item["text"]) for item in propositions
    ]
    embeddings = _normalize_rows(encode_texts(embedding_texts))
    fact_vectors = embeddings[: len(facts)]
    proposition_vectors = embeddings[len(facts) :]
    similarities = fact_vectors @ proposition_vectors.T

    visual_fact_indexes: dict[str, list[int]] = defaultdict(list)
    text_proposition_indexes: dict[str, list[int]] = defaultdict(list)
    for index, fact in enumerate(facts):
        for entity_id in fact.get("entity_ids", []):
            visual_fact_indexes[str(entity_id)].append(index)
    for index, proposition in enumerate(propositions):
        for entity_id in proposition.get("entity_ids", []):
            text_proposition_indexes[str(entity_id)].append(index)

    scored = []
    for candidate in candidates:
        visual_id = str(candidate["visual_id"])
        text_id = str(candidate["text_id"])
        fact_indexes = visual_fact_indexes.get(visual_id, [])
        proposition_indexes = text_proposition_indexes.get(text_id, [])
        if not fact_indexes or not proposition_indexes:
            continue
        best = max(
            (
                float(similarities[fact_index, proposition_index]),
                fact_index,
                proposition_index,
            )
            for fact_index in fact_indexes
            for proposition_index in proposition_indexes
        )
        scored.append(
            {
                **candidate,
                "correspondence_score": best[0],
                "visual_fact_id": facts[best[1]]["fact_id"],
                "visual_fact": facts[best[1]]["text"],
                "transcript_proposition_id": propositions[best[2]][
                    "proposition_id"
                ],
                "transcript_proposition": propositions[best[2]]["text"],
            }
        )

    by_visual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_visual[item["visual_id"]].append(item)
        by_text[item["text_id"]].append(item)
    for values in (*by_visual.values(), *by_text.values()):
        values.sort(
            key=lambda item: float(item["correspondence_score"]),
            reverse=True,
        )

    accepted = []
    diagnostics = []
    for item in scored:
        visual_rank = by_visual[item["visual_id"]]
        text_rank = by_text[item["text_id"]]
        visual_best = visual_rank[0] is item
        text_best = text_rank[0] is item
        visual_margin = (
            float(item["correspondence_score"])
            - float(visual_rank[1]["correspondence_score"])
            if len(visual_rank) > 1
            else 1.0
        )
        text_margin = (
            float(item["correspondence_score"])
            - float(text_rank[1]["correspondence_score"])
            if len(text_rank) > 1
            else 1.0
        )
        passed = (
            float(item["correspondence_score"]) >= threshold
            and visual_best
            and text_best
            and visual_margin >= margin
            and text_margin >= margin
        )
        diagnostics.append(
            {
                **item,
                "mutual_best": visual_best and text_best,
                "visual_margin": visual_margin,
                "text_margin": text_margin,
                "passed": passed,
            }
        )
        if passed:
            accepted.append(item)

    return accepted, {
        "status": "matched" if accepted else "independent",
        "threshold": threshold,
        "margin": margin,
        "pairs": diagnostics,
    }


class OpenCLIPTextEncoder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Cross-modal correspondence requires open-clip-torch."
            ) from exc
        configured = str(
            self.config.get("correspondence_device", "auto")
        ).lower()
        self.device = (
            "cuda"
            if configured == "auto" and torch.cuda.is_available()
            else ("cpu" if configured == "auto" else configured)
        )
        if self.config.get("pipeline_strict", False) and self.device != "cuda":
            raise RuntimeError(
                "Strict pipeline requires CUDA for OpenCLIP correspondence."
            )
        model_name = str(self.config.get("openclip_model", "ViT-B-32"))
        pretrained = str(
            self.config.get("openclip_pretrained", "laion2b_s34b_b79k")
        )
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self.load()
        import torch

        batch_size = int(
            self.config.get("correspondence_embedding_batch_size", 64)
        )
        outputs = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    texts[offset : offset + batch_size]
                ).to(self.device)
                vectors = self.model.encode_text(tokens)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
                outputs.append(vectors.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
