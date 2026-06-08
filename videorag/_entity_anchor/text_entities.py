from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from .._unified_graph.registry import normalize_alias
from .._unified_graph.schema import normalize_entity_type


PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "him",
    "her",
    "them",
    "this",
    "that",
    "these",
    "those",
}
GENERIC_NOUNS = {
    "thing",
    "something",
    "anything",
    "person",
    "people",
    "man",
    "woman",
    "someone",
}
ENTITY_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "PRODUCT": "object",
    "WORK_OF_ART": "object",
    "NORP": "concept",
    "LAW": "concept",
    "LANGUAGE": "concept",
}


def _load_nlp(config: dict[str, Any]):
    try:
        import spacy
    except ImportError:
        return None, "regex_fallback"
    model_name = str(config.get("spacy_model", "en_core_web_trf"))
    try:
        return spacy.load(model_name), model_name
    except OSError:
        try:
            return spacy.load("en_core_web_sm"), "en_core_web_sm"
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp, "spacy_blank_en"


def _speaker_spans(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = []
    current = None
    for word in sorted(words, key=lambda item: float(item["start"])):
        speaker = str(word.get("speaker_id") or "SPEAKER_UNKNOWN")
        if (
            current is None
            or current["speaker_id"] != speaker
            or float(word["start"]) - current["end"] > 0.8
        ):
            if current:
                spans.append(current)
            current = {
                "speaker_id": speaker,
                "start": float(word["start"]),
                "end": float(word["end"]),
                "words": [str(word["text"])],
            }
        else:
            current["words"].append(str(word["text"]))
            current["end"] = float(word["end"])
    if current:
        spans.append(current)
    for span in spans:
        span["text"] = " ".join(span.pop("words")).strip()
    return spans


class TextEntityExtractor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.nlp, self.backend = _load_nlp(config)
        self.counters: dict[str, int] = defaultdict(int)
        self.claim_counter = 0
        self.mention_index: dict[tuple[str, str], str] = {}
        self.alias_vectors: dict[tuple[str, str], np.ndarray] = {}

    def _new_id(self, entity_type: str) -> str:
        normalized = normalize_entity_type(entity_type)
        self.counters[normalized] += 1
        return f"T_{normalized.upper()}_{self.counters[normalized]:03d}"

    def _mention_vector(self, text: str) -> np.ndarray | None:
        if self.nlp is None:
            return None
        vector = np.asarray(self.nlp(text).vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or norm == 0:
            return None
        return vector / norm

    def _mention_id(self, text: str, entity_type: str) -> str:
        key = (normalize_entity_type(entity_type), normalize_alias(text))
        if key not in self.mention_index:
            vector = self._mention_vector(text)
            scored = []
            for candidate_key, candidate_id in self.mention_index.items():
                if candidate_key[0] != key[0]:
                    continue
                lexical = SequenceMatcher(
                    None,
                    key[1],
                    candidate_key[1],
                ).ratio()
                candidate_vector = self.alias_vectors.get(candidate_key)
                semantic = (
                    float(np.dot(vector, candidate_vector))
                    if vector is not None and candidate_vector is not None
                    else 0.0
                )
                scored.append((max(lexical, semantic), candidate_id))
            scored.sort(reverse=True)
            threshold = float(
                self.config.get("text_alias_similarity_threshold", 0.88)
            )
            if (
                scored
                and scored[0][0] >= threshold
                and (
                    len(scored) == 1
                    or scored[0][0] - scored[1][0] >= 0.05
                )
            ):
                self.mention_index[key] = scored[0][1]
            else:
                self.mention_index[key] = self._new_id(entity_type)
            if vector is not None:
                self.alias_vectors[key] = vector
        return self.mention_index[key]

    def _extract_mentions_spacy(self, doc) -> list[dict[str, Any]]:
        candidates: dict[tuple[int, int], dict[str, Any]] = {}
        for entity in getattr(doc, "ents", []):
            entity_type = ENTITY_TYPE_MAP.get(entity.label_, "concept")
            candidates[(entity.start_char, entity.end_char)] = {
                "text": entity.text.strip(),
                "entity_type": entity_type,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "method": f"spacy_ner:{entity.label_}",
                "confidence": 0.90,
            }
        try:
            noun_chunks = list(doc.noun_chunks)
        except (ValueError, AttributeError):
            noun_chunks = []
        for chunk in noun_chunks:
            text = chunk.text.strip()
            normalized = normalize_alias(text)
            if (
                not normalized
                or normalized in PRONOUNS
                or normalized in GENERIC_NOUNS
                or len(normalized) < 3
            ):
                continue
            key = (chunk.start_char, chunk.end_char)
            candidates.setdefault(
                key,
                {
                    "text": text,
                    "entity_type": "object"
                    if getattr(chunk.root, "pos_", "") in {"NOUN", "PROPN"}
                    else "concept",
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "method": "spacy_noun_chunk",
                    "confidence": 0.65,
                },
            )
        return sorted(candidates.values(), key=lambda item: item["start_char"])

    def _extract_mentions_regex(self, text: str) -> list[dict[str, Any]]:
        mentions = []
        for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            value = match.group(0).strip()
            if normalize_alias(value) in GENERIC_NOUNS:
                continue
            mentions.append(
                {
                    "text": value,
                    "entity_type": "person",
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "method": "regex_proper_noun",
                    "confidence": 0.45,
                }
            )
        return mentions

    def _claims(self, doc, text: str) -> list[dict[str, Any]]:
        claims = []
        try:
            sentences = list(getattr(doc, "sents", [])) if doc is not None else []
        except (ValueError, AttributeError):
            sentences = []
        if not sentences:
            sentences = [None]
        for sentence in sentences:
            sentence_text = sentence.text.strip() if sentence is not None else text.strip()
            if not sentence_text:
                continue
            subject = None
            predicate = None
            obj = None
            if sentence is not None:
                for token in sentence:
                    if token.dep_ in {"nsubj", "nsubjpass"} and subject is None:
                        subject = token.text
                    if token.pos_ in {"VERB", "AUX"} and predicate is None:
                        predicate = token.lemma_.lower()
                    if token.dep_ in {"dobj", "obj", "attr", "pobj"} and obj is None:
                        obj = token.text
            if predicate or len(sentence_text.split()) >= 4:
                self.claim_counter += 1
                claims.append(
                    {
                        "text_id": f"T_CLAIM_{self.claim_counter:03d}",
                        "entity_type": "claim",
                        "text": sentence_text,
                        "subject_text": subject,
                        "predicate": predicate or "states",
                        "object_text": obj,
                        "confidence": 0.75 if predicate else 0.55,
                    }
                )
        return claims

    def extract_segment(
        self,
        segment: dict[str, Any],
        words: list[dict[str, Any]],
    ) -> dict[str, Any]:
        mentions: dict[str, dict[str, Any]] = {}
        claims: list[dict[str, Any]] = []
        rewritten_lines = []
        spans = _speaker_spans(words)
        for span in spans:
            text = span["text"]
            doc = self.nlp(text) if self.nlp is not None else None
            raw_mentions = (
                self._extract_mentions_spacy(doc)
                if doc is not None
                else self._extract_mentions_regex(text)
            )
            replacements = []
            for mention in raw_mentions:
                text_id = self._mention_id(
                    mention["text"],
                    mention["entity_type"],
                )
                record = mentions.setdefault(
                    text_id,
                    {
                        "text_id": text_id,
                        "entity_type": mention["entity_type"],
                        "label": mention["text"],
                        "aliases": [],
                        "confidence": mention["confidence"],
                        "methods": [],
                        "segment_id": segment["segment_id"],
                        "start": span["start"],
                        "end": span["end"],
                    },
                )
                if mention["text"] not in record["aliases"]:
                    record["aliases"].append(mention["text"])
                if mention["method"] not in record["methods"]:
                    record["methods"].append(mention["method"])
                replacements.append(
                    (
                        int(mention["start_char"]),
                        int(mention["end_char"]),
                        f"[{text_id}]",
                    )
                )
            rewritten = text
            for start_char, end_char, replacement in sorted(
                replacements, reverse=True
            ):
                rewritten = rewritten[:start_char] + replacement + rewritten[end_char:]
            span_claims = self._claims(doc, text)
            for claim in span_claims:
                claim.update(
                    {
                        "speaker_id": span["speaker_id"],
                        "segment_id": segment["segment_id"],
                        "start": span["start"],
                        "end": span["end"],
                    }
                )
                claims.append(claim)
            rewritten_lines.append(
                f"[{span['start']:.2f}s -> {span['end']:.2f}s] "
                f"[{span['speaker_id']}]: {rewritten}"
            )
        return {
            "segment_id": segment["segment_id"],
            "backend": self.backend,
            "mentions": list(mentions.values()),
            "claims": claims,
            "rewritten_transcript": "\n".join(rewritten_lines),
        }
