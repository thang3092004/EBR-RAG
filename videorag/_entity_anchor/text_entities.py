from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from .._unified_graph.registry import normalize_alias
from .._unified_graph.schema import normalize_entity_type
from .text_memory import (
    GENERIC_REFERENCE_SPECS,
    REFERENCE_SPECS,
    TextDiscourseMemory,
    is_indefinite_introduction,
    reference_spec,
)


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
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
OBLIQUE_DEPS = {"pobj", "obl"}
POSSESSOR_DEPS = {"poss"}


def _load_nlp(config: dict[str, Any]):
    try:
        import spacy
    except ImportError as exc:
        if config.get("pipeline_strict", False):
            raise RuntimeError(
                "Strict pipeline requires spaCy and the configured model."
            ) from exc
        return None, "regex_fallback"
    model_name = str(config.get("spacy_model", "en_core_web_trf"))
    try:
        return spacy.load(model_name), model_name
    except OSError as exc:
        if config.get("pipeline_strict", False):
            raise RuntimeError(
                "Strict pipeline requires the configured spaCy model without "
                f"fallback: {model_name}"
            ) from exc
        try:
            return spacy.load("en_core_web_sm"), "en_core_web_sm"
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp, "spacy_blank_en"


def _transcript_spans(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = []
    current = None
    for word in sorted(words, key=lambda item: float(item["start"])):
        if current is None or float(word["start"]) - current["end"] > 0.8:
            if current:
                spans.append(current)
            current = {
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


def _role_from_dep(dep: str | None) -> str:
    normalized = str(dep or "").lower()
    if normalized in SUBJECT_DEPS:
        return "subject"
    if normalized in OBJECT_DEPS:
        return "object"
    if normalized in OBLIQUE_DEPS:
        return "oblique"
    if normalized in POSSESSOR_DEPS:
        return "possessor"
    return "other" if normalized else "unknown"


def _number_from_token(token) -> str:
    try:
        values = token.morph.get("Number")
    except (AttributeError, KeyError):
        values = []
    if "Sing" in values:
        return "singular"
    if "Plur" in values:
        return "plural"
    return "unknown"


def _head_word(text: str) -> str:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return words[-1] if words else ""


def _generic_entity_type(head: str) -> str:
    expected = GENERIC_REFERENCE_SPECS.get(head, ({"object"}, "unknown", None))[0]
    if "person" in expected:
        return "person"
    return "object"


class TextEntityExtractor:
    def __init__(
        self,
        config: dict[str, Any],
        state: dict[str, Any] | None = None,
    ):
        self.config = config
        self.memory_enabled = not bool(
            config.get("disable_transcript_memory", False)
        )
        self.nlp, self.backend = _load_nlp(config)
        self.counters: dict[str, int] = defaultdict(int)
        self.mention_index: dict[tuple[str, str], str] = {}
        self.alias_vectors: dict[tuple[str, str], np.ndarray] = {}
        state = state or {}
        self.counters.update(
            {
                str(key): int(value)
                for key, value in state.get("counters", {}).items()
            }
        )
        for item in state.get("mention_index", []):
            key = (str(item["entity_type"]), str(item["normalized"]))
            self.mention_index[key] = str(item["text_id"])
        for item in state.get("alias_vectors", []):
            key = (str(item["entity_type"]), str(item["normalized"]))
            self.alias_vectors[key] = np.asarray(
                item.get("vector", []),
                dtype=np.float32,
            )
        self.memory = TextDiscourseMemory(config, state.get("memory"))

    def to_state(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "mention_index": [
                {
                    "entity_type": key[0],
                    "normalized": key[1],
                    "text_id": text_id,
                }
                for key, text_id in self.mention_index.items()
            ],
            "alias_vectors": [
                {
                    "entity_type": key[0],
                    "normalized": key[1],
                    "vector": vector.tolist(),
                }
                for key, vector in self.alias_vectors.items()
            ],
            "memory": self.memory.to_state(),
        }

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
            if not self.memory_enabled:
                self.mention_index[key] = self._new_id(entity_type)
                return self.mention_index[key]
            vector = self._mention_vector(text)
            scored = []
            for candidate_key, candidate_id in self.mention_index.items():
                if candidate_key[0] != key[0]:
                    continue
                lexical = SequenceMatcher(None, key[1], candidate_key[1]).ratio()
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

    def _extract_items_spacy(self, doc) -> list[dict[str, Any]]:
        candidates: dict[tuple[int, int], dict[str, Any]] = {}
        for entity in getattr(doc, "ents", []):
            entity_type = ENTITY_TYPE_MAP.get(entity.label_, "concept")
            candidates[(entity.start_char, entity.end_char)] = {
                "kind": "entity",
                "text": entity.text.strip(),
                "entity_type": entity_type,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "method": f"spacy_ner:{entity.label_}",
                "confidence": 0.90,
                "role": _role_from_dep(getattr(entity.root, "dep_", "")),
                "number": _number_from_token(entity.root),
                "generic": False,
            }
        try:
            noun_chunks = list(doc.noun_chunks)
        except (ValueError, AttributeError):
            noun_chunks = []
        reference_ranges = set()
        for chunk in noun_chunks:
            text = chunk.text.strip()
            head = str(getattr(chunk.root, "lemma_", "") or chunk.root.text).lower()
            spec = reference_spec(text, head)
            if spec is not None:
                if is_indefinite_introduction(text, head):
                    candidates[(chunk.start_char, chunk.end_char)] = {
                        "kind": "entity",
                        "text": text,
                        "entity_type": _generic_entity_type(head),
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "method": "generic_introduction",
                        "confidence": 0.50,
                        "role": _role_from_dep(getattr(chunk.root, "dep_", "")),
                        "number": spec["number"],
                        "descriptor": spec["descriptor"],
                        "generic": True,
                    }
                else:
                    candidates[(chunk.start_char, chunk.end_char)] = {
                        "kind": "reference",
                        "text": text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "method": "generic_reference",
                        "role": _role_from_dep(getattr(chunk.root, "dep_", "")),
                        **spec,
                    }
                    reference_ranges.add((chunk.start_char, chunk.end_char))
                continue
            normalized = normalize_alias(text)
            if not normalized or len(normalized) < 3:
                continue
            start_char = chunk.start_char
            end_char = chunk.end_char
            possessive_tokens = [
                token
                for token in chunk
                if reference_spec(token.text, token.text.lower()) is not None
                and _role_from_dep(getattr(token, "dep_", "")) == "possessor"
            ]
            if possessive_tokens:
                start_char = int(chunk.root.idx)
                end_char = start_char + len(chunk.root.text)
                text = chunk.root.text
            candidates.setdefault(
                (start_char, end_char),
                {
                    "kind": "entity",
                    "text": text,
                    "entity_type": (
                        "object"
                        if getattr(chunk.root, "pos_", "") in {"NOUN", "PROPN"}
                        else "concept"
                    ),
                    "start_char": start_char,
                    "end_char": end_char,
                    "method": "spacy_noun_chunk",
                    "confidence": 0.65,
                    "role": _role_from_dep(getattr(chunk.root, "dep_", "")),
                    "number": _number_from_token(chunk.root),
                    "generic": False,
                },
            )
        for token in doc:
            spec = reference_spec(token.text, token.text.lower())
            if spec is None:
                continue
            token_range = (int(token.idx), int(token.idx) + len(token.text))
            if any(start <= token_range[0] and token_range[1] <= end for start, end in reference_ranges):
                continue
            candidates[token_range] = {
                "kind": "reference",
                "text": token.text,
                "start_char": token_range[0],
                "end_char": token_range[1],
                "method": "spacy_reference",
                "role": _role_from_dep(getattr(token, "dep_", "")),
                "number": (
                    spec["number"]
                    if spec["number"] != "unknown"
                    else _number_from_token(token)
                ),
                "expected_types": spec["expected_types"],
                "descriptor": spec["descriptor"],
                "resolvable": spec["resolvable"],
            }
        return sorted(candidates.values(), key=lambda item: item["start_char"])

    def _extract_items_regex(self, text: str) -> list[dict[str, Any]]:
        candidates: dict[tuple[int, int], dict[str, Any]] = {}
        generic_words = "|".join(
            sorted(GENERIC_REFERENCE_SPECS, key=len, reverse=True)
        )
        generic_pattern = re.compile(
            rf"\b(?:(?:a|an|another|some|one|the|this|that|these|those)\s+)?"
            rf"(?:{generic_words})\b",
            re.IGNORECASE,
        )
        for match in generic_pattern.finditer(text):
            value = match.group(0)
            head = _head_word(value)
            spec = reference_spec(value, head)
            if spec is None:
                continue
            if is_indefinite_introduction(value, head):
                item = {
                    "kind": "entity",
                    "text": value,
                    "entity_type": _generic_entity_type(head),
                    "method": "regex_generic_introduction",
                    "confidence": 0.45,
                    "role": "unknown",
                    "number": spec["number"],
                    "descriptor": spec["descriptor"],
                    "generic": True,
                }
            else:
                item = {
                    "kind": "reference",
                    "text": value,
                    "method": "regex_generic_reference",
                    "role": "unknown",
                    **spec,
                }
            candidates[(match.start(), match.end())] = {
                **item,
                "start_char": match.start(),
                "end_char": match.end(),
            }
        reference_pattern = re.compile(
            r"\b(" + "|".join(
                sorted(REFERENCE_SPECS, key=len, reverse=True)
            ) + r"|i|me|my|mine|you|your|yours|we|us|our|ours)\b",
            re.IGNORECASE,
        )
        for match in reference_pattern.finditer(text):
            value = match.group(0)
            spec = reference_spec(value, value.lower())
            candidates[(match.start(), match.end())] = {
                "kind": "reference",
                "text": value,
                "start_char": match.start(),
                "end_char": match.end(),
                "method": "regex_reference",
                "role": "unknown",
                **(spec or {}),
            }
        proper_pattern = re.compile(
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        )
        for match in proper_pattern.finditer(text):
            value = match.group(0).strip()
            if reference_spec(value, value.lower()) is not None:
                continue
            if any(
                start <= match.start() and match.end() <= end
                for start, end in candidates
            ):
                continue
            candidates[(match.start(), match.end())] = {
                "kind": "entity",
                "text": value,
                "entity_type": "person",
                "start_char": match.start(),
                "end_char": match.end(),
                "method": "regex_proper_noun",
                "confidence": 0.45,
                "role": "unknown",
                "number": "singular",
                "generic": False,
            }
        return sorted(candidates.values(), key=lambda item: item["start_char"])

    def _event_from_doc(
        self,
        doc,
        rewritten: str,
        entity_ids: list[str],
        span: dict[str, Any],
        segment: dict[str, Any],
    ) -> dict[str, Any]:
        predicate = ""
        if doc is not None:
            root = next(
                (
                    token
                    for token in doc
                    if str(getattr(token, "dep_", "")).upper() == "ROOT"
                ),
                None,
            )
            if root is not None:
                predicate = str(
                    getattr(root, "lemma_", "") or root.text
                ).lower()
        return {
            "segment_id": segment["segment_id"],
            "start": float(span["start"]),
            "end": float(span["end"]),
            "predicate": predicate,
            "entity_ids": list(dict.fromkeys(entity_ids)),
            "text": rewritten,
        }

    def _record_for_entity(
        self,
        mentions: dict[str, dict[str, Any]],
        *,
        text_id: str,
        label: str,
        entity_type: str,
        confidence: float,
        method: str,
        segment: dict[str, Any],
        span: dict[str, Any],
        role: str,
        generic: bool,
        occurrence_kind: str,
        surface: str,
    ) -> dict[str, Any]:
        record = mentions.setdefault(
            text_id,
            {
                "text_id": text_id,
                "entity_type": entity_type,
                "label": label,
                "aliases": [],
                "confidence": confidence,
                "methods": [],
                "segment_id": segment["segment_id"],
                "start": span["start"],
                "end": span["end"],
                "generic": generic,
                "occurrences": [],
                "reference_forms": [],
            },
        )
        record["confidence"] = max(
            float(record.get("confidence", 0.0)),
            float(confidence),
        )
        record["start"] = min(float(record["start"]), float(span["start"]))
        record["end"] = max(float(record["end"]), float(span["end"]))
        if occurrence_kind == "explicit" and surface not in record["aliases"]:
            record["aliases"].append(surface)
        if occurrence_kind == "reference" and surface not in record["reference_forms"]:
            record["reference_forms"].append(surface)
        if method not in record["methods"]:
            record["methods"].append(method)
        record["occurrences"].append(
            {
                "start": float(span["start"]),
                "end": float(span["end"]),
                "text": surface,
                "role": role,
                "kind": occurrence_kind,
                "confidence": float(confidence),
            }
        )
        return record

    def extract_segment(
        self,
        segment: dict[str, Any],
        words: list[dict[str, Any]],
    ) -> dict[str, Any]:
        mentions: dict[str, dict[str, Any]] = {}
        references = []
        rewritten_lines = []
        segment_start = float(
            segment.get(
                "start",
                min(
                    (float(word["start"]) for word in words),
                    default=0.0,
                ),
            )
        )
        segment_end = float(
            segment.get(
                "end",
                max(
                    (float(word["end"]) for word in words),
                    default=segment_start,
                ),
            )
        )
        memory_context = (
            self.memory.prompt_context(current_time=segment_start)
            if self.memory_enabled
            else {
                "short_term": {
                    "recent_mentions": [],
                    "recent_events": [],
                    "active_entities": [],
                },
                "long_term": {"entities": []},
            }
        )
        spans = _transcript_spans(words)
        segment_index = int(segment.get("index", 0))
        for span in spans:
            text = span["text"]
            doc = self.nlp(text) if self.nlp is not None else None
            items = (
                self._extract_items_spacy(doc)
                if doc is not None
                else self._extract_items_regex(text)
            )
            replacements = []
            event_entity_ids = []
            for item in items:
                if item["kind"] == "entity":
                    text_id = (
                        self._new_id(item["entity_type"])
                        if item.get("generic")
                        else self._mention_id(
                            item["text"],
                            item["entity_type"],
                        )
                    )
                    self._record_for_entity(
                        mentions,
                        text_id=text_id,
                        label=item["text"],
                        entity_type=item["entity_type"],
                        confidence=float(item["confidence"]),
                        method=item["method"],
                        segment=segment,
                        span=span,
                        role=item.get("role", "unknown"),
                        generic=bool(item.get("generic", False)),
                        occurrence_kind="explicit",
                        surface=item["text"],
                    )
                    if self.memory_enabled:
                        self.memory.observe_entity(
                            text_id=text_id,
                            entity_type=item["entity_type"],
                            label=item["text"],
                            start=float(span["start"]),
                            end=float(span["end"]),
                            segment_id=segment["segment_id"],
                            segment_index=segment_index,
                            role=item.get("role", "unknown"),
                            number=item.get("number", "unknown"),
                            descriptor=item.get("descriptor"),
                            generic=bool(item.get("generic", False)),
                            confidence=float(item["confidence"]),
                        )
                    event_entity_ids.append(text_id)
                    replacement = f"[{text_id}]"
                else:
                    reference_payload = {
                        **item,
                        "start": float(span["start"]),
                        "end": float(span["end"]),
                        "segment_id": segment["segment_id"],
                        "segment_index": segment_index,
                    }
                    if self.memory_enabled:
                        reference = self.memory.resolve_reference(
                            reference_payload
                        )
                    else:
                        reference = {
                            "reference_id": self.memory.new_reference_id(),
                            "text": item["text"],
                            "start": float(span["start"]),
                            "end": float(span["end"]),
                            "role": item.get("role", "unknown"),
                            "expected_types": list(
                                item.get("expected_types", [])
                            ),
                            "number": item.get("number", "unknown"),
                            "descriptor": item.get("descriptor"),
                            "status": "unresolved",
                            "resolved_text_id": None,
                            "confidence": 0.0,
                            "candidates": [],
                        }
                    references.append(reference)
                    resolved_id = reference.get("resolved_text_id")
                    if resolved_id:
                        state = self.memory.entities[resolved_id]
                        self._record_for_entity(
                            mentions,
                            text_id=resolved_id,
                            label=state["canonical_label"],
                            entity_type=state["entity_type"],
                            confidence=float(reference["confidence"]),
                            method="memory_coreference",
                            segment=segment,
                            span=span,
                            role=item.get("role", "unknown"),
                            generic=bool(state.get("generic", False)),
                            occurrence_kind="reference",
                            surface=item["text"],
                        )
                        if self.memory_enabled:
                            self.memory.observe_entity(
                                text_id=resolved_id,
                                entity_type=state["entity_type"],
                                label=state["canonical_label"],
                                start=float(span["start"]),
                                end=float(span["end"]),
                                segment_id=segment["segment_id"],
                                segment_index=segment_index,
                                role=item.get("role", "unknown"),
                                number=item.get("number", "unknown"),
                                descriptor=item.get("descriptor"),
                                generic=bool(state.get("generic", False)),
                                confidence=float(reference["confidence"]),
                                reference_text=item["text"],
                            )
                        event_entity_ids.append(resolved_id)
                        replacement = f"[{resolved_id}]"
                    else:
                        replacement = (
                            f'<unresolved-ref id="{reference["reference_id"]}">'
                            f'{item["text"]}</unresolved-ref>'
                        )
                replacements.append(
                    (
                        int(item["start_char"]),
                        int(item["end_char"]),
                        replacement,
                    )
                )
            rewritten = text
            occupied = []
            for start_char, end_char, replacement in sorted(
                replacements,
                key=lambda value: (value[0], value[1] - value[0]),
                reverse=True,
            ):
                if any(
                    start_char < used_end and end_char > used_start
                    for used_start, used_end in occupied
                ):
                    continue
                rewritten = (
                    rewritten[:start_char]
                    + replacement
                    + rewritten[end_char:]
                )
                occupied.append((start_char, end_char))
            rewritten_lines.append(
                f"[{span['start']:.2f}s -> {span['end']:.2f}s]: {rewritten}"
            )
            event = self._event_from_doc(
                doc,
                rewritten,
                event_entity_ids,
                span,
                segment,
            )
            if self.memory_enabled:
                self.memory.add_event(event)
        return {
            "segment_id": segment["segment_id"],
            "backend": self.backend,
            "mentions": list(mentions.values()),
            "references": references,
            "memory_context": memory_context,
            "memory_after": (
                self.memory.prompt_context(current_time=segment_end)
                if self.memory_enabled
                else {
                    "short_term": {
                        "recent_mentions": [],
                        "recent_events": [],
                        "active_entities": [],
                    },
                    "long_term": {"entities": []},
                }
            ),
            "rewritten_transcript": "\n".join(rewritten_lines),
        }
