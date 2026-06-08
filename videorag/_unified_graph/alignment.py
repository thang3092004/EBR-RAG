from __future__ import annotations

import json
import gc
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..pipeline.stage_runner import atomic_write_json, read_json
from .registry import EntityRegistry, normalize_alias
from .schema import EdgeOccurrence, ProvenanceRecord


ALIGNMENT_PROMPT = """You are grounding one video segment into an entity graph.

Rules:
1. Use only entity IDs and merge candidates explicitly listed below.
2. Never invent an ID.
3. Merge only when the frames and transcript clearly refer to the same entity.
4. A speaker is a person only when the supplied speaker-person link says so.
5. Return directed factual relations visible or explicitly stated in this segment.
6. If uncertain, do not merge and omit the uncertain edge.
7. For every edge, list only the evidence channels actually supporting it:
   "visual", "transcript", or both.

Return JSON only:
{{
  "caption": "A concise segment description using the supplied IDs.",
  "entity_merges": [
    {{
      "visual_id": "PERSON_001",
      "text_id": "T_PERSON_001",
      "reason": "short evidence-based reason"
    }}
  ],
  "edges": [
    {{
      "source": "PERSON_001",
      "predicate": "holds",
      "target": "OBJECT_001",
      "confidence": 0.0,
      "modalities": ["visual"]
    }}
  ]
}}

Segment: {segment_id}
Time: {start:.2f}-{end:.2f}s
Mode: {mode}

Visible entities:
{visual_entities}

Transcript entities:
{text_entities}

Allowed merge candidates:
{merge_candidates}

Accepted speaker-person links:
{speaker_links}

Recent event memory:
{recent_memory}

Entity state:
{entity_state}

Rewritten transcript:
{transcript}
"""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("MiniCPM response does not contain a JSON object.")
    return json.loads(match.group(0))


class MiniCPMAligner:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self._owns_model = model is None or tokenizer is None
        self.model_path = str(
            config.get("caption_model_path", "./MiniCPM-V-2_6-int4")
        )

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MiniCPM alignment requires torch and transformers."
            ) from exc
        model_path = Path(self.model_path)
        resolved = (
            str(model_path.resolve())
            if model_path.exists()
            else "openbmb/MiniCPM-V-2_6-int4"
        )
        device_map = str(self.config.get("caption_device", "cuda"))
        self.model = AutoModel.from_pretrained(
            resolved,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=str(
                self.config.get("caption_attention", "sdpa")
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
        )
        self.model.eval()

    def align(self, prompt: str, frame_paths: list[str]) -> dict[str, Any]:
        self.load()
        images = [
            Image.open(path).convert("RGB")
            for path in frame_paths
            if Path(path).exists()
        ]
        try:
            content = images + [prompt]
            messages = [{"role": "user", "content": content}]
            response = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=int(self.config.get("caption_max_slice_nums", 2)),
                max_new_tokens=int(self.config.get("caption_max_tokens", 450)),
            )
            try:
                return _extract_json(str(response))
            except (ValueError, json.JSONDecodeError):
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": str(response)},
                    {
                        "role": "user",
                        "content": "Rewrite the same answer as valid JSON only. Do not add commentary.",
                    },
                ]
                repaired = self.model.chat(
                    image=None,
                    msgs=repair_messages,
                    tokenizer=self.tokenizer,
                    use_image_id=False,
                    max_slice_nums=int(
                        self.config.get("caption_max_slice_nums", 2)
                    ),
                    max_new_tokens=int(
                        self.config.get("caption_max_tokens", 450)
                    ),
                )
                return _extract_json(str(repaired))
        finally:
            for image in images:
                image.close()

    def close(self) -> None:
        if not self._owns_model:
            return
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _visible_entities(
    segment_id: str,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if str(observation.get("segment_id")) == segment_id:
            by_entity[str(observation["entity_id"])].append(observation)
    result = []
    for entity_id, items in by_entity.items():
        result.append(
            {
                "entity_id": entity_id,
                "entity_type": items[0].get("entity_type", "unknown"),
                "label": items[0].get("label", ""),
                "first_seen": min(float(item["time"]) for item in items),
                "last_seen": max(float(item["time"]) for item in items),
                "confidence": sum(float(item["confidence"]) for item in items)
                / max(len(items), 1),
            }
        )
    return sorted(result, key=lambda item: item["entity_id"])


def _candidate_merges(
    visual_entities: list[dict[str, Any]],
    text_mentions: list[dict[str, Any]],
    registry: EntityRegistry,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates = []
    deterministic: dict[str, str] = {}
    for mention in text_mentions:
        label = str(mention.get("label", ""))
        normalized = normalize_alias(label)
        existing = registry.resolve(label)
        if (
            existing
            and mention["entity_type"] in {
                "person",
                "organization",
                "location",
                "concept",
            }
            and normalized not in {
            "person",
            "man",
            "woman",
            "object",
            "thing",
            }
        ):
            deterministic[mention["text_id"]] = existing
            continue
        compatible = [
            visual
            for visual in visual_entities
            if visual["entity_type"] == mention["entity_type"]
            or {
                visual["entity_type"],
                mention["entity_type"],
            }
            <= {"object", "concept"}
        ]
        compatible.sort(
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        for visual in compatible[:4]:
            candidates.append(
                {
                    "visual_id": visual["entity_id"],
                    "text_id": mention["text_id"],
                    "text_label": label,
                    "entity_type": mention["entity_type"],
                }
            )
    return candidates[:12], deterministic


def _rewrite_ids(text: str, registry: EntityRegistry) -> str:
    replacements = sorted(
        registry.alias_to_global.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, global_id in replacements:
        text = text.replace(f"[{alias}]", global_id)
        text = re.sub(rf"\b{re.escape(alias)}\b", global_id, text)
    return text


def _edge_id(video_id: str, counter: int) -> str:
    safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
    return f"EDGE_{safe_video}_{counter:07d}"


def _validate_alignment(
    raw: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {
        (candidate["visual_id"], candidate["text_id"]) for candidate in candidates
    }
    merges = []
    for merge in raw.get("entity_merges", []):
        pair = (str(merge.get("visual_id")), str(merge.get("text_id")))
        if pair in allowed:
            merges.append(
                {
                    "visual_id": pair[0],
                    "text_id": pair[1],
                    "reason": str(merge.get("reason", "")),
                }
            )
    edges = []
    for edge in raw.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        predicate = str(edge.get("predicate", "")).strip().lower().replace(" ", "_")
        if source and target and predicate and source != target:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "predicate": predicate,
                    "confidence": max(
                        0.0,
                        min(float(edge.get("confidence", 0.5)), 1.0),
                    ),
                    "modalities": [
                        modality
                        for modality in edge.get(
                            "modalities",
                            ["visual", "transcript"],
                        )
                        if modality in {"visual", "transcript"}
                    ]
                    or ["visual", "transcript"],
                }
            )
    return {
        "caption": str(raw.get("caption", "")).strip(),
        "entity_merges": merges,
        "edges": edges,
    }


def align_all_segments(
    video_id: str,
    segments: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    frame_selections: dict[str, dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    speaker_links: list[dict[str, Any]],
    initial_registry: EntityRegistry,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    progress=None,
    aligner: MiniCPMAligner | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_path / "alignment_state.json"
    state = read_json(state_path, {})
    registry = (
        EntityRegistry(state["registry"])
        if state.get("registry")
        else initial_registry
    )
    recent_events = list(state.get("recent_events", []))
    long_term_state = dict(state.get("long_term_state", {}))
    next_index = int(state.get("next_index", 0))
    edge_counter = int(state.get("edge_counter", 0))
    aligner = aligner or MiniCPMAligner(config)
    profile_map = {item["segment_id"]: item for item in profiles}

    accepted_speaker_map: dict[str, str] = {}
    for link in speaker_links:
        person_global = registry.resolve(str(link["person_id"]))
        if person_global:
            accepted_speaker_map[str(link["speaker_id"])] = person_global
            registry.add_alias(
                str(link["speaker_id"]),
                person_global,
                source="audio_visual",
                label="active speaker",
                confidence=float(link["score"]),
            )

    results: dict[str, dict[str, Any]] = {}
    for previous_index in range(next_index):
        segment_id = segments[previous_index]["segment_id"]
        previous = read_json(checkpoint_path / f"{segment_id}.json")
        if previous:
            results[segment_id] = previous

    for index, segment in enumerate(segments[next_index:], start=next_index):
        segment_id = str(segment["segment_id"])
        text_result = text_results.get(
            segment_id,
            {"mentions": [], "claims": [], "rewritten_transcript": ""},
        )
        visible = _visible_entities(segment_id, observations)
        candidates, deterministic = _candidate_merges(
            visible,
            text_result.get("mentions", []),
            registry,
        )
        for text_id, global_id in deterministic.items():
            mention = next(
                item
                for item in text_result["mentions"]
                if item["text_id"] == text_id
            )
            registry.add_alias(
                text_id,
                global_id,
                source="transcript",
                label=mention["label"],
                confidence=float(mention.get("confidence", 0.0)),
                segment_id=segment_id,
            )

        prompt_mentions = []
        for mention in text_result.get("mentions", []):
            resolved = registry.resolve(mention["text_id"])
            prompt_mentions.append(
                {
                    **mention,
                    "text_id": resolved or mention["text_id"],
                    **(
                        {}
                        if resolved
                        else {"provisional_id": mention["text_id"]}
                    ),
                }
            )
        prompt_transcript = _rewrite_ids(
            text_result.get("rewritten_transcript", ""),
            registry,
        )
        entity_state = [
            {
                "entity_id": node.entity_id,
                "name": node.canonical_name,
                "type": node.entity_type,
                "last_seen": node.last_seen,
                "recent_relations": long_term_state.get(
                    node.entity_id,
                    [],
                )[-3:],
            }
            for node in registry.entities.values()
            if node.last_seen is None
            or node.last_seen >= float(segment["start"]) - 120.0
        ][-20:]
        prompt = ALIGNMENT_PROMPT.format(
            segment_id=segment_id,
            start=float(segment["start"]),
            end=float(segment["end"]),
            mode=profile_map.get(segment_id, {}).get("mode", "balanced"),
            visual_entities=json.dumps(visible, ensure_ascii=False),
            text_entities=json.dumps(
                prompt_mentions, ensure_ascii=False
            ),
            merge_candidates=json.dumps(candidates, ensure_ascii=False),
            speaker_links=json.dumps(speaker_links, ensure_ascii=False),
            recent_memory=json.dumps(recent_events[-8:], ensure_ascii=False),
            entity_state=json.dumps(entity_state, ensure_ascii=False),
            transcript=prompt_transcript,
        )
        frames = [
            item["path"]
            for item in frame_selections.get(segment_id, {}).get("frames", [])
        ]
        raw_alignment = aligner.align(prompt, frames)
        alignment = _validate_alignment(raw_alignment, candidates)

        for merge in alignment["entity_merges"]:
            mention = next(
                item
                for item in text_result["mentions"]
                if item["text_id"] == merge["text_id"]
            )
            registry.add_alias(
                merge["text_id"],
                merge["visual_id"],
                source="transcript",
                label=mention["label"],
                confidence=float(mention.get("confidence", 0.0)),
                segment_id=segment_id,
            )

        for mention in text_result.get("mentions", []):
            if registry.resolve(mention["text_id"]):
                continue
            global_id = registry.ensure_entity(
                mention["entity_type"],
                mention["label"],
                source="transcript",
                confidence=float(mention.get("confidence", 0.0)),
            )
            registry.add_alias(
                mention["text_id"],
                global_id,
                source="transcript",
                label=mention["label"],
                confidence=float(mention.get("confidence", 0.0)),
                segment_id=segment_id,
            )
            registry.add_provenance(
                global_id,
                ProvenanceRecord(
                    source="transcript",
                    video_id=video_id,
                    segment_id=segment_id,
                    start=float(mention["start"]),
                    end=float(mention["end"]),
                    text=mention["label"],
                    confidence=float(mention.get("confidence", 0.0)),
                ),
            )

        edges: list[dict[str, Any]] = []
        for raw_edge in alignment["edges"]:
            source = registry.resolve(raw_edge["source"])
            target = registry.resolve(raw_edge["target"])
            if not source or not target:
                continue
            edge_counter += 1
            edge = EdgeOccurrence(
                edge_id=_edge_id(video_id, edge_counter),
                source_id=source,
                target_id=target,
                predicate=raw_edge["predicate"],
                start=float(segment["start"]),
                end=float(segment["end"]),
                segment_id=segment_id,
                confidence=float(raw_edge["confidence"]),
                modalities=raw_edge["modalities"],
                provenance=[
                    ProvenanceRecord(
                        source="minicpm_alignment",
                        video_id=video_id,
                        segment_id=segment_id,
                        start=float(segment["start"]),
                        end=float(segment["end"]),
                        confidence=float(raw_edge["confidence"]),
                    )
                ],
            )
            edge_payload = edge.to_dict()
            edge_payload["storage_id"] = segment.get(
                "storage_id",
                f"{video_id}_{segment.get('index', index)}",
            )
            edges.append(edge_payload)

        for claim in text_result.get("claims", []):
            claim_global = registry.ensure_entity(
                "claim",
                claim["text"],
                source="transcript",
                confidence=float(claim.get("confidence", 0.0)),
            )
            registry.add_alias(
                claim["text_id"],
                claim_global,
                source="transcript",
                label=claim["text"],
                confidence=float(claim.get("confidence", 0.0)),
                segment_id=segment_id,
            )
            registry.add_provenance(
                claim_global,
                ProvenanceRecord(
                    source="transcript",
                    video_id=video_id,
                    segment_id=segment_id,
                    start=float(claim["start"]),
                    end=float(claim["end"]),
                    text=claim["text"],
                    confidence=float(claim.get("confidence", 0.0)),
                ),
            )
            speaker_alias = str(claim.get("speaker_id") or "SPEAKER_UNKNOWN")
            speaker_global = (
                accepted_speaker_map.get(speaker_alias)
                or registry.resolve(speaker_alias)
            )
            if not speaker_global:
                speaker_global = registry.ensure_entity(
                    "speaker",
                    "unattributed speaker",
                    source="transcript",
                    confidence=0.25,
                )
                registry.add_alias(
                    speaker_alias,
                    speaker_global,
                    source="transcript",
                    label="unattributed speaker",
                    confidence=0.25,
                    segment_id=segment_id,
                )
            registry.add_provenance(
                speaker_global,
                ProvenanceRecord(
                    source="audio",
                    video_id=video_id,
                    segment_id=segment_id,
                    start=float(claim["start"]),
                    end=float(claim["end"]),
                    text=speaker_alias,
                    confidence=(
                        0.90 if speaker_alias in accepted_speaker_map else 0.25
                    ),
                ),
            )
            edge_counter += 1
            claim_edge = EdgeOccurrence(
                    edge_id=_edge_id(video_id, edge_counter),
                    source_id=speaker_global,
                    target_id=claim_global,
                    predicate="says",
                    start=float(claim["start"]),
                    end=float(claim["end"]),
                    segment_id=segment_id,
                    confidence=float(claim.get("confidence", 0.0)),
                    modalities=["transcript"],
                    provenance=[
                        ProvenanceRecord(
                            source="transcript",
                            video_id=video_id,
                            segment_id=segment_id,
                            start=float(claim["start"]),
                            end=float(claim["end"]),
                            text=claim["text"],
                            confidence=float(claim.get("confidence", 0.0)),
                        )
                    ],
                ).to_dict()
            claim_edge["storage_id"] = segment.get(
                "storage_id",
                f"{video_id}_{segment.get('index', index)}",
            )
            edges.append(claim_edge)

        caption = _rewrite_ids(alignment["caption"], registry)
        rewritten_transcript = _rewrite_ids(
            text_result.get("rewritten_transcript", ""),
            registry,
        )
        recent_events.extend(
            [
                {
                    "source": edge["source_id"],
                    "predicate": edge["predicate"],
                    "target": edge["target_id"],
                    "segment_id": segment_id,
                }
                for edge in edges
            ]
        )
        recent_events = recent_events[-int(config.get("entity_memory_recent_events", 8)) :]
        memory_delta = []
        for edge in edges:
            relation = {
                "predicate": edge["predicate"],
                "target": edge["target_id"],
                "segment_id": segment_id,
                "time": [edge["start"], edge["end"]],
            }
            long_term_state.setdefault(edge["source_id"], []).append(relation)
            long_term_state[edge["source_id"]] = long_term_state[
                edge["source_id"]
            ][-5:]
            memory_delta.append(
                f"{edge['source_id']} {edge['predicate']} {edge['target_id']}"
            )
        segment_result = {
            "segment_id": segment_id,
            "caption": caption,
            "rewritten_transcript": rewritten_transcript,
            "edges": edges,
            "accepted_merges": alignment["entity_merges"],
            "candidate_merges": candidates,
            "frame_paths": frames,
            "mode": profile_map.get(segment_id, {}).get("mode", "balanced"),
            "memory_delta": memory_delta,
        }
        atomic_write_json(checkpoint_path / f"{segment_id}.json", segment_result)
        results[segment_id] = segment_result
        atomic_write_json(
            state_path,
            {
                "next_index": index + 1,
                "edge_counter": edge_counter,
                "recent_events": recent_events,
                "long_term_state": long_term_state,
                "registry": registry.to_dict(),
            },
        )
        if progress is not None:
            progress.set(
                index + 1,
                entities=len(registry.entities),
                edges=edge_counter,
            )

    return {
        "segments": results,
        "registry": registry.to_dict(),
        "edge_count": edge_counter,
        "recent_events": recent_events,
        "long_term_state": long_term_state,
    }
