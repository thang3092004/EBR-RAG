from __future__ import annotations

import importlib.util as _iutil
import json
import gc
import logging
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _detect_attn_impl() -> str:
    spec = _iutil.find_spec("flash_attn")
    if spec is not None and spec.origin is not None:
        return "flash_attention_2"
    return "sdpa"

from PIL import Image

from ..pipeline.stage_runner import atomic_write_json, read_json
from .registry import EntityRegistry, normalize_alias
from .schema import EdgeOccurrence, ProvenanceRecord, bbox_position


GENERIC_VISUAL_LABELS = {
    "person", "animal", "object", "vehicle", "plant",
    "food", "sports", "outdoor", "indoor",
}


VISUAL_CAPTION_PROMPT = """Describe what is happening in this video segment.

Your description must include:
1. All events and actions occurring in the scene
2. The overall context and atmosphere
3. For each visible person or distinct object, describe:
   - Appearance (clothing color, size, distinguishing features)
   - Position in frame (e.g. "standing on the left", "in the foreground center")
   - What they are doing

Be specific about spatial relationships between entities and their positions.
"""


CROSSMODAL_MERGE_SYSTEM = """You are building an entity knowledge graph from a video. \
You match visual tracked objects to transcript entities, extract relationships, \
and maintain an accumulative entity memory across batches. Return valid JSON only."""


CROSSMODAL_MERGE_PROMPT = """=== KNOWN ENTITIES (accumulated so far) ===
{entity_memory_json}

=== VISUAL ENTITIES FROM OBJECT TRACKING (this batch) ===
{visual_entities_with_positions}

=== TRANSCRIPT ENTITIES (this batch) ===
{transcript_entities_with_names}

=== SEGMENTS TO PROCESS ===
{segments_data}

=== YOUR TASKS ===

**Task 1 — Entity matching:**
For each visual entity (V_xxx) that appears in a segment, match it to a transcript
entity (T_xxx) or known entity from memory, using:
- Appearance description in the visual caption vs canonical_name
- Position in frame (V_ entity position vs caption description like "on the left")
- Temporal co-occurrence (both appear in the same segment timeframe)

Return matches as a list:
{{"visual_id": "V_ANIMAL_001", "text_id": "T_ANIMAL_003", "reason": "..."}}
Use null for text_id if no transcript match is found.

**Task 2 — Relationship extraction:**
Extract entity relationships and events from each segment.
Use canonical IDs where known (from matches + memory), provisional IDs otherwise.
Format each as:
{{"source": "ANIMAL_001", "predicate": "chased_by", "target": "ANIMAL_015",
  "segment_id": "SEG_007", "confidence": 0.7,
  "modalities": ["visual", "transcript"],
  "description": "predator chases prey at high speed"}}

**Task 3 — Memory update:**
For each known/matched entity, provide updated description for this batch.
Format: {{"entity_id": "ANIMAL_001", "new_description": "fleeing from predator (seg7)"}}

Return valid JSON with keys: "matches", "relationships", "memory_updates"
"""


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("Response does not contain a JSON object.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def _load_frame_images(frame_paths: list[str]) -> list[Image.Image]:
    return [
        Image.open(path).convert("RGB")
        for path in frame_paths
        if Path(path).exists()
    ]


class MiniCPMCaptioner:
    """MiniCPM visual captioning — plain text output, no entity IDs."""

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
                "MiniCPM captioning requires torch and transformers."
            ) from exc
        model_path = Path(self.model_path)
        if self.config.get("pipeline_strict", False) and not model_path.is_dir():
            raise FileNotFoundError(
                "Strict pipeline requires a local MiniCPM model directory: "
                f"{model_path.resolve()}"
            )
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
                self.config.get("caption_attention", _detect_attn_impl())
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
        )
        self.model.eval()

    def _chat_text(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        max_tokens: int,
        max_slice_nums: int | None = None,
    ) -> str:
        effective_slices = max_slice_nums if max_slice_nums is not None else int(
            self.config.get("caption_max_slice_nums", 2)
        )
        content = images + [prompt]
        messages = [{"role": "user", "content": content}]
        response = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            use_image_id=False,
            max_slice_nums=effective_slices,
            max_new_tokens=max_tokens,
        )
        return str(response).strip()

    def _chat_batch_text(
        self,
        prompts: list[str],
        images_list: list[list[Image.Image]],
        *,
        max_tokens: int,
        max_slice_nums: int | None = None,
    ) -> list[str]:
        import torch

        effective_slices = max_slice_nums if max_slice_nums is not None else int(
            self.config.get("caption_max_slice_nums", 2)
        )
        msgs = [
            [{"role": "user", "content": imgs + [prompt]}]
            for imgs, prompt in zip(images_list, prompts)
        ]
        try:
            responses = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=effective_slices,
                max_new_tokens=max_tokens,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            return [
                self._chat_text(
                    prompts[i],
                    images_list[i],
                    max_tokens=max_tokens,
                    max_slice_nums=max_slice_nums,
                )
                for i in range(len(prompts))
            ]
        return [str(r).strip() for r in responses]

    def caption(
        self,
        frame_paths: list[str],
        *,
        max_tokens: int | None = None,
        max_slice_nums: int | None = None,
    ) -> str:
        self.load()
        effective = max_tokens if max_tokens is not None else int(
            self.config.get("caption_max_tokens", 450)
        )
        images = _load_frame_images(frame_paths)
        try:
            return self._chat_text(
                VISUAL_CAPTION_PROMPT, images, max_tokens=effective,
                max_slice_nums=max_slice_nums,
            )
        finally:
            for image in images:
                image.close()

    def caption_preloaded(
        self,
        images: list[Image.Image],
        *,
        max_tokens: int | None = None,
        max_slice_nums: int | None = None,
    ) -> str:
        self.load()
        effective = max_tokens if max_tokens is not None else int(
            self.config.get("caption_max_tokens", 450)
        )
        return self._chat_text(
            VISUAL_CAPTION_PROMPT, images, max_tokens=effective,
            max_slice_nums=max_slice_nums,
        )

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


# ---- keep backward-compatible alias for unified_ingest.py imports ----
MiniCPMAligner = MiniCPMCaptioner


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


def _edge_id(video_id: str, counter: int) -> str:
    safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
    return f"EDGE_{safe_video}_{counter:07d}"


def _deduplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (edge["source"], edge["predicate"], edge["target"])
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = dict(edge)
            continue
        existing["confidence"] = max(
            float(existing["confidence"]),
            float(edge["confidence"]),
        )
        existing["modalities"] = sorted(
            set(existing.get("modalities", []))
            | set(edge.get("modalities", []))
        )
    return list(deduplicated.values())


# =====================================================================
# Sub-step 8a — MiniCPM Visual Captioning (parallel, stateless)
# =====================================================================

def run_visual_captioning(
    video_id: str,
    segments: list[dict[str, Any]],
    frame_selections: dict[str, dict[str, Any]],
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    captioner: MiniCPMCaptioner | None = None,
) -> dict[str, str]:
    """Run MiniCPM captioning for all segments. Returns {segment_id: caption_text}."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    captions_path = checkpoint_path / "captions.json"
    captions: dict[str, str] = dict(read_json(captions_path, {}))

    def _frame_paths(seg: dict[str, Any]) -> list[str]:
        return [
            item["path"]
            for item in frame_selections.get(
                str(seg["segment_id"]), {}
            ).get("frames", [])
        ]

    segments_todo = [
        seg for seg in segments
        if str(seg["segment_id"]) not in captions
    ]

    if not segments_todo:
        return captions

    captioner = captioner or MiniCPMCaptioner(config)
    captioner.load()

    batch_size = int(config.get("caption_visual_batch_size", 2))
    max_tokens = int(config.get("caption_max_tokens", 450))
    max_slice_nums = int(config.get("caption_visual_slice_nums", 1))
    batches = [
        segments_todo[i: i + batch_size]
        for i in range(0, len(segments_todo), batch_size)
    ]
    n_done = len(segments) - len(segments_todo)

    with (
        tqdm(
            total=len(segments),
            initial=n_done,
            unit="seg",
            desc=f"[{video_id}] visual caption",
            dynamic_ncols=True,
        ) as pbar,
        ThreadPoolExecutor(max_workers=batch_size) as executor,
    ):
        pending_futures: list[Future[list[Image.Image]]] = [
            executor.submit(_load_frame_images, _frame_paths(seg))
            for seg in batches[0]
        ]

        for bi, batch in enumerate(batches):
            batch_images = [f.result() for f in pending_futures]

            if bi + 1 < len(batches):
                pending_futures = [
                    executor.submit(_load_frame_images, _frame_paths(seg))
                    for seg in batches[bi + 1]
                ]
            else:
                pending_futures = []

            active_idx: list[int] = []
            active_images_list: list[list[Image.Image]] = []
            for local_i, (seg, images) in enumerate(zip(batch, batch_images)):
                if images:
                    active_idx.append(local_i)
                    active_images_list.append(images)

            if active_idx:
                batch_results = captioner._chat_batch_text(
                    [VISUAL_CAPTION_PROMPT] * len(active_idx),
                    active_images_list,
                    max_tokens=max_tokens,
                    max_slice_nums=max_slice_nums,
                )
            else:
                batch_results = []

            result_iter = iter(zip(active_idx, batch_results))
            next_active = next(result_iter, None)
            for local_i, (seg, images) in enumerate(zip(batch, batch_images)):
                segment_id = str(seg["segment_id"])
                if next_active is not None and next_active[0] == local_i:
                    _, caption_text = next_active
                    next_active = next(result_iter, None)
                else:
                    caption_text = ""
                for img in images:
                    img.close()
                captions[segment_id] = caption_text

            pbar.update(len(batch))
            atomic_write_json(captions_path, captions)

    return captions


# =====================================================================
# Sub-step 8b — Cross-modal Entity Merge (GPT-4o-mini, sequential)
# =====================================================================

def _filter_entity_memory(
    entity_memory: dict[str, Any],
    active_entity_ids: set[str],
    recent_segment_ids: set[str],
    *,
    global_threshold: int = 5,
    max_entities: int = 25,
) -> dict[str, Any]:
    keep: dict[str, int] = {}

    for eid in active_entity_ids:
        if eid in entity_memory:
            keep[eid] = 1

    for eid in list(keep.keys()):
        for rel_str in entity_memory.get(eid, {}).get("relationships", []):
            parts = rel_str.split()
            if len(parts) >= 2:
                neighbor_id = parts[1].strip("(),.")
                if neighbor_id in entity_memory and neighbor_id not in keep:
                    keep[neighbor_id] = 2

    for eid, data in entity_memory.items():
        if eid not in keep:
            if any(s in recent_segment_ids for s in data.get("segments_seen", [])):
                keep[eid] = 3

    for eid, data in entity_memory.items():
        if eid not in keep:
            if len(data.get("segments_seen", [])) >= global_threshold:
                keep[eid] = 4

    sorted_eids = sorted(keep.keys(), key=lambda e: keep[e])[:max_entities]
    return {eid: entity_memory[eid] for eid in sorted_eids if eid in entity_memory}


def _build_merge_prompt(
    batch_segments: list[dict[str, Any]],
    captions: dict[str, str],
    visual_entities: list[dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    entity_memory: dict[str, Any],
    *,
    transcripts: dict[str, str] | None = None,
    recent_segment_ids: set[str] | None = None,
    max_memory_entities: int = 25,
) -> str:
    from collections import Counter

    transcripts = transcripts or {}

    _active_ids: set[str] = set()
    for _seg in batch_segments:
        _sid = str(_seg["segment_id"])
        for _obs in observations:
            if str(_obs.get("segment_id")) == _sid:
                _active_ids.add(str(_obs["entity_id"]))
        for _mention in text_results.get(_sid, {}).get("mentions", []):
            _active_ids.add(_mention["text_id"])

    filtered_memory = _filter_entity_memory(
        entity_memory,
        _active_ids,
        recent_segment_ids or set(),
        max_entities=max_memory_entities,
    )
    entity_memory_json = json.dumps(filtered_memory, ensure_ascii=False) if filtered_memory else "{}"

    ve_position: dict[str, str] = {
        ve["entity_id"]: ve.get("dominant_position") or "unknown"
        for ve in visual_entities
    }

    batch_visual_ids: set[str] = set()
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        for obs in observations:
            if str(obs.get("segment_id")) == seg_id:
                batch_visual_ids.add(str(obs["entity_id"]))

    v_lines = []
    for ve in visual_entities:
        eid = ve["entity_id"]
        if eid not in batch_visual_ids:
            continue
        pos = ve_position.get(eid, "unknown")
        v_lines.append(
            f"- {eid}: {ve['entity_type']} ({ve['label']}), "
            f"dominant position across video: {pos}"
        )
    visual_entities_str = "\n".join(v_lines) if v_lines else "(none)"

    t_seen: dict[str, dict[str, Any]] = {}
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        for mention in text_results.get(seg_id, {}).get("mentions", []):
            tid = mention["text_id"]
            if tid not in t_seen:
                t_seen[tid] = mention

    t_lines = []
    for tid, mention in t_seen.items():
        surface = (
            mention["occurrences"][0]["text"]
            if mention.get("occurrences")
            else mention["label"]
        )
        start = mention.get("start", 0.0)
        t_lines.append(
            f"- {tid}: label=\"{mention['label']}\", type={mention['entity_type']}, "
            f"surface_text=\"{surface}\", first_seen={start:.1f}s"
        )
    transcript_entities_str = "\n".join(t_lines) if t_lines else "(none)"

    segments_data_parts = []
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        caption = captions.get(seg_id, "")
        raw_transcript = transcripts.get(seg_id, "")

        seg_obs_by_entity: dict[str, list[dict[str, Any]]] = {}
        for obs in observations:
            if str(obs.get("segment_id")) == seg_id:
                eid = str(obs["entity_id"])
                seg_obs_by_entity.setdefault(eid, []).append(obs)

        seg_v_parts = []
        for eid, obs_list in sorted(seg_obs_by_entity.items()):
            pos_counts: Counter = Counter()
            for obs in obs_list:
                p = bbox_position(obs.get("bbox"))
                if p:
                    pos_counts[p] += 1
            seg_pos = pos_counts.most_common(1)[0][0] if pos_counts else ve_position.get(eid, "unknown")
            ve_info = next((ve for ve in visual_entities if ve["entity_id"] == eid), {})
            seg_v_parts.append(
                f"{eid} ({ve_info.get('label', '?')}, pos_in_segment: {seg_pos})"
            )
        seg_v_str = ", ".join(seg_v_parts) if seg_v_parts else "(none)"

        seg_t_parts = []
        for mention in text_results.get(seg_id, {}).get("mentions", []):
            surface = (
                mention["occurrences"][0]["text"]
                if mention.get("occurrences")
                else mention["label"]
            )
            seg_t_parts.append(
                f"{mention['text_id']} (\"{surface}\", {mention['entity_type']})"
            )
        seg_t_str = ", ".join(seg_t_parts) if seg_t_parts else "(none)"

        segments_data_parts.append(
            f"SEGMENT {seg_id} [{seg['start']:.1f}s - {seg['end']:.1f}s]:\n"
            f"Visual caption: {caption}\n"
            f"Transcript: {raw_transcript}\n"
            f"Visual entities in this segment: {seg_v_str}\n"
            f"Transcript entities in this segment: {seg_t_str}"
        )
    segments_data_str = "\n\n".join(segments_data_parts)

    return CROSSMODAL_MERGE_PROMPT.format(
        entity_memory_json=entity_memory_json,
        visual_entities_with_positions=visual_entities_str,
        transcript_entities_with_names=transcript_entities_str,
        segments_data=segments_data_str,
    )


def _validate_merge_result(
    raw: dict[str, Any],
    visual_ids: set[str],
    text_ids: set[str],
    registry_entity_ids: set[str] | None = None,
) -> dict[str, Any]:
    registry_entity_ids = registry_entity_ids or set()
    matches = []
    for match in raw.get("matches", []):
        vid = str(match.get("visual_id", "")).strip()
        tid = match.get("text_id")
        if tid is not None:
            tid = str(tid).strip()
        if vid in visual_ids:
            if tid is None or tid in text_ids or tid in registry_entity_ids:
                matches.append({
                    "visual_id": vid,
                    "text_id": tid,
                    "reason": str(match.get("reason", "")),
                })

    relationships = []
    for rel in raw.get("relationships", []):
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        predicate = str(rel.get("predicate", "")).strip().lower().replace(" ", "_")
        if source and target and predicate and source != target:
            relationships.append({
                "source": source,
                "target": target,
                "predicate": predicate,
                "segment_id": str(rel.get("segment_id", "")),
                "confidence": max(0.0, min(float(rel.get("confidence", 0.5)), 1.0)),
                "modalities": [
                    m for m in rel.get("modalities", ["visual", "transcript"])
                    if m in {"visual", "transcript"}
                ] or ["visual", "transcript"],
                "description": str(rel.get("description", "")),
            })

    memory_updates = []
    for update in raw.get("memory_updates", []):
        eid = str(update.get("entity_id", "")).strip()
        desc = str(update.get("new_description", "")).strip()
        if eid and desc:
            memory_updates.append({"entity_id": eid, "new_description": desc})

    return {
        "matches": matches,
        "relationships": relationships,
        "memory_updates": memory_updates,
    }


async def _call_gpt4o_mini(prompt: str, system_prompt: str) -> str:
    from .._llm import openai_complete_if_cache
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
    )


def run_crossmodal_merge(
    video_id: str,
    segments: list[dict[str, Any]],
    captions: dict[str, str],
    visual_entities: list[dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    initial_registry: EntityRegistry,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    loop=None,
    progress=None,
    transcripts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run cross-modal entity merge using GPT-4o-mini in sequential batches."""
    import asyncio

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_path / "merge_state.json"
    state = read_json(state_path, {})
    registry = (
        EntityRegistry(state["registry"])
        if state.get("registry")
        else initial_registry
    )
    entity_memory: dict[str, Any] = dict(state.get("entity_memory", {}))
    next_batch = int(state.get("next_batch", 0))
    edge_counter = int(state.get("edge_counter", 0))

    batch_size = int(config.get("crossmodal_batch_size", 8))
    crossmodal_enabled = not bool(config.get("disable_crossmodal_alignment", False))

    batches = [
        segments[i: i + batch_size]
        for i in range(0, len(segments), batch_size)
    ]

    results: dict[str, dict[str, Any]] = dict(state.get("segment_results", {}))

    if loop is None:
        loop = asyncio.new_event_loop()
        owns_loop = True
    else:
        owns_loop = False

    try:
        for bi in range(next_batch, len(batches)):
            batch = batches[bi]

            if not crossmodal_enabled:
                for seg in batch:
                    seg_id = str(seg["segment_id"])
                    results[seg_id] = {
                        "segment_id": seg_id,
                        "caption": captions.get(seg_id, ""),
                        "edges": [],
                        "accepted_merges": [],
                    }
                    for mention in text_results.get(seg_id, {}).get("mentions", []):
                        global_id = registry.resolve(mention["text_id"])
                        if not global_id:
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
                                segment_id=seg_id,
                            )
                        _add_mention_provenance(
                            registry, global_id, mention, video_id, seg_id,
                        )
                if progress is not None:
                    progress.set(
                        sum(len(b) for b in batches[: bi + 1]),
                        entities=len(registry.entities),
                        edges=edge_counter,
                    )
                continue

            all_segment_ids = [str(s["segment_id"]) for s in segments]
            _first_seg_id = str(batch[0]["segment_id"]) if batch else ""
            _first_idx = all_segment_ids.index(_first_seg_id) if _first_seg_id in all_segment_ids else 0
            _recent_segment_ids: set[str] = set()
            for i in range(_first_idx - 1, max(_first_idx - 21, -1), -1):
                seg = segments[i]
                if seg.get("has_shot_at_start", False):
                    break
                _recent_segment_ids.add(str(seg["segment_id"]))

            prompt = _build_merge_prompt(
                batch, captions, visual_entities, text_results,
                observations, entity_memory,
                transcripts=transcripts or {},
                recent_segment_ids=_recent_segment_ids,
                max_memory_entities=int(config.get("entity_memory_max_context", 25)),
            )

            try:
                raw_response = loop.run_until_complete(
                    _call_gpt4o_mini(prompt, CROSSMODAL_MERGE_SYSTEM)
                )
                raw_result = _extract_json(raw_response)
            except Exception as exc:
                logger.warning(
                    "crossmodal_merge batch %d failed (video=%s): %s",
                    bi, video_id, exc,
                )
                if config.get("pipeline_strict", False):
                    raise
                raw_result = {"matches": [], "relationships": [], "memory_updates": []}

            batch_visual_ids: set[str] = set()
            batch_text_ids: set[str] = set()
            for seg in batch:
                seg_id = str(seg["segment_id"])
                for obs in observations:
                    if str(obs.get("segment_id")) == seg_id:
                        batch_visual_ids.add(str(obs["entity_id"]))
                for mention in text_results.get(seg_id, {}).get("mentions", []):
                    batch_text_ids.add(mention["text_id"])

            validated = _validate_merge_result(
                raw_result, batch_visual_ids, batch_text_ids,
                registry_entity_ids=set(registry.entities.keys()),
            )

            for match in validated["matches"]:
                if match["text_id"] is None:
                    continue
                mention = None
                for seg in batch:
                    seg_id = str(seg["segment_id"])
                    for m in text_results.get(seg_id, {}).get("mentions", []):
                        if m["text_id"] == match["text_id"]:
                            mention = m
                            break
                    if mention:
                        break
                if mention:
                    registry.add_alias(
                        match["text_id"],
                        match["visual_id"],
                        source="crossmodal_merge",
                        label=mention["label"],
                        confidence=float(mention.get("confidence", 0.0)),
                        segment_id=seg_id,
                    )
                    node = registry.entities.get(match["visual_id"])
                    if node and mention.get("label"):
                        if (
                            node.canonical_name in GENERIC_VISUAL_LABELS
                            or node.canonical_name == node.entity_id
                        ):
                            node.canonical_name = mention["label"]
                    if match["visual_id"] in entity_memory and mention.get("label"):
                        entity_memory[match["visual_id"]]["canonical_name"] = mention["label"]

                if mention is None and match["text_id"] in registry.entities:
                    memory_node = registry.entities[match["text_id"]]
                    vis_node = registry.entities.get(match["visual_id"])
                    if vis_node:
                        registry.add_alias(
                            match["text_id"],
                            match["visual_id"],
                            source="crossmodal_merge",
                            label=memory_node.canonical_name or vis_node.canonical_name,
                            confidence=0.7,
                            segment_id=str(batch[0]["segment_id"]),
                        )
                        old_mem = entity_memory.pop(match["text_id"], None)
                        if old_mem:
                            new_mem = entity_memory.setdefault(match["visual_id"], {
                                "canonical_name": "",
                                "accumulated_descriptions": [],
                                "relationships": [],
                                "segments_seen": [],
                            })
                            new_mem["accumulated_descriptions"] = (
                                old_mem.get("accumulated_descriptions", [])
                                + new_mem["accumulated_descriptions"]
                            )[-10:]
                            for r in old_mem.get("relationships", []):
                                if r not in new_mem["relationships"]:
                                    new_mem["relationships"].append(r)
                            new_mem["relationships"] = new_mem["relationships"][-10:]
                            for s in old_mem.get("segments_seen", []):
                                if s not in new_mem["segments_seen"]:
                                    new_mem["segments_seen"].append(s)
                    if (
                        vis_node
                        and memory_node.canonical_name
                        and memory_node.canonical_name not in GENERIC_VISUAL_LABELS
                        and memory_node.canonical_name != vis_node.entity_id
                    ):
                        if (
                            vis_node.canonical_name in GENERIC_VISUAL_LABELS
                            or vis_node.canonical_name == vis_node.entity_id
                        ):
                            vis_node.canonical_name = memory_node.canonical_name
                        if match["visual_id"] in entity_memory:
                            entity_memory[match["visual_id"]]["canonical_name"] = (
                                memory_node.canonical_name
                            )

            for seg in batch:
                seg_id = str(seg["segment_id"])
                for mention in text_results.get(seg_id, {}).get("mentions", []):
                    global_id = registry.resolve(mention["text_id"])
                    if not global_id:
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
                            segment_id=seg_id,
                        )
                    _add_mention_provenance(
                        registry, global_id, mention, video_id, seg_id,
                    )

            for update in validated["memory_updates"]:
                eid = update["entity_id"]
                canonical = registry.resolve(eid) or eid
                entry = entity_memory.setdefault(canonical, {
                    "canonical_name": "",
                    "accumulated_descriptions": [],
                    "relationships": [],
                    "segments_seen": [],
                })
                node = registry.entities.get(canonical)
                if node:
                    entry["canonical_name"] = node.canonical_name
                entry["accumulated_descriptions"].append(update["new_description"])
                entry["accumulated_descriptions"] = entry["accumulated_descriptions"][-10:]
                for seg in batch:
                    seg_id = str(seg["segment_id"])
                    seg_has_entity = any(
                        str(obs.get("entity_id")) == canonical
                        for obs in observations
                        if str(obs.get("segment_id")) == seg_id
                    ) or any(
                        m["text_id"] == eid or registry.resolve(m["text_id"]) == canonical
                        for m in text_results.get(seg_id, {}).get("mentions", [])
                    )
                    if seg_has_entity and seg_id not in entry["segments_seen"]:
                        entry["segments_seen"].append(seg_id)

            seg_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
            dropped_rels: list[dict[str, str]] = []
            for rel in validated["relationships"]:
                source = registry.resolve(rel["source"]) or rel["source"]
                target = registry.resolve(rel["target"]) or rel["target"]
                if source not in registry.entities or target not in registry.entities:
                    dropped_rels.append({
                        "batch": str(bi),
                        "raw_source": rel["source"],
                        "raw_target": rel["target"],
                        "resolved_source": source,
                        "resolved_target": target,
                        "predicate": rel.get("predicate", ""),
                    })
                    continue
                edge_counter += 1
                seg_id = rel.get("segment_id", "")
                seg_for_time = next(
                    (s for s in batch if str(s["segment_id"]) == seg_id),
                    batch[0],
                )
                edge = EdgeOccurrence(
                    edge_id=_edge_id(video_id, edge_counter),
                    source_id=source,
                    target_id=target,
                    predicate=rel["predicate"],
                    start=float(seg_for_time["start"]),
                    end=float(seg_for_time["end"]),
                    segment_id=seg_id,
                    confidence=float(rel["confidence"]),
                    modalities=rel["modalities"],
                    description=rel.get("description", ""),
                    provenance=[
                        ProvenanceRecord(
                            source="gpt4o_mini_crossmodal",
                            video_id=video_id,
                            segment_id=seg_id,
                            start=float(seg_for_time["start"]),
                            end=float(seg_for_time["end"]),
                            confidence=float(rel["confidence"]),
                        )
                    ],
                )
                edge_payload = edge.to_dict()
                edge_payload["storage_id"] = seg_for_time.get(
                    "storage_id",
                    f"{video_id}_{seg_for_time.get('index', 0)}",
                )
                seg_edges[seg_id].append(edge_payload)

                mem_entry = entity_memory.setdefault(source, {
                    "canonical_name": "",
                    "accumulated_descriptions": [],
                    "relationships": [],
                    "segments_seen": [],
                })
                mem_entry["relationships"].append(
                    f"{rel['predicate']} {target} ({seg_id})"
                )
                mem_entry["relationships"] = mem_entry["relationships"][-10:]
                if seg_id and seg_id not in mem_entry["segments_seen"]:
                    mem_entry["segments_seen"].append(seg_id)

            if dropped_rels:
                logger.info(
                    "batch %d: %d/%d relationships dropped (unresolved IDs)",
                    bi, len(dropped_rels), len(dropped_rels) + sum(len(v) for v in seg_edges.values()),
                )
                drops_path = checkpoint_path / "relationship_drops.jsonl"
                with open(drops_path, "a", encoding="utf-8") as f:
                    for d in dropped_rels:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")

            for seg in batch:
                seg_id = str(seg["segment_id"])
                edges = seg_edges.get(seg_id, [])
                entity_ids = sorted({
                    node
                    for e in edges
                    for node in (e["source_id"], e["target_id"])
                })
                results[seg_id] = {
                    "segment_id": seg_id,
                    "caption": captions.get(seg_id, ""),
                    "edges": edges,
                    "entity_ids": entity_ids,
                    "accepted_merges": [
                        m for m in validated["matches"]
                        if m["text_id"] is not None
                    ],
                }

            atomic_write_json(
                state_path,
                {
                    "next_batch": bi + 1,
                    "edge_counter": edge_counter,
                    "entity_memory": entity_memory,
                    "registry": registry.to_dict(),
                    "segment_results": results,
                },
            )
            if progress is not None:
                progress.set(
                    sum(len(b) for b in batches[: bi + 1]),
                    entities=len(registry.entities),
                    edges=edge_counter,
                    batch=f"{bi + 1}/{len(batches)}",
                )
    finally:
        if owns_loop:
            loop.close()

    return {
        "segments": results,
        "registry": registry.to_dict(),
        "edge_count": edge_counter,
        "entity_memory": entity_memory,
    }


def _add_mention_provenance(
    registry: EntityRegistry,
    global_id: str,
    mention: dict[str, Any],
    video_id: str,
    segment_id: str,
) -> None:
    occurrences = mention.get("occurrences") or [
        {
            "start": mention["start"],
            "end": mention["end"],
            "text": mention["label"],
            "confidence": mention.get("confidence", 0.0),
        }
    ]
    for occurrence in occurrences:
        registry.add_provenance(
            global_id,
            ProvenanceRecord(
                source="transcript",
                video_id=video_id,
                segment_id=segment_id,
                start=float(occurrence["start"]),
                end=float(occurrence["end"]),
                text=str(occurrence.get("text", mention["label"])),
                confidence=float(
                    occurrence.get("confidence", mention.get("confidence", 0.0))
                ),
            ),
        )


# =====================================================================
# Main entry point — align_all_segments (8a + 8b combined)
# =====================================================================

def align_all_segments(
    video_id: str,
    segments: list[dict[str, Any]],
    frame_selections: dict[str, dict[str, Any]],
    text_results: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    initial_registry: EntityRegistry,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    progress=None,
    aligner: MiniCPMCaptioner | None = None,
    visual_entities: list[dict[str, Any]] | None = None,
    loop=None,
    transcripts: dict[str, str] | None = None,
    # legacy kwargs kept for backward compat — ignored
    correspondence_encoder=None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # ---- Sub-step 8a: MiniCPM visual captioning ----
    captions = run_visual_captioning(
        video_id,
        segments,
        frame_selections,
        config,
        checkpoint_path,
        captioner=aligner,
    )

    if aligner is not None:
        aligner.close()

    # ---- Sub-step 8b: GPT-4o-mini cross-modal merge ----
    result = run_crossmodal_merge(
        video_id,
        segments,
        captions,
        visual_entities or [],
        text_results,
        observations,
        initial_registry,
        config,
        checkpoint_path,
        loop=loop,
        progress=progress,
        transcripts=transcripts,
    )

    return result
