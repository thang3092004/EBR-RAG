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


VISUAL_CAPTION_PROMPT = """Describe what is visible in this video segment in 2-4 concise sentences.

Rules:
- State FACTS only: who/what is visible, what they are doing, where.
- Do NOT speculate or hedge (no "possibly", "suggesting", "likely", "indicating").
- Do NOT describe mood, atmosphere, or make interpretive comments.
- Do NOT start with "The video segment shows/captures/depicts".
- Use specific names if visible in text overlays or recognizable (e.g. species names).
- For each distinct person or animal: note appearance and position in frame.
- Maximum 100 words.
"""


# =====================================================================
# GPT-4o-mini prompts — GraphRAG/LightRAG style unified extraction
# =====================================================================

GRAPHRAG_SYSTEM = """---Role---
You are a Knowledge Graph Specialist extracting entities and relationships from video content.
You receive visual captions (what is seen) and audio transcripts (what is heard) for each video segment.
You maintain an accumulative entity memory across processing batches.
Return valid JSON only."""


GRAPHRAG_PROMPT = """---Known Entities---
Entities accumulated from previous batches. Reuse their EXACT names when encountered again.

{entity_memory_json}

---Entity Types---
- Person: Named or role-identified humans (e.g. DAVID ATTENBOROUGH, NARRATOR, RESEARCHER)
- Animal: Living creatures (e.g. LION, SCARFACE, DOLPHIN). Use SINGULAR form.
- Object: Physical objects relevant to narrative (e.g. NEST, FISH, ROCK)
- Location: Geographic places, habitats (e.g. SAVANNA, CORAL REEF, RIVER)

---Segments---
{segments_data}

---Task---
Extract entities and relationships from ALL segments above.

**1. Entity Extraction:**
For each meaningful entity found in visual captions OR transcripts:
- "entity_name": Canonical name, CAPITALIZED, SINGULAR. Use specific names from transcript \
when available (SCARFACE not MONKEY). For unnamed people use roles (NARRATOR, DIVER). \
If entity matches one in Known Entities, use the EXACT same name.
- "entity_type": One of [person, animal, object, location]
- "entity_description": Concise description of appearance, role, distinguishing features. \
Third person. No pronouns.
- "source_segments": List of segment IDs where this entity appears

Do NOT extract: video production elements (the scene, the camera, BBC Earth), \
abstract concepts (atmosphere, tension).
If same entity has different names in caption vs transcript, unify under ONE name.

**2. Relationship Extraction:**
For each pair of clearly related entities:
- "source_entity": Must match an extracted entity name
- "target_entity": Must match an extracted entity name
- "predicate": Simple present-tense base verb (chase, attack, inhabit, defend)
- "keywords": High-level thematic keywords, comma-separated (e.g. "predation, survival")
- "description": Brief factual explanation
- "segment_id": Where observed
- "confidence": 0.0 to 1.0

Do NOT create relationships for video editing transitions. Do NOT duplicate relationships.

**3. Memory Update:**
For entities in Known Entities whose state changed:
- "entity_name": EXACT match from Known Entities
- "new_description": What is NEW (do not repeat old descriptions)

---Output Format---
{{
  "entities": [
    {{"entity_name": "LION", "entity_type": "animal", "entity_description": "Adult male lion with full mane, dominant predator in the pride", "source_segments": ["SEG_00040", "SEG_00041"]}}
  ],
  "relationships": [
    {{"source_entity": "LION", "target_entity": "BUFFALO", "predicate": "hunt", "keywords": "predation, survival", "description": "The lion stalks and chases the buffalo herd", "segment_id": "SEG_00040", "confidence": 0.9}}
  ],
  "memory_updates": [
    {{"entity_name": "LION", "new_description": "Coordinates group attack on buffalo near the river"}}
  ]
}}
"""


GLEANING_PROMPT = """MANY entities and relationships may have been missed in the previous extraction.
Review the same segments again and extract any that were overlooked.

Do NOT re-output entities or relationships already correctly extracted.
Only output NEW additions or corrections.
If nothing was missed, return: {{"entities": [], "relationships": [], "memory_updates": []}}

Use the same JSON output format."""


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


MiniCPMAligner = MiniCPMCaptioner


def _edge_id(video_id: str, counter: int) -> str:
    safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", video_id)
    return f"EDGE_{safe_video}_{counter:07d}"


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
# Sub-step 8b — GPT-4o-mini entity+relationship extraction (GraphRAG style)
# =====================================================================

def _filter_entity_memory(
    entity_memory: dict[str, Any],
    recent_segment_ids: set[str],
    *,
    global_threshold: int = 5,
    max_entities: int = 25,
) -> dict[str, Any]:
    keep: dict[str, int] = {}

    for eid in list(entity_memory.keys()):
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


def _build_graphrag_prompt(
    batch_segments: list[dict[str, Any]],
    captions: dict[str, str],
    transcripts: dict[str, str],
    entity_memory: dict[str, Any],
    *,
    recent_segment_ids: set[str] | None = None,
    max_memory_entities: int = 25,
) -> str:
    filtered_memory = _filter_entity_memory(
        entity_memory,
        recent_segment_ids or set(),
        max_entities=max_memory_entities,
    )
    display_memory: dict[str, Any] = {}
    for eid, data in filtered_memory.items():
        key = data.get("canonical_name") or eid
        if key in display_memory:
            key = f"{key} ({eid})"
        display_memory[key] = {
            "type": data.get("entity_type", ""),
            "descriptions": data.get("accumulated_descriptions", [])[-3:],
            "segments_seen": len(data.get("segments_seen", [])),
        }
    entity_memory_json = json.dumps(display_memory, ensure_ascii=False) if display_memory else "{}"

    segments_data_parts = []
    for seg in batch_segments:
        seg_id = str(seg["segment_id"])
        caption = captions.get(seg_id, "")
        transcript = transcripts.get(seg_id, "")
        segments_data_parts.append(
            f"SEGMENT {seg_id} [{seg['start']:.1f}s - {seg['end']:.1f}s]:\n"
            f"Visual caption: {caption}\n"
            f"Transcript: {transcript}"
        )
    segments_data_str = "\n\n".join(segments_data_parts)

    return GRAPHRAG_PROMPT.format(
        entity_memory_json=entity_memory_json,
        segments_data=segments_data_str,
    )


PREDICATE_NORMALIZE = {
    "chases": "chase", "attacks": "attack", "approaches": "approach",
    "inhabits": "inhabit", "interacts_with": "interact_with",
    "observes": "observe", "hunts": "hunt", "stalks": "stalk",
    "confronts": "confront", "defends": "defend", "flees": "flee",
    "stands_near": "stand_near", "swims": "swim", "flies": "fly",
    "fights": "fight", "eats": "eat", "feeds_on": "feed_on",
    "shares_habitat_with": "share_habitat",
    "shares_space_with": "share_habitat",
    "shares_environment_with": "share_habitat",
    "occupies_space_with": "share_habitat",
    "forages_for": "forage", "forages_in": "forage",
    "gathers_around": "gather", "gathers_at": "gather",
    "engages_with": "engage", "engages_in": "engage",
    "engages": "engage", "transitions_to": "transition",
    "displays_similar_behavior": "resemble",
    "potentially_threatens": "threaten", "potential_threatens": "threaten",
    "threatens": "threaten",
}

ENTITY_NAME_NORMALIZE = {
    "DOLPHINS": "DOLPHIN", "LIONS": "LION", "ANTS": "ANT",
    "SNAKES": "SNAKE", "SEALS": "SEAL", "WOLVES": "WOLF",
    "MONKEYS": "MONKEY", "BEARS": "BEAR", "CRABS": "CRAB",
    "BIRDS": "BIRD", "SHARKS": "SHARK", "OTTERS": "OTTER",
    "PENGUINS": "PENGUIN", "ELEPHANTS": "ELEPHANT",
    "HIPPOS": "HIPPO", "HIPPOPOTAMUS": "HIPPO",
    "KILLER WHALES": "ORCA", "KILLER WHALE": "ORCA",
    "MONITOR LIZARD": "LACE MONITOR",
    "REINDEER": "CARIBOU",
    "WILD BOARS": "WILD BOAR", "BOARS": "WILD BOAR",
    "SEA LIONS": "SEA LION",
    "ATLANTIC LOBSTER": "LOBSTER",
    "ANTOLOPE": "ANTELOPE", "ANTELOPES": "ANTELOPE",
    "OVIRAPTORID DINOSAUR": "OVIRAPTORID",
    "HAIR": "HARE",
}

ENTITY_NOISE = {
    "BBC EARTH", "SUBMARINE", "THE SCENE", "THE VIDEO",
    "THE ATMOSPHERE", "THE CAMERA", "THE FOCUS",
    "THE NARRATOR", "ANIMAL", "OBJECT",
}


def _normalize_entity_name(name: str) -> str:
    upper = name.strip().upper()
    return ENTITY_NAME_NORMALIZE.get(upper, upper)


def _normalize_predicate(pred: str) -> str:
    normalized = pred.strip().lower().replace(" ", "_")
    return PREDICATE_NORMALIZE.get(normalized, normalized)


def _validate_graphrag_result(raw: dict[str, Any]) -> dict[str, Any]:
    entities = []
    for ent in raw.get("entities", []):
        name = _normalize_entity_name(str(ent.get("entity_name", "")))
        if not name or len(name) < 2 or name in ENTITY_NOISE:
            continue
        etype = str(ent.get("entity_type", "object")).strip().lower()
        if etype not in {"person", "animal", "object", "location", "event", "concept"}:
            etype = "object"
        if name == "HARE" and etype == "object":
            etype = "animal"
        entities.append({
            "entity_name": name,
            "entity_type": etype,
            "entity_description": str(ent.get("entity_description", "")).strip(),
            "source_segments": [str(s) for s in ent.get("source_segments", [])],
        })

    seen_entities: dict[str, int] = {}
    deduped_entities = []
    for ent in entities:
        name = ent["entity_name"]
        if name in seen_entities:
            existing = deduped_entities[seen_entities[name]]
            existing["source_segments"] = sorted(
                set(existing["source_segments"]) | set(ent["source_segments"])
            )
            if ent["entity_description"] and not existing["entity_description"]:
                existing["entity_description"] = ent["entity_description"]
        else:
            seen_entities[name] = len(deduped_entities)
            deduped_entities.append(ent)

    relationships = []
    for rel in raw.get("relationships", []):
        source = _normalize_entity_name(str(rel.get("source_entity", "")))
        target = _normalize_entity_name(str(rel.get("target_entity", "")))
        predicate = _normalize_predicate(str(rel.get("predicate", "")))
        if source and target and predicate and source != target:
            if source in ENTITY_NOISE or target in ENTITY_NOISE:
                continue
            relationships.append({
                "source_entity": source,
                "target_entity": target,
                "predicate": predicate,
                "keywords": str(rel.get("keywords", "")),
                "segment_id": str(rel.get("segment_id", "")),
                "confidence": max(0.0, min(float(rel.get("confidence", 0.5)), 1.0)),
                "description": str(rel.get("description", "")),
            })

    memory_updates = []
    for update in raw.get("memory_updates", []):
        name = _normalize_entity_name(str(update.get("entity_name", "")))
        desc = str(update.get("new_description", "")).strip()
        if name and desc and name not in ENTITY_NOISE:
            memory_updates.append({"entity_name": name, "new_description": desc})

    return {
        "entities": deduped_entities,
        "relationships": relationships,
        "memory_updates": memory_updates,
    }


async def _call_gpt4o_mini(prompt: str, system_prompt: str, history_messages=None) -> str:
    from .._llm import openai_complete_if_cache
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
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
    initial_entity_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run GPT-4o-mini entity+relationship extraction in sequential batches."""
    import asyncio

    transcripts = transcripts or {}
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
    if not entity_memory and initial_entity_memory:
        entity_memory = dict(initial_entity_memory)
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
                        "entity_ids": [],
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

            prompt = _build_graphrag_prompt(
                batch, captions, transcripts, entity_memory,
                recent_segment_ids=_recent_segment_ids,
                max_memory_entities=int(config.get("entity_memory_max_context", 25)),
            )

            try:
                raw_response = loop.run_until_complete(
                    _call_gpt4o_mini(prompt, GRAPHRAG_SYSTEM)
                )
                raw_result = _extract_json(raw_response)
            except Exception as exc:
                logger.warning(
                    "graphrag_extract batch %d failed (video=%s): %s",
                    bi, video_id, exc,
                )
                if config.get("pipeline_strict", False):
                    raise
                raw_result = {"entities": [], "relationships": [], "memory_updates": []}

            validated = _validate_graphrag_result(raw_result)

            # ---- Gleaning: retry to catch missed entities (LightRAG pattern) ----
            max_gleaning = int(config.get("extraction_gleaning_rounds", 1))
            for gleaning_round in range(max_gleaning):
                if not validated["entities"] and not validated["relationships"]:
                    break
                try:
                    gleaning_resp = loop.run_until_complete(
                        _call_gpt4o_mini(
                            GLEANING_PROMPT,
                            GRAPHRAG_SYSTEM,
                            history_messages=[
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": raw_response},
                            ],
                        )
                    )
                    gleaning_result = _extract_json(gleaning_resp)
                    gleaning_valid = _validate_graphrag_result(gleaning_result)
                    if not gleaning_valid["entities"] and not gleaning_valid["relationships"]:
                        break
                    existing_names = {e["entity_name"] for e in validated["entities"]}
                    for ent in gleaning_valid["entities"]:
                        if ent["entity_name"] not in existing_names:
                            validated["entities"].append(ent)
                            existing_names.add(ent["entity_name"])
                    validated["relationships"].extend(gleaning_valid["relationships"])
                    validated["memory_updates"].extend(gleaning_valid["memory_updates"])
                except Exception:
                    break

            # ---- Register extracted entities ----
            for ent in validated["entities"]:
                existing_id = registry.resolve(ent["entity_name"])
                if existing_id:
                    global_id = existing_id
                    if global_id not in registry.entities:
                        registry.ensure_entity(
                            ent["entity_type"],
                            ent["entity_name"],
                            entity_id=global_id,
                            source="cross_video",
                            confidence=0.7,
                        )
                else:
                    global_id = registry.ensure_entity(
                        ent["entity_type"],
                        ent["entity_name"],
                        source="gpt4o_mini",
                        confidence=0.7,
                    )
                mem = entity_memory.setdefault(global_id, {
                    "canonical_name": ent["entity_name"],
                    "entity_type": ent["entity_type"],
                    "accumulated_descriptions": [],
                    "relationships": [],
                    "segments_seen": [],
                })
                mem["canonical_name"] = ent["entity_name"]
                mem["entity_type"] = ent["entity_type"]
                if ent["entity_description"]:
                    mem["accumulated_descriptions"].append(ent["entity_description"])
                    mem["accumulated_descriptions"] = mem["accumulated_descriptions"][-10:]
                for sid in ent.get("source_segments", []):
                    if sid and sid not in mem["segments_seen"]:
                        mem["segments_seen"].append(sid)

            # ---- Process relationships ----
            seg_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
            dropped_rels: list[dict[str, str]] = []
            for rel in validated["relationships"]:
                source = registry.resolve(rel["source_entity"])
                target = registry.resolve(rel["target_entity"])
                if not source or not target:
                    dropped_rels.append({
                        "batch": str(bi),
                        "raw_source": rel["source_entity"],
                        "raw_target": rel["target_entity"],
                        "resolved_source": source or "",
                        "resolved_target": target or "",
                        "predicate": rel["predicate"],
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
                    modalities=["visual", "transcript"],
                    description=f"{rel.get('description', '')} [{rel.get('keywords', '')}]".strip(),
                    provenance=[
                        ProvenanceRecord(
                            source="gpt4o_mini_graphrag",
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

            # ---- Process memory updates ----
            for update in validated["memory_updates"]:
                resolved = registry.resolve(update["entity_name"])
                if not resolved:
                    continue
                entry = entity_memory.setdefault(resolved, {
                    "canonical_name": update["entity_name"],
                    "accumulated_descriptions": [],
                    "relationships": [],
                    "segments_seen": [],
                })
                entry["accumulated_descriptions"].append(update["new_description"])
                entry["accumulated_descriptions"] = entry["accumulated_descriptions"][-10:]

            # ---- Build segment results ----
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
    initial_entity_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

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
        initial_entity_memory=initial_entity_memory,
    )

    return result
