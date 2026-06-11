from __future__ import annotations

import importlib.util as _iutil
import json
import gc
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _detect_attn_impl() -> str:
    """Return 'flash_attention_2' if real flash-attn (.so) is available, else 'sdpa'."""
    spec = _iutil.find_spec("flash_attn")
    if spec is not None and spec.origin is not None:
        return "flash_attention_2"
    return "sdpa"

from PIL import Image

from ..pipeline.stage_runner import atomic_write_json, read_json
from .correspondence import (
    OpenCLIPTextEncoder,
    gate_merge_candidates,
    transcript_propositions,
    visual_facts,
)
from .registry import EntityRegistry, normalize_alias
from .schema import EdgeOccurrence, ProvenanceRecord


VISUAL_FACT_PROMPT = """Analyze only the supplied video frames.

Rules:
1. Use only the visible entity IDs listed below.
2. Do not use transcript knowledge or infer names.
3. Describe observable appearance, action, pose, and interaction.
4. Never invent an entity ID.
5. Omit uncertain relations.

Return JSON only:
{{
  "visual_caption": "A concise description using visible entity IDs.",
  "entity_descriptions": [
    {{
      "entity_id": "PERSON_001",
      "description": "A seated person wearing a dark shirt."
    }}
  ],
  "visual_edges": [
    {{
      "source": "PERSON_001",
      "predicate": "holds",
      "target": "OBJECT_001",
      "confidence": 0.0
    }}
  ]
}}

Segment: {segment_id}
Time: {start:.2f}-{end:.2f}s
Visible entities:
{visual_entities}
"""


ALIGNMENT_PROMPT = """You are grounding one video segment into an entity graph.

Rules:
1. Use only entity IDs and merge candidates explicitly listed below.
2. Never invent an ID.
3. Merge only when the frames and transcript clearly refer to the same entity.
4. Return directed factual relations visible or explicitly stated in this segment.
5. If uncertain, do not merge and omit the uncertain edge.
6. For every edge, list only the evidence channels actually supporting it:
   "visual", "transcript", or both.
7. Transcript references marked unresolved are context, not new entities.
   Use only an existing supplied entity ID when the memory and current
   evidence make the antecedent clear; otherwise omit the uncertain edge.
8. Allowed merge candidates have passed a train-free correspondence gate.
   A passing score permits consideration but does not prove identity.
9. Never merge merely because one person is visible while a person's name is
   mentioned. Require the paired visual fact and transcript proposition to
   describe the same grounded participant or interaction.

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

Visible entities:
{visual_entities}

Transcript entities:
{text_entities}

Allowed merge candidates:
{merge_candidates}

Visual-only facts:
{visual_facts}

Transcript propositions:
{transcript_propositions}

Recent event memory:
{recent_memory}

Entity state:
{entity_state}

Transcript discourse memory:
{transcript_memory}

Current transcript references:
{transcript_references}

Rewritten transcript:
{transcript}
"""


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("MiniCPM response does not contain a JSON object.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def _load_frame_images(frame_paths: list[str]) -> list[Image.Image]:
    return [
        Image.open(path).convert("RGB")
        for path in frame_paths
        if Path(path).exists()
    ]


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

    def _chat_with_images(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        max_tokens: int,
        max_slice_nums: int | None = None,
    ) -> dict[str, Any]:
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
            # Repair always uses the full token budget regardless of the original
            # call's limit — a truncated response needs more room to be rewritten.
            repair_max_tokens = int(self.config.get("caption_max_tokens", 450))
            repaired = self.model.chat(
                image=None,
                msgs=repair_messages,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=effective_slices,
                max_new_tokens=repair_max_tokens,
            )
            try:
                return _extract_json(str(repaired))
            except (ValueError, json.JSONDecodeError):
                # Model could not produce valid JSON even after repair; return
                # an empty dict so the segment is skipped gracefully rather
                # than crashing the whole pipeline.
                return {}

    def _chat_batch(
        self,
        prompts: list[str],
        images_list: list[list[Image.Image]],
        *,
        max_tokens: int,
        max_slice_nums: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run one MiniCPM batch call for multiple segments simultaneously.

        MiniCPM batch mode is triggered when msgs is a list-of-lists.
        Images must be embedded in message content; image=None is required.
        Returns one result dict per input (empty dict on unrecoverable failure).
        On CUDA OOM falls back to sequential single-item calls and frees cache.
        Caller is responsible for closing all PIL Images after this returns.
        """
        import torch

        effective_slices = max_slice_nums if max_slice_nums is not None else int(
            self.config.get("caption_max_slice_nums", 2)
        )
        # Batch format: outer list = batch items, inner list = conversation turns
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
            # OOM fallback: process each item individually
            return [
                self._chat_with_images(
                    prompts[i],
                    images_list[i],
                    max_tokens=max_tokens,
                    max_slice_nums=max_slice_nums,
                )
                for i in range(len(prompts))
            ]
        results: list[dict[str, Any]] = []
        for i, response in enumerate(responses):
            try:
                results.append(_extract_json(str(response)))
            except (ValueError, json.JSONDecodeError):
                # Single-item repair for this batch item (images still alive)
                results.append(
                    self._chat_with_images(
                        prompts[i],
                        images_list[i],
                        max_tokens=max_tokens,
                        max_slice_nums=max_slice_nums,
                    )
                )
        return results

    def align(
        self,
        prompt: str,
        frame_paths: list[str],
        *,
        max_tokens: int | None = None,
        max_slice_nums: int | None = None,
    ) -> dict[str, Any]:
        self.load()
        effective = max_tokens if max_tokens is not None else int(
            self.config.get("caption_max_tokens", 450)
        )
        images = _load_frame_images(frame_paths)
        try:
            return self._chat_with_images(
                prompt, images, max_tokens=effective, max_slice_nums=max_slice_nums
            )
        finally:
            for image in images:
                image.close()

    def align_preloaded(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        max_tokens: int | None = None,
        max_slice_nums: int | None = None,
    ) -> dict[str, Any]:
        """Run inference with already-loaded PIL Images. Caller is responsible for closing them."""
        self.load()
        effective = max_tokens if max_tokens is not None else int(
            self.config.get("caption_max_tokens", 450)
        )
        return self._chat_with_images(
            prompt, images, max_tokens=effective, max_slice_nums=max_slice_nums
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
            and not mention.get("generic")
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
    *,
    visual_ids: set[str] | None = None,
    text_ids: set[str] | None = None,
) -> dict[str, Any]:
    visual_ids = visual_ids or set()
    text_ids = text_ids or set()
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
    merged_visual_ids = {
        merge["visual_id"]
        for merge in merges
    }
    edges = []
    for edge in raw.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        predicate = (
            str(edge.get("predicate", "")).strip().lower().replace(" ", "_")
        )
        source_visual_only = source in visual_ids and source not in text_ids
        target_visual_only = target in visual_ids and target not in text_ids
        source_text_only = source in text_ids and source not in visual_ids
        target_text_only = target in text_ids and target not in visual_ids
        unsupported_cross_modal = (
            source_visual_only
            and target_text_only
            and source not in merged_visual_ids
        ) or (
            target_visual_only
            and source_text_only
            and target not in merged_visual_ids
        )
        if unsupported_cross_modal:
            continue
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


def _validate_visual_analysis(
    raw: dict[str, Any],
    visible_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {
        str(entity["entity_id"])
        for entity in visible_entities
    }
    descriptions = []
    for item in raw.get("entity_descriptions", []):
        entity_id = str(item.get("entity_id", "")).strip()
        description = str(item.get("description", "")).strip()
        if entity_id in allowed and description:
            descriptions.append(
                {
                    "entity_id": entity_id,
                    "description": description,
                }
            )
    edges = []
    for edge in raw.get("visual_edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        predicate = (
            str(edge.get("predicate", "")).strip().lower().replace(" ", "_")
        )
        if (
            source in allowed
            and target in allowed
            and source != target
            and predicate
        ):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "predicate": predicate,
                    "confidence": max(
                        0.0,
                        min(float(edge.get("confidence", 0.5)), 1.0),
                    ),
                    "modalities": ["visual"],
                }
            )
    return {
        "visual_caption": str(raw.get("visual_caption", "")).strip(),
        "entity_descriptions": descriptions,
        "visual_edges": edges,
    }


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
    aligner: MiniCPMAligner | None = None,
    correspondence_encoder=None,
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
    crossmodal_enabled = not bool(config.get("disable_crossmodal_alignment", False))
    owns_correspondence_encoder = crossmodal_enabled and correspondence_encoder is None
    if owns_correspondence_encoder:
        correspondence_encoder = OpenCLIPTextEncoder(config)

    visual_max_tokens = int(config.get("caption_visual_max_tokens", 200))
    visual_slice_nums = int(config.get("caption_visual_slice_nums", 1))
    alignment_max_tokens = int(config.get("caption_max_tokens", 450))

    # load already-completed alignment results
    results: dict[str, dict[str, Any]] = {}
    for previous_index in range(next_index):
        segment_id = segments[previous_index]["segment_id"]
        previous = read_json(checkpoint_path / f"{segment_id}.json")
        if previous:
            results[segment_id] = previous

    def _frame_paths(seg: dict[str, Any]) -> list[str]:
        return [
            item["path"]
            for item in frame_selections.get(
                str(seg["segment_id"]), {}
            ).get("frames", [])
        ]

    # -----------------------------------------------------------------
    # PASS 1 — Visual captioning (stateless, prefetch frames)
    # visual_pass.json caches results so resume and ablation reuse work.
    # -----------------------------------------------------------------
    visual_pass_path = checkpoint_path / "visual_pass.json"
    visual_pass: dict[str, dict[str, Any]] = dict(read_json(visual_pass_path, {}))

    segments_needing_visual = [
        seg for seg in segments
        if str(seg["segment_id"]) not in visual_pass
        and str(seg["segment_id"]) not in results
    ]

    if segments_needing_visual:
        aligner.load()
        batch_size = int(config.get("caption_visual_batch_size", 4))
        # Group segments into batches for parallel GPU processing.
        # VRAM budget at batch_size=4: ~4.6 GB extra vs ~10 GB headroom (RTX 3090).
        batches = [
            segments_needing_visual[i : i + batch_size]
            for i in range(0, len(segments_needing_visual), batch_size)
        ]
        n_already_done = len(segments) - len(segments_needing_visual)
        with (
            tqdm(
                total=len(segments),
                initial=n_already_done,
                unit="seg",
                desc=f"[{video_id}] visual caption (pass 1)",
                dynamic_ncols=True,
            ) as vbar,
            ThreadPoolExecutor(max_workers=batch_size) as executor,
        ):
            # Prefetch first batch's frames in parallel before GPU work starts
            pending_futures: list[Future[list[Image.Image]]] = [
                executor.submit(_load_frame_images, _frame_paths(seg))
                for seg in batches[0]
            ]

            for bi, batch in enumerate(batches):
                # Collect prefetched images for this batch
                batch_images: list[list[Image.Image]] = [
                    f.result() for f in pending_futures
                ]

                # Kick off prefetch for next batch before GPU call begins
                if bi + 1 < len(batches):
                    pending_futures = [
                        executor.submit(_load_frame_images, _frame_paths(seg))
                        for seg in batches[bi + 1]
                    ]
                else:
                    pending_futures = []

                # Separate segments that need MiniCPM from those that can be skipped
                active_idx: list[int] = []
                active_prompts: list[str] = []
                active_visible: list[list[dict[str, Any]]] = []

                for local_i, (seg, images) in enumerate(zip(batch, batch_images)):
                    segment_id = str(seg["segment_id"])
                    visible = _visible_entities(segment_id, observations)
                    if visible and images:
                        active_idx.append(local_i)
                        active_visible.append(visible)
                        active_prompts.append(
                            VISUAL_FACT_PROMPT.format(
                                segment_id=segment_id,
                                start=float(seg["start"]),
                                end=float(seg["end"]),
                                visual_entities=json.dumps(
                                    visible, ensure_ascii=False
                                ),
                            )
                        )

                # Single batch call for all active segments; images stay alive for repair
                if active_idx:
                    active_images = [batch_images[i] for i in active_idx]
                    batch_results = aligner._chat_batch(
                        active_prompts,
                        active_images,
                        max_tokens=visual_max_tokens,
                        max_slice_nums=visual_slice_nums,
                    )
                else:
                    batch_results = []

                # Map results back and close images
                result_iter = iter(zip(active_idx, active_visible, batch_results))
                next_active = next(result_iter, None)

                for local_i, (seg, images) in enumerate(zip(batch, batch_images)):
                    segment_id = str(seg["segment_id"])
                    if next_active is not None and next_active[0] == local_i:
                        _, visible, raw_visual = next_active
                        visual_analysis = _validate_visual_analysis(raw_visual, visible)
                        next_active = next(result_iter, None)
                    else:
                        visual_analysis = {
                            "visual_caption": "",
                            "entity_descriptions": [],
                            "visual_edges": [],
                        }
                    for img in images:
                        img.close()
                    visual_pass[segment_id] = visual_analysis

                vbar.update(len(batch))
                atomic_write_json(visual_pass_path, visual_pass)

    # -----------------------------------------------------------------
    # PASS 2 — Alignment (stateful, memory-aware, prefetch frames)
    # OpenCLIP runs inline here so registry stays progressive across
    # segments (deterministic merges from earlier segs inform later ones).
    # Frame prefetch overlaps with the OpenCLIP + CPU work before each
    # MiniCPM call.
    # -----------------------------------------------------------------
    remaining_segments = segments[next_index:]

    with ThreadPoolExecutor(max_workers=1) as executor:
        next_future_p2: Future[list[Image.Image]] | None = None
        next_prefetch_id_p2: str | None = None

        if remaining_segments:
            next_future_p2 = executor.submit(
                _load_frame_images, _frame_paths(remaining_segments[0])
            )
            next_prefetch_id_p2 = str(remaining_segments[0]["segment_id"])

        for index, segment in enumerate(remaining_segments, start=next_index):
            segment_id = str(segment["segment_id"])
            local_idx = index - next_index

            text_result = text_results.get(
                segment_id,
                {"mentions": [], "rewritten_transcript": ""},
            )
            visible = _visible_entities(segment_id, observations)
            visual_analysis = visual_pass.get(
                segment_id,
                {"visual_caption": "", "entity_descriptions": [], "visual_edges": []},
            )

            # collect pre-fetched frames
            align_images = (
                next_future_p2.result()
                if next_future_p2 is not None and next_prefetch_id_p2 == segment_id
                else _load_frame_images(_frame_paths(segment))
            )

            # kick off prefetch for next segment before OpenCLIP/CPU work
            if local_idx + 1 < len(remaining_segments):
                nxt = remaining_segments[local_idx + 1]
                next_prefetch_id_p2 = str(nxt["segment_id"])
                next_future_p2 = executor.submit(_load_frame_images, _frame_paths(nxt))
            else:
                next_future_p2 = None

            # candidate merges use live registry for progressive deterministic resolution
            raw_candidates, deterministic = _candidate_merges(
                visible,
                text_result.get("mentions", []),
                registry,
            )
            if not crossmodal_enabled:
                deterministic = {}
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
            transcript_memory = _rewrite_ids(
                json.dumps(
                    text_result.get("memory_context", {}),
                    ensure_ascii=False,
                ),
                registry,
            )
            transcript_references = _rewrite_ids(
                json.dumps(
                    text_result.get("references", []),
                    ensure_ascii=False,
                ),
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

            # OpenCLIP correspondence gate — runs while next frames are prefetching
            facts = visual_facts(visual_analysis)
            propositions = transcript_propositions(text_result, segment_id)
            threshold = float(
                config.get("correspondence_similarity_threshold", 0.28)
            )
            margin = float(
                config.get("correspondence_similarity_margin", 0.04)
            )
            if (
                crossmodal_enabled
                and raw_candidates
                and facts
                and propositions
                and correspondence_encoder is not None
            ):
                candidates, correspondence = gate_merge_candidates(
                    raw_candidates,
                    facts,
                    propositions,
                    correspondence_encoder.encode,
                    threshold=threshold,
                    margin=margin,
                )
            elif not crossmodal_enabled:
                candidates = []
                correspondence = {
                    "status": "disabled_by_ablation",
                    "threshold": threshold,
                    "margin": margin,
                    "pairs": [],
                }
            else:
                candidates = []
                correspondence = {
                    "status": "no_correspondence_evidence",
                    "threshold": threshold,
                    "margin": margin,
                    "pairs": [],
                }

            prompt_propositions = [
                {
                    **proposition,
                    "text": _rewrite_ids(proposition["text"], registry),
                    "entity_ids": [
                        registry.resolve(entity_id) or entity_id
                        for entity_id in proposition.get("entity_ids", [])
                    ],
                }
                for proposition in propositions
            ]
            prompt = ALIGNMENT_PROMPT.format(
                segment_id=segment_id,
                start=float(segment["start"]),
                end=float(segment["end"]),
                visual_entities=json.dumps(visible, ensure_ascii=False),
                text_entities=json.dumps(
                    prompt_mentions, ensure_ascii=False
                ),
                merge_candidates=json.dumps(candidates, ensure_ascii=False),
                visual_facts=json.dumps(facts, ensure_ascii=False),
                transcript_propositions=json.dumps(
                    prompt_propositions,
                    ensure_ascii=False,
                ),
                recent_memory=json.dumps(recent_events[-8:], ensure_ascii=False),
                entity_state=json.dumps(entity_state, ensure_ascii=False),
                transcript_memory=transcript_memory,
                transcript_references=transcript_references,
                transcript=prompt_transcript,
            )

            # MiniCPM alignment — frames are pre-fetched and waiting
            try:
                raw_alignment = aligner.align_preloaded(
                    prompt,
                    align_images,
                    max_tokens=alignment_max_tokens,
                )
            finally:
                for img in align_images:
                    img.close()

            text_ids = {
                str(mention["text_id"])
                for mention in text_result.get("mentions", [])
            }
            text_ids.update(
                str(mention["text_id"])
                for mention in prompt_mentions
            )
            alignment = _validate_alignment(
                raw_alignment,
                candidates,
                visual_ids={
                    str(entity["entity_id"])
                    for entity in visible
                },
                text_ids=text_ids,
            )
            alignment["edges"] = _deduplicate_edges(
                visual_analysis["visual_edges"] + alignment["edges"]
            )

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
                        segment_id=segment_id,
                    )
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
                                occurrence.get(
                                    "confidence",
                                    mention.get("confidence", 0.0),
                                )
                            ),
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
                            source=(
                                "minicpm_visual"
                                if raw_edge["modalities"] == ["visual"]
                                else (
                                    "minicpm_transcript"
                                    if raw_edge["modalities"] == ["transcript"]
                                    else "minicpm_alignment"
                                )
                            ),
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
            recent_events = recent_events[
                -int(config.get("entity_memory_recent_events", 8)):
            ]
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
                "raw_candidate_merges": raw_candidates,
                "correspondence": correspondence,
                "visual_analysis": visual_analysis,
                "transcript_propositions": propositions,
                "frame_paths": _frame_paths(segment),
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
                    correspondence=correspondence["status"],
                    gated_candidates=len(candidates),
                    accepted_merges=len(alignment["entity_merges"]),
                )

    if owns_correspondence_encoder and correspondence_encoder is not None:
        correspondence_encoder.close()
    correspondence_stats: dict[str, int] = defaultdict(int)
    for segment in results.values():
        status = str(
            segment.get("correspondence", {}).get("status", "unknown")
        )
        correspondence_stats[status] += 1
    return {
        "segments": results,
        "registry": registry.to_dict(),
        "edge_count": edge_counter,
        "recent_events": recent_events,
        "long_term_state": long_term_state,
        "correspondence_stats": dict(correspondence_stats),
    }
