from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .._op import get_chunks
from .._unified_graph.alignment import MiniCPMCaptioner, align_all_segments
from .._unified_graph.builder import build_unified_graph, validate_unified_graph
from .._unified_graph.registry import EntityRegistry, normalize_alias
from .._videoutil.asr_v2 import (
    assign_words_to_segments,
    transcribe_full_video,
)
from .._videoutil.frame_selector import select_segment_frames
from .._videoutil.media_probe import probe_video
from .._videoutil.shot_detection import detect_shots_and_motion
from .._videoutil.smart_segment import fixed_segment, smart_segment
from .._utils import compute_mdhash_id, logger
from .stage_runner import StageRunner, atomic_write_json, read_json
from .unified_stages import UNIFIED_STAGE_DEFINITIONS


def _scalar_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_scalar_config(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _scalar_config(item)
            for key, item in value.items()
            if not callable(item)
        }
    return str(value)


def build_pipeline_config(vrag) -> dict[str, Any]:
    raw = asdict(vrag) if is_dataclass(vrag) else dict(vars(vrag))
    excluded = {
        "llm",
        "entity_extraction_func",
        "chunk_func",
        "convert_response_to_json_func",
        "key_string_value_json_storage_cls",
        "vector_db_storage_cls",
        "vs_vector_db_storage_cls",
        "graph_storage_cls",
    }
    config = {
        key: _scalar_config(value)
        for key, value in raw.items()
        if key not in excluded and not callable(value)
    }
    config["caption_backend"] = "MiniCPM-V-2_6-int4"
    return config


def _load_global_registry(working_dir: str | Path) -> EntityRegistry:
    path = Path(working_dir) / "pipeline_v2" / "global_registry.json"
    return EntityRegistry(read_json(path, {}))


def _new_video_registry(working_dir: str | Path) -> EntityRegistry:
    global_registry = _load_global_registry(working_dir)
    registry = EntityRegistry({"counters": dict(global_registry.counters), "entities": []})
    registry.name_to_global = dict(global_registry.name_to_global)
    return registry


def _merge_global_registry(
    working_dir: str | Path,
    video_id: str,
    registry_payload: dict[str, Any],
) -> list[str]:
    global_registry = _load_global_registry(working_dir)
    removed_entity_ids = []
    for entity_id, node in list(global_registry.entities.items()):
        belonged_to_video = any(
            alias.alias_id.startswith(f"{video_id}::")
            for alias in node.aliases
        ) or any(
            item.video_id == video_id
            for item in node.provenance
        )
        node.aliases = [
            alias
            for alias in node.aliases
            if not alias.alias_id.startswith(f"{video_id}::")
        ]
        node.provenance = [
            item for item in node.provenance if item.video_id != video_id
        ]
        if belonged_to_video and not node.provenance and not node.aliases:
            del global_registry.entities[entity_id]
            removed_entity_ids.append(entity_id)
    global_registry.alias_to_global = {
        alias.alias_id: entity_id
        for entity_id, node in global_registry.entities.items()
        for alias in node.aliases
    }
    global_registry.name_to_global = {
        normalize_alias(node.canonical_name): entity_id
        for entity_id, node in global_registry.entities.items()
    }
    for entity in registry_payload.get("entities", []):
        entity_id = entity["entity_id"]
        global_registry.ensure_entity(
            entity["entity_type"],
            entity.get("canonical_name", entity_id),
            entity_id=entity_id,
            source=None,
            confidence=float(entity.get("confidence", 0.0)),
            attributes=dict(entity.get("attributes", {})),
        )
        node = global_registry.entities[entity_id]
        node.sources = list(entity.get("sources", []))
        node.first_seen = entity.get("first_seen")
        node.last_seen = entity.get("last_seen")
        for alias in entity.get("aliases", []):
            global_registry.add_alias(
                f"{video_id}::{alias['alias_id']}",
                entity_id,
                source=alias.get("source", "unknown"),
                label=alias.get("label", ""),
                confidence=float(alias.get("confidence", 0.0)),
                segment_id=(
                    alias.get("segment_ids", [None])[0]
                    if alias.get("segment_ids")
                    else None
                ),
            )
        for provenance in entity.get("provenance", []):
            from .._unified_graph.schema import ProvenanceRecord

            global_registry.add_provenance(
                entity_id,
                ProvenanceRecord(**provenance),
            )
    for entity_type, count in registry_payload.get("counters", {}).items():
        global_registry.counters[entity_type] = max(
            global_registry.counters[entity_type],
            int(count),
        )
    atomic_write_json(
        Path(working_dir) / "pipeline_v2" / "global_registry.json",
        global_registry.to_dict(),
    )
    return removed_entity_ids


def _extract_clip(
    video_path: str,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, output_path)


class UnifiedIngestPipeline:
    def __init__(
        self,
        vrag,
        *,
        resume: bool = True,
        restart_stage: str | None = None,
        force: bool = False,
    ):
        self.vrag = vrag
        self.config = build_pipeline_config(vrag)
        self.resume = resume
        self.restart_stage = restart_stage
        self.force = force
        self.loop = asyncio.new_event_loop()

    def _await(self, awaitable):
        return self.loop.run_until_complete(awaitable)

    def close(self) -> None:
        self.loop.close()

    def run(self, video_paths: list[str]) -> list[dict[str, Any]]:
        reports = []
        try:
            with tqdm(
                total=len(video_paths),
                desc="Unified V2 videos",
                unit="video",
                dynamic_ncols=True,
            ) as video_progress:
                for video_position, video_path in enumerate(video_paths, start=1):
                    video_id = Path(video_path).stem
                    video_progress.set_postfix_str(video_id, refresh=True)
                    logger.info(
                        "[Unified V2] Video %d/%d: %s",
                        video_position,
                        len(video_paths),
                        video_id,
                    )
                    try:
                        reports.append(self.run_video(video_path))
                    except Exception as exc:
                        if not self.config.get(
                            "pipeline_continue_on_error",
                            False,
                        ):
                            raise
                        logger.exception(
                            "[Unified V2] Video failed; continuing: %s",
                            video_id,
                        )
                        reports.append(
                            {
                                "video_id": video_id,
                                "status": "failed",
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                },
                            }
                        )
                    video_progress.update(1)
            return reports
        finally:
            self.close()

    def run_video(self, video_path: str) -> dict[str, Any]:
        video_path = str(Path(video_path).resolve())
        video_id = Path(video_path).stem
        input_stat = Path(video_path).stat()
        run_config = {
            **self.config,
            "input_path": video_path,
            "input_size_bytes": input_stat.st_size,
            "input_mtime_ns": input_stat.st_mtime_ns,
        }
        runner = StageRunner(
            self.vrag.working_dir,
            video_id,
            run_config,
            UNIFIED_STAGE_DEFINITIONS,
            resume=self.resume,
            restart_stage=self.restart_stage,
            force=self.force,
        )

        with runner.run_scope():
            runner.run_stage("probe", lambda context: self._stage_probe(context, video_path))
            runner.run_stage("asr", lambda context: self._stage_asr(context, runner))
            runner.run_stage(
                "shot_detection",
                lambda context: self._stage_shots(context, runner),
            )
            runner.run_stage(
                "segmentation",
                lambda context: self._stage_segmentation(context, runner, video_id),
            )
            runner.run_stage(
                "frame_selection",
                lambda context: self._stage_frames(context, runner),
            )
            runner.run_stage(
                "alignment_caption",
                lambda context: self._stage_alignment(context, runner, video_id),
            )
            runner.run_stage(
                "graph_build",
                lambda context: self._stage_graph(context, runner, video_id),
            )
            runner.run_stage(
                "embedding_index",
                lambda context: self._stage_index(context, runner, video_id),
            )
            runner.run_stage(
                "validation",
                lambda context: self._stage_validation(context, runner),
            )

        return read_json(runner.report_path, {})

    def _stage_probe(self, context, video_path: str) -> dict[str, Any]:
        probe = probe_video(
            video_path,
            strict=bool(self.config.get("pipeline_strict", False)),
        )
        context.write_json("probe.json", probe)
        context.report_metrics(**probe)
        context.log(
            f"duration={probe['duration']:.2f}s fps={probe['fps']:.3f} "
            f"resolution={probe['width']}x{probe['height']}"
        )
        return {"output": str(context.path("probe.json")), **probe}

    def _stage_asr(self, context, runner: StageRunner) -> dict[str, Any]:
        probe = read_json(runner.output("probe", "probe.json"))
        with context.progress(
            total=float(probe["duration"]),
            unit="s",
            description="ASR full video",
        ) as progress:
            if probe.get("has_audio") is False:
                if self.config.get("pipeline_strict", False):
                    raise RuntimeError(
                        "Strict pipeline requires an audio stream for ASR."
                    )
                result = {
                    "model": "none",
                    "device": "none",
                    "compute_type": "none",
                    "language": None,
                    "language_probability": 0.0,
                    "duration": float(probe["duration"]),
                    "vad_filter": False,
                    "words": [],
                    "utterances": [],
                    "reason": "no_audio_stream",
                }
                progress.set(float(probe["duration"]), reason="no_audio_stream")
            else:
                result = transcribe_full_video(
                    probe["path"],
                    self.config,
                    progress=progress,
                )
        context.write_json("asr.json", result)
        context.report_metrics(
            words=len(result["words"]),
            utterances=len(result["utterances"]),
            model=result["model"],
            device=result["device"],
        )
        return {
            "output": str(context.path("asr.json")),
            "words": len(result["words"]),
            "utterances": len(result["utterances"]),
            "model": result["model"],
        }

    def _stage_shots(self, context, runner: StageRunner) -> dict[str, Any]:
        probe = read_json(runner.output("probe", "probe.json"))
        with context.progress(
            total=float(probe["duration"]),
            unit="s",
            description="Shot and motion analysis",
        ) as progress:
            result = detect_shots_and_motion(
                probe["path"],
                float(probe["duration"]),
                self.config,
                progress=progress,
            )
        context.write_json("shots.json", result)
        context.report_metrics(
            backend=result["backend"],
            shots=len(result["boundaries"]),
            samples=len(result["samples"]),
        )
        return {
            "output": str(context.path("shots.json")),
            "backend": result["backend"],
            "shots": len(result["boundaries"]),
        }

    def _stage_segmentation(
        self,
        context,
        runner: StageRunner,
        video_id: str,
    ) -> dict[str, Any]:
        probe = read_json(runner.output("probe", "probe.json"))
        asr = read_json(runner.output("asr", "asr.json"))
        shots = read_json(runner.output("shot_detection", "shots.json"))
        strategy = str(
            self.config.get("segmentation_strategy", "adaptive")
        ).strip().lower()
        if strategy == "adaptive":
            segments = smart_segment(
                float(probe["duration"]),
                asr["words"],
                shots["boundaries"],
                self.config,
            )
        elif strategy == "fixed":
            segments = fixed_segment(
                float(probe["duration"]),
                length=float(
                    self.config.get("fixed_segment_seconds", 30.0)
                ),
                context=float(
                    self.config.get("segment_context_seconds", 1.5)
                ),
            )
        else:
            raise ValueError(
                "segmentation_strategy must be 'adaptive' or 'fixed', "
                f"got {strategy!r}"
            )
        for segment in segments:
            segment["storage_id"] = f"{video_id}_{segment['index']}"
        context.write_json("segments.json", segments)
        durations = [float(item["duration"]) for item in segments]
        context.report_metrics(
            segments=len(segments),
            min_duration=min(durations, default=0.0),
            max_duration=max(durations, default=0.0),
            mean_duration=float(np.mean(durations)) if durations else 0.0,
            strategy=strategy,
        )
        return {
            "output": str(context.path("segments.json")),
            "segments": len(segments),
            "strategy": strategy,
        }

    def _stage_frames(self, context, runner: StageRunner) -> dict[str, Any]:
        probe = read_json(runner.output("probe", "probe.json"))
        segments = read_json(runner.output("segmentation", "segments.json"))
        shots = read_json(runner.output("shot_detection", "shots.json"))
        observations: list[dict[str, Any]] = []
        selections: dict[str, dict[str, Any]] = {}
        with context.progress(
            total=len(segments),
            unit="seg",
            description="Diverse frame selection",
        ) as progress:
            for index, segment in enumerate(segments):
                segment_id = segment["segment_id"]
                result_path = context.path(f"segments/{segment_id}.json")
                existing = read_json(result_path)
                if existing:
                    selections[segment_id] = existing
                    progress.set(index + 1, resumed=True)
                    continue
                selection = select_segment_frames(
                    probe["path"],
                    segment,
                    shots,
                    observations,
                    context.path("frames"),
                    self.config,
                )
                atomic_write_json(result_path, selection)
                selections[segment_id] = selection
                progress.set(
                    index + 1,
                    frames=len(selection.get("frames", [])),
                )
        context.write_json("frame_selections.json", selections)
        context.report_metrics(
            selected_frames=sum(
                len(item.get("frames", [])) for item in selections.values()
            ),
        )
        return {
            "segments": len(selections),
            "selected_frames": sum(
                len(item.get("frames", [])) for item in selections.values()
            ),
        }

    def _stage_alignment(
        self,
        context,
        runner: StageRunner,
        video_id: str,
    ) -> dict[str, Any]:
        segments = read_json(runner.output("segmentation", "segments.json"))
        selections = read_json(
            runner.output("frame_selection", "frame_selections.json")
        )
        asr = read_json(runner.output("asr", "asr.json"))
        by_segment = assign_words_to_segments(asr["words"], segments)
        transcripts: dict[str, str] = {
            str(seg["segment_id"]): " ".join(
                str(w["text"])
                for w in sorted(
                    by_segment.get(str(seg["segment_id"]), []),
                    key=lambda w: float(w["start"]),
                )
            )
            for seg in segments
        }

        registry = _new_video_registry(self.vrag.working_dir)
        global_registry = _load_global_registry(self.vrag.working_dir)
        registry.name_to_global.update(global_registry.name_to_global)
        global_memory_path = (
            Path(self.vrag.working_dir) / "pipeline_v2" / "global_entity_memory.json"
        )
        initial_entity_memory = read_json(global_memory_path, {})

        if (
            getattr(self.vrag, "caption_model", None) is None
            or getattr(self.vrag, "caption_tokenizer", None) is None
        ):
            self.vrag.load_caption_model()
        captioner = MiniCPMCaptioner(
            self.config,
            model=getattr(self.vrag, "caption_model", None),
            tokenizer=getattr(self.vrag, "caption_tokenizer", None),
        )

        with context.progress(
            total=len(segments),
            unit="seg",
            description="Caption + entity extraction",
        ) as progress:
            result = align_all_segments(
                video_id,
                segments,
                selections,
                {},
                [],
                registry,
                self.config,
                context.path("segments"),
                progress=progress,
                aligner=captioner,
                loop=self.loop,
                transcripts=transcripts,
                initial_entity_memory=initial_entity_memory,
            )
        atomic_write_json(global_memory_path, result.get("entity_memory", {}))
        stale_entity_ids = _merge_global_registry(
            self.vrag.working_dir,
            video_id,
            result["registry"],
        )
        result["stale_entity_ids"] = stale_entity_ids
        context.write_json("alignment.json", result)
        context.report_metrics(
            entities=len(result["registry"]["entities"]),
            edges=result["edge_count"],
            caption_model="MiniCPM-V-2_6-int4",
            merge_model="gpt-4o-mini",
            stale_entities=len(stale_entity_ids),
        )
        return {
            "output": str(context.path("alignment.json")),
            "entities": len(result["registry"]["entities"]),
            "edges": result["edge_count"],
            "caption_model": "MiniCPM-V-2_6-int4",
            "merge_model": "gpt-4o-mini",
        }

    def _segment_storage_payload(
        self,
        runner: StageRunner,
        video_id: str,
    ) -> dict[str, dict[str, Any]]:
        segments = read_json(runner.output("segmentation", "segments.json"))
        alignment = read_json(
            runner.output("alignment_caption", "alignment.json")
        )
        selections = read_json(
            runner.output("frame_selection", "frame_selections.json")
        )
        asr = read_json(runner.output("asr", "asr.json"))
        by_segment = assign_words_to_segments(asr["words"], segments)
        registry_data = alignment.get("registry", {})
        entity_names: dict[str, str] = {}
        for entity in registry_data.get("entities", []):
            entity_names[entity["entity_id"]] = entity.get(
                "canonical_name", entity["entity_id"]
            )

        payload = {}
        for segment in segments:
            segment_id = segment["segment_id"]
            aligned = alignment["segments"].get(segment_id, {})

            words = by_segment.get(segment_id, [])
            original_transcript = " ".join(
                str(w["text"]) for w in sorted(
                    words, key=lambda w: float(w["start"])
                )
            )

            visible_ids = sorted(
                {
                    node
                    for edge in aligned.get("edges", [])
                    for node in (edge["source_id"], edge["target_id"])
                }
                | set(aligned.get("entity_ids", []))
            )
            entity_memory = (
                "Entity Memory:\n"
                + "\n".join(
                    f"- {eid} ({entity_names.get(eid, eid)})"
                    for eid in visible_ids
                )
                if visible_ids
                else ""
            )
            caption = aligned.get("caption", "")
            parts = []
            if entity_memory:
                parts.append(entity_memory)
            parts.append(f"Caption:\n{caption}")
            parts.append(f"Transcript:\n{original_transcript}")
            content = "\n\n".join(parts)
            payload[str(segment["index"])] = {
                "content": content,
                "time": f"{segment['start']:.3f}-{segment['end']:.3f}",
                "transcript": original_transcript,
                "caption": caption,
                "entity_memory": entity_memory,
                "frame_times": [
                    frame["time"]
                    for frame in selections.get(segment_id, {}).get("frames", [])
                ],
                "segment_id": segment_id,
                "storage_id": segment["storage_id"],
            }
        return payload

    def _stage_graph(
        self,
        context,
        runner: StageRunner,
        video_id: str,
    ) -> dict[str, Any]:
        alignment = read_json(
            runner.output("alignment_caption", "alignment.json")
        )
        removal_stats = {"removed_edges": 0, "removed_nodes": 0}
        if hasattr(self.vrag.chunk_entity_relation_graph, "remove_video"):
            removal_stats = self._await(
                self.vrag.chunk_entity_relation_graph.remove_video(video_id)
            )
        graph_stats = self._await(
            build_unified_graph(
                self.vrag.chunk_entity_relation_graph,
                alignment["registry"],
                alignment["segments"],
                clear=False,
                entity_memory=alignment.get("entity_memory", {}),
            )
        )
        segment_payload = self._segment_storage_payload(runner, video_id)
        self._await(self.vrag.video_path_db.upsert({video_id: read_json(
            runner.output("probe", "probe.json")
        )["path"]}))
        self._await(self.vrag.video_segments.upsert({video_id: segment_payload}))
        self._await(self.vrag.video_segments.index_done_callback())
        self._await(self.vrag.video_path_db.index_done_callback())
        self._await(self.vrag.chunk_entity_relation_graph.index_done_callback())
        context.write_json("graph_stats.json", graph_stats)
        context.report_metrics(**graph_stats, **removal_stats)
        return {**graph_stats, **removal_stats}

    def _stage_index(
        self,
        context,
        runner: StageRunner,
        video_id: str,
    ) -> dict[str, Any]:
        probe = read_json(runner.output("probe", "probe.json"))
        segments = read_json(runner.output("segmentation", "segments.json"))
        segment_payload = self._segment_storage_payload(runner, video_id)

        # Visual feature reuse: ImageBind embeddings depend only on the video
        # clips (segmentation). Profiles that seed segmentation from full_framework
        # have identical clips, so identical features. If all segment features for
        # this video are already present in the VDB (seeded), skip the expensive
        # clip extraction + ImageBind embedding.
        reuse_visual = False
        _vsf_client = getattr(self.vrag.video_segment_feature_vdb, "_client", None)
        if _vsf_client is not None and segments:
            wanted_ids = [f"{video_id}_{seg['index']}" for seg in segments]
            try:
                present = {item["__id__"] for item in _vsf_client.get(wanted_ids)}
                reuse_visual = all(wid in present for wid in wanted_ids)
            except Exception:
                reuse_visual = False

        old_video_data = getattr(self.vrag.video_segments, "_data", {}).get(
            video_id,
            {},
        )
        old_visual_ids = [
            f"{video_id}_{segment_index}"
            for segment_index in old_video_data
        ]
        if not reuse_visual and old_visual_ids and hasattr(
            getattr(self.vrag.video_segment_feature_vdb, "_client", None),
            "delete",
        ):
            self.vrag.video_segment_feature_vdb._client.delete(old_visual_ids)

        old_chunk_ids = []
        for chunk_id, chunk in list(
            getattr(self.vrag.text_chunks, "_data", {}).items()
        ):
            segment_ids = chunk.get("video_segment_id", [])
            if any(
                str(segment_id).startswith(f"{video_id}_")
                for segment_id in segment_ids
            ):
                old_chunk_ids.append(chunk_id)
                del self.vrag.text_chunks._data[chunk_id]
        if old_chunk_ids and hasattr(
            getattr(self.vrag.chunks_vdb, "_client", None),
            "delete",
        ):
            self.vrag.chunks_vdb._client.delete(old_chunk_ids)
        cache_dir = (
            Path(self.vrag.working_dir)
            / "_cache"
            / video_id
        )
        segment_index2name = {}
        with context.progress(
            total=len(segments),
            unit="seg",
            description="Segment clips and embedding indexes",
        ) as progress:
            for index, segment in enumerate(segments):
                name = (
                    f"stable-{segment['index']}-"
                    f"{segment['start']:.3f}-{segment['end']:.3f}"
                )
                segment_index2name[str(segment["index"])] = name
                clip_path = cache_dir / f"{name}.{self.vrag.video_output_format}"
                if not reuse_visual and (
                    self.config.get("pipeline_strict", False)
                    or not clip_path.exists()
                ):
                    _extract_clip(
                        probe["path"],
                        clip_path,
                        float(segment["start"]),
                        float(segment["end"]),
                    )
                progress.set(index + 1, clips=index + 1)

        chunks = get_chunks(
            new_videos={video_id: segment_payload},
            chunk_func=self.vrag.chunk_func,
            max_token_size=self.vrag.chunk_token_size,
        )
        new_chunk_ids = self._await(
            self.vrag.text_chunks.filter_keys(list(chunks.keys()))
        )
        new_chunks = {
            key: value for key, value in chunks.items() if key in new_chunk_ids
        }
        if new_chunks:
            if self.vrag.chunks_vdb is not None:
                self._await(self.vrag.chunks_vdb.upsert(new_chunks))
            self._await(self.vrag.text_chunks.upsert(new_chunks))

        alignment = read_json(
            runner.output("alignment_caption", "alignment.json")
        )
        stale_entity_vdb_ids = [
            compute_mdhash_id(entity_id, prefix="ent-")
            for entity_id in alignment.get("stale_entity_ids", [])
        ]
        if stale_entity_vdb_ids and hasattr(
            getattr(self.vrag.entities_vdb, "_client", None),
            "delete",
        ):
            self.vrag.entities_vdb._client.delete(stale_entity_vdb_ids)
        entity_data = {}
        for entity in alignment["registry"]["entities"]:
            aliases = " ".join(
                alias.get("label", "") for alias in entity.get("aliases", [])
            )
            content = (
                f"{entity['entity_id']} {entity.get('canonical_name', '')} "
                f"{entity.get('entity_type', '')} {aliases}"
            )
            entity_data[compute_mdhash_id(entity["entity_id"], prefix="ent-")] = {
                "content": content,
                "entity_name": entity["entity_id"],
            }
        if self.vrag.entities_vdb is not None and entity_data:
            self._await(self.vrag.entities_vdb.upsert(entity_data))

        if not reuse_visual:
            self._await(
                self.vrag.video_segment_feature_vdb.upsert(
                    video_id,
                    segment_index2name,
                    self.vrag.video_output_format,
                )
            )
        self._await(self.vrag._insert_done())
        context.report_metrics(
            chunks=len(chunks),
            new_chunks=len(new_chunks),
            indexed_entities=len(entity_data),
            visual_segments=len(segments),
        )
        if cache_dir.exists() and not self.config.get("keep_segment_cache", False):
            shutil.rmtree(cache_dir)
        return {
            "chunks": len(chunks),
            "new_chunks": len(new_chunks),
            "entities": len(entity_data),
            "visual_segments": len(segments),
        }

    def _stage_validation(self, context, runner: StageRunner) -> dict[str, Any]:
        graph_report = self._await(
            validate_unified_graph(self.vrag.chunk_entity_relation_graph)
        )
        segments = read_json(runner.output("segmentation", "segments.json"))
        alignment = read_json(
            runner.output("alignment_caption", "alignment.json")
        )
        missing_segments = [
            segment["segment_id"]
            for segment in segments
            if segment["segment_id"] not in alignment["segments"]
        ]
        report = {
            **graph_report,
            "missing_aligned_segments": missing_segments,
            "valid": graph_report["valid"] and not missing_segments,
        }
        context.write_json("validation_report.json", report)
        context.report_metrics(**report)
        if not report["valid"]:
            raise RuntimeError(
                "Unified graph validation failed. See validation_report.json."
            )
        return report
