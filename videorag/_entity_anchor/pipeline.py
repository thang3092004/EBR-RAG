from __future__ import annotations

import json
import os
from typing import Any

from .._utils import logger
from .linker import build_tracklets, link_tracklets
from .memory import build_segment_memory
from .schema import EntityAnchorResult
from .tracker import run_ultralytics_tracking


def _storage_dir(working_dir: str, configured_dir: str) -> str:
    if os.path.isabs(configured_dir):
        return configured_dir
    return os.path.join(working_dir, configured_dir)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_entity_anchor(
    video_name: str,
    video_path: str,
    segment_times_info: dict[str, dict],
    global_config: dict,
) -> EntityAnchorResult:
    """Build train-free entity IDs and per-segment entity memory."""
    if not global_config.get("enable_entity_anchoring", False):
        return EntityAnchorResult(video_name=video_name, entities=[], tracklets=[], segment_memory={})

    observations = run_ultralytics_tracking(
        video_name=video_name,
        video_path=video_path,
        segment_times_info=segment_times_info,
        global_config=global_config,
    )
    tracklets = build_tracklets(observations)
    entities = link_tracklets(tracklets, global_config)
    top_k = int(global_config.get("entity_memory_top_k", 12))
    segment_memory = build_segment_memory(entities, segment_times_info, top_k=top_k)

    diagnostics = {
        "num_observations": len(observations),
        "num_tracklets": len(tracklets),
        "num_entities": len(entities),
        "tracker_model": global_config.get("entity_tracking_model", "yolov8n.pt"),
        "tracker": global_config.get("entity_tracking_tracker", "botsort.yaml"),
    }
    result = EntityAnchorResult(
        video_name=video_name,
        entities=entities,
        tracklets=tracklets,
        segment_memory=segment_memory,
        diagnostics=diagnostics,
    )

    working_dir = global_config["working_dir"]
    configured_dir = global_config.get("entity_anchor_storage_dir", "entity_anchor")
    output_dir = _storage_dir(working_dir, configured_dir)
    _write_json(os.path.join(output_dir, f"{video_name}_entity_anchor.json"), result.to_dict())
    _write_json(os.path.join(output_dir, f"{video_name}_segment_memory.json"), segment_memory)

    logger.info(
        f"[Entity Anchoring] {video_name}: "
        f"{diagnostics['num_tracklets']} tracklets -> {diagnostics['num_entities']} entities."
    )
    return result
