from __future__ import annotations

from typing import Any


BASELINE_SCENARIO = "video_rag_baseline"
FULL_SCENARIO = "full_framework"


INGESTION_PROFILES: dict[str, dict[str, Any]] = {
    FULL_SCENARIO: {
        "ablation_profile": FULL_SCENARIO,
    },
    "no_adaptive_segmentation": {
        "ablation_profile": "no_adaptive_segmentation",
        "segmentation_strategy": "fixed",
        "fixed_segment_seconds": 30.0,
    },
    "no_entity_memory": {
        "ablation_profile": "no_entity_memory",
        "entity_memory_max_context": 0,
    },
    "no_gleaning": {
        "ablation_profile": "no_gleaning",
        "extraction_gleaning_rounds": 0,
    },
    "no_crossmodal_alignment": {
        "ablation_profile": "no_crossmodal_alignment",
        "disable_crossmodal_alignment": True,
    },
}


QUERY_SCENARIOS: dict[str, dict[str, Any]] = {
    BASELINE_SCENARIO: {
        "artifact_profile": BASELINE_SCENARIO,
        "mode": "videorag",
    },
    FULL_SCENARIO: {
        "artifact_profile": FULL_SCENARIO,
        "mode": "EBR_RAG",
    },
    "no_adaptive_segmentation": {
        "artifact_profile": "no_adaptive_segmentation",
        "mode": "EBR_RAG",
    },
    "no_entity_memory": {
        "artifact_profile": "no_entity_memory",
        "mode": "EBR_RAG",
    },
    "no_gleaning": {
        "artifact_profile": "no_gleaning",
        "mode": "EBR_RAG",
    },
    "no_crossmodal_alignment": {
        "artifact_profile": "no_crossmodal_alignment",
        "mode": "EBR_RAG",
    },
    "no_debate_refinement": {
        "artifact_profile": FULL_SCENARIO,
        "mode": "EBR_RAG",
        "max_rounds": 0,
    },
    "critique_sees_evidence": {
        "artifact_profile": FULL_SCENARIO,
        "mode": "EBR_RAG",
        "debate_critique_see_evidence": True,
    },
    "defender_no_tools": {
        "artifact_profile": FULL_SCENARIO,
        "mode": "EBR_RAG",
        "debate_defender_disable_tools": True,
    },
}


def ingestion_profile_overrides(profile: str) -> dict[str, Any]:
    try:
        return dict(INGESTION_PROFILES[profile])
    except KeyError as exc:
        valid = ", ".join(INGESTION_PROFILES)
        raise ValueError(
            f"Unknown ingestion profile {profile!r}. Valid profiles: {valid}"
        ) from exc


def query_scenario_config(scenario: str) -> dict[str, Any]:
    try:
        return dict(QUERY_SCENARIOS[scenario])
    except KeyError as exc:
        valid = ", ".join(QUERY_SCENARIOS)
        raise ValueError(
            f"Unknown query scenario {scenario!r}. Valid scenarios: {valid}"
        ) from exc
