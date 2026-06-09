from videorag.ablation import (
    BASELINE_SCENARIO,
    FULL_SCENARIO,
    INGESTION_PROFILES,
    QUERY_SCENARIOS,
)


def test_minimal_ablation_matrix_has_eight_query_scenarios():
    assert tuple(QUERY_SCENARIOS) == (
        BASELINE_SCENARIO,
        FULL_SCENARIO,
        "no_adaptive_segmentation",
        "no_transcript_memory",
        "no_visual_identity_linking",
        "no_crossmodal_alignment",
        "no_debate_refinement",
        "critique_sees_evidence",
    )


def test_debate_ablations_share_full_framework_artifacts():
    assert (
        QUERY_SCENARIOS["no_debate_refinement"]["artifact_profile"]
        == FULL_SCENARIO
    )
    assert (
        QUERY_SCENARIOS["critique_sees_evidence"]["artifact_profile"]
        == FULL_SCENARIO
    )


def test_each_graph_ablation_changes_one_major_ingestion_block():
    assert INGESTION_PROFILES["no_adaptive_segmentation"] == {
        "ablation_profile": "no_adaptive_segmentation",
        "segmentation_strategy": "fixed",
        "fixed_segment_seconds": 30.0,
    }
    assert INGESTION_PROFILES["no_transcript_memory"][
        "disable_transcript_memory"
    ]
    assert INGESTION_PROFILES["no_visual_identity_linking"][
        "disable_visual_identity_linking"
    ]
    assert INGESTION_PROFILES["no_crossmodal_alignment"][
        "disable_crossmodal_alignment"
    ]
