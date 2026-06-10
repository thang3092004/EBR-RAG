# Ablation Matrix — Protocol and Run Scripts

## Collections Used

`longervideos/dataset.json` — collections 0, 6, 11 only.

| ID | Folder | Type | Videos | Questions | Evidence Type |
|---|---|---|---:|---:|---|
| 0 | `fights-in-animal-kingdom` | Documentary | 1 | 11 | Visual-heavy |
| 6 | `daubechies-wavelet-lecture` | Lecture | 4 | 25 | Transcript + slides |
| 11 | `primetime-emmy-awards` | Entertainment | 3 | 17 | Mixed modal |

Total: 8 videos, 53 questions. These three represent the three evidence modes deliberately.

## INGESTION_PROFILES (`videorag/ablation.py`)

These profiles control how videos are ingested. Each produces a separate artifact set.

| Profile key | Config overrides | What it measures |
|---|---|---|
| `full_framework` | _(none)_ | Unified V2 baseline |
| `no_adaptive_segmentation` | `segmentation_strategy="fixed"`, `fixed_segment_seconds=30.0` | Value of adaptive segmentation |
| `no_transcript_memory` | `disable_transcript_memory=True` | Value of discourse memory / coreference |
| `no_visual_identity_linking` | `disable_visual_identity_linking=True` | Value of cross-tracklet identity |
| `no_crossmodal_alignment` | `disable_crossmodal_alignment=True` | Value of visual-text entity alignment |

The `video_rag_baseline` scenario uses a **separate** ingestion (the legacy non-Unified pipeline, `use_unified_graph=False`).

## QUERY_SCENARIOS (`videorag/ablation.py`)

These map onto ingestion artifacts (via `artifact_profile`) and query-time settings.

| Scenario key | Artifact profile | Query mode | Extra config |
|---|---|---|---|
| `video_rag_baseline` | `video_rag_baseline` | `videorag` | — |
| `full_framework` | `full_framework` | `EBR_RAG` | — |
| `no_adaptive_segmentation` | `no_adaptive_segmentation` | `EBR_RAG` | — |
| `no_transcript_memory` | `no_transcript_memory` | `EBR_RAG` | — |
| `no_visual_identity_linking` | `no_visual_identity_linking` | `EBR_RAG` | — |
| `no_crossmodal_alignment` | `no_crossmodal_alignment` | `EBR_RAG` | — |
| `no_debate_refinement` | **full_framework** | `EBR_RAG` | `max_rounds=0` |
| `critique_sees_evidence` | **full_framework** | `EBR_RAG` | `debate_critique_see_evidence=True` |

A6 (`no_debate_refinement`) and A7 (`critique_sees_evidence`) **reuse A1's ingestion artifacts**. This is why 8 query scenarios require only 6 ingestion profiles.

## Ablation IDs (Thesis Mapping)

| Thesis ID | Scenario key | Ingestion profile needed |
|---|---|---|
| A0 | `video_rag_baseline` | `video_rag_baseline` (separate legacy ingest) |
| A1 | `full_framework` | `full_framework` |
| A2 | `no_adaptive_segmentation` | `no_adaptive_segmentation` |
| A3 | `no_transcript_memory` | `no_transcript_memory` |
| A4 | `no_visual_identity_linking` | `no_visual_identity_linking` |
| A5 | `no_crossmodal_alignment` | `no_crossmodal_alignment` |
| A6 | `no_debate_refinement` | `full_framework` (reused) |
| A7 | `critique_sees_evidence` | `full_framework` (reused) |

## Artifact Reuse (Stage-Level Sharing)

`reproduce/run_ablation_matrix.py` — `REUSABLE_FULL_STAGES` dict.  
When ingesting for a non-full-framework profile, certain early stages can be **soft-linked** from the `full_framework` artifact set to avoid redundant GPU work:

| Profile | Reusable stages from `full_framework` |
|---|---|
| `no_adaptive_segmentation` | `probe`, `asr`, `shot_detection` |
| `no_transcript_memory` | `probe`, `asr`, `shot_detection`, `segmentation`, `tracking_base`, `frame_selection` |
| `no_visual_identity_linking` | `probe`, `asr`, `shot_detection`, `segmentation`, `text_entities` |
| `no_crossmodal_alignment` | `probe`, `asr`, `shot_detection`, `segmentation`, `tracking_base`, `frame_selection`, `text_entities` |

The runner copies (or symlinks) the shared stage output directories so StageRunner sees them as `done` without re-running.

## Running the Ablation Matrix

```bash
# Default: all 8 scenarios, collections 0 6 11
python reproduce/run_ablation_matrix.py

# Specific collections only
python reproduce/run_ablation_matrix.py --collections 0 6

# Specific scenarios only
python reproduce/run_ablation_matrix.py --scenarios full_framework no_debate_refinement

# Force re-ingest (ignore cached artifacts)
python reproduce/run_ablation_matrix.py --force-ingest

# Ingest only (no query)
python reproduce/run_ablation_matrix.py --ingest-only

# Query only (artifacts already exist)
python reproduce/run_ablation_matrix.py --query-only
```

## Running the Legacy Baseline

```bash
python reproduce/run_baseline_naive.py --collections 0 6 11
```

Despite the script name `run_baseline_naive.py`, it runs `mode="videorag"` (the VideoRAG pipeline, not a simple dense-only naive RAG).

## Output Structure

Answers are written to:
```
reproduce/all_answers/<collection_id>-<description>/answers-<scenario>/answer_<id>.md
```

Each `.md` file contains:
```markdown
# Answer

<answer text>

# Rationale

<rationale text>
```

## Configuration Contract for Controlled Ablations

- Only change the one flag that the scenario measures.
- Never change `initial_text_k`, `initial_graph_k`, `initial_visual_k`, `max_evidence`, `max_rounds` between A1–A7 unless you are specifically running A6 (`max_rounds=0`).
- The `ablation_profile` string in `VideoRAG` config must match the profile name so artifacts are stored in the right namespace.
- If you change retrieval or graph schema, verify artifact compatibility with A0 baseline by checking that `NetworkXStorage` (legacy undirected) still reads correctly with the old graph files.
