from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from videorag.ablation import (
    BASELINE_SCENARIO,
    FULL_SCENARIO,
    INGESTION_PROFILES,
    QUERY_SCENARIOS,
    ingestion_profile_overrides,
    query_scenario_config,
)
from videorag.pipeline.stage_runner import (
    PIPELINE_VERSION,
    atomic_write_json,
    read_json,
    stable_hash,
)


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi")
DEFAULT_COLLECTIONS = ("0", "6", "11")
DEFAULT_INGESTION_PROFILES = (
    BASELINE_SCENARIO,
    FULL_SCENARIO,
    "no_adaptive_segmentation",
    "no_entity_memory",
    "no_gleaning",
    "no_crossmodal_alignment",
)
DEFAULT_QUERY_SCENARIOS = tuple(QUERY_SCENARIOS)
REUSABLE_FULL_STAGES = {
    "no_adaptive_segmentation": (
        "probe",
        "asr",
        "shot_detection",
    ),
    "no_entity_memory": (
        "probe",
        "asr",
        "shot_detection",
        "segmentation",
        "frame_selection",
    ),
    "no_gleaning": (
        "probe",
        "asr",
        "shot_detection",
        "segmentation",
        "frame_selection",
    ),
    "no_crossmodal_alignment": (
        "probe",
        "asr",
        "shot_detection",
        "segmentation",
        "frame_selection",
    ),
}

# Profiles whose MiniCPM captions are identical to A1's.
# captions.json is seeded from full_framework workdir so captioning is skipped.
CAPTION_REUSABLE_PROFILES = {"no_entity_memory", "no_gleaning", "no_crossmodal_alignment"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dataset(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collection(dataset: dict, collection_id: str) -> dict:
    rows = dataset.get(str(collection_id))
    if not rows:
        raise KeyError(f"Collection {collection_id!r} is absent from dataset.")
    item = dict(rows[0])
    item["collection_id"] = str(collection_id)
    item["slug"] = f"{collection_id}-{item['description']}"
    return item


def _youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    values = parse_qs(parsed.query).get("v", [])
    if values:
        return values[0]
    raise ValueError(f"Cannot extract YouTube ID from {url!r}")


def _collection_videos(collection: dict, video_root: Path) -> list[str]:
    video_dir = video_root / collection["slug"] / "videos"
    if not video_dir.is_dir():
        raise FileNotFoundError(
            f"Video directory is missing: {video_dir}\n"
            "Run longervideos/prepare_data.py and download the collection first."
        )
    paths = []
    missing = []
    for url in collection.get("video_url", []):
        video_id = _youtube_id(url)
        matches = sorted(
            path
            for path in video_dir.glob(f"{video_id}.*")
            if path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if len(matches) == 1:
            paths.append(str(matches[0].resolve()))
        elif not matches:
            missing.append(video_id)
        else:
            raise ValueError(
                f"Multiple files match video ID {video_id!r}: "
                + ", ".join(str(path) for path in matches)
            )
    if missing:
        raise FileNotFoundError(
            f"Collection {collection['slug']} is missing videos: "
            + ", ".join(missing)
        )
    return paths


def _workdir(work_root: Path, collection: dict, artifact_profile: str) -> Path:
    return work_root / collection["slug"] / artifact_profile


def _hardlink_or_copy(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _seed_profile_from_full(
    full_workdir: Path,
    target_workdir: Path,
    videos: list[str],
    profile: str,
) -> dict:
    stages = REUSABLE_FULL_STAGES.get(profile, ())
    if not stages:
        return {"status": "not_applicable", "seeded_videos": 0}

    seeded = 0
    skipped = 0
    unavailable = []
    for video_path in videos:
        video_id = Path(video_path).stem
        source_video_dir = full_workdir / "pipeline_v2" / video_id
        target_video_dir = target_workdir / "pipeline_v2" / video_id
        target_manifest_path = target_video_dir / "manifest.json"
        if target_manifest_path.exists():
            skipped += 1
            continue

        source_manifest = read_json(
            source_video_dir / "manifest.json",
            {},
        )
        source_states = source_manifest.get("stages", {})
        source_is_reusable = (
            source_manifest.get("pipeline_version") == PIPELINE_VERSION
            and all(
                source_states.get(stage, {}).get("status") == "done"
                and (source_video_dir / stage).is_dir()
                for stage in stages
            )
        )
        if not source_is_reusable:
            unavailable.append(video_id)
            continue

        target_video_dir.mkdir(parents=True, exist_ok=True)
        seeded_states = {}
        for stage in stages:
            source_stage_dir = source_video_dir / stage
            target_stage_dir = target_video_dir / stage
            if source_stage_dir.is_dir() and not target_stage_dir.exists():
                shutil.copytree(
                    source_stage_dir,
                    target_stage_dir,
                    copy_function=_hardlink_or_copy,
                )
            seeded_states[stage] = json.loads(
                json.dumps(source_states[stage])
            )
            seeded_states[stage]["seeded_from"] = str(
                source_video_dir
            )

        atomic_write_json(
            target_manifest_path,
            {
                "pipeline_version": PIPELINE_VERSION,
                "video_id": video_id,
                "created_at": _utc_now(),
                "seeded_from_profile": FULL_SCENARIO,
                "seeded_stages": list(stages),
                "stages": seeded_states,
            },
        )
        # Seed captions.json from A1 for profiles that reuse MiniCPM captions.
        # Captions are purely frame-based (no entity memory or gleaning config),
        # so they are identical across profiles that share the same segmentation.
        if profile in CAPTION_REUSABLE_PROFILES:
            src_captions = (
                source_video_dir
                / "alignment_caption"
                / "segments"
                / "captions.json"
            )
            if src_captions.exists():
                dst_seg_dir = (
                    target_video_dir / "alignment_caption" / "segments"
                )
                dst_seg_dir.mkdir(parents=True, exist_ok=True)
                dst_captions = dst_seg_dir / "captions.json"
                if not dst_captions.exists():
                    _hardlink_or_copy(str(src_captions), str(dst_captions))

        seeded += 1

    # Seed visual segment feature VDB from full_framework for profiles that
    # share segmentation. ImageBind clip embeddings are identical when the
    # clips (segment boundaries) are identical, so _stage_index can skip the
    # expensive clip extraction + ImageBind embedding entirely.
    if profile in CAPTION_REUSABLE_PROFILES:
        src_vdb = full_workdir / "vdb_video_segment_feature_v2.json"
        dst_vdb = target_workdir / "vdb_video_segment_feature_v2.json"
        if src_vdb.exists() and not dst_vdb.exists():
            target_workdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_vdb), str(dst_vdb))

    return {
        "status": "complete",
        "seeded_videos": seeded,
        "skipped_existing": skipped,
        "unavailable_videos": unavailable,
        "stages": list(stages),
    }


def _new_vrag(
    workdir: Path,
    artifact_profile: str,
    args,
) -> "VideoRAG":
    from videorag import VideoRAG
    from videorag._llm import openai_config

    common = {
        "llm": openai_config,
        "working_dir": str(workdir),
        "caption_model_path": args.caption_model,
    }
    if artifact_profile == BASELINE_SCENARIO:
        return VideoRAG(
            **common,
            use_unified_graph=False,
        )
    return VideoRAG(
        **common,
        use_unified_graph=True,
        asr_model=args.asr_model,
        pipeline_continue_on_error=args.continue_on_error,
        pipeline_strict=args.strict_pipeline,
        keep_segment_cache=args.keep_segment_cache,
        **ingestion_profile_overrides(artifact_profile),
    )


def _sum_nested_counts(target: dict[str, int], values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)):
            target[str(key)] = target.get(str(key), 0) + int(value)


def _collect_artifact_report(
    workdir: Path,
    collection: dict,
    artifact_profile: str,
    vrag: "VideoRAG",
) -> dict:
    report = {
        "created_at": _utc_now(),
        "collection_id": collection["collection_id"],
        "collection": collection["slug"],
        "category": collection.get("type", "unknown"),
        "artifact_profile": artifact_profile,
        "workdir": str(workdir.resolve()),
        "videos_expected": len(collection.get("video_url", [])),
    }
    graph = getattr(
        getattr(vrag, "chunk_entity_relation_graph", None),
        "_graph",
        None,
    )
    if graph is not None:
        report["graph"] = {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        }
    if artifact_profile == BASELINE_SCENARIO:
        report["segments"] = sum(
            len(value)
            for value in getattr(vrag.video_segments, "_data", {}).values()
            if isinstance(value, dict)
        )
        atomic_write_json(workdir / "ablation_artifact_report.json", report)
        return report

    pipeline_root = workdir / "pipeline_v2"
    stage_totals = {
        "segments": 0,
        "segment_boundaries": 0,
        "boundaries_inside_words": 0,
        "alignment_edges": 0,
        "alignment_entities": 0,
    }
    segment_durations = []
    edge_modalities: dict[str, int] = {}
    completed_videos = 0
    for manifest_path in sorted(pipeline_root.glob("*/manifest.json")):
        manifest = read_json(manifest_path, {})
        stages = manifest.get("stages", {})
        if stages and all(
            state.get("status") == "done"
            for state in stages.values()
        ):
            completed_videos += 1
        stage_totals["segments"] += int(
            stages.get("segmentation", {})
            .get("metrics", {})
            .get("segments", 0)
        )
        segments = read_json(
            manifest_path.parent / "segmentation" / "segments.json",
            [],
        )
        asr = read_json(
            manifest_path.parent / "asr" / "asr.json",
            {},
        )
        segment_durations.extend(
            float(segment.get("duration", 0.0))
            for segment in segments
        )
        boundaries = [
            float(segment["end"])
            for segment in segments[:-1]
        ]
        stage_totals["segment_boundaries"] += len(boundaries)
        for boundary in boundaries:
            if any(
                float(word.get("start", 0.0))
                < boundary
                < float(word.get("end", 0.0))
                for word in asr.get("words", [])
            ):
                stage_totals["boundaries_inside_words"] += 1
        alignment = stages.get("alignment_caption", {}).get("metrics", {})
        stage_totals["alignment_edges"] += int(alignment.get("edges", 0))
        alignment_path = (
            manifest_path.parent
            / "alignment_caption"
            / "alignment.json"
        )
        payload = read_json(alignment_path, {})
        stage_totals["alignment_entities"] += len(
            payload.get("registry", {}).get("entities", [])
        )
        for segment in payload.get("segments", {}).values():
            for edge in segment.get("edges", []):
                modalities = "+".join(
                    sorted(edge.get("modalities", []))
                ) or "unknown"
                edge_modalities[modalities] = (
                    edge_modalities.get(modalities, 0) + 1
                )
    report.update(
        {
            "videos_completed": completed_videos,
            "metrics": stage_totals,
            "derived_metrics": {
                "mean_segment_seconds": (
                    statistics.fmean(segment_durations)
                    if segment_durations
                    else 0.0
                ),
                "min_segment_seconds": min(
                    segment_durations,
                    default=0.0,
                ),
                "max_segment_seconds": max(
                    segment_durations,
                    default=0.0,
                ),
                "boundary_word_cut_rate": (
                    stage_totals["boundaries_inside_words"]
                    / max(stage_totals["segment_boundaries"], 1)
                ),
            },
            "edge_modalities": edge_modalities,
        }
    )
    atomic_write_json(workdir / "ablation_artifact_report.json", report)
    return report


def _ingest_profile(
    collection: dict,
    videos: list[str],
    profile: str,
    args,
) -> dict:
    workdir = _workdir(args.work_root, collection, profile)
    workdir.mkdir(parents=True, exist_ok=True)
    seed_report = {"status": "disabled", "seeded_videos": 0}
    if (
        args.reuse_full_artifacts
        and profile in REUSABLE_FULL_STAGES
        and not args.force
        and not args.no_resume
    ):
        seed_report = _seed_profile_from_full(
            _workdir(
                args.work_root,
                collection,
                FULL_SCENARIO,
            ),
            workdir,
            videos,
            profile,
        )
    vrag = _new_vrag(workdir, profile, args)
    started = time.time()
    try:
        if profile == BASELINE_SCENARIO:
            vrag.insert_video(videos)
            reports = []
        else:
            reports = vrag.insert_video(
                videos,
                resume=not args.no_resume,
                restart_stage=args.restart_stage,
                force=args.force,
            )
        artifact_report = _collect_artifact_report(
            workdir,
            collection,
            profile,
            vrag,
        )
        failed_reports = [
            report
            for report in reports
            if report.get("status") not in {"complete", "completed"}
        ]
        return {
            "status": (
                "complete"
                if not failed_reports
                else "partial_failure"
            ),
            "elapsed_seconds": time.time() - started,
            "failed_videos": len(failed_reports),
            "seed_report": seed_report,
            "reports": reports,
            "artifact_report": artifact_report,
        }
    finally:
        if hasattr(vrag, "caption_model"):
            vrag.caption_model = None
            vrag.caption_tokenizer = None
        del vrag
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_ingestion(dataset: dict, args) -> None:
    profiles = tuple(args.ingestion_profiles or DEFAULT_INGESTION_PROFILES)
    if args.strict_pipeline and BASELINE_SCENARIO in profiles:
        raise ValueError(
            "--strict-pipeline applies to Unified V2 profiles only; select "
            "full_framework and/or controlled unified ingestion profiles."
        )
    invalid = [
        profile
        for profile in profiles
        if profile != BASELINE_SCENARIO
        and profile not in INGESTION_PROFILES
    ]
    if invalid:
        raise ValueError(f"Unknown ingestion profiles: {', '.join(invalid)}")

    total = len(args.collections) * len(profiles)
    with tqdm(total=total, desc="Ablation ingestion", unit="artifact") as progress:
        for collection_id in args.collections:
            collection = _collection(dataset, collection_id)
            videos = _collection_videos(collection, args.video_root)
            for profile in profiles:
                progress.set_postfix(
                    collection=collection["slug"],
                    profile=profile,
                    refresh=True,
                )
                try:
                    result = _ingest_profile(
                        collection,
                        videos,
                        profile,
                        args,
                    )
                    atomic_write_json(
                        _workdir(
                            args.work_root,
                            collection,
                            profile,
                        )
                        / "ablation_ingest_result.json",
                        {
                            "collection": collection,
                            "profile": profile,
                            **result,
                        },
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    traceback.print_exc()
                    workdir = _workdir(
                        args.work_root,
                        collection,
                        profile,
                    )
                    workdir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        workdir / "ablation_ingest_result.json",
                        {
                            "status": "failed",
                            "collection": collection,
                            "profile": profile,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    )
                progress.update(1)


def _query_param(scenario: str) -> "QueryParam":
    from videorag import QueryParam

    config = query_scenario_config(scenario)
    if config["mode"] == "videorag":
        return QueryParam(
            mode="videorag",
            wo_reference=True,
        )
    return QueryParam(
        mode="EBR_RAG",
        ebr_top_k=4,
        initial_text_k=4,
        initial_graph_k=4,
        initial_visual_k=4,
        max_evidence=16,
        max_rounds=int(config.get("max_rounds", 2)),
        max_tool_calls_per_round=2,
        max_total_tool_calls=4,
        graph_context_token_cap=1800,
        debate_critique_see_evidence=bool(
            config.get("debate_critique_see_evidence", False)
        ),
        debate_defender_disable_tools=bool(
            config.get("debate_defender_disable_tools", False)
        ),
        debate_single_hypothesis=False,
        return_detailed=True,
        wo_reference=True,
    )


def _result_paths(
    output_root: Path,
    collection: dict,
    scenario: str,
    query_id: str | int,
) -> tuple[Path, Path]:
    directory = output_root / collection["slug"] / scenario
    return (
        directory / f"answer_{query_id}.md",
        directory / f"result_{query_id}.json",
    )


def _result_is_complete(path: Path, config_hash: str) -> bool:
    payload = read_json(path, {})
    return bool(
        payload.get("status") == "complete"
        and payload.get("config_hash") == config_hash
        and str(payload.get("answer", "")).strip()
    )


async def _run_query_scenario(
    vrag: "VideoRAG",
    collection: dict,
    scenario: str,
    question: dict,
    args,
) -> str:
    param = _query_param(scenario)
    query_id = question["id"]
    answer_path, result_path = _result_paths(
        args.output_root,
        collection,
        scenario,
        query_id,
    )
    query_config = {
        "scenario": scenario,
        "artifact_profile": query_scenario_config(scenario)[
            "artifact_profile"
        ],
        "question": question["question"],
        "query_param": asdict(param),
        "workdir": vrag.working_dir,
    }
    config_hash = stable_hash(query_config)
    if not args.no_resume and _result_is_complete(
        result_path,
        config_hash,
    ):
        return "resumed"

    answer_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    response = await vrag.aquery(question["question"], param=param)
    if isinstance(response, dict):
        answer = str(response.get("answer", "")).strip()
        details = response
    else:
        answer = str(response).strip()
        details = {"answer": answer}
    if not answer:
        raise ValueError(
            f"{collection['slug']} {scenario} question {query_id}: empty answer"
        )
    answer_path.write_text(answer, encoding="utf-8")
    atomic_write_json(
        result_path,
        {
            "status": "complete",
            "created_at": _utc_now(),
            "config_hash": config_hash,
            "collection_id": collection["collection_id"],
            "collection": collection["slug"],
            "category": collection.get("type", "unknown"),
            "scenario": scenario,
            "artifact_profile": query_config["artifact_profile"],
            "query_id": query_id,
            "question": question["question"],
            "answer": answer,
            "elapsed_seconds": time.time() - started,
            "query_config": query_config,
            "details": details,
        },
    )
    return "complete"


async def run_queries(dataset: dict, args) -> None:
    scenarios = tuple(args.query_scenarios or DEFAULT_QUERY_SCENARIOS)
    invalid = [
        scenario for scenario in scenarios if scenario not in QUERY_SCENARIOS
    ]
    if invalid:
        raise ValueError(f"Unknown query scenarios: {', '.join(invalid)}")
    total = sum(
        len(_collection(dataset, collection_id).get("questions", []))
        * len(scenarios)
        for collection_id in args.collections
    )
    shared_caption_model = None
    shared_caption_tokenizer = None
    vrag_cache: dict[tuple[str, str], "VideoRAG"] = {}
    try:
        with tqdm(total=total, desc="Ablation queries", unit="question") as progress:
            for collection_id in args.collections:
                collection = _collection(dataset, collection_id)
                for scenario in scenarios:
                    scenario_config = query_scenario_config(scenario)
                    artifact_profile = scenario_config["artifact_profile"]
                    key = (collection["slug"], artifact_profile)
                    if key not in vrag_cache:
                        workdir = _workdir(
                            args.work_root,
                            collection,
                            artifact_profile,
                        )
                        if not workdir.is_dir():
                            raise FileNotFoundError(
                                f"Missing artifact workdir: {workdir}"
                            )
                        vrag = _new_vrag(
                            workdir,
                            artifact_profile,
                            args,
                        )
                        if shared_caption_model is None:
                            vrag.load_caption_model()
                            shared_caption_model = vrag.caption_model
                            shared_caption_tokenizer = vrag.caption_tokenizer
                        else:
                            vrag.caption_model = shared_caption_model
                            vrag.caption_tokenizer = shared_caption_tokenizer
                        vrag_cache[key] = vrag
                    vrag = vrag_cache[key]
                    for question in collection.get("questions", []):
                        progress.set_postfix(
                            collection=collection["slug"],
                            scenario=scenario,
                            query=question["id"],
                            refresh=True,
                        )
                        try:
                            await _run_query_scenario(
                                vrag,
                                collection,
                                scenario,
                                question,
                                args,
                            )
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            traceback.print_exc()
                            _, result_path = _result_paths(
                                args.output_root,
                                collection,
                                scenario,
                                question["id"],
                            )
                            atomic_write_json(
                                result_path,
                                {
                                    "status": "failed",
                                    "created_at": _utc_now(),
                                    "collection": collection["slug"],
                                    "scenario": scenario,
                                    "query_id": question["id"],
                                    "error": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                },
                            )
                        progress.update(1)
    finally:
        for vrag in vrag_cache.values():
            vrag.caption_model = None
            vrag.caption_tokenizer = None
        vrag_cache.clear()
        shared_caption_model = None
        shared_caption_tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def write_status_report(dataset: dict, args) -> Path:
    rows = []
    for collection_id in args.collections:
        collection = _collection(dataset, collection_id)
        for scenario in DEFAULT_QUERY_SCENARIOS:
            complete = 0
            failed = 0
            latencies = []
            rounds = []
            tool_calls = []
            evidence_counts = []
            evidence_by_source: dict[str, int] = {}
            for question in collection.get("questions", []):
                _, result_path = _result_paths(
                    args.output_root,
                    collection,
                    scenario,
                    question["id"],
                )
                payload = read_json(result_path, {})
                if payload.get("status") == "complete":
                    complete += 1
                    latencies.append(
                        float(payload.get("elapsed_seconds", 0.0))
                    )
                    details = payload.get("details", {})
                    if isinstance(details, dict):
                        rounds.append(
                            int(details.get("rounds_run", 0))
                        )
                        tool_calls.append(
                            int(details.get("tool_calls_made", 0))
                        )
                        evidence = details.get("evidence", [])
                        if isinstance(evidence, list):
                            evidence_counts.append(len(evidence))
                            for item in evidence:
                                source = str(
                                    item.get("source")
                                    or item.get("type")
                                    or "unknown"
                                )
                                evidence_by_source[source] = (
                                    evidence_by_source.get(source, 0) + 1
                                )
                elif payload.get("status") == "failed":
                    failed += 1
            rows.append(
                {
                    "collection": collection["slug"],
                    "category": collection.get("type", "unknown"),
                    "scenario": scenario,
                    "questions": len(collection.get("questions", [])),
                    "complete": complete,
                    "failed": failed,
                    "mean_query_seconds": (
                        statistics.fmean(latencies)
                        if latencies
                        else 0.0
                    ),
                    "mean_rounds": (
                        statistics.fmean(rounds)
                        if rounds
                        else 0.0
                    ),
                    "mean_tool_calls": (
                        statistics.fmean(tool_calls)
                        if tool_calls
                        else 0.0
                    ),
                    "mean_evidence_count": (
                        statistics.fmean(evidence_counts)
                        if evidence_counts
                        else 0.0
                    ),
                    "evidence_by_source": evidence_by_source,
                }
            )
    path = args.output_root / "ablation_status.json"
    atomic_write_json(path, {"created_at": _utc_now(), "rows": rows})
    return path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe minimal ablation runner for VideoRAG baseline, "
            "unified graph construction, and debate."
        )
    )
    parser.add_argument(
        "phase",
        choices=("ingest", "query", "all", "status"),
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_COLLECTIONS),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "longervideos" / "dataset.json",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=ROOT / "longervideos",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "longervideos" / "ablation-workdirs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reproduce" / "minimal_ablation_answers",
    )
    parser.add_argument(
        "--ingestion-profiles",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--query-scenarios",
        nargs="+",
        default=None,
    )
    parser.add_argument("--restart-stage")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-segment-cache", action="store_true")
    parser.add_argument(
        "--strict-pipeline",
        action="store_true",
        help=(
            "Disable runtime fallbacks and cross-profile artifact reuse, "
            "require local models, and stop on the first error. Completed "
            "stages may still be resumed unless --no-resume is set."
        ),
    )
    parser.add_argument(
        "--no-reuse-full-artifacts",
        dest="reuse_full_artifacts",
        action="store_false",
        help=(
            "Run every unified profile from scratch instead of seeding "
            "safe completed stages from Full."
        ),
    )
    parser.set_defaults(reuse_full_artifacts=True)
    parser.add_argument(
        "--asr-model",
        default="Systran/faster-distil-whisper-large-v3",
    )
    parser.add_argument("--caption-model", default="./MiniCPM-V-2_6-int4")
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    args = _parse_args()
    if args.strict_pipeline and args.continue_on_error:
        raise ValueError(
            "--strict-pipeline cannot be combined with --continue-on-error."
        )
    if args.strict_pipeline and args.restart_stage:
        raise ValueError(
            "--strict-pipeline cannot be combined with --restart-stage."
        )
    if (
        args.strict_pipeline
        and not Path(args.caption_model).expanduser().is_dir()
    ):
        raise FileNotFoundError(
            "--strict-pipeline requires --caption-model to be a local directory: "
            f"{args.caption_model}"
        )
    if args.strict_pipeline and not Path(args.asr_model).expanduser().is_dir():
        raise FileNotFoundError(
            "--strict-pipeline requires --asr-model to be a local directory: "
            f"{args.asr_model}"
        )
    args.dataset = args.dataset.expanduser().resolve()
    args.video_root = args.video_root.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args.dataset)

    if args.phase in {"ingest", "all"}:
        run_ingestion(dataset, args)
    if args.phase in {"query", "all"}:
        asyncio.run(run_queries(dataset, args))
    status_path = write_status_report(dataset, args)
    print(f"Ablation status: {status_path}")


if __name__ == "__main__":
    main()
