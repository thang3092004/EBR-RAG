from __future__ import annotations

import json
import gc
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def run_speaker_diarization(
    media_path: str,
    config: dict[str, Any],
    *,
    progress=None,
) -> dict[str, Any]:
    if not config.get("enable_diarization", True):
        return {"available": False, "reason": "disabled", "turns": []}
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        return {
            "available": False,
            "reason": "pyannote_not_installed",
            "turns": [],
        }

    token = (
        config.get("huggingface_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        return {
            "available": False,
            "reason": "huggingface_token_missing",
            "turns": [],
        }
    model_name = str(
        config.get(
            "diarization_model",
            "pyannote/speaker-diarization-3.1",
        )
    )
    try:
        try:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_name, token=token)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"diarization_model_unavailable: {type(exc).__name__}",
            "turns": [],
        }
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    annotation = pipeline(media_path)
    turns = []
    iterable = (
        annotation.itertracks(yield_label=True)
        if hasattr(annotation, "itertracks")
        else annotation.speaker_diarization.itertracks(yield_label=True)
    )
    speaker_map: dict[str, str] = {}
    for turn, _, raw_speaker in iterable:
        raw_speaker = str(raw_speaker)
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"SPEAKER_{len(speaker_map) + 1:03d}"
        turns.append(
            {
                "speaker_id": speaker_map[raw_speaker],
                "raw_speaker": raw_speaker,
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
        if progress is not None:
            progress.set(float(turn.end), speakers=len(speaker_map))
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "available": True,
        "model": model_name,
        "turns": turns,
        "speaker_count": len(speaker_map),
    }


def assign_speakers_to_words(
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            overlap = max(
                0.0,
                min(end, float(turn["end"])) - max(start, float(turn["start"])),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        output.append(
            {
                **word,
                "speaker_id": (
                    best_turn["speaker_id"] if best_turn else "SPEAKER_UNKNOWN"
                ),
                "speaker_overlap": best_overlap,
            }
        )
    return output


def run_talknet_adapter(
    video_path: str,
    tracks_path: str,
    output_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run an optional TalkNet wrapper that writes standardized JSON.

    The configured command may contain {video}, {tracks}, and {output}. The
    wrapper output must contain a `links` list with speaker_id, person_id,
    score, and overlap_seconds.
    """
    if not config.get("enable_talknet", False):
        return {"available": False, "reason": "disabled", "links": []}
    command_template = str(config.get("talknet_command") or "").strip()
    if not command_template:
        return {
            "available": False,
            "reason": "talknet_command_not_configured",
            "links": [],
        }
    command = [
        token.format(
            video=str(Path(video_path).resolve()),
            tracks=str(Path(tracks_path).resolve()),
            output=str(Path(output_path).resolve()),
        )
        for token in shlex.split(command_template)
    ]
    subprocess.run(command, check=True)
    with open(output_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    links = payload.get("links", [])
    if not isinstance(links, list):
        raise ValueError("TalkNet output must contain a list named `links`.")
    return {"available": True, "command": command[0], "links": links}


def accepted_speaker_links(
    talknet_result: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config.get("speaker_person_threshold", 0.80))
    margin = float(config.get("speaker_person_margin", 0.15))
    minimum_overlap = float(config.get("speaker_person_min_overlap", 1.0))
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for link in talknet_result.get("links", []):
        by_speaker.setdefault(str(link["speaker_id"]), []).append(link)
    accepted = []
    for speaker_id, candidates in by_speaker.items():
        ranked = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        best = ranked[0]
        second_score = float(ranked[1].get("score", 0.0)) if len(ranked) > 1 else 0.0
        score = float(best.get("score", 0.0))
        overlap = float(best.get("overlap_seconds", 0.0))
        if (
            score >= threshold
            and score - second_score >= margin
            and overlap >= minimum_overlap
        ):
            accepted.append(
                {
                    "speaker_id": speaker_id,
                    "person_id": str(best["person_id"]),
                    "score": score,
                    "margin": score - second_score,
                    "overlap_seconds": overlap,
                }
            )
    return accepted


def build_audio_entity_memory(
    speaker_links: list[dict[str, Any]],
) -> dict[str, str]:
    return {
        str(link["speaker_id"]): str(link["person_id"])
        for link in speaker_links
    }
