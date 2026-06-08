from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


CONTENT_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _percentile_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(array, 10))
    high = float(np.percentile(array, 90))
    if math.isclose(low, high):
        return [0.0 if value <= 0 else 0.5 for value in values]
    return [
        float(np.clip((value - low) / (high - low), 0.0, 1.0))
        for value in values
    ]


def _within(value: float, start: float, end: float) -> bool:
    return start <= value < end or math.isclose(value, end)


def profile_modalities(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    shot_data: dict[str, Any],
    observations: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, float]] = []
    recent_tokens: list[set[str]] = []

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        duration = max(end - start, 1e-6)
        segment_words = [
            word
            for word in words
            if _within(
                (float(word["start"]) + float(word["end"])) / 2.0,
                start,
                end,
            )
        ]
        boundaries = [
            item
            for item in shot_data.get("boundaries", [])
            if _within(float(item["time"]), start, end)
        ]
        samples = [
            item
            for item in shot_data.get("samples", [])
            if _within(float(item["time"]), start, end)
        ]
        segment_observations = [
            item
            for item in observations
            if _within(float(item.get("time", 0.0)), start, end)
        ]

        track_ids = {
            str(item.get("tracklet_id") or item.get("local_track_id"))
            for item in segment_observations
        }
        entity_changes = len(track_ids)
        motion = (
            float(np.mean([float(item.get("motion_score", 0.0)) for item in samples]))
            if samples
            else 0.0
        )
        text_change = (
            float(
                np.mean(
                    [
                        float(item.get("text_change_score", 0.0))
                        for item in samples
                    ]
                )
            )
            if samples
            else 0.0
        )
        speech_seconds = sum(
            max(
                0.0,
                min(float(word["end"]), end) - max(float(word["start"]), start),
            )
            for word in segment_words
        )
        speech_ratio = min(speech_seconds / duration, 1.0)
        content_tokens = {
            token.lower()
            for word in segment_words
            for token in CONTENT_WORD.findall(str(word.get("text", "")))
        }
        recent_union = set().union(*recent_tokens[-3:]) if recent_tokens else set()
        novelty = (
            len(content_tokens - recent_union) / max(len(content_tokens), 1)
            if content_tokens
            else 0.0
        )
        recent_tokens.append(content_tokens)
        asr_quality = (
            float(
                np.mean(
                    [
                        max(0.0, min(float(word.get("probability", 0.0)), 1.0))
                        for word in segment_words
                    ]
                )
            )
            if segment_words
            else 0.0
        )
        raw_rows.append(
            {
                "shot_rate": len(boundaries) / duration,
                "entity_rate": entity_changes / duration,
                "motion": motion,
                "ocr_change": text_change,
                "speech_ratio": speech_ratio,
                "content_rate": len(content_tokens) / duration,
                "asr_quality": asr_quality,
                "transcript_novelty": novelty,
            }
        )

    feature_names = list(raw_rows[0]) if raw_rows else []
    normalized: dict[str, list[float]] = {
        name: _percentile_normalize([row[name] for row in raw_rows])
        for name in feature_names
    }
    profiles: list[dict[str, Any]] = []
    for index, (segment, raw) in enumerate(zip(segments, raw_rows)):
        visual = (
            0.30 * normalized["shot_rate"][index]
            + 0.30 * normalized["entity_rate"][index]
            + 0.20 * normalized["motion"][index]
            + 0.20 * normalized["ocr_change"][index]
        )
        speech = (
            0.30 * normalized["speech_ratio"][index]
            + 0.25 * normalized["content_rate"][index]
            + 0.25 * normalized["asr_quality"][index]
            + 0.20 * normalized["transcript_novelty"][index]
        )
        balance = (visual - speech) / (visual + speech + 1e-6)
        profiles.append(
            {
                "segment_id": segment["segment_id"],
                "visual_score": visual,
                "speech_score": speech,
                "balance_raw": balance,
                "confidence": max(visual, speech),
                "raw_features": raw,
            }
        )

    for index, profile in enumerate(profiles):
        previous = profiles[max(0, index - 1)]["balance_raw"]
        current = profile["balance_raw"]
        following = profiles[min(len(profiles) - 1, index + 1)]["balance_raw"]
        smoothed = 0.25 * previous + 0.50 * current + 0.25 * following
        profile["balance"] = smoothed
        low_threshold = float(config.get("modality_low_information_threshold", 0.2))
        if (
            profile["visual_score"] < low_threshold
            and profile["speech_score"] < low_threshold
        ):
            mode = "low_information"
        elif smoothed > float(config.get("modality_visual_threshold", 0.35)):
            mode = "visual_rich"
        elif smoothed < float(config.get("modality_speech_threshold", -0.35)):
            mode = "speech_rich"
        else:
            mode = "balanced"
        profile["mode"] = mode
    return profiles
