from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BoundaryCandidate:
    time: float
    shot: float = 0.0
    pause: float = 0.0
    sentence_end: float = 0.0
    speech_split: float = 0.0
    forced: bool = False

    @property
    def score(self) -> float:
        return (
            0.40 * self.shot
            + 0.35 * self.pause
            + 0.15 * self.sentence_end
            + 0.10 * (1.0 if self.forced else 0.0)
        )


def _inside_word(time_s: float, words: list[dict[str, Any]]) -> bool:
    starts = [float(word["start"]) for word in words]
    index = bisect.bisect_right(starts, time_s) - 1
    if index < 0:
        return False
    word = words[index]
    return float(word["start"]) + 1e-3 < time_s < float(word["end"]) - 1e-3


def _nearest_word_boundary(
    target: float,
    words: list[dict[str, Any]],
    search_radius: float = 2.0,
) -> float:
    options = [
        float(word["end"])
        for word in words
        if abs(float(word["end"]) - target) <= search_radius
    ]
    if not options:
        return target
    return min(options, key=lambda value: abs(value - target))


def build_boundary_candidates(
    duration: float,
    words: list[dict[str, Any]],
    shot_boundaries: list[dict[str, Any]],
    *,
    target: float,
    maximum: float,
) -> list[BoundaryCandidate]:
    merged: dict[float, BoundaryCandidate] = {}

    def candidate_at(time_s: float) -> BoundaryCandidate:
        rounded = round(max(0.0, min(float(time_s), duration)), 3)
        return merged.setdefault(rounded, BoundaryCandidate(time=rounded))

    candidate_at(0.0).forced = True
    candidate_at(duration).forced = True

    for boundary in shot_boundaries:
        item = candidate_at(float(boundary["time"]))
        item.shot = max(item.shot, float(boundary.get("score", 1.0)))

    for index, word in enumerate(words):
        end = float(word["end"])
        next_start = (
            float(words[index + 1]["start"])
            if index + 1 < len(words)
            else duration
        )
        gap = max(0.0, next_start - end)
        text = str(word.get("text", "")).strip()
        item = candidate_at(end)
        item.pause = max(item.pause, min(gap / 1.0, 1.0))
        if text.endswith((".", "?", "!", ":", ";")):
            item.sentence_end = 1.0

    forced_time = target
    while forced_time < duration:
        safe_time = _nearest_word_boundary(forced_time, words)
        item = candidate_at(safe_time)
        item.forced = True
        forced_time += target

    hard_time = maximum
    while hard_time < duration:
        safe_time = _nearest_word_boundary(hard_time, words, search_radius=1.0)
        item = candidate_at(safe_time)
        item.forced = True
        hard_time += maximum

    for item in merged.values():
        item.speech_split = 1.0 if _inside_word(item.time, words) else 0.0
    return sorted(merged.values(), key=lambda item: item.time)


def smart_segment(
    duration: float,
    words: list[dict[str, Any]],
    shot_boundaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    target = float(config.get("segment_target_seconds", 24.0))
    minimum = float(config.get("segment_min_seconds", 8.0))
    maximum = float(config.get("segment_max_seconds", 45.0))
    context = float(config.get("segment_context_seconds", 1.5))
    if not (0 < minimum <= target <= maximum):
        raise ValueError("Segment duration must satisfy 0 < min <= target <= max.")

    candidates = build_boundary_candidates(
        duration,
        words,
        shot_boundaries,
        target=target,
        maximum=maximum,
    )
    count = len(candidates)
    costs = [math.inf] * count
    previous: list[int | None] = [None] * count
    costs[0] = 0.0

    for right in range(1, count):
        right_time = candidates[right].time
        for left in range(right - 1, -1, -1):
            left_time = candidates[left].time
            length = right_time - left_time
            if length > maximum + 1e-6:
                break
            is_final = right == count - 1
            if length < minimum and not is_final:
                continue
            if not math.isfinite(costs[left]):
                continue
            length_cost = ((length - target) / max(target, 1e-6)) ** 2
            short_final_penalty = (
                ((minimum - length) / minimum) ** 2 if is_final and length < minimum else 0.0
            )
            transition_cost = (
                length_cost
                + short_final_penalty
                - candidates[right].score
                + 4.0 * candidates[right].speech_split
            )
            total = costs[left] + transition_cost
            if total < costs[right]:
                costs[right] = total
                previous[right] = left

    if previous[-1] is None:
        # A deterministic hard fallback for pathological timestamp input.
        boundaries = [0.0]
        now = maximum
        while now < duration:
            boundaries.append(_nearest_word_boundary(now, words, search_radius=1.0))
            now += maximum
        boundaries.append(duration)
    else:
        indices = []
        cursor: int | None = count - 1
        while cursor is not None:
            indices.append(cursor)
            cursor = previous[cursor]
        indices.reverse()
        boundaries = [candidates[index].time for index in indices]

    segments: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        segment_id = f"SEG_{index:05d}"
        segments.append(
            {
                "segment_id": segment_id,
                "index": index,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "context_start": max(0.0, float(start) - context),
                "context_end": min(duration, float(end) + context),
            }
        )
    return segments


def fixed_segment(
    duration: float,
    *,
    length: float = 30.0,
    context: float = 1.5,
) -> list[dict[str, Any]]:
    if duration < 0:
        raise ValueError("Video duration cannot be negative.")
    if length <= 0:
        raise ValueError("Fixed segment length must be positive.")
    if duration == 0:
        return []

    segments = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + length, duration)
        segments.append(
            {
                "segment_id": f"SEG_{index:05d}",
                "index": index,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "context_start": max(0.0, float(start) - context),
                "context_end": min(duration, float(end) + context),
            }
        )
        start = end
        index += 1
    return segments
