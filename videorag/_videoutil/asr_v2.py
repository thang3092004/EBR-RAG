from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Any


def _default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def transcribe_full_video(
    video_path: str,
    config: dict[str, Any],
    *,
    progress=None,
) -> dict[str, Any]:
    """Transcribe the original media once and keep absolute word timestamps."""
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Full-video ASR requires torch and faster-whisper."
        ) from exc

    configured_model = config.get(
        "asr_model",
        config.get("whisper_model", "Systran/faster-distil-whisper-large-v3"),
    )
    local_model = os.path.abspath("./faster-distil-whisper-large-v3")
    model_name = local_model if os.path.exists(local_model) else configured_model
    requested_device = str(config.get("asr_device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    compute_type = str(
        config.get("asr_compute_type") or _default_compute_type(device)
    )
    language = config.get("asr_language") or None
    vad_filter = bool(config.get("asr_vad_filter", True))

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        video_path,
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        beam_size=int(config.get("asr_beam_size", 5)),
    )

    words: list[dict[str, Any]] = []
    utterances: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments_iter):
        segment_words = []
        for word in segment.words or []:
            start = float(word.start if word.start is not None else segment.start)
            end = float(word.end if word.end is not None else segment.end)
            probability = float(getattr(word, "probability", 0.0) or 0.0)
            item = {
                "start": start,
                "end": end,
                "text": str(word.word or "").strip(),
                "probability": probability,
                "segment_index": segment_index,
            }
            if item["text"]:
                words.append(item)
                segment_words.append(item)
        utterance = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": str(segment.text or "").strip(),
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(
                getattr(segment, "no_speech_prob", 0.0) or 0.0
            ),
            "words": segment_words,
        }
        utterances.append(utterance)
        if progress is not None:
            progress.set(
                utterance["end"],
                utterances=len(utterances),
                words=len(words),
            )

    result = {
        "model": str(model_name),
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", language),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "vad_filter": vad_filter,
        "words": words,
        "utterances": utterances,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def assign_words_to_segments(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    assigned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    segment_index = 0
    ordered = sorted(segments, key=lambda item: float(item["start"]))
    for word in sorted(words, key=lambda item: float(item["start"])):
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        while (
            segment_index + 1 < len(ordered)
            and midpoint >= float(ordered[segment_index]["end"])
        ):
            segment_index += 1
        if not ordered:
            break
        segment = ordered[segment_index]
        if float(segment["start"]) <= midpoint <= float(segment["end"]):
            assigned[str(segment["segment_id"])].append(word)
    return dict(assigned)
