from __future__ import annotations


def build_audio_entity_memory(*args, **kwargs) -> dict[str, str]:
    """Placeholder for SPEAKER_ID memory.

    Audio is deliberately optional in the first implementation because CG-Bench
    mini is mostly visual/OCR/entity-action oriented. This function gives the
    next phase a stable hook for WhisperX/pyannote/TalkNet integration.
    """
    return {}
