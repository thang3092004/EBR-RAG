"""Lazy video utility exports.

ImageBind and MiniCPM are intentionally imported only by the stages that need
them. This keeps light-weight tasks such as segmentation and manifest
inspection usable without loading the full vision stack.
"""

_EXPORTS = {
    "split_video": (".split", "split_video"),
    "saving_video_segments": (".split", "saving_video_segments"),
    "speech_to_text": (".asr", "speech_to_text"),
    "segment_caption": (".caption", "segment_caption"),
    "merge_segment_information": (".caption", "merge_segment_information"),
    "retrieved_segment_caption": (".caption", "retrieved_segment_caption"),
    "encode_video_segments": (".feature", "encode_video_segments"),
    "encode_string_query": (".feature", "encode_string_query"),
    "probe_video": (".media_probe", "probe_video"),
    "transcribe_full_video": (".asr_v2", "transcribe_full_video"),
    "assign_words_to_segments": (".asr_v2", "assign_words_to_segments"),
    "detect_shots_and_motion": (".shot_detection", "detect_shots_and_motion"),
    "smart_segment": (".smart_segment", "smart_segment"),
    "select_segment_frames": (".frame_selector", "select_segment_frames"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attribute = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
