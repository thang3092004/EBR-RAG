import os
import json
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

# Setup robust flash_attn mock to prevent importlib.util.find_spec errors
flash_attn_spec = ModuleSpec("flash_attn", None)
flash_attn_mock = MagicMock()
flash_attn_mock.__spec__ = flash_attn_spec
flash_attn_mock.__path__ = []
sys.modules["flash_attn"] = flash_attn_mock
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
sys.modules["flash_attn.bert_padding"] = MagicMock()

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

from .._utils import log_openai_usage

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames


def _get_segment_entity_memory(segment_entity_memory, index):
    if not segment_entity_memory:
        return ""
    return segment_entity_memory.get(str(index), segment_entity_memory.get(index, ""))


def _caption_entity_instruction(entity_memory):
    if not entity_memory:
        return ""
    return (
        "Tracked entity IDs for this segment:\n"
        f"{entity_memory}\n\n"
        "When describing visible tracked entities, use the exact IDs above "
        "(for example PERSON_001 or OBJECT_003) instead of pronouns like he, she, or it. "
        "If an action involves a tracked entity, attach the action to the ID.\n\n"
    )


def _usage_token_count(usage, key):
    if not usage:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def _openai_usage_cost_usd(model_name, input_tokens, output_tokens):
    if model_name == "gpt-4o-mini":
        return (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.60)
    return 0.0


def _safe_checkpoint_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _caption_checkpoint_path(working_dir, video_name, model_name):
    if not working_dir:
        return None
    checkpoint_dir = os.path.join(working_dir, "_caption_checkpoints")
    filename = f"{_safe_checkpoint_name(video_name)}__{_safe_checkpoint_name(model_name)}.json"
    return os.path.join(checkpoint_dir, filename)


def _load_caption_checkpoint(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(k): v for k, v in data.get("captions", {}).items() if v}


def _save_caption_checkpoint(path, model_name, captions):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model_name,
        "captions": {str(k): v for k, v in captions.items()},
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def segment_caption(
    video_name,
    video_path,
    segment_index2name,
    transcripts,
    segment_times_info,
    caption_result,
    error_queue,
    segment_entity_memory=None,
    working_dir=None,
):
    if os.environ.get("USE_GPT4O_CAPTION") == "True":
        import requests
        import base64
        from io import BytesIO

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        caption_model_name = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")
        checkpoint_path = _caption_checkpoint_path(working_dir, video_name, caption_model_name)
        caption_checkpoint = _load_caption_checkpoint(checkpoint_path)
        for cached_index, cached_caption in caption_checkpoint.items():
            if cached_index in segment_index2name:
                caption_result[cached_index] = cached_caption

        def encode_frame_to_base64(frame_array):
            img = Image.fromarray(frame_array.astype("uint8")).resize((640, 360))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            with VideoFileClip(video_path) as video:
                prompt_tokens_total = 0
                completion_tokens_total = 0
                caption_cost_usd = 0.0
                progress = tqdm(
                    segment_index2name,
                    desc=f"{caption_model_name} Captioning {video_name}",
                    unit="seg",
                )
                progress.set_postfix({"cached": len(caption_checkpoint)})
                for index in progress:
                    index_key = str(index)
                    if index_key in caption_checkpoint:
                        caption_result[index] = caption_checkpoint[index_key]
                        continue

                    frame_times = segment_times_info[index]["frame_times"]

                    # Láº¥y tá»‘i Ä‘a 5 frame Ä‘á»ƒ tiáº¿t kiá»‡m token
                    sampled_times = frame_times[::max(1, len(frame_times) // 5)][:5]
                    frame_b64_list = []
                    for t in sampled_times:
                        frame = video.get_frame(t)
                        frame_b64_list.append(encode_frame_to_base64(frame))

                    transcript = transcripts[index]
                    entity_memory = _get_segment_entity_memory(segment_entity_memory, index)
                    entity_instruction = _caption_entity_instruction(entity_memory)

                    # Build message vá»›i áº£nh vÃ  transcript
                    content = []
                    for b64 in frame_b64_list:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                        })
                    content.append({
                        "type": "text",
                        "text": (
                            entity_instruction
                            + f"The transcript of the current video segment:\n{transcript}\n\n"
                            "Based on the frames above and the transcript, provide a concise but detailed "
                            "description (caption) of what is happening in this video segment in English. "
                            "Focus on visual details, actions, and key information."
                        )
                    })

                    payload = {
                        "model": caption_model_name,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 300,
                        "temperature": 0.2,
                    }

                    resp = requests.post(
                        f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    response_json = resp.json()
                    usage = response_json.get("usage")
                    log_openai_usage(
                        caption_model_name,
                        "chat.completions.caption",
                        usage,
                        {"video_name": video_name, "segment_index": index},
                    )
                    prompt_tokens_total += _usage_token_count(usage, "prompt_tokens")
                    completion_tokens_total += _usage_token_count(usage, "completion_tokens")
                    caption_cost_usd += _openai_usage_cost_usd(
                        caption_model_name,
                        _usage_token_count(usage, "prompt_tokens"),
                        _usage_token_count(usage, "completion_tokens"),
                    )
                    progress.set_postfix(
                        {
                            "in_tok": prompt_tokens_total,
                            "out_tok": completion_tokens_total,
                            "caption_usd": f"{caption_cost_usd:.4f}",
                        }
                    )
                    caption_text = response_json["choices"][0]["message"]["content"].strip()
                    caption_text = caption_text.replace("\n", " ")
                    caption_result[index] = caption_text
                    caption_checkpoint[index_key] = caption_text
                    _save_caption_checkpoint(checkpoint_path, caption_model_name, caption_checkpoint)

        except Exception as e:
            error_queue.put(f"Error in segment_caption_gpt4o:\n{str(e)}")
            raise RuntimeError(str(e))
        return

    try:
        model_path = os.path.abspath("./MiniCPM-V-2_6-int4")
        if not os.path.exists(model_path):
            model_path = "openbmb/MiniCPM-V-2_6-int4"
        checkpoint_model_name = "MiniCPM-V-2_6-int4"
        checkpoint_path = _caption_checkpoint_path(working_dir, video_name, checkpoint_model_name)
        caption_checkpoint = _load_caption_checkpoint(checkpoint_path)
        for cached_index, cached_caption in caption_checkpoint.items():
            if cached_index in segment_index2name:
                caption_result[cached_index] = cached_caption

        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()

        with VideoFileClip(video_path) as video:
            progress = tqdm(segment_index2name, desc=f"Captioning Video {video_name}")
            progress.set_postfix({"cached": len(caption_checkpoint)})
            for index in progress:
                index_key = str(index)
                if index_key in caption_checkpoint:
                    caption_result[index] = caption_checkpoint[index_key]
                    continue

                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                entity_memory = _get_segment_entity_memory(segment_entity_memory, index)
                query = (
                    _caption_entity_instruction(entity_memory)
                    + f"The transcript of the current video:\n{segment_transcript}.\n"
                    "Now provide a description (caption) of the video in English."
                )
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                caption_text = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                caption_text = caption_text.replace("\n", "").replace("<|endoftext|>", "")
                caption_result[index] = caption_text
                caption_checkpoint[index_key] = caption_text
                _save_caption_checkpoint(checkpoint_path, checkpoint_model_name, caption_checkpoint)
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(
    segment_index2name,
    segment_times_info,
    transcripts,
    captions,
    segment_entity_memory=None,
):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        entity_memory = _get_segment_entity_memory(segment_entity_memory, index)
        entity_memory_section = f"Entity Memory:\n{entity_memory}\n" if entity_memory else ""
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = (
            f"{entity_memory_section}"
            f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        )
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["entity_memory"] = entity_memory
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments

def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()

    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        entity_memory = video_segments._data[video_name][index].get("entity_memory", "")
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = (
            _caption_entity_instruction(entity_memory)
            + f"The transcript of the current video:\n{segment_transcript}.\n"
            f"Now provide a very detailed description (caption) of the video in English "
            f"and extract relevant information about: {refine_knowledge}'"
        )
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        entity_memory_section = f"Entity Memory:\n{entity_memory}\n" if entity_memory else ""
        caption_result[this_segment] = (
            f"{entity_memory_section}"
            f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        )
        torch.cuda.empty_cache()

    return caption_result
