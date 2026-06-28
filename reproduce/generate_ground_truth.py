"""
Generate ground truth answers for RAGAS evaluation using GPT-4o.

Reads full transcript + visual captions from ingested data, then asks GPT-4o
to answer each benchmark question comprehensively.

Usage:
    # Collection 0 only (11 questions) — evaluate quality first
    python reproduce/generate_ground_truth.py --collections 0

    # All 3 collections
    python reproduce/generate_ground_truth.py --collections 0 6 11

    # Resume after crash (skips already-generated answers)
    python reproduce/generate_ground_truth.py --collections 0

    # Use a different model
    python reproduce/generate_ground_truth.py --collections 0 --model gpt-4o-mini

Requires: OPENAI_API_KEY environment variable set.
Output:  reproduce/all_answers/<collection>/answers-groundtruth/answer_<id>.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "longervideos" / "dataset.json"
WORKDIR_BASE = ROOT / "longervideos" / "videorag-workdir"
OUTPUT_BASE = ROOT / "reproduce" / "all_answers"

COLLECTION_FOLDERS = {
    "0": "0-fights-in-animal-kingdom",
    "6": "6-daubechies-wavelet-lecture",
    "11": "11-primetime-emmy-awards",
}

SAMPLE_QUESTION = "Describe the dramatic fight between two male ibex competing for access to females at a waterhole."

SAMPLE_ANSWER = """\
### The Ibex Battle for Dominance

The dramatic encounter between two male ibex is a stark display of power and competition, typical during the mating rituals of these animals. Set against a rugged, rocky landscape, the scene unfolds near a vital waterhole that serves as a crucial resource, especially during the hot summer months. The struggle highlights the importance of access to water not just for hydration but as a locus where females congregate, making it a prime spot for mating opportunities.

#### The Clash

As the video footage reveals, the male ibex engage in a fierce head-to-head battle, showcasing their impressive horns that are specially reinforced to withstand the impacts of such confrontations. Initially, both males stand poised and ready, muscles tense, as they gauge each other's strength and resolve. The tension in the air is palpable; neither ibex is willing to back down as they prepare for the formidable challenge ahead.

Once the clash begins, both ibex charge at each other with remarkable speed and agility, locking their horns in a show of brute force. Their movements create clouds of dust that rise into the air, underscoring the intensity of the battle. Their interactions are not merely physical but also psychological, as they attempt to outmaneuver each other for dominance.

#### The Outcome

Throughout this encounter, there are moments where one ibex appears to gain a slight upper hand, pushing its opponent back and displaying signs of triumph. However, this victory is fleeting, as the determination of both animals is relentless. The surrounding environment — a rocky terrain dotted with sparse vegetation — serves to highlight the harshness of their habitat, adding an element of drama to the fierce competition.

Overall, the spectacle of the ibex fight encapsulates not just the physical struggle for dominance but also the broader themes of survival, competition, and the sophisticated behaviors exhibited by these remarkable creatures in their natural habitat."""

SYSTEM_PROMPT = f"""\
You are creating GROUND TRUTH reference answers for a long-video question-answering benchmark.

You receive the COMPLETE transcript (what is spoken) and visual captions (what is seen on screen) \
from one or more long videos. Your task: answer each question as thoroughly and accurately as \
possible, using ONLY the provided data.

STRICT RULES:
1. Every factual claim must come from the transcript or captions. Do NOT add external knowledge.
2. Use specific details: names, actions, descriptions, approximate timestamps when relevant.
3. Be comprehensive — if multiple segments discuss the topic, synthesize all of them.
4. Structure with Markdown: use ### headings and #### subheadings for clarity.
5. Write 300-600 words. Enough to be thorough, not so long that it pads.
6. Write directly — NO meta-commentary ("Based on the transcript...", "The captions show..."). \
Just state the facts as if you watched the video.
7. If the data does not contain enough information to fully answer, say so explicitly rather \
than guessing.

EXAMPLE — Question: "{SAMPLE_QUESTION}"

{SAMPLE_ANSWER}

Match this format: Markdown with headings, specific details, 300-600 words, no citations."""


def load_dataset() -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context_for_collection(collection_id: str) -> str:
    """Build combined transcript + captions for all videos in a collection."""
    folder = COLLECTION_FOLDERS[collection_id]
    workdir = WORKDIR_BASE / folder / "full_framework" / "pipeline_v2"
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir not found: {workdir}")

    from videorag._videoutil.asr_v2 import assign_words_to_segments

    parts: list[str] = []
    video_dirs = sorted(
        d for d in workdir.iterdir()
        if d.is_dir() and (d / "asr" / "asr.json").exists()
    )

    for video_dir in video_dirs:
        video_id = video_dir.name
        asr = json.loads((video_dir / "asr" / "asr.json").read_text("utf-8"))
        segments = json.loads(
            (video_dir / "segmentation" / "segments.json").read_text("utf-8")
        )

        captions = {}
        for cap_candidate in [
            video_dir / "alignment_caption" / "segments" / "captions.json",
            video_dir / "alignment_caption" / "captions.json",
        ]:
            if cap_candidate.exists():
                captions = json.loads(cap_candidate.read_text("utf-8"))
                break

        by_segment = assign_words_to_segments(asr["words"], segments)

        parts.append(f"\n{'='*60}\nVIDEO: {video_id}\n{'='*60}")
        for seg in segments:
            sid = seg["segment_id"]
            cap = captions.get(sid, "")
            trans = " ".join(
                w["text"]
                for w in sorted(
                    by_segment.get(sid, []),
                    key=lambda w: float(w["start"]),
                )
            )
            block = f"\n[{seg['start']:.1f}s – {seg['end']:.1f}s] segment {sid}"
            if trans.strip():
                block += f"\nTranscript: {trans}"
            if cap.strip():
                block += f"\nCaption: {cap}"
            parts.append(block)

    return "\n".join(parts)


def count_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token for English)."""
    return len(text) // 4


def compress_caption(caption: str) -> str:
    """Remove MiniCPM boilerplate while preserving factual content."""
    import re

    if not caption:
        return ""

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', caption.strip())

    keep = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        # Skip pure boilerplate openers
        if s.startswith("The video segment showcases") or s.startswith("The video segment captures"):
            # Keep only the part after the boilerplate intro if it has content
            match = re.search(r'(?:showcases|captures|depicts|presents)\s+(.+)', s)
            if match:
                s = match.group(1).strip().capitalize()
            else:
                continue
        # Skip pure atmosphere/filler sentences
        if s.startswith("The overall atmosphere") or s.startswith("Overall, the atmosphere"):
            continue
        if s.startswith("This close-up offers") or s.startswith("This provides a"):
            continue
        # Clean hedging language
        s = re.sub(r',?\s*possibly\s+', ' ', s)
        s = re.sub(r',?\s*likely\s+(?=a |an |the )', ' ', s)
        s = re.sub(r',?\s*suggesting\s+(?:a |that )', ', ', s)
        s = re.sub(r',?\s*indicating\s+(?:a |that |it )', ', ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s) > 10:
            keep.append(s)

    return " ".join(keep)


def fit_context(context: str, token_limit: int = 115_000) -> str:
    """Compress captions to fit within GPT-4o context limit."""
    if count_tokens(context) <= token_limit:
        return context

    lines = context.split("\n")
    compressed = []
    for line in lines:
        if line.startswith("Caption: "):
            raw = line[len("Caption: "):]
            compressed.append("Caption: " + compress_caption(raw))
        else:
            compressed.append(line)

    return "\n".join(compressed)


def truncate_caption(caption: str, max_sentences: int = 3) -> str:
    """Keep only first N sentences of a caption to reduce token count."""
    if not caption:
        return ""
    sentences = caption.replace("\n\n", "\n").split(". ")
    if len(sentences) <= max_sentences:
        return caption
    return ". ".join(sentences[:max_sentences]) + "."


def _drop_caption_when_transcript(context: str) -> str:
    """Drop caption lines for segments that also have a transcript."""
    lines = context.split("\n")
    result = []
    i = 0
    while i < len(lines):
        if (
            lines[i].startswith("Caption: ")
            and i >= 1
            and lines[i - 1].startswith("Transcript: ")
        ):
            i += 1
            continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def fit_context_to_limit(context: str, token_limit: int = 115_000) -> str:
    """Progressively compress context to fit within token limit.

    Priority: transcript > caption. Transcript carries narration/speech
    which is the primary factual source; captions describe visuals.
    """
    if count_tokens(context) <= token_limit:
        return context

    # Step 1: truncate captions (3 → 2 → 1 sentences)
    for max_sent in [3, 2, 1]:
        lines = context.split("\n")
        trimmed = []
        for line in lines:
            if line.startswith("Caption: "):
                cap = line[len("Caption: "):]
                trimmed.append("Caption: " + truncate_caption(cap, max_sent))
            else:
                trimmed.append(line)
        result = "\n".join(trimmed)
        if count_tokens(result) <= token_limit:
            return result

    # Step 2: drop captions entirely for segments that have transcript
    result = _drop_caption_when_transcript(result)
    if count_tokens(result) <= token_limit:
        return result

    # Step 3: drop ALL remaining captions (caption-only segments)
    lines = result.split("\n")
    result = "\n".join(l for l in lines if not l.startswith("Caption: "))
    return result


def call_openai(
    system: str,
    user: str,
    model: str = "gpt-4o",
    max_retries: int = 6,
) -> str:
    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            wait = min(60, 2 ** (attempt + 2))
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            if attempt < max_retries - 1:
                wait = min(30, 2 ** (attempt + 1))
                print(f"    API error: {e}, retry in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def generate_answers(
    collection_id: str,
    questions: list[dict],
    context: str,
    model: str,
):
    from tqdm import tqdm

    folder = COLLECTION_FOLDERS[collection_id]
    output_dir = OUTPUT_BASE / folder / "answers-groundtruth"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx_tokens = count_tokens(context)
    sys_tokens = count_tokens(SYSTEM_PROMPT)
    print(f"  Context: ~{ctx_tokens:,} tokens, System: ~{sys_tokens:,} tokens")
    print(f"  Total input per call: ~{ctx_tokens + sys_tokens:,} tokens")

    if ctx_tokens + sys_tokens > 900_000:
        print(f"  ERROR: Input ~{ctx_tokens + sys_tokens:,} tokens exceeds 1M context limit.")
        sys.exit(1)

    pbar = tqdm(questions, desc=f"  Collection {collection_id}", unit="q")
    generated = 0
    skipped = 0
    for q in pbar:
        qid = q["id"]
        out_path = output_dir / f"answer_{qid}.md"

        if out_path.exists() and out_path.stat().st_size > 50:
            skipped += 1
            pbar.set_postfix(skip=skipped, gen=generated, current=f"Q{qid} exists")
            continue

        pbar.set_postfix(skip=skipped, gen=generated, current=f"Q{qid}...")

        user_msg = (
            f"QUESTION: {q['question']}\n\n"
            f"FULL VIDEO DATA (transcript + visual captions):\n"
            f"{context}"
        )

        answer = call_openai(SYSTEM_PROMPT, user_msg, model=model)
        out_path.write_text(answer, encoding="utf-8")
        generated += 1
        pbar.set_postfix(skip=skipped, gen=generated, current=f"Q{qid} done")

    print(f"  Result: {generated} generated, {skipped} skipped (already exist)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth answers for RAGAS evaluation"
    )
    parser.add_argument(
        "--collections", nargs="+", default=["0"],
        choices=["0", "6", "11"],
        help="Collection IDs (default: 0)",
    )
    parser.add_argument(
        "--model", default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1, 1M context)",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)

    dataset = load_dataset()

    for cid in args.collections:
        folder = COLLECTION_FOLDERS[cid]
        print(f"\n{'='*60}")
        print(f"Collection {cid}: {folder}")
        print(f"{'='*60}")

        questions = dataset[cid][0]["questions"]
        output_dir = OUTPUT_BASE / folder / "answers-groundtruth"
        existing = len(list(output_dir.glob("answer_*.md"))) if output_dir.exists() else 0
        print(f"  Questions: {len(questions)}, already done: {existing}/{len(questions)}")

        if existing == len(questions):
            print("  All done, skipping.")
            continue

        print("  Loading transcript + captions...")
        context = build_context_for_collection(cid)
        print(f"  Loaded: {len(context):,} chars (~{count_tokens(context):,} tokens)")

        model_limits = {"gpt-4o": 115_000, "gpt-4o-mini": 115_000}
        token_limit = model_limits.get(args.model, 900_000)
        context = fit_context_to_limit(context, token_limit=token_limit)
        print(f"  After compression: {len(context):,} chars (~{count_tokens(context):,} tokens)")

        generate_answers(cid, questions, context, model=args.model)

    print(f"\nDone! Answers at: reproduce/all_answers/*/answers-groundtruth/")


if __name__ == "__main__":
    main()
