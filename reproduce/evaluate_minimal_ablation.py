from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from videorag.ablation import FULL_SCENARIO, QUERY_SCENARIOS
from videorag.pipeline.stage_runner import atomic_write_json, read_json


DEFAULT_COLLECTIONS = ("0", "6", "11")
DEFAULT_COMPARISONS = tuple(
    scenario
    for scenario in QUERY_SCENARIOS
    if scenario != FULL_SCENARIO
)

SYSTEM_PROMPT = """You are a blind evaluator of answers to long-video questions.
Judge only answer quality. Do not infer which system produced an answer.
Prefer answers that are relevant, factually cautious, complete, evidence-grounded,
and concise. Return JSON only."""

USER_TEMPLATE = """Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate both answers using:
- relevance
- factual_caution
- completeness
- evidence_grounding
- conciseness

Return this JSON object:
{{
  "winner": "A" | "B" | "tie",
  "scores": {{
    "A": {{
      "relevance": 1,
      "factual_caution": 1,
      "completeness": 1,
      "evidence_grounding": 1,
      "conciseness": 1
    }},
    "B": {{
      "relevance": 1,
      "factual_caution": 1,
      "completeness": 1,
      "evidence_grounding": 1,
      "conciseness": 1
    }}
  }},
  "reason": "brief explanation"
}}

Each score must be an integer from 1 to 5."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collection(dataset: dict, collection_id: str) -> dict:
    item = dict(dataset[str(collection_id)][0])
    item["collection_id"] = str(collection_id)
    item["slug"] = f"{collection_id}-{item['description']}"
    return item


def _answer_path(
    answers_root: Path,
    collection: dict,
    scenario: str,
    question_id: str | int,
) -> Path:
    return (
        answers_root
        / collection["slug"]
        / scenario
        / f"answer_{question_id}.md"
    )


def _comparison_order(key: str) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return digest[0] % 2 == 0


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "complete":
                ids.add(str(payload.get("comparison_id", "")))
    return ids


async def _judge(client, semaphore, request: dict, model: str) -> dict:
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request["prompt"]},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                result = json.loads(
                    response.choices[0].message.content or "{}"
                )
                winner_label = str(result.get("winner", "tie")).upper()
                winner_scenario = (
                    request["label_to_scenario"].get(winner_label)
                    if winner_label in {"A", "B"}
                    else "tie"
                )
                return {
                    **request["metadata"],
                    "status": "complete",
                    "created_at": _utc_now(),
                    "winner_label": winner_label,
                    "winner_scenario": winner_scenario,
                    "judge": result,
                }
            except Exception as exc:
                if attempt == 2:
                    return {
                        **request["metadata"],
                        "status": "failed",
                        "created_at": _utc_now(),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                await asyncio.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _requests(dataset: dict, args, existing: set[str]) -> list[dict]:
    requests = []
    for collection_id in args.collections:
        collection = _collection(dataset, collection_id)
        for scenario in args.comparisons:
            for question in collection.get("questions", []):
                full_path = _answer_path(
                    args.answers_root,
                    collection,
                    FULL_SCENARIO,
                    question["id"],
                )
                ablation_path = _answer_path(
                    args.answers_root,
                    collection,
                    scenario,
                    question["id"],
                )
                if not full_path.is_file() or not ablation_path.is_file():
                    continue
                full_answer = full_path.read_text(encoding="utf-8")
                ablation_answer = ablation_path.read_text(encoding="utf-8")
                for repeat in range(args.repeats):
                    comparison_id = (
                        f"{collection['slug']}::q{question['id']}::"
                        f"{FULL_SCENARIO}_vs_{scenario}::r{repeat}"
                    )
                    if comparison_id in existing:
                        continue
                    full_is_a = _comparison_order(comparison_id)
                    if full_is_a:
                        answer_a, answer_b = full_answer, ablation_answer
                        label_to_scenario = {
                            "A": FULL_SCENARIO,
                            "B": scenario,
                        }
                    else:
                        answer_a, answer_b = ablation_answer, full_answer
                        label_to_scenario = {
                            "A": scenario,
                            "B": FULL_SCENARIO,
                        }
                    requests.append(
                        {
                            "metadata": {
                                "comparison_id": comparison_id,
                                "collection_id": collection["collection_id"],
                                "collection": collection["slug"],
                                "category": collection.get("type", "unknown"),
                                "question_id": question["id"],
                                "scenario": scenario,
                                "repeat": repeat,
                                "label_to_scenario": label_to_scenario,
                            },
                            "label_to_scenario": label_to_scenario,
                            "prompt": USER_TEMPLATE.format(
                                question=question["question"],
                                answer_a=answer_a,
                                answer_b=answer_b,
                            ),
                        }
                    )
    return requests


def _write_summary(results_path: Path, summary_path: Path) -> None:
    grouped: dict[tuple[str, str], dict] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "complete":
                continue
            key = (str(row["collection"]), str(row["scenario"]))
            group = grouped.setdefault(
                key,
                {
                    "collection": key[0],
                    "category": row.get("category", "unknown"),
                    "scenario": key[1],
                    "full_wins": 0,
                    "ties": 0,
                    "full_losses": 0,
                    "full_scores": [],
                    "ablation_scores": [],
                },
            )
            winner = row.get("winner_scenario")
            if winner == FULL_SCENARIO:
                group["full_wins"] += 1
            elif winner == "tie":
                group["ties"] += 1
            else:
                group["full_losses"] += 1
            label_map = row.get("label_to_scenario", {})
            scores = row.get("judge", {}).get("scores", {})
            for label in ("A", "B"):
                values = scores.get(label, {})
                if not isinstance(values, dict):
                    continue
                numeric = [
                    float(value)
                    for value in values.values()
                    if isinstance(value, (int, float))
                ]
                if not numeric:
                    continue
                target = (
                    "full_scores"
                    if label_map.get(label) == FULL_SCENARIO
                    else "ablation_scores"
                )
                group[target].append(statistics.fmean(numeric))
    rows = []
    for group in grouped.values():
        total = (
            group["full_wins"]
            + group["ties"]
            + group["full_losses"]
        )
        rows.append(
            {
                **{
                    key: value
                    for key, value in group.items()
                    if key not in {"full_scores", "ablation_scores"}
                },
                "comparisons": total,
                "full_win_rate": group["full_wins"] / max(total, 1),
                "full_non_loss_rate": (
                    group["full_wins"] + group["ties"]
                )
                / max(total, 1),
                "mean_full_score": (
                    statistics.fmean(group["full_scores"])
                    if group["full_scores"]
                    else 0.0
                ),
                "mean_ablation_score": (
                    statistics.fmean(group["ablation_scores"])
                    if group["ablation_scores"]
                    else 0.0
                ),
            }
        )
    atomic_write_json(
        summary_path,
        {"created_at": _utc_now(), "rows": rows},
    )


async def _run(args) -> None:
    from openai import AsyncOpenAI

    dataset = _load_dataset(args.dataset)
    existing = _existing_ids(args.output)
    requests = _requests(dataset, args, existing)
    if not requests:
        print("No pending answer pairs to evaluate.")
        if args.output.exists():
            _write_summary(args.output, args.summary)
        return
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            _judge(client, semaphore, request, args.model)
        )
        for request in requests
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Blind pairwise evaluation",
        ):
            result = await task
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    await client.close()
    _write_summary(args.output, args.summary)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe blind pairwise evaluation: Full Framework versus "
            "each baseline/ablation."
        )
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_COLLECTIONS),
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        choices=DEFAULT_COMPARISONS,
        default=list(DEFAULT_COMPARISONS),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "longervideos" / "dataset.json",
    )
    parser.add_argument(
        "--answers-root",
        type=Path,
        default=ROOT / "reproduce" / "minimal_ablation_answers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproduce"
        / "minimal_ablation_answers"
        / "pairwise_judgments.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / "reproduce"
        / "minimal_ablation_answers"
        / "pairwise_summary.json",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.answers_root = args.answers_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.summary = args.summary.expanduser().resolve()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
