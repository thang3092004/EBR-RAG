"""
Real-time quantitative evaluation — replaces the 4-step batch workflow.

Methodology matches the original paper EXACTLY:
  - Same sys_prompt, user_prompt, response_format (Pydantic schema → Score: int)
  - Same validation: Score must be int in [1,2,3,4,5], retry until valid
  - Same aggregation: collect all runs × questions, then np.mean()
  - Same domain breakdown: lecture / documentary / entertainment
  - Baseline: answers-naiverag  (NaiveRAG is the reference for all methods)
  - Run each question N times (default 5) and average (paper methodology)

Usage (collection 0, EBR-RAG vs NaiveRAG):
  python quant_eval_realtime.py \
    --baseline-root ../all_answers \
    --eval-root     ../full-framework-results \
    --baseline-scenario answers-naiverag \
    --eval-scenarios full_framework \
    --collections 0 \
    --run-time 5 \
    --output results_col0_ebr_vs_naiverag.json
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from openai.lib._parsing._completions import type_to_response_format_param
from tqdm import tqdm

# Load .env from repo root (contains OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Response schema — identical to batch_quant_eval_upload.py
# Forces Score to be int; Explanation to be str.
# ---------------------------------------------------------------------------
class Criterion(BaseModel):
    Score: int
    Explanation: str

class EvalResult(BaseModel):
    Comprehensiveness: Criterion
    Empowerment: Criterion
    Trustworthiness: Criterion
    Depth: Criterion
    Density: Criterion
    Overall_Score: Criterion = Field(alias="Overall Score")

    model_config = {"populate_by_name": True}

RESPONSE_FORMAT = type_to_response_format_param(EvalResult)

# ---------------------------------------------------------------------------
# Prompts — byte-for-byte identical to batch_quant_eval_upload.py
# ---------------------------------------------------------------------------
SYS_PROMPT = """
---Role---
You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.
"""

USER_PROMPT_TEMPLATE = """You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Trustworthiness**: Does the answer provide sufficient detail and align with common knowledge, enhancing its credibility?
- **Depth**: Does the answer provide in-depth analysis or details, rather than just superficial information?
- **Density**: Does the answer contain relevant information without less informative or redundant content?

For the evaluated answer labeled "Evaluation Answer," assign a score from 1 to 5 for each criterion compared to the baseline answer labeled "Baseline Answer." Then, assign an overall score based on these criteria.
The evaluation scores are defined as follows:
- 1: Strongly worse than the baseline answer
- 2: Weakly worse than the baseline answer
- 3: Moderate compared to the baseline answer
- 4: Weakly better than the baseline answer
- 5: Strongly better than the baseline answer


Here is the question:
{query}

Here are the answers:

**Baseline Answer:**
{baseline_answer}

**Evaluation Answer:**
{evaluation_answer}


Evaluate the answer using the criteria listed above and provide detailed explanations for the scores.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }}
    "Overall Score": {{
        "Score": "[1 - 5]",
        "Explanation": "[Provide explanation here]"
    }}
}}"""

METRICS = ["Comprehensiveness", "Empowerment", "Trustworthiness", "Depth", "Density", "Overall Score"]


# ---------------------------------------------------------------------------
# Validation — identical to batch_quant_eval_parse.py check_response_valid()
# ---------------------------------------------------------------------------
def check_response_valid(data: dict) -> None:
    valid_keys = ["Comprehensiveness", "Empowerment", "Trustworthiness", "Depth", "Density", "Overall Score"]
    assert len(data) == 6, f"Expected 6 keys, got {len(data)}: {list(data.keys())}"
    assert set(data.keys()) == set(valid_keys), f"Wrong keys: {list(data.keys())}"
    for key in valid_keys:
        assert data[key]["Score"] in [1, 2, 3, 4, 5], \
            f"{key} Score={data[key]['Score']!r} not in [1,2,3,4,5]"
        assert "Explanation" in data[key], f"Missing Explanation in {key}"


# ---------------------------------------------------------------------------
# API call with structured output + validation + unlimited retry (like parse.py)
# ---------------------------------------------------------------------------
async def call_judge(client: AsyncOpenAI, sem: asyncio.Semaphore, request: dict) -> dict:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user",   "content": request["prompt"]},
    ]
    async with sem:
        attempt = 0
        while True:
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=RESPONSE_FORMAT,
                    # No temperature set → default (1.0), same as original batch
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                check_response_valid(parsed)
                return {
                    "custom_id": request["custom_id"],
                    "result": parsed,
                    "error": None,
                }
            except Exception as e:
                attempt += 1
                if attempt <= 3:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                # After 3 attempts give up and log error (unlike parse.py which retries forever,
                # we cap at 3 to avoid hanging in async context)
                return {
                    "custom_id": request["custom_id"],
                    "result": None,
                    "error": str(e),
                }


# ---------------------------------------------------------------------------
# Aggregation — mirrors batch_quant_eval_calculate.py exactly:
#   for each run file × each question: append score to list
#   final score = np.mean(all appended scores)
# ---------------------------------------------------------------------------
def aggregate_scores(
    all_results: list[dict],
    dataset: dict,
    eval_scenario: str,
) -> dict:
    col_domain: dict[str, str] = {}
    for cid, meta_list in dataset.items():
        col_domain[cid] = meta_list[0].get("type", "unknown")

    domain_list = ["lecture", "documentary", "entertainment"]

    # domain_score[domain][metric] = [score, score, ...]  (all runs × questions)
    domain_score: dict[str, dict[str, list]] = {
        d: {m: [] for m in METRICS} for d in domain_list
    }
    overall_score: dict[str, list] = {m: [] for m in METRICS}

    for item in all_results:
        if item.get("error") or not item.get("result"):
            continue
        col_id = item["col_id"]
        domain = col_domain.get(col_id, "unknown")
        result = item["result"]
        for metric in METRICS:
            score_val = result[metric]["Score"]   # guaranteed int 1-5 by validation
            if domain in domain_score:
                domain_score[domain][metric].append(score_val)
            overall_score[metric].append(score_val)

    return {"domain": domain_score, "all": overall_score}


# ---------------------------------------------------------------------------
# Print — format matches batch_quant_eval_calculate.py output
# ---------------------------------------------------------------------------
def print_score_table(
    agg: dict,
    eval_scenario: str,
    baseline_scenario: str,
    run_time: int,
    collections: list[str],
) -> None:
    domain_list = ["lecture", "documentary", "entertainment"]

    print(f"\n{'====' * 8}")
    print(f"Evaluation : {eval_scenario}")
    print(f"Baseline   : {baseline_scenario}  (runs per question = {run_time})")
    print(f"Collections: {', '.join(collections)}")
    print(f"{'====' * 8}")

    for domain in domain_list:
        scores = agg["domain"].get(domain, {})
        if not any(scores.get(m) for m in METRICS):
            continue
        print(domain)
        for metric in METRICS:
            vals = scores.get(metric, [])
            mean = np.array(vals).mean() if vals else float("nan")
            print(f"  {mean:.2f} {metric}")
        print("----")

    print("All")
    for metric in METRICS:
        vals = agg["all"].get(metric, [])
        mean = np.array(vals).mean() if vals else float("nan")
        print(f"  {mean:.2f} {metric}")
    print("====" * 8)

    n_calls = sum(len(v) for v in agg["all"].values() if v) // len(METRICS)
    print(f"Total scored calls: {n_calls}  (= {n_calls // max(run_time,1)} questions × {run_time} runs)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    _repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Real-time quantitative evaluation matching paper methodology exactly"
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=_repo_root / "reproduce" / "full-framework-results",
        help="Root for eval answers: <root>/<col_slug>/<scenario>/answer_<id>.md",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Root for baseline answers (default: same as --eval-root). "
             "Use reproduce/all_answers for the downloaded VideoRAG paper answers.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_repo_root / "longervideos" / "dataset.json",
    )
    parser.add_argument(
        "--baseline-scenario",
        default="answers-naiverag",
        help="Baseline subdirectory. Paper default: 'answers-naiverag'",
    )
    parser.add_argument(
        "--eval-scenarios",
        nargs="+",
        default=["full_framework"],
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["0", "6", "11"],
    )
    parser.add_argument(
        "--run-time",
        type=int,
        default=5,
        help="Independent judge runs per question (paper uses 5)",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "quant_eval_realtime_results.json",
    )
    args = parser.parse_args()

    baseline_root: Path = args.baseline_root if args.baseline_root is not None else args.eval_root

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset: dict = json.load(f)

    # ------------------------------------------------------------------
    # Build request list (base × run_time copies)
    # ------------------------------------------------------------------
    base_requests: list[dict] = []

    print("=== Collecting evaluation pairs ===")
    print(f"  baseline root     : {baseline_root}")
    print(f"  eval root         : {args.eval_root}")
    print(f"  baseline scenario : {args.baseline_scenario}")
    print(f"  eval scenarios    : {args.eval_scenarios}")
    print(f"  run-time          : {args.run_time}x per question")

    for col_id in args.collections:
        meta_list = dataset.get(col_id)
        if not meta_list:
            print(f"  Warning: collection {col_id} not in dataset")
            continue
        col_meta = meta_list[0]
        col_slug = f"{col_id}-{col_meta['description']}"
        questions = col_meta["questions"]
        baseline_dir = baseline_root / col_slug / args.baseline_scenario

        for eval_scenario in args.eval_scenarios:
            eval_dir = args.eval_root / col_slug / eval_scenario
            found = 0
            for q in questions:
                q_id = q["id"]
                baseline_path = baseline_dir / f"answer_{q_id}.md"
                eval_path = eval_dir / f"answer_{q_id}.md"

                if not baseline_path.exists():
                    print(f"  [MISSING baseline] {baseline_path}")
                    continue
                if not eval_path.exists():
                    print(f"  [MISSING eval]     {eval_path}")
                    continue

                baseline_ans = baseline_path.read_text(encoding="utf-8")
                eval_ans = eval_path.read_text(encoding="utf-8")
                prompt = USER_PROMPT_TEMPLATE.format(
                    query=q["question"],
                    baseline_answer=baseline_ans,
                    evaluation_answer=eval_ans,
                )
                base_requests.append({
                    "base_id": (
                        f"{col_id}-{col_meta['description']}"
                        f"++query{q_id}"
                        f"++base++{args.baseline_scenario}"
                        f"++evaluate++{eval_scenario}"
                    ),
                    "prompt": prompt,
                    "col_id": col_id,
                    "domain": col_meta.get("type", "unknown"),
                    "scenario": eval_scenario,
                    "q_id": q_id,
                })
                found += 1
            print(f"  [{col_id}] {col_slug} / {eval_scenario}: {found}/{len(questions)} pairs")

    if not base_requests:
        print("\nNo pairs found. Check that baseline and eval answer files exist.")
        sys.exit(1)

    # Expand to run_time copies
    all_requests: list[dict] = [
        {**req, "custom_id": f"{req['base_id']}++run{k}"}
        for k in range(args.run_time)
        for req in base_requests
    ]

    total_calls = len(all_requests)
    print(
        f"\n=== {total_calls} API calls "
        f"({len(base_requests)} pairs × {args.run_time} runs, "
        f"concurrency={args.concurrency}) ==="
    )

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    id_to_req = {r["custom_id"]: r for r in all_requests}

    tasks = [call_judge(client, sem, r) for r in all_requests]
    results: list[dict] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=total_calls):
        result = await coro
        req = id_to_req[result["custom_id"]]
        result["col_id"] = req["col_id"]
        result["domain"] = req["domain"]
        result["scenario"] = req["scenario"]
        result["q_id"] = req["q_id"]
        results.append(result)

    errors = [r for r in results if r.get("error")]
    print(f"\nDone: {len(results)} calls | {len(errors)} errors")
    if errors:
        for e in errors[:5]:
            print(f"  ERROR {e['custom_id']}: {e['error']}")

    # Save raw results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw results → {args.output}")

    # Print per eval scenario
    for eval_scenario in args.eval_scenarios:
        scenario_results = [r for r in results if r.get("scenario") == eval_scenario]
        agg = aggregate_scores(scenario_results, dataset, eval_scenario)
        print_score_table(agg, eval_scenario, args.baseline_scenario, args.run_time, args.collections)


if __name__ == "__main__":
    asyncio.run(main())
