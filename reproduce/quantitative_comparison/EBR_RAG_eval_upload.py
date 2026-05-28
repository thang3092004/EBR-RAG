"""
EBR-RAG Evaluation — adds 'answers-EBR-RAG' to the official VideoRAG benchmark evaluation.

This script is a drop-in replacement for batch_quant_eval_upload.py that includes
the EBR-RAG answers in the comparison. Run this AFTER:
    1. EBR_RAG_longervideos.py --mode all     (generates answers)
    2. This script                               (submits batch evaluation to OpenAI)
    3. batch_quant_eval_download.py              (downloads results)
    4. batch_quant_eval_parse.py                 (parses results)
    5. batch_quant_eval_calculate.py with EBR_RAG added to evaluate_model list

Usage (from the reproduce/quantitative_comparison/ folder):
    cd reproduce/quantitative_comparison
    # Only generate request files (no submission):
    python EBR_RAG_eval_upload.py --only-generate

    # Generate and submit to OpenAI Batch API:
    python EBR_RAG_eval_upload.py
"""
import os
import re
import time
import json
import argparse
import jsonlines
import tiktoken
import itertools
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.lib._parsing._completions import type_to_response_format_param

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY before running.")

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

sys_prompt = """
---Role---
You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.
"""

prompt = """
You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.

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
}}
"""

class Criterion(BaseModel):
    Score: int
    Explanation: str

class Result(BaseModel):
    Comprehensiveness: Criterion
    Empowerment: Criterion
    Trustworthiness: Criterion
    Depth: Criterion
    Density: Criterion
    Overall_Score: Criterion = Field(alias="Overall Score")

result_response_format = type_to_response_format_param(Result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit EBR-RAG evaluation batches")
    parser.add_argument("--run-time", type=int, default=5, help="Number of repeated runs (paper uses 5)")
    parser.add_argument("--max-enqueued-tokens", type=int, default=1800000, help="Approx token budget per batch file")
    parser.add_argument("--only-generate", action="store_true", help="Only generate request json files, do not submit")
    args = parser.parse_args()

    with open('../../longervideos/dataset.json', 'r') as f:
        questions = json.load(f)

    baseline_answer_dir = 'answers-naiverag'
    base_dir = 'EBR_RAG_comparison'
    evaluation_answer_dirs = [
        'answers-videorag',             # Original VideoRAG
        'answers-EBR-RAG-baseline',     # EBR-RAG
    ]

    requests = []
    request_token_counts = []
    total_token_count = 0
    skipped = 0

    for _id in questions:
        video_list_name = questions[_id][0]['description']
        video_querys = questions[_id][0]['questions']
        data_path = f"../all_answers/{_id}-{video_list_name}"

        for _evaluation_answer_dir in evaluation_answer_dirs:
            baseline_work_dir = os.path.join(data_path, baseline_answer_dir)
            evaluation_work_dir = os.path.join(data_path, _evaluation_answer_dir)

            for i in range(len(video_querys)):
                query_id = video_querys[i]["id"]
                query = video_querys[i]["question"]

                baseline_path = os.path.join(baseline_work_dir, f'answer_{query_id}.md')
                evaluation_path = os.path.join(evaluation_work_dir, f'answer_{query_id}.md')

                if not os.path.exists(baseline_path):
                    print(f"[SKIP] Missing baseline: {baseline_path}")
                    skipped += 1
                    continue
                if not os.path.exists(evaluation_path):
                    print(f"[SKIP] Missing evaluation: {evaluation_path}")
                    skipped += 1
                    continue

                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_answer = f.read()
                with open(evaluation_path, 'r', encoding='utf-8') as f:
                    evaluation_answer = f.read()

                request_prompt = prompt.format(
                    query=query,
                    baseline_answer=baseline_answer,
                    evaluation_answer=evaluation_answer,
                )
                request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++base++{baseline_answer_dir}++evaluate++{_evaluation_answer_dir}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": request_prompt},
                        ],
                        "response_format": result_response_format,
                    },
                }
                request_token_count = len(encoding.encode(request_prompt))
                requests.append(request_data)
                request_token_counts.append(request_token_count)
                total_token_count += request_token_count

    print(f"\nTotal requests: {len(requests)}  |  Skipped: {skipped}")
    print(f"Estimated price for {args.run_time} runs: ${total_token_count / 1_000_000 * 0.075 * args.run_time:.2f}")

    # Split into chunks
    max_enqueued_tokens_per_batch = args.max_enqueued_tokens
    os.makedirs(f'batch_requests/{base_dir}', exist_ok=True)

    request_chunks, token_chunks = [], []
    current_chunk, current_chunk_tokens = [], 0
    for request, token_count in zip(requests, request_token_counts):
        if current_chunk and current_chunk_tokens + token_count > max_enqueued_tokens_per_batch:
            request_chunks.append(current_chunk)
            token_chunks.append(current_chunk_tokens)
            current_chunk, current_chunk_tokens = [], 0
        current_chunk.append(request)
        current_chunk_tokens += token_count
    if current_chunk:
        request_chunks.append(current_chunk)
        token_chunks.append(current_chunk_tokens)

    print(f"Split into {len(request_chunks)} batch file(s)")

    submitted_count = 0
    for k in range(args.run_time):
        for part_idx, (request_chunk, token_chunk) in enumerate(zip(request_chunks, token_chunks)):
            request_json_file_path = f'batch_requests/{base_dir}/{int(time.time())}_run{k}_part{part_idx}.json'
            with jsonlines.open(request_json_file_path, mode="w") as writer:
                for request in request_chunk:
                    writer.write(request)
            print(f"Batch file written: {request_json_file_path} (tokens~{token_chunk})")

            if args.only_generate:
                continue

            client = OpenAI()
            batch_input_file = client.files.create(
                file=open(request_json_file_path, "rb"), purpose="batch"
            )
            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"EBR-RAG run{k}-part{part_idx}"},
            )
            print(f"Run {k} Part {part_idx}: Batch {batch.id} created.")
            submitted_count += 1

    if args.only_generate:
        print("\nDone: request files generated only (no submission).")
    else:
        print(f"\nDone: {submitted_count} batch job(s) submitted.")
