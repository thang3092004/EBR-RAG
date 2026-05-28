import os
# Optional hardcoded key (user requested). Leave empty to use environment variable.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY (hardcoded or environment variable) before running this script.")
import re
import time
import json
import argparse
import jsonlines
import tiktoken
from pydantic import BaseModel, Field
from typing import Literal

from tqdm import tqdm
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.lib._parsing._completions import type_to_response_format_param

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

sys_prompt = """
You are an expert tasked with evaluating two answers to the same question based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.
"""

prompt = """
You will evaluate two answers to the same question based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**.


For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these criteria.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner on the above criteria]"
    }}
}}
"""

class Criterion(BaseModel):
    Winner: Literal["Answer 1", "Answer 2"]
    Explanation: str

class Result(BaseModel):
    Comprehensiveness: Criterion
    Empowerment: Criterion
    Trustworthiness: Criterion
    Depth: Criterion
    Density: Criterion
    Overall_Winner: Criterion = Field(alias="Overall Winner")

result_response_format = type_to_response_format_param(Result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and submit winrate evaluation batches")
    parser.add_argument("--run-time", type=int, default=5, help="Number of repeated runs (paper uses 5)")
    parser.add_argument("--max-enqueued-tokens", type=int, default=1800000, help="Approx token budget per submitted batch file")
    parser.add_argument("--submit-run", type=int, default=None, help="Submit only this run index (e.g., 0)")
    parser.add_argument("--submit-part", type=int, default=None, help="Submit only this part index within a run (e.g., 0)")
    parser.add_argument("--only-generate", action="store_true", help="Only generate request json files, do not submit")
    args = parser.parse_args()

    with open('../../longervideos/dataset.json', 'r') as f:
        questions = json.load(f)
    our_answer_dir = 'answers-videorag'
    
    # overall comparsion - rag
    base_dir = 'overall_comparison_rag'
    com_answer_dir = [
        'answers-naiverag', 
        'answers-graphrag-local', 
        'answers-graphrag-global',
        'answers-lightrag-hybrid',
    ]
    
    requests = []
    request_token_counts = []
    total_token_count = 0
    for _id in questions:
        video_list_name = questions[_id][0]['description']
        video_querys = questions[_id][0]['questions']
        data_path = f"../all_answers/{_id}-{video_list_name}"
        for _com_answer_dir in com_answer_dir:
            our_work_dir = os.path.join(data_path, our_answer_dir)
            com_work_dir = os.path.join(data_path, _com_answer_dir)
            for i in range(len(questions[_id][0]['questions'])):
                # query
                query_id = questions[_id][0]['questions'][i]["id"]
                query = questions[_id][0]['questions'][i]["question"]
                # our answer
                with open(os.path.join(our_work_dir, f'answer_{query_id}.md'), 'r', encoding='utf-8') as f:
                    our_answer = f.read()
                # com answer
                with open(os.path.join(com_work_dir, f'answer_{query_id}.md'), 'r', encoding='utf-8') as f:
                    com_answer = f.read()
                ori_prompt = prompt.format(query=query, answer1=our_answer, answer2=com_answer)
                rev_prompt = prompt.format(query=query, answer1=com_answer, answer2=our_answer)
                
                ori_request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++{our_answer_dir}++{_com_answer_dir}++ori",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": ori_prompt},
                        ],
                        "response_format": result_response_format
                    },
                }
                rev_request_data = {
                    "custom_id": f"{_id}-{video_list_name}++query{query_id}++{_com_answer_dir}++{our_answer_dir}++rev",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": rev_prompt},
                        ],
                        "response_format": result_response_format
                    },
                }
                ori_token_count = len(encoding.encode(ori_prompt))
                rev_token_count = len(encoding.encode(rev_prompt))

                requests.append(ori_request_data)
                request_token_counts.append(ori_token_count)
                requests.append(rev_request_data)
                request_token_counts.append(rev_token_count)

                total_token_count += ori_token_count
                total_token_count += rev_token_count

    run_time = args.run_time
    max_enqueued_tokens_per_batch = args.max_enqueued_tokens
    os.makedirs(f'batch_requests/{base_dir}', exist_ok=True)

    request_chunks = []
    token_chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for request, token_count in zip(requests, request_token_counts):
        if current_chunk and current_chunk_tokens + token_count > max_enqueued_tokens_per_batch:
            request_chunks.append(current_chunk)
            token_chunks.append(current_chunk_tokens)
            current_chunk = []
            current_chunk_tokens = 0
        current_chunk.append(request)
        current_chunk_tokens += token_count
    if current_chunk:
        request_chunks.append(current_chunk)
        token_chunks.append(current_chunk_tokens)

    print(f"Total requests: {len(requests)}")
    print(f"Split into {len(request_chunks)} batch file(s) with max {max_enqueued_tokens_per_batch} tokens each.")
    print(f"Estimated price: {total_token_count / 1000000 * 0.075 * run_time}$")

    submitted_count = 0
    for k in range(run_time):
        for part_idx, (request_chunk, token_chunk) in enumerate(zip(request_chunks, token_chunks)):
            if args.submit_run is not None and k != args.submit_run:
                continue
            if args.submit_part is not None and part_idx != args.submit_part:
                continue

            request_json_file_path = f'batch_requests/{base_dir}/{int(time.time())}_run{k}_part{part_idx}.json'
            with jsonlines.open(request_json_file_path, mode="w") as writer:
                for request in request_chunk:
                    writer.write(request)
            print(f"Batch API requests written to {request_json_file_path} (tokens~{token_chunk})")

            if args.only_generate:
                continue

            client = OpenAI()
            batch_input_file = client.files.create(
                file=open(request_json_file_path, "rb"), purpose="batch"
            )
            batch_input_file_id = batch_input_file.id

            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"runtime{k}-part{part_idx}: {request_json_file_path}"},
            )
            print(f"RunTime {k} Part {part_idx}: Batch {batch.id} has been created.")
            submitted_count += 1

    if args.only_generate:
        print("Done: request files generated only (no submission).")
    else:
        print(f"Done: submitted {submitted_count} batch job(s).")