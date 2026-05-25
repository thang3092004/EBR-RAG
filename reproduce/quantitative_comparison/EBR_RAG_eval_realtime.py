import os
import json
import asyncio
import tqdm
from openai import AsyncOpenAI

# ===========================================================================
# EBR-RAG Real-time Evaluation Script
# Uses EXACT logic and prompts from the official benchmark.
# ===========================================================================

SYS_PROMPT = """You are an expert evaluating an answer against a baseline answer based on these criteria: **Comprehensiveness**, **Empowerment**, **Trustworthiness**, **Depth** and **Density**."""

USER_PROMPT_TEMPLATE = """
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

async def call_judge(client, sem, request_data):
    """Wait for semaphore then call OpenAI API."""
    async with sem:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": request_data["prompt"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return {
                "custom_id": request_data["custom_id"],
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": response.choices[0].message.content}}
                        ]
                    }
                },
                "error": None
            }
        except Exception as e:
            return {
                "custom_id": request_data["custom_id"],
                "error": str(e)
            }

async def main():
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        default_headers={"User-Agent": "Mozilla/5.0"} # Trình duyệt giả lập để tránh bị chặn IP đơn thuần
    )
    sem = asyncio.Semaphore(10) # 10 concurrent requests to respect RPM/TPM
    
    dataset_path = "../../longervideos/dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        questions_dataset = json.load(f)

    # Configuration for comparison
    processed_ids = ['0', '13', '17']
    evaluation_answer_dirs = ['answers-EBR-RAG-baseline']
    baseline_dir_name = 'answers-naiverag'
    
    evaluation_pairs = []

    print(f"--- Collecting answers for Real-time Evaluation (Sets: {processed_ids}) ---")

    for _id in processed_ids:
        meta_list = questions_dataset.get(_id, [])
        if not meta_list: continue
        
        collection_desc = meta_list[0]['description']
        questions = meta_list[0]['questions']
        base_dir = f"../all_answers/{_id}-{collection_desc}"
        
        for _eval_dir in evaluation_answer_dirs:
            for q in questions:
                q_id = q['id']
                query_text = q['question']
                
                path_target = os.path.join(base_dir, _eval_dir, f"answer_{q_id}.md")
                path_baseline = os.path.join(base_dir, baseline_dir_name, f"answer_{q_id}.md")
                
                if os.path.exists(path_target) and os.path.exists(path_baseline):
                    with open(path_target, 'r', encoding='utf-8') as f1, \
                         open(path_baseline, 'r', encoding='utf-8') as f2:
                        
                        target_ans = f1.read()
                        base_ans = f2.read()
                        
                        prompt = USER_PROMPT_TEMPLATE.format(
                            query=query_text,
                            baseline_answer=base_ans,
                            evaluation_answer=target_ans
                        )
                        
                        evaluation_pairs.append({
                            "custom_id": f"{_id}-{collection_desc}++query{q_id}++base++naiverag++evaluate++{_eval_dir}",
                            "prompt": prompt
                        })

    if not evaluation_pairs:
        print("No matches found to evaluate. Check paths.")
        return

    print(f"--- Found {len(evaluation_pairs)} pairs to judge. Starting API calls... ---")
    
    tasks = [call_judge(client, sem, pair) for pair in evaluation_pairs]
    
    results = []
    # Use tqdm to show progress bar
    for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        results.append(await coro)

    # Save results to a file compatible with existing parsers
    output_dir = "batch_requests/EBR_RAG_comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "realtime_evaluation_results.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n--- DONE! ---")
    print(f"Results saved to: {output_file}")
    print("You can now run your parsing script on this file.")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable.")
    else:
        asyncio.run(main())
