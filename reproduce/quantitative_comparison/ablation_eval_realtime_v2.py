"""
Realtime evaluation for ablation scenarios (collections 6, 11, 19)
Calls gpt-4o-mini directly without using Batch API
"""
import os
import json
import asyncio
from openai import AsyncOpenAI
import argparse
from tqdm.asyncio import tqdm

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
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Score": {{
        "Score": [1-5],
        "Explanation": "[Provide explanation here]"
    }}
}}
"""

async def call_judge(client, sem, request_data):
    """Call OpenAI API with semaphore for rate limiting."""
    async with sem:
        for attempt in range(3):
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
                content = response.choices[0].message.content
                try:
                    parsed = json.loads(content)
                    return {
                        "custom_id": request_data["custom_id"],
                        "result": parsed,
                        "error": None
                    }
                except Exception as e:
                    if attempt == 2:
                        return {
                            "custom_id": request_data["custom_id"],
                            "error": f"JSON Parse: {str(e)}",
                            "raw": content
                        }
            except Exception as e:
                if attempt == 2:
                    return {
                        "custom_id": request_data["custom_id"],
                        "error": str(e)
                    }
                await asyncio.sleep(2 ** attempt)
        return {
            "custom_id": request_data["custom_id"],
            "error": "Max retries exceeded"
        }

async def main():
    parser = argparse.ArgumentParser(description="Realtime ablation evaluation")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent API calls")
    parser.add_argument("--output", type=str, default="ablation_realtime_v2_results.jsonl", help="Output file")
    args = parser.parse_args()

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    # Load dataset
    dataset_path = "../../longervideos/dataset.json"
    with open(dataset_path, 'r') as f:
        questions_dataset = json.load(f)

    # Collections to evaluate
    collection_ids = ['19']
    
    # Ablation scenarios
    ablation_scenarios = [
        'full_framework',
        'no_semantic_nodes',
        'no_tan_nodes',
        'no_semantic_edges',
        'no_temporal_edges',
        'no_cross_modal_edges',
        'no_debate',
        'critique_with_evidence',
        'defender_no_tools',
        'baseline_with_debate',
    ]
    
    baseline_dir = 'answers-naiverag'

    evaluation_pairs = []

    print("=== Collecting evaluation pairs ===")

    for col_id in collection_ids:
        meta_list = questions_dataset.get(col_id, [])
        if not meta_list:
            print(f"Warning: Collection {col_id} not found in dataset")
            continue
        
        collection_desc = meta_list[0]['description']
        questions = meta_list[0]['questions']
        base_dir = f"../all_answers/{col_id}-{collection_desc}"
        
        print(f"\n[{col_id}] {collection_desc}: {len(questions)} questions")
        
        for q in questions:
            q_id = q['id']
            query_text = q['question']
            
            # Load baseline answer
            baseline_path = os.path.join(base_dir, baseline_dir, f"answer_{q_id}.md")
            if not os.path.exists(baseline_path):
                continue
            
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_ans = f.read()
            
            # Evaluate each scenario
            for scenario in ablation_scenarios:
                scenario_path = os.path.join(base_dir, scenario, f"answer_{q_id}.md")
                
                if not os.path.exists(scenario_path):
                    continue
                
                with open(scenario_path, 'r', encoding='utf-8') as f:
                    scenario_ans = f.read()
                
                prompt = USER_PROMPT_TEMPLATE.format(
                    query=query_text,
                    baseline_answer=baseline_ans,
                    evaluation_answer=scenario_ans
                )
                
                evaluation_pairs.append({
                    "custom_id": f"{col_id}-{collection_desc}++q{q_id}++{scenario}",
                    "prompt": prompt,
                    "col_id": col_id,
                    "scenario": scenario
                })

    print(f"\n=== Found {len(evaluation_pairs)} pairs to evaluate ===")
    
    if not evaluation_pairs:
        print("No evaluation pairs found!")
        return

    # Call API for all pairs
    print(f"\n=== Starting API calls (concurrency={args.concurrency}) ===")
    
    tasks = [call_judge(client, sem, pair) for pair in evaluation_pairs]
    results = []
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print stats
    errors = sum(1 for r in results if r.get('error'))
    print(f"\n=== Results saved to {args.output} ===")
    print(f"Total: {len(results)} | Successful: {len(results) - errors} | Errors: {errors}")

if __name__ == "__main__":
    asyncio.run(main())
