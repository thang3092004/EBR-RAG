import os
import json
import asyncio
import tqdm
import numpy as np
from openai import AsyncOpenAI
import argparse
from dotenv import load_dotenv

from pathlib import Path

# ===========================================================================
# EBR-RAG Full Ablation & Framework Real-time Evaluation Script
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
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }},
    "Trustworthiness": {{
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }},
    "Depth": {{
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }},
    "Density": {{
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Score": {{
        "Score": [1 - 5],
        "Explanation": "[Provide explanation here]"
    }}
}}
"""

async def call_judge(client, sem, request_data):
    """Wait for semaphore then call OpenAI API."""
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
                # Attempt to parse json to ensure it's valid, cleaning if necessary
                try:
                    parsed = json.loads(content)
                    return {
                        "custom_id": request_data["custom_id"],
                        "result": parsed,
                        "error": None
                    }
                except Exception as e:
                    if attempt == 2:
                        return {"custom_id": request_data["custom_id"], "error": f"JSON Parse error: {str(e)}", "raw": content}
                    continue
            except Exception as e:
                if attempt == 2:
                    return {
                        "custom_id": request_data["custom_id"],
                        "error": str(e)
                    }
                await asyncio.sleep(2 * (attempt + 1))
    return {"custom_id": request_data["custom_id"], "error": "Max retries reached"}

async def main():
    # Load .env from project root
    repo_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", type=str, default="ablation_realtime_results.json")
    parser.add_argument("--collections", nargs='+', type=str, default=['0','13','17'],
                        help="Collection IDs to evaluate (e.g. 6 11 19)")
    parser.add_argument("--evaluation-dirs", nargs='+', type=str,
                        default=[
                            'full_framework',
                            'no_semantic_nodes',
                            'no_tan_nodes',
                            'no_semantic_edges',
                            'no_temporal_edges',
                            'no_cross_modal_edges',
                            'no_debate',
                            'critique_with_evidence',
                            'defender_no_tools',
                        ],
                        help="List of evaluation answer directories in all_answers/<id>-<name>/")
    parser.add_argument("--baseline-dir", type=str, default='answers-naiverag',
                        help="Baseline answer directory name")
    args = parser.parse_args()

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    
    # Dataset path relative to project root
    dataset_path = repo_root / "longervideos/dataset.json"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        questions_dataset = json.load(f)

    # Collection IDs to process (can be passed via --collections)
    processed_ids = args.collections
    evaluation_answer_dirs = args.evaluation_dirs
    baseline_answer_dir = args.baseline_dir
    
    evaluation_pairs = []

    print(f"--- Collecting answers for Real-time Evaluation ---")

    for _id in processed_ids:
        meta_list = questions_dataset.get(_id, [])
        if not meta_list: continue
        
        collection_desc = meta_list[0]['description']
        questions = meta_list[0]['questions']
        # Path relative to project root
        base_dir = repo_root / "reproduce/all_answers" / f"{_id}-{collection_desc}"
        
        for q in questions:
            q_id = q['id']
            query_text = q['question']
            
            baseline_path = base_dir / baseline_answer_dir / f"answer_{q_id}.md"
            if not baseline_path.exists():
                continue
            
            with open(baseline_path, 'r', encoding='utf-8') as fb:
                base_ans = fb.read()

            for eval_dir in evaluation_answer_dirs:
                target_path = base_dir / eval_dir / f"answer_{q_id}.md"
                
                if target_path.exists():
                    with open(target_path, 'r', encoding='utf-8') as f1:
                        target_ans = f1.read()
                        
                        prompt = USER_PROMPT_TEMPLATE.format(
                            query=query_text,
                            baseline_answer=base_ans,
                            evaluation_answer=target_ans
                        )
                        
                        evaluation_pairs.append({
                            "custom_id": f"{_id}-{collection_desc}++query{q_id}++base++{baseline_answer_dir}++evaluate++{eval_dir}",
                            "prompt": prompt,
                            "eval_dir": eval_dir
                        })

    if not evaluation_pairs:
        print("No matches found to evaluate. Check paths.")
        return

    print(f"--- Found {len(evaluation_pairs)} pairs to judge. Starting API calls... ---")
    
    tasks = [call_judge(client, sem, pair) for pair in evaluation_pairs]
    
    results = []
    for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        results.append(await coro)

    # Save raw results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n--- Analysis ---")
    
    metrics = ['Comprehensiveness', 'Empowerment', 'Trustworthiness', 'Depth', 'Density', 'Overall Score']
    summary = {}
    
    for eval_dir in evaluation_answer_dirs:
        summary[eval_dir] = {m: [] for m in metrics}

    success_count = 0
    fail_count = 0
    for r in results:
        if r.get("error"):
            fail_count += 1
            continue
        
        success_count += 1
        cid = r["custom_id"]
        # Extract eval_dir from custom_id
        # Format: 0-Collection++query1++base++answers-naiverag++evaluate++answers-no_graph
        target_dir = cid.split("++evaluate++")[-1]
        
        data = r["result"]
        for m in metrics:
            val = data.get(m, {}).get("Score")
            if val is not None:
                try:
                    summary[target_dir][m].append(float(val))
                except:
                    pass

    print(f"Success: {success_count} | Failed: {fail_count}")

    # Generate the Comparison Table
    headers = ["Method", "Comp", "Emp", "Trust", "Depth", "Dens", "Overall"]
    rows = []
    
    # Map dir names to friendly names
    name_map = {
        "full_framework": "Full Framework",
        "baseline_with_debate": "Baseline with Debate",
        "no_semantic_nodes": "w/o Semantic Nodes",
        "no_tan_nodes": "w/o TAN Nodes",
        "no_semantic_edges": "w/o Semantic Edges",
        "no_temporal_edges": "w/o Temporal Edges",
        "no_cross_modal_edges": "w/o Cross-Modal Edges",
        "no_debate": "w/o Debate",
        "critique_with_evidence": "Critique with Evidence",
        "defender_no_tools": "Defender No Tools",
    }

    for eval_dir in evaluation_answer_dirs:
        row = [name_map.get(eval_dir, eval_dir)]
        for m in metrics:
            scores = summary[eval_dir][m]
            if scores:
                avg = np.mean(scores)
                row.append(f"{avg:.2f}")
            else:
                row.append("N/A")
        rows.append(row)

    # Print markdown table
    print("\n### Comparison Table\n")
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_str)
    print(sep_str)
    for row in rows:
        print("| " + " | ".join(row) + " |")

    # Also save summary to JSON
    summary_final = {}
    for eval_dir, m_data in summary.items():
        summary_final[eval_dir] = {m: (float(np.mean(vals)) if vals else 0) for m, vals in m_data.items()}
    
    with open("ablation_summary.json", "w") as f:
        json.dump(summary_final, f, indent=4)

    print(f"\nSummary saved to ablation_summary.json")

if __name__ == "__main__":
    asyncio.run(main())
