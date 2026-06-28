"""
RAGAS evaluation using the official ragas library v0.1.x (Es et al., 2023).

Metrics:
  1. Faithfulness — fraction of claims in answer supported by retrieved contexts
  2. Answer Relevancy — how well the answer addresses the question
  3. Answer Correctness — factual F1 + semantic similarity vs ground truth

Usage:
    pip install ragas datasets

    # Collection 0 only
    python reproduce/ragas_eval.py --collections 0

    # All collections
    python reproduce/ragas_eval.py --collections 0 6 11

Requires: OPENAI_API_KEY environment variable set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATASET_PATH = ROOT / "longervideos" / "dataset.json"

COLLECTION_FOLDERS = {
    "0": "0-fights-in-animal-kingdom",
    "6": "6-daubechies-wavelet-lecture",
    "11": "11-primetime-emmy-awards",
}


def build_ragas_dataset(
    dataset: dict,
    results_root: Path,
    groundtruth_root: Path,
    collections: list[str],
    scenario: str,
):
    """Build a HuggingFace Dataset with columns: question, answer, contexts, ground_truth."""
    from datasets import Dataset

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for col_id in collections:
        meta_list = dataset.get(col_id)
        if not meta_list:
            print(f"  Warning: collection {col_id} not in dataset")
            continue
        col_meta = meta_list[0]
        col_slug = f"{col_id}-{col_meta['description']}"
        col_questions = col_meta["questions"]

        result_dir = results_root / col_slug / scenario
        gt_dir = groundtruth_root / col_slug / "answers-groundtruth"

        found = 0
        for q in col_questions:
            q_id = q["id"]
            result_path = result_dir / f"result_{q_id}.json"
            gt_path = gt_dir / f"answer_{q_id}.md"

            if not result_path.exists():
                print(f"  [MISSING result] {result_path}")
                continue
            if not gt_path.exists():
                print(f"  [MISSING groundtruth] {gt_path}")
                continue

            result_data = json.loads(result_path.read_text(encoding="utf-8"))
            answer = str(result_data.get("answer", "")).strip()
            evidence = result_data.get("details", {}).get("evidence", [])
            ctx = [
                str(e.get("snippet", ""))
                for e in evidence
                if str(e.get("snippet", "")).strip()
            ]
            gt = gt_path.read_text(encoding="utf-8").strip()

            if not answer:
                print(f"  [EMPTY answer] {result_path}")
                continue
            if not ctx:
                print(f"  [NO contexts] {result_path}")
                continue

            questions.append(q["question"])
            answers.append(answer)
            contexts_list.append(ctx)
            ground_truths.append(gt)
            found += 1

        print(f"  [{col_id}] {col_slug}: {found}/{len(col_questions)} samples")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation using the official ragas library"
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=ROOT / "reproduce" / "full-framework-results",
    )
    parser.add_argument(
        "--groundtruth-root", type=Path,
        default=ROOT / "reproduce" / "all_answers",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--scenario", default="full_framework")
    parser.add_argument("--collections", nargs="+", default=["0", "6", "11"])
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reproduce" / "ragas_results.json",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
    except ImportError:
        print("ERROR: ragas not installed. Run: pip install ragas datasets")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=== Building evaluation dataset ===")
    print(f"  results root   : {args.results_root}")
    print(f"  groundtruth    : {args.groundtruth_root}")
    print(f"  scenario       : {args.scenario}")

    eval_dataset = build_ragas_dataset(
        dataset,
        args.results_root,
        args.groundtruth_root,
        args.collections,
        args.scenario,
    )

    if len(eval_dataset) == 0:
        print("\nNo samples found. Check result/groundtruth paths.")
        sys.exit(1)

    metrics = [faithfulness, answer_relevancy, answer_correctness]
    metric_names = ["faithfulness", "answer_relevancy", "answer_correctness"]
    print(f"\n=== {len(eval_dataset)} samples, metrics: {', '.join(metric_names)} ===")
    print("  Running evaluation (this calls OpenAI API)...")

    result = evaluate(eval_dataset, metrics=metrics)

    print(f"\n{'='*60}")
    print(f"RAGAS Results — {args.scenario}")
    print(f"{'='*60}")
    for name in metric_names:
        score = result.get(name)
        if score is not None:
            print(f"  {name:<30} {score:.4f}")
        else:
            print(f"  {name:<30} N/A")

    df = result.to_pandas()
    output_data = {
        "scenario": args.scenario,
        "collections": args.collections,
        "num_samples": len(eval_dataset),
        "aggregate": {
            name: float(result.get(name, 0)) for name in metric_names
        },
        "per_question": json.loads(df.to_json(orient="records", force_ascii=False)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
