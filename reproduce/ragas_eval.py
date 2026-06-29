"""
RAGAS evaluation — SEQUENTIAL, per-question, resumable.

Unlike ragas.evaluate() over a whole batch (the original approach: if it crashes
mid-run you lose ALL progress and the spent API cost, and resume is awkward), this
scores ONE question at a time and atomically checkpoints after EVERY question.
Safe for a single expensive run: kill it any time, re-run the same command, and it
skips everything already scored.

Metrics:
  Context-free (run on EVERY sample, comparable across scenarios — primary ablation signal):
    - answer_relevancy   : answer vs question
    - answer_correctness : answer vs ground_truth (factual F1 + semantic similarity)
  Context-dependent (run only on samples that have a retrieved evidence pool):
    - faithfulness       : answer claims supported by ITS OWN contexts
                           (per-scenario grounding measure, NOT a cross-scenario ranking)
    - context_recall     : ground-truth claims present in retrieved contexts
                           (retrieval / graph completeness — key for the graph ablations)
    - context_precision  : relevant contexts ranked high
                           (SLOWEST: ~1 LLM call per question*context; --no-precision to skip)

Contexts are read from the UNCAPPED evidence text (`snippet_full`) the debate agents
actually saw, so faithfulness/context metrics judge against the true context.

Install in an ISOLATED venv (ragas 0.1.x pins openai<2 and would otherwise downgrade
the pipeline venv). This script imports ONLY ragas/datasets — never the videorag package —
so the isolated venv needs nothing else:
    python -m venv .venv-ragas
    .venv-ragas/bin/pip install "ragas>=0.1,<0.2" datasets python-dotenv openai
    .venv-ragas/bin/python reproduce/ragas_eval.py --collections 0 6 11

Requires OPENAI_API_KEY (in .env or environment).
"""
from __future__ import annotations

import argparse
import json
import math
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

# Hardcoded (not imported from videorag.ablation) so the script runs in the
# isolated .venv-ragas which does not have the videorag package installed.
DEFAULT_SCENARIOS = [
    "video_rag_baseline", "full_framework", "no_adaptive_segmentation",
    "no_entity_memory", "no_gleaning", "no_crossmodal_alignment",
    "no_debate_refinement", "critique_sees_evidence", "defender_no_tools",
]

FREE_METRIC_NAMES = ["answer_relevancy", "answer_correctness"]


def _slug(dataset: dict, col_id: str):
    meta = dataset.get(col_id)
    return f"{col_id}-{meta[0]['description']}" if meta else None


def load_records(dataset, results_root, gt_root, collections, scenario):
    """Return [(qid, record)] for every question. record['status'] is 'ok' when an
    answer + ground_truth exist; otherwise it flags what is missing (kept so the
    checkpoint records why a question was not scored)."""
    records = []
    for col_id in collections:
        slug = _slug(dataset, col_id)
        if slug is None:
            print(f"  [warn] collection {col_id} not in dataset")
            continue
        result_dir = results_root / slug / scenario
        gt_dir = gt_root / slug / "answers-groundtruth"
        for q in dataset[col_id][0]["questions"]:
            qid = f"{col_id}:{q['id']}"
            rp = result_dir / f"result_{q['id']}.json"
            gp = gt_dir / f"answer_{q['id']}.md"
            if not rp.exists():
                records.append((qid, {"status": "missing_result", "path": str(rp)}))
                continue
            if not gp.exists():
                records.append((qid, {"status": "missing_groundtruth", "path": str(gp)}))
                continue
            data = json.loads(rp.read_text(encoding="utf-8"))
            answer = str(data.get("answer", "")).strip()
            evidence = data.get("details", {}).get("evidence", [])
            ctx = [
                (str(e.get("snippet_full") or e.get("snippet") or "")).strip()
                for e in evidence
            ]
            ctx = [c for c in ctx if c]
            gt = gp.read_text(encoding="utf-8").strip()
            if not answer:
                records.append((qid, {"status": "empty_answer"}))
                continue
            records.append((qid, {
                "status": "ok",
                "question": q["question"],
                "answer": answer,
                "contexts": ctx if ctx else [""],
                "ground_truth": gt,
                "has_context": bool(ctx),
            }))
    return records


def evaluate_one(rec, metric_objs, Dataset, evaluate):
    """Run RAGAS on a single-row dataset and return {metric_name: float|None}."""
    ds = Dataset.from_dict({
        "question": [rec["question"]],
        "answer": [rec["answer"]],
        "contexts": [rec["contexts"]],
        "ground_truth": [rec["ground_truth"]],
    })
    result = evaluate(ds, metrics=metric_objs)
    df = result.to_pandas()
    row = df.iloc[0]
    reserved = {
        "question", "answer", "contexts", "ground_truth",
        "user_input", "response", "retrieved_contexts", "reference",
    }
    out = {}
    for col in df.columns:
        if col in reserved:
            continue
        try:
            f = float(row[col])
            out[col] = None if math.isnan(f) else f
        except (TypeError, ValueError):
            continue
    return out


def _nanmean(vals):
    xs = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return sum(xs) / len(xs) if xs else None


def _save(path: Path, state: dict):
    """Atomic write: write to .tmp then rename, so a kill mid-write never corrupts
    the checkpoint (critical for a one-shot resumable run)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Sequential, resumable RAGAS evaluation")
    ap.add_argument("--collections", nargs="+", default=["0", "6", "11"])
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--results-root", type=Path,
                    default=ROOT / "reproduce" / "minimal_ablation_answers",
                    help="Where query writes result_<id>.json (matches query --output-root)")
    ap.add_argument("--groundtruth-root", type=Path,
                    default=ROOT / "reproduce" / "all_answers")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduce" / "ragas_results")
    ap.add_argument("--no-precision", action="store_true",
                    help="Skip context_precision (slowest: ~1 LLM call per question*context).")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, answer_correctness,
            context_recall, context_precision,
        )
    except ImportError:
        print("ERROR: ragas not installed. In an ISOLATED venv run:")
        print("  pip install 'ragas>=0.1,<0.2' datasets python-dotenv openai")
        sys.exit(1)

    free_objs = [answer_relevancy, answer_correctness]
    ctx_objs = [faithfulness, context_recall] + ([] if args.no_precision else [context_precision])
    ctx_names = ["faithfulness", "context_recall"] + ([] if args.no_precision else ["context_precision"])

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        out_path = args.output_dir / f"ragas_{scenario}.json"
        if out_path.exists():
            state = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            state = {
                "scenario": scenario,
                "collections": args.collections,
                "context_precision_included": not args.no_precision,
                "per_question": {},
                "aggregate": {},
            }
        done = state["per_question"]

        records = load_records(dataset, args.results_root, args.groundtruth_root,
                               args.collections, scenario)
        scorable = sum(1 for _, r in records if r.get("status") == "ok")
        already = sum(1 for qid, _ in records if done.get(qid, {}).get("_complete"))
        print(f"\n=== {scenario}: {len(records)} questions | {scorable} scorable | "
              f"{already} already done (resume) ===")

        for qid, rec in records:
            if done.get(qid, {}).get("_complete"):
                continue  # resume: skip completed (errors have _complete=False -> retried)
            if rec.get("status") != "ok":
                done[qid] = {"status": rec["status"], "_complete": True}
                _save(out_path, state)
                continue

            scores = {"status": "ok", "has_context": rec["has_context"], "_complete": False}
            try:
                scores.update(evaluate_one(rec, free_objs, Dataset, evaluate))
                if rec["has_context"]:
                    scores.update(evaluate_one(rec, ctx_objs, Dataset, evaluate))
                scores["_complete"] = True
            except Exception as exc:  # transient API/metric failure -> retry on resume
                scores["error"] = str(exc)[:300]
                scores["_complete"] = False

            done[qid] = scores
            _save(out_path, state)  # checkpoint EVERY question
            shown = " ".join(
                f"{k}={scores[k]:.3f}" for k in (FREE_METRIC_NAMES + ctx_names)
                if isinstance(scores.get(k), (int, float))
            )
            flag = "" if scores["_complete"] else f" [ERROR: {scores.get('error','')[:60]}]"
            print(f"  {qid}: {shown}{flag}")

        # aggregate over successfully-scored questions (NaN/None ignored).
        # Discover metric keys from the actual per-question scores so we capture
        # whatever names RAGAS used, not just the expected ones.
        meta_keys = {"status", "has_context", "_complete", "error"}
        metric_keys = []
        for v in done.values():
            if v.get("status") != "ok":
                continue
            for k in v:
                if k not in meta_keys and k not in metric_keys:
                    metric_keys.append(k)
        # keep the canonical ordering first, then any extras
        ordered = [m for m in (FREE_METRIC_NAMES + ctx_names) if m in metric_keys]
        ordered += [m for m in metric_keys if m not in ordered]
        agg = {}
        for name in ordered:
            agg[name] = _nanmean([
                v.get(name) for v in done.values() if v.get("status") == "ok"
            ])
        state["aggregate"] = agg
        state["num_scored"] = sum(1 for v in done.values()
                                  if v.get("status") == "ok" and v.get("_complete"))
        _save(out_path, state)
        print("  AGGREGATE: " + " ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=NA" for k, v in agg.items()
        ) + f"  ({state['num_scored']} scored)  -> {out_path.name}")

    print(f"\nDone. Per-scenario results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
