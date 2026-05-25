import json
import re
from pathlib import Path

BASE = Path('/workspace/EBR-RAG')
results_path = BASE / 'ablation_realtime_results.json'
dataset_path = BASE / 'longervideos' / 'dataset.json'

with open(results_path, 'r', encoding='utf-8') as f:
    txt = f.read()
    txt_strip = txt.lstrip()
    if txt_strip.startswith('['):
        results = json.loads(txt)
    else:
        # try jsonl
        results = [json.loads(l) for l in txt.splitlines() if l.strip()]

# load dataset
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# scenarios from script
scenarios = [
    'full_framework',
    'no_semantic_nodes',
    'no_tan_nodes',
    'no_semantic_edges',
    'no_temporal_edges',
    'no_cross_modal_edges',
    'no_debate',
    'critique_with_evidence',
    'defender_no_tools',
]

# collect stats
total = len(results)
errors = sum(1 for r in results if r.get('error'))

from collections import Counter, defaultdict
col_counts = Counter()
col_scenario_counts = defaultdict(Counter)
present_ids = set()

for r in results:
    cid = r.get('custom_id')
    if not cid:
        continue
    present_ids.add(cid)
    # collection id
    m = re.match(r"^(\d+)-", cid)
    col = m.group(1) if m else 'unknown'
    col_counts[col] += 1
    # scenario: try to find '++evaluate++X' or last ++ token
    scen = None
    m2 = re.search(r"\+\+evaluate\+\+([a-zA-Z0-9_]+)$", cid)
    if m2:
        scen = m2.group(1)
    else:
        parts = cid.split('++')
        if parts:
            scen = parts[-1]
    if scen:
        col_scenario_counts[col][scen] += 1

# For collections of interest build expected ids
targets = ['6','11','19']
expected = set()
missing = {}
for col in targets:
    meta = dataset.get(str(int(col)), [])
    if not meta:
        missing[col] = {'error': 'collection not in dataset'}
        continue
    desc = meta[0]['description']
    questions = meta[0]['questions']
    qids = [q['id'] for q in questions]
    for q in qids:
        for s in scenarios:
            # generate multiple possible id formats observed
            expected1 = f"{col}-{desc}++q{q}++{s}"
            expected2 = f"{col}-{desc}++query{q}++base++answers-naiverag++evaluate++{s}"
            expected3 = f"{col}-{desc}++q{q}++base++answers-naiverag++evaluate++{s}"
            expected.add(expected1); expected.add(expected2); expected.add(expected3)

for col in targets:
    # collect those expected related to this col (any of the formats)
    col_expected = [eid for eid in expected if eid.startswith(f"{col}-")]
    col_present = [eid for eid in present_ids if eid.startswith(f"{col}-")]
    # consider an expected pair present if any of its formats is in present_ids
    # we built expected to include multiple formats; so missing = expected per q/s minus present intersect expected
    present_expected = set(col_present) & set(col_expected)
    missing_list = sorted(list(set(col_expected) - present_expected))
    missing[col] = {
        'expected_total_variants': len(col_expected),
        'present_matching_expected': len(present_expected),
        'missing_count': len(missing_list),
        'missing_sample': missing_list[:50]
    }

out = {
    'total_entries': total,
    'error_entries': errors,
    'per_collection_counts': dict(col_counts),
    'per_collection_scenario_counts': {k: dict(v) for k,v in col_scenario_counts.items()},
    'missing': missing,
}
print(json.dumps(out, indent=2, ensure_ascii=False))
