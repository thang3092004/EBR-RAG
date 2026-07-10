# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

This repo is **VideoRAG** — a retrieval-augmented generation framework for *extremely long-context video* understanding (paper: Ren et al., arXiv:2502.01549) — **extended with a new `EBR-RAG` pipeline** (Evidence-Based Reasoning: iterative evidence-verification via multi-agent debate with on-demand tool-calling).

Two answering paradigms live side-by-side and share the same ingested index:

- **`videorag`** — the original single-shot dual-channel RAG (graph/entity + dense-text + ImageBind visual retrieval → one LLM answer). This is the baseline used in all official evaluations.
- **`EBR_RAG`** — the headline addition. Same retrieval, then a Generator → (Critique → tool-empowered Defender) × N rounds → Judge debate loop that returns a structured verdict `{answer, rationale, confidence, citations, transcript, evidence, ...}`.

The core RAG machinery is adapted from **nano-graphrag** / **LightRAG**, which explains the `_`-prefixed module layout and the pluggable storage/LLM abstractions.

> ⚠️ Research code, not a packaged library. There is **no `setup.py`/`pyproject.toml`, no test suite, and no CI.** Many eval scripts still require you to hand-edit API keys / batch IDs inside the file before running. Verify by running, not by looking for tests.

## Repository layout

| Path | Purpose |
|------|---------|
| `videorag/videorag.py` | The `VideoRAG` dataclass — top-level orchestrator. `insert_video()` (ingest) and `query()`/`aquery()` (dispatch by `param.mode`). |
| `videorag/base.py` | `QueryParam` (mode + all knobs) and the `Base*Storage` abstract interfaces. |
| `videorag/_llm.py` | `LLMConfig` + named provider configs (OpenAI, Azure, Ollama, DeepSeek+bge). All completion/embedding funcs are async with a JSON KV cache. |
| `videorag/_op.py` | **Core ops (48 KB).** Chunking, entity/relation extraction & graph merge, entity/visual retrieval, and the `videorag_query` / `videorag_query_multiple_choice` answer generators. |
| `videorag/prompt.py` | `PROMPTS` dict: entity extraction, keyword extraction, RAG response templates, etc. |
| `videorag/_splitter.py`, `videorag/_utils.py` | Token splitting; hashing (`compute_mdhash_id`, `compute_args_hash`), async helpers (`limit_async_func_call`, `always_get_an_event_loop`), the global `logger`, JSON locators. |
| `videorag/_storage/` | Pluggable backends: KV=`JsonKVStorage`, vector=`NanoVectorDBStorage` (default) or `HNSWVectorStorage`, graph=`NetworkXStorage` (default) or `Neo4jStorage`. |
| `videorag/_videoutil/` | Ingestion GPU pipeline: `split` (ffmpeg clips), `asr` (faster-whisper), `caption` (MiniCPM-V VLM), `feature` (ImageBind embeddings). |
| `videorag/pipeline/EBR_RAG.py` | `EBR_RAG_answer()` — the EBR-RAG entry point. Stage-1 dual retrieval + budgeting + query-aware re-captioning, then hands off to the debate. |
| `videorag/debate/` | `run_debate()` orchestrator (`debate_manager.py`), `DebateConfig`/`DebateState` (`state.py`), `EvidenceItem` (`evidence_types.py`). |
| `videorag/agents/` | Debate personas: `ROLE_CONFIGS` (`roles.py`: generator/critique/defender/judge) + their system prompts (`agents_prompts.py`, `*_MCQ` and `*_OPEN` variants). |
| `videorag/tools/` | The 3 agent tools (`schemas.py` = `ALL_TOOLS`) and their implementations: `text_tools`, `graph_tools`, `vision_tools`, plus `formatters.py`. |
| `run_ingest.py` | Convenience ingestion of 3 representative LongerVideos collections (reads `.env`). |
| `videorag_longervideos.py` | Original per-collection ingest + `videorag`-mode answer script (`--collection`, `--cuda`). |
| `demo.py`, `examples/` | Interactive demo; DeepSeek-config ingest/query examples. |
| `longervideos/` | The **LongerVideos benchmark**: `dataset.json` (collections → questions), `prepare_data.py`, `download.sh`. |
| `reproduce/` | Evaluation harnesses (win-rate + quantitative 5-point + ablation matrix), OpenAI batch flows, HF upload. See [Evaluation](#evaluation-reproduce). |
| `scripts/`, `scratch/`, `notesbooks/` | Ad-hoc counting scripts, throwaway tests, the demo notebook. |

## Ingestion & query lifecycle

**Ingest** — `VideoRAG.insert_video([paths])` → for each video (steps are resumable; each checks storage and skips if already done):
1. `split_video` → 30 s clips + audio (`_videoutil/split.py`).
2. `speech_to_text` → transcripts via faster-whisper (`asr.py`, **English hard-coded**).
3. In **two parallel processes**: `saving_video_segments` (writes clips) + `segment_caption` (MiniCPM-V captions). If either process dies, both are killed and the error is logged to `error_log_videorag.txt` and re-raised (fail-fast).
4. `video_segment_feature_vdb.upsert` → ImageBind visual embeddings.
5. `ainsert()` → chunk (`chunking_by_video_segments`) → dense-index chunks → `extract_entities()` builds the entity/relation knowledge graph.

**Query** — `VideoRAG.query(q, QueryParam(mode=...))` → `aquery()` dispatches on `param.mode`:
- `"videorag"` → `videorag_query(...)` in `_op.py` (entity+chunk+visual retrieval → one answer).
- `"videorag_multiple_choice"` → `videorag_query_multiple_choice(...)`.
- `"EBR_RAG"` → `EBR_RAG_answer(vrag, q, param)`:
  1. Stage-1 retrieval: `search_text_evidence` + `search_graph_evidence` + `search_visual_segment` (each `top_k=8`, **hard-coded** `base_top_k`), dedup, truncate to 24.
  2. Query-aware re-captioning of retrieved segments (needs `load_caption_model()` first, else silently skipped).
  3. `run_debate()`: Generator draft → for each round: Critique attacks → Defender refines (may call tools to pull ≤1 new evidence item/call) → finally Judge synthesizes the verdict.
  4. MCQ auto-detected from `(A)…(B)…` patterns in the query; changes prompt variants and forces structured JSON parsing.

## Common commands

There is no build step. Everything is `python <script>` from repo root inside the `videorag` conda env.

```bash
# --- Setup (see README for the full GPU dep list; requirements.txt is a subset) ---
conda create --name videorag python=3.11 && conda activate videorag
pip install -r requirements.txt
# Then download checkpoints into repo root: MiniCPM-V-2_6-int4/, faster-distil-whisper-large-v3/, .checkpoints/imagebind_huge.pth
cp .env.example .env    # put OPENAI_API_KEY here (run_ingest.py loads .env; most other scripts read os.environ or set the key inline)

# --- Ingest (GPU required for caption/feature; ASR+VLM+ImageBind checkpoints must exist) ---
python run_ingest.py                                   # 3 representative collections into ./longervideos/videorag-workdir/<name>
python videorag_longervideos.py --collection 4-rag-lecture --cuda 0   # ingest + answer one collection in videorag mode

# --- Query programmatically ---
python demo.py                                         # interactive

# --- Evaluation (all under reproduce/, run in the listed order; several need keys/IDs edited into the file) ---
cd reproduce && wget https://archive.org/download/videorag/all_answers.zip && unzip all_answers.zip

# Ablation matrix (EBR_RAG variants) — collections default to 6, 11, 19
python reproduce/run_ablation_matrix.py --collections 6 11 19
python reproduce/run_baseline_naive.py --collections 6 11 19          # videorag-mode baseline answers

# Win-rate comparison (OpenAI batch API; 4 ordered steps)
cd reproduce/winrate_comparison
python batch_winrate_eval_upload.py && python batch_winrate_eval_download.py \
  && python batch_winrate_eval_parse.py && python batch_winrate_eval_calculate.py

# Quantitative 5-point comparison (same 4-step batch shape)
cd reproduce/quantitative_comparison
python batch_quant_eval_upload.py && python batch_quant_eval_download.py \
  && python batch_quant_eval_parse.py && python batch_quant_eval_calculate.py
```

## Configuration model

- **LLM/provider is config-driven.** Construct `VideoRAG(llm=<some LLMConfig>, working_dir=...)`. Named configs in `_llm.py`: `openai_config` (default), `openai_4o_mini_config`, `azure_openai_config`, `ollama_config`, `deepseek_bge_config`. All default to `gpt-4o-mini` + `text-embedding-3-small` (1536-dim). To switch providers, pass a different `LLMConfig`; to switch models, edit the config's `*_model_name` fields.
- **`QueryParam`** (`base.py`) carries the mode and all query knobs. EBR-RAG-specific: `max_rounds` (default 2), `ebr_top_k`, `return_detailed`, and the three `debate_*` ablation flags. `wo_reference=True` by default (no clip citations appended).
- **Working dir is the whole state.** `insert_video` writes `kv_store_*.json`, `vdb_*.json`, `graph_chunk_entity_relation.graphml` into `working_dir`; querying re-instantiates `VideoRAG` pointing at the same dir. Ingest is **resumable** — re-running skips fully-processed videos/segments.

## Gotchas & conventions (read before editing)

- **`CLAUDE.md` and any `*.md` are git-ignored.** `.gitignore` has `*.md` with only `!README.md` excepted. To commit this file you must add `!CLAUDE.md` to `.gitignore`. Note LongerVideos answers are written as `.md` and are intentionally ignored.
- **Secrets & env vars:** `OPENAI_API_KEY` is required. `OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`, `DEEPSEEK_API_KEY`, `SILICONFLOW_API_KEY` are read when using those providers. History note: hardcoded keys were purged in commit `606b3da`; `videorag_longervideos.py` still contains an empty `os.environ["OPENAI_API_KEY"] = ""` line you must fill or delete. **Never commit real keys.**
- **`multiprocessing.set_start_method('spawn')` is mandatory** before any ingest (CUDA + spawn). All ingest entry scripts do this under `if __name__ == '__main__':`.
- **GPU + local checkpoints required for ingest and for EBR-RAG re-captioning:** `./MiniCPM-V-2_6-int4` (falls back to the HF hub id), `./faster-distil-whisper-large-v3`, `./.checkpoints/imagebind_huge.pth`. Querying in `videorag`/`EBR_RAG` mode needs `videorag.load_caption_model(debug=False)` first; `debug=True` skips loading the VLM (EBR-RAG then skips the re-caption refinement step silently).
- **ASR is English-only** — `WhisperModel` language is fixed in `_videoutil/asr.py`.
- **EBR-RAG budgets are hard-coded, not fully driven by `QueryParam`.** `pipeline/EBR_RAG.py` sets `base_top_k = 8` regardless of `param.ebr_top_k` (the docstring claiming it reads `ebr_top_k` is stale), and caps evidence at `24 + max_rounds*3`. Defender tool calls fetch only 1 item each, ≤3 calls/round, ≤25 total, capped at `universal_cap` (~52).
- **Async LLM cache correctness:** the KV cache guards against `if_cache_return["return"] is None` (fixed in commit `f8227df`) so a cached failure isn't replayed. The OpenAI async client is **re-created per call on purpose** (`get_openai_async_client_instance`) — do not cache it globally; sharing an `AsyncOpenAI` across event loops makes httpx hang.
- **Return type of EBR_RAG mode is context-dependent:** `aquery` returns a **plain answer string** unless `param.return_detailed=True`, which returns the full dict. Eval scripts that expect a dict set this flag; `run_ablation_matrix.py` hard-fails if it gets a string or an empty answer.
- **Ablation flag `use_tm_graph`** on `VideoRAG` swaps `extract_entities`→`extract_entities_tm` and adds a `_tm` namespace suffix to storage files (parallel index; won't collide with the standard one).
- **Naming conventions:** `_`-prefixed modules (`_op`, `_llm`, `_utils`, `_storage`, `_videoutil`) are internal/adapted-from-nano-graphrag; everything is `async` down to the storage layer (drive it with `always_get_an_event_loop()` or `asyncio.run`). Storage classes are injected as `*_cls` types on the `VideoRAG` dataclass — to add a backend, implement the matching `Base*Storage` interface in `base.py` and swap the class.
- **`dataset.json` schema:** top-level dict keyed by string collection id (`"0".."21"`); each value is a 1-element list whose object has `video_url`, `description`, `type` (lecture/documentary/entertainment), and `questions: [{id, question}]`. The `--collection` flags use the `"<id>-<slug>"` form (e.g. `6-daubechies-wavelet-lecture`); code splits on `-` to recover the numeric id.
- **Some `reproduce/` scripts require manual edits** (batch IDs, output file IDs, parsed filenames) between steps — the README win-rate/quantitative sections spell out the exact order.

## Where to make common changes

- **Change the answer LLM / provider** → pick or add an `LLMConfig` in `_llm.py`, pass via `VideoRAG(llm=...)`.
- **Tune EBR-RAG behavior** → retrieval budgets/caps in `pipeline/EBR_RAG.py`; debate rounds/roles in `debate/debate_manager.py` + `agents/roles.py`; persona wording in `agents/agents_prompts.py`; add/modify agent tools in `tools/` + register the schema in `tools/schemas.py` (`ALL_TOOLS`) and dispatch in `EBR_RAG.py::dispatch_tool`.
- **Change retrieval / graph / chunking** → `_op.py`.
- **Change prompts** for the classic pipeline → `prompt.py`.
- **Add a new query mode** → branch in `videorag.py::aquery` + a new function in `_op.py` (or a new `pipeline/` module) + a `Literal` entry in `QueryParam.mode`.
