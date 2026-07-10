# EBR-RAG — Interactive Demo

A self-contained Streamlit app that runs the **full EBR-RAG pipeline on one short video**:

1. **① Ingest** — download a YouTube video and watch a **live per-stage trace** of processing
   (split → ASR → caption → visual features → chunk → entity graph).
2. **② Hỏi đáp** — ask a question in `EBR_RAG` mode and watch the **multi-agent debate**
   (Generator → Critique → Defender+tools → Judge), with answer, evidence, and metadata.
3. **③ Đồ thị tri thức** — explore the extracted knowledge graph interactively (pyvis).

Default video: TED-Ed *"What staying up all night does to your brain"* (~5 min).

## Requirements

- **GPU** (RTX 3090 / 24 GB is plenty), CUDA, `ffmpeg`.
- The VideoRAG stack + 3 model checkpoints (MiniCPM-V, faster-whisper, ImageBind).
- An **OpenAI API key** (ingest uses `gpt-4o-mini` + `text-embedding-3-small`; the debate uses `gpt-4o-mini`).

## Setup (one time)

Run from the **repo root**, inside your GPU python env (e.g. `/venv/main` or a `videorag` conda env):

```bash
cd /path/to/EBR-RAG
source /venv/main/bin/activate          # or: conda activate videorag
bash demo_app/setup.sh                   # installs deps + downloads checkpoints (per README) + yt-dlp
```

Then paste your key into the repo-root `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

Always launch **from the repo root** (so `./MiniCPM-V-2_6-int4` and `./.checkpoints/imagebind_huge.pth` resolve):

```bash
streamlit run demo_app/app.py
```

Open the local URL, go to **① Ingest**, keep the default URL (or paste another short YouTube video),
click **Tải & Ingest**, then move to **② Hỏi đáp** and **③ Đồ thị tri thức**.

## How it works (for maintainers)

| File | Role |
|------|------|
| `app.py` | Streamlit UI (3 tabs); spawn-safe `main()` guard; live-trace streaming loops. |
| `pipeline.py` | `ingest_with_trace()` / `query_with_trace()` — background-thread wrappers over `VideoRAG`. |
| `trace.py` | `QueueLogHandler`, **forces the `nano-graphrag`/`debate_manager` loggers to INFO** (they default to WARNING → trace would otherwise be empty), ingest stage instrumentation + disk poller. |
| `graph_view.py` | Reads `graph_chunk_entity_relation.graphml` and renders an interactive pyvis network. |
| `ytdl.py` | yt-dlp download helper. |
| `config.py` | Paths, OpenAI `LLMConfig`, demo video URL. |
| `setup.sh` | Installs deps + downloads the 3 checkpoints per the main README. |

### Notes & limitations
- **Run from repo root.** Otherwise the code silently falls back to re-downloading MiniCPM-V from the HF hub.
- **Ingest Step 3 (caption+save)** runs in `multiprocessing` subprocesses, so its progress can't reach the log handler; the trace shows it as an indeterminate stage with a "clips saved" counter from a disk poller.
- **Query-aware re-captioning is skipped** by default (caption model left unloaded) for speed. Set `load_real_caption=True` in `pipeline.query_with_trace` for full fidelity (loads MiniCPM-V into VRAM at query time).
- `result["citations"]` is currently always `[]` in the pipeline (the `_validate_citations` helper is defined but never called); the demo surfaces the per-evidence `validated` flag instead.
- Ingest artifacts go to `demo_app/workdir/<video>/` and downloaded videos to `demo_app/videos/` (both git-ignored).
