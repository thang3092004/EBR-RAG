"""
Thin wrappers over VideoRAG for the demo. Both entry points are meant to run in a
background thread; they communicate with the Streamlit UI purely through the trace
``log_queue`` (see trace.py) and a shared ``result_holder`` dict.
"""
import os
import glob
import threading
import multiprocessing

from .trace import setup_trace_logging, instrument_ingest, poll_segment_saving

_imagebind_patched = False
_cached_imagebind_embedder = None


def ensure_spawn():
    """CUDA + the caption/save subprocesses require the 'spawn' start method."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def enable_imagebind_cache():
    """Keep one ImageBind model resident in GPU RAM across visual retrievals.

    Without this, every ``search_visual_segment`` reloads the ~4.5GB model, which makes
    EBR-RAG queries painfully slow. Patches NanoVectorDBVideoSegmentStorage.query.
    """
    global _imagebind_patched
    if _imagebind_patched:
        return

    async def cached_query(self, query: str, top_k=None):
        global _cached_imagebind_embedder
        from imagebind.models import imagebind_model
        from videorag._videoutil import encode_string_query
        if _cached_imagebind_embedder is None:
            _cached_imagebind_embedder = imagebind_model.imagebind_huge(pretrained=True).to("cuda")
            _cached_imagebind_embedder.eval()
        embedding = encode_string_query(query, _cached_imagebind_embedder)[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k if top_k is not None else self.top_k,
            better_than_threshold=-1,
        )
        return [{**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results]

    import videorag._storage.vdb_nanovectordb as vdb_mod
    vdb_mod.NanoVectorDBVideoSegmentStorage.query = cached_query
    _imagebind_patched = True


def _ingest_summary(workdir: str) -> dict:
    """Best-effort counts for the completion card (never raises)."""
    from . import config
    from . import graph_view
    summary = {"segments": None, "chunks": None, "entities": None, "edges": None}
    try:
        import json
        seg_file = os.path.join(workdir, "kv_store_video_segments.json")
        if os.path.exists(seg_file):
            data = json.load(open(seg_file))
            summary["segments"] = sum(len(v) for v in data.values())
        chunk_file = os.path.join(workdir, "kv_store_text_chunks.json")
        if os.path.exists(chunk_file):
            summary["chunks"] = len(json.load(open(chunk_file)))
    except Exception:
        pass
    try:
        g = graph_view.load_graph(config.graphml_path(workdir))
        if g is not None:
            summary["entities"] = g.number_of_nodes()
            summary["edges"] = g.number_of_edges()
    except Exception:
        pass
    return summary


def ingest_with_trace(video_path: str, workdir: str, log_queue, result_holder: dict,
                      video_output_format: str = "mp4"):
    """Background-thread target: ingest one video, streaming stage/log events."""
    ensure_spawn()
    teardown = setup_trace_logging(log_queue)
    video_name = os.path.basename(video_path).split(".")[0]
    stop_event = threading.Event()
    poller = threading.Thread(
        target=poll_segment_saving,
        args=(log_queue, workdir, video_name, video_output_format, stop_event),
        daemon=True,
    )
    try:
        from videorag import VideoRAG
        from .config import get_llm_config
        vr = VideoRAG(llm=get_llm_config(), working_dir=workdir)
        poller.start()
        with instrument_ingest(log_queue):
            vr.insert_video(video_path_list=[video_path])
        result_holder["result"] = _ingest_summary(workdir)
    except Exception as e:  # noqa: BLE001
        result_holder["error"] = f"{type(e).__name__}: {e}"
        log_queue.put(("error", result_holder["error"]))
    finally:
        stop_event.set()
        teardown()
        log_queue.put(("done", None))


def query_with_trace(workdir: str, question: str, max_rounds: int, log_queue,
                     result_holder: dict, load_real_caption: bool = False):
    """Background-thread target: run an EBR-RAG query, streaming debate logs.

    Returns (via result_holder["result"]) the full detailed dict:
    {answer, rationale, confidence, citations, transcript, evidence,
     rounds_run, tool_calls_made}.
    """
    ensure_spawn()
    teardown = setup_trace_logging(log_queue)
    try:
        from videorag import VideoRAG, QueryParam
        from .config import get_llm_config
        vr = VideoRAG(llm=get_llm_config(), working_dir=workdir)
        # By default we leave caption_model unset so EBR-RAG cleanly SKIPS query-aware
        # re-captioning (the `if not c_model` guard). Set load_real_caption=True for full
        # fidelity (loads MiniCPM-V into VRAM and re-captions retrieved segments).
        if load_real_caption:
            vr.load_caption_model(debug=False)
        try:
            enable_imagebind_cache()
        except Exception:
            pass  # visual retrieval still works, just slower
        param = QueryParam(
            mode="EBR_RAG",
            wo_reference=True,
            return_detailed=True,
            max_rounds=int(max_rounds),
        )
        result_holder["result"] = vr.query(query=question, param=param)
    except Exception as e:  # noqa: BLE001
        result_holder["error"] = f"{type(e).__name__}: {e}"
        log_queue.put(("error", result_holder["error"]))
    finally:
        teardown()
        log_queue.put(("done", None))
