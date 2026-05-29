"""
EBR-RAG Demo — Single-file Gradio app
Cho phép hỏi đáp trên 3 collection đã ingest và so sánh trực quan các kịch bản ablation.

Usage:
    python demo.py
    (Requires OPENAI_API_KEY in environment or .env file)
"""

import os
import time
import queue
import logging
import threading
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

# ─── BYPASS CAPTION MONKEYPATCH ───────────────────────────────────────────────
def bypass_retrieved_segment_caption(
    caption_model, caption_tokenizer, refine_knowledge, retrieved_segments,
    video_path_db, video_segments, num_sampled_frames
):
    print("\n=== [DEMO PATCH] Bypassing Re-Captioning (Refinement Disabled for maximum speed) ===")
    caption_result = {}
    for this_segment in retrieved_segments:
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        
        # Get the pre-saved static caption and transcript directly from the DB
        existing_content = video_segments._data[video_name][index]["content"]
        caption_result[this_segment] = existing_content
        
    return caption_result

_cached_imagebind_embedder = None

async def cached_nanovectordb_video_segment_query(self, query: str, top_k=None):
    global _cached_imagebind_embedder
    from imagebind.models import imagebind_model
    from videorag._videoutil import encode_string_query
    import torch
    
    if _cached_imagebind_embedder is None:
        print("\n=== [DEMO PATCH] Loading ImageBind model into GPU RAM ===")
        device = "cuda"
        _cached_imagebind_embedder = imagebind_model.imagebind_huge(pretrained=True).to(device)
        _cached_imagebind_embedder.eval()
        print("✅ [DEMO PATCH] ImageBind loaded on GPU successfully.")
        
    embedding = encode_string_query(query, _cached_imagebind_embedder)
    embedding = embedding[0]
    
    results = self._client.query(
        query=embedding,
        top_k=top_k if top_k is not None else self.top_k,
        better_than_threshold=-1,
    )
    results = [
        {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
    ]
    return results

# Apply monkeypatching before imports/run
try:
    import videorag._videoutil.caption as caption_mod
    import videorag._videoutil as videoutil_mod
    import videorag._op as op_mod
    import videorag._storage.vdb_nanovectordb as vdb_mod

    caption_mod.retrieved_segment_caption = bypass_retrieved_segment_caption
    videoutil_mod.retrieved_segment_caption = bypass_retrieved_segment_caption
    op_mod.retrieved_segment_caption = bypass_retrieved_segment_caption
    
    vdb_mod.NanoVectorDBVideoSegmentStorage.query = cached_nanovectordb_video_segment_query
    print("✅ [DEMO PATCH] Successfully patched retrieved_segment_caption to bypass refinement.")
    print("✅ [DEMO PATCH] Successfully patched ImageBind video query with in-memory caching.")
except Exception as patch_err:
    print(f"❌ [DEMO PATCH] Failed to patch demo: {patch_err}")

# ─── LIVE STATUS LOG HANDLER ──────────────────────────────────────────────────
class GradioLogHandler(logging.Handler):
    """Captures log records and pushes them into a queue for streaming to UI."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put(("log", self.format(record)))
        except Exception:
            pass


# ─── CONFIG ───────────────────────────────────────────────────────────────────
COLLECTIONS = {
    "6 — Daubechies Wavelet Lecture": {
        "id": "6",
        "workdir": "./longervideos/videorag-workdir/6-daubechies-wavelet-lecture",
        "youtube_urls": [
            "https://www.youtube.com/watch?v=RkGK0MloK0E",
            "https://www.youtube.com/watch?v=1s9zZ6ERAko",
            "https://www.youtube.com/watch?v=RNqXpdwd9AA",
            "https://www.youtube.com/watch?v=2zaJZ_F7Xrk",
        ],
        "sample_questions": [
            "What are the inherent limitations of time-frequency localization?",
            "How do different window functions impact time-frequency analysis?",
            "What is the significance of the vanishing moments property in wavelets?",
        ],
    },
    "11 — Primetime Emmy Awards": {
        "id": "11",
        "workdir": "./longervideos/videorag-workdir/11-primetime-emmy-awards",
        "youtube_urls": [
            "https://www.youtube.com/watch?v=dYX809pLH00",
            "https://www.youtube.com/watch?v=5gWAZV8KoEw",
            "https://www.youtube.com/watch?v=0udZzgn1UcM",
        ],
        "sample_questions": [
            "How do local news stations engage viewers and earn their trust?",
            "Why is the truth and factual reporting important, especially in local news?",
            "What impact has the Emmy Awards had on the television industry?",
        ],
    },
    "19 — Jeff Bezos": {
        "id": "19",
        "workdir": "./longervideos/videorag-workdir/19-jeff-bezos",
        "youtube_urls": [
            "https://www.youtube.com/watch?v=s71nJQqzYRQ",
            "https://www.youtube.com/watch?v=DcWqzZ3I2cY",
            "https://www.youtube.com/watch?v=zN1PyNwjHpc",
        ],
        "sample_questions": [
            "What shaped Jeff Bezos's problem-solving approach?",
            "What is Bezos's 'Day One' philosophy?",
            "What are Bezos's key principles for effective leadership?",
        ],
    },
}

ABLATION_SCENARIOS = {
    "VideoRAG (Baseline)": {
        "mode": "videorag",
        "use_tm_graph": False,
        "debate_critique_see_evidence": False,
        "debate_defender_disable_tools": False,
        "max_rounds": 2,
        "description": "Pipeline gốc VideoRAG: Graph cơ bản + truy xuất đa phương thức, không có debate.",
    },
    "EBR-RAG (Full Framework)": {
        "mode": "EBR_RAG",
        "use_tm_graph": True,   # Full framework dùng TM Graph
        "debate_critique_see_evidence": False,
        "debate_defender_disable_tools": False,
        "max_rounds": 3,
        "description": "EBR-RAG đầy đủ: TM Graph + tranh luận đa tác nhân + tool-calling.",
    },
    "EBR-RAG (Baseline with Debate)": {
        "mode": "EBR_RAG",
        "use_tm_graph": False,  # Graph cơ bản, không TM Graph
        "debate_critique_see_evidence": False,
        "debate_defender_disable_tools": False,
        "max_rounds": 3,
        "description": "Ablation: EBR-RAG dùng graph cơ bản (không TM Graph) + tranh luận đầy đủ.",
    },
    "EBR-RAG (No Debate)": {
        "mode": "EBR_RAG",
        "use_tm_graph": True,
        "debate_critique_see_evidence": False,
        "debate_defender_disable_tools": False,
        "max_rounds": 0,        # max_rounds=0 → tắt debate
        "description": "Ablation: Tắt hoàn toàn debate (max_rounds=0), chỉ dùng TM Graph.",
    },
    "EBR-RAG (Critique Sees Evidence)": {
        "mode": "EBR_RAG",
        "use_tm_graph": True,
        "debate_critique_see_evidence": True,
        "debate_defender_disable_tools": False,
        "max_rounds": 3,
        "description": "Ablation: Critic được xem trực tiếp bằng chứng của Defender (giảm tính độc lập).",
    },
    "EBR-RAG (Defender No Tools)": {
        "mode": "EBR_RAG",
        "use_tm_graph": True,
        "debate_critique_see_evidence": False,
        "debate_defender_disable_tools": True,
        "max_rounds": 3,
        "description": "Ablation: Defender không được dùng tool-calling để truy xuất bằng chứng.",
    },
}

# ─── VIDEORAG LOADER ───────────────────────────────────────────────────────────
_cache = {}  # (workdir, use_tm_graph) -> VideoRAG instance

def get_vrag(workdir: str, use_tm_graph: bool):
    key = (workdir, use_tm_graph)
    if key not in _cache:
        from videorag import VideoRAG
        from videorag._llm import openai_config
        vr = VideoRAG(
            llm=openai_config,
            working_dir=workdir,
            use_tm_graph=use_tm_graph,
        )
        # Prevent loading MiniCPM-V-2_6-int4 into VRAM completely
        vr.load_caption_model(debug=True)
        # Give dummy mock values so the EBR_RAG.py check (if not c_model or not c_tok) passes
        vr.caption_model = "MOCKED_API_MODEL"
        vr.caption_tokenizer = "MOCKED_API_TOKENIZER"
        _cache[key] = vr
    return _cache[key]


def _format_status(lines: list, elapsed: float) -> str:
    """Format live log lines as a markdown status block."""
    recent = lines[-20:] if len(lines) > 20 else lines
    log_text = "\n".join(recent) if recent else "Đang khởi tạo pipeline..."
    return (
        f"**🔄 Đang xử lý... ({elapsed:.0f}s)**\n\n"
        f"```\n{log_text}\n```"
    )


def run_query(collection_name, scenario_name, question, max_rounds):
    """Generator: yields (status_md, meta_md) pairs while running, then yields final answer."""
    if not question.strip():
        yield "⚠️ Vui lòng nhập câu hỏi.", ""
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        yield "⚠️ Chưa có OPENAI_API_KEY. Vui lòng thêm vào file .env hoặc biến môi trường.", ""
        return

    col = COLLECTIONS[collection_name]
    scen = ABLATION_SCENARIOS[scenario_name]
    workdir = col["workdir"]
    use_tm = scen["use_tm_graph"]

    # ── Set up live log streaming ──
    log_queue: queue.Queue = queue.Queue()
    handler = GradioLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(message)s"))

    vr_logger = logging.getLogger("nano-graphrag")
    debate_logger = logging.getLogger("videorag.debate.debate_manager")
    for lg in (vr_logger, debate_logger):
        lg.addHandler(handler)

    result_holder: dict = {}

    def _run():
        try:
            vrag = get_vrag(workdir, use_tm)
            from videorag.base import QueryParam
            # Nếu kịch bản là No Debate (0) thì giữ 0, ngược lại ưu tiên giá trị từ slider
            effective_max_rounds = 0 if scen.get("max_rounds") == 0 else int(max_rounds)
            param = QueryParam(
                mode=scen["mode"],
                wo_reference=True,
                return_detailed=(scen["mode"] == "EBR_RAG"),
                max_rounds=effective_max_rounds,
                debate_critique_see_evidence=scen.get("debate_critique_see_evidence", False),
                debate_defender_disable_tools=scen.get("debate_defender_disable_tools", False),
            )
            result_holder["result"] = vrag.query(query=question, param=param)
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            log_queue.put(("done", None))
            for lg in (vr_logger, debate_logger):
                lg.removeHandler(handler)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    t0 = time.time()

    status_lines = []
    while True:
        try:
            kind, msg = log_queue.get(timeout=0.4)
        except queue.Empty:
            if not thread.is_alive():
                break
            yield _format_status(status_lines, time.time() - t0), ""
            continue

        if kind == "done":
            break
        status_lines.append(msg)
        yield _format_status(status_lines, time.time() - t0), ""

    thread.join(timeout=10)
    elapsed = time.time() - t0

    if "error" in result_holder:
        yield f"❌ Lỗi khi chạy query: {result_holder['error']}", ""
        return

    result = result_holder.get("result")
    if result is None:
        yield "❌ Không nhận được kết quả từ pipeline.", ""
        return

    # ── Format final output ──
    if isinstance(result, dict):
        answer = result.get("answer", "")
        rationale = result.get("rationale", "")
        confidence = result.get("confidence", 0.0)
        rounds = result.get("rounds_run", 0)
        tool_calls = result.get("tool_calls_made", 0)
        citations = result.get("citations", [])

        meta = (
            f"**⏱ Thời gian:** {elapsed:.1f}s | "
            f"**🔄 Rounds:** {rounds} | "
            f"**🔧 Tool calls:** {tool_calls} | "
            f"**📊 Confidence:** {confidence:.0%}"
        )
        if citations:
            cite_str = "\n".join([f"- {c}" for c in citations[:5]])
            meta += f"\n\n**📌 Citations:**\n{cite_str}"

        full_answer = answer
        if rationale:
            full_answer += f"\n\n---\n**💡 Rationale:** {rationale}"
        yield full_answer, meta
    else:
        yield str(result), f"⏱ Thời gian: {elapsed:.1f}s"


def update_questions(collection_name):
    questions = COLLECTIONS[collection_name]["sample_questions"]
    links = COLLECTIONS[collection_name]["youtube_urls"]
    link_md = "\n".join([f"- [{url}]({url})" for url in links])
    return (
        gr.update(choices=questions, value=questions[0]),
        f"**🎬 Video sources:**\n{link_md}",
    )


def fill_question(sample_q):
    return sample_q


# ─── GRADIO UI ─────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.scenario-desc { font-size: 0.85em; color: #666; margin-top: 4px; padding: 6px 10px; background: #f5f5f5; border-radius: 6px; }
"""

with gr.Blocks(title="EBR-RAG Demo") as demo:
    gr.Markdown("# 🎬 EBR-RAG Interactive Demo", elem_id="title")
    gr.Markdown("So sánh câu trả lời của các kịch bản ablation trên 3 collection video đã được ingest sẵn.", elem_id="subtitle")

    with gr.Row():
        # ── LEFT PANEL ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Chọn Collection")
            collection_dd = gr.Dropdown(
                choices=list(COLLECTIONS.keys()),
                value=list(COLLECTIONS.keys())[0],
                label="Collection",
                interactive=True,
            )
            video_links_md = gr.Markdown()

            gr.Markdown("### 💬 Câu hỏi gợi ý")
            sample_dd = gr.Dropdown(
                choices=COLLECTIONS[list(COLLECTIONS.keys())[0]]["sample_questions"],
                value=COLLECTIONS[list(COLLECTIONS.keys())[0]]["sample_questions"][0],
                label="Chọn câu hỏi mẫu",
                interactive=True,
            )
            use_sample_btn = gr.Button("📋 Dùng câu hỏi này", variant="secondary", size="sm")

            gr.Markdown("### ⚗️ Kịch bản Ablation")
            scenario_dd = gr.Dropdown(
                choices=list(ABLATION_SCENARIOS.keys()),
                value=list(ABLATION_SCENARIOS.keys())[1],  # EBR-RAG Full by default
                label="Kịch bản",
                interactive=True,
            )
            scenario_desc_md = gr.Markdown(
                value=f"<div class='scenario-desc'>{ABLATION_SCENARIOS[list(ABLATION_SCENARIOS.keys())[1]]['description']}</div>"
            )

            max_rounds_sl = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="🔄 Override max_rounds (mặc định dùng giá trị của kịch bản)",
                interactive=True,
            )

        # ── RIGHT PANEL ─────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### ❓ Câu hỏi của bạn")
            question_box = gr.Textbox(
                label="Nhập câu hỏi",
                placeholder="Ví dụ: What is Bezos's Day One philosophy?",
                lines=3,
            )

            run_btn = gr.Button("🚀 Chạy Query", variant="primary")

            gr.Markdown("### 📝 Câu trả lời")
            answer_box = gr.Markdown(label="Answer")

            gr.Markdown("### 📊 Metadata")
            meta_box = gr.Markdown(label="Metadata")

    # ── SIDE-BY-SIDE COMPARISON ──────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 🔬 So sánh Song Song 2 Kịch Bản")
    with gr.Row():
        with gr.Column():
            scen_a = gr.Dropdown(
                choices=list(ABLATION_SCENARIOS.keys()),
                value=list(ABLATION_SCENARIOS.keys())[0],
                label="Kịch bản A",
            )
        with gr.Column():
            scen_b = gr.Dropdown(
                choices=list(ABLATION_SCENARIOS.keys()),
                value=list(ABLATION_SCENARIOS.keys())[1],
                label="Kịch bản B",
            )

    compare_btn = gr.Button("⚖️ So sánh", variant="primary")
    with gr.Row():
        with gr.Column():
            out_a = gr.Markdown(label="Kết quả A")
            meta_a = gr.Markdown()
        with gr.Column():
            out_b = gr.Markdown(label="Kết quả B")
            meta_b = gr.Markdown()

    # ─── EVENT HANDLERS ──────────────────────────────────────────────────────
    collection_dd.change(
        fn=update_questions,
        inputs=[collection_dd],
        outputs=[sample_dd, video_links_md],
    )

    use_sample_btn.click(fn=fill_question, inputs=[sample_dd], outputs=[question_box])

    def update_scenario_desc(s):
        return f"<div class='scenario-desc'>{ABLATION_SCENARIOS[s]['description']}</div>"
    scenario_dd.change(fn=update_scenario_desc, inputs=[scenario_dd], outputs=[scenario_desc_md])

    # Single run — streaming generator
    run_btn.click(
        fn=run_query,
        inputs=[collection_dd, scenario_dd, question_box, max_rounds_sl],
        outputs=[answer_box, meta_box],
    )

    # Comparison run (drains the generator to get final result for each scenario)
    def run_comparison(collection_name, scen_a_name, scen_b_name, question, max_rounds):
        res_a = meta_a_val = res_b = meta_b_val = ""
        for val_a, m_a in run_query(collection_name, scen_a_name, question, max_rounds):
            res_a, meta_a_val = val_a, m_a
        for val_b, m_b in run_query(collection_name, scen_b_name, question, max_rounds):
            res_b, meta_b_val = val_b, m_b
        return res_a, meta_a_val, res_b, meta_b_val

    compare_btn.click(
        fn=run_comparison,
        inputs=[collection_dd, scen_a, scen_b, question_box, max_rounds_sl],
        outputs=[out_a, meta_a, out_b, meta_b],
    )

    demo.load(
        fn=lambda c: update_questions(c)[1],
        inputs=[collection_dd],
        outputs=[video_links_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=CSS)
