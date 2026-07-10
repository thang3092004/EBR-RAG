"""
EBR-RAG — Interactive Demo (Streamlit)

Three tabs:
  ①  Ingest a YouTube video with a live per-stage processing trace
  ②  Ask a question and watch the multi-agent debate (EBR-RAG) unfold
  ③  Explore the extracted knowledge graph interactively

Run FROM THE REPO ROOT so model-checkpoint paths (./MiniCPM-V-2_6-int4,
./.checkpoints/imagebind_huge.pth) resolve:

    streamlit run demo_app/app.py
"""
import os
import sys
import time
import queue
import threading

# ── Import bootstrap ──────────────────────────────────────────────────────────
# `streamlit run demo_app/app.py` executes this file as a top-level script with no
# package context, so make the repo root importable and load submodules via the
# `demo_app` package (keeps the relative imports inside those modules working).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from demo_app import config, ytdl, graph_view, pipeline
from demo_app.trace import INGEST_STAGES, INGEST_STAGE_ORDER, ingest_stage_from_log

load_dotenv(os.path.join(_ROOT, ".env"), override=True)
pipeline.ensure_spawn()

SAMPLE_QUESTIONS = [
    "What happens to the brain when you stay up all night?",
    "How does sleep deprivation affect memory and emotions?",
    "Why do we feel sleepy — what is the role of adenosine?",
]

STAGE_ICON = {"pending": "⚪", "active": "🔄", "done": "✅"}
ROLE_STYLE = {
    "Generator": ("🧠", "#2b4c7e"),
    "Critique":  ("🔍", "#7e4a2b"),
    "Defender":  ("🛡️", "#2b7e4a"),
    "Judge":     ("⚖️", "#5a2b7e"),
    "Message":   ("💬", "#444"),
}


# ── Stepper helpers (ingest) ──────────────────────────────────────────────────
def _mark_stage(status: dict, key: str, done: bool = False):
    if key not in INGEST_STAGE_ORDER:
        return
    idx = INGEST_STAGE_ORDER.index(key)
    for i, k in enumerate(INGEST_STAGE_ORDER):
        if i < idx and status[k] != "done":
            status[k] = "done"
    if done:
        status[key] = "done"
    elif status[key] == "pending":
        status[key] = "active"


def _render_stepper(ph, status: dict, caption_saved: int = 0):
    lines = []
    for key, label in INGEST_STAGES:
        s = status[key]
        extra = ""
        if key == "caption_save" and s == "active" and caption_saved:
            extra = f" — {caption_saved} clip đã lưu"
        lines.append(f"{STAGE_ICON[s]} **{label}**{extra}")
    ph.markdown("\n\n".join(lines))


# ── Transcript parsing (query) ────────────────────────────────────────────────
def _parse_transcript(transcript):
    out = []
    for msg in transcript or []:
        c = (msg.get("content") or "").strip()
        if c.startswith("["):
            tag, _, rest = c[1:].partition("]")
            role = tag.split()[0] if tag else "Message"
            out.append((role, tag.strip(), rest.strip().lstrip(":").strip()))
        else:
            out.append(("Message", "", c))
    return out


def _render_card(role: str, header: str, body: str):
    icon, color = ROLE_STYLE.get(role, ROLE_STYLE["Message"])
    safe = (body or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(
        f"""<div style="border-left:4px solid {color};background:rgba(127,127,127,0.08);
        padding:8px 12px;margin:6px 0;border-radius:4px;">
        <b>{icon} {header or role}</b><br><span style="font-size:0.92em;">{safe}</span></div>""",
        unsafe_allow_html=True,
    )


# ── Live trace runner ─────────────────────────────────────────────────────────
def _stream_ingest(video_path: str, workdir: str):
    log_queue: queue.Queue = queue.Queue()
    holder: dict = {}
    t = threading.Thread(
        target=pipeline.ingest_with_trace,
        args=(video_path, workdir, log_queue, holder),
        daemon=True,
    )
    t.start()

    status = {k: "pending" for k, _ in INGEST_STAGES}
    status["download"] = "done"  # already downloaded before ingest
    stepper_ph = st.empty()
    log_ph = st.empty()
    logs, caption_saved = [], 0
    _render_stepper(stepper_ph, status)

    while True:
        try:
            kind, payload = log_queue.get(timeout=0.3)
        except queue.Empty:
            if not t.is_alive():
                break
            continue
        if kind == "done":
            break
        elif kind == "log":
            logs.append(payload)
            stg = ingest_stage_from_log(payload)
            if stg:
                _mark_stage(status, stg)
            log_ph.code("\n".join(logs[-25:]), language="log")
        elif kind == "stage":
            _mark_stage(status, payload["stage"], done=(payload["status"] == "done"))
        elif kind == "progress":
            caption_saved = payload.get("current", caption_saved)
            _mark_stage(status, "caption_save")
        elif kind == "error":
            logs.append(f"❌ {payload}")
            log_ph.code("\n".join(logs[-25:]), language="log")
        _render_stepper(stepper_ph, status, caption_saved)

    t.join(timeout=10)
    # Mark everything done if it finished cleanly.
    if "error" not in holder:
        for k in status:
            status[k] = "done"
    _render_stepper(stepper_ph, status, caption_saved)
    log_ph.code("\n".join(logs[-25:]) or "(no logs)", language="log")
    return holder


def _stream_query(workdir: str, question: str, max_rounds: int):
    log_queue: queue.Queue = queue.Queue()
    holder: dict = {}
    t = threading.Thread(
        target=pipeline.query_with_trace,
        args=(workdir, question, max_rounds, log_queue, holder),
        daemon=True,
    )
    t.start()

    log_ph = st.empty()
    logs = []
    t0 = time.time()
    while True:
        try:
            kind, payload = log_queue.get(timeout=0.3)
        except queue.Empty:
            if not t.is_alive():
                break
            log_ph.markdown(f"🔄 Đang chạy debate… ({time.time()-t0:.0f}s)")
            continue
        if kind == "done":
            break
        elif kind == "log":
            logs.append(payload)
            log_ph.code("\n".join(logs[-20:]), language="log")
        elif kind == "error":
            logs.append(f"❌ {payload}")
            log_ph.code("\n".join(logs[-20:]), language="log")
    t.join(timeout=10)
    log_ph.code("\n".join(logs[-20:]) or "(no logs)", language="log")
    return holder


# ── Tabs ──────────────────────────────────────────────────────────────────────
def _tab_ingest():
    st.subheader("① Ingest video + trace xử lý")
    st.caption("Tải một video YouTube ngắn rồi theo dõi từng bước pipeline xử lý video.")
    url = st.text_input("YouTube URL", value=config.DEMO_VIDEO_URL)

    col1, col2 = st.columns([1, 3])
    with col1:
        go = st.button("⬇️ Tải & Ingest", type="primary", use_container_width=True)

    if go:
        if not config.openai_key_present():
            st.error("Chưa có OPENAI_API_KEY. Dán key vào file `.env` ở thư mục gốc repo rồi chạy lại.")
            return
        # 1) Download
        with st.status("Đang tải video…", expanded=True) as s:
            try:
                info = ytdl.probe(url)
                st.write(f"🎬 **{info['title']}**  ·  {info['duration']//60}m{info['duration']%60:02d}s")
                video_path = ytdl.download_youtube(url, config.VIDEO_DIR, log=lambda m: st.write(m))
                s.update(label="Đã tải xong video.", state="complete")
            except Exception as e:  # noqa: BLE001
                s.update(label="Tải video thất bại.", state="error")
                st.error(str(e))
                return

        video_name = os.path.basename(video_path).split(".")[0]
        workdir = config.workdir_for(video_name)
        st.session_state["workdir"] = workdir
        st.session_state["video_name"] = video_name
        st.session_state["video_path"] = video_path

        # 2) Ingest with live stage trace
        st.markdown("#### Tiến trình ingest")
        holder = _stream_ingest(video_path, workdir)

        if "error" in holder:
            st.error(f"Ingest lỗi: {holder['error']}")
        else:
            st.session_state["ingested"] = True
            summ = holder.get("result", {})
            st.success("✅ Ingest hoàn tất!")
            c = st.columns(4)
            c[0].metric("Segments", summ.get("segments", "—"))
            c[1].metric("Chunks", summ.get("chunks", "—"))
            c[2].metric("Entities", summ.get("entities", "—"))
            c[3].metric("Edges", summ.get("edges", "—"))
            st.info("→ Chuyển sang tab **② Hỏi đáp** để đặt câu hỏi.")

    # Show existing state
    if st.session_state.get("ingested"):
        st.divider()
        st.caption(f"Đã ingest: `{st.session_state.get('video_name')}` → `{st.session_state.get('workdir')}`")


def _tab_query():
    st.subheader("② Hỏi đáp + trace agent (EBR-RAG)")
    workdir = st.session_state.get("workdir")
    if not workdir or not os.path.exists(config.graphml_path(workdir)):
        st.warning("Chưa có dữ liệu đã ingest. Hãy ingest video ở tab ① trước.")
        return

    question = st.text_area("Câu hỏi", value=SAMPLE_QUESTIONS[0], height=80)
    with st.expander("Câu hỏi gợi ý"):
        for q in SAMPLE_QUESTIONS:
            st.markdown(f"- {q}")
    max_rounds = st.slider("Số vòng debate (max_rounds)", 1, 5,
                           st.session_state.get("max_rounds", 2))
    run = st.button("🚀 Chạy EBR-RAG", type="primary")

    if run:
        if not config.openai_key_present():
            st.error("Chưa có OPENAI_API_KEY trong `.env`.")
            return
        if not question.strip():
            st.warning("Nhập câu hỏi trước đã.")
            return

        st.markdown("#### Trace trực tiếp")
        holder = _stream_query(workdir, question, max_rounds)
        if "error" in holder:
            st.error(f"Query lỗi: {holder['error']}")
            return
        result = holder.get("result")
        if not isinstance(result, dict):
            st.error("Pipeline không trả về kết quả chi tiết.")
            return

        st.session_state["last_result"] = result

        # Metadata
        st.markdown("#### Kết quả")
        m = st.columns(3)
        m[0].metric("🔄 Rounds", result.get("rounds_run", 0))
        m[1].metric("🔧 Tool calls", result.get("tool_calls_made", 0))
        m[2].metric("📊 Confidence", f"{result.get('confidence', 0.0):.0%}")

        st.markdown("##### 💬 Câu trả lời")
        st.markdown(result.get("answer", "") or "_(trống)_")
        if result.get("rationale"):
            with st.expander("💡 Rationale"):
                st.markdown(result["rationale"])

        # Agent timeline
        st.markdown("##### 🎭 Diễn tiến tranh luận (agent timeline)")
        for role, header, body in _parse_transcript(result.get("transcript", [])):
            _render_card(role, header, body)
        _render_card("Judge", "Judge — Phán quyết cuối", result.get("answer", ""))

        # Evidence
        ev = result.get("evidence", [])
        st.markdown(f"##### 📚 Bằng chứng ({len(ev)})")
        st.caption("Lưu ý: `citations` hiện luôn rỗng trong pipeline; dùng cột `validated` trên bằng chứng.")
        if ev:
            try:
                import pandas as pd
                df = pd.DataFrame(ev)
                cols = [c for c in ["id", "type", "score", "video_name", "time_range", "validated", "snippet"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
            except Exception:
                st.json(ev)


def _tab_graph():
    st.subheader("③ Đồ thị tri thức (tương tác)")
    workdir = st.session_state.get("workdir")
    if not workdir:
        st.warning("Chưa có dữ liệu đã ingest. Hãy ingest video ở tab ① trước.")
        return
    g = graph_view.load_graph(config.graphml_path(workdir))
    if g is None:
        st.warning("Không tìm thấy đồ thị (`graph_chunk_entity_relation.graphml`) hoặc đồ thị rỗng.")
        return

    stats = graph_view.graph_stats(g)
    c = st.columns(2)
    c[0].metric("Thực thể (nodes)", stats["num_nodes"])
    c[1].metric("Quan hệ (edges)", stats["num_edges"])
    st.markdown(graph_view.legend_html(stats), unsafe_allow_html=True)

    with st.expander("Bộ lọc hiển thị", expanded=True):
        types = graph_view.entity_types(g)
        selected = st.multiselect("Loại thực thể", options=types, default=types)
        cc = st.columns(3)
        min_degree = cc[0].slider("Bậc tối thiểu", 0, 10, 0)
        max_nodes = cc[1].slider("Số node tối đa", 50, 800, 300, step=50)
        physics = cc[2].checkbox("Physics (tự dàn layout)", value=True)

    html = graph_view.build_html(
        g, selected_types=set(selected), min_degree=min_degree,
        physics=physics, max_nodes=max_nodes,
    )
    components.html(html, height=680, scrolling=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái")
        st.write("🔑 OPENAI_API_KEY:", "✅" if config.openai_key_present() else "❌ (thêm vào `.env`)")
        st.markdown("**Checkpoints:**")
        for name, path in config.CHECKPOINTS.items():
            st.write(("✅ " if os.path.exists(path) else "❌ ") + name)
        st.divider()
        st.session_state["max_rounds"] = st.slider(
            "Số vòng debate mặc định", 1, 5, st.session_state.get("max_rounds", 2))
        st.caption("Chạy từ thư mục gốc repo:\n`streamlit run demo_app/app.py`")


def main():
    st.set_page_config(page_title="EBR-RAG Demo", page_icon="🎬", layout="wide")
    st.title("🎬 EBR-RAG — Interactive Demo")
    st.caption("Ingest video → trace xử lý → hỏi đáp với tranh luận đa tác nhân → đồ thị tri thức.")
    _sidebar()
    tab1, tab2, tab3 = st.tabs(["① Ingest", "② Hỏi đáp", "③ Đồ thị tri thức"])
    with tab1:
        _tab_ingest()
    with tab2:
        _tab_query()
    with tab3:
        _tab_graph()


if __name__ == "__main__":
    main()
