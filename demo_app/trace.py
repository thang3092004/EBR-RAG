"""
Live-trace plumbing shared by the ingest and query tabs.

Design notes (verified against the videorag source):
- Every module in ``videorag`` logs through ONE logger named ``"nano-graphrag"``
  (``from .._utils import logger``), EXCEPT the debate manager which uses
  ``logging.getLogger("videorag.debate.debate_manager")``.
- Neither logger has a level configured anywhere in the package, so they inherit
  the root default of WARNING. **All the INFO trace lines we care about are dropped
  unless we explicitly setLevel(INFO).** This is the #1 gotcha — see setup_trace_logging.
- Step 3 of ingest (caption + save clips) runs in ``multiprocessing.Process``
  children, so their tqdm/progress never reaches a parent log handler. We surface
  coarse progress for that stage with a disk poller (count saved .mp4 clips).

Queue event protocol — every item is a ``(kind, payload)`` tuple:
    ("log",      "formatted message")          # a raw log line
    ("stage",    {"stage": key, "status": "start"|"done", "detail": str})
    ("progress", {"stage": key, "current": int, "total": int|None})
    ("error",    "message")
    ("done",     None)                          # worker finished
"""
import os
import glob
import time
import queue
import logging
import threading
from contextlib import contextmanager

LOGGER_NAMES = ("nano-graphrag", "videorag.debate.debate_manager")

# Canonical ingest stages, in order, for the UI stepper.
INGEST_STAGES = [
    ("download", "Tải video"),
    ("split", "Cắt segment (30s)"),
    ("asr", "Nhận dạng giọng nói (Whisper)"),
    ("caption_save", "Tạo caption + lưu clip (MiniCPM-V)"),
    ("features", "Mã hoá đặc trưng hình ảnh (ImageBind)"),
    ("chunk", "Chunk văn bản"),
    ("entity", "Trích xuất đồ thị thực thể"),
]
INGEST_STAGE_ORDER = [k for k, _ in INGEST_STAGES]


class QueueLogHandler(logging.Handler):
    """Push every formatted log record into a queue for the UI to drain."""

    def __init__(self, log_queue: "queue.Queue"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put(("log", self.format(record)))
        except Exception:
            pass


def setup_trace_logging(log_queue: "queue.Queue", level: int = logging.INFO):
    """Attach a QueueLogHandler to both videorag loggers and force their level.

    Returns a ``teardown()`` callable that removes the handler and restores levels.
    """
    handler = QueueLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(message)s"))

    touched = []
    for name in LOGGER_NAMES:
        lg = logging.getLogger(name)
        prev_level = lg.level
        lg.setLevel(level)           # <-- the critical fix: WARNING -> INFO
        lg.addHandler(handler)
        touched.append((lg, prev_level))

    def teardown():
        for lg, prev_level in touched:
            try:
                lg.removeHandler(handler)
                lg.setLevel(prev_level)
            except Exception:
                pass

    return teardown


# ── Ingest stage detection from log markers ───────────────────────────────────

def ingest_stage_from_log(msg: str):
    """Map a raw log line to the ingest stage it implies (or None)."""
    if "video_segment_feature" in msg:
        return "features"
    if "[New Chunks] inserting" in msg or "Insert chunks for naive RAG" in msg:
        return "chunk"
    if "[Entity Extraction]" in msg:
        return "entity"
    if "Writing graph with" in msg:
        return "entity"   # graph write == entity stage essentially finished
    return None


# ── Ingest stage instrumentation ──────────────────────────────────────────────
# split_video / speech_to_text are imported into the videorag.videorag namespace
# and called by name inside insert_video(), so patching them there is enough. They
# run in the MAIN process, unlike the caption/save subprocesses.

@contextmanager
def instrument_ingest(log_queue: "queue.Queue"):
    """Temporarily wrap the main-process ingest step functions to emit stage events."""
    import videorag.videorag as vv

    originals = {}

    def wrap(name, stage_key, label):
        orig = getattr(vv, name)
        originals[name] = orig

        def wrapped(*args, **kwargs):
            log_queue.put(("stage", {"stage": stage_key, "status": "start", "detail": label}))
            try:
                result = orig(*args, **kwargs)
            finally:
                log_queue.put(("stage", {"stage": stage_key, "status": "done", "detail": label}))
            # After ASR finishes, the caption+save subprocess block begins.
            if stage_key == "asr":
                log_queue.put(("stage", {"stage": "caption_save", "status": "start",
                                         "detail": "Captioning + saving (subprocess)"}))
            return result

        setattr(vv, name, wrapped)

    wrap("split_video", "split", "Splitting video")
    wrap("speech_to_text", "asr", "Speech recognition")
    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(vv, name, orig)


def poll_segment_saving(log_queue, workdir, video_name, video_output_format, stop_event, total=None):
    """Background poller: count saved .mp4 clips so the caption/save subprocess
    (invisible to the log handler) shows live progress. Best-effort; never raises.
    """
    cache_dir = os.path.join(workdir, "_cache", video_name)
    last = -1
    while not stop_event.is_set():
        try:
            n = len(glob.glob(os.path.join(cache_dir, f"*.{video_output_format}")))
            if n != last:
                last = n
                log_queue.put(("progress", {"stage": "caption_save", "current": n, "total": total}))
        except Exception:
            pass
        time.sleep(1.0)
